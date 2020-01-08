#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import random
import tensorflow as tf
import time
import bisect
import lc0_az_policy_map
import proto.net_pb2 as pb

from net import Net

class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited, [-1, self.reshape_size, 1, 1]), 2, axis=1)
        return tf.nn.sigmoid(gammas) * x + betas


class ApplyPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs):
        h_conv_pol_flat = tf.reshape(inputs, [-1, 80*8*8])
        return tf.matmul(h_conv_pol_flat, tf.cast(self.fc1, h_conv_pol_flat.dtype))

class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model']['filters']
        self.RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks']
        self.SE_ratio = self.cfg['model']['se_ratio']
        self.policy_channels = self.cfg['model'].get('policy_channels', 32)
        precision = self.cfg['training'].get('precision', 'single')
        loss_scale = self.cfg['training'].get('loss_scale', 128)

        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        policy_head = self.cfg['model'].get('policy', 'convolution')
        value_head  = self.cfg['model'].get('value', 'wdl')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None

        if policy_head == "classical":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CONVOLUTION
        else:
            raise ValueError(
                "Unknown policy head format: {}".format(policy_head))

        if value_head == "classical":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_CLASSICAL
            self.wdl = False
        elif value_head == "wdl":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_WDL
            self.wdl = True
        else:
            raise ValueError(
                "Unknown value head format: {}".format(value_head))

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']], True)
        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    def init_net_v2(self):
        input_var = tf.keras.Input(shape=(112, 8*8))
        x_planes = tf.keras.layers.Reshape([112, 8, 8])(input_var)
        self.model = tf.keras.Model(inputs=input_var, outputs=self.construct_net_v2(x_planes))

    def infer(self, input_batch):
        y,z = self.model(input_batch)
        y = tf.cast(y, tf.float32)
        z = tf.cast(z, tf.float32)
        return y, tf.nn.softmax(z)

    def replace_weights_v2(self, new_weights_orig):
        new_weights = [w for w in new_weights_orig]
        # self.model.weights ordering doesn't match up nicely, so first shuffle the new weights to match up.
        # input order is (for convolutional policy):
        # policy conv
        # policy bn * 4
        # policy raw conv and bias
        # value conv
        # value bn * 4
        # value dense with bias
        # value dense with bias
        #
        # output order is (for convolutional policy):
        # value conv
        # policy conv
        # value bn * 4
        # policy bn * 4
        # policy raw conv and bias
        # value dense with bias
        # value dense with bias
        new_weights[-5] = new_weights_orig[-10]
        new_weights[-6] = new_weights_orig[-11]
        new_weights[-7] = new_weights_orig[-12]
        new_weights[-8] = new_weights_orig[-13]
        new_weights[-9] = new_weights_orig[-14]
        new_weights[-10] = new_weights_orig[-15]
        new_weights[-11] = new_weights_orig[-5]
        new_weights[-12] = new_weights_orig[-6]
        new_weights[-13] = new_weights_orig[-7]
        new_weights[-14] = new_weights_orig[-8]
        new_weights[-15] = new_weights_orig[-16]
        new_weights[-16] = new_weights_orig[-9]

        all_evals = []
        offset = 0
        last_was_gamma = False
        for e, weights in enumerate(self.model.weights):
            source_idx = e+offset
            if weights.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if e == 0:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weights[source_idx])):
                        if (i % (num_inputs*9))//9 == rule50_input:
                            new_weights[source_idx][i] = new_weights[source_idx][i]*99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[source_idx], shape=shape)
                weights.assign(
                    tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[source_idx], shape=shape)
                weights.assign(
                    tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Can't populate renorm weights, but the current new_weight will need using elsewhere.
                if 'renorm' in weights.name:
                    offset-=1
                    continue
                # betas without gamms need to skip the gamma in the input.
                if 'beta:' in weights.name and not last_was_gamma:
                    source_idx+=1
                    offset+=1
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[source_idx], shape=weights.shape)
                if 'stddev:' in weights.name:
                    weights.assign(tf.math.sqrt(new_weight + 1e-5))
                else:
                    weights.assign(new_weight)
                # need to use the variance to also populate the stddev for renorm, so adjust offset.
                if 'variance:' in weights.name and self.renorm_enabled:
                    offset-=1
            last_was_gamma = 'gamma:' in weights.name

    def batch_norm_v2(self, input, scale=False):
        return tf.keras.layers.BatchNormalization(
            epsilon=1e-5, axis=1, fused=True, center=True,
            scale=scale)(input)

    def squeeze_excitation_v2(self, inputs, channels):
        assert channels % self.SE_ratio == 0

        pooled = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(inputs)
        squeezed = tf.keras.layers.Activation('relu')(tf.keras.layers.Dense(channels // self.SE_ratio, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg)(pooled))
        excited = tf.keras.layers.Dense(2 * channels, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg)(squeezed)
        return ApplySqueezeExcitation()([inputs, excited])

    def conv_block_v2(self, inputs, filter_size, output_channels, bn_scale=False):
        conv = tf.keras.layers.Conv2D(output_channels, filter_size, use_bias=False, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, data_format='channels_first')(inputs)
        return tf.keras.layers.Activation('relu')(self.batch_norm_v2(conv, scale=bn_scale))

    def residual_block_v2(self, inputs, channels):
        conv1 = tf.keras.layers.Conv2D(channels, 3, use_bias=False, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, data_format='channels_first')(inputs)
        out1 = tf.keras.layers.Activation('relu')(self.batch_norm_v2(conv1, scale=False))
        conv2 = tf.keras.layers.Conv2D(channels, 3, use_bias=False, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, data_format='channels_first')(out1)
        out2 = self.squeeze_excitation_v2(self.batch_norm_v2(conv2, scale=True), channels)
        return tf.keras.layers.Activation('relu')(tf.keras.layers.add([inputs, out2]))

    def construct_net_v2(self, inputs):
        flow = self.conv_block_v2(inputs, filter_size=3, output_channels=self.RESIDUAL_FILTERS, bn_scale=True)
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block_v2(flow, self.RESIDUAL_FILTERS)
        # Policy head
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_CONVOLUTION:
            conv_pol = self.conv_block_v2(flow, filter_size=3, output_channels=self.RESIDUAL_FILTERS)
            conv_pol2 = tf.keras.layers.Conv2D(80, 3, use_bias=True, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg, data_format='channels_first')(conv_pol)
            h_fc1 = ApplyPolicyMap()(conv_pol2)
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_CLASSICAL:
            conv_pol = self.conv_block_v2(flow, filter_size=1, output_channels=self.policy_channels)
            h_conv_pol_flat = tf.keras.layers.Flatten()(conv_pol)
            h_fc1 = tf.keras.layers.Dense(1858, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg)(h_conv_pol_flat)
        else:
            raise ValueError(
                "Unknown policy head type {}".format(self.POLICY_HEAD))

        # Value head
        conv_val = self.conv_block_v2(flow, filter_size=1, output_channels=32)
        h_conv_val_flat = tf.keras.layers.Flatten()(conv_val)
        h_fc2 = tf.keras.layers.Dense(128, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, activation='relu')(h_conv_val_flat)
        if self.wdl:
            h_fc3 = tf.keras.layers.Dense(3, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg)(h_fc2)
        else:
            h_fc3 = tf.keras.layers.Dense(1, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, activation='tanh')(h_fc2)
        return h_fc1, h_fc3

