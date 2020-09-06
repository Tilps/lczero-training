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
from functools import reduce
import operator


class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
                                            [-1, self.reshape_size, 1, 1]),
                                 2,
                                 axis=1)
        return tf.nn.sigmoid(gammas) * x + betas


class SqueezeExcitationLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, channels, name, **kwargs):
        super(SqueezeExcitationLayer, self).__init__(name=name, **kwargs)
        assert channels % config_source.SE_ratio == 0
        self.pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_first')
        self.dense1 = tf.keras.layers.Dense(
            channels // config_source.SE_ratio,
            kernel_initializer='glorot_normal',
            kernel_regularizer=config_source.l2reg,
            name=name + '/se/dense1')
        self.activation = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(2 * channels,
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=config_source.l2reg,
                                        name=name + '/se/dense2')
        self.applier = ApplySqueezeExcitation()

    def call(self, inputs):
        pooled = self.pool(inputs)
        squeezed = self.activation(self.dense1(pooled))
        excited = self.dense2(squeezed)
        return self.applier([inputs, excited])


class ConvBlockLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, filter_size, output_channels, name, bn_scale=False, **kwargs):
        super(ConvBlockLayer, self).__init__(name=name, **kwargs)
        self.conv = tf.keras.layers.Conv2D(output_channels,
                                      filter_size,
                                      use_bias=False,
                                      padding='same',
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=config_source.l2reg,
                                      data_format='channels_first',
                                      name=name + '/conv2d')
        self.batch_norm = config_source.make_batch_norm(name=name + '/bn', scale=bn_scale)
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, inputs, training=None):
        conved = self.conv(inputs)
        bned = self.batch_norm(conved, training=training)
        return self.activation(bned)

class ResidualBlockLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, channels, name, bn_scale=False, **kwargs):
        super(ResidualBlockLayer, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(channels,
                                      3,
                                      use_bias=False,
                                      padding='same',
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=config_source.l2reg,
                                      data_format='channels_first',
                                      name=name + '/1/conv2d')
        self.batch_norm1 = config_source.make_batch_norm(name=name + '/1/bn', scale=False)
        self.activation = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(channels,
                                      3,
                                      use_bias=False,
                                      padding='same',
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=config_source.l2reg,
                                      data_format='channels_first',
                                      name=name + '/2/conv2d')
        self.batch_norm2 = config_source.make_batch_norm(name=name + '/2/bn', scale=True)
        self.SE = SqueezeExcitationLayer(config_source, channels, name=name+'/se')

    def call(self, inputs, training=None):
        conved1 = self.conv1(inputs)
        bned1 = self.batch_norm1(conved1, training=training)
        activated1 = self.activation(bned1)
        conved2 = self.conv2(activated1)
        bned2 = self.batch_norm2(conved2, training=training)
        seed = self.SE(bned2)
        return self.activation(tf.keras.layers.add([inputs, seed]))


class InitialStackLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, name, **kwargs):
        super(InitialStackLayer, self).__init__(name=name, **kwargs)
        self.conv = ConvBlockLayer(config_source, 3, config_source.RESIDUAL_FILTERS, name=name+'/input')
        self.body = []
        for i in range(config_source.INITIAL_RESIDUAL_BLOCKS):
            self.body.append(ResidualBlockLayer(config_source, config_source.RESIDUAL_FILTERS, name=name+'/residual_{}'.format(i+1)))

    def call(self, inputs, training=None):
        flow = self.conv(inputs, training=training)
        for residual in self.body:
            flow = residual(flow, training=training)
        return flow


class MainStackLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, name, **kwargs):
        super(MainStackLayer, self).__init__(name=name, **kwargs)
        self.body = []
        for i in range(config_source.RESIDUAL_BLOCKS):
            self.body.append(ResidualBlockLayer(config_source, config_source.RESIDUAL_FILTERS, name=name+'/residual_{}'.format(i+1)))

    def call(self, inputs, training=None):
        flow = inputs
        for residual in self.body:
            flow = residual(flow, training=training)
        return flow


class ApplyPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs):
        h_conv_pol_flat = tf.reshape(inputs, [-1, 80 * 8 * 8])
        return tf.matmul(h_conv_pol_flat,
                         tf.cast(self.fc1, h_conv_pol_flat.dtype))


class PolicyHeadLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, name, **kwargs):
        super(PolicyHeadLayer, self).__init__(name=name, **kwargs)
        self.conv1 = ConvBlockLayer(config_source, 3, config_source.RESIDUAL_FILTERS, name=name+'/policy1')
        self.conv2 = tf.keras.layers.Conv2D(
                80,
                3,
                use_bias=True,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=config_source.l2reg,
                bias_regularizer=config_source.l2reg,
                data_format='channels_first',
                name=name+'/policy')
        self.applier = ApplyPolicyMap()

    def call(self, inputs, training=None):
        conved1 = self.conv1(inputs, training=training)
        conved2 = self.conv2(conved1)
        return self.applier(conved2)

class ValueHeadLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, name, **kwargs):
        super(ValueHeadLayer, self).__init__(name=name, **kwargs)
        self.conv = ConvBlockLayer(config_source, 1, 32, name=name+'/value')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128,
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=config_source.l2reg,
                                      activation='relu',
                                      name=name+'/value/dense1')
        self.dense2 = tf.keras.layers.Dense(3,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=config_source.l2reg,
                                          bias_regularizer=config_source.l2reg,
                                          name=name+'/value/dense2')

    def call(self, inputs, training=None):
        conved = self.conv(inputs, training=training)
        flattened = self.flatten(conved)
        densed1 = self.dense1(flattened)
        return self.dense2(densed1)

class MovesLeftHeadLayer(tf.keras.layers.Layer):
    def __init__(self, config_source, name, **kwargs):
        super(MovesLeftHeadLayer, self).__init__(name=name, **kwargs)
        self.conv = ConvBlockLayer(config_source, 1, 8, name=name+'/moves_left')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128,
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=config_source.l2reg,
                                      activation='relu',
                                      name=name+'/moves_left/dense1')
        self.dense2 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=config_source.l2reg,
                                          activation='relu',
                                          name=name+'/moves_left/dense2')

    def call(self, inputs, training=None):
        conved = self.conv(inputs, training=training)
        flattened = self.flatten(conved)
        densed1 = self.dense1(flattened)
        return self.dense2(densed1)


def partial_stop(flow, allow):
    return allow * flow + tf.stop_gradient((1.0 - allow) * flow)


class RecursiveStackModel(tf.keras.Model):
    def __init__(self, config_source, name, **kwargs):
        super(RecursiveStackModel, self).__init__(name=name, **kwargs)
        self.reshape = tf.keras.layers.Reshape([112, 8, 8])
        self.first = InitialStackLayer(config_source, name=name+'/initial')
        self.recursive = MainStackLayer(config_source, name=name+'/main')
        self.policy = PolicyHeadLayer(config_source, name=name+'/p')
        self.value = ValueHeadLayer(config_source, name=name+'/v')
        self.movesleft = MovesLeftHeadLayer(config_source, name=name+'/ml')
        self.unroll = 10 

    def call(self, inputs, training=None):
        reshaped = self.reshape(inputs)
        firsted = self.first(reshaped, training=training)
        results = [(self.policy(firsted, training=training), self.value(firsted, training=training), self.movesleft(firsted, training=training))]
        flow = firsted
        for i in range(self.unroll):
            flow = self.recursive(partial_stop(flow, 0.5), training=training)
            results.append((self.policy(flow, training=training), self.value(flow, training=training), self.movesleft(flow, training=training)))
        return results


class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = os.path.join(self.cfg['training']['path'],
                                     self.cfg['name'])

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model']['filters']
        self.INITIAL_RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks'] ##TODO: give it its own setting.
        self.RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks']
        self.SE_ratio = self.cfg['model']['se_ratio']
        precision = self.cfg['training'].get('precision', 'single')
        loss_scale = self.cfg['training'].get('loss_scale', 128)
        self.virtual_batch_size = self.cfg['model'].get(
            'virtual_batch_size', None)

        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else loss_scale

        input_mode = self.cfg['model'].get('input_type', 'classic')

        self.INPUT_MODE = None

        if input_mode == "classic":
            self.INPUT_MODE = 1
        elif input_mode == "frc_castling":
            self.INPUT_MODE = 2
        elif input_mode == "canonical":
            self.INPUT_MODE = 3
        elif input_mode == "canonical_100":
            self.INPUT_MODE = 4
        elif input_mode == "canonical_armageddon":
            self.INPUT_MODE = 132
        else:
            raise ValueError(
                "Unknown input mode format: {}".format(input_mode))

        self.swa_enabled = self.cfg['training'].get('swa', False)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg['training'].get('swa_max_n', 0)

        self.renorm_enabled = self.cfg['training'].get('renorm', False)
        self.renorm_max_r = self.cfg['training'].get('renorm_max_r', 1)
        self.renorm_max_d = self.cfg['training'].get('renorm_max_d', 0)
        self.renorm_momentum = self.cfg['training'].get(
            'renorm_momentum', 0.99)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                   'GPU')
        tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']], True)
        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False,
                                       dtype=tf.int64)

    def init_v2(self, train_dataset, test_dataset, validation_dataset=None):
        self.train_dataset = train_dataset
        self.train_iter = iter(train_dataset)
        self.test_dataset = test_dataset
        self.test_iter = iter(test_dataset)
        self.validation_dataset = validation_dataset
        self.init_net_v2()

    def init_for_play(self):
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        self.model = RecursiveStackModel(self, name='model')
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        #self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.root_dir,
            max_to_keep=50,
            keep_checkpoint_every_n_hours=24,
            checkpoint_name=self.cfg['name'])

    def init_net_v2(self):
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        self.model = RecursiveStackModel(self, name='model')

        # swa_count initialized reguardless to make checkpoint code simpler.
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
        self.swa_weights = None
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_weights = [
                tf.Variable(w, trainable=False) for w in self.model.weights
            ]

        self.active_lr = 0.01
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=lambda: self.active_lr, momentum=0.9, nesterov=True)
        self.orig_optimizer = self.optimizer
        if self.loss_scale != 1:
            self.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self.optimizer, self.loss_scale)

        def correct_policy(target, output):
            output = tf.cast(output, tf.float32)
            # Calculate loss on policy head
            if self.cfg['training'].get('mask_legal_moves'):
                # extract mask for legal moves from target policy
                move_is_legal = tf.greater_equal(target, 0)
                # replace logits of illegal moves with large negative value (so that it doesn't affect policy of legal moves) without gradient
                illegal_filler = tf.zeros_like(output) - 1.0e10
                output = tf.where(move_is_legal, output, illegal_filler)
            # y_ still has -1 on illegal moves, flush them to 0
            target = tf.nn.relu(target)
            return target, output

        def policy_loss(target, output):
            target, output = correct_policy(target, output)
            policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_loss_fn = policy_loss

        def policy_accuracy(target, output):
            target, output = correct_policy(target, output)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.policy_accuracy_fn = policy_accuracy

        self.policy_accuracy_fn = policy_accuracy

        def moves_left_mean_error_fn(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(tf.abs(target - output))

        self.moves_left_mean_error = moves_left_mean_error_fn

        def policy_entropy(target, output):
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            return tf.math.negative(
                tf.reduce_mean(
                    tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                                  axis=1)))

        self.policy_entropy_fn = policy_entropy

        def policy_uniform_loss(target, output):
            uniform = tf.where(tf.greater_equal(target, 0),
                               tf.ones_like(target), tf.zeros_like(target))
            balanced_uniform = uniform / tf.reduce_sum(
                uniform, axis=1, keepdims=True)
            target, output = correct_policy(target, output)
            policy_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(balanced_uniform),
                                                        logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_uniform_loss_fn = policy_uniform_loss

        q_ratio = self.cfg['training'].get('q_ratio', 1)
        assert 0 <= q_ratio <= 1

        # Linear conversion to scalar to compute MSE with, for comparison to old values
        wdl = tf.expand_dims(tf.constant([1.0, 0.0, -1.0]), 1)

        self.qMix = lambda z, q: q * q_ratio + z * (1 - q_ratio)

        def value_loss(target, output):
            output = tf.cast(output, tf.float32)
            value_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            return tf.reduce_mean(input_tensor=value_cross_entropy)

        self.value_loss_fn = value_loss

        def mse_loss(target, output):
            output = tf.cast(output, tf.float32)
            scalar_z_conv = tf.matmul(tf.nn.softmax(output), wdl)
            scalar_target = tf.matmul(target, wdl)
            return tf.reduce_mean(input_tensor=tf.math.squared_difference(
                scalar_target, scalar_z_conv))

        self.mse_loss_fn = mse_loss

        def moves_left_loss(target, output):
            # Scale the loss to similar range as other losses.
            scale = 20.0
            target = target / scale
            output = tf.cast(output, tf.float32) / scale
            huber = tf.keras.losses.Huber(10.0 / scale)
            return tf.reduce_mean(huber(target, output))

        self.moves_left_loss_fn = moves_left_loss

        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']

        moves_loss_w = self.cfg['training']['moves_left_loss_weight']

        def _lossMix(policy, value, moves_left):
            return pol_loss_w * policy + val_loss_w * value + moves_loss_w * moves_left

        self.lossMix = _lossMix

        def accuracy(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.accuracy_fn = accuracy

        self.avg_policy_loss = []
        self.avg_value_loss = []
        self.avg_moves_left_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None
        self.last_steps = None
        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
        self.lr = self.cfg['training']['lr_values'][0]
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-test".format(self.cfg['name'])))
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-train".format(self.cfg['name'])))
        if vars(self).get('validation_dataset', None) is not None:
            self.validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-validation".format(self.cfg['name'])))
        if self.swa_enabled:
            self.swa_writer = tf.summary.create_file_writer(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-swa-test".format(self.cfg['name'])))
            self.swa_validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-swa-validation".format(self.cfg['name'])))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer,
                                              model=self.model,
                                              global_step=self.global_step,
                                              swa_count=self.swa_count)
        self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.root_dir,
            max_to_keep=50,
            keep_checkpoint_every_n_hours=24,
            checkpoint_name=self.cfg['name'])


    def restore_v2(self, partial=False):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            if partial:
                self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
            else:
                self.checkpoint.restore(self.manager.latest_checkpoint)

    def process_loop_v2(self, batch_size, test_batches, batch_splits=1):
        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = self.global_step.read_value()
        total_steps = self.cfg['training']['total_steps']
        for _ in range(steps % total_steps, total_steps):
            self.process_v2(batch_size,
                            test_batches,
                            batch_splits=batch_splits)

    @tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    @tf.function()
    def process_inner_loop(self, x, y, z, q, m):
        policy_losses = []
        value_losses = []
        mse_losses = []
        moves_left_losses = []
        with tf.GradientTape() as tape:
            outputs = self.model(x, training=True)
            reg_term = sum(self.model.losses)
            total_loss = reg_term
            for policy, value, moves_left in outputs:
                policy_loss = self.policy_loss_fn(y, policy)
                value_loss = self.value_loss_fn(self.qMix(z, q), value)
                moves_left_loss = self.moves_left_loss_fn(m, moves_left)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                moves_left_losses.append(moves_left_loss)

                total_loss += self.lossMix(policy_loss, value_loss,
                                      moves_left_loss)
            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)
        for policy, value, moves_left in outputs:
            mse_losses.append(self.mse_loss_fn(self.qMix(z, q), value))
        return policy_losses, value_losses, mse_losses, moves_left_losses, reg_term, tape.gradient(
            total_loss, self.model.trainable_weights)

    def process_v2(self, batch_size, test_batches, batch_splits=1):
        if not self.time_start:
            self.time_start = time.time()

        # Get the initial steps value before we do a training step.
        steps = self.global_step.read_value()
        if not self.last_steps:
            self.last_steps = steps

        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2

        # Run test before first step to see delta since end of last run.
        if steps % self.cfg['training']['total_steps'] == 0:
            # Steps is given as one higher than current in order to avoid it
            # being equal to the value the end of a run is stored against.
            #self.calculate_test_summaries_v2(test_batches, steps + 1)
            #if self.swa_enabled:
            #    self.calculate_swa_summaries_v2(test_batches, steps + 1)
            nothing = 0

        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_size * self.cfg[
                'training'].get('num_batch_splits', 1)
            raise ValueError(
                'batch_size must be a multiple of {}'.format(required_factor))

        # Determine learning rate
        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = steps % self.cfg['training']['total_steps']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1,
                                        tf.float32) / self.warmup_steps

        # need to add 1 to steps because steps will be incremented after gradient update
        if (steps +
                1) % self.cfg['training']['train_avg_report_steps'] == 0 or (
                    steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.read_weights()

        # Run training for this batch
        grads = None
        for _ in range(batch_splits):
            x, y, z, q, m = next(self.train_iter)
            policy_losses, value_losses, mse_losses, moves_left_losses, reg_term, new_grads = self.process_inner_loop(
                x, y, z, q, m)
            if not grads:
                grads = new_grads
            else:
                grads = [tf.math.add(a, b) for (a, b) in zip(grads, new_grads)]
            # Keep running averages
            self.avg_policy_loss.append(policy_losses)
            self.avg_value_loss.append(value_losses)
            self.avg_moves_left_loss.append(moves_left_losses)
            self.avg_mse_loss.append(mse_losses)
            self.avg_reg_term.append(reg_term)
        # Gradients of batch splits are summed, not averaged like usual, so need to scale lr accordingly to correct for this.
        self.active_lr = self.lr / batch_splits
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['training'].get('max_grad_norm',
                                                 10000.0) * batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))

        # Update steps.
        self.global_step.assign_add(1)
        steps = self.global_step.read_value()

        if steps % self.cfg['training'][
                'train_avg_report_steps'] == 0 or steps % self.cfg['training'][
                    'total_steps'] == 0:
            pol_loss_w = self.cfg['training']['policy_loss_weight']
            val_loss_w = self.cfg['training']['value_loss_weight']
            moves_loss_w = self.cfg['training']['moves_left_loss_weight']
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (tf.cast(steps_elapsed, tf.float32) /
                                      elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [[0]], axis=0)
            avg_moves_left_loss = np.mean(self.avg_moves_left_loss or [[0]], axis=0)
            avg_value_loss = np.mean(self.avg_value_loss or [[0]], axis=0)
            avg_mse_loss = np.mean(self.avg_mse_loss or [[0]], axis=0)
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            for i in range(len(avg_policy_loss)):
                print(
                    "step {} depth: {}, lr={:g} policy={:g} value={:g} mse={:g} moves={:g} reg={:g} total={:g} ({:g} pos/s)"
                    .format(
                        steps, i, self.lr, avg_policy_loss[i], avg_value_loss[i],
                        avg_mse_loss[i], avg_moves_left_loss[i], avg_reg_term,
                        pol_loss_w * avg_policy_loss[i] +
                        val_loss_w * avg_value_loss[i] + avg_reg_term +
                        moves_loss_w * avg_moves_left_loss[i], speed))

            after_weights = self.read_weights()
            with self.train_writer.as_default():
                for i in range(len(avg_policy_loss)):
                    tf.summary.scalar("Policy Loss {}".format(i), avg_policy_loss[i], step=steps)
                    tf.summary.scalar("Value Loss {}".format(i), avg_value_loss[i], step=steps)
                    tf.summary.scalar("Moves Left Loss {}".format(i),
                                      avg_moves_left_loss[i],
                                      step=steps)
                    tf.summary.scalar("MSE Loss {}".format(i), avg_mse_loss[i], step=steps)
                tf.summary.scalar("Reg term", avg_reg_term, step=steps)
                tf.summary.scalar("LR", self.lr, step=steps)
                tf.summary.scalar("Gradient norm",
                                  grad_norm / batch_splits,
                                  step=steps)
                self.compute_update_ratio_v2(before_weights, after_weights,
                                             steps)
            self.train_writer.flush()
            self.time_start = time_end
            self.last_steps = steps
            self.avg_policy_loss = []
            self.avg_moves_left_loss = []
            self.avg_value_loss = []
            self.avg_mse_loss = []
            self.avg_reg_term = []

        if self.swa_enabled and steps % self.cfg['training']['swa_steps'] == 0:
            self.update_swa_v2()

        # Calculate test values every 'test_steps', but also ensure there is
        # one at the final step so the delta to the first step can be calculted.
        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg[
                'training']['total_steps'] == 0:
            self.calculate_test_summaries_v2(test_batches, steps)
            if self.swa_enabled:
                self.calculate_swa_summaries_v2(test_batches, steps)

        #if self.validation_dataset is not None and (
        #        steps % self.cfg['training']['validation_steps'] == 0
        #        or steps % self.cfg['training']['total_steps'] == 0):
        #    if self.swa_enabled:
        #        self.calculate_swa_validations_v2(steps)
        #    else:
        #        self.calculate_test_validations_v2(steps)

        # Save session and weights at end, and also optionally every 'checkpoint_steps'.
        if steps % self.cfg['training']['total_steps'] == 0 or (
                'checkpoint_steps' in self.cfg['training']
                and steps % self.cfg['training']['checkpoint_steps'] == 0):
            evaled_steps = steps.numpy()
            self.manager.save(checkpoint_number=evaled_steps)
            print("Model saved in file: {}".format(
                self.manager.latest_checkpoint))

    def calculate_swa_summaries_v2(self, test_batches, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print('swa', end=' ')
        self.calculate_test_summaries_v2(test_batches, steps)
        self.test_writer = true_test_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y, z, q, m):
        policy_losses = []
        policy_accuracies = []
        policy_entropies = []
        policy_uls = []
        value_losses = []
        mse_losses = []
        value_accuracies = []
        moves_left_losses = []
        moves_left_mean_errors = []
        outputs = self.model(x, training=False)
        for policy, value, moves_left in outputs:
            policy_loss = self.policy_loss_fn(y, policy)
            policy_accuracy = self.policy_accuracy_fn(y, policy)
            policy_entropy = self.policy_entropy_fn(y, policy)
            policy_ul = self.policy_uniform_loss_fn(y, policy)
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = self.accuracy_fn(self.qMix(z, q), value)
            moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            moves_left_mean_error = self.moves_left_mean_error(m, moves_left)
            policy_losses.append(policy_loss)
            policy_accuracies.append(policy_accuracy)
            policy_entropies.append(policy_entropy)
            policy_uls.append(policy_ul)
            value_losses.append(value_loss)
            mse_losses.append(mse_loss)
            value_accuracies.append(value_accuracy)
            moves_left_losses.append(moves_left_loss)
            moves_left_mean_errors.append(moves_left_mean_error)

        return policy_losses, value_losses, moves_left_losses, mse_losses, policy_accuracies, value_accuracies, moves_left_mean_errors, policy_entropies, policy_uls

    def calculate_test_summaries_v2(self, test_batches, steps):
        sum_policy_accuracy = []
        sum_value_accuracy = []
        sum_moves_left = []
        sum_moves_left_mean_error = []
        sum_mse = []
        sum_policy = []
        sum_value = []
        sum_policy_entropy = []
        sum_policy_ul = []
        for _ in range(0, test_batches):
            x, y, z, q, m = next(self.test_iter)
            policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul = self.calculate_test_summaries_inner_loop(
                x, y, z, q, m)
            sum_policy_accuracy.append(policy_accuracy)
            sum_policy_entropy.append(policy_entropy)
            sum_policy_ul.append(policy_ul)
            sum_mse.append(mse_loss)
            sum_policy.append(policy_loss)
            sum_value_accuracy.append(value_accuracy)
            sum_value.append(value_loss)
            sum_moves_left.append(moves_left_loss)
            sum_moves_left_mean_error.append(moves_left_mean_error)
        sum_policy_accuracy = np.mean(sum_policy_accuracy, axis=0)
        sum_value_accuracy = np.mean(sum_value_accuracy, axis=0)
        sum_moves_left = np.mean(sum_moves_left, axis=0)
        sum_moves_left_mean_error = np.mean(sum_moves_left_mean_error, axis=0)
        sum_mse = np.mean(sum_mse, axis=0)
        sum_policy = np.mean(sum_policy, axis=0)
        sum_value = np.mean(sum_value, axis=0)
        sum_policy_entropy = np.mean(sum_policy_entropy, axis=0)
        sum_policy_ul = np.mean(sum_policy_ul, axis=0)

        with self.test_writer.as_default():
            for i in range(len(sum_policy)):
                tf.summary.scalar("Policy Loss {}".format(i), sum_policy[i], step=steps)
                tf.summary.scalar("Value Loss {}".format(i), sum_value[i], step=steps)
                tf.summary.scalar("MSE Loss {}".format(i), sum_mse[i], step=steps)
                tf.summary.scalar("Policy Accuracy {}".format(i),
                                  sum_policy_accuracy[i],
                                  step=steps)
                tf.summary.scalar("Policy Entropy {}".format(i), sum_policy_entropy[i], step=steps)
                tf.summary.scalar("Policy UL {}".format(i), sum_policy_ul[i], step=steps)
                tf.summary.scalar("Value Accuracy {}".format(i),
                                  sum_value_accuracy[i],
                                  step=steps)
                tf.summary.scalar("Moves Left Loss {}".format(i),
                                  sum_moves_left[i],
                                  step=steps)
                tf.summary.scalar("Moves Left Mean Error {}".format(i),
                                  sum_moves_left_mean_error[i],
                                  step=steps)
            for w in self.model.weights:
                tf.summary.histogram(w.name, w, step=steps)
        self.test_writer.flush()

        for i in range(len(sum_policy)):
            print("step {} depth {}, policy={:g} value={:g} policy accuracy={:g}% value accuracy={:g}% mse={:g} policy entropy={:g} policy ul={:g} moves={:g} moves mean={:g}".\
                format(steps, i, sum_policy[i], sum_value[i], sum_policy_accuracy[i], sum_value_accuracy[i], sum_mse[i], sum_policy_entropy[i], sum_policy_ul[i], sum_moves_left[i], sum_moves_left_mean_error[i]))

    def calculate_swa_validations_v2(self, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_validation_writer, self.validation_writer = self.validation_writer, self.swa_validation_writer
        print('swa', end=' ')
        self.calculate_test_validations_v2(steps)
        self.validation_writer = true_validation_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def calculate_test_validations_v2(self, steps):
        sum_policy_accuracy = 0
        sum_value_accuracy = 0
        sum_moves_left = 0
        sum_moves_left_mean_error = 0
        sum_mse = 0
        sum_policy = 0
        sum_value = 0
        sum_policy_entropy = 0
        sum_policy_ul = 0
        counter = 0
        for (x, y, z, q, m) in self.validation_dataset:
            policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul = self.calculate_test_summaries_inner_loop(
                x, y, z, q, m)
            sum_policy_accuracy += policy_accuracy
            sum_policy_entropy += policy_entropy
            sum_policy_ul += policy_ul
            sum_mse += mse_loss
            sum_policy += policy_loss
            if self.moves_left:
                sum_moves_left += moves_left_loss
                sum_moves_left_mean_error += moves_left_mean_error
            counter += 1
            if self.wdl:
                sum_value_accuracy += value_accuracy
                sum_value += value_loss
        sum_policy_accuracy /= counter
        sum_policy_accuracy *= 100
        sum_policy /= counter
        sum_policy_entropy /= counter
        sum_policy_ul /= counter
        sum_value /= counter
        if self.wdl:
            sum_value_accuracy /= counter
            sum_value_accuracy *= 100
        if self.moves_left:
            sum_moves_left /= counter
            sum_moves_left_mean_error /= counter
        # Additionally rescale to [0, 1] so divide by 4
        sum_mse /= (4.0 * counter)
        with self.validation_writer.as_default():
            tf.summary.scalar("Policy Loss", sum_policy, step=steps)
            tf.summary.scalar("Value Loss", sum_value, step=steps)
            tf.summary.scalar("MSE Loss", sum_mse, step=steps)
            tf.summary.scalar("Policy Accuracy",
                              sum_policy_accuracy,
                              step=steps)
            tf.summary.scalar("Policy Entropy", sum_policy_entropy, step=steps)
            tf.summary.scalar("Policy UL", sum_policy_ul, step=steps)
            if self.wdl:
                tf.summary.scalar("Value Accuracy",
                                  sum_value_accuracy,
                                  step=steps)
            if self.moves_left:
                tf.summary.scalar("Moves Left Loss",
                                  sum_moves_left,
                                  step=steps)
                tf.summary.scalar("Moves Left Mean Error",
                                  sum_moves_left_mean_error,
                                  step=steps)
        self.validation_writer.flush()

        print("step {}, validation: policy={:g} value={:g} policy accuracy={:g}% value accuracy={:g}% mse={:g} policy entropy={:g} policy ul={:g}".\
            format(steps, sum_policy, sum_value, sum_policy_accuracy, sum_value_accuracy, sum_mse, sum_policy_entropy, sum_policy_ul), end='')

        if self.moves_left:
            print(" moves={:g} moves mean={:g}".format(
                sum_moves_left, sum_moves_left_mean_error))
        else:
            print()

    @tf.function()
    def compute_update_ratio_v2(self, before_weights, after_weights, steps):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [
            after - before
            for after, before in zip(after_weights, before_weights)
        ]
        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [
            tf.math.reduce_euclidean_norm(w) for w in before_weights
        ]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.))
                  for d, w, tensor in zip(delta_norms, weight_norms,
                                          self.model.weights)
                  if not 'moving' in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar('update_ratios/' + name, ratio, step=steps)
        # Filtering is hard, so just push infinities/NaNs to an unreasonably large value.
        ratios = [
            tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299,
                    lambda: 200.) for (_, r) in ratios
        ]
        tf.summary.histogram('update_ratios_log10',
                             tf.stack(ratios),
                             buckets=1000,
                             step=steps)

    def update_swa_v2(self):
        num = self.swa_count.read_value()
        for (w, swa) in zip(self.model.weights, self.swa_weights):
            swa.assign(swa.read_value() * (num / (num + 1.)) + w.read_value() *
                       (1. / (num + 1.)))
        self.swa_count.assign(min(num + 1., self.swa_max_n))

    def make_batch_norm(self, name, scale=False):
        if self.renorm_enabled:
            clipping = {
                "rmin": 1.0 / self.renorm_max_r,
                "rmax": self.renorm_max_r,
                "dmax": self.renorm_max_d
            }
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                fused=False,
                center=True,
                scale=scale,
                renorm=True,
                renorm_clipping=clipping,
                renorm_momentum=self.renorm_momentum,
                name=name)
        else:
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                center=True,
                scale=scale,
                virtual_batch_size=self.virtual_batch_size,
                name=name)
