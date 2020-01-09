#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
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

import argparse
import os
import yaml
import sys
import mmap
import time
import tensorflow as tf
import struct
from tfprocess import TFProcess
from net import Net

def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    file_to_mmap = cfg['infer']['mmap_file']
    print('Waiting for mmap', end='')
    while not os.path.exists(file_to_mmap):
        print('.', end='', flush=True)
        time.sleep(1)
    print('')
    with open(file_to_mmap, "r+b") as f:
        data_view = mmap.mmap(f.fileno(), 0)
        print('Waiting for network', end='')
        while (struct.unpack_from("Q", data_view, 0)[0] != 1):
            print('.', struct.unpack_from("Q", data_view, 0), end='', flush=True)
            time.sleep(1)
        print('')
        struct.pack_into("Q", data_view, 0, 0)
        net_length = struct.unpack_from("Q", data_view, 16)[0]
        net = Net()
        net.parse_proto_from_data(data_view[32:32+net_length])
        cfg['model']['filters'] = net.filters()
        cfg['model']['residual_blocks'] = net.blocks()
        cfg['model']['se_ratio'] = net.se_ratio()

        tfprocess = TFProcess(cfg)
        tfprocess.init_net_v2()
        tfprocess.replace_weights_v2(net.get_weights())

        def infer(input_batch):
            return tfprocess.infer(input_batch)
        fast_infer = tf.function(infer, input_signature=[tf.TensorSpec(shape=(None, 112, 64), dtype=tf.float32)])
        @tf.function()
        def infer_from_view(data_view, data_len):
            ib = tf.io.decode_raw(data_view[32:32+112*64*4*data_len], tf.float32)
            ib = tf.reshape(ib, [-1, 112, 64])
            p,wdl = fast_infer(ib)
            return p, wdl

        struct.pack_into("Q", data_view, 8, 1)
        while True:
            enter = time.clock()
            while (struct.unpack_from("Q", data_view, 0)[0] != 2):
                continue
            waited = time.clock()
            struct.pack_into("Q", data_view, 0, 0)
            data_len = struct.unpack_from("Q", data_view, 16)[0]
            infer_start = time.clock()
            p,wdl = infer_from_view(data_view, data_len)
            infer_done = time.clock()
            memoryview(data_view)[32+112*64*4*1024:32+112*64*4*1024+1858*4*data_len] = memoryview(p).cast('B')
            memoryview(data_view)[32+112*64*4*1024+1858*4*1024:32+112*64*4*1024+1858*4*1024+3*4*data_len] = memoryview(wdl).cast('B')
            struct.pack_into("Q", data_view, 8, 2)
            sent_data = time.clock()
            print(data_len)
            print(waited-enter, infer_start-waited, infer_done-infer_start, sent_data-infer_done)

        data_view.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for running inference on demand via mmap file.')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
        help='yaml configuration with inference parameters')

    main(argparser.parse_args())
