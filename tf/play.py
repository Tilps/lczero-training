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
import glob
import gzip
import random
import chess
import math
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from tfprocess import TFProcess
from policy_index import policy_index
from timeit import default_timer as timer


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    tfprocess = TFProcess(cfg)

    tfprocess.init_for_play()

    tfprocess.restore_v2(True)

    input_data = np.zeros((1,112,8,8), dtype=float)
    # pre heat the net by running an inference with empty input.
    @tf.function
    def first(input_data):
        return tfprocess.model.first(input_data, training=False)
    @tf.function
    def recurse(input_data):
        return tfprocess.model.recursive(input_data, training=False)
    hidden_state1 = first(input_data)
    recurse(hidden_state1)
    tfprocess.model.policy(hidden_state1, training=False)
    tfprocess.model.value(hidden_state1, training=False)

    board = chess.Board()


    while True:
        instruction = input()
        if instruction == 'uci':
            print('id name The blob')
            print('id author The blob Authors')
            print('uciok')
        elif instruction.startswith('position '):
            pos_start = timer()
            # update board and calculate input
            board.reset()
            input_data = np.zeros((1,112,8,8), dtype=float)
            parts = instruction.split()
            started = False
            flip = len(parts) > 2 and len(parts) % 2 == 0
            for i in range(len(parts)):
                if started:
                    hist_depth = len(parts) - i
                    if hist_depth <= 7:
                        for j in range(6):
                            for sq in board.pieces(j+1, not flip):
                                row = sq // 8
                                if flip:
                                    row = 7 - row
                                input_data[0, hist_depth*13+j, row, sq % 8] = 1.0
                            for sq in board.pieces(j+1, flip):
                                row = sq // 8
                                if flip:
                                    row = 7 - row
                                input_data[0, hist_depth*13+j+6, row, sq % 8] = 1.0
                        if board.is_repetition(2):
                            input_data[0, hist_depth*13+12,:,:] = 1.0
                    board.push_uci(parts[i])
                if parts[i] == 'moves':
                    started = True
            for j in range(6):
                for sq in board.pieces(j+1, not flip):
                    row = sq // 8
                    if flip:
                        row = 7 - row
                    input_data[0, j, row, sq % 8] = 1.0
                for sq in board.pieces(j+1, flip):
                    row = sq // 8
                    if flip:
                        row = 7 - row
                    input_data[0, j+6, row, sq % 8] = 1.0
            if board.is_repetition(2):
                input_data[0, 12,:,:] = 1.0
            
            # chess 960 castling, but without the chess960 support...
            if board.has_queenside_castling_rights(True):
                row = 0
                if flip:
                    row = 7 - row
                input_data[0, 104, row, 0] = 1.0
            if board.has_queenside_castling_rights(False):
                row = 7
                if flip:
                    row = 7 - row
                input_data[0, 104, row, 0] = 1.0
            if board.has_kingside_castling_rights(True):
                row = 0
                if flip:
                    row = 7 - row
                input_data[0, 105, row, 7] = 1.0
            if board.has_kingside_castling_rights(False):
                row = 7
                if flip:
                    row = 7 - row
                input_data[0, 105, row, 7] = 1.0
            if board.has_legal_en_passant():
                sq = board.ep_square
                input_data[0, 108, 7, sq % 8] = 1.0
            input_data[0, 109, :, :] = board.halfmove_clock / 100.0
            if flip:
                input_data[0, 110, :, :] = 1.0
            input_data[0, 111, :, :] = 1.0
            # TODO: Correct history_to_keep for castling rights change, and passing of en-passant rights.
            history_to_keep = board.halfmove_clock
            if history_to_keep < 7:
                for i in range(103, history_to_keep*13+12, -1):
                    input_data[0, i, :, :] = 0.0
                
            # TODO: finish implementing transform correctly.
            transform = 0
            if not board.has_castling_rights(True) and not board.has_castling_rights(False):
                king_sq = board.pieces(chess.KING, not flip).pop()
                if flip:
                    king_sq = king_sq + 8 * (7 - 2*(king_sq // 8))

                if king_sq % 8 < 4:
                    transform |= 1
                    king_sq = king_sq + (7 - 2*(king_sq % 8))
                if len(board.pieces(chess.PAWN, not flip).union(board.pieces(chess.PAWN, flip))) == 0:
                    if king_sq // 8 >= 4:
                        transform |= 2
                        king_sq = king_sq + 8 * (7 - 2*(king_sq // 8))
                    if king_sq // 8 > 7 - king_sq % 8:
                        transform |= 4
                    # elif king_sq // 8 == 7 - king_sq % 8: the hard logic goes here...
            if transform != 0:
                if (transform & 1) != 0:
                    np.flip(input_data, 3)
                if (transform & 2) != 0:
                    np.flip(input_data, 2)
                if (transform & 4) != 0:
                    np.transpose(input_data, (0, 1, 3, 2))
                    np.flip(input_data, 2)
                    np.flip(input_data, 3)
            pos_end = timer()
            #print('timed {}'.format(pos_end-pos_start))
            


        elif instruction.startswith('go '):
            go_start = timer()
            # Do evil things that are not uci compliant... This loop should be on a different thread so it can be interrupted by stop.
            hidden_state1 = first(input_data)
            go_mid = timer()
            #print('timed {}'.format(go_mid-go_start))
            for i in range(cmd.unroll):
                hidden_state1 = recurse(hidden_state1)
            policy = tfprocess.model.policy(hidden_state1, training=False).numpy()
            bestmove = '0000'
            bestpolicy = None
            def mirrorMaybe(move, mirror):
                if mirror:
                    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
                else:
                    return move
            legal_moves = set([mirrorMaybe(x, flip).uci() for x in board.legal_moves])
            # iterate over 1858 options, check if they are in legal move set, and determine which has maximal network output value.
            for i in range(1858):
                if policy_index[i] in legal_moves:
                    p = policy[0,i]
                    print('info string policy {}, {}'.format(mirrorMaybe(chess.Move.from_uci(policy_index[i]), flip).uci(), p))
                    if bestpolicy is None or p > bestpolicy:
                        bestmove = policy_index[i]
                        bestpolicy = p
            bestmove = mirrorMaybe(chess.Move.from_uci(bestmove), flip).uci()
            value = tf.nn.softmax(tfprocess.model.value(hidden_state1, training=False)).numpy()
            go_end = timer()
            q = value[0,0] - value[0,2]
            cp = int(90 * math.tan(1.5637541897 * q))
            print('info depth 1 seldepth 1 time {} nodes {} score cp {} nps {} pv {} '.format(int((go_end-go_start)*1000), cmd.unroll + 1, cp, int((cmd.unroll + 1)/(go_end-go_start)), bestmove))
            print('bestmove {}'.format(bestmove))
        elif instruction == 'quit':
            return
        elif instruction == 'isready':
            print('readyok')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg',
                           type=argparse.FileType('r'),
                           help='yaml configuration with training parameters')
    argparser.add_argument('--unroll',
                           type=int,
                           help='Override time management with forced unroll.')

    main(argparser.parse_args())
