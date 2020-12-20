#!/usr/bin/env python3

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
import logging
from tfprocess import TFProcess
from timeit import default_timer as timer

def calculate_input(instruction, board):
    # update board and calculate input
    board.reset()
    input_data = np.zeros((1,24,8,8), dtype=np.float32)
    parts = instruction.split()
    started = False
    if len(parts) > 2 and parts[1] == "fen":
        last_idx = len(parts)
        if "moves" in parts:
            last_idx = parts.index("moves")
        # fen is from parts[2] to parts[last_idx-1] inclusive.
        board.set_fen(' '.join(parts[2:last_idx]))

    def castling_rights(board):
        result = 0
        if board.has_queenside_castling_rights(True):
            result |= 1
        if board.has_queenside_castling_rights(False):
            result |= 2
        if board.has_kingside_castling_rights(True):
            result |= 4
        if board.has_kingside_castling_rights(False):
            result |= 8
        return result
    
    for i in range(len(parts)):
        if started:
            board.push_uci(parts[i])
        if parts[i] == 'moves':
            started = True
    flip = board.turn == chess.BLACK
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
    
    # chess 960 castling, but without the chess960 support...
    if board.has_queenside_castling_rights(True):
        row = 0
        if flip:
            row = 7 - row
        input_data[0, 12, row, 0] = 1.0
    if board.has_queenside_castling_rights(False):
        row = 7
        if flip:
            row = 7 - row
        input_data[0, 12, row, 0] = 1.0
    if board.has_kingside_castling_rights(True):
        row = 0
        if flip:
            row = 7 - row
        input_data[0, 13, row, 7] = 1.0
    if board.has_kingside_castling_rights(False):
        row = 7
        if flip:
            row = 7 - row
        input_data[0, 13, row, 7] = 1.0
    if board.has_legal_en_passant():
        sq = board.ep_square
        input_data[0, 16, 7, sq % 8] = 1.0
    input_data[0, 17, :, :] = board.halfmove_clock / 100.0
    # TODO: Allow to depopultae some of the inputs.
    for i in range(18, 24):
        input_data[0, i, :, :] = 1.0
        
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
            elif king_sq // 8 == 7 - king_sq % 8:
                def choose_transform(bitboard, transform, flip):
                    if flip:
                        bitboard = chess.flip_vertical(bitboard)
                    if (transform & 1) != 0:
                        bitboard = chess.flip_horizontal(bitboard)
                    if (transform & 2) != 0:
                        bitboard = chess.flip_vertical(bitboard)
                    alternative = chess.flip_anti_diagonal(bitboard)
                    if alternative < bitboard:
                        return 1
                    if alternative > bitboard:
                        return -1
                    return 0
                def should_transform_ad(board, transform, flip):
                    allbits = int(board.pieces(chess.PAWN, not flip).union(board.pieces(chess.PAWN, flip)).union(board.pieces(chess.KNIGHT, not flip)).union(board.pieces(chess.KNIGHT, flip)).union(board.pieces(chess.BISHOP, not flip)).union(board.pieces(chess.BISHOP, flip)).union(board.pieces(chess.ROOK, not flip)).union(board.pieces(chess.ROOK, flip)).union(board.pieces(chess.QUEEN, not flip)).union(board.pieces(chess.QUEEN, flip)).union(board.pieces(chess.KING, not flip)).union(board.pieces(chess.KING, flip)))
                    outcome = choose_transform(allbits, transform, flip)
                    if outcome == 1:
                        return True
                    if outcome == -1:
                        return False
                    stmbits = int(board.pieces(chess.PAWN, not flip).union(board.pieces(chess.KNIGHT, not flip)).union(board.pieces(chess.BISHOP, not flip)).union(board.pieces(chess.ROOK, not flip)).union(board.pieces(chess.QUEEN, not flip)).union(board.pieces(chess.KING, not flip)))
                    outcome = choose_transform(stmbits, transform, flip)
                    if outcome == 1:
                        return True
                    if outcome == -1:
                        return False
                    kingbits = int(board.pieces(chess.KING, not flip).union(board.pieces(chess.KING, flip)))
                    outcome = choose_transform(kingbits, transform, flip)
                    if outcome == 1:
                        return True
                    if outcome == -1:
                        return False
                    queenbits = int(board.pieces(chess.QUEEN, not flip).union(board.pieces(chess.QUEEN, flip)))
                    outcome = choose_transform(queenbits, transform, flip)
                    if outcome == 1:
                        return True
                    if outcome == -1:
                        return False
                    rookbits = int(board.pieces(chess.ROOK, not flip).union(board.pieces(chess.ROOK, flip)))
                    outcome = choose_transform(rookbits, transform, flip)
                    if outcome == 1:
                        return True
                    if outcome == -1:
                        return False
                    knightbits = int(board.pieces(chess.KNIGHT, not flip).union(board.pieces(chess.KNIGHT, flip)))
                    outcome = choose_transform(knightbits, transform, flip)
                    if outcome == 1:
                        return True
                    if outcome == -1:
                        return False
                    bishopbits = int(board.pieces(chess.BISHOP, not flip).union(board.pieces(chess.BISHOP, flip)))
                    outcome = choose_transform(bishopbits, transform, flip)
                    if outcome == 1:
                        return True
                    if outcome == -1:
                        return False

                    return False
                if should_transform_ad(board, transform, flip):
                    transform |= 4
                
    if transform != 0:
        if (transform & 1) != 0:
            np.flip(input_data, 3)
        if (transform & 2) != 0:
            np.flip(input_data, 2)
        if (transform & 4) != 0:
            np.transpose(input_data, (0, 1, 3, 2))
            np.flip(input_data, 2)
            np.flip(input_data, 3)
    return input_data, flip, transform


def main(cmd):
    tf.get_logger().setLevel(logging.ERROR)
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    tfprocess = TFProcess(cfg)

    tfprocess.init_for_play()

    tfprocess.restore_v2()
    for (swa, w) in zip(tfprocess.swa_weights, tfprocess.model.weights):
        w.assign(swa.read_value())

    board = chess.Board()

    # pre heat the net by running an inference with startpos input.
    @tf.function
    def first(input_data):
        return tfprocess.model(input_data, training=False)

    input_data, flip, transform = calculate_input('position startpos', board)

    outputs = first(input_data)

    while True:
        instruction = input()
        if instruction == 'uci':
            print('id name Reverser')
            print('id author Reverser Authors')
            print('uciok')
        elif instruction.startswith('position '):
            pos_start = timer()
            input_data, flip,transform = calculate_input(instruction, board)
            pos_end = timer()
            #print('timed {}'.format(pos_end-pos_start))

        elif instruction.startswith('go '):
            go_start = timer()
            # Do evil things that are not uci compliant... This loop should be on a different thread so it can be interrupted by stop, if this was actually uci :P
            policy, moves, r50_est = first(input_data)
            max_value = 0
            max_set = False
            max_idx = -1
            for i in range(64*128):
                # Skip squares that currently have a piece, they are all illegal. (TODO: confirm this is true even with 960 castling)
                # TODO: apply 'transform' in reverse to get board location.
                pos_sq = i % 64
                if flip:
                    pos_sq = pos_sq + 8 * (7 - 2*(pos_sq // 8))
                if board.piece_at(pos_sq) is not None:
                    continue
                val = policy[0,i]
                if not max_set or val > max_value:
                    max_set = True
                    max_value = val
                    max_idx = i
            print('Policy value:',max_value,'index:',max_idx)
            print('Move type:', max_idx//64, 'Starting Square:', max_idx % 64)
            sq = max_idx % 64
            row = sq // 8
            col = sq % 8
            if flip:
                row = 7 - row
            print('Square:',"abcdefgh"[col]+"12345678"[row])
            print('Moves from start:',moves[0,0])
            print('Rule 50 est:',r50_est[0,0])
            bestmove = '0000'
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
