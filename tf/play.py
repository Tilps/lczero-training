#!/usr/bin/env python3

import argparse
import os
import yaml
import sys
import glob
import gzip
import random
import chess
import chess.pgn
import math
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import logging
from tfprocess import TFProcess
from timeit import default_timer as timer

def calculate_input(instruction, board):
    print('received:', instruction)
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

    # This section commented out until there is smarter logic about when to enable or disable this information provided to the net.    
##    # chess 960 castling, but without the chess960 support...
##    if board.has_queenside_castling_rights(True):
##        row = 0
##        if flip:
##            row = 7 - row
##        input_data[0, 12, row, 0] = 1.0
##    if board.has_queenside_castling_rights(False):
##        row = 7
##        if flip:
##            row = 7 - row
##        input_data[0, 12, row, 0] = 1.0
##    if board.has_kingside_castling_rights(True):
##        row = 0
##        if flip:
##            row = 7 - row
##        input_data[0, 13, row, 7] = 1.0
##    if board.has_kingside_castling_rights(False):
##        row = 7
##        if flip:
##            row = 7 - row
##        input_data[0, 13, row, 7] = 1.0
##    if board.has_legal_en_passant():
##        sq = board.ep_square
##        input_data[0, 16, 7, sq % 8] = 1.0
##    input_data[0, 17, :, :] = board.halfmove_clock / 100.0
##    # TODO: Allow to depopultae some of the inputs.
##    for i in range(18, 24):
##        input_data[0, i, :, :] = 1.0
    if flip:
        input_data[0, 14, :, :] = 1.0

    return input_data, flip


def reverseTransformSq(sq, flip):
    if flip:
        sq = sq + 8 * (7 - 2*(sq // 8))
    return sq


def reverseTransformDir(x_delta, y_delta, flip):
    if flip:
        y_delta = -y_delta
    return x_delta, y_delta


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

    input_data, flip = calculate_input('position startpos', board)

    outputs = first(input_data)

    reco_moves = []

    while True:
        instruction = input()
        if instruction == 'uci':
            print('id name Reverser')
            print('id author Reverser Authors')
            print('uciok')
        elif instruction.startswith('position '):
            reco_moves = []
            pos_start = timer()
            input_data, flip = calculate_input(instruction, board)
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
                # Skip squares that currently have a piece, they are all illegal. (TODO: this can be false with 960 castling, if king stays still)
                pos_sq = reverseTransformSq(i % 64, flip)
                if board.piece_at(pos_sq) is not None:
                    continue
                val = policy[0,i]
                if not max_set or val > max_value:
                    max_set = True
                    max_value = val
                    max_idx = i
            for i in range(64*128):
                # Skip squares that currently have a piece, they are all illegal. (TODO: this can be false with 960 castling, if king stays still)
                pos_sq = reverseTransformSq(i % 64, flip)
                if board.piece_at(pos_sq) is not None:
                    continue
                val = policy[0,i]
                if val > max_value - 3.:
                    print('Policy value:',val,'index:',i)
                    print('Move type:', i//64, 'Starting Square:', i % 64)
                    row = pos_sq // 8
                    col = pos_sq % 8
                    print('Square:',"abcdefgh"[col]+"12345678"[row])
            print()
            print('Max Policy value:',max_value,'index:',max_idx)
            print('Move type:', max_idx//64, 'Starting Square:', max_idx % 64)
            from_sq = reverseTransformSq(max_idx % 64, flip)
            row = from_sq // 8
            col = from_sq % 8
            print('Square:',"abcdefgh"[col]+"12345678"[row])
            move_type = max_idx // 64
            if move_type < 48:
                direction = move_type // 6
                cap_type = move_type % 6
                x_delta = 0
                y_delta = 0
                if direction == 0:
                    x_delta, y_delta = 2, 1
                elif direction == 1:
                    x_delta, y_delta = 1, 2
                elif direction == 2:
                    x_delta, y_delta = -2, 1
                elif direction == 3:
                    x_delta, y_delta = -1, 2
                elif direction == 4:
                    x_delta, y_delta = 2, -1
                elif direction == 5:
                    x_delta, y_delta = 1, -2
                elif direction == 6:
                    x_delta, y_delta = -2, -1
                elif direction == 7:
                    x_delta, y_delta = -1, -2
                x_delta, y_delta = reverseTransformDir(x_delta, y_delta, flip)
                to_row = row - y_delta
                to_col = col - x_delta
                print('KnightMove','To:',"abcdefgh"[to_col]+"12345678"[to_row],'Captured:',cap_type)
                sq = to_row*8+to_col
                piece_moved = board.remove_piece_at(sq)
                if piece_moved is None:
                    print('Illegal KnightMove')
                board.set_piece_at(from_sq, piece_moved)
                if cap_type == 1:
                    board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 2:
                    board.set_piece_at(sq, chess.Piece(chess.ROOK, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 3:
                    board.set_piece_at(sq, chess.Piece(chess.BISHOP, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 4:
                    board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 5:
                    board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK if flip else chess.WHITE))
                board.turn = chess.WHITE if flip else chess.BLACK
                input_data, flip = calculate_input('position fen '+board.fen(), board)
                reco_moves.append("abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row])
            elif move_type < 96:
                direction = (move_type-48) // 6
                cap_type = (move_type-48) % 6
                x_delta = 0
                y_delta = 0
                if direction == 0:
                    x_delta, y_delta = 1, 0
                elif direction == 1:
                    x_delta, y_delta = 1, 1
                elif direction == 2:
                    x_delta, y_delta = 0, 1
                elif direction == 3:
                    x_delta, y_delta = -1, 1
                elif direction == 4:
                    x_delta, y_delta = -1, 0
                elif direction == 5:
                    x_delta, y_delta = -1, -1
                elif direction == 6:
                    x_delta, y_delta = 0, -1
                elif direction == 7:
                    x_delta, y_delta = 1, -1
                x_delta, y_delta = reverseTransformDir(x_delta, y_delta, flip)
                to_row = row - y_delta
                to_col = col - x_delta
                while True:
                    if to_row < 0 or to_row > 7 or to_col < 0 or to_col > 7:
                        print('Illegal slide')
                        break
                    sq = to_row*8+to_col
                    if board.piece_at(sq) is not None:
                        break
                    to_row = to_row - y_delta
                    to_col = to_col - x_delta
                sq = to_row*8+to_col
                print('SlideMove','To:',"abcdefgh"[to_col]+"12345678"[to_row],'Captured:',cap_type)
                piece_moved = board.remove_piece_at(sq)
                if piece_moved is None:
                    print('Illegal slide')
                board.set_piece_at(from_sq, piece_moved)
                if cap_type == 1:
                    board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 2:
                    board.set_piece_at(sq, chess.Piece(chess.ROOK, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 3:
                    board.set_piece_at(sq, chess.Piece(chess.BISHOP, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 4:
                    board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 5:
                    board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK if flip else chess.WHITE))
                board.turn = chess.WHITE if flip else chess.BLACK
                input_data, flip = calculate_input('position fen '+board.fen(), board)
                reco_moves.append("abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row])
            elif move_type < 98:
                x_delta = -1
                y_delta = 1
                if move_type == 97:
                    x_delta = 1
                x_delta, y_delta = reverseTransformDir(x_delta, y_delta, flip)
                to_row = row - y_delta
                to_col = col - x_delta
                sq = to_row*8+to_col
                print('Enpassant', 'To:',"abcdefgh"[to_col]+"12345678"[to_row])
                piece_moved = board.remove_piece_at(sq)
                if piece_moved is None:
                    print('Illegal enpassant')
                board.set_piece_at(from_sq, piece_moved)
                board.set_piece_at(row*8+to_col, chess.Piece(chess.PAWN, chess.BLACK if flip else chess.WHITE))
                board.turn = chess.WHITE if flip else chess.BLACK
                input_data, flip = calculate_input('position fen '+board.fen(), board)
                reco_moves.append("abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row])
            elif move_type < 106:
                # TODO: support 960
                y_delta = 0
                if move_type == 98:
                    x_delta = -1
                elif move_type == 105:
                    x_delta = 1
                else:
                    print('Unsupported castling')
                x_delta, y_delta = reverseTransformDir(x_delta, y_delta, flip)
                if x_delta > 0:
                    print('Castling, O-O')
                    piece_moved = board.remove_piece_at(row*8+6)
                    if piece_moved is None:
                        print('Illegal castling')
                    board.set_piece_at(from_sq, piece_moved)
                    piece_moved = board.remove_piece_at(row*8+5)
                    if piece_moved is None:
                        print('Illegal castling')
                    board.set_piece_at(row*8+7, piece_moved)
                else:
                    print('Castling, O-O-O')
                    piece_moved = board.remove_piece_at(row*8+2)
                    if piece_moved is None:
                        print('Illegal castling')
                    board.set_piece_at(from_sq, piece_moved)
                    piece_moved = board.remove_piece_at(row*8+3)
                    if piece_moved is None:
                        print('Illegal castling')
                    board.set_piece_at(row*8, piece_moved)
                board.turn = chess.WHITE if flip else chess.BLACK
                input_data, flip = calculate_input('position fen '+board.fen(), board)
                reco_moves.append("abcdefgh"[col]+"12345678"[row]+"abcdefgh"[7 if x_delta > 0 else 0]+"12345678"[row])
            elif move_type < 117:
                y_delta = 1
                if move_type == 106:
                    x_delta = 0
                elif move_type < 112:
                    x_delta = -1
                    cap_type = move_type-107
                else:
                    x_delta = 1
                    cap_type = move_type-112
                x_delta, y_delta = reverseTransformDir(x_delta, y_delta, flip)
                to_row = row - y_delta
                to_col = col - x_delta
                sq = to_row*8+to_col
                print('Promotion','To:',"abcdefgh"[to_col]+"12345678"[to_row],'Captured:',cap_type)
                piece_moved = board.remove_piece_at(sq)
                if piece_moved is None:
                    print('Illegal promotion')
                board.set_piece_at(from_sq, chess.Piece(chess.PAWN, chess.WHITE if flip else chess.BLACK))
                if cap_type == 1:
                    board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 2:
                    board.set_piece_at(sq, chess.Piece(chess.ROOK, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 3:
                    board.set_piece_at(sq, chess.Piece(chess.BISHOP, chess.BLACK if flip else chess.WHITE))
                elif cap_type == 4:
                    board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.BLACK if flip else chess.WHITE))
                board.turn = chess.WHITE if flip else chess.BLACK
                input_data, flip = calculate_input('position fen '+board.fen(), board)
                reco_moves.append("abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row]+piece_moved.symbol().lower())
            print('Moves from start:',moves[0,0])
            print('Rule 50 est:',r50_est[0,0])
            bestmove = '0000'
            print('Recomoves:',list(reversed(reco_moves)))
            if board.board_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR":
                board.reset()
                for uci_move in reversed(reco_moves):
                    board.push_uci(uci_move)
                print(chess.pgn.Game.from_board(board))
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

    main(argparser.parse_args())
