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

STRICT_RULE_50 = False


def utility_calc(moves, target_moves):
    return -tf.nn.relu(moves-target_moves)


def check_extra_valid(board):
    # If 16 white pieces, there can be no more un-captures by black pawns, so they are stuck in the files they are in. Therefore they must not be doubled.
    if chess.popcount(board.occupied_co[chess.WHITE]) == 16:
        pawn_mask = board.pawns & board.occupied_co[chess.BLACK]
        for bb_file in chess.BB_FILES:
            if chess.popcount(pawn_mask & bb_file) > 1:
                return False
    # And vice versa.
    if chess.popcount(board.occupied_co[chess.BLACK]) == 16:
        pawn_mask = board.pawns & board.occupied_co[chess.WHITE]
        for bb_file in chess.BB_FILES:
            if chess.popcount(pawn_mask & bb_file) > 1:
                return False
    # TODO: if full pawn wall in place, only knights can be outside.
    return True


class SearchNode:
    def __init__(self):
        self.input_data = None
        self.flip = None
        self.board = None
        self.state = None
        self.visits = 0
        self.total_move_est = 0
        self.policy = None
        self.policy_idx = None
        self.children = None
        self.bad = False
        self.good = False

    def visit(self, eval_func, target_moves):
        if self.bad:
            print("Bad node unavoidable")
            return -500
        if self.good:
            return utility_calc(0, target_moves)
        if self.input_data is None:
            self.input_data, self.flip = calculate_input_from_board(self.board, self.state)
        if self.visits == 0:
            self.visits = 1
            policy, moves, r50_est, sorted_high_policy, sorted_indicies = eval_func(self.input_data)
            self.policy = sorted_high_policy
            self.policy_index = sorted_indicies
            value = utility_calc(moves[0,0], target_moves)
            self.total_move_est = value
            self.children = []
            return value
        best_child = -1
        best_child_score = -1000
        U_mult = 5 * math.sqrt(self.visits)
        for i in range(len(self.policy)):
            Q = -1
            U = self.policy[i] * U_mult
            if i < len(self.children):
                child = self.children[i]
                Q = child.total_move_est / child.visits
                U = U / (1 + child.visits)
            elif i > len(self.children):
                break
            score = Q+U
            if score > best_child_score:
                best_child_score = score
                best_child = i
        if best_child == len(self.children):
            new_child = SearchNode()
            new_child.board = self.board.copy()
            new_child.state = self.state.copy()
            try:
                updateBoardForIndex(new_child.board, new_child.state, self.policy_index[best_child], self.flip)
            except ValueError as err:
                print(err)
                new_child.bad = True
                new_child.total_move_est = -500
                new_child.visits = 1
                self.children.append(new_child)
                # Try again, to emulate prefiltering illegal moves.
                return self.visit(eval_func, target_moves)
            else:
                if new_child.board.board_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" and new_child.board.turn == chess.WHITE:
                    new_child.good = True
                    new_child.visits = 1
                    new_child.total_move_est = utility_calc(0, target_moves - 1)
            self.children.append(new_child)
        res = self.children[best_child].visit(eval_func, target_moves - 1)
        self.visits = self.visits + 1
        self.total_move_est = self.total_move_est + res
        return res


class OptionalState:
    def __init__(self):
        self.ks_castle_white = None
        self.ks_castle_black = None
        self.qs_castle_white = None
        self.qs_castle_black = None
        self.enpassant = None
        self.rule50 = None
        self.first_time = True

    def update_for_rule50_reset(self):
        if self.rule50 is not None:
            if self.rule50 > 0:
                if STRICT_RULE_50:
                    raise ValueError("Resetting rule 50 too early.")
                print("Resetting rule 50 early!!")
            self.rule50 = None

    def update_for_new_board(self, board):
        if self.rule50 is not None:
            self.rule50 = self.rule50 - 1
            if self.rule50 < 0:
                if STRICT_RULE_50:
                    raise ValueError("Failed to reset rule 50 in time.")
                print("Failed to reset rule 50 in time!")
                self.rule50 = None
        # If there are no candidate enpassant pawns we could set self.enpassant to -1.
        # However only setting it to -1 in that case provides no value, since -1 is only useful if there are enpassant candidate pawns.
        self.enpassant = None
        # TODO: technically False to None transition for castling should only happen on rook or king move affecting the specific castling option.
        # Additionally forcing False for misplaced king or rook is useless as it only provides additional meaning when the king and rook are correct.
        if self.qs_castle_white is not None and not self.qs_castle_white:
            self.qs_castle_white = None
        if self.qs_castle_black is not None and not self.qs_castle_black:
            self.qs_castle_black = None
        if self.ks_castle_white is not None and not self.ks_castle_white:
            self.ks_castle_white = None
        if self.ks_castle_black is not None and not self.ks_castle_black:
            self.ks_castle_black = None

    def update_for_castling(self, ks, flip):
        if ks:
            if flip:
                self.ks_castle_white = True
            else:
                self.ks_castle_black = True
        else:
            if flip:
                self.qs_castle_white = True
            else:
                self.qs_castle_black = True

    def update_for_enpassant(self, col):
        self.enpassant = col

    def update_for_first_time(self, board):
        if self.first_time:
            self.first_time = False
            self.qs_castle_white = board.has_queenside_castling_rights(True)
            self.qs_castle_black = board.has_queenside_castling_rights(False)
            self.ks_castle_white = board.has_kingside_castling_rights(True)
            self.ks_castle_black = board.has_kingside_castling_rights(False)
            if board.has_legal_en_passant():
                self.enpassant = board.ep_square % 8
            else:
                self.enpassant = -1
            self.rule50 = board.halfmove_clock

    def copy(self):
        res = OptionalState();
        res.ks_castle_white = self.ks_castle_white
        res.ks_castle_black = self.ks_castle_black
        res.qs_castle_white = self.qs_castle_white
        res.qs_castle_black = self.qs_castle_black
        res.enpassant = self.enpassant
        res.rule50 = self.rule50
        res.first_time = self.first_time
        return res



def calculate_input(instruction, board, state):
    print('received:', instruction)
    # update board and calculate input
    board.reset()
    parts = instruction.split()
    started = False
    if len(parts) > 2 and parts[1] == "fen":
        last_idx = len(parts)
        if "moves" in parts:
            last_idx = parts.index("moves")
        # fen is from parts[2] to parts[last_idx-1] inclusive.
        board.set_fen(' '.join(parts[2:last_idx]))
    
    for i in range(len(parts)):
        if started:
            board.push_uci(parts[i])
        if parts[i] == 'moves':
            started = True
    state.update_for_first_time(board)
    print('OptionalState:',state.qs_castle_white,state.qs_castle_black,state.ks_castle_white,state.ks_castle_black,state.enpassant,state.rule50)
    return calculate_input_from_board(board, state)


def calculate_input_from_board(board, state):
    input_data = np.zeros((1,24,8,8), dtype=np.float32)
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
    if state.qs_castle_white is not None:
        input_data[0, 20 if flip else 18, : , :] = 1.0
        if state.qs_castle_white:
            row = 0
            if flip:
                row = 7 - row
            input_data[0, 12, row, 0] = 1.0
    if state.qs_castle_black is not None:
        input_data[0, 18 if flip else 20, : , :] = 1.0
        if state.qs_castle_black:
            row = 7
            if flip:
                row = 7 - row
            input_data[0, 12, row, 0] = 1.0
    if state.ks_castle_white is not None:
        input_data[0, 21 if flip else 19, : , :] = 1.0
        if state.ks_castle_white:
            row = 0
            if flip:
                row = 7 - row
            input_data[0, 13, row, 7] = 1.0
    if state.ks_castle_black is not None:
        input_data[0, 19 if flip else 21, : , :] = 1.0
        if state.ks_castle_black:
            row = 7
            if flip:
                row = 7 - row
            input_data[0, 13, row, 7] = 1.0
    if state.enpassant is not None:
        input_data[0, 22, : , :] = 1.0
        if state.enpassant > -1:
            input_data[0, 16, 7, state.enpassant] = 1.0
    if state.rule50 is not None:
        input_data[0, 23, :, :] = 1.0
        input_data[0, 17, :, :] = state.rule50 / 100.0
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


def updateBoardForIndex(board, state, max_idx, flip):
    from_sq = reverseTransformSq(max_idx % 64, flip)
    if board.piece_at(from_sq) is not None:
        # TODO: If adding support for 960 - this could be a valid castling move.
        raise ValueError("From square not empty.")
    row = from_sq // 8
    col = from_sq % 8
    #print('Square:',"abcdefgh"[col]+"12345678"[row])
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
        if to_row < 0 or to_row > 7 or to_col < 0 or to_col > 7:
            raise ValueError("Knight move off the board")
        #print('KnightMove','To:',"abcdefgh"[to_col]+"12345678"[to_row],'Captured:',cap_type)
        sq = to_row*8+to_col
        piece_moved = board.remove_piece_at(sq)
        if piece_moved is None:
            raise ValueError("To square for knight move was empty.")
        if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
            raise ValueError("Opponent piece moved.")
        if piece_moved.piece_type != chess.KNIGHT:
            raise ValueError("Piece other than a knight doing knight moves.")
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
        if cap_type > 0:
            state.update_for_rule50_reset()
        state.update_for_new_board(board)
        if not board.is_valid() or not check_extra_valid(board):
            raise ValueError("Board invalid after move")
        return "abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row]
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
                raise ValueError("Slide move goes off board")
            sq = to_row*8+to_col
            if board.piece_at(sq) is not None:
                break
            to_row = to_row - y_delta
            to_col = to_col - x_delta
        sq = to_row*8+to_col
        #print('SlideMove','To:',"abcdefgh"[to_col]+"12345678"[to_row],'Captured:',cap_type)
        piece_moved = board.remove_piece_at(sq)
        if piece_moved is None:
            raise ValueError("Slide move goes off board")
        if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
            raise ValueError("Opponent piece moved.")            
        if piece_moved.piece_type == chess.KNIGHT:
            raise ValueError("Knight doing slide moves.")
        if piece_moved.piece_type == chess.ROOK and direction not in [0,2,4,6]:
            raise ValueError("Rook doing diagonal moves.")
        if piece_moved.piece_type == chess.BISHOP and direction not in [1,3,5,7]:
            raise ValueError("Bishop doing horiz/vert moves.")
        if piece_moved.piece_type == chess.KING and (abs(to_row-row) > 1 or abs(to_col-col) > 1):
            raise ValueError("King sliding too far.")
        if piece_moved.piece_type == chess.PAWN and direction not in [1,2,3]:
            raise ValueError("Pawns can only move forward.")
        if piece_moved.piece_type == chess.PAWN and (abs(to_row-row) + abs(to_col-col) > 2):
            raise ValueError("Pawn sliding too far.")
        if piece_moved.piece_type == chess.PAWN and abs(to_row-row) == 2 and row not in [1,6]:
            raise ValueError("Pawn sliding too far for current rank.")
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
        if cap_type > 0 or board.piece_at(from_sq).piece_type == chess.PAWN:
            state.update_for_rule50_reset()
        state.update_for_new_board(board)
        if not board.is_valid() or not check_extra_valid(board):
            raise ValueError("Board invalid after move")
        return "abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row]
    elif move_type < 98:
        x_delta = -1
        y_delta = 1
        if move_type == 97:
            x_delta = 1
        x_delta, y_delta = reverseTransformDir(x_delta, y_delta, flip)
        to_row = row - y_delta
        to_col = col - x_delta
        sq = to_row*8+to_col
        #print('Enpassant', 'To:',"abcdefgh"[to_col]+"12345678"[to_row])
        # TODO: check enpassant captured position is empty.
        piece_moved = board.remove_piece_at(sq)
        if piece_moved is None:
            raise ValueError("Illegal enpassant")
        if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
            raise ValueError("Opponent piece moved.")
        if piece_moved.piece_type != chess.PAWN:
            raise ValueError("Piece other than a pawn doing enpassant.")
        board.set_piece_at(from_sq, piece_moved)
        board.set_piece_at(row*8+to_col, chess.Piece(chess.PAWN, chess.BLACK if flip else chess.WHITE))
        board.turn = chess.WHITE if flip else chess.BLACK
        state.update_for_rule50_reset()
        state.update_for_new_board(board)
        state.update_for_enpassant(to_col)
        if not board.is_valid() or not check_extra_valid(board):
            raise ValueError("Board invalid after move")
        return "abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row]
    elif move_type < 106:
        # TODO: support 960
        y_delta = 0
        if move_type == 98:
            x_delta = -1
        elif move_type == 105:
            x_delta = 1
        else:
            raise ValueError("Unsupported Castling")
        x_delta, y_delta = reverseTransformDir(x_delta, y_delta, flip)
        if x_delta > 0:
            #print('Castling, O-O')
            piece_moved = board.remove_piece_at(row*8+6)
            if piece_moved is None:
                raise ValueError("Illegal castling")
            if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
                raise ValueError("Opponent piece moved.")       
            if piece_moved.piece_type != chess.KING:
                raise ValueError("Piece other than a king doing castling.")
            board.set_piece_at(from_sq, piece_moved)
            piece_moved = board.remove_piece_at(row*8+5)
            if piece_moved is None:
                raise ValueError("Illegal castling")
            if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
                raise ValueError("Opponent piece moved.")            
            if piece_moved.piece_type != chess.ROOK:
                raise ValueError("Piece other than a rook in castling.")
            #TODO: verify destination spot is empty (unless supporting 960 castling)
            board.set_piece_at(row*8+7, piece_moved)
        else:
            #print('Castling, O-O-O')
            piece_moved = board.remove_piece_at(row*8+2)
            if piece_moved is None:
                raise ValueError("Illegal castling")
            if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
                raise ValueError("Opponent piece moved.")            
            if piece_moved.piece_type != chess.KING:
                raise ValueError("Piece other than a king doing castling.")
            board.set_piece_at(from_sq, piece_moved)
            piece_moved = board.remove_piece_at(row*8+3)
            if piece_moved is None:
                raise ValueError("Illegal castling")
            if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
                raise ValueError("Opponent piece moved.")            
            if piece_moved.piece_type != chess.ROOK:
                raise ValueError("Piece other than a rook in castling.")
            #TODO: verify destination spot is empty (unless supporting 960 castling)
            board.set_piece_at(row*8, piece_moved)
        board.turn = chess.WHITE if flip else chess.BLACK
        state.update_for_new_board(board)
        state.update_for_castling(x_delta > 0, flip)
        if not board.is_valid() or not check_extra_valid(board):
            raise ValueError("Board invalid after move")
        return "abcdefgh"[col]+"12345678"[row]+"abcdefgh"[7 if x_delta > 0 else 0]+"12345678"[row]
    elif move_type < 117:
        y_delta = 1
        if move_type == 106:
            x_delta = 0
            cap_type = 0
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
        #print('Promotion','To:',"abcdefgh"[to_col]+"12345678"[to_row],'Captured:',cap_type)
        piece_moved = board.remove_piece_at(sq)
        if piece_moved is None:
            raise ValueError("Illegal promotion")
        if piece_moved.color != (chess.WHITE if flip else chess.BLACK):
            raise ValueError("Opponent piece moved.")            
        if piece_moved.piece_type == chess.KING or piece_moved.piece_type == chess.PAWN:
            raise ValueError("Promotion to pawn or king, shouldn't happen.")
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
        state.update_for_rule50_reset()
        state.update_for_new_board(board)
        if not board.is_valid() or not check_extra_valid(board):
            raise ValueError("Board invalid after move")
        return "abcdefgh"[col]+"12345678"[row]+"abcdefgh"[to_col]+"12345678"[to_row]+piece_moved.symbol().lower()
    else:
        raise ValueError("Illegal move type")


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
        policy,moves,r50_est = tfprocess.model(input_data, training=False)
        occupency_policy_mask = tf.reshape(tf.tile(tf.reduce_max(input_data[:,0:12,:, :], axis=1, keepdims=True), [1, 128, 1, 1]), [-1, 128*64])
        illegal_filler = tf.zeros_like(policy) - 1.0e10
        policy = tf.where(tf.equal(occupency_policy_mask, 0), policy, illegal_filler)
        max_idx = tf.argmax(policy, 1)
        max_value = tf.gather(policy, max_idx, axis=-1)
        indicies = tf.where(tf.greater_equal(policy, max_value - 9.))
        # TODO: this code 'works' only if the batch size of input is 1.
        high_policy = tf.gather_nd(policy, indicies)
        sort_order = tf.argsort(high_policy, direction="DESCENDING")
        sorted_high_policy = tf.gather(high_policy, sort_order)
        sorted_high_policy = tf.nn.softmax(sorted_high_policy)
        sorted_indicies = tf.gather(indicies, sort_order)
        return policy, moves, r50_est, sorted_high_policy, sorted_indicies[:,1]


    state = OptionalState()
    input_data, flip = calculate_input('position startpos', board, state)

    outputs = first(input_data)

    reco_moves = []
    target_moves = None

    instruction_override = ""
    while True:
        instruction = instruction_override if instruction_override != "" else input()
        if instruction == 'uci':
            print('id name Reverser')
            print('id author Reverser Authors')
            print('uciok')
        elif instruction.startswith('position '):
            reco_moves = []
            target_moves = None
            state = OptionalState()
            pos_start = timer()
            input_data, flip = calculate_input(instruction, board, state)
            pos_end = timer()
            #print('timed {}'.format(pos_end-pos_start))

        elif instruction.startswith('go '):
            go_start = timer()
            # Do evil things that are not uci compliant... This loop should be on a different thread so it can be interrupted by stop, if this was actually uci :P
            policy, moves, r50_est, sorted_high_policy, sorted_indicies = first(input_data)
            max_idx = tf.argmax(policy, 1)[0]
            max_value = policy[0, max_idx]
            print()
            print('Moves from start:',moves[0,0])
            print('Rule 50 est:',r50_est[0,0])
            print('Max Policy value:',max_value,'index:',max_idx)
            print('Move type:', max_idx//64, 'Starting Square:', max_idx % 64)
            if target_moves is None:
                # Add some bonus here to account for moves estimate not being very accurate.
                # Can't go too big though or it'll create 3 folds too frequently.
                target_moves = moves[0,0] + 8
                # If fullmove_number is larger, use that instead. Users who provide a value of 1 without thinking won't be affected since the base target_moves is always greater already.
                alternative = (board.fullmove_number - 1) * 2 + (1 if flip else 0)
                target_moves = max(target_moves, alternative)
            else:
                target_moves = target_moves - 1
            root_node = SearchNode()
            root_node.board = board
            root_node.flip = flip
            root_node.state = state
            root_node.input_data = input_data
            root_node.policy = sorted_high_policy
            root_node.policy_index = sorted_indicies
            root_node.visits = 1
            root_node.total_move_est = utility_calc(moves[0,0], target_moves)
            root_node.children = []
            for i in range(100):
                root_node.visit(first, target_moves)
            max_count = -1
            for i in range(len(root_node.children)):
                if root_node.children[i].visits > max_count:
                    max_idx = root_node.policy_index[i]
                    max_count = root_node.children[i].visits
                print('Child:',root_node.children[i].visits, root_node.policy_index[i])
            
            reco_moves.append(updateBoardForIndex(board, state, max_idx, flip))
            input_data, flip = calculate_input('position fen '+board.fen(), board, state)
            bestmove = '0000'
            print('Recomoves:',list(reversed(reco_moves)))
            instruction_override = instruction
            if board.board_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" and not flip:
                board.reset()
                for uci_move in reversed(reco_moves):
                    board.push_uci(uci_move)
                print(chess.pgn.Game.from_board(board))
                instruction_override = ""
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
