# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from Chessnut import Game
import random
import time

# Chess bot code
class AdvancedChessBot:
    def __init__(self):
        self.piece_values = {
            " ": 0, "p": 100, "P": 100, "n": 320, "N": 320, "b": 330, "B": 330, "r": 500,
            "R": 500, "q": 900, "Q": 900, "k": 20000, "K": 20000
        }
        self.board_maps = {
            'P': {
                'early': { "a2": 0, "b2": 5, "c2": 10, "d2": 20, "e2": 20, "f2": 10, "g2": 5, "h2": 0,
                          "a3": 5, "b3": 10, "c3": 15, "d3": 25, "e3": 25, "f3": 15, "g3": 10, "h3": 5 },
                'late': { "a7": 50, "b7": 50, "c7": 50, "d7": 50, "e7": 50, "f7": 50, "g7": 50, "h7": 50 }
            },
            'N': {'placement': { "b1": 5, "c1": 10, "f1": 10, "g1": 5, "b8": 5, "c8": 10, "f8": 10, "g8": 5 }},
            'K': {
                'early': { "e1": -20, "d1": -10, "f1": -10, "e8": -20, "d8": -10, "f8": -10 },
                'late': { "e1": 10, "d1": 5, "f1": 5, "e8": 10, "d8": 5, "f8": 5 }
            }
        }
        self.CAPTURE_BONUS = 50
        self.THREAT_PENALTY = 30
        self.MOBILITY_WEIGHT = 0.1
        self.CENTER_CONTROL_WEIGHT = 5

    def evaluate_position(self, game_board):
        score = 0
        piece_counts = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0, 'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0}
        center_squares = ['d4', 'd5', 'e4', 'e5']
        
        for square_index in range(64):
            piece = game_board.get_piece(square_index)
            if piece != ' ':
                piece_upper = piece.upper()
                value = self.piece_values.get(piece_upper, 0)
                square = Game.i2xy(square_index)
                if piece_upper in self.board_maps:
                    for map_type, map_data in self.board_maps[piece_upper].items():
                        if square in map_data:
                            bonus = map_data[square]
                            value += bonus if piece.isupper() else -bonus
                score += value if piece.isupper() else -value
                piece_counts[piece_upper] += 1  # Increment count for both uppercase and lowercase pieces

        for square in center_squares:
            piece = game_board.get_piece(Game.xy2i(square))
            if piece != ' ':
                score += self.CENTER_CONTROL_WEIGHT if piece.isupper() else -self.CENTER_CONTROL_WEIGHT

        return score

    def evaluate_move_safety(self, game_instance, move):
        # Now using `obs.board` directly, which contains the FEN string
        test_game = Game(fen=game_instance.get_fen())  # Use the FEN string from `game_instance.get_fen()`
        test_game.apply_move(move)
        return test_game.status != Game.CHECK

    def select_best_move(self, game_state, time_limit=1.5):
        start_time = time.time()
        game_instance = Game(fen=game_state.board)  # Use `game_state.board` directly as the FEN string
        moves = list(game_instance.get_moves())
        
        if game_state.remainingOverageTime < 2:
            return random.choice(moves)

        move_scores = []
        initial_evaluation = self.evaluate_position(game_instance.board)

        for move in moves:
            if not self.evaluate_move_safety(game_instance, move):
                continue

            test_game = Game(fen=game_state.board)  # Use `game_state.board` directly as the FEN string
            test_game.apply_move(move)
            post_move_eval = self.evaluate_position(test_game.board)
            evaluation_diff = post_move_eval - initial_evaluation

            score = evaluation_diff
            capture_piece = test_game.board.get_piece(Game.xy2i(move[2:4]))
            if capture_piece != ' ':
                score += self.CAPTURE_BONUS

            if 'q' in move.lower():
                score += 100

            if test_game.status == Game.CHECK:
                score += 30

            move_scores.append((move, score))

        move_scores.sort(key=lambda x: x[1], reverse=True)

        try:
            for move, score in move_scores:
                if time.time() - start_time > time_limit:
                    break
                if score > 0:
                    return move
        except Exception:
            pass

        return random.choice(moves)

# Wrapper function to interface with Kaggle environment
def chess_bot(obs):
    bot = AdvancedChessBot()
    time_limit = 1.35 * (obs.remainingOverageTime / 10)
    game = Game(fen=obs.board)  # Use `obs.board` directly as the FEN string
    return bot.select_best_move(obs, time_limit)


