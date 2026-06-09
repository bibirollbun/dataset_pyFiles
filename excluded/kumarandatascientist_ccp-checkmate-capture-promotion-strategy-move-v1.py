import requests
requests.get('http://www.google.com',timeout=10).ok
!pip install --upgrade kaggle-environments
from kaggle_environments import make
env = make("chess", debug=True)


!pip install Chessnut


%%writefile submission.py
import json
import time
import csv
import os
import glob
from Chessnut import Game
import random

class StrategicChessAI:
    def __init__(self):
        self.win_streak = 0
        self.mu = 600
        self.log_file = f"chess_log_{int(time.time())}.csv"
        self._init_csv()

    def _init_csv(self):
        with open(self.log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['player1', 'player2', 'total_moves', 'status', 'timestamp'])

    def calculate_move(self, game):
        moves = list(game.get_moves())

        # Check for checkmates first
        for move in moves:
            temp_game = Game(game.get_fen())
            temp_game.apply_move(move)
            if temp_game.status == Game.CHECKMATE:
                return move

        # Then check for captures
        capture_moves = []
        for move in moves:
            target_pos = Game.xy2i(move[2:4])
            if game.board.get_piece(target_pos) != ' ':
                capture_moves.append(move)
        if capture_moves:
            return random.choice(capture_moves)

        # Then check for promotions
        promotion_moves = [move for move in moves if 'q' in move.lower()]
        if promotion_moves:
            return random.choice(promotion_moves)

        # Fallback to random move
        return random.choice(moves)

def agent(observation, configuration):
    """
    Kaggle competition entry point that conforms to the expected API:
    - observation: contains 'board' (FEN string) and 'remainingTime' 
    - configuration: contains game parameters
    Returns move in UCI format
    """
    ai = StrategicChessAI()
    game = Game(observation.board)
    return ai.calculate_move(game)


from kaggle_environments import make

# Create the chess environment
env = make("chess", debug=True)

# Run the agent against a random opponent
result = env.run(["submission.py", "random"])

# Print the agent's final status, reward, and remaining overage time
print("Agent exit status/reward/time left:")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.get("remainingOverageTime", "N/A"))

# Render the game
env.render(mode="ipython", width=800, height=800)





