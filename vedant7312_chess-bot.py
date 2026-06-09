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


%%capture
!pip install --upgrade kaggle-environments


from kaggle_environments import make
env = make("chess", debug=True)



result = env.run(["random", "random"])
env.render(mode="ipython", width=1000, height=1000) 








!pip install chess





!pip install cairosvg


# Import necessary libraries
import chess
import chess.engine
import numpy as np
import random
import chess.svg
from IPython.display import SVG, display


%%writefile main.py

# Create a simple chess engine
class SimpleChessEngine:
    def __init__(self):
        self.board = chess.Board()

    def make_move(self):
        legal_moves = list(self.board.legal_moves)
        move = random.choice(legal_moves)
        self.board.push(move)
        return move

    def reset(self):
        self.board.reset()

    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self):
        result = self.board.result()
        if result == '1-0':
            return 1  # White wins
        elif result == '0-1':
            return -1  # Black wins
        else:
            return 0  # Draw

# Simulate a game between two simple chess engines
def simulate_game(engine1, engine2):
    engine1.reset()
    engine2.reset()
    while not engine1.is_game_over():
        engine1.make_move()
        if engine1.is_game_over():
            break
        engine2.make_move()
    return engine1.get_result(), engine1.board

# Create two instances of the simple chess engine
engine1 = SimpleChessEngine()
engine2 = SimpleChessEngine()

# Simulate a game and get the result and final board position
result, final_board = simulate_game(engine1, engine2)
print(f"Game result: {result}")

# Display the final board position
display(SVG(chess.svg.board(final_board)))




