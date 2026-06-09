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


import requests
requests.get('http://www.google.com',timeout=10).ok

!pip install --upgrade kaggle-environments
!pip install pygame


from kaggle_environments import make
env = make("chess", debug=True)


result = env.run(["random", "random"])
env.render(mode="ipython", width=1000, height=1000) 


%%writefile main.py
from Chessnut import Game
import random

def chess_bot(obs):


    game = Game(obs.board)
    moves = list(game.get_moves())

    # 1.For checkmate
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # 2. For captures
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    # 3. For queen promotions
    for move in moves:
        if "q" in move.lower():
            return move

    # 4. Random move if no checkmates or captures
    return random.choice(moves)


def my_agent(obs, config):
    return chess_bot(obs)

result = env.run([my_agent, "random"])

env.render(mode="ipython", width=1000, height=1000)



agent_0_reward = result[-1][0]["reward"]
agent_1_reward = result[-1][1]["reward"]

print("Game result:")
print(f"Your agent's reward: {agent_0_reward}")
print(f"Opponent's reward: {agent_1_reward}")


win_count = 0
loss_count = 0
draw_count = 0
n_games = 100  # Number of games to test
for _ in range(n_games):
    game_result = env.run([my_agent, "random"])[-1]
    reward = game_result[0]["reward"]
    if reward == 1:
        win_count += 1
    elif reward == -1:
        loss_count += 1
    else:
        draw_count += 1

print(f"Performance after {n_games} games:")
print(f"Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")




