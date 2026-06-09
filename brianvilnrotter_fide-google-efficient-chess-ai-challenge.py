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


# from Your First Chess Bot (https://www.kaggle.com/code/bovard/your-first-chess-bot)
import requests
requests.get('http://www.google.com',timeout=10).ok # This is to ensure that the notebook has access to the internet


# Now to setup the chess environment!
from kaggle_environments import make
env = make("chess",debug=True) # pygame is not installed, so deprication warnings are ignored


%%writefile submission.py
# Now to create an agent
from Chessnut import Game
import pandas as pd
import random,warnings
warnings.filterwarnings('ignore')

piece_value = {"q":0.6,"r":0.5,"n":0.3,"b":0.3,"p":0.1," ":0.0,"k":0.0}

def is_targeted(move,moves):
    return sum([1 if move[2:4] in list(map(lambda x:x[2:4],moves)) else 0]) > 1

def get_group_targeted(game,Game,move,moves) -> list:
    return [piece_value[piece] for piece in [game.board.get_piece(Game.xy2i(space)).lower() for space in [move[2:4] for move in moves]]]
    
def chess_bot(obs):
    game = Game(obs.board)

    # set the players
    if game.state.player == 'w':
        player,opp = ('w','b')
    else:
        player,opp = ('b','w')
    
    moves = list(game.get_moves())
    opp_moves = list(game.get_moves(player=opp))
    
    pd_moves = pd.DataFrame({'Moves':list(game.get_moves())})  
    pd_moves["Value"] = 0.0
    applied_board = []
    for move in moves:
        g = Game(obs.board)
        g.apply_move(move)
        applied_board.append(g)
    pd_moves['next'] = applied_board  

    # add values equal to all the pieces that opponent can reach next turn
    #pd_moves["Value"] += [sum(-1*get_group_targeted(game,Game,move,opp_moves)) for move in pd_moves["Moves"]]
    # subtract values equal to all the pieces that opponent can take
    #pd_moves["Value"] += [sum(get_group_targeted(game,Game,move,moves)) for move in pd_moves["Moves"]]
    # below is the added value per piece being captured
    pd_moves["Value"] += [piece_value[game.board.get_piece(Game.xy2i(move[2:4])).lower()] if game.board.get_piece(Game.xy2i(move[2:4])) != " " else 0 for move in moves]
    # below is checking for queen promotion
    pd_moves["Value"] += [0.7 if "q" in move.lower() else 0 for move in moves]
    # move piece if being attacked
    pd_moves["Value"] += [piece_value[game.board.get_piece(Game.xy2i(move[:2])).lower()] if game.board.get_piece(Game.xy2i(move[:2])) not in [' ','k','K'] else 0 for move in moves]
    # okay if protected
    pd_moves["Value"] += [-1*piece_value[game.board.get_piece(Game.xy2i(move[:2])).lower()] if game.board.get_piece(Game.xy2i(move[:2])) not in [' ','k','K'] else 0 for move in moves]
    # check if moves are attacked but protected
    pd_moves["Value"] += [piece_value[game.board.get_piece(Game.xy2i(move[2:4])).lower()] if is_targeted(move,moves) and is_targeted(move,opp_moves) else 0 for move in moves]
    # definitely do the move if it is checkmate
    pd_moves['Value'] += [99 if row[2].status == Game.CHECKMATE else 0 for _,row in pd_moves.iterrows()]
    # definitely do not move if stalemate
    pd_moves['Value'] += [-99 if row[2].status == Game.STALEMATE else 0 for _,row in pd_moves.iterrows()]
    # prioritize move if protected check
    pd_moves['Value'] += [1 if row[2] and is_targeted(pd_moves.loc[i,'Moves'],pd_moves['Moves']) else 0 for i,row in pd_moves.iterrows()]

    print(pd_moves['Value'])
    #print(pd_moves.loc[pd_moves["Value"] == pd_moves["Value"].max(),["Moves","Value"]])
    # Random move if no checkmates or captures
    return random.choice(pd_moves.loc[pd_moves["Value"] == pd_moves["Value"].max(),"Moves"].tolist())


# now lets test the agent
result = env.run(["submission.py","submission.py"])
print("Agent exit status/reward/time left:")
# look for the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t",agent.status,"/",agent.reward,"/",agent.observation.remainingOverageTime)
print("/n")
env.render(mode="ipython",width=1000,height=1000)




