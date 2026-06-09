# first let's make sure you have internet enabled
import requests
requests.get('http://www.google.com',timeout=10).ok


from warnings import filterwarnings
filterwarnings('ignore')


%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments


from kaggle_environments import make
env = make("chess", debug=True)


result = env.run(["random", "random"])
env.render(mode="ipython", width=1000, height=1000) 


%%writefile submission.py
from Chessnut import Game
import random

def chess_bot(obs):
  
    game = Game(obs.board)
    moves = list(game.get_moves())

    # 1. Check a subset of moves for checkmate
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # 2. Check for captures
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    # 3. Check for queen promotions
    for move in moves:
        if "q" in move.lower():
            return move

    # 4. Random move if no checkmates or captures
    return random.choice(moves)


from kaggle_environments import make
from submission import chess_bot   
env = make("chess", debug=True)
env.run([chess_bot, "random"])
env.render(mode="ipython", width=1000, height=1000)

