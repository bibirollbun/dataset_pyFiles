#!/usr/bin/env python
# coding: utf-8
"""
FIDE & Google Efficient Chess AI Challenge
---------------------------------------------------------------
This script demonstrates:
 - Environment setup (including package installs)
 - Creating and writing several chess agent files
 - Running tests and tournaments between agents
 - Generating plots and logging game outcomes
 - Creating a submission file from one of the agents

Requirements (will be installed if missing):
    requests, kaggle-environments, Chessnut, matplotlib, seaborn, tqdm, scikit-learn

Author: Your Name
Date: YYYY-MM-DD
"""

import sys
import subprocess
import os
import time
import random
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =============================================================================
# Helper Function: pip install (if required)
# =============================================================================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# =============================================================================
# Ensure Required Packages are Installed
# =============================================================================
try:
    import requests
except ImportError:
    install("requests")
    import requests

try:
    from kaggle_environments import make
except ImportError:
    install("kaggle-environments")
    from kaggle_environments import make

try:
    from Chessnut import Game
except ImportError:
    install("Chessnut")
    from Chessnut import Game

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    install("seaborn")
    import seaborn as sns

try:
    from tqdm import tqdm
except ImportError:
    install("tqdm")
    from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    install("scikit-learn")
    from sklearn.metrics import confusion_matrix

# =============================================================================
# Step 1: Check Internet Connectivity
# =============================================================================
def check_internet():
    try:
        response = requests.get('http://www.google.com', timeout=10)
        if response.ok:
            print("Internet connectivity: OK")
        else:
            print("Internet connectivity: FAILED")
    except Exception as e:
        print("Internet connectivity: FAILED", e)

check_internet()

# =============================================================================
# Step 2: Upgrade kaggle-environments Package
# =============================================================================
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "kaggle-environments"])

# =============================================================================
# Step 3: Set Up the Chess Environment
# =============================================================================
env = make("chess", debug=True)

# Run one game between two random agents to ensure environment works
result = env.run(["random", "random"])
print("Random vs Random game result:")
for agent in result[-1]:
    print("Status:", agent.status, "/ Reward:", agent.reward)
# (In a Jupyter notebook you might use: env.render(mode="ipython", width=700, height=700))

# =============================================================================
# Step 4: Write the Initial Agent File (initial_agent.py)
# =============================================================================
initial_agent_code = '''\
from Chessnut import Game
import random

def heuristic_chess_bot(obs):
    """
    A heuristic-based chess bot that prioritizes:
      - Checkmates
      - Captures
      - Promotions
      - Random moves
    Args:
        obs: Object with 'board' representing board state as FEN string
    Returns:
        Move in UCI notation
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    random.shuffle(moves)  # Randomize moves to add variation

    # Prioritize checkmates
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # Check for captures
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    # Check for promotions
    for move in moves:
        if "q" in move.lower():  # Queen promotion
            return move

    # Default to random move
    return random.choice(moves)
'''
with open("initial_agent.py", "w") as f:
    f.write(initial_agent_code)
print("initial_agent.py written.")

# =============================================================================
# Step 5: Test the Initial Agent Against a Random Agent
# =============================================================================
print("\nTesting initial_agent.py against random agent:")
result = env.run(["initial_agent.py", "random"])
for agent in result[-1]:
    # Using getattr for remainingOverageTime since it might not exist in some environments.
    print("Status:", agent.status, "/ Reward:", agent.reward, "/ Time left:", getattr(agent.observation, 'remainingOverageTime', 'N/A'))

# =============================================================================
# Step 6: Write Agent LittleDeepBlue_0_5_5.py
# =============================================================================
littledeepblue_0_5_5_code = '''\
from Chessnut import Game
import time
import random

PIECE_VALUES = {'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000}

def evaluate_board(game):
    """
    Evaluate the board position from the perspective of the current player.
    Positive scores are good for the current player, negative for the opponent.
    """
    board_str = str(game.board)
    score = 0
    for char in board_str:
        if char.isalpha():
            if char.isupper():
                score += PIECE_VALUES[char.lower()]
            else:
                score -= PIECE_VALUES[char]
    return score

def minimax(game, depth, alpha, beta, maximizing_player, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return None, None
    if depth == 0 or game.status >= 2:
        return evaluate_board(game), None
    moves = list(game.get_moves())
    if not moves:
        return evaluate_board(game), None
    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in moves:
            if time.time() - start_time > time_limit:
                break
            new_game = Game(str(game))
            new_game.apply_move(move)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, False, start_time, time_limit)
            if eval_score is None:
                continue
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            if time.time() - start_time > time_limit:
                break
            new_game = Game(str(game))
            new_game.apply_move(move)
            eval_score, _ = minimax(new_game, depth - 1, alpha, beta, True, start_time, time_limit)
            if eval_score is None:
                continue
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def heuristic_chess_bot(obs):
    game = Game(obs.board)
    moves = list(game.get_moves())
    best_score = float('-inf')
    best_move = None
    for move in moves:
        move_score = 0
        from_square = move[:2]
        to_square = move[2:4]
        piece_moved = game.board.get_piece(Game.xy2i(from_square))
        target_piece = game.board.get_piece(Game.xy2i(to_square))
        if target_piece != ' ':
            move_score += PIECE_VALUES[target_piece.lower()] - PIECE_VALUES[piece_moved.lower()]
        if piece_moved.lower() in ['n', 'b', 'q']:
            if (piece_moved.isupper() and from_square[1] == '1') or (piece_moved.islower() and from_square[1] == '8'):
                move_score += 10
        if "q" in move.lower():
            move_score += PIECE_VALUES['q']
        if move in ['e1g1', 'e1c1', 'e8g8', 'e8c8']:
            move_score += 50
        if move_score > best_score:
            best_score = move_score
            best_move = move
    if best_move:
        return best_move
    else:
        return random.choice(moves)
'''
with open("LittleDeepBlue_0_5_5.py", "w") as f:
    f.write(littledeepblue_0_5_5_code)
print("LittleDeepBlue_0_5_5.py written.")

# =============================================================================
# Step 7: Write Agent LittleDeepBlue_0_10_4_debug.py (Debug Version)
# =============================================================================
littledeepblue_0_10_4_debug_code = '''\
from Chessnut import Game
import time
import random

DEBUG = True

PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}

def log_message(message):
    """Helper function to print messages if DEBUG is True."""
    if DEBUG:
        current_time = time.time()
        print(f"[{current_time:.2f}] {message}")

def enhanced_heuristic_with_time_limit(obs, per_move_time, remaining_time):
    """
    Enhanced heuristic evaluation with adjusted scoring to prioritize captures and protect endangered pieces.
    Includes detailed logging via log_message.
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    random.shuffle(moves)
    log_message(f"Available moves: {moves}")

    best_score = float('-inf')
    best_move = None
    start_time = time.time()

    for move in moves:
        move_start = time.time()
        elapsed = move_start - start_time
        remaining = remaining_time - elapsed
        if remaining <= 0:
            log_message("Time limit for heuristic evaluation reached. Stopping.")
            break

        log_message(f"Evaluating move: {move}. Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s")

        if elapsed >= per_move_time:
            log_message(f"Skipping move {move} due to per-move time limit.")
            continue

        move_score = 0
        from_square = move[:2]
        to_square = move[2:4]

        new_game = Game(str(game))
        new_game.apply_move(move)

        if new_game.status == Game.CHECKMATE:
            log_message(f"Checkmate detected with move: {move}. Returning immediately.")
            return move

        piece_moved = game.board.get_piece(Game.xy2i(from_square))
        target_piece = game.board.get_piece(Game.xy2i(to_square))

        if target_piece != ' ':
            capture_value = PIECE_VALUES[target_piece.lower()]
            move_score += capture_value * 100
            log_message(f"Move {move} captures {target_piece}. Capture value: {capture_value}. Updated score: {move_score}")

            piece_value = PIECE_VALUES.get(piece_moved.lower(), 0)
            net_gain = capture_value - piece_value
            move_score += net_gain * 50
            log_message(f"Net gain from exchange: {net_gain}. Updated score: {move_score}")

        if 'q' in move.lower():
            move_score += PIECE_VALUES['q'] * 100
            log_message(f"Move {move} promotes to queen. Updated score: {move_score}")

        if to_square in ['d4', 'e4', 'd5', 'e5']:
            move_score += 20
            log_message(f"Move {move} controls center. Updated score: {move_score}")

        if new_game.status == Game.CHECK:
            move_score += 30
            log_message(f"Move {move} gives check. Updated score: {move_score}")

        opponent_moves = list(new_game.get_moves())
        for opp_move in opponent_moves:
            opp_target_square = opp_move[2:4]
            own_piece = new_game.board.get_piece(Game.xy2i(opp_target_square))
            if own_piece != ' ' and own_piece.isupper():
                endangered_piece_value = PIECE_VALUES.get(own_piece.lower(), 0)
                move_score -= endangered_piece_value * 100
                log_message(f"Own piece {own_piece} at {opp_target_square} is endangered. Penalizing move. Updated score: {move_score}")

        if move_score > best_score:
            best_score = move_score
            best_move = move
            log_message(f"New best move found: {best_move}. Best score updated to: {best_score}")

        move_end = time.time()
        log_message(f"Move {move} evaluation completed in {move_end - move_start:.2f}s.")

    log_message(f"Best move chosen: {best_move} with final score: {best_score}")
    return best_move if best_score > float('-inf') else None

def agent(obs, config):
    try:
        start_time = time.time()
        total_time_limit = 8.0
        per_move_time_limit = 0.1

        log_message("Starting agent evaluation")

        while True:
            elapsed_time = time.time() - start_time
            remaining_time = total_time_limit - elapsed_time

            log_message(f"Total elapsed: {elapsed_time:.2f}s, Remaining time: {remaining_time:.2f}s")

            if remaining_time <= 0.1:
                log_message("Time is almost up. Defaulting to a random move.")
                break

            move = enhanced_heuristic_with_time_limit(obs, per_move_time=per_move_time_limit, remaining_time=remaining_time)
            if move:
                log_message(f"Heuristic selected move: {move}")
                return move

        game = Game(obs.board)
        random_move = random.choice(list(game.get_moves()))
        log_message(f"Random move chosen: {random_move}")
        return random_move

    except Exception as e:
        log_message(f"Error in agent: {e}")
        game = Game(obs.board)
        fallback_move = random.choice(list(game.get_moves()))
        log_message(f"Fallback random move chosen: {fallback_move}")
        return fallback_move
'''
with open("LittleDeepBlue_0_10_4_debug.py", "w") as f:
    f.write(littledeepblue_0_10_4_debug_code)
print("LittleDeepBlue_0_10_4_debug.py written.")

# =============================================================================
# Step 8: Write Agent LittleDeepBlue_0_10_4.py (Non-debug version)
# =============================================================================
littledeepblue_0_10_4_code = '''\
from Chessnut import Game
import time
import random

DEBUG = False

PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}

def log_message(message):
    """Helper function to print messages if DEBUG is True."""
    if DEBUG:
        current_time = time.time()
        print(f"[{current_time:.2f}] {message}")

def enhanced_heuristic_with_time_limit(obs, per_move_time, remaining_time):
    """
    Enhanced heuristic evaluation with adjusted scoring to prioritize captures and protect endangered pieces.
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    random.shuffle(moves)

    best_score = float('-inf')
    best_move = None
    start_time = time.time()

    for move in moves:
        move_start = time.time()
        elapsed = move_start - start_time
        remaining = remaining_time - elapsed
        if remaining <= 0:
            break

        if elapsed >= per_move_time:
            continue

        move_score = 0
        from_square = move[:2]
        to_square = move[2:4]

        new_game = Game(str(game))
        new_game.apply_move(move)

        if new_game.status == Game.CHECKMATE:
            return move

        piece_moved = game.board.get_piece(Game.xy2i(from_square))
        target_piece = game.board.get_piece(Game.xy2i(to_square))

        if target_piece != ' ':
            capture_value = PIECE_VALUES[target_piece.lower()]
            move_score += capture_value * 100

            piece_value = PIECE_VALUES.get(piece_moved.lower(), 0)
            net_gain = capture_value - piece_value
            move_score += net_gain * 50

        if 'q' in move.lower():
            move_score += PIECE_VALUES['q'] * 100

        if to_square in ['d4', 'e4', 'd5', 'e5']:
            move_score += 20

        if new_game.status == Game.CHECK:
            move_score += 30

        opponent_moves = list(new_game.get_moves())
        for opp_move in opponent_moves:
            opp_target_square = opp_move[2:4]
            own_piece = new_game.board.get_piece(Game.xy2i(opp_target_square))
            if own_piece != ' ' and own_piece.isupper():
                endangered_piece_value = PIECE_VALUES.get(own_piece.lower(), 0)
                move_score -= endangered_piece_value * 100

        if move_score > best_score:
            best_score = move_score
            best_move = move

    return best_move if best_score > float('-inf') else None

def agent(obs, config):
    try:
        start_time = time.time()
        total_time_limit = 8.0
        per_move_time_limit = 0.1

        while True:
            elapsed_time = time.time() - start_time
            remaining_time = total_time_limit - elapsed_time

            if remaining_time <= 0.1:
                break

            move = enhanced_heuristic_with_time_limit(obs, per_move_time=per_move_time_limit, remaining_time=remaining_time)
            if move:
                return move

        game = Game(obs.board)
        random_move = random.choice(list(game.get_moves()))
        return random_move

    except Exception:
        game = Game(obs.board)
        fallback_move = random.choice(list(game.get_moves()))
        return fallback_move
'''
with open("LittleDeepBlue_0_10_4.py", "w") as f:
    f.write(littledeepblue_0_10_4_code)
print("LittleDeepBlue_0_10_4.py written.")

# =============================================================================
# Step 9: Test the Agents
# =============================================================================
print("\n--- Testing Agents ---")

# Test A: LittleDeepBlue_0_10_4_debug.py vs Random
print("\nTest: LittleDeepBlue_0_10_4_debug.py vs random agent")
result = env.run(["LittleDeepBlue_0_10_4_debug.py", "random"])
for agent_info in result[-1]:
    print("Status:", agent_info.status, "/ Reward:", agent_info.reward, 
          "/ Time left:", getattr(agent_info.observation, 'remainingOverageTime', 'N/A'))

# Test B: LittleDeepBlue_0_10_4_debug.py vs LittleDeepBlue_0_10_4.py
print("\nTest: LittleDeepBlue_0_10_4_debug.py vs LittleDeepBlue_0_10_4.py")
result = env.run(["LittleDeepBlue_0_10_4_debug.py", "LittleDeepBlue_0_10_4.py"])
for agent_info in result[-1]:
    print("Status:", agent_info.status, "/ Reward:", agent_info.reward, 
          "/ Time left:", getattr(agent_info.observation, 'remainingOverageTime', 'N/A'))

# Test C: LittleDeepBlue_0_10_4_debug.py vs initial_agent.py
print("\nTest: LittleDeepBlue_0_10_4_debug.py vs initial_agent.py")
result = env.run(["LittleDeepBlue_0_10_4_debug.py", "initial_agent.py"])
for agent_info in result[-1]:
    print("Status:", agent_info.status, "/ Reward:", agent_info.reward, 
          "/ Time left:", getattr(agent_info.observation, 'remainingOverageTime', 'N/A'))

# =============================================================================
# Step 10: Tournament Testing Between Agents
# =============================================================================
print("\n--- Tournament: LittleDeepBlue_0_10_4.py vs initial_agent.py ---")
num_games = 50
wins = [0, 0]  # wins[0] for agent1, wins[1] for agent2
draws = 0
rewards = [0, 0]

# Store results for visualization
results = {
    "Agent 1 Wins": [],
    "Agent 2 Wins": [],
    "Draws": [],
    "Agent 1 Rewards": [],
    "Agent 2 Rewards": []
}

for game in tqdm(range(num_games), desc="Tournament Games"):
    result = env.run(["LittleDeepBlue_0_10_4.py", "initial_agent.py"])
    for i, agent_info in enumerate(result[-1]):
        if agent_info.status == "WINNER":
            wins[i] += 1
        elif agent_info.status == "DRAW":
            draws += 1
        rewards[i] += agent_info.reward if agent_info.reward is not None else 0
    agent_1_status = result[-1][0].status
    agent_2_status = result[-1][1].status
    if agent_1_status != "Timeout" and agent_2_status != "Timeout":
        print(f"Game {game + 1}: Agent 1 Status: {agent_1_status}, Agent 2 Status: {agent_2_status}")

    # Append results for visualization
    results["Agent 1 Wins"].append(wins[0])
    results["Agent 2 Wins"].append(wins[1])
    results["Draws"].append(draws)
    results["Agent 1 Rewards"].append(rewards[0])
    results["Agent 2 Rewards"].append(rewards[1])

avg_rewards = [reward / num_games for reward in rewards]
print(f"\nTotal games played: {num_games}")
print(f"Agent 1 Wins: {wins[0]}")
print(f"Agent 2 Wins: {wins[1]}")
print(f"Draws: {draws}")
print(f"Agent 1 Average Reward: {avg_rewards[0]}")
print(f"Agent 2 Average Reward: {avg_rewards[1]}")

# =============================================================================
# Step 11: Visualization of Results
# =============================================================================
# Plot cumulative wins and draws over time
plt.figure(figsize=(12, 6))
plt.plot(results["Agent 1 Wins"], label="Agent 1 Wins (LittleDeepBlue_0_10_4)")
plt.plot(results["Agent 2 Wins"], label="Agent 2 Wins (initial_agent)")
plt.plot(results["Draws"], label="Draws", linestyle="--")
plt.xlabel("Game Number")
plt.ylabel("Cumulative Count")
plt.title("Tournament Results: Wins and Draws Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot cumulative rewards over time
plt.figure(figsize=(12, 6))
plt.plot(results["Agent 1 Rewards"], label="Agent 1 Rewards (LittleDeepBlue_0_10_4)")
plt.plot(results["Agent 2 Rewards"], label="Agent 2 Rewards (initial_agent)")
plt.xlabel("Game Number")
plt.ylabel("Cumulative Reward")
plt.title("Tournament Results: Cumulative Rewards Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix for Agent Performance
# Convert results into a confusion matrix
y_true = ["Agent 1" if wins[0] > wins[1] else "Agent 2" if wins[1] > wins[0] else "Draw" for _ in range(num_games)]
y_pred = ["Agent 1" if rewards[0] > rewards[1] else "Agent 2" if rewards[1] > rewards[0] else "Draw" for _ in range(num_games)]

cm = confusion_matrix(y_true, y_pred, labels=["Agent 1", "Agent 2", "Draw"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Agent 1", "Agent 2", "Draw"], yticklabels=["Agent 1", "Agent 2", "Draw"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix: Agent Performance")
plt.show()

# =============================================================================
# Step 12: Create Submission File by Copying an Agent File
# =============================================================================
# Here we copy LittleDeepBlue_0_5_5.py to submission.py as an example.
shutil.copyfile("LittleDeepBlue_0_5_5.py", "submission.py")
print("\nSubmission file created: submission.py")

# =============================================================================
# Final Instructions for Submission:
# 1. Download (or save) this main.py file.
# 2. Go to the Kaggle submissions page:
#    https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/submissions
# 3. Click "Submit Agent" and upload submission.py.
# 4. Press Submit!
#
# =============================================================================
# References:
# - https://www.chessprogramming.org/Main_Page
# - https://www.freecodecamp.org/news/simple-chess-ai-step-by-step-1d55a9266977/
# - https://github.com/thomasahle/sunfish
# - https://github.com/apostolisv/chess-ai
# - Additional video and blog references as needed.
# =============================================================================




