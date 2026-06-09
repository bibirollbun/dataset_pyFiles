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
import pandas as pd
from Chessnut import Game
import random
from tqdm import tqdm
from datetime import datetime

class StrategicChessAI:
    def __init__(self):
        self.win_streak = 0
        self.mu = 600
        self.log_file = f"chess_log_{int(time.time())}.csv"
        self._init_csv()
        self.cleanup_old_files()

    def _init_csv(self):
        with open(self.log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['player1', 'player2', 'total_moves', 'status', 'timestamp', 'date_time'])

    def log_game(self, moves, result):
        timestamp = int(time.time() * 1000)
        date_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        json_file = f"{result}_chessgame_at_{timestamp}.json"

        # Save the game moves to a JSON file
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'date_time': date_time,
                'moves': moves,
                'result': result
            }, f)

        # Log the result in the CSV file
        with open(self.log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["Player1", "Player2", len(moves), result, timestamp, date_time])

    def cleanup_old_files(self):
        json_files = sorted(glob.glob('*_chessgame_at_*.json'), key=os.path.getmtime)
        while len(json_files) > 100:
            os.remove(json_files.pop(0))

    def calculate_move(self, game):
        moves = list(game.get_moves())

        # 1. Check for checkmates
        for move in moves:
            g = Game(game.get_fen())
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

        # 4. Random move as fallback
        return random.choice(moves)

    def analyze_loss_and_learn(self):
        loss_files = sorted(glob.glob('loss_chessgame_at_*.json'), key=os.path.getmtime)
        if not loss_files:
            return

        # Load the most recent loss file
        with open(loss_files[-1]) as f:
            loss_data = json.load(f)

        print(f"Analyzing loss at {loss_data['timestamp']} with moves: {loss_data['moves']}")
        # Implement a learning strategy based on the loss data

    def play_game(self):
        game = Game()
        moves = []

        while game.status == 0:
            move = self.calculate_move(game)
            game.apply_move(move)
            moves.append(move)

            if len(moves) > 50:  # Force end if no mate in 50
                break

        # Determine result
        result = 'win' if game.status == Game.CHECKMATE else 'loss' if game.status == Game.STALEMATE else 'tie'

        # Update rating and log game
        if result == 'win':
            self.mu += 15 + (self.win_streak * 3)
            self.win_streak += 1
        else:
            self.win_streak = 0

        self.log_game(moves, result)

    def generate_statistics(self):
        # Read the log file and calculate statistics
        df = pd.read_csv(self.log_file)
        total_games = len(df)
        wins = len(df[df['status'] == 'win'])
        losses = len(df[df['status'] == 'loss'])
        ties = len(df[df['status'] == 'tie'])

        win_percentage = (wins / total_games) * 100 if total_games > 0 else 0
        loss_percentage = (losses / total_games) * 100 if total_games > 0 else 0
        tie_percentage = (ties / total_games) * 100 if total_games > 0 else 0

        print("Game Statistics:")
        print(f"Total Games: {total_games}")
        print(f"Wins: {wins} ({win_percentage:.2f}%)")
        print(f"Losses: {losses} ({loss_percentage:.2f}%)")
        print(f"Ties: {ties} ({tie_percentage:.2f}%)")

        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_percentage': win_percentage,
            'loss_percentage': loss_percentage,
            'tie_percentage': tie_percentage
        }

    def convert_json_to_dataframe(self):
        json_files = glob.glob('*_chessgame_at_*.json')
        all_data = []

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_data.append(data)

        df = pd.DataFrame(all_data)
        print("\nAll JSON Data as DataFrame:")
        print(df)
        return df

def chess_bot(obs):
    """
    Simple chess bot that prioritizes checkmates, then captures, queen promotions, then randomly moves.

    Args:
        obs: An object with a 'board' attribute representing the current board state as a FEN string.

    Returns:
        A string representing the chosen move in UCI notation (e.g., "e2e4")
    """
    ai = StrategicChessAI()

    # 0. Parse the current board state and generate legal moves using Chessnut library
    game = Game(obs.board)
    return ai.calculate_move(game)

if __name__ == "__main__":
    ai = StrategicChessAI()

    with tqdm(total=2, desc="Strategic Chess AI Progression") as pbar:
        while ai.mu < 3000:
            ai.play_game()
            ai.analyze_loss_and_learn()
            pbar.update(ai.mu - pbar.n)

            if ai.mu % 500 == 0:
                print(f"\nCurrent Rating: {ai.mu}")

    print(f"\nSupreme Victory Achieved! Final Rating: {ai.mu}")
    stats = ai.generate_statistics()
    print("\nFinal Statistics:")
    print(stats)

    # Convert JSON logs to DataFrame and display
    ai.convert_json_to_dataframe()



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





