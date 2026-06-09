# pip install libraries
!pip install pygame Chessnut
!pip install --upgrade kaggle-environments


from kaggle_environments import make
env = make("chess", debug=True)


%cd /kaggle/input/fruktik/frukt
!gcc -std=c++11 -O2 -o /kaggle/working/frukt *.cpp



%%writefile /kaggle/working/main.py
%%writefile /kaggle/working/main.py

import subprocess
import chess
import chess.pgn
import random
import os

class ChessEngine:
    def __init__(self, engine_path, opening_book_path=None):
        self.log_lines = []
        # Start the engine process.
        self.engine = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._initialize_engine()
        # Load the opening book if a path is provided.
        self.opening_book = self._load_opening_book(opening_book_path) if opening_book_path else {}

    def _initialize_engine(self):
        # Start UCI mode.
        self._send_command("uci")
        while True:
            output = self._read_output()
            if output == "uciok":
                break

        # --- ENHANCED SETTINGS FOR MAXIMUM PERFORMANCE ---
        self._send_command("setoption name Hash value 512")
        self._send_command("setoption name Threads value 8")
        self._send_command("setoption name Skill Level value 20")
        self._send_command("setoption name Contempt value -200")
        self._send_command("setoption name UCI_LimitStrength value false")
        self._send_command("setoption name Ponder value true")
        self._send_command("setoption name MultiPV value 1")

    def _load_opening_book(self, book_path):
        """
        Loads opening moves from a PGN file into a dictionary mapping FEN strings to lists of moves.
        Each move is stored as a chess.Move object.
        """
        if not os.path.exists(book_path):
            print(f"Opening book file not found: {book_path}")
            return {}
        opening_book = {}
        with open(book_path, "r") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    fen_key = board.fen()
                    if fen_key not in opening_book:
                        opening_book[fen_key] = []
                    opening_book[fen_key].append(move)
                    board.push(move)
        print(f"Loaded opening book with {len(opening_book)} unique positions from {book_path}")
        return opening_book

    def _send_command(self, command):
        """Send a command to the engine and log it."""
        self.log_lines.append("Command: " + command)
        self.engine.stdin.write(command + "\n")
        self.engine.stdin.flush()

    def _read_output(self):
        """Read a line of output from the engine and log it."""
        output = self.engine.stdout.readline().strip()
        self.log_lines.append("Output: " + output)
        return output

    def _evaluate_king_safety(self, board):
        """
        A simple evaluation of king safety.
        Counts the number of friendly pieces adjacent to the king and applies a penalty if the king is in check.
        """
        king_square = board.king(board.turn)
        if king_square is None:
            return 0  # Game over or invalid board state

        defenders = 0
        # Check squares immediately around the king.
        for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
            piece = board.piece_at(square)
            if piece is not None and piece.color == board.turn:
                defenders += 1

        in_check_penalty = 3 if board.is_check() else 0
        safety_score = defenders - in_check_penalty
        return safety_score

    def get_best_move(self, fen, movetime=100):
        """
        Get the best move for the given position.
        1. Uses an opening book if the current FEN is found.
        2. Otherwise, adjusts the search time based on king safety and queries the engine.
        """
        board = chess.Board(fen)

        # 1. Opening Book Lookup
        if self.opening_book and fen in self.opening_book:
            possible_moves = self.opening_book[fen]
            if possible_moves:
                chosen_move = random.choice(possible_moves)
                self.log_lines.append(f"Using opening book move: {chosen_move.uci()} for position {fen}")
                return chosen_move.uci()

        # 2. Adjust movetime based on king safety.
        king_safety = self._evaluate_king_safety(board)
        adjusted_movetime = int(movetime * 2) if king_safety < 0 else int(movetime * 1.1)

        self._send_command(f"position fen {fen}")
        self._send_command(f"go movetime {adjusted_movetime}")

        best_move = None
        while True:
            output = self._read_output()
            if output.startswith("bestmove"):
                best_move = output.split()[1]
                break

        return best_move

    def stop(self):
        """Stop the engine process and write logs to a text file."""
        self._send_command("quit")
        self.engine.terminate()
        self.engine.wait()

        log_file_path = "/kaggle/working/engine_logs.txt"
        with open(log_file_path, "w") as log_file:
            for line in self.log_lines:
                log_file.write(line + "\n")
        print(f"Engine logs saved to {log_file_path}")

# Global instance of the ChessEngine.
ultima = None

def chess_bot(obs):
    """
    Given an observation dictionary with a 'board' key containing the FEN,
    this function returns the best move as determined by the engine (or from the opening book).
    """
    global ultima
    fen = obs['board']

    engine_path = '/kaggle_simulations/agent/frukt'
    opening_book_path = '/my-opening-book/my_openings.pgn'  # Replace with your opening book

    if ultima is None:
        ultima = ChessEngine(engine_path, opening_book_path)

    best_move = ultima.get_best_move(fen)
    return best_move



%cd /kaggle/working
!tar -czvf submission.tar.gz main.py frukt


# Move all necessary files
!cp -r /kaggle/input/aiagenticoptimizedcodellm/other/default/1/* /kaggle/working/
import sys 
sys.path.append('/kaggle/working/AIAgenticOptimizedCodeLLM.py')


import os
from kaggle_secrets import UserSecretsClient
from google import genai
from IPython.display import display, HTML
from AIAgenticOptimizedCodeLLM import AIAgenticOptimizedCodeLLM 


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Get the API key using Kaggle's secrets (adjust as needed for your environment)
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")

    # Define file paths for metrics, original code, and the output HTML file.
    metrics_file_path = "/kaggle/input/log-text/logdata_file.txt"
    code_file_path = "/kaggle/working/main.py"
    optimized_html_file_path = "/kaggle/working/main_v1.html"

    # User prompt for improvements
    user_prompt = """
1. any grandmaster openbook movements for this uci is available
2.   here --> # Enable pondering so the engine continues thinking during the opponent's turn.
        self._send_command("setoption name Ponder value true")
        # Force the engine to focus on a single best move.
        self._send_command("setoption name MultiPV value 1")

        any new strategy that protect king from all checkmates as well as parallel agressive move for winning?
"""

    optimizer = AIAgenticOptimizedCodeLLM(api_key, metrics_file_path, code_file_path, optimized_html_file_path)
    html_response = optimizer.optimize_model_code(user_prompt)

    #print("Final HTML Response:")
    #print(html_response)
    #print("Final HTML Response:")
    display(HTML(html_response))

