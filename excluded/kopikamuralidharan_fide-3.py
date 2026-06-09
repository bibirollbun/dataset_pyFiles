!pip install python-chess


import warnings
warnings.simplefilter("ignore", DeprecationWarning)



!pip install --upgrade setuptools pip


!pip install --upgrade pygame



import sys
sys.path.append('/kaggle/input/chessbot')

from chess_bot import chess_bot



!chmod +x /kaggle/input/chessengine/stockfish.exe
!cp /kaggle/input/chessengine/stockfish.exe /kaggle/working/



import chess
import chess.engine
import time

# Define a global instance of the engine
engine = None

def chess_bot(obs):
    """Main function to interact with Kaggle chess environment."""
    global engine
    fen = obs['board']

    # Set engine path (update if uploading the engine as a dataset)
    engine_path = "/kaggle/input/chessengine/stockfish.exe"  # Update with correct Kaggle dataset path

    # Initialize engine if not already running
    if engine is None:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    # Get the best move with a limited time
    start_time = time.time()
    try:
        result = engine.play(chess.Board(fen), chess.engine.Limit(time=0.5))  # 0.5 sec per move
        best_move = result.move.uci()
    except Exception as e:
        print("Error:", e)
        best_move = "e2e4"  # Default safe move
    end_time = time.time()

    print(f"Move selected: {best_move} | Time taken: {end_time - start_time:.2f}s")
    
    return best_move



!apt-get install -y stockfish



import chess
import chess.engine
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessBot:
    def __init__(self, engine_path):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.logger = logging.getLogger(__name__)

    def get_best_move(self, fen, time_limit=0.5):
        board = chess.Board(fen)
        start_time = time.time()
        try:
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            best_move = result.move.uci()
        except Exception as e:
            self.logger.error(f"Error: {e}")
            best_move = "e2e4"  # Default safe move
        end_time = time.time()
        self.logger.info(f"Move selected: {best_move} | Time taken: {end_time - start_time:.2f}s")
        return best_move

    def close(self):
        self.engine.quit()

# Example usage
if __name__ == "__main__":
    engine_path = "/usr/games/stockfish"
    bot = ChessBot(engine_path)
    
    # Example FEN string
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    best_move = bot.get_best_move(fen)
    print(f"Best move: {best_move}")
    
    bot.close()


import chess
import chess.engine
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessBot:
    def __init__(self, engine_path="/usr/games/stockfish"):  # Use built-in Linux Stockfish
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.logger = logging.getLogger(__name__)

    def get_best_move(self, fen, time_limit=0.5):
        """Returns the best move and all legal moves from a given FEN position."""
        board = chess.Board(fen)
        start_time = time.time()

        try:
            # Get the best move from Stockfish
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            best_move = result.move.uci()

            # Get all legal moves
            legal_moves = [move.uci() for move in board.legal_moves]

        except Exception as e:
            self.logger.error(f"Error: {e}")
            best_move = "e2e4"  # Default safe move
            legal_moves = []

        end_time = time.time()

        self.logger.info(f"Best Move: {best_move} | All Legal Moves: {legal_moves} | Time taken: {end_time - start_time:.2f}s")

        return {
            "best_move": best_move,
            "legal_moves": legal_moves
        }

    def close(self):
        """Closes the Stockfish engine."""
        self.engine.quit()

# Example usage
if __name__ == "__main__":
    bot = ChessBot()  # No need to specify engine_path
    
    # Example FEN string (Initial Position)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Get best move and all legal moves
    result = bot.get_best_move(fen)
    
    print(f"Best Move: {result['best_move']}")
    print(f"All Legal Moves: {result['legal_moves']}")
    
    bot.close()



import chess
import chess.engine
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessBot:
    def __init__(self, engine_path="/usr/games/stockfish"):  # Use built-in Linux Stockfish
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.logger = logging.getLogger(__name__)

    def get_best_move(self, fen, time_limit=0.5):
        """Returns the best move and all legal moves from a given FEN position."""
        board = chess.Board(fen)
        start_time = time.time()

        try:
            # Get the best move from Stockfish
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            best_move = result.move.uci()

            # Get all legal moves
            legal_moves = [move.uci() for move in board.legal_moves]

        except Exception as e:
            self.logger.error(f"Error: {e}")
            best_move = "e2e4"  # Default safe move
            legal_moves = []

        end_time = time.time()

        self.logger.info(f"Best Move: {best_move} | All Legal Moves: {legal_moves} | Time taken: {end_time - start_time:.2f}s")

        return {
            "best_move": best_move,
            "legal_moves": legal_moves
        }

    def close(self):
        """Closes the Stockfish engine."""
        self.engine.quit()

# Example usage
if __name__ == "__main__":
    bot = ChessBot()  # No need to specify engine_path
    
    # Example FEN string (Initial Position)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Get best move and all legal moves
    result = bot.get_best_move(fen)
    
    print(f"Best Move: {result['best_move']}")
    print(f"All Legal Moves: {result['legal_moves']}")
    
    bot.close()



import chess
import chess.engine
import time
import logging
import chess.svg
from IPython.display import display, SVG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessBot:
    def __init__(self, engine_path="/usr/games/stockfish"):  # Use built-in Linux Stockfish
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.logger = logging.getLogger(__name__)

    def get_best_move(self, board, time_limit=0.5):
        """Get the best move and engine evaluation for the current position."""
        start_time = time.time()
        
        try:
            # Get best move
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            best_move = result.move.uci()

            # Get position evaluation
            info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
            score = info.get("score", None)

            # Convert evaluation score to readable format
            if score is not None:
                if score.is_mate():
                    eval_str = f"Mate in {score.mate()}"
                else:
                    eval_str = f"CP: {score.relative.cp}"
            else:
                eval_str = "No evaluation available"

        except Exception as e:
            self.logger.error(f"Error: {e}")
            best_move = "e2e4"  # Default safe move
            eval_str = "Error occurred"

        end_time = time.time()

        self.logger.info(f"Best Move: {best_move} | Evaluation: {eval_str} | Time taken: {end_time - start_time:.2f}s")

        return best_move, eval_str

    def close(self):
        """Closes the Stockfish engine."""
        self.engine.quit()

# Function to display the chessboard
def display_board(board, highlight_move=None):
    print("\nBoard Position:")
    print(board)  # ASCII representation

    if highlight_move:
        move = chess.Move.from_uci(highlight_move)
        board_svg = chess.svg.board(board=board, lastmove=move, size=400)
    else:
        board_svg = chess.svg.board(board=board, size=400)

    display(SVG(board_svg))

# Function to simulate a full game
def play_full_game():
    bot = ChessBot()
    board = chess.Board()

    print("\nğŸ”¹ Starting a New Chess Game ğŸ”¹")

    move_count = 1
    while not board.is_game_over():
        print(f"\nâ™Ÿï¸� Move {move_count}: {'White' if board.turn == chess.WHITE else 'Black'} to move")

        # Get best move and evaluation
        best_move, evaluation = bot.get_best_move(board)
        board.push_uci(best_move)  # Make the move on the board

        # Display updated board with highlighted move
        display_board(board, highlight_move=best_move)

        # Print move details
        print(f"âœ… Best Move Played: {best_move}")
        print(f"ğŸ“Š Position Evaluation: {evaluation}")
        
        move_count += 1
        time.sleep(1)  # Small delay to better visualize moves

    # Game is over, print the result
    print("\nğŸ”´ Game Over ğŸ”´")
    print(f"Result: {board.result()} (1-0 = White wins, 0-1 = Black wins, 1/2-1/2 = Draw)")

    bot.close()

# Run the automatic game simulation
play_full_game()


