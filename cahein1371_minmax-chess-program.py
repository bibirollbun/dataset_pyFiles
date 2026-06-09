# first let's make sure you have internet enabled
import requests
requests.get('http://www.google.com',timeout=60).ok


%%capture
!pip install --upgrade kaggle-environments chessnut


# Import libraries
from kaggle_environments import make


%%writefile main.py
from Chessnut import Game
import random
import time

# Piece values for evaluation
PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 100,
                "p": -1, "n": -3, "b": -3, "r": -5, "q": -9, "k": -100}

# Transposition table for caching results
transposition_table = {}

# Helper functions
def fen_to_board_array(fen):
    """Convert FEN format to a board array"""
    board = []
    for row in fen.split()[0].split('/'):
        for char in row:
            if char.isdigit():
                board.extend([" "] * int(char))
            else:
                board.append(char)
    return board

def quick_evaluate(fen):
    """Simple evaluation function: evaluates based on piece values"""
    board = fen_to_board_array(fen)
    return sum(PIECE_VALUES.get(piece, 0) for piece in board)

def positional_evaluation(board):
    """Positional evaluation: evaluates control of the center and piece positions"""
    central_squares = ['d4', 'e4', 'd5', 'e5']
    position_score = 0
    board_array = fen_to_board_array(board.get_fen())  # Get the board from FEN
    for i, piece in enumerate(board_array):
        if piece != " ":
            row, col = divmod(i, 8)
            square = chr(97 + col) + str(8 - row)  # Convert to coordinates like 'd4', 'e5'
            position_score += 1 if square in central_squares else 0
    return position_score

def evaluate_board(board):
    """Board evaluation function: combines piece value and positional evaluation"""
    return quick_evaluate(board.get_fen()) + positional_evaluation(board)

def order_moves(board, moves):
    """Order moves to improve alpha-beta pruning efficiency"""
    scored_moves = []
    for move in moves:
        new_board = Game(board.get_fen())
        new_board.apply_move(move)
        score = evaluate_board(new_board)
        scored_moves.append((move, score))
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return [move for move, _ in scored_moves]

def filter_moves(board, moves, max_moves=5):
    """Filter the top N moves"""
    scored_moves = []
    for move in moves:
        new_board = Game(board.get_fen())
        new_board.apply_move(move)
        score = evaluate_board(new_board)
        scored_moves.append((move, score))
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return [move for move, _ in scored_moves[:max_moves]]

def determine_dynamic_depth(board):
    """Dynamically adjust the search depth"""
    piece_count = sum(1 for char in board if char in PIECE_VALUES)
    if piece_count > 20:
        return 2  # Early game
    elif piece_count > 10:
        return 3  # Midgame
    else:
        return 4  # Endgame

def minimax_with_transposition(board, depth, alpha, beta, maximizing_player):
    """Minimax with transposition table"""
    # Check transposition table
    if board.get_fen() in transposition_table:
        return transposition_table[board.get_fen()]

    if depth == 0 or not list(board.get_moves()):  # No legal moves
        return evaluate_board(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.get_moves():
            new_board = Game(board.get_fen())
            new_board.apply_move(move)
            eval = minimax_with_transposition(new_board, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[board.get_fen()] = max_eval  # Save the result to the transposition table
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.get_moves():
            new_board = Game(board.get_fen())
            new_board.apply_move(move)
            eval = minimax_with_transposition(new_board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[board.get_fen()] = min_eval  # Save the result to the transposition table
        return min_eval

def iterative_deepening(board, max_depth, time_limit=0.2):
    """Iterative deepening search"""
    best_move = None
    start_time = time.time()

    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            break  # Exit if the time limit is exceeded
        best_move = None
        moves = list(board.get_moves())
        moves = order_moves(board, moves)  # Optimize move ordering

        for move in moves:
            new_board = Game(board.get_fen())
            new_board.apply_move(move)
            eval = minimax_with_transposition(new_board, depth, float('-inf'), float('inf'), True)
            if best_move is None or eval > best_move[1]:
                best_move = (move, eval)

            # Exit early if time limit is near
            if time.time() - start_time > time_limit * 0.5:  # Buffer time
                break

    return best_move[0] if best_move else "0000"

def chess_bot(obs):
    """Chess bot"""
    game = Game(obs.board)
    if not game.get_moves():
        return "0000"  # No legal moves

    # Dynamically determine depth
    depth = determine_dynamic_depth(obs.board)

    # Use iterative deepening to find the best move
    best_move = iterative_deepening(game, depth)
    return best_move



# Initialize the Kaggle chess environment
env = make("chess", debug=True)

# Run the game: Bot vs Random agent
result = env.run(["main.py", "random"])


# Display agent results
print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")

# Render the game visually
env.render(mode="ipython", width=1000, height=1000)




