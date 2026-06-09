%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments


# Now let's set up the chess environment!
from kaggle_environments import make
env = make("chess", debug=True)


%%writefile main.py
from Chessnut import Game
import random

def init_chess_bot(obs):
    """
    Simple chess bot that prioritizes queen promotions, then captures, then randomly moves.

    Args:
        obs: An object with a 'board' attribute representing the current board state as a FEN string.

    Returns:
        A string representing the chosen move in UCI notation (e.g., "e2e4")
    """
    # Initialize the chess game with the current board state
    game = Game(obs.board)

    # Get all possible moves
    moves = game.get_moves()

    # Prioritize queen promotions
    promotion_moves = []
    for move in moves:
        if 'q' in move:
            promotion_moves.append(move)

    if promotion_moves:
        return random.choice(promotion_moves)

    # Prioritize captures
    capture_moves = []
    for move in moves:
        if 'x' in move:
            capture_moves.append(move)

    if capture_moves:
        return random.choice(capture_moves)

    # Randomly select a move
    return random.choice(moves)


# Testing the agent
result = env.run(["main.py", "random"])
for agent in result[-1]:
    print("Status:", agent.status, "/ Reward:", agent.reward, "/ Time left:", agent.observation.remainingOverageTime)
env.render(mode="ipython", width=700, height=700)


%%writefile improving_agent.py
from Chessnut import Game
import random

def init_chess_bot(obs):
    """
    Simple chess bot that prioritizes queen promotions, then captures, then controls the center, then develops pieces, then randomly moves.

    Args:
        obs: An object with a 'board' attribute representing the current board state as a FEN string.

    Returns:
        A string representing the chosen move in UCI notation (e.g., "e2e4")
    """
    # Initialize the chess game with the current board state
    game = Game(obs.board)

    # Get all possible moves
    moves = game.get_moves()

    # Prioritize queen promotions
    promotion_moves = [move for move in moves if 'q' in move]
    if promotion_moves:
        return random.choice(promotion_moves)

    # Prioritize captures
    capture_moves = [move for move in moves if 'x' in move]
    if capture_moves:
        return random.choice(capture_moves)

    # Prioritize controlling the center
    center_moves = [move for move in moves if move[:2] in ['d4', 'd5', 'e4', 'e5'] or move[2:] in ['d4', 'd5', 'e4', 'e5']]
    if center_moves:
        return random.choice(center_moves)

    # Prioritize developing pieces
    piece_development_moves = [move for move in moves if move[:2] in ['b1', 'c1', 'f1', 'g1'] or move[2:] in ['b3', 'c3', 'f3', 'g3']]
    if piece_development_moves:
        return random.choice(piece_development_moves)

    # Randomly select a move
    return random.choice(moves)


result = env.run(["improving_agent.py", "random"])
print("Agent exit status/reward/time left: ")
# look at the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")
# render the game
env.render(mode="ipython",  width=700, height=700)


def order_moves(game):
    """
    Orders legal moves to improve alpha-beta efficiency.
    Prioritizes captures and checks.
    """
    def move_priority(move):
        game.apply_move(move)
        score = 0
        if game.status() == "checkmate":  # Prioritize checkmate
            score += 100000
        elif game.status() == "check":  # Prioritize checks
            score += 1000
        elif move[1] in "QRBN":  # Capturing high-value pieces
            score += 100 * "QRBN".index(move[1])
        game.undo()
        return -score  # Sort in descending order

    return sorted(game.get_moves(), key=move_priority)


def alpha_beta(game, depth, alpha, beta, maximizing_player):
    if depth == 0 or game.status() != "playing":
        return evaluate_board(game)

    legal_moves = order_moves(game)  # Use optimized move ordering
    best_move = None

    if maximizing_player:
        max_eval = -float("inf")
        for move in legal_moves:
            game.apply_move(move)
            eval_score = alpha_beta(game, depth - 1, alpha, beta, False)
            game.undo()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return max_eval if depth > 1 else best_move
    else:
        min_eval = float("inf")
        for move in legal_moves:
            game.apply_move(move)
            eval_score = alpha_beta(game, depth - 1, alpha, beta, True)
            game.undo()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return min_eval


def evaluate_board(game):
    """
    Improved evaluation function with piece-square tables.
    """
    piece_values = {'P': 100, 'N': 300, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}

    piece_square_bonus = {
        'P': [0, 5, 10, 20, 30, 40, 50, 60],  # Encourage advancing pawns
        'N': [30, 20, 10, 5, 5, 10, 20, 30],  # Centralize knights
        'B': [10, 10, 20, 30, 30, 20, 10, 10],  # Bishops favor open diagonals
        'R': [5, 10, 15, 20, 20, 15, 10, 5],  # Rooks favor open files
        'Q': [5, 10, 15, 20, 20, 15, 10, 5],  # Queens similar to rooks
        'K': [-50, -40, -30, -20, -20, -30, -40, -50]  # King safety (favor back rank)
    }

    score = 0
    board = game.board()

    for i, piece in enumerate(board):
        if piece != '.':
            value = piece_values.get(piece.upper(), 0)
            file_index = i % 8
            rank_index = i // 8 if piece.isupper() else 7 - (i // 8)  # Flip for black pieces

            # Apply piece-square table
            bonus = piece_square_bonus.get(piece.upper(), [0]*8)[rank_index]
            total_value = value + bonus if piece.isupper() else -(value + bonus)

            score += total_value

    return score


from kaggle_environments import make

env = make("chess", debug=True)

def chess_agent(observation, configuration):
    """
    Kaggle Chess AI Agent: Uses Alpha-Beta with Move Ordering & Improved Evaluation.
    """
    game = Game()
    game.apply_fen(observation["fen"])
    best_move = alpha_beta(game, depth=3, alpha=-float('inf'), beta=float('inf'), maximizing_player=True)
    return best_move if best_move else random.choice(game.get_moves())

# Run a test match
env.run([chess_agent, "random"])
env.render(mode="ipython", width=600, height=600)

