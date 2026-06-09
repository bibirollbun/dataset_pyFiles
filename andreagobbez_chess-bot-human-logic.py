!pip install python-chess


from kaggle_environments import make
env = make("chess", debug=True)

result = env.run(["random", "random"])
env.render(mode="ipython", width=1000, height=1000) 


%%writefile main.py
import chess
import random

def choose_move(chess_board, bot_color):
    """
    Choose a valid move for the bot following a prioritized strategy.
    :param chess_board: Current chess board (chess.Board object).
    :param bot_color: Color of the bot (True for White, False for Black).
    :return: A valid move in string format.
    """
    legal_moves = list(chess_board.legal_moves)

    # Helper function to evaluate piece importance
    def piece_value(piece):
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 1000}
        return values.get(piece.piece_type, 0)

    # Checkmate opportunity
    for move in legal_moves:
        chess_board.push(move)
        if chess_board.is_checkmate():
            chess_board.pop()
            return str(move)
        chess_board.pop()

    # Capture a more valuable piece with a less valuable one
    for move in legal_moves:
        if chess_board.is_capture(move):
            captured_piece = chess_board.piece_at(move.to_square)
            moved_piece = chess_board.piece_at(move.from_square)
            if captured_piece and piece_value(captured_piece) > piece_value(moved_piece):
                return str(move)

    # Capture a piece of equal value
    for move in legal_moves:
        if chess_board.is_capture(move):
            captured_piece = chess_board.piece_at(move.to_square)
            moved_piece = chess_board.piece_at(move.from_square)
            if captured_piece and piece_value(captured_piece) == piece_value(moved_piece):
                return str(move)

    # Give check if safe or leads to advantageous recapture
    for move in legal_moves:
        chess_board.push(move)
        if chess_board.is_check():
            if not chess_board.is_attacked_by(not bot_color, move.to_square):
                chess_board.pop()
                return str(move)
            else:
                # Check if recapture is possible after giving check
                if any(chess_board.is_capture(recapture_move) for recapture_move in chess_board.legal_moves):
                    chess_board.pop()
                    return str(move)
        chess_board.pop()

    # Safe random move excluding pawns
    safe_moves = []
    for move in legal_moves:
        moved_piece = chess_board.piece_at(move.from_square)
        if moved_piece.piece_type != chess.PAWN:
            if not chess_board.is_attacked_by(not bot_color, move.to_square):
                safe_moves.append(move)

    if safe_moves:
        return str(random.choice(safe_moves))

    # Fallback: Any random move
    if legal_moves:
        return str(random.choice(legal_moves))

    # No moves available (stalemate or checkmate)
    return None

def human_chess_logic(chess_board_state):
    """
    Main logic for the bot's move selection.
    :param chess_board_state: A dictionary with 'board' (FEN string) and 'mark' (bot's color as 'white' or 'black').
    :return: The bot's chosen move in string format.
    """
    try:
        bot_color = chess_board_state['mark']
        bot_color = chess.WHITE if bot_color == "white" else chess.BLACK
        chess_board = chess.Board(chess_board_state['board'])

        if chess_board.is_game_over():
            # Game over conditions
            return None

        move = choose_move(chess_board, bot_color)
        return move
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


result = env.run(["main.py", "random"])
print("Agent exit status/reward/time left: ")
# look at the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")
# render the game
env.render(mode="ipython", width=1000, height=1000) 

