from kaggle_environments import evaluate

# Run the minimax agent against a random agent
minimax_vs_random_results = evaluate(
    "connectx",
    [minimax_agent, "random"],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("Results against random opponent (Minimax vs Random):")
print(minimax_vs_random_results)

# Run the random agent against the minimax agent (to see how minimax performs as player 2)
random_vs_minimax_results = evaluate(
    "connectx",
    ["random", minimax_agent],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("\nResults against minimax opponent (Random vs Minimax):")
print(random_vs_minimax_results)


!pip install kaggle-environments


from kaggle_environments import make

env = make('connectx')


print("Observation space:", env.specification.observation)
print("Action space:", env.specification.action)


from kaggle_environments import evaluate

# Run the agent against itself
self_play_results = evaluate(
    "connectx", # Environment name
    [my_agent, my_agent], # Agents to play against each other (pass the function object)
    num_episodes=10 # Number of episodes to run
)

print("Results against itself:")
print(self_play_results)

# Run the agent against a random agent
random_opponent_results = evaluate(
    "connectx", # Environment name
    [my_agent, "random"], # Agent against random opponent (pass the function object for my_agent)
    num_episodes=10 # Number of episodes to run
)

print("\nResults against random opponent:")
print(random_opponent_results)


def my_agent(observation, configuration):
    """
    A random agent that picks a valid column to drop a checker.
    """
    board = observation.board
    columns = configuration.columns
    for col in range(columns):
        if board[col] == 0:
            return col
    # This part should ideally not be reached in a standard game before it ends
    return 0 # Return a default value or raise an error if no valid moves are found


print(env.specification)


print(env.specification)


print("Observation space:", env.specification.observation)
print("Action space:", env.specification.action)


def my_agent(observation, configuration):
    """
    A random agent that picks a valid column to drop a checker.
    """
    board = observation.board
    columns = configuration.columns
    for col in range(columns):
        if board[col] == 0:
            return col
    # This part should ideally not be reached in a standard game before it ends
    return 0 # Return a default value or raise an error if no valid moves are found



import matplotlib.pyplot as plt

# Results from minimax vs random (minimax is player 1)
# [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
minimax_vs_random_results = [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]

# Results from random vs minimax (random is player 1, minimax is player 2)
# [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
random_vs_minimax_results = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

# Calculate wins, losses, and draws for minimax when it's Player 1
minimax_p1_wins = sum(1 for result in minimax_vs_random_results if result[0] == 1)
minimax_p1_losses = sum(1 for result in minimax_vs_random_results if result[0] == -1)
minimax_p1_draws = sum(1 for result in minimax_vs_random_results if result[0] == 0)

# Calculate wins, losses, and draws for minimax when it's Player 2
minimax_p2_wins = sum(1 for result in random_vs_minimax_results if result[1] == 1)
minimax_p2_losses = sum(1 for result in random_vs_minimax_results if result[1] == -1)
minimax_p2_draws = sum(1 for result in random_vs_minimax_results if result[1] == 0)

labels = ['Minimax (P1) Wins', 'Minimax (P1) Losses', 'Minimax (P1) Draws',
          'Minimax (P2) Wins', 'Minimax (P2) Losses', 'Minimax (P2) Draws']
counts = [minimax_p1_wins, minimax_p1_losses, minimax_p1_draws,
          minimax_p2_wins, minimax_p2_losses, minimax_p2_draws]

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color=['green', 'red', 'gray', 'green', 'red', 'gray'])
plt.ylabel('Number of Episodes')
plt.title('Minimax Agent vs Random Agent Results (5 Episodes Each)')
plt.ylim(0, max(counts) + 1) # Adjust y-axis limit for better visualization
plt.show()


from kaggle_environments import evaluate

# Run the minimax agent against a random agent
minimax_vs_random_results = evaluate(
    "connectx",
    [minimax_agent, "random"],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("Results against random opponent (Minimax vs Random):")
print(minimax_vs_random_results)

# Run the random agent against the minimax agent (to see how minimax performs as player 2)
random_vs_minimax_results = evaluate(
    "connectx",
    ["random", minimax_agent],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("\nResults against minimax opponent (Random vs Minimax):")
print(random_vs_minimax_results)


import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Significantly increased scores for immediate wins and penalties for immediate losses.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Immediate win - Give a very high score
    if window.count(piece) == inarow:
        score += 1000000

    # Potential win (3 in a row with 1 empty) - Increase score
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 1000

    # Potential win (2 in a row with 2 empty) - Slightly increased score
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 100

    # Immediate loss (opponent has a winning window) - Very high penalty
    if window.count(opponent_piece) == inarow:
         score -= 1000000

    # Opponent potential win (3 in a row with 1 empty) - Increased penalty
    elif window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 500


    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration):
            return (None, 100000000000000)
        elif check_win(board, 2, configuration):
            return (None, -10000000000000)
        else:
            return (None, 0)

    if depth == 0:
        return (None, score_position(board, 1 if maximizingPlayer else 2, configuration))

    if maximizingPlayer:
        value = -np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            if new_score > value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)

    else: # Minimizing player
        value = np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            if new_score < value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            beta = min(beta, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)


def minimax_agent(observation, configuration, search_depth=4): # Default depth set to 4
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 4.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)


from kaggle_environments import evaluate

# Run the agent against itself
self_play_results = evaluate(
    "connectx", # Environment name
    [my_agent, my_agent], # Agents to play against each other (pass the function object)
    num_episodes=10 # Number of episodes to run
)

print("Results against itself:")
print(self_play_results)

# Run the agent against a random agent
random_opponent_results = evaluate(
    "connectx", # Environment name
    [my_agent, "random"], # Agent against random opponent (pass the function object for my_agent)
    num_episodes=10 # Number of episodes to run
)

print("\nResults against random opponent:")
print(random_opponent_results)


import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """Drops a piece into the specified column."""
    new_board = board.copy()
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    return board # Should not reach here if is_valid_location is checked

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board[r * columns + c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board[(r + i) * columns + c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board[(r + i) * columns + c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board[(r - i) * columns + c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """Evaluates the score of a window of cells."""
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    if window.count(piece) == inarow:
        score += 100
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 2

    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, piece, configuration):
    """Evaluates the score of the entire board for a given piece."""
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """Minimax algorithm for finding the optimal move."""
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration): # Player 1 wins
            return (None, 100000000000000)
        elif check_win(board, 2, configuration): # Player 2 wins
            return (None, -10000000000000)
        else: # Game is a draw
            return (None, 0)

    if depth == 0:
        return (None, score_position(board, 1 if maximizingPlayer else 2, configuration))

    if maximizingPlayer:
        value = -np.inf
        column = np.random.choice(valid_locations) # Initialize with a random valid move
        for col in valid_locations:
            row = next(r for r in range(configuration.rows - 1, -1, -1) if board[r * configuration.columns + col] == 0)
            b_copy = board.copy()
            b_copy[row * configuration.columns + col] = 1 # Assume player 1 is maximizing
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else: # Minimizing player
        value = np.inf
        column = np.random.choice(valid_locations) # Initialize with a random valid move
        for col in valid_locations:
            row = next(r for r in range(configuration.rows - 1, -1, -1) if board[r * configuration.columns + col] == 0)
            b_copy = board.copy()
            b_copy[row * configuration.columns + col] = 2 # Assume player 2 is minimizing
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def minimax_agent(observation, configuration):
    """ConnectX agent that uses the Minimax algorithm to choose a move."""
    board = observation.board
    player = observation.mark
    # Adjusting for the minimax function which assumes player 1 is maximizing
    # If the current player is 2, we call minimax with maximizingPlayer=False
    # and expect the result for player 2. The score_position function also needs to be adjusted.
    if player == 1:
        col, minimax_score = minimax(board, 3, -np.inf, np.inf, True, configuration)
    else: # Player 2
         # To use the existing minimax function which maximizes for player 1,
         # we need to think about the scores from player 2's perspective.
         # A high score for player 1 is a low score for player 2.
         # When player 2 is deciding, it wants to minimize player 1's score,
         # which is equivalent to maximizing its own (negative) score.
         # However, the minimax function is written to maximize for player 1.
         # A simpler approach for player 2 is to run minimax as the maximizing player
         # on a modified board where player 2's pieces are 1 and player 1's are 2,
         # and then reverse the resulting score's sign.
         # Alternatively, we can pass the current player's mark to minimax and
         # adjust the scoring and win conditions within minimax.
         # Let's modify the minimax and score_position functions to take the current player.

        # For now, let's assume the minimax function is always finding the best move for player 1.
        # This will make player 2 act as if it's player 1 trying to win. This is incorrect.

        # Let's refine the minimax call for player 2. Player 2 wants to minimize the outcome
        # from player 1's perspective. So, when it's player 2's turn (minimizing player in the original minimax),
        # we call minimax with maximizingPlayer=False. The scores returned are from player 1's perspective.
        # A lower score is better for player 2.
        col, minimax_score = minimax(board, 3, -np.inf, np.inf, False, configuration)


    # Ensure the chosen column is valid
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (e.g., depth 0 and no terminal state)
        # or the returned column is somehow invalid, choose the first valid column.
        return valid_locations[0]

    return int(col)



from kaggle_environments import evaluate

# Run the minimax agent against a random agent
minimax_vs_random_results = evaluate(
    "connectx",
    [minimax_agent, "random"],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("Results against random opponent (Minimax vs Random):")
print(minimax_vs_random_results)

# Run the random agent against the minimax agent (to see how minimax performs as player 2)
random_vs_minimax_results = evaluate(
    "connectx",
    ["random", minimax_agent],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("\nResults against minimax opponent (Random vs Minimax):")
print(random_vs_minimax_results)


import random

def get_episode_outcome(steps):
    """Determines the outcome of a single episode from the steps."""
    last_step = steps[-1]
    player1_reward = last_step[0]['reward']
    player2_reward = last_step[1]['reward'] # Get player 2's reward to handle draws correctly

    if player1_reward == 1:
        return 'Player 1 Wins'
    elif player2_reward == 1: # Check player 2's reward for a win
        return 'Player 2 Wins'
    else:
        return 'Draw' # If neither player won, it's a draw

def print_training_summary(agent1_name, agent2_name, outcomes, num_episodes):
    """Prints a summary of the training results."""
    player1_wins = outcomes.count('Player 1 Wins')
    player2_wins = outcomes.count('Player 2 Wins')
    draws = outcomes.count('Draw')

    print("\n--- Training Results Summary ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Player 1 ({agent1_name}) Wins: {player1_wins} ({player1_wins/num_episodes:.2%})")
    print(f"Player 2 ({agent2_name}) Wins: {player2_wins} ({player2_wins/num_episodes:.2%})")
    print(f"Draws: {draws} ({draws/num_episodes:.2%})")
    print("--------------------------------")


def train_agent(agent1, agent2, env, num_episodes):
    """
    Trains two agents by simulating games between them.

    Args:
        agent1: The first agent function.
        agent2: The second agent function.
        env: The game environment.
        num_episodes: The number of episodes to play.

    Returns:
        list: A list of outcomes for each episode ('Player 1 Wins', 'Player 2 Wins', 'Draw').
    """
    outcomes = [] # List to store the outcome of each episode

    # Loop through the specified number of episodes for training
    for episode in range(num_episodes):
        # Reset the environment and run the game
        # The env.run method handles the game loop and steps internally
        # It returns a list of observations, rewards, and other info for each step
        steps = env.run([agent1, agent2])

        # Determine and record the outcome of the episode
        outcome = get_episode_outcome(steps)
        outcomes.append(outcome)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Finished episode {episode + 1}/{num_episodes}")

    # Print a summary of the training results after all episodes are complete
    print_training_summary(agent1.__name__, agent2.__name__, outcomes, num_episodes)

    return outcomes # Return the list of outcomes


# Call the training function with the minimax agent playing against itself
print("Starting self-play training with Minimax agent...")
minimax_self_play_outcomes = train_agent(minimax_agent, minimax_agent, env, num_episodes=100) # Increased episodes


# Iterate through the test cases and run each one
for case in test_cases:
    run_test_case(case, minimax_agent)


class MockObservation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class MockConfiguration:
    def __init__(self, columns, rows, inarow):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

def run_test_case(test_case, agent_function):
    """
    Runs a single test case for the minimax agent and checks if the output matches the expected move.

    Args:
        test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
        agent_function (function): The agent function to test (e.g., minimax_agent).
    """
    name = test_case["name"]
    board = test_case["board"]
    mark = test_case["mark"]
    expected_move = test_case["expected_move"]

    # Create mock observation and configuration objects
    mock_observation = MockObservation(board, mark)
    mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)


    # Call the agent function
    actual_move = agent_function(mock_observation, mock_configuration)

    # Compare actual and expected moves
    if actual_move == expected_move:
        print(f"Test case '{name}' PASSED.")
    else:
        print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")


# Define test cases for the minimax agent
test_cases = [
    {
        "name": "Empty board",
        "board": [0] * 42, # 6 rows * 7 columns
        "mark": 1,
        "expected_move": 3 # Center column is often strategically good
    },
    {
        "name": "Winning move (Player 1)",
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 1,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 1,
        "expected_move": 3 # Player 1 must block in column 3
    },
     {
        "name": "Potential future win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                  0, 0, 0, 0, 0, 0, 0],
        "mark": 1,
        "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
    },
     {
        "name": "Winning move (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 2,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 2,
        "expected_move": 3 # Player 2 must block in column 3
    },
]


import matplotlib.pyplot as plt

# Run the training and get the outcomes
# Assuming the last training run you want to visualize is minimax_agent vs minimax_agent for 50 episodes
training_outcomes = train_agent(minimax_agent, minimax_agent, env, num_episodes=100) # Changed to 100 episodes

# Calculate wins, losses, and draws from the outcomes
player1_wins = training_outcomes.count('Player 1 Wins')
player2_wins = training_outcomes.count('Player 2 Wins')
draws = training_outcomes.count('Draw')

labels = ['Player 1 Wins', 'Player 2 Wins', 'Draws']
counts = [player1_wins, player2_wins, draws]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['green', 'blue', 'gray'])
plt.ylabel('Number of Episodes')
plt.title('Training Results (Minimax vs Minimax - 100 Episodes)') # Updated title
plt.ylim(0, max(counts) + 1) # Adjust y-axis limit for better visualization
plt.show()


def minimax_agent(observation, configuration):
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (dict): A dictionary containing the game observation, including:
            - 'board' (list): A 1D list representing the game board (0: empty, 1: Player 1, 2: Player 2).
            - 'mark' (int): The current player's mark (1 or 2).
        configuration (dict): A dictionary containing the game configuration, including:
            - 'columns' (int): The number of columns on the board.
            - 'rows' (int): The number of rows on the board.
            - 'inarow' (int): The number of checkers in a row required to win.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move.
    # The maximizingPlayer argument is set based on whether the current player is 1 (maximizing) or 2 (minimizing from player 1's perspective).
    # The search depth is set to 3.
    col, minimax_score = minimax(board, 3, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        return valid_locations[0]

    return int(col)


def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration['inarow']

    # High score for a winning window
    if window.count(piece) == inarow:
        score += 100
    # Good score for a potential winning line with one empty spot
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 5
    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 2

    # Penalize the opponent having a potential winning line with one empty spot
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 4

    return score


%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """Drops a piece into the specified column."""
    new_board = board.copy()
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    return board # Should not reach here if is_valid_location is checked

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board[r * columns + c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board[(r + i) * columns + c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board[(r + i) * columns + c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board[(r - i) * columns + c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """Evaluates the score of a window of cells."""
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    if window.count(piece) == inarow:
        score += 100
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 2

    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, piece, configuration):
    """Evaluates the score of the entire board for a given piece."""
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """Minimax algorithm for finding the optimal move."""
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration): # Player 1 wins
            return (None, 100000000000000)
        elif check_win(board, 2, configuration): # Player 2 wins
            return (None, -10000000000000)
        else: # Game is a draw
            return (None, 0)

    if depth == 0:
        return (None, score_position(board, 1 if maximizingPlayer else 2, configuration))

    if maximizingPlayer:
        value = -np.inf
        column = np.random.choice(valid_locations) # Initialize with a random valid move
        for col in valid_locations:
            row = next(r for r in range(configuration.rows - 1, -1, -1) if board[r * configuration.columns + col] == 0)
            b_copy = board.copy()
            b_copy[row * configuration.columns + col] = 1 # Assume player 1 is maximizing
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else: # Minimizing player
        value = np.inf
        column = np.random.choice(valid_locations) # Initialize with a random valid move
        for col in valid_locations:
            row = next(r for r in range(configuration.rows - 1, -1, -1) if board[r * configuration.columns + col] == 0)
            b_copy = board.copy()
            b_copy[row * configuration.columns + col] = 2 # Assume player 2 is minimizing
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def minimax_agent(observation, configuration):
    """ConnectX agent that uses the Minimax algorithm to choose a move."""
    board = observation.board
    player = observation.mark

    # The minimax function is designed to maximize for player 1.
    # When player 2 is playing, we want to minimize player 1's score,
    # which is what the 'minimizingPlayer' branch in minimax does.
    # So, we just need to pass the correct player to the minimax function
    # and whether they are the maximizing player in the current context.
    # If the current player is 1, they are the maximizing player.
    # If the current player is 2, they are the minimizing player from player 1's perspective,
    # but they are maximizing their own score, which is the negative of player 1's score.
    # The current minimax implementation assumes player 1 is always maximizing their score
    # and player 2 is always minimizing player 1's score. This aligns with the game's zero-sum nature.
    # So, we just need to call minimax with maximizingPlayer=True if the current player is 1,
    # and maximizingPlayer=False if the current player is 2.

    col, minimax_score = minimax(board, 3, -np.inf, np.inf, player == 1, configuration)


    # Ensure the chosen column is valid
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (e.g., depth 0 and no terminal state)
        # or the returned column is somehow invalid, choose the first valid column.
        return valid_locations[0]

    return int(col)


import random
from itertools import combinations
import matplotlib.pyplot as plt

def get_episode_outcome(steps):
    """
    Determines the outcome of a single episode from the list of steps.

    Args:
        steps (list): A list of observations, rewards, and other info for each step of the episode,
                      as returned by env.run().

    Returns:
        str: The outcome of the episode ('Player 1 Wins', 'Player 2 Wins', or 'Draw').
    """
    last_step = steps[-1]
    player1_reward = last_step[0]['reward']
    player2_reward = last_step[1]['reward'] # Get player 2's reward to handle draws correctly

    if player1_reward == 1:
        return 'Player 1 Wins'
    elif player2_reward == 1: # Check player 2's reward for a win
        return 'Player 2 Wins'
    else:
        return 'Draw' # If neither player won, it's a draw

def print_training_summary(agent1_name, agent2_name, outcomes, num_episodes):
    """Prints a summary of the training results for a single match-up."""
    player1_wins = outcomes.count('Player 1 Wins')
    player2_wins = outcomes.count('Player 2 Wins')
    draws = outcomes.count('Draw')

    print(f"\n--- Training Results Summary: {agent1_name} vs {agent2_name} ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"{agent1_name} Wins (as Player 1): {player1_wins} ({player1_wins/num_episodes:.2%})")
    print(f"{agent2_name} Wins (as Player 2): {player2_wins} ({player2_wins/num_episodes:.2%})")
    print(f"Draws: {draws} ({draws/num_episodes:.2%})")
    print("--------------------------------------------------")

def visualize_matchup_results(agent1_name, agent2_name, outcomes, num_episodes):
    """Visualizes the results of a single agent matchup."""
    player1_wins = outcomes.count('Player 1 Wins')
    player2_wins = outcomes.count('Player 2 Wins')
    draws = outcomes.count('Draw')

    labels = [f'{agent1_name} Wins (P1)', f'{agent2_name} Wins (P2)', 'Draws']
    counts = [player1_wins, player2_wins, draws]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color=['green', 'blue', 'gray'])
    plt.ylabel('Number of Episodes')
    plt.title(f'Matchup Results: {agent1_name} vs {agent2_name} ({num_episodes} Episodes)')
    plt.ylim(0, max(counts) + 1) # Adjust y-axis limit for better visualization
    plt.show()


def train_agents_round_robin(agents, env, num_episodes_per_matchup):
    """
    Trains multiple agents by simulating games in a round-robin fashion and visualizes results.

    Args:
        agents: A list of agent functions or names.
        env: The game environment.
        num_episodes_per_matchup: The number of episodes each pair of agents plays.
    """
    print("Starting round-robin training...")

    # Create pairs of agents for round-robin
    agent_pairs = list(combinations(agents, 2))

    for agent1, agent2 in agent_pairs:
        agent1_name = getattr(agent1, '__name__', agent1) # Get name if it's a function, otherwise use string
        agent2_name = getattr(agent2, '__name__', agent2)

        print(f"\nTraining: {agent1_name} vs {agent2_name}")
        outcomes = []

        for episode in range(num_episodes_per_matchup):
            # Reset the environment and run the game
            steps = env.run([agent1, agent2])

            # Determine and record the outcome
            outcome = get_episode_outcome(steps)
            outcomes.append(outcome)

            # Print progress within a matchup
            if (episode + 1) % (num_episodes_per_matchup // 5) == 0 or (episode + 1) == num_episodes_per_matchup:
                 print(f"  Finished episode {episode + 1}/{num_episodes_per_matchup} for this matchup")


        # Print summary for the current matchup
        print_training_summary(agent1_name, agent2_name, outcomes, num_episodes_per_matchup)

        # Visualize results for the current matchup
        visualize_matchup_results(agent1_name, agent2_name, outcomes, num_episodes_per_matchup)


    print("\nRound-robin training complete.")


# Example usage with multiple agents:
# Assuming you have minimax_agent and my_agent defined, and want to include the built-in "random" agent
# You can add more agent functions to this list.
agent_list = [minimax_agent, my_agent, "random"] # Add your agent functions/names here

# Call the round-robin training function
train_agents_round_robin(agent_list, env, num_episodes_per_matchup=10) # Adjust episodes per matchup as needed


import random

def train_agent(agent1, agent2, env, num_episodes):
    """
    Trains two agents by simulating games between them.

    Args:
        agent1: The first agent function.
        agent2: The second agent function.
        env: The game environment.
        num_episodes: The number of episodes to play.
    """
    outcomes = []

    for episode in range(num_episodes):
        # Reset the environment
        env.reset() # Reset returns a list of observations for each agent
        # Initial observation after reset is in env.state[0]['observation']
        observation = env.state[0]['observation']
        done = False
        step = 0

        # The env.run method handles the game loop and steps internally
        # We can pass the agents directly to env.run
        steps = env.run([agent1, agent2])

        # The outcome is in the last step of the game
        last_step = steps[-1]
        # The reward indicates the outcome: 1 for player 1 win, -1 for player 2 win, 0 for draw/loss
        reward = last_step[0]['reward'] # Reward for player 1

        # Record the outcome
        if reward == 1:
            outcomes.append('Player 1 Wins')
        elif reward == -1:
            outcomes.append('Player 2 Wins')
        else:
             # Check player 2's reward for their outcome
            player2_reward = last_step[1]['reward']
            if player2_reward == 1:
                 outcomes.append('Player 2 Wins')
            elif player2_reward == -1:
                 outcomes.append('Player 1 Wins')
            else:
                 outcomes.append('Draw') # Both rewards are 0 for a draw


        if (episode + 1) % 10 == 0:
            print(f"Finished episode {episode + 1}/{num_episodes}")


    # Print a summary of the training results
    player1_wins = outcomes.count('Player 1 Wins')
    player2_wins = outcomes.count('Player 2 Wins')
    draws = outcomes.count('Draw')

    print("\n--- Training Results Summary ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Player 1 ({agent1.__name__}) Wins: {player1_wins} ({player1_wins/num_episodes:.2%})")
    print(f"Player 2 ({agent2.__name__}) Wins: {player2_wins} ({player2_wins/num_episodes:.2%})")
    print(f"Draws: {draws} ({draws/num_episodes:.2%})")
    print("--------------------------------")

# Call the training function with the minimax agent playing against itself
print("Starting self-play training with Minimax agent...")
train_agent(minimax_agent, minimax_agent, env, num_episodes=50)


%%writefile submission.py

import numpy as np
import time
import random

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Significantly increased scores for immediate wins and penalties for immediate losses.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Immediate win - Give a very high score
    if window.count(piece) == inarow:
        score += 1000000

    # Potential win (3 in a row with 1 empty) - Increase score
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 1000

    # Potential win (2 in a row with 2 empty) - Slightly increased score
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 100

    # Immediate loss (opponent has a winning window) - Very high penalty
    if window.count(opponent_piece) == inarow:
         score -= 1000000

    # Opponent potential win (3 in a row with 1 empty) - Increased penalty
    elif window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 500


    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration):
            return (None, 100000000000000)
        elif check_win(board, 2, configuration):
            return (None, -10000000000000)
        else:
            return (None, 0)

    if depth == 0:
        return (None, score_position(board, 1 if maximizingPlayer else 2, configuration))

    if maximizingPlayer:
        value = -np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            if new_score > value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)

    else: # Minimizing player
        value = np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            if new_score < value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            beta = min(beta, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)


def iterative_deepening_agent(observation, configuration, max_time=1.8):
    """
    ConnectX agent that uses Iterative Deepening with Minimax and Alpha-Beta Pruning.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        max_time (float): The maximum allowed time (in seconds) for the agent to make a move.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]

    if not valid_locations:
        return 0

    best_move = random.choice(valid_locations)
    start_time = time.time()
    depth = 1

    while time.time() - start_time < max_time:
        try:
            current_player_is_maximizing = (player == 1)
            col, score = minimax(board, depth, -np.inf, np.inf, current_player_is_maximizing, configuration)

            if col is not None and col in valid_locations:
                best_move = col

            depth += 1

        except Exception as e:
            # print(f"Error during minimax search at depth {depth}: {e}") # Comment out for Kaggle submission
            break

    current_valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if best_move is None or best_move not in current_valid_locations:
        if current_valid_locations:
             return random.choice(current_valid_locations)
        else:
            return 0

    return int(best_move)


%pycat submission.py


# Sample window (e.g., a horizontal section of the board)
# Let's assume inarow is 4
sample_window_1 = [0, 1, 1, 1] # Player 1 has 3 pieces and 1 empty spot
sample_window_2 = [0, 2, 2, 0] # Player 2 has 2 pieces and 2 empty spots
sample_window_3 = [1, 1, 1, 1] # Player 1 wins
sample_window_4 = [2, 2, 2, 2] # Player 2 wins
sample_window_5 = [0, 0, 0, 0] # Empty window
sample_window_6 = [1, 2, 1, 2] # Mixed pieces

# Sample configuration (assuming default ConnectX board)
sample_configuration = {
    'rows': 6,
    'columns': 7,
    'inarow': 4
}


# Evaluate the sample windows for Player 1 (piece = 1)
score1_p1 = evaluate_window(sample_window_1, 1, sample_configuration)
score2_p1 = evaluate_window(sample_window_2, 1, sample_configuration)
score3_p1 = evaluate_window(sample_window_3, 1, sample_configuration)
score4_p1 = evaluate_window(sample_window_4, 1, sample_configuration)
score5_p1 = evaluate_window(sample_window_5, 1, sample_configuration)
score6_p1 = evaluate_window(sample_window_6, 1, sample_configuration)

print(f"Evaluating sample_window_1 ({sample_window_1}) for Player 1: {score1_p1}")
print(f"Evaluating sample_window_2 ({sample_window_2}) for Player 1: {score2_p1}") # Corrected variable name
print(f"Evaluating sample_window_3 ({sample_window_3}) for Player 1: {score3_p1}")
print(f"Evaluating sample_window_4 ({sample_window_4}) for Player 1: {score4_p1}")
print(f"Evaluating sample_window_5 ({sample_window_5}) for Player 1: {score5_p1}")
print(f"Evaluating sample_window_6 ({sample_window_6}) for Player 1: {score6_p1}")

print("-" * 30)

# Evaluate the sample windows for Player 2 (piece = 2)
score1_p2 = evaluate_window(sample_window_1, 2, sample_configuration)
score2_p2 = evaluate_window(sample_window_2, 2, sample_configuration)
score3_p2 = evaluate_window(sample_window_3, 2, sample_configuration)
score4_p2 = evaluate_window(sample_window_4, 2, sample_configuration)
score5_p2 = evaluate_window(sample_window_5, 2, sample_configuration)
score6_p2 = evaluate_window(sample_window_6, 2, sample_configuration)

print(f"Evaluating sample_window_1 ({sample_window_1}) for Player 2: {score1_p2}")
print(f"Evaluating sample_window_2 ({sample_window_2}) for Player 2: {score2_p2}")
print(f"Evaluating sample_window_3 ({sample_window_3}) for Player 2: {score3_p2}")
print(f"Evaluating sample_window_4 ({sample_window_4}) for Player 2: {score4_p2}")
print(f"Evaluating sample_window_5 ({sample_window_5}) for Player 2: {score5_p2}")
print(f"Evaluating sample_window_6 ({sample_window_6}) for Player 2: {score6_p2}")


from kaggle_environments import evaluate

# Run the minimax agent against a random agent
minimax_vs_random_results = evaluate(
    "connectx",
    [minimax_agent, "random"],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("Results against random opponent (Minimax vs Random):")
print(minimax_vs_random_results)

# Run the random agent against the minimax agent (to see how minimax performs as player 2)
random_vs_minimax_results = evaluate(
    "connectx",
    ["random", minimax_agent],
    num_episodes=5 # Running a smaller number of episodes for quicker testing
)

print("\nResults against minimax opponent (Random vs Minimax):")
print(random_vs_minimax_results)


def minimax_agent(observation, configuration):
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (dict): A dictionary containing the game observation, including:
            - 'board' (list): A 1D list representing the game board (0: empty, 1: Player 1, 2: Player 2).
            - 'mark' (int): The current player's mark (1 or 2).
        configuration (dict): A dictionary containing the game configuration, including:
            - 'columns' (int): The number of columns on the board.
            - 'rows' (int): The number of rows on the board.
            - 'inarow' (int): The number of checkers in a row required to win.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move.
    # The maximizingPlayer argument is set based on whether the current player is 1 (maximizing) or 2 (minimizing from player 1's perspective).
    # The search depth is set to 3.
    col, minimax_score = minimax(board, 3, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        return valid_locations[0]

    return int(col)


# Define test cases for the minimax agent
test_cases = [
    {
        "name": "Empty board",
        "board": [0] * 42, # 6 rows * 7 columns
        "mark": 1,
        "expected_move": 3 # Center column is often strategically good
    },
    {
        "name": "Winning move (Player 1)",
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 1,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 1,
        "expected_move": 3 # Player 1 must block in column 3
    },
     {
        "name": "Potential future win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                  0, 0, 0, 0, 0, 0, 0],
        "mark": 1,
        "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
    },
     {
        "name": "Winning move (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 2,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 2,
        "expected_move": 3 # Player 2 must block in column 3
    },
]


def run_test_case(test_case, agent_function):
    """
    Runs a single test case for the minimax agent and checks if the output matches the expected move.

    Args:
        test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
        agent_function (function): The agent function to test (e.g., minimax_agent).
    """
    name = test_case["name"]
    board = test_case["board"]
    mark = test_case["mark"]
    expected_move = test_case["expected_move"]

    # Create mock observation and configuration dictionaries
    # Using a fixed configuration for testing standard ConnectX
    mock_observation = {
        'board': board,
        'mark': mark
    }
    mock_configuration = {
        'columns': 7,
        'rows': 6,
        'inarow': 4
    }

    # Call the agent function
    actual_move = agent_function(mock_observation, mock_configuration)

    # Compare actual and expected moves
    if actual_move == expected_move:
        print(f"Test case '{name}' PASSED.")
    else:
        print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")



# Iterate through the test cases and run each one
for case in test_cases:
    run_test_case(case, minimax_agent)


for case in test_cases:
    run_test_case(case, minimax_agent)


class MockObservation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class MockConfiguration:
    def __init__(self, columns, rows, inarow):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

def run_test_case(test_case, agent_function):
    """
    Runs a single test case for the minimax agent and checks if the output matches the expected move.

    Args:
        test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
        agent_function (function): The agent function to test (e.g., minimax_agent).
    """
    name = test_case["name"]
    board = test_case["board"]
    mark = test_case["mark"]
    expected_move = test_case["expected_move"]

    # Create mock observation and configuration objects
    mock_observation = MockObservation(board, mark)
    mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)


    # Call the agent function
    actual_move = agent_function(mock_observation, mock_configuration)

    # Compare actual and expected moves
    if actual_move == expected_move:
        print(f"Test case '{name}' PASSED.")
    else:
        print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")

# Iterate through the test cases and run each one
for case in test_cases:
    run_test_case(case, minimax_agent)


def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


import numpy as np

class MockObservation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class MockConfiguration:
    def __init__(self, columns, rows, inarow):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # High score for a winning window
    if window.count(piece) == inarow:
        score += 100
    # Good score for a potential winning line with one empty spot
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 5
    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 2

    # Penalize the opponent having a potential winning line with one empty spot
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

def minimax_agent(observation, configuration, search_depth=3):
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)

def run_test_case(test_case, agent_function):
    """
    Runs a single test case for the minimax agent and checks if the output matches the expected move.

    Args:
        test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
        agent_function (function): The agent function to test (e.g., minimax_agent).
    """
    name = test_case["name"]
    board = test_case["board"]
    mark = test_case["mark"]
    expected_move = test_case["expected_move"]

    # Create mock observation and configuration objects
    mock_observation = MockObservation(board, mark)
    mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)


    # Call the agent function
    actual_move = agent_function(mock_observation, mock_configuration)

    # Compare actual and expected moves
    if actual_move == expected_move:
        print(f"Test case '{name}' PASSED.")
    else:
        print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")

# Iterate through the test cases and run each one
for case in test_cases:
    run_test_case(case, minimax_agent)


# Review the output from the test case execution.
# The output is already available in the previous cell.
# Analyze the failed test cases:
# Test case 'Empty board' FAILED: Expected 3, Got 0
# Test case 'Potential future win (Player 1)' FAILED: Expected 1, Got 0

# Hypothesis for 'Empty board' failure:
# Expected move 3 is the center column, which is generally a good strategic move in ConnectX
# because it maximizes the number of potential winning lines.
# The agent chose column 0. This might be because the initial random choice within minimax
# happened to be 0, and with a shallow search depth, the evaluation function might not
# sufficiently differentiate the strategic value of the center column from other columns
# on an empty board. The score_position function's center column scoring might not be
# weighted highly enough or the depth is too shallow to see the long-term benefits.

# Hypothesis for 'Potential future win (Player 1)' failure:
# The board state shows Player 1 has three pieces in a horizontal row, with empty spaces
# on either side (columns 1 and 5). Dropping a piece in either column 1 or 5 would create a winning line.
# The expected move is 1. The agent chose 0.
# This suggests the evaluation function or the search depth is failing to recognize the immediate
# winning opportunity in columns 1 or 5.
# - The evaluate_window function should give a high score for a window with 3 pieces and 1 empty spot.
# - The score_position function sums these window scores.
# - The minimax algorithm should prioritize moves leading to high scores (wins).
# Possible reasons for failure:
# 1. The score assigned by evaluate_window for a potential winning line (3 pieces, 1 empty) is not high enough
#    relative to other board features, causing the minimax to overlook it at the given depth.
# 2. The search depth (currently 3) might be too shallow to guarantee finding this winning move if it requires
#    the opponent to not block it in the next step (which they wouldn't, as they are random).
# 3. There might be an issue in how the score_position sums up the scores from different windows,
#    potentially diluting the value of the winning opportunity.
# 4. The minimax algorithm's logic for selecting the best column based on the returned score might have an issue,
#    although the basic structure seems correct.

print("Analysis of failed test cases completed. Hypotheses documented in comments.")



%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Increased scores for potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 1000 # Increased winning score significantly

    # Good score for a potential winning line with one empty spot
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 50 # Increased score for potential win

    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 20 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Increased penalty for opponent's potential win to prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 40 # Increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 5 # Increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


def minimax_agent(observation, configuration, search_depth=3): # Default depth set to 3
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 3.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)


# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent is loaded.

# Re-run the test cases with the updated minimax_agent
for case in test_cases:
    run_test_case(case, minimax_agent)


%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Increased scores for potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 1000 # Increased winning score significantly

    # Good score for a potential winning line with one empty spot
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 50 # Increased score for potential win

    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 20 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Increased penalty for opponent's potential win to prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 40 # Increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 5 # Increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


def minimax_agent(observation, configuration, search_depth=4): # Increased depth to 4
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 4.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)


# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (increased depth) is loaded.

# Re-run the test cases with the updated minimax_agent
for case in test_cases:
    run_test_case(case, minimax_agent)


%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Increased scores for potential wins and increased penalty for opponent's potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 10000 # Increased winning score significantly

    # Good score for a potential winning line with one empty spot
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 100 # Further increased score for potential win

    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 30 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Further increased penalty for opponent's potential win to strongly prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 200 # Further increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 10 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


def minimax_agent(observation, configuration, search_depth=4): # Depth kept at 4
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 4.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)



# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (updated scoring and depth) is loaded.

# Re-run the test cases with the updated minimax_agent
for case in test_cases:
    run_test_case(case, minimax_agent)


# Create a mock empty board observation and configuration
empty_board = [0] * (6 * 7) # 6 rows, 7 columns
mock_observation_empty = MockObservation(empty_board, 1) # Player 1's turn
mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)

print("Evaluating moves on an empty board for Player 1:")

# Iterate through each column and evaluate the resulting board state after dropping a piece
# We will use a slightly increased depth for this evaluation to see the scores better
eval_depth = 5 # Use a deeper evaluation just for this analysis

for col in range(mock_configuration.columns):
    # Check if the column is valid (always true on an empty board)
    if is_valid_location(empty_board, col, mock_configuration):
        # Simulate dropping a piece in this column
        temp_board = drop_piece(empty_board, col, 1, mock_configuration) # Player 1 drops a piece

        # Evaluate the resulting board state from Player 1's perspective
        # We call minimax with depth 0 to just get the heuristic score of the resulting board
        # The maximizingPlayer is True here because we are evaluating from Player 1's perspective
        score = minimax(temp_board, 0, -np.inf, np.inf, True, mock_configuration)[1]

        print(f"  Column {col}: Score = {score}")



%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Increased scores for potential wins and increased penalty for opponent's potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 10000 # Increased winning score significantly

    # Good score for a potential winning line with one empty spot
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 100 # Further increased score for potential win

    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 30 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Further increased penalty for opponent's potential win to strongly prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 200 # Further increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 10 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


def minimax_agent(observation, configuration, search_depth=5): # Increased depth to 5
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 5.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)



%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Increased scores for potential wins and increased penalty for opponent's potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 10000 # Increased winning score significantly

    # Good score for a potential winning line with one empty spot
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 100 # Further increased score for potential win

    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 30 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Further increased penalty for opponent's potential win to strongly prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 200 # Further increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 10 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


def minimax_agent(observation, configuration, search_depth=6): # Increased depth to 6
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 6.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)

# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (increased depth) is loaded.

class MockObservation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class MockConfiguration:
    def __init__(self, columns, rows, inarow):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

def run_test_case(test_case, agent_function):
    """
    Runs a single test case for the minimax agent and checks if the output matches the expected move.

    Args:
        test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
        agent_function (function): The agent function to test (e.g., minimax_agent).
    """
    name = test_case["name"]
    board = test_case["board"]
    mark = test_case["mark"]
    expected_move = test_case["expected_move"]

    # Create mock observation and configuration objects
    mock_observation = MockObservation(board, mark)
    mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)


    # Call the agent function
    actual_move = agent_function(mock_observation, mock_configuration)

    # Compare actual and expected moves
    if actual_move == expected_move:
        print(f"Test case '{name}' PASSED.")
    else:
        print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")

# Define test cases for the minimax agent
test_cases = [
    {
        "name": "Empty board",
        "board": [0] * 42, # 6 rows * 7 columns
        "mark": 1,
        "expected_move": 3 # Center column is often strategically good
    },
    {
        "name": "Winning move (Player 1)",
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 1,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 1,
        "expected_move": 3 # Player 1 must block in column 3
    },
     {
        "name": "Potential future win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                  0, 0, 0, 0, 0, 0, 0],
        "mark": 1,
        "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
    },
     {
        "name": "Winning move (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 2,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 2,
        "expected_move": 3 # Player 2 must block in column 3
    },
]

# Iterate through the test cases and run each one
for case in test_cases:
    run_test_case(case, minimax_agent)


# Re-run the test cases with the updated minimax_agent
for case in test_cases:
    run_test_case(case, minimax_agent)


%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Further increased scores for potential wins and increased penalty for opponent's potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 100000 # Significantly increased winning score

    # Good score for a potential winning line with one empty spot (e.g., 3 in a row)
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 500 # Further increased score for potential win

    # Decent score for a potential winning line with two empty spots (e.g., 2 in a row with two open ends)
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 50 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Further increased penalty for opponent's potential win to strongly prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 1000 # Significantly increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) if valid_locations else None # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


def minimax_agent(observation, configuration, search_depth=6): # Depth kept at 6
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 6.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)



# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (updated scoring) is loaded.

# Re-run the test cases with the updated minimax_agent
for case in test_cases:
    run_test_case(case, minimax_agent)


# Create a mock empty board observation and configuration
empty_board = [0] * (6 * 7) # 6 rows, 7 columns
mock_observation_empty = MockObservation(empty_board, 1) # Player 1's turn
mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)

print("Evaluating scores for first moves on an empty board for Player 1:")

# Iterate through each column and evaluate the resulting board state after dropping a piece
# We will use the score_position function directly to see the heuristic score without minimax search
for col in range(mock_configuration.columns):
    # Check if the column is valid (always true on an empty board)
    if is_valid_location(empty_board, col, mock_configuration):
        # Simulate dropping a piece in this column
        temp_board = drop_piece(empty_board, col, 1, mock_configuration) # Player 1 drops a piece

        # Evaluate the resulting board state from Player 1's perspective
        score = score_position(temp_board, 1, mock_configuration)

        print(f"  Column {col}: Score = {score}")


%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + inarow - 1 - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Further increased scores for potential wins and increased penalty for opponent's potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 100000 # Significantly increased winning score

    # Good score for a potential winning line with one empty spot (e.g., 3 in a row)
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 500 # Further increased score for potential win

    # Decent score for a potential winning line with two empty spots (e.g., 2 in a row with two open ends)
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 50 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Further increased penalty for opponent's potential win to strongly prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 1000 # Significantly increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = None # Initialize with None
        # Prioritize columns closer to the center
        prioritized_columns = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))


        # Iterate through all valid moves (prioritizing center)
        for col in prioritized_columns:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col
            # If the scores are equal, prefer the column closer to the center (already handled by prioritized_columns order)
            elif new_score == value and abs(col - configuration.columns // 2) < abs(column - configuration.columns // 2):
                column = col


            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = None # Initialize with None
         # Prioritize columns closer to the center
        prioritized_columns = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        # Iterate through all valid moves (prioritizing center)
        for col in prioritized_columns:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col
             # If the scores are equal, prefer the column closer to the center (already handled by prioritized_columns order)
            elif new_score == value and abs(col - configuration.columns // 2) < abs(column - configuration.columns // 2):
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value


def minimax_agent(observation, configuration, search_depth=6): # Depth kept at 6
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 6.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)

# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (increased depth) is loaded.

class MockObservation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class MockConfiguration:
    def __init__(self, columns, rows, inarow):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

def run_test_case(test_case, agent_function):
    """
    Runs a single test case for the minimax agent and checks if the output matches the expected move.

    Args:
        test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
        agent_function (function): The agent function to test (e.g., minimax_agent).
    """
    name = test_case["name"]
    board = test_case["board"]
    mark = test_case["mark"]
    expected_move = test_case["expected_move"]

    # Create mock observation and configuration objects
    mock_observation = MockObservation(board, mark)
    mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)


    # Call the agent function
    actual_move = agent_function(mock_observation, mock_configuration)

    # Compare actual and expected moves
    if actual_move == expected_move:
        print(f"Test case '{name}' PASSED.")
    else:
        print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")

# Define test cases for the minimax agent
test_cases = [
    {
        "name": "Empty board",
        "board": [0] * 42, # 6 rows * 7 columns
        "mark": 1,
        "expected_move": 3 # Center column is often strategically good
    },
    {
        "name": "Winning move (Player 1)",
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 1,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 1,
        "expected_move": 3 # Player 1 must block in column 3
    },
     {
        "name": "Potential future win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                  0, 0, 0, 0, 0, 0, 0],
        "mark": 1,
        "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
    },
     {
        "name": "Winning move (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 2,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 2,
        "expected_move": 3 # Player 2 must block in column 3
    },
]


# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (prioritized center) is loaded.

# Re-run the test cases with the updated minimax_agent
for case in test_cases:
    run_test_case(case, minimax_agent)


%%writefile submission.py

import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration.rows - 1, -1, -1):
        if new_board[row * configuration.columns + col] == 0:
            new_board[row * configuration.columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + inarow - 1 - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Further increased scores for potential wins and increased penalty for opponent's potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 100000 # Significantly increased winning score

    # Good score for a potential winning line with one empty spot (e.g., 3 in a row)
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 500 # Further increased score for potential win

    # Decent score for a potential winning line with two empty spots (e.g., 2 in a row with two open ends)
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 50 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Further increased penalty for opponent's potential win to strongly prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 1000 # Significantly increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        best_columns = [] # List to store columns that achieve the maximum value

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and collect columns that achieve this value
            if new_score > value:
                value = new_score
                best_columns = [col] # Start a new list of best columns
            elif new_score == value:
                best_columns.append(col) # Add this column to the list of best columns


            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        # If there are multiple best columns, choose the one closest to the center
        if best_columns:
            # Sort best_columns by their distance from the center and pick the first one
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            # Should not happen if valid_locations is not empty, but return a default if needed
            return (valid_locations[0] if valid_locations else 0, value)


    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        best_columns = [] # List to store columns that achieve the minimum value

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and collect columns that achieve this value
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                best_columns = [col] # Start a new list of best columns
            elif new_score == value:
                best_columns.append(col) # Add this column to the list of best columns

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        # If there are multiple best columns, choose the one closest to the center
        if best_columns:
             # Sort best_columns by their distance from the center and pick the first one
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
             # Should not happen if valid_locations is not empty, but return a default if needed
            return (valid_locations[0] if valid_locations else 0, value)


def minimax_agent(observation, configuration, search_depth=6): # Depth kept at 6
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        search_depth (int): The maximum depth the minimax algorithm will search. Defaults to 6.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        if valid_locations:
            return valid_locations[0]
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(col)

# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (increased depth) is loaded.

class MockObservation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

class MockConfiguration:
    def __init__(self, columns, rows, inarow):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

def run_test_case(test_case, agent_function):
    """
    Runs a single test case for the minimax agent and checks if the output matches the expected move.

    Args:
        test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
        agent_function (function): The agent function to test (e.g., minimax_agent).
    """
    name = test_case["name"]
    board = test_case["board"]
    mark = test_case["mark"]
    expected_move = test_case["expected_move"]

    # Create mock observation and configuration objects
    mock_observation = MockObservation(board, mark)
    mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)


    # Call the agent function
    actual_move = agent_function(mock_observation, mock_configuration)

    # Compare actual and expected moves
    if actual_move == expected_move:
        print(f"Test case '{name}' PASSED.")
    else:
        print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")

# Define test cases for the minimax agent
test_cases = [
    {
        "name": "Empty board",
        "board": [0] * 42, # 6 rows * 7 columns
        "mark": 1,
        "expected_move": 3 # Center column is often strategically good
    },
    {
        "name": "Winning move (Player 1)",
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 1,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 1,
        "expected_move": 3 # Player 1 must block in column 3
    },
     {
        "name": "Potential future win (Player 1)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                  0, 0, 0, 0, 0, 0, 0],
        "mark": 1,
        "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
    },
     {
        "name": "Winning move (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0,
                  0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
        "mark": 2,
        "expected_move": 3
    },
    {
        "name": "Blocking opponent win (Player 2)",
        "board": [0] * 42,
        "board": [0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
        "mark": 2,
        "expected_move": 3 # Player 2 must block in column 3
    },
]


# Assuming the test_cases list and run_test_case function are already defined
# and the submission.py with the modified minimax_agent (updated tie-breaking) is loaded.

# Re-run the test cases with the updated minimax_agent
for case in test_cases:
    run_test_case(case, minimax_agent)


import numpy as np
import time

# Keep the existing helper functions (is_valid_location, drop_piece, check_win, evaluate_window, score_position, is_terminal_node, minimax)
# from the previous successful implementation.
# You can assume these functions are available in the environment or should be included.

# Define the iterative deepening agent
def iterative_deepening_agent(observation, configuration, max_time=1.8):
    """
    ConnectX agent that uses Iterative Deepening with Minimax and Alpha-Beta Pruning.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        max_time (float): The maximum allowed time (in seconds) for the agent to make a move.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]

    # If there are no valid moves, return a default (shouldn't happen in a standard game)
    if not valid_locations:
        return 0

    best_move = valid_locations[0]  # Initialize with the first valid move
    start_time = time.time()
    depth = 1

    # Perform iterative deepening
    while time.time() - start_time < max_time:
        try:
            # Perform minimax search with the current depth
            # We need to pass the current player correctly to minimax
            current_player_is_maximizing = (player == 1)
            col, score = minimax(board, depth, -np.inf, np.inf, current_player_is_maximizing, configuration)

            # If minimax returned a valid move, update the best move found so far
            if col is not None and col in valid_locations:
                best_move = col
            else:
                # This case should ideally not happen if minimax is implemented correctly
                # and finds a valid move when one exists.
                pass # Keep the last valid best_move found

            # Increase the search depth for the next iteration
            depth += 1

            # Optional: Add a small delay or check remaining time more frequently
            # for finer-grained time control if needed.

        except Exception as e:
            # Handle potential errors during the search at deeper levels
            print(f"Error during minimax search at depth {depth}: {e}")
            # If an error occurs, break the loop and use the best move found so far
            break

        # Stop deepening if time is running out or the next depth is likely to exceed the time limit
        # This is a simple heuristic; more sophisticated time management could be implemented.
        # For now, we rely on the outer time check.

    # Ensure the returned move is still valid in the current board state
    # This is a safety check, though the last valid move found by minimax should be valid
    if best_move not in [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]:
        # If the best move found is somehow no longer valid (e.g., board changed unexpectedly,
        # though not in standard kaggle environments), fall back to the first valid move.
         return [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)][0]


    return int(best_move)

# Define the minimax and helper functions here or ensure they are imported/available
# from the previous successful step where the minimax agent was defined.

# --- Assuming the following functions are available from previous steps ---
# import numpy as np
# def is_valid_location(board, col, configuration): ...
# def drop_piece(board, col, piece, configuration): ...
# def check_win(board, piece, configuration): ...
# def evaluate_window(window, piece, configuration): ...
# def score_position(board, piece, configuration): ...
# def is_terminal_node(board, configuration): ...
# def minimax(board, depth, alpha, beta, maximizingPlayer, configuration): ...
# -------------------------------------------------------------------------

# You would typically replace the original minimax_agent definition with iterative_deepening_agent
# for testing and submission.

# Example: Testing the iterative_deepening_agent against itself or a random agent
# from kaggle_environments import evaluate
# env = make('connectx') # Assuming env is already created
# print("Starting evaluation with Iterative Deepening agent...")
# id_vs_random_results = evaluate(
#     "connectx",
#     [iterative_deepening_agent, "random"],
#     num_episodes=5
# )
# print("Results against random opponent (Iterative Deepening vs Random):")
# print(id_vs_random_results)


%%writefile submission.py

import numpy as np
import time
import random # Import random for potential fallback moves

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    if col < 0 or col >= configuration.columns:
        return False
    # In a 1D board representation, the top row of a column 'col' is at index 'col'.
    # However, in the ConnectX environment's 1D representation, the board is flattened row by row.
    # The top row is at indices 0 to columns-1.
    # So, to check if a column is valid, we just need to see if the cell at the top of that column is 0.
    # The top cell of column 'col' is at index 'col'.
    return board[col] == 0


def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    rows = configuration.rows
    columns = configuration.columns

    # Find the lowest empty row in the selected column
    # The board is 1D, flattened row by row.
    # The cell at row r, column c is at index r * columns + c.
    for row in range(rows - 1, -1, -1):
        if new_board[row * columns + col] == 0:
            new_board[row * columns + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Further increased scores for potential wins and increased penalty for opponent's potential wins.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Highest score for a winning window
    if window.count(piece) == inarow:
        score += 100000 # Significantly increased winning score

    # Good score for a potential winning line with one empty spot (e.g., 3 in a row)
    # Increased the score for a "Connect 3" with an open end
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 500 # Further increased score for potential win

    # Decent score for a potential winning line with two empty spots (e.g., 2 in a row with two open ends)
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 50 # Increased score slightly

    # Penalize the opponent having a potential winning line with one empty spot
    # Further increased penalty for opponent's potential win to strongly prioritize blocking
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 1000 # Significantly increased penalty

    # Consider adding scores for other configurations if needed

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    # Increased scoring for pieces in the center column
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score


def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)


def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        best_columns = [] # List to store columns that achieve the maximum value

        # Iterate through all valid moves
        # Prioritize columns closer to the center for exploration order, but minimax still explores all valid branches
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and collect columns that achieve this value
            if new_score > value:
                value = new_score
                best_columns = [col] # Start a new list of best columns
            elif new_score == value:
                best_columns.append(col) # Add this column to the list of best columns


            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        # If there are multiple best columns, choose the one closest to the center
        if best_columns:
            # Sort best_columns by their distance from the center and pick the first one
            # This adds a strategic bias towards the center among equally good moves.
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            # Should not happen if valid_locations is not empty, but return a random valid move as a fallback
            return (random.choice(valid_locations) if valid_locations else 0, value)


    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        best_columns = [] # List to store columns that achieve the minimum value

        # Iterate through all valid moves (prioritizing center for exploration order)
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and collect columns that achieve this value
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                best_columns = [col] # Start a new list of best columns
            elif new_score == value:
                best_columns.append(col) # Add this column to the list of best columns

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        # If there are multiple best columns, choose the one closest to the center
        if best_columns:
             # Sort best_columns by their distance from the center and pick the first one
             # This adds a strategic bias towards the center among equally good moves.
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
             # Should not happen if valid_locations is not empty, but return a random valid move as a fallback
            return (random.choice(valid_locations) if valid_locations else 0, value)


def iterative_deepening_agent(observation, configuration, max_time=1.8):
    """
    ConnectX agent that uses Iterative Deepening with Minimax and Alpha-Beta Pruning.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        max_time (float): The maximum allowed time (in seconds) for the agent to make a move.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]

    # If there are no valid moves, return a default (shouldn't happen in a standard game)
    if not valid_locations:
        return 0

    # Initialize best_move with a random valid move as a fallback
    best_move = random.choice(valid_locations)
    start_time = time.time()
    depth = 1

    # Perform iterative deepening
    while time.time() - start_time < max_time:
        try:
            # Calculate remaining time and adjust search depth or time for this iteration
            remaining_time = max_time - (time.time() - start_time)
            # A simple approach: Allocate a portion of remaining time, or just continue deepening
            # until the total time runs out. For simplicity, we'll just keep deepening
            # as long as there's time left. More complex strategies exist.

            # Perform minimax search with the current depth
            current_player_is_maximizing = (player == 1)
            # Note: Minimax is called with the current board state.
            col, score = minimax(board, depth, -np.inf, np.inf, current_player_is_maximizing, configuration)


            # If minimax returned a valid move, update the best move found so far
            if col is not None and col in valid_locations:
                best_move = col

            # Increase the search depth for the next iteration
            depth += 1

             # Optional: Check for time more frequently inside the minimax loop or here
             # to cut off searches that are taking too long.

        except Exception as e:
            # Handle potential errors during the search at deeper levels
            #print(f"Error during minimax search at depth {depth}: {e}") # Comment out for Kaggle submission
            # If an error occurs (e.g., resource limits), break the loop and use the best move found so far
            break

        # Stop deepening if we reached a very high depth (game has limited moves)
        # Or if the last search iteration took a significant amount of time,
        # anticipating the next one might exceed the limit.
        # A simple check: if the time for the last depth was too long. This requires tracking time per depth.
        # For simplicity, we just rely on the total time limit.


    # Ensure the returned move is still valid in the current board state
    # This is a safety check.
    current_valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if best_move is None or best_move not in current_valid_locations:
        # If the best move found is somehow no longer valid (e.g., board changed unexpectedly,
        # though not in standard kaggle environments), or was None, fall back to a random valid move.
        if current_valid_locations:
             return random.choice(current_valid_locations)
        else:
            # If no valid locations, this is an unexpected state, but return 0 as a last resort
            return 0


    return int(best_move)



from kaggle_environments import evaluate, make

# Assuming env is already created from a previous step, if not, create it:
try:
    env
except NameError:
    env = make('connectx')

# Run the iterative_deepening_agent against a random agent
print("Starting evaluation with Iterative Deepening agent...")
id_vs_random_results = evaluate(
    "connectx",
    [iterative_deepening_agent, "random"],
    num_episodes=10 # Running 10 episodes for a quick test
)

print("\nResults against random opponent (Iterative Deepening vs Random):")
print(id_vs_random_results)

# Run the random agent against the iterative_deepening_agent (to see how ID performs as player 2)
print("\nStarting evaluation (Random vs Iterative Deepening)...")
random_vs_id_results = evaluate(
    "connectx",
    ["random", iterative_deepening_agent],
    num_episodes=10 # Running 10 episodes for a quick test
)

print("\nResults against Iterative Deepening opponent (Random vs Iterative Deepening):")
print(random_vs_id_results)


# Assuming the test_cases list and run_test_case function are already defined
# from previous steps and are available in the environment.
# The iterative_deepening_agent is now defined in submission.py and should be loaded.

# Define MockObservation and MockConfiguration classes if not already defined
try:
    MockObservation
except NameError:
    class MockObservation:
        def __init__(self, board, mark):
            self.board = board
            self.mark = mark

try:
    MockConfiguration
except NameError:
    class MockConfiguration:
        def __init__(self, columns, rows, inarow):
            self.columns = columns
            self.rows = rows
            self.inarow = inarow

# Define run_test_case function if not already defined
try:
    run_test_case
except NameError:
    def run_test_case(test_case, agent_function):
        """
        Runs a single test case for the agent and checks if the output matches the expected move.

        Args:
            test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
            agent_function (function): The agent function to test (e.g., iterative_deepening_agent).
        """
        name = test_case["name"]
        board = test_case["board"]
        mark = test_case["mark"]
        expected_move = test_case["expected_move"]

        # Create mock observation and configuration objects
        mock_observation = MockObservation(board, mark)
        mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)

        # Call the agent function
        # Note: We might need to pass max_time to iterative_deepening_agent in tests,
        # but for simple test cases, the default time should be sufficient.
        actual_move = agent_function(mock_observation, mock_configuration)

        # Compare actual and expected moves
        if actual_move == expected_move:
            print(f"Test case '{name}' PASSED.")
        else:
            print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")


# Define test_cases if not already defined
try:
    test_cases
except NameError:
    # Define test cases for the agent
    test_cases = [
        {
            "name": "Empty board",
            "board": [0] * 42, # 6 rows * 7 columns
            "mark": 1,
            "expected_move": 3 # Center column is often strategically good
        },
        {
            "name": "Winning move (Player 1)",
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
            "mark": 1,
            "expected_move": 3
        },
        {
            "name": "Blocking opponent win (Player 1)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
            "mark": 1,
            "expected_move": 3 # Player 1 must block in column 3
        },
         {
            "name": "Potential future win (Player 1)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                      0, 0, 0, 0, 0, 0, 0],
            "mark": 1,
            "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
        },
         {
            "name": "Winning move (Player 2)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
            "mark": 2,
            "expected_move": 3
        },
        {
            "name": "Blocking opponent win (Player 2)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
            "mark": 2,
            "expected_move": 3 # Player 2 must block in column 3
        },
    ]


# Re-run the test cases with the iterative_deepening_agent
print("\nRunning test cases with Iterative Deepening agent:")
for case in test_cases:
    run_test_case(case, iterative_deepening_agent)


%%writefile submission.py

import numpy as np
import time
import random

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    if col < 0 or col >= configuration.columns:
        return False
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    rows = configuration.rows
    columns = configuration.columns
    for row in range(rows - 1, -1, -1):
        if new_board[row * columns + col] == 0:
            new_board[row * columns + col] = piece
            return new_board
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Significantly increased scores for immediate wins and penalties for immediate losses.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Immediate win
    if window.count(piece) == inarow:
        score += 1000000 # Very high score for a winning window

    # Potential win (3 in a row with 1 empty)
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 1000 # Increased score for potential win

    # Potential win (2 in a row with 2 empty)
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 100 # Increased score slightly

    # Immediate loss (opponent has a winning window)
    if window.count(opponent_piece) == inarow:
         score -= 1000000 # Very high penalty for opponent winning

    # Opponent potential win (3 in a row with 1 empty)
    elif window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 500 # Increased penalty for opponent's potential win

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration):
            return (None, 100000000000000)
        elif check_win(board, 2, configuration):
            return (None, -10000000000000)
        else:
            return (None, 0)

    if depth == 0:
        return (None, score_position(board, 1 if maximizingPlayer else 2, configuration))

    if maximizingPlayer:
        value = -np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            if new_score > value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)

    else: # Minimizing player
        value = np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            if new_score < value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            beta = min(beta, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)


def iterative_deepening_agent(observation, configuration, max_time=1.8):
    """
    ConnectX agent that uses Iterative Deepening with Minimax and Alpha-Beta Pruning.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        max_time (float): The maximum allowed time (in seconds) for the agent to make a move.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]

    if not valid_locations:
        return 0

    best_move = random.choice(valid_locations)
    start_time = time.time()
    depth = 1

    while time.time() - start_time < max_time:
        try:
            current_player_is_maximizing = (player == 1)
            col, score = minimax(board, depth, -np.inf, np.inf, current_player_is_maximizing, configuration)

            if col is not None and col in valid_locations:
                best_move = col

            depth += 1

        except Exception as e:
            # print(f"Error during minimax search at depth {depth}: {e}") # Comment out for Kaggle submission
            break

    current_valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if best_move is None or best_move not in current_valid_locations:
        if current_valid_locations:
             return random.choice(current_valid_locations)
        else:
            return 0

    return int(best_move)



# Assuming the test_cases list and run_test_case function are already defined
# from previous steps and are available in the environment.
# The iterative_deepening_agent with the refined evaluation is now defined in submission.py and should be loaded.

# Define MockObservation and MockConfiguration classes if not already defined
try:
    MockObservation
except NameError:
    class MockObservation:
        def __init__(self, board, mark):
            self.board = board
            self.mark = mark

try:
    MockConfiguration
except NameError:
    class MockConfiguration:
        def __init__(self, columns, rows, inarow):
            self.columns = columns
            self.rows = rows
            self.inarow = inarow

# Define run_test_case function if not already defined
try:
    run_test_case
except NameError:
    def run_test_case(test_case, agent_function):
        """
        Runs a single test case for the agent and checks if the output matches the expected move.

        Args:
            test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
            agent_function (function): The agent function to test (e.g., iterative_deepening_agent).
        """
        name = test_case["name"]
        board = test_case["board"]
        mark = test_case["mark"]
        expected_move = test_case["expected_move"]

        # Create mock observation and configuration objects
        mock_observation = MockObservation(board, mark)
        mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)

        # Call the agent function
        # Note: We might need to pass max_time to iterative_deepening_agent in tests,
        # but for simple test cases, the default time should be sufficient.
        actual_move = agent_function(mock_observation, mock_configuration)

        # Compare actual and expected moves
        if actual_move == expected_move:
            print(f"Test case '{name}' PASSED.")
        else:
            print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")


# Define test_cases if not already defined
try:
    test_cases
except NameError:
    # Define test cases for the agent
    test_cases = [
        {
            "name": "Empty board",
            "board": [0] * 42, # 6 rows * 7 columns
            "mark": 1,
            "expected_move": 3 # Center column is often strategically good
        },
        {
            "name": "Winning move (Player 1)",
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
            "mark": 1,
            "expected_move": 3
        },
        {
            "name": "Blocking opponent win (Player 1)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
            "mark": 1,
            "expected_move": 3 # Player 1 must block in column 3
        },
         {
            "name": "Potential future win (Player 1)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                      0, 0, 0, 0, 0, 0, 0],
            "mark": 1,
            "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
        },
         {
            "name": "Winning move (Player 2)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
            "mark": 2,
            "expected_move": 3
        },
        {
            "name": "Blocking opponent win (Player 2)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
            "mark": 2,
            "expected_move": 3 # Player 2 must block in column 3
        },
    ]


# Re-run the test cases with the iterative_deepening_agent
print("\nRunning test cases with Iterative Deepening agent (Refined Evaluation):")
for case in test_cases:
    run_test_case(case, iterative_deepening_agent)


# Assuming the test_cases list and run_test_case function are already defined
# from previous steps and are available in the environment.
# The iterative_deepening_agent with the refined evaluation is defined in submission.py.

# Modify the minimax function temporarily for debugging purposes
# Add print statements to trace scores and chosen columns

# Note: This is for debugging within the notebook. For actual submission,
# these print statements should be removed from submission.py.

# We need to redefine the minimax function here with print statements
# to use it in the test cases.

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    if col < 0 or col >= configuration.columns:
        return False
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    rows = configuration.rows
    columns = configuration.columns
    for row in range(rows - 1, -1, -1):
        if new_board[row * columns + col] == 0:
            new_board[row * columns + col] = piece
            return new_board
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Significantly increased scores for immediate wins and penalties for immediate losses.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Immediate win
    if window.count(piece) == inarow:
        score += 1000000 # Very high score for a winning window

    # Potential win (3 in a row with 1 empty)
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 1000 # Increased score for potential win

    # Potential win (2 in a row with 2 empty)
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 100 # Increased score slightly

    # Immediate loss (opponent has a winning window)
    if window.count(opponent_piece) == inarow:
         score -= 1000000 # Very high penalty for opponent winning

    # Opponent potential win (3 in a row with 1 empty)
    elif window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 500 # Increased penalty for opponent's potential win

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration):
            return (None, 100000000000000)
        elif check_win(board, 2, configuration):
            return (None, -10000000000000)
        else:
            return (None, 0)

    if depth == 0:
        score = score_position(board, 1 if maximizingPlayer else 2, configuration)
        # print(f"  Depth 0 evaluation: Score = {score}") # Debug print
        return (None, score)

    if maximizingPlayer:
        value = -np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        # print(f"Minimax (Max, Depth {depth}): Exploring valid moves {prioritized_locations}") # Debug print

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # print(f"  Minimax (Max, Depth {depth}): Move {col} leads to score {new_score}") # Debug print

            if new_score > value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            alpha = max(alpha, value)
            if alpha >= beta:
                # print(f"  Minimax (Max, Depth {depth}): Alpha-beta pruning at move {col}. Alpha={alpha}, Beta={beta}") # Debug print
                break

        if best_columns:
            # print(f"Minimax (Max, Depth {depth}): Best columns found: {best_columns} with value {value}") # Debug print
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            # print(f"Minimax (Max, Depth {depth}): Choosing column {best_column}") # Debug print
            return best_column, value
        else:
            # print(f"Minimax (Max, Depth {depth}): No best columns found, returning random valid move") # Debug print
            return (random.choice(valid_locations) if valid_locations else 0, value)

    else: # Minimizing player
        value = np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        # print(f"Minimax (Min, Depth {depth}): Exploring valid moves {prioritized_locations}") # Debug print

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # print(f"  Minimax (Min, Depth {depth}): Move {col} leads to score {new_score}") # Debug print


            if new_score < value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            beta = min(beta, value)
            if alpha >= beta:
                # print(f"  Minimax (Min, Depth {depth}): Alpha-beta pruning at move {col}. Alpha={alpha}, Beta={beta}") # Debug print
                break

        if best_columns:
            # print(f"Minimax (Min, Depth {depth}): Best columns found: {best_columns} with value {value}") # Debug print
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            # print(f"Minimax (Min, Depth {depth}): Choosing column {best_column}") # Debug print
            return best_column, value
        else:
             # print(f"Minimax (Min, Depth {depth}): No best columns found, returning random valid move") # Debug print
            return (random.choice(valid_locations) if valid_locations else 0, value)


def iterative_deepening_agent(observation, configuration, max_time=1.8):
    """
    ConnectX agent that uses Iterative Deepening with Minimax and Alpha-Beta Pruning.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        max_time (float): The maximum allowed time (in seconds) for the agent to make a move.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]

    if not valid_locations:
        return 0

    best_move = random.choice(valid_locations)
    start_time = time.time()
    depth = 1

    # print(f"Iterative Deepening Agent: Starting search for player {player}...") # Debug print

    while time.time() - start_time < max_time:
        try:
            # print(f"Iterative Deepening Agent: Searching at depth {depth}") # Debug print
            current_player_is_maximizing = (player == 1)
            col, score = minimax(board, depth, -np.inf, np.inf, current_player_is_maximizing, configuration)

            if col is not None and col in valid_locations:
                best_move = col
                # print(f"Iterative Deepening Agent: Found best move {best_move} with score {score} at depth {depth}") # Debug print


            depth += 1

        except Exception as e:
            # print(f"Error during minimax search at depth {depth}: {e}") # Debug print
            break

    # print(f"Iterative Deepening Agent: Time limit reached or error, returning best move found: {best_move}") # Debug print

    current_valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if best_move is None or best_move not in current_valid_locations:
        if current_valid_locations:
             # print(f"Iterative Deepening Agent: Fallback to random valid move: {current_valid_locations[0]}") # Debug print
             return random.choice(current_valid_locations)
        else:
            # print("Iterative Deepening Agent: No valid locations, returning 0") # Debug print
            return 0

    return int(best_move)

# Assuming the test_cases list and run_test_case function are already defined
# from previous steps and are available in the environment.
# The iterative_deepening_agent with the refined evaluation is now defined here.

# Define MockObservation and MockConfiguration classes if not already defined
try:
    MockObservation
except NameError:
    class MockObservation:
        def __init__(self, board, mark):
            self.board = board
            self.mark = mark

try:
    MockConfiguration
except NameError:
    class MockConfiguration:
        def __init__(self, columns, rows, inarow):
            self.columns = columns
            self.rows = rows
            self.inarow = inarow

# Define run_test_case function if not already defined
try:
    run_test_case
except NameError:
    def run_test_case(test_case, agent_function):
        """
        Runs a single test case for the agent and checks if the output matches the expected move.

        Args:
            test_case (dict): A dictionary containing the test case details ('name', 'board', 'mark', 'expected_move').
            agent_function (function): The agent function to test (e.g., iterative_deepening_agent).
        """
        name = test_case["name"]
        board = test_case["board"]
        mark = test_case["mark"]
        expected_move = test_case["expected_move"]

        print(f"\n--- Running Test Case: '{name}' ---") # Debug print

        # Create mock observation and configuration objects
        mock_observation = MockObservation(board, mark)
        mock_configuration = MockConfiguration(columns=7, rows=6, inarow=4)

        # Call the agent function
        actual_move = agent_function(mock_observation, mock_configuration)

        # Compare actual and expected moves
        if actual_move == expected_move:
            print(f"Test case '{name}' PASSED.")
        else:
            print(f"Test case '{name}' FAILED: Expected {expected_move}, Got {actual_move}")
        print("-" * (len(name) + 20)) # Debug print


# Define test_cases if not already defined
try:
    test_cases
except NameError:
    # Define test cases for the agent
    test_cases = [
        {
            "name": "Empty board",
            "board": [0] * 42, # 6 rows * 7 columns
            "mark": 1,
            "expected_move": 3 # Center column is often strategically good
        },
        {
            "name": "Winning move (Player 1)",
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
            "mark": 1,
            "expected_move": 3
        },
        {
            "name": "Blocking opponent win (Player 1)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
            "mark": 1,
            "expected_move": 3 # Player 1 must block in column 3
        },
         {
            "name": "Potential future win (Player 1)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 1, 1, 0, 0, # Player 1 has 3 in a row, needs col 1 or 5
                      0, 0, 0, 0, 0, 0, 0],
            "mark": 1,
            "expected_move": 1 # Either 1 or 5 could be valid, choose 1 for now
        },
         {
            "name": "Winning move (Player 2)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0], # Player 2 needs one more in column 3
            "mark": 2,
            "expected_move": 3
        },
        {
            "name": "Blocking opponent win (Player 2)",
            "board": [0] * 42,
            "board": [0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0], # Player 1 needs one more in column 3
            "mark": 2,
            "expected_move": 3 # Player 2 must block in column 3
        },
    ]


# Re-run the test cases with the iterative_deepening_agent and debug prints
print("\nRunning test cases with Iterative Deepening agent (Debugging Enabled):")
for case in test_cases:
    run_test_case(case, iterative_deepening_agent)


%%writefile submission.py

import numpy as np
import time
import random

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    if col < 0 or col >= configuration.columns:
        return False
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    rows = configuration.rows
    columns = configuration.columns
    for row in range(rows - 1, -1, -1):
        if new_board[row * columns + col] == 0:
            new_board[row * columns + col] = piece
            return new_board
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    Significantly increased scores for immediate wins and penalties for immediate losses.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration.inarow

    # Immediate win
    if window.count(piece) == inarow:
        score += 1000000 # Very high score for a winning window

    # Potential win (3 in a row with 1 empty)
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 1000 # Increased score for potential win

    # Potential win (2 in a row with 2 empty)
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 100 # Increased score slightly

    # Immediate loss (opponent has a winning window)
    if window.count(opponent_piece) == inarow:
         score -= 1000000 # Very high penalty for opponent winning

    # Opponent potential win (3 in a row with 1 empty)
    elif window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 500 # Increased penalty for opponent's potential win

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    Adjusted center column scoring.
    """
    score = 0
    rows = configuration.rows
    columns = configuration.columns
    inarow = configuration.inarow
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 20 # Further increased center column weight

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (object): Game configuration object with attributes.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    is_terminal = is_terminal_node(board, configuration)

    if is_terminal:
        if check_win(board, 1, configuration):
            return (None, 100000000000000)
        elif check_win(board, 2, configuration):
            return (None, -10000000000000)
        else:
            return (None, 0)

    if depth == 0:
        score = score_position(board, 1 if maximizingPlayer else 2, configuration)
        return (None, score)

    if maximizingPlayer:
        value = -np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 1
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            if new_score > value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)

    else: # Minimizing player
        value = np.inf
        best_columns = []
        prioritized_locations = sorted(valid_locations, key=lambda col: abs(col - configuration.columns // 2))

        for col in prioritized_locations:
            b_copy = board.copy()
            row = next(r for r in range(configuration.rows - 1, -1, -1) if b_copy[r * configuration.columns + col] == 0)
            b_copy[row * configuration.columns + col] = 2
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            if new_score < value:
                value = new_score
                best_columns = [col]
            elif new_score == value:
                best_columns.append(col)

            beta = min(beta, value)
            if alpha >= beta:
                break

        if best_columns:
            best_column = sorted(best_columns, key=lambda col: abs(col - configuration.columns // 2))[0]
            return best_column, value
        else:
            return (random.choice(valid_locations) if valid_locations else 0, value)


def iterative_deepening_agent(observation, configuration, max_time=1.8):
    """
    ConnectX agent that uses Iterative Deepening with Minimax and Alpha-Beta Pruning.

    Args:
        observation (object): The game observation object with attributes 'board' and 'mark'.
        configuration (object): The game configuration object with attributes 'columns', 'rows', and 'inarow'.
        max_time (float): The maximum allowed time (in seconds) for the agent to make a move.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark
    valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]

    if not valid_locations:
        return 0

    best_move = random.choice(valid_locations)
    start_time = time.time()
    depth = 1

    while time.time() - start_time < max_time:
        try:
            current_player_is_maximizing = (player == 1)
            col, score = minimax(board, depth, -np.inf, np.inf, current_player_is_maximizing, configuration)

            if col is not None and col in valid_locations:
                best_move = col

            depth += 1

        except Exception as e:
            # print(f"Error during minimax search at depth {depth}: {e}") # Comment out for Kaggle submission
            break

    current_valid_locations = [col for col in range(configuration.columns) if is_valid_location(board, col, configuration)]
    if best_move is None or best_move not in current_valid_locations:
        if current_valid_locations:
             return random.choice(current_valid_locations)
        else:
            return 0

    return int(best_move)


