import numpy as np
import time
import itertools

np.random.seed(9)


class RewardMap:
    """
    EV-based approach using a single shared EV grid across all relics.
    
    In this design each time we add a reward region (via add_reward_region or 
    narrow_reward_location) we update the global EV grid and also record the relic’s 
    metadata (including its center, region size, and remaining expected reward).
    
    Later, when an actual reward is discovered, we find the relic (by center) that is 
    nearest to the tile where the reward was found and “deduct” from all tiles in a 
    5x5 area around that relic.
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # A single EV grid that holds the sum of EV contributions from all relics.
        self.prob_grid = np.zeros((rows, cols), dtype=float) + .05
        # A list of relic records. Each relic is represented as a dict with keys:
        # 'center' (tuple), 'region_size' (int), 'estimated' (float) and 'remaining' (float).
        self.relics = {}
    
    def add_reward_region(self, center_x, center_y, region_size, estimated_reward):
        """
        Add a new relic’s reward region. The expected reward (estimate_reward) is 
        uniformly distributed over the region_size x region_size block centered at (center_x, center_y).
        The EV grid is updated, and a relic record is stored.

        Parameters:
            center_x, center_y: int, int - the center of the reward region.
            region_size: int - the side length of the square region.
            estimate_reward: float - the total expected reward in the region.
        """

        if (center_x, center_y) in self.relics:
            return
        
        start = time.time()
        half = region_size // 2
        valid_tiles = []
        for dx in range(-half, half+1):
            for dy in range(-half, half+1):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < self.cols and 0 <= y < self.rows:
                    valid_tiles.append((x, y))
        if valid_tiles:
            increment = estimated_reward / len(valid_tiles)
            for (vx, vy) in valid_tiles:
                self.prob_grid[vx, vy] += increment
        
        # Save the relic’s metadata.
        self.relics[(center_x, center_y)] = {
            'center': (center_x, center_y),
            'region_size': region_size,
            'estimated': estimated_reward,
            'remaining': estimated_reward
        }

    def sum(self):
        return self.prob_grid.sum()
    
    def multi_step_observation(self, visited_tiles, found_reward):
        """
        Update the map of expected apple probabilities based on an observation.
        
        Parameters:
          map_probs: dict mapping (x, y) coordinates to prior P(apple).
          visited_cells: list of (x, y) coordinates visited by the agents.
          observed_apples: total number of apples found among visited cells (0, 1, or 2).
          
        Returns:
          A new dictionary with updated probabilities for the visited cells.
          Cells not in visited_cells remain unchanged.
        """
        map_probs = self.prob_grid
        # Extract the priors for the visited cells.
        visited_priors = {cell: map_probs[cell] for cell in visited_tiles}
        
        total_likelihood = 0.0
        config_likelihoods = {}
        
        # Enumerate all configurations (as subsets of visited_cells) that have exactly found_reward
        if len(visited_tiles) == 1 and found_reward:
            combos = [(visited_tiles)]
        elif len(visited_tiles) == 1 and not found_reward:
            combos = [()]
        else:
            combos = list(itertools.combinations(visited_tiles, found_reward))
        for combo in combos:
            config = set(combo)
            likelihood = 1.0
            for cell in visited_tiles:
                if cell in config:
                    likelihood *= visited_priors[cell]
                else:
                    likelihood *= (1 - visited_priors[cell])
            config_key = frozenset(config)
            config_likelihoods[config_key] = likelihood
            total_likelihood += likelihood
    
        # If no configuration has nonzero probability, then we cannot update
        if total_likelihood == 0:
            return map_probs  # Alternatively, you might want to leave visited cells unchanged.
        
        # For each visited cell, sum the likelihoods over all configurations that include that cell.
        # updated_map = map_probs.copy()
        for cell in visited_tiles:
            cell_likelihood = sum(like for config, like in config_likelihoods.items() if cell in config)
            self.prob_grid[cell] = cell_likelihood / total_likelihood
        return self.prob_grid
    
    def print_total_ev(self):
        """
        Print the shared EV grid (rounded to 2 decimals).
        """
        print(np.round(self.prob_grid, 2))
    
    def __repr__(self):
        # Display both the global EV grid and the relics' information.
        print(self.prob_grid)
        return 'RewardMap'
    
    def heatmap(self, size=None, show_text=True):

        if size is None:
            size = (8, 8)

        import matplotlib.pyplot as plt
        # we need to label each cell with its grid location and use size
        # to set the figure size
        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(self.prob_grid, cmap='viridis', interpolation='nearest', 
        vmin=np.min(self.prob_grid), vmax=np.max(self.prob_grid))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Value")
        if show_text:
            for i in range(self.prob_grid.shape[0]):
                for j in range(self.prob_grid.shape[1]):
                    txt = f'{i}, {j}'
                    ax.text(j, i, txt, ha='center', va='center')
        # we want the colors to be associated the values in the grid
        plt.imshow(self.prob_grid, cmap='viridis')


np.set_printoptions(linewidth=140)


reward_map = RewardMap(rows=12, cols=12)
# Add region
reward_map.add_reward_region(center_x=8, center_y=8, region_size=3, estimated_reward=3.0)
reward_map.add_reward_region(center_x=2, center_y=2, region_size=3, estimated_reward=3.0)
# Plot the heatmap
reward_map.heatmap()
# Perform multi-step observation
visited_tiles = [(8, 9), (9, 9), (7, 7)]
print(visited_tiles)
reward_map.multi_step_observation(visited_tiles, found_reward=2)
reward_map.heatmap()
visited_tiles = [(8, 9), (9, 9), (7, 8)]
print(visited_tiles)
reward_map.multi_step_observation(visited_tiles, found_reward=2)
reward_map.heatmap()
reward_map.multi_step_observation(visited_tiles, found_reward=0)
# uncomment if you want to see the grid
#print(np.round(reward_map.prob_grid, 2))
reward_map.heatmap()


print(np.round(reward_map.prob_grid, 2))
visited_tiles = [(2, 3)]
print(visited_tiles)
reward_map.multi_step_observation(visited_tiles, found_reward=2)
reward_map.heatmap()
visited_tiles = [(3, 3)]
print(visited_tiles)
reward_map.multi_step_observation(visited_tiles, found_reward=0)
reward_map.heatmap()


def weighted_random_choice_2d_row_col(array_2d, weights_2d):
    """
    Selects a single random element from a 2D NumPy array using probabilities as weights,
    and returns the row and column indices of the selected element.

    Args:
        array_2d: A 2D NumPy array.
        weights_2d: A 2D NumPy array of the same shape as array_2d, representing probabilities.

    Returns:
        A tuple (row, col) representing the row and column indices of the selected element.
    """
    if not isinstance(array_2d, np.ndarray) or array_2d.ndim != 2:
        raise ValueError("array_2d must be a 2D NumPy array.")
    if not isinstance(weights_2d, np.ndarray) or weights_2d.ndim != 2:
        raise ValueError("weights_2d must be a 2D NumPy array.")
    if array_2d.shape != weights_2d.shape:
        raise ValueError("array_2d and weights_2d must have the same shape.")
    if not np.all(weights_2d >= 0):
        raise ValueError("weights_2d must contain non-negative values.")

    rows, cols = array_2d.shape
    flat_weights = weights_2d.flatten()
    flat_weights = flat_weights / np.sum(flat_weights)  # Normalize weights

    selected_index = np.random.choice(rows * cols, p=flat_weights)
    return selected_index
    # selected_row = selected_index // cols
    # selected_col = selected_index % cols

    # return (selected_row, selected_col)


import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.special import softmax

def run_true_greedy_sampling_simulation(reward_map, num_iterations=50, true_reward_locations=None, use_softmax=True, temperature=0.1, step_size=2):
    """
    Run a true greedy sampling simulation on the reward map.
    This implementation allows revisiting locations and selects cells based on softmax probabilities.
    
    Parameters:
        reward_map: RewardMap object
        num_iterations: int - number of iterations to run
        true_reward_locations: list of (x, y) tuples - the actual locations of rewards
        use_softmax: bool - whether to use softmax selection instead of pure argmax
        temperature: float - temperature parameter for softmax (lower = more greedy)
        step_size: int - number of cells to explore per iteration
    
    Returns:
        history: list of probability grids over time
        visited_history: list of visited locations
        rewards_found: list of rewards found at each step
    """
    # Create a deep copy of the reward map to avoid modifying the original
    greedy_map = copy.deepcopy(reward_map)
    
    if true_reward_locations is None:
        raise ValueError("true_reward_locations must be provided for greedy simulation")
    
    # Store history of probability grids
    history = [greedy_map.prob_grid.copy()]
    visited_history = []
    rewards_found = []
    
    # Keep track of found rewards (each reward can only be found once)
    found_rewards_set = set()
    
    for i in range(num_iterations):
        # Get current probability grid
        current_probs = greedy_map.prob_grid.copy()
        
        # Apply the P > 0.1 threshold
        probability_mask = greedy_map.prob_grid > 0.1
        # Mask out cells with probability <= 0.1
        current_probs = np.where(probability_mask, current_probs, 0.0)
        
        # Batch of cells to visit this iteration
        batch_visited_tiles = []
        
        for _ in range(step_size):
            if use_softmax:
                # Apply softmax to probabilities to make selection probabilistic
                flat_probs = current_probs.flatten()
                # Check if there are any viable cells left
                if np.sum(flat_probs) > 0:
                    # Apply temperature scaling to control greediness
                    flat_probs_softmax = softmax(flat_probs / temperature)
                    # Sample based on softmax probabilities
                    flat_idx = np.random.choice(len(flat_probs_softmax), p=flat_probs_softmax)
                else:
                    # If no cells with P > 0.1 left, fall back to highest probability cell
                    flat_idx = np.argmax(greedy_map.prob_grid.flatten())
            else:
                # Pure greedy selection
                if np.max(current_probs) > 0:
                    flat_idx = np.argmax(current_probs)
                else:
                    # If no cells with P > 0.1 left, fall back to highest probability cell
                    flat_idx = np.argmax(greedy_map.prob_grid.flatten())
            
            # Convert flat index to 2D coordinates
            y, x = np.unravel_index(flat_idx, current_probs.shape)
            visited_tile = (x, y)
            batch_visited_tiles.append(visited_tile)
            
            # Remove this cell from consideration for this batch
            current_probs[y, x] = 0
        
        raw_rewards_in_batch = sum(1 for tile in batch_visited_tiles if tile in true_reward_locations)
        new_rewards_in_batch = 0
        for tile in batch_visited_tiles:
            if tile in true_reward_locations and tile not in found_rewards_set:
                new_rewards_in_batch += 1
                found_rewards_set.add(tile)
        
        # Update the map with raw observations
        greedy_map.multi_step_observation(batch_visited_tiles, raw_rewards_in_batch)
        
        # Store history
        history.append(greedy_map.prob_grid.copy())
        visited_history.extend(batch_visited_tiles)  # Add all visited tiles to history
        rewards_found.append(new_rewards_in_batch)
    
    print("True Greedy visited tiles:", visited_history)
    print("True Greedy rewards found:", rewards_found)
    print("True Greedy total rewards found:", sum(rewards_found))
    
    return history, visited_history, rewards_found

def run_epsilon_greedy_sampling_simulation(reward_map, num_iterations=50, true_reward_locations=None, epsilon=0.1, step_size=2):
    """
    Run an epsilon-greedy sampling simulation on the reward map.
    With probability epsilon, chooses random locations.
    With probability 1-epsilon, chooses the highest probability locations.
    
    Parameters:
        reward_map: RewardMap object
        num_iterations: int - number of iterations to run
        true_reward_locations: list of (x, y) tuples - the actual locations of rewards
        epsilon: float - probability of random exploration (0-1)
        step_size: int - number of cells to explore per iteration
    
    Returns:
        history: list of probability grids over time
        visited_history: list of visited locations
        rewards_found: list of rewards found at each step
    """
    import copy
    import numpy as np
    import itertools

    # Create a deep copy of the reward map to avoid modifying the original
    epsilon_greedy_map = copy.deepcopy(reward_map)
    
    if true_reward_locations is None:
        raise ValueError("true_reward_locations must be provided for epsilon-greedy simulation")
    
    # Store history of probability grids
    history = [epsilon_greedy_map.prob_grid.copy()]
    visited_history = []
    rewards_found = []
    
    # Keep track of visited locations and found rewards
    visited_set = set()  # For informational purposes only
    found_rewards_set = set()
    
    for i in range(num_iterations):
        # Batch of cells to visit this iteration
        batch_visited_tiles = []
        current_probs = epsilon_greedy_map.prob_grid.copy()
        
        # Loop over step_size: choose step_size cells per iteration
        for _ in range(step_size):
            # Apply the P > 0.1 threshold for the current state
            probability_mask = epsilon_greedy_map.prob_grid > 0.1
            filtered_probs = np.where(probability_mask, current_probs, 0.0)
            
            # With probability epsilon, explore randomly
            if np.random.random() < epsilon:
                # Select a random location from cells with P > 0.1
                high_prob_cells = np.where(probability_mask)
                if len(high_prob_cells[0]) > 0:
                    idx = np.random.choice(len(high_prob_cells[0]))
                    y, x = high_prob_cells[0][idx], high_prob_cells[1][idx]
                else:
                    x = np.random.randint(0, epsilon_greedy_map.cols)
                    y = np.random.randint(0, epsilon_greedy_map.rows)
            else:
                # Exploitation: select the highest probability location
                if np.max(filtered_probs) > 0:
                    flat_idx = np.argmax(filtered_probs)
                    y, x = np.unravel_index(flat_idx, current_probs.shape)
                else:
                    flat_idx = np.argmax(current_probs)
                    y, x = np.unravel_index(flat_idx, current_probs.shape)
            
            visited_tile = (x, y)
            batch_visited_tiles.append(visited_tile)
            
            # Remove this cell from consideration in this batch
            current_probs[y, x] = 0
            
            # Track unique visited locations (for informational purposes)
            visited_set.add(visited_tile)
        
        # Determine number of rewards observed in this batch.
        # raw_rewards_in_batch is the count of cells in the batch that are true reward locations.
        raw_rewards_in_batch = sum(1 for tile in batch_visited_tiles if tile in true_reward_locations)
        new_rewards_in_batch = 0
        for tile in batch_visited_tiles:
            if tile in true_reward_locations and tile not in found_rewards_set:
                new_rewards_in_batch += 1
                found_rewards_set.add(tile)
        
        # Update the map with raw observations
        epsilon_greedy_map.multi_step_observation(batch_visited_tiles, raw_rewards_in_batch)
        
        # Store history
        history.append(epsilon_greedy_map.prob_grid.copy())
        visited_history.extend(batch_visited_tiles)  # Add all visited tiles to history
        rewards_found.append(new_rewards_in_batch)
    
    print("Epsilon-Greedy visited tiles:", visited_history)
    print("Epsilon-Greedy rewards found:", rewards_found)
    print("Epsilon-Greedy total rewards found:", sum(rewards_found))
    print(f"Epsilon-Greedy unique locations visited: {len(visited_set)}/{epsilon_greedy_map.rows * epsilon_greedy_map.cols}")
    
    return history, visited_history, rewards_found


# Alternative implementation of run_thompson_sampling_simulation if needed
def run_thompson_sampling_simulation(reward_map, num_iterations=50, true_reward_locations=None, step_size=2):
    """
    Run a Thompson sampling simulation on the reward map.
    
    Parameters:
        reward_map: RewardMap object
        num_iterations: int - number of iterations to run
        true_reward_locations: list of (x, y) tuples - the actual locations of rewards
        step_size: int - number of cells to explore per iteration
    
    Returns:
        history: list of probability grids over time
        visited_history: list of visited locations
        rewards_found: list of rewards found at each step
        actual_locations: list of actual reward locations
    """
    # Create a deep copy of the reward map to avoid modifying the original
    thompson_map = copy.deepcopy(reward_map)
    
    # If true reward locations not provided, generate them
    if true_reward_locations is None:
        # For demonstration, randomly place rewards based on the initial probability grid
        initial_probs = thompson_map.prob_grid.copy().flatten()
        normalized_probs = initial_probs / initial_probs.sum()
        n_rewards = max(3, int(thompson_map.rows * thompson_map.cols * 0.05))  # Around 5% of cells have rewards
        flat_indices = np.random.choice(thompson_map.rows * thompson_map.cols, 
                                     size=n_rewards, replace=False, p=normalized_probs)
        true_reward_locations = []
        for idx in flat_indices:
            y, x = np.unravel_index(idx, (thompson_map.rows, thompson_map.cols))
            true_reward_locations.append((x, y))
    
    # Store history of probability grids
    history = [thompson_map.prob_grid.copy()]
    visited_history = []
    rewards_found = []
    
    # Keep track of found rewards (each reward can only be found once)
    found_rewards_set = set()
    
    for i in range(num_iterations):
        # Sample from the beta distribution for each cell
        alpha = thompson_map.prob_grid + 1  # Add 1 for Beta(1,1) prior
        beta = 1 - thompson_map.prob_grid + 1  # Add 1 for Beta(1,1) prior
        
        sampled_grid = np.random.beta(alpha, beta)
        
        # Apply the P > 0.1 threshold (mask out cells with low probability)
        probability_mask = thompson_map.prob_grid > 0.1
        # Use a large negative number to ensure these cells aren't selected
        sampled_grid = np.where(probability_mask, sampled_grid, -1.0)
        
        # Batch of cells to visit this iteration
        batch_visited_tiles = []
        
        for _ in range(step_size):
            # Find the highest value in the sampled grid
            flat_idx = np.argmax(sampled_grid)
            y, x = np.unravel_index(flat_idx, sampled_grid.shape)
            visited_tile = (x, y)
            batch_visited_tiles.append(visited_tile)
            
            # Remove this cell from consideration for this batch
            sampled_grid[y, x] = -1  # Set to a value lower than possible Beta samples
        
        # Count rewards - FIXED INDENTATION: This should be inside the loop
        raw_rewards_in_batch = sum(1 for tile in batch_visited_tiles if tile in true_reward_locations)
        new_rewards_in_batch = 0
        for tile in batch_visited_tiles:
            if tile in true_reward_locations and tile not in found_rewards_set:
                new_rewards_in_batch += 1
                found_rewards_set.add(tile)
        
        # Update the map with raw observations
        thompson_map.multi_step_observation(batch_visited_tiles, raw_rewards_in_batch)
        
        # Store history
        history.append(thompson_map.prob_grid.copy())
        visited_history.extend(batch_visited_tiles)  # Add all visited tiles to history
        rewards_found.append(new_rewards_in_batch)
    
    print("Thompson Sampling visited tiles:", visited_history)
    print("Thompson Sampling rewards found:", rewards_found)
    print("Thompson Sampling total rewards found:", sum(rewards_found))
    
    return history, visited_history, rewards_found, true_reward_locations

def compare_all_strategies(reward_map, num_iterations=50, true_reward_locations=None, epsilon=0.1, softmax_temp=0.1, step_size=2):
    """
    Compare Thompson sampling with various greedy strategies.
    
    Parameters:
        reward_map: RewardMap object
        num_iterations: int - number of iterations to run
        true_reward_locations: list of (x, y) tuples - the actual locations of rewards
        epsilon: float - probability of random exploration for epsilon-greedy
        softmax_temp: float - temperature parameter for softmax in greedy
        step_size: int - number of cells to explore per iteration
    """
    # Create deep copies for all algorithms
    thompson_reward_map = copy.deepcopy(reward_map)
    softmax_greedy_map = copy.deepcopy(reward_map)
    epsilon_greedy_map = copy.deepcopy(reward_map)
    
    # Run Thompson sampling
    thompson_history, thompson_visited, thompson_rewards, actual_locations = run_thompson_sampling_simulation(
        thompson_reward_map, num_iterations, true_reward_locations, step_size)
    
    # If true_reward_locations was None, use the generated locations from Thompson
    if true_reward_locations is None:
        true_reward_locations = actual_locations
    
    # Run softmax greedy
    softmax_greedy_history, softmax_greedy_visited, softmax_greedy_rewards = run_true_greedy_sampling_simulation(
        softmax_greedy_map, num_iterations, actual_locations, use_softmax=True, temperature=softmax_temp, step_size=step_size)
    
    # Run epsilon-greedy
    epsilon_greedy_history, epsilon_greedy_visited, epsilon_greedy_rewards = run_epsilon_greedy_sampling_simulation(
        epsilon_greedy_map, num_iterations, actual_locations, epsilon, step_size)
    
    # Calculate cumulative rewards
    thompson_cum_rewards = np.cumsum(thompson_rewards)
    softmax_greedy_cum_rewards = np.cumsum(softmax_greedy_rewards)
    epsilon_greedy_cum_rewards = np.cumsum(epsilon_greedy_rewards)
    
    # Create comparative plots
    fig = plt.figure(figsize=(18, 16))
    
    # 1. Cumulative rewards comparison
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.plot(range(1, len(thompson_cum_rewards) + 1), thompson_cum_rewards, 'b-', 
             linewidth=2, label='Thompson Sampling')
    ax1.plot(range(1, len(softmax_greedy_cum_rewards) + 1), softmax_greedy_cum_rewards, 'g-', 
             linewidth=2, label=f'Softmax Greedy (T={softmax_temp})')
    ax1.plot(range(1, len(epsilon_greedy_cum_rewards) + 1), epsilon_greedy_cum_rewards, 'c-', 
             linewidth=2, label=f'Epsilon-Greedy (ε={epsilon})')
    
    # Add optimal line (can't find more rewards than exist)
    x_range = np.arange(1, num_iterations + 1)
    optimal = np.minimum(x_range * step_size, len(actual_locations))
    ax1.plot(x_range, optimal, 'k--', alpha=0.5, label='Optimal (if perfect knowledge)')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Cumulative Reward Comparison')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Location revisits analysis
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    
    # Count frequency of each visited location
    thompson_counts = {}
    softmax_greedy_counts = {}
    epsilon_greedy_counts = {}
    
    for loc in thompson_visited:
        thompson_counts[loc] = thompson_counts.get(loc, 0) + 1
    for loc in softmax_greedy_visited:
        softmax_greedy_counts[loc] = softmax_greedy_counts.get(loc, 0) + 1
    for loc in epsilon_greedy_visited:
        epsilon_greedy_counts[loc] = epsilon_greedy_counts.get(loc, 0) + 1
    
    # Calculate unique locations visited and max revisits
    thompson_unique = len(thompson_counts)
    softmax_greedy_unique = len(softmax_greedy_counts)
    epsilon_greedy_unique = len(epsilon_greedy_counts)
    
    thompson_max_revisits = max(thompson_counts.values()) if thompson_counts else 0
    softmax_greedy_max_revisits = max(softmax_greedy_counts.values()) if softmax_greedy_counts else 0
    epsilon_greedy_max_revisits = max(epsilon_greedy_counts.values()) if epsilon_greedy_counts else 0
    
    # Create a histogram of location visit counts
    max_revisits = max(thompson_max_revisits, 
                     softmax_greedy_max_revisits, epsilon_greedy_max_revisits)
    bins = np.arange(1, max_revisits + 2) - 0.5
    
    thompson_hist = [sum(1 for v in thompson_counts.values() if v == i) for i in range(1, len(bins))]
    softmax_greedy_hist = [sum(1 for v in softmax_greedy_counts.values() if v == i) for i in range(1, len(bins))]
    epsilon_greedy_hist = [sum(1 for v in epsilon_greedy_counts.values() if v == i) for i in range(1, len(bins))]
    
    x = np.arange(1, len(bins))
    width = 0.2
    
    ax2.bar(x - 1.5*width, thompson_hist, width, label='Thompson')
    ax2.bar(x + 0.5*width, softmax_greedy_hist, width, label='Softmax Greedy')
    ax2.bar(x + 1.5*width, epsilon_greedy_hist, width, label='Epsilon-Greedy')
    
    ax2.set_xlabel('Number of visits to same location')
    ax2.set_ylabel('Count of locations')
    ax2.set_title('Location Revisit Analysis')
    ax2.set_xticks(x)
    ax2.legend()
    
    # 3. Final probability maps
    grid_spec = plt.GridSpec(2, 2, wspace=0.2, hspace=0.3)
    ax3 = plt.subplot(grid_spec[1, 0])
    ax4 = plt.subplot(grid_spec[1, 1])
    
    # Thompson final map
    im1 = ax3.imshow(thompson_history[-1], cmap='viridis')
    ax3.set_title('Thompson Final Map')
    fig.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
    
    # Mark true reward locations
    for x, y in actual_locations:
        ax3.plot(y, x, 'r*', markersize=8)
    
    # Softmax Greedy final map
    im2 = ax4.imshow(softmax_greedy_history[-1], cmap='viridis')
    ax4.set_title('Softmax Greedy Final Map')
    fig.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    
    # Mark true reward locations
    for x, y in actual_locations:
        ax4.plot(y, x, 'r*', markersize=8)
    
    # Add summary statistics
    stats_text = (
        f"Performance After {num_iterations} Iterations (step_size={step_size}):\n"
        f"Strategy                   | Total Rewards | Unique Locations | Max Revisits\n"
        f"----------------------------|--------------|-----------------|------------\n"
        f"Thompson Sampling           | {sum(thompson_rewards)}/{len(actual_locations)} | {thompson_unique} | {thompson_max_revisits}\n"
        f"Softmax Greedy (T={softmax_temp})   | {sum(softmax_greedy_rewards)}/{len(actual_locations)} | {softmax_greedy_unique} | {softmax_greedy_max_revisits}\n"
        f"Epsilon-Greedy (ε={epsilon})   | {sum(epsilon_greedy_rewards)}/{len(actual_locations)} | {epsilon_greedy_unique} | {epsilon_greedy_max_revisits}"
    )
    
    plt.figtext(0.5, 0.01, stats_text, ha='center', va='center', fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the text
    plt.show()
    
    return {
        'thompson': sum(thompson_rewards),
        'softmax_greedy': sum(softmax_greedy_rewards),
        'epsilon_greedy': sum(epsilon_greedy_rewards),
        'max_possible': len(actual_locations)
    }


reward_map = RewardMap(rows=12, cols=12)
reward_map.add_reward_region(center_x=8, center_y=8, region_size=5, estimated_reward=5.0)
reward_map.add_reward_region(center_x=2, center_y=2, region_size=5, estimated_reward=5.0)

# # Define true reward locations (or let the function generate them)
true_reward_locations = [(7, 7), (8, 8), (9, 9), (2, 2), (2, 3), (3, 2), (4,4)]

# Compare Thompson sampling with greedy sampling
comparison_results = compare_all_strategies(reward_map, num_iterations=40, true_reward_locations=true_reward_locations, step_size=2)

