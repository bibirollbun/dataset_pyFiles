# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing libraries 
import json                                    # for reading and writing JSON data files.
import matplotlib.pyplot as plt                # for visualizing the grids as images.
import numpy as np                             # for numerical operations and array manipulation.
from itertools import permutations, product             # to generate all possible combinations of rules.
from scipy.ndimage import label, find_objects  # for advanced image processing tasks 
import collections 
from pprint import pprint  # Used for pretty-printing complex data structures.


# Define the main class for the ARC Solver.
class ARCSolver:
    """
    A class to encapsulate the logic for solving and visualizing ARC tasks.
    """
    def __init__(self, file_path):
        """
        Initializes the solver with the path to the dataset file.
        
        Args:
            file_path (str): The path to the ARC JSON data file.
        """
        self.file_path = file_path
        self.tasks = {}
        self.submission = {}  # Add an attribute to store all task predictions.
        self.inferred_rules = {}  # Add an attribute to store inferred rules.

    def load_tasks(self):
        """
        Loads the ARC tasks from the JSON file.
        
        Returns:
            bool: True if tasks are loaded successfully, False otherwise.
        """
        try:
            with open(self.file_path, 'r') as f:
                self.tasks = json.load(f)
            print("ARC tasks loaded successfully.")
            return True
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found. Please ensure the data is in the correct directory.")
            print("Please download the official ARC dataset from the GitHub repository and place the 'training.json' file in this directory.")
            return False
        except Exception as e:
            print(f"An error occurred while loading tasks: {e}")
            return False

    # --- Basic Transformation Helper Functions (no parameters) ---
    def _rotate_90(self, grid):
        """Helper function to rotate a grid 90 degrees clockwise."""
        return np.rot90(np.array(grid), k=3).tolist()

    def _flip_vertical(self, grid):
        """Helper function to flip a grid vertically."""
        return np.flipud(np.array(grid)).tolist()

    def _flip_horizontal(self, grid):
        """Helper function to flip a grid horizontally."""
        return np.fliplr(np.array(grid)).tolist()
    
    def _trim_borders(self, grid):
        """Helper function to remove the 1-pixel border from the grid."""
        grid_np = np.array(grid)
        if grid_np.shape[0] > 2 and grid_np.shape[1] > 2:
            return grid_np[1:-1, 1:-1].tolist()
        return grid_np.tolist()
        
    def _invert_colors(self, grid):
        """Helper function to invert the colors (0->9, 1->8, etc.) assuming 0-9 values."""
        inverted_grid = 9 - np.array(grid)
        return inverted_grid.tolist()

    def _crop_to_content(self, grid):
        """Helper function to crop the grid to the bounding box of non-zero values."""
        grid_np = np.array(grid)
        rows = np.any(grid_np, axis=1)
        cols = np.any(grid_np, axis=0)
        if not np.any(rows) or not np.any(cols):
            return grid_np.tolist()
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
        return grid_np[min_row:max_row+1, min_col:max_col+1].tolist()
        
    def _fill_with_most_frequent_color(self, grid):
        """Fills the entire grid with the most frequent non-zero color."""
        grid_np = np.array(grid)
        flat_grid = grid_np.flatten()
        non_zero_colors = flat_grid[flat_grid != 0]

        if non_zero_colors.size == 0:
            return grid_np.tolist()

        most_common_color = collections.Counter(non_zero_colors).most_common(1)[0][0]
        
        return np.full(grid_np.shape, most_common_color, dtype=int).tolist()

    # --- Advanced Logic-Based Transformations (no parameters) ---
    def _remove_isolated_pixels(self, grid):
        """Removes any pixel not adjacent to another pixel of the same color."""
        grid_np = np.array(grid)
        output_grid = grid_np.copy()
        rows, cols = grid_np.shape
        for r in range(rows):
            for c in range(cols):
                if grid_np[r, c] != 0:
                    is_isolated = True
                    if r > 0 and grid_np[r-1, c] == grid_np[r, c]: is_isolated = False
                    if r < rows-1 and grid_np[r+1, c] == grid_np[r, c]: is_isolated = False
                    if c > 0 and grid_np[r, c-1] == grid_np[r, c]: is_isolated = False
                    if c < cols-1 and grid_np[r, c+1] == grid_np[r, c]: is_isolated = False
                    if is_isolated:
                        output_grid[r, c] = 0
        return output_grid.tolist()

    def _fill_holes(self, grid):
        """Fills holes in a shape."""
        grid_np = np.array(grid)
        padded_grid = np.pad(grid_np, 1, mode='constant', constant_values=1)
        labeled_holes, num_labels = label(padded_grid == 0)
        
        for i in range(1, num_labels + 1):
            if not np.any(labeled_holes[0, :] == i) and \
               not np.any(labeled_holes[-1, :] == i) and \
               not np.any(labeled_holes[:, 0] == i) and \
               not np.any(labeled_holes[:, -1] == i):
                hole_coords = np.where(labeled_holes == i)
                surrounding_color = 0
                for r, c in zip(*hole_coords):
                    if r > 0 and padded_grid[r-1, c] != 0: surrounding_color = padded_grid[r-1, c]
                    elif r < padded_grid.shape[0]-1 and padded_grid[r+1, c] != 0: surrounding_color = padded_grid[r+1, c]
                    elif c > 0 and padded_grid[r, c-1] != 0: surrounding_color = padded_grid[r, c-1]
                    elif c < padded_grid.shape[1]-1 and padded_grid[r, c+1] != 0: surrounding_color = padded_grid[r, c+1]
                    if surrounding_color != 0:
                        break
                
                if surrounding_color != 0:
                    for r, c in zip(*hole_coords):
                        padded_grid[r, c] = surrounding_color
        
        return padded_grid[1:-1, 1:-1].tolist()

    def _find_largest_object(self, grid):
        """Keeps only the largest connected object in the grid."""
        grid_np = np.array(grid)
        labeled_objects, num_objects = label(grid_np)
        
        if num_objects == 0:
            return grid_np.tolist()
        
        object_sizes = np.bincount(labeled_objects.ravel())
        largest_object_label = np.argmax(object_sizes[1:]) + 1
        
        output_grid = np.where(labeled_objects == largest_object_label, grid_np, 0)
        
        return output_grid.tolist()

    # --- Parameterized Transformations ---
    def _recolor_by_value(self, grid, old_color, new_color):
        """Helper function to change all instances of one color to another."""
        return np.where(np.array(grid) == old_color, new_color, grid).tolist()

    # --- Rule Inference Methods ---
    def _test_rule(self, task, rule_function):
        """Tests a single rule or a combination of rules on all training pairs."""
        num_train_pairs = len(task['train'])
        correct_matches = 0
        
        for train_pair in task['train']:
            input_grid = np.array(train_pair['input'])
            output_grid = np.array(train_pair['output'])
            
            try:
                predicted_output = np.array(rule_function(input_grid))
                if np.array_equal(predicted_output, output_grid):
                    correct_matches += 1
            except Exception:
                pass
        
        return correct_matches, num_train_pairs

    def _infer_and_solve_efficient(self, task):
        """
        Infers a rule using a more efficient, non-brute-force approach.
        
        Returns:
            A tuple of (rule_function, rule_name_string) or (lambda, "No rule found").
        """
        # 1. First, check for simple, single-step rules.
        unparameterized_rules = {
            '_flip_vertical': self._flip_vertical,
            '_flip_horizontal': self._flip_horizontal,
            '_rotate_90': self._rotate_90,
            '_trim_borders': self._trim_borders,
            '_crop_to_content': self._crop_to_content,
            '_invert_colors': self._invert_colors,
            '_remove_isolated_pixels': self._remove_isolated_pixels,
            '_fill_holes': self._fill_holes,
            '_find_largest_object': self._find_largest_object,
            '_fill_with_most_frequent_color': self._fill_with_most_frequent_color
        }

        for rule_name, rule_func in unparameterized_rules.items():
            correct_matches, num_pairs = self._test_rule(task, rule_func)
            if correct_matches == num_pairs:
                return rule_func, rule_name
        
        # 2. Next, infer and test specific recoloring rules.
        color_diffs = collections.defaultdict(list)
        for pair in task['train']:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            if input_grid.shape != output_grid.shape: continue
            for old_c in np.unique(input_grid):
                if old_c == 0: continue
                mask = (input_grid == old_c)
                output_colors = output_grid[mask]
                output_colors = output_colors[output_colors != old_c]
                if len(output_colors) > 0:
                    new_c = collections.Counter(output_colors).most_common(1)[0][0]
                    if old_c != new_c:
                        color_diffs[old_c].append(new_c)

        for old_c, new_cs in color_diffs.items():
            if len(new_cs) == len(task['train']) and all(c == new_cs[0] for c in new_cs):
                new_c = new_cs[0]
                def recolor_rule(grid): return self._recolor_by_value(grid, old_c, new_c)
                correct_matches, num_pairs = self._test_rule(task, recolor_rule)
                if correct_matches == num_pairs:
                    return recolor_rule, f"_recolor_by_value(from={old_c}, to={new_c})"

        # 3. Finally, test a limited set of high-probability combinations.
        high_prob_combinations = [
            (self._crop_to_content, self._rotate_90),
            (self._trim_borders, self._invert_colors),
        ]
        
        for rule_1, rule_2 in high_prob_combinations:
            def combined_rule(grid):
                return rule_2(rule_1(grid))
            
            rule_name = f"{rule_1.__name__} -> {rule_2.__name__}"
            correct_matches, num_pairs = self._test_rule(task, combined_rule)
            if correct_matches == num_pairs:
                return combined_rule, rule_name

        return lambda grid: grid, "No rule found"

    def save_inferred_rules_to_json(self, filename="inferred_rules.json"):
        """Saves the inferred rules to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.inferred_rules, f, indent=4)
        print(f"\nInferred rules saved to '{filename}'.")
        
    def run(self):
        """
        Main method to run the entire process: loading, solving, and saving results.
        """
        if not self.load_tasks():
            return

        for task_id, task in self.tasks.items():
            print(f"\n--- Processing task: {task_id} ---")

            inferred_rule, inferred_rule_name = self._infer_and_solve_efficient(task)
            
            self.inferred_rules[task_id] = inferred_rule_name
            
            if inferred_rule_name != "No rule found":
                print(f"Inferred Rule: {inferred_rule_name}")
            else:
                print("No rule found for this task.")
            
            print("\nTraining Pairs:")
            for i, pair in enumerate(task['train']):
                print(f"  Training Pair {i+1}:")
                print("    Input:")
                pprint(pair['input'])
                print("    Output:")
                pprint(pair['output'])
            
            predictions = [inferred_rule(np.array(pair['input'])) for pair in task['test']]
            
            self.submission[task_id] = predictions

            print("\nTest Predictions:")
            for i, pair in enumerate(task['test']):
                print(f"  Test Pair {i+1}:")
                print("    Input:")
                pprint(pair['input'])
                print("    Prediction:")
                pprint(predictions[i])
            print("-" * 30)
        
        # --- Final step: Save all results to JSON files ---
        print("\n--- Saving results to JSON files ---")

        # 1. Save all task predictions to submission.json
        submission_file_path = "submission.json"
        with open(submission_file_path, 'w') as f:
            json.dump(self.submission, f, indent=4)
        print(f"âœ… All task predictions saved to '{submission_file_path}'.")

        # 2. Save all inferred rules to inferred_rules.json
        self.save_inferred_rules_to_json()
        
# --- Main execution block for running the solver ---
if __name__ == "__main__":
    # The official ARC training dataset is named 'arc-agi_training_challenges.json'
    # in the standard distribution.
    file_path = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
    
    solver = ARCSolver(file_path)
    
    # === Control to run for a single task or all tasks ===
    # Change this variable to 'single' to run only the first task.
    # Change it to 'all' to run all tasks in the dataset.
    run_mode = "all"
    
    if run_mode == "single":
        if solver.load_tasks() and solver.tasks:
            first_task_id = list(solver.tasks.keys())[0]
            task_to_run = {first_task_id: solver.tasks[first_task_id]}
            print(f"Running for a single task: {first_task_id}")
            solver.tasks = task_to_run
            solver.run()
        else:
            print("No tasks found to run.")
    elif run_mode == "all":
        print("Running for all tasks in the dataset.")
        solver.run()
    else:
        print(f"Invalid run mode: '{run_mode}'. Please use 'single' or 'all'.")




