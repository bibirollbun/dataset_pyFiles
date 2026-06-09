# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations 
from collections import deque


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations 
from collections import deque
import os

# Define the standard path for the test challenges
TEST_DATA_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"

# ---------------------
class ARCSolver:    
    """
    A class to encapsulate the logic for solving and visualizing ARC tasks.
    This provides a more organized and scalable structure for the project.
    """
    def __init__(self, file_path):        
        """
        Initializes the solver with the path to the dataset file.
                
        Args:
            file_path (str): The path to the ARC JSON data file.
        """
        self.file_path = file_path
        self.tasks = {}
        self.submission = {}
        # Define a safe placeholder for invalid predictions
        self.SAFE_PLACEHOLDER = [[0]]

    @staticmethod
    def flattener(grid: list) -> str:        
        """
        Converts a predicted 2D grid (list of lists of ints) into the required 
        pipe-delimited string format for the ARC competition (e.g., "|12|34|").
        Handles invalid grids by returning a safe empty string.
        """
        # Ensure the grid is valid for processing
        if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
            return ""

        formatted_rows = []
        for row in grid:
            try:
                # Convert integers to strings and join without a delimiter
                formatted_row = "".join(map(str, row))
                formatted_rows.append(formatted_row)
            except Exception:
                # Skip corrupted rows
                continue

        # Join all formatted rows with the pipe delimiter and enclose in pipes
        return "|" + "|".join(formatted_rows) + "|"

    def load_tasks(self):        
        """
        Loads the ARC tasks from the JSON file.
                
        Returns:
            bool: True if tasks are loaded successfully, False otherwise.
        """
        try:
            if os.path.isfile(self.file_path):
                with open(self.file_path, 'r') as f:
                    self.tasks = json.load(f)
                print(f"ARC tasks loaded successfully from single file: {len(self.tasks)} tasks found.")
                return True
            else:
                print(f"Error: Path '{self.file_path}' is not a recognized file. Please adjust the path.")
                return False
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found. Please ensure the data is in the correct directory.")
            return False
        except Exception as e:
            print(f"An error occurred while loading tasks: {e}")
            return False

    # @staticmethod
    # def visualize_grid(grid, title="ARC Grid"):        
    #     """
    #     Static method to plot a 2D grid as a colored image.
    #     NOTE: This is commented out for execution environment stability.
    #     """
    #     fig, ax = plt.subplots()
    #     ax.imshow(grid, cmap='viridis')
    #     ax.set_title(title)
    #     plt.show()
    #     pass


    # --- Basic Transformation Helper Functions (Now taking and returning list[list[int]]) ---
    
    def _rotate_90(self, grid):        
        """Helper function to rotate a grid 90 degrees clockwise. Expects NumPy array, Returns list."""
        # np.rot90(k=1) is 90 deg counter-clockwise. k=3 is 90 deg clockwise.
        return np.rot90(np.array(grid), k=3).tolist()

    def _flip_vertical(self, grid):        
        """Helper function to flip a grid vertically. Expects NumPy array, Returns list."""
        return np.flipud(np.array(grid)).tolist()

    def _flip_horizontal(self, grid):        
        """Helper function to flip a grid horizontally. Expects NumPy array, Returns list."""
        return np.fliplr(np.array(grid)).tolist()

    def _recolor_by_value(self, grid, old_color, new_color):        
        """Helper function to change all instances of one color to another. Expects NumPy array, Returns list."""
        return np.where(np.array(grid) == old_color, new_color, np.array(grid)).tolist()

    def _trim_borders(self, grid):        
        """Helper function to remove the 1-pixel border from the grid. Expects NumPy array, Returns list."""
        grid_array = np.array(grid)
        if grid_array.shape[0] > 2 and grid_array.shape[1] > 2:
            return grid_array[1:-1, 1:-1].tolist()
        return grid_array.tolist()
        
    def _invert_colors(self, grid):        
        """Helper function to invert the colors (0->9, 1->8, etc.) assuming 0-9 values. Expects NumPy array, Returns list."""
        grid_array = np.array(grid)
        inverted_grid = 9 - grid_array
        return inverted_grid.tolist()

    def _crop_to_content(self, grid):        
        """Helper function to crop the grid to the bounding box of non-zero values. Expects NumPy array, Returns list."""
        grid_array = np.array(grid)
        rows = np.any(grid_array, axis=1)
        cols = np.any(grid_array, axis=0)
        if not np.any(rows) or not np.any(cols):
            return grid_array.tolist() # Return original if no content found
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
        return grid_array[min_row:max_row+1, min_col:max_col+1].tolist()
        
    # --- Advanced Logic-Based Transformations ---
    def _remove_isolated_pixels(self, grid):        
        """        
        Helper function to remove any pixel that is not adjacent (up, down, left, right)
        to another pixel of the same color. Expects NumPy array, Returns list.
        """
        grid_array = np.array(grid)
        output_grid = grid_array.copy()
        rows, cols = grid_array.shape
        
        for r in range(rows):
            for c in range(cols):
                if grid_array[r, c] != 0: # Check only non-empty cells
                    color = grid_array[r, c]
                    is_isolated = True
                    # Check neighbors (up, down, left, right)
                    if r > 0 and grid_array[r-1, c] == color: is_isolated = False
                    if r < rows-1 and grid_array[r+1, c] == color: is_isolated = False
                    if c > 0 and grid_array[r, c-1] == color: is_isolated = False
                    if c < cols-1 and grid_array[r, c+1] == color: is_isolated = False
                    
                    if is_isolated:
                        output_grid[r, c] = 0 # Remove the isolated pixel
        return output_grid.tolist()

    def _fill_with_dominant_color(self, grid):
        """
        Helper function: Finds the most frequent non-zero color in the grid and 
        returns a new grid of the same dimensions filled entirely with that color.
        If no non-zero colors are found, returns the original grid.
        Expects list[list[int]], Returns list[list[int]].
        """
        grid_array = np.array(grid)
        
        # 1. Count non-zero colors
        # Get unique values and their counts
        colors, counts = np.unique(grid_array, return_counts=True)
        color_counts = dict(zip(colors, counts))
        
        # Remove background color (0) from consideration
        color_counts.pop(0, None)
        
        if not color_counts:
            # No content, return original grid dimensions
            return grid_array.tolist()
            
        # 2. Find the dominant color
        dominant_color = max(color_counts, key=color_counts.get)
        
        # 3. Create a new grid of the same shape filled with the dominant color
        rows, cols = grid_array.shape
        # Use dtype=int to ensure the grid elements are integers
        new_grid = np.full((rows, cols), dominant_color, dtype=int)
        
        return new_grid.tolist()

    def _double_size(self, grid):
        """
        Helper function: Upscales the grid size by 2x by repeating each pixel 2x2.
        Expects list[list[int]], Returns list[list[int]].
        """
        grid_array = np.array(grid)
        
        # Use np.kron (Kronecker product) to repeat each element 2x2, effectively doubling size
        upscaled_grid = np.kron(grid_array, np.ones((2, 2), dtype=int))
        
        return upscaled_grid.tolist()


    def _test_rule(self, task, rule_function):        
        """Tests a single rule or a combination of rules on all training pairs."""
        num_train_pairs = len(task['train'])
        correct_matches = 0
        
        for train_pair in task['train']:
            input_grid = np.array(train_pair['input'])
            output_grid = np.array(train_pair['output'])
            
            try:
                # Apply the rule, which is expected to return a list[list[int]]
                predicted_grid_list = rule_function(input_grid.tolist())
                
                # Convert back to np.array for reliable comparison
                if np.array_equal(np.array(predicted_grid_list), output_grid):
                    correct_matches += 1
            except Exception:
                # In case a transformation fails for a specific grid size
                pass
        
        return correct_matches, num_train_pairs


    def infer_and_solve_combined_rules(self, task):        
        """
        This function infers a rule by trying single and combined transformations.
                
        Returns:
            function: The best-matching rule function (takes a list/np.array, returns list[list[int]]), 
                      or a lambda that returns the input grid unchanged.
        """
        # All functions take a grid (list or array) and return a list[list[int]]
        all_rules = [
            self._flip_vertical,
            self._flip_horizontal,
            self._rotate_90,
            self._trim_borders,
            self._crop_to_content,
            self._invert_colors,
            self._remove_isolated_pixels,
            self._fill_with_dominant_color,
            self._double_size, # <-- NEW RULE ADDED
            # Identity (returns input grid unchanged)
            lambda grid: grid.tolist()
        ]
        
        # We will try single rules first
        for rule in all_rules:
            correct_matches, num_pairs = self._test_rule(task, rule)
            
            if correct_matches == len(task['train']):
                print(f"Inferred rule for task: {rule.__name__ if hasattr(rule, '__name__') else 'Identity'} (Single Rule)")
                return rule

        # If single rules fail, try combinations of 2 transformations (order matters)
        for rule1 in all_rules:
            for rule2 in all_rules:
                
                # Use a combined rule function that correctly applies r1 then r2
                # Note: r1's output (list) is converted to array before r2 (if r2 expects array, 
                # but all our helpers are designed to handle either and return list[list[int]] for simplicity)
                def combined_rule(grid_input, r1=rule1, r2=rule2):
                    # Start with the list[list[int]] input grid
                    intermediate_grid = r1(grid_input) 
                    final_grid = r2(intermediate_grid)
                    return final_grid

                correct_matches, num_pairs = self._test_rule(task, combined_rule)

                if correct_matches == len(task['train']):
                    name1 = rule1.__name__ if hasattr(rule1, '__name__') else 'Identity'
                    name2 = rule2.__name__ if hasattr(rule2, '__name__') else 'Identity'
                    print(f"Inferred combined rule for task: {name1} -> {name2}")
                    return combined_rule
        
        # Return a placeholder rule if no combination is found (returns input grid unchanged)
        print("Inferred rule for task: Identity (Fallback)")
        return lambda grid: grid.tolist()


    def run(self):        
        """
        Main method to run the entire process: loading, solving, and saving.
        """
        if not self.load_tasks():
            return

        # Process all tasks for the submission
        num_tasks_processed = 0
        for task_id, task in self.tasks.items():
            print(f"\nProcessing task: {task_id}")
            num_tasks_processed += 1

            # --- Rule Inference ---
            inferred_rule = self.infer_and_solve_combined_rules(task)
            
            # --- Test Case Prediction ---
            predictions = []
            if 'test' in task:
                for test_pair in task['test']:
                    input_grid = test_pair['input'] # Input is list[list[int]]

                    # 1. Attempt 1: Inferred Rule (Highest Confidence)
                    try:
                        predicted_grid_A1 = inferred_rule(input_grid)
                    except Exception:
                        predicted_grid_A1 = self.SAFE_PLACEHOLDER
                    
                    # 2. Attempt 2: Simple Rotation (90 degrees clockwise) - Generic Guess
                    try:
                        predicted_grid_A2 = self._rotate_90(input_grid) 
                    except Exception:
                        predicted_grid_A2 = self.SAFE_PLACEHOLDER

                    # 3. Attempt 3: Simple Horizontal Flip - Generic Guess
                    try:
                        predicted_grid_A3 = self._flip_horizontal(input_grid)
                    except Exception:
                        predicted_grid_A3 = self.SAFE_PLACEHOLDER

                    # Apply safety placeholder if output is invalid or empty
                    A1 = predicted_grid_A1 if predicted_grid_A1 and isinstance(predicted_grid_A1, list) and isinstance(predicted_grid_A1[0], list) else self.SAFE_PLACEHOLDER
                    A2 = predicted_grid_A2 if predicted_grid_A2 and isinstance(predicted_grid_A2, list) and isinstance(predicted_grid_A2[0], list) else self.SAFE_PLACEHOLDER
                    A3 = predicted_grid_A3 if predicted_grid_A3 and isinstance(predicted_grid_A3, list) and isinstance(predicted_grid_A3[0], list) else self.SAFE_PLACEHOLDER

                    # Format and store the three predictions
                    predictions.append({
                        "attempt_1": self.flattener(A1),
                        "attempt_2": self.flattener(A2),
                        "attempt_3": self.flattener(A3)
                    })
                
                # Store the list of three-attempt predictions for the task
                self.submission[task_id] = predictions

        # Save the submission file
        try:
            with open('submission.json', 'w') as f:
                json.dump(self.submission, f, indent=4)
            print(f"\n✅ Successfully created submission file 'submission.json' with {num_tasks_processed} tasks.")
        except Exception as e:
            print(f"\nError saving submission.json: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    solver = ARCSolver(TEST_DATA_PATH)
    solver.run()





