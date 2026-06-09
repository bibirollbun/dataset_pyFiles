import os
import json
import copy
import random
import math
import itertools
import functools
from collections import defaultdict, Counter
from pprint import pprint

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
%matplotlib inline

from tqdm.notebook import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


import tqdm
from random import sample
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


class ARCDataset:
    def __init__(self, train_path=None, train_solutions_path=None, test_path=None, eval_path=None, eval_solutions_path=None):
        self.train_data = self._load_json(train_path) if train_path else {}
        self.train_solutions = self._load_json(train_solutions_path) if train_solutions_path else {}
        self.test_data = self._load_json(test_path) if test_path else {}
        self.eval_data = self._load_json(eval_path) if eval_path else {}
        self.eval_solutions = self._load_json(eval_solutions_path) if eval_solutions_path else {}

        # Define colormap and normalization
        self.ARC_COLORMAP = colors.ListedColormap([
            '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#000000'  # 11th color for blank
        ])
        self.ARC_NORM = colors.Normalize(vmin=0, vmax=10)

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_task(self, task_id, split='train'):
        if split == 'train':
            return self.train_data.get(task_id), self.train_solutions.get(task_id)
        elif split == 'test':
            return self.test_data.get(task_id), None
        elif split == 'eval':
            return self.eval_data.get(task_id), self.eval_solutions.get(task_id)
        else:
            raise ValueError("split must be 'train', 'test', or 'eval'")

    def resize_grid_to_30x30(self, grid):
        """
        Resizes a grid to 30x30, adding blank pixels (value 10) as needed,
        and positions the original grid in the bottom-left corner.
        """
        new_grid = [[10] * 30 for _ in range(30)]
        original_height = len(grid)
        original_width = len(grid[0])
        
        for i in range(original_height):
            for j in range(original_width):
                new_grid[30 - original_height + i][j] = grid[i][j]

        return new_grid

    def revert_from_30x30(self, grid):
        """
        Converts a tweaked (30x30) grid back to its original raw format
        by removing blank pixels.
        """
        non_blank_rows = [i for i, row in enumerate(grid) if any(cell != 10 for cell in row)]
        non_blank_cols = [j for j in range(len(grid[0])) if any(row[j] != 10 for row in grid)]
        
        min_row, max_row = min(non_blank_rows), max(non_blank_rows)
        min_col, max_col = min(non_blank_cols), max(non_blank_cols)

        return [row[min_col:max_col + 1] for row in grid[min_row:max_row + 1]]

    def create_tweaked_training_data(self, task_data):
        """
        Creates tweaked (30x30) versions of training inputs and outputs.
        Returns two lists: tweaked_inputs and tweaked_outputs.
        """
        tweaked_inputs = [self.resize_grid_to_30x30(example['input']) for example in task_data.get('train', [])]
        tweaked_outputs = [self.resize_grid_to_30x30(example['output']) for example in task_data.get('train', [])]
        return tweaked_inputs, tweaked_outputs

    def create_tweaked_unsolved_input(self, task_data):
        """
        Creates tweaked (30x30) version of an unsolved task input.
        Returns a single tweaked grid.
        """
        test_examples = task_data.get('test', [])
        if not test_examples:
            raise ValueError("No test examples found in task data.")
        return self.resize_grid_to_30x30(test_examples[0]['input'])

    def plot_raw_task(self, task_data, task_solution, title="Raw Task Visualization"):
        """
        Plots the raw task data (inputs and outputs) along with the task solution in a 3x2 grid layout.
        """
        train_examples = task_data.get('train', [])
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))
        plt.suptitle(title, fontsize=16)

        for col, example in enumerate(train_examples):
            # Row 1: Raw inputs
            axs[0, col].imshow(example['input'], cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[0, col].set_title(f"Raw Input {col + 1}")
            axs[0, col].axis('off')

            # Row 2: Raw outputs
            axs[1, col].imshow(example['output'], cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[1, col].set_title(f"Raw Output {col + 1}")
            axs[1, col].axis('off')

            if(col==0):
                # Row 3: Raw task solution
                axs[2, col].imshow(task_solution[col], cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
                axs[2, col].set_title(f"Raw Solution {col + 1}")
                axs[2, col].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_tweaked_task(self, task_data, task_solution, title="Tweaked Task Visualization"):
        """
        Plots the tweaked (30x30) version of the task data (inputs and outputs)
        along with the tweaked task solution in a 3x2 grid layout.
        """
        train_examples = task_data.get('train', [])
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))
        plt.suptitle(title, fontsize=16)

        for col, example in enumerate(train_examples):
            # Row 1: Tweaked inputs
            tweaked_input = self.resize_grid_to_30x30(example['input'])
            axs[0, col].imshow(tweaked_input, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[0, col].set_title(f"Tweaked Input {col + 1}")
            axs[0, col].axis('off')

            # Row 2: Tweaked outputs
            tweaked_output = self.resize_grid_to_30x30(example['output'])
            axs[1, col].imshow(tweaked_output, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[1, col].set_title(f"Tweaked Output {col + 1}")
            axs[1, col].axis('off')

            if(col==0):
                # Row 3: Tweaked solutions
                tweaked_solution = self.resize_grid_to_30x30(task_solution[col])
                axs[2, col].imshow(tweaked_solution, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
                axs[2, col].set_title(f"Tweaked Solution {col + 1}")
                axs[2, col].axis('off')

        plt.tight_layout()
        plt.show()

    def recolor_and_create_dataframes(self, tweaked_inputs, tweaked_outputs, tweaked_test_input):
        """
        Recolors grids based on frequency of colors and returns them as Pandas DataFrames.
        Most frequent color is recolored as 11, next as 12, etc.
        """
        import pandas as pd
        from collections import Counter

        def recolor_grid(grid, coloring_scheme="default"):
            """
            Recolors the grid based on the chosen coloring scheme.
        
            Parameters:
            - grid: 2D list representing the grid to be recolored.
            - coloring_scheme: String specifying the recoloring scheme. 
              - "default": Recolors based on frequency of colors.
              - "orientation": Recolors based on the grid traversal starting from the bottom-left.
        
            Returns:
            - recolored_df: Pandas DataFrame of the recolored grid (30x30), with new color values.
            """
            import pandas as pd
            from collections import Counter
        
            if coloring_scheme == "default":
                # Default scheme: Based on frequency of colors
                # Flatten grid to count occurrences of colors
                flat_grid = [cell for row in grid for cell in row]
                color_counts = Counter(flat_grid)
                # Map colors to new values based on frequency
                sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x], reverse=True)
                recolor_map = {color: 11 + idx for idx, color in enumerate(sorted_colors)}
                # Apply recoloring
                recolored_grid = [[recolor_map[cell] for cell in row] for row in grid]
        
            elif coloring_scheme == "orientation":
                # Orientation scheme: Start from the bottom-left and assign colors row-by-row
                new_color = 11  # Start assigning colors from 11
                recolor_map = {}
                recolored_grid = []
        
                for i in range(len(grid) - 1, -1, -1):  # Iterate rows from bottom to top
                    recolored_row = []
                    for j in range(len(grid[i])):  # Iterate columns from left to right
                        color = grid[i][j]
                        # Assign a new color if the color hasn't been mapped yet
                        if color not in recolor_map:
                            recolor_map[color] = new_color
                            new_color += 1
                        recolored_row.append(recolor_map[color])
                    recolored_grid.insert(0, recolored_row)  # Insert row to form correct top-to-bottom orientation
        
            else:
                raise ValueError("Invalid coloring_scheme. Choose 'default' or 'orientation'.")
        
            # Convert to DataFrame
            recolored_df = pd.DataFrame(recolored_grid, index=range(29, -1, -1), columns=range(30))
            return recolored_df


        # Recolor all grids
        recolored_inputs = [recolor_grid(grid) for grid in tweaked_inputs]
        recolored_outputs = [recolor_grid(grid) for grid in tweaked_outputs]
        recolored_test_input = recolor_grid(tweaked_test_input)

        return recolored_inputs + recolored_outputs + [recolored_test_input]


DATA_PATH = '/kaggle/input/arc-prize-2025'
dataset = ARCDataset(
    train_path=f'{DATA_PATH}/arc-agi_training_challenges.json',
    train_solutions_path=f'{DATA_PATH}/arc-agi_training_solutions.json',
    test_path=f'{DATA_PATH}/arc-agi_test_challenges.json',
    eval_path=f'{DATA_PATH}/arc-agi_evaluation_challenges.json',
    eval_solutions_path=f'{DATA_PATH}/arc-agi_evaluation_solutions.json',
)


task_data, task_solution = dataset.get_task('00576224', split='train')


dataset.plot_raw_task(task_data, task_solution, title="Raw Task Visualization")


dataset.plot_tweaked_task(task_data, task_solution, title="Tweaked Task Visualization")


# Step 1: Generate tweaked data
tweaked_inputs, tweaked_outputs = dataset.create_tweaked_training_data(task_data)
tweaked_test_input = dataset.create_tweaked_unsolved_input(task_data)

# Step 2: Recolor and create DataFrames
dataframes = dataset.recolor_and_create_dataframes(tweaked_inputs, tweaked_outputs, tweaked_test_input)

# Access each DataFrame
input_df1 = dataframes[0]
input_df2 = dataframes[1]
output_df1 = dataframes[2]
output_df2 = dataframes[3]
test_input_df = dataframes[4]


input_df2


def create_color_pair_dataframes(input_df1, input_df2, output_df1, output_df2, input_formula, output_formula):
    """
    Creates two sets of paired DataFrames (input1-output1, input2-output2) based on pixel color mappings.
    Each cell in the DataFrame contains a tuple (input_color, output_color) based on formulas.

    Parameters:
    - input_df1: DataFrame of the first input grid (30x30).
    - input_df2: DataFrame of the second input grid (30x30).
    - output_df1: DataFrame of the first output grid (30x30).
    - output_df2: DataFrame of the second output grid (30x30).
    - input_formula: Function/formula to determine the pixel index in the input grid.
    - output_formula: Function/formula to determine the pixel index in the output grid.

    Returns:
    - input1_output1_df: DataFrame of paired colors (input_df1, output_df1).
    - input2_output2_df: DataFrame of paired colors (input_df2, output_df2).
    """
    import pandas as pd

    def bound_index(index):
        """
        Ensure the index is within the range of 0 to 29.
        """
        return max(0, min(29, index))

    def generate_pair_dataframe(input_df, output_df, input_formula, output_formula):
        """
        Helper function to create one paired DataFrame for an input-output pair.
        Each cell contains a tuple (input_color, output_color).
        """
        pair_df = pd.DataFrame(index=input_df.index, columns=input_df.columns)

        for i in range(30):  # Rows
            for j in range(30):  # Columns
                # Determine pixel indices for input and output using the provided formulas
                input_pixel_index = (bound_index(input_formula(i, j)[0]), bound_index(input_formula(i, j)[1]))
                output_pixel_index = (bound_index(output_formula(i, j)[0]), bound_index(output_formula(i, j)[1]))

                # Extract colors for the determined indices
                input_color = input_df.at[input_pixel_index[0], input_pixel_index[1]]
                output_color = output_df.at[output_pixel_index[0], output_pixel_index[1]]

                # Assign color pairs as tuples
                pair_df.at[i, j] = (input_color, output_color)

        return pair_df

    # Create the paired DataFrames for input1-output1 and input2-output2
    input1_output1_df = generate_pair_dataframe(input_df1, output_df1, input_formula, output_formula)
    input2_output2_df = generate_pair_dataframe(input_df2, output_df2, input_formula, output_formula)

    return input1_output1_df, input2_output2_df



# Example formulas for pixel indices
input_formula = lambda i, j: (i, j)  # Use pixel (i, j) in input
output_formula = lambda i, j: (i, j)  # Use pixel (i, j+1) in output

# Call the function to create paired DataFrames
input1_output1_df, input2_output2_df = create_color_pair_dataframes(
    input_df1, input_df2, output_df1, output_df2, input_formula, output_formula
)


input1_output1_df


def recolor_grid(grid, coloring_scheme="default"):
            """
            Recolors the grid based on the chosen coloring scheme.
        
            Parameters:
            - grid: 2D list representing the grid to be recolored.
            - coloring_scheme: String specifying the recoloring scheme. 
              - "default": Recolors based on frequency of colors.
              - "orientation": Recolors based on the grid traversal starting from the bottom-left.
        
            Returns:
            - recolored_df: Pandas DataFrame of the recolored grid (30x30), with new color values.
            """
            import pandas as pd
            from collections import Counter
        
            if coloring_scheme == "default":
                # Default scheme: Based on frequency of colors
                # Flatten grid to count occurrences of colors
                flat_grid = [cell for row in grid for cell in row]
                color_counts = Counter(flat_grid)
                # Map colors to new values based on frequency
                sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x], reverse=True)
                recolor_map = {color: 11 + idx for idx, color in enumerate(sorted_colors)}
                # Apply recoloring
                recolored_grid = [[recolor_map[cell] for cell in row] for row in grid]
        
            elif coloring_scheme == "orientation":
                # Orientation scheme: Start from the bottom-left and assign colors row-by-row
                new_color = 11  # Start assigning colors from 11
                recolor_map = {}
                recolored_grid = []
        
                for i in range(len(grid) - 1, -1, -1):  # Iterate rows from bottom to top
                    recolored_row = []
                    for j in range(len(grid[i])):  # Iterate columns from left to right
                        color = grid[i][j]
                        # Assign a new color if the color hasn't been mapped yet
                        if color not in recolor_map:
                            recolor_map[color] = new_color
                            new_color += 1
                        recolored_row.append(recolor_map[color])
                    recolored_grid.insert(0, recolored_row)  # Insert row to form correct top-to-bottom orientation
        
            else:
                raise ValueError("Invalid coloring_scheme. Choose 'default' or 'orientation'.")
        
            # Convert to DataFrame
            recolored_df = pd.DataFrame(recolored_grid, index=range(29, -1, -1), columns=range(30))
            return recolored_df


def create_similarity_dataframe(df1, df2):
    """
    Creates a 30x30 DataFrame indicating similarity between two input DataFrames of color pairs.
    Each cell is 1 if the pair (input_color, output_color) matches across both DataFrames, else 0.

    Parameters:
    - df1: First DataFrame containing color pairs (30x30).
    - df2: Second DataFrame containing color pairs (30x30).

    Returns:
    - similarity_df: DataFrame of size 30x30 with 0s and 1s.
    """
    import pandas as pd

    # Initialize an empty 30x30 DataFrame for similarity
    similarity_df = pd.DataFrame(index=df1.index, columns=df1.columns)

    for i in range(30):  # Rows
        for j in range(30):  # Columns
            # Compare the pairs in the two DataFrames
            similarity_df.at[i, j] = 1 if df1.at[i, j] == df2.at[i, j] else 0

    return similarity_df


# Assume df1 and df2 are outputs from the previous function
similarity_df = create_similarity_dataframe(input1_output1_df, input2_output2_df)

# Display the similarity DataFrame
print("Similarity DataFrame:")
similarity_df


def generate_test_output_with_mapping(similarity_matrix, test_input_df, input_df, output_df):
    """
    Generates a new output image for the test input by applying color mappings derived from
    input-output grids wherever the similarity matrix has 1. Pixels with similarity 0
    are set to gray (color 12).

    Parameters:
    - similarity_matrix: DataFrame with similarity values (1s and 0s).
    - test_input_df: DataFrame of the test input grid (30x30).
    - input_df: DataFrame of the training input grid (30x30) used for color mapping.
    - output_df: DataFrame of the training output grid (30x30) used for color mapping.

    Returns:
    - raw_image_df: DataFrame of the resulting raw image with 30x30 pixels.
    """
    import pandas as pd

    # Generate the color mapping from input_df to output_df based on the similarity matrix
    color_mapping = {}
    for i in range(30):
        for j in range(30):
            # If similarity_matrix is 1, add the color pair from input to output
            if similarity_matrix.at[i, j] == 1:
                input_color = input_df.at[i, j]
                output_color = output_df.at[i, j]
                color_mapping[input_color] = output_color

    # Create the new grid for the test output
    new_output_grid = []
    for i in range(30):
        new_row = []
        for j in range(30):
            test_input_color = test_input_df.at[i, j]
            if similarity_matrix.at[i, j] == 1:
                # Apply the mapped output color for pixels with similarity 1
                new_row.append(color_mapping.get(test_input_color, 12))  # Map based on color_mapping
            else:
                # Set gray (color 12) for pixels with similarity 0
                new_row.append(12)
        new_output_grid.append(new_row)

    # Convert the grid to a DataFrame
    raw_image_df = pd.DataFrame(new_output_grid, index=range(29, -1, -1), columns=range(30))
    return raw_image_df


similarity_matrix 


# Example similarity matrix, test input, and training grids
similarity_matrix = create_similarity_dataframe(input1_output1_df, input2_output2_df)
input_df = input1_output1_df  # Use input1-output1 mapping as example
output_df = output_df1  # Training output grid

# Generate the resulting raw image
raw_image_df = generate_test_output_with_mapping(similarity_matrix, test_input_df, input_df, output_df)

# Plot the resulting image
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(raw_image_df, cmap="tab20", interpolation="nearest")
plt.title("Resulting Image with Gray Pixels for Similarity=0")
plt.axis("off")
plt.show()


raw_image_df[raw_image_df.values!=12]











raw_grid = dataset.revert_from_30x30(tweaked_inputs[0])


df = pd.DataFrame(tweaked_test_input, columns = [str(29-i) for i in range(30)]) 


df.reindex(index=df.index[::-1])























def convert_to_30x30(task_data):
    def resize_grid_to_30x30(grid):
        # Create a new 30x30 grid filled with "blank" pixels (value 10)
        new_grid = [[10] * 30 for _ in range(30)]
        
        # Original grid dimensions
        original_height = len(grid)
        original_width = len(grid[0])
        
        # Place the original grid in the bottom-left corner
        for i in range(original_height):
            for j in range(original_width):
                new_grid[30 - original_height + i][j] = grid[i][j]
        
        return new_grid

    # Apply resizing to all tasks
    for key in ['train', 'test']:
        for example in task_data.get(key, []):
            example['input'] = resize_grid_to_30x30(example['input'])
            if 'output' in example:  # Resize output only if it exists
                example['output'] = resize_grid_to_30x30(example['output'])
    
    return task_data


def convert_to_raw_format(task_data):
    def trim_grid_to_original(grid):
        # Determine rows and columns containing non-blank pixels
        non_blank_rows = [i for i, row in enumerate(grid) if any(cell != 10 for cell in row)]
        non_blank_cols = [j for j in range(len(grid[0])) if any(row[j] != 10 for row in grid)]
        
        # Trim the grid based on non-blank bounds
        min_row, max_row = min(non_blank_rows), max(non_blank_rows)
        min_col, max_col = min(non_blank_cols), max(non_blank_cols)
        
        return [row[min_col:max_col + 1] for row in grid[min_row:max_row + 1]]

    # Apply trimming to all tasks
    for key in ['train', 'test']:
        for example in task_data.get(key, []):
            example['input'] = trim_grid_to_original(example['input'])
            if 'output' in example:  # Trim output only if it exists
                example['output'] = trim_grid_to_original(example['output'])
    
    return task_data


# Sample task data
sample_task = {
    'train': [
        {'input': [[7, 9], [4, 3]],
         'output': [[7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3],
                    [9, 7, 9, 7, 9, 7],
                    [3, 4, 3, 4, 3, 4],
                    [7, 9, 7, 9, 7, 9],
                    [4, 3, 4, 3, 4, 3]]}
    ],
    'test': [{'input': [[3, 2], [7, 8]]}]
}

# Convert to 30x30
converted_task = convert_to_30x30(sample_task)
print("Converted to 30x30:", converted_task)


# Convert back to raw format
raw_task = convert_to_raw_format(converted_task)
print("Converted back to raw format:", raw_task)



































# defining a handful of basic primitives

def tophalf(grid):
    """ upper half """
    return grid[:len(grid) // 2]


def rot90(grid):
    """ clockwise rotation by 90 degrees """
    return list(zip(*grid[::-1]))


def hmirror(grid):
    """ mirroring along horizontal """
    return grid[::-1]


def compress(grid):
    """ removes frontiers """
    ri = [i for i, r in enumerate(grid) if len(set(r)) == 1]
    ci = [j for j, c in enumerate(zip(*grid)) if len(set(c)) == 1]
    return [[v for j, v in enumerate(r) if j not in ci] for i, r in enumerate(grid) if i not in ri]


def trim(grid):
    """ removes border """
    return [r[1:-1] for r in grid[1:-1]]


# defining the DSL as the set of the primitives

DSL_primitives = {tophalf, rot90, hmirror, compress, trim}
primitive_names = {p.__name__ for p in DSL_primitives}
print(f'DSL consists of {len(DSL_primitives)} primitives: {primitive_names}')


# the maximum composition depth to consider
MAX_DEPTH = 2

# construct the program strings of all programs expressible by composing at most MAX_DEPTH primitives

program_strings = []
for depth in range(1, MAX_DEPTH+1):
    primitive_tuples = itertools.product(*[primitive_names]*depth)
    for primitives in primitive_tuples:
        left_side = "".join([p + "(" for p in primitives])
        right_side = ')' * depth
        program_string = f'lambda grid: {left_side}grid{right_side}'
        program_strings.append(program_string)


# print some of the program strings
print(f'Space to search consists of {len(program_strings)} programs:\n')
print('\n'.join([*program_strings[:10], '...']))


# map program strings to programs
programs = {prog_str: eval(prog_str) for prog_str in program_strings}


def search_programs_for_tasks(train_challenges, programs, max_depth=4):
    """
    Searches for valid programs that solve the training challenges.
    Returns a dictionary of best guesses for each task.
    """
    guesses = {}
    
    # Iterate over all training tasks
    for key, task in tqdm.tqdm(train_challenges.items()):
        train_inputs = [example['input'] for example in task['train']]
        train_outputs = [example['output'] for example in task['train']]
        hypotheses = []

        # Iterate over all programs in the DSL
        for program_string, program in programs.items():
            try:
                # Check if the program solves all training examples
                if all([program(i) == o for i, o in zip(train_inputs, train_outputs)]):
                    hypotheses.append(program_string)
            except Exception:  # Skip programs that raise errors
                pass

        # Select the first valid program as the guess
        if len(hypotheses) > 0:
            print(f'Found {len(hypotheses)} candidate programs for task {key}!')
            guesses[key] = hypotheses[0]
    
    print(f'\nMade guesses for {len(guesses)} tasks')
    return guesses


def evaluate_predictions(guesses, train_challenges, train_solutions):
    """
    Evaluates predictions by checking if the guessed programs solve all test examples.
    Returns a dictionary of correctly solved tasks.
    """
    solved = {}

    # Iterate over all tasks with a valid guess
    for key, program_string in guesses.items():
        test_inputs = [example['input'] for example in train_challenges[key]['test']]
        program = eval(program_string)

        # Check if the program solves all test examples
        if all([program(i) == o for i, o in zip(test_inputs, train_solutions[key])]):
            solved[key] = program_string
    
    print(f'Predictions correct for {len(solved)}/{len(guesses)} tasks')
    return solved


def main(train_challenges, programs, train_solutions):
    """
    Main function to search for programs and evaluate their predictions.
    Incorporates the 30x30 grid resizing logic.
    """
    # Step 1: Search for programs that solve training tasks
    guesses = search_programs_for_tasks(train_challenges, programs)

    # Step 2: Evaluate the guesses against test solutions
    solved = evaluate_predictions(guesses, train_challenges, train_solutions)

    return guesses, solved


train_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
train_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'


with open(train_challenges_path) as fp:
    train_challenges = json.load(fp)
with open(train_solutions_path) as fp:
    train_solutions = json.load(fp)


guesses, solved = main(train_challenges, programs, train_solutions)


def visualize_training_with_guesses(task_data, guessed_outputs, title="Training Visualization with Guesses", figsize=(12, 15)):
    """
    Visualizes training data with six rows and two columns.
    Each column corresponds to a training example, showing:
    - Row 1: Raw training inputs
    - Row 2: Tweaked (30x30) training inputs
    - Row 3: Raw training outputs
    - Row 4: Tweaked (30x30) training outputs
    - Row 5: Raw guesses for the training outputs
    - Row 6: Tweaked (30x30) guesses for the training outputs
    """
    train_examples = task_data.get('train', [])
    assert len(train_examples) == len(guessed_outputs), "Each training example must have a corresponding guessed output."

    fig, axs = plt.subplots(6, len(train_examples), figsize=figsize)
    plt.suptitle(title, fontsize=16)

    for col, example in enumerate(train_examples):
        # Row 1: Raw training inputs
        axs[0, col].imshow(example['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0, col].set_title(f"Raw Input {col + 1}")
        axs[0, col].axis('off')

        # Row 2: Tweaked (30x30) training inputs
        tweaked_input = resize_to_30x30(example['input'])
        axs[1, col].imshow(tweaked_input, cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[1, col].set_title(f"Tweaked Input {col + 1}")
        axs[1, col].axis('off')

        # Row 3: Raw training outputs
        axs[2, col].imshow(example['output'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[2, col].set_title(f"Raw Output {col + 1}")
        axs[2, col].axis('off')

        # Row 4: Tweaked (30x30) training outputs
        tweaked_output = resize_to_30x30(example['output'])
        axs[3, col].imshow(tweaked_output, cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[3, col].set_title(f"Tweaked Output {col + 1}")
        axs[3, col].axis('off')

        # Row 5: Raw guesses for the training outputs
        axs[4, col].imshow(guessed_outputs[col], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[4, col].set_title(f"Raw Guess {col + 1}")
        axs[4, col].axis('off')

        # Row 6: Tweaked (30x30) guesses for the training outputs
        tweaked_guess = resize_to_30x30(guessed_outputs[col])
        axs[5, col].imshow(tweaked_guess, cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[5, col].set_title(f"Tweaked Guess {col + 1}")
        axs[5, col].axis('off')

    plt.tight_layout()
    plt.show()



# Example task data
training_task = {
    'train': [
        {'input': [[7, 9], [4, 3]], 'output': [[7, 9, 7, 9], [4, 3, 4, 3]]},
        {'input': [[8, 6], [6, 4]], 'output': [[8, 6, 8, 6], [6, 4, 6, 4]]},
    ]
}

# Example guessed outputs (raw format)
guessed_outputs = [
    [[7, 9, 7, 9], [4, 3, 4, 3]],
    [[8, 6, 8, 6], [6, 4, 6, 4]],
]

# Visualize the training data with guesses
visualize_training_with_guesses(training_task, guessed_outputs)







