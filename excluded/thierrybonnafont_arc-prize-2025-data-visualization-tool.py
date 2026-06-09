# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
from matplotlib.colors import ListedColormap

def visualize_arc_tasks(
    task_file_path: str,
    solutions_file_path: str = None,
    start_index: int = 0,
    end_index: int = None
):
    """
    Loads and visualizes ARC grids, with a single-line layout for inputs and outputs.
    All labels and comments are in English for community sharing.

    Args:
        task_file_path (str): Path to the JSON file of challenges.
        solutions_file_path (str): Path to the JSON file of solutions.
        start_index (int): Starting index for the slice of tasks to display.
        end_index (int): Ending index for the slice of tasks to display.
    """
    arc_colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#8B0000'
    ]
    cmap = ListedColormap(arc_colors)

    with open(task_file_path, 'r') as f:
        tasks_dict = json.load(f)
    
    solutions = None
    if solutions_file_path:
        with open(solutions_file_path, 'r') as f:
            solutions = json.load(f)

    tasks_to_process = list(tasks_dict.items())[start_index:end_index]

    for task_id, task_data in tasks_to_process:
        print(f"--- Task {task_id} ---")

        train_pairs = task_data['train']
        test_pairs = task_data['test']
        
        # Determine the total number of pairs to display
        num_train = len(train_pairs)
        num_test = len(test_pairs)
        num_pairs = num_train + num_test

        # Get solution for the current task
        task_solution = solutions.get(task_id) if solutions else None

        # Create a single figure with two rows for inputs and outputs
        fig, axes = plt.subplots(2, num_pairs, figsize=(4 * num_pairs, 8))

        # Handle a single pair edge case for proper indexing
        if num_pairs == 1:
            axes = np.array([axes]).reshape(2, 1)

        # Plot all input grids in the first row
        for i, pair in enumerate(train_pairs):
            ax = axes[0, i]
            ax.imshow(pair['input'], cmap=cmap, vmin=0, vmax=9)
            ax.set_title(f"Train {i+1} - Input")
            ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
            ax.set_xticks(np.arange(len(pair['input'][0])))
            ax.set_yticks(np.arange(len(pair['input'])))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)
        
        for i, pair in enumerate(test_pairs):
            ax = axes[0, num_train + i]
            ax.imshow(pair['input'], cmap=cmap, vmin=0, vmax=9)
            ax.set_title(f"Test {i+1} - Input")
            ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
            ax.set_xticks(np.arange(len(pair['input'][0])))
            ax.set_yticks(np.arange(len(pair['input'])))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)

        # Plot all output grids in the second row
        for i, pair in enumerate(train_pairs):
            ax = axes[1, i]
            ax.imshow(pair['output'], cmap=cmap, vmin=0, vmax=9)
            ax.set_title(f"Train {i+1} - Output")
            ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
            ax.set_xticks(np.arange(len(pair['output'][0])))
            ax.set_yticks(np.arange(len(pair['output'])))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)

        for i, pair in enumerate(test_pairs):
            ax = axes[1, num_train + i]
            
            # Display test solution if available
            if task_solution and i < len(task_solution):
                output_grid = task_solution[i]
                ax.imshow(output_grid, cmap=cmap, vmin=0, vmax=9)
                ax.set_title(f"Test {i+1} - Solution")
                ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
                ax.set_xticks(np.arange(len(output_grid[0])))
                ax.set_yticks(np.arange(len(output_grid)))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='both', which='both', length=0)
            else:
                ax.set_title(f"Test {i+1} - Solution (N/A)")
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage with corrected paths and slice
# visualize_arc_tasks(
#     'path/to/arc-agi_training_challenges.json',
#     'path/to/arc-agi_training_solutions.json',
#     start_index=0,
#     end_index=2
# )


visualize_arc_tasks(
     '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json',
     '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json',
     start_index=43,
     end_index=45
 )


import json

def find_multiple_test_tasks(file_path: str):
    """
    Finds and prints the IDs of tasks that have multiple test pairs.

    Args:
        file_path (str): The path to the ARC training challenges JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            tasks_dict = json.load(f)

        print("Searching for tasks with more than one test pair...")
        found_tasks = []
        for task_id, task_data in tasks_dict.items():
            if 'test' in task_data and len(task_data['test']) > 1:
                found_tasks.append(task_id)

        if found_tasks:
            print(f"Found {len(found_tasks)} tasks with multiple test pairs:")
            # Now, find their ordinal index for easier access
            all_task_ids = list(tasks_dict.keys())
            for task_id in found_tasks:
                ordinal_index = all_task_ids.index(task_id)
                print(f"- Task ID: {task_id}, Ordinal Index: {ordinal_index}")
        else:
            print("No tasks with multiple test pairs found in the file.")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print("Error: The file is not a valid JSON.")

# Example usage:
# find_multiple_test_tasks('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json')


find_multiple_test_tasks('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json')

