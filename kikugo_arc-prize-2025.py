import os
import json
import copy
import random
import math
import itertools
from collections import defaultdict, Counter, namedtuple
import traceback 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
%matplotlib inline

from scipy.ndimage import label, find_objects 

from tqdm.notebook import tqdm


# Define the ARC colormap and normalization
ARC_COLORMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF']
)
ARC_NORM = colors.Normalize(vmin=0, vmax=9) # vMax=9 is typically sufficient

# Function to display a single grid
def show_grid(grid, title=None, figsize=None):
    """Displays a single ARC grid using matplotlib."""
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    elif not isinstance(grid, list):
        return
    if not grid or not isinstance(grid[0], list) or len(grid[0]) == 0:
        return
    rows = len(grid)
    cols = len(grid[0])
    if not figsize:
        figsize = (max(1, cols * 0.5), max(1, rows * 0.5))
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap=ARC_COLORMAP, norm=ARC_NORM, interpolation='nearest')
    plt.xticks(np.arange(cols))
    plt.yticks(np.arange(rows))
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.tick_params(length=0) 
    if title:
        plt.title(title, fontsize=12)
    plt.show()

# Function to visualize a full task (train/test pairs)
def visualize_task(task_data, task_solutions=None, title="ARC Task Visualization", figsize=None):
    """Visualizes all train and test pairs for a given ARC task."""
    if not task_data: return
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    if not train_examples and not test_examples: return
    has_solution = task_solutions is not None
    num_train = len(train_examples)
    num_test = len(test_examples)
    cols = max(1, num_train + num_test)
    
    if figsize is None: 
        max_width, max_height = 1, 1
        all_ex = train_examples + test_examples
        for idx, ex in enumerate(all_ex):
             try: 
                 inp_h, inp_w = np.array(ex['input']).shape
                 max_width = max(max_width, inp_w)
                 max_height = max(max_height, inp_h)
                 if idx < num_train: 
                     out_h, out_w = np.array(ex['output']).shape
                     max_width = max(max_width, out_w)
                     max_height = max(max_height, out_h)
                 elif has_solution and idx - num_train < len(task_solutions):
                     sol_h, sol_w = np.array(task_solutions[idx - num_train]).shape
                     max_width = max(max_width, sol_w)
                     max_height = max(max_height, sol_h)
             except Exception: pass
        fig_width = max(8, cols * max(1.5, max_width * 0.3))
        fig_height = max(4, 2 * max(1.5, max_height * 0.3))
        figsize=(fig_width, fig_height)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, cols, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(title, fontsize=16, y=0.98)

    for idx, example in enumerate(train_examples):
        try:
            ax_in = fig.add_subplot(gs[0, idx])
            ax_out = fig.add_subplot(gs[1, idx])
            ax_in.imshow(example['input'], cmap=ARC_COLORMAP, norm=ARC_NORM, interpolation='nearest')
            ax_in.set_title(f"Train {idx+1}: Input")
            ax_in.set_xticks([]); ax_in.set_yticks([]); ax_in.grid(False)
            ax_out.imshow(example['output'], cmap=ARC_COLORMAP, norm=ARC_NORM, interpolation='nearest')
            ax_out.set_title(f"Train {idx+1}: Output")
            ax_out.set_xticks([]); ax_out.set_yticks([]); ax_out.grid(False)
        except Exception as e: print(f"Error plotting train pair {idx}: {e}")

    for idx, example in enumerate(test_examples):
        try:
            ax_in = fig.add_subplot(gs[0, num_train + idx])
            ax_out = fig.add_subplot(gs[1, num_train + idx])
            ax_in.imshow(example['input'], cmap=ARC_COLORMAP, norm=ARC_NORM, interpolation='nearest')
            ax_in.set_title(f"Test {idx+1}: Input")
            ax_in.set_xticks([]); ax_in.set_yticks([]); ax_in.grid(False)
            output_title = f"Test {idx+1}: Output"
            if has_solution and idx < len(task_solutions):
                try:
                    ax_out.imshow(task_solutions[idx], cmap=ARC_COLORMAP, norm=ARC_NORM, interpolation='nearest')
                    output_title += " (Solution)"
                except Exception as e:
                    ax_out.text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=10)
                    output_title += f": Error ({e})"
            else:
                ax_out.text(0.5, 0.5, '?', ha='center', va='center', fontsize=20)
                output_title += ": ?"
            ax_out.set_title(output_title)
            ax_out.set_xticks([]); ax_out.set_yticks([]); ax_out.grid(False)
        except Exception as e: print(f"Error plotting test pair {idx}: {e}")
        
    if cols == 0: # Handle empty task case
        ax = fig.add_subplot(gs[0,0])
        ax.text(0.5, 0.5, 'No data', ha='center', va='center'); ax.axis('off')
    plt.show()


class ARCDataset:
    """Handles loading and accessing ARC task data from JSON files."""
    def __init__(self, base_path):
        print("Initializing ARCDataset...")
        self.base_path = base_path
        self.train_path = f'{base_path}/arc-agi_training_challenges.json'
        self.train_solutions_path = f'{base_path}/arc-agi_training_solutions.json'
        self.eval_path = f'{base_path}/arc-agi_evaluation_challenges.json'
        self.eval_solutions_path = f'{base_path}/arc-agi_evaluation_solutions.json'
        self.test_path = f'{base_path}/arc-agi_test_challenges.json'

        self.train_data = self._load_json(self.train_path)
        self.train_solutions = self._load_json(self.train_solutions_path)
        self.eval_data = self._load_json(self.eval_path)
        self.eval_solutions = self._load_json(self.eval_solutions_path)
        self.test_data = self._load_json(self.test_path) 
        print("ARCDataset initialized.")
        self._print_loaded_counts()

    def _load_json(self, path):
        """Loads a JSON file."""
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}. Returning empty dict.")
            return {}
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return {}

    def _print_loaded_counts(self):
        """Prints the number of tasks loaded for each split."""
        print("\n--- Dataset Summary ---")
        print(f"Training challenges: {len(self.train_data)}")
        print(f"Evaluation challenges: {len(self.eval_data)}")
        print(f"Test challenges (submission target): {len(self.test_data)}")
        print("-----------------------\n")

    def get_task(self, task_id, split='train'):
        """Retrieves a specific task and its solution (if available)."""
        if split == 'train':
            task = self.train_data.get(task_id)
            solution = self.train_solutions.get(task_id)
        elif split == 'test': 
            task = self.test_data.get(task_id)
            solution = None 
        elif split == 'eval': 
            task = self.eval_data.get(task_id)
            solution = self.eval_solutions.get(task_id)
        else:
            raise ValueError("split must be 'train', 'test', or 'eval'")
        return task, solution

print("ARCDataset class defined.")

DATA_PATH = '/kaggle/input/arc-prize-2025'
dataset = ARCDataset(DATA_PATH)

# Visualize one task from training set ---
# task_id_to_show = '00576224' 
# vis_task_data, vis_task_sol = dataset.get_task(task_id_to_show, 'train')
# if vis_task_data:
#     print(f"\nVisualizing sample Training Task: {task_id_to_show}")
#     visualize_task(vis_task_data, vis_task_sol, title=f"Sample Task: {task_id_to_show}")


def analyze_dataset_split(task_dict, solutions_dict=None):
    """Analyzes tasks within a specific split (train, eval)."""
    stats = defaultdict(list) # Use defaultdict for easier appending
    print(f"Analyzing {len(task_dict)} tasks...")
    
    for task_id, task in tqdm(task_dict.items(), desc="Analyzing Tasks"):
        train_examples = task.get('train', [])
        test_examples = task.get('test', []) 

        stats['task_id'].append(task_id)
        stats['num_train_pairs'].append(len(train_examples))
        stats['num_test_pairs'].append(len(test_examples))

        task_colors = set()
        input_shapes_train = []
        output_shapes_train = []
        any_equal_train = False
        all_shapes_match_train = True if train_examples else None 

        for ex in train_examples:
            try:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                input_shapes_train.append(inp.shape)
                output_shapes_train.append(out.shape)
                task_colors.update(np.unique(inp))
                task_colors.update(np.unique(out))
                if np.array_equal(inp, out): any_equal_train = True
                if inp.shape != out.shape: all_shapes_match_train = False
            except Exception: pass # Ignore errors in malformed pairs

        stats['train_input_shapes'].append(input_shapes_train)
        stats['train_output_shapes'].append(output_shapes_train)
        stats['input_equals_output_train'].append(any_equal_train)
        stats['output_size_matches_input_train'].append(all_shapes_match_train)

        input_shapes_test = [np.array(ex['input']).shape for ex in test_examples if 'input' in ex]
        stats['test_input_shapes'].append(input_shapes_test)

        output_shapes_test = []
        if solutions_dict and task_id in solutions_dict:
            task_solutions = solutions_dict[task_id]
            # Check if solutions format matches test example count
            if isinstance(task_solutions, list) and len(task_solutions) == len(test_examples):
                 for sol in task_solutions:
                      try:
                         sol_np = np.array(sol)
                         output_shapes_test.append(sol_np.shape)
                         task_colors.update(np.unique(sol_np))
                      except Exception: output_shapes_test.append(None) # Append None if solution is invalid
            else: # Mismatch or invalid format
                output_shapes_test = [None] * len(test_examples) 
        stats['test_output_shapes'].append(output_shapes_test)

        stats['colors'].append(task_colors)
        stats['max_color_value'].append(max(task_colors) if task_colors else -1)

    print("Analysis complete.")
    return pd.DataFrame(stats)


def plot_grid_size_distributions(df, split_name="Training"):
    """Plots histograms of input and output grid dimensions from train pairs."""
    all_input_shapes = [shape for shapes_list in df['train_input_shapes'] for shape in shapes_list if len(shape) == 2]
    all_output_shapes = [shape for shapes_list in df['train_output_shapes'] for shape in shapes_list if len(shape) == 2]
    if not all_input_shapes and not all_output_shapes: return # Skip if no valid shapes
    
    input_heights = [s[0] for s in all_input_shapes]
    input_widths = [s[1] for s in all_input_shapes]
    output_heights = [s[0] for s in all_output_shapes]
    output_widths = [s[1] for s in all_output_shapes]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Grid Size Distributions ({split_name} Set - Train Pairs)', fontsize=14)
    axs[0, 0].hist(input_heights, bins=range(1, 32), align='left', rwidth=0.8); axs[0, 0].set_title('Input Heights'); axs[0, 0].set_ylabel('Frequency'); axs[0, 0].set_xticks(range(1, 32, 2))
    axs[0, 1].hist(input_widths, bins=range(1, 32), align='left', rwidth=0.8); axs[0, 1].set_title('Input Widths'); axs[0, 1].set_xticks(range(1, 32, 2))
    axs[1, 0].hist(output_heights, bins=range(1, 32), align='left', rwidth=0.8); axs[1, 0].set_title('Output Heights'); axs[1, 0].set_xlabel('Height'); axs[1, 0].set_ylabel('Frequency'); axs[1, 0].set_xticks(range(1, 32, 2))
    axs[1, 1].hist(output_widths, bins=range(1, 32), align='left', rwidth=0.8); axs[1, 1].set_title('Output Widths'); axs[1, 1].set_xlabel('Width'); axs[1, 1].set_xticks(range(1, 32, 2))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def plot_color_stats(df, split_name="Training"):
    """Plots distribution of number of unique colors and max color value."""
    num_colors = df['colors'].apply(len)
    max_color = df['max_color_value']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4)); fig.suptitle(f'Color Statistics ({split_name} Set)', fontsize=14)
    axs[0].hist(num_colors, bins=range(1, 13), align='left', rwidth=0.8); axs[0].set_title('Unique Colors per Task'); axs[0].set_xlabel('# Unique Colors'); axs[0].set_ylabel('Frequency'); axs[0].set_xticks(range(1, 12))
    axs[1].hist(max_color[max_color >= 0], bins=range(0, 11), align='left', rwidth=0.8); axs[1].set_title('Max Color Index per Task'); axs[1].set_xlabel('Max Color Index (0-9)'); axs[1].set_ylabel('Frequency'); axs[1].set_xticks(range(0, 10))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def plot_boolean_flags(df, split_name="Training"):
    """Plots counts for boolean flags like input==output in train pairs."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)); fig.suptitle(f'Task Properties ({split_name} Set - Train Pairs)', fontsize=14)
    # Handle potential NaN values if a task had no train pairs
    df['input_equals_output_train'].fillna(False).value_counts().sort_index().plot(kind='bar', ax=axs[0], rot=0); axs[0].set_title('Any Train Input == Output?'); axs[0].set_xticklabels(['False', 'True']); axs[0].set_ylabel('Number of Tasks')
    df['output_size_matches_input_train'].fillna(True).value_counts().sort_index().plot(kind='bar', ax=axs[1], rot=0); axs[1].set_title('All Train Output Shapes == Input Shapes?'); axs[1].set_xticklabels(['False', 'True'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]); plt.show()


# Analyze the training data
if dataset.train_data and dataset.train_solutions:
    print("Starting analysis of the Training dataset...")
    train_eda_df = analyze_dataset_split(dataset.train_data, dataset.train_solutions)
    print("\nTraining data analysis complete. Displaying results.")

    print("\n--- Head of Training EDA DataFrame ---")
    print(train_eda_df.head())
    print("-------------------------------------\n")

    plot_grid_size_distributions(train_eda_df, split_name="Training")
    plot_color_stats(train_eda_df, split_name="Training")
    plot_boolean_flags(train_eda_df, split_name="Training")


import numpy as np

def transform_identity(grid_np):
  """Returns a copy of the input grid."""
  return grid_np.copy()

def transform_rotate(grid_np, k=1):
  """Rotates the grid 90 degrees clockwise k times."""
  return np.rot90(grid_np, k=-k)

def transform_reflect(grid_np, axis=0):
  """
  Reflects the grid horizontally (axis=1) or vertically (axis=0).
  axis=0: Flip top-to-bottom (vertical reflection).
  axis=1: Flip left-to-right (horizontal reflection).
  """
  return np.flip(grid_np, axis=axis)

example_grid = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

print("Original Grid:")
show_grid(example_grid)

print("\nRotated 90 deg clockwise (k=1):")
show_grid(transform_rotate(example_grid, k=1))

print("\nRotated 180 deg clockwise (k=2):")
show_grid(transform_rotate(example_grid, k=2))

print("\nReflected Vertically (axis=0):")
show_grid(transform_reflect(example_grid, axis=0))

print("\nReflected Horizontally (axis=1):")
show_grid(transform_reflect(example_grid, axis=1))

print("\nBasic transformation functions defined.")


import numpy as np
import traceback

def solve_task(task_data):
    """
    Attempts to solve an ARC task by finding a simple consistent transformation.
    Tests Identity, Rotations (90, 180, 270), Reflections (H, V).
    """
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', []) 
    
    if not train_pairs:
        print(f"Task {task_data.get('task_id', 'Unknown')} has no training pairs. Using fallback.")
        predictions = []
        for test_pair in test_pairs:
             input_grid_np = np.array(test_pair['input'])
             pred_1 = input_grid_np.tolist() 
             pred_2 = np.zeros((3,3), dtype=int).tolist() 
             predictions.append({"attempt_1": pred_1, "attempt_2": pred_2})
        return predictions

    # Define the transformations to test
    transformations_to_test = [
        ('identity', transform_identity, {}),
        ('rotate_90', transform_rotate, {'k': 1}),
        ('rotate_180', transform_rotate, {'k': 2}),
        ('rotate_270', transform_rotate, {'k': 3}),
        ('reflect_v', transform_reflect, {'axis': 0}),
        ('reflect_h', transform_reflect, {'axis': 1}),
        # Add more transformations here later
        # e.g., ('recolor', transform_recolor, {'map': ...})
        # e.g., ('crop', transform_crop, {'bbox': ...})
    ]

    consistent_rules = []

    # Test each transformation
    for rule_name, transform_func, args in transformations_to_test:
        is_consistent = True
        for pair in train_pairs:
            input_np = np.array(pair['input'])
            output_np = np.array(pair['output'])
            
            try:
                predicted_output_np = transform_func(input_np, **args)
                
                if not np.array_equal(predicted_output_np, output_np):
                    is_consistent = False
                    break 
            except Exception as e:
                is_consistent = False
                break

        if is_consistent:
            consistent_rules.append({'name': rule_name, 'func': transform_func, 'args': args})

    
    predictions = []
    selected_rule = None
    if consistent_rules:
        selected_rule = consistent_rules[0] 
        # print(f"Selected rule: {selected_rule['name']}") #for debugging
    # else:
        # print("No simple geometric rule was consistent for all train pairs.") # Optional

    for test_pair in test_pairs:
        input_grid_np = np.array(test_pair['input'])
        pred_1_list = None
        pred_2_list = None

        # Attempt 1: Apply selected rule if found
        if selected_rule:
            try:
                pred_1_np = selected_rule['func'](input_grid_np, **selected_rule['args'])
                if pred_1_np.ndim == 2 and pred_1_np.shape[0] > 0 and pred_1_np.shape[1] > 0:
                     pred_1_list = pred_1_np.tolist()
                else:
                     print(f"Warning: Rule '{selected_rule['name']}' produced invalid shape {pred_1_np.shape}. Falling back.")
                     pred_1_list = input_grid_np.tolist()
            except Exception as e:
                print(f"Error applying rule '{selected_rule['name']}' to test input: {e}")
                # print(traceback.format_exc()) # More detailed error
                pred_1_list = input_grid_np.tolist() 
        else:
            pred_1_list = input_grid_np.tolist()

        # Attempt 2: Simple fallback (copy input) - could be more sophisticated
        # If attempt 1 was already copy input, maybe try zeros?
        if selected_rule and selected_rule['name'] == 'identity':
             pred_2_list = np.zeros_like(input_grid_np, dtype=int).tolist() 
        elif selected_rule and not np.array_equal(np.array(pred_1_list), input_grid_np):
             pred_2_list = input_grid_np.tolist() # If pred_1 wasn't copy, try copy for pred_2
        else:
             pred_2_list = np.zeros((3,3), dtype=int).tolist() 

        # Final check for None and ensure list format
        if pred_1_list is None: pred_1_list = [[0]]
        if pred_2_list is None: pred_2_list = [[0]]
        
        predictions.append({
            "attempt_1": pred_1_list,
            "attempt_2": pred_2_list
        })

    # Final validation (same as before)
    if len(predictions) != len(test_pairs):
         print(f"Error: Prediction count mismatch! ({len(predictions)} vs {len(test_pairs)})")
         predictions = []
         for _ in test_pairs:
              dummy_pred = [[0]]
              predictions.append({"attempt_1": dummy_pred, "attempt_2": dummy_pred})

    return predictions

print("Solver function `solve_task` updated with basic hypothesis testing.")

# --- Test the updated solver on a known simple task ---
simple_rot_task_id = '1190e5a7' # This task *might* be simple rotation/reflection
print(f"\nTesting updated solver on task: {simple_rot_task_id}")
task_data, _ = dataset.get_task(simple_rot_task_id, split='train') # Need only train data to find rule

if task_data:
    visualize_task(task_data, dataset.train_solutions.get(simple_rot_task_id), title=f"Task {simple_rot_task_id} (Train Set)")
    
    print("\nRunning solver...")
    predictions_for_task = solve_task(task_data)
    
    # Show the predicted outputs for the test input of this task
    if task_data['test']:
        test_input_grid = task_data['test'][0]['input']
        print("\nTest Input:")
        show_grid(test_input_grid)
        
        print("\nPredicted Output (Attempt 1):")
        show_grid(predictions_for_task[0]['attempt_1'])
        
        print("\nPredicted Output (Attempt 2):")
        show_grid(predictions_for_task[0]['attempt_2'])
        
        # Compare with actual solution if available (from training solutions file)
        actual_solution = dataset.train_solutions.get(simple_rot_task_id)
        if actual_solution:
            print("\nActual Test Output (from training_solutions):")
            show_grid(actual_solution[0])
            
            # Check if attempt 1 was correct
            if np.array_equal(predictions_for_task[0]['attempt_1'], actual_solution[0]):
                print("\n---> Attempt 1 prediction MATCHES the solution! <---")
            else:
                print("\n---> Attempt 1 prediction does NOT match the solution. <---")
    else:
        print("Task has no test cases in the provided data.")
        
else:
    print(f"Task {simple_rot_task_id} not found in dataset.")


import numpy as np

# --- Add a new transformation function ---

def transform_swap_colors(grid_np, c1, c2):
  """Swaps all occurrences of color c1 with c2 and vice-versa."""
  new_grid = grid_np.copy()
  mask_c1 = (new_grid == c1)
  mask_c2 = (new_grid == c2)
  new_grid[mask_c1] = c2
  new_grid[mask_c2] = c1
  return new_grid

# --- Update solve_task to include color swaps ---

def solve_task(task_data):
    """
    Attempts to solve an ARC task. Tests basic geometry and simple color swaps.
    """
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', []) 
    
    if not train_pairs:
        # (Keep the same fallback logic for no-training-pair tasks)
        predictions = []
        for test_pair in test_pairs:
             input_grid_np = np.array(test_pair['input'])
             pred_1 = input_grid_np.tolist() 
             pred_2 = np.zeros((3,3), dtype=int).tolist() 
             predictions.append({"attempt_1": pred_1, "attempt_2": pred_2})
        return predictions

    # --- Define the transformations to test ---
    transformations_to_test = [
        ('identity', transform_identity, {}),
        ('rotate_90', transform_rotate, {'k': 1}),
        ('rotate_180', transform_rotate, {'k': 2}),
        ('rotate_270', transform_rotate, {'k': 3}),
        ('reflect_v', transform_reflect, {'axis': 0}),
        ('reflect_h', transform_reflect, {'axis': 1}),
    ]
    
    # --- Dynamically add color swap rules ---
    # Find all unique colors present in the training inputs/outputs
    all_colors = set()
    for pair in train_pairs:
        all_colors.update(np.unique(pair['input']))
        all_colors.update(np.unique(pair['output']))
    
    # Create rules for swapping every pair of distinct colors found
    import itertools
    color_list = sorted(list(all_colors))
    for c1, c2 in itertools.combinations(color_list, 2):
         rule_name = f'swap_{c1}_{c2}'
         transformations_to_test.append(
             (rule_name, transform_swap_colors, {'c1': c1, 'c2': c2})
         )
         
    # --- (Rest of the function remains the same) ---
    # --- Hypothesis testing loop ---
    consistent_rules = []
    for rule_name, transform_func, args in transformations_to_test:
        is_consistent = True
        for pair in train_pairs:
            input_np = np.array(pair['input'])
            output_np = np.array(pair['output'])
            try:
                predicted_output_np = transform_func(input_np, **args)
                if not np.array_equal(predicted_output_np, output_np):
                    is_consistent = False
                    break 
            except Exception as e:
                is_consistent = False
                break
        if is_consistent:
            consistent_rules.append({'name': rule_name, 'func': transform_func, 'args': args})
            # print(f"Found consistent rule: {rule_name}") # Debugging

    # --- Rule Selection and Application ---
    predictions = []
    selected_rule = None
    if consistent_rules:
        selected_rule = consistent_rules[0] # Still pick the first one
        # print(f"Selected rule: {selected_rule['name']}") # Debugging

    for test_pair in test_pairs:
        input_grid_np = np.array(test_pair['input'])
        pred_1_list = None
        pred_2_list = None

        if selected_rule:
            try:
                pred_1_np = selected_rule['func'](input_grid_np, **selected_rule['args'])
                if pred_1_np.ndim == 2 and pred_1_np.shape[0] > 0 and pred_1_np.shape[1] > 0:
                     pred_1_list = pred_1_np.tolist()
                else:
                     # print(f"Warning: Rule '{selected_rule['name']}' produced invalid shape {pred_1_np.shape}. Falling back.")
                     pred_1_list = input_grid_np.tolist() 
            except Exception as e:
                # print(f"Error applying rule '{selected_rule['name']}' to test input: {e}")
                pred_1_list = input_grid_np.tolist() 
        else:
            pred_1_list = input_grid_np.tolist()

        # Attempt 2 Fallback Logic (same as before for now)
        if selected_rule and selected_rule['name'] == 'identity':
             pred_2_list = np.zeros_like(input_grid_np, dtype=int).tolist() 
        elif selected_rule and not np.array_equal(np.array(pred_1_list), input_grid_np):
             pred_2_list = input_grid_np.tolist() 
        else:
             pred_2_list = np.zeros((3,3), dtype=int).tolist()

        if pred_1_list is None: pred_1_list = [[0]]
        if pred_2_list is None: pred_2_list = [[0]]
        
        predictions.append({
            "attempt_1": pred_1_list,
            "attempt_2": pred_2_list
        })

    if len(predictions) != len(test_pairs):
         # (Keep validation logic)
         print(f"Error: Prediction count mismatch! ({len(predictions)} vs {len(test_pairs)})")
         predictions = []
         for _ in test_pairs:
              dummy_pred = [[0]]
              predictions.append({"attempt_1": dummy_pred, "attempt_2": dummy_pred})

    return predictions

print("`transform_swap_colors` defined and `solve_task` updated to test color swaps.")

# --- Test again on a task that *might* involve color changes ---
# Task '1f876c06' often involves color mapping
color_task_id = '1f876c06' 
print(f"\nTesting updated solver on task: {color_task_id}")
task_data, _ = dataset.get_task(color_task_id, split='train') 

if task_data:
    visualize_task(task_data, dataset.train_solutions.get(color_task_id), title=f"Task {color_task_id} (Train Set)")
    
    print("\nRunning solver...")
    predictions_for_task = solve_task(task_data) # This will now test color swaps too
    
    if task_data['test']:
        test_input_grid = task_data['test'][0]['input']
        print("\nTest Input:")
        show_grid(test_input_grid)
        
        print("\nPredicted Output (Attempt 1):")
        show_grid(predictions_for_task[0]['attempt_1'])
        
        print("\nPredicted Output (Attempt 2):")
        show_grid(predictions_for_task[0]['attempt_2'])
        
        actual_solution = dataset.train_solutions.get(color_task_id)
        if actual_solution:
            print("\nActual Test Output (from training_solutions):")
            show_grid(actual_solution[0])
            if np.array_equal(predictions_for_task[0]['attempt_1'], actual_solution[0]):
                print("\n---> Attempt 1 prediction MATCHES the solution! <---")
            else:
                print("\n---> Attempt 1 prediction does NOT match the solution. <---")
    else:
        print("Task has no test cases in the provided data.")
        
else:
    print(f"Task {color_task_id} not found in dataset.")


import numpy as np
import itertools # Need this again

# --- Add a new transformation function ---

def transform_recolor_single(grid_np, color_from, color_to):
  """Changes all occurrences of color_from to color_to."""
  new_grid = grid_np.copy()
  new_grid[new_grid == color_from] = color_to
  return new_grid

# --- Update solve_task to include single color recolors ---

def solve_task(task_data):
    """
    Attempts to solve an ARC task. Tests basic geometry, color swaps, 
    and single color recoloring.
    """
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', []) 
    
    if not train_pairs:
        # (Keep fallback logic)
        predictions = []
        for test_pair in test_pairs:
             input_grid_np = np.array(test_pair['input'])
             pred_1 = input_grid_np.tolist() 
             pred_2 = np.zeros((3,3), dtype=int).tolist() 
             predictions.append({"attempt_1": pred_1, "attempt_2": pred_2})
        return predictions

    # --- Define the base transformations ---
    transformations_to_test = [
        ('identity', transform_identity, {}),
        ('rotate_90', transform_rotate, {'k': 1}),
        ('rotate_180', transform_rotate, {'k': 2}),
        ('rotate_270', transform_rotate, {'k': 3}),
        ('reflect_v', transform_reflect, {'axis': 0}),
        ('reflect_h', transform_reflect, {'axis': 1}),
    ]
    
    # --- Dynamically add color swap AND single recolor rules ---
    all_colors = set()
    for pair in train_pairs:
        all_colors.update(np.unique(pair['input']))
        all_colors.update(np.unique(pair['output']))
    
    color_list = sorted(list(all_colors))
    
    # Add color swaps (as before)
    for c1, c2 in itertools.combinations(color_list, 2):
         rule_name = f'swap_{c1}_{c2}'
         transformations_to_test.append(
             (rule_name, transform_swap_colors, {'c1': c1, 'c2': c2})
         )
         
    # Add single color recolors
    # Try changing each color 'c_from' to each other color 'c_to'
    for c_from in color_list:
        for c_to in color_list:
            if c_from != c_to: # No point changing a color to itself
                 rule_name = f'recolor_{c_from}_to_{c_to}'
                 transformations_to_test.append(
                     (rule_name, transform_recolor_single, {'color_from': c_from, 'color_to': c_to})
                 )

    # --- (Rest of the function remains the same: Hypothesis testing, Rule Selection, Application) ---
    consistent_rules = []
    # print(f"Total rules to test: {len(transformations_to_test)}") # Debug: See how many rules we have
    for rule_name, transform_func, args in transformations_to_test:
        is_consistent = True
        for pair in train_pairs:
            input_np = np.array(pair['input'])
            output_np = np.array(pair['output'])
            try:
                predicted_output_np = transform_func(input_np, **args)
                if not np.array_equal(predicted_output_np, output_np):
                    is_consistent = False
                    break 
            except Exception as e:
                is_consistent = False
                break
        if is_consistent:
            consistent_rules.append({'name': rule_name, 'func': transform_func, 'args': args})
            # print(f"Found consistent rule: {rule_name}") # Debugging

    predictions = []
    selected_rule = None
    if consistent_rules:
        selected_rule = consistent_rules[0] 
        # print(f"Selected rule: {selected_rule['name']}") # Debugging

    for test_pair in test_pairs:
        input_grid_np = np.array(test_pair['input'])
        pred_1_list = None
        pred_2_list = None

        if selected_rule:
            try:
                pred_1_np = selected_rule['func'](input_grid_np, **selected_rule['args'])
                if pred_1_np.ndim == 2 and pred_1_np.shape[0] > 0 and pred_1_np.shape[1] > 0:
                     pred_1_list = pred_1_np.tolist()
                else:
                     pred_1_list = input_grid_np.tolist() 
            except Exception as e:
                pred_1_list = input_grid_np.tolist() 
        else:
            pred_1_list = input_grid_np.tolist()

        # Attempt 2 Fallback Logic (same)
        if selected_rule and selected_rule['name'] == 'identity':
             pred_2_list = np.zeros_like(input_grid_np, dtype=int).tolist() 
        elif selected_rule and not np.array_equal(np.array(pred_1_list), input_grid_np):
             pred_2_list = input_grid_np.tolist() 
        else:
             pred_2_list = np.zeros((3,3), dtype=int).tolist()

        if pred_1_list is None: pred_1_list = [[0]]
        if pred_2_list is None: pred_2_list = [[0]]
        
        predictions.append({
            "attempt_1": pred_1_list,
            "attempt_2": pred_2_list
        })

    if len(predictions) != len(test_pairs):
         # (Keep validation logic)
         print(f"Error: Prediction count mismatch! ({len(predictions)} vs {len(test_pairs)})")
         predictions = []
         for _ in test_pairs:
              dummy_pred = [[0]]
              predictions.append({"attempt_1": dummy_pred, "attempt_2": dummy_pred})

    return predictions


print("`transform_recolor_single` defined and `solve_task` updated.")

# --- Test on a task that might involve single recoloring ---
# Task '0a938d79' seems to change blue (1) shapes to red (2)
recolor_task_id = '0a938d79' 
print(f"\nTesting updated solver on task: {recolor_task_id}")
task_data, _ = dataset.get_task(recolor_task_id, split='train') 

if task_data:
    visualize_task(task_data, dataset.train_solutions.get(recolor_task_id), title=f"Task {recolor_task_id} (Train Set)")
    
    print("\nRunning solver...")
    predictions_for_task = solve_task(task_data) 
    
    if task_data['test']:
        test_input_grid = task_data['test'][0]['input']
        print("\nTest Input:")
        show_grid(test_input_grid)
        
        print("\nPredicted Output (Attempt 1):")
        show_grid(predictions_for_task[0]['attempt_1'])
        
        print("\nPredicted Output (Attempt 2):")
        show_grid(predictions_for_task[0]['attempt_2'])
        
        actual_solution = dataset.train_solutions.get(recolor_task_id)
        if actual_solution:
            print("\nActual Test Output (from training_solutions):")
            show_grid(actual_solution[0])
            if np.array_equal(predictions_for_task[0]['attempt_1'], actual_solution[0]):
                print("\n---> Attempt 1 prediction MATCHES the solution! <---")
            else:
                print("\n---> Attempt 1 prediction does NOT match the solution. <---")
    else:
        print("Task has no test cases in the provided data.")
        
else:
    print(f"Task {recolor_task_id} not found in dataset.")


import numpy as np
from scipy.ndimage import label, find_objects # For finding connected components
from collections import namedtuple

# Define a simple structure to hold object information
Object = namedtuple('Object', ['mask', 'color', 'bbox', 'pixel_coords'])

def find_grid_objects(grid_np, ignore_color=0, connectivity=1):
    """
    Finds distinct objects (connected components) in a grid for each color.
    
    Args:
        grid_np (np.array): The input grid.
        ignore_color (int): The color to treat as background (usually 0).
        connectivity (1 or 2): Connectivity for labeling (1: 4-way, 2: 8-way).

    Returns:
        list[Object]: A list of namedtuples, each describing a found object.
                       Returns empty list if no objects found or grid is empty.
    """
    objects = []
    if grid_np.size == 0:
        return objects
        
    unique_colors = np.unique(grid_np)
    
    for color in unique_colors:
        if color == ignore_color:
            continue
            
        # Create a binary mask for the current color
        binary_mask = (grid_np == color)
        
        # Label connected components for this color
        # structure defines connectivity:
        # [[0,1,0], [1,1,1], [0,1,0]] for 4-way
        # [[1,1,1], [1,1,1], [1,1,1]] for 8-way
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]]) if connectivity == 1 else np.array([[1,1,1],[1,1,1],[1,1,1]])
        labeled_mask, num_labels = label(binary_mask, structure=structure)
        
        if num_labels > 0:
            # Find the bounding boxes for each labeled component
            bboxes = find_objects(labeled_mask) # Returns list of slice tuples
            
            # Extract info for each distinct object of this color
            for i in range(num_labels):
                obj_label = i + 1
                obj_mask = (labeled_mask == obj_label)
                obj_bbox_slices = bboxes[i] # Slices for (row, col)
                
                # Get pixel coordinates relative to the full grid
                obj_coords = np.argwhere(obj_mask) # List of [row, col] pairs
                
                # Create the Object tuple
                obj = Object(mask=obj_mask, 
                             color=color, 
                             bbox=obj_bbox_slices, 
                             pixel_coords=obj_coords)
                objects.append(obj)
                
    return objects

# --- Example Usage ---
example_grid_objects = np.array([
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 2],
    [0, 0, 3, 3, 2],
    [0, 0, 3, 3, 0],
    [1, 0, 0, 0, 0] 
])

print("Example Grid for Object Detection:")
show_grid(example_grid_objects)

print("\nFinding objects (ignoring color 0, 8-way connectivity)...")
found_objects = find_grid_objects(example_grid_objects, ignore_color=0, connectivity=2)

print(f"\nFound {len(found_objects)} objects:")
for i, obj in enumerate(found_objects):
    print(f"  Object {i+1}:")
    print(f"    Color: {obj.color}")
    # Bbox is tuple of slices: (slice(row_start, row_stop), slice(col_start, col_stop))
    print(f"    BBox (slices): {obj.bbox}") 
    bbox_rows, bbox_cols = obj.bbox
    print(f"    BBox (coords): Rows {bbox_rows.start}-{bbox_rows.stop-1}, Cols {bbox_cols.start}-{bbox_cols.stop-1}")
    print(f"    Pixel Count: {len(obj.pixel_coords)}")
    # print(f"    Mask Shape: {obj.mask.shape}") # Mask is full grid size
    # print(f"    Pixel Coords (first 5): {obj.pixel_coords[:5].tolist()}") # Show some coords

    # Visualize the object's mask
    # obj_vis_grid = np.zeros_like(example_grid_objects)
    # obj_vis_grid[obj.mask] = obj.color
    # show_grid(obj_vis_grid, title=f"Object {i+1} (Color {obj.color}) Mask")


print("\nObject detection function `find_grid_objects` defined.")


import numpy as np
# We need find_grid_objects from the previous cell

# --- Add the new transformation function ---

def transform_keep_largest_objects(grid_np, ignore_color=0, connectivity=1):
    """
    Finds all objects, identifies the largest ones (by pixel count),
    and returns a new grid containing only those largest objects.
    If multiple objects share the max size, all are kept.
    """
    objects = find_grid_objects(grid_np, ignore_color=ignore_color, connectivity=connectivity)
    
    if not objects:
        # If no objects found (e.g., empty grid or only ignore_color), return empty grid of same shape
        return np.full_like(grid_np, ignore_color) 
        
    # Calculate sizes and find the maximum size
    object_sizes = [len(obj.pixel_coords) for obj in objects]
    max_size = max(object_sizes)
    
    # Create a new grid, initially filled with the ignore_color
    output_grid = np.full_like(grid_np, ignore_color)
    
    # Add back only the largest objects
    for obj, size in zip(objects, object_sizes):
        if size == max_size:
            # Place the object's pixels back onto the output grid
            # obj.mask is a boolean array of the same shape as grid_np
            output_grid[obj.mask] = obj.color 
            
    return output_grid

# --- Update solve_task to include this new transformation ---

def solve_task(task_data):
    """
    Attempts to solve an ARC task. Tests basic geometry, color changes,
    and keeping the largest object(s).
    """
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', []) 
    
    if not train_pairs:
        # (Keep fallback logic)
        predictions = []
        for test_pair in test_pairs:
             input_grid_np = np.array(test_pair['input'])
             pred_1 = input_grid_np.tolist() 
             pred_2 = np.zeros((3,3), dtype=int).tolist() 
             predictions.append({"attempt_1": pred_1, "attempt_2": pred_2})
        return predictions

    # --- Define the base transformations ---
    transformations_to_test = [
        ('identity', transform_identity, {}),
        ('rotate_90', transform_rotate, {'k': 1}),
        ('rotate_180', transform_rotate, {'k': 2}),
        ('rotate_270', transform_rotate, {'k': 3}),
        ('reflect_v', transform_reflect, {'axis': 0}),
        ('reflect_h', transform_reflect, {'axis': 1}),
        # Add the new object-based rule
        # We might want to test both connectivity options
        ('keep_largest_obj_conn1', transform_keep_largest_objects, {'ignore_color': 0, 'connectivity': 1}),
        ('keep_largest_obj_conn2', transform_keep_largest_objects, {'ignore_color': 0, 'connectivity': 2}),
    ]
    
    # --- Dynamically add color rules (same as before) ---
    all_colors = set()
    for pair in train_pairs:
        all_colors.update(np.unique(pair['input']))
        all_colors.update(np.unique(pair['output']))
    color_list = sorted(list(all_colors))
    # Add color swaps
    for c1, c2 in itertools.combinations(color_list, 2):
         rule_name = f'swap_{c1}_{c2}'
         transformations_to_test.append(
             (rule_name, transform_swap_colors, {'c1': c1, 'c2': c2})
         )
    # Add single color recolors
    for c_from in color_list:
        for c_to in color_list:
            if c_from != c_to:
                 rule_name = f'recolor_{c_from}_to_{c_to}'
                 transformations_to_test.append(
                     (rule_name, transform_recolor_single, {'color_from': c_from, 'color_to': c_to})
                 )

    # --- (Hypothesis testing, Rule Selection, Application - remain the same structure) ---
    consistent_rules = []
    # print(f"Total rules to test: {len(transformations_to_test)}") # Debug
    for rule_name, transform_func, args in transformations_to_test:
        is_consistent = True
        for pair in train_pairs:
            input_np = np.array(pair['input'])
            output_np = np.array(pair['output'])
            try:
                # Important: Make sure output shapes match if the rule implies it
                # For 'keep_largest', the shape *should* match the input shape.
                predicted_output_np = transform_func(input_np, **args)
                if not np.array_equal(predicted_output_np, output_np):
                    is_consistent = False
                    break 
            except Exception as e:
                # print(f"Rule {rule_name} failed during train check: {e}") # Debugging
                is_consistent = False
                break
        if is_consistent:
            consistent_rules.append({'name': rule_name, 'func': transform_func, 'args': args})
            # print(f"Found consistent rule: {rule_name}") # Debugging

    predictions = []
    selected_rule = None
    if consistent_rules:
        # Simple selection: Prioritize non-identity rules if available?
        # Or just stick with the first one found for now.
        selected_rule = consistent_rules[0] 
        # print(f"Selected rule: {selected_rule['name']}") # Debugging

    for test_pair in test_pairs:
        input_grid_np = np.array(test_pair['input'])
        pred_1_list = None
        pred_2_list = None

        if selected_rule:
            try:
                pred_1_np = selected_rule['func'](input_grid_np, **selected_rule['args'])
                # Basic validation
                if pred_1_np.ndim == 2 and pred_1_np.shape[0] > 0 and pred_1_np.shape[1] > 0:
                     # Check if output shape matches input shape, as expected for this rule
                     if pred_1_np.shape == input_grid_np.shape:
                         pred_1_list = pred_1_np.tolist()
                     else:
                         # print(f"Warning: Rule '{selected_rule['name']}' output shape {pred_1_np.shape} != input shape {input_grid_np.shape}. Falling back.")
                         pred_1_list = input_grid_np.tolist() # Fallback: copy input if shape changed unexpectedly
                else:
                     pred_1_list = input_grid_np.tolist() 
            except Exception as e:
                # print(f"Error applying rule '{selected_rule['name']}' to test input: {e}")
                pred_1_list = input_grid_np.tolist() 
        else:
            pred_1_list = input_grid_np.tolist()

        # Attempt 2 Fallback Logic (same)
        if selected_rule and selected_rule['name'] == 'identity':
             pred_2_list = np.zeros_like(input_grid_np, dtype=int).tolist() 
        elif selected_rule and pred_1_list != input_grid_np.tolist(): # Check if attempt 1 was different from input
             pred_2_list = input_grid_np.tolist() # Try copy input for attempt 2
        else: # No rule found, or rule resulted in copy input anyway
             pred_2_list = np.zeros((3,3), dtype=int).tolist() 

        if pred_1_list is None: pred_1_list = [[0]]
        if pred_2_list is None: pred_2_list = [[0]]
        
        predictions.append({
            "attempt_1": pred_1_list,
            "attempt_2": pred_2_list
        })

    if len(predictions) != len(test_pairs):
         # (Keep validation logic)
         print(f"Error: Prediction count mismatch! ({len(predictions)} vs {len(test_pairs)})")
         predictions = []
         for _ in test_pairs:
              dummy_pred = [[0]]
              predictions.append({"attempt_1": dummy_pred, "attempt_2": dummy_pred})

    return predictions


print("`transform_keep_largest_objects` defined and `solve_task` updated.")

# --- Test on a task that might involve keeping the largest object ---
# Task '25d8a9c8' 
largest_obj_task_id = '25d8a9c8' 
print(f"\nTesting updated solver on task: {largest_obj_task_id}")
task_data, _ = dataset.get_task(largest_obj_task_id, split='train') 

if task_data:
    visualize_task(task_data, dataset.train_solutions.get(largest_obj_task_id), title=f"Task {largest_obj_task_id} (Train Set)")
    
    print("\nRunning solver...")
    predictions_for_task = solve_task(task_data) # Solver now includes 'keep_largest'
    
    if task_data['test']:
        test_input_grid = task_data['test'][0]['input']
        print("\nTest Input:")
        show_grid(test_input_grid)
        
        print("\nPredicted Output (Attempt 1):")
        show_grid(predictions_for_task[0]['attempt_1'])
        
        print("\nPredicted Output (Attempt 2):")
        show_grid(predictions_for_task[0]['attempt_2'])
        
        actual_solution = dataset.train_solutions.get(largest_obj_task_id)
        if actual_solution:
            print("\nActual Test Output (from training_solutions):")
            show_grid(actual_solution[0])
            if np.array_equal(predictions_for_task[0]['attempt_1'], actual_solution[0]):
                print("\n---> Attempt 1 prediction MATCHES the solution! <---")
            else:
                print("\n---> Attempt 1 prediction does NOT match the solution. <---")
    else:
        print("Task has no test cases in the provided data.")
        
else:
    print(f"Task {largest_obj_task_id} not found in dataset.")


# --- Basic Geometric Transformations ---
def transform_identity(grid_np, **kwargs):
  return grid_np.copy()

def transform_rotate(grid_np, k=1, **kwargs):
  return np.rot90(grid_np, k=-k) # Clockwise rotation

def transform_reflect(grid_np, axis=0, **kwargs):
  return np.flip(grid_np, axis=axis) # axis 0: vertical, axis 1: horizontal

# --- Color Transformations ---
def transform_swap_colors(grid_np, c1, c2, **kwargs):
  new_grid = grid_np.copy()
  mask_c1 = (new_grid == c1)
  mask_c2 = (new_grid == c2)
  new_grid[mask_c1] = c2
  new_grid[mask_c2] = c1
  return new_grid

def transform_recolor_single(grid_np, color_from, color_to, **kwargs):
  new_grid = grid_np.copy()
  new_grid[new_grid == color_from] = color_to
  return new_grid

# --- Object Detection Helper ---
Object = namedtuple('Object', ['mask', 'color', 'bbox', 'pixel_coords'])

def find_grid_objects(grid_np, ignore_color=0, connectivity=1):
    """Finds distinct objects (connected components) in a grid for each color."""
    objects = []
    if grid_np.size == 0: return objects
    unique_colors = np.unique(grid_np)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]]) if connectivity == 1 else np.ones((3,3), dtype=bool)
    
    for color in unique_colors:
        if color == ignore_color: continue
        binary_mask = (grid_np == color)
        labeled_mask, num_labels = label(binary_mask, structure=structure)
        if num_labels > 0:
            bboxes = find_objects(labeled_mask)
            if len(bboxes) == num_labels: # Ensure find_objects returns expected number
                for i in range(num_labels):
                    obj_label = i + 1
                    obj_mask = (labeled_mask == obj_label)
                    obj_bbox_slices = bboxes[i]
                    obj_coords = np.argwhere(obj_mask)
                    if obj_coords.size > 0: # Ensure object is not empty
                        obj = Object(mask=obj_mask, color=color, bbox=obj_bbox_slices, pixel_coords=obj_coords)
                        objects.append(obj)
            # else: print(f"Warning: Mismatch between num_labels {num_labels} and bboxes {len(bboxes)}") # Keep commented
    return objects

# --- Object-Based Transformations ---
def transform_keep_largest_objects(grid_np, ignore_color=0, connectivity=1, **kwargs):
    """Returns grid containing only the object(s) with the maximum pixel count."""
    objects = find_grid_objects(grid_np, ignore_color=ignore_color, connectivity=connectivity)
    if not objects: return np.full_like(grid_np, ignore_color) if grid_np.size > 0 else np.array([[ignore_color]])
    object_sizes = [len(obj.pixel_coords) for obj in objects]
    max_size = max(object_sizes) if object_sizes else 0
    if max_size == 0: return np.full_like(grid_np, ignore_color) if grid_np.size > 0 else np.array([[ignore_color]])
    
    output_grid = np.full_like(grid_np, ignore_color)
    for obj, size in zip(objects, object_sizes):
        if size == max_size:
            output_grid[obj.mask] = obj.color
    return output_grid


def solve_task(task_data):
    """
    Attempts to solve an ARC task by finding a consistent simple transformation
    from a predefined list (geometry, color changes, keep largest object).
    """
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', []) 
    task_id = task_data.get('task_id', 'Unknown') # Get task_id if available in data

    # --- Fallback for tasks with no training data ---
    if not train_pairs:
        predictions = []
        for test_pair in test_pairs:
             try:
                 input_grid_np = np.array(test_pair['input'])
                 pred_1 = input_grid_np.tolist()
                 pred_2 = np.zeros_like(input_grid_np, dtype=int).tolist() if input_grid_np.size > 0 else [[0,0,0],[0,0,0],[0,0,0]]
                 if not pred_1: pred_1 = [[0]] # Ensure not empty list
                 if not pred_2: pred_2 = [[0]] 
             except Exception: # Handle malformed test_pair input
                 pred_1 = [[0]]
                 pred_2 = [[0]]
             predictions.append({"attempt_1": pred_1, "attempt_2": pred_2})
        return predictions

    # --- Define Base Transformations ---
    transformations_to_test = [
        ('identity', transform_identity, {}),
        ('rotate_90', transform_rotate, {'k': 1}),
        ('rotate_180', transform_rotate, {'k': 2}),
        ('rotate_270', transform_rotate, {'k': 3}),
        ('reflect_v', transform_reflect, {'axis': 0}),
        ('reflect_h', transform_reflect, {'axis': 1}),
        ('keep_largest_obj_conn1', transform_keep_largest_objects, {'ignore_color': 0, 'connectivity': 1}),
        ('keep_largest_obj_conn2', transform_keep_largest_objects, {'ignore_color': 0, 'connectivity': 2}),
    ]
    
    # --- Dynamically Add Color Rules ---
    all_colors = set()
    for pair in train_pairs: 
        try:
             all_colors.update(np.unique(pair['input']))
             all_colors.update(np.unique(pair['output']))
        except Exception: pass 
    color_list = sorted(list(all_colors))
    
    for c1, c2 in itertools.combinations(color_list, 2):
         transformations_to_test.append(
             (f'swap_{c1}_{c2}', transform_swap_colors, {'c1': c1, 'c2': c2})
         )
    for c_from in color_list:
        for c_to in color_list:
            if c_from != c_to:
                 transformations_to_test.append(
                     (f'recolor_{c_from}_to_{c_to}', transform_recolor_single, {'color_from': c_from, 'color_to': c_to})
                 )

    # --- Hypothesis Testing Loop ---
    consistent_rules = []
    for rule_name, transform_func, args in transformations_to_test:
        is_consistent = True
        for pair_idx, pair in enumerate(train_pairs):
            try:
                input_np = np.array(pair['input'])
                output_np = np.array(pair['output'])
                predicted_output_np = transform_func(input_np, **args)
                if not np.array_equal(predicted_output_np, output_np):
                    is_consistent = False; break 
            except Exception as e: is_consistent = False; break
        if is_consistent: 
            consistent_rules.append({'name': rule_name, 'func': transform_func, 'args': args})

    # --- Rule Selection and Application ---
    predictions = []
    selected_rule = None
    if consistent_rules:
        selected_rule = consistent_rules[0] # Still using first found rule

    # Generate predictions for each test input
    for test_idx, test_pair in enumerate(test_pairs):
        pred_1_list = [[0]] 
        pred_2_list = [[0]] 
        try:
            input_grid_np = np.array(test_pair['input'])
            
            # --- Attempt 1 ---
            if selected_rule:
                pred_1_np = selected_rule['func'](input_grid_np, **selected_rule['args'])
                if pred_1_np.ndim == 2 and pred_1_np.shape[0] > 0 and pred_1_np.shape[1] > 0:
                     pred_1_list = pred_1_np.tolist()
                else: pred_1_list = input_grid_np.tolist() # Fallback
            else: pred_1_list = input_grid_np.tolist() # Fallback: copy input

            # --- Attempt 2 ---
            if selected_rule and selected_rule['name'] != 'identity':
                 pred_2_list = input_grid_np.tolist() # Try copy input
            else: # No rule or identity rule -> try zeros
                 pred_2_list = np.zeros_like(input_grid_np, dtype=int).tolist() if input_grid_np.size > 0 else [[0,0,0],[0,0,0],[0,0,0]]

            # Ensure predictions are not empty lists
            if not pred_1_list: pred_1_list = [[0]]
            if not pred_2_list: pred_2_list = [[0]]
            
        except Exception as e: # Broad exception catch during prediction
             try: # Try copy input as fallback 1
                 pred_1_list = np.array(test_pair['input']).tolist()
                 if not pred_1_list: pred_1_list = [[0]]
             except: pred_1_list = [[0]]
             pred_2_list = [[0]] # Safest fallback for attempt 2

        predictions.append({
            "attempt_1": pred_1_list,
            "attempt_2": pred_2_list
        })

    # Final validation of prediction count
    if len(predictions) != len(test_pairs):
         predictions = []
         num_test_cases = len(task_data.get('test', [])) # Recalculate safely
         for _ in range(num_test_cases):
              dummy_pred = [[0]]
              predictions.append({"attempt_1": dummy_pred, "attempt_2": dummy_pred})
    return predictions


def generate_submission(solver_func, test_task_dict, filename="submission.json"):
    """
    Generates the submission JSON file by applying the solver_func to each task.
    """
    submission = {}
    if not test_task_dict:
         print("Error: test_task_dict is empty. Cannot generate submission.")
         return None
    print(f"Generating submission for {len(test_task_dict)} test tasks...")
    tasks_with_id = {tid: {**tdata, 'task_id': tid} for tid, tdata in test_task_dict.items()}

    for task_id, task_data in tqdm(tasks_with_id.items(), desc="Solving Tasks"):
        try:
            if 'test' not in task_data or not isinstance(task_data['test'], list):
                 raise ValueError(f"Task {task_id} missing or invalid 'test' field.")
            task_predictions = solver_func(task_data) 
            if not isinstance(task_predictions, list) or len(task_predictions) != len(task_data['test']):
                 raise ValueError(f"Solver returned invalid prediction list length for task {task_id}.")
            
            valid_predictions = []
            for i, pred_dict in enumerate(task_predictions): # Iterate with index
                 if not isinstance(pred_dict, dict) or "attempt_1" not in pred_dict or "attempt_2" not in pred_dict:
                     raise ValueError(f"Invalid prediction dict structure for task {task_id}, item {i}.")
                 for attempt in ["attempt_1", "attempt_2"]: # Validate grid format
                     grid = pred_dict[attempt]
                     if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                          # Attempt to fix simple np array issue, otherwise raise error
                          try: pred_dict[attempt] = np.array(grid).tolist()
                          except: raise ValueError(f"Grid {attempt} not list of lists for task {task_id}, item {i}.")
                     if not pred_dict[attempt]: # Handle empty grid prediction -> make it [[0]]
                          pred_dict[attempt] = [[0]]
                 valid_predictions.append(pred_dict)
            submission[task_id] = valid_predictions
            
        except Exception as e:
            print(f"\nError processing task {task_id}: {e}. Generating default fallback.")
            dummy_predictions = []
            num_test_cases = len(task_data.get('test', []))
            for _ in range(num_test_cases):
                dummy_grid = [[0]]
                dummy_predictions.append({"attempt_1": dummy_grid, "attempt_2": dummy_grid})
            submission[task_id] = dummy_predictions

    output_path = os.path.join('/kaggle/working', filename)
    try:
        with open(output_path, 'w') as f: json.dump(submission, f)
        print(f"\nSubmission file successfully saved to: {output_path}")
    except Exception as e:
        print(f"\nError writing submission file to {output_path}: {e}"); return None
    return output_path 

# --- Generate the Submission ---
print("\nStarting final submission generation...")
if dataset and hasattr(dataset, 'test_data') and dataset.test_data:
     submission_file = generate_submission(solve_task, dataset.test_data, filename="submission.json")
     if submission_file: print(f"\nSubmission generation complete. File: {submission_file}")
     else: print("\nSubmission generation failed.")
else: print("\nError: Test data not loaded correctly. Cannot generate submission.")

