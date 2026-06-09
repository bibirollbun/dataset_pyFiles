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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import json
import os
from pathlib import Path
import collections # For counting elements
import seaborn as sns # For enhanced plots like heatmaps


BASE_PATH = '/kaggle/input/arc-prize-2025/'

# Standard ARC color map
CMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
)
NORM = colors.Normalize(vmin=0, vmax=9)


def load_json(file_path):
    """Loads a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_arc_dataset(base_path):
    """Loads the training and evaluation datasets."""
    training_challenges_path = Path(base_path) / 'arc-agi_training_challenges.json'
    training_solutions_path = Path(base_path) / 'arc-agi_training_solutions.json'
    evaluation_challenges_path = Path(base_path) / 'arc-agi_evaluation_challenges.json'
    evaluation_solutions_path = Path(base_path) / 'arc-agi_evaluation_solutions.json'

    train_challenges = load_json(training_challenges_path)
    train_solutions = load_json(training_solutions_path)
    eval_challenges = load_json(evaluation_challenges_path)
    eval_solutions = load_json(evaluation_solutions_path) # Solutions for local evaluation

    return train_challenges, train_solutions, eval_challenges, eval_solutions


def plot_grid(ax, grid, title=""):
    """Plots a single ARC grid."""
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid) # Ensure it's a numpy array for imshow

    ax.imshow(grid, cmap=CMAP, norm=NORM)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_xticks([x - 0.5 for x in range(1, grid.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, grid.shape[0])], minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_task(task, task_id, solutions=None, include_solutions=False):
    """
    Plots all train pairs and test inputs for a given task.
    Optionally plots ground truth solutions for test inputs if provided.
    """
    n_train_pairs = len(task['train'])
    n_test_pairs = len(task['test'])
    
    cols = n_train_pairs * 2 + n_test_pairs * (2 if include_solutions else 1)
    
    fig, axs = plt.subplots(1, cols, figsize=(3 * cols, 3.5))
    if cols == 1: # Handle case where only 1 subplot is created (e.g., 1 test input, no train)
        axs = [axs] 
    else:
        axs = axs.flatten()
    
    plot_idx = 0

    for i, pair in enumerate(task['train']):
        plot_grid(axs[plot_idx], pair['input'], title=f'Train {i} Input')
        plot_idx += 1
        plot_grid(axs[plot_idx], pair['output'], title=f'Train {i} Output')
        plot_idx += 1

    for i, pair in enumerate(task['test']):
        plot_grid(axs[plot_idx], pair['input'], title=f'Test {i} Input')
        plot_idx += 1
        if include_solutions and solutions:
            test_solution = None
            if task_id in solutions:
                # Find the specific test solution for this index
                test_solution = solutions[task_id][i]['output']
            if test_solution:
                plot_grid(axs[plot_idx], test_solution, title=f'Test {i} GT Output')
                plot_idx += 1
            else:
                axs[plot_idx].axis('off')
                axs[plot_idx].set_title(f'Test {i} No GT')
                plot_idx += 1
    
    for i in range(plot_idx, len(axs)):
        fig.delaxes(axs[i])

    fig.suptitle(f"Task: {task_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def get_grid_features(grid_data, dataset_type="train"):
    """
    Extracts basic features from input/output grids for EDA.
    Returns a list of dictionaries, one for each grid.
    """
    features = []
    
    for task_id, task in grid_data.items():
        # Process training pairs
        for i, io_pair in enumerate(task['train']):
            input_grid = np.array(io_pair['input'])
            output_grid = np.array(io_pair['output'])

            features.append({
                'task_id': task_id,
                'pair_idx': i,
                'type': 'train_input',
                'rows': input_grid.shape[0],
                'cols': input_grid.shape[1],
                'total_cells': input_grid.size,
                'unique_colors': len(np.unique(input_grid)),
                'color_counts': collections.Counter(input_grid.flatten()),
                'aspect_ratio': input_grid.shape[0] / input_grid.shape[1] if input_grid.shape[1] > 0 else 0
            })
            features.append({
                'task_id': task_id,
                'pair_idx': i,
                'type': 'train_output',
                'rows': output_grid.shape[0],
                'cols': output_grid.shape[1],
                'total_cells': output_grid.size,
                'unique_colors': len(np.unique(output_grid)),
                'color_counts': collections.Counter(output_grid.flatten()),
                'aspect_ratio': output_grid.shape[0] / output_grid.shape[1] if output_grid.shape[1] > 0 else 0
            })
        
        # Process test inputs (outputs are unknown for submission)
        if dataset_type == "eval":
            for i, io_pair in enumerate(task['test']):
                input_grid = np.array(io_pair['input'])
                features.append({
                    'task_id': task_id,
                    'pair_idx': i,
                    'type': 'test_input',
                    'rows': input_grid.shape[0],
                    'cols': input_grid.shape[1],
                    'total_cells': input_grid.size,
                    'unique_colors': len(np.unique(input_grid)),
                    'color_counts': collections.Counter(input_grid.flatten()),
                    'aspect_ratio': input_grid.shape[0] / input_grid.shape[1] if input_grid.shape[1] > 0 else 0
                })
    return pd.DataFrame(features)


def plot_grid_size_distribution(df, title_suffix=""):
    """Plots histograms of grid rows, columns, and total cells."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(df['rows'], bins=range(1, 31), kde=True, ax=axes[0])
    axes[0].set_title(f'Distribution of Rows {title_suffix}')
    axes[0].set_xlabel('Rows')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(range(0, 31, 2))

    sns.histplot(df['cols'], bins=range(1, 31), kde=True, ax=axes[1])
    axes[1].set_title(f'Distribution of Columns {title_suffix}')
    axes[1].set_xlabel('Columns')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks(range(0, 31, 2))
    
    sns.histplot(df['total_cells'], bins=50, kde=True, ax=axes[2])
    axes[2].set_title(f'Distribution of Total Cells {title_suffix}')
    axes[2].set_xlabel('Total Cells (Rows * Cols)')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def plot_color_distribution(df, title_suffix=""):
    """Plots the overall distribution of colors."""
    all_colors_counter = collections.Counter()
    for counts_dict in df['color_counts']:
        all_colors_counter.update(counts_dict)
    
    colors_df = pd.DataFrame(all_colors_counter.items(), columns=['Color', 'Frequency']).sort_values('Color')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Color', y='Frequency', data=colors_df, palette=CMAP.colors[:10]) # Use ARC colors
    plt.title(f'Overall Color Distribution {title_suffix}')
    plt.xlabel('Color ID')
    plt.ylabel('Total Pixel Count')
    plt.xticks(range(10), labels=[str(i) for i in range(10)])
    plt.show()


def plot_size_relationship(df, title_suffix=""):
    """Plots relationship between input and output grid sizes."""
    
    # Filter for training pairs only, as they have both input and output
    train_io_df = df[df['type'].isin(['train_input', 'train_output'])].copy()
    
    # Create a unique ID for each train I/O pair to link them
    train_io_df['io_pair_id'] = train_io_df['task_id'] + '_' + train_io_df['pair_idx'].astype(str)

    # Pivot to get input and output sizes side by side
    pivot_df = train_io_df.pivot(index='io_pair_id', columns='type', values=['rows', 'cols', 'total_cells', 'aspect_ratio'])
    
    # Flatten multi-index columns
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    # Fix: Pass string column names for x and y
    sns.scatterplot(x='train_input_total_cells', y='train_output_total_cells', data=pivot_df)
    plt.title(f'Input vs Output Total Cells {title_suffix}')
    plt.xlabel('Input Total Cells')
    plt.ylabel('Output Total Cells')
    plt.plot([0, 900], [0, 900], color='red', linestyle='--', label='Input == Output') # Max grid size 30x30 = 900
    plt.legend()

    plt.subplot(1, 2, 2)
    # Fix: Pass string column names for x and y
    sns.scatterplot(x='train_input_aspect_ratio', y='train_output_aspect_ratio', data=pivot_df)
    plt.title(f'Input vs Output Aspect Ratio {title_suffix}')
    plt.xlabel('Input Aspect Ratio (Rows/Cols)')
    plt.ylabel('Output Aspect Ratio (Rows/Cols)')
    plt.plot([0, 30], [0, 30], color='red', linestyle='--', label='Input == Output')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_size_diff_distribution(df, title_suffix=""):
    """Plots the distribution of the difference in total cells between input and output."""
    
    train_io_df = df[df['type'].isin(['train_input', 'train_output'])].copy()
    train_io_df['io_pair_id'] = train_io_df['task_id'] + '_' + train_io_df['pair_idx'].astype(str)
    pivot_df = train_io_df.pivot(index='io_pair_id', columns='type', values='total_cells')
    pivot_df['size_diff'] = pivot_df['train_output'] - pivot_df['train_input']

    plt.figure(figsize=(8, 6))
    sns.histplot(pivot_df['size_diff'], bins=50, kde=True)
    plt.title(f'Distribution of (Output - Input) Total Cells Difference {title_suffix}')
    plt.xlabel('Total Cells Difference')
    plt.ylabel('Count')
    plt.axvline(0, color='red', linestyle='--', label='No size change')
    plt.legend()
    plt.show()


def plot_unique_colors_distribution(df, title_suffix=""):
    """Plots the distribution of unique colors per grid."""
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['unique_colors'], bins=range(1, 11), kde=False) # Max 10 colors (0-9)
    plt.title(f'Distribution of Unique Colors per Grid {title_suffix}')
    plt.xlabel('Number of Unique Colors')
    plt.ylabel('Count of Grids')
    plt.xticks(range(1, 11))
    plt.show()


def get_object_count(grid):
    """
    A simple connected component labeling to estimate 'number of objects'.
    This is a heuristic and might not perfectly align with human 'objects'.
    """
    from skimage.measure import label
    # Convert 0s (background) to -1 to differentiate from actual objects if needed
    # For object counting, we typically label connected regions of NON-BACKGROUND pixels.
    # A simple approach is to treat 0 as background, and all other colors as potential parts of objects.
    
    # Create a binary mask where non-zero pixels are 1 (potential object) and 0s are 0 (background)
    binary_grid = (grid != 0).astype(int)
    
    # Label connected components
    labeled_grid = label(binary_grid, connectivity=1) # 1 for 4-connectivity, 2 for 8-connectivity
    
    # Number of unique labels (excluding 0, which is background if present)
    return len(np.unique(labeled_grid)) - 1 if 0 in np.unique(labeled_grid) else len(np.unique(labeled_grid))


def plot_object_count_distribution(df, title_suffix="", task_data_for_object_count=None):
    """
    Estimates and plots the distribution of the number of 'objects' in input grids.
    This is a heuristic.
    """
    object_counts = []
    
    # You were correctly identifying that this needs the raw challenges data.
    # Now, 'task_data_for_object_count' is available as a parameter.
    if task_data_for_object_count is None:
        print("Warning: task_data_for_object_count not provided. Cannot calculate object counts.")
        return

    print(f"\nCalculating object counts for {title_suffix} (this might take a moment)...")
    for task_id, task in task_data_for_object_count.items():
        for io_pair in task['train']:
            input_grid = np.array(io_pair['input'])
            object_counts.append(get_object_count(input_grid))
        # Also include test inputs for a more complete picture of task complexity
        if 'test' in task:
            for test_pair in task['test']:
                input_grid = np.array(test_pair['input'])
                object_counts.append(get_object_count(input_grid))

    if not object_counts:
        print(f"No object counts to plot for {title_suffix}.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(object_counts, bins=max(object_counts) + 1 if object_counts else 1, kde=False)
    plt.title(f'Distribution of Number of Objects in Input Grids {title_suffix}')
    plt.xlabel('Number of Objects')
    plt.ylabel('Count of Grids')
    plt.xticks(range(0, max(object_counts) + 1, 2) if object_counts else [0])
    plt.show()


if __name__ == "__main__":
    print("Loading ARC datasets for EDA...")
    train_challenges, train_solutions, eval_challenges, eval_solutions = load_arc_dataset(BASE_PATH)
    print("Datasets loaded.")

    print("\n--- EDA for Training Set (Challenges) ---")
    train_df = get_grid_features(train_challenges, dataset_type="train")
    print(f"Number of training grid entries (input/output pairs): {len(train_df)}")
    print("\nBasic statistics for training set grids:")
    print(train_df[['rows', 'cols', 'total_cells', 'unique_colors', 'aspect_ratio']].describe())

    plot_grid_size_distribution(train_df[train_df['type'] == 'train_input'], "(Training Input)")
    plot_grid_size_distribution(train_df[train_df['type'] == 'train_output'], "(Training Output)")
    
    plot_color_distribution(train_df, "(Training Set)")
    plot_unique_colors_distribution(train_df, "(Training Set)")
    
    plot_size_relationship(train_df, "(Training Set)")
    plot_size_diff_distribution(train_df, "(Training Set)")
    
    # For object count, we need the raw challenges data again.
    # Pass train_challenges for object counting in training data
    plot_object_count_distribution(train_df, "(Training Inputs)", task_data_for_object_count=train_challenges)


    print("\n--- EDA for Evaluation Set (Challenges) ---")
    # For evaluation, we only have inputs for test, but train pairs are also there
    eval_df = get_grid_features(eval_challenges, dataset_type="eval")
    print(f"Number of evaluation grid entries (input/output/test_input pairs): {len(eval_df)}")
    print("\nBasic statistics for evaluation set grids:")
    print(eval_df[['rows', 'cols', 'total_cells', 'unique_colors', 'aspect_ratio']].describe())

    plot_grid_size_distribution(eval_df[eval_df['type'].isin(['train_input', 'test_input'])], "(Evaluation Inputs)")
    plot_grid_size_distribution(eval_df[eval_df['type'] == 'train_output'], "(Evaluation Train Output)")
    
    plot_color_distribution(eval_df, "(Evaluation Set)")
    plot_unique_colors_distribution(eval_df, "(Evaluation Set)")
    
    # Relationship plots only make sense for train pairs
    plot_size_relationship(eval_df, "(Evaluation Set Train Pairs)")
    plot_size_diff_distribution(eval_df, "(Evaluation Set Train Pairs)")

    # Pass eval_challenges for object counting in evaluation data
    plot_object_count_distribution(eval_df, "(Evaluation Inputs)", task_data_for_object_count=eval_challenges)
    
    print("\n--- Example: Plotting a few specific tasks from evaluation set ---")
    for i, (task_id, task_data) in enumerate(list(eval_challenges.items())[:2]): # Plot first 2 eval tasks
        plot_task(task_data, task_id) # No solutions shown for eval tasks

    print("\nEDA Complete.")
    print("Remember to interpret these plots to understand the dataset characteristics.")
    print("For instance, look for common grid sizes, dominant colors, and typical size changes between input and output.")

