import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


# Define paths to the datasets
training_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
evaluation_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json'
evaluation_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
sample_submission_path = '/kaggle/input/arc-prize-2025/sample_submission.json'
training_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
test_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'



# Load the datasets
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

training_solutions = load_json(training_solutions_path)
evaluation_solutions = load_json(evaluation_solutions_path)
evaluation_challenges = load_json(evaluation_challenges_path)
sample_submission = load_json(sample_submission_path)
training_challenges = load_json(training_challenges_path)
test_challenges = load_json(test_challenges_path)



# --- Data Analysis and Visualization ---

# 1. Basic Statistics
def dataset_statistics(challenges, dataset_name):
    """
    Calculates and prints basic statistics about the dataset.
    """
    num_tasks = len(challenges)
    input_sizes = []
    output_sizes = []

    for task_id, task in challenges.items():
        for io_pair in task['train']:
            input_sizes.append(np.array(io_pair['input']).shape)
            output_sizes.append(np.array(io_pair['output']).shape)

    print(f"--- {dataset_name} Statistics ---")
    print(f"Number of tasks: {num_tasks}")
    print(f"Number of examples: {len(input_sizes)}")
    print(f"Unique Input shapes: {len(set(input_sizes))}")
    print(f"Unique Output shapes: {len(set(output_sizes))}")
    print("-" * 30)

dataset_statistics(training_challenges, "Training Challenges")
dataset_statistics(evaluation_challenges, "Evaluation Challenges")
#dataset_statistics(test_challenges, "Test Challenges")



# 2. Visualization of Sample Tasks
def plot_task(task):
    """
    Visualizes a single ARC task with inputs and outputs.
    """
    num_train_pairs = len(task['train'])
    num_test_pairs = len(task['test'])

    fig, axes = plt.subplots(num_train_pairs + num_test_pairs, 2, figsize=(8, 4 * (num_train_pairs + num_test_pairs)))
    fig.suptitle(f"Task ID: {task_id}", fontsize=16)

    for i, io_pair in enumerate(task['train']):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        ax1.imshow(np.array(io_pair['input']), cmap='tab10')
        ax2.imshow(np.array(io_pair['output']), cmap='tab10')
        ax1.set_title(f'Train Input {i+1}')
        ax2.set_title(f'Train Output {i+1}')
        ax1.axis('off')
        ax2.axis('off')

    for i, io_pair in enumerate(task['test']):
        ax1 = axes[num_train_pairs + i, 0]
        ax2 = axes[num_train_pairs + i, 1]
        ax1.imshow(np.array(io_pair['input']), cmap='tab10')
        ax2.axis('off')  # We don't have test outputs to display
        ax1.set_title(f'Test Input {i+1}')
        ax2.set_title(f'Test Output {i+1} (Prediction)')
        ax1.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()

# Plot a sample task from the training set
task_id = list(training_challenges.keys())[0]
plot_task(training_challenges[task_id])

# 3. Color Distribution Analysis
def color_distribution(challenges, dataset_name):
    """
    Analyzes and visualizes the distribution of colors in the dataset.
    """
    colors = []
    for task_id, task in challenges.items():
        for io_pair in task['train']:
            colors.extend(np.array(io_pair['input']).flatten())
            colors.extend(np.array(io_pair['output']).flatten())


    color_counts = Counter(colors)
    sorted_colors = sorted(color_counts.items()) # Sort to ensure consistent color mapping

    color_values = [item[0] for item in sorted_colors]  # List of colors present
    counts = [item[1] for item in sorted_colors]  # List of color counts

    plt.figure(figsize=(10, 6))
    plt.bar(color_values, counts, color=[plt.cm.tab10(i) for i in color_values])  # Consistent coloring
    plt.xlabel("Color")
    plt.ylabel("Frequency")
    plt.title(f"Color Distribution in {dataset_name}")
    plt.xticks(color_values) # Set x-ticks to color values
    plt.grid(axis='y', alpha=0.75)
    plt.show()

color_distribution(training_challenges, "Training Challenges")
color_distribution(evaluation_challenges, "Evaluation Challenges")



# 4. Input/Output Size Correlation
def input_output_size_correlation(challenges, dataset_name):
    """
    Analyzes and visualizes the correlation between input and output grid sizes.
    """
    input_sizes = []
    output_sizes = []
    for task_id, task in challenges.items():
        for io_pair in task['train']:
            input_sizes.append(np.array(io_pair['input']).shape[0] * np.array(io_pair['input']).shape[1])
            output_sizes.append(np.array(io_pair['output']).shape[0] * np.array(io_pair['output']).shape[1])

    # Only use training data for the plot, because test data does not have outputs.
    plt.figure(figsize=(8, 6))
    plt.scatter(input_sizes, output_sizes, alpha=0.5)
    plt.xlabel("Input Grid Size (Number of Cells)")
    plt.ylabel("Output Grid Size (Number of Cells)")
    plt.title(f"Input/Output Size Correlation in {dataset_name} (Training Data Only)")
    plt.grid(True)
    plt.show()

input_output_size_correlation(training_challenges, "Training Challenges")

# 5. Distribution of Grid Sizes (Heatmap)
def grid_size_heatmap(challenges, dataset_name):
    """
    Creates a heatmap visualizing the distribution of grid dimensions.
    """
    rows = []
    cols = []
    for task_id, task in challenges.items():
        for io_pair in task['train']:
            rows.append(np.array(io_pair['input']).shape[0])
            cols.append(np.array(io_pair['input']).shape[1])
            rows.append(np.array(io_pair['output']).shape[0])
            cols.append(np.array(io_pair['output']).shape[1])

    grid_size_counts = Counter(zip(rows, cols))
    max_row = max(rows)
    max_col = max(cols)
    heatmap_data = np.zeros((max_row + 1, max_col + 1))  # +1 to include the max sizes

    for (row, col), count in grid_size_counts.items():
        heatmap_data[row, col] = count

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="viridis", annot=False, cbar_kws={'label': 'Frequency'})
    plt.xlabel("Column Size")
    plt.ylabel("Row Size")
    plt.title(f"Grid Size Distribution Heatmap - {dataset_name}")
    plt.show()

grid_size_heatmap(training_challenges, "Training Challenges")

# 6. Color Pair Co-occurrence Matrix
def color_cooccurrence_matrix(challenges, dataset_name):
    """
    Calculates and visualizes the co-occurrence of color pairs in input grids.
    """
    cooccurrence_matrix = np.zeros((10, 10))  # Assuming 10 colors (0-9)

    for task_id, task in challenges.items():
        for io_pair in task['train']:
            input_grid = np.array(io_pair['input'])
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    color1 = input_grid[i, j]
                    # Check neighbors (up, down, left, right)
                    neighbors = []
                    if i > 0:
                        neighbors.append(input_grid[i-1, j])
                    if i < input_grid.shape[0] - 1:
                        neighbors.append(input_grid[i+1, j])
                    if j > 0:
                        neighbors.append(input_grid[i, j-1])
                    if j < input_grid.shape[1] - 1:
                        neighbors.append(input_grid[i, j+1])

                    for color2 in neighbors:
                        cooccurrence_matrix[color1, color2] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cooccurrence_matrix, cmap="coolwarm", annot=True, fmt=".0f",
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Co-occurrence Count'})
    plt.title(f"Color Co-occurrence Matrix (Adjacent Cells) - {dataset_name}")
    plt.xlabel("Color 2")
    plt.ylabel("Color 1")
    plt.show()

color_cooccurrence_matrix(training_challenges, "Training Challenges")

# 7. Task Complexity (Number of Objects) Distribution
def task_complexity_distribution(challenges, dataset_name):
    """
    Estimates task complexity by counting connected components (objects) in input grids and visualizes its distribution.
    This is a simplified estimate and may not be accurate for all tasks.
    """
    from skimage import measure  # scikit-image

    num_objects_list = []
    for task_id, task in challenges.items():
        for io_pair in task['train']:
            input_grid = np.array(io_pair['input'])
            # Label connected regions (objects)
            labels = measure.label(input_grid, connectivity=1)  # 1 for 4-connectivity, 2 for 8-connectivity (3D)
            num_objects = labels.max()  # The highest label number is the number of objects
            num_objects_list.append(num_objects)

    plt.figure(figsize=(10, 6))
    plt.hist(num_objects_list, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Number of Connected Objects")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Task Complexity (Number of Objects) - {dataset_name}")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

task_complexity_distribution(training_challenges, "Training Challenges")

# 8. Ratio of Unique Colors in Input Grids
def unique_color_ratio(challenges, dataset_name):
    """
    Calculates and visualizes the ratio of unique colors to total cells in input grids.
    """
    ratios = []
    for task_id, task in challenges.items():
        for io_pair in task['train']:
            input_grid = np.array(io_pair['input'])
            num_unique_colors = len(np.unique(input_grid))
            total_cells = input_grid.size
            ratio = num_unique_colors / total_cells
            ratios.append(ratio)

    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel("Ratio of Unique Colors to Total Cells")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Unique Color Ratio - {dataset_name}")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

unique_color_ratio(training_challenges, "Training Challenges")

# 9. Distribution of Difference Between Input and Output Sizes

def io_size_difference(challenges, dataset_name):

    size_differences = []
    for task_id, task in challenges.items():
        for io_pair in task['train']:
            input_grid = np.array(io_pair['input'])
            output_grid = np.array(io_pair['output'])
            size_difference = input_grid.size - output_grid.size
            size_differences.append(size_difference)

    plt.figure(figsize=(10, 6))
    plt.hist(size_differences, bins=20, color='orange', edgecolor='black')
    plt.xlabel("Difference Between Input and Output Grid Size")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Input/Output Grid Size Differences - {dataset_name}")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

io_size_difference(training_challenges, "Training Challenges")




def create_dummy_submission(challenges):
    """
    Creates a valid dummy submission with a *very* basic pattern-based approach.
    """
    submission = {}
    for task_id, task in challenges.items():
        submission[task_id] = []
        for io_pair in task['test']:
            input_grid = np.array(io_pair['input'])  # NumPy for easier shape access
            output_shape = None

            # Infer output shape from training data (as before)
            for train_pair in task['train']:
                output_shape = np.array(train_pair['output']).shape
                break
            if output_shape is None:
                output_shape = input_grid.shape

            # Attempt a simple pattern: copy the input, but shift colors
            dummy_output_grid = (input_grid % 9 + 1).astype(int)  # Shift colors, wrap around 0-9
            #Resize if output shape is different
            if output_shape != input_grid.shape:
                dummy_output_grid = np.zeros(output_shape, dtype=int)
                min_row = min(output_shape[0], input_grid.shape[0])
                min_col = min(output_shape[1], input_grid.shape[1])
                dummy_output_grid[:min_row, :min_col] = (input_grid[:min_row, :min_col] % 9 +1).astype(int)


            dummy_output_grid = dummy_output_grid.tolist()

            # Create the prediction dictionary:
            prediction = {"attempt_1": dummy_output_grid, "attempt_2": dummy_output_grid}
            submission[task_id].append(prediction)
    return submission

# Create a dummy submission using the evaluation challenges
dummy_submission = create_dummy_submission(evaluation_challenges)

# --- Save Submission ---
def save_submission(submission, filename="submission.json"):
    """
    Saves the submission to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(submission, f)

save_submission(dummy_submission)

# --- Load and Display Head of Submission ---
loaded_submission = load_json("submission.json")
print("Head of Submission:")
print(dict(list(loaded_submission.items())[:3]))  # Print the first 3 task IDs in submission


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Define the color map to match the ARC app
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#85144B', '#98FB98', '#FFA500'])
norm = colors.BoundaryNorm(np.arange(11) - 0.5, 10)


def plot_task(task, task_solutions, i, t):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app.  Handles cases where a solution
    isn't available for plotting.
    """
    fs = 12
    num_train = len(task['train'])
    num_test = 1  # Keep this as 1 for ARC tasks

    w = num_train + num_test
    fig, axs = plt.subplots(2, w, figsize=(2 * w, 2 * 2))  # Increased figsize for readability
    plt.suptitle(f'Set #{i}, {t}:', fontsize=fs, fontweight='bold', y=1)


    for j in range(num_train):
        plot_one(axs[0, j], task, 'train', 'input', cmap, norm)
        plot_one(axs[1, j], task, 'train', 'output', cmap, norm)


    plot_one(axs[0, num_train], task, 'test', 'input', cmap, norm)

    # Plot the solution (if available)
    if task_solutions is not None:
        solution_matrix = task_solutions
        axs[1, num_train].imshow(solution_matrix, cmap=cmap, norm=norm)
        axs[1, num_train].grid(True, which='both', color='lightgrey', linewidth=0.5)
        axs[1, num_train].set_yticks([x - 0.5 for x in range(1 + len(solution_matrix))])
        axs[1, num_train].set_xticks([x - 0.5 for x in range(1 + len(solution_matrix[0]))])
        axs[1, num_train].set_xticklabels([])
        axs[1, num_train].set_yticklabels([])
        axs[1, num_train].set_title('Test output', fontsize=fs - 2)

    else:
        axs[1, num_train].text(0.5, 0.5, "Solution\nNot Available", ha="center", va="center", fontsize=10)
        axs[1, num_train].axis("off")  # Hide the axes if no solution

    # Draw separating lines (mostly decorative)

    for m in range(1, num_train):
        axs[1, num_train].plot([m, m], [0, 1], '--', linewidth=1, color='black', transform=axs[1, num_train].transAxes)  #Use transform to fix the line issue
    axs[1, num_train].plot([num_train, num_train], [0, 1], '-', linewidth=3, color='black', transform=axs[1, num_train].transAxes)

    # Customize the figure's appearance
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.15, top=0.85) #added this for the title

    print(f'#{i}, {t}')  # For fast and convenient search
    plt.show()
    print()



def plot_one(ax, task, train_or_test, input_or_output, cmap, norm):
    fs = 12
    input_matrix = np.array(task[train_or_test][0][input_or_output])
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs - 2)



for i in range(40, 54):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t][0]
    plot_task(task,  task_solution, i, t)




