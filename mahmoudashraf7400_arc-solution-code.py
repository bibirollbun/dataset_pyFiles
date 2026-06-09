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



# A function to visualize the grid
def plot_grid(grid):
    """
    Plots a 2D grid as a colored image.
    Each integer in the grid corresponds to a color.
    """
    # Create a new figure and axes for the plot
    fig, ax = plt.subplots()
    
    # Use imshow to display the grid as an image
    # We use a specific color map to map integers to colors
    # The 'viridis' color map is a good default, but you can change it  to {'plasma','magma','cividis',....}
    ax.imshow(grid, cmap='viridis')
    
    # Set the ticks and labels to display the grid structure
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels(np.arange(grid.shape[1]))
    ax.set_yticklabels(np.arange(grid.shape[0]))
    ax.set_title("ARC Grid") 

   # Add a grid to the plot for better readability
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    
 # Display the plot
    plt.show()


def solve_task(task):
    """
    This is a placeholder function for solving a single ARC task.
    You will implement your core logic here.
    
    Args:
        task (dict): A dictionary containing 'train' and 'test' data for a task.
        
    Returns:
        list: A list of predicted output grids for the test inputs.
        The format should be compatible with the Kaggle submission.
        
    Note: For this example, we simply return the test input as the output.
    You must replace this with your actual solving algorithm.
    """
    predictions = []
    for test_pair in task['test']:
        # This is a very simple (and incorrect) baseline solution:
        # just return the input grid as the output.
        # Your real solution will need to be much more complex.
        predictions.append(test_pair['input'])
    return predictions



# Main script
if __name__ == "__main__":
    # The name of the dataset file
    file_path = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
    
    submission = {}
    
    try:
        # Load the JSON data from the file
        with open(file_path, 'r') as f:
            tasks = json.load(f)
            
        # Select the first few tasks for a demonstration
        for task_id in list(tasks.keys())[:3]: # Visualizing the first 3 tasks as an example
            task = tasks[task_id]
            
            print(f"Loading and visualizing task: {task_id}")
            
            # Iterate over the training examples in the task
            for i, pair in enumerate(task['train']):
                print(f"--- Training Pair {i+1} ---")
                
                # Convert the input and output grids to numpy arrays
                input_grid = np.array(pair['input'])
                output_grid = np.array(pair['output'])
                
                # Plot the input grid
                print("Input Grid:")
                plot_grid(input_grid)
                
                # Plot the output grid
                print("Output Grid:")
                plot_grid(output_grid)
            
            # Solve the task and generate a prediction
            predicted_outputs = solve_task(task)
            submission[task_id] = predicted_outputs
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please make sure the data is in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")



# After processing all tasks, save the predictions to a submission file
with open('submission.json', 'w') as f:
    json.dump(submission, f, indent=4)
print("\nSubmission file 'submission.json' has been created.")





# After processing all tasks, save the predictions to a submission file
with open('submission.json', 'w') as f:
    json.dump(submission, f, indent=4)
print("\nSubmission file 'submission.json' has been created.")

