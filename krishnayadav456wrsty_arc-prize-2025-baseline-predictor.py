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
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = '/kaggle/input/arc-prize-2025/'

with open(Path(DATA_DIR) / 'arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)

with open(Path(DATA_DIR) / 'arc-agi_evaluation_challenges.json') as f:
    eval_challenges = json.load(f)

with open(Path(DATA_DIR) / 'arc-agi_test_challenges.json') as f:
    test_challenges = json.load(f)

print(f"Number of training tasks: {len(train_challenges)}")
print(f"Number of evaluation tasks: {len(eval_challenges)}")
print(f"Number of test tasks: {len(test_challenges)}")





def plot_grid(grid, title=""):
    """
    Visualize a grid of integers (0-9) as a colored matrix.
    """
    plt.figure(figsize=(5,5))
    plt.imshow(grid, cmap='tab10', vmin=0, vmax=9)
    plt.title(title)
    plt.axis('off')
    plt.show()

def grid_diff(a, b):
    """
    Calculate difference between two grids by comparing only overlapping area.
    """
    min_rows = min(len(a), len(b))
    min_cols = min(len(a[0]), len(b[0]))
    a_sub = np.array(a)[:min_rows, :min_cols]
    b_sub = np.array(b)[:min_rows, :min_cols]
    return np.sum(a_sub != b_sub)





sample_task = list(train_challenges.values())[0]  

print("Keys in task:", sample_task.keys())
print("Number of train pairs:", len(sample_task['train']))
print("Number of test inputs:", len(sample_task['test']))

# Visualize first train input/output pair and test input
train_input_0 = sample_task['train'][0]['input']
train_output_0 = sample_task['train'][0]['output']
test_input_0 = sample_task['test'][0]['input']

plot_grid(train_input_0, "Train Input")
plot_grid(train_output_0, "Train Output")
plot_grid(test_input_0, "Test Input")





def predict_output(task):
    train_pairs = task['train']
    test_input = task['test'][0]['input']
    
    best_diff = np.inf
    best_output = None
    
    for pair in train_pairs:
        train_input = pair['input']
        train_output = pair['output']
        
        diff = grid_diff(test_input, train_input)
        
        if diff < best_diff:
            best_diff = diff
            best_output = train_output
            
    return best_output





predicted_output = predict_output(sample_task)
plot_grid(predicted_output, "Predicted Output")



# === Main Heuristic Prediction Logic ===
def smart_predict(task):
    predictions = []
    train = task["train"]
    test = task["test"]
    example_output = train[-1]["output"]

    for test_input in test:
        # Attempt 1: Repeat last output grid from training
        attempt_1 = copy_shape_from_train(train[-1])

        # Attempt 2: Use majority color from test input, fill same shape
        attempt_2 = fill_with_color_like_input(test_input["input"])

        predictions.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
    return predictions


submission_outputs = []

for task_id, task in test_challenges.items():
    pred_output = predict_output(task)
    submission_outputs.append({"task_id": task_id, "output": pred_output})

print(f"Generated predictions for {len(submission_outputs)} test tasks.")


# submission = {"outputs": submission_outputs}

# with open("submission.json", "w") as f:
#     json.dump(submission, f)

# print("Submission file saved as submission.json")


import json
from pathlib import Path

def load_test_data():
    try:
     
        path = Path("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json")
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
      
        path = Path("/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json")
        with open(path) as f:
            data = json.load(f)

 
    if isinstance(data, dict) and len(data) == 1:
       
        return list(data.values())[0]
    return data

test_data = load_test_data()

def baseline_predict(task):
    last_output = task["train"][-1]["output"]
    return [{"attempt_1": last_output, "attempt_2": last_output} for _ in task["test"]]

# Create predictions for all tasks
submission = {task_id: baseline_predict(task) for task_id, task in test_data.items()}

# Write to /kaggle/working/submission.json (required path)
output_path = Path("/kaggle/working/submission.json")
with open(output_path, "w") as f:
    json.dump(submission, f)

print("✅ Submission file created successfully:", output_path)


