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
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path
import random
import time
from copy import deepcopy
from typing import List, Dict, Callable, Optional, Tuple
from tqdm import tqdm


# --- Config ---
DATA_PATH_TRAIN = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
DATA_PATH_TRAIN_SOL = "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json"
DATA_PATH_VALID = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"
DATA_PATH_VALID_SOL = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json"
DATA_PATH_TEST = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Load Data ---
with open(DATA_PATH_TRAIN, "r") as f:
    train_challenges = json.load(f)
with open(DATA_PATH_VALID, "r") as f:
    valid_challenges = json.load(f)
with open(DATA_PATH_TEST, "r") as f:
    test_challenges = json.load(f)

print(f"✅ Loaded {len(train_challenges)} training pairs, {len(valid_challenges)} validation pairs, {len(test_challenges)} test pairs.")
print(type(train_challenges))
sample = list(train_challenges.keys())[0]
print(sample)
train_challenges[sample]['train']


# --- Sample Check ---
task_id = random.choice(list(train_challenges.keys()))
sample = train_challenges[task_id]
print("Sample task ID:", task_id)
print("Keys:", sample.keys())
print("Train samples:", len(sample["train"]))
print("Test samples:", len(sample["test"]))



# --- Multi-Test Check ---
multiple_tests_count = 0

for task_id, task_data in test_challenges.items():
    test_samples = task_data.get("test", [])
    if len(test_samples) > 1:
        multiple_tests_count += 1
        print(f"Task ID: {task_id}")
        print(f"Number of test samples: {len(test_samples)}")


# Visualize a grid with matplotlib
def show_grid(grid, title=None, cmap_name='tab20'):
    arr = np.array(grid)
    num_colors = int(arr.max()) + 1
    cmap = plt.cm.get_cmap(cmap_name, num_colors)

    plt.imshow(arr, cmap=cmap, vmin=0, vmax=num_colors - 1)
    plt.xticks([]), plt.yticks([])
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

# Visualize a task (input/output pairs)
def visualize_task(task, task_id=None):
    print(f"Task ID: {task_id}")
    for i, pair in enumerate(task['train']):
        print(f"Train Example {i+1}")
        show_grid(pair['input'], "Input")
        show_grid(pair['output'], "Output")

    for i, pair in enumerate(task['test']):
        print(f"Test Example {i+1}")
        show_grid(pair['input'], "Input")


# Example: Visualize a random training task
task_id = random.choice(list(train_challenges.keys()))
visualize_task(train_challenges[task_id], task_id)


# --- Transforms Types ---

def identity(grid):
    return [row[:] for row in grid]

def flip_horizontal(grid):
    return [row[::-1] for row in grid]

def flip_vertical(grid):
    return grid[::-1]

def transpose(grid):
    n_rows = len(grid)
    n_cols = len(grid[0])
    new_grid = []

    for j in range(n_cols):
        new_row = []
        for i in range(n_rows):
            new_row.append(grid[i][j])
        new_grid.append(new_row)
    return new_grid

def rotate_90(grid):
    return [list(row)[::-1] for row in transpose(grid)]

def rotate_180(grid):
    return [row[::-1] for row in grid[::-1]]

def rotate_270(grid):
    return [list(row) for row in transpose(grid)][::-1]

def color_map(grid):
    return [[1 if c==0 else 0 if c==1 else c for c in row] for row in grid]

def scaling_down(grid):
    n_rows, n_cols = len(grid), len(grid[0])
    new_rows = max(1, n_rows // 2)
    new_cols = max(1, n_cols // 2)
    
    new_grid = [
        [grid[i*2][j*2] for j in range(new_cols)]
        for i in range(new_rows)
    ]
    return new_grid


def scaling_up(grid):
    n_rows, n_cols = len(grid), len(grid[0])
    new_rows = n_rows*2
    new_cols = n_cols*2
    
    new_grid = [
        [grid[i//2][j//2] for j in range(new_cols)]
        for i in range(new_rows)
    ]
    return new_grid


def fill_row(grid):
    return [grid[0][:] for _ in grid]  


def fill_col(grid):
    n_rows, n_cols = len(grid), len(grid[0])
    pattern = [grid[i][0] for i in range(n_rows)] 
    return [[pattern[i] for j in range(n_cols)] for i in range(n_rows)]


def flood_fill(grid):
    n_rows, n_cols = len(grid), len(grid[0])
    target = grid[0][0]
    new_color = 1
    if target == new_color:
        return grid
    stack = [(0, 0)]
    while stack:
        i, j = stack.pop()
        if 0 <= i < n_rows and 0 <= j < n_cols and grid[i][j] == target:
            grid[i][j] = new_color
            stack.extend([(i-1,j),(i+1,j),(i,j-1),(i,j+1)])
    return grid



TRANSFORMS = {
    "identity": identity,
    "flip_h": flip_horizontal,
    "flip_v": flip_vertical,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "transpose": transpose,
    "color_map": color_map,
    "scaling_down": scaling_down,
    "scaling_up": scaling_up,
    "fill_row": fill_row,
    "fill_col": fill_col,
    "flood_fill": flood_fill,
}



# --- Grid Match ---
def grids_equal(grid1, grid2):
    if len(grid1) != len(grid2):
      return False
    else :
      for i in range(len(grid1)):
        if len(grid1[i]) != len(grid2[i]):
          return False
        else :
          for j in range(len(grid1[i])):
            if grid1[i][j] != grid2[i][j]:
              return False
    return True


# --- Transformation Solver ---
class TransformSolver:
    def __init__(self, transforms: Dict[str, Callable]):
        self.transforms = transforms
        self.task_best_transform: Dict[str, str] = {}

    def best_transform(self, task: Dict) -> Tuple[Optional[str], float]:
        train_samples = task.get("train", [])
        scores = {}
        
        for name, func in self.transforms.items():
            correct, total = 0, 0
            for sample in train_samples:
                input_grid = sample.get("input")
                output_grid = sample.get("output")
                pred = func(input_grid)
                if grids_equal(pred, output_grid):
                    correct += 1
                total += 1 
                
            if total > 0:
                scores[name] = correct / total
        
        if not scores:
            return None, 0.0
        
        best = max(scores, key=scores.get)
        return best, scores[best]


    def train(self, train_tasks: Dict[str, Dict], valid_tasks: Dict[str, Dict], epochs=100):
        for epoch in range(1, epochs+1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            
            # Identify best transform per task
            for task_id, task_data in train_tasks.items():
                best_transform, _ = self.best_transform(task_data)
                self.task_best_transform[task_id] = best_transform or "identity"

            # Evaluate training accuracy
            train_acc = self.evaluate(train_tasks)
            val_acc = self.evaluate(valid_tasks)
            print(f"Train Accuracy: {train_acc*100:.2f}% | Validation Accuracy: {val_acc*100:.2f}%")


    def evaluate(self, tasks: Dict[str, Dict]) -> float: 
        correct, total = 0, 0

        for task_id, task_data in tasks.items():
            name = self.task_best_transform.get(task_id, "identity")
            func = self.transforms.get(name, identity)
            
            for sample in task_data.get("train", []):  # <-- use task_data["train"]
                input_grid = sample.get("input")
                output_grid = sample.get("output")
                pred = func(input_grid)
                if grids_equal(pred, output_grid):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0


    def predict(self, test_tasks: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        results = {}

        for task_id, task_data in test_tasks.items():
            name = self.task_best_transform.get(task_id, "identity")
            func = self.transforms.get(name, identity)

            predictions = []
            for sample in task_data.get("test", []):
                input_grid = sample.get("input", [])
                try:
                    pred = func(input_grid)
                except Exception:
                    pred = [row[:] for row in input_grid]
                
                predictions.append({"attempt_1": pred, "attempt_2": pred})
            
            results[task_id] = predictions

        return results



# --- Define Main ---
def main():
    solver = TransformSolver(TRANSFORMS)
    solver.train(train_challenges, valid_challenges, epochs=100)
    submission = solver.predict(test_challenges)
    output_path = "./submission.json"

    # --- Save submission ---
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    print("Submission saved.")


    with open(output_path, "r") as f:
        submission = json.load(f)
    print(f"Number of tasks: {len(submission)}")

    max_show = 3
    for i, (task_id, attempts) in enumerate(submission.items()):
        if i > max_show:
            break

        print(f"Task ID: {task_id}")

        if isinstance(attempts, list) and len(attempts) > 0:
            attempt = attempts[0]

            for i, grid in attempt.items():
                print(f"{i}: {len(grid)}x{len(grid[0]) if grid else 0} grid")
                print(grid)


if __name__ == "__main__":
    main()




