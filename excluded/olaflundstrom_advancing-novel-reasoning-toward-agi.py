import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("Random seeds set for reproducibility.")


import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load evaluation challenges
eval_challenges_path = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"
with open(eval_challenges_path, "r") as f:
    eval_challenges = json.load(f)

num_tasks = len(eval_challenges)
print(f"Total Evaluation Tasks: {num_tasks}")

rows_list = []
cols_list = []
digits = []

for task_id, task in list(eval_challenges.items())[:100]:
    for pair in task.get("train", []):
        input_grid = pair.get("input")
        if isinstance(input_grid, list) and len(input_grid) > 0:
            rows = len(input_grid)
            cols = len(input_grid[0])
            rows_list.append(rows)
            cols_list.append(cols)
            for row in input_grid:
                digits.extend(row)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(rows_list, kde=True, bins=range(1, max(rows_list)+2), color="skyblue")
plt.title("Distribution of Grid Rows")
plt.xlabel("Rows")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.histplot(cols_list, kde=True, bins=range(1, max(cols_list)+2), color="salmon")
plt.title("Distribution of Grid Columns")
plt.xlabel("Columns")
plt.ylabel("Count")
plt.show()

digit_counts = pd.Series(digits).value_counts().sort_index()
plt.figure(figsize=(8, 5))
sns.barplot(x=digit_counts.index, y=digit_counts.values, palette="viridis")
plt.title("Frequency of Digits in Training Inputs (Sample of 100 Tasks)")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.show()

print("Rows Summary:", pd.Series(rows_list).describe())
print("Columns Summary:", pd.Series(cols_list).describe())


import cv2

def adjust_grid(prototype, target_shape):
    """
    Adjust a prototype grid to the target shape using center cropping if the prototype is larger,
    or zero-padding if it is smaller. Returns a list of lists of integers.
    """
    prot = np.array(prototype, dtype=np.uint8)
    target_h, target_w = target_shape
    h, w = prot.shape
    
    # If shapes are equal, return the prototype as is
    if (h, w) == (target_h, target_w):
        return prototype
    
    # Create a new grid of zeros of target shape
    new_grid = np.zeros((target_h, target_w), dtype=np.uint8)
    
    # Determine the region to copy from the prototype
    # For rows
    if h >= target_h:
        start_h = (h - target_h) // 2
        end_h = start_h + target_h
        prot_rows = target_h
    else:
        start_h = 0
        end_h = h
        prot_rows = h
    # For columns
    if w >= target_w:
        start_w = (w - target_w) // 2
        end_w = start_w + target_w
        prot_cols = target_w
    else:
        start_w = 0
        end_w = w
        prot_cols = w

    # Compute placement indices in new_grid
    pad_h = (target_h - prot_rows) // 2
    pad_w = (target_w - prot_cols) // 2

    # Determine region from prototype to copy
    prot_crop = prot[start_h:start_h+prot_rows, start_w:start_w+prot_cols]
    new_grid[pad_h:pad_h+prot_crop.shape[0], pad_w:pad_w+prot_crop.shape[1]] = prot_crop
    return new_grid.tolist()

def clean_grid(grid):
    """
    Ensure the grid is a list of lists of ints. If grid is empty or None, return [[0]].
    """
    if grid is None or not grid:
        return [[0]]
    return [[int(cell) for cell in row] for row in grid]

def generate_improved_prediction(task):
    """
    Generate predictions for a task using the first training output as a prototype.
    Adjust the prototype to exactly match the dimensions of each test input.
    Returns a list of dictionaries, one per test input, each with keys "attempt_1" and "attempt_2".
    """
    predictions = []
    train_pairs = task.get("train", [])
    prototype = None
    if train_pairs:
        first_output = train_pairs[0].get("output")
        prototype = clean_grid(first_output)
    
    test_inputs = task.get("test", [])
    if not isinstance(test_inputs, list):
        test_inputs = [test_inputs]
    
    for test_grid in test_inputs:
        if isinstance(test_grid, list) and test_grid and isinstance(test_grid[0], list):
            target_shape = (len(test_grid), len(test_grid[0]))
        else:
            target_shape = (1, 1)
        
        if prototype is not None:
            pred_grid = adjust_grid(prototype, target_shape)
        else:
            pred_grid = [[0 for _ in range(target_shape[1])] for _ in range(target_shape[0])]
        
        predictions.append({
            "attempt_1": pred_grid,
            "attempt_2": pred_grid
        })
    return predictions

# Generate improved submission for all evaluation tasks
improved_submission = {}
for task_id, task in eval_challenges.items():
    improved_submission[task_id] = generate_improved_prediction(task)

# Visualize a sample task
sample_task_id = list(eval_challenges.keys())[0]
sample_task = eval_challenges[sample_task_id]
print("Sample Task ID:", sample_task_id)
print("Test Input (Sample):", sample_task.get("test"))
print("Improved Prediction (Sample):", improved_submission[sample_task_id][0])


def validate_submission(submission, challenges):
    errors = []
    for task_id, predictions in submission.items():
        challenge = challenges.get(task_id, {})
        test_inputs = challenge.get("test", [])
        if not isinstance(test_inputs, list):
            test_inputs = [test_inputs]
        if len(predictions) != len(test_inputs):
            errors.append(f"Task {task_id}: Number of predictions ({len(predictions)}) does not match number of test inputs ({len(test_inputs)}).")
        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict) or "attempt_1" not in pred or "attempt_2" not in pred:
                errors.append(f"Task {task_id}, prediction {i}: Missing required keys.")
                continue
            for key in ["attempt_1", "attempt_2"]:
                grid = pred[key]
                if not (isinstance(grid, list) and grid and isinstance(grid[0], list)):
                    errors.append(f"Task {task_id}, prediction {i}, {key}: Not a proper grid (list of lists).")
                else:
                    for row in grid:
                        if any(not isinstance(cell, int) for cell in row):
                            errors.append(f"Task {task_id}, prediction {i}, {key}: Non-integer value found.")
    if errors:
        print("Validation Errors:")
        for err in errors:
            print(err)
    else:
        print("Submission format validated successfully.")

validate_submission(improved_submission, eval_challenges)


# Load evaluation solutions file
eval_solutions_path = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json"
with open(eval_solutions_path, "r") as f:
    eval_solutions = json.load(f)

def exact_match(pred, truth):
    """Return True if two grids are exactly equal."""
    return pred == truth

def compute_score(predictions, solutions):
    total = 0
    correct = 0
    for task_id in solutions:
        truth_outputs = solutions[task_id]  # List of ground truth outputs
        pred_outputs = predictions.get(task_id, [])
        if len(truth_outputs) != len(pred_outputs):
            print(f"Warning: Task {task_id} - number of predictions does not match ground truth.")
            continue
        for truth, pred in zip(truth_outputs, pred_outputs):
            total += 1
            if exact_match(pred.get("attempt_1"), truth) or exact_match(pred.get("attempt_2"), truth):
                correct += 1
    return correct / total if total > 0 else 0

score = compute_score(improved_submission, eval_solutions)
print(f"Baseline Score (Improved Heuristic): {score*100:.2f}%")

submission_path = "submission.json"
with open(submission_path, "w") as f:
    json.dump(improved_submission, f, indent=2)
print(f"Submission file saved to {submission_path}")

