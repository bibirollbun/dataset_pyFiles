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


import os
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from   matplotlib import colors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path


DATA_DIR = "/mnt/data"
TRAIN_FILE = os.path.join(DATA_DIR, "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json")
TRAIN_SOL_FILE = os.path.join(DATA_DIR, "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json")
EVAL_FILE = os.path.join(DATA_DIR, "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json")
EVAL_SOL_FILE = os.path.join(DATA_DIR, "/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json")
TEST_FILE = os.path.join(DATA_DIR, "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json")
SUBMISSION_TEMPLATE = os.path.join(DATA_DIR, "/kaggle/input/arc-prize-2025/sample_submission.json")


def plot_grid(grid: List[List[int]], title=""):
    plt.imshow(grid, cmap="tab10", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(ticks=range(10))
    plt.show()

def plot_task(task: Dict, task_id: str):
    print(f"=== Task ID: {task_id} ===")
    n_train = len(task["train"])
    n_test = len(task["test"])
    fig, axs = plt.subplots(nrows=n_train + n_test, ncols=2, figsize=(6, 3 * (n_train + n_test)))

    for i, pair in enumerate(task["train"]):
        axs[i, 0].imshow(pair["input"], cmap="tab10", interpolation="nearest")
        axs[i, 0].set_title(f"Train Input {i}")
        axs[i, 1].imshow(pair["output"], cmap="tab10", interpolation="nearest")
        axs[i, 1].set_title(f"Train Output {i}")
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")

    for j, test in enumerate(task["test"]):
        axs[n_train + j, 0].imshow(test["input"], cmap="tab10", interpolation="nearest")
        axs[n_train + j, 0].set_title(f"Test Input {j}")
        axs[n_train + j, 0].axis("off")
        axs[n_train + j, 1].axis("off")

    plt.tight_layout()
    plt.show()


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

train_data = load_json(TRAIN_FILE)
train_solutions = load_json(TRAIN_SOL_FILE)
eval_data = load_json(EVAL_FILE)
eval_solutions = load_json(EVAL_SOL_FILE)
test_data = load_json(TEST_FILE)


def run_eda(data_dict: Dict, name: str):
    print(f"\n=== EDA for {name} ===")
    n_tasks = len(data_dict)
    print(f"Total tasks: {n_tasks}")

    grid_sizes = []
    color_counts = Counter()

    for task in data_dict.values():
        for pair in task["train"]:
            inp = np.array(pair["input"])
            out = np.array(pair["output"])
            grid_sizes.append(inp.shape)
            color_counts.update(inp.flatten())
            color_counts.update(out.flatten())
        for test in task["test"]:
            inp = np.array(test["input"])
            grid_sizes.append(inp.shape)
            color_counts.update(inp.flatten())

    sizes = Counter(grid_sizes)
    print("\nMost common grid sizes:")
    for size, count in sizes.most_common(5):
        print(f"  {size}: {count} tasks")

    print("\nTop 10 most frequent colors:")
    for color, count in color_counts.most_common(10):
        print(f"  Color {color}: {count} times")

    plt.bar(color_counts.keys(), color_counts.values(), color='tab:blue')
    plt.xticks(range(10))
    plt.title(f"Color Frequency in {name}")
    plt.xlabel("Color Index")
    plt.ylabel("Frequency")
    plt.show()


run_eda(train_data, "Training Set")
run_eda(eval_data, "Evaluation Set")
run_eda(test_data, "Test Set")


example_task_id = list(train_data.keys())[0]
plot_task(train_data[example_task_id], example_task_id)


def rule_based_predict(input_grid):
    return input_grid

def pattern_match_predict(task: Dict):
    train_outputs = [pair["output"] for pair in task["train"]]
    shapes = [np.array(out).shape for out in train_outputs]
    if all(shape == shapes[0] for shape in shapes):
        common_out = train_outputs[0]
        return lambda x: common_out
    else:
        return rule_based_predict


def predict_task(task: Dict):
    predictions = []
    predictor = pattern_match_predict(task)
    for test in task["test"]:
        inp = test["input"]
        pred1 = predictor(inp)
        pred2 = predictor(inp)
        predictions.append({"attempt_1": pred1, "attempt_2": pred2})
    return predictions


def generate_submission(test_tasks):
    submission = {}
    for task_id, task in test_tasks.items():
        submission[task_id] = predict_task(task)
    return submission

def save_submission(submission_dict, path="submission.json"):
    with open(path, "w") as f:
        json.dump(submission_dict, f)
    print(f"Submission saved to {path}")


def infer_rule(train_pairs):
    """Simple rule inference: check if pattern is tiling, mirroring, or color swap"""
    in_grid = np.array(train_pairs[0]['input'])
    out_grid = np.array(train_pairs[0]['output'])
    if out_grid.shape[0] % in_grid.shape[0] == 0 and out_grid.shape[1] % in_grid.shape[1] == 0:
        return lambda x: np.tile(x, (out_grid.shape[0] // x.shape[0], out_grid.shape[1] // x.shape[1])).tolist()
    return lambda x: x


def plot_grid(grid, title=""):
    plt.imshow(grid, cmap="tab10", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(ticks=range(10))
    plt.show()


def infer_rule(train_pairs):
    """Simple rule inference: check if pattern is tiling, mirroring, or color swap"""
    in_grid = np.array(train_pairs[0]['input'])
    out_grid = np.array(train_pairs[0]['output'])
    if out_grid.shape[0] % in_grid.shape[0] == 0 and out_grid.shape[1] % in_grid.shape[1] == 0:
        return lambda x: np.tile(x, (out_grid.shape[0] // x.shape[0], out_grid.shape[1] // x.shape[1])).tolist()
    return lambda x: x


class ARCGridDataset(Dataset):
    def __init__(self, data_dict, sol_dict):
        self.inputs = []
        self.outputs = []
        for k in data_dict:
            task = data_dict[k]
            solutions = sol_dict[k]
            for pair, out in zip(task['train'], solutions):
                inp = np.array(pair['input'])
                out = np.array(out)
                if inp.shape == out.shape:
                    self.inputs.append(inp)
                    self.outputs.append(out)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.outputs[idx], dtype=torch.long)
        return x, y

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 16)
        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.head = nn.Conv2d(32, 10, 1)

    def forward(self, x):
        x = self.embed(x).permute(0, 3, 1, 2) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.head(x)



def hybrid_predict(task):
    rule = infer_rule(task['train'])
    predictions = []
    for test in task['test']:
        input_grid = np.array(test['input'])
        rule_pred = rule(input_grid)
        plot_grid(input_grid, "Test Input")
        plot_grid(rule_pred, "Predicted Output (Rule-Based)")
        predictions.append({"attempt_1": rule_pred, "attempt_2": rule_pred})
    return predictions


def build_submission(dataset):
    submission = {}
    for task_id, task in dataset.items():
        print(f"Visualizing task {task_id}...")
        submission[task_id] = hybrid_predict(task)
    return submission

def save_submission(preds, filename="submission_hybrid.json"):
    with open(filename, "w") as f:
        json.dump(preds, f)
    print("Saved:", filename)


submission = build_submission(dict(list(train_data.items())[:3]))
save_submission(submission)


def plot_grid(grid, title="", ax=None):
    cmap = plt.get_cmap("tab10") 
    arr = np.array(grid)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title)
    ax.axis("off")

task_ids = list(train_data.keys())[:10]

for idx, task_id in enumerate(task_ids):
    task = train_data[task_id]
    solution = train_solutions[task_id]
    print(f"\nTask {idx} - ID: {task_id}")
    print(f"Number of train pairs: {len(task['train'])}")
    print(f"Number of test inputs: {len(task['test'])}")

    n_rows = len(task['train']) + len(task['test'])
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 3 * n_rows))

    for i, pair in enumerate(task['train']):
        plot_grid(pair['input'], f"Train {i+1} - Input", axes[i, 0])
        plot_grid(pair['output'], f"Train {i+1} - Output", axes[i, 1])

    for j, test_pair in enumerate(task['test']):
        plot_grid(test_pair['input'], f"Test {j+1} - Input", axes[len(task['train']) + j, 0])
        plot_grid(solution[j], f"Test {j+1} - GT Output", axes[len(task['train']) + j, 1])

    plt.tight_layout()
    plt.suptitle(f"Task ID: {task_id}", fontsize=16, y=1.02)
    plt.show()

all_colors = []
for i, (task_id, task) in enumerate(train_data.items()):
    if i >= 1000:
        break
    for pair in task["train"]:
        all_colors.extend(np.array(pair["input"]).flatten())

color_counts = Counter(all_colors)

plt.figure(figsize=(8, 4))
sns.barplot(x=list(color_counts.keys()), y=list(color_counts.values()), palette="tab10")
plt.title("Distribusi Warna (0–9) dalam 1000 Task Pertama")
plt.xlabel("Warna")
plt.ylabel("Frekuensi")
plt.show()

