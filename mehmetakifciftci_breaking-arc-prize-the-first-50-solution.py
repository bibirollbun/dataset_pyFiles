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

ARC_COLORMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF']
)
ARC_NORM = colors.Normalize(vmin=0, vmax=10)

def show_grid(grid, title=None, figsize=None):
    if not figsize:
        figsize = (len(grid[0]) * 0.6, len(grid) * 0.6)
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap=ARC_COLORMAP, norm=ARC_NORM)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if title:
        plt.title(title, fontsize=12)
    plt.show()



class ARCDataset:
    def __init__(self, train_path=None, train_solutions_path=None, 
                       test_path=None, 
                       eval_path=None, eval_solutions_path=None):
        self.train_data = self._load_json(train_path) if train_path else {}
        self.train_solutions = self._load_json(train_solutions_path) if train_solutions_path else {}
        self.test_data = self._load_json(test_path) if test_path else {}
        self.eval_data = self._load_json(eval_path) if eval_path else {}
        self.eval_solutions = self._load_json(eval_solutions_path) if eval_solutions_path else {}

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



def visualize_task(task_data, task_solutions=None, title="ARC Task", figsize=(12, 6)):
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    has_solution = task_solutions is not None

    num_train = len(train_examples)
    num_test = len(test_examples)
    cols = num_train + num_test

    fig, axs = plt.subplots(2, cols, figsize=figsize)
    plt.suptitle(title, fontsize=16)

    for idx, example in enumerate(train_examples):
        axs[0, idx].imshow(example['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0, idx].set_title("Train Input")
        axs[0, idx].axis('off')
        axs[1, idx].imshow(example['output'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[1, idx].set_title("Train Output")
        axs[1, idx].axis('off')

    for idx, example in enumerate(test_examples):
        axs[0, num_train + idx].imshow(example['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0, num_train + idx].set_title("Test Input")
        axs[0, num_train + idx].axis('off')

        if has_solution:
            axs[1, num_train + idx].imshow(task_solutions[idx], cmap=ARC_COLORMAP, norm=ARC_NORM)
            axs[1, num_train + idx].set_title("Test Output")
        else:
            axs[1, num_train + idx].set_title("Test Output: ?")
        axs[1, num_train + idx].axis('off')

    plt.tight_layout()
    plt.show()



DATA_PATH = '/kaggle/input/arc-prize-2025'
dataset = ARCDataset(
    train_path=f'{DATA_PATH}/arc-agi_training_challenges.json',
    train_solutions_path=f'{DATA_PATH}/arc-agi_training_solutions.json',
    test_path=f'{DATA_PATH}/arc-agi_test_challenges.json',
    eval_path=f'{DATA_PATH}/arc-agi_evaluation_challenges.json',
    eval_solutions_path=f'{DATA_PATH}/arc-agi_evaluation_solutions.json',
)

task_data, task_solution = dataset.get_task('00576224', split='train')
visualize_task(task_data, task_solution, title='Task 00576224')



def analyze_dataset(dataset):
    stats = {
        'task_id': [],
        'num_train_pairs': [],
        'num_test_pairs': [],
        'input_shapes': [],
        'output_shapes': [],
        'max_colors': [],
        'any_input_equals_output': [],
    }

    for task_id, task in dataset.train_data.items():
        train_examples = task['train']
        test_examples = task['test']
        
        all_inputs = [np.array(e['input']) for e in train_examples]
        all_outputs = [np.array(e['output']) for e in train_examples]

        input_shapes = [arr.shape for arr in all_inputs]
        output_shapes = [arr.shape for arr in all_outputs]

        all_colors = set()
        for a, b in zip(all_inputs, all_outputs):
            all_colors.update(np.unique(a))
            all_colors.update(np.unique(b))

        any_equal = any(np.array_equal(a, b) for a, b in zip(all_inputs, all_outputs))

        stats['task_id'].append(task_id)
        stats['num_train_pairs'].append(len(train_examples))
        stats['num_test_pairs'].append(len(test_examples))
        stats['input_shapes'].append(input_shapes)
        stats['output_shapes'].append(output_shapes)
        stats['max_colors'].append(len(all_colors))
        stats['any_input_equals_output'].append(any_equal)

    return pd.DataFrame(stats)



def plot_grid_size_distribution(df):
    input_sizes = df['input_shapes'].apply(lambda x: (x[0], x[1]))
    input_w = input_sizes.apply(lambda x: x[1])
    input_h = input_sizes.apply(lambda x: x[0])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(input_w, bins=20)
    plt.title('Input Grid Widths')
    plt.xlabel('Width')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.hist(input_h, bins=20)
    plt.title('Input Grid Heights')
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()



def plot_color_distribution(df):
    plt.figure(figsize=(6, 4))
    plt.hist(df['max_colors'], bins=range(1, 13))
    plt.title('Color Count per Task')
    plt.xlabel('Number of Unique Colors')
    plt.ylabel('Count')
    plt.xticks(range(1, 12))
    plt.show()



def plot_input_output_equality(df):
    equal = df['any_input_equals_output'].value_counts()
    plt.figure(figsize=(4, 3))
    equal.plot(kind='bar', color=['green', 'red'])
    plt.title('Input == Output (in train examples)')
    plt.xticks(ticks=[0, 1], labels=['Not Equal', 'Equal'], rotation=0)
    plt.ylabel('Count')
    plt.show()



eda_df = analyze_dataset(dataset)



plot_grid_size_distribution(eda_df)


plot_color_distribution(eda_df)


plot_input_output_equality(eda_df)


def generate_dummy_submission(test_data, filename='submission.json'):
    submission = {}

    for task_id, task in test_data.items():
        task_outputs = []
        for test_case in task['test']:
            input_grid = np.array(test_case['input'])
            h, w = input_grid.shape

            dummy_output_1 = [[0 for _ in range(w)] for _ in range(h)]
            dummy_output_2 = [[0 for _ in range(w)] for _ in range(h)]

            task_outputs.append({
                "attempt_1": dummy_output_1,
                "attempt_2": dummy_output_2
            })

        submission[task_id] = task_outputs

    # Save to the correct path so Kaggle captures it
    output_path = os.path.join('/kaggle/working', filename)

    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"submission.json successfully saved to: {output_path}")



generate_dummy_submission(dataset.test_data)

