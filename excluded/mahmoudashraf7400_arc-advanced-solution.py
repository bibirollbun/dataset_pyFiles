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
import os
import glob
import numpy as np
import heapq
from typing import List, Callable, Tuple
from scipy.optimize import linear_sum_assignment  # For Hungarian

# Union-Find for Blob Labeling (Perception)
class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, p: int) -> int:
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p: int, q: int):
        rootP, rootQ = self.find(p), self.find(q)
        if rootP == rootQ:
            return
        if self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
        elif self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
        else:
            self.parent[rootQ] = rootP
            self.rank[rootP] += 1

def label_blobs(grid: np.ndarray) -> List[dict]:
    """Label connected components (blobs) using Union-Find."""
    h, w = grid.shape
    uf = UnionFind(h * w)
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0: continue  # Background
            idx = i * w + j
            if j + 1 < w and grid[i, j] == grid[i, j + 1]:
                uf.union(idx, idx + 1)
            if i + 1 < h and grid[i, j] == grid[i + 1, j]:
                uf.union(idx, idx + w)
    blobs = {}
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0:
                root = uf.find(i * w + j)
                if root not in blobs:
                    blobs[root] = {'color': grid[i, j], 'pixels': []}
                blobs[root]['pixels'].append((i, j))
    return list(blobs.values())

# Expanded DSL Primitives (More for complex rules)
def rotate(grid: List[List[int]], angle: int) -> List[List[int]]:
    arr = np.array(grid)
    return np.rot90(arr, k=angle // 90 % 4).tolist()

def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    return np.fliplr(np.array(grid)).tolist()

def change_color(grid: List[List[int]], old_color: int, new_color: int) -> List[List[int]]:
    arr = np.array(grid)
    arr[arr == old_color] = new_color
    return arr.tolist()

def fill_border(grid: List[List[int]], color: int) -> List[List[int]]:
    arr = np.array(grid)
    h, w = arr.shape
    arr[0, :] = color; arr[-1, :] = color; arr[:, 0] = color; arr[:, -1] = color
    return arr.tolist()

def scale_grid(grid: List[List[int]], factor: int) -> List[List[int]]:
    arr = np.array(grid)
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1).tolist()

def mosaic(grid: List[List[int]], reps: int) -> List[List[int]]:
    arr = np.array(grid)
    return np.tile(arr, (reps, reps)).tolist()

def extract_subgrid(grid: List[List[int]], blob: dict) -> List[List[int]]:
    """Extract subgrid from blob pixels."""
    pixels = blob['pixels']
    min_i, max_i = min(p[0] for p in pixels), max(p[0] for p in pixels)
    min_j, max_j = min(p[1] for p in pixels), max(p[1] for p in pixels)
    sub = np.zeros((max_i - min_i + 1, max_j - min_j + 1), dtype=int)
    for i, j in pixels:
        sub[i - min_i, j - min_j] = blob['color']
    return sub.tolist()

# Parameters
ANGLES = [90, 180, 270]
COLORS = range(10)
FACTORS = [2, 3]
REPS = [2, 3]

OPERATIONS = [
    (rotate, [int]), (flip_horizontal, []), (change_color, [int, int]),
    (fill_border, [int]), (scale_grid, [int]), (mosaic, [int]),
    # Add more...
]

import itertools

def apply_sequence(grid: List[List[int]], sequence: List[Tuple[Callable, Tuple]], blobs: List[dict] = None) -> List[List[int]]:
    current = [row[:] for row in grid]
    for func, params in sequence:
        if func == extract_subgrid:
            if blobs:
                current = func(current, blobs[0])  # Example for first blob
        else:
            current = func(current, *params)
    return current

def grids_equal(g1, g2): return np.array_equal(np.array(g1), np.array(g2))

def feature_vector(grid: np.ndarray, blobs: List[dict]) -> np.ndarray:
    """Simple 5-D feature vector (expand to 50-D as in aviad12g)."""
    fv = np.zeros(5)
    fv[0] = len(blobs)  # Num blobs
    if blobs:
        fv[1] = np.mean([len(b['pixels']) for b in blobs])  # Avg size
    return fv

def hungarian_cost(fv1: np.ndarray, fv2: np.ndarray) -> float:
    """Hungarian for assignment cost (simplified)."""
    cost_matrix = np.abs(fv1[:, None] - fv2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()

def heuristic(g1: List[List[int]], g2: List[List[int]]) -> float:
    arr1, arr2 = np.array(g1), np.array(g2)
    blobs1, blobs2 = label_blobs(arr1), label_blobs(arr2)
    fv1, fv2 = feature_vector(arr1, blobs1), feature_vector(arr2, blobs2)
    euclid = np.linalg.norm(fv1 - fv2)
    hung = hungarian_cost(fv1.reshape(-1, 1), fv2.reshape(-1, 1)) if len(fv1) > 0 else 0
    return 0.5 * euclid + 0.5 * hung

def a_star_search(train_pairs: List[dict], max_depth: int = 5) -> List[List[Tuple[Callable, Tuple]]]:
    start = ([], 0.0)
    pq = []
    heapq.heappush(pq, (0.0, [], 0.0))  # priority, seq, cost
    visited = set()
    best_sequences = []

    while pq:
        _, seq, cost = heapq.heappop(pq)
        if tuple(map(id, seq)) in visited: continue  # Approx tuple for funcs
        visited.add(tuple(map(id, seq)))

        if len(seq) > max_depth: continue

        matches = 0
        for pair in train_pairs:
            input_grid = pair['input']
            blobs = label_blobs(np.array(input_grid))
            transformed = apply_sequence(input_grid, seq, blobs)
            if grids_equal(transformed, pair['output']):
                matches += 1
        if matches == len(train_pairs):
            best_sequences.append(seq)
            if len(best_sequences) >= 2: return best_sequences

        for op, param_types in OPERATIONS:
            param_options = [ANGLES if op == rotate else FACTORS if op in [scale_grid, mosaic] else COLORS for _ in param_types]
            for params in itertools.product(*param_options) if param_options else [()]:
                new_seq = seq + [(op, params)]
                new_cost = cost + 1
                h = max(heuristic(apply_sequence(pair['input'], new_seq), pair['output']) for pair in train_pairs)
                priority = new_cost + h
                heapq.heappush(pq, (priority, new_seq, new_cost))

    return best_sequences

# Main (Kaggle setup)
test_path = '/kaggle/input/arc-prize-2025/arc-agi_test_v2/'
test_files = glob.glob(os.path.join(test_path, '*.json'))
submission = {}

for file in test_files:
    task_id = os.path.basename(file).rstrip('.json')
    with open(file, 'r') as f:
        task = json.load(f)
    train_pairs = task.get('train', [])
    test_pairs = task['test']
    best_sequences = a_star_search(train_pairs)
    task_predictions = []
    for test_pair in test_pairs:
        input_grid = test_pair['input']
        blobs = label_blobs(np.array(input_grid))
        attempts = [apply_sequence(input_grid, seq, blobs) for seq in best_sequences]
        while len(attempts) < 2:
            attempts.append([row[:] for row in input_grid])
        task_predictions.append({'attempt_1': attempts[0], 'attempt_2': attempts[1]})
    submission[task_id] = task_predictions

with open('submission.json', 'w') as f:
    json.dump(submission, f)

print("Enhanced submission with perception and heuristics generated.")

