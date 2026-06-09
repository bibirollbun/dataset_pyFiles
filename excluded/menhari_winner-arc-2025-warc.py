""" 
ARC Solver with Kaggle Dataset Integration

This script implements a solver for the Abstraction and Reasoning Corpus (ARC) challenge tasks.
The solver uses a combination of grid transformations (rotate, flip, translate, scale)
and similarity measures (feature embedding + color histograms) to match test inputs
against training pairs and generate predicted outputs.

Key features:
- Automatically detects the Kaggle input dataset directory containing ARC JSON files.
- Falls back to a local './data' folder if run outside Kaggle.
- Loads challenge tasks and available solutions (training and evaluation).
- Generates predictions by voting among top similar candidates found via transformations.
- Evaluates predictions with Intersection-over-Union (IoU) scores if ground truths are available.
- Visualizes inputs, predictions, and ground truths side-by-side.
- Saves submission files in JSON and CSV formats ready for Kaggle submission.

Note: Before running on Kaggle, ensure the ARC dataset containing the necessary JSON files
is added to the notebook via the "Add data" button on the right sidebar.
(herman)""" 

import os
import json
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import label
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import pandas as pd

# Set random seed for reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# --- Auto-detect Kaggle input data folder with expected files ---
def find_data_dir():
    kaggle_base = '/kaggle/input'
    expected_files = {
        'arc-agi_evaluation_challenges.json',
        'arc-agi_evaluation_solutions.json',
        'arc-agi_training_solutions.json'
    }
    if os.path.exists(kaggle_base):
        for folder in os.listdir(kaggle_base):
            full_path = os.path.join(kaggle_base, folder)
            if os.path.isdir(full_path):
                files = set(os.listdir(full_path))
                if expected_files.issubset(files):
                    print(f"Found dataset folder: {full_path}")
                    return full_path
    print("Kaggle dataset folder not found, falling back to './data'")
    return './data'

DATA_DIR = find_data_dir()
print("Files found in DATA_DIR:", os.listdir(DATA_DIR))

EVAL_CHALLENGES = os.path.join(DATA_DIR, 'arc-agi_evaluation_challenges.json')
EVAL_SOLUTIONS = os.path.join(DATA_DIR, 'arc-agi_evaluation_solutions.json')
TRAINING_SOLUTIONS = os.path.join(DATA_DIR, 'arc-agi_training_solutions.json')

# ========== Utility Functions ==========
def grid_to_numpy(grid: List[List[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int8)

def numpy_to_grid(array: np.ndarray) -> List[List[int]]:
    return array.tolist()

def resize_grid(grid: np.ndarray, new_shape: Tuple[int,int]) -> np.ndarray:
    old_h, old_w = grid.shape
    new_h, new_w = new_shape
    row_ratio = old_h / new_h
    col_ratio = old_w / new_w
    resized = np.zeros((new_h, new_w), dtype=grid.dtype)
    for r in range(new_h):
        for c in range(new_w):
            src_r = min(int(r * row_ratio), old_h - 1)
            src_c = min(int(c * col_ratio), old_w - 1)
            resized[r, c] = grid[src_r, src_c]
    return resized

# ========== Transformations ==========
def rotate(grid, k=1):
    return np.rot90(grid, k=k)

def flip(grid, direction='h'):
    return np.fliplr(grid) if direction == 'h' else np.flipud(grid)

def translate(grid, dx=0, dy=0):
    shifted = np.zeros_like(grid)
    h, w = grid.shape
    x_start_src = max(0, -dx)
    x_end_src = w - max(0, dx)
    y_start_src = max(0, -dy)
    y_end_src = h - max(0, dy)
    x_start_dst = max(0, dx)
    x_end_dst = w - max(0, -dx)
    y_start_dst = max(0, dy)
    y_end_dst = h - max(0, -dy)
    shifted[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = grid[y_start_src:y_end_src, x_start_src:x_end_src]
    return shifted

def scale(grid, factor=1.0):
    if factor == 1.0:
        return grid.copy()
    h, w = grid.shape
    new_h = max(1, int(h * factor))
    new_w = max(1, int(w * factor))
    return resize_grid(grid, (new_h, new_w))

# ========== Similarity Measures ==========
def color_histogram_similarity(g1: np.ndarray, g2: np.ndarray) -> float:
    max_color = max(g1.max(), g2.max(), 9)
    hist1, _ = np.histogram(g1, bins=np.arange(max_color+2))
    hist2, _ = np.histogram(g2, bins=np.arange(max_color+2))
    hist1 = hist1 / (hist1.sum() + 1e-5)
    hist2 = hist2 / (hist2.sum() + 1e-5)
    return np.dot(hist1, hist2) / (np.linalg.norm(hist1)*np.linalg.norm(hist2) + 1e-5)

def extract_features(grid: np.ndarray) -> np.ndarray:
    features = []
    h, w = grid.shape
    features.extend([h, w, h*w])
    unique_colors, counts = np.unique(grid, return_counts=True)
    features.append(len(unique_colors))
    hist = np.zeros(10)
    for color, count in zip(unique_colors, counts):
        if 0 <= color < 10:
            hist[color] = count
    features.extend(hist / (h * w + 1e-5))
    num_objects = 0
    total_object_area = 0
    if grid.any():
        labeled_array, num_features = label(grid > 0)
        if num_features > 0:
            num_objects = num_features
            object_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
            total_object_area = sum(object_sizes)
            features.extend([np.mean(object_sizes), np.std(object_sizes), max(object_sizes)])
        else:
            features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0])
    features.append(num_objects)
    features.append(total_object_area / (h * w + 1e-5))
    return np.array(features, dtype=np.float32)

def combined_similarity(g1: np.ndarray, g2: np.ndarray) -> float:
    feat1 = extract_features(g1)
    feat2 = extract_features(g2)
    sim_embed = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0, 0]
    sim_hist = color_histogram_similarity(g1, g2)
    return 0.7 * sim_embed + 0.3 * sim_hist

# ========== Voting ==========
def majority_vote(predictions: List[np.ndarray], shape: Tuple[int,int]) -> np.ndarray:
    h, w = shape
    stacked = np.stack([resize_grid(p, shape) for p in predictions])
    final = np.zeros(shape, dtype=stacked.dtype)
    for y in range(h):
        for x in range(w):
            vals, counts = np.unique(stacked[:, y, x], return_counts=True)
            final[y,x] = vals[np.argmax(counts)]
    return final

# ========== Solver ==========
class ArcSolver:
    def __init__(self, task: Dict[str, Any]):
        self.train_pairs = task['train']
        self.test_pairs = task['test']

    def solve(self) -> List[Dict[str, List[List[int]]]]:
        results = []
        for pair in self.test_pairs:
            inp = grid_to_numpy(pair['input'])
            preds = self.advanced_solve(inp)
            results.append({
                'attempt_1': numpy_to_grid(preds[0]),
                'attempt_2': numpy_to_grid(preds[1])
            })
        return results

    def advanced_solve(self, test_in: np.ndarray) -> List[np.ndarray]:
        candidates = []
        for flip_dir in ['none', 'h', 'v']:
            base = test_in.copy()
            if flip_dir != 'none':
                base = flip(base, flip_dir)
            for rot in range(4):
                r = rotate(base, rot)
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        t = translate(r, dx, dy)
                        for scale_factor in [1.0, 0.9, 1.1]:
                            s = scale(t, scale_factor)
                            candidates.append(s)

        scored_candidates = []
        for cand in candidates:
            best_sim = -1
            best_output = None
            for train in self.train_pairs:
                train_in = grid_to_numpy(train['input'])
                train_out = grid_to_numpy(train['output'])
                sim_in = combined_similarity(cand, train_in)
                sim_out = combined_similarity(cand, train_out)
                sim = max(sim_in, sim_out)
                if sim > best_sim:
                    best_sim = sim
                    best_output = train_out
            if best_output is not None:
                scored_candidates.append((best_sim, best_output))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = min(5, len(scored_candidates))
        top_outputs = [out for _, out in scored_candidates[:top_k]]

        if top_outputs:
            final_pred_1 = majority_vote(top_outputs, test_in.shape)
        else:
            final_pred_1 = np.zeros_like(test_in)

        if scored_candidates:
            best_output = scored_candidates[0][1]
            final_pred_2 = resize_grid(best_output, test_in.shape)
        else:
            final_pred_2 = test_in.copy()

        return [final_pred_1, final_pred_2]

# ========== Evaluation and Save ==========
def iou_score(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape != truth.shape:
        pred = resize_grid(pred, truth.shape)
    colors = np.unique(np.concatenate([pred.flatten(), truth.flatten()]))
    colors = colors[colors != 0]
    ious = []
    for c in colors:
        pred_mask = (pred == c)
        truth_mask = (truth == c)
        intersection = np.logical_and(pred_mask, truth_mask).sum()
        union = np.logical_or(pred_mask, truth_mask).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    if len(ious) == 0:
        return 1.0 if np.array_equal(pred, truth) else 0.0
    return np.mean(ious)

def plot_grid(ax, grid, title=""):
    cmap = colors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = colors.Normalize(vmin=0, vmax=9)
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def visualize_and_evaluate(submission: Dict, challenges: Dict, training_solutions: Dict, evaluation_solutions: Dict, max_samples: Optional[int] = 5):
    task_ids = list(submission.keys())
    if max_samples:
        task_ids = random.sample(task_ids, min(max_samples, len(task_ids)))

    total_score = 0
    total_tasks = 0

    for task_id in task_ids:
        task_data = challenges[task_id]
        preds = submission[task_id]
        has_solution = task_id in evaluation_solutions or task_id in training_solutions
        solutions = evaluation_solutions.get(task_id) or training_solutions.get(task_id)

        print(f"--- Task: {task_id} ---")

        for i, test_pair in enumerate(task_data['test']):
            input_grid = grid_to_numpy(test_pair['input'])
            pred_1 = grid_to_numpy(preds[i]['attempt_1'])
            pred_2 = grid_to_numpy(preds[i]['attempt_2'])

            num_plots = 3 + (1 if has_solution else 0)
            fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 4, 4))

            plot_grid(axes[0], input_grid, title=f"Test {i} Input")
            plot_grid(axes[1], pred_1, title="Attempt 1")
            plot_grid(axes[2], pred_2, title="Attempt 2")

            if has_solution:
                sol = solutions[i]
                if isinstance(sol, dict) and 'output' in sol:
                    true_output = grid_to_numpy(sol['output'])
                else:
                    true_output = grid_to_numpy(sol)

                plot_grid(axes[3], true_output, title="True Output")
                score1 = iou_score(pred_1, true_output)
                score2 = iou_score(pred_2, true_output)
                task_score = max(score1, score2)
                total_score += task_score
                print(f"  Test Case {i}: Score = {task_score:.2f} (Attempt 1: {score1:.2f}, Attempt 2: {score2:.2f})")

            plt.tight_layout()
            plt.show()

        total_tasks += len(task_data['test'])

    if total_tasks > 0:
        avg_score = total_score / total_tasks
        print(f"\n--- Overall Average Score: {avg_score:.3f} ---")

# ========== Load/Save ==========





import os
import json
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import label
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import pandas as pd

# Set random seed for reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# --- Auto-detect Kaggle input data folder with expected files ---
def find_data_dir():
    kaggle_base = '/kaggle/input'
    expected_files = {
        'arc-agi_evaluation_challenges.json',
        'arc-agi_evaluation_solutions.json',
        'arc-agi_training_solutions.json'
    }
    if os.path.exists(kaggle_base):
        for folder in os.listdir(kaggle_base):
            full_path = os.path.join(kaggle_base, folder)
            if os.path.isdir(full_path):
                files = set(os.listdir(full_path))
                if expected_files.issubset(files):
                    print(f"Found dataset folder: {full_path}")
                    return full_path
    print("Kaggle dataset folder not found, falling back to './data'")
    return './data'

DATA_DIR = find_data_dir()
print("Files found in DATA_DIR:", os.listdir(DATA_DIR))

EVAL_CHALLENGES = os.path.join(DATA_DIR, 'arc-agi_evaluation_challenges.json')
EVAL_SOLUTIONS = os.path.join(DATA_DIR, 'arc-agi_evaluation_solutions.json')
TRAINING_SOLUTIONS = os.path.join(DATA_DIR, 'arc-agi_training_solutions.json')

# ========== Utility Functions ==========
def grid_to_numpy(grid: List[List[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int8)

def numpy_to_grid(array: np.ndarray) -> List[List[int]]:
    return array.tolist()

def resize_grid(grid: np.ndarray, new_shape: Tuple[int,int]) -> np.ndarray:
    old_h, old_w = grid.shape
    new_h, new_w = new_shape
    row_ratio = old_h / new_h
    col_ratio = old_w / new_w
    resized = np.zeros((new_h, new_w), dtype=grid.dtype)
    for r in range(new_h):
        for c in range(new_w):
            src_r = min(int(r * row_ratio), old_h - 1)
            src_c = min(int(c * col_ratio), old_w - 1)
            resized[r, c] = grid[src_r, src_c]
    return resized

# ... [code omitted for brevity - all code before the last function stays the same] ...

# ========== Load/Save ==========
def save_submission(submission: Dict, filename: str = 'submission.json'):
    with open(filename, 'w') as f:
        json.dump(submission, f)

def save_submission_csv(submission: Dict, filename: str = 'submission.csv'):
    def grid_to_str(grid):
        return '|' + '|'.join([' '.join(map(str, row)) for row in grid]) + '|'

    rows = []
    for task_id, preds in submission.items():
        for i, pred in enumerate(preds):
            output_id = f"{task_id}_{i}"
            output_str = grid_to_str(pred['attempt_1'])
            rows.append({'output_id': output_id, 'output': output_str})

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

def load_arc_data(challenge_file: str, solution_file: Optional[str] = None) -> Tuple[Dict, Optional[Dict]]:
    with open(challenge_file, 'r') as f:
        challenges = json.load(f)
    solutions = None
    if solution_file:
        with open(solution_file, 'r') as f:
            solutions = json.load(f)
    return challenges, solutions

# ========== Main ==========
if __name__ == '__main__':
    print("Loading data...")
    challenges, _ = load_arc_data(EVAL_CHALLENGES)
    training_solution = {}
    evaluation_solution = {}

    try:
        with open(TRAINING_SOLUTIONS, 'r') as f:
            training_solution = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training solutions - {e}")

    try:
        with open(EVAL_SOLUTIONS, 'r') as f:
            evaluation_solution = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load evaluation solutions - {e}")

    print("Solving tasks...")
    submission = {}
    for task_id, task in tqdm(challenges.items(), desc="Solving Tasks"):
        submission[task_id] = ArcSolver(task).solve()

    print("Saving submission files...")
    save_submission(submission)
    save_submission_csv(submission)
    print("âœ… Saved submission.json and submission.csv")

    visualize_and_evaluate(submission, challenges, training_solution, evaluation_solution, max_samples=5)
    print("âœ… Evaluation complete")
    print("Done!")


