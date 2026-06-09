import json



def rotate90(grid):
    return [list(row) for row in zip(*grid[::-1])]

def rotate180(grid):
    return rotate90(rotate90(grid))

def rotate270(grid):
    return rotate90(rotate180(grid))

def flip_horizontal(grid):
    return [row[::-1] for row in grid]

def flip_vertical(grid):
    return grid[::-1]

def copy_grid(grid):
    return [list(row) for row in grid]

def is_uniform(grid):
    flat = [c for row in grid for c in row]
    return len(set(flat)) == 1


with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_tasks = json.load(f)

with open('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
    train_tasks = json.load(f)

submission = {}


import json

# â€”â€”â€”â€”â€”â€” Helper Transform Functions â€”â€”â€”â€”â€”â€”
def rotate90(grid):
    return [list(row) for row in zip(*grid[::-1])]

def rotate180(grid):
    return rotate90(rotate90(grid))

def rotate270(grid):
    return rotate90(rotate180(grid))

def flip_horizontal(grid):
    return [row[::-1] for row in grid]

def flip_vertical(grid):
    return grid[::-1]

def is_uniform(grid):
    flat = [c for row in grid for c in row]
    return len(set(flat)) == 1 if flat else False

def copy_grid(grid):
    return [list(row) for row in grid]

# â€”â€”â€”â€”â€”â€” Load ARC-AGI-2 Challenges â€”â€”â€”â€”â€”â€”
with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_dict = json.load(f)
with open('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
    train_dict = json.load(f)  # not used now, but available

submission = {}

# â€”â€”â€”â€”â€”â€” Process Each Task â€”â€”â€”â€”â€”â€”
for task_id, task in eval_dict.items():
    train_pairs = task.get('train', [])
    test_pairs = task.get('test', [])

    use_identity = False
    use_fill = False
    use_transform = None

    # Rule 1: Identity mapping
    if train_pairs and all(s['input'] == s['output'] for s in train_pairs):
        use_identity = True
    # Rule 2: Uniform fill with single input color
    elif train_pairs and all(is_uniform(s['output']) for s in train_pairs):
        ok = True
        for s in train_pairs:
            inp = s['input']
            flat = [c for row in inp for c in row if c != 0]
            out_flat = [c for row in s['output'] for c in row]
            if not (out_flat and len(set(out_flat)) == 1 and flat):
                ok = False
                break
            common = max(set(flat), key=flat.count)
            if common != out_flat[0]:
                ok = False
                break
        if ok:
            use_fill = True
    # Rule 3: Consistent transformation
    elif train_pairs:
        for tf in (rotate90, rotate180, rotate270, flip_horizontal, flip_vertical):
            if all(tf(s['input']) == s['output'] for s in train_pairs):
                use_transform = tf
                break

    submission[task_id] = []
    for test in test_pairs:
        inp = test['input']
        attempt1 = copy_grid(inp)

        if use_identity:
            attempt2 = copy_grid(inp)
        elif use_fill:
            flat = [c for row in inp for c in row if c != 0]
            fill = max(set(flat), key=flat.count) if flat else 0
            h, w = len(inp), len(inp[0])
            attempt2 = [[fill]*w for _ in range(h)]
        elif use_transform:
            attempt2 = use_transform(inp)
        else:
            attempt2 = flip_horizontal(inp)

        submission[task_id].append({
            "attempt_1": attempt1,
            "attempt_2": attempt2
        })

# â€”â€”â€”â€”â€”â€” Save Submission â€”â€”â€”â€”â€”â€”
with open('submission.json', 'w') as f:
    json.dump(submission, f)
print(f"âœ… submission.json created with {len(submission)} tasks.")



# ================== Imports ==================
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

# ================== Globals ==================
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ================== Data Location ==================
def find_data_dir():
    kaggle_base = '/kaggle/input'
    for folder in os.listdir(kaggle_base):
        full_path = os.path.join(kaggle_base, folder)
        if os.path.isdir(full_path):
            files = set(os.listdir(full_path))
            if 'arc-agi_test_challenges.json' in files:
                print(f"âœ… Found dataset folder: {full_path}")
                return full_path
    raise FileNotFoundError("â�Œ ARC dataset folder not found in /kaggle/input.")

DATA_DIR = find_data_dir()
print("ğŸ“� Files in DATA_DIR:", os.listdir(DATA_DIR))

# ================== File Paths ==================
TRAIN_SOLUTIONS = os.path.join(DATA_DIR, 'arc-agi_training_solutions.json')
EVAL_CHALLENGES = os.path.join(DATA_DIR, 'arc-agi_evaluation_challenges.json')
EVAL_SOLUTIONS = os.path.join(DATA_DIR, 'arc-agi_evaluation_solutions.json')
TEST_CHALLENGES = os.path.join(DATA_DIR, 'arc-agi_test_challenges.json')

# ================== Grid Utils ==================
def grid_to_numpy(grid): return np.array(grid, dtype=np.int8)
def numpy_to_grid(array): return array.tolist()

def resize_grid(grid, shape):
    h, w = grid.shape
    new_h, new_w = shape
    result = np.zeros((new_h, new_w), dtype=grid.dtype)
    rh, rw = h / new_h, w / new_w
    for i in range(new_h):
        for j in range(new_w):
            result[i, j] = grid[min(h - 1, int(i * rh)), min(w - 1, int(j * rw))]
    return result

# ================== Transforms ==================
def rotate(grid, k=1): return np.rot90(grid, k=k)
def flip(grid, dir='h'): return np.fliplr(grid) if dir == 'h' else np.flipud(grid)

def translate(grid, dx=0, dy=0):
    h, w = grid.shape
    result = np.zeros_like(grid)
    xs, xe = max(0, -dx), w - max(0, dx)
    ys, ye = max(0, -dy), h - max(0, dy)
    xt, yt = max(0, dx), max(0, dy)
    result[yt:yt+(ye-ys), xt:xt+(xe-xs)] = grid[ys:ye, xs:xe]
    return result

def scale(grid, factor=1.0):
    if factor == 1.0:
        return grid.copy()
    new_shape = (max(1, int(grid.shape[0]*factor)), max(1, int(grid.shape[1]*factor)))
    return resize_grid(grid, new_shape)

# ================== Similarity ==================
def extract_features(grid):
    h, w = grid.shape
    feats = [h, w, h*w]
    unique, counts = np.unique(grid, return_counts=True)
    feats.append(len(unique))
    hist = np.zeros(10)
    for u, c in zip(unique, counts):
        if 0 <= u < 10:
            hist[u] = c
    feats.extend(hist / (h*w + 1e-5))
    labeled, num = label(grid > 0)
    if num > 0:
        sizes = [np.sum(labeled == i) for i in range(1, num+1)]
        feats.extend([np.mean(sizes), np.std(sizes), max(sizes), num, np.sum(labeled > 0) / (h*w+1e-5)])
    else:
        feats.extend([0]*5)
    return np.array(feats, dtype=np.float32)

def combined_similarity(g1, g2):
    f1, f2 = extract_features(g1), extract_features(g2)
    sim1 = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0, 0]
    hist1 = np.histogram(g1, bins=range(11))[0]
    hist2 = np.histogram(g2, bins=range(11))[0]
    hist1, hist2 = hist1 / (hist1.sum()+1e-5), hist2 / (hist2.sum()+1e-5)
    sim2 = np.dot(hist1, hist2) / (np.linalg.norm(hist1)*np.linalg.norm(hist2)+1e-5)
    return 0.7 * sim1 + 0.3 * sim2

# ================== Voting ==================
def majority_vote(grids, shape):
    stack = np.stack([resize_grid(g, shape) for g in grids])
    out = np.zeros(shape, dtype=stack.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            vals, counts = np.unique(stack[:, i, j], return_counts=True)
            out[i, j] = vals[np.argmax(counts)]
    return out

# ================== Solver ==================
class ArcSolver:
    def __init__(self, task): self.train, self.test = task['train'], task['test']

    def solve(self):
        return [self.solve_one(grid_to_numpy(p['input'])) for p in self.test]

    def solve_one(self, test_in):
        cands = []
        for flip_dir in ['none', 'h', 'v']:
            base = test_in if flip_dir == 'none' else flip(test_in.copy(), flip_dir)
            for rot in range(4):
                r = rotate(base, rot)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        t = translate(r, dx, dy)
                        for s in [1.0, 0.9, 1.1]:
                            cands.append(scale(t, s))
        scored = []
        for cand in cands:
            best_sim, best_out = -1, None
            for p in self.train:
                train_in = grid_to_numpy(p['input'])
                train_out = grid_to_numpy(p['output'])
                sim = max(combined_similarity(cand, train_in), combined_similarity(cand, train_out))
                if sim > best_sim:
                    best_sim, best_out = sim, train_out
            if best_out is not None:
                scored.append((best_sim, best_out))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [out for _, out in scored[:min(5, len(scored))]]
        pred1 = majority_vote(top, test_in.shape) if top else np.zeros_like(test_in)
        pred2 = resize_grid(scored[0][1], test_in.shape) if scored else test_in.copy()
        return {
            'attempt_1': numpy_to_grid(pred1),
            'attempt_2': numpy_to_grid(pred2)
        }

# ================== Save ==================
def save_submission(sub, name='submission.json'):
    with open(name, 'w') as f: json.dump(sub, f)
    print(f"âœ… Saved {name}")

def save_csv(sub, name='submission.csv'):
    rows = []
    for task_id, preds in sub.items():
        for i, pred in enumerate(preds):
            out = pred['attempt_1']
            flat = '|' + '|'.join([' '.join(map(str, row)) for row in out]) + '|'
            rows.append({'output_id': f"{task_id}_{i}", 'output': flat})
    pd.DataFrame(rows).to_csv(name, index=False)
    print(f"âœ… Saved {name}")

# ================== Main ==================
if __name__ == '__main__':
    # Auto choose test or eval
    try:
        with open(TEST_CHALLENGES) as f:
            challenges = json.load(f)
        mode = 'test'
        print("ğŸ“¦ Mode: TEST")
    except:
        with open(EVAL_CHALLENGES) as f: challenges = json.load(f)
        with open(EVAL_SOLUTIONS) as f: eval_sols = json.load(f)
        mode = 'eval'
        print("ğŸ“¦ Mode: EVALUATION")

    print(f"ğŸ”¢ {len(challenges)} tasks found")
    submission = {}
    for task_id, task in tqdm(challenges.items(), desc="Solving"):
        try:
            solver = ArcSolver(task)
            result = solver.solve()
            submission[task_id] = result
        except Exception as e:
            print(f"âš ï¸� Error in {task_id}: {e}")

    save_submission(submission)
    save_csv(submission)

    if mode == 'eval':
        from IPython.display import display
        def iou(a, b):
            if a.shape != b.shape:
                a = resize_grid(a, b.shape)
            return np.mean([(np.logical_and(a == c, b == c).sum() / np.logical_or(a == c, b == c).sum())
                            for c in np.unique(b) if c != 0]) or float(np.array_equal(a, b))

        score = 0
        total = 0
        for tid in random.sample(list(submission.keys()), min(5, len(submission))):
            print(f"ğŸ”� {tid}")
            task = challenges[tid]
            sols = eval_sols.get(tid, [])
            preds = submission[tid]
            for i, (pair, pred) in enumerate(zip(task['test'], preds)):
                inp = grid_to_numpy(pair['input'])
                pred1 = grid_to_numpy(pred['attempt_1'])
                pred2 = grid_to_numpy(pred['attempt_2'])
                true = grid_to_numpy(sols[i]['output']) if i < len(sols) else np.zeros_like(pred1)
                fig, axs = plt.subplots(1, 4, figsize=(16, 3))
                for ax, grid, label in zip(axs, [inp, pred1, pred2, true], ["Input", "Attempt 1", "Attempt 2", "Truth"]):
                    plot_grid(ax, grid, label)
                plt.show()
                iou1, iou2 = iou(pred1, true), iou(pred2, true)
                print(f"   IOU1: {iou1:.2f} | IOU2: {iou2:.2f}")
                score += max(iou1, iou2)
                total += 1
        if total:
            print(f"ğŸ“Š Mean IOU: {score / total:.4f}")


