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


# ==========================================================
# ARC-AGI-2 Solver v6.1 — Leaderboard Ready
# ==========================================================

import os, json, numpy as np
from collections import Counter

# ---------------------- Utilities --------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)

# ---------------------- Helpers ----------------------------
def invert_colors(grid):
    g = np.array(grid)
    max_c = g.max() if g.size > 0 else 9
    return ((max_c - g) % 10).tolist()

def transpose_grid(grid):
    return np.array(grid).T.tolist()

def mirror_horizontal(grid):
    return np.flip(np.array(grid), axis=1).tolist()

def mirror_vertical(grid):
    return np.flip(np.array(grid), axis=0).tolist()

def recolor_by_frequency(inp, out):
    """Find dominant color mapping by frequency."""
    cin, cout = Counter(np.array(inp).flatten()), Counter(np.array(out).flatten())
    mapping = {}
    for (a, _), (b, _) in zip(cin.most_common(), cout.most_common()):
        if a != b:
            mapping[a] = b
    return mapping

# ---------------------- Apply pipeline ----------------------
def apply_pipeline(grid, pipeline):
    g = np.array(grid)
    for step in pipeline:
        t = step["type"]
        if t == "rotate":
            g = np.rot90(g, step.get("k", 1))
        elif t == "flip_h":
            g = np.flip(g, axis=1)
        elif t == "flip_v":
            g = np.flip(g, axis=0)
        elif t == "transpose":
            g = g.T
        elif t == "invert":
            g = (np.max(g) - g) % 10
        elif t == "recolor":
            cmap = step.get("map", {})
            g = np.vectorize(lambda x: cmap.get(x, x))(g)
    return g.tolist()

# ---------------------- Infer pipeline ----------------------
def infer_pipeline(inp, out):
    inp, out = np.array(inp), np.array(out)
    if inp.shape == out.shape:
        # Try simple global ops
        for k in range(4):
            if np.array_equal(np.rot90(inp, k), out):
                return [{"type": "rotate", "k": k}]
        if np.array_equal(np.flip(inp, 0), out):
            return [{"type": "flip_v"}]
        if np.array_equal(np.flip(inp, 1), out):
            return [{"type": "flip_h"}]
        if np.array_equal(inp.T, out):
            return [{"type": "transpose"}]
        if np.array_equal((np.max(inp) - inp) % 10, out):
            return [{"type": "invert"}]

        # Try recolor
        cmap = recolor_by_frequency(inp, out)
        if cmap:
            recolored = np.vectorize(lambda x: cmap.get(x, x))(inp)
            if np.array_equal(recolored, out):
                return [{"type": "recolor", "map": cmap}]
            else:
                return [{"type": "recolor", "map": cmap}]
    else:
        # Size difference -> maybe cropping/padding
        if inp.shape[0] <= out.shape[0] and inp.shape[1] <= out.shape[1]:
            return [{"type": "recolor", "map": recolor_by_frequency(inp, out)}]
    return []

# ---------------------- Validate pipeline ----------------------
def validate_pipeline(task, pipeline):
    matches = 0
    for pair in task["train"]:
        pred = np.array(apply_pipeline(pair["input"], pipeline))
        true = np.array(pair["output"])
        if pred.shape == true.shape:
            if np.mean(pred == true) > 0.9:
                matches += 1
    return matches / len(task["train"])

# ---------------------- Solver ----------------------
def solve_task(task):
    pipelines = []
    for pair in task["train"]:
        pipe = infer_pipeline(pair["input"], pair["output"])
        if pipe:
            pipelines.append(pipe)
    if not pipelines:
        # fallback guesses
        return [{
            "attempt_1": mirror_horizontal(task["test"][0]["input"]),
            "attempt_2": invert_colors(task["test"][0]["input"])
        }]

    best_pipe, best_score = None, -1
    for p in pipelines:
        s = validate_pipeline(task, p)
        if s > best_score:
            best_pipe, best_score = p, s

    if not best_pipe:
        return [{
            "attempt_1": task["test"][0]["input"],
            "attempt_2": transpose_grid(task["test"][0]["input"])
        }]

    preds = []
    for t in task["test"]:
        preds.append({
            "attempt_1": apply_pipeline(t["input"], best_pipe),
            "attempt_2": mirror_vertical(t["input"])
        })
    return preds

# ---------------------- Main ----------------------
def main():
    DATA_DIR = "/kaggle/input/arc-prize-2025" if os.path.exists("/kaggle/input") else "."
    TEST_PATH = f"{DATA_DIR}/arc-agi_test_challenges.json"
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = load_json(TEST_PATH)
    print(f"Loaded {len(tasks)} test tasks.")

    submission = {}
    for tid, task in tasks.items():
        print(f"\n=== Solving {tid} ===")
        preds = solve_task(task)
        submission[tid] = preds

    save_path = os.path.join(OUTPUT_DIR, "submission.json")
    save_json(submission, save_path)
    print(f"\n✅ Submission saved to {save_path}")

if __name__ == "__main__":
    main()





