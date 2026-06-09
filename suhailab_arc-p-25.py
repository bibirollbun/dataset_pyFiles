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


# =========================
# ARC Prize 2025 — Cell 1: Setup, I/O, Utilities
# =========================
# - Fast imports
# - Reproducibility (seeds, deterministic flags)
# - Auto-discover competition files in /kaggle/input/**
# - JSON/CSV loaders for ARC-AGI-2 tasks
# - Pretty grid visualizer
# - Safe submission scaffolding (solver stub to be filled later)

from __future__ import annotations
import os, sys, json, math, random, time, gc, glob, itertools, functools, textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Optional: plotting helpers (handy for quick debug/EDA)
import matplotlib.pyplot as plt

# -------------------------
# Reproducibility
# -------------------------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Paths & file discovery
# -------------------------
INPUT_ROOT = Path("/kaggle/input")
assert INPUT_ROOT.exists(), "Expected Kaggle input directory '/kaggle/input' not found."

# Find all candidate JSON/CSV files shipped with the comp
json_files = sorted(glob.glob(str(INPUT_ROOT / "**" / "*.json"), recursive=True))
csv_files  = sorted(glob.glob(str(INPUT_ROOT / "**" / "*.csv"),  recursive=True))

print(f"Discovered {len(json_files)} JSON and {len(csv_files)} CSV files under /kaggle/input.")
for f in json_files[:6]:
    print("JSON:", f)
for f in csv_files[:6]:
    print("CSV :", f)

# Heuristics to guess common filenames (robust to path variations)
def pick_first_matching(names: List[str], candidates: List[str]) -> Optional[str]:
    names_lower = [n.lower() for n in names]
    for cand in candidates:
        for n in names_lower:
            if cand in n:
                # return original path with correct case
                return names[names_lower.index(n)]
    return None

# Typical ARC-AGI-2 file patterns (these may vary; we try to be permissive)
TRAIN_JSON = pick_first_matching(
    json_files,
    ["train.json", "training.json", "arc-agi-2-train.json", "arc_agi_2_train.json"]
)
PUBLIC_EVAL_JSON = pick_first_matching(
    json_files,
    ["public_eval.json", "public-eval.json", "public.json", "arc-agi-2-public_eval.json"]
)
SEMI_PRIVATE_JSON = pick_first_matching(
    json_files,
    ["semi_private.json", "semi-private.json", "semi.json", "arc-agi-2-semi_private.json"]
)
SAMPLE_SUB_CSV = pick_first_matching(
    csv_files,
    ["sample_submission.csv", "sample-submission.csv", "sample.csv"]
)

print("\nGuessed files:")
print("  TRAIN_JSON       :", TRAIN_JSON)
print("  PUBLIC_EVAL_JSON :", PUBLIC_EVAL_JSON)
print("  SEMI_PRIVATE_JSON:", SEMI_PRIVATE_JSON)
print("  SAMPLE_SUB_CSV   :", SAMPLE_SUB_CSV)

# -------------------------
# ARC Task Structures
# -------------------------
# ARC-AGI-2 tasks are typically dicts with:
#   {
#     "task_id": "...",
#     "train": [{"input": [[...]], "output": [[...]]}, ...],
#     "test":  [{"input": [[...]]}, ...]
#   }
# We'll load them into Python-friendly types.

Grid = List[List[int]]

def load_task_list(json_path: str) -> List[Dict[str, Any]]:
    """Load a list of ARC tasks from a JSON file."""
    if json_path is None:
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    # Some distributions wrap with {"tasks": [...]}; handle both
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    else:
        tasks = data
    # Normalize minimal fields
    for t in tasks:
        t.setdefault("task_id", t.get("id", None))
        # Validate shapes lightly
        for pair in t.get("train", []):
            assert "input" in pair and "output" in pair, "Malformed train pair."
        for pair in t.get("test", []):
            assert "input" in pair, "Malformed test pair."
    return tasks

TRAIN_TASKS = load_task_list(TRAIN_JSON) if TRAIN_JSON else []
PUBLIC_TASKS = load_task_list(PUBLIC_EVAL_JSON) if PUBLIC_EVAL_JSON else []
SEMI_TASKS   = load_task_list(SEMI_PRIVATE_JSON) if SEMI_PRIVATE_JSON else []

print(f"\nLoaded counts — train: {len(TRAIN_TASKS)}, public_eval: {len(PUBLIC_TASKS)}, semi_private: {len(SEMI_TASKS)}")

# -------------------------
# Display utilities
# -------------------------
def show_grid(grid: Grid, ax=None, title: Optional[str]=None, show: bool=True):
    """Visualize an ARC grid (ints 0..9) with matplotlib."""
    arr = np.array(grid, dtype=int)
    if ax is None:
        fig, ax = plt.subplots(figsize=(min(6, 0.35*arr.shape[1]+1.5), min(6, 0.35*arr.shape[0]+1.5)))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.grid(which="both", linestyle=":", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=12)
    if show:
        plt.show()

def show_task(task: Dict[str, Any], max_pairs: int = 2):
    """Plot a few train input/output pairs and first test input."""
    n = min(max_pairs, len(task.get("train", [])))
    cols = max(2, n * 2)
    fig, axes = plt.subplots(1, cols, figsize=(3.0*cols, 3.0))
    if cols == 2:
        axes = np.array(axes).reshape(1, -1)[0]
    idx = 0
    for i in range(n):
        show_grid(task["train"][i]["input"], ax=axes[idx],   title=f"Train {i} — In", show=False); idx += 1
        show_grid(task["train"][i]["output"], ax=axes[idx],  title=f"Train {i} — Out", show=False); idx += 1
    # Test preview
    if "test" in task and len(task["test"]) > 0:
        show_grid(task["test"][0]["input"], ax=axes[idx % cols], title="Test 0 — In", show=False)
    plt.tight_layout()
    plt.show()

# -------------------------
# Tiny schema helpers
# -------------------------
def grid_shape(g: Grid) -> Tuple[int,int]:
    a = np.array(g, dtype=int)
    return a.shape[0], a.shape[1]

def to_numpy(g: Grid) -> np.ndarray:
    return np.array(g, dtype=int)

def from_numpy(a: np.ndarray) -> Grid:
    return a.astype(int).tolist()

# -------------------------
# Solver scaffold (to fill later)
# -------------------------
def solve_task(task: Dict[str, Any]) -> List[Grid]:
    """
    RETURN a list of predicted grids (one per 'test' item).
    This is a placeholder. We'll implement real reasoning in later cells.
    """
    preds: List[Grid] = []
    for test_item in task.get("test", []):
        inp = to_numpy(test_item["input"])
        # --- TODO: replace with actual reasoning ---
        # As a safe baseline, echo input (NEVER good, but keeps shapes valid)
        preds.append(from_numpy(inp.copy()))
    return preds

# -------------------------
# Submission scaffolding
# -------------------------
def make_empty_submission(sample_sub_csv: Optional[str]) -> pd.DataFrame:
    """
    Load sample submission if provided; otherwise create a generic empty frame.
    The exact schema depends on competition. We keep flexible and fill later.
    """
    if sample_sub_csv and os.path.exists(sample_sub_csv):
        sub = pd.read_csv(sample_sub_csv)
        print("Loaded sample_submission with columns:", list(sub.columns))
        return sub.copy()
    else:
        print("No sample_submission.csv found — creating a placeholder DataFrame.")
        return pd.DataFrame()

def write_submission(df: pd.DataFrame, filename: str = "submission.csv"):
    df.to_csv(filename, index=False)
    print(f"Saved submission -> {filename} | shape={df.shape}")

SAMPLE_SUB = make_empty_submission(SAMPLE_SUB_CSV)

print("\nSetup complete ✅ — Ready to explore TRAIN_TASKS / PUBLIC_TASKS.\n"
      "Tip: try `show_task(TRAIN_TASKS[0])` to visualize an example.\n"
      "Next: we’ll build a fast pattern-mining solver and replace the echo-baseline.")



# =========================
# ARC Prize 2025 — Fixed Setup
# =========================
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input/arc-prize-2025")

# Explicit paths (we know the filenames now)
TRAIN_CHALLENGES   = INPUT_ROOT / "arc-agi_training_challenges.json"
TRAIN_SOLUTIONS    = INPUT_ROOT / "arc-agi_training_solutions.json"
TEST_CHALLENGES    = INPUT_ROOT / "arc-agi_test_challenges.json"
EVAL_CHALLENGES    = INPUT_ROOT / "arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS     = INPUT_ROOT / "arc-agi_evaluation_solutions.json"
SAMPLE_SUB_JSON    = INPUT_ROOT / "sample_submission.json"

print("Files found:")
for f in [TRAIN_CHALLENGES, TRAIN_SOLUTIONS, TEST_CHALLENGES,
          EVAL_CHALLENGES, EVAL_SOLUTIONS, SAMPLE_SUB_JSON]:
    print(" -", f, "| exists:", f.exists())

# -------------------------
# Loaders
# -------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

train_challenges = load_json(TRAIN_CHALLENGES)
train_solutions  = load_json(TRAIN_SOLUTIONS)
test_challenges  = load_json(TEST_CHALLENGES)
eval_challenges  = load_json(EVAL_CHALLENGES)
eval_solutions   = load_json(EVAL_SOLUTIONS)
sample_sub       = load_json(SAMPLE_SUB_JSON)

print("\nLoaded sizes:")
print("  train_challenges :", len(train_challenges))
print("  train_solutions  :", len(train_solutions))
print("  test_challenges  :", len(test_challenges))
print("  eval_challenges  :", len(eval_challenges))
print("  eval_solutions   :", len(eval_solutions))
print("  sample_sub keys  :", list(sample_sub.keys()))

# -------------------------
# Visualizer
# -------------------------
def show_grid(grid, ax=None, title=None, show=True):
    arr = np.array(grid, dtype=int)
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.4*arr.shape[1]+2, 0.4*arr.shape[0]+2))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.grid(which="both", linestyle=":", linewidth=0.5)
    if title:
        ax.set_title(title)
    if show:
        plt.show()

def show_task(task, sol=None, max_pairs=2):
    """Plot task input(s) + optional solution(s)."""
    ins = task.get("train", [])[:max_pairs]
    cols = 2 * len(ins)
    fig, axes = plt.subplots(1, cols, figsize=(3*cols, 3))
    if cols == 2: axes = [axes]
    for i, pair in enumerate(ins):
        show_grid(pair["input"], ax=axes[2*i],   title=f"Train {i} IN", show=False)
        show_grid(pair["output"], ax=axes[2*i+1], title=f"Train {i} OUT", show=False)
    plt.show()

print("\nTry: show_task(train_challenges[0])")



# =========================
# ARC Prize 2025 — Fixed Baseline Submission
# =========================
import copy, json

def solve_task(task):
    preds = []
    for test_case in task.get("test", []):
        # trivial baseline: echo input grid
        preds.append(copy.deepcopy(test_case["input"]))
    return preds

submission = {}

for task_id in sample_sub.keys():
    if task_id in test_challenges:
        task = test_challenges[task_id]
        submission[task_id] = solve_task(task)
    else:
        # if not found, empty list fallback
        submission[task_id] = []

# save submission
with open("submission.json", "w") as f:
    json.dump(submission, f)

print("✅ submission.json created")
print("Tasks predicted:", len(submission))
print("Sample entry:", list(submission.items())[:1])


