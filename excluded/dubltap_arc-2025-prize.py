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


# Minimal valid ARC submission artifact
import os, json
out_path = "/kaggle/working/submission.json"
with open(out_path, "w") as f:
    json.dump({}, f)
print("Wrote:", out_path, "Size:", os.path.getsize(out_path), "bytes")



# ARC 2025 — write a VALID /kaggle/working/submission.json
# Strategy:
# 1) If a sample_submission.json exists under /kaggle/input, copy it (always correct format).
# 2) Else, find the official test tasks JSON, read each test input's shape,
#    and emit zero-grids of the same shape: {"<task_id>": [grid_for_test1, grid_for_test2, ...]}.

import os, json, glob, shutil

WORK = "/kaggle/working"
OUT = os.path.join(WORK, "submission.json")
INPUT = "/kaggle/input"

def try_copy_sample_submission():
    # Look for something like sample_submission.json in any attached input
    for p in glob.glob(os.path.join(INPUT, "**", "*sample*submission*.json"), recursive=True):
        try:
            # Validate it's JSON
            with open(p, "r") as f:
                _ = json.load(f)
            shutil.copy(p, OUT)
            print(f"✅ Copied sample submission: {p} -> {OUT}")
            return True
        except Exception as e:
            pass
    return False

def find_test_tasks_file():
    # Heuristic: look for JSON/JSONL that likely contains the PUBLIC/TEST tasks
    # Prefer filenames with both "test" and "task"
    candidates = []
    for pat in ["**/*.json", "**/*.jsonl"]:
        for p in glob.glob(os.path.join(INPUT, pat), recursive=True):
            name = os.path.basename(p).lower()
            if "test" in name and "task" in name:
                candidates.append(p)
    # fallback: any file with "public" and "task"
    if not candidates:
        for pat in ["**/*.json", "**/*.jsonl"]:
            for p in glob.glob(os.path.join(INPUT, pat), recursive=True):
                name = os.path.basename(p).lower()
                if "public" in name and "task" in name:
                    candidates.append(p)
    # pick the shortest path (usually the main one)
    return sorted(set(candidates), key=len)[0] if candidates else None

def load_json_or_jsonl(path):
    with open(path, "r") as f:
        head = f.read(2); f.seek(0)
        if head.strip().startswith("{") or head.strip().startswith("["):
            return json.load(f)  # JSON
        rows = []
        for line in open(path, "r"):
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except: pass
        return rows

def iter_tasks(obj):
    # Normalize various common formats into a list of task dicts
    if isinstance(obj, dict):
        if "tasks" in obj and isinstance(obj["tasks"], list):
            for t in obj["tasks"]: 
                if isinstance(t, dict): yield t
        else:
            for k, v in obj.items():
                if isinstance(v, dict):
                    if "id" not in v: v = {"id": k, **v}
                    yield v
    elif isinstance(obj, list):
        for t in obj:
            if isinstance(t, dict): 
                yield t

def zero_grid_like(grid):
    return [[0 for _ in row] for row in grid]

def build_submission_from_tests(test_path):
    data = load_json_or_jsonl(test_path)
    preds = {}
    n_tasks = 0
    n_outputs = 0
    for task in iter_tasks(data):
        tid = task.get("id") or task.get("task_id") or task.get("uuid") or task.get("name")
        tests = task.get("test") or task.get("tests") or []
        outputs = []
        for item in tests:
            # Common schema: {"input": <grid>} — create zero grid of same HxW
            grid = item.get("input")
            if isinstance(grid, list) and grid and isinstance(grid[0], list):
                outputs.append(zero_grid_like(grid))
            else:
                # If shape is unclear, default to an empty 0x0 grid (still valid JSON)
                outputs.append([])
        if tid:
            # ARC-style simplest schema: {"task_id": [<grid>, <grid>, ...]}
            preds[str(tid)] = outputs
            n_tasks += 1
            n_outputs += len(outputs)
    with open(OUT, "w") as f:
        json.dump(preds, f)
    print(f"✅ Wrote {OUT} with {n_tasks} tasks / {n_outputs} outputs (zero-grids).")
    return n_tasks > 0

# ---- main ----
done = try_copy_sample_submission()
if not done:
    test_path = find_test_tasks_file()
    if test_path is None:
        # last resort: write an empty, but valid JSON (will likely be format-rejected, but avoids missing-file)
        with open(OUT, "w") as f: json.dump({}, f)
        print("⚠️ Could not find a test tasks file; wrote empty submission.json")
    else:
        print("Found test tasks file:", test_path)
        ok = build_submission_from_tests(test_path)
        if not ok:
            # fallback if parsing failed
            with open(OUT, "w") as f: json.dump({}, f)
            print("⚠️ Parsing failed; wrote empty submission.json")

print("Exists:", os.path.exists(OUT), "Size:", os.path.getsize(OUT))


