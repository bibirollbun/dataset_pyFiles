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


# -------- cell 1: baseline solver --------
def solve_task(task: dict):
    """
    Toy baseline:
    • If the first train output has the same width as the test input,
      return the input unchanged.
    • Otherwise return the input mirrored horizontally.
    """
    ref_out = task["train"][0]["output"]
    outputs = []

    for test in task["test"]:
        inp = test["input"]
        if len(inp[0]) == len(ref_out[0]):          # same width
            out = [row[:] for row in inp]          # identity
        else:
            out = [row[::-1] for row in inp]       # flip
        outputs.append(out)

    return outputs            


# -------- cell 2: build submission --------
import json, os, tqdm

DATA_DIR = "/kaggle/input/arc-prize-2025"

with open(f"{DATA_DIR}/arc-agi_test_challenges.json") as f:
    test_tasks = json.load(f)

submission = {}
for task_id, task in tqdm.tqdm(test_tasks.items()):
    preds = solve_task(task)
    assert len(preds) == len(task["test"]), f"Mismatch in {task_id}"
    submission[task_id] = [
        {"attempt_1": p, "attempt_2": p} for p in preds
    ]

sub_path = "/kaggle/working/submission.json"
with open(sub_path, "w") as f:
    json.dump(submission, f)

print("✅  submission.json created at", sub_path)


# -------- cell 3: quick local evaluation --------
import json, tqdm, numpy as np

DATA_DIR = "/kaggle/input/arc-prize-2025"
with open(f"{DATA_DIR}/arc-agi_evaluation_challenges.json") as f:
    eval_tasks = json.load(f)
with open(f"{DATA_DIR}/arc-agi_evaluation_solutions.json") as f:
    eval_solutions = json.load(f)

correct = total = 0
for tid, task in tqdm.tqdm(eval_tasks.items()):
    preds = solve_task(task)
    for pred, true in zip(preds, eval_solutions[tid]):
        total += 1
        if np.array(pred).tolist() == true:
            correct += 1
print(f"Local eval accuracy: {correct}/{total} = {correct/total:.2%}")


# pick a failing task
for tid, task in eval_tasks.items():
    preds = solve_task(task)
    if preds[0] != eval_solutions[tid][0]:
        bad_id = tid; break

print("Example failing task:", bad_id)
from matplotlib import pyplot as plt, numpy as np
def show(g): plt.imshow(np.array(g), cmap="tab10", vmin=0, vmax=9); plt.axis("off")

pair = task["train"][0]
show(pair["input"]); plt.show()
show(pair["output"]); plt.show()

