from pathlib import Path
import json

# ✅ Define your input folder and output file
INPUT_PATH = Path("/kaggle/input/tubmission-json")
OUTPUT_PATH = Path("/kaggle/working/submission.json")

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)

def solve(task):
    """Return a zero grid with the same shape as input."""
    grid = None
    if isinstance(task, dict) and "test" in task:
        grid = task["test"][0].get("input")
    elif isinstance(task, dict) and "input" in task:
        grid = task["input"]
    elif isinstance(task, list) and len(task) > 0 and "input" in task[0]:
        grid = task[0]["input"]
    else:
        return [[0]]

    h, w = len(grid), len(grid[0]) if grid else (0, 0)
    return [[0 for _ in range(w)] for _ in range(h)]

# ✅ Build submission
submission = {}
test_files = sorted(INPUT_PATH.glob("*.json"))

if not test_files:
    print("⚠️ No files found in", INPUT_PATH)
else:
    for path in test_files:
        task_id = f"test/{path.stem}"
        task_data = load_json(path)
        submission[task_id] = [solve(task_data)]

with open(OUTPUT_PATH, "w") as f:
    json.dump(submission, f)

print(f"✅ Built predictions for {len(submission)} test tasks.")
print(f"✅ Saved submission file to: {OUTPUT_PATH}")



def solve(task):
    """
    Very simple ARC baseline: return a zero grid of the same size as the test input.
    """
    grid = None
    if isinstance(task, dict) and "test" in task:
        grid = task["test"][0].get("input")
    elif isinstance(task, dict) and "input" in task:
        grid = task["input"]
    elif isinstance(task, list) and len(task) > 0 and "input" in task[0]:
        grid = task[0]["input"]
    else:
        return [[0]]

    h, w = len(grid), len(grid[0]) if grid else (0, 0)
    return [[0 for _ in range(w)] for _ in range(h)]



submission = {}
test_files = sorted(INPUT_PATH.glob("*.json"))

if not test_files:
    print("⚠️ No files found in", INPUT_PATH)
else:
    for path in test_files:
        task_id = f"test/{path.stem}"
        task_data = load_json(path)
        submission[task_id] = [solve(task_data)]

print(f"✅ Built predictions for {len(submission)} test tasks.")



!ls -lh /kaggle/working/
!head /kaggle/working/submission.json


