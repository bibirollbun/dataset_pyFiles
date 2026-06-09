from IPython.display import Image, display

# Display the image directly
display(Image(filename="/kaggle/input/arc-prize1/ARC.png"))



# This code cell can be hidden from public view in Kaggle
from IPython.display import Image, display

# Path to your image
img_path = "/kaggle/input/arc-model/ARC.png"

# Display the image
display(Image(filename=img_path))



import json
import numpy as np
from pathlib import Path

# Adjust path if running locally vs Kaggle
DATA_DIR = Path("../input/arc-prize-2025")

train_challenges = json.load(open(DATA_DIR / "arc-agi_training_challenges.json"))
train_solutions = json.load(open(DATA_DIR / "arc-agi_training_solutions.json"))
test_challenges  = json.load(open(DATA_DIR / "arc-agi_test_challenges.json"))
eval_challenges  = json.load(open(DATA_DIR / "arc-agi_evaluation_challenges.json"))
eval_solutions   = json.load(open(DATA_DIR / "arc-agi_evaluation_solutions.json"))

print("Train tasks:", len(train_challenges))
print("Test tasks:", len(test_challenges))
print("Eval tasks:", len(eval_challenges))



def to_numpy(grid):
    return np.array(grid, dtype=int)

def from_numpy(arr):
    return arr.tolist()

def detect_color_mapping(train_inputs, train_outputs):
    rules = []
    for inp, out in zip(train_inputs, train_outputs):
        inp_arr, out_arr = to_numpy(inp), to_numpy(out)
        if inp_arr.shape == out_arr.shape:
            diff = (inp_arr != out_arr)
            if np.any(diff):
                mapping = {}
                for i_val, o_val in zip(inp_arr[diff], out_arr[diff]):
                    mapping[int(i_val)] = int(o_val)
                if mapping:
                    rules.append(mapping)
    return rules if rules else [{}]

def apply_rule(inp_grid, rules):
    arr = to_numpy(inp_grid).copy()
    for rule in rules:
        for k, v in rule.items():
            arr[arr == k] = v
    return from_numpy(arr)

def detect_transform(inp, out):  # your new helper
    inp_arr, out_arr = to_numpy(inp), to_numpy(out)
    if inp_arr.shape == out_arr.shape:
        if np.array_equal(out_arr, np.rot90(inp_arr, 1)):
            return "rot90"
        if np.array_equal(out_arr, np.rot90(inp_arr, 2)):
            return "rot180"
        if np.array_equal(out_arr, np.flipud(inp_arr)):
            return "flip_ud"
        if np.array_equal(out_arr, np.fliplr(inp_arr)):
            return "flip_lr"
    return None



def solve_task(task):
    """
    Solve one task: infer rule from training examples,
    apply to test inputs, return 2 attempts.
    """
    train_examples = task["train"]
    test_examples = task["test"]

    train_inputs = [ex["input"] for ex in train_examples]
    train_outputs = [ex["output"] for ex in train_examples]

    rules = detect_color_mapping(train_inputs, train_outputs)

    outputs = []
    for test_ex in test_examples:
        inp = test_ex["input"]
        attempt_1 = apply_rule(inp, rules)
        attempt_2 = inp  # fallback: just copy input grid
        outputs.append({
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        })
    return outputs



submission = {}
for task_id, task in test_challenges.items():
    submission[task_id] = solve_task(task)

with open("submission.json", "w") as f:
    json.dump(submission, f)

print("✅ Created submission.json with", len(submission), "tasks")


