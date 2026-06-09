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


import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from scipy.ndimage import label

# ------------------------------
# Binary Mapping (Unique Keys Only)
# ------------------------------
BINARY_MAPPING = {
    "1101100010100111": 6,
    "1101100010101000": 8,
    "1101100010101110": 4,
    "1101100010110001": 7,
    "1101100010101100": 2,
    "1101100010100011": 3,
    "1101100010100000": 0,
    "1101101110100011": 9,
    "1101100010110010": 5,
    "1101100010110100": 1,
    "00100000": 2
}

def binary_to_decimal(binary: str) -> int:
    try:
        return int(binary, 2)
    except ValueError:
        return sum(int(c) for c in binary if c.isdigit())

def min_max_scale(decimal: int, min_val: int = 32, max_val: int = 219) -> int:
    """Scale decimal to 0â€“9 range."""
    return round((decimal - min_val) / (max_val - min_val) * 9)

def map_pattern_to_number(pattern: str) -> int:
    if pattern in BINARY_MAPPING:
        return BINARY_MAPPING[pattern]
    decimal = binary_to_decimal(pattern)
    return min_max_scale(decimal)

# ------------------------------
# Grid Utils
# ------------------------------
def load_binary_grid(grid: List[List[int]]) -> np.ndarray:
    try:
        return np.array([[int(val) for val in row] for row in grid])
    except Exception:
        return np.zeros((1, 1), dtype=int)

def crop_to_bbox(grid: np.ndarray) -> np.ndarray:
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    if not rows.any() or not cols.any():
        return grid
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return grid[rmin:rmax+1, cmin:cmax+1]

def smart_resize(grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize safely with padding/cropping."""
    h, w = target_shape
    gh, gw = grid.shape
    if (gh, gw) == (h, w):
        return grid
    fixed = np.zeros((h, w), dtype=int)
    min_h, min_w = min(h, gh), min(w, gw)
    fixed[:min_h, :min_w] = grid[:min_h, :min_w]
    return fixed



class IntegratedSolver:
    def __init__(self):
        self.color_rules: Dict[int, int] = {}
        self.background_color: int = 0
        self.valid_numbers: set = set([0])
        self.number_frequency: Counter = Counter()
        self.train_outputs: List[np.ndarray] = []

    def learn(self, train_examples: List[dict]):
        self.color_rules = {}
        self.background_color = 0
        self.valid_numbers = set([0])
        self.number_frequency = Counter()
        self.train_outputs = []

        for ex in train_examples:
            inp = load_binary_grid(ex["input"])
            out = load_binary_grid(ex["output"])
            self.train_outputs.append(out)

            # Binary pattern â†’ number mapping
            binary_str = ''.join(str(x % 2) for x in inp.flatten())
            mapped_num = map_pattern_to_number(binary_str)
            if 0 <= mapped_num <= 9:
                self.valid_numbers.add(mapped_num)
                self.number_frequency[mapped_num] += 1

            # Collect valid numbers from output
            for num in np.unique(out):
                if 0 <= num <= 9:
                    self.valid_numbers.add(num)
                    self.number_frequency[num] += 1

            # Color replacement rules
            for a, b in zip(inp.flatten(), out.flatten()):
                if a != b:
                    self.color_rules[a] = b

            vals, counts = np.unique(inp, return_counts=True)
            self.background_color = vals[np.argmax(counts)]

    # --- Rules ---
    def apply_binary_mapping(self, g: np.ndarray) -> np.ndarray:
        binary_str = ''.join(str(x % 2) for x in g.flatten())
        mapped_num = map_pattern_to_number(binary_str)
        if mapped_num in self.valid_numbers:
            return np.full_like(g, mapped_num)
        return g

    def apply_color_rules(self, g): return np.vectorize(lambda x: self.color_rules.get(x, x))(g)
    def apply_majority_color(self, g): return np.full_like(g, Counter(g.flatten()).most_common(1)[0][0])
    def apply_symmetry(self, g): return np.flip(g, axis=1)
    def apply_rotation(self, g): return np.rot90(g, k=-1)

    def apply_background_preserve(self, g):
        mask = g != self.background_color
        new_g = g.copy()
        new_g[mask] = self.apply_color_rules(g)[mask]
        return new_g

    def apply_flood_fill(self, g):
        labeled, num = label(g != self.background_color)
        if num == 0: return g
        counts = Counter(labeled.flatten()); counts.pop(0, None)
        largest = max(counts, key=counts.get)
        return np.where(labeled == largest, g, self.background_color)

    # --- Scoring ---
    def score_prediction(self, pred: np.ndarray) -> float:
        score = 0
        for truth in self.train_outputs:
            if pred.shape == truth.shape: score += 2
            score += len(set(np.unique(pred)) & set(np.unique(truth)))
            _, po = label(pred != self.background_color)
            _, to = label(truth != self.background_color)
            if po == to: score += 1
        return score

    # --- Solve ---
    def solve(self, train_examples: List[dict], test_input: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        self.learn(train_examples)
        g = load_binary_grid(test_input)
        target_shape = g.shape or (1, 1)

        rules = [
            self.apply_binary_mapping,
            self.apply_color_rules,
            self.apply_background_preserve,
            self.apply_symmetry,
            self.apply_rotation,
            self.apply_majority_color,
            self.apply_flood_fill,
        ]

        attempts = []
        for rule in rules:
            try:
                att = rule(g)
                if att.shape != target_shape:
                    att = smart_resize(att, target_shape)
                attempts.append((att, self.score_prediction(att)))
            except Exception as e:
                print(f"âš ï¸� Rule error: {e}")

        if not attempts:
            fallback_num = self.number_frequency.most_common(1)[0][0] if self.number_frequency else max(self.valid_numbers)
            fallback = np.full(target_shape, fallback_num)
            return fallback, fallback

        attempts.sort(key=lambda x: x[1], reverse=True)
        return attempts[0][0], attempts[1][0] if len(attempts) > 1 else attempts[0][0]


# ------------------------------
# Resize Wrapper for Submission
# ------------------------------
def resize_to_output(grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Wrapper around smart_resize for submission phase."""
    return smart_resize(grid, target_shape)



import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

# ------------------------------
# Safe Execution Wrapper
# ------------------------------
def safe_execute(func):
    """Wrapper to catch errors during execution."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"âš ï¸� Error in {func.__name__}: {e}")
            return {}
    return wrapper

# ------------------------------
# Helper Functions
# ------------------------------
def plot_grid(grid: np.ndarray, title: str) -> None:
    """Show single grid with colors."""
    plt.imshow(grid, cmap='tab20', vmin=0, vmax=9, interpolation="nearest")
    plt.title(title)
    plt.axis("off")

def compare_grids(input_grid, pred1, pred2, task_id, test_id):
    """Side-by-side comparison plots for debug."""
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    for ax, grid, title in zip(
        axes, [input_grid, pred1, pred2],
        [f"Input", "Attempt 1", "Attempt 2"]
    ):
        ax.imshow(grid, cmap="tab20", vmin=0, vmax=9, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(f"Task {task_id} - Test {test_id} Comparison")
    plt.show()

def check_submission_format(submission: Dict) -> None:
    """Ensure Kaggle submission has correct keys."""
    print("\nğŸ”� Checking submission schema...")
    for tid, attempts in submission.items():
        for i in [1, 2]:
            key = f"attempt_{i}"
            if key not in attempts:
                print(f"â�Œ Missing {key} in {tid}")
    print("âœ… Submission schema looks correct!")

# ------------------------------
# Submission Generator
# ------------------------------
@safe_execute
def generate_submission(sample_json: Dict) -> Dict:
    submission = {}
    solver = IntegratedSolver()   # ğŸ‘ˆ from Cell 2
    error_log = {}
    logs = []   # for CSV logging

    if not sample_json:
        print("âš ï¸� Input JSON is empty!")
        return submission

    for task_id, task in sample_json.items():
        print(f"\nğŸ“Œ Processing task: {task_id}")
        train_examples = task.get("train", [])
        test_inputs = task.get("test", [])
        submission[task_id] = {}
        error_log[task_id] = {"shape_mismatch": 0}

        if not train_examples or not test_inputs:
            print(f"âš ï¸� Skipping {task_id} (empty train/test)")
            submission[task_id] = {"attempt_1": [[1]], "attempt_2": [[1]]}
            continue

        # Collect valid numbers from training outputs
        valid_numbers = set([0])
        for example in train_examples:
            out = load_binary_grid(example.get("output", []))
            valid_numbers.update([num for num in np.unique(out) if 0 <= num <= 9])

        # Predict each test input
        for i, test_input in enumerate(test_inputs):
            input_grid = load_binary_grid(test_input.get("input", []))
            if input_grid.size == 0:
                print(f"âš ï¸� Empty input in {task_id}, Test {i}, fallback used")
                input_grid = np.zeros((1, 1), dtype=int)

            target_shape = input_grid.shape

            # Predictions
            try:
                attempt_1, attempt_2 = solver.solve(train_examples, test_input.get("input", []))
            except Exception as e:
                print(f"âš ï¸� Solve error {task_id} Test {i}: {e}")
                attempt_1 = np.full(target_shape, max(valid_numbers))
                attempt_2 = attempt_1

            # Resize if mismatch
            if attempt_1.shape != target_shape:
                error_log[task_id]["shape_mismatch"] += 1
                attempt_1 = resize_to_output(attempt_1, target_shape)
            if attempt_2.shape != target_shape:
                error_log[task_id]["shape_mismatch"] += 1
                attempt_2 = resize_to_output(attempt_2, target_shape)

            # Clip to valid numbers
            attempt_1 = np.where(np.isin(attempt_1, list(valid_numbers)), attempt_1, max(valid_numbers))
            attempt_2 = np.where(np.isin(attempt_2, list(valid_numbers)), attempt_2, max(valid_numbers))
            attempt_1 = np.clip(attempt_1, 0, 9)
            attempt_2 = np.clip(attempt_2, 0, 9)

            # Save in submission
            submission[task_id]["attempt_1"] = attempt_1.tolist()
            submission[task_id]["attempt_2"] = attempt_2.tolist()

            # ğŸ”¹ Debug Visualization
            compare_grids(input_grid, attempt_1, attempt_2, task_id, i)

        # Log summary
        logs.append({
            "task_id": task_id,
            "train_size": len(train_examples),
            "test_size": len(test_inputs),
            "shape_mismatches": error_log[task_id]["shape_mismatch"]
        })

    # Save debug logs
    pd.DataFrame(logs).to_csv("debug_logs.csv", index=False)
    print("\nğŸ“� Debug logs saved as debug_logs.csv")

    # Schema check
    check_submission_format(submission)
    return submission

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    try:
        with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
            SAMPLE_JSON = json.load(f)
        print(f"âœ… Loaded JSON with {len(SAMPLE_JSON)} tasks")
    except FileNotFoundError:
        print("âš ï¸� Error: test_challenges.json not found")
        SAMPLE_JSON = {}

    submission = generate_submission(SAMPLE_JSON)
    if submission:
        with open("submission.json", "w") as f:
            json.dump(submission, f, indent=2)
        print("âœ… Submission saved as submission.json")
    else:
        print("â�Œ Failed to generate submission")



import numpy as np
import pandas as pd
from scipy.ndimage import label

def grid_accuracy(pred: np.ndarray, truth: np.ndarray) -> Dict:
    """Compare prediction vs ground truth on multiple metrics."""
    if pred.shape != truth.shape:
        return {"exact": 0, "cell_acc": 0, "iou": 0}

    exact = int(np.array_equal(pred, truth))

    total = pred.size
    correct = np.sum(pred == truth)
    cell_acc = correct / total if total > 0 else 0

    # IoU on non-background (ignoring 0)
    mask_pred = pred != 0
    mask_truth = truth != 0
    intersection = np.logical_and(mask_pred, mask_truth).sum()
    union = np.logical_or(mask_pred, mask_truth).sum()
    iou = intersection / union if union > 0 else 0

    return {"exact": exact, "cell_acc": cell_acc, "iou": iou}

def evaluate_accuracy(sample_json: Dict, solver) -> pd.DataFrame:
    logs = []
    for task_id, task in sample_json.items():
        train_examples = task.get("train", [])
        if not train_examples:
            continue

        # Run solver on training inputs (self-consistency test)
        for i, example in enumerate(train_examples):
            inp = load_binary_grid(example.get("input", []))
            out = load_binary_grid(example.get("output", []))
            if inp.size == 0 or out.size == 0:
                continue

            try:
                pred1, pred2 = solver.solve(train_examples, example.get("input", []))
            except Exception as e:
                print(f"âš ï¸� Solver failed on {task_id} train {i}: {e}")
                continue

            # Resize if mismatch
            if pred1.shape != out.shape:
                pred1 = smart_resize(pred1, out.shape)
            if pred2.shape != out.shape:
                pred2 = smart_resize(pred2, out.shape)

            # Metrics
            m1 = grid_accuracy(pred1, out)
            m2 = grid_accuracy(pred2, out)

            logs.append({
                "task_id": task_id,
                "train_id": i,
                "attempt": 1,
                "exact": m1["exact"],
                "cell_acc": m1["cell_acc"],
                "iou": m1["iou"],
            })
            logs.append({
                "task_id": task_id,
                "train_id": i,
                "attempt": 2,
                "exact": m2["exact"],
                "cell_acc": m2["cell_acc"],
                "iou": m2["iou"],
            })

    df = pd.DataFrame(logs)
    if not df.empty:
        print("\nğŸ“Š Accuracy Summary:")
        print(df.groupby("attempt")[["exact", "cell_acc", "iou"]].mean())
        df.to_csv("accuracy_logs.csv", index=False)
        print("âœ… Saved accuracy_logs.csv")
    else:
        print("âš ï¸� No accuracy data collected.")
    return df

# ------------------------------
# Main Accuracy Test
# ------------------------------
if __name__ == "__main__":
    try:
        with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
            SAMPLE_JSON = json.load(f)
    except FileNotFoundError:
        print("âš ï¸� Error: test_challenges.json not found")
        SAMPLE_JSON = {}

    solver = IntegratedSolver()
    _ = evaluate_accuracy(SAMPLE_JSON, solver)



import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Utility functions
# ------------------------------

def show_grid(grid, title="Grid"):
    """Grid ko visualize karo color map ke sath"""
    arr = np.array(grid)
    plt.imshow(arr, cmap="tab20", vmin=0, vmax=9, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.show()

def explain_task(task_json):
    """Ek training task ko detail me samjho"""
    print("ğŸ“Œ Task Details:")
    
    # Training examples
    train_examples = task_json.get("train", [])
    print(f"  ğŸ”¹ Total Training Examples: {len(train_examples)}")
    
    for i, ex in enumerate(train_examples):
        print(f"\n--- Training Example {i} ---")
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        print(f" Input shape: {inp.shape}, unique colors: {np.unique(inp)}")
        print(f" Output shape: {out.shape}, unique colors: {np.unique(out)}")
        
        # Visualize input and output
        show_grid(inp, f"Train {i} - Input")
        show_grid(out, f"Train {i} - Output")

    # Test examples
    test_examples = task_json.get("test", [])
    print(f"\n  ğŸ”¹ Total Test Examples: {len(test_examples)}")
    for i, ex in enumerate(test_examples):
        inp = np.array(ex["input"])
        print(f"\n--- Test Example {i} ---")
        print(f" Input shape: {inp.shape}, unique colors: {np.unique(inp)}")
        show_grid(inp, f"Test {i} - Input")

# ------------------------------
# Load and test
# ------------------------------

# Example: file load karna
with open("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", "r") as f:
    data = json.load(f)

# Ek random task pick karo
task_id, task = list(data.items())[0]
print(f"âœ… Loaded Task ID: {task_id}")
explain_task(task)
import numpy as np
from scipy.ndimage import label

def analyze_training_examples(task_id: str, train_examples: list, test_examples: list):
    print(f"\nğŸ”� Loaded Task ID: {task_id}")
    print(f"ğŸ“Œ Task Details:")
    print(f"  ğŸ”¹ Total Training Examples: {len(train_examples)}")

    # --- Training Examples ---
    for i, ex in enumerate(train_examples):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])

        print(f"\n--- Training Example {i} ---")
        print(f" Input shape: {inp.shape}, unique colors: {np.unique(inp)}")
        print(f" Output shape: {out.shape}, unique colors: {np.unique(out)}")

        # Check if resizing
        resize_factor = (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
        if resize_factor[0].is_integer() and resize_factor[1].is_integer():
            print(f" ğŸ”� Likely scaling: {resize_factor}x")
        elif inp.shape == out.shape:
            print(" ğŸ”� Same size â†’ transformation may be symmetry/rotation/color change")
        else:
            print(" ğŸ”� Uneven resize â†’ complex transformation")

        # Compare colors
        added_colors = set(np.unique(out)) - set(np.unique(inp))
        removed_colors = set(np.unique(inp)) - set(np.unique(out))
        if added_colors:
            print(f" ğŸ”� New colors added in output: {added_colors}")
        if removed_colors:
            print(f" ğŸ”� Colors removed in output: {removed_colors}")
        if not added_colors and not removed_colors:
            print(" ğŸ”� No color change")

        # Object count check
        _, input_objs = label(inp != 0)
        _, output_objs = label(out != 0)
        print(f" ğŸ”� Objects: Input={input_objs}, Output={output_objs}")

    # --- Test Examples ---
    print(f"\n  ğŸ”¹ Total Test Examples: {len(test_examples)}")
    for j, ex in enumerate(test_examples):
        inp = np.array(ex["input"])
        print(f"\n--- Test Example {j} ---")
        print(f" Input shape: {inp.shape}, unique colors: {np.unique(inp)}")



import numpy as np
from scipy.ndimage import label

def analyze_training_examples(task_id: str, train_examples: list, test_examples: list):
    print(f"\nğŸ”� Loaded Task ID: {task_id}")
    print(f"ğŸ“Œ Task Details:")
    print(f"  ğŸ”¹ Total Training Examples: {len(train_examples)}")

    # --- Training Examples ---
    for i, ex in enumerate(train_examples):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])

        print(f"\n--- Training Example {i} ---")
        print(f" Input shape: {inp.shape}, unique colors: {np.unique(inp)}")
        print(f" Output shape: {out.shape}, unique colors: {np.unique(out)}")

        # Check if resizing
        resize_factor = (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
        if resize_factor[0].is_integer() and resize_factor[1].is_integer():
            print(f" ğŸ”� Likely scaling: {resize_factor}x")
        elif inp.shape == out.shape:
            print(" ğŸ”� Same size â†’ transformation may be symmetry/rotation/color change")
        else:
            print(" ğŸ”� Uneven resize â†’ complex transformation")

        # Compare colors
        added_colors = set(np.unique(out)) - set(np.unique(inp))
        removed_colors = set(np.unique(inp)) - set(np.unique(out))
        if added_colors:
            print(f" ğŸ”� New colors added in output: {added_colors}")
        if removed_colors:
            print(f" ğŸ”� Colors removed in output: {removed_colors}")
        if not added_colors and not removed_colors:
            print(" ğŸ”� No color change")

        # Object count check
        _, input_objs = label(inp != 0)
        _, output_objs = label(out != 0)
        print(f" ğŸ”� Objects: Input={input_objs}, Output={output_objs}")

    # --- Test Examples ---
    print(f"\n  ğŸ”¹ Total Test Examples: {len(test_examples)}")
    for j, ex in enumerate(test_examples):
        inp = np.array(ex["input"])
        print(f"\n--- Test Example {j} ---")
        print(f" Input shape: {inp.shape}, unique colors: {np.unique(inp)}")



import json

# 1) JSON load karo
with open("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", "r") as f:
    data = json.load(f)

# 2) Koi ek task_id pick karo (jaise tumne 00576224 kiya tha)
task_id = "00576224"
task = data[task_id]

# 3) Call analyzer function
analyze_training_examples(
    task_id,
    train_examples=task["train"],
    test_examples=task["test"]
)



import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.ndimage import label
import pandas as pd

def analyze_training_examples(task_id: str, train_examples: list):
    print(f"\nğŸ”� Analyzing Task: {task_id}")
    logs = []

    for i, ex in enumerate(train_examples):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])

        # --- Shapes & Ratios ---
        in_shape, out_shape = inp.shape, out.shape
        scale_y = out_shape[0] / in_shape[0] if in_shape[0] > 0 else 0
        scale_x = out_shape[1] / in_shape[1] if in_shape[1] > 0 else 0

        # --- Colors ---
        in_colors = np.unique(inp)
        out_colors = np.unique(out)
        color_map = {c: out_colors[0] for c in in_colors if c not in out_colors}  # naive mapping

        # --- Objects ---
        in_labels, in_objects = label(inp != inp[0,0])  # assume background = top-left
        out_labels, out_objects = label(out != out[0,0])

        # --- Background ---
        bg_in = Counter(inp.flatten()).most_common(1)[0][0]
        bg_out = Counter(out.flatten()).most_common(1)[0][0]

        # --- Symmetry/Rotation Checks ---
        symmetry_match = np.array_equal(np.flip(inp, 1), out)
        rotation_match = np.array_equal(np.rot90(inp), out)

        # --- Exact Match ---
        exact_match = np.array_equal(inp, out)

        # --- Visualization ---
        fig, axes = plt.subplots(1, 2, figsize=(6,3))
        axes[0].imshow(inp, cmap="tab20", vmin=0, vmax=9)
        axes[0].set_title(f"Train {i} Input")
        axes[0].axis("off")
        axes[1].imshow(out, cmap="tab20", vmin=0, vmax=9)
        axes[1].set_title(f"Train {i} Output")
        axes[1].axis("off")
        plt.show()

        # --- Log Data ---
        logs.append({
            "example": i,
            "in_shape": in_shape,
            "out_shape": out_shape,
            "scale_x": round(scale_x,2),
            "scale_y": round(scale_y,2),
            "in_colors": list(in_colors),
            "out_colors": list(out_colors),
            "color_map_guess": color_map,
            "in_objects": in_objects,
            "out_objects": out_objects,
            "bg_in": int(bg_in),
            "bg_out": int(bg_out),
            "exact_match": exact_match,
            "symmetry_match": symmetry_match,
            "rotation_match": rotation_match
        })

    # Summary Table
    df = pd.DataFrame(logs)
    print("\nğŸ“Š Training Summary Table:")
    display(df)

    # Consistency Check
    print("\nâœ… Consistency Check:")
    if df["scale_x"].nunique() == 1 and df["scale_y"].nunique() == 1:
        print(f"   â�� Same scaling factor: {df['scale_x'].iloc[0]}x , {df['scale_y'].iloc[0]}y")
    if all(df["in_objects"] == df["out_objects"]):
        print("   â�� Object counts consistent across all examples")
    if all(df["bg_in"] == df["bg_out"]):
        print("   â�� Background color preserved")
    if any(df["symmetry_match"]):
        print("   â�� At least one example suggests symmetry")
    if any(df["rotation_match"]):
        print("   â�� At least one example suggests rotation")
    if any(df["exact_match"]):
        print("   â�� At least one example is exact copy (input==output)")

    return df



# Example run (Task ko load karke analyze karna)
import json

# Load ARC training JSON
with open("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", "r") as f:
    TRAIN_JSON = json.load(f)

# Pick ek task
task_id = list(TRAIN_JSON.keys())[0]   # first task ka ID
train_examples = TRAIN_JSON[task_id]["train"]

# Run analyzer
df = analyze_training_examples(task_id, train_examples)





