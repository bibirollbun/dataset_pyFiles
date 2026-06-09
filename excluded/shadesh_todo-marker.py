# ===============================================================
# ARC Prize 2025 - Competition Solver
# ===============================================================

# --- Imports ---
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter
import copy
from itertools import permutations


# ===============================================================
# Data Paths
# ===============================================================
DATA_DIR = Path("/kaggle/input/arc-prize-2025")
TRAIN_FILE = DATA_DIR / "arc-agi_training_challenges.json"
TEST_FILE = DATA_DIR / "arc-agi_test_challenges.json"


# ===============================================================
# Utility Functions
# ===============================================================
def load_json(file_path):
    """Loads a JSON file and returns its content."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    """Saves a dictionary to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def grid_to_numpy(grid):
    """Converts a list of lists grid to a numpy array."""
    return np.array(grid)


def numpy_to_grid(arr):
    """Converts a numpy array back to a list of lists."""
    return arr.tolist()


def get_objects(grid, background_color=0):
    """
    Identifies connected components (objects) in the grid using flood-fill.
    Returns a list of cropped object grids.
    """
    grid_copy = np.copy(grid)
    h, w = grid_copy.shape
    visited = np.zeros_like(grid_copy, dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if grid_copy[r, c] != background_color and not visited[r, c]:
                obj_grid = np.zeros_like(grid_copy, dtype=int)
                stack = [(r, c)]
                visited[r, c] = True
                min_r = max_r = r
                min_c = max_c = c

                while stack:
                    curr_r, curr_c = stack.pop()
                    obj_grid[curr_r, curr_c] = grid_copy[curr_r, curr_c]

                    min_r, max_r = min(min_r, curr_r), max(max_r, curr_r)
                    min_c, max_c = min(min_c, curr_c), max(max_c, curr_c)

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and \
                           grid_copy[nr, nc] != background_color and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                # Crop the object
                cropped_obj = obj_grid[min_r:max_r + 1, min_c:max_c + 1]
                objects.append(cropped_obj)

    return objects


def find_largest_object(grid):
    """Returns the largest object by pixel count."""
    objects = get_objects(grid)
    if not objects:
        return None
    return max(objects, key=lambda obj: np.sum(obj > 0))


def remove_color(grid, color):
    """Removes all pixels of a specific color."""
    new_grid = np.copy(grid)
    new_grid[new_grid == color] = 0
    return new_grid


def fill_hollow(grid):
    """Heuristic placeholder for filling hollow shapes."""
    return grid  # TODO: Implement real fill logic


def map_color_to_count(grid):
    """Maps each color to its frequency count."""
    counts = Counter(grid.flatten())
    new_grid = np.copy(grid)
    for color, count in counts.items():
        new_grid[new_grid == color] = count
    return new_grid


def get_output_dims(input_grid, output_grid):
    """Infers output dimensions based on input-output relation."""
    return output_grid.shape


def apply_rule(rule, input_grid):
    """Applies a specific rule (function + args) to a grid."""
    func, *args = rule
    return func(input_grid, *args) if args else func(input_grid)


# ===============================================================
# DSL Operations
# ===============================================================
DSL_OPS = {
    "rotate_90": lambda g: np.rot90(g, -1),
    "flip_horizontal": lambda g: np.fliplr(g),
    "flip_vertical": lambda g: np.flipud(g),
    "mirror_horizontally_at_middle": lambda g: np.vstack((g, np.flipud(g))),
    "mirror_vertically_at_middle": lambda g: np.hstack((g, np.fliplr(g))),
    "isolate_largest_object": find_largest_object,
    "remove_background": lambda g: g[g != 0].reshape(1, -1),  # Simplified
}


# ===============================================================
# Rule Class & Program Induction
# ===============================================================
class Rule:
    """Represents a single step in a program (operation + args)."""

    def __init__(self, op_name, *args):
        self.op_name = op_name
        self.args = args

    def apply(self, grid):
        func = DSL_OPS.get(self.op_name)
        if func is None:
            return None  # Invalid op

        if self.op_name in ["rotate_90", "flip_horizontal", "flip_vertical",
                            "isolate_largest_object"]:
            return func(grid)

        return None


def verify_program(program, train_pairs):
    """Checks if a program works for all training pairs."""
    for pair in train_pairs:
        input_grid = grid_to_numpy(pair['input'])
        expected_output = grid_to_numpy(pair['output'])

        current_grid = input_grid
        for rule in program:
            current_grid = rule.apply(current_grid)
            if current_grid is None:
                return False

        if not np.array_equal(current_grid, expected_output):
            return False

    return True


def find_programs(train_pairs, max_depth=2):
    """Finds all valid programs up to max_depth."""
    valid_programs = []

    # One-step programs
    for op_name in DSL_OPS:
        program = [Rule(op_name)]
        if verify_program(program, train_pairs):
            valid_programs.append(program)

    # Two-step programs
    if max_depth > 1:
        for p1 in list(valid_programs):
            for op_name2 in DSL_OPS:
                p2 = p1 + [Rule(op_name2)]
                if verify_program(p2, train_pairs):
                    valid_programs.append(p2)

    return valid_programs


# ===============================================================
# Main Logic
# ===============================================================
def main():
    """Main solver function."""
    try:
        challenges = load_json(TEST_FILE)
    except FileNotFoundError:
        print("Test file not found. Using training challenges instead.")
        challenges = load_json(TRAIN_FILE)

    submission = {}

    for task_id, task_data in challenges.items():
        print(f"Solving task: {task_id}")

        train_pairs = task_data['train']
        test_inputs = task_data['test']

        found_programs = find_programs(train_pairs)

        predictions = []
        for test_pair in test_inputs:
            input_grid = grid_to_numpy(test_pair['input'])

            # --- Attempt 1 ---
            output_1 = None
            if found_programs:
                best_program = found_programs[0]
                output_1_np = input_grid
                for rule in best_program:
                    output_1_np = rule.apply(output_1_np)
                    if output_1_np is None:
                        break
                if output_1_np is not None:
                    output_1 = numpy_to_grid(output_1_np)

            # --- Attempt 2 ---
            output_2 = None
            if len(found_programs) > 1:
                second_best_program = found_programs[1]
                output_2_np = input_grid
                for rule in second_best_program:
                    output_2_np = rule.apply(output_2_np)
                    if output_2_np is None:
                        break
                if output_2_np is not None:
                    output_2 = numpy_to_grid(output_2_np)

            # --- Fallbacks ---
            if output_1 is None:
                output_1 = test_pair['input']  # Identity
            if output_2 is None:
                try:
                    output_2 = numpy_to_grid(np.rot90(input_grid, -1))
                except Exception:
                    output_2 = test_pair['input']

            predictions.append({
                "attempt_1": output_1,
                "attempt_2": output_2
            })

        submission[task_id] = predictions

    save_json(submission, "submission.json")
    print("✅ Submission file created successfully!")


# ===============================================================
# Run
# ===============================================================
if __name__ == "__main__":
    main()


