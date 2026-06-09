import json
import pandas as pd
import numpy as np
import json
from pathlib import Path

import matplotlib.pyplot as plt
from pathlib import Path


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def ARC_PRIZE_2025():
    data_dir = Path('/kaggle/input/arc-prize-2025')
    train_file = data_dir / 'arc-agi_training_challenges.json'
    test_file  = data_dir / 'arc-agi_test_challenges.json'
    sample_submission_file = data_dir / 'sample_submission.json'

    train_data = load_json(train_file)
    test_data = load_json(test_file)
    sample_submission = load_json(sample_submission_file)

    print(f"Total training tasks: {len(train_data)}")
    print(f"Total test tasks: {len(test_data)}")
    print(f"Sample submission keys: {list(sample_submission.keys())[:5]}")
    return train_data, test_data, sample_submission 
train_data, test_data, sample_submission = ARC_PRIZE_2025()


def data_cleaning(train_data, test_data, sample_submission):

    print("========== DATASET OVERVIEW ==========")
    print(f"Train dataset size : {len(train_data.keys())}")
    print(f"Test dataset size  : {len(test_data.keys())}")
    print("======================================\n")

    train_df = pd.DataFrame.from_dict(train_data, orient="index")
    test_df = pd.DataFrame.from_dict(test_data, orient="index")
    sample_df = pd.DataFrame.from_dict(sample_submission, orient="index")


    print(">>> Checking for NULL values:")
    if train_df.isnull().values.any() or test_df.isnull().values.any() or sample_df.isnull().values.any():
        print("   Null values found!\n")
        print("   Train null counts:\n", train_df.isnull().sum(), "\n")
        print("   Test null counts:\n", test_df.isnull().sum(), "\n")
        print("   Sample null counts:\n", sample_df.isnull().sum(), "\n")
    else:
        print("   No null values detected.\n")


    print(">>> Checking for NaN values:")
    if train_df.isna().values.any() or test_df.isna().values.any() or sample_df.isna().values.any():
        print("   NaN values found!\n")
        print("   Train NaN counts:\n", train_df.isna().sum(), "\n")
        print("   Test NaN counts:\n", test_df.isna().sum(), "\n")
        print("   Sample NaN counts:\n", sample_df.isna().sum(), "\n")
    else:
        print("   No NaN values detected.\n")


train_data, test_data, sample_submission = ARC_PRIZE_2025()
data_cleaning(train_data, test_data, sample_submission)


# Store each puzzle input/output as a numpy.ndarray of type np.uint8, shape (H, W), values ∈ {0,…,9}.
# This will make ops like rotate, flip, crop, etc. super fast.

def to_grid_matrix(matrix):
    return np.arry(matrix, dtype = np.uint8)
def to_form_grid(matrix):
    return matrix.tolist()


def to_grid_matrix(example):
    return np.array(example, dtype=np.int32)         # was np.arry

def from_grid(grid):
    return grid.tolist()                            # you used 'grid' but the function name was different

# ---------- Ops (pure functions) ----------
def rotate(grid, k=1):
    return np.rot90(grid, k)                        # you had 'gird' and mismatched param

def flip(grid, axis=0):                             # 0=vertical, 1=horizontal
    return np.flip(grid, axis=axis)

def transpose(grid):
    return grid.T                                   # no need for k

def recolor(grid, mapping):
    lut = np.arange(256, dtype=np.uint8)            # was np.arrange
    for src, tgt in mapping.items():
        lut[src] = tgt
    return lut[grid]


from scipy.ndimage import label

def extract_components(grid, ignore_color=0):
    
    mask = grid != ignore_color
    labeled, n = label(mask)
    return labeled, n


class Program:
    def __init__(self, ops, name=""):
        """
        ops: list of callables (each takes a grid and returns a grid)
        """
        self.ops = ops
        self.name = name or "+".join([op.__name__ if hasattr(op, "__name__") else str(op) for op in ops])

    def run(self, grid):
        out = grid
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self):
        return f"Program({self.name}, ops={len(self.ops)})"



def fits_example(input_grid, output_grid, program):
    gi = to_grid_matrix(input_grid)
    go = to_grid_matrix(output_grid)
    out = program.run(gi)
    return out.shape == go.shape and np.array_equal(out, go)


def fits_task(task, program):
    for pair in task["train"]:
        inp = to_grid_matrix(pair["input"])
        out = to_grid_matrix(pair["output"])
        if not fits_example(inp, out, program):
            return False
    return True



def prog_identity():
    return Program([lambda g: g], "identity")

def prog_rotate(k):
    return Program([lambda g, kk=k: rotate(g, kk)], f"rotate{k}")

def prog_flip(axis):
    return Program([lambda g, ax=axis: flip(g, ax)], f"flip{axis}")

def prog_transpose():
    return Program([lambda g: transpose(g)], "transpose")

def prog_recolor(mapping):
    return Program([lambda g, m=mapping: recolor(g, m)], "recolor")

# Two-op examples
def prog_rotate_then_recolor(k, mapping):
    return Program([
        lambda g, kk=k: rotate(g, kk),
        lambda g, m=mapping: recolor(g, m)
    ], f"rotate{k}+recolor")

def prog_flip_then_recolor(axis, mapping):
    return Program([
        lambda g, ax=axis: flip(g, ax),
        lambda g, m=mapping: recolor(g, m)
    ], f"flip{axis}+recolor")

def prog_transpose_then_recolor(mapping):
    return Program([
        lambda g: transpose(g),
        lambda g, m=mapping: recolor(g, m)
    ], "transpose+recolor")



def description_length(program):
    L = len(program.ops)
    return L

def score_program(task, program, lam=0.05):
    fit = 1.0 if fits_task(task, program) else 0.0
    mdl = description_length(program)
    return fit - lam * mdl 
    return len(program.ops)



def generate_candidates(task):
    cands = []
    cands.append(Program(lambda g: g, "identity"))
    for k in [1, 2, 3]:
        cands.append(Program(lambda g, k=k: rotate(g, k), f"rotate{k}"))
    for axis in [0, 1]:
        cands.append(Program(lambda g, axis=axis: flip(g, axis), f"flip{axis}"))
    cands.append(Program(lambda g: transpose(g), "transpose"))

    # Try to infer the Global Recorder mapping form the first_train_pair

    first = task["train"][0]
    mapping = infer_global_color_mapping(first["input"], first["output"])
    if mapping:
        cands.append(prog_recolor(mapping))

    # Combine with geometry variants

        for k in (1, 2, 3):
            cands.append(prog_rotate_then_recolor(k, mapping))
        for ax in (0, 1):
            cands.append(prog_flip_then_recolor(ax, mapping))
        cands.append(prog_transpose_then_recolor(mapping))

    # Later add symmetry, crop/pad, object-wise ops, etc.

    return cands


import time

def solve_task_simple(task, time_budget=2.0):
    """
    Returns up to two programs (best, diverse runner-up).
    """
    t0 = time.time()
    cands = generate_candidates(task)

    scored = []
    for prog in cands:
        sc = score_program(task, prog)
        scored.append((sc, prog))
        if time.time() - t0 > time_budget:
            break

    scored.sort(key=lambda x: (-x[0], description_length(x[1])))

    winners = [p for s, p in scored if s >= 1.0]  # since fit=1.0 - lam*mdl < 1, use fit check:
    if not winners:
        top = [p for _, p in scored[:2]]
        return top

    best = winners[0]
    second = None
    for cand in winners[1:]:
        if len(cand.ops) != len(best.ops) or cand.ops[0].__code__.co_firstlineno != best.ops[0].__code__.co_firstlineno:
            second = cand
            break
    if not second and len(winners) > 1:
        second = winners[1]

    return [best] if second is None else [best, second]



def predict_outputs_for_task(task, progs):

    out = []
    for test in task["test"]:
        gi = to_grid_matrix(test["input"])
        # attempt_1
        a1 = progs[0].run(gi) if len(progs) >= 1 else gi
        # attempt_2
        if len(progs) >= 2:
            a2 = progs[1].run(gi)
        else:
            # provide a different heuristic as second attempt (e.g., identity vs transpose)
            a2 = transpose(gi) if a1.shape == transpose(gi).shape else a1

        out.append({
            "attempt_1": from_grid(a1),
            "attempt_2": from_grid(a2),
        })
    return out



def build_submission(test_data, train_data_lookup, per_task_time=2.0):
    """
    test_data: dict[task_id] -> {train: [...], test: [...]}
    train_data_lookup: use this if your test_data structure references the same format.
                       (ARC provides both similarly-shaped dicts)
    Returns a dict ready to json.dump as submission.json
    """
    submission = {}
    for task_id, task in test_data.items():
        # In ARC’s JSON, test_data tasks also include 'train' examples you can learn from.
        # If not, you may need to fetch them from train_data by task_id. Often they’re separate sets,
        # but in ARC-AGI format, each 'task' has its own 'train' + 'test'.
        full_task = task

        progs = solve_task_simple(full_task, time_budget=per_task_time)
        preds = predict_outputs_for_task(full_task, progs)
        submission[task_id] = preds
    return submission



def evaluate_on_train(train_data, per_task_time=2.0):
    correct, total = 0, 0

    for task_id, task in train_data.items():
        progs = solve_task_simple(task, time_budget=per_task_time)
        preds = predict_outputs_for_task(task, progs)

        # evaluate only on training pairs (they always have outputs)
        for pred, gold in zip(preds, task["train"]):
            gold_out = to_grid_matrix(gold["output"])
            p1 = to_grid_matrix(pred.get("attempt_1"))
            p2 = to_grid_matrix(pred.get("attempt_2"))

            if np.array_equal(p1, gold_out) or np.array_equal(p2, gold_out):
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    return correct, total, acc



import numpy as np

def to_grid_matrix(grid):
    """Convert list-of-lists into numpy array for safety."""
    return np.array(grid, dtype=int)

def infer_global_color_mapping(in_grid, out_grid):
    """
    Try to infer a bijective color mapping from input -> output, if one exists.
    Returns a dict (mapping input_color -> output_color) or None if inconsistent.
    """
    g_in  = to_grid_matrix(in_grid)
    g_out = to_grid_matrix(out_grid)

    if g_in.shape != g_out.shape:
        return None  # shape mismatch, can't do 1-to-1 mapping

    mapping = {}
    for i in range(g_in.shape[0]):
        for j in range(g_in.shape[1]):
            a, b = g_in[i, j], g_out[i, j]
            if a in mapping and mapping[a] != b:
                return None  # conflict in mapping
            mapping[a] = b

    return mapping



# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models   # <-- fixed import

# GRID_SIZE = 30
# NUM_COLORS = 10

# --- Padding + One-Hot encoding ---
# def pad_and_onehot(grid, grid_size=GRID_SIZE, num_colors=NUM_COLORS):
#    padded = np.zeros((grid_size, grid_size), dtype=int)
#    h, w = grid.shape
#    padded[:h, :w] = grid   # put original grid in top-left
#    return tf.one_hot(padded, depth=num_colors)

# --- Dataset prep ---
# X, Y = [], []
# for task_id, task in train_data.items():
#    for pair in task['train']:
#        inp = np.array(pair['input'])
#        out = np.array(pair['output'])
#        X.append(pad_and_onehot(inp))
#        Y.append(pad_and_onehot(out))

# X = np.stack(X)
# Y = np.stack(Y)

# print("Data Shape:", X.shape, Y.shape)   # <-- fixed print

# --- CNN model ---
# model = models.Sequential([
#    layers.Input(shape=(GRID_SIZE, GRID_SIZE, NUM_COLORS)),
#    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
#    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
#    layers.Conv2D(NUM_COLORS, (1,1), activation="softmax")   # <-- fixed Convo2D -> Conv2D
# ])

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # <-- fixed optimize->optimizer

# --- Training ---
# model.fit(X, Y, epochs=20, batch_size=20)

# --- Prediction ---
# def predict_grid(inp_grid):
#    x = pad_and_onehot(np.array(inp_grid))[None, ...]   # <-- fixed pred_and_onehot -> pad_and_onehot
#    pred = model.predict(x)
#    pred_grid = np.argmax(pred[0], axis=-1)   # take argmax over colors
#    return pred_grid



# print("X dtype:", X.dtype, "Y dtype:", Y.dtype)
# print("X shape:", X.shape, "Y shape:", Y.shape)


# def evaluate_on_train(train_data):
#    correct, total = 0, 0
#    for task_id, task in train_data.items():
#        for pair in task['train']:
#            pred = predict_grid(pair['input'])
#            target = np.array(pair['output'])
#            if np.array_equal(pred[:target.shape[0], :target.shape[1]], target):
#                correct += 1
#            total += 1
#    acc = correct / total if total > 0 else 0
#    return correct, total, acc


# model = models.Sequential([
#    layers.Input(shape=(GRID_SIZE, GRID_SIZE, NUM_COLORS)),
#    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
#    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
#    layers.MaxPooling2D((2,2)),
#    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
#    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
#    layers.UpSampling2D((2,2)),   # to restore size
#    layers.Conv2D(NUM_COLORS, (1,1), activation="softmax")
# ])


class Program:
    def __init__(self, ops, name=""):
        # Ensure ops is always a list
        if callable(ops):
            self.ops = [ops]
        else:
            self.ops = list(ops)
        self.name = name  # keep a human-readable name for debugging

    def run(self, grid):
        out = grid
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self):
        return f"Program({self.name})"



correct, total, acc = evaluate_on_train(train_data)
print(f"Train-set accuracy = {correct}/{total} = {acc:.2%}")


   submission = {}

# Loop through every task in test_data
for task_id, task in test_data.items():
    task_predictions = []
    for test_example in task["test"]:
        h = len(test_example["input"])
        w = len(test_example["input"][0])
        
        # Here you’d call your solver. For now, output zeros.
        pred1 = [[0] * w for _ in range(h)]
        pred2 = [[0] * w for _ in range(h)]
        
        task_predictions.append({
            "attempt_1": pred1,
            "attempt_2": pred2
        })
    
    submission[task_id] = task_predictions

# Save to submission.json
with open('submission.json', 'w') as f:
    json.dump(submission, f, indent=4)

print("submission.json created with", len(submission), "tasks.")


