# creating dataset by combining eval challenges, train challenges and train solutions (NO EVAL SOLUTIONS) and then augments dihedrally
from pathlib import Path
import json

DATA_ROOT = Path("/kaggle/input/arc-prize-2024")
TRAIN_CHALLENGES = DATA_ROOT / "arc-agi_training_challenges.json"
TRAIN_SOLUTIONS = DATA_ROOT / "arc-agi_training_solutions.json"
EVAL_CHALLENGES = DATA_ROOT / "arc-agi_evaluation_challenges.json"

OUT_PATH = Path("/kaggle/working/assets/challenges_dihedral_both.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with path.open("r") as f:
        return json.load(f)

def extract_output(sol):
    return sol["output"] if isinstance(sol, dict) and "output" in sol else sol

def attach_train_solutions(challenges, solutions):
    combined = {}
    for task_id, task in challenges.items():
        train_pairs = [dict(pair) for pair in task.get("train", [])]
        test_pairs = [dict(pair) for pair in task.get("test", [])]
        sol_list = solutions.get(task_id)
        if sol_list is None:
            raise KeyError(f"Missing solutions for task {task_id}")
        if len(test_pairs) != len(sol_list):
            raise ValueError(
                f"Solution count mismatch for {task_id}: {len(test_pairs)} test vs {len(sol_list)} sols"
            )
        for pair, sol in zip(test_pairs, sol_list):
            pair["output"] = extract_output(sol)
        task_copy = dict(task)
        task_copy["train"] = train_pairs
        task_copy["test"] = test_pairs
        combined[task_id] = task_copy
    return combined

def move_test_to_train(task_map):
    moved = {}
    for task_id, task in task_map.items():
        train_pairs = [dict(pair) for pair in task.get("train", [])]
        test_pairs = [dict(pair) for pair in task.get("test", [])]
        new_task = dict(task)
        new_task["train"] = train_pairs + test_pairs
        new_task.pop("test", None)
        new_task.pop("name", None)
        moved[task_id] = new_task
    return moved

def merge_task_maps(*maps):
    merged = {}
    for task_map in maps:
        for task_id, task in task_map.items():
            if task_id in merged:
                raise ValueError(f"Duplicate task id: {task_id}")
            merged[task_id] = task
    return {task_id: merged[task_id] for task_id in sorted(merged)}

def copy_grid(grid):
    return [list(row) for row in grid]

def rotate90(grid):
    if not grid:
        return []
    return [list(row) for row in zip(*grid[::-1])]

def rotate180(grid):
    return [list(reversed(row)) for row in reversed(grid)]

def rotate270(grid):
    if not grid:
        return []
    return [list(row) for row in zip(*grid)][::-1]

def flip_horizontal(grid):
    return [list(reversed(row)) for row in grid]

def flip_vertical(grid):
    return [list(row) for row in reversed(grid)]

def flip_main_diagonal(grid):
    if not grid:
        return []
    return [list(row) for row in zip(*grid)]

def flip_anti_diagonal(grid):
    return flip_vertical(rotate90(grid))

TRANSFORMS = [
    ("identity", copy_grid),
    ("rot90", rotate90),
    ("rot180", rotate180),
    ("rot270", rotate270),
    ("flip_horizontal", flip_horizontal),
    ("flip_vertical", flip_vertical),
    ("flip_main_diagonal", flip_main_diagonal),
    ("flip_anti_diagonal", flip_anti_diagonal),
]

def augment_pairs(pairs):
    augmented = []
    for pair in pairs:
        input_grid = pair["input"]
        output_grid = pair.get("output")
        for _, transform in TRANSFORMS:
            new_pair = {"input": transform(input_grid)}
            if output_grid is not None:
                new_pair["output"] = transform(output_grid)
            augmented.append(new_pair)
    return augmented

def augment_dataset(challenges):
    augmented = {}
    for task_id, payload in challenges.items():
        new_payload = dict(payload)
        if "train" in payload:
            new_payload["train"] = augment_pairs(list(payload.get("train", [])))
        if "test" in payload:
            new_payload["test"] = augment_pairs(list(payload.get("test", [])))
        augmented[task_id] = new_payload
    return augmented

train_challenges = load_json(TRAIN_CHALLENGES)
train_solutions = load_json(TRAIN_SOLUTIONS)
eval_challenges = load_json(EVAL_CHALLENGES)

train_both = move_test_to_train(attach_train_solutions(train_challenges, train_solutions))
combined = merge_task_maps(train_both, eval_challenges)

augmented = augment_dataset(combined)
OUT_PATH.write_text(json.dumps(augmented, indent=2))
print(f"Wrote {OUT_PATH} (tasks: {len(augmented)})")



# Choose runconfig
runconfig = ["concept", 11]  # runs the fastest, expect 7%, about x min on the T4
# runconfig = ["concept", 101] # expect 23%, about y min on the T4


import argparse
import importlib
from pathlib import Path
import os
import sys

WORKDIR = Path("/kaggle/working")
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

CODE_DIR = Path("/kaggle/input/mdlarc/pytorch/fp16_for_t4/1")  # read-only mount
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import utils, tinytransformer, train

importlib.reload(utils)  # pick up code changes during iteration
importlib.reload(tinytransformer)
importlib.reload(train)

args = {
    # run config
    "num_workers": 0,
    "device": "cuda",  # 'cuda' | 'mps' | 'cpu'
    "do_validate": False,
    "name": "arc1-cleanenv-30M-vvwide-bs32-101ep-100color-ccdb-18dec0430",  # download file name
    "GPU": "A100-noaugreg",  # just for logging purposes
    # paths - must pass as Path("<path_to_dir>")
    "train_log_file": Path("runs/training_log.txt"),
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": None,  # Path("runs/tiny.pt"),  # or None to start from scratch
    "data_path": Path("assets/challenges_dihedral_both.json"),
    # hyperparameters
    "epochs": runconfig[1],
    "batch_size": 32,
    "val_batch_size": 300,
    "enable_color_aug_train": True,
    "max_color_augments_train": (runconfig[1] - 1),
    "color_aug_seed": 42,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "dropout": 0.1,
    "seed": 42,
    # Model Architecture
    "d_model": 768,  # 128, 256, 512, 768 | 128, 384, 640
    "n_heads": 12,  # 4, 8, 8/16, 12 | 4, 12, 10
    "d_ff": 3072,  # 512, 1024, 2048, 3072 | 512, 1536, 2560
    "n_layers": 4,  # 4, 6, 16, 16 | 24, 28, 24
    # Visibility toggles
    "log_train_strings": False,
    "log_train_limit": 10,
    "log_inference_prompt": False,
}
cfg = argparse.Namespace(**args)

runs_dir = Path("runs")
runs_dir.mkdir(parents=True, exist_ok=True)
with (runs_dir / "config.txt").open("w") as f:
    for k, v in args.items():
        f.write(f"{k}: {v}\n")

model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)


# Training only

from time import perf_counter

t_start = perf_counter()

# ---
# direct
train.train_model(
    cfg,
    model=model,
    dataloader=dataloader,
    dataset=dataset,
    device=device,
    data_path=data_path,
)


# # periodic checkpointing
# cfg.save_path = Path(f"runs/tiny-{cfg.epochs}.pt")
# for i in range(3):
#   if i != 0:
#     cfg.checkpoint_path = cfg.save_path
#     cfg.save_path = Path(f"runs/tiny-{cfg.epochs*(i+1)}.pt")
#   train.train_model(cfg, model=model, dataloader=dataloader, dataset=dataset, device=device, data_path=data_path)
# ---

t_duration = perf_counter() - t_start
print(f"Training took {t_duration:.2f}s")

with open(Path("runs/timing.txt"), "w") as f:
    f.write(f"Training: {t_duration:.4f} s\n")


# cleaning up memory to run inference
import gc
import torch

# 1. Delete global references to free memory
# Deleting 'model' ensures Cell 4 reloads a fresh instance from the checkpoint,
# preventing memory fragmentation or leftover gradients from training.
for name in ["model", "dataset", "dataloader", "optimizer", "scheduler"]:
    if name in globals():
        del globals()[name]

# 2. Reset compiled graph caches (crucial if torch.compile was used)
if hasattr(torch, "_dynamo"):
    torch._dynamo.reset()

# 3. Force garbage collection and clear GPU memory
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print(f"GPU cleaned. Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


import torch
import inference
import tinytransformer
import utils
import sys
import json
import importlib
from pathlib import Path
from time import perf_counter

# Reload modules to pick up changes
importlib.reload(tinytransformer)
importlib.reload(inference)
importlib.reload(utils)

# Define your paths constants
PATH_BOTH = Path("assets/challenges_dihedral_both.json")

# Config List: (Run Name, Max Color Augments, Dataset Path)
EVAL_CONFIGS = [("eval", runconfig[1] - 1, PATH_BOTH)]

# Global settings shared across runs
EVAL_BATCH_SIZE = 260
SPLITS = ["test"]
CHECKPOINT_PATH = Path("runs/tiny.pt")
SOLUTIONS_PRESENT = False
EVAL_TASK_IDS = None  # Set to None to evaluate full dataset, or ["00576224", ...] for specific tasks
LOG_CORRECT_GRIDS = False  # Print the actual grid, IDs, and augmentation indices for fully correct grids


# Helper class for logging to file and console
class TeeLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def run_evaluation_pipeline(run_name, max_color_augments, dataset_path, device):
    print(f"\n{'=' * 60}")
    print(f"STARTING PIPELINE: {run_name} (Color Augs: {max_color_augments})")
    print(f"{'=' * 60}\n")

    # 1. Setup Directories
    base_run_dir = Path("runs") / run_name
    base_run_dir.mkdir(parents=True, exist_ok=True)

    eval_log_path = base_run_dir / "eval_log.txt"
    aaivr_log_path = base_run_dir / "aaivr.txt"
    submission_path = base_run_dir / "submission.json"

    # 2. Update Config
    cfg.checkpoint_path = CHECKPOINT_PATH
    cfg.data_path = dataset_path
    cfg.enable_color_aug_eval = max_color_augments > 0
    cfg.max_color_augments_eval = max_color_augments

    # 3. Build/Rebuild Model & Data
    # We rebuild the dataloader every time to handle the different color augmentation settings
    print("Building model and dataloader for config...")

    # Load checkpoint explicitly to pass to build function
    checkpoint = torch.load(
        cfg.checkpoint_path, map_location=device, weights_only=False
    )

    # Check if model exists in global scope to reuse weights, else create
    global model
    if "model" in globals() and model is not None:
        model.load_state_dict(
            checkpoint["model_state"] if "model_state" in checkpoint else checkpoint,
            strict=False,
        )
        model.eval()
        # Rebuild only dataset/loader
        _, dataset, dataloader, device, _ = train.build_model_and_data(
            cfg, checkpoint=checkpoint
        )
    else:
        model, dataset, dataloader, device, _ = train.build_model_and_data(cfg)

    # 4. Run Inference (Logic from old Cell 3)
    def log_eval(msg):
        print(msg)
        with open(eval_log_path, "a") as f:
            f.write(msg + "\n")

    color_mappings_eval = None
    color_apply_fn = None
    if cfg.enable_color_aug_eval and cfg.max_color_augments_eval > 0:
        color_seed = cfg.color_aug_seed or cfg.seed
        color_mappings_eval = utils.generate_color_mapping_tensors(
            cfg.max_color_augments_eval, color_seed
        )
        color_apply_fn = lambda split: True

    evaluation = inference.evaluate_model_on_dataset(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=EVAL_BATCH_SIZE,
        log_prompts=args["log_inference_prompt"],
        splits=SPLITS,
        color_mappings=color_mappings_eval,
        color_apply_fn=color_apply_fn,
        task_ids=EVAL_TASK_IDS,
        include_targets=SOLUTIONS_PRESENT,
    )

    # Redirect stdout for AAIVR logging
    if hasattr(sys.stdout, "log"):
        sys.stdout = sys.stdout.terminal  # Reset if needed
    sys.stdout = TeeLogger(str(aaivr_log_path))

    try:
        test_results = evaluation.get("test", {}).get("results", [])
        dataset_has_dihedral_augments = "dihedral_both" in str(cfg.data_path)

        aaivr_results = []
        if test_results:
            aaivr_results = utils.run_aaivr_on_results(
                test_results,
                is_dihedral_augmented=dataset_has_dihedral_augments,
                color_aug_seed=cfg.color_aug_seed,
                max_color_augments=cfg.max_color_augments_eval,
            )

            # # Print Stats (will go to console + aaivr.txt)
            # utils.summarize_aaivr_pass_at_k(aaivr_results)
            # if aaivr_results:
            #     tasks_map = {}
            #     for res in aaivr_results:
            #         if res.task_id not in tasks_map:
            #             tasks_map[res.task_id] = []
            #         tasks_map[res.task_id].append(res)

            #     arc_score = 0.0
            #     total_tasks = len(tasks_map)

            #     for t_id, pairs in tasks_map.items():
            #         n_pairs = len(pairs)
            #         if n_pairs > 0:
            #             n_solved = sum(1 for p in pairs if p.pass_at_k)
            #             arc_score += (n_solved / n_pairs)

            #     max_score = total_tasks
            #     pct = (arc_score / max_score * 100) if max_score > 0 else 0.0
            #     print(f"Official ARC style scoring: {arc_score:.2f}/{max_score} ({pct:.2f}%)")
        else:
            print("No test results for AAIVR.")

    finally:
        # Always restore stdout
        if hasattr(sys.stdout, "terminal"):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal

    # 6. Generate Submission (Logic from old Cell 5)
    print(f"Generating submission.json for {run_name}...")
    submission_data = {}
    temp_grouping = {}

    if aaivr_results:
        for item in aaivr_results:
            t_id = item.task_id
            p_idx = item.original_pair_index
            if t_id not in temp_grouping:
                temp_grouping[t_id] = {}

            top_grids = item.selected_outputs[:2]
            if not top_grids:
                top_grids = [[[0]]]  # Fallback

            pair_dict = {
                "attempt_1": top_grids[0],
                "attempt_2": top_grids[1] if len(top_grids) > 1 else top_grids[0],
            }
            temp_grouping[t_id][p_idx] = pair_dict

        for t_id, pairs_map in temp_grouping.items():
            sorted_indices = sorted(pairs_map.keys())
            submission_data[t_id] = [pairs_map[idx] for idx in sorted_indices]

    with open(submission_path, "w") as f:
        json.dump(submission_data, f)

    print(f"Finished {run_name}. Submission saved to {submission_path}")


# --- Execute the Loop (Modified with Timing) ---
timing_path = Path("runs/timing.txt")

for name, aug_count, d_path in EVAL_CONFIGS:  # <--- Unpack 3 items
    t_start = perf_counter()

    run_evaluation_pipeline(name, aug_count, d_path, device)

    t_duration = perf_counter() - t_start
    print(f"Run {name} took {t_duration:.2f}s")

    with open(timing_path, "a") as f:
        f.write(f"Evaluation {name}: {t_duration:.4f} s\n")

print("\nAll evaluation runs completed.")


# visualisation WITHOUT loading solutions
import json
import matplotlib.pyplot as plt
from pathlib import Path

EVAL_SUB_FOLDER = EVAL_CONFIGS[0][0]
submission_file = Path(f"runs/{EVAL_SUB_FOLDER}/submission.json")

if not submission_file.exists():
    print(f"Error: Could not find submission file: {submission_file}")
else:
    # Submission-only visualization (attempts without ground truth)
    with open(submission_file, "r") as f:
        subs = json.load(f)

    print(f"Visualizing submissions for {len(subs)} tasks (no solutions)...")

    for task_id, attempts_list in subs.items():
        for i, attempts in enumerate(attempts_list):
            att1 = attempts.get("attempt_1")
            att2 = attempts.get("attempt_2")

            grids_to_plot = []
            if att1 is not None:
                grids_to_plot.append(att1)
            if att2 is not None:
                grids_to_plot.append(att2)

            if not grids_to_plot:
                print(f"Skipping {task_id} pair {i} (no attempts)")
                continue

            header = f"Task: {task_id} | Pair: {i} | Status: submission-only"
            print(f"Plotting {header}")

            try:
                utils.plot_grids(grids_to_plot, title=header)
            except Exception as e:
                print(f"Skipping plot for {task_id} due to error: {e}")



import json

# Replace these with your actual file paths
SOLUTIONS_FILE = Path("/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json")
SUBMISSION_FILE = Path(f"runs/{EVAL_SUB_FOLDER}/submission.json")


def score(sol_path, sub_path):
    with open(sol_path, "r") as f:
        solutions = json.load(f)

    with open(sub_path, "r") as f:
        submissions = json.load(f)

    calc_score = 0.0
    max_total_score = len(solutions)
    fully_solved_tasks = []

    for task_id, ground_truth_grids in solutions.items():
        # If task is missing in submission, score is 0 for that task
        if task_id not in submissions:
            continue

        task_attempts = submissions[task_id]

        # Determine number of pairs to score for this task
        num_pairs = len(ground_truth_grids)
        pairs_solved = 0

        # Iterate through each test pair in the task
        for i in range(min(len(task_attempts), num_pairs)):
            truth = ground_truth_grids[i]
            attempts = task_attempts[i]

            # Check if either attempt matches the ground truth
            # Python lists compare by value (deep equality) automatically
            if attempts.get("attempt_1") == truth or attempts.get("attempt_2") == truth:
                pairs_solved += 1

        # Add fractional score (e.g., 1/2 = 0.5)
        if num_pairs > 0:
            calc_score += pairs_solved / num_pairs
            if pairs_solved == num_pairs:
                fully_solved_tasks.append(task_id)

    percentage = 100 * (calc_score / max_total_score) if max_total_score > 0 else 0.0
    print(f"Official ARC style scoring: {calc_score}/{max_total_score} ({percentage}%)")
    print(f"Fully correct tasks ({len(fully_solved_tasks)}):")
    for task_id in fully_solved_tasks:
        print(task_id)


if __name__ == "__main__":
    score(SOLUTIONS_FILE, SUBMISSION_FILE)



# Compare mode requires the solutions file; submission mode does not
# visualisation
import json
import matplotlib.pyplot as plt
from pathlib import Path

EVAL_SUB_FOLDER = EVAL_CONFIGS[0][0]
submission_file = Path(f"runs/{EVAL_SUB_FOLDER}/submission.json")
solutions_file = Path("/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json")


if not solutions_file.exists():
    print(
        f"Error: Could not find solutions file for compare mode:\n{solutions_file}"
    )
else:
    # Load Data
    with open(submission_file, "r") as f:
        subs = json.load(f)
    with open(solutions_file, "r") as f:
        sols = json.load(f)

    print(f"Visualizing comparison for {len(subs)} tasks...")

    for task_id, attempts_list in subs.items():
        # Get Ground Truth (list of grids)
        if task_id not in sols:
            print(f"Warning: Task {task_id} not found in solutions.json")
            continue

        gt_grids = sols[task_id]
        print(gt_grids)
        for i, attempts in enumerate(attempts_list):
            if i >= len(gt_grids):
                break

            # 1. Retrieve Grids
            gt = gt_grids[i]
            att1 = attempts.get("attempt_1")
            att2 = attempts.get("attempt_2")

            # 2. Check Correctness
            pass1 = (att1 == gt) if att1 is not None else False
            pass2 = (att2 == gt) if att2 is not None else False

            if pass1 and pass2:
                status = "Pass - both"
            elif pass1:
                status = "Pass - 1"
            elif pass2:
                status = "Pass - 2"
            else:
                status = "Fail"

            # 3. Visualize
            # Construct list: [Ground Truth, Attempt 1, Attempt 2]
            grids_to_plot = [gt]
            if att1 is not None:
                grids_to_plot.append(att1)
            if att2 is not None:
                grids_to_plot.append(att2)

            header = f"Task: {task_id} | Pair: {i} | Status: {status}"
            print(f"Plotting {header}")

            # utils.plot_grids handles the matplotlib figure creation
            try:
                utils.plot_grids(grids_to_plot, title=header)
            except Exception as e:
                print(f"Skipping plot for {task_id} due to error: {e}")

