import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *


import copy
import importlib.util
import json
import re
import sys
import traceback
import numpy as np
import os
import shutil
from tqdm import tqdm 

def simple_verify_program(task_num, examples):
    task_name, task_path = "task_with_imports", "/kaggle/working/task.py"
    spec = importlib.util.spec_from_file_location(task_name, task_path)
    if spec is None:
        print("Error: Unable to import task.py.")
        return
    module = sys.modules[task_name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "p"):
        print("Error: Unable to locate function p() in task.py.")
        return
    program = getattr(module, "p")
    if not callable(program):
        print("Error: Function p() in task.py is not callable.")
        return

    def verify(example_subset):
        right, wrong, expected, error = 0, 0, None, ""
        for example in example_subset:
            example_copy = copy.deepcopy(example)
            try:
                result = program(example_copy["input"])
                result = json.dumps(result)
                result = result.replace("true", "1").replace("false", "0")
                unsafe_chars = re.compile(r"[^0-9,\[\]\s\.]")
                if unsafe_chars.search(result):
                    raise ValueError(f"Invalid output from user code: {result[:500]}")
                result = json.loads(result)
                user_output = np.array(result)
                label_output = np.array(example_copy["output"])
                if numpy.array_equal(user_output, label_output):
                    right += 1
                else:
                    expected = copy.deepcopy(example)
                    wrong += 1
            except:
                error = traceback.format_exc()
                wrong += 1
                #if error: print(f"Error: {error}")
        return right, wrong, expected
    
    arc_agi_right, arc_agi_wrong, arc_agi_expected = verify(examples["train"] + examples["test"])
    arc_gen_right, arc_gen_wrong, arc_gen_expected = verify(examples["arc-gen"])
    #print(f"Results on ARC-AGI examples: {arc_agi_right} pass, {arc_agi_wrong} fail")
    #print(f"Results on ARC-GEN examples: {arc_gen_right} pass, {arc_gen_wrong} fail")
    return arc_agi_right, arc_agi_wrong, arc_gen_right, arc_gen_wrong


import os
import zipfile
from io import BytesIO
import warnings
from itertools import combinations, permutations

# Ignore SyntaxWarning globally
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Paths containing submission.zip
paths = ["/kaggle/input/python-minifier-applied",
         "/kaggle/input/road-to-400-collaboration-39-unsolved-tasks", 
         "/kaggle/input/road-to-400-collaboration", 
         "/kaggle/input/code-golf-ensemble-local-score-391-400-dsl",
         "/kaggle/input/oh-barnacles",
         "/kaggle/input/big-zippa",
         "/kaggle/input/top-scores-remix-with-visualizations-and-insight",
         "/kaggle/input/neuroips-4-of-some-lessons-learned"
        ]

for idx, path in enumerate(paths):
    dest_folder = f"/kaggle/working/submission_{idx}"
    os.makedirs(dest_folder, exist_ok=True)
    
    zip_path = os.path.join(path, "submission.zip")
    if not os.path.exists(zip_path):
        print(f"Zip not found: {zip_path}")
        continue

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_folder)

    print(f"Extracted all files from {zip_path} → {dest_folder}")


src = "/kaggle/input/gcgc-data-solutions"
dst = f"/kaggle/working/submission_{len(paths)}"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".py"):
        shutil.copy(os.path.join(src, f), dst)
paths.append(src)


src = "/kaggle/input/arc-code-golf-starter-solutions"
dst = f"/kaggle/working/submission_{len(paths)}"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".py"):
        shutil.copy(os.path.join(src, f), dst)
paths.append(src)


import os, json, hashlib, shutil
from tqdm import tqdm

# ====== Fixed paths (no flags, no input writes) ======
INPUT_CACHE_PATH = "/kaggle/input/verify-cache/verify_cache.json"
WORKING_CACHE_PATH = "/kaggle/working/verify_cache.json"

# Derive cache dataset info for clear messaging at the end
INPUT_DATASET_DIR = os.path.dirname(INPUT_CACHE_PATH)               # e.g., /kaggle/input/verify-cache
INPUT_DATASET_NAME = os.path.basename(INPUT_DATASET_DIR)            # e.g., verify-cache

def _log(msg: str):
    try:
        tqdm.write(str(msg))
    except Exception:
        print(str(msg))

def _read_json(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        _log(f"[CACHE LOAD WARNING] Could not read {path}: {e}")
    return {}

def _save_json_atomic(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

def _load_cache_union() -> dict:
    """
    Load cache from INPUT and WORKING; return their union.
    - WORKING entries override INPUT for the same key.
    - We never delete keys; later we only add new ones (monotonic growth).
    """
    base = _read_json(INPUT_CACHE_PATH)
    work = _read_json(WORKING_CACHE_PATH)
    if work:
        base.update(work)
    return base

def _hash_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _examples_signature(examples) -> str:
    """
    Make a stable-ish signature of the examples used by verification.
    Falls back to str(...) if not JSON-serializable.
    """
    try:
        data = json.dumps(examples, sort_keys=True, ensure_ascii=False)
        return _hash_bytes(data.encode("utf-8"))
    except Exception:
        return _hash_bytes(str(examples).encode("utf-8"))

def _dataset_slug_from_input_path(p: str) -> str:
    """
    Return the top-level dataset folder under /kaggle/input from a path.
    Robust to extra subfolders or files.
    """
    parts = os.path.normpath(p).split(os.sep)
    try:
        i = parts.index("input")
        if i + 1 < len(parts):
            return parts[i + 1]
    except ValueError:
        pass
    return os.path.basename(os.path.normpath(p))

# ====== Main loop with mandatory miss reporting ======
submission = "/kaggle/working/submission"
os.makedirs(submission, exist_ok=True)

cache = _load_cache_union()
initial_cache_size = len(cache)

cache_hits = 0
cache_misses = 0
had_miss = False  # tracks if *any* miss occurred this run

lb_score = 0
unsolved_tasks = []
num_tasks = 400

for task_num in tqdm(range(1, num_tasks + 1), desc="Tasks"):
    task_name = f"task{task_num:03d}.py"
    smallest_size = None
    smallest_idx = None  # use None to indicate 'unset'

    # Load examples once per task
    examples = load_examples(task_num)
    ex_sig = _examples_signature(examples)

    # Check all submission paths
    for idx in range(len(paths)):
        folder = f"submission_{idx}"
        task_path = os.path.join(folder, task_name)

        if not os.path.exists(task_path):
            continue

        # Prepare cache key
        file_hash = _hash_file(task_path)
        cache_key = f"{task_num}::{idx}::{file_hash}::{ex_sig}"

        # Use cache or compute and add (report every miss)
        if cache_key in cache:
            result = cache[cache_key]
            cache_hits += 1
            agi_right = result["agi_right"]
            agi_wrong = result["agi_wrong"]
            agen_right = result["agen_right"]
            gen_wrong = result["gen_wrong"]
            is_valid = (agi_wrong + gen_wrong == 0)
        else:
            cache_misses += 1
            had_miss = True

            # NEW: print the exact source dataset path for this task
            source_path = paths[idx]  # e.g., /kaggle/input/road-to-400-collaboration-39-unsolved-tasks
            _log(
                f"[CACHE MISS] task={task_num} idx={idx} "
                f"src={source_path} "
                f"file={os.path.basename(task_path)} "
                f"hash={file_hash[:12]} exsig={ex_sig[:12]}"
            )

            # Verify and then append to cache (monotonic growth)
            shutil.copy(task_path, "task.py")
            agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
            is_valid = (agi_wrong + gen_wrong == 0)

            cache[cache_key] = {
                "agi_right": agi_right,
                "agi_wrong": agi_wrong,
                "agen_right": agen_right,
                "gen_wrong": gen_wrong,
                "task_num": task_num,
                "idx": idx,
                "file_hash": file_hash,
                "examples_sig": ex_sig,
            }

        if is_valid:
            size = os.path.getsize(task_path)
            if smallest_size is None or size < smallest_size:
                smallest_size = size
                smallest_idx = idx

    # Decide best file (if any)
    if smallest_idx is not None:
        smallest_file = os.path.join(f"submission_{smallest_idx}", task_name)
        if os.path.exists(smallest_file):
            # Determine source dataset path and slug for display
            source_path = paths[smallest_idx]
            source_slug = _dataset_slug_from_input_path(source_path)

            # Logging (now includes dataset slug)
            if smallest_size is not None:
                mark = ""
            else:
                shutil.copy(smallest_file, "task.py")
                agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
                mark = f" >>> agi_wrong: {agi_wrong}, gen_wrong: {gen_wrong}"

            print(
                f'Task_id = {task_num}: the best solution ({smallest_size} bytes) = '
                f'{source_path} "{source_slug}"' + mark
            )

            shutil.copy(smallest_file, os.path.join(submission, task_name))

            # Score
            if smallest_size is None:
                unsolved_tasks.append(task_num)
                lb_score += 0.001
            else:
                lb_score += max(1, 2500 - smallest_size)
        else:
            unsolved_tasks.append(task_num)
            print(f"Task_id = {task_num}: Selected index had no file present")
    else:
        print(f"Task_id = {task_num}: No solution found")
        unsolved_tasks.append(task_num)

# ====== Wrap-up ======
final_cache_size = len(cache)
added_keys = final_cache_size - initial_cache_size

print("Unsolved tasks:", unsolved_tasks)
print("LB score (approx):", lb_score)
print(f"Cache hits: {cache_hits}, misses: {cache_misses}")
print(f"Cache size: {initial_cache_size} -> {final_cache_size} "
      f"({'grew' if final_cache_size>initial_cache_size else 'unchanged'})")

# If any miss happened, write the full, updated cache to WORKING and print dataset-aware instructions.
if had_miss:
    _save_json_atomic(WORKING_CACHE_PATH, cache)
    print(
        f"\n>>> Updated cache written to {WORKING_CACHE_PATH}\n"
        f"Cache dataset: '{INPUT_DATASET_NAME}' (mounted at {INPUT_DATASET_DIR})\n"
        "To update your input dataset:\n"
        f"  1) Open the 'Output' tab and download verify_cache.json from {WORKING_CACHE_PATH}.\n"
        f"  2) Go to your Kaggle dataset named '{INPUT_DATASET_NAME}' (the one mounted at {INPUT_DATASET_DIR}).\n"
        "  3) Click 'New Version', upload the file as verify_cache.json (overwrite the old one), and save.\n"
        "Future runs will pick it up automatically from /kaggle/input.\n"
        f"(Added {added_keys} new cache entr{'y' if added_keys==1 else 'ies'} this run.)\n"
    )
else:
    print(
        f"\n>>> No cache misses this run; input dataset update not needed.\n"
        f"Cache dataset: '{INPUT_DATASET_NAME}' (mounted at {INPUT_DATASET_DIR})."
    )


import zipfile

submission_zip = f"{submission}.zip"

with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    task_count = 0
    for task_num in range(1, 401):
        task_id = f"{task_num:03d}"
        src_path = f"{submission}/task{task_id}.py"
        
        if os.path.exists(src_path):
            zipf.write(src_path, arcname=f"task{task_id}.py")
            task_count += 1

print(f"Created submission zip with {task_count} tasks: {submission_zip}")

# Display zip file size
zip_size = os.path.getsize(submission_zip)
print(f"Submission zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")


print("LB Score: ", lb_score)
print(f"{len(unsolved_tasks)} Unsolved Tasks: {unsolved_tasks}")


# Task visualization: https://www.kaggle.com/code/jacekwl/a-bit-more-of-code-golf-255-400-visualization

for idx in range(1,len(unsolved_tasks)+1): # only display 100 at once to not cause any memory issues
    task_num = unsolved_tasks[idx-1]
    task_id = f"{task_num:03d}"
    examples = load_examples(task_num)
    show_examples(examples['train'][:1] + examples['test'][:1] )
    plt.figure(idx).suptitle(
        f"task{task_id}" + [' : solved', ' : unsolved'][task_num in unsolved_tasks],
        fontsize=16,color=['green', 'red'][task_num in unsolved_tasks])

