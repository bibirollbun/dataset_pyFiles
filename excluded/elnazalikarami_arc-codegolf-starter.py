# Competition-safe imports (stdlib only)
from pathlib import Path
import json, re, os, sys, itertools, functools, math, random, zipfile




CANDIDATE_INPUT_DIRS = [
    Path("/kaggle/input/google-code-golf-2025"),

    Path("/kaggle/working/google-code-golf-2025") # if you unzip there
]

DATA_DIR = None
for p in CANDIDATE_INPUT_DIRS:
    if p.exists():
        DATA_DIR = p
        break

print("DATA_DIR:", DATA_DIR)



def load_all_tasks(base: Path):
    """
    Load all task JSONs under `base`. Returns a dict: {task_id: {'train': [...], 'test': [...]}}.
    """
    tasks = {}
    if base is None:
        return tasks
    # Common ARC layouts: tasks/ and/or json/; adjust if needed.
    candidates = [base, base/"tasks", base/"json", base/"arc"]
    files = []
    for c in candidates:
        if c.exists():
            files += list(c.rglob("*.json"))
    for fp in sorted(files):
        try:
            obj = json.loads(fp.read_text())
        except Exception as e:
            try:
                obj = json.load(open(fp, "r"))
            except:
                continue
        tid = fp.stem
        tasks[tid] = obj
    return tasks

TASKS = load_all_tasks(DATA_DIR)
print("Loaded tasks:", len(TASKS))
if TASKS:
    sample_id = sorted(TASKS)[0]
    print("Example task id:", sample_id)



def grids_equal(a, b):
    return a == b

def run_fn_on_grid(fn, g):
    return fn([row[:] for row in g])  # defensive copy

def evaluate_on_task(fn, task):
    ok = True
    for ex in task.get("train", []):
        pred = run_fn_on_grid(fn, ex["input"])
        if not grids_equal(pred, ex["output"]):
            ok = False
            break
    return ok

def byte_len_of_function(fn):
    import inspect
    src = inspect.getsource(fn)
    return len(src.encode("utf-8"))

def score_for_bytes(nbytes):
    return max(1, 2500 - nbytes)



# Identity (returns grid unchanged) — tiny but rarely correct.
def id0(g):
    return g

# Constant fill with the most frequent color in the input.
def fill_mode(g):
    from collections import Counter
    flat = sum(g, [])
    m = Counter(flat).most_common(1)[0][0]
    return [[m]*len(g[0]) for _ in g]

# Remap one color to another if it appears (simple color-map example).
def map51(g):
    # replace color 5 with 1
    return [[(1 if v==5 else v) for v in row] for row in g]



# Global default (fall back)
DEFAULT_FN = id0

# Override examples; replace 'task001' with real ids (e.g., hashes) in the dataset.
SOLUTION_REGISTRY = {
    # "task001": fill_mode,
    # "task002": map51,
}

def get_fn_for_task(tid):
    return SOLUTION_REGISTRY.get(tid, DEFAULT_FN)



def validate_some(n=5):
    tids = list(sorted(TASKS))[:n]
    rows = []
    for tid in tids:
        fn = get_fn_for_task(tid)
        ok = evaluate_on_task(fn, TASKS[tid])
        rows.append((tid, ok, byte_len_of_function(fn)))
    return rows

try:
    print(validate_some(5))
except Exception as e:
    print("Validation skipped (no JSONs found yet). Error:", e)



def to_minified_source(fn):
    """
    Convert a Python function object to a tiny single-file source defining `p(g)`.
    You can make this much shorter by hand-golfing.
    """
    import inspect, textwrap, re
    src = inspect.getsource(fn)
    src = re.sub(r"def\s+\w+\(g\):", "def p(g):", src)
    src = textwrap.dedent(src)
    return src

def write_submission_zip(tasks: dict, out_zip="submission.zip"):
    out_path = Path(out_zip)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i, tid in enumerate(sorted(tasks)):
            fn = get_fn_for_task(tid)
            code = to_minified_source(fn)
            fname = f"task{(i+1):03d}.py"
            z.writestr(fname, code)
    print("Wrote:", out_path, "size=", out_path.stat().st_size, "bytes")

def write_blank_400(out_zip="submission.zip"):
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i in range(1,401):
            z.writestr(f"task{i:03d}.py", "def p(g):\n return g\n")
    print("Wrote:", out_zip, "size=", Path(out_zip).stat().st_size, "bytes")



# Try to build from loaded tasks; otherwise write a blank scaffold
if TASKS:
    write_submission_zip(TASKS, "submission.zip")
else:
    write_blank_400("submission.zip")


