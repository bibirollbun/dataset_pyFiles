# Fix 1-D attempt outputs into ARC grid format, validate, and zip
import json
from pathlib import Path
from zipfile import ZipFile

src = Path("/kaggle/working/submission.json")   # adjust if your filename is different
dst = Path("/kaggle/working/submission_fixed.json")
zip_out = Path("/kaggle/working/arc_submission_final.zip")

if not src.exists():
    raise FileNotFoundError(f"Source not found: {src}")

def is_grid_like(v):
    return isinstance(v, list) and v and all(isinstance(r, list) for r in v)

def fix_value(v):
    # If already a grid (list of lists) — keep
    if is_grid_like(v):
        return v
    # If list of ints (1-D) -> wrap as one-row grid
    if isinstance(v, list) and all(isinstance(x, int) for x in v):
        return [v]
    # If single int -> make 1x1 grid
    if isinstance(v, int):
        return [[v]]
    # If empty list -> return [[]] (empty single row)
    if v == []:
        return [[]]
    # If something else (dict?) — try to find nested grid
    if isinstance(v, dict):
        # look for likely keys
        for key in ("output","grid","arr","input","result"):
            if key in v:
                return fix_value(v[key])
    # fallback — return [[0]] and log
    return [[0]]

data = json.loads(src.read_text(encoding='utf-8'))
fixed = {}
fixed_count = 0
total = 0
problems = []

for tid, attempts in data.items():
    total += 1
    if not isinstance(attempts, list) or not attempts:
        problems.append((tid, "no_attempts_or_bad_type"))
        # create a safe default
        fixed[tid] = [{"attempt_1": [[0]]}]
        continue
    new_attempts = []
    for i, att in enumerate(attempts, start=1):
        if not isinstance(att, dict) or not att:
            # unknown format — wrap in attempt_1 as fallback
            new_attempts.append({"attempt_1": [[0]]})
            problems.append((tid, f"bad_attempt_type_{i}"))
            continue
        # For each attempt dict, preserve keys but fix values
        new_att = {}
        for k, v in att.items():
            new_v = fix_value(v)
            if not is_grid_like(new_v):
                problems.append((tid, f"fix_failed_for_{k}"))
            new_att[k] = new_v
            # detect if we changed shape (1D -> fixed)
            if isinstance(v, list) and all(isinstance(x, int) for x in v) and isinstance(new_v, list) and isinstance(new_v[0], list):
                fixed_count += 1
        new_attempts.append(new_att)
    fixed[tid] = new_attempts

# Write fixed submission
dst.write_text(json.dumps(fixed, separators=(",", ":"), ensure_ascii=False))
print("Wrote fixed submission:", dst, "tasks:", len(fixed), "1D→grid fixes:", fixed_count)
if problems:
    print("Sample problems (up to 10):", problems[:10])

# Run a quick structural validator
def validate_submission(path):
    d = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(d, dict):
        return False, "top-level not a dict"
    for tid, attempts in d.items():
        if not isinstance(tid, str):
            return False, f"task id not string: {tid}"
        if not isinstance(attempts, list) or not attempts:
            return False, f"attempts must be non-empty list for {tid}"
        for att in attempts:
            if not isinstance(att, dict) or not att:
                return False, f"each attempt must be a non-empty dict for {tid}"
            # check first value is grid-like
            val = next(iter(att.values()))
            if not (isinstance(val, list) and val and all(isinstance(row, list) for row in val)):
                return False, f"value for {tid} not a grid-like list-of-lists"
    return True, "OK"

ok, msg = validate_submission(dst)
print("Validation:", ok, msg)

# Create zip for easy download
with ZipFile(zip_out, "w") as z:
    z.write(dst, dst.name)
print("Wrote zip:", zip_out, "size:", zip_out.stat().st_size)



# Filesystem hints & explicit candidate paths (updated per your structure)
import os
from pathlib import Path
print("Working dir:", os.getcwd())

# Candidate paths for competition/test JSONs (in order of preference)
CANDIDATE_TEST_JSON_PATHS = [
    # Kaggle competition/dataset usual locations
    "/kaggle/input/arc-prize-2025",
    "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
    "/kaggle/input/COMPETITIONS/ARC Prize 2025",
    "/kaggle/input/COMPETITIONS/ARC Prize 2025/arc-agi_test_challenges.json",
    "/kaggle/input/COMPETITIONS/ARC Prize 2025/arc-agi_evaluation_challenges.json",
    "/kaggle/input/COMPETITIONS/ARC-Prize-2025",
    # fallback dataset names you listed
    "/kaggle/input/arc-solver-optimized",
    "/kaggle/input/arc_optimized_delivery",
    "/kaggle/input/arc-boot",
    # local unpacked dirs (when running locally or after manual unzip)
    "/mnt/data/unpacked_arc-prize-2025_(1)",
    "/mnt/data/unpacked_archive_(2)/arc_optimized_delivery",
    "/mnt/data",
    "/kaggle/working",
]

# Candidate solver package roots (package code locations)
CANDIDATE_SOLVER_PKG_ROOTS = [
    # typical Kaggle dataset layout
    "/kaggle/input/arc-solver-optimized/solver_package_optimized/solver_package_optimized",
    "/kaggle/input/arc_optimized_delivery/solver_package_optimized/solver_package_optimized",
    # local unpacked copies
    "/mnt/data/unpacked_archive_(2)/arc_optimized_delivery/solver_package_optimized/solver_package_optimized",
    "/mnt/data/unpacked_archive_(2)/arc_optimized_delivery/solver_package_optimized",
    "/mnt/data/unpacked_arc-prize-2025_(1)",
    "/mnt/data",
]

# Utility: find the first existing path from a list (file or dir)
def find_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

print("First existing test path:", find_existing(CANDIDATE_TEST_JSON_PATHS))
print("First existing solver pkg root:", find_existing(CANDIDATE_SOLVER_PKG_ROOTS))


# Integrate optimized ARC solver package into the notebook environment (updated with candidate roots)
import sys, os
from pathlib import Path

# Use candidate roots defined earlier in CANDIDATE_SOLVER_PKG_ROOTS if available, else fallback to common locations
candidates = globals().get("CANDIDATE_SOLVER_PKG_ROOTS", [
    "/kaggle/input/arc-solver-optimized/solver_package_optimized/solver_package_optimized",
    "/kaggle/input/arc_optimized_delivery/solver_package_optimized/solver_package_optimized",
    "/mnt/data/unpacked_archive_(2)/arc_optimized_delivery/solver_package_optimized/solver_package_optimized",
    "/mnt/data/unpacked_archive_(2)/arc_optimized_delivery/solver_package_optimized",
    "/mnt/data/unpacked_arc-prize-2025_(1)",
    "/mnt/data"
])
solver_pkg_root = None
for p in candidates:
    if Path(p).exists():
        solver_pkg_root = str(Path(p))
        break

if solver_pkg_root is None:
    # As a last resort, try to locate a directory named 'arc_solver_upgrade' under /kaggle/input or /mnt/data
    fallback = None
    for base in ["/kaggle/input", "/mnt/data"]:
        for root, dirs, files in os.walk(base):
            if "arc_solver_upgrade" in dirs:
                fallback = os.path.join(root, "arc_solver_upgrade")
                break
        if fallback:
            solver_pkg_root = fallback
            break

if solver_pkg_root is None:
    print("WARNING: No solver package root found in candidate locations. Please ensure the solver package is available under one of CANDIDATE_SOLVER_PKG_ROOTS.")
else:
    if solver_pkg_root not in sys.path:
        sys.path.insert(0, solver_pkg_root)
    print("Inserted solver package path:", solver_pkg_root)

# Try importing common entrypoints
try:
    import unified_arc_solver_v3_4 as arc_unified
    print("Imported unified_arc_solver_v3_4 as arc_unified")
except Exception as e:
    print("Could not import unified_arc_solver_v3_4:", e)
try:
    import expert_solver as expert_solver_module
    print("Imported expert_solver module")
except Exception as e:
    print("Could not import expert_solver:", e)


# Imports & quick info
import json, glob, os
from pathlib import Path
print("Notebook ready. Current working dir:", os.getcwd())
print("Searching /kaggle/input for ARC-style test files...")


# list_test_tasks(): discover and return iterable of (task_id, [test_inputs])
import json, glob, os
from pathlib import Path

def _try_parse_taskfile(p):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None
    tasks = []
    # Case A: top-level dict mapping task_id -> task obj (common ARC format)
    if isinstance(data, dict):
        for tid, obj in data.items():
            if isinstance(obj, dict) and ('test' in obj or 'train' in obj):
                tests = []
                # obj['test'] may be list of grids or list of {'input':..}
                for t in obj.get('test', []):
                    if isinstance(t, dict) and 'input' in t:
                        tests.append(t['input'])
                    else:
                        tests.append(t)
                # If 'test' empty but 'train' present, skip (we need test inputs)
                if tests:
                    tasks.append((tid, tests))
        if tasks:
            return tasks
    # Case B: list of task objects (each with 'id' or similar)
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                tid = entry.get('id') or entry.get('task_id') or entry.get('name')
                if not tid:
                    tid = str(hash(json.dumps(entry)))[-8:]
                tests = []
                if 'test' in entry:
                    for t in entry['test']:
                        if isinstance(t, dict) and 'input' in t:
                            tests.append(t['input'])
                        else:
                            tests.append(t)
                if tests:
                    tasks.append((tid, tests))
        if tasks:
            return tasks
    return None

def list_test_tasks():
    # Use explicit candidate paths if present
    candidates = globals().get("CANDIDATE_TEST_JSON_PATHS", [
        "/kaggle/input/arc-prize-2025",
        "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
        "/kaggle/input/COMPETITIONS/ARC Prize 2025",
        "/kaggle/input/COMPETITIONS/ARC Prize 2025/arc-agi_test_challenges.json",
        "/kaggle/input/COMPETITIONS/ARC Prize 2025/arc-agi_evaluation_challenges.json",
        "/kaggle/input/arc-solver-optimized",
        "/kaggle/input/arc_optimized_delivery",
        "/mnt/data/unpacked_arc-prize-2025_(1)",
        "/mnt/data/unpacked_archive_(2)/arc_optimized_delivery",
        "/mnt/data",
        "/kaggle/working",
    ])

    # If candidate is a directory, look for relevant JSON files inside it
    for c in candidates:
        p = Path(c)
        if not p.exists():
            continue
        if p.is_file() and p.suffix == ".json":
            res = _try_parse_taskfile(str(p))
            if res:
                print("Loaded test tasks from:", p)
                return res
        if p.is_dir():
            # check known filenames in the directory
            for fname in ["arc-agi_test_challenges.json", "arc-agi_evaluation_challenges.json", "arc-agi_test_challenges.json", "arc-agi_training_challenges.json", "sample_submission.json"]:
                fp = p / fname
                if fp.exists():
                    res = _try_parse_taskfile(str(fp))
                    if res:
                        print("Loaded test tasks from:", fp)
                        return res
            # otherwise try any json
            for fp in p.rglob("*.json"):
                res = _try_parse_taskfile(str(fp))
                if res:
                    print("Loaded test tasks from:", fp)
                    return res

    # Last resort: search all json under /kaggle/input and /mnt/data
    for base in ["/kaggle/input", "/mnt/data", "/kaggle/working"]:
        for fp in Path(base).rglob("*.json"):
            res = _try_parse_taskfile(str(fp))
            if res:
                print("Loaded test tasks from (fallback):", fp)
                return res
    raise FileNotFoundError("No ARC-style test JSON found under candidate locations. Place your test JSON under one of the candidate paths.")


# Integrated predict_one: uses integrated solver package when available.
import inspect, sys

def _is_grid_like(obj):
    if isinstance(obj, list) and obj and all(isinstance(row, list) for row in obj):
        return True
    return False

def _make_pairs_from_task(task):
    # task is expected to be a dict with 'train' (list of pairs) and 'test' (list of pairs)
    train_pairs = []
    test_pairs = []
    if isinstance(task, dict):
        if 'train' in task and isinstance(task['train'], list):
            for p in task['train']:
                # allow either raw grid or {'input':..., 'output':...}
                if isinstance(p, dict):
                    inp = p.get('input') or p.get('inp') or p.get('x') or p.get('input_grid') or None
                    out = p.get('output') or p.get('output_grid') or p.get('y') or None
                    entry = {}
                    if inp is not None: entry['input'] = inp
                    if out is not None: entry['output'] = out
                    if entry: train_pairs.append(entry)
                elif isinstance(p, list):
                    train_pairs.append({'input': p})
        if 'test' in task and isinstance(task['test'], list):
            for p in task['test']:
                if isinstance(p, dict) and 'input' in p:
                    test_pairs.append({'input': p['input']})
                elif isinstance(p, list):
                    test_pairs.append({'input': p})
    return train_pairs, test_pairs

def predict_one(task_id, task_obj):
    # task_obj may be a dict with 'train' and 'test'; or the old loader format (task_id, [tests])
    # prefer using unified solver if available
    g = globals()
    # prepare train/test pairs in expected format
    train_pairs, test_pairs = _make_pairs_from_task(task_obj if isinstance(task_obj, dict) else {'test': task_obj if isinstance(task_obj, list) else []})
    # Try arc_unified.solve_arc_task(train_pairs, test_pairs)
    try:
        arc = g.get('arc_unified')
        if arc and hasattr(arc, 'solve_arc_task'):
            # call solver - note: solver expects List[Pair] for train and test
            out = arc.solve_arc_task(train_pairs, test_pairs)
            # solver may return list of outputs per test, or a grid directly for single test; try to handle
            if isinstance(out, list) and out:
                # assume list of grids corresponding to test_pairs
                return out[0] if _is_grid_like(out[0]) else out
            if _is_grid_like(out):
                return out
    except Exception as e:
        print("arc_unified.solve_arc_task failed:", e)
    # Try expert_solver's solve_one or solve functions
    try:
        exp = g.get('expert_solver_module')
        if exp:
            # try solve_one(train_pairs, test_pairs) or exp.solve(train_pairs, test_pairs)
            if hasattr(exp, 'solve_one'):
                try:
                    res = exp.solve_one(train_pairs, test_pairs)
                    if _is_grid_like(res):
                        return res
                except Exception:
                    pass
            if hasattr(exp, 'solve'):
                try:
                    res = exp.solve(train_pairs, test_pairs)
                    if _is_grid_like(res):
                        return res
                except Exception:
                    pass
    except Exception as e:
        print("expert_solver attempts failed:", e)
    # fallback: if task_obj is dict with test, return the first test input
    if isinstance(task_obj, dict) and 'test' in task_obj and isinstance(task_obj['test'], list) and task_obj['test']:
        first = task_obj['test'][0]
        if isinstance(first, dict) and 'input' in first:
            return first['input']
        if isinstance(first, list):
            return first
    # old-style: if task_obj is a list of grids, return the first
    if isinstance(task_obj, list):
        return task_obj[0] if task_obj else [[0]]
    return [[0]]


# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()



# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

arc_prize_2025_path = kagglehub.competition_download('arc-prize-2025')
skyeloitz_arc_solver_optimized_path = kagglehub.dataset_download('skyeloitz/arc-solver-optimized')
skyeloitz_arc_boot_path = kagglehub.dataset_download('skyeloitz/arc-boot')

print('Data source import complete.')



# Diagnostics: list /kaggle/input and /kaggle/working (run first)
from pathlib import Path
import os, json # Corrected: json should be imported directly
print("=== /kaggle/input (top 200 entries) ===")
p = Path("/kaggle/input")
if p.exists():
    for i, entry in enumerate(sorted(p.iterdir())):
        if i>200: break
        print(i, entry.name, "(dir)" if entry.is_dir() else "")
else:
    print("/kaggle/input not present in this environment")
print("\n=== /kaggle/working (top 200 entries) ===")
pw = Path("/kaggle/working")
if pw.exists():
    for i, entry in enumerate(sorted(pw.iterdir())):
        if i>200: break
        print(i, entry.name, "(dir)" if entry.is_dir() else "")
else:
    print("/kaggle/working not present in this environment")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: # Helpers: config, logging, atomic write, JSON loaders, and candidate merging

# 
# # Helpers: config, logging, atomic write, JSON loaders, and candidate merging
# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # CONFIG
# WORKING_DIR = Path("/kaggle/working")
# INPUT_DIR = Path("/kaggle/input")
# REPORT_PATH = WORKING_DIR / "execution_report.json"
# FINAL_SUB = WORKING_DIR / "submission.json"
# SOLVER_WEIGHTS_PATH = WORKING_DIR / "solver_weights.json"
# LOG_PATH = WORKING_DIR / "finalize.log"
# SMOOTHING_ALPHA = 1.0
# WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
# # Logging (console + file)
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: # Train solver weights from training solutions (returns solver_weights)

# 
# # Train solver weights from training solutions (returns solver_weights)
# from collections import defaultdict
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights:
#                         raw_weights[s] = 0.01; total += 0.01
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#     return solver_weights, solver_stats


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: # Finalize submission using weighted voting

# # Finalize submission using weighted voting
# def find_and_load_challenge(candidates):
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try: return json.loads(p.read_text(encoding="utf-8")), p
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try: return json.loads(f.read_text(encoding="utf-8")), f
#                     except Exception: continue
#             for f in p.rglob("*.json"):
#                 try: return json.loads(f.read_text(encoding="utf-8")), f
#                 except Exception: continue
#     for f in INPUT_DIR.rglob("*arc*challenges*.json"):
#         try: return json.loads(f.read_text(encoding="utf-8")), f
#         except Exception: continue
#     return None, None
# 
# def finalize_submission(partial, solver_weights, challenge=None):
#     # load challenge if not provided
#     if challenge is None:
#         challenge_candidates = [WORKING_DIR, INPUT_DIR / "arc-prize-2025", INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         challenge_path_used = "provided"
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue
#             scores = defaultdict(float); contributors = defaultdict(list)
#             for c in cands:
#                 sname = c.get("solver") or "unknown"
#                 g = c.get("grid")
#                 if g is None: continue
#                 conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                 w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                 scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             if not scores:
#                 chosen_grids.append(cands[0].get("grid")); continue
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# # Run helper to produce submission and write outputs
# def run_all(dry_run=False):
#     # discover partials
#     partial_paths = []
#     for p in WORKING_DIR.glob("*.json"):
#         if p.name in {FINAL_SUB.name, REPORT_PATH.name, SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in INPUT_DIR.iterdir() if INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
#     # load training solutions
#     train_solution_files = []
#     for d in INPUT_DIR.iterdir() if INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in INPUT_DIR.rglob("*.json") if INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=SMOOTHING_ALPHA)
#     atomic_write_json(SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", SOLVER_WEIGHTS_PATH)
#     final_submission, report = finalize_submission(partial, solver_weights, challenge=None)
#     if not dry_run:
#         atomic_write_json(FINAL_SUB, final_submission)
#         report["summary"] = {"total_tasks": len(final_submission)}
#         atomic_write_json(REPORT_PATH, report)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights)}


import json
from pathlib import Path

submission_path = Path("/kaggle/working/submission.json")

if submission_path.exists():
    with open(submission_path, 'r', encoding='utf-8') as f:
        submission_content = json.load(f)
    display(submission_content)
else:
    print(f"The file {submission_path} does not exist.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json

# import json
# import os
# from pathlib import Path
# 
# class Config:
#     def __init__(self, working_dir=None, input_dir=None, smoothing_alpha=None):
#         self.WORKING_DIR = Path(working_dir) if working_dir else Path("/kaggle/working")
#         self.INPUT_DIR = Path(input_dir) if input_dir else Path("/kaggle/input")
# 
#         self.REPORT_PATH = self.WORKING_DIR / "execution_report.json"
#         self.FINAL_SUB = self.WORKING_DIR / "submission.json"
#         self.SOLVER_WEIGHTS_PATH = self.WORKING_DIR / "solver_weights.json"
#         self.LOG_PATH = self.WORKING_DIR / "finalize.log"
#         self.SMOOTHING_ALPHA = smoothing_alpha if smoothing_alpha is not None else 1.0
# 
#         self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def load_from_json(self, config_file_path):
#         if not Path(config_file_path).is_file():
#             return
#         with open(config_file_path, 'r', encoding='utf-8') as f:
#             overrides = json.load(f)
#         for key, value in overrides.items():
#             if hasattr(self, key):
#                 # Handle Path objects correctly
#                 if 'DIR' in key.upper() or 'PATH' in key.upper() or 'SUB' in key.upper():
#                     setattr(self, key, Path(value))
#                 else:
#                     setattr(self, key, value)
# 
# print("Config class defined.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#     return solver_weights, solver_stats
# 
# def find_and_load_challenge(config, candidates):
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try: return json.loads(p.read_text(encoding="utf-8")), p
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try: return json.loads(f.read_text(encoding="utf-8")), f
#                     except Exception: continue
#             for f in p.rglob("*.json"):
#                 try: return json.loads(f.read_text(encoding="utf-8")), f
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try: return json.loads(f.read_text(encoding="utf-8")), f
#         except Exception: continue
#     return None, None
# 
# def finalize_submission(partial, solver_weights, config, challenge=None):
#     # load challenge if not provided
#     if challenge is None:
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         challenge_path_used = "provided"
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
#             scores = defaultdict(float); contributors = defaultdict(list)
#             for c in cands:
#                 sname = c.get("solver") or "unknown"
#                 g = c.get("grid")
#                 if g is None: continue
#                 conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                 w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                 scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             if not scores:
#                 chosen_grids.append(cands[0].get("grid")); continue # Fallback if no valid grids from candidates
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     final_submission, report = finalize_submission(partial, solver_weights, config, challenge=None)
#     if not dry_run:
#         atomic_write_json(config.FINAL_SUB, final_submission)
#         report["summary"] = {"total_tasks": len(final_submission), "config_used": {k: str(v) for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}}
#         atomic_write_json(config.REPORT_PATH, report)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": config.__dict__}
# 
# print("Helpers and run_all function updated to use Config class.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0, "total_confidence_sum":0.0, "task_ids_attempted":set()})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 solver_stats[solver]["task_ids_attempted"].add(tid)
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#                     conf = c.get("confidence")
#                     if conf is not None and isinstance(conf,(int,float)):
#                         solver_stats[solver]["total_confidence_sum"] += float(conf)
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
# 
#             # Calculate additional metrics
#             st["average_confidence"] = st["total_confidence_sum"] / st["correct"] if st["correct"] > 0 else 0.0
#             st["num_tasks_attempted"] = len(st["task_ids_attempted"])
#             # Convert set to list for JSON serialization if needed later
#             st["task_ids_attempted"] = sorted(list(st["task_ids_attempted"]))
# 
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#                         solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                         solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"]))
# 
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#             for s in sols:
#                 solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                 solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"]))
# 
#     return solver_weights, solver_stats
# 
# def find_and_load_challenge(config, candidates):
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try: return json.loads(p.read_text(encoding="utf-8")), p
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try: return json.loads(f.read_text(encoding="utf-8")), f
#                     except Exception: continue
#             for f in p.rglob("*.json"):
#                 try: return json.loads(f.read_text(encoding="utf-8")), f
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try: return json.loads(f.read_text(encoding="utf-8")), f
#         except Exception: continue
#     return None, None
# 
# def finalize_submission(partial, solver_weights, config, challenge=None):
#     # load challenge if not provided
#     if challenge is None:
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         challenge_path_used = "provided"
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
#             scores = defaultdict(float); contributors = defaultdict(list)
#             for c in cands:
#                 sname = c.get("solver") or "unknown"
#                 g = c.get("grid")
#                 if g is None: continue
#                 conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                 w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                 scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             if not scores:
#                 chosen_grids.append(cands[0].get("grid")); continue # Fallback if no valid grids from candidates
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     # Optionally save solver_stats as well for diagnostics/visualization
#     atomic_write_json(config.WORKING_DIR / "solver_stats.json", {s: {k: v if not isinstance(v, set) else sorted(list(v)) for k, v in stats.items()} for s, stats in solver_stats.items()})
# 
#     final_submission, report = finalize_submission(partial, solver_weights, config, challenge=None)
#     if not dry_run:
#         atomic_write_json(config.FINAL_SUB, final_submission)
#         report["summary"] = {"total_tasks": len(final_submission), "config_used": {k: str(v) for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}}
#         report["solver_metrics"] = solver_stats # Add solver statistics to the report
#         atomic_write_json(config.REPORT_PATH, report)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": config.__dict__}


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # --- Config Class (Updated) ---
# class Config:
#     def __init__(self, working_dir=None, input_dir=None, smoothing_alpha=None, voting_strategy=None):
#         self.WORKING_DIR = Path(working_dir) if working_dir else Path("/kaggle/working")
#         self.INPUT_DIR = Path(input_dir) if input_dir else Path("/kaggle/input")
# 
#         self.REPORT_PATH = self.WORKING_DIR / "execution_report.json"
#         self.FINAL_SUB = self.WORKING_DIR / "submission.json"
#         self.SOLVER_WEIGHTS_PATH = self.WORKING_DIR / "solver_weights.json"
#         self.LOG_PATH = self.WORKING_DIR / "finalize.log"
#         self.SMOOTHING_ALPHA = smoothing_alpha if smoothing_alpha is not None else 1.0
#         self.VOTING_STRATEGY = voting_strategy if voting_strategy else "weighted" # New: Default voting strategy
# 
#         self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def load_from_json(self, config_file_path):
#         if not Path(config_file_path).is_file():
#             return
#         with open(config_file_path, 'r', encoding='utf-8') as f:
#             overrides = json.load(f)
#         for key, value in overrides.items():
#             if hasattr(self, key):
#                 # Handle Path objects correctly
#                 if 'DIR' in key.upper() or 'PATH' in key.upper() or 'SUB' in key.upper():
#                     setattr(self, key, Path(value))
#                 else:
#                     setattr(self, key, value)
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0, "total_confidence_sum":0.0, "task_ids_attempted":set()})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 solver_stats[solver]["task_ids_attempted"].add(tid)
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#                     conf = c.get("confidence")
#                     if conf is not None and isinstance(conf,(int,float)):
#                         solver_stats[solver]["total_confidence_sum"] += float(conf)
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
# 
#             # Calculate additional metrics
#             st["average_confidence"] = st["total_confidence_sum"] / st["correct"] if st["correct"] > 0 else 0.0
#             st["num_tasks_attempted"] = len(st["task_ids_attempted"])
#             # Convert set to list for JSON serialization if needed later
#             st["task_ids_attempted"] = sorted(list(st["task_ids_attempted"])) # Convert set to list here
# 
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#                         solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                         solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#             for s in sols:
#                 solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                 solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#     return solver_weights, solver_stats
# 
# def find_and_load_challenge(config, candidates):
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try: return json.loads(p.read_text(encoding="utf-8")), p
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try: return json.loads(f.read_text(encoding="utf-8")), f
#                     except Exception: continue
#             for f in p.rglob("*.json"):
#                 try: return json.loads(f.read_text(encoding="utf-8")), f
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try: return json.loads(f.read_text(encoding="utf-8")), f
#         except Exception: continue
#     return None, None
# 
# # --- finalize_submission (Updated) ---
# def finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy="weighted"):
#     # load challenge if not provided
#     if challenge is None:
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         challenge_path_used = "provided"
# 
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
# 
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
# 
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
# 
#             scores = defaultdict(float)
#             contributors = defaultdict(list)
# 
#             if voting_strategy == "weighted":
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             elif voting_strategy == "unweighted":
#                 # Simple majority vote
#                 for c in cands:
#                     g = c.get("grid")
#                     if g is None: continue
#                     k = grid_key(g)
#                     scores[k] += 1 # Each occurrence counts as one vote
#                     contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#             elif voting_strategy == "confidence-based":
#                 # Use confidence as the primary weight. If no confidence, treat as 0 or 1.
#                 any_confidence = False
#                 for c in cands:
#                     if c.get("confidence") is not None and isinstance(c.get("confidence"), (int, float)):
#                         any_confidence = True
#                         break
# 
#                 if not any_confidence:
#                     logger.warning(f"Task {tid}, test {ti}: No confidence scores found for confidence-based voting. Falling back to unweighted voting.")
#                     # Fallback to unweighted voting if no confidence is available
#                     for c in cands:
#                         g = c.get("grid")
#                         if g is None: continue
#                         k = grid_key(g)
#                         scores[k] += 1
#                         contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#                 else:
#                     for c in cands:
#                         sname = c.get("solver") or "unknown"
#                         g = c.get("grid")
#                         if g is None: continue
#                         conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                         score = conf_factor # Only confidence matters
#                         k = grid_key(g)
#                         scores[k] += score; contributors[k].append({"solver":sname,"weight":1,"confidence":conf})
#             else:
#                 logger.warning(f"Unknown voting strategy '{voting_strategy}'. Falling back to weighted voting.")
#                 # Default to weighted if strategy is unknown
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
# 
#             if not scores:
#                 chosen_grids.append([[0]]); continue # Fallback if no valid grids from candidates or voting method yields no scores
# 
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     # Optionally save solver_stats as well for diagnostics/visualization
#     atomic_write_json(config.WORKING_DIR / "solver_stats.json", {s: {k: v if not isinstance(v, list) else v for k, v in stats.items()} for s, stats in solver_stats.items()}) # Fix set serialization to list here
# 
#     final_submission, report = finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy=config.VOTING_STRATEGY)
#     if not dry_run:
#         atomic_write_json(config.FINAL_SUB, final_submission)
#         report["summary"] = {"total_tasks": len(final_submission), "config_used": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}} # Fix path serialization
#         report["solver_metrics"] = {s: {k: v if not isinstance(v, set) else sorted(list(v)) for k, v in stats.items()} for s, stats in solver_stats.items()} # Add solver statistics to the report, ensure set to list
#         atomic_write_json(config.REPORT_PATH, report)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}} # Fix path serialization here
# 
# print("Config class and relevant functions updated to support multiple voting strategies.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # --- Config Class (Updated) ---
# class Config:
#     def __init__(self, working_dir=None, input_dir=None, smoothing_alpha=None, voting_strategy=None):
#         self.WORKING_DIR = Path(working_dir) if working_dir else Path("/kaggle/working")
#         self.INPUT_DIR = Path(input_dir) if input_dir else Path("/kaggle/input")
# 
#         self.REPORT_PATH = self.WORKING_DIR / "execution_report.json"
#         self.FINAL_SUB = self.WORKING_DIR / "submission.json"
#         self.SOLVER_WEIGHTS_PATH = self.WORKING_DIR / "solver_weights.json"
#         self.LOG_PATH = self.WORKING_DIR / "finalize.log"
#         self.SMOOTHING_ALPHA = smoothing_alpha if smoothing_alpha is not None else 1.0
#         self.VOTING_STRATEGY = voting_strategy if voting_strategy else "weighted" # New: Default voting strategy
# 
#         self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def load_from_json(self, config_file_path):
#         if not Path(config_file_path).is_file():
#             return
#         with open(config_file_path, 'r', encoding='utf-8') as f:
#             overrides = json.load(f)
#         for key, value in overrides.items():
#             if hasattr(self, key):
#                 # Handle Path objects correctly
#                 if 'DIR' in key.upper() or 'PATH' in key.upper() or 'SUB' in key.upper():
#                     setattr(self, key, Path(value))
#                 else:
#                     setattr(self, key, value)
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0, "total_confidence_sum":0.0, "task_ids_attempted":set()})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 solver_stats[solver]["task_ids_attempted"].add(tid)
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#                     conf = c.get("confidence")
#                     if conf is not None and isinstance(conf,(int,float)):
#                         solver_stats[solver]["total_confidence_sum"] += float(conf)
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
# 
#             # Calculate additional metrics
#             st["average_confidence"] = st["total_confidence_sum"] / st["correct"] if st["correct"] > 0 else 0.0
#             st["num_tasks_attempted"] = len(st["task_ids_attempted"])
#             # Convert set to list for JSON serialization if needed later
#             st["task_ids_attempted"] = sorted(list(st["task_ids_attempted"]))
# 
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#                         solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                         solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"]))
# 
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#             for s in sols:
#                 solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                 solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"]))
# 
#     return solver_weights, solver_stats
# 
# def find_and_load_challenge(config, candidates):
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try: return json.loads(p.read_text(encoding="utf-8")), p
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try: return json.loads(f.read_text(encoding="utf-8")), f
#                     except Exception: continue
#             for f in p.rglob("*.json"):
#                 try: return json.loads(f.read_text(encoding="utf-8")), f
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try: return json.loads(f.read_text(encoding="utf-8")), f
#         except Exception: continue
#     return None, None
# 
# # --- finalize_submission (Updated) ---
# def finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy="weighted"):
#     # load challenge if not provided
#     if challenge is None:
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         challenge_path_used = "provided"
# 
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
# 
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
# 
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
# 
#             scores = defaultdict(float)
#             contributors = defaultdict(list)
# 
#             if voting_strategy == "weighted":
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             elif voting_strategy == "unweighted":
#                 # Simple majority vote
#                 for c in cands:
#                     g = c.get("grid")
#                     if g is None: continue
#                     k = grid_key(g)
#                     scores[k] += 1 # Each occurrence counts as one vote
#                     contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#             elif voting_strategy == "confidence-based":
#                 # Use confidence as the primary weight. If no confidence, treat as 0 or 1.
#                 any_confidence = False
#                 for c in cands:
#                     if c.get("confidence") is not None and isinstance(c.get("confidence"), (int, float)):
#                         any_confidence = True
#                         break
# 
#                 if not any_confidence:
#                     logger.warning(f"Task {tid}, test {ti}: No confidence scores found for confidence-based voting. Falling back to unweighted voting.")
#                     # Fallback to unweighted voting if no confidence is available
#                     for c in cands:
#                         g = c.get("grid")
#                         if g is None: continue
#                         k = grid_key(g)
#                         scores[k] += 1
#                         contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#                 else:
#                     for c in cands:
#                         sname = c.get("solver") or "unknown"
#                         g = c.get("grid")
#                         if g is None: continue
#                         conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                         score = conf_factor # Only confidence matters
#                         k = grid_key(g)
#                         scores[k] += score; contributors[k].append({"solver":sname,"weight":1,"confidence":conf})
#             else:
#                 logger.warning(f"Unknown voting strategy '{voting_strategy}'. Falling back to weighted voting.")
#                 # Default to weighted if strategy is unknown
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
# 
#             if not scores:
#                 chosen_grids.append([[0]]); continue # Fallback if no valid grids from candidates or voting method yields no scores
# 
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# # --- New Helper Function for Benchmarking ---
# def grid_to_set(grid):
#     if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
#         return set() # Return empty set for invalid or empty grids
#     s = set()
#     for r_idx, row in enumerate(grid):
#         for c_idx, color in enumerate(row):
#             s.add((r_idx, c_idx, color))
#     return s
# 
# def grid_jaccard_similarity(grid1, grid2):
#     set1 = grid_to_set(grid1)
#     set2 = grid_to_set(grid2)
#     if not set1 and not set2: # Both grids are empty or invalid, consider them 100% similar in a trivial way
#         return 1.0
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     if union == 0:
#         return 0.0 # Should not happen if at least one set is non-empty, but for safety
#     return intersection / union
# 
# # --- New Benchmark Function ---
# def benchmark_submission(final_submission, true_solutions):
#     total_correct_predictions = 0
#     total_predictions = 0
#     total_jaccard_scores = 0.0
#     task_benchmarks = {}
# 
#     for tid, predicted_output in final_submission.items():
#         true_grids = get_solution_grids_for_task(tid, true_solutions)
#         if not true_grids:
#             logger.debug(f"No true solutions found for task {tid}, skipping benchmarking.")
#             continue
# 
#         # Predicted output is typically a list containing a single dict of attempts
#         if isinstance(predicted_output, list) and predicted_output:
#             predicted_attempts = predicted_output[0]
#         else:
#             logger.warning(f"Unexpected format for predicted output for task {tid}: {predicted_output}")
#             continue
# 
#         num_tests = len(true_grids)
#         task_benchmarks[tid] = {"tests": []}
# 
#         for i in range(num_tests):
#             # Get predicted grid for the current test
#             pred_grid = predicted_attempts.get(f"attempt_{i+1}", [[0]]) # Default to empty grid
#             true_grid = true_grids[i]
# 
#             correct = 0
#             jaccard_score = 0.0
# 
#             if grid_key(pred_grid) == grid_key(true_grid):
#                 correct = 1
#                 jaccard_score = 1.0 # Jaccard is 1.0 if grids are identical
#             else:
#                 jaccard_score = grid_jaccard_similarity(pred_grid, true_grid)
# 
#             total_correct_predictions += correct
#             total_jaccard_scores += jaccard_score
#             total_predictions += 1
# 
#             task_benchmarks[tid]["tests"].append({
#                 "test_idx": i,
#                 "accuracy": correct,
#                 "jaccard_similarity": jaccard_score
#             })
# 
#     overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0.0
#     average_jaccard = total_jaccard_scores / total_predictions if total_predictions > 0 else 0.0
# 
#     return {
#         "overall_accuracy": overall_accuracy,
#         "average_jaccard_similarity": average_jaccard,
#         "total_predictions_evaluated": total_predictions,
#         "task_details": task_benchmarks
#     }
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     # Optionally save solver_stats as well for diagnostics/visualization
#     atomic_write_json(config.WORKING_DIR / "solver_stats.json", {s: {k: v if not isinstance(v, list) else v for k, v in stats.items()} for s, stats in solver_stats.items()}) # Fix set serialization to list here
# 
#     final_submission, report = finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy=config.VOTING_STRATEGY)
# 
#     # --- Integrate benchmarking here ---
#     if train_solutions: # Only benchmark if true solutions are available
#         benchmarking_results = benchmark_submission(final_submission, train_solutions)
#         report["benchmarking_results"] = benchmarking_results
#         logger.info("Benchmarking completed. Overall Accuracy: %.2f, Average Jaccard: %.2f", benchmarking_results["overall_accuracy"], benchmarking_results["average_jaccard_similarity"])
#     else:
#         logger.warning("No true solutions found for benchmarking.")
# 
# 
#     if not dry_run:
#         atomic_write_json(config.FINAL_SUB, final_submission)
#         report["summary"] = {"total_tasks": len(final_submission), "config_used": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}} # Fix path serialization
#         report["solver_metrics"] = {s: {k: v if not isinstance(v, set) else sorted(list(v)) for k, v in stats.items()} for s, stats in solver_stats.items()} # Add solver statistics to the report, ensure set to list
#         atomic_write_json(config.REPORT_PATH, report)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}}


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import pandas as pd
# import json
# from pathlib import Path
# 
# # 1. Load the execution_report.json file
# report_path = Path("/kaggle/working/execution_report.json")
# 
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report = json.load(f)
#     print("Execution report loaded successfully.")
# else:
#     print(f"Error: {report_path} not found.")
#     report = {}
# 
# # Check if report has necessary data
# if not report or "solver_metrics" not in report or "benchmarking_results" not in report:
#     print("Report does not contain enough data for visualizations. Please ensure run_all was executed.")
# else:
#     # 2. Visualize Solver Weights Distribution
#     solver_weights = report.get("solver_metrics", {})
#     if solver_weights:
#         weights = {s: d.get('weight') for s, d in solver_weights.items() if 'weight' in d}
#         if not weights: # Fallback if 'weight' isn't directly in solver_metrics (e.g. from solver_weights.json)
#             solver_weights_path = Path("/kaggle/working/solver_weights.json")
#             if solver_weights_path.exists():
#                 with open(solver_weights_path, 'r', encoding='utf-8') as f:
#                     weights = json.load(f)
# 
#         if weights:
#             solvers = list(weights.keys())
#             values = list(weights.values())
# 
#             plt.figure(figsize=(10, 6))
#             plt.bar(solvers, values, color='skyblue')
#             plt.xlabel('Solver')
#             plt.ylabel('Weight')
#             plt.title('Solver Weights Distribution')
#             plt.xticks(rotation=45, ha='right')
#             plt.tight_layout()
#             plt.show()
#         else:
#             print("No solver weights found for visualization.")
#     else:
#         print("No solver metrics found in the report.")
# 
#     # 3. Visualize Per-Solver Accuracy and Average Confidence
#     solver_metrics = report.get("solver_metrics", {})
#     if solver_metrics:
#         solver_names = []
#         accuracies = []
#         avg_confidences = []
# 
#         for s_name, metrics in solver_metrics.items():
#             solver_names.append(s_name)
#             accuracy = metrics.get('correct', 0) / metrics.get('attempts', 1) if metrics.get('attempts', 0) > 0 else 0
#             accuracies.append(accuracy)
#             avg_confidences.append(metrics.get('average_confidence', 0.0))
# 
#         # Plot Solver Accuracy
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, accuracies, color='lightcoral')
#         plt.xlabel('Solver')
#         plt.ylabel('Accuracy')
#         plt.title('Per-Solver Accuracy')
#         plt.xticks(rotation=45, ha='right')
#         plt.ylim(0, 1) # Accuracy is between 0 and 1
#         plt.tight_layout()
#         plt.show()
# 
#         # Plot Solver Average Confidence
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, avg_confidences, color='lightgreen')
#         plt.xlabel('Solver')
#         plt.ylabel('Average Confidence')
#         plt.title('Per-Solver Average Confidence')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No solver metrics found for accuracy/confidence visualization.")
# 
#     # 4. Visualize Distribution of Task Difficulties
#     benchmarking_results = report.get("benchmarking_results", {})
#     if benchmarking_results and "task_details" in benchmarking_results:
#         jaccard_scores = []
#         total_tasks = 0
#         perfectly_solved_tasks = 0
# 
#         for tid, details in benchmarking_results["task_details"].items():
#             total_tasks += 1
#             task_perfectly_solved = True
#             for test in details["tests"]:
#                 jaccard_scores.append(test["jaccard_similarity"])
#                 if test["jaccard_similarity"] < 1.0:
#                     task_perfectly_solved = False
#             if task_perfectly_solved:
#                 perfectly_solved_tasks += 1
# 
#         if jaccard_scores:
#             plt.figure(figsize=(10, 6))
#             plt.hist(jaccard_scores, bins=20, color='purple', edgecolor='black')
#             plt.xlabel('Jaccard Similarity')
#             plt.ylabel('Number of Tests')
#             plt.title('Distribution of Task Jaccard Similarities')
#             plt.grid(axis='y', alpha=0.75)
#             plt.tight_layout()
#             plt.show()
#         else:
#             print("No Jaccard scores available for histogram visualization.")
# 
#         # Solved vs Unsolved Tasks (based on all tests in a task being perfectly solved)
#         if total_tasks > 0:
#             unsolved_tasks = total_tasks - perfectly_solved_tasks
#             labels = ['Perfectly Solved Tasks', 'Partially/Unsolved Tasks']
#             sizes = [perfectly_solved_tasks, unsolved_tasks]
#             colors = ['#66b3ff', '#ff9999']
#             explode = (0.1, 0) # explode 1st slice
# 
#             plt.figure(figsize=(8, 8))
#             plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#             plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
#             plt.title('Proportion of Perfectly Solved vs. Partially/Unsolved Tasks')
#             plt.tight_layout()
#             plt.show()
# 
#             print(f"Total tasks evaluated: {total_tasks}")
#             print(f"Tasks perfectly solved (all tests passed with Jaccard 1.0): {perfectly_solved_tasks}")
#             print(f"Tasks partially/unsolved: {unsolved_tasks}")
#         else:
#             print("No tasks found in benchmarking results to visualize solved vs unsolved.")
#     else:
#         print("No benchmarking results found for task difficulty visualization.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import pandas as pd
# import json
# from pathlib import Path
# 
# # Ensure run_all is executed to generate/update execution_report.json
# # The run_all function is defined in a previous cell (00138927).
# # We need to ensure it runs before attempting to load its output report.
# print("Executing run_all to generate the latest execution report...")
# result = run_all(dry_run=False)
# print("run_all execution complete. Result:", result)
# 
# # 1. Load the execution_report.json file
# report_path = Path("/kaggle/working/execution_report.json")
# 
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report = json.load(f)
#     print("Execution report loaded successfully for visualization.")
# else:
#     print(f"Error: {report_path} not found after run_all execution.")
#     report = {}
# 
# # Check if report has necessary data
# if not report or "solver_metrics" not in report or "benchmarking_results" not in report:
#     print("Report still does not contain enough data for visualizations, even after re-running run_all.")
# else:
#     # 2. Visualize Solver Weights Distribution
#     solver_weights = report.get("solver_metrics", {})
#     # Extract weights from the top-level solver_weights in the report, if not present then check for 'weight' key in individual solver metrics
#     # The 'weight' directly comes from the solver_weights.json, not the metrics sub-dictionary in execution_report
#     solver_weights_from_file = {}
#     solver_weights_path = Path("/kaggle/working/solver_weights.json")
#     if solver_weights_path.exists():
#         with open(solver_weights_path, 'r', encoding='utf-8') as f:
#             solver_weights_from_file = json.load(f)
# 
#     if solver_weights_from_file:
#         solvers = list(solver_weights_from_file.keys())
#         values = list(solver_weights_from_file.values())
# 
#         plt.figure(figsize=(10, 6))
#         plt.bar(solvers, values, color='skyblue')
#         plt.xlabel('Solver')
#         plt.ylabel('Weight')
#         plt.title('Solver Weights Distribution')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No solver weights found for visualization (from solver_weights.json).")
# 
#     # 3. Visualize Per-Solver Accuracy and Average Confidence
#     solver_metrics = report.get("solver_metrics", {})
#     if solver_metrics:
#         solver_names = []
#         accuracies = []
#         avg_confidences = []
# 
#         for s_name, metrics in solver_metrics.items():
#             solver_names.append(s_name)
#             accuracy = metrics.get('correct', 0) / metrics.get('attempts', 1) if metrics.get('attempts', 0) > 0 else 0
#             accuracies.append(accuracy)
#             # Ensure average_confidence exists, if not, set to 0.0
#             avg_confidences.append(metrics.get('average_confidence', 0.0))
# 
#         # Plot Solver Accuracy
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, accuracies, color='lightcoral')
#         plt.xlabel('Solver')
#         plt.ylabel('Accuracy')
#         plt.title('Per-Solver Accuracy')
#         plt.xticks(rotation=45, ha='right')
#         plt.ylim(0, 1) # Accuracy is between 0 and 1
#         plt.tight_layout()
#         plt.show()
# 
#         # Plot Solver Average Confidence
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, avg_confidences, color='lightgreen')
#         plt.xlabel('Solver')
#         plt.ylabel('Average Confidence')
#         plt.title('Per-Solver Average Confidence')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No solver metrics found for accuracy/confidence visualization.")
# 
#     # 4. Visualize Distribution of Task Difficulties
#     benchmarking_results = report.get("benchmarking_results", {})
#     if benchmarking_results and "task_details" in benchmarking_results:
#         jaccard_scores = []
#         total_tasks = 0
#         perfectly_solved_tasks = 0
# 
#         for tid, details in benchmarking_results["task_details"].items():
#             total_tasks += 1
#             task_perfectly_solved = True
#             if "tests" in details:
#                 for test in details["tests"]:
#                     jaccard_scores.append(test.get("jaccard_similarity", 0.0))
#                     if test.get("jaccard_similarity", 0.0) < 1.0:
#                         task_perfectly_solved = False
#             else:
#                 # If no tests are detailed, assume not perfectly solved or handle as a special case
#                 task_perfectly_solved = False
#                 logger.warning(f"Task {tid} in benchmarking results has no 'tests' detail.")
# 
#             if task_perfectly_solved:
#                 perfectly_solved_tasks += 1
# 
#         if jaccard_scores:
#             plt.figure(figsize=(10, 6))
#             plt.hist(jaccard_scores, bins=20, color='purple', edgecolor='black')
#             plt.xlabel('Jaccard Similarity')
#             plt.ylabel('Number of Tests')
#             plt.title('Distribution of Task Jaccard Similarities')
#             plt.grid(axis='y', alpha=0.75)
#             plt.tight_layout()
#             plt.show()
#         else:
#             print("No Jaccard scores available for histogram visualization.")
# 
#         # Solved vs Unsolved Tasks (based on all tests in a task being perfectly solved)
#         if total_tasks > 0:
#             unsolved_tasks = total_tasks - perfectly_solved_tasks
#             labels = ['Perfectly Solved Tasks', 'Partially/Unsolved Tasks']
#             sizes = [perfectly_solved_tasks, unsolved_tasks]
#             colors = ['#66b3ff', '#ff9999']
#             explode = (0.1, 0) # explode 1st slice
# 
#             plt.figure(figsize=(8, 8))
#             plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#             plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
#             plt.title('Proportion of Perfectly Solved vs. Partially/Unsolved Tasks')
#             plt.tight_layout()
#             plt.show()
# 
#             print(f"Total tasks evaluated: {total_tasks}")
#             print(f"Tasks perfectly solved (all tests passed with Jaccard 1.0): {perfectly_solved_tasks}")
#             print(f"Tasks partially/unsolved: {unsolved_tasks}")
#         else:
#             print("No tasks found in benchmarking results to visualize solved vs unsolved.")
#     else:
#         print("No benchmarking results found for task difficulty visualization.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # --- Config Class (Updated) ---
# class Config:
#     def __init__(self, working_dir=None, input_dir=None, smoothing_alpha=None, voting_strategy=None):
#         self.WORKING_DIR = Path(working_dir) if working_dir else Path("/kaggle/working")
#         self.INPUT_DIR = Path(input_dir) if input_dir else Path("/kaggle/input")
# 
#         self.REPORT_PATH = self.WORKING_DIR / "execution_report.json"
#         self.FINAL_SUB = self.WORKING_DIR / "submission.json"
#         self.SOLVER_WEIGHTS_PATH = self.WORKING_DIR / "solver_weights.json"
#         self.LOG_PATH = self.WORKING_DIR / "finalize.log"
#         self.SMOOTHING_ALPHA = smoothing_alpha if smoothing_alpha is not None else 1.0
#         self.VOTING_STRATEGY = voting_strategy if voting_strategy else "weighted" # New: Default voting strategy
# 
#         self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def load_from_json(self, config_file_path):
#         if not Path(config_file_path).is_file():
#             return
#         with open(config_file_path, 'r', encoding='utf-8') as f:
#             overrides = json.load(f)
#         for key, value in overrides.items():
#             if hasattr(self, key):
#                 # Handle Path objects correctly
#                 if 'DIR' in key.upper() or 'PATH' in key.upper() or 'SUB' in key.upper():
#                     setattr(self, key, Path(value))
#                 else:
#                     setattr(self, key, value)
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0, "total_confidence_sum":0.0, "task_ids_attempted":set()})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 solver_stats[solver]["task_ids_attempted"].add(tid)
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#                     conf = c.get("confidence")
#                     if conf is not None and isinstance(conf,(int,float)):
#                         solver_stats[solver]["total_confidence_sum"] += float(conf)
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
# 
#             # Calculate additional metrics
#             st["average_confidence"] = st["total_confidence_sum"] / st["correct"] if st["correct"] > 0 else 0.0
#             st["num_tasks_attempted"] = len(st["task_ids_attempted"])
#             # Convert set to list for JSON serialization if needed later
#             st["task_ids_attempted"] = sorted(list(st["task_ids_attempted"])) # Convert set to list here
# 
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#                         solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                         solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#             for s in sols:
#                 solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                 solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#     return solver_weights, solver_stats
# 
# # --- find_and_load_challenge (Modified) ---
# def find_and_load_challenge(config, candidates):
#     all_challenges = {}
#     found_any_challenge = False
# 
#     # Prioritize loading challenge tasks from the standard ARC test path
#     arc_test_path = config.INPUT_DIR / "arc-prize-2025" / "test"
#     if arc_test_path.exists() and arc_test_path.is_dir():
#         logger.info(f"Searching for challenge files in {arc_test_path}")
#         for f in arc_test_path.rglob("*.json"):
#             data, _ = load_json_if_file(f)
#             if data:
#                 if isinstance(data, dict):
#                     all_challenges.update(data)
#                     found_any_challenge = True
#                 else:
#                     logger.warning(f"Skipping non-dict JSON file in ARC test path: {f}")
# 
#     if found_any_challenge:
#         # Convert combined dictionary to a list of tasks if it makes sense for further processing
#         # Assuming challenge data is a dict of {task_id: task_details}
#         return all_challenges, str(arc_test_path)
# 
#     # Fallback to original candidates if no specific ARC test challenges found
#     logger.warning("No structured ARC test challenges found in expected path. Falling back to generic search.")
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try:
#                 data, used = load_json_if_file(p)
#                 if isinstance(data, dict): return data, p # Return the first valid dict file
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try:
#                         data, used = load_json_if_file(f)
#                         if isinstance(data, dict): return data, f # Return the first valid dict file
#                     except Exception: continue
#             for f in p.rglob("*.json"): # This iterates over ALL json files recursively.
#                 try:
#                     data, used = load_json_if_file(f)
#                     if isinstance(data, dict): return data, f # Return the first valid dict file
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try:
#             data, used = load_json_if_file(f)
#             if isinstance(data, dict): return data, f # Return the first valid dict file
#         except Exception: continue
#     return None, None
# 
# # --- finalize_submission (Updated) ---
# def finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy="weighted"):
#     # load challenge if not provided
#     if challenge is None:
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         challenge_path_used = "provided"
# 
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
# 
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
# 
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
# 
#             scores = defaultdict(float)
#             contributors = defaultdict(list)
# 
#             if voting_strategy == "weighted":
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             elif voting_strategy == "unweighted":
#                 # Simple majority vote
#                 for c in cands:
#                     g = c.get("grid")
#                     if g is None: continue
#                     k = grid_key(g)
#                     scores[k] += 1 # Each occurrence counts as one vote
#                     contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#             elif voting_strategy == "confidence-based":
#                 # Use confidence as the primary weight. If no confidence, treat as 0 or 1.
#                 any_confidence = False
#                 for c in cands:
#                     if c.get("confidence") is not None and isinstance(c.get("confidence"), (int, float)):
#                         any_confidence = True
#                         break
# 
#                 if not any_confidence:
#                     logger.warning(f"Task {tid}, test {ti}: No confidence scores found for confidence-based voting. Falling back to unweighted voting.")
#                     # Fallback to unweighted voting if no confidence is available
#                     for c in cands:
#                         g = c.get("grid")
#                         if g is None: continue
#                         k = grid_key(g)
#                         scores[k] += 1
#                         contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#                 else:
#                     for c in cands:
#                         sname = c.get("solver") or "unknown"
#                         g = c.get("grid")
#                         if g is None: continue
#                         conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                         score = conf_factor # Only confidence matters
#                         k = grid_key(g)
#                         scores[k] += score; contributors[k].append({"solver":sname,"weight":1,"confidence":conf})
#             else:
#                 logger.warning(f"Unknown voting strategy '{voting_strategy}'. Falling back to weighted voting.")
#                 # Default to weighted if strategy is unknown
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
# 
#             if not scores:
#                 chosen_grids.append([[0]]); continue # Fallback if no valid grids from candidates or voting method yields no scores
# 
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# # --- New Helper Function for Benchmarking ---
# def grid_to_set(grid):
#     if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
#         return set() # Return empty set for invalid or empty grids
#     s = set()
#     for r_idx, row in enumerate(grid):
#         for c_idx, color in enumerate(row):
#             s.add((r_idx, c_idx, color))
#     return s
# 
# def grid_jaccard_similarity(grid1, grid2):
#     set1 = grid_to_set(grid1)
#     set2 = grid_to_set(grid2)
#     if not set1 and not set2: # Both grids are empty or invalid, consider them 100% similar in a trivial way
#         return 1.0
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     if union == 0:
#         return 0.0 # Should not happen if at least one set is non-empty, but for safety
#     return intersection / union
# 
# # --- New Benchmark Function ---
# def benchmark_submission(final_submission, true_solutions):
#     total_correct_predictions = 0
#     total_predictions = 0
#     total_jaccard_scores = 0.0
#     task_benchmarks = {}
# 
#     for tid, predicted_output in final_submission.items():
#         true_grids = get_solution_grids_for_task(tid, true_solutions)
#         if not true_grids:
#             logger.debug(f"No true solutions found for task {tid}, skipping benchmarking.")
#             continue
# 
#         # Predicted output is typically a list containing a single dict of attempts
#         if isinstance(predicted_output, list) and predicted_output:
#             predicted_attempts = predicted_output[0]
#         else:
#             logger.warning(f"Unexpected format for predicted output for task {tid}: {predicted_output}")
#             continue
# 
#         num_tests = len(true_grids)
#         task_benchmarks[tid] = {"tests": []}
# 
#         for i in range(num_tests):
#             # Get predicted grid for the current test
#             pred_grid = predicted_attempts.get(f"attempt_{i+1}", [[0]]) # Default to empty grid
#             true_grid = true_grids[i]
# 
#             correct = 0
#             jaccard_score = 0.0
# 
#             if grid_key(pred_grid) == grid_key(true_grid):
#                 correct = 1
#                 jaccard_score = 1.0 # Jaccard is 1.0 if grids are identical
#             else:
#                 jaccard_score = grid_jaccard_similarity(pred_grid, true_grid)
# 
#             total_correct_predictions += correct
#             total_jaccard_scores += jaccard_score
#             total_predictions += 1
# 
#             task_benchmarks[tid]["tests"].append({
#                 "test_idx": i,
#                 "accuracy": correct,
#                 "jaccard_similarity": jaccard_score
#             })
# 
#     overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0.0
#     average_jaccard = total_jaccard_scores / total_predictions if total_predictions > 0 else 0.0
# 
#     return {
#         "overall_accuracy": overall_accuracy,
#         "average_jaccard_similarity": average_jaccard,
#         "total_predictions_evaluated": total_predictions,
#         "task_details": task_benchmarks
#     }
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     # Optionally save solver_stats as well for diagnostics/visualization
#     atomic_write_json(config.WORKING_DIR / "solver_stats.json", {s: {k: v if not isinstance(v, list) else v for k, v in stats.items()} for s, stats in solver_stats.items()}) # Fix set serialization to list here
# 
#     final_submission, report = finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy=config.VOTING_STRATEGY)
# 
#     # --- Integrate benchmarking here ---
#     if train_solutions: # Only benchmark if true solutions are available
#         benchmarking_results = benchmark_submission(final_submission, train_solutions)
#         report["benchmarking_results"] = benchmarking_results
#         logger.info("Benchmarking completed. Overall Accuracy: %.2f, Average Jaccard: %.2f", benchmarking_results["overall_accuracy"], benchmarking_results["average_jaccard_similarity"])
#     else:
#         logger.warning("No true solutions found for benchmarking.")
# 
# 
#     if not dry_run:
#         atomic_write_json(config.FINAL_SUB, final_submission)
#         report["summary"] = {"total_tasks": len(final_submission), "config_used": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}} # Fix path serialization
#         report["solver_metrics"] = {s: {k: v if not isinstance(v, set) else sorted(list(v)) for k, v in stats.items()} for s, stats in solver_stats.items()} # Add solver statistics to the report, ensure set to list
#         atomic_write_json(config.REPORT_PATH, report)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}}
# 
# print("find_and_load_challenge function updated to prioritize ARC test challenges.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import pandas as pd
# import json
# from pathlib import Path
# 
# # Ensure run_all is executed to generate/update execution_report.json
# # The run_all function is defined in a previous cell (00138927).
# # We need to ensure it runs before attempting to load its output report.
# print("Executing run_all to generate the latest execution report...")
# result = run_all(dry_run=False)
# print("run_all execution complete. Result:", result)
# 
# # 1. Load the execution_report.json file
# report_path = Path("/kaggle/working/execution_report.json")
# 
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report = json.load(f)
#     print("Execution report loaded successfully for visualization.")
# else:
#     print(f"Error: {report_path} not found after run_all execution.")
#     report = {}
# 
# # Check if report has necessary data
# if not report or "solver_metrics" not in report or "benchmarking_results" not in report:
#     print("Report still does not contain enough data for visualizations, even after re-running run_all.")
# else:
#     # 2. Visualize Solver Weights Distribution
#     solver_weights = report.get("solver_metrics", {})
#     # Extract weights from the top-level solver_weights in the report, if not present then check for 'weight' key in individual solver metrics
#     # The 'weight' directly comes from the solver_weights.json, not the metrics sub-dictionary in execution_report
#     solver_weights_from_file = {}
#     solver_weights_path = Path("/kaggle/working/solver_weights.json")
#     if solver_weights_path.exists():
#         with open(solver_weights_path, 'r', encoding='utf-8') as f:
#             solver_weights_from_file = json.load(f)
# 
#     if solver_weights_from_file:
#         solvers = list(solver_weights_from_file.keys())
#         values = list(solver_weights_from_file.values())
# 
#         plt.figure(figsize=(10, 6))
#         plt.bar(solvers, values, color='skyblue')
#         plt.xlabel('Solver')
#         plt.ylabel('Weight')
#         plt.title('Solver Weights Distribution')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No solver weights found for visualization (from solver_weights.json).")
# 
#     # 3. Visualize Per-Solver Accuracy and Average Confidence
#     solver_metrics = report.get("solver_metrics", {})
#     if solver_metrics:
#         solver_names = []
#         accuracies = []
#         avg_confidences = []
# 
#         for s_name, metrics in solver_metrics.items():
#             solver_names.append(s_name)
#             accuracy = metrics.get('correct', 0) / metrics.get('attempts', 1) if metrics.get('attempts', 0) > 0 else 0
#             accuracies.append(accuracy)
#             # Ensure average_confidence exists, if not, set to 0.0
#             avg_confidences.append(metrics.get('average_confidence', 0.0))
# 
#         # Plot Solver Accuracy
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, accuracies, color='lightcoral')
#         plt.xlabel('Solver')
#         plt.ylabel('Accuracy')
#         plt.title('Per-Solver Accuracy')
#         plt.xticks(rotation=45, ha='right')
#         plt.ylim(0, 1) # Accuracy is between 0 and 1
#         plt.tight_layout()
#         plt.show()
# 
#         # Plot Solver Average Confidence
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, avg_confidences, color='lightgreen')
#         plt.xlabel('Solver')
#         plt.ylabel('Average Confidence')
#         plt.title('Per-Solver Average Confidence')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No solver metrics found for accuracy/confidence visualization.")
# 
#     # 4. Visualize Distribution of Task Difficulties
#     benchmarking_results = report.get("benchmarking_results", {})
#     if benchmarking_results and "task_details" in benchmarking_results:
#         jaccard_scores = []
#         total_tasks = 0
#         perfectly_solved_tasks = 0
# 
#         for tid, details in benchmarking_results["task_details"].items():
#             total_tasks += 1
#             task_perfectly_solved = True
#             if "tests" in details:
#                 for test in details["tests"]:
#                     jaccard_scores.append(test.get("jaccard_similarity", 0.0))
#                     if test.get("jaccard_similarity", 0.0) < 1.0:
#                         task_perfectly_solved = False
#             else:
#                 # If no tests are detailed, assume not perfectly solved or handle as a special case
#                 task_perfectly_solved = False
#                 logger.warning(f"Task {tid} in benchmarking results has no 'tests' detail.")
# 
#             if task_perfectly_solved:
#                 perfectly_solved_tasks += 1
# 
#         if jaccard_scores:
#             plt.figure(figsize=(10, 6))
#             plt.hist(jaccard_scores, bins=20, color='purple', edgecolor='black')
#             plt.xlabel('Jaccard Similarity')
#             plt.ylabel('Number of Tests')
#             plt.title('Distribution of Task Jaccard Similarities')
#             plt.grid(axis='y', alpha=0.75)
#             plt.tight_layout()
#             plt.show()
#         else:
#             print("No Jaccard scores available for histogram visualization.")
# 
#         # Solved vs Unsolved Tasks (based on all tests in a task being perfectly solved)
#         if total_tasks > 0:
#             unsolved_tasks = total_tasks - perfectly_solved_tasks
#             labels = ['Perfectly Solved Tasks', 'Partially/Unsolved Tasks']
#             sizes = [perfectly_solved_tasks, unsolved_tasks]
#             colors = ['#66b3ff', '#ff9999']
#             explode = (0.1, 0) # explode 1st slice
# 
#             plt.figure(figsize=(8, 8))
#             plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#             plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
#             plt.title('Proportion of Perfectly Solved vs. Partially/Unsolved Tasks')
#             plt.tight_layout()
#             plt.show()
# 
#             print(f"Total tasks evaluated: {total_tasks}")
#             print(f"Tasks perfectly solved (all tests passed with Jaccard 1.0): {perfectly_solved_tasks}")
#             print(f"Tasks partially/unsolved: {unsolved_tasks}")
#         else:
#             print("No tasks found in benchmarking results to visualize solved vs unsolved.")
#     else:
#         print("No benchmarking results found for task difficulty visualization.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # --- Config Class (Updated) ---
# class Config:
#     def __init__(self, working_dir=None, input_dir=None, smoothing_alpha=None, voting_strategy=None):
#         self.WORKING_DIR = Path(working_dir) if working_dir else Path("/kaggle/working")
#         self.INPUT_DIR = Path(input_dir) if input_dir else Path("/kaggle/input")
# 
#         self.REPORT_PATH = self.WORKING_DIR / "execution_report.json"
#         self.FINAL_SUB = self.WORKING_DIR / "submission.json"
#         self.SOLVER_WEIGHTS_PATH = self.WORKING_DIR / "solver_weights.json"
#         self.LOG_PATH = self.WORKING_DIR / "finalize.log"
#         self.SMOOTHING_ALPHA = smoothing_alpha if smoothing_alpha is not None else 1.0
#         self.VOTING_STRATEGY = voting_strategy if voting_strategy else "weighted" # New: Default voting strategy
# 
#         self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def load_from_json(self, config_file_path):
#         if not Path(config_file_path).is_file():
#             return
#         with open(config_file_path, 'r', encoding='utf-8') as f:
#             overrides = json.load(f)
#         for key, value in overrides.items():
#             if hasattr(self, key):
#                 # Handle Path objects correctly
#                 if 'DIR' in key.upper() or 'PATH' in key.upper() or 'SUB' in key.upper():
#                     setattr(self, key, Path(value))
#                 else:
#                     setattr(self, key, value)
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0, "total_confidence_sum":0.0, "task_ids_attempted":set()})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 solver_stats[solver]["task_ids_attempted"].add(tid)
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#                     conf = c.get("confidence")
#                     if conf is not None and isinstance(conf,(int,float)):
#                         solver_stats[solver]["total_confidence_sum"] += float(conf)
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
# 
#             # Calculate additional metrics
#             st["average_confidence"] = st["total_confidence_sum"] / st["correct"] if st["correct"] > 0 else 0.0
#             st["num_tasks_attempted"] = len(st["task_ids_attempted"])
#             # Convert set to list for JSON serialization if needed later
#             st["task_ids_attempted"] = sorted(list(st["task_ids_attempted"])) # Convert set to list here
# 
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#                         solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                         solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#             for s in sols:
#                 solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                 solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#     return solver_weights, solver_stats
# 
# def find_and_load_challenge(config, candidates):
#     all_challenges = {}
#     found_any_challenge = False
# 
#     # Prioritize loading challenge tasks from the standard ARC test path
#     arc_test_path = config.INPUT_DIR / "arc-prize-2025" / "test"
#     if arc_test_path.exists() and arc_test_path.is_dir():
#         logger.info(f"Searching for challenge files in {arc_test_path}")
#         for f in arc_test_path.rglob("*.json"):
#             data, _ = load_json_if_file(f)
#             if data:
#                 if isinstance(data, dict):
#                     all_challenges.update(data)
#                     found_any_challenge = True
#                 else:
#                     logger.warning(f"Skipping non-dict JSON file in ARC test path: {f}")
# 
#     if found_any_challenge:
#         # Convert combined dictionary to a list of tasks if it makes sense for further processing
#         # Assuming challenge data is a dict of {task_id: task_details}
#         return all_challenges, str(arc_test_path)
# 
#     # Fallback to original candidates if no specific ARC test challenges found
#     logger.warning("No structured ARC test challenges found in expected path. Falling back to generic search.")
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try:
#                 data, used = load_json_if_file(p)
#                 if isinstance(data, dict): return data, p # Return the first valid dict file
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try:
#                         data, used = load_json_if_file(f)
#                         if isinstance(data, dict): return data, f # Return the first valid dict file
#                     except Exception: continue
#             for f in p.rglob("*.json"): # This iterates over ALL json files recursively.
#                 try:
#                     data, used = load_json_if_file(f)
#                     if isinstance(data, dict): return data, f # Return the first valid dict file
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try:
#             data, used = load_json_if_file(f)
#             if isinstance(data, dict): return data, f # Return the first valid dict file
#         except Exception: continue
#     return None, None
# 
# # --- finalize_submission (Updated) ---
# def finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy="weighted"):
#     # load challenge if not provided
#     if challenge is None:
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         challenge_path_used = "provided"
# 
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
# 
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
# 
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
# 
#             scores = defaultdict(float)
#             contributors = defaultdict(list)
# 
#             if voting_strategy == "weighted":
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             elif voting_strategy == "unweighted":
#                 # Simple majority vote
#                 for c in cands:
#                     g = c.get("grid")
#                     if g is None: continue
#                     k = grid_key(g)
#                     scores[k] += 1 # Each occurrence counts as one vote
#                     contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#             elif voting_strategy == "confidence-based":
#                 # Use confidence as the primary weight. If no confidence, treat as 0 or 1.
#                 any_confidence = False
#                 for c in cands:
#                     if c.get("confidence") is not None and isinstance(c.get("confidence"), (int, float)):
#                         any_confidence = True
#                         break
# 
#                 if not any_confidence:
#                     logger.warning(f"Task {tid}, test {ti}: No confidence scores found for confidence-based voting. Falling back to unweighted voting.")
#                     # Fallback to unweighted voting if no confidence is available
#                     for c in cands:
#                         g = c.get("grid")
#                         if g is None: continue
#                         k = grid_key(g)
#                         scores[k] += 1
#                         contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#                 else:
#                     for c in cands:
#                         sname = c.get("solver") or "unknown"
#                         g = c.get("grid")
#                         if g is None: continue
#                         conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                         score = conf_factor # Only confidence matters
#                         k = grid_key(g)
#                         scores[k] += score; contributors[k].append({"solver":sname,"weight":1,"confidence":conf})
#             else:
#                 logger.warning(f"Unknown voting strategy '{voting_strategy}'. Falling back to weighted voting.")
#                 # Default to weighted if strategy is unknown
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
# 
#             if not scores:
#                 chosen_grids.append([[0]]); continue # Fallback if no valid grids from candidates or voting method yields no scores
# 
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# # --- New Helper Function for Benchmarking ---
# def grid_to_set(grid):
#     if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
#         return set() # Return empty set for invalid or empty grids
#     s = set()
#     for r_idx, row in enumerate(grid):
#         for c_idx, color in enumerate(row):
#             s.add((r_idx, c_idx, color))
#     return s
# 
# def grid_jaccard_similarity(grid1, grid2):
#     set1 = grid_to_set(grid1)
#     set2 = grid_to_set(grid2)
#     if not set1 and not set2: # Both grids are empty or invalid, consider them 100% similar in a trivial way
#         return 1.0
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     if union == 0:
#         return 0.0 # Should not happen if at least one set is non-empty, but for safety
#     return intersection / union
# 
# # --- New Benchmark Function ---
# def benchmark_submission(final_submission, true_solutions):
#     total_correct_predictions = 0
#     total_predictions = 0
#     total_jaccard_scores = 0.0
#     task_benchmarks = {}
# 
#     for tid, predicted_output in final_submission.items():
#         true_grids = get_solution_grids_for_task(tid, true_solutions)
#         if not true_grids:
#             logger.debug(f"No true solutions found for task {tid}, skipping benchmarking.")
#             continue
# 
#         # Predicted output is typically a list containing a single dict of attempts
#         if isinstance(predicted_output, list) and predicted_output:
#             predicted_attempts = predicted_output[0]
#         else:
#             logger.warning(f"Unexpected format for predicted output for task {tid}: {predicted_output}")
#             continue
# 
#         num_tests = len(true_grids)
#         task_benchmarks[tid] = {"tests": []}
# 
#         for i in range(num_tests):
#             # Get predicted grid for the current test
#             pred_grid = predicted_attempts.get(f"attempt_{i+1}", [[0]]) # Default to empty grid
#             true_grid = true_grids[i]
# 
#             correct = 0
#             jaccard_score = 0.0
# 
#             if grid_key(pred_grid) == grid_key(true_grid):
#                 correct = 1
#                 jaccard_score = 1.0 # Jaccard is 1.0 if grids are identical
#             else:
#                 jaccard_score = grid_jaccard_similarity(pred_grid, true_grid)
# 
#             total_correct_predictions += correct
#             total_jaccard_scores += jaccard_score
#             total_predictions += 1
# 
#             task_benchmarks[tid]["tests"].append({
#                 "test_idx": i,
#                 "accuracy": correct,
#                 "jaccard_similarity": jaccard_score
#             })
# 
#     overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0.0
#     average_jaccard = total_jaccard_scores / total_predictions if total_predictions > 0 else 0.0
# 
#     return {
#         "overall_accuracy": overall_accuracy,
#         "average_jaccard_similarity": average_jaccard,
#         "total_predictions_evaluated": total_predictions,
#         "task_details": task_benchmarks
#     }
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     # Optionally save solver_stats as well for diagnostics/visualization
#     atomic_write_json(config.WORKING_DIR / "solver_stats.json", {s: {k: v if not isinstance(v, list) else v for k, v in stats.items()} for s, stats in solver_stats.items()}) # Fix set serialization to list here
# 
#     # Pass train_solutions as the challenge to finalize_submission for benchmarking
#     final_submission, report = finalize_submission(partial, solver_weights, config, challenge=train_solutions, voting_strategy=config.VOTING_STRATEGY)
# 
#     # --- Integrate benchmarking here ---
#     if train_solutions: # Only benchmark if true solutions are available
#         benchmarking_results = benchmark_submission(final_submission, train_solutions)
#         report["benchmarking_results"] = benchmarking_results
#         logger.info("Benchmarking completed. Overall Accuracy: %.2f, Average Jaccard: %.2f", benchmarking_results["overall_accuracy"], benchmarking_results["average_jaccard_similarity"])
#     else:
#         logger.warning("No true solutions found for benchmarking.")
# 
# 
#     if not dry_run:
#         atomic_write_json(config.FINAL_SUB, final_submission)
#         report["summary"] = {"total_tasks": len(final_submission), "config_used": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}} # Fix path serialization
#         report["solver_metrics"] = {s: {k: v if not isinstance(v, set) else sorted(list(v)) for k, v in stats.items()} for s, stats in solver_stats.items()} # Add solver statistics to the report, ensure set to list
#         atomic_write_json(config.REPORT_PATH, report)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}}
# 
# print("run_all function updated to use train_solutions for finalizing submission and benchmarking.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import pandas as pd
# import json
# from pathlib import Path
# 
# # Ensure run_all is executed to generate/update execution_report.json
# # The run_all function is defined in a previous cell (f7070890).
# # We need to ensure it runs before attempting to load its output report.
# print("Executing run_all to generate the latest execution report...")
# result = run_all(dry_run=False)
# print("run_all execution complete. Result:", result)
# 
# # 1. Load the execution_report.json file
# report_path = Path("/kaggle/working/execution_report.json")
# 
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report = json.load(f)
#     print("Execution report loaded successfully for visualization.")
# else:
#     print(f"Error: {report_path} not found after run_all execution.")
#     report = {}
# 
# # Check if report has necessary data
# if not report or "solver_metrics" not in report or "benchmarking_results" not in report:
#     print("Report still does not contain enough data for visualizations, even after re-running run_all.")
# else:
#     # 2. Visualize Solver Weights Distribution
#     solver_weights = report.get("solver_metrics", {})
#     # Extract weights from the top-level solver_weights in the report, if not present then check for 'weight' key in individual solver metrics
#     # The 'weight' directly comes from the solver_weights.json, not the metrics sub-dictionary in execution_report
#     solver_weights_from_file = {}
#     solver_weights_path = Path("/kaggle/working/solver_weights.json")
#     if solver_weights_path.exists():
#         with open(solver_weights_path, 'r', encoding='utf-8') as f:
#             solver_weights_from_file = json.load(f)
# 
#     if solver_weights_from_file:
#         solvers = list(solver_weights_from_file.keys())
#         values = list(solver_weights_from_file.values())
# 
#         plt.figure(figsize=(10, 6))
#         plt.bar(solvers, values, color='skyblue')
#         plt.xlabel('Solver')
#         plt.ylabel('Weight')
#         plt.title('Solver Weights Distribution')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No solver weights found for visualization (from solver_weights.json).")
# 
#     # 3. Visualize Per-Solver Accuracy and Average Confidence
#     solver_metrics = report.get("solver_metrics", {})
#     if solver_metrics:
#         solver_names = []
#         accuracies = []
#         avg_confidences = []
# 
#         for s_name, metrics in solver_metrics.items():
#             solver_names.append(s_name)
#             accuracy = metrics.get('correct', 0) / metrics.get('attempts', 1) if metrics.get('attempts', 0) > 0 else 0
#             accuracies.append(accuracy)
#             # Ensure average_confidence exists, if not, set to 0.0
#             avg_confidences.append(metrics.get('average_confidence', 0.0))
# 
#         # Plot Solver Accuracy
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, accuracies, color='lightcoral')
#         plt.xlabel('Solver')
#         plt.ylabel('Accuracy')
#         plt.title('Per-Solver Accuracy')
#         plt.xticks(rotation=45, ha='right')
#         plt.ylim(0, 1) # Accuracy is between 0 and 1
#         plt.tight_layout()
#         plt.show()
# 
#         # Plot Solver Average Confidence
#         plt.figure(figsize=(12, 6))
#         plt.bar(solver_names, avg_confidences, color='lightgreen')
#         plt.xlabel('Solver')
#         plt.ylabel('Average Confidence')
#         plt.title('Per-Solver Average Confidence')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No solver metrics found for accuracy/confidence visualization.")
# 
#     # 4. Visualize Distribution of Task Difficulties
#     benchmarking_results = report.get("benchmarking_results", {})
#     if benchmarking_results and "task_details" in benchmarking_results:
#         jaccard_scores = []
#         total_tasks = 0
#         perfectly_solved_tasks = 0
# 
#         for tid, details in benchmarking_results["task_details"].items():
#             total_tasks += 1
#             task_perfectly_solved = True
#             if "tests" in details:
#                 for test in details["tests"]:
#                     jaccard_scores.append(test.get("jaccard_similarity", 0.0))
#                     if test.get("jaccard_similarity", 0.0) < 1.0:
#                         task_perfectly_solved = False
#             else:
#                 # If no tests are detailed, assume not perfectly solved or handle as a special case
#                 task_perfectly_solved = False
#                 logger.warning(f"Task {tid} in benchmarking results has no 'tests' detail.")
# 
#             if task_perfectly_solved:
#                 perfectly_solved_tasks += 1
# 
#         if jaccard_scores:
#             plt.figure(figsize=(10, 6))
#             plt.hist(jaccard_scores, bins=20, color='purple', edgecolor='black')
#             plt.xlabel('Jaccard Similarity')
#             plt.ylabel('Number of Tests')
#             plt.title('Distribution of Task Jaccard Similarities')
#             plt.grid(axis='y', alpha=0.75)
#             plt.tight_layout()
#             plt.show()
#         else:
#             print("No Jaccard scores available for histogram visualization.")
# 
#         # Solved vs Unsolved Tasks (based on all tests in a task being perfectly solved)
#         if total_tasks > 0:
#             unsolved_tasks = total_tasks - perfectly_solved_tasks
#             labels = ['Perfectly Solved Tasks', 'Partially/Unsolved Tasks']
#             sizes = [perfectly_solved_tasks, unsolved_tasks]
#             colors = ['#66b3ff', '#ff9999']
#             explode = (0.1, 0) # explode 1st slice
# 
#             plt.figure(figsize=(8, 8))
#             plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#             plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
#             plt.title('Proportion of Perfectly Solved vs. Partially/Unsolved Tasks')
#             plt.tight_layout()
#             plt.show()
# 
#             print(f"Total tasks evaluated: {total_tasks}")
#             print(f"Tasks perfectly solved (all tests passed with Jaccard 1.0): {perfectly_solved_tasks}")
#             print(f"Tasks partially/unsolved: {unsolved_tasks}")
#         else:
#             print("No tasks found in benchmarking results to visualize solved vs unsolved.")
#     else:
#         print("No benchmarking results found for task difficulty visualization.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # --- Config Class (Updated) ---
# class Config:
#     def __init__(self, working_dir=None, input_dir=None, smoothing_alpha=None, voting_strategy=None, submission_challenge_mode=None, custom_submission_challenge_path=None):
#         self.WORKING_DIR = Path(working_dir) if working_dir else Path("/kaggle/working")
#         self.INPUT_DIR = Path(input_dir) if input_dir else Path("/kaggle/input")
# 
#         self.REPORT_PATH = self.WORKING_DIR / "execution_report.json"
#         self.FINAL_SUB = self.WORKING_DIR / "submission.json"
#         self.SOLVER_WEIGHTS_PATH = self.WORKING_DIR / "solver_weights.json"
#         self.LOG_PATH = self.WORKING_DIR / "finalize.log"
#         self.SMOOTHING_ALPHA = smoothing_alpha if smoothing_alpha is not None else 1.0
#         self.VOTING_STRATEGY = voting_strategy if voting_strategy else "weighted" # New: Default voting strategy
#         # New attributes for challenge set selection
#         self.SUBMISSION_CHALLENGE_MODE = submission_challenge_mode if submission_challenge_mode else "competition_test_set" # Options: "competition_test_set", "train_set_for_submission", "custom_path"
#         self.CUSTOM_SUBMISSION_CHALLENGE_PATH = Path(custom_submission_challenge_path) if custom_submission_challenge_path else None
# 
#         self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def load_from_json(self, config_file_path):
#         if not Path(config_file_path).is_file():
#             return
#         with open(config_file_path, 'r', encoding='utf-8') as f:
#             overrides = json.load(f)
#         for key, value in overrides.items():
#             if hasattr(self, key):
#                 # Handle Path objects correctly
#                 if 'DIR' in key.upper() or 'PATH' in key.upper() or 'SUB' in key.upper():
#                     setattr(self, key, Path(value))
#                 else:
#                     setattr(self, key, value)
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0, "total_confidence_sum":0.0, "task_ids_attempted":set()})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 solver_stats[solver]["task_ids_attempted"].add(tid)
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#                     conf = c.get("confidence")
#                     if conf is not None and isinstance(conf,(int,float)):
#                         solver_stats[solver]["total_confidence_sum"] += float(conf)
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
# 
#             # Calculate additional metrics
#             st["average_confidence"] = st["total_confidence_sum"] / st["correct"] if st["correct"] > 0 else 0.0
#             st["num_tasks_attempted"] = len(st["task_ids_attempted"])
#             # Convert set to list for JSON serialization if needed later
#             st["task_ids_attempted"] = sorted(list(st["task_ids_attempted"])) # Convert set to list here
# 
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#                         solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                         solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#             for s in sols:
#                 solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                 solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#     return solver_weights, solver_stats
# 
# # --- find_and_load_challenge (Modified) ---
# def find_and_load_challenge(config, candidates, explicit_path=None):
#     all_challenges = {}
#     found_any_challenge = False
# 
#     if explicit_path and Path(explicit_path).exists():
#         logger.info(f"Prioritizing challenge files from explicit path: {explicit_path}")
#         p = Path(explicit_path)
#         if p.is_file():
#             data, _ = load_json_if_file(p)
#             if data and isinstance(data, dict):
#                 all_challenges.update(data)
#                 found_any_challenge = True
#         elif p.is_dir():
#             for f in p.rglob("*.json"):
#                 data, _ = load_json_if_file(f)
#                 if data and isinstance(data, dict):
#                     all_challenges.update(data)
#                     found_any_challenge = True
# 
#     if found_any_challenge:
#         return all_challenges, str(explicit_path)
# 
#     # Prioritize loading challenge tasks from the standard ARC test path if no explicit path or if explicit path yielded nothing
#     arc_test_path = config.INPUT_DIR / "arc-prize-2025" / "test"
#     if arc_test_path.exists() and arc_test_path.is_dir():
#         logger.info(f"Searching for challenge files in {arc_test_path}")
#         for f in arc_test_path.rglob("*.json"):
#             data, _ = load_json_if_file(f)
#             if data:
#                 if isinstance(data, dict):
#                     all_challenges.update(data)
#                     found_any_challenge = True
#                 else:
#                     logger.warning(f"Skipping non-dict JSON file in ARC test path: {f}")
# 
#     if found_any_challenge:
#         # Convert combined dictionary to a list of tasks if it makes sense for further processing
#         # Assuming challenge data is a dict of {task_id: task_details}
#         return all_challenges, str(arc_test_path)
# 
#     # Fallback to original candidates if no specific ARC test challenges found
#     logger.warning("No structured ARC test challenges found in expected path. Falling back to generic search.")
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try:
#                 data, used = load_json_if_file(p)
#                 if isinstance(data, dict): return data, p # Return the first valid dict file
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try:
#                         data, used = load_json_if_file(f)
#                         if isinstance(data, dict): return data, f # Return the first valid dict file
#                     except Exception: continue
#             for f in p.rglob("*.json"): # This iterates over ALL json files recursively.
#                 try:
#                     data, used = load_json_if_file(f)
#                     if isinstance(data, dict): return data, f # Return the first valid dict file
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try:
#             data, used = load_json_if_file(f)
#             if isinstance(data, dict): return data, f # Return the first valid dict file
#         except Exception: continue
#     return None, None
# 
# # --- finalize_submission (Updated) ---
# def finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy="weighted"):
#     # load challenge if not provided
#     if challenge is None:
#         # Default behavior: try to find challenge set from standard locations
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         # Challenge was explicitly provided (e.g., train_solutions or a custom path)
#         challenge_path_used = "provided_explicitly"
# 
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
# 
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
# 
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
# 
#             scores = defaultdict(float)
#             contributors = defaultdict(list)
# 
#             if voting_strategy == "weighted":
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             elif voting_strategy == "unweighted":
#                 # Simple majority vote
#                 for c in cands:
#                     g = c.get("grid")
#                     if g is None: continue
#                     k = grid_key(g)
#                     scores[k] += 1 # Each occurrence counts as one vote
#                     contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#             elif voting_strategy == "confidence-based":
#                 # Use confidence as the primary weight. If no confidence, treat as 0 or 1.
#                 any_confidence = False
#                 for c in cands:
#                     if c.get("confidence") is not None and isinstance(c.get("confidence"), (int, float)):
#                         any_confidence = True
#                         break
# 
#                 if not any_confidence:
#                     logger.warning(f"Task {tid}, test {ti}: No confidence scores found for confidence-based voting. Falling back to unweighted voting.")
#                     # Fallback to unweighted voting if no confidence is available
#                     for c in cands:
#                         g = c.get("grid")
#                         if g is None: continue
#                         k = grid_key(g)
#                         scores[k] += 1
#                         contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#                 else:
#                     for c in cands:
#                         sname = c.get("solver") or "unknown"
#                         g = c.get("grid")
#                         if g is None: continue
#                         conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                         score = conf_factor # Only confidence matters
#                         k = grid_key(g)
#                         scores[k] += score; contributors[k].append({"solver":sname,"weight":1,"confidence":conf})
#             else:
#                 logger.warning(f"Unknown voting strategy '{voting_strategy}'. Falling back to weighted voting.")
#                 # Default to weighted if strategy is unknown
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
# 
#             if not scores:
#                 chosen_grids.append([[0]]); continue # Fallback if no valid grids from candidates or voting method yields no scores
# 
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# # --- New Helper Function for Benchmarking ---
# def grid_to_set(grid):
#     if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
#         return set() # Return empty set for invalid or empty grids
#     s = set()
#     for r_idx, row in enumerate(grid):
#         for c_idx, color in enumerate(row):
#             s.add((r_idx, c_idx, color))
#     return s
# 
# def grid_jaccard_similarity(grid1, grid2):
#     set1 = grid_to_set(grid1)
#     set2 = grid_to_set(grid2)
#     if not set1 and not set2: # Both grids are empty or invalid, consider them 100% similar in a trivial way
#         return 1.0
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     if union == 0:
#         return 0.0 # Should not happen if at least one set is non-empty, but for safety
#     return intersection / union
# 
# # --- New Benchmark Function ---
# def benchmark_submission(final_submission, true_solutions):
#     total_correct_predictions = 0
#     total_predictions = 0
#     total_jaccard_scores = 0.0
#     task_benchmarks = {}
# 
#     for tid, predicted_output in final_submission.items():
#         true_grids = get_solution_grids_for_task(tid, true_solutions)
#         if not true_grids:
#             logger.debug(f"No true solutions found for task {tid}, skipping benchmarking.")
#             continue
# 
#         # Predicted output is typically a list containing a single dict of attempts
#         if isinstance(predicted_output, list) and predicted_output:
#             predicted_attempts = predicted_output[0]
#         else:
#             logger.warning(f"Unexpected format for predicted output for task {tid}: {predicted_output}")
#             continue
# 
#         num_tests = len(true_grids)
#         task_benchmarks[tid] = {"tests": []}
# 
#         for i in range(num_tests):
#             # Get predicted grid for the current test
#             pred_grid = predicted_attempts.get(f"attempt_{i+1}", [[0]]) # Default to empty grid
#             true_grid = true_grids[i]
# 
#             correct = 0
#             jaccard_score = 0.0
# 
#             if grid_key(pred_grid) == grid_key(true_grid):
#                 correct = 1
#                 jaccard_score = 1.0 # Jaccard is 1.0 if grids are identical
#             else:
#                 jaccard_score = grid_jaccard_similarity(pred_grid, true_grid)
# 
#             total_correct_predictions += correct
#             total_jaccard_scores += jaccard_score
#             total_predictions += 1
# 
#             task_benchmarks[tid]["tests"].append({
#                 "test_idx": i,
#                 "accuracy": correct,
#                 "jaccard_similarity": jaccard_score
#             })
# 
#     overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0.0
#     average_jaccard = total_jaccard_scores / total_predictions if total_predictions > 0 else 0.0
# 
#     return {
#         "overall_accuracy": overall_accuracy,
#         "average_jaccard_similarity": average_jaccard,
#         "total_predictions_evaluated": total_predictions,
#         "task_details": task_benchmarks
#     }
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     # Optionally save solver_stats as well for diagnostics/visualization
#     atomic_write_json(config.WORKING_DIR / "solver_stats.json", {s: {k: v if not isinstance(v, list) else v for k, v in stats.items()} for s, stats in solver_stats.items()}) # Fix set serialization to list here
# 
#     # --- Determine the challenge set for the official submission ---
#     submission_challenge_set = None
#     if config.SUBMISSION_CHALLENGE_MODE == "competition_test_set":
#         logger.info("SUBMISSION_CHALLENGE_MODE: competition_test_set. finalize_submission will search standard paths.")
#         submission_challenge_set = None # This signals finalize_submission to auto-discover
#     elif config.SUBMISSION_CHALLENGE_MODE == "train_set_for_submission":
#         logger.info("SUBMISSION_CHALLENGE_MODE: train_set_for_submission. Using train_solutions for submission.")
#         submission_challenge_set = train_solutions
#     elif config.SUBMISSION_CHALLENGE_MODE == "custom_path":
#         if config.CUSTOM_SUBMISSION_CHALLENGE_PATH and config.CUSTOM_SUBMISSION_CHALLENGE_PATH.exists():
#             logger.info(f"SUBMISSION_CHALLENGE_MODE: custom_path. Loading challenges from {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}")
#             # find_and_load_challenge needs config and candidates, passing custom_path as explicit
#             # The candidates are just a fallback if explicit_path doesn't yield anything
#             submission_challenge_set, _ = find_and_load_challenge(config, [], explicit_path=config.CUSTOM_SUBMISSION_CHALLENGE_PATH)
#             if submission_challenge_set is None:
#                 logger.error(f"Failed to load custom submission challenges from {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}.")
#                 raise FileNotFoundError(f"Custom submission challenge path invalid or empty: {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}")
#         else:
#             logger.error(f"SUBMISSION_CHALLENGE_MODE is 'custom_path' but CUSTOM_SUBMISSION_CHALLENGE_PATH is not set or does not exist: {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}")
#             raise ValueError("Custom submission challenge path not configured or not found.")
#     else:
#         logger.warning(f"Unknown SUBMISSION_CHALLENGE_MODE: {config.SUBMISSION_CHALLENGE_MODE}. Falling back to competition_test_set.")
#         submission_challenge_set = None
# 
#     # --- Generate the official submission ---
#     final_submission_for_output, report_for_output = finalize_submission(partial, solver_weights, config, challenge=submission_challenge_set, voting_strategy=config.VOTING_STRATEGY)
#     logger.info("Official submission predictions generated.")
# 
#     # --- Generate predictions specifically for benchmarking (always using train_solutions) ---
#     final_submission_for_benchmark, _ = finalize_submission(partial, solver_weights, config, challenge=train_solutions, voting_strategy=config.VOTING_STRATEGY)
#     logger.info("Benchmarking predictions generated using training set.")
# 
#     # --- Perform benchmarking ---
#     if train_solutions:
#         benchmarking_results = benchmark_submission(final_submission_for_benchmark, train_solutions)
#         report_for_output["benchmarking_results"] = benchmarking_results
#         logger.info("Benchmarking completed. Overall Accuracy: %.2f, Average Jaccard: %.2f", benchmarking_results["overall_accuracy"], benchmarking_results["average_jaccard_similarity"])
#     else:
#         logger.warning("No true solutions found for benchmarking.")
# 
#     if not dry_run:
#         # --- Save outputs ---
#         atomic_write_json(config.FINAL_SUB, final_submission_for_output)
#         report_for_output["summary"] = {"total_tasks": len(final_submission_for_output), "config_used": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}}
#         report_for_output["solver_metrics"] = {s: {k: v if not isinstance(v, set) else sorted(list(v)) for k, v in stats.items()} for s, stats in solver_stats.items()}
#         atomic_write_json(config.REPORT_PATH, report_for_output)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}}


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # --- Config Class (Updated) ---
# class Config:
#     def __init__(self, working_dir=None, input_dir=None, smoothing_alpha=None, voting_strategy=None, submission_challenge_mode=None, custom_submission_challenge_path=None):
#         self.WORKING_DIR = Path(working_dir) if working_dir else Path("/kaggle/working")
#         self.INPUT_DIR = Path(input_dir) if input_dir else Path("/kaggle/input")
# 
#         self.REPORT_PATH = self.WORKING_DIR / "execution_report.json"
#         self.FINAL_SUB = self.WORKING_DIR / "submission.json"
#         self.SOLVER_WEIGHTS_PATH = self.WORKING_DIR / "solver_weights.json"
#         self.LOG_PATH = self.WORKING_DIR / "finalize.log"
#         self.SMOOTHING_ALPHA = smoothing_alpha if smoothing_alpha is not None else 1.0
#         self.VOTING_STRATEGY = voting_strategy if voting_strategy else "weighted" # New: Default voting strategy
#         # New attributes for challenge set selection
#         self.SUBMISSION_CHALLENGE_MODE = submission_challenge_mode if submission_challenge_mode else "competition_test_set" # Options: "competition_test_set", "train_set_for_submission", "custom_path"
#         self.CUSTOM_SUBMISSION_CHALLENGE_PATH = Path(custom_submission_challenge_path) if custom_submission_challenge_path else None
# 
#         self.WORKING_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def load_from_json(self, config_file_path):
#         if not Path(config_file_path).is_file():
#             return
#         with open(config_file_path, 'r', encoding='utf-8') as f:
#             overrides = json.load(f)
#         for key, value in overrides.items():
#             if hasattr(self, key):
#                 # Handle Path objects correctly
#                 if 'DIR' in key.upper() or 'PATH' in key.upper() or 'SUB' in key.upper():
#                     setattr(self, key, Path(value))
#                 else:
#                     setattr(self, key, value)
# 
# # Logging (console + file) - Moved outside run_all to avoid re-initializing handlers
# logger = logging.getLogger("arc_finalize")
# logger.setLevel(logging.DEBUG)
# # Only add handlers if they don't already exist to prevent duplicate logging
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
#     fh = logging.FileHandler(Path("/kaggle/working") / "finalize.log", mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(ch); logger.addHandler(fh)
# 
# def atomic_write_json(p: Path, data):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_name(p.name + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, p)
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def find_json_files_under(dirpath: Path, pattern="*.json"):
#     if not dirpath.exists():
#         return []
#     files = list(dirpath.glob(pattern))
#     if files:
#         return files
#     return list(dirpath.rglob(pattern))
# 
# def merge_partial_submissions(paths: Iterable[Path]):
#     partial = defaultdict(list)
#     parse_stats = {"scanned": 0, "parsed": 0, "skipped": 0, "errors": []}
#     for p in paths:
#         parse_stats["scanned"] += 1
#         data, used = load_json_if_file(p)
#         if data is None:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "parse_failed_or_not_file"})
#             continue
#         parse_stats["parsed"] += 1
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, list):
#                     partial[k].extend(v)
#                 else:
#                     partial[k].append(v)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict):
#                     continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     v = item.get("prediction") or item.get("output") or item.get("solutions") or item
#                     partial[tid].append(v)
#                 else:
#                     if len(item) == 1:
#                         tid = next(iter(item.keys()))
#                         partial[tid].append(item[tid])
#                         continue
#         else:
#             parse_stats["skipped"] += 1
#             parse_stats["errors"].append({"path": str(p), "reason": "unexpected_top_level_type", "type": str(type(data))})
#     return partial, parse_stats
# 
# def extract_candidates(entry_list, expected_tests=1):
#     per_test = [[] for _ in range(max(1, expected_tests))]
#     if not entry_list:
#         return per_test
#     for item in entry_list:
#         solver = None; pred = None
#         if isinstance(item, dict) and "solver" in item and ("prediction" in item or "grid" in item or "output" in item):
#             solver = item.get("solver"); pred = item.get("prediction") or item.get("output") or item.get("grid")
#         elif isinstance(item, dict) and any(k.startswith("attempt_") for k in item.keys()):
#             if expected_tests == 1:
#                 val = item.get("attempt_1") or item.get("attempt") or item.get("prediction")
#                 per_test[0].append({"solver": item.get("solver", "final_format"), "grid": val, "confidence": item.get("confidence")})
#             else:
#                 for ti in range(expected_tests):
#                     k = f"attempt_{ti+1}"
#                     if k in item:
#                         per_test[ti].append({"solver": item.get("solver", "final_format"), "grid": item.get(k), "confidence": None})
#             continue
#         else:
#             pred = item
# 
#         if isinstance(pred, list) and all(isinstance(x, list) for x in pred) and len(pred) == expected_tests:
#             for ti, g in enumerate(pred):
#                 per_test[ti].append({"solver": solver, "grid": g, "confidence": None})
#         elif isinstance(pred, list) and expected_tests == 1:
#             per_test[0].append({"solver": solver, "grid": pred[0] if len(pred) > 0 else pred, "confidence": None})
#         elif isinstance(pred, dict) and "grid" in pred:
#             per_test[0].append({"solver": solver, "grid": pred["grid"], "confidence": pred.get("confidence")})
#         else:
#             if expected_tests == 1:
#                 per_test[0].append({"solver": solver, "grid": pred, "confidence": None})
#     return per_test
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         return [v for v in sol.values()]
#     if isinstance(sol, list):
#         return sol
#     return [sol]
# 
# def train_solver_weights(partial, train_solutions, alpha=1.0):
#     solver_stats = defaultdict(lambda: {"attempts":0, "correct":0, "total_confidence_sum":0.0, "task_ids_attempted":set()})
#     overlap_tasks = [tid for tid in train_solutions.keys() if tid in partial]
#     logger.info("Computing solver stats from %d overlapping tasks", len(overlap_tasks))
#     for tid in overlap_tasks:
#         sol_grids = get_solution_grids_for_task(tid, train_solutions)
#         if not sol_grids: continue
#         n_tests = len(sol_grids)
#         candidates = extract_candidates(partial.get(tid, []), expected_tests=n_tests)
#         for ti in range(n_tests):
#             expected_grid = sol_grids[ti]
#             expected_key = grid_key(expected_grid)
#             for c in candidates[ti]:
#                 solver = c.get("solver") or "unknown"
#                 pred_grid = c.get("grid")
#                 if pred_grid is None: continue
#                 solver_stats[solver]["attempts"] += 1
#                 solver_stats[solver]["task_ids_attempted"].add(tid)
#                 if grid_key(pred_grid) == expected_key:
#                     solver_stats[solver]["correct"] += 1
#                     conf = c.get("confidence")
#                     if conf is not None and isinstance(conf,(int,float)):
#                         solver_stats[solver]["total_confidence_sum"] += float(conf)
#     # compute smoothed weights
#     raw_weights = {}; total = 0.0
#     any_attempts = any(v["attempts"]>0 for v in solver_stats.values())
#     if any_attempts:
#         for s, st in solver_stats.items():
#             a = st["attempts"]; c = st["correct"]
#             score = (c + alpha) / (a + 2.0*alpha)
#             raw_weights[s] = float(score); total += raw_weights[s]
# 
#             # Calculate additional metrics
#             st["average_confidence"] = st["total_confidence_sum"] / st["correct"] if st["correct"] > 0 else 0.0
#             st["num_tasks_attempted"] = len(st["task_ids_attempted"])
#             # Convert set to list for JSON serialization if needed later
#             st["task_ids_attempted"] = sorted(list(st["task_ids_attempted"])) # Convert set to list here
# 
#         # tiny weight for unseen solvers in partial
#         for tid, entries in partial.items():
#             for e in entries:
#                 if isinstance(e, dict) and "solver" in e:
#                     s = e["solver"]
#                     if s not in raw_weights: # Check if solver already has a weight
#                         raw_weights[s] = 0.01; total += 0.01 # Add a small default weight for solvers that never made an attempt but generated a prediction
#                         solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                         solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#         solver_weights = {s: (w/total) for s,w in raw_weights.items()} if total > 0 else {"unknown": 1.0}
#     else:
#         sols = {e.get("solver") for entries in partial.values() for e in entries if isinstance(e, dict) and "solver" in e}
#         if not sols:
#             solver_weights = {"unknown":1.0}
#         else:
#             # If no attempts were made but solvers generated predictions, distribute weights equally
#             solver_weights = {s: 1.0/len(sols) for s in sols}
#             for s in sols:
#                 solver_stats[s]["num_tasks_attempted"] = len(solver_stats[s]["task_ids_attempted"])
#                 solver_stats[s]["task_ids_attempted"] = sorted(list(solver_stats[s]["task_ids_attempted"])) # Convert set to list here
# 
#     return solver_weights, solver_stats
# 
# # --- find_and_load_challenge (Modified) ---
# def find_and_load_challenge(config, candidates, explicit_path=None):
#     all_challenges = {}
#     found_any_challenge = False
# 
#     if explicit_path and Path(explicit_path).exists():
#         logger.info(f"Prioritizing challenge files from explicit path: {explicit_path}")
#         p = Path(explicit_path)
#         if p.is_file():
#             data, _ = load_json_if_file(p)
#             if data and isinstance(data, dict):
#                 all_challenges.update(data)
#                 found_any_challenge = True
#         elif p.is_dir():
#             for f in p.rglob("*.json"):
#                 data, _ = load_json_if_file(f)
#                 if data and isinstance(data, dict):
#                     all_challenges.update(data)
#                     found_any_challenge = True
# 
#     if found_any_challenge:
#         return all_challenges, str(explicit_path)
# 
#     # Prioritize loading challenge tasks from the standard ARC test path if no explicit path or if explicit path yielded nothing
#     arc_test_path = config.INPUT_DIR / "arc-prize-2025" / "test"
#     if arc_test_path.exists() and arc_test_path.is_dir():
#         logger.info(f"Searching for challenge files in {arc_test_path}")
#         for f in arc_test_path.rglob("*.json"):
#             data, _ = load_json_if_file(f)
#             if data:
#                 if isinstance(data, dict):
#                     all_challenges.update(data)
#                     found_any_challenge = True
#                 else:
#                     logger.warning(f"Skipping non-dict JSON file in ARC test path: {f}")
# 
#     if found_any_challenge:
#         # Convert combined dictionary to a list of tasks if it makes sense for further processing
#         # Assuming challenge data is a dict of {task_id: task_details}
#         return all_challenges, str(arc_test_path)
# 
#     # Fallback to original candidates if no specific ARC test challenges found
#     logger.warning("No structured ARC test challenges found in expected path. Falling back to generic search.")
#     for p in candidates:
#         if not p.exists(): continue
#         if p.is_file():
#             try:
#                 data, used = load_json_if_file(p)
#                 if isinstance(data, dict): return data, p # Return the first valid dict file
#             except Exception: continue
#         if p.is_dir():
#             for pat in ("arc*challenges.json","*challenges.json","*.json"):
#                 for f in sorted(p.glob(pat)):
#                     if not f.is_file(): continue
#                     try:
#                         data, used = load_json_if_file(f)
#                         if isinstance(data, dict): return data, f # Return the first valid dict file
#                     except Exception: continue
#             for f in p.rglob("*.json"): # This iterates over ALL json files recursively.
#                 try:
#                     data, used = load_json_if_file(f)
#                     if isinstance(data, dict): return data, f # Return the first valid dict file
#                 except Exception: continue
#     for f in config.INPUT_DIR.rglob("*arc*challenges*.json"):
#         try:
#             data, used = load_json_if_file(f)
#             if isinstance(data, dict): return data, f # Return the first valid dict file
#         except Exception: continue
#     return None, None
# 
# # --- finalize_submission (Updated) ---
# def finalize_submission(partial, solver_weights, config, challenge=None, voting_strategy="weighted"):
#     # load challenge if not provided
#     if challenge is None:
#         # Default behavior: try to find challenge set from standard locations
#         challenge_candidates = [config.WORKING_DIR, config.INPUT_DIR / "arc-prize-2025", config.INPUT_DIR]
#         challenge, challenge_path_used = find_and_load_challenge(config, challenge_candidates)
#         if challenge is None:
#             raise FileNotFoundError("Cannot find evaluation challenge JSON to finalize submission.")
#     else:
#         # Challenge was explicitly provided (e.g., train_solutions or a custom path)
#         challenge_path_used = "provided_explicitly"
# 
#     # build task list
#     if isinstance(challenge, dict):
#         task_items = list(challenge.items())
#     else:
#         task_items = [(t.get("id", str(i)), t) for i, t in enumerate(challenge)]
#     task_test_counts = {}
#     for tid, tdata in task_items:
#         tests = None
#         if isinstance(tdata, dict):
#             tests = tdata.get("test") or tdata.get("tests")
#             if tests is not None:
#                 task_test_counts[tid] = len(tests); continue
#         task_test_counts[tid] = 1
# 
#     final_submission = {}; report = {"tasks":{}}
#     for tid, n_tests in task_test_counts.items():
#         report["tasks"].setdefault(tid, {"chosen":{}}) # Moved this line to ensure initialization
#         entries = partial.get(tid, [])
#         candidates_per_test = extract_candidates(entries, expected_tests=n_tests)
#         chosen_grids = []
# 
#         for ti in range(n_tests):
#             cands = candidates_per_test[ti]
#             if not cands:
#                 chosen_grids.append([[0]]); continue # Default to empty grid if no candidates
# 
#             scores = defaultdict(float)
#             contributors = defaultdict(list)
# 
#             if voting_strategy == "weighted":
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
#             elif voting_strategy == "unweighted":
#                 # Simple majority vote
#                 for c in cands:
#                     g = c.get("grid")
#                     if g is None: continue
#                     k = grid_key(g)
#                     scores[k] += 1 # Each occurrence counts as one vote
#                     contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#             elif voting_strategy == "confidence-based":
#                 # Use confidence as the primary weight. If no confidence, treat as 0 or 1.
#                 any_confidence = False
#                 for c in cands:
#                     if c.get("confidence") is not None and isinstance(c.get("confidence"), (int, float)):
#                         any_confidence = True
#                         break
# 
#                 if not any_confidence:
#                     logger.warning(f"Task {tid}, test {ti}: No confidence scores found for confidence-based voting. Falling back to unweighted voting.")
#                     # Fallback to unweighted voting if no confidence is available
#                     for c in cands:
#                         g = c.get("grid")
#                         if g is None: continue
#                         k = grid_key(g)
#                         scores[k] += 1
#                         contributors[k].append({"solver":c.get("solver"),"weight":1,"confidence":c.get("confidence")})
#                 else:
#                     for c in cands:
#                         sname = c.get("solver") or "unknown"
#                         g = c.get("grid")
#                         if g is None: continue
#                         conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                         score = conf_factor # Only confidence matters
#                         k = grid_key(g)
#                         scores[k] += score; contributors[k].append({"solver":sname,"weight":1,"confidence":conf})
#             else:
#                 logger.warning(f"Unknown voting strategy '{voting_strategy}'. Falling back to weighted voting.")
#                 # Default to weighted if strategy is unknown
#                 for c in cands:
#                     sname = c.get("solver") or "unknown"
#                     g = c.get("grid")
#                     if g is None: continue
#                     conf = c.get("confidence"); conf_factor = 1.0 + float(conf) if (conf is not None and isinstance(conf,(int,float))) else 1.0
#                     w = solver_weights.get(sname, 0.01); score = w * conf_factor; k = grid_key(g)
#                     scores[k] += score; contributors[k].append({"solver":sname,"weight":w,"confidence":conf})
# 
#             if not scores:
#                 chosen_grids.append([[0]]); continue # Fallback if no valid grids from candidates or voting method yields no scores
# 
#             ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#             top_k = ranked[0][0]; chosen1 = json.loads(top_k)
#             chosen_grids.append(chosen1)
#             report["tasks"][tid]["chosen"][str(ti)] = {"winner_score": scores[top_k], "runner_score": scores.get(ranked[1][0],0.0) if len(ranked)>1 else 0.0, "contributors": contributors[top_k][:5]}
#         attempt = {};
#         for idx,g in enumerate(chosen_grids): attempt[f"attempt_{idx+1}"] = g
#         final_submission[tid] = [attempt]; report["tasks"][tid]["final_attempt"] = attempt
#     return final_submission, report
# 
# # --- New Helper Function for Benchmarking ---
# def grid_to_set(grid):
#     if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
#         return set() # Return empty set for invalid or empty grids
#     s = set()
#     for r_idx, row in enumerate(grid):
#         for c_idx, color in enumerate(row):
#             s.add((r_idx, c_idx, color))
#     return s
# 
# def grid_jaccard_similarity(grid1, grid2):
#     set1 = grid_to_set(grid1)
#     set2 = grid_to_set(grid2)
#     if not set1 and not set2: # Both grids are empty or invalid, consider them 100% similar in a trivial way
#         return 1.0
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     if union == 0:
#         return 0.0 # Should not happen if at least one set is non-empty, but for safety
#     return intersection / union
# 
# # --- New Benchmark Function ---
# def benchmark_submission(final_submission, true_solutions):
#     total_correct_predictions = 0
#     total_predictions = 0
#     total_jaccard_scores = 0.0
#     task_benchmarks = {}
# 
#     for tid, predicted_output in final_submission.items():
#         true_grids = get_solution_grids_for_task(tid, true_solutions)
#         if not true_grids:
#             logger.debug(f"No true solutions found for task {tid}, skipping benchmarking.")
#             continue
# 
#         # Predicted output is typically a list containing a single dict of attempts
#         if isinstance(predicted_output, list) and predicted_output:
#             predicted_attempts = predicted_output[0]
#         else:
#             logger.warning(f"Unexpected format for predicted output for task {tid}: {predicted_output}")
#             continue
# 
#         num_tests = len(true_grids)
#         task_benchmarks[tid] = {"tests": []}
# 
#         for i in range(num_tests):
#             # Get predicted grid for the current test
#             pred_grid = predicted_attempts.get(f"attempt_{i+1}", [[0]]) # Default to empty grid
#             true_grid = true_grids[i]
# 
#             correct = 0
#             jaccard_score = 0.0
# 
#             if grid_key(pred_grid) == grid_key(true_grid):
#                 correct = 1
#                 jaccard_score = 1.0 # Jaccard is 1.0 if grids are identical
#             else:
#                 jaccard_score = grid_jaccard_similarity(pred_grid, true_grid)
# 
#             total_correct_predictions += correct
#             total_jaccard_scores += jaccard_score
#             total_predictions += 1
# 
#             task_benchmarks[tid]["tests"].append({
#                 "test_idx": i,
#                 "accuracy": correct,
#                 "jaccard_similarity": jaccard_score
#             })
# 
#     overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0.0
#     average_jaccard = total_jaccard_scores / total_predictions if total_predictions > 0 else 0.0
# 
#     return {
#         "overall_accuracy": overall_accuracy,
#         "average_jaccard_similarity": average_jaccard,
#         "total_predictions_evaluated": total_predictions,
#         "task_details": task_benchmarks
#     }
# 
# # Run helper to produce submission and write outputs
# def run_all(config_file_path=None, dry_run=False):
#     config = Config() # Create a default config instance
#     if config_file_path: # Load overrides if path is provided
#         config.load_from_json(config_file_path)
# 
#     # Update logger to use config.LOG_PATH
#     for handler in logger.handlers:
#         if isinstance(handler, logging.FileHandler):
#             logger.removeHandler(handler)
#     fh = logging.FileHandler(config.LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG)
#     fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"))
#     logger.addHandler(fh)
# 
#     # discover partials
#     partial_paths = []
#     for p in config.WORKING_DIR.glob("*.json"):
#         if p.name in {config.FINAL_SUB.name, config.REPORT_PATH.name, config.SOLVER_WEIGHTS_PATH.name}: continue
#         partial_paths.append(p)
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         partial_paths.extend(find_json_files_under(d, pattern="*.json"))
#     partial_paths = sorted({p.resolve() for p in partial_paths if p.is_file()})
#     logger.info("Partial JSON files discovered: %d", len(partial_paths))
#     partial, parse_stats = merge_partial_submissions(partial_paths)
#     atomic_write_json(config.WORKING_DIR / "partial_snapshot.json", {"tasks": len(partial), "parse_stats": parse_stats})
# 
#     # load training solutions
#     train_solution_files = []
#     for d in config.INPUT_DIR.iterdir() if config.INPUT_DIR.exists() else []:
#         if "arc" in d.name.lower() or "train" in d.name.lower() or "solution" in d.name.lower():
#             train_solution_files.extend(find_json_files_under(d, pattern="*.json"))
#     for p in config.INPUT_DIR.rglob("*.json") if config.INPUT_DIR.exists() else []:
#         if "solution" in p.name.lower() or "train" in p.name.lower():
#             train_solution_files.append(p)
#     train_solution_files = sorted({p.resolve() for p in train_solution_files})
#     train_solutions = {}
#     for p in train_solution_files:
#         data, used = load_json_if_file(p)
#         if data is None: continue
#         if isinstance(data, dict): train_solutions.update(data)
#         elif isinstance(data, list):
#             for item in data:
#                 if not isinstance(item, dict): continue
#                 tid = item.get("id") or item.get("task_id") or item.get("name")
#                 if tid:
#                     out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                     train_solutions[tid] = out
#     atomic_write_json(config.WORKING_DIR / "train_snapshot.json", {"tasks": len(train_solutions), "files": [str(x) for x in train_solution_files[:20]]})
# 
#     solver_weights, solver_stats = train_solver_weights(partial, train_solutions, alpha=config.SMOOTHING_ALPHA)
#     atomic_write_json(config.SOLVER_WEIGHTS_PATH, solver_weights)
#     logger.info("Solver weights saved: %s", config.SOLVER_WEIGHTS_PATH)
# 
#     # Optionally save solver_stats as well for diagnostics/visualization
#     atomic_write_json(config.WORKING_DIR / "solver_stats.json", {s: {k: v if not isinstance(v, list) else v for k, v in stats.items()} for s, stats in solver_stats.items()}) # Fix set serialization to list here
# 
#     # --- Determine the challenge set for the official submission ---
#     submission_challenge_set = None
#     if config.SUBMISSION_CHALLENGE_MODE == "competition_test_set":
#         logger.info("SUBMISSION_CHALLENGE_MODE: competition_test_set. finalize_submission will search standard paths.")
#         submission_challenge_set = None # This signals finalize_submission to auto-discover
#     elif config.SUBMISSION_CHALLENGE_MODE == "train_set_for_submission":
#         logger.info("SUBMISSION_CHALLENGE_MODE: train_set_for_submission. Using train_solutions for submission.")
#         submission_challenge_set = train_solutions
#     elif config.SUBMISSION_CHALLENGE_MODE == "custom_path":
#         if config.CUSTOM_SUBMISSION_CHALLENGE_PATH and config.CUSTOM_SUBMISSION_CHALLENGE_PATH.exists():
#             logger.info(f"SUBMISSION_CHALLENGE_MODE: custom_path. Loading challenges from {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}")
#             # find_and_load_challenge needs config and candidates, passing custom_path as explicit
#             # The candidates are just a fallback if explicit_path doesn't yield anything
#             submission_challenge_set, _ = find_and_load_challenge(config, [], explicit_path=config.CUSTOM_SUBMISSION_CHALLENGE_PATH)
#             if submission_challenge_set is None:
#                 logger.error(f"Failed to load custom submission challenges from {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}.")
#                 raise FileNotFoundError(f"Custom submission challenge path invalid or empty: {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}")
#         else:
#             logger.error(f"SUBMISSION_CHALLENGE_MODE is 'custom_path' but CUSTOM_SUBMISSION_CHALLENGE_PATH is not set or does not exist: {config.CUSTOM_SUBMISSION_CHALLENGE_PATH}")
#             raise ValueError("Custom submission challenge path not configured or not found.")
#     else:
#         logger.warning(f"Unknown SUBMISSION_CHALLENGE_MODE: {config.SUBMISSION_CHALLENGE_MODE}. Falling back to competition_test_set.")
#         submission_challenge_set = None
# 
#     # --- Generate the official submission ---
#     final_submission_for_output, report_for_output = finalize_submission(partial, solver_weights, config, challenge=submission_challenge_set, voting_strategy=config.VOTING_STRATEGY)
#     logger.info("Official submission predictions generated.")
# 
#     # --- Generate predictions specifically for benchmarking (always using train_solutions) ---
#     final_submission_for_benchmark, _ = finalize_submission(partial, solver_weights, config, challenge=train_solutions, voting_strategy=config.VOTING_STRATEGY)
#     logger.info("Benchmarking predictions generated using training set.")
# 
#     # --- Perform benchmarking ---
#     if train_solutions:
#         benchmarking_results = benchmark_submission(final_submission_for_benchmark, train_solutions)
#         report_for_output["benchmarking_results"] = benchmarking_results
#         logger.info("Benchmarking completed. Overall Accuracy: %.2f, Average Jaccard: %.2f", benchmarking_results["overall_accuracy"], benchmarking_results["average_jaccard_similarity"])
#     else:
#         logger.warning("No true solutions found for benchmarking.")
# 
#     if not dry_run:
#         # --- Save outputs ---
#         atomic_write_json(config.FINAL_SUB, final_submission_for_output)
#         report_for_output["summary"] = {"total_tasks": len(final_submission_for_output), "config_used": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items() if isinstance(v, Path) or isinstance(v, (int, float, str))}}
#         report_for_output["solver_metrics"] = {s: {k: v if not isinstance(v, set) else sorted(list(v)) for k, v in stats.items()} for s, stats in solver_stats.items()}
#         atomic_write_json(config.REPORT_PATH, report_for_output)
#         logger.info("Final submission and report written to /kaggle/working")
#     return {"partial_files": len(partial_paths), "partial_tasks": len(partial), "train_tasks": len(train_solutions), "solvers": len(solver_weights), "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}}


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json

# import json
# from pathlib import Path
# import logging # Import logging as load_json_if_file might use it
# from collections import defaultdict # Needed for helper function get_solution_grids_for_task if it uses it.
# 
# # Assuming logger is configured in a previous cell. If not, a minimal configuration might be needed.
# # For self-containment, adding a basic logger setup here if not already present.
# logger = logging.getLogger("arc_finalize")
# if not logger.handlers:
#     ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
#     logger.addHandler(ch)
# 
# # --- Helper functions (copied for self-containment in this debug cell) ---
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         if 'train' in sol and 'test' in sol: # Specific ARC training task format
#              out = []
#              for t_entry in sol['test']:
#                  if 'output' in t_entry:
#                      out.append(t_entry['output'])
#                  elif 'grid' in t_entry:
#                      out.append(t_entry['grid'])
#                  else:
#                      out.append(t_entry) # Fallback if test entry is just the grid
#              return out
#         return [v for v in sol.values()] # Fallback, original behavior
#     if isinstance(sol, list):
#         return sol
#     return [sol] # sol is a single grid
# # --- End of Helper functions ---
# 
# 
# # Define paths to the report and train snapshot files
# report_path = Path("/kaggle/working/execution_report.json")
# train_snapshot_path = Path("/kaggle/working/train_snapshot.json")
# 
# # Load the execution report
# report_for_output = {}
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report_for_output = json.load(f)
#     print("Execution report loaded successfully.")
# else:
#     print(f"Error: {report_path} not found.")
# 
# # Load the train snapshot and reconstruct train_solutions
# train_solutions = {}
# if train_snapshot_path.exists():
#     with open(train_snapshot_path, 'r', encoding='utf-8') as f:
#         train_snapshot = json.load(f)
#     print(f"Train snapshot loaded. Found {train_snapshot.get('tasks', 0)} tasks.")
# 
#     # Reconstruct train_solutions from the files listed in the snapshot
#     for file_path_str in train_snapshot.get("files", []):
#         p = Path(file_path_str)
#         # Only load from actual solution files, not challenge files or sample submission
#         if "solution" in p.name.lower(): # Match logic from run_all for solution files
#             data, _ = load_json_if_file(p)
#             if data is None:
#                 logger.warning(f"Could not load data from {p}. Skipping.")
#                 continue
#             if isinstance(data, dict):
#                 train_solutions.update(data)
#             elif isinstance(data, list): # If it's a list, it's typically a list of task objects
#                 for item in data:
#                     if not isinstance(item, dict): continue
#                     tid = item.get("id") or item.get("task_id") or item.get("name")
#                     if tid:
#                         out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                         train_solutions[tid] = out
#     print(f"Reconstructed train_solutions with {len(train_solutions)} tasks.")
# else:
#     print(f"Train snapshot not found at {train_snapshot_path}. Cannot reconstruct true solutions.")
# 
# # Determine the task_id to compare
# task_id = None
# if 'benchmarking_results' in report_for_output and 'task_details' in report_for_output['benchmarking_results']:
#     if len(report_for_output['benchmarking_results']['task_details']) > 0:
#         # Pick the first valid task_id from the benchmarking results for which there are actual test results
#         for tid_candidate, details in report_for_output['benchmarking_results']['task_details'].items():
#             # Check if this task ID exists in the reconstructed train_solutions and has test cases
#             if tid_candidate in train_solutions and get_solution_grids_for_task(tid_candidate, train_solutions):
#                 task_id = tid_candidate
#                 break
#         if task_id:
#             print(f"Using valid task ID for comparison from benchmarking_results: {task_id}")
#         else:
#             print("No tasks with valid true solutions found in benchmarking_results. Falling back to '00d62c1b'.")
#             task_id = '00d62c1b' # Fallback
#     else:
#         print("No tasks in benchmarking_results. Falling back to '00d62c1b'.")
#         task_id = '00d62c1b' # Fallback
# else:
#     print("Benchmarking results not found in report. Falling back to '00d62c1b'.")
#     task_id = '00d62c1b' # Fallback
# 
# 
# print(f"--- Task ID: {task_id} ---")
# print("Predicted Output (from benchmark run):")
# predicted_output = None
# if 'tasks' in report_for_output and task_id in report_for_output['tasks']:
#     task_report = report_for_output['tasks'].get(task_id)
#     if task_report and 'final_attempt' in task_report:
#         # predicted_output is a dictionary like {'attempt_1': grid}
#         predicted_output = task_report['final_attempt']
# 
# if predicted_output:
#     display(predicted_output)
# else:
#     print(f"No predicted output found for task {task_id} in report_for_output.")
# 
# print("\nTrue Solution (from training set):")
# true_solution = None
# if train_solutions and task_id in train_solutions:
#     sol_grids = get_solution_grids_for_task(task_id, train_solutions)
#     if sol_grids:
#         true_solution = sol_grids[0] # Assuming we are interested in the first test case
# 
# if true_solution:
#     display(true_solution)
# else:
#     print(f"No true solution found for task {task_id} in reconstructed train_solutions.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # Assuming logger is configured in a previous cell. If not, a minimal configuration might be needed.
# # For self-containment, adding a basic logger setup here if not already present.
# logger = logging.getLogger("arc_finalize")
# if not logger.handlers:
#     ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
#     logger.addHandler(ch)
# 
# # --- Helper functions (copied for self-containment in this debug cell) ---
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         if 'train' in sol and 'test' in sol: # Specific ARC training task format
#              out = []
#              for t_entry in sol['test']:
#                  if 'output' in t_entry:
#                      out.append(t_entry['output'])
#                  elif 'grid' in t_entry:
#                      out.append(t_entry['grid'])
#                  else:
#                      out.append(t_entry) # Fallback if test entry is just the grid
#              return out
#         return [v for v in sol.values()] # Fallback, original behavior
#     if isinstance(sol, list):
#         return sol
#     return [sol] # sol is a single grid
# # --- End of Helper functions ---
# 
# 
# # Define paths to the report and train snapshot files
# report_path = Path("/kaggle/working/execution_report.json")
# train_snapshot_path = Path("/kaggle/working/train_snapshot.json")
# 
# # Load the execution report
# report_for_output = {}
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report_for_output = json.load(f)
#     print("Execution report loaded successfully.")
# else:
#     print(f"Error: {report_path} not found.")
# 
# # Load the train snapshot and reconstruct train_solutions
# train_solutions = {}
# if train_snapshot_path.exists():
#     with open(train_snapshot_path, 'r', encoding='utf-8') as f:
#         train_snapshot = json.load(f)
#     print(f"Train snapshot loaded. Found {train_snapshot.get('tasks', 0)} tasks.")
# 
#     # Reconstruct train_solutions from the files listed in the snapshot
#     for file_path_str in train_snapshot.get("files", []):
#         p = Path(file_path_str)
#         # Only load from actual solution files, not challenge files or sample submission
#         if "solution" in p.name.lower(): # Match logic from run_all for solution files
#             data, _ = load_json_if_file(p)
#             if data is None:
#                 logger.warning(f"Could not load data from {p}. Skipping.")
#                 continue
#             if isinstance(data, dict):
#                 train_solutions.update(data)
#             elif isinstance(data, list): # If it's a list, it's typically a list of task objects
#                 for item in data:
#                     if not isinstance(item, dict): continue
#                     tid = item.get("id") or item.get("task_id") or item.get("name")
#                     if tid:
#                         out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                         train_solutions[tid] = out
#     print(f"Reconstructed train_solutions with {len(train_solutions)} tasks.")
# else:
#     print(f"Train snapshot not found at {train_snapshot_path}. Cannot reconstruct true solutions.")
# 
# # Determine the task_id to compare
# task_id = None
# if 'benchmarking_results' in report_for_output and 'task_details' in report_for_output['benchmarking_results']:
#     if len(report_for_output['benchmarking_results']['task_details']) > 0:
#         for tid_candidate, details in report_for_output['benchmarking_results']['task_details'].items():
#             if tid_candidate in train_solutions and get_solution_grids_for_task(tid_candidate, train_solutions):
#                 task_id = tid_candidate
#                 break
#         if task_id:
#             print(f"Using valid task ID for comparison from benchmarking_results: {task_id}")
#         else:
#             print("No tasks with valid true solutions found in benchmarking_results. Falling back to '00d62c1b'.")
#             task_id = '00d62c1b' # Fallback
#     else:
#         print("No tasks in benchmarking_results. Falling back to '00d62c1b'.")
#         task_id = '00d62c1b' # Fallback
# else:
#     print("Benchmarking results not found in report. Falling back to '00d62c1b'.")
#     task_id = '00d62c1b' # Fallback
# 
# 
# # --- Detailed Analysis for the selected task_id ---
# print(f"\n--- Detailed Analysis for Task ID: {task_id} ---\n")
# 
# predicted_output_raw = None
# if 'tasks' in report_for_output and task_id in report_for_output['tasks']:
#     task_report = report_for_output['tasks'].get(task_id)
#     if task_report and 'final_attempt' in task_report:
#         # The `final_attempt` is a dict like {'attempt_1': grid_content, 'attempt_2': grid_content}
#         # We are interested in the grid content of the first attempt (attempt_1) for comparison
#         predicted_output_raw = task_report['final_attempt'].get('attempt_1')
# 
# true_solution_raw = None
# if train_solutions and task_id in train_solutions:
#     sol_grids = get_solution_grids_for_task(task_id, train_solutions)
#     if sol_grids:
#         true_solution_raw = sol_grids[0] # Assuming we are interested in the first test case
# 
# def get_element_type(grid_data):
#     if not isinstance(grid_data, list) or not grid_data:
#         return "N/A"
#     first_element = grid_data[0]
#     if isinstance(first_element, list) and first_element:
#         return type(first_element[0]).__name__
#     return type(first_element).__name__
# 
# 
# print("=== Predicted Grid Details ===")
# print(f"Raw content: {json.dumps(predicted_output_raw, ensure_ascii=False)}")
# print(f"Type: {type(predicted_output_raw).__name__}")
# if isinstance(predicted_output_raw, list) and predicted_output_raw:
#     print(f"Element type (first level): {type(predicted_output_raw[0]).__name__}")
#     if isinstance(predicted_output_raw[0], list) and predicted_output_raw[0]:
#         print(f"Element type (second level): {type(predicted_output_raw[0][0]).__name__}")
# else:
#     print(f"Element type: {get_element_type(predicted_output_raw)}")
# print(f"Grid Key: {grid_key(predicted_output_raw)}")
# 
# print("\n=== True Solution Grid Details ===")
# print(f"Raw content: {json.dumps(true_solution_raw, ensure_ascii=False)}")
# print(f"Type: {type(true_solution_raw).__name__}")
# if isinstance(true_solution_raw, list) and true_solution_raw:
#     print(f"Element type (first level): {type(true_solution_raw[0]).__name__}")
#     if isinstance(true_solution_raw[0], list) and true_solution_raw[0]:
#         print(f"Element type (second level): {type(true_solution_raw[0][0]).__name__}")
# else:
#     print(f"Element type: {get_element_type(true_solution_raw)}")
# print(f"Grid Key: {grid_key(true_solution_raw)}")
# 
# print("\n=== Grid Key Comparison ===")
# if predicted_output_raw is not None and true_solution_raw is not None:
#     are_keys_identical = (grid_key(predicted_output_raw) == grid_key(true_solution_raw))
#     print(f"Grid Key values are identical: {are_keys_identical}")
#     if not are_keys_identical:
#         print("Reason for difference: The JSON representations (grid keys) of the predicted and true grids do not match. This could be due to differences in content, ordering (if not handled by sort_keys), or data types within the grids.")
# else:
#     print("Cannot compare grid keys: One or both grids are None.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # Assuming logger is configured in a previous cell. If not, a minimal configuration might be needed.
# # For self-containment, adding a basic logger setup here if not already present.
# logger = logging.getLogger("arc_finalize")
# if not logger.handlers:
#     ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
#     logger.addHandler(ch)
# 
# # --- Helper functions (copied for self-containment in this debug cell) ---
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         if 'train' in sol and 'test' in sol: # Specific ARC training task format
#              out = []
#              for t_entry in sol['test']:
#                  if 'output' in t_entry:
#                      out.append(t_entry['output'])
#                  elif 'grid' in t_entry:
#                      out.append(t_entry['grid'])
#                  else:
#                      out.append(t_entry)
#              return out
#         return [v for v in sol.values()] # Fallback, original behavior
#     if isinstance(sol, list):
#         return sol
#     return [sol] # sol is a single grid
# # --- End of Helper functions ---
# 
# 
# # Define paths to the report and train snapshot files
# report_path = Path("/kaggle/working/execution_report.json")
# train_snapshot_path = Path("/kaggle/working/train_snapshot.json")
# 
# # Load the execution report
# report_for_output = {}
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report_for_output = json.load(f)
#     print("Execution report loaded successfully.")
# else:
#     print(f"Error: {report_path} not found.")
# 
# # Load the train snapshot and reconstruct train_solutions
# train_solutions = {}
# if train_snapshot_path.exists():
#     with open(train_snapshot_path, 'r', encoding='utf-8') as f:
#         train_snapshot = json.load(f)
#     print(f"Train snapshot loaded. Found {train_snapshot.get('tasks', 0)} tasks.")
# 
#     # Reconstruct train_solutions from the files listed in the snapshot
#     for file_path_str in train_snapshot.get("files", []):
#         p = Path(file_path_str)
#         # Only load from actual solution files, not challenge files or sample submission
#         if "solution" in p.name.lower(): # Match logic from run_all for solution files
#             data, _ = load_json_if_file(p)
#             if data is None:
#                 logger.warning(f"Could not load data from {p}. Skipping.")
#                 continue
#             if isinstance(data, dict):
#                 train_solutions.update(data)
#             elif isinstance(data, list): # If it's a list, it's typically a list of task objects
#                 for item in data:
#                     if not isinstance(item, dict):
#                         continue
#                     tid = item.get("id") or item.get("task_id") or item.get("name")
#                     if tid:
#                         out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                         train_solutions[tid] = out
#     print(f"Reconstructed train_solutions with {len(train_solutions)} tasks.")
# else:
#     print(f"Train snapshot not found at {train_snapshot_path}. Cannot reconstruct true solutions.")
# 
# # Determine the task_id to compare
# task_id = None
# if 'benchmarking_results' in report_for_output and 'task_details' in report_for_output['benchmarking_results']:
#     if len(report_for_output['benchmarking_results']['task_details']) > 0:
#         for tid_candidate, details in report_for_output['benchmarking_results']['task_details'].items():
#             if tid_candidate in train_solutions and get_solution_grids_for_task(tid_candidate, train_solutions):
#                 task_id = tid_candidate
#                 break
#         if task_id:
#             print(f"Using valid task ID for comparison from benchmarking_results: {task_id}")
#         else:
#             print("No tasks with valid true solutions found in benchmarking_results. Falling back to '00d62c1b'.")
#             task_id = '00d62c1b' # Fallback
#     else:
#         print("No tasks in benchmarking_results. Falling back to '00d62c1b'.")
#         task_id = '00d62c1b' # Fallback
# else:
#     print("Benchmarking results not found in report. Falling back to '00d62c1b'.")
#     task_id = '00d62c1b' # Fallback
# 
# 
# # --- Detailed Analysis for the selected task_id ---
# print(f"\n--- Detailed Analysis for Task ID: {task_id} ---\n")
# 
# predicted_output_raw = None
# if 'tasks' in report_for_output and task_id in report_for_output['tasks']:
#     task_report = report_for_output['tasks'].get(task_id)
#     if task_report and 'final_attempt' in task_report and 'attempt_1' in task_report['final_attempt']:
#         # Use get_solution_grids_for_task to correctly extract the grid from the 'attempt_1' content
#         extracted_grids = get_solution_grids_for_task(task_id, {task_id: task_report['final_attempt']['attempt_1']})
#         if extracted_grids and len(extracted_grids) > 0:
#             predicted_output_raw = extracted_grids[0] # Assuming we are interested in the first test case's output
#         else:
#             logger.warning(f"Task {task_id}: Could not extract a valid predicted grid from 'attempt_1' content.")
#     else:
#         logger.warning(f"Task {task_id}: 'final_attempt' or 'attempt_1' not found in report.")
# else:
#     logger.warning(f"Task {task_id} not found in execution report.")
# 
# true_solution_raw = None
# if train_solutions and task_id in train_solutions:
#     sol_grids = get_solution_grids_for_task(task_id, train_solutions)
#     if sol_grids:
#         true_solution_raw = sol_grids[0] # Assuming we are interested in the first test case
# 
# def get_element_type(grid_data):
#     if not isinstance(grid_data, list) or not grid_data:
#         return "N/A"
#     first_element = grid_data[0]
#     if isinstance(first_element, list) and first_element:
#         return type(first_element[0]).__name__
#     return type(first_element).__name__
# 
# 
# print("=== Predicted Grid Details ===")
# if predicted_output_raw:
#     print(f"Raw content: {json.dumps(predicted_output_raw, ensure_ascii=False)}")
#     print(f"Type: {type(predicted_output_raw).__name__}")
#     if isinstance(predicted_output_raw, list) and predicted_output_raw:
#         print(f"Element type (first level): {type(predicted_output_raw[0]).__name__}")
#         if isinstance(predicted_output_raw[0], list) and predicted_output_raw[0]:
#             print(f"Element type (second level): {type(predicted_output_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(predicted_output_raw)}")
#     print(f"Grid Key: {grid_key(predicted_output_raw)}")
# else:
#     print("Predicted grid is None or empty.")
# 
# print("\n=== True Solution Grid Details ===")
# if true_solution_raw:
#     print(f"Raw content: {json.dumps(true_solution_raw, ensure_ascii=False)}")
#     print(f"Type: {type(true_solution_raw).__name__}")
#     if isinstance(true_solution_raw, list) and true_solution_raw:
#         print(f"Element type (first level): {type(true_solution_raw[0]).__name__}")
#         if isinstance(true_solution_raw[0], list) and true_solution_raw[0]:
#             print(f"Element type (second level): {type(true_solution_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(true_solution_raw)}")
#     print(f"Grid Key: {grid_key(true_solution_raw)}")
# else:
#     print("True solution grid is None or empty.")
# 
# print("\n=== Grid Key Comparison ===")
# if predicted_output_raw is not None and true_solution_raw is not None:
#     are_keys_identical = (grid_key(predicted_output_raw) == grid_key(true_solution_raw))
#     print(f"Grid Key values are identical: {are_keys_identical}")
#     if not are_keys_identical:
#         print("Reason for difference: The JSON representations (grid keys) of the predicted and true grids do not match. This could be due to differences in content, ordering (if not handled by sort_keys), or data types within the grids.")
# else:
#     print("Cannot compare grid keys: One or both grids are missing.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # Assuming logger is configured in a previous cell. If not, a minimal configuration might be needed.
# # For self-containment, adding a basic logger setup here if not already present.
# logger = logging.getLogger("arc_finalize")
# if not logger.handlers:
#     ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
#     logger.addHandler(ch)
# 
# # --- Helper functions (copied for self-containment in this debug cell) ---
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         if 'train' in sol and 'test' in sol: # Specific ARC training task format
#              out = []
#              for t_entry in sol['test']:
#                  if 'output' in t_entry:
#                      out.append(t_entry['output'])
#                  elif 'grid' in t_entry:
#                      out.append(t_entry['grid'])
#                  else:
#                      out.append(t_entry)
#              return out
#         return [v for v in sol.values()] # Fallback, original behavior
#     if isinstance(sol, list):
#         return sol
#     return [sol] # sol is a single grid
# # --- End of Helper functions ---
# 
# 
# # Define paths to the report and train snapshot files
# report_path = Path("/kaggle/working/execution_report.json")
# train_snapshot_path = Path("/kaggle/working/train_snapshot.json")
# 
# # Load the execution report
# report_for_output = {}
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report_for_output = json.load(f)
#     print("Execution report loaded successfully.")
# else:
#     print(f"Error: {report_path} not found.")
# 
# # Load the train snapshot and reconstruct train_solutions
# train_solutions = {}
# if train_snapshot_path.exists():
#     with open(train_snapshot_path, 'r', encoding='utf-8') as f:
#         train_snapshot = json.load(f)
#     print(f"Train snapshot loaded. Found {train_snapshot.get('tasks', 0)} tasks.")
# 
#     # Reconstruct train_solutions from the files listed in the snapshot
#     for file_path_str in train_snapshot.get("files", []):
#         p = Path(file_path_str)
#         # Only load from actual solution files, not challenge files or sample submission
#         if "solution" in p.name.lower(): # Match logic from run_all for solution files
#             data, _ = load_json_if_file(p)
#             if data is None:
#                 logger.warning(f"Could not load data from {p}. Skipping.")
#                 continue
#             if isinstance(data, dict):
#                 train_solutions.update(data)
#             elif isinstance(data, list): # If it's a list, it's typically a list of task objects
#                 for item in data:
#                     if not isinstance(item, dict):
#                         continue
#                     tid = item.get("id") or item.get("task_id") or item.get("name")
#                     if tid:
#                         out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                         train_solutions[tid] = out
#     print(f"Reconstructed train_solutions with {len(train_solutions)} tasks.")
# else:
#     print(f"Train snapshot not found at {train_snapshot_path}. Cannot reconstruct true solutions.")
# 
# # Determine the task_id to compare
# task_id = None
# if 'benchmarking_results' in report_for_output and 'task_details' in report_for_output['benchmarking_results']:
#     if len(report_for_output['benchmarking_results']['task_details']) > 0:
#         for tid_candidate, details in report_for_output['benchmarking_results']['task_details'].items():
#             if tid_candidate in train_solutions and get_solution_grids_for_task(tid_candidate, train_solutions):
#                 task_id = tid_candidate
#                 break
#         if task_id:
#             print(f"Using valid task ID for comparison from benchmarking_results: {task_id}")
#         else:
#             print("No tasks with valid true solutions found in benchmarking_results. Falling back to '00d62c1b'.")
#             task_id = '00d62c1b' # Fallback
#     else:
#         print("No tasks in benchmarking_results. Falling back to '00d62c1b'.")
#         task_id = '00d62c1b' # Fallback
# else:
#     print("Benchmarking results not found in report. Falling back to '00d62c1b'.")
#     task_id = '00d62c1b' # Fallback
# 
# 
# # --- Detailed Analysis for the selected task_id ---
# print(f"\n--- Detailed Analysis for Task ID: {task_id} ---\n")
# 
# predicted_output_raw = None
# if 'tasks' in report_for_output and task_id in report_for_output['tasks']:
#     task_report = report_for_output['tasks'].get(task_id)
#     if task_report and 'final_attempt' in task_report and 'attempt_1' in task_report['final_attempt']:
#         # Use get_solution_grids_for_task to correctly extract the grid from the 'attempt_1' content
#         extracted_grids = get_solution_grids_for_task(task_id, {task_id: task_report['final_attempt']['attempt_1']})
#         if extracted_grids and len(extracted_grids) > 0:
#             predicted_output_raw = extracted_grids[0] # Assuming we are interested in the first test case's output
#         else:
#             logger.warning(f"Task {task_id}: Could not extract a valid predicted grid from 'attempt_1' content.")
#     else:
#         logger.warning(f"Task {task_id}: 'final_attempt' or 'attempt_1' not found in report.")
# else:
#     logger.warning(f"Task {task_id} not found in execution report.")
# 
# true_solution_raw = None
# if train_solutions and task_id in train_solutions:
#     sol_grids = get_solution_grids_for_task(task_id, train_solutions)
#     if sol_grids:
#         true_solution_raw = sol_grids[0] # Assuming we are interested in the first test case
# 
# def get_element_type(grid_data):
#     if not isinstance(grid_data, list) or not grid_data:
#         return "N/A"
#     first_element = grid_data[0]
#     if isinstance(first_element, list) and first_element:
#         return type(first_element[0]).__name__
#     return type(first_element).__name__
# 
# 
# print("=== Predicted Grid Details ===")
# if predicted_output_raw:
#     print(f"Raw content: {json.dumps(predicted_output_raw, ensure_ascii=False)}")
#     print(f"Type: {type(predicted_output_raw).__name__}")
#     if isinstance(predicted_output_raw, list) and predicted_output_raw:
#         print(f"Element type (first level): {type(predicted_output_raw[0]).__name__}")
#         if isinstance(predicted_output_raw[0], list) and predicted_output_raw[0]:
#             print(f"Element type (second level): {type(predicted_output_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(predicted_output_raw)}")
#     print(f"Grid Key: {grid_key(predicted_output_raw)}")
# else:
#     print("Predicted grid is None or empty.")
# 
# print("\n=== True Solution Grid Details ===")
# if true_solution_raw:
#     print(f"Raw content: {json.dumps(true_solution_raw, ensure_ascii=False)}")
#     print(f"Type: {type(true_solution_raw).__name__}")
#     if isinstance(true_solution_raw, list) and true_solution_raw:
#         print(f"Element type (first level): {type(true_solution_raw[0]).__name__}")
#         if isinstance(true_solution_raw[0], list) and true_solution_raw[0]:
#             print(f"Element type (second level): {type(true_solution_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(true_solution_raw)}")
#     print(f"Grid Key: {grid_key(true_solution_raw)}")
# else:
#     print("True solution grid is None or empty.")
# 
# print("\n=== Grid Key Comparison ===")
# if predicted_output_raw is not None and true_solution_raw is not None:
#     are_keys_identical = (grid_key(predicted_output_raw) == grid_key(true_solution_raw))
#     print(f"Grid Key values are identical: {are_keys_identical}")
#     if not are_keys_identical:
#         print("Reason for difference: The JSON representations (grid keys) of the predicted and true grids do not match. This could be due to differences in content, ordering (if not handled by sort_keys), or data types within the grids.")
# else:
#     print("Cannot compare grid keys: One or both grids are missing.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # Assuming logger is configured in a previous cell. If not, a minimal configuration might be needed.
# # For self-containment, adding a basic logger setup here if not already present.
# logger = logging.getLogger("arc_finalize")
# if not logger.handlers:
#     ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
#     logger.addHandler(ch)
# 
# # --- Helper functions (copied for self-containment in this debug cell) ---
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         if 'train' in sol and 'test' in sol: # Specific ARC training task format
#              out = []
#              for t_entry in sol['test']:
#                  if 'output' in t_entry:
#                      out.append(t_entry['output'])
#                  elif 'grid' in t_entry:
#                      out.append(t_entry['grid'])
#                  else:
#                      out.append(t_entry)
#              return out
#         return [v for v in sol.values()] # Fallback, original behavior
#     if isinstance(sol, list):
#         return sol
#     return [sol] # sol is a single grid
# # --- End of Helper functions ---
# 
# 
# # Define paths to the report and train snapshot files
# report_path = Path("/kaggle/working/execution_report.json")
# train_snapshot_path = Path("/kaggle/working/train_snapshot.json")
# 
# # Load the execution report
# report_for_output = {}
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report_for_output = json.load(f)
#     print("Execution report loaded successfully.")
# else:
#     print(f"Error: {report_path} not found.")
# 
# # Load the train snapshot and reconstruct train_solutions
# train_solutions = {}
# if train_snapshot_path.exists():
#     with open(train_snapshot_path, 'r', encoding='utf-8') as f:
#         train_snapshot = json.load(f)
#     print(f"Train snapshot loaded. Found {train_snapshot.get('tasks', 0)} tasks.")
# 
#     # Reconstruct train_solutions from the files listed in the snapshot
#     for file_path_str in train_snapshot.get("files", []):
#         p = Path(file_path_str)
#         # Only load from actual solution files, not challenge files or sample submission
#         if "solution" in p.name.lower(): # Match logic from run_all for solution files
#             data, _ = load_json_if_file(p)
#             if data is None:
#                 logger.warning(f"Could not load data from {p}. Skipping.")
#                 continue
#             if isinstance(data, dict):
#                 train_solutions.update(data)
#             elif isinstance(data, list): # If it's a list, it's typically a list of task objects
#                 for item in data:
#                     if not isinstance(item, dict):
#                         continue
#                     tid = item.get("id") or item.get("task_id") or item.get("name")
#                     if tid:
#                         out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                         train_solutions[tid] = out
#     print(f"Reconstructed train_solutions with {len(train_solutions)} tasks.")
# else:
#     print(f"Train snapshot not found at {train_snapshot_path}. Cannot reconstruct true solutions.")
# 
# # Determine the task_id to compare
# task_id = None
# if 'benchmarking_results' in report_for_output and 'task_details' in report_for_output['benchmarking_results']:
#     if len(report_for_output['benchmarking_results']['task_details']) > 0:
#         for tid_candidate, details in report_for_output['benchmarking_results']['task_details'].items():
#             if tid_candidate in train_solutions and get_solution_grids_for_task(tid_candidate, train_solutions):
#                 task_id = tid_candidate
#                 break
#         if task_id:
#             print(f"Using valid task ID for comparison from benchmarking_results: {task_id}")
#         else:
#             print("No tasks with valid true solutions found in benchmarking_results. Falling back to '00d62c1b'.")
#             task_id = '00d62c1b' # Fallback
#     else:
#         print("No tasks in benchmarking_results. Falling back to '00d62c1b'.")
#         task_id = '00d62c1b' # Fallback
# else:
#     print("Benchmarking results not found in report. Falling back to '00d62c1b'.")
#     task_id = '00d62c1b' # Fallback
# 
# 
# # --- Detailed Analysis for the selected task_id ---
# print(f"\n--- Detailed Analysis for Task ID: {task_id} ---\n")
# 
# predicted_output_raw = None
# if 'tasks' in report_for_output and task_id in report_for_output['tasks']:
#     task_report = report_for_output['tasks'].get(task_id)
#     if task_report and 'final_attempt' in task_report and 'attempt_1' in task_report['final_attempt']:
#         # Use get_solution_grids_for_task to correctly extract the grid from the 'attempt_1' content
#         extracted_grids = get_solution_grids_for_task(task_id, {task_id: task_report['final_attempt']['attempt_1']})
#         if extracted_grids and len(extracted_grids) > 0:
#             predicted_output_raw = extracted_grids[0] # Assuming we are interested in the first test case's output
#         else:
#             logger.warning(f"Task {task_id}: Could not extract a valid predicted grid from 'attempt_1' content.")
#     else:
#         logger.warning(f"Task {task_id}: 'final_attempt' or 'attempt_1' not found in report.")
# else:
#     logger.warning(f"Task {task_id} not found in execution report.")
# 
# true_solution_raw = None
# if train_solutions and task_id in train_solutions:
#     sol_grids = get_solution_grids_for_task(task_id, train_solutions)
#     if sol_grids:
#         true_solution_raw = sol_grids[0] # Assuming we are interested in the first test case
# 
# def get_element_type(grid_data):
#     if not isinstance(grid_data, list) or not grid_data:
#         return "N/A"
#     first_element = grid_data[0]
#     if isinstance(first_element, list) and first_element:
#         return type(first_element[0]).__name__
#     return type(first_element).__name__
# 
# 
# print("=== Predicted Grid Details ===")
# if predicted_output_raw:
#     print(f"Raw content: {json.dumps(predicted_output_raw, ensure_ascii=False)}")
#     print(f"Type: {type(predicted_output_raw).__name__}")
#     if isinstance(predicted_output_raw, list) and predicted_output_raw:
#         print(f"Element type (first level): {type(predicted_output_raw[0]).__name__}")
#         if isinstance(predicted_output_raw[0], list) and predicted_output_raw[0]:
#             print(f"Element type (second level): {type(predicted_output_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(predicted_output_raw)}")
#     print(f"Grid Key: {grid_key(predicted_output_raw)}")
# else:
#     print("Predicted grid is None or empty.")
# 
# print("\n=== True Solution Grid Details ===")
# if true_solution_raw:
#     print(f"Raw content: {json.dumps(true_solution_raw, ensure_ascii=False)}")
#     print(f"Type: {type(true_solution_raw).__name__}")
#     if isinstance(true_solution_raw, list) and true_solution_raw:
#         print(f"Element type (first level): {type(true_solution_raw[0]).__name__}")
#         if isinstance(true_solution_raw[0], list) and true_solution_raw[0]:
#             print(f"Element type (second level): {type(true_solution_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(true_solution_raw)}")
#     print(f"Grid Key: {grid_key(true_solution_raw)}")
# else:
#     print("True solution grid is None or empty.")
# 
# print("\n=== Grid Key Comparison ===")
# if predicted_output_raw is not None and true_solution_raw is not None:
#     are_keys_identical = (grid_key(predicted_output_raw) == grid_key(true_solution_raw))
#     print(f"Grid Key values are identical: {are_keys_identical}")
#     if not are_keys_identical:
#         print("Reason for difference: The JSON representations (grid keys) of the predicted and true grids do not match. This could be due to differences in content, ordering (if not handled by sort_keys), or data types within the grids.")
# else:
#     print("Cannot compare grid keys: One or both grids are missing.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import json, os, sys, logging, math

# import json, os, sys, logging, math
# from pathlib import Path
# from collections import defaultdict
# from typing import Iterable, Tuple
# 
# # Assuming logger is configured in a previous cell. If not, a minimal configuration might be needed.
# # For self-containment, adding a basic logger setup here if not already present.
# logger = logging.getLogger("arc_finalize")
# if not logger.handlers:
#     ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
#     logger.addHandler(ch)
# 
# # --- Helper functions (copied for self-containment in this debug cell) ---
# def load_json_if_file(p: Path):
#     try:
#         if p.is_file():
#             txt = p.read_text(encoding="utf-8")
#             return json.loads(txt), p
#     except Exception as e:
#         logger.debug("Failed parsing %s: %s", p, e)
#     return None, None
# 
# def grid_key(g):
#     try:
#         return json.dumps(g, sort_keys=True, ensure_ascii=False)
#     except Exception:
#         return repr(g)
# 
# def get_solution_grids_for_task(tid, train_solutions):
#     sol = train_solutions.get(tid)
#     if sol is None:
#         return None
#     if isinstance(sol, dict):
#         if "test" in sol and isinstance(sol["test"], list):
#             out = []
#             for t in sol["test"]:
#                 if isinstance(t, dict):
#                     out.append(t.get("output") or t.get("grid") or t)
#                 else:
#                     out.append(t)
#             return out
#         if "output" in sol:
#             return sol["output"] if isinstance(sol["output"], list) else [sol["output"]]
#         if 'train' in sol and 'test' in sol: # Specific ARC training task format
#              out = []
#              for t_entry in sol['test']:
#                  if 'output' in t_entry:
#                      out.append(t_entry['output'])
#                  elif 'grid' in t_entry:
#                      out.append(t_entry['grid'])
#                  else:
#                      out.append(t_entry) # Fallback if test entry is just the grid
#              return out
#         return [v for v in sol.values()] # Fallback, original behavior
#     if isinstance(sol, list):
#         return sol
#     return [sol] # sol is a single grid
# # --- End of Helper functions ---
# 
# 
# # Define paths to the report and train snapshot files
# report_path = Path("/kaggle/working/execution_report.json")
# train_snapshot_path = Path("/kaggle/working/train_snapshot.json")
# 
# # Load the execution report
# report_for_output = {}
# if report_path.exists():
#     with open(report_path, 'r', encoding='utf-8') as f:
#         report_for_output = json.load(f)
#     print("Execution report loaded successfully.")
# else:
#     print(f"Error: {report_path} not found.")
# 
# # Load the train snapshot and reconstruct train_solutions
# train_solutions = {}
# if train_snapshot_path.exists():
#     with open(train_snapshot_path, 'r', encoding='utf-8') as f:
#         train_snapshot = json.load(f)
#     print(f"Train snapshot loaded. Found {train_snapshot.get('tasks', 0)} tasks.")
# 
#     # Reconstruct train_solutions from the files listed in the snapshot
#     for file_path_str in train_snapshot.get("files", []):
#         p = Path(file_path_str)
#         # Only load from actual solution files, not challenge files or sample submission
#         if "solution" in p.name.lower(): # Match logic from run_all for solution files
#             data, _ = load_json_if_file(p)
#             if data is None:
#                 logger.warning(f"Could not load data from {p}. Skipping.")
#                 continue
#             if isinstance(data, dict):
#                 train_solutions.update(data)
#             elif isinstance(data, list): # If it's a list, it's typically a list of task objects
#                 for item in data:
#                     if not isinstance(item, dict):
#                         continue
#                     tid = item.get("id") or item.get("task_id") or item.get("name")
#                     if tid:
#                         out = item.get("output") or item.get("solution") or item.get("solutions") or item.get("test") or item
#                         train_solutions[tid] = out
#     print(f"Reconstructed train_solutions with {len(train_solutions)} tasks.")
# else:
#     print(f"Train snapshot not found at {train_snapshot_path}. Cannot reconstruct true solutions.")
# 
# # Determine the task_id to compare
# task_id = None
# if 'benchmarking_results' in report_for_output and 'task_details' in report_for_output['benchmarking_results']:
#     if len(report_for_output['benchmarking_results']['task_details']) > 0:
#         for tid_candidate, details in report_for_output['benchmarking_results']['task_details'].items():
#             if tid_candidate in train_solutions and get_solution_grids_for_task(tid_candidate, train_solutions):
#                 task_id = tid_candidate
#                 break
#         if task_id:
#             print(f"Using valid task ID for comparison from benchmarking_results: {task_id}")
#         else:
#             print("No tasks with valid true solutions found in benchmarking_results. Falling back to '00d62c1b'.")
#             task_id = '00d62c1b' # Fallback
#     else:
#         print("No tasks in benchmarking_results. Falling back to '00d62c1b'.")
#         task_id = '00d62c1b' # Fallback
# else:
#     print("Benchmarking results not found in report. Falling back to '00d62c1b'.")
#     task_id = '00d62c1b' # Fallback
# 
# 
# # --- Detailed Analysis for the selected task_id ---
# print(f"\n--- Detailed Analysis for Task ID: {task_id} ---\n")
# 
# predicted_output_raw = None
# if 'tasks' in report_for_output and task_id in report_for_output['tasks']:
#     task_report = report_for_output['tasks'].get(task_id)
#     if task_report and 'final_attempt' in task_report and 'attempt_1' in task_report['final_attempt']:
#         predicted_content = task_report['final_attempt']['attempt_1']
#         # If the content is a full task object, extract its first test's output
#         if isinstance(predicted_content, dict) and 'test' in predicted_content and isinstance(predicted_content['test'], list) and len(predicted_content['test']) > 0:
#             first_test_entry = predicted_content['test'][0]
#             if isinstance(first_test_entry, dict) and 'output' in first_test_entry:
#                 predicted_output_raw = first_test_entry['output']
#             elif isinstance(first_test_entry, dict) and 'grid' in first_test_entry:
#                 predicted_output_raw = first_test_entry['grid']
#             else:
#                 # Fallback if first_test_case is just the grid itself (unlikely in this context)
#                 predicted_output_raw = first_test_entry
#         # If it's directly a grid, assign it
#         elif isinstance(predicted_content, list) and all(isinstance(row, list) for row in predicted_content):
#             predicted_output_raw = predicted_content
#         else:
#             logger.warning(f"Task {task_id}: 'attempt_1' content has unexpected structure: {type(predicted_content)}")
#     else:
#         logger.warning(f"Task {task_id}: 'final_attempt' or 'attempt_1' not found in report.")
# else:
#     logger.warning(f"Task {task_id} not found in execution report.")
# 
# true_solution_raw = None
# if train_solutions and task_id in train_solutions:
#     sol_grids = get_solution_grids_for_task(task_id, train_solutions)
#     if sol_grids:
#         true_solution_raw = sol_grids[0] # Assuming we are interested in the first test case
# 
# def get_element_type(grid_data):
#     if not isinstance(grid_data, list) or not grid_data:
#         return "N/A"
#     first_element = grid_data[0]
#     if isinstance(first_element, list) and first_element:
#         return type(first_element[0]).__name__
#     return type(first_element).__name__
# 
# 
# print("=== Predicted Grid Details ===")
# if predicted_output_raw:
#     print(f"Raw content: {json.dumps(predicted_output_raw, ensure_ascii=False)}")
#     print(f"Type: {type(predicted_output_raw).__name__}")
#     if isinstance(predicted_output_raw, list) and predicted_output_raw:
#         print(f"Element type (first level): {type(predicted_output_raw[0]).__name__}")
#         if isinstance(predicted_output_raw[0], list) and predicted_output_raw[0]:
#             print(f"Element type (second level): {type(predicted_output_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(predicted_output_raw)}")
#     print(f"Grid Key: {grid_key(predicted_output_raw)}")
# else:
#     print("Predicted grid is None or empty.")
# 
# print("\n=== True Solution Grid Details ===")
# if true_solution_raw:
#     print(f"Raw content: {json.dumps(true_solution_raw, ensure_ascii=False)}")
#     print(f"Type: {type(true_solution_raw).__name__}")
#     if isinstance(true_solution_raw, list) and true_solution_raw:
#         print(f"Element type (first level): {type(true_solution_raw[0]).__name__}")
#         if isinstance(true_solution_raw[0], list) and true_solution_raw[0]:
#             print(f"Element type (second level): {type(true_solution_raw[0][0]).__name__}")
#     else:
#         print(f"Element type: {get_element_type(true_solution_raw)}")
#     print(f"Grid Key: {grid_key(true_solution_raw)}")
# else:
#     print("True solution grid is None or empty.")
# 
# print("\n=== Grid Key Comparison ===")
# if predicted_output_raw is not None and true_solution_raw is not None:
#     are_keys_identical = (grid_key(predicted_output_raw) == grid_key(true_solution_raw))
#     print(f"Grid Key values are identical: {are_keys_identical}")
#     if not are_keys_identical:
#         print("Reason for difference: The JSON representations (grid keys) of the predicted and true grids do not match. This could be due to differences in content, ordering (if not handled by sort_keys), or data types within the grids.")
# else:
#     print("Cannot compare grid keys: One or both grids are missing.")


# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: import numpy as np

# import numpy as np
# 
# def print_grid(g, label="Grid"):
#     # Print part of grid if large
#     max_rows = 8
#     print(f"{label} ({len(g)} x {len(g[0])}):")
#     for idx, row in enumerate(g):
#         if idx < max_rows:
#             print(row)
#         else:
#             print("...")
#             break
# 
# def count_diff(a, b):
#     rows, cols = len(b), len(b[0])
#     return sum(a[r][c] != b[r][c] for r in range(rows) for c in range(cols))
# 
# # Defensive conversion for input_grid
# print("input_grid type:", type(predicted_output_raw))
# if isinstance(predicted_output_raw, dict):
#     # Try first value or key 'input' or key 'output'
#     keys = list(predicted_output_raw.keys())
#     print("input_grid dict keys:", keys)
#     grid_candidate = None
#     for k in ['input', 'output', 0]:
#         if k in predicted_output_raw:
#             grid_candidate = predicted_output_raw[k]
#             break
#     if not grid_candidate:
#         grid_candidate = list(predicted_output_raw.values())[0]
#     input_grid = grid_candidate
# else:
#     input_grid = predicted_output_raw
# 
# if not (isinstance(input_grid, list) and input_grid and isinstance(input_grid[0], list)):
#     raise ValueError("input_grid is not a list of lists. Type: {}".format(type(input_grid)))
# 
# target = true_solution_raw
# if not (isinstance(target, list) and target and isinstance(target[0], list)):
#     raise ValueError("target is not a list of lists. Type: {}".format(type(target)))
# 
# rows, cols = len(target), len(target[0])
# input_rows, input_cols = len(input_grid), len(input_grid[0])
# 
# print("\nStrategy 1: All subgrids of target size")
# # Try all subgrids of target size within input
# subgrids = []
# for i in range(input_rows - rows + 1):
#     for j in range(input_cols - cols + 1):
#         candidate = [input_grid[i + r][j:j + cols] for r in range(rows)]
#         diff = count_diff(candidate, target)
#         subgrids.append(('subgrid', diff, (i,j), candidate))
# subgrids.sort(key=lambda x: x[1])
# for key,diff,pos,cand in subgrids[:2]:
#     print(f"\n{key} starting at {pos}, diff: {diff}")
#     print_grid(cand)
# 
# def flip_grid(grid, mode):
#     if mode == 'h':
#         return [row[::-1] for row in grid]
#     elif mode == 'v':
#         return grid[::-1]
#     return grid
# 
# print("\nStrategy 2: Flips")
# for mode in ['h','v']:
#     fg = flip_grid(target, mode)
#     print(f"\n{mode}-flipped target:")
#     print_grid(fg)
#     for i in range(input_rows - rows + 1):
#         for j in range(input_cols - cols + 1):
#             candidate = [input_grid[i + r][j:j + cols] for r in range(rows)]
#             diff = count_diff(candidate, fg)
#             if diff == 0:
#                 print(f"Match with {mode}-flipped region at {i},{j}")
# 
# def rotate_grid(grid, k):
#     arr = np.array(grid)
#     rotated = np.rot90(arr, k=k).tolist()
#     return rotated
# 
# print("\nStrategy 3: Rotations")
# for k in range(1,4):
#     rot = rotate_grid(target, k)
#     print(f"\nTarget rotated {90*k} deg:")
#     print_grid(rot)
#     for i in range(input_rows - len(rot) + 1):
#         for j in range(input_cols - len(rot[0]) + 1):
#             candidate = [input_grid[i+r][j:j+len(rot[0])] for r in range(len(rot))]
#             diff = count_diff(candidate, rot)
#             if diff == 0:
#                 print(f"Match with rotation {90*k} deg at {i},{j}")
# 
# input_flat = [v for row in input_grid for v in row]
# target_flat = [v for row in target for v in row]
# input_colors = sorted(set(input_flat))
# target_colors = sorted(set(target_flat))
# if len(input_colors) == len(target_colors):
#     from itertools import permutations
#     print("\nStrategy 4: Color remap")
#     for perm in permutations(input_colors):
#         mapping = dict(zip(perm, target_colors))
#         remapped = [[mapping[v] for v in row] for row in input_grid]
#         if remapped[:rows] == target:
#             print("\nFull grid color-remap match with mapping:", mapping)
#             print_grid(remapped[:rows])
#         for i in range(input_rows - rows + 1):
#             for j in range(input_cols - cols + 1):
#                 candidate = [ [mapping[v] for v in input_grid[i+r][j:j+cols]] for r in range(rows)]
#                 if candidate == target:
#                     print(f"\nRemap subgrid match at ({i},{j}) using {mapping}")
#                     print_grid(candidate)
# 
# print("\nStrategy 5: Transposed (swapped axes)")
# transposed = [list(x) for x in zip(*input_grid)]
# if len(transposed) >= rows and len(transposed[0]) >= cols:
#     for i in range(len(transposed) - rows + 1):
#         for j in range(len(transposed[0]) - cols + 1):
#             candidate = [transposed[i + r][j:j + cols] for r in range(rows)]
#             if candidate == target:
#                 print(f"Found transposed-region match at ({i},{j})")
#                 print_grid(candidate)
# 
# print("\nStrategy 6: Padding/cropping if nearly correct shape")
# for dr in [-1, 0, 1]:
#     for dc in [-1, 0, 1]:
#         rr = rows + dr
#         cc = cols + dc
#         if rr > 0 and cc > 0 and input_rows >= rr and input_cols >= cc:
#             candidate = [input_grid[r][:cc] for r in range(rr)]
#             if len(candidate) == rows and all(len(c)==cols for c in candidate):
#                 diff = count_diff(candidate, target)
#                 if diff == 0:
#                     print(f"Found crop/pad match at rows:{rr}, cols:{cc}")
#                     print_grid(candidate)
# 
# print("\n=== True Solution Grid ===")
# print_grid(target, label="Target Grid")






# REMOVED FOR SUBMISSION: developer/eval-only cell
# This cell was commented out to produce a safe submission notebook.
# If you need it for debugging, open the original notebook.

# Original first non-empty line: # Orientation-normalization utility (add to evaluation/debugging pipeline)

# # Orientation-normalization utility (add to evaluation/debugging pipeline)
# # This cell adds dihedral transforms enumeration and a normalize_orientation() helper.
# import numpy as np
# from typing import Dict, Any
# 
# def _to_np(grid):
#     \"\"\"Accept list-of-lists or ndarray; return ndarray of ints.\"\"\"
#     arr = np.array(grid, dtype=int)
#     return arr
# 
# def generate_dihedral_transforms(grid: np.ndarray):
#     \"\"\"Yield (name, transformed_grid) for the 8 dihedral transforms:
#     rotations by 0/90/180/270 and each rotated version optionally flipped left-right.
#     \"\"\"
#     for k in range(4):
#         r = np.rot90(grid, k=k)              # rotation by 90*k degrees CCW
#         yield f"rot_{k*90}", r
#         yield f"rot_{k*90}_flip_lr", np.fliplr(r)
# 
# def mismatch_score(a: np.ndarray, b: np.ndarray) -> int:
#     \"\"\"Simple mismatch count (number of cells not equal).\"\"\"
#     if a.shape != b.shape:
#         return 10**9
#     return int(np.sum(a != b))
# 
# def normalize_orientation(pred_grid, true_grid) -> Dict[str, Any]:
#     \"\"\"Return the best transformed prediction that minimizes mismatch with true_grid.
#     Returns dict with keys: best_pred (ndarray), best_name (str), best_score (int), all_scores (list).
#     \"\"\"
#     p = _to_np(pred_grid)
#     t = _to_np(true_grid)
# 
#     best = None
#     best_name = None
#     best_score = 10**12
#     scores = []
# 
#     for name, cand in generate_dihedral_transforms(p):
#         if cand.shape != t.shape:
#             score = 10**9
#         else:
#             score = mismatch_score(cand, t)
#         scores.append((name, score))
#         if score < best_score:
#             best_score = score
#             best = cand.copy()
#             best_name = name
# 
#     return {
#         'best_pred': best,
#         'best_name': best_name,
#         'best_score': best_score,
#         'all_scores': scores
#     }
# 
# # Quick usage example (for dev/eval only):
# # res = normalize_orientation(predicted_grid, true_grid)
# # normalized = res['best_pred']
# # print(res['best_name'], res['best_score'])


# CLEAN submission writer - place this as the LAST runnable cell in the notebook.
import json
from pathlib import Path
import inspect

out_path = Path("/kaggle/working/submission.json")

# Helper discovery: try to find a test task loader and a predict function in globals
loader_candidates = ["list_test_tasks", "load_test_tasks", "get_test_tasks", "iter_test_tasks"]
predict_candidates = ["predict_one", "predict", "solve_task", "model_predict"]

globals_obj = globals()

def find_callable(names):
    for n in names:
        obj = globals_obj.get(n)
        if callable(obj):
            return n, obj
    return None, None

loader_name, loader_fn = find_callable(loader_candidates)
predict_name, predict_fn = find_callable(predict_candidates)

if loader_fn is None:
    raise RuntimeError("No test-data loader found. Implement one of: " + ", ".join(loader_candidates) + 
                       " that returns an iterable of (task_id, test_inputs).")

if predict_fn is None:
    raise RuntimeError("No prediction function found. Implement one of: " + ", ".join(predict_candidates) + 
                       " that accepts (task_id, test_input) and returns a grid (list-of-lists of ints).")

# Run deterministic inference over test set
submission = {}
count = 0
for task_id, test_inputs in loader_fn():
    attempts = []
    for idx, test_input in enumerate(test_inputs):
        pred_grid = predict_fn(task_id, test_input)
        attempts.append({f"attempt_{idx+1}": pred_grid})
    submission[task_id] = attempts
    count += 1

# Write compact JSON
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(submission, separators=(",", ":"), ensure_ascii=False))
print(f"Wrote submission for {count} tasks to: {out_path}")

