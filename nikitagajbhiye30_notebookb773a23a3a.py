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


# ================================
# 1. Imports
# ================================
import os, json, zipfile
from pathlib import Path

# Root path where JSON tasks are mounted
TASKS_PATH = Path("/kaggle/input/google-code-golf-2025")

# Where we'll save Python solutions
OUTPUT_DIR = Path("/kaggle/working/solutions")
OUTPUT_DIR.mkdir(exist_ok=True)

# ================================
# 2. Helper to load a task
# ================================
def load_task(task_id):
    """Load a taskXXX.json file as dict"""
    fname = TASKS_PATH / f"task{task_id:03d}.json"
    with open(fname) as f:
        return json.load(f)

# Example: load task001
task = load_task(1)
print("Keys:", task.keys())
print("Train examples:", len(task["train"]))
print("First train input:", task["train"][0]["input"])
print("First train output:", task["train"][0]["output"])

# ================================
# 3. Write your solution here
# Replace with the actual logic
# ================================
def p(g):
    """
    Example solution function for task001.
    You MUST edit per-task logic manually.
    g: input grid (list of lists of ints)
    return: output grid
    """
    # Example: identity transformation
    return [row[:] for row in g]

# ================================
# 4. Test the solution
# ================================
def test_solution(task, fn):
    """Check if fn(g) == expected for all train/test examples"""
    for split in ["train", "test"]:
        for i, pair in enumerate(task[split]):
            inp, out = pair["input"], pair["output"]
            pred = fn([row[:] for row in inp])  # copy to avoid mutation
            if pred != out:
                print(f"❌ Mismatch in {split} example {i}")
                return False
    print("✅ All examples passed")
    return True

# Validate current solution on task001
ok = test_solution(task, p)

# ================================
# 5. Save solution as taskXXX.py
# ================================
def save_solution(task_id, fn_src):
    """Save function source as taskXXX.py"""
    out_file = OUTPUT_DIR / f"task{task_id:03d}.py"
    with open(out_file, "w") as f:
        f.write(fn_src)
    print("Saved", out_file)

# Example: saving identity solution for task001
fn_src = """def p(g):\n return [row[:] for row in g]\n"""
save_solution(1, fn_src)

# ================================
# 6. Package all into submission.zip
# ================================
def make_submission():
    zip_path = Path("/kaggle/working/submission.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for py_file in sorted(OUTPUT_DIR.glob("task*.py")):
            z.write(py_file, py_file.name)
    print("Created", zip_path)

# Package current solutions
make_submission()



import json

# Load task001.json
with open("/kaggle/input/google-code-golf-2025/task001.json") as f:
    task001 = json.load(f)

print("Train examples:", len(task001["train"]))
for i, pair in enumerate(task001["train"]):
    print(f"\n--- Train Example {i} ---")
    print("Input:")
    for row in pair["input"]:
        print(row)
    print("Output:")
    for row in pair["output"]:
        print(row)



def p(g):
 h=len(g);z=[[0]*h for _ in g];o=[]
 for r in g:
  for i in range(h):
   o.append([])
   for c in r:o[-1]+=g[i]if c else z[i]
 return o



fn_src = """def p(g):
 h=len(g);z=[[0]*h for _ in g];o=[]
 for r in g:
  for i in range(h):
   o.append([])
   for c in r:o[-1]+=g[i]if c else z[i]
 return o
"""
save_solution(1, fn_src)



task = load_task(1)

from solutions.task001 import p   # load your saved solution
test_solution(task, p)



make_submission()



def p(g):h=len(g);z=[[0]*h for _ in g];o=[];[o.append([c and g[i]or z[i]for c in r for _ in[0]][0]*0)or None for r in g for i in range(h)];return o



import os, json, zipfile
from pathlib import Path

TASKS_PATH = Path("/kaggle/input/google-code-golf-2025")
OUTPUT_DIR = Path("/kaggle/working/solutions")
OUTPUT_DIR.mkdir(exist_ok=True)

# Helper: load one task
def load_task(task_id):
    fname = TASKS_PATH / f"task{task_id:03d}.json"
    with open(fname) as f:
        return json.load(f)

# Example: identity solver (placeholder)
def identity_solver():
    return "def p(g):return[g[:]for g in g]"

# Save a solution
def save_solution(task_id, fn_src):
    out_file = OUTPUT_DIR / f"task{task_id:03d}.py"
    with open(out_file, "w") as f:
        f.write(fn_src)

# Loop over all 400 tasks
for tid in range(1, 401):
    # TODO: Replace identity_solver() with real logic for each task
    save_solution(tid, identity_solver())

# Package everything
def make_submission():
    zip_path = Path("/kaggle/working/submission.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for py_file in sorted(OUTPUT_DIR.glob("task*.py")):
            z.write(py_file, py_file.name)
    print("Created", zip_path)

make_submission()



# Auto-solver framework for Google Code Golf 2025 (ARC-style tasks)
# Place into a Kaggle notebook and run. Adjust heuristics if you want.

import json, zipfile, math
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy

TASKS_PATH = Path("/kaggle/input/google-code-golf-2025")
OUT_DIR = Path("/kaggle/working/solutions")
OUT_DIR.mkdir(exist_ok=True)
ZIP_PATH = Path("/kaggle/working/submission.zip")

# Utility grid helpers
def h_w(g): return len(g), len(g[0])
def rot90(g): return [list(row) for row in zip(*g[::-1])]
def rot180(g): return [row[::-1] for row in g[::-1]]
def rot270(g): return [list(row) for row in zip(*g)][::-1]
def flip_h(g): return [row[::-1] for row in g]
def flip_v(g): return g[::-1]
def transpose(g): return [list(row) for row in zip(*g)]
def copy_g(g): return [row[:] for row in g]
def zeros(h,w): return [[0]*w for _ in range(h)]

# NN scale (repeat pixels)
def scale_nn(g, k):
    h,w = h_w(g)
    out = []
    for r in g:
        for _ in range(k):
            row = []
            for c in r:
                row += [c]*k
            out.append(row)
    return out

# block-tiling pattern: for each input cell, place block (block_src) if predicate(cell) else zero-block
def block_tile_by_cell(g, block_src, pred):
    ih,iw = h_w(g)
    bh,bw = h_w(block_src)
    out=[]
    for bi in range(ih):
        for sub_i in range(bh):
            row=[]
            for bj in range(iw):
                if pred(g[bi][bj]):
                    row += block_src[sub_i][:]
                else:
                    row += [0]*bw
            out.append(row)
    return out

# color map remap
def remap_colors(g, cmap):
    return [[cmap.get(c,c) for c in row] for row in g]

# equality test
def same(a,b):
    if a is None or b is None: return False
    if len(a)!=len(b) or len(a[0])!=len(b[0]): return False
    return all(a[i]==b[i] for i in range(len(a)))

# Load a task dict
def load_task(task_path):
    with open(task_path) as f:
        return json.load(f)

# Candidate solver generators (each returns function f(g) -> grid)
def cand_identity(): return lambda g: copy_g(g)
def cand_rot90(): return lambda g: rot90(g)
def cand_rot180(): return lambda g: rot180(g)
def cand_rot270(): return lambda g: rot270(g)
def cand_flip_h(): return lambda g: flip_h(g)
def cand_flip_v(): return lambda g: flip_v(g)
def cand_transpose(): return lambda g: transpose(g)

# color remap guesser: if output is same shape but colors permuted
def cand_color_remap(train_pairs):
    # infer mapping from first example where both shapes equal
    for pair in train_pairs:
        A,B = pair['input'], pair['output']
        if h_w(A) == h_w(B):
            # mapping from A->B where A cell nonzero maybe maps
            mapping = {}
            ok=True
            for i in range(len(A)):
                for j in range(len(A[0])):
                    a,b = A[i][j], B[i][j]
                    if a in mapping and mapping[a]!=b: ok=False
                    mapping[a]=b
            if ok:
                return lambda g, mapping=mapping: remap_colors(g,mapping)
    return None

# scale (nearest neighbor) candidate if output is integer multiple of input
def cand_scale_if_matches(train_pairs):
    # try to infer k from first pair where sizes are multiples
    for pair in train_pairs:
        A,B = pair['input'], pair['output']
        ah,aw = h_w(A); bh,bw = h_w(B)
        if bh%ah==0 and bw%aw==0:
            kh,kw = bh//ah, bw//aw
            if kh==kw and kh>1:
                k=kh
                return lambda g,k=k: scale_nn(g,k)
    return None

# block-tiling heuristics: common pattern in ARC tasks (like task001)
def cand_block_tile_variants(train_pairs):
    """
    Detect pattern where output size = (ih*bh) x (iw*bw) and blocks correspond to input cells.
    Try variants: block_src = input or block_src = rotated input; predicate: cell!=0 or cell==targetcolor.
    """
    for pair in train_pairs:
        A,B = pair['input'], pair['output']
        ih,iw=h_w(A); bh = len(B)//ih if ih and len(B)%ih==0 else None
        if bh is None: continue
        bw = len(B[0])//iw if iw and len(B[0])%iw==0 else None
        if bw is None: continue
        # candidate block size
        # Extract blocks
        bh=int(bh); bw=int(bw)
        # Try to reconstruct mapping of blocks to whether block equals some reference
        blocks = {}
        ok=True
        # collect unique block patterns by block position
        for bi in range(ih):
            for bj in range(iw):
                # extract block at that position
                top = bi*bh; left = bj*bw
                block = [B[top+r][left:left+bw] for r in range(bh)]
                blocks[(bi,bj)] = block
        # try if any block equals input grid (maybe scaled or not)
        # block_src candidates: A, rot90(A), rot180(A), rot270(A), transpose(A)
        block_srcs = [A, rot90(A), rot180(A), rot270(A), transpose(A)]
        # also try zero block
        zero_block = [[0]*bw for _ in range(bh)]
        for src in block_srcs:
            sh,sw=h_w(src)
            if sh==bh and sw==bw:
                # check if for each (bi,bj), block is either src or zero
                ok2=True
                target_cells=[]
                for bi in range(ih):
                    for bj in range(iw):
                        block = blocks[(bi,bj)]
                        if block==src:
                            target_cells.append(1)
                        elif block==zero_block:
                            target_cells.append(0)
                        else:
                            ok2=False; break
                    if not ok2: break
                if ok2:
                    # We can return solver: for input cell nonzero -> place src block else zero_block
                    def solver(g, src=src):
                        return block_tile_by_cell(g, src, lambda v: v!=0)
                    return solver
        # Another variant: block equals a single color tile formed by input cell value
        # e.g., if block is all equal to some color equal to A[bi][bj]
        ok3=True
        for bi in range(ih):
            for bj in range(iw):
                block = blocks[(bi,bj)]
                vals=set(x for r in block for x in r)
                if len(vals)==1:
                    continue
                else:
                    ok3=False; break
            if not ok3: break
        if ok3:
            # build solver: each output block filled with color equal to input cell value
            def solver(g):
                ih,iw=h_w(g)
                bh,bw = bh,bw
                out=[]
                for bi in range(ih):
                    for sub_i in range(bh):
                        row=[]
                        for bj in range(iw):
                            color = g[bi][bj]
                            row += [color]*bw
                        out.append(row)
                return out
            return solver
    return None

# try heuristics in an ordered list
def find_solver_for_task(task):
    train = task.get("train", [])
    # if no train examples, skip
    if not train: return None, "no-train"
    # prepare pairs
    pairs = [{"input":p["input"], "output":p["output"]} for p in train]
    # candidate list (some are parameterized)
    cands = []
    # basic transforms (same-size)
    cands += [cand_identity(), cand_rot90(), cand_rot180(), cand_rot270(), cand_flip_h(), cand_flip_v(), cand_transpose()]
    # color remap
    cr = cand_color_remap(pairs)
    if cr: cands.append(cr)
    # scale
    sc = cand_scale_if_matches(pairs)
    if sc: cands.append(sc)
    # block tile variants
    bt = cand_block_tile_variants(pairs)
    if bt: cands.append(bt)
    # Also try: place input at positions where cell==target color (like earlier more general)
    # general block tile with block_src = input (if block size equals input)
    def block_src_input_if_matches(pairs):
        for pair in pairs:
            A,B = pair['input'], pair['output']
            ih,iw=h_w(A); bh = len(B)//ih if ih and len(B)%ih==0 else None
            if bh is None: continue
            bw = len(B[0])//iw if iw and len(B[0])%iw==0 else None
            if bw is None: continue
            if bh==len(A) and bw==len(A[0]):
                return lambda g: block_tile_by_cell(g, copy_g(g), lambda v: v!=0)
        return None
    gblock = block_src_input_if_matches(pairs)
    if gblock: cands.append(gblock)

    # For each candidate, test against all train examples
    for fn in cands:
        ok=True
        for p in pairs:
            inp, out = p['input'], p['output']
            try:
                pred = fn([row[:] for row in inp])
            except Exception as e:
                ok=False; break
            if not same(pred, out):
                ok=False; break
        if ok:
            return fn, "auto"
    return None, "none"

# Convert a solver function to Python source string to save
def solver_to_src(fn, task_id):
    """
    We will try to match known solver patterns and return a compact source.
    For unknown callables, we fallback to a wrapper that calls a precomputed mapping for training examples only.
    """
    # match simple names by comparing to known functions' behavior on a sample grid
    # For safety, we'll generate explicit source based on identity, rotations, flips, scale, block-tile patterns detected in find_solver_for_task (we returned those solvers).
    # We'll inspect fn by calling it on a small sample (size 3 identity) and comparing to patterns.
    # But simpler: we will store a tiny wrapper that embeds the solver logic as code we know for the common cases.
    # We'll determine type by applying fn to a canonical grid and comparing to transformations.
    g3 = [[1,2,3],[4,5,6],[7,8,9]]
    out=fn(copy_g(g3))
    # compare to known transforms
    if same(out, g3): return "def p(g):\n return [row[:] for row in g]\n"
    if same(out, rot90(g3)): return "def p(g):\n return [list(r) for r in zip(*g[::-1])]\n"
    if same(out, rot180(g3)): return "def p(g):\n return [row[::-1] for row in g[::-1]]\n"
    if same(out, rot270(g3)): return "def p(g):\n return [list(r) for r in zip(*g)][::-1]\n"
    if same(out, flip_h(g3)): return "def p(g):\n return [row[::-1] for row in g]\n"
    if same(out, flip_v(g3)): return "def p(g):\n return g[::-1]\n"
    # scale (detect if size increased)
    ih,iw=h_w(g3); oh,ow=h_w(out)
    if oh%ih==0 and ow%iw==0 and oh//ih==ow//iw>1:
        k=oh//ih
        return f"def p(g):\n h=len(g);w=len(g[0]);o=[]\n for r in g:\n  for _ in range({k}):\n   row=[]\n   for c in r: row += [c]*{k}\n   o.append(row)\n return o\n"
    # block tiling pattern (common): detect if out size is ih*bh etc and blocks equal either input or zero-block
    # We will implement a safe generic block-tiler that uses input as block source and tests cell!=0
    # This matches many tasks like task001.
    # Provide compact code:
    return "def p(g):\n h=len(g);z=[[0]*h for _ in g];o=[]\n for r in g:\n  for i in range(h):\n   o.append([])\n   for c in r:o[-1]+=g[i] if c else z[i]\n return o\n"

# Main loop: iterate over JSON files in input dir
task_files = sorted([p for p in TASKS_PATH.glob("task*.json")])
solved = []
unsolved = []
for tfile in task_files:
    tid = int(tfile.stem.replace("task",""))
    task = load_task(tfile)
    fn, status = find_solver_for_task(task)
    if fn is not None:
        src = solver_to_src(fn, tid)
        outpath = OUT_DIR / f"task{tid:03d}.py"
        with open(outpath,"w") as f:
            f.write("# Auto-generated by heuristic solver\n")
            f.write(src)
        solved.append(tid)
        print(f"Task {tid:03d} solved automatically ({status})")
    else:
        # fallback: write identity (safe placeholder) so submission contains file
        outpath = OUT_DIR / f"task{tid:03d}.py"
        with open(outpath,"w") as f:
            f.write("# Placeholder: no auto-solution found; identity fallback\n")
            f.write("def p(g):\n return [row[:] for row in g]\n")
        unsolved.append(tid)
        print(f"Task {tid:03d} NOT solved automatically")

# Package submission zip
with zipfile.ZipFile(ZIP_PATH, "w") as z:
    for pyfile in sorted(OUT_DIR.glob("task*.py")):
        z.write(pyfile, pyfile.name)
print("Submission zip created at:", ZIP_PATH)
print("Solved count:", len(solved), "Unsolved count:", len(unsolved))
print("Solved tasks (sample):", solved[:50])


