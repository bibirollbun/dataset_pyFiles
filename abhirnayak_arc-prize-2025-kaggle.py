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


# ULTIMATE HYBRID ARC-AGI SOLVER
# Paste this whole cell into a Kaggle Notebook and run (GPU runtime).
# Requirements:
#  - Kaggle kernel with GPU (T4x2 recommended)
#  - Input datasets in ../input/arc-prize-2025 as provided by the competition
#  - (Optional but recommended) DarkAGICCompress input at /kaggle/input/darkagicompressarc

# ---------------------------
# 1) Environment + imports
# ---------------------------
import os, sys, time, json, random, math, traceback
from copy import deepcopy
from collections import Counter, defaultdict
import multiprocessing
from multiprocessing import Process, Manager, Queue

import numpy as np
import torch

# SKLEARN (Kaggle provides sklearn)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Set deterministic seeds & GPU settings
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.set_float32_matmul_precision('high')  # newer PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

# Kaggle ARC paths
DATA_DIR = '/kaggle/input/arc-prize-2025'
TEST_PATH = os.path.join(DATA_DIR, 'arc-agi_test_challenges.json')
EVAL_PATH = os.path.join(DATA_DIR, 'arc-agi_evaluation_challenges.json')
TRAIN_PATH = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.json')
OUT_PATH = '/kaggle/working/submission.json'
MODEL_DIR = '/kaggle/working/arc_models'
os.makedirs(MODEL_DIR, exist_ok=True)
DARK_PATH = '/kaggle/input/ARaChnida'

# ---------------------------
# 2) Utility I/O helpers
# ---------------------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# ---------------------------
# 3) Grid utilities & primitives (from your code, tightened)
# ---------------------------
def grid_to_np(grid):
    return np.array(grid, dtype=np.int64)

def np_to_grid(arr):
    return arr.astype(np.int64).tolist()

def sanitize_grid(grid):
    try:
        arr = grid_to_np(grid)
        if arr.ndim != 2 or arr.size == 0: return [[0]]
        arr = np.clip(arr, 0, 9)
        return arr.tolist()
    except Exception:
        return [[0]]

def equal_grids(a, b):
    try:
        return np.array_equal(grid_to_np(a), grid_to_np(b))
    except Exception:
        return False

def most_common_color(grid):
    try:
        a = grid_to_np(grid).flatten()
        if a.size == 0: return 0
        return int(Counter(a.tolist()).most_common(1)[0][0])
    except Exception:
        return 0

def extract_components(grid, connectivity=4):
    arr = grid_to_np(grid)
    h,w = arr.shape
    visited = np.zeros((h,w), dtype=bool)
    comps = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity==4 else [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for y in range(h):
        for x in range(w):
            if visited[y,x]: continue
            color = int(arr[y,x])
            q=[(y,x)]; visited[y,x]=True; pts=[]
            for cy,cx in q:
                pts.append((cy,cx))
                for dy,dx in dirs:
                    ny,nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and (not visited[ny,nx]) and arr[ny,nx]==color:
                        visited[ny,nx]=True; q.append((ny,nx))
            ys=[p[0] for p in pts]; xs=[p[1] for p in pts]
            bbox=(min(ys), max(ys), min(xs), max(xs))
            comps.append({'color':color, 'pts':pts, 'bbox':bbox, 'size':len(pts)})
    return comps

def detect_symmetry_full(grid):
    arr = grid_to_np(grid)
    flags = {'vertical': False, 'horizontal': False, 'rot180': False, 'diag': False}
    try:
        if np.array_equal(arr, np.fliplr(arr)): flags['vertical'] = True
        if np.array_equal(arr, np.flipud(arr)): flags['horizontal'] = True
        if np.array_equal(arr, np.rot90(arr,2)): flags['rot180'] = True
        if arr.shape[0]==arr.shape[1] and np.array_equal(arr, arr.T): flags['diag'] = True
    except Exception:
        pass
    return flags

def detect_repetition(grid, max_block=6):
    arr = grid_to_np(grid)
    h,w = arr.shape
    for th in range(1, min(max_block,h)+1):
        if h % th != 0: continue
        for tw in range(1, min(max_block,w)+1):
            if w % tw != 0: continue
            block = arr[:th, :tw]
            tiled = np.tile(block, (h//th, w//tw))
            if np.array_equal(tiled, arr): return True, (th,tw)
    return False, None

# ---------- primitives ----------
def prim_relabel(grid, mapping):
    arr = grid_to_np(grid).copy()
    for a,b in mapping.items():
        arr[arr==int(a)] = int(b)
    return np_to_grid(arr)

def prim_translate(grid, dy, dx, canvas_shape=None, fill=0):
    arr = grid_to_np(grid); h,w = arr.shape
    H,W = (h,w) if canvas_shape is None else canvas_shape
    out = np.full((H,W), fill, dtype=arr.dtype)
    for y in range(h):
        for x in range(w):
            ny, nx = y+dy, x+dx
            if 0<=ny<H and 0<=nx<W:
                out[ny,nx] = arr[y,x]
    return np_to_grid(out)

def prim_tile(grid, ty, tx):
    return np_to_grid(np.tile(grid_to_np(grid), (ty,tx)))

def prim_mirror(grid, axis):
    arr = grid_to_np(grid)
    if axis in ('v','vertical'): return np_to_grid(np.fliplr(arr))
    if axis in ('h','horizontal'): return np_to_grid(np.flipud(arr))
    if axis in ('diag','transpose'): return np_to_grid(arr.T)
    return np_to_grid(arr)

def prim_fill_border_of_largest(grid, color):
    arr = grid_to_np(grid).copy()
    comps = extract_components(grid)
    if not comps: return np_to_grid(arr)
    largest = max(comps, key=lambda c:c['size'])
    y0,y1,x0,x1 = largest['bbox']
    arr[y0, x0:x1+1] = int(color)
    arr[y1, x0:x1+1] = int(color)
    arr[y0:y1+1, x0] = int(color)
    arr[y0:y1+1, x1] = int(color)
    return np_to_grid(arr)

def prim_color_cycle(grid, offset, preserve_zero=True):
    arr = grid_to_np(grid).copy()
    unique = np.unique(arr)
    if unique.size==0: return np_to_grid(arr)
    maxcol = int(unique.max()); mod = maxcol + 1
    if mod <= 1: return np_to_grid(arr)
    out = arr.copy()
    for c in unique:
        if preserve_zero and int(c)==0: continue
        out[arr==c] = ((int(c) + offset) % mod)
    return np_to_grid(out)

def prim_move_component_to_bbox(grid, color, target_bbox, canvas_shape=None, fill=0):
    arr = grid_to_np(grid)
    H,W = arr.shape
    if canvas_shape is None: canvas_shape = (H,W)
    comps = extract_components(grid)
    found = None
    for c in comps:
        if c['color'] == color:
            found = c; break
    if found is None: return np_to_grid(arr)
    y0,y1,x0,x1 = found['bbox']
    comp = arr[y0:y1+1, x0:x1+1]
    ch,cw = comp.shape
    out = np.full(canvas_shape, fill, dtype=arr.dtype)
    ay,ax = target_bbox[0], target_bbox[2]
    for y in range(ch):
        for x in range(cw):
            by = ay+y; bx = ax+x
            if 0<=by<canvas_shape[0] and 0<=bx<canvas_shape[1]:
                out[by,bx] = comp[y,x]
    return np_to_grid(out)

# ---------------------------
# 4) Program DSL
# ---------------------------
class Program:
    def __init__(self, ops=None):
        self.ops = ops or []

    def append(self, op, params=None):
        self.ops.append((op, params or {}))

    def copy(self):
        return Program([(op, deepcopy(params)) for op,params in self.ops])

    def run(self, grid, library=None):
        cur = deepcopy(grid)
        for op, params in self.ops:
            try:
                if op == 'relabel': cur = prim_relabel(cur, params['mapping'])
                elif op == 'rotate': cur = prim_mirror(cur, 'diag') if params.get('k',0)==0 else np_to_grid(np.rot90(grid_to_np(cur), params['k']))
                elif op == 'flip': cur = prim_mirror(cur, params['axis'])
                elif op == 'translate': cur = prim_translate(cur, params['dy'], params['dx'], canvas_shape=params.get('canvas'), fill=params.get('fill',0))
                elif op == 'fill_border': cur = prim_fill_border_of_largest(cur, params['color'])
                elif op == 'tile': cur = prim_tile(cur, params['ty'], params['tx'])
                elif op == 'mirror': cur = prim_mirror(cur, params['axis'])
                elif op == 'color_cycle': cur = prim_color_cycle(cur, params.get('offset',1))
                elif op == 'move_comp_bbox': cur = prim_move_component_to_bbox(cur, params['color'], params['target_bbox'], canvas_shape=params.get('canvas'), fill=params.get('fill',0))
                elif op == 'call_lib':
                    idx = params['idx']
                    if library and 0 <= idx < len(library): cur = library[idx].run(cur, library=library)
                    else: return None
                elif op == 'overlay':
                    cur = prim_translate(cur, 0,0)  # no-op placeholder (overlay not heavily used)
                else:
                    return None
            except Exception:
                return None
            if cur is None: return None
        return cur

    def __repr__(self):
        return 'Program(' + ' -> '.join([f"{op}:{params}" for op,params in self.ops]) + ')'

# ---------------------------
# 5) Partial scoring / caching
# ---------------------------
_eval_cache = {}
def grid_signature(arr):
    try:
        a = grid_to_np(arr)
        return (a.shape, int(a.sum()) ^ int((a.astype(np.int64)**2).sum() & 0xFFFFFFFF))
    except Exception:
        return None

def cached_run(prog, inp, library):
    try:
        sig = (repr(prog), grid_signature(inp))
    except Exception:
        sig = (repr(prog), None)
    if sig in _eval_cache: return _eval_cache[sig]
    out = prog.run(inp, library=library)
    _eval_cache[sig] = out
    return out

def partial_match_score(a,b):
    try:
        aa = grid_to_np(a); bb = grid_to_np(b)
    except Exception:
        return 0.0
    if aa.shape != bb.shape:
        H = min(aa.shape[0], bb.shape[0]); W = min(aa.shape[1], bb.shape[1])
        if H == 0 or W == 0: return 0.0
        return float((aa[:H,:W] == bb[:H,:W]).sum()) / (H*W)
    return float((aa==bb).sum()) / aa.size

def partial_score_program(prog, train_pairs, library):
    if prog is None: return -1.0
    s=0.0
    for pair in train_pairs:
        out = cached_run(prog, pair['input'], library)
        if out is None: return -1.0
        s += partial_match_score(out, pair['output'])
    return s

def test_program_exact(prog, train_pairs, library):
    if prog is None: return False
    for pair in train_pairs:
        out = cached_run(prog, pair['input'], library)
        if out is None or not equal_grids(out, pair['output']): return False
    return True

# ---------------------------
# 6) Rule miner + templates
# ---------------------------
def rule_miner(train_pairs):
    if not train_pairs: return None
    tin = grid_to_np(train_pairs[0]['input']); tout = grid_to_np(train_pairs[0]['output'])
    # exact relabel
    if tin.shape == tout.shape:
        mapping_counts = defaultdict(Counter)
        for y in range(tin.shape[0]):
            for x in range(tin.shape[1]):
                mapping_counts[int(tin[y,x])][int(tout[y,x])] += 1
        final = {}
        for a,cnt in mapping_counts.items():
            final[a] = cnt.most_common(1)[0][0]
        cand = prim_relabel(tin.tolist(), final)
        if equal_grids(cand, tout.tolist()):
            p=Program(); p.append('relabel', {'mapping':final}); return p
    # uniform output
    uniq_out = np.unique(tout)
    if uniq_out.size == 1:
        p = Program(); p.append('tile', {'ty':1,'tx':1}); p.append('relabel', {'mapping':{most_common_color(tin): int(uniq_out[0])}}); return p
    # translate single component
    in_comps = extract_components(tin); out_comps = extract_components(tout)
    if len(in_comps) == 1 and len(out_comps) == 1:
        ci = in_comps[0]; co = out_comps[0]
        dy = co['bbox'][0] - ci['bbox'][0]; dx = co['bbox'][2] - ci['bbox'][2]
        p = Program(); p.append('translate', {'dy':dy,'dx':dx,'canvas':tout.shape}); return p
    # symmetry
    sym = detect_symmetry_full(tout)
    if sym['vertical']:
        p=Program(); p.append('mirror', {'axis':'v'}); return p
    if sym['horizontal']:
        p=Program(); p.append('mirror', {'axis':'h'}); return p
    # fill border
    try:
        if np.array_equal(grid_to_np(prim_fill_border_of_largest(tin.tolist(), most_common_color(tout))), tout):
            p=Program(); p.append('fill_border', {'color':most_common_color(tout)}); return p
    except Exception:
        pass
    return None

# ---------------------------
# 7) Strong initial seeds + MCTS
# ---------------------------
def propose_color_mappings(train_in, train_out, top_k=3):
    ia = grid_to_np(train_in); oa = grid_to_np(train_out)
    H = min(ia.shape[0], oa.shape[0]); W = min(ia.shape[1], oa.shape[1])
    mapping_counts = defaultdict(Counter)
    for y in range(H):
        for x in range(W):
            mapping_counts[int(ia[y,x])][int(oa[y,x])] += 1
    if not mapping_counts: return [{}]
    base = {a: counter.most_common(1)[0][0] for a,counter in mapping_counts.items()}
    candidates = [base]
    for a,counter in mapping_counts.items():
        for b,_ in counter.most_common(top_k):
            nm = dict(base); nm[a]=b; candidates.append(nm)
    uniq=[] 
    for m in candidates:
        if m not in uniq: uniq.append(m)
    return uniq[:30]

def propose_translations(train_in, train_out, max_shift=3):
    ia = grid_to_np(train_in); oa = grid_to_np(train_out)
    H,O = oa.shape[0], oa.shape[1]
    candidates=[]
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            matched = 0
            for y in range(ia.shape[0]):
                for x in range(ia.shape[1]):
                    ny, nx = y+dy, x+dx
                    if 0<=ny<H and 0<=nx<O and ia[y,x] == oa[ny,nx]: matched += 1
            if matched>0: candidates.append((dy,dx))
    if not candidates: candidates=[(0,0)]
    uniq=[]
    for c in candidates:
        if c not in uniq: uniq.append(c)
    return uniq[:40]

def propose_bboxes(grid, max_candidates=40):
    comps = extract_components(grid)
    bbs = [c['bbox'] for c in comps]
    H,W = grid_to_np(grid).shape
    proposals=[]
    for (y0,y1,x0,x1) in bbs:
        for pad in [0,1,2,3]:
            ny0 = max(0,y0-pad); ny1 = min(H-1,y1+pad); nx0 = max(0,x0-pad); nx1 = min(W-1,x1+pad)
            proposals.append((ny0,ny1,nx0,nx1))
    proposals.append((0,H-1,0,W-1))
    uniq=[]
    for p in proposals:
        if p not in uniq: uniq.append(p)
    return uniq[:max_candidates]

def strong_initial_seeds(train_pairs, library):
    seeds=[]
    if not train_pairs: return seeds
    tin = grid_to_np(train_pairs[0]['input']); tout = grid_to_np(train_pairs[0]['output'])
    for m in propose_color_mappings(tin,tout,top_k=3):
        p=Program(); p.append('relabel', {'mapping':m}); seeds.append(p)
    colors = [c for c in np.unique(tin)]
    for c in colors[:4]:
        pts = np.argwhere(tin==c); pts_out = np.argwhere(tout==c)
        if pts.size and pts_out.size:
            ci = pts.mean(axis=0); co = pts_out.mean(axis=0)
            dy,dx = int(round(co[0]-ci[0])), int(round(co[1]-ci[1]))
            p=Program(); p.append('translate', {'dy':dy,'dx':dx,'canvas':tout.shape}); seeds.append(p)
    for k in [1,2,3]:
        p=Program(); p.append('rotate', {'k':k}); seeds.append(p)
    for ax in ['h','v']:
        p=Program(); p.append('flip', {'axis':ax}); seeds.append(p)
    for bb in propose_bboxes(tout, max_candidates=6):
        p=Program(); p.append('tile', {'ty':1, 'tx':1}); p.append('fill_border', {'color':most_common_color(tout)})
        seeds.append(p)
    rep, block = detect_repetition(tin)
    if rep and block:
        th,tw = block; p=Program(); p.append('tile', {'ty':tin.shape[0]//th, 'tx':tin.shape[1]//tw}); seeds.append(p)
    flags = detect_symmetry_full(tout)
    if flags['vertical']: p=Program(); p.append('mirror', {'axis':'v'}); seeds.append(p)
    if flags['horizontal']: p=Program(); p.append('mirror', {'axis':'h'}); seeds.append(p)
    if library:
        for i,lp in enumerate(library[:6]):
            p=Program(); p.append('call_lib', {'idx':i}); seeds.append(p)
    # de-dup
    uniq=set(); out=[]
    for s in seeds:
        k = repr(s)
        if k not in uniq: uniq.add(k); out.append(s)
    return out[:220]

# MCTS node & search (lightweight)
class MCTSNode:
    def __init__(self, program, parent=None, prior=1.0):
        self.program = program; self.parent = parent; self.children = []; self.visits = 0; self.value = 0.0; self.prior = prior; self.expanded=False
    def q(self): return self.value/self.visits if self.visits>0 else 0.0
    def ucb(self, c_puct, parent_visits): return self.q() + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)

def prog_novelty(prog):
    ops = [op for op,_ in prog.ops]
    return len(set(ops)) + 0.02*len(ops)

def select_leaf(node, c_puct, novelty_weight=0.06):
    cur = node
    while cur.expanded and cur.children:
        parent_visits = cur.visits if cur.visits>0 else 1
        scores=[]
        for child in cur.children:
            base = child.ucb(c_puct, parent_visits)
            novelty = prog_novelty(child.program)
            scores.append(base + novelty_weight * novelty)
        idx = int(np.argmax(scores)); cur = cur.children[idx]
    return cur

def expand_candidates_for_mcts(prog, train_pairs, library, guidance=None, max_expand=60):
    if not train_pairs: return []
    sample_in = train_pairs[0]['input']; sample_out = train_pairs[0]['output']
    ops=[]
    for m in propose_color_mappings(sample_in, sample_out, top_k=2): ops.append(('relabel', {'mapping':m}))
    for dy,dx in propose_translations(sample_in, sample_out, max_shift=3)[:10]: ops.append(('translate', {'dy':dy,'dx':dx,'canvas':grid_to_np(sample_out).shape}))
    for k in [1,2,3]: ops.append(('rotate', {'k':k}))
    for ax in ['h','v']: ops.append(('flip', {'axis':ax}))
    for bb in propose_bboxes(sample_out, max_candidates=6): ops.append(('tile', {'ty':1,'tx':1}))
    ops.append(('color_cycle', {'offset':1}))
    # naive ranking: keep relabels & translations first
    ranked = ops
    expansions=[]
    for op_name, params in ranked[:max_expand]:
        npg = prog.copy(); npg.append(op_name, params); expansions.append(npg)
    return expansions

def rollout_policy_value(program, train_pairs, library, value_model=None, depth_limit=3):
    cur = program.copy()
    for _ in range(depth_limit):
        cands = expand_candidates_for_mcts(cur, train_pairs, library, max_expand=12)
        if not cands: break
        best=None; best_val=-1e9
        for c in cands:
            if SKLEARN_AVAILABLE and value_model and getattr(value_model,'model',None) is not None:
                val = value_model.predict(train_pairs, c)
            else:
                val = partial_score_program(c, train_pairs, library)
            if val > best_val:
                best_val = val; best = c
        if best is None: break
        cur = best
    if SKLEARN_AVAILABLE and value_model and getattr(value_model,'model',None) is not None:
        return value_model.predict(train_pairs, cur)
    return partial_score_program(cur, train_pairs, library)

def backprop(node, value):
    cur = node
    while cur is not None:
        cur.visits += 1; cur.value += value; cur = cur.parent

def best_prog_from_root(root):
    best=None; best_score=-1e9
    for child in root.children:
        sc = child.value/child.visits if child.visits>0 else child.prior
        if sc>best_score: best_score=sc; best=child.program
    return best

def search_mcts(train_pairs, library, guidance, value_model, time_limit=6.0, n_iters=300, c_puct=1.0, rollout_depth=3):
    start = time.time()
    root = MCTSNode(Program(), parent=None, prior=1.0)
    seeds = strong_initial_seeds(train_pairs, library)
    if seeds:
        for s in seeds[:12]:
            root.children.append(MCTSNode(s, parent=root, prior=1.0/len(seeds)))
    root.expanded = True
    best_prog=None; best_val=-1e9; it=0
    while time.time()-start < time_limit and it < n_iters:
        it += 1
        leaf = select_leaf(root, c_puct)
        if not leaf.expanded:
            exps = expand_candidates_for_mcts(leaf.program, train_pairs, library, max_expand=40)
            if not exps:
                leaf.expanded=True
            else:
                priors=[1.0]*len(exps); s=sum(priors)
                leaf.children = [MCTSNode(p, parent=leaf, prior=pr/s) for p,pr in zip(exps,priors)]
                leaf.expanded=True
        node_to_roll = leaf
        if leaf.expanded and leaf.children:
            node_to_roll = random.choice(leaf.children)
        val = rollout_policy_value(node_to_roll.program, train_pairs, library, value_model, depth_limit=rollout_depth)
        if val is None: val = -1.0
        backprop(node_to_roll, val)
        if val > best_val:
            best_val = val; best_prog = node_to_roll.program.copy()
        if best_prog is not None and test_program_exact(best_prog, train_pairs, library):
            return best_prog
    if best_prog is None:
        best_prog = best_prog_from_root(root)
    return best_prog

# ---------------------------
# 8) Library management & templates
# ---------------------------
LIB_PATH = os.path.join(MODEL_DIR, 'arc_library.pkl')
TEMPLATE_PATH = os.path.join(MODEL_DIR, 'arc_templates.pkl')

import pickle
def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_library(path=LIB_PATH):
    if os.path.exists(path):
        try: return load_pickle(path)
        except: return []
    return []

def save_library(lib, path=LIB_PATH):
    try: save_pickle(lib, path)
    except: pass

def add_prog_to_library(lib, prog, train_pairs):
    emb = task_embedding_from_train(train_pairs)
    entry = {'prog':prog, 'emb':emb}
    lib.insert(0, entry)
    if len(lib) > 800: lib.pop()
    save_library(lib)

def retrieve_similar_library(lib, train_pairs, top_k=6):
    if not lib: return []
    emb = task_embedding_from_train(train_pairs).astype(float)
    embs = np.vstack([entry['emb'].astype(float) for entry in lib])
    def norm(v): nv = v - v.mean(); s = np.linalg.norm(nv); return nv/(s+1e-9)
    embn = norm(emb); embsn = np.vstack([norm(e) for e in embs])
    sims = embsn.dot(embn); idx = np.argsort(-sims)[:top_k]
    return [lib[i]['prog'] for i in idx]

# ---------------------------
# 9) Task embedding (compact)
# ---------------------------
def task_embedding_from_train(train_pairs):
    if not train_pairs: return np.zeros(28, dtype=float)
    p = train_pairs[0]
    in_arr = grid_to_np(p['input']); out_arr = grid_to_np(p['output'])
    emb = [in_arr.shape[0], in_arr.shape[1], out_arr.shape[0], out_arr.shape[1]]
    hist = Counter(in_arr.flatten().tolist())
    for c in range(10): emb.append(hist.get(c,0))
    comps = extract_components(p['input']); sizes = sorted([c['size'] for c in comps], reverse=True)[:5]
    while len(sizes) < 5: sizes.append(0)
    emb.extend(sizes)
    emb = np.array(emb, dtype=float)
    mean = emb.mean(); norm = np.linalg.norm(emb-mean)
    if norm > 0: emb = (emb-mean)/norm
    return emb

# --------------------------- 
# 10) Heuristic fallback (fixed)
# --------------------------- 
def heuristic_fallback(train_pairs, inp):
    attempts = []
    
    # --- Rule miner based ---
    try:
        p = rule_miner(train_pairs)
        if p is not None:
            out = p.run(inp)
            if out is not None:
                attempts.append(out)
    except Exception:
        pass

    # --- Simple color mapping ---
    if train_pairs:
        try:
            maps = propose_color_mappings(train_pairs[0]['input'], train_pairs[0]['output'], top_k=2)
            for m in maps[:2]:
                try:
                    attempts.append(prim_relabel(inp, m))
                except Exception:
                    pass
        except Exception:
            pass

    # --- Symmetry based ---
    try:
        if any(detect_symmetry_full(train_pairs[0]['output']).values()):
            # try vertical mirror
            attempts.append(prim_mirror(inp, 'v'))
            # try horizontal mirror
            attempts.append(prim_mirror(inp, 'h'))
    except Exception:
        pass

    # --- Border fill ---
    try:
        attempts.append(prim_fill_border_of_largest(inp, most_common_color(inp)))
    except Exception as e:
        print(f"[WARN] prim_fill_border_of_largest failed: {e}")
        attempts.append(deepcopy(inp))

    # --- Color cycle ---
    try:
        attempts.append(prim_color_cycle(inp, 1))
    except Exception:
        pass

    # --- Always keep original as fallback ---
    attempts.append(deepcopy(inp))

    # --- Select two distinct grids ---
    a = attempts[0] if attempts else deepcopy(inp)
    b = attempts[1] if len(attempts) > 1 else deepcopy(inp)

    return sanitize_grid(a), sanitize_grid(b)


# ---------------------------
# 11) Solve single task (core)
# ---------------------------
def solve_task_single(task_id, task, lib_entries, guidance, value_model, time_budget, failure_log_local=None):
    train_pairs = task.get('train', []); tests = task.get('test', [])
    lib_programs = [e['prog'] for e in lib_entries] if lib_entries else []
    # 1) library exact reuse
    for lp in lib_programs:
        try:
            if test_program_exact(lp, train_pairs, lib_programs):
                outputs=[]
                for t in tests:
                    out1 = lp.run(t['input'], library=lib_programs)
                    h1,h2 = heuristic_fallback(train_pairs, t['input'])
                    outputs.append((out1 if out1 is not None else t['input'], h1))
                return outputs
        except Exception:
            pass
    # 2) rule miner
    rm = None
    try:
        rm = rule_miner(train_pairs)
    except Exception:
        rm = None
    if rm is not None:
        outputs=[]
        for t in tests:
            o = rm.run(t['input'])
            outputs.append((o if o is not None else t['input'], t['input']))
        try: add_prog_to_library(lib_entries, rm, train_pairs)
        except: pass
        return outputs
    # 3) repetition/template quick attempt
    try:
        rep, blk = detect_repetition(train_pairs[0]['input']) if train_pairs else (False,None)
        if rep:
            th,tw = blk
            p = Program(); p.append('tile', {'ty':train_pairs[0]['input'].shape[0]//th if isinstance(train_pairs[0]['input'], np.ndarray) else 1, 'tx':1})
            # crude
    except Exception:
        pass
    # 4) component match
    try:
        matches = []  # not implemented full match_components, quicker skip
    except Exception:
        matches = []
    # 5) heavy MCTS
    per_search = max(1.0, min(time_budget*0.6, 8.0))
    prog = None
    try:
        prog = search_mcts(train_pairs, lib_programs, guidance, value_model, time_limit=per_search, n_iters=400, c_puct=1.0, rollout_depth=3)
    except Exception:
        prog = None
    results=[]; failure_entry={}
    for t in tests:
        inp = t['input']; cand_outs=[]
        if prog is not None:
            try:
                o = prog.run(inp, library=lib_programs)
                if o is not None: cand_outs.append(o)
            except Exception:
                pass
        try:
            h1,h2 = heuristic_fallback(train_pairs, inp); cand_outs.append(h1); cand_outs.append(h2)
        except Exception:
            pass
        try:
            retrieved = retrieve_similar_library(lib_entries, train_pairs, top_k=4)
            for rp in retrieved:
                try:
                    o = rp.run(inp, library=lib_programs); cand_outs.append(o)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            cand_outs.append(prim_color_cycle(inp, 1))
            cand_outs.append(prim_fill_border_of_largest(inp, most_common_color(inp)))
            cand_outs.append(prim_mirror(inp, 'v') if False else deepcopy(inp))
        except Exception:
            pass
        cand_outs.append(deepcopy(inp))
        # choose two distinct outputs
        a = cand_outs[0] if cand_outs else inp
        b = cand_outs[1] if len(cand_outs)>1 else inp
        results.append((sanitize_grid(a), sanitize_grid(b)))
        failure_entry['candidates'] = len(cand_outs)
    try:
        if prog is not None and test_program_exact(prog, train_pairs, lib_programs):
            add_prog_to_library(lib_entries, prog, train_pairs)
    except Exception:
        pass
    if failure_log_local is not None:
        failure_log_local[str(task_id)] = {'prog_repr': repr(prog) if prog is not None else None, 'candidates_tried': failure_entry.get('candidates', 0)}
    return results

# ---------------------------
# 12) Top-level submission builder for symbolic-only fallback
# ---------------------------
def build_submission_symbolic_only(test_path, train_path=None, out_path=OUT_PATH, per_task_time=8.0):
    challenges = load_json(test_path)
    training_tasks = None
    if train_path and os.path.exists(train_path):
        try:
            training_tasks = load_json(train_path)
        except Exception:
            training_tasks = None
    lib_entries = load_library()
    guidance = None; value_model = None
    # seed templates using a small bank
    bank = [
        {'type':'relabel_all'},
        {'type':'relabel_majority'},
        {'type':'translate_centroid'},
        {'type':'tile_block'},
        {'type':'mirror_if_needed'},
        {'type':'move_component_to_border'},
        {'type':'fill_border_of_largest_outcolor'},
        {'type':'repeat_pattern_bbox'},
    ]
    # try seeding from training tasks
    if training_tasks:
        added=0
        for tid,task in list(training_tasks.items()):
            if added>400: break
            train_pairs = task.get('train', [])
            if not train_pairs: continue
            for tmpl in bank:
                # try template_instantiate quick version
                try:
                    # attempt relabel_majority & relabel_all
                    tin = grid_to_np(train_pairs[0]['input']); tout = grid_to_np(train_pairs[0]['output'])
                    if tmpl['type']=='relabel_majority':
                        incol = Counter(tin.flatten().tolist()).most_common(1)[0][0]
                        outcol = Counter(tout.flatten().tolist()).most_common(1)[0][0]
                        p=Program(); p.append('relabel', {'mapping':{incol:outcol}})
                        if test_program_exact(p, train_pairs, [e['prog'] for e in lib_entries] if lib_entries else []):
                            add_prog_to_library(lib_entries, p, train_pairs); added+=1; break
                except Exception:
                    pass
    # solve
    items = list(challenges.items())
    final_results = {}
    failure_log = {}
    total = len(items); count = 0
    print(f"Symbolic-only: Solving {total} tasks sequentially; per_task_time={per_task_time}s")
    for tid, task in items:
        tstart = time.time()
        try:
            preds = solve_task_single(tid, task, lib_entries, guidance, value_model, per_task_time, failure_log_local=failure_log)
        except Exception as e:
            traceback.print_exc()
            preds=[]
            for t in task.get('test', []):
                g = deepcopy(t['input']); preds.append((g,g))
        final_results[str(tid)] = preds
        count += 1
        print(f"[{count}/{total}] Task {tid} done in {time.time()-tstart:.2f}s")
    save_json(failure_log, os.path.join(MODEL_DIR, 'failure_log.json'))
    # format like sample submission if available
    if os.path.exists(SAMPLE_SUB):
        sample = load_json(SAMPLE_SUB)
        out = {}
        for tid, template in sample.items():
            if tid in final_results:
                preds = final_results[tid]
                entries=[]
                for i, templ_entry in enumerate(template):
                    if i < len(preds):
                        a1,a2 = preds[i]
                    else:
                        a1,a2 = [[0]], [[0]]
                    new_entry = {}
                    for k in templ_entry.keys():
                        if k.lower().startswith('attempt'):
                            if k.lower().endswith('1'): new_entry[k] = sanitize_grid(a1)
                            elif k.lower().endswith('2'): new_entry[k] = sanitize_grid(a2)
                            else:
                                new_entry[k] = sanitize_grid(a1)
                        else:
                            new_entry[k] = templ_entry[k]
                    entries.append(new_entry)
                out[tid] = entries
            else:
                out[tid] = template
        save_json(out, out_path)
        return out
    out = {}
    for tid, preds in final_results.items():
        entries=[]
        for a1,a2 in preds:
            entries.append({'attempt_1': sanitize_grid(a1), 'attempt_2': sanitize_grid(a2)})
        out[tid] = entries
    save_json(out, out_path)
    return out

# ---------------------------
# 13) GPU compressor integration & scheduler
# ---------------------------
# We'll import darkagicompress modules if present.
DARK_AVAILABLE = os.path.exists(DARK_PATH)
if DARK_AVAILABLE:
    sys.path.append(DARK_PATH)
    try:
        import solve_task as compressor_solve_task  # expected to provide solve_task.solve_task API
        print("[INFO] DarkAGICCompress modules loaded from:", DARK_PATH)
    except Exception as e:
        print("[WARN] Could not import DarkAGICCompress modules:", e)
        DARK_AVAILABLE = False

def parallelize_compressor(gpu_quotas, task_usages, task_names_local, n_iterations, end_time, verbose=False):
    # Implementation adapted from your parallelize_runs snippet but simplified for per-task queue.
    n_gpus = len(gpu_quotas)
    n_cpus = multiprocessing.cpu_count()
    tasks_started = {t:False for t in task_names_local}
    tasks_finished = {t:False for t in task_names_local}
    processes = {t:None for t in task_names_local}
    process_gpu_ids = {t:None for t in task_names_local}
    manager = Manager()
    memory_dict = manager.dict()
    solutions_dict = manager.dict()
    error_queue = manager.Queue()
    try:
        while not all(tasks_finished.values()):
            if not error_queue.empty():
                raise ValueError(error_queue.get())
            # check finished
            for t in task_names_local:
                if tasks_started[t] and not tasks_finished[t]:
                    p = processes[t]
                    if p is not None:
                        p.join(timeout=0)
                        if not p.is_alive():
                            tasks_finished[t] = True
                            gpu_quotas[process_gpu_ids[t]] += task_usages[task_names_local.index(t)]
                            if verbose: print(f"[compressor] {t} finished on gpu {process_gpu_ids[t]}")
            # schedule new
            for gpu_id in range(n_gpus):
                for t in task_names_local:
                    if tasks_started[t]: continue
                    enough_quota = gpu_quotas[gpu_id] > task_usages[task_names_local.index(t)]
                    enough_cpus = sum(map(int, tasks_started.values())) - sum(map(int, tasks_finished.values())) < n_cpus
                    if enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[task_names_local.index(t)]
                        args = (t, "test", end_time, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue)
                        p = Process(target=compressor_solve_task.solve_task, args=args)
                        p.start()
                        processes[t] = p
                        tasks_started[t] = True
                        process_gpu_ids[t] = gpu_id
                        if verbose: print(f"[compressor] started {t} on gpu {gpu_id}")
            time.sleep(0.5)
    finally:
        # convert results to normal dicts
        mem = dict(memory_dict); sols = dict(solutions_dict)
        return mem, sols

def run_gpu_compressor_for_tasks(task_list, end_time, verbose=True):
    if not DARK_AVAILABLE:
        print("[WARN] DarkAGICCompress not available. Skipping GPU compressor.")
        return {}
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[WARN] No CUDA GPUs detected. Skipping GPU compressor.")
        return {}
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
    gpu_task_quotas = [int(m // (4 * 1024**3)) for m in gpu_memory_quotas]
    task_usages = [1 for _ in range(len(task_list))]
    # quick warmup (2 iterations) to measure memory usage
    mem, sols = parallelize_compressor(gpu_task_quotas[:], task_usages, task_list, n_iterations=2, end_time=end_time, verbose=False)
    # sort tasks by memory usage if available
    tasks_sorted = sorted(mem.items(), key=lambda x: x[1], reverse=True) if mem else [(t,1) for t in task_list]
    ordered_tasks = [t for t,_ in tasks_sorted]
    # run timed test phases to estimate per-step time
    test_steps = 8
    safe_gpu_memory_quotas = [m - 6 * 1024**3 for m in gpu_memory_quotas]
    _mem2, _sol2 = parallelize_compressor(safe_gpu_memory_quotas[:], task_usages, ordered_tasks, n_iterations=test_steps, end_time=end_time, verbose=False)
    # compute total steps we can do in remaining time
    # crude heuristics: run as many rounds of test_steps as time allows
    time_left = end_time - time.time()
    estimated_one_round = max(30.0, test_steps * 2.0)  # baseline guess
    n_rounds = max(1, int(time_left // estimated_one_round))
    # run full
    mem_final, sols_final = parallelize_compressor(safe_gpu_memory_quotas[:], task_usages, ordered_tasks, n_iterations=test_steps * n_rounds, end_time=end_time, verbose=verbose)
    print(f"[compressor] GPU run completed. tasks solved: {len(sols_final)}")
    return sols_final

# ---------------------------
# 14) Orchestration: hybrid driver
# ---------------------------
def hybrid_driver(use_gpu_compressor=True, time_budget_hours=12.0):
    # load dataset (prefer evaluation if present; else test)
    split_file = TEST_PATH if os.path.exists(TEST_PATH) else EVAL_PATH
    challenges = load_json(split_file)
    task_names = list(challenges.keys())
    n_tasks = len(task_names)
    print(f"[driver] Loaded {n_tasks} tasks from {split_file}")

    start_time = time.time(); end_time = start_time + time_budget_hours*3600 - 20*60  # 20 min safety
    # Step A: Symbolic first (fast)
    print("[driver] Running symbolic solver first (fast CPU pass)...")
    sym_out = build_submission_symbolic_only(split_file, train_path=TRAIN_PATH, out_path=os.path.join(MODEL_DIR, 'symbolic_partial.json'), per_task_time=6.0)
    # determine unsolved
    unsolved = []
    # symbolic output format may be different; check which tasks have non-trivial attempts
    for tid in task_names:
        # read symbolic results if available
        if str(tid) in sym_out:
            entries = sym_out[str(tid)]
            solved_flag = False
            for entry in entries:
                # heuristically check attempt fields
                if 'attempt_1' in entry:
                    if entry['attempt_1'] != [[0]] and entry['attempt_1'] is not None:
                        solved_flag = True; break
                elif 'output' in entry:
                    solved_flag = True; break
            if not solved_flag:
                unsolved.append(tid)
        else:
            unsolved.append(tid)
    print(f"[driver] Symbolic solved approx {n_tasks - len(unsolved)} tasks; {len(unsolved)} remain for GPU compressor")

    # Step B: GPU compressor only for unsolved tasks
    compressor_results = {}
    if use_gpu_compressor and len(unsolved)>0 and DARK_AVAILABLE:
        print(f"[driver] Running GPU compressor on {len(unsolved)} tasks; GPUs: {torch.cuda.device_count()}")
        compressor_results = run_gpu_compressor_for_tasks(unsolved, end_time=end_time, verbose=True)
    else:
        if not DARK_AVAILABLE:
            print("[driver] GPU compressor not available; skipping")
        else:
            print("[driver] No unsolved tasks or compressor not requested; skipping GPU")

    # Step C: Merge outputs: prioritise symbolic outputs, else compressor outputs
    final = {}
    # load symbolic partial results file to inspect structure
    try:
        sym_json = load_json(os.path.join(MODEL_DIR, 'symbolic_partial.json'))
    except Exception:
        sym_json = {}

    for tid in task_names:
        tid_str = str(tid)
        if tid_str in sym_json:
            final[tid_str] = sym_json[tid_str]
        elif tid in compressor_results:
            final[tid_str] = compressor_results[tid]
        else:
            # fallback fill
            # fetch task test shape and create trivial attempts
            t = challenges[tid]
            entries=[]
            for test in t.get('test', []):
                entries.append({'attempt_1': sanitize_grid(test['input']), 'attempt_2': sanitize_grid(test['input'])})
            final[tid_str] = entries

    # save final submission
    save_json(final, OUT_PATH)
    print("[driver] Saved submission to", OUT_PATH)
    print("[driver] Done. Runtime:", time.time()-start_time, "seconds")
    return final

# ---------------------------
# 15) Run
# ---------------------------
if __name__ == "__main__":
    # NOTE: On Kaggle set runtime to GPU, choose T4x2 GPUs for best throughput.
    print("Starting HYBRID ARC solver. GPU available:", torch.cuda.is_available(), "CUDA device count:", torch.cuda.device_count())
    out = hybrid_driver(use_gpu_compressor=True, time_budget_hours=12.0)
    print("Submission saved. Final tasks:", len(out))



