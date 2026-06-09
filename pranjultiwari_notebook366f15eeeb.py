# DigitalSoulARC Pro - Hybrid Heuristic ARC Solver
# Drop into a Kaggle notebook cell and run
import os, sys, json, time, math, traceback, copy, itertools, functools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Environment / Silencing noisy CUDA plugin messages & safety
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# limit visible CUDA devices if necessary:
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# ---------------------------------------------------------------------------

# Paths (Kaggle standard)
ROOT = "/kaggle/input/arc-prize-2025"
CH_PATH = os.path.join(ROOT, "arc-agi_test_challenges.json")
SAMPLE_PATH = os.path.join(ROOT, "sample_submission.json")
OUT_PATH = "submission.json"

# Runtime config
USE_LLM = True                  # Set True to enable optional LLM-assisted transform classification
LLM_MODEL_DIR = "/kaggle/input/codellama-7b/other/default/1"  # example
MAX_WORKERS = min(6, (os.cpu_count() or 2))  # parallel tasks
MAX_SEQUENCE_TRY = 2             # try single transform and chains up to this length
TIMEOUT_PER_TASK = 30            # seconds (not used to kill threads but a guidance for complexity)
VERBOSE = True

# ---------------------------------------------------------------------------
# Optional imports (torch/transformers) only if USE_LLM True
model = tokenizer = None
if USE_LLM:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if VERBOSE: print("[INFO] Device:", DEVICE)
        if Path(LLM_MODEL_DIR).exists():
            if VERBOSE: print("[INFO] Loading classifier LLM from:", LLM_MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR, local_files_only=True, truncation_side="left")
            model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_DIR, local_files_only=True, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, low_cpu_mem_usage=True)
            model.eval()
            if VERBOSE: print("[INFO] LLM loaded (classifier-only mode).")
        else:
            if VERBOSE: print("[WARN] LLM model path not found — continuing with heuristics only.")
            USE_LLM = False
    except Exception as e:
        print("[WARN] LLM init failed:", e)
        USE_LLM = False

# ---------------------------------------------------------------------------
# Basic libs
import numpy as np
try:
    from scipy.ndimage import label
except Exception:
    label = None

# Pydantic for structured parsing (optional); fallback if not present
try:
    from pydantic import BaseModel
    from typing import List
    class GridModel(BaseModel):
        grid: List[List[int]]
    HAS_PYDANTIC = True
except Exception:
    HAS_PYDANTIC = False

# ---------------------------------------------------------------------------
# Utilities
def load_json(p):
    with open(p) as f:
        return json.load(f)
def save_json(o,p):
    with open(p,"w") as f:
        json.dump(o,f)
def as_np(grid):
    return np.array(grid, dtype=int)
def as_list(np_arr):
    return np_arr.tolist() if isinstance(np_arr, np.ndarray) else grid_safe_list(np_arr)
def grid_safe_list(g):
    # ensure int lists
    return [[int(x) for x in row] for row in g]

def validate_grid_candidate(g, shape=None):
    # strong validation: must be 2D list ints, shape match optional
    try:
        if HAS_PYDANTIC:
            GridModel(grid=g)
        else:
            assert isinstance(g, list)
            h = len(g)
            assert h>0 and h<=30
            w = len(g[0])
            assert w>0 and w<=30
            for r in g:
                assert isinstance(r, list) and len(r)==w
                for c in r:
                    assert isinstance(c, (int, np.integer))
        if shape:
            return (len(g), len(g[0])) == tuple(shape)
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# OBJECT / REGION helpers
def extract_objects(grid):
    g = as_np(grid)
    objs=[]
    if label is None:
        # fallback: connected components by 4-neighbour - simple BFS for non-zero regions
        h,w = g.shape
        vis = np.zeros_like(g, dtype=bool)
        for y in range(h):
            for x in range(w):
                if g[y,x]==0 or vis[y,x]: continue
                color = int(g[y,x])
                stack=[(y,x)]; mask=[]
                while stack:
                    yy,xx = stack.pop()
                    if yy<0 or yy>=h or xx<0 or xx>=w or vis[yy,xx] or g[yy,xx]==0: continue
                    vis[yy,xx]=True; mask.append((yy,xx))
                    stack.extend([(yy+1,xx),(yy-1,xx),(yy,xx+1),(yy,xx-1)])
                coords = np.array(mask)
                objs.append({"coords":coords, "color":color, "size":len(coords)})
    else:
        labeled, n = label(g != 0)
        for i in range(1, n+1):
            mask = (labeled==i)
            coords = np.argwhere(mask)
            colors, counts = np.unique(g[mask], return_counts=True)
            color = int(colors[np.argmax(counts)])
            objs.append({"coords":coords, "mask":mask, "color":color, "size":len(coords)})
    objs.sort(key=lambda o: -o["size"])
    return objs

def bbox_from_coords(coords):
    ys = coords[:,0]; xs = coords[:,1]
    return ys.min(), xs.min(), ys.max()+1, xs.max()+1

# ---------------------------------------------------------------------------
# TRANSFORMATIONS (the big list) - each returns a function f(grid)->grid
def tf_identity():
    return lambda g: as_list(as_np(g))

def tf_flip_h():
    return lambda g: as_list(np.fliplr(as_np(g)))

def tf_flip_v():
    return lambda g: as_list(np.flipud(as_np(g)))

def tf_rotate90():
    return lambda g: as_list(np.rot90(as_np(g), k=1))

def tf_rotate180():
    return lambda g: as_list(np.rot90(as_np(g), k=2))

def tf_rotate270():
    return lambda g: as_list(np.rot90(as_np(g), k=3))

def tf_transpose():
    return lambda g: as_list(as_np(g).T)

def tf_diag_reflect_main():  # reflect across main diagonal (transpose)
    return tf_transpose()

def tf_diag_reflect_anti():  # reflect across anti-diagonal
    def fn(g):
        a = as_np(g)
        return as_list(np.fliplr(np.rot90(a, 1)))  # rotate+flip approximates anti-diagonal reflect
    return fn

def tf_crop_to_bbox_of_color(color):
    def fn(g):
        a = as_np(g)
        coords = np.argwhere(a==color)
        if coords.size==0: return as_list(a*0)
        y0,x0,y1,x1 = bbox_from_coords(coords)
        return as_list(a[y0:y1, x0:x1])
    return fn

def tf_keep_largest_object():
    def fn(g):
        a = as_np(g); out = np.zeros_like(a)
        objs = extract_objects(a)
        if not objs: return as_list(out)
        m = objs[0]
        for (y,x) in m["coords"]:
            out[y,x] = m["color"]
        return as_list(out)
    return fn

def tf_keep_smallest_object():
    def fn(g):
        a = as_np(g); out = np.zeros_like(a)
        objs = extract_objects(a)
        if not objs: return as_list(out)
        m = objs[-1]
        for (y,x) in m["coords"]:
            out[y,x] = m["color"]
        return as_list(out)
    return fn

def tf_color_map(mapping):
    # mapping: dict old->new
    def fn(g):
        a = as_np(g).copy()
        out = np.zeros_like(a)
        for old, new in mapping.items():
            out[a==old] = int(new)
        # keep zeros
        out[a==0] = 0
        return as_list(out)
    return fn

def detect_color_map(inp, out):
    # tries to detect mapping where out = color_map(inp)
    inp_a = as_np(inp); out_a = as_np(out)
    unique_in = np.unique(inp_a)
    mapping = {}
    for u in unique_in:
        if u==0: continue
        mask = (inp_a==u)
        vals, counts = np.unique(out_a[mask], return_counts=True)
        if len(vals)==0:
            mapping[u]=0
        else:
            mapping[u] = int(vals[np.argmax(counts)])
    # sanity: apply and compare
    applied = as_np(inp).copy()*0
    for k,v in mapping.items(): applied[as_np(inp)==k] = v
    if np.array_equal(applied, out_a):
        return mapping
    return None

def tf_tile(tile_pattern):
    # tile_pattern smaller grid => tile to target size based on test input size at runtime
    def fn(g):
        a = as_np(g); th,tw = a.shape
        pat = as_np(tile_pattern)
        ph,pw = pat.shape
        out = np.zeros((th,tw), dtype=int)
        for y in range(th):
            for x in range(tw):
                out[y,x] = int(pat[y%ph, x%pw])
        return as_list(out)
    return fn

def tf_majority_infill():
    # fill zeros by majority color in same row/column or global mode
    def fn(g):
        a = as_np(g).copy()
        h,w = a.shape
        for y in range(h):
            for x in range(w):
                if a[y,x]==0:
                    # row mode
                    row = a[y,:]; vals = row[row!=0]
                    if vals.size>0:
                        a[y,x] = int(np.bincount(vals).argmax())
                        continue
                    col = a[:,x]; vals = col[col!=0]
                    if vals.size>0:
                        a[y,x] = int(np.bincount(vals).argmax())
                        continue
                    # global
                    vals = a[a!=0]
                    a[y,x] = int(np.bincount(vals).argmax()) if vals.size>0 else 0
        return as_list(a)
    return fn

def tf_shift_object_to_corner(color, corner="tl"):
    def fn(g):
        a = as_np(g); out = np.zeros_like(a)
        coords = np.argwhere(a==color)
        if coords.size==0: return as_list(out)
        y0,x0,y1,x1 = bbox_from_coords(coords)
        h,w = a.shape
        ph = y1-y0; pw = x1-x0
        if corner=="tl":
            sy,sx = 0,0
        elif corner=="tr":
            sy,sx = 0, w-pw
        elif corner=="bl":
            sy,sx = h-ph, 0
        else:
            sy,sx = h-ph, w-pw
        out[sy:sy+ph, sx:sx+pw] = np.asarray(a[y0:y1, x0:x1])
        return as_list(out)
    return fn

def tf_overlay_object_at(grid_src, color_from, color_to, dy, dx):
    # not used directly, provided as helper for composition
    pass

# Compose functions: apply sequence fns
def compose_fns(fn_list):
    def composed(g):
        cur = g
        for f in fn_list:
            cur = f(cur)
        return cur
    return composed

# Build catalog of candidate transforms (callables) with name tags
def build_transform_catalog(task=None):
    catalog = []
    # basic geometric
    catalog.append(("identity", tf_identity()))
    catalog.append(("flip_h", tf_flip_h()))
    catalog.append(("flip_v", tf_flip_v()))
    catalog.append(("rot90", tf_rotate90()))
    catalog.append(("rot180", tf_rotate180()))
    catalog.append(("rot270", tf_rotate270()))
    catalog.append(("transpose", tf_transpose()))
    catalog.append(("diag_reflect_anti", tf_diag_reflect_anti()))
    # object-centric
    catalog.append(("keep_largest_obj", tf_keep_largest_object()))
    catalog.append(("keep_smallest_obj", tf_keep_smallest_object()))
    catalog.append(("majority_infill", tf_majority_infill()))
    # if we have a task, detect common tile pattern (fast heuristic) and add
    if task:
        # try to find small repeated motif in outputs of train (if outputs smaller than inputs)
        for p in task.get("train", []):
            inp = as_np(p["input"]); out = as_np(p["output"])
            if out.shape[0]<=inp.shape[0] and out.shape[1]<=inp.shape[1]:
                # treat out as tile
                catalog.append((f"tile_from_example_{len(catalog)}", tf_tile(out)))
    return catalog

# ---------------------------------------------------------------------------
# Matching / rule discovery
def apply_and_validate(fn, g, expected):
    try:
        cand = fn(g)
        if not validate_grid_candidate(cand, shape=(len(expected), len(expected[0])) if expected else None):
            return False, None
        return True, cand
    except Exception:
        return False, None

def find_transform_from_examples(task, max_sequence=MAX_SEQUENCE_TRY):
    """
    Try to find a deterministic transform (single or sequence of transforms) that maps each train input->train output.
    Returns (name, fn) or (None,None)
    """
    catalog = build_transform_catalog(task)
    # quick color-map detection: if outputs are same shape and positions but colors remapped
    try:
        # shape-preserving mapping detection across all pairs
        shape_preserving = True
        for p in task["train"]:
            if np.array(as_np(p["input"]).shape).tolist() != np.array(as_np(p["output"]).shape).tolist():
                shape_preserving = False; break
        if shape_preserving:
            # try to detect a single color mapping that explains all pairs
            global_map = {}
            consistent = True
            for p in task["train"]:
                mapping = detect_color_map(p["input"], p["output"])
                if mapping is None:
                    consistent=False; break
                # merge mapping
                for k,v in mapping.items():
                    if k in global_map and global_map[k]!=v:
                        consistent=False; break
                    global_map[k]=v
                if not consistent: break
            if consistent and len(global_map)>0:
                fn = tf_color_map(global_map)
                # test strictly
                ok_all = True
                for p in task["train"]:
                    ok, cand = apply_and_validate(fn, p["input"], p["output"])
                    if not ok or not np.array_equal(as_np(cand), as_np(p["output"])):
                        ok_all=False; break
                if ok_all:
                    return ("color_map", fn)
    except Exception:
        pass

    # Try single transforms
    for name,fn in catalog:
        ok_all=True
        for p in task["train"]:
            ok, cand = apply_and_validate(fn, p["input"], p["output"])
            if not ok:
                ok_all=False; break
            if not np.array_equal(as_np(cand), as_np(p["output"])):
                ok_all=False; break
        if ok_all:
            return (name, fn)

    # Try sequences of transforms (depth-limited)
    if max_sequence>=2:
        names = [n for n,_ in catalog]
        fns = [f for _,f in catalog]
        for (i,j) in itertools.product(range(len(fns)), repeat=2):
            fn_seq = compose_fns([fns[i], fns[j]])
            name = f"{names[i]}+{names[j]}"
            ok_all=True
            for p in task["train"]:
                ok, cand = apply_and_validate(fn_seq, p["input"], p["output"])
                if not ok or not np.array_equal(as_np(cand), as_np(p["output"])):
                    ok_all=False; break
            if ok_all:
                return (name, fn_seq)

    return (None, None)

# ---------------------------------------------------------------------------
# Spatial/object translation detection
def detect_translation_between(inp, out):
    """
    Detects dominant translation (dy, dx, color) between input and output grids.
    Works safely for all array sizes and boundaries.
    """
    a = np.array(inp, dtype=np.int32)
    b = np.array(out, dtype=np.int32)
    h, w = a.shape

    for color in np.unique(a):
        if color == 0:
            continue
        coords_in = np.argwhere(a == color)
        coords_out = np.argwhere(b == color)
        if coords_in.size == 0 or coords_out.size == 0:
            continue

        dy = int(round(coords_out[:, 0].mean() - coords_in[:, 0].mean()))
        dx = int(round(coords_out[:, 1].mean() - coords_in[:, 1].mean()))

        shifted = coords_in + np.array([dy, dx])

        # ✅ Clip to valid range (avoid out-of-bounds errors)
        valid_mask = (
            (shifted[:, 0] >= 0) & (shifted[:, 0] < h) &
            (shifted[:, 1] >= 0) & (shifted[:, 1] < w)
        )
        shifted = shifted[valid_mask]

        if shifted.size == 0:
            continue

        # ✅ Count how many shifted pixels still match in output
        overlap = 0
        for y, x in shifted:
            try:
                if b[y, x] == color:
                    overlap += 1
            except IndexError:
                # should never happen after clipping, but be extra safe
                continue

        if overlap / max(1, len(coords_in)) > 0.9:
            return dy, dx, int(color)
    return None

# ---------------------------------------------------------------------------
# LLM-assisted classification (optional): given train pairs, ask model which transform from the catalog best fits
def llm_classify_transform(task, catalog_names):
    if not USE_LLM or model is None or tokenizer is None:
        return None
    try:
        # build short prompt instructing classification from list of transform names
        short_list = "\n".join([f"- {n}" for n in catalog_names])
        train_text = "\n\n".join([f"Input:\n{p['input']}\nOutput:\n{p['output']}" for p in task["train"]])
        prompt = f"You are a classifier. Given training input->output grid pairs, identify which transform from the following list best explains them. Reply ONLY the transform name exactly as in the list.\nTransforms:\n{short_list}\n\nTraining examples:\n{train_text}\n\nAnswer with one transform name:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(next(model.parameters()).device)
        gen_cfg = dict(max_new_tokens=32, do_sample=False, num_beams=4, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_cfg)
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        # pick first line, sanitize
        name = text.splitlines()[0].strip()
        if name in catalog_names:
            return name
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Main per-task solver: strong heuristic-first approach and deterministic application
def solve_task_tid(tid, task):
    start = time.time()
    try:
        # 1) try to find a transform by examples
        name, fn = find_transform_from_examples(task, max_sequence=MAX_SEQUENCE_TRY)
        if name:
            # apply to each test
            preds=[]
            for tcase in task["test"]:
                out = fn(tcase["input"])
                if not validate_grid_candidate(out, shape=(len(tcase["input"]), len(tcase["input"][0]))) :
                    # try resizing (ensure same shape by cropping/padding)
                    out_a = as_np(out); in_a = as_np(tcase["input"])
                    if out_a.shape != in_a.shape:
                        # pad/crop center
                        oh,ow = out_a.shape; ih,iw = in_a.shape
                        H = max(oh, ih); W = max(ow, iw)
                        big = np.zeros((H,W), dtype=int)
                        big[:oh,:ow] = out_a
                        crop = big[:ih,:iw]
                        out = as_list(crop)
                preds.append({"attempt_1": out, "attempt_2": out})
            return tid, preds

        # 2) attempt color mapping as fallback
        # try detect color mapping per pair and check consistency
        cmap = None
        try:
            mappings = [detect_color_map(p["input"], p["output"]) for p in task["train"]]
            if all(m is not None for m in mappings):
                merged = {}
                ok=True
                for m in mappings:
                    for k,v in m.items():
                        if k in merged and merged[k]!=v:
                            ok=False; break
                        merged[k]=v
                    if not ok: break
                if ok and len(merged)>0:
                    cmap = merged
        except Exception:
            cmap=None
        if cmap:
            fnc = tf_color_map(cmap)
            preds=[]
            for tcase in task["test"]:
                out = fnc(tcase["input"])
                # ensure shape
                inp_shape = as_np(tcase["input"]).shape
                if as_np(out).shape != inp_shape:
                    # try to embed mapping preserving shape by applying mapping per cell but shape unchanged: done in tf_color_map
                    pass
                preds.append({"attempt_1": out, "attempt_2": out})
            return tid, preds

        # 3) try translations between input/output within train set (object movement)
        # if train pairs show consistent vector for a particular color, apply same translation
        translation_candidates = []
        for p in task["train"]:
            det = detect_translation_between(p["input"], p["output"])
            if det:
                translation_candidates.append(det)
        if translation_candidates:
            # pick most common (dy,dx,color)
            from collections import Counter
            common = Counter(translation_candidates).most_common(1)[0][0]
            dy,dx,color = common
            def apply_translation(g):
                a = as_np(g); out = np.zeros_like(a)
                coords = np.argwhere(a==color)
                for (y,x) in coords:
                    ny, nx = y+dy, x+dx
                    if 0<=ny<a.shape[0] and 0<=nx<a.shape[1]:
                        out[ny,nx]=color
                return as_list(out)
            preds=[]
            for tcase in task["test"]:
                out = apply_translation(tcase["input"])
                preds.append({"attempt_1": out, "attempt_2": out})
            return tid, preds

        # 4) strong heuristic enumeration (try many transforms + quick checks) - fast checks only
        catalog = build_transform_catalog(task)
        catalog_names = [n for n,_ in catalog]
        # Optionally use LLM to pick one of catalog_names (classifier)
        pick_name = None
        if USE_LLM and model is not None:
            try:
                pick_name = llm_classify_transform(task, catalog_names)
                if VERBOSE and pick_name: print(f"[LLM-classify] Task {tid} -> {pick_name}")
            except Exception:
                pick_name = None
        # If classifier picked, try that transform first
        trial_order = []
        if pick_name:
            for n,f in catalog:
                if n==pick_name:
                    trial_order.append((n,f))
            # then the rest
            trial_order += [(n,f) for n,f in catalog if n!=pick_name]
        else:
            trial_order = catalog

        # Try each transform and validate across all train pairs (fast)
        for name,fn in trial_order:
            ok_all=True
            for p in task["train"]:
                ok, cand = apply_and_validate(fn, p["input"], p["output"])
                if not ok:
                    ok_all=False; break
                if not np.array_equal(as_np(cand), as_np(p["output"])):
                    ok_all=False; break
            if ok_all:
                # apply to tests
                preds=[]
                for tcase in task["test"]:
                    out = fn(tcase["input"])
                    preds.append({"attempt_1": out, "attempt_2": out})
                return tid, preds

        # 5) tiling/pattern replication detection: try to see if train outputs are small patterns extracted and tiled
        # quick approach: check if any train output is significantly smaller than its input and appears repeated in input
        for p in task["train"]:
            inp, outp = as_np(p["input"]), as_np(p["output"])
            if outp.size==0: continue
            if outp.shape[0] <= inp.shape[0] and outp.shape[1] <= inp.shape[1]:
                # try to tile outp to input shape and then compare to outp of other pairs
                fn_tile = tf_tile(outp)
                # test across all train examples: whether output equals tiled version or maps to it by color mapping
                ok_all=True
                for q in task["train"]:
                    cand = as_np(fn_tile(q["input"]))
                    # if shapes differ, tile to shape of q
                    cand = cand[:as_np(q["output"]).shape[0], :as_np(q["output"]).shape[1]]
                    if not np.array_equal(cand, as_np(q["output"])):
                        ok_all=False; break
                if ok_all:
                    preds=[]
                    for tcase in task["test"]:
                        preds.append({"attempt_1": fn_tile(tcase["input"]), "attempt_2": fn_tile(tcase["input"])})
                    return tid, preds

        # 6) majority infill: try completing zeros in inputs to match outputs (pattern completion)
        for name,fn in [("majority_infill", tf_majority_infill())]:
            ok_all=True
            for p in task["train"]:
                ok,cand = apply_and_validate(fn, p["input"], p["output"])
                if not ok or not np.array_equal(as_np(cand), as_np(p["output"])):
                    ok_all=False; break
            if ok_all:
                preds=[]
                for tcase in task["test"]:
                    out = fn(tcase["input"])
                    preds.append({"attempt_1": out, "attempt_2": out})
                return tid, preds

        # 7) If nothing else, attempt to produce a conservative "recomposition" answer:
        #    - detect largest object in each train output and re-place it into test input bounding area (best-effort)
        preds=[]
        for tcase in task["test"]:
            inp = as_np(tcase["input"])
            # pick largest non-zero color across all train outputs and place that shape centered
            shapes=[]
            for p in task["train"]:
                outp = as_np(p["output"])
                objs = extract_objects(outp)
                if objs: shapes.append(objs[0])  # largest
            if shapes:
                shape = shapes[0]
                # center it in input
                h,w = inp.shape
                coords = shape["coords"]
                y0,x0,y1,x1 = bbox_from_coords(coords)
                ph,pw = y1-y0, x1-x0
                sy = max(0,(h-ph)//2); sx = max(0,(w-pw)//2)
                out = np.zeros_like(inp)
                # place shape pixels with its color
                patch = np.zeros((ph,pw), dtype=int)
                for (yy,xx) in coords:
                    patch[yy-y0, xx-x0] = shape["color"]
                ph, pw = patch.shape
                sy, sx = max(0, sy), max(0, sx)
                eh, ew = min(out.shape[0], sy + ph), min(out.shape[1], sx + pw)
                out[sy:eh, sx:ew] = patch[:eh - sy, :ew - sx]
                preds.append({"attempt_1": as_list(out), "attempt_2": as_list(out)})
            else:
                # fallback: echo input
                preds.append({"attempt_1": as_list(inp), "attempt_2": as_list(inp)})
        return tid, preds

    except Exception as e:
        if VERBOSE: print(f"[ERROR] Task {tid} crashed heuristics:", e, traceback.format_exc()[:400])
        # fail-safe: return inputs as attempts
        preds = []
        for tcase in task["test"]:
            preds.append({"attempt_1": as_list(as_np(tcase["input"])), "attempt_2": as_list(as_np(tcase["input"]))})
        return tid, preds
    finally:
        if VERBOSE:
            dur = time.time()-start
            if dur>1.0:
                print(f"[INFO] Task {tid} processed in {dur:.2f}s")

# ---------------------------------------------------------------------------
# Solve all tasks (parallel)
def solve_all_and_write(ch_path=CH_PATH, sample_path=SAMPLE_PATH, out_path=OUT_PATH, max_workers=MAX_WORKERS):
    test_data = load_json(ch_path)
    sample_sub = load_json(sample_path)
    tasks = list(sample_sub.keys())
    submission = {}
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(solve_task_tid, tid, test_data[tid]): tid for tid in tasks if tid in test_data}
        completed=0
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                tid_ret, preds = fut.result()
                submission[str(tid_ret)] = preds
            except Exception as e:
                # fill with sample as fallback
                submission[str(tid)] = sample_sub[tid]
            completed += 1
            if VERBOSE and completed%20==0:
                print(f"[INFO] Solved {completed}/{len(tasks)} tasks.")
    # ensure every sample key present
    final = {}
    for tid in sample_sub:
        if tid in submission:
            final[tid] = submission[tid]
        else:
            final[tid] = sample_sub[tid]
    save_json(final, out_path)
    total = time.time()-start_all
    print(f"[DONE] Submission saved to {out_path}. Total time: {total:.1f}s")
    return final

# ---------------------------------------------------------------------------
# Run
if __name__ == "__main__":
    t0 = time.time()
    print("[START] DigitalSoulARC Pro - Heuristic-first solver")
    out = solve_all_and_write()
    print("[FINISH] Elapsed:", time.time()-t0)





