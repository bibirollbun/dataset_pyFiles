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


import os, json, math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque

 
DATA_DIR = "/kaggle/input/arc-prize-2025/"   

EVAL_CH  = os.path.join(DATA_DIR, "arc-agi_evaluation_challenges.json")
EVAL_SOL = os.path.join(DATA_DIR, "arc-agi_evaluation_solutions.json")

TEST_CH   = os.path.join(DATA_DIR, "arc-agi_test_challenges.json")
TRAIN_CH  = os.path.join(DATA_DIR, "arc-agi_training_challenges.json")

TRAIN_SOL   = os.path.join(DATA_DIR, "arc-agi_training_solutions.json")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.json")
 
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

train_ch = load_json(TRAIN_CH)
train_sol = load_json(TRAIN_SOL)
eval_ch = load_json(EVAL_CH)
eval_sol = load_json(EVAL_SOL)
test_ch = load_json(TEST_CH)  # placeholder locally; real file appears on rerun

len(train_ch), len(eval_ch), len(test_ch)



train_ch = load_json(TRAIN_CH)      # dict: task_id -> {"train":[...], "test":[...]}
train_sol = load_json(TRAIN_SOL)    # dict or list
eval_ch = load_json(EVAL_CH)
eval_sol = load_json(EVAL_SOL)
test_ch = load_json(TEST_CH)        # dict: used to build submission

print("Loaded:",
      f"TRAIN {len(train_ch)} challenges,",
      f"EVAL {len(eval_ch)} challenges,",
      f"TEST {len(test_ch)} challenges")

# ------------- Core helpers -------------
def grid_to_np(g):  # list[list[int]] -> np.ndarray[int]
    return np.array(g, dtype=int)

def np_to_grid(a):  # np.ndarray[int] -> list[list[int]]
    return a.astype(int).tolist()

def same_shape(a, b):
    return grid_to_np(a).shape == grid_to_np(b).shape

def palette(A):
    return sorted(np.unique(A).tolist())

def border_mode_color(A):
    H, W = A.shape
    if H == 0 or W == 0:
        return 0
    border = np.concatenate([A[0, :], A[-1, :], A[:, 0], A[:, -1]])
    return Counter(border.tolist()).most_common(1)[0][0]

def infer_bg_color_from_output(out_grid):
    return border_mode_color(grid_to_np(out_grid))

# Visualization (optional)
def plot_grid(grid, title="Grid"):
    arr = grid_to_np(grid)
    plt.imshow(arr, cmap="tab20", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.show()

# ------------- Dihedral transforms -------------
def t_identity(a):   return a
def t_rot90(a):      return np.rot90(a, 1)
def t_rot180(a):     return np.rot90(a, 2)
def t_rot270(a):     return np.rot90(a, 3)
def t_flipud(a):     return np.flipud(a)
def t_fliplr(a):     return np.fliplr(a)
def t_transpose(a):  return a.T
def t_antitrans(a):  return np.rot90(a, 1).T

DIHEDRAL = [
    ("identity",   t_identity),
    ("rot90",      t_rot90),
    ("rot180",     t_rot180),
    ("rot270",     t_rot270),
    ("flipud",     t_flipud),
    ("fliplr",     t_fliplr),
    ("transpose",  t_transpose),
    ("antitrans",  t_antitrans),
]

# ------------- Simple rules -------------
# Identity
def rule_identity(inp):
    return grid_to_np(inp)

def fits_rule_identity(train_pairs):
    for p in train_pairs:
        if not np.array_equal(grid_to_np(p["input"]), grid_to_np(p["output"])):
            return False
    return True

# Global color permutation
def infer_color_map(inp, out):
    A = grid_to_np(inp)
    B = grid_to_np(out)
    if A.shape != B.shape:
        return None
    mapping = {}
    for v_in, v_out in zip(A.flatten(), B.flatten()):
        if v_in in mapping and mapping[v_in] != v_out:
            return None
        mapping[v_in] = v_out
    # optional injective check
    inv = {}
    for k,v in mapping.items():
        if v in inv and inv[v] != k:
            return None
        inv[v] = k
    return mapping

def apply_color_map(inp, cmap):
    A = grid_to_np(inp)
    B = A.copy()
    for c in np.unique(A):
        B[A == c] = cmap.get(c, c)
    return B

def fits_rule_color_perm(train_pairs):
    cmap = None
    for p in train_pairs:
        m = infer_color_map(p["input"], p["output"])
        if m is None:
            return None
        if cmap is None:
            cmap = dict(m)
        else:
            for k, v in m.items():
                if k in cmap and cmap[k] != v:
                    return None
                cmap[k] = v
    return cmap

# Integer scaling (pixel repeat)
def is_integer_scale(inp, out):
    A = grid_to_np(inp); B = grid_to_np(out)
    H1, W1 = A.shape; H2, W2 = B.shape
    if H2 % H1 != 0 or W2 % W1 != 0:
        return None
    sy, sx = H2 // H1, W2 // W1
    return (sy, sx)

def upscale_repeat(inp, sy, sx):
    A = grid_to_np(inp)
    return np.repeat(np.repeat(A, sy, axis=0), sx, axis=1)

def fits_rule_integer_scale(train_pairs):
    scale = None
    for p in train_pairs:
        s = is_integer_scale(p["input"], p["output"])
        if s is None:
            return None
        pred = upscale_repeat(p["input"], *s)
        if not np.array_equal(pred, grid_to_np(p["output"])):
            return None
        if scale is None:
            scale = s
        elif scale != s:
            return None
    return scale

# ------------- Dihedral rules -------------
def fits_rule_dihedral(train_pairs):
    for name, T in DIHEDRAL:
        ok = True
        for p in train_pairs:
            A = T(grid_to_np(p["input"]))
            B = grid_to_np(p["output"])
            if A.shape != B.shape or not np.array_equal(A, B):
                ok = False
                break
        if ok:
            return (name, T)
    return None

def fits_rule_dihedral_colorperm(train_pairs):
    for name, T in DIHEDRAL:
        cmap = None
        ok = True
        for p in train_pairs:
            A = T(grid_to_np(p["input"]))
            B = grid_to_np(p["output"])
            m = infer_color_map(A, B)
            if m is None:
                ok = False
                break
            if cmap is None:
                cmap = dict(m)
            else:
                for k, v in m.items():
                    if k in cmap and cmap[k] != v:
                        ok = False
                        break
                    cmap[k] = v
            if not ok:
                break
        if ok:
            return (name, T, cmap)
    return None

def apply_T_then_cmap(grid, T, cmap):
    A = T(grid_to_np(grid))
    B = A.copy()
    for c in np.unique(A):
        B[A == c] = cmap.get(c, c)
    return B

# ------------- Translation rules -------------
def shift_with_bg(A, dy, dx, bg):
    A = grid_to_np(A)
    H, W = A.shape
    B = np.full((H, W), bg, dtype=int)
    y_src0 = max(0, -dy); y_src1 = min(H, H - dy)
    x_src0 = max(0, -dx); x_src1 = min(W, W - dx)
    y_dst0 = max(0,  dy); y_dst1 = min(H,  H + dy)
    x_dst0 = max(0,  dx); x_dst1 = min(W,  W + dx)
    if y_src0 < y_src1 and x_src0 < x_src1:
        B[y_dst0:y_dst1, x_dst0:x_dst1] = A[y_src0:y_src1, x_src0:x_src1]
    return B

def fits_rule_translation(train_pairs, search_limit=10):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for p in train_pairs:
        if grid_to_np(p["input"]).shape != grid_to_np(p["output"]).shape:
            return None
    dy_range = range(-search_limit, search_limit + 1)
    dx_range = range(-search_limit, search_limit + 1)
    for dy in dy_range:
        for dx in dx_range:
            ok = True
            for p in train_pairs:
                pred = shift_with_bg(p["input"], dy, dx, bg)
                if not np.array_equal(pred, grid_to_np(p["output"])):
                    ok = False
                    break
            if ok:
                return (dy, dx, bg)
    return None

def fits_rule_dihedral_translation(train_pairs, search_limit=10):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for name, T in DIHEDRAL:
        Ts = []
        shapes_ok = True
        for p in train_pairs:
            A = T(grid_to_np(p["input"]))
            B = grid_to_np(p["output"])
            if A.shape != B.shape:
                shapes_ok = False
                break
            Ts.append(A)
        if not shapes_ok:
            continue
        dy_range = range(-search_limit, search_limit + 1)
        dx_range = range(-search_limit, search_limit + 1)
        for dy in dy_range:
            for dx in dx_range:
                ok = True
                for A, p in zip(Ts, train_pairs):
                    B = grid_to_np(p["output"])
                    pred = shift_with_bg(A, dy, dx, bg)
                    if not np.array_equal(pred, B):
                        ok = False
                        break
                if ok:
                    return (name, T, dy, dx, bg)
    return None

def fits_rule_translation_colorperm(train_pairs, search_limit=10):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for p in train_pairs:
        if grid_to_np(p["input"]).shape != grid_to_np(p["output"]).shape:
            return None
    dy_range = range(-search_limit, search_limit + 1)
    dx_range = range(-search_limit, search_limit + 1)
    for dy in dy_range:
        for dx in dx_range:
            cmap = None
            ok = True
            for p in train_pairs:
                A_shift = shift_with_bg(p["input"], dy, dx, bg)
                m = infer_color_map(A_shift, p["output"])
                if m is None:
                    ok = False
                    break
                if cmap is None:
                    cmap = dict(m)
                else:
                    for k, v in m.items():
                        if k in cmap and cmap[k] != v:
                            ok = False
                            break
                        cmap[k] = v
                if not ok:
                    break
            if ok:
                return (dy, dx, bg, cmap)
    return None

def apply_shift_then_cmap(grid, dy, dx, bg, cmap):
    A = shift_with_bg(grid, dy, dx, bg)
    B = A.copy()
    for c in np.unique(A):
        B[A == c] = cmap.get(c, c)
    return B

# ------------- Morphology / components / bbox -------------
def foreground_mask(A, bg): return (A != bg)

def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return (ys.min(), ys.max()+1, xs.min(), xs.max()+1)

def crop_to_bbox(A, bg):
    A = grid_to_np(A)
    m = foreground_mask(A, bg)
    box = bbox_from_mask(m)
    if box is None:
        return np.full((1,1), bg, dtype=int)
    y0, y1, x0, x1 = box
    return A[y0:y1, x0:x1]

def keep_largest_component(A, bg):
    A = grid_to_np(A)
    H, W = A.shape
    fg = (A != bg)
    visited = np.zeros_like(fg, dtype=bool)
    best = []
    for y in range(H):
        for x in range(W):
            if fg[y,x] and not visited[y,x]:
                q = deque([(y,x)])
                visited[y,x] = True
                coords = [(y,x)]
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if 0 <= ny < H and 0 <= nx < W and fg[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx] = True
                            q.append((ny,nx))
                            coords.append((ny,nx))
                if len(coords) > len(best):
                    best = coords
    B = np.full_like(A, bg)
    for (y,x) in best:
        B[y,x] = A[y,x]
    return B

def center_pad_to_shape(A, out_shape, bg):
    A = grid_to_np(A)
    H2, W2 = out_shape
    H1, W1 = A.shape
    B = np.full((H2, W2), bg, dtype=int)
    y0 = (H2 - H1) // 2
    x0 = (W2 - W1) // 2
    y1 = y0 + H1
    x1 = x0 + W1
    Ay0 = max(0, -y0); Ax0 = max(0, -x0)
    Ay1 = min(H1, H2 - y0); Ax1 = min(W1, W2 - x0)
    By0 = max(0, y0);      Bx0 = max(0, x0)
    By1 = By0 + (Ay1 - Ay0); Bx1 = Bx0 + (Ax1 - Ax0)
    if Ay0 < Ay1 and Ax0 < Ax1:
        B[By0:By1, Bx0:Bx1] = A[Ay0:Ay1, Ax0:Ax0+ (Ax1 - Ax0)]
        # fix bug: previous slice used Ax0 twice; use Ax1 properly:
        B[By0:By1, Bx0:Bx1] = A[Ay0:Ay1, Ax0:Ax1]
    return B

# Strict center pad rule
def fits_rule_center_pad_strict(train_pairs):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for p in train_pairs:
        Ain = grid_to_np(p["input"])
        Aout = grid_to_np(p["output"])
        pred = center_pad_to_shape(Ain, Aout.shape, bg)
        if not np.array_equal(pred, Aout):
            return None
    out_shapes = {tuple(grid_to_np(p["output"]).shape) for p in train_pairs}
    if len(out_shapes) != 1:
        return None
    target_shape = next(iter(out_shapes))
    return (target_shape, bg)

# Crop bbox (plain)
def fits_rule_crop_bbox(train_pairs):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for p in train_pairs:
        Ain = grid_to_np(p["input"])
        Aout = grid_to_np(p["output"])
        if not np.array_equal(crop_to_bbox(Ain, bg), Aout):
            return None
    return bg

# Keep largest CC (plain)
def fits_rule_keep_largest_cc(train_pairs):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for p in train_pairs:
        Ain = grid_to_np(p["input"])
        Aout = grid_to_np(p["output"])
        pred = keep_largest_component(Ain, bg)
        if pred.shape != Aout.shape or not np.array_equal(pred, Aout):
            return None
    return bg

# Binary mask recolor
def fits_rule_binary_mask_recolor(train_pairs):
    bg_in  = Counter([border_mode_color(grid_to_np(p["input"]))  for p in train_pairs]).most_common(1)[0][0]
    bg_out = Counter([border_mode_color(grid_to_np(p["output"])) for p in train_pairs]).most_common(1)[0][0]
    fg_color = None
    for p in train_pairs:
        Ain = grid_to_np(p["input"])
        Aout = grid_to_np(p["output"])
        pal = palette(Aout)
        if len(pal) > 2:
            return None
        cand = [c for c in pal if c != bg_out]
        this_fg = cand[0] if cand else bg_out
        if not np.array_equal((Ain != bg_in), (Aout == this_fg)):
            return None
        if fg_color is None:
            fg_color = this_fg
        elif fg_color != this_fg:
            return None
    return (bg_in, bg_out, fg_color)

def solver_binary_mask_recolor(bg_in, bg_out, fg_color):
    def solve(g):
        A = grid_to_np(g)
        B = np.full_like(A, bg_out)
        B[A != bg_in] = fg_color
        return B
    return solve

# Dihedral -> CropBBox
def fits_rule_dihedral_cropbbox(train_pairs):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for name, T in DIHEDRAL:
        ok = True
        for p in train_pairs:
            Ain = T(grid_to_np(p["input"]))
            Aout = grid_to_np(p["output"])
            pred = crop_to_bbox(Ain, bg)
            if pred.shape != Aout.shape or not np.array_equal(pred, Aout):
                ok = False
                break
        if ok:
            return (name, T, bg)
    return None

# Background swap
def fits_rule_background_swap(train_pairs):
    bg_in  = Counter([border_mode_color(grid_to_np(p["input"]))  for p in train_pairs]).most_common(1)[0][0]
    bg_out = Counter([border_mode_color(grid_to_np(p["output"])) for p in train_pairs]).most_common(1)[0][0]
    for p in train_pairs:
        A = grid_to_np(p["input"])
        B = grid_to_np(p["output"])
        if A.shape != B.shape:
            return None
        C = A.copy()
        C[A == bg_in] = bg_out
        if not np.array_equal(C, B):
            return None
    return (bg_in, bg_out)

def solver_background_swap(bg_in, bg_out):
    def solve(g):
        A = grid_to_np(g)
        B = A.copy()
        B[A == bg_in] = bg_out
        return B
    return solve

# LCC then color perm
def fits_rule_lcc_then_colorperm(train_pairs):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    cmap = None
    for p in train_pairs:
        A = keep_largest_component(grid_to_np(p["input"]), bg)
        B = grid_to_np(p["output"])
        m = infer_color_map(A, B)
        if m is None:
            return None
        if cmap is None:
            cmap = dict(m)
        else:
            for k,v in m.items():
                if k in cmap and cmap[k] != v:
                    return None
                cmap[k] = v
    return (bg, cmap)

def apply_lcc_then_cmap(grid, bg, cmap):
    A = keep_largest_component(grid_to_np(grid), bg)
    for c in np.unique(A):
        A[A == c] = cmap.get(c, c)
    return A

# Per-color translation (single consistent shift)
def infer_shift_by_color(inp, out):
    A = grid_to_np(inp)
    B = grid_to_np(out)
    if A.shape != B.shape:
        return None
    colors = np.unique(np.concatenate([A.flatten(), B.flatten()]))
    shifts = []
    for c in colors:
        posA = np.argwhere(A == c)
        posB = np.argwhere(B == c)
        if len(posA) == 0 and len(posB) == 0:
            continue
        if len(posA) != len(posB):
            return None
        if len(posA) == 0:
            continue
        dy_dx = set()
        for (ya,xa),(yb,xb) in zip(sorted(map(tuple,posA.tolist())),
                                   sorted(map(tuple,posB.tolist()))):
            dy_dx.add((yb-ya, xb-xa))
        if len(dy_dx) != 1:
            return None
        shifts.append(dy_dx.pop())
    if not shifts:
        return None
    dy, dx = shifts[0]
    for s in shifts[1:]:
        if s != (dy, dx):
            return None
    return (dy, dx)

def fits_rule_percolor_translation(train_pairs):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    shift = None
    for p in train_pairs:
        s = infer_shift_by_color(p["input"], p["output"])
        if s is None:
            return None
        if shift is None:
            shift = s
        elif shift != s:
            return None
    return (shift[0], shift[1], bg)

def solver_percolor_translation(dy, dx, bg):
    def solve(g):
        return shift_with_bg(g, dy, dx, bg)
    return solve

# ------------- Block downscale (majority) -------------
def is_divisible_shape(A, B):
    h1, w1 = A.shape
    h2, w2 = B.shape
    return (h1 % h2 == 0) and (w1 % w2 == 0)

def block_majority_downscale(A, factor_y, factor_x):
    A = grid_to_np(A)
    H, W = A.shape
    h = H // factor_y
    w = W // factor_x
    out = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            block = A[i*factor_y:(i+1)*factor_y, j*factor_x:(j+1)*factor_x].flatten()
            try:
                out[i, j] = mode(block.tolist())
            except:
                vals, counts = np.unique(block, return_counts=True)
                out[i, j] = vals[np.argmax(counts)]
    return out

def fits_rule_block_downscale_majority(train_pairs):
    fy, fx = None, None
    for p in train_pairs:
        Ain = grid_to_np(p["input"])
        Aout = grid_to_np(p["output"])
        if not is_divisible_shape(Ain, Aout):
            return None
        fyi = Ain.shape[0] // Aout.shape[0]
        fxi = Ain.shape[1] // Aout.shape[1]
        if fy is None:
            fy, fx = fyi, fxi
        elif (fy, fx) != (fyi, fxi):
            return None
        pred = block_majority_downscale(Ain, fy, fx)
        if not np.array_equal(pred, Aout):
            return None
    return (fy, fx)

def solver_block_downscale_majority(fy, fx):
    def solve(g):
        return block_majority_downscale(grid_to_np(g), fy, fx)
    return solve

# ------------- Tiling (repeat motif) -------------
def is_integer_tiling(inp, out):
    A = grid_to_np(inp); B = grid_to_np(out)
    H1, W1 = A.shape; H2, W2 = B.shape
    if H2 % H1 != 0 or W2 % W1 != 0:
        return None
    ty, tx = H2 // H1, W2 // W1
    tiled = np.tile(A, (ty, tx))
    if np.array_equal(tiled, B):
        return (ty, tx)
    return None

def fits_rule_tiling(train_pairs):
    til = None
    for p in train_pairs:
        t = is_integer_tiling(p["input"], p["output"])
        if t is None:
            return None
        if til is None:
            til = t
        elif til != t:
            return None
    return til

def solver_tiling(ty, tx):
    def solve(g):
        A = grid_to_np(g)
        return np.tile(A, (ty, tx))
    return solve

# ------------- Outline extraction -------------
def compute_outline(A, bg):
    A = grid_to_np(A)
    H, W = A.shape
    B = np.full_like(A, bg)
    for y in range(H):
        for x in range(W):
            c = A[y, x]
            if c == bg:
                continue
            # if any neighbor differs -> edge
            for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0 <= ny < H and 0 <= nx < W and A[ny,nx] != c:
                    B[y,x] = c
                    break
    return B

def fits_rule_outline(train_pairs):
    bg = Counter([infer_bg_color_from_output(p["output"]) for p in train_pairs]).most_common(1)[0][0]
    for p in train_pairs:
        Ain = grid_to_np(p["input"])
        Aout = grid_to_np(p["output"])
        pred = compute_outline(Ain, bg)
        if not np.array_equal(pred, Aout):
            return None
    return bg

def solver_outline(bg):
    def solve(g):
        return compute_outline(grid_to_np(g), bg)
    return solve

# ------------- Rule hit diagnostics -------------
try:
    RULE_HITS
except NameError:
    RULE_HITS = Counter()

def mark_hit(name):
    try:
        RULE_HITS[name] += 1
    except Exception:
        pass

# ------------- Solver selection (ordered) -------------
def solve_task_from_train(task):
    train_pairs = task["train"]

    # 0) Identity
    if fits_rule_identity(train_pairs):
        mark_hit("identity")
        def solver(g): return rule_identity(g)
        return solver

    # 1) Dihedral only
    d = fits_rule_dihedral(train_pairs)
    if d is not None:
        name, T = d
        mark_hit("dihedral")
        def solver(g): return T(grid_to_np(g))
        return solver

    # 2) Dihedral + color permutation
    dc = fits_rule_dihedral_colorperm(train_pairs)
    if dc is not None:
        name, T, cmap = dc
        mark_hit("dihedral_colorperm")
        def solver(g): return apply_T_then_cmap(g, T, cmap)
        return solver

    # 3) Pure translation
    tr = fits_rule_translation(train_pairs, search_limit=10)
    if tr is not None:
        dy, dx, bg = tr
        mark_hit("translation")
        def solver(g): return shift_with_bg(g, dy, dx, bg)
        return solver

    # 4) Dihedral + translation
    dtr = fits_rule_dihedral_translation(train_pairs, search_limit=10)
    if dtr is not None:
        name, T, dy, dx, bg = dtr
        mark_hit("dihedral_translation")
        def solver(g): return shift_with_bg(T(grid_to_np(g)), dy, dx, bg)
        return solver

    # 5) Translation + color perm
    trc = fits_rule_translation_colorperm(train_pairs, search_limit=10)
    if trc is not None:
        dy, dx, bg, cmap = trc
        mark_hit("translation_colorperm")
        def solver(g): return apply_shift_then_cmap(g, dy, dx, bg, cmap)
        return solver

    # 6) Dihedral -> CropBBox
    dcb = fits_rule_dihedral_cropbbox(train_pairs)
    if dcb is not None:
        name, T, bg = dcb
        mark_hit("dihedral_cropbbox")
        def solver(g): return crop_to_bbox(T(grid_to_np(g)), bg)
        return solver

    # 7) Background swap
    bgs = fits_rule_background_swap(train_pairs)
    if bgs is not None:
        bg_in, bg_out = bgs
        mark_hit("background_swap")
        return solver_background_swap(bg_in, bg_out)

    # 8) LCC -> ColorPerm
    lcccp = fits_rule_lcc_then_colorperm(train_pairs)
    if lcccp is not None:
        bg, cmap = lcccp
        mark_hit("lcc_then_colorperm")
        def solver(g): return apply_lcc_then_cmap(g, bg, cmap)
        return solver

    # 9) Per-color translation
    pct = fits_rule_percolor_translation(train_pairs)
    if pct is not None:
        dy, dx, bg = pct
        mark_hit("percolor_translation")
        return solver_percolor_translation(dy, dx, bg)

    # 10) Crop bbox (plain)
    bg = fits_rule_crop_bbox(train_pairs)
    if bg is not None:
        mark_hit("crop_bbox")
        def solver(g): return crop_to_bbox(grid_to_np(g), bg)
        return solver

    # 11) Keep largest CC (plain)
    bg = fits_rule_keep_largest_cc(train_pairs)
    if bg is not None:
        mark_hit("keep_largest_cc")
        def solver(g): return keep_largest_component(grid_to_np(g), bg)
        return solver

    # 12) Binary mask recolor
    bm = fits_rule_binary_mask_recolor(train_pairs)
    if bm is not None:
        bg_in, bg_out, fg_color = bm
        mark_hit("binary_mask_recolor")
        def solver(g): return solver_binary_mask_recolor(bg_in, bg_out, fg_color)(g)
        return solver

    # 13) Center pad (strict)
    cp = fits_rule_center_pad_strict(train_pairs)
    if cp is not None:
        target_shape, bg = cp
        mark_hit("center_pad_strict")
        def solver(g, ts=target_shape, b=bg):
            return center_pad_to_shape(grid_to_np(g), ts, b)
        return solver

    # 14) Block downscale majority
    bd = fits_rule_block_downscale_majority(train_pairs)
    if bd is not None:
        fy, fx = bd
        mark_hit("block_downscale_majority")
        return solver_block_downscale_majority(fy, fx)

    # 15) Tiling
    til = fits_rule_tiling(train_pairs)
    if til is not None:
        ty, tx = til
        mark_hit("tiling")
        return solver_tiling(ty, tx)

    # 16) Outline extraction
    ol = fits_rule_outline(train_pairs)
    if ol is not None:
        bg = ol
        mark_hit("outline")
        return solver_outline(bg)

    # FINAL fallback — always callable
    mark_hit("fallback_identity")
    def solver(g):
        return grid_to_np(g)
    return solver

# ------------- Solutions coercion -------------
def coerce_solutions_to_map(solutions, challenges):
    if isinstance(solutions, dict):
        return solutions
    if isinstance(solutions, list) and len(solutions) > 0 and isinstance(solutions[0], dict) and 'task_id' in solutions[0]:
        return {s['task_id']: s for s in solutions}
    if isinstance(solutions, list):
        sol_map = {}
        for i, tid in enumerate(challenges.keys()):
            if i < len(solutions):
                sol_map[tid] = solutions[i]
        return sol_map
    raise TypeError("Unrecognized solutions structure")

def extract_true_tests(sol_record):
    if isinstance(sol_record, dict):
        if 'test' in sol_record:
            return sol_record['test']
        if 'output' in sol_record:
            return sol_record['output']
    return sol_record

train_sol_map = coerce_solutions_to_map(train_sol, train_ch)
eval_sol_map  = coerce_solutions_to_map(eval_sol,  eval_ch)

# ------------- Scoring -------------
def exact_match(A, B):
    A = grid_to_np(A); B = grid_to_np(B)
    return A.shape == B.shape and np.array_equal(A, B)

def predict_test_grids(task, solver):
    preds = []
    if not callable(solver):
        solver = lambda g: grid_to_np(g)
    for t in task["test"]:
        try:
            pred = solver(t["input"])
        except Exception:
            pred = grid_to_np(t["input"])
        preds.append(pred.astype(int).tolist())
    return preds

def score_split(challenges, solutions_map):
    total = 0
    correct = 0
    for tid, task in challenges.items():
        if tid not in solutions_map:
            continue
        solver = solve_task_from_train(task)
        preds = predict_test_grids(task, solver)
        true_outs = extract_true_tests(solutions_map[tid])
        for p, gt in zip(preds, true_outs):
            total += 1
            if exact_match(p, gt):
                correct += 1
    acc = (correct / total) if total else 0.0
    return correct, total, acc

# ------------- Run scoring -------------
train_correct, train_total, train_acc = score_split(train_ch, train_sol_map)
eval_correct,  eval_total,  eval_acc  = score_split(eval_ch,  eval_sol_map)

print(f"\nTRAIN: {train_correct}/{train_total} = {train_acc:.3f}")
print(f"EVAL : {eval_correct}/{eval_total} = {eval_acc:.3f}")
print("Top rule hits:", RULE_HITS.most_common(20))

# ------------- Build & save submission -------------
def build_submission(challenges):
    sub = {}
    for tid, task in challenges.items():
        solver = solve_task_from_train(task)
        preds = predict_test_grids(task, solver)
        sub[tid] = {"test": preds}
    return sub

# Save to /kaggle/working/submission.json
submission = build_submission(test_ch)
OUTPUT_PATH = "/kaggle/working/submission.json"
with open(OUTPUT_PATH, "w") as f:
    json.dump(submission, f)

print(f"Submission written to: {OUTPUT_PATH}")



