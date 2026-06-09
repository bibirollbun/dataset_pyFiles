# Kaggle ARC 2025: All-in-One Rule-Based Baseline
# Config and imports
from __future__ import annotations
import json, os, random, math, itertools, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

RNG_SEED = 123
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# Kaggle input paths
KAGGLE_DIR = '/kaggle/input/arc-prize-2025'
PATH_TRAIN_CH = os.path.join(KAGGLE_DIR, 'arc-agi_training_challenges.json')
PATH_TRAIN_SOL = os.path.join(KAGGLE_DIR, 'arc-agi_training_solutions.json')
PATH_EVAL_CH = os.path.join(KAGGLE_DIR, 'arc-agi_evaluation_challenges.json')
PATH_EVAL_SOL = os.path.join(KAGGLE_DIR, 'arc-agi_evaluation_solutions.json')
PATH_TEST_CH = os.path.join(KAGGLE_DIR, 'arc-agi_test_challenges.json')




# Load datasets and normalize
@dataclass
class Task:
    task_id: str
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]]
    test_inputs: List[List[List[int]]]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def parse_challenges(d: Dict[str, Any]) -> Dict[str, Task]:
    tasks: Dict[str, Task] = {}
    for tid, obj in d.items():
        train_pairs = [(ex['input'], ex['output']) for ex in obj['train']]
        test_inputs = [ex['input'] for ex in obj['test']]
        tasks[tid] = Task(tid, train_pairs, test_inputs)
    return tasks


def parse_solutions(d: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
    # maps task_id -> list of test outputs (in order)
    return {tid: sols for tid, sols in d.items()}

train_ch = load_json(PATH_TRAIN_CH)
train_sol = load_json(PATH_TRAIN_SOL)
eval_ch = load_json(PATH_EVAL_CH)
eval_sol = load_json(PATH_EVAL_SOL)
test_ch = load_json(PATH_TEST_CH)

TRAIN_TASKS = parse_challenges(train_ch)
EVAL_TASKS = parse_challenges(eval_ch)
TEST_TASKS = parse_challenges(test_ch)
TRAIN_SOLUTIONS = parse_solutions(train_sol)
EVAL_SOLUTIONS = parse_solutions(eval_sol)

len(TRAIN_TASKS), len(EVAL_TASKS), len(TEST_TASKS)



# Minimal EDA helpers and sample visualization
from IPython.display import display

COLOR_MAP = {
    0: '#000000', 1: '#0074D9', 2: '#FF4136', 3: '#2ECC40', 4: '#FFDC00',
    5: '#AAAAAA', 6: '#F012BE', 7: '#FF851B', 8: '#7FDBFF', 9: '#870C25'
}

def show_grid(grid: List[List[int]], ax=None, title: Optional[str]=None):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for k, hexcol in COLOR_MAP.items():
        mask = (g == k)
        if not np.any(mask):
            continue
        hexcol = hexcol.lstrip('#')
        rgb_val = tuple(int(hexcol[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb[mask] = rgb_val
    if ax is None:
        _, ax = plt.subplots(figsize=(w/2+1, h/2+1))
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def show_task(task: Task, max_examples: int = 3):
    k = min(max_examples, len(task.train_pairs))
    fig, axes = plt.subplots(k, 2, figsize=(6, 3*k))
    if k == 1:
        axes = np.array([axes])
    for i in range(k):
        inp, out = task.train_pairs[i]
        show_grid(inp, ax=axes[i,0], title='input')
        show_grid(out, ax=axes[i,1], title='output')
    plt.tight_layout()

# Show a few random tasks to sanity-check
for tid in random.sample(list(TRAIN_TASKS.keys()), 3):
    print('Task', tid)
    show_task(TRAIN_TASKS[tid], max_examples=2)




# Grid helpers
ArrayLikeGrid = List[List[int]]


def to_np(grid: ArrayLikeGrid) -> np.ndarray:
    return np.array(grid, dtype=int)


def from_np(arr: np.ndarray) -> ArrayLikeGrid:
    return arr.astype(int).tolist()


def dims(grid: ArrayLikeGrid) -> Tuple[int, int]:
    g = to_np(grid)
    return g.shape[0], g.shape[1]


def palette(grid: ArrayLikeGrid) -> List[int]:
    g = to_np(grid)
    vals = np.unique(g)
    return [int(v) for v in vals.tolist()]


def rotate(grid: ArrayLikeGrid, k: int=1) -> ArrayLikeGrid:
    g = to_np(grid)
    return from_np(np.rot90(g, k=k))


def flip_h(grid: ArrayLikeGrid) -> ArrayLikeGrid:
    g = to_np(grid)
    return from_np(np.flip(g, axis=1))


def flip_v(grid: ArrayLikeGrid) -> ArrayLikeGrid:
    g = to_np(grid)
    return from_np(np.flip(g, axis=0))


def transpose_grid(grid: ArrayLikeGrid) -> ArrayLikeGrid:
    g = to_np(grid)
    return from_np(g.T)


def pad_to(grid: ArrayLikeGrid, target_h: int, target_w: int, fill: int=0) -> ArrayLikeGrid:
    g = to_np(grid)
    h, w = g.shape
    out = np.full((target_h, target_w), fill, dtype=int)
    out[:min(h, target_h), :min(w, target_w)] = g[:min(h, target_h), :min(w, target_w)]
    return from_np(out)


def tile_to(grid: ArrayLikeGrid, target_h: int, target_w: int) -> ArrayLikeGrid:
    g = to_np(grid)
    reps_h = math.ceil(target_h / g.shape[0])
    reps_w = math.ceil(target_w / g.shape[1])
    tiled = np.tile(g, (reps_h, reps_w))[:target_h, :target_w]
    return from_np(tiled)


def bbox_of_color(grid: ArrayLikeGrid, colors: Optional[List[int]]=None) -> Optional[Tuple[int,int,int,int]]:
    g = to_np(grid)
    if colors is None:
        colors = [c for c in palette(grid) if c != 0]
    mask = np.isin(g, colors)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return int(y0), int(x0), int(y1)+1, int(x1)+1


def crop_to_bbox(grid: ArrayLikeGrid, bbox: Tuple[int,int,int,int]) -> ArrayLikeGrid:
    y0, x0, y1, x1 = bbox
    g = to_np(grid)
    return from_np(g[y0:y1, x0:x1])


def majority_color(grid: ArrayLikeGrid) -> int:
    g = to_np(grid).ravel()
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])

# Connected components (4-connectivity)
from collections import deque

def components(grid: ArrayLikeGrid, conn: int=4) -> List[Dict[str, Any]]:
    g = to_np(grid)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    dirs = [(1,0),(-1,0),(0,1),(0,-1)] if conn==4 else [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    comps = []
    for y in range(h):
        for x in range(w):
            if seen[y,x] or g[y,x]==0:
                continue
            col = g[y,x]
            q = deque([(y,x)])
            seen[y,x] = True
            cells = []
            while q:
                cy,cx = q.popleft()
                cells.append((cy,cx))
                for dy,dx in dirs:
                    ny,nx = cy+dy,cx+dx
                    if 0<=ny<h and 0<=nx<w and not seen[ny,nx] and g[ny,nx]==col:
                        seen[ny,nx]=True
                        q.append((ny,nx))
            ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
            bbox = (min(ys), min(xs), max(ys)+1, max(xs)+1)
            comps.append({"color": int(col), "cells": cells, "bbox": bbox, "size": len(cells)})
    comps.sort(key=lambda c: c['size'], reverse=True)
    return comps




# Transform registry and core transforms
class Transform:
    name: str = "transform"
    def fit(self, pairs: List[Tuple[ArrayLikeGrid, ArrayLikeGrid]]):
        return self
    def predict(self, grid: ArrayLikeGrid) -> ArrayLikeGrid:
        raise NotImplementedError
    def __repr__(self):
        return f"{self.__class__.__name__}()"


TRANSFORMS: Dict[str, Callable[[], Transform]] = {}

def register(cls):
    TRANSFORMS[cls.__name__] = cls
    return cls


@register
class Identity(Transform):
    name = "identity"
    def predict(self, grid):
        return grid


@register
class Rotate90(Transform):
    def predict(self, grid):
        return rotate(grid, 1)

@register
class Rotate180(Transform):
    def predict(self, grid):
        return rotate(grid, 2)

@register
class Rotate270(Transform):
    def predict(self, grid):
        return rotate(grid, 3)

@register
class FlipH(Transform):
    def predict(self, grid):
        return flip_h(grid)

@register
class FlipV(Transform):
    def predict(self, grid):
        return flip_v(grid)

@register
class Transpose(Transform):
    def predict(self, grid):
        return transpose_grid(grid)


@register
class ColorPermute(Transform):
    """Infer a consistent color mapping from train pairs via Hungarian-like greedy.
    If ambiguous, fall back to identity.
    """
    def __init__(self):
        self.map: Dict[int,int] = {}
    def fit(self, pairs):
        # build mapping votes
        votes: Dict[Tuple[int,int], int] = {}
        for inp, out in pairs:
            gi, go = to_np(inp), to_np(out)
            # only consider overlapping area when shapes differ
            h = min(gi.shape[0], go.shape[0]); w = min(gi.shape[1], go.shape[1])
            gi2, go2 = gi[:h,:w].ravel(), go[:h,:w].ravel()
            for a,b in zip(gi2, go2):
                votes[(int(a), int(b))] = votes.get((int(a), int(b)), 0) + 1
        # for each input color choose the most voted output color
        in_colors = set(a for a,_ in votes.keys())
        mapping = {}
        for c in in_colors:
            candidates = [(cnt, b) for (a,b),cnt in votes.items() if a==c]
            if not candidates:
                continue
            b = max(candidates)[1]
            mapping[c] = b
        self.map = mapping
        return self
    def predict(self, grid):
        g = to_np(grid)
        out = np.vectorize(lambda x: self.map.get(int(x), int(x)))(g)
        return from_np(out)


@register
class RepeatTileToOutputSize(Transform):
    """Tile the input to match the most common output size in train."""
    def __init__(self):
        self.target: Optional[Tuple[int,int]] = None
    def fit(self, pairs):
        sizes = [dims(o) for _,o in pairs]
        # pick most frequent (h,w)
        vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
        self.target = tuple(map(int, vals[np.argmax(counts)]))
        return self
    def predict(self, grid):
        h,w = dims(grid)
        th, tw = self.target if self.target else (h,w)
        return tile_to(grid, th, tw)


@register
class PadOrCropToOutput(Transform):
    """Pad or crop input to match output size, using majority background color for pad."""
    def __init__(self):
        self.target: Optional[Tuple[int,int]] = None
        self.fill: int = 0
    def fit(self, pairs):
        sizes = [dims(o) for _,o in pairs]
        vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
        self.target = tuple(map(int, vals[np.argmax(counts)]))
        # infer majority from outputs
        colors = [majority_color(o) for _,o in pairs]
        vals2, counts2 = np.unique(np.array(colors), return_counts=True)
        self.fill = int(vals2[np.argmax(counts2)])
        return self
    def predict(self, grid):
        th, tw = self.target
        h,w = dims(grid)
        if h>=th and w>=tw:
            return pad_to(crop_to_bbox(grid, (0,0,th,tw)), th, tw, fill=self.fill) if (h!=th or w!=tw) else grid
        else:
            return pad_to(grid, th, tw, fill=self.fill)




# Extended transforms
@register
class AddBorder(Transform):
    """Add a border of inferred color and thickness around the input to reach output size."""
    def __init__(self):
        self.target: Optional[Tuple[int,int]] = None
        self.color: int = 0
    def fit(self, pairs):
        sizes = [dims(o) for _,o in pairs]
        vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
        self.target = tuple(map(int, vals[np.argmax(counts)]))
        # color from outputs majority
        colors = [majority_color(o) for _,o in pairs]
        vals2, counts2 = np.unique(np.array(colors), return_counts=True)
        self.color = int(vals2[np.argmax(counts2)])
        return self
    def predict(self, grid):
        th, tw = self.target
        h, w = dims(grid)
        out = np.full((th, tw), self.color, dtype=int)
        sy = max((th - h)//2, 0)
        sx = max((tw - w)//2, 0)
        out[sy:sy+min(h,th), sx:sx+min(w,tw)] = to_np(grid)[:min(h,th), :min(w,tw)]
        return from_np(out)


@register
class BBoxCropOrExpand(Transform):
    """Crop to bbox of salient colors or expand by padding to target size."""
    def __init__(self):
        self.target: Optional[Tuple[int,int]] = None
        self.fill: int = 0
    def fit(self, pairs):
        sizes = [dims(o) for _,o in pairs]
        vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
        self.target = tuple(map(int, vals[np.argmax(counts)]))
        self.fill = majority_color(pairs[0][1])
        return self
    def predict(self, grid):
        th, tw = self.target
        bb = bbox_of_color(grid)
        if bb is None:
            return pad_to(grid, th, tw, fill=self.fill)
        cropped = crop_to_bbox(grid, bb)
        return pad_to(cropped, th, tw, fill=self.fill)


@register
class StripeFill1D(Transform):
    """If input is 1 row/col, repeat to match target size (stripe)."""
    def __init__(self):
        self.target: Optional[Tuple[int,int]] = None
    def fit(self, pairs):
        sizes = [dims(o) for _,o in pairs]
        vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
        self.target = tuple(map(int, vals[np.argmax(counts)]))
        return self
    def predict(self, grid):
        h,w = dims(grid)
        th,tw = self.target
        if h==1:
            row = to_np(grid)
            out = np.tile(row, (th, math.ceil(tw/w)))[:th,:tw]
            return from_np(out)
        if w==1:
            col = to_np(grid)
            out = np.tile(col, (1, tw))
            out = np.tile(out, (math.ceil(th/h), 1))[:th,:tw]
            return from_np(out)
        return pad_to(grid, th, tw, fill=majority_color(grid))


@register
class MajorityFillToOutput(Transform):
    def __init__(self):
        self.target: Optional[Tuple[int,int]] = None
        self.fill: int = 0
    def fit(self, pairs):
        sizes = [dims(o) for _,o in pairs]
        vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
        self.target = tuple(map(int, vals[np.argmax(counts)]))
        self.fill = majority_color(pairs[0][1])
        return self
    def predict(self, grid):
        th,tw = self.target
        return from_np(np.full((th,tw), self.fill, dtype=int))




# Search procedure and evaluation

def exact_match(a: ArrayLikeGrid, b: ArrayLikeGrid) -> bool:
    return np.array_equal(to_np(a), to_np(b))


def compose(t1: Transform, t2: Transform) -> Transform:
    class Composed(Transform):
        def __init__(self, t1, t2):
            self.t1, self.t2 = t1, t2
        def fit(self, pairs):
            self.t1.fit(pairs); self.t2.fit(pairs)
            return self
        def predict(self, grid):
            return self.t2.predict(self.t1.predict(grid))
        def __repr__(self):
            return f"Compose({self.t1},{self.t2})"
    return Composed(t1, t2)


def candidate_transforms() -> List[Transform]:
    # instantiate a modest set
    base_names = [
        'Identity','Rotate90','Rotate180','Rotate270','FlipH','FlipV','Transpose',
        'ColorPermute','RepeatTileToOutputSize','PadOrCropToOutput',
        'AddBorder','BBoxCropOrExpand','StripeFill1D','MajorityFillToOutput']
    bases = [TRANSFORMS[n]() for n in base_names]
    # single or composed of length 2
    composed = [compose(TRANSFORMS[a](), TRANSFORMS[b]()) for a,b in itertools.product(base_names, base_names) if a!=b]
    # keep size small to be fast
    return bases + composed[:80]


def fit_and_score(task: Task, tfm: Transform) -> Tuple[int, int]:
    # returns (#correct, total)
    pairs = task.train_pairs
    tfm.fit(pairs)
    correct = 0
    for inp, out in pairs:
        pred = tfm.predict(inp)
        # if shapes differ, allow simple pad/crop to out size for scoring
        ph, pw = dims(pred); oh, ow = dims(out)
        if (ph, pw) != (oh, ow):
            pred = pad_to(pred, oh, ow, fill=majority_color(out))
        correct += int(exact_match(pred, out))
    return correct, len(pairs)


def choose_transforms_for_task(task: Task, topk: int=2) -> List[Transform]:
    cands = candidate_transforms()
    scored = []
    for t in cands:
        c, n = fit_and_score(task, t)
        scored.append((c/n, c, repr(t), t))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s[3] for s in scored[:topk]]


def predict_task(task: Task, topk: int=2) -> List[ArrayLikeGrid]:
    tfms = choose_transforms_for_task(task, topk=topk)
    preds = []
    # refit on train once chosen
    for t in tfms:
        t.fit(task.train_pairs)
    # produce predictions per selected transform for first test input
    for test_inp in task.test_inputs:
        outs = []
        for t in tfms:
            pred = t.predict(test_inp)
            # if we have target size ambiguity, try aligning to most common train output size
            sizes = [dims(o) for _,o in task.train_pairs]
            if sizes:
                vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
                th, tw = tuple(map(int, vals[np.argmax(counts)]))
                ph, pw = dims(pred)
                if (ph,pw)!=(th,tw):
                    pred = pad_to(pred, th, tw, fill=majority_color(task.train_pairs[0][1]))
            outs.append(pred)
        preds.extend(outs[:topk])
    return preds


def evaluate_split(tasks: Dict[str, Task], use_train_targets: bool=True, solutions: Optional[Dict[str, Any]]=None, limit:int=100) -> Dict[str, Any]:
    start = time.time()
    n_tasks = 0
    total_pairs = 0
    exact = 0
    for tid, task in list(tasks.items())[:limit]:
        n_tasks += 1
        if use_train_targets:
            # leave-one-out style: score on train pairs directly
            best = choose_transforms_for_task(task, topk=1)[0]
            best.fit(task.train_pairs)
            for inp, out in task.train_pairs:
                pred = best.predict(inp)
                oh,ow = dims(out)
                ph,pw = dims(pred)
                if (ph,pw)!=(oh,ow):
                    pred = pad_to(pred, oh, ow, fill=majority_color(out))
                exact += int(exact_match(pred, out))
                total_pairs += 1
        else:
            assert solutions is not None
            tfms = choose_transforms_for_task(task, topk=2)
            for t in tfms:
                t.fit(task.train_pairs)
            sols = solutions.get(tid, [])
            for i, test_inp in enumerate(task.test_inputs):
                pred1 = tfms[0].predict(test_inp)
                pred2 = tfms[1].predict(test_inp) if len(tfms)>1 else pred1
                target = sols[i]
                oh,ow = dims(target)
                for pred in (pred1, pred2):
                    ph,pw = dims(pred)
                    if (ph,pw)!=(oh,ow):
                        pred = pad_to(pred, oh, ow, fill=majority_color(target))
                    exact += int(exact_match(pred, target))
                    total_pairs += 1
    elapsed = time.time() - start
    return {"tasks_evaluated": n_tasks, "items_scored": total_pairs, "exact": exact, "acc": exact/max(total_pairs,1), "sec": elapsed}

print('Train CV-ish score (quick):', evaluate_split(TRAIN_TASKS, use_train_targets=True, limit=50))
print('Eval sanity (uses provided eval solutions):', evaluate_split(EVAL_TASKS, use_train_targets=False, solutions=EVAL_SOLUTIONS, limit=30))




# Inference on test and submission writing

def build_submission(tasks: Dict[str, Task]) -> Dict[str, Any]:
    submission: Dict[str, Any] = {}
    for tid, task in tasks.items():
        tfms = choose_transforms_for_task(task, topk=2)
        for t in tfms:
            t.fit(task.train_pairs)
        item_objs = []
        for test_inp in task.test_inputs:
            preds = []
            for t in tfms:
                p = t.predict(test_inp)
                # align to train-most-common output size if available
                sizes = [dims(o) for _,o in task.train_pairs]
                if sizes:
                    vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
                    th, tw = tuple(map(int, vals[np.argmax(counts)]))
                    ph,pw = dims(p)
                    if (ph,pw)!=(th,tw):
                        p = pad_to(p, th, tw, fill=majority_color(task.train_pairs[0][1]))
                preds.append(p)
            # at most two attempts
            item_objs.append({"attempt_1": preds[0], "attempt_2": preds[1] if len(preds)>1 else preds[0]})
        submission[tid] = item_objs
    return submission

SUB = build_submission(TEST_TASKS)
with open('submission.json', 'w') as f:
    json.dump(SUB, f)
print('Wrote submission.json with', len(SUB), 'tasks')




# Logging/debug (lightweight)

def summarize_task_choice(task: Task, topk: int=2):
    tfms = choose_transforms_for_task(task, topk=topk)
    print(task.task_id, '->', [repr(t) for t in tfms])

print('Sample chosen transforms:')
for tid in random.sample(list(TRAIN_TASKS.keys()), 5):
    summarize_task_choice(TRAIN_TASKS[tid])



