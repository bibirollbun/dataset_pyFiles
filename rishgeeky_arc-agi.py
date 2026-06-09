#!/usr/bin/env python3
"""
ARC Prize 2025 – Hybrid Baseline Solver
Updated July 2025
• Adds colour-permutation, uniform-fill and translation
• Keeps the original geometric transforms and composite search
• Structure kept to a single file for easy Kaggle submission
• Ready-to-run: will create submission.json in the proper format
"""

import json, os, time
from itertools import permutations, product
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm.auto import tqdm


# ════════════════════════
# 0- CONFIG
# ════════════════════════
@dataclass
class Config:
    data_dir: str = "/kaggle/input/arc-prize-2025"
    test_file: str = "arc-agi_test_challenges.json"
    max_depth: int = 2             # for composite geometric search
    verbose: bool = True


CFG = Config()


# ════════════════════════
# 1- I/O HELPERS
# ════════════════════════
def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] Bad JSON: {path} – {e}")
        return {}


def np_grid(lst: List[List[int]]) -> np.ndarray:
    return np.asarray(lst, dtype=np.int8)


def lst_grid(arr: np.ndarray) -> List[List[int]]:
    return arr.astype(int).tolist()


def valid_grid(arr: np.ndarray) -> bool:
    return (
        arr.size > 0
        and arr.shape[0] <= 30
        and arr.shape[1] <= 30
        and arr.min() >= 0
        and arr.max() <= 9
    )


# ════════════════════════
# 2- BASIC GEOMETRIC OPS
# ════════════════════════
class Geo:
    @staticmethod
    def identity(g): return g.copy()
    @staticmethod
    def rot90(g): return np.rot90(g, k=-1)
    @staticmethod
    def rot180(g): return np.rot90(g, k=2)
    @staticmethod
    def rot270(g): return np.rot90(g, k=1)
    @staticmethod
    def flip_h(g): return np.fliplr(g)
    @staticmethod
    def flip_v(g): return np.flipud(g)
    @staticmethod
    def transpose(g): return g.T
    @staticmethod
    def anti_transpose(g): return np.rot90(g.T, k=2)


BASIC_TRANSFORMS = [
    ("identity", Geo.identity),
    ("rot90", Geo.rot90),
    ("rot180", Geo.rot180),
    ("rot270", Geo.rot270),
    ("flip_h", Geo.flip_h),
    ("flip_v", Geo.flip_v),
    ("transpose", Geo.transpose),
    ("anti_transpose", Geo.anti_transpose),
]


# ════════════════════════
# 3- COLOUR UTILITIES
# ════════════════════════
def apply_colour_map(g: np.ndarray, cmap: Dict[int, int]) -> np.ndarray:
    out = g.copy()
    for k, v in cmap.items():
        out[g == k] = v
    return out


def bijective_colour_map(
    g_in: np.ndarray, g_out: np.ndarray
) -> Optional[Dict[int, int]]:
    if g_in.shape != g_out.shape:
        return None
    in_cols = list(np.unique(g_in))
    out_cols = list(np.unique(g_out))
    if len(in_cols) != len(out_cols):
        return None
    for perm in permutations(out_cols):
        test_map = dict(zip(in_cols, perm))
        if np.array_equal(apply_colour_map(g_in, test_map), g_out):
            return test_map
    return None


# ════════════════════════
# 4- TRANSLATION SEARCH
# ════════════════════════
def find_shift(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[int, int]]:
    if a.shape != b.shape:
        return None
    h, w = a.shape
    for dy in range(-h + 1, h):
        for dx in range(-w + 1, w):
            if np.array_equal(np.roll(np.roll(a, dy, 0), dx, 1), b):
                return dy, dx
    return None


# ════════════════════════
# 5- PATTERN MATCHER
# ════════════════════════
class PatternMatcher:
    def __init__(self):
        self.transform = None   # callable
        self.name = "identity"

    # ––––– master routine –––––
    def fit(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        if self._uniform_fill(pairs):
            return
        if self._colour_perm(pairs):
            return
        if self._translation(pairs):
            return
        if self._single_geo(pairs):
            return
        self._composite_geo(pairs)  # last resort

    # 5.1 uniform-fill (whole grid becomes one colour)
    def _uniform_fill(self, pairs) -> bool:
        out_cols = {tuple(np.unique(o)) for _, o in pairs}
        if len(out_cols) == 1 and len(next(iter(out_cols))) == 1:
            c = next(iter(out_cols))[0]
            self.transform = lambda g, col=c: np.full_like(g, col)
            self.name = f"fill_{c}"
            return True
        return False

    # 5.2 colour permutation
    def _colour_perm(self, pairs) -> bool:
        cmap = bijective_colour_map(*pairs[0])
        if cmap and all(
            np.array_equal(apply_colour_map(g, cmap), o) for g, o in pairs
        ):
            self.transform = lambda g, m=cmap: apply_colour_map(g, m)
            self.name = f"colour_perm_{cmap}"
            return True
        return False

    # 5.3 global translation
    def _translation(self, pairs) -> bool:
        shift = find_shift(*pairs[0])
        if shift and all(find_shift(g, o) == shift for g, o in pairs):
            dy, dx = shift
            self.transform = lambda g, y=dy, x=dx: np.roll(
                np.roll(g, y, 0), x, 1
            )
            self.name = f"shift_{dy}_{dx}"
            return True
        return False

    # 5.4 single geometric transform
    def _single_geo(self, pairs) -> bool:
        for name, func in BASIC_TRANSFORMS:
            if all(np.array_equal(func(g), o) for g, o in pairs):
                self.transform, self.name = func, name
                return True
        return False

    # 5.5 two-step composites (depth ≤ CFG.max_depth)
    def _composite_geo(self, pairs) -> bool:
        for depth in range(2, CFG.max_depth + 1):
            for combo in product(BASIC_TRANSFORMS, repeat=depth):
                funcs = [f for _, f in combo]
                def chained(g, fs=funcs):
                    for f in fs:
                        g = f(g)
                    return g
                if all(np.array_equal(chained(g), o) for g, o in pairs):
                    self.transform = chained
                    self.name = "+".join(n for n, _ in combo)
                    return True
        # fallback to identity
        self.transform, self.name = Geo.identity, "identity"
        return False


# ════════════════════════
# 6- SOLVER
# ════════════════════════
class ARCSolver:
    def __init__(self):
        self.matcher = PatternMatcher()

    def solve(self, task: Dict[str, Any]) -> List[List[List[int]]]:
        pairs = [
            (np_grid(t["input"]), np_grid(t["output"]))
            for t in task.get("train", [])
        ]
        self.matcher.fit(pairs)
        preds = []
        for test_case in task.get("test", []):
            g = np_grid(test_case["input"])
            try:
                out = self.matcher.transform(g)
                if not valid_grid(out):
                    out = g
            except Exception:
                out = g
            preds.append(lst_grid(out))
        return preds


# ════════════════════════
# 7- SUBMISSION HANDLER
# ════════════════════════
def make_submission():
    path = os.path.join(CFG.data_dir, CFG.test_file)
    tasks = load_json(path)
    if not tasks:
        return
    solver = ARCSolver()
    submission = {}
    solved = 0
    t0 = time.time()

    for tid, task in tqdm(tasks.items(), desc="tasks"):
        preds = solver.solve(task)
        submission[tid] = [
            {"attempt_1": p, "attempt_2": p} for p in preds
        ]
        if solver.matcher.name != "identity":
            solved += 1

    with open("submission.json", "w") as f:
        json.dump(submission, f, separators=(",", ":"))

    dt = time.time() - t0
    print(
        f"\n✅ submission.json written – {solved}/{len(tasks)} "
        f"non-identity solutions in {dt:.1f}s"
    )
    print("Top transform types included:", solver.matcher.name)


# ════════════════════════
# 8- QUICK SELF-TEST
# ════════════════════════
if __name__ == "__main__":
    # tiny sanity check
    toy = {
        "train": [
            {
                "input": [[1, 0], [0, 1]],
                "output": [[0, 1], [1, 0]],
            }
        ],
        "test": [{"input": [[1, 1], [0, 0]]}],
    }
    print("▶ Self-test:", ARCSolver().solve(toy)[0])
    # create real submission when running on Kaggle
    if os.path.exists(CFG.data_dir):
        make_submission()


