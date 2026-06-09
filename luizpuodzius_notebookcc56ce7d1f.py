import os, json, sys
sys.path.append('/kaggle/input/fcp-solver')
from fcp_arc_solver_v_2 import FCP_ARC_Solver, generate_submission

path = '/kaggle/input/arc-prize-2025'
challenge_path = f'{path}/arc-agi_test_challenges.json'

with open(challenge_path, 'r') as f:
    challenges = json.load(f)

print(f"Loaded {len(challenges)} ARC tasks.")



import sys
sys.path.append('/kaggle/input/fcp-solver')

from fcp_arc_solver_v_2 import FCP_ARC_Solver, generate_submission



%%writefile /kaggle/working/fcp_arc_solver_v_2.py
# ---# fcp_arc_solver_v_2.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from collections import defaultdict, Counter

# -----------------------------
# Utilities
# -----------------------------

Coord = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)

def to_np(grid: Any) -> np.ndarray:
    g = np.array(grid, dtype=int)
    assert g.ndim == 2
    return g

def bbox_from_pixels(pixels: List[Coord]) -> BBox:
    rs, cs = zip(*pixels)
    return (min(rs), min(cs), max(rs), max(cs))

# -----------------------------
# Field Objects & Extraction
# -----------------------------

@dataclass
class FieldObject:
    color: int
    pixels: List[Coord]
    bbox: BBox
    mask: np.ndarray  # tight bbox mask (1=object)

    @property
    def size(self) -> int:
        return len(self.pixels)

class FCPExtractor:
    def __init__(self, background: Optional[int] = 0):
        self.background = background

    def extract(self, grid: np.ndarray) -> Dict[str, Any]:
        return {
            "objects": self._extract_objects(grid),
            "background": self._infer_background(grid),
            "symmetries": self._find_symmetries(grid),
            "patterns": self._identify_patterns(grid),
        }

    def _infer_background(self, grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _extract_objects(self, grid: np.ndarray) -> List[FieldObject]:
        H, W = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        bg = self._infer_background(grid) if self.background is None else self.background
        objs: List[FieldObject] = []

        def neigh(r: int, c: int):
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W:
                    yield rr, cc

        for r in range(H):
            for c in range(W):
                if visited[r,c]:
                    continue
                color = int(grid[r,c])
                if color == bg:
                    visited[r,c] = True
                    continue
                stack = [(r,c)]
                comp: List[Coord] = []
                visited[r,c] = True
                while stack:
                    rr, cc = stack.pop()
                    if grid[rr,cc] != color:
                        continue
                    comp.append((rr,cc))
                    for nr,nc in neigh(rr,cc):
                        if not visited[nr,nc] and grid[nr,nc] == color:
                            visited[nr,nc] = True
                            stack.append((nr,nc))
                if comp:
                    r0,c0,r1,c1 = bbox_from_pixels(comp)
                    mask = np.zeros((r1-r0+1, c1-c0+1), dtype=int)
                    for pr,pc in comp:
                        mask[pr-r0, pc-c0] = 1
                    objs.append(FieldObject(color=color, pixels=comp, bbox=(r0,c0,r1,c1), mask=mask))
        objs.sort(key=lambda o: (-o.size, o.bbox))
        return objs

    def _find_symmetries(self, grid: np.ndarray) -> Dict[str, bool]:
        sym: Dict[str,bool] = {}
        if np.array_equal(grid, np.fliplr(grid)): sym["mirror_lr"] = True
        if np.array_equal(grid, np.flipud(grid)): sym["mirror_ud"] = True
        if np.array_equal(grid, np.rot90(grid,1)): sym["rot90"] = True
        if np.array_equal(grid, np.rot90(grid,2)): sym["rot180"] = True
        if np.array_equal(grid, np.rot90(grid,3)): sym["rot270"] = True
        bg = self._infer_background(grid)
        fg = np.where(grid == bg, -1, grid)
        if np.array_equal(fg, np.fliplr(fg)): sym["mirror_lr_fg"] = True
        if np.array_equal(fg, np.flipud(fg)): sym["mirror_ud_fg"] = True
        return sym

    def _identify_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        def period_1d(arr: np.ndarray) -> Optional[int]:
            n = len(arr)
            for p in range(1, n//2+1):
                if n % p == 0 and all((arr[i] == arr[i % p]) for i in range(n)):
                    return p
            return None
        return {
            "row_periods": [period_1d(row) for row in grid],
            "col_periods": [period_1d(col) for col in grid.T],
        }

# -----------------------------
# Transformations (Field-Coherent)
# -----------------------------

class FCPTransforms:
    @staticmethod
    def color_map(grid: np.ndarray, mapping: Dict[int,int]) -> np.ndarray:
        if not mapping: return grid.copy()
        out = grid.copy()
        for k,v in mapping.items():
            out[grid == k] = v
        return out

    @staticmethod
    def mirror(grid: np.ndarray, axis: str) -> np.ndarray:
        if axis == "lr": return np.fliplr(grid)
        if axis == "ud": return np.flipud(grid)
        raise ValueError("axis must be 'lr' or 'ud'")

    @staticmethod
    def rotate(grid: np.ndarray, k: int) -> np.ndarray:
        return np.rot90(grid, k % 4)

    @staticmethod
    def crop_to_fg(grid: np.ndarray) -> np.ndarray:
        bg = int(np.bincount(grid.ravel()).argmax())
        ys, xs = np.where(grid != bg)
        if len(ys) == 0: return grid.copy()
        r0, r1 = ys.min(), ys.max()
        c0, c1 = xs.min(), xs.max()
        return grid[r0:r1+1, c0:c1+1]

    @staticmethod
    def center_largest(grid: np.ndarray) -> np.ndarray:
        ex = FCPExtractor(background=None)
        info = ex.extract(grid)
        objs = info["objects"]; bg = info["background"]
        if not objs: return grid.copy()
        big = objs[0]
        H,W = grid.shape
        out = np.full((H,W), bg, dtype=int)
        (r0,c0,r1,c1) = big.bbox
        h,w = (r1-r0+1), (c1-c0+1)
        rr,cc = (H-h)//2, (W-w)//2
        for pr,pc in big.pixels:
            out[rr + (pr-r0), cc + (pc-c0)] = big.color
        return out

    @staticmethod
    def keep_border(grid: np.ndarray, width: int, fill_bg: bool=True, bg_color: Optional[int]=None) -> np.ndarray:
        H,W = grid.shape
        out = grid.copy()
        max_w = max(0, min(H,W)//2)
        w = max(0, min(int(width), max_w))
        if w == 0 or H == 0 or W == 0: return out
        if fill_bg and (bg_color is not None) and (w < H and w < W):
            out[w:H-w, w:W-w] = bg_color
        return out

    @staticmethod
    def tile_fill(grid: np.ndarray, tile_h: int, tile_w: int, template: Optional[np.ndarray]=None) -> np.ndarray:
        H,W = grid.shape
        template = grid[:tile_h,:tile_w].copy() if template is None else template
        out = np.zeros_like(grid)
        for r0 in range(0, H, tile_h):
            for c0 in range(0, W, tile_w):
                r1 = min(r0+tile_h, H)
                c1 = min(c0+tile_w, W)
                out[r0:r1, c0:c1] = template[:(r1-r0), :(c1-c0)]
        return out

    # Pixel-wise upscaling (each pixel becomes a filled block)
    @staticmethod
    def repeat(grid: np.ndarray, factor: int=3) -> np.ndarray:
        factor = int(factor)
        if factor <= 1: return grid.copy()
        return np.kron(grid, np.ones((factor, factor), dtype=int))

    @staticmethod
    def resize_repeat(grid: np.ndarray, rh: int, rw: int) -> np.ndarray:
        rh, rw = int(rh), int(rw)
        if rh <= 1 and rw <= 1: return grid.copy()
        return np.kron(grid, np.ones((max(1,rh), max(1,rw)), dtype=int))

    @staticmethod
    def repeat_fg(grid: np.ndarray, factor: int=3) -> np.ndarray:
        bg = int(np.bincount(grid.ravel()).argmax())
        mask = (grid != bg).astype(int)
        up_mask = np.kron(mask, np.ones((factor,factor), dtype=int))
        up_grid = np.kron(grid, np.ones((factor,factor), dtype=int))
        out = np.full_like(up_grid, bg)
        out[up_mask == 1] = up_grid[up_mask == 1]
        return out

# -----------------------------
# Learner
# -----------------------------

class FCPLearner:
    def __init__(self):
        self.rule: Optional[Dict[str,Any]] = None
        self.second_best: Optional[Dict[str,Any]] = None
        self.debug_scored: List[Dict[str,Any]] = []

    def learn(self, train_pairs: List[Dict[str,np.ndarray]]) -> Dict[str,Any]:
        candidates: List[Dict[str,Any]] = []

        # Baselines / heuristics
        for w in (1,2,3):
            candidates.append({"type":"keep_border", "width":w, "fill_bg":True})
        candidates.append({"type":"crop_fg"})
        candidates.append({"type":"center_largest"})
        candidates.append({"type":"identity"})

        # Color map (no geometry)
        cmap_plain = self._learn_color_map_under_geom(train_pairs, k=0, mirror=None)
        candidates.append({"type":"geom_cmap", "k":0, "mirror":None, "mapping":cmap_plain})

        # Geom + colormap
        for k,m in [(0,None),(1,None),(2,None),(3,None),(0,'lr'),(0,'ud')]:
            mapping = self._learn_color_map_under_geom(train_pairs, k=k, mirror=m)
            candidates.append({"type":"geom_cmap", "k":k, "mirror":m, "mapping":mapping})

        # Pure geom
        for ax in ("lr","ud"):
            candidates.append({"type":"mirror", "axis":ax})
        for k in (1,2,3):
            candidates.append({"type":"rotate", "k":k})

        # Pixel-wise scaling
        for f in (2,3):
            candidates.append({"type":"repeat", "factor":f})
            candidates.append({"type":"repeat_fg", "factor":f})

        # Learn integer ratio from training examples
        ratio_rule = self._learn_resize_rule(train_pairs)
        if ratio_rule:
            candidates.append(ratio_rule)

        # Color map then scale
        for f in (2,3):
            if cmap_plain:
                candidates.append({"type":"color_repeat", "factor":f, "mapping":cmap_plain})

        # Score
        scored: List[Tuple[int,int,Dict[str,Any]]] = []
        for cand in candidates:
            s, ok = self._score_rule(train_pairs, cand)
            scored.append((s, ok, cand))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        valid = [t for t in scored if t[1] > 0]
        pool = valid if valid else scored
        self.debug_scored = [{"score":s, "ok_pairs":ok, "rule":cand} for (s,ok,cand) in pool]

        best = pool[0][2] if pool else {"type":"identity"}
        second = pool[1][2] if len(pool)>1 else {"type":"identity"}
        self.rule = best
        self.second_best = second
        return best

    def apply(self, x: np.ndarray, rule: Optional[Dict[str,Any]]=None) -> np.ndarray:
        rule = rule or self.rule or {"type":"identity"}
        t = rule["type"]

        if t == "identity": return x.copy()
        if t == "crop_fg": return FCPTransforms.crop_to_fg(x)
        if t == "center_largest": return FCPTransforms.center_largest(x)
        if t == "mirror": return FCPTransforms.mirror(x, rule["axis"])
        if t == "rotate": return FCPTransforms.rotate(x, rule["k"])
        if t == "geom_cmap":
            y = FCPTransforms.rotate(x, rule.get("k",0)) if rule.get("k",0) else x
            if rule.get("mirror"): y = FCPTransforms.mirror(y, rule["mirror"])
            return FCPTransforms.color_map(y, rule.get("mapping", {}))
        if t == "tile_fill":
            return FCPTransforms.tile_fill(x, rule["tile_h"], rule["tile_w"], rule.get("template"))
        if t == "keep_border":
            bg = rule.get("bg")
            if bg is None: bg = int(np.bincount(x.ravel()).argmax())
            return FCPTransforms.keep_border(x, rule["width"], fill_bg=rule.get("fill_bg",True), bg_color=bg)
        if t == "repeat": return FCPTransforms.repeat(x, rule["factor"])
        if t == "repeat_fg": return FCPTransforms.repeat_fg(x, rule["factor"])
        if t == "resize_repeat": return FCPTransforms.resize_repeat(x, rule["rh"], rule["rw"])
        if t == "color_repeat":
            y = FCPTransforms.color_map(x, rule.get("mapping", {}))
            return FCPTransforms.repeat(y, rule["factor"])

        raise ValueError(f"Unknown rule {t}")

    # --- helpers ---

    def _score_rule(self, train_pairs: List[Dict[str,np.ndarray]], rule: Dict[str,Any]) -> Tuple[int,int]:
        score = 0
        ok = 0
        for ex in train_pairs:
            x, y = ex["input"], ex["output"]
            yhat = self.apply(x, rule)
            if yhat.shape != y.shape:
                continue
            ok += 1
            score += int(np.array_equal(yhat, y))
        return score, ok

    def _learn_tile_rule(self, train_pairs: List[Dict[str,np.ndarray]]) -> Optional[Dict[str,Any]]:
        tile_candidates = []
        for ex in train_pairs:
            x, y = ex["input"], ex["output"]
            if x.shape != y.shape: continue
            H,W = y.shape
            hs = [h for h in range(1, min(6,H)+1) if H % h == 0]
            ws = [w for w in range(1, min(6,W)+1) if W % w == 0]
            for h in hs:
                for w in ws:
                    tmpl = y[:h,:w]
                    tiled = FCPTransforms.tile_fill(np.zeros_like(y), h, w, tmpl)
                    if np.array_equal(tiled, y):
                        tile_candidates.append((h,w,tmpl.copy()))
        if not tile_candidates:
            return None
        (bh,bw), _ = Counter((h,w) for (h,w,_) in tile_candidates).most_common(1)[0]
        tmpl = next(t for (h,w,t) in tile_candidates if h==bh and w==bw)
        return {"type":"tile_fill", "tile_h":int(bh), "tile_w":int(bw), "template":tmpl}

    def _learn_color_map_under_geom(self, train_pairs: List[Dict[str,np.ndarray]], k:int=0, mirror:Optional[str]=None) -> Dict[int,int]:
        counts: Dict[int, Dict[int,int]] = defaultdict(lambda: defaultdict(int))
        for ex in train_pairs:
            x, y = ex["input"], ex["output"]
            xg = FCPTransforms.rotate(x, k) if k else x
            if mirror: xg = FCPTransforms.mirror(xg, mirror)
            if xg.shape != y.shape: continue
            for a,b in zip(xg.ravel(), y.ravel()):
                counts[int(a)][int(b)] += 1
        if not counts: return {}
        triples = []
        for a,row in counts.items():
            for b,c in row.items():
                if c>0: triples.append((c,a,b))
        triples.sort(reverse=True)
        used_in, used_out, mapping = set(), set(), {}
        for c,a,b in triples:
            if a in used_in or b in used_out: continue
            mapping[a] = b
            used_in.add(a); used_out.add(b)
        return {int(kk): int(vv) for kk,vv in mapping.items()}

    def _learn_resize_rule(self, train_pairs: List[Dict[str,np.ndarray]]) -> Optional[Dict[str,Any]]:
        ratios = []
        for ex in train_pairs:
            x, y = ex["input"], ex["output"]
            if x.size == 0 or y.size == 0: continue
            rh = y.shape[0] / x.shape[0]
            rw = y.shape[1] / x.shape[1]
            if rh.is_integer() and rw.is_integer():
                ratios.append((int(rh), int(rw)))
        if not ratios: return None
        (rh,rw), _ = Counter(ratios).most_common(1)[0]
        if rh == 1 and rw == 1: return None
        return {"type":"resize_repeat", "rh":int(rh), "rw":int(rw)}

# -----------------------------
# Solver wrapper + submission
# -----------------------------

class FCP_ARC_Solver:
    def __init__(self):
        self.extractor = FCPExtractor(background=None)
        self.learner = FCPLearner()

    def detect_field_structure(self, grid: np.ndarray) -> Dict[str,Any]:
        return self.extractor.extract(grid)

    def _learn_transformation(self, train_pairs: List[Dict[str,np.ndarray]]) -> Dict[str,Any]:
        return self.learner.learn(train_pairs)

    def _apply_transformation(self, grid: np.ndarray, transformations: Dict[str,Any], variant:int=0) -> np.ndarray:
        return self.learner.apply(grid, transformations)

    def solve_task(self, train_pairs: List[Dict[str,np.ndarray]], test_input: np.ndarray) -> List[np.ndarray]:
        rule = self._learn_transformation(train_pairs)
        attempts: List[np.ndarray] = []
        # attempt 1: best rule
        attempts.append(self._apply_transformation(test_input, rule, variant=0))
        # attempt 2: meaningful diversity
        alt = getattr(self.learner, "second_best", None)
        if alt is not None and alt.get("type") != rule.get("type"):
            attempts.append(self.learner.apply(test_input, alt))
        else:
            attempts.append(FCPTransforms.rotate(test_input, 1))
        return attempts

def generate_submission(solver: FCP_ARC_Solver, test_challenges: Dict[str,Any]) -> Dict[str,Any]:
    sub: Dict[str,Any] = {}
    for task_id, task in test_challenges.items():
        train_pairs = [{"input": to_np(p["input"]), "output": to_np(p["output"])} for p in task.get("train", [])]
        test_input = to_np(task["test"][0]["input"])
        attempts = solver.solve_task(train_pairs, test_input)
        sub[task_id] = [{"attempt_1": attempts[0].tolist()}, {"attempt_2": attempts[1].tolist()}]
    return sub

# -----------------------------
# CLI (write submission.json when this file is run directly)
# -----------------------------
if __name__ == "__main__":
    import argparse, os, glob
    from textwrap import dedent

    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", type=str, required=False, help="Path to ARC challenges JSON")
    ap.add_argument("--out", type=str, default="submission.json")
    args = ap.parse_args()

    challenge_path = args.challenges
    if challenge_path is None:
        candidates = [
            "arc-agi_training_challenges.json",
            "arc-agi-training-challenges.json",
            "arc-agi-test-challenges.json",
            "arc-agi_test_challenges.json",
            "arc-agi-evaluation-challenges.json",
            "arc-agi_evaluation_challenges.json",
        ]
        found = []
        for pat in candidates + ["*.json"]:
            found.extend(glob.glob(pat))
        priority = [p for p in candidates if os.path.exists(p)]
        if priority:
            challenge_path = priority[0]
        else:
            found_arc = [p for p in found if "arc" in p.lower() and ("challenge" in p.lower() or "eval" in p.lower())]
            challenge_path = found_arc[0] if found_arc else None

    if challenge_path is None:
        print(dedent("""
        [FCP] No --challenges path provided and auto-detect failed.
        Place your ARC JSON in this folder and name it one of:
          - arc-agi_training_challenges.json
          - arc-agi-test-challenges.json
          - arc-agi-evaluation-challenges.json
        Or pass the full path with --challenges.
        """))
        raise SystemExit(2)

    with open(challenge_path, "r") as f:
        challenges = json.load(f)

    print(f"[FCP] Using challenges: {challenge_path}")
    solver = FCP_ARC_Solver()
    submission = generate_submission(solver, challenges)

    with open(args.out, "w") as f:
        json.dump(submission, f)
    print(f"Wrote {args.out} with {len(submission)} tasks.")
 --




import sys, json
sys.path.append('/kaggle/working')

from fcp_arc_solver_v_2 import FCP_ARC_Solver, generate_submission

DATA_DIR = '/kaggle/input/arc-prize-2025'
with open(f'{DATA_DIR}/arc-agi_test_challenges.json', 'r') as f:
    challenges = json.load(f)

solver = FCP_ARC_Solver()
submission = generate_submission(solver, challenges)

with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)
print("Saved /kaggle/working/submission.json")



!ls /kaggle/input/fcp-solver



import sys, json, numpy as np
sys.path.append('/kaggle/input/fcp-solver')  # your solver dataset
from fcp_arc_solver_v_2 import FCP_ARC_Solver, generate_submission

with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json','r') as f:
    challenges = json.load(f)

solver = FCP_ARC_Solver()
submission = generate_submission(solver, challenges)

with open('/kaggle/working/submission.json','w') as f:
    json.dump(submission, f)

print('Wrote /kaggle/working/submission.json with', len(submission), 'tasks.')



# show the first task id and the shapes of both attempts
first_task = next(iter(submission))
a1 = np.array(submission[first_task][0]['attempt_1'])
a2 = np.array(submission[first_task][1]['attempt_2'])
print('First task:', first_task, '| attempt_1:', a1.shape, '| attempt_2:', a2.shape)
print(a1)  # optional: remove if the grid is large



import json, numpy as np

with open('/kaggle/working/submission.json','r') as f:
    sub = json.load(f)

problems = []
for tid, attempts in sub.items():
    if not isinstance(attempts, list) or len(attempts) != 2:
        problems.append((tid, 'needs exactly two attempts')); continue
    try:
        a1 = np.array(attempts[0]['attempt_1'], dtype=int)
        a2 = np.array(attempts[1]['attempt_2'], dtype=int)
    except Exception as e:
        problems.append((tid, f'bad array format: {e}')); continue
    if a1.ndim != 2 or a2.ndim != 2 or a1.size == 0 or a2.size == 0:
        problems.append((tid, 'attempts must be non-empty 2D arrays')); continue
    if not (np.all((a1>=0)&(a1<=9)) and np.all((a2>=0)&(a2<=9))):
        problems.append((tid, 'values must be integers 0â€“9'))
        
print('Tasks:', len(sub), '| issues:', len(problems))
print(problems[:5])


