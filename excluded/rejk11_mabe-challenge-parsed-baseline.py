# MABe Mouse Behavior Detection - Advanced Baseline
# This notebook contains an advanced baseline model for mouse behavior detection
# Run with: !python baseline.py --mode validate --prior-scope mixed --eps 12 -vv

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

import polars as pl
import numpy as np
from tqdm.auto import tqdm


# ========================
# Configuration
# ========================
IS_KAGGLE=False
if os.path.exists("/kaggle/working"):
    IS_KAGGLE=True 
@dataclass(frozen=True)
class Config:
    data_root: Path = Path(os.getenv("MABE_DATA_ROOT", default= "/kaggle/input/MABe-mouse-behavior-detection" if IS_KAGGLE else "./MABe-mouse-behavior-detection"))
    submission_file: str = os.getenv("MABE_SUBMISSION", "submission.csv")
    row_id_col: str = os.getenv("MABE_ROW_ID_COL", "row_id")

    @property
    def train_csv(self) -> Path: return self.data_root / "train.csv"
    @property
    def test_csv(self) -> Path: return self.data_root / "test.csv"
    @property
    def train_annot_dir(self) -> Path: return self.data_root / "train_annotation"
    @property
    def train_track_dir(self) -> Path: return self.data_root / "train_tracking"
    @property
    def test_track_dir(self) -> Path: return self.data_root / "test_tracking"

    @property
    def submission_schema(self) -> Dict[str, pl.DataType]:
        return {
            "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
            "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
        }

    @property
    def solution_schema(self) -> Dict[str, pl.DataType]:
        return {
            "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
            "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
            "lab_id": pl.Utf8, "behaviors_labeled": pl.Utf8,
        }

logger = logging.getLogger(__name__)

class HostVisibleError(Exception): 
    pass

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)


# ========================
# Utility Functions & Validators
# ========================

def safe_json_loads(s: Optional[str]) -> List[str]:
    if s is None: return []
    if isinstance(s, list): return [str(x) for x in s]
    if not isinstance(s, str): return []
    s = s.strip()
    if not s: return []
    try:
        return json.loads(s)
    except Exception:
        try: return json.loads(s.replace("'", '"'))
        except Exception: return []

def validate_schema(df: pl.DataFrame, schema: Dict[str, pl.DataType], name: str) -> pl.DataFrame:
    missing = set(schema.keys()) - set(df.columns)
    if missing: raise ValueError(f"{name} is missing columns: {missing}")
    casts = [pl.col(col).cast(dtype) for col, dtype in schema.items() if df[col].dtype != dtype]
    return df.with_columns(casts) if casts else df

def validate_frame_ranges(df: pl.DataFrame, name: str) -> None:
    if not (df["start_frame"] <= df["stop_frame"]).all():
        raise ValueError(f"{name}: start_frame > stop_frame detected")

def _norm_mouse_id(x: str | int) -> str:
    s = str(x)
    return s if s.startswith("mouse") else f"mouse{s}"

def _norm_triplet(agent: str | int, target: str | int, action: str) -> str:
    return f"{_norm_mouse_id(agent)},{_norm_mouse_id(target)},{action}"

def _range_frames(start: int, stop: int) -> Iterable[int]:
    return range(start, stop)  # [start, stop)


# Additional utility functions for intervals and allocation

def merge_intervals(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not intervals: return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s,e in intervals[1:]:
        ps,pe = merged[-1]
        if s <= pe: merged[-1] = (ps, max(pe, e))
        else: merged.append((s,e))
    return merged

def split_interval(s: int, e: int, parts: int) -> List[Tuple[int,int]]:
    if parts <= 1: return [(s,e)]
    L = e - s
    step = L // parts
    rem = L % parts
    out = []
    cur = s
    for i in range(parts):
        extra = 1 if i < rem else 0
        nxt = cur + step + extra
        out.append((cur, min(nxt, e)))
        cur = nxt
    return out

def largest_remainder_allocation(total: int, weights: List[float]) -> List[int]:
    if total <= 0 or not weights: return [0]*len(weights)
    s = sum(weights) or 1.0
    w = [x/s for x in weights]
    raw = [total*x for x in w]
    base = [int(v) for v in raw]
    remainder = total - sum(base)
    if remainder > 0:
        fr = sorted([(i, raw[i]-base[i]) for i in range(len(w))], key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            base[fr[i % len(w)][0]] += 1
    return base


# ========================
# Metrics (F-beta Score Calculation)
# ========================

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1.0) -> float:
    label_frames: Dict[str, Set[int]] = defaultdict(set)
    for row in lab_solution.to_dicts():
        label_frames[row["label_key"]].update(_range_frames(row["start_frame"], row["stop_frame"]))

    active_by_video: Dict[int, Set[str]] = {}
    for row in lab_solution.select(["video_id", "behaviors_labeled"]).unique().to_dicts():
        s: Set[str] = set()
        for item in safe_json_loads(row["behaviors_labeled"]):
            parts = [p.strip() for p in str(item).replace("'", "").split(",")]
            if len(parts) == 3:
                a, t, act = parts
                s.add(_norm_triplet(a, t, act))
        active_by_video[int(row["video_id"])] = s

    prediction_frames: Dict[str, Set[int]] = defaultdict(set)
    for video_id in lab_solution["video_id"].unique():
        active = active_by_video.get(int(video_id), set())
        predicted_mouse_pairs: Dict[str, Set[int]] = defaultdict(set)
        for row in lab_submission.filter(pl.col("video_id") == video_id).to_dicts():
            triple_norm = _norm_triplet(row["agent_id"], row["target_id"], row["action"])
            if triple_norm not in active:
                continue
            pred_key = row["prediction_key"]
            agent_target = f"{row['agent_id']},{row['target_id']}"
            new_frames = set(_range_frames(row["start_frame"], row["stop_frame"]))
            new_frames -= prediction_frames[pred_key]
            if predicted_mouse_pairs[agent_target] & new_frames:
                raise HostVisibleError("Multiple predictions for the same frame from one agent/target pair")
            prediction_frames[pred_key].update(new_frames)
            predicted_mouse_pairs[agent_target].update(new_frames)

    tps: Dict[str, int] = defaultdict(int)
    fns: Dict[str, int] = defaultdict(int)
    fps: Dict[str, int] = defaultdict(int)
    distinct_actions: Set[str] = set()

    for key, pred_frames in prediction_frames.items():
        action = key.split("_")[-1]
        distinct_actions.add(action)
        gt_frames = label_frames.get(key, set())
        tps[action] += len(pred_frames & gt_frames)
        fns[action] += len(gt_frames - pred_frames)
        fps[action] += len(pred_frames - gt_frames)

    for key, gt_frames in label_frames.items():
        action = key.split("_")[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(gt_frames)

    if not distinct_actions:
        return 0.0

    beta2 = beta * beta
    f_scores: List[float] = []
    for action in distinct_actions:
        tp, fn, fp = tps[action], fns[action], fps[action]
        denom = (1 + beta2) * tp + beta2 * fn + fp
        f_scores.append(0.0 if denom == 0 else (1 + beta2) * tp / denom)
    return sum(f_scores) / len(f_scores)


# Main F-beta score function

def mouse_fbeta(solution: pl.DataFrame, submission: pl.DataFrame, beta: float = 1.0, cfg: Optional[Config] = None) -> float:
    cfg = cfg or Config()
    solution = validate_schema(solution, cfg.solution_schema, "Solution")
    submission = validate_schema(submission, cfg.submission_schema, "Submission")
    validate_frame_ranges(solution, "Solution")
    validate_frame_ranges(submission, "Submission")

    solution_videos = solution["video_id"].unique()
    submission = submission.filter(pl.col("video_id").is_in(solution_videos))

    def add_key(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
        return df.with_columns(
            pl.concat_str(
                [
                    pl.col("video_id").cast(pl.Utf8),
                    pl.col("agent_id").cast(pl.Utf8),
                    pl.col("target_id").cast(pl.Utf8),
                    pl.col("action"),
                ],
                separator="_",
            ).alias(col_name)
        )

    solution = add_key(solution, "label_key")
    submission = add_key(submission, "prediction_key")

    lab_scores: List[float] = []
    for lab_id in solution["lab_id"].unique():
        lab_solution = solution.filter(pl.col("lab_id") == lab_id)
        lab_videos = lab_solution["video_id"].unique()
        lab_submission = submission.filter(pl.col("video_id").is_in(lab_videos))
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores) if lab_scores else 0.0

def score(solution: pl.DataFrame, submission: pl.DataFrame, row_id_column_name: str = "", beta: float = 1.0, cfg: Optional[Config] = None) -> float:
    if row_id_column_name:
        solution = solution.drop(row_id_column_name, strict=False)
        submission = submission.drop(row_id_column_name, strict=False)
    return mouse_fbeta(solution, submission, beta=beta, cfg=cfg)


# ========================
# Solution Building & Video Spans
# ========================

def create_solution_df(dataset: pl.DataFrame, cfg: Optional[Config] = None) -> pl.DataFrame:
    cfg = cfg or Config()
    records: List[pl.DataFrame] = []
    for row in tqdm(dataset.to_dicts(), total=len(dataset), desc="Building solution"):
        lab_id: str = row["lab_id"]
        if lab_id.startswith("MABe22"): continue
        video_id: int = row["video_id"]
        annot_path = cfg.train_annot_dir / lab_id / f"{video_id}.parquet"
        if not annot_path.exists():
            logger.warning("No annotations for %s", annot_path)
            continue
        try:
            annot = pl.read_parquet(annot_path).with_columns(
                [
                    pl.lit(lab_id).alias("lab_id"),
                    pl.lit(video_id).alias("video_id"),
                    pl.lit(row["behaviors_labeled"]).alias("behaviors_labeled"),
                    pl.concat_str([pl.lit("mouse"), pl.col("agent_id").cast(pl.Utf8)]).alias("agent_id"),
                    pl.concat_str([pl.lit("mouse"), pl.col("target_id").cast(pl.Utf8)]).alias("target_id"),
                ]
            )
            for col, dtype in (cfg.solution_schema).items():
                if col in annot.columns and annot[col].dtype != dtype:
                    annot = annot.with_columns(pl.col(col).cast(dtype))
            annot = annot.select([c for c in cfg.solution_schema.keys() if c in annot.columns])
            records.append(annot)
        except Exception as e:
            logger.error("Failed to load %s: %s", annot_path, e)
            continue
    if not records: raise ValueError("No annotation files loaded.")
    solution = pl.concat(records, how="vertical")
    solution = validate_schema(solution, cfg.solution_schema, "Solution")
    return solution


# Video spans builder

def build_video_spans(dataset: pl.DataFrame, split: str, cfg: Optional[Config] = None) -> Dict[int, Tuple[int,int]]:
    """
    Map video_id -> (min_frame, max_frame+1).
    """
    cfg = cfg or Config()
    track_dir = cfg.train_track_dir if split == "train" else cfg.test_track_dir
    spans: Dict[int, Tuple[int,int]] = {}
    for row in tqdm(dataset.to_dicts(), total=len(dataset), desc="Scanning spans"):
        lab_id = row["lab_id"]
        if lab_id.startswith("MABe22"): continue
        vid = row["video_id"]
        path = track_dir / lab_id / f"{vid}.parquet"
        if not path.exists(): continue
        try:
            df = pl.read_parquet(path).select(["video_frame"])
            s = int(df["video_frame"].min())
            e = int(df["video_frame"].max()) + 1
            spans[int(vid)] = (s,e)
        except Exception as e:
            logger.warning("Span read failed for %s: %s", path, e)
    return spans


# ========================
# Prior Computation (Duration & Timing)
# ========================

def compute_action_priors(solution: pl.DataFrame, eps: float = 1.0) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Returns:
      per_lab_weight: {lab: {action: weight_share}}
      global_weight: {action: weight_share}
      per_lab_med_dur: {lab: {action: median_duration_frames}}
      global_med_dur: {action: median_duration_frames}
    """
    sol = solution.with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("dur"))
    # shares
    by_lab = sol.group_by(["lab_id", "action"]).agg(pl.col("dur").sum().alias("dur_sum"))
    global_ = sol.group_by(["action"]).agg(pl.col("dur").sum().alias("dur_sum"))
    actions = set(global_["action"].to_list())

    per_lab_weight: Dict[str, Dict[str, float]] = defaultdict(dict)
    for lab in by_lab["lab_id"].unique():
        sub = by_lab.filter(pl.col("lab_id") == lab)
        dmap = {r["action"]: float(r["dur_sum"]) for r in sub.to_dicts()}
        for a in actions: dmap[a] = dmap.get(a, 0.0) + eps
        total = sum(dmap.values()) or 1.0
        per_lab_weight[str(lab)] = {a: dmap[a]/total for a in actions}

    gmap = {r["action"]: float(r["dur_sum"]) for r in global_.to_dicts()}
    for a in actions: gmap[a] = gmap.get(a, 0.0) + eps
    gtotal = sum(gmap.values()) or 1.0
    global_weight = {a: gmap[a]/gtotal for a in actions}

    # median durations
    med_by_lab = sol.group_by(["lab_id", "action"]).median().select(["lab_id","action","dur"])
    per_lab_med_dur: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in med_by_lab.to_dicts():
        per_lab_med_dur[str(r["lab_id"])][str(r["action"])] = int(r["dur"])
    med_global = sol.group_by(["action"]).median().select(["action","dur"])
    global_med_dur: Dict[str, int] = {r["action"]: int(r["dur"]) for r in med_global.to_dicts()}

    return per_lab_weight, global_weight, per_lab_med_dur, global_med_dur


# Timing priors computation

def compute_timing_priors(solution: pl.DataFrame, video_spans: Dict[int, Tuple[int,int]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Median start percentile per (lab, action) and global.
    start_pct = (start_frame - video_start)/(video_stop - video_start)
    """
    # attach start_pct
    def start_pct_func(row) -> float:
        vid = int(row["video_id"])
        if vid not in video_spans: return 0.5
        s,e = video_spans[vid]
        denom = max(1, e - s)
        return float(max(0, min(1, (int(row["start_frame"]) - s) / denom)))

    rows = []
    for r in solution.select(["lab_id","action","video_id","start_frame"]).to_dicts():
        rows.append({"lab_id": r["lab_id"], "action": r["action"], "start_pct": start_pct_func(r)})
    df = pl.DataFrame(rows)
    by_lab = df.group_by(["lab_id","action"]).median().select(["lab_id","action","start_pct"])
    per_lab: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in by_lab.to_dicts():
        per_lab[str(r["lab_id"])][str(r["action"])] = float(r["start_pct"])
    g = df.group_by(["action"]).median().select(["action","start_pct"])
    global_: Dict[str, float] = {r["action"]: float(r["start_pct"]) for r in g.to_dicts()}
    return per_lab, global_


# ========================
# Tracking Features → Windows
# ========================

def _strip_mouse_prefix(s: str | int) -> str:
    s = str(s)
    return s[5:] if s.startswith("mouse") else s

def _pair_features(df: pl.DataFrame, agent_raw: str, target_raw: str, downsample: int = 1) -> Optional[pl.DataFrame]:
    """
    Tính features cho một cặp (agent,target) bằng NumPy:
      - dist  = sqrt((ax-bx)^2 + (ay-by)^2)
      - rel_speed = speed_a - speed_b, với speed = sqrt(dx^2 + dy^2) khung-kề-khung
      - ddist = diff(dist)
    Trả về Polars DataFrame: ["frame","dist","rel_speed","ddist"] đã sort theo frame.
    downsample: lấy mỗi N khung (N>=1). Giá trị 2–3 giúp tăng tốc đáng kể.
    """
    # --- auto-detect schema ---
    frame_candidates = ["video_frame","frame","frame_idx"]
    id_candidates    = ["mouse_id","id","track_id","agent_id"]
    x_candidates     = ["x","x_pos","x_position","x_mm","centroid_x","cx"]
    y_candidates     = ["y","y_pos","y_position","y_mm","centroid_y","cy"]

    cols = set(df.columns)
    frame_col = next((c for c in frame_candidates if c in cols), None)
    id_col    = next((c for c in id_candidates    if c in cols), None)
    x_col     = next((c for c in x_candidates     if c in cols), None)
    y_col     = next((c for c in y_candidates     if c in cols), None)
    if not all([frame_col, id_col, x_col, y_col]):
        return None

    # --- chuẩn hoá ID ---
    a_id = _strip_mouse_prefix(agent_raw)
    t_id = _strip_mouse_prefix(target_raw)

    # --- lấy tối thiểu các cột cần thiết & chuyển sang pandas để dùng NumPy ---
    # (to_pandas trên 4 cột nhỏ rất nhanh; tránh join nhiều lần trong Polars)
    pdf = df.select([frame_col, id_col, x_col, y_col]).to_pandas()
    # ép kiểu an toàn
    pdf[frame_col] = pdf[frame_col].astype(np.int64, copy=False)
    pdf[id_col]    = pdf[id_col].astype(str, copy=False)

    a = pdf[pdf[id_col] == a_id].copy()
    b = pdf[pdf[id_col] == t_id].copy()
    if a.empty or b.empty:
        return None

    # ưu tiên một bản ghi / frame / mouse (nếu trùng) để merge gọn
    a.drop_duplicates(subset=[frame_col], keep="first", inplace=True)
    b.drop_duplicates(subset=[frame_col], keep="first", inplace=True)

    merged = a.merge(b, on=frame_col, how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        return None
    merged.sort_values(frame_col, inplace=True)

    # --- NumPy vectors ---
    ax = merged[f"{x_col}_a"].to_numpy(dtype=np.float64, copy=False)
    ay = merged[f"{y_col}_a"].to_numpy(dtype=np.float64, copy=False)
    bx = merged[f"{x_col}_b"].to_numpy(dtype=np.float64, copy=False)
    by = merged[f"{y_col}_b"].to_numpy(dtype=np.float64, copy=False)
    frames = merged[frame_col].to_numpy(dtype=np.int64, copy=False)

    # downsample nếu cần
    if downsample > 1:
        sl = slice(0, None, int(downsample))
        ax, ay, bx, by, frames = ax[sl], ay[sl], bx[sl], by[sl], frames[sl]
        if ax.size == 0:
            return None

    dx = ax - bx
    dy = ay - by
    dist = np.sqrt(dx*dx + dy*dy)

    # tốc độ từng chuột (dịch khung-kề-khung; prepend để giữ chiều)
    dax = np.diff(ax, prepend=ax[0])
    day = np.diff(ay, prepend=ay[0])
    dbx = np.diff(bx, prepend=bx[0])
    dby = np.diff(by, prepend=by[0])
    speed_a = np.sqrt(dax*dax + day*day)
    speed_b = np.sqrt(dbx*dbx + dby*dby)

    rel_speed = speed_a - speed_b
    ddist = np.diff(dist, prepend=dist[0])

    # --- trả về Polars DF cho _make_windows(...) hiện tại ---
    feat = pl.DataFrame(
        {
            "frame": frames,
            "dist": dist,
            "rel_speed": rel_speed,
            "ddist": ddist,
        }
    ).sort("frame")

    return feat


# Window creation from features

def _make_windows(feat: pl.DataFrame, min_len: int, q_dist: float = 0.40, q_rel: float = 0.60, q_ddist: float = 0.40) -> List[Tuple[int,int]]:
    """
    Local percentiles per pair for robust thresholds.
    Condition:
      (dist <= Pq_dist) or (rel_speed >= Pq_rel and ddist <= Pq_ddist)
    """
    if len(feat) == 0:
        return []
    # quantiles
    qd = float(feat["dist"].quantile(q_dist))
    qr = float(feat["rel_speed"].quantile(q_rel))
    qdd = float(feat["ddist"].quantile(q_ddist))  # typically negative
    # boolean mask
    cond = (pl.col("dist") <= qd) | ((pl.col("rel_speed") >= qr) & (pl.col("ddist") <= qdd))
    mask = feat.select(cond.alias("m")).to_series().to_list()
    frames = feat["frame"].to_list()

    windows: List[Tuple[int,int]] = []
    run: Optional[List[int]] = None
    for i, flag in enumerate(mask):
        if flag and run is None:
            run = [frames[i], frames[i]]
        elif flag and run is not None:
            run[1] = frames[i]
        elif (not flag) and run is not None:
            s,e = run[0], run[1]+1
            if e - s >= min_len:
                windows.append((s,e))
            run = None
    if run is not None:
        s,e = run[0], run[1]+1
        if e - s >= min_len:
            windows.append((s,e))
    return merge_intervals(windows)


# ========================
# Advanced Baseline Prediction
# ========================

def _order_actions_by_timing(actions: List[str], lab_id: str,
                             timing_lab: Dict[str, Dict[str, float]],
                             timing_global: Dict[str, float],
                             canonical: Dict[str,int]) -> List[str]:
    def score(a: str) -> float:
        if lab_id in timing_lab and a in timing_lab[lab_id]:
            return timing_lab[lab_id][a]
        return timing_global.get(a, 0.5)
    # earlier → smaller
    return sorted(actions, key=lambda a: (score(a), canonical.get(a, 99)))

def _clip_rare_actions(weights_map: Dict[str,float], actions: List[str], p_min: float, cap: float) -> Dict[str,float]:
    w = {a: max(0.0, float(weights_map.get(a, 0.0))) for a in actions}
    # if an action extremely rare, cap its share
    for a in actions:
        if w[a] < p_min:
            w[a] = min(w[a], cap)
    s = sum(w.values()) or 1.0
    return {a: w[a]/s for a in actions}


# Segment allocation and smoothing functions

def _allocate_segments_in_windows(windows: List[Tuple[int,int]],
                                  ordered_actions: List[str],
                                  weights: Dict[str,float],
                                  med_dur: Dict[str,int],
                                  total_frames: int) -> List[Tuple[str,int,int]]:
    """
    Allocate contiguous segments across union(windows) sequentially following ordered_actions.
    Length per action ~ max(weight*total, median_duration) but clipped by remaining frames.
    """
    # flatten windows into a sequence of positions
    win_idx = 0
    cur_s, cur_e = (windows[0] if windows else (0,0))
    remain = sum(e-s for s,e in windows)
    out: List[Tuple[str,int,int]] = []

    for a in ordered_actions:
        if remain <= 0: break
        want = int(weights.get(a, 0.0) * total_frames)
        want = max(want, int(med_dur.get(a, 0) or 0))
        want = min(want, remain)
        got = 0
        while got < want and win_idx < len(windows):
            s,e = cur_s, cur_e
            if s >= e:
                win_idx += 1
                if win_idx >= len(windows): break
                cur_s, cur_e = windows[win_idx]
                continue
            take = min(want - got, e - s)
            out.append((a, s, s+take))
            got += take
            remain -= take
            cur_s = s + take
            if cur_s >= e and win_idx < len(windows):
                win_idx += 1
                if win_idx < len(windows):
                    cur_s, cur_e = windows[win_idx]
    return out

def _smooth_segments(segments: List[Tuple[str,int,int]], min_len: int, gap_close: int) -> List[Tuple[str,int,int]]:
    if not segments: return []
    # sort
    segments = sorted(segments, key=lambda x: (x[1], x[2], x[0]))
    # remove too short
    segments = [seg for seg in segments if seg[2] - seg[1] >= min_len]
    if not segments: return []
    # merge same-action with small gap
    out = [segments[0]]
    for a,s,e in segments[1:]:
        pa,ps,pe = out[-1]
        if a == pa and s - pe <= gap_close:
            out[-1] = (a, ps, e)
        else:
            out.append((a,s,e))
    return out


# Main prediction function (Part 1)

def predict_without_ml(dataset: pl.DataFrame, data_split: str, cfg: Optional[Config] = None,
                       priors_per_lab: Optional[Dict[str, Dict[str, float]]] = None,
                       priors_global: Optional[Dict[str, float]] = None,
                       meddur_per_lab: Optional[Dict[str, Dict[str, int]]] = None,
                       meddur_global: Optional[Dict[str, int]] = None,
                       timing_lab: Optional[Dict[str, Dict[str, float]]] = None,
                       timing_global: Optional[Dict[str, float]] = None,
                       prior_scope: str = "mixed",
                       use_windows: bool = True,
                       min_len: int = 10,
                       gap_close: int = 5,
                       p_min: float = 0.03,
                       cap: float = 0.02) -> pl.DataFrame:
    """
    Advanced heuristic:
      - Optionally create proximity windows from tracking per (agent,target) using pairwise features.
      - Allocate within union(windows) following action order by timing prior; lengths by weight & median duration.
      - Smooth & gap-close small segments; rare-action clipping.
    """
    cfg = cfg or Config()
    track_dir = cfg.test_track_dir if data_split == "test" else cfg.train_track_dir
    records: List[Tuple[int, str, str, str, int, int]] = []
    canonical = {"approach": 0, "avoid": 1, "chase": 2, "chaseattack": 3, "attack": 4, "mount": 5, "submit": 6}
    # Main prediction function (Part 2 - Processing loop)

    for row in tqdm(dataset.to_dicts(), total=len(dataset), desc=f"Predicting ({data_split})"):
        lab_id: str = row["lab_id"]
        if lab_id.startswith("MABe22"): continue
        video_id: int = row["video_id"]
        path = track_dir / lab_id / f"{video_id}.parquet"
        if not path.exists():
            logger.warning("Tracking file not found: %s", path)
            continue

        try:
            trk = pl.read_parquet(path)
            start_frame = int(trk["video_frame"].min())
            stop_frame = int(trk["video_frame"].max()) + 1
            video_frames = stop_frame - start_frame
            if video_frames <= 0: continue

            # parse behaviors
            raw_list = safe_json_loads(row["behaviors_labeled"])
            triples: List[List[str]] = []
            for b in raw_list:
                parts = [p.strip() for p in str(b).replace("'", "").split(",")]
                if len(parts) == 3:
                    triples.append(parts)
            if not triples:  # no actions labeled for this video
                continue

            beh_df = pl.DataFrame(triples, schema=["agent","target","action"], orient="row").with_columns(
                [pl.col("agent").cast(pl.Utf8), pl.col("target").cast(pl.Utf8), pl.col("action").cast(pl.Utf8)]
            )
        # Main prediction function (Part 3 - Per agent-target processing)

            # per (agent,target)
            for (agent, target), group in beh_df.group_by(["agent","target"]):
                actions = sorted(list(set(group["action"].to_list())), key=lambda a: canonical.get(a, 99))
                if not actions: continue

                # choose priors
                if prior_scope == "lab" and priors_per_lab is not None:
                    w_map = priors_per_lab.get(str(lab_id), {})
                    md_map = meddur_per_lab.get(str(lab_id), {}) if meddur_per_lab else {}
                elif prior_scope == "global" and priors_global is not None:
                    w_map = priors_global
                    md_map = meddur_global or {}
                else:
                    w_map = (priors_per_lab or {}).get(str(lab_id), {}) or (priors_global or {})
                    md_map = (meddur_per_lab or {}).get(str(lab_id), {}) or (meddur_global or {})

                # rare-action clipping & renorm
                weights = _clip_rare_actions(w_map, actions, p_min=p_min, cap=cap)

                # order actions by timing prior
                ordered_actions = _order_actions_by_timing(
                    actions, str(lab_id), timing_lab or {}, timing_global or {}, canonical
                )

                # windows
                windows: List[Tuple[int,int]]
                if use_windows:
                    feat = _pair_features(trk, _norm_mouse_id(agent), _norm_mouse_id(target))
                    if feat is None:
                        windows = [(start_frame, stop_frame)]
                    else:
                        windows = _make_windows(feat, min_len=min_len)
                        if not windows:
                            windows = [(start_frame, stop_frame)]
                else:
                    windows = [(start_frame, stop_frame)]

                windows = merge_intervals(windows)
                allowed_total = sum(e - s for s,e in windows)
                if allowed_total <= 0:
                    continue

                # allocate segments along windows
                segs = _allocate_segments_in_windows(
                    windows=windows,
                    ordered_actions=ordered_actions,
                    weights=weights,
                    med_dur=md_map,
                    total_frames=allowed_total
                )

                segs = _smooth_segments(segs, min_len=min_len, gap_close=gap_close)

                # emit rows
                for a, s, e in segs:
                    if e > s:
                        records.append((
                            video_id,
                            _norm_mouse_id(agent), _norm_mouse_id(target),
                            a, int(s), int(e)
                        ))

        except Exception as e:
            logger.error("Error processing %s: %s", path, e)
            continue
    # Main prediction function (Part 4 - Return results)

    if not records:
        raise ValueError("No predictions generated.")

    df = pl.DataFrame(
        records,
        schema={
            "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
            "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
        },
        orient="row",
    )
    df = validate_schema(df, cfg.submission_schema, "Submission")
    validate_frame_ranges(df, "Submission")
    return df


# ========================
# Main Functions for Validation and Submission
# ========================

def run_validate(cfg: Config, beta: float, prior_scope: str, eps: float,
                 use_windows: bool, min_len: int, gap_close: int, p_min: float, cap: float) -> float:
    logger.info("Loading train data for validation: %s", cfg.train_csv)
    train = pl.read_csv(cfg.train_csv)
    train_subset = train.filter(~pl.col("lab_id").str.starts_with("MABe22"))

    logger.info("Building solution dataframe & spans...")
    solution = create_solution_df(train_subset, cfg)
    spans = build_video_spans(train_subset, "train", cfg)

    logger.info("Computing priors (eps=%.2f) & timing...", eps)
    per_lab, global_w, med_lab, med_glob = compute_action_priors(solution, eps=eps)
    timing_lab, timing_glob = compute_timing_priors(solution, spans)

    logger.info("Generating predictions (advanced)...")
    submission_train = predict_without_ml(
        train_subset, "train", cfg,
        priors_per_lab=per_lab, priors_global=global_w,
        meddur_per_lab=med_lab, meddur_global=med_glob,
        timing_lab=timing_lab, timing_global=timing_glob,
        prior_scope=prior_scope,
        use_windows=use_windows, min_len=min_len, gap_close=gap_close,
        p_min=p_min, cap=cap
    )

    logger.info("Scoring (beta=%.3f)...", beta)
    val_score = score(solution, submission_train, cfg.row_id_col, beta=beta, cfg=cfg)
    print(f"[RESULT] Validation F1: {val_score:.6f}")
    return val_score


# Submission generation function

def run_submit(cfg: Config, prior_scope: str, eps: float,
               use_windows: bool, min_len: int, gap_close: int, p_min: float, cap: float) -> None:
    logger.info("Loading train (for priors) and test data...")
    train = pl.read_csv(cfg.train_csv)
    train_subset = train.filter(~pl.col("lab_id").str.starts_with("MABe22"))
    solution = create_solution_df(train_subset, cfg)
    spans = build_video_spans(train_subset, "train", cfg)
    per_lab, global_w, med_lab, med_glob = compute_action_priors(solution, eps=eps)
    timing_lab, timing_glob = compute_timing_priors(solution, spans)

    test = pl.read_csv(cfg.test_csv)

    logger.info("Generating predictions (advanced, test)...")
    submission_test = predict_without_ml(
        test, "test", cfg,
        priors_per_lab=per_lab, priors_global=global_w,
        meddur_per_lab=med_lab, meddur_global=med_glob,
        timing_lab=timing_lab, timing_global=timing_glob,
        prior_scope=prior_scope,
        use_windows=use_windows, min_len=min_len, gap_close=gap_close,
        p_min=p_min, cap=cap
    )

    ordered = list(cfg.submission_schema.keys())
    submission_test = submission_test.select(ordered).with_row_index(cfg.row_id_col)

    logger.info("Saving submission to %s", cfg.submission_file)
    submission_test.write_csv(cfg.submission_file)

    try:
        with open(cfg.submission_file, "r") as f:
            for _ in range(10):
                line = f.readline()
                if not line: break
                logger.info("SUBMISSION PREVIEW | %s", line.strip())
    except Exception as e:
        logger.warning("Preview failed: %s", e)


# ========================
# Example Usage and Testing
# ========================

# Initialize configuration and setup logging
setup_logging(verbosity=2)
warnings.filterwarnings("ignore")

# Create configuration instance
cfg = Config()
print(f"Data root: {cfg.data_root}")
print(f"Train CSV: {cfg.train_csv}")
print(f"Test CSV: {cfg.test_csv}")

# Check if data files exist
print(f"Train CSV exists: {cfg.train_csv.exists()}")
print(f"Test CSV exists: {cfg.test_csv.exists()}")
print(f"Train annotation dir exists: {cfg.train_annot_dir.exists()}")
print(f"Train tracking dir exists: {cfg.train_track_dir.exists()}")


# Run validation with default parameters
# This will validate the model using the training data

print("Running validation...")
try:
    val_score = run_validate(
        cfg=cfg,
        beta=1.0,  # F1 score (beta=1)
        prior_scope="mixed",  # Use lab-specific priors with global fallback
        eps=12.0,  # Laplace smoothing parameter
        use_windows=True,  # Use proximity windows
        min_len=10,  # Minimum segment length
        gap_close=5,  # Gap closing threshold
        p_min=0.03,  # Rare action threshold
        cap=0.02  # Rare action cap
    )
    print(f"Validation completed successfully! F1 Score: {val_score:.6f}")
except Exception as e:
    print(f"Validation failed: {e}")
    import traceback
    traceback.print_exc()


# Generate submission for test data
# Uncomment and run this cell to generate a submission file

print("Generating submission...")
try:
    run_submit(
        cfg=cfg,
        prior_scope="mixed",
        eps=12.0,
        use_windows=True,
        min_len=10,
        gap_close=5,
        p_min=0.03,
        cap=0.02
    )
    print("Submission generated successfully!")
except Exception as e:
    print(f"Submission generation failed: {e}")
    import traceback
    traceback.print_exc()

