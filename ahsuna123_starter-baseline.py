from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import itertools

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ========================
# Config
# ========================

@dataclass(frozen=True)
class Config:
    data_root: Path = Path(os.getenv("MABE_DATA_ROOT", "/kaggle/input/MABe-mouse-behavior-detection"))
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

class HostVisibleError(Exception): pass

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(ascus)s | %(levelname)s | %(name)s | %(message)s", force=True)

# ========================
# Utils & Validators
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
    return range(start, stop)

def merge_intervals(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not intervals: return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s,e in intervals[1:]:
        ps,pe = merged[-1]
        if s <= pe: merged[-1] = (ps, max(pe, e))
        else: merged.append((s,e))
    return merged

# ========================
# Core Functions
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

def build_video_spans(dataset: pl.DataFrame, split: str, cfg: Optional[Config] = None) -> Dict[int, Tuple[int,int]]:
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

def compute_action_priors(solution: pl.DataFrame, eps: float = 1.0) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, int]], Dict[str, int]]:
    sol = solution.with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("dur"))
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

    med_by_lab = sol.group_by(["lab_id", "action"]).median().select(["lab_id","action","dur"])
    per_lab_med_dur: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in med_by_lab.to_dicts():
        per_lab_med_dur[str(r["lab_id"])][str(r["action"])] = int(r["dur"])
    med_global = sol.group_by(["action"]).median().select(["action","dur"])
    global_med_dur: Dict[str, int] = {r["action"]: int(r["dur"]) for r in med_global.to_dicts()}

    return per_lab_weight, global_weight, per_lab_med_dur, global_med_dur

def compute_timing_priors(solution: pl.DataFrame, video_spans: Dict[int, Tuple[int,int]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
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

def _strip_mouse_prefix(s: str | int) -> str:
    s = str(s)
    return s[5:] if s.startswith("mouse") else s

# ========================
# ENHANCED FEATURE EXTRACTION
# ========================

def _pair_features_enhanced(df: pl.DataFrame, agent_raw: str, target_raw: str, downsample: int = 1) -> Optional[pl.DataFrame]:
    """Enhanced feature extraction with behavioral indicators"""
    frame_candidates = ["video_frame","frame","frame_idx"]
    id_candidates = ["mouse_id","id","track_id","agent_id"]
    x_candidates = ["x","x_pos","x_position","x_mm","centroid_x","cx"]
    y_candidates = ["y","y_pos","y_position","y_mm","centroid_y","cy"]

    cols = set(df.columns)
    frame_col = next((c for c in frame_candidates if c in cols), None)
    id_col = next((c for c in id_candidates if c in cols), None)
    x_col = next((c for c in x_candidates if c in cols), None)
    y_col = next((c for c in y_candidates if c in cols), None)
    if not all([frame_col, id_col, x_col, y_col]):
        return None

    a_id = _strip_mouse_prefix(agent_raw)
    t_id = _strip_mouse_prefix(target_raw)

    pdf = df.select([frame_col, id_col, x_col, y_col]).to_pandas()
    pdf[frame_col] = pdf[frame_col].astype(np.int64, copy=False)
    pdf[id_col] = pdf[id_col].astype(str, copy=False)

    a = pdf[pdf[id_col] == a_id].copy()
    b = pdf[pdf[id_col] == t_id].copy()
    if a.empty or b.empty:
        return None

    a.drop_duplicates(subset=[frame_col], keep="first", inplace=True)
    b.drop_duplicates(subset=[frame_col], keep="first", inplace=True)

    merged = a.merge(b, on=frame_col, how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        return None
    merged.sort_values(frame_col, inplace=True)

    ax = merged[f"{x_col}_a"].to_numpy(dtype=np.float64, copy=False)
    ay = merged[f"{y_col}_a"].to_numpy(dtype=np.float64, copy=False)
    bx = merged[f"{x_col}_b"].to_numpy(dtype=np.float64, copy=False)
    by = merged[f"{y_col}_b"].to_numpy(dtype=np.float64, copy=False)
    frames = merged[frame_col].to_numpy(dtype=np.int64, copy=False)

    if downsample > 1:
        sl = slice(0, None, int(downsample))
        ax, ay, bx, by, frames = ax[sl], ay[sl], bx[sl], by[sl], frames[sl]
        if ax.size == 0:
            return None

    # Basic features
    dx = ax - bx
    dy = ay - by
    dist = np.sqrt(dx*dx + dy*dy)

    dax = np.diff(ax, prepend=ax[0])
    day = np.diff(ay, prepend=ay[0])
    dbx = np.diff(bx, prepend=bx[0])
    dby = np.diff(by, prepend=by[0])
    speed_a = np.sqrt(dax*dax + day*day)
    speed_b = np.sqrt(dbx*dbx + dby*dby)

    rel_speed = speed_a - speed_b
    ddist = np.diff(dist, prepend=dist[0])

    # Enhanced features
    angle_between = np.arctan2(dy, dx)
    angle_change = np.abs(np.diff(angle_between, prepend=angle_between[0]))
    
    # Acceleration
    acc_a = np.diff(speed_a, prepend=speed_a[0])
    acc_b = np.diff(speed_b, prepend=speed_b[0])
    rel_acc = acc_a - acc_b
    
    # Proximity indicators
    dist_pct = np.percentile(dist, [10, 25, 50, 75, 90])
    very_close = (dist <= dist_pct[0]).astype(float)
    close = (dist <= dist_pct[1]).astype(float)
    approaching = (ddist < 0).astype(float)
    
    # Temporal smoothing
    window_size = min(5, len(dist))
    if window_size > 1:
        from scipy.ndimage import uniform_filter1d
        dist_smooth = uniform_filter1d(dist, size=window_size)
        speed_trend_a = uniform_filter1d(speed_a, size=window_size)
        speed_trend_b = uniform_filter1d(speed_b, size=window_size)
    else:
        dist_smooth = dist
        speed_trend_a = speed_a
        speed_trend_b = speed_b
    
    # Interaction intensity
    interaction_score = (very_close * 2 + close) * (speed_a + speed_b) / 2
    
    feat = pl.DataFrame({
        "frame": frames,
        "dist": dist,
        "rel_speed": rel_speed,
        "ddist": ddist,
        "angle_between": angle_between,
        "angle_change": angle_change,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "rel_acc": rel_acc,
        "very_close": very_close,
        "close": close,
        "approaching": approaching,
        "dist_smooth": dist_smooth,
        "speed_trend_a": speed_trend_a,
        "speed_trend_b": speed_trend_b,
        "interaction_score": interaction_score,
        "speed_a": speed_a,
        "speed_b": speed_b
    }).sort("frame")

    return feat

# ========================
# BEHAVIOR-SPECIFIC DETECTION
# ========================

def _make_behavior_specific_windows(feat: pl.DataFrame, behavior: str, min_len: int) -> List[Tuple[int,int]]:
    """Create windows tailored to specific behaviors"""
    if len(feat) == 0:
        return []
    
    # Behavior-specific parameters
    behavior_configs = {
        'approach': {
            'dist_q': 0.70,
            'speed_q': 0.40,
            'use_angle': True,
            'use_approaching': True,
            'primary_cond': lambda: (pl.col("approaching") == 1) & (pl.col("dist") <= pl.col("dist").quantile(0.70))
        },
        'chase': {
            'dist_q': 0.60,
            'speed_q': 0.70,
            'use_acceleration': True,
            'use_rel_speed': True,
            'primary_cond': lambda: (pl.col("rel_speed") >= pl.col("rel_speed").quantile(0.70)) & (pl.col("acc_a") >= pl.col("acc_a").quantile(0.60))
        },
        'attack': {
            'dist_q': 0.15,
            'speed_q': 0.60,
            'use_proximity': True,
            'use_interaction': True,
            'primary_cond': lambda: (pl.col("very_close") == 1) & (pl.col("interaction_score") >= pl.col("interaction_score").quantile(0.80))
        },
        'mount': {
            'dist_q': 0.10,
            'speed_q': 0.30,
            'use_proximity': True,
            'use_low_speed': True,
            'primary_cond': lambda: (pl.col("very_close") == 1) & (pl.col("speed_a") <= pl.col("speed_a").quantile(0.40))
        },
        'avoid': {
            'dist_q': 0.80,
            'speed_q': 0.60,
            'inverse_approach': True,
            'primary_cond': lambda: (pl.col("approaching") == 0) & (pl.col("rel_speed") <= pl.col("rel_speed").quantile(0.30))
        },
        'chaseattack': {
            'dist_q': 0.30,
            'speed_q': 0.75,
            'use_acceleration': True,
            'use_proximity': True,
            'primary_cond': lambda: (pl.col("close") == 1) & (pl.col("rel_speed") >= pl.col("rel_speed").quantile(0.75)) & (pl.col("acc_a") >= pl.col("acc_a").quantile(0.70))
        },
        'submit': {
            'dist_q': 0.20,
            'speed_q': 0.20,
            'use_proximity': True,
            'use_low_speed': True,
            'primary_cond': lambda: (pl.col("close") == 1) & (pl.col("speed_a") <= pl.col("speed_a").quantile(0.30)) & (pl.col("speed_b") <= pl.col("speed_b").quantile(0.30))
        }
    }
    
    config = behavior_configs.get(behavior, {
        'dist_q': 0.50, 
        'speed_q': 0.60,
        'primary_cond': lambda: (pl.col("dist") <= pl.col("dist").quantile(0.50)) & (pl.col("rel_speed") >= pl.col("rel_speed").quantile(0.60))
    })
    
    # Apply behavior-specific condition
    try:
        cond = config['primary_cond']()
        mask = feat.select(cond.alias("m")).to_series().to_list()
    except Exception:
        # Fallback to simple condition
        qd = float(feat["dist"].quantile(config['dist_q']))
        qs = float(feat["rel_speed"].quantile(config['speed_q']))
        mask = feat.select(((pl.col("dist") <= qd) & (pl.col("rel_speed") >= qs)).alias("m")).to_series().to_list()
    
    frames = feat["frame"].to_list()
    
    # Convert mask to windows
    windows = []
    run = None
    for i, flag in enumerate(mask):
        if flag and run is None:
            run = [frames[i], frames[i]]
        elif flag and run is not None:
            run[1] = frames[i]
        elif (not flag) and run is not None:
            s, e = run[0], run[1] + 1
            if e - s >= min_len:
                windows.append((s, e))
            run = None
    
    if run is not None:
        s, e = run[0], run[1] + 1
        if e - s >= min_len:
            windows.append((s, e))
    
    return merge_intervals(windows)

# ========================
# SEQUENCE MODELING
# ========================

def _model_behavior_sequences(segments: List[Tuple[str,int,int]], video_frames: int) -> List[Tuple[str,int,int]]:
    """Model sequential dependencies between behaviors"""
    if not segments:
        return segments
    
    # Define realistic behavior transitions
    transitions = {
        'approach': ['chase', 'attack', 'mount', 'avoid'],
        'chase': ['attack', 'chaseattack', 'mount', 'approach'],
        'attack': ['mount', 'chase', 'chaseattack'],
        'chaseattack': ['attack', 'mount', 'chase'],
        'mount': ['submit', 'attack'],
        'avoid': ['approach'],  # Can lead to approach if situation changes
        'submit': []  # Usually terminal
    }
    
    # Sort segments by start time
    segments = sorted(segments, key=lambda x: x[1])
    improved_segments = []
    
    for i, (action, start, end) in enumerate(segments):
        should_include = True
        
        if i > 0 and improved_segments:
            prev_action, prev_start, prev_end = improved_segments[-1]
            gap = start - prev_end
            
            # Check for realistic sequences
            if 0 <= gap <= 45:  # Within 45 frames (~1.5 seconds)
                valid_transitions = transitions.get(prev_action, [])
                
                # Handle illogical sequences
                if action == prev_action and gap <= 10:
                    # Merge same behaviors that are very close
                    improved_segments[-1] = (action, prev_start, max(prev_end, end))
                    continue
                elif valid_transitions and action not in valid_transitions:
                    # Check for impossible transitions
                    impossible_pairs = [
                        ('attack', 'approach'), ('mount', 'approach'), 
                        ('submit', 'chase'), ('submit', 'attack')
                    ]
                    if (prev_action, action) in impossible_pairs:
                        # Skip this segment as it's unlikely
                        continue
        
        if should_include:
            improved_segments.append((action, start, end))
    
    return improved_segments

# ========================
# ENSEMBLE METHODS
# ========================

def merge_ensemble_predictions(predictions_list: List[pl.DataFrame], consensus_threshold: float = 0.6) -> pl.DataFrame:
    """Merge multiple prediction sets using consensus voting"""
    if not predictions_list:
        return pl.DataFrame(schema={
            "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
            "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
        })
    
    if len(predictions_list) == 1:
        return predictions_list[0]
    
    # Collect all predictions
    all_records = []
    for pred_df in predictions_list:
        for record in pred_df.to_dicts():
            all_records.append(record)
    
    # Group by key and find consensus
    grouped = defaultdict(list)
    for record in all_records:
        key = (record['video_id'], record['agent_id'], record['target_id'], record['action'])
        grouped[key].append((record['start_frame'], record['stop_frame']))
    
    # Apply consensus threshold
    min_votes = max(1, int(len(predictions_list) * consensus_threshold))
    final_records = []
    
    for (vid, agent, target, action), intervals in grouped.items():
        if len(intervals) >= min_votes:
            # Merge overlapping intervals
            merged = merge_intervals(intervals)
            for start, end in merged:
                final_records.append((vid, agent, target, action, start, end))
    
    if not final_records:
        # Fallback to best single prediction
        return predictions_list[0]
    
    return pl.DataFrame(
        final_records,
        schema={
            "video_id": pl.Int64, "agent_id": pl.Utf8, "target_id": pl.Utf8,
            "action": pl.Utf8, "start_frame": pl.Int64, "stop_frame": pl.Int64,
        },
        orient="row"
    )

# ========================
# PARAMETER OPTIMIZATION
# ========================

def optimize_parameters(solution_df: pl.DataFrame, train_sample: pl.DataFrame, cfg: Config) -> Dict:
    """Simple parameter optimization based on data characteristics"""
    
    # Analyze data characteristics
    solution_df = solution_df.with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("duration"))
    
    behavior_stats = {}
    for action in solution_df['action'].unique():
        action_data = solution_df.filter(pl.col('action') == action)
        behavior_stats[action] = {
            'count': len(action_data),
            'mean_duration': action_data['duration'].mean(),
            'median_duration': action_data['duration'].median(),
            'std_duration': action_data['duration'].std() or 1.0
        }
    
    # Adaptive parameter selection based on data
    avg_duration = solution_df['duration'].mean()
    std_duration = solution_df['duration'].std() or 1.0
    
    if avg_duration < 20:  # Short behaviors
        return {
            'min_len': max(5, int(avg_duration * 0.3)),
            'gap_close': 2,
            'p_min': 0.03,
            'cap': 0.04
        }
    elif avg_duration > 50:  # Long behaviors
        return {
            'min_len': max(15, int(avg_duration * 0.2)),
            'gap_close': 8,
            'p_min': 0.07,
            'cap': 0.08
        }
    else:  # Medium behaviors
        return {
            'min_len': max(10, int(avg_duration * 0.25)),
            'gap_close': 4,
            'p_min': 0.05,
            'cap': 0.06
        }

# ========================
# MAIN PREDICTION FUNCTIONS
# ========================

def _order_actions_by_timing(actions: List[str], lab_id: str,
                             timing_lab: Dict[str, Dict[str, float]],
                             timing_global: Dict[str, float],
                             canonical: Dict[str,int]) -> List[str]:
    def score(a: str) -> float:
        if lab_id in timing_lab and a in timing_lab[lab_id]:
            return timing_lab[lab_id][a]
        return timing_global.get(a, 0.5)
    return sorted(actions, key=lambda a: (score(a), canonical.get(a, 99)))

def _clip_rare_actions(weights_map: Dict[str,float], actions: List[str], p_min: float, cap: float) -> Dict[str,float]:
    w = {a: max(0.0, float(weights_map.get(a, 0.0))) for a in actions}
    for a in actions:
        if w[a] < p_min:
            w[a] = min(w[a], cap)
    s = sum(w.values()) or 1.0
    return {a: w[a]/s for a in actions}

def _allocate_segments_in_windows(windows: List[Tuple[int,int]],
                                  ordered_actions: List[str],
                                  weights: Dict[str,float],
                                  med_dur: Dict[str,int],
                                  total_frames: int) -> List[Tuple[str,int,int]]:
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
    segments = sorted(segments, key=lambda x: (x[1], x[2], x[0]))
    segments = [seg for seg in segments if seg[2] - seg[1] >= min_len]
    if not segments: return []
    out = [segments[0]]
    for a,s,e in segments[1:]:
        pa,ps,pe = out[-1]
        if a == pa and s - pe <= gap_close:
            out[-1] = (a, ps, e)
        else:
            out.append((a,s,e))
    return out

def predict_without_ml_improved(dataset: pl.DataFrame, data_split: str, cfg: Optional[Config] = None,
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
                               cap: float = 0.02,
                               use_enhanced_features: bool = True,
                               use_behavior_specific: bool = True,
                               use_sequences: bool = True) -> pl.DataFrame:
    """Improved prediction with all enhancements"""
    
    cfg = cfg or Config()
    track_dir = cfg.test_track_dir if data_split == "test" else cfg.train_track_dir
    records: List[Tuple[int, str, str, str, int, int]] = []
    canonical = {"approach": 0, "avoid": 1, "chase": 2, "chaseattack": 3, "attack": 4, "mount": 5, "submit": 6}

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

            raw_list = safe_json_loads(row["behaviors_labeled"])
            triples: List[List[str]] = []
            for b in raw_list:
                parts = [p.strip() for p in str(b).replace("'", "").split(",")]
                if len(parts) == 3:
                    triples.append(parts)
            if not triples:
                continue

            beh_df = pl.DataFrame(triples, schema=["agent","target","action"], orient="row").with_columns(
                [pl.col("agent").cast(pl.Utf8), pl.col("target").cast(pl.Utf8), pl.col("action").cast(pl.Utf8)]
            )

            for (agent, target), group in beh_df.group_by(["agent","target"]):
                actions = sorted(list(set(group["action"].to_list())), key=lambda a: canonical.get(a, 99))
                if not actions: continue

                # Get priors
                if prior_scope == "lab" and priors_per_lab is not None:
                    w_map = priors_per_lab.get(str(lab_id), {})
                    md_map = meddur_per_lab.get(str(lab_id), {}) if meddur_per_lab else {}
                elif prior_scope == "global" and priors_global is not None:
                    w_map = priors_global
                    md_map = meddur_global or {}
                else:
                    w_map = (priors_per_lab or {}).get(str(lab_id), {}) or (priors_global or {})
                    md_map = (meddur_per_lab or {}).get(str(lab_id), {}) or (meddur_global or {})

                weights = _clip_rare_actions(w_map, actions, p_min=p_min, cap=cap)
                ordered_actions = _order_actions_by_timing(
                    actions, str(lab_id), timing_lab or {}, timing_global or {}, canonical
                )

                # Generate windows for each behavior
                all_windows = []
                if use_windows:
                    if use_enhanced_features:
                        feat = _pair_features_enhanced(trk, _norm_mouse_id(agent), _norm_mouse_id(target))
                    else:
                        feat = _pair_features_basic(trk, _norm_mouse_id(agent), _norm_mouse_id(target))
                    
                    if feat is None:
                        all_windows = [(start_frame, stop_frame)]
                    else:
                        if use_behavior_specific:
                            # Create behavior-specific windows
                            for action in actions:
                                behavior_windows = _make_behavior_specific_windows(feat, action, min_len)
                                all_windows.extend(behavior_windows)
                        else:
                            all_windows = _make_windows_basic(feat, min_len)
                        
                        if not all_windows:
                            all_windows = [(start_frame, stop_frame)]
                else:
                    all_windows = [(start_frame, stop_frame)]

                all_windows = merge_intervals(all_windows)
                allowed_total = sum(e - s for s,e in all_windows)
                if allowed_total <= 0:
                    continue

                # Allocate segments
                segs = _allocate_segments_in_windows(
                    windows=all_windows,
                    ordered_actions=ordered_actions,
                    weights=weights,
                    med_dur=md_map,
                    total_frames=allowed_total
                )

                # Smooth segments
                segs = _smooth_segments(segs, min_len=min_len, gap_close=gap_close)
                
                # Apply sequence modeling
                if use_sequences:
                    segs = _model_behavior_sequences(segs, video_frames)

                # Add to records
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

    if not records:
        return pl.DataFrame(schema=cfg.submission_schema)

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

def _pair_features_basic(df: pl.DataFrame, agent_raw: str, target_raw: str, downsample: int = 1) -> Optional[pl.DataFrame]:
    """Basic feature extraction (fallback)"""
    frame_candidates = ["video_frame","frame","frame_idx"]
    id_candidates = ["mouse_id","id","track_id","agent_id"]
    x_candidates = ["x","x_pos","x_position","x_mm","centroid_x","cx"]
    y_candidates = ["y","y_pos","y_position","y_mm","centroid_y","cy"]

    cols = set(df.columns)
    frame_col = next((c for c in frame_candidates if c in cols), None)
    id_col = next((c for c in id_candidates if c in cols), None)
    x_col = next((c for c in x_candidates if c in cols), None)
    y_col = next((c for c in y_candidates if c in cols), None)
    if not all([frame_col, id_col, x_col, y_col]):
        return None

    a_id = _strip_mouse_prefix(agent_raw)
    t_id = _strip_mouse_prefix(target_raw)

    pdf = df.select([frame_col, id_col, x_col, y_col]).to_pandas()
    pdf[frame_col] = pdf[frame_col].astype(np.int64, copy=False)
    pdf[id_col] = pdf[id_col].astype(str, copy=False)

    a = pdf[pdf[id_col] == a_id].copy()
    b = pdf[pdf[id_col] == t_id].copy()
    if a.empty or b.empty:
        return None

    a.drop_duplicates(subset=[frame_col], keep="first", inplace=True)
    b.drop_duplicates(subset=[frame_col], keep="first", inplace=True)

    merged = a.merge(b, on=frame_col, how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        return None
    merged.sort_values(frame_col, inplace=True)

    ax = merged[f"{x_col}_a"].to_numpy(dtype=np.float64, copy=False)
    ay = merged[f"{y_col}_a"].to_numpy(dtype=np.float64, copy=False)
    bx = merged[f"{x_col}_b"].to_numpy(dtype=np.float64, copy=False)
    by = merged[f"{y_col}_b"].to_numpy(dtype=np.float64, copy=False)
    frames = merged[frame_col].to_numpy(dtype=np.int64, copy=False)

    if downsample > 1:
        sl = slice(0, None, int(downsample))
        ax, ay, bx, by, frames = ax[sl], ay[sl], bx[sl], by[sl], frames[sl]
        if ax.size == 0:
            return None

    dx = ax - bx
    dy = ay - by
    dist = np.sqrt(dx*dx + dy*dy)

    dax = np.diff(ax, prepend=ax[0])
    day = np.diff(ay, prepend=ay[0])
    dbx = np.diff(bx, prepend=bx[0])
    dby = np.diff(by, prepend=by[0])
    speed_a = np.sqrt(dax*dax + day*day)
    speed_b = np.sqrt(dbx*dbx + dby*dby)

    rel_speed = speed_a - speed_b
    ddist = np.diff(dist, prepend=dist[0])

    feat = pl.DataFrame(
        {
            "frame": frames,
            "dist": dist,
            "rel_speed": rel_speed,
            "ddist": ddist,
        }
    ).sort("frame")

    return feat

def _make_windows_basic(feat: pl.DataFrame, min_len: int, q_dist: float = 0.40, q_rel: float = 0.60, q_ddist: float = 0.40) -> List[Tuple[int,int]]:
    """Basic window detection"""
    if len(feat) == 0:
        return []
    qd = float(feat["dist"].quantile(q_dist))
    qr = float(feat["rel_speed"].quantile(q_rel))
    qdd = float(feat["ddist"].quantile(q_ddist))
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

def predict_with_ensemble(dataset: pl.DataFrame, data_split: str, cfg: Config, **kwargs) -> pl.DataFrame:
    """Ensemble prediction with multiple configurations"""
    
    # Multiple configurations optimized for different scenarios
    configs = [
        # Sensitive detection
        {
            'min_len': 8, 'gap_close': 2, 'p_min': 0.02, 'cap': 0.03, 
            'use_enhanced_features': True, 'use_behavior_specific': True, 'use_sequences': True
        },
        # Balanced detection  
        {
            'min_len': 12, 'gap_close': 3, 'p_min': 0.04, 'cap': 0.05,
            'use_enhanced_features': True, 'use_behavior_specific': True, 'use_sequences': True
        },
        # Conservative detection
        {
            'min_len': 18, 'gap_close': 5, 'p_min': 0.06, 'cap': 0.07,
            'use_enhanced_features': True, 'use_behavior_specific': False, 'use_sequences': True
        },
        # Fallback configuration
        {
            'min_len': 15, 'gap_close': 4, 'p_min': 0.05, 'cap': 0.06,
            'use_enhanced_features': False, 'use_behavior_specific': False, 'use_sequences': False
        }
    ]
    
    predictions = []
    
    for i, config in enumerate(configs):
        print(f"Running ensemble model {i+1}/{len(configs)}...")
        try:
            # Merge config with existing kwargs
            merged_kwargs = {**kwargs}
            merged_kwargs.update(config)
            
            pred = predict_without_ml_improved(dataset, data_split, cfg, **merged_kwargs)
            if len(pred) > 0:
                predictions.append(pred)
        except Exception as e:
            logger.warning(f"Ensemble model {i+1} failed: {e}")
            continue
    
    if not predictions:
        # Fallback to basic prediction
        return predict_without_ml_improved(dataset, data_split, cfg, **kwargs)
    
    # Merge using consensus
    return merge_ensemble_predictions(predictions, consensus_threshold=0.5)

# ========================
# MAIN EXECUTION
# ========================

def main():
    print("MABe Mouse Behavior Detection - Enhanced Model")
    print("=" * 60)
    
    # Setup
    setup_logging(verbosity=1)
    warnings.filterwarnings("ignore")
    
    cfg = Config()
    
    # Load data
    print("\nLoading datasets...")
    train = pl.read_csv(cfg.train_csv)
    train_subset = train.filter(~pl.col("lab_id").str.starts_with("MABe22"))
    print(f"Loaded {len(train)} training samples ({len(train_subset)} after filtering)")
    
    print("\nBuilding solution dataframe...")
    solution = create_solution_df(train_subset, cfg)
    print(f"Processed {len(solution):,} annotations")
    
    print("\nComputing video spans...")
    spans = build_video_spans(train_subset, "train", cfg)
    print(f"Computed spans for {len(spans)} videos")
    
    print("\nComputing priors...")
    per_lab, global_w, med_lab, med_glob = compute_action_priors(solution, eps=0.3)
    timing_lab, timing_glob = compute_timing_priors(solution, spans)
    print(f"Computed priors for {len(global_w)} actions across {len(per_lab)} labs")
    
    print("\nOptimizing parameters...")
    optimal_params = optimize_parameters(solution, train_subset.sample(min(500, len(train_subset))), cfg)
    print(f"Optimal parameters: {optimal_params}")
    
    print("\nLoading test data...")
    test = pl.read_csv(cfg.test_csv)
    print(f"Loaded {len(test)} test samples")
    
    print("\nGenerating predictions with ensemble...")
    submission_test = predict_with_ensemble(
        test, "test", cfg,
        priors_per_lab=per_lab, 
        priors_global=global_w,
        meddur_per_lab=med_lab, 
        meddur_global=med_glob,
        timing_lab=timing_lab, 
        timing_global=timing_glob,
        prior_scope="mixed",
        use_windows=True,
        **optimal_params
    )
    print(f"Generated {len(submission_test):,} predictions")
    
    # Add row_id and save
    ordered = list(cfg.submission_schema.keys())
    submission_test = submission_test.select(ordered).with_row_index(cfg.row_id_col)
    
    print(f"\nSaving submission to {cfg.submission_file}...")
    submission_test.write_csv(cfg.submission_file)
    print("Submission saved successfully!")
    
    # Print summary
    if len(submission_test) > 0:
        actions = submission_test.group_by("action").count().sort("count", descending=True)
        print(f"\nPrediction Summary:")
        for row in actions.head(5).to_dicts():
            print(f"  {row['action']}: {row['count']} predictions")
        
        durations = (submission_test["stop_frame"] - submission_test["start_frame"]).to_list()
        print(f"\nDuration Statistics:")
        print(f"  Mean: {np.mean(durations):.1f} frames")
        print(f"  Median: {np.median(durations):.1f} frames")
        print(f"  Videos covered: {submission_test['video_id'].n_unique()}")

if __name__ == "__main__":
    main()

