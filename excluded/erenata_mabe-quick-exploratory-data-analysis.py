from __future__ import annotations

import os
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Style and warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 120

# Logging
logger = logging.getLogger(__name__)
def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)

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

class HostVisibleError(Exception): pass

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

def merge_intervals(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not intervals: return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s,e in intervals[1:]:
        ps,pe = merged[-1]
        if s <= pe: merged[-1] = (ps, max(pe, e))
        else: merged.append((s,e))
    return merged


setup_logging(verbosity=1)
cfg = Config()

print("Loading train/test metadata...")
train = pl.read_csv(cfg.train_csv)
test = pl.read_csv(cfg.test_csv)
train_subset = train.filter(~pl.col("lab_id").str.starts_with("MABe22"))
print(f"Train rows: {len(train)} (after filter: {len(train_subset)}) | Test rows: {len(test)}")

def create_solution_df(dataset: pl.DataFrame, cfg: Optional[Config] = None) -> pl.DataFrame:
    cfg = cfg or Config()
    records: List[pl.DataFrame] = []
    for row in tqdm(dataset.to_dicts(), total=len(dataset), desc="Building solution"):
        lab_id: str = row["lab_id"]
        video_id: int = row["video_id"]
        annot_path = cfg.train_annot_dir / lab_id / f"{video_id}.parquet"
        if not annot_path.exists():
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
            # keep only required columns, cast if needed
            keep_cols = list(Config().solution_schema.keys())
            annot = annot.select([c for c in keep_cols if c in annot.columns])
            annot = validate_schema(annot, Config().solution_schema, "Solution")
            records.append(annot)
        except Exception as e:
            logger.warning("Failed to load %s: %s", annot_path, e)
            continue
    if not records: raise ValueError("No annotation files loaded.")
    solution = pl.concat(records, how="vertical")
    return solution

def build_video_spans(dataset: pl.DataFrame, split: str, cfg: Optional[Config] = None) -> Dict[int, Tuple[int,int]]:
    cfg = cfg or Config()
    track_dir = cfg.train_track_dir if split == "train" else cfg.test_track_dir
    spans: Dict[int, Tuple[int,int]] = {}
    for row in tqdm(dataset.to_dicts(), total=len(dataset), desc="Scanning spans"):
        lab_id = row["lab_id"]
        if str(lab_id).startswith("MABe22"): continue
        vid = row["video_id"]
        path = track_dir / lab_id / f"{vid}.parquet"
        if not path.exists(): continue
        try:
            df = pl.read_parquet(path).select(["video_frame"])
            s = int(df["video_frame"].min())
            e = int(df["video_frame"].max()) + 1
            spans[int(vid)] = (s,e)
        except Exception as e:
            continue
    return spans

def compute_action_priors(solution: pl.DataFrame, eps: float = 1.0):
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

def compute_timing_priors(solution: pl.DataFrame, video_spans: Dict[int, Tuple[int,int]]):
    def start_pct_func(row) -> float:
        vid = int(row["video_id"])
        if vid not in video_spans:
            return 0.5
        s, e = video_spans[vid]
        denom = max(1, e - s)
        return float(max(0, min(1, (int(row["start_frame"]) - s) / denom)))
    rows = [
        {"lab_id": r["lab_id"], "action": r["action"], "start_pct": start_pct_func(r)}
        for r in solution.select(["lab_id", "action", "video_id", "start_frame"]).to_dicts()
    ]
    df = pl.DataFrame(rows)
    by_lab = df.group_by(["lab_id", "action"]).median().select(["lab_id", "action", "start_pct"])
    per_lab = defaultdict(dict)
    for r in by_lab.to_dicts():
        per_lab[str(r["lab_id"])][str(r["action"])] = float(r["start_pct"])
    g = df.group_by(["action"]).median().select(["action", "start_pct"])
    global_ = {r["action"]: float(r["start_pct"]) for r in g.to_dicts()}
    return per_lab, global_

print("Building solution from annotations...")
solution = create_solution_df(train_subset, cfg)
print(f"Annotations loaded: {len(solution):,}")

print("Computing frame spans...")
spans = build_video_spans(train_subset, "train", cfg)
print(f"Video spans computed: {len(spans)}")

print("Computing priors...")
per_lab, global_w, med_lab, med_glob = compute_action_priors(solution, eps=0.5)
timing_lab, timing_glob = compute_timing_priors(solution, spans)
print(f"Actions in priors: {len(global_w)} | Labs with priors: {len(per_lab)}")


def visualize_data_overview(train_df: pl.DataFrame, solution_df: pl.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1) Top Labs
    ax = axes[0,0]
    lab_counts = train_df.group_by("lab_id").count().sort("count", descending=True)
    top = lab_counts.head(12)
    ax.barh(top["lab_id"].to_list()[::-1], top["count"].to_list()[::-1], color="#4C72B0")
    ax.set_title("Videos per Lab (Top 12)"); ax.set_xlabel("Count"); ax.grid(axis="x", alpha=0.3)

    # 2) Action counts (solution)
    ax = axes[0,1]
    act_counts = solution_df.group_by("action").count().sort("count", descending=True)
    acts = act_counts["action"].to_list()[:20][::-1]
    vals = act_counts["count"].to_list()[:20][::-1]
    ax.barh(acts, vals, color="#64B5CD")
    ax.set_title("Top Behaviors (Train)"); ax.set_xlabel("Count"); ax.grid(axis="x", alpha=0.3)

    # 3) Duration histogram
    ax = axes[0,2]
    sd = solution_df.with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("duration"))["duration"].to_list()
    if sd:
        ax.hist(sd, bins=40, color="#C44E52", edgecolor="black", linewidth=0.3)
        ax.axvline(np.median(sd), color="black", linestyle="--", alpha=0.7, label=f"Median={np.median(sd):.1f}")
        ax.legend()
    ax.set_title("Behavior Duration (frames)"); ax.set_xlabel("Frames"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)

    # 4) Pairs
    ax = axes[1,0]
    pair_counts = solution_df.group_by(["agent_id", "target_id"]).count().sort("count", descending=True).head(10)
    labels = [f"{r['agent_id']}->{r['target_id']}" for r in pair_counts.to_dicts()][::-1]
    values = pair_counts["count"].to_list()[::-1]
    ax.barh(labels, values, color="#E17C05")
    ax.set_title("Top Pairs (Train)"); ax.set_xlabel("Count"); ax.grid(axis="x", alpha=0.3)

    # 5) Behaviors per video
    ax = axes[1,1]
    per_video = solution_df.group_by("video_id").count()["count"].to_list()
    if per_video:
        ax.hist(per_video, bins=30, color="#8172B2", edgecolor="black", linewidth=0.3)
        ax.axvline(np.mean(per_video), color="black", linestyle="--", alpha=0.7, label=f"Mean={np.mean(per_video):.1f}")
        ax.legend()
    ax.set_title("Behaviors per Video"); ax.set_xlabel("Count"); ax.set_ylabel("Videos"); ax.grid(axis="y", alpha=0.3)

    # 6) Summary
    ax = axes[1,2]; ax.axis("off")
    n_vid = int(train_df["video_id"].n_unique())
    n_lab = int(train_df["lab_id"].n_unique())
    n_anns = len(solution_df)
    n_act = int(solution_df["action"].n_unique())
    dur = solution_df.with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("duration"))["duration"]
    text = f"""DATASET SUMMARY
Videos: {n_vid}
Labs: {n_lab}
Annotations: {n_anns:,}
Unique behaviors: {n_act}
Mean duration: {float(dur.mean() if len(dur)>0 else 0):.1f} frames"""
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3))

    plt.tight_layout(); plt.show()

print("Overview visuals...")
visualize_data_overview(train_subset, solution)


def _strip_mouse_prefix(s: str | int) -> str:
    s = str(s)
    return s[5:] if s.startswith("mouse") else s

def _pair_features(df: pl.DataFrame, agent_raw: str, target_raw: str, downsample: int = 1) -> Optional[pl.DataFrame]:
    frame_candidates = ["video_frame","frame","frame_idx"]
    id_candidates = ["mouse_id","id","track_id","agent_id"]
    x_candidates = ["x","x_pos","x_position","x_mm","centroid_x","cx"]
    y_candidates = ["y","y_pos","y_position","y_mm","centroid_y","cy"]

    cols = set(df.columns)
    frame_col = next((c for c in frame_candidates if c in cols), None)
    id_col = next((c for c in id_candidates if c in cols), None)
    x_col = next((c for c in x_candidates if c in cols), None)
    y_col = next((c for c in y_candidates if c in cols), None)
    if not all([frame_col, id_col, x_col, y_col]): return None

    a_id = _strip_mouse_prefix(agent_raw)
    t_id = _strip_mouse_prefix(target_raw)

    pdf = df.select([frame_col, id_col, x_col, y_col]).to_pandas()
    pdf[frame_col] = pdf[frame_col].astype(np.int64, copy=False)
    pdf[id_col] = pdf[id_col].astype(str, copy=False)

    a = pdf[pdf[id_col] == a_id].copy()
    b = pdf[pdf[id_col] == t_id].copy()
    if a.empty or b.empty: return None

    a.drop_duplicates(subset=[frame_col], inplace=True)
    b.drop_duplicates(subset=[frame_col], inplace=True)

    merged = a.merge(b, on=frame_col, how="inner", suffixes=("_a", "_b"))
    if merged.empty: return None
    merged.sort_values(frame_col, inplace=True)

    ax = merged[f"{x_col}_a"].to_numpy(dtype=np.float64)
    ay = merged[f"{y_col}_a"].to_numpy(dtype=np.float64)
    bx = merged[f"{x_col}_b"].to_numpy(dtype=np.float64)
    by = merged[f"{y_col}_b"].to_numpy(dtype=np.float64)
    frames = merged[frame_col].to_numpy(dtype=np.int64)

    if downsample and downsample > 1:
        sl = slice(0, None, int(downsample))
        ax, ay, bx, by, frames = ax[sl], ay[sl], bx[sl], by[sl], frames[sl]
        if ax.size == 0: return None

    dx, dy = ax - bx, ay - by
    dist = np.sqrt(dx*dx + dy*dy)
    dax, day = np.diff(ax, prepend=ax[0]), np.diff(ay, prepend=ay[0])
    dbx, dby = np.diff(bx, prepend=bx[0]), np.diff(by, prepend=by[0])
    rel_speed = np.sqrt(dax*dax + day*day) - np.sqrt(dbx*dbx + dby*dby)
    ddist = np.diff(dist, prepend=dist[0])

    return pl.DataFrame({"frame": frames, "dist": dist, "rel_speed": rel_speed, "ddist": ddist}).sort("frame")

def _make_windows(feat: pl.DataFrame, min_len: int, q_dist: float = 0.40, q_rel: float = 0.60, q_ddist: float = 0.40) -> List[Tuple[int,int]]:
    if len(feat) == 0: return []
    qd = float(feat["dist"].quantile(q_dist))
    qr = float(feat["rel_speed"].quantile(q_rel))
    qdd = float(feat["ddist"].quantile(q_ddist))
    cond = (pl.col("dist") <= qd) | ((pl.col("rel_speed") >= qr) & (pl.col("ddist") <= qdd))
    mask = feat.select(cond.alias("m")).to_series().to_list()
    frames = feat["frame"].to_list()

    windows: List[Tuple[int,int]] = []
    run = None
    for i, flag in enumerate(mask):
        if flag and run is None: run = [frames[i], frames[i]]
        elif flag and run is not None: run[1] = frames[i]
        elif (not flag) and run is not None:
            s,e = run[0], run[1]+1
            if e - s >= min_len: windows.append((s,e))
            run = None
    if run is not None:
        s,e = run[0], run[1]+1
        if e - s >= min_len: windows.append((s,e))
    return merge_intervals(windows)

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

def predict_without_ml(dataset: pl.DataFrame, data_split: str, cfg: Optional[Config] = None,
                       priors_per_lab: Optional[Dict[str, Dict[str, float]]] = None,
                       priors_global: Optional[Dict[str, float]] = None,
                       meddur_per_lab: Optional[Dict[str, Dict[str, int]]] = None,
                       meddur_global: Optional[Dict[str, int]] = None,
                       timing_lab: Optional[Dict[str, Dict[str, float]]] = None,
                       timing_global: Optional[Dict[str, float]] = None,
                       prior_scope: str = "mixed",
                       use_windows: bool = True,
                       min_len: int = 15,
                       gap_close: int = 3,
                       p_min: float = 0.05,
                       cap: float = 0.05) -> pl.DataFrame:
    cfg = cfg or Config()
    track_dir = cfg.test_track_dir if data_split == "test" else cfg.train_track_dir
    records: List[Tuple[int, str, str, str, int, int]] = []
    canonical = {"approach": 0, "avoid": 1, "chase": 2, "chaseattack": 3, "attack": 4, "mount": 5, "submit": 6}

    for row in tqdm(dataset.to_dicts(), total=len(dataset), desc=f"Predicting ({data_split})"):
        lab_id: str = row["lab_id"]
        video_id: int = row["video_id"]
        path = track_dir / lab_id / f"{video_id}.parquet"
        if not path.exists():
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
                if allowed_total <= 0: continue

                segs = _allocate_segments_in_windows(
                    windows=windows,
                    ordered_actions=ordered_actions,
                    weights=weights,
                    med_dur=md_map,
                    total_frames=allowed_total
                )
                segs = _smooth_segments(segs, min_len=min_len, gap_close=gap_close)

                for a, s, e in segs:
                    if e > s:
                        records.append((video_id, _norm_mouse_id(agent), _norm_mouse_id(target), a, int(s), int(e)))

        except Exception as e:
            logger.warning("Error processing %s: %s", path, e)
            continue

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


print("Computing overview and priors completed above.")
print("Loading test and generating predictions...")

prediction_params = dict(
    prior_scope="mixed",
    use_windows=True,
    min_len=15,
    gap_close=3,
    p_min=0.05,
    cap=0.05
)

submission_test = predict_without_ml(
    test, "test", cfg,
    priors_per_lab=per_lab, priors_global=global_w,
    meddur_per_lab=med_lab, meddur_global=med_glob,
    timing_lab=timing_lab, timing_global=timing_glob,
    **prediction_params
)
print(f"Predictions generated: {len(submission_test):,}")

# Quick prediction QA visuals (safe for small test)
def visualize_predictions(pred_df: pl.DataFrame):
    if len(pred_df) == 0:
        print("No predictions to visualize."); return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    # Action histogram
    ax = axes[0]
    pc = pred_df.group_by("action").count().sort("count", descending=True)
    a = pc["action"].to_list()[:20][::-1]; c = pc["count"].to_list()[:20][::-1]
    ax.barh(a, c, color="#64B5CD"); ax.set_title("Predicted Actions"); ax.set_xlabel("Count"); ax.grid(axis="x", alpha=0.3)
    # Duration
    ax = axes[1]
    pdur = pred_df.with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("duration"))["duration"].to_list()
    ax.hist(pdur, bins=30, color="#C44E52", edgecolor="black", linewidth=0.3)
    if pdur:
        ax.axvline(np.median(pdur), color="black", linestyle="--", alpha=0.7, label=f"Median={np.median(pdur):.1f}")
        ax.legend()
    ax.set_title("Predicted Duration (frames)"); ax.set_xlabel("Frames"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()

visualize_predictions(submission_test)

# Save submission
ordered = list(cfg.submission_schema.keys())
submission_out = submission_test.select(ordered).with_row_index(cfg.row_id_col)
submission_out.write_csv(cfg.submission_file)
print(f"Saved submission: {cfg.submission_file} | rows: {len(submission_out)}")

# Final concise summary
print("Summary")
print(f"- Train videos (filtered): {int(train_subset['video_id'].n_unique())}")
print(f"- Labs: {int(train_subset['lab_id'].n_unique())}")
print(f"- Annotations: {len(solution)}")
print(f"- Unique behaviors: {int(solution['action'].n_unique())}")
print(f"- Predictions: {len(submission_out)}")


# Requires: `submission_out` (from Cell 6), `test`
qa_actions = (
    submission_out
    .with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("duration"))
    .group_by("action")
    .agg([
        pl.count().alias("num_events"),
        pl.col("duration").mean().alias("mean_dur"),
        pl.col("duration").median().alias("median_dur"),
    ])
    .sort("num_events", descending=True)
)
qa_videos = (
    submission_out
    .group_by("video_id")
    .agg([
        pl.count().alias("num_events"),
        pl.col("action").n_unique().alias("num_actions"),
    ])
    .sort("num_events", descending=True)
)

print("Per-action summary (top 20):")
display(qa_actions.head(20).to_pandas())

print("\nPer-video summary (top 20):")
display(qa_videos.head(20).to_pandas())


# Requires: predict_without_ml, cfg, test, priors (per_lab, global_w, med_lab, med_glob, timing_lab, timing_glob)

def try_params(min_len: int, gap_close: int):
    preds = predict_without_ml(
        test, "test", cfg,
        priors_per_lab=per_lab, priors_global=global_w,
        meddur_per_lab=med_lab, meddur_global=med_glob,
        timing_lab=timing_lab, timing_global=timing_glob,
        prior_scope="mixed", use_windows=True,
        min_len=min_len, gap_close=gap_close,
        p_min=0.05, cap=0.05
    )
    return preds.select(pl.count()).item()

grid = [(12,3), (10,3), (10,5), (8,5), (6,5)]
results = []
for ml, gc_ in grid:
    try:
        n = try_params(ml, gc_)
        results.append((ml, gc_, int(n)))
        print(f"min_len={ml}, gap_close={gc_} -> events={n}")
    except Exception as e:
        print(f"min_len={ml}, gap_close={gc_} -> failed: {e}")

print("\nGrid results (sorted by events desc):")
for ml, gc_, n in sorted(results, key=lambda x: x[2], reverse=True):
    print(f"- min_len={ml}, gap_close={gc_}, events={n}")


# If you already ran the light sweep cell, reuse `results`. Otherwise, define a default grid fallback.
try:
    assert 'results' in globals() and len(results) > 0
except Exception:
    grid = [(12,3), (10,3), (10,5), (8,5), (6,5)]
    results = []
    for ml, gc_ in grid:
        try:
            n = predict_without_ml(
                test, "test", cfg,
                priors_per_lab=per_lab, priors_global=global_w,
                meddur_per_lab=med_lab, meddur_global=med_glob,
                timing_lab=timing_lab, timing_global=timing_glob,
                prior_scope="mixed", use_windows=True,
                min_len=ml, gap_close=gc_, p_min=0.05, cap=0.05
            ).select(pl.count()).item()
            results.append((ml, gc_, int(n)))
        except Exception as e:
            results.append((ml, gc_, -1))

# Heuristic selector: prefer the top third by event count, then choose the middle of that subset
valid = [(ml, gc_, n) for (ml, gc_, n) in results if n >= 0]
if not valid:
    print("No valid results from sweep; keeping previous submission.")
else:
    valid_sorted = sorted(valid, key=lambda x: x[2], reverse=True)
    top_k = max(1, len(valid_sorted) // 3)
    candidates = valid_sorted[:top_k]
    chosen = candidates[len(candidates)//2]  # middle of the top slice
    chosen_min_len, chosen_gap = chosen[0], chosen[1]

    print("Sweep results (top 10 by events):")
    for r in valid_sorted[:10]:
        print(f"min_len={r[0]}, gap_close={r[1]} -> events={r[2]}")

    print(f"\nChosen params: min_len={chosen_min_len}, gap_close={chosen_gap}")

    # Re-run predictions with chosen params
    submission_tuned = predict_without_ml(
        test, "test", cfg,
        priors_per_lab=per_lab, priors_global=global_w,
        meddur_per_lab=med_lab, meddur_global=med_glob,
        timing_lab=timing_lab, timing_global=timing_glob,
        prior_scope="mixed", use_windows=True,
        min_len=chosen_min_len, gap_close=chosen_gap, p_min=0.05, cap=0.05
    )

    # Quick QA
    qa = (
        submission_tuned
        .with_columns((pl.col("stop_frame") - pl.col("start_frame")).alias("duration"))
        .group_by("action")
        .agg([pl.count().alias("num_events"), pl.col("duration").median().alias("median_dur")])
        .sort("num_events", descending=True)
    )
    print("\nPer-action (top 15):")
    display(qa.head(15).to_pandas())

    # Save tuned submission
    ordered_cols = list(cfg.submission_schema.keys())
    submission_tuned_out = submission_tuned.select(ordered_cols).with_row_index(cfg.row_id_col)
    tuned_path = "submission.csv"  # overwrite as final
    submission_tuned_out.write_csv(tuned_path)
    print(f"\nSaved tuned submission: {tuned_path} | rows: {len(submission_tuned_out)}")

