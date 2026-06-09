%%writefile baseline.py
#!/usr/bin/env python3
import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import polars as pl
from tqdm.auto import tqdm

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

class HostVisibleError(Exception): pass

# ========================
# Utils
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
    casts = []
    for col, dtype in schema.items():
        if col in df.columns and df[col].dtype != dtype:
            casts.append(pl.col(col).cast(dtype))
    return df.with_columns(casts) if casts else df

def validate_frame_ranges(df: pl.DataFrame, name: str) -> None:
    if not (df["start_frame"] <= df["stop_frame"]).all():
        raise ValueError(f"{name}: start_frame > stop_frame detected")

def _norm_mouse_id(x: str | int) -> str:
    s = str(x)
    return s if s.startswith("mouse") else f"mouse{s}"

def _norm_triplet(agent: str | int, target: str | int, action: str) -> str:
    return f"{_norm_mouse_id(agent)},{_norm_mouse_id(target)},{action}"

def _range_frames(start: int, stop: int):
    return range(start, stop)

# ========================
# Metrics
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
            if triple_norm not in active: continue
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

    if not distinct_actions: return 0.0

    beta2 = beta * beta
    f_scores: List[float] = []
    for action in distinct_actions:
        tp, fn, fp = tps[action], fns[action], fps[action]
        denom = (1 + beta2) * tp + beta2 * fn + fp
        f_scores.append(0.0 if denom == 0 else (1 + beta2) * tp / denom)
    return sum(f_scores) / len(f_scores)

def mouse_fbeta(solution: pl.DataFrame, submission: pl.DataFrame, beta: float = 1.0, cfg: Optional[Config] = None) -> float:
    cfg = cfg or Config()
    solution = validate_schema(solution, cfg.solution_schema, "Solution")
    submission = validate_schema(submission, cfg.submission_schema, "Submission")
    validate_frame_ranges(solution, "Solution")
    validate_frame_ranges(submission, "Submission")

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
# Build solution
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
            # cast numeric cols to Int64
            annot = annot.with_columns([
                pl.col("video_id").cast(pl.Int64),
                pl.col("start_frame").cast(pl.Int64),
                pl.col("stop_frame").cast(pl.Int64),
            ])
            annot = annot.select([c for c in cfg.solution_schema.keys() if c in annot.columns])
            records.append(annot)
        except Exception as e:
            print(f"[WARN] Failed to load {annot_path}: {e}")
            continue
    if not records: raise ValueError("No annotation files loaded.")
    solution = pl.concat(records, how="vertical")
    solution = validate_schema(solution, cfg.solution_schema, "Solution")
    return solution

# ========================
# Main
# ========================

def run_submit(cfg: Config, eps: float, prior_scope: str) -> None:
    print("[INFO] Loading train (priors) and test...")
    train = pl.read_csv(cfg.train_csv)
    train_subset = train.filter(~pl.col("lab_id").str.starts_with("MABe22"))
    solution = create_solution_df(train_subset, cfg)

    test = pl.read_csv(cfg.test_csv)
    # simple dummy predictions (all background)
    submission = test.select(["video_id"]).with_columns([
        pl.lit("mouse1").alias("agent_id"),
        pl.lit("mouse2").alias("target_id"),
        pl.lit("background").alias("action"),
        pl.lit(0).alias("start_frame"),
        pl.lit(10).alias("stop_frame"),
    ])
    submission = submission.select(list(cfg.submission_schema.keys())).with_row_index(cfg.row_id_col)
    submission.write_csv(cfg.submission_file)
    print(f"[INFO] Saved submission to {cfg.submission_file}")
    print(submission.head())

def main() -> None:
    parser = argparse.ArgumentParser(description="MABe Mouse Baseline (fixed)")
    parser.add_argument("--mode", choices=["validate", "submit", "all"], default="submit")
    parser.add_argument("--prior-scope", choices=["lab", "global", "mixed"], default="mixed")
    parser.add_argument("--eps", type=float, default=1.0)
    args = parser.parse_args()

    cfg = Config()

    if args.mode in ("submit", "all"):
        run_submit(cfg, args.eps, args.prior_scope)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()



!python baseline.py --mode submit --prior-scope mixed --eps 1.0





