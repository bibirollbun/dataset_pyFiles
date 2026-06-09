!pip install -q --no-index --find-links=/kaggle/input/mabe-package xgboost==3.1.1


import datetime
import gc
import itertools
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

sys.path.append("/kaggle/usr/lib/mabe-f-beta")
from metric import score


# const
INPUT_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
TRAIN_TRACKING_DIR = INPUT_DIR / "train_tracking"
TRAIN_ANNOTATION_DIR = INPUT_DIR / "train_annotation"
TEST_TRACKING_DIR = INPUT_DIR / "test_tracking"

WORKING_DIR = Path("/kaggle/working")

INDEX_COLS = [
    "video_id",
    "agent_mouse_id",
    "target_mouse_id",
    "video_frame",
]

BODY_PARTS = [
    "ear_left",
    "ear_right",
    "nose",
    "neck",
    "body_center",
    "lateral_left",
    "lateral_right",
    "hip_left",
    "hip_right",
    "tail_base",
    "tail_tip",
]

SELF_BEHAVIORS = [
    "biteobject",
    "climb",
    "dig",
    "exploreobject",
    "freeze",
    "genitalgroom",
    "huddle",
    "rear",
    "rest",
    "run",
    "selfgroom",
]

PAIR_BEHAVIORS = [
    "allogroom",
    "approach",
    "attack",
    "attemptmount",
    "avoid",
    "chase",
    "chaseattack",
    "defend",
    "disengage",
    "dominance",
    "dominancegroom",
    "dominancemount",
    "ejaculate",
    "escape",
    "flinch",
    "follow",
    "intromit",
    "mount",
    "reciprocalsniff",
    "shepherd",
    "sniff",
    "sniffbody",
    "sniffface",
    "sniffgenital",
    "submit",
    "tussle",
]


# read data
train_dataframe = pl.read_csv(INPUT_DIR / "train.csv")


# preprocess behavior labels
train_behavior_dataframe = (
    train_dataframe.filter(pl.col("behaviors_labeled").is_not_null())
    .select(
        pl.col("lab_id"),
        pl.col("video_id"),
        pl.col("behaviors_labeled").map_elements(eval, return_dtype=pl.List(pl.Utf8)).alias("behaviors_labeled_list"),
    )
    .explode("behaviors_labeled_list")
    .rename({"behaviors_labeled_list": "behaviors_labeled_element"})
    .select(
        pl.col("lab_id"),
        pl.col("video_id"),
        pl.col("behaviors_labeled_element").str.split(",").list[0].str.replace_all("'", "").alias("agent"),
        pl.col("behaviors_labeled_element").str.split(",").list[1].str.replace_all("'", "").alias("target"),
        pl.col("behaviors_labeled_element").str.split(",").list[2].str.replace_all("'", "").alias("behavior"),
    )
)

train_self_behavior_dataframe = train_behavior_dataframe.filter(pl.col("behavior").is_in(SELF_BEHAVIORS))
train_pair_behavior_dataframe = train_behavior_dataframe.filter(pl.col("behavior").is_in(PAIR_BEHAVIORS))


%%writefile self_features.py

def make_self_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(body_part_1, body_part_2):
        # agentã�®ä½“ã�®å�„éƒ¨ä½�é–“è·�é›¢(cm)
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"agent_x_{body_part_1}") - pl.col(f"agent_x_{body_part_2}")).pow(2)
            + (pl.col(f"agent_y_{body_part_1}") - pl.col(f"agent_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(body_part, period_ms):
        # éƒ¨ä½�ã�®æ�¨å®šé€Ÿåº¦(cm/s)
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            ((pl.col(f"agent_x_{body_part}").diff()).pow(2) + (pl.col(f"agent_y_{body_part}").diff()).pow(2)).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def elongation():
        # ä¼¸é•·åº¦
        d1 = body_parts_distance("nose", "tail_base")
        d2 = body_parts_distance("ear_left", "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle():
        # ä½“è§’åº¦(deg)
        v1x = pl.col("agent_x_nose") - pl.col("agent_x_body_center")
        v1y = pl.col("agent_y_nose") - pl.col("agent_y_body_center")
        v2x = pl.col("agent_x_tail_base") - pl.col("agent_x_body_center")
        v2y = pl.col("agent_y_tail_base") - pl.col("agent_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)

    n_mice = (
        (metadata["mouse1_strain"] is not None)
        + (metadata["mouse2_strain"] is not None)
        + (metadata["mouse3_strain"] is not None)
        + (metadata["mouse4_strain"] is not None)
    )
    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()

    result = []

    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])
    pivot_trackings = {mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id) for mouse_id in range(1, n_mice + 1)}

    for agent_mouse_id in range(1, n_mice + 1):
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": -1,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        pivot = pivot_trackings[agent_mouse_id].select(
            pl.col("video_frame"),
            pl.exclude("video_frame").name.prefix("agent_"),
        )
        columns = pivot.columns
        pivot = pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}") for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}") for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
        )

        features = pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(-1).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            *[
                body_parts_distance(body_part_1, body_part_2).alias(f"aa__{body_part_1}__{body_part_2}__distance")
                for body_part_1, body_part_2 in itertools.combinations(BODY_PARTS, 2)
            ],
            *[
                body_part_speed(body_part, period_ms).alias(f"agent__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            elongation().alias("agent__elongation"),
            body_angle().alias("agent__body_angle"),
        )

        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")


%%writefile pair_features.py

def make_pair_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(agent_or_target_1, body_part_1, agent_or_target_2, body_part_2):
        # agent-targetã�®ä½“ã�®å�„éƒ¨ä½�é–“è·�é›¢(cm)
        assert agent_or_target_1 == "agent" or agent_or_target_1 == "target"
        assert agent_or_target_2 == "agent" or agent_or_target_2 == "target"
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"{agent_or_target_1}_x_{body_part_1}") - pl.col(f"{agent_or_target_2}_x_{body_part_2}")).pow(2)
            + (pl.col(f"{agent_or_target_1}_y_{body_part_1}") - pl.col(f"{agent_or_target_2}_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(agent_or_target, body_part, period_ms):
        # éƒ¨ä½�ã�®æ�¨å®šé€Ÿåº¦(cm/s)
        assert agent_or_target == "agent" or agent_or_target == "target"
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            (
                (pl.col(f"{agent_or_target}_x_{body_part}").diff()).pow(2)
                + (pl.col(f"{agent_or_target}_y_{body_part}").diff()).pow(2)
            ).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True)

    def elongation(agent_or_target):
        # ä¼¸é•·åº¦(cm)
        assert agent_or_target == "agent" or agent_or_target == "target"
        d1 = body_parts_distance(agent_or_target, "nose", agent_or_target, "tail_base")
        d2 = body_parts_distance(agent_or_target, "ear_left", agent_or_target, "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle(agent_or_target):
        # ä½“è§’åº¦(deg)
        assert agent_or_target == "agent" or agent_or_target == "target"
        v1x = pl.col(f"{agent_or_target}_x_nose") - pl.col(f"{agent_or_target}_x_body_center")
        v1y = pl.col(f"{agent_or_target}_y_nose") - pl.col(f"{agent_or_target}_y_body_center")
        v2x = pl.col(f"{agent_or_target}_x_tail_base") - pl.col(f"{agent_or_target}_x_body_center")
        v2y = pl.col(f"{agent_or_target}_y_tail_base") - pl.col(f"{agent_or_target}_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)

    def body_center_distance_rolling_agg(agg, period_ms):
        # è·�é›¢ã�®ç§»å‹•é›†è¨ˆç‰¹å¾´é‡�
        assert agg in ["mean", "std", "var", "min", "max"] # é›†è¨ˆé–¢æ•°
        expr = body_parts_distance("agent", "body_center", "target", "body_center")
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))

        if agg == "mean":
            return expr.rolling_mean(window_size=window_frames, center=True, min_samples=1)
        elif agg == "std":
            return expr.rolling_std(window_size=window_frames, center=True, min_samples=1)
        elif agg == "var":
            return expr.rolling_var(window_size=window_frames, center=True, min_samples=1)
        elif agg == "min":
            return expr.rolling_min(window_size=window_frames, center=True, min_samples=1)
        elif agg == "max":
            return expr.rolling_max(window_size=window_frames, center=True, min_samples=1)
        else:
            raise ValueError()

    n_mice = (
        (metadata["mouse1_strain"] is not None)
        + (metadata["mouse2_strain"] is not None)
        + (metadata["mouse3_strain"] is not None)
        + (metadata["mouse4_strain"] is not None)
    )
    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()

    result = []

    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])
    pivot_trackings = {mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id) for mouse_id in range(1, n_mice + 1)}

    for agent_mouse_id, target_mouse_id in itertools.permutations(range(1, n_mice + 1), 2):
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": target_mouse_id,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        merged_pivot = (
            pivot_trackings[agent_mouse_id]
            .select(
                pl.col("video_frame"),
                pl.exclude("video_frame").name.prefix("agent_"),
            )
            .join(
                pivot_trackings[target_mouse_id].select(
                    pl.col("video_frame"),
                    pl.exclude("video_frame").name.prefix("target_"),
                ),
                on="video_frame",
                how="inner",
            )
        )
        columns = merged_pivot.columns
        merged_pivot = merged_pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}") for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}") for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_x_{bp}") for bp in BODY_PARTS if f"target_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_y_{bp}") for bp in BODY_PARTS if f"target_y_{bp}" not in columns],
        )

        features = merged_pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(target_mouse_id).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            *[
                body_parts_distance("agent", agent_body_part, "target", target_body_part).alias(
                    f"at__{agent_body_part}__{target_body_part}__distance"
                )
                for agent_body_part, target_body_part in itertools.product(BODY_PARTS, repeat=2)
            ],
            *[
                body_part_speed("agent", body_part, period_ms).alias(f"agent__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            *[
                body_part_speed("target", body_part, period_ms).alias(f"target__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            elongation("agent").alias("agent__elongation"),
            elongation("target").alias("target__elongation"),
            body_angle("agent").alias("agent__body_angle"),
            body_angle("target").alias("target__body_angle"),
        )

        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")


%run -i self_features.py
%run -i pair_features.py

def process_video(row):
    """Process a single video to extract self and pair features."""
    lab_id = row["lab_id"]
    video_id = row["video_id"]

    tracking_path = TRAIN_TRACKING_DIR / f"{lab_id}/{video_id}.parquet"
    tracking = pl.read_parquet(tracking_path)

    self_features = make_self_features(metadata=row, tracking=tracking)
    pair_features = make_pair_features(metadata=row, tracking=tracking)

    self_features.write_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
    pair_features.write_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")

    return video_id


# make data
(WORKING_DIR / "self_features").mkdir(exist_ok=True, parents=True)
(WORKING_DIR / "pair_features").mkdir(exist_ok=True, parents=True)

rows = list(train_dataframe.filter(pl.col("behaviors_labeled").is_not_null()).rows(named=True))
results = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(process_video)(row) for row in rows)

print(f"Processed {len(results)} videos successfully")

del rows, results
gc.collect()


def tune_threshold(oof_action, y_action):
    thresholds = np.arange(0, 1.005, 0.005)
    scores = [f1_score(y_action, (oof_action >= th), zero_division=0) for th in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx]


def train_validate(lab_id: str, behavior: str, indices: pl.DataFrame, features: pl.DataFrame, labels: pl.Series):
    # çµ�æ�œã‚’ä¿�å­˜ã�™ã‚‹ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�®ãƒ‘ã‚¹ã‚’ä½œæˆ�
    result_dir = WORKING_DIR / "results" / lab_id / behavior
    # ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�Œå­˜åœ¨ã�—ã�ªã�„å ´å�ˆã�¯ä½œæˆ�ï¼ˆè¦ªãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã‚‚å�«ã‚�ã�¦ï¼‰
    result_dir.mkdir(exist_ok=True, parents=True)

    # ãƒ©ãƒ™ãƒ«ã�®å�ˆè¨ˆã�Œ0ã�®å ´å�ˆï¼ˆæ­£ä¾‹ã�Œ1ã�¤ã‚‚ã�ªã�„å ´å�ˆï¼‰ã�®å‡¦ç�†
    if labels.sum() == 0:
        # F1ã‚¹ã‚³ã‚¢ã‚’0ã�¨ã�—ã�¦ä¿�å­˜
        with open(result_dir / "f1.txt", "w") as f:
            f.write("0.0\n")
        # ã�™ã�¹ã�¦ã�®äºˆæ¸¬å€¤ã‚’0ã�¨ã�—ã�Ÿçµ�æ�œãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã‚’ä½œæˆ�
        oof_prediction_dataframe = indices.with_columns(
            pl.Series("fold", [-1] * len(labels), dtype=pl.Int8),  # ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ç•ªå�·ï¼ˆ-1ã�¯æœªä½¿ç”¨ã‚’æ„�å‘³ï¼‰
            pl.Series("prediction", [0.0] * len(labels), dtype=pl.Float32),  # äºˆæ¸¬ç¢ºç�‡
            pl.Series("predicted_label", [0] * len(labels), dtype=pl.Int8),  # äºˆæ¸¬ãƒ©ãƒ™ãƒ«
        )
        # çµ�æ�œã‚’parquetå½¢å¼�ã�§ä¿�å­˜
        oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")
        return 0.0

    # Out-of-Foldäºˆæ¸¬çµ�æ�œã‚’ä¿�å­˜ã�™ã‚‹ã�Ÿã‚�ã�®é…�åˆ—ã‚’åˆ�æœŸåŒ–
    folds = np.ones(len(labels), dtype=np.int8) * -1  # å�„ã‚µãƒ³ãƒ—ãƒ«ã�Œå±�ã�™ã‚‹ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ç•ªå�·
    oof_predictions = np.zeros(len(labels), dtype=np.float32)  # äºˆæ¸¬ç¢ºç�‡
    oof_prediction_labels = np.zeros(len(labels), dtype=np.int8)  # äºˆæ¸¬ãƒ©ãƒ™ãƒ«ï¼ˆ0ã�¾ã�Ÿã�¯1ï¼‰

    # 3åˆ†å‰²ã�®å±¤åŒ–ã‚°ãƒ«ãƒ¼ãƒ—äº¤å·®æ¤œè¨¼ã‚’å®Ÿè¡Œ
    # StratifiedGroupKFoldã�¯ã€�ãƒ©ãƒ™ãƒ«ã�®åˆ†å¸ƒã‚’ä¿�ã�¡ã�¤ã�¤ã€�å�Œã�˜ã‚°ãƒ«ãƒ¼ãƒ—ï¼ˆvideo_idï¼‰ã�Œè¤‡æ•°ã�®ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�«åˆ†ã�‹ã‚Œã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹
    for fold, (train_idx, valid_idx) in enumerate(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
            X=features,  # ç‰¹å¾´é‡�
            y=labels,  # ãƒ©ãƒ™ãƒ«
            groups=indices.get_column("video_id"),  # ã‚°ãƒ«ãƒ¼ãƒ—åŒ–ã�®åŸºæº–ï¼ˆå�Œã�˜å‹•ç”»IDã�¯å�Œã�˜ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�«ï¼‰
        )
    ):
        # å�„ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�®çµ�æ�œã‚’ä¿�å­˜ã�™ã‚‹ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã‚’ä½œæˆ�
        result_dir_fold = result_dir / f"fold_{fold}"
        result_dir_fold.mkdir(exist_ok=True, parents=True)

        # è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�«åˆ†å‰²
        X_train = features[train_idx]  # è¨“ç·´ç”¨ç‰¹å¾´é‡�
        y_train = labels[train_idx]  # è¨“ç·´ç”¨ãƒ©ãƒ™ãƒ«
        X_valid = features[valid_idx]  # æ¤œè¨¼ç”¨ç‰¹å¾´é‡�
        y_valid = labels[valid_idx]  # æ¤œè¨¼ç”¨ãƒ©ãƒ™ãƒ«

        # ã‚¯ãƒ©ã‚¹ä¸�å�‡è¡¡ã�«å¯¾å‡¦ã�™ã‚‹ã�Ÿã‚�ã�®é‡�ã�¿ã‚’è¨ˆç®—
        # è² ä¾‹ã�®æ•° / æ­£ä¾‹ã�®æ•° = æ­£ä¾‹ã�«ã�‹ã�‘ã‚‹é‡�ã�¿
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        # XGBoostã�®ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã‚’è¨­å®š
        params = {
            "objective": "binary:logistic",  # äºŒå€¤åˆ†é¡�å•�é¡Œ
            "eval_metric": "logloss",  # è©•ä¾¡æŒ‡æ¨™ï¼šå¯¾æ•°æ��å¤±
            "device": "cpu",  # ä½¿ç”¨ãƒ‡ãƒ�ã‚¤ã‚¹
            "tree_method": "hist",  # ãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ ãƒ™ãƒ¼ã‚¹ã�®é«˜é€Ÿã�ªã‚¢ãƒ«ã‚´ãƒªã‚ºãƒ 
            "learning_rate": 0.05,  # å­¦ç¿’ç�‡
            "max_depth": 6,  # æœ¨ã�®æœ€å¤§æ·±ã�•
            "min_child_weight": 5,  # å­�ãƒ�ãƒ¼ãƒ‰ã�®æœ€å°�é‡�ã�¿
            "subsample": 0.8,  # å�„æœ¨ã�§ä½¿ç”¨ã�™ã‚‹ã‚µãƒ³ãƒ—ãƒ«ã�®å‰²å�ˆ
            "colsample_bytree": 0.8,  # å�„æœ¨ã�§ä½¿ç”¨ã�™ã‚‹ç‰¹å¾´é‡�ã�®å‰²å�ˆ
            "scale_pos_weight": scale_pos_weight,  # æ­£ä¾‹ã�®é‡�ã�¿
            "max_bin": 64,  # ãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ ã�®ãƒ“ãƒ³æ•°
            "seed": 42,  # ä¹±æ•°ã‚·ãƒ¼ãƒ‰
        }
        
        # XGBoostç”¨ã�®ãƒ‡ãƒ¼ã‚¿è¡Œåˆ—ã‚’ä½œæˆ�ï¼ˆè¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¯é‡�å­�åŒ–è¡Œåˆ—ã€�æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�¯é€šå¸¸ã�®è¡Œåˆ—ï¼‰
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train, feature_names=features.columns, max_bin=64)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=features.columns)

        # è©•ä¾¡çµ�æ�œã‚’ä¿�å­˜ã�™ã‚‹è¾�æ›¸
        evals_result = {}
        
        # æ—©æœŸçµ‚äº†ã�®ã‚³ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯ã‚’è¨­å®š
        # æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�®å¯¾æ•°æ��å¤±ã�Œ10ãƒ©ã‚¦ãƒ³ãƒ‰æ”¹å–„ã�—ã�ªã�„å ´å�ˆã€�å­¦ç¿’ã‚’å�œæ­¢
        early_stopping_callback = xgb.callback.EarlyStopping(
            rounds=10,  # æ”¹å–„ã�Œè¦‹ã‚‰ã‚Œã�ªã�„é€£ç¶šãƒ©ã‚¦ãƒ³ãƒ‰æ•°
            metric_name="logloss",  # ç›£è¦–ã�™ã‚‹æŒ‡æ¨™
            data_name="valid",  # ç›£è¦–ã�™ã‚‹ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆ
            maximize=False,  # å°�ã�•ã�„æ–¹ã�Œè‰¯ã�„æŒ‡æ¨™
            save_best=True,  # æœ€è‰¯ã�®ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜
        )
        
        # ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ã‚’å®Ÿè¡Œ
        model = xgb.train(
            params,  # ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
            dtrain=dtrain,  # è¨“ç·´ãƒ‡ãƒ¼ã‚¿
            num_boost_round=250,  # æœ€å¤§ãƒ–ãƒ¼ã‚¹ãƒ†ã‚£ãƒ³ã‚°ãƒ©ã‚¦ãƒ³ãƒ‰æ•°
            evals=[(dtrain, "train"), (dvalid, "valid")],  # è©•ä¾¡ã�™ã‚‹ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆ
            callbacks=[early_stopping_callback],  # ã‚³ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯
            evals_result=evals_result,  # è©•ä¾¡çµ�æ�œã�®ä¿�å­˜å…ˆ
            verbose_eval=0,  # ãƒ­ã‚°å‡ºåŠ›ã�®é »åº¦ï¼ˆ0ã�¯å‡ºåŠ›ã�ªã�—ï¼‰
        )

        # æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�«å¯¾ã�—ã�¦äºˆæ¸¬ã‚’å®Ÿè¡Œï¼ˆç¢ºç�‡å€¤ã‚’å�–å¾—ï¼‰
        fold_predictions = model.predict(dvalid)

        # F1ã‚¹ã‚³ã‚¢ã‚’æœ€å¤§åŒ–ã�™ã‚‹æœ€é�©ã�ªé–¾å€¤ã‚’èª¿æ•´
        threshold = tune_threshold(fold_predictions, y_valid)
        
        # Out-of-Foldäºˆæ¸¬çµ�æ�œã‚’ä¿�å­˜
        folds[valid_idx] = fold  # ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ç•ªå�·
        oof_predictions[valid_idx] = fold_predictions  # äºˆæ¸¬ç¢ºç�‡
        oof_prediction_labels[valid_idx] = (fold_predictions >= threshold).astype(np.int8)  # é–¾å€¤ã�§äºŒå€¤åŒ–

        # ã�“ã�®ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�®çµ�æ�œã‚’ä¿�å­˜
        # å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜
        model.save_model(result_dir_fold / "model.json")
        # æœ€é�©ã�ªé–¾å€¤ã‚’ä¿�å­˜
        with open(result_dir_fold / "threshold.txt", "w") as f:
            f.write(f"{threshold}\n")

        # ç‰¹å¾´é‡�ã�®é‡�è¦�åº¦ã‚’ãƒ—ãƒ­ãƒƒãƒˆï¼ˆä¸Šä½�20å€‹ã€�ã‚²ã‚¤ãƒ³åŸºæº–ï¼‰
        xgb.plot_importance(model, max_num_features=20, importance_type="gain", values_format="{v:.2f}")
        plt.tight_layout()
        plt.savefig(result_dir_fold / "feature_importance.png")
        plt.close()

        # å­¦ç¿’æ›²ç·šï¼ˆå¯¾æ•°æ��å¤±ã�®æ�¨ç§»ï¼‰ã‚’ãƒ—ãƒ­ãƒƒãƒˆ
        lgb.plot_metric(evals_result, metric="logloss")
        plt.tight_layout()
        plt.savefig(result_dir_fold / "metric.png")
        plt.close()

        # ãƒ¡ãƒ¢ãƒªã‚’è§£æ”¾
        gc.collect()

    # ã�™ã�¹ã�¦ã�®ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�®äºˆæ¸¬çµ�æ�œã‚’ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã�«ã�¾ã�¨ã‚�ã‚‹
    oof_prediction_dataframe = indices.with_columns(
        pl.Series("fold", folds, dtype=pl.Int8),  # ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ç•ªå�·
        pl.Series("prediction", oof_predictions, dtype=pl.Float32),  # äºˆæ¸¬ç¢ºç�‡
        pl.Series("predicted_label", oof_prediction_labels, dtype=pl.Int8),  # äºˆæ¸¬ãƒ©ãƒ™ãƒ«
    )
    
    # å…¨ä½“ã�®F1ã‚¹ã‚³ã‚¢ã‚’è¨ˆç®—
    f1 = f1_score(labels, oof_prediction_labels, zero_division=0)
    # F1ã‚¹ã‚³ã‚¢ã‚’ãƒ•ã‚¡ã‚¤ãƒ«ã�«ä¿�å­˜
    with open(result_dir / "f1.txt", "w") as f:
        f.write(f"{f1}\n")

    # äºˆæ¸¬çµ�æ�œãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã‚’ä¿�å­˜
    oof_prediction_dataframe.write_parquet(result_dir / "oof_predictions.parquet")

    # F1ã‚¹ã‚³ã‚¢ã‚’è¿”ã�™
    return f1


groups = train_self_behavior_dataframe.group_by("lab_id", "behavior", maintain_order=True)
total_groups = len(list(groups))
start_time = time.perf_counter()

for idx, ((lab_id, behavior), group) in tqdm(enumerate(groups), total=total_groups):
    if idx == 0:
        tqdm.write(
            f"|{'LAB':^25}|{'BEHAVIOR':^15}|{'SAMPLES':^10}|{'POSITIVE':^10}|{'FEATURES':^10}|{'F1':^10}|{'ELAPSED TIME':^15}|",
            end="\n",
        )

    tqdm.write(f"|{lab_id:^25}|{behavior:^15}|", end="")
    index_list = []
    feature_list = []
    label_list = []

    for row in group.rows(named=True):
        video_id = row["video_id"]
        agent = row["agent"]

        agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))

        data = pl.scan_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet").filter(
            (pl.col("agent_mouse_id") == agent_mouse_id)
        )
        index = data.select(INDEX_COLS).collect(engine="streaming")
        feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")

        # read annotation
        annotation_path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
        if annotation_path.exists():
            annotation = (
                pl.scan_parquet(annotation_path)
                .filter((pl.col("action") == behavior) & (pl.col("agent_id") == agent_mouse_id))
                .collect()
            )
        else:
            annotation = pl.DataFrame(
                schema={
                    "agent_id": pl.Int8,
                    "target_id": pl.Int8,
                    "action": str,
                    "start_frame": pl.Int16,
                    "stop_frame": pl.Int16,
                }
            )

        label_frames = set()
        for annotation_row in annotation.rows(named=True):
            label_frames.update(range(annotation_row["start_frame"], annotation_row["stop_frame"]))
        label = index.select(pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label"))

        if label.get_column("label").sum() == 0:
            continue

        index_list.append(index)
        feature_list.append(feature)
        label_list.append(label.get_column("label"))

    if not index_list:
        elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
        tqdm.write(f"{0:>10,}|{0:>10,}|{0:>10,}|{'-':>10}|{str(elapsed_time):>15}|", end="\n")
        continue

    indices = pl.concat(index_list, how="vertical")
    features = pl.concat(feature_list, how="vertical")
    labels = pl.concat(label_list, how="vertical")

    del index_list, feature_list, label_list
    gc.collect()

    tqdm.write(f"{len(indices):>10,}|{labels.sum():>10,}|{len(features.columns):>10,}|", end="")

    f1 = train_validate(lab_id, behavior, indices, features, labels)
    tqdm.write(f"{f1:>10.2f}|", end="")

    elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
    tqdm.write(f"{str(elapsed_time):>15}|", end="\n")

    gc.collect()


groups = train_pair_behavior_dataframe.group_by("lab_id", "behavior", maintain_order=True)
total_groups = len(list(groups))
start_time = time.perf_counter()

for idx, ((lab_id, behavior), group) in tqdm(enumerate(groups), total=total_groups):
    if idx == 0:
        tqdm.write(
            f"|{'LAB':^25}|{'BEHAVIOR':^15}|{'SAMPLES':^10}|{'POSITIVE':^10}|{'FEATURES':^10}|{'F1':^10}|{'ELAPSED TIME':^15}|",
            end="\n",
        )

    tqdm.write(f"|{lab_id:^25}|{behavior:^15}|", end="")
    index_list = []
    feature_list = []
    label_list = []

    for row in group.rows(named=True):
        video_id = row["video_id"]
        agent = row["agent"]
        target = row["target"]

        agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
        target_mouse_id = int(re.search(r"mouse(\d+)", target).group(1))

        data = pl.scan_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet").filter(
            (pl.col("agent_mouse_id") == agent_mouse_id) & (pl.col("target_mouse_id") == target_mouse_id)
        )
        index = data.select(INDEX_COLS).collect(engine="streaming")
        feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")

        # read annotation
        annotation_path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
        if annotation_path.exists():
            annotation = (
                pl.scan_parquet(annotation_path)
                .filter(
                    (pl.col("action") == behavior)
                    & (pl.col("agent_id") == agent_mouse_id)
                    & (pl.col("target_id") == target_mouse_id)
                )
                .collect()
            )
        else:
            annotation = pl.DataFrame(
                schema={
                    "agent_id": pl.Int8,
                    "target_id": pl.Int8,
                    "action": str,
                    "start_frame": pl.Int16,
                    "stop_frame": pl.Int16,
                }
            )

        label_frames = set()
        for annotation_row in annotation.rows(named=True):
            label_frames.update(range(annotation_row["start_frame"], annotation_row["stop_frame"]))
        label = index.select(pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label"))

        if label.get_column("label").sum() == 0:
            continue

        index_list.append(index)
        feature_list.append(feature)
        label_list.append(label.get_column("label"))

    if not index_list:
        elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
        tqdm.write(f"{0:>10,}|{0:>10,}|{0:>10,}|{'-':>10}|{str(elapsed_time):>15}|", end="\n")
        continue

    indices = pl.concat(index_list, how="vertical")
    features = pl.concat(feature_list, how="vertical")
    labels = pl.concat(label_list, how="vertical")

    del index_list, feature_list, label_list
    gc.collect()

    tqdm.write(f"{len(indices):>10,}|{labels.sum():>10,}|{len(features.columns):>10,}|", end="")

    f1 = train_validate(lab_id, behavior, indices, features, labels)
    tqdm.write(f"{f1:>10.2f}|", end="")

    elapsed_time = datetime.timedelta(seconds=int(time.perf_counter() - start_time))
    tqdm.write(f"{str(elapsed_time):>15}|", end="\n")

    gc.collect()


%%writefile robustify.py

def robustify(submission: pl.DataFrame, dataset: pl.DataFrame, train_test: str = "train"):
    traintest_directory = INPUT_DIR / f"{train_test}_tracking"

    old_submission = submission.clone()
    submission = submission.filter(pl.col("start_frame") < pl.col("stop_frame"))
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")

    old_submission = submission.clone()
    group_list = []
    for _, group in submission.group_by("video_id", "agent_id", "target_id"):
        group = group.sort("start_frame")
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, row in enumerate(group.rows(named=True)):
            if row["start_frame"] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row["stop_frame"]
        group_list.append(group.filter(pl.Series("mask", mask)))

    submission = pl.concat(group_list)

    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")

    s_list = []
    for row in dataset.rows(named=True):
        lab_id = row["lab_id"]
        video_id = row["video_id"]
        if row["behaviors_labeled"] is None:
            continue

        if video_id in submission.get_column("video_id").to_list():
            continue

        if isinstance(row["behaviors_labeled"], str):
            continue

        print(f"Video {video_id} has no predictions.")

        path = traintest_directory / f"/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = json.loads(row["behaviors_labeled"])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(",") for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=["agent", "target", "action"])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(["agent", "target"]):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, action_row in enumerate(actions.itertuples(index=False)):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row["action"], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat(
            [
                submission,
                pd.DataFrame(s_list, columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]),
            ]
        )
        print("ERROR: Filled empty videos")

    return submission


# ã‚°ãƒ«ãƒ¼ãƒ—ã�”ã�¨ã�®Out-of-Foldäºˆæ¸¬çµ�æ�œã‚’ä¿�å­˜ã�™ã‚‹ãƒªã‚¹ãƒˆ
group_oof_predictions = []

# ãƒ‡ãƒ¼ã‚¿ã‚’ lab_id, video_id, agent, target ã�§ã‚°ãƒ«ãƒ¼ãƒ—åŒ–
# maintain_order=True ã�§å…ƒã�®é †åº�ã‚’ä¿�æŒ�
groups = train_behavior_dataframe.group_by("lab_id", "video_id", "agent", "target", maintain_order=True)

# å�„ã‚°ãƒ«ãƒ¼ãƒ—ã�«å¯¾ã�—ã�¦å‡¦ç�†ã‚’å®Ÿè¡Œï¼ˆé€²æ�—ãƒ�ãƒ¼ã‚’è¡¨ç¤ºï¼‰
for (lab_id, video_id, agent, target), group in tqdm(groups, total=len(list(groups))):
    # agentï¼ˆè¡Œå‹•ä¸»ä½“ï¼‰ã�‹ã‚‰ãƒ�ã‚¦ã‚¹IDã‚’æŠ½å‡º
    # ä¾‹: "mouse1" â†’ 1
    agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
    
    # targetï¼ˆè¡Œå‹•å¯¾è±¡ï¼‰ã�‹ã‚‰ãƒ�ã‚¦ã‚¹IDã‚’æŠ½å‡º
    # "self"ï¼ˆè‡ªåˆ†è‡ªèº«ï¼‰ã�®å ´å�ˆã�¯ -1ã€�ã��ã‚Œä»¥å¤–ã�¯ãƒ�ã‚¦ã‚¹IDã‚’å�–å¾—
    target_mouse_id = -1 if target == "self" else int(re.search(r"mouse(\d+)", target).group(1))

    # ã�“ã�®ã‚°ãƒ«ãƒ¼ãƒ—ã�®å�„è¡Œå‹•ã�®äºˆæ¸¬çµ�æ�œã‚’ä¿�å­˜ã�™ã‚‹ãƒªã‚¹ãƒˆ
    prediction_dataframe_list = []

    # ã‚°ãƒ«ãƒ¼ãƒ—å†…ã�®å�„è¡Œï¼ˆå�„è¡Œå‹•ï¼‰ã‚’å‡¦ç�†
    for row in group.rows(named=True):
        behavior = row["behavior"]  # è¡Œå‹•ã�®ç¨®é¡�ï¼ˆä¾‹: "grooming", "sniffing"ã�ªã�©ï¼‰

        # ã�“ã�®è¡Œå‹•ã�®OOFäºˆæ¸¬çµ�æ�œãƒ•ã‚¡ã‚¤ãƒ«ã�®ãƒ‘ã‚¹ã‚’æ§‹ç¯‰
        oof_path = WORKING_DIR / "results" / lab_id / behavior / "oof_predictions.parquet"
        
        # ãƒ•ã‚¡ã‚¤ãƒ«ã�Œå­˜åœ¨ã�—ã�ªã�„å ´å�ˆã�¯ã‚¹ã‚­ãƒƒãƒ—
        if not oof_path.exists():
            continue

        # äºˆæ¸¬çµ�æ�œã‚’èª­ã�¿è¾¼ã�¿ã€�è©²å½“ã�™ã‚‹video_idã€�agentã€�targetã�§ãƒ•ã‚£ãƒ«ã‚¿ãƒªãƒ³ã‚°
        prediction = (
            pl.scan_parquet(oof_path)  # é�…å»¶èª­ã�¿è¾¼ã�¿ï¼ˆãƒ¡ãƒ¢ãƒªåŠ¹ç�‡çš„ï¼‰
            .filter(
                (pl.col("video_id") == video_id)  # å‹•ç”»IDã�Œä¸€è‡´
                & (pl.col("agent_mouse_id") == agent_mouse_id)  # è¡Œå‹•ä¸»ä½“ã�Œä¸€è‡´
                & (pl.col("target_mouse_id") == target_mouse_id)  # è¡Œå‹•å¯¾è±¡ã�Œä¸€è‡´
            )
            .select(
                *INDEX_COLS,  # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ã‚’é�¸æŠ�
                # äºˆæ¸¬ç¢ºç�‡ã�¨äºˆæ¸¬ãƒ©ãƒ™ãƒ«ã‚’æ�›ã�‘å�ˆã‚�ã�›ã�¦ã€�ã�“ã�®è¡Œå‹•ã�®ã‚¹ã‚³ã‚¢ã‚’è¨ˆç®—
                # äºˆæ¸¬ãƒ©ãƒ™ãƒ«ã�Œ0ã�®å ´å�ˆã�¯ã‚¹ã‚³ã‚¢ã‚‚0ã�«ã�ªã‚‹
                (pl.col("prediction") * pl.col("predicted_label")).alias(behavior)
            )
            .collect()  # å®Ÿéš›ã�«ãƒ‡ãƒ¼ã‚¿ã‚’èª­ã�¿è¾¼ã‚“ã�§å®Ÿè¡Œ
        )

        # ãƒ•ã‚£ãƒ«ã‚¿å¾Œã�«è¡Œã�Œã�ªã�„å ´å�ˆï¼ˆè©²å½“ãƒ‡ãƒ¼ã‚¿ã�Œã�ªã�„å ´å�ˆï¼‰ã�¯ã‚¹ã‚­ãƒƒãƒ—
        if len(prediction) == 0:
            continue

        # ã�“ã�®è¡Œå‹•ã�®äºˆæ¸¬çµ�æ�œã‚’ãƒªã‚¹ãƒˆã�«è¿½åŠ 
        prediction_dataframe_list.append(prediction)

    # ã�“ã�®ã‚°ãƒ«ãƒ¼ãƒ—ã�§äºˆæ¸¬çµ�æ�œã�Œ1ã�¤ã‚‚ã�ªã�„å ´å�ˆã�¯ã‚¹ã‚­ãƒƒãƒ—
    if not prediction_dataframe_list:
        continue

    # è¤‡æ•°ã�®è¡Œå‹•ã�®äºˆæ¸¬çµ�æ�œã‚’æ¨ªæ–¹å�‘ã�«çµ�å�ˆ
    # how="align"ã�§ã€�ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ã‚’åŸºæº–ã�«æ•´åˆ—ã�—ã�¦çµ�å�ˆ
    prediction_dataframe = pl.concat(prediction_dataframe_list, how="align")

    # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ä»¥å¤–ã�®åˆ—å��ï¼ˆå�„è¡Œå‹•å��ï¼‰ã‚’å�–å¾—
    cols = prediction_dataframe.select(pl.exclude(INDEX_COLS)).columns
    
    # å�„ãƒ•ãƒ¬ãƒ¼ãƒ ã�§æœ€ã‚‚ç¢ºä¿¡åº¦ã�®é«˜ã�„è¡Œå‹•ã‚’é�¸æŠ�
    prediction_labels_dataframe = prediction_dataframe.with_columns(
        pl.struct(pl.exclude(INDEX_COLS))  # å…¨è¡Œå‹•ã�®ã‚¹ã‚³ã‚¢ã‚’æ§‹é€ ä½“ã�«ã�¾ã�¨ã‚�ã‚‹
        .map_elements(
            # å�„è¡Œã�«å¯¾ã�—ã�¦å®Ÿè¡Œã�™ã‚‹é–¢æ•°
            lambda row: "none" if sum(row.values()) == 0  # å…¨ã‚¹ã‚³ã‚¢ã�Œ0ã�ªã‚‰"none"
                       else (cols[np.argmax(list(row.values()))]),  # æœ€å¤§ã‚¹ã‚³ã‚¢ã�®è¡Œå‹•ã‚’é�¸æŠ�
            return_dtype=pl.String,
        )
        .alias("prediction")  # æ–°ã�—ã�„åˆ—å��ã‚’"prediction"ã�¨ã�™ã‚‹
    ).select(INDEX_COLS + ["prediction"])  # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ã�¨äºˆæ¸¬åˆ—ã�®ã�¿ã‚’é�¸æŠ�

    # é€£ç¶šã�™ã‚‹å�Œã�˜è¡Œå‹•ã‚’ã�¾ã�¨ã‚�ã�¦ã€�è¡Œå‹•ã�®é–‹å§‹ã�¨çµ‚äº†ãƒ•ãƒ¬ãƒ¼ãƒ ã‚’ç‰¹å®š
    group_oof_prediction = (
        prediction_labels_dataframe
        .filter((pl.col("prediction") != pl.col("prediction").shift(1)))  # å‰�ã�®è¡Œã�¨ç•°ã�ªã‚‹è¡Œå‹•ã�®ã�¿ã‚’æŠ½å‡ºï¼ˆå¢ƒç•Œç‚¹ï¼‰
        .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))  # æ¬¡ã�®å¢ƒç•Œç‚¹ã‚’çµ‚äº†ãƒ•ãƒ¬ãƒ¼ãƒ ã�¨ã�™ã‚‹
        .filter(pl.col("prediction") != "none")  # "none"ï¼ˆè¡Œå‹•ã�ªã�—ï¼‰ã‚’é™¤å¤–
        .select(
            pl.col("video_id"),  # å‹•ç”»ID
            ("mouse" + pl.col("agent_mouse_id").cast(str)).alias("agent_id"),  # "mouse1"å½¢å¼�ã�«å¤‰æ�›
            # target_mouse_idã�Œ-1ã�ªã‚‰"self"ã€�ã��ã‚Œä»¥å¤–ã�¯"mouse2"å½¢å¼�ã�«å¤‰æ�›
            pl.when(pl.col("target_mouse_id") == -1)
            .then(pl.lit("self"))
            .otherwise("mouse" + pl.col("target_mouse_id").cast(str))
            .alias("target_id"),
            pl.col("prediction").alias("action"),  # è¡Œå‹•å��
            pl.col("video_frame").alias("start_frame"),  # é–‹å§‹ãƒ•ãƒ¬ãƒ¼ãƒ 
            pl.col("stop_frame"),  # çµ‚äº†ãƒ•ãƒ¬ãƒ¼ãƒ 
        )
    )

    # ã�“ã�®ã‚°ãƒ«ãƒ¼ãƒ—ã�®äºˆæ¸¬çµ�æ�œã‚’ãƒªã‚¹ãƒˆã�«è¿½åŠ 
    group_oof_predictions.append(group_oof_prediction)

%run -i robustify.py

oof_predictions = pl.concat(group_oof_predictions, how="vertical")
oof_predictions = robustify(oof_predictions, train_dataframe, train_test="train")
oof_predictions.with_row_index("row_id").write_csv(WORKING_DIR / "oof_predictions.csv")



def compute_validation_metrics(submission, verbose=True):
    """Compute and display validation metrics for single vs pair behaviors."""
    # solution_df
    dataset = pl.read_csv(INPUT_DIR / "train.csv").to_pandas()

    solution = []
    for _, row in dataset.iterrows():
        lab_id = row["lab_id"]
        if lab_id.startswith("MABe22"):
            continue

        video_id = row["video_id"]
        path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            continue

        annot["lab_id"] = lab_id
        annot["video_id"] = video_id
        annot["behaviors_labeled"] = row["behaviors_labeled"]
        annot["target_id"] = np.where(
            annot.target_id != annot.agent_id, annot["target_id"].apply(lambda s: f"mouse{s}"), "self"
        )
        annot["agent_id"] = annot["agent_id"].apply(lambda s: f"mouse{s}")
        solution.append(annot)

    solution = pd.concat(solution)

    try:
        # Separate single and pair behaviors
        submission_single = submission[submission["target_id"] == "self"].copy()
        submission_pair = submission[submission["target_id"] != "self"].copy()

        # Filter solution to match submission videos
        solution_videos = set(submission["video_id"].unique())
        solution = solution[solution["video_id"].isin(solution_videos)]

        if len(solution) == 0:
            return

        # Compute overall F1 score
        overall_f1 = score(solution, submission, "row_id", beta=1.0)
        print(f"\n{'=' * 60}")
        print("PERFORMANCE METRICS")
        print(f"{'=' * 60}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Total predictions: {len(submission)}")
        print(f"  - Single behaviors: {len(submission_single)}")
        print(f"  - Pair behaviors: {len(submission_pair)}")

        # Compute per-action F1 scores using existing scoring function
        solution_pl = pl.DataFrame(solution)
        submission_pl = pl.DataFrame(submission)

        # Add label_key and prediction_key
        solution_pl = solution_pl.with_columns(
            pl.concat_str(
                [
                    pl.col("video_id").cast(pl.Utf8),
                    pl.col("agent_id").cast(pl.Utf8),
                    pl.col("target_id").cast(pl.Utf8),
                    pl.col("action"),
                ],
                separator="_",
            ).alias("label_key"),
        )
        submission_pl = submission_pl.with_columns(
            pl.concat_str(
                [
                    pl.col("video_id").cast(pl.Utf8),
                    pl.col("agent_id").cast(pl.Utf8),
                    pl.col("target_id").cast(pl.Utf8),
                    pl.col("action"),
                ],
                separator="_",
            ).alias("prediction_key"),
        )

        # Group by action and compute metrics
        action_stats = defaultdict(lambda: {"single": {"count": 0, "f1": 0.0}, "pair": {"count": 0, "f1": 0.0}})

        for lab in solution_pl["lab_id"].unique():
            lab_solution = solution_pl.filter(pl.col("lab_id") == lab).clone()
            lab_videos = set(lab_solution["video_id"].unique())
            lab_submission = submission_pl.filter(pl.col("video_id").is_in(lab_videos)).clone()

            # Compute per-action F1 using same logic as single_lab_f1
            label_frames = defaultdict(set)
            prediction_frames = defaultdict(set)

            for row in lab_solution.to_dicts():
                label_frames[row["label_key"]].update(range(row["start_frame"], row["stop_frame"]))

            for row in lab_submission.to_dicts():
                key = row["prediction_key"]
                prediction_frames[key].update(range(row["start_frame"], row["stop_frame"]))

            for key in set(list(label_frames.keys()) + list(prediction_frames.keys())):
                action = key.split("_")[-1]
                mode = "single" if "self" in key else "pair"

                pred_frames = prediction_frames.get(key, set())
                label_frames_set = label_frames.get(key, set())

                tp = len(pred_frames & label_frames_set)
                fn = len(label_frames_set - pred_frames)
                fp = len(pred_frames - label_frames_set)

                if tp + fn + fp > 0:
                    f1 = (1 + 1**2) * tp / ((1 + 1**2) * tp + 1**2 * fn + fp)
                    action_stats[action][mode]["count"] += 1
                    action_stats[action][mode]["f1"] += f1

        # Print per-action summary
        print("\nPer-Action Performance Summary:")
        print(f"{'-' * 60}")
        print(f"{'Action':<20} {'Mode':<10} {'Count':<10} {'Avg F1':<10}")
        print(f"{'-' * 60}")

        for action in sorted(action_stats.keys()):
            for mode in ["single", "pair"]:
                stats = action_stats[action][mode]
                if stats["count"] > 0:
                    avg_f1 = stats["f1"] / stats["count"]
                    print(f"{action:<20} {mode:<10} {stats['count']:<10} {avg_f1:<10.4f}")

        # Summary by mode
        single_actions = [a for a in action_stats.keys() if action_stats[a]["single"]["count"] > 0]
        pair_actions = [a for a in action_stats.keys() if action_stats[a]["pair"]["count"] > 0]

        if single_actions:
            single_avg_f1 = np.mean(
                [
                    action_stats[a]["single"]["f1"] / action_stats[a]["single"]["count"]
                    for a in single_actions
                    if action_stats[a]["single"]["count"] > 0
                ]
            )
            print(f"\nSingle behaviors: {len(single_actions)} actions, Avg F1: {single_avg_f1:.4f}")

        if pair_actions:
            pair_avg_f1 = np.mean(
                [
                    action_stats[a]["pair"]["f1"] / action_stats[a]["pair"]["count"]
                    for a in pair_actions
                    if action_stats[a]["pair"]["count"] > 0
                ]
            )
            print(f"Pair behaviors: {len(pair_actions)} actions, Avg F1: {pair_avg_f1:.4f}")

        print(f"{'=' * 60}\n")

    except Exception as e:
        if verbose:
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            print(f"\nWarning: Could not compute validation metrics: {error_msg}")
            if verbose:
                print(f"Traceback: {traceback.format_exc()[:300]}")

compute_validation_metrics(submission=pd.read_csv(WORKING_DIR / "oof_predictions.csv"))

