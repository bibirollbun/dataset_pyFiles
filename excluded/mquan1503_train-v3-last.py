import pandas as pd
import polars as pl
import numpy as np
import json
from tqdm import tqdm
import itertools
import gc
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.metrics import f1_score
import pickle
import torch
import joblib
import gc
import polars as pl
from collections import defaultdict
from pathlib import Path
import re
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import os
from joblib import Parallel, delayed


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

SELF_ACTIONS = [
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

PAIR_ACTIONS = [
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
INDEX_COLS = [
    "video_id",
    "agent_mouse_id",
    "target_mouse_id",
    "video_frame",
]

MODEL_CONFIG = {
    'x': {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.0,
        'min_child_weight': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'logloss',
        'verbosity': 0,
        'seed': 42,
        'n_jobs': -1
    },

    'c': {
        'iterations': 300,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'verbose': False,
        'task_type': 'CPU',
        'random_seed': 42,
        'thread_count': -1
    },

    'l': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 40,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 1,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'device': 'cpu',
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
}

# TRAINING | SUBMIT
STATUS = 'SUBMIT'


if STATUS == 'TRAINING':
    train_dataframe = pl.read_csv("/kaggle/input/MABe-mouse-behavior-detection/train.csv")
    
    train_dataframe = (
        train_dataframe
        .with_row_index()
        .filter(pl.col("index") != 8640)
        .filter(pl.col("behaviors_labeled").is_not_null())
        .drop("index")
    )
    
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
    
    train_self_behavior_dataframe = train_behavior_dataframe.filter(pl.col("behavior").is_in(SELF_ACTIONS))
    train_pair_behavior_dataframe = train_behavior_dataframe.filter(pl.col("behavior").is_in(PAIR_ACTIONS))


def make_self_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(body_part_1, body_part_2):
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"agent_x_{body_part_1}") - pl.col(f"agent_x_{body_part_2}")).pow(2)
            + (pl.col(f"agent_y_{body_part_1}") - pl.col(f"agent_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(body_part, period_ms):
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            ((pl.col(f"agent_x_{body_part}").diff()).pow(2) + (pl.col(f"agent_y_{body_part}").diff()).pow(2)).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def elongation():
        d1 = body_parts_distance("nose", "tail_base")
        d2 = body_parts_distance("ear_left", "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle():
        v1x = pl.col("agent_x_nose") - pl.col("agent_x_body_center")
        v1y = pl.col("agent_y_nose") - pl.col("agent_y_body_center")
        v2x = pl.col("agent_x_tail_base") - pl.col("agent_x_body_center")
        v2y = pl.col("agent_y_tail_base") - pl.col("agent_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)   

    def freeze_score(period_ms=1000):
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        speed = body_part_speed("body_center", 200)
        low_speed = (speed < 0.5).cast(pl.Float64)        
        freeze_ratio = low_speed.rolling_mean(window_size=window_frames, center=True, min_samples=1)        
        x_var = pl.col("agent_x_body_center").rolling_var(window_size=window_frames, center=True, min_samples=1)
        y_var = pl.col("agent_y_body_center").rolling_var(window_size=window_frames, center=True, min_samples=1)
        position_stability = 1.0 / (x_var + y_var + 1e-06)
        return freeze_ratio * position_stability  # ← SỬA: BỎ .alias()
    
    def movement_curvature(period_ms=500):
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        vel_x = pl.col("agent_x_body_center").diff().fill_null(0)
        vel_y = pl.col("agent_y_body_center").diff().fill_null(0)
        acc_x = vel_x.diff().fill_null(0)
        acc_y = vel_y.diff().fill_null(0)
        
        cross_product = vel_x * acc_y - vel_y * acc_x
        vel_magnitude = (vel_x.pow(2) + vel_y.pow(2)).sqrt() + 1e-06
        
        curvature = cross_product.abs() / vel_magnitude.pow(3)
        return curvature.rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def body_part_acceleration(body_part, period_ms):
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        speed = body_part_speed(body_part, period_ms)
        return speed.diff().rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def body_part_jerk(body_part, period_ms):
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        accel = body_part_acceleration(body_part, period_ms)
        return accel.diff().rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def head_lowering(period_ms=500):
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))        
        head_height_diff = (pl.col("agent_y_body_center") - pl.col("agent_y_nose")) / metadata["pix_per_cm_approx"]        
        head_low_ratio = (head_height_diff > 2.0).cast(pl.Float64)
        return head_low_ratio.rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def grooming_features():
        features = []        
        for period_ms in [200, 500]:
            window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
            nose_body_dist = body_parts_distance("nose", "body_center")
            dist_diff = nose_body_dist.diff().abs()
            oscillation_freq = dist_diff.rolling_mean(window_size=window_frames, center=True, min_samples=1)
            features.append(oscillation_freq.alias(f"agent__groom_oscillation_{period_ms}ms"))
        
        area_window = max(1, int(round(2000 * metadata["frames_per_second"] / 1000.0)))
        x_range = pl.col("agent_x_body_center").rolling_max(window_size=area_window) - \
                  pl.col("agent_x_body_center").rolling_min(window_size=area_window)
        y_range = pl.col("agent_y_body_center").rolling_max(window_size=area_window) - \
                  pl.col("agent_y_body_center").rolling_min(window_size=area_window)
        activity_area = (x_range * y_range) / (metadata["pix_per_cm_approx"] ** 2)
        features.append(activity_area.alias("agent__grooming_activity_area"))
        
        return features
       
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
            freeze_score(1000).alias("agent__freeze_score"),
            movement_curvature(500).alias("agent__movement_curvature"),
            head_lowering(500).alias("agent__head_lowering"),
            body_part_acceleration("body_center", 500).alias("agent__body_center__acceleration_500ms"),
            body_part_acceleration("nose", 500).alias("agent__nose__acceleration_500ms"),
            body_part_jerk("body_center", 500).alias("agent__body_center__jerk_500ms"),
            body_part_jerk("nose", 500).alias("agent__nose__jerk_500ms"),
            *grooming_features()
        )

        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")


def make_pair_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    
    def body_parts_distance(agent_or_target_1, body_part_1, agent_or_target_2, body_part_2):
        assert agent_or_target_1 == "agent" or agent_or_target_1 == "target"
        assert agent_or_target_2 == "agent" or agent_or_target_2 == "target"
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"{agent_or_target_1}_x_{body_part_1}") - pl.col(f"{agent_or_target_2}_x_{body_part_2}")).pow(2)
            + (pl.col(f"{agent_or_target_1}_y_{body_part_1}") - pl.col(f"{agent_or_target_2}_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(agent_or_target, body_part, period_ms):
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
        assert agent_or_target == "agent" or agent_or_target == "target"
        d1 = body_parts_distance(agent_or_target, "nose", agent_or_target, "tail_base")
        d2 = body_parts_distance(agent_or_target, "ear_left", agent_or_target, "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle(agent_or_target):
        assert agent_or_target == "agent" or agent_or_target == "target"
        v1x = pl.col(f"{agent_or_target}_x_nose") - pl.col(f"{agent_or_target}_x_body_center")
        v1y = pl.col(f"{agent_or_target}_y_nose") - pl.col(f"{agent_or_target}_y_body_center")
        v2x = pl.col(f"{agent_or_target}_x_tail_base") - pl.col(f"{agent_or_target}_x_body_center")
        v2y = pl.col(f"{agent_or_target}_y_tail_base") - pl.col(f"{agent_or_target}_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)

    def body_center_distance_rolling_agg(agg, period_ms):
        assert agg in ["mean", "std", "var", "min", "max"]
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

    def approach_features():
        current_dist = body_parts_distance("agent", "body_center", "target", "body_center")
        features = []
        for lag_ms in [100, 250, 500]:
            lag_frames = max(1, int(round(lag_ms * metadata["frames_per_second"] / 1000.0)))
            
            past_dist = current_dist.shift(lag_frames).fill_null(strategy="forward")
            approach_rate = (past_dist - current_dist) / (lag_ms / 1000.0)  # cm/s
            
            for period_ms in [500, 1000]:
                window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
                mean_rate = approach_rate.rolling_mean(window_size=window_frames, center=True, min_samples=1)
                features.append(mean_rate.alias(f"approach_rate_{lag_ms}lag_{period_ms}avg"))
        
        return features

    def relative_orientation_features():
        dx = pl.col("target_x_body_center") - pl.col("agent_x_body_center")
        dy = pl.col("target_y_body_center") - pl.col("agent_y_body_center")        
        agent_dir_x = pl.col("agent_x_nose") - pl.col("agent_x_tail_base")
        agent_dir_y = pl.col("agent_y_nose") - pl.col("agent_y_tail_base")        
        dot = dx * agent_dir_x + dy * agent_dir_y
        mag_agent = (agent_dir_x.pow(2) + agent_dir_y.pow(2)).sqrt() + 1e-06
        mag_to_target = (dx.pow(2) + dy.pow(2)).sqrt() + 1e-06
        
        facing_score = dot / (mag_agent * mag_to_target)
        
        features = []
        
        facing_each_other = (facing_score > 0.7).cast(pl.Float64)
        side_by_side = ((facing_score >= -0.3) & (facing_score <= 0.3)).cast(pl.Float64)
        facing_away = (facing_score < -0.7).cast(pl.Float64)
        
        features.extend([
            facing_each_other.alias("facing_each_other"),
            side_by_side.alias("side_by_side"),
            facing_away.alias("facing_away"),
            facing_score.alias("relative_orientation_score"),
        ])
        
        return features

    def chase_features():
        dist = body_parts_distance("agent", "body_center", "target", "body_center")        
        agent_vx = pl.col("agent_x_body_center").diff()
        agent_vy = pl.col("agent_y_body_center").diff()
        target_vx = pl.col("target_x_body_center").diff()
        target_vy = pl.col("target_y_body_center").diff()
        agent_speed = (agent_vx.pow(2) + agent_vy.pow(2)).sqrt() * metadata["frames_per_second"]
        target_speed = (target_vx.pow(2) + target_vy.pow(2)).sqrt() * metadata["frames_per_second"]        
        dx = pl.col("target_x_body_center") - pl.col("agent_x_body_center")
        dy = pl.col("target_y_body_center") - pl.col("agent_y_body_center")        
        agent_proj = (agent_vx * dx + agent_vy * dy) / (dist + 1e-06)
        target_proj = (target_vx * (-dx) + target_vy * (-dy)) / (dist + 1e-06)
        features = []
        chase_score = (agent_proj > 0).cast(pl.Float64) * (target_proj > 0).cast(pl.Float64)
        speed_diff = agent_speed - target_speed
        for period_ms in [500, 1000, 2000]:
            window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
            features.extend([
                chase_score.rolling_mean(window_size=window_frames, center=True, min_samples=1)
                .alias(f"chase_score_{period_ms}ms"),
                speed_diff.rolling_mean(window_size=window_frames, center=True, min_samples=1)
                .alias(f"speed_difference_{period_ms}ms"),
            ])
        return features
    
    def mounting_features():
        height_diff = (pl.col("target_y_body_center") - pl.col("agent_y_body_center")) / metadata["pix_per_cm_approx"]        
        vertical_dist = height_diff.abs()        
        agent_below_target = (height_diff > 5.0).cast(pl.Float64)        
        dx = pl.col("target_x_body_center") - pl.col("agent_x_body_center")
        dy = pl.col("target_y_body_center") - pl.col("agent_y_body_center")        
        target_dir_x = pl.col("target_x_nose") - pl.col("target_x_tail_base")
        target_dir_y = pl.col("target_y_nose") - pl.col("target_y_tail_base")        
        behind_score = -(dx * target_dir_x + dy * target_dir_y) / ((dx.pow(2) + dy.pow(2)).sqrt() * (target_dir_x.pow(2) + target_dir_y.pow(2)).sqrt() + 1e-06)
        features = [
            height_diff.alias("mounting_height_difference"),
            vertical_dist.alias("mounting_vertical_distance"),
            agent_below_target.alias("agent_below_target"),
            behind_score.alias("mounting_behind_score"),
        ]
        
        return features
    
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

            *approach_features(),
            *relative_orientation_features(),
            *chase_features(),
            *mounting_features()
        )

        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")


# def process_video(row):
#     lab_id = row["lab_id"]
#     video_id = row["video_id"]

#     tracking_path = f"/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet"
#     tracking = pl.read_parquet(tracking_path)

#     self_features = make_self_features(metadata=row, tracking=tracking)
#     pair_features = make_pair_features(metadata=row, tracking=tracking)

#     self_features.write_parquet(f"self_features/{video_id}.parquet")
#     pair_features.write_parquet(f"pair_features/{video_id}.parquet")

#     return video_id

# if STATUS == "TRAINING":
#     os.makedirs("self_features", exist_ok=True)
#     os.makedirs("pair_features", exist_ok=True)
#     rows = list(train_dataframe.filter(pl.col("behaviors_labeled").is_not_null()).rows(named=True))
#     results = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(process_video)(row) for row in rows)
#     del rows, results
#     gc.collect()


def score_for_threshold(th, oof, y):
    preds = (oof >= th)
    return f1_score(y, preds, zero_division=0)

def tune_threshold(oof, y, n_jobs=-1):
    thresholds = np.arange(0.0, 1.0001, 0.005)
    f1_scores = Parallel(n_jobs=n_jobs)(
        delayed(score_for_threshold)(th, oof, y) for th in thresholds
    )
    f1_scores = np.array(f1_scores)

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    return best_threshold, best_f1


def tune_model(X, y, type_model):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        pos_weight = neg_count / max(pos_count, 1)
        pos_weight = min(pos_weight, 50)
        
        if type_model == 'x':
            xgb_config = MODEL_CONFIG['x'].copy()
            xgb_config['scale_pos_weight'] = pos_weight
            model = xgb.XGBClassifier(**xgb_config)
            
        elif type_model == 'c':
            cat_config = MODEL_CONFIG['c'].copy()
            cat_config['scale_pos_weight'] = pos_weight
            model = CatBoostClassifier(**cat_config)
            
        else:
            lgb_config = MODEL_CONFIG['l'].copy()
            lgb_config['scale_pos_weight'] = pos_weight
            model = lgb.LGBMClassifier(**lgb_config)

        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_probs

        del model, X_train, X_val, y_train, y_val
        gc.collect()
        torch.cuda.empty_cache()

    best_threshold, best_f1 = tune_threshold(oof_predictions, y)

    total_pos_count = y.sum()
    total_neg_count = len(y) - total_pos_count
    total_pos_weight = total_neg_count / max(total_pos_count, 1)
    
    if type_model == 'x':
        xgb_config = MODEL_CONFIG['x'].copy()
        xgb_config['scale_pos_weight'] = total_pos_weight
        final_model = xgb.XGBClassifier(**xgb_config)
        
    elif type_model == 'c':
        cat_config = MODEL_CONFIG['c'].copy()
        cat_config['scale_pos_weight'] = total_pos_weight
        final_model = CatBoostClassifier(**cat_config)
        
    else:
        lgb_config = MODEL_CONFIG['l'].copy()
        lgb_config['scale_pos_weight'] = total_pos_weight
        final_model = lgb.LGBMClassifier(**lgb_config)

    final_model.fit(X, y)

    return final_model, best_threshold, best_f1


MAX_NEGATIVES = 4_000_000

def train_validate_1(lab_id: str, behavior: str,
                     indices: pl.DataFrame, features: pl.DataFrame, labels: pl.Series,
                     max_negatives: int = MAX_NEGATIVES):
    
    result_dir = f"results/{lab_id}/{behavior}"
    os.makedirs(result_dir, exist_ok=True)
    
    if labels.sum() == 0:
        return None, 0.0
    
    df = features.with_columns(labels.alias("label"))
    
    pos_df = df.filter(pl.col("label") == 1)
    neg_df = df.filter(pl.col("label") == 0)
    
    if len(neg_df) > max_negatives:
        neg_df = neg_df.sample(n=max_negatives, shuffle=True)

    df_sampled = pl.concat([pos_df, neg_df])
    
    X = df_sampled.drop("label").to_numpy()
    y = df_sampled["label"].to_numpy()
    
    final_xgb, th_xgb, f1_xgb = tune_model(X, y, "x")
    
    save_dir = os.path.join(result_dir, "models_1")
    os.makedirs(save_dir, exist_ok=True)
    
    final_xgb.save_model(os.path.join(save_dir, "model_x.json"))
    with open(os.path.join(save_dir, "threshold_x.txt"), "w") as f:
        f.write(str(th_xgb))
        
    print(f'XGB_Level1: th-{th_xgb:.3f}, f1-{f1_xgb:.3f}')

    return final_xgb, th_xgb


def train_validate_2(lab_id: str, behavior: str,
                     indices: pl.DataFrame, features: pl.DataFrame, labels: pl.Series,
                     model1_dict: dict,
                     max_negatives: int = MAX_NEGATIVES):
    
    result_dir = f"results/{lab_id}/{behavior}"
    os.makedirs(result_dir, exist_ok=True)
    
    if labels.sum() == 0:
        return None, 0.0

    df = features.with_columns(labels.alias("label"))

    pos_df = df.filter(pl.col("label") == 1)
    neg_df = df.filter(pl.col("label") == 0)

    if len(neg_df) > max_negatives:
        neg_df = neg_df.sample(n=max_negatives, shuffle=True)

    df_sampled = pl.concat([pos_df, neg_df])
    df_sampled = df_sampled.sample(n=df_sampled.height, shuffle=True)
    
    X_original = df_sampled.drop("label").to_numpy()
    y = df_sampled["label"].to_numpy()
    
    predictions_list = []

    for other_lab_id, (model1, thr) in model1_dict.get(behavior, {}).items():  
        if other_lab_id == lab_id:
            continue
        try:
            prob = model1.predict_proba(X_original)[:, 1]
            pred_binary = (prob >= thr).astype(float)
            predictions_list.append(prob.reshape(-1, 1))
            predictions_list.append(pred_binary.reshape(-1, 1))
        except Exception:
            continue

    if predictions_list:
        predictions_array = np.hstack(predictions_list)
        X_enhanced = np.hstack([X_original, predictions_array])
    else:
        X_enhanced = X_original
    
    final_xgb, th_xgb, f1_xgb = tune_model(X_enhanced, y, "x")
    
    save_dir = os.path.join(result_dir, "models_2")
    os.makedirs(save_dir, exist_ok=True)
    
    final_xgb.save_model(os.path.join(save_dir, "model_x.json"))
    with open(os.path.join(save_dir, "threshold_x.txt"), "w") as f:
        f.write(str(th_xgb))
        
    print(f'XGB_Level2: th-{th_xgb:.3f}, f1-{f1_xgb:.3f}')
    return final_xgb, th_xgb



# if STATUS == 'TRAINING':
#     print("=" * 60)
#     print("LEVEL 1")
#     print("=" * 60)
    
#     groups = train_self_behavior_dataframe.group_by(["lab_id", "behavior"], maintain_order=True)
#     groups_list = list(groups)
#     total = len(groups_list)
    
#     model1_dict = {}
    
#     for (lab_id, behavior), group in tqdm(groups_list, total=total):
#         print(f"\n=== LEVEL1 - LAB: {lab_id} | BEHAVIOR: {behavior} ===")
        
#         index_list = []
#         feature_list = []
#         label_list = []
        
#         for row in group.iter_rows(named=True):
#             video_id = row["video_id"]
#             agent = row["agent"]
#             agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
            
#             feature_path = f"self_features/{video_id}.parquet"
#             data = pl.scan_parquet(feature_path).filter(
#                 pl.col("agent_mouse_id") == agent_mouse_id
#             )
            
#             index = data.select(INDEX_COLS).collect(engine="streaming")
#             feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")
            
#             ann_path = f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet"
#             if os.path.exists(ann_path):
#                 annotation = (
#                     pl.scan_parquet(ann_path)
#                     .filter(
#                         (pl.col("action") == behavior) &
#                         (pl.col("agent_id") == agent_mouse_id)
#                     )
#                     .collect()
#                 )
#             else:
#                 annotation = pl.DataFrame()
            
#             label_frames = set()
#             for ann in annotation.iter_rows(named=True):
#                 label_frames.update(range(ann["start_frame"], ann["stop_frame"]))
            
#             label = index.select(
#                 pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label")
#             )
            
#             if label["label"].sum() == 0:
#                 continue
            
#             index_list.append(index)
#             feature_list.append(feature)
#             label_list.append(label["label"])
        
#         if not index_list:
#             continue
        
#         indices = pl.concat(index_list)
#         features = pl.concat(feature_list)
#         labels = pl.concat(label_list)

#         if labels.min() == labels.max():
#             continue
        
#         print(f"Samples: {len(indices)}")
#         print(f"Positive: {labels.sum()}")
#         print(f"Features: {len(features.columns)}")
        
#         final_xgb, thr_xgb = train_validate_1(lab_id, behavior, indices, features, labels)
        
#         if final_xgb is not None:
#             if behavior not in model1_dict:
#                 model1_dict[behavior] = {}
#             model1_dict[behavior][lab_id] = (final_xgb, thr_xgb)
        
#         del indices, features, labels
#         gc.collect()
    
#     print("\n" + "=" * 60)
#     print("LEVEL 2")
#     print("=" * 60)
    
#     for (lab_id, behavior), group in tqdm(groups_list, total=total):
#         print(f"\n=== LEVEL2 - LAB: {lab_id} | BEHAVIOR: {behavior} ===")
        
#         if behavior not in model1_dict:
#             continue
        
#         index_list = []
#         feature_list = []
#         label_list = []
        
#         for row in group.iter_rows(named=True):
#             video_id = row["video_id"]
#             agent = row["agent"]
#             agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
            
#             feature_path = f"self_features/{video_id}.parquet"
#             data = pl.scan_parquet(feature_path).filter(
#                 pl.col("agent_mouse_id") == agent_mouse_id
#             )
            
#             index = data.select(INDEX_COLS).collect(engine="streaming")
#             feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")
            
#             ann_path = f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet"
#             if os.path.exists(ann_path):
#                 annotation = (
#                     pl.scan_parquet(ann_path)
#                     .filter(
#                         (pl.col("action") == behavior) &
#                         (pl.col("agent_id") == agent_mouse_id)
#                     )
#                     .collect()
#                 )
#             else:
#                 annotation = pl.DataFrame()
            
#             label_frames = set()
#             for ann in annotation.iter_rows(named=True):
#                 label_frames.update(range(ann["start_frame"], ann["stop_frame"]))
            
#             label = index.select(
#                 pl.col("video_frame").is_in(label_frames).cast(pl.Int8).alias("label")
#             )
            
#             if label["label"].sum() == 0:
#                 continue
            
#             index_list.append(index)
#             feature_list.append(feature)
#             label_list.append(label["label"])
        
#         if not index_list:
#             continue
        
#         indices = pl.concat(index_list)
#         features = pl.concat(feature_list)
#         labels = pl.concat(label_list)

#         if labels.min() == labels.max():
#             continue
        
#         print(f"Samples: {len(indices)}")
#         print(f"Positive: {labels.sum()}")
#         print(f"Features: {len(features.columns)}")
        
#         train_validate_2(lab_id, behavior, indices, features, labels, model1_dict)
        
#         del indices, features, labels
#         gc.collect()
#     del model1_dict
#     gc.collect()


if STATUS == 'TRAINING':
    print("=" * 60)
    print("PAIR BEHAVIORS - LEVEL 1")
    print("=" * 60)
    
    groups = train_pair_behavior_dataframe.group_by(
        ["lab_id", "behavior"], 
        maintain_order=True
    )
    groups_list = list(groups)
    total = len(groups_list)
    
    model1_dict = {}
    
    for (lab_id, behavior), group in tqdm(groups_list, total=total):
        print(f"\n=== LEVEL1_PAIR - LAB: {lab_id} | BEHAVIOR: {behavior} ===")

        index_list = []
        feature_list = []
        label_list = []

        for row in group.iter_rows(named=True):
            video_id = row["video_id"]
            agent = row["agent"]
            target = row["target"]

            agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
            target_mouse_id = int(re.search(r"mouse(\d+)", target).group(1))

            feature_path = f"pair_features/{video_id}.parquet"
            if not os.path.exists(feature_path):
                continue

            data = (
                pl.scan_parquet(feature_path)
                .filter(
                    (pl.col("agent_mouse_id") == agent_mouse_id) &
                    (pl.col("target_mouse_id") == target_mouse_id)
                )
            )

            index = data.select(INDEX_COLS).collect(engine="streaming")
            feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")

            ann_path = (
                f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/"
                f"{lab_id}/{video_id}.parquet"
            )

            if os.path.exists(ann_path):
                annotation = (
                    pl.scan_parquet(ann_path)
                    .filter(
                        (pl.col("action") == behavior) &
                        (pl.col("agent_id") == agent_mouse_id) &
                        (pl.col("target_id") == target_mouse_id)
                    )
                    .collect()
                )
            else:
                annotation = pl.DataFrame()

            label_frames = set()
            for ann in annotation.iter_rows(named=True):
                label_frames.update(range(ann["start_frame"], ann["stop_frame"]))

            label = index.select(
                pl.col("video_frame")
                .is_in(label_frames)
                .cast(pl.Int8)
                .alias("label")
            )

            if label["label"].sum() == 0:
                continue

            index_list.append(index)
            feature_list.append(feature)
            label_list.append(label["label"])

        if not index_list:
            continue

        indices = pl.concat(index_list)
        features = pl.concat(feature_list)
        labels = pl.concat(label_list)
        
        if labels.min() == labels.max():
            continue

        print(f"Samples: {len(indices)}")
        print(f"Positive: {labels.sum()}")
        print(f"Features: {len(features.columns)}")

        final_xgb, thr_xgb = train_validate_1(lab_id, behavior, indices, features, labels)
        
        if final_xgb is not None:
            if behavior not in model1_dict:
                model1_dict[behavior] = {}
            model1_dict[behavior][lab_id] = (final_xgb, thr_xgb)
        
        del indices, features, labels
        gc.collect()
    
    print("\n" + "=" * 60)
    print("PAIR BEHAVIORS - LEVEL 2")
    print("=" * 60)
    
    for (lab_id, behavior), group in tqdm(groups_list, total=total):
        print(f"\n=== LEVEL2_PAIR - LAB: {lab_id} | BEHAVIOR: {behavior} ===")
        
        if behavior not in model1_dict:
            continue
        
        index_list = []
        feature_list = []
        label_list = []

        for row in group.iter_rows(named=True):
            video_id = row["video_id"]
            agent = row["agent"]
            target = row["target"]

            agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
            target_mouse_id = int(re.search(r"mouse(\d+)", target).group(1))

            feature_path = f"pair_features/{video_id}.parquet"
            if not os.path.exists(feature_path):
                continue

            data = (
                pl.scan_parquet(feature_path)
                .filter(
                    (pl.col("agent_mouse_id") == agent_mouse_id) &
                    (pl.col("target_mouse_id") == target_mouse_id)
                )
            )

            index = data.select(INDEX_COLS).collect(engine="streaming")
            feature = data.select(pl.exclude(INDEX_COLS)).collect(engine="streaming")

            ann_path = (
                f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/"
                f"{lab_id}/{video_id}.parquet"
            )

            if os.path.exists(ann_path):
                annotation = (
                    pl.scan_parquet(ann_path)
                    .filter(
                        (pl.col("action") == behavior) &
                        (pl.col("agent_id") == agent_mouse_id) &
                        (pl.col("target_id") == target_mouse_id)
                    )
                    .collect()
                )
            else:
                annotation = pl.DataFrame()

            label_frames = set()
            for ann in annotation.iter_rows(named=True):
                label_frames.update(range(ann["start_frame"], ann["stop_frame"]))

            label = index.select(
                pl.col("video_frame")
                .is_in(label_frames)
                .cast(pl.Int8)
                .alias("label")
            )

            if label["label"].sum() == 0:
                continue

            index_list.append(index)
            feature_list.append(feature)
            label_list.append(label["label"])

        if not index_list:
            continue

        indices = pl.concat(index_list)
        features = pl.concat(feature_list)
        labels = pl.concat(label_list)
        
        if labels.min() == labels.max():
            continue

        print(f"Samples: {len(indices)}")
        print(f"Positive: {labels.sum()}")
        print(f"Features: {len(features.columns)}")
        
        train_validate_2(lab_id, behavior, indices, features, labels, model1_dict)
        
        del indices, features, labels
        gc.collect()
    
    del model1_dict
    gc.collect()


def robustify(submission: pl.DataFrame, dataset: pl.DataFrame, train_test: str = "train"):
    traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{train_test}_tracking"

    old = submission
    submission = submission.filter(
        (pl.col("stop_frame") - pl.col("start_frame")) > 2
    )
    if len(submission) != len(old):
        print("ERROR: Dropped frames with start >= stop")

    old = submission
    cleaned_groups = []

    for _, group in submission.group_by("video_id", "agent_id", "target_id"):
        group = group.sort("start_frame")

        mask = np.ones(len(group), dtype=bool)
        last_stop = -1

        for i, row in enumerate(group.rows(named=True)):
            if row["start_frame"] < last_stop:
                mask[i] = False
            else:
                last_stop = row["stop_frame"]

        cleaned_groups.append(group.filter(pl.Series("mask", mask)))

    submission = pl.concat(cleaned_groups)

    merged_groups = []
    for _, group in submission.group_by(["video_id", "agent_id", "target_id", "action"]):
        
        g = group.sort("start_frame").to_pandas()
    
        if len(g) == 0:
            continue

        merged = []
        current = g.iloc[0]
        
        for i in range(1, len(g)):
            row = g.iloc[i]
            if row.start_frame < current.start_frame:
                continue
            if row.stop_frame <= current.stop_frame:
                continue
            gap = row.start_frame - current.stop_frame
            if 0 <= gap <= 2:
                current.stop_frame = row.stop_frame
            else:
                merged.append(current.copy())
                current = row.copy()
        merged.append(current.copy())
        merged_groups.append(pl.from_pandas(pd.DataFrame(merged)))
    submission = pl.concat(merged_groups)

    if len(submission) != len(old):
        print("ERROR: Dropped duplicate/overlap frames")

    s_list = []

    submission_videos = set(submission.get_column("video_id").to_list())

    for r in dataset.rows(named=True):
        lab_id = r["lab_id"]
        video_id = r["video_id"]
        labeled = r["behaviors_labeled"]

        if labeled is None:
            continue

        if video_id in submission_videos:
            continue

        if isinstance(labeled, str):
            continue

        print(f"Video {video_id} has no predictions.")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        behaviors = json.loads(labeled)
        behaviors = sorted({b.replace("'", "") for b in behaviors})
        behaviors = [x.split(",") for x in behaviors]
        behaviors = pd.DataFrame(behaviors, columns=["agent", "target", "action"])

        start = vid.video_frame.min()
        stop = vid.video_frame.max() + 1

        grouped = behaviors.groupby(["agent", "target"])

        for (agent, target), acts in grouped:
            batch_len = int(np.ceil((stop - start) / len(acts)))

            for i, act in enumerate(acts.itertuples(index=False)):
                st = start + i * batch_len
                sp = min(st + batch_len, stop)
                s_list.append((video_id, agent, target, act.action, st, sp))

    if s_list:
        print("ERROR: Filled empty videos")
        fill_df = pd.DataFrame(
            s_list,
            columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
        )
        submission = pl.concat([submission, pl.from_pandas(fill_df)])

    return submission


if STATUS == "SUBMIT":
    test_dataframe = pl.read_csv("/kaggle/input/MABe-mouse-behavior-detection/test.csv")
    test_behavior_dataframe = (
        test_dataframe
        .filter(pl.col("behaviors_labeled").is_not_null())
        .select(
            pl.col("lab_id"),
            pl.col("video_id"),
            pl.col("behaviors_labeled")
                .map_elements(eval, return_dtype=pl.List(pl.Utf8))
                .alias("behaviors_labeled_list"),
        )
        .explode("behaviors_labeled_list")
        .rename({"behaviors_labeled_list": "behaviors_labeled_element"})
        .select(
            pl.col("lab_id"),
            pl.col("video_id"),
            pl.col("behaviors_labeled_element")
                .str.split(",")
                .list[0]
                .str.replace_all("'", "")
                .alias("agent"),
            pl.col("behaviors_labeled_element")
                .str.split(",")
                .list[1]
                .str.replace_all("'", "")
                .alias("target"),
            pl.col("behaviors_labeled_element")
                .str.split(",")
                .list[2]
                .str.replace_all("'", "")
                .alias("behavior"),
        )
    )
    
    test_self_behavior_dataframe = test_behavior_dataframe.filter(
        pl.col("behavior").is_in(SELF_ACTIONS)
    )
    
    test_pair_behavior_dataframe = test_behavior_dataframe.filter(
        pl.col("behavior").is_in(PAIR_ACTIONS)
    )


if STATUS == "SUBMIT":

    self_features_dir = "self_features"
    pair_features_dir = "pair_features"
    
    os.makedirs(self_features_dir, exist_ok=True)
    os.makedirs(pair_features_dir, exist_ok=True)
    
    rows = test_dataframe.rows(named=True)
    
    for row in tqdm(rows, total=len(rows)):
        lab_id = row["lab_id"]
        video_id = row["video_id"]
        
        tracking_path = f"/kaggle/input/MABe-mouse-behavior-detection/test_tracking/{lab_id}/{video_id}.parquet"
        tracking = pl.read_parquet(tracking_path)
        
        self_features = make_self_features(metadata=row, tracking=tracking)
        pair_features = make_pair_features(metadata=row, tracking=tracking)
        
        self_features.write_parquet(f"{self_features_dir}/{video_id}.parquet")
        pair_features.write_parquet(f"{pair_features_dir}/{video_id}.parquet")


if STATUS == "SUBMIT":

    model1_map = {}
    model2_map = {}
    
    for lab_id in os.listdir('/kaggle/input/train-v3-not-enough/results (1)'):
        for act in os.listdir(f'/kaggle/input/train-v3-not-enough/results (1)/{lab_id}'):
            if 'models_1' in os.listdir(f'/kaggle/input/train-v3-not-enough/results (1)/{lab_id}/{act}'):
                model1 = xgb.Booster()
                model1.load_model(f'/kaggle/input/train-v3-not-enough/results (1)/{lab_id}/{act}/models_1/model_x.json')
                with open(f'/kaggle/input/train-v3-not-enough/results (1)/{lab_id}/{act}/models_1/threshold_x.txt', 'r', encoding='utf-8') as f:
                    threshold = float(f.read())
                if lab_id not in model1_map:
                    model1_map[lab_id] = {}
                model1_map[lab_id][act] = (model1, threshold)
            if 'models_2' in os.listdir(f'/kaggle/input/train-v3-not-enough/results (1)/{lab_id}/{act}'):
                model2 = xgb.Booster()
                model2.load_model(f'/kaggle/input/train-v3-not-enough/results (1)/{lab_id}/{act}/models_2/model_x.json')
                with open(f'/kaggle/input/train-v3-not-enough/results (1)/{lab_id}/{act}/models_2/threshold_x.txt', 'r', encoding='utf-8') as f:
                    threshold = float(f.read())
                if lab_id not in model2_map:
                    model2_map[lab_id] = {}
                model2_map[lab_id][act] = (model2, threshold)
    for lab_id in os.listdir('/kaggle/input/train-v3-not-enough/results (2)'):
        for act in os.listdir(f'/kaggle/input/train-v3-not-enough/results (2)/{lab_id}'):
            if 'models_1' in os.listdir(f'/kaggle/input/train-v3-not-enough/results (2)/{lab_id}/{act}'):
                model1 = xgb.Booster()
                model1.load_model(f'/kaggle/input/train-v3-not-enough/results (2)/{lab_id}/{act}/models_1/model_x.json')
                with open(f'/kaggle/input/train-v3-not-enough/results (2)/{lab_id}/{act}/models_1/threshold_x.txt', 'r', encoding='utf-8') as f:
                    threshold = float(f.read())
                if lab_id not in model1_map:
                    model1_map[lab_id] = {}
                model1_map[lab_id][act] = (model1, threshold)
            if 'models_2' in os.listdir(f'/kaggle/input/train-v3-not-enough/results (2)/{lab_id}/{act}'):
                model2 = xgb.Booster()
                model2.load_model(f'/kaggle/input/train-v3-not-enough/results (2)/{lab_id}/{act}/models_2/model_x.json')
                with open(f'/kaggle/input/train-v3-not-enough/results (2)/{lab_id}/{act}/models_2/threshold_x.txt', 'r', encoding='utf-8') as f:
                    threshold = float(f.read())
                if lab_id not in model2_map:
                    model2_map[lab_id] = {}
                model2_map[lab_id][act] = (model2, threshold)
    
    group_submissions = []
    groups = test_behavior_dataframe.group_by("lab_id", "video_id", "agent", "target", maintain_order=True)

    for (lab_id, video_id, agent, target), group in tqdm(groups, total=len(list(groups))):
        agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
        target_mouse_id = -1 if target == "self" else int(re.search(r"mouse(\d+)", target).group(1))
        
        if target == "self":
            index = (
                pl.scan_parquet(f"self_features/{video_id}.parquet")
                .filter(pl.col("agent_mouse_id") == agent_mouse_id)
                .select(INDEX_COLS)
                .collect()
            )
            feature = (
                pl.scan_parquet(f"self_features/{video_id}.parquet")
                .filter(pl.col("agent_mouse_id") == agent_mouse_id)
                .select(pl.exclude(INDEX_COLS))
                .collect()
            )
        else:
            index = (
                pl.scan_parquet(f"pair_features/{video_id}.parquet")
                .filter((pl.col("agent_mouse_id") == agent_mouse_id) & 
                        (pl.col("target_mouse_id") == target_mouse_id))
                .select(INDEX_COLS)
                .collect()
            )
            feature = (
                pl.scan_parquet(f"pair_features/{video_id}.parquet")
                .filter((pl.col("agent_mouse_id") == agent_mouse_id) & 
                        (pl.col("target_mouse_id") == target_mouse_id))
                .select(pl.exclude(INDEX_COLS))
                .collect()
            )
        
        prediction_dataframe = index.clone()
        
        for row in group.rows(named=True):
            behavior = row["behavior"]
            predictions = []
            prediction_labels = []
            
            # fold_dirs = list(("/absolute/path/to/results" / lab_id / behavior).glob("fold_*"))
            # if not fold_dirs:
            #     continue
            
            # for fold_dir in fold_dirs:
            #     with open(fold_dir / "threshold.txt", "r") as f:
            #         threshold = float(f.read().strip())
                
            #     model = xgb.Booster(model_file=fold_dir / "model.json")
                
            #     dtest = xgb.DMatrix(feature, feature_names=feature.columns)
            #     fold_predictions = model.predict(dtest)
                
            #     predictions.append(fold_predictions)
            #     prediction_labels.append((fold_predictions >= threshold).astype(np.int8))

            try:
                model, thr = model1_map[lab_id][behavior]
                dtest = xgb.DMatrix(feature, feature_names=feature.columns)
                model_predictions = model.predict(dtest)
                predictions.append(model_predictions)
                prediction_labels.append((model_predictions >= thr).astype(np.int8))
            except:
                continue
            
            prediction_dataframe = prediction_dataframe.with_columns(
                *[
                    pl.Series(name=f"{behavior}_{fold}", 
                             values=predictions[fold] * prediction_labels[fold], 
                             dtype=pl.Float32)
                    for fold in range(len(predictions))
                ]
            )
        
        cols = prediction_dataframe.select(pl.exclude(INDEX_COLS)).columns
        if not cols:
            tqdm.write(f"Warning: No predictions for {lab_id}, {video_id}, {agent}, {target}")
            continue
        
        prediction_labels_dataframe = prediction_dataframe.with_columns(
            pl.struct(pl.col(cols))
            .map_elements(
                lambda row: "none" if sum(row.values()) == 0 else cols[np.argmax(list(row.values()))].split("_")[0],
                return_dtype=pl.String,
            )
            .alias("prediction")
        ).select(INDEX_COLS + ["prediction"])
        
        group_submission = (
            prediction_labels_dataframe
            .filter(pl.col("prediction") != pl.col("prediction").shift(1))
            .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))
            .filter(pl.col("prediction") != "none")
            .select(
                pl.col("video_id"),
                ("mouse" + pl.col("agent_mouse_id").cast(str)).alias("agent_id"),
                pl.when(pl.col("target_mouse_id") == -1)
                .then(pl.lit("self"))
                .otherwise("mouse" + pl.col("target_mouse_id").cast(str))
                .alias("target_id"),
                pl.col("prediction").alias("action"),
                pl.col("video_frame").alias("start_frame"),
                pl.col("stop_frame"),
            )
        )
        
        group_submissions.append(group_submission)
    
    submission = pl.concat(group_submissions, how="vertical").sort(
        "video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"
    )
    
    submission = robustify(submission, test_dataframe, train_test="test")
    
    submission.with_row_index("row_id").write_csv("submission.csv")

