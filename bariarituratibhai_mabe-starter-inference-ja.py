!pip install -q --no-index --find-links=/kaggle/input/mabe-package xgboost==3.1.1


!cp /kaggle/input/mabe-starter-train-ja/self_features.py .
!cp /kaggle/input/mabe-starter-train-ja/pair_features.py .
!cp /kaggle/input/mabe-starter-train-ja/robustify.py .
!cp -r /kaggle/input/mabe-starter-train-ja/results .


import gc
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
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
test_dataframe = pl.read_csv(INPUT_DIR / "test.csv")


# preprocess behavior labels
test_behavior_dataframe = (
    test_dataframe.filter(pl.col("behaviors_labeled").is_not_null())
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

test_self_behavior_dataframe = test_behavior_dataframe.filter(pl.col("behavior").is_in(SELF_BEHAVIORS))
test_pair_behavior_dataframe = test_behavior_dataframe.filter(pl.col("behavior").is_in(PAIR_BEHAVIORS))


%run -i self_features.py
%run -i pair_features.py
%run -i robustify.py


(WORKING_DIR / "self_features").mkdir(exist_ok=True, parents=True)
(WORKING_DIR / "pair_features").mkdir(exist_ok=True, parents=True)

rows = test_dataframe.rows(named=True)

for row in tqdm(rows, total=len(rows)):
    lab_id = row["lab_id"]
    video_id = row["video_id"]

    tracking_path = TEST_TRACKING_DIR / f"{lab_id}/{video_id}.parquet"
    tracking = pl.read_parquet(tracking_path)

    self_features = make_self_features(metadata=row, tracking=tracking)
    pair_features = make_pair_features(metadata=row, tracking=tracking)

    self_features.write_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
    pair_features.write_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")

    del self_features, pair_features
    gc.collect()


#（lab_id, video_id, agent, target 
group_submissions = []

# lab_id, video_id, agent, target 
# maintain_order=True 
groups = list(test_behavior_dataframe.group_by("lab_id", "video_id", "agent", "target", maintain_order=True))


for (lab_id, video_id, agent, target), group in tqdm(groups, total=len(list(groups))):
    # agentID 
    # 例: "mouse1" → 1
    agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
    
    # target（ID 
    # "self" -1、
    # : "mouse2" → 2, "self" → -1
    target_mouse_id = -1 if target == "self" else int(re.search(r"mouse(\d+)", target).group(1))

    # ===== 特徴量データの読み込み =====
    if target == "self":
        # 自己行動（rear など）の場合: self_features ディレクトリから読み込み
        
        # インデックス列（video_id, agent_mouse_id, video_frame など）を読み込み
        index = (
            pl.scan_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id))  # 対象マウスでフィルタ
            .select(INDEX_COLS)  # インデックス列のみ選択
            .collect()  # 遅延評価を実行してデータを取得
        )
        
        # 特徴量列（速度、距離、角度など）を読み込み
        feature = (
            pl.scan_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id))
            .select(pl.exclude(INDEX_COLS))  # インデックス列以外を選択
            .collect()
        )
    else:
        # ペア行動（attack, chase など）の場合: pair_features ディレクトリから読み込み
        
        # インデックス列を読み込み（agent と target の両方でフィルタ）
        index = (
            pl.scan_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id) & (pl.col("target_mouse_id") == target_mouse_id))
            .select(INDEX_COLS)
            .collect()
        )
        
        # 特徴量列を読み込み
        feature = (
            pl.scan_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id) & (pl.col("target_mouse_id") == target_mouse_id))
            .select(pl.exclude(INDEX_COLS))
            .collect()
        )

    # 予測結果を格納する DataFrame を作成（インデックス列のコピー）
    prediction_dataframe = index.clone()

    # ===== 各行動（behavior）に対して予測を実行 =====
    for row in group.rows(named=True):
        behavior = row["behavior"]  # 現在の行動名（例: "attack", "rear"）

        # 各 fold（交差検証の分割）の予測結果を格納するリスト
        predictions = []  # 予測確率
        prediction_labels = []  # 予測ラベル（閾値で 0/1 に変換したもの）

        # 保存された fold ディレクトリを取得
        # 例: results/AdaptableSnail/attack/fold_0, fold_1, fold_2
        fold_dirs = list((WORKING_DIR / "results" / lab_id / behavior).glob("fold_*"))
        if not fold_dirs:
            # 訓練されたモデルが見つからない場合はスキップ
            continue

        # 各 fold のモデルで予測を実行
        for fold_dir in fold_dirs:
            # 保存された最適閾値を読み込み
            with open(fold_dir / "threshold.txt", "r") as f:
                threshold = float(f.read().strip())
            
            # XGBoost モデルを読み込み
            model = xgb.Booster(model_file=fold_dir / "model.json")
            
            # 特徴量を XGBoost の入力形式（DMatrix）に変換
            dtest = xgb.DMatrix(feature, feature_names=feature.columns)
            
            # モデルで予測を実行（確率値を取得）
            fold_predictions = model.predict(dtest)
            
            # 予測確率を保存
            predictions.append(fold_predictions)
            
            # 閾値を適用してラベル化（1: 行動あり, 0: 行動なし）
            prediction_labels.append((fold_predictions >= threshold).astype(np.int8))

        # 予測結果を DataFrame に追加
        # 各 fold の「予測確率 × 予測ラベル」を列として追加
        # （ラベルが 0 の場合は確率も 0 になり、1 の場合は確率がそのまま残る）
        prediction_dataframe = prediction_dataframe.with_columns(
            *[
                pl.Series(name=f"{behavior}_{fold}", values=predictions[fold] * prediction_labels[fold], dtype=pl.Float32)
                for fold in range(len(fold_dirs))
            ]
        )

    # ===== 最も確率が高い行動を選択 =====
    
    # インデックス列以外の列名を取得（各行動の予測列）
    cols = prediction_dataframe.select(pl.exclude(INDEX_COLS)).columns
    if not cols:
        # 予測列が 1 つもない場合は警告を表示してスキップ
        tqdm.write(f"Warning: No predictions found for {lab_id}, {video_id}, {agent}, {target}")
        continue

    # 各フレームで最も確率が高い行動を選択
    prediction_labels_dataframe = prediction_dataframe.with_columns(
        pl.struct(pl.col(cols))  # 全予測列を構造体にまとめる
        .map_elements(
            # 各行（フレーム）に対して以下の処理を実行:
            # - すべての予測値が 0 なら "none"（行動なし）
            # - それ以外は最大値を持つ行動名を返す
            lambda row: "none" if sum(row.values()) == 0 else (cols[np.argmax(list(row.values()))]).split("_")[0],
            return_dtype=pl.String,
        )
        .alias("prediction")  # 新しい列名を "prediction" とする
    ).select(INDEX_COLS + ["prediction"])  # インデックス列と予測列のみ選択

    # ===== 連続する同じ行動をイベントにまとめる =====
    
    group_submission = (
        prediction_labels_dataframe
        .filter((pl.col("prediction") != pl.col("prediction").shift(1)))  # 行動が変化したフレームのみ残す
        .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))  # 次の変化点を終了フレームとする
        .filter(pl.col("prediction") != "none")  # "none"（行動なし）を除外
        .select(
            # 提出形式に合わせて列を選択・変換
            pl.col("video_id"),
            ("mouse" + pl.col("agent_mouse_id").cast(str)).alias("agent_id"),  # 例: 1 → "mouse1"
            pl.when(pl.col("target_mouse_id") == -1)  # target_mouse_id が -1 なら
            .then(pl.lit("self"))  # "self" に変換
            .otherwise("mouse" + pl.col("target_mouse_id").cast(str))  # それ以外は "mouseN"
            .alias("target_id"),
            pl.col("prediction").alias("action"),  # 行動名
            pl.col("video_frame").alias("start_frame"),  # 開始フレーム
            pl.col("stop_frame"),  # 終了フレーム
        )
    )

    # このグループの提出データをリストに追加
    group_submissions.append(group_submission)

# ===== 全グループの予測結果を結合 =====

# 全グループの提出データを縦方向に結合
submission = pl.concat(group_submissions, how="vertical").sort(
    "video_id",
    "agent_id",
    "target_id",
    "action",
    "start_frame",
    "stop_frame",
)

# 提出データの堅牢化処理（重複削除、フレームの修正など）
submission = robustify(submission, test_dataframe, train_test="test")

# 行番号（row_id）を追加して CSV ファイルとして保存
submission.with_row_index("row_id").write_csv(WORKING_DIR / "submission.csv")


!head submission.csv

