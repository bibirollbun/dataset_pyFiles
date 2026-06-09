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


# å�„ã‚°ãƒ«ãƒ¼ãƒ—ï¼ˆlab_id, video_id, agent, target ã�®çµ„ã�¿å�ˆã‚�ã�›ï¼‰ã�”ã�¨ã�®äºˆæ¸¬çµ�æ�œã‚’æ ¼ç´�ã�™ã‚‹ãƒªã‚¹ãƒˆ
group_submissions = []

# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã‚’ lab_id, video_id, agent, target ã�§ã‚°ãƒ«ãƒ¼ãƒ—åŒ–
# maintain_order=True ã�§å…ƒã�®é †åº�ã‚’ä¿�æŒ�
groups = list(test_behavior_dataframe.group_by("lab_id", "video_id", "agent", "target", maintain_order=True))

# å�„ã‚°ãƒ«ãƒ¼ãƒ—ã�«å¯¾ã�—ã�¦é †ç•ªã�«å‡¦ç�†ã‚’å®Ÿè¡Œï¼ˆé€²æ�—ãƒ�ãƒ¼ã‚’è¡¨ç¤ºï¼‰
for (lab_id, video_id, agent, target), group in tqdm(groups, total=len(list(groups))):
    # agentï¼ˆè¡Œå‹•ã‚’èµ·ã�“ã�™ãƒ�ã‚¦ã‚¹ï¼‰ã�®ID ã‚’æŠ½å‡º
    # ä¾‹: "mouse1" â†’ 1
    agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
    
    # targetï¼ˆè¡Œå‹•ã�®å¯¾è±¡ï¼‰ã�®ID ã‚’æŠ½å‡º
    # "self"ï¼ˆè‡ªå·±è¡Œå‹•ï¼‰ã�®å ´å�ˆã�¯ -1ã€�ã��ã‚Œä»¥å¤–ã�¯ãƒ�ã‚¦ã‚¹IDã‚’æŠ½å‡º
    # ä¾‹: "mouse2" â†’ 2, "self" â†’ -1
    target_mouse_id = -1 if target == "self" else int(re.search(r"mouse(\d+)", target).group(1))

    # ===== ç‰¹å¾´é‡�ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿ =====
    if target == "self":
        # è‡ªå·±è¡Œå‹•ï¼ˆrear ã�ªã�©ï¼‰ã�®å ´å�ˆ: self_features ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�‹ã‚‰èª­ã�¿è¾¼ã�¿
        
        # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ï¼ˆvideo_id, agent_mouse_id, video_frame ã�ªã�©ï¼‰ã‚’èª­ã�¿è¾¼ã�¿
        index = (
            pl.scan_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id))  # å¯¾è±¡ãƒ�ã‚¦ã‚¹ã�§ãƒ•ã‚£ãƒ«ã‚¿
            .select(INDEX_COLS)  # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ã�®ã�¿é�¸æŠ�
            .collect()  # é�…å»¶è©•ä¾¡ã‚’å®Ÿè¡Œã�—ã�¦ãƒ‡ãƒ¼ã‚¿ã‚’å�–å¾—
        )
        
        # ç‰¹å¾´é‡�åˆ—ï¼ˆé€Ÿåº¦ã€�è·�é›¢ã€�è§’åº¦ã�ªã�©ï¼‰ã‚’èª­ã�¿è¾¼ã�¿
        feature = (
            pl.scan_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id))
            .select(pl.exclude(INDEX_COLS))  # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ä»¥å¤–ã‚’é�¸æŠ�
            .collect()
        )
    else:
        # ãƒšã‚¢è¡Œå‹•ï¼ˆattack, chase ã�ªã�©ï¼‰ã�®å ´å�ˆ: pair_features ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�‹ã‚‰èª­ã�¿è¾¼ã�¿
        
        # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ã‚’èª­ã�¿è¾¼ã�¿ï¼ˆagent ã�¨ target ã�®ä¸¡æ–¹ã�§ãƒ•ã‚£ãƒ«ã‚¿ï¼‰
        index = (
            pl.scan_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id) & (pl.col("target_mouse_id") == target_mouse_id))
            .select(INDEX_COLS)
            .collect()
        )
        
        # ç‰¹å¾´é‡�åˆ—ã‚’èª­ã�¿è¾¼ã�¿
        feature = (
            pl.scan_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")
            .filter((pl.col("agent_mouse_id") == agent_mouse_id) & (pl.col("target_mouse_id") == target_mouse_id))
            .select(pl.exclude(INDEX_COLS))
            .collect()
        )

    # äºˆæ¸¬çµ�æ�œã‚’æ ¼ç´�ã�™ã‚‹ DataFrame ã‚’ä½œæˆ�ï¼ˆã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ã�®ã‚³ãƒ”ãƒ¼ï¼‰
    prediction_dataframe = index.clone()

    # ===== å�„è¡Œå‹•ï¼ˆbehaviorï¼‰ã�«å¯¾ã�—ã�¦äºˆæ¸¬ã‚’å®Ÿè¡Œ =====
    for row in group.rows(named=True):
        behavior = row["behavior"]  # ç�¾åœ¨ã�®è¡Œå‹•å��ï¼ˆä¾‹: "attack", "rear"ï¼‰

        # å�„ foldï¼ˆäº¤å·®æ¤œè¨¼ã�®åˆ†å‰²ï¼‰ã�®äºˆæ¸¬çµ�æ�œã‚’æ ¼ç´�ã�™ã‚‹ãƒªã‚¹ãƒˆ
        predictions = []  # äºˆæ¸¬ç¢ºç�‡
        prediction_labels = []  # äºˆæ¸¬ãƒ©ãƒ™ãƒ«ï¼ˆé–¾å€¤ã�§ 0/1 ã�«å¤‰æ�›ã�—ã�Ÿã‚‚ã�®ï¼‰

        # ä¿�å­˜ã�•ã‚Œã�Ÿ fold ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã‚’å�–å¾—
        # ä¾‹: results/AdaptableSnail/attack/fold_0, fold_1, fold_2
        fold_dirs = list((WORKING_DIR / "results" / lab_id / behavior).glob("fold_*"))
        if not fold_dirs:
            # è¨“ç·´ã�•ã‚Œã�Ÿãƒ¢ãƒ‡ãƒ«ã�Œè¦‹ã�¤ã�‹ã‚‰ã�ªã�„å ´å�ˆã�¯ã‚¹ã‚­ãƒƒãƒ—
            continue

        # å�„ fold ã�®ãƒ¢ãƒ‡ãƒ«ã�§äºˆæ¸¬ã‚’å®Ÿè¡Œ
        for fold_dir in fold_dirs:
            # ä¿�å­˜ã�•ã‚Œã�Ÿæœ€é�©é–¾å€¤ã‚’èª­ã�¿è¾¼ã�¿
            with open(fold_dir / "threshold.txt", "r") as f:
                threshold = float(f.read().strip())
            
            # XGBoost ãƒ¢ãƒ‡ãƒ«ã‚’èª­ã�¿è¾¼ã�¿
            model = xgb.Booster(model_file=fold_dir / "model.json")
            
            # ç‰¹å¾´é‡�ã‚’ XGBoost ã�®å…¥åŠ›å½¢å¼�ï¼ˆDMatrixï¼‰ã�«å¤‰æ�›
            dtest = xgb.DMatrix(feature, feature_names=feature.columns)
            
            # ãƒ¢ãƒ‡ãƒ«ã�§äºˆæ¸¬ã‚’å®Ÿè¡Œï¼ˆç¢ºç�‡å€¤ã‚’å�–å¾—ï¼‰
            fold_predictions = model.predict(dtest)
            
            # äºˆæ¸¬ç¢ºç�‡ã‚’ä¿�å­˜
            predictions.append(fold_predictions)
            
            # é–¾å€¤ã‚’é�©ç”¨ã�—ã�¦ãƒ©ãƒ™ãƒ«åŒ–ï¼ˆ1: è¡Œå‹•ã�‚ã‚Š, 0: è¡Œå‹•ã�ªã�—ï¼‰
            prediction_labels.append((fold_predictions >= threshold).astype(np.int8))

        # äºˆæ¸¬çµ�æ�œã‚’ DataFrame ã�«è¿½åŠ 
        # å�„ fold ã�®ã€Œäºˆæ¸¬ç¢ºç�‡ Ã— äºˆæ¸¬ãƒ©ãƒ™ãƒ«ã€�ã‚’åˆ—ã�¨ã�—ã�¦è¿½åŠ 
        # ï¼ˆãƒ©ãƒ™ãƒ«ã�Œ 0 ã�®å ´å�ˆã�¯ç¢ºç�‡ã‚‚ 0 ã�«ã�ªã‚Šã€�1 ã�®å ´å�ˆã�¯ç¢ºç�‡ã�Œã��ã�®ã�¾ã�¾æ®‹ã‚‹ï¼‰
        prediction_dataframe = prediction_dataframe.with_columns(
            *[
                pl.Series(name=f"{behavior}_{fold}", values=predictions[fold] * prediction_labels[fold], dtype=pl.Float32)
                for fold in range(len(fold_dirs))
            ]
        )

    # ===== æœ€ã‚‚ç¢ºç�‡ã�Œé«˜ã�„è¡Œå‹•ã‚’é�¸æŠ� =====
    
    # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ä»¥å¤–ã�®åˆ—å��ã‚’å�–å¾—ï¼ˆå�„è¡Œå‹•ã�®äºˆæ¸¬åˆ—ï¼‰
    cols = prediction_dataframe.select(pl.exclude(INDEX_COLS)).columns
    if not cols:
        # äºˆæ¸¬åˆ—ã�Œ 1 ã�¤ã‚‚ã�ªã�„å ´å�ˆã�¯è­¦å‘Šã‚’è¡¨ç¤ºã�—ã�¦ã‚¹ã‚­ãƒƒãƒ—
        tqdm.write(f"Warning: No predictions found for {lab_id}, {video_id}, {agent}, {target}")
        continue

    # å�„ãƒ•ãƒ¬ãƒ¼ãƒ ã�§æœ€ã‚‚ç¢ºç�‡ã�Œé«˜ã�„è¡Œå‹•ã‚’é�¸æŠ�
    prediction_labels_dataframe = prediction_dataframe.with_columns(
        pl.struct(pl.col(cols))  # å…¨äºˆæ¸¬åˆ—ã‚’æ§‹é€ ä½“ã�«ã�¾ã�¨ã‚�ã‚‹
        .map_elements(
            # å�„è¡Œï¼ˆãƒ•ãƒ¬ãƒ¼ãƒ ï¼‰ã�«å¯¾ã�—ã�¦ä»¥ä¸‹ã�®å‡¦ç�†ã‚’å®Ÿè¡Œ:
            # - ã�™ã�¹ã�¦ã�®äºˆæ¸¬å€¤ã�Œ 0 ã�ªã‚‰ "none"ï¼ˆè¡Œå‹•ã�ªã�—ï¼‰
            # - ã��ã‚Œä»¥å¤–ã�¯æœ€å¤§å€¤ã‚’æŒ�ã�¤è¡Œå‹•å��ã‚’è¿”ã�™
            lambda row: "none" if sum(row.values()) == 0 else (cols[np.argmax(list(row.values()))]).split("_")[0],
            return_dtype=pl.String,
        )
        .alias("prediction")  # æ–°ã�—ã�„åˆ—å��ã‚’ "prediction" ã�¨ã�™ã‚‹
    ).select(INDEX_COLS + ["prediction"])  # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹åˆ—ã�¨äºˆæ¸¬åˆ—ã�®ã�¿é�¸æŠ�

    # ===== é€£ç¶šã�™ã‚‹å�Œã�˜è¡Œå‹•ã‚’ã‚¤ãƒ™ãƒ³ãƒˆã�«ã�¾ã�¨ã‚�ã‚‹ =====
    
    group_submission = (
        prediction_labels_dataframe
        .filter((pl.col("prediction") != pl.col("prediction").shift(1)))  # è¡Œå‹•ã�Œå¤‰åŒ–ã�—ã�Ÿãƒ•ãƒ¬ãƒ¼ãƒ ã�®ã�¿æ®‹ã�™
        .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))  # æ¬¡ã�®å¤‰åŒ–ç‚¹ã‚’çµ‚äº†ãƒ•ãƒ¬ãƒ¼ãƒ ã�¨ã�™ã‚‹
        .filter(pl.col("prediction") != "none")  # "none"ï¼ˆè¡Œå‹•ã�ªã�—ï¼‰ã‚’é™¤å¤–
        .select(
            # æ��å‡ºå½¢å¼�ã�«å�ˆã‚�ã�›ã�¦åˆ—ã‚’é�¸æŠ�ãƒ»å¤‰æ�›
            pl.col("video_id"),
            ("mouse" + pl.col("agent_mouse_id").cast(str)).alias("agent_id"),  # ä¾‹: 1 â†’ "mouse1"
            pl.when(pl.col("target_mouse_id") == -1)  # target_mouse_id ã�Œ -1 ã�ªã‚‰
            .then(pl.lit("self"))  # "self" ã�«å¤‰æ�›
            .otherwise("mouse" + pl.col("target_mouse_id").cast(str))  # ã��ã‚Œä»¥å¤–ã�¯ "mouseN"
            .alias("target_id"),
            pl.col("prediction").alias("action"),  # è¡Œå‹•å��
            pl.col("video_frame").alias("start_frame"),  # é–‹å§‹ãƒ•ãƒ¬ãƒ¼ãƒ 
            pl.col("stop_frame"),  # çµ‚äº†ãƒ•ãƒ¬ãƒ¼ãƒ 
        )
    )

    # ã�“ã�®ã‚°ãƒ«ãƒ¼ãƒ—ã�®æ��å‡ºãƒ‡ãƒ¼ã‚¿ã‚’ãƒªã‚¹ãƒˆã�«è¿½åŠ 
    group_submissions.append(group_submission)

# ===== å…¨ã‚°ãƒ«ãƒ¼ãƒ—ã�®äºˆæ¸¬çµ�æ�œã‚’çµ�å�ˆ =====

# å…¨ã‚°ãƒ«ãƒ¼ãƒ—ã�®æ��å‡ºãƒ‡ãƒ¼ã‚¿ã‚’ç¸¦æ–¹å�‘ã�«çµ�å�ˆ
submission = pl.concat(group_submissions, how="vertical").sort(
    "video_id",
    "agent_id",
    "target_id",
    "action",
    "start_frame",
    "stop_frame",
)

# æ��å‡ºãƒ‡ãƒ¼ã‚¿ã�®å …ç‰¢åŒ–å‡¦ç�†ï¼ˆé‡�è¤‡å‰Šé™¤ã€�ãƒ•ãƒ¬ãƒ¼ãƒ ã�®ä¿®æ­£ã�ªã�©ï¼‰
submission = robustify(submission, test_dataframe, train_test="test")

# è¡Œç•ªå�·ï¼ˆrow_idï¼‰ã‚’è¿½åŠ ã�—ã�¦ CSV ãƒ•ã‚¡ã‚¤ãƒ«ã�¨ã�—ã�¦ä¿�å­˜
submission.with_row_index("row_id").write_csv(WORKING_DIR / "submission.csv")


!head submission.csv

