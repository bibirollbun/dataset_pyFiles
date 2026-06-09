# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from pathlib import Path
import os
import sys

# ------------------------------------------------------------
# Input dataset checks
# ------------------------------------------------------------
COMP_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
STARTER_DIR = Path("/kaggle/input/mabe-starter-train-ja")
MABE_PKG_DIR = Path("/kaggle/input/mabe-package")

if not COMP_DIR.exists():
    raise FileNotFoundError(
        "Competition dataset 'MABe Challenge - Social Action Recognition in Mice' "
        "must be attached as an input."
    )

if not STARTER_DIR.exists():
    raise FileNotFoundError(
        "Dataset 'mabe-starter-train-ja' is not attached. "
        "Click 'Add input' and add it before running."
    )

if not MABE_PKG_DIR.exists():
    raise FileNotFoundError(
        "Dataset 'mabe-package' is not attached. "
        "It provides the offline xgboost wheel used by the starter models."
    )
#defines the dfs


# ------------------------------------------------------------
# Install xgboost from offline wheel (no internet)
# ------------------------------------------------------------
!pip install -q --no-index --find-links=/kaggle/input/mabe-package xgboost==3.1.1

# ------------------------------------------------------------
# Copy helper scripts and trained models from starter dataset
# ------------------------------------------------------------
!cp /kaggle/input/mabe-starter-train-ja/self_features.py .
!cp /kaggle/input/mabe-starter-train-ja/pair_features.py .
!cp /kaggle/input/mabe-starter-train-ja/robustify.py .
!cp -r /kaggle/input/mabe-starter-train-ja/results .


INPUT_DIR = COMP_DIR
TRAIN_TRACKING_DIR = INPUT_DIR / "train_tracking"
TRAIN_ANNOTATION_DIR = INPUT_DIR / "train_annotation"
TEST_TRACKING_DIR = INPUT_DIR / "test_tracking"

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

SELF_FEATURE_DIR = WORKING_DIR / "self_features"
PAIR_FEATURE_DIR = WORKING_DIR / "pair_features"
SELF_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
PAIR_FEATURE_DIR.mkdir(parents=True, exist_ok=True)



import gc
import re
import ast
import itertools
from pathlib import Path

import numpy as np

# polars is preinstalled on Kaggle GPU/CPU images
try:
    import polars as pl
except ImportError:
    raise ImportError(
        "polars is not available in this environment. "
        "Use a Kaggle GPU/CPU notebook image where polars is preinstalled."
    )

import xgboost as xgb
from tqdm.auto import tqdm


# Helper scripts from starter notebook
%run -i self_features.py
%run -i pair_features.py
%run -i robustify.py


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



def parse_behaviors_column(behaviors_str):
    if behaviors_str is None:
        return []
    return ast.literal_eval(behaviors_str)

def build_behavior_dataframe(test_df: pl.DataFrame) -> pl.DataFrame:
    """
    Expand behaviors_labeled into one row per (lab, video, agent, target, behavior).
    """
    behavior_df = (
        test_df
        .filter(pl.col("behaviors_labeled").is_not_null())
        .select(["lab_id", "video_id", "behaviors_labeled"])
        .with_columns(
            pl.col("behaviors_labeled")
            .map_elements(
                parse_behaviors_column,
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("behaviors_labeled_list")
        )
        .explode("behaviors_labeled_list")
        .rename({"behaviors_labeled_list": "behaviors_labeled_element"})
        .with_columns(
            pl.col("behaviors_labeled_element").str.split(",").list.get(0)
            .str.replace_all("[()' ]", "")
            .alias("agent"),
            pl.col("behaviors_labeled_element").str.split(",").list.get(1)
            .str.replace_all("[()' ]", "")
            .alias("target"),
            pl.col("behaviors_labeled_element").str.split(",").list.get(2)
            .str.replace_all("[()' ]", "")
            .alias("behavior"),
        )
        .select(["lab_id", "video_id", "agent", "target", "behavior"])
    )
    return behavior_df

def extract_mouse_id(mouse_str: str) -> int:
    """
    Convert 'mouse1' -> 1, 'mouse2' -> 2, 'self' -> -1.
    """
    if mouse_str == "self":
        return -1
    m = re.search(r"mouse(\d+)", mouse_str)
    if m:
        return int(m.group(1))
    raise ValueError(f"Unexpected mouse id format: {mouse_str}")

def load_features_for_group(lab_id, video_id, agent, target):
    """
    Load per frame features for a given (lab, video, agent, target) group.
    Returns:
      index_df   - DataFrame with INDEX_COLS
      feature_df - DataFrame with feature columns only
    """
    agent_mouse_id = extract_mouse_id(agent)
    target_mouse_id = extract_mouse_id(target)

    if target == "self":
        feature_path = SELF_FEATURE_DIR / f"{video_id}.parquet"
        scan = pl.scan_parquet(feature_path).filter(
            pl.col("agent_mouse_id") == agent_mouse_id
        )
    else:
        feature_path = PAIR_FEATURE_DIR / f"{video_id}.parquet"
        scan = pl.scan_parquet(feature_path).filter(
            (pl.col("agent_mouse_id") == agent_mouse_id)
            & (pl.col("target_mouse_id") == target_mouse_id)
        )

    full_df = scan.collect()
    if full_df.height == 0:
        return full_df, full_df

    index_df = full_df.select(INDEX_COLS)
    feature_df = full_df.select(pl.exclude(INDEX_COLS))
    return index_df, feature_df

def load_models_for_behavior(lab_id: str, behavior: str):
    """
    Load all fold models and thresholds for a given (lab, behavior).
    Returns list of (model, threshold).
    """
    behavior_dir = WORKING_DIR / "results" / lab_id / behavior
    fold_dirs = sorted(behavior_dir.glob("fold_*"))
    models = []
    for fold_dir in fold_dirs:
        model_file = fold_dir / "model.json"
        thr_file = fold_dir / "threshold.txt"
        if not model_file.exists() or not thr_file.exists():
            continue
        with open(thr_file, "r") as f:
            threshold = float(f.read().strip())
        model = xgb.Booster(model_file=str(model_file))
        models.append((model, threshold))
    return models

def predict_for_group(
    lab_id: str,
    video_id: int,
    agent: str,
    target: str,
    group_behaviors: pl.DataFrame,
    ):
    """
    Run inference for one group of (lab_id, video_id, agent, target).

    Improvements:
      - Aggregate folds per behavior into a single score column
        (mean of thresholded probabilities).
      - Pick best behavior per frame using those aggregated scores.
    """
    index_df, feature_df = load_features_for_group(lab_id, video_id, agent, target)

    if feature_df.height == 0:
        return None

    # Create XGBoost DMatrix once per group and reuse across behaviors
    dtest = xgb.DMatrix(feature_df.to_pandas(), feature_names=feature_df.columns)

    prediction_df = index_df.clone()
    used_cols = []
    unique_behaviors = (group_behaviors.select("behavior").unique()["behavior"].to_list()
    )

    for behavior in unique_behaviors:
        models = load_models_for_behavior(lab_id, behavior)
        if not models:
            # No trained model for this (lab, behavior) in the starter models
            continue

        # Aggregate over folds: mean of thresholded probabilities
        agg_scores = np.zeros(feature_df.height, dtype=np.float32)

        for model, threshold in models:
            probs = model.predict(dtest)
            labels = (probs >= threshold).astype(np.int8)
            agg_scores += probs * labels

        agg_scores /= max(len(models), 1)

        col_name = behavior
        prediction_df = prediction_df.with_columns(
            pl.Series(name=col_name, values=agg_scores)
        )
        used_cols.append(col_name)

    if not used_cols:
        return None

    # Pick best behavior per frame (over behaviors only)
    cols = used_cols

    prediction_labels_df = (
        prediction_df
        .with_columns(
            pl.struct(pl.col(cols))
            .map_elements(
                lambda row: (
                    "none"
                    if sum(row.values()) == 0
                    else cols[int(np.argmax(list(row.values())))]
                ),
                return_dtype=pl.String,
            )
            .alias("prediction")
        )
        .select(INDEX_COLS + ["prediction"])
    )

    # Convert per frame labels into time segments
    agent_mouse_id = extract_mouse_id(agent)
    target_mouse_id = extract_mouse_id(target) 
    group_submission = (
        prediction_labels_df
        .filter(pl.col("prediction") != pl.col("prediction").shift(1))
        .with_columns(
            pl.col("video_frame").shift(-1).alias("stop_frame")
        )
        .filter(pl.col("prediction") != "none")
        .select(
            pl.col("video_id"),
            (pl.lit("mouse") + pl.lit(agent_mouse_id).cast(pl.Utf8)).alias("agent_id"),
            pl.when(pl.lit(target_mouse_id) == -1)
            .then(pl.lit("self"))
            .otherwise(pl.lit("mouse") + pl.lit(target_mouse_id).cast(pl.Utf8))
            .alias("target_id"),
            pl.col("prediction").alias("action"),
            pl.col("video_frame").alias("start_frame"),
            pl.col("stop_frame"),
        )
    )

    return group_submission


    


print('Load test metadata:')
test_df = pl.read_csv(COMP_DIR/'test.csv')
print(test_df.head())

#behavior_df, list of behaviors? why not label encoder?

behavior_df = build_behavior_dataframe(test_df)

print('behavior_df')

print(behavior_df.head())

groups = list(
    behavior_df.group_by("lab_id", "video_id", "agent", "target", maintain_order=True)
)
print(f"Number of (lab, video, agent, target) groups: {len(groups)}")

print("Generating self and pair features for all test videos...")

rows = test_df.rows(named=True)


for row in tqdm(rows, total=len(rows)): #make self/pair features and it seems like the agent and target is predefined in the make_self_features from the tracking and the metadata...
    lab_id = test_df['lab_id'].item()
    video_id = test_df['video_id'].item()

    tracking_path = TEST_TRACKING_DIR / f"{lab_id}/{video_id}.parquet"
    print(tracking_path)
    tracking = pl.read_parquet(tracking_path)

    self_feat = make_self_features(metadata=row, tracking=tracking)

    pair_feat = make_pair_features(metadata=row, tracking=tracking)
    
    print(self_feat)
    print(pair_feat)

    self_feat.write_parquet(SELF_FEATURE_DIR / f"{video_id}.parquet")
    pair_feat.write_parquet(PAIR_FEATURE_DIR / f"{video_id}.parquet")

    del self_feat, pair_feat, tracking
    gc.collect()

print('Running inference and creating group submissions:')

group_submissions = []

for (lab_id, video_id, agent, target), group in tqdm(groups, total=len(groups)):
    group_submission = predict_for_group(
        lab_id=lab_id,
        video_id=video_id,
        agent=agent,
        target=target,
        group_behaviors=group,
    )

    if group_submission is not None and group_submission.height > 0:
        group_submissions.append(group_submission)

if not group_submissions:
    raise RuntimeError(
        "No submissions were generated. "
        "Check that starter models exist under /kaggle/working/results."
    )

submission = pl.concat(group_submissions, how='vertical').sort(
        "video_id",
        "agent_id",
        "target_id",
        "action",
        "start_frame",
        "stop_frame",
    )

print("Initial submission rows:", submission.height)

# ============================================================
# 4. Robustify and final clean up
# ============================================================
print("Running robustify on submission...")
submission = robustify(submission, test_df, train_test="test")

# Keep only valid intervals
submission = submission.filter(pl.col("start_frame") < pl.col("stop_frame"))

# Drop ultra short segments (likely noise)
submission = submission.with_columns(
    (pl.col("stop_frame") - pl.col("start_frame")).alias("duration")
).filter(pl.col("duration") >= 2).drop("duration")

print("Rows after robustify, validity check and duration filter:", submission.height)

submission = submission.to_pandas()

submission = submission.reset_index()

submission.index.name = 'row_id'

print(submission.head())

submission.to_csv('submission.csv')

print('csv saved to submission.csv')

# Add row_id and save as submission.csv
#final_submission = submission.with_row_index("row_id")
#print(final_submission.head())
#output_dir = 'kaggle/working'
#Path(output_dir).mkdir(parents=True, exist_ok=True)
# The parents=True argument creates necessary parent directories ('kaggle', then 'working')
# The exist_ok=True prevents an error if the directory already exists (good practice for re-runs)

# --- Now write the CSV file ---
#output_path = os.path.join(output_dir, 'submission.csv')
#final_submission.write_csv(output_path, encoding="utf8")

#print(f"CSV file has been created successfully at {output_path}")



