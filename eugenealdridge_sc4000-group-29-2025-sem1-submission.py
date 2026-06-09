# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- 1. Imports --- 
import os
import json
import glob
import polars as pl
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import kaggle_evaluation.mcts_inference_server


# --- 2. Configuration --- 

# Path to the dataset containing the pre-trained models
MODEL_DATASET_PATH = "/kaggle/input/sc4000-group29-models-2025-sem1/cb-greedy depth- optuma lr - ML4000"

# Weight for the Test-Time Augmentation (TTA) predictions.
# 0.5 = 50/50 split (original notebook's behavior)
TTA_WEIGHT = 0.7


# --- 3. Feature Engineering Functions --- 

def augment_inversion(inp_df):
    """Creates a new dataframe with agent1 and agent2 swapped."""
    # This function is already compatible with modern polars
    return inp_df.with_columns(
        pl.col("agent1").alias("agent2"),
        pl.col("agent2").alias("agent1"),
        (1 - pl.col("original_agent_order")).alias("original_agent_order"),
        (1 - pl.col("AdvantageP1")).alias("AdvantageP1"),
        (1 - pl.col("AdvantageP2")).alias("AdvantageP2"),
        pl.when(pl.col("SwapOptionPlayerID") == "P2").then(pl.lit("P1")).otherwise(pl.lit("Any")).alias("SwapOptionPlayerID"),
    )


def extract_agent_details(inp_df):
    """Parses the agent string columns into new features."""
    
    # --- FIXED --- 
    # Changed `.str.split_fixed` back to `.str.split_exact`
    # which is the correct name in modern polars (v1.x.x)
    inp_df = (
        inp_df.with_columns(
                pl.col("agent1").str.split_exact("-", 5) # <-- This line was fixed
                .struct.rename_fields(["agent1_dropcol", "agent1_selection", "agent1_expconst", "agent1_playout", "agent1_scorebounds"])
                .alias("fields")
            ).unnest("fields")
            .with_columns(
                pl.col("agent2").str.split_exact("-", 5) # <-- This line was fixed
                .struct.rename_fields(["agent2_dropcol", "agent2_selection", "agent2_expconst", "agent2_playout", "agent2_scorebounds"])
                .alias("fields")
            ).unnest("fields")
        ).drop(["agent1_dropcol", "agent2_dropcol"])

    # This part was already compatible
    inp_df = inp_df.with_columns(
                ((pl.col("AdvantageP1") * pl.col("Completion")) + (pl.col("Drawishness")/2)).alias("adv_p1_adj"),
                ((pl.col("AdvantageP2") * pl.col("Completion")) + (pl.col("Drawishness")/2)).alias("adv_p2_adj"),
        )    
    return inp_df


# --- 4. Load Models and Columns ---
print(f"Loading models from: {MODEL_DATASET_PATH}")

model_filepaths = sorted(glob.glob(f"{MODEL_DATASET_PATH}/model*.cb"))
print(f"Found {len(model_filepaths)} model files.")

with open(f'{MODEL_DATASET_PATH}/used_cols.json') as f:
    COLUMNS_TO_USE = json.load(f)

# Remove raw agent strings if they slipped into the JSON
ban_cols = {"agent1", "agent2"}
if any(c in COLUMNS_TO_USE for c in ban_cols):
    print("⚠️ Removing raw agent string columns from used_cols.json at runtime:", sorted(ban_cols & set(COLUMNS_TO_USE)))
    COLUMNS_TO_USE = [c for c in COLUMNS_TO_USE if c not in ban_cols]

print(f"Loaded {len(COLUMNS_TO_USE)} feature names (after sanitization).")

models_list = []
for path in model_filepaths:
    m = CatBoostRegressor()
    m.load_model(path)
    models_list.append(m)

print(f"Successfully loaded {len(models_list)} models.")

# Helper: which of the used columns should be treated as categorical
AGENT_PART_COLS = [
    "agent1_selection","agent1_expconst","agent1_playout","agent1_scorebounds",
    "agent2_selection","agent2_expconst","agent2_playout","agent2_scorebounds",
]

# Build categorical index list from COLUMNS_TO_USE
CAT_COL_NAMES = [c for c in AGENT_PART_COLS if c in COLUMNS_TO_USE]
CAT_COL_IDXS  = [COLUMNS_TO_USE.index(c) for c in CAT_COL_NAMES]
print(f"Categorical cols used: {CAT_COL_NAMES} -> indices {CAT_COL_IDXS}")



# --- 5. Define the Predict Function ---
from catboost import Pool

def predict(test: pl.DataFrame, sample_sub: pl.DataFrame):
    """The main prediction function called by the inference server."""
    # 1) Base features
    base_test_df = test.with_columns(
        pl.lit(0).alias("original_agent_order"),
        (1 - pl.col("AdvantageP1")).alias("AdvantageP2"),
        pl.when(pl.col("SwapOption") == 1).then(pl.lit("P2")).otherwise(pl.lit("Any")).alias("SwapOptionPlayerID"),
    )

    # 2) Derived features
    test_pl_normal = extract_agent_details(base_test_df)
    test_pl_aug    = extract_agent_details(augment_inversion(base_test_df))

    # Convert to pandas
    test_pd_normal = test_pl_normal.to_pandas()
    test_pd_aug    = test_pl_aug.to_pandas()

    # 3) Ensure expconst are strings for CatBoost categoricals
    for c in ("agent1_expconst", "agent2_expconst"):
        if c in test_pd_normal.columns:
            test_pd_normal[c] = test_pd_normal[c].astype(str)
        if c in test_pd_aug.columns:
            test_pd_aug[c] = test_pd_aug[c].astype(str)

    # 4) Slice columns exactly as used in training
    Xn = test_pd_normal[COLUMNS_TO_USE]
    Xa = test_pd_aug[COLUMNS_TO_USE]

    # 5) Build CatBoost Pools with categorical feature indices
    pool_n = Pool(Xn, cat_features=CAT_COL_IDXS)
    pool_a = Pool(Xa, cat_features=CAT_COL_IDXS)

    # 6) Ensemble predictions
    normal_predictions = []
    aug_predictions = []

    for model in models_list:
        normal_predictions.append(model.predict(pool_n))
        aug_predictions.append(-model.predict(pool_a))

    mean_normal_pred = np.mean(normal_predictions, axis=0)
    mean_aug_pred    = np.mean(aug_predictions, axis=0)

    blended = (1 - TTA_WEIGHT) * mean_normal_pred + (TTA_WEIGHT) * mean_aug_pred
    scaled  = blended * 1.15 + 0.033
    #scaled  = blended
    final   = np.clip(scaled, -1, 1)

    return sample_sub.with_columns(pl.lit(final).alias('utility_agent1'))



# --- 6. Run Inference Server --- 

inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
            '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
        )
    )

