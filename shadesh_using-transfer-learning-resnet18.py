# ==========================================================
# RSNA Intracranial Aneurysm Detection - Full Multi-Label Pipeline
# Kaggle-Compatible (uses CSV + test DICOMs)
# ==========================================================

!pip install polars -q
!pip install pydicom -q
!pip install xgboost -q

import os
import shutil
import pydicom
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import kaggle_evaluation.rsna_inference_server as rsna_server

# ==========================================================
# Step 1: Paths and Data
# ==========================================================
data_path = "/kaggle/input/rsna-intracranial-aneurysm-detection/"

train_csv = pd.read_csv(os.path.join(data_path, "train.csv"))
train_localizers = pd.read_csv(os.path.join(data_path, "train_localizers.csv"))

ID_COL = "SeriesInstanceUID"
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# ==========================================================
# Step 2: Feature Engineering
# ==========================================================
# Feature: number of localizers per series
train_features = (
    train_localizers.groupby(ID_COL)
    .size()
    .reset_index(name="num_localizers")
)

train_df = train_csv.merge(train_features, on=ID_COL, how="left")
train_df["num_localizers"].fillna(0, inplace=True)

# You can add more metadata features here
X = train_df[["num_localizers"]].values
y = train_df[LABEL_COLS].values

# ==========================================================
# Step 3: Train Multi-Label Models
# ==========================================================
models = {}
val_scores = {}

for i, col in enumerate(LABEL_COLS):
    y_col = y[:, i]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_col, test_size=0.2, random_state=42
    )

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    clf.fit(X_train, y_train)
    models[col] = clf

    # Validation AUC
    try:
        val_preds = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_preds)
        val_scores[col] = auc
    except:
        val_scores[col] = None

print("Validation AUCs:")
for col, auc in val_scores.items():
    print(f"{col}: {auc}")

# ==========================================================
# Step 4: Prediction Function for Kaggle
# ==========================================================
def predict(series_path: str):
    """
    Predict function for RSNA Aneurysm Detection.
    Args:
        series_path (str): Path to series folder.
    Returns:
        pd.DataFrame: Predictions for all 14 labels.
    """
    series_id = os.path.basename(series_path)

    # Feature: number of DICOM slices in test folder
    try:
        num_files = len(os.listdir(series_path))
    except:
        num_files = 0

    X_test = np.array([[num_files]])

    preds = {}
    for col in LABEL_COLS:
        if col in models:
            preds[col] = float(models[col].predict_proba(X_test)[:, 1][0])
        else:
            preds[col] = 0.0

    df = pd.DataFrame([[series_id, *preds.values()]], columns=[ID_COL, *LABEL_COLS])
    return df.drop(columns=[ID_COL])

# ==========================================================
# Step 5: Inference Server (Kaggle)
# ==========================================================
inference_server = rsna_server.RSNAInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet("/kaggle/working/submission.parquet").head())

# Clean up
shutil.rmtree("/kaggle/shared", ignore_errors=True)


