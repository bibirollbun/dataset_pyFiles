import pandas as pd
import numpy as np

!pip install tab-transformer-pytorch scikit-learn


import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from tab_transformer_pytorch import FTTransformer
from pandas.api.types import CategoricalDtype

df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

df.drop(['id'], axis=1, inplace=True)

# -----------------------------
# CONFIG
# -----------------------------
TARGET_COL = "diagnosed_diabetes"

cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]

num_cols = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
]

# =====================================================
# 1. PREPROCESS TRAIN DF (CATEGORICAL ENCODING + SCALING)
# =====================================================

# Ensure correct dtypes
df[TARGET_COL] = df[TARGET_COL].astype("float32")

cardinalities = []
cat_categories = {}  # store category mapping for later (test_df)

for col in cat_cols:
    df[col] = df[col].astype("category")
    cat_categories[col] = df[col].cat.categories.tolist()
    df[col] = df[col].cat.codes
    cardinalities.append(df[col].nunique())

# Scale numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


batch_size = 1024 * 10

class DiabetesTestDataset(Dataset):
    """No labels, for test_df prediction."""
    def __init__(self, X, cat_cols, num_cols):
        self.cat_data = X[cat_cols].values.astype("int64")
        self.num_data = X[num_cols].values.astype("float32")

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        x_cat = torch.tensor(self.cat_data[idx], dtype=torch.long)
        x_num = torch.tensor(self.num_data[idx], dtype=torch.float32)
        return x_cat, x_num



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# 6. LOAD BEST MODEL FOR INFERENCE
# =====================================================

best_model = FTTransformer(
    categories=tuple(cardinalities),
    num_continuous=len(num_cols),
    dim=32,
    dim_out=1,
    depth=4,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
).to(device)

best_model.load_state_dict(torch.load("/kaggle/input/diabetes-prediction-ft-transformer/best_ft_transformer_auc.pt", map_location=device))
best_model.eval()

# =====================================================
# 7. PREPROCESS test_df AND PREDICT
# =====================================================

test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
ids = test_df.pop('id')

# Ensure all feature columns present in test_df; ignore target if it exists
if TARGET_COL in test_df.columns:
    test_features = test_df.drop(columns=[TARGET_COL]).copy()
else:
    test_features = test_df.copy()

# --- Encode categoricals with SAME categories as train ---
for col in cat_cols:
    # Use training categories; unseen values -> NaN -> code -1 -> map to 0
    dtype = CategoricalDtype(categories=cat_categories[col])
    test_features[col] = test_features[col].astype(dtype)
    codes = test_features[col].cat.codes  # -1 for NaN/unseen
    codes = codes.where(codes >= 0, 0)    # map -1 to 0 to stay in range
    test_features[col] = codes.astype("int64")

# --- Scale numerical features with SAME scaler ---
test_features[num_cols] = scaler.transform(test_features[num_cols])

# Create test dataset & loader
test_dataset = DiabetesTestDataset(test_features, cat_cols, num_cols)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Run inference ---
all_test_probs = []
all_test_preds = []

with torch.no_grad():
    for x_cat, x_num in test_loader:
        x_cat, x_num = x_cat.to(device), x_num.to(device)
        logits = best_model(x_cat, x_num).squeeze(1)
        probs = torch.sigmoid(logits)

        all_test_probs.extend(probs.cpu().numpy())
        all_test_preds.extend((probs > 0.5).long().cpu().numpy())

# Attach predictions to test_df
test_df_pred = test_df.copy()
test_df_pred["diabetes_prob"] = all_test_probs
test_df_pred["diagnosed_diabetes"] = all_test_preds  # 0 / 1
test_df_pred['id'] =  ids
# test_df_pred now has predictions.
# Example: print first few rows
print(test_df_pred.head())



submission = test_df_pred[['id', 'diagnosed_diabetes']]


submission.head()


submission.diagnosed_diabetes.min(), submission.diagnosed_diabetes.mean(), submission.diagnosed_diabetes.max()


submission.to_csv("submission.csv", index=False)




