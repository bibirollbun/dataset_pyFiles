# =========================
# Cell 1 â€” Imports & Config
# =========================
import os, gc, random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_PATH = "/kaggle/input/nfl-big-data-bowl-2026-analytics/train/input_2023_w01.csv"
TEST_PATH  = "/kaggle/input/nfl-big-data-bowl-2026-analytics/train/input_2023_w02.csv"
ID_COL = "nfl_id"



# =========================
# Cell 2 â€” Load Data
# =========================
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)



# =========================
# Cell 3 â€” Preprocessing
# =========================
def preprocess(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("category").cat.codes
        df[c] = df[c].fillna(-999)
    return df

train_p = preprocess(train)
test_p  = preprocess(test)

FEATURES = [c for c in train_p.columns if c not in [ID_COL]]
print("Features:", len(FEATURES))



# =========================
# Cell 4 â€” Baseline Model
# =========================
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": SEED,
    "verbosity": -1
}

X = train_p[FEATURES]
y = train_p["x"]  # Example target: player x-coordinate

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
preds = np.zeros(len(test_p))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    dtrain = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
    dval   = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])
    
    model = lgb.train(params, dtrain, valid_sets=[dtrain, dval],
                      valid_names=["train","valid"],
                      verbose_eval=200,
                      early_stopping_rounds=100)
    
    oof[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
    preds += model.predict(test_p[FEATURES], num_iteration=model.best_iteration) / kf.n_splits

print("OOF RMSE:", mean_squared_error(y, oof, squared=False))



# =========================
# Cell 5 â€” Inference
# =========================
def predict_position(df):
    return model.predict(df[FEATURES], num_iteration=model.best_iteration)

sample_preds = predict_position(test_p.head(10))
print("Sample predictions:", sample_preds)



# =========================
# Cell 6 â€” Submission
# =========================
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    "x_pred": preds
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved")
submission.head()


