from sklearn.linear_model import ARDRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from pathlib import Path

import pandas as pd
import numpy as np
import warnings
import gc

warnings.filterwarnings("ignore")


root_path = Path("/kaggle/input/drw-crypto-market-prediction")

train_path = root_path / "train.parquet"
test_path = root_path / "test.parquet"
sample_submission_path = root_path / "sample_submission.csv"

target = "label"
seed = 42

xgb_params = {
    "learning_rate": 0.02,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1500,
    "reg_alpha": 10,
    "reg_lambda": 100,
    "subsample": 0.05,

    "n_jobs": -1,
    "random_state": seed,
    "verbosity": 0
}

hyperparameters = {
    'features': 70, # ~10%
}


train = pd.read_parquet(train_path).astype(np.float16) # to prevent OOM in kaggle
test = pd.read_parquet(test_path).astype(np.float16) # to prevent OOM in kaggle


parts = np.array_split(np.arange(len(train)), 3)
train_df = train.iloc[np.hstack([parts[0], parts[2]])].reset_index(drop=True)
val_df = train.iloc[parts[1]].reset_index(drop=True)

label = train_df[target]
val_label = val_df[target]

train_df.drop(columns=[target],inplace=True)
val_df.drop(columns=[target],inplace=True)

test_df = test[train_df.columns]


features = hyperparameters['features']
selector = SelectKBest(f_regression, k=features)
X_selected = selector.fit_transform(train_df, label)
X_test_selected = selector.transform(val_df)
selected_indices = selector.get_support()
selected_columns = train_df.columns[selected_indices]

filtered_columns = ["ask_qty", "bid_qty", "sell_qty", "volume"]

for i, feature in enumerate(selected_columns):
    X_train = (train_df[feature].to_numpy()).reshape(-1,1)
    X_val = (val_df[feature].to_numpy()).reshape(-1,1)
    lr = LinearRegression()
    lr.fit(X_train, label)
    val_preds = lr.predict(X_val)
    score = pearsonr(val_label, val_preds)[0]
    if score <= 0:
        continue
    filtered_columns.append(feature)


ard_regressor = ARDRegression()
ard_regressor.fit(train_df[filtered_columns],label)
val_preds_ard = ard_regressor.predict(val_df[filtered_columns])
test_preds_ard = ard_regressor.predict(test_df[filtered_columns])

xgb_regressor = XGBRegressor(**xgb_params)
xgb_regressor.fit(train_df[filtered_columns],label)
val_preds_xgb = xgb_regressor.predict(val_df[filtered_columns])
test_preds_xgb = xgb_regressor.predict(test_df[filtered_columns])

print(f"Blending")
best_score = 0
best_ratio = 0
for ratio  in np.linspace(0, 1, 11):
    val_preds = val_preds_xgb * ratio + val_preds_ard * (1 - ratio) 
    score = pearsonr(val_label, val_preds)[0]
    print(f"ratio: {ratio:.1f} * xgb + {1 - ratio:.1f} * ard, score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_ratio = ratio

test_preds = test_preds_xgb * best_ratio + test_preds_ard * (1 - best_ratio) 
    
submission = pd.read_csv(sample_submission_path)
submission["prediction"] = test_preds
submission.to_csv(f"submission.csv", index=False)

