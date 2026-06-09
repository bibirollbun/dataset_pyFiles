# ==========================
# ðŸŽµ Kaggle Beats Per Minute Prediction (Ensemble Edition)
# ==========================

# 1. Imports
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import optuna

# ==========================
# 2. Load Data
# ==========================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# ==========================
# 3. Preprocessing
# ==========================
def preprocess(df, ref_df=None):
    df = df.copy()
    outlier_cols = [
        "AudioLoudness", "VocalContent", "AcousticQuality", 
        "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore"
    ]
    if ref_df is None:
        ref_df = df
    for col in outlier_cols:
        low, high = ref_df[col].quantile([0.005, 0.995])
        df[col] = df[col].clip(lower=low, upper=high)
    pos_skew = ["VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood"]
    for col in pos_skew:
        df[col] = np.log1p(df[col])
    df["AudioLoudness"] = np.cbrt(df["AudioLoudness"])
    return df

train_fe = preprocess(train)
test_fe = preprocess(test, ref_df=train)

X = train_fe.drop(columns=["id", "BeatsPerMinute"])
y = train_fe["BeatsPerMinute"]
X_test = test_fe.drop(columns=["id"])

# ==========================
# 4. LightGBM Hyperparameter Tuning (Optuna)
# ==========================
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "seed": 42
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)



print("Best RMSE:", study.best_trial.value)
print("Best Params:", study.best_trial.params)


best_params = study.best_trial.params
best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "seed": 42
})

# ==========================
# 5. Define Models
# ==========================
lgb_model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", 
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    verbose=0
)

# ==========================
# 6. Stacking Ensemble
# ==========================
estimators = [
    ("lgb", lgb_model),
    ("xgb", xgb_model),
    ("cat", cat_model)
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV(),
    n_jobs=-1
)

# Train ensemble
stack.fit(X, y)


# ==========================
# 7. Predictions & Submission
# ==========================
test_preds = stack.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_preds
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created with LightGBM + XGBoost + CatBoost Stacking")


submission.head()

