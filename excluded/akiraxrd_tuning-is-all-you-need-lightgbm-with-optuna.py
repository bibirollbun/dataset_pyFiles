import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

import optuna

import warnings
from pathlib import Path


SEED = 42
INPUT_DIR = Path("/kaggle/input/playground-series-s5e9")
TRAIN_DIR = INPUT_DIR / "train.csv"
TEST_DIR = INPUT_DIR / "test.csv"


train_df = pd.read_csv(TRAIN_DIR, index_col="id")
test_df = pd.read_csv(TEST_DIR, index_col="id")

TARGET_COLUMN = "BeatsPerMinute"
X = train_df.copy().drop(columns=TARGET_COLUMN)
y = train_df.copy()[TARGET_COLUMN]

X_test = test_df.copy()

X.head()


# def objective(trial):
#     params = {
#         "metric": "rmse",
#         "boosting_type": "gbdt",
#         "verbosity": -1,
#         "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 20, 500),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
#         "n_estimators": 10100
#     }

#     num_folds = 3
#     kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
#     rmse_scores: list[float] = []
    
#     for i, (train_idx, val_idx) in enumerate(kf.split(X), 1):
#         print(f"\n[KFOLD] Fold {i:3d}/{num_folds}")
        
#         x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model = lgb.LGBMRegressor(**params)
        
#         callbacks = [
#             lgb.early_stopping(stopping_rounds=200),
#             lgb.log_evaluation(100)
#         ]
        
#         model.fit(
#             x_train, y_train,
#             eval_set=[(x_val, y_val)],
#             callbacks=callbacks
#         )

#         preds = model.predict(x_val)
#         rmse_score = np.sqrt(mean_squared_error(y_val, preds))
#         rmse_scores.append(rmse_score)
        
#     return np.mean(rmse_scores)


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=15)

# print(f"\nBest parameters: {study.best_params}")


try:
    best_params = study.best_params

except:
    best_params = {
        'learning_rate': 0.039802115601828794,
        'num_leaves': 276,
        'max_depth': 4,
        'min_child_samples': 11,
        'subsample': 0.7839053571377562,
        'colsample_bytree': 0.7548005467520535,
        'reg_alpha': 1.1550197166349566e-08,
        'reg_lambda': 5.995317758331191e-05
    }

model = lgb.LGBMRegressor(**best_params)
model.fit(
    X, y,
    eval_set=[(X, y)]
)
print("[INFO] Model successfully trained")


preds = model.predict(X_test)
submission = pd.DataFrame({
    "id": X_test.index,
    TARGET_COLUMN: preds
})
submission = submission.set_index("id")

submission.sample(5)


submission.to_csv("submission.csv")

