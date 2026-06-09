import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


SEED = 42
N_FOLDS = 5


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

TARGET = "accident_risk"
ID_COL = "id"

X = train.drop([TARGET, ID_COL], axis=1)
y = train[TARGET]
X_test = test.drop(ID_COL, axis=1)


# Identifying categorical columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

# Combining the train + test for consistent encoding
full = pd.concat([X, X_test], axis=0)

# One-hot encoding
full = pd.get_dummies(full, columns=cat_cols, drop_first=True)

# Splitting back
X = full.iloc[:len(X)].reset_index(drop=True)
X_test = full.iloc[len(X):].reset_index(drop=True)

print("Train shape:", X.shape)
print("Test shape:", X_test.shape)


kf = KFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)


lgb_params = {"n_estimators": 2000,"learning_rate": 0.02,"num_leaves": 64,"subsample": 0.8,
              "colsample_bytree": 0.8,"reg_alpha": 0.1,"reg_lambda": 0.1,"random_state": SEED,"n_jobs": -1}


xgb_params = {"n_estimators": 2000,"learning_rate": 0.02,"max_depth": 8,"subsample": 0.8,"colsample_bytree": 0.8,
    "reg_alpha": 0.1,"reg_lambda": 0.1,"tree_method": "hist","random_state": SEED,"n_jobs": -1}


cat_params = {"iterations": 2000,"learning_rate": 0.03,"depth": 8,"loss_function": "RMSE","random_seed": SEED,
              "verbose": False}


def train_model(model, X, y, X_test):
    oof = np.zeros(len(X))
    preds_test = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict(X_val)
        preds_test += model.predict(X_test) / N_FOLDS

        rmse = np.sqrt(mean_squared_error(y_val, oof[val_idx]))
        scores.append(rmse)

        print(f"Fold {fold+1} RMSE: {rmse:.5f}")

    print(f"Mean RMSE: {np.mean(scores):.5f} | Std: {np.std(scores):.5f}\n")
    return oof, preds_test


lgb_oof, lgb_test = train_model(LGBMRegressor(**lgb_params), X, y, X_test)

xgb_oof, xgb_test = train_model(XGBRegressor(**xgb_params), X, y, X_test)

cat_oof, cat_test = train_model(CatBoostRegressor(**cat_params), X, y, X_test)


X_stack = np.column_stack([lgb_oof, xgb_oof, cat_oof])
X_test_stack = np.column_stack([lgb_test, xgb_test, cat_test])


meta_model = Ridge(alpha=1.0)

meta_oof = np.zeros(len(X))
meta_test = np.zeros(len(X_test))
scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_stack)):
    X_tr, X_val = X_stack[tr_idx], X_stack[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    meta_model.fit(X_tr, y_tr)
    meta_oof[val_idx] = meta_model.predict(X_val)
    meta_test += meta_model.predict(X_test_stack) / N_FOLDS

    rmse = np.sqrt(mean_squared_error(y_val, meta_oof[val_idx]))
    scores.append(rmse)
    print(f"Meta Fold {fold+1} RMSE: {rmse:.5f}")

print(f"\nSTACK RMSE: {np.mean(scores):.5f} | Std: {np.std(scores):.5f}")


submission = pd.DataFrame({
    "id": test[ID_COL],
    "accident_risk": meta_test
})

submission.to_csv("submission.csv", index=False)
submission.head()

