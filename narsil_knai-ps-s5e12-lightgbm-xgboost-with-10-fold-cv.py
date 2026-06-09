# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]


y.mean()


# --- Simple categorical encoding ---
cat_cols = X.select_dtypes(include="object").columns.tolist()


cat_cols


X


# for col in cat_cols:
#     X[col] = X[col].astype("category").cat.codes
#     test[col] = test[col].astype("category").cat.codes


for col in cat_cols:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

    # Align test categories to X
    test[col] = test[col].cat.set_categories(X[col].cat.categories)

    # Now encode
    X[col] = X[col].cat.codes
    test[col] = test[col].cat.codes


X


# --- CV Setup ---
n_splits_ = 10

skf = StratifiedKFold(n_splits=n_splits_, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
#oof_xgb = np.zeros(len(X))
pred_lgb = np.zeros(len(test))
#pred_xgb = np.zeros(len(test))


# --- LightGBM ---
lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    #learning_rate=0.1,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=1000,
    #n_estimators=500,
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    print('fold: ', fold)
    print('train indices:', trn_idx)
    print('val indices:', val_idx)
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)
    oof_lgb[val_idx] = model.predict_proba(X_val)[:,1]
    print('liczba wierszy w zbiorze walidacyjnym w obecnym foldzie (out of fold predicions, oof)', X_val.shape[0])
    pred_lgb += model.predict_proba(test)[:,1] / n_splits_
    print('liczba wierszy w zbiorze testowym', test.shape[0])

print("LGB OOF AUC:", roc_auc_score(y, oof_lgb))


#Wyniki na 10 foldach (splitach)
#AUC = 0.7246 - 100 drzew
#AUC = 0.7261 - 500 drzew


from itertools import product


param_grid = {
    "learning_rate": [0.03, 0.1],
    "num_leaves": [63],
    "feature_fraction": [0.8],
    "bagging_fraction": [0.8],
    "bagging_freq": [1],
    "n_estimators": [500, 1000],
}

# Base params (fixed across experiments)
base_params = dict(
    objective="binary",
    metric="auc",
    # add other fixed params here if you want, e.g.:
    # random_state=42,
    # n_jobs=-1,
)

# ------------------------------------------------------------
# 2) Cartesian product -> experiments DataFrame
# ------------------------------------------------------------
keys = list(param_grid.keys())
values = [param_grid[k] for k in keys]
experiments = pd.DataFrame(list(product(*values)), columns=keys)

# Add tracking fields (these will be updated per experiment)
experiments["oof_auc"] = np.nan
experiments["status"] = "pending"


experiments


#opcjonalnie: Optuna https://optuna.readthedocs.io/en/v2.0.0/reference/generated/optuna.integration.lightgbm.LightGBMTuner.html


for exp_idx, exp_row in experiments.iterrows():
    print(f"\n=== Experiment {exp_idx+1}/{len(experiments)} ===")
    print(exp_row[keys].to_dict())

    # Build params for this experiment
    lgb_params = base_params.copy()
    lgb_params.update(exp_row[keys].to_dict())

    # Fresh arrays per experiment
    oof_lgb = np.zeros(len(X), dtype=float)
    pred_lgb = np.zeros(len(test), dtype=float)

    try:
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
            print("FOLD:", fold)
            X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr, y_tr)

            oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
            pred_lgb += model.predict_proba(test)[:, 1] / n_splits_

        oof_auc = roc_auc_score(y, oof_lgb)
        experiments.loc[exp_idx, "oof_auc"] = oof_auc
        experiments.loc[exp_idx, "status"] = "done"

        print("OOF AUC:", oof_auc)

    except Exception as e:
        experiments.loc[exp_idx, "status"] = f"failed: {type(e).__name__}"
        print("FAILED:", e)


# ------------------------------------------------------------
# 4) Pick the best row (highest OOF AUC)
# ------------------------------------------------------------
best_idx = experiments["oof_auc"].idxmax()
best_row = experiments.loc[best_idx]

print("\n====================")
print("BEST EXPERIMENT")
print("====================")
print("Index:", best_idx)
print("Best OOF AUC:", best_row["oof_auc"])
print("Best params:", best_row[keys].to_dict())

# If you want a sorted leaderboard of experiments:
experiments_sorted = experiments.sort_values("oof_auc", ascending=False)
display(experiments_sorted.head(20))


# # --- XGBoost ---
# xgb_model = xgb.XGBClassifier(
#     n_estimators=2000,
#     max_depth=5,
#     learning_rate=0.03,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric="auc",
#     random_state=42
# )

# for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
#     X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

#     xgb_model.fit(X_tr, y_tr)
#     oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:,1]
#     pred_xgb += xgb_model.predict_proba(test)[:,1] / 10

# print("XGB OOF AUC:", roc_auc_score(y, oof_xgb))


# --- Final Ensemble ---
#test_pred = 0.5 * pred_lgb + 0.5 * pred_xgb
test_pred = 1.0 * pred_lgb

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")








