# !pip install pandas numpy matplotlib lightgbm xgboost catboost scikit-learn


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

train.head(), train.shape, test.shape


TARGET = "diagnosed_diabetes"
ID = "id"

X = train.drop([TARGET, ID], axis=1)
y = train[TARGET]
X_test = test.drop(ID, axis=1)


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# # =========================================================
# # SETUP
# # =========================================================
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import Ridge

# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostClassifier

# from category_encoders import TargetEncoder

# SEED = 42
# N_FOLDS = 5

# # =========================================================
# # LOAD DATA
# # =========================================================
# train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
# test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# TARGET = "diagnosed_diabetes"
# ID = "id"

# cat_cols = [
#     'gender', 'ethnicity', 'education_level',
#     'income_level', 'smoking_status', 'employment_status'
# ]

# X = train.drop([TARGET, ID], axis=1)
# y = train[TARGET]
# X_test = test.drop(ID, axis=1)

# print("Train:", train.shape, "Test:", test.shape)

# # =========================================================
# # CATBOOST DATA (RAW CATEGORICALS)
# # =========================================================
# X_cat = X.copy()
# X_test_cat = X_test.copy()

# # =========================================================
# # TARGET ENCODING FOR LGB + XGB (NO LEAKAGE)
# # =========================================================
# te = TargetEncoder(cols=cat_cols, smoothing=10)

# X_te = X.copy()
# X_test_te = X_test.copy()

# X_te[cat_cols] = te.fit_transform(X[cat_cols], y)
# X_test_te[cat_cols] = te.transform(X_test[cat_cols])

# # =========================================================
# # STRATIFIED FOLDS
# # =========================================================
# skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# oof_lgb = np.zeros(len(X))
# oof_xgb = np.zeros(len(X))
# oof_cat = np.zeros(len(X))

# pred_lgb = np.zeros(len(X_test))
# pred_xgb = np.zeros(len(X_test))
# pred_cat = np.zeros(len(X_test))

# # =========================================================
# # LIGHTGBM
# # =========================================================
# print("\n=== LightGBM ===")

# params_lgb = {
#     "objective": "binary",
#     "metric": "auc",
#     "learning_rate": 0.01,
#     "num_leaves": 64,
#     "min_data_in_leaf": 30,
#     "feature_fraction": 0.85,
#     "bagging_fraction": 0.85,
#     "bagging_freq": 1,
#     "lambda_l1": 0.1,
#     "lambda_l2": 0.3,
#     "verbosity": -1,
#     "seed": SEED
# }

# for fold, (tr, val) in enumerate(skf.split(X_te, y), 1):
#     print(f"Fold {fold}")

#     model = lgb.LGBMClassifier(**params_lgb, n_estimators=8000)

#     model.fit(
#         X_te.iloc[tr], y.iloc[tr],
#         eval_set=[(X_te.iloc[val], y.iloc[val])],
#         callbacks=[
#             lgb.early_stopping(120),
#             lgb.log_evaluation(300)
#         ]
#     )

#     oof_lgb[val] = model.predict_proba(X_te.iloc[val])[:, 1]
#     pred_lgb += model.predict_proba(X_test_te)[:, 1] / N_FOLDS

# print("LGB AUC:", roc_auc_score(y, oof_lgb))

# # =========================================================
# # CATBOOST (GPU)
# # =========================================================
# print("\n=== CatBoost ===")

# for fold, (tr, val) in enumerate(skf.split(X_cat, y), 1):
#     print(f"Fold {fold}")

#     model = CatBoostClassifier(
#         iterations=3500,
#         learning_rate=0.02,
#         depth=8,
#         l2_leaf_reg=6,
#         random_strength=1.5,
#         bagging_temperature=0.7,
#         loss_function="Logloss",
#         eval_metric="AUC",
#         task_type="GPU",
#         devices="0",
#         random_seed=SEED,
#         verbose=300
#     )

#     model.fit(
#         X_cat.iloc[tr], y.iloc[tr],
#         eval_set=(X_cat.iloc[val], y.iloc[val]),
#         cat_features=cat_cols,
#         use_best_model=True
#     )

#     oof_cat[val] = model.predict_proba(X_cat.iloc[val])[:, 1]
#     pred_cat += model.predict_proba(X_test_cat)[:, 1] / N_FOLDS

# print("Cat AUC:", roc_auc_score(y, oof_cat))

# # =========================================================
# # XGBOOST (GPU)
# # =========================================================
# print("\n=== XGBoost ===")

# params_xgb = {
#     "objective": "binary:logistic",
#     "eval_metric": "auc",
#     "eta": 0.015,
#     "max_depth": 7,
#     "min_child_weight": 3,
#     "subsample": 0.85,
#     "colsample_bytree": 0.85,
#     "gamma": 0.1,
#     "lambda": 1.5,
#     "alpha": 0.2,
#     "tree_method": "gpu_hist",
#     "seed": SEED
# }

# for fold, (tr, val) in enumerate(skf.split(X_te, y), 1):
#     print(f"Fold {fold}")

#     dtrain = xgb.DMatrix(X_te.iloc[tr], y.iloc[tr])
#     dval   = xgb.DMatrix(X_te.iloc[val], y.iloc[val])
#     dtest  = xgb.DMatrix(X_test_te)

#     model = xgb.train(
#         params_xgb,
#         dtrain,
#         num_boost_round=3500,
#         evals=[(dval, "valid")],
#         early_stopping_rounds=120,
#         verbose_eval=300
#     )

#     oof_xgb[val] = model.predict(dval)
#     pred_xgb += model.predict(dtest) / N_FOLDS

# print("XGB AUC:", roc_auc_score(y, oof_xgb))

# # =========================================================
# # STACKING (RIDGE)
# # =========================================================
# print("\n=== STACKING ===")

# stack_train = np.column_stack([oof_lgb, oof_cat, oof_xgb])
# stack_test  = np.column_stack([pred_lgb, pred_cat, pred_xgb])

# meta = Ridge(alpha=1.0)
# meta.fit(stack_train, y)

# final_pred = meta.predict(stack_test)

# # =========================================================
# # SUBMISSION
# # =========================================================
# submission = pd.DataFrame({
#     "id": test["id"],
#     "diagnosed_diabetes": final_pred
# })

# submission.to_csv("submission.csv", index=False)
# print("Saved submission.csv")



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

print("=== Load data ===")
print("train", train.shape, "test", test.shape)

# ---------------------------------------------------------
# 1. Identify categorical columns
# ---------------------------------------------------------
cat_cols = ['gender', 'ethnicity', 'education_level',
            'income_level', 'smoking_status', 'employment_status']

print("Categorical cols:", cat_cols)

# ---------------------------------------------------------
# 2. Copy data for CatBoost (raw categoricals)
# ---------------------------------------------------------
X_cat = train.drop(["diagnosed_diabetes"], axis=1).copy()
X_test_cat = test.copy()

# ---------------------------------------------------------
# 3. Encode only for LightGBM & XGBoost
# ---------------------------------------------------------
train_enc = train.copy()
test_enc = test.copy()

from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    train_enc[col] = le.fit_transform(train_enc[col])
    test_enc[col]  = le.transform(test_enc[col])

# Dataset for LGB/XGB
X = train_enc.drop(["diagnosed_diabetes"], axis=1)
y = train_enc["diagnosed_diabetes"]
X_test = test_enc.copy()

# ---------------------------------------------------------
# Setup folds
# ---------------------------------------------------------
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Storage
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))

pred_lgb = np.zeros(len(test))
pred_xgb = np.zeros(len(test))
pred_cat = np.zeros(len(test))

# ---------------------------------------------------------
# 4. LIGHTGBM (CPU)
# ---------------------------------------------------------
print("\n=== TRAINING LightGBM (CPU) ===\n")

params_lgb = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.015,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "verbosity": -1
}

from lightgbm import early_stopping, log_evaluation

fold = 1
for tr_idx, val_idx in skf.split(X, y):
    print(f"[START] LightGBM Fold {fold}")

    model = lgb.LGBMClassifier(**params_lgb, n_estimators=5000)

    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[
            early_stopping(stopping_rounds=80),
            log_evaluation(period=200)
        ]
    )

    oof_lgb[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    pred_lgb += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold += 1

print("LightGBM OOF AUC:", roc_auc_score(y, oof_lgb))

# ---------------------------------------------------------
# 5. CATBOOST (GPU — raw categoricals)
# ---------------------------------------------------------
print("\n=== TRAINING CatBoost (GPU) ===")

fold = 1
for tr_idx, val_idx in skf.split(X_cat, y):
    print(f"[START] CatBoost Fold {fold}")

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        task_type='GPU',
        devices='0',
        verbose=200,
        random_seed=42
    )

    model.fit(
        X_cat.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=(X_cat.iloc[val_idx], y.iloc[val_idx]),
        use_best_model=True,
        cat_features=cat_cols
    )

    oof_cat[val_idx] = model.predict_proba(X_cat.iloc[val_idx])[:, 1]
    pred_cat += model.predict_proba(X_test_cat)[:, 1] / N_FOLDS

    fold += 1

print("CatBoost OOF AUC:", roc_auc_score(y, oof_cat))

# ---------------------------------------------------------
# 6. XGBOOST (GPU)
# ---------------------------------------------------------
print("\n=== TRAINING XGBoost (GPU) ===")

params_xgb = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "gpu_hist",
}

fold = 1
for tr_idx, val_idx in skf.split(X, y):
    print(f"[START] XGB Fold {fold}")

    dtrain = xgb.DMatrix(X.iloc[tr_idx], y.iloc[tr_idx])
    dval   = xgb.DMatrix(X.iloc[val_idx], y.iloc[val_idx])
    dtest  = xgb.DMatrix(X_test)

    model = xgb.train(
        params_xgb,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "valid")],
        early_stopping_rounds=80,
        verbose_eval=200
    )

    oof_xgb[val_idx] = model.predict(dval)
    pred_xgb += model.predict(dtest) / N_FOLDS

    fold += 1

print("XGB OOF AUC:", roc_auc_score(y, oof_xgb))

# ---------------------------------------------------------
# 7. Stacking + Final prediction
# ---------------------------------------------------------
print("\n=== STACKING ===")

stack_train = np.vstack([oof_lgb, oof_cat, oof_xgb]).T
stack_test  = np.vstack([pred_lgb, pred_cat, pred_xgb]).T

from sklearn.linear_model import LogisticRegression

meta = LogisticRegression()
meta.fit(stack_train, y)

final_pred = meta.predict_proba(stack_test)[:, 1]

sub = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": final_pred
})

sub.to_csv("submission.csv", index=False)

print("\n=== COMPLETE ===")
print("Final submission saved → submission.csv")


