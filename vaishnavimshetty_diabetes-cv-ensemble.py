# ============================================
# 1. Imports
# ============================================
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation, early_stopping

from xgboost import XGBClassifier

# ============================================
# 2. Load Data
# ============================================
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print(train.head())

# ============================================
# 3. Encode Categorical Columns
# ============================================
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)

    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ============================================
# 4. Define Features and Target
# ============================================
TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"

features = [col for col in train.columns if col not in [TARGET_COL, ID_COL]]

X = train[features]
y = train[TARGET_COL]
X_test = test[features]

print("Feature count:", len(features))

# ============================================
# 5. CV + Seeds (FAST)
# ============================================
SEEDS = [42, 2024]   # 2 seeds
N_FOLDS = 5          # 5 folds

# ============================================
# 6. LightGBM CV Training (FAST)
# ============================================
def train_lgbm_cv_fast(X, y, X_test, seeds, n_folds=5):
    oof_preds = np.zeros(len(X))
    test_preds_all_seeds = []
    auc_scores = []

    for seed in seeds:
        print(f"\n===== LGBM | SEED {seed} =====")
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed
        )

        test_preds_this_seed = np.zeros(len(X_test))
        oof_seed = np.zeros(len(X))

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n--- LGBM | Seed {seed} | Fold {fold}/{n_folds} ---")

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.02,
                objective="binary",
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=-1,
                num_leaves=64,
                random_state=seed,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                callbacks=[
                    log_evaluation(200),
                    early_stopping(200)
                ]
            )

            valid_pred = model.predict_proba(X_valid)[:, 1]
            oof_seed[valid_idx] = valid_pred

            fold_auc = roc_auc_score(y_valid, valid_pred)
            auc_scores.append(fold_auc)
            print(f"LGBM | Seed {seed} | Fold {fold} AUC: {fold_auc:.5f}")

            test_pred = model.predict_proba(X_test)[:, 1]
            test_preds_this_seed += test_pred / n_folds

        seed_auc = roc_auc_score(y, oof_seed)
        print(f"\nLGBM | Seed {seed} | OOF AUC: {seed_auc:.5f}")

        oof_preds += oof_seed / len(seeds)
        test_preds_all_seeds.append(test_preds_this_seed)

    test_preds_mean = np.mean(test_preds_all_seeds, axis=0)

    overall_auc = roc_auc_score(y, oof_preds)
    print("\n=======================================")
    print(f"LGBM | Overall OOF AUC (all seeds blended): {overall_auc:.5f}")
    print("=======================================")

    return oof_preds, test_preds_mean, auc_scores

# ============================================
# 7. XGBoost CV Training (FAST)
# ============================================
def train_xgb_cv_fast(X, y, X_test, seeds, n_folds=5):
    oof_preds = np.zeros(len(X))
    test_preds_all_seeds = []
    auc_scores = []

    for seed in seeds:
        print(f"\n===== XGB | SEED {seed} =====")
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed
        )

        test_preds_this_seed = np.zeros(len(X_test))
        oof_seed = np.zeros(len(X))

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n--- XGB | Seed {seed} | Fold {fold}/{n_folds} ---")

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = XGBClassifier(
                n_estimators=2000,
                learning_rate=0.02,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
                use_label_encoder=False
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=200,
                early_stopping_rounds=200
            )

            valid_pred = model.predict_proba(X_valid)[:, 1]
            oof_seed[valid_idx] = valid_pred

            fold_auc = roc_auc_score(y_valid, valid_pred)
            auc_scores.append(fold_auc)
            print(f"XGB | Seed {seed} | Fold {fold} AUC: {fold_auc:.5f}")

            test_pred = model.predict_proba(X_test)[:, 1]
            test_preds_this_seed += test_pred / n_folds

        seed_auc = roc_auc_score(y, oof_seed)
        print(f"\nXGB | Seed {seed} | OOF AUC: {seed_auc:.5f}")

        oof_preds += oof_seed / len(seeds)
        test_preds_all_seeds.append(test_preds_this_seed)

    test_preds_mean = np.mean(test_preds_all_seeds, axis=0)

    overall_auc = roc_auc_score(y, oof_preds)
    print("\n=======================================")
    print(f"XGB | Overall OOF AUC (all seeds blended): {overall_auc:.5f}")
    print("=======================================")

    return oof_preds, test_preds_mean, auc_scores

# ============================================
# 8. Run LGBM and XGBoost
# ============================================
lgbm_oof, lgbm_test, lgbm_aucs = train_lgbm_cv_fast(X, y, X_test, SEEDS, N_FOLDS)
xgb_oof,  xgb_test,  xgb_aucs  = train_xgb_cv_fast(X, y, X_test, SEEDS, N_FOLDS)

print("\nLGBM mean AUC:", np.mean(lgbm_aucs))
print("XGB mean AUC :", np.mean(xgb_aucs))

# ============================================
# 9. Blend Predictions (LGBM + XGB)
# ============================================
# You can tweak these weights: try 0.5/0.5, 0.6/0.4, 0.7/0.3, etc.
w_lgbm = 0.6
w_xgb  = 0.4

blend_test = w_lgbm * lgbm_test + w_xgb * xgb_test

blend_oof = w_lgbm * lgbm_oof + w_xgb * xgb_oof
blend_auc = roc_auc_score(y, blend_oof)
print(f"\nBlended OOF AUC (LGBM {w_lgbm} + XGB {w_xgb}): {blend_auc:.5f}")




# ============================================
# 10. Create Submission from Blended Predictions
# ============================================
submission = sample_submission.copy()
submission["diagnosed_diabetes"] = blend_test

submission_file_name = "submission_lgbm_xgb_blend.csv"
submission.to_csv(submission_file_name, index=False)

print("\nSaved submission as:", submission_file_name)
print(submission.head())

