# ============================================
# 1. Imports
# ============================================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report,
    f1_score
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

# ============================================
# 2. Load Data
# ============================================

train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)
train.head()


# ============================================
# 3. Split target & combine for preprocessing
# ============================================

y = train["loan_paid_back"]
X = train.drop("loan_paid_back", axis=1)

combined = pd.concat([X, test], ignore_index=True)

num_cols = combined.select_dtypes(include=["int64", "float64"]).columns
cat_cols = combined.select_dtypes(include=["object"]).columns

print("Numeric cols:", list(num_cols))
print("Categorical cols:", list(cat_cols))


# ============================================
# 4. Missing value handling + Label Encoding
# ============================================

# numeric â†’ median
for col in num_cols:
    combined[col] = combined[col].fillna(combined[col].median())

# categorical â†’ mode + LabelEncode
le = LabelEncoder()
for col in cat_cols:
    combined[col] = combined[col].fillna(combined[col].mode()[0])
    combined[col] = le.fit_transform(combined[col])



# ============================================
# 5. Base Financial Feature Engineering
# ============================================

# EMI approx
combined["emi"] = combined["loan_amount"] * combined["interest_rate"] / 100

# Ratios
combined["income_to_emi_ratio"] = combined["annual_income"] / (combined["emi"] + 1)
combined["loan_to_income_ratio"] = combined["loan_amount"] / (combined["annual_income"] + 1)

# credit risk helper
combined["credit_risk"] = combined["credit_score"] / (combined["debt_to_income_ratio"] + 1)



# ============================================
# 6. Risk Buckets (qcut based)
# ============================================

for base_col, new_col in [
    ("credit_score", "credit_bucket"),
    ("annual_income", "income_bucket"),
    ("debt_to_income_ratio", "dti_bucket")
]:
    try:
        combined[new_col] = pd.qcut(
            combined[base_col],
            5,
            labels=False,
            duplicates="drop"
        )
    except Exception as e:
        print(f"Bucket creation failed for {base_col}: {e}")
        combined[new_col] = 0



# ============================================
# 7. SHAP-inspired Interaction Features
# (from your SHAP plot: employment_status,
#  credit_risk, credit_score, dti_bucket,
#  grade_subgrade etc.)
# ============================================

# NOTE: employment_status & grade_subgrade are label-encoded already.

combined["cs_emp_interaction"] = combined["credit_score"] * combined["employment_status"]
combined["risk_dti_ratio"] = combined["credit_risk"] / (combined["dti_bucket"] + 1)
combined["score_grade_combo"] = combined["credit_score"] * combined["grade_subgrade"]



# ============================================
# 8. Split combined back into train & test features
# ============================================

X_processed  = combined.iloc[: len(train)]
X_test_final = combined.iloc[len(train):]

print("X_processed:", X_processed.shape)
print("X_test_final:", X_test_final.shape)



# ============================================
# 9. Train / Validation Split + Imbalance ratio
# ============================================

X_train, X_val, y_train, y_val = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = pos / neg
print("neg:", neg, "pos:", pos, "scale_pos_weight:", scale)



# ============================================
# 10. LightGBM V2 (Optuna best params) â­�
# ============================================

best_params_lgb = {
    "n_estimators": 2425,
    "learning_rate": 0.04799514605781845,
    "max_depth": 4,
    "num_leaves": 35,
    "subsample": 0.9906130959928888,
    "colsample_bytree": 0.6077506330189334,
    "min_child_samples": 57,
    "reg_lambda": 4.798855044917759,
    "scale_pos_weight": scale,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 200
}

lgb_best = LGBMClassifier(**best_params_lgb, verbosity=-1)

lgb_best.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc"
)

lgb_val_proba = lgb_best.predict_proba(X_val)[:, 1]
lgb_val_pred  = (lgb_val_proba >= 0.5).astype(int)

print("LGBM V2 ROC-AUC:", roc_auc_score(y_val, lgb_val_proba))
print("\nLGBM V2 Classification Report:\n", classification_report(y_val, lgb_val_pred))



# ============================================
# 11. XGBoost V2
# ============================================

xgb_model_v2 = XGBClassifier(
    n_estimators=1200,
    max_depth=8,
    learning_rate=0.025,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.2,
    scale_pos_weight=scale,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model_v2.fit(X_train, y_train)
xgb_val_proba_v2 = xgb_model_v2.predict_proba(X_val)[:, 1]
xgb_val_pred_v2  = (xgb_val_proba_v2 >= 0.5).astype(int)

print("XGB V2 ROC-AUC:", roc_auc_score(y_val, xgb_val_proba_v2))
print("\nXGB V2 Classification Report:\n", classification_report(y_val, xgb_val_pred_v2))



# ============================================
# 12. CatBoost V2
# ============================================

cat_model_v2 = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.025,
    loss_function="Logloss",
    eval_metric="AUC",
    scale_pos_weight=scale,
    random_state=42,
    verbose=False  # silent
)

cat_model_v2.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_val_proba_v2 = cat_model_v2.predict_proba(X_val)[:, 1]
cat_val_pred_v2  = (cat_val_proba_v2 >= 0.5).astype(int)

print("CAT V2 ROC-AUC:", roc_auc_score(y_val, cat_val_proba_v2))
print("\nCAT V2 Classification Report:\n", classification_report(y_val, cat_val_pred_v2))



# ============================================
# 13. Ensemble Weight Search (Validation)
#     (LGB + XGB + CAT) â†’ choose best weights
# ============================================

print("LGBM V2:", roc_auc_score(y_val, lgb_val_proba))
print("XGB  V2:", roc_auc_score(y_val, xgb_val_proba_v2))
print("CAT  V2:", roc_auc_score(y_val, cat_val_proba_v2))

best_auc = 0
best_weights = (1.0, 0.0, 0.0)  # default

weights_range = np.arange(0.0, 1.01, 0.05)

for w_lgb in weights_range:
    for w_xgb in weights_range:
        w_cat = 1.0 - (w_lgb + w_xgb)
        if w_cat < 0:
            continue

        blend_val_proba = (
            w_lgb * lgb_val_proba +
            w_xgb * xgb_val_proba_v2 +
            w_cat * cat_val_proba_v2
        )

        auc = roc_auc_score(y_val, blend_val_proba)

        if auc > best_auc:
            best_auc = auc
            best_weights = (w_lgb, w_xgb, w_cat)

print("Best Ensemble ROC-AUC:", best_auc)
print("Best Weights (LGB, XGB, CAT):", best_weights)



# Small manual weight search around safe region
cand_weights = [
    (0.98, 0.02, 0.00),
    (0.97, 0.01, 0.02),
    (0.96, 0.02, 0.02),
    (0.95, 0.03, 0.02),
]

print("Validation AUC for candidate blends:\n")
for (w_lgb, w_xgb, w_cat) in cand_weights:
    blend_val = (
        w_lgb * lgb_val_proba +
        w_xgb * xgb_val_proba_v2 +
        w_cat * cat_val_proba_v2
    )
    auc = roc_auc_score(y_val, blend_val)
    print(f"Weights (LGB={w_lgb}, XGB={w_xgb}, CAT={w_cat}) -> AUC = {auc:.8f}")



from sklearn.model_selection import StratifiedKFold

# 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X_processed))          # out-of-fold preds for full train
test_pred_folds = np.zeros((5, len(X_test_final)))  # per-fold test preds

fold = 0
for train_idx, val_idx in skf.split(X_processed, y):
    fold += 1
    print(f"\n===== Fold {fold} =====")
    
    X_tr, X_vl = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
    
    params_fold = best_params_lgb.copy()
    
    model = LGBMClassifier(**params_fold, verbosity=-1)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        eval_metric="auc"
    )
    
    # Validation predictions for this fold
    val_proba = model.predict_proba(X_vl)[:, 1]
    oof_pred[val_idx] = val_proba
    fold_auc = roc_auc_score(y_vl, val_proba)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")
    
    # Test predictions for this fold
    test_pred_folds[fold - 1, :] = model.predict_proba(X_test_final)[:, 1]

# CV AUC (OOF) on full train
cv_auc = roc_auc_score(y, oof_pred)
print("\n===== CV LGBM (5-Fold) AUC on full train =====")
print(f"CV AUC: {cv_auc:.6f}")



# Average test probs over folds
test_pred_mean = test_pred_folds.mean(axis=0)

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_pred_mean
})

submission.to_csv("submission.csv", index=False)
print("Created file: submission.csv")
submission.head()



from sklearn.model_selection import StratifiedKFold

# Copy best params
params_seed2025 = best_params_lgb.copy()
params_seed2025["random_state"] = 2025   # NEW seed
params_seed2025["early_stopping_rounds"] = 200

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

oof_2025 = np.zeros(len(X_processed))
test_pred_2025 = np.zeros((5, len(X_test_final)))

fold = 0
for tr_idx, vl_idx in skf.split(X_processed, y):
    fold += 1
    print(f"\n===== Fold {fold} (seed 2025) =====")
    
    X_tr, X_vl = X_processed.iloc[tr_idx], X_processed.iloc[vl_idx]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[vl_idx]
    
    model = LGBMClassifier(**params_seed2025, verbosity=-1)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        eval_metric="auc"
    )
    
    oof_pred_fold = model.predict_proba(X_vl)[:, 1]
    oof_2025[vl_idx] = oof_pred_fold
    print(f"Fold AUC (seed 2025): {roc_auc_score(y_vl, oof_pred_fold):.6f}")
    
    test_pred_2025[fold-1] = model.predict_proba(X_test_final)[:, 1]

# CV AUC
auc_2025 = roc_auc_score(y, oof_2025)
print("\n===== LGBM CV AUC (seed 2025) =====")
print(auc_2025)

# Final test average
test_pred_mean_2025 = test_pred_2025.mean(axis=0)

# Submission
submission_lgb_cv_seed2025 = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_pred_mean_2025
})

submission_lgb_cv_seed2025.to_csv("submission_lgb_cv_seed2025.csv", index=False)
print("\nCreated submission_lgb_cv_seed2025.csv")
submission_lgb_cv_seed2025.head(10)



params_seed7 = best_params_lgb.copy()
params_seed7["random_state"] = 7
params_seed7["early_stopping_rounds"] = 200

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

oof_7 = np.zeros(len(X_processed))
test_pred_7 = np.zeros((5, len(X_test_final)))

fold = 0
for tr_idx, vl_idx in skf.split(X_processed, y):
    fold += 1
    print(f"\n===== Fold {fold} (seed 7) =====")
    
    X_tr, X_vl = X_processed.iloc[tr_idx], X_processed.iloc[vl_idx]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[vl_idx]
    
    model = LGBMClassifier(**params_seed7,verbosity=-1)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        eval_metric="auc"
    )
    
    oof_pred_fold = model.predict_proba(X_vl)[:, 1]
    oof_7[vl_idx] = oof_pred_fold
    print(f"Fold AUC (seed 7): {roc_auc_score(y_vl, oof_pred_fold):.6f}")
    
    test_pred_7[fold-1] = model.predict_proba(X_test_final)[:, 1]

auc_7 = roc_auc_score(y, oof_7)
print("\n===== LGBM CV AUC (seed 7) =====")
print(auc_7)

test_pred_mean_7 = test_pred_7.mean(axis=0)

sub7 = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_pred_mean_7
})

submission.to_csv("submission.csv", index=False)
print("\nCreated submission.csv")



import warnings
warnings.filterwarnings("ignore")

drop_cols_2 = ["grade_subgrade", "education_level"]   # TWO weak features (you can change if needed)

X_drop2 = X_processed.drop(columns=drop_cols_2)
X_test_drop2 = X_test_final.drop(columns=drop_cols_2)

params_drop2 = best_params_lgb.copy()
params_drop2["random_state"] = 2025
params_drop2["early_stopping_rounds"] = 200

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

oof_drop2 = np.zeros(len(X_drop2))
test_pred_drop2 = np.zeros((5, len(X_test_drop2)))

fold = 0
for tr_idx, vl_idx in skf.split(X_drop2, y):
    fold += 1
    print(f"\n===== Fold {fold} (Drop2) =====")
    
    X_tr, X_vl = X_drop2.iloc[tr_idx], X_drop2.iloc[vl_idx]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[vl_idx]
    
    model = LGBMClassifier(
        **params_drop2,
        verbosity=-1     # NO WARNINGS
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        eval_metric="auc"
    )
    
    oof_pred_fold = model.predict_proba(X_vl)[:, 1]
    oof_drop2[vl_idx] = oof_pred_fold
    print(f"Fold AUC (Drop2): {roc_auc_score(y_vl, oof_pred_fold):.6f}")
    
    test_pred_drop2[fold-1] = model.predict_proba(X_test_drop2)[:, 1]

auc_drop2 = roc_auc_score(y, oof_drop2)
print("\n===== LGBM CV AUC (Drop2) =====")
print(auc_drop2)

test_pred_mean_drop2 = test_pred_drop2.mean(axis=0)

sub_drop2 = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_pred_mean_drop2
})

submission.to_csv("submission.csv", index=False)
print("\nCreated submission.csv")





