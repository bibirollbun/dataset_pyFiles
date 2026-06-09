!pip install -U scikit-learn imbalanced-learn


# ============================================================
# ğŸ“˜ BLENDING MODEL: XGBoost + LightGBM 
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ğŸ“¥ LOAD DATA
# ============================================================
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

target_col = "loan_paid_back"

X = train.drop(columns=[target_col, "id"])
y = train[target_col].astype(float)

test_ids = test["id"]
X_test = test.drop(columns=["id"])


# ============================================================
# âœ¨ FIX SKEWED FEATURES
# ============================================================
X["annual_income"] = np.log1p(X["annual_income"])
X_test["annual_income"] = np.log1p(X_test["annual_income"])

X["debt_to_income_ratio"] = np.sqrt(X["debt_to_income_ratio"])
X_test["debt_to_income_ratio"] = np.sqrt(X_test["debt_to_income_ratio"])


# ============================================================
# ğŸ”¤ ENCODE CATEGORICAL
# ============================================================
cat_cols = X.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder()
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])


# ============================================================
# âœ‚ï¸� SPLIT DATA
# ============================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# âš–ï¸� SMOTE BALANCING
# ============================================================
sm = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_res.value_counts(normalize=True))


# ============================================================
# ğŸ”§ PROVIDED BEST PARAMS FOR XGB & LGBM
# ============================================================

xgb_params = {
    'n_estimators': 1780,
    'learning_rate': 0.020074350734455924,
    'max_depth': 12,
    'min_child_weight': 4,
    'subsample': 0.9011694206152446,
    'colsample_bytree': 0.6284973519533058,
    'gamma': 3.0258868756964996,
    'reg_alpha': 0.7757649369022213,
    'reg_lambda': 0.004554755722702163,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'tree_method': 'hist'
}

lgb_params = {
    "learning_rate": 0.046216594722431385,
    "num_leaves": 81,
    "max_depth": 4,
    "min_child_samples": 141,
    "subsample": 0.7706417252454062,
    "colsample_bytree": 0.714224110342554,
    "reg_alpha": 2.1924967649292544,
    "reg_lambda": 2.057985586700453,
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 2000,
    "random_state": 42,
    "boosting_type": "gbdt"
}


# ============================================================
# ğŸ�‹ï¸� TRAIN BOTH MODELS
# ============================================================

model_xgb = XGBClassifier(**xgb_params)
model_xgb.fit(X_train_res, y_train_res)


model_lgb = LGBMClassifier(**lgb_params)
model_lgb.fit(X_train_res, y_train_res)


# ============================================================
# ğŸ”® VALIDATION PREDICTIONS
# ============================================================
pred_xgb = model_xgb.predict_proba(X_valid)[:, 1]
pred_lgb = model_lgb.predict_proba(X_valid)[:, 1]

# Simple blend: average
blend_valid = (pred_xgb + pred_lgb) / 2

auc_xgb = roc_auc_score(y_valid, pred_xgb)
auc_lgb = roc_auc_score(y_valid, pred_lgb)
auc_blend = roc_auc_score(y_valid, blend_valid)

print(f"AUC XGB:   {auc_xgb:.5f}")
print(f"AUC LGBM:  {auc_lgb:.5f}")
print(f"AUC Blend: {auc_blend:.5f}  <-- ğŸ”¥ Best wins")


# ============================================================
# ğŸ�† TRAIN ON FULL DATA (with SMOTE)
# ============================================================
sm_full = SMOTE(sampling_strategy=0.5, random_state=42)
X_res, y_res = sm_full.fit_resample(X, y)

final_xgb = XGBClassifier(**xgb_params)
final_xgb.fit(X_res, y_res)

final_lgb = LGBMClassifier(**lgb_params)
final_lgb.fit(X_res, y_res)


# ============================================================
# ğŸ§ª TEST PREDICTIONS (BLEND)
# ============================================================
test_pred_xgb = final_xgb.predict_proba(X_test)[:, 1]
test_pred_lgb = final_lgb.predict_proba(X_test)[:, 1]

test_blend = (test_pred_xgb + test_pred_lgb) / 2


# ============================================================
# ğŸ“¤ SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": test_blend
})
submission.to_csv("submission.csv", index=False)

print("âœ… Submission saved as submission_blend.csv")


submission

