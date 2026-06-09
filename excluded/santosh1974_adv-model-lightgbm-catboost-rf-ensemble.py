# ------------------------------
# 1. Imports
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



# ------------------------------
# 2. Load Data
# ------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# ------------------------------
# 3. Identify Features
# ------------------------------
target = "y"
id_col = "id"

cat_features = train.select_dtypes(include=['object']).columns.tolist()
num_features = train.select_dtypes(exclude=['object']).columns.tolist()
if target in num_features:
    num_features.remove(target)
if id_col in num_features:
    num_features.remove(id_col)

print(f"Categorical features: {cat_features}")
print(f"Numerical features: {num_features}")




# ------------------------------
# 4. Prepare Data
# ------------------------------
X = train.drop([target, id_col], axis=1)
y = train[target]
X_test = test.drop([id_col], axis=1)




# ------------------------------
# 5. Cross-validation Setup
# ------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for ensembling
oof_preds_lgb = np.zeros(len(X))
oof_preds_cb = np.zeros(len(X))
oof_preds_rf = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))
test_preds_rf = np.zeros(len(X_test))




# ------------------------------
# 6. Models
# ------------------------------
lgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "objective": "binary",
    "random_state": 42
}

cb_params = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 0
}

rf_params = {
    "n_estimators": 300,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1
}




# ---- Make categoricals proper 'category' dtype for LightGBM ----
cat_features = train.select_dtypes(include=['object']).columns.tolist()

X = train.drop([target, id_col], axis=1)
y = train[target]
X_test = test.drop([id_col], axis=1)

for col in cat_features:
    # Build a unified set of categories across train+test
    all_vals = pd.concat([X[col], X_test[col]], axis=0).astype('category')
    cats = all_vals.cat.categories
    X[col] = pd.Categorical(X[col], categories=cats)
    X_test[col] = pd.Categorical(X_test[col], categories=cats)



from lightgbm import early_stopping, log_evaluation

from sklearn.preprocessing import OneHotEncoder

# ------------------------------
# 7. Cross-validation Loop  (RF part fixed)
# ------------------------------
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"FOLD {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # -------- LightGBM (unchanged) --------
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(50), log_evaluation(0)]
    )
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # -------- CatBoost (unchanged) --------
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        verbose=False
    )
    oof_preds_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]
    test_preds_cb += cb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # -------- RandomForest (fixed: one-hot encode cats) --------
    # Identify numeric columns (everything except categorical ones)
    num_features = [c for c in X.columns if c not in cat_features]

    # Fit OHE on TRAIN fold only, ignore unknowns in VAL/TEST
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    ohe.fit(X_train[cat_features])

    # Build RF matrices
    Xtr_rf = np.hstack([
        X_train[num_features].to_numpy(),
        ohe.transform(X_train[cat_features])
    ])
    Xval_rf = np.hstack([
        X_val[num_features].to_numpy(),
        ohe.transform(X_val[cat_features])
    ])
    Xtest_rf = np.hstack([
        X_test[num_features].to_numpy(),
        ohe.transform(X_test[cat_features])
    ])

    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(Xtr_rf, y_train)
    oof_preds_rf[val_idx] = rf_model.predict_proba(Xval_rf)[:, 1]
    test_preds_rf += rf_model.predict_proba(Xtest_rf)[:, 1] / skf.n_splits




# ------------------------------
# 8. Evaluate Models
# ------------------------------
print("LightGBM CV AUC:", roc_auc_score(y, oof_preds_lgb))
print("CatBoost CV AUC:", roc_auc_score(y, oof_preds_cb))
print("RandomForest CV AUC:", roc_auc_score(y, oof_preds_rf))




# ------------------------------
# 9. Ensemble Predictions
# ------------------------------
oof_preds_ensemble = (oof_preds_lgb + oof_preds_cb + oof_preds_rf) / 3
test_preds_ensemble = (test_preds_lgb + test_preds_cb + test_preds_rf) / 3

print("Ensemble CV AUC:", roc_auc_score(y, oof_preds_ensemble))




# ------------------------------
# 10. Submission
# ------------------------------
submission = pd.DataFrame({
    id_col: test[id_col],
    target: test_preds_ensemble
})
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")




