
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from xgboost import XGBClassifier









train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

TARGET = "loan_paid_back"
ID_COL = "id"

train[TARGET] = train[TARGET].astype(int)


def encode_grade_subgrade(s):
    if pd.isnull(s):
        return np.nan
    s = str(s).upper().strip()
    if len(s) < 2:
        return np.nan
    letter = s[0]
    digit = s[1]
    try:
        return (ord(letter) - ord('A')) * 5 + (int(digit) - 1)
    except:
        return np.nan

train["grade_subgrade_num"] = train["grade_subgrade"].apply(encode_grade_subgrade)
test["grade_subgrade_num"]  = test["grade_subgrade"].apply(encode_grade_subgrade)





num_features = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "grade_subgrade_num"
]

cat_features = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose"
]



num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])


X = train[num_features + cat_features]
y = train[TARGET]
X_test = test[num_features + cat_features]




clf_lr = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=2000))
])

oof_lr = np.zeros(len(y))
test_lr = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Logistic Regression ===")
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    clf_lr.fit(X.iloc[tr], y.iloc[tr])
    oof_lr[val] = clf_lr.predict_proba(X.iloc[val])[:, 1]
    test_lr += clf_lr.predict_proba(X_test)[:, 1] / skf.n_splits
    print(f"Fold {fold} AUC = {roc_auc_score(y.iloc[val], oof_lr[val]):.5f}")

print("LR OOF AUC:", roc_auc_score(y, oof_lr))



X_train_full = preprocessor.fit_transform(X)
X_test_full = preprocessor.transform(X_test)

if hasattr(X_train_full, "toarray"):
    X_train_full = X_train_full.toarray()
if hasattr(X_test_full, "toarray"):
    X_test_full = X_test_full.toarray()

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbosity": -1
}

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(test))

print("\n=== LightGBM ===")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y), 1):
    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    tr_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        lgb_params,
        tr_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            early_stopping(100),
            log_evaluation(200)
        ]
    )
    
    oof_lgb[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_lgb += model.predict(X_test_full, num_iteration=model.best_iteration) / skf.n_splits
    
    print(f"Fold {fold} AUC = {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}")

print("LGB OOF AUC:", roc_auc_score(y, oof_lgb))



oof_xgb = np.zeros(len(y))
test_xgb = np.zeros(len(test))

xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist"
)

print("\n=== XGBoost ===")
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    xgb_model.fit(
        preprocessor.fit_transform(X.iloc[tr]),
        y.iloc[tr],
        eval_set=[(preprocessor.transform(X.iloc[val]), y.iloc[val])],
        early_stopping_rounds=100,
        verbose=False
    )
    
    oof_xgb[val] = xgb_model.predict_proba(preprocessor.transform(X.iloc[val]))[:, 1]
    test_xgb += xgb_model.predict_proba(preprocessor.transform(X_test))[:, 1] / skf.n_splits
    
    print(f"Fold {fold} AUC = {roc_auc_score(y.iloc[val], oof_xgb[val]):.5f}")

print("XGB OOF AUC:", roc_auc_score(y, oof_xgb))



w_lr = 0.15
w_lgb = 0.50
w_xgb = 0.35

oof_ensemble = w_lr*oof_lr + w_lgb*oof_lgb + w_xgb*oof_xgb
test_ensemble = w_lr*test_lr + w_lgb*test_lgb + w_xgb*test_xgb

print("\n=== FINAL ENSEMBLE ===")
print("Ensemble OOF AUC:", roc_auc_score(y, oof_ensemble))





submission = pd.DataFrame({
    "id": test[ID_COL],
    "loan_paid_back": test_ensemble.clip(0, 1)
})

submission.to_csv("submission.csv", index=False)
submission

