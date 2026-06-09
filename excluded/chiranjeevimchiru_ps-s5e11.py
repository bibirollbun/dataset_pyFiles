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


# ============================================================

# ============================================================
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

TARGET = "loan_paid_back"
ID_COL = "id"

print(train.shape, test.shape)

train[TARGET] = train[TARGET].astype(int)


def encode_grade_subgrade(s):
    if pd.isnull(s):
        return np.nan
    s = str(s).strip().upper()
    if len(s) < 2:
        return np.nan
    letter = s[0]
    digit = s[1]
    try:
        letter_rank = ord(letter) - ord('A')   # A→0, B→1, ..., G→6
        sub = int(digit) - 1                   # 1→0, ..., 5→4
        return letter_rank * 5 + sub           # 0–34 scale
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_scores = []
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    clf_lr.fit(X.iloc[tr], y.iloc[tr])
    preds = clf_lr.predict_proba(X.iloc[val])[:, 1]
    auc = roc_auc_score(y.iloc[val], preds)
    lr_scores.append(auc)
    print(f"LR Fold {fold} AUC: {auc:.5f}")

print("\nLogistic Regression Mean AUC:", np.mean(lr_scores))


X_train_full = preprocessor.fit_transform(X)
X_test_full = preprocessor.transform(X_test)

# convert to dense if needed
if hasattr(X_train_full, "toarray"):
    X_train_full = X_train_full.toarray()
if hasattr(X_test_full, "toarray"):
    X_test_full = X_test_full.toarray()



lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "seed": 42,
    "verbosity": -1
}



oof_preds = np.zeros(len(y))
test_preds = np.zeros(X_test_full.shape[0])

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y), 1):
    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    tr_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    print(f"\n---- LIGHTGBM FOLD {fold} ----")
    model = lgb.train(
        lgb_params,
        tr_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(100)
        ]
    )
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test_full, num_iteration=model.best_iteration) / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold} AUC = {fold_auc:.5f}")

print("\nOverall OOF AUC:", roc_auc_score(y, oof_preds))


submission = pd.DataFrame({
    "id": test[ID_COL],
    "loan_paid_back": test_preds.clip(0, 1)
})
submission
submission.to_csv("submission.csv", index=False)

