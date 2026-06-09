#Bhakti Vyas
#Predicting loan payback 


# import libraries



import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
FOLDS = 5
np.random.seed(SEED)


INPUT_DIR = "/kaggle/input/playground-series-s5e11"
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

TARGET = train.columns[-1]
X_raw = train.drop(columns=[TARGET])
y = train[TARGET].astype(int)
test_raw = test.copy()

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain head:")
print(train.head())


def add_domain_features(df):
    df = df.copy()
    if "applicant_income" in df.columns and "coapplicant_income" in df.columns:
        df["total_income"] = df["applicant_income"].fillna(0) + df["coapplicant_income"].fillna(0)
    if "loan_amount" in df.columns and "total_income" in df.columns:
        df["loan_income_ratio"] = df["loan_amount"] / (df["total_income"] + 1e-6)
    if "credit_score" in df.columns:
        df["credit_bucket"] = pd.cut(
            df["credit_score"],
            bins=[-np.inf, 580, 670, 740, 800, np.inf],
            labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"]
        )
    for c in ["applicant_income", "coapplicant_income", "total_income", "loan_amount"]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    return df

X_fe = add_domain_features(X_raw)
test_fe = add_domain_features(test_raw)

num_cols = X_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_fe.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric features:", len(num_cols))
print("Categorical features:", len(cat_cols))


plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols[:9], 1):
    plt.subplot(3, 3, i)
    sns.histplot(X_fe[col].dropna(), kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols[:9], 1):
    plt.subplot(3, 3, i)
    X_fe[col].value_counts().plot.bar(color="salmon")
    plt.title(f"Counts of {col}")
plt.tight_layout()
plt.show()


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

# Diagnostic check
preprocessor.fit(X_fe)
print("Transformed shape:", preprocessor.transform(X_fe.head()).shape)
print("First 10 features:", preprocessor.get_feature_names_out()[:10])


lgbm = lgb.LGBMClassifier(
    n_estimators=1500, learning_rate=0.03, num_leaves=63,
    subsample=0.9, colsample_bytree=0.8, random_state=SEED, n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=1500, learning_rate=0.03, max_depth=6,
    subsample=0.9, colsample_bytree=0.8, random_state=SEED,
    n_jobs=-1, eval_metric="auc"
)

cat = CatBoostClassifier(
    iterations=1500, learning_rate=0.03, depth=8,
    l2_leaf_reg=3.0, loss_function="Logloss", eval_metric="AUC",
    random_state=SEED, verbose=False
)

pipe_lgbm = Pipeline([("pre", preprocessor), ("clf", lgbm)])
pipe_xgb  = Pipeline([("pre", preprocessor), ("clf", xgb)])
pipe_cat  = Pipeline([("pre", preprocessor), ("clf", cat)])

print("Base models initialized successfully.")


skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_lgbm = np.zeros(len(X_fe))
oof_xgb  = np.zeros(len(X_fe))
oof_cat  = np.zeros(len(X_fe))

test_pred_lgbm = np.zeros((len(test_fe), FOLDS))
test_pred_xgb  = np.zeros((len(test_fe), FOLDS))
test_pred_cat  = np.zeros((len(test_fe), FOLDS))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_fe, y), 1):
    X_tr, X_va = X_fe.iloc[trn_idx], X_fe.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]

    pipe_lgbm.fit(X_tr, y_tr)
    oof_lgbm[val_idx] = pipe_lgbm.predict_proba(X_va)[:, 1]
    test_pred_lgbm[:, fold-1] = pipe_lgbm.predict_proba(test_fe)[:, 1]

    pipe_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = pipe_xgb.predict_proba(X_va)[:, 1]
    test_pred_xgb[:, fold-1] = pipe_xgb.predict_proba(test_fe)[:, 1]

    pipe_cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = pipe_cat.predict_proba(X_va)[:, 1]
    test_pred_cat[:, fold-1] = pipe_cat.predict_proba(test_fe)[:, 1]

    print(f"Fold {fold} AUCs — LGBM: {roc_auc_score(y_va, oof_lgbm[val_idx]):.5f}, "
          f"XGB: {roc_auc_score(y_va, oof_xgb[val_idx]):.5f}, "
          f"CAT: {roc_auc_score(y_va, oof_cat[val_idx]):.5f}")

print("OOF AUC — LGBM:", roc_auc_score(y, oof_lgbm))
print("OOF AUC — XGB :", roc_auc_score(y, oof_xgb))
print("OOF AUC — CAT :", roc_auc_score(y, oof_cat))


stack_train = pd.DataFrame({"lgbm": oof_lgbm, "xgb": oof_xgb, "cat": oof_cat})
stack_test = pd.DataFrame({
    "lgbm": test_pred_lgbm.mean(axis=1),
    "xgb": test_pred_xgb.mean(axis=1),
    "cat": test_pred_cat.mean(axis=1)
})

meta = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=SEED)
meta.fit(stack_train, y)

oof_stack = meta.predict_proba(stack_train)[:, 1]
stack_auc = roc_auc_score(y, oof_stack)
print(f"OOF AUC — Stacked: {stack_auc:.6f}")

# ROC curve
fpr, tpr, _ = roc_curve(y, oof_stack)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"Stacked AUC = {stack_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


final_preds = meta.predict_proba(stack_test)[:, 1]
final_preds = np.clip(final_preds, 0.001, 0.999)  # small clipping for stability

submission = sample_submission.copy()
submission[submission.columns[-1]] = final_preds
submission.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully!")
print(submission.head())

