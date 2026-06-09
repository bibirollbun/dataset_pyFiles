import importlib, subprocess, sys

# Install LightGBM if needed
try:
    import lightgbm as lgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import skew

plt.style.use("seaborn-v0_8")
sns.set(font_scale=1.0)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

train.head()



print("\nInfo:")
print(train.info())

print("\nMissing values (% top 20):")
(train.isna().mean() * 100).sort_values(ascending=False).head(20)



target = "loan_paid_back"   # adjust only if competition uses a different name

train[target].value_counts(normalize=True)



fig, ax = plt.subplots(figsize=(5,4))
sns.countplot(x=target, data=train, ax=ax)
ax.set_title("Target Distribution")
ax.bar_label(ax.containers[0], fmt="%.0f")
plt.show()



numerical_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(train[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print("Histograms for numerical features displayed successfully.")


# Exclude ID-like columns from feature lists if present
id_cols = []
for col in train.columns:
    if "id" in col.lower():
        id_cols.append(col)

print("ID-like columns:", id_cols)

feature_cols = [c for c in train.columns if c not in id_cols + [target]]

num_cols = train[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = train[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric features:", len(num_cols))
print("Categorical features:", len(cat_cols))



missing = train[feature_cols].isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]

if not missing.empty:
    plt.figure(figsize=(8, min(0.4*len(missing), 10)))
    sns.barplot(x=missing.values*100, y=missing.index)
    plt.xlabel("Missing (%)")
    plt.title("Missing Values by Feature")
    plt.show()
else:
    print("No missing values in features.")



df_train = train.copy()
df_test  = test.copy()

# 1) Basic imputations
for col in num_cols:
    median_val = df_train[col].median()
    df_train[col].fillna(median_val, inplace=True)
    df_test[col].fillna(median_val, inplace=True)

for col in cat_cols:
    df_train[col].fillna("Missing", inplace=True)
    df_test[col].fillna("Missing", inplace=True)

# 2) Optional ratio features if columns exist
def safe_add_ratio(col_num, col_den, new_name):
    if col_num in df_train.columns and col_den in df_train.columns:
        df_train[new_name] = df_train[col_num] / (df_train[col_den] + 1e-3)
        df_test[new_name]  = df_test[col_num]  / (df_test[col_den]  + 1e-3)
        print(f"Created feature: {new_name}")

# These names may/may not exist in this playground dataset, so we guard them
safe_add_ratio("loan_amount", "annual_income", "loan_to_income")
safe_add_ratio("loan_amount", "monthly_income", "loan_to_monthly_income")
safe_add_ratio("current_balance", "credit_limit", "credit_util_ratio")
safe_add_ratio("annual_income", "age", "income_per_age")

# Update numeric feature list after engineered ratios
num_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [c for c in num_cols if c not in id_cols + [target]]

print("Numeric columns after feature engineering:", len(num_cols))



# Log-transform highly skewed positive numeric features
log_cols = []
for col in num_cols:
    if df_train[col].min() > -0.99:  # avoid weird negatives
        sk = skew(df_train[col])
        if abs(sk) > 1:
            new_col = col + "_log"
            df_train[new_col] = np.log1p(df_train[col])
            df_test[new_col]  = np.log1p(df_test[col])
            log_cols.append(new_col)

print("Log-transformed columns:", log_cols)

# Refresh lists
num_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in num_cols if c not in id_cols + [target]]

print("Final numeric:", len(num_cols))
print("Final categorical:", len(cat_cols))



# Combine for consistent one-hot encoding
train_features = df_train.drop(columns=[target])
test_features  = df_test.copy()

combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

X = combined_encoded.iloc[:len(train)]
X_test = combined_encoded.iloc[len(train):].reset_index(drop=True)

y = df_train[target].values

print("Encoded train shape:", X.shape)
print("Encoded test shape :", X_test.shape)



X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "n_estimators": 10000,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "subsample_freq": 1,
    "min_child_samples": 40,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

model = lgb.LGBMClassifier(**params)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=200),
    ],
)

valid_pred = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
print("Holdout ROC AUC:", roc_auc_score(y_valid, valid_pred))



NFOLDS = 5

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

feature_importances = pd.DataFrame()
feature_importances["feature"] = X.columns

models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold} / {NFOLDS} =====")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200),
        ],
    )

    val_pred = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
    oof_pred[val_idx] = val_pred

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

    test_fold_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
    test_pred += test_fold_pred / NFOLDS

    models.append(model)

    # collect feature importance
    feature_importances[f"fold_{fold}"] = model.booster_.feature_importance(
        importance_type="gain"
    )

# Overall OOF AUC
oof_auc = roc_auc_score(y, oof_pred)
print(f"\nOOF AUC: {oof_auc:.6f}")



feature_importances["avg_gain"] = feature_importances[[f"fold_{i}" for i in range(1, NFOLDS+1)]].mean(axis=1)
feat_imp = feature_importances.sort_values("avg_gain", ascending=False)

top_n = 40
plt.figure(figsize=(8, min(0.35*top_n, 12)))
sns.barplot(x="avg_gain", y="feature", data=feat_imp.head(top_n))
plt.title("Top Feature Importances (Average Gain)")
plt.tight_layout()
plt.show()






