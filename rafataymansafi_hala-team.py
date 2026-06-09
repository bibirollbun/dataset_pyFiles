# ====================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# ØªØ¬Ø§Ù‡Ù„ Ø§Ù„ØªØ­Ø°ÙŠØ±Ø§Øª
warnings.filterwarnings("ignore")

# ØªØ¹Ø¯ÙŠÙ„ Ø·Ø±ÙŠÙ‚Ø© Ø¹Ø±Ø¶ Ø§Ù„Ø£Ø±Ù‚Ø§Ù… Ù�ÙŠ Pandas (Ø§Ø®ØªÙŠØ§Ø±ÙŠ Ù„Ù„ØªÙˆØ¶ÙŠØ­)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")


# ====================================================
train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("ğŸ”¹ Preview:")
display(train.head())

print("\nğŸ”¹ Info:")
print(train.info())

print("\nğŸ”¹ Basic statistics:")
display(train.describe())


# ====================================================
# Target distribution
sns.countplot(x="target", data=train)
plt.title("Target Distribution (0 vs 1)")
plt.show()
time_cols = ['day', 'month']
for c in time_cols:
    sns.countplot(x=c, data=train)
    plt.title(f"Distribution of {c} vs target")
    plt.show()



# ====================================================
TARGET = "target"
ID_COL = "id"

y = train[TARGET]
X = train.drop([TARGET, ID_COL], axis=1)
X_test = test.drop([ID_COL], axis=1)

# Identify categorical columns
def is_categorical(col):
    c = col.lower()
    return (
        c.startswith("bin_") or
        c.startswith("nom_") or
        c.startswith("ord_") or
        c in ["day", "month"]
    )

cat_cols = [c for c in X.columns if is_categorical(c)]
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Categorical cols: {len(cat_cols)} | Numeric cols: {len(num_cols)}")


# ====================================================
if num_cols:
    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# âœ… Fix: CatBoost requires categorical features as str/int
for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

cat_features_idx = [X.columns.get_loc(c) for c in cat_cols]


from catboost import cv, Pool
from catboost import CatBoostClassifier

N_FOLDS = 5
SEED = 42

params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 700,              # Ø¹Ø¯Ø¯ Ø£Ù‚Ù„ = Ø£Ø³Ø±Ø¹
    "learning_rate": 0.07,          # Ø£Ø¹Ù„Ù‰ Ù‚Ù„ÙŠÙ„Ø§Ù‹ Ù„ØªØ¹ÙˆÙŠØ¶ ØªÙ‚Ù„ÙŠÙ„ iterations
    "depth": 5,                     # Ø¹Ù…Ù‚ Ø£Ù‚Ù„ â†’ Ø£Ø³Ø±Ø¹
    "l2_leaf_reg": 8,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "random_seed": SEED,
    "task_type": "CPU",             # ØºÙŠÙ‘Ø±Ù‡Ø§ "GPU" Ø¥Ø°Ø§ Ù…ØªÙˆÙ�Ø± Ø¹Ù†Ø¯Ùƒ
    "verbose": 100,
    "early_stopping_rounds": 50     # ÙŠÙˆÙ‚Ù� Ø¹Ù†Ø¯ Ø£Ù�Ø¶Ù„ Ù†ØªÙŠØ¬Ø© Ù…Ø¨ÙƒØ±Ù‹Ø§
}

train_pool = Pool(X, y, cat_features=cat_features_idx)

cv_results = cv(
    params=params,
    pool=train_pool,
    fold_count=N_FOLDS,
    shuffle=True,
    stratified=True,
    seed=SEED
)

print("Best CV AUC:", cv_results["test-AUC-mean"].max())


# ====================================================
submission[TARGET] = test_preds
submission.to_csv("submission.csv", index=False)
print("âœ… Saved submission_catboost_optimized.csv")


# ====================================================
# Validation sample
sample_val = X.iloc[:5]
sample_pool_val = Pool(sample_val, cat_features=cat_features_idx)
print("\nğŸ§ª Sample Validation predictions:", model.predict_proba(sample_pool_val)[:, 1])

# Test sample
sample_test = X_test.sample(5, random_state=1)
sample_pool_test = Pool(sample_test, cat_features=cat_features_idx)
print("\nğŸ§ª Sample Test predictions:", model.predict_proba(sample_pool_test)[:, 1])


# Ù†Ù�ØªØ±Ø¶ Ø£Ù† Ø¹Ù†Ø¯Ùƒ Ù…ÙˆØ¯ÙŠÙ„ Ù…Ø¯Ø±Ù‘Ù�Ø¨ Ø§Ø³Ù…Ù‡ final_model
# Ùˆ test Ù‡Ùˆ Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±
test_pool = Pool(X_test, cat_features=cat_features_idx)

# Ø§Ù„Ø­ØµÙˆÙ„ Ø¹Ù„Ù‰ Ø§Ø­ØªÙ…Ø§Ù„Ø§Øª target=1
test_preds = final_model.predict_proba(test_pool)[:, 1]

# ØªØ¬Ù‡ÙŠØ² Ù…Ù„Ù� submission
submission = sub.copy()  # sub Ù‡Ùˆ sample_submission.csv
submission["target"] = test_preds
submission.to_csv("submission_catboost_optimized.csv", index=False)

print("âœ… Saved submission_catboost_optimized.csv")


