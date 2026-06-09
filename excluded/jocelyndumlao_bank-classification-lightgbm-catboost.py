import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train_df.head().style.background_gradient(cmap='plasma')


test_df.head().style.background_gradient(cmap='plasma')


submission.head().style.background_gradient(cmap='plasma')


print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


train_df.info()


train_df.describe().style.background_gradient(cmap='tab20c')


# Drop 'id'
train_df.drop("id", axis=1, inplace=True)
test_ids = test_df["id"]
test_df.drop("id", axis=1, inplace=True)


# Handle categorical columns
cat_cols = train_df.select_dtypes(include="object").columns.tolist()

# Fill missing values (if any)
for col in cat_cols:
    train_df[col] = train_df[col].fillna("Unknown")
    test_df[col] = test_df[col].fillna("Unknown")


# Target variable
y = train_df["y"]
X = train_df.drop("y", axis=1)

# Encode categorical variables
X = pd.get_dummies(X, columns=cat_cols)
test_encoded = pd.get_dummies(test_df, columns=cat_cols)

# Align train & test
X, test_encoded = X.align(test_encoded, join='left', axis=1)
test_encoded = test_encoded.fillna(0)


plt.style.use("seaborn-v0_8-darkgrid")
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Exploratory Data Analysis", fontsize=18, fontweight="bold")

# 1. Target distribution
sns.countplot(x=y, ax=axs[0,0], palette="coolwarm")
axs[0,0].set_title("Target Distribution (y)")

# 2. Age distribution
sns.histplot(train_df["age"], bins=30, kde=True, ax=axs[0,1], color="skyblue")
axs[0,1].set_title("Age Distribution")

# 3. Balance by target
sns.boxplot(x=y, y=train_df["balance"], ax=axs[1,0], palette="Set2")
axs[1,0].set_title("Balance vs Target")

# 4. Job type counts
sns.countplot(y=train_df["job"], order=train_df["job"].value_counts().index, ax=axs[1,1])
axs[1,1].set_title("Job Distribution")

# 5. Duration distribution by target
sns.kdeplot(x=train_df["duration"], hue=y, ax=axs[2,0], fill=True)
axs[2,0].set_title("Duration Distribution by Target")

# 6. Campaign count distribution
sns.histplot(train_df["campaign"], bins=20, ax=axs[2,1], color="orange")
axs[2,1].set_title("Campaign Count Distribution")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



n_splits = 7
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(X.shape[0])
test_preds_lgb = np.zeros(test_encoded.shape[0])
roc_scores_lgb = []

params_lgb = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nðŸ“‚ Fold {fold+1}/{n_splits}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params_lgb,
        train_set,
        num_boost_round=15000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(500)]
    )
    
    val_preds = model.predict(X_val)
    oof_preds_lgb[val_idx] = val_preds
    test_preds_lgb += model.predict(test_encoded) / n_splits
    
    score = roc_auc_score(y_val, val_preds)
    roc_scores_lgb.append(score)
    print(f"ROC-AUC: {score:.5f}")



plt.figure(figsize=(8,6))
RocCurveDisplay.from_predictions(y, oof_preds_lgb)
plt.title("LightGBM ROC Curve")
plt.show()

print(f"\nâœ… LightGBM Mean ROC-AUC: {np.mean(roc_scores_lgb):.5f}")



oof_preds_cb = np.zeros(X.shape[0])
test_preds_cb = np.zeros(test_encoded.shape[0])
roc_scores_cb = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nðŸ“‚ CatBoost Fold {fold+1}/{n_splits}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.01,
        depth=6,
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=0)
    
    val_preds = model.predict_proba(X_val)[:,1]
    oof_preds_cb[val_idx] = val_preds
    test_preds_cb += model.predict_proba(test_encoded)[:,1] / n_splits
    
    score = roc_auc_score(y_val, val_preds)
    roc_scores_cb.append(score)
    print(f"ROC-AUC: {score:.5f}")



plt.figure(figsize=(8,6))
RocCurveDisplay.from_predictions(y, oof_preds_cb)
plt.title("CatBoost ROC Curve")
plt.show()

print(f"\nâœ… CatBoost Mean ROC-AUC: {np.mean(roc_scores_cb):.5f}")



submission["id"] = test_ids
submission["y"] = test_preds_lgb
submission.head()

submission.to_csv("submission.csv", index=False)
print("\nðŸ“„ Submission file saved as submission.csv")


submission.head()

