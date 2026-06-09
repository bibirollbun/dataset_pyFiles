# Import Libraries
import numpy as np
import pandas as pd
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation
from sklearn.metrics import mean_squared_error


# Set purple theme for plots
sns.set_palette("Purples")
plt.style.use("seaborn-v0_8-darkgrid")

# =====================
# Data Loading
# =====================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Quick check
print("Train shape :", train.shape)
print("Test shape  :", test.shape)




train.head()


test.head()


# Overview of the dataset
print("ğŸ”� Info")
print(train.info())

print("\nğŸ“Š Missing values per column")
print(train.isnull().sum())


# Target distribution
plt.figure(figsize=(8,5))
sns.histplot(train["BeatsPerMinute"], bins=40, kde=True, color="purple")
plt.title("Distribution of BeatsPerMinute (Target)", fontsize=14, color="purple")
plt.xlabel("BeatsPerMinute")
plt.ylabel("Count")
plt.show()

# Quick descriptive statistics
train.describe().T


# Compute correlation matrix
corr_matrix = train.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=False, cmap="Purples", cbar=True)
plt.title("Correlation Matrix", fontsize=14, color="purple")
plt.show()

# Top correlations with the target
target_corr = corr_matrix["BeatsPerMinute"].sort_values(ascending=False)
print("ğŸ”� Correlation of features with BeatsPerMinute:\n")
print(target_corr)



# Pick top correlated features with BeatsPerMinute (excluding target itself)
corr_matrix = train.corr(numeric_only=True)
top_features = corr_matrix["BeatsPerMinute"].drop("BeatsPerMinute").abs().sort_values(ascending=False).head(4).index

# Scatterplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    sns.scatterplot(
        x=train[feature],
        y=train["BeatsPerMinute"],
        alpha=0.5,
        color="purple",
        ax=axes[i]
    )
    axes[i].set_title(f"{feature} vs BeatsPerMinute", color="purple")

plt.tight_layout()
plt.show()



# Drop non-predictive column
X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]

# Test features (without id)
X_test = test.drop(columns=["id"])

print("âœ… Feature matrix shape :", X.shape)
print("âœ… Target vector shape  :", y.shape)
print("âœ… Test set shape       :", X_test.shape)


from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_params = {
    'n_estimators': 371,
    'max_leaves': 127,
    'min_child_weight': 1.5,
    'max_depth': 2,
    'grow_policy': 'lossguide',
    'learning_rate': 0.03865068678819597,
    'tree_method': 'hist',          # with device='cuda' this will run on GPU (xgboost>=2.0)
    'subsample': 0.85,
    'colsample_bylevel': 0.7,
    'colsample_bytree': 0.75,
    'colsample_bynode': 0.85,
    'sampling_method': 'uniform',
    'reg_alpha': 2.5,
    'reg_lambda': 0.8,
    'enable_categorical': True,     # ok if you later pass categorical cols; harmless if all numeric
    'max_cat_to_onehot': 1,
    'device': 'cuda',               # GPU
    'n_jobs': -1,
    'random_state': 42,
}


# Model
model = XGBRegressor(**xgb_params
)

# Cross-validation
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    score = mean_squared_error(y_val, preds, squared=False)
    rmse_scores.append(score)
    print(f"Fold {fold}: RMSE = {score:.4f}")

print("\nâœ… Mean RMSE:", np.mean(rmse_scores).round(4))



# Fit the model on the entire training set
model.fit(X, y)

# Predict on test set
test_preds = model.predict(X_test)

# Build submission
submission = sample_submission.copy()
submission["BeatsPerMinute"] = test_preds

# Save (Kaggle will pick this up from working directory)
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created: submission.csv")
submission.head()


