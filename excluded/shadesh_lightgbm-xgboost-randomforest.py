import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ========================
# 1. Load Data
# ========================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# ========================
# 2. Basic Exploration
# ========================
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

print("\nTrain head:")
print(train.head())

print("\nMissing values in train:")
print(train.isnull().sum())

print("\nMissing values in test:")
print(test.isnull().sum())

print("\nTrain info:")
print(train.info())

# ========================
# 3. EDA - Visualizations
# ========================
plt.figure(figsize=(8,5))
sns.histplot(train['BeatsPerMinute'], bins=50, kde=True)
plt.title("Target Distribution: Beats Per Minute")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(train.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot of first few features + target
sns.pairplot(train.iloc[:,1:6].join(train['BeatsPerMinute']))
plt.show()

# Boxplots for first 5 features
for col in train.columns[1:6]:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=train[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Distribution plots for first 5 features
for col in train.columns[1:6]:
    plt.figure(figsize=(8,5))
    sns.histplot(train[col], bins=40, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# ========================
# 4. Data Preparation
# ========================
X = train.drop(["BeatsPerMinute", "id"], axis=1)
y = train["BeatsPerMinute"]

X_test = test.drop(["id"], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 5. Modeling - LightGBM (Fixed)
# ========================
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Validation performance
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print("Validation RMSE:", rmse)

# ========================
# 6. Feature Importance
# ========================
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(20).plot(kind='barh', figsize=(10,6))
plt.title("Top 20 Feature Importances")
plt.show()

# ========================
# 7. Submission
# ========================
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "ID": test["id"],
    "BeatsPerMinute": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

