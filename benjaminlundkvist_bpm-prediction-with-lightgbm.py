import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# Check the first few rows
train.head()


print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nMissing values:\n", train.isnull().sum())


# Distribution of target variable
plt.figure(figsize=(8,5))
sns.histplot(train["BeatsPerMinute"], bins=50, kde=True, color="skyblue")
plt.title("Distribution of Beats Per Minute (Target)")
plt.xlabel("BPM")
plt.ylabel("Count")
plt.show()


# Correlation heatmap
plt.figure(figsize=(10,8))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.show()


features = [col for col in train.columns if col not in ["id", "BeatsPerMinute"]]

X = train[features]
y = train["BeatsPerMinute"]
X_test = test[features]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=40,
    min_child_samples=30,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=2,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)


# Training RMSE
train_preds = model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_preds, squared=False)

# Validation RMSE
val_preds = model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_preds, squared=False)

# Print the evaluation results
print(f"LightGBM train RMSE: {train_rmse:.4f}")
print(f"LightGBM validation RMSE: {val_rmse:.4f}")


test_predictions = model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


submission.head()

