# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# Define target and ID column names
TARGET = "accident_risk"
ID = "id"


print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# Plot the distribution of the target variable
plt.figure(figsize=(12,4))
sns.histplot(train[TARGET], kde=True, bins=30)
plt.title("Target Distribution: Accident Risk")
plt.show()


# Identify categorical and numerical columns
cat_cols = train.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = [col for col in train.columns if col not in cat_cols + [TARGET, ID]]

# Quick correlation heatmap for numerical features vs target
plt.figure(figsize=(8,6))
sns.heatmap(train[num_cols+[TARGET]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlation with Accident Risk")
plt.show()


# Function to label encode categorical features
def encode_categorical(train_df, test_df, cols):
    for col in cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0]))
    return train_df, test_df

# Apply encoding
train, test = encode_categorical(train, test, cat_cols)

# Scale numerical features
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# Prepare train and test sets
X = train[cat_cols + num_cols]
y = train[TARGET]
X_test = test[cat_cols + num_cols]


kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RMSE metric
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# LightGBM parameters (from best Optuna trial)
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.0224,
    "num_leaves": 168,
    "feature_fraction": 0.896,
    "bagging_fraction": 0.884,
    "bagging_freq": 2,
    "min_data_in_leaf": 20,
    "random_state": 42,
    "device": "gpu" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
}

# Out-of-fold predictions
oof = np.zeros(len(X))
for train_idx, val_idx in kf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=500,
                      callbacks=[lgb.early_stopping(stopping_rounds=50)])
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

print("CV RMSE:", rmse(y, oof))


# Train final model on all data
final_model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=500)

# Make predictions on test set
preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Create submission file
submission = pd.DataFrame({ID: test[ID], TARGET: preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")


lgb.plot_importance(final_model, max_num_features=20, importance_type="gain", figsize=(10,6))
plt.title("Top 20 Feature Importances")
plt.show()

