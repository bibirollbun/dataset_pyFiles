import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# Display basic info
print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)

# Check for missing values
print("Missing values:\n", train_df.isnull().sum())

# Explore dataset
print(train_df.head())




# Separate features and target
X = train_df.drop(columns=["id", "rainfall"])  # Drop ID and target column
y = train_df["rainfall"]  # Target column

# Encode categorical variables (if any)
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])




# Standardize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)
test_df[X.columns] = scaler.transform(test_df.drop(columns=["id"]))

# Split into train-validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Train model (XGBoost)
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


# Evaluate on validation set
y_val_pred = model.predict_proba(X_val)[:, 1]
roc_score = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC-AUC Score: {roc_score:.4f}")

# Make test predictions
test_preds = model.predict_proba(test_df.drop(columns=["id"]))[:, 1]


# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")

