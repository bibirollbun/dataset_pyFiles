import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)


# Drop ID column and separate features/target
X = train_df.drop(columns=["id", "rainfall"])
y = train_df["rainfall"]
test_ids = test_df["id"]
test_df = test_df.drop(columns=["id"])

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])

# Standardize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)
test_df[X.columns] = scaler.transform(test_df)


# Split into train-validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)
logreg_preds = logreg.predict_proba(X_val)[:, 1]
logreg_score = roc_auc_score(y_val, logreg_preds)


# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict_proba(X_val)[:, 1]
rf_score = roc_auc_score(y_val, rf_preds)

# Train XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
xgb_score = roc_auc_score(y_val, xgb_preds)


# Print model performance
print(f"ROC-AUC Scores:\n Logistic Regression: {logreg_score:.4f}\n Random Forest: {rf_score:.4f}\n XGBoost: {xgb_score:.4f}")

# Feature importance analysis using XGBoost
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
top_features = feature_importance.head(20)['Feature'].tolist()  # Keep top 20 features



# Retrain XGBoost with selected features
X_train_selected = X_train[top_features]
X_val_selected = X_val[top_features]
test_selected = test_df[top_features]

xgb_model.fit(X_train_selected, y_train)
xgb_preds_selected = xgb_model.predict_proba(X_val_selected)[:, 1]
xgb_score_selected = roc_auc_score(y_val, xgb_preds_selected)

print(f"Optimized XGBoost ROC-AUC after Feature Selection: {xgb_score_selected:.4f}")


# Make final test predictions using best model (XGBoost with selected features)
final_test_preds = xgb_model.predict_proba(test_selected)[:, 1]


# Create submission file
submission = pd.DataFrame({"id": test_ids, "rainfall": final_test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")

