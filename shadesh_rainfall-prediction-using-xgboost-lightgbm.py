# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# Display dataset information
print("\nðŸ”¹ Train Data Shape:", train_df.shape)
print("\nðŸ”¹ Test Data Shape:", test_df.shape)
print("\nðŸ”¹ Train Data Preview:\n", train_df.head())



# Check for missing values
print("\nðŸ”¹ Missing Values in Train Data:\n", train_df.isnull().sum())

# Check data types
print("\nðŸ”¹ Data Types:\n", train_df.dtypes)

# Statistical summary of numerical columns
print("\nðŸ”¹ Statistical Summary:\n", train_df.describe())


# Drop ID column and separate target
X = train_df.drop(columns=["id", "rainfall"])  
y = train_df["rainfall"]  

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])


# Standardize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)
test_df[X.columns] = scaler.transform(test_df.drop(columns=["id"]))

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Initialize models
xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)

# Train models
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Validate models
xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
lgb_pred = lgb_model.predict_proba(X_val)[:, 1]


# Evaluate models
xgb_score = roc_auc_score(y_val, xgb_pred)
lgb_score = roc_auc_score(y_val, lgb_pred)

print(f"\nðŸ”¹ XGBoost Validation ROC-AUC Score: {xgb_score:.4f}")
print(f"\nðŸ”¹ LightGBM Validation ROC-AUC Score: {lgb_score:.4f}")

# Feature Importance Analysis (Using XGBoost)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nðŸ”¹ Top 10 Most Important Features:\n", feature_importance.head(10))


# Plot Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10), palette="viridis")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.show()


# Drop less important features (Optional Step)
selected_features = feature_importance[feature_importance["Importance"] > 0.01]["Feature"].values
X_selected = X[selected_features]
test_selected = test_df[selected_features]

# Retrain with selected features
xgb_model.fit(X_selected, y)
test_preds = xgb_model.predict_proba(test_selected)[:, 1]


# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission file saved!")

