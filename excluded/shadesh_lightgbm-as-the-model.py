# ========================================================
# Kaggle Playground Series - Season 5, Episode 8
# Binary Classification with a Bank Dataset
# Complete solution with LightGBM
# ========================================================

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# Step 2: Basic data exploration
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())
print("\nFirst few rows of train data:\n", train.head())

# Step 3: Data Preprocessing
# Check for missing values
print("\nMissing values in train:\n", train.isnull().sum())
print("\nMissing values in test:\n", test.isnull().sum())

# Encode categorical variables
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Step 4: Define features and target
X = train.drop(columns=['id', 'y'])  # Drop ID and target
y = train['y']  # Target variable
X_test = test.drop(columns=['id'])  # Test features

# Step 5: Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# Step 6: Initialize and train the LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    objective='binary',
    metric='auc',
    early_stopping_round=50,
    verbosity=1
)

# Train with early stopping
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100)
    ]
)

# Step 7: Evaluate the model
y_val_pred = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"\nValidation ROC AUC: {roc_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 important features:\n", feature_importance.head(10))

# Step 8: Make predictions on test set
y_test_pred = model.predict_proba(X_test)[:, 1]

# Step 9: Prepare submission
sample_submission['y'] = y_test_pred
print("\nSubmission head:\n", sample_submission.head())

# Step 10: Save submission
sample_submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved successfully!")

