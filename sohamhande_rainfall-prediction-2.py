import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# Load the data
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# Display basic info
print("Train Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())



# Check for missing values
print("\nMissing Values in Train:")
print(train_df.isnull().sum())
print("\nMissing Values in Test:")
print(test_df.isnull().sum())



# EDA - Visualizations
plt.figure(figsize=(8, 5))
sns.countplot(x='rainfall', data=train_df)
plt.title('Target Variable Distribution')
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# Handling missing values
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)


# Encoding categorical variables
label_encoders = {}
for col in train_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    if col in test_df.columns:
        test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le


# Splitting data
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']


# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Scaling numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_df)


# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_




# Model evaluation
y_pred = best_rf.predict(X_val)
print("RandomForest Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# Try XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)


# XGBoost Evaluation
y_pred_xgb = xgb_model.predict(X_val)
print("XGBoost Accuracy:", accuracy_score(y_val, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_val, y_pred_xgb))



# Predictions on test set
test_predictions = xgb_model.predict_proba(X_test)[:, 1]


submission = pd.read_csv(sample_submission_path)
submission['rainfall'] = test_predictions
submission.to_csv('submission2.csv', index=False)

print("Submission file 'submission2.csv' created successfully!")




