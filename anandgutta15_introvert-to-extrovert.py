# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score


import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test_ids = test['id'].copy()


train.head()


test.head()


train.isnull().sum()


train.info()


# Convert numeric columns to proper types
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

for col in numeric_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

# Fill missing values in numeric columns with median (more robust than mean)
for col in numeric_cols:
    train_median = train[col].median()
    train[col].fillna(train_median, inplace=True)
    test[col].fillna(train_median, inplace=True)  # Use train median for consistency

# Handle categorical columns
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

# Fill missing categorical values with mode
for col in categorical_cols:
    train_mode = train[col].mode()[0]
    train[col].fillna(train_mode, inplace=True)
    test[col].fillna(train_mode, inplace=True)  # Use train mode for consistency

print("Missing values after cleaning:")
print("Train:", train.isnull().sum().sum())
print("Test:", test.isnull().sum().sum())



# Display the cleaned train DataFrame
train.head()


# Initialize label encoders
label_encoders = {}

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
train['Personality'] = target_encoder.fit_transform(train['Personality'])

print("Categorical encoding completed.")
print("Target classes:", target_encoder.classes_)



# Select features for training
feature_cols = [col for col in train.columns if col not in ['id', 'Personality']]

X = train[feature_cols]
y = train['Personality']
X_test = test[feature_cols]

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Feature scaling (optional but recommended for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing completed!")



# Hyperparameter Grids
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

catboost_params = {
    'iterations': [300, 500],
    'learning_rate': [0.01, 0.1],
    'depth': [6, 8, 10],
    'l2_leaf_reg': [1, 5, 9],
}
xgb_params = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
lgb_params = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 8],
    'subsample': [0.8, 1.0]
}
rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}
lr_params = {
    'penalty': ['l2'],
    'C': np.logspace(-2, 2, 5),
    'solver': ['lbfgs', 'liblinear']
}


# Model Optimization and Training Functions
def optimize_and_eval(model, params, X, y):
    search = RandomizedSearchCV(
        model, params, 
        n_iter=8,
        scoring='f1', cv=cv, random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_score_, search.best_params_



model_scores = {}

# CatBoost
cat_model, cat_score, cat_params = optimize_and_eval(
    CatBoostClassifier(verbose=0, random_state=42), catboost_params, X_train, y_train)
y_pred_cat = cat_model.predict(X_val)
cat_acc = accuracy_score(y_val, y_pred_cat)
cat_f1 = f1_score(y_val, y_pred_cat)
model_scores['CatBoost'] = (cat_acc, cat_f1, cat_model, cat_params)

# XGBoost
xgb_model, xgb_score, xgb_params_ = optimize_and_eval(
    xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params, X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
xgb_acc = accuracy_score(y_val, y_pred_xgb)
xgb_f1 = f1_score(y_val, y_pred_xgb)
model_scores['XGBoost'] = (xgb_acc, xgb_f1, xgb_model, xgb_params_)

# LightGBM
lgb_model, lgb_score, lgb_params_ = optimize_and_eval(
    lgb.LGBMClassifier(random_state=42), lgb_params, X_train, y_train)
y_pred_lgb = lgb_model.predict(X_val)
lgb_acc = accuracy_score(y_val, y_pred_lgb)
lgb_f1 = f1_score(y_val, y_pred_lgb)
model_scores['LightGBM'] = (lgb_acc, lgb_f1, lgb_model, lgb_params_)


# RandomForest
rf_model, rf_score, rf_params_ = optimize_and_eval(
    RandomForestClassifier(random_state=42), rf_params, X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, y_pred_rf)
rf_f1 = f1_score(y_val, y_pred_rf)
model_scores['RandomForest'] = (rf_acc, rf_f1, rf_model, rf_params_)

# Logistic Regression (uses scaled)
lr_model, lr_score, lr_params_ = optimize_and_eval(
    LogisticRegression(random_state=42), lr_params, X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_val_scaled)
lr_acc = accuracy_score(y_val, y_pred_lr)
lr_f1 = f1_score(y_val, y_pred_lr)
model_scores['LogisticRegression'] = (lr_acc, lr_f1, lr_model, lr_params_)


# Print Results
print("Model Evaluation Results:")
for name, (acc, f1, _, params) in model_scores.items():
    print(f"{name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}, Best Params = {params}")

# 11. Choose Best Model & Retrain on All Train Data
# (Choose by highest F1/Accuracy)
best_model_name = max(model_scores.items(), key=lambda x: x[1][1])[0]
print(f"\nBest Model: {best_model_name}")

best_model = model_scores[best_model_name][2]

# Scale data if necessary
if best_model_name in ['LogisticRegression']:
    best_model.fit(scaler.fit_transform(X), y)
    test_pred = best_model.predict(scaler.transform(X_test))
else:
    best_model.fit(X, y)
    test_pred = best_model.predict(X_test)


# Prepare Submission
final_preds = target_encoder.inverse_transform(test_pred)
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': final_preds
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created at /kaggle/working/submission.csv")


df = pd.read_csv('/kaggle/working/submission.csv')
df.head()  




