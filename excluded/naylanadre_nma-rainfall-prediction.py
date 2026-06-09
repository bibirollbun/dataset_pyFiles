# 1. Import Libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress warnings

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel

# 2. Auto-Detect Dataset Folder
base_path = '/kaggle/input'
dataset_folder = None
for folder in os.listdir(base_path):
    if 'playground-series-s5e3' in folder.lower():
        dataset_folder = os.path.join(base_path, folder)
        break

if not dataset_folder:
    raise FileNotFoundError("Dataset folder for Playground Series S5E3 not found. Make sure you attached the dataset in the sidebar.")

print(f"Using dataset folder: {dataset_folder}")
print("Available files:", os.listdir(dataset_folder))

# 3. Load Data
train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))

# ğŸ”¹ Standardize column names (lowercase, remove spaces)
train.columns = train.columns.str.lower().str.strip()
test.columns = test.columns.str.lower().str.strip()

# ğŸ”¹ Fix incorrect column name spelling
column_map = {'temparature': 'temperature'}
train.rename(columns=column_map, inplace=True)
test.rename(columns=column_map, inplace=True)

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
print("Columns in Train Dataset:", train.columns.tolist())

# 4. Handle Missing Values ğŸš€
# Identify numerical columns (excluding target 'rainfall' from train)
num_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [col for col in num_cols if col in test.columns]  # Ensure the column exists in test set too

# Fill missing values with mean for numerical columns
imputer = SimpleImputer(strategy="mean")
train[num_cols] = imputer.fit_transform(train[num_cols])
test[num_cols] = imputer.transform(test[num_cols])

# 5. Feature Engineering ğŸš€
if 'date' in train.columns:
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])
    train['month'] = train['date'].dt.month
    train['day_of_year'] = train['date'].dt.dayofyear
    test['month'] = test['date'].dt.month
    test['day_of_year'] = test['date'].dt.dayofyear

# âœ… Interaction Features
if 'temperature' in train.columns and 'humidity' in train.columns:
    train['temp_humidity'] = train['temperature'] * train['humidity']
    test['temp_humidity'] = test['temperature'] * test['humidity']

if 'temperature' in train.columns and 'windspeed' in train.columns:
    train['temp_wind'] = train['temperature'] * train['windspeed']
    test['temp_wind'] = test['temperature'] * test['windspeed']

if 'humidity' in train.columns and 'windspeed' in train.columns:
    train['humidity_wind'] = train['humidity'] * train['windspeed']
    test['humidity_wind'] = test['humidity'] * test['windspeed']

# âœ… Handle Categorical Variables
categorical_cols = train.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
    test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# âœ… Ensure Test Set Matches Train Set
missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0  # Add missing columns in test set
test = test[train.columns]

# 6. Feature Selection (Dropping Unnecessary Columns)
drop_cols = ['id']
if 'date' in train.columns:
    drop_cols.append('date')

# Ensure 'rainfall' is not included in test set
X = train.drop(columns=drop_cols + ['rainfall'])  # Train features
y = train['rainfall']  # Target variable

# Drop 'rainfall' from test dataset if it exists
X_test = test.drop(columns=drop_cols, errors='ignore')
X_test = X_test[X.columns]  # Align columns to match train dataset

# 7. Standardize Features ğŸš€
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 8. Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Feature Selection ğŸš€
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

sfm = SelectFromModel(rf, threshold='median')  
X_train_selected = sfm.transform(X_train)
X_val_selected = sfm.transform(X_val)
X_test_selected = sfm.transform(X_test)

# 10. Train Models ğŸš€
best_models = {}

# Logistic Regression
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg.fit(X_train_selected, y_train)
best_models['Logistic Regression'] = log_reg

# Random Forest
rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
rf.fit(X_train_selected, y_train)
best_models['Random Forest'] = rf

# XGBoost
xgb = XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]), eval_metric='logloss', random_state=42)
xgb.fit(X_train_selected, y_train)
best_models['XGBoost'] = xgb

# LightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train_selected, y_train)
best_models['LightGBM'] = lgbm

# CatBoost
catboost = CatBoostClassifier(verbose=0, random_state=42)
catboost.fit(X_train_selected, y_train)
best_models['CatBoost'] = catboost

# 11. Stacking Model ğŸš€
stacking_model = StackingClassifier(
    estimators=[
        ('log_reg', best_models['Logistic Regression']),
        ('rf', best_models['Random Forest']),
        ('xgb', best_models['XGBoost']),
        ('lgbm', best_models['LightGBM']),
        ('catboost', best_models['CatBoost'])
    ],
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train_selected, y_train)
y_val_pred = stacking_model.predict_proba(X_val_selected)[:, 1]
stacking_auc = roc_auc_score(y_val, y_val_pred)

print(f"\nğŸ�† Stacking Model Validation AUC: {stacking_auc:.4f}")

# 12. Predict on Test Set
test['rainfall'] = stacking_model.predict_proba(X_test_selected)[:, 1]

# 13. Save Submission
submission = test[['id', 'rainfall']]
submission.to_csv('submission.csv', index=False)

print("âœ… Final optimized submission file saved as 'submission.csv'")


