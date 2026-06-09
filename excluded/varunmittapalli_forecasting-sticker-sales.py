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


# ==========================================================
# ðŸ§  Playground Series S5E1 - Forecasting Sticker Sales 
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

print("âœ… Data Loaded Successfully")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nColumns:", train.columns.tolist())

target_col = 'num_sold' if 'num_sold' in train.columns else train.columns[-1]

# ----------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (GRAPHS)
# ----------------------------------------------------------
plt.figure(figsize=(10,5))
sns.histplot(train[target_col], bins=50, kde=True)
plt.title("Distribution of Target (num_sold)")
plt.show()

if 'date' in train.columns:
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])

    plt.figure(figsize=(12,5))
    sns.lineplot(data=train, x='date', y=target_col, color='teal')
    plt.title("Sales Trend Over Time")
    plt.show()

if 'country' in train.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=train, x='country', y=target_col, palette='Set2')
    plt.title("Sales by Country")
    plt.show()

if 'store' in train.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=train, x='store', y=target_col, palette='cool')
    plt.title("Sales by Store")
    plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(train.select_dtypes(include=np.number).corr(), annot=True, cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

# Additional insights
if 'month' not in train.columns and 'date' in train.columns:
    train['month'] = train['date'].dt.month
    plt.figure(figsize=(10,5))
    sns.barplot(x='month', y=target_col, data=train.groupby('month')[target_col].mean().reset_index())
    plt.title("Average Sales per Month")
    plt.show()

if 'store' in train.columns:
    top_stores = train.groupby('store')[target_col].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_stores.index, y=top_stores.values, palette='magma')
    plt.title("Top 10 Stores by Total Sales")
    plt.show()

# ----------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------
if 'date' in train.columns:
    for df in [train, test]:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['week'] = df['date'].dt.isocalendar().week.astype(int)

cat_cols = train.select_dtypes(include='object').columns.tolist()
print("\nCategorical Columns:", cat_cols)

le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# ----------------------------------------------------------
# 4. MODEL TRAINING (LOG TRANSFORMED)
# ----------------------------------------------------------
X = train.drop([target_col, 'date'], axis=1, errors='ignore')
y = train[target_col]
X_test = test.drop(['date'], axis=1, errors='ignore')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# log-transform to stabilize
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train_log)
print("âœ… Model Trained Successfully")

# ----------------------------------------------------------
# 5. VALIDATION SCORE (RMSLE) â€” FIXED
# ----------------------------------------------------------
val_preds_log = model.predict(X_valid)
val_preds = np.expm1(val_preds_log)

# Replace any invalid predictions (NaN or negative)
val_preds = np.nan_to_num(val_preds, nan=0.0)
val_preds = np.maximum(0, val_preds)

# Ensure y_valid has no NaNs or negatives
y_valid_clean = np.nan_to_num(y_valid, nan=0.0)
y_valid_clean = np.maximum(0, y_valid_clean)

# RMSLE calculation
val_score = np.sqrt(mean_squared_log_error(y_valid_clean, val_preds))
print(f"ðŸ“‰ Local Validation RMSLE: {val_score:.5f}")

# ----------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ----------------------------------------------------------
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance (LightGBM)")
plt.show()

# ----------------------------------------------------------
# 7. FINAL PREDICTIONS & SUBMISSION
# ----------------------------------------------------------
test_preds_log = model.predict(X_test)
test_preds = np.expm1(test_preds_log)

# Clean up final predictions
test_preds = np.nan_to_num(test_preds, nan=0.0)
test_preds = np.maximum(0, test_preds)

submission = sample_submission.copy()
submission['num_sold'] = test_preds
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file created successfully: submission.csv")
submission.head()

