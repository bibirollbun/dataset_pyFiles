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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from catboost import CatBoostClassifier


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv") 


df.head()


# Drop unwanted columns
columns_to_drop = ['id']  
df = df.drop(columns=columns_to_drop, errors='ignore')




# Encode categorical columns for XGBoost (CatBoost can handle them natively)
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target
X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#XGBoost 
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print(f"XGBoost ROC-AUC: {auc_xgb:.4f}")

# CatBoost 
cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train, y_train, cat_features=cat_features_indices)
y_pred_proba_cat = cat_model.predict_proba(X_test)[:, 1]
auc_cat = roc_auc_score(y_test, y_pred_proba_cat)
print(f"CatBoost ROC-AUC: {auc_cat:.4f}")



# Load test dataset
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")  # replace with your test dataset path

# Keep the 'id' for submission
submission_ids = test_df['id']

# Drop unwanted columns and encode categorical features
test_df = test_df.drop(columns=['id'], errors='ignore')

# Encode categorical columns for XGBoost (CatBoost can handle them natively)
for col in categorical_cols:
    if col in test_df.columns:
        test_df[col] = LabelEncoder().fit_transform(test_df[col])

# Predict probabilities using XGBoost
y_test_pred_xgb = xgb_model.predict_proba(test_df)[:, 1]

# OR Predict probabilities using CatBoost
cat_features_indices = [test_df.columns.get_loc(col) for col in categorical_cols if col in test_df.columns]
y_test_pred_cat = cat_model.predict_proba(test_df)[:, 1]

# Create submission DataFrame (using CatBoost predictions here)
submission = pd.DataFrame({
    'id': submission_ids,
    'diagnosed_diabetes': y_test_pred_cat  # or y_test_pred_xgb
})

# Save submission file
submission.to_csv("diabetes_submission.csv", index=False)
print("Submission file created: diabetes_submission.csv")





