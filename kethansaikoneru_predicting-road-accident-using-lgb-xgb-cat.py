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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


def create_features(df, top_features=None):
    df = df.copy()
    # Your previous feature engineering...
    # Add interaction/polynomial features
    # Add the new 'score' feature
    df['score'] = df['curvature'] * df['speed_limit'] / (df['num_reported_accidents'] + 1)
    # You may repeat with other combinations if relevant
    # Add more features below as needed
    return df


numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
for drop_col in ['id', 'accident_risk']:
    if drop_col in numeric_cols:
        numeric_cols.remove(drop_col)
correlations = train[numeric_cols + ['accident_risk']].corr()['accident_risk'].sort_values(ascending=False)
top_features = correlations.head(4).index.tolist()
if 'accident_risk' in top_features:
    top_features.remove('accident_risk')



# Apply feature engineering
train_fe = create_features(train, top_features=top_features)
test_fe = create_features(test, top_features=top_features)



# Prepare train and test data (drop id, target columns)
target = train_fe['accident_risk'].values
train_ids = train_fe['id'].values
test_ids = test_fe['id'].values
X_train = train_fe.drop(['id', 'accident_risk'], axis=1)
X_test = test_fe.drop(['id'], axis=1)


# Ensure columns align
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols: X_test[col] = 0
extra_cols = set(X_test.columns) - set(X_train.columns)
if len(extra_cols) > 0: X_test = X_test.drop(columns=extra_cols)
X_test = X_test[X_train.columns]


# Encode categorical variables
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le



# CV Helper
def get_cv_scores(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(fold_score)
        print(f"Fold {fold+1} RMSE: {fold_score:.5f}")
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"\nMean CV RMSE: {mean_cv:.5f} (+/- {std_cv:.5f})")
    return oof_preds, cv_scores, mean_cv


# Train base models and collect OOF/test predictions
lgb_model = LGBMRegressor(random_state=42)
lgb_oof, _, lgb_cv = get_cv_scores(lgb_model, X_train, target)
lgb_model.fit(X_train, target)
lgb_pred = lgb_model.predict(X_test)

cat_model = CatBoostRegressor(verbose=0, random_state=42)
cat_oof, _, cat_cv = get_cv_scores(cat_model, X_train, target)
cat_model.fit(X_train, target)
cat_pred = cat_model.predict(X_test)

xgb_model = XGBRegressor(random_state=42)
xgb_oof, _, xgb_cv = get_cv_scores(xgb_model, X_train, target)
xgb_model.fit(X_train, target)
xgb_pred = xgb_model.predict(X_test)


# Ensemble: mean or Ridge blending
# Simple mean ensemble
ensemble_pred = (lgb_pred + cat_pred + xgb_pred) / 3



# Ridge regression blending (optional, improves performance)
blend_oof = np.column_stack([lgb_oof, cat_oof, xgb_oof])
blend_test = np.column_stack([lgb_pred, cat_pred, xgb_pred])
ridge = Ridge()
ridge.fit(blend_oof, target)
ensemble_pred_ridge = ridge.predict(blend_test)




# Choose which ensemble to submit (mean or ridge)
final_pred = np.clip(ensemble_pred_ridge, 0, 1)


# Save submission
sample_submission['accident_risk'] = final_pred
sample_submission.to_csv('submission.csv', index=False)

print("Submission saved as submission.csv")




