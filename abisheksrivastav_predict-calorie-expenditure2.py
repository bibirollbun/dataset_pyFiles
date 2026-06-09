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
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Preprocess 'Sex'
train['Sex'] = train['Sex'].str.lower().str.strip()
test['Sex'] = test['Sex'].str.lower().str.strip()

# Feature engineering
def add_features(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HR_per_min'] = df['Heart_Rate'] / df['Duration'].replace(0, np.nan)
    df['Temp_per_min'] = df['Body_Temp'] / df['Duration'].replace(0, np.nan)
    df['Age_BMI'] = df['Age'] * df['BMI']
    df['Heart_Weight'] = df['Heart_Rate'] / df['Weight'].replace(0, np.nan)
    df['Duration_BMI'] = df['Duration'] * df['BMI']
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Log_Weight'] = np.log1p(df['Weight'])
    df.fillna(0, inplace=True)
    return df

train = add_features(train)
test = add_features(test)

# Target
y = train['Calories']
X = train.drop(['Calories', 'id'], axis=1)
X_test = test.drop('id', axis=1)

# Label encode
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])

# Log1p target for RMSLE
y_log = np.log1p(y)

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lgb, oof_preds_xgb, oof_preds_cat = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_preds_lgb, test_preds_xgb, test_preds_cat = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=31, random_state=42
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100, verbose=0)
    oof_preds_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6, random_state=42,
        verbose=0, early_stopping_rounds=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    oof_preds_cat[val_idx] = cat_model.predict(X_val)
    test_preds_cat += cat_model.predict(X_test) / kf.n_splits

# Final predictions (simple average)
oof_final = (
    0.5 * oof_preds_lgb +
    0.3 * oof_preds_cat +
    0.2 * oof_preds_xgb
)

# Weight based on validation performance 
test_final = (
    0.5 * test_preds_lgb +
    0.3 * test_preds_cat +
    0.2 * test_preds_xgb
)


# Evaluate
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_log), np.expm1(oof_final)))
print(f'OOF RMSLE Ensemble: {rmsle:.5f}')

# Submit
submission['Calories'] = np.expm1(test_final).clip(0)
submission.to_csv('submission_ensemble.csv', index=False)


