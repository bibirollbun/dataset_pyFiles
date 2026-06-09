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
import lightgbm as lgb
import gc
from sklearn.model_selection import KFold

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/final_music_train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/final_music_test.csv')
sample_sub = pd.read_csv('/kaggle/input/sample-submission-set-6/sample_submission.csv')

# Features and target
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']
gc.collect()

# Setup for faster training
N_SPLITS = 3
cv = KFold(N_SPLITS, random_state=42, shuffle=True)
y_pred = np.zeros(len(test_df))
rmse_scores = []

# Cross-validation loop (no encoding)
for idx_train, idx_valid in cv.split(X, y):
    X_train, y_train_fold = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid_fold = X.iloc[idx_valid], y.iloc[idx_valid]
    X_test = test_df[X.columns].copy()

    model = lgb.LGBMRegressor(
        n_estimators=800,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.015,
        objective='l2',
        metric='rmse',
        verbosity=-1,
        device='cpu'
    )

    model.fit(
        X_train, y_train_fold,
        eval_set=[(X_valid, y_valid_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )

    rmse_scores.append(model.best_score_['valid_0']['rmse'])
    y_pred += model.predict(X_test)

# Average predictions
pred_lgbm = y_pred / N_SPLITS

# Create submission using correct IDs
submission_lgbm = sample_sub.copy()
submission_lgbm['Listening_Time_minutes'] = pred_lgbm
submission_lgbm.to_csv('submission6.csv', index=False)

# RMSE output
print(f"\n✅ Best Validation RMSE: {min(rmse_scores):.4f}")

