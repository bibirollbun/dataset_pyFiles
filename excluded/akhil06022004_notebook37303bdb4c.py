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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv')
test = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv')

# Store test IDs
test_ids = test.pop('id') if 'id' in test.columns else test.index
train.drop(columns=['id'], errors='ignore', inplace=True)

# Separate features and target
X = train.drop(columns=['target'])
y = train['target']

def preprocess_features(df, train_columns=None):
    df['interaction'] = df.iloc[:, 0] * df.iloc[:, 1]
    df['polynomial'] = df.iloc[:, 0] ** 2
    return df[train_columns] if train_columns else df

# Process features
train_columns = X.columns.tolist()
X = preprocess_features(X, train_columns)
test = preprocess_features(test, train_columns)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# Define models
models = {
    'lgb': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, max_depth=10, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42),
    'xgb': xgb.XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42),
    'catboost': cb.CatBoostRegressor(iterations=500, learning_rate=0.02, depth=8, verbose=False, random_state=42)
}

# Stacking with cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
meta_train = np.zeros((X.shape[0], len(models)))
meta_test = np.zeros((test.shape[0], len(models)))

for i, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    test_preds = np.zeros((test.shape[0], n_splits))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        meta_train[val_idx, i] = model.predict(X_val)
        test_preds[:, fold] = model.predict(test_scaled)
        
        print(f"{name} - Fold {fold + 1} R2 Score: {r2_score(y_val, meta_train[val_idx, i]):.4f}")
    
    meta_test[:, i] = test_preds.mean(axis=1)

# Train final meta-model
meta_model = RidgeCV()
meta_model.fit(meta_train, y)
final_predictions = meta_model.predict(meta_test)

# Save submission
submission = pd.DataFrame({'id': test_ids, 'target': final_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved.")

# Print meta-model contributions
for name, coef in zip(models.keys(), meta_model.coef_):
    print(f"{name} contribution: {coef:.4f}")



