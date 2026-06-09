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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from xgboost import XGBRegressor


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_df.head()


train_df['BPM'] = train_df['Heart_Rate'] * train_df['Duration']
test_df['BPM'] = test_df['Heart_Rate'] * test_df['Duration']

encoder = LabelEncoder()
train_df['Sex'] = encoder.fit_transform(train_df['Sex'])
test_df['Sex'] = encoder.transform(test_df['Sex'])


X = train_df.drop('Calories', axis=1)
y = train_df['Calories']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
rmsle_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    y_train_log = np.log1p(y_train)
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train_log)
    
    
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)  
    

    val_preds = np.maximum(val_preds, 0)
    
    mae = mean_absolute_error(y_val, val_preds)
    rmsle = np.sqrt(mean_squared_log_error(y_val, val_preds))
    
    print(f"Fold {fold + 1} - MAE: {mae:.2f}, RMSLE: {rmsle:.4f}")
    
    mae_scores.append(mae)
    rmsle_scores.append(rmsle)

print(f"\nAverage MAE: {np.mean(mae_scores):.2f}")
print(f"Average RMSLE: {np.mean(rmsle_scores):.4f}")


y_log = np.log1p(y)
final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=47)
final_model.fit(X_scaled, y_log)


test_preds_log = final_model.predict(test_scaled)
test_preds = np.expm1(test_preds_log)
test_preds = np.maximum(test_preds, 0)


submission = sample_submission.copy()
submission['Calories'] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

