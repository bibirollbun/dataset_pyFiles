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
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler, LabelEncoder
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import FunctionTransformer

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

train.drop(columns=['id'], inplace=True)
test_id = test['id']
test.drop(columns=['id'], inplace=True)


for df in [train, test]:
    for feature in ['Body_Temp', 'Heart_Rate', 'Height', 'Weight']:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['Duration_Heart_Rate'] = df['Duration'] * df['Heart_Rate']
    df['Duration_Body_Temp'] = df['Duration'] * df['Body_Temp']
    df['Heart_Rate_Body_Temp'] = df['Heart_Rate'] * df['Body_Temp']
    df['Weight_per_Age'] = df['Weight'] / (df['Age'] + 1)
    df['Temp_per_Heart'] = df['Body_Temp'] / (df['Heart_Rate'] + 1)
    df['Height_per_Age'] = df['Height'] / (df['Age'] + 1)
    df['Heart_Duration_Ratio'] = df['Heart_Rate'] / (df['Duration'] + 1)
    df['Duration_Squared'] = df['Duration'] ** 2
    df['Body_Temp_Squared'] = df['Body_Temp'] ** 2

features_to_scale = [
    'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
    'BMI', 'Duration_Heart_Rate', 'Duration_Body_Temp', 'Heart_Rate_Body_Temp',
    'Weight_per_Age', 'Temp_per_Heart', 'Height_per_Age', 'Heart_Duration_Ratio',
    'Duration_Squared', 'Body_Temp_Squared']

scaler = RobustScaler()
train[features_to_scale] = scaler.fit_transform(train[features_to_scale])
test[features_to_scale] = scaler.transform(test[features_to_scale])

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


X = train.drop(columns=['Calories'])
y = np.log1p(train['Calories'])  
X_test = test.copy()

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

meta_features_train = np.zeros((X.shape[0], 3))
meta_features_test = np.zeros((X_test.shape[0], 3))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== Fold {fold + 1} =====")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(tree_method='hist', device='cuda', eval_metric='rmse', random_state=42)
    xgb_model.fit(X_tr, y_tr)
    meta_features_train[val_idx, 0] = xgb_model.predict(X_val)
    meta_features_test[:, 0] += xgb_model.predict(X_test) / n_splits

    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(device='gpu', verbose=-1, random_state=42)
    lgb_model.fit(X_tr, y_tr)
    meta_features_train[val_idx, 1] = lgb_model.predict(X_val)
    meta_features_test[:, 1] += lgb_model.predict(X_test) / n_splits

    # CatBoost
    print("Training CatBoost...")
    cat_model = CatBoostRegressor(verbose=0, task_type='GPU', random_seed=42)
    cat_model.fit(X_tr, y_tr)
    meta_features_train[val_idx, 2] = cat_model.predict(X_val)
    meta_features_test[:, 2] += cat_model.predict(X_test) / n_splits

best_params = {'alpha': 97.98252991347158, 'tol': 7.525332977441315e-05}

ridge = Ridge(**best_params)
ridge.fit(meta_features_train, y) 
ridge_preds_log = ridge.predict(meta_features_train)

ridge_preds = np.expm1(ridge_preds_log)  
y_true = np.expm1(y)                    

rmsle_train = mean_squared_log_error(y_true, ridge_preds, squared=False)
print(f"\n RMSLE: {rmsle_train:.5f}")


final_preds_log = ridge.predict(meta_features_test)
final_preds = np.expm1(final_preds_log)  


submission = pd.DataFrame({"id": test_id, "Calories": final_preds})
submission.to_csv("stacking_submission.csv", index=False)
submission.head()

