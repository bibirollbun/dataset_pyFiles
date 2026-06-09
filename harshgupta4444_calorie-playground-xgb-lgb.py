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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])



def add_features(df):
    # BMI (Body Mass Index)
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    # Metabolic equivalents (simplified)
    df['MET'] = df['Duration'] * (df['Heart_Rate'] / df['Age'])
    
    # Interaction terms
    df['HR_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Temp_Duration'] = df['Body_Temp'] * df['Duration']
    
    # Physiological metrics
    df['Cardio_Load'] = df['Heart_Rate'] * df['Duration'] / df['Age']
    return df


train = add_features(train)
test = add_features(test)


features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Sex', 
            'Weight', 'Height', 'BMI', 'MET', 'HR_Duration', 
            'Temp_Duration', 'Cardio_Load']
target = 'Calories'


X = train[features]
y = train[target]


y_log = np.log1p(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42
)


xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmsle',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': 42,
    'n_jobs': -1
}


xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)


lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity': -1,
    'random_state': 42
}


train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)


def ensemble_predict(X):
    xgb_pred = xgb_model.predict(X)
    lgb_pred = lgb_model.predict(X)
    return (xgb_pred + lgb_pred) / 2


val_pred = ensemble_predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(val_pred)))
print(f"Validation RMSLE: {rmsle:.5f}")


X_test = test[features]
X_test_scaled = scaler.transform(X_test)


test_pred_log = ensemble_predict(X_test_scaled)
test_pred = np.expm1(test_pred_log)


submission = pd.DataFrame({'id': test['id'], 'Calories': test_pred})
submission.to_csv('submission3.csv', index=False)
print("Submission file created!")




