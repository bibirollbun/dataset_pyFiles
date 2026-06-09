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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import optuna


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


train_df.head()


train_df.isna().sum() 


# binary encoding

train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)


def new_features(df):
    
    # Body mass index
    df['BMR'] = np.where(
        df['Sex'] == 1,
        (66.1 + 13.8 * df['Weight'] + 5 * df['Height'] - 6.8 * df['Age']),
        (655 + 9.5 * df['Weight'] + 1.9 * df['Height'] - 4.7 * df['Age'])
    )
    
    # Training intensity
    df['HR_per_kg'] = df['Heart_Rate']/df['Weight']
    df['Temp_per_kg'] = df['Body_Temp']/df['Weight']
    
    # Age Groups
    df['Age_Group'] = df['Age'].apply(lambda x: 0 if x <= 35 else 1 if x <= 55 else 2)
    
    return df


# Launch func

train_df = new_features(train_df)
test_df = new_features(test_df)


# df split

X_train = train_df.drop(['Calories', 'id'], axis=1)
y_train = train_df['Calories']

X_test = test_df.drop(['id'], axis=1)


# selection split

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,   
    random_state=42,     
    shuffle=True          
)


# optimized params
params = {'boosting_type': 'gbdt',
         'colsample_bytree': 0.93,
         'importance_type': 'split',
         'learning_rate': 0.081,
         'max_depth': 9,
         'min_child_samples': 30,
         'min_child_weight': 0.001,
         'min_split_gain': 0.0,
         'n_estimators': 243,
         'num_leaves': 115,
         'subsample': 0.956,
         'subsample_for_bin': 200000,
         'random_state': 42
         }

# LightGBM
model_lgb = lgb.LGBMRegressor(**params, verbose=-1)

# Model fit
model_lgb.fit(X_train, y_train)

# Prediction
y_pred_lgb= model_lgb.predict(X_train)
y_pred_lgb_val= model_lgb.predict(X_val)
y_pred_lgb_test= model_lgb.predict(X_test)

# Prediction
print(f'RMSLE_train: {round(mean_squared_log_error(y_pred_lgb, y_train, squared = False),5)}')
print('------------------------------')
print(f'RMSLE_test: {round(mean_squared_log_error(y_pred_lgb_val, y_val, squared = False),5)}')


submission=pd.DataFrame({'id':test_df['id'] ,'Calories': y_pred_lgb_test})
submission.to_csv('submission.csv', index=False)

