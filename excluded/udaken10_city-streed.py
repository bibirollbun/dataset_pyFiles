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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from catboost import Pool


train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')
train_df.head()


# AL, LAなどは市を表し、13、10などはストリートを表現すると思われるので、_でsplitして市と数値に分ける
train_df['city'] = train_df['ID'].str.split('_').str[0]
train_df['num'] = train_df['ID'].str.split('_').str[1]

test_df['city'] = test_df['ID'].str.split('_').str[0]
test_df['num'] = test_df['ID'].str.split('_').str[1]

train_df


# 	cityについて、OneHotEncodingする

from sklearn.preprocessing import OneHotEncoder

train = train_df.copy()
test = test_df.copy()

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train[['city']])

train_onehot_df = pd.DataFrame(enc.transform(train[['city']]).toarray(), columns=enc.get_feature_names_out())
test_onehot_df = pd.DataFrame(enc.transform(test[['city']]).toarray(), columns=enc.get_feature_names_out())

train = pd.concat([train, train_onehot_df], axis=1)
test = pd.concat([test, test_onehot_df], axis=1)

train


# 'num'についてもOneHotEncodingする

enc_2 = OneHotEncoder(handle_unknown='ignore')
enc_2.fit(train[['num']])

train_onehot_df_2 = pd.DataFrame(enc_2.transform(train[['num']]).toarray(), columns=enc_2.get_feature_names_out())
test_onehot_df_2 = pd.DataFrame(enc_2.transform(test[['num']]).toarray(), columns=enc_2.get_feature_names_out())

train = pd.concat([train, train_onehot_df_2], axis=1)
test = pd.concat([test, test_onehot_df_2], axis=1)

train


target = train_df['HOMELESS_RATE']
train = train.drop(['ID','city','num'], axis = 1)
test = test.drop(['ID','city', 'num'], axis = 1)
test.columns.values


train.drop('HOMELESS_RATE', axis = 1, inplace = True)



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from catboost import Pool

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)

# catboostRegressorのパラメーターをグリッドサーチする
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 1, 2],
    'random_strength': [0, 1, 2],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}

model_cat = CatBoostRegressor()
grid_search = GridSearchCV(estimator=model_cat, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 最適なパラメータを表示
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)

rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
print("RMSE: ", rmse)


prediction = grid_search.predict(test)
prediction


sub['HOMELESS_RATE'] = prediction
sub


sub.to_csv('submission.csv', index = False)




