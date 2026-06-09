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


train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')


train_df.head()


train_df.info()


train_df.shape


# Check null values

train_df.isnull().sum()/ len(train_df) * 100


train_df.dropna(inplace=True, axis=0)


# Check null values

train_df.isnull().sum()/ len(train_df) * 100


train_df['Podcast_Name'].unique()


train_df['Genre'].unique()


train_df.head()


# For now, I am removing Podcast Name and Episode Title which doesn't make sense to model in prediction

train_df.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)


test_df.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)


train_df.head()


# Coverting Categorical Features into Numerical 

# genre_dum = pd.get_dummies(train_df['Genre'], dtype=int)
# train_df = pd.concat([train_df,genre_dum], axis=1)


def transform_to_numerical(train_df, columns):
    for col in columns : 
        genre_dum = pd.get_dummies(train_df[col], dtype=int)
        train_df = pd.concat([train_df,genre_dum], axis=1)
    
    train_df.drop(columns= columns, inplace=True)
    return train_df

columns = ['Genre', 'Publication_Day','Publication_Time','Episode_Sentiment']


train_df = transform_to_numerical(train_df, columns)
test_df = transform_to_numerical(test_df, columns)


train_df.head()


y = train_df.pop('Listening_Time_minutes')
X = train_df


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X,y)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y, train_size=0.7, random_state=42)


X_train.shape


X_val.shape


import sklearn
sklearn.metrics.get_scorer_names()


%%time
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

xgb = XGBRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
} # 3 * 3 * 3 * 3 = 81 * no_of_cv = 243

xgb_gridcv = GridSearchCV(estimator=xgb,param_grid=param_grid, scoring = 'neg_root_mean_squared_error', n_jobs=-1, cv=4, verbose=True )

xgb_gridcv.fit(X_train,y_train)

xgb_gridcv.best_estimator_


xgb_with_tuning = xgb_gridcv.best_estimator_


import xgboost as xgboost
xgboost.plot_importance(xgb_with_tuning)


xgb_with_tuning.fit(X_train,y_train)



X_test = test_df
X_test = scaler.transform(X_test)


y_test_pred = xgb_with_tuning.predict(X_test)


id = test_df.pop('id')


sub = pd.DataFrame({'id': id, 'Listening_Time_minutes': y_test_pred })
sub.head()


sub.to_csv('submission.csv', index=0)




