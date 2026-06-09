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


from tqdm import tqdm
from itertools import combinations


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col = 'id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.head()


target = train['Listening_Time_minutes']
target


# train['Episode_Title'] をEpisoce と　数値に分けます

train[['Episode', 'Episode_Number']] = train['Episode_Title'].str.split('Episode ', expand=True)
train['Episode_Number'] = train['Episode_Number'].astype(float)
train.drop(['Episode_Title','Episode'], axis=1, inplace=True)

test[['Episode', 'Episode_Number']] = test['Episode_Title'].str.split('Episode ', expand=True)
test['Episode_Number'] = test['Episode_Number'].astype(float)
test.drop(['Episode_Title','Episode'], axis=1, inplace=True)



target_col = 'Listening_Time_minutes'
target = train[target_col]
train.drop(target_col, axis=1, inplace=True)


train.isnull().sum()


for col in train.columns:
    print(col, train[col].unique())
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(train[col].mode()[0],inplace = True)

train.describe()


train.isnull().sum()


def round_num(df):
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].round().astype('int')
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].round().astype('int')
    df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].round().astype('int')
    df['Number_of_Ads'] = df['Number_of_Ads'].round().astype('int')
    df['Episode_Number'] = df['Episode_Number'].round().astype('int')
    return df



train_df = round_num(train)
test_df = round_num(test)


cat_cols = ['Podcast_Name','Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


train_df


from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.2, random_state=42)


from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Define the objective function for Bayesian Optimization
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    val = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    model = RandomForestRegressor(n_estimators=val,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42,
                                  criterion='squared_error',
                                  n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    return -rmse  # BayesianOptimization maximizes


# Define the parameter bounds for Bayesian Optimization
pbounds = {
    'n_estimators': (100, 1000),
    'max_depth': (5, 30),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
}

# Initialize the Bayesian Optimization object
optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds, random_state=42)


# Perform Bayesian Optimization
optimizer.maximize(init_points=15, n_iter=45)

# Get the best parameters and RMSE
print(optimizer.max)

# Train the model with the best parameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

print('best_params: ', best_params)


best_model = RandomForestRegressor(**best_params, random_state=42, criterion='squared_error')
best_model.fit(X_train, y_train)


# Make predictions on the test set
y_pred_best = best_model.predict(test_df)

# Create submission file
sub['Listening_Time_minutes'] = y_pred_best
sub.to_csv('/kaggle/working/submission.csv', index=False)
sub


best_model




