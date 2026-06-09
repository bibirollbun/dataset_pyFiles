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

# Read the data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')


#train_data.head()
X = train_data.copy()
# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['Listening_Time_minutes'], inplace=True)
y = X.Listening_Time_minutes              
X.drop(['Listening_Time_minutes'], axis=1, inplace=True)


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


y


numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].dtype in ['object']]


numeric_cols


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


def get_pipline(n_estimators,learn):
    return Pipeline(steps=[('preprocessor',preprocessor),
                                 ('model',XGBRegressor(n_estimators=n_estimators,
                                                       learning_rate=learn,random_state=0))])
    

def get_score(n_estimators,learn):
    # Replace this body with your own code
    my_pipline = get_pipline(n_estimators,learn)
    scores = -1*cross_val_score(my_pipline,X,y,cv=3,scoring='neg_mean_absolute_error')
    return scores.mean()


my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                                 ('model',XGBRegressor(n_estimators=3,
                                                       learning_rate=0.01,random_state=0))])
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())


#get_score(100,0.05)


# Preprocessing of test data, fit model
my_pipeline = get_pipline(100,0.05)
my_pipeline.fit(X_train, y_train)


preds_test = my_pipeline.predict(test_data) # Your code here
# Save test predictions to file
output = pd.DataFrame({'id': test_data.index,
                       'Listening_Time_minutes': preds_test})
output.to_csv('submission.csv', index=False)

