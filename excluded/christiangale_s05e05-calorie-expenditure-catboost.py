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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings(action='ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv').set_index('id')


print(train.duplicated().sum())
print(train.info())


def add_feature(df):
    
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['BMR'] = 10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] + np.where(df['Sex'] == 1, 5, -161)
    df['Activity_Level'] = df['Heart_Rate'] * df['Duration']

    return df


train = add_feature(train)
test = add_feature(test)


sns.barplot(data=train, x=train['Sex'].unique(), y=train['Sex'].value_counts())


sns.displot(train['Calories'], kde=True, bins=40)


cat_cols = [i for i in train.drop(columns='Calories', axis=1).columns if train[i].dtype == 'object']
num_cols = [i for i in train.drop(columns='Calories', axis=1).columns if train[i].dtype != 'object']


num_cols


for i in num_cols:
    sns.displot(train[i], kde=True,bins=40, height=4, aspect=1.5)


sns.heatmap(train[
            ['Age',
            'Height',
            'Weight',
            'Duration',
            'Heart_Rate',
            'Body_Temp',
            'BMI',
            'BMR',
            'Activity_Level',
            'Calories']
            ].corr(), annot=True, linewidth=.5, fmt=".2f")


numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

category_transformer = Pipeline([
    ('oe', OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ('cat', category_transformer, cat_cols),
    ('num', numeric_transformer, num_cols)
])


cat_params = {
    'iterations': 1714,
    'learning_rate': 0.02222610449905819,
    'depth': 9,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'verbose': 0,
    'subsample': 0.8721520155095129,
    'l2_leaf_reg': 6.777086298024515,
    "early_stopping_rounds":  61
}


my_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', CatBoostRegressor(**cat_params))
])


X = train.drop(columns='Calories', axis=1)
y = np.log1p(train['Calories'])


kf = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(my_pipe, X, y, cv=kf, scoring='neg_mean_squared_error')
print(np.sqrt(-scores.mean()))


my_pipe.fit(X, y)
y_pred = my_pipe.predict(test)


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission = submission.set_index('id')
submission['Calories'] = np.expm1(y_pred)
submission.to_csv('./submission.csv')

