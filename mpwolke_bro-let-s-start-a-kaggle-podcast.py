# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.tail(3)


train.info()


train.isnull().sum()


test.isnull().sum()


train.describe().loc[['mean','min','max']].T


categorical_columns = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


#By Dustin Ober https://www.kaggle.com/code/dustinober/kjv-eda

for col in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train, x=col)
    plt.xticks(rotation=40)
    plt.title(f'Distribution of {col}')
    plt.show()


numerical_columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']


#By Dustin Ober https://www.kaggle.com/code/dustinober/kjv-eda

for col in numerical_columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(data=train, x=col, kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()


# columns with NaN values 
cols_fillna = ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads"]

# replace 'NaN' with '0' in these columns
for col in cols_fillna:
    train[col].fillna('0',inplace=True)
    test[col].fillna('0',inplace=True)


# Check if there are any missing values left
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head()


import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


X = train.drop(columns=['Listening_Time_minutes'])
y = train[['Listening_Time_minutes']]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#By Alaa Sweed  https://www.kaggle.com/code/alaasweed/playground-s5e4-how-long-do-you-listen/notebook

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()


# pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    ]
)



#By Kunal Aldar https://www.kaggle.com/code/kunalaldar/crop-yield-prediction-rf-regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# Feature selection using KBest method to reduce curse of dimensionality
kbest = SelectKBest(score_func=f_regression, k=5)


# After hyperparameter tuning best parameters are selected to reduce runtime and increse performance of model
rf = RandomForestRegressor(max_features=0.75, max_samples=0.75, n_estimators=400, n_jobs=-1)


#By Kunal Aldar https://www.kaggle.com/code/kunalaldar/crop-yield-prediction-rf-regressor

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kbest', kbest),
    ('RF_regressor', rf)
])


pipe.fit(X_train,y_train.values.ravel())


y_pred = pipe.predict(X_test)


from sklearn.metrics import accuracy_score, r2_score
r2_score(y_test,y_pred)

