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


# Importing Libraries

import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Read Training Data

df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df = df.drop(['id'], axis=1)


df.info()


df.describe()


# Differntiate Numeric and Categorical Data

numeric_columns = df.select_dtypes(include='number').columns.tolist()
categorical_columns = df.select_dtypes(exclude='number').columns.tolist()
print("Numeric: ",numeric_columns)
print("Categorical: ",categorical_columns)


# Data Imputation

df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
df[categorical_columns] = df[categorical_columns].fillna(df[numeric_columns].mode().iloc[0])



# Check for null values

df.isna().any()


# Visualize correlation between numeric data

sns.heatmap(df[numeric_columns].corr(),annot=True,cmap='coolwarm')


# Label Encoding for Categorical Data

label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Perform Feature Engineering to produce more valuable information

def feature_engineering(data):

    # Ads density for an episode
    data['Ads_per_minute'] = data['Number_of_Ads']/data['Episode_Length_minutes']

    # Weekend or not
    data['is_Weekend'] = data['Publication_Day'].isin(['Saturday','Sunday']).astype(int)

    # Length of the podcast
    data['length'] = pd.cut(data['Episode_Length_minutes'], bins=[0, 30, 60, 90, 350],labels=['short', 'medium', 'long', 'very_long'])

    return data


# Data preparation

X = df.drop(['Listening_Time_minutes'], axis=1)
y = df['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Perform Feature Engineering on train and test dataset

train = feature_engineering(X_train)
test = feature_engineering(X_test)

train.shape,test.shape


# XGBoost model after Optuna

model = xgb.XGBRegressor(
    booster='gbtree',
    tree_method='hist',
    device='cuda',
    objective='reg:squarederror',
    enable_categorical=True,
    n_estimators=682,
    learning_rate=0.04498875724122863,
    max_depth=12,
    reg_alpha=0.8598352293101645,
    reg_lambda=0.38321932381046153,
    subsample=0.8849420233634961,
    gamma=0.4203805964021932,
    colsample_bytree=0.6105849873359358,
    random_state=42
)


# Training

model.fit(X_train, y_train)


# Prediction on Test Data

# dtest = xgb.DMatrix(X_test, enable_categorical=False)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

rmse


# PREDICTION

test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_df.shape


# Storing id for Submission

ids = test_df['id']
test_df = test_df.drop(['id'], axis=1)
test_df.isna().sum()


# Data Imputation

test_df = test_df.fillna(test_df.median(numeric_only=True))


# Performing Feature Engineering 

test_df = feature_engineering(test_df)


# LabelEncoding Test Dataset

test_label_encoders = {}

for col in test_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])
    test_label_encoders[col] = le


# Predicting on Test Data

y_test_pred = model.predict(test_df)


# Creating a Dataframe for submission

ans = pd.DataFrame({'id':ids,'Listening_Time_minutes':y_test_pred})
ans.to_csv('submission.csv',index=False)

