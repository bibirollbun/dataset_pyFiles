import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_data.head()


test_data.head()


train_data.shape


test_data.shape


train_data.info()


train_data.drop(columns=['id', 'Episode_Title', 'Podcast_Name'], inplace=True)
test_data.drop(columns=['id', 'Episode_Title', 'Podcast_Name'], inplace=True)


common_cols = train_data.columns.intersection(test_data.columns)
categorical_cols = train_data[common_cols].select_dtypes(include=['object']).columns
numerical_cols = train_data[common_cols].select_dtypes(include=['int64', 'float64']).columns


categorical_cols, numerical_cols


train_data.isnull().sum()


test_data.isnull().sum()


#Handle missing values in train data
train_data['Episode_Length_minutes'].fillna(train_data['Episode_Length_minutes'].median(), inplace=True)
train_data['Guest_Popularity_percentage'].fillna(train_data['Guest_Popularity_percentage'].median(), inplace=True)
train_data = train_data[train_data['Number_of_Ads']<10]


#Handle missing values in test data
test_data['Episode_Length_minutes'].fillna(train_data['Episode_Length_minutes'].median(), inplace=True)
test_data['Guest_Popularity_percentage'].fillna(test_data['Guest_Popularity_percentage'].median(), inplace=True)


train_data.isnull().sum()


X = train_data[common_cols]
y = train_data['Listening_Time_minutes']


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor



categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(
    n_estimators=901,
    max_depth=14,
    learning_rate=0.023634721917800305,
    subsample=0.9422518609857451,
    reg_lambda=3.4146535302491694,
    random_state=42
))
])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)


# Load test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_ids = test_df['id']

X_test = test_df[common_cols]


preds = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': preds
})


submission.to_csv("submission.csv", index=False)
submission

