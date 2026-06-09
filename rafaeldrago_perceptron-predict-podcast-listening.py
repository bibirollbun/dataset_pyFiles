import pandas as pd
import numpy as np
import os
import warnings 
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


submission = test.copy()


df


test


df.describe(include='all')


df = df.drop('id',axis=1)
test = test.drop('id',axis=1)


def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':  
            df[column] = df[column].fillna(df[column].mode()[0])  # Fill with mode
        else:
            df[column] = df[column].fillna(df[column].mean())     # Fill with mean
    return df

df = fill_missing_values(df)
test = fill_missing_values(test)


def encode_columns(df, cols_encoder):
    le = LabelEncoder()
    for col in cols_encoder:
        df[col] = le.fit_transform(df[col])
    return df

cols_encoder = ['Podcast_Name', 'Episode_Title']
df = encode_columns(df, cols_encoder)
test = encode_columns(test, cols_encoder)


def encode_categorical_with_dummies(df):
    categorical_cols = df.select_dtypes(exclude='number').columns
    df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_dummies

df = encode_categorical_with_dummies(df)
test = encode_categorical_with_dummies(test)


df


test


X = df.drop('Listening_Time_minutes', axis=1)
y = df['Listening_Time_minutes']

y_test = test


model = Sequential([
    Dense(8,input_dim=26, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
    ])
    
model.summary()


model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)


submission = submission.drop(['Podcast_Name','Episode_Title','Episode_Length_minutes','Genre','Host_Popularity_percentage','Publication_Day','Publication_Time','Guest_Popularity_percentage','Number_of_Ads','Episode_Sentiment'],axis=1)


y_test = model.predict(test).flatten()
submission = pd.DataFrame({
    'id': submission['id'],
    'Listening_Time_minutes': y_test
})

submission.to_csv('submission', index=False)


submission

