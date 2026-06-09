import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
df_train.head()


df_train.info()


def preprocess_data(df):    
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns    
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df
    
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)


correlation_values = df_train.corr()['diagnosed_diabetes'].drop('diagnosed_diabetes')
correlation_values.plot(kind='barh', figsize=(10, 6))


def improve_features(df):
    df = df.drop(columns=[
        'smoking_status',
        'ethnicity',
        # 'employment_status',
        # 'income_level', 
        # 'education_level'
        # 'gender',
        # 'sleep_hours_per_day',
        # 'alcohol_consumption_per_week',
    ])

    return df

df_train = improve_features(df_train)
df_test = improve_features(df_test)


correlation_values = df_train.corr()
correlation_values


X = df_train.drop(columns=['diagnosed_diabetes'])
y = df_train['diagnosed_diabetes']


model=CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0
)
model.fit(X,y)
submission=model.predict(df_test)


submission = pd.DataFrame({
    'id': df_sample['id'],
    'diagnosed_diabetes': submission
})
submission.to_csv('submission.csv', index=False)
submission[:5]

