import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.metrics import accuracy_score


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df_train.drop('id' , axis = 1 , inplace =True)
df_test.drop('id' , axis = 1 , inplace =True)


cat_col = df_train.select_dtypes(include='O').columns.tolist()
for col in cat_col:
    if df_train[col].isna().any():
        print(f'{col}: {df_train[col].isna().sum()}')


df_train.dropna(subset=['Stage_fear' ,'Drained_after_socializing'] , axis=0 , inplace=True)


cat_coltest = df_test.select_dtypes(include='O').columns.tolist()
for col in cat_coltest:
    if df_test[col].isna().any():
        print(f'{col}: {df_test[col].isna().sum()}')


df_test['Stage_fear'] = df_test['Stage_fear'].apply(lambda x: df_test['Stage_fear'].mode()[0] if pd.isna(x) else x)
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].apply(lambda x: df_test['Drained_after_socializing'].mode()[0] if pd.isna(x) else x)


num_col = df_train.select_dtypes(include='number').columns.tolist()
for col in num_col:
    if df_train[col].isna().any():
        print(f'{col}: {df_train[col].isna().sum()}')


df_train.dropna(subset=['Time_spent_Alone' , 'Social_event_attendance' , 'Going_outside' ,'Friends_circle_size', 'Post_frequency'],
               axis=0 , inplace=True)


num_coltest = df_test.select_dtypes(include='number').columns.tolist()
for col in num_coltest:
    if df_test[col].isna().any():
        print(f'{col}: {df_test[col].isna().sum()}')


df_test['Time_spent_Alone'] = df_test['Time_spent_Alone'].apply(lambda x: df_test['Time_spent_Alone'].mean() if pd.isna(x) else x)
df_test['Social_event_attendance'] = df_test['Social_event_attendance'].apply(lambda x: df_test['Social_event_attendance'].mean() if pd.isna(x) else x)
df_test['Going_outside'] = df_test['Going_outside'].apply(lambda x: df_test['Going_outside'].mean() if pd.isna(x) else x)
df_test['Friends_circle_size'] = df_test['Friends_circle_size'].apply(lambda x: df_test['Friends_circle_size'].mean() if pd.isna(x) else x)
df_test['Post_frequency'] = df_test['Post_frequency'].apply(lambda x: df_test['Post_frequency'].mean() if pd.isna(x) else x)


df_train['Stage_fear'] = df_train['Stage_fear'].map({'No': 0, 'Yes': 1}).astype(int)
df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].map({'No': 0, 'Yes': 1}).astype(int)


df_test['Stage_fear'] = df_test['Stage_fear'].map({'No': 0, 'Yes': 1}).astype(int)
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].map({'No': 0, 'Yes': 1}).astype(int)


x_train = df_train.drop('Personality' , axis=1)
y_train = df_train['Personality']
x_test = df_test.copy()


pl = make_pipeline( StandardScaler() , LogisticRegression())


pl.fit(x_train , y_train)


pl.score(x_train , y_train)


pred = pl.predict(x_test)


submission = pd.DataFrame({
    'id': range(18524, 18524 + len(x_test)),
    'Personality': pred
})


submission.to_csv('final_submission.csv', index=False)


pd.read_csv('/kaggle/input/final-subm/final_submission.csv')

