import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')

df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No':0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No':0})
df['Personality'] = df['Personality'].map({'Extrovert': 1.0, 'Introvert':0.0})

df.head()


df = df.fillna(df.mean(numeric_only=True))
df["extrovert_fetures"] = df[['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']].prod(axis=1)
df["introvert_fetures"] = df[['Time_spent_Alone', 'Stage_fear', 'Drained_after_socializing']].prod(axis=1)
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
df.head()


df['preds'] = np.where(df['extrovert_fetures'] >= df['introvert_fetures'], 1, 0)
df.head()


print(accuracy_score(df.preds, df.Personality))


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')

test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No':0})
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No':0})

test_df = test_df.fillna(test_df.mean(numeric_only=True))
test_df["extrovert_fetures"] = test_df[['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']].prod(axis=1)
test_df["introvert_fetures"] = test_df[['Time_spent_Alone', 'Stage_fear', 'Drained_after_socializing']].prod(axis=1)

scaler = MinMaxScaler()
test_df = pd.DataFrame(scaler.fit_transform(test_df), columns=test_df.columns, index=test_df.index)

test_df['preds'] = np.where(test_df['extrovert_fetures'] >= test_df['introvert_fetures'], 1, 0)

test_df.head()


output = pd.DataFrame({'Personality': test_df.preds})
output['Personality'] = output['Personality'].map({1:"Extrovert", 0:"Introvert"})
output.head()


output.to_csv('submission.csv')

