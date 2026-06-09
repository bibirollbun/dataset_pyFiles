# 'e_users', 'promotion_1', 'promotion_2', 'promotion_3'の分布を確認してみる
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

%matplotlib inline


# データの読み込み
train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv', parse_dates=['datetime'])
test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv', parse_dates=['datetime'])


# train_ofとtest_dfの結合
df = pd.concat([train_df, test_df], axis=0)


# 特徴量エンジニアリング（時刻に対して）
def gene_features(X):
    df = X.copy()

    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['days'] = df['datetime'].dt.dayofyear
    df['week'] = df['datetime'].dt.isocalendar().week
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour

    return df


# 特徴量の作成
total_df = gene_features(df)


# まずは'e_users'の分布を確認
col = 'e_users'
plt.figure(figsize=(10, 5))
sns.histplot(df[col], bins=100)
plt.title(col + ' distribution')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.grid()
plt.show()


# 'promotion_1'の分布を確認
col = 'promotion_1'
plt.figure(figsize=(10, 5))
sns.histplot(df[col], bins=100)
plt.title(col + ' distribution')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.grid()
plt.show()


# 'promotion_2'の分布を確認
col = 'promotion_2'
plt.figure(figsize=(10, 5))
sns.histplot(df[col], bins=100)
plt.title(col + ' distribution')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.grid()
plt.show()


# 'promotion_3'の分布を確認
col = 'promotion_3'
plt.figure(figsize=(10, 5))
sns.histplot(df[col], bins=100)
plt.title(col + ' distribution')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.grid()
plt.show()


col = 'e_users'
cand = (total_df['month'] >= 5) & (total_df['month'] <= 10)
not_busy_season = total_df[cand]
busy_season = total_df[~cand]
plt.figure(figsize=(10, 5))
sns.histplot(not_busy_season[col], bins=60, label='not busy season(5,6,7,8,9,10)')
sns.histplot(busy_season[col], bins=100, label='busy season(1,2,3,4,11,12)')
plt.legend()
plt.show()


col = 'promotion_1'
cand = (total_df['month'] >= 5) & (total_df['month'] <= 10)
not_busy_season = total_df[cand]
busy_season = total_df[~cand]
plt.figure(figsize=(10, 5))
sns.histplot(not_busy_season[col], bins=100, label='not busy season(5,6,7,8,9,10)')
sns.histplot(busy_season[col], bins=100, label='busy season(1,2,3,4,11,12)')
plt.legend()
plt.show()


col = 'promotion_2'
cand = (total_df['month'] >= 5) & (total_df['month'] <= 10)
not_busy_season = total_df[cand]
busy_season = total_df[~cand]
plt.figure(figsize=(10, 5))
sns.histplot(not_busy_season[col], bins=100, label='not busy season(5,6,7,8,9,10)')
sns.histplot(busy_season[col], bins=100, label='busy season(1,2,3,4,11,12)')
plt.legend()
plt.show()


col = 'promotion_3'
cand = (total_df['month'] >= 5) & (total_df['month'] <= 10)
not_busy_season = total_df[cand]
busy_season = total_df[~cand]
plt.figure(figsize=(10, 5))
sns.histplot(not_busy_season[col], bins=100, label='not busy season(5,6,7,8,9,10)')
sns.histplot(busy_season[col], bins=100, label='busy season(1,2,3,4,11,12)')
plt.legend()
plt.show()




