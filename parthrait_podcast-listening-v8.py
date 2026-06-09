import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
pd.options.mode.copy_on_write = True

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df = pd.concat([df_train,df_test], axis=0, ignore_index=True)
df.drop(columns=['id'], inplace=True)
df = df.drop_duplicates()


# Outlier removal
df['Episode_Length_minutes'] = np.clip(df['Episode_Length_minutes'], 0, 120)
df['Host_Popularity_percentage'] = np.clip(df['Host_Popularity_percentage'], 20, 100)
df['Guest_Popularity_percentage'] = np.clip(df['Guest_Popularity_percentage'], 0, 100)
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0


# Numerical Features Analysis
num_features = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                'Guest_Popularity_percentage', 'Number_of_Ads']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, feature in enumerate(num_features):
    row = idx // 2
    col = idx % 2
    sns.scatterplot(data=df, x=feature, y='Listening_Time_minutes', 
                    ax=axes[row][col], alpha=0.3)
    axes[row][col].set_title(f'{feature} vs Listening Time')
plt.tight_layout()
plt.show()


# Categorical Features Analysis

cat_features = ['Publication_Day', 'Publication_Time', 'Genre', 'Episode_Sentiment']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, feature in enumerate(cat_features):
    row = idx // 2
    col = idx % 2
    sns.boxplot(data=df, x=feature, y='Listening_Time_minutes', ax=axes[row][col])
    axes[row][col].set_title(f'{feature} Distribution')
    axes[row][col].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()



# Correlation Analysis
plt.figure(figsize=(12, 8))
corr_matrix = df[num_features + ['Listening_Time_minutes']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()


# Efficient Combination Processing
def process_combinations_fast(df, columns_to_encode, pair_size, max_batch_size=2000):
    str_df = df[columns_to_encode].astype(str)
    le = LabelEncoder()
    
    if isinstance(pair_size, int):
        pair_size = [pair_size]

    total_new_cols = 0
    for r in pair_size:
        print(f"\nProcessing {r}-combinations...")
        combos_iter = combinations(columns_to_encode, r)
        n_combinations = np.math.comb(len(columns_to_encode), r)
        print(f"Total {r}-combinations to process: {n_combinations}")

        batch_cols = []
        batch_names = []

        with tqdm(total=n_combinations, desc=f"{r}-combinations") as pbar:
            while True:
                batch_cols.clear()
                batch_names.clear()

                for _ in range(max_batch_size):
                    try:
                        cols = next(combos_iter)
                        batch_cols.append(list(cols))
                        batch_names.append('+'.join(cols))
                    except StopIteration:
                        break

                if not batch_cols:
                    break

                for cols, new_name in zip(batch_cols, batch_names):
                    result = str_df[cols[0]].copy()
                    for col in cols[1:]:
                        result += str_df[col]
                    df[new_name] = le.fit_transform(result) + 1
                    pbar.update(1)

                total_new_cols += len(batch_cols)

        print(f"Completed {r}-combinations. Total columns now: {len(df.columns)}")
    return df


# Encode categorical features
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['Publication_Day'] = df['Publication_Day'].map(day_mapping)

time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
df['Publication_Time'] = df['Publication_Time'].map(time_mapping)

sentiment_map = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)

df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=True).astype(int)

le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col]) + 1

