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


# ğŸ“¦ Step 1: Import required libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns



# ğŸ“„ Step 2: Load and preview the dataset
df = pd.read_csv("/kaggle/input/sample/ranking_sample.csv")
df.head()


# ğŸ§¹ Step 3: Basic data cleaning
df.drop_duplicates(inplace=True)

# Convert to datetime
df['requestDate'] = pd.to_datetime(df['requestDate'])
df['legs0_departureAt'] = pd.to_datetime(df['legs0_departureAt'])
df['legs0_arrivalAt'] = pd.to_datetime(df['legs0_arrivalAt'])

if 'legs1_departureAt' in df.columns:
    df['legs1_departureAt'] = pd.to_datetime(df['legs1_departureAt'])
    df['legs1_arrivalAt'] = pd.to_datetime(df['legs1_arrivalAt'])


# ğŸ§  Step 4: Feature engineering
df['total_duration'] = df['legs0_duration']
if 'legs1_duration' in df.columns:
    df['total_duration'] += df['legs1_duration']

df['days_before_departure'] = (df['legs0_departureAt'] - df['requestDate']).dt.days
df['tax_ratio'] = df['taxes'] / df['totalPrice']

# Boolean to int
df['isVip'] = df['isVip'].astype(int)
df['hasFrequentFlyer'] = df['frequentFlyer'].apply(lambda x: int(pd.notnull(x) and str(x).strip() != ''))
df['bySelf'] = df['bySelf'].astype(int)

# Fill missing values
df.fillna(0, inplace=True)


# ğŸ�·ï¸� Step 5: Define features and target
features = [
    'totalPrice', 'taxes', 'tax_ratio', 'total_duration',
    'isVip', 'frequentFlyer', 'bySelf', 'days_before_departure',
    'pricingInfo_passengerCount'
]


df[features].dtypes


df['total_duration'] = pd.to_numeric(df['total_duration'], errors='coerce')  # convert or set NaN
df['total_duration'].fillna(0, inplace=True)  # Ğ·Ğ°Ğ¼ĞµĞ½Ğ¸Ğ¼ NaN Ğ½Ğ° 0


df['frequentFlyer'] = df['frequentFlyer'].apply(lambda x: 0 if pd.isna(x) or x in ['0', '', 'None'] else 1)


df[features].dtypes


# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ğ¸Ğ¼ ĞµÑ�Ñ‚ÑŒ Ğ»Ğ¸ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¸ Ñ�Ñ€ĞµĞ´Ğ¸ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹
print(df['total_duration'].apply(type).value_counts())
print(df['frequentFlyer'].apply(type).value_counts())


X = df[features]
y = df['selected']
group = df.groupby('ranker_id').size().to_list()


# âœ‚ï¸� Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
group_train = df.loc[X_train.index].groupby('ranker_id').size().to_list()


# ğŸš‚ Step 7: Train LightGBM Ranker
ranker = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100
)

ranker.fit(X_train, y_train, group=group_train)


# ğŸ“Š Step 8: Prediction and ranking
df_test = df.loc[X_test.index].copy()
df_test['pred'] = ranker.predict(X_test)
df_test['rank'] = df_test.groupby('ranker_id')['pred'].rank(ascending=False, method='first')
df_test[['ranker_id', 'pred', 'rank']].head(10)


# ğŸ”� Step 9: Feature importance
lgb.plot_importance(ranker, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances")
plt.show()


# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° test-Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
test_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')

print(test_df.shape)
test_df.head()


# ğŸ§¹ Step 3: Basic data cleaning
test_df.drop_duplicates(inplace=True)

# Convert to datetime
test_df['requestDate'] = pd.to_datetime(test_df['requestDate'])
test_df['legs0_departureAt'] = pd.to_datetime(test_df['legs0_departureAt'])
test_df['legs0_arrivalAt'] = pd.to_datetime(test_df['legs0_arrivalAt'])

if 'legs1_departureAt' in test_df.columns:
    test_df['legs1_departureAt'] = pd.to_datetime(test_df['legs1_departureAt'])
    test_df['legs1_arrivalAt'] = pd.to_datetime(test_df['legs1_arrivalAt'])


# ğŸ§  Step 4: Feature engineering
test_df['total_duration'] = test_df['legs0_duration']
if 'legs1_duration' in test_df.columns:
    test_df['total_duration'] += test_df['legs1_duration']

test_df['days_before_departure'] = (test_df['legs0_departureAt'] - test_df['requestDate']).dt.days
test_df['tax_ratio'] = test_df['taxes'] / test_df['totalPrice']

# Boolean to int
test_df['isVip'] = test_df['isVip'].astype(int)
test_df['hasFrequentFlyer'] = test_df['frequentFlyer'].apply(lambda x: int(pd.notnull(x) and str(x).strip() != ''))
test_df['bySelf'] = test_df['bySelf'].astype(int)

# Fill missing values
test_df.fillna(0, inplace=True)


test_df['total_duration'] = pd.to_numeric(test_df['total_duration'], errors='coerce')  # convert or set NaN
test_df['total_duration'].fillna(0, inplace=True)  # Ğ·Ğ°Ğ¼ĞµĞ½Ğ¸Ğ¼ NaN Ğ½Ğ° 0


test_df['frequentFlyer'] = test_df['frequentFlyer'].apply(lambda x: 0 if pd.isna(x) or x in ['0', '', 'None'] else 1)


print(test_df['total_duration'].apply(type).value_counts())
print(test_df['frequentFlyer'].apply(type).value_counts())


# Ğ“Ğ¾Ñ‚Ğ¾Ğ²Ğ¸Ğ¼ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
features = [
    'totalPrice', 'taxes', 'tax_ratio', 'total_duration',
    'isVip', 'frequentFlyer', 'bySelf', 'days_before_departure',
    'pricingInfo_passengerCount'
]

X_test = test_df[features].copy()

# Ğ£Ğ±ĞµĞ´Ğ¸Ğ¼Ñ�Ñ�, Ñ‡Ñ‚Ğ¾ Ğ²Ñ�Ñ‘ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ğ¾Ğµ
for col in X_test.columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)


# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ (Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ´Ğ¾Ğ»Ğ¶Ğ½Ğ° Ğ±Ñ‹Ñ‚ÑŒ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ° Ñ€Ğ°Ğ½ĞµĞµ)
test_df['pred'] = ranker.predict(X_test)

# ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ğµ Ñ€Ğ°Ğ½Ğ³Ğ° Ğ²Ğ½ÑƒÑ‚Ñ€Ğ¸ ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¹ Ğ³Ñ€ÑƒĞ¿Ğ¿Ñ‹
test_df['rank'] = test_df.groupby('ranker_id')['pred'].rank(ascending=False, method='first')



# Ğ¨Ğ°Ğ³ 3: Ğ¤Ğ¸Ğ»ÑŒÑ‚Ñ€Ğ°Ñ†Ğ¸Ñ� top-3
top3_df = test_df[test_df['rank'] <= 3].copy()


top3_df.head()


top3_df.shape


print(len(test_df.groupby('ranker_id')))


print(test_df.groupby('ranker_id').size())


# Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ğ¸Ñ‚ÑŒ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸, Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€, Ñ�Ğ°Ğ¼Ñ‹Ğ¼ Ğ±Ğ¾Ğ»ÑŒÑˆĞ¸Ğ¼ Ñ€Ğ°Ğ½ĞºĞ¾Ğ¼
submission['rank'] = submission['rank'].fillna(9999).astype(int)


submission = test_df[['Id', 'rank']].copy()
submission['rank'] = submission['rank'].astype(int)  # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ rank Ğº Ñ†ĞµĞ»Ğ¾Ğ¼Ñƒ Ñ‚Ğ¸Ğ¿Ñƒ
submission.to_csv('with_traning_set1.csv', index=False)


print(submission.head())
print(submission.dtypes)
print(submission.shape)

