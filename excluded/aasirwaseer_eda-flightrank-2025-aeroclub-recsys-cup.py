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


import pyarrow.parquet as pq

# Read just schema (no data)
train_schema = pq.read_schema('/kaggle/input/aeroclub-recsys-2025/train.parquet')
print("Train schema:\n", train_schema)



cols_to_use = [
    "Id", "ranker_id", "selected", "totalPrice", "taxes", "pricingInfo_passengerCount", 
    "requestDate", "legs0_departureAt", "legs0_duration", "legs1_arrivalAt",
    "frequentFlyer", "isVip", "bySelf", "companyID", "searchRoute", "sex"
]

train_small = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet', columns=cols_to_use).head(100_000)

# Basic EDA
print("Shape:", train_small.shape)
print("Sample:")
print(train_small.head())

# Check unique sessions and group size
print("Unique ranker_id:", train_small['ranker_id'].nunique())
print("Group sizes:")
print(train_small.groupby('ranker_id').size().value_counts().sort_index().head(10))

# Confirm label distribution
print("Selected value counts:")
print(train_small['selected'].value_counts())



import numpy as np

df = train_small.copy()

# Normalize D.HH:MM:SS to P format that pandas can understand
df['duration_mins'] = pd.to_timedelta(df['legs0_duration'].str.replace(r'(\d+)\.(\d+:\d+:\d+)', r'\1 days \2', regex=True)).dt.total_seconds() / 60


# -- Parse datetime
df['departure_dt'] = pd.to_datetime(df['legs0_departureAt'])
df['request_dt'] = pd.to_datetime(df['requestDate'])
df['days_to_departure'] = (df['departure_dt'] - df['request_dt']).dt.days

# -- Time of day (hour)
df['departure_hour'] = df['departure_dt'].dt.hour

# -- Price per passenger
df['price_per_passenger'] = df['totalPrice'] / df['pricingInfo_passengerCount'].clip(lower=1)

# -- Group-wise relative rank features
df['rank_price'] = df.groupby('ranker_id')['totalPrice'].rank(method='min', ascending=True)
df['rank_duration'] = df.groupby('ranker_id')['duration_mins'].rank(method='min', ascending=True)
df['rank_taxes'] = df.groupby('ranker_id')['taxes'].rank(method='min', ascending=True)

# -- Convert frequent flyer (multi airline string) to count
df['ff_count'] = df['frequentFlyer'].fillna('').apply(lambda x: len(str(x).split('/')))

# -- Optional: encode categorical route
df['searchRoute_encoded'] = df['searchRoute'].astype('category').cat.codes

df[['duration_mins', 'days_to_departure', 'departure_hour', 'price_per_passenger', 'rank_price', 'rank_duration', 'ff_count']].describe()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Adjust display
sns.set(style="whitegrid")
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# Assuming df is your DataFrame

# =======================
# 1. Visualizations
# =======================

plt.figure(figsize=(8, 4))
sns.histplot(df['price_per_passenger'], bins=100, kde=True)
plt.title('Price per Passenger Distribution')
plt.xscale('log')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['duration_mins'], bins=100, kde=True)
plt.title('Flight Duration Distribution (minutes)')
plt.xscale('log')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='duration_mins', y='price_per_passenger', data=df, alpha=0.3)
plt.title('Price vs Duration')
plt.xscale('log'); plt.yscale('log')
plt.show()

# =======================
# 2. Feature Engineering
# =======================

df['log_price'] = np.log1p(df['price_per_passenger'])
df['log_duration'] = np.log1p(df['duration_mins'])
df['price_per_min'] = df['price_per_passenger'] / df['duration_mins']

# Arrival hour and overnight flag
df['arrival_hour'] = (df['departure_hour'] + df['duration_mins'] // 60) % 24
df['overnight'] = ((df['departure_hour'] + df['duration_mins'] // 60) >= 24).astype(int)

# Early morning / late night flags
df['is_morning'] = ((df['departure_hour'] >= 5) & (df['departure_hour'] <= 9)).astype(int)
df['is_night'] = ((df['departure_hour'] >= 21) | (df['departure_hour'] <= 4)).astype(int)

# Optional: log-transformed ranks
df['log_rank_price'] = np.log1p(df['rank_price'])
df['log_rank_duration'] = np.log1p(df['rank_duration'])

# =======================
# 3. Correlation Heatmap
# =======================

plt.figure(figsize=(10, 6))
corr = df[['price_per_passenger', 'duration_mins', 'days_to_departure', 
           'departure_hour', 'rank_price', 'rank_duration', 'ff_count',
           'price_per_min', 'overnight', 'is_night', 'is_morning']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# =======================
# 4. Export Features (Optional)
# =======================

useful_cols = [
    'price_per_passenger', 'duration_mins', 'days_to_departure', 'departure_hour',
    'rank_price', 'rank_duration', 'ff_count',
    'price_per_min', 'overnight', 'is_night', 'is_morning',
    'log_price', 'log_duration'
]

df[useful_cols].to_parquet("flight_features_clean.parquet")


