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


import os
print(os.listdir("/kaggle/input"))



import os

# Let's list what's inside the avazu dataset folder
print("Files inside avazu-ctr-prediction:")
print(os.listdir("/kaggle/input/avazu-ctr-prediction"))



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path

# Paths
RAW_PATH = Path("/kaggle/input/avazu-ctr-prediction/train.gz")
PROC_PATH = Path("/kaggle/working/train_processed.parquet")
ENCODERS_PATH = Path("/kaggle/working/encoders.joblib")
SCALER_PATH = Path("/kaggle/working/scaler.joblib")

# Parameters
RANDOM_STATE = 42
SAMPLE_ROWS = None  # set to an integer for faster testing



# Load compressed train file
df = pd.read_csv(RAW_PATH, nrows=SAMPLE_ROWS, compression='gzip')
print("Train shape:", df.shape)
df.head()



# Fill missing values
df.fillna("unknown", inplace=True)






# Target column
target_col = 'click'

# Time column
time_col = 'hour'

# User/Ad columns for aggregates
ad_col = 'id'
user_col = 'device_id'  # Using device_id as user identifier
session_col = None  # Not available



# Convert Avazu 'hour' column to datetime
df['hour'] = df['hour'].astype(str)
df['datetime'] = pd.to_datetime(df['hour'], format='%y%m%d%H', errors='coerce')

# Extract features
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.weekday
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype('int')



# Global click rate
global_ctr = df[target_col].mean()

# Ad popularity (CTR per ad)
ad_stats = df.groupby(ad_col)[target_col].agg(['sum','count']).rename(columns={'sum':'ad_clicks','count':'ad_impr'})
ad_stats['ad_ctr'] = ad_stats['ad_clicks'] / (ad_stats['ad_impr'] + 1e-9)
ad_map = ad_stats['ad_ctr'].to_dict()
df['ad_popularity'] = df[ad_col].map(ad_map).fillna(global_ctr)

# User past CTR (device_id)
user_stats = df.groupby(user_col)[target_col].agg(['sum','count']).rename(columns={'sum':'user_clicks','count':'user_impr'})
user_stats['user_ctr'] = user_stats['user_clicks'] / (user_stats['user_impr'] + 1e-9)
user_map = user_stats['user_ctr'].to_dict()
df['user_past_ctr'] = df[user_col].map(user_map).fillna(global_ctr)

# Session length (not available, set default 1)
df['session_len'] = 1



# Categorical columns
cat_cols = [
    'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
    'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
    'device_model', 'device_type', 'device_conn_type',
    'C14','C15','C16','C17','C18','C19','C20','C21'
]

# Fit label encoders
cat_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    cat_encoders[col] = le

# Save encoders
joblib.dump(cat_encoders, ENCODERS_PATH)
print("Saved categorical encoders to:", ENCODERS_PATH)



# Numeric features
num_cols = ['ad_popularity','user_past_ctr','session_len','hour_of_day','day_of_week']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save scaler
joblib.dump(scaler, SCALER_PATH)
print("Saved scaler to:", SCALER_PATH)



# Save processed train dataset (Parquet for speed & size)
df.to_parquet(PROC_PATH, index=False)
print("Saved processed train dataset to:", PROC_PATH)
df.head()





