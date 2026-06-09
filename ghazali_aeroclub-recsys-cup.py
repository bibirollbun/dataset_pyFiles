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


!pip install lightgbm pyarrow


import pandas as pd

train_path = '/kaggle/input/aeroclub-recsys-2025/train.parquet'
test_path = '/kaggle/input/aeroclub-recsys-2025/test.parquet'

important_columns = [
    'Id', 'ranker_id', 'selected', 'totalPrice', 'taxes',
    'frequentFlyer', 'isVip', 'bySelf', 'isAccess3D',
    'legs0_duration', 'legs1_duration', 
    'pricingInfo_isAccessTP', 'pricingInfo_passengerCount'
]

train_df = pd.read_parquet(train_path, columns=important_columns)
test_df = pd.read_parquet(train_path, columns=[col for col in important_columns if col != 'selected'])


from sklearn.preprocessing import LabelEncoder

exclude_cols = [
    'Id', 'ranker_id', 'profileId', 'requestDate', 'searchRoute',
    'legs0_arrivalAt', 'legs0_departureAt', 'legs1_arrivalAt', 'legs1_departureAt',
    'selected'  # only in train
]

feature_cols = [col for col in train_df.columns if col not in exclude_cols]

train_df[feature_cols] = train_df[feature_cols].fillna(-1)
test_df[feature_cols] = test_df[feature_cols].fillna(-1)

object_cols = train_df.select_dtypes(include=['object', 'bool']).columns

from sklearn.preprocessing import LabelEncoder
import hashlib

cat_cols = train_df.select_dtypes(include=['object', 'bool']).columns

low_card_cols = [col for col in cat_cols if train_df[col].nunique() <= 200]
high_card_cols = [col for col in cat_cols if col not in low_card_cols]

for col in low_card_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

def hash_encode(series, n_buckets=1000):
    return series.astype(str).apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % n_buckets)

for col in high_card_cols:
    train_df[col] = hash_encode(train_df[col])
    test_df[col] = hash_encode(test_df[col])


from lightgbm import LGBMRanker

group_sizes = train_df.groupby('ranker_id').size()
large_groups = group_sizes[group_sizes > 10000].index
print(f"Removing {len(large_groups)} oversized groups")

train_df = train_df[~train_df['ranker_id'].isin(large_groups)]

X_train = train_df[feature_cols]
y_train = train_df['selected']
group_train = train_df.groupby('ranker_id').size().values

ranker = LGBMRanker(
    n_estimators=100,
    random_state=42,
    objective='lambdarank'  
)

ranker.fit(X_train, y_train, group=group_train)



X_test = test_df[feature_cols]
test_df['score'] = ranker.predict(X_test)

test_df['rank'] = test_df.groupby('ranker_id')['score'].rank(ascending=False, method='first').astype(int)



submission_df = test_df[['Id', 'ranker_id', 'rank']].copy()
submission_df.rename(columns={'rank': 'selected'}, inplace=True)

submission_df.to_csv("/kaggle/working/submission.csv", index=False)


def hit_rate_at_3(df, true_col='selected', rank_col='rank'):
    hit_count = 0
    total = 0

    for _, group in df.groupby('ranker_id'):
        # Get all rows where selected == 1
        true_rows = group[group[true_col] == 1]
        top3 = group.nsmallest(3, rank_col)

        if not true_rows.empty:
            total += 1
            # Check if any of the true rows are in top-3
            if true_rows.index[0] in top3.index:
                hit_count += 1

    if total == 0:
        print("⚠️ No valid ranker_id groups with selected=1 found. HitRate@3 undefined.")
        return 0.0

    return hit_count / total

# --- Evaluate (safe check) ---
if 'selected' in test_df.columns:
    hr3 = hit_rate_at_3(test_df)
    print(f"✅ HitRate@3: {hr3:.4f}")
else:
    print("⚠️ 'selected' column not found in test_df. Cannot compute HitRate@3.")

