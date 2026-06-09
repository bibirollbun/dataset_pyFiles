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


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/sample-for-aeroclub-recsys-2025/ranking_sample.csv")
df.head()


from sklearn.preprocessing import MultiLabelBinarizer




print(df.columns.tolist())


from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def process(df):
    df = df.copy()
    
    # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ� durations
    df['legs0_duration'] = pd.to_numeric(df['legs0_duration'], errors='coerce').fillna(0)
    df['legs1_duration'] = pd.to_numeric(df['legs1_duration'], errors='coerce').fillna(0)
    df['total_duration'] = df['legs0_duration'] + df['legs1_duration']

    # tax_ratio = taxes / totalPrice
    df['taxes'] = pd.to_numeric(df['taxes'], errors='coerce').fillna(0)
    df['totalPrice'] = pd.to_numeric(df['totalPrice'], errors='coerce').replace(0, 1)
    df['tax_ratio'] = df['taxes'] / df['totalPrice']

    # days_before_departure = (legs0_departureAt - requestDate).days
    df['legs0_departureAt'] = pd.to_datetime(df['legs0_departureAt'], errors='coerce')
    df['requestDate'] = pd.to_datetime(df['requestDate'], errors='coerce')
    df['days_before_departure'] = (df['legs0_departureAt'] - df['requestDate']).dt.days.fillna(-1)

    # price_per_passenger
    df['pricingInfo_passengerCount'] = pd.to_numeric(df['pricingInfo_passengerCount'], errors='coerce').replace(0, 1)
    df['price_per_passenger'] = df['totalPrice'] / df['pricingInfo_passengerCount']

    # has_baggage â€” ĞµÑ�Ğ»Ğ¸ Ñ�ÑƒĞ¼Ğ¼Ğ°Ñ€Ğ½Ğ¾ Ğ² legs0_segments0_baggageAllowance_quantity ĞµÑ�Ñ‚ÑŒ Ğ±Ğ°Ğ³Ğ°Ğ¶
    df['legs0_segments0_baggageAllowance_quantity'] = pd.to_numeric(
        df['legs0_segments0_baggageAllowance_quantity'], errors='coerce').fillna(0)
    df['has_baggage'] = (df['legs0_segments0_baggageAllowance_quantity'] > 0).astype(int)

    # refundable Ğ¸ exchangeable Ğ¸Ğ· Ñ�Ñ‚Ğ°Ñ‚ÑƒÑ�Ğ¾Ğ² Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»
    df['miniRules0_statusInfos'] = df['miniRules0_statusInfos'].fillna(0).astype(int)
    df['miniRules1_statusInfos'] = df['miniRules1_statusInfos'].fillna(0).astype(int)
    df['is_refundable'] = (df['miniRules0_statusInfos'] != 0).astype(int)
    df['is_exchangeable'] = (df['miniRules1_statusInfos'] != 0).astype(int)

    # Ğ�Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ° frequentFlyer (one-hot)
    df['frequentFlyer'] = df['frequentFlyer'].fillna('None')
    df['frequentFlyer_split'] = df['frequentFlyer'].astype(str).str.split('/')
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    ff_encoded = pd.DataFrame(
        mlb.fit_transform(df['frequentFlyer_split']),
        columns=[f'ff_{c}' for c in mlb.classes_],
        index=df.index
    )
    df = df.drop(columns=['frequentFlyer', 'frequentFlyer_split'])
    df = pd.concat([df, ff_encoded], axis=1)

    # ĞŸÑ€Ğ¸Ğ²ĞµĞ´ĞµĞ½Ğ¸Ğµ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ñ… Ğº int
    for col in ['isVip', 'bySelf']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df


train_df = process(df)


features = [
    'tax_ratio',
    'days_before_departure',
    'price_per_passenger',
    'has_baggage',
    'is_refundable',
    'is_exchangeable',
    'isVip',
    'bySelf',
    'total_duration',
]

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ²Ñ�Ğµ one-hot ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ frequentFlyer
features += [col for col in train_df.columns if col.startswith('ff_')]

X_train = train_df[features]
y_train = train_df['selected']
group_train = train_df.groupby('ranker_id').size().to_numpy()




import lightgbm as lgb

model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[3],
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train, group=group_train)







