import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from supplemental_english import REGION_CODES, GOVERNMENT_CODES
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


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


train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
train.head()


train.shape


train.isnull().sum()


train.duplicated().sum()


test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
test.head()


test.shape


test.isnull().sum()


test.duplicated().sum()


file_path = os.path.abspath('/kaggle/input/russian-car-plates-prices-prediction/supplemental_english.py') 
sys.path.append(file_path)


max_len = max(len(v) for v in REGION_CODES.values())


for k in REGION_CODES:
    while len(REGION_CODES[k]) < max_len:
        REGION_CODES[k].append(None)


region_codes_df = pd.DataFrame(REGION_CODES)
region_codes_df


region_codes_df = region_codes_df.melt(var_name='region', value_name='region_code').dropna().reset_index(drop=True)
region_codes_df


region_codes_df['region_code'] = region_codes_df['region_code'].astype(str)


region_codes_df.head(5)


region_codes_df.dtypes


records = []

for (letters, (num_from, num_to), region), (description, is_forbidden, road_advantage, significance) in GOVERNMENT_CODES.items():
    records.append({
        "letters": letters,
        "number_from": num_from,
        "number_to": num_to,
        "region": region,
        "description": description,
        "is_forbidden": is_forbidden,
        "road_advantage": road_advantage,
        "significance": significance
    })


records


government_codes_df = pd.DataFrame(records)
government_codes_df.head()


import re

def parse_plate(plate):
    match = re.match(r'([A-Z])(\d{3})([A-Z]{2})(\d{2,3})$', plate)
    if match:
        return match.groups()  # letters1, number, letters2, region_code
    return None, None, None, None


train[['letter1', 'number', 'letter2', 'region_code']] = (train['plate'].apply(lambda p: pd.Series(parse_plate(p))))


test[['letter1', 'number', 'letter2', 'region_code']] = (test['plate'].apply(lambda p: pd.Series(parse_plate(p))))


train.head()


test.head()


train['plate_letters'] = train['letter1'] + train['letter2']


train['plate_number'] = pd.to_numeric(train['number'], errors='coerce')


train['region_code'] = train['region_code'].astype(str)


government_codes_df['region'] = government_codes_df['region'].astype(str)


merged = train.merge(government_codes_df, left_on=['plate_letters', 'region_code'], right_on=['letters', 'region'],
    how='left', suffixes=('', '_gov'))


merged['is_government'] = ((merged['plate_number'] >= merged['number_from']) & (merged['plate_number'] <= merged['number_to']))


merged['is_government'] = merged['is_government'].fillna(False)


train['is_government'] = merged['is_government']
train = train.drop(['letter1', 'letter2', 'number'], axis=1)
train.head()


test['plate_letters'] = test['letter1'] + test['letter2']


test['plate_number'] = pd.to_numeric(test['number'], errors='coerce')


test['region_code'] = test['region_code'].astype(str)


government_codes_df['region'] = government_codes_df['region'].astype(str)


merged = test.merge(government_codes_df, left_on=['plate_letters', 'region_code'], right_on=['letters', 'region'], how='left',
    suffixes=('', '_gov'))


merged['is_government'] = ((merged['plate_number'] >= merged['number_from']) & (merged['plate_number'] <= merged['number_to']))


merged['is_government'] = merged['is_government'].fillna(False)


test['is_government'] = merged['is_government']
test = test.drop(['letter1', 'letter2', 'number'], axis=1)


train = train.merge(region_codes_df, on='region_code', how='left')
test = test.merge(region_codes_df, on='region_code', how='left')


train.head()


test.head()


cols = ['plate_letters', 'is_government', 'region']

for col in cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


all_plates = pd.concat([train['plate'], test['plate']]).astype(str)
all_plates


le = LabelEncoder()
le.fit(all_plates)
train['plate'] = le.transform(train['plate'].astype(str))
test['plate'] = le.transform(test['plate'].astype(str))


def date_change(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df.drop(['date'], axis=1)
    return df


train = date_change(train)
test = date_change(test)


train['region_code'] = train['region_code'].astype('int64')
test['region_code'] = test['region_code'].astype('int64')


train.head()


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    smape_val = np.where(denominator == 0, 0, diff / denominator)
    return np.mean(smape_val) * 100
smape_scorer = make_scorer(smape, greater_is_better=False)


test_ids = test['id']
y = train['price']
X = train[['plate', 'region_code', 'plate_letters', 'plate_number', 'is_government', 'region', 'year', 'month']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train), len(y_train), len(X_test), len(y_test)


correlation = X.assign(target=y).corr()['target'].drop('target')
print(correlation.sort_values(ascending=False))


model = XGBRegressor(colsample_bytree=1.0, gamma=0, learning_rate=0.01, max_depth=6, n_estimators=1500, reg_alpha=0, reg_lambda=2, subsample=0.8)
model.fit(X_train, y_train, verbose=False)
pred = model.predict(X_test)
smape(y_test, pred)


test_sub = test[['plate', 'region_code', 'plate_letters', 'plate_number', 'is_government', 'region', 'year', 'month']]


submission_predictions = model.predict(test_sub)


submission = pd.DataFrame({'id': test_ids.values, 'price': submission_predictions})
submission.head(5)


submission.to_csv('Solution.csv', index = False)




