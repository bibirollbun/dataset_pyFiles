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


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import gc
import os
import sys
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics, preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

from tqdm import tqdm

sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format

print('Files in input folder:', os.listdir("../input"))


def reduceMemUsage(df):
    startMem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        colType = df[col].dtype
    
        if colType != object:
            c_min = df[col].min()
            c_max = df[col].max()
    
            if pd.api.types.is_integer_dtype(colType):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    endMem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage reduced: {startMem:.2f} MB → {endMem:.2f} MB '
          f'({100 * (startMem - endMem) / startMem:.1f}% reduction)')
    
    return df


train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_trn = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
test_trn = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


train_id = reduceMemUsage(train_id)
train_trn = reduceMemUsage(train_trn)
test_id = reduceMemUsage(test_id)
test_trn = reduceMemUsage(test_trn)


print(train_id.shape, test_id.shape)
print(train_trn.shape, test_trn.shape)


[c for c in train_trn.columns if c not in test_trn.columns]


test_trn = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")


test_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")


[c for c in test_trn.columns if c not in test_trn.columns]


fc = train_trn['isFraud'].value_counts(normalize = True).to_frame()
fc.plot.bar()
fc.T


fig, ax = plt.subplots(2, 1, figsize = (16,8))

train_trn['_seq_day'] = train_trn['TransactionDT'] // (24*60*60)
train_trn['_seq_week'] = train_trn['_seq_day'] // 7

train_trn.groupby('_seq_day')['isFraud'].mean().to_frame().plot.line(ax = ax[0])
train_trn.groupby('_seq_week')['isFraud'].mean().to_frame().plot.line(ax = ax[1])

plt.tight_layout(pad = 3.0)
ax[0].set_title('Daily Fraud Rate')
ax[0].set_xlabel('Days Since Start')
ax[0].set_ylabel('Mean isFraud')

ax[1].set_title('Weekly Fraud Rate')
ax[1].set_xlabel('Weeks Since Start')
ax[1].set_ylabel('Mean isFraud')

ax[0].grid(True)
ax[1].grid(True)

sns.set(style='whitegrid')


import datetime
startDate = datetime.datetime.strptime('2024-09-08', "%Y-%m-%d")
lastDate = startDate + datetime.timedelta(seconds = train_trn['TransactionDT'].max())


print(lastDate)


train_trn.head()


train_trn['Date'] = train_trn['TransactionDT'].apply(lambda x: (startDate + datetime.timedelta(seconds = x)))
test_trn['Date'] = test_trn['TransactionDT'].apply(lambda x: (startDate + datetime.timedelta(seconds = x)))


train_trn.head()


test_trn.head()


test_trn['ymd'] = test_trn['Date'].dt.year.astype(str) + '-' + test_trn['Date'].dt.month.astype(str) + '-' + test_trn['Date'].dt.day.astype(str)
test_trn['year_month'] = test_trn['Date'].dt.year.astype(str) + '-' + test_trn['Date'].dt.month.astype(str)
test_trn['weekday'] = test_trn['Date'].dt.dayofweek
test_trn['hour'] = test_trn['Date'].dt.hour
test_trn['day'] = test_trn['Date'].dt.day


train_trn['ymd'] = train_trn['Date'].dt.year.astype(str) + '-' + train_trn['Date'].dt.month.astype(str) + '-' + train_trn['Date'].dt.day.astype(str)
train_trn['year_month'] = train_trn['Date'].dt.year.astype(str) + '-' + train_trn['Date'].dt.month.astype(str)
train_trn['weekday'] = train_trn['Date'].dt.dayofweek
train_trn['hour'] = train_trn['Date'].dt.hour
train_trn['day'] = train_trn['Date'].dt.day


train_trn.head()


test_trn.head()


fig,ax = plt.subplots(4, 1, figsize=(16,12))

train_trn.groupby('weekday')['isFraud'].mean().to_frame().plot.bar(ax=ax[0])
train_trn.groupby('hour')['isFraud'].mean().to_frame().plot.bar(ax=ax[1])
train_trn.groupby('day')['isFraud'].mean().to_frame().plot.bar(ax=ax[2])
train_trn.groupby('year_month')['isFraud'].mean().to_frame().plot.bar(ax=ax[3])

plt.tight_layout(pad = 3.0)


df = train_trn.groupby(['ymd'])['isFraud'].agg(['count','mean','sum'])
df.sort_values(by='mean',ascending=False)[:10].T


df.sort_values(by='count',ascending=False)[:10].T


fig, ax = plt.subplots(2, 1, figsize = (16,12))
ax[0].scatter(df['count'], df['mean'], s=10)
ax[1].scatter(df['count'], df['sum'], s=10)

ax[0].set_xlabel('Count')
ax[0].set_ylabel('Mean')

ax[1].set_xlabel('Count')
ax[1].set_ylabel('Sum')

plt.tight_layout(pad = 3.0)


train_trn['weekday_hour'] = train_trn['weekday'].astype(str) + '_' + train_trn['hour'].astype(str)
train_trn.groupby('weekday_hour')['isFraud'].mean().to_frame().plot.line(figsize=(16,3))


df = train_trn.groupby('weekday')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False)


df = train_trn.groupby('hour')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False).head(10)


df = train_trn.groupby('weekday_hour')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False).head(10)


train_trn['TransactionAmt'] = train_trn['TransactionAmt'].astype('float32')
train_trn['_amount_qcut10'] = pd.qcut(train_trn['TransactionAmt'],10)
df = train_trn.groupby('_amount_qcut10')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False)


test_trn['TransactionAmt'] = test_trn['TransactionAmt'].astype('float32')
test_trn['_amount_qcut10'] = pd.qcut(test_trn['TransactionAmt'],10)


fraud_id = train_trn[train_trn['isFraud'] == 1]['TransactionID']
fraud_id_in_trn = [i for i in fraud_id if i in train_id['TransactionID'].values]
print(f'fraud data count : {len(fraud_id)}, and in trn: {len(fraud_id_in_trn)}')


train_id.head()


train_id.columns


train_id['id_30'].head()


id_cols = [col for col in train_id.columns if col.startswith('id_')]

train_id[id_cols].head().T


train_full = pd.merge(train_trn, train_id, on = 'TransactionID', how = 'inner')


test_full = pd.merge(test_trn, test_id, on = 'TransactionID', how = 'inner')


train_full.head()


test_full.head()


len(train_full)


len(test_full)


train_full_f0 = train_full[train_full['isFraud'] == 0]
train_full_f1 = train_full[train_full['isFraud'] == 1]


def plotHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([train_full_f0[col], train_full_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])


def plotCategoryRateBar(col, topN=np.nan, figsize=(8,3)):
    a, b = train_full_f0, train_full_f1
    if topN == topN: # isNotNan
        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]
        subA = a.loc[a[col].isin(vals.index.values), col]
        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})
    else:
        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})
    df.sort_values('fraud', ascending=False).plot.bar(figsize=figsize)


id_cols = [col for col in train_full.columns if col.startswith('id_')]
print(f'Total id columns: {len(id_cols)}')


from pandas.api.types import is_numeric_dtype

def plot_all_ids(id_cols):
    for col in id_cols:
        print(f'Plotting: {col}')
        try:
            if is_numeric_dtype(train_full[col]):
                plotHistByFraud(col)
                plt.title(f'Histogram of {col} by Fraud')
                plt.show()
            else:
                plotCategoryRateBar(col, topN=10)
                plt.title(f'Category Rate of {col} by Fraud')
                plt.show()
        except Exception as e:
            print(f'Could not plot {col}: {e}')



plot_all_ids(id_cols)


plotCategoryRateBar('DeviceType')
plotCategoryRateBar('DeviceInfo',10)


plotCategoryRateBar('DeviceType')
plotCategoryRateBar('DeviceInfo',10)


train_id.head().T


print(train_full.columns.tolist())


print(test_full.columns.tolist())


v_columns = [col for col in train_trn.columns if col.startswith('V')]

for col in v_columns:
    print(f"\nColumn: {col}")
    print(train_trn[col].head(5))


v_cols = [col for col in train_full.columns if col.startswith('V')]
train_full_NoV = train_full.drop(columns=v_cols)


v_cols = [col for col in test_full.columns if col.startswith('V')]
test_full = test_full.drop(columns=v_cols)


train_full_NoV.head()


test_full.head()


id_columns = [col for col in train_full_NoV.columns if col.startswith('id_')]

cols_to_drop = [col for col in id_columns if col not in ['id_30', 'id_31']]

train_full_NoV = train_full_NoV.drop(columns=cols_to_drop)



id_columns = [col for col in test_full.columns if col.startswith('id-')]

cols_to_drop = [col for col in id_columns if col not in ['id-30', 'id-31']]

test_full = test_full.drop(columns=cols_to_drop)



print(train_full_NoV.columns.tolist())


print(test_full.columns.tolist())


colsCard = [col for col in train_full_NoV.columns if col.startswith('card')]
print(train_full_NoV[colsCard])


train_full_NoV = train_full_NoV.rename(columns={
    'card1': 'card_id',
    'card2': 'issuer_bank_code',
    'card4': 'card_network',
    'card5': 'card_bin',
    'card6': 'card_type'
})

train_full_NoV.drop(columns = ['card3'])


test_full = test_full.rename(columns={
    'card1': 'card_id',
    'card2': 'issuer_bank_code',
    'card4': 'card_network',
    'card5': 'card_bin',
    'card6': 'card_type'
})

test_full.drop(columns = ['card3'])


print(train_full_NoV.columns.tolist())


print(test_full.columns.tolist())


colsCard = [col for col in train_full_NoV.columns if col.startswith('C')]
print(train_full_NoV[colsCard])


correlations = train_full_NoV[[f'C{i}' for i in range(1, 15)] + ['isFraud']].corr()
print(correlations['isFraud'].sort_values(ascending=False))


import seaborn as sns
import matplotlib.pyplot as plt

c_cols = [f'C{i}' for i in range(1, 15)]

plt.figure(figsize=(20, 25))
for i, col in enumerate(c_cols, 1):
    plt.subplot(5, 3, i)
    sns.boxplot(data=train_full_NoV, x='isFraud', y=col)
    plt.title(f"Fraud Distribution vs. {col}")
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()



train_full_NoV = train_full_NoV.rename(columns={
    'C1': 'recent_txn_count',
    'C2': 'card_usage_frequency',
    'C3': 'shared_device_count',
    'C4': 'billing_address_usage',
    'C5': 'shipping_address_usage',
    'C6': 'device_browser_combo_count',
    'C7': 'transaction_type_count',
    'C8': 'device_usage_frequency',
    'C9': 'inactive_device_count',
    'C10': 'merchant_category_count',
    'C11': 'location_terminal_count',
    'C12': 'rolling_txn_count_short_term',
    'C13': 'rolling_txn_count_mid_term',
    'C14': 'rolling_txn_count_long_term'
})


test_full = test_full.rename(columns={
    'C1': 'recent_txn_count',
    'C2': 'card_usage_frequency',
    'C3': 'shared_device_count',
    'C4': 'billing_address_usage',
    'C5': 'shipping_address_usage',
    'C6': 'device_browser_combo_count',
    'C7': 'transaction_type_count',
    'C8': 'device_usage_frequency',
    'C9': 'inactive_device_count',
    'C10': 'merchant_category_count',
    'C11': 'location_terminal_count',
    'C12': 'rolling_txn_count_short_term',
    'C13': 'rolling_txn_count_mid_term',
    'C14': 'rolling_txn_count_long_term'
})


print(train_full_NoV.columns.tolist())


print(test_full.columns.tolist())


d_cols = [col for col in train_full_NoV.columns if col.startswith('D')]

plt.figure(figsize=(20, 25))

for i, col in enumerate(d_cols, 1):
    plt.subplot(5, 3, i)
    sns.boxplot(data=train_full_NoV, x='isFraud', y=col)
    plt.title(f"{col} vs isFraud")
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()



train_full_NoV = train_full_NoV.rename(columns={
    'D1': 'days_since_prev_txn',                        # Short-term transaction gap
    'D2': 'days_since_first_txn',                       # Time since first known txn
    'D3': 'device_session_txn_gap',                     # Session-related txn gap
    'D4': 'txn_gap_same_card',                          # Time delta with same card
    'D5': 'txn_gap_same_billing_addr',                  # Time delta with same billing address
    'D6': 'days_since_last_login',                      # Last login delta
    'D7': 'days_since_last_device_use',                 # Last device use delta
    'D8': 'txn_gap_same_state',                         # Time since transaction in same state
    'D9': 'address_reuse_duration',                     # Possibly reused address duration
    'D10': 'billing_shipping_time_diff',                # Days between billing and shipping
    'D11': 'days_since_card_registration',              # Card registration or first use
    'D12': 'rolling_txn_time_short_term',               # Time feature (short window)
    'D13': 'rolling_txn_time_mid_term',                 # Time feature (mid window)
    'D14': 'rolling_txn_time_long_term',                # Time feature (long window)
    'D15': 'rolling_txn_time_extended'                  # Very long-term user behavior
})


test_full = test_full.rename(columns={
    'D1': 'days_since_prev_txn',                        # Short-term transaction gap
    'D2': 'days_since_first_txn',                       # Time since first known txn
    'D3': 'device_session_txn_gap',                     # Session-related txn gap
    'D4': 'txn_gap_same_card',                          # Time delta with same card
    'D5': 'txn_gap_same_billing_addr',                  # Time delta with same billing address
    'D6': 'days_since_last_login',                      # Last login delta
    'D7': 'days_since_last_device_use',                 # Last device use delta
    'D8': 'txn_gap_same_state',                         # Time since transaction in same state
    'D9': 'address_reuse_duration',                     # Possibly reused address duration
    'D10': 'billing_shipping_time_diff',                # Days between billing and shipping
    'D11': 'days_since_card_registration',              # Card registration or first use
    'D12': 'rolling_txn_time_short_term',               # Time feature (short window)
    'D13': 'rolling_txn_time_mid_term',                 # Time feature (mid window)
    'D14': 'rolling_txn_time_long_term',                # Time feature (long window)
    'D15': 'rolling_txn_time_extended'                  # Very long-term user behavior
})


train_full_NoV.head().T


test_full.head().T


print(train_full_NoV.columns.tolist())


train_full_NoV = train_full_NoV.rename(columns = {
    'id_30' : 'Operating_system',
    'id_31' : 'Browser_type'
})


test_full = test_full.rename(columns = {
    'id-30' : 'Operating_system',
    'id-31' : 'Browser_type'
})


import seaborn as sns
import matplotlib.pyplot as plt

for col in [f'M{i}' for i in range(1, 10)]:
    if train_full_NoV[col].nunique(dropna=True) < 2:
        print(f"Skipping {col} due to insufficient data")
        continue
    plt.figure(figsize=(5, 3))
    sns.barplot(data=train_full_NoV, x=col, y='isFraud', estimator='mean')
    plt.title(f'Fraud Rate by {col}')
    plt.ylabel('Fraud Rate')
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



missing_m_cols = train_full_NoV[[f'M{i}' for i in range(1, 10)]].isnull().mean().sort_values(ascending=False)
print((missing_m_cols * 100).round(2))


print(train_full_NoV['M4'])


train_full_NoV['M4'] = train_full_NoV['M4'].fillna('Unknown')


test_full['M4'] = test_full['M4'].fillna('Unknown')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_full_NoV['M4'] = le.fit_transform(train_full_NoV['M4'])


le = LabelEncoder()
test_full['M4'] = le.fit_transform(test_full['M4'])


print(train_full_NoV['M4'])


print(test_full['M4'])


train_full_NoV.drop(columns = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9', 'card3'], inplace = True)


test_full.drop(columns = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9', 'card3'], inplace = True)


def get_time_bucket(hour):
    hour = int(hour)
    if 5 <= hour < 12: return 'morning'
    elif 12 <= hour < 17: return 'afternoon'
    elif 17 <= hour < 21: return 'evening'
    else: return 'night'

train_full_NoV['hour_bucket'] = train_full_NoV['hour'].astype(int).apply(get_time_bucket)


test_full['hour_bucket'] = test_full['hour'].astype(int).apply(get_time_bucket)


train_full_NoV['_is_weekend'] = train_full_NoV['weekday'].astype(int).isin([5, 6]).astype(int)


test_full['_is_weekend'] = test_full['weekday'].astype(int).isin([5, 6]).astype(int)


hour_counts = train_full_NoV['hour'].value_counts(normalize=True)
train_full_NoV['hour_density'] = train_full_NoV['hour'].map(hour_counts.to_dict())


hour_counts = test_full['hour'].value_counts(normalize=True)
test_full['hour_density'] = test_full['hour'].map(hour_counts.to_dict())


train_full_NoV.drop(['Date', 'ymd', 'year_month', 'day'], axis=1, inplace=True)


test_full.drop(['Date', 'ymd', 'year_month', 'day'], axis=1, inplace=True)


print(train_full_NoV.columns.tolist())


print(test_full.columns.tolist())


# 1. Combine email domain and addr1
train_full_NoV['_P_emaildomain__addr1'] = train_full_NoV['P_emaildomain'].fillna('unknown') + '__' + train_full_NoV['addr1'].astype(str)

# 2. Combine card_id and issuer_bank_code
train_full_NoV['_card_id__issuer'] = train_full_NoV['card_id'].astype(str) + '__' + train_full_NoV['issuer_bank_code'].astype(str)

# 3. Combine card_id and addr1
train_full_NoV['_card_id__addr1'] = train_full_NoV['card_id'].astype(str) + '__' + train_full_NoV['addr1'].astype(str)

# 4. Combine issuer_bank_code and addr1
train_full_NoV['_issuer__addr1'] = train_full_NoV['issuer_bank_code'].astype(str) + '__' + train_full_NoV['addr1'].astype(str)

# 5. Combine full card identifier with addr1
train_full_NoV['_cardid_issuer__addr1'] = train_full_NoV['_card_id__issuer'] + '__' + train_full_NoV['addr1'].astype(str)


# 1. Combine email domain and addr1
test_full['_P_emaildomain__addr1'] = test_full['P_emaildomain'].fillna('unknown') + '__' + test_full['addr1'].astype(str)

# 2. Combine card_id and issuer_bank_code
test_full['_card_id__issuer'] = test_full['card_id'].astype(str) + '__' + test_full['issuer_bank_code'].astype(str)

# 3. Combine card_id and addr1
test_full['_card_id__addr1'] = test_full['card_id'].astype(str) + '__' + test_full['addr1'].astype(str)

# 4. Combine issuer_bank_code and addr1
test_full['_issuer__addr1'] = test_full['issuer_bank_code'].astype(str) + '__' + test_full['addr1'].astype(str)

# 5. Combine full card identifier with addr1
test_full['_cardid_issuer__addr1'] = test_full['_card_id__issuer'] + '__' + test_full['addr1'].astype(str)


cross_features = [
    '_P_emaildomain__addr1',
    '_card_id__issuer',
    '_card_id__addr1',
    '_issuer__addr1',
    '_cardid_issuer__addr1'
]

for col in cross_features:
    freq_map = train_full_NoV[col].value_counts().to_dict()
    train_full_NoV[col + '_freq'] = train_full_NoV[col].map(freq_map)


for col in cross_features:
    freq_map = test_full[col].value_counts().to_dict()
    test_full[col + '_freq'] = test_full[col].map(freq_map)


train_full_NoV['_amount_decimal'] = ((train_full_NoV['TransactionAmt'] - train_full_NoV['TransactionAmt'].astype(int)) * 1000).astype(int)
train_full_NoV['_amount_decimal_len'] = train_full_NoV['TransactionAmt'].apply(
    lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]) if '.' in str(x) else 0
)
train_full_NoV['_amount_fraction'] = train_full_NoV['TransactionAmt'].apply(
    lambda x: float('0.' + re.sub('^[0-9]|\.|0+$', '', str(x))) if '.' in str(x) else 0.0
)


test_full['_amount_decimal'] = ((test_full['TransactionAmt'] - test_full['TransactionAmt'].astype(int)) * 1000).astype(int)
test_full['_amount_decimal_len'] = test_full['TransactionAmt'].apply(
    lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]) if '.' in str(x) else 0
)
test_full['_amount_fraction'] = test_full['TransactionAmt'].apply(
    lambda x: float('0.' + re.sub('^[0-9]|\.|0+$', '', str(x))) if '.' in str(x) else 0.0
)


train_full_NoV.rename(columns={'M4': 'match_status'}, inplace=True)


test_full.rename(columns={'M4': 'match_status'}, inplace=True)


print(train_full_NoV.columns.tolist())


print(test_full.columns.tolist())


train_full_NoV.rename(columns={
    '_P_emaildomain__addr1': 'P_emaildomain_addr1',
    '_card_id__issuer': 'card_id_issuer',
    '_card_id__addr1': 'card_id_addr1',
    '_issuer__addr1': 'issuer_addr1',
    '_cardid_issuer__addr1': 'cardid_issuer_addr1',
    '_P_emaildomain__addr1_freq': 'P_emaildomain_addr1_freq',
    '_card_id__issuer_freq': 'card_id_issuer_freq',
    '_card_id__addr1_freq': 'card_id_addr1_freq',
    '_issuer__addr1_freq': 'issuer_addr1_freq',
    '_cardid_issuer__addr1_freq': 'cardid_issuer_addr1_freq',
    '_amount_decimal': 'amount_decimal',
    '_amount_decimal_len': 'amount_decimal_len',
    '_amount_fraction': 'amount_fraction'
}, inplace=True)


test_full.rename(columns={
    '_P_emaildomain__addr1': 'P_emaildomain_addr1',
    '_card_id__issuer': 'card_id_issuer',
    '_card_id__addr1': 'card_id_addr1',
    '_issuer__addr1': 'issuer_addr1',
    '_cardid_issuer__addr1': 'cardid_issuer_addr1',
    '_P_emaildomain__addr1_freq': 'P_emaildomain_addr1_freq',
    '_card_id__issuer_freq': 'card_id_issuer_freq',
    '_card_id__addr1_freq': 'card_id_addr1_freq',
    '_issuer__addr1_freq': 'issuer_addr1_freq',
    '_cardid_issuer__addr1_freq': 'cardid_issuer_addr1_freq',
    '_amount_decimal': 'amount_decimal',
    '_amount_decimal_len': 'amount_decimal_len',
    '_amount_fraction': 'amount_fraction'
}, inplace=True)


print(train_full_NoV.columns.tolist())


print(test_full.columns.tolist())


numerical_cols = [
    'TransactionAmt', 'recent_txn_count', 'card_usage_frequency', 'shared_device_count',
    'billing_address_usage', 'shipping_address_usage', 'device_browser_combo_count',
    'transaction_type_count', 'device_usage_frequency', 'inactive_device_count',
    'merchant_category_count', 'location_terminal_count',
    'rolling_txn_count_short_term', 'rolling_txn_count_mid_term', 'rolling_txn_count_long_term',
    'days_since_prev_txn', 'days_since_first_txn', 'device_session_txn_gap', 'txn_gap_same_card',
    'txn_gap_same_billing_addr', 'days_since_last_login', 'days_since_last_device_use',
    'txn_gap_same_state', 'address_reuse_duration', 'billing_shipping_time_diff',
    'days_since_card_registration', 'rolling_txn_time_short_term', 'rolling_txn_time_mid_term',
    'rolling_txn_time_long_term', 'rolling_txn_time_extended',
    '_seq_day', '_seq_week', 'hour_density',
    'P_emaildomain_addr1_freq', 'card_id_issuer_freq', 'card_id_addr1_freq',
    'issuer_addr1_freq', 'cardid_issuer_addr1_freq',
    'amount_decimal', 'amount_decimal_len', 'amount_fraction'
]



categorical_cols = [
    'ProductCD', 'card_id', 'issuer_bank_code', 'card_network', 'card_bin', 'card_type',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'weekday', 'hour', 'weekday_hour',
    '_amount_qcut10', 'Operating_system', 'Browser_type', 'DeviceType', 'DeviceInfo',
    'hour_bucket', '_is_weekend', 'match_status',
    'P_emaildomain_addr1', 'card_id_issuer', 'card_id_addr1', 'issuer_addr1', 'cardid_issuer_addr1'
]



from sklearn.preprocessing import LabelEncoder

for col in categorical_cols:
    if col in train_full_NoV.columns and col in test_full.columns:
        le = LabelEncoder()
        
        combined_values = list(train_full_NoV[col].astype(str).values) + list(test_full[col].astype(str).values)
        le.fit(combined_values)
        
        train_full_NoV[col] = le.transform(list(train_full_NoV[col].astype(str).values))
        test_full[col] = le.transform(list(test_full[col].astype(str).values))


low_card_cols = ['hour_bucket', 'match_status', '_is_weekend']

train_full_NoV = pd.get_dummies(train_full_NoV, columns=low_card_cols, drop_first=True)


import pandas as pd

BASE_DATE = pd.to_datetime('2024-09-08')

test_full['Date'] = BASE_DATE + pd.to_timedelta(test_full['TransactionDT'], unit='s')

test_full['weekday'] = test_full['Date'].dt.weekday
test_full['hour'] = test_full['Date'].dt.hour

test_full['weekday_hour'] = test_full['weekday'] * 24 + test_full['hour']



test_full.drop(columns=['Date', 'weekday', 'hour'], inplace=True)


train_cols = set(train_full_NoV.columns)
test_cols = set(test_full.columns)

if train_cols == test_cols:
    print("Both datasets have the same columns.")
else:
    print("Columns differ!")

    print("Columns in train but not in test:")
    print(train_cols - test_cols)

    print("\n Columns in test but not in train:")
    print(test_cols - train_cols)



train_full_NoV.drop(columns = ['_seq_week', 'weekday', '_seq_day', 'hour'], inplace = True)


train_cols = set(train_full_NoV.columns)
test_cols = set(test_full.columns)

if train_cols == test_cols:
    print("Both datasets have the same columns.")
else:
    print("Columns differ!")

    print("Columns in train but not in test:")
    print(train_cols - test_cols)

    print("\n Columns in test but not in train:")
    print(test_cols - train_cols)



def generate_ratio_features(df, value_cols, group_cols, stat_ops):
    for val_col in value_cols:
        if val_col not in df.columns:
            continue
        for grp_col in group_cols:
            if grp_col not in df.columns:
                continue

            df[grp_col] = df[grp_col].astype(str)

            for stat in stat_ops:
                try:
                    stat_series = df.groupby(grp_col)[val_col].transform(stat)
                    feature_name = f'{val_col}_to_{stat}_{grp_col}'
                    df[feature_name] = df[val_col] / stat_series
                except Exception as e:
                    print(f"Skipped {val_col} grouped by {grp_col} ({stat}): {e}")
    return df



value_cols = [
    'TransactionAmt',
    'card_usage_frequency',
    'recent_txn_count',
    'rolling_txn_count_short_term',
    'amount_fraction'
]

group_cols = [
    'card_id',
    'issuer_bank_code',
    'card_network',
    'addr1',
    'P_emaildomain'
]

stat_ops = ['mean', 'std']

train_full_NoV = generate_ratio_features(train_full_NoV, value_cols, group_cols, stat_ops)
test_full = generate_ratio_features(test_full, value_cols, group_cols, stat_ops)


test_full.head().T


def safe_split_email_column(df, col_name):

    split_cols = df[col_name].astype(str).str.split('.', expand=True)

    for i in range(3 - split_cols.shape[1]):
        split_cols[split_cols.shape[1]] = 'None'

    split_cols.columns = [f'{col_name}_{i+1}' for i in range(3)]
    
    return split_cols

p_split_train = safe_split_email_column(train_full_NoV, 'P_emaildomain')
r_split_train = safe_split_email_column(train_full_NoV, 'R_emaildomain')

train_full_NoV = pd.concat([train_full_NoV, p_split_train, r_split_train], axis=1)

p_split_test = safe_split_email_column(test_full, 'P_emaildomain')
r_split_test = safe_split_email_column(test_full, 'R_emaildomain')

test_full = pd.concat([test_full, p_split_test, r_split_test], axis=1)


print(test_full.columns.tolist())


print(train_full_NoV.columns.tolist())


test_full.drop(columns = ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3'], inplace = True)
train_full_NoV.drop(columns = ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3'], inplace = True)


many_null_cols = [col for col in train_full_NoV.columns if train_full_NoV[col].isnull().mean() > 0.9]
many_null_cols_test = [col for col in test_full.columns if test_full[col].isnull().mean() > 0.9]


big_top_value_cols = [
    col for col in train_full_NoV.columns 
    if train_full_NoV[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
]
big_top_value_cols_test = [
    col for col in test_full.columns 
    if test_full[col].value_counts(dropna=False, normalize=True).values[0] > 0.9
]


one_value_cols = [col for col in train_full_NoV.columns if train_full_NoV[col].nunique(dropna=False) <= 1]
one_value_cols_test = [col for col in test_full.columns if test_full[col].nunique(dropna=False) <= 1]


cols_to_drop = set(
    many_null_cols + 
    many_null_cols_test + 
    big_top_value_cols + 
    big_top_value_cols_test + 
    one_value_cols + 
    one_value_cols_test
)


cols_to_drop.discard('isFraud')
cols_to_drop = list(cols_to_drop)


train_full_NoV.drop(columns=[col for col in cols_to_drop if col in train_full_NoV.columns], inplace=True)
test_full.drop(columns=[col for col in cols_to_drop if col in test_full.columns], inplace=True)

print(f"✅ Dropped {len(cols_to_drop)} uninformative columns (where applicable).")


X = train_full_NoV.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train_full_NoV['isFraud']

X_test = test_full.drop(['TransactionDT', 'TransactionID'], axis=1)

test_full = test_full[["TransactionDT", "TransactionID"]]


def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)
X = clean_inf_nan(X)
X_test = clean_inf_nan(X_test)


from sklearn.preprocessing import LabelEncoder

object_cols = list(set(X.select_dtypes(include='object').columns) & set(X_test.select_dtypes(include='object').columns))

le_dict = {}

for col in object_cols:
    le = LabelEncoder()
    combined_values = list(X[col].astype(str).values) + list(X_test[col].astype(str).values)
    le.fit(combined_values)

    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

    le_dict[col] = le 

print(f"Label-encoded {len(object_cols)} object columns.")



from sklearn.preprocessing import LabelEncoder

object_cols = X.select_dtypes(include='object').columns.tolist()

for col in object_cols:
    if col in X.columns and col in X_test.columns:
        le = LabelEncoder()
        le.fit(list(X[col].astype(str).values) + list(X_test[col].astype(str).values))
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

print(f"Label-encoded {len(object_cols)} object columns.")


object_cols = list(set(X.select_dtypes(include='object').columns) & 
                   set(X_test.select_dtypes(include='object').columns))

for col in object_cols:
    le = LabelEncoder()
    combined_vals = list(X[col].astype(str)) + list(X_test[col].astype(str))
    le.fit(combined_vals)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


gc.collect()


print("Train feature count:", X.shape[1])
print("Test feature count:", X_test.shape[1])
print("Train columns not in test:", set(X.columns) - set(X_test.columns))
print("Test columns not in train:", set(X_test.columns) - set(X.columns))


cat_cols = ['hour_bucket', 'match_status', '_is_weekend']
X_test = pd.get_dummies(X_test, columns=cat_cols)
X_test = X_test.reindex(columns=X.columns, fill_value=0)


from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np

def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc',
                               plot_feature_importance=False, verbose=100, early_stopping_rounds=100,
                               n_estimators=5000, averaging='usual', n_jobs=-1):
    
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    models = []
    
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started...')
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=n_jobs)
            model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_metric=eval_metric,
              callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=True)])
            
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test)[:, 1]
        
        oof[valid_index] = y_pred_valid
        scores.append(roc_auc_score(y_valid, y_pred_valid))
        prediction += y_pred
        models.append(model)
    
    prediction /= folds.n_splits
    print('CV mean score:', np.mean(scores))

    result_dict = {
        'oof': oof,
        'prediction': prediction,
        'models': models,
        'scores': scores
    }

    return result_dict



from sklearn.model_selection import KFold
import lightgbm as lgb
import gc

n_fold = 5


from sklearn.model_selection import TimeSeriesSplit
folds = TimeSeriesSplit(n_splits=n_fold)

from sklearn.model_selection import KFold
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

params = {
    'num_leaves': 256,
    'min_child_samples': 79,
    'objective': 'binary',
    'max_depth': 13,
    'learning_rate': 0.03,
    'boosting_type': 'gbdt',
    'subsample_freq': 3,
    'subsample': 0.9,
    'bagging_seed': 11,
    'metric': 'auc',
    'verbosity': -1,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'colsample_bytree': 0.9,
}

result_dict_lgb = train_model_classification(
    X=X,
    X_test=X_test,
    y=y,
    params=params,
    folds=folds,
    model_type='lgb',
    eval_metric='auc',
    plot_feature_importance=True,
    verbose=500,
    early_stopping_rounds=200,
    n_estimators=5000,
    averaging='usual',
    n_jobs=-1
)

gc.collect()



oof = np.zeros(len(X))
from sklearn.metrics import accuracy_score
threshold = 0.5

y_pred_binary = (oof >= threshold).astype(int)

accuracy = accuracy_score(y, y_pred_binary)

print(f"Validation Accuracy (threshold={threshold}): {accuracy:.4f}")


print(X.columns.tolist())


raw_input = {
    'TransactionAmt': 123.45,
    'ProductCD': 'W',
    'card_id': 'C12345',
    'issuer_bank_code': 302,
    'card_network': 'visa',
    'card_bin': 4147,
    'card_type': 'debit',
    'addr1': 325,
    'addr2': 87,
    'dist2': 12.0,
    'P_emaildomain': 'gmail.com',
    'R_emaildomain': 'gmail.com',
    'DeviceType': 'mobile',
    'DeviceInfo': 'Samsung SM-G960F',
    'TransactionDT': 86400
}


def build_model_input(raw_input, le_dict, model_columns, agg_dict):
    import pandas as pd
    import datetime
    df = pd.DataFrame([raw_input])

    base_date = datetime.datetime.strptime("2024-09-08", "%Y-%m-%d")
    df['Date'] = df['TransactionDT'].apply(lambda x: base_date + datetime.timedelta(seconds=x))
    df['hour'] = df['Date'].dt.hour
    df['weekday'] = df['Date'].dt.weekday
    df['_is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    df['hour_bucket'] = df['hour'].map(lambda h: 1 if 6 <= h < 12 else (2 if 12 <= h < 18 else (3 if 18 <= h < 24 else 0)))
    df['match_status'] = df['_is_weekend'] + df['hour_bucket']

    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    for col in ['card_id', 'issuer_bank_code', 'card_network', 'addr1', 'P_emaildomain']:
        for f in ['TransactionAmt', 'card_usage_frequency', 'recent_txn_count',
                  'rolling_txn_count_short_term', 'amount_fraction']:
            for stat in ['mean', 'std']:
                colname = f'{f}_to_{stat}_{col}'
                if colname in model_columns:
                    group_key = df[col].iloc[0]
                    val = agg_dict.get(colname, {}).get(group_key, np.nan)
                    df[colname] = val

    for combo in ['P_emaildomain_addr1', 'card_id_issuer', 'card_id_addr1', 'issuer_addr1', 'cardid_issuer_addr1']:
        df[f'{combo}_freq'] = agg_dict.get(f'{combo}_freq', {}).get(df.get(combo.split('_')[0], 'unknown'), 0)

    df['amount_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
    df['amount_fraction'] = df['TransactionAmt'] % 1
    df['amount_decimal_len'] = df['TransactionAmt'].apply(lambda x: len(str(x).split('.')[-1].rstrip('0')))
    df = pd.get_dummies(df, columns=['hour_bucket', 'match_status', '_is_weekend'])
    df = df.reindex(columns=model_columns, fill_value=0)

    return df


model_input = build_model_input(raw_input, le_dict, model_columns, agg_dict)
prediction = model.predict_proba(model_input)[:, 1][0]
print(f"Fraud probability: {prediction:.4f}")


import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")


