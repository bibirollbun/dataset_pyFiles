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
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from datetime import datetime 
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


train = pd.read_csv('/kaggle/input/Store-Sales-Forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/Store-Sales-Forecasting/test.csv', parse_dates=['date'])
stores = pd.read_csv('/kaggle/input/Store-Sales-Forecasting/stores.csv')
oil = pd.read_csv('/kaggle/input/Store-Sales-Forecasting/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('/kaggle/input/Store-Sales-Forecasting/holidays_events.csv', parse_dates=['date'])
transactions = pd.read_csv('/kaggle/input/Store-Sales-Forecasting/transactions.csv', parse_dates=['date'])


train.head()


train.shape


train.info()


train.isnull().sum()


train['family'].value_counts()


stores.head()


stores.info()


stores.isnull().sum()


stores.shape


stores['city'].value_counts()


stores['state'].value_counts()


stores['type'].value_counts()


daily_sales = train.groupby('date')['sales'].sum().reset_index()

plt.figure(figsize=(15, 6))
plt.plot(daily_sales['date'], daily_sales['sales'], linewidth=1)
plt.title('Günlük Toplam Satışlar', fontsize=16, fontweight='bold')
plt.xlabel('Tarih')
plt.ylabel('Satış')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('daily_sales.png', dpi=300, bbox_inches='tight')
plt.show()  # Grafiği görüntüle


plt.figure(figsize=(12, 8))
top_families = train.groupby('family')['sales'].sum().sort_values(ascending=True).tail(15)
top_families.plot(kind='barh')
plt.title('En Çok Satan 15 Ürün Ailesi', fontsize=16, fontweight='bold')
plt.xlabel('Toplam Satış')
plt.ylabel('Ürün Ailesi')
plt.tight_layout()
plt.savefig('top_families.png', dpi=300, bbox_inches='tight') 


plt.figure(figsize=(12, 6))
top_stores = train.groupby('store_nbr')['sales'].sum().sort_values(ascending=False).head(15)
plt.bar(range(len(top_stores)), top_stores.values)
plt.title('En Çok Satan 15 Mağaza', fontsize=16, fontweight='bold')
plt.xlabel('Mağaza Numarası')
plt.ylabel('Toplam Satış')
plt.xticks(range(len(top_stores)), top_stores.index)
plt.tight_layout()
plt.savefig('top_stores.png', dpi=300, bbox_inches='tight')


plt.figure(figsize=(15, 6))
plt.plot(oil['date'], oil['dcoilwtico'], linewidth=1)
plt.title('Petrol Fiyatları (WTI)', fontsize=16, fontweight='bold')
plt.xlabel('Tarih')
plt.ylabel('Fiyat ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('oil_prices.png', dpi=300, bbox_inches='tight')


train['day_of_week'] = train['date'].dt.dayofweek
weekly_sales = train.groupby('day_of_week')['sales'].mean()
days = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

plt.figure(figsize=(10, 6))
plt.bar(range(7), weekly_sales.values)
plt.title('Hafta İçi Ortalama Satışlar', fontsize=16, fontweight='bold')
plt.xlabel('Gün')
plt.ylabel('Ortalama Satış')
plt.xticks(range(7), days, rotation=45)
plt.tight_layout()
plt.savefig('weekly_sales.png', dpi=300, bbox_inches='tight') 


oil.head()


oil.isnull().sum()


oil['dcoilwtico'] = oil['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')


transactions.head()


transactions.isnull().sum()


holidays.head()


holidays.isnull().sum()


holidays['type'].value_counts()


holidays['description'].value_counts()


# Ulusal tatiller
national_holidays = holidays[holidays['locale'] == 'National'].copy()
national_holidays['is_national_holiday'] = 1
national_holidays = national_holidays[['date', 'is_national_holiday']].drop_duplicates()

# Bölgesel tatiller
regional_holidays = holidays[holidays['locale'] == 'Regional'].copy()
regional_holidays['is_regional_holiday'] = 1
regional_holidays = regional_holidays[['date', 'is_regional_holiday']].drop_duplicates()

# Yerel tatiller
local_holidays = holidays[holidays['locale'] == 'Local'].copy()
local_holidays['is_local_holiday'] = 1
local_holidays = local_holidays[['date', 'is_local_holiday']].drop_duplicates()

# Çalışma günü mi tatil mi
work_holidays = holidays[holidays['type'] == 'Work Day'].copy()
work_holidays['is_work_day'] = 1
work_holidays = work_holidays[['date', 'is_work_day']].drop_duplicates()

# Transfer edilmiş tatil
transferred_holidays = holidays[holidays['transferred'] == True].copy()
transferred_holidays['is_transferred'] = 1
transferred_holidays = transferred_holidays[['date', 'is_transferred']].drop_duplicates()


train = train.merge(stores, on='store_nbr', how='left')
train = train.merge(oil, on='date', how='left')
train = train.merge(transactions, on=['date', 'store_nbr'], how='left')


# Tatil bilgilerini ekle
train = train.merge(national_holidays, on='date', how='left')
train = train.merge(regional_holidays, on='date', how='left')
train = train.merge(local_holidays, on='date', how='left')
train = train.merge(work_holidays, on='date', how='left')
train = train.merge(transferred_holidays, on='date', how='left')

# Tatil sütunlarındaki NaN'ları 0 ile doldur
train['is_national_holiday'] = train['is_national_holiday'].fillna(0)
train['is_regional_holiday'] = train['is_regional_holiday'].fillna(0)
train['is_local_holiday'] = train['is_local_holiday'].fillna(0)
train['is_work_day'] = train['is_work_day'].fillna(0)
train['is_transferred'] = train['is_transferred'].fillna(0)


test = test.merge(stores, on='store_nbr', how='left') 
test = test.merge(oil, on='date', how='left') 
test = test.merge(transactions, on=['date', 'store_nbr'], how='left')


# Tatil bilgileri
test = test.merge(national_holidays, on='date', how='left')
test = test.merge(regional_holidays, on='date', how='left')
test = test.merge(local_holidays, on='date', how='left')
test = test.merge(work_holidays, on='date', how='left')
test = test.merge(transferred_holidays, on='date', how='left')

test['is_national_holiday'] = test['is_national_holiday'].fillna(0)
test['is_regional_holiday'] = test['is_regional_holiday'].fillna(0)
test['is_local_holiday'] = test['is_local_holiday'].fillna(0)
test['is_work_day'] = test['is_work_day'].fillna(0)
test['is_transferred'] = test['is_transferred'].fillna(0)


# TRAIN için tarih özellikleri
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['day_of_week'] = train['date'].dt.dayofweek
train['day_of_year'] = train['date'].dt.dayofyear
train['week_of_year'] = train['date'].dt.isocalendar().week
train['quarter'] = train['date'].dt.quarter
train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)
train['is_month_start'] = train['date'].dt.is_month_start.astype(int)
train['is_month_end'] = train['date'].dt.is_month_end.astype(int)
train['is_quarter_start'] = train['date'].dt.is_quarter_start.astype(int)
train['is_quarter_end'] = train['date'].dt.is_quarter_end.astype(int)


# TEST için tarih özellikleri
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.dayofweek
test['day_of_year'] = test['date'].dt.dayofyear
test['week_of_year'] = test['date'].dt.isocalendar().week
test['quarter'] = test['date'].dt.quarter
test['is_weekend'] = (test['day_of_week'] >= 5).astype(int)
test['is_month_start'] = test['date'].dt.is_month_start.astype(int)
test['is_month_end'] = test['date'].dt.is_month_end.astype(int)
test['is_quarter_start'] = test['date'].dt.is_quarter_start.astype(int)
test['is_quarter_end'] = test['date'].dt.is_quarter_end.astype(int)


train = train.sort_values(['store_nbr', 'family', 'date'])

# 7 gün önceki satışlar
train['sales_lag_7'] = train.groupby(['store_nbr', 'family'])['sales'].shift(7)

# 14 gün önceki satışlar
train['sales_lag_14'] = train.groupby(['store_nbr', 'family'])['sales'].shift(14)

# 28 gün önceki satışlar
train['sales_lag_28'] = train.groupby(['store_nbr', 'family'])['sales'].shift(28)

# 7 günlük hareketli ortalama
train['sales_rolling_mean_7'] = train.groupby(['store_nbr', 'family'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
)

# 14 günlük hareketli ortalama
train['sales_rolling_mean_14'] = train.groupby(['store_nbr', 'family'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
)

# 28 günlük hareketli ortalama
train['sales_rolling_mean_28'] = train.groupby(['store_nbr', 'family'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=28, min_periods=1).mean()
)


categorical_cols = ['family', 'city', 'state', 'type', 'cluster']


label_encodings = {}
for col in categorical_cols: 
    all_values = pd.concat([train[col], test[col]]).unique()
    label_encodings[col] = {val: idx for idx, val in enumerate(all_values)}
     
    train[f'{col}_encoded'] = train[col].map(label_encodings[col])
    test[f'{col}_encoded'] = test[col].map(label_encodings[col])



train['transactions'] = train['transactions'].fillna(train['transactions'].median())
test['transactions'] = test['transactions'].fillna(train['transactions'].median())


lag_cols = [col for col in train.columns if 'lag' in col or 'rolling' in col]
for col in lag_cols:
    train[col] = train[col].fillna(0)


train.loc[train['sales'] < 0, 'sales'] = 0


train.to_csv('train_processed.csv', index=False)
test.to_csv('test_processed.csv', index=False)


import pickle
with open('label_encodings.pkl', 'wb') as f:
    pickle.dump(label_encodings, f)


feature_cols = [col for col in train.columns if col not in 
                ['id', 'date', 'sales', 'family', 'city', 'state', 'type']]


# Kullanılmayacak sütunlar
drop_cols = ['id', 'date', 'sales', 'family', 'city', 'state', 'type']

# Mevcut sütunlardan olmayanları çıkar
drop_cols = [col for col in drop_cols if col in train.columns]

# Özellik sütunları
feature_cols = [col for col in train.columns if col not in drop_cols]


X = train[feature_cols]
y = train['sales']


X.shape


y.shape


# Zaman serisinde rastgele split yapamayız, son günleri validation olarak ayırıyoruz
# Son 15 günü validation için ayır
train_dates = train['date'].unique()
train_dates = sorted(train_dates)


split_date = train_dates[-16]  # Son 15 gün validation

train_mask = train['date'] < split_date
val_mask = train['date'] >= split_date


X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[val_mask]
y_val = y[val_mask]


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1
}


#  LIGHTGBM DATASET OLUŞTURMA
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)


model = lgb.train(
    params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)


# Validation tahminleri
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)

# Negatif tahminleri 0 yap
y_val_pred = np.maximum(y_val_pred, 0)


from sklearn.metrics import mean_squared_error, mean_absolute_error


# Metrikler
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1))) * 100


# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
})
feature_importance = feature_importance.sort_values('importance', ascending=False)


# Feature importance grafiği
plt.figure(figsize=(10, 12))
top_features = feature_importance.head(30)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance (Gain)')
plt.title('Top 30 Özellik Önemliliği', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ feature_importance.png kaydedildi")


# Rastgele bir örnek seç (belirli bir mağaza ve ürün)
sample_store = train['store_nbr'].mode()[0]
sample_family_col = [col for col in train.columns if 'family_encoded' in col][0]
sample_family = train[sample_family_col].mode()[0]

sample_mask = (train['store_nbr'] == sample_store) & (train[sample_family_col] == sample_family)
sample_data = train[sample_mask].copy()
sample_data = sample_data.sort_values('date')

# Son 60 günü al
sample_data = sample_data.tail(60)

# Bu sample için tahmin yap
sample_X = sample_data[feature_cols]
sample_y_true = sample_data['sales'].values
sample_y_pred = model.predict(sample_X, num_iteration=model.best_iteration)
sample_y_pred = np.maximum(sample_y_pred, 0)


plt.figure(figsize=(15, 6))
plt.plot(sample_data['date'], sample_y_true, label='Gerçek Satış', marker='o', linewidth=2)
plt.plot(sample_data['date'], sample_y_pred, label='Tahmin', marker='s', linewidth=2, alpha=0.7)
plt.xlabel('Tarih')
plt.ylabel('Satış')
plt.title(f'Gerçek vs Tahmin (Mağaza {sample_store}, Son 60 Gün)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')




