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
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df.info()


df.describe()


df.isnull().sum()


df.set_index('date')
df.head()
df['num_sold'].plot(figsize=(15,5))


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['num_sold'])


df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day']=df['date'].dt.day
df['dayofweek']=df['date'].dt.dayofweek
df['is_weekend']=df['dayofweek'].apply(lambda x:1 if x>=5 else 0)
df['season']=pd.cut(df['month'],[0,3,6,9,12], labels=['winter','spring','summer','fall'])
df.drop('date', axis=1, inplace=True)



# Tarih sütununu datetime formatına dönüştürme
test['date'] = pd.to_datetime(test['date'])

# Zaman bazlı özellikler ekleme
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek
test['is_weekend'] = test['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# Mevsim sütunu ekleme
test['season'] = pd.cut(test['month'], bins=[0, 3, 6, 9, 12], labels=['winter', 'spring', 'summer', 'fall'])

# Test setinde tarih sütununu kaldırma (modelleme için gereksiz)
test.drop('date', axis=1, inplace=True)

# Test setinin başını kontrol etme
print(test.head())


plt.figure(figsize=(12, 6))
sns.boxplot(x='country', y='num_sold', data=df)
plt.title='Sales by Country'
plt.xlabel='Country'
plt.ylabel='Sales'


avg_sales_by_season=df.groupby('season')['num_sold'].mean()
sns.barplot(x=avg_sales_by_season.index,y=avg_sales_by_season.values)
plt.show()


avg_country=df.groupby('country')['num_sold'].mean()
sns.barplot(x=avg_country.index, y=avg_country.values)


avg_day=df.groupby('dayofweek')['num_sold'].mean()
sns.barplot(x=avg_day.index,y=avg_day.values)


# Kategorik sütunlar
categorical_cols = ['country', 'store', 'product', 'season']

# Eğitim seti üzerinde Get Dummies işlemi
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Test seti üzerinde Get Dummies işlemi
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)




df.head()


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Eksik veri maskesini oluştur
missing_mask = df['num_sold'].isnull()

# Eksik olmayan verilerle model eğitimi
train_data = df[~missing_mask].drop(columns=['num_sold'])
train_target = df[~missing_mask]['num_sold']

# Eksik olan veriler
test_data = df[missing_mask].drop(columns=['num_sold'])

# RandomForestRegressor ile model oluştur
model = RandomForestRegressor(random_state=42)
model.fit(train_data, train_target)

# Eksik olan değerleri tahmin et
df.loc[missing_mask, 'num_sold'] = model.predict(test_data)


df.isnull().sum()


sns.histplot(df['num_sold'], bins=50, kde=True)


df['num_sold_sqrt'] = np.sqrt(df['num_sold'])


sns.histplot(df['num_sold_sqrt'], bins=50, kde=True)



from sklearn.model_selection import train_test_split

# Bağımsız değişkenlerden num_sold ve num_sold_log'u çıkar
X = df.drop(columns=['num_sold','num_sold_sqrt'])  # Hedef değişkenler hariç tüm sütunlar
y = df['num_sold_sqrt']  # SQRT dönüşümlü hedef değişken

# Eğitim ve doğrulama setine ayırma
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Boyutları kontrol etme
print("Eğitim seti boyutu:", X_train.shape, y_train.shape)
print("Doğrulama seti boyutu:", X_val.shape, y_val.shape)



from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Bağımlı ve bağımsız değişkenler
X = df.drop(columns=['num_sold','num_sold_sqrt'])  # Hedef değişken hariç tüm sütunlar
y = df['num_sold_sqrt']  # Hedef değişken: num_sold

# Eğitim ve doğrulama setine ayırma
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM modeli oluştur
model = LGBMRegressor(
    objective='regression',   # Regresyon problemi için uygun hedef
    learning_rate=0.1,       # Öğrenme oranı
    max_depth=8,              # Maksimum derinlik
    n_estimators=1000,        # Ağaç sayısı
    random_state=42           # Rastgelelik için sabit
)

from lightgbm import early_stopping

# Modeli eğitme
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mape',
    callbacks=[early_stopping(stopping_rounds=50)]
)


# Test setindeki tahminler
test_predictions_sqrt = model.predict(test)

# Orijinal ölçeğe geri dönüş (karesini alarak)
test_predictions = np.square(test_predictions_sqrt)


# Tahmin sonuçlarını bir DataFrame'e kaydetme
submission = pd.DataFrame({
    'id': test['id'],            # Test setindeki id sütununu koruyun
    'num_sold': test_predictions  # Orijinal ölçeğe döndürülmüş tahminler
})

# Submission dosyasını CSV olarak kaydetme
submission.to_csv('submission.csv', index=False)

# İlk 5 tahmini kontrol etme
print(submission.head())



