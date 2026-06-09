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


import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')


df.head()


df.info()


df['date'] = pd.to_datetime(df['date'])


df.isnull().sum()


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6])


df.duplicated().sum()


df.groupby(['store', 'item'])['date'].count()



df['sales'].describe()
df['sales'].plot(kind='box')


df['sales'] = df['sales'].astype('int16')


df.describe()


# Zaman serisi grafiği (tüm ürünler ve mağazalar için)
df.groupby('date')['sales'].sum().plot(kind='line', figsize=(12,6))
plt.title('Toplam Satış (Tarih Bazında)')
plt.xlabel('Tarih')
plt.ylabel('Toplam Satış')
plt.show()



# Mağaza bazında satışların toplamı
df.groupby('store')['sales'].sum().plot(kind='bar', figsize=(12,6))
plt.title('Mağaza Bazında Toplam Satış')
plt.xlabel('Mağaza')
plt.ylabel('Toplam Satış')
plt.show()

# Ürün bazında satışların toplamı
df.groupby('item')['sales'].sum().plot(kind='bar', figsize=(12,6))
plt.title('Ürün Bazında Toplam Satış')
plt.xlabel('Ürün')
plt.ylabel('Toplam Satış')
plt.show()



# Haftanın gününe göre satış ortalaması
df.groupby('dayofweek')['sales'].mean().plot(kind='bar', figsize=(12,6))
plt.title('Haftanın Gününe Göre Ortalama Satış')
plt.xlabel('Hafta Günü')
plt.ylabel('Ortalama Satış')
plt.xticks(range(7), ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'])
plt.show()



df['lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
df['lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)
df['lag_30'] = df.groupby(['store', 'item'])['sales'].shift(30)


df['rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].shift(1).rolling(window=7).mean()
df['rolling_std_7'] = df.groupby(['store', 'item'])['sales'].shift(1).rolling(window=7).std()


df['sales_diff'] = df.groupby(['store', 'item'])['sales'].diff()


result = adfuller(df[df['store']==1][df['item']==1]['sales'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")



df[['sales', 'lag_1', 'rolling_mean_7']].corr()


df_model = df.dropna()


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
X = df_model.drop(columns=['sales', 'date', 'lag_1', 'rolling_mean_7'])
y = df_model['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Tahmin ve metrikler
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"XGBoost MAE: {mae_xgb:.2f}")
print(f"XGBoost RMSE: {rmse_xgb:.2f}")



from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

# XGBoost model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Parametre dağılımları
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': randint(1, 10)
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,  # toplam denenecek parametre kombinasyonu
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# En iyi sonuçlar
print("En iyi parametreler:", random_search.best_params_)
print("En iyi MAE (negatif):", random_search.best_score_)



best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Optimize XGBoost MAE: {mae:.2f}")
print(f"Optimize XGBoost RMSE: {rmse:.2f}")



import joblib

# Eğitilmiş modelin kaydedilmesi
joblib.dump(best_model, 'xgboost_model.pkl')



pip install shap


pip install torch --upgrade


import matplotlib.pyplot as plt
from xgboost import plot_importance

# Özellik önem grafiği
plt.figure(figsize=(10, 6))
plot_importance(best_model, max_num_features=10)  # İlk 10 özelliği göster
plt.title("XGBoost Feature Importance")
plt.show()



import shap
import pandas as pd
import matplotlib.pyplot as plt

# SHAP için ağaç tabanlı explainer'ı kullan
explainer = shap.TreeExplainer(best_model)

# X_test'ten örnek al (örneğin 1000 satır)
X_sample = X_test.sample(n=1000, random_state=42)

# SHAP değerlerini hesapla
shap_values = explainer.shap_values(X_sample)

# SHAP özet grafiğini çiz
shap.summary_plot(shap_values, X_sample)


