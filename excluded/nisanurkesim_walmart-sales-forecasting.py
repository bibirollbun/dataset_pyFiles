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


#Kütüphaneler
import numpy as np               
import pandas as pd
import datetime as dt   
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.preprocessing import LabelEncoder  
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
import lightgbm as lgb 

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
sns.set_style('whitegrid')

                      
import warnings          
warnings.filterwarnings('ignore')


dp = '/kaggle/input/walmart-recruiting-store-sales-forecasting/'

train = pd.read_csv(dp + 'train.csv.zip', compression='zip')
features = pd.read_csv(dp + 'features.csv.zip', compression='zip')
stores = pd.read_csv(dp + 'stores.csv') 

dp_temp = pd.merge(train, stores, on='Store', how='left')
features_clean = features.drop(columns=['IsHoliday'])
final_dp = pd.merge(dp_temp, features_clean, on=['Store', 'Date'], how='left')

final_dp['Date'] = pd.to_datetime(final_dp['Date'])

final_dp.fillna({
    'MarkDown1': 0, 'MarkDown2': 0, 'MarkDown3': 0, 'MarkDown4': 0, 'MarkDown5': 0
}, inplace=True)

print(f"Orijinal Train Satır Sayısı: {train.shape[0]}")
print(f"Birleştirilmiş Final Satır Sayısı: {final_dp.shape[0]}")
print(f"Sütun Sayısı: {final_dp.shape[1]}")

final_dp.head()


plt.figure(figsize=(15, 12))
#1. grafik
plt.subplot(3, 1, 1) 
sns.histplot(final_dp['Weekly_Sales'], bins=50, color='teal')
plt.title('Haftalık Satış Dağılımı (Hedef Değişken)', fontsize=14)
plt.xlabel('Satış Miktarı ($)')
plt.ylabel('Frekans')

# 2. grafik
plt.subplot(3, 1, 2) 
daily_sales = final_dp.groupby('Date')['Weekly_Sales'].sum().reset_index()
sns.lineplot(data=daily_sales, x='Date', y='Weekly_Sales', color='red', linewidth=2)
plt.title('Zaman İçindeki Toplam Satış Trendi (Mevsimsellik)', fontsize=14)
plt.xlabel('Tarih')
plt.ylabel('Toplam Satış')

# 3. grafik
plt.subplot(3, 1, 3) 
numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
corr_matrix = final_dp[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=14)

plt.tight_layout() 
plt.show()


features_cols = ['Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']
X = final_dp[features_cols]
y = final_dp['Weekly_Sales']

X['IsHoliday'] = X['IsHoliday'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"BASELINE RMSE SKORU: {rmse:.2f}")
print(f"BASELINE MAE SKORU : {mae:.2f}")
print(f" Ortalama {rmse:.0f} dolarlık bir hata payımız var.")


final_dp['Year'] = final_dp['Date'].dt.year
final_dp['Month'] = final_dp['Date'].dt.month
final_dp['Week'] = final_dp['Date'].dt.isocalendar().week 
final_dp['Day'] = final_dp['Date'].dt.day


def get_holiday_type(row):
    if row['IsHoliday'] == 1:
        date_str = str(row['Date']).split(' ')[0]
        
        if date_str in ['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08']:
            return 1 # Super Bowl
        elif date_str in ['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06']:
            return 2 # Labor Day
        elif date_str in ['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']:
            return 3 # Thanksgiving 
        elif date_str in ['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']:
            return 4 # Christmas
    return 0 # Tatil Değil

final_dp['Holiday_Type'] = final_dp.apply(get_holiday_type, axis=1)


type_mapping = {"A": 3, "B": 2, "C": 1}
final_dp['Type_Encoded'] = final_dp['Type'].map(type_mapping)

print(f" Kolonlar: {final_dp.columns.tolist()}")
final_dp.head()


features_cols = [
    'Store', 'Dept', 'Type_Encoded', 'Size',      # Mağaza Özellikleri
    'Week', 'Month', 'Year', 'Holiday_Type',      # Zaman ve Tatil 
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', # Ekonomi
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5' # Promosyonlar
]

X = final_dp[features_cols]
y = final_dp['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(" Feature Eng. + Random Forest")
print(f"RMSE (Hata Payı): {rmse:.2f} $")
print(f"MAE  (Mutlak Hata): {mae:.2f} $")
print(f"Baseline Farkı: {21820 - rmse:.0f} dolar iyileşme")


param_grid = {
    'n_estimators': [50, 100, 200],       # Ağaç sayısı
    'max_depth': [10, 20, None],          # Derinlik
    'min_samples_split': [2, 5, 10]       # Bölünme kuralı
}

#RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator = RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions = param_grid,
    n_iter = 5, # 5 farklı kombinasyon
    cv = 3,     # 3 katlı doğrulama
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='neg_root_mean_squared_error' # Hatayı minimize etmeye çalışmak
)

rf_random.fit(X_train, y_train)
print(f"En İyi Ayarlar: {rf_random.best_params_}")

best_model = rf_random.best_estimator_
y_pred_opt = best_model.predict(X_test)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))

print(f"Optimize Edilmiş RMSE: {rmse_opt:.2f}")




import joblib
# Feature Importance Görselleştirme
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
plt.title('Feature Importance (Model Karar Ağırlıkları)')
plt.show()

# Modeli Kaydetme
model_filename = 'walmart_rf_model_optimized.pkl'
joblib.dump(best_model, model_filename)

print(f" Final Model Kaydedildi: {model_filename}")


