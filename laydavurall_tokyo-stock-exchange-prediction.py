# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os

base_path = '/kaggle/input/jpx-tokyo-stock-exchange-prediction'
print("Ana klasör içeriği:", os.listdir(base_path))

# Eğer 'train_files' varsa, onun içeriğine de bakalım
if 'train_files' in os.listdir(base_path):
    print("train_files klasör içeriği:", os.listdir(f"{base_path}/train_files"))
    
import pandas as pd #DataFrame olusturub incelemek icin
import numpy as np #istatistik
import lightgbm as lgb #ai modeli olusturmak
from sklearn.model_selection import train_test_split #veri setini model/egitim olarak ayirmak
from sklearn.metrics import mean_squared_error #hata payi hesaplama (kucukse iyidir)
import joblib #modeli dosyaya kaydetmek
import matplotlib.pyplot as plt #görsellestirmek

#Hatayı görmezden gel
import warnings #basit hataları filtrelemek
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# 1. Veriyi oku
df = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['SecuritiesCode', 'Date'])

# 2. Teknik Göstergeler RSI/MACD
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)  # Sonsuzlukları NaN yap
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    macd_line = macd_line.replace([np.inf, -np.inf], np.nan).fillna(0)
    signal_line = signal_line.replace([np.inf, -np.inf], np.nan).fillna(0)
    macd_hist = macd_hist.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return macd_line, signal_line, macd_hist
    
# 3. Özellik mühendisliği
def create_features(data):
    data = data.copy()
    data['close_to_open'] = data['Close'] / data['Open']
    data['high_to_low'] = data['High'] / data['Low']
    data['return_1d'] = data.groupby('SecuritiesCode')['Close'].pct_change(1, fill_method=None)
    data['ma_5'] = data.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(5).mean())
    data['ma_5_diff'] = data['Close'] - data['ma_5']

    # RSI
    data['rsi'] = data.groupby('SecuritiesCode')['Close'].transform(lambda x: calculate_rsi(x, 14))
    # MACD
    data['macd'] = data.groupby('SecuritiesCode')['Close'].transform(lambda x: calculate_macd(x)[0])
    data['macd_signal'] = data.groupby('SecuritiesCode')['Close'].transform(lambda x: calculate_macd(x)[1])
    data['macd_hist'] = data['macd'] - data['macd_signal']
    return data
df = create_features(df)

# 4. Hedef değişkeni oluştur
df['Target'] = df.groupby('SecuritiesCode')['Target'].shift(-1)

# 5. NaN'leri temizle
features = [
    'close_to_open', 'high_to_low', 'return_1d',
    'ma_5', 'ma_5_diff', 'rsi',
    'macd', 'macd_signal', 'macd_hist'
]
df = df.dropna(subset=features + ['Target'])
df = df.replace([np.inf, -np.inf], np.nan)  # Sonsuzlukları NaN yap
df = df.dropna(subset=features + ['Target'])  # Geçersizleri at

X = df[features]
y = df['Target']

# 6. Eğitim/Test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 7. Model eğitimi
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

param_grid = {
    'num_leaves': [31],
    'learning_rate': [0.1],
    'max_depth': [-1, 10],
    'min_child_samples': [20]
}#Tek çekirdek icin tek column ayırdım yoksa uzun sürüyor 

model = lgb.LGBMRegressor(n_jobs=-1, random_state=42)

tscv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV RMSE:", np.sqrt(-grid.best_score_))

best_model = grid.best_estimator_

# Burada dikkat et! best_model ile predict et
y_pred = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse_test)




# 8. Tahmin ve skor
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# 9. Sonuçları kaydet
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
results.to_csv('predictions_with_technical_indicators.csv', index=False)

# 10. Modeli kaydet
joblib.dump(model, 'lgbm_model_technical.pkl')

# 11. Görselleştirme
plt.figure(figsize=(12, 4))
plt.plot(results['Actual'], label='Actual')
plt.plot(results['Predicted'], label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

# 1. Sektör bilgisini oku
stock_list = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv')

# Sektör bilgisi olarak 'SecuritiesCode' ve '17SectorName' alınıyor
stock_sector = stock_list[['SecuritiesCode', '17SectorName']]

# Tahmin sonuçları dataframe
df_results = df.loc[y_test.index].copy()
df_results['Predicted'] = y_pred

# Sektör bilgisi ile birleştir
df_results = df_results.merge(stock_sector, on='SecuritiesCode', how='left')

# Sektoru grupla ve ortalama al
sector_trends = df_results.groupby('17SectorName')['Predicted'].mean().reset_index()

# Trendleri azalan şekilde sıralamak icin
sector_trends = sector_trends.sort_values(by='Predicted', ascending=False)

print("Sektörel ortalama tahmin trendleri:")
print(sector_trends)

# matplotlib ilse gorsellestirilir
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6)) #12,6 boyutlarında
sector_trends['17SectorName'] = sector_trends['17SectorName'].str.encode('ascii', errors='ignore').str.decode('ascii') # japonca karakter almasın 
plt.bar(sector_trends['17SectorName'], sector_trends['Predicted'], color='pink') #bar grafik cizer
plt.xticks(rotation=45, ha='right')
plt.title('Sektörel Ortalama Tahmin Değerleri')
plt.ylabel('Ortalama Tahmin Edilen Hedef')
plt.tight_layout()
plt.show()




y_pred = best_model.predict(X_test)

import matplotlib.pyplot as plt
# Grafik 1
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Gerçek Değerler')
plt.plot(y_pred, label='Tahminler', alpha=0.7)
plt.title('Gerçek vs Tahmin (LightGBM)')
plt.xlabel('Örnek Numarası')
plt.ylabel('Hedef Değer')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Grafik 2
plt.title("500 veri ile karsilastirma")
plt.plot(y_test.values[:500], label='500 Veri')
plt.plot(y_pred[:500], label='Tahmin', alpha=0.7)

