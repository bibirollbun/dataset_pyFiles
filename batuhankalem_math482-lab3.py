# --- HÜCRE 1: Kütüphaneler ve Veri Yükleme ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Veri setlerini yükle
train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=['date'])
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date'])
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")

print("Eğitim verisi boyutu:", train_df.shape)
print("Test verisi boyutu:", test_df.shape)


# --- HÜCRE 2: Veriye Genel Bakış ve Eksik Değerler ---

print("--- Veri Tipleri ve Bilgiler ---")
train_df.info()

print("\n--- Eksik Değerler ---")
# 'num_sold' hedef değişkenindeki eksik değerler, muhtemelen satış olmadığı anlamına geliyor.
# Bu yüzden bunları 0 ile dolduracağız.
print(train_df.isnull().sum())
train_df['num_sold'].fillna(0, inplace=True)


# --- HÜCRE 3: Hedef Değişkenin ve Zaman Serisinin Görselleştirilmesi ---

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# 1. Hedef Değişkenin Dağılımı
sns.histplot(train_df['num_sold'], bins=100, kde=True, ax=axes[0])
axes[0].set_title('Satılan Sticker Sayısının Dağılımı (num_sold)')
axes[0].set_xlabel('Satış Adedi')

# 2. Zaman Serisi Grafiği
daily_sales = train_df.groupby('date')['num_sold'].sum()
daily_sales.plot(ax=axes[1], label='Günlük Toplam Satış')
daily_sales.rolling(window=365).mean().plot(ax=axes[1], label='Yıllık Hareketli Ortalama (Trend)', color='red', linestyle='--')
axes[1].set_title('Günlük Toplam Satışlar ve Trend')
axes[1].set_ylabel('Toplam Satış')

plt.tight_layout()
plt.show()


# --- HÜCRE 4: Model 1 - Genel Ortalama Baseline ---

global_mean = train_df['num_sold'].mean()
submission_mean = sample_submission.copy()
submission_mean['num_sold'] = global_mean
submission_mean['num_sold'] = submission_mean['num_sold'].round().astype(int)
submission_mean.to_csv('submission_mean.csv', index=False)
print(f"Genel Ortalama Modeli oluşturuldu. Tahmin: {global_mean:.2f}")


# --- HÜCRE 5: Model 2 - Mevsimsel Ortalama Baseline ---

# Özellik mühendisliği yapmadan önce, bazı temel özelliklerle daha akıllı bir baseline oluşturalım
train_temp = train_df.copy()
test_temp = test_df.copy()

# Haftanın günü özelliğini ekle
train_temp['dayofweek'] = train_temp['date'].dt.dayofweek
test_temp['dayofweek'] = test_temp['date'].dt.dayofweek

# Gruplara göre ortalama satışları hesapla
group_features = ['country', 'store', 'product', 'dayofweek']
average_sales = train_temp.groupby(group_features)['num_sold'].mean()

# Test setine bu ortalamaları ekle
test_preds_seasonal = test_temp.merge(average_sales.reset_index(), on=group_features, how='left')['num_sold']

# Eğitim setinde hiç görülmemiş bir grup varsa, bu değerler NaN olur. Onları genel ortalama ile doldur.
test_preds_seasonal.fillna(global_mean, inplace=True)

# Gönderim dosyasını oluştur
submission_seasonal = sample_submission.copy()
submission_seasonal['num_sold'] = test_preds_seasonal.round().astype(int)
submission_seasonal.to_csv('submission_seasonal.csv', index=False)
print("Mevsimsel Ortalama Modeli oluşturuldu.")


# --- HÜCRE 6: Özellik Mühendisliği (Tüm Veri Üzerinde) ---

# Train ve Test setlerini birleştirerek işlemleri tek seferde ve tutarlı bir şekilde yapalım
# Önce hedef değişkeni ve ID'leri ayıralım
y = train_df['num_sold']
train_ids = train_df['id']
test_ids = test_df['id']

# Birleştirme
df_combined = pd.concat([train_df.drop('num_sold', axis=1), test_df], ignore_index=True)
df_combined = df_combined.sort_values(['country', 'store', 'product', 'date']) # Sıralama önemli!

# 1. Temel Tarih Özellikleri
df_combined['year'] = df_combined['date'].dt.year
df_combined['month'] = df_combined['date'].dt.month
df_combined['day'] = df_combined['date'].dt.day
df_combined['dayofweek'] = df_combined['date'].dt.dayofweek
df_combined['weekofyear'] = df_combined['date'].dt.isocalendar().week.astype(int)

# 2. Gelişmiş Zamansal Özellikler (Lag & Rolling)
# Bu özellikleri hesaplamak için 'num_sold' sütununa ihtiyacımız var.
# Geçici olarak ekleyip, sonra kaldıracağız.
df_combined['num_sold'] = y

group_cols = ['country', 'store', 'product']
# Lag features (7 gün ve 14 gün öncesi)
for lag in [7, 14, 28, 365]:
    df_combined[f'sales_lag_{lag}'] = df_combined.groupby(group_cols)['num_sold'].shift(lag)

# Rolling window features
windows = [7, 28]
for window in windows:
    # Önce hareketli ortalamayı hesapla
    rolling_mean = df_combined.groupby(group_cols)['num_sold'].shift(1).rolling(window=window, min_periods=1).mean()
    df_combined[f'sales_rolling_mean_{window}'] = rolling_mean
    
    # YENİ EKLENEN KISIM: Hareketli ortalamanın türevini (günlük farkını) ekle
    # Bu, son 'window' gündeki satış trendinin yönünü (artıyor mu, azalıyor mu) gösterir.
    df_combined[f'sales_rolling_mean_{window}_diff'] = rolling_mean.diff().fillna(0)
    
    # Hareketli standart sapmayı da ekleyelim
    df_combined[f'sales_rolling_std_{window}'] = df_combined.groupby(group_cols)['num_sold'].shift(1).rolling(window=window, min_periods=1).std()

# Artık 'num_sold' sütununu kaldırabiliriz
df_combined.drop('num_sold', axis=1, inplace=True)

# 3. Kategorik Özellikler
df_combined = pd.get_dummies(df_combined, columns=['country', 'store', 'product'], drop_first=True)

# Oluşturulan yeni özelliklerdeki NaN değerleri (serinin başlangıcı) 0 ile dolduralım
df_combined.fillna(0, inplace=True)

# Orijinal 'date' ve 'id' sütunlarını kaldıralım
df_final = df_combined.drop(columns=['date', 'id'])

# Veriyi tekrar train ve test olarak ayıralım
X_final = df_final.iloc[:len(train_df)]
X_test_final = df_final.iloc[len(train_df):]

print("Tüm özellik mühendisliği adımları tamamlandı.")
print(f"Son eğitim verisi boyutu: {X_final.shape}")


# --- HÜCRE 7: Veriyi Bölme ve MAPE Fonksiyonu ---

# 2016 yılını doğrulama (validation) seti olarak ayıralım
val_year = 2016
train_indices = X_final[X_final['year'] < val_year].index
val_indices = X_final[X_final['year'] == val_year].index

X_train = X_final.loc[train_indices]
y_train = y.loc[train_indices]

X_val = X_final.loc[val_indices]
y_val = y.loc[val_indices]

# Sıfıra bölme hatasını önleyen MAPE fonksiyonu
def calculate_mape_masking(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

print("Eğitim ve doğrulama setleri oluşturuldu.")


# --- HÜCRE 8: Model 3 - LightGBM (Tüm Özelliklerle) ---

# Sabit hiperparametreler
lgbm_params = {
    'objective': 'regression_l1', 'metric': 'mape', 'n_estimators': 2000,
    'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
}

print("LightGBM modelini eğitmeye başlıyoruz...")
model = lgb.LGBMRegressor(**lgbm_params)

# Early stopping ile modeli eğit
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='mape',
          callbacks=[lgb.early_stopping(100, verbose=True)])

# Validation skoru
val_preds = model.predict(X_val)
val_preds[val_preds < 0] = 0
mape_score = calculate_mape_masking(y_val, val_preds)
print(f"\nLightGBM - Validation MAPE: {mape_score:.4f}%")

# Test seti tahminleri
test_preds = model.predict(X_test_final)
test_preds[test_preds < 0] = 0
submission_lgbm = sample_submission.copy()
submission_lgbm['num_sold'] = test_preds.round().astype(int)
submission_lgbm.to_csv('submission_lgbm_final.csv', index=False)
print("submission_lgbm_final.csv oluşturuldu.")


# --- HÜCRE 9: Özellik Önemi Grafiği ---

# Modelin en önemli gördüğü 20 özelliği çizdirelim
lgb.plot_importance(model, figsize=(12, 10), max_num_features=20)
plt.title('LightGBM - En Önemli 20 Özellik')
plt.show()




