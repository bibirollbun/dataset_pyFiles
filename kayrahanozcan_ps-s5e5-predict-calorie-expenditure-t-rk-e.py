# Gerekli KÃ¼tÃ¼phaneleri YÃ¼kleyelim
import pandas as pd
import numpy as np
import lightgbm as lgb # HÄ±zlÄ± ve etkili bir model seÃ§imi
from sklearn.model_selection import train_test_split # Ä°steÄŸe baÄŸlÄ± olarak yerel doÄŸrulama iÃ§in
from sklearn.metrics import mean_squared_log_error # DeÄŸerlendirme metriÄŸimiz iÃ§in

# RMSLE metrik fonksiyonu (sklearn'de zaten var ama kullanÄ±mÄ± hatÄ±rlatmak iÃ§in)
# Sklearn'deki mean_squared_log_error zaten log(1+y_true) ve log(1+y_pred) arasÄ±ndaki kare farklarÄ±nÄ±n ortalamasÄ±nÄ± alÄ±r.
# Bizim yapmamÄ±z gereken, modelimizi log(1+Calories) Ã¼zerine eÄŸitmek veya
# modelimizin Ã§Ä±ktÄ±sÄ±nÄ± RMSLE metrik fonksiyonuna uygun ÅŸekilde kullanmaktÄ±r.
# Genellikle hedef deÄŸiÅŸkeni log(1+y) olarak dÃ¶nÃ¼ÅŸtÃ¼rÃ¼p RMSE modellemek RMSLE iÃ§in iyi sonuÃ§ verir.
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# --- 1. Veri YÃ¼kleme ---
print("Veriler yÃ¼kleniyor...")
# Veri yollarÄ±nÄ± belirtiyoruz
train_path = '/kaggle/input/playground-series-s5e5/train.csv'
test_path = '/kaggle/input/playground-series-s5e5/test.csv'
sample_submission_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'

# CSV dosyalarÄ±nÄ± DataFrame'lere yÃ¼klÃ¼yoruz
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(sample_submission_path)

print("Veriler baÅŸarÄ±yla yÃ¼klendi.")

# Veri setlerine ilk bakÄ±ÅŸ
print("\nEÄŸitim verisi (train_df) ilk 5 satÄ±r:")
print(train_df.head())

print("\nTest verisi (test_df) ilk 5 satÄ±r:")
print(test_df.head())

print("\nÃ–rnek gÃ¶nderim dosyasÄ± (sample_submission_df) ilk 5 satÄ±r:")
print(sample_submission_df.head())

# Veri setlerinin boyutlarÄ±
print(f"\nEÄŸitim verisi boyutu: {train_df.shape}")
print(f"Test verisi boyutu: {test_df.shape}")

# --- 2. KeÅŸifsel Veri Analizi (EDA) - Temel BakÄ±ÅŸ ---
print("\nEÄŸitim verisi hakkÄ±nda bilgi:")
train_df.info()

print("\nEÄŸitim verisi istatistikleri:")
print(train_df.describe())

print("\nTest verisi hakkÄ±nda bilgi:")
test_df.info() # Test verisinde 'Calories' sÃ¼tunu olmadÄ±ÄŸÄ±nÄ± kontrol edin, bu hedefimizdir.

# Eksik deÄŸer kontrolÃ¼ (BaÅŸlangÄ±Ã§ iÃ§in genellikle bu yarÄ±ÅŸmalarda eksik deÄŸer olmaz, ama kontrol etmek iyidir)
print("\nEÄŸitim verisindeki eksik deÄŸerlerin toplamÄ±:")
print(train_df.isnull().sum().sum()) # Toplam eksik deÄŸer sayÄ±sÄ±

print("\nTest verisindeki eksik deÄŸerlerin toplamÄ±:")
print(test_df.isnull().sum().sum())

# Hedef deÄŸiÅŸkenin daÄŸÄ±lÄ±mÄ± (Calories)
print("\n'Calories' hedef deÄŸiÅŸkeni daÄŸÄ±lÄ±mÄ± istatistikleri:")
print(train_df['Calories'].describe())

# RMSLE metriÄŸi iÃ§in hedef deÄŸiÅŸkeni log(1+y) formatÄ±na dÃ¶nÃ¼ÅŸtÃ¼rmek yaygÄ±n bir stratejidir.
# Hedef deÄŸiÅŸkenin daÄŸÄ±lÄ±mÄ±na bakarak bu dÃ¶nÃ¼ÅŸÃ¼mÃ¼n uygun olup olmadÄ±ÄŸÄ±nÄ± deÄŸerlendirebilirsiniz.
# Genellikle saÄŸa Ã§arpÄ±k (skewed) daÄŸÄ±lÄ±mlar iÃ§in etkilidir.
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.histplot(train_df['Calories'], kde=True)
# plt.title('Calories DaÄŸÄ±lÄ±mÄ±')
# plt.show()
#
# sns.histplot(np.log1p(train_df['Calories']), kde=True, color='green') # log1p = log(1+x)
# plt.title('log(1+Calories) DaÄŸÄ±lÄ±mÄ±')
# plt.show()
# DaÄŸÄ±lÄ±ma bakarak log dÃ¶nÃ¼ÅŸÃ¼mÃ¼nÃ¼n faydalÄ± olacaÄŸÄ±nÄ± gÃ¶rebiliriz.

# --- 3. Veri Ã–n Ä°ÅŸleme ---
print("\nVeri Ã¶n iÅŸleme baÅŸlatÄ±lÄ±yor...")

# id sÃ¼tununu Ã¶zelliklerden ayÄ±rÄ±yoruz, Ã§Ã¼nkÃ¼ model eÄŸitimi iÃ§in gerekli deÄŸil
train_ids = train_df['id']
test_ids = test_df['id']

train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Hedef deÄŸiÅŸkeni (Calories) ayÄ±rÄ±p log(1+Calories) dÃ¶nÃ¼ÅŸÃ¼mÃ¼nÃ¼ uyguluyoruz
y = np.log1p(train_df['Calories'])
X = train_df.drop('Calories', axis=1) # Ã–zelliklerimiz

# Test veri seti
X_test = test_df.copy()

# Kategorik deÄŸiÅŸkenleri belirleyelim (Genellikle 'Gender' gibi)
# Veri setini inceleyerek veya .info() Ã§Ä±ktÄ±sÄ±na bakarak kategorik sÃ¼tunlarÄ± bulabilirsiniz.
# Bu veri setinde 'Gender' sÃ¼tunu kategorik olabilir.
categorical_features = X.select_dtypes(include=['object']).columns
print(f"\nBulunan kategorik sÃ¼tunlar: {list(categorical_features)}")

# Kategorik deÄŸiÅŸkenlere One-Hot Encoding uygulayalÄ±m
# Bu, makine Ã¶ÄŸrenimi modellerinin anlayabileceÄŸi sayÄ±sal formata dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r
if len(categorical_features) > 0:
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True) # drop_first=True ile dummy deÄŸiÅŸken tuzaÄŸÄ±nÄ± Ã¶nleriz
    X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
    print("\nKategorik sÃ¼tunlara One-Hot Encoding uygulandÄ±.")

# EÄŸitim ve test veri setlerinin sÃ¼tunlarÄ±nÄ±n tutarlÄ± olduÄŸundan emin olalÄ±m
# get_dummies farklÄ± sÃ¼tunlar oluÅŸturabilir eÄŸer train ve test setlerinde farklÄ± kategoriler varsa
# veya birinde olup diÄŸerinde olmayan kategoriler varsa.
# Bu durumda, sÃ¼tunlarÄ± eÅŸleÅŸtirmek en iyi uygulamadÄ±r.
common_cols = list(set(X.columns) & set(X_test.columns))
X = X[common_cols]
X_test = X_test[common_cols]

# EÄŸer One-Hot Encoding sonucunda bazÄ± sÃ¼tunlar sadece train'de veya sadece test'te oluÅŸursa,
# eÅŸleÅŸtirme sÄ±rasÄ±nda eksik kalanlara 0 deÄŸeri atayabiliriz.
missing_cols_test = set(X.columns) - set(X_test.columns)
for c in missing_cols_test:
    X_test[c] = 0

missing_cols_train = set(X_test.columns) - set(X.columns)
for c in missing_cols_train:
    X[c] = 0

# SÃ¼tun sÄ±ralamasÄ±nÄ± eÅŸleÅŸtirelim
X_test = X_test[X.columns]

print(f"\nÃ–n iÅŸleme sonrasÄ± eÄŸitim verisi boyutu: {X.shape}")
print(f"Ã–n iÅŸleme sonrasÄ± test verisi boyutu: {X_test.shape}")


print("Veri Ã¶n iÅŸleme tamamlandÄ±.")

# --- 4. Model EÄŸitimi ---
print("\nLightGBM modeli eÄŸitiliyor...")

# LGBMRegressor modelini tanÄ±mlayalÄ±m
# Hiperparametreler baÅŸlangÄ±Ã§ seviyesi iÃ§in varsayÄ±lan veya yaygÄ±n kullanÄ±lan deÄŸerler olabilir.
# Daha iyi sonuÃ§lar iÃ§in hiperparametre ayarlamasÄ± (tuning) yapÄ±lmasÄ± gerekir.
lgbm = lgb.LGBMRegressor(random_state=42) # random_state sonuÃ§larÄ±n tekrarlanabilir olmasÄ±nÄ± saÄŸlar

# Modeli log(1+Calories) Ã¼zerine eÄŸitiyoruz
lgbm.fit(X, y)

print("Model eÄŸitimi tamamlandÄ±.")

# --- 5. Tahmin Yapma ve GÃ¶nderme DosyasÄ± OluÅŸturma ---
print("\nTest veri seti Ã¼zerinde tahminler yapÄ±lÄ±yor...")

# EÄŸitilmiÅŸ model ile test veri seti Ã¼zerinde tahmin yapÄ±yoruz
predictions_log = lgbm.predict(X_test)

# Tahminlerimiz log(1+Calories) formatÄ±nda olduÄŸu iÃ§in, orijinal Kalori deÄŸerine geri dÃ¶nÃ¼ÅŸtÃ¼rÃ¼yoruz
# np.expm1(x) = exp(x) - 1 dÃ¶nÃ¼ÅŸÃ¼mÃ¼nÃ¼ kullanÄ±yoruz.
predictions = np.expm1(predictions_log)

# Kalori deÄŸerleri negatif olamaz, bu yÃ¼zden tahminleri 0'Ä±n altÄ±na dÃ¼ÅŸmemesi iÃ§in sÄ±nÄ±rlandÄ±ralÄ±m
predictions[predictions < 0] = 0

print("Tahminler tamamlandÄ±.")

# GÃ¶nderme dosyasÄ±nÄ± oluÅŸturalÄ±m
# sample_submission.csv formatÄ±nda olmalÄ±: id,Calories
submission_df = pd.DataFrame({'id': test_ids, 'Calories': predictions})

# GÃ¶nderme dosyasÄ±nÄ± CSV formatÄ±nda kaydedelim
submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' dosyasÄ± baÅŸarÄ±yla oluÅŸturuldu.")
print("Ä°lk 5 tahmin:")
print(submission_df.head())

print("\nNot: Bu bir baÅŸlangÄ±Ã§ not defteridir. Daha iyi sonuÃ§lar iÃ§in ÅŸunlarÄ± deneyebilirsiniz:")
print("- Daha detaylÄ± KeÅŸifsel Veri Analizi (EDA)")
print("- Yeni Ã¶zellik mÃ¼hendisliÄŸi (feature engineering)")
print("- FarklÄ± modeller (XGBoost, CatBoost, Ridge vb.)")
print("- Model hiperparametrelerinin ayarlanmasÄ± (Hyperparameter Tuning)")
print("- Ã‡apraz DoÄŸrulama (Cross-Validation)")
print("- Orijinal veri setini kullanma (yarÄ±ÅŸma aÃ§Ä±klamasÄ±na gÃ¶re)")

