# --- 2. Gerekli Kütüphanelerin Yüklenmesi ---

# Bu hücrede projemiz için gerekli olan tüm Python kütüphanelerini yüklüyoruz.

# Optuna, hiperparametre optimizasyonu için kullanacağımız güçlü bir kütüphane.
# '-q' parametresi, kurulum sırasında daha az çıktı gösterilmesini sağlar.
!pip install -q optuna

# Sayısal hesaplamalar ve dizi işlemleri için temel kütüphane.
import numpy as np
# Verileri tablolar (DataFrame) halinde okumak, işlemek ve analiz etmek için kullanılır.
import pandas as pd
# Veri görselleştirme, yani grafikler çizmek için kullanılır.
import matplotlib.pyplot as plt
# Daha estetik ve bilgilendirici grafikler çizmek için kullanılan bir başka görselleştirme kütüphanesi.
import seaborn as sns

# Makine öğrenmesi modelleri ve yardımcı araçlar için Scikit-learn kütüphanesi.
# Veriyi eğitim ve test setlerine ayırmak için kullanılır.
from sklearn.model_selection import train_test_split, KFold
# Hata metriklerini hesaplamak için kullanılır (bizim için RMSE).
from sklearn.metrics import mean_squared_error
# Yeni özellikler oluşturmak için kullanılır.
from sklearn.preprocessing import PolynomialFeatures

# LightGBM, hızlı ve yüksek performanslı bir Gradient Boosting modelidir.
import lightgbm as lgb
# Optuna, en iyi model hiperparametrelerini otomatik olarak bulmamıza yardımcı olur.
import optuna

# Gereksiz uyarı mesajlarını gizlemek için kullanılır.
import warnings
warnings.filterwarnings('ignore')

# Her şeyin başarıyla yüklendiğini teyit eden bir mesaj.
print("Kütüphaneler başarıyla yüklendi.")


# --- Veri Setlerini Yükleme ve Birleştirme ---

# Yarışma için sağlanan sentetik eğitim ve test verilerini yüklüyoruz.
# try-except bloğu, kodun hem yerel makinede hem de Kaggle ortamında çalışmasını sağlar.
try:
    # Eğitim verisini pandas DataFrame olarak oku.
    train_df = pd.read_csv("train.csv")
    # Test verisini pandas DataFrame olarak oku.
    test_df = pd.read_csv("test.csv")
    # Örnek gönderim dosyasını yükle.
    sample_submission = pd.read_csv("sample_submission.csv")
    
    # Orijinal veri setini yükle.
    original_train_df = pd.read_csv("Train.csv")
except FileNotFoundError:
    # Kaggle ortamı için dosya yolları.
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
    
    # Orijinal veri setinin Kaggle yolları.
    original_train_df = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")

# Orijinal ve sentetik eğitim verilerini birleştirelim.
# Bu, modelin daha fazla veri görmesini ve daha iyi genelleme yapmasını sağlar.
# Sütun isimleri farklı olabilir, bu yüzden onları standart hale getirelim.
# Orijinal veri setindeki sütun adlarını sentetik veri setindekiyle aynı yapıyoruz.
original_train_df.rename(columns={'ID': 'id', 'BeatsPerMinute': 'BeatsPerMinute'}, inplace=True)
# İki DataFrame'i birleştiriyoruz. ignore_index=True, yeni bir sıralı indeks oluşturur.
combined_train_df = pd.concat([train_df, original_train_df], ignore_index=True)

# --- Veriye İlk Bakış ---

# Birleştirilmiş eğitim verisinin boyutlarını yazdır (satır, sütun sayısı).
print("Birleştirilmiş Eğitim Verisi Boyutu:", combined_train_df.shape)
# Test verisinin boyutlarını yazdır.
print("Test Verisi Boyutu:", test_df.shape)

# Verinin ilk 5 satırını göstererek genel yapısını inceleyelim.
print("\nBirleştirilmiş Eğitim Verisinin İlk 5 Satırı:")
display(combined_train_df.head())

# Veri setindeki sütunların tiplerini ve boş değer olup olmadığını kontrol edelim.
print("\nEğitim Verisi Bilgileri:")
combined_train_df.info()

# --- Hedef Değişken (BeatsPerMinute) Analizi ---

# Hedef değişkenimizin dağılımını görselleştirelim.
plt.figure(figsize=(12, 6)) # Grafiğin boyutunu ayarla.
# Seaborn kütüphanesi ile bir histogram çiziyoruz. kde=True, yoğunluk eğrisini de ekler.
sns.histplot(combined_train_df['BeatsPerMinute'], bins=50, kde=True, color='skyblue')
# Grafiğe bir başlık ekle.
plt.title('BPM (Dakikadaki Vuruş Sayısı) Dağılımı', fontsize=16)
# X eksenini etiketle.
plt.xlabel('BeatsPerMinute')
# Y eksenini etiketle.
plt.ylabel('Frekans (Sayı)')
# Grafiği göster.
plt.show()

print("EDA Notu: BPM dağılımı kabaca normal bir dağılıma benziyor, ancak bazı tepe noktaları var. Bu, belirli tempo aralıklarının müzikte daha yaygın olduğunu gösterebilir.")

# --- Özellikler Arasındaki Korelasyon Analizi ---

# Sayısal özellikler arasındaki ilişkiyi görmek için bir korelasyon matrisi oluşturalım.
# 'id' sütunu bir tanımlayıcı olduğu için korelasyondan çıkarıyoruz.
correlation_matrix = combined_train_df.drop('id', axis=1).corr()

# Korelasyon matrisini bir ısı haritası (heatmap) ile görselleştirelim.
plt.figure(figsize=(14, 10)) # Grafiğin boyutunu ayarla.
# annot=True, değerleri hücrelerin içine yazar. cmap, renk paletini belirler.
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
# Grafiğe başlık ekle.
plt.title('Özellikler Arasındaki Korelasyon Matrisi', fontsize=16)
# Grafiği göster.
plt.show()


# --- 4. Özellik Mühendisliği (Feature Engineering) ---

# Orijinal verileri bozmamak için kopyalarını oluşturalım.
train_fe = combined_train_df.copy()
test_fe = test_df.copy()

print("Mevcut Sütun Adları (Başlangıçta):")
print(train_fe.columns.tolist())
print("-" * 30)

# Tahmin için kullanılmayacak sütunları ayıralım.
features = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]

# --- ÖZELLİK EŞLEŞTİRME ---
energy_col = 'Energy'
danceability_col = 'RhythmScore'
loudness_col = 'AudioLoudness'
acousticness_col = 'AcousticQuality'
valence_col = 'MoodScore'

# 1. Etkileşim Özellikleri
train_fe['energy_danceability_interaction'] = train_fe[energy_col] * train_fe[danceability_col]
test_fe['energy_danceability_interaction'] = test_fe[energy_col] * test_fe[danceability_col]

train_fe['loudness_acousticness_interaction'] = train_fe[loudness_col] * train_fe[acousticness_col]
test_fe['loudness_acousticness_interaction'] = test_fe[loudness_col] * test_fe[acousticness_col]

# 2. Polinomsal Özellikler
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False) # interaction_only=False to get squared terms too

poly_cols = [energy_col, danceability_col, loudness_col, valence_col]
poly_features_train = poly.fit_transform(train_fe[poly_cols])
poly_features_test = poly.transform(test_fe[poly_cols])

poly_feature_names = poly.get_feature_names_out(poly_cols)
poly_train_df = pd.DataFrame(poly_features_train, columns=poly_feature_names)
poly_test_df = pd.DataFrame(poly_features_test, columns=poly_feature_names)

# --- HATA DÜZELTMESİ: Tekrarlanan sütunları birleştirmeden önce kaldıralım ---
# PolynomialFeatures, hem orijinal sütunları (örn: 'RhythmScore') hem de yeni etkileşim
# terimlerini (örn: 'RhythmScore^2', 'RhythmScore AudioLoudness') oluşturur.
# Orijinal DataFrame ile birleştirdiğimizde, orijinal sütunlar tekrarlanmış olur.
# Bu yüzden, yeni oluşturduğumuz DataFrame'den orijinal sütunları çıkarıyoruz.
poly_train_df = poly_train_df.drop(columns=poly_cols)
poly_test_df = poly_test_df.drop(columns=poly_cols)

print(f"\nPolinomsal özellikler oluşturuldu. {len(poly_train_df.columns)} yeni etkileşim/kare terimi eklenecek.")

# Orijinal DataFrame'lerle yeni ve benzersiz özellikleri birleştiriyoruz.
train_fe = pd.concat([train_fe.reset_index(drop=True), poly_train_df], axis=1)
test_fe = pd.concat([test_fe.reset_index(drop=True), poly_test_df], axis=1)

# Yeni oluşturulan özelliklerle birlikte tüm özelliklerin listesini güncelliyoruz.
features = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]

# --- KONTROL ADIMI: Tekrarlanan sütun olup olmadığını kontrol edelim ---
duplicate_columns = train_fe[features].columns[train_fe[features].columns.duplicated()]
if len(duplicate_columns) > 0:
    print(f"\nUYARI: Tekrarlanan sütunlar bulundu: {duplicate_columns.tolist()}")
else:
    print("\nKontrol başarılı: Tekrarlanan sütun bulunamadı.")
    
print("\nÖzellik mühendisliği tamamlandı.")
print(f"Eğitim setindeki yeni özellik sayısı: {len(features)}")


# --- Optuna ile Hiperparametre Optimizasyonu ---

# Optuna'nın optimize edeceği bir "amaç fonksiyonu" tanımlıyoruz.
def objective(trial):
    # Bu fonksiyon, Optuna'nın her denemesinde farklı hiperparametrelerle bir model eğitir
    # ve modelin hata skorunu (RMSE) döndürür. Optuna, bu skoru minimize etmeye çalışır.

    # Hiperparametreler için arama uzayını tanımlıyoruz.
    # Optuna, her denemede bu aralıklardan değerler seçecektir.
    params = {
        'objective': 'regression_l1',  # L1 kaybı, aykırı değerlere karşı daha dayanıklıdır.
        'metric': 'rmse',              # Değerlendirme metriğimiz.
        'n_estimators': 2000,          # Ağaç sayısı, erken durdurma ile en iyisi bulunacak.
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # L1 düzenlileştirme
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2 düzenlileştirme
        'random_state': 42,
        'n_jobs': -1
    }

    # Hızlı bir optimizasyon için verinin küçük bir kısmını ayırıyoruz.
    X_train, X_val, y_train, y_val = train_test_split(
        train_fe[features], train_fe['BeatsPerMinute'], test_size=0.2, random_state=42
    )

    # LightGBM modelini önerilen parametrelerle oluşturuyoruz.
    model = lgb.LGBMRegressor(**params)
    # Modeli eğitiyoruz. early_stopping_rounds, performans artmazsa eğitimi durdurur.
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # Doğrulama seti üzerinde tahminler yapıyoruz.
    preds = model.predict(X_val)
    # RMSE skorunu hesaplıyoruz.
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    # Skoru Optuna'ya döndürüyoruz.
    return rmse

# Bir 'study' nesnesi oluşturuyoruz. 'direction='minimize'' Optuna'ya skoru küçültmesini söyler.
study = optuna.create_study(direction='minimize', study_name='LGBM Optimization')
# Optimizasyon sürecini başlatıyoruz. n_trials, kaç farklı kombinasyon deneneceğini belirtir.
# Daha fazla deneme daha iyi sonuç verebilir ama daha uzun sürer.
study.optimize(objective, n_trials=30)

# Optimizasyon bittiğinde en iyi parametreleri alıyoruz.
best_params = study.best_params
# Bazı sabit parametreleri ekliyoruz.
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['boosting_type'] = 'gbdt'

print("\nOptuna tarafından bulunan en iyi hiperparametreler:")
print(best_params)


# --- K-Fold Çapraz Doğrulama ile Eğitim ve Tahmin ---

# K-Fold için kat (bölüm) sayısını belirliyoruz.
NFOLDS = 5
# KFold nesnesini oluşturuyoruz. shuffle=True, veriyi karıştırır.
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# OOF (Out-of-Fold) tahminlerini saklamak için boş bir dizi oluşturuyoruz.
# Bu, modelin eğitim verisi üzerindeki genelleştirme performansını ölçmemizi sağlar.
oof_preds = np.zeros(train_fe.shape[0])
# Test verisi tahminlerini saklamak için boş bir dizi oluşturuyoruz.
# Her kat için yapılan tahminleri toplayıp sonra ortalamasını alacağız.
test_preds = np.zeros(test_fe.shape[0])

# Tüm özelliklerin listesini yeniden oluşturalım.
features = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]
X = train_fe[features]
y = train_fe['BeatsPerMinute']
X_test = test_fe[features]

# K-Fold döngüsünü başlatıyoruz.
# 'enumerate' hem kat numarasını hem de indeksleri verir.
for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    # Her katın başında hangi katta olduğumuzu yazdırıyoruz.
    print(f"========== KAT {fold + 1}/{NFOLDS} BAŞLADI ==========")
    
    # Veriyi bu kat için eğitim ve doğrulama setlerine ayırıyoruz.
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Modeli, Optuna'dan aldığımız en iyi parametrelerle başlatıyoruz.
    model = lgb.LGBMRegressor(**best_params, n_estimators=10000) # Erken durdurma için yüksek n_estimators.
    
    # Modeli eğitiyoruz.
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(200, verbose=False)]) # Performans 200 tur artmazsa dur.
    
    # Doğrulama seti üzerinde tahminler yapıp OOF dizisine kaydediyoruz.
    oof_preds[val_idx] = model.predict(X_val)
    # Test seti üzerinde tahminler yapıp toplam tahmin dizisine ekliyoruz.
    test_preds += model.predict(X_test) / NFOLDS

# --- Eğitim Sonu Değerlendirme ---

# Tüm OOF tahminlerini kullanarak genel RMSE skorunu hesaplıyoruz.
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nGenel Out-of-Fold (OOF) RMSE Skoru: {oof_rmse:.5f}")


# --- 7. Sonuçların Gönderilmesi ve Ek Analizler ---

# 7.1. Gönderim Dosyasının Oluşturulması
# Son olarak, K-Fold sürecinde elde ettiğimiz ortalama test tahminlerini kullanarak
# yarışmanın istediği formatta bir submission.csv dosyası oluşturuyoruz.

# --- HATA DÜZELTMESİ: Sütun adını yarışmanın istediği gibi 'ID' yapalım ---
# Yarışma, gönderim dosyasında 'ID' (büyük harf) ve 'BeatsPerMinute' sütunları bekliyor.
# Bizim test_df DataFrame'imizde bu sütun 'id' (küçük harf) olarak adlandırılmış olabilir.
# Bu yüzden, yeni DataFrame'i oluştururken doğru adlandırmayı kullanmalıyız.

# Gönderim için bir DataFrame oluşturuyoruz.
# Sütun adı olarak 'ID' (büyük harf) kullanıyoruz ve değerleri test_df['id'] (küçük harf) sütunundan alıyoruz.
submission = pd.DataFrame({
    'ID': test_df['id'],  # Çıktı sütununun adı 'ID' olmalı
    'BeatsPerMinute': test_preds
})

# DataFrame'i CSV dosyası olarak kaydediyoruz. index=False, satır numaralarının yazılmasını engeller.
submission.to_csv('submission.csv', index=False)

print("\nGönderim dosyası 'submission.csv' başarıyla oluşturuldu!")
# Dosyanın ilk birkaç satırını göstererek formatı ve sütun adlarını kontrol ediyoruz.
# Çıktıda 'ID' sütununun büyük harfle yazıldığını teyit edin.
display(submission.head())

# --- Tahminlerin Dağılımını Görselleştirme ---

# Test seti için yaptığımız tahminlerin dağılımını çizdirelim.
# Bu, eğitim verisinin dağılımıyla benzer olup olmadığını kontrol etmek için iyi bir yöntemdir.
plt.figure(figsize=(12, 6))
sns.histplot(submission['BeatsPerMinute'], bins=50, kde=True, color='purple')
plt.title('Test Seti Tahminlerinin Dağılımı', fontsize=16)
plt.xlabel('Tahmin Edilen BeatsPerMinute')
plt.ylabel('Frekans')
plt.show()

