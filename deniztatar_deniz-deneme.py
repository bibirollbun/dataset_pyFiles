# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Input data files are available in the read-only "../input/" directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Verileri okuma
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# İlk bakış
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())


# Eksik değer kontrolü
print("Train eksik değerler:")
print(train.isna().sum())
print("\nTest eksik değerler:")
print(test.isna().sum())

# Veri tiplerini kontrol etme
print("\nTrain veri tipleri:")
print(train.dtypes)


# İstatistiksel özet
print("Train istatistikleri:")
print(train.describe())

print("\nTest istatistikleri:")
print(test.describe())


# Hedef değişken dağılımını inceleme
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(train['BeatsPerMinute'], kde=True)
plt.title('Hedef Değişken Dağılımı (BeatsPerMinute)')
plt.show()


# Feature Engineering fonksiyonu
def feature_engineering(df):
    df = df.copy()
    
    # Mevcut feature'ların etkileşimleri
    df['Rhythm_Energy'] = df['RhythmScore'] * df['Energy']
    df['Rhythm_Loudness'] = df['RhythmScore'] * df['AudioLoudness']
    df['Duration_Minutes'] = df['TrackDurationMs'] / 60000  
    df['Duration_Energy_Ratio'] = df['TrackDurationMs'] / (df['Energy'] * 10000 + 1)  
    df['RhythmScore_Squared'] = df['RhythmScore'] ** 2
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Log_Duration'] = np.log1p(df['TrackDurationMs']) 
    df['Acoustic_Instrumental_Ratio'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 0.01) 
    df['Vocal_Energy'] = df['VocalContent'] * df['Energy']
    df['Live_Energy'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['Mood_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Audio_Intensity'] = (df['Energy'] * np.abs(df['AudioLoudness'])) / 10  
    df['Performance_Character'] = (df['LivePerformanceLikelihood'] + df['MoodScore']) / 2
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 0.01)
    df['Rhythm_Duration_Density'] = df['RhythmScore'] / (df['Duration_Minutes'] + 0.01)
    
    return df

# Feature engineering uygula
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

print("Yeni train shape:", train_fe.shape)
print("Yeni test shape:", test_fe.shape)


# Yeni oluşturulan feature'ları kontrol et
original_cols = set(train.columns)
new_cols = [col for col in train_fe.columns if col not in original_cols]
print("Yeni feature'lar:", new_cols)
print("Toplam yeni feature sayısı:", len(new_cols))


# Gerekli kütüphaneleri import et
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# Feature matrix X ve target vector y'yi hazırla
# ID sütununu çıkar ve hedef değişkeni ayır
X = train_fe.drop(['id', 'BeatsPerMinute'], axis=1)
y = train_fe['BeatsPerMinute']
test_ids = test_fe['id']
test_fe = test_fe.drop(['id'], axis=1)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Test features shape:", test_fe.shape)


# Preprocessor - sayısal değişkenleri standardize et
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
    ]
)


# LightGBM modelini tanımla - optimize edilmiş hiperparametreler
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,  # Daha fazla estimator
    learning_rate=0.01,
    max_depth=12,  # Daha derin ağaçlar
    num_leaves=63,  # Daha fazla yaprak
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    importance_type='gain'
)


# Pipeline oluştur
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lgb_model)
])


# Cross-validation ile model performansını değerlendirme
# GroupKFold yerine standart KFold kullanıyoruz çünkü datasetimizde grup yapısı yok
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(pipeline, X, y, 
                          cv=kf, 
                          scoring='neg_mean_squared_error',
                          n_jobs=-1)

print("CV MSE Scores:", -cv_scores)
print("CV MSE Ortalama:", -cv_scores.mean())
print("CV MSE Std:", cv_scores.std())


# Modeli tüm train verisi ile eğit
print("Model eğitiliyor...")
pipeline.fit(X, y)
print("Model eğitimi tamamlandı!")


# Feature importance'yi görselleştir
feature_names = X.columns
importances = pipeline.named_steps['model'].feature_importances_

# Önem sırasına göre sırala
indices = np.argsort(importances)[::-1]

# En önemli 15 feature'ı göster
plt.figure(figsize=(12, 8))
plt.title("Feature Importance (Top 15)")
plt.bar(range(15), importances[indices[:15]])
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=45)
plt.tight_layout()
plt.show()


# Test tahminlerini yap
print("Test tahminleri yapılıyor...")
test_predictions = pipeline.predict(test_fe)

# Tahminleri kontrol et
print("Tahmin istatistikleri:")
print(f"Min: {test_predictions.min():.2f}")
print(f"Max: {test_predictions.max():.2f}")
print(f"Ortalama: {test_predictions.mean():.2f}")
print(f"Std: {test_predictions.std():.2f}")


# Submission dosyasını oluştur
submission = pd.DataFrame({
    'id': test_ids,
    'BeatsPerMinute': test_predictions
})

# Tahminleri kontrol et
print("Submission önizleme:")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")

# Dosyayı kaydet
submission.to_csv('submission.csv', index=False)
print("Submission dosyası kaydedildi!")

