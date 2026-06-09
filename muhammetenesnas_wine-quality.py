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


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")



# Veri setlerini yükle
train_df = pd.read_csv('/kaggle/input/chydv-hackathon-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/chydv-hackathon-2025/test.csv')
sub = pd.read_csv('/kaggle/input/chydv-hackathon-2025/sample_submission.csv')

# Özellik ve hedef değişkenleri ayır
X = train_df.drop(columns=['quality', 'id'])
y = train_df['quality'] - 3  # Başlangıçta 3 çıkarıyoruz
y = y.values.ravel()

# Test verisi için özellikleri ayarla
X_test = test_df.drop(columns=['id'])

############################################
# SMOTE ile Oversampling ve Değerlendirme
############################################
def evaluate_smote(X, y, test_data):
    print("\n--- SMOTE Değerlendirmesi ---")
    print("Orijinal sınıf dağılımı:")
    print(pd.Series(y).value_counts().sort_index())
    
    # Eğitim verisi için ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Test verisini, eğitim verisiyle öğrenilen scaler ile dönüştür
    test_scaled = scaler.transform(test_data)
    
    # SMOTE uygulama
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
    
    print("\nSMOTE sonrası sınıf dağılımı:")
    print(pd.Series(y_smote).value_counts().sort_index())
    
    # Cross-validation ile değerlendirme
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X_smote, y_smote, cv=5, scoring='accuracy')
    print(f"\nRandom Forest CV Accuracy with SMOTE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Tam veri seti üzerinde eğitim
    rf.fit(X_smote, y_smote)
    
    # Test verisi üzerinde tahmin
    smote_predictions = rf.predict(test_scaled)
    
    return rf, cv_scores.mean(), smote_predictions

############################################
# RandomOverSampler ile Oversampling ve Değerlendirme
############################################
def evaluate_ros(X, y, test_data):
    print("\n--- RandomOverSampler Değerlendirmesi ---")
    print("Orijinal sınıf dağılımı:")
    print(pd.Series(y).value_counts().sort_index())
    
    # Eğitim verisi için ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Test verisini, eğitim verisiyle öğrenilen scaler ile dönüştür
    test_scaled = scaler.transform(test_data)
    
    # RandomOverSampler uygulama
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X_scaled, y)
    
    print("\nRandomOverSampler sonrası sınıf dağılımı:")
    print(pd.Series(y_ros).value_counts().sort_index())
    
    # Cross-validation ile değerlendirme
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X_ros, y_ros, cv=5, scoring='accuracy')
    print(f"\nRandom Forest CV Accuracy with ROS: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Tam veri seti üzerinde eğitim
    rf.fit(X_ros, y_ros)
    
    # Test verisi üzerinde tahmin
    ros_predictions = rf.predict(test_scaled)
    
    return rf, cv_scores.mean(), ros_predictions

# Oversampling tekniklerini değerlendir
rf_smote, smote_score, smote_preds = evaluate_smote(X, y, X_test)
rf_ros, ros_score, ros_preds = evaluate_ros(X, y, X_test)

# En iyi modeli ve tahminleri bul
scores = {
    "SMOTE": smote_score,
    "ROS": ros_score
}

best_method = max(scores, key=scores.get)
print(f"\n*** En iyi yöntem: {best_method} (Accuracy: {scores[best_method]:.4f}) ***")

# En iyi yöntemin tahminlerini kullan
if best_method == "SMOTE":
    best_preds = smote_preds
else:
    best_preds = ros_preds

# Tahminleri orijinal ölçeğe geri ayarla (başlangıçta 3 çıkardığımız için) ve submission oluştur
final_predictions = best_preds + 3
sub['quality'] = final_predictions
sub.to_csv(f'submission_{best_method}.csv', index=False)
print(f"\nTahminler '{best_method}_submission.csv' dosyasına kaydedildi.")

############################################
# Ensemble Model Değerlendirmesi
############################################
print("\n--- Ensemble Model Değerlendirmesi ---")
def create_ensemble_prediction(models, X_test_data):
    # Her modelden tahmin al - sadece None olmayan modelleri kullan
    valid_models = [m for m in models if m is not None]
    all_preds = np.array([model.predict(X_test_data) for model in valid_models])
    
    # Her örnek için mod (en yaygın tahmin) al
    ensemble_preds = np.zeros(X_test_data.shape[0], dtype=int)
    for i in range(X_test_data.shape[0]):
        values, counts = np.unique(all_preds[:, i], return_counts=True)
        ensemble_preds[i] = values[np.argmax(counts)]
    
    return ensemble_preds

# Ensemble için de, eğitim verisiyle fit edilmiş scaler kullanarak test verisini dönüştürelim
ensemble_scaler = StandardScaler()
ensemble_scaler.fit(X)  # Eğitim verisinden fit
test_scaled_ensemble = ensemble_scaler.transform(X_test)

# Ensemble tahmini oluştur
ensemble_models = [rf_smote, rf_ros]
ensemble_preds = create_ensemble_prediction(ensemble_models, test_scaled_ensemble)

# Ensemble tahminlerini orijinal ölçeğe geri ayarla ve submission oluştur
ensemble_final = ensemble_preds + 3
sub['quality'] = ensemble_final
sub.to_csv('submission.csv', index=False)
print("\nEnsemble tahminleri 'submission.csv' dosyasına kaydedildi.")





