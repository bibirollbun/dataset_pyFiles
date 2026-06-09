# Veri işleme ve analiz için temel kütüphaneler
import numpy as np
import pandas as pd
import gc

# Veri görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# Makine öğrenmesi modeli
import lightgbm as lgb

# Yardımcı araçlar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Uyarıları bastırmak için
import warnings
warnings.filterwarnings('ignore')

# Grafiklerin stilini belirleme
plt.style.use('ggplot')
sns.set_style('whitegrid')

print("Gerekli kütüphaneler başarıyla yüklendi.")


# Veri setlerinin yollarını belirtme
TRAIN_PATH = '/kaggle/input/playground-series-s5e11/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e11/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/playground-series-s5e11/sample_submission.csv'

# Veri setlerini yükleme
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission_df = pd.read_csv(SAMPLE_SUB_PATH)

print(f"Eğitim verisi boyutu: {train_df.shape}")
print(f"Test verisi boyutu: {test_df.shape}")


# Eğitim ve test verilerini birleştirerek tüm işlemleri tek seferde yapıyoruz
combined_df = pd.concat([train_df.drop('loan_paid_back', axis=1), test_df], ignore_index=True)
print("Ekstrem özellik mühendisliği başlıyor...")

# 1. Temel Oranlar ve Etkileşimler
combined_df['income_to_loan_ratio'] = combined_df['annual_income'] / combined_df['loan_amount']
combined_df['credit_to_income_ratio'] = combined_df['credit_score'] / combined_df['annual_income']
combined_df['loan_to_credit_ratio'] = combined_df['loan_amount'] / combined_df['credit_score']

# 2. YENİ (EKSTREM): Çok Seviyeli Gruplama ve Agregasyon
# İki farklı kategorik özelliği birleştirerek çok daha granüler ve güçlü gruplar oluşturuyoruz.
multi_level_groups = [
    ('grade_subgrade', 'employment_status'),
    ('grade_subgrade', 'education_level'),
    ('employment_status', 'education_level')
]

agg_features_ext = {
    'annual_income': ['mean', 'max', 'min', 'std', 'nunique'], # nunique eklendi
    'credit_score': ['mean', 'max', 'min', 'std', 'nunique'],
    'loan_amount': ['mean', 'max', 'min', 'std'],
    'interest_rate': ['mean', 'std']
}

print("Çok seviyeli gruplama özellikleri oluşturuluyor...")
for g1, g2 in multi_level_groups:
    grouped_stats = combined_df.groupby([g1, g2]).agg(agg_features_ext)
    grouped_stats.columns = [f'{g1}_{g2}_{feature}_{stat}' for feature, stat in grouped_stats.columns.to_flat_index()]
    combined_df = combined_df.merge(grouped_stats, on=[g1, g2], how='left')

# 3. Grup Ortalamasından Fark Özellikleri
print("Grup farkı özellikleri oluşturuluyor...")
# Önce tek seviyeli gruplamaları hesapla
single_group_cols = ['education_level', 'employment_status', 'grade_subgrade']
for col in single_group_cols:
    for feature in ['annual_income', 'credit_score']:
        mean_val = combined_df.groupby(col)[feature].transform('mean')
        combined_df[f'{col}_{feature}_diff_from_mean'] = combined_df[feature] - mean_val

# 4. Veri Temizliği ve Hazırlığı
categorical_features = combined_df.select_dtypes(include=['object']).columns
combined_df = pd.get_dummies(combined_df, columns=categorical_features, drop_first=True)

# 5. YENİ (EKSTREM): GPU için Veri Tipi Optimizasyonu
# Bellek kullanımını yarıya indirir ve GPU işlemlerini hızlandırır.
print("Veri tipleri GPU için optimize ediliyor (float32)...")
for col in combined_df.select_dtypes(include=['float64']).columns:
    combined_df[col] = combined_df[col].astype(np.float32)

# Boş değerleri doldurma
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df.fillna(0, inplace=True) # Median yerine 0 ile doldurmak bazen daha robust olabilir

# 6. Veriyi Ayırma ve Son Hazırlıklar
train_processed = combined_df.iloc[:len(train_df)]
test_processed = combined_df.iloc[len(train_df):]
train_processed['loan_paid_back'] = train_df['loan_paid_back']

X = train_processed.drop(['id', 'loan_paid_back'], axis=1)
y = train_processed['loan_paid_back']
X_test = test_processed.drop(['id'], axis=1)

del combined_df, train_processed, test_processed
gc.collect()

print(f"\nÖzellik mühendisliği tamamlandı. Yeni eğitim verisi boyutu: {X.shape}")


# Gerekli kütüphaneyi ekliyoruz
from joblib import Parallel, delayed

# Ortak Ayarlar
NFOLDS = 10
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
SEED = 42

# --- EKSTREM HIZ ve SKOR için Optimize Edilmiş LightGBM Parametreleri ---
# Not: gpu_device_id parametresini buradan kaldırıyoruz, çünkü her görevde dinamik olarak atanacak.
lgb_params_extreme_speed = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 20000,
    'learning_rate': 0.005,
    'num_leaves': 90,
    'max_depth': 10,
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.65,
    'subsample': 0.65,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'device': 'gpu',
    'force_col_wise': True,
    'gpu_use_dp': False
}

# Tek bir katmanı eğitecek olan fonksiyonu tanımlıyoruz
def train_fold(n_fold, train_idx, valid_idx, X, y, X_test, params):
    """
    Belirtilen katman verilerini alır, GPU ataması yapar, modeli eğitir
    ve tahminleri geri döndürür.
    """
    print(f"Fold {n_fold + 1} eğitimi başlıyor...")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    # Her görev için parametreleri kopyalayıp GPU ID'sini dinamik olarak atıyoruz
    # n_fold % 2 işlemi, katmanları 0 ve 1 numaralı GPU'lar arasında paylaştırır.
    current_params = params.copy()
    current_params['gpu_device_id'] = n_fold % 2
    
    model = lgb.LGBMClassifier(**current_params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
              callbacks=[lgb.early_stopping(500, verbose=False)])
    
    oof_pred = model.predict_proba(X_valid)[:, 1]
    sub_pred = model.predict_proba(X_test)[:, 1]
    
    fold_auc = roc_auc_score(y_valid, oof_pred)
    print(f"Fold {n_fold + 1} AUC: {fold_auc:.6f}")
    
    # Belleği temizle
    del model, X_train, y_train, X_valid, y_valid
    gc.collect()
    
    return valid_idx, oof_pred, sub_pred

# --- Paralel Eğitim Başlangıcı ---
print("\n--- ÇİFT GPU ile Paralel Eğitim Başlıyor ---")
print("Aynı anda 2 katman (fold) eğitilecek.")

# n_jobs=2, aynı anda 2 görevin (her biri bir GPU'da) çalışacağını belirtir.
parallel = Parallel(n_jobs=2, backend='threading', verbose=0)

# Döngüyü paralel hale getiriyoruz
results = parallel(
    delayed(train_fold)(
        n_fold,
        train_idx,
        valid_idx,
        X, y, X_test,
        lgb_params_extreme_speed
    )
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y))
)

# Sonuçları birleştirme
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])

for valid_idx, oof_pred, sub_pred in results:
    oof_preds[valid_idx] = oof_pred
    sub_preds += sub_pred / NFOLDS

# Modelin genel performansını değerlendirme
mean_oof_auc = roc_auc_score(y, oof_preds)
print(f"\nGenel Ortalama OOF AUC Skoru: {mean_oof_auc:.6f}")


submission_df = pd.DataFrame({'id': test_df['id'], 'loan_paid_back': sub_preds})
submission_df.to_csv('submission_lgbm_extreme.csv', index=False)

print("\nEkstrem optimize edilmiş LightGBM gönderim dosyası 'submission_lgbm_extreme.csv' başarıyla oluşturuldu.")
display(submission_df.head())

