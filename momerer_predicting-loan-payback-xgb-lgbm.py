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

# Veri setlerinin yollarını belirleyelim
train_path = '/kaggle/input/playground-series-s5e11/train.csv'
test_path = '/kaggle/input/playground-series-s5e11/test.csv'
submission_path = '/kaggle/input/playground-series-s5e11/sample_submission.csv'

# Veri setlerini yükleyelim
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(submission_path)
    
    print(f"Train verisi yüklendi. Shape: {train_df.shape}")
    print(f"Test verisi yüklendi. Shape: {test_df.shape}")
    print(f"Örnek submission dosyası yüklendi. Shape: {sample_submission_df.shape}")

except FileNotFoundError as e:
    print(f"HATA: Dosya bulunamadı. Lütfen yolları kontrol edin.")
    print(e)


# Veri çerçevelerinin ilk 5 satırını (head) gösterelim
print("--- Train Verisi İlk 5 Satır (head) ---")
print(train_df.head())
print("\n" + "="*50 + "\n")

print("--- Test Verisi İlk 5 Satır (head) ---")
print(test_df.head())
print("\n" + "="*50 + "\n")

# Veri çerçevelerinin yapısını (info) inceleyelim (Veri tipleri ve eksik veriler)
print("--- Train Verisi Bilgisi (info) ---")
# .info() çıktısı doğrudan konsola yazdırılır, print() içine almaya gerek yok.
train_df.info()
print("\n" + "="*50 + "\n")

print("--- Test Verisi Bilgisi (info) ---")
test_df.info()
print("\n" + "="*50 + "\n")

# Train ve Test setlerindeki sütunları karşılaştırarak hedef değişkeni bulalım
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

# Test setinde olmayan sütunları bulalım
target_col = list(train_cols - test_cols)

if len(target_col) == 1:
    target_variable_name = target_col[0]
    print(f"Tespit edilen Hedef Değişken (Target Variable): '{target_variable_name}'")
    
    # Hedef değişkenin dağılımını kontrol edelim
    print(f"\n--- Hedef Değişken Dağılımı (%) ---")
    print(train_df[target_variable_name].value_counts(normalize=True) * 100)
    
    # Dağılımı görselleştirelim
    plt.figure(figsize=(7, 5))
    sns.countplot(x=target_variable_name, data=train_df, palette='pastel')
    plt.title(f'Hedef Değişken Dağılımı ({target_variable_name})', fontsize=14)
    plt.ylabel('Sayı (Count)')
    plt.xlabel('Sınıf')
    
    # Eksen etiketlerini daha okunaklı hale getirelim (eğer gerekirse)
    plt.xticks(rotation=0) 
    
    # Grafiği kaydet
    plt.tight_layout() # Düzeni optimize et
    plt.savefig('target_distribution.png')
    
    print("\nHedef değişken dağılım grafiği 'target_distribution.png' olarak kaydedildi.")
    
elif len(target_col) == 0:
    print("HATA: Train ve Test setleri aynı sütunlara sahip. Hedef değişken bulunamadı.")
else:
    print(f"HATA: Birden fazla potansiyel hedef değişken bulundu: {target_col}")


# Hedef değişkeni ve ID sütununu ayıralım
target_col = 'loan_paid_back'
id_col = 'id'

# Sütun türlerini ayıralım (ID ve Target hariç)
features = train_df.columns.difference([target_col, id_col])

numerical_features = train_df[features].select_dtypes(include=np.number).columns.tolist()
categorical_features = train_df[features].select_dtypes(include='object').columns.tolist()

print(f"Sayısal Özellikler (Numerical): {numerical_features}")
print(f"Kategorik Özellikler (Categorical): {categorical_features}")
print("\n" + "="*50 + "\n")

# --- 1. Sayısal Özelliklerin İstatistikleri ---
print("--- Sayısal Özellikler İstatistikleri (Train) ---")
# .T (transpose) ile daha okunaklı hale getirelim
print(train_df[numerical_features].describe().T)
print("\n" + "="*50 + "\n")

# --- 2. Sayısal Özelliklerin Dağılımları (Histogramlar) ---
print("--- Sayısal Özelliklerin Dağılımları (Histogramlar) ---")
plt.figure(figsize=(18, 12))
plt.suptitle('Sayısal Özelliklerin Dağılımı (Train)', fontsize=16)
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1) # 2 satır, 3 sütunlu bir grid
    sns.histplot(train_df[col], kde=True, bins=50, color='blue', alpha=0.6, label='Train')
    sns.histplot(test_df[col], kde=True, bins=50, color='orange', alpha=0.6, label='Test')
    plt.title(f'{col} Dağılımı')
    plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ana başlık için yer ayır
plt.savefig('numerical_distributions.png')
print("Sayısal özellik dağılım grafikleri 'numerical_distributions.png' olarak kaydedildi.")
print("Grafikler, train (mavi) ve test (turuncu) setlerinin dağılımlarını karşılaştırır.")
print("\n" + "="*50 + "\n")

# --- 3. Kategorik Özelliklerin Analizi (Benzersiz Değerler) ---
print("--- Kategorik Özellikler Analizi (Benzersiz Değerler) ---")

# Benzersiz değer sayılarını bir DataFrame'de gösterelim
nunique_df = pd.DataFrame({
    'Özellik': categorical_features,
    'Benzersiz Değer Sayısı (Train)': [train_df[col].nunique() for col in categorical_features],
    'Benzersiz Değer Sayısı (Test)': [test_df[col].nunique() for col in categorical_features]
})
print(nunique_df.to_markdown(index=False)) # .to_markdown() ile daha güzel bir tablo çıktısı
print("\n" + "-"*30 + "\n")

# Benzersiz değerlerin kendilerini inceleyelim
print("--- Kategorik Özelliklerin İçeriği ---")
for col in categorical_features:
     print(f"\n--- '{col}' Benzersiz Değerleri (Train) ---")
     # Değerleri sıralayarak yazdıralım
     print(np.sort(train_df[col].unique()))
     
     # Test setinde train'de olmayan bir değer var mı kontrol edelim
     train_values = set(train_df[col].unique())
     test_values = set(test_df[col].unique())
     if not test_values.issubset(train_values):
         print(f"!!! DİKKAT: '{col}' test setinde, train'de olmayan şu değerleri içeriyor: {test_values - train_values}")


from sklearn.preprocessing import OrdinalEncoder

# Orijinal veri çerçevelerini korumak için kopyalarını oluşturalım
# Bu aynı zamanda 'SettingWithCopyWarning' uyarısını almamızı engeller
train_encoded = train_df.copy()
test_encoded = test_df.copy()

print("--- Ordinal Encoding Başlatıldı ---")

# 1. education_level için manuel haritalama (mapping)
# 'Other' kategorisini en düşük seviye olarak (0) tanımlayalım
education_map = {
    'Other': 0,
    'High School': 1,
    "Bachelor's": 2,
    "Master's": 3,
    'PhD': 4
}

train_encoded['education_level_encoded'] = train_encoded['education_level'].map(education_map)
test_encoded['education_level_encoded'] = test_encoded['education_level'].map(education_map)

print("--- 'education_level' Kodlama Sonucu (İlk 5 Satır) ---")
print(train_encoded[['education_level', 'education_level_encoded']].head())
print("\n" + "-"*30 + "\n")


# 2. grade_subgrade için OrdinalEncoder
# Kategorilerin sırasını (A1...F5) train setinden alıp,
# hem train hem de test setine uygulayacağız.
grade_categories = np.sort(train_df['grade_subgrade'].unique())

# Encoder'ı kategorileri belirterek başlatalım
ordinal_encoder = OrdinalEncoder(
    categories=[grade_categories], 
    dtype=int  # Çıktı tipi integer olsun
)

# SADECE train verisine fit et
ordinal_encoder.fit(train_encoded[['grade_subgrade']])

# Hem train hem test verisini transform et
train_encoded['grade_subgrade_encoded'] = ordinal_encoder.transform(train_encoded[['grade_subgrade']])
test_encoded['grade_subgrade_encoded'] = ordinal_encoder.transform(test_encoded[['grade_subgrade']])

print("--- 'grade_subgrade' Kodlama Sonucu (Örnek Satırlar) ---")
# Farklı 'grade'leri görebilmek için .head(15) yerine .sample(10) alalım
print(train_encoded[['grade_subgrade', 'grade_subgrade_encoded']].sample(10, random_state=42))
print("\n" + "-"*30 + "\n")

# --- Özellik listelerimizi güncelleyelim ---
# Artık sayısal olan yeni özellikleri 'numerical_features' listesine ekleyelim
# ve eskilerini 'categorical_features' listesinden çıkaralım.

# 'numerical_features' ve 'categorical_features' listelerinin 
# bir önceki hücrede tanımlandığını varsayıyoruz.
try:
    numerical_features.extend(['education_level_encoded', 'grade_subgrade_encoded'])
    categorical_features.remove('education_level')
    categorical_features.remove('grade_subgrade')
    
    print(f"Güncel Sayısal Özellikler: {numerical_features}")
    print(f"Kalan Kategorik Özellikler (OHE için): {categorical_features}")

except NameError:
    print("HATA: 'numerical_features' veya 'categorical_features' listeleri bulunamadı.")
    print("Lütfen bir önceki hücrenin (Hücre 3) tam olarak çalıştığından emin olun.")


# Bir önceki hücrede tanımlanan 'categorical_features' listesini kullanalım
# OHE yapılacak sütunlar: ['employment_status', 'gender', 'loan_purpose', 'marital_status']

print(f"One-Hot Encoding uygulanacak {len(categorical_features)} özellik: {categorical_features}")

# Orijinal 'train_encoded' ve 'test_encoded' veri çerçevelerini koruyalım
train_final = train_encoded.copy()
test_final = test_encoded.copy()

# OHE'nin hem train hem de testte tutarlı olmasını sağlamak için birleştirelim
# Hedef değişkeni geçici olarak ayır
target = train_final[target_col]
train_final = train_final.drop(columns=[target_col])

# Birleştirme öncesi setleri tanımak için bir işaretçi ekleyelim
train_final['dataset_marker'] = 'train'
test_final['dataset_marker'] = 'test'

# İki seti birleştir
combined_df = pd.concat([train_final, test_final], ignore_index=True)

print(f"Birleştirilmiş veri boyutu: {combined_df.shape}")

# One-Hot Encoding uygula
# drop_first=False kullanıyoruz, çünkü ağaç bazlı modeller (LightGBM, XGBoost)
# için bu genellikle daha iyi çalışır ve yorumlanabilirliği artırır.
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_features, drop_first=False)

print(f"OHE sonrası birleştirilmiş veri boyutu: {combined_df_encoded.shape}")

# Veri setlerini tekrar ayıralım
train_final = combined_df_encoded[combined_df_encoded['dataset_marker'] == 'train'].copy()
test_final = combined_df_encoded[combined_df_encoded['dataset_marker'] == 'test'].copy()

# İşaretçi sütununu kaldıralım
train_final = train_final.drop(columns=['dataset_marker'])
test_final = test_final.drop(columns=['dataset_marker'])

# Hedef değişkeni train setine geri ekleyelim
train_final[target_col] = target

# Orijinal (object tipi) kategorik sütunları da temizleyelim
# 'categorical_features' listesindeki orijinal sütun adları
original_cat_cols_to_drop = [col for col in categorical_features if col in train_final.columns]
if original_cat_cols_to_drop:
    train_final = train_final.drop(columns=original_cat_cols_to_drop)
    test_final = test_final.drop(columns=original_cat_cols_to_drop)

print("\n" + "="*50 + "\n")
print("--- One-Hot Encoding Tamamlandı ---")
print(f"Yeni Train Verisi Boyutu: {train_final.shape}")
print(f"Yeni Test Verisi Boyutu: {test_final.shape}")

print("\n--- Yeni Sütunlardan Bazı Örnekler (Train) ---")
# Yeni oluşturulan OHE sütunlarından bazılarını gösterelim
ohe_cols = [col for col in train_final.columns if any(cat_col in col for cat_col in categorical_features)]
print(train_final[ohe_cols].head())


# %load_ext cudf.pandas

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import lightgbm as lgb
import gc

# --- 1. Veri Yükleme ---
print("--- Veri Yükleniyor ---")
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
orig = orig.drop_duplicates()
print(f'Train Shape: {train.shape}, Test Shape: {test.shape}, Orig Shape (temizlenmiş): {orig.shape}')

TARGET = 'loan_paid_back'
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
BASE = [col for col in train.columns if col not in ['id', TARGET]]

# --- 2. Etkileşim Özellikleri (INTER) ---
print("\n--- Etkileşim Özellikleri Oluşturuluyor (INTER) ---")
INTER = []
for col1, col2 in combinations(BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test, orig]:
        if col1 in df.columns and col2 in df.columns:
            df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
print(f'{len(INTER)} INTER Features created.')

# --- 3. Harici Veri Özellikleri (ORIG) - (mean, count, std, median) ---
print("\n--- Harici Veri Özellikleri Oluşturuluyor (ORIG) ---")
ORIG = []
for col in BASE:
    if col not in orig.columns:
        print(f"Uyarı: '{col}' sütunu 'orig' veri setinde bulunamadı. Atlanıyor.")
        continue
    
    agg_funcs = ['mean', 'count', 'std', 'median']
    aggs = orig.groupby(col)[TARGET].agg(agg_funcs).reset_index()
    aggs.columns = [col] + [f'orig_{func}_{col}' for func in agg_funcs]
    aggs[f'orig_std_{col}'] = aggs[f'orig_std_{col}'].fillna(0)
    
    train = train.merge(aggs, on=col, how='left')
    test = test.merge(aggs, on=col, how='left')
    ORIG.extend([f'orig_{func}_{col}' for func in agg_funcs])

print(f'{len(ORIG)} ORIG Features created.') # 44 olmalı

# --- 4. Özellik Listesi ve Veri Hazırlığı ---
FEATURES = BASE + ORIG + INTER
FEATURES = [col for col in FEATURES if col in test.columns]
CATS = [col for col in CATS if col in FEATURES] 
print(f'\nToplam {len(FEATURES)} özellik ile eğitime başlanacak.')

X = train[FEATURES]
y = train[TARGET]

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# --- 5. Target Encoder Sınıfı (Değişiklik yok) ---
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train
            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0 and avg_variance_within > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

# --- 6. Model 1: XGBoost Eğitimi (GPU) (Çalışıyor) ---
print("\n--- MODEL 1: XGBoost EĞİTİMİ (GPU) BAŞLIYOR ---")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'enable_categorical': True,
    'tree_method': 'hist',
    'device': 'cuda',
    'predictor': 'gpu_predictor', 
}

oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(test))

# XGB için OOF skorunu önceki çalışmadan biliyoruz, 
# ama blend için tahminleri (test_preds_xgb) yeniden oluşturmamız lazım.
# Bu yüzden bu döngü tekrar çalışmalı.
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- XGB Fold {fold}/{N_SPLITS} ---')
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test[FEATURES].copy()
    
    INTER_to_encode = [col for col in INTER if col in X_train.columns]
    TE = TargetEncoder(cols_to_encode=INTER_to_encode, cv=5, smooth='auto', aggs=['mean'], drop_original=True)
    X_train = TE.fit_transform(X_train, y_train)
    X_val = TE.transform(X_val)
    X_test = TE.transform(X_test)
    
    CATS_in_train = [col for col in CATS if col in X_train.columns]
    X_train[CATS_in_train] = X_train[CATS_in_train].astype('category')
    X_val[CATS_in_train] = X_val[CATS_in_train].astype('category')
    X_test[CATS_in_train] = X_test[CATS_in_train].astype('category')

    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1000)
    val_preds = model_xgb.predict_proba(X_val)[:, 1]
    oof_preds_xgb[val_idx] = val_preds
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'XGB Fold {fold} AUC: {fold_score:.6f}')
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS
    del X_train, X_val, y_train, y_val, X_test, TE
    gc.collect()

overall_auc_xgb = roc_auc_score(y, oof_preds_xgb)
print(f'====================\nOverall XGB OOF AUC: {overall_auc_xgb:.6f}\n====================')

# --- 7. Model 2: LightGBM Eğitimi (CPU - DÜZELTİLDİ) ---
print("\n--- MODEL 2: LightGBM EĞİTİMİ (CPU) BAŞLIYOR (Yavaş olacak) ---")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'verbose': -1,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    
    # --- GPU PARAMETRELERİ (KALDIRILDI) ---
    # 'device': 'gpu',
    # 'device_type': 'opencl',
    
    # --- CPU PARAMETRESİ (EKLENDİ) ---
    'n_jobs': -1, 
    # --- BİTTİ ---
}

oof_preds_lgbm = np.zeros(len(X))
test_preds_lgbm = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- LGBM Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test[FEATURES].copy()

    INTER_to_encode = [col for col in INTER if col in X_train.columns]
    TE = TargetEncoder(cols_to_encode=INTER_to_encode, cv=5, smooth='auto', aggs=['mean'], drop_original=True)
    X_train = TE.fit_transform(X_train, y_train)
    X_val = TE.transform(X_val)
    X_test = TE.transform(X_test)
    
    CATS_in_train = [col for col in CATS if col in X_train.columns]
    X_train[CATS_in_train] = X_train[CATS_in_train].astype('category')
    X_val[CATS_in_train] = X_val[CATS_in_train].astype('category')
    X_test[CATS_in_train] = X_test[CATS_in_train].astype('category')

    model_lgbm = lgb.LGBMClassifier(**lgb_params)
    
    model_lgbm.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   eval_metric='auc',
                   callbacks=[lgb.early_stopping(100, verbose=1000)],
                   categorical_feature=CATS_in_train) 

    val_preds = model_lgbm.predict_proba(X_val)[:, 1]
    oof_preds_lgbm[val_idx] = val_preds
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'LGBM Fold {fold} AUC: {fold_score:.6f}')
    test_preds_lgbm += model_lgbm.predict_proba(X_test)[:, 1] / N_SPLITS
    del X_train, X_val, y_train, y_val, X_test, TE
    gc.collect()

overall_auc_lgbm = roc_auc_score(y, oof_preds_lgbm)
print(f'====================\nOverall LGBM OOF AUC: {overall_auc_lgbm:.6f}\n====================')

# --- 8. Blending (Karışım) ve Final Skor ---
print("\n--- MODELLER KARIŞTIRILIYOR (BLENDING) ---")

oof_blend = (oof_preds_xgb * 0.5) + (oof_preds_lgbm * 0.5)
overall_auc_blend = roc_auc_score(y, oof_blend)

print(f'====================')
print(f'XGB OOF AUC (GPU):    {overall_auc_xgb:.6f}')
print(f'LGBM OOF AUC (CPU):   {overall_auc_lgbm:.6f}')
print(f'BLEND OOF AUC:        {overall_auc_blend:.6f}') 
print(f'====================')

# --- 9. Submission Dosyası Oluşturma ---
test_blend = (test_preds_xgb * 0.5) + (test_preds_lgbm * 0.5)

sub_df = pd.DataFrame({'id': test['id'], TARGET: test_blend})
sub_filename = f'blend_xgb_lgbm_cpu_cv_{overall_auc_blend:.6f}.csv'
sub_df.to_csv(sub_filename, index=False)

print(f"Submission dosyası kaydedildi: {sub_filename}")
print(sub_df.head())

