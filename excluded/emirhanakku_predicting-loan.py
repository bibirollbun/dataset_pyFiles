import pandas as pd
import numpy as np

# Veri setini yüklüyoruz (Kaggle Notebook'ta genellikle bu yoldadır, değilse path'i güncelle)
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# İlk bakış
print("Train Seti Boyutu:", train_df.shape)
print("\n--- İlk 5 Satır ---")
display(train_df.head())

print("\n--- Sütun İsimleri ve Tipleri ---")
print(train_df.info())


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Gereksiz 'id' sütununu atalım, modelin kafasını karıştırmasın.
if 'id' in train_df.columns:
    train_df = train_df.drop('id', axis=1)

# 2. Hedef Değişken (Loan Paid Back) Dengesi
print("--- Hedef Değişken Dağılımı (%) ---")
print(train_df['loan_paid_back'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(x='loan_paid_back', data=train_df)
plt.title('Kredi Geri Ödeme Durumu (0: Ödenmedi, 1: Ödendi)')
plt.show()

# 3. Kategorik Verilerin İçine Bakalım (Özellikle Grade_Subgrade)
cat_cols = ['grade_subgrade', 'employment_status', 'loan_purpose']
for col in cat_cols:
    print(f"\n--- {col} (İlk 5 Değer) ---")
    print(train_df[col].value_counts().head(5))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# --- 1. ÖN İŞLEME (PREPROCESSING) ---

# Test setindeki ID'leri saklayalım (Submission için lazım olacak)
test_ids = test_df['id']

# Train ve Test'i ayırt etmek için geçici bir etiket ekleyip birleştirelim
# Böylece LabelEncoder ikisinde de aynı dönüşümü yapar.
train_df['is_train'] = 1
test_df['is_train'] = 0

# Test setinde 'loan_paid_back' yok, geçici olarak dolduralım (birleştirmek için)
test_df['loan_paid_back'] = np.nan

# İkisini alt alta birleştir
full_df = pd.concat([train_df, test_df], axis=0)

# Gereksiz sütunları atalım
cols_to_drop = ['id']
full_df = full_df.drop(cols_to_drop, axis=1)

# Kategorik sütunları bul ve sayıya çevir (Label Encoding)
cat_cols = full_df.select_dtypes(include=['object']).columns

print(f"Dönüştürülecek Kategorik Sütunlar: {list(cat_cols)}")

le = LabelEncoder()
for col in cat_cols:
    # String'e çevirip (hatayı önlemek için) encode ediyoruz
    full_df[col] = le.fit_transform(full_df[col].astype(str))

# Tekrar ayıralım
train_final = full_df[full_df['is_train'] == 1].drop(['is_train'], axis=1)
test_final = full_df[full_df['is_train'] == 0].drop(['is_train', 'loan_paid_back'], axis=1)

# --- 2. MODELLEME (XGBOOST) ---

X = train_final.drop('loan_paid_back', axis=1)
y = train_final['loan_paid_back']

# Validasyon seti ayıralım (%80 eğitim, %20 kontrol)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nModel eğitiliyor")

# Basit bir XGBoost kuralım
model = XGBClassifier(
    n_estimators=1000,      # Ağaç sayısı
    learning_rate=0.05,     # Öğrenme hızı
    max_depth=6,            # Ağaç derinliği
    eval_metric='auc',      # Yarışma metriği genelde AUC olur
    early_stopping_rounds=50, # İyileşme durursa eğitimi kes
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100 # Her 100 adımda bir bilgi ver
)

# --- 3. SONUÇ VE SUBMISSION ---

# Validasyon skoruna bakalım
val_preds = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f"\n Validation ROC-AUC Skoru: {auc_score:.5f}")

# Test seti için tahmin yap
test_preds = model.predict_proba(test_final)[:, 1]

# Submission dosyasını hazırla
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': test_preds
})

# Dosyayı kaydet
submission.to_csv('submission.csv', index=False)
print("\n 'submission.csv' dosyası hazır! Kaggle'a yükleyebilirsin.")


# --- 1. GELİŞMİŞ ÖN İŞLEME & FEATURE ENGINEERING ---

# Verileri tekrar tazeleyelim
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# id'leri ayıralım
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Train/Test birleştir
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['loan_paid_back'] = np.nan
full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# --- YENİ ÖZELLİKLER EKLEME (Sihir Burada) ---

# 1. Grade'i Parçalama (C3 -> Grade: C, Sub: 3)
# Grade zaten harf sırasına göre risk içerir, bunu map edelim
grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}

# İlk harfi alıp puana çeviriyoruz
full_df['grade_score'] = full_df['grade_subgrade'].apply(lambda x: grade_map.get(x[0], 0))
# İkinci karakteri (rakamı) alıp int yapıyoruz
full_df['subgrade_score'] = full_df['grade_subgrade'].apply(lambda x: int(x[1]) if len(x)>1 else 0)

# Kombine bir risk skoru (Örn: A1=1.1, C3=3.3 gibi düşünebilir model)
full_df['risk_score'] = full_df['grade_score'] * 10 + full_df['subgrade_score']

# 2. Finansal Oranlar
# Kredi Miktarı / Yıllık Gelir (Kişi bu borcun altında ezilir mi?)
# 0'a bölme hatası olmasın diye +1 ekliyoruz
full_df['loan_to_income'] = full_df['loan_amount'] / (full_df['annual_income'] + 1)

# 3. Tahmini Aylık Ödeme Yükü (Basit Faiz Formülü ile Yaklaşım)
# P * r (Kabaca yıllık faiz yükü / 12)
full_df['monthly_burden'] = (full_df['loan_amount'] * full_df['interest_rate']) / 12

# --- ENCODING (Kalanlar için) ---
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade'] # grade_subgrade'i de bırakalım, model belki oradan da bir şey yakalar
le = LabelEncoder()
for col in cat_cols:
    full_df[col] = le.fit_transform(full_df[col].astype(str))

# Tekrar Ayır
train_final = full_df[full_df['is_train'] == 1].drop(['is_train'], axis=1)
test_final = full_df[full_df['is_train'] == 0].drop(['is_train', 'loan_paid_back'], axis=1)

X = train_final.drop('loan_paid_back', axis=1)
y = train_final['loan_paid_back']

# --- 2. MODEL EĞİTİMİ (XGBOOST - Biraz daha güçlendirilmiş) ---

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Model yeni özelliklerle eğitiliyor")

model = XGBClassifier(
    n_estimators=1500,      # Biraz daha fazla ağaç
    learning_rate=0.03,     # Daha yavaş ve dikkatli öğrensin
    max_depth=8,            # Biraz daha derin ilişkiler kursun
    min_child_weight=3,     # Overfitting engellemek için
    subsample=0.8,          # Her ağaçta verinin %80'ini görsün (çeşitlilik)
    colsample_bytree=0.8,   # Sütunların %80'ini görsün
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=200
)

# --- 3. SONUÇ ---
val_preds = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f"\n YENİ Validation ROC-AUC Skoru: {auc_score:.5f}")

# Feature Importance (Hangi özellik işe yaramış bakalım)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- En Önemli 5 Özellik ---")
print(importances.head(5))

# Submission
test_preds = model.predict_proba(test_final)[:, 1]
submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': test_preds})
submission.to_csv('submission_v2.csv', index=False)
print("\✅ 'submission_v2.csv' hazır! Kaggle'a yükle.")


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- ÖNCEKİ VERİ HAZIRLIĞINI KORUYORUZ ---
# (Eğer notebook'u kapatmadıysan veriler hafızada duruyordur, 
# ama garanti olsun diye X ve y'nin hazır olduğundan emin olalım)

# XGBoost için hazırladığımız X ve y değişkenlerini kullanacağız.
# Eğer değişkenler silindiyse yukarıdaki feature engineering kodunu bir kez daha çalıştır.

print("Modeller hazırlanıyor.")

# --- 1. MODEL: LightGBM (Hızlı ve Öfkeli) ---
# LightGBM kategorik verilerle çok iyi anlaşır.

lgbm_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 10,
    'num_leaves': 64,             # Yaprak sayısı (LGBM için kritik)
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Validasyon seti ayıralım (Yine %20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n LightGBM Eğitiliyor...")
lgbm_model = lgb.LGBMClassifier(**lgbm_params)
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=200)]
)

# LGBM Skoru
lgbm_val_preds = lgbm_model.predict_proba(X_val)[:, 1]
print(f"LightGBM Validation AUC: {roc_auc_score(y_val, lgbm_val_preds):.5f}")

# --- 2. MODEL: XGBoost (Zaten bildiğimiz güç) ---
print("\n XGBoost Eğitiliyor (Tekrar)...")
xgb_model = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=200
)

# XGB Skoru
xgb_val_preds = xgb_model.predict_proba(X_val)[:, 1]
print(f"XGBoost Validation AUC: {roc_auc_score(y_val, xgb_val_preds):.5f}")

# --- 3. ENSEMBLE (BÜYÜK FİNAL) ---
# İki modelin tahminlerini birleştiriyoruz.
# Genelde en iyi modelin ağırlığı biraz daha fazla verilebilir ama şimdilik 50-50 başlayalım.

ensemble_val_preds = (lgbm_val_preds * 0.5) + (xgb_val_preds * 0.5)
ensemble_score = roc_auc_score(y_val, ensemble_val_preds)

print(f"\ENSEMBLE Validation AUC: {ensemble_score:.5f}")
print("(Tek modellerden daha yüksek olmalı!)")

# --- SUBMISSION ---
lgbm_test_preds = lgbm_model.predict_proba(test_final)[:, 1]
xgb_test_preds = xgb_model.predict_proba(test_final)[:, 1]

final_preds = (lgbm_test_preds * 0.5) + (xgb_test_preds * 0.5)

submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': final_preds})
submission.to_csv('submission_ensemble.csv', index=False)
print("\n'submission_ensemble.csv' hazır! Bu sefer rekor gelebilir.")


from catboost import CatBoostClassifier

print("CatBoost")

# --- 1. CatBoost Eğitimi ---
cat_params = {
    'iterations': 2000,
    'learning_rate': 0.03,
    'depth': 6,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 200,
    'early_stopping_rounds': 50,
    'allow_writing_files': False # Gereksiz dosya oluşturmasın
}

cat_model = CatBoostClassifier(**cat_params)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# CatBoost Skoru
cat_val_preds = cat_model.predict_proba(X_val)[:, 1]
print(f"CatBoost Validation AUC: {roc_auc_score(y_val, cat_val_preds):.5f}")

# --- 2. ÜÇLÜ ENSEMBLE (XGB + LGBM + CAT) ---

# Daha önceki modellerin tahminleri hafızada duruyor varsayıyoruz:
# xgb_val_preds, lgbm_val_preds, xgb_test_preds, lgbm_test_preds

# Validasyon Skoru (3 Modelin Ortalaması)
ensemble_val_preds_v3 = (xgb_val_preds + lgbm_val_preds + cat_val_preds) / 3
score_v3 = roc_auc_score(y_val, ensemble_val_preds_v3)

print(f"\n 3'lü ENSEMBLE Validation AUC: {score_v3:.5f}")
print(f"Eski Skorun (2'li): {ensemble_score:.5f}")

# --- 3. SUBMISSION ---
cat_test_preds = cat_model.predict_proba(test_final)[:, 1]

# Test tahminlerini birleştir
final_preds_v3 = (xgb_test_preds + lgbm_test_preds + cat_test_preds) / 3

submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': final_preds_v3})
submission.to_csv('submission_cat_ensemble.csv', index=False)
print("\n 'submission_cat_ensemble.csv' hazır")


# --- 1. VERİYİ SIFIRDAN AL VE GÜÇLENDİR ---
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['loan_paid_back'] = np.nan
full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# --- MEVCUT ÖZELLİKLER (Zaten yapmıştık) ---
grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
full_df['grade_score'] = full_df['grade_subgrade'].apply(lambda x: grade_map.get(x[0], 0))
full_df['subgrade_score'] = full_df['grade_subgrade'].apply(lambda x: int(x[1]) if len(x)>1 else 0)
full_df['risk_score'] = full_df['grade_score'] * 10 + full_df['subgrade_score']

# ---  YENİ EKLENEN KRİTİK ÖZELLİKLER (Feature V2) ---

# 1. Kişinin üzerindeki mevcut toplam borç (TL bazında)
full_df['total_debt_amount'] = full_df['annual_income'] * full_df['debt_to_income_ratio']

# 2. Harcanabilir Yıllık Gelir (Gelir - Mevcut Borç)
# (Not: Eksi çıkmaması için mutlak değer veya minimum sınır koymuyoruz, matematiksel ilişkiyi model çözer)
full_df['disposable_income'] = full_df['annual_income'] - full_df['total_debt_amount']

# 3. Kredi Miktarı / Harcanabilir Gelir (Kişi bu krediyi öderken aç kalır mı?)
# 0'a bölme hatasını önlemek için paydaya ufak bir sayı ekliyoruz
full_df['loan_burden_ratio'] = full_df['loan_amount'] / (full_df['disposable_income'] + 100)

# 4. Kredi Skoru ve Gelir İlişkisi
full_df['income_per_credit_score'] = full_df['annual_income'] / (full_df['credit_score'] + 1)

# --- ENCODING ---
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
le = LabelEncoder()
for col in cat_cols:
    full_df[col] = le.fit_transform(full_df[col].astype(str))

train_final = full_df[full_df['is_train'] == 1].drop(['is_train'], axis=1)
test_final = full_df[full_df['is_train'] == 0].drop(['is_train', 'loan_paid_back'], axis=1)

X = train_final.drop('loan_paid_back', axis=1)
y = train_final['loan_paid_back']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. MODELLERİ YENİDEN EĞİT (Hızlı Versiyon) ---
print("Modeller yeni")

# XGBoost
xgb = XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=8, eval_metric='auc', 
                    early_stopping_rounds=50, random_state=42, n_jobs=-1, verbose=0)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb.predict_proba(X_val)[:, 1]
xgb_test = xgb.predict_proba(test_final)[:, 1]
print(f"XGB Score: {roc_auc_score(y_val, xgb_pred):.5f}")

# LightGBM
lgbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, max_depth=10, num_leaves=64,
                          metric='auc', random_state=42, n_jobs=-1, verbose=-1)
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
lgbm_pred = lgbm.predict_proba(X_val)[:, 1]
lgbm_test = lgbm.predict_proba(test_final)[:, 1]
print(f"LGBM Score: {roc_auc_score(y_val, lgbm_pred):.5f}")

# CatBoost
cat = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=6, eval_metric='AUC', 
                         random_seed=42, verbose=0, early_stopping_rounds=50, allow_writing_files=False)
cat.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_pred = cat.predict_proba(X_val)[:, 1]
cat_test = cat.predict_proba(test_final)[:, 1]
print(f"Cat Score: {roc_auc_score(y_val, cat_pred):.5f}")

# --- 3. AĞIRLIKLI ENSEMBLE (Weighted Blend) ---
# CatBoost ve LGBM genelde daha sağlamdır, onlara daha çok güvenelim.
# Formül: %40 Cat + %35 LGBM + %25 XGB

w_cat = 0.40
w_lgb = 0.35
w_xgb = 0.25

ensemble_val = (cat_pred * w_cat) + (lgbm_pred * w_lgb) + (xgb_pred * w_xgb)
print(f"\n Weighted Ensemble Score: {roc_auc_score(y_val, ensemble_val):.5f}")

# Submission
final_preds = (cat_test * w_cat) + (lgbm_test * w_lgb) + (xgb_test * w_xgb)
submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': final_preds})
submission.to_csv('submission_v4_features.csv', index=False)
print("\n'submission_v4_features.csv' +")

