import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


train_df.shape, test_df.shape


print(train_df.isna().sum(), test_df.isna().sum())


train_df.drop_duplicates()
test_df.drop_duplicates()


print(train_df.columns, test_df.columns)


train_df.info()


num_cols = train_df.select_dtypes('number')

plt.figure(figsize=(10, 5))
sns.heatmap(num_cols.corr(), annot=True, cmap='inferno', fmt='.2f')
plt.show()


train_df['loan_to_income_ratio'] = train_df['loan_amount'] / train_df['annual_income']
test_df['loan_to_income_ratio'] = test_df['loan_amount'] / test_df['annual_income']


plt.figure(figsize=(10, 5))
sns.scatterplot(data=train_df, x='credit_score', y='interest_rate', hue='loan_paid_back', alpha=0.6)
plt.axvline(train_df['credit_score'].mean(), color='red', linestyle='--', label='Rata-rata Score')
plt.axhline(train_df['interest_rate'].mean(), color='blue', linestyle='--', label='Rata-rata Bunga')

plt.title("Deteksi Anomali: Credit Score vs Interest Rate")
plt.legend()
plt.show()


# 1. Tentukan Batas "Rendah"
# Misal: Kita anggap rendah jika di bawah persentil 25 (Q1)
batas_score_rendah = train_df['credit_score'].quantile(0.25)
batas_bunga_rendah = train_df['interest_rate'].quantile(0.25)

print(f"Batas Score Rendah (< 25%): {batas_score_rendah}")
print(f"Batas Bunga Rendah (< 25%): {batas_bunga_rendah}")

# 2. Filter Data Anomali
# Logika: Score < Batas Rendah DAN Bunga < Batas Rendah
anomali_df = train_df[
    (train_df['credit_score'] < batas_score_rendah) & 
    (train_df['interest_rate'] < batas_bunga_rendah)
]

# 3. Tampilkan Hasil
print(f"\nDitemukan {len(anomali_df)} data anomali.")
if len(anomali_df) > 0:
    print(anomali_df[['credit_score', 'interest_rate', 'loan_paid_back']].head(10))


train_df.shape


# 1. Tandai data anomali
# (Pastikan Anda sudah mendefinisikan batas_score_rendah dan batas_bunga_rendah sebelumnya)
train_df['is_anomali'] = (
    (train_df['credit_score'] < batas_score_rendah) & 
    (train_df['interest_rate'] < batas_bunga_rendah)
).astype(int)

test_df['is_anomali'] = (
    (test_df['credit_score'] < batas_score_rendah) & 
    (test_df['interest_rate'] < batas_bunga_rendah)
).astype(int)

# 2. Bandingkan Tingkat Gagal Bayar (loan_paid_back)
# Ingat: 1 = Bayar, 0 = Gagal (tergantung dataset Anda, sesuaikan logicnya)
perbandingan = train_df.groupby('is_anomali')['loan_paid_back'].agg(['mean', 'count'])

print(perbandingan)


# Fitur Interaksi: Indikator 'Lucky Borrower' (Dapat bunga murah padahal skor jelek)
train_df['Lucky_Borrower'] = 0
train_df.loc[
    (train_df['credit_score'] < batas_score_rendah) & 
    (train_df['interest_rate'] < batas_bunga_rendah), 
    'Lucky_Borrower'
] = 1

test_df['Lucky_Borrower'] = 0
test_df.loc[
    (test_df['credit_score'] < batas_score_rendah) & 
    (test_df['interest_rate'] < batas_bunga_rendah), 
    'Lucky_Borrower'
] = 1

# Cek korelasinya dengan target
print("Korelasi Fitur Baru:", train_df[['Lucky_Borrower', 'loan_paid_back']].corr())


train_df.head()


cat_cols = train_df.select_dtypes('object')

for col in cat_cols.columns:
    print('-------------------------\n')
    print(train_df[col].value_counts())


num_cols = train_df.select_dtypes('number')

plt.figure(figsize=(10, 5))
sns.heatmap(num_cols.corr(), annot=True, cmap='inferno', fmt='.2f')
plt.show()


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

sample_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# ==========================================
# 1. PRE-PROCESSING (Clean & Simple)
# ==========================================

# Gabungkan train dan test sebentar untuk encoding yang konsisten
train_df['is_train'] = 1
test_df['is_train'] = 0
all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# A. Ordinal Mapping (Pendidikan - Menjaga Urutan)
edu_map = {
    'High School': 1, 'Other': 1, "Bachelor's": 2, "Master's": 3, 'PhD': 4
}
all_data['education_level_encoded'] = all_data['education_level'].map(edu_map).fillna(1)

# B. Label Encoding (Grade - Menjaga Urutan)
le = LabelEncoder()
# Ubah ke string dulu jaga-jaga ada yang error
all_data['grade_subgrade'] = all_data['grade_subgrade'].astype(str)
all_data['grade_encoded'] = le.fit_transform(all_data['grade_subgrade'])

# C. Auto-Category untuk Kolom Nominal (Gender, dll)
# LightGBM suka tipe data 'category', tidak perlu One-Hot manual yang bikin berat
nominal_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']
for col in nominal_cols:
    all_data[col] = all_data[col].astype('category')

# D. Pisahkan Kembali Train dan Test
train_final = all_data[all_data['is_train'] == 1].drop(columns=['is_train'])
test_final = all_data[all_data['is_train'] == 0].drop(columns=['is_train', 'loan_paid_back'])

# E. Definisi X dan y
# Buang kolom teks asli yang tidak dipakai
cols_to_drop = ['education_level', 'grade_subgrade', 'loan_paid_back']
X = train_final.drop(columns=cols_to_drop)
y = train_final['loan_paid_back']
X_test_submission = test_final.drop(columns=['education_level', 'grade_subgrade']) # Sesuaikan kolom drop untuk test

print("Preprocessing Selesai. Siap Training.")
print(f"Shape X: {X.shape}, Shape X_test: {X_test_submission.shape}")

# ==========================================
# 2. TRAINING: ENSEMBLE LIGHTGBM (5 FOLDS)
# ==========================================

# Parameter "Pro" (Slow Learning: LR kecil, Estimators banyak)
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 3000,   # Jumlah pohon banyak
    'learning_rate': 0.01,  # Belajar pelan-pelan
    'max_depth': 6,
    'num_leaves': 31,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

# Siapkan wadah untuk hasil ensemble
test_predictions = np.zeros(len(X_test_submission))
oof_preds = np.zeros(len(X)) # Out-of-Fold predictions untuk evaluasi lokal
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

print(f"\nMemulai 5-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Split data per fold
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Train Model
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(500)
        ]
    )
    
    # Simpan prediksi validasi (untuk cek skor lokal)
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Prediksi data test (akumulasi untuk dirata-rata)
    test_predictions += model.predict_proba(X_test_submission)[:, 1] / folds
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")

# ==========================================
# 3. EVALUASI & SUBMISSION
# ==========================================

print("\n" + "="*30)
print(f"OOF AUC Score (Rata-rata Local): {roc_auc_score(y, oof_preds):.5f}")
print("="*30)

# Tentukan Threshold Optimal (Misal dari OOF data)
threshold = 0.45
final_class = np.where(test_predictions > threshold, 1, 0)

# Simpan Submission
submission = pd.DataFrame({
    'id': sample_df['id'],
    'loan_paid_back': final_class
})

submission.to_csv('submission.csv', index=False)

