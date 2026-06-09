# Install library yang dibutuhkan
!pip install xgboost lightgbm catboost scikit-learn pandas numpy


import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

# Import Model-Model Terbaik
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Konfigurasi
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
SEED = 42

print("Setup Complete. Libraries Loaded.")


def load_and_preprocess():
    # 1. Load Data
    train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
    
    # Gabungkan sementara untuk preprocessing yang konsisten
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- FEATURE ENGINEERING ---
    # Ide: Kombinasi fitur kesehatan seringkali lebih prediktif daripada fitur tunggal
    
    # Contoh: BMI Risk (Asumsi kolom Weight dan Height ada, jika tidak, sesuaikan)
    # Jika dataset ini mirip NHANES, biasanya ada kolom BMI langsung.
    # Kita buat interaksi umum:
    
    # Cek kolom yang ada (sesuaikan dengan nama kolom asli di dataset s5e12 saat kamu run)
    # Misal ada 'Age', 'BMI', 'GenHlth', 'PhysHlth', dll.
    
    # Log Transform untuk data yang 'skewed' (miring)
    skewed_cols = ['bmi', 'phys_hlth', 'ment_hlth'] # Sesuaikan nama kolom asli
    for col in df.columns:
        if col.lower() in skewed_cols:
            df[f'{col}_log'] = np.log1p(df[col])
            
    # Binning Age (Mengelompokkan umur sering membantu Tree-based model)
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=5, labels=False)
        
    # Interaction Features (Sangat Kuat untuk XGBoost/LGBM)
    # GenHlth * BMI seringkali menjadi indikator kuat diabetes
    if 'GenHlth' in df.columns and 'BMI' in df.columns:
        df['Health_Risk_Index'] = df['GenHlth'] * df['BMI']
    
    # --- END FEATURE ENGINEERING ---

    # Encoding Categorical Features
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Kembalikan ke Train dan Test
    train_df = df[df['is_train'] == 1].drop(['is_train', 'id'], axis=1)
    test_df = df[df['is_train'] == 0].drop(['is_train', 'id', 'diagnosed_diabetes'], axis=1)
    
    return train_df, test_df, submission

train_df, test_df, submission = load_and_preprocess()
print(f"Data Shape: Train {train_df.shape}, Test {test_df.shape}")


# Parameter yang disetel manual untuk performa tinggi di data tabular
lgbm_params = {
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'max_depth': 8,
    'num_leaves': 64,
    'objective': 'binary',
    'metric': 'auc',
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
}

xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'random_state': SEED,
    'n_jobs': -1,
    'enable_categorical': True # XGBoost versi baru support ini
}

cat_params = {
    'iterations': 2000,
    'learning_rate': 0.03,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
    'random_seed': SEED,
    'allow_writing_files': False
}


def train_model(model_class, params, X, y, X_test, model_name):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    print(f"--- Training {model_name} ---")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Inisialisasi Model
        if model_name == 'CatBoost':
            model = model_class(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=0)
        elif model_name == 'LGBM':
            model = model_class(**params)
            # LGBM menggunakan callback untuk early stopping di versi baru
            from lightgbm import early_stopping, log_evaluation
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[early_stopping(100, verbose=False)])
        else: # XGBoost
            model = model_class(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False) # Early stopping dihandle manual atau via param construct
            
        # Prediksi
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / kf.get_n_splits()
        
        score = roc_auc_score(y_val, val_pred)
        # print(f"Fold {fold+1} AUC: {score:.5f}")
        
    print(f"--> Overall CV AUC {model_name}: {roc_auc_score(y, oof_preds):.5f}")
    return oof_preds, test_preds

# Siapkan X dan y
X = train_df.drop('diagnosed_diabetes', axis=1)
y = train_df['diagnosed_diabetes']

# Jalankan Training untuk 3 Model
oof_lgbm, preds_lgbm = train_model(LGBMClassifier, lgbm_params, X, y, test_df, 'LGBM')
oof_xgb, preds_xgb = train_model(XGBClassifier, xgb_params, X, y, test_df, 'XGBoost')
oof_cat, preds_cat = train_model(CatBoostClassifier, cat_params, X, y, test_df, 'CatBoost')


# Strategi Weighted Average (Ubah bobot ini berdasarkan hasil print skor CV di atas)
# Total bobot harus 1.0
w_lgbm = 0.35
w_xgb = 0.30
w_cat = 0.35

final_preds = (preds_lgbm * w_lgbm) + (preds_xgb * w_xgb) + (preds_cat * w_cat)

print("Ensemble Completed.")


submission['diagnosed_diabetes'] = final_preds
submission.to_csv('submissio_ensemble.csv', index=False)
print("File 'submission_ensemble.csv' saved successfully!")

# Cek distribusi prediksi (pastikan tidak ada yang aneh, misal semua 0 atau semua 1)
print(submission['diagnosed_diabetes'].describe())

