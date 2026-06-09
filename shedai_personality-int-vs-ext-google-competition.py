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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# 1. Veri Yükleme
train_path = '/kaggle/input/playground-series-s5e7/train.csv'
test_path  = '/kaggle/input/playground-series-s5e7/test.csv'
subm_path  = 'submission.csv'

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# 2. Özellik / Hedef Ayrımı
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
X_test = test.drop(['id'], axis=1)
ids = test['id']

# 3. Sayısal ve Kategorik Değişkenleri Belirleme
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 4. Eksik Değer İmputasyonu
#   - Sayısallar: Medyan
#   - Kategorikler: Mod
num_imputer = SimpleImputer(strategy='median')
X_num      = num_imputer.fit_transform(X[num_cols])
X_test_num = num_imputer.transform(X_test[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat       = cat_imputer.fit_transform(X[cat_cols])
X_test_cat  = cat_imputer.transform(X_test[cat_cols])

# 5. Kategoriklerin Sıralı Kodlanması
ord_enc       = OrdinalEncoder()
X_cat_enc     = ord_enc.fit_transform(X_cat)
X_test_cat_enc= ord_enc.transform(X_test_cat)

# 6. İşlenmiş Özellik Matrisi
X_proc      = np.hstack([X_num,       X_cat_enc])
X_test_proc = np.hstack([X_test_num,  X_test_cat_enc])

# 7. Hedef Değişkenin Etiketlenmesi
le    = LabelEncoder()
y_enc = le.fit_transform(y)  # Introvert→0, Extrovert→1 (örnek)

# 8. Model Tanımı ve Çapraz Doğrulama
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_proc, y_enc, cv=cv, scoring='accuracy')
print("Beş Katlı Stratifiye CV Doğruluk Skorları:", np.round(scores, 4))
print("Ortalama CV Doğruluğu:        ", np.round(scores.mean(), 4))

# 9. Tüm Veriyle Eğit ve Teste Tahmin Üret
model.fit(X_proc, y_enc)
pred_enc = model.predict(X_test_proc)
pred      = le.inverse_transform(pred_enc)

# 10. Submission Dosyası Oluşturma
submission = pd.DataFrame({
    'id': ids,
    'Personality': pred
})
submission.to_csv(subm_path, index=False)
print(f"Submission dosyası kaydedildi: {subm_path}")



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
# from category_encoders import TargetEncoder   # Ekstra: pip install category-encoders

# 1) Veri Yükleme
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
X     = train.drop(['id','Personality'], axis=1)
y     = train['Personality']

# 2) Kolon Tiplerini Tespit Et
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 3) Ortak Ayarlar
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4) Pipeline A: Temel (Ordinal + Medyan)
pipe_A = Pipeline([
    ('pre', ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ord', OrdinalEncoder())
        ]), cat_cols),
    ])),
    ('clf', model)
])

# 5) Pipeline B: One-Hot Encoding
pipe_B = Pipeline([
    ('pre', ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]), cat_cols),
    ])),
    ('clf', model)
])

# 6) Pipeline C: Eksik Göstergesi + One-Hot
pipe_C = Pipeline([
    ('pre', ColumnTransformer([
        ('num', SimpleImputer(strategy='median', add_indicator=True), num_cols),
        ('cat', Pipeline([
            ('impute',    SimpleImputer(strategy='constant', fill_value='MISSING', add_indicator=True)),
            ('ohe',       OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]), cat_cols),
    ])),
    ('clf', model)
])

# 7) Pipeline D: (İsteğe Bağlı) Hedef Kodlama (Target Encoding)
# pipe_D = Pipeline([
#     ('pre', ColumnTransformer([
#         ('num', SimpleImputer(strategy='median'), num_cols),
#         ('cat', Pipeline([
#             ('impute', SimpleImputer(strategy='most_frequent')),
#             ('tgt',   TargetEncoder())
#         ]), cat_cols),
#     ])),
#     ('clf', model)
# ])

# 8) Değerlendirme
pipelines = {'Ordinal+Medyan': pipe_A,
             'OHE':            pipe_B,
             'Ind+OHE':        pipe_C,
             # 'TargetEnc':      pipe_D
            }

for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"{name:12s} → CV Acc: {scores.mean():.4f}  (std {scores.std():.4f})")



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- 1. Load Data ---
# Load the training, testing, and sample submission files.
# The file paths are assumed to be as specified in the problem description.
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
except FileNotFoundError:
    print("--- File Loading Error ---")
    print("The specified file paths were not found.")
    print("Please ensure the CSV files are in the correct directory:")
    print("/kaggle/input/playground-series-s5e7/")
    # As a fallback for local execution, you can try loading from the current directory.
    # To use this, place your CSVs in the same folder as the script.
    try:
        print("\nAttempting to load files from the current directory...")
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sample_submission_df = pd.read_csv('sample_submission.csv')
        print("Files loaded successfully from the current directory.")
    except FileNotFoundError:
        print("Fallback failed. Could not find CSV files in the current directory either.")
        exit() # Exit the script if data cannot be loaded.


print("--- Data Loaded Successfully ---")
print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")
print("-" * 30)


# --- 2. Preprocessing and Feature Engineering ---

# The 'id' column is just an identifier and not a predictive feature.
# We'll keep the test IDs for the submission file.
train_ids = train_df['id']
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# The target variable 'Personality' is categorical (Introvert/Extrovert).
# Machine learning models require numerical input, so we'll encode these labels.
# 'Introvert' will become 0 and 'Extrovert' will become 1.
label_encoder = LabelEncoder()
train_df['Personality'] = label_encoder.fit_transform(train_df['Personality'])

# Separate features (X) from the target (y)
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
X_test = test_df

# *** FIX STARTS HERE ***
# Identify categorical features (those with 'object' dtype)
categorical_features = X.select_dtypes(include=['object']).columns
print(f"Found categorical features: {list(categorical_features)}")

# Convert categorical features to numerical codes
# We loop through each categorical column and use pandas' .cat.codes
# to assign a unique integer to each unique string value.
for col in categorical_features:
    # Combine train and test to ensure all categories are seen
    combined = pd.concat([X[col], X_test[col]], axis=0)
    # Convert to category type
    combined_cat = pd.Categorical(combined)
    # Assign the integer codes back to the respective dataframes
    X[col] = combined_cat[:len(X)].codes
    X_test[col] = combined_cat[len(X):].codes
# *** FIX ENDS HERE ***


print("\n--- Preprocessing Complete ---")
print("Categorical features have been encoded.")
print("Final feature types in training data:")
print(X.dtypes.value_counts())
print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("-" * 30)


# --- 3. Model Training (LightGBM with Cross-Validation) ---

# We will use a LightGBM classifier, which is a powerful and efficient gradient boosting model,
# excellent for tabular data competitions.

# To make our model robust, we'll use Stratified K-Fold Cross-Validation.
# This ensures that each 'fold' of the data has the same proportion of Introverts and Extroverts
# as the original dataset, which is important for imbalanced datasets.

# Model Parameters for LightGBM
# These are a good starting point and can be tuned for better performance.
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'num_leaves': 20,
    'max_depth': 5,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

print("--- Starting Model Training ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    
    # Use early stopping to find the optimal number of boosting rounds
    callbacks = [lgb.early_stopping(100, verbose=False)]
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='accuracy',
              callbacks=callbacks)

    # Store predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

# Evaluate overall out-of-fold (OOF) accuracy
oof_accuracy = accuracy_score(y, np.round(oof_preds))
print("-" * 30)
print(f"--- Training Complete ---")
print(f"Overall OOF Accuracy: {oof_accuracy:.5f}")
print("-" * 30)


# --- 4. Create Submission File ---

# The model predicts probabilities. We'll convert them to 0s and 1s
# by rounding (threshold of 0.5).
final_predictions_encoded = np.round(test_preds).astype(int)

# Now, we convert the encoded predictions back to the original labels ('Introvert'/'Extrovert').
final_predictions = label_encoder.inverse_transform(final_predictions_encoded)

# Create the submission DataFrame in the required format.
submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})

# Save the submission file.
submission_df.to_csv('submission.csv', index=False)

print("--- Submission File Created ---")
print("File saved as 'submission.csv'")
print(submission_df.head())
print("-" * 30)



import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna # Optuna kütüphanesini içe aktarıyoruz
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- 1. Veri Yükleme ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
except FileNotFoundError:
    print("--- Dosya Yükleme Hatası ---")
    print("Mevcut dizinden dosyalar yüklenmeye çalışılıyor...")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sample_submission_df = pd.read_csv('sample_submission.csv')
        print("Dosyalar başarıyla yüklendi.")
    except FileNotFoundError:
        print("Dosyalar bulunamadı. Lütfen dosya yollarını kontrol edin.")
        exit()

print("--- Veri Başarıyla Yüklendi ---")
print(f"Eğitim verisi boyutu: {train_df.shape}")
print(f"Test verisi boyutu: {test_df.shape}")
print("-" * 30)


# --- 2. Ön İşleme ---
train_ids = train_df['id']
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

target_encoder = LabelEncoder()
train_df['Personality'] = target_encoder.fit_transform(train_df['Personality'])

X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
X_test = test_df

categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    combined = pd.concat([X[col], X_test[col]], axis=0)
    combined_cat = pd.Categorical(combined)
    X[col] = combined_cat[:len(X)].codes
    X_test[col] = combined_cat[len(X):].codes

print("--- Ön İşleme Tamamlandı ---")


# --- 3. Hiperparametre Optimizasyonu (Optuna) ---

def objective(trial):
    """
    Optuna'nın her denemede çalıştıracağı ve optimize edeceği fonksiyon.
    Farklı hiperparametreleri dener ve doğruluk skorunu döndürür.
    """
    # Denenecek hiperparametre aralıklarını belirliyoruz
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1000, # Daha sonra early stopping ile en iyisi bulunacak
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Optimizasyon için daha az fold
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        callbacks = [lgb.early_stopping(50, verbose=False)]
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='accuracy',
                  callbacks=callbacks)
        oof_preds[val_idx] = model.predict(X_val)

    accuracy = accuracy_score(y, oof_preds)
    return accuracy

print("--- Hiperparametre Optimizasyonu Başlatılıyor ---")
# Optuna çalışmasını (study) oluşturuyoruz. Amacımız doğruluğu 'maksimize etmek'.
study = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
# Optimizasyonu 25 deneme ile sınırlıyoruz. Bu sayıyı artırarak daha iyi sonuçlar bulabilirsiniz.
study.optimize(objective, n_trials=25)

print("--- Optimizasyon Tamamlandı ---")
print(f"En iyi deneme (Best trial):")
print(f"  Değer (Accuracy): {study.best_value:.5f}")
print(f"  En iyi parametreler:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print("-" * 30)

# --- 4. En İyi Parametrelerle Final Modelini Eğitme ---
best_params = study.best_params
# Optuna'nın bulduğu en iyi parametrelere sabit parametreleri ekliyoruz
final_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 2000, # Yüksek bir değer, early stopping en iyisini bulacak
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    **best_params # Optuna'nın bulduğu en iyi parametreleri ekle
}


N_SPLITS = 10 # Final model için daha fazla fold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

print("--- Final Modeli Eğitiliyor ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**final_params)
    callbacks = [lgb.early_stopping(100, verbose=False)]
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='accuracy',
              callbacks=callbacks)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

oof_accuracy = accuracy_score(y, np.round(oof_preds))
print("-" * 30)
print(f"--- Eğitim Tamamlandı ---")
print(f"Final Model OOF Doğruluk: {oof_accuracy:.5f}")
print("-" * 30)

# --- 5. Teslim Dosyası Oluşturma ---
final_predictions_encoded = np.round(test_preds).astype(int)
final_predictions = target_encoder.inverse_transform(final_predictions_encoded)
submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
submission_df.to_csv('submission_optimized.csv', index=False)

print("--- Teslim Dosyası Oluşturuldu ---")
print("Dosya 'submission_optimized.csv' olarak kaydedildi.")
print(submission_df.head())
print("-" * 30)



import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- 1. Veri Yükleme ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
except FileNotFoundError:
    print("Mevcut dizinden dosyalar yükleniyor...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

print("--- Veri Başarıyla Yüklendi ---")

# --- 2. Ön İşleme ve ÖZELLİK MÜHENDİSLİĞİ ---

# ID ve Hedef Değişkeni Ayırma
train_ids = train_df['id']
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

target_encoder = LabelEncoder()
train_df['Personality'] = target_encoder.fit_transform(train_df['Personality'])

X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
X_test = test_df

# Kategorik özellikleri sayısal hale getirme
categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    combined = pd.concat([X[col], X_test[col]], axis=0)
    combined_cat = pd.Categorical(combined)
    X[col] = combined_cat[:len(X)].codes
    X_test[col] = combined_cat[len(X):].codes

print("Temel ön işleme tamamlandı.")

# *** YENİ ADIM: ÖZELLİK MÜHENDİSLİĞİ ***
print("Yeni özellikler oluşturuluyor...")
for df in [X, X_test]:
    # Sayısal özelliklerin adlarını al (tüm özellikler artık sayısal)
    numeric_cols = df.columns
    
    # Yeni toplulaştırma özelliklerini ekle
    df['feat_mean'] = df[numeric_cols].mean(axis=1)
    df['feat_sum'] = df[numeric_cols].sum(axis=1)
    df['feat_std'] = df[numeric_cols].std(axis=1)
    df['feat_min'] = df[numeric_cols].min(axis=1)
    df['feat_max'] = df[numeric_cols].max(axis=1)

print(f"Yeni özellikler eklendi. Yeni özellik sayısı: {X.shape[1]}")
print("-" * 30)

# --- 3. Hiperparametre Optimizasyonu (Optuna) ---
# (Bu adımı tekrar çalıştırmak yerine, bir önceki adımdaki en iyi parametreleri
# kullanmak süreci hızlandırır. Ancak en doğru sonuç için tekrar çalıştırmak en iyisidir.)
# Bu örnekte, süreci göstermek için Optuna'yı tekrar çalıştırıyoruz.

def objective(trial):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model = lgb.LGBMClassifier(**params)
        callbacks = [lgb.early_stopping(50, verbose=False)]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='accuracy', callbacks=callbacks)
        oof_preds[val_idx] = model.predict(X_val)
    accuracy = accuracy_score(y, oof_preds)
    return accuracy

print("--- Hiperparametre Optimizasyonu Başlatılıyor ---")
study = optuna.create_study(direction='maximize', study_name='lgbm_feature_eng')
study.optimize(objective, n_trials=25) # n_trials'ı artırarak daha iyi sonuç arayabilirsiniz
best_params = study.best_params
print(f"En iyi OOF skoru: {study.best_value:.5f}")
print("-" * 30)

# --- 4. Final Modelini Eğitme ve Teslim Dosyası Oluşturma ---
final_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'n_estimators': 2000, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    **best_params
}

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

print("--- Final Modeli Eğitiliyor ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    model = lgb.LGBMClassifier(**final_params)
    callbacks = [lgb.early_stopping(100, verbose=False)]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='accuracy', callbacks=callbacks)
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

# --- 5. Teslim Dosyası Oluşturma ---
final_predictions = target_encoder.inverse_transform(np.round(test_preds).astype(int))
submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
submission_df.to_csv('submission_feature_eng.csv', index=False)
print("--- Teslim Dosyası 'submission_feature_eng.csv' olarak oluşturuldu. ---")



import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- 1. Veri Yükleme ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
except FileNotFoundError:
    print("Mevcut dizinden dosyalar yükleniyor...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

print("--- Veri Başarıyla Yüklendi ---")

# --- 2. Ön İşleme ---
train_ids = train_df['id']
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

target_encoder = LabelEncoder()
train_df['Personality'] = target_encoder.fit_transform(train_df['Personality'])

X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
X_test = test_df

# CatBoost'a hangi sütunların kategorik olduğunu söylüyoruz.
categorical_features_indices = np.where(X.dtypes == 'object')[0]
print(f"Kategorik özelliklerin indeksleri: {list(categorical_features_indices)}")

# *** HATA DÜZELTME ADIMI BAŞLANGIÇ ***
# CatBoost, kategorik özelliklerde NaN (float) yerine string bekler.
# Bu yüzden object tipindeki sütunlardaki boş değerleri bir metinle dolduruyoruz.
categorical_feature_names = X.columns[categorical_features_indices]
print(f"Doldurulacak kategorik özellikler: {list(categorical_feature_names)}")

for col in categorical_feature_names:
    X[col].fillna('Missing', inplace=True)
    X_test[col].fillna('Missing', inplace=True)

print("Kategorik sütunlardaki boş değerler 'Missing' ile dolduruldu.")
# *** HATA DÜZELTME ADIMI SON ***
print("-" * 30)


# --- 3. CatBoost Modeli ile Eğitim ---
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
oof_accuracy_scores = []

print("--- CatBoost Modeli Eğitiliyor ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=6,
        loss_function='Logloss',
        eval_metric='Accuracy',
        cat_features=categorical_features_indices,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    oof_accuracy_scores.append(accuracy_score(y_val, np.round(model.predict(X_val))))
    
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

print("-" * 30)
print(f"--- Eğitim Tamamlandı ---")
print(f"Ortalama OOF Doğruluk: {np.mean(oof_accuracy_scores):.5f}")
print("-" * 30)

# --- 4. Teslim Dosyası Oluşturma ---
final_predictions_encoded = np.round(test_preds).astype(int)
final_predictions = target_encoder.inverse_transform(final_predictions_encoded)

submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
submission_df.to_csv('submission_catboost_fixed.csv', index=False)

print("--- Teslim Dosyası Oluşturuldu ---")
print("Dosya 'submission_catboost_fixed.csv' olarak kaydedildi.")
print(submission_df.head())
print("-" * 30)



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# --- 1. Veri Yükleme ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
except FileNotFoundError:
    print("Mevcut dizinden dosyalar yükleniyor...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

print("--- Veri Başarıyla Yüklendi ---")

# --- 2. Gelişmiş Veri Ön İşleme (Imputation Baseline) ---
train_ids = train_df['id']
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

target_encoder = LabelEncoder()
train_df['Personality'] = target_encoder.fit_transform(train_df['Personality'])

combined_df = pd.concat([train_df.drop('Personality', axis=1), test_df], axis=0)

categorical_cols = combined_df.select_dtypes(include=['object']).columns
numerical_cols = combined_df.select_dtypes(include=np.number).columns

print("--- Imputation Stratejisi: Medyan / Sabit Değer ---")

# *** HATA DÜZELTME ADIMI 1: SAYISAL DOLDURMA ***
# Sayısal sütunlardaki boş değerleri medyan ile doldur
for col in numerical_cols:
    if combined_df[col].isnull().any():
        median_val = combined_df[col].median()
        combined_df[col].fillna(median_val, inplace=True)
        print(f"Sayısal sütun '{col}' içindeki boş değerler medyan ({median_val}) ile dolduruldu.")

# Kategorik sütunlardaki boş değerleri 'Missing' ile doldur
for col in categorical_cols:
    combined_df[col].fillna('Missing', inplace=True)

print("Tüm boş değerler dolduruldu.")

# One-Hot Encoding ve Ölçeklendirme
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)
scaler = StandardScaler()
combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])

X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]
y = train_df['Personality']

# Giriş verisinde NaN kontrolü
assert not X.isnull().values.any(), "X içinde hala NaN var!"
assert not X_test.isnull().values.any(), "X_test içinde hala NaN var!"

print(f"Ön işleme sonrası özellik sayısı: {X.shape[1]}")
print("-" * 30)

# --- 3. Daha Stabil Derin Öğrenme Modeli Mimarisi ---
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=[input_shape]),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # *** HATA DÜZELTME ADIMI 2: DAHA STABİL OPTIMIZER ***
    # Düşük öğrenme oranı ve gradient clipping ile modeli daha stabil hale getir
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 4. Model Eğitimi (Cross-Validation ile) ---
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
oof_accuracy_scores = []

print("--- Derin Öğrenme Modeli Eğitiliyor ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = build_model(X.shape[1])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=200, batch_size=64,
              callbacks=[early_stopping, reduce_lr],
              verbose=0)

    val_preds = model.predict(X_val).flatten()
    
    # Tahminlerde NaN kontrolü
    if np.isnan(val_preds).any():
        print(f"UYARI: Fold {fold+1} NaN tahminler üretti! Bu fold atlanıyor.")
        continue # Bu fold'u atla

    oof_preds[val_idx] = val_preds
    oof_accuracy_scores.append(accuracy_score(y_val, np.round(val_preds)))
    test_preds += model.predict(X_test).flatten() / N_SPLITS
    print(f"  Fold Accuracy: {oof_accuracy_scores[-1]:.5f}")

print("-" * 30)
print(f"--- Eğitim Tamamlandı ---")
if oof_accuracy_scores:
    print(f"Ortalama OOF Doğruluk: {np.mean(oof_accuracy_scores):.5f}")
else:
    print("Tüm fold'lar NaN ürettiği için doğruluk hesaplanamadı.")
print("-" * 30)

# --- 5. Teslim Dosyası Oluşturma ---
final_predictions_encoded = np.round(test_preds).astype(int)
final_predictions = target_encoder.inverse_transform(final_predictions_encoded)
submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
submission_df.to_csv('submission_dl_imputed_baseline.csv', index=False)
print("--- Teslim Dosyası 'submission_dl_imputed_baseline.csv' olarak oluşturuldu. ---")



import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# 1. Veri Yükleme
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
X     = train.drop(['id','Personality'], axis=1)
y     = train['Personality']
X_test= test.drop(['id'], axis=1)
ids   = test['id']

# 2. Özellik Tiplerini Belirleme
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 3. Ön-işleme Tanımı: Medyan imputing + One-Hot Encoding
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imp', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]), cat_cols)
])

# 4. Model Tanımları
models = {
    'RandomForest':     RandomForestClassifier(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost':          XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
}

# 5. Ensemble: Stacking
estimators = []
for name, clf in models.items():
    estimators.append((name, Pipeline([('pre', preprocessor), ('clf', clf)])))
stack = StackingClassifier(estimators=estimators,
                           final_estimator=LogisticRegression(),
                           cv=5)

# 6. Tüm Yöntemleri Listele
pipelines = {**models, 'Stacking': stack}

# 7. Değerlendirme
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
le   = LabelEncoder()
y_enc= le.fit_transform(y)

print("=== Five-Fold CV Accuracy ===")
for name, clf in pipelines.items():
    if name != 'Stacking':
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    else:
        pipe = stack
    scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring='accuracy')
    print(f"{name:15s}: {scores.mean():.4f} ± {scores.std():.4f}")

# 8. En İyi Model ile Tahmin ve Kaydetme (Örnek: Stacking)
best_model = stack
best_model.fit(X, y_enc)
preds = best_model.predict(X_test)
preds = le.inverse_transform(preds)

submission = pd.DataFrame({'id': ids, 'Personality': preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Ensemble tahminler submission.csv olarak kaydedildi.")



import pandas as pd
import numpy as np
import warnings

# Ön İşleme Araçları
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb

# Ayarlar
warnings.filterwarnings('ignore')

# --- 1. Veri Yükleme ---
print("Veri setleri yükleniyor...")
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
except FileNotFoundError:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')

# --- 2. Veri Hazırlığı ve Ön İşleme Pipeline'ı ---
print("Ön işleme pipeline'ı oluşturuluyor...")

# Hedef değişkeni ve özellikleri ayırma
X = train_df.drop("Personality", axis=1)
y_raw = train_df["Personality"]
X_test = test_df.copy() # Orijinal test verisini koru

# ID sütunlarını daha sonra kullanmak için sakla
train_ids = X.pop('id')
test_ids = X_test.pop('id')

# Hedef değişkeni kodlama (Introvert/Extrovert -> 0/1)
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Sayısal ve kategorik sütunları belirleme
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Sayısal veriler için ön işleme adımları
# Adım 1: Boş değerleri medyan ile doldur
# Adım 2: Veriyi standartlaştır (ortalaması 0, standart sapması 1)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Kategorik veriler için ön işleme adımları
# Adım 1: Boş değerleri en sık tekrar eden değerle doldur
# Adım 2: One-Hot Encoding uygula (her kategori için yeni bir sütun oluştur)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# ColumnTransformer ile bu iki pipeline'ı birleştirme
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Geriye kalan sütunlara dokunma
)

# --- 3. Ensemble Modeli (Voting Classifier) ---
print("Ensemble modeli (Voting Classifier) tanımlanıyor...")

# Ensemble içinde kullanılacak temel modeller
clf1 = GradientBoostingClassifier(random_state=42)
clf2 = lgb.LGBMClassifier(random_state=42)
clf3 = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
clf4 = RandomForestClassifier(random_state=42)
clf5 = LogisticRegression(random_state=42)

# Voting Classifier'ı oluşturma
# voting='soft': Modellerin tahmin olasılıklarının ortalamasını alır (genellikle daha iyi sonuç verir)
eclf1 = VotingClassifier(
    estimators=[('gb', clf1), ('lgbm', clf2), ('xgb', clf3), ('rf', clf4), ('lr', clf5)],
    voting='soft',
    weights=[0.15, 0.35, 0.3, 0.1, 0.1] # Notebook'taki ağırlıklar
)

# Tüm pipeline'ı oluşturma: Önce ön işleme, sonra modelleme
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', eclf1)])

# --- 4. Modeli Eğitme ve Tahmin Yapma ---
print("Model tüm eğitim verisiyle eğitiliyor...")
full_pipeline.fit(X, y)

print("Test verisi üzerinde tahminler yapılıyor...")
test_predictions_encoded = full_pipeline.predict(X_test)

# Tahminleri orijinal etiketlere (Introvert/Extrovert) geri çevirme
test_predictions = le.inverse_transform(test_predictions_encoded)

# --- 5. Teslim Dosyasını Oluşturma ---
print("Teslim dosyası oluşturuluyor...")
submission_df = pd.DataFrame({'id': test_ids, 'Personality': test_predictions})
submission_df.to_csv('/kaggle/working/submission_ensemble.csv', index=False)

print("\nİşlem Tamamlandı! 'submission_ensemble.csv' dosyası oluşturuldu.")
print(submission_df.head())



import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna # Optuna kütüphanesini içe aktarıyoruz
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- 1. Veri Yükleme ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
except FileNotFoundError:
    print("--- Dosya Yükleme Hatası ---")
    print("Mevcut dizinden dosyalar yüklenmeye çalışılıyor...")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sample_submission_df = pd.read_csv('sample_submission.csv')
        print("Dosyalar başarıyla yüklendi.")
    except FileNotFoundError:
        print("Dosyalar bulunamadı. Lütfen dosya yollarını kontrol edin.")
        exit()

print("--- Veri Başarıyla Yüklendi ---")
print(f"Eğitim verisi boyutu: {train_df.shape}")
print(f"Test verisi boyutu: {test_df.shape}")
print("-" * 30)


# --- 2. Ön İşleme ---
train_ids = train_df['id']
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

target_encoder = LabelEncoder()
train_df['Personality'] = target_encoder.fit_transform(train_df['Personality'])

X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
X_test = test_df

categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    combined = pd.concat([X[col], X_test[col]], axis=0)
    combined_cat = pd.Categorical(combined)
    X[col] = combined_cat[:len(X)].codes
    X_test[col] = combined_cat[len(X):].codes

print("--- Ön İşleme Tamamlandı ---")


# --- 3. Hiperparametre Optimizasyonu (Optuna) ---

def objective(trial):
    """
    Optuna'nın her denemede çalıştıracağı ve optimize edeceği fonksiyon.
    Farklı hiperparametreleri dener ve doğruluk skorunu döndürür.
    """
    # Denenecek hiperparametre aralıklarını belirliyoruz
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1000, # Daha sonra early stopping ile en iyisi bulunacak
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Optimizasyon için daha az fold
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        callbacks = [lgb.early_stopping(50, verbose=False)]
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='accuracy',
                  callbacks=callbacks)
        oof_preds[val_idx] = model.predict(X_val)

    accuracy = accuracy_score(y, oof_preds)
    return accuracy

print("--- Hiperparametre Optimizasyonu Başlatılıyor ---")
# Optuna çalışmasını (study) oluşturuyoruz. Amacımız doğruluğu 'maksimize etmek'.
study = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
# Optimizasyonu 25 deneme ile sınırlıyoruz. Bu sayıyı artırarak daha iyi sonuçlar bulabilirsiniz.
study.optimize(objective, n_trials=25)

print("--- Optimizasyon Tamamlandı ---")
print(f"En iyi deneme (Best trial):")
print(f"  Değer (Accuracy): {study.best_value:.5f}")
print(f"  En iyi parametreler:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print("-" * 30)

# --- 4. En İyi Parametrelerle Final Modelini Eğitme ---
best_params = study.best_params
# Optuna'nın bulduğu en iyi parametrelere sabit parametreleri ekliyoruz
final_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 2000, # Yüksek bir değer, early stopping en iyisini bulacak
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    **best_params # Optuna'nın bulduğu en iyi parametreleri ekle
}


N_SPLITS = 10 # Final model için daha fazla fold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

print("--- Final Modeli Eğitiliyor ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**final_params)
    callbacks = [lgb.early_stopping(100, verbose=False)]
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='accuracy',
              callbacks=callbacks)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

oof_accuracy = accuracy_score(y, np.round(oof_preds))
print("-" * 30)
print(f"--- Eğitim Tamamlandı ---")
print(f"Final Model OOF Doğruluk: {oof_accuracy:.5f}")
print("-" * 30)

# --- 5. Teslim Dosyası Oluşturma (Gelişmiş) ---
# *** YENİ ADIM BAŞLANGICI ***

# 1. Standart tahminleri yap
# Önce tüm tahminleri standart yuvarlama (0.5 eşiği) ile yapıyoruz.
final_predictions_encoded = np.round(test_preds).astype(int)

# 2. En kararsız tahminleri bul
# Modelin tahmin olasılığının 0.5'e olan uzaklığını hesaplıyoruz.
# Bu uzaklık ne kadar küçükse, model o kadar kararsızdır.
distance_from_half = np.abs(test_preds - 0.5)

# 3. Değiştirilecek tahmin sayısını belirle
# Sorunuzda belirttiğiniz gibi 55 satırı hedefliyoruz.
N_TO_FLIP = 55

# 4. En kararsız N tahmini tersine çevir
if len(test_preds) >= N_TO_FLIP:
    # 0.5'e en yakın N tane tahminin indeksini buluyoruz.
    indices_to_flip = np.argsort(distance_from_half)[:N_TO_FLIP]
    
    print(f"Modelin en kararsız olduğu {N_TO_FLIP} tahmin bulunuyor.")
    print("Bu tahminlerin sonuçları tersine çevriliyor...")
    
    # Bu indekslerdeki tahminleri tersine çeviriyoruz (0 ise 1, 1 ise 0 yap).
    original_values = final_predictions_encoded[indices_to_flip].copy()
    final_predictions_encoded[indices_to_flip] = 1 - final_predictions_encoded[indices_to_flip]

    # Kontrol için birkaç değişikliği yazdır
    print("\nÖrnek Değişiklikler (index: orijinal -> yeni):")
    for i in range(min(5, N_TO_FLIP)):
        idx = indices_to_flip[i]
        print(f"  {test_ids[idx]}: {original_values[i]} -> {final_predictions_encoded[idx]} (Olasılık: {test_preds[idx]:.4f})")

else:
    print(f"UYARI: Değiştirilecek satır sayısı ({N_TO_FLIP}) toplam satır sayısından ({len(test_preds)}) fazla.")

# *** YENİ ADIM SONU ***

# Sonuçları orijinal etiketlere dönüştür
final_predictions = target_encoder.inverse_transform(final_predictions_encoded)
submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
submission_df.to_csv('submission_flipped.csv', index=False)

print("\n--- Teslim Dosyası Oluşturuldu ---")
print("Dosya 'submission_flipped.csv' olarak kaydedildi.")
print(submission_df.head())
print("-" * 30)


