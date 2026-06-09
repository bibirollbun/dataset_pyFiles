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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MODELO FINAL - MEJOR CONFIGURACIÓN")
print("="*60)
print("Parámetros optimizados del Grid Search (Combinación 203)")
print("Kappa esperado: 0.3741 (+/- 0.0054)")
print("="*60)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Paths originales
train_path = '/kaggle/input/petfinder-adoption-prediction/train/train.csv'
test_path = '/kaggle/input/petfinder-adoption-prediction/test/test.csv'

# Paths de features preprocesadas (AJUSTAR SEGÚN TUS DATASETS EN KAGGLE)
train_text_path = '/kaggle/input/petfinder-text-features/train_text_features.csv'
test_text_path = '/kaggle/input/petfinder-text-features/test_text_features.csv'
train_image_path = '/kaggle/input/petfinder-image-features/train_image_features.csv'
test_image_path = '/kaggle/input/petfinder-image-features/test_image_features.csv'

# Cargar datos principales
print("\nCargando datos principales...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Guardar el target
y = train_df['AdoptionSpeed'].copy()

# Cargar features preprocesadas
print("\nCargando features preprocesadas...")
train_text_features = pd.read_csv(train_text_path)
test_text_features = pd.read_csv(test_text_path)
train_image_features = pd.read_csv(train_image_path)
test_image_features = pd.read_csv(test_image_path)

print(f"Text features - Train: {train_text_features.shape}, Test: {test_text_features.shape}")
print(f"Image features - Train: {train_image_features.shape}, Test: {test_image_features.shape}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_tabular_features(df):
    """Crea características adicionales de datos tabulares"""
    df = df.copy()
    
    # Features de combinaciones
    df['Age_MaturitySize'] = df['Age'] * df['MaturitySize']
    df['Age_FurLength'] = df['Age'] * df['FurLength']
    df['Fee_Health'] = df['Fee'] * df['Health']
    df['Fee_Vaccinated'] = df['Fee'] * df['Vaccinated']
    df['Fee_Dewormed'] = df['Fee'] * df['Dewormed']
    df['Fee_Sterilized'] = df['Fee'] * df['Sterilized']
    
    # Features de agrupación
    df['Health_Total'] = df['Vaccinated'] + df['Dewormed'] + df['Sterilized']
    df['is_fully_healthy'] = (df['Health_Total'] == 3).astype(int)
    
    # Features de color
    df['Color_count'] = (df['Color1'] != 0).astype(int) + \
                        (df['Color2'] != 0).astype(int) + \
                        (df['Color3'] != 0).astype(int)
    df['is_single_color'] = (df['Color_count'] == 1).astype(int)
    df['is_multicolor'] = (df['Color_count'] > 1).astype(int)
    
    # Features de video/foto
    df['Total_media'] = df['PhotoAmt'] + df['VideoAmt']
    df['has_video'] = (df['VideoAmt'] > 0).astype(int)
    df['has_photo'] = (df['PhotoAmt'] > 0).astype(int)
    df['media_richness'] = df['PhotoAmt'] * 1 + df['VideoAmt'] * 2
    
    # Features de fee
    df['is_free'] = (df['Fee'] == 0).astype(int)
    df['fee_log'] = np.log1p(df['Fee'])
    
    # Features de edad
    df['age_in_years'] = df['Age'] / 12
    df['is_puppy_kitten'] = (df['Age'] <= 3).astype(int)
    df['is_young'] = ((df['Age'] > 3) & (df['Age'] <= 12)).astype(int)
    df['is_adult'] = ((df['Age'] > 12) & (df['Age'] <= 60)).astype(int)
    df['is_senior'] = (df['Age'] > 60).astype(int)
    
    # Features de tipo
    df['is_dog'] = (df['Type'] == 1).astype(int)
    df['is_cat'] = (df['Type'] == 2).astype(int)
    
    return df

print("\nCreando features tabulares...")
train_df = create_tabular_features(train_df)
test_df = create_tabular_features(test_df)

# ============================================================================
# MERGE DE FEATURES
# ============================================================================

print("\nMergeando features...")
train_df = train_df.merge(train_text_features, on='PetID', how='left', suffixes=('', '_text'))
train_df = train_df.merge(train_image_features, on='PetID', how='left', suffixes=('', '_image'))
test_df = test_df.merge(test_text_features, on='PetID', how='left', suffixes=('', '_text'))
test_df = test_df.merge(test_image_features, on='PetID', how='left', suffixes=('', '_image'))

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# Eliminar columnas duplicadas
duplicate_cols = [col for col in train_df.columns if col.endswith('_text') or col.endswith('_image')]
if duplicate_cols:
    print(f"Eliminando {len(duplicate_cols)} columnas duplicadas...")
    train_df = train_df.drop(columns=duplicate_cols)
    test_df = test_df.drop(columns=[col for col in duplicate_cols if col in test_df.columns])

# ============================================================================
# PREPARAR DATOS
# ============================================================================

features_to_drop = ['PetID', 'Name', 'Description', 'RescuerID', 'AdoptionSpeed', 'sentiment_language']
object_cols = train_df.select_dtypes(include=['object']).columns.tolist()
features_to_drop.extend(object_cols)
features_to_drop = list(set(features_to_drop))

X = train_df.drop([col for col in features_to_drop if col in train_df.columns], axis=1)
X_test = test_df.drop([col for col in features_to_drop if col in test_df.columns], axis=1)

object_cols_remaining = X.select_dtypes(include=['object']).columns.tolist()
if object_cols_remaining:
    print(f"⚠️ Eliminando columnas object restantes: {object_cols_remaining}")
    X = X.drop(columns=object_cols_remaining)
    X_test = X_test.drop(columns=[col for col in object_cols_remaining if col in X_test.columns])

print(f"\n{'='*60}")
print(f"INFORMACIÓN DEL DATASET")
print(f"{'='*60}")
print(f"Número de features: {X.shape[1]}")
print(f"Número de muestras train: {X.shape[0]}")
print(f"Número de muestras test: {X_test.shape[0]}")
print(f"\nDistribución de clases:")
print(y.value_counts().sort_index())

# ============================================================================
# PARÁMETROS ÓPTIMOS - COMBINACIÓN 203 (MEJOR DEL GRID SEARCH)
# ============================================================================

params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'max_depth': 5,
    'min_child_samples': 10,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

print(f"\n{'='*60}")
print("PARÁMETROS DEL MODELO FINAL")
print(f"{'='*60}")
print(f"Basados en Grid Search - Combinación 203")
print(f"Kappa esperado (3-fold): 0.3741 (+/- 0.0054)")
print(f"\nHiperparámetros:")
for key, value in params.items():
    if key not in ['objective', 'num_class', 'metric', 'boosting_type', 'random_state', 'n_jobs', 'verbose']:
        print(f"  {key:25s} = {value}")

# ============================================================================
# ENTRENAMIENTO CON 5-FOLD CV
# ============================================================================

n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = np.zeros((len(X), 5))
test_predictions = np.zeros((len(X_test), 5))
kappa_scores = []
feature_importance_list = []

print(f"\n{'='*60}")
print("ENTRENAMIENTO CON CROSS-VALIDATION")
print(f"{'='*60}")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'─'*60}")
    print(f"FOLD {fold + 1}/{n_folds}")
    print(f"{'─'*60}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predicciones OOF
    oof_predictions[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Predicciones de test
    test_predictions += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
    
    # Calcular Cohen's Kappa
    oof_pred_classes = np.argmax(oof_predictions[val_idx], axis=1)
    kappa = cohen_kappa_score(y_val, oof_pred_classes, weights='quadratic')
    kappa_scores.append(kappa)
    
    # Guardar feature importance
    fold_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain'),
        'fold': fold + 1
    })
    feature_importance_list.append(fold_importance)
    
    print(f"\n✓ Fold {fold + 1} Kappa Score: {kappa:.4f}")

# ============================================================================
# RESULTADOS FINALES
# ============================================================================

oof_pred_classes_all = np.argmax(oof_predictions, axis=1)
overall_kappa = cohen_kappa_score(y, oof_pred_classes_all, weights='quadratic')

print(f"\n{'='*60}")
print("RESULTADOS FINALES")
print(f"{'='*60}")
print(f"Overall OOF Kappa Score: {overall_kappa:.4f}")
print(f"Mean CV Kappa Score: {np.mean(kappa_scores):.4f} (+/- {np.std(kappa_scores):.4f})")
print(f"\nKappa por fold:")
for i, kappa in enumerate(kappa_scores):
    print(f"  Fold {i+1}: {kappa:.4f}")

print(f"\nComparación con Grid Search:")
print(f"  Grid Search (3-fold):  0.3741 (+/- 0.0054)")
print(f"  Este modelo (5-fold):  {overall_kappa:.4f} (+/- {np.std(kappa_scores):.4f})")
print(f"  Diferencia:            {overall_kappa - 0.3741:+.4f}")



# ============================================================================
# CREAR SUBMISSION
# ============================================================================

submission = pd.DataFrame({
    'PetID': test_df['PetID'],
    'AdoptionSpeed': np.argmax(test_predictions, axis=1)
})

submission.to_csv('submission.csv', index=False)

print(f"\n{'='*60}")
print("SUBMISSION CREADO ✅")
print(f"{'='*60}")
print(f"✓ Archivo: submission.csv")
print(f"✓ Shape: {submission.shape}")
print(f"\nDistribución de predicciones:")
dist = submission['AdoptionSpeed'].value_counts().sort_index()
for speed, count in dist.items():
    percentage = (count / len(submission)) * 100
    print(f"  Clase {speed}: {count:4d} ({percentage:5.2f}%)")




print(f"\n{'='*60}")
print("PROCESO COMPLETADO EXITOSAMENTE ✅")
print(f"{'='*60}")


