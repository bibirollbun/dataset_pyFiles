# Carga de Librerias

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import lightgbm as lgb


# Definicion de Paths y variables

COMP_PATH = '/kaggle/input/petfinder-adoption-prediction/'
IMG_PATH = '/kaggle/input/petfinder-preprocesado-img-large-all/'
TEXT_PATH = '/kaggle/input/petfinder-preprocesado-texto-all/'
N_FOLDS = 5
SEED = 42


# Carga de datos

train = pd.read_csv(f'{COMP_PATH}train/train.csv')
test = pd.read_csv(f'{COMP_PATH}test/test.csv')


# Embeddings: imágenes (ViT-large) + texto (DeBERTa)

train = train.merge(pd.read_parquet(f'{IMG_PATH}train.parquet'), left_on='PetID', right_index=True, how='left')
train = train.merge(pd.read_parquet(f'{TEXT_PATH}train_text.parquet'), left_on='PetID', right_index=True, how='left')
test = test.merge(pd.read_parquet(f'{IMG_PATH}test.parquet'), left_on='PetID', right_index=True, how='left')
test = test.merge(pd.read_parquet(f'{TEXT_PATH}test_text.parquet'), left_on='PetID', right_index=True, how='left')

print(f"Train: {train.shape}, Test: {test.shape}")


# RESCUERID features

all_data = pd.concat([train[['RescuerID', 'Type', 'Breed1', 'Fee', 'Quantity']],
                      test[['RescuerID', 'Type', 'Breed1', 'Fee', 'Quantity']]], ignore_index=True)

rescuer_stats = all_data.groupby('RescuerID').agg({
    'Type': ['count', 'nunique'],
    'Breed1': 'nunique',
    'Fee': 'mean',
    'Quantity': 'sum'
}).reset_index()

rescuer_stats.columns = ['RescuerID', 'res_pet_count', 'res_type_nunique',
                         'res_breed_nunique', 'res_fee_mean', 'res_quantity_sum']

train = train.merge(rescuer_stats, on='RescuerID', how='left')
test = test.merge(rescuer_stats, on='RescuerID', how='left')

print(f"Train después de rescuer features: {train.shape}")


# Preparar datos

exclude = ['PetID', 'Name', 'Description', 'AdoptionSpeed', 'RescuerID']
feature_cols = [c for c in train.columns if c not in exclude]
X = train[feature_cols]
y = train['AdoptionSpeed']
X_test = test[feature_cols]

print(f"Features: {len(feature_cols)}")


# Entrenar lightGBM

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'n_estimators': 1000,
    'random_state': SEED,
    'verbose': -1
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMRegressor(**params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds[val_idx] = model.predict(X.iloc[val_idx])
    test_preds += model.predict(X_test) / N_FOLDS

    kappa = cohen_kappa_score(y.iloc[val_idx], np.round(oof_preds[val_idx]).astype(int).clip(0,4), weights='quadratic')
    print(f"Fold {fold}: QWK = {kappa:.4f}")

print(f"\nOOF QWK (rounded): {cohen_kappa_score(y, np.round(oof_preds).astype(int).clip(0,4), weights='quadratic'):.4f}")


# Threshold optimization

def qwk_loss(th, y_true, y_pred):
    return -cohen_kappa_score(y_true, np.digitize(y_pred, np.sort(th)), weights='quadratic')

result = minimize(qwk_loss, [0.5, 1.5, 2.5, 3.5], args=(y, oof_preds), method='Nelder-Mead')
thresholds = np.sort(result.x)

print(f"\nThresholds óptimos: {thresholds.round(4)}")
print(f"OOF QWK (optimizado): {cohen_kappa_score(y, np.digitize(oof_preds, thresholds), weights='quadratic'):.4f}")


# Submit
submission = pd.DataFrame({'PetID': test['PetID'], 'AdoptionSpeed': np.digitize(test_preds, thresholds)})
submission.to_csv('submission.csv', index=False)
print("\n✓ submission.csv guardado")

