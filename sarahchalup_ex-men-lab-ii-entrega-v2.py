!pip install -q catboost


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor
from scipy.optimize import minimize

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

### Para mantener la semilla
import os
import random

print("âœ… LibrerÃ­as importadas")


SEED = 42
N_FOLDS = 5
TARGET = 'AdoptionSpeed'
np.random.seed(SEED)

### Semilla de Python      
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)

np.random.seed(SEED)
print(f"âœ… Semillas fijadas globalmente a {SEED}")


train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")

# Cargar features preprocesadas
#train_text = pd.read_parquet("../input/examen-lab-ii-preprocesado-texto-entrega-final/train.parquet")
#test_text = pd.read_parquet("../input/examen-lab-ii-preprocesado-texto-entrega-final/test.parquet")
train_img = pd.read_parquet("../input/examen-lab-ii-preprocesado-img-entrega-final/train.parquet")
test_img = pd.read_parquet("../input/examen-lab-ii-preprocesado-img-entrega-final/test.parquet")

train_text = pd.read_parquet("../input/preprocesado-de-texto-v5/train.parquet")
test_text = pd.read_parquet("../input/preprocesado-de-texto-v5/test.parquet")


print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Text features: {train_text.shape[1]}, Image features: {train_img.shape[1]}")


# === DATOS EXTERNOS DE MALASIA ===
# PoblaciÃ³n por estado de Malasia (2019)
state_population = {
    41336: 6500000,   # Selangor
    41325: 1800000,   # Kuala Lumpur
    41327: 1700000,   # Johor
    41401: 2500000,   # Sabah
    41326: 2100000,   # Perak
    41330: 1100000,   # Penang
    41324: 1000000,   # Kedah
    41332: 800000,    # Pahang
    41335: 1200000,   # Sarawak
    41342: 500000,    # Negeri Sembilan
    41334: 900000,    # Kelantan
    41329: 700000,    # Melaka
    41328: 1200000,   # Terengganu
    41331: 250000,    # Perlis
    41333: 100000,    # Labuan
}

train['State_population'] = train['State'].map(state_population).fillna(500000)
test['State_population'] = test['State'].map(state_population).fillna(500000)
print("âœ“ State population aÃ±adido")


# === BREED FEATURES ===
def clean_breed(df):
    df = df.copy()
    # Si Breed1 == Breed2, poner Breed2 = 0
    df.loc[df['Breed1'] == df['Breed2'], 'Breed2'] = 0
    # Pure breed flag
    df['pure_breed'] = (df['Breed2'] == 0).astype(int)
    return df

train = clean_breed(train)
test = clean_breed(test)
print("âœ“ Breed features aÃ±adidas")


# === RESCUERID AGGREGATIONS  ===
def add_rescuer_features(train_df, test_df):
    # Reset index para tener PetID como columna
    train_temp = train_df.reset_index()
    test_temp = test_df.reset_index()
    
    # Combinar para calcular estadÃ­sticas
    all_data = pd.concat([train_temp, test_temp], axis=0)
    
    # Agregaciones por RescuerID
    rescuer_agg = all_data.groupby('RescuerID').agg({
        'PetID': 'count',
        'Age': ['mean', 'std', 'min', 'max'],
        'Fee': ['mean', 'std', 'sum'],
        'Quantity': ['sum', 'mean'],
        'PhotoAmt': ['mean', 'sum'],
        'VideoAmt': 'sum',
        'Type': 'nunique',
        'Breed1': 'nunique',
    })
    
    # Flatten column names
    rescuer_agg.columns = ['res_' + '_'.join([str(c) for c in col]).strip() for col in rescuer_agg.columns]
    rescuer_agg = rescuer_agg.reset_index()
    
    # Merge con train y test
    train_out = train_temp.merge(rescuer_agg, on='RescuerID', how='left').set_index('PetID')
    test_out = test_temp.merge(rescuer_agg, on='RescuerID', how='left').set_index('PetID')
    
    return train_out, test_out

train, test = add_rescuer_features(train, test)
print(f"âœ“ RescuerID aggregations aÃ±adidas ({len([c for c in train.columns if c.startswith('res_')])} features)")


# === OTRAS FEATURES ÃšTILES ===
def add_basic_features(df):
    df = df.copy()
    df['Age_in_years'] = df['Age'] / 12
    df['Fee_is_free'] = (df['Fee'] == 0).astype(int)
    df['Has_name'] = (df['Name'].notna() & (df['Name'] != '')).astype(int)
    df['Health_score'] = df['Vaccinated'] + df['Dewormed'] + df['Sterilized']
    df['Total_colors'] = (df['Color1'] != 0).astype(int) + (df['Color2'] != 0).astype(int) + (df['Color3'] != 0).astype(int)
    
    # Edad relativa al promedio del RescuerID (tÃ©cnica del 27th place)
    df['Age_vs_rescuer_mean'] = df['Age'] - df['res_Age_mean']
    
    return df

train = add_basic_features(train)
test = add_basic_features(test)
print("âœ“ Features bÃ¡sicas aÃ±adidas")


# Convertir categÃ³ricas
CAT_COLS = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 
            'FurLength', 'MaturitySize', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']

for col in CAT_COLS:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


# Unir features
train_full = train.join(train_text).join(train_img)
test_full = test.join(test_text).join(test_img)

# Eliminar columnas no necesarias
cols_to_drop = ['Name', 'Description', 'RescuerID', 'PhotoAmt', 'VideoAmt']
train_full = train_full.drop(columns=[c for c in cols_to_drop if c in train_full.columns])
test_full = test_full.drop(columns=[c for c in cols_to_drop if c in test_full.columns])

# Separar X e y
y_train = train_full[TARGET]
X_train = train_full.drop(columns=[TARGET]).select_dtypes(exclude=['object'])
X_test = test_full.select_dtypes(exclude=['object'])

# Asegurar mismas columnas
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# ============================================
# OPTIMIZACIÃ“N LIGERA CON OPTUNA 
# ============================================

### Agregando una semilla
from optuna.samplers import TPESampler

def objective(trial):
    params = {
        'objective': 'regression',
        'verbosity': -1,
        'random_state': SEED,
        'n_estimators': 500,
        
        # Solo 4 parÃ¡metros a optimizar
        'learning_rate': trial.suggest_categorical('learning_rate', [0.03, 0.05, 0.08]),
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'num_leaves': trial.suggest_int('num_leaves', 31, 80),
        'min_child_samples': trial.suggest_categorical('min_child_samples', [30, 50, 80]),
    }
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        preds = model.predict(X_val)
        preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)
        scores.append(quadratic_weighted_kappa(y_val, preds_rounded))
    
    return np.mean(scores)

print("ğŸ”� Optimizando hiperparÃ¡metros...")

### Definimos el sampler con la semilla fija
sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction='maximize', sampler=sampler)
#study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15, show_progress_bar=True)

print(f"\nâœ… Mejor score: {study.best_value:.4f}")
print(f"   Mejores params: {study.best_params}")



# LightGBM - con parÃ¡metros de Optuna
lgb_reg_params = {
    # ParÃ¡metros fijos 
    'objective': 'regression',
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbosity': -1,
    'random_state': SEED,
    
    # ParÃ¡metros optimizados por Optuna 
    **study.best_params
}

# CatBoost - defaults buenos
cat_params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 7,
    'l2_leaf_reg': 10,
    'random_seed': SEED,
    'verbose': False,
}

print("âœ“ ParÃ¡metros definidos")
print(f"  LightGBM: {lgb_reg_params}")


# Preparar datos para CatBoost
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

for col in X_train_cat.select_dtypes(include=['category']).columns:
    X_train_cat[col] = X_train_cat[col].cat.codes
    X_test_cat[col] = X_test_cat[col].cat.codes


# Entrenar con K-Fold (RegresiÃ³n)
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print(f"\nFold {fold+1}/{N_FOLDS}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    X_tr_cat, X_val_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBM Regressor
    lgb_model = LGBMRegressor(**lgb_reg_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / N_FOLDS
    
    # CatBoost Regressor
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), early_stopping_rounds=50)
    oof_cat[val_idx] = cat_model.predict(X_val_cat)
    test_cat += cat_model.predict(X_test_cat) / N_FOLDS
    
    print(f"  LGB val: {np.sqrt(((oof_lgb[val_idx] - y_val)**2).mean()):.4f}")
    print(f"  CAT val: {np.sqrt(((oof_cat[val_idx] - y_val)**2).mean()):.4f}")


# Blend simple (0.5 cada uno)
oof_blend = 0.5 * oof_lgb + 0.5 * oof_cat
test_blend = 0.5 * test_lgb + 0.5 * test_cat

print(f"OOF blend RMSE: {np.sqrt(((oof_blend - y_train)**2).mean()):.4f}")


# OptimizaciÃ³n de umbrales 
class OptimizedRounder:
    def __init__(self):
        self.coef_ = None
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3, 4])
        return -quadratic_weighted_kappa(y, preds)
    
    def fit(self, X, y):
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = minimize(self._kappa_loss, initial_coef, args=(X, y), method='nelder-mead').x
        return self
    
    def predict(self, X):
        return pd.cut(X, [-np.inf] + list(np.sort(self.coef_)) + [np.inf], labels=[0, 1, 2, 3, 4]).astype(int)

rounder = OptimizedRounder()
rounder.fit(oof_blend, y_train)

print(f"Umbrales optimizados: {np.sort(rounder.coef_)}")

oof_preds = rounder.predict(oof_blend)
cv_score = quadratic_weighted_kappa(y_train, oof_preds)
print(f"\nğŸ�¯ CV Score (QWK): {cv_score:.5f}")


# Predicciones finales
test_preds = rounder.predict(test_blend)

submission = pd.DataFrame({
    'PetID': X_test.index,
    'AdoptionSpeed': test_preds
}).set_index('PetID')

print("DistribuciÃ³n de predicciones:")
print(submission['AdoptionSpeed'].value_counts().sort_index())


submission.to_csv('submission.csv')

print("\n" + "="*50)
print("âœ… RESUMEN")
print("="*50)
print(f"  â€¢ Modelos: LightGBM + CatBoost (regresiÃ³n)")
print(f"  â€¢ CV Score (QWK): {cv_score:.5f}")
print("="*50)


!head submission.csv




