import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import warnings
# Ignorar advertencias para una ejecución limpia
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
TRAIN_PATH = '/kaggle/input/petfinder-adoption-prediction/train/train.csv'
TEST_PATH = '/kaggle/input/petfinder-adoption-prediction/test/test.csv'
BREED_LABELS = '/kaggle/input/petfinder-adoption-prediction/breed_labels.csv'
COLOR_LABELS = '/kaggle/input/petfinder-adoption-prediction/color_labels.csv'
STATE_LABELS = '/kaggle/input/petfinder-adoption-prediction/state_labels.csv'

# Paths de features procesadas (asumiendo que se han agregado como input dataset)
TRAIN_SENTIMENT = '/kaggle/input/process-text/train_sentiment_features.csv'
TEST_SENTIMENT = '/kaggle/input/process-text/test_sentiment_features.csv'
TRAIN_TEXT = '/kaggle/input/process-text/train_text_features.csv'
TEST_TEXT = '/kaggle/input/process-text/test_text_features.csv'
TRAIN_IMAGE = '/kaggle/input/process-image/train_image_features.csv'
TEST_IMAGE = '/kaggle/input/process-image/test_image_features.csv'

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def feature_engineering(train, test, breeds, colors, states):
    """Feature engineering completo"""

    # Concatenar para procesamiento conjunto
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)

    # === FEATURES BÁSICAS ===
    # Texto
    df['Name_length'] = df['Name'].fillna('').apply(len)
    df['Description_length'] = df['Description'].fillna('').apply(len)
    df['Name_word_count'] = df['Name'].fillna('').apply(lambda x: len(x.split()))
    df['Description_word_count'] = df['Description'].fillna('').apply(lambda x: len(x.split()))

    # Agregaciones por RescuerID
    for col in ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt']:
        grouped = df.groupby('RescuerID')[col].agg(['mean', 'std', 'count'])
        df[f'RescuerID_{col}_mean'] = df['RescuerID'].map(grouped['mean'])
        df[f'RescuerID_{col}_std'] = df['RescuerID'].map(grouped['std']).fillna(0)
        df[f'RescuerID_{col}_count'] = df['RescuerID'].map(grouped['count'])

    # Breed purity
    df['is_pure_breed'] = (df['Breed2'] == 0).astype(int)
    df['is_mixed_breed'] = (df['Breed2'] != 0).astype(int)

    # Color combinations
    df['has_multiple_colors'] = ((df['Color2'] != 0) | (df['Color3'] != 0)).astype(int)
    df['num_colors'] = (df['Color1'] != 0).astype(int) + (df['Color2'] != 0).astype(int) + (df['Color3'] != 0).astype(int)

    # Health features
    df['health_total'] = (df['Vaccinated'] + df['Dewormed'] + df['Sterilized'])
    df['is_healthy'] = ((df['Health'] == 1) & (df['Vaccinated'] == 1) &
                        (df['Dewormed'] == 1) & (df['Sterilized'] == 1)).astype(int)

    # Fee features
    df['has_fee'] = (df['Fee'] > 0).astype(int)
    df['log_fee'] = np.log1p(df['Fee'])

    # Age categories
    df['age_category'] = pd.cut(df['Age'], bins=[0, 3, 12, 60, 300], labels=['baby', 'young', 'adult', 'senior'])
    df = pd.get_dummies(df, columns=['age_category'], drop_first=True)

    # Interaction features
    df['Age_Quantity_interaction'] = df['Age'] * df['Quantity']
    df['PhotoAmt_VideoAmt_sum'] = df['PhotoAmt'] + df['VideoAmt']
    df['Fee_PhotoAmt_ratio'] = df['Fee'] / (df['PhotoAmt'] + 1)

    # Separar train y test
    train_processed = df[df['is_train'] == 1].drop('is_train', axis=1).reset_index(drop=True)
    test_processed = df[df['is_train'] == 0].drop('is_train', axis=1).reset_index(drop=True)

    return train_processed, test_processed

# ============================================================================
# MODELO Y ENTRENAMIENTO
# ============================================================================
def quadratic_weighted_kappa(y_true, y_pred):
    """Calcula QWK - métrica de la competencia"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

class OptimizedRounder:
    """Optimiza los thresholds para maximizar QWK"""
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return -quadratic_weighted_kappa(y, X_p)

    def fit(self, X, y):
        from scipy.optimize import minimize
        loss_partial = lambda coef: self._kappa_loss(coef, X, y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead').x
        return self

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

# Función para el entrenamiento del modelo LightGBM
def train_lgb_model(X_train, y_train, X_val, y_val, params=None):
    """Entrena un modelo LightGBM"""

    # Parámetros por defecto si no se especifican (no se usan en este script modificado)
    if params is None:
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': -1,
            'min_child_samples': 20, 'subsample': 0.8, 'subsample_freq': 1,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }

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
            lgb.log_evaluation(period=500)
        ]
    )

    return model

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def main():
    """Pipeline principal de ejecución"""

    print("=" * 80)
    print("PART 3: MAIN MODEL TRAINING")
    print("=" * 80)

    # 1. Cargar datos base
    print("\n1. Cargando datos base...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    breeds = pd.read_csv(BREED_LABELS)
    colors = pd.read_csv(COLOR_LABELS)
    states = pd.read_csv(STATE_LABELS)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Target distribution:\n{train['AdoptionSpeed'].value_counts().sort_index()}")

    # 2. Cargar features procesadas
    print("\n2. Cargando features procesadas...")
    try:
        train_sentiment = pd.read_csv(TRAIN_SENTIMENT)
        test_sentiment = pd.read_csv(TEST_SENTIMENT)
        print(f"  ✓ Sentiment features cargadas")
    except:
        print("  ✗ ERROR: No se encontraron sentiment features")
        print("    Asegúrate de ejecutar primero 1_text_processing.ipynb")
        print("    y agregar su output como input dataset")
        return

    try:
        train_text = pd.read_csv(TRAIN_TEXT)
        test_text = pd.read_csv(TEST_TEXT)
        print(f"  ✓ Text features cargadas")
    except:
        print("  ✗ ERROR: No se encontraron text features")
        return

    try:
        train_image = pd.read_csv(TRAIN_IMAGE)
        test_image = pd.read_csv(TEST_IMAGE)
        print(f"  ✓ Image features cargadas")
    except:
        print("  ✗ ERROR: No se encontraron image features")
        print("    Asegúrate de ejecutar primero 2_image_processing.ipynb")
        print("    y agregar su output como input dataset")
        return

    # 3. Feature engineering
    print("\n3. Aplicando feature engineering...")
    train, test = feature_engineering(train, test, breeds, colors, states)

    # 4. Merge con features procesadas
    print("\n4. Combinando todas las features...")
    train = train.merge(train_sentiment, on='PetID', how='left')
    train = train.merge(train_text, on='PetID', how='left')
    train = train.merge(train_image, on='PetID', how='left')

    test = test.merge(test_sentiment, on='PetID', how='left')
    test = test.merge(test_text, on='PetID', how='left')
    test = test.merge(test_image, on='PetID', how='left')

    print(f"Train final shape: {train.shape}")
    print(f"Test final shape: {test.shape}")

    # 5. Preparar datos para el modelo
    print("\n5. Preparando datos para entrenamiento...")
    target = train['AdoptionSpeed']
    train_id = train['PetID']
    test_id = test['PetID']

    # Columnas a excluir
    exclude_cols = ['PetID', 'Name', 'RescuerID', 'Description', 'AdoptionSpeed']
    feature_cols = [col for col in train.columns if col not in exclude_cols]

    X = train[feature_cols]
    X_test = test[feature_cols]
    y = target

    print(f"Features: {len(feature_cols)}")
    print(f"Train shape: {X.shape}")
    print(f"Test shape: {X_test.shape}")

    # 6. Parámetros Fijos (Reemplazo del Random Search)
    print("\n6. Usando parámetros fijos...")

    # Definir los parámetros óptimos encontrados en la búsqueda previa
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'learning_rate': 0.04,
        'num_leaves': 90,
        'max_depth': 10,
        'min_child_samples': 10,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.00,
        'reg_lambda': 0.00, # L2 nula
    }

    print("  Parámetros a usar:")
    for key, value in best_params.items():
        if key not in ['objective', 'metric', 'boosting_type', 'random_state', 'n_jobs', 'verbose', 'subsample_freq']:
            print(f"  {key}: {value}")

    # 7. Cross-validation con StratifiedKFold
    print("\n7. Entrenando modelos con 5-Fold CV...")
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Entrenar modelo con los parámetros fijos
        model = train_lgb_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, params=best_params)

        # Predicciones
        oof_predictions[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
        test_predictions += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

        # Score del fold
        fold_score = quadratic_weighted_kappa(
            y_val_fold,
            np.round(oof_predictions[val_idx]).astype(int)
        )
        scores.append(fold_score)
        print(f"Fold {fold + 1} QWK Score: {fold_score:.4f}")

        gc.collect()

    print(f"\n{'='*80}")
    print(f"Mean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"{'='*80}")

    # 8. Optimizar thresholds
    print("\n8. Optimizando thresholds...")
    rounder = OptimizedRounder()
    rounder.fit(oof_predictions, y)

    optimized_predictions = rounder.predict(oof_predictions, rounder.coef_)
    final_score = quadratic_weighted_kappa(y, optimized_predictions)

    print(f"Optimized thresholds: {rounder.coef_}")
    print(f"Final OOF QWK Score: {final_score:.4f}")

    # 9. Predicciones finales
    print("\n9. Generando predicciones finales...")
    final_test_predictions = rounder.predict(test_predictions, rounder.coef_)

    # 10. Crear submission
    print("\n10. Creando archivo de submission...")
    submission = pd.DataFrame({
        'PetID': test_id,
        'AdoptionSpeed': final_test_predictions.astype(int)
    })

    submission.to_csv('submission.csv', index=False)
    print("\n✓ Submission guardada como 'submission.csv'")
    print(f"\nDistribución de predicciones:")
    print(submission['AdoptionSpeed'].value_counts().sort_index())

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 80)

# ============================================================================
# EJECUTAR
# ============================================================================
if __name__ == '__main__':
    main()

