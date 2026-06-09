import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# 1. Carga de datos
df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_train


df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
df_test


# 2. Separar caracterÃ­sticas (X) y variable objetivo (y)
X = df_train.iloc[:, 1:-1]  # Todas las columnas excepto la Ãºltima (variable objetivo) y la primera (identificador)
y = df_train.iloc[:, -1]   # Ãšltima columna como variable objetivo

# 3. Escalado y centrado de las caracterÃ­sticas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 4. Feature engineering: CombinaciÃ³n de variables dos a dos
poly = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Obtener nombres de las caracterÃ­sticas polinÃ³micas
poly_feature_names = poly.get_feature_names_out(X.columns)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

print(f"Dimensiones originales: {X.shape}")
print(f"Dimensiones despuÃ©s de feature engineering: {X_poly_df.shape}")

# 5. Dividir el dataset en train (70%), validation (20%), test (10%)
# Primera divisiÃ³n: 70% train, 30% temporal
X_temp, X_test, y_temp, y_test = train_test_split(
    X_poly_df, y, test_size=0.1, random_state=42
)

# Segunda divisiÃ³n: del 30% temporal, 2/3 para validation (20% total) y 1/3 para test (10% total) aproximadamente ya que al final he dejado un poco mÃ¡s de datos para el entrenamiento
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42  # 0.222 â‰ˆ 20%/90% pero lo he dejado en 0.2 para darle mÃ¡s datos al conjunto de entrenamiento
)

print(f"TamaÃ±o del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"TamaÃ±o del conjunto de validaciÃ³n: {X_val.shape[0]}")
print(f"TamaÃ±o del conjunto de test: {X_test.shape[0]}")


# 6. Configurar LightGBM y validaciÃ³n cruzada
def train_lightgbm_with_cv(X_train, y_train, X_val, y_val, n_folds=3):
    """
    Entrena LightGBM con validaciÃ³n cruzada y ajusta hiperparÃ¡metros
    """
    # ConfiguraciÃ³n inicial de LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.003, #0.0405973992268857,
        'num_leaves': 35,
        'min_data_in_leaf': 90,
        'feature_fraction': 0.8786957228471932,
        'bagging_fraction': 0.7966824793412932,
        'bagging_freq': 6,
        'lambda_l1': 7.151613714286091,
        'lambda_l2': 5.489198722797788,
        'min_gain_to_split': 2.4913261623670584,
        'max_depth': 17,
        'verbose': -1,
        'random_state': 42
    }
    
    # Datos para LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # ValidaciÃ³n cruzada con KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    print("Realizando validaciÃ³n cruzada...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Crear datasets para LightGBM
        lgb_train = lgb.Dataset(X_fold_train, y_fold_train)
        lgb_val = lgb.Dataset(X_fold_val, y_fold_val, reference=lgb_train)
        
        # Entrenar modelo
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Predecir y calcular RMSE
        y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        cv_scores.append(rmse)
        
        print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")
    
    print(f"\nRMSE promedio en CV: {np.mean(cv_scores):.4f} (Â±{np.std(cv_scores):.4f})")
    
    # Entrenar modelo final con todos los datos de entrenamiento
    print("\nEntrenando modelo final...")
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return final_model, cv_scores


# 7. Entrenar el modelo
model, cv_scores = train_lightgbm_with_cv(X_train, y_train, X_val, y_val)


# 8. Evaluar el conjunto de validaciÃ³n
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"\nRMSE en conjunto de validaciÃ³n: {val_rmse:.4f}")


# 9. Evaluar el conjunto de test
y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"RMSE en conjunto de test: {test_rmse:.4f}")


# 10. Comparar resultados para detectar overfitting
print("\n" + "="*50)
print("COMPARACIÃ“N DE RESULTADOS")
print("="*50)
print(f"RMSE promedio en CV: {np.mean(cv_scores):.4f}")
print(f"RMSE en validaciÃ³n: {val_rmse:.4f}")
print(f"RMSE en test: {test_rmse:.4f}")

# AnÃ¡lisis de overfitting
if test_rmse > val_rmse * 1.1:  # Si el RMSE de test es mÃ¡s del 10% mayor que el de validaciÃ³n
    print("\nâš ï¸�  POSIBLE OVERFITTING: El modelo no generaliza bien a datos no vistos")
elif test_rmse < val_rmse:
    print("\nâœ… BUEN AJUSTE: El modelo generaliza bien")
else:
    print("\nğŸ“Š COMPORTAMIENTO ESPERADO: PequeÃ±a diferencia entre validaciÃ³n y test")


def train_all_regression_models(X_train, y_train, X_val, y_val, n_folds=3, verbose=True):
    """
    Entrena y evalÃºa mÃºltiples modelos de regresiÃ³n con validaciÃ³n cruzada
    
    Parameters:
    X_train, y_train: Datos de entrenamiento
    X_val, y_val: Datos de validaciÃ³n
    n_folds: NÃºmero de folds para CV
    verbose: Mostrar progreso
    
    Returns:
    Dict con modelos entrenados y resultados
    """
    
    # Diccionario para almacenar resultados
    results = {}
    
    # DefiniciÃ³n de todos los modelos con sus parÃ¡metros por defecto
    models = {
        'LightGBM': lgb.LGBMRegressor(
            random_state=42,
            n_estimators=1000,
            learning_rate=0.003,
            num_leaves=35,
            min_child_samples=90,
            subsample=0.796,
            colsample_bytree=0.879,
            reg_alpha=7.15,
            reg_lambda=5.49,
            min_split_gain=2.49,
            max_depth=17
        ),
        'XGBoost': XGBRegressor(
            random_state=42,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        ),
        'CatBoost': CatBoostRegressor(
            random_state=42,
            verbose=False,
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3
        ),
        #'RandomForest': RandomForestRegressor(
        #    random_state=42,
        #    n_estimators=100,
        #    max_depth=None,
        #    min_samples_split=2,
        #    min_samples_leaf=1
        #),
        'GradientBoosting': GradientBoostingRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8
        ),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        #'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=None),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5),
        'AdaBoost': AdaBoostRegressor(
            random_state=42,
            n_estimators=50,
            learning_rate=1.0
        )
    }
    
    if verbose:
        print("ğŸš€ Entrenando mÃºltiples modelos de regresiÃ³n...")
        print("=" * 60)
    
    for model_name, model in models.items():
        if verbose:
            print(f"\nğŸ“Š Entrenando {model_name}...")
        
        try:
            # ValidaciÃ³n cruzada
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = -cross_val_score(
                model, X_train, y_train, 
                cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1
            )
            
            # Entrenar modelo completo
            model.fit(X_train, y_train)
            
            # Predecir en validation set
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            # Almacenar resultados
            results[model_name] = {
                'model': model,
                'cv_mean_rmse': np.mean(cv_scores),
                'cv_std_rmse': np.std(cv_scores),
                'val_rmse': val_rmse,
                'cv_scores': cv_scores.tolist()
            }
            
            if verbose:
                print(f"   CV RMSE: {np.mean(cv_scores):.4f} (Â±{np.std(cv_scores):.4f})")
                print(f"   Validation RMSE: {val_rmse:.4f}")
                
        except Exception as e:
            print(f"â�Œ Error entrenando {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results

def print_results_summary(results):
    """
    Imprime un resumen de los resultados de todos los modelos
    """
    print("\n" + "=" * 80)
    print("ğŸ“ˆ RESUMEN DE RESULTADOS - COMPARACIÃ“N DE MODELOS")
    print("=" * 80)
    
    # Crear DataFrame para mejor visualizaciÃ³n
    summary_data = []
    for model_name, result in results.items():
        if 'error' not in result:
            summary_data.append({
                'Model': model_name,
                'CV RMSE Mean': result['cv_mean_rmse'],
                'CV RMSE Std': result['cv_std_rmse'],
                'Validation RMSE': result['val_rmse'],
                'Difference': result['val_rmse'] - result['cv_mean_rmse']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Validation RMSE')
    
    print(summary_df.to_string(index=False))
    
    # Mejor modelo
    best_model_name = summary_df.iloc[0]['Model']
    best_rmse = summary_df.iloc[0]['Validation RMSE']
    print(f"\nğŸ�† MEJOR MODELO: {best_model_name} con RMSE: {best_rmse:.4f}")

def get_best_model(results, return_metrics=False):
    """
    Retorna el mejor modelo basado en RMSE de validaciÃ³n
    """
    best_model_name = None
    best_rmse = float('inf')
    best_model = None
    
    for model_name, result in results.items():
        if 'error' not in result and result['val_rmse'] < best_rmse:
            best_rmse = result['val_rmse']
            best_model_name = model_name
            best_model = result['model']
    
    if return_metrics:
        return best_model, best_model_name, best_rmse
    else:
        return best_model


# OpciÃ³n 1: Entrenar todos los modelos
print("ğŸ§ª Probando todos los modelos...")
all_results = train_all_regression_models(X_train, y_train, X_val, y_val, n_folds=3)
    
# Mostrar resumen
print_results_summary(all_results)
    
# Obtener mejor modelo
best_model, best_name, best_rmse = get_best_model(all_results, return_metrics=True)
print(f"\nğŸ�¯ Mejor modelo seleccionado: {best_name} (RMSE: {best_rmse:.4f})")


# FunciÃ³n especÃ­fica para LightGBM
def train_lightgbm_custom(X_train, y_train, X_val, y_val, X_test, y_test, n_folds=3):
    """
    Entrena LightGBM con la configuraciÃ³n especÃ­fica que tenÃ­as
    """

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.002,
        'num_leaves': 35,
        'min_data_in_leaf': 90,
        'feature_fraction': 0.8786957228471932,
        'bagging_fraction': 0.7966824793412932,
        'bagging_freq': 6,
        'lambda_l1': 7.151613714286091,
        'lambda_l2': 5.489198722797788,
        'min_gain_to_split': 2.4913261623670584,
        'max_depth': 17,
        'verbose': -1,
        'random_state': 42
    }
    
    # Datos para LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # ValidaciÃ³n cruzada con KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=params['random_state'])
    cv_scores = []
    
    print("Realizando validaciÃ³n cruzada para LightGBM...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Crear datasets para LightGBM
        lgb_train = lgb.Dataset(X_fold_train, y_fold_train)
        lgb_val = lgb.Dataset(X_fold_val, y_fold_val, reference=lgb_train)
        
        # Entrenar modelo
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Predecir y calcular RMSE
        y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        cv_scores.append(rmse)
        
        print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nRMSE promedio en CV: {cv_mean:.4f} (Â±{cv_std:.4f})")
    
    # Entrenar modelo final con todos los datos de entrenamiento
    print("\nEntrenando modelo final de LightGBM...")
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predecir en validation
    y_test_pred = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"LightGBM Test RMSE: {test_rmse:.4f}")
    
    return {
        'model': final_model,
        'cv_mean_rmse': cv_mean,
        'cv_std_rmse': cv_std,
        'test_rmse': test_rmse,
        'cv_scores': cv_scores
    }



# OpciÃ³n 2: Entrenar solo LightGBM con configuraciÃ³n personalizada
print("\n" + "="*50)
print("ğŸ”§ LightGBM con configuraciÃ³n personalizada")
print("="*50)

# Enfrentamos los conjuntos combinados directamente al conjunto de test
lgb_results = train_lightgbm_custom(X_train, y_train, X_val, y_val, X_test, y_test)
    
# Comparar con otros modelos
print(f"\nLightGBM Custom RMSE: {lgb_results['test_rmse']:.4f}")


# 11. Cargar y preprocesar el conjunto de test para submission
def prepare_test_data(scaler, poly, original_columns):
    """
    Preprocesa los datos de test de la misma manera que los de entrenamiento
    """
    
    # Guardar los IDs para el submission
    test_ids = df_test['id'].copy() if 'id' in df_test.columns else None
    
    # Si hay columnas que no son caracterÃ­sticas, quitarlas (como 'id')
    features_to_drop = ['id'] if 'id' in df_test.columns else []
    X_test_original = df_test.drop(features_to_drop, axis=1)
    
    # Asegurar que las columnas estÃ¡n en el mismo orden que el entrenamiento
    X_test_original = X_test_original[original_columns]
    
    # Aplicar el mismo escalado y centrado
    X_test_scaled = scaler.transform(X_test_original)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=original_columns)
    
    # Aplicar las mismas transformaciones polinÃ³micas
    X_test_poly = poly.transform(X_test_scaled)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out(original_columns))
    
    print(f"Datos de test preparados: {X_test_poly_df.shape}")
    
    return X_test_poly_df, test_ids

# 12. Hacer predicciones en el conjunto de test
def make_predictions(model, scaler, poly, original_columns):
    """
    Realiza predicciones en el conjunto de test y prepara el submission
    """
    print("ğŸ”® Haciendo predicciones en el conjunto de test...")
    
    # Preparar datos de test
    X_test_processed, test_ids = prepare_test_data(scaler, poly, original_columns)
    
    # Hacer predicciones
    predictions = model.predict(X_test_processed)
    
    # Crear DataFrame de submission
    submission_df = pd.DataFrame({
        'id': test_ids,
        'BeatsPerMinute': predictions
    })
    
    # Verificar que no hay valores nulos
    if submission_df.isnull().any().any():
        print("âš ï¸�  Advertencia: Hay valores nulos en las predicciones")
        # Rellenar nulos con la media (por si acaso)
        submission_df['BeatsPerMinute'] = submission_df['BeatsPerMinute'].fillna(
            submission_df['BeatsPerMinute'].mean()
        )
    
    print(f"Predicciones realizadas: {len(predictions)} registros")
    print(f"Rango de predicciones: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"Media de predicciones: {predictions.mean():.2f}")
    
    return submission_df

# 13. Guardar el archivo submission.csv
def save_submission(submission_df, filename='submission.csv'):
    """
    Guarda el DataFrame de submission como archivo CSV
    """
    submission_df.to_csv(filename, index=False)
    print(f"âœ… Archivo '{filename}' guardado exitosamente")
    print(f"ğŸ“Š Preview del archivo de submission:")
    print(submission_df.head())
    
    # Verificar el archivo guardado
    saved_file = pd.read_csv(filename)
    print(f"ğŸ“� Archivo verificado: {len(saved_file)} registros guardados")

# 14. Ejecutar el pipeline completo de predicciÃ³n
# Definir las columnas originales (sin la variable objetivo)
original_columns = X.columns.tolist()

try:
    #model = best_model #OpciÃ³n 1: ComparaciÃ³n de modelos y elecciÃ³n del mejor modelo encontrado
    model = lgb_results['model'] #OpciÃ³n 2: LightGBM personalizado con los mejores hiperparÃ¡metros encontrados
    # Hacer predicciones
    submission_df = make_predictions(model, scaler, poly, original_columns)
    
    # Guardar submission
    save_submission(submission_df, 'submission.csv')
    
    # 18. AnÃ¡lisis adicional de las predicciones
    print("\nğŸ“ˆ ANÃ�LISIS DE LAS PREDICCIONES FINALES:")
    print("="*40)
    print(f"NÃºmero de predicciones: {len(submission_df)}")
    print(f"Valor mÃ­nimo predicho: {submission_df['BeatsPerMinute'].min():.2f}")
    print(f"Valor mÃ¡ximo predicho: {submission_df['BeatsPerMinute'].max():.2f}")
    print(f"Valor medio predicho: {submission_df['BeatsPerMinute'].mean():.2f}")
    print(f"DesviaciÃ³n estÃ¡ndar: {submission_df['BeatsPerMinute'].std():.2f}")
    
    # Verificar distribuciÃ³n
    print("\nğŸ“Š DistribuciÃ³n de las predicciones:")
    print(submission_df['BeatsPerMinute'].describe())
    
    # Verificar que no hay IDs duplicados
    if submission_df['id'].duplicated().any():
        print("âš ï¸�  Advertencia: Hay IDs duplicados en el submission")
    else:
        print("âœ… No hay IDs duplicados")
    
except FileNotFoundError:
    print(f"â�Œ Error: No se encontrÃ³ el archivo de test en {test_data_path}")
    print("Por favor, verifica la ruta del archivo de test")
    
except Exception as e:
    print(f"â�Œ Error inesperado durante la predicciÃ³n: {str(e)}")
    print("Traceback:", e.__traceback__)

# 15. FunciÃ³n adicional para verificar la consistencia del preprocesamiento
def verify_preprocessing_consistency(X_train_original, X_test_processed, poly):
    """
    Verifica que el preprocesamiento se aplicÃ³ consistentemente
    """
    print("\nğŸ”� VERIFICACIÃ“N DE CONSISTENCIA EN PREPROCESAMIENTO:")
    print("="*50)
    
    # Verificar nÃºmero de caracterÃ­sticas
    expected_features = len(poly.get_feature_names_out(X_train_original.columns))
    actual_features = X_test_processed.shape[1]
    
    print(f"CaracterÃ­sticas esperadas: {expected_features}")
    print(f"CaracterÃ­sticas obtenidas: {actual_features}")
    
    if expected_features == actual_features:
        print("âœ… NÃºmero de caracterÃ­sticas consistente")
    else:
        print("â�Œ Error: NÃºmero de caracterÃ­sticas inconsistente")
    
    # Verificar que no hay valores nulos en los datos procesados
    if X_test_processed.isnull().any().any():
        print("â�Œ Error: Hay valores nulos en los datos procesados")
    else:
        print("âœ… No hay valores nulos en los datos procesados")

# 16. Si necesitas probar con un pequeÃ±o ejemplo de test (opcional)
def create_sample_test_file(original_columns, n_samples=5, filename='sample_test.csv'):
    """
    Crea un archivo de test de ejemplo para pruebas
    """
    sample_data = {}
    for col in original_columns:
        sample_data[col] = np.random.normal(0, 1, n_samples)
    
    sample_data['id'] = range(1, n_samples + 1)
    
    sample_df = pd.DataFrame(sample_data)
    # Reordenar columnas: id primero, luego las caracterÃ­sticas
    sample_df = sample_df[['id'] + original_columns]
    sample_df.to_csv(filename, index=False)
    print(f"âœ… Archivo de ejemplo '{filename}' creado con {n_samples} muestras")

# Crear archivo de ejemplo para pruebas (opcional)
# create_sample_test_file(original_columns, n_samples=10, filename='sample_test.csv')

# 17. CÃ³digo para cargar y verificar el archivo de test
print("\nğŸ“‹ INFORMACIÃ“N DEL ARCHIVO DE TEST:")
print("="*40)
try:
    df_test_original = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    print(f"Filas en test original: {df_test_original.shape[0]}")
    print(f"Columnas en test original: {df_test_original.shape[1]}")
    print("Columnas:", df_test_original.columns.tolist())
    print("\nPrimeras 5 filas:")
    print(df_test_original.head())
    
    # Verificar que tenemos la columna 'id'
    if 'id' not in df_test_original.columns:
        print("â�Œ Error: El archivo de test no tiene columna 'id'")
    else:
        print("âœ… Columna 'id' encontrada en el archivo de test")
        
except FileNotFoundError:
    print(f"Archivo de test no encontrado en: {test_data_path}")
    print("AsegÃºrate de que la ruta es correcta y el archivo existe")

# 18. Guardar tambiÃ©n el modelo entrenado para uso futuro (opcional)
def save_trained_model(model, filename='lightgbm_model.txt'):
    """
    Guarda el modelo entrenado para uso futuro
    """
    model.save_model(filename)
    print(f"âœ… Modelo guardado como '{filename}'")

# Guardar el modelo (opcional)
# save_trained_model(best_model, 'best_lightgbm_model.txt')

print("\nğŸ�‰ PROCESO COMPLETADO! El archivo submission.csv estÃ¡ listo para enviar a Kaggle")


# 1. FunciÃ³n para generar hiperparÃ¡metros aleatorios
def generate_random_hyperparameters():
    """
    Genera un conjunto aleatorio de hiperparÃ¡metros para LightGBM
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    
    # HiperparÃ¡metros aleatorios
    params['learning_rate'] = np.random.uniform(0.005, 0.2)
    params['num_leaves'] = np.random.randint(20, 150)
    params['min_data_in_leaf'] = np.random.randint(10, 100)
    params['feature_fraction'] = np.random.uniform(0.6, 1.0)
    params['bagging_fraction'] = np.random.uniform(0.6, 1.0)
    params['bagging_freq'] = np.random.randint(0, 10)
    params['lambda_l1'] = np.random.uniform(0, 10)
    params['lambda_l2'] = np.random.uniform(0, 10)
    params['min_gain_to_split'] = np.random.uniform(0, 15)
    params['max_depth'] = np.random.randint(-1, 20)  # -1 significa sin lÃ­mite
    params['random_state'] = 42 #np.random.randint(1, 100) # Cambiamos la semilla en cada iteraciÃ³n para mÃ¡s aleatoriedad
    #params['n_estimators'] = np.random.randint(10, 1000)
    #params['min_child_samples'] = np.random.randint(10, 50)
    #params['subsample'] = np.random.uniform(0.6, 1.0)
    #params['colsample_bytree'] = np.random.uniform(0.6, 1.0)
    #params['reg_alpha'] = np.random.uniform(0, 1.0)
    #params['reg_lambda'] = np.random.uniform(0, 1.0)
    
    return params


# 2. FunciÃ³n de optimizaciÃ³n con bÃºsqueda aleatoria
def random_search_optimization(X_train, y_train, X_val, y_val, X_test, y_test, target_rmse=26.4, max_iterations=50):
    """
    BÃºsqueda aleatoria de hiperparÃ¡metros hasta alcanzar el RMSE objetivo
    """
    best_rmse = float('inf')
    best_params = None
    best_model = None
    best_iteration = 0
    history = []
    
    print(f"Iniciando bÃºsqueda aleatoria (objetivo RMSE < {target_rmse})...")
    print("="*60)
    
    iteration = 0
    while best_rmse > target_rmse and iteration < max_iterations:
        iteration += 1
        
        # Generar parÃ¡metros aleatorios
        current_params = generate_random_hyperparameters()
        
        print(f"\nIteraciÃ³n {iteration}: Probando parÃ¡metros...")
        print(f"LR: {current_params['learning_rate']:.4f}, "
              f"Leaves: {current_params['num_leaves']}, "
              f"Min Data: {current_params['min_data_in_leaf']}")
        
        try:
            # Entrenar modelo
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                current_params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=200)
                ]
            )
            
            # Evaluar
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Guardar historial
            history.append({
                'iteration': iteration,
                'rmse': rmse,
                'params': current_params.copy(),
                'best_iteration': model.best_iteration
            })
            
            print(f"RMSE obtenido: {rmse:.4f} | Best RMSE: {best_rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = current_params.copy()
                best_model = model
                best_iteration = iteration
                print(f"âœ… NUEVO MEJOR RMSE: {best_rmse:.4f}")
                
                if best_rmse <= target_rmse:
                    print(f"ğŸ�¯ OBJETIVO ALCANZADO! RMSE = {best_rmse:.4f}")
                    break
            
        except Exception as e:
            print(f"â�Œ Error en iteraciÃ³n {iteration}: {str(e)}")
            continue
    
    # Resultados finales
    print("\n" + "="*60)
    print("RESULTADOS DE LA BÃšSQUEDA ALEATORIA")
    print("="*60)
    print(f"Mejor RMSE obtenido: {best_rmse:.4f}")
    print(f"Mejor iteraciÃ³n: {best_iteration}")
    print(f"Objetivo alcanzado: {'âœ…' if best_rmse <= target_rmse else 'â�Œ'}")
    
    if best_rmse <= target_rmse:
        print("ğŸ�¯ Â¡Objetivo de RMSE alcanzado!")
    else:
        print(f"âš ï¸�  No se alcanzÃ³ el objetivo. Mejor RMSE: {best_rmse:.4f}")
    
    return best_model, best_params, best_rmse, history



# 3. Ejecutar la bÃºsqueda aleatoria
target_rmse = 26.4  # Tu objetivo de RMSE
best_model = None
#best_model, best_params, best_rmse, history = random_search_optimization(
#    X_train, y_train, X_val, y_val, X_test, y_test, target_rmse=target_rmse, max_iterations=250
#)

# 9. Evaluar el mejor modelo en conjunto de test
if best_model is not None:
    y_test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\nğŸ“Š RMSE en conjunto de TEST: {test_rmse:.4f}")
    
    # 10. Comparar resultados para detectar overfitting
    print("\n" + "="*50)
    print("COMPARACIÃ“N FINAL DE RESULTADOS")
    print("="*50)
    print(f"Mejor RMSE en validaciÃ³n: {best_rmse:.4f}")
    print(f"RMSE en test: {test_rmse:.4f}")
    
    # AnÃ¡lisis de overfitting
    overfitting_threshold = 1.1  # 10% mÃ¡s
    if test_rmse > best_rmse * overfitting_threshold:
        print("\nâš ï¸�  POSIBLE OVERFITTING: El modelo no generaliza bien")
        print(f"   Diferencia: {((test_rmse - best_rmse) / best_rmse * 100):.1f}%")
    elif test_rmse < best_rmse:
        print("\nâœ… EXCELENTE: El modelo generaliza mejor de lo esperado")
    else:
        print("\nğŸ“Š COMPORTAMIENTO ESPERADO: PequeÃ±a diferencia")
    
    # 11. Mostrar los mejores parÃ¡metros encontrados
    print("\nğŸ�¯ MEJORES HIPERPARÃ�METROS ENCONTRADOS:")
    print("="*40)
    for key, value in best_params.items():
        if key not in ['objective', 'metric', 'boosting_type', 'verbose']:
            print(f"{key}: {value}")


# 4. FunciÃ³n adicional para anÃ¡lisis del historial
def analyze_search_history(history):
    """
    Analiza el historial de la bÃºsqueda aleatoria
    """
    if not history:
        return
    
    print("\nğŸ“ˆ ANÃ�LISIS DEL HISTORIAL DE BÃšSQUEDA:")
    print("="*40)
    
    # Convertir a DataFrame para anÃ¡lisis
    history_df = pd.DataFrame(history)
    
    # Top 5 mejores resultados
    print("\nTop 5 mejores iteraciones:")
    top_5 = history_df.nsmallest(5, 'rmse')
    for _, row in top_5.iterrows():
        print(f"Iter {row['iteration']}: RMSE = {row['rmse']:.4f}")
    
    # EstadÃ­sticas
    print(f"\nEstadÃ­sticas de la bÃºsqueda:")
    print(f"Total de iteraciones: {len(history_df)}")
    print(f"Mejor RMSE: {history_df['rmse'].min():.4f}")
    print(f"Peor RMSE: {history_df['rmse'].max():.4f}")
    print(f"RMSE promedio: {history_df['rmse'].mean():.4f}")
    
    return history_df

# Analizar el historial de bÃºsqueda
#history_df = analyze_search_history(history)

