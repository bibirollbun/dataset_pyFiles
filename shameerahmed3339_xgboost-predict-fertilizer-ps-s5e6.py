import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
import warnings
warnings.simplefilter('ignore')  # Suprimir advertencias para una salida más limpia


def mapk(y_true, y_pred, k=3):
    """
    
    """
    actual = [[label] for label in y_true]  # Convertir a lista de listas
    
    apk_values = []
    for a, p in zip(actual, y_pred):
        # Limitar predicciones a k
        p = p[:k]
        
        # Calcular precision para cada posición donde hay un acierto
        score = 0.0
        num_hits = 0.0
        
        for i, pred in enumerate(p):
            if pred in a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        # Si no hay etiquetas relevantes, el AP es 0
        if not a:
            apk_values.append(0.0)
        else:
            apk_values.append(score / min(len(a), k))
    
    return np.mean(apk_values)


def feature_engineering(df):
    # Crear copia para evitar warnings
    df = df.copy()
    
    # Eliminar columnas no relevantes
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    
    # Convertir columnas categóricas a tipo 'category'
    for col in ['Soil Type', 'Crop Type']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Características adicionales que podrían ser útiles
    
    # Relación N-P-K
    if all(col in df.columns for col in ['Nitrogen', 'Phosphorous', 'Potassium']):
        # Suma total de nutrientes
        df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
        
        # Ratios entre nutrientes (evitando divisiones por cero)
        df['N_to_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
        df['N_to_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
        df['P_to_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
        
        # Proporción de cada nutriente respecto al total
        df['N_proportion'] = df['Nitrogen'] / (df['Total_NPK'] + 1e-5)
        df['P_proportion'] = df['Phosphorous'] / (df['Total_NPK'] + 1e-5)
        df['K_proportion'] = df['Potassium'] / (df['Total_NPK'] + 1e-5)
    
    # Interacciones entre variables climáticas y del suelo
    if all(col in df.columns for col in ['Temparature', 'Humidity', 'Moisture']):
        # Índice de estrés hídrico (simplificado)
        df['Water_Stress_Index'] = df['Temparature'] / (df['Humidity'] + df['Moisture'] + 1e-5)
        
        # Interacciones
        df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
        df['Temp_Moisture'] = df['Temparature'] * df['Moisture']
    
    return df


# Cargar datos de entrenamiento y prueba
print("Cargando datos...")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# Guardar IDs de test para la presentación final
test_ids = df_test['id'].values

# Aplicar ingeniería de características
print("Aplicando ingeniería de características...")
X_train = feature_engineering(df_train)
X_test = feature_engineering(df_test)

# Separar la variable objetivo
y_train = X_train.pop('Fertilizer Name')

# Codificar las etiquetas
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Obtener la lista de clases
fertilizer_classes = label_encoder.classes_
num_classes = len(fertilizer_classes)

print(f"Número de clases de fertilizantes: {num_classes}")
print(f"Clases de fertilizantes: {fertilizer_classes}")


# Configuración de validación cruzada estratificada para mantener distribución de clases
seed = 42  # Semilla para reproducibilidad
n_folds = 5  # Número de particiones para validación cruzada
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# Inicializar arrays para almacenar predicciones
oof_predictions = np.zeros((len(X_train), num_classes))  # Out-of-fold para validación
test_predictions = np.zeros((len(X_test), num_classes))  # Predicciones de test (promediadas)

# Listas para almacenar métricas de cada fold
fold_scores = []


print(f"\nIniciando entrenamiento con {n_folds} folds...")
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train_encoded)):
    print(f"\n=== Fold {fold+1}/{n_folds} ===")
    
    # Dividir datos
    X_tr = X_train.iloc[train_idx]
    y_tr = y_train_encoded[train_idx]
    X_val = X_train.iloc[valid_idx]
    y_val = y_train_encoded[valid_idx]
    
    print(f"Tamaño conjunto de entrenamiento: {X_tr.shape[0]} filas")
    print(f"Tamaño conjunto de validación: {X_val.shape[0]} filas")
    
    # Convertir a DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    
    # Parámetros para clasificación multiclase con XGBoost
    params = {
        'objective': 'multi:softprob',  # Probabilidades para múltiples clases
        'eval_metric': 'mlogloss',      # Log loss para clasificación multiclase
        'num_class': num_classes,       # Número de clases
        'max_depth': 6,                # Profundidad máxima de los árboles
        'eta': 0.03,                   # Learning rate (tasa de aprendizaje)
        'subsample': 0.8,               # Fracción de muestras usadas en cada árbol
        'colsample_bytree': 0.7,        # Fracción de características usadas en cada árbol
        'min_child_weight': 2,          # Peso mínimo necesario en un nodo hijo
        'gamma': 0.1,                   # Regularización de complejidad mínima por nodo
        'seed': seed,                   # Semilla para reproducibilidad
        'tree_method': 'hist',          # Método de construcción de árboles (histograma, más rápido)
        'device': 'cuda'
        #'nthread': 4                    # Número de hilos para paralelización
    }
    
    # Entrenamiento del modelo XGBoost
    print("Entrenando modelo...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=20000,                          # Número máximo de rondas (árboles)
        evals=[(dtrain, 'train'), (dvalid, 'valid')],   # Conjuntos de evaluación
        early_stopping_rounds=50,                      # Detener si no hay mejora en 50 rondas
        verbose_eval=100                                # Mostrar métricas cada 100 iteraciones
    )
    
    # Predicciones de validación
    val_preds = model.predict(dvalid)
    oof_predictions[valid_idx] = val_preds
    
    # Predicciones de test
    test_preds = model.predict(dtest)
    test_predictions += test_preds / n_folds
    
    # Calcular MAP@3 para este fold
    val_pred_labels = []
    for pred_probs in val_preds:
        # Obtener los índices de las 3 clases con mayor probabilidad
        top_3_indices = np.argsort(pred_probs)[::-1][:3]
        # Convertir índices a nombres de fertilizantes
        val_pred_labels.append([fertilizer_classes[idx] for idx in top_3_indices])
    
    # Calcular MAP@3
    map3_score = mapk(y_train.iloc[valid_idx].values, val_pred_labels, k=3)
    fold_scores.append(map3_score)
    
    print(f"MAP@3 en fold {fold+1}: {map3_score:.6f}")
    
    # Feature importance - solo en el último fold
    if fold == n_folds - 1:  # Verifica si es el último fold
        fig, ax = plt.subplots(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax)
        plt.title(f'Feature Importance - Fold {fold+1}')
        plt.tight_layout()
        plt.show()
        plt.close()





# Calcular MAP@3 global usando predicciones out-of-fold (OOF)
print("\n=== Resultados finales ===")

# Convertir predicciones OOF a listas de top-3 fertilizantes
oof_pred_labels = []
for pred_probs in oof_predictions:
    top_3_indices = np.argsort(pred_probs)[::-1][:3]
    oof_pred_labels.append([fertilizer_classes[idx] for idx in top_3_indices])

# Calcular MAP@3 global
map3_global = mapk(y_train.values, oof_pred_labels, k=3)
print(f"MAP@3 global (OOF): {map3_global:.6f}")
print(f"MAP@3 promedio por fold: {np.mean(fold_scores):.6f}")
print(f"Desviación estándar MAP@3: {np.std(fold_scores):.6f}")

# Gráfico de MAP@3 por fold
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_folds+1), fold_scores, color='skyblue')
plt.axhline(y=map3_global, color='r', linestyle='--', label=f'MAP@3 Global: {map3_global:.6f}')
plt.title('MAP@3 por Fold')
plt.xlabel('Número de Fold')
plt.ylabel('MAP@3')
plt.xticks(range(1, n_folds+1))
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
plt.close()


# Preparar archivo de envío con formato requerido por la competencia
print("\nPreparando archivo de envío...")
submission = pd.DataFrame({'id': test_ids})

# Convertir predicciones de test a top-3 fertilizantes
test_pred_labels = []
for pred_probs in test_predictions:
    top_3_indices = np.argsort(pred_probs)[::-1][:3]
    test_pred_labels.append(' '.join([fertilizer_classes[idx] for idx in top_3_indices]))

submission['Fertilizer Name'] = test_pred_labels

# Guardar archivo de envío en formato CSV (requerido por Kaggle)
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Archivo de envío guardado en: {submission_path}")

# ==============================
# Fin del proceso
# ==============================
print("\n¡Proceso completado con éxito!")

