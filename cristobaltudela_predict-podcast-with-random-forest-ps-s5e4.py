# Importar las bibliotecas necesarias para entrenamiento con RandomForest
import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import kagglehub


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Cargar los datos de entrenamiento y prueba
path = kagglehub.dataset_download("ysthehurricane/podcast-listening-time-prediction-dataset")
print("Path to dataset files:", path)

original_data = pd.read_csv(f'{path}/podcast_dataset.csv')

ruta_base = '/kaggle/input/playground-series-s5e4/'
train_data = pd.read_csv(f'{ruta_base}train.csv')
train_data = train_data.drop(['id'], axis=1)

# Concatenar datos originales con sintéticos y eliminar duplicados
train_data = pd.concat([train_data, original_data], axis=0, ignore_index=True)
train_data = train_data.drop_duplicates()  # Asignar el resultado de vuelta al DataFrame

# Eliminar filas donde 'Listening_Time_minutes' es NaN ANTES de separar
train_data = train_data.dropna(subset=['Listening_Time_minutes'])

test_data = pd.read_csv(f'{ruta_base}test.csv')

# Explorando datos
print(f"Datos originales: {original_data.shape}")
print(f"Datos de entrenamiento: {train_data.shape}")
print(f"Datos de prueba: {test_data.shape}")

# Mostrar información sobre valores nulos (ahora solo en features)
print("\nValores nulos en entrenamiento (solo features):")
print(train_data.drop('Listening_Time_minutes', axis=1).isnull().sum())

# Separar características y variable objetivo (ahora y_train no tiene NaN)
features = train_data.drop(['Listening_Time_minutes'], axis=1)
#test_data = test_data.drop(['Podcast_Name'], axis=1)

y_train = train_data['Listening_Time_minutes'].values

# Identificar columnas numéricas y categóricas
cat_cols = [col for col in features.columns if features[col].dtype == 'object']
num_cols = [col for col in features.columns if col not in cat_cols]

# Imputar la mediana para características numéricas
num_imputer = SimpleImputer(strategy='median')
features[num_cols] = num_imputer.fit_transform(features[num_cols])

# Imputar la moda para características categóricas (por si acaso hay nulos)
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    features[cat_cols] = cat_imputer.fit_transform(features[cat_cols])

# Verificación de que y_train no tiene NaN (opcional)
print(f"\nValores NaN en y_train: {np.isnan(y_train).sum()} (debe ser 0)")

# Preparar los datos de prueba
features_test = test_data.drop(['id'], axis=1)
test_ids = test_data['id']

# Aplicar las mismas transformaciones a los datos de prueba
features_test[num_cols] = num_imputer.transform(features_test[num_cols])
if len(cat_cols) > 0:
    features_test[cat_cols] = cat_imputer.transform(features_test[cat_cols])

# Mostrar información actualizada
print("\nCaracterísticas numéricas:", len(num_cols))
print("Variables numéricas:", ', '.join(num_cols))
print("\nCaracterísticas categóricas:", len(cat_cols))
print("Variables categóricas:", ', '.join(cat_cols))

print("\nMuestra de variables numéricas después de imputación:")
print(features[num_cols].head())
print("\nMuestra de variables categóricas después de imputación:")
print(features[cat_cols].head())


# Transformar variables categóricas específicas a numéricas
print("\nTransformando variables categóricas específicas a numéricas...")

# Definir mapeos para las tres variables categóricas seleccionadas basados en tiempo promedio de escucha
# Ordenados de mayor a menor tiempo promedio (mayor valor = más escuchado)

# Mapeo para Genre basado en tiempo promedio de escucha
genre_mapping = {
    'Music': 0,         # 46.58 min (más escuchado)
    'True Crime': 1,    # 46.04 min
    'Health': 2,        # 45.74 min
    'Education': 3,     # 45.74 min
    'Technology': 4,    # 45.63 min
    'Business': 5,      # 45.54 min
    'Lifestyle': 6,     # 45.52 min
    'Sports': 7,        # 44.94 min
    'Comedy': 8,        # 44.43 min
    'News': 9           # 44.41 min (menos escuchado)
}

# Mapeo para Publication_Day
day_mapping = {
    'Monday': 0, 
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3, 
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

# Mapeo para variable Publication_Time
time_mapping = {
    'Morning': 0,
    'Afternoon': 1,
    'Evening': 2,
    'Night': 3,
}

# Crear nuevas columnas con valores numéricos para las variables seleccionadas
features['Genre_Numeric'] = features['Genre'].map(genre_mapping)
features['Publication_Day_Numeric'] = features['Publication_Day'].map(day_mapping)
features['Publication_Time_Numeric'] = features['Publication_Time'].map(time_mapping)

episode_counts = features['Episode_Title'].value_counts(normalize=True)
features['Episode_Title_Freq'] = features['Episode_Title'].map(episode_counts)

# Aplicar la misma transformación a los datos de prueba
features_test['Genre_Numeric'] = features_test['Genre'].map(genre_mapping)
features_test['Publication_Day_Numeric'] = features_test['Publication_Day'].map(day_mapping)
features_test['Publication_Time_Numeric'] = features_test['Publication_Time'].map(time_mapping)

episode_counts = features_test['Episode_Title'].value_counts(normalize=True)
features_test['Episode_Title_Freq'] = features_test['Episode_Title'].map(episode_counts)

# Extraer el número del episodio como una variable numérica
print("\nExtrayendo número de episodio como variable numérica...")
features['Episode_Number'] = features['Episode_Title'].str.extract('Episode (\d+)').astype(int)
features_test['Episode_Number'] = features_test['Episode_Title'].str.extract('Episode (\d+)').astype(int)

# Verificar los resultados de la extracción
print("\nVerificación de la extracción del número de episodio:")
print(features[['Episode_Title', 'Episode_Number']].head())

# Eliminar Episode_Title de los datos de entrenamiento y prueba
features.drop(columns=['Episode_Title'], inplace=True)
features_test.drop(columns=['Episode_Title'], inplace=True)

# Verificar los resultados de la transformación
print("\nVerificación de la codificación numérica:")
print(features[['Genre', 'Genre_Numeric', 
                'Publication_Day', 'Publication_Day_Numeric', 
                'Publication_Time', 'Publication_Time_Numeric']].head())

# Actualizar las listas de columnas numéricas y categóricas
num_cols += ['Genre_Numeric', 'Publication_Day_Numeric', 'Publication_Time_Numeric', 'Episode_Number', 'Episode_Title_Freq']

# MANTENER REPRESENTACIÓN DUAL - NO eliminar variables originales de cat_cols
# Esto permite tener tanto la representación categórica original (para One-Hot Encoding)
# como la nueva representación numérica basada en tiempos promedio de escucha


# Imprimir información actualizada después de la transformación
print(f"\nCaracterísticas numéricas después de la transformación: {len(num_cols)}")
print(f"Variables numéricas después de la transformación: {', '.join(num_cols)}")
print(f"\nCaracterísticas categóricas después de la transformación: {len(cat_cols)}")
print(f"Variables categóricas después de la transformación: {', '.join(cat_cols)}")
print("\nNOTA: Se mantiene la representación dual - variables categóricas originales y sus versiones numéricas")


# Asegurar que 'Episode_Title' ya no esté en cat_cols
if 'Episode_Title' in cat_cols:
    cat_cols.remove('Episode_Title')

# Crear pipeline de preprocesamiento
print("\nCreando pipeline de preprocesamiento...")

# Preprocesador para variables categóricas
cat_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='desconocido')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Preprocesador para variables numéricas
num_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combinando los preprocesadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_processor, num_cols),
        ('cat', cat_processor, cat_cols)
    ],
    remainder='drop'
)

# Aplicar preprocesamiento a los datos
print("Aplicando preprocesamiento a los datos...")
X_train_processed = preprocessor.fit_transform(features)
X_test_processed = preprocessor.transform(features_test)

print(f"Forma de X_train después de preprocesamiento: {X_train_processed.shape}")
print(f"Número de características utilizadas: {X_train_processed.shape[1]}")


# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(10, 6))
sns.histplot(y_train, kde=True)
plt.title('Distribución de Listening_Time_minutes')
plt.show()
plt.close()


# Configuración fija para RandomForest
params = {'n_estimators': 450, 'max_depth': None, 'min_samples_split': 2, 'max_features': None}

# Crear y entrenar modelo
print("Entrenando RandomForest con parámetros fijos:")
for param, value in params.items():
    print(f"  - {param}: {value}")

model = RandomForestRegressor(**params, random_state=42, n_jobs=-1, verbose=1)
start_time = time.time()
model.fit(X_train_processed, y_train)
end_time = time.time()
print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")


# Realizar predicciones en los datos de prueba
print("\nRealizando predicciones...")
predictions = model.predict(X_test_processed)

# Asegurar que las predicciones no sean negativas (tiempo de escucha no puede ser negativo)
predictions = np.maximum(0, predictions)

# Crear el DataFrame de resultados
results = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': predictions
})

# Guardar los resultados en un archivo CSV
output_path = '/kaggle/working/submission.csv'
results.to_csv(output_path, index=False)
print(f"Resultados guardados en {output_path}")

# Mostrar las primeras filas del resultado
print("\nPrimeras filas del resultado:")
print(results.head())


# Guardar el modelo y el preprocesador
print("\nGuardando modelo y preprocesador...")
#joblib.dump(model, f'{ruta_base}modelo_regresion.pkl')
#joblib.dump(preprocessor, f'{ruta_base}preprocesador.pkl')


# Calcular métricas en el conjunto de entrenamiento
y_train_pred = model.predict(X_train_processed)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

print(f"\nMétricas en conjunto de entrenamiento:")
print(f"RMSE: {train_rmse:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f}")


# Análisis de residuos
residuals = y_train - y_train_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribución de Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.show()
plt.close()


# Gráfico de dispersión: Valores reales vs. predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, alpha=0.3)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title('Valores Reales vs. Predichos')
plt.xlabel('Tiempo de Escucha Real (minutos)')
plt.ylabel('Tiempo de Escucha Predicho (minutos)')
plt.show()
plt.close()


# Visualizar las 20 características más importantes
feature_importances = model.feature_importances_
plt.figure(figsize=(12, 8))
indices = np.argsort(feature_importances)[-20:]  # Top 20 características
plt.barh(range(len(indices)), feature_importances[indices])
plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
plt.title('Top 20 Características Más Importantes - RandomForest')
plt.show()
plt.close()


# Visualizar métricas de evaluación en un gráfico de barras
plt.figure(figsize=(10, 6))
metrics = ['RMSE', 'R²', 'MAE']
values = [train_rmse, train_r2, train_mae]
colors = ['#FF9999', '#66B2FF', '#99FF99']

# Crear gráfico de barras para las métricas
bars = plt.bar(metrics, values, color=colors, width=0.5)

# Añadir etiquetas y título
plt.title('Métricas de Evaluación del Modelo Final', fontsize=15)
plt.ylabel('Valor', fontsize=12)
plt.xticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir los valores encima de las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=11)

# Guardar gráfico
plt.show()
plt.close()


# Imprimir resumen final
print("\nResumen del modelo:")
print(f"Modelo: RandomForest con los parámetros {params}")
print(f"RMSE en entrenamiento: {train_rmse:.4f}")
print(f"R² en entrenamiento: {train_r2:.4f}")
print(f"MAE en entrenamiento: {train_mae:.4f}")

