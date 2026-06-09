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


import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col="id")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
X = train.drop(['Personality'], axis=1)
y = train['Personality']


train.info()


train.head()


categorical_columns = ["Stage_fear", "Drained_after_socializing"]
numerical_columns = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]


print('Valores faltantes por columna')
pd.concat([train.isna().sum(0), (train.isna().sum()/train.shape[0])*100, test.isna().sum(0), (test.isna().sum()/test.shape[0])*100], axis=1).rename({0: 'train', 1:'%', 2: 'test', 3:'%'}, axis=1)


train[numerical_columns].describe()


for col in numerical_columns:
  sns.displot(train, x=col, hue='Personality', discrete=True)


train[categorical_columns].describe()


sns.countplot(data=train, x="Stage_fear", hue='Personality')



sns.countplot(data=train, x="Drained_after_socializing", hue='Personality')


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# preprocesamiento para datos numericos
numerical_transformer = Pipeline(
    steps=[
        ('mputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())])

# preprocesamiento para datos categoricos
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal_encoder', OrdinalEncoder())
])

# preprocesamiento para datos numericos y categoricos
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Modelo de regresión logística
model = LogisticRegression(random_state=42, max_iter=1000)

# Pipeline completo con preprocesamiento y modelo
my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor), 
        ('classifier', model)
    ]
)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Codificar la variable objetivo
from sklearn.preprocessing import LabelEncoder

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Ajustar LabelEncoder SOLO con datos de entrenamiento
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)  

# Aplicar pipeline
X_train_processed = my_pipeline.fit(X_train, y_train_encoded)

# Predicción con el modelo entrenado
preds = my_pipeline.predict(X_val)



from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

class_names = le.classes_

cm = confusion_matrix(y_val_encoded, preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión absoluta
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], cbar_kws={'label': 'Cantidad'})
axes[0].set_title('Matriz de Confusión (Valores Absolutos) - Modelo Baseline', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicción', fontsize=12)
axes[0].set_ylabel('Valor Real', fontsize=12)

# Matriz de confusión porcentajes
cm_normalized = confusion_matrix(y_val_encoded, preds, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], cbar_kws={'label': 'Proporción'})
axes[1].set_title('Matriz de Confusión Normalizada (Porcentajes) - Modelo Baseline', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicción', fontsize=12)
axes[1].set_ylabel('Valor Real', fontsize=12)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("MÉTRICAS DE CALIDAD DEL MODELO BASELINE")
print("="*60)

# Métricas por clase
precision = precision_score(y_val_encoded, preds, average=None)
recall = recall_score(y_val_encoded, preds, average=None)
f1 = f1_score(y_val_encoded, preds, average=None)

print(f"\n{'Clase':<15} {'Precisión':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 60)
for i, class_name in enumerate(class_names):
    print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")

# Accuracy
print(f"\n{'Accuracy':<15} {accuracy_score(y_val_encoded, preds):<12.4f}")



# GridSearchCV integrado con el pipeline completo
from sklearn.model_selection import GridSearchCV

# Definir la grilla de parámetros para búsqueda exhaustiva
# Expandimos la grilla para explorar más combinaciones y encontrar mejores resultados
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100],  # Más valores de regularización (10 valores)
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],  # Incluimos elasticnet también (3 valores)
    'classifier__solver': ['liblinear', 'lbfgs', 'saga'],  # Agregamos saga para elasticnet (3 valores)
    'classifier__max_iter': [500, 1000, 1500, 2000]  # Más opciones de iteraciones (4 valores)
}

# Nota sobre compatibilidad de solvers y penalties:
# - 'liblinear': compatible con 'l1' y 'l2'
# - 'lbfgs': compatible solo con 'l2'
# - 'saga': compatible con 'l1', 'l2' y 'elasticnet'
# GridSearchCV manejará automáticamente las combinaciones inválidas

# Inicializar el modelo en el pipeline con parámetros por defecto
# Estos son los parámetros base que se usarán si no se especifican en la búsqueda
my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor), 
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]
)

# GridSearchCV con el pipeline completo
grid_search_pipeline = GridSearchCV(
    estimator=my_pipeline,  
    param_grid=param_grid,
    cv=5,  
    scoring='accuracy',
    n_jobs=-1,  
    verbose=1
)

# Ajustar el modelo 
total_combinations = (len(param_grid['classifier__C']) * 
                     len(param_grid['classifier__penalty']) * 
                     len(param_grid['classifier__solver']) * 
                     len(param_grid['classifier__max_iter']))
total_fits = total_combinations * 5  

print("Iniciando GridSearchCV con pipeline completo...")
print("Esto incluye preprocesamiento automático en cada fold de CV")
print(f"\nParámetros de la grilla:")
print(f"  - C: {len(param_grid['classifier__C'])} valores")
print(f"  - penalty: {len(param_grid['classifier__penalty'])} valores")
print(f"  - solver: {len(param_grid['classifier__solver'])} valores")
print(f"  - max_iter: {len(param_grid['classifier__max_iter'])} valores")
print(f"\nTotal de combinaciones a probar: {total_combinations}")
print(f"Total de fits: {total_fits} ({total_combinations} combinaciones × 5 folds)")
print("\nNota: Algunas combinaciones pueden ser inválidas (ej: lbfgs con l1), GridSearchCV las omitirá automáticamente")
print("Iniciando búsqueda...\n")


grid_search_pipeline.fit(X_train, y_train_encoded) 

# Resultados
print(f"\nMejores parámetros: {grid_search_pipeline.best_params_}")
print(f"Mejor score (CV): {grid_search_pipeline.best_score_:.4f}")

# Obtener el mejor modelo (pipeline completo optimizado)
best_pipeline = grid_search_pipeline.best_estimator_

# Evaluar en conjunto de validación
y_pred_pipeline_encoded = grid_search_pipeline.predict(X_val)  
print(f"\nAccuracy en validación: {accuracy_score(y_val_encoded, y_pred_pipeline_encoded):.4f}")

# Convertir predicciones a etiquetas originales para comparación
y_pred_pipeline = le.inverse_transform(y_pred_pipeline_encoded)



# Evaluación final del modelo optimizado con datos de validación
print("="*60)
print("EVALUACIÓN FINAL DEL MODELO OPTIMIZADO")
print("="*60)

# Matriz de confusión del modelo optimizado
cm_optimized = confusion_matrix(y_val, y_pred_pipeline)
cm_optimized_normalized = confusion_matrix(y_val, y_pred_pipeline, normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión absoluta
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], cbar_kws={'label': 'Cantidad'})
axes[0].set_title('Matriz de Confusión - Modelo Optimizado (Valores Absolutos)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicción', fontsize=12)
axes[0].set_ylabel('Valor Real', fontsize=12)

# Matriz de confusión normalizada
sns.heatmap(cm_optimized_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], cbar_kws={'label': 'Proporción'})
axes[1].set_title('Matriz de Confusión - Modelo Optimizado (Porcentajes)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicción', fontsize=12)
axes[1].set_ylabel('Valor Real', fontsize=12)

plt.tight_layout()
plt.show()

# Métricas detalladas del modelo optimizado
print("\n" + "="*60)
print("MÉTRICAS DE CALIDAD DEL MODELO OPTIMIZADO")
print("="*60)

# Métricas por clase
precision_opt = precision_score(y_val, y_pred_pipeline, average=None)
recall_opt = recall_score(y_val, y_pred_pipeline, average=None)
f1_opt = f1_score(y_val, y_pred_pipeline, average=None)

print(f"\n{'Clase':<15} {'Precisión':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 60)
for i, class_name in enumerate(class_names):
    print(f"{class_name:<15} {precision_opt[i]:<12.4f} {recall_opt[i]:<12.4f} {f1_opt[i]:<12.4f}")

# Accuracy
print(f"\n{'Accuracy':<15} {accuracy_score(y_val, y_pred_pipeline):<12.4f}")



print("Generando predicciones para el conjunto de prueba...")

predictions_encoded = best_pipeline.predict(test)

# Verificar el tipo de predicciones
print(f"Tipo de predicciones: {type(predictions_encoded)}")
print(f"Valores únicos en predicciones: {np.unique(predictions_encoded)}")
print(f"Primeras 5 predicciones: {predictions_encoded[:5]}")

# Convertir predicciones de vuelta a etiquetas originales
predictions_labels = le.inverse_transform(predictions_encoded)

# Crear submission
submission = sample_submission.copy()
submission['Personality'] = predictions_labels

print("\n=== PRIMERAS 10 PREDICCIONES ===")
print(submission.head(10))

print(f"\n=== DISTRIBUCIÓN DE PREDICCIONES EN TEST ===")
print(submission['Personality'].value_counts())
print(f"\nTotal de predicciones: {len(submission)}")

# Guardar submission
submission.to_csv('submission.csv', index=False)


