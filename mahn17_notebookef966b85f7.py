import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Para el Baseline
from sklearn.ensemble import HistGradientBoostingClassifier # Modelo principal
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Carga de los archivos desde la ruta de Kaggle
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que los archivos 'train.csv' y 'test.csv' estén en la ruta correcta.")


# Separación de X, y, y datos de prueba
X = train_df.drop(['id', 'Personality'], axis=1)
y_text = train_df['Personality']
X_test_final = test_df.drop('id', axis=1)
test_ids = test_df['id']

# Codificación de la variable objetivo (Y)
le = LabelEncoder()
y = le.fit_transform(y_text)
print(f"Clases codificadas: {list(le.classes_)} (0, 1)")

#División para Entrenamiento y Validación (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definición de Columnas
num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Columna con alto skewness (según el análisis del usuario: 1.13)
col_time = ['Time_spent_Alone']

# Resto de columnas numéricas (bajo skewness/simétricas)
num_cols_sin_time = num_cols.drop('Time_spent_Alone').tolist()


print("\n Estructura y Datos Faltantes")
print(X.info())
print("\nDatos Faltantes (NaN):")
print(X.isnull().sum())

print("\n Distribución Numérica (Skewness y Outliers)")

outliers_dict = {}

for col in num_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filtrar outliers
    outliers = X[(X[col] < Q1 - 1.5*IQR) | (X[col] > Q3 + 1.5*IQR)]
    
    outliers_dict[col] = outliers.shape[0]  # cuántos outliers tiene cada columna
print("\n IQR:")
print(outliers_dict)

skew_values = X.select_dtypes(include=['number']).skew()
print("\n Skew:")
print(skew_values)

print("\n Distribución de la Variable Objetivo:")
print(y_text.value_counts(normalize=True))

print("\n Naturaleza de las Variables Categóricas:")
print(f"Stage_fear únicos: {X['Stage_fear'].unique()}")
print(f"Drained_after_socializing únicos: {X['Drained_after_socializing'].unique()}")


# Preprocesamiento básico para Visualización
X_viz = X.fillna(X.median(numeric_only=True)).fillna('Missing')
X_viz = pd.get_dummies(X_viz, drop_first=True)
X_viz_scaled = StandardScaler().fit_transform(X_viz)

# Aplicación de PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_viz_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_text, alpha=0.5)
plt.title('Visualización con PCA (Separación Lineal)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.show()

# Aplicación de t-SNE
sample_size = 2000
if len(X_viz_scaled) > sample_size:
    idx = np.random.choice(len(X_viz_scaled), sample_size, replace=False)
    X_tsne_sample = X_viz_scaled[idx]
    y_tsne_sample = y_text.iloc[idx]
else:
    X_tsne_sample = X_viz_scaled
    y_tsne_sample = y_text

tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=30)
X_tsne = tsne.fit_transform(X_tsne_sample)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_tsne_sample, alpha=0.7)
plt.title('Visualización con t-SNE (Estructura No Lineal)')
plt.show()


 #Pipeline para Time_spent_Alone (Alto Skewness)
log_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Imputación antes de la transformación logarítmica
    ('log_transform', FunctionTransformer(np.log1p, validate=True)), # np.log(1+x)
    ('scaler', StandardScaler())
])

#Pipeline para otras numéricas (Simétricas/Bajo Skewness)
std_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#Pipeline para categóricas (Imputación constante por 'falta')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='falta')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

#Combinación de Transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('log', log_transformer, col_time),
        ('std', std_transformer, num_cols_sin_time),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)


# Modelo Inicial
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])
baseline_pipeline.fit(X_train, y_train)

y_pred_baseline = baseline_pipeline.predict(X_val)
f1_baseline = f1_score(y_val, y_pred_baseline)
print(f"\n Resultado del Baseline (Regresión Logística): F1-score = {f1_baseline:.4f}")


# Modelo Candidato
hgbc_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Espacio de búsqueda y GridSearchCV
param_grid = {
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_leaf_nodes': [15, 31, 50],
}

grid_search = GridSearchCV(
    hgbc_pipeline,
    param_grid,
    cv=3, 
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

print("Iniciando búsqueda de hiperparámetros para HistGradientBoosting...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n Resultado de GridSearchCV:")
print(f"Mejores Hiperparámetros: {grid_search.best_params_}")
print(f"Mejor F1-score (Validación Cruzada): {grid_search.best_score_:.4f}")


# k-Cross-Fold Validation 
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1', n_jobs=-1)
print(f" Resultados de 5-Fold Cross-Validation (F1-score):")
print(f"Scores: {cv_scores}")
print(f"Promedio (mu): {cv_scores.mean():.4f}")
print(f"Desviación Estándar (sigma): {cv_scores.std():.4f}.")

# Evaluación en el conjunto de validación (X_val)
y_pred_best = best_model.predict(X_val)
y_proba_best = best_model.predict_proba(X_val)[:, 1]

# Métricas de Entrenamiento y Validación
print("\n 6.2 Reporte de Clasificación en el Conjunto de Validación:")
print(classification_report(y_val, y_pred_best, target_names=le.classes_))

# Matriz de Confusión
cm = confusion_matrix(y_val, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusión del Modelo Retenido')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Curva ROC y AUC
fpr, tpr, thresholds = roc_curve(y_val, y_proba_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



#Re-entrenar con todos los datos de entrenamiento (X, y) para el máximo rendimiento
best_model.fit(X, y) 

# Predicción en el conjunto de prueba de Kaggle
predictions_encoded = best_model.predict(X_test_final)
predictions_text = le.inverse_transform(predictions_encoded)

#Creación del Submission
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': predictions_text
})

submission_df.to_csv('submission.csv', index=False)
print("Archivo 'submission.csv' generado correctamente.")
print("\nLas primeras 5 predicciones son:")
print(submission_df.head())


sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
print(sample.head())

