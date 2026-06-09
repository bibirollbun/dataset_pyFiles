import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings 
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
test=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv',header=None)
trainLabels=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv',header=None)


train=train.copy()
train['target']=trainLabels
train


import matplotlib.pyplot as plt
import seaborn as sns

# Filtrar los datos donde target == 1
df_target1 = train[train['target'] == 1]

# Crear una grilla de 10x4 subplots
fig, axes = plt.subplots(10, 4, figsize=(20, 25))  # tamaño ajustable
fig.suptitle('Boxplots por columna (target = 1)', fontsize=18)

# Aplanar los ejes para poder iterar fácilmente
axes = axes.flatten()

# Generar los boxplots
for i in range(40):
    sns.boxplot(x=df_target1.iloc[:, i], ax=axes[i])
    axes[i].set_title(f'Columna {i}')
    axes[i].set_xlabel('')

# Ajustar espacio entre subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()



# Filtrar los datos donde target == 0
df_target1 = train[train['target'] == 0]

# Crear una grilla de 10x4 subplots
fig, axes = plt.subplots(10, 4, figsize=(20, 25))  # tamaño ajustable
fig.suptitle('Boxplots por columna (target = 0)', fontsize=18)

# Aplanar los ejes para poder iterar fácilmente
axes = axes.flatten()

# Generar los boxplots
for i in range(40):
    sns.boxplot(x=df_target1.iloc[:, i], ax=axes[i])
    axes[i].set_title(f'Columna {i}')
    axes[i].set_xlabel('')

# Ajustar espacio entre subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


def eliminar_outliers_iqr(df, columnas):
    for col in columnas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Aplicar a todas las columnas numéricas
columnas_numericas = list(range(40))
train = eliminar_outliers_iqr(train.copy(), columnas_numericas)



train.iloc[:, :41]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(train.iloc[:, :40])

# Convertir de nuevo a DataFrame si lo deseas
X_scaled = pd.DataFrame(X_scaled, columns=[f'col_{i}' for i in range(40)])



scaler = MinMaxScaler()
X_scaled_test = scaler.fit_transform(test.iloc[:, :40])

# Convertir de nuevo a DataFrame si lo deseas
X_scaled_test = pd.DataFrame(X_scaled_test, columns=[f'col_{i}' for i in range(40)])


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


X, y = X_scaled, train['target']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()

# Definimos la grilla de hiperparámetros
param_grid = {
    'n_neighbors': list(range(1, 25)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
}

# GridSearchCV con validación cruzada de 10 folds
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Mejores parámetros encontrados:")
print(grid.best_params_)

print(f"\nMejor accuracy en validación cruzada: {grid.best_score_:.4f}")

# Evaluar con el mejor modelo
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)

# Resultados
print("\nAccuracy en test set:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Instanciar el modelo base
rf = RandomForestClassifier(random_state=42)

# 2. Definir la grilla de hiperparámetros
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# 3. Ejecutar GridSearchCV
grid_rf = GridSearchCV(rf, param_grid_rf, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

# 4. Evaluar resultados
print("Mejores parámetros encontrados para Random Forest:")
print(grid_rf.best_params_)
print(f"\nMejor accuracy en validación cruzada: {grid_rf.best_score_:.4f}")

# 5. Evaluar en test
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\nAccuracy en test set:", accuracy_score(y_test, y_pred_rf))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred_rf))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred_rf))



from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


# Modelo base
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Grid de hiperparámetros
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# GridSearchCV
grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb,
                        cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

# Entrenar
grid_xgb.fit(X_train, y_train)

print("Mejores parámetros encontrados para XGBoost:")
print(grid_xgb.best_params_)
print(f"\nMejor accuracy en validación cruzada: {grid_xgb.best_score_:.4f}")

best_xgb = grid_xgb.best_estimator_

# Predecir en test
y_pred_xgb = best_xgb.predict(X_test)

# Evaluar
print("\nAccuracy en test set:", accuracy_score(y_test, y_pred_xgb))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred_xgb))



y_pred = best_knn.predict(X_scaled_test)
# Convertir el array a tabla
df_pred = pd.DataFrame({
    'ID': range(1, len(y_pred)+1), 
    'Solution': y_pred
})
df_pred


df_pred.to_csv('Submission.csv', index=False)

