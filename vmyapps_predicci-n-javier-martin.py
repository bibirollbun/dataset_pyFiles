import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import optuna

# Cargar datos
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Eliminar columnas irrelevantes
train.drop(columns=['id', 'day'], inplace=True)
test.drop(columns=['id', 'day'], inplace=True)

# Separar variables
X = train.drop(columns=['rainfall'])
y = train['rainfall']

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# Separar en train y validación
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo inicial LightGBM
model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluación inicial
preds = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, preds)
print(f'AUC-ROC en validación: {auc:.4f}')

# Optimización con Optuna
def objective(trial):
    num_leaves = trial.suggest_int("num_leaves", 20, 100)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    model = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Entrenar modelo con mejores parámetros
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_scaled, y)

# Predicciones finales
final_preds = best_model.predict_proba(test_scaled)[:, 1]

# Guardar archivo de envío
submission = pd.DataFrame({'id': pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')['id'], 
                           'rainfall': final_preds})
submission.to_csv('submission.csv', index=False)

print(submission.head())  # Verificar formato

