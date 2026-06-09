# ğŸ“Œ 1ï¸�âƒ£ Importar LibrerÃ­as
import pandas as pd
import numpy as np
import joblib
import optuna
import lightgbm as lgb
import xgboost as xgb
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# ğŸ“Œ 2ï¸�âƒ£ Verificar y Cargar Datos
def load_data():
    kaggle_path = "/kaggle/input"
    local_path = "."

    paths = [os.path.join(kaggle_path, d) for d in os.listdir(kaggle_path)] if os.path.exists(kaggle_path) else [local_path]

    for path in paths:
        train_path, test_path = os.path.join(path, "train.csv"), os.path.join(path, "test.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"âœ… Datos encontrados en: {path}")
            return pd.read_csv(train_path), pd.read_csv(test_path)

    raise FileNotFoundError("â�Œ ERROR: Archivos train.csv o test.csv no encontrados.")

train, test = load_data()

# ğŸ“Œ 3ï¸�âƒ£ Feature Engineering Avanzado
def feature_engineering(df):
    df["feature_sum"] = df.select_dtypes(include=np.number).sum(axis=1)
    df["feature_mean"] = df.select_dtypes(include=np.number).mean(axis=1)
    df["feature_std"] = df.select_dtypes(include=np.number).std(axis=1)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# ğŸ“Œ 4ï¸�âƒ£ Preprocesamiento
num_features = train.select_dtypes(include=['int64', 'float64']).columns.drop("Price")
cat_features = train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

# ğŸ“Œ 5ï¸�âƒ£ ReducciÃ³n de Dimensiones con PCA
pca = PCA(n_components=30)  # Ajustar si es necesario

# ğŸ“Œ 6ï¸�âƒ£ DivisiÃ³n de Datos
X = train.drop(columns=["Price"])
y = train["Price"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸ“Œ 7ï¸�âƒ£ CodificaciÃ³n de Variables CategÃ³ricas (Para LightGBM)
def encode_categorical(df):
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convertimos a string por seguridad
    return df

X_train = encode_categorical(X_train)
X_val = encode_categorical(X_val)
X_test = encode_categorical(test.drop(columns=["id"]))

# ğŸ“Œ 8ï¸�âƒ£ OptimizaciÃ³n de HiperparÃ¡metros con Optuna
def optimize_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    return mean_squared_error(y_val, y_pred, squared=False)

study = optuna.create_study(direction="minimize")
study.optimize(optimize_lgb, n_trials=20)
best_params = study.best_params

# ğŸ“Œ 9ï¸�âƒ£ Modelos Optimizados
lgb_model = lgb.LGBMRegressor(**best_params)
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6)
ridge = Ridge(alpha=1.0)

# ğŸ“Œ ğŸ”Ÿ Stacking Avanzado
stacking_model = StackingRegressor(estimators=[
    ('lgb', lgb_model),
    ('xgb', xgb_model),
    ('ridge', ridge)
], final_estimator=Ridge())

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", pca),
    ("model", stacking_model)
])

# ğŸ“Œ ğŸš€ Entrenamiento
pipeline.fit(X_train, y_train)

# ğŸ“Œ EvaluaciÃ³n
y_pred = pipeline.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"âœ… RMSE Stacking: {rmse:.5f}")

# ğŸ“Š GrÃ¡fico del Error Residual
plt.figure(figsize=(6, 4))
sns.histplot(y_val - y_pred, bins=30, kde=True, color="blue", label="Stacking")
plt.axvline(0, color="black", linestyle="--")
plt.title("DistribuciÃ³n del Error Residual")
plt.legend()
plt.show()

# ğŸ“Œ Generar Predicciones para Test
test["Price"] = pipeline.predict(X_test)

# ğŸ“Œ Guardar Predicciones
submission = test[["id", "Price"]]
submission.to_csv("submission.csv", index=False)
print("âœ… Archivo submission.csv generado correctamente.")

# ğŸ“Œ Guardar Modelo
joblib.dump(pipeline, "stacking_model.pkl")
print("âœ… Modelo guardado correctamente.")

