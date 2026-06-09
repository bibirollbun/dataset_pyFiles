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


# ===============================================
# IoT Preprocessing Challenge - Starter Notebook (Educativo)
# ===============================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# === Cargar los datos ===
train = pd.read_csv("/kaggle/input/io-t-data-preprocessing-kaggle-summer-school-urjc/train.csv")
test = pd.read_csv("/kaggle/input/io-t-data-preprocessing-kaggle-summer-school-urjc/test.csv")
sample_submission = pd.read_csv("/kaggle/input/io-t-data-preprocessing-kaggle-summer-school-urjc/sample_submission.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)





# === Análisis Exploratorio de Datos (EDA) ===

# === Columna objetivo ===
TARGET = "benign"

# Convertir valores a 0/1 si son TRUE/FALSE
for df in [train, test]:
    df.replace({"TRUE": 1, "FALSE": 0, True: 1, False: 0}, inplace=True)
    df.infer_objects(copy=False)


# Crear lista de características
features = [col for col in train.columns if col not in ["id", TARGET]]

# === Separar características numéricas y categóricas ===
categorical_cols = train[features].select_dtypes(include=["object"]).columns.tolist()
numerical_cols = train[features].select_dtypes(include=["int64", "float64", "bool", "int"]).columns.tolist()


# 1. Vista rápida de las primeras filas
print("Primeras 5 filas:")
display(train.head())

# 2. Información general: tipos y valores nulos
print("\nInformación del DataFrame:")
train.info()

print("\nValores nulos por columna:")
print(train.isna().sum().sort_values(ascending=False))

# 3. Estadísticas descriptivas de variables numéricas
print("\nEstadísticas de variables numéricas:")
display(train[numerical_cols].describe().T)

# 4. Distribución de la variable objetivo
plt.figure(figsize=(6,4))
sns.countplot(x=TARGET, data=train)
plt.title("Distribución de la clase objetivo")
plt.show()

# 5. Histograma de las variables numéricas
train[numerical_cols].hist(bins=20, figsize=(12,10))
plt.suptitle("Histogramas de variables numéricas")
plt.tight_layout()
plt.show()

# 6. Boxplots para detectar outliers
plt.figure(figsize=(12, len(numerical_cols)*2))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(len(numerical_cols), 1, i)
    sns.boxplot(x=train[col])
    plt.title(f"Boxplot de {col}")
plt.tight_layout()
plt.show()

# 7. Matriz de correlación entre numéricas
corr = train[numerical_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mapa de calor de correlaciones")
plt.show()

# 8. Análisis de variables categóricas
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(y=col, data=train, order=train[col].value_counts().index)
    plt.title(f"Frecuencias de {col}")
    plt.show()



# === Configuración de la opción de preprocesamiento ===
preprocessing_option = "standard+ohe"  # opciones: "raw", "standard+ohe", "minmax+label", etc.


# ====================================
# SWITCH-STYLE PREPROCESSING OPTIONS
# ====================================
if preprocessing_option == "raw":
    # Codificar columnas categóricas con LabelEncoder (mínimo necesario para funcionar)
    train_raw = train.copy()
    test_raw = test.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        train_raw[col] = le.fit_transform(train_raw[col].astype(str))
        test_raw[col] = le.transform(test_raw[col].astype(str))

    X_train = train_raw[features].values
    X_test = test_raw[features].values


elif preprocessing_option == "standard+ohe":
    train_encoded = pd.get_dummies(train[categorical_cols])
    test_encoded = pd.get_dummies(test[categorical_cols])
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    scaler = StandardScaler()
    train_scaled_num = scaler.fit_transform(train[numerical_cols])
    test_scaled_num = scaler.transform(test[numerical_cols])

    X_train = np.hstack([train_scaled_num, train_encoded.values])
    X_test = np.hstack([test_scaled_num, test_encoded.values])

elif preprocessing_option == "minmax+label":
    # Codificar categorías con LabelEncoder
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        le_dict[col] = le

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])

else:
    raise ValueError(f"Opción no válida: {preprocessing_option}")

# === Codificar la variable objetivo ===
le_target = LabelEncoder()
y = le_target.fit_transform(train[TARGET])

# === Dividir para evaluación local ===
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.3, stratify=y, random_state=42)

# === Entrenar modelo ===
clf = LogisticRegression(max_iter=100, solver="lbfgs", random_state=42)
clf.fit(X_tr, y_tr)

# === Predicción en validación ===
y_pred = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)[:, 1]  # para AUC

# === Métricas de evaluación ===
print("=== Métricas de Validación ===")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1-score:", f1_score(y_val, y_pred))

# AUC solo si tenemos 2 clases
if len(np.unique(y_val)) == 2:
    print("ROC AUC:", roc_auc_score(y_val, y_prob))

print("\nReporte de clasificación:")
target_names_str = [str(cls) for cls in le_target.classes_]
print(classification_report(y_val, y_pred, target_names=target_names_str))

# === Predicción final para Kaggle ===
final_pred = clf.predict(X_test)
final_labels = le_target.inverse_transform(final_pred)

submission = test[["id"]].copy()
submission[TARGET] = final_labels
submission.to_csv("submission.csv", index=False)

print("✅ Archivo 'submission.csv' generado.")
submission.head()

