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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

print("Tamaño del train:", train.shape)
print("Tamaño del test:", test.shape)
print("Tamaño del sample_submission:", sample_submission.shape)

# Vista rápida de los datos
train.head()


train.info()


train.describe()


train.isnull().sum()


sns.countplot(data=train, x="Personality")
plt.title("Distribución de Introvert vs Extrovert")
plt.show()



# dist. numéricas
train.hist(bins=30, figsize=(15, 10))
plt.show()


for col in train.select_dtypes(include="object"):
    print(f"\nValores únicos de {col}:\n", train[col].unique())


test.head()
test.isnull().sum()

print("Tipos de datos únicos en train:", train.dtypes.unique().tolist())
print("Tipos de datos únicos en test:", test.dtypes.unique().tolist())


cols_num = train.select_dtypes(include=["int64", "float64"]).columns
cols_cat = train.select_dtypes(include=["object"]).columns

print("Número de columnas numéricas:", len(cols_num))
print("Número de columnas categóricas:", len(cols_cat))


faltantes_train = train.isnull().sum().sort_values(ascending=False)
faltantes_train = faltantes_train[faltantes_train > 0]

if not faltantes_train.empty:
    plt.figure(figsize=(10,5))
    sns.barplot(x=faltantes_train.index, y=faltantes_train.values)
    plt.xticks(rotation=90)
    plt.title("Valores faltantes por columna (Train)")
    plt.show()

faltantes_test = test.isnull().sum().sort_values(ascending=False)
faltantes_test = faltantes_test[faltantes_test > 0]

if not faltantes_test.empty:
    plt.figure(figsize=(10,5))
    sns.barplot(x=faltantes_test.index, y=faltantes_test.values)
    plt.xticks(rotation=90)
    plt.title("Valores faltantes por columna (Test)")
    plt.show()



train["Time_spent_Alone"].describe()

sns.histplot(
    data=train,
    x="Time_spent_Alone",
    hue="Personality",
    bins=12,
    kde=True,
    multiple="stack"
)
plt.title("Distribución del 'Tiempo a solas' según: Personalidad")
plt.show()


print("Valores únicos de 'Going_outside':")
print(train["Going_outside"].unique())


print("\nRango de 'Going_outside':")
print("  Máx:", train["Going_outside"].max())
print("  Mín:", train["Going_outside"].min())


print("\nRango de 'Friends_circle_size':")
print("  Máx:", train["Friends_circle_size"].max())
print("  Mín:", train["Friends_circle_size"].min())


print("\nRango de 'Post_frequency':")
print("  Máx:", train["Post_frequency"].max())
print("  Mín:", train["Post_frequency"].min())


def codificar_binarias(df):
    """
    se convierte n algunas columnas que solo tienen 'yes / no' 
    a valores numéricos (1 / 0). 
    Se convierte el target:
       - Extrovert -> 1
       - Introvert -> 0
    """
    bin_cols = [c for c in df.columns if df[c].dtype == "object" 
                and set(df[c].dropna().unique()) <= {"Yes", "No"}]

    for col in bin_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
        
    if "Personality" in df.columns:
        df["Personality"] = df["Personality"].map({
            "Introvert": 0,
            "Extrovert": 1
        })
        
    return df

train_encoded = codificar_binarias(train.copy())
test_encoded = codificar_binarias(test.copy())


# copia para pca
pca_df = train_encoded.copy()

# eliminamos nans
pca_df = pca_df.dropna()

num_cols = pca_df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["id", "Personality"]]

# escalamos solo para pca
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_df[num_cols])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=pca_df["Personality"],
    palette="coolwarm"
)
plt.title("PCA")
plt.show()

#print("Varianza explicada por PC1 y PC2:", pca.explained_variance_ratio_)




X = train_encoded.drop(columns=["Personality", "id"])
y = train_encoded["Personality"]

X_test = test_encoded.drop(columns=["id"])


numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print("Numéricas:", numeric_cols)
print("Categóricas:", categorical_cols)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)

logreg = LogisticRegression(max_iter=1000, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", logreg)
])



from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)


# baseline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# dummy baseline 
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_scores = cross_val_score(dummy, X, y, cv=cv, scoring="accuracy")
print("Dummy (most_frequent) accuracy por fold:", np.round(dummy_scores, 4))
print("Dummy (most_frequent) accuracy promedio:", dummy_scores.mean())

# baseline de regresion logistica usando el pipeline
lr_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print("\nLogReg accuracy por fold:", np.round(lr_scores, 4))
print("LogReg accuracy promedio:", lr_scores.mean())



# preds por cross vañl para sacar la matriz de confusión y ROC 
y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
y_proba_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]  


# matriz de confusión
cm = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Introvert","Extrovert"], yticklabels=["Introvert","Extrovert"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de confusión (cross validation)")
plt.show()


print("\nReporte de clasificación (cross validation):")
print(classification_report(y, y_pred_cv, target_names=["Introvert","Extrovert"]))


roc_auc = roc_auc_score(y, y_proba_cv)
print("ROC-AUC (cross validation):", roc_auc)

fpr, tpr, _ = roc_curve(y, y_proba_cv)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"LogReg (CV) AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC (validación cruzada)")
plt.legend()
plt.show()


print(y.unique())



pipeline.fit(X, y)



test_preds = pipeline.predict(X_test)



from sklearn.model_selection import GridSearchCV

# ddef el pipeline de nuevo con el nombre del modelo que cambiaremos
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

# h iperparámetros a buscar
param_grid_lr = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs", "liblinear"]
}

grid_lr = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid_lr,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid_lr.fit(X, y)

print("Mejores parámetros (Logistic Regression):", grid_lr.best_params_)
print("Mejor score (Logistic Regression):", grid_lr.best_score_)



from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

param_grid_rf = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

grid_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid_rf,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid_rf.fit(X, y)

print("Mejores parámetros (Random Forest):", grid_rf.best_params_)
print("Mejor score (Random Forest):", grid_rf.best_score_)



pipeline_final = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    ))
])



scores = cross_val_score(pipeline_final, X, y, cv=5, scoring="accuracy")
print("Cross-validation accuracy promedio:", scores.mean())



# Entrenar en todo el set de entrenamiento
pipeline_final.fit(X, y)

# Predicciones sobre el set de entrenamiento
train_preds = pipeline_final.predict(X)


acc = accuracy_score(y, train_preds)
print(f"Accuracy en el set de entrenamiento: {acc:.4f}")



print("\nReporte de clasificación:")
print(classification_report(y, train_preds, target_names=["Introvert", "Extrovert"]))


# Matriz de confusión
cm = confusion_matrix(y, train_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Introvert","Extrovert"],
            yticklabels=["Introvert","Extrovert"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de confusión: Modelo Final")
plt.show()



# preds sobre el test set
test_preds = pipeline_final.predict(X_test)


# Convertir a etiquetas para Kaggle
test_labels = np.where(test_preds == 1, "Extrovert", "Introvert")

# Crear archivo de submission
submission = pd.DataFrame({
    "id": test_encoded["id"],
    "Personality": test_labels
})
submission.to_csv("submission.csv", index=False)
print("Archivo submission_final.csv generado")

