#Librerías y configuración
import os, random, sys, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import lightgbm as lgb 

SEED = 42
np.random.seed(SEED); random.seed(SEED)


#Carga de datos
train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")

train.shape, test.shape, sample.shape


# MUESTREO 0.01%
frac_sample = 0.0001  # 0.01%
train = train.sample(frac=frac_sample, random_state=42).reset_index(drop=True)
print("Tamaño tras muestreo 0.01%:", train.shape)


train.head()


train.columns
train['class'].value_counts(normalize=True)


# Definición de variables y objetivo
TARGET = "class"
ID_COL = "id"

# Convertir la variable objetivo de texto ('p','e') a binaria (1,0)
y = train[TARGET].map({'p': 1, 'e': 0}).copy()

# Separar los conjuntos de datos
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

# Identificar columnas categóricas y numéricas
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number", "float", "int"]).columns.tolist()

print("Categóricas:", len(cat_cols), "Numéricas:", len(num_cols))
print("Ejemplo categóricas:", cat_cols[:5])
print("Ejemplo numéricas:", num_cols[:5])


# PREPROCESAMIENTO
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "class" 

X = train.drop(columns=[TARGET])
y = train[TARGET]

# Detecta tipos
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print(f"Categóricas ({len(cat_cols)}):", cat_cols[:10])
print(f"Numéricas   ({len(num_cols)}):", num_cols[:10])

# Transformadores por tipo
#    - Categóricas: imputación por moda + One-Hot 
#    - Numéricas: imputación por moda + estandarización
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  
])

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", StandardScaler(with_mean=True, with_std=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ],
    remainder="drop"
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Ajusta SOLO el preprocesador (sin PCA aún)
X_train_transformed = preprocessor.fit_transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)

print("Tipo de X_train_transformed:", type(X_train_transformed))


# PCA: grafica de varianza 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# Para graficar la varianza explicada, densificamos temporalmente el conjunto transformado
def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

X_train_dense = _to_dense(X_train_transformed)

# PCA “exploratorio” para elegir número de componentes
pca_probe = PCA(svd_solver="full", random_state=42)
pca_probe.fit(X_train_dense)

expl_var = pca_probe.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

plt.figure(figsize=(7,4))
plt.plot(cum_var, marker="o")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("PCA - Varianza explicada acumulada")
plt.grid(True)
plt.show()

# Elegimos n_components a partir de un umbral
threshold = 0.95
n_comp = int(np.searchsorted(cum_var, threshold) + 1)
print(f"Componentes necesarios para {threshold*100:.0f}%: {n_comp}")

# pipeline FINAL: Preprocesamiento
to_dense = FunctionTransformer(_to_dense, accept_sparse=True)

pipe_with_pca = Pipeline(steps=[
    ("pre", preprocessor),                              # produce sparse
    ("to_dense", to_dense),                             # densifica solo aquí
    ("pca", PCA(n_components=n_comp, svd_solver="full", random_state=42)),
])

X_train_pca = pipe_with_pca.fit_transform(X_train, y_train)
X_valid_pca = pipe_with_pca.transform(X_valid)

print("Shapes PCA:", X_train_pca.shape, X_valid_pca.shape)


# Balance de clases
import matplotlib.pyplot as plt
import pandas as pd

def to_binary_series(y_series):
    POSITIVE_TOKENS = {"1","p","poisonous","venenoso","yes","true"}
    if pd.api.types.is_numeric_dtype(y_series):
        return (y_series.astype(float) > 0).astype(int)
    y_str = y_series.astype(str).str.lower().str.strip()
    return y_str.apply(lambda v: 1 if v in POSITIVE_TOKENS else 0).astype(int)

y_bin = to_binary_series(y)
counts = y_bin.value_counts().reindex([0,1], fill_value=0)

labels = ["No venenoso (0)", "Venenoso (1)"]
plt.figure(figsize=(5,3))
plt.bar([0,1], counts.values)
plt.xticks([0,1], labels)
plt.title("Balance de clases")
plt.ylabel("Número de muestras")
plt.tight_layout()
plt.show()


import numpy as np

card = X.nunique().sort_values(ascending=False)
top = 20
plt.figure(figsize=(8, max(3, 0.35*min(top, len(card)))))
plt.barh(card.index[:top], card.values[:top])
plt.gca().invert_yaxis()
plt.title(f"Cardinalidad por columna (top {top})")
plt.xlabel("Nº de valores únicos")
plt.tight_layout()
plt.show()


for c in num_cols:
    plt.figure(figsize=(5,3))
    plt.hist(X[c].values, bins=50)
    plt.title(f"Distribución de {c}")
    plt.xlabel(c); plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()



if len(num_cols) >= 2:
    corr = X[num_cols].corr()
    plt.figure(figsize=(6,4))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right')
    plt.yticks(range(len(num_cols)), num_cols)
    plt.colorbar(label="correlación de Pearson")
    plt.title("Matriz de correlación (variables numéricas)")
    plt.tight_layout()
    plt.show()
else:
    print("No hay suficientes columnas numéricas para correlación.")



# Esquema de validación
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# clase positiva (venenoso)
POSITIVE_TOKENS = {"1", "p", "poisonous", "venenoso", "yes", "true"}

def to_binary_series(y_series: pd.Series) -> pd.Series:
    """Convierte y a 0/1 solo para cálculo de proporciones."""
    if pd.api.types.is_numeric_dtype(y_series):
        return (y_series.astype(float) > 0).astype(int)
    y_str = y_series.astype(str).str.lower().str.strip()
    return y_str.apply(lambda v: 1 if v in POSITIVE_TOKENS else 0).astype(int)

y_bin = to_binary_series(y)

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Verificación rápida del equilibrio de clases por fold
fold_proportions = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    prop_train = y_bin.iloc[train_idx].mean()
    prop_val   = y_bin.iloc[val_idx].mean()
    fold_proportions.append((fold, prop_train, prop_val))

print("Proporción de clase positiva (venenoso) por fold:")
for f, tr, va in fold_proportions:
    print(f"Fold {f}: Train={tr:.3f} | Val={va:.3f}")

# distribución global
print("\nDistribución global:")
print(pd.Series({"train/valid folds": len(fold_proportions),
                 "global_positive": y_bin.mean()}))



# MODELO BASE (LightGBM)
import numpy as np
import lightgbm as lgb

# Hiperparámetros
lgb_params = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.1,
    n_estimators=1500,
    max_depth=-1,
    num_leaves=63,
    feature_fraction=0.9,  
    bagging_fraction=0.9,
    bagging_freq=1,
    subsample=0.9,
    colsample_bytree=0.9,
    min_data_in_leaf=2,        
    min_child_samples=1,       
    min_gain_to_split=0.0,    
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

model = lgb.LGBMClassifier(**lgb_params)


# ENTRENAMIENTO Y EVALUACIÓN con PCA
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 'p' (venenoso) -> 1; 'e' (comestible) -> 0
def to_binary_series(y_series):
    import pandas as pd
    POSITIVE_TOKENS = {"1", "p", "poisonous", "venenoso", "yes", "true"}
    if pd.api.types.is_numeric_dtype(y_series):
        return (y_series.astype(float) > 0).astype(int)
    y_str = y_series.astype(str).str.lower().str.strip()
    return y_str.apply(lambda v: 1 if v in POSITIVE_TOKENS else 0).astype(int)

y_train_bin = to_binary_series(y_train)
y_valid_bin = to_binary_series(y_valid)

# Entrenar sobre PCA
model.fit(
    X_train_pca, y_train_bin,
    eval_set=[(X_valid_pca, y_valid_bin)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
)

# 4) Probabilidades y predicciones
valid_proba = model.predict_proba(X_valid_pca)[:, 1]
valid_pred  = (valid_proba >= 0.5).astype(int)

# 5) Métricas
acc = accuracy_score(y_valid_bin, valid_pred)
prec = precision_score(y_valid_bin, valid_pred, zero_division=0)
rec = recall_score(y_valid_bin, valid_pred, zero_division=0)
f1 = f1_score(y_valid_bin, valid_pred, zero_division=0)
auc = roc_auc_score(y_valid_bin, valid_proba)

print("Métricas (VALIDACIÓN, PCA)")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print(f"Best iteration (early stopping): {getattr(model, 'best_iteration_', 'N/A')}")


# === Tabla de métricas promedio y desviación ===
import pandas as pd
import matplotlib.pyplot as plt

metrics_df = pd.DataFrame([{
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "roc_auc": auc
}])

# Calcular promedio y desviación
summary_mean = metrics_df.mean(numeric_only=True).round(4)

summary_table = pd.DataFrame({"mean": summary_mean})
display(summary_table)

# Gráfico de barras comparativo
plt.figure(figsize=(7, 4))
plt.bar(summary_table.index, summary_table["mean"], capsize=4)
plt.title("Desempeño promedio (validación PCA)")
plt.ylabel("Valor de métrica")
plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()



# Matriz de confusión (validación PCA)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_valid_bin, valid_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Comestible (0)", "Venenoso (1)"])

plt.figure(figsize=(4, 4))
disp.plot(cmap="Blues", colorbar=False)
plt.title("Matriz de confusión - Validación PCA")
plt.show()


# ENTRENAMIENTO FINAL
import numpy as np
import pandas as pd
import lightgbm as lgb

# ('p' -> 1, 'e' -> 0)
def to_binary_series(y_series):
    POSITIVE_TOKENS = {"1", "p", "poisonous", "venenoso", "yes", "true"}
    if pd.api.types.is_numeric_dtype(y_series):
        return (y_series.astype(float) > 0).astype(int)
    y_str = y_series.astype(str).str.lower().str.strip()
    return y_str.apply(lambda v: 1 if v in POSITIVE_TOKENS else 0).astype(int)

y_bin_all = to_binary_series(y)

# Ajustar pipeline PCA con todo el conjunto de entrenamiento
pipe_with_pca_full = pipe_with_pca
pipe_with_pca_full.fit(X, y_bin_all)

# Transformar los conjuntos de datos
X_all_pca  = pipe_with_pca_full.transform(X)
X_test_pca = pipe_with_pca_full.transform(test)

# Entrenar el modelo final LightGBM con todos los datos
model_full = lgb.LGBMClassifier(**lgb_params)
model_full.fit(X_all_pca, y_bin_all)

# Generar predicciones en el conjunto de prueba
test_proba = model_full.predict_proba(X_test_pca)[:, 1]
test_pred  = (test_proba >= 0.5).astype(int)

# Construir DataFrame de envío
if 'sample' in globals():
    sub = sample.copy()
    out_col = None
    for c in ('class', 'target', 'label'):
        if c in sub.columns:
            out_col = c
            break
    if out_col is None:
        out_col = 'class'
        if 'id' not in sub.columns:
            sub = pd.DataFrame({'id': np.arange(len(test_pred))})
        sub[out_col] = test_pred
    else:
        sub[out_col] = test_pred
else:
    sub = pd.DataFrame({'id': np.arange(len(test_pred)), 'class': test_pred})

# Guardar archivo CSV
out_path = "submission.csv"
try:
    import os
    if os.path.exists("/kaggle/working"):
        out_path = "/kaggle/working/submission.csv"
except Exception:
    pass

sub.to_csv(out_path, index=False)
print(f"Archivo de envío guardado en: {out_path}")
display(sub.head())



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Distribución de probabilidades predichas
plt.figure(figsize=(6,4))
sns.histplot(test_proba, bins=50, kde=True, color="skyblue")
plt.title("Distribución de probabilidades predichas (P = venenoso)")
plt.xlabel("Probabilidad predicha")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Conteo de clases predichas (0 = comestible, 1 = venenoso)
unique, counts = np.unique(test_pred, return_counts=True)
plt.figure(figsize=(5,3))
plt.bar(["Comestible (0)", "Venenoso (1)"], counts, color=["#66bb6a", "#ef5350"])
plt.title("Distribución de clases predichas")
plt.ylabel("Número de muestras")
plt.tight_layout()
plt.show()


# Boxplot de confianza por clase
plt.figure(figsize=(6,4))
sns.boxplot(x=test_pred, y=test_proba, palette=["#66bb6a", "#ef5350"])
plt.title("Confianza de predicción por clase asignada")
plt.xlabel("Clase predicha (0 = comestible, 1 = venenoso)")
plt.ylabel("Probabilidad asignada por el modelo")
plt.tight_layout()
plt.show()


# Histograma de la zona de incertidumbre (predicciones entre 0.4 y 0.6)
uncertain = (test_proba >= 0.4) & (test_proba <= 0.6)
plt.figure(figsize=(5,3))
plt.hist(test_proba[uncertain], bins=20, color="orange", edgecolor="black")
plt.title(f"Región de incertidumbre (0.4 ≤ P ≤ 0.6): {uncertain.sum()} muestras")
plt.xlabel("Probabilidad predicha")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()


# Resumen textual
print("Resumen:")
print(f"- Total de predicciones: {len(test_proba):,}")
print(f"- Clases predichas: {dict(zip(unique, counts))}")
print(f"- Promedio de probabilidad: {test_proba.mean():.4f}")
print(f"- Predicciones 'inciertas' (0.4 ≤ p ≤ 0.6): {uncertain.sum():,} ({uncertain.mean()*100:.2f}%)")


# Análisis de errores (estimado)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

uncertain_cases = X_test[(test_proba >= 0.4) & (test_proba <= 0.6)].copy()
uncertain_cases["probabilidad"] = test_proba[(test_proba >= 0.4) & (test_proba <= 0.6)]

print(f"Total de casos 'inciertos': {len(uncertain_cases):,} ({len(uncertain_cases)/len(X_test)*100:.3f}%)")

cols_to_plot = ["cap-color", "stem-surface", "gill-color", "ring-type", "habitat"]
cols_to_plot = [c for c in cols_to_plot if c in uncertain_cases.columns][:4]

if cols_to_plot:
    for col in cols_to_plot:
        plt.figure(figsize=(7,3))
        sns.countplot(data=uncertain_cases, x=col, order=uncertain_cases[col].value_counts().index)
        plt.title(f"Distribución de '{col}' en casos inciertos")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Distribución general de probabilidades en casos inciertos
plt.figure(figsize=(6,4))
sns.histplot(uncertain_cases["probabilidad"], bins=30, color="orange")
plt.title("Distribución de probabilidades (casos inciertos)")
plt.xlabel("Probabilidad predicha de 'venenoso'")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()


