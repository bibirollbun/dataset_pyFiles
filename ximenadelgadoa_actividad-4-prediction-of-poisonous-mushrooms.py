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

import lightgbm as lgb  # baseline recomendado

SEED = 42
np.random.seed(SEED); random.seed(SEED)


#Carga de datos
train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")

train.shape, test.shape, sample.shape


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


#Preprocesamiento

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

#Revisión de nulos y cardinalidad
nulls_train = X.isna().sum().sort_values(ascending=False)
nulls_test  = X_test.isna().sum().sort_values(ascending=False)
print("Nulos en train (top):\n", nulls_train.head(10), "\n")
print("Nulos en test  (top):\n", nulls_test.head(10), "\n")

card_train = X.nunique().sort_values(ascending=False)
print("Cardinalidad (n° de valores únicos) - top:\n", card_train.head(10), "\n")

#Eliminar columnas constantes
const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if const_cols:
    print("Eliminando columnas constantes:", const_cols)
    X.drop(columns=const_cols, inplace=True)
    X_test.drop(columns=const_cols, inplace=True)
    cat_cols = [c for c in cat_cols if c not in const_cols]
    num_cols = [c for c in num_cols if c not in const_cols]

#Imputación de faltantes
X[cat_cols] = X[cat_cols].fillna("missing")
X_test[cat_cols] = X_test[cat_cols].fillna("missing")

medianas = X[num_cols].median(numeric_only=True)
X[num_cols] = X[num_cols].fillna(medianas)
X_test[num_cols] = X_test[num_cols].fillna(medianas)

#Tipos consistentes
for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

#Label Encoding consistente
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    vals = pd.concat([X[c], X_test[c]], axis=0)
    le.fit(vals)
    X[c] = le.transform(X[c])
    X_test[c] = le.transform(X_test[c])
    encoders[c] = le

#comprobaciones finales
print(f"Shape X: {X.shape} | X_test: {X_test.shape}")
print("Tipos después del encoding (muestra):")
print(X.dtypes.head())



import matplotlib.pyplot as plt
import numpy as np

vals = np.bincount(y.astype(int))
labels = ["No venenoso (0)", "Venenoso (1)"]

plt.figure(figsize=(5,3))
plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), labels, rotation=0)
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



#Esquema de validación

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Verificación rápida del equilibrio de clases
fold_proportions = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    prop_train = y_train_fold.mean()
    prop_val   = y_val_fold.mean()
    fold_proportions.append((fold, prop_train, prop_val))

# Mostrar proporciones
print("Proporción de clase 1 (venenoso) por fold:")
for f, tr, va in fold_proportions:
    print(f"Fold {f}: Train={tr:.3f} | Val={va:.3f}")



#Modelo base (LightGBM)

import lightgbm as lgb

# Hiperparámetros iniciales (baseline)
lgb_params = dict(
    objective="binary",          # problema de clasificación binaria
    metric="auc",                # métrica interna
    boosting_type="gbdt",        # Gradient Boosting clásico
    learning_rate=0.05,          # velocidad de aprendizaje moderada
    num_leaves=48,               # complejidad de los árboles
    max_depth=-1,                
    n_estimators=300,           # máximo de árboles
    feature_fraction=0.7,        # fracción de features usadas por árbol
    bagging_fraction=0.7,        # fracción de muestras usadas
    bagging_freq=1,              
    verbose=-1,
    random_state=42,
    n_jobs=-1                    # usar todos los núcleos disponibles
)

print("Parámetros base del modelo:")
for k, v in lgb_params.items():
    print(f"  {k}: {v}")


#Entrenamiento y Evaluación

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import numpy as np
import pandas as pd
import lightgbm as lgb

THRESHOLD = 0.5 

oof_proba = np.zeros(len(X))       
oof_pred  = np.zeros(len(X), int)   
fold_metrics = []                   
cm_sum = np.zeros((2, 2), dtype=int) 

fprs, tprs, aucs = [], [], []      

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Probabilidades y predicciones del fold
    va_proba = model.predict_proba(X_va)[:, 1]
    va_pred  = (va_proba >= THRESHOLD).astype(int)

    # Guardar OOF
    oof_proba[va_idx] = va_proba
    oof_pred[va_idx]  = va_pred

    # Métricas del fold
    acc  = accuracy_score(y_va, va_pred)
    prec = precision_score(y_va, va_pred, zero_division=0)
    rec  = recall_score(y_va, va_pred, zero_division=0)
    f1   = f1_score(y_va, va_pred, zero_division=0)
    auc  = roc_auc_score(y_va, va_proba)

    fold_metrics.append({
        "fold": fold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "best_iteration_": getattr(model, "best_iteration_", None)
    })

    # Matriz de confusión acumulada
    cm_sum += confusion_matrix(y_va, va_pred, labels=[0, 1])

    # Datos para curva ROC del fold
    fpr, tpr, _ = roc_curve(y_va, va_proba)
    fprs.append(fpr); tprs.append(tpr); aucs.append(auc)


# DataFrame con métricas por fold + resumen
metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
display(metrics_df.style.format("{:.4f}"))

print("\n=== Promedio (10 folds) ===")
print(metrics_df.mean(numeric_only=True).round(4))

print("\n=== Desviación estándar (10 folds) ===")
print(metrics_df.std(numeric_only=True).round(4))

# Métricas OOF globales (con el mismo THRESHOLD)
oof_acc  = accuracy_score(y, oof_pred)
oof_prec = precision_score(y, oof_pred, zero_division=0)
oof_rec  = recall_score(y, oof_pred, zero_division=0)
oof_f1   = f1_score(y, oof_pred, zero_division=0)
oof_auc  = roc_auc_score(y, oof_proba)

print("\n=== Métricas OOF (global) ===")
print({
    "accuracy": round(oof_acc, 4),
    "precision": round(oof_prec, 4),
    "recall": round(oof_rec, 4),
    "f1": round(oof_f1, 4),
    "roc_auc": round(oof_auc, 4)
})


#Tabla de métricas promedio y desviación
import pandas as pd
import matplotlib.pyplot as plt

summary_mean = metrics_df.mean(numeric_only=True).round(4)
summary_std  = metrics_df.std(numeric_only=True).round(4)
summary_table = pd.DataFrame({"mean": summary_mean, "std": summary_std})
display(summary_table)

# Gráfico de barras comparativo
plt.figure(figsize=(7,4))
plt.bar(summary_table.index, summary_table["mean"], yerr=summary_table["std"], capsize=4)
plt.title("Desempeño promedio (5 folds)")
plt.ylabel("Valor de métrica")
plt.ylim(0.98, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# Matriz de confusión acumulada
import matplotlib.pyplot as plt
import numpy as np

classes = ["No venenoso (0)", "Venenoso (1)"]

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm_sum, interpolation="nearest")

# Etiquetas y formato
ax.set_title("Matriz de confusión (acumulada en validación cruzada)")
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=20, ha="right")
ax.set_yticklabels(classes)
ax.set_xlabel("Predicción del modelo")
ax.set_ylabel("Clase real")

# Números en cada celda
for i in range(cm_sum.shape[0]):
    for j in range(cm_sum.shape[1]):
        ax.text(j, i, format(cm_sum[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_sum[i, j] > cm_sum.max()/2 else "black")

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Tasas de error por clase
FP = cm_sum[0,1]  # falsos positivos
FN = cm_sum[1,0]  # falsos negativos
TP = cm_sum[1,1]
TN = cm_sum[0,0]

total = cm_sum.sum()
print(f"Total muestras: {total:,}")
print(f"Falsos positivos: {FP:,} ({FP/total:.4%})")
print(f"Falsos negativos: {FN:,} ({FN/total:.4%})")
print(f"Verdaderos positivos: {TP:,}")
print(f"Verdaderos negativos: {TN:,}")


# Curva ROC por fold
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6,5))

# Graficar las curvas ROC individuales
for fpr, tpr in zip(fprs, tprs):
    plt.plot(fpr, tpr, alpha=0.2, color="gray")

# Promedio de AUC y referencia diagonal
mean_auc = np.mean(aucs)
plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Aleatorio (AUC=0.5)")
plt.title(f"Curvas ROC por fold (AUC promedio = {mean_auc:.4f})")
plt.xlabel("False Positive Rate (1 - Especificidad)")
plt.ylabel("True Positive Rate (Sensibilidad)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(y, oof_proba)
f1_scores = 2 * (prec * rec) / (prec + rec)
best_idx = np.argmax(f1_scores)
print(f"Umbral óptimo para F1 = {thr[best_idx]:.3f} | F1={f1_scores[best_idx]:.4f}")

plt.figure(figsize=(6,4))
plt.plot(thr, prec[:-1], label="Precision")
plt.plot(thr, rec[:-1], label="Recall")
plt.plot(thr, f1_scores[:-1], label="F1", linestyle="--")
plt.axvline(thr[best_idx], color="red", linestyle="--", label=f"Umbral óptimo={thr[best_idx]:.3f}")
plt.title("Precision, Recall y F1 según el umbral")
plt.xlabel("Umbral de decisión")
plt.ylabel("Valor de métrica")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



#Entrenamiento final con todos los datos
import lightgbm as lgb
import pandas as pd

# Copia de los mejores parámetros
lgb_params_final = dict(lgb_params)
lgb_params_final.update({
    "n_estimators": 300,
    "num_leaves": 48,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "learning_rate": 0.05,
    "random_state": 42
})

#Entrenamiento completo 
final_model = lgb.LGBMClassifier(**lgb_params_final)
final_model.fit(X, y, categorical_feature=cat_cols)

#Predicciones sobre el conjunto de test
test_proba = final_model.predict_proba(X_test)[:, 1]

#Umbral para convertir probabilidad → clase
THRESHOLD_FINAL = 0.5
test_pred = (test_proba >= THRESHOLD_FINAL).astype(int)

#Crear DataFrame
submission = pd.DataFrame({
    "id": test["id"],
    "class": test_pred
})

#Guardar CSV 
submission.to_csv("submission.csv", index=False)
print("Archivo 'submission.csv' creado correctamente")
print(submission.head())



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


# Importancia de características
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Obtener importancias desde el modelo final
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values(by="importance", ascending=False)

# Mostrar top 20
top_n = 20
plt.figure(figsize=(8, max(4, 0.35 * top_n)))
plt.barh(importance_df["feature"][:top_n][::-1], importance_df["importance"][:top_n][::-1])
plt.title(f"Importancia de variables (top {top_n}) — LightGBM")
plt.xlabel("Importancia relativa (ganancia total)")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# Mostrar tabla resumen
display(importance_df.head(20).style.background_gradient(cmap="YlGnBu"))



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


