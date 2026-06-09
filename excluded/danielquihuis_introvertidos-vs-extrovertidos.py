import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # suprime FutureWarning repetidos
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support

RND = 42  # reproducibilidad

print("scikit-learn version:", sklearn.__version__)

# 1) Cargar datos (archivos ya presentes en el entorno: train.csv, test.csv, sample_submission.csv)
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())

# 2) EDA rápido (tipos, nulos, estadísticas)
print("\n--- info ---")
print(train.info())
print("\n--- nulos por columna ---")
print(train.isnull().sum())
print("\n--- estadisticas numéricas ---")
display(train.describe())

# 3) Definir target, id y features: QUITAR 'id' de las features
TARGET = 'Personality'
ID_COL = 'id'

# Comprobar columnas
all_cols = train.columns.tolist()
print("\nColumnas:", all_cols)

# Construir X quitando la columna id y target
X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET].copy()

# Para test, quitar id pero guardarlo para submission
X_test = test.drop(columns=[ID_COL]).copy()
test_ids = test[ID_COL].copy()

# Detectar numéricas y categóricas automáticamente (después de quitar id)
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print("\nNuméricas:", numeric_cols)
print("Categóricas:", categorical_cols)

# 4) Preprocesamiento (imputación + escalado + one-hot). Usar median para num (más robusto)
#    Construyo OneHotEncoder con compatibilidad para versiones antiguas/nuevas de sklearn
try:
    # scikit-learn >= 1.2: parameter name sparse_output
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    # versiones antiguas: sparse
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # median evita sesgo por outliers
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # moda para categorías faltantes
    ('onehot', ohe)
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], remainder='drop')  # drop any other columns

# Probar preprocessor y revisar shape
X_pre = preprocessor.fit_transform(X)
print("\nPreprocessed train shape:", X_pre.shape)
# Nota: X_pre es NumPy array listo para PCA/modelo

# 5) PCA (visualización) - aplicar sobre X_pre
pca = PCA(n_components=2, random_state=RND)
X_pca = pca.fit_transform(X_pre)
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(7,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, alpha=0.6)
plt.title("PCA (2 componentes) - coloreado por Personality")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

# 6) t-SNE (muestra para velocidad). Si falla, lo saltamos
try:
    sample_frac = 0.25  # reduce si quieres más rapidez
    sample_idx = train.sample(frac=sample_frac, random_state=RND).index
    X_sample = X.loc[sample_idx]
    y_sample = y.loc[sample_idx]
    X_sample_pre = preprocessor.transform(X_sample)

    tsne = TSNE(n_components=2, random_state=RND, perplexity=30, n_iter=800, init='pca')
    t0 = time.time()
    X_tsne = tsne.fit_transform(X_sample_pre)
    print("t-SNE time:", round(time.time() - t0, 2), "s")

    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_sample, alpha=0.6)
    plt.title("t-SNE (muestra)")
    plt.show()
except Exception as e:
    print("t-SNE omitido o muy lento:", e)

# 7) Decisión: clasificación (target categórico)
print("\nDECISIÓN: problema de CLASIFICACIÓN porque 'Personality' es categórico (Extrovert/Introvert).")

# 8) Baseline: Logistic Regression (pipeline)
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, random_state=RND, solver='lbfgs'))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
scores = cross_val_score(baseline_pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
print("\nBaseline LogisticRegression CV accuracy (5-fold):", scores, "mean:", scores.mean())

# Holdout split para inspección adicional
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RND)
baseline_pipeline.fit(X_train, y_train)
y_pred_base = baseline_pipeline.predict(X_val)
print("Baseline holdout accuracy:", accuracy_score(y_val, y_pred_base))
print(classification_report(y_val, y_pred_base))

# 9) Modelos candidatos y RandomizedSearchCV (RF y GradientBoosting)
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(random_state=RND, n_jobs=-1))
])

gb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', GradientBoostingClassifier(random_state=RND))
])

rf_param_dist = {
    'clf__n_estimators': [100, 200, 400],
    'clf__max_depth': [None, 6, 12],
    'clf__min_samples_split': [2, 5, 10]
}

gb_param_dist = {
    'clf__n_estimators': [50, 100, 200],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5, 8]
}

print("\nIniciando RandomizedSearchCV para RandomForest (esto puede tardar unos minutos)...")
rf_search = RandomizedSearchCV(
    rf_pipe, rf_param_dist, n_iter=8, cv=3, scoring='accuracy', random_state=RND, n_jobs=-1, verbose=1
)
rf_search.fit(X_train, y_train)
print("RF best params:", rf_search.best_params_, "best score:", rf_search.best_score_)

print("\nIniciando RandomizedSearchCV para GradientBoosting ...")
gb_search = RandomizedSearchCV(
    gb_pipe, gb_param_dist, n_iter=8, cv=3, scoring='accuracy', random_state=RND, n_jobs=-1, verbose=1
)
gb_search.fit(X_train, y_train)
print("GB best params:", gb_search.best_params_, "best score:", gb_search.best_score_)

# Selección del mejor por CV score
if rf_search.best_score_ >= gb_search.best_score_:
    best_search = rf_search
    best_name = 'RandomForest'
else:
    best_search = gb_search
    best_name = 'GradientBoosting'

print("\nModelo seleccionado:", best_name)

# 10) Evaluación en holdout con el mejor modelo
best_model = best_search.best_estimator_
y_val_pred = best_model.predict(X_val)

print("\nHoldout accuracy (best):", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (holdout)')
plt.show()

# ROC (binarizar target). Usamos LabelBinarizer para convertir a 0/1
lb = LabelBinarizer()
y_val_bin = lb.fit_transform(y_val).ravel()
if hasattr(best_model, "predict_proba"):
    y_val_proba = best_model.predict_proba(X_val)[:,1]
    fpr, tpr, _ = roc_curve(y_val_bin, y_val_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (holdout)')
    plt.legend()
    plt.show()
else:
    print("El modelo no provee predict_proba; no se puede dibujar ROC.")

# 11) Learning curve (curva de aprendizaje)
train_sizes, train_scores, val_scores = learning_curve(best_model, X, y, cv=skf, scoring='accuracy', n_jobs=-1,
                                                       train_sizes=np.linspace(0.1,1.0,5))
train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(6,4))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Train score')
plt.plot(train_sizes, val_scores_mean, 'o-', label='CV score')
plt.xlabel('Train size'); plt.ylabel('Accuracy'); plt.title('Learning Curve (best model)')
plt.legend(); plt.grid(True)
plt.show()

# 12) Cross-validation final (k-fold) con el mejor modelo
cv_scores = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
print("\nCross-val scores (5 folds):", cv_scores, "mean:", cv_scores.mean())

# Métricas promedio (precision, recall, f1) por CV (Extrovert como pos_label)
precisions, recalls, f1s = [], [], []
for train_idx, val_idx in skf.split(X, y):
    X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]
    best_model.fit(X_tr, y_tr)
    yv_pred = best_model.predict(X_v)
    p, r, f, _ = precision_recall_fscore_support(y_v, yv_pred, average='binary', pos_label='Extrovert')
    precisions.append(p); recalls.append(r); f1s.append(f)
print(f"\nCV Precision (Extrovert): mean {np.mean(precisions):.3f}, Recall mean {np.mean(recalls):.3f}, F1 mean {np.mean(f1s):.3f}")

# 13) Re-entrenar con todo el set de train y predecir test para submission
best_model.fit(X, y)
test_preds = best_model.predict(X_test)

submission = pd.DataFrame({'id': test_ids, TARGET: test_preds})
submission.to_csv('submission.csv', index=False)
print("\nSubmission creada: submission.csv (lista para subir).")

# 14) Resumen final y recomendaciones
print("\n--- RESUMEN ---")
print(f"Modelo seleccionado: {best_name}")
print("Decisión: Clasificación (target categórico).")
print("Preprocessing: imputación (median para num, most_frequent para cat), escalado, one-hot encoding; todo en pipeline.")
print("Evaluaciones incluidas: baseline (LogisticRegression), búsqueda de hiperparámetros RandomizedSearchCV, CV k-fold, ROC, learning curve, confusion matrix.")
print("Acciones importantes realizadas: 'id' eliminado de features (evita leakage), OneHotEncoder configurado con compatibilidad y warnings suprimidos.")
print("Archivo submission.csv listo")


