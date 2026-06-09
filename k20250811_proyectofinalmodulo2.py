# ManipulaciÃ³n de datos
import pandas as pd
import numpy as np

# VisualizaciÃ³n
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, make_scorer

# Gradient Boosting
from xgboost import XGBClassifier
# import lightgbm as lgb

# Manejo de desbalance
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline


# ------------------------------
# Funciones Ãºtiles
# ------------------------------
def normalized_gini_coefficient(y_true, y_prob):
    """
    Calcula el coeficiente de Gini normalizado
    """

    return 2 * roc_auc_score(y_true, y_prob) - 1

def gini(y_true, y_prob):
    """
    Calcula el coeficiente de Gini
    """
    # Ordenar por probabilidad predicha (descendente)
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calcular curva de Lorenz
    n = len(y_true)
    cumsum_true = np.cumsum(y_true_sorted)
    cumsum_index = np.arange(1, n + 1)
    
    # Gini = 2 * AUC - 1
    # Pero calculamos directamente usando la fÃ³rmula del coeficiente de Gini
    gini_coef = (2 * np.sum(cumsum_true * cumsum_index) / (n * np.sum(y_true))) - (n + 1) / n
    return gini_coef

# Crear scorer personalizado para Gini
gini_scorer = make_scorer(normalized_gini_coefficient, greater_is_better=True, needs_proba=True)


train = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
test  = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')


train.shape, test.shape


train.info()


# DistribuciÃ³n del target
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=train)
plt.title("DistribuciÃ³n de la variable target")
plt.xlabel("Target (0 = No ReclamaciÃ³n, 1 = ReclamaciÃ³n)")
plt.ylabel("Frecuencia")
plt.show()

# Porcentaje de clases
target_counts = train['target'].value_counts(normalize=True) * 100
print("DistribuciÃ³n del target (%):")
print(target_counts)


# Porcentaje de valores faltantes por columna (donde -1 aparece)
missing_pct = (train == -1).sum() / len(train)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

# Mostrar columnas con valores faltantes
print("Columnas con valores faltantes (-1):")
print(missing_pct)

# VisualizaciÃ³n
plt.figure(figsize=(10,6))
missing_pct.plot(kind='bar')
plt.title("Porcentaje de valores faltantes por columna (-1)")
plt.ylabel("Porcentaje")
plt.tight_layout()
plt.show()



# Identificar tipos de caracterÃ­sticas por sufijo
categorical_features = train.columns[train.columns.str.endswith('cat')].tolist()
binary_features = train.columns[train.columns.str.endswith('bin')].tolist()
numeric_features = [col for col in train.columns if col not in categorical_features + binary_features + ['id', 'target']]


# GrÃ¡ficos de caracterÃ­sticas categÃ³ricas vs target
n_rows = len(categorical_features)
fig, axs = plt.subplots(n_rows, 1, figsize=(20,  n_rows * 4))

for ax, col in zip(axs, categorical_features):
    sns.barplot(x=col, y='target', data=train, ax=ax)
    ax.set_title(f'{col} vs target')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# CorrelaciÃ³n entre variables numÃ©ricas
plt.figure(figsize=(40,20))
corr_matrix = train[numeric_features].corr()
sns.heatmap(corr_matrix, annot = True)
plt.plot()


selected_features = ['ps_car_13', 'ps_reg_03', 'ps_ind_15']

for feat in selected_features:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='target', y=feat, data=train)
    plt.title(f"{feat} vs target")
    plt.show()


# Preparar X, y
X = train.drop(['id', 'target'], axis=1)
y = train['target']

# Test set
X_test_raw = test.drop('id', axis=1)
test_ids = test['id']


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['missing_count'] = (df == -1).sum(axis=1)
    if {'ps_car_13','ps_reg_03'}.issubset(df.columns):
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    bin_cols = [c for c in df.columns if c.endswith('_bin')]
    if len(bin_cols):
        df['sum_bin'] = df[bin_cols].replace(-1,0).sum(axis=1)
    for col in ['ps_reg_03','ps_car_12','ps_car_13']:
        if col in df.columns:
            vals = df[col].replace(-1, np.nan)
            try:
                df[col+'_binned'] = pd.qcut(vals, q=10, duplicates='drop').cat.codes
            except:
                pass
    return df

X_eng = create_new_features(X)
X_test_eng = create_new_features(X_test_raw)

# Actualizar listas de features
binary_features = [c for c in X_eng.columns if c.endswith('_bin')]
categorical_features = [c for c in X_eng.columns if c.endswith('_cat') or c.endswith('_binned')]
numeric_features = [c for c in X_eng.columns if c not in binary_features + categorical_features]


numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('bin', binary_transformer, binary_features)
], remainder='drop')


log_reg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Pipeline SIN SMOTE (solo preprocessor + classifier)
pipes = {
    'LogReg': Pipeline([
        ('pre', preprocessor),
        ('classifier', log_reg)
    ]),
    'XGBoost': Pipeline([
        ('pre', preprocessor),
        ('classifier', XGBClassifier(
            eval_metric='logloss', 
            random_state=42, 
            objective='binary:logistic',
            scale_pos_weight=len(y[y==0])/len(y[y==1])
        ))
    ])
} 


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'gini': gini_scorer
}

cv_results = {}
for name, pipe in pipes.items():
    print(f"\nâ€” Modelo: {name}")
    res = cross_validate(pipe, X_eng, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=1)
    cv_results[name] = {k:(np.mean(v), np.std(v)) for k,v in res.items() if k.startswith('test_')}
    for metric, (m,s) in cv_results[name].items():
        print(f"{metric.replace('test_','')}: {m:.4f} Â± {s:.4f}")


# HiperparÃ¡metros a probar
param_grids = {
    'LogReg': {
        'classifier__C': [0.1, 1.0],
        # 'classifier__solver': ['liblinear', 'lbfgs']
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.1, 0.01],
        'classifier__max_depth': [3, 6, 9]
    }
}

# GridSearchCV para ambos modelos
best_models = {}
for name, pipe in pipes.items():
    print(f"\nğŸ”� Optimizando {name}...")
    
    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grids[name],
        scoring=gini_scorer,
        cv=skf,
        n_jobs=1,
        verbose=2
    )

    print(f"Entrenando modelo {name}...")
    grid_search.fit(X_eng, y)
    
    print(f"âœ… Mejores hiperparÃ¡metros para {name}:")
    print(grid_search.best_params_)
    print(f"âœ… Mejor Gini Score: {grid_search.best_score_:.4f}")
    print(f"âœ… ROC AUC equivalente: {(grid_search.best_score_ + 1) / 2:.4f}")
    
    best_models[name] = grid_search.best_estimator_


# 1. Train/test split para evaluaciÃ³n final (hold-out)
X_tr, X_va, y_tr, y_va = train_test_split(X_eng, y, test_size=0.2, stratify=y, random_state=42)

# 2. Construir pipeline final con el mejor modelo completo (preprocesamiento + clasificador)
final_pipeline = grid_search.best_estimator_

# 3. Entrenar en train (split)
final_pipeline.fit(X_tr, y_tr)

# 4. Predecir en validaciÃ³n
proba_va = final_pipeline.predict_proba(X_va)[:, 1]
pred_va = (proba_va > 0.5).astype(int)

# 5. Calcular mÃ©tricas
auc_va = roc_auc_score(y_va, proba_va)
gini_va = 2 * auc_va - 1

print(f"ROC-AUC (hold-out): {auc_va:.6f}")
print(f"Gini (hold-out): {gini_va:.6f}")
print("Confusion Matrix:")
print(confusion_matrix(y_va, pred_va))
print("\nClassification Report:")
print(classification_report(y_va, pred_va, digits=4))

# 6. Entrenar pipeline final en todo el dataset (train + validaciÃ³n)
final_pipeline.fit(X_eng, y)

# 7. Aplicar feature engineering al test set
X_test_eng = create_new_features(X_test_raw)

# 8. Predecir probabilidades para test set (usando pipeline completo)
proba_test = final_pipeline.predict_proba(X_test_eng)[:, 1]

# 9. Crear submission dataframe (ajusta 'test_ids' si tu columna id tiene otro nombre)
submission = pd.DataFrame({'id': test_ids, 'target': proba_test})

# 10. Guardar archivo CSV submission
submission.to_csv('submission.csv', index=False)
print("\nArchivo submission.csv guardado con Ã©xito.")
print(submission['target'].describe())

