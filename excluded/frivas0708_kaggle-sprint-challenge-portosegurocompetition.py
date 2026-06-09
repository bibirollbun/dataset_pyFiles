# Kaggle Porto Seguro Safe Driver Prediction â€” versiÃ³n optimizada y limpia
# Incluye: EDA, Feature Engineering, Pipelines, 2+ modelos, Stratified K-Fold, Gini scorer,
# GridSearchCV para el mejor modelo, evaluaciÃ³n final y submission.
# Optimizado: menos uso de CPU, reducciÃ³n de datos opcional, warnings eliminados.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')  # <--- elimina todos los warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# ------------------------------
# ConfiguraciÃ³n visual
# ------------------------------
plt.style.use('ggplot')
sns.set_palette("Set2")
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 120)

# ------------------------------
# Funciones Ãºtiles
# ------------------------------
def normalized_gini_coefficient(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1

gini_scorer = make_scorer(normalized_gini_coefficient, greater_is_better=True, needs_proba=True)

def safe_print(title, obj):
    print("\n" + "="*20 + f" {title} " + "="*20)
    print(obj)

def ensure_cols(df, cols):
    return [c for c in cols if c in df.columns]

# ------------------------------
# Cargar datos
# ------------------------------
print("Loading data...")
DATA_DIR = Path('/kaggle/input/porto-seguro-safe-driver-prediction')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

# ------------------------------
# Reducir tamaÃ±o de datos para menos CPU (opcional)
# ------------------------------
SUBSAMPLE_FRAC = 0.25  # usa 25% de los datos para pruebas rÃ¡pidas
train = train.sample(frac=SUBSAMPLE_FRAC, random_state=42).reset_index(drop=True)

safe_print('Train shape', train.shape)
safe_print('Test shape', test.shape)

# ------------------------------
# Separar variables y target
# ------------------------------
X = train.drop(['id','target'], axis=1)
y = train['target']
test_ids = test['id']
X_test_raw = test.drop('id', axis=1)

# ------------------------------
# Feature Engineering
# ------------------------------
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

# ------------------------------
# Pipelines
# ------------------------------
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # <--- sin warning
])

binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('bin', binary_transformer, binary_features)
], remainder='drop')

# ------------------------------
# Modelos
# ------------------------------
log_reg = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=2)
rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=2, class_weight='balanced', random_state=42)
lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=2,
    verbose=-1  # <--- silencia LightGBM
)

pipes = {
    'LogReg': Pipeline([('pre', preprocessor), ('model', log_reg)]),
    'RandomForest': Pipeline([('pre', preprocessor), ('model', rf)]),
    'LightGBM': Pipeline([('pre', preprocessor), ('model', lgbm)])
}

# ------------------------------
# ValidaciÃ³n Cruzada (Stratified K-Fold)
# ------------------------------
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # <--- 3 folds para menos CPU

scoring = {
    'roc_auc':'roc_auc',
    'gini':gini_scorer,
    'precision':'precision',
    'recall':'recall',
    'f1':'f1'
}

cv_results = {}
for name, pipe in pipes.items():
    print(f"\nâ€” Modelo: {name}")
    res = cross_validate(pipe, X_eng, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=2)
    cv_results[name] = {k:(np.mean(v), np.std(v)) for k,v in res.items() if k.startswith('test_')}
    for metric, (m,s) in cv_results[name].items():
        print(f"{metric.replace('test_','')}: {m:.4f} Â± {s:.4f}")

# Mejor modelo por Gini
best_by_gini = max(cv_results.items(), key=lambda kv: kv[1]['test_gini'][0])[0]
print(f"\n>>> Mejor modelo por Gini: {best_by_gini}")

# ------------------------------
# Grid Search HiperparÃ¡metros
# ------------------------------
if best_by_gini == 'LightGBM':
    grid = {
        'model__n_estimators':[200,300],
        'model__learning_rate':[0.05,0.1],
        'model__num_leaves':[31,63]
    }
elif best_by_gini == 'RandomForest':
    grid = {
        'model__n_estimators':[100,200],
        'model__max_depth':[None,15]
    }
else:  # LogReg
    grid = {'model__C':[0.1,1.0]}

best_pipe = pipes[best_by_gini]

grid_search = GridSearchCV(best_pipe, param_grid=grid, scoring=gini_scorer, cv=skf, n_jobs=2, verbose=1)
grid_search.fit(X_eng, y)
print(f"Mejores hiperparÃ¡metros: {grid_search.best_params_}")
print(f"Mejor Gini (CV): {grid_search.best_score_:.6f}")

final_pipeline = grid_search.best_estimator_

# ------------------------------
# EvaluaciÃ³n hold-out
# ------------------------------
X_tr, X_va, y_tr, y_va = train_test_split(X_eng, y, test_size=0.2, stratify=y, random_state=42)
final_pipeline.fit(X_tr, y_tr)
proba_va = final_pipeline.predict_proba(X_va)[:,1]
pred_va = (proba_va>0.5).astype(int)
auc_va = roc_auc_score(y_va, proba_va)
gini_va = 2*auc_va-1

safe_print('ROC-AUC (hold-out)', f"{auc_va:.6f}")
safe_print('Gini (hold-out)', f"{gini_va:.6f}")
safe_print('Confusion Matrix', confusion_matrix(y_va, pred_va))
safe_print('Classification Report', classification_report(y_va, pred_va, digits=4))

# ------------------------------
# Entrenamiento final y Submission
# ------------------------------
final_pipeline.fit(X_eng, y)
proba_test = final_pipeline.predict_proba(X_test_eng)[:,1]
submission = pd.DataFrame({'id': test_ids, 'target': proba_test})
SUB_PATH = Path('/kaggle/working/submission.csv')
submission.to_csv(SUB_PATH, index=False)
safe_print('Submission guardado en', str(SUB_PATH))
safe_print('Pred stats', submission['target'].describe())

print("\nÂ¡Listo! Sube submission.csv a Kaggle. ðŸš€")




