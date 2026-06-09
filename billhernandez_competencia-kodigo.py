import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
%matplotlib inline



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, make_scorer, precision_score



df_train = pd.read_csv("/kaggle/input/porto-seguro-safe-driver-prediction/train.csv")
df_test  = pd.read_csv("/kaggle/input/porto-seguro-safe-driver-prediction/test.csv")
df_sample = pd.read_csv("/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv")



df_train.info()


df_train.head()


df_train.describe().T


df_train['target'].value_counts()


df_train['target'].value_counts(normalize=True)


plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df_train)
plt.title('Distribución de target (counts)')
plt.show()


menos1 = (df_train == -1).sum().sort_values(ascending=False)
menos1 = menos1[menos1 > 0]  # solo columnas con -1
df_menos1= pd.DataFrame({
    'conteo_menos1': menos1,
    'porcentaje_menos1': (menos1 / len(df_train) * 100)
})
display(df_menos1.head(40))


plt.figure(figsize=(10,6))
missing_series = (df_train == -1).sum() / len(df_train)
missing_series = missing_series[missing_series>0].sort_values(ascending=False)
missing_series.head(30).plot.bar()
plt.ylabel('Fracción de filas con -1')
plt.title('Fracción de -1 (missing codificado) por feature (top 30)')
plt.show()


train_na = df_train.replace(-1, np.nan)


num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ['id','target']]
print("Número de columnas numéricas (excluyendo id/target):", len(num_cols))


plt.figure(figsize=(14,10))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(3,2,i)
    sns.histplot(train_na[col].dropna(), bins=40)
    plt.title(col)
plt.tight_layout()
plt.show()


cat_cols = [c for c in train.columns if c.endswith('_cat')]
low_card_cols = [c for c in train.columns if train[c].nunique() <= 10 and c not in ['id','target'] and c not in cat_cols]
cat_cols = list(dict.fromkeys(cat_cols + low_card_cols))  # unir sin duplicados
print("Columnas categóricas detectadas:", cat_cols)


for c in cat_cols[:6]:
    display(pd.crosstab(train[c], train['target'], margins=False))
    plt.figure(figsize=(6,3))
    sns.countplot(x=c, data=train, order=sorted(train[c].dropna().unique()))
    plt.title(f"Count of {c}")
    plt.show()



corr = df_train.corr()
corr_target = corr['target'].drop('target').sort_values(key=lambda x: x.abs(), ascending=False)
display(corr_target.head(30))



plt.figure(figsize=(8,10))
sns.barplot(x=corr_target.abs().values[:20], y=corr_target.index[:20])
plt.title('Correlación absoluta con target (top 20)')
plt.xlabel('abs(corr)')
plt.show()


top_feats = corr_target.index[:20].tolist()
plt.figure(figsize=(12,10))
sns.heatmap(train[top_feats + ['target']].corr(), center=0, cmap='coolwarm')
plt.title('Heatmap correlaciones (top 20 con target)')
plt.show()


top_num = [c for c in corr_target.index if c in num_cols][:3]
print("Top num features para graficar vs target:", top_num)

for f in top_num:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='target', y=f, data=train_na)
    plt.title(f'Boxplot de {f} por target')
    plt.show()

    plt.figure(figsize=(8,4))
    sns.kdeplot(data=train_na, x=f, hue='target', fill=True, common_norm=False)
    plt.title(f'Distribución de {f} separada por target')
    plt.show()

#  - Para categóricas: mostrar mean(target) por categoría (target rate)
cat_for_plot = cat_cols[:3]  # tomar hasta 3 categóricas
print("Categorías para graficar target rate:", cat_for_plot)
for c in cat_for_plot:
    plt.figure(figsize=(8,4))
    grp = train.groupby(c)['target'].mean().sort_index()
    grp.plot(kind='bar')
    plt.ylabel('Mean target (claim rate)')
    plt.title(f'Target rate by {c}')
    plt.show()




X = df_train.drop(columns=['target'])
y = df_train['target']


df = df_train.copy()

# Separar variables
num_cols = [c for c in df.columns if '_cat' not in c and c not in ['id', 'target']]
cat_cols = [c for c in df.columns if '_cat' in c]

# Imputar numéricas (-1) con la mediana
for col in num_cols:
    df[col] = df[col].replace(-1, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Imputar categóricas (-1) con la moda
for col in cat_cols:
    df[col] = df[col].replace(-1, np.nan)
    df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Imputación completada.")


train['missing_count'] = (train == -1).sum(axis=1)
df['missing_count'] = train['missing_count']


scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print("✅ Escalado completado para variables numéricas.")


skewness = df[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
print("Top variables más sesgadas:\n", skewness.head(10))



for col in num_cols:
    if abs(df[col].skew()) > 1:
        df[col] = np.log1p(df[col])  # log(1 + x)



# Todas las columnas categóricas que existen
cat_cols = [c for c in df.columns if '_cat' in c]
print("Columnas categóricas detectadas:", cat_cols)



# Umbral: baja cardinalidad ≤10 categorías
low_cardinality = [c for c in cat_cols if df[c].nunique() <= 10]
high_cardinality = [c for c in cat_cols if df[c].nunique() > 10]

print("Baja cardinalidad:", low_cardinality)
print("Alta cardinalidad:", high_cardinality)




df = pd.get_dummies(df, columns=low_cardinality, drop_first=True)
print("✅ One-Hot Encoding completado para baja cardinalidad.")





le = LabelEncoder()
for c in high_cardinality:
    df[c] = le.fit_transform(df[c])
print("✅ Label Encoding completado para alta cardinalidad.")




def target_encode_kfold(train_series, target, test_series=None, n_splits=5, smoothing=1.0):
    train_series = train_series.fillna("##MISSING##").astype(str)
    prior = target.mean()
    oof = pd.Series(index=train_series.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(train_series):
        tr_s = train_series.iloc[train_idx]
        tr_y = target.iloc[train_idx]
        stats = tr_y.groupby(tr_s).agg(['sum','count'])
        means = stats['sum'] / stats['count']
        counts = stats['count']
        smooth = (counts * means + smoothing * prior) / (counts + smoothing)
        oof.iloc[val_idx] = train_series.iloc[val_idx].map(smooth).fillna(prior)
    
    test_encoded = None
    if test_series is not None:
        test_series = test_series.fillna("##MISSING##").astype(str)
        train_stats = target.groupby(train_series).agg(['sum','count'])
        means = train_stats['sum'] / train_stats['count']
        counts = train_stats['count']
        smooth = (counts * means + smoothing * prior) / (counts + smoothing)
        test_encoded = test_series.map(smooth).fillna(prior).astype(float)
    
    return oof.astype(float), test_encoded



num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])



cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])



preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)



pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(solver='liblinear', random_state=42))
])


pipeline


pipeline.fit(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


models = {
    "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}



results = []

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
    
    # Entrenar
    pipe.fit(X_train, y_train)
    
    # Predecir en test
    y_pred = pipe.predict(X_test)
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    
    results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC-AUC": roc})
    
# Mostrar resultados
results_df = pd.DataFrame(results)
print(results_df)



def gini(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1

# Crear scorer para cross_validate
gini_scorer = make_scorer(gini, needs_proba=True)



models = {
    "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



cv_results = {}

scoring = {
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'gini': gini_scorer
}

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
    
    scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    cv_results[name] = {metric: np.mean(scores['test_' + metric]) for metric in scoring.keys()}

# Mostrar resultados
cv_df = pd.DataFrame(cv_results).T
print(cv_df)



pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])


param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.1, 0.01],
    'classifier__max_depth': [3, 6, 9]
}



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



from sklearn.metrics import make_scorer, roc_auc_score

def gini(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1

gini_scorer = make_scorer(gini, needs_proba=True)



grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=gini_scorer,
    cv=skf,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X, y)



print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor Gini:", grid_search.best_score_)






