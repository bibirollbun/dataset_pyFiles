import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from scipy.stats import randint, uniform, loguniform

try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

import joblib

from sklearn.base import clone


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


print('train shape:', train.shape)
print('test shape:', test.shape)


train.columns.tolist()


train.head()


print('Train missing per column:')
print(train.isna().sum())
print('Test missing per column:')
print(test.isna().sum())


print('Target distribution:')
print(train['loan_paid_back'].value_counts())
print('Target balance (proportion):')
print(train['loan_paid_back'].value_counts(normalize=True))


id_col = 'id'
target_col = 'loan_paid_back'
feature_cols = [c for c in train.columns if c not in [id_col, target_col]]
num_cols = train[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]
print('Numeric columns:', num_cols)
print('Categorical columns:', cat_cols)


for c in cat_cols:
    print(c, '->', train[c].nunique())


train[num_cols].describe().T


for c in num_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(train[c], kde=False, bins=50)
    plt.title(c)
    plt.tight_layout()
    plt.show()


for c in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=target_col, y=c, data=train.sample(30000, random_state=42))
    plt.title(f'{c} by {target_col}')
    plt.tight_layout()
    plt.show()


for c in cat_cols:
    print('Column:', c)
    print(train[c].value_counts().head(10))


cat_example = 'loan_purpose'
if cat_example in cat_cols:
    temp = train.groupby(cat_example)[target_col].agg(['mean','count']).sort_values('count', ascending=False)
    display(temp.head(20))


X = train[feature_cols]
y = train[target_col]


num_pipeline = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
cat_pipeline = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])


pipelines = {
    'LogisticRegression': Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000, solver='saga'))]),
    'RandomForest': Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))]),
    'LightGBM': Pipeline([('pre', preprocessor),('clf', lgb.LGBMClassifier(n_estimators=200, random_state=42))]),
    'XGBoost': Pipeline([('pre', preprocessor),('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))]),
    'CatBoost': Pipeline([('pre', preprocessor),('clf', CatBoostClassifier(verbose=0, random_state=42))])
}


FAST = True
SAMPLE_SIZE = 100000
if FAST and len(X) > SAMPLE_SIZE:
    sample_idx = np.random.choice(X.index, size=SAMPLE_SIZE, replace=False)
    X_fast = X.loc[sample_idx]
    y_fast = y.loc[sample_idx]
else:
    X_fast = X
    y_fast = y

results = {}
for name, pipe in pipelines.items():
    print('Scoring', name)
    try:
        scores = cross_val_score(pipe, X_fast, y_fast, cv=3, scoring='roc_auc')
        results[name] = scores
        print(name, 'ROC AUC mean:', scores.mean(), 'std:', scores.std())
    except Exception as e:
        print('Failed', name, e)

res_df = pd.DataFrame({k: v for k, v in results.items()})
display(res_df)


pre_catboost = ColumnTransformer([('num', SimpleImputer(strategy='median'), num_cols)], remainder='passthrough')
X_sample_transformed = pre_catboost.fit_transform(X_fast)
cat_indices = list(range(len(num_cols), X_sample_transformed.shape[1]))


num_pipeline_cb = Pipeline([
    ('impute', SimpleImputer(strategy='median'))
])

pre_catboost = ColumnTransformer([
('num', SimpleImputer(strategy='median'), num_cols),
], remainder='passthrough')

cb_pipe = Pipeline([
    ('pre', pre_catboost),
    ('clf', CatBoostClassifier(random_state=42, verbose=0))
])


param_dist = {
    'iterations': randint(1000, 8000),
    'learning_rate': loguniform(1e-4, 2e-1),
    'od_type': ['Iter', 'IncToDec', None],
    'od_wait': randint(50, 400),                  
    'depth': randint(4, 12),
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
    'border_count': randint(32, 255),
    'l2_leaf_reg': loguniform(1e-2, 100.0),
    'random_strength': uniform(0.0, 5.0),
    'bagging_temperature': uniform(0.0, 2.0),
    'rsm': uniform(0.2, 1.0),
    'subsample': uniform(0.5, 1.0),
    'leaf_estimation_iterations': randint(1, 25),
    'leaf_estimation_method': ['Newton', 'Gradient'],
    'boosting_type': ['Plain', 'Ordered'],
    'loss_function': ['Logloss', 'CrossEntropy'],
    'auto_class_weights': ['Balanced', None],
    'ctr_border_count': randint(50, 200),
    'one_hot_max_size': randint(0, 20),
    'score_function': ['Cosine', 'L2', 'NewtonCosine', 'NewtonL2'],
}


X_full_transformed = pre_catboost.fit_transform(X)

cb = CatBoostClassifier(eval_metric='AUC', cat_features=cat_indices, random_state=42, verbose=0)

rs = RandomizedSearchCV(cb, param_distributions=param_dist, n_iter=500, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
rs.fit(X_full_transformed, y)

print('Best parameters:', rs.best_params_)
print('Best CV AUC:', rs.best_score_)


X_full_transformed = pre_catboost.fit_transform(X)
test_transformed = pre_catboost.transform(test[feature_cols])
cat_indices_full = list(range(len(num_cols), X_full_transformed.shape[1]))


cb_model_full = CatBoostClassifier(
    iterations=rs.best_params_['iterations'],
    depth=rs.best_params_['depth'],
    learning_rate=rs.best_params_['learning_rate'],
    l2_leaf_reg=rs.best_params_['l2_leaf_reg'],
    eval_metric='AUC',
    cat_features=cat_indices_full,
    random_state=42,
    verbose=50
)
cb_model_full.fit(X_full_transformed, y)


preds = cb_model_full.predict_proba(test_transformed)[:, 1]
submission = pd.DataFrame({'id': test['id'], 'loan_paid_back': preds})
submission.to_csv('submission.csv', index=False)
print('Submission saved to submission.csv')
display(submission.head())




