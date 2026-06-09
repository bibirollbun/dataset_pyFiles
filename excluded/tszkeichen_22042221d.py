import numpy as np
import pandas as pd


from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix

import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

RANDOM_STATE = 42


train=pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/train.csv')
test = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv')

train.shape, test.shape


train.target.value_counts()


train.isnull().sum()


X_train = train.drop(['id','target'], axis=1)
y_train = train.target
X_test = test.drop(['id'], axis=1)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics



pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, penalty='l1'))
])

param_grid = {
    'lr__C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    'lr__solver': ['liblinear','saga']
}

lr_grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,                   
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

lr_grid_search.fit(X_train, y_train)
print('===Lasso Classifier===')
print(f"The Best of Parameters: {lr_grid_search.best_params_}")
print(f"The Best of cross-validation score:{lr_grid_search.best_score_}")


def print_gridsearch_scores(grid: GridSearchCV):
    results = pd.DataFrame(grid.cv_results_)
    cols = ['param_lr__C', 'param_lr__solver',
            'mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[cols].copy()
    results.rename(columns={
        'param_lr__C': 'C',
        'param_lr__solver': 'solver',
        'mean_test_score': 'mean_AUC',
        'std_test_score': 'std_AUC',
        'rank_test_score': 'rank'
    }, inplace=True)
    results = results.sort_values('rank').reset_index(drop=True)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n=== Scores for each Lasso model ===")
    print(results.to_string(index=False))
    
print_gridsearch_scores(lr_grid_search)


from sklearn.linear_model import RidgeClassifier
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeClassifier(random_state=RANDOM_STATE, max_iter=10000))
])

ridge_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
}

ridge_grid_search = GridSearchCV(
    estimator=ridge_pipe,
    param_grid=ridge_grid,
    cv=cv,                   
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

ridge_grid_search.fit(X_train, y_train)
print('===Ridge Classifier===')
print(f"The Best of Parameters: {ridge_grid_search.best_params_}")
print(f"The Best of cross-validation score:{ridge_grid_search.best_score_}")


def print_gridsearch_scores(grid: GridSearchCV):
    results = pd.DataFrame(grid.cv_results_)
    cols = ['param_ridge__alpha',
            'mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[cols].copy()
    results.rename(columns={
        'param_ridge__alpha': 'ridge_alpha',
        'mean_test_score': 'mean_AUC',
        'std_test_score': 'std_AUC',
        'rank_test_score': 'rank'
    }, inplace=True)
    results = results.sort_values('rank').reset_index(drop=True)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n=== Scores for each Ridge Classification model ===")
    print(results.to_string(index=False))
    
print_gridsearch_scores(ridge_grid_search)


from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1))
])

rf_grid = {
    'rf__n_estimators'    : [300, 500],
    'rf__max_depth'       : [3, 4],
    'rf__min_samples_leaf': [5],
    'rf__max_features'    : ['sqrt']
}

rf_grid_search = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_grid,
    cv=cv,                   
    scoring='roc_auc',
    n_jobs=-1,
    verbose=3
)

rf_grid_search.fit(X_train, y_train)
print('===Random Forest===')
print(f"The Best of Parameters: {rf_grid_search.best_params_}")
print(f"The Best of cross-validation score:{rf_grid_search.best_score_}")


def print_gridsearch_scores(grid: GridSearchCV):
    results = pd.DataFrame(grid.cv_results_)
    cols = ['param_rf__n_estimators','param_rf__max_depth','param_rf__min_samples_leaf','param_rf__max_features',
            'mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[cols].copy()
    results.rename(columns={
        'param_rf__n_estimators': 'n_estimators',
        'param_rf__max_depth':'max_depth',
        'param_rf__min_samples_leaf':'min_samples_leaf',
        'param_rf__max_features':'max_features',
        'mean_test_score': 'mean_AUC',
        'std_test_score': 'std_AUC',
        'rank_test_score': 'rank'
    }, inplace=True)
    results = results.sort_values('rank').reset_index(drop=True)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n=== Scores for each Random Forest model ===")
    print(results.to_string(index=False))
    
print_gridsearch_scores(rf_grid_search)


from sklearn.ensemble import HistGradientBoostingClassifier

hgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('hgb', HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, scoring='roc_auc'))
])

hgb_grid = {
    'hgb__max_iter'        : [100, 200],
    'hgb__max_depth'       : [3, 4],
    'hgb__learning_rate'   : [0.05, 0.1],
    'hgb__l2_regularization': [ 1.0]
}

hgb_grid_search = GridSearchCV(
    estimator=hgb_pipe,
    param_grid=hgb_grid,
    cv=cv,                   
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

hgb_grid_search.fit(X_train, y_train)
print('===HistGradientBoosting===')
print(f"The Best of Parameters: {hgb_grid_search.best_params_}")
print(f"The Best of cross-validation score:{hgb_grid_search.best_score_}")


def print_gridsearch_scores(grid: GridSearchCV):
    results = pd.DataFrame(grid.cv_results_)
    cols = ['param_hgb__max_iter','param_hgb__max_depth','param_hgb__learning_rate','param_hgb__l2_regularization',
            'mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[cols].copy()
    results.rename(columns={
        'param_hgb__max_iter': 'max_iter',
        'param_hgb__max_depth':'max_depth',
        'param_hgb__learning_rate':'learning_rate',
        'param_hgb__l2_regularization':'l2_regularization',
        'mean_test_score': 'mean_AUC',
        'std_test_score': 'std_AUC',
        'rank_test_score': 'rank'
    }, inplace=True)
    results = results.sort_values('rank').reset_index(drop=True)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n=== Scores for each HistGradientBoosting model ===")
    print(results.to_string(index=False))
    
print_gridsearch_scores(hgb_grid_search)


from sklearn.naive_bayes import GaussianNB

nb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

nb_grid = {}   # no hyper-parameters
nb_grid_search = GridSearchCV(
    estimator=nb_pipe,
    param_grid=nb_grid,
    cv=cv,                   
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

nb_grid_search.fit(X_train, y_train)
print('===Gaussian Naive Bayes===')
#print(f"The Best of Parameters: {nb_grid_search.best_params_}")
print(f"The Best of cross-validation score:{nb_grid_search.best_score_}")


best_model = lr_grid_search.best_estimator_
pred = best_model.predict_proba(X_test)


sample_submission.target = pred[:, 1]
sample_submission.to_csv("/kaggle/working/submission.csv", index=False)

