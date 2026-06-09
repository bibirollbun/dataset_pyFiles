run_debug = False
run_submission = True

seed = 10000 if run_submission else 100


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

directory_name = '/kaggle/input/playground-series-s5e11'

for dirname, _, filenames in os.walk(directory_name):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_train = pd.read_csv(f'{directory_name}/train.csv', index_col='id')
df_test = pd.read_csv(f'{directory_name}/test.csv', index_col='id')


target_feature = 'loan_paid_back'

df_train.drop(target_feature, axis=1).isnull().sum() + df_test.isnull().sum()


print(f'Unique values for target feature = {df_train[target_feature].unique()}')
df_train.head(10)


categorical_features = ['gender', 'marital_status', 'education_level', 'loan_purpose', 'grade_subgrade', 'employment_status']

# Combine the training and test datasets so that we can one-hot encode the MTRANS feature in both sets at the same time.
df_full = pd.concat([df_train, df_test])

print('Categorical data in full dataset')
for feature in categorical_features:
    print(f"Categories in feature {feature} = {[i for i in df_full[feature].unique()]}") 


def credit_risk(grade_subgrade: str) -> int:
    grade = ord(grade_subgrade[0]) - 65
    subgrade = grade_subgrade[1]

    return 5 * grade + int(subgrade)

def education_level_ord(education_level: str) -> int:
    match education_level:
        case 'Other': return 0
        case 'High School': return 1
        case 'Bachelor\'s': return 2
        case 'Master\'s': return 3
        case 'PhD': return 4


# Combine the training and test datasets so that we can encode both sets at the same time.
df_full = pd.concat([df_train, df_test])

df_full['credit_risk'] = df_full['grade_subgrade'].map(credit_risk)
df_full['education_Other'] = [1 if i == 'Other' else 0 for i in df_full['education_level']]
df_full['education_ord'] = df_full['education_level'].map(education_level_ord)

df_full = pd.get_dummies(df_full, columns=['gender', 'marital_status', 'employment_status'], drop_first=True, dtype=int)
df_full = df_full.drop(['grade_subgrade', 'education_level'], axis=1)

df_full['credit_index'] = df_full['credit_score'] / df_full['interest_rate']
df_full['loan_burden'] = df_full['loan_amount'] / df_full['annual_income']

df_full.head(10)


# Separate the training and test sets using the indices of each.
df_train = df_full.loc[df_train.index]
df_test = df_full.loc[df_test.index]
df_test = df_test.drop(target_feature, axis=1)


from sklearn.model_selection import train_test_split

X = df_train.copy()
y = df_train[target_feature].copy()
X_test = df_test.copy()

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=seed)


for feature in ['loan_purpose']:
    lp_freqs = X_train[feature].value_counts() / X_train.shape[0]
    lp_means = X_train.groupby(feature)[target_feature].mean()

    if feature == 'loan_purpose':
        X[feature] = X[feature].map(lp_means)
        X_test[feature] = X_test[feature].map(lp_means)

X = X.drop(target_feature, axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=seed)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(X, y):
    X_y = X.join(y, on='id')
    
    _, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(X_y.corr(),
                cmap='RdBu',
                annot=True,
                fmt=".2f",
                ax=ax)
    plt.show()

plot_correlation_matrix(X, y)


from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score

roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

base_objective = 'reg:squarederror'
base_eval_metric = 'auc'

def create_base_model():
    return LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=seed,
        verbose=2 if run_debug else -1)

def score_model(X, y, model=None, print_score=True, return_model=False):
    Xt, Xv, yt, yv = train_test_split(X, y, random_state=seed)
    _model = create_base_model() if model is None else model
    
    cv = 5
    cv_score = cross_val_score(_model, Xt, yt, scoring=roc_auc_scorer, cv=cv)
    
    if run_debug:
        print(f'Cross validation scores from latest run = {cv_score}.')
    
    _score = cv_score.mean()
    
    if print_score:
        print(f'Baseline CV score with {type(_model)} model = {_score:.5f} ± {cv_score.std()/np.sqrt(cv):.5f}.')
    
    if return_model:
        return _score, _model
    return _score


base_score, base_model = score_model(X_train, y_train, return_model=True)

best_score = base_score
best_model = base_model


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [1000] if run_submission else [100],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [10, 20, 50, -1],
    'subsample': [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'min_child_weight': [0.001, 0.01, 0.1, 1, 10, 100],
    'reg_lambda': [0, 0.5, 1, 2, 5, 10],
    'reg_alpha': [0, 0.1, 0.5],
    'num_leaves': [31, 63, 127, 255],
    'min_data_in_leaf': [20, 40, 80],
    'feature_fraction': [0.7, 0.8, 0.9],
    'bagging_fraction': [0.7, 0.8, 0.9],
}

lgb_model = LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=seed,
    verbose=2 if run_debug else -1)

search = RandomizedSearchCV(
    lgb_model,
    param_distributions=param_grid,
    n_iter=20 if run_submission else 3,
    scoring=roc_auc_scorer,
    cv=5,
    n_jobs=1,
    random_state=seed
)

search.fit(X_train, y_train)

print("Best CV ROC AUC score:", search.best_score_)
print("Best params:", search.best_params_)

score, model = score_model(X_train, y_train, model=search.best_estimator_, return_model=True)


if score > base_score:
    best_score = score
    best_model = model


prev_best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'random_state': seed,
    'verbose': 2 if run_debug else -1,
    'subsample': 0.7,
    'reg_lambda': 10,
    'reg_alpha': 0.5,
    'num_leaves': 63,
    'n_estimators': 1000 if run_submission else 100,
    'min_data_in_leaf': 80,
    'min_child_weight': 10,
    'max_depth': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'colsample_bytree': 1.0,
    'bagging_fraction': 0.7
}

lgb_model = LGBMClassifier(**prev_best_params)

score, model = score_model(X_train, y_train, model=lgb_model, return_model=True)

if score > base_score:
    best_score = score
    best_model = model


import shap

if run_submission:
    best_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(best_model.booster_)
    shap_values = explainer.shap_values(X_val)
    shap.summary_plot(shap_values, X_val)


from sklearn.inspection import permutation_importance
import pandas as pd

def stabilise_features(model, Xt, yt, Xv, yv, debug=False):
    params = model.get_params()
    params['early_stopping_rounds'] = None
    cur_model = LGBMClassifier(**params)

    cur_model.fit(Xt, yt)
    perm_imp = permutation_importance(cur_model, Xv, yv, n_repeats=5, scoring=roc_auc_scorer, random_state=seed)
    perm_imp = pd.Series(abs(perm_imp.importances_mean), index=X_val.columns).sort_values(ascending=False)

    cur_auc = score_model(Xt, yt, model=model)
    cur_features = X_train.columns
    delta_auc = 0
    
    if debug:
        print(f'Permutation importance scores:\n{perm_imp}')
        print(f'Initial AUC = {cur_auc}.')

    idx = 1
    num_idxs = perm_imp.size

    if debug:
        print(f'Num indexes = {num_idxs}.')

    while idx < num_idxs:
        if debug:
            print(f'Index = {idx}.')
        new_features = perm_imp[:-idx].index
        Xt_ = Xt[new_features]
        new_auc = score_model(Xt_, yt, model=cur_model)
        delta_auc = new_auc - cur_auc

        if debug:
            print(f'New features = {new_features}.')
            print(f'New AUC = {new_auc}.')
            print(f'Delta AUC = {delta_auc}.')

        if delta_auc <= 0:
            return cur_features, cur_auc, cur_model

        cur_features = new_features
        cur_auc = new_auc
        idx += 1


reduced_features, score, model = stabilise_features(best_model, X_train, y_train, X_val, y_val)
X_rc = X[reduced_features]
X_rc_train = X_train[reduced_features]
X_rc_val = X_val[reduced_features]


if score > best_score:
    best_score = score
    best_model = model
    X = X_rc
    X_train = X_rc_train
    X_val = X_rc_val
    X_test = X_test[reduced_features]


params = best_model.get_params()
params['early_stopping_rounds'] = 50
params['n_estimators'] = 5000 if run_submission else 500
params['eval_metric'] = ['logloss', 'rmse']

model = LGBMClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

params['early_stopping_rounds'] = None
params['n_estimators'] = model.best_iteration_

y_preds = []
for s in range(5):
    s = seed + s * 10000
    params['random_state'] = s
    model = LGBMClassifier(**params)
    model.fit(X, y)
    y_pred = model.predict_proba(X_test)[:, 1]
    y_preds.append(y_pred)

y_pred = np.clip(np.mean(y_preds, axis=0), 0.0, 1.0)
y_pred = pd.DataFrame(y_pred, index=X_test.index, columns=[target_feature])

y_pred


y_pred.to_csv('submission.csv')

