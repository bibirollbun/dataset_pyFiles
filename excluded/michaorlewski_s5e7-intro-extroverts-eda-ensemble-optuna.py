import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import optuna
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')

train_df.head()


train_df.info()


train_df.describe()


train_df.isna().sum()


test_df.isna().sum()


num_features = test_df.select_dtypes('float64').columns

plt.figure(figsize=(8, 8))
for i, col in enumerate(num_features):
    plt.subplot(3, 2, i+1)
    sns.histplot(train_df[col], bins=10)
    sns.histplot(test_df[col], bins=10)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
for i, col in enumerate(num_features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=train_df[col], y=train_df['Personality'])

plt.tight_layout()
plt.show()


cat_features = test_df.select_dtypes('object').columns

plt.figure(figsize=(8, 8))
for i, col in enumerate(cat_features):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=train_df[col], hue=train_df['Personality'])

plt.tight_layout()
plt.show()


def calc_chi2_pvalue(df, feature, target):
    counts = df[[feature, target]].groupby([feature, target]).size().reset_index(name='Count')
    counts_pivoted = counts.pivot(index=target, values='Count', columns=feature)
    chi2, pvalue, _, _ = stats.chi2_contingency(counts_pivoted)
    return chi2, pvalue

for col in cat_features:
    chi2, pvalue = calc_chi2_pvalue(train_df, col, 'Personality')
    print(f'{col}: chi2 = {chi2:.4f}, p-value = {pvalue}')


train_df['Personality'].value_counts()


X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, X.select_dtypes('float64').columns),
    ('cat', cat_pipeline, X.select_dtypes('object').columns)
], remainder='passthrough')

X = preprocessor.fit_transform(X, y)
X_test = preprocessor.transform(test_df)

le = LabelEncoder()
y = le.fit_transform(y)


def cross_validate(model, X, y, name, verbose=True):
    if verbose:
        print(f'------- {name} -------')
    kf = StratifiedKFold(n_splits=4)

    train_acc, valid_acc = [], []
    train_pre, valid_pre = [], []
    train_rec, valid_rec = [], []
    train_f1, valid_f1 = [], []
    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)

        train_acc.append(accuracy_score(y_train, train_pred))
        valid_acc.append(accuracy_score(y_valid, valid_pred))
        train_pre.append(precision_score(y_train, train_pred))
        valid_pre.append(precision_score(y_valid, valid_pred))
        train_rec.append(recall_score(y_train, train_pred))
        valid_rec.append(recall_score(y_valid, valid_pred))
        train_f1.append(f1_score(y_train, train_pred))
        valid_f1.append(f1_score(y_valid, valid_pred))

        if verbose:
            print(f'Split {i+1}, validation confusion matrix')
            print(confusion_matrix(y_valid, valid_pred))

    if verbose:
        print('Train:')
        print(f'\tAccuracy: {np.mean(train_acc):.4f}')
        print(f'\tPrecision: {np.mean(train_pre):.4f}')
        print(f'\tRecall: {np.mean(train_rec):.4f}')
        print(f'\tF1-score: {np.mean(train_f1):.4f}')
        print('Valid:')
        print(f'\tAccuracy: {np.mean(valid_acc):.4f}')
        print(f'\tPrecision: {np.mean(valid_pre):.4f}')
        print(f'\tRecall: {np.mean(valid_rec):.4f}')
        print(f'\tF1-score: {np.mean(valid_f1):.4f}')

    return np.mean(valid_f1)


def objective(trial):
    lgbm = LGBMClassifier(
        n_estimators=trial.suggest_int('n_estimators', 50, 2000),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3),
        num_leaves=trial.suggest_int('num_leaves', 31, 4096),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        min_split_gain=trial.suggest_float('min_split_gain', 0, 5),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        n_jobs=-1,
        verbose=-1,
    )

    score = cross_validate(lgbm, X, y, 'LGBMClassifier', verbose=False)
    return score

study_lgbm = optuna.create_study(direction='maximize', study_name='LGBMClassifier')
# study_lgbm.optimize(objective, n_trials=200, show_progress_bar=True)
# print(study_lgbm.best_params)


best_params = {'n_estimators': 173, 'learning_rate': 0.2227098051285123, 'num_leaves': 1943,
               'max_depth': 11, 'min_split_gain': 4.585058417323775, 'min_child_weight': 5,
               'subsample': 0.5515913400964882, 'colsample_bytree': 0.5157484943139989,
               'reg_alpha': 1.6928421102635325, 'reg_lambda': 4.3381382972960125}

lgbm = LGBMClassifier(**best_params, verbose=-1, n_jobs=-1)
cross_validate(lgbm, X, y, 'LGBMClassifier')

lgbm.fit(X, y)
lgbm_preds = lgbm.predict_proba(X_test)


def objective(trial):
    xgb = XGBClassifier(
        n_estimators=trial.suggest_int('n_estimators', 50, 2000),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        gamma=trial.suggest_float('gamma', 0, 5),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        n_jobs=-1,
    )

    score = cross_validate(xgb, X, y, 'XGBClassifier', verbose=False)
    return score

study_xgb = optuna.create_study(direction='maximize', study_name='XGBClassifier')
# study_xgb.optimize(objective, n_trials=200, show_progress_bar=True)
# print(study_xgb.best_params)


best_params = {'n_estimators': 418, 'learning_rate': 0.19125584067394716, 'max_depth': 8,
               'gamma': 1.5149664677821972, 'min_child_weight': 2, 'subsample': 0.9292300233081432,
               'colsample_bytree': 0.60991021323087, 'reg_alpha': 0.31464001625577853, 'reg_lambda': 2.643656463178053}

xgb = XGBClassifier(**best_params, n_jobs=-1)
cross_validate(xgb, X, y, 'XGBClassifier')

xgb.fit(X, y)
xgb_preds = xgb.predict_proba(X_test)


preds = [lgbm_preds, xgb_preds]
labels = np.argmax(np.mean(preds, axis=0), axis=1)
final_preds = le.inverse_transform(labels)


submission = pd.DataFrame({'id': test_df.index, 'Personality': final_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)

