!pip install --upgrade scikit-learn


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

df.head()


df.columns


df.info()


df['education_level'].unique()


df['employment_status'].unique()


df['loan_purpose'].nunique()


df['grade_subgrade'].nunique()


from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold


def target_encode_skf(train_df, test_df, col='grade_subgrade', target_col='loan_paid_back', n_splits=5):
    val_encoded = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True)

    for train_idx, val_idx in skf.split(np.zeros(len(train_df[target_col])), train_df[target_col]):
        X_train = train_df.iloc[train_idx]
        X_val = train_df.iloc[val_idx]
        y_train = train_df[target_col].iloc[train_idx]

        noise = 0.03
        te = TargetEncoder(smooth = 10.0, target_type = 'binary')

        te.fit(X_train[[col]], y_train)

        val_te = te.transform(X_val[[col]]).ravel()
        val_te += np.random.normal(0, noise, size=len(val_te))
        val_encoded[val_idx] = val_te

        test_te = te.transform(test_df[[col]]).ravel() 
        test_te += np.random.normal(0, noise, size=len(test_te))
        test_encoded += test_te / n_splits

    return val_encoded, test_encoded


df['grade_subgrade_te'], test_df['grade_subgrade_te'] = target_encode_skf(df, test_df)


df.head()


import matplotlib.pyplot as plt
import seaborn as sns


num_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    if col not in ['id', 'loan_paid_back']:
        sns.histplot(x = col, data = df)
        plt.show()


def create_features(df):
    df = df.copy()

    df['loan_to_income'] = df['loan_amount'] / df['annual_income']
    df['loan_x_dti'] = df['loan_amount'] * df['debt_to_income_ratio']
    df['interest_x_loan'] = df['interest_rate'] * df['loan_amount']
    df['interest_x_dti'] = df['interest_rate'] * df['debt_to_income_ratio']
    df['credit_x_dti'] = df['credit_score'] * df['debt_to_income_ratio']
    df['credit_x_interest'] = df['credit_score'] * df['interest_rate']

    df['log_annual_income'] = np.log1p(df['annual_income'])
    df['log_debt_to_income'] = np.log1p(df['debt_to_income_ratio'])
    df['log_loan_amount'] = np.log1p(df['loan_amount'])
    df['log_credit_score'] = np.log1p(df['credit_score'])

    df['bin_debt_to_income'] = pd.qcut(df['debt_to_income_ratio'], q=10, labels=False)
    df['bin_annual_income'] = pd.qcut(df['annual_income'], q=10, labels=False)
    df['bin_loan_amount'] = pd.qcut(df['loan_amount'], q=10, labels=False)

    df['grade_subgrade_te_x_dti'] = df['grade_subgrade_te'] * df['debt_to_income_ratio']
    df['grade_subgrade_te_x_loan'] = df['grade_subgrade_te'] * df['loan_amount']
    df['grade_subgrade_te_x_interest'] = df['grade_subgrade_te'] * df['interest_rate']

    return df


X = df.drop(['id', 'loan_paid_back', 'grade_subgrade'], axis = 1)
y = df['loan_paid_back']


df_sample = df.sample(frac=0.2)
X_sample = df_sample.drop(['id', 'loan_paid_back', 'grade_subgrade'], axis = 1)
y_sample = df_sample['loan_paid_back']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


num_cols = X.select_dtypes(include=np.number).columns.tolist()
num_cols


cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']


preprocessor = ColumnTransformer([
    ('one_hot', OneHotEncoder(), cat_cols),
    ('num', 'passthrough', num_cols)
])


import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }

    model = xgb.XGBClassifier(**params, early_stopping_rounds=50, eval_metric='auc', verbosity=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    cv = skf.split(np.zeros(len(y_sample)), y_sample)

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv):
        X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

        X_train = create_features(X_train)
        X_val = create_features(X_val)

        preprocessor.fit(X_train)
        X_train_encoded = preprocessor.transform(X_train)
        X_val_encoded = preprocessor.transform(X_val)

        model.fit(
            X_train_encoded, y_train, 
            eval_set=[(X_val_encoded, y_val)], 
            verbose = False
        )

        y_pred = model.predict_proba(X_val_encoded)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        fold_scores.append(score)

        trial.report(np.mean(fold_scores), step = fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# study = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())
# study.optimize(objective, n_trials = 50, n_jobs=8)


xgb_best_params = {'learning_rate': 0.08137818512804033, 'max_depth': 4, 'n_estimators': 860, 'colsample_bytree': 0.5095811336814081, 'subsample': 0.9965077011984054, 'reg_alpha': 1.516310161701913e-08, 'reg_lambda': 0.0283645036397172}


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
        'n_jobs': 1,
        'verbosity': -1,
        'eval_metric': 'auc',
        'objective': 'binary'
    }

    model = lgb.LGBMClassifier(**params, early_stopping_round=50)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    cv = skf.split(np.zeros(len(y_sample)), y_sample)

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv):
        X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

        X_train = create_features(X_train)
        X_val = create_features(X_val)

        preprocessor.fit(X_train)
        X_train_encoded = preprocessor.transform(X_train)
        X_val_encoded = preprocessor.transform(X_val)

        model.fit(
            X_train_encoded, y_train, 
            eval_set=[(X_val_encoded, y_val)]
        )

        y_pred = model.predict_proba(X_val_encoded)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        fold_scores.append(score)

        trial.report(np.mean(fold_scores), step = fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# study = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())
# study.optimize(objective, n_trials = 50, n_jobs=8)


lgb_best_params = {'learning_rate': 0.08086041918711771, 'max_depth': 9, 'n_estimators': 1945, 'feature_fraction': 0.5620235135794959, 'bagging_fraction': 0.7806601309622704, 'lambda_l1': 8.100598818680405, 'lambda_l2': 0.36450455037460305, 'min_data_in_leaf': 151}


def objective(trial):    
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 6),
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'random_strength': trial.suggest_float('random_strength', 0, 0.2),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 64),
        'eval_metric': 'AUC'
    }

    model = cat.CatBoostClassifier(**params, early_stopping_rounds=50, thread_count=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    cv = skf.split(np.zeros(len(y_sample)), y_sample)

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv):
        X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

        X_train = create_features(X_train)
        X_val = create_features(X_val)

        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            cat_features=cat_cols,
            verbose = False
        )

        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        fold_scores.append(score)

        trial.report(np.mean(fold_scores), step = fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# study = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())
# study.optimize(objective, n_trials = 50)


cat_best_params = {'learning_rate': 0.022296876074331236, 'depth': 6, 'iterations': 933, 'colsample_bylevel': 0.8933406380783512, 'l2_leaf_reg': 2.0546440313515553, 'min_data_in_leaf': 20, 'random_strength': 0.11191953243175205, 'bagging_temperature': 0.6754511892370539, 'border_count': 57}


create_features(X).columns


def generate_stack_features(model, model_type, X, y, X_test, cat_features=cat_cols):
    val_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    cv = skf.split(X, y)

    for fold_idx, (train_idx, val_idx) in enumerate(cv):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_full = create_features(X_train)
        X_val_full = create_features(X_val)
        X_test_full = create_features(X_test)

        if model_type == 'cat':
            X_train_encoded = X_train_full
            X_val_encoded = X_val_full
            X_test_encoded = X_test_full

            model.fit(X_train_encoded, y_train, 
                  eval_set=[(X_val_encoded, y_val)],
                  cat_features=cat_features,
                  verbose=False)
        else:
            preprocessor.fit(X_train_full)
            X_train_encoded = preprocessor.transform(X_train_full)
            X_val_encoded = preprocessor.transform(X_val_full)
            X_test_encoded = preprocessor.transform(X_test_full)

            model.fit(X_train_encoded, y_train, 
                  eval_set=[(X_val_encoded, y_val)])

        val_preds[val_idx] = model.predict_proba(X_val_encoded)[:, 1]
        test_preds += model.predict_proba(X_test_encoded)[:, 1] / 5

    return val_preds, test_preds


X_test = test_df.drop(['id', 'grade_subgrade'], axis=1)

X_test.head()


best_xgb_model = xgb.XGBClassifier(**xgb_best_params, eval_metric='auc', verbosity=0)

val_preds, test_preds = generate_stack_features(best_xgb_model, 'xgb', X, y, X_test)


best_lgb_model = lgb.LGBMClassifier(**lgb_best_params, verbosity=-1, eval_metric='auc', objective='binary')

val_preds_lgb, test_preds_lgb = generate_stack_features(best_lgb_model, 'lgb', X, y, X_test)


best_cat_model = cat.CatBoostClassifier(**cat_best_params, eval_metric='AUC')

val_preds_cat, test_preds_cat = generate_stack_features(best_cat_model, 'cat', X, y, X_test)


X_meta = np.column_stack([val_preds, val_preds_lgb, val_preds_cat])
X_meta_test = np.column_stack([test_preds, test_preds_lgb, test_preds_cat])


X_meta


from sklearn.linear_model import LogisticRegression

meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(X_meta, y)

preds = meta_model.predict_proba(X_meta_test)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': (preds > 0.5).astype(int)
})

submission.head()


submission.to_csv("submission.csv", index=False)

