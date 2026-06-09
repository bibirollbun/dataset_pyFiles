pip install optuna-integration


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
import catboost as catb

import optuna
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback, CatBoostPruningCallback

from prettytable import PrettyTable

import warnings
warnings.filterwarnings('ignore')


# load data
df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e11/test.csv')
df_subm = pd.read_csv(r'/kaggle/input/playground-series-s5e11/sample_submission.csv', usecols = ['id'])

df_train.head()


# remove id column
df_train.drop('id', axis = 1, inplace = True)
df_test.drop('id', axis = 1, inplace = True)

# check num of duplicated rows
print(f'Num of duplicated rows in train dataset: {len(df_train[df_train.duplicated()])}')


# check num of nulls
pd.DataFrame([df_train.isnull().sum(), df_test.isnull().sum()]).T\
    .rename(columns = {0: 'Num of nulls in train dataset', 1: 'Num of nulls in test dataset'})



# distribution of columns
df = pd.concat([df_train, df_test])
df.loc[df['loan_paid_back'].isnull(), 'dataset'] = 'test'
df.loc[df['loan_paid_back'].notnull(), 'dataset'] = 'train'

plt.figure(figsize = (15, 15))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(5, 3, i)
    if df[col].dtypes in (float, int) and col != 'loan_paid_back':
        b = sbn.histplot(data = df, x = col, hue = 'dataset')
    elif col == 'loan_paid_back':
        b = sbn.countplot(data = df.fillna('Null'), x = col, hue = 'dataset')
    else:
        b = sbn.countplot(data = df, x = col, order = df[col].value_counts().index, hue = 'dataset')
    b.tick_params(axis = 'both', labelsize = 7)
    plt.grid(axis = 'y')
    plt.title(f'Comp. of {col} distr. in train and test dataset')
plt.tight_layout()
plt.show()

df.drop('dataset', axis = 1, inplace = True)


# pairplot
sbn.pairplot(data = df_train.groupby('loan_paid_back', group_keys = False)[df_train.columns]\
             .apply(lambda x: x.sample(frac = 0.2)), hue = 'loan_paid_back', diag_kind = 'hist')


# correlation
sbn.heatmap(data = df_train.corr(numeric_only = True), annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Correlation of numeric data in train dataset')


# distribution of columns by target
plt.figure(figsize = (15, 15))
for i, col in enumerate(df_train.select_dtypes('object').columns, 1):
    plt.subplot(5, 3, i)
    b = sbn.countplot(data = df_train, x = col, hue = 'loan_paid_back', order = df_train[col].value_counts().index)
    b.tick_params(axis = 'both', labelsize = 7)
    plt.grid(axis = 'y')
    plt.title(f'Distribution of {col} by target')
    plt.tight_layout()
plt.show()


# repayment rate
plt.figure(figsize = (15, 15))
for i, col in enumerate(df_train.select_dtypes('object').columns, 1):
    plt.subplot(5, 3, i)
    repayment_rate = df_train.groupby(col)['loan_paid_back'].mean().sort_values(ascending = False)
    b = sbn.barplot(x = repayment_rate.index, y = repayment_rate.values,
                    palette = 'tab10')
    b.tick_params(axis = 'both', labelsize = 7)
    plt.ylabel('repayment_rate')
    plt.grid(axis = 'y')
    plt.title(f'Repayment rate by {col}')
plt.tight_layout()
plt.show()


# create new columns
df['monthly_income'] = df['annual_income'] / 12
df['estimated_monthly_payment'] = (df['loan_amount'] * (df['interest_rate'] / 100)) / 12
df['payment_to_income_ratio'] = df['estimated_monthly_payment'] / (df['monthly_income'] + 1)
df['loan_to_annual_income'] = df['loan_amount'] / (df['annual_income'] + 1)
df['high_risk_flag'] = ((df['credit_score'] < 650) & (df['debt_to_income_ratio'] > 0.43)).astype(int)
df['income_adequacy'] = df['annual_income'] / (df['loan_amount'] + 1)
df['estimated_total_debt'] = df['annual_income'] * df['debt_to_income_ratio']
df['remaining_income_after_payment'] = df['monthly_income'] - df['estimated_monthly_payment']

# create FICO column
df.loc[df['credit_score'] <= 850, 'FICO_rating'] = 'Exceptional'
df.loc[df['credit_score'] <= 799, 'FICO_rating'] = 'Very Good'
df.loc[df['credit_score'] <= 739, 'FICO_rating'] = 'Good'
df.loc[df['credit_score'] <= 669, 'FICO_rating'] = 'Fair'
df.loc[df['credit_score'] <= 579, 'FICO_rating'] = 'Poor'


# create grade and subgrade columns and loan rank
df['grade'] = df['grade_subgrade'].apply(lambda x: x[0])
df['subgrade'] = df['grade_subgrade'].apply(lambda x: x[1])

grade_note = {'A': 0, 'B': 5, 'C': 10, 'D': 15, 'E': 20, 'F': 25}
df['loan_rank'] = df['grade'].map(grade_note) + df['subgrade'].astype(int)

df.drop('subgrade', axis = 1, inplace = True)


# change object into category type
cat_cols = [col for col in df.select_dtypes('object').columns]
for col in cat_cols:
    df[col] = df[col].astype('category')

# log transformation for numeric columns
for col in [col for col in df.select_dtypes('float') if col != 'loan_paid_back'] + ['credit_score']:
    df[col] = np.log1p(df[col])

df.head()


def optuna_lgbm(
        train_set: pd.DataFrame,
        target: pd.Series,
        categorical: list,
        cv_strategy,
        cv_n_splits: int,
        n_estimators: int,
        early_stopping_round: int,
        eval_metric: str,
        random_state: int,
        verbose: int,
        direction: str,
        n_trials: int,
        timeout: int,
        n_jobs: int
):
        def objective(trial):
                params = {
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'lambda_l1': trial.suggest_int('lambda_l1', 0, 100, step = 5),
                        'lambda_l2': trial.suggest_int('lambda_l2', 0, 100, step = 5),
                        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 0.95, step = 0.1),
                        'bagging_freq': trial.suggest_categorical('bagging_freq', [1]),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.95, step = 0.1),
                        'subsample': trial.suggest_float('subsample', 0.4, 1.0, log = True),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0, log = True),
                        'max_features': trial.suggest_categorical('max_features', choices = ['auto', 'sqrt', 'log2']),
                        'device': trial.suggest_categorical('device', ['gpu'])
                }

                model = lgb.LGBMClassifier(**params, 
                                        n_estimators = n_estimators,
                                        early_stopping_rounds = early_stopping_round,
                                        random_state = random_state,
                                        verbose = verbose,                                    
                                        n_jobs = n_jobs,
                                        )

                cv = cv_strategy(n_splits = cv_n_splits, shuffle = True)
                scores = []
                for train_idx, test_idx in cv.split(train_set, target):
                        X_train, y_train = train_set.iloc[train_idx], target.iloc[train_idx]
                        X_test, y_test = train_set.iloc[test_idx], target.iloc[test_idx]
                
                model.fit(X_train, 
                        y_train, 
                        eval_set = [(X_test, y_test)], 
                        eval_metric = eval_metric,
                        categorical_feature = categorical, 
                        callbacks = [LightGBMPruningCallback(trial, eval_metric)]
                        )
                preds = model.predict_proba(X_test)[:, 1]
                
                score = roc_auc_score(y_test, preds)
                scores.append(score)

                return np.mean(scores)

        study = optuna.create_study(direction = direction, load_if_exists = True)
        study.optimize(objective, n_trials = n_trials, timeout = timeout, n_jobs = n_jobs, show_progress_bar = True)

        best_params = study.best_params

        return best_params


def optuna_xgb(
        train_set: pd.DataFrame,
        target: pd.Series,
        categorical: bool,
        cv_strategy,
        cv_n_splits: int,
        n_estimators: int,
        early_stopping_round: int,
        eval_metric: str,
        random_state: int,
        verbose: int,
        direction: str,
        n_trials: int,
        timeout: int,
        n_jobs: int
):
        def objective(trial):
                params = {
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'lambda': trial.suggest_int('lambda', 0, 100, step = 5),
                        'alpha': trial.suggest_int('alpha', 0, 100, step = 5),
                        'seed': trial.suggest_categorical('seed', [random_state]),
                        'eval_metric': trial.suggest_categorical('eval_metric', [eval_metric]),
                        'n_jobs': trial.suggest_categorical('n_jobs', [n_jobs]),
                        'device': trial.suggest_categorical('device', ['cuda'])
                }

                cv = cv_strategy(n_splits = cv_n_splits, shuffle = True)
                scores = []
                for train_idx, test_idx in cv.split(train_set, target):
                        X_train, y_train = train_set.iloc[train_idx], target.iloc[train_idx]
                        X_test, y_test = train_set.iloc[test_idx], target.iloc[test_idx]

                        X_train_dmatrix = xgb.DMatrix(X_train, label = y_train, enable_categorical = categorical)
                        X_test_dmatrix = xgb.DMatrix(X_test, label = y_test, enable_categorical = categorical)
                        
                        model = xgb.train(params, 
                                          X_train_dmatrix,
                                          num_boost_round = n_estimators, 
                                          early_stopping_rounds = early_stopping_round, 
                                          evals = [(X_test_dmatrix, 'eval')],                                          
                                          callbacks = [XGBoostPruningCallback(trial, f'eval-{eval_metric}')], 
                                          verbose_eval = verbose
                                          )
                        preds = model.predict(X_test_dmatrix)
                        score = roc_auc_score(y_test, preds)
                        scores.append(score)

                return np.mean(scores)

        study = optuna.create_study(direction = direction, load_if_exists = True)
        study.optimize(objective, n_trials = n_trials, timeout = timeout, n_jobs = n_jobs, show_progress_bar = True)

        best_params = study.best_params

        return best_params


def optuna_catboost(
    train_set: pd.DataFrame,
    target: pd.Series,
    categorical: list,
    cv_strategy,
    cv_n_splits: int,
    n_estimators: int,
    early_stopping_round: int,
    eval_metric: str,
    random_state: int,
    verbose: int,
    direction: str,
    n_trials: int,
    timeout: int
):
    train_pool = catb.Pool(train_set, label = target, cat_features = categorical)
    train_pool.quantize()

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli']),
            'task_type': trial.suggest_categorical('task_type', ['GPU'])
        }

        model = catb.CatBoostClassifier(**params,
                                        iterations = n_estimators, 
                                        early_stopping_rounds = early_stopping_round,
                                        eval_metric = str.upper(eval_metric), 
                                        cat_features = categorical,  
                                        random_seed = random_state 
                                        )

        cv = cv_strategy(n_splits = cv_n_splits, shuffle = True)
        scores = []
        for train_idx, test_idx in cv.split(train_set, target):
            y_test = target.iloc[test_idx]

            X_train_pool = train_pool.slice(train_idx)
            X_test_pool = train_pool.slice(test_idx)
            
            model.fit(X_train_pool, 
                      eval_set = [(X_test_pool)], 
                      # callbacks = [CatBoostPruningCallback(trial, str.upper(eval_metric))], 
                      verbose = verbose
                      )
            preds = model.predict_proba(X_test_pool)[:, 1]
            
            score = roc_auc_score(y_test, preds)
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction = direction, load_if_exists = True)
    study.optimize(objective, n_trials = n_trials, timeout = timeout, show_progress_bar = True)

    best_params = study.best_params

    return best_params


def final_lgbm_xgb_catb_train_fit(
        params_lgbm: dict, 
        params_xgb: dict, 
        params_catb: dict,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        target: pd.Series,
        categorical: list,
        cv_strategy,
        cv_n_split: int,
        n_estimators: int,
        early_stopping_round: int,
        eval_metric: str,
        random_state: int,
        verbose: list,
        n_jobs: int
):
        
        # outputs
        preds_lgbm_dftrain, preds_xgb_dftrain, preds_catb_dftrain = np.zeros(len(target)), np.zeros(len(target)), np.zeros(len(target))
        preds_lgbm_dftest, preds_xgb_dftest, preds_catb_dftest = np.zeros(len(test_set)), np.zeros(len(test_set)), np.zeros(len(test_set))
        score_lgbm, score_xgb, score_catb = [], [], []

        # prepare datasets for xgb and catb
        df_test_dmatrix = xgb.DMatrix(test_set, enable_categorical = categorical[1], feature_names = list(train_set.columns))

        train_pool = catb.Pool(train_set, label = target, cat_features = categorical[0])
        train_pool.quantize()

        # cross validation
        cv = cv_strategy(n_splits = cv_n_split, shuffle = True)
        for i, (train_idx, test_idx) in enumerate(cv.split(train_set, target), 1):
                X_train, y_train = train_set.iloc[train_idx], target.iloc[train_idx]
                X_test, y_test = train_set.iloc[test_idx], target.iloc[test_idx]
                print(f'\n*****Fold {i}*****')

                # LGBM
                print('\tComputing LGBM predictions...')
                model = lgb.LGBMClassifier(**params_lgbm, 
                                           random_state = random_state, 
                                           early_stopping_rounds = early_stopping_round, 
                                           n_estimators = n_estimators, 
                                           n_jobs = n_jobs, 
                                           verbose = verbose[0])
                
                model.fit(X_train, 
                          y_train, 
                          categorical_feature = categorical[0], 
                          eval_metric = eval_metric, 
                          eval_set = [(X_test, y_test)])
                
                preds = model.predict_proba(X_test)[:, 1]
                preds_lgbm_dftrain[test_idx] = preds
                score = roc_auc_score(y_test, preds)
                score_lgbm.append(score)

                preds = model.predict_proba(test_set)[:, 1] / cv.n_splits
                preds_lgbm_dftest += preds
                print(f'\t\tScore: {score}')
                
                # XGB   
                print('\tComputing XGB predictions...')             
                X_train_dmatrix = xgb.DMatrix(X_train, label = y_train, enable_categorical = categorical[1])
                X_test_dmatrix = xgb.DMatrix(X_test, label = y_test, enable_categorical = categorical[1])        

                model = xgb.train(params_xgb, 
                                   X_train_dmatrix, 
                                   evals = [(X_test_dmatrix, 'eval')], 
                                   early_stopping_rounds = early_stopping_round, 
                                   num_boost_round = n_estimators,
                                   verbose_eval = verbose[1])
                
                preds = model.predict(X_test_dmatrix)
                preds_xgb_dftrain[test_idx] = preds
                score = roc_auc_score(y_test, preds)
                score_xgb.append(score)
                preds = model.predict(df_test_dmatrix) / cv.n_splits
                preds_xgb_dftest += preds
                print(f'\t\tScore: {score}')

                # CatBoost
                print('\tComputing CatBoost predictions...')  
                model = catb.CatBoostClassifier(**params_catb,
                                                random_state = random_state, 
                                                iterations = n_estimators, 
                                                early_stopping_rounds = early_stopping_round, 
                                                eval_metric = str.upper(eval_metric), 
                                                cat_features = categorical[0]
                                                )
                
                X_train_pool = train_pool.slice(train_idx)
                X_test_pool = train_pool.slice(test_idx)

                model.fit(X_train_pool, 
                          verbose = verbose[2], 
                          eval_set = [(X_test_pool)])
                
                preds = model.predict_proba(X_test_pool)[:, 1]
                preds_catb_dftrain[test_idx] = preds
                score = roc_auc_score(y_test, preds)
                score_catb.append(score)
                preds = model.predict_proba(test_set)[:, 1] / cv.n_splits
                preds_catb_dftest += preds
                print(f'\t\tScore: {score}')

        print('\nDone')
        print(f'\nMean score LGBM:     {np.mean(score_lgbm)}' + ' +- ' + f'{np.std(score_lgbm)}')
        print(f'Mean score XGB:      {np.mean(score_xgb)}' + ' +- ' + f'{np.std(score_xgb)}')
        print(f'Mean score CatBoost: {np.mean(score_catb)}' + ' +- ' + f'{np.std(score_catb)}')

        return preds_lgbm_dftrain, preds_xgb_dftrain, preds_catb_dftrain, \
                preds_lgbm_dftest, preds_xgb_dftest, preds_catb_dftest, \
                score_lgbm, score_xgb, score_catb



def summary_models_results(
        scores_lgbm = list,
        scores_xgb = list,
        scores_catb = list
):
    # mean score
    score_lgbm_mean = np.round(np.mean(scores_lgbm), 6)
    score_xgb_mean = np.round(np.mean(scores_xgb), 6)
    score_catb_mean = np.round(np.mean(scores_catb), 6)

    # std score
    score_lgbm_std = np.round(np.std(scores_lgbm), 6)
    score_xgb_std = np.round(np.std(scores_xgb), 6)
    score_catb_std = np.round(np.std(scores_catb), 6)

    # Pretty Table
    print('Scores:')
    myTable = PrettyTable(['', 'LGBM', 'XGB', 'CatBoost'])

    for j in range(len(scores_lgbm)):
        myTable.add_row([f'Fold {j + 1}', np.round(scores_lgbm[j], 6), np.round(scores_xgb[j], 6), np.round(scores_catb[j], 6)])
        
    myTable.add_divider()
    myTable.add_row(['Mean score', score_lgbm_mean, score_xgb_mean, score_catb_mean], divider = True)
    myTable.add_row(['Std score', score_lgbm_std, score_xgb_std, score_catb_std], divider = True)
    myTable.align[''] = 'l'
    myTable.align['LGBM'] = 'r'
    myTable.align['XGB'] = 'r'
    myTable.align['CatBoost'] = 'r'
    print(myTable)


def blended_lgbm_preds(
        train: list,
        test: list,
        target: pd.Series,
        cv_strategy,
        cv_n_splits: int,
        n_estimators: int,
        early_stopping_round: int,
        eval_metric: str,
        random_state: int,
        verbose: int,
        direction: str,
        n_trials: int,
        timeout: int,
        n_jobs: int
):      
        # define datasets
        X = pd.DataFrame(train).T
        df_test = pd.DataFrame(test).T

        # optuna prediction
        def objective(trial):
                params = {
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'lambda_l1': trial.suggest_int('lambda_l1', 0, 100, step = 5),
                        'lambda_l2': trial.suggest_int('lambda_l2', 0, 100, step = 5),
                        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 0.95, step = 0.1),
                        'bagging_freq': trial.suggest_categorical('bagging_freq', [1]),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.95, step = 0.1),
                        'subsample': trial.suggest_float('subsample', 0.4, 1.0, log = True),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0, log = True),
                        'max_features': trial.suggest_categorical('max_features', choices = ['auto', 'sqrt', 'log2']),
                        'device': trial.suggest_categorical('device', ['gpu'])
                }

                model = lgb.LGBMClassifier(**params, 
                                           random_state = random_state, 
                                           early_stopping_rounds = early_stopping_round, 
                                           n_estimators = n_estimators, 
                                           n_jobs = n_jobs, 
                                           verbose = verbose
                                           )
                
                cv = cv_strategy(n_splits = cv_n_splits, shuffle = True)
                scores = []
                for train_idx, test_idx in cv.split(X, target):

                        X_train, y_train = X.iloc[train_idx], target.iloc[train_idx]
                        X_test, y_test = X.iloc[test_idx], target.iloc[test_idx]

                        model.fit(X_train, 
                                  y_train, 
                                  eval_metric = eval_metric,
                                  eval_set = [(X_test, y_test)], 
                                  callbacks = [LightGBMPruningCallback(trial, eval_metric)]
                                  )
                        preds = model.predict_proba(X_test)[:, 1]
                        
                        score = roc_auc_score(y_test, preds)
                        scores.append(score)

                return np.mean(scores)

        study = optuna.create_study(direction = direction, load_if_exists = True)
        study.optimize(objective, n_trials = n_trials, timeout = timeout, n_jobs = n_jobs, show_progress_bar = True)
        best_params_model_blender = study.best_params

        # configuration of blender model
        model_blender = lgb.LGBMClassifier(**best_params_model_blender, 
                                           random_state = random_state, 
                                           early_stopping_rounds = early_stopping_round, 
                                           n_estimators = n_estimators, 
                                           n_jobs = n_jobs, 
                                           verbose = verbose
                                           )
        
        # outpusts
        preds_model_blender_dftrain, preds_model_blender_dftest = np.zeros(len(X)), np.zeros(len(df_test))
        score_model_blender = []

        # cross validation of blender model
        cv = StratifiedKFold(n_splits = cv_n_splits, shuffle = True)
        for i, (train_idx, test_idx) in enumerate(cv.split(X, target), 1):

                X_train, y_train = X.iloc[train_idx], target.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], target.iloc[test_idx]
                print(f'\n*****Fold {i}*****')

                # cross validation
                print('\tComputing blender model predictions...')

                model_blender.fit(X_train, 
                                  y_train, 
                                  eval_metric = eval_metric, 
                                  eval_set = [(X_test, y_test)]
                                  )
                
                preds = model_blender.predict_proba(X_test)[:, 1]
                preds_model_blender_dftrain[test_idx] = preds
                score = roc_auc_score(y_test, preds)
                score_model_blender.append(score)
                preds = model_blender.predict_proba(df_test)[:, 1] / cv.n_splits
                preds_model_blender_dftest += preds
                print(f'\t\tScore: {score}')
                
        print('\nDone')
        print(f'\nMean score: {np.mean(score_model_blender)}' + ' +- ' + f'{np.std(score_model_blender)}')

        return best_params_model_blender, preds_model_blender_dftrain, preds_model_blender_dftest, score_model_blender


def blended_xgb_preds(
        train: list,
        test: list,
        target: pd.Series,
        cv_strategy,
        cv_n_splits: int,
        n_estimators: int,
        early_stopping_round: int,
        eval_metric: str,
        random_state: int,
        verbose: int,
        direction: str,
        n_trials: int,
        timeout: int,
        n_jobs: int
):      
        # define datasets
        X = pd.DataFrame(train).T
        df_test = pd.DataFrame(test).T
        df_test_dmatrix = xgb.DMatrix(df_test, feature_names = [str(x) for x in df_test.columns])

        # optuna prediction
        def objective(trial):
                params = {
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'lambda': trial.suggest_int('lambda', 0, 100, step = 5),
                        'alpha': trial.suggest_int('alpha', 0, 100, step = 5),
                        'seed': trial.suggest_categorical('seed', [random_state]),
                        'eval_metric': trial.suggest_categorical('eval_metric', [eval_metric]),
                        'n_jobs': trial.suggest_categorical('n_jobs', [n_jobs]),
                        'device': trial.suggest_categorical('device', ['cuda'])
                }
                
                cv = cv_strategy(n_splits = cv_n_splits, shuffle = True)
                scores = []
                for train_idx, test_idx in cv.split(X, target):

                        X_train, y_train = X.iloc[train_idx], target.iloc[train_idx]
                        X_test, y_test = X.iloc[test_idx], target.iloc[test_idx]

                        X_train_dmatrix = xgb.DMatrix(X_train, label = y_train)
                        X_test_dmatrix = xgb.DMatrix(X_test, label = y_test)

                        model = xgb.train(params, 
                                   X_train_dmatrix,
                                   num_boost_round = n_estimators, 
                                   early_stopping_rounds = early_stopping_round, 
                                   evals = [(X_test_dmatrix, 'eval')], 
                                   callbacks = [XGBoostPruningCallback(trial, 'eval-auc')],
                                   verbose_eval = verbose
                                   )
                
                        preds = model.predict(X_test_dmatrix)
                        score = roc_auc_score(y_test, preds)
                        scores.append(score)

                return np.mean(scores)

        study = optuna.create_study(direction = direction, load_if_exists = True)
        study.optimize(objective, n_trials = n_trials, timeout = timeout, n_jobs = n_jobs, show_progress_bar = True)
        best_params_model_blender = study.best_params

        # outpusts
        preds_model_blender_dftrain, preds_model_blender_dftest = np.zeros(len(X)), np.zeros(len(df_test))
        score_model_blender = []

        # cross validation of blender model
        cv = cv_strategy(n_splits = cv_n_splits, shuffle = True)
        for i, (train_idx, test_idx) in enumerate(cv.split(X, target), 1):

                X_train, y_train = X.iloc[train_idx], target.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], target.iloc[test_idx]

                X_train_dmatrix = xgb.DMatrix(X_train, label = y_train)
                X_test_dmatrix = xgb.DMatrix(X_test, label = y_test)
                print(f'\n*****Fold {i}*****')

                # cross validation
                print('\tComputing blender model predictions...')

                model_blender = xgb.train(best_params_model_blender, 
                                   X_train_dmatrix, 
                                   num_boost_round = n_estimators,
                                   early_stopping_rounds = early_stopping_round,
                                   evals = [(X_test_dmatrix, 'eval')],                                 
                                   verbose_eval = verbose
                                   )
                
                preds = model_blender.predict(X_test_dmatrix)
                preds_model_blender_dftrain[test_idx] = preds
                score = roc_auc_score(y_test, preds)
                score_model_blender.append(score)
                preds = model_blender.predict(df_test_dmatrix) / cv.n_splits
                preds_model_blender_dftest += preds
                print(f'\tScore: {score}')
                
        print('\nDone')
        print(f'Mean score: {np.mean(score_model_blender)}' + ' +- ' + f'{np.std(score_model_blender)}')
              
        return best_params_model_blender, preds_model_blender_dftrain, preds_model_blender_dftest, score_model_blender


def weighted_preds(
        predictions_dftrain_to_weight: list, 
        predictions_dftest: list,
        target = pd.Series,
        n_trials = int,
        timeout = int
):
    # optuna compute weights
    def objective(trial):
        w_1 = trial.suggest_float('w_1', 0, 1)
        w_2 = trial.suggest_float('w_2', 0, 1)
        w_3 = 1 - (w_1 + w_2)
        
        if w_3 < 0 or w_3 > 1:
            raise optuna.exceptions.TrialPruned()
            
        ensamble_preds = w_1 * predictions_dftrain_to_weight[0] + \
                        w_2 * predictions_dftrain_to_weight[1] + \
                        w_3 * predictions_dftrain_to_weight[2]
            
        score = roc_auc_score(target, ensamble_preds)
        return score
    
    study = optuna.create_study(direction = 'maximize', load_if_exists = True)
    study.optimize(objective, n_trials = n_trials, timeout = timeout, n_jobs = -1, show_progress_bar = True)
    
    # outputs
    weights = study.best_params.copy()
    weights[f'w{len(predictions_dftrain_to_weight)}'] = 1 - sum(weights.values())

    lista_dftrain = [a * b for (a, b) in zip(weights.values(), predictions_dftrain_to_weight)]
    lista_dftest = [a * b for (a, b) in zip(weights.values(), predictions_dftest)]

    weighted_preds_dftrain = np.zeros(len(predictions_dftrain_to_weight[0]))
    for i in range(len(predictions_dftrain_to_weight[0])):
        weighted_preds_dftrain[i] = np.sum([lista_dftrain[k][i] for k in range(len(weights))])
    
    weighted_preds_dftest = np.zeros(len(predictions_dftest[0]))
    for i in range(len(predictions_dftest[0])):
        weighted_preds_dftest[i] = np.sum([lista_dftest[k][i] for k in range(len(weights))])

    score_weighted = roc_auc_score(target, weighted_preds_dftrain)

    return weights, weighted_preds_dftrain, weighted_preds_dftest, score_weighted


# cross validation settings
CV_STRATEGY = StratifiedKFold
CV_N_SPLITS = 5

# model settings
N_ESTIMATORS = 5000
EARLY_STOPPING_ROUND = 100
RANDOM_STATE = 42
EVAL_METRIC = 'auc'

# optuna settings
DIRECTION = 'maximize'
N_TRIALS = 20000
TIMEOUT = 36000
N_JOBS = -1

# optuna blender settings
N_TRIALS_BLENDER = 3000
TIMEOUT_BLENDER = 7200

# optuna weight settings
N_TRIALS_WEIGHT = 3000
TIMEOUT_WEIGHT = 7200


# prepare datasets
df_train = df[df['loan_paid_back'].notnull()]
df_test = df[~df['loan_paid_back'].notnull()].drop('loan_paid_back', axis = 1)

X = df_train.drop('loan_paid_back', axis = 1)
y = df_train['loan_paid_back'].astype(int)


# best_params_lgbm = optuna_lgbm(
#     train_set = X,
#     target = y,
#     categorical = cat_cols,
#     cv_strategy = CV_STRATEGY,
#     cv_n_splits = CV_N_SPLITS,
#     n_estimators = N_ESTIMATORS,
#     early_stopping_round = EARLY_STOPPING_ROUND,
#     eval_metric = EVAL_METRIC,
#     random_state = RANDOM_STATE,
#     verbose = -1,
#     direction = DIRECTION,
#     n_trials = N_TRIALS,
#     timeout = TIMEOUT,
#     n_jobs = N_JOBS
# )


# best_params_xgb = optuna_xgb(
#     train_set = X,
#     target = y,
#     categorical = True,
#     cv_strategy = CV_STRATEGY,
#     cv_n_splits = CV_N_SPLITS,
#     n_estimators = N_ESTIMATORS,
#     early_stopping_round = EARLY_STOPPING_ROUND,
#     eval_metric = EVAL_METRIC,
#     random_state = RANDOM_STATE,
#     verbose = 0,
#     direction = DIRECTION,
#     n_trials = N_TRIALS,
#     timeout = TIMEOUT,
#     n_jobs = N_JOBS
# )


# best_params_catb = optuna_catboost(
#     train_set = X,
#     target = y,
#     categorical = cat_cols,
#     cv_strategy = CV_STRATEGY,
#     cv_n_splits = CV_N_SPLITS,
#     n_estimators = N_ESTIMATORS,
#     early_stopping_round = EARLY_STOPPING_ROUND,
#     eval_metric = EVAL_METRIC,
#     random_state = RANDOM_STATE,
#     verbose = 0,
#     direction = DIRECTION,
#     n_trials = N_TRIALS,
#     timeout = TIMEOUT
# )


best_params_lgbm = {
    'learning_rate': 0.12612578622088044, 
    'max_depth': 11, 
    'lambda_l1': 5, 
    'lambda_l2': 10, 
    'min_gain_to_split': 2.2758924091155075, 
    'bagging_fraction': 0.9, 
    'bagging_freq': 1, 
    'feature_fraction': 0.9, 
    'subsample': 0.7266526707048253, 
    'colsample_bylevel': 0.4563300419462343, 
    'max_features': 'log2',
    'device': 'gpu'
}
best_params_xgb = {
    'max_depth': 10,
    'learning_rate': 0.09723697367097107,
    'subsample': 0.928217753576159,
    'colsample_bytree': 0.9998224783074757,
    'min_child_weight': 7,
    'gamma': 0.1644811935453648,
    'lambda': 70,
    'alpha': 10,
    'seed': 42,
    'eval_metric': 'auc',
    'n_jobs': -1,
    'device': 'cuda'
}
best_params_catb = {
    'learning_rate': 0.07537731882111856, 
    'subsample': 0.9481358585886993, 
    'bootstrap_type': 'Bernoulli',
    'task_type': 'GPU'
}


preds_lgbm_dftrain, preds_xgb_dftrain, preds_catb_dftrain, \
preds_lgbm_dftest, preds_xgb_dftest, preds_catb_dftest, \
score_lgbm, score_xgb, score_catb = final_lgbm_xgb_catb_train_fit(
    params_lgbm = best_params_lgbm,
    params_xgb = best_params_xgb,
    params_catb = best_params_catb,
    train_set = X,
    test_set = df_test,
    target = y,
    categorical = [cat_cols, True], 
    cv_strategy = CV_STRATEGY,
    cv_n_split = CV_N_SPLITS,
    n_estimators = N_ESTIMATORS,
    early_stopping_round = EARLY_STOPPING_ROUND,
    eval_metric = EVAL_METRIC, 
    random_state = RANDOM_STATE,    
    verbose = [-1, 0, 0],
    n_jobs = N_JOBS
)


summary_models_results(score_lgbm, score_xgb, score_catb)


weights, weighted_preds_dftrain, weighted_preds_dftest, score_weighted = weighted_preds(
    predictions_dftrain_to_weight = [preds_lgbm_dftrain, preds_xgb_dftrain, preds_catb_dftrain],
    predictions_dftest = [preds_lgbm_dftest, preds_xgb_dftest, preds_catb_dftest],
    target = y,
    n_trials = N_TRIALS_WEIGHT,
    timeout = TIMEOUT_WEIGHT
) 


best_params_model_blender_lgbm, preds_model_blender_lgbm_dftrain, preds_model_blender_lgbm_dftest, score_model_blender_lgbm = blended_lgbm_preds(
    train = [preds_lgbm_dftrain, preds_xgb_dftrain, preds_catb_dftrain, weighted_preds_dftrain],
    test = [preds_lgbm_dftest, preds_xgb_dftest, preds_catb_dftest, weighted_preds_dftest],
    target = y,
    cv_strategy = CV_STRATEGY,
    cv_n_splits = CV_N_SPLITS,
    n_estimators = N_ESTIMATORS,
    early_stopping_round = EARLY_STOPPING_ROUND,
    eval_metric = EVAL_METRIC,
    random_state = RANDOM_STATE,    
    verbose = -1,
    direction = DIRECTION,
    n_trials = N_TRIALS_BLENDER,
    timeout = TIMEOUT_BLENDER,
    n_jobs = N_JOBS
)


best_params_model_blender_xgb, preds_model_blender_xgb_dftrain, preds_model_blender_xgb_dftest, score_model_blender_xgb = blended_xgb_preds(
    train = [preds_lgbm_dftrain, preds_xgb_dftrain, preds_catb_dftrain, weighted_preds_dftrain],
    test = [preds_lgbm_dftest, preds_xgb_dftest, preds_catb_dftest, weighted_preds_dftest],
    target = y,
    cv_strategy = CV_STRATEGY,
    cv_n_splits = CV_N_SPLITS,
    n_estimators = N_ESTIMATORS,
    early_stopping_round = EARLY_STOPPING_ROUND,
    eval_metric = EVAL_METRIC,
    random_state = RANDOM_STATE,    
    verbose = 0,
    direction = DIRECTION,
    n_trials = N_TRIALS_BLENDER,
    timeout = TIMEOUT_BLENDER,
    n_jobs = N_JOBS
)


weights_2, weighted_preds_dftrain_2, weighted_preds_dftest_2, score_weighted_2 = weighted_preds(
    predictions_dftrain_to_weight = [weighted_preds_dftrain, preds_model_blender_lgbm_dftrain, preds_model_blender_xgb_dftrain],
    predictions_dftest = [weighted_preds_dftest, preds_model_blender_lgbm_dftest, preds_model_blender_xgb_dftest],
    target = y,
    n_trials = N_TRIALS_WEIGHT,
    timeout = TIMEOUT_WEIGHT
) 


mean_score_lgbm = np.round(np.mean(score_lgbm), 6)
mean_score_xgb = np.round(np.mean(score_xgb), 6)
mean_score_catb = np.round(np.mean(score_catb), 6)
mean_score_lgbm_blender = np.round(np.mean(score_model_blender_lgbm), 6)
mean_score_xgb_blender = np.round(np.mean(score_model_blender_xgb), 6)

std_score_lgbm = np.round(np.std(score_lgbm), 6)
std_score_xgb = np.round(np.std(score_xgb), 6)
std_score_catb = np.round(np.std(score_catb), 6)
std_score_lgbm_blender = np.round(np.std(score_model_blender_lgbm), 6)
std_score_xgb_blender = np.round(np.std(score_model_blender_xgb), 6)


scores = {
    '(L) LGBM score': [mean_score_lgbm, std_score_lgbm, preds_lgbm_dftest],
    '(X) XGB score': [mean_score_xgb, std_score_xgb, preds_xgb_dftest],
    '(C) CatB score': [mean_score_catb, std_score_catb, preds_catb_dftest],
    '(W) L_X_C weighted score': [np.round(score_weighted, 6), '-', weighted_preds_dftest],
    '(BL) L_X_C_W LGBM blender score': [mean_score_lgbm_blender, std_score_lgbm_blender, preds_model_blender_lgbm_dftest],
    '(BX) L_X_C_W XGB blender score': [mean_score_xgb_blender, std_score_xgb_blender, preds_model_blender_xgb_dftest],
    '(W2) W_BL_BX weighted score': [np.round(score_weighted_2, 6), '-', weighted_preds_dftest_2]
}

myTable = PrettyTable(['Model', 'Score', 'Std'])

for key in scores.keys():
    myTable.add_row([key, scores[key][0], scores[key][1]], divider = True)
myTable.align['Model'] = 'l'
myTable.align['Score'] = 'r'
myTable.align['Std'] = 'r'
print(myTable)

best_model = max(scores, key = scores.get)
print(f'\nBest model: {best_model}: {scores[best_model][0]}')


df_subm['loan_paid_back'] = scores[best_model][2]
df_subm.head()


df_subm.to_csv('submission.csv', index = False)

