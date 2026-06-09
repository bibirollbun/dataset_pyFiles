# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
#import kagglehub
#kagglehub.login()


# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

#march_machine_learning_mania_2025_path = kagglehub.competition_download('march-machine-learning-mania-2025')

#print('Data source import complete.')


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


#from google.colab import drive
#drive.mount('/content/drive')


import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import *
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.ensemble import StackingRegressor, StackingClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


PATH = '/kaggle/input/march-machine-learning-mania-2025/**'


PATH


data = {p.split('/')[-1].split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(PATH)}


data


teams = pd.concat([data['MTeams'], data['WTeams']])
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
teams_spelling.columns = ['TeamID', 'TeamNameCount']
teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
del teams_spelling


season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
gcities = pd.concat([data['MGameCities'], data['WGameCities']])
seasons = pd.concat([data['MSeasons'], data['WSeasons']])


seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}
cities = data['Cities']
sub = data['SampleSubmissionStage2']
del data


season_cresults['ST'] = 'S'
season_dresults['ST'] = 'S'
tourney_cresults['ST'] = 'T'
tourney_dresults['ST'] = 'T'


games = pd.concat((season_dresults, tourney_dresults), axis = 0, ignore_index = True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})


games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'],r['LTeamID']]))), axis = 1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis = 1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis = 1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis = 1)


games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)


games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0] == r['WTeamID'] else 0., axis = 1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis = 1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)


games.info()


games.shape


pd.DataFrame(games.isna().sum())


games.describe(exclude = np.number).T


games.describe().T


c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
               'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl','LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']


gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]


games = games[games['ST'] == 'T']


sub['WLoc'] = 3

sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)

sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])

sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)

sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']

sub = sub.fillna(-1)


sub.head()


sub.shape


sub.info()


sub.describe(exclude = np.number).T


sub.describe().T


games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')


games[['IDTeams', 'IDTeams_c_score']]


games.describe(exclude = np.number).T


games.describe().T


col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST',
                                             'Team1', 'Team2',
                                             'IDTeams', 'IDTeam1', 'IDTeam2',
                                             'WTeamID', 'WScore',
                                             'LTeamID', 'LScore',
                                             'NumOT',
                                             'Pred',
                                             'ScoreDiff',
                                             'ScoreDiffNorm','WLoc'] + c_score_col]


pd.DataFrame(games.isna().sum() > 0).value_counts()


#! pip install skimpy


#from skimpy import skim
#X = games[col].fillna(-1)
#skim(X)


# Selecting training data
X = games[col].fillna(-1)
y = games[['Season', 'Pred']]
sub_X = sub[col].fillna(-1)
seasons = X['Season'].unique()
CV = []
def kfold_Classifier(df, model, df_test = None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Pipeline = fillna + ss + model
    pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'mean')),
                         ('scaler', StandardScaler()),
                         ('model', model)])

    # Fit model
    pipeline.fit(X_train, y_train['Pred'])

    # Predictions
    pred_games = pipeline.predict_proba(X_val)[:, 1].clip(0.001, 0.999) # Вероятность класса 1

    # Interactive cross val
    score_val = brier_score_loss(y_val['Pred'], pred_games)
    loss = log_loss(y_val['Pred'].values, pred_games)
    CV.append(loss)

    # Test Predictions
    if df_test is not None:
        pred_sub = pipeline.predict_proba(df_test)[:, 1].clip(0.001, 0.999)

    print(f'\n Local CV is {np.mean(CV):.3f}')
    return pred_games, y_val['Pred'].values


# Selecting training data
X = games[col].fillna(-1)
y = games[['Season', 'Pred']]
sub_X = sub[col].fillna(-1)
seasons = X['Season'].unique()
CV = []
def kfold_Classifier_Torch(df, model, df_test = None):
    # Import Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Pipeline = fillna + ss + model
    pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'mean')),
                         ('scaler', StandardScaler()),
                         ('model', model)])

    #to numpy(np.float32)
    X_train = X_train.astype('float32').to_numpy().astype(np.float32)
    X_val = X_val.astype('float32').to_numpy().astype(np.float32)
    y_train = y_train['Pred'].astype('float32').to_numpy().astype(np.float32)
    y_val = y_val['Pred'].astype('float32').to_numpy().astype(np.float32)

    # Fit model
    pipeline.fit(X_train, y_train)

    # Predictions
    pred_games = pipeline.predict_proba(X_val)[:, 1].clip(0.001, 0.999) # Вероятность класса 1

    # Interactive cross val
    score_val = brier_score_loss(y_val, pred_games)
    loss = log_loss(y_val.values, pred_games)
    CV.append(loss)

    # Test Predictions
    if df_test is not None:
        pred_sub = pipeline.predict_proba(df_test)[:, 1].clip(0.001, 0.999)

    print(f'\n Local CV is {np.mean(CV):.3f}')
    return pred_games, y_val.values


def objective(trial):
    cat_params = dict(
        iterations=trial.suggest_int("iterations", 100, 1000, step = 100),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        depth=trial.suggest_int("depth", 7, 15, step = 2),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bootstrap_type='Bernoulli',  # Аналог bagging_temperature для классификации
        subsample=trial.suggest_float('subsample', 0.5, 1.0),  # Контролирует бэггинг
        thread_count = -1,  # Аналог task_type='CPU'
        early_stopping_rounds=200,
        verbose=False,
        loss_function='Logloss'  # Добавлена целевая функция для классификации
    )

    model = CatBoostClassifier(**cat_params)
    y_pred , y_val = kfold_Classifier(X, model, sub_X)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42) #NSGAIISampler
study_1 = optuna.create_study(direction="minimize", sampler = sampler)
study_1.optimize(objective, n_trials=10)
'''


best_params_1 = {'iterations': 700, 'learning_rate': 0.0008612579192594886, 'depth': 11, 'l2_leaf_reg': 0.002931587042311714, 'subsample': 0.5924272277627636,
                 'bootstrap_type': 'Bernoulli', 'thread_count': -1, 'early_stopping_rounds': 200,
                 'verbose': False, 'loss_function': 'Logloss'}
final_model_1 = CatBoostClassifier(**best_params_1)


#plot_optimization_history(study_1)


#plot_param_importances(study_1)


def objective(trial):
    xgb_params = dict(
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50),
        max_depth = trial.suggest_int("max_depth", 7, 15, step=2),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        reg_alpha = trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        subsample = trial.suggest_float("subsample", 0.5, 0.9),
        gamma = trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.22, 0.9),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 3),
        reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,      # Отключаем устаревший функционал
        tree_method='hist',           # Оптимизация для CPU
        enable_categorical=False      # Отключаем автоматическую обработку категорий
    )

    model = XGBClassifier(**xgb_params)
    y_pred , y_val = kfold_Classifier(X, model, sub_X)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42) #NSGAIISampler
study_2 = optuna.create_study(direction="minimize", sampler = sampler)
study_2.optimize(objective, n_trials=10)
'''


best_params_2 = {'n_estimators': 400, 'max_depth': 9, 'learning_rate': 0.012172847081122434, 'reg_alpha': 5.065486063975357e-06, 'subsample': 0.8208787923016159,
                 'gamma': 0.0014096175149815868, 'colsample_bytree': 0.8910831168883517,
                 'min_child_weight': 3, 'reg_lambda': 9.853225172032558e-06, 'objective': 'binary:logistic', 'eval_metric': 'logloss',
                 'use_label_encoder': False, 'tree_method': 'hist', 'enable_categorical': False}
final_model_2 = XGBClassifier(**best_params_2)


#plot_optimization_history(study_2)


#plot_param_importances(study_2)


# Gradient boosting regressor does not work, because on the predicted data it produces nan values, I could not figure it out
def objective(trial):
    gbm_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step = 50),
        max_depth=trial.suggest_int("max_depth", 6, 16, step = 2),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        subsample=trial.suggest_float("subsample", 0.40, 0.90),
        min_samples_split=trial.suggest_float("min_samples_split", 0.3, 0.9),
    )
    model = GradientBoostingClassifier(**gbm_params)
    y_pred, y_val = kfold_Classifier(X, model, sub_X)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42) #NSGAIISampler
study_3 = optuna.create_study(direction="minimize", sampler = sampler)
study_3.optimize(objective, n_trials=10)
'''


best_params_3 = {'n_estimators': 250, 'max_depth': 8, 'learning_rate': 0.01120760621186057, 'subsample': 0.615972509321058,
                 'min_samples_split': 0.47473748411882516}
final_model_3 = GradientBoostingClassifier(**best_params_3)


#plot_optimization_history(study_3)


#plot_param_importances(study_3)


def objective(trial):
    lgbm_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
        max_depth=trial.suggest_int("max_depth", 6, 16, step=2),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        subsample=trial.suggest_float("subsample", 0.4, 0.9),
        max_bin=trial.suggest_int("max_bin", 100, 300, step=20),
        feature_fraction=trial.suggest_float("feature_fraction", 0.1, 0.5),
        num_leaves=trial.suggest_int("num_leaves", 20, 100, step=10),  # Добавляем параметр num_leaves
        min_child_samples=trial.suggest_int("min_child_samples", 10, 50, step=10),  # Добавляем параметр min_child_samples
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1e-1, log = True),  # Добавляем параметр reg_alpha
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1e-1, log = True),  # Добавляем параметр reg_lambda
        objective='binary')  # Указываем, что задача бинарной классификации

    model = LGBMClassifier(**lgbm_params, verbosity = -1)
    y_pred , y_val = kfold_Classifier(X, model, sub_X)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42) #NSGAIISampler
study_4 = optuna.create_study(direction="minimize", sampler = sampler)
study_4.optimize(objective, n_trials=10)
'''


best_params_4 = {'n_estimators': 650, 'max_depth': 6, 'learning_rate': 0.00383962929980417, 'subsample': 0.5831809216468459, 'max_bin': 200,
                 'feature_fraction': 0.41407038455720546, 'num_leaves': 30, 'min_child_samples': 30,
                 'reg_alpha': 0.015304852121831466, 'reg_lambda': 0.001238513729886093, 'objective': 'binary', 'verbosity': -1}
final_model_4 = LGBMClassifier(**best_params_4)


#plot_optimization_history(study_4)


#plot_param_importances(study_4)


def objective(trial):
    laso_params = dict(
        penalty='l1',
        solver='liblinear',
        C=trial.suggest_float("C", 1e-4, 1, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 1000))

    model = LogisticRegression(**laso_params)
    y_pred , y_val = kfold_Classifier(X, model, sub_X)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42) #NSGAIISampler
study_5 = optuna.create_study(direction="minimize", sampler = sampler)
study_5.optimize(objective, n_trials=25)
'''


best_params_5 = {'C': 0.027308349581482736, 'max_iter': 747, 'penalty': 'l1', 'solver': 'liblinear'}
final_model_5 = LogisticRegression(**best_params_5)


#plot_optimization_history(study_5)


#plot_param_importances(study_5)


def objective(trial):
    ridge_params = dict(
        penalty='l2',
        solver='liblinear',
        C=trial.suggest_float("C", 1e-4, 1.0, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 1000))

    model = LogisticRegression(**ridge_params)
    y_pred , y_val = kfold_Classifier(X, model, sub_X)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42) #NSGAIISampler
study_6 = optuna.create_study(direction="minimize", sampler = sampler)
study_6.optimize(objective, n_trials=10)
'''


best_params_6 = {'C': 0.005342937261279773, 'max_iter': 362, 'penalty': 'l2', 'solver': 'liblinear'}
final_model_6 = LogisticRegression(**best_params_6)


#plot_optimization_history(study_6)


#plot_param_importances(study_6)


def objective(trial):
    elastic_params = dict(
        penalty='elasticnet',
        solver='saga',
        C=trial.suggest_float("C", 1e-4, 1.0, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        max_iter=trial.suggest_int("max_iter", 100, 1000))

    model = LogisticRegression(**elastic_params)
    y_pred , y_val = kfold_Classifier(X, model, sub_X)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42) #NSGAIISampler
study_7 = optuna.create_study(direction="minimize", sampler = sampler)
study_7.optimize(objective, n_trials=25)
'''


best_params_7 = {'C': 0.0174986549461868, 'l1_ratio': 0.756297251336319, 'max_iter': 547, 'penalty': 'elasticnet', 'solver': 'saga'}
final_model_7 = LogisticRegression(**best_params_7)


#plot_optimization_history(study_7)


#plot_param_importances(study_7)


#! pip install skorch


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import skorch
from skorch.callbacks import EarlyStopping # Not Used

class NeuralNetBinaryClassifier(skorch.NeuralNetBinaryClassifier):
    def fit(self, X, y, **fit_params):
        return super().fit(X, np.asarray(y, dtype=np.float32), **fit_params)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, dropout_1):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()

        # Входной слой
        self.layers.append(nn.Linear(input_dim, hidden_dim_1))
        self.layers.append(nn.BatchNorm1d(hidden_dim_1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_1))

        self.layers.append(nn.Linear(hidden_dim_1, hidden_dim_2))
        #self.layers.append(nn.BatchNorm1d(hidden_dim_2))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim_2, hidden_dim_3))
        #self.layers.append(nn.BatchNorm1d(hidden_dim_3))
        self.layers.append(nn.ReLU())

        # Выходной слой
        self.output = nn.Linear(hidden_dim_3, 1)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return torch.sigmoid(self.output(X)).squeeze()
'''


'''
df = tourney_results_df_copy
features = ["SeedA", "SeedB", 'WinRatioA', 'GapAvgA', 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff']
seasons = df['Season'].unique()
target = "WinA"
pred_final_list = []
for season in seasons[1:]:
    #Iterator
    df_train = df[df['Season'] < season].reset_index(drop=True).copy()
    df_val = df[df['Season'] == season].reset_index(drop=True).copy()
    df_test = test_df.copy()
    #Fillna
    df_train.fillna(-1, inplace=True)
    df_val.fillna(-1, inplace=True)
    df_test.fillna(-1, inplace=True)
    #Scaling
    df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)
    new_df = pd.concat([df_train, df_val], ignore_index = True)
    print("Train", season, df_train['Season'].unique())
    print("Valid", season, df_val['Season'].unique())

# Проверка входных данных (skorch работает как с numpy(), так и с tensor(), но не с DF)
X_train = df_train[features].astype('float32').to_numpy().astype(np.float32)
y_train = df_train[target].astype('float32').to_numpy().astype(np.float32)
X_val = df_val[features].astype('float32').to_numpy().astype(np.float32)
y_val = df_val[target].astype('float32').to_numpy().astype(np.float32)
X_full = df[features].astype('float32').to_numpy().astype(np.float32)
y_full = df[target].astype('float32').to_numpy().astype(np.float32)
'''


def objective(trial):
    # Параметры для нейросети
    nn_params = {
        'module__input_dim': X_train.shape[1],
        'module__hidden_dim_1': trial.suggest_int("hidden_dim_1", 16, 64, step = 1),
        'module__hidden_dim_2': trial.suggest_int("hidden_dim_2", 10, 32, step = 1),
        'module__hidden_dim_3': trial.suggest_int("hidden_dim_3", 10, 32, step = 1),
        'module__dropout_1': trial.suggest_float("dropout_1", 0.1, 0.5),
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'optimizer__weight_decay': trial.suggest_float("weight_decay", 1e-4, 1e-2, log = True),
    }

    # Создаем модель через skorch
    model = NeuralNetBinaryClassifier(
        module=SimpleNN,
        **nn_params,
        optimizer=torch.optim.SGD,
        criterion=nn.BCELoss,
        batch_size=128,
        max_epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose = 0)

    y_pred , y_test = kfold_Classifier_Torch(X, model)
    score = brier_score_loss(y_test, y_pred)
    return score


'''
# Best - 0.17
sampler = TPESampler(seed=42)
study_8 = optuna.create_study(direction="minimize", sampler=sampler)
study_8.optimize(objective, n_trials=20)
'''


#plot_optimization_history(study_8)


#plot_param_importances(study_8)


#study_8.best_params


'''
model_args = {
    # Фиксированные параметры, не участвующие в оптимизации
    'module__input_dim': X_train.shape[1],
    'optimizer': torch.optim.Adam,
    'criterion': nn.BCELoss,
    'batch_size': 512,  # Если batch_size не был параметром Optuna
    'max_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'verbose': 0
}

# Добавляем оптимизированные параметры с правильными префиксами
for key, value in study_8.best_params.items():
    if key.startswith(('hidden_dim_', 'dropout_')):
        model_args[f'module__{key}'] = value
    elif key == 'weight_decay':
        model_args['optimizer__weight_decay'] = value
    else:
        model_args[key] = value  #lr

best_nn = NeuralNetBinaryClassifier(
    SimpleNN,
    **model_args)
'''


#model_args


def objective(trial):
    ET_params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 15, 25, step = 2),
        min_samples_split=trial.suggest_int("min_samples_split", 3, 7, step = 1),
        max_features = 'sqrt',
        random_state = 42)

    model = ExtraTreesClassifier(**ET_params)
    y_pred , y_val = kfold_Classifier(tourney_results_df_copy, model, test_df)
    score = brier_score_loss(y_val, y_pred)
    return score


'''
sampler = TPESampler(seed=42)
study_9 = optuna.create_study(direction="minimize", sampler = sampler)
study_9.optimize(objective, n_trials=10)
'''


best_params_9 = {'n_estimators': 250, 'max_depth': 19, 'min_samples_split': 3, 'max_features': 'sqrt', 'random_state': 42}
final_model_9 = ExtraTreesClassifier(**best_params_9)


#plot_optimization_history(study_9)


#plot_param_importances(study_9)


# Добавление в стек
estimators_1 = [("final_model_5", final_model_5),
                ("final_model_6", final_model_6),
                ("final_model_7", final_model_7)]
stacking_classifier_1 = StackingClassifier(estimators=estimators_1, final_estimator = LogisticRegression(C = 0.0001))


'''
# We will search for alpha using GridSearchCV (best C = 1)
grid_params = {'final_estimator__C': [0.0001, 0.01, 1, 10]}
ss = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)
stack_search_1 = GridSearchCV(stacking_classifier_1, param_grid = grid_params,
                            scoring = make_scorer(brier_score_loss, greater_is_better=False), cv = ss, n_jobs = -1)
stack_search_1.fit(X_full, y_full)
'''


#stack_search_1.best_params_


estimators_1 = [("final_model_5", final_model_5),
                ("final_model_6", final_model_6),
                ("final_model_7", final_model_7)]
stacking_classifier_1 = StackingClassifier(estimators=estimators_1, final_estimator = LogisticRegression(C = 0.001))
estimators_2 = [
            #('PyTorch_nn', best_nn), bad predictions with this nn
            ("final_model_1", final_model_1),
            ("final_model_2", final_model_2),
            ("final_model_3", final_model_3),
            ("final_model_4", final_model_4),
            #("final_model_9", final_model_9),
            ("stacking_classifier_1", stacking_classifier_1)]
stacking_classifier_2 = StackingClassifier(estimators=estimators_2, final_estimator = LogisticRegression(C = 0.001, solver = 'newton-cg'))


'''
# 1. GridSearchCV (long-long)
grid_params_2 = {'final_estimator__C': [0.0001, 0.01, 1, 10]}
ss_2 = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)
stack_search_2 = GridSearchCV(stacking_classifier_2, param_grid = grid_params_2,
                            scoring = make_scorer(brier_score_loss, greater_is_better=False), cv = ss_2, n_jobs = -1)
stack_search_2.fit(X_full, y_full)
'''


# 1. GridSearchCV. C - best
#stack_search_2.best_params_


X_full = X.drop(['IDTeams_c_score'], axis = 1)
stacking_classifier_2.fit(X_full, y['Pred'])
pred_final_0 = stacking_classifier_2.predict_proba(X_full)[:, 1].clip(0.001, 0.999)

score = brier_score_loss(y['Pred'], pred_final_0)
print(f"Score on Train Data (full dataset) is - {score}")

IsReg = IsotonicRegression(out_of_bounds="clip")
IsReg.fit(pred_final_0, y['Pred'])


pred_final_1 = stacking_classifier_2.predict_proba(sub_X.drop(['IDTeams_c_score'], axis = 1))[:, 1].clip(0.001, 0.999)
pred_final_2 = IsReg.transform(pred_final_1)


# Распределение целевой переменной на обучающем датасете
# Подсчет количества наблюдений для каждого класса
counts = y['Pred'].value_counts()
# Построение графика
plt.bar(counts.index, counts.values, color=['blue', 'orange'])
plt.title('Распределение целевой переменной')
plt.xlabel('Целевая переменная')
plt.ylabel('Количество наблюдений')
plt.xticks([0, 1], ['0', '1'])
plt.show()


# Подсчет количества наблюдений для каждого класса
counts = y['Pred'].value_counts()
# Построение круговой диаграммы
plt.pie(counts, labels=['0', '1'], autopct='%1.1f%%', colors=['blue', 'orange'])
plt.title('Распределение целевой переменной')
plt.show()


df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")
submission_df_stack = pd.DataFrame({
                              'ID': df['ID'],
                              'Pred': pred_final_2})

# Saving to csv
submission_df_stack.to_csv('submission.csv', index=False)


plt.hist(submission_df_stack['Pred'], bins=10, color='red', alpha=0.7);




