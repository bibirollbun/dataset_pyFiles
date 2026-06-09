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


import numpy as np
import pandas as pd
from sklearn import *
import glob



path = '/kaggle/input/march-machine-learning-mania-2025/**'
data = {p.split('/')[-1].split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}
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
#games = pd.concat((season_cresults, tourney_cresults), axis=0, ignore_index=True)
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)

c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

games = games[games['ST']=='T']

sub['WLoc'] = 3
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
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

games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col]


# !pip install plotly


import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'
answer = pd.read_csv('/kaggle/input/lb0-00000-answer/answer.csv')

fig = px.histogram(
    answer, 
    x='Pred',
    title='Distribution of Predictions',
    labels={'Pred': 'Prediction Value', 'count': 'Count'},
    text_auto=True,
    color='Pred',
    color_discrete_map={0: '#FFA500', 1: '#007FFF'}
)

fig.update_layout(
    bargap=0.2,
    showlegend=False,
    xaxis=dict(tickmode='array', tickvals=[0, 1]),
    yaxis_title='Count'
)

fig.show()


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, brier_score_loss
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import json
from datetime import datetime
import optuna

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X = games[col].fillna(-1)
sub_X = sub[col].fillna(-1)
print(sub_X[sub_X['Season'] >= 2021].shape)
print(X[X['Season'] >= 2021].shape)


# take only 2024 as validation
X = X[X['Season'] < 2024]
games = games[games['Season'] < 2024]
sub_X_test = sub_X.copy()
sub_X = sub_X[sub_X['Season'] == 2024]
answer['Season'] = answer['ID'].map(lambda x: x.split('_')[0])
answer = answer[answer['Season'] == '2024']
val_answer = answer['Pred']

fig = px.histogram(
    answer, 
    x='Pred',
    title='Distribution of Predictions',
    labels={'Pred': 'Prediction Value', 'count': 'Count'},
    text_auto=True,
    color='Pred',
    color_discrete_map={0: '#FFA500', 1: '#007FFF'}
)

fig.update_layout(
    bargap=0.2,
    showlegend=False,
    xaxis=dict(tickmode='array', tickvals=[0, 1]),
    yaxis_title='Count'
)

fig.show()

print(sub_X.shape, val_answer.shape, sub_X_test.shape)


X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
sub_X_imputed = imputer.transform(sub_X)
sub_X_scaled = scaler.transform(sub_X_imputed)


def objective_et_rf(trial):
    et_params = {
        'et__n_estimators': trial.suggest_int('et__n_estimators', 200, 370),  # Reduced upper bound
        'et__max_depth': trial.suggest_int('et__max_depth', 10, 15),  # Reduced max depth
        'et__min_samples_split': trial.suggest_int('et__min_samples_split', 2, 5),  # Increased min samples
        'et__max_features': trial.suggest_categorical('et__max_features', ['sqrt', 'log2']),  # Removed None option
        'et__criterion': trial.suggest_categorical('et__criterion', ['squared_error', 'absolute_error']),
        'et__n_jobs': -1,
        'et__random_state': 42
    }
    rf_params = {
        'rf__n_estimators': trial.suggest_int('rf__n_estimators',200, 370),  # Moderate number of trees
        'rf__max_depth': trial.suggest_int('rf__max_depth', 10, 20),  # Limited depth
        'rf__min_samples_split': trial.suggest_int('rf__min_samples_split', 2, 5),  # Higher min samples
        'rf__max_features': trial.suggest_categorical('rf__max_features', ['sqrt', 'log2']),  # Restrict features
        'rf__bootstrap': True,  # Enable bootstrapping
        'rf__n_jobs': -1,
        'rf__random_state': 42
    }
    rf_params = {k.replace('rf__', ''): v for k, v in rf_params.items() if k.startswith('rf__')}
    et_params = {k.replace('et__', ''): v for k, v in et_params.items() if k.startswith('et__')}
    et = ExtraTreesRegressor(**et_params)
    rf = RandomForestRegressor(**rf_params)
    voting_regressor = VotingRegressor(estimators=[('et', et), ('rf', rf)])
    model = Pipeline(steps=[
        ('voting', voting_regressor)
    ])
    model.fit(X_scaled, games['Pred'])
    train_pred = model.predict(sub_X_scaled).clip(0.001, 0.999)
    # ir = IsotonicRegression(out_of_bounds='clip')
    # ir.fit(train_pred, games['Pred'])
    # cv_scores = cross_val_score(model, sub_X_scaled, val_answer, cv=5, scoring="neg_mean_squared_error")
    loss = mean_squared_error(train_pred, val_answer)
    return loss


def objective_et_xgb(trial):
    et_params = {
        'et__n_estimators': trial.suggest_int('et__n_estimators', 200, 300),
        'et__max_depth': trial.suggest_int('et__max_depth', 10, 20),
        'et__min_samples_split': trial.suggest_int('et__min_samples_split', 2, 4),
        'et__max_features': trial.suggest_categorical('et__max_features', ['sqrt', 'log2']),
        'et__criterion': trial.suggest_categorical('et__criterion', ['squared_error', 'absolute_error']),
        'et__n_jobs': -1,
        'et__random_state': 42
    }
    
    xgb_params = {
        'xgb__n_estimators': trial.suggest_int('xgb__n_estimators', 200, 300),
        'xgb__max_depth': trial.suggest_int('xgb__max_depth', 6, 9),  # Reduced depth
        'xgb__learning_rate': trial.suggest_float('xgb__learning_rate', 0.01, 0.1),  # Lower learning rate
        'xgb__min_child_weight': trial.suggest_int('xgb__min_child_weight', 1, 7),  # Prevent overfitting
        'xgb__gamma': trial.suggest_float('xgb__gamma', 0.001, 1.0),  # Min loss reduction
        'xgb__reg_lambda': trial.suggest_float('xgb__reg_lambda', 0.001, 10.0, log=True),  # L2 regularization
        'xgb__reg_alpha': trial.suggest_float('xgb__reg_alpha', 0.001, 1.0),  # L1 regularization
        'xgb__subsample': trial.suggest_float('xgb__subsample', 0.6, 0.8),  # Lower subsample
        'xgb__colsample_bytree': trial.suggest_float('xgb__colsample_bytree', 0.6, 0.8),  # Lower feature sampling
        'xgb__random_state': 42,
        'device':"cuda",
        'xgb__tree_method': 'hist'
    }

    xgb_params = {k.replace('xgb__', ''): v for k, v in xgb_params.items() if k.startswith('xgb__')}
    et_params = {k.replace('et__', ''): v for k, v in et_params.items() if k.startswith('et__')}
    
    et = ExtraTreesRegressor(**et_params)
    xgb = XGBRegressor(**xgb_params)
    
    voting_regressor = VotingRegressor(estimators=[('et', et), ('xgb', xgb)])
    model = Pipeline(steps=[
        ('voting', voting_regressor)
    ])
    
    model.fit(X_scaled, games['Pred'])
    train_pred = model.predict(sub_X_scaled).clip(0.001, 0.999)
    loss = mean_squared_error(train_pred, val_answer)
    return loss


def objective_et_xgb_lgbm_cat(trial):
    et_params = {
        'et__n_estimators': trial.suggest_int('et__n_estimators', 200, 300),
        'et__max_depth': trial.suggest_int('et__max_depth', 10, 20),
        'et__min_samples_split': trial.suggest_int('et__min_samples_split', 2, 4),
        'et__max_features': trial.suggest_categorical('et__max_features', ['sqrt', 'log2']),
        'et__criterion': trial.suggest_categorical('et__criterion', ['squared_error', 'absolute_error']),
        'et__n_jobs': -1,
        'et__random_state': 42
    }
    
    xgb_params = {
        'xgb__n_estimators': trial.suggest_int('xgb__n_estimators', 200, 300),
        'xgb__max_depth': trial.suggest_int('xgb__max_depth', 6, 9),  # Reduced depth
        'xgb__learning_rate': trial.suggest_float('xgb__learning_rate', 0.01, 0.1),  # Lower learning rate
        'xgb__min_child_weight': trial.suggest_int('xgb__min_child_weight', 1, 7),  # Prevent overfitting
        'xgb__gamma': trial.suggest_float('xgb__gamma', 0.001, 1.0),  # Min loss reduction
        'xgb__reg_lambda': trial.suggest_float('xgb__reg_lambda', 0.001, 10.0, log=True),  # L2 regularization
        'xgb__reg_alpha': trial.suggest_float('xgb__reg_alpha', 0.001, 1.0),  # L1 regularization
        'xgb__subsample': trial.suggest_float('xgb__subsample', 0.6, 0.8),  # Lower subsample
        'xgb__colsample_bytree': trial.suggest_float('xgb__colsample_bytree', 0.6, 0.8),  # Lower feature sampling
        'xgb__random_state': 42,
        'device':"cuda",
        'xgb__tree_method': 'hist'
    }

    lgb_params = {
        'lgb__n_estimators': trial.suggest_int('lgb__n_estimators', 200, 300),
        'lgb__max_depth': trial.suggest_int('lgb__max_depth', 6, 9),
        'lgb__learning_rate': trial.suggest_float('lgb__learning_rate', 0.01, 0.1),
        'lgb__num_leaves': trial.suggest_int('lgb__num_leaves', 20, 50),
        'lgb__subsample': trial.suggest_float('lgb__subsample', 0.6, 0.8),
        'lgb__colsample_bytree': trial.suggest_float('lgb__colsample_bytree', 0.6, 0.8),
        'lgb__reg_alpha': trial.suggest_float('lgb__reg_alpha', 0.001, 1.0),
        'lgb__reg_lambda': trial.suggest_float('lgb__reg_lambda', 0.001, 10.0, log=True),
        'lgb__random_state': 42,
        'lgb__n_jobs': -1
    }

    cat_params = {
        'cat__iterations': trial.suggest_int('cat__iterations', 200, 300),
        'cat__depth': trial.suggest_int('cat__depth', 6, 9),
        'cat__learning_rate': trial.suggest_float('cat__learning_rate', 0.01, 0.1),
        'cat__l2_leaf_reg': trial.suggest_float('cat__l2_leaf_reg', 0.001, 10.0, log=True),
        'cat__random_strength': trial.suggest_float('cat__random_strength', 0.1, 10.0),
        'cat__subsample': trial.suggest_float('cat__subsample', 0.6, 0.8),
        'cat__random_state': 42,
        'cat__verbose': False
    }
    xgb_params = {k.replace('xgb__', ''): v for k, v in xgb_params.items() if k.startswith('xgb__')}
    et_params = {k.replace('et__', ''): v for k, v in et_params.items() if k.startswith('et__')}
    lgb_params = {k.replace('lgb__', ''): v for k, v in lgb_params.items() if k.startswith('lgb__')}
    cat_params = {k.replace('cat__', ''): v for k, v in cat_params.items() if k.startswith('cat__')}

    # Initialize models
    et = ExtraTreesRegressor(**et_params)
    xgb = XGBRegressor(**xgb_params)
    lgb = LGBMRegressor(**lgb_params)
    cat = CatBoostRegressor(**cat_params)

    # Create voting regressor with all models
    voting_regressor = VotingRegressor(estimators=[
        ('et', et), 
        ('xgb', xgb), 
        ('lgb', lgb), 
        ('cat', cat)
    ])
    
    model = Pipeline(steps=[
        ('voting', voting_regressor)
    ])

    model.fit(X_scaled, games['Pred'])
    train_pred = model.predict(sub_X_scaled).clip(0.001, 0.999)
    loss = mean_squared_error(train_pred, val_answer)
    return loss


optuna.logging.set_verbosity(optuna.logging.WARNING) # hide output
def run_opt(obj_func, name, trials=20):
    study = optuna.create_study(direction='minimize')
    study.optimize(obj_func, n_trials=trials,show_progress_bar=True)
    best_params = study.best_params
    best_params['score'] = study.best_trial.value
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = 'best_params_'+ name + f'_{timestamp}.json'

    with open(filename, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(best_params)
    return best_params


best_params = run_opt(objective_et_xgb, 'et_xgb')


from sklearn.isotonic import IsotonicRegression
xgb_best_params = {k.replace('xgb__', ''): v for k, v in best_params.items() if k.startswith('xgb__')}
et_best_params = {k.replace('et__', ''): v for k, v in best_params.items() if k.startswith('et__')}

et = ExtraTreesRegressor(**et_best_params)
xgb = XGBRegressor(**xgb_best_params)


voting_regressor = VotingRegressor(estimators=[('et', et), ('xgb', xgb)])
pipe = Pipeline(steps=[
    ('voting', voting_regressor)
])


sub_X_test_imputed = imputer.transform(sub_X_test)
sub_X_test_scaled = scaler.transform(sub_X_test_imputed)

pipe.fit(X_scaled, games['Pred'])


pred = pipe.predict(sub_X_test_scaled).clip(0.001, 0.999)
train_pred = pipe.predict(X_scaled).clip(0.001, 0.999)


ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(train_pred, games['Pred'])
sub['Pred'] = pred


sub[['ID', 'Pred']].to_csv('submission.csv', index=False)
sub[['ID', 'Pred']].head()


sub['Pred'] = sub['Pred'].round(1)
fig = px.histogram(
    sub, 
    x='Pred',
    title='Distribution of Predictions',
    labels={'Pred': 'Prediction Value', 'count': 'Count'},
    text_auto=True,
    color='Pred',
    color_discrete_map={0: '#FFA500', 1: '#007FFF'}
)

fig.update_layout(
    bargap=0.2,
    showlegend=False,
    xaxis=dict(tickmode='array', tickvals=[0, 1]),
    yaxis_title='Count'
)

fig.show()


val_pred = pipe.predict(sub_X_scaled).clip(0.001, 0.999)
val_pred = pd.DataFrame({
    "Pred": val_pred.round(1)
})


fig = px.histogram(
    val_pred, 
    x='Pred',
    title='Distribution of Predictions',
    labels={'Pred': 'Prediction Value', 'count': 'Count'},
    text_auto=True,
    color='Pred',
    color_discrete_map={0: '#FFA500', 1: '#007FFF'}
)

fig.update_layout(
    bargap=0.2,
    showlegend=False,
    xaxis=dict(tickmode='array', tickvals=[0, 1]),
    yaxis_title='Count'
)

fig.show()

