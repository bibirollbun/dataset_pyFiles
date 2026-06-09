# NCAA Basketball Match Prediction Using Ensemble Learning with Optuna Optimization

import numpy as np
import pandas as pd
import glob
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

# Load data from Kaggle dataset
path = '/kaggle/input/march-machine-learning-mania-2025/**'
data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}

# Merge and preprocess datasets
teams = pd.concat([data['MTeams'], data['WTeams']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
sub = data['SampleSubmissionStage2']

# Combine season and tournament results
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)

# Generate matchup identifiers
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)

# Remove duplicate team seeds to ensure unique index
seeds = seeds.drop_duplicates(subset=['Season', 'TeamID'], keep='first')
seeds['Seed'] = seeds['Seed'].str[1:3].astype(float)

# Map team seeds to games
games['Team1Seed'] = games.merge(seeds, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')['Seed'].fillna(-1)
games['Team2Seed'] = games.merge(seeds, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')['Seed'].fillna(-1)

games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']

games = games.fillna(-1)

# Feature selection
features = ['SeedDiff']

# Generate target variable
games['Pred'] = games.apply(lambda r: 1. if r['WTeamID'] == r['Team1'] else 0., axis=1)

# Data scaling and missing value imputation
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X = games[features].fillna(-1)
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
y = games['Pred']

# Define base models
rf = RandomForestRegressor(n_estimators=200, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=200, objective='binary:logistic', random_state=42)

# Stacking ensemble model with Ridge as final estimator
estimators = [('rf', rf), ('gb', gb), ('xgb', xgb_model)]
st_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
st_model.fit(X_scaled, y)

# Model predictions and evaluation
pred = st_model.predict(X_scaled).clip(0.0001, 0.9999)
print('Log Loss:', log_loss(games['Pred'], pred))

# Preprocess test data for submission
sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
sub['Team1'] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
sub['Team2'] = sub['ID'].apply(lambda x: int(x.split('_')[2]))

sub['Team1Seed'] = sub.merge(seeds, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')['Seed'].fillna(-1)
sub['Team2Seed'] = sub.merge(seeds, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')['Seed'].fillna(-1)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']

# Generate predictions for test data
X_sub = sub[features].fillna(-1)
X_sub_imputed = imputer.transform(X_sub)
X_sub_scaled = scaler.transform(X_sub_imputed)
sub['Pred'] = st_model.predict(X_sub_scaled).clip(0.0001, 0.9999)

# Save submission file
sub[['ID', 'Pred']].to_csv('submission.csv', index=False)


