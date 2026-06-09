import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb


import warnings
warnings.filterwarnings('ignore')


DATA_PATH = "/kaggle/input/march-machine-learning-mania-2025"

teams = pd.read_csv(f"{DATA_PATH}/MTeams.csv")
seasons = pd.read_csv(f"{DATA_PATH}/MSeasons.csv")
seeds = pd.read_csv(f"{DATA_PATH}/MNCAATourneySeeds.csv")
regular_results = pd.read_csv(f"{DATA_PATH}/MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv(f"{DATA_PATH}/MNCAATourneyCompactResults.csv")
detailed_results = pd.read_csv(f"{DATA_PATH}/MRegularSeasonDetailedResults.csv")
massey_ordinals = pd.read_csv(f"{DATA_PATH}/MMasseyOrdinals.csv")


print(teams.head())
print(seasons.head())
print(seeds.head())
print(regular_results.head())
print(tourney_results.head())
print(detailed_results.head())
print(massey_ordinals.head())


print(teams.shape)
print(seasons.shape)
print(seeds.shape)
print(regular_results.shape)
print(tourney_results.shape)
print(detailed_results.shape)
print(massey_ordinals.shape)


# Checking for null values

print(teams.isnull().sum())
print(seeds.isnull().sum())
print(regular_results.isnull().sum())
print(tourney_results.isnull().sum())
print(detailed_results.isnull().sum())
print(massey_ordinals.isnull().sum())


# Distribution of team's rankings
plt.figure(figsize=(10, 6))
sns.histplot(massey_ordinals['OrdinalRank'], bins=50)
plt.title('Distribution of Team Rankings')
plt.xlabel('Ranking')
plt.ylabel('Frequency')
plt.show()


win_loss_ratio = regular_results.groupby('Season')[['WScore', 'LScore']].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=win_loss_ratio, x='Season', y='WScore', label='Winning Score')
sns.lineplot(data=win_loss_ratio, x='Season', y='LScore', label='Losing Score')
plt.title('Average Winning and Losing Scores by Season')
plt.xlabel('Season')
plt.ylabel('Average Score')
plt.legend()
plt.show()


def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

seeds['SeedInt'] = seeds['Seed'].apply(seed_to_int)
seeds.head()


# Merging seeds with tournament results
tourney_results = pd.merge(tourney_results, seeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
tourney_results.rename(columns={'SeedInt': 'WSeed'}, inplace=True)
tourney_results = pd.merge(tourney_results, seeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
tourney_results.rename(columns={'SeedInt': 'LSeed'}, inplace=True)
tourney_results.head()


tourney_results['SeedDiff'] = tourney_results['WSeed'] - tourney_results['LSeed']
tourney_results.head()


# Calculating win rates from regular season results
regular_results['WScore_diff'] = regular_results['WScore'] - regular_results['LScore']
win_diff = regular_results.groupby(['Season', 'WTeamID'])['WScore_diff'].sum().reset_index()
win_diff.columns = ['Season', 'TeamID', 'WinDiff']
win_diff.head()


tourney_results.columns


# Merging win_diff with tourney results for WTeamID
tourney_results = pd.merge(tourney_results, win_diff, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
tourney_results.rename(columns={'WinDiff': 'WWinDiff', 'TeamID': 'TeamID_W'}, inplace=True)

# Merging win_diff with tourney results for LTeamID
tourney_results = pd.merge(tourney_results, win_diff, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
tourney_results.rename(columns={'WinDiff': 'LWinDiff', 'TeamID': 'TeamID_L'}, inplace=True)

# Dropping unnecessary 'TeamID' columns
if 'TeamID_W' in tourney_results.columns:
    tourney_results.drop(['TeamID_W'], axis=1, inplace=True)
if 'TeamID_L' in tourney_results.columns:
    tourney_results.drop(['TeamID_L'], axis=1, inplace=True)

tourney_results.head()


tourney_results.columns


tourney_results = tourney_results.loc[:,~tourney_results.columns.duplicated()]


# Creating Win Rate Difference feature
tourney_results['WinRateDiff'] = tourney_results['WWinDiff'] - tourney_results['LWinDiff']
tourney_results.head()


# Features and target variables
features = ['SeedDiff', 'WinRateDiff']
X = tourney_results[features]
y = (tourney_results['WTeamID'] < tourney_results['LTeamID']).astype(int)


X.head()


y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_test.head()


X_test.head()


y_train.head()


y_test.head()


# LGB Model Training
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)


# Evaluating LightGBM model
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)
print('Log Loss:', log_loss(y_test, y_pred_proba))
print('Accuracy:', accuracy_score(y_test, y_pred))


# Random Forest model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Evaluating Random Forest model
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = (y_pred_proba_rf > 0.5).astype(int)
print('RF Log Loss:', log_loss(y_test, y_pred_proba_rf))
print('RF Accuracy:', accuracy_score(y_test, y_pred_rf))


# Gradient Boosting Classifier model training
gbc_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc_model.fit(X_train, y_train)


# Evaluating Gradient Boosting Classifier model
y_pred_proba_gbc = gbc_model.predict_proba(X_test)[:, 1]
y_pred_gbc = (y_pred_proba_gbc > 0.5).astype(int)
print('GBC Log Loss:', log_loss(y_test, y_pred_proba_gbc))
print('GBC Accuracy:', accuracy_score(y_test, y_pred_gbc))


param_grid = {
    'num_leaves': [31, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(lgb.LGBMClassifier(), param_grid, cv=5, scoring='neg_log_loss', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print('Best params:', grid_search.best_params_)
print('Best log loss:', -grid_search.best_score_)


param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='neg_log_loss', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print('Best params:', grid_search_rf.best_params_)
print('Best log loss:', -grid_search_rf.best_score_)


param_grid_gbc = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid_gbc, cv=5, scoring='neg_log_loss', verbose=1, n_jobs=-1)
grid_search_gbc.fit(X_train, y_train)

print('Best params:', grid_search_gbc.best_params_)
print('Best log loss:', -grid_search_gbc.best_score_)


estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gbc', GradientBoostingClassifier(n_estimators=100))
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=lgb.LGBMClassifier())
stacking_model.fit(X_train, y_train)

# Evaluate Stacking Model
y_pred_proba_stack = stacking_model.predict_proba(X_test)[:, 1]
y_pred_stack = (y_pred_proba_stack > 0.5).astype(int)
print('Stacking Log Loss:', log_loss(y_test, y_pred_proba_stack))
print('Stacking Accuracy:', accuracy_score(y_test, y_pred_stack))


gbc_best_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=100)
gbc_best_model.fit(X_train, y_train)


# Evaluating GBC model (on test data)
y_pred_proba_gbc = gbc_best_model.predict_proba(X_test)[:, 1]
y_pred_gbc = (y_pred_proba_gbc > 0.5).astype(int)
print('GBC Log Loss:', log_loss(y_test, y_pred_proba_gbc))
print('GBC Accuracy:', accuracy_score(y_test, y_pred_gbc))


# Preparing the final model using the entire dataset
gbc_final_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=100)
gbc_final_model.fit(X, y)


# Preparing test data (SampleSubmissionStage2)
submission = pd.read_csv(f"{DATA_PATH}/SampleSubmissionStage2.csv")

submission['Season'] = submission['ID'].apply(lambda x: int(x.split('_')[0]))
submission['TeamA'] = submission['ID'].apply(lambda x: int(x.split('_')[1]))
submission['TeamB'] = submission['ID'].apply(lambda x: int(x.split('_')[2]))
submission.head()


# Unique identifier for teams in submission data
teams_in_submission = set(submission['TeamA']).union(set(submission['TeamB']))
teams_in_seeds = set(seeds['TeamID'])
missing_teams = teams_in_submission - teams_in_seeds

# Adding missing teams to seeds data with default seed value
default_seed = seeds['SeedInt'].max() + 1
missing_seeds = pd.DataFrame({
    'Season': [submission['Season'].unique()[0]] * len(missing_teams),
    'TeamID': list(missing_teams),
    'Seed': ['Z99'] * len(missing_teams),
    'SeedInt': [default_seed] * len(missing_teams)
})

seeds = pd.concat([seeds, missing_seeds], ignore_index=True)

# Merging seeds with submission data for TeamA & TeamB
submission = pd.merge(submission, seeds[['Season', 'TeamID', 'SeedInt']], how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
submission.rename(columns={'SeedInt': 'SeedA'}, inplace=True)
submission.drop(columns=['TeamID'], inplace=True)

submission = pd.merge(submission, seeds[['Season', 'TeamID', 'SeedInt']], how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
submission.rename(columns={'SeedInt': 'SeedB'}, inplace=True)
submission.drop(columns=['TeamID'], inplace=True)

# Filling any remaining missing values in SeedA & SeedB with the default seed value
submission['SeedA'].fillna(default_seed, inplace=True)
submission['SeedB'].fillna(default_seed, inplace=True)

submission['SeedDiff'] = submission['SeedA'] - submission['SeedB']

# Checking for missing values
print("Missing SeedA values after adding default seeds:", submission['SeedA'].isna().sum())
print("Missing SeedB values after adding default seeds:", submission['SeedB'].isna().sum())

submission.head()


# Merging win_diff with submission data for TeamA and TeamB
submission = pd.merge(submission, win_diff[['Season', 'TeamID', 'WinDiff']], how='left', left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'], suffixes=('', '_A'))
submission.rename(columns={'WinDiff': 'WinDiffA'}, inplace=True)
submission.drop(columns=['TeamID'], inplace=True)

submission = pd.merge(submission, win_diff[['Season', 'TeamID', 'WinDiff']], how='left', left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'], suffixes=('', '_B'))
submission.rename(columns={'WinDiff': 'WinDiffB'}, inplace=True)
submission.drop(columns=['TeamID'], inplace=True)
submission.head()


# Filling any remaining missing values in WinDiffA and WinDiffB with 0 (no win data)
submission['WinDiffA'].fillna(0, inplace=True)
submission['WinDiffB'].fillna(0, inplace=True)

submission['WinRateDiff'] = submission['WinDiffA'] - submission['WinDiffB']
submission.head()


submission.columns


# Features for submission data
submission_features = submission[['SeedDiff', 'WinRateDiff']]


# Imputing missing values in submission data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
submission_features_imputed = imputer.transform(submission_features)


y_pred_proba_submission = gbc_final_model.predict_proba(submission_features)[:, 1]


submission['Pred'] = y_pred_proba_submission
print(submission.head())
submission[['ID', 'Pred']].to_csv('submission.csv', index=False)










