# Cell 1: Imports and Data Loading
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

# For hyperparameter tuning
import optuna

warnings.filterwarnings("ignore", category=FutureWarning)

# Define the data path (using wildcard to load all CSV files)
DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/**'

# Load all CSV files into a dictionary
data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') 
        for p in glob.glob(DATA_PATH)}

# Check loaded data keys
print("Data keys loaded:", list(data.keys()))


# Cell 2: Basic Data Preparation and Seeds Processing

# Combine men's and women's teams data
teams = pd.concat([data['MTeams'], data['WTeams']], ignore_index=True)

# Combine men's and women's team spellings and merge with teams data
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']], ignore_index=True)
teams_spelling = teams_spelling.groupby('TeamID', as_index=False)['TeamNameSpelling'].count()
teams_spelling.columns = ['TeamID', 'TeamNameCount']
teams = pd.merge(teams, teams_spelling, how='left', on='TeamID')
del teams_spelling

# Combine detailed season and tournament results
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']], ignore_index=True)
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']], ignore_index=True)

# Combine seeds (men + women)
seeds_df = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']], ignore_index=True)
# Create a seeds dictionary in format "Season_TeamID": seed_number (e.g., "2017_1104": 3)
seeds = {
    '_'.join(map(str, [int(row[0]), row[2]])): int(row[1][1:3])
    for row in seeds_df[['Season', 'Seed', 'TeamID']].values
}

# Choose the submission file (Stage2 if available, otherwise Stage1)
if 'SampleSubmissionStage2' in data:
    sub = data['SampleSubmissionStage2']
else:
    sub = data['SampleSubmissionStage1']

# Label game type: 'S' for season and 'T' for tournament
season_dresults['ST'] = 'S'
tourney_dresults['ST'] = 'T'

# Merge season and tournament games
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)

# Convert 'WLoc' from A/H/N to numeric (optional)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

# Create unique game IDs and order teams consistently
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

# Map seeds to each team (fill missing with 0)
games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

# Create the target variable: Pred = 1 if Team1 (sorted order) is the winner
games['Pred'] = games.apply(lambda r: 1.0 if r['Team1'] == r['WTeamID'] else 0.0, axis=1)

# Process submission file similarly: extract Season, Team1, and Team2 and map seeds
sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
sub['Team1'] = sub['ID'].apply(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].apply(lambda x: x.split('_')[2])
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)

print("Basic data preparation completed!")


# Cell 3: Advanced Feature Engineering - Aggregated Team Stats

def create_team_features(df):
    """
    This function aggregates team statistics for each (Season, TeamID) 
    using the regular season detailed results.
    """
    # Calculate shooting percentages (handle division by zero using replace)
    df['WFGPerc'] = (df['WFGM'] / df['WFGA']).replace([np.inf, -np.inf], 0)
    df['WFG3Perc'] = (df['WFGM3'] / df['WFGA3']).replace([np.inf, -np.inf], 0)
    df['WFTPerc'] = (df['WFTM'] / df['WFTA']).replace([np.inf, -np.inf], 0)
    df['LFGPerc'] = (df['LFGM'] / df['LFGA']).replace([np.inf, -np.inf], 0)
    df['LFG3Perc'] = (df['LFGM3'] / df['LFGA3']).replace([np.inf, -np.inf], 0)
    df['LFTPerc'] = (df['LFTM'] / df['LFTA']).replace([np.inf, -np.inf], 0)
    
    # Aggregate stats for winning teams
    wstats = df.groupby(['Season', 'WTeamID']).agg({
        'WScore': ['mean', 'sum'],
        'WFGPerc': 'mean',
        'WFG3Perc': 'mean',
        'WFTPerc': 'mean',
        'WOR': 'mean',
        'WDR': 'mean',
        'WAst': 'mean',
        'WTO': 'mean',
        'WStl': 'mean',
        'WBlk': 'mean',
        'WPF': 'mean'
    })
    wstats.columns = ["_".join(x) for x in wstats.columns.ravel()]
    wstats.reset_index(inplace=True)
    wstats.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
    
    # Aggregate stats for losing teams
    lstats = df.groupby(['Season', 'LTeamID']).agg({
        'LScore': ['mean', 'sum'],
        'LFGPerc': 'mean',
        'LFG3Perc': 'mean',
        'LFTPerc': 'mean',
        'LOR': 'mean',
        'LDR': 'mean',
        'LAst': 'mean',
        'LTO': 'mean',
        'LStl': 'mean',
        'LBlk': 'mean',
        'LPF': 'mean'
    })
    lstats.columns = ["_".join(x) for x in lstats.columns.ravel()]
    lstats.reset_index(inplace=True)
    lstats.rename(columns={'LTeamID': 'TeamID'}, inplace=True)
    
    # Merge winning and losing stats (outer join to account for teams appearing in one role only)
    merged = pd.merge(wstats, lstats, on=['Season', 'TeamID'], how='outer').fillna(0)
    
    # Example combined feature: overall average score per game
    merged['Score_mean'] = (merged['WScore_mean'] + merged['LScore_mean']) / 2.0
    return merged

# Create aggregated statistics from season detailed results only (using season_dresults)
season_stats = create_team_features(season_dresults)

print("Advanced feature engineering (aggregated team stats) completed!")


# Cell 4 (Revised): Merge Aggregated Stats into Games and Submission Data

# For Team1: rename 'TeamID' to 'Team1' and add suffix only to non-key columns
temp_stats_T1 = season_stats.copy()
temp_stats_T1 = temp_stats_T1.rename(columns={'TeamID': 'Team1'})
cols_to_suffix = [col for col in temp_stats_T1.columns if col not in ['Season', 'Team1']]
temp_stats_T1 = temp_stats_T1.rename(columns={col: col + '_T1' for col in cols_to_suffix})

# Merge aggregated stats for Team1 into games DataFrame
games = pd.merge(
    games, 
    temp_stats_T1,
    on=['Season', 'Team1'],
    how='left'
)

# For Team2: rename 'TeamID' to 'Team2' and add suffix only to non-key columns
temp_stats_T2 = season_stats.copy()
temp_stats_T2 = temp_stats_T2.rename(columns={'TeamID': 'Team2'})
cols_to_suffix = [col for col in temp_stats_T2.columns if col not in ['Season', 'Team2']]
temp_stats_T2 = temp_stats_T2.rename(columns={col: col + '_T2' for col in cols_to_suffix})

# Merge aggregated stats for Team2 into games DataFrame
games = pd.merge(
    games,
    temp_stats_T2,
    on=['Season', 'Team2'],
    how='left'
)

# Create difference feature example: difference in average score between Team1 and Team2
games['ScoreMean_diff'] = games['Score_mean_T1'] - games['Score_mean_T2']

# --- For Submission DataFrame ---

# Ensure join columns in submission DataFrame are numeric
sub['Team1'] = pd.to_numeric(sub['Team1'])
sub['Team2'] = pd.to_numeric(sub['Team2'])

# For Team1 in submission data
temp_stats_T1_sub = season_stats.copy()
temp_stats_T1_sub = temp_stats_T1_sub.rename(columns={'TeamID': 'Team1'})
cols_to_suffix = [col for col in temp_stats_T1_sub.columns if col not in ['Season', 'Team1']]
temp_stats_T1_sub = temp_stats_T1_sub.rename(columns={col: col + '_T1' for col in cols_to_suffix})

sub = pd.merge(
    sub,
    temp_stats_T1_sub,
    on=['Season', 'Team1'],
    how='left'
)

# For Team2 in submission data
temp_stats_T2_sub = season_stats.copy()
temp_stats_T2_sub = temp_stats_T2_sub.rename(columns={'TeamID': 'Team2'})
cols_to_suffix = [col for col in temp_stats_T2_sub.columns if col not in ['Season', 'Team2']]
temp_stats_T2_sub = temp_stats_T2_sub.rename(columns={col: col + '_T2' for col in cols_to_suffix})

sub = pd.merge(
    sub,
    temp_stats_T2_sub,
    on=['Season', 'Team2'],
    how='left'
)

# Create the difference feature for submission data
sub['ScoreMean_diff'] = sub['Score_mean_T1'] - sub['Score_mean_T2']

# Fill any remaining missing values
games.fillna(-1, inplace=True)
sub.fillna(-1, inplace=True)

print("Merged aggregated stats into games and submission data successfully!")


# Cell 5: Feature Selection for Modeling

# Exclude non-feature columns from games DataFrame
exclude_cols = [
    'ID', 'Season', 'DayNum', 'WTeamID', 'LTeamID', 'WScore', 'LScore',
    'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
    'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
    'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
    'WLoc', 'ST', 'Team1', 'Team2', 'IDTeam1', 'IDTeam2', 'Pred'
]
all_cols = games.columns.tolist()
model_cols = [c for c in all_cols if c not in exclude_cols]

print("Selected model columns:", model_cols)
print("Shape of training data X:", games[model_cols].shape)

# Define training features and target
X = games[model_cols]
y = games['Pred']


# Cell 6: Hyperparameter Tuning with Optuna using GroupKFold

# Preprocess data: impute missing values and scale features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Use Season as groups to avoid data leakage between different seasons
groups = games['Season'].values

# Define the objective function for Optuna optimization
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'seed': 42,
        'verbose': -1
    }
    
    # Use GroupKFold for cross-validation based on Season groups
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    
    for train_idx, valid_idx in gkf.split(X_scaled, y, groups=groups):
        X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
        
        # Train LightGBM model using callbacks for early stopping and disable logging
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5),
                lgb.log_evaluation(period=0)  # disable logging
            ]
        )
        
        valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
        cv_score = log_loss(y_valid, valid_preds)
        cv_scores.append(cv_score)
    
    return np.mean(cv_scores)

# Run the hyperparameter tuning for a limited number of trials (increase n_trials for a better search)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best hyperparameters found:", best_params)
print("Best CV Log Loss:", study.best_value)


# Cell 7: Train Final Model on All Data with Best Hyperparameters

# Update parameters with best found parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'seed': 42
}
params.update(best_params)

# Train final LightGBM model on full training data
lgb_full = lgb.Dataset(X_scaled, label=y)
best_num_boost_round = 1000  # Optionally, you can set this based on CV results
model_final = lgb.train(params, lgb_full, num_boost_round=best_num_boost_round)

# Evaluate training performance
train_preds = model_final.predict(X_scaled)
train_loss = log_loss(y, train_preds)
print("Final model log loss on full training data:", train_loss)


# Cell 8: Prediction on Submission Data and Optional Calibration

# Prepare submission features using the same model columns and preprocessing
X_sub = sub[model_cols]
X_sub_imputed = imputer.transform(X_sub)
X_sub_scaled = scaler.transform(X_sub_imputed)

# Predict using the final model
preds = model_final.predict(X_sub_scaled)

# Optional: Calibrate predictions using Isotonic Regression
ir = IsotonicRegression(out_of_bounds='clip')
# Here we use the training predictions and labels for calibration.
ir.fit(train_preds, y)
preds_cal = ir.transform(preds)

# Add predictions to submission DataFrame
sub['Pred'] = preds_cal  # Or use 'preds' directly if calibration is not desired

# Save submission file
sub[['ID', 'Pred']].to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created!")


# Cell 9: Visualization of Calibrated Predictions Distribution

sns.set(style="ticks", context="talk", palette="deep")
plt.figure(figsize=(12, 6))
sns.histplot(preds_cal, bins=30, kde=True, color="#1f77b4", edgecolor="white", linewidth=1.2, alpha=0.8)
plt.title("Distribution of Calibrated Predictions", fontsize=16)
plt.xlabel("Winning Probability", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
sns.despine()
plt.show()

print("All cells executed successfully!")

