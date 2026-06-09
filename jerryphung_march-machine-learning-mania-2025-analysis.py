# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import kagglehub

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 1. Data Loading


march_machine_learning_mania_2025_path = kagglehub.competition_download('march-machine-learning-mania-2025')
march_machine_learning_mania_2025_path


# Load key CSV files into DataFrames
tourney_results = pd.read_csv(march_machine_learning_mania_2025_path + '/MNCAATourneyCompactResults.csv')   # Tournament game outcomes
season_results = pd.read_csv(march_machine_learning_mania_2025_path + '/MRegularSeasonDetailedResults.csv') # Regular season game details
seeds = pd.read_csv(march_machine_learning_mania_2025_path + '/MNCAATourneySeeds.csv')              # Tournament seeds for each team
teams = pd.read_csv(march_machine_learning_mania_2025_path + '/MTeams.csv')                         # Team IDs and names

# Quick peek at the data
print(tourney_results.head(3))
print(seeds.head(3))


# Aggregate wins (for each team as winner)
wins = season_results.groupby(['Season', 'WTeamID']).agg(
    Wins=('WTeamID', 'count'),
    TotalPointsFor=('WScore', 'sum'),
    TotalPointsAgainst=('LScore', 'sum')
).reset_index().rename(columns={'WTeamID': 'TeamID'})

# Aggregate losses (for each team as loser)
losses = season_results.groupby(['Season', 'LTeamID']).agg(
    Losses=('LTeamID', 'count'),
    TotalPointsForLoss=('LScore', 'sum'),
    TotalPointsAgainstLoss=('WScore', 'sum')
).reset_index().rename(columns={'LTeamID': 'TeamID'})


# Merge wins and losses
team_stats = pd.merge(wins, losses, on=['Season', 'TeamID'], how='outer')
team_stats['Wins'] = team_stats['Wins'].fillna(0)
team_stats['Losses'] = team_stats['Losses'].fillna(0)
team_stats['Games'] = team_stats['Wins'] + team_stats['Losses']
team_stats['WinPct'] = team_stats['Wins'] / team_stats['Games']
team_stats['AvgPointsFor'] = (team_stats['TotalPointsFor'].fillna(0) + team_stats['TotalPointsForLoss'].fillna(0)) / team_stats['Games']
team_stats['AvgPointsAgainst'] = (team_stats['TotalPointsAgainst'].fillna(0) + team_stats['TotalPointsAgainstLoss'].fillna(0)) / team_stats['Games']


# Work with a copy of tournament results
tourney = tourney_results.copy()

# Process seeds for winners:
seeds_win = seeds[['Season', 'TeamID', 'Seed']].copy()
# Extract numeric part of the seed (e.g., "W01" -> 1)
seeds_win['SeedNumber'] = seeds_win['Seed'].str.extract('(\d+)').astype(int)
# Rename columns so that they clearly refer to the winning team's seed data.
# This helps avoid confusion and collisions with columns for the losing team 
seeds_win = seeds_win.rename(columns={'Seed': 'Seed_W', 'TeamID': 'TeamID_W', 'SeedNumber': 'SeedNumber_W'})
tourney = pd.merge(tourney, seeds_win, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID_W'], how='left')

# # Process seeds for losers:
seeds_loss = seeds[['Season', 'TeamID', 'Seed']].copy()
seeds_loss['SeedNumber'] = seeds_loss['Seed'].str.extract('(\d+)').astype(int)
# Rename columns so that they clearly refer to the losing team's seed data.
# This helps avoid confusion and collisions with columns for the winning team 
seeds_loss = seeds_loss.rename(columns={'Seed': 'Seed_L', 'TeamID': 'TeamID_L', 'SeedNumber': 'SeedNumber_L'})
tourney = pd.merge(tourney, seeds_loss, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID_L'], how='left')


# Merge team stats for winning team
team_stats_win = team_stats.rename(columns={
    'TeamID': 'TeamID_W',
    'WinPct': 'WinPct_W',
    'AvgPointsFor': 'AvgPointsFor_W',
    'AvgPointsAgainst': 'AvgPointsAgainst_W'
})
tourney = pd.merge(tourney, team_stats_win[['Season', 'TeamID_W', 'WinPct_W', 'AvgPointsFor_W', 'AvgPointsAgainst_W']],
                    on=['Season', 'TeamID_W'], how='left')


team_stats_win


tourney


tourney


# Merge team stats for losing team
team_stats_loss = team_stats.rename(columns={
    'TeamID': 'TeamID_L',
    'WinPct': 'WinPct_L',
    'AvgPointsFor': 'AvgPointsFor_L',
    'AvgPointsAgainst': 'AvgPointsAgainst_L'
})
tourney = pd.merge(tourney, team_stats_loss[['Season', 'TeamID_L', 'WinPct_L', 'AvgPointsFor_L', 'AvgPointsAgainst_L']],
                    on=['Season', 'TeamID_L'], how='left')



# Compute matchup features
tourney['SeedDiff'] = tourney['SeedNumber_W'] - tourney['SeedNumber_L']
tourney['WinPctDiff'] = tourney['WinPct_W'] - tourney['WinPct_L']
tourney['AvgPtDiff'] = tourney['AvgPointsFor_W'] - tourney['AvgPointsFor_L']
tourney['AvgOppPtDiff'] = tourney['AvgPointsAgainst_W'] - tourney['AvgPointsAgainst_L']

# The target is 1 since these are actual results (the winner is always team1 in these rows)
tourney['Target'] = 1

# %% [markdown]
# To balance the training data (so the model sees examples where team1 loses), we create an inverse matchup by flipping the features and setting Target = 0.

# %% [code]
inverse_tourney = tourney.copy()
inverse_tourney['SeedDiff'] = -inverse_tourney['SeedDiff']
inverse_tourney['WinPctDiff'] = -inverse_tourney['WinPctDiff']
inverse_tourney['AvgPtDiff'] = -inverse_tourney['AvgPtDiff']
inverse_tourney['AvgOppPtDiff'] = -inverse_tourney['AvgOppPtDiff']
inverse_tourney['Target'] = 0


# Combine the original and inverse matchups
train_data = pd.concat([tourney, inverse_tourney], ignore_index=True)

# Keep only the key matchup features and the target
feature_cols = ['SeedDiff', 'WinPctDiff', 'AvgPtDiff', 'AvgOppPtDiff']
X = train_data[feature_cols]
y = train_data['Target']


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of Seed Difference
sns.histplot(data=train_data, x='SeedDiff', kde=True)
plt.title('Distribution of Seed Difference')
plt.xlabel('SeedDiff')
plt.ylabel('Frequency')
plt.show()

# If you want to see how SeedDiff differs between matchups Team1 wins vs. loses:
sns.histplot(data=train_data, x='SeedDiff', hue='Target', kde=True)
plt.title('Distribution of Seed Difference by Outcome')
plt.xlabel('SeedDiff')
plt.ylabel('Frequency')
plt.show()



sns.boxplot(data=train_data, x='Target', y='SeedDiff')
plt.title('SeedDiff by Game Outcome (Target)')
plt.xlabel('Target (0 = Team1 lost, 1 = Team1 won)')
plt.ylabel('SeedDiff')
plt.show()



# Correlation among main features plus the target
feature_subset = ['SeedDiff', 'WinPctDiff', 'AvgPtDiff', 'AvgOppPtDiff', 'Target']
corr_matrix = train_data[feature_subset].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title("Correlation Matrix (Features vs Target)")
plt.show()



 # We scale the features, split the data into training/validation sets, and train a Logistic Regression model. We also train a Random Forest and an XGBoost model and combine them using a Voting Ensemble.

# %% [code]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# %% [code]
from sklearn.impute import SimpleImputer

# Create an imputer that replaces NaNs with the median value for each feature
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training features and transform both training and validation sets
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)


# Now, train Logistic Regression with the imputed data
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_imputed, y_train)

print("Logistic Regression training score:", log_reg.score(X_train_imputed, y_train))
print("Logistic Regression validation score:", log_reg.score(X_val_imputed, y_val))

# %% [code]
# Evaluate using log loss
from sklearn.metrics import log_loss
val_pred_proba = log_reg.predict_proba(X_val_imputed)[:,1]
print("Logistic Regression Validation Log Loss:", log_loss(y_val, val_pred_proba))


from sklearn.metrics import roc_curve, auc

# Assuming y_val are the true labels and val_pred_proba are predicted probabilities
fpr, tpr, thresholds = roc_curve(y_val, val_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1], color='red', linestyle='--')  # reference line
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()



from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_val, val_pred_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title('Calibration Curve')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability in each bin')
plt.legend()
plt.show()



# Train additional models: RandomForest and XGBoost
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_imputed, y_train)


xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                    use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_imputed, y_train)



# Create ensemble using soft voting (averaging predicted probabilities)
ensemble = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('xgb', xgb)], voting='soft')
ensemble.fit(X_train_imputed, y_train)
ensemble_val_pred = ensemble.predict_proba(X_val_imputed)[:,1]
print("Ensemble Validation Log Loss:", log_loss(y_val, ensemble_val_pred))


# Load sample submission file (for Stage 1)
men_sample_submission = pd.read_csv(march_machine_learning_mania_2025_path+'/SampleSubmissionStage1.csv')


def parse_id(matchup_id):
    parts = matchup_id.split('_')
    return int(parts[0]), int(parts[1]), int(parts[2])


test_features = []
# Iterate over each matchup in the sample submission
for idx, row in men_sample_submission.iterrows():
    matchup_id = row['ID']
    season, team1, team2 = parse_id(matchup_id)
    
    # Retrieve team statistics for team1 and team2
    t1_stats_df = team_stats[(team_stats['Season'] == season) & (team_stats['TeamID'] == team1)]
    t2_stats_df = team_stats[(team_stats['Season'] == season) & (team_stats['TeamID'] == team2)]
    # If stats are missing for either team, skip this matchup
    if t1_stats_df.empty or t2_stats_df.empty:
        continue
    t1_stats = t1_stats_df.iloc[0]
    t2_stats = t2_stats_df.iloc[0]
    
    # Retrieve seed information for both teams
    s1 = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == team1)]
    s2 = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == team2)]
    if s1.empty:
        t1_seed = -1
    else:
        t1_seed = int(s1['Seed'].str.extract('(\d+)').iloc[0, 0])
    if s2.empty:
        t2_seed = -1
    else:
        t2_seed = int(s2['Seed'].str.extract('(\d+)').iloc[0, 0])
    
    # Compute matchup features
    feat = {}
    feat['SeedDiff'] = t1_seed - t2_seed
    feat['WinPctDiff'] = t1_stats['WinPct'] - t2_stats['WinPct']
    feat['AvgPtDiff'] = t1_stats['AvgPointsFor'] - t2_stats['AvgPointsFor']
    feat['AvgOppPtDiff'] = t1_stats['AvgPointsAgainst'] - t2_stats['AvgPointsAgainst']
    test_features.append(feat)


test_df = pd.DataFrame(test_features)
print("Test features sample:")
print(test_df.head())

# Scale test features using the same scaler
X_test_scaled = scaler.transform(test_df[feature_cols])

# Generate predictions with the ensemble
test_preds = ensemble.predict_proba(X_test_scaled)[:,1]

# NOTE: Ensure the order of rows in test_df aligns with sample_submission.
# For simplicity, here we assume they are in the same order.
men_sample_submission = men_sample_submission.iloc[:len(test_preds)].copy()
men_sample_submission['Pred'] = test_preds


w_tourney_results = pd.read_csv(march_machine_learning_mania_2025_path + '/WNCAATourneyCompactResults.csv')   # Tournament game outcomes
w_season_results = pd.read_csv(march_machine_learning_mania_2025_path + '/WRegularSeasonDetailedResults.csv') # Regular season game details
w_seeds = pd.read_csv(march_machine_learning_mania_2025_path + '/WNCAATourneySeeds.csv')              # Tournament seeds for each team
w_teams = pd.read_csv(march_machine_learning_mania_2025_path + '/WTeams.csv')                         # Team IDs and names

# Quick peek at the data
print(w_tourney_results.head(3))
print(w_seeds.head(3))


w_wins = w_season_results.groupby(['Season', 'WTeamID']).agg(
    Wins=('WTeamID', 'count'),
    TotalPointsFor=('WScore', 'sum'),
    TotalPointsAgainst=('LScore', 'sum')
).reset_index().rename(columns={'WTeamID': 'TeamID'})

# Aggregate losses for women's season
w_losses = w_season_results.groupby(['Season', 'LTeamID']).agg(
    Losses=('LTeamID', 'count'),
    TotalPointsForLoss=('LScore', 'sum'),
    TotalPointsAgainstLoss=('WScore', 'sum')
).reset_index().rename(columns={'LTeamID': 'TeamID'})


# Merge wins and losses and compute averages for women
w_team_stats = pd.merge(w_wins, w_losses, on=['Season', 'TeamID'], how='outer')
w_team_stats['Wins'] = w_team_stats['Wins'].fillna(0)
w_team_stats['Losses'] = w_team_stats['Losses'].fillna(0)
w_team_stats['Games'] = w_team_stats['Wins'] + w_team_stats['Losses']
w_team_stats['WinPct'] = w_team_stats['Wins'] / w_team_stats['Games']
w_team_stats['AvgPointsFor'] = (w_team_stats['TotalPointsFor'].fillna(0) + w_team_stats['TotalPointsForLoss'].fillna(0)) / w_team_stats['Games']
w_team_stats['AvgPointsAgainst'] = (w_team_stats['TotalPointsAgainst'].fillna(0) + w_team_stats['TotalPointsAgainstLoss'].fillna(0)) / w_team_stats['Games']


w_tourney = w_tourney_results.copy()

# Process women's seeds for winners:
w_seeds_win = w_seeds[['Season', 'TeamID', 'Seed']].copy()
w_seeds_win['SeedNumber'] = w_seeds_win['Seed'].str.extract('(\d+)').astype(int)
w_seeds_win = w_seeds_win.rename(columns={'Seed': 'Seed_W', 'TeamID': 'TeamID_W', 'SeedNumber': 'SeedNumber_W'})
w_tourney = pd.merge(w_tourney, w_seeds_win, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID_W'], how='left')

# Process women's seeds for losers:
w_seeds_loss = w_seeds[['Season', 'TeamID', 'Seed']].copy()
w_seeds_loss['SeedNumber'] = w_seeds_loss['Seed'].str.extract('(\d+)').astype(int)
w_seeds_loss = w_seeds_loss.rename(columns={'Seed': 'Seed_L', 'TeamID': 'TeamID_L', 'SeedNumber': 'SeedNumber_L'})
w_tourney = pd.merge(w_tourney, w_seeds_loss, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID_L'], how='left')


# Merge team stats for winning team (women)
w_team_stats_win = w_team_stats.rename(columns={
    'TeamID': 'TeamID_W',
    'WinPct': 'WinPct_W',
    'AvgPointsFor': 'AvgPointsFor_W',
    'AvgPointsAgainst': 'AvgPointsAgainst_W'
})
w_tourney = pd.merge(w_tourney, w_team_stats_win[['Season', 'TeamID_W', 'WinPct_W', 'AvgPointsFor_W', 'AvgPointsAgainst_W']],
                    on=['Season', 'TeamID_W'], how='left')


# Merge team stats for losing team (women)
w_team_stats_loss = w_team_stats.rename(columns={
    'TeamID': 'TeamID_L',
    'WinPct': 'WinPct_L',
    'AvgPointsFor': 'AvgPointsFor_L',
    'AvgPointsAgainst': 'AvgPointsAgainst_L'
})
w_tourney = pd.merge(w_tourney, w_team_stats_loss[['Season', 'TeamID_L', 'WinPct_L', 'AvgPointsFor_L', 'AvgPointsAgainst_L']],
                    on=['Season', 'TeamID_L'], how='left')

# Compute matchup features for women
w_tourney['SeedDiff'] = w_tourney['SeedNumber_W'] - w_tourney['SeedNumber_L']
w_tourney['WinPctDiff'] = w_tourney['WinPct_W'] - w_tourney['WinPct_L']
w_tourney['AvgPtDiff'] = w_tourney['AvgPointsFor_W'] - w_tourney['AvgPointsFor_L']
w_tourney['AvgOppPtDiff'] = w_tourney['AvgPointsAgainst_W'] - w_tourney['AvgPointsAgainst_L']

w_tourney['Target'] = 1  # winning team is team1

# Create inverse matchups for women
w_inverse_tourney = w_tourney.copy()
w_inverse_tourney['SeedDiff'] = -w_inverse_tourney['SeedDiff']
w_inverse_tourney['WinPctDiff'] = -w_inverse_tourney['WinPctDiff']
w_inverse_tourney['AvgPtDiff'] = -w_inverse_tourney['AvgPtDiff']
w_inverse_tourney['AvgOppPtDiff'] = -w_inverse_tourney['AvgOppPtDiff']
w_inverse_tourney['Target'] = 0


w_train_data = pd.concat([w_tourney, w_inverse_tourney], ignore_index=True)

# Use same feature columns for women as men
women_feature_cols = ['SeedDiff', 'WinPctDiff', 'AvgPtDiff', 'AvgOppPtDiff']
w_X = w_train_data[women_feature_cols]
w_y = w_train_data['Target']


w_X = pd.DataFrame(imputer.fit_transform(w_X), columns=women_feature_cols)
w_X_scaled = scaler.transform(w_X)

w_X_train, w_X_val, w_y_train, w_y_val = train_test_split(w_X_scaled, w_y, test_size=0.2, random_state=42)


w_log_reg = LogisticRegression(max_iter=1000)
w_log_reg.fit(w_X_train, w_y_train)

w_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
w_rf.fit(w_X_train, w_y_train)

w_xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                      use_label_encoder=False, eval_metric='logloss', random_state=42)
w_xgb.fit(w_X_train, w_y_train)

w_ensemble = VotingClassifier(estimators=[('lr', w_log_reg), ('rf', w_rf), ('xgb', w_xgb)], voting='soft')
w_ensemble.fit(w_X_train, w_y_train)


w_val_pred = w_ensemble.predict_proba(w_X_val)[:,1]
print("Women Ensemble Validation Log Loss:", log_loss(w_y_val, w_val_pred))


# For women's test predictions:
women_sample_submission = pd.read_csv(march_machine_learning_mania_2025_path+'/SampleSubmissionStage1.csv')# (Assuming women's test matchups are provided in SampleSubmissionStage2.csv)
women_test_features = []


for idx, row in women_sample_submission.iterrows():
    matchup_id = row['ID']
    season, team1, team2 = parse_id(matchup_id)
    
    # Retrieve team stats for women's teams
    t1_stats_df = w_team_stats[(w_team_stats['Season'] == season) & (w_team_stats['TeamID'] == team1)]
    t2_stats_df = w_team_stats[(w_team_stats['Season'] == season) & (w_team_stats['TeamID'] == team2)]
    if t1_stats_df.empty or t2_stats_df.empty:
        continue
    t1_stats = t1_stats_df.iloc[0]
    t2_stats = t2_stats_df.iloc[0]
    
    # Retrieve seeds for women
    s1 = w_seeds[(w_seeds['Season'] == season) & (w_seeds['TeamID'] == team1)]
    s2 = w_seeds[(w_seeds['Season'] == season) & (w_seeds['TeamID'] == team2)]
    if s1.empty:
        t1_seed = -1
    else:
        t1_seed = int(s1['Seed'].str.extract('(\d+)').iloc[0,0])
    if s2.empty:
        t2_seed = -1
    else:
        t2_seed = int(s2['Seed'].str.extract('(\d+)').iloc[0,0])
    
    feat = {
        'SeedDiff': t1_seed - t2_seed,
        'WinPctDiff': t1_stats['WinPct'] - t2_stats['WinPct'],
        'AvgPtDiff': t1_stats['AvgPointsFor'] - t2_stats['AvgPointsFor'],
        'AvgOppPtDiff': t1_stats['AvgPointsAgainst'] - t2_stats['AvgPointsAgainst']
    }
    women_test_features.append(feat)


women_test_df = pd.DataFrame(women_test_features)
women_X_test_scaled = scaler.transform(women_test_df[women_feature_cols])
women_test_preds = w_ensemble.predict_proba(women_X_test_scaled)[:,1]
women_sample_submission = women_sample_submission.iloc[:len(women_test_preds)].copy()
women_sample_submission['Pred'] = women_test_preds


# Save submission file
final_submission = pd.concat([men_sample_submission, women_sample_submission], ignore_index=True)
submission_file = 'submission.csv'
final_submission.to_csv(submission_file, index=False)
print("Final submission file saved as", submission_file)

