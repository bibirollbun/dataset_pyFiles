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


import pandas as pd

# Load core data
m_teams = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv")
w_teams = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WTeams.csv")
m_seasons = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv")
w_seasons = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv")

# Load game results
m_compact = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv")
w_compact = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv")
m_tourney = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
w_tourney = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv")

# Load advanced stats
m_detailed = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv")
w_detailed = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv")

# Load rankings (critical for performance)
massey = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv")



def calculate_efficiency(df):
    # Offensive Efficiency = (Points Scored / Possessions)
    df['WPoss'] = df['WFGA'] - df['WOR'] + df['WTO'] + 0.475*df['WFTA']
    df['LPoss'] = df['LFGA'] - df['LOR'] + df['LTO'] + 0.475*df['LFTA']
    df['WOffEff'] = df['WScore'] / df['WPoss']
    df['LDefEff'] = df['LScore'] / df['LPoss']
    return df

m_detailed = calculate_efficiency(m_detailed)
w_detailed = calculate_efficiency(w_detailed)


def create_team_features(games_df, team_id_col):
    features = games_df.groupby(team_id_col).agg({
        'WScore': ['mean', 'std'],
        'LScore': ['mean', 'std'],
        'WOffEff': 'mean',
        'LDefEff': 'mean'
    })
    features.columns = ['_'.join(col).strip() for col in features.columns]
    return features

m_team_stats = create_team_features(m_detailed, 'WTeamID')
w_team_stats = create_team_features(w_detailed, 'WTeamID')


def get_final_rankings(massey_df):
    # Get latest pre-tournament rankings
    final_ranks = massey_df[massey_df['SystemName'] == 'POM']  # Use Pomeroy rankings
    final_ranks = final_ranks.sort_values(['Season', 'RankingDayNum'], ascending=False)
    final_ranks = final_ranks.drop_duplicates(['Season', 'TeamID'])
    return final_ranks[['Season', 'TeamID', 'OrdinalRank']]

pomeroy_ranks = get_final_rankings(massey)


import re
def clean_seed(seed):
    return int(re.sub('[^0-9]', '', seed[:3]))

m_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
m_seeds['Seed'] = m_seeds['Seed'].apply(clean_seed)


import pandas as pd
from sklearn.model_selection import train_test_split



m_tourney = m_tourney[['Season', 'WTeamID', 'LTeamID']]
m_tourney['Result'] = 1  # Assuming the dataset records winning teams


m_team_stats = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv")
# Process team stats if needed (aggregate features per team)

# Merge features for each team pair
train_data = pd.merge(
    m_tourney,
    m_team_stats.add_prefix('T1_'),
    left_on='WTeamID', right_index=True
)
train_data = pd.merge(
    train_data,
    m_team_stats.add_prefix('T2_'),
    left_on='LTeamID', right_index=True
)

# Ensure 'Result' column exists before dropping
if 'Result' in train_data.columns:
    X = train_data.drop(['Result'], axis=1)
    y = train_data['Result']

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    print("Error: 'Result' column not found in train_data.")



print(y_train.unique())  # Check unique labels in y_train



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



y = y.astype(int)  # Convert to integers



print(y_train.value_counts())  # Ensure both classes are present



print(y.value_counts())  # Ensure both 0 and 1 exist



# Ensure tourney_games is properly defined
tourney_games = m_tourney[['Season', 'WTeamID', 'LTeamID']].copy()

# Create labels for both winning and losing cases
train_data = pd.concat([
    pd.DataFrame({
        'Season': tourney_games['Season'],
        'Team1': tourney_games['WTeamID'],
        'Team2': tourney_games['LTeamID'],
        'Result': 1  # Win
    }),
    pd.DataFrame({
        'Season': tourney_games['Season'],
        'Team1': tourney_games['LTeamID'],
        'Team2': tourney_games['WTeamID'],
        'Result': 0  # Loss
    })
], ignore_index=True)  # Reset index after concatenation

# Check distribution
print(train_data['Result'].value_counts())  # Should show both 1s and 0s



X = train_data.drop(['Result'], axis=1)
y = train_data['Result'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratify ensures balanced classes
)

print(y_train.value_counts())  # Should show both 1s and 0s



xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)



from sklearn.metrics import roc_auc_score

xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
lgb_preds = lgb_model.predict_proba(X_val)[:, 1]

ensemble_pred = (xgb_preds * 0.5) + (lgb_preds * 0.5)
auc_score = roc_auc_score(y_val, ensemble_pred)
print(f"Ensemble AUC Score: {auc_score:.4f}")



# Calculate win ratio
team_wins = m_compact.groupby('WTeamID').size()
team_losses = m_compact.groupby('LTeamID').size()
total_games = team_wins.add(team_losses, fill_value=0)

win_ratio = team_wins / total_games
win_ratio = win_ratio.fillna(0)  # Handle teams with no games

# Merge win ratio into dataset
train_data['T1_win_ratio'] = train_data['Team1'].map(win_ratio)
train_data['T2_win_ratio'] = train_data['Team2'].map(win_ratio)



print(train_data.columns)



from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

# Ensure labels are binary
y = y.astype(int)

# Train-test split with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check if both classes exist
print(y_train.value_counts())

# XGBoost
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6
)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.01,
    n_estimators=1000
)

# Fit models
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Ensemble predictions
ensemble_pred = (xgb_model.predict_proba(X_val)[:,1] * 0.5 +
                 lgb_model.predict_proba(X_val)[:,1] * 0.5)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ROC Curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, xgb_model.predict_proba(X_val)[:, 1])
fpr_lgb, tpr_lgb, _ = roc_curve(y_val, lgb_model.predict_proba(X_val)[:, 1])
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_val, ensemble_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost (AUC: {:.3f})".format(auc(fpr_xgb, tpr_xgb)))
plt.plot(fpr_lgb, tpr_lgb, label="LightGBM (AUC: {:.3f})".format(auc(fpr_lgb, tpr_lgb)))
plt.plot(fpr_ensemble, tpr_ensemble, label="Ensemble (AUC: {:.3f})".format(auc(fpr_ensemble, tpr_ensemble)), linestyle="--")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Feature Importance - XGBoost
plt.figure(figsize=(8, 6))
xgb_importance = xgb_model.feature_importances_
sns.barplot(x=xgb_importance, y=X.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()


# Confusion Matrix
y_pred = (ensemble_pred > 0.5).astype(int)
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


def generate_matchups(season, teams):
    from itertools import combinations
    matchups = [f"{season}_{t1}_{t2}" 
               for t1, t2 in combinations(teams, 2)]
    return pd.DataFrame({'ID': matchups})

# Get 2025 teams (example)
m2025_teams = m_teams[m_teams['LastD1Season'] >= 2025]['TeamID']
submission_df = generate_matchups(2025, m2025_teams)


print(submission_df.columns)  # Should output: Index(['ID', 'Pred'], dtype='object')



print("Train Features:", X_train.columns)



print(submission_df.shape)  # Should be (N, 2), where N is the number of rows



print(submission_df.head())



final_preds = (xgb_model.predict_proba(final_features)[:,1] * 0.5 + 
               lgb_model.predict_proba(final_features)[:,1] * 0.5)

submission_df['Pred'] = final_preds
submission_df.to_csv('champion_submission.csv', index=False)

print("Submission file saved successfully!")




submission_df[['Season', 'Team1', 'Team2']] = submission_df['ID'].str.split('_', expand=True).astype(int)

# Now, use it as final_features
final_features = submission_df[['Season', 'Team1', 'Team2']]
print(final_features.head())  # Verify the output

# Make predictions
final_preds = (xgb_model.predict_proba(final_features)[:, 1] * 0.5 +
               lgb_model.predict_proba(final_features)[:, 1] * 0.5)

submission_df['Pred'] = final_preds
submission_df.to_csv('champion_submission.csv', index=False)

print("Submission file saved successfully!")





plt.figure(figsize=(10, 6))
sns.histplot(final_preds, bins=30, kde=True, color='blue')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.show()



submission_df['Pred'] = final_preds

# Top 10 most confident predictions
print("Top 10 Most Confident Predictions:")
print(submission_df.nlargest(10, 'Pred'))

# Bottom 10 least confident predictions
print("\nBottom 10 Least Confident Predictions:")
print(submission_df.nsmallest(10, 'Pred'))



team_probs = submission_df.groupby('Team1')['Pred'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
team_probs.head(10).plot(kind='bar', color='green')
plt.xlabel("Team ID")
plt.ylabel("Average Predicted Win Probability")
plt.title("Top 10 Teams by Predicted Win Probability")
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(x=submission_df["Season"], y=submission_df["Pred"], palette="coolwarm")
plt.xlabel("Season")
plt.ylabel("Predicted Probability")
plt.title("Season-Wise Prediction Trends")
plt.xticks(rotation=45)
plt.show()



# Selecting a specific team for visualization
team_id = 1101  # Change this to any Team ID you want to analyze

plt.figure(figsize=(12, 6))
matchups = submission_df[(submission_df["Team1"] == team_id) | (submission_df["Team2"] == team_id)]
sns.barplot(x=matchups["Team2"], y=matchups["Pred"], palette="viridis")
plt.xlabel("Opponent Team ID")
plt.ylabel("Predicted Win Probability")
plt.title(f"Predicted Win Probability for Team {team_id} Against Opponents")
plt.xticks(rotation=45)
plt.show()



match_matrix = submission_df.pivot(index="Team1", columns="Team2", values="Pred")

plt.figure(figsize=(12, 8))
sns.heatmap(match_matrix, cmap="coolwarm", annot=False)
plt.xlabel("Team 2")
plt.ylabel("Team 1")
plt.title("Heatmap of Predicted Win Probabilities")
plt.show()

