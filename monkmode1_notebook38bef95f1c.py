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



data_path = "/kaggle/input/march-machine-learning-mania-2025/"
import pandas as pd

# Function to load both Men's and Women's files
def load_mw_files(filename):
    men_df = pd.read_csv(data_path + f"M{filename}.csv")
    women_df = pd.read_csv(data_path + f"W{filename}.csv")
    return men_df, women_df

# 1️⃣ Load Tournament Results (Core Data)
m_tourney_compact, w_tourney_compact = load_mw_files("NCAATourneyCompactResults")
m_tourney_detailed, w_tourney_detailed = load_mw_files("NCAATourneyDetailedResults")

# 2️⃣ Load Tournament Seeds & Slots (Tournament Structure)
m_tourney_seeds, w_tourney_seeds = load_mw_files("NCAATourneySeeds")
m_tourney_slots, w_tourney_slots = load_mw_files("NCAATourneySlots")

# 3️⃣ Load Regular Season Data (Historical Team Strength)
m_regular_compact, w_regular_compact = load_mw_files("RegularSeasonCompactResults")
m_regular_detailed, w_regular_detailed = load_mw_files("RegularSeasonDetailedResults")

# 4️⃣ Load Team & Conference Information (Metadata)
m_teams, w_teams = load_mw_files("Teams")
m_team_conferences, w_team_conferences = load_mw_files("TeamConferences")

# 5️⃣ Load Advanced Statistics (Optional but Useful)

print("✅ Men's & Women's Data Loaded Successfully!")






m_conference=pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeamConferences.csv')
w_conference=pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeamConferences.csv')
m_ordinals=pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv')



def compute_team_stats(df):
    team_stats = df.groupby(['Season', 'WTeamID']).agg(
        Wins=('WTeamID', 'count'),
        PointsScored=('WScore', 'sum'),
        PointsAllowed=('LScore', 'sum')
    ).reset_index()
    
    # Rename for clarity
    team_stats.rename(columns={'WTeamID': 'TeamID'}, inplace=True)
    
    # Compute additional features
    team_stats['GamesPlayed'] = team_stats['Wins'] + df.groupby(['Season', 'LTeamID']).size().reset_index(name='Losses')['Losses']
    team_stats['WinRate'] = team_stats['Wins'] / team_stats['GamesPlayed']
    team_stats['AvgPointsScored'] = team_stats['PointsScored'] / team_stats['GamesPlayed']
    team_stats['AvgPointsAllowed'] = team_stats['PointsAllowed'] / team_stats['GamesPlayed']
    
    return team_stats

# Compute stats for both M & W datasets
m_team_stats = compute_team_stats(m_regular_compact)
w_team_stats = compute_team_stats(w_regular_compact)



import re

def extract_seed(df):
    df['SeedNumber'] = df['Seed'].apply(lambda x: int(re.sub("[^0-9]", "", x)))
    return df[['Season', 'TeamID', 'SeedNumber']]

# Process M & W seeds
m_seeds_clean = extract_seed(m_tourney_seeds)
w_seeds_clean = extract_seed(w_tourney_seeds)



def merge_tournament_data(tourney_df, team_stats, seeds):
    # Merge winner's stats
    tourney_df = tourney_df.merge(team_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    tourney_df = tourney_df.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('', '_Winner'))
    
    # Merge loser's stats
    tourney_df = tourney_df.merge(team_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
    tourney_df = tourney_df.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('_Winner', '_Loser'))

    # Drop redundant columns
    
    return tourney_df

# Apply for M & W datasets
m_tourney_merged = merge_tournament_data(m_tourney_compact, m_team_stats, m_seeds_clean)
w_tourney_merged = merge_tournament_data(w_tourney_compact, w_team_stats, w_seeds_clean)



def get_latest_massey_ranking(df):
    latest_rankings = df.groupby(['Season', 'TeamID']).last().reset_index()
    return latest_rankings[['Season', 'TeamID', 'OrdinalRank']]

m_latest_rankings = get_latest_massey_ranking(m_ordinals)

# Merge and explicitly drop duplicate 'TeamID' columns
m_tourney_merged = m_tourney_merged.merge(
    m_latest_rankings.rename(columns={'TeamID': 'WTeamID', 'OrdinalRank': 'OrdinalRank_Winner'}),
    on=['Season', 'WTeamID'], how='left'
)

m_tourney_merged = m_tourney_merged.merge(
    m_latest_rankings.rename(columns={'TeamID': 'LTeamID', 'OrdinalRank': 'OrdinalRank_Loser'}),
    on=['Season', 'LTeamID'], how='left'
)



import re

def extract_seed(seed):
    return int(re.sub("[^0-9]", "", seed))  # Remove non-numeric characters

m_tourney_seeds['SeedValue'] = m_tourney_seeds['Seed'].apply(extract_seed)
w_tourney_seeds['SeedValue'] = w_tourney_seeds['Seed'].apply(extract_seed)



# Compute Win % per Team
def compute_win_percentage(df):
    win_counts = df.groupby(['Season', 'WTeamID']).size().reset_index(name='Wins')
    loss_counts = df.groupby(['Season', 'LTeamID']).size().reset_index(name='Losses')

    # Merge wins & losses
    win_loss = win_counts.merge(loss_counts, left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'], how='outer')
    win_loss.fillna(0, inplace=True)

    # Compute win ratio
    win_loss['TotalGames'] = win_loss['Wins'] + win_loss['Losses']
    win_loss['WinRatio'] = win_loss['Wins'] / win_loss['TotalGames']

    return win_loss[['Season', 'WTeamID', 'WinRatio']]

m_win_ratios = compute_win_percentage(m_regular_compact)
w_win_ratios = compute_win_percentage(w_regular_compact)



print(m_tourney_merged.columns)
print(m_tourney_seeds.columns)



print(m_tourney_seeds.columns)



m_tourney_seeds = m_tourney_seeds.rename(columns={'TeamID': 'TeamID_Seed'})



m_tourney_merged = m_tourney_merged.merge(
    m_tourney_seeds[['Season', 'TeamID_Seed', 'SeedNumber']],
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID_Seed'],
    how='left'
).rename(columns={'SeedNumber': 'WSeedNumber'}).drop(columns=['TeamID_Seed'])



m_tourney_merged = m_tourney_merged.merge(
    m_tourney_seeds[['Season', 'TeamID_Seed', 'SeedNumber']],  # Use correct column names
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID_Seed'],  
    how='left'
).rename(columns={'SeedNumber': 'LSeedNumber'}).drop(columns=['TeamID_Seed'])



print(m_tourney_merged.columns)
print(m_tourney_merged[['Season', 'WTeamID', 'WSeedNumber', 'LTeamID', 'LSeedNumber']].head())



m_tourney_merged


m_tourney_merged["TeamID1"] = m_tourney_merged[["WTeamID", "LTeamID"]].min(axis=1)
m_tourney_merged["TeamID2"] = m_tourney_merged[["WTeamID", "LTeamID"]].max(axis=1)



m_tourney_merged["Target"] = (m_tourney_merged["TeamID1"] == m_tourney_merged["WTeamID"]).astype(int)



m_tourney_merged.drop(columns=["WTeamID", "LTeamID", "TeamID_Winner", "TeamID"], inplace=True)



m_tourney_merged["SeedDiff"] = m_tourney_merged["WSeedNumber"] - m_tourney_merged["LSeedNumber"]
m_tourney_merged["WinRateDiff"] = m_tourney_merged["WinRate_x"] - m_tourney_merged["WinRate_y"]



match=m_tourney_merged.copy()


match.columns


match['TeamID1'] = match[['TeamID_x', 'TeamID_y']].min(axis=1)  # Lower ID
match['TeamID2'] = match[['TeamID_x', 'TeamID_y']].max(axis=1)  # Higher ID
match['Target'] = (match['TeamID_x'] == match['TeamID1']).astype(int)  # 1 if lower ID won, 0 otherwise



match.Target.skew()


print(match.isnull().sum())



match['OrdinalRank_Winner'].fillna(match['OrdinalRank_Winner'].median(), inplace=True)
match['OrdinalRank_Loser'].fillna(match['OrdinalRank_Loser'].median(), inplace=True)



print(match.isnull().sum())



columns_to_drop = ['WScore', 'LScore', 'WLoc', 'NumOT']  # Scores & location are not needed
match.drop(columns=columns_to_drop, inplace=True)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_columns = ['WinRate_x', 'WinRate_y', 'SeedDiff', 'WinRateDiff', 
                  'AvgPointsScored_x', 'AvgPointsScored_y', 
                  'AvgPointsAllowed_x', 'AvgPointsAllowed_y']

match[scaled_columns] = scaler.fit_transform(match[scaled_columns])



from sklearn.model_selection import train_test_split

X = match.drop(columns=['Target'])  # Features
y = match['Target']  # Target column (1 if lower ID team wins, else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss

# Initialize model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predict class labels
y_pred = nb_model.predict(X_test)

# Predict probabilities
y_prob = nb_model.predict_proba(X_test)[:, 1]  # Probability of winning for TeamID1

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
log_loss_score = log_loss(y_test, y_prob)

print(f'Accuracy: {accuracy:.4f}')
print(f'Log Loss: {log_loss_score:.4f}')



from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.plot(prob_pred, prob_true, marker="o", label="Naïve Bayes")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.legend()
plt.title("Calibration Curve")
plt.show()



submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')





import pandas as pd
import itertools

# Load seeds data
men_seeds = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")
women_seeds = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv")






# Extract numeric part of seed (assuming format like 'W01' or 'M16')
def extract_seed_number(seed):
    return int(''.join(filter(str.isdigit, seed)))

men_seeds["SeedNum"] = men_seeds["Seed"].apply(extract_seed_number)
women_seeds["SeedNum"] = women_seeds["Seed"].apply(extract_seed_number)

# Generate all possible matchups for men and women
def generate_matchups(seeds_df, year, prefix):
    teams = seeds_df[seeds_df["Season"] == year][["TeamID", "SeedNum"]]
    matchups = list(itertools.combinations(teams.values, 2))  # Generate all team pairs
    matchup_df = pd.DataFrame(matchups, columns=["Team1", "Seed1", "Team2", "Seed2"])
    
    # Ensure lower TeamID comes first
    matchup_df["LowerID"], matchup_df["HigherID"] = matchup_df[["Team1", "Team2"]].min(axis=1), matchup_df[["Team1", "Team2"]].max(axis=1)
    matchup_df["SeedDiff"] = matchup_df["Seed1"] - matchup_df["Seed2"]
    
    matchup_df["ID"] = matchup_df.apply(lambda row: f"{year}_{row['LowerID']}_{row['HigherID']}", axis=1)
    return matchup_df[["ID", "SeedDiff"]]

# Generate matchups for 2025




import pandas as pd
import itertools

def generate_matchups(seeds_df, year, prefix):
    teams = seeds_df[seeds_df["Season"] == year][["TeamID", "SeedNum"]].values  # Extract as NumPy array
    matchups = [(t1[0], t1[1], t2[0], t2[1]) for t1, t2 in itertools.combinations(teams, 2)]  # Unpack elements

    matchup_df = pd.DataFrame(matchups, columns=["Team1", "Seed1", "Team2", "Seed2"])

    # Ensure lower TeamID comes first
    matchup_df["LowerTeamID"] = matchup_df[["Team1", "Team2"]].min(axis=1)
    matchup_df["HigherTeamID"] = matchup_df[["Team1", "Team2"]].max(axis=1)
    matchup_df["SeedDiff"] = matchup_df["Seed1"] - matchup_df["Seed2"]

    matchup_df["ID"] = f"{year}_" + matchup_df["LowerTeamID"].astype(str) + "_" + matchup_df["HigherTeamID"].astype(str)

    return matchup_df[["ID", "SeedDiff"]]

# Generate matchups for 2025
men_matchups = generate_matchups(men_seeds, 2025, "M")
women_matchups = generate_matchups(women_seeds, 2025, "W")

# Combine both
all_matchups = pd.concat([men_matchups, women_matchups])

print(all_matchups.head())  # Check output



w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')


def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):
    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16

# Reformat the data
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(extract_game_info).tolist()
seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# Merge seed information for TeamID1
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed information for TeamID2
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])


def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):
    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16

# Reformat the data
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(extract_game_info).tolist()
seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# Merge seed information for TeamID1
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed information for TeamID2
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])


seed_df = seed_df.drop_duplicates(subset=['Season', 'TeamID'])



submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])



from sklearn.metrics import brier_score_loss, mean_squared_error

solution_df = submission_df.copy()
solution_df['Pred'] = 1

# Now calculate the Brier score
y_true = solution_df['Pred']
y_pred = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")



submission_df.to_csv('/kaggle/working/submission.csv', index=False)


submission_df


submission_df = submission_df[['ID', 'Pred']]



submission=submission_df.copy()



submission.to_csv('/kaggle/working/submission.csv', index=False)




