# load libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# Load data
m_tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
w_tourney_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
m_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
w_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')


# Basic data exploration
print(m_tourney_results.head())
print(w_tourney_results.head())
print(m_seeds.head())
print(w_seeds.head())


# Merge seeds with results
m_tourney_results = m_tourney_results.merge(m_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
m_tourney_results = m_tourney_results.merge(m_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))
w_tourney_results = w_tourney_results.merge(w_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
w_tourney_results = w_tourney_results.merge(w_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))


# Feature engineering
m_tourney_results['SeedDiff'] = m_tourney_results['Seed_L'].apply(lambda x: int(x[1:3])) - m_tourney_results['Seed_W'].apply(lambda x: int(x[1:3]))
w_tourney_results['SeedDiff'] = w_tourney_results['Seed_L'].apply(lambda x: int(x[1:3])) - w_tourney_results['Seed_W'].apply(lambda x: int(x[1:3]))


# Prepare training data
X_m = m_tourney_results[['SeedDiff']]
y_m = m_tourney_results['WTeamID'] < m_tourney_results['LTeamID']
X_w = w_tourney_results[['SeedDiff']]
y_w = w_tourney_results['WTeamID'] < w_tourney_results['LTeamID']


# Split data into training and testing sets
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.2, random_state=42)


# Train logistic regression model
model_m = LogisticRegression()
model_m.fit(X_train_m, y_train_m)
model_w = LogisticRegression()
model_w.fit(X_train_w, y_train_w)


# Make predictions
preds_m = model_m.predict_proba(X_test_m)[:, 1]
preds_w = model_w.predict_proba(X_test_w)[:, 1]


# Evaluate model
brier_score_m = brier_score_loss(y_test_m, preds_m)
brier_score_w = brier_score_loss(y_test_w, preds_w)
print(f'Brier score for men\'s model: {brier_score_m}')
print(f'Brier score for women\'s model: {brier_score_w}')


""" import pandas as pd
import itertools

teams = pd.concat([m_seeds[['Season', 'TeamID', 'Seed']], w_seeds[['Season', 'TeamID', 'Seed']]])
teams = teams.drop_duplicates()

# Function to process a batch of team pairs
def process_batch(batch, season, model_m, model_w):
    results = []
    for team1, team2 in batch:
        seed1 = teams.loc[teams['TeamID'] == team1, 'Seed'].values[0]
        seed2 = teams.loc[teams['TeamID'] == team2, 'Seed'].values[0]
        seed_diff = int(seed2[1:3]) - int(seed1[1:3])  # Extract numeric part of seed

        pred = model_m.predict_proba([[seed_diff]])[0][1] if team1 < 2000 else model_w.predict_proba([[seed_diff]])[0][1]
        results.append([f'{season}_{team1}_{team2}', pred])
    return results

# Process in chunks and write directly to CSV
batch_size = 1000
csv_filename = 'submission.csv'

# Initialize CSV with header
pd.DataFrame(columns=['ID', 'Pred']).to_csv(csv_filename, index=False)

batch_count = 0  # Counter for number of batches

season = 2025
team_pairs = list(itertools.combinations(teams['TeamID'], 2))  # Generate team pairs
team_pairs = list(set(team_pairs))  # Remove duplicates
batch = []  # Temporary batch storage

for pair in team_pairs:
    batch.append(pair)

    # Process when batch reaches batch_size
    if len(batch) == batch_size:
        batch_results = process_batch(batch, season, model_m, model_w)

        # Append results to CSV in chunks
        pd.DataFrame(batch_results, columns=['ID', 'Pred']).to_csv(csv_filename, mode='a', header=False, index=False)

        batch_count += 1
        print(f"Processed batch {batch_count}")  # Print batch count

        batch = []  # Reset batch

# Process any remaining pairs in batch (if not exactly batch_size)
if batch:
    batch_results = process_batch(batch, season, model_m, model_w)
    pd.DataFrame(batch_results, columns=['ID', 'Pred']).to_csv(csv_filename, mode='a', header=False, index=False)

    batch_count += 1
    print(f"Processed batch {batch_count}")  # Print batch count """

