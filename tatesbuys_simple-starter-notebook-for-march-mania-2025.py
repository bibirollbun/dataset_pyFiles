import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error


w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')


seed_df.head()


submission_df.head()


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


# Calculate seed difference
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']

# Update 'Pred' column
submission_df['Pred'] = 0.5 + (0.03 * submission_df['SeedDiff'])

# Drop unnecessary columns
submission_df = submission_df[['ID', 'Pred']].fillna(0.5)

# Preview your submission
submission_df.head()


stats = submission_df.iloc[:, 1].describe()
print(stats)


# Create a dataframe of ground truth values
solution_df = submission_df.copy()
solution_df['Pred'] = 1

# Now calculate the Brier score
y_true = solution_df['Pred']
y_pred = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")


submission_df.to_csv('/kaggle/working/submission.csv', index=False)

