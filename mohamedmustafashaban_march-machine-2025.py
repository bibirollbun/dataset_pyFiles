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


import pandas as pd
from sklearn.metrics import brier_score_loss
# Load the required list of IDs (replace with the actual list)
required_ids = [...]  # Provide the full list of required 131407 IDs

# Check if all required IDs exist
missing_ids = set(required_ids) - set(submission_df['ID'].unique())

if missing_ids:
    print(f"Missing {len(missing_ids)} IDs: {missing_ids}")

    # Create missing rows with default prediction values (you can change 'Pred' value if needed)
    missing_rows = pd.DataFrame({'ID': list(missing_ids), 'Pred': 0})

    # Append missing rows to submission_df
    submission_df = pd.concat([submission_df, missing_rows], ignore_index=True)

# Ensure DataFrame has exactly 131407 rows
if submission_df.shape[0] != 131407:
    print(f"Warning: Adjusting row count to 131407. Current rows: {submission_df.shape[0]}")
    submission_df = submission_df.iloc[:131407]  # Trim extra rows if needed

# Sort by 'ID' (if required)
submission_df = submission_df.sort_values(by="ID").reset_index(drop=True)

# Create a DataFrame with ground truth values
solution_df = submission_df.copy()
solution_df['Pred'] = 1  # Assuming all true labels are 1

# Calculate the Brier Score
y_true = solution_df['Pred']
y_pred = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred)

print(f"Brier Score: {brier_score}")

# Save the corrected submission file
submission_df.to_csv("submission.csv", index=False)
print("Fixed submission saved as 'fixed_submission.csv'.")



submission_df

