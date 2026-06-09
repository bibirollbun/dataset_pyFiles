import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.metrics import brier_score_loss


data_dir = "/kaggle/input/march-machine-learning-mania-2025/"


import os
for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))


m_results = pd.read_csv(data_dir+'/MNCAATourneyCompactResults.csv')
m_results = m_results[m_results['Season'] >= 2021]
m_results


w_results = pd.read_csv(data_dir+'/WNCAATourneyCompactResults.csv')
w_results = w_results[w_results['Season'] >= 2021]
w_results


results = pd.concat([m_results, w_results], ignore_index=True)
results


results['Team1'] = results[['WTeamID', 'LTeamID']].min(axis=1)
results['Team2'] = results[['WTeamID', 'LTeamID']].max(axis=1)
results


results['submission_ID'] = results['Season'].apply(str) + '_' + results['Team1'].apply(str) + '_' + results['Team2'].apply(str)
results


results['Truth'] = (results['Team1'] == results['WTeamID']).astype(int)
results


final_gt = results[['submission_ID', 'Truth']]
final_gt


def evaluate_submission_on_stage1(submission_df):
    comparison_df = submission_df.merge(final_gt, left_on='ID', right_on='submission_ID', how='inner')
    y_true = comparison_df['Truth']
    y_pred = comparison_df['Pred']
    brier_score = brier_score_loss(y_true, y_pred)
    print(f"Brier Score: {brier_score:.3f}")


submission = pd.read_csv(data_dir+'/SampleSubmissionStage1.csv')
submission


evaluate_submission_on_stage1(submission)


# Concat M and W data
m_seeds = pd.read_csv(data_dir + "MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(data_dir + "WNCAATourneySeeds.csv")
seeds_df = pd.concat([m_seeds, w_seeds], ignore_index=True)
seeds_df


# Transform seeds to integers
seeds_df['seed_int'] = seeds_df['Seed'].apply(lambda x: int(x[1:3]))
seeds_df = seeds_df[['Season', 'TeamID', 'seed_int']]
seeds_df


baseline = submission.copy()


# Preprocessing before merge
baseline['Season'] = baseline['ID'].apply(lambda x: int(x.split('_')[0]))
baseline['Team1'] = baseline['ID'].apply(lambda x: int(x.split('_')[1]))
baseline['Team2'] = baseline['ID'].apply(lambda x: int(x.split('_')[2]))
baseline


# Getting seeds for Team1
baseline = pd.merge(baseline, seeds_df, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
baseline.rename(columns={'seed_int': 'Team1Seed'}, inplace=True)
baseline.drop(columns=['TeamID'], inplace=True)

# Getting seeds for Team2
baseline = pd.merge(baseline, seeds_df, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')
baseline.rename(columns={'seed_int': 'Team2Seed'}, inplace=True)
baseline.drop(columns=['TeamID'], inplace=True)
baseline


baseline['SeedDiff'] = baseline['Team1Seed'] - baseline['Team2Seed']
baseline


baseline['SeedDiff'].fillna(0, inplace=True)


baseline['Pred'] = (-np.sign(baseline['SeedDiff']) + 1)/2.0
baseline


baseline_sub = baseline[['ID', 'Pred']]
baseline_sub


evaluate_submission_on_stage1(baseline_sub)


baseline2 = baseline.copy()


# Smoothing predictions based on SeedDiff values 
# Pred = 1 for maximum positive SeedDiff (1 vs 16)
# Pred = 0.5 for SeedDiff = 0 (same SeedDiff or unknown seeds)
# Pred = 0 for maximum negative SeedDiff (16 vs 1)
baseline2['Pred'] = 0.5 - 0.5*baseline2['SeedDiff']/15.0
baseline2


baseline2_sub = baseline2[['ID', 'Pred']]
baseline2_sub


evaluate_submission_on_stage1(baseline2_sub)


submission2 = pd.read_csv(data_dir+'/SampleSubmissionStage2.csv')
submission2


# Concat M and W data
m_seeds = pd.read_csv(data_dir + "MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(data_dir + "WNCAATourneySeeds.csv")
seeds_df = pd.concat([m_seeds, w_seeds], ignore_index=True)

# Transform seeds to integers
seeds_df['seed_int'] = seeds_df['Seed'].apply(lambda x: int(x[1:3]))
seeds_df = seeds_df[['Season', 'TeamID', 'seed_int']]

baseline = submission2.copy()

# Preprocessing before merge
baseline['Season'] = baseline['ID'].apply(lambda x: int(x.split('_')[0]))
baseline['Team1'] = baseline['ID'].apply(lambda x: int(x.split('_')[1]))
baseline['Team2'] = baseline['ID'].apply(lambda x: int(x.split('_')[2]))

# Getting seeds for Team1
baseline = pd.merge(baseline, seeds_df, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
baseline.rename(columns={'seed_int': 'Team1Seed'}, inplace=True)
baseline.drop(columns=['TeamID'], inplace=True)

# Getting seeds for Team2
baseline = pd.merge(baseline, seeds_df, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')
baseline.rename(columns={'seed_int': 'Team2Seed'}, inplace=True)
baseline.drop(columns=['TeamID'], inplace=True)

# Compute SeedDiff
baseline['SeedDiff'] = baseline['Team1Seed'] - baseline['Team2Seed']
baseline['SeedDiff'].fillna(0, inplace=True)

# Create prediction based on Seed Diff
baseline['Pred'] = 0.5 - 0.5*baseline['SeedDiff']/15.0

# Get final submission df
baseline2_sub = baseline[['ID', 'Pred']]
baseline2_sub


baseline2_sub[baseline2_sub['Pred'] != 0.5]


baseline2_sub.to_csv('submission.csv', index=False)




