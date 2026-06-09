import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

players_df = pd.read_csv('/kaggle/input/databowl/players.csv')
plays_df = pd.read_csv('/kaggle/input/databowl/plays.csv')
tracking_week1_df = pd.read_csv('/kaggle/input/databowl/tracking_week_1.csv')
tracking_week2_df = pd.read_csv('/kaggle/input/databowl/tracking_week_2.csv')
submission_df = pd.DataFrame({
    "Id": [1, 2, 3],
    "Prediction": [0.5, 0.6, 0.7]
})


# Filter players dataset for WRs, RBs, and TEs
relevant_positions = ["WR", "RB", "TE"]
players_filtered = players_df[players_df['position'].isin(relevant_positions)]
players_filtered


# Extract relevant tracking data
tracking_data = pd.concat([tracking_week1_df, tracking_week2_df])
tracking_data_filtered = tracking_data[tracking_data['nflId'].isin(players_filtered['nflId'])]
tracking_data_filtered


# Determine the frame just before the snap for each play
snap_frames = tracking_data_filtered[tracking_data_filtered['event'] == 'ball_snap'].groupby(['gameId', 'playId']).first().reset_index()
snap_frames


# Calculate the total distance moved by each player from the start of the tracking to the snap frame
pre_snap_movement = pd.merge(tracking_data_filtered, snap_frames[['gameId', 'playId', 'frameId']], on=['gameId', 'playId'], how='left')
pre_snap_movement = pre_snap_movement[pre_snap_movement['frameId_x'] <= pre_snap_movement['frameId_y']]
pre_snap_movement = pre_snap_movement.groupby(['gameId', 'playId'])['dis'].sum().reset_index()
pre_snap_movement


passing_plays = plays_df[plays_df['passResult'].notna()]
passing_plays


merged_data = pd.merge(passing_plays, pre_snap_movement, on=['gameId', 'playId'])
merged_data


merged_data['completionFlag'] = merged_data['passResult'].apply(lambda x: 1 if x == 'C' else 0)
merged_data


# Calculate Pearson correlation coefficient
pearson_corr = merged_data['dis'].corr(merged_data['completionFlag'], method='pearson')

# Calculate Spearman correlation coefficient
spearman_corr = merged_data['dis'].corr(merged_data['completionFlag'], method='spearman')

pearson_corr, spearman_corr


from sklearn.linear_model import LogisticRegression

# Define the independent variable (pre-snap movement distance) and the dependent variable (pass completion flag)
X = merged_data[['dis']]
y = merged_data['completionFlag']

# Initialize the logistic regression model
log_reg = LogisticRegression()

# Fit the model to the data
log_reg.fit(X, y)

# Get the coefficient of the pre-snap movement distance
coef = log_reg.coef_[0][0]

coef

# Generate completion probabilities
merged_data['completionProbability'] = log_reg.predict_proba(X)[:, 1]

# Prepare the submission DataFrame
submission_df = merged_data[['playId', 'completionProbability']].copy()
submission_df.rename(columns={'playId': 'id', 'completionProbability': 'prediction'}, inplace=True)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Verify the submission
test_submission = pd.read_csv('submission.csv')
print(test_submission.head())


submission_df.to_csv('submission.csv', index=False)

