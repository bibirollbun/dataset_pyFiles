import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

#########################################
# 1. Prepare Seed Data and Helper Functions
#########################################

# Load seed data for women and men tournaments
w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)

# Helper function to extract numerical seed value from the seed string
def extract_seed_value(seed_str):
    try:
        return int(seed_str[1:])  # remove the first character (often a letter)
    except ValueError:
        return 16  # default to worst seed if extraction fails

seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

#########################################
# 2. Train the Gradient Boosting Classifier
#########################################

# Load historical matchup data for both men's and women's tournaments
w_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
m_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
historical_df = pd.concat([m_results, w_results], axis=0)

# --- Merge seed values for winners ---
historical_df = pd.merge(
    historical_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
historical_df = historical_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# --- Merge seed values for losers ---
historical_df = pd.merge(
    historical_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
historical_df = historical_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# Create a feature: seed difference
historical_df['SeedDiff'] = historical_df['SeedValue1'] - historical_df['SeedValue2']
# Define outcome: 1 indicates that the team in the WTeamID (with SeedValue1) won
historical_df['Outcome'] = 1

# --- Invert match-ups to balance the dataset ---
inverse_df = historical_df.copy()
inverse_df[['WTeamID', 'LTeamID']] = inverse_df[['LTeamID', 'WTeamID']]
inverse_df[['SeedValue1', 'SeedValue2']] = inverse_df[['SeedValue2', 'SeedValue1']]
inverse_df['SeedDiff'] = -inverse_df['SeedDiff']
inverse_df['Outcome'] = 0

# Combine the original and inverted data
final_training_data = pd.concat([historical_df, inverse_df])

# ---- Add polynomial features ----
final_training_data['SeedValue1_sq'] = final_training_data['SeedValue1'] ** 2
final_training_data['SeedValue2_sq'] = final_training_data['SeedValue2'] ** 2
final_training_data['SeedProduct'] = final_training_data['SeedValue1'] * final_training_data['SeedValue2']

# Define the feature set including our new polynomial features
features = ['SeedValue1', 'SeedValue2', 'SeedDiff', 'SeedValue1_sq', 'SeedValue2_sq', 'SeedProduct']
X = final_training_data[features]
y = final_training_data['Outcome']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Hyperparameter tuning via GridSearchCV ----
param_grid = {
    'n_estimators': [150, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3],
    'subsample': [0.9, 1.0],
    'min_samples_leaf': [1, 2]
}
gbc = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gbc, param_grid, scoring='neg_log_loss', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Select the best model
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the best model on the validation set
y_val_pred = best_model.predict_proba(X_val)[:, 1]
val_log_loss = log_loss(y_val, y_val_pred)
print("Validation Log Loss:", val_log_loss)

#########################################
# 3. Prepare Submission Data and Predict
#########################################

# Load the submission file
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')

# Helper function to extract game info from the submission ID string
def extract_game_info(id_str):
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

# Create Season, TeamID1, and TeamID2 columns
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(
    lambda x: pd.Series(extract_game_info(x))
)

# --- Merge seed information for TeamID1 ---
submission_df = pd.merge(
    submission_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID1'],
    right_on=['Season', 'TeamID'],
    how='left'
)
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# --- Merge seed information for TeamID2 ---
submission_df = pd.merge(
    submission_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID2'],
    right_on=['Season', 'TeamID'],
    how='left'
)
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# Fill any missing seed values with 16 (assume worst-case)
submission_df['SeedValue1'] = submission_df['SeedValue1'].fillna(16)
submission_df['SeedValue2'] = submission_df['SeedValue2'].fillna(16)
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']

# ---- Add polynomial features for submission data ----
submission_df['SeedValue1_sq'] = submission_df['SeedValue1'] ** 2
submission_df['SeedValue2_sq'] = submission_df['SeedValue2'] ** 2
submission_df['SeedProduct'] = submission_df['SeedValue1'] * submission_df['SeedValue2']

# Prepare features and make predictions
submission_features = submission_df[features]
submission_df['Pred'] = best_model.predict_proba(submission_features)[:, 1]

#########################################
# 4. (Optional) Evaluate Predictions
#########################################

# (This is just a demonstration; in a real submission the ground truth is unknown.)
# Here we create a dummy ground truth with all ones.
solution_df = submission_df.copy()
solution_df['Pred'] = 1  
y_true = solution_df['Pred']
y_pred_submission = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred_submission)
print("Brier Score:", brier_score)

#########################################
# 5. Save the Submission File
#########################################

submission_df = submission_df[['ID', 'Pred']].fillna(0.5)
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved.")


