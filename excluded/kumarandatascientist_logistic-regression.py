import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import joblib  # used by GridSearchCV
import category_encoders as ce  # target encoder
from contextlib import contextmanager

# ---------------------------
# Custom tqdm_joblib Context Manager
# ---------------------------
@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given by `tqdm_object`.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# ---------------------------
# 1. Prepare Seed Data and Helper Function
# ---------------------------
w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)

def extract_seed_value(seed_str):
    try:
        return int(seed_str[1:])  # remove first character (often a letter)
    except ValueError:
        return 16  # default if extraction fails

seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# ---------------------------
# 2. Prepare Training Data with Polynomial & Target-Encoded Features
# ---------------------------
w_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
m_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
historical_df = pd.concat([m_results, w_results], axis=0)

# Merge seed values for winning teams (WTeamID)
historical_df = pd.merge(
    historical_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
historical_df = historical_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed values for losing teams (LTeamID)
historical_df = pd.merge(
    historical_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'],
    how='left'
)
historical_df = historical_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# Create base feature: seed difference and outcome (1 means the first team won)
historical_df['SeedDiff'] = historical_df['SeedValue1'] - historical_df['SeedValue2']
historical_df['Outcome'] = 1

# Invert match-ups to balance the classes
inverse_df = historical_df.copy()
inverse_df[['WTeamID', 'LTeamID']] = inverse_df[['LTeamID', 'WTeamID']]
inverse_df[['SeedValue1', 'SeedValue2']] = inverse_df[['SeedValue2', 'SeedValue1']]
inverse_df['SeedDiff'] = -inverse_df['SeedDiff']
inverse_df['Outcome'] = 0

# Combine original and inverted data
training_data = pd.concat([historical_df, inverse_df])

# ---------------------------
# Add polynomial features
# ---------------------------
training_data['SeedValue1_sq'] = training_data['SeedValue1'] ** 2
training_data['SeedValue2_sq'] = training_data['SeedValue2'] ** 2
training_data['SeedProduct']   = training_data['SeedValue1'] * training_data['SeedValue2']

# ---------------------------
# Add team ID features for target encoding
# ---------------------------
# We use the original matchup order as features:
training_data['TeamID1'] = training_data['WTeamID']
training_data['TeamID2'] = training_data['LTeamID']

# Define feature list including both numeric and categorical features
features = ['SeedValue1', 'SeedValue2', 'SeedDiff', 
            'SeedValue1_sq', 'SeedValue2_sq', 'SeedProduct', 
            'TeamID1', 'TeamID2']

X = training_data[features]
y = training_data['Outcome']

# ---------------------------
# Apply target encoding on the categorical team ID columns
# ---------------------------
encoder = ce.TargetEncoder(cols=['TeamID1', 'TeamID2'])
X_encoded = encoder.fit_transform(X, y)

# Split data (using stratification)
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 3. Hyperparameter Tuning with Logistic Regression using GridSearchCV
# ---------------------------
# We tune the regularization strength "C". (Lower C = stronger regularization.)
lr = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid_search_lr = GridSearchCV(lr, param_grid, scoring='neg_log_loss', cv=5, n_jobs=-1, verbose=0)
total_iter_lr = len(param_grid['C']) * 5  # number of candidates * 5-fold CV

with tqdm_joblib(tqdm(desc="Logistic Regression Grid Search", total=total_iter_lr)):
    grid_search_lr.fit(X_train, y_train)

print("Best Logistic Regression Hyperparameters:", grid_search_lr.best_params_)
best_lr = grid_search_lr.best_estimator_

y_val_pred = best_lr.predict_proba(X_val)[:, 1]
val_log_loss = log_loss(y_val, y_val_pred)
print("Validation Log Loss:", val_log_loss)

# ---------------------------
# 4. Prepare Submission Data, Apply Target Encoding, and Predict
# ---------------------------
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')

def extract_game_info(id_str):
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(
    lambda x: pd.Series(extract_game_info(x))
)

# Merge seed info for TeamID1
submission_df = pd.merge(
    submission_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID1'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed info for TeamID2
submission_df = pd.merge(
    submission_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID2'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# If seed values are missing, default them to 16.
submission_df['SeedValue1'] = submission_df['SeedValue1'].fillna(16)
submission_df['SeedValue2'] = submission_df['SeedValue2'].fillna(16)
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']

submission_df['SeedValue1_sq'] = submission_df['SeedValue1'] ** 2
submission_df['SeedValue2_sq'] = submission_df['SeedValue2'] ** 2
submission_df['SeedProduct']   = submission_df['SeedValue1'] * submission_df['SeedValue2']

# For submission, we also include the team IDs as categorical features.
submission_features = submission_df[features]

# Apply the same target encoder that was fit on the training data.
submission_features_encoded = encoder.transform(submission_features)

submission_df['Pred'] = best_lr.predict_proba(submission_features_encoded)[:, 1]

# ---------------------------
# 5. (Optional) Evaluate Predictions with Dummy Ground Truth
# ---------------------------
solution_df = submission_df.copy()
solution_df['Pred'] = 1  # Dummy ground truth; replace with real values if available.
y_true = solution_df['Pred']
y_pred_submission = submission_df['Pred']
brier = brier_score_loss(y_true, y_pred_submission)
print("Brier Score:", brier)

# ---------------------------
# 6. Save the Submission File
# ---------------------------
submission_df[['ID', 'Pred']].fillna(0.5).to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved.")


