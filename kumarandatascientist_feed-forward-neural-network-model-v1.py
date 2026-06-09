import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# ============================================================================
# 1. Set seeds for reproducibility
# ============================================================================
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ============================================================================
# 2. Prepare Seed Data and Helper Function
# ============================================================================
# Load seeds data (Women's and Men's tournaments)
w_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)

def extract_seed_value(seed_str):
    """Extracts numeric seed value from a seed string (e.g., 'W05')."""
    try:
        return int(seed_str[1:])  # remove the first character (often a letter)
    except (ValueError, TypeError):
        return 16  # default if extraction fails

seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# ============================================================================
# 3. Prepare Training Data with Polynomial Features
# ============================================================================
# Load tournament results data
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

# Create base feature: seed difference and outcome indicator
historical_df['SeedDiff'] = historical_df['SeedValue1'] - historical_df['SeedValue2']
historical_df['Outcome'] = 1  # Outcome: 1 means team with SeedValue1 (WTeamID) won

# Invert match-ups to balance the classes
inverse_df = historical_df.copy()
inverse_df[['WTeamID', 'LTeamID']] = inverse_df[['LTeamID', 'WTeamID']]
inverse_df[['SeedValue1', 'SeedValue2']] = inverse_df[['SeedValue2', 'SeedValue1']]
inverse_df['SeedDiff'] = -inverse_df['SeedDiff']
inverse_df['Outcome'] = 0

# Combine original and inverted data
training_data = pd.concat([historical_df, inverse_df])

# Add polynomial features
training_data['SeedValue1_sq'] = training_data['SeedValue1'] ** 2
training_data['SeedValue2_sq'] = training_data['SeedValue2'] ** 2
training_data['SeedProduct']   = training_data['SeedValue1'] * training_data['SeedValue2']

# Define features and target
features = ['SeedValue1', 'SeedValue2', 'SeedDiff', 'SeedValue1_sq', 'SeedValue2_sq', 'SeedProduct']
X = training_data[features].values  # Convert to NumPy array for Keras
y = training_data['Outcome'].values

# Split data (with stratification to maintain class balance)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=seed_value, stratify=y
)

# ============================================================================
# 4. Define the Optuna Objective Function
# ============================================================================
def objective(trial):
    """
    Build and train a neural network with hyperparameters suggested by Optuna.
    Returns the validation log loss.
    """
    # Suggest hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    dense_units1 = trial.suggest_int("dense_units1", 16, 64, step=16)
    dense_units2 = trial.suggest_int("dense_units2", 8, 32, step=8)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    
    # Build the model
    model = Sequential([
        Dense(dense_units1, activation='relu',
              input_shape=(X_train.shape[1],),
              kernel_regularizer=regularizers.l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(dense_units2, activation='relu',
              kernel_regularizer=regularizers.l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping callback to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                               restore_best_weights=True, verbose=0)
    
    # Train the model
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate model on validation set
    y_val_pred = model.predict(X_val).ravel()  # Flatten predictions
    val_loss = log_loss(y_val, y_val_pred)
    
    return val_loss

# ============================================================================
# 5. Run Optuna Study for 5 Trials
# ============================================================================
study = optuna.create_study(direction='minimize', 
                            sampler=optuna.samplers.TPESampler(seed=seed_value))
study.optimize(objective, n_trials=5)

print("\nOptuna Best Trial:")
trial = study.best_trial
print("  Validation Log Loss: {:.6f}".format(trial.value))
print("  Best Hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# ============================================================================
# 6. Build and Train the Final Model Using Best Hyperparameters
# ============================================================================
best_params = study.best_trial.params

final_model = Sequential([
    Dense(best_params["dense_units1"], activation='relu',
          input_shape=(X_train.shape[1],),
          kernel_regularizer=regularizers.l2(best_params["l2_reg"])),
    Dropout(best_params["dropout_rate"]),
    Dense(best_params["dense_units2"], activation='relu',
          kernel_regularizer=regularizers.l2(best_params["l2_reg"])),
    Dropout(best_params["dropout_rate"]),
    Dense(1, activation='sigmoid')
])

final_model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Use early stopping to prevent overfitting
early_stop_final = EarlyStopping(monitor='val_loss', patience=10, 
                                 restore_best_weights=True, verbose=1)

history_final = final_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=best_params["batch_size"],
    validation_data=(X_val, y_val),
    callbacks=[early_stop_final],
    verbose=1
)

# Evaluate final model on validation set
y_val_pred_final = final_model.predict(X_val).ravel()
final_val_loss = log_loss(y_val, y_val_pred_final)
print("\nFinal Model Validation Log Loss: {:.6f}".format(final_val_loss))

# ============================================================================
# 7. Prepare Submission Data and Predict
# ============================================================================
submission_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')

def extract_game_info(id_str):
    """Extracts season and team IDs from the submission ID."""
    parts = id_str.split('_')
    season = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return season, teamID1, teamID2

# Extract season and team IDs from the 'ID' column
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
)
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed info for TeamID2
submission_df = pd.merge(
    submission_df,
    seed_df[['Season', 'TeamID', 'SeedValue']],
    left_on=['Season', 'TeamID2'],
    right_on=['Season', 'TeamID'],
    how='left'
)
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# Fill missing seed values with default 16 if necessary
submission_df['SeedValue1'] = submission_df['SeedValue1'].fillna(16)
submission_df['SeedValue2'] = submission_df['SeedValue2'].fillna(16)
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']

# Create polynomial features for submission data
submission_df['SeedValue1_sq'] = submission_df['SeedValue1'] ** 2
submission_df['SeedValue2_sq'] = submission_df['SeedValue2'] ** 2
submission_df['SeedProduct']   = submission_df['SeedValue1'] * submission_df['SeedValue2']

# Prepare the feature matrix for prediction
submission_features = submission_df[features].values
submission_df['Pred'] = final_model.predict(submission_features).ravel()

# ============================================================================
# 8. (Optional) Evaluate Predictions with Dummy Ground Truth and Save Submission
# ============================================================================
# (Replace the dummy ground truth with actual values when available.)
solution_df = submission_df.copy()
solution_df['Pred'] = 1  # Dummy ground truth for demonstration purposes
y_true_dummy = solution_df['Pred']
y_pred_submission = submission_df['Pred']
brier = brier_score_loss(y_true_dummy, y_pred_submission)
print("Brier Score (with dummy ground truth):", brier)

# Save the submission file
submission_df[['ID', 'Pred']].fillna(0.5).to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved.")


