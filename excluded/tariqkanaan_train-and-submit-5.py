import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Lambda, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report

# =============================================================================
# Data Preparation Function (Training)
# =============================================================================
def prepare_matchup_data(results_file, game_cities_file, gender_label):
    """
    Process season results and game cities files to produce matchup pairs.
    Adds features such as rolling averages, travel distance, and pads sequences.
    """
    # Load the datasets.
    results = pd.read_csv(results_file)
    game_cities = pd.read_csv(game_cities_file)
    
    # Build wins dataframe.
    wins = results[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore', 'WLoc', 'NumOT']].copy()
    wins['Win'] = 1
    wins = wins.rename(columns={'WTeamID': 'TeamID',
                                'WScore': 'PointsFor',
                                'LScore': 'PointsAgainst'})
    wins = wins.merge(
        game_cities[['Season', 'DayNum', 'WTeamID', 'CityID']],
        left_on=['Season', 'DayNum', 'TeamID'],
        right_on=['Season', 'DayNum', 'WTeamID'],
        how='left'
    ).drop(columns=['WTeamID'])
    
    # Build losses dataframe.
    losses = results[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore', 'WLoc', 'NumOT']].copy()
    losses['Win'] = 0
    losses = losses.rename(columns={'LTeamID': 'TeamID',
                                    'LScore': 'PointsFor',
                                    'WScore': 'PointsAgainst'})
    losses = losses.merge(
        game_cities[['Season', 'DayNum', 'LTeamID', 'CityID']],
        left_on=['Season', 'DayNum', 'TeamID'],
        right_on=['Season', 'DayNum', 'LTeamID'],
        how='left'
    ).drop(columns=['LTeamID'])
    
    # Combine wins and losses.
    team_games = pd.concat([wins, losses], axis=0)
    team_games = team_games.sort_values(['Season', 'TeamID', 'DayNum']).reset_index(drop=True)
    
    # Feature engineering.
    team_games['HomeGame'] = (team_games['WLoc'] == 'H').astype(int)
    window = 5
    team_games['MA_PointsFor'] = team_games.groupby(['Season', 'TeamID'])['PointsFor'] \
                                          .transform(lambda x: x.rolling(window, min_periods=1).mean())
    team_games['MA_PointsAgainst'] = team_games.groupby(['Season', 'TeamID'])['PointsAgainst'] \
                                              .transform(lambda x: x.rolling(window, min_periods=1).mean())
    team_games['MA_WinRate'] = team_games.groupby(['Season', 'TeamID'])['Win'] \
                                         .transform(lambda x: x.rolling(window, min_periods=1).mean())
    team_games['DaysSinceLast'] = team_games.groupby(['Season', 'TeamID'])['DayNum'].diff().fillna(0)
    
    # Add travel distance: assume 0 for home and 50 for away.
    team_games['TravelDistance'] = np.where(team_games['HomeGame'] == 1, 0, 50)
    
    # Define sequence features (order matters for the model input).
    seq_features = ['MA_PointsFor', 'MA_PointsAgainst', 'MA_WinRate', 
                    'HomeGame', 'TravelDistance', 'DaysSinceLast']
    team_games[seq_features] = team_games[seq_features].fillna(0)
    
    # Pad sequences to fixed length (10 games).
    def pad_sequence(seq, n=10, pad_value=0):
        if seq.shape[0] >= n:
            return seq[-n:]
        else:
            pad = np.full((n - seq.shape[0], seq.shape[1]), pad_value)
            return np.vstack([pad, seq])
    
    team_sequences = team_games.groupby(['Season', 'TeamID'])[seq_features] \
        .apply(lambda df: pad_sequence(df.values, n=10)) \
        .reset_index(name='Sequence')
    
    # =============================================================================
    # Build Matchup Pairs
    # =============================================================================
    matchup_df = results[['Season', 'DayNum', 'WTeamID', 'LTeamID']].copy()
    
    # Merge to get the winning team's sequence.
    matchup_df = matchup_df.merge(team_sequences[['Season', 'TeamID', 'Sequence']],
                                  left_on=['Season', 'WTeamID'],
                                  right_on=['Season', 'TeamID'],
                                  how='left') \
                           .rename(columns={'Sequence': 'TeamA_Sequence'}) \
                           .drop(columns=['TeamID'])
    
    # Merge to get the losing team's sequence.
    matchup_df = matchup_df.merge(team_sequences[['Season', 'TeamID', 'Sequence']],
                                  left_on=['Season', 'LTeamID'],
                                  right_on=['Season', 'TeamID'],
                                  how='left') \
                           .rename(columns={'Sequence': 'TeamB_Sequence'}) \
                           .drop(columns=['TeamID'])
    
    # Label 1 indicates Team A wins (by construction).
    matchup_df['Label'] = 1
    # Create negative examples by swapping team sequences.
    swapped = matchup_df.copy().rename(columns={'TeamA_Sequence': 'TeamB_Sequence',
                                                 'TeamB_Sequence': 'TeamA_Sequence'})
    swapped['Label'] = 0
    combined_matchups = pd.concat([matchup_df, swapped], axis=0).reset_index(drop=True)
    
    # Add the gender label.
    combined_matchups['Gender'] = gender_label
    return combined_matchups

# Process Men's and Women's data.
men_matchups = prepare_matchup_data(
    '/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv',
    '/kaggle/input/march-machine-learning-mania-2025/MGameCities.csv',
    'Men'
)

women_matchups = prepare_matchup_data(
    '/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv',
    '/kaggle/input/march-machine-learning-mania-2025/WGameCities.csv',
    'Women'
)

combined_all_matchups = pd.concat([men_matchups, women_matchups], axis=0).reset_index(drop=True)

# Optional: Verify that each team sequence is of the expected shape (10, 6).
def check_sequence_shapes(seq_series, expected_shape=(10, 6)):
    for i, seq in enumerate(seq_series):
        if seq.shape != expected_shape:
            print(f"Sequence {i} has shape {seq.shape}")

check_sequence_shapes(combined_all_matchups['TeamA_Sequence'])

# Prepare arrays for training.
X_team_a = np.stack(combined_all_matchups['TeamA_Sequence'].values)  # Shape: (N, 10, 6)
X_team_b = np.stack(combined_all_matchups['TeamB_Sequence'].values)  # Shape: (N, 10, 6)
y = combined_all_matchups['Label'].values

print("\nCombined All Matchups Sample:")
print(combined_all_matchups.head())
print("\nLabel Counts:")
print(combined_all_matchups['Label'].value_counts())
print("\nGender Counts:")
print(combined_all_matchups['Gender'].value_counts())
print("\nShapes:")
print("Team A sequences:", X_team_a.shape)
print("Team B sequences:", X_team_b.shape)
print("Labels:", y.shape)

# =============================================================================
# Revised Model Architecture with Asymmetric Merging
# =============================================================================
def build_team_encoder(input_shape):
    """
    Defines a simple encoder to process a team's sequence.
    """
    inp = Input(shape=input_shape)
    x = LSTM(32, return_sequences=True)(inp)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    return Model(inp, x, name="TeamEncoder")

team_input_shape = (10, 6)
input_a = Input(shape=team_input_shape, name="team_a_seq")
input_b = Input(shape=team_input_shape, name="team_b_seq")

# Build the shared team encoder.
team_encoder = build_team_encoder(team_input_shape)
encoded_a = team_encoder(input_a)
encoded_b = team_encoder(input_b)

# --- Asymmetric Merging Block ---
raw_concat = Concatenate()([encoded_a, encoded_b])
diff = Lambda(lambda x: x[0] - x[1])([encoded_a, encoded_b])
abs_diff = Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_a, encoded_b])
elem_mul = Lambda(lambda x: x[0] * x[1])([encoded_a, encoded_b])
merged_features = Concatenate()([raw_concat, diff, abs_diff, elem_mul])

# Final dense layers.
x = Dense(128, activation='relu')(merged_features)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid', name="win_probability")(x)

model = Model(inputs=[input_a, input_b], outputs=output, name="Matchup_Predictor")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =============================================================================
# Training the Revised Model
# =============================================================================
history = model.fit(
    [X_team_a, X_team_b],
    y,
    epochs=200,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# =============================================================================
# Model Evaluation on Real Data
# =============================================================================
# For validation, select a specific season (e.g., 2024).
validation_season = 2024
validation_games = combined_all_matchups[combined_all_matchups['Season'] == validation_season]

if validation_games.empty:
    raise ValueError("No games found for the specified validation season. Adjust the season or split method.")

X_val_a = np.stack(validation_games['TeamA_Sequence'].values)
X_val_b = np.stack(validation_games['TeamB_Sequence'].values)
y_val = validation_games['Label'].values

# Generate win probability predictions.
y_pred_probs = model.predict([X_val_a, X_val_b])
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Compute evaluation metrics.
accuracy = accuracy_score(y_val, y_pred)
loss_val = log_loss(y_val, y_pred_probs)
auc_val = roc_auc_score(y_val, y_pred_probs)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation Log Loss: {loss_val:.4f}")
print(f"Validation AUC-ROC: {auc_val:.4f}")

# Detailed classification report.
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["Loss", "Win"]))

# Plot the Confusion Matrix.
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Loss", "Predicted Win"],
            yticklabels=["Actual Loss", "Actual Win"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# After training is completed
model.save('my_model_1.keras')  # Saves the model in HDF5 format




import pandas as pd
import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# Helper Function: Generate Team Sequences for a Single Dataset
# -----------------------------------------------------------------------------
def get_team_sequences(results_file, game_cities_file):
    """
    Reads results and game cities data, computes rolling averages and other features,
    pads the sequences to 10 games, and returns a DataFrame with one row per team per season.
    """
    results = pd.read_csv(results_file)
    game_cities = pd.read_csv(game_cities_file)
    
    # Process wins.
    wins = results[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore', 'WLoc', 'NumOT']].copy()
    wins['Win'] = 1
    wins = wins.rename(columns={'WTeamID': 'TeamID',
                                'WScore': 'PointsFor',
                                'LScore': 'PointsAgainst'})
    wins = wins.merge(
        game_cities[['Season', 'DayNum', 'WTeamID', 'CityID']],
        left_on=['Season', 'DayNum', 'TeamID'],
        right_on=['Season', 'DayNum', 'WTeamID'],
        how='left'
    ).drop(columns=['WTeamID'])
    
    # Process losses.
    losses = results[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore', 'WLoc', 'NumOT']].copy()
    losses['Win'] = 0
    losses = losses.rename(columns={'LTeamID': 'TeamID',
                                    'LScore': 'PointsFor',
                                    'WScore': 'PointsAgainst'})
    losses = losses.merge(
        game_cities[['Season', 'DayNum', 'LTeamID', 'CityID']],
        left_on=['Season', 'DayNum', 'TeamID'],
        right_on=['Season', 'DayNum', 'LTeamID'],
        how='left'
    ).drop(columns=['LTeamID'])
    
    # Combine wins and losses.
    team_games = pd.concat([wins, losses], axis=0)
    team_games = team_games.sort_values(['Season', 'TeamID', 'DayNum']).reset_index(drop=True)
    
    # Feature engineering.
    team_games['HomeGame'] = (team_games['WLoc'] == 'H').astype(int)
    window = 5
    team_games['MA_PointsFor'] = team_games.groupby(['Season', 'TeamID'])['PointsFor']\
                                          .transform(lambda x: x.rolling(window, min_periods=1).mean())
    team_games['MA_PointsAgainst'] = team_games.groupby(['Season', 'TeamID'])['PointsAgainst']\
                                              .transform(lambda x: x.rolling(window, min_periods=1).mean())
    team_games['MA_WinRate'] = team_games.groupby(['Season', 'TeamID'])['Win']\
                                         .transform(lambda x: x.rolling(window, min_periods=1).mean())
    team_games['DaysSinceLast'] = team_games.groupby(['Season', 'TeamID'])['DayNum'].diff().fillna(0)
    team_games['TravelDistance'] = np.where(team_games['HomeGame'] == 1, 0, 50)
    
    # Define sequence features.
    seq_features = ['MA_PointsFor', 'MA_PointsAgainst', 'MA_WinRate', 
                    'HomeGame', 'TravelDistance', 'DaysSinceLast']
    team_games[seq_features] = team_games[seq_features].fillna(0)
    
    # Pad sequences to fixed length (10 games).
    def pad_sequence(seq, n=10, pad_value=0):
        if seq.shape[0] >= n:
            return seq[-n:]
        else:
            pad = np.full((n - seq.shape[0], seq.shape[1]), pad_value)
            return np.vstack([pad, seq])
    
    team_sequences = team_games.groupby(['Season', 'TeamID'])[seq_features]\
        .apply(lambda df: pad_sequence(df.values, n=10))\
        .reset_index(name='Sequence')
    
    return team_sequences

# -----------------------------------------------------------------------------
# Function: Combine Men’s and Women’s Team Sequences
# -----------------------------------------------------------------------------
def get_combined_team_sequences(men_results_file, men_game_cities_file,
                                women_results_file, women_game_cities_file):
    men_sequences = get_team_sequences(men_results_file, men_game_cities_file)
    women_sequences = get_team_sequences(women_results_file, women_game_cities_file)
    combined_sequences = pd.concat([men_sequences, women_sequences], axis=0).reset_index(drop=True)
    return combined_sequences

# -----------------------------------------------------------------------------
# Helper Function: Retrieve a Team's Sequence from the Combined Data
# -----------------------------------------------------------------------------
def get_team_sequence(team_sequences_df, season, team_id):
    """
    Returns the padded sequence for a given team and season.
    If not found, returns None.
    """
    row = team_sequences_df[(team_sequences_df['Season'] == season) &
                            (team_sequences_df['TeamID'] == team_id)]
    if row.empty:
        return None
    else:
        return row.iloc[0]['Sequence']

# -----------------------------------------------------------------------------
# Function: Predict Single Matchup
# -----------------------------------------------------------------------------
def predict_single_matchup(lower_team_id, higher_team_id, season, team_sequences_df, model):
    """
    Given the lower and higher team IDs (as integers) and a season,
    returns the predicted probability (between 0 and 1) that the team
    with the lower ID wins.
    """
    seq_lower = get_team_sequence(team_sequences_df, season, lower_team_id)
    seq_higher = get_team_sequence(team_sequences_df, season, higher_team_id)
    
    # If either sequence is missing, default to 0.5.
    if seq_lower is None or seq_higher is None:
        return 0.5
    
    # Expand dimensions to match model input shape (1, 10, 6)
    seq_lower = np.expand_dims(seq_lower, axis=0)
    seq_higher = np.expand_dims(seq_higher, axis=0)
    
    # Use the model to predict the win probability.
    pred_prob = model.predict([seq_lower, seq_higher])[0][0]
    return pred_prob

# -----------------------------------------------------------------------------
# Function: Generate Submission File
# -----------------------------------------------------------------------------
def generate_submission(sample_submission_file, men_results_file, men_game_cities_file,
                        women_results_file, women_game_cities_file, model, output_file):
    """
    Reads the sample submission file to determine the required matchups,
    generates predictions for every matchup, and writes the results to a CSV file.
    """
    # Load sample submission file.
    submission = pd.read_csv(sample_submission_file)
    
    # Build the combined team sequences.
    team_sequences_df = get_combined_team_sequences(men_results_file, men_game_cities_file,
                                                    women_results_file, women_game_cities_file)
    
    predictions = []
    # Iterate through each matchup specified in the sample submission.
    for idx, row in submission.iterrows():
        match_id = row['ID']  # Format: SSSS_XXXX_YYYY (e.g., "2025_1101_1102")
        parts = match_id.split('_')
        season = int(parts[0])
        team1 = int(parts[1])
        team2 = int(parts[2])
        
        # Ensure that the lower team ID is first.
        lower_team = min(team1, team2)
        higher_team = max(team1, team2)
        
        pred = predict_single_matchup(lower_team, higher_team, season, team_sequences_df, model)
        predictions.append(pred)
    
    # Update predictions in submission DataFrame.
    submission['Pred'] = predictions
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

# -----------------------------------------------------------------------------
# Example Usage: Generate Submission File
# -----------------------------------------------------------------------------
# File paths for men's and women's data (update these paths as needed).
men_results_file = '/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv'
men_game_cities_file = '/kaggle/input/march-machine-learning-mania-2025/MGameCities.csv'
women_results_file = '/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv'
women_game_cities_file = '/kaggle/input/march-machine-learning-mania-2025/WGameCities.csv'

# Sample submission file provided by the competition.
sample_submission_file = '/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv'
output_submission_file = 'submission1.csv'

# (Assuming your trained model is loaded in the variable `model`.)
# For example, if you saved your model weights:
# model.load_weights('best_model.h5')

# Generate the submission file.
generate_submission(sample_submission_file, men_results_file, men_game_cities_file,
                    women_results_file, women_game_cities_file, model, output_submission_file)


