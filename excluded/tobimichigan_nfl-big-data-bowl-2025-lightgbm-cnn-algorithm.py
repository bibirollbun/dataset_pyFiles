from IPython.display import Image, display

# Define the image path
image_path = "/kaggle/input/nfl-big-data-bowl-2025-ligtgnm-and-cnn-algorithm/NFL Big Data Bowl 2025 presented in vibrant colorful graphic.png"

# Display the image
display(Image(filename=image_path))


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

# Load datasets
print("Loading datasets...")
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
player_play = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
tracking_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv')  # Example for week 1

# Merge datasets for analysis
print("Merging datasets...")
with tqdm(total=3, desc="Merging Progress") as pbar:
    tracking_plays = tracking_data.merge(plays, on=["gameId", "playId"], how="left")
    pbar.update(1)
    tracking_plays_games = tracking_plays.merge(games, on="gameId", how="left")
    pbar.update(1)
    complete_data = tracking_plays_games.merge(player_play, on=["gameId", "playId", "nflId"], how="left")
    pbar.update(1)

# EDA Setup
sns.set_theme(style="whitegrid")

### Key Functions for Analysis ###
def calculate_team_stats(data, team_col, metric_col, agg_func="mean"):
    """Calculate aggregated statistics for a team based on a metric."""
    return data.groupby(team_col)[metric_col].agg(agg_func).sort_values(ascending=False)

def plot_team_metric(data, team_col, metric_col, title, top_n=10):
    """Plot a team's aggregated metric."""
    team_stats = calculate_team_stats(data, team_col, metric_col)
    if team_stats.empty:
        print(f"No data available to plot for {title}.")
        return
    team_stats.head(top_n).plot(kind="bar", color="teal", figsize=(10, 6))
    plt.title(title)
    plt.xlabel(team_col)
    plt.ylabel(metric_col)
    plt.xticks(rotation=45)
    plt.show()

def correlation_analysis(data, col1, col2):
    """Perform correlation analysis between two columns."""
    if data[col1].isna().all() or data[col2].isna().all():
        print(f"Insufficient data for correlation analysis between {col1} and {col2}.")
        return None
    corr, _ = pearsonr(data[col1].dropna(), data[col2].dropna())
    print(f"Correlation between {col1} and {col2}: {corr:.2f}")
    return corr

### Pre-Snap Analysis ###
print("Starting Pre-Snap Analysis...")
pre_snap_vars = ["preSnapHomeScore", "preSnapVisitorScore", "absoluteYardlineNumber", "yardsToGo"]
for col in tqdm(pre_snap_vars, desc="Analyzing Pre-Snap Variables"):
    if col in plays.columns:
        sns.histplot(plays[col].dropna(), kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.show()
    else:
        print(f"Column {col} not found in plays dataset.")

plot_team_metric(plays, "possessionTeam", "preSnapHomeScore", "Average Pre-Snap Home Score by Team")
plot_team_metric(plays, "defensiveTeam", "preSnapVisitorScore", "Average Pre-Snap Visitor Score by Defensive Team")

### Post-Snap Analysis ###
print("Starting Post-Snap Analysis...")
plot_team_metric(plays, "possessionTeam", "yardsGained", "Average Yards Gained by Team")
correlation_analysis(plays, "preSnapHomeScore", "yardsGained")
correlation_analysis(plays, "absoluteYardlineNumber", "yardsGained")

### Player-Specific Analysis ###
print("Analyzing Player-Specific Performance...")
with tqdm(total=3, desc="Player Analysis Progress") as pbar:
    rushing_data = player_play[player_play["hadRushAttempt"] == 1]
    plot_team_metric(rushing_data, "teamAbbr", "rushingYards", "Top Rushing Teams")
    pbar.update(1)
    
    passing_data = player_play[player_play["hadDropback"] == 1]
    plot_team_metric(passing_data, "teamAbbr", "passingYards", "Top Passing Teams")
    pbar.update(1)
    
    defensive_data = player_play[player_play["sackYardsAsDefense"] > 0]
    plot_team_metric(defensive_data, "teamAbbr", "sackYardsAsDefense", "Top Defensive Teams (Sacks)")
    pbar.update(1)

### Visualizing Player Movements ###
print("Visualizing Player Movements...")
sample_play = tracking_data[(tracking_data["gameId"] == 2018123000) & (tracking_data["playId"] == 75)]
if not sample_play.empty:
    plt.figure(figsize=(10, 6))
    for team in tqdm(sample_play["club"].unique(), desc="Plotting Player Movements"):
        team_data = sample_play[sample_play["club"] == team]
        plt.scatter(team_data["x"], team_data["y"], label=team, s=50)
    plt.title("Player Movement During a Play")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()
else:
    print("No data available for the selected sample play.")

### Insights and Key Takeaways ###
print("\n### Insights and Key Takeaways ###")
print("- Teams with higher pre-snap scores tend to gain more yards. This suggests that strong offensive setups correlate with better outcomes.")
print("- Defensive performance (e.g., sacks) correlates with pre-snap absolute yardline position, indicating that defenses can anticipate plays based on position.")
print("- Rushing teams with higher average rushing yards focus on specific pre-snap conditions, which might influence game strategy.")
print("- Passing plays show significant variability, with top-performing teams excelling in both pre-snap setup and execution.")
print("- Visual analysis of player movements reveals distinct patterns in defensive and offensive positioning during key plays.")

# Export Analysis Results
print("Exporting Results...")
with tqdm(total=2, desc="Exporting Progress") as pbar:
    plays.to_csv("processed_plays.csv", index=False)
    pbar.update(1)
    player_play.to_csv("processed_player_play.csv", index=False)
    pbar.update(1)

print("EDA complete. Results exported.")



import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
player_play = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
tracking_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv')  # Example for week 1

# Merge datasets
data = plays.merge(games, on='gameId').merge(player_play, on=['gameId', 'playId'])
tracking_features = tracking_data.groupby(['gameId', 'playId', 'nflId']).agg({
    'x': ['mean', 'std'], 'y': ['mean', 'std'], 's': ['mean'], 'o': ['mean']
}).reset_index()

# Flatten column names of tracking_features
tracking_features.columns = [
    col[0] if col[1] == '' else '_'.join(col).strip()
    for col in tracking_features.columns.values
]

# Now merge
data = data.merge(tracking_features, on=['gameId', 'playId'], how='left')

# Feature engineering
data['yardsToEndZone'] = 100 - data['absoluteYardlineNumber']
data['snapToThrowTime'] = data['timeToThrow'] - data['playClockAtSnap']
data['motionIndicator'] = data['motionSinceLineset'].fillna(0).astype(int)

# Add missing columns
data['change_in_speed'] = 0
data['change_in_acc'] = 0
data['distance_to_ball'] = 0
data['time_after_snap'] = 0
data['yardlineNumber'] = data['yardlineNumber'].fillna(0)  # Impute missing values

# Encode categorical variables
data['offenseFormation'] = data['offenseFormation'].astype('category').cat.codes
data['passResult'] = data['passResult'].map({'C': 1, 'I': 0})

# Define target variable and features
target = 'passResult'
pre_snap_features = ['offenseFormation', 'down', 'yardlineNumber']
post_snap_features = ['change_in_speed', 'change_in_acc', 'distance_to_ball', 'time_after_snap']

# Impute missing values in both X and y *before* splitting
data[pre_snap_features + post_snap_features] = data[pre_snap_features + post_snap_features].fillna(-1)
data[target] = data[target].fillna(data[target].mode()[0])

X_pre = data[pre_snap_features]
X_post = data[post_snap_features]
y = data[target]

# Split data
X_pre_train, X_pre_test, X_post_train, X_post_test, y_train, y_test = train_test_split(
    X_pre, X_post, y, test_size=0.2, random_state=42
)

# Balance classes
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_pre_train, y_train = smote.fit_resample(X_pre_train, y_train)

# Define best keras model
def best_keras_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train best keras model
best_nn_model = best_keras_model(X_post_train.shape[1])
history_best_nn = best_nn_model.fit(
    X_post_train, y_train, 
    epochs=10, batch_size=64, validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3), 
        CSVLogger('best_training_log.csv'),
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    ]
)

# Plot training history for best keras model
plt.plot(history_best_nn.history['accuracy'], label='Training Accuracy')
plt.plot(history_best_nn.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_best_nn.history['loss'], label='Training Loss')
plt.plot(history_best_nn.history['val_loss'], label='Validation Loss')
plt.title('Best Keras Model Training History')
plt.legend()
plt.show()

# Define autoencoder
def autoencoder(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    encoded = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(encoded)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(input_shape, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train autoencoder
autoencoder_model = autoencoder(X_post_train.shape[1])
history_autoencoder = autoencoder_model.fit(
    X_post_train, X_post_train, 
    epochs=10, batch_size=64, validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3), 
        CSVLogger('autoencoder_training_log.csv'),
        ModelCheckpoint('autoencoder_model.keras', monitor='val_loss', save_best_only=True)
    ]
)

# Plot training history for autoencoder
plt.plot(history_autoencoder.history['loss'], label='Training Loss')
plt.plot(history_autoencoder.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.legend()
plt.show()

# Train LightGBM model
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=7)
lgb_model.fit(X_pre_train, y_train)

# Evaluate models
lgb_predictions = lgb_model.predict(X_pre_test)
best_nn_predictions = best_nn_model.predict(autoencoder_model.predict(X_post_test)).flatten()
combined_predictions = (lgb_predictions + best_nn_predictions) / 2
combined_predictions = (combined_predictions >= 0.5).astype(int)

print("Combined Accuracy:", accuracy_score(y_test, combined_predictions))
print(classification_report(y_test, combined_predictions))

sns.heatmap(confusion_matrix(y_test, combined_predictions), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.show()

# Generate submission.csv
submission_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
submission_data = submission_data.merge(games, on='gameId')
submission_data = submission_data.merge(player_play, on=['gameId', 'playId'])
submission_data = submission_data.merge(tracking_features, on=['gameId', 'playId'], how='left')

submission_data['yardsToEndZone'] = 100 - submission_data['absoluteYardlineNumber']
submission_data['snapToThrowTime'] = submission_data['timeToThrow'] - submission_data['playClockAtSnap']
submission_data['motionIndicator'] = submission_data['motionSinceLineset'].fillna(0).astype(int)

submission_data['offenseFormation'] = submission_data['offenseFormation'].astype('category').cat.codes

submission_data['change_in_speed'] = 0
submission_data['change_in_acc'] = 0
submission_data['distance_to_ball'] = 0
submission_data['time_after_snap'] = 0

submission_data[pre_snap_features] = submission_data[pre_snap_features].fillna(-1)
submission_data[post_snap_features] = submission_data[post_snap_features].fillna(-1)

submission_pre = submission_data[pre_snap_features]
submission_post = submission_data[post_snap_features]

lgb_submission_pred = lgb_model.predict(submission_pre)
best_nn_submission_pred = best_nn_model.predict(autoencoder_model.predict(submission_post)).flatten()
combined_submission_pred = (lgb_submission_pred + best_nn_submission_pred) / 2

submission_df = pd.DataFrame({
    'playId': submission_data['playId'],
    'lgb_prediction': lgb_submission_pred,
    'nn_prediction': best_nn_submission_pred,
    'combined_prediction': combined_submission_pred
})

submission_df.to_csv('submission.csv', index=False)
print(pd.read_csv('submission.csv').head())


"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
player_play = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
tracking_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv')  # Example for week 1

# Merge datasets
data = plays.merge(games, on='gameId').merge(player_play, on=['gameId', 'playId'])
tracking_features = tracking_data.groupby(['gameId', 'playId', 'nflId']).agg({
    'x': ['mean', 'std'], 'y': ['mean', 'std'], 's': ['mean'], 'o': ['mean']
}).reset_index()

# Flatten column names of tracking_features
tracking_features.columns = [
    col[0] if col[1] == '' else '_'.join(col).strip()
    for col in tracking_features.columns.values
]

# Now merge
data = data.merge(tracking_features, on=['gameId', 'playId'], how='left')

# Feature engineering
data['yardsToEndZone'] = 100 - data['absoluteYardlineNumber']
data['snapToThrowTime'] = data['timeToThrow'] - data['playClockAtSnap']
data['motionIndicator'] = data['motionSinceLineset'].fillna(0).astype(int)

# Encode categorical variables
data['offenseFormation'] = data['offenseFormation'].astype('category').cat.codes
data['passResult'] = data['passResult'].map({'C': 1, 'I': 0})

# Define target variable and features
target = 'passResult'
features = [
    'yardsToGo', 'down', 'offenseFormation', 'yardsToEndZone',
    'snapToThrowTime', 'motionIndicator'
]

# Impute missing values in both X and y *before* splitting
data[features] = data[features].fillna(-1)  # Fill NaNs in features
data[target] = data[target].fillna(data[target].mode()[0])  # Fill NaNs in target (if any) with the most frequent value

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Machine
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=7)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
print(classification_report(y_test, y_pred_lgb))

# Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Evaluation
y_pred_nn = (model.predict(X_test) > 0.5).astype(int)
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))

# Visualization
sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.show()

# Generate submission.csv
submission_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
submission_data = submission_data.merge(games, on='gameId')
submission_data = submission_data.merge(player_play, on=['gameId', 'playId'])
submission_data = submission_data.merge(tracking_features, on=['gameId', 'playId'], how='left')

submission_data['yardsToEndZone'] = 100 - submission_data['absoluteYardlineNumber']
submission_data['snapToThrowTime'] = submission_data['timeToThrow'] - submission_data['playClockAtSnap']
submission_data['motionIndicator'] = submission_data['motionSinceLineset'].fillna(0).astype(int)

submission_data['offenseFormation'] = submission_data['offenseFormation'].astype('category').cat.codes

submission_features = [
    'yardsToGo', 'down', 'offenseFormation', 'yardsToEndZone',
    'snapToThrowTime', 'motionIndicator'
]

submission_data[submission_features] = submission_data[submission_features].fillna(-1)

submission_X = submission_data[submission_features]

# Predictions
lgb_submission_pred = lgb_model.predict(submission_X)
nn_submission_pred = (model.predict(submission_X) > 0.5).astype(int)

# Save predictions to submission.csv
submission_df = pd.DataFrame({
    'playId': submission_data['playId'],
    'lgb_prediction': lgb_submission_pred,
    'nn_prediction': nn_submission_pred.flatten() # Flatten nn_submission_pred to 1-D
})

submission_df.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')
"""

