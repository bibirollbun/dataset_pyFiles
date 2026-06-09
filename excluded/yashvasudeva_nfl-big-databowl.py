import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


path = "/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
# Listing input and output files
input_files = sorted([f for f in os.listdir(path) if f.startswith("input_") and f.endswith(".csv")])
output_files = sorted([f for f in os.listdir(path) if f.startswith("output_") and f.endswith(".csv")])

print(f"Number of input files: {len(input_files)}")
print(f"Number of output files: {len(output_files)}")

# Loading all input dataframes into one
input_dfs = [pd.read_csv(os.path.join(path, f)) for f in input_files]
df_input = pd.concat(input_dfs, ignore_index=True)
print("Combined input shape:", df_input.shape)

# Loading all output dataframes into one
output_dfs = [pd.read_csv(os.path.join(path, f)) for f in output_files]
df_output = pd.concat(output_dfs, ignore_index=True)
print("Combined output shape:", df_output.shape)

print("\n--- Input Data ---")
display(df_input.head())

print("\n--- Output Data ---")
display(df_output.head())



print("\n--- Input Data Info ---")
df_input.info()

print("\n--- Missing Values of Input Data ---")
print(df_input.isnull().sum().sort_values(ascending=False).head(10))




# Numeric summary
print("\n--- Numeric Description (Input) ---")
display(df_input.describe().T)

# Correlation heatmap for numeric columns (if not too many)
numeric_cols = df_input.select_dtypes(include=np.number).columns
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_input[numeric_cols].corr(), cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap - Input Data")
    plt.show()

# A Check for duplicates
print("\nDuplicate Rows in Input:", df_input.duplicated().sum())
print("Duplicate Rows in Output:", df_output.duplicated().sum())



id_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
player_info_cols = ['player_name', 'player_height', 'player_weight', 'player_role']
position_cols = ['x', 'y', 's', 'a', 'dir', 'o']
context_cols = ['play_direction', 'absolute_yardline_number']
target_cols = ['ball_land_x', 'ball_land_y']

print("Input Columns:", df_input.columns.tolist())
print("\nTotal Columns:", len(df_input.columns))


# --- Numerical Feature Distributions ---
num_cols = df_input.select_dtypes(include=np.number).columns
df_input[num_cols].hist(figsize=(15, 12), bins=30)
plt.suptitle("Numerical Feature Distributions", y=1.02)
plt.show()

# --- Player Role Distribution ---
plt.figure(figsize=(8, 4))
df_input['player_role'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Player Role Distribution")
plt.xlabel("Role")
plt.ylabel("Count")
plt.show()

# --- Play Direction Count ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df_input, x='play_direction', palette='coolwarm')
plt.title("Play Direction Distribution")
plt.show()

# --- Height & Weight ---
# Convert player_height like "6-1" to inches
def height_to_inches(h):
    if isinstance(h, str) and '-' in h:
        feet, inches = h.split('-')
        return int(feet)*12 + int(inches)
    return np.nan

df_input['player_height_inches'] = df_input['player_height'].apply(height_to_inches)

plt.figure(figsize=(8, 4))
sns.scatterplot(x='player_height_inches', y='player_weight', data=df_input, alpha=0.3)
plt.title("Player Height vs Weight")
plt.show()

# --- Correlation heatmap (for movement stats) ---
movement_cols = ['s', 'a', 'dir', 'o']
plt.figure(figsize=(6, 5))
sns.heatmap(df_input[movement_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Movement Feature Correlations")
plt.show()

# --- Sample Play Trajectory (Input vs Output) ---
sample_play = df_input['play_id'].iloc[0]
sample_game = df_input['game_id'].iloc[0]

df_play_input = df_input[(df_input['game_id'] == sample_game) & (df_input['play_id'] == sample_play)]
df_play_output = df_output[(df_output['game_id'] == sample_game) & (df_output['play_id'] == sample_play)]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', data=df_play_input, hue='player_role', alpha=0.6)
plt.scatter(df_play_output['x'], df_play_output['y'], color='red', label='Predicted Path', s=50)
plt.title(f"Player Movements (Game {sample_game}, Play {sample_play})")
plt.legend()
plt.show()

# --- Frame-wise motion analysis for a sample player ---
sample_player = df_play_input['nfl_id'].iloc[0]
df_player_frames = df_play_input[df_play_input['nfl_id'] == sample_player]

plt.figure(figsize=(8, 4))
plt.plot(df_player_frames['frame_id'], df_player_frames['s'], label='Speed')
plt.plot(df_player_frames['frame_id'], df_player_frames['a'], label='Acceleration')
plt.xlabel('Frame ID')
plt.ylabel('Value')
plt.title(f"Speed and Acceleration over Frames - Player {sample_player}")
plt.legend()
plt.show()


# ===== 1️Feature Engineering =====

# Convert height if not already done
def height_to_inches(h):
    if isinstance(h, str) and '-' in h:
        f, i = h.split('-')
        return int(f) * 12 + int(i)
    return np.nan

df_input['player_height_inches'] = df_input['player_height'].apply(height_to_inches)

# Encode categorical variables
cat_cols = ['player_role', 'play_direction']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_input[col] = le.fit_transform(df_input[col].astype(str))
    label_encoders[col] = le

# Computing new features
df_input['speed_acc_ratio'] = df_input['s'] / (df_input['a'] + 1e-3)
df_input['dir_cos'] = np.cos(np.deg2rad(df_input['dir']))
df_input['dir_sin'] = np.sin(np.deg2rad(df_input['dir']))
df_input['orientation_cos'] = np.cos(np.deg2rad(df_input['o']))
df_input['orientation_sin'] = np.sin(np.deg2rad(df_input['o']))

# Aggregate per player per play
agg_funcs = {
    's': ['mean', 'max'],
    'a': ['mean', 'max'],
    'speed_acc_ratio': 'mean',
    'player_height_inches': 'first',
    'player_weight': 'first',
    'player_role': 'first',
    'play_direction': 'first',
    'absolute_yardline_number': 'first',
    'dir_cos': 'mean',
    'dir_sin': 'mean',
    'orientation_cos': 'mean',
    'orientation_sin': 'mean',
    'ball_land_x': 'first',
    'ball_land_y': 'first'
}

features = df_input.groupby(['game_id', 'play_id', 'nfl_id']).agg(agg_funcs)
features.columns = ['_'.join(col).strip() for col in features.columns.values]
features = features.reset_index()

print("Feature table shape:", features.shape)
display(features.head())

# ===== 2️Merge with Output Data =====

df_merged = pd.merge(
    features,
    df_output.groupby(['game_id', 'play_id', 'nfl_id'])[['x', 'y']].mean().reset_index(),
    on=['game_id', 'play_id', 'nfl_id'],
    how='inner'
)

print("Merged Data Shape:", df_merged.shape)
display(df_merged.head())

# ===== Train-Test Split =====

X = df_merged.drop(['x', 'y'], axis=1)
y = df_merged[['x', 'y']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Baseline Model =====

model_x = RandomForestRegressor(n_estimators=100, random_state=42)
model_y = RandomForestRegressor(n_estimators=100, random_state=42)

model_x.fit(X_train, y_train['x'])
model_y.fit(X_train, y_train['y'])

# ===== Evaluation =====

y_pred_x = model_x.predict(X_test)
y_pred_y = model_y.predict(X_test)

rmse_x = np.sqrt(mean_squared_error(y_test['x'], y_pred_x))
rmse_y = np.sqrt(mean_squared_error(y_test['y'], y_pred_y))

r2_x = r2_score(y_test['x'], y_pred_x)
r2_y = r2_score(y_test['y'], y_pred_y)

print(f"X Prediction RMSE: {rmse_x:.3f}, R²: {r2_x:.3f}")
print(f"Y Prediction RMSE: {rmse_y:.3f}, R²: {r2_y:.3f}")

# ===== Feature Importance =====

import matplotlib.pyplot as plt
import seaborn as sns

feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance_x': model_x.feature_importances_,
    'importance_y': model_y.feature_importances_
})
feat_imp['avg_importance'] = (feat_imp['importance_x'] + feat_imp['importance_y']) / 2
feat_imp = feat_imp.sort_values(by='avg_importance', ascending=False).head(15)

plt.figure(figsize=(8, 5))
sns.barplot(x='avg_importance', y='feature', data=feat_imp, palette='viridis')
plt.title("Top 15 Feature Importances (Average of X & Y models)")
plt.show()



y_pred_x = model_x.predict(X_test)
y_pred_y = model_y.predict(X_test)

df_results = X_test.copy()
df_results['x_true'] = y_test['x'].values
df_results['y_true'] = y_test['y'].values
df_results['x_pred'] = y_pred_x
df_results['y_pred'] = y_pred_y

# --- Scatterplot of true vs predicted ---
plt.figure(figsize=(6,6))
sns.scatterplot(x='x_true', y='x_pred', data=df_results, alpha=0.4)
plt.plot([df_results['x_true'].min(), df_results['x_true'].max()],
         [df_results['x_true'].min(), df_results['x_true'].max()],
         color='red', linestyle='--')
plt.title("True vs Predicted X Coordinates")
plt.xlabel("True X")
plt.ylabel("Predicted X")
plt.show()

plt.figure(figsize=(6,6))
sns.scatterplot(x='y_true', y='y_pred', data=df_results, alpha=0.4, color='orange')
plt.plot([df_results['y_true'].min(), df_results['y_true'].max()],
         [df_results['y_true'].min(), df_results['y_true'].max()],
         color='red', linestyle='--')
plt.title("True vs Predicted Y Coordinates")
plt.xlabel("True Y")
plt.ylabel("Predicted Y")
plt.show()

# Plot Predicted vs Actual Player Movement for a Random Play
sample = df_results.sample(1, random_state=42)
game_id = sample['game_id'].values[0]
play_id = sample['play_id'].values[0]

df_actual = df_output[(df_output['game_id'] == game_id) & (df_output['play_id'] == play_id)]
df_pred = df_results[(df_results['game_id'] == game_id) & (df_results['play_id'] == play_id)]

plt.figure(figsize=(8,6))
sns.scatterplot(x='x', y='y', data=df_actual, color='blue', alpha=0.4, label='Actual')
sns.scatterplot(x='x_pred', y='y_pred', data=df_pred, color='red', alpha=0.6, label='Predicted')
plt.title(f"Predicted vs Actual Movement (Game {game_id}, Play {play_id})")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()
plt.show()

# Error Distribution Visualization
df_results['error_x'] = abs(df_results['x_true'] - df_results['x_pred'])
df_results['error_y'] = abs(df_results['y_true'] - df_results['y_pred'])
df_results['error_total'] = np.sqrt(df_results['error_x']**2 + df_results['error_y']**2)

plt.figure(figsize=(8,4))
sns.histplot(df_results['error_total'], bins=30, kde=True, color='purple')
plt.title("Total Position Error Distribution")
plt.xlabel("Euclidean Error (yards)")
plt.ylabel("Count")
plt.show()

# Heatmap of Spatial Error on the Field
plt.figure(figsize=(10,6))
sns.kdeplot(
    x=df_results['x_true'], 
    y=df_results['y_true'], 
    weights=df_results['error_total'], 
    cmap="coolwarm", 
    fill=True, 
    thresh=0.05
)
plt.title("Spatial Heatmap of Model Error")
plt.xlabel("X position (yards)")
plt.ylabel("Y position (yards)")
plt.show()


# LSTM sequence model: sequence -> next-frame (x,y)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from math import sqrt
from sklearn.metrics import mean_squared_error

# -------- PARAMETERS --------
SEQ_LEN = 8      # number of past frames used as input
PRED_STEPS = 1   # predict 1 step ahead (you can roll for multi-step)
BATCH_SIZE = 256
EPOCHS = 30
RANDOM_STATE = 42

# -------- PREPARE FRAME-LEVEL TABLE --------
df_frames = df_input.merge(
    df_output.rename(columns={'x':'x_true','y':'y_true'}),
    on=['game_id','play_id','nfl_id','frame_id'],
    how='left'
)

# keep only rows with true targets 
df_frames = df_frames.dropna(subset=['x_true','y_true']).copy()
df_frames = df_frames.sort_values(['game_id','play_id','nfl_id','frame_id'])

# -------- CHOSEN FEATURES (frame-level) --------
frame_feat_cols = ['x', 'y', 's', 'a', 'dir_cos','dir_sin',
                   'orientation_cos','orientation_sin','speed_acc_ratio',
                   'player_role', 'player_height_inches', 'player_weight',
                   'play_direction','absolute_yardline_number']

# ensure these exist
for c in frame_feat_cols:
    if c not in df_frames.columns:
        raise ValueError(f"Missing required column: {c}")

# -------- BUILD SLIDING WINDOWS PER PLAYER-PLAY --------
X_seqs = []
y_next = []
meta = []  # store (game_id, play_id, nfl_id, frame_id_of_target)

group_cols = ['game_id','play_id','nfl_id']
for _, grp in df_frames.groupby(group_cols):
    grp = grp.sort_values('frame_id')
    feats = grp[frame_feat_cols].values
    targets = grp[['x_true','y_true']].values
    frames = grp['frame_id'].values
    if len(feats) < SEQ_LEN + PRED_STEPS:
        continue
    # sliding windows
    for i in range(len(feats) - SEQ_LEN - (PRED_STEPS-1)):
        X_seqs.append(feats[i:i+SEQ_LEN])
        # predict next frame (single step)
        y_next.append(targets[i+SEQ_LEN])  # frame immediately after sequence
        meta.append((grp['game_id'].iloc[0], grp['play_id'].iloc[0], grp['nfl_id'].iloc[0], frames[i+SEQ_LEN]))

X_seqs = np.array(X_seqs)   # shape: (N, SEQ_LEN, n_features)
y_next = np.array(y_next)   # shape: (N, 2)

print("Built sequences:", X_seqs.shape, y_next.shape)

# -------- TRAIN/TEST SPLIT BY PLAY --------
meta_df = pd.DataFrame(meta, columns=['game_id','play_id','nfl_id','target_frame'])
meta_df['play_uid'] = meta_df['game_id'].astype(str) + "_" + meta_df['play_id'].astype(str)
play_uids = meta_df['play_uid'].unique()
train_uids, test_uids = train_test_split(play_uids, test_size=0.2, random_state=RANDOM_STATE)

train_mask = meta_df['play_uid'].isin(train_uids).values
X_train = X_seqs[train_mask]
X_test  = X_seqs[~train_mask]
y_train = y_next[train_mask]
y_test  = y_next[~train_mask]

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes :", X_test.shape, y_test.shape)

# -------- SCALE FEATURES  --------
n_feats = X_train.shape[2]
scaler = StandardScaler()
# fit on training frames flattened
X_train_flat = X_train.reshape(-1, n_feats)
scaler.fit(X_train_flat)
# transform
X_train = scaler.transform(X_train_flat).reshape(-1, SEQ_LEN, n_feats)
X_test = scaler.transform(X_test.reshape(-1, n_feats)).reshape(-1, SEQ_LEN, n_feats)

# We will not scale target coordinates here (predict in original yard units).
# Optional: scale y targets and invert after predictions.

# -------- BUILD LSTM MODEL --------
tf.random.set_seed(RANDOM_STATE)
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, n_feats)),
    layers.Masking(mask_value=0.0),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)   # predict x and y
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# callbacks
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# -------- TRAIN --------
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, rlr],
    verbose=2
)

# -------- EVALUATE --------
y_pred = model.predict(X_test)
rmse_x = sqrt(mean_squared_error(y_test[:,0], y_pred[:,0]))
rmse_y = sqrt(mean_squared_error(y_test[:,1], y_pred[:,1]))
rmse_total = sqrt(mean_squared_error(np.linalg.norm(y_test - y_pred, axis=1), np.zeros_like(y_test[:,0])) )
print(f"LSTM RMSE X: {rmse_x:.3f}, Y: {rmse_y:.3f}")
# Euclidean RMSE (mean euclidean error)
eucl_errors = np.linalg.norm(y_test - y_pred, axis=1)
print("Mean Euclidean Error (yards):", eucl_errors.mean())

# -------- VISUALIZING SOME PREDICTIONS FOR RANDOM SAMPLES --------
import matplotlib.pyplot as plt
idxs = np.random.choice(len(y_test), size=5, replace=False)
for idx in idxs:
    seq = X_test[idx]  # scaled
    # unscale for plotting sequence coords: inverse transform only x,y features positions inside features array
    # find index of x and y in frame_feat_cols:
    ix = frame_feat_cols.index('x'); iy = frame_feat_cols.index('y')
    seq_unscaled = scaler.inverse_transform(seq)
    seq_x = seq_unscaled[:, ix]
    seq_y = seq_unscaled[:, iy]
    true_x, true_y = y_test[idx]
    pred_x, pred_y = y_pred[idx]
    plt.figure(figsize=(6,4))
    plt.plot(seq_x, seq_y, marker='o', label='history (past frames)')
    plt.scatter([true_x],[true_y], color='green', s=60, label='true next')
    plt.scatter([pred_x],[pred_y], color='red', s=60, label='pred next')
    plt.title('Sequence -> next-frame prediction (green=true, red=pred)')
    plt.xlabel('x (yards)')
    plt.ylabel('y (yards)')
    plt.legend()
    plt.show()

# -------- ROLL FOR MULTI-STEP PREDICTIONS --------
def roll_predict(initial_seq_scaled, steps=5):
    """
    initial_seq_scaled: np.array shape (SEQ_LEN, n_feats) already scaled
    returns list of predicted (x,y) for next steps (does not update other non-pos features)
    """
    seq = initial_seq_scaled.copy()
    preds = []
    for _ in range(steps):
        p = model.predict(seq.reshape(1,SEQ_LEN,n_feats))[0]
        preds.append(p)
        # To roll forward, we need to create next input frame features.
        # Simplest approach: shift sequence and append a new frame where x,y are predicted
        # and keep other features equal to last frame (approximation).
        last_frame = scaler.inverse_transform(seq[-1].reshape(1,-1))[0]
        next_frame = last_frame.copy()
        next_frame[ix] = p[0]
        next_frame[iy] = p[1]
        # scale next_frame and append to seq
        next_frame_scaled = scaler.transform(next_frame.reshape(1,-1))[0]
        seq = np.vstack([seq[1:], next_frame_scaled])
    return np.array(preds)

# Example roll: pick one test sample
sample_seq_scaled = X_test[0]
multi_preds = roll_predict(sample_seq_scaled, steps=5)
print("Rolled preds (5 steps):", multi_preds)



# Load supplementary dataset
supp_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')

# Basic info
print("Shape:", supp_df.shape)
print("\nColumns:\n", supp_df.columns.tolist())
print("\nMissing values per column:\n", supp_df.isnull().sum().sort_values(ascending=False).head(10))

# Sample records
display(supp_df.head(3))

# Data types summary
print("\nData types:\n", supp_df.dtypes)

# Quick statistics for numerical columns
display(supp_df.describe().T)

# Check categorical column distributions
categorical_cols = supp_df.select_dtypes(include='object').columns
for col in categorical_cols[:5]:
    print(f"\n{col} value counts:\n", supp_df[col].value_counts().head())



# --- Clean and Prepare Supplementary Data ---

# Standardize column names
supp_df.columns = (
    supp_df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

# Convert date and time columns
supp_df['game_date'] = pd.to_datetime(supp_df['game_date'], errors='coerce')

# Convert time column safely (some may not be valid times)
supp_df['game_time_eastern'] = pd.to_datetime(
    supp_df['game_time_eastern'], errors='coerce', format='%H:%M:%S'
).dt.time

# Check for duplicates
duplicates = supp_df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Safely convert categorical/object columns to lowercase
cat_cols = supp_df.select_dtypes(include='object').columns
for col in cat_cols:
    supp_df[col] = supp_df[col].astype(str).str.lower().str.strip()

# Handle missing values in numeric columns
num_cols = supp_df.select_dtypes(include=np.number).columns
supp_df[num_cols] = supp_df[num_cols].fillna(0)

# Basic validation summary
print("\nSupplementary Data Cleaned Successfully!")
print(f"Rows: {supp_df.shape[0]}, Columns: {supp_df.shape[1]}")
print(f"Categorical Columns: {len(cat_cols)} | Numerical Columns: {len(num_cols)}")



# Top teams by number of plays
top_teams = supp_df['home_team_abbr'].value_counts().head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=top_teams.index, y=top_teams.values)
plt.title('Top 10 Teams by Number of Plays')
plt.show()

# Distribution of yards gained
plt.figure(figsize=(8,4))
sns.histplot(supp_df['yards_gained'], bins=30, kde=True)
plt.title('Distribution of Yards Gained')
plt.show()

# Correlation heatmap of key numeric features
plt.figure(figsize=(10,6))
sns.heatmap(supp_df[['yards_gained', 'expected_points', 'expected_points_added']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Key Game Metrics')
plt.show()



# --- Team and Play Insights ---

# Average yards gained per team
team_yards = supp_df.groupby('home_team_abbr')['yards_gained'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=team_yards.index, y=team_yards.values, palette='crest')
plt.title('Average Yards Gained per Home Team')
plt.ylabel('Average Yards')
plt.xlabel('Team')
plt.xticks(rotation=45)
plt.show()

# Distribution of expected points added (EPA)
plt.figure(figsize=(8,4))
sns.histplot(supp_df['expected_points_added'], bins=40, kde=True, color='teal')
plt.title('Distribution of Expected Points Added (EPA)')
plt.show()

# Boxplot of yards gained per quarter
plt.figure(figsize=(8,4))
sns.boxplot(data=supp_df, x='quarter', y='yards_gained', palette='cool')
plt.title('Yards Gained per Quarter')
plt.show()

# Relationship between expected points and yards gained
plt.figure(figsize=(7,5))
sns.scatterplot(data=supp_df.sample(2000, random_state=42), x='expected_points', y='yards_gained', alpha=0.6)
plt.title('Expected Points vs. Yards Gained')
plt.show()

# Check correlation across all numeric columns
corr = supp_df.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Overall Feature Correlation')
plt.show()


# --- Feature Preparation ---

# Fill NaNs safely
supp_df = supp_df.fillna({
    'yards_gained': 0,
    'expected_points': 0,
    'expected_points_added': 0
})

# Encode categorical columns (for modeling later)
cat_cols = ['home_team_abbr', 'visitor_team_abbr', 'team_coverage_type']
for col in cat_cols:
    if col in supp_df.columns:
        supp_df[col] = supp_df[col].astype('category').cat.codes

# Select key features for predictive modeling (can modify later)
feature_cols = [
    'quarter', 'down', 'yards_gained', 'expected_points', 'expected_points_added',
    'pre_penalty_yards_gained', 'penalty_yards', 'home_team_abbr', 'visitor_team_abbr'
]

# Drop missing columns that don’t exist
feature_cols = [col for col in feature_cols if col in supp_df.columns]
X = supp_df[feature_cols]

print(f"Feature matrix prepared with shape: {X.shape}")
print(f"Included features: {feature_cols}")



# Target variable
target_col = 'yards_gained'

# Ensure the target exists
if target_col not in supp_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset")

# Drop rows with missing target
model_df = supp_df.dropna(subset=[target_col])

# Select numeric columns for modeling
X = model_df.select_dtypes(include=[np.number]).drop(columns=[target_col])
y = model_df[target_col]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# --- Baseline Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# --- Training and Evaluation ---
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\nModel Performance Summary:")
print(results_df)

# --- Plot comparison ---
plt.figure(figsize=(8,4))
sns.barplot(data=results_df, x="Model", y="RMSE", palette="viridis")
plt.title("Model Comparison (Lower RMSE is Better)")
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4)
    })
    print(f"\n{name}")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))



results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\nModel Performance Comparison:")
print(results_df)



best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")

print(f"Best model saved as 'best_model.pkl' — {best_model_name}")








