# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.cluster import KMeans


import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

print("All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")


def load_nfl_data():
    """Load and explore the NFL Big Data Bowl 2026 dataset"""
    
    # Load supplementary data
    supplementary_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv'
    supplementary_data = pd.read_csv(supplementary_path)
    
    # Load sample input and output files
    input_files = [
        '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w01.csv',
        '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w02.csv'
    ]
    
    output_files = [
        '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w01.csv',
        '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w02.csv'
    ]
    
    # Load multiple weeks of data
    input_data = pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)
    output_data = pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)
    
    return supplementary_data, input_data, output_data


print("Loading NFL data...")
supplementary_data, input_data, output_data = load_nfl_data()

print("Dataset shapes:")
print(f"Supplementary Data: {supplementary_data.shape}")
print(f"Input Data: {input_data.shape}")
print(f"Output Data: {output_data.shape}")


print("\nSupplementary Data Columns:")
print(supplementary_data.columns.tolist())

print("\nInput Data Columns:")
print(input_data.columns.tolist())

print("\nOutput Data Columns:")
print(output_data.columns.tolist())


print("\nSupplementary Data Info:")
print(supplementary_data.info())


print("Missing values in Supplementary Data:")
missing_supp = supplementary_data.isnull().sum()
print(missing_supp[missing_supp > 0])

print("\nMissing values in Input Data:")
missing_input = input_data.isnull().sum()
print(missing_input[missing_input > 0])

print("\nMissing values in Output Data:")
print(output_data.isnull().sum())


def clean_nfl_data(supplementary_data, input_data, output_data):
    """Clean and preprocess the NFL data"""
    
    # Create copies to avoid modifying original data
    supp_clean = supplementary_data.copy()
    input_clean = input_data.copy()
    output_clean = output_data.copy()
    
    # Handle missing values in supplementary data
    supp_clean['yardline_side'] = supp_clean['yardline_side'].fillna('UNK')
    supp_clean['route_of_targeted_receiver'] = supp_clean['route_of_targeted_receiver'].fillna('UNKNOWN')
    supp_clean['play_action'] = supp_clean['play_action'].fillna('UNK')
    supp_clean['dropback_type'] = supp_clean['dropback_type'].fillna('UNK')
    supp_clean['pass_location_type'] = supp_clean['pass_location_type'].fillna('UNK')
    supp_clean['team_coverage_man_zone'] = supp_clean['team_coverage_man_zone'].fillna('UNK')
    supp_clean['team_coverage_type'] = supp_clean['team_coverage_type'].fillna('UNK')
    supp_clean['penalty_yards'] = supp_clean['penalty_yards'].fillna(0)
    
    # Clean input data - handle numeric conversions
    numeric_columns = ['player_height', 'player_weight', 's', 'a', 'dir', 'o', 'x', 'y']
    for col in numeric_columns:
        if col in input_clean.columns:
            input_clean[f'{col}_clean'] = pd.to_numeric(input_clean[col], errors='coerce')
    
    # Remove rows with critical missing values
    input_clean = input_clean.dropna(subset=['x_clean', 'y_clean', 's_clean'])
    
    return supp_clean, input_clean, output_clean


print("Cleaning data...")
supp_clean, input_clean, output_clean = clean_nfl_data(supplementary_data, input_data, output_data)

print(f"Cleaned Input Data Shape: {input_clean.shape}")


def create_football_field(ax=None):
    """Create a football field for visualization"""
    if ax is None:
        ax = plt.gca()
    
    # Create rectangle representing the field
    rect = Rectangle((0, 0), 120, 53.3, linewidth=2, edgecolor='black', facecolor='green', alpha=0.2)
    ax.add_patch(rect)
    
    # Add yard lines
    for x in range(10, 120, 10):
        ax.axvline(x=x, color='white', alpha=0.5, linestyle='-', linewidth=1)
    
    # Add 50-yard line
    ax.axvline(x=60, color='white', alpha=1, linestyle='-', linewidth=2)
    
    # Add end zones
    endzone1 = Rectangle((0, 0), 10, 53.3, linewidth=2, edgecolor='black', facecolor='darkblue', alpha=0.3)
    endzone2 = Rectangle((110, 0), 10, 53.3, linewidth=2, edgecolor='black', facecolor='darkblue', alpha=0.3)
    ax.add_patch(endzone1)
    ax.add_patch(endzone2)
    
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal')
    ax.set_facecolor('green')
    
    return ax


# 1 Player Position Distribution
plt.figure(figsize=(12, 6))
position_counts = input_clean['player_position'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(position_counts)))

bars = plt.bar(position_counts.index, position_counts.values, color=colors)
plt.title('Player Position Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Position', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# 2 Player Movement Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Speed distribution
axes[0,0].hist(input_clean['s_clean'].dropna(), bins=50, color='skyblue', alpha=0.7)
axes[0,0].set_title('Player Speed Distribution', fontweight='bold')
axes[0,0].set_xlabel('Speed (yards/s)')
axes[0,0].set_ylabel('Frequency')

# Acceleration distribution
axes[0,1].hist(input_clean['a_clean'].dropna(), bins=50, color='lightcoral', alpha=0.7)
axes[0,1].set_title('Player Acceleration Distribution', fontweight='bold')
axes[0,1].set_xlabel('Acceleration (yards/s²)')
axes[0,1].set_ylabel('Frequency')

# Speed by position
speed_by_pos = input_clean.groupby('player_position')['s_clean'].mean().sort_values(ascending=False)
axes[1,0].bar(speed_by_pos.index, speed_by_pos.values, color='lightgreen')
axes[1,0].set_title('Average Speed by Position', fontweight='bold')
axes[1,0].set_xlabel('Position')
axes[1,0].set_ylabel('Average Speed (yards/s)')
axes[1,0].tick_params(axis='x', rotation=45)

# Direction distribution
axes[1,1].hist(input_clean['dir_clean'].dropna(), bins=50, color='gold', alpha=0.7)
axes[1,1].set_title('Player Direction Distribution', fontweight='bold')
axes[1,1].set_xlabel('Direction (degrees)')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# 3 Field Position Heatmap
plt.figure(figsize=(15, 8))
ax = create_football_field()

# Create heatmap of player positions
valid_positions = input_clean.dropna(subset=['x_clean', 'y_clean'])
heatmap = ax.hexbin(valid_positions['x_clean'], valid_positions['y_clean'], 
                    gridsize=50, cmap='YlOrRd', alpha=0.8, mincnt=1)
plt.colorbar(heatmap, ax=ax, label='Player Density')

ax.set_title('Player Position Heatmap on Football Field', fontsize=16, fontweight='bold')
plt.show()


# 4 Game Situation Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Down distribution
down_counts = supp_clean['down'].value_counts().sort_index()
axes[0,0].bar(down_counts.index, down_counts.values, color='lightblue')
axes[0,0].set_title('Play Distribution by Down', fontweight='bold')
axes[0,0].set_xlabel('Down')
axes[0,0].set_ylabel('Number of Plays')

# Yards to go distribution
axes[0,1].hist(supp_clean['yards_to_go'], bins=20, color='lightgreen', alpha=0.7)
axes[0,1].set_title('Yards to Go Distribution', fontweight='bold')
axes[0,1].set_xlabel('Yards to Go')
axes[0,1].set_ylabel('Frequency')

# Quarter distribution
quarter_counts = supp_clean['quarter'].value_counts().sort_index()
axes[1,0].bar(quarter_counts.index, quarter_counts.values, color='lightcoral')
axes[1,0].set_title('Play Distribution by Quarter', fontweight='bold')
axes[1,0].set_xlabel('Quarter')
axes[1,0].set_ylabel('Number of Plays')

# Pass result distribution
if 'pass_result' in supp_clean.columns:
    pass_results = supp_clean['pass_result'].value_counts()
    axes[1,1].bar(pass_results.index, pass_results.values, color='gold')
    axes[1,1].set_title('Pass Result Distribution', fontweight='bold')
    axes[1,1].set_xlabel('Pass Result')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


def create_advanced_features(input_data, supplementary_data):
    """Create advanced features for player movement prediction"""
    
    # Merge input and supplementary data
    merged_data = input_data.merge(
        supplementary_data[['game_id', 'play_id', 'down', 'yards_to_go', 'quarter', 
                          'possession_team', 'defensive_team', 'pass_result']],
        on=['game_id', 'play_id'], 
        how='left'
    )
    
    # Position-based features
    merged_data['is_defensive_player'] = merged_data['player_side'] == 'defense'
    merged_data['is_offensive_player'] = merged_data['player_side'] == 'offense'
    
    # Distance to ball features (if ball position available)
    if all(col in merged_data.columns for col in ['ball_land_x', 'ball_land_y', 'x_clean', 'y_clean']):
        merged_data['distance_to_ball_land'] = np.sqrt(
            (merged_data['x_clean'] - merged_data['ball_land_x'])**2 + 
            (merged_data['y_clean'] - merged_data['ball_land_y'])**2
        )
    
    # Game situation features
    merged_data['is_third_down'] = (merged_data['down'] == 3).astype(int)
    merged_data['is_fourth_down'] = (merged_data['down'] == 4).astype(int)
    merged_data['is_short_yards'] = (merged_data['yards_to_go'] <= 3).astype(int)
    merged_data['is_long_yards'] = (merged_data['yards_to_go'] >= 10).astype(int)
    
    # Player role encoding
    role_encoder = LabelEncoder()
    merged_data['player_role_encoded'] = role_encoder.fit_transform(merged_data['player_role'].fillna('UNK'))
    
    # Position grouping
    position_groups = {
        'QB': 'QB', 'RB': 'RB', 'FB': 'RB',
        'WR': 'WR', 'TE': 'TE',
        'T': 'OL', 'G': 'OL', 'C': 'OL',
        'DE': 'DL', 'DT': 'DL', 'NT': 'DL',
        'LB': 'LB', 'MLB': 'LB', 'OLB': 'LB',
        'CB': 'DB', 'FS': 'DB', 'SS': 'DB', 'S': 'DB'
    }
    
    merged_data['position_group'] = merged_data['player_position'].map(position_groups).fillna('OTHER')
    
    # Encode position groups
    pos_group_encoder = LabelEncoder()
    merged_data['position_group_encoded'] = pos_group_encoder.fit_transform(merged_data['position_group'])
    
    return merged_data

print("Creating advanced features...")
featured_data = create_advanced_features(input_clean, supp_clean)
print(f"Featured data shape: {featured_data.shape}")


def prepare_model_data(featured_data, output_data):
    """Prepare data for machine learning models - CORRECTED VERSION"""
    
    # Merge with output data to get target positions
    model_data = featured_data.merge(
        output_data[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']],
        on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
        how='inner',
        suffixes=('', '_target')
    )
    
    # Feature columns for modeling
    feature_columns = [
        'x_clean', 'y_clean', 's_clean', 'a_clean', 'dir_clean', 'o_clean',
        'player_role_encoded', 'position_group_encoded',
        'is_defensive_player', 'is_offensive_player',
        'down', 'yards_to_go', 'quarter', 'is_third_down', 'is_fourth_down',
        'is_short_yards', 'is_long_yards'
    ]
    
    # Add distance features if available
    if 'distance_to_ball_land' in model_data.columns:
        feature_columns.append('distance_to_ball_land')
    
    # Target columns - ONLY x and y coordinates
    target_columns = ['x', 'y']  # Using the original column names from output data
    
    # Remove rows with missing values in features or targets
    available_features = [col for col in feature_columns if col in model_data.columns]
    model_data_clean = model_data[available_features + target_columns].dropna()
    
    X = model_data_clean[available_features]
    y = model_data_clean[target_columns]
    
    print(f"Target columns: {target_columns}")
    print(f"Target shape: {y.shape}")
    print(f"Target sample:\n{y.head()}")
    
    return X, y, available_features


print("Preparing model data...")
X, y, feature_columns = prepare_model_data(featured_data, output_clean)

print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")
print(f"Feature columns: {feature_columns}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
print(f"Testing set - X: {X_test.shape}, y: {y_test.shape}")


print(f"y_train columns: {y_train.columns.tolist()}")
print(f"y_train sample:\n{y_train.head()}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)


# 6.1 Random Forest Model
print("Training Random Forest Model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


rf_model.fit(X_train_scaled, y_train)


y_pred_rf = rf_model.predict(X_test_scaled)


rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"Random Forest Performance:")
print(f"MAE: {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")


print("Training XGBoost Model...")

# XGBoost for multi-output regression
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)


y_pred_xgb = xgb_model.predict(X_test_scaled)


xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"XGBoost Performance:")
print(f"MAE: {xgb_mae:.4f}")
print(f"RMSE: {xgb_rmse:.4f}")


print("Training Neural Network Model...")

# Get the correct input and output dimensions
input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]  # This should be 2 (x, y)

print(f"Neural Network - Input dimension: {input_dim}, Output dimension: {output_dim}")


nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(output_dim)  # Output: x and y coordinates
])

nn_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


nn_model.summary()


print("Training neural network...")
history = nn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)


y_pred_nn_scaled = nn_model.predict(X_test_scaled)
y_pred_nn = target_scaler.inverse_transform(y_pred_nn_scaled)


nn_mae = mean_absolute_error(y_test, y_pred_nn)
nn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))

print(f"Neural Network Performance:")
print(f"MAE: {nn_mae:.4f}")
print(f"RMSE: {nn_rmse:.4f}")


# Plot training history for neural network
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()


# Compare model performance
models_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
    'MAE': [rf_mae, xgb_mae, nn_mae],
    'RMSE': [rf_rmse, xgb_rmse, nn_rmse]
})

print("Model Performance Comparison:")
print(models_comparison)


# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# MAE comparison
bars1 = axes[0].bar(models_comparison['Model'], models_comparison['MAE'], 
                   color=['skyblue', 'lightgreen', 'lightcoral'])
axes[0].set_title('Model Comparison - Mean Absolute Error (MAE)', fontweight='bold')
axes[0].set_ylabel('MAE (yards)')
axes[0].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

# RMSE comparison
bars2 = axes[1].bar(models_comparison['Model'], models_comparison['RMSE'], 
                   color=['skyblue', 'lightgreen', 'lightcoral'])
axes[1].set_title('Model Comparison - Root Mean Squared Error (RMSE)', fontweight='bold')
axes[1].set_ylabel('RMSE (yards)')
axes[1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importances (Random Forest)', fontweight='bold')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()


def visualize_predictions_comparison(y_true, y_pred, model_name, sample_size=100):
    """Visualize actual vs predicted positions"""
    
    # Sample data for visualization
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_sample = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Actual vs Predicted X coordinates
    axes[0].scatter(y_true_sample.iloc[:, 0], y_pred_sample[:, 0], alpha=0.6, color='blue')
    axes[0].plot([y_true_sample.iloc[:, 0].min(), y_true_sample.iloc[:, 0].max()],
                [y_true_sample.iloc[:, 0].min(), y_true_sample.iloc[:, 0].max()], 
                'r--', linewidth=2)
    axes[0].set_xlabel('Actual X Position')
    axes[0].set_ylabel('Predicted X Position')
    axes[0].set_title(f'{model_name} - X Coordinate Prediction')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted Y coordinates
    axes[1].scatter(y_true_sample.iloc[:, 1], y_pred_sample[:, 1], alpha=0.6, color='green')
    axes[1].plot([y_true_sample.iloc[:, 1].min(), y_true_sample.iloc[:, 1].max()],
                [y_true_sample.iloc[:, 1].min(), y_true_sample.iloc[:, 1].max()], 
                'r--', linewidth=2)
    axes[1].set_xlabel('Actual Y Position')
    axes[1].set_ylabel('Predicted Y Position')
    axes[1].set_title(f'{model_name} - Y Coordinate Prediction')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate error statistics
    x_error = np.abs(y_true_sample.iloc[:, 0] - y_pred_sample[:, 0])
    y_error = np.abs(y_true_sample.iloc[:, 1] - y_pred_sample[:, 1])
    total_error = np.sqrt(x_error**2 + y_error**2)
    
    print(f"{model_name} Error Analysis:")
    print(f"X Error - Mean: {x_error.mean():.3f}, Std: {x_error.std():.3f}")
    print(f"Y Error - Mean: {y_error.mean():.3f}, Std: {y_error.std():.3f}")
    print(f"Total Distance Error - Mean: {total_error.mean():.3f}, Std: {total_error.std():.3f}")


print("Random Forest Prediction Analysis:")
visualize_predictions_comparison(y_test, y_pred_rf, "Random Forest")

print("\nXGBoost Prediction Analysis:")
visualize_predictions_comparison(y_test, y_pred_xgb, "XGBoost")

print("\nNeural Network Prediction Analysis:")
visualize_predictions_comparison(y_test, y_pred_nn, "Neural Network")


def analyze_position_performance(featured_data, y_test, y_pred, test_indices):
    """Analyze model performance by player position"""
    
    # Get the original indices from the test set
    if hasattr(X_test, 'index'):
        test_data = featured_data.loc[X_test.index[test_indices]]
    else:
        # If we don't have indices, use the first n rows
        test_data = featured_data.iloc[test_indices]
    
    # Add predictions to test data
    test_data = test_data.copy()
    test_data['pred_x'] = y_pred[:, 0]
    test_data['pred_y'] = y_pred[:, 1]
    test_data['actual_x'] = y_test.iloc[test_indices, 0].values if hasattr(y_test, 'iloc') else y_test[test_indices, 0]
    test_data['actual_y'] = y_test.iloc[test_indices, 1].values if hasattr(y_test, 'iloc') else y_test[test_indices, 1]
    
    # Calculate errors
    test_data['x_error'] = np.abs(test_data['pred_x'] - test_data['actual_x'])
    test_data['y_error'] = np.abs(test_data['pred_y'] - test_data['actual_y'])
    test_data['distance_error'] = np.sqrt(test_data['x_error']**2 + test_data['y_error']**2)
    
    # Analyze by position
    position_performance = test_data.groupby('player_position').agg({
        'distance_error': ['mean', 'std', 'count'],
        'x_error': 'mean',
        'y_error': 'mean'
    }).round(3)
    
    position_performance.columns = ['_'.join(col).strip() for col in position_performance.columns.values]
    position_performance = position_performance.sort_values('distance_error_mean')
    
    return position_performance, test_data


sample_indices = np.arange(min(12709, len(y_test)))
position_perf, test_data_with_pred = analyze_position_performance(featured_data, y_test, y_pred_rf, sample_indices)

print("Position-Specific Performance (Random Forest):")
print(position_perf)


# Plot position performance
plt.figure(figsize=(12, 6))
positions = position_perf.index
errors = position_perf['distance_error_mean']
std_errors = position_perf['distance_error_std']

plt.bar(positions, errors, yerr=std_errors, capsize=5, color='lightblue', alpha=0.7)
plt.title('Average Prediction Error by Player Position', fontweight='bold')
plt.xlabel('Player Position')
plt.ylabel('Average Distance Error (yards)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels
for i, (pos, error) in enumerate(zip(positions, errors)):
    plt.text(i, error + 0.05, f'{error:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()




