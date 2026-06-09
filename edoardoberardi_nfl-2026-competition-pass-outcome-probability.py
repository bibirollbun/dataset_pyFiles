import random
import os
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.lines import Line2D  # Imported for Custom Legend
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML, YouTubeVideo

output_files = []


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'output_' in filename:
            output_files.append(os.path.join(dirname, filename))


# Import of supplementary data 
df_supp = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv')
df_supp['play_unique_id'] = df_supp['game_id'].astype(str) + '_' + df_supp['play_id'].astype(str)

# Reduce the number of useful columns to the only two that are necessary for
# the calculation of the Pass Outcome Probability (POP)
df_supp_compact = df_supp[['play_unique_id', 'pass_result']]


# Create df combining the data coming from week 1 to week 17
# This process is done for the output (Post-thrown information) as well as the 
# input df (Pre-thrown information containing players useful information)
df = pd.DataFrame()
file_path = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/'

for i in range(1, len(output_files)):
    if i < 10:
        file_name = 'output_2023_w0' + str(i) + '.csv'
    else:
        file_name = 'output_2023_w' + str(i) + '.csv'

    print(file_name)
    
    df_temp = pd.read_csv(file_path + file_name)
    df_temp['play_unique_id'] = df_temp['game_id'].astype(str) + '_' + df_temp['play_id'].astype(str)
    df_temp['compact_id'] = df_temp['game_id'].astype(str) + '_' + df_temp['play_id'].astype(str) + '_' + df_temp['nfl_id'].astype(str) + '_' + df_temp['frame_id'].astype(str)

    df = pd.concat([df, df_temp], ignore_index=True)

df_input = pd.DataFrame()
for i in range(1, len(output_files)):
    if i < 10:
        file_name = 'input_2023_w0' + str(i) + '.csv'
    else:
        file_name = 'input_2023_w' + str(i) + '.csv'

    print(file_name)
    
    df_temp = pd.read_csv(file_path + file_name)
    df_temp['play_unique_id'] = df_temp['game_id'].astype(str) + '_' + df_temp['play_id'].astype(str)
    df_temp['compact_id'] = df_temp['game_id'].astype(str) + '_' + df_temp['play_id'].astype(str) + '_' + df_temp['nfl_id'].astype(str) + '_' + df_temp['frame_id'].astype(str)
    
    df_input = pd.concat([df_input, df_temp], ignore_index=True)


# Creating the same columns for the week 18 which will not be used for training or testing but only for the final visualizations

df_input_visualization = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/input_2023_w18.csv')

df_input_visualization['play_unique_id'] = df_input_visualization['game_id'].astype(str) + '_' + df_input_visualization['play_id'].astype(str)
df_input_visualization['compact_id'] = df_input_visualization['game_id'].astype(str) + '_' + df_input_visualization['play_id'].astype(str) + '_' + df_input_visualization['nfl_id'].astype(str) + '_' + df_input_visualization['frame_id'].astype(str)


df_output_visualization = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train/output_2023_w18.csv')

df_output_visualization['play_unique_id'] = df_output_visualization['game_id'].astype(str) + '_' + df_output_visualization['play_id'].astype(str)
df_output_visualization['compact_id'] = df_output_visualization['game_id'].astype(str) + '_' + df_output_visualization['play_id'].astype(str) + '_' + df_output_visualization['nfl_id'].astype(str) + '_' + df_output_visualization['frame_id'].astype(str)


# Join supplementary useful information with the Post-thrown df for the training as well as the visualization df
df = df.join(df_supp_compact.set_index('play_unique_id'), on = 'play_unique_id', how='left')
df_output_visualization = df_output_visualization.join(df_supp_compact.set_index('play_unique_id'), on = 'play_unique_id', how='left')

# Removal of usell columns
useful_columns = ['compact_id', 'player_side', 'player_role', 'num_frames_output', 'ball_land_x', 'ball_land_y']
df_input_useful_col = df_input[useful_columns]
df_input_useful_col_visualization = df_input_visualization[useful_columns]

# Get final dfs combining only necessary columns from input, output and supplementary data
df_pre_computed = df.join(df_input_useful_col.set_index('compact_id'), on= 'compact_id',how='left')
df_pre_computed_visualization = df_output_visualization.join(df_input_useful_col_visualization.set_index('compact_id'), on= 'compact_id',how='left')


def compute_nfs_style_kinematics(df: pd.DataFrame, fps: int=10.0, smooth_window: int=3) -> pd.DataFrame:
    """
    Reconstruct NFL-style tracking fields after the throw:
        speed (s), acceleration (a), orientation (o), direction (dir), Velocity(v),
        vx, vy, ax, ay, speed_mph.
    
    Enhance df with derived metrics from x and y coordinates containe in the output df. Reproduce the 
    information that are provided in the input files (Pre-throw)
    Smooth_window is used to take into consideration the movement of the last n frames to calcualte 
    the new features by using a rolling mean
    """
    df = df.reset_index(drop=True)

    dt = 1.0 / fps

    df = df.sort_values(['game_id','play_id','nfl_id','frame_id'])

    # Ensure proper order since the window should roll on consecutive frames
    df = df.sort_values(['game_id','play_id','nfl_id','frame_id']).copy()
    gcols = ['game_id','play_id','nfl_id']

    # Smooth positions 
    if smooth_window > 1:
        df['x_s'] = df.groupby(gcols)['x'].transform(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
        )
        df['y_s'] = df.groupby(gcols)['y'].transform(
            lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
        )
        x_col, y_col = 'x_s', 'y_s'
    else:
        x_col, y_col = 'x', 'y'

    # Velocity via central differences 
    x_fwd = df.groupby(gcols)[x_col].shift(-1)
    x_bwd = df.groupby(gcols)[x_col].shift( 1)
    y_fwd = df.groupby(gcols)[y_col].shift(-1)
    y_bwd = df.groupby(gcols)[y_col].shift( 1)

    vx = (x_fwd - x_bwd) / (2*dt)
    vy = (y_fwd - y_bwd) / (2*dt)

    # Edges
    vx.loc[x_bwd.isna()] = (x_fwd - df[x_col]) / dt
    vx.loc[x_fwd.isna()] = (df[x_col] - x_bwd) / dt

    vy.loc[y_bwd.isna()] = (y_fwd - df[y_col]) / dt
    vy.loc[y_fwd.isna()] = (df[y_col] - y_bwd) / dt
    
    # Velocity 
    df['vx'], df['vy'] = vx, vy

    # Speed (clipped at 13 yards per seconds)
    df['s'] = np.hypot(df['vx'], df['vy'])
    df['s'] = df['s'].clip(0, 13)  # ~13 yds/s = ~26 mph

    # Direction of motion in degrees
    df['dir'] = (np.degrees(np.arctan2(df['vy'], df['vx'])) % 360)

    # Orientation
    df['o'] = df['dir']

    # Acceleration via central differences on velocity
    vx_fwd = df.groupby(gcols)['vx'].shift(-1)
    vx_bwd = df.groupby(gcols)['vx'].shift( 1)
    vy_fwd = df.groupby(gcols)['vy'].shift(-1)
    vy_bwd = df.groupby(gcols)['vy'].shift( 1)

    ax = (vx_fwd - vx_bwd) / (2*dt)
    ay = (vy_fwd - vy_bwd) / (2*dt)

    # Edge fix
    ax.loc[vx_bwd.isna()] = (vx_fwd - df['vx']) / dt
    ax.loc[vx_fwd.isna()] = (df['vx'] - vx_bwd) / dt
    ay.loc[vy_bwd.isna()] = (vy_fwd - df['vy']) / dt
    ay.loc[vy_fwd.isna()] = (df['vy'] - vy_bwd) / dt

    df['ax'], df['ay'] = ax, ay

    # Acceleration magnitude
    df['a'] = np.hypot(df['ax'], df['ay'])

    # Clip unrealistic bursts (>10 yds/s²)
    df['a'] = df['a'].clip(0, 12)

    # MPH field for convenience 
    df['speed_mph'] = df['s'] * 2.04545

    return df


# Computing features that exists in the pre-throw df but not in the post-throw df

df_pre_computed = df_pre_computed.reset_index(drop=True)
out = compute_nfs_style_kinematics(df_pre_computed, fps=10.0, smooth_window=5)

df_pre_computed_visualization = df_pre_computed_visualization.reset_index(drop=True)
out_visualization= compute_nfs_style_kinematics(df_pre_computed_visualization, fps=10.0, smooth_window=5)


def compute_ball_tracking_metrics(df: pd.DataFrame, eps:float=1e-6, use_accel_arrival:bool=True) -> pd.DataFrame:
    """
    Expects df with columns:
      ['game_id','play_id','nfl_id','frame_id',
       'x','y','vx','vy','ax','ay','ball_land_x','ball_land_y']
    Returns df with added columns:
      r_x, r_y, dist_to_ball,
      unit_rx, unit_ry,
      pursuit_angle_deg,
      v_radial, a_radial, a_lateral,
      t_arr_const, t_arr_quad (if use_accel_arrival)
    """

    # vector to ball
    df['r_x'] = df['ball_land_x'] - df['x']
    df['r_y'] = df['ball_land_y'] - df['y']
    df['dist_to_ball'] = np.hypot(df['r_x'], df['r_y'])

    # unit vector toward ball (eps used for stabilitz in case dist_to_ball is 0)
    df['unit_rx'] = df['r_x'] / (df['dist_to_ball'] + eps)
    df['unit_ry'] = df['r_y'] / (df['dist_to_ball'] + eps)

    # velocity/acceleration magnitude
    df['speed'] = np.hypot(df['vx'], df['vy'])
    df['accel_mag'] = np.hypot(df['ax'], df['ay'])

    # radial speed and radial accel (dot products)
    df['v_radial'] = df['vx'] * df['unit_rx'] + df['vy'] * df['unit_ry']
    df['a_radial'] = df['ax'] * df['unit_rx'] + df['ay'] * df['unit_ry']

    # lateral acceleration magnitude
    df['a_lateral'] = np.sqrt(np.clip(df['accel_mag']**2 - df['a_radial']**2, 0.0, None))

    # pursuit angle (deg): angle between v vector and r vector
    # compute cos_theta safely
    cos_theta = (df['v_radial']) / (df['speed'] + eps)
    cos_theta = cos_theta.clip(-1.0, 1.0)
    df['pursuit_angle_deg'] = np.degrees(np.arccos(cos_theta))

    # signed pursuit angle (optional): use cross product sign
    # cross = v_x * r_y - v_y * r_x
    cross = df['vx'] * df['r_y'] - df['vy'] * df['r_x']
    # df['pursuit_angle_signed_deg'] = df['pursuit_angle_deg'] * np.sign(cross)

    # Time-to-Arrival: constant radial speed --> key metrics for the final calculation of POP
    # if v_radial <= 0 (not moving toward target), otherwise the player is movig toward the target
    df['t_arr_const'] = np.where(df['v_radial'] > eps, df['dist_to_ball'] / df['v_radial'], np.inf)

    # Time-to-Arrival using quadratic (0.5*a t^2 + v t - dist = 0) --> Newton’s Equation of Motion
    if use_accel_arrival:
        a = 0.5 * df['a_radial'] # Teh quadratic term (acceleration)
        b = df['v_radial'] # The linear term (acceleration)
        c = -df['dist_to_ball']

        # discriminant b^2 - 4ac
        disc = b**2 - 4.0 * a * c

        # default to inf
        df['t_arr_quad'] = np.inf

        # valid where disc >=0 and (a != 0 or b>0)
        valid = disc >= 0
        sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))

        # two roots: (-b +/- sqrt_disc) / (2a)
        # choose positive root
        # handle a == 0 (linear) separately, a can be negative if the player is accelerating away 
        linear_mask = np.isclose(a, 0.0)
        # linear case t = -c / b  (but b==v_radial)
        lin_valid = linear_mask & (df['v_radial'] > eps)
        df.loc[lin_valid, 't_arr_quad'] = df.loc[lin_valid, 'dist_to_ball'] / df.loc[lin_valid, 'v_radial']

        # quadratic case where a != 0
        quad_mask = (~linear_mask) & valid
        if quad_mask.any():
            a_q = a[quad_mask]
            b_q = b[quad_mask]
            sd = sqrt_disc[quad_mask]
            # compute both roots
            t1 = (-b_q + sd) / (2.0 * a_q)
            t2 = (-b_q - sd) / (2.0 * a_q)
            # pick smalllest positive root
            t_candidates = np.vstack([t1, t2])
            # replace negative with inf then min along axis
            t_candidates[t_candidates <= eps] = np.inf
            tpos = np.min(t_candidates, axis=0)
            df.loc[quad_mask, 't_arr_quad'] = tpos

        # fallback: where t_arr_quad is inf, use t_arr_const
        df['t_arrival'] = np.where(np.isfinite(df['t_arr_quad']), df['t_arr_quad'], df['t_arr_const'])
    else:
        df['t_arrival'] = df['t_arr_const']

    # ensure numerical robustness in the df
    df['pursuit_angle_deg'] = df['pursuit_angle_deg'].fillna(0.0)
    df['v_radial'] = df['v_radial'].fillna(0.0)
    df['a_radial'] = df['a_radial'].fillna(0.0)
    df['a_lateral'] = df['a_lateral'].fillna(0.0)
    df['t_arrival'] = df['t_arrival'].replace([np.inf, np.nan], np.inf)

    return df


def add_time_comparison_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculating time and speed differences between players and frames
    """
    # Determine the "Final Frame" (when the ball lands) for each play
    # It assumes the max frame_id in the data is the arrival momen, since the post-thrown df ends when the ball is caught/lands.
    df['arrival_frame'] = df.groupby(['game_id', 'play_id'])['frame_id'].transform('max')

    # Calculate Time Remaining (in seconds) --> (Final Frame - Current Frame) * 0.1s
    df['time_remaining'] = (df['arrival_frame'] - df['frame_id']) * 0.1

    # Calculate the Critical Delta
    df['time_delta'] = df['t_arrival'] - df['time_remaining']

    # Handle "Infinite" Time-to-Arrivals
    # If a player is moving away from the ball, t_arrival is inf.
    # The value is "rounded" to a large number (e.g., 99s) to make it ML computable.
    df['time_delta'] = df['time_delta'].replace([np.inf, -np.inf], 10.0) # Player is very late

    return df


# Compute Time-to-Arrival and comparison metrics for the training df and the visualization df

computed_metrics = compute_ball_tracking_metrics(out)
computed_metrics_full = add_time_comparison_metrics(computed_metrics)

computed_metrics_visualization = compute_ball_tracking_metrics(out_visualization)
computed_metrics_full_visualization = add_time_comparison_metrics(computed_metrics_visualization)


# Visualization Game: 2024010600 Play: 178 --> PIT-BAL
# Visualization Game: 2024010713 Play: 459 --> BUF-MIA
# Visualization Game: 2024010706 Play: 588 --> CHI-GB

df_single_play_visualization = computed_metrics_full_visualization[(computed_metrics_full_visualization['game_id'] ==  2024010713) & (computed_metrics_full_visualization['play_id'] == 459)]


def create_enhanced_training_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced version with temporal features and defender selection based on distance at each frame.
    Adding first and second defender to the movement of the target receiver and add the defender_count 
    which can influence on the output of the pass
    """
    identifier_col = ['game_id', 'play_id', 'frame_id']
    
    # Isolate targets and defenders
    targets = df[df['player_role'] == 'Targeted Receiver'].copy()
    defenders = df[df['player_side'] == 'Defense'].copy()
    
    # Sort defenders by time_delta (closest Time-to-Arrival)
    defenders = defenders.sort_values(['game_id', 'play_id', 'frame_id', 'time_delta'])

    # Get primary defender (closest) and count total defenders
    primary_defender = defenders.groupby(identifier_col).first().reset_index()
    def_counts = defenders.groupby(identifier_col).size().reset_index(name='defender_count')

    
    # Get second closest defender if exists (important for coverage)
    secondary_defender = defenders.groupby(identifier_col).nth(1).reset_index()
    secondary_defender = secondary_defender[['game_id', 'play_id', 'frame_id', 'time_delta', 'dist_to_ball']].rename(columns={
        'time_delta': 'def2_time_delta',
        'dist_to_ball': 'def2_dist_ball'
    })
    
    # Rename target columns
    target_cols = ['game_id', 'play_id', 'frame_id', 'time_delta', 'dist_to_ball', 'speed', 
                   'pass_result', 'time_remaining', 'v_radial', 'a_radial', 'pursuit_angle_deg']
    targets = targets[target_cols].rename(columns={
        'time_delta': 'target_time_delta',
        'dist_to_ball': 'target_dist_ball',
        'speed': 'target_speed',
        'v_radial': 'target_v_radial',
        'a_radial': 'target_a_radial',
        'pursuit_angle_deg': 'target_pursuit_angle'
    })
    
    # Rename primary defender columns
    def_cols = ['game_id', 'play_id', 'frame_id', 'time_delta', 'dist_to_ball', 
                'speed', 'v_radial', 'a_radial', 'pursuit_angle_deg']
    primary_defender = primary_defender[def_cols].rename(columns={
        'time_delta': 'def_time_delta',
        'dist_to_ball': 'def_dist_ball',
        'speed': 'def_speed',
        'v_radial': 'def_v_radial',
        'a_radial': 'def_a_radial',
        'pursuit_angle_deg': 'def_pursuit_angle'
    })

    # Merge everything
    train_df = pd.merge(targets, primary_defender, on=identifier_col, how='inner')

    train_df = pd.merge(train_df, def_counts, on=identifier_col, how='left')
    train_df = pd.merge(train_df, secondary_defender, on=identifier_col, how='left')    
    
    # Create duel features (differences between players)
    train_df['time_advantage'] = train_df['def_time_delta'] - train_df['target_time_delta']
    train_df['dist_advantage'] = train_df['def_dist_ball'] - train_df['target_dist_ball']
    train_df['speed_advantage'] = train_df['target_speed'] - train_df['def_speed']
    train_df['v_radial_advantage'] = train_df['target_v_radial'] - train_df['def_v_radial']
    
    # sort based on identifiers
    train_df = train_df.sort_values(identifier_col)
    
    # Change in advantages over time
    for col in ['time_advantage', 'dist_advantage']:
        train_df[f'{col}_change'] = train_df.groupby(['game_id', 'play_id'])[col].diff()
        train_df[f'{col}_change'] = train_df[f'{col}_change'].fillna(0)
    
    # Frame position (normalized)
    train_df['frame_pct'] = train_df['frame_id'] / train_df.groupby(['game_id', 'play_id'])['frame_id'].transform('max')
    
    return train_df


# Get the final dfs before starting training the model
train_dataset = create_enhanced_training_set(computed_metrics_full)
dataset_visualization = create_enhanced_training_set(df_single_play_visualization)


# Prepare data
label_map = {'C': 1, 'I': 0, 'IN': 2}
train_dataset['target'] = train_dataset['pass_result'].map(label_map)

# Enhanced feature set with basic duel feauters, temporal features, individual metrics and def metrics
features = ['time_advantage', 'dist_advantage', 'speed_advantage', 'v_radial_advantage',
    'time_remaining', 'frame_pct','time_advantage_change', 'dist_advantage_change',
    'target_speed', 'def_speed', 'target_v_radial', 'def_v_radial', 'target_pursuit_angle', 'def_pursuit_angle',
    'defender_count', 'def2_time_delta', 'def2_dist_ball']

# Handle missing values
for col in features:
    if col in train_dataset.columns:
        train_dataset[col] = train_dataset[col].fillna(10.0)

X = train_dataset[features]
y = train_dataset['target']

# Create groups: each play is a group
train_dataset['group_id'] = train_dataset['game_id'].astype(str) + '_' + train_dataset['play_id'].astype(str)
groups = train_dataset['group_id']

# GroupKFold --> ensure that a play cannot have some frame that are used to train and some other used for validation
# No data leakage and preservation of "sequential information"
gkf = GroupKFold(n_splits=5)

# results of each fold
fold_scores = []
oof_predictions = np.zeros((len(X), 3))  # Out-of-fold (validation) predictions for calibration
models = []

print("Starting GroupKFold Cross-Validation...")
print("_______________________________________")

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"\nFold {fold}/5")
    print("_______________________________________")

    # Split test and validation for each fold 
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Check class distribution
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Train plays: {groups.iloc[train_idx].nunique()}, Val plays: {groups.iloc[val_idx].nunique()}")
    print(f"Val class distribution:\n{y_val.value_counts(normalize=True)}")
    
    # Train model
    model = xgb.XGBClassifier(
        objective='multi:softprob', # Multi class problem (3 --> Complete, Incomplete, Intercept)
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        tree_method='hist'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )


    # Predict on validation fold
    val_probs = model.predict_proba(X_val)
    oof_predictions[val_idx] = val_probs
    
    # Evaluate fold
    fold_logloss = log_loss(y_val, val_probs)
    fold_auc = roc_auc_score(y_val, val_probs, multi_class='ovr')
    
    print(f"Fold {fold} Log Loss: {fold_logloss:.4f}")
    print(f"Fold {fold} AUC: {fold_auc:.4f}")
    
    fold_scores.append({
        'fold': fold,
        'logloss': fold_logloss,
        'auc': fold_auc,
        'best_iteration': model.best_iteration
    })
    
    models.append(model)

print("***************************************")
print("Cross-Validation Results:")
print("***************************************")

# Overall out-of-fold scores
overall_logloss = log_loss(y, oof_predictions)
overall_auc = roc_auc_score(y, oof_predictions, multi_class='ovr')

print(f"\nOverall OOF Log Loss: {overall_logloss:.4f}")
print(f"Overall OOF AUC: {overall_auc:.4f}")

# Per-fold statistics
fold_df = pd.DataFrame(fold_scores)
print(f"\nPer-Fold Statistics:")
print(fold_df)
print(f"\nMean Log Loss: {fold_df['logloss'].mean():.4f} (+/- {fold_df['logloss'].std():.4f})")
print(f"Mean AUC: {fold_df['auc'].mean():.4f} (+/- {fold_df['auc'].std():.4f})")

# Check prediction distribution vs actual
print("_______________________________________")
print("Prediction Calibration Check:")
print("_______________________________________")
print("Actual class distribution:")
print(y.value_counts(normalize=True))
print("\nPredicted class distribution:")
print(pd.DataFrame(oof_predictions, columns=['I', 'C', 'IN']).mean())


# Select the best model from the 5 folds
best_model = models[0]

# Prepare data
label_map = {'C': 1, 'I': 0, 'IN': 2}
dataset_visualization['target'] = dataset_visualization['pass_result'].map(label_map)

oof_predictions_visualization = np.zeros((len(dataset_visualization), 3))

# Enhanced feature set with basic duel feauters, temporal features, individual metrics and def metrics
features = ['time_advantage', 'dist_advantage', 'speed_advantage', 'v_radial_advantage',
    'time_remaining', 'frame_pct','time_advantage_change', 'dist_advantage_change',
    'target_speed', 'def_speed', 'target_v_radial', 'def_v_radial', 'target_pursuit_angle', 'def_pursuit_angle',
    'defender_count', 'def2_time_delta', 'def2_dist_ball']

# Handle missing values
for col in features:
    if col in dataset_visualization.columns:
        dataset_visualization[col] = dataset_visualization[col].fillna(10.0)

X = dataset_visualization[features]
y = dataset_visualization['target']

# Get the Pass Outcome Probabilities for the selected play
best_model.predict(X)
val_probs_visualization = model.predict_proba(X)

oof_predictions_visualization[dataset_visualization.index] = val_probs_visualization


IMAGE_PATH = '/kaggle/input/nfl-field/NFL_field.png' 

# Attach Predictions
if hasattr(oof_predictions_visualization, 'values'):
    preds = oof_predictions_visualization.values
else:
    preds = oof_predictions_visualization

# Conver array POP into df
dataset_visualization['prob_incomplete'] = preds[:, 0]
dataset_visualization['prob_catch'] = preds[:, 1]
dataset_visualization['prob_int'] = preds[:, 2]

# Ensuring game and play ids
sample_play = dataset_visualization.iloc[0]
game_id_vis = sample_play['game_id']
play_id_vis = sample_play['play_id']
print(f"Visualizing Game: {game_id_vis} Play: {play_id_vis}")

# Get Tracking Data
play_tracking = computed_metrics_full_visualization[
    (computed_metrics_full_visualization['game_id'] == game_id_vis) &
    (computed_metrics_full_visualization['play_id'] == play_id_vis)
].copy()

# D. Get Probability Data
play_probs = dataset_visualization[
    (dataset_visualization['game_id'] == game_id_vis) &
    (dataset_visualization['play_id'] == play_id_vis)
].sort_values('frame_id').set_index('frame_id')

# Get Landing Spot
land_x = play_tracking['ball_land_x'].iloc[0]
land_y = play_tracking['ball_land_y'].iloc[0]

# Get Passer Info
right_play_passer = df_input_visualization[(df_input_visualization['player_role'] == 'Passer') & (df_input_visualization['game_id'] == game_id_vis) & (df_input_visualization['play_id'] == play_id_vis)]

# Extract Coordinates from the First Frame
passer_data = right_play_passer.sort_values('frame_id', ascending=False).iloc[0]

if not passer_data.empty:
    # We take the very first frame of the passer to represent the throw start
    qb_start_row = passer_data
    qb_x = qb_start_row['x']
    qb_y = qb_start_row['y']
else:
    # Final fallback if no QB/Passer found
    qb_x, qb_y = np.nan, np.nan
    print("Warning: No Passer or QB found in tracking data.")
    

# VISUALIZATION SETUP

def create_field_with_image(img_path):
    fig, ax = plt.subplots(1, figsize=(12, 6.33))
    
    try:
        img = mpimg.imread(img_path)
        ax.imshow(img, extent=[0, 120, 0, 53.3], zorder=0)
    except FileNotFoundError:
        print("Image not found. Using default green field.")
        ax.set_facecolor('darkgreen')
        for x in range(10, 111, 10):
            ax.plot([x, x], [0, 53.3], color='white', linewidth=2, alpha=0.5)

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.axis('off')
    return fig, ax

fig, ax = create_field_with_image(IMAGE_PATH)

# Plot elements
# Trajectory Line (Passer -> Landing Spot)
if pd.notna(qb_x) and pd.notna(land_x):
    ax.plot([qb_x, land_x], [qb_y, land_y], color='white', linestyle=':', linewidth=2, alpha=0.7, zorder=1)

# QB Start Marker
if pd.notna(qb_x):
    ax.scatter(qb_x, qb_y, marker='p', c='white', edgecolors='black', s=150, zorder=6)

# Landing Spot (Yellow X)
if pd.notna(land_x):
    ax.scatter(land_x, land_y, marker='x', c='yellow', s=200, linewidth=3, zorder=2)

# Player Scatters
offense_scat = ax.scatter([], [], c='cyan', s=150, edgecolors='white', linewidth=2, zorder=5)
defense_scat = ax.scatter([], [], c='red', s=150, edgecolors='black', linewidth=1, zorder=4)

# Probability Halo
prob_halo = patches.Circle((land_x if pd.notna(land_x) else 0, land_y if pd.notna(land_y) else 0), 
                           radius=4.0, color='white', alpha=0.0, zorder=3)
ax.add_patch(prob_halo)

# Info Box
info_box = ax.text(60, 48, 'Loading...', ha='center', va='top', fontsize=12,
                   bbox=dict(facecolor='black', alpha=0.8, edgecolor='white'), color='white', zorder=10)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Offense',
           markerfacecolor='cyan', markersize=10, markeredgecolor='white', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', label='Defense',
           markerfacecolor='red', markersize=10, markeredgecolor='black', linestyle='None'),
    Line2D([0], [0], marker='p', color='w', label='Passer Start',
           markerfacecolor='white', markersize=10, markeredgecolor='black', linestyle='None'),
    Line2D([0], [0], marker='x', color='w', label='Landing Spot',
           markeredgecolor='yellow', markersize=10, linestyle='None'),
    Line2D([0], [0], color='white', linestyle=':', label='Trajectory')
]

leg = ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
                frameon=True, fancybox=True, framealpha=0.8, borderpad=1)
leg.get_frame().set_facecolor('black')
leg.get_frame().set_edgecolor('white')
for text in leg.get_texts():
    text.set_color('white')

# Animation Loop
frames = sorted(play_tracking['frame_id'].unique())

def update(frame_idx):
    fid = frames[frame_idx]
    
    # Update Spatial (Players)
    frame_data = play_tracking[play_tracking['frame_id'] == fid]
    off = frame_data[frame_data['player_side'] == 'Offense']
    defs = frame_data[frame_data['player_side'] == 'Defense']
    
    if not off.empty:
        offense_scat.set_offsets(off[['x', 'y']].values)
    if not defs.empty:
        defense_scat.set_offsets(defs[['x', 'y']].values)
        
    # Update Probabilities
    if fid in play_probs.index:
        row = play_probs.loc[fid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
            
        p_catch = row['prob_catch']
        p_int = row['prob_int']
        p_inc = row['prob_incomplete']
        
        # Update Text
        txt = f"Frame: {fid}\nCATCH: {p_catch:.1%} | INT: {p_int:.1%} | INC: {p_inc:.1%}"
        info_box.set_text(txt)
        
        # HALO LOGIC (Locked to Landing Spot)
        # Green if prob of catch is more than 50% Red elif intercept pro is more than 15 %
        # Is white if the two threshold is not reached
        if pd.notna(land_x) and pd.notna(land_y):
            prob_halo.center = (land_x, land_y)
            
            if p_catch > 0.50:
                prob_halo.set_color('lime')
                prob_halo.set_alpha(p_catch * 0.6) 
            elif p_int > 0.15:
                prob_halo.set_color('red')
                alpha_val = p_int * 3.0 
                prob_halo.set_alpha(alpha_val if alpha_val < 0.8 else 0.8)
            else:
                prob_halo.set_color('white')
                prob_halo.set_alpha(0.15)
        else:
            prob_halo.set_alpha(0)

    return offense_scat, defense_scat, prob_halo, info_box

anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())


# Save the visualization as mp4 file

# writer = FFMpegWriter(
#     fps=10,            # matches interval=100 ms
#     metadata=dict(artist='You'),
#     bitrate=1800
# )

# anim.save(
#     "BUF_MIA_459.mp4",
#     writer=writer
# )



# Display of the youtube video containing the real play and the visualization
YouTubeVideo(
    'brTVaBH7vVQ',
    width=1000,
    height=500
)

