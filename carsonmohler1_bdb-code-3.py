# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from pathlib import Path as FilePath
import time
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
from catboost import CatBoostClassifier

# Custom utilities
import sys
sys.path.append('/kaggle/input/bdb-utils2')
sys.path.append('/kaggle/input/bdb-models')

from utils import (
    load_data, create_foundation, calculate_separation_at_throw,
    extract_route_features, calculate_defensive_metrics,
    calculate_direction_features, detect_wheel_pattern, classify_primary_direction
)
from models import (
    ROUTE_FEATURE_COLS, ROUTE_MODEL_PARAMS, CATCH_MODEL_PARAMS,
    TIMING_WINDOWS, TIMING_FRAMES, CATCH_MODEL_EXCLUDE_COLS
)


# Load data
BASE_PATH = FilePath('/kaggle/input/bdb-data')

input_dfs = []
output_dfs = []

for week in range(1, 19):
    week_str = f"w{week:02d}"
    input_file = BASE_PATH / f"input_2023_{week_str}.csv"
    output_file = BASE_PATH / f"output_2023_{week_str}.csv"
    
    if input_file.exists():
        df = pd.read_csv(input_file)
        df['week'] = week
        input_dfs.append(df)
    
    if output_file.exists():
        df = pd.read_csv(output_file)
        df['week'] = week
        output_dfs.append(df)

input_data = pd.concat(input_dfs, ignore_index=True)
output = pd.concat(output_dfs, ignore_index=True)
supplementary = pd.read_csv(BASE_PATH / 'supplementary_data.csv', low_memory=False)

print(f"Input data: {len(input_data):,} rows")
print(f"Output data: {len(output):,} rows")
print(f"Supplementary: {len(supplementary):,} rows")

del input_dfs, output_dfs


# Valid plays: completions/incompletions, no penalties
valid_plays = supplementary[
    (supplementary['pass_result'].isin(['C', 'I'])) &
    (supplementary['play_nullified_by_penalty'] == 'N')
][['game_id', 'play_id']].drop_duplicates()

print(f"Valid plays: {len(valid_plays)}")

# Final frame per play (throw moment)
final_frames = input_data.groupby(['game_id', 'play_id'])['frame_id'].max().reset_index()
final_frames.columns = ['game_id', 'play_id', 'final_frame_id']

# Player info lookup
player_info = input_data[['game_id', 'play_id', 'nfl_id', 'player_name', 
                          'player_position', 'player_role', 'player_side']].drop_duplicates()

# Final frame positions
final_frame_data = input_data.merge(
    final_frames, on=['game_id', 'play_id']
).query('frame_id == final_frame_id').copy()

# Split offense/defense
offense_final = final_frame_data[final_frame_data['player_side'] == 'Offense'].copy()
defense_final = final_frame_data[final_frame_data['player_side'] == 'Defense'].copy()

print(f"Offense at throw: {len(offense_final)}")
print(f"Defense at throw: {len(defense_final)}")


# Calculate separation at throw
offense_valid = offense_final.merge(valid_plays, on=['game_id', 'play_id'])
defense_valid = defense_final.merge(valid_plays, on=['game_id', 'play_id'])

receivers = offense_valid[['game_id', 'play_id', 'nfl_id', 'x', 'y']].copy()
receivers.columns = ['game_id', 'play_id', 'rec_nfl_id', 'rec_x', 'rec_y']

defenders = defense_valid[['game_id', 'play_id', 'nfl_id', 'x', 'y']].copy()
defenders.columns = ['game_id', 'play_id', 'def_nfl_id', 'def_x', 'def_y']

pairs = receivers.merge(defenders, on=['game_id', 'play_id'])
pairs['distance'] = np.sqrt(
    (pairs['rec_x'] - pairs['def_x'])**2 + 
    (pairs['rec_y'] - pairs['def_y'])**2
)

idx_min = pairs.groupby(['game_id', 'play_id', 'rec_nfl_id'])['distance'].idxmin()
nearest = pairs.loc[idx_min].copy()

separation_df = nearest[['game_id', 'play_id', 'rec_nfl_id', 'distance', 'def_nfl_id']].copy()
separation_df.columns = ['game_id', 'play_id', 'nfl_id', 'separation_at_throw', 'defender_nfl_id']

print(f"Separation calculated for {len(separation_df)} receivers")
print(f"Mean separation: {separation_df['separation_at_throw'].mean():.2f} yards")

del pairs, receivers, defenders, nearest, idx_min


# Get receiver info
receiver_info = player_info[player_info['player_side'] == 'Offense'][
    ['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position']
].copy()
receiver_info.columns = ['game_id', 'play_id', 'nfl_id', 'receiver_name', 'receiver_position']

# Get defender info
defender_info = player_info[player_info['player_side'] == 'Defense'][
    ['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position']
].copy()
defender_info.columns = ['game_id', 'play_id', 'defender_nfl_id', 'defender_name', 'defender_position']

# Get targeted receiver IDs
targeted_receivers = player_info[player_info['player_role'] == 'Targeted Receiver'][
    ['game_id', 'play_id', 'nfl_id']
].copy()
targeted_receivers['is_targeted'] = True

# Get supplementary columns
supp_cols = [
    'game_id', 'play_id', 'possession_team', 'defensive_team',
    'pass_result', 'down', 'yards_to_go', 'quarter', 'pass_length',
    'yards_gained', 'route_of_targeted_receiver', 'play_action',
    'offense_formation', 'pass_location_type', 'team_coverage_type',
    'defenders_in_the_box', 'play_description'
]
available_cols = [c for c in supp_cols if c in supplementary.columns]
supp_filtered = supplementary[available_cols].drop_duplicates(subset=['game_id', 'play_id'])
supp_filtered = supp_filtered.merge(valid_plays, on=['game_id', 'play_id'])

# Build dataset
preliminary_project_set = separation_df.copy()
preliminary_project_set = preliminary_project_set.merge(receiver_info, on=['game_id', 'play_id', 'nfl_id'], how='left')
preliminary_project_set = preliminary_project_set.merge(defender_info, on=['game_id', 'play_id', 'defender_nfl_id'], how='left')
preliminary_project_set = preliminary_project_set.merge(targeted_receivers, on=['game_id', 'play_id', 'nfl_id'], how='left')
preliminary_project_set['is_targeted'] = preliminary_project_set['is_targeted'].fillna(False)
preliminary_project_set = preliminary_project_set.merge(supp_filtered, on=['game_id', 'play_id'], how='left')

# Add derived columns
preliminary_project_set['route'] = np.where(
    preliminary_project_set['is_targeted'],
    preliminary_project_set['route_of_targeted_receiver'],
    np.nan
)
preliminary_project_set['caught'] = np.where(
    preliminary_project_set['is_targeted'],
    (preliminary_project_set['pass_result'] == 'C').astype(float),
    np.nan
)
preliminary_project_set['is_touchdown'] = preliminary_project_set['play_description'].str.upper().str.contains('TOUCHDOWN', na=False)
preliminary_project_set['receiver_team'] = preliminary_project_set['possession_team']
preliminary_project_set['defender_team'] = preliminary_project_set['defensive_team']

print(f"Preliminary dataset: {preliminary_project_set.shape}")
print(f"Targeted receivers: {preliminary_project_set['is_targeted'].sum()}")


group_cols = ['game_id', 'play_id', 'nfl_id']

offense_tracking = input_data[input_data['player_side'] == 'Offense'].merge(
    valid_plays, on=['game_id', 'play_id']
)
offense_tracking = offense_tracking.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])

print(f"Offensive tracking rows: {len(offense_tracking):,}")

# Tier 1: Basic position metrics
first_frame = offense_tracking.groupby(group_cols).first().reset_index()
last_frame = offense_tracking.groupby(group_cols).last().reset_index()

route_features = first_frame[['game_id', 'play_id', 'nfl_id']].copy()
route_features['start_x'] = first_frame['x']
route_features['start_y'] = first_frame['y']
route_features['end_x'] = last_frame['x']
route_features['end_y'] = last_frame['y']

frame_counts = offense_tracking.groupby(group_cols)['frame_id'].count().reset_index()
frame_counts.columns = ['game_id', 'play_id', 'nfl_id', 'num_frames']
route_features = route_features.merge(frame_counts, on=group_cols)

route_features['net_vertical'] = route_features['end_x'] - route_features['start_x']
route_features['net_lateral_signed'] = route_features['start_y'] - route_features['end_y']
route_features['net_lateral'] = route_features['net_lateral_signed'].abs()
route_features['receiver_side'] = np.where(route_features['start_y'] > 26.65, 'right', 'left')

route_features['net_lateral_from_center'] = np.where(
    route_features['receiver_side'] == 'right',
    route_features['end_y'] - route_features['start_y'],
    route_features['start_y'] - route_features['end_y']
)

route_features['displacement'] = np.sqrt(
    route_features['net_vertical']**2 + route_features['net_lateral']**2
)
route_features['distance_to_endzone'] = 120 - route_features['end_x']
route_features['distance_from_center_end'] = (route_features['end_y'] - 26.65).abs()
route_features['distance_from_sideline_end'] = np.where(
    route_features['receiver_side'] == 'right',
    53.3 - route_features['end_y'],
    route_features['end_y']
)

route_features['crossed_middle'] = (
    ((route_features['start_y'] < 26.65) & (route_features['end_y'] > 26.65)) |
    ((route_features['start_y'] > 26.65) & (route_features['end_y'] < 26.65))
)
route_features['started_wide'] = (route_features['start_y'] - 26.65).abs() > 15
route_features['started_near_sideline'] = np.where(
    route_features['receiver_side'] == 'right',
    53.3 - route_features['start_y'] < 10,
    route_features['start_y'] < 10
)
route_features['ended_in_redzone'] = route_features['distance_to_endzone'] < 20
route_features['moving_toward_sideline'] = route_features['net_lateral_from_center'] > 0

print(f"Tier 1 complete: {len(route_features.columns)} features")


# Tier 2: Frame-to-frame metrics
offense_tracking['x_next'] = offense_tracking.groupby(group_cols)['x'].shift(-1)
offense_tracking['y_next'] = offense_tracking.groupby(group_cols)['y'].shift(-1)
offense_tracking['frame_distance'] = np.sqrt(
    (offense_tracking['x_next'] - offense_tracking['x'])**2 +
    (offense_tracking['y_next'] - offense_tracking['y'])**2
)

total_distance = offense_tracking.groupby(group_cols)['frame_distance'].sum().reset_index()
total_distance.columns = ['game_id', 'play_id', 'nfl_id', 'total_distance']
route_features = route_features.merge(total_distance, on=group_cols)

route_features['route_efficiency'] = np.where(
    route_features['total_distance'] > 0,
    route_features['displacement'] / route_features['total_distance'],
    0
)
route_features['curvature'] = 1 - route_features['route_efficiency']
route_features['avg_speed'] = np.where(
    route_features['num_frames'] > 0,
    route_features['total_distance'] / route_features['num_frames'],
    0
)
route_features['lateral_to_vertical_ratio'] = np.where(
    route_features['net_vertical'].abs() > 0.1,
    route_features['net_lateral'].abs() / route_features['net_vertical'].abs(),
    999
)

y_stats = offense_tracking.groupby(group_cols)['y'].agg(['max', 'min']).reset_index()
y_stats.columns = ['game_id', 'play_id', 'nfl_id', 'max_y', 'min_y']
route_features = route_features.merge(y_stats, on=group_cols)
route_features['max_width'] = np.maximum(
    (route_features['max_y'] - route_features['start_y']).abs(),
    (route_features['min_y'] - route_features['start_y']).abs()
)

x_min = offense_tracking.groupby(group_cols)['x'].min().reset_index()
x_min.columns = ['game_id', 'play_id', 'nfl_id', 'min_x']
route_features = route_features.merge(x_min, on=group_cols)
route_features['went_backward'] = route_features['min_x'] < route_features['start_x']
route_features = route_features.drop(columns=['max_y', 'min_y', 'min_x'])

print(f"Tier 2 complete: {len(route_features.columns)} features")


# Tier 3: Direction features
print("Calculating direction features (this takes ~30-60 seconds)...")

direction_features = offense_tracking.groupby(group_cols).apply(
    calculate_direction_features, include_groups=False
).reset_index()
route_features = route_features.merge(direction_features, on=group_cols)

print(f"Tier 3 complete: {len(route_features.columns)} features")


# Tier 4: Pattern detection
wheel_detection = offense_tracking.groupby(group_cols).apply(
    detect_wheel_pattern, include_groups=False
).reset_index()
wheel_detection.columns = ['game_id', 'play_id', 'nfl_id', 'is_wheel_pattern']
route_features = route_features.merge(wheel_detection, on=group_cols)

route_features['primary_direction'] = route_features.apply(classify_primary_direction, axis=1)
route_features['break_timing_ratio'] = 1.0

# Add player position
position_info = player_info[player_info['player_side'] == 'Offense'][
    ['game_id', 'play_id', 'nfl_id', 'player_position']
].drop_duplicates()
route_features = route_features.merge(position_info, on=group_cols, how='left')

print(f"Route features complete: {route_features.shape}")
print(f"Wheel patterns detected: {route_features['is_wheel_pattern'].sum()}")


# Merge route features to preliminary dataset
route_cols_to_add = [c for c in route_features.columns if c not in preliminary_project_set.columns or c in group_cols]
preliminary_project_set = preliminary_project_set.merge(
    route_features[route_cols_to_add],
    on=group_cols,
    how='left'
)

print(f"Dataset after route features: {preliminary_project_set.shape}")


# Prepare route classification data
route_df = preliminary_project_set[
    (preliminary_project_set['is_targeted'] == True) & 
    (preliminary_project_set['route'].notna())
].copy()

print(f"Targeted receivers with known routes: {len(route_df)}")

# One-hot encode
route_df_encoded = pd.get_dummies(route_df, columns=['receiver_side', 'primary_direction', 'receiver_position'])

feature_cols_encoded = ROUTE_FEATURE_COLS + [col for col in route_df_encoded.columns 
                                              if col.startswith(('receiver_side_', 'primary_direction_', 'receiver_position_'))]

X = route_df_encoded[feature_cols_encoded].copy()
y = route_df_encoded['route'].copy()

X = X.replace([np.inf, -np.inf], 999)
X = X.fillna(X.median())

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Route classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols_encoded)}")


# Train route classifier
route_model = xgb.XGBClassifier(
    **ROUTE_MODEL_PARAMS,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

route_model.fit(X_train, y_train)

y_pred = route_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Route Classification Accuracy: {test_accuracy:.4f}")

y_test_decoded = le.inverse_transform(y_test)
y_pred_decoded = le.inverse_transform(y_pred)
print(classification_report(y_test_decoded, y_pred_decoded))


# Predict routes for non-targeted receivers
labeled = preliminary_project_set[preliminary_project_set['route'].notna()].copy()
unlabeled = preliminary_project_set[preliminary_project_set['route'].isna()].copy()

print(f"Labeled (targeted): {len(labeled)}")
print(f"Unlabeled (non-targeted): {len(unlabeled)}")

unlabeled_encoded = pd.get_dummies(unlabeled, columns=['receiver_side', 'primary_direction', 'receiver_position'])

for col in feature_cols_encoded:
    if col not in unlabeled_encoded.columns:
        unlabeled_encoded[col] = 0

X_unlabeled = unlabeled_encoded[feature_cols_encoded].copy()
X_unlabeled = X_unlabeled.replace([np.inf, -np.inf], 999)
X_unlabeled = X_unlabeled.fillna(X.median())

predicted_encoded = route_model.predict(X_unlabeled)
prediction_probs = route_model.predict_proba(X_unlabeled)
prediction_confidence = prediction_probs.max(axis=1)

predicted_routes = le.inverse_transform(predicted_encoded)

unlabeled['predicted_route'] = predicted_routes
unlabeled['prediction_confidence'] = prediction_confidence

labeled['predicted_route'] = labeled['route']
labeled['prediction_confidence'] = 1.0

preliminary_project_set = pd.concat([labeled, unlabeled], ignore_index=True)
preliminary_project_set = preliminary_project_set.sort_values(['game_id', 'play_id', 'nfl_id']).reset_index(drop=True)

preliminary_project_set['current_route'] = preliminary_project_set['predicted_route'].fillna(
    preliminary_project_set['route']
)

print(f"All {len(preliminary_project_set)} receivers now have route assignments")


# Calculate receiver stats
targeted = preliminary_project_set[preliminary_project_set['is_targeted'] == True].copy()
targeted = targeted.merge(supplementary[['game_id', 'play_id', 'week']], on=['game_id', 'play_id'], how='left')

# Overall stats
receiver_overall = targeted.groupby('nfl_id').agg(
    total_catches=('caught', 'sum'),
    total_targets=('caught', 'count'),
    catch_rate=('caught', 'mean'),
    avg_yards_per_target=('yards_gained', 'mean')
).reset_index()

# Route-specific stats
route_stats = targeted.groupby(['nfl_id', 'current_route']).agg(
    catches_on_route=('caught', 'sum'),
    targets_on_route=('caught', 'count'),
    catch_rate_on_route=('caught', 'mean'),
    avg_yards_on_route=('yards_gained', 'mean')
).reset_index()
route_stats = route_stats.rename(columns={'current_route': 'route'})

# Recent form (last 3 weeks)
targeted_sorted = targeted.sort_values(['nfl_id', 'week'])
receiver_weeks = targeted_sorted.groupby('nfl_id')['week'].apply(lambda x: pd.Series(x.unique())).reset_index()
receiver_weeks.columns = ['nfl_id', 'week_rank', 'week']
weeks_per_receiver = receiver_weeks.groupby('nfl_id').size().reset_index(name='total_weeks')
receiver_weeks = receiver_weeks.merge(weeks_per_receiver, on='nfl_id')
receiver_weeks['week_order'] = receiver_weeks.groupby('nfl_id').cumcount()
receiver_weeks['is_last_3'] = receiver_weeks['week_order'] >= (receiver_weeks['total_weeks'] - 3)
last_3_weeks = receiver_weeks[receiver_weeks['is_last_3']][['nfl_id', 'week']]
targeted_last_3 = targeted.merge(last_3_weeks, on=['nfl_id', 'week'])

recent_form = targeted_last_3.groupby('nfl_id').agg(
    targets_last_3=('caught', 'count'),
    catches_last_3=('caught', 'sum'),
    catch_rate_last_3=('caught', 'mean'),
    yards_last_3=('yards_gained', 'sum')
).reset_index()

receiver_stats = receiver_overall.merge(recent_form, on='nfl_id', how='left')

# Merge to dataset
complete_with_stats = preliminary_project_set.merge(receiver_stats, on='nfl_id', how='left')
complete_with_stats = complete_with_stats.merge(
    route_stats,
    left_on=['nfl_id', 'current_route'],
    right_on=['nfl_id', 'route'],
    how='left',
    suffixes=('', '_route')
)

if 'route_route' in complete_with_stats.columns:
    complete_with_stats = complete_with_stats.drop(columns=['route_route'])

# Impute missing values
league_avg_catch_rate = targeted['caught'].mean()
league_avg_yards = targeted['yards_gained'].mean()

complete_with_stats['total_targets'] = complete_with_stats['total_targets'].fillna(0)
complete_with_stats['total_catches'] = complete_with_stats['total_catches'].fillna(0)
complete_with_stats['catch_rate'] = complete_with_stats['catch_rate'].fillna(league_avg_catch_rate)
complete_with_stats['avg_yards_per_target'] = complete_with_stats['avg_yards_per_target'].fillna(league_avg_yards)
complete_with_stats['targets_on_route'] = complete_with_stats['targets_on_route'].fillna(0)
complete_with_stats['catches_on_route'] = complete_with_stats['catches_on_route'].fillna(0)
complete_with_stats['catch_rate_on_route'] = complete_with_stats['catch_rate_on_route'].fillna(league_avg_catch_rate)
complete_with_stats['avg_yards_on_route'] = complete_with_stats['avg_yards_on_route'].fillna(league_avg_yards)
complete_with_stats['targets_last_3'] = complete_with_stats['targets_last_3'].fillna(0)
complete_with_stats['catches_last_3'] = complete_with_stats['catches_last_3'].fillna(0)
complete_with_stats['catch_rate_last_3'] = complete_with_stats['catch_rate_last_3'].fillna(league_avg_catch_rate)
complete_with_stats['yards_last_3'] = complete_with_stats['yards_last_3'].fillna(0)

print(f"Dataset with receiver stats: {complete_with_stats.shape}")

del targeted, targeted_sorted, receiver_weeks, weeks_per_receiver, last_3_weeks
del targeted_last_3, receiver_overall, recent_form, receiver_stats, route_stats


# Add context features
existing_cols = set(complete_with_stats.columns)
desired_supp_features = [
    'yardline_number', 'pre_snap_home_score', 'pre_snap_visitor_score',
    'receiver_alignment', 'dropback_type', 'team_coverage_man_zone'
]
new_features = [c for c in desired_supp_features if c not in existing_cols]

if new_features:
    merge_cols = ['game_id', 'play_id'] + new_features
    available_merge_cols = [c for c in merge_cols if c in supplementary.columns]
    complete_with_context = complete_with_stats.merge(
        supplementary[available_merge_cols].drop_duplicates(subset=['game_id', 'play_id']),
        on=['game_id', 'play_id'],
        how='left'
    )
else:
    complete_with_context = complete_with_stats.copy()

print(f"Dataset with context: {complete_with_context.shape}")


# QB stats
qb_identities = input_data[input_data['player_role'] == 'Passer'][
    ['game_id', 'play_id', 'nfl_id']
].drop_duplicates()
qb_identities.columns = ['game_id', 'play_id', 'qb_nfl_id']

complete_with_qb = complete_with_context.merge(qb_identities, on=['game_id', 'play_id'], how='left')

# Calculate QB season stats
supp_with_qb = supplementary[
    (supplementary['pass_result'].isin(['C', 'I'])) &
    (supplementary['play_nullified_by_penalty'] == 'N')
].merge(qb_identities, on=['game_id', 'play_id'], how='left')

supp_with_qb['is_complete'] = (supp_with_qb['pass_result'] == 'C').astype(int)

qb_overall = supp_with_qb.groupby('qb_nfl_id').agg(
    qb_total_attempts=('play_id', 'count'),
    qb_total_completions=('is_complete', 'sum'),
    qb_total_yards=('yards_gained', 'sum'),
    qb_avg_depth_of_target=('pass_length', 'mean')
).reset_index()

qb_overall['qb_completion_pct'] = qb_overall['qb_total_completions'] / qb_overall['qb_total_attempts']
qb_overall['qb_yards_per_attempt'] = qb_overall['qb_total_yards'] / qb_overall['qb_total_attempts']

# Third down stats
third_down = supp_with_qb[supp_with_qb['down'] == 3].groupby('qb_nfl_id').agg(
    td_comp=('is_complete', 'sum'),
    td_att=('is_complete', 'count')
).reset_index()
third_down['qb_third_down_comp_pct'] = third_down['td_comp'] / third_down['td_att']
qb_overall = qb_overall.merge(third_down[['qb_nfl_id', 'qb_third_down_comp_pct']], on='qb_nfl_id', how='left')

# Deep ball stats
deep = supp_with_qb[supp_with_qb['pass_length'] >= 20].groupby('qb_nfl_id').agg(
    deep_comp=('is_complete', 'sum'),
    deep_att=('is_complete', 'count')
).reset_index()
deep['qb_deep_comp_pct'] = deep['deep_comp'] / deep['deep_att']
qb_overall = qb_overall.merge(deep[['qb_nfl_id', 'qb_deep_comp_pct']], on='qb_nfl_id', how='left')

# QB route-specific stats
qb_route_stats = supp_with_qb.groupby(['qb_nfl_id', 'route_of_targeted_receiver']).agg(
    completions=('is_complete', 'sum'),
    attempts=('is_complete', 'count')
).reset_index()
qb_route_stats['qb_comp_pct_on_route'] = qb_route_stats['completions'] / qb_route_stats['attempts']
qb_route_stats = qb_route_stats.rename(columns={'route_of_targeted_receiver': 'predicted_route'})

# Merge QB stats
complete_with_qb = complete_with_qb.merge(qb_overall, on='qb_nfl_id', how='left')
complete_with_qb = complete_with_qb.merge(
    qb_route_stats[['qb_nfl_id', 'predicted_route', 'qb_comp_pct_on_route']],
    on=['qb_nfl_id', 'predicted_route'],
    how='left'
)

# Impute missing QB stats
league_avg_qb_comp = qb_overall['qb_completion_pct'].mean()
complete_with_qb['qb_completion_pct'] = complete_with_qb['qb_completion_pct'].fillna(league_avg_qb_comp)
complete_with_qb['qb_comp_pct_on_route'] = complete_with_qb['qb_comp_pct_on_route'].fillna(league_avg_qb_comp)
complete_with_qb['qb_third_down_comp_pct'] = complete_with_qb['qb_third_down_comp_pct'].fillna(league_avg_qb_comp)
complete_with_qb['qb_deep_comp_pct'] = complete_with_qb['qb_deep_comp_pct'].fillna(league_avg_qb_comp)

print(f"Dataset with QB stats: {complete_with_qb.shape}")


# Route depth categories
ROUTE_DEPTH_CATEGORY = {
    'SCREEN': 'SHORT', 'FLAT': 'SHORT', 'HITCH': 'SHORT', 'SLANT': 'SHORT',
    'IN': 'MEDIUM', 'ANGLE': 'MEDIUM', 'OUT': 'MEDIUM', 'CROSS': 'MEDIUM',
    'CORNER': 'LONG', 'WHEEL': 'LONG', 'POST': 'LONG', 'GO': 'LONG'
}

complete_with_qb['route_depth_category'] = complete_with_qb['predicted_route'].map(ROUTE_DEPTH_CATEGORY)

# Count routes per play by depth
play_route_counts = complete_with_qb.groupby(['game_id', 'play_id']).agg(
    num_short=('route_depth_category', lambda x: (x == 'SHORT').sum()),
    num_medium=('route_depth_category', lambda x: (x == 'MEDIUM').sum()),
    num_long=('route_depth_category', lambda x: (x == 'LONG').sum()),
    num_screen=('predicted_route', lambda x: (x == 'SCREEN').sum()),
    avg_net_vertical=('net_vertical', 'mean')
).reset_index()

play_route_counts['num_quick_routes'] = play_route_counts['num_short'] + play_route_counts['num_medium']

# Classify concept
conditions = [
    (play_route_counts['num_screen'] >= 1),
    (play_route_counts['num_long'] >= 2) & (play_route_counts['avg_net_vertical'] > 5),
    (play_route_counts['num_quick_routes'] >= 3),
]
choices = ['SCREEN', 'DEEP', 'QUICK']
play_route_counts['concept_base'] = np.select(conditions, choices, default='QUICK')

# Add play action
play_action_map = complete_with_qb[['game_id', 'play_id', 'play_action']].drop_duplicates()
play_route_counts = play_route_counts.merge(play_action_map, on=['game_id', 'play_id'], how='left')

play_route_counts['concept_final'] = np.where(
    play_route_counts['play_action'] == True,
    play_route_counts['concept_base'] + '-PA',
    play_route_counts['concept_base']
)

play_route_counts['concept_final'] = play_route_counts['concept_final'].replace({
    'SCREEN-PA': 'QUICK-PA'
})

# Merge to dataset
complete_with_qb = complete_with_qb.merge(
    play_route_counts[['game_id', 'play_id', 'concept_final']],
    on=['game_id', 'play_id'],
    how='left'
)

print(f"Concept distribution:")
print(complete_with_qb['concept_final'].value_counts())


# Defensive metrics at throw
offense_at_throw = offense_final.merge(valid_plays, on=['game_id', 'play_id'])
defense_at_throw = defense_final.merge(valid_plays, on=['game_id', 'play_id'])

receivers = offense_at_throw[['game_id', 'play_id', 'nfl_id', 'x', 'y']].copy()
receivers.columns = ['game_id', 'play_id', 'nfl_id', 'rec_x', 'rec_y']

defenders = defense_at_throw[['game_id', 'play_id', 'nfl_id', 'x', 'y', 'o']].copy()
defenders.columns = ['game_id', 'play_id', 'def_nfl_id', 'def_x', 'def_y', 'def_o']

pairs = receivers.merge(defenders, on=['game_id', 'play_id'])
pairs['distance'] = np.sqrt(
    (pairs['rec_x'] - pairs['def_x'])**2 +
    (pairs['rec_y'] - pairs['def_y'])**2
)

# Defender counts
pairs['within_5'] = (pairs['distance'] <= 5).astype(int)
pairs['within_10'] = (pairs['distance'] <= 10).astype(int)

defender_counts = pairs.groupby(['game_id', 'play_id', 'nfl_id']).agg(
    defenders_within_5_yards_at_throw=('within_5', 'sum'),
    defenders_within_10_yards_at_throw=('within_10', 'sum')
).reset_index()

# Nearest defender angle
idx_nearest = pairs.groupby(['game_id', 'play_id', 'nfl_id'])['distance'].idxmin()
nearest_defenders = pairs.loc[idx_nearest].copy()

nearest_defenders['dx'] = nearest_defenders['rec_x'] - nearest_defenders['def_x']
nearest_defenders['dy'] = nearest_defenders['rec_y'] - nearest_defenders['def_y']
nearest_defenders['angle_to_receiver'] = np.arctan2(
    nearest_defenders['dy'], nearest_defenders['dx']
) * 180 / np.pi
nearest_defenders['angle_diff'] = np.abs(
    nearest_defenders['angle_to_receiver'] - nearest_defenders['def_o']
)
nearest_defenders['angle_diff'] = np.where(
    nearest_defenders['angle_diff'] > 180,
    360 - nearest_defenders['angle_diff'],
    nearest_defenders['angle_diff']
)

nearest_defender_angle = nearest_defenders[['game_id', 'play_id', 'nfl_id', 'angle_diff']].copy()
nearest_defender_angle.columns = ['game_id', 'play_id', 'nfl_id', 'nearest_defender_angle_at_throw']

# Defenders near QB
qb_at_throw = offense_at_throw[offense_at_throw['player_role'] == 'Passer'][
    ['game_id', 'play_id', 'x', 'y']
].copy()
qb_at_throw.columns = ['game_id', 'play_id', 'qb_x', 'qb_y']

defenders_vs_qb = defense_at_throw.merge(qb_at_throw, on=['game_id', 'play_id'])
defenders_vs_qb['dist_to_qb'] = np.sqrt(
    (defenders_vs_qb['x'] - defenders_vs_qb['qb_x'])**2 +
    (defenders_vs_qb['y'] - defenders_vs_qb['qb_y'])**2
)
defenders_vs_qb['near_qb'] = (defenders_vs_qb['dist_to_qb'] <= 4).astype(int)

defenders_near_qb = defenders_vs_qb.groupby(['game_id', 'play_id']).agg(
    defenders_near_qb_at_throw=('near_qb', 'sum')
).reset_index()

# Merge defensive metrics
complete_with_qb = complete_with_qb.merge(defender_counts, on=['game_id', 'play_id', 'nfl_id'], how='left')
complete_with_qb = complete_with_qb.merge(nearest_defender_angle, on=['game_id', 'play_id', 'nfl_id'], how='left')
complete_with_qb = complete_with_qb.merge(defenders_near_qb, on=['game_id', 'play_id'], how='left')

print(f"Dataset with defensive metrics: {complete_with_qb.shape}")

del pairs, defender_counts, nearest_defenders, nearest_defender_angle, defenders_near_qb


# Defender season stats
player_sides = player_info[['nfl_id', 'player_side']].drop_duplicates()
output_with_sides = output.merge(player_sides, on='nfl_id', how='left')
defenders_output = output_with_sides[output_with_sides['player_side'] == 'Defense'].copy()
defenders_output = defenders_output.sort_values(['nfl_id', 'game_id', 'play_id', 'frame_id'])

defenders_output['x_next'] = defenders_output.groupby(['nfl_id', 'game_id', 'play_id'])['x'].shift(-1)
defenders_output['y_next'] = defenders_output.groupby(['nfl_id', 'game_id', 'play_id'])['y'].shift(-1)
defenders_output['frame_next'] = defenders_output.groupby(['nfl_id', 'game_id', 'play_id'])['frame_id'].shift(-1)

defenders_output['frame_distance'] = np.sqrt(
    (defenders_output['x_next'] - defenders_output['x'])**2 +
    (defenders_output['y_next'] - defenders_output['y'])**2
)
defenders_output['frame_time'] = (defenders_output['frame_next'] - defenders_output['frame_id']) / 10.0
defenders_output['frame_speed'] = np.where(
    defenders_output['frame_time'] > 0,
    defenders_output['frame_distance'] / defenders_output['frame_time'],
    np.nan
)

defender_speed_stats = defenders_output.groupby('nfl_id').agg(
    defender_avg_closing_speed=('frame_speed', 'mean'),
    speed_sample_size=('frame_speed', 'count')
).reset_index()
defender_speed_stats.columns = ['defender_nfl_id', 'defender_avg_closing_speed', 'speed_sample_size']

# Catch rate allowed
targeted_with_def = complete_with_qb[complete_with_qb['is_targeted'] == True][
    ['game_id', 'play_id', 'nfl_id', 'defender_nfl_id', 'caught']
].copy()

defender_coverage = targeted_with_def.groupby('defender_nfl_id').agg(
    catches_allowed=('caught', 'sum'),
    targets_covered=('caught', 'count'),
    catch_rate_allowed=('caught', 'mean')
).reset_index()

defender_stats = defender_speed_stats.merge(
    defender_coverage[['defender_nfl_id', 'catch_rate_allowed', 'targets_covered']],
    on='defender_nfl_id',
    how='outer'
)

league_avg_speed = defender_stats['defender_avg_closing_speed'].mean()
league_avg_catch_allowed = defender_stats['catch_rate_allowed'].mean()

defender_stats['defender_avg_closing_speed'] = defender_stats['defender_avg_closing_speed'].fillna(league_avg_speed)
defender_stats['catch_rate_allowed'] = defender_stats['catch_rate_allowed'].fillna(league_avg_catch_allowed)

complete_with_qb = complete_with_qb.merge(
    defender_stats[['defender_nfl_id', 'defender_avg_closing_speed', 'catch_rate_allowed']],
    on='defender_nfl_id',
    how='left'
)

complete_with_qb['defender_avg_closing_speed'] = complete_with_qb['defender_avg_closing_speed'].fillna(league_avg_speed)
complete_with_qb['catch_rate_allowed'] = complete_with_qb['catch_rate_allowed'].fillna(league_avg_catch_allowed)

print(f"Dataset with defender stats: {complete_with_qb.shape}")


# Prepare data
complete_with_qb_full = complete_with_qb.copy()

# Handle duplicate columns from merges
all_cols = complete_with_qb.columns.tolist()
x_cols = [col for col in all_cols if col.endswith('_x')]
y_cols = [col for col in all_cols if col.endswith('_y')]

for col in y_cols:
    base_name = col[:-2]
    x_version = base_name + '_x'
    if x_version in x_cols:
        complete_with_qb = complete_with_qb.drop(columns=[col])
        complete_with_qb = complete_with_qb.rename(columns={x_version: base_name})
    else:
        complete_with_qb = complete_with_qb.drop(columns=[col])

# Filter to targeted receivers
modeling_data = complete_with_qb[complete_with_qb['is_targeted'] == True].copy()

cols_to_exclude = [col for col in CATCH_MODEL_EXCLUDE_COLS if col in modeling_data.columns]

y = modeling_data['caught'].copy()
X = modeling_data.drop(columns=cols_to_exclude)

# Identify categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()
cat_features = categorical_cols + boolean_cols

print(f"Features: {X.shape[1]}")
print(f"Categorical: {len(cat_features)}")


# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Handle NaNs in categorical columns
for col in cat_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna('Missing').astype(str)
        X_val[col] = X_val[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)

print(f"Train: {len(X_train)} ({y_train.mean()*100:.1f}% completion)")
print(f"Val: {len(X_val)} ({y_val.mean()*100:.1f}% completion)")
print(f"Test: {len(X_test)} ({y_test.mean()*100:.1f}% completion)")


# Calculate class weights
n_completions = y_train.sum()
n_incompletions = len(y_train) - n_completions
weight_ratio = n_completions / n_incompletions
class_weights = {0: weight_ratio, 1: 1.0}

print(f"Class weights: {class_weights}")

# Train model
catch_model = CatBoostClassifier(
    **CATCH_MODEL_PARAMS,
    class_weights=class_weights,
    cat_features=cat_features,
    random_state=42,
    verbose=100,
    eval_metric='AUC'
)

catch_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    use_best_model=True
)


# Evaluate
y_test_pred = catch_model.predict(X_test)
y_test_prob = catch_model.predict_proba(X_test)[:, 1]

print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
print(classification_report(y_test, y_test_pred, target_names=['Incomplete', 'Complete']))


start_time = time.time()

model_features = list(X_train.columns)

# Filter out QBs
all_receivers = complete_with_qb_full[
    complete_with_qb_full['nfl_id'] != complete_with_qb_full['qb_nfl_id']
].copy()

print(f"Total receivers (excluding QBs): {len(all_receivers):,}")

# Prepare features
available_features = [f for f in model_features if f in all_receivers.columns]
missing_features = [f for f in model_features if f not in all_receivers.columns]

if missing_features:
    print(f"Warning: {len(missing_features)} missing features will be filled with 0")

X_all = all_receivers[available_features].copy()

for col in missing_features:
    X_all[col] = 0

X_all = X_all[model_features]

for col in cat_features:
    if col in X_all.columns:
        X_all[col] = X_all[col].fillna('Missing').astype(str)

# Predict
catch_probs = catch_model.predict_proba(X_all)[:, 1]
all_receivers['catch_probability'] = catch_probs

# Compute rankings
all_receivers['prob_rank'] = all_receivers.groupby(['game_id', 'play_id'])['catch_probability'].rank(ascending=False)
all_receivers['is_best_option'] = all_receivers.groupby(['game_id', 'play_id'])['catch_probability'].transform('max') == all_receivers['catch_probability']
all_receivers['timing_window_frames'] = all_receivers['concept_final'].map(TIMING_FRAMES).fillna(20)

print(f"Inference complete in {time.time() - start_time:.2f} seconds")
print(f"Predictions: {len(all_receivers):,}")


# Create predictions dataframe
output_cols = [
    'game_id', 'play_id', 'nfl_id',
    'catch_probability', 'prob_rank', 'is_best_option',
    'concept_final', 'timing_window_frames',
    'receiver_position', 'predicted_route', 'current_route',
    'is_targeted', 'caught',
    'separation_at_throw', 'separation_at_decision',
    'defenders_within_5_yards_at_throw', 'defenders_within_10_yards_at_throw',
    'catch_rate_on_route', 'catch_rate_allowed',
    'qb_nfl_id', 'qb_completion_pct',
    'start_x', 'start_y', 'end_x', 'end_y', 'net_vertical', 'net_lateral'
]

output_cols = [c for c in output_cols if c in all_receivers.columns]
predictions_df = all_receivers[output_cols].copy()

print(f"Predictions dataset: {predictions_df.shape}")


# QB Target Distribution by Model Rank
targeted = predictions_df[predictions_df['is_targeted'] == True]
rank_counts = targeted['prob_rank'].value_counts().sort_index()
total = len(targeted)

print("QB Target Distribution by Model Rank:")
print("=" * 45)
for rank in sorted(rank_counts.index):
    count = rank_counts[rank]
    pct = count / total * 100
    print(f"Rank {int(rank)}: {count:,} targets ({pct:.1f}%)")

print(f"\nCatch Rates by Rank:")
for rank in sorted(targeted['prob_rank'].unique()):
    subset = targeted[targeted['prob_rank'] == rank]
    catch_rate = subset['caught'].mean()
    print(f"Rank {int(rank)}: {catch_rate:.1%}")


# Model vs QB Decision Analysis
targeted = predictions_df[predictions_df['is_targeted'] == True].copy()

agreed = targeted[targeted['is_best_option'] == True]
disagreed = targeted[targeted['is_best_option'] == False]

print(f"Overall completion rate: {targeted['caught'].mean()*100:.1f}%")
print(f"\nWhen model AGREED with QB ({len(agreed):,} plays):")
print(f"  Completion rate: {agreed['caught'].mean()*100:.1f}%")
print(f"\nWhen model DISAGREED with QB ({len(disagreed):,} plays):")
print(f"  Completion rate: {disagreed['caught'].mean()*100:.1f}%")
print(f"\n  Difference: {(agreed['caught'].mean() - disagreed['caught'].mean())*100:.1f} percentage points")

# What was the model's alternative?
disagreed_plays = disagreed[['game_id', 'play_id']].drop_duplicates()
model_preferred = predictions_df[
    predictions_df[['game_id', 'play_id']].apply(tuple, axis=1).isin(
        disagreed_plays.apply(tuple, axis=1)
    ) & (predictions_df['is_best_option'] == True)
]

print(f"\nOn plays where model disagreed:")
print(f"  QB's target avg catch prob: {disagreed['catch_probability'].mean()*100:.1f}%")
print(f"  Model's best option avg catch prob: {model_preferred['catch_probability'].mean()*100:.1f}%")
print(f"  Missed opportunity gap: {(model_preferred['catch_probability'].mean() - disagreed['catch_probability'].mean())*100:.1f}pp")


# Visualization: Catch Rate by Model Agreement
agreed_catch_rate = agreed['caught'].mean()
disagreed_catch_rate = disagreed['caught'].mean()
league_avg_catch_rate = targeted['caught'].mean()

n_icons = 100
grid_size = 10

def get_football_marker():
    verts = np.array([
        [-1.0,  0.0], [-0.7,  0.25], [-0.4,  0.38], [-0.1,  0.42],
        [ 0.1,  0.42], [ 0.4,  0.38], [ 0.7,  0.25], [ 1.0,  0.0],
        [ 0.7, -0.25], [ 0.4, -0.38], [ 0.1, -0.42], [-0.1, -0.42],
        [-0.4, -0.38], [-0.7, -0.25], [-1.0,  0.0],
    ])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

football_marker = get_football_marker()

def create_icon_grid(catch_rate):
    n_catches = int(round(catch_rate * n_icons))
    x_coords, y_coords, colors = [], [], []
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            x_coords.append(col)
            y_coords.append(row)
            colors.append('#2ecc71' if idx < n_catches else '#e74c3c')
            idx += 1
    return np.array(x_coords), np.array(y_coords), colors

fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor='#1a1a2e')

catch_color = '#2ecc71'
incomplete_color = '#e74c3c'
marker_size = 500

for ax, catch_rate, title in zip(
    axes, 
    [agreed_catch_rate, disagreed_catch_rate],
    ['QB Agreed with Model', 'QB Disagreed with Model']
):
    ax.set_facecolor('#1a1a2e')
    x, y, colors = create_icon_grid(catch_rate)
    
    for xi, yi, color in zip(x, y, colors):
        ax.scatter(xi, yi, marker=football_marker, s=marker_size, 
                   c=color, edgecolors='white', linewidths=1.2, zorder=2)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15)
    
    rate_color = catch_color if catch_rate > 0.8 else incomplete_color
    ax.text(4.5, -2, f'{catch_rate:.1%}', fontsize=28, fontweight='bold',
            ha='center', va='top', color=rate_color)
    ax.text(4.5, -3, 'Catch Rate', fontsize=12, ha='center', va='top', color='#888888')
    
    ax.set_xlim(-1, 10)
    ax.set_ylim(-4, 10)
    ax.set_aspect('equal')
    ax.axis('off')

fig.suptitle('Catch Rate by Model Agreement', fontsize=22, fontweight='bold', 
             color='white', y=0.96)

legend_elements = [
    mpatches.Patch(facecolor=catch_color, edgecolor='white', label='Catch'),
    mpatches.Patch(facecolor=incomplete_color, edgecolor='white', label='Incomplete')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
           bbox_to_anchor=(0.5, 0.91), fontsize=12, frameon=False,
           labelcolor='white')

fig.text(0.5, 0.06, f'League Average: {league_avg_catch_rate:.1%}', 
         ha='center', fontsize=14, color='#aaaaaa', style='italic')

plt.tight_layout(rect=[0, 0.08, 1, 0.88])
plt.show()


# Save predictions
predictions_df.to_parquet('precomputed_predictions.parquet', index=False)
print("Predictions saved to precomputed_predictions.parquet")

# Save app version (smaller)
app_cols = [
    'game_id', 'play_id', 'nfl_id',
    'catch_probability', 'prob_rank', 'is_best_option',
    'concept_final', 'timing_window_frames',
    'receiver_position', 'predicted_route',
    'is_targeted', 'caught',
    'separation_at_throw'
]
app_cols = [c for c in app_cols if c in predictions_df.columns]
predictions_app = predictions_df[app_cols].copy()
predictions_app.to_parquet('predictions_for_app.parquet', index=False)
print("App predictions saved to predictions_for_app.parquet")


# Use predictions_df from Section 12
precomputed = predictions_df

# 1. Season-level receiver catches
targeted_plays = complete_with_qb_full[complete_with_qb_full['is_targeted'] == True].copy()

season_catches = targeted_plays.groupby('nfl_id').agg(
    total_catches_season=('caught', 'sum'),
    total_targets_season=('is_targeted', 'count')
).reset_index()

season_catches['total_catches_season'] = season_catches['total_catches_season'].astype(int)
season_catches['total_targets_season'] = season_catches['total_targets_season'].astype(int)

print(f"Receivers with targets: {len(season_catches)}")

# 2. Receivers app file
receivers_app = precomputed.copy()

# Add names from player_info if not present
if 'receiver_name' not in receivers_app.columns:
    name_lookup = player_info[['nfl_id', 'player_name']].drop_duplicates()
    receivers_app = receivers_app.merge(name_lookup, on='nfl_id', how='left')
    receivers_app['receiver_name'] = receivers_app['player_name']
    receivers_app.drop(columns=['player_name'], inplace=True, errors='ignore')

# Add position from player_info if not present  
if 'receiver_position' not in receivers_app.columns:
    pos_lookup = player_info[['nfl_id', 'player_position']].drop_duplicates()
    receivers_app = receivers_app.merge(pos_lookup, on='nfl_id', how='left')
    receivers_app['receiver_position'] = receivers_app['player_position']
    receivers_app.drop(columns=['player_position'], inplace=True, errors='ignore')

receivers_app = receivers_app.merge(season_catches, on='nfl_id', how='left')
receivers_app['total_catches_season'] = receivers_app['total_catches_season'].fillna(0).astype(int)
receivers_app['total_targets_season'] = receivers_app['total_targets_season'].fillna(0).astype(int)

if 'timing_window_frames' in receivers_app.columns:
    receivers_app['timing_frames'] = receivers_app['timing_window_frames']

receiver_cols = [
    'game_id', 'play_id', 'nfl_id',
    'receiver_name', 'receiver_position',
    'predicted_route', 'catch_probability', 'prob_rank', 'is_best_option',
    'concept_final', 'timing_frames',
    'is_targeted', 'caught',
    'separation_at_throw', 'separation_at_decision',
    'total_catches_season', 'total_targets_season'
]
receiver_cols = [c for c in receiver_cols if c in receivers_app.columns]
receivers_app = receivers_app[receiver_cols].copy()

print(f"receivers_app: {receivers_app.shape}")

# 3. Plays app file
plays_app = receivers_app[['game_id', 'play_id', 'concept_final', 'timing_frames']].drop_duplicates()

targeted_receivers = receivers_app[receivers_app['is_targeted'] == True][
    ['game_id', 'play_id', 'nfl_id', 'caught']
].drop_duplicates()
targeted_receivers.columns = ['game_id', 'play_id', 'target_nfl_id', 'target_caught']

plays_app = plays_app.merge(targeted_receivers, on=['game_id', 'play_id'], how='left')

plays_app['play_result'] = np.where(
    plays_app['target_caught'] == 1, 'CAUGHT',
    np.where(plays_app['target_caught'] == 0, 'INCOMPLETE', 'UNKNOWN')
)
plays_app.drop(columns=['target_caught'], inplace=True, errors='ignore')

print(f"plays_app: {plays_app.shape}")

# 4. Tracking app file
play_lookup = receivers_app[['game_id', 'play_id']].drop_duplicates()

input_filtered = input_data.merge(play_lookup, on=['game_id', 'play_id'], how='inner')
input_filtered['source'] = 'input'

output_filtered = output.merge(play_lookup, on=['game_id', 'play_id'], how='inner')
output_filtered['source'] = 'output'

tracking_cols = ['game_id', 'play_id', 'frame_id', 'nfl_id', 'x', 'y', 'player_side', 'player_role', 'source']
input_cols = [c for c in tracking_cols if c in input_filtered.columns]
output_cols = [c for c in tracking_cols if c in output_filtered.columns]

input_filtered = input_filtered[input_cols]
output_filtered = output_filtered[output_cols]

tracking_app = pd.concat([input_filtered, output_filtered], ignore_index=True)

print(f"tracking_app: {tracking_app.shape}")

del input_filtered, output_filtered

# 5. Supplementary (as-is)
supplementary_app = supplementary.copy()
print(f"supplementary_app: {supplementary_app.shape}")

# 6. Save all files
receivers_app.to_parquet('receivers_app.parquet', index=False)
plays_app.to_parquet('plays_app.parquet', index=False)
tracking_app.to_parquet('tracking_app.parquet', index=False)
supplementary_app.to_parquet('supplementary_app.parquet', index=False)

print('\nApp data files saved:')
print('  - receivers_app.parquet')
print('  - plays_app.parquet')
print('  - tracking_app.parquet')
print('  - supplementary_app.parquet')


!pip install dash -q


# =============================================================================
# SECTION 16: QB DECISION TRAINER APP
# =============================================================================

from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import plotly.graph_objects as go

# Split tracking into input/output
input_data_app = tracking_app[tracking_app['source'] == 'input'].copy()
output_data_app = tracking_app[tracking_app['source'] == 'output'].copy()

# Get QB names from player_info
qb_lookup = player_info[player_info['player_role'] == 'Passer'][['game_id', 'play_id', 'player_name']].drop_duplicates()
qb_lookup = qb_lookup.rename(columns={'player_name': 'qb_name'})
plays_app_with_qb = plays_app.merge(qb_lookup, on=['game_id', 'play_id'], how='left')
plays_app_with_qb['qb_name'] = plays_app_with_qb['qb_name'].fillna('QB')

print(f"Input frames: {len(input_data_app):,}")
print(f"Output frames: {len(output_data_app):,}")
print(f"Unique QBs: {plays_app_with_qb['qb_name'].nunique()}")

# =============================================================================
# PLAY LOADER FUNCTION
# =============================================================================

def load_play_for_coaching(game_id, play_id):
    """Load all data needed for a single play in the coaching app."""
    preds = receivers_app[
        (receivers_app['game_id'] == game_id) & 
        (receivers_app['play_id'] == play_id)
    ].copy()
    
    if len(preds) == 0:
        return None
    
    play_info = plays_app_with_qb[
        (plays_app_with_qb['game_id'] == game_id) & 
        (plays_app_with_qb['play_id'] == play_id)
    ]
    
    if len(play_info) == 0:
        return None
    
    play_info = play_info.iloc[0]
    
    supp = supplementary_app[
        (supplementary_app['game_id'] == game_id) & 
        (supplementary_app['play_id'] == play_id)
    ]
    
    supp_info = supp.iloc[0] if len(supp) > 0 else None
    
    input_frames = sorted(input_data_app[
        (input_data_app['game_id'] == game_id) & 
        (input_data_app['play_id'] == play_id)
    ]['frame_id'].unique().tolist())
    
    output_frames = sorted(output_data_app[
        (output_data_app['game_id'] == game_id) & 
        (output_data_app['play_id'] == play_id)
    ]['frame_id'].unique().tolist())
    
    if len(input_frames) == 0:
        return None
    
    best = preds[preds['is_best_option'] == True]
    if len(best) == 0:
        best = preds.loc[preds['catch_probability'].idxmax()]
    else:
        best = best.iloc[0]
    
    best_option = {
        'nfl_id': int(best['nfl_id']),
        'receiver_name': best.get('receiver_name', 'Unknown'),
        'receiver_position': best.get('receiver_position', '??'),
        'predicted_route': best.get('predicted_route', '?'),
        'catch_probability': float(best['catch_probability'])
    }
    
    actual_target = None
    targeted = preds[preds['is_targeted'] == True]
    if len(targeted) > 0:
        t = targeted.iloc[0]
        actual_target = {
            'nfl_id': int(t['nfl_id']),
            'receiver_name': t.get('receiver_name', 'Unknown'),
            'receiver_position': t.get('receiver_position', '??'),
            'predicted_route': t.get('predicted_route', '?'),
            'catch_probability': float(t['catch_probability'])
        }
    
    timing_frames = int(play_info.get('timing_frames', 20))
    pause_idx = min(timing_frames, len(input_frames) - 1)
    pause_frame = input_frames[pause_idx]
    
    first_frame = input_frames[0]
    qb_data = input_data_app[
        (input_data_app['game_id'] == game_id) & 
        (input_data_app['play_id'] == play_id) &
        (input_data_app['frame_id'] == first_frame) &
        (input_data_app['player_role'] == 'Passer')
    ]
    
    if len(qb_data) > 0:
        qb_pos = [float(qb_data['x'].iloc[0]), float(qb_data['y'].iloc[0])]
    else:
        qb_pos = [25.0, 26.65]
    
    off_data = input_data_app[
        (input_data_app['game_id'] == game_id) & 
        (input_data_app['play_id'] == play_id) &
        (input_data_app['frame_id'] == first_frame) &
        (input_data_app['player_side'] == 'Offense')
    ]
    
    def_data = input_data_app[
        (input_data_app['game_id'] == game_id) & 
        (input_data_app['play_id'] == play_id) &
        (input_data_app['frame_id'] == first_frame) &
        (input_data_app['player_side'] == 'Defense')
    ]
    
    if len(off_data) > 0 and len(def_data) > 0:
        avg_off_x = off_data['x'].mean()
        avg_def_x = def_data['x'].mean()
        attacking_right = avg_def_x > avg_off_x
    else:
        attacking_right = True
    
    preds_sorted = preds.sort_values('total_catches_season', ascending=False)
    receiver_rankings = {
        int(row['nfl_id']): idx + 1 
        for idx, (_, row) in enumerate(preds_sorted.iterrows())
    }
    
    result = play_info.get('play_result', 'UNKNOWN')
    qb_name = play_info.get('qb_name', 'QB')
    
    if supp_info is not None:
        possession_team = supp_info.get('possession_team', 'OFF')
        defensive_team = supp_info.get('defensive_team', 'DEF')
        yards_to_go_raw = supp_info.get('yards_to_go', 10)
        yards_to_go = int(yards_to_go_raw) if not pd.isna(yards_to_go_raw) else 10
        yards_gained_raw = supp_info.get('yards_gained', 0)
        yards_gained = int(yards_gained_raw) if not pd.isna(yards_gained_raw) else 0
        ep_raw = supp_info.get('expected_points', 0)
        ep = float(ep_raw) if not pd.isna(ep_raw) else 0.0
        epa_raw = supp_info.get('expected_points_added', 0)
        epa = float(epa_raw) if not pd.isna(epa_raw) else 0.0
        home_team = supp_info.get('home_team_abbr', 'HOME')
        visitor_team = supp_info.get('visitor_team_abbr', 'AWAY')
        home_score_raw = supp_info.get('pre_snap_home_score', 0)
        home_score = int(home_score_raw) if not pd.isna(home_score_raw) else 0
        visitor_score_raw = supp_info.get('pre_snap_visitor_score', 0)
        visitor_score = int(visitor_score_raw) if not pd.isna(visitor_score_raw) else 0
        yardline_raw = supp_info.get('yardline_number', 50)
        yardline = int(yardline_raw) if not pd.isna(yardline_raw) else 50
        down_raw = supp_info.get('down', 1)
        down = int(down_raw) if not pd.isna(down_raw) else 1
        is_touchdown = 'TOUCHDOWN' in str(supp_info.get('play_description', '')).upper()
    else:
        possession_team, defensive_team = 'OFF', 'DEF'
        yards_to_go, yards_gained = 10, 0
        ep, epa = 0.0, 0.0
        home_team, visitor_team = 'HOME', 'AWAY'
        home_score, visitor_score = 0, 0
        yardline, down = 50, 1
        is_touchdown = False
    
    if possession_team == home_team:
        off_score, def_score = home_score, visitor_score
        off_team, def_team = home_team, visitor_team
    else:
        off_score, def_score = visitor_score, home_score
        off_team, def_team = visitor_team, home_team
    
    if attacking_right:
        los_x = 10 + yardline
        first_down_x = los_x + yards_to_go
    else:
        los_x = 110 - yardline
        first_down_x = los_x - yards_to_go
    
    return {
        'game_id': game_id,
        'play_id': play_id,
        'input_frames': input_frames,
        'output_frames': output_frames,
        'pause_frame': pause_frame,
        'best_nfl_id': best_option['nfl_id'],
        'best_name': best_option['receiver_name'],
        'best_position': best_option['receiver_position'],
        'best_route': best_option['predicted_route'],
        'best_prob': best_option['catch_probability'],
        'actual_nfl_id': actual_target['nfl_id'] if actual_target else None,
        'actual_name': actual_target['receiver_name'] if actual_target else None,
        'actual_route': actual_target['predicted_route'] if actual_target else None,
        'actual_prob': actual_target['catch_probability'] if actual_target else None,
        'result': result,
        'receiver_rankings': receiver_rankings,
        'qb_pos': qb_pos,
        'attacking_right': attacking_right,
        'qb_name': qb_name,
        'off_team': off_team,
        'def_team': def_team,
        'off_score': off_score,
        'def_score': def_score,
        'down': down,
        'yards_to_go': yards_to_go,
        'yards_gained': yards_gained,
        'los_x': los_x,
        'first_down_x': first_down_x,
        'ep': ep,
        'epa': epa,
        'is_touchdown': is_touchdown
    }

# =============================================================================
# CREATE APP
# =============================================================================

def create_app():
    app = Dash(__name__)
    
    POSITION_COLORS = {'WR': '#4fc3f7', 'TE': '#81c784', 'RB': '#ffb74d', 'FB': '#ff8a65'}
    DEFAULT_COLOR = '#b0bec5'
    
    play_list = plays_app_with_qb[['game_id', 'play_id']].drop_duplicates().values.tolist()
    
    TIMER_OPTIONS = [
        {'label': 'Off', 'value': 0},
        {'label': '2s', 'value': 2},
        {'label': '5s', 'value': 5},
        {'label': '10s', 'value': 10},
    ]
    BASE_FRAME_DURATION = 120
    
    def create_field_figure(los_x=None, first_down_x=None):
        fig = go.Figure()
        field_width = 53.3
        field_length = 120
        
        fig.add_shape(type="rect", x0=0, y0=0, x1=field_length, y1=field_width,
                      fillcolor="#2e7d32", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=field_width,
                      fillcolor="#1b5e20", line=dict(color="white", width=2), layer="below")
        fig.add_shape(type="rect", x0=110, y0=0, x1=120, y1=field_width,
                      fillcolor="#1b5e20", line=dict(color="white", width=2), layer="below")
        
        for yard in range(10, 111, 5):
            width = 2 if yard % 10 == 0 else 1
            fig.add_shape(type="line", x0=yard, y0=0, x1=yard, y1=field_width,
                          line=dict(color="white", width=width), layer="below")
        
        for yard in range(10, 100, 10):
            display_num = yard if yard <= 50 else 100 - yard
            fig.add_annotation(x=yard+10, y=field_width/2, text=str(display_num),
                              font=dict(size=20, color="white"), showarrow=False, opacity=0.5)
        
        if los_x is not None:
            fig.add_shape(type="line", x0=los_x, y0=0, x1=los_x, y1=field_width,
                          line=dict(color="#2196f3", width=3), layer="below")
        
        if first_down_x is not None and 10 <= first_down_x <= 110:
            fig.add_shape(type="line", x0=first_down_x, y0=0, x1=first_down_x, y1=field_width,
                          line=dict(color="#ffd700", width=3), layer="below")
        
        fig.update_layout(
            xaxis=dict(range=[-5, 125], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
            yaxis=dict(range=[-5, field_width+5], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", fixedrange=True),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='#1a1a1a',
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            autosize=True,
            showlegend=True,
            legend=dict(bgcolor='rgba(30,30,30,0.9)', bordercolor='white', borderwidth=1,
                       font=dict(color='white', size=10), x=1.01, y=1.0)
        )
        return fig
    
    def get_frame_data(game_id, play_id, frame_id, source='input'):
        if source == 'input':
            data = input_data_app
        else:
            data = output_data_app
        
        frame = data[
            (data['game_id'] == game_id) &
            (data['play_id'] == play_id) &
            (data['frame_id'] == frame_id)
        ]
        return frame.to_dict('records')
    
    app.layout = html.Div([
        html.H2("QB Decision Trainer", style={'textAlign': 'center', 'color': 'white', 'marginBottom': '10px'}),
        
        html.Div([
            html.Button("New Play", id='new-play-btn', n_clicks=0,
                       style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px'}),
            html.Button("Play ▶", id='play-btn', n_clicks=0,
                       style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px'}),
            html.Span("Timer: ", style={'color': 'white', 'marginRight': '5px'}),
            dcc.Dropdown(id='timer-dropdown', options=TIMER_OPTIONS, value=0,
                        style={'width': '80px', 'display': 'inline-block'}),
            html.Span(id='timer-display', style={'color': '#ffd700', 'marginLeft': '10px', 'fontSize': '18px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),
        
        html.Div(id='play-info', style={'textAlign': 'center', 'color': 'white', 'marginBottom': '10px'}),
        
        dcc.Graph(id='field-graph', figure=create_field_figure(), 
                  config={'displayModeBar': False}),
        
        html.Div(id='feedback-panel', style={'display': 'none'}),
        html.Div(id='instructions', children="Click 'New Play' to start",
                style={'textAlign': 'center', 'color': '#888', 'marginTop': '10px'}),
        
        dcc.Store(id='play-data'),
        dcc.Store(id='app-phase', data='idle'),
        dcc.Store(id='current-frame', data=0),
        dcc.Store(id='selected-receiver'),
        dcc.Store(id='timer-remaining'),
        dcc.Interval(id='animation-interval', interval=BASE_FRAME_DURATION, disabled=True),
        dcc.Interval(id='timer-interval', interval=1000, disabled=True),
    ], style={'backgroundColor': '#1a1a1a', 'padding': '20px', 'minHeight': '100vh'})
    
    @app.callback(
        Output('play-data', 'data'),
        Output('field-graph', 'figure'),
        Output('play-info', 'children'),
        Output('app-phase', 'data'),
        Output('instructions', 'children'),
        Output('feedback-panel', 'style'),
        Input('new-play-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def load_new_play(n_clicks):
        import random
        game_id, play_id = random.choice(play_list)
        play_data = load_play_for_coaching(game_id, play_id)
        
        if play_data is None:
            return None, create_field_figure(), "Error loading play", 'idle', "Click 'New Play'", {'display': 'none'}
        
        fig = create_field_figure(los_x=play_data['los_x'], first_down_x=play_data['first_down_x'])
        
        first_frame = play_data['input_frames'][0]
        frame_data = get_frame_data(game_id, play_id, first_frame)
        
        preds = receivers_app[
            (receivers_app['game_id'] == game_id) & 
            (receivers_app['play_id'] == play_id)
        ]
        receiver_ids = set(preds['nfl_id'].tolist())
        receiver_info = {row['nfl_id']: row for _, row in preds.iterrows()}
        
        for p in frame_data:
            nfl_id = p['nfl_id']
            if p['player_side'] == 'Defense':
                fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers',
                    marker=dict(size=10, color='#ef5350'), showlegend=False, hoverinfo='skip'))
            elif p['player_role'] == 'Passer':
                fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers',
                    marker=dict(size=14, color='white', symbol='square'), 
                    name=play_data['qb_name'], hoverinfo='skip'))
            elif nfl_id in receiver_ids:
                pred = receiver_info[nfl_id]
                pos = pred.get('receiver_position', '??')
                name = pred.get('receiver_name', 'Unknown')
                route = pred.get('predicted_route', '?')
                rank = play_data['receiver_rankings'].get(nfl_id, 99)
                color = POSITION_COLORS.get(pos, DEFAULT_COLOR)
                
                name_parts = str(name).split() if name else ['?']
                display_name = f"{name_parts[0][0]}. {name_parts[-1]}" if len(name_parts) >= 2 else str(name)
                
                fig.add_trace(go.Scatter(
                    x=[p['x']], y=[p['y']], mode='markers+text',
                    marker=dict(size=16, color=color, line=dict(width=1, color='white')),
                    text=str(rank) if rank <= 5 else '',
                    textposition='middle center',
                    textfont=dict(size=9, color='white'),
                    name=f"{rank}. {display_name} ({pos})",
                    customdata=[nfl_id],
                    hovertemplate=f"{display_name}<br>{route}<extra></extra>"
                ))
            else:
                fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers',
                    marker=dict(size=8, color='#607d8b'), showlegend=False, hoverinfo='skip'))
        
        info = f"{play_data['off_team']} {play_data['off_score']} - {play_data['def_team']} {play_data['def_score']} | {play_data['down']} & {play_data['yards_to_go']}"
        
        return play_data, fig, info, 'paused', "Click a receiver to make your choice", {'display': 'none'}
    
    @app.callback(
        Output('feedback-panel', 'children'),
        Output('feedback-panel', 'style', allow_duplicate=True),
        Output('selected-receiver', 'data'),
        Output('instructions', 'children', allow_duplicate=True),
        Input('field-graph', 'clickData'),
        State('play-data', 'data'),
        State('app-phase', 'data'),
        prevent_initial_call=True
    )
    def handle_selection(click_data, play_data, phase):
        if phase != 'paused' or click_data is None or play_data is None:
            return "", {'display': 'none'}, None, no_update
        
        point = click_data['points'][0]
        if 'customdata' not in point:
            return "", {'display': 'none'}, None, no_update
        
        selected_id = point['customdata']
        if isinstance(selected_id, list):
            selected_id = selected_id[0]
        
        preds = receivers_app[
            (receivers_app['game_id'] == play_data['game_id']) & 
            (receivers_app['play_id'] == play_data['play_id'])
        ]
        
        selected_row = preds[preds['nfl_id'] == selected_id]
        if len(selected_row) == 0:
            return "", {'display': 'none'}, None, no_update
        
        selected_row = selected_row.iloc[0]
        selected_prob = selected_row['catch_probability']
        selected_route = selected_row.get('predicted_route', '?')
        selected_name = selected_row.get('receiver_name', 'Unknown')
        
        is_best = selected_id == play_data['best_nfl_id']
        
        if is_best:
            color = '#4caf50'
            msg = "Great choice!"
        elif selected_prob >= play_data['best_prob'] * 0.8:
            color = '#ff9800'
            msg = "Good option"
        else:
            color = '#f44336'
            msg = "Risky choice"
        
        feedback = html.Div([
            html.P(f"Your pick: {selected_name}", style={'color': color, 'fontWeight': 'bold', 'margin': '5px'}),
            html.P(f"{selected_route} | {selected_prob*100:.0f}% catch prob", style={'color': 'white', 'margin': '5px'}),
            html.Hr(style={'borderColor': '#444'}),
            html.P(f"Best option: {play_data['best_name']}", style={'color': '#ffd700', 'margin': '5px'}),
            html.P(f"{play_data['best_route']} | {play_data['best_prob']*100:.0f}%", style={'color': 'white', 'margin': '5px'}),
            html.P(msg, style={'color': color, 'fontWeight': 'bold', 'marginTop': '10px'}),
        ])
        
        style = {
            'display': 'block',
            'position': 'fixed',
            'right': '20px',
            'top': '100px',
            'backgroundColor': '#2a2a2a',
            'padding': '15px',
            'borderRadius': '8px',
            'border': f'2px solid {color}',
            'width': '200px'
        }
        
        return feedback, style, selected_id, "Click 'Play' to see the result"
    
    @app.callback(
        Output('field-graph', 'figure', allow_duplicate=True),
        Output('app-phase', 'data', allow_duplicate=True),
        Output('animation-interval', 'disabled'),
        Output('current-frame', 'data'),
        Output('instructions', 'children', allow_duplicate=True),
        Input('play-btn', 'n_clicks'),
        State('play-data', 'data'),
        State('app-phase', 'data'),
        prevent_initial_call=True
    )
    def start_animation(n_clicks, play_data, phase):
        if play_data is None or phase not in ['paused', 'selected']:
            return no_update, no_update, True, 0, no_update
        
        return no_update, 'playing', False, 0, "Watch the play unfold..."
    
    @app.callback(
        Output('field-graph', 'figure', allow_duplicate=True),
        Output('current-frame', 'data', allow_duplicate=True),
        Output('animation-interval', 'disabled', allow_duplicate=True),
        Output('app-phase', 'data', allow_duplicate=True),
        Output('instructions', 'children', allow_duplicate=True),
        Input('animation-interval', 'n_intervals'),
        State('play-data', 'data'),
        State('current-frame', 'data'),
        State('app-phase', 'data'),
        prevent_initial_call=True
    )
    def animate_frame(n_intervals, play_data, current_frame, phase):
        if play_data is None or phase != 'playing':
            return no_update, no_update, True, no_update, no_update
        
        all_frames = play_data['input_frames'] + play_data['output_frames']
        
        if current_frame >= len(all_frames):
            result = play_data['result']
            result_text = f"Result: {result}"
            if result == 'CAUGHT':
                result_text += f" for {play_data.get('yards_gained', 0)} yards"
                if play_data.get('is_touchdown'):
                    result_text += " TOUCHDOWN!"
            return no_update, current_frame, True, 'finished', result_text
        
        frame_id = all_frames[current_frame]
        source = 'input' if frame_id in play_data['input_frames'] else 'output'
        frame_data = get_frame_data(play_data['game_id'], play_data['play_id'], frame_id, source)
        
        fig = create_field_figure(los_x=play_data['los_x'], first_down_x=play_data['first_down_x'])
        
        preds = receivers_app[
            (receivers_app['game_id'] == play_data['game_id']) & 
            (receivers_app['play_id'] == play_data['play_id'])
        ]
        receiver_ids = set(preds['nfl_id'].tolist())
        receiver_info = {row['nfl_id']: row for _, row in preds.iterrows()}
        
        for p in frame_data:
            nfl_id = p['nfl_id']
            if p['player_side'] == 'Defense':
                fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers',
                    marker=dict(size=10, color='#ef5350'), showlegend=False, hoverinfo='skip'))
            elif p['player_role'] == 'Passer':
                fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers',
                    marker=dict(size=14, color='white', symbol='square'),
                    name=play_data['qb_name'], hoverinfo='skip'))
            elif nfl_id in receiver_ids:
                pred = receiver_info[nfl_id]
                pos = pred.get('receiver_position', '??')
                color = POSITION_COLORS.get(pos, DEFAULT_COLOR)
                rank = play_data['receiver_rankings'].get(nfl_id, 99)
                
                if nfl_id == play_data.get('actual_nfl_id'):
                    color = '#ffd700'
                
                fig.add_trace(go.Scatter(
                    x=[p['x']], y=[p['y']], mode='markers+text',
                    marker=dict(size=16, color=color, line=dict(width=1, color='white')),
                    text=str(rank) if rank <= 5 else '',
                    textposition='middle center',
                    textfont=dict(size=9, color='white'),
                    showlegend=False, hoverinfo='skip'
                ))
            else:
                fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers',
                    marker=dict(size=8, color='#607d8b'), showlegend=False, hoverinfo='skip'))
        
        return fig, current_frame + 1, False, 'playing', f"Frame {current_frame + 1}/{len(all_frames)}"
    
    return app

# =============================================================================
# RUN APP
# =============================================================================
app = create_app()
# For Kaggle: use inline mode
# For local: change to jupyter_mode='external' for clickable link
app.run(debug=False, port=8050, jupyter_mode='external')

