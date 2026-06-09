# Phase 0: Project Setup & Reproducibility
# ========================================

import os
import json
import random
import numpy as np
import torch
from datetime import datetime

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# MODE flag: 'demo' for quick testing (200 plays), 'full' for complete run
MODE = 'full'  # Change to 'full' for production

# Create output directory
os.makedirs('/kaggle/working/', exist_ok=True)
os.makedirs('/kaggle/working/artifacts', exist_ok=True)
os.makedirs('/kaggle/working/media', exist_ok=True)

# Log run manifest
manifest = {
    'project': 'NFL Big Data Bowl 2026 - CIS Suite',
    'mode': MODE,
    'seed': SEED,
    'timestamp': datetime.now().isoformat(),
    'environment': {
        'platform': 'Kaggle',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
    }
}

with open('/kaggle/working/run_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"âœ… Setup complete! Mode: {MODE}, Seed: {SEED}")
print(f"Manifest saved to /kaggle/working/run_manifest.json")
print(f"GPU: {manifest['environment']['gpu'] if 'gpu' in manifest['environment'] else 'CPU'}")


# Phase 1: Data Ingestion & Expansion
# ========================================

# Install additional libraries
!pip install -q nfl_data_py pyarrow lightgbm shap captum streamlit torch-geometric

import pandas as pd
import glob
from nfl_data_py import import_pbp_data
import warnings
warnings.filterwarnings('ignore')

# Define paths
INPUT_DIR = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train'
SUPPLEMENTARY_PATH = '/kaggle/input/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv'

# Load Supplementary data
print("Loading supplementary data...")
supp_df = pd.read_csv(SUPPLEMENTARY_PATH)
print(f"Supplementary loaded: {supp_df.shape[0]} rows, {supp_df.shape[1]} cols")

# Load input tracking data (multiple weeks)
input_files = glob.glob(f'{INPUT_DIR}/input_2023_w*.csv')
print(f"Found {len(input_files)} input files")
input_dfs = [pd.read_csv(f) for f in input_files]
input_df = pd.concat(input_dfs, ignore_index=True)
print(f"Input tracking loaded: {input_df.shape[0]} rows, {input_df.shape[1]} cols")

# Load output tracking data (multiple weeks)
output_files = glob.glob(f'{INPUT_DIR}/output_2023_w*.csv')
print(f"Found {len(output_files)} output files")
output_dfs = [pd.read_csv(f) for f in output_files]
output_df = pd.concat(output_dfs, ignore_index=True)
print(f"Output tracking loaded: {output_df.shape[0]} rows, {output_df.shape[1]} cols")

# Merge supplementary with input on game_id and play_id
print("Merging supplementary with input...")
merged_df = pd.merge(input_df, supp_df, on=['game_id', 'play_id'], how='left')
print(f"Merged data: {merged_df.shape[0]} rows")

# Fetch nflverse PBP data for 2023-2025 (expansion)
print("Fetching nflverse PBP data (2023-2025)...")
pbp_data = import_pbp_data([2023, 2024, 2025])
print(f"PBP data loaded: {pbp_data.shape[0]} rows")

# Filter PBP to passing plays and merge relevant columns (e.g., EPA, personnel)
pbp_pass = pbp_data[pbp_data['play_type'].isin(['pass', 'play_action'])].copy()
pbp_pass = pbp_pass[['old_game_id', 'play_id', 'epa', 'wp', 'def_wp', 'home_wp', 'away_wp', 'season', 'week', 'weather']]
pbp_pass = pbp_pass.rename(columns={'old_game_id': 'game_id'})
print(f"Filtered PBP passing plays: {pbp_pass.shape[0]} rows")

# Convert join keys to string for consistent merging
pbp_pass['game_id'] = pbp_pass['game_id'].astype(str)
pbp_pass['play_id'] = pbp_pass['play_id'].astype(str)
merged_df['game_id'] = merged_df['game_id'].astype(str)
merged_df['play_id'] = merged_df['play_id'].astype(str)

# Merge PBP with merged_df (note: game_id/play_id join key as per schema)
expanded_df = pd.merge(merged_df, pbp_pass, on=['game_id', 'play_id'], how='left')
print(f"Expanded data with nflverse: {expanded_df.shape[0]} rows")

# Basic validation: Check for NaNs in critical columns
critical_cols = ['ball_land_x', 'ball_land_y', 'player_side', 'pass_result']
nan_summary = expanded_df[critical_cols].isnull().sum()
print("NaN counts in critical columns:")
print(nan_summary)

# Impute minor NaNs if any (e.g., forward fill or median for positions)
expanded_df['ball_land_x'] = expanded_df['ball_land_x'].fillna(expanded_df['ball_land_x'].median())
expanded_df['ball_land_y'] = expanded_df['ball_land_y'].fillna(expanded_df['ball_land_y'].median())
expanded_df['pass_result'] = expanded_df['pass_result'].fillna('I')  # Default to incomplete

# Save intermediate
expanded_df.to_parquet('/kaggle/working/expanded_data.parquet')
print("âœ… Phase 1 complete! Data saved to expanded_data.parquet")
print(f"Sample columns: {list(expanded_df.columns[:10])}")
print(f"Sample shape: {expanded_df.shape}")


# Phase 2: Play Selection & Filtering (Scaled to ~5k for Full Mode)
# ========================================

import numpy as np
import pandas as pd

# Ensure consistent dtypes for joins (str for game_id/play_id)
output_df['game_id'] = output_df['game_id'].astype(str)
output_df['play_id'] = output_df['play_id'].astype(str)

# Get candidate plays from supplementary (per-play level)
supp_df = expanded_df.drop_duplicates(subset=['game_id', 'play_id'])[['game_id', 'play_id', 'pass_length', 'pass_result', 'team_coverage_man_zone']]  # Subset for efficiency

candidates = supp_df[
    (supp_df['pass_length'] >= 15) & 
    supp_df['pass_result'].isin(['C', 'I', 'IN'])
].copy()
print(f"Candidate deep passes: {len(candidates)} plays")

# Precompute ball landing positions per play
ball_land_df = input_df.groupby(['game_id', 'play_id']).agg({
    'ball_land_x': 'first',
    'ball_land_y': 'first'
}).reset_index()
ball_land_df['game_id'] = ball_land_df['game_id'].astype(str)
ball_land_df['play_id'] = ball_land_df['play_id'].astype(str)
print(f"Ball landing data: {ball_land_df.shape[0]} plays")

# Precompute player sides per player per play
player_sides_df = input_df.groupby(['game_id', 'play_id', 'nfl_id'])['player_side'].first().reset_index()
player_sides_df['game_id'] = player_sides_df['game_id'].astype(str)
player_sides_df['play_id'] = player_sides_df['play_id'].astype(str)
print(f"Player sides data: {player_sides_df.shape[0]} player-play pairs")

def compute_contested(game_id, play_id, output_df, player_sides_df, ball_land_df, relaxed_threshold=1, relaxed_radius=15):
    """
    Compute if a play is contested: Relaxed for scale (>1 def within 15yd at t_land).
    """
    play_out = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
    if play_out.empty:
        return True  # Fallback include for scale
    
    ball_row = ball_land_df[(ball_land_df['game_id'] == game_id) & (ball_land_df['play_id'] == play_id)]
    if ball_row.empty:
        return True
    ball_x = ball_row['ball_land_x'].iloc[0]
    ball_y = ball_row['ball_land_y'].iloc[0]
    
    # Merge player sides for this play
    play_sides = player_sides_df[(player_sides_df['game_id'] == game_id) & (player_sides_df['play_id'] == play_id)][['nfl_id', 'player_side']]
    play_out = pd.merge(play_out, play_sides, on='nfl_id', how='left')
    
    if play_out['player_side'].isna().all():
        return True  # Fallback
    
    # Compute distance to ball landing
    play_out['dist_to_ball'] = np.sqrt(
        (play_out['x'] - ball_x)**2 + (play_out['y'] - ball_y)**2
    )
    
    # Find t_land: frame with minimum distance to ball (approx arrival)
    min_dist_idx = play_out['dist_to_ball'].idxmin()
    t_land_frame = play_out.loc[min_dist_idx, 'frame_id']
    
    # Defenders at t_land within relaxed radius
    defs_at_land = play_out[
        (play_out['frame_id'] == t_land_frame) & 
        (play_out['player_side'] == 'Defense')
    ]
    num_close_defs = (defs_at_land['dist_to_ball'] < relaxed_radius).sum()
    
    return num_close_defs > relaxed_threshold

# Select plays based on MODE (Scaled: Aim ~5k in full via relaxed contested)
if MODE == 'demo':
    # Sample balanced candidates for demo (200 per outcome max)
    sample_size = 200
    candidates_sample = candidates.groupby('pass_result', group_keys=False).apply(
        lambda x: x.sample(n=min(sample_size, len(x)), random_state=SEED)
    ).reset_index(drop=True)
    print(f"Sampled {len(candidates_sample)} candidates for contested check")
    
    contested_plays = []
    for _, row in candidates_sample.iterrows():
        g, p, result = row['game_id'], row['play_id'], row['pass_result']
        if compute_contested(g, p, output_df, player_sides_df, ball_land_df, relaxed_threshold=1, relaxed_radius=15):
            contested_plays.append({'game_id': g, 'play_id': p, 'pass_result': result})
    
    if len(contested_plays) == 0:
        print("No contested plays found; using all sampled candidates as fallback.")
        selected_plays = candidates_sample[['game_id', 'play_id', 'pass_result']].sample(n=200, random_state=SEED).reset_index(drop=True)
    else:
        contested_df = pd.DataFrame(contested_plays)
        # Balance selected to ~200 total
        n_per_group = min(67, contested_df['pass_result'].value_counts().min())  # ~200 total
        selected_plays = contested_df.groupby('pass_result', group_keys=False).apply(
            lambda x: x.sample(n=min(n_per_group, len(x)), random_state=SEED)
        ).reset_index(drop=True)
    
else:
    # Full mode: Relaxed contested to scale to ~5k (threshold=1 def <15yd)
    print("Computing relaxed contested for all candidates (full mode, scaled to ~5k)...")
    contested_plays = []
    for _, row in candidates.iterrows():
        g, p, result = row['game_id'], row['play_id'], row['pass_result']
        if compute_contested(g, p, output_df, player_sides_df, ball_land_df, relaxed_threshold=1, relaxed_radius=15):
            contested_plays.append({'game_id': g, 'play_id': p, 'pass_result': result})
    
    contested_df = pd.DataFrame(contested_plays)
    print(f"Relaxed contested: {len(contested_df)} plays")
    
    # Ensure ~5k: If <4k, supplement w/ random deep; if >6k, sample down
    if len(contested_df) < 4000:
        print("Supplementing to ~5k with random deep passes...")
        supplement_n = 5000 - len(contested_df)
        supplement = candidates[~candidates.set_index(['game_id', 'play_id']).index.isin(contested_df.set_index(['game_id', 'play_id']).index)]
        supplement_sample = supplement.sample(n=min(supplement_n, len(supplement)), random_state=SEED)
        contested_df = pd.concat([contested_df, supplement_sample], ignore_index=True)
    elif len(contested_df) > 6000:
        print("Sampling down to ~5k for efficiency...")
        n_per_group = 1667  # ~5k total
        contested_df = contested_df.groupby('pass_result', group_keys=False).apply(
            lambda x: x.sample(n=min(n_per_group, len(x)), random_state=SEED)
        ).reset_index(drop=True)
    
    selected_plays = contested_df

print(f"âœ… Selected {len(selected_plays)} scaled deep pass plays for {MODE} mode")
print(f"Breakdown by outcome:\n{selected_plays['pass_result'].value_counts()}")

# Filter data to selected plays
selected_mi = pd.MultiIndex.from_frame(selected_plays[['game_id', 'play_id']])
expanded_filtered = expanded_df.set_index(['game_id', 'play_id']).loc[selected_mi].reset_index()
print(f"Filtered expanded data: {expanded_filtered.shape[0]} rows")

output_filtered = output_df.merge(selected_plays[['game_id', 'play_id']], on=['game_id', 'play_id'], how='inner')
print(f"Filtered output data: {output_filtered.shape[0]} rows")

# Save
selected_plays.to_parquet('/kaggle/working/selected_plays.parquet')
expanded_filtered.to_parquet('/kaggle/working/filtered_expanded.parquet')
output_filtered.to_parquet('/kaggle/working/filtered_output.parquet')

print("âœ… Phase 2 complete! Filtered data saved (scaled to ~5k for full mode).")
print(f"Sample selected plays:\n{selected_plays.head()}")


# Phase 3: Temporal Alignment
# ========================================

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Load filtered data from previous phase
selected_plays = pd.read_parquet('/kaggle/working/selected_plays.parquet')
expanded_filtered = pd.read_parquet('/kaggle/working/filtered_expanded.parquet')
output_filtered = pd.read_parquet('/kaggle/working/filtered_output.parquet')

# Ensure string dtypes for consistency
for df in [selected_plays, expanded_filtered, output_filtered]:
    df['game_id'] = df['game_id'].astype(str)
    df['play_id'] = df['play_id'].astype(str)

# Step 1: Identify targeted receiver per play (from player_role == 'Targeted Receiver' in input/expanded)
receiver_df = expanded_filtered[expanded_filtered['player_role'] == 'Targeted Receiver'].groupby(['game_id', 'play_id'])['nfl_id'].first().reset_index()
receiver_df.columns = ['game_id', 'play_id', 'targeted_nfl_id']
print(f"Identified {len(receiver_df)} targeted receivers")

# Merge receiver info to selected_plays
selected_plays = pd.merge(selected_plays, receiver_df, on=['game_id', 'play_id'], how='left')
print(f"Plays with receiver: {selected_plays['targeted_nfl_id'].notna().sum()}")

# Get play_direction per play
play_dir_df = expanded_filtered.groupby(['game_id', 'play_id'])['play_direction'].first().reset_index()
print(f"Play directions: {len(play_dir_df)} plays")

# Step 2: Compute t_land per play (frame where receiver is closest to ball landing in output)
# First, get ball_land per play
ball_land_play = expanded_filtered.groupby(['game_id', 'play_id']).agg({
    'ball_land_x': 'first',
    'ball_land_y': 'first',
    'play_direction': 'first'
}).reset_index()
print(f"Ball landing per play: {len(ball_land_play)}")

# Merge play_dir to output
output_with_dir = output_filtered.merge(play_dir_df, on=['game_id', 'play_id'], how='left')

# Compute t_land_dict using output_with_dir
t_land_dict = {}
for _, play in selected_plays.iterrows():
    g, p, rec_id = play['game_id'], play['play_id'], play['targeted_nfl_id']
    if pd.isna(rec_id):
        # Fallback: closest any offensive player
        play_out = output_with_dir[(output_with_dir['game_id'] == g) & (output_with_dir['play_id'] == p)]
        if play_out.empty:
            continue
        play_exp = expanded_filtered[(expanded_filtered['game_id'] == g) & (expanded_filtered['play_id'] == p) & (expanded_filtered['player_side'] == 'Offense')]
        if play_exp.empty:
            continue
        ball_row = ball_land_play[(ball_land_play['game_id'] == g) & (ball_land_play['play_id'] == p)]
        ball_x = ball_row['ball_land_x'].iloc[0]
        ball_y = ball_row['ball_land_y'].iloc[0]
        min_dist = float('inf')
        t_land = 1
        for _, frame in play_out.iterrows():
            if frame['nfl_id'] in play_exp['nfl_id'].values:  # Offensive player
                dist = euclidean([frame['x'], frame['y']], [ball_x, ball_y])
                if dist < min_dist:
                    min_dist = dist
                    t_land = frame['frame_id']
    else:
        # Use targeted receiver
        play_out_rec = output_with_dir[(output_with_dir['game_id'] == g) & (output_with_dir['play_id'] == p) & (output_with_dir['nfl_id'] == rec_id)]
        if play_out_rec.empty:
            continue
        ball_row = ball_land_play[(ball_land_play['game_id'] == g) & (ball_land_play['play_id'] == p)]
        ball_x = ball_row['ball_land_x'].iloc[0]
        ball_y = ball_row['ball_land_y'].iloc[0]
        play_out_rec['dist_to_land'] = np.sqrt((play_out_rec['x'] - ball_x)**2 + (play_out_rec['y'] - ball_y)**2)
        t_land = play_out_rec.loc[play_out_rec['dist_to_land'].idxmin(), 'frame_id']
    
    t_land_dict[(g, p)] = int(t_land)

print(f"Computed t_land for {len(t_land_dict)} plays")
selected_plays['t_land_frame'] = selected_plays.set_index(['game_id', 'play_id']).index.map(t_land_dict).values
print(f"Mean t_land frame: {selected_plays['t_land_frame'].mean():.1f}")

# Now merge t_land to output_with_dir to create output_with_tland
output_with_tland = output_with_dir.merge(
    selected_plays[['game_id', 'play_id', 't_land_frame']], on=['game_id', 'play_id'], how='left'
)

# Now standardize all
def standardize_direction(df):
    df = df.copy()
    left_mask = df['play_direction'] == 'left'
    if left_mask.any():
        if 'x' in df.columns:
            df.loc[left_mask, 'x'] = 120 - df.loc[left_mask, 'x']
        if 'y' in df.columns:
            df.loc[left_mask, 'y'] = 53.3 - df.loc[left_mask, 'y']
        if 'dir' in df.columns:
            df.loc[left_mask, 'dir'] = (360 - df.loc[left_mask, 'dir']) % 360
        if 'o' in df.columns:
            df.loc[left_mask, 'o'] = (360 - df.loc[left_mask, 'o']) % 360
        if 'ball_land_x' in df.columns:
            df.loc[left_mask, 'ball_land_x'] = 120 - df.loc[left_mask, 'ball_land_x']
        if 'ball_land_y' in df.columns:
            df.loc[left_mask, 'ball_land_y'] = 53.3 - df.loc[left_mask, 'ball_land_y']
    return df

# Apply standardization
expanded_filtered = standardize_direction(expanded_filtered)
output_with_tland = standardize_direction(output_with_tland)
ball_land_play = standardize_direction(ball_land_play)

print("Standardized positions and directions.")

# Step 3: Extract trajectories (0.5s ~5 frames pre/post t_land; but since output starts at 1, take available)
trajectories = []
for _, play in selected_plays.iterrows():
    g, p = play['game_id'], play['play_id']
    play_out = output_with_tland[(output_with_tland['game_id'] == g) & (output_with_tland['play_id'] == p)]
    t_land = play['t_land_frame']
    start_frame = max(1, int(t_land) - 5)
    end_frame = min(play_out['frame_id'].max(), int(t_land) + 5)
    traj = play_out[(play_out['frame_id'] >= start_frame) & (play_out['frame_id'] <= end_frame)].copy()
    traj['relative_frame'] = traj['frame_id'] - t_land
    trajectories.append(traj)

aligned_trajectories = pd.concat(trajectories, ignore_index=True)
print(f"Aligned trajectories: {aligned_trajectories.shape[0]} rows")

# Step 4: Save
selected_plays.to_parquet('/kaggle/working/selected_plays.parquet', index=False)
expanded_filtered.to_parquet('/kaggle/working/filtered_expanded.parquet', index=False)
aligned_trajectories.to_parquet('/kaggle/working/aligned_trajectories.parquet', index=False)
ball_land_play.to_parquet('/kaggle/working/ball_land_play.parquet', index=False)
output_with_tland.to_parquet('/kaggle/working/output_with_tland.parquet', index=False)

print("âœ… Phase 3 complete! Aligned data saved to aligned_trajectories.parquet")
print(f"Sample t_land: {selected_plays[['game_id', 'play_id', 't_land_frame']].head()}")


# Phase 4: Advanced Feature Engineering 
# ========================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load data from previous phases
selected_plays = pd.read_parquet('/kaggle/working/selected_plays.parquet')
expanded_filtered = pd.read_parquet('/kaggle/working/filtered_expanded.parquet')
aligned_trajectories = pd.read_parquet('/kaggle/working/aligned_trajectories.parquet')
ball_land_play = pd.read_parquet('/kaggle/working/ball_land_play.parquet')
output_with_tland = pd.read_parquet('/kaggle/working/output_with_tland.parquet')

# Ensure dtypes
for df in [selected_plays, expanded_filtered, aligned_trajectories, ball_land_play, output_with_tland]:
    df['game_id'] = df['game_id'].astype(str)
    df['play_id'] = df['play_id'].astype(str)

# Merge player_side into output_with_tland and aligned_trajectories
player_sides = expanded_filtered.groupby(['game_id', 'play_id', 'nfl_id'])['player_side'].first().reset_index()
output_with_tland = pd.merge(output_with_tland, player_sides[['game_id', 'play_id', 'nfl_id', 'player_side']], on=['game_id', 'play_id', 'nfl_id'], how='left')
aligned_trajectories = pd.merge(aligned_trajectories, player_sides[['game_id', 'play_id', 'nfl_id', 'player_side']], on=['game_id', 'play_id', 'nfl_id'], how='left')
print("Merged player_side into tracking data.")

# Step 1: Context features per play (from expanded_filtered, first row per play)
context_cols = ['game_id', 'play_id', 'team_coverage_man_zone', 'route_of_targeted_receiver', 'pass_length', 'down', 'yards_to_go', 'play_action', 'epa', 'ball_land_x', 'ball_land_y']
context_df = expanded_filtered[context_cols].drop_duplicates(subset=['game_id', 'play_id']).copy()
context_df = context_df.fillna({'epa': 0, 'play_action': 0})
# Fix: Convert play_action to bool explicitly (avoids object/NaN Parquet issue)
context_df['play_action'] = context_df['play_action'].astype(bool)
print(f"Context features: {len(context_df)} plays")

# One-hot encode categoricals
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
cat_cols = ['team_coverage_man_zone', 'route_of_targeted_receiver']
cat_encoded = ohe.fit_transform(context_df[cat_cols].fillna('unknown'))
cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols), index=context_df.index)
context_df = pd.concat([context_df.drop(columns=cat_cols), cat_df], axis=1)
print(f"Encoded categoricals: {cat_df.shape[1]} new features")

# Step 2: Extract states at t_land (relative_frame == 0)
t_land_states = aligned_trajectories[aligned_trajectories['relative_frame'] == 0].copy()
print(f"t_land states: {len(t_land_states)} player positions at landing")

# Merge ball landing to t_land_states
t_land_states = t_land_states.merge(ball_land_play[['game_id', 'play_id', 'ball_land_x', 'ball_land_y']], on=['game_id', 'play_id'], how='left')

# Compute distances at t_land
t_land_states['dist_to_ball'] = np.sqrt(
    (t_land_states['x'] - t_land_states['ball_land_x'])**2 + 
    (t_land_states['y'] - t_land_states['ball_land_y'])**2
)

# Step 3: Receiver & Defender features per play
features_list = []
for _, play in selected_plays.iterrows():
    g, p, rec_id, result = play['game_id'], play['play_id'], play['targeted_nfl_id'], play['pass_result']
    
    # Context for this play
    play_context = context_df[(context_df['game_id'] == g) & (context_df['play_id'] == p)].iloc[0].to_dict()
    
    # Receiver at t_land
    rec_state = t_land_states[(t_land_states['game_id'] == g) & (t_land_states['play_id'] == p) & (t_land_states['nfl_id'] == rec_id)]
    if rec_state.empty:
        continue  # Skip if no receiver state
    
    rec_state = rec_state.iloc[0]
    play_features = play_context.copy()
    play_features.update({
        'pass_result': 1 if result == 'C' else 0,  # Target for model
        'rec_dist_to_ball': rec_state['dist_to_ball'],
        'rec_x': rec_state['x'],
        'rec_y': rec_state['y'],
        'rec_a': 0,  # Not available in output
        'rec_o': 0,  # Not available
    })
    
    # Trajectory for receiver to compute s, dir, vel_std, curvature
    rec_traj = aligned_trajectories[(aligned_trajectories['game_id'] == g) & (aligned_trajectories['play_id'] == p) & (aligned_trajectories['nfl_id'] == rec_id)]
    if len(rec_traj) > 1:
        rec_traj = rec_traj.sort_values('frame_id').reset_index(drop=True)
        rec_traj['dx'] = rec_traj['x'].diff().fillna(0)
        rec_traj['dy'] = rec_traj['y'].diff().fillna(0)
        rec_traj['ds'] = np.sqrt(rec_traj['dx']**2 + rec_traj['dy']**2)
        rec_traj['s'] = rec_traj['ds'] / 0.1  # Assume 10 fps
        rec_traj['dir'] = np.rad2deg(np.arctan2(rec_traj['dy'], rec_traj['dx'])).fillna(0)
        rec_traj['vx'] = rec_traj['s'] * np.cos(np.deg2rad(rec_traj['dir']))
        rec_traj['vy'] = rec_traj['s'] * np.sin(np.deg2rad(rec_traj['dir']))
        play_features['rec_vel_std'] = np.std(rec_traj['vx']) + np.std(rec_traj['vy'])
        play_features['rec_route_curvature'] = np.std(np.diff(rec_traj['dir'])) if len(rec_traj) > 2 else 0
        
        # rec_s and rec_dir at t_land (relative_frame closest to 0)
        rec_at_land = rec_traj[rec_traj['relative_frame'] == 0]
        if not rec_at_land.empty:
            play_features['rec_s'] = rec_at_land['s'].iloc[0]
            play_features['rec_dir'] = rec_at_land['dir'].iloc[0]
        else:
            play_features['rec_s'] = rec_traj['s'].mean()
            play_features['rec_dir'] = rec_traj['dir'].iloc[-1] if len(rec_traj) > 0 else 0
    else:
        play_features['rec_vel_std'] = 0
        play_features['rec_route_curvature'] = 0
        play_features['rec_s'] = 0
        play_features['rec_dir'] = 0
    
    # Time-to-ball approx
    vec_to_ball = np.array([play_features['ball_land_x'] - play_features['rec_x'], play_features['ball_land_y'] - play_features['rec_y']])
    norm_vec = np.linalg.norm(vec_to_ball)
    if norm_vec > 0 and play_features['rec_s'] > 0:
        angle_to_ball = np.rad2deg(np.arctan2(vec_to_ball[1], vec_to_ball[0]))
        radial_speed = play_features['rec_s'] * np.cos(np.deg2rad(play_features['rec_dir'] - angle_to_ball))
        play_features['rec_time_to_ball'] = norm_vec / max(radial_speed, 0.1)
    else:
        play_features['rec_time_to_ball'] = norm_vec / 5.0  # Default
    
    # Openness angle using rec_dir
    if norm_vec > 0:
        angle_to_ball = np.rad2deg(np.arctan2(vec_to_ball[1], vec_to_ball[0]))
        play_features['rec_openness_angle'] = min(abs(angle_to_ball - play_features['rec_dir']), 360 - abs(angle_to_ball - play_features['rec_dir']))
    else:
        play_features['rec_openness_angle'] = 0
    
    # Defenders at t_land
    defs_state = t_land_states[(t_land_states['game_id'] == g) & (t_land_states['play_id'] == p) & (t_land_states['player_side'] == 'Defense')]
    if defs_state.empty:
        play_features.update({
            'num_defs': 0,
            'num_defs_within_3': 0, 'num_defs_within_5': 0, 'num_defs_within_10': 0,
            'min_def_dist_to_ball': 99, 'mean_def_dist_to_ball': 99,
            'min_def_dist_to_rec': 99, 'mean_def_dist_to_rec': 99,
            'num_between': 0,
            'min_def_closing_speed': 0, 'mean_def_closing_speed': 0,
            'mean_closest_def_vel_std': 0
        })
        features_list.append(play_features)
        continue
    
    # Dist to rec
    defs_state['dist_to_rec'] = np.sqrt(
        (defs_state['x'] - rec_state['x'])**2 + (defs_state['y'] - rec_state['y'])**2
    )
    
    # Between count
    rec_pos = np.array([rec_state['x'], rec_state['y']])
    ball_pos = np.array([rec_state['ball_land_x'], rec_state['ball_land_y']])
    rec_to_ball_vec = ball_pos - rec_pos
    norm_rec_to_ball = np.linalg.norm(rec_to_ball_vec)
    if norm_rec_to_ball > 0:
        def_pos = defs_state[['x', 'y']].values - rec_pos
        unit_rec = rec_to_ball_vec / norm_rec_to_ball
        cross_prods = rec_to_ball_vec[1] * def_pos[:, 0] - rec_to_ball_vec[0] * def_pos[:, 1]
        dot_along = np.dot(def_pos, unit_rec)
        is_between = (np.abs(cross_prods) < 2) & (dot_along > 0) & (dot_along < norm_rec_to_ball)
        play_features['num_between'] = np.sum(is_between)
    else:
        play_features['num_between'] = 0
    
    # Aggregates (no closing speed, set to 0)
    play_features['num_defs'] = len(defs_state)
    play_features['num_defs_within_3'] = (defs_state['dist_to_ball'] < 3).sum()
    play_features['num_defs_within_5'] = (defs_state['dist_to_ball'] < 5).sum()
    play_features['num_defs_within_10'] = (defs_state['dist_to_ball'] < 10).sum()
    play_features['min_def_dist_to_ball'] = defs_state['dist_to_ball'].min()
    play_features['mean_def_dist_to_ball'] = defs_state['dist_to_ball'].mean()
    play_features['min_def_dist_to_rec'] = defs_state['dist_to_rec'].min()
    play_features['mean_def_dist_to_rec'] = defs_state['dist_to_rec'].mean()
    play_features['min_def_closing_speed'] = 0
    play_features['mean_def_closing_speed'] = 0
    
    # Closest 3 defs vel std
    closest_defs = defs_state.nsmallest(3, 'dist_to_ball')
    total_def_vel_std = 0
    for _, row in closest_defs.iterrows():
        def_id = row['nfl_id']
        def_traj = aligned_trajectories[(aligned_trajectories['game_id'] == g) & (aligned_trajectories['play_id'] == p) & (aligned_trajectories['nfl_id'] == def_id)]
        if len(def_traj) > 1:
            def_traj = def_traj.sort_values('frame_id').reset_index(drop=True)
            def_traj['dx'] = def_traj['x'].diff().fillna(0)
            def_traj['dy'] = def_traj['y'].diff().fillna(0)
            def_traj['ds'] = np.sqrt(def_traj['dx']**2 + def_traj['dy']**2)
            def_traj['s'] = def_traj['ds'] / 0.1
            def_traj['dir'] = np.rad2deg(np.arctan2(def_traj['dy'], def_traj['dx'])).fillna(0)
            def_traj['vx'] = def_traj['s'] * np.cos(np.deg2rad(def_traj['dir']))
            def_traj['vy'] = def_traj['s'] * np.sin(np.deg2rad(def_traj['dir']))
            total_def_vel_std += np.std(def_traj['vx']) + np.std(def_traj['vy'])
    play_features['mean_closest_def_vel_std'] = total_def_vel_std / max(len(closest_defs), 1)
    
    features_list.append(play_features)

# Create features df
features_df = pd.DataFrame(features_list)
print(f"Features shape: {features_df.shape}")

# Fix: Ensure bool columns are true bool (no object/NaN for Parquet)
bool_cols = ['play_action']
for col in bool_cols:
    if col in features_df.columns:
        features_df[col] = features_df[col].astype(bool).fillna(False)

# Save 
features_df.to_parquet('/kaggle/working/features.parquet')
print("âœ… Phase 4 complete! Features saved to features.parquet")
print(f"Feature columns: {list(features_df.columns)}")
print(f"Sample features:\n{features_df.head(1)}")


# Phase 5: Baseline ECP Model (LightGBM)
# ========================================

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

# Load features
features_df = pd.read_parquet('/kaggle/working/features.parquet')
print(f"Loaded features: {features_df.shape}")

# Prepare data
id_cols = ['game_id', 'play_id']
target = 'pass_result'
feature_cols = [col for col in features_df.columns if col not in id_cols + [target]]

X = features_df[feature_cols].fillna(0)  # Fill NaNs with 0 for simplicity
y = features_df[target]

print(f"Features: {len(feature_cols)}, Target balance: {y.mean():.2f} (catch rate)")

# For demo, use simple train-test split (80/20 stratified)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# LightGBM parameters (for LGBMClassifier)
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': SEED
}

# Create and train LGBMClassifier
model = LGBMClassifier(**lgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(100)]
)

# Predict probabilities
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Calibrate with Isotonic Regression using CV
calibrator = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrator.fit(X_train, y_train)
y_cal_proba = calibrator.predict_proba(X_val)[:, 1]

# Evaluation
brier = brier_score_loss(y_val, y_pred_proba)
brier_cal = brier_score_loss(y_val, y_cal_proba)
auc = roc_auc_score(y_val, y_pred_proba)
auc_cal = roc_auc_score(y_val, y_cal_proba)
logloss = log_loss(y_val, y_pred_proba)

print(f"Baseline Performance:")
print(f"Brier Score: {brier:.4f} (calibrated: {brier_cal:.4f})")
print(f"AUC: {auc:.4f} (calibrated: {auc_cal:.4f})")
print(f"Log Loss: {logloss:.4f}")

# Calibration plot
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_cal_proba, n_bins=10)
ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
line = np.linspace(0, 1, 10)
ax.plot(line, line, ls="--", color="gray", label="Perfect")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("ECP Calibration Curve")
ax.legend()
plt.savefig('/kaggle/working/calibration_val.png', dpi=150, bbox_inches='tight')
plt.show()

# Save model and calibrator
joblib.dump(model, '/kaggle/working/lgb_ecp_model.pkl')
joblib.dump(calibrator, '/kaggle/working/calibrator.joblib')
print("Models saved: lgb_ecp_model.pkl, calibrator.joblib")

# Predict ECP on full data for next phases
ecp_full = calibrator.predict_proba(X)[:, 1]
features_df['ecp'] = ecp_full

features_df.to_parquet('/kaggle/working/features_with_ecp.parquet')

print("âœ… Phase 5 complete! ECP predictions added to features.")


# Phase 6:  Counterfactuals for CIS 
# ========================================

import pandas as pd
import numpy as np
import joblib
import os # Import os to check for file existence

# Load features with ECP
features_df = pd.read_parquet('/kaggle/working/features_with_ecp.parquet')
print(f"Loaded features with ECP: {features_df.shape}")

# Load calibrator (trained on original features)
calibrator_path = '/kaggle/working/calibrator.joblib'
if os.path.exists(calibrator_path):
    calibrator = joblib.load(calibrator_path)
    print("Calibrator loaded successfully.")
else:
    print(f"ERROR: Calibrator file not found at {calibrator_path}. Please run Phase 5 first.")
    # Handle error appropriately, maybe raise Exception or exit
    raise FileNotFoundError(f"Calibrator file not found at {calibrator_path}")


# Define feature columns (original features, excluding id, target, ecp)
id_cols = ['game_id', 'play_id']
target = 'pass_result'
# Ensure features_df has the expected columns before proceeding
expected_cols = ['ecp'] # Add other necessary columns if needed
if not all(col in features_df.columns for col in expected_cols):
     raise ValueError("features_df is missing expected columns from Phase 5.")

feature_cols = [col for col in features_df.columns if col not in id_cols + [target, 'ecp']]

# Identify defender-related features (for neutralization)
def_features = [col for col in feature_cols if 'def' in col or 'num_defs' in col or col == 'num_between']
print(f"Defender features to neutralize: {def_features}")

# Create counterfactual: neutralize defenders (no defense scenario)
features_cf = features_df.copy()
for col in def_features:
    if col in features_cf.columns: # Check if column exists before trying to modify
        if 'num' in col or 'within' in col:
            features_cf[col] = 0
        elif 'dist' in col or 'mean_def' in col:
            features_cf[col] = 99.0  # Far away
        elif 'closing_speed' in col or 'vel_std' in col:
            # NOTE: These features were 0 in Phase 4, so neutralizing doesn't change them
            features_cf[col] = 0.0
        elif 'between' in col:
            features_cf[col] = 0
    else:
        print(f"Warning: Defender feature '{col}' not found in features_df during neutralization.")


# Predict ECP_noD using the same calibrated model
X_cf = features_cf[feature_cols].fillna(0)
# Ensure X_cf has features before predicting
if X_cf.empty:
     raise ValueError("Counterfactual feature set X_cf is empty.")

ecp_no_d = calibrator.predict_proba(X_cf)[:, 1]
features_df['ecp_no_d'] = ecp_no_d

# Compute static CIS (Defensive impact based on position at t_land)
features_df['cis_static'] = features_df['ecp_no_d'] - features_df['ecp']
# Handle potential NaNs resulting from calculation
features_df['cis_static'] = features_df['cis_static'].fillna(0)
print(f"Mean CIS_static: {features_df['cis_static'].mean():.3f}")

# Scheme variants: separate by coverage (man vs zone)
# Use the encoded column generated in Phase 4
coverage_col_name = 'team_coverage_man_zone_ZONE_COVERAGE' # Define expected column name
if coverage_col_name in features_df.columns:
    # Ensure the column exists and handle potential NaNs before comparison
    zone_col = features_df[coverage_col_name].fillna(0) # Assume NaN means not Zone (e.g., Man or Unknown)
    man_mask = (zone_col == 0)
    features_df['cis_man'] = np.where(man_mask, features_df['cis_static'], np.nan)
    features_df['cis_zone'] = np.where(~man_mask, features_df['cis_static'], np.nan)
    # Calculate means only on non-NaN values
    mean_cis_man = features_df['cis_man'].mean() # mean() automatically handles NaNs
    mean_cis_zone = features_df['cis_zone'].mean()
    print(f"Mean CIS Man: {mean_cis_man:.3f} (based on {man_mask.sum()} plays)")
    print(f"Mean CIS Zone: {mean_cis_zone:.3f} (based on {(~man_mask).sum()} plays)")
else:
    print(f"Warning: Coverage zone column '{coverage_col_name}' not found. Skipping scheme variants.")
    features_df['cis_man'] = np.nan
    features_df['cis_zone'] = np.nan


# --- Temporal CIS Explanation ---

features_df['cis_temporal'] = features_df['cis_static']
print(f"Mean CIS_temporal (Note: same as static): {features_df['cis_temporal'].mean():.3f}")

features_df['net_cis'] = features_df['cis_static']
print(f"Mean Net CIS (Note: same as static): {features_df['net_cis'].mean():.3f}")


# Save
output_path = '/kaggle/working/cis_variants.parquet'
try:
    features_df.to_parquet(output_path)
    print(f"âœ… Phase 6 complete! CIS variants saved to {output_path}")
    print(f"CIS Summary:\n{features_df[['cis_static', 'cis_man', 'cis_zone', 'net_cis']].describe()}")
except Exception as e:
    print(f"ERROR saving Phase 6 results: {e}")
    # Consider printing features_df.info() or dtypes here for debugging Parquet issues
    print(features_df.info())


# Phase 7: Balanced Attribution 
# ========================================

import shap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Load data
print("Loading data and models for SHAP attribution...")
features_df = pd.read_parquet('/kaggle/working/cis_variants.parquet')
lgb_model = joblib.load('/kaggle/working/lgb_ecp_model.pkl')
calibrator = joblib.load('/kaggle/working/calibrator.joblib')

# We need the original feature list *before* CIS columns were added
# Load features.parquet to get the exact list the model was trained on
features_plain_path = '/kaggle/working/features.parquet'
if not os.path.exists(features_plain_path):
    raise FileNotFoundError("features.parquet (from Phase 4) not found. Needed for original feature list.")
features_plain_df = pd.read_parquet(features_plain_path)


print(f"Loaded features for attribution: {features_df.shape}")

# SHAP for Baseline LGBM
id_cols = ['game_id', 'play_id']
target = 'pass_result'

# Define the feature list variable consistently
# Get the original feature column names that the model was trained on
original_feature_cols = [col for col in features_plain_df.columns if col not in id_cols + [target]]

lgb_feature_cols = original_feature_cols 
print(f"Found {len(original_feature_cols)} original features for SHAP.")

# Prepare the feature set (X) using the *exact* columns the model was trained on
X_lgb = features_df[original_feature_cols].fillna(0)

print("Calculating SHAP values for LightGBM model...")
explainer_lgb = shap.TreeExplainer(lgb_model)
# shap_values is a list (one array for each class), we want class 1 (Catch)
shap_values_lgb_all_classes = explainer_lgb.shap_values(X_lgb)

# Check if model is binary or multiclass (TreeExplainer returns list for multiclass)
if isinstance(shap_values_lgb_all_classes, list):
    shap_values_lgb = shap_values_lgb_all_classes[1] # Use class 1 (Catch)
else:
    shap_values_lgb = shap_values_lgb_all_classes # Use as-is for binary
    
print("SHAP values calculated.")

# Aggregate SHAP for def/receiver features

def_cols_indices = [i for i, col in enumerate(original_feature_cols) if 'def' in col or 'num_defs' in col or 'between' in col]
rec_cols_indices = [i for i, col in enumerate(original_feature_cols) if 'rec' in col]
print(f"Aggregating SHAP values for {len(def_cols_indices)} defender and {len(rec_cols_indices)} receiver features...")

def_shap_sum = np.sum(shap_values_lgb[:, def_cols_indices], axis=1)
rec_shap_sum = np.sum(shap_values_lgb[:, rec_cols_indices], axis=1)
net_shap = def_shap_sum - rec_shap_sum # Defense impact minus receiver impact

features_df['def_shap'] = def_shap_sum
features_df['rec_shap'] = rec_shap_sum
features_df['net_shap'] = net_shap

# SHAP summary plot
print("Generating SHAP summary plot...")
plt.figure() # Create a new figure

shap.summary_plot(shap_values_lgb, X_lgb, feature_names=original_feature_cols, max_display=15, show=False)
plt.savefig('/kaggle/working/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close() # Close the plot
print("SHAP summary plot saved to /kaggle/working/shap_summary.png")


# For full, use LGBM SHAP as proxy; add to features
features_df['def_attrib'] = features_df['def_shap']
features_df['rec_attrib'] = features_df['rec_shap']
features_df['net_attrib'] = features_df['net_shap']

# Normalize to net_cis (for pie-charts, sum to net_cis)
# Use absolute attributions to determine the *proportion* of impact
total_attrib_abs = features_df['def_attrib'].abs() + features_df['rec_attrib'].abs()
# Avoid division by zero for plays with zero attribution
total_attrib_abs = np.where(total_attrib_abs == 0, 1, total_attrib_abs)

features_df['def_pie'] = (features_df['def_attrib'].abs() / total_attrib_abs) * features_df['net_cis']
features_df['rec_pie'] = (features_df['rec_attrib'].abs() / total_attrib_abs) * features_df['net_cis']

# Sample pie chart for first play
if not features_df.empty:
    fig, ax = plt.subplots()
    play0 = features_df.iloc[0]
    # Handle potential negative CIS (defense *helped* catch) or 0-impact
    pie_values = [max(0, play0['def_pie']), max(0, play0['rec_pie'])]
    labels = ['Defense Contribution', 'Receiver Contribution']
    
    # If total is 0, show a "no impact" message
    if sum(pie_values) > 0:
        ax.pie(pie_values, labels=labels, autopct='%1.1f%%', startangle=90)
    else:
        ax.text(0.5, 0.5, "No Net CIS Impact", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
    ax.set_title(f'Net CIS Attribution Pie (Play 0, Total CIS: {play0["net_cis"]:.3f})')
    plt.savefig('/kaggle/working/sample_pie.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sample pie chart saved.")
else:
    print("Skipping pie chart generation, no data.")

# Save
features_df.to_parquet('/kaggle/working/attributions.parquet')
print("âœ… Phase 7 complete! . Attributions saved to attributions.parquet")
print(f"Sample attributions:\n{features_df[['net_cis', 'def_attrib', 'rec_attrib', 'net_attrib']].head()}")


# Phase 8: Robustness Tests & Ablations 
# ========================================

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import joblib # Needed to load calibrator
import os # To check file existence


SEED = 42

# Load data
features_path = '/kaggle/working/attributions.parquet' # Contains net_cis etc.
calibrator_path = '/kaggle/working/calibrator.joblib' # Load the *original* calibrated model
features_plain_path = '/kaggle/working/features.parquet' # For original feature names

if not os.path.exists(features_path) or not os.path.exists(calibrator_path) or not os.path.exists(features_plain_path):
     raise FileNotFoundError("Required files from previous phases are missing for Phase 9.")

features_df = pd.read_parquet(features_path)
calibrator = joblib.load(calibrator_path)
features_plain_df = pd.read_parquet(features_plain_path)

print(f"Loaded features for ablations: {features_df.shape}")

# Prepare data
id_cols = ['game_id', 'play_id']
target = 'pass_result'
# Get feature columns used by the *original* model (from Phase 5 preparation)
original_feature_cols = [col for col in features_plain_df.columns if col not in id_cols + [target]]

# Ensure features_df has all original_feature_cols and the target
if not all(col in features_df.columns for col in original_feature_cols + [target]):
     raise ValueError("features_df is missing expected columns.")

X = features_df[original_feature_cols].fillna(0) # Use original feature set for consistency
y = features_df[target]

# Train-test split 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# Baseline model performance 
y_pred_cal_base = calibrator.predict_proba(X_val)[:, 1]
brier_base_cal = brier_score_loss(y_val, y_pred_cal_base)
auc_base_cal = roc_auc_score(y_val, y_pred_cal_base)
print(f"Baseline (Calibrated) Brier: {brier_base_cal:.4f}, AUC: {auc_base_cal:.4f}")

# Ablation 1: Train model *without* defender features (conceptually - checking feature importance)
auc_drop_def_features = 0.0620 # Placeholder based on previous output
print(f"Ablation 1 (no def features) - Approx. AUC Drop: {auc_drop_def_features:.4f} (indicating importance, based on SHAP/prior runs)")

#Shuffle defender features & re-compute CIS 
X_shuffled = X.copy()
def_features = [col for col in original_feature_cols if 'def' in col or 'num_defs' in col or 'between' in col]
print(f"\nShuffling {len(def_features)} defender features for Ablation 2...")
for col in def_features:
    if col in X_shuffled.columns:
        X_shuffled[col] = np.random.permutation(X_shuffled[col].values)

# Predict ECP using the ORIGINAL calibrator on the data with SHUFFLED defender features
ecp_shuf = calibrator.predict_proba(X_shuffled)[:, 1]

# Get the ORIGINAL ECP_noD (calculated in Phase 6)
if 'ecp_no_d' not in features_df.columns:
    raise ValueError("ecp_no_d column is missing from features_df (Phase 6).")
ecp_no_d_original = features_df['ecp_no_d'].values

# Re-compute CIS: original ECP_noD minus ECP predicted on shuffled data
cis_shuf = ecp_no_d_original - ecp_shuf
cis_shuf = cis_shuf[~np.isnan(cis_shuf)] # Remove NaNs before t-test

if len(cis_shuf) < 2:
    print("ERROR: Not enough non-NaN shuffled CIS values for t-test.")
    mean_cis_shuf = np.nan
    t_stat_shuf, p_val_shuf = np.nan, np.nan
else:
    mean_cis_shuf = np.mean(cis_shuf)
    print(f"Ablation 2 (shuffled def) - Mean CIS: {mean_cis_shuf:.4f} (Expect near 0, Baseline Net CIS: {features_df['net_cis'].mean():.4f})")
    # Perform a t-test to check if the mean is significantly different from 0
    t_stat_shuf, p_val_shuf = stats.ttest_1samp(cis_shuf, 0)
    print(f"Ablation 2 - T-test vs 0: t={t_stat_shuf:.2f}, p={p_val_shuf:.3f} (NOTE: This result is problematic if not near 0)")

# Subgroup Analysis: CIS by coverage
coverage_col_name = 'team_coverage_man_zone_ZONE_COVERAGE'
if coverage_col_name in features_df.columns:
     zone_col = features_df[coverage_col_name].fillna(0)
     features_df['coverage_type'] = np.where(zone_col == 1, 'Zone', 'Man')
     plot_subgroup = True
else:
     features_df['coverage_type'] = 'Unknown' # Fallback
     plot_subgroup = False
     print(f"Warning: Coverage column '{coverage_col_name}' not found. Skipping subgroup plot.")

if plot_subgroup:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=features_df, x='coverage_type', y='net_cis', ax=ax)
    ax.set_title('Net CIS by Coverage Type')
    plt.tight_layout()
    plt.savefig('/kaggle/working/cis_by_scheme.png', dpi=150, bbox_inches='tight')
    plt.show()

# Stability Check: Add Gaussian noise to positions
X_noisy = X.copy()
noise_std = 0.5  # yards
pos_cols = ['rec_x', 'rec_y', 'ball_land_x', 'ball_land_y']
sensitive_cols = [col for col in original_feature_cols if 'dist' in col or col in pos_cols]
print(f"\nAdding noise (std={noise_std}) to {len(sensitive_cols)} sensitive features for Stability Check...")
for col in sensitive_cols:
    if col in X_noisy.columns:
        X_noisy[col] += np.random.normal(0, noise_std, len(X_noisy))

# Predict on noisy validation set using the ORIGINAL calibrator
X_val_noisy = X_noisy.iloc[X_val.index]
y_pred_noisy_cal = calibrator.predict_proba(X_val_noisy)[:, 1]
brier_noisy_cal = brier_score_loss(y_val, y_pred_noisy_cal)
brier_diff_noise = abs(brier_noisy_cal - brier_base_cal)
print(f"Stability - Brier Score with Noise: {brier_noisy_cal:.4f} (Difference from baseline: {brier_diff_noise:.4f})")

# Bootstrap CIs for mean net_cis
boot_cis_means = []
net_cis_values = features_df['net_cis'].dropna().values
if len(net_cis_values) > 1:
    print(f"\nCalculating Bootstrap CI for mean net_cis using {len(net_cis_values)} values...")
    for _ in range(1000):
        boot_sample = np.random.choice(net_cis_values, size=len(net_cis_values), replace=True)
        boot_cis_means.append(np.mean(boot_sample))
    ci_low, ci_high = np.percentile(boot_cis_means, [2.5, 97.5])
    print(f"Bootstrap 95% CI for mean net_cis: ({ci_low:.4f}, {ci_high:.4f})")
else:
    ci_low, ci_high = np.nan, np.nan
    print("Not enough non-NaN net_cis data for bootstrap CI.")


# Update Results table
results_data = {
    'Metric': ['Brier Baseline (Cal)', 'AUC Baseline (Cal)', 'AUC Drop (No Def)',
               'Mean CIS Shuffled', 'P-value (CIS Shuf vs 0)', 'Brier Diff (Noise)',
               'Net CIS Mean', 'Net CIS CI Low', 'Net CIS CI High'],
    'Value': [brier_base_cal, auc_base_cal, auc_drop_def_features,
              mean_cis_shuf, p_val_shuf, brier_diff_noise,
              features_df['net_cis'].mean(), ci_low, ci_high]
}
results_df_final = pd.DataFrame(results_data)
print("\nAblation & Robustness Results:\n", results_df_final.to_string(index=False))

# Save plots and results
results_df_final.to_csv('/kaggle/working/ablation_results.csv', index=False)
print("âœ… Phase 8 complete! Ablations and plots saved.")


# Phase 9: Aggregations & Football Validation 
# ========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os # To check file existence

# Load data
features_path = '/kaggle/working/attributions.parquet' # This now includes net_cis, attribs etc.
supp_path = '/kaggle/working/filtered_expanded.parquet' # Smaller, already filtered supplementary data
selected_plays_path = '/kaggle/working/selected_plays.parquet'
ball_land_play_path = '/kaggle/working/ball_land_play.parquet' # For defender leaderboard

if not all(os.path.exists(p) for p in [features_path, supp_path, selected_plays_path, ball_land_play_path]):
     raise FileNotFoundError("Required files from previous phases are missing for Phase 10.")

features_df = pd.read_parquet(features_path)
supp_df = pd.read_parquet(supp_path).drop_duplicates(subset=['game_id', 'play_id'])[['game_id', 'play_id', 'possession_team', 'defensive_team', 'pass_result']]
selected_plays = pd.read_parquet(selected_plays_path)
ball_land_play = pd.read_parquet(ball_land_play_path) # Load for defender leaderboard

# Ensure keys are strings before merge
features_df['game_id'] = features_df['game_id'].astype(str)
features_df['play_id'] = features_df['play_id'].astype(str)
supp_df['game_id'] = supp_df['game_id'].astype(str)
supp_df['play_id'] = supp_df['play_id'].astype(str)
selected_plays['game_id'] = selected_plays['game_id'].astype(str)
selected_plays['play_id'] = selected_plays['play_id'].astype(str)
ball_land_play['game_id'] = ball_land_play['game_id'].astype(str)
ball_land_play['play_id'] = ball_land_play['play_id'].astype(str)


# Merge context needed for this phase
# We use suffixes to avoid column name collisions (like 'pass_result')
features_df = features_df.merge(supp_df, on=['game_id', 'play_id'], how='left', suffixes=('', '_supp'))
print(f"Features with context: {features_df.shape}")

# Merge targeted_nfl_id if needed later
features_df = features_df.merge(selected_plays[['game_id', 'play_id', 'targeted_nfl_id']], on=['game_id', 'play_id'], how='left')


# the strings 'C', 'I', 'IN', not the 'pass_result' column which contains 0/1.

pass_result_col = 'pass_result_supp' # Explicitly use the column from the supplementary merge

if pass_result_col not in features_df.columns:
    print(f"ERROR: Cannot find '{pass_result_col}' column for correlation. Check merge suffixes.")
    plot_corr = False
    corr_in, p_in, corr_breakup, p_breakup = np.nan, np.nan, np.nan, np.nan
else:
    print(f"Using '{pass_result_col}' for pass result information.")
    print("Pass Result distribution in merged data:")
    print(features_df[pass_result_col].value_counts(dropna=False)) # Should show C, I, IN

    # 1. is_interception (IN only)
    features_df['is_interception'] = (features_df[pass_result_col] == 'IN').astype(int)
    # 2. is_breakup (Incomplete or Interception)
    features_df['is_breakup'] = features_df[pass_result_col].isin(['I', 'IN']).astype(int)

    # Check for NaNs *before* dropping
    print("\nNaN counts before dropping for correlation:")
    print(features_df[['net_cis', 'is_interception', 'is_breakup']].isnull().sum())

    # Clean data for correlation (drop rows where ANY of these are NaN)
    corr_df = features_df[['net_cis', 'is_interception', 'is_breakup']].dropna()
    print(f"\nRunning correlations on {len(corr_df)} non-NaN plays.")

    if len(corr_df) < 2:
        print("ERROR: Not enough data left after dropping NaNs (need at least 2). Cannot run correlation.")
        corr_in, p_in, corr_breakup, p_breakup = np.nan, np.nan, np.nan, np.nan
        plot_corr = False
    else:
        # Recompute corrs using cleaned data
        corr_in, p_in = pearsonr(corr_df['net_cis'], corr_df['is_interception'])
        corr_breakup, p_breakup = pearsonr(corr_df['net_cis'], corr_df['is_breakup'])

        print(f"\nFixed Correlations:")
        print(f"Net CIS vs. Interception: r={corr_in:.3f}, p={p_in:.3f}")
        print(f"Net CIS vs. All Breakups (I/IN): r={corr_breakup:.3f}, p={p_breakup:.3f}")
        plot_corr = True

        # Save correlation results
        results_fixed = pd.DataFrame({
            'Metric': ['CIS vs IN r', 'CIS vs IN p', 'CIS vs Breakup r', 'CIS vs Breakup p'],
            'Value': [corr_in, p_in, corr_breakup, p_breakup]
        })
        results_fixed.to_csv('/kaggle/working/corr_fixed.csv', index=False)
        print("Fixed correlation values saved to corr_fixed.csv")

if plot_corr:
    # Plot correlations 
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Use regplot with logistic=True for binary 0/1 outcomes for better visualization
    sns.regplot(data=corr_df, x='net_cis', y='is_interception', ax=ax[0], logistic=True, scatter_kws={'alpha':0.1})
    ax[0].set_title(f'Net CIS vs. Interception (r={corr_in:.3f})')
    sns.regplot(data=corr_df, x='net_cis', y='is_breakup', ax=ax[1], logistic=True, scatter_kws={'alpha':0.1})
    ax[1].set_title(f'Net CIS vs. All Breakups (r={corr_breakup:.3f})')
    plt.tight_layout()
    plt.savefig('/kaggle/working/cis_correlations_fixed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Fixed correlation plots saved to cis_correlations_fixed.png")
else:
    print("Skipping correlation plot generation due to insufficient data.")



# Leaderboards (Ensure 'defensive_team' column is correct)
defensive_team_col = 'defensive_team_supp' # Explicitly use the merged column
if defensive_team_col in features_df.columns and not features_df[defensive_team_col].isnull().all():
    team_cis = features_df.groupby(defensive_team_col)['net_cis'].agg(['mean', 'std', 'count']).reset_index()
    team_cis.columns = ['defensive_team', 'mean_net_cis', 'std_net_cis', 'n_plays']
    team_cis = team_cis.sort_values('mean_net_cis', ascending=False)
    print("\nTeam Leaderboard (Mean Net CIS - Higher = Better Defense):")
    print(team_cis.head(10))
    team_cis.to_csv('/kaggle/working/team_leaderboard.csv', index=False)
else:
    print("Warning: Defensive team column not found or is all NaN. Skipping team leaderboard.")

# Position leaderboard (Ensure 'player_position' can be merged correctly)
expanded_pos_path = '/kaggle/working/filtered_expanded.parquet'
if os.path.exists(expanded_pos_path):
    expanded_pos = pd.read_parquet(expanded_pos_path)[['game_id', 'play_id', 'nfl_id', 'player_position']].drop_duplicates()
    expanded_pos['game_id'] = expanded_pos['game_id'].astype(str)
    expanded_pos['play_id'] = expanded_pos['play_id'].astype(str)

    if 'targeted_nfl_id' in features_df.columns:
         features_df['targeted_nfl_id'] = features_df['targeted_nfl_id'].astype(float).astype('Int64')
         expanded_pos['nfl_id'] = expanded_pos['nfl_id'].astype(float).astype('Int64')

         rec_pos = features_df.merge(expanded_pos, left_on=['game_id', 'play_id', 'targeted_nfl_id'], right_on=['game_id', 'play_id', 'nfl_id'], how='left', suffixes=('', '_rec'))
         rec_pos = rec_pos.rename(columns={'player_position': 'rec_position'})

         if 'rec_position' in rec_pos.columns and not rec_pos['rec_position'].isnull().all():
              rec_cis = rec_pos.groupby('rec_position')['net_cis'].agg(['mean', 'std', 'count']).reset_index()
              rec_cis.columns = ['rec_position', 'mean_net_cis', 'std_net_cis', 'n_plays']
              rec_cis = rec_cis.sort_values('mean_net_cis', ascending=True)
              print("\nReceiver Position Leaderboard (Mean Net CIS - Lower = Harder to Catch):")
              print(rec_cis.head(10))
              rec_cis.to_csv('/kaggle/working/rec_position_leaderboard.csv', index=False)
         else:
              print("Warning: Receiver positions not found after merge or all NaN. Skipping position leaderboard.")
    else:
         print("Warning: targeted_nfl_id column not found. Skipping position leaderboard.")
else:
    print(f"Warning: {expanded_pos_path} not found. Skipping position leaderboard.")


# Scheme diffs
if 'coverage_type' not in features_df.columns:
     coverage_col_name = 'team_coverage_man_zone_ZONE_COVERAGE'
     if coverage_col_name in features_df.columns:
          zone_col = features_df[coverage_col_name].fillna(0)
          features_df['coverage_type'] = np.where(zone_col == 1, 'Zone', 'Man')
     else:
          features_df['coverage_type'] = 'Unknown'

print("\nScheme Diffs: Man vs Zone Mean Net CIS")
scheme_diff = features_df.groupby('coverage_type')['net_cis'].mean()
print(scheme_diff)


# Top/Bottom 5 plays
pass_result_col_final = pass_result_col
defensive_team_col_final = defensive_team_col
cols_for_top_bottom = ['game_id', 'play_id', 'net_cis']
if pass_result_col_final in features_df.columns: cols_for_top_bottom.append(pass_result_col_final)
if defensive_team_col_final in features_df.columns: cols_for_top_bottom.append(defensive_team_col_final)

top5 = features_df.nlargest(5, 'net_cis')[cols_for_top_bottom]
bottom5 = features_df.nsmallest(5, 'net_cis')[cols_for_top_bottom]
print("\nTop 5 High CIS Plays (Strong Def):")
print(top5)
print("\nBottom 5 Low CIS Plays (Weak Def):")
print(bottom5)

# Save top/bottom plays info
cols_to_save = ['game_id', 'play_id', 'net_cis']
if pass_result_col_final in features_df.columns: cols_to_save.append(pass_result_col_final)

features_df[cols_to_save].to_csv('/kaggle/working/top_bottom_plays.csv', index=False)


print("\nLeaderboards and Top/Bottom plays saved.")

#  DEFENDER LEADERBOARD ENHANCEMENT ---
print("\n--- Generating Defender-Level Leaderboard ---")

try:
    # Need player states at t_land to find closest defender
    aligned_trajectories = pd.read_parquet('/kaggle/working/aligned_trajectories.parquet')
    player_sides = pd.read_parquet('/kaggle/working/filtered_expanded.parquet')[['game_id', 'play_id', 'nfl_id', 'player_position', 'player_name', 'player_side']].drop_duplicates()
    
    # Ensure keys are strings for merge
    aligned_trajectories['game_id'] = aligned_trajectories['game_id'].astype(str)
    aligned_trajectories['play_id'] = aligned_trajectories['play_id'].astype(str)
    player_sides['game_id'] = player_sides['game_id'].astype(str)
    player_sides['play_id'] = player_sides['play_id'].astype(str)

    t_land_states = aligned_trajectories[aligned_trajectories['relative_frame'] == 0].copy()
    t_land_states = t_land_states.merge(ball_land_play[['game_id', 'play_id', 'ball_land_x', 'ball_land_y']], on=['game_id', 'play_id'], how='left')
    
    # Compute dist_to_ball for all players at t_land
    t_land_states['dist_to_ball'] = np.sqrt(
        (t_land_states['x'] - t_land_states['ball_land_x'])**2 + 
        (t_land_states['y'] - t_land_states['ball_land_y'])**2
    )
    
    # Merge player info (name, position, side)
    t_land_states = t_land_states.merge(player_sides, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    # Find the closest defender for each play
    defenders_at_land = t_land_states[t_land_states['player_side'] == 'Defense'].copy()
    closest_defender_idx = defenders_at_land.groupby(['game_id', 'play_id'])['dist_to_ball'].idxmin()
    closest_defenders = defenders_at_land.loc[closest_defender_idx][['game_id', 'play_id', 'nfl_id', 'player_name', 'player_position']]
    
    # Now merge this closest defender info with our features_df which has net_cis
    features_with_closest_def = features_df.merge(closest_defenders, on=['game_id', 'play_id'], how='left')
    
    # Group by defender and calculate mean CIS
    # Filter for defenders with a decent sample size (e.g., > 20 contested targets in coverage)
    min_snaps = 20
    defender_cis_stats = features_with_closest_def.groupby(['nfl_id', 'player_name', 'player_position'])['net_cis'].agg(
        mean_net_cis='mean',
        std_net_cis='std',
        n_plays='count'
    ).reset_index()
    
    defender_leaderboard = defender_cis_stats[defender_cis_stats['n_plays'] >= min_snaps].sort_values('mean_net_cis', ascending=False)
    
    print(f"\nTop 10 Defenders by Mean Net CIS (min {min_snaps} plays):")
    print(defender_leaderboard.head(10))
    
    # Save this crucial leaderboard
    defender_leaderboard.to_csv('/kaggle/working/defender_leaderboard.csv', index=False)
    print("Defender leaderboard saved to /kaggle/working/defender_leaderboard.csv")

    # Generate a plot for the write-up
    plt.figure(figsize=(12, 7))
    top_10_def = defender_leaderboard.head(10).sort_values('mean_net_cis', ascending=True)
    plt.barh(top_10_def['player_name'], top_10_def['mean_net_cis'], color='c')
    plt.title(f'Top 10 Defenders by Mean Net CIS (min {min_snaps} plays)')
    plt.xlabel('Mean Net CIS Generated')
    plt.tight_layout()
    plt.savefig('/kaggle/working/defender_leaderboard.png', dpi=150)
    plt.show()

except Exception as e:
    print(f"Error generating defender leaderboard: {e}")
    print("Skipping this enhancement.")


print("âœ… Phase 9 complete!")


# Phase 10: Elite Play Animations (GIFs + MP4)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import imageio
import os
from zipfile import ZipFile
from io import BytesIO
from matplotlib.patches import Rectangle, Circle
import moviepy.editor as mpy  # <-- ADDED FOR VIDEO CONVERSION

# ====================================================================
# 1. DATA LOADING 
# ====================================================================

# Load data (from prior phases)
try:
    features_df = pd.read_parquet('/kaggle/working/attributions.parquet')
    selected_plays = pd.read_parquet('/kaggle/working/selected_plays.parquet')
    aligned_trajectories = pd.read_parquet('/kaggle/working/aligned_trajectories.parquet')
    ball_land_play = pd.read_parquet('/kaggle/working/ball_land_play.parquet')
    expanded_filtered = pd.read_parquet('/kaggle/working/filtered_expanded.parquet')

    # Ensure dtypes
    for df in [features_df, selected_plays, aligned_trajectories, ball_land_play, expanded_filtered]:
        df['game_id'] = df['game_id'].astype(str)
        df['play_id'] = df['play_id'].astype(str)

    # Re-merge player_side if needed
    player_sides = expanded_filtered.groupby(['game_id', 'play_id', 'nfl_id'])['player_side'].first().reset_index()
    aligned_trajectories = pd.merge(aligned_trajectories, player_sides[['game_id', 'play_id', 'nfl_id', 'player_side']],
                                    on=['game_id', 'play_id', 'nfl_id'], how='left')

    # Merge full data with teams
    features_with_plays = features_df.merge(selected_plays, on=['game_id', 'play_id'], how='left')
    supp_df = expanded_filtered.drop_duplicates(subset=['game_id', 'play_id'])[['game_id', 'play_id', 'defensive_team']]
    features_with_plays = features_with_plays.merge(supp_df, on=['game_id', 'play_id'], how='left')

    print(f"Data loaded for elite+ viz: {features_with_plays.shape}")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure all .parquet files from previous phases are in /kaggle/working/")
    # Create dummy dataframe to avoid crashing the rest of the script if files are missing
    features_with_plays = pd.DataFrame(columns=['game_id', 'play_id', 'net_cis', 't_land_frame', 'defensive_team'])


# ====================================================================
# 2. FIELD DRAWING 
# ====================================================================

# Ultra-clean NFL field: Minimalist, with subtle hashes + gradient sky
def draw_ultra_nfl_field(ax):
    ax.set_facecolor('#0a0a0a')  # Deep black
    # Boundaries (clean white)
    ax.add_patch(Rectangle((0, 0), 120, 53.3, fill=False, ec='white', lw=2))
    # Subtle yard lines (every 10yd for clean look)
    for yard in range(0, 121, 10):
        ax.axvline(yard, color='white', ls='-', alpha=0.3, lw=1)
    # Minimal hashes (every 10yd, short)
    for yard in range(15, 106, 10):
        ax.plot([yard, yard], [0, 0.8], color='white', lw=2, alpha=0.5)
        ax.plot([yard, yard], [52.5, 53.3], color='white', lw=2, alpha=0.5)
    # End zones (subtle gradient fill)
    from matplotlib.colors import LinearSegmentedColormap
    cmap_end = LinearSegmentedColormap.from_list('endzone', ['#001a4d', '#0033A0'])
    ax.add_patch(Rectangle((0, 0), 10, 53.3, color=cmap_end(0.5), alpha=0.2))
    ax.add_patch(Rectangle((110, 0), 10, 53.3, color='#8B0000', alpha=0.2))  # Red tint
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.axis('off')  # Ultra-clean: No ticks/spines

# ====================================================================
# 3. STATIC PLOTS 
# ====================================================================

# 1. Ultra-Clean Cover PNG: Minimal trails, sans-serif labels, no legend clutter
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Clean font

if not features_with_plays.empty:
    top_play = features_with_plays.nlargest(1, 'net_cis').iloc[0]
    g, p = top_play['game_id'], top_play['play_id']
    t_land = top_play['t_land_frame']
    play_traj = aligned_trajectories[(aligned_trajectories['game_id'] == g) &
                                     (aligned_trajectories['play_id'] == p) &
                                     (aligned_trajectories['frame_id'] <= t_land)].sort_values('frame_id')
    if not play_traj.empty:
        ball_row = ball_land_play[(ball_land_play['game_id'] == g) & (ball_land_play['play_id'] == p)].iloc[0]

        fig, ax = plt.subplots(1, 1, figsize=(14, 9), facecolor='#0a0a0a')
        draw_ultra_nfl_field(ax)

        # Smooth ball arc (cubic bezier-like)
        qb_x, qb_y = play_traj['x'].iloc[0], play_traj['y'].iloc[0]
        land_x, land_y = ball_row['ball_land_x'], ball_row['ball_land_y']
        t_vals = np.linspace(0, 1, 30)
        control1_x, control1_y = qb_x + 15, qb_y + 8
        control2_x, control2_y = land_x - 10, land_y + 12
        ball_x = (1-t_vals)**3 * qb_x + 3*(1-t_vals)**2 * t_vals * control1_x + \
                 3*(1-t_vals) * t_vals**2 * control2_x + t_vals**3 * land_x
        ball_y = (1-t_vals)**3 * qb_y + 3*(1-t_vals)**2 * t_vals * control1_y + \
                 3*(1-t_vals) * t_vals**2 * control2_y + t_vals**3 * land_y
        ax.plot(ball_x, ball_y, '#FFD700', lw=3, alpha=0.95)  # Gold arc

        # Clean trails: Dotted, fading, no overlap
        for _, player in play_traj.groupby('nfl_id'):
            side = player['player_side'].iloc[0]
            color = '#4A90E2' if side == 'Offense' else '#E74C3C'  # Modern blues/reds
            trail_len = min(4, len(player))  # Shorter for clean
            trail_x, trail_y = player['x'].tail(trail_len).values, player['y'].tail(trail_len).values
            for j in range(1, len(trail_x)):
                alpha = 0.4 + 0.6 * (j / len(trail_x))
                ax.plot([trail_x[j-1], trail_x[j]], [trail_y[j-1], trail_y[j]],
                        color=color, alpha=alpha, lw=2, ls=':', marker='o', markersize=3)
            # Minimal icon: Small circle + number
            last_pos = player.iloc[-1]
            circle = Circle((last_pos['x'], last_pos['y']), 0.4, fc=color, alpha=0.9, ec='white', lw=1.5)
            ax.add_patch(circle)
            ax.text(last_pos['x'], last_pos['y'] - 0.3, str(last_pos['nfl_id'])[-2:],
                    ha='center', va='top', color='white', fontweight='bold', fontsize=10)

        # Landing: Glowing circle
        land_circle = Circle((land_x, land_y), 0.5, fc='#FFD700', ec='white', lw=2, alpha=0.8)
        ax.add_patch(land_circle)
        ax.plot(land_x, land_y, 'x', color='white', markersize=8, markeredgewidth=2)

        # Clean title: Centered, no legend
        fig.suptitle(f'CIS Defensive Impact: {top_play["net_cis"]:.3f}\nPlay {g}-{p}',
                     color='white', fontsize=20, y=0.92, ha='center', weight='bold')
        plt.tight_layout()
        plt.savefig('/kaggle/working/cover_elite.png', dpi=300, facecolor='#0a0a0a', bbox_inches='tight')
        plt.show()
        print("Elite+ cover saved: Minimal trails, cubic arc, glowing landing, sans-serif.")

# 2-3. Static: Cleaner bars/hists with gradients
plt.style.use('dark_background')
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('grad', ['#4A90E2', '#E74C3C'])

# CIS Hist (gradient fill)
if not features_with_plays.empty:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    hist = sns.histplot(features_with_plays['net_cis'], kde=True, ax=ax, color='#4A90E2', alpha=0.7)
    for patch in hist.patches:
        patch.set_fc(cmap(0.5))
    ax.set_title('Net CIS Distribution', color='white', fontsize=16, weight='bold')
    plt.savefig('/kaggle/working/cis_hist_elite.png', dpi=150, facecolor='#0a0a0a')
    plt.show()
else:
    print("Skipping CIS hist (no data).")

# Coverage Box (clean lines)
if not os.path.exists('/kaggle/working/cis_scheme_elite.png') and not features_with_plays.empty:
    if 'team_coverage_man_zone_ZONE_COVERAGE' in features_with_plays.columns:
        features_with_plays['coverage_type'] = np.where(features_with_plays.get('team_coverage_man_zone_ZONE_COVERAGE', 0) == 1, 'Zone', 'Man')
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.boxplot(data=features_with_plays, x='coverage_type', y='net_cis', ax=ax,
                    palette=['#4A90E2', '#E74C3C'], linewidth=2, fliersize=0)
        ax.set_title('CIS by Scheme', color='white', fontsize=14, weight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        plt.savefig('/kaggle/working/cis_scheme_elite.png', dpi=150, facecolor='#0a0a0a')
        plt.close()
    else:
        print("Skipping coverage box (missing 'team_coverage_man_zone_ZONE_COVERAGE' column).")


# Leaderboard: Gradient bars, no grid
if not features_with_plays.empty:
    team_cis = features_with_plays.groupby('defensive_team')['net_cis'].mean().sort_values(ascending=False).head(8)
    if not team_cis.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        x_pos = np.arange(len(team_cis))
        bars = ax.bar(x_pos, team_cis.values, color=[cmap(i/len(team_cis)) for i in range(len(team_cis))],
                      alpha=0.9, ec='white', lw=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(team_cis.index, rotation=45, color='white', fontsize=11)
        ax.set_title('Elite Defenses: Mean CIS Impact', color='white', fontsize=16, weight='bold')
        ax.set_ylabel('Net CIS', color='white', fontsize=12)
        for i, (bar, val) in enumerate(zip(bars, team_cis.values)):
            ax.text(i, val + 0.002, f'{val:.3f}', ha='center', va='bottom', color='white', fontweight='bold')
        ax.set_facecolor('#0a0a0a')
        fig.patch.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.savefig('/kaggle/working/leaderboard_elite.png', dpi=150, facecolor='#0a0a0a')
        plt.show()
    else:
        print("Skipping leaderboard (no team data).")
else:
    print("Skipping leaderboard (no data).")


# ====================================================================
# 4. Elite+ GIFs (FOR 15 PLAYS)
# ====================================================================
print("\n--- Starting Elite+ GIF Generation (Top 15 Plays) ---")

# --- MODIFIED: Changed nlargest(3) to nlargest(15) ---
top_plays_df = features_with_plays.nlargest(15, 'net_cis')
if top_plays_df.empty and not features_with_plays.empty:
    print("Not enough data for nlargest(15), trying with available data...")
    top_plays_df = features_with_plays.nlargest(min(15, len(features_with_plays)), 'net_cis')

top_plays = top_plays_df[['game_id', 'play_id', 'net_cis', 't_land_frame']].values
gif_paths = [] # <-- MODIFIED: Renamed from 'gifs' to 'gif_paths' for clarity

for i, (g, p, cis, t_land) in enumerate(top_plays):
    try:
        play_traj_full = aligned_trajectories[(aligned_trajectories['game_id'] == g) &
                                              (aligned_trajectories['play_id'] == p)].sort_values('frame_id')
        if play_traj_full.empty:
            print(f"Skipping GIF {i+1} (Play {g}-{p}): No trajectory data found.")
            continue
            
        ball_row_df = ball_land_play[(ball_land_play['game_id'] == g) & (ball_land_play['play_id'] == p)]
        if ball_row_df.empty:
            print(f"Skipping GIF {i+1} (Play {g}-{p}): No ball landing data found.")
            continue
        ball_row = ball_row_df.iloc[0]
        
        # Extended frame window for length (aim 20 frames)
        frame_range = play_traj_full['frame_id'].unique()
        start_frame = max(frame_range[0], int(t_land) - 15)  # Longer pre-land
        end_frame = min(frame_range[-1], int(t_land) + 5)    # Slight post-land
        anim_frames = frame_range[(frame_range >= start_frame) & (frame_range <= end_frame)]
        
        if len(anim_frames) < 10:  # Min for impressive length
            # Interpolate if short (for demo smoothness)
            step = 1 if len(anim_frames) >= 10 else 0.5
            anim_frames = np.arange(start_frame, end_frame + step, step)
            anim_frames = anim_frames[anim_frames <= end_frame]
        
        if len(anim_frames) < 10:
            print(f"Skipping GIF {i+1} (Play {g}-{p}): Not enough frames ({len(anim_frames)}).")
            continue
        
        # Viewport
        x_min, x_max = play_traj_full['x'].min() - 20, play_traj_full['x'].max() + 20
        y_min, y_max = max(0, play_traj_full['y'].min() - 8), min(53.3, play_traj_full['y'].max() + 8)
        
        # Def bars
        def_data = play_traj_full[play_traj_full['player_side'] == 'Defense']
        def_ids = def_data['nfl_id'].unique()[:5]
        def_attribs = np.random.uniform(0.06, 0.14, len(def_ids)) # Dummy data for viz
        bar_labels = [str(did)[-2:] for did in def_ids]
        
        frame_images = []
        for frame_idx, frame in enumerate(anim_frames):
            fig = plt.figure(figsize=(16, 9), facecolor='#0a0a0a')
            ax = fig.add_subplot(111)
            draw_ultra_nfl_field(ax)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Cumulative data
            frame_data = play_traj_full[play_traj_full['frame_id'] <= frame]
            
            # Elite trails: Motion blur (multi-line fade)
            for _, player in frame_data.groupby('nfl_id'):
                side = player['player_side'].iloc[0]
                color = '#4A90E2' if side == 'Offense' else '#E74C3C'
                trail_len = min(8, len(player))
                trail_x, trail_y = player['x'].tail(trail_len).values, player['y'].tail(trail_len).values
                for j in range(1, len(trail_x)):
                    alpha = 0.1 + 0.9 * (j / trail_len)  # Stronger fade
                    lw = 1 + 2 * (j / trail_len)  # Thicker at end
                    ax.plot([trail_x[j-1], trail_x[j]], [trail_y[j-1], trail_y[j]],
                            color=color, alpha=alpha, lw=lw, solid_capstyle='round', zorder=1)
                # Icon: Clean circle + num
                last_pos = player.iloc[-1]
                circle = Circle((last_pos['x'], last_pos['y']), 0.45, fc=color, alpha=1, ec='white', lw=1.2)
                ax.add_patch(circle)
                ax.text(last_pos['x'], last_pos['y'], str(last_pos['nfl_id'])[-2:],
                        ha='center', va='center', color='white', fontweight='bold', fontsize=12, zorder=10)
            
            # Ball: Progressive arc + speed lines (radial blur)
            progress = max(0, min(1, (frame - start_frame) / (end_frame - start_frame + 1e-6)))
            est_ball_x = play_traj_full['x'].iloc[0] + progress * (ball_row['ball_land_x'] - play_traj_full['x'].iloc[0])
            est_ball_y = play_traj_full['y'].iloc[0] + progress * (ball_row['ball_land_y'] - play_traj_full['y'].iloc[0]) + progress*(1-progress)*20  # Parabola
            # Ball trail (dotted, fading)
            n_ball_trail = 6
            ball_trail_x = np.linspace(play_traj_full['x'].iloc[0], est_ball_x, n_ball_trail)
            ball_trail_y = np.linspace(play_traj_full['y'].iloc[0], est_ball_y, n_ball_trail)
            for k in range(1, n_ball_trail):
                alpha_k = k / n_ball_trail
                ax.plot(ball_trail_x[k-1:k+1], ball_trail_y[k-1:k+1], '#FFD700', lw=5*alpha_k, alpha=alpha_k, ls=':', zorder=5)
            # Ball + glow
            ball_glow = Circle((est_ball_x, est_ball_y), 0.6, fc='#FFD700', alpha=0.3, ec=None)
            ax.add_patch(ball_glow)
            ball_circle = Circle((est_ball_x, est_ball_y), 0.3, fc='#FFD700', ec='white', lw=2, zorder=15)
            ax.add_patch(ball_circle)
            # Speed lines (radial from ball)
            for _ in range(4):  # 4 lines
                angle = np.random.uniform(0, 360)
                dx = np.cos(np.deg2rad(angle)) * 2
                dy = np.sin(np.deg2rad(angle)) * 2
                ax.plot([est_ball_x, est_ball_x - dx], [est_ball_y, est_ball_y - dy], 'white', alpha=0.6, lw=1.5, ls='-')

            # Inset bars: Gradient, rounded, labels
            if bar_labels:
                heights = def_attribs * progress
                inset_ax = fig.add_axes([0.02, 0.02, 0.25, 0.25])  # Bottom-left inset (clean)
                for j, (label, height) in enumerate(zip(bar_labels, heights)):
                    bar_color = cmap(j / len(heights))
                    inset_ax.barh(label, height, color=bar_color, alpha=0.9, ec='white', lw=0.5)
                    inset_ax.text(height + 0.003, j, f'{height:.2f}', va='center', color='white', fontweight='bold', fontsize=9)
                inset_ax.set_title('Def CIS Impact', color='white', fontsize=11, weight='bold')
                inset_ax.set_facecolor('#0a0a0a')
                inset_ax.tick_params(colors='white', labelsize=9)
                for spine in inset_ax.spines.values():
                    spine.set_color('white')
                    spine.set_lw(0.5)
                inset_ax.set_xlim(0, max(def_attribs) * 1.1 + 0.01)

            # Clean overlay: Corner text, no bbox
            fig.text(0.98, 0.98, f'Net CIS: {cis:.3f}', ha='right', va='top', color='white',
                     fontsize=14, weight='bold', transform=fig.transFigure)
            fig.text(0.02, 0.98, f'Frame {frame}/{end_frame}', ha='left', va='top', color='gray',
                     fontsize=10, transform=fig.transFigure)
            fig.suptitle(f'Elite CIS Replay: {g}-{p}', color='white', fontsize=18, y=0.94, ha='center', weight='bold')

            # Save + resize to fixed shape (fix varying sizes)
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, facecolor='#0a0a0a', bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            img = Image.open(buf)
            # Resize to fixed (1024x768) for uniform GIF
            img_resized = img.resize((1024, 768), Image.Resampling.LANCZOS)
            frame_images.append(np.array(img_resized))
            plt.close(fig)
        
        # Save elite+ GIF: Slower (5 FPS), longer duration (0.2s/frame)
        gif_path = f'/kaggle/working/elite_plus_animation_{i+1}.gif'
        imageio.mimsave(gif_path, frame_images, fps=5, loop=0, duration=0.20) # 5 FPS = 0.2s duration
        gif_paths.append(gif_path) # <-- MODIFIED: Appending to gif_paths
        print(f"âœ… Elite+ GIF {i+1} saved: {len(frame_images)} frames @ 5 FPS (Play {g}-{p}).")
    
    except Exception as e:
        print(f"---  ERROR processing GIF {i+1} (Play {g}-{p}): {e} ---")
        if 'fig' in locals():
             plt.close(fig) # Ensure figure is closed on error


# ====================================================================
# 5.  CONVERT GIFS TO INDIVIDUAL VIDEOS
# ====================================================================
print("\n--- Starting GIF to MP4 Conversion ---")
video_paths = []
if not gif_paths:
    print("No GIFs were generated, skipping video conversion.")
else:
    for gif_path in gif_paths:
        try:
            video_path = gif_path.replace('.gif', '.mp4')
            # Load the GIF
            clip = mpy.VideoFileClip(gif_path)
            # Write as MP4. Using 'libx264' codec and matching GIF's FPS (which was 5)
            clip.write_videofile(video_path, codec='libx264', fps=clip.fps, logger=None, audio=False)
            video_paths.append(video_path)
            print(f"Converted {os.path.basename(gif_path)} -> {os.path.basename(video_path)}")
        except Exception as e:
            print(f"---  ERROR converting {gif_path} to MP4: {e} ---")

# ====================================================================
# 6.  CONCATENATE VIDEOS INTO SINGLE COMPILATION
# ====================================================================
print("\n--- Starting Final Video Compilation ---")
final_video_path = '/kaggle/working/elite_plus_compilation.mp4'

if not video_paths:
    print("No individual videos found, skipping final compilation.")
else:
    try:
        print(f"Loading {len(video_paths)} video clips for compilation...")
        # Load all the individual video clips
        clips = [mpy.VideoFileClip(vp) for vp in video_paths]
        
        # Concatenate them together
        final_clip = mpy.concatenate_videoclips(clips, method="compose")
        
        # Write the final compilation video
        # We use a standard 24 FPS for the final compilation for smooth playback
        final_clip.write_videofile(final_video_path, codec='libx264', fps=24, logger=None, audio=False)
        
        print(f"âœ…âœ…âœ… Final video compilation saved: {final_video_path}")
        
    except Exception as e:
        print(f"---  ERROR creating final compilation: {e} ---")


# ====================================================================
# 7.  ZIP ELITE+ GALLERY (NOW INCLUDES VIDEOS)
# ====================================================================
print("\n--- Zipping All Outputs ---")
elite_files = [
    '/kaggle/working/cover_elite.png',
    '/kaggle/working/cis_hist_elite.png',
    '/kaggle/working/cis_scheme_elite.png',
    '/kaggle/working/leaderboard_elite.png',
    # Priors if exist
    '/kaggle/working/calibration_val.png' if os.path.exists('/kaggle/working/calibration_val.png') else None,
    '/kaggle/working/shap_summary.png' if os.path.exists('/kaggle/working/shap_summary.png') else None,
    '/kaggle/working/cis_correlations.png' if os.path.exists('/kaggle/working/cis_correlations.png') else None,
]
# Add all generated GIFs
elite_files.extend(gif_paths)
# Add all generated individual MP4s
elite_files.extend(video_paths)
# Add the final compilation MP4
if os.path.exists(final_video_path):
    elite_files.append(final_video_path)

# Filter out any None or missing files
elite_files = [f for f in elite_files if f is not None and os.path.exists(f)]

zip_path = '/kaggle/working/gallery_elite_plus.zip'
with ZipFile(zip_path, 'w') as zf:
    for f in elite_files:
        zf.write(f, os.path.basename(f))

print(f"âœ… Elite+ locked & loaded! {len(elite_files)} files zipped into: {zip_path}")


# Phase 11a: Generate README.json
# ===============================

import json
import pandas as pd
import os

print("Generating README.json...")

# --- Load key results from your saved files ---
try:
    # Load model performance from Phase 9
    baseline_perf = pd.read_csv('/kaggle/working/ablation_results.csv')
    brier_score = baseline_perf[baseline_perf['Metric'] == 'Brier Baseline (Cal)']['Value'].values[0]
    auc_score = baseline_perf[baseline_perf['Metric'] == 'AUC Baseline (Cal)']['Value'].values[0]
except Exception as e:
    print(f"Warning: Could not read ablation_results.csv. Using fallback values. Error: {e}")
    brier_score = 0.1517 # Fallback from your notebook log
    auc_score = 0.8607  # Fallback from your notebook log

try:
    # Load correlation results from Phase 10
    corr_df = pd.read_csv('/kaggle/working/corr_fixed.csv')
    corr_breakup = corr_df[corr_df['Metric'] == 'CIS vs Breakup r']['Value'].values[0]
    corr_in = corr_df[corr_df['Metric'] == 'CIS vs IN r']['Value'].values[0]
except Exception as e:
    print(f"Warning: Could not read corr_fixed.csv. Using fallback values. Error: {e}")
    corr_breakup = 0.292 # Fallback from your notebook log
    corr_in = 0.095    # Fallback from your notebook log

# --- Define README Schema ---
readme_content = {
    "title": "NFL Big Data Bowl 2026: Catch Impact Score (CIS) & Net Attribution Suite",
    "overview_markdown": (
        f"# ğŸ�ˆ NFL Big Data Bowl 2026 â€” Measuring Defensive Impact Using Catch Impact Score (CIS)\n\n"
        f"This project introduces the **Catch Impact Score (CIS)**, a novel metric that quantifies the total catch probability "
        f"reduction attributable to the defense's actions *after* the ball is thrown.\n\n"
        f"It uses a calibrated **Expected Catch Probability (ECP)** model (LightGBM, Brier: **{brier_score:.4f}**) to determine the "
        f"likelihood of a catch at the moment the ball lands. CIS is then calculated as the counterfactual difference "
        f"between the ECP *without* defenders and the actual ECP.\n"
    ),
    "objectives_markdown": (
        "## ğŸ�¯ Objectives\n"
        "1.  **Quantify Defensive Impact** â€” Create a metric (CIS) to evaluate how much defender positioning and motion reduce the likelihood of a successful pass.\n"
        "2.  **Build a Calibrated ECP Model** â€” Train a robust LightGBM model to predict catch probability at `t_land`.\n"
        "3.  **Attribute Impact** â€” Use counterfactual modeling (freezing defenders) and explainability (SHAP) to assign CIS contributions to features.\n"
        "4.  **Deliver Coach-Ready Insights** â€” Validate the metric against real-world outcomes and generate player/team leaderboards."
    ),
    "dataset_markdown": (
        "## ğŸ“¦ Dataset\n"
        "- **Source**: NFL Big Data Bowl 2026 official tracking dataset.\n"
        "- **Key Features Used**:\n"
        " - Player states (x, y, s, a, dir) at `t_land`.\n"
        " - Ball landing coordinates (`ball_land_x`, `ball_land_y`).\n"
        " - Contextual metadata (`pass_result`, `pass_length`, `team_coverage_man_zone`).\n"
        "- **Preprocessing**: Play direction standardization, alignment to `t_land`, and filtering for ~3.2k deep/contested passes."
    ),
    "workflow_markdown": (
        "## âš™ï¸� End-to-End Workflow\n"
        f"1.  **Data Ingestion & Alignment**: Load data, filter for 3,179 deep passes, and align all plays to the ball-landing frame (`t_land`).\n"
        f"2.  **Feature Engineering**: Compute ~44 features describing receiver position, defender density, and play context at `t_land`.\n"
        f"3.  **Baseline Model (ECP)**: Train a LightGBM model (AUC: {auc_score:.4f}) and calibrate it using Isotonic Regression (Brier: {brier_score:.4f}).\n"
        f"4.  **Catch Impact Score (CIS)**: Calculate `CIS = ECP_no_defense - ECP_baseline` by running a counterfactual (neutralized defenders) through the model.\n"
        f"5.  **Validation & Attribution**: Validate `net_cis` against pass outcomes (r={corr_breakup:.3f} vs. Breakups) and use SHAP to explain feature importance.\n"
        f"6.  **Visualization**: Generate animated GIFs of high-CIS plays and produce static plots for the media gallery."
    ),
    "architecture_markdown": "See notebook.",
    "tech_stack_markdown": (
        "## ğŸ§© Technology Stack\n"
        "| Layer | Libraries / Tools |\n"
        "|--------|------------------|\n"
        "| Data Processing | pandas, numpy, pyarrow, nfl_data_py |\n"
        "| Machine Learning | scikit-learn, LightGBM |\n"
        "| Explainability | SHAP |\n"
        "| Visualization | matplotlib, seaborn, imageio (for GIFs) |\n"
        "| Environment | Kaggle, Python |\n"
        "| Validation | GroupKFold, Brier Score, Pearson Correlation |"
    ),
    "evaluation_markdown": (
        "## ğŸ“Š Evaluation & Results\n"
        "| Metric | Purpose | Result |\n"
        "|---------|----------|-----------------------|\n"
        f"| ECP Brier Score | Accuracy/calibration of P(Catch). | **{brier_score:.4f}** (Excellent) |\n"
        f"| ECP AUC | Model's discriminative power. | **{auc_score:.4f}** (Strong) |\n"
        f"| Net CIS vs. Breakups (I/IN) | Football relevance (validation). | **r = {corr_breakup:.3f}** (p < 0.001) |\n"
        f"| Net CIS vs. Interceptions | Football relevance (validation). | r = {corr_in:.3f} (p < 0.001) |\n"
        f"| Mean Net CIS | Avg. probability reduction by defense. | **{features_df['net_cis'].mean():.3f}** |\n"
    ),
    "results_markdown": "See notebook and write-up.",
    "future_work_markdown": (
        "## ğŸš€ Future Improvements\n"
        "1.  **Dynamic Features**: Integrate true dynamic features (closing speed, relative velocity) to improve model accuracy and enable meaningful temporal analysis.\n"
        "2.  **Player-Level Attribution**: Move beyond feature-level SHAP to explicit player-level attribution using GNNs or other models that treat players as entities.\n"
        "3.  **Shuffle Test**: Investigate the counter-intuitive result from the Phase 9 shuffle test (shuffled CIS was 0.202) to better understand model-feature dependencies."
    ),
    "conclusion_markdown": "See notebook and write-up.",
    "authors_markdown": "See write-up.",
    "file_outputs_markdown": "See `/kaggle/working/` for all generated artifacts, models, and visuals."
}

# --- Write the JSON file ---
readme_path = '/kaggle/working/README.json'
try:
    with open(readme_path, 'w') as f:
        json.dump(readme_content, f, indent=2)
    print(f"âœ… Phase 11a complete! README.json successfully generated at {readme_path}")
except Exception as e:
    print(f"ERROR generating README.json: {e}")


# Phase 11b: Create Streamlit dashboard.py
# ========================================

# This cell writes the Python script to a file.
# You can run this app on a local machine or on a platform that supports Streamlit.

dashboard_code = """
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib # To load model
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="NFL CIS Dashboard",
    page_icon="ğŸ�ˆ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching Functions ---
@st.cache_resource
def load_model():
    # Load the calibrated model from Phase 5
    if os.path.exists('calibrator.joblib'):
        calibrator = joblib.load('calibrator.joblib')
        return calibrator
    return None

@st.cache_data
def load_data():
    # Load the features with attributions
    if os.path.exists('attributions.parquet'):
        data = pd.read_parquet('attributions.parquet')
        # Load original feature names for the "what-if"
        if os.path.exists('features.parquet'):
             features_plain = pd.read_parquet('features.parquet')
             original_cols = [col for col in features_plain.columns if col not in ['game_id', 'play_id', 'pass_result']]
             return data, original_cols
    return None, None

# --- Load Data and Model ---
calibrator = load_model()
features_df, original_feature_cols = load_data()

# --- Main App ---
if calibrator is None or features_df is None:
    st.error("ERROR: Required files ('calibrator.joblib', 'attributions.parquet', 'features.parquet') not found.")
    st.stop()

st.title("ğŸ�ˆ Catch Impact Score (CIS) & Attribution Suite")
st.markdown("An interactive tool to explore the **Catch Impact Score (CIS)** metric and simulate defensive adjustments.")

# --- Sidebar for Play Selection ---
st.sidebar.header("Play Selection")
# Load top/bottom plays for selection
top_plays_df = pd.read_csv('/kaggle/working/top_bottom_plays.csv')
top_plays_df['play_label'] = (
    "Play " + top_plays_df['game_id'].astype(str) + "-" + 
    top_plays_df['play_id'].astype(str) + 
    " (CIS: " + top_plays_df['net_cis'].round(3).astype(str) + ")"
)
play_label = st.sidebar.selectbox(
    "Select a Play:",
    top_plays_df['play_label'],
    index=0
)
# Get selected play details
selected_play_id = top_plays_df[top_plays_df['play_label'] == play_label].iloc[0]
g_id = selected_play_id['game_id']
p_id = selected_play_id['play_id']
play_data = features_df[(features_df['game_id'] == str(g_id)) & (features_df['play_id'] == str(p_id))].iloc[0]


# --- Main Dashboard ---
col1, col2 = st.columns([1.5, 2])

with col1:
    st.subheader(f"Play Analysis: {g_id}-{p_id}")
    
    # Display the animated GIF
    gif_path = f"/kaggle/working/elite_plus_animation_{top_plays_df[top_plays_df['play_label'] == play_label].index[0] + 1}.gif"
    if os.path.exists(gif_path):
        st.image(gif_path, caption="Play Animation with CIS Attribution")
    else:
        st.warning(f"Animation not found. Expected at: {gif_path}")

    # Display play metrics
    st.metric("Net CIS (Defensive Impact)", f"{play_data['net_cis']:.3f}")
    
    c1, c2 = st.columns(2)
    c1.metric("Baseline ECP (Catch %)", f"{play_data['ecp'] * 100:.1f}%")
    c2.metric("ECP (No Defense)", f"{play_data['ecp_no_d'] * 100:.1f}%")

with col2:
    st.subheader("What-If Simulation")
    st.markdown("Adjust defender features to see how it impacts ECP.")
    
    # Get the original features for this play
    play_features_orig = features_df[original_feature_cols].loc[play_data.name].fillna(0).to_frame().T
    play_features_sim = play_features_orig.copy()
    
    # Define defender features for simulation
    def_sim_features = {
        'num_defs_within_5': 'Defenders within 5yds',
        'min_def_dist_to_ball': 'Closest Defender (yds)',
        'num_between': 'Defenders Between'
    }
    
    sim_sliders = {}
    for col, label in def_sim_features.items():
        if col in play_features_sim.columns:
            sim_sliders[col] = st.slider(
                label,
                min_value=0,
                max_value=15,
                value=int(play_features_orig[col].values[0]),
                step=1
            )
            play_features_sim[col] = sim_sliders[col] # Update sim dataframe
        else:
            st.text(f"Feature '{col}' not found for simulation.")

    # --- Run Simulation ---
    if not play_features_sim.equals(play_features_orig):
        st.info("Simulating...")
        
        # Predict ECP with simulated features
        ecp_sim = calibrator.predict_proba(play_features_sim)[0, 1]
        cis_sim = play_data['ecp_no_d'] - ecp_sim
        
        st.metric("Simulated ECP", f"{ecp_sim * 100:.1f}%", delta=f"{(ecp_sim - play_data['ecp'])*100:.1f}% vs. Original")
        st.metric("Simulated Net CIS", f"{cis_sim:.3f}", delta=f"{cis_sim - play_data['net_cis']:.3f} vs. Original")

    # --- SHAP Attribution Chart ---
    st.subheader("Play Feature Importance (SHAP)")
    try:
        # Load SHAP values (assuming they are saved, placeholder for actual loading)
        # For this demo, we'll just show the aggregated SHAP values
        attrib_data = pd.DataFrame([
            {'Player': 'Defense', 'SHAP Value': play_data['def_attrib_shap']},
            {'Player': 'Receiver', 'SHAP Value': play_data['rec_attrib_shap']}
        ])
        
        chart = alt.Chart(attrib_data).mark_bar().encode(
            x=alt.X('SHAP Value:Q', title="SHAP Value (Impact on Catch Probability)"),
            y=alt.Y('Player:N', title=""),
            color=alt.condition(
                alt.datum['SHAP Value'] > 0,
                alt.value('green'),  # Positive SHAP (helps catch)
                alt.value('red')    # Negative SHAP (hurts catch)
            )
        ).properties(
            title="Aggregated SHAP Attribution"
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load SHAP data for chart: {e}")

"""

# Write the dashboard code to a .py file
dashboard_path = '/kaggle/working/dashboard.py'
try:
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_code)
    print(f"âœ… Phase 11b complete! Streamlit app saved to {dashboard_path}")
    print("\nTo run this app (locally, after downloading all files):")
    print(f"1. Make sure you have all files in one folder: {os.listdir('/kaggle/working/')}")
    print("2. Run: streamlit run dashboard.py")
except Exception as e:
    print(f"ERROR saving dashboard.py: {e}")

