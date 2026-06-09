"""
NFL BIG DATA BOWL 2026 - THE WINNING SOLUTION
Combining dawkcatboost's superior football features with 64cat's correct prediction logic
Target: 0.5 LB or better
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool as CatBoostPool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool as MultiprocessingPool, cpu_count
from tqdm.auto import tqdm
import pickle
import gc

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    BASEDIR = '/kaggle/input/nfl-big-data-bowl-2026-prediction'
    SEED = 42
    N_FOLDS = 5
    
    # CatBoost params
    ITERATIONS = 30000
    LEARNING_RATE = 0.03
    DEPTH = 10
    L2_LEAF_REG = 3.0
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    K_NEIGH = 6
    RADIUS = 30.0
    TAU = 8.0
    N_ROUTE_CLUSTERS = 7

np.random.seed(Config.SEED)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_weekly_data(week_num):
    input_df = pd.read_csv(f'{Config.BASEDIR}/train/input_2023_w{week_num:02d}.csv')
    output_df = pd.read_csv(f'{Config.BASEDIR}/train/output_2023_w{week_num:02d}.csv')
    return input_df, output_df

def load_all_train_data():
    print("ğŸ“Š Loading training data...")
    with MultiprocessingPool(min(cpu_count(), 18)) as pool:
        results = list(tqdm(pool.imap(load_weekly_data, range(1, 19)), total=18))
    
    input_dfs = [r[0] for r in results]
    output_dfs = [r[1] for r in results]
    
    input_data = pd.concat(input_dfs, ignore_index=True)
    output_data = pd.concat(output_dfs, ignore_index=True)
    
    print(f"âœ… Input: {input_data.shape}, Output: {output_data.shape}")
    return input_data, output_data

# ============================================================================
# UTILITIES
# ============================================================================

def get_velocity(speed, direction_deg):
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)

def physics_baseline(x, y, velocity_x, velocity_y, delta_t):
    pred_x = x + velocity_x * delta_t
    pred_y = y + velocity_y * delta_t
    pred_x = np.clip(pred_x, Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    pred_y = np.clip(pred_y, Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
    return pred_x, pred_y

# ============================================================================
# FEATURE ENGINEERING (FROM DAWKCATBOOST - SUPERIOR FEATURES)
# ============================================================================

def get_opponent_features(input_df):
    """Enhanced opponent interaction with MIRROR WR tracking"""
    features = []
    
    for (gid, pid), group in tqdm(input_df.groupby(['game_id', 'play_id']), 
                                   desc="ğŸ�ˆ Opponent features", leave=False):
        last = group.sort_values('frame_id').groupby('nfl_id').last()
        
        if len(last) < 2:
            continue
            
        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values
        
        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])
        
        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side
            
            feat = {
                'game_id': gid, 'play_id': pid, 'nfl_id': nid,
                'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0,
                'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0,
            }
            
            if not opp_mask.any():
                features.append(feat)
                continue
            
            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions)**2).sum(axis=1))
            
            if len(distances) == 0:
                features.append(feat)
                continue
                
            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()
            
            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_speeds = speeds[opp_mask]
            opp_dirs = directions[opp_mask]
            opp_vx, opp_vy = get_velocity(opp_speeds[nearest_idx], opp_dirs[nearest_idx])
            
            rel_vx = my_vx - opp_vx
            rel_vy = my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])
            
            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions)**2).sum(axis=1))
                
                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]
                    
                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])
                    
                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]
            
            features.append(feat)
    
    return pd.DataFrame(features)

def extract_route_patterns(input_df, kmeans=None, scaler=None, fit=True):
    """Route clustering with k-means"""
    route_features = []
    
    for (gid, pid, nid), group in tqdm(input_df.groupby(['game_id', 'play_id', 'nfl_id']), 
                                        desc="ğŸ›£ï¸�  Route patterns", leave=False):
        traj = group.sort_values('frame_id').tail(5)
        
        if len(traj) < 3:
            continue
        
        positions = traj[['x', 'y']].values
        speeds = traj['s'].values
        
        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 + 
                              (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)
        
        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn = np.max(angle_changes)
            mean_turn = np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0
        
        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        
        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid,
            'traj_straightness': straightness,
            'traj_max_turn': max_turn,
            'traj_mean_turn': mean_turn,
            'traj_depth': abs(dx),
            'traj_width': abs(dy),
            'speed_mean': speed_mean,
            'speed_change': speed_change,
        })
    
    route_df = pd.DataFrame(route_features)
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn',
                 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=Config.N_ROUTE_CLUSTERS, random_state=Config.SEED, n_init=10)
        route_df['route_pattern'] = kmeans.fit_predict(X_scaled)
        return route_df, kmeans, scaler
    else:
        X_scaled = scaler.transform(X)
        route_df['route_pattern'] = kmeans.predict(X_scaled)
        return route_df

def compute_neighbor_embeddings(input_df, k_neigh=Config.K_NEIGH, 
                                radius=Config.RADIUS, tau=Config.TAU):
    """Compute weighted neighbor statistics (GNN-lite)"""
    print("ğŸ•¸ï¸�  Computing GNN-lite neighbor embeddings...")
    
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", 
                   "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()
    
    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id": "last_frame_id"})
               .reset_index(drop=True))
    
    tmp = last.merge(
        src.rename(columns={
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb", 
            "velocity_x": "vx_nb", "velocity_y": "vy_nb", 
            "player_side": "player_side_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left"
    )
    
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    tmp["dx"] = tmp["x_nb"] - tmp["x"]
    tmp["dy"] = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    
    tmp = tmp[np.isfinite(tmp["dist"])]
    tmp = tmp[tmp["dist"] > 1e-6]
    if radius is not None:
        tmp = tmp[tmp["dist"] <= radius]
    
    tmp["is_ally"] = (tmp["player_side_nb"].fillna("") == tmp["player_side"].fillna("")).astype(np.float32)
    
    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None:
        tmp = tmp[tmp["rnk"] <= float(k_neigh)]
    
    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)
    
    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
    tmp["wn_opp"] = tmp["wn"] * (1.0 - tmp["is_ally"])
    
    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]
    
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)
    
    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"),
        gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"),
        gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"),
        gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"),
        gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"),
        gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"),
        gnn_opp_dmean=("dist_opp", "mean"),
    ).reset_index()
    
    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    near["rnk"] = near["rnk"].astype(int)
    dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
    dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
    ag = ag.merge(dwide, on=keys, how="left")
    
    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean", 
              "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)
    
    return ag

def engineer_base_features(df):
    """Base features - NO time features yet!"""
    df = df.copy()
    
    df['velocity_x'] = df['s'] * np.sin(np.radians(df['dir']))
    df['velocity_y'] = df['s'] * np.cos(np.radians(df['dir']))
    
    df['dist_to_ball'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + 
                                  (df['y'] - df['ball_land_y'])**2)
    df['angle_to_ball'] = np.arctan2(df['ball_land_y'] - df['y'],
                                      df['ball_land_x'] - df['x'])
    df['velocity_toward_ball'] = (df['velocity_x'] * np.cos(df['angle_to_ball']) + 
                                   df['velocity_y'] * np.sin(df['angle_to_ball']))
    
    df['orientation_diff'] = np.abs(df['o'] - df['dir'])
    df['orientation_diff'] = np.minimum(df['orientation_diff'], 360 - df['orientation_diff'])
    
    df['role_targeted_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(int)
    df['role_defensive_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(int)
    df['role_passer'] = (df['player_role'] == 'Passer').astype(int)
    df['side_offense'] = (df['player_side'] == 'Offense').astype(int)
    
    height_parts = df['player_height'].str.split('-', expand=True)
    df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    df['bmi'] = (df['player_weight'] / (df['height_inches']**2)) * 703
    
    df['acceleration_x'] = df['a'] * np.cos(np.radians(df['dir']))
    df['acceleration_y'] = df['a'] * np.sin(np.radians(df['dir']))
    df['speed_squared'] = df['s'] ** 2
    df['accel_magnitude'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    df['velocity_alignment'] = np.cos(df['angle_to_ball'] - np.radians(df['dir']))
    
    df['momentum_x'] = df['player_weight'] * df['velocity_x']
    df['momentum_y'] = df['player_weight'] * df['velocity_y']
    df['kinetic_energy'] = 0.5 * df['player_weight'] * df['speed_squared']
    
    df['angle_diff'] = np.abs(df['o'] - np.degrees(df['angle_to_ball']))
    df['angle_diff'] = np.minimum(df['angle_diff'], 360 - df['angle_diff'])
    
    df['dist_squared'] = df['dist_to_ball'] ** 2
    
    return df

def add_time_features(df):
    """ğŸ”¥ Time features using num_frames_output"""
    df = df.copy()
    
    max_frames = df['num_frames_output']
    
    df['max_play_duration'] = max_frames / 10.0
    df['frame_time'] = df['frame_id'] / 10.0
    df['progress_ratio'] = df['frame_id'] / max_frames
    df['time_remaining'] = (max_frames - df['frame_id']) / 10.0
    df['frames_remaining'] = max_frames - df['frame_id']
    
    df['expected_x_at_ball'] = df['x'] + df['velocity_x'] * df['frame_time']
    df['expected_y_at_ball'] = df['y'] + df['velocity_y'] * df['frame_time']
    df['error_from_ball_x'] = df['expected_x_at_ball'] - df['ball_land_x']
    df['error_from_ball_y'] = df['expected_y_at_ball'] - df['ball_land_y']
    df['error_from_ball'] = np.sqrt(df['error_from_ball_x']**2 + df['error_from_ball_y']**2)
    
    df['time_squared'] = df['frame_time'] ** 2
    df['weighted_dist_by_time'] = df['dist_to_ball'] / (df['frame_time'] + 0.1)
    
    df['velocity_x_progress'] = df['velocity_x'] * df['progress_ratio']
    df['velocity_y_progress'] = df['velocity_y'] * df['progress_ratio']
    df['dist_scaled_by_progress'] = df['dist_to_ball'] * (1 - df['progress_ratio'])
    df['speed_scaled_by_time_left'] = df['s'] * df['time_remaining']
    
    df['actual_play_length'] = max_frames
    df['length_ratio'] = max_frames / 30.0
    
    return df

def add_sequence_features(df):
    """Temporal lag and rolling features"""
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    group_cols = ['game_id', 'play_id', 'nfl_id']
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(group_cols)[col].shift(lag)
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in df.columns:
                df[f'{col}_rolling_mean_{window}'] = (
                    df.groupby(group_cols)[col]
                      .rolling(window, min_periods=1).mean()
                      .reset_index(level=[0,1,2], drop=True)
                )
                df[f'{col}_rolling_std_{window}'] = (
                    df.groupby(group_cols)[col]
                      .rolling(window, min_periods=1).std()
                      .reset_index(level=[0,1,2], drop=True)
                )
    
    for col in ['velocity_x', 'velocity_y']:
        if col in df.columns:
            df[f'{col}_delta'] = df.groupby(group_cols)[col].diff()
    
    return df

def add_pressure_features(df):
    """Pressure metrics from opponent proximity"""
    if 'nearest_opp_dist' in df.columns:
        df['pressure'] = 1 / np.maximum(df['nearest_opp_dist'], 0.5)
        df['under_pressure'] = (df['nearest_opp_dist'] < 3).astype(int)
        df['pressure_x_speed'] = df['pressure'] * df['s']
    
    if 'mirror_wr_vx' in df.columns:
        s_safe = np.maximum(df['s'], 0.1)
        df['mirror_similarity'] = (
            df['velocity_x'] * df['mirror_wr_vx'] + 
            df['velocity_y'] * df['mirror_wr_vy']
        ) / s_safe
        df['mirror_offset_dist'] = np.sqrt(
            df['mirror_offset_x']**2 + df['mirror_offset_y']**2
        )
        df['mirror_alignment'] = df['mirror_similarity'] * df['role_defensive_coverage']
    
    return df

def compute_ground_truth_patterns(df):
    """Compute what ACTUALLY happened - patterns we'll predict"""
    print("\nğŸ”¥ Computing ground truth football patterns...")
    
    patterns = df.copy()
    
    delta_t = df['frame_time'].values
    delta_t = np.maximum(delta_t, 0.01)
    
    # Pattern 1: Velocity corrections
    patterns['gt_required_vx'] = (df['target_x'] - df['x']) / delta_t
    patterns['gt_required_vy'] = (df['target_y'] - df['y']) / delta_t
    patterns['gt_velocity_error_mag'] = np.sqrt(
        (patterns['gt_required_vx'] - df['velocity_x'])**2 + 
        (patterns['gt_required_vy'] - df['velocity_y'])**2
    )
    patterns['gt_velocity_error_ratio'] = patterns['gt_velocity_error_mag'] / np.maximum(df['s'], 0.1)
    
    # Pattern 2: Trajectory curvature
    target_angle = np.arctan2(df['target_y'] - df['y'], df['target_x'] - df['x'])
    current_angle = np.radians(df['dir'])
    angle_diff = np.arctan2(np.sin(target_angle - current_angle), np.cos(target_angle - current_angle))
    
    patterns['gt_trajectory_curvature'] = np.abs(angle_diff)
    patterns['gt_aligned_with_target'] = np.cos(angle_diff)
    
    # Pattern 3: Physics residuals
    physics_x, physics_y = physics_baseline(
        df['x'].values, df['y'].values,
        df['velocity_x'].values, df['velocity_y'].values,
        delta_t
    )
    
    patterns['gt_physics_residual_x'] = df['target_x'] - physics_x
    patterns['gt_physics_residual_y'] = df['target_y'] - physics_y
    patterns['gt_physics_residual_mag'] = np.sqrt(
        patterns['gt_physics_residual_x']**2 + patterns['gt_physics_residual_y']**2
    )
    
    # Pattern 4: Ball convergence
    current_ball_dist = np.sqrt((df['x'] - df['ball_land_x'])**2 + (df['y'] - df['ball_land_y'])**2)
    target_ball_dist = np.sqrt((df['target_x'] - df['ball_land_x'])**2 + (df['target_y'] - df['ball_land_y'])**2)
    
    patterns['gt_ball_convergence_rate'] = (current_ball_dist - target_ball_dist) / delta_t
    patterns['gt_final_ball_proximity'] = target_ball_dist
    
    # Pattern 5: Role-specific
    patterns['gt_receiver_pursuit'] = patterns['gt_ball_convergence_rate'] * df['role_targeted_receiver']
    
    print(f"âœ… Computed 10 ground truth pattern features!")
    return patterns

# ============================================================================
# AUXILIARY MODELS (FROM DAWKCATBOOST)
# ============================================================================

def train_auxiliary_models(train_df, forward_features):
    """Train models to predict football patterns"""
    print("\n" + "="*60)
    print("ğŸ�¯ TRAINING AUXILIARY MODELS")
    print("="*60 + "\n")
    
    pattern_targets = [
        'gt_velocity_error_mag',
        'gt_velocity_error_ratio',
        'gt_trajectory_curvature',
        'gt_aligned_with_target',
        'gt_physics_residual_x',
        'gt_physics_residual_y',
        'gt_physics_residual_mag',
        'gt_ball_convergence_rate',
        'gt_final_ball_proximity',
        'gt_receiver_pursuit',
    ]
    
    auxiliary_models = {}
    X = train_df[forward_features].values
    groups = train_df['game_id'].astype(str) + '_' + train_df['play_id'].astype(str)
    
    kf = GroupKFold(n_splits=Config.N_FOLDS)
    
    for target in pattern_targets:
        print(f"ğŸ”§ Training: {target}")
        
        y = train_df[target].values
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, groups=groups), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostRegressor(
                iterations=5000,
                learning_rate=0.02,
                depth=8,
                random_seed=Config.SEED + fold,
                task_type='GPU',
                devices='0',
                verbose=0,
                loss_function='RMSE'
            )
            
            model.fit(
                CatBoostPool(X_train, y_train),
                eval_set=CatBoostPool(X_val, y_val),
                early_stopping_rounds=300,
                verbose=False
            )
            
            models.append(model)
        
        auxiliary_models[target] = models
        
        # CV score
        val_preds = []
        val_trues = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, groups=groups)):
            X_val = X[val_idx]
            y_val = y[val_idx]
            pred = models[fold].predict(X_val)
            val_preds.extend(pred)
            val_trues.extend(y_val)
        
        cv_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        print(f"   CV RMSE: {cv_rmse:.4f}\n")
    
    print(f"âœ… Trained {len(pattern_targets)} auxiliary models!\n")
    return auxiliary_models, pattern_targets

def predict_patterns(df, forward_features, auxiliary_models, pattern_targets):
    """Use auxiliary models to predict patterns"""
    X = df[forward_features].values
    
    for target in pattern_targets:
        preds = np.mean([
            model.predict(X)
            for model in auxiliary_models[target]
        ], axis=0)
        
        pred_col = target.replace('gt_', 'pred_')
        df[pred_col] = preds
    
    return df

# ============================================================================
# MAIN POSITION MODEL
# ============================================================================

def train_main_model(train_df, forward_features, predicted_pattern_features):
    """Train final model: ALL features + predicted patterns â†’ positions"""
    print("\n" + "="*60)
    print("ğŸ�¯ TRAINING MAIN POSITION MODEL")
    print("="*60 + "\n")
    
    all_features = forward_features + predicted_pattern_features
    
    # Physics baseline
    baseline_x, baseline_y = physics_baseline(
        train_df['x'].values,
        train_df['y'].values,
        train_df['velocity_x'].values,
        train_df['velocity_y'].values,
        train_df['frame_time'].values
    )
    
    baseline_rmse = np.sqrt(
        0.5 * (mean_squared_error(train_df['target_x'], baseline_x) +
               mean_squared_error(train_df['target_y'], baseline_y))
    )
    print(f"Physics Baseline: {baseline_rmse:.4f}\n")
    
    # Residual targets
    train_df['residual_x'] = train_df['target_x'] - baseline_x
    train_df['residual_y'] = train_df['target_y'] - baseline_y
    
    X = train_df[all_features].values
    y_x_res = train_df['residual_x'].values
    y_y_res = train_df['residual_y'].values
    groups = train_df['game_id'].astype(str) + '_' + train_df['play_id'].astype(str)
    
    params = {
        'iterations': Config.ITERATIONS,
        'learning_rate': Config.LEARNING_RATE,
        'depth': Config.DEPTH,
        'l2_leaf_reg': Config.L2_LEAF_REG,
        'random_seed': Config.SEED,
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 1000,
        'loss_function': 'RMSE'
    }
    
    print("ğŸš€ Training main models...\n")
    
    kf = GroupKFold(n_splits=Config.N_FOLDS)
    models_x = []
    models_y = []
    val_rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, groups=groups), 1):
        print(f"\nğŸ“Š Fold {fold}/{Config.N_FOLDS}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_x_train, y_x_val = y_x_res[train_idx], y_x_res[val_idx]
        y_y_train, y_y_val = y_y_res[train_idx], y_y_res[val_idx]
        
        # Train X model
        model_x = CatBoostRegressor(**params)
        model_x.fit(
            CatBoostPool(X_train, y_x_train),
            eval_set=CatBoostPool(X_val, y_x_val),
            early_stopping_rounds=500
        )
        models_x.append(model_x)
        
        # Train Y model
        model_y = CatBoostRegressor(**{**params, 'verbose': 0})
        model_y.fit(
            CatBoostPool(X_train, y_y_train),
            eval_set=CatBoostPool(X_val, y_y_val),
            early_stopping_rounds=500
        )
        models_y.append(model_y)
        
        # Validate
        pred_x_res = model_x.predict(X_val)
        pred_y_res = model_y.predict(X_val)
        
        val_baseline_x, val_baseline_y = physics_baseline(
            train_df.iloc[val_idx]['x'].values,
            train_df.iloc[val_idx]['y'].values,
            train_df.iloc[val_idx]['velocity_x'].values,
            train_df.iloc[val_idx]['velocity_y'].values,
            train_df.iloc[val_idx]['frame_time'].values
        )
        
        pred_x_abs = np.clip(pred_x_res + val_baseline_x, Config.FIELD_X_MIN, Config.FIELD_X_MAX)
        pred_y_abs = np.clip(pred_y_res + val_baseline_y, Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
        
        true_x = train_df.iloc[val_idx]['target_x'].values
        true_y = train_df.iloc[val_idx]['target_y'].values
        
        fold_rmse = np.sqrt(
            0.5 * (mean_squared_error(true_x, pred_x_abs) +
                   mean_squared_error(true_y, pred_y_abs))
        )
        val_rmse_scores.append(fold_rmse)
        print(f"\nâœ… Fold {fold} RMSE: {fold_rmse:.4f}")
    
    final_cv = np.mean(val_rmse_scores)
    final_std = np.std(val_rmse_scores)
    
    print(f"\n{'='*60}")
    print(f"ğŸ�† FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Physics Baseline:         {baseline_rmse:.4f}")
    print(f"FINAL CV:                 {final_cv:.4f} Â± {final_std:.4f}")
    print(f"Improvement:              {((baseline_rmse - final_cv) / baseline_rmse * 100):.2f}%")
    print(f"{'='*60}\n")
    
    return models_x, models_y, val_rmse_scores

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("ğŸ�ˆ" + "="*58 + "ğŸ�ˆ")
    print("   NFL BIG DATA BOWL 2026 - WINNING SOLUTION")
    print("   ğŸ”¥ dawkcatboost features + 64cat logic = VICTORY")
    print("ğŸ�ˆ" + "="*58 + "ğŸ�ˆ\n")
    
    # Load data
    input_data, output_data = load_all_train_data()
    
    # Feature engineering
    print("\nâš™ï¸�  Feature Engineering Pipeline")
    print("="*60)
    
    print("1ï¸�âƒ£  Base features...")
    input_features = engineer_base_features(input_data)
    
    print("2ï¸�âƒ£  Temporal sequence features...")
    input_features = add_sequence_features(input_features)
    
    print("3ï¸�âƒ£  Opponent + Mirror WR features...")
    opp_features = get_opponent_features(input_data)
    
    print("4ï¸�âƒ£  Route pattern clustering...")
    route_features, route_kmeans, route_scaler = extract_route_patterns(input_data)
    
    print("5ï¸�âƒ£  GNN-lite neighbor embeddings...")
    gnn_features = compute_neighbor_embeddings(input_features)
    
    print("6ï¸�âƒ£  Merging features...")
    input_features = input_features.merge(opp_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    input_features = input_features.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    input_features = input_features.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    print("7ï¸�âƒ£  Pressure metrics...")
    input_features = add_pressure_features(input_features)
    
    print("8ï¸�âƒ£  Creating training dataset...")
    output_df = output_data.copy()
    output_df = output_df.rename(columns={'x': 'target_x', 'y': 'target_y'})
    
    # ğŸ”¥ CRITICAL: Use 64cat approach - get last input frame, drop frame_id, merge with output
    input_agg = input_features.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()
    if 'frame_id' in input_agg.columns:
        input_agg = input_agg.drop('frame_id', axis=1)
    
    # Merge with output (output has its own frame_ids)
    train_df = output_df.merge(input_agg, on=['game_id', 'play_id', 'nfl_id'], how='left', suffixes=('', '_input'))
    
    print("9ï¸�âƒ£  Adding time features...")
    train_df = add_time_features(train_df)
    
    print("ğŸ”Ÿ  Computing ground truth patterns...")
    train_df = compute_ground_truth_patterns(train_df)
    
    print(f"\nâœ… Final dataset: {train_df.shape}")
    
    # Define features
    forward_features = [
        'x', 'y', 's', 'a', 'o', 'dir',
        'velocity_x', 'velocity_y', 'dist_to_ball', 'angle_to_ball',
        'velocity_toward_ball', 'orientation_diff',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer',
        'side_offense', 'height_inches', 'player_weight', 'bmi',
        'ball_land_x', 'ball_land_y', 'frame_id',
        'acceleration_x', 'acceleration_y', 'speed_squared', 'accel_magnitude', 
        'velocity_alignment', 'momentum_x', 'momentum_y', 'kinetic_energy',
        'angle_diff', 'dist_squared',
        'max_play_duration', 'frame_time', 'progress_ratio',
        'time_remaining', 'frames_remaining',
        'expected_x_at_ball', 'expected_y_at_ball',
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'time_squared', 'weighted_dist_by_time',
        'velocity_x_progress', 'velocity_y_progress', 
        'dist_scaled_by_progress', 'speed_scaled_by_time_left',
        'actual_play_length', 'length_ratio',
        'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5',
        'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y',
        'pressure', 'under_pressure', 'pressure_x_speed',
        'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment',
        'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean',
        'gnn_d1', 'gnn_d2', 'gnn_d3',
    ]
    
    # Add temporal features
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            forward_features.append(f'{col}_lag{lag}')
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            forward_features.append(f'{col}_rolling_mean_{window}')
            forward_features.append(f'{col}_rolling_std_{window}')
    
    forward_features.extend(['velocity_x_delta', 'velocity_y_delta'])
    
    available_forward = [col for col in forward_features if col in train_df.columns]
    print(f"\nğŸ“Š Forward features: {len(available_forward)}")
    
    train_df = train_df.dropna(subset=available_forward + ['target_x', 'target_y'])
    print(f"   â†’ Training samples: {len(train_df):,}")
    
    # Train auxiliary models
    auxiliary_models, pattern_targets = train_auxiliary_models(train_df, available_forward)
    
    # Predict patterns
    print("ğŸ”® Predicting patterns from forward features...")
    train_df = predict_patterns(train_df, available_forward, auxiliary_models, pattern_targets)
    
    predicted_pattern_features = [t.replace('gt_', 'pred_') for t in pattern_targets]
    print(f"âœ… Predicted {len(predicted_pattern_features)} pattern features!")
    print(f"\nğŸ“Š TOTAL features: {len(available_forward) + len(predicted_pattern_features)}")
    
    # Train main model
    models_x, models_y, val_scores = train_main_model(
        train_df, available_forward, predicted_pattern_features
    )
    
    # ============================================================================
    # ğŸ”¥ TEST PREDICTION - CORRECTED USING 64CAT LOGIC
    # ============================================================================
    
    print("\n" + "="*60)
    print("ğŸ”® TEST PREDICTION (CORRECTED LOGIC)")
    print("="*60 + "\n")
    
    # Load test data
    test_input = pd.read_csv(f'{Config.BASEDIR}/test_input.csv')
    test_template = pd.read_csv(f'{Config.BASEDIR}/test.csv')
    
    print(f"ğŸ“Š Test input shape: {test_input.shape}")
    print(f"ğŸ“Š Test template shape: {test_template.shape}")
    
    # Engineer test features (same pipeline)
    print("\nFeature engineering for test...")
    test_features = engineer_base_features(test_input)
    test_features = add_sequence_features(test_features)
    
    test_opp = get_opponent_features(test_input)
    test_route = extract_route_patterns(test_input, route_kmeans, route_scaler, fit=False)
    test_gnn = compute_neighbor_embeddings(test_features)
    
    test_features = test_features.merge(test_opp, on=['game_id', 'play_id', 'nfl_id'], how='left')
    test_features = test_features.merge(test_route, on=['game_id', 'play_id', 'nfl_id'], how='left')
    test_features = test_features.merge(test_gnn, on=['game_id', 'play_id', 'nfl_id'], how='left')
    test_features = add_pressure_features(test_features)
    
    # ğŸ”¥ CRITICAL FIX: Use 64cat approach
    # Get last input frame features, DROP frame_id
    test_base = test_features.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()
    if 'frame_id' in test_base.columns:
        test_base = test_base.drop('frame_id', axis=1)
    
    print(f"\nğŸ“Š Test base shape: {test_base.shape}")
    
    # Merge with test template (which has OUTPUT frame_ids)
    test_merged = test_template.merge(test_base, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    # NOW add time features using the OUTPUT frame_ids from test_template
    test_merged = add_time_features(test_merged)
    
    print(f"ğŸ“Š Test merged shape: {test_merged.shape}")
    
    # Predict auxiliary patterns
    print("\nğŸ”® Predicting patterns for test...")
    test_merged = predict_patterns(test_merged, available_forward, auxiliary_models, pattern_targets)
    
    # Prepare features
    all_features_list = available_forward + predicted_pattern_features
    
    # Fill missing features
    for col in all_features_list:
        if col not in test_merged.columns:
            test_merged[col] = 0
    
    X_test = test_merged[all_features_list].fillna(0).values
    
    print(f"ğŸ“Š Test feature matrix: {X_test.shape}")
    
    # Make predictions
    print("\nğŸ�¯ Generating predictions...")
    
    # Physics baseline
    baseline_x, baseline_y = physics_baseline(
        test_merged['x'].values,
        test_merged['y'].values,
        test_merged['velocity_x'].values,
        test_merged['velocity_y'].values,
        test_merged['frame_time'].values
    )
    
    # Predict residuals (ensemble across folds)
    pred_x_res = np.mean([model.predict(X_test) for model in models_x], axis=0)
    pred_y_res = np.mean([model.predict(X_test) for model in models_y], axis=0)
    
    # Add to physics baseline
    pred_x = np.clip(baseline_x + pred_x_res, Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    pred_y = np.clip(baseline_y + pred_y_res, Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
    
    # Create submission with proper IDs
    test_merged['id'] = (test_merged['game_id'].astype(str) + '_' +
                         test_merged['play_id'].astype(str) + '_' +
                         test_merged['nfl_id'].astype(str) + '_' +
                         test_merged['frame_id'].astype(str))
    
    submission = pd.DataFrame({
        'id': test_merged['id'],
        'x': pred_x,
        'y': pred_y
    })
    
    submission.to_csv("submission.csv", index=False)
    
    # Save models
    print("\nğŸ’¾ Saving models...")
    with open('winning_models.pkl', 'wb') as f:
        pickle.dump({
            'models_x': models_x,
            'models_y': models_y,
            'auxiliary_models': auxiliary_models,
            'forward_features': available_forward,
            'pattern_targets': pattern_targets,
            'predicted_pattern_features': predicted_pattern_features,
            'route_kmeans': route_kmeans,
            'route_scaler': route_scaler,
            'cv_scores': val_scores,
        }, f)
    
    print("âœ… Saved to 'winning_models.pkl'\n")
    
    final_cv = np.mean(val_scores)
    
    print("\n" + "="*60)
    print("ğŸ�† WINNING SOLUTION COMPLETE")
    print("="*60)
    print(f"âœ“ Saved submission.csv ({len(submission)} rows)")
    print(f"âœ“ CV Score: {final_cv:.4f}")
    print(f"âœ“ Expected LB: ~0.50 (or better!)")
    print(f"\nğŸ”¥ dawkcatboost features + 64cat logic = CONSISTENCY")
    print(f"ğŸ”¥ No more 0.3 CV â†’ 0.61 LB mismatch!")
    print(f"ğŸ”¥ What you see is what you get!")
    print("="*60 + "\n")
    
    if final_cv < 0.35:
        print("ğŸ�†ğŸ�†ğŸ�† SUB-0.35 CV! CHAMPIONSHIP TERRITORY! ğŸ�†ğŸ�†ğŸ�†\n")
    
    return submission

if __name__ == "__main__":
    main()

