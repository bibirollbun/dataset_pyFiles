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


# ==============================
# MABe Mouse Behavior Detection
# NO-LIMITS PIPELINE - MULTI-CLASS VERSION
# TARGET: F1 > 0.80
# ==============================

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import gc
import warnings
import os
warnings.filterwarnings('ignore')

# --- 1. ENHANCED Configuration ---
class EnhancedConfig:
    DATA_PATH = Path("/kaggle/input/MABe-mouse-behavior-detection")
    SEED = 42
    N_FOLDS = 5
    # OPTIMIZED PARAMETERS FOR F1 > 0.80
    N_ESTIMATORS = 2000
    LEARNING_RATE = 0.01
    NUM_LEAVES = 127
    MIN_CHILD_SAMPLES = 50
    SUBSAMPLE = 0.7
    COLSAMPLE_BYTREE = 0.7
    REG_ALPHA = 0.3
    REG_LAMBDA = 0.3
    MAX_DEPTH = -1
    MIN_SPLIT_GAIN = 0.01
    SUBSAMPLE_FREQ = 1
    RANDOM_STATE = SEED
    USE_GPU = True
    TEMP_DIR = Path("/kaggle/working/temp_features_enhanced")
    # ENHANCED STRATEGY
    MAX_FRAMES_PER_VIDEO = 50000
    MIN_VIDEO_LENGTH = 50
    EARLY_STOPPING_ROUNDS = 100
    USE_SCALING = True
    # MEMORY MANAGEMENT
    BATCH_SIZE = 500000
    USE_DISK_STORAGE = True
    # ENHANCED FEATURES
    USE_ENHANCED_FEATURES = True
    FEATURE_SELECTION_THRESHOLD = 0.001

cfg = EnhancedConfig()
cfg.TEMP_DIR.mkdir(exist_ok=True)

# --- 2. ESSENTIAL UTILITY FUNCTIONS ---
def load_data():
    print("ğŸ“¥ Loading data...")
    train = pd.read_csv(cfg.DATA_PATH / "train.csv", low_memory=False)
    test = pd.read_csv(cfg.DATA_PATH / "test.csv", low_memory=False)
    print(f"âœ… Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test

def process_metadata(train):
    print("ğŸ”§ Processing metadata...")
    return train

def create_features(train):
    print("ğŸ”§ Creating features...")
    if "behaviors_labeled" in train.columns:
        train_labeled = train[~train["behaviors_labeled"].isnull()].copy()
        print(f"âœ… Labeled samples: {len(train_labeled)}")
    else:
        train_labeled = train.copy()
    return train, train_labeled

def load_tracking_annotation(meta_row, mode="train"):
    lab_id, video_id = meta_row["lab_id"], meta_row["video_id"]
    track_fp = cfg.DATA_PATH / f"{mode}_tracking" / str(lab_id) / f"{video_id}.parquet"
    annot_fp = cfg.DATA_PATH / f"{mode}_annotation" / str(lab_id) / f"{video_id}.parquet"
    
    tracking = None
    annotation = pd.DataFrame()
    
    if track_fp.exists():
        try:
            tracking = pd.read_parquet(track_fp)
        except Exception:
            pass
    
    if annot_fp.exists() and mode == "train":
        try:
            annotation = pd.read_parquet(annot_fp)
        except Exception:
            pass
    
    return tracking, annotation

def save_batch_to_disk(batch_data, batch_num):
    batch_path = cfg.TEMP_DIR / f"batch_{batch_num}.npz"
    X_batch, y_batch, groups_batch = batch_data
    np.savez_compressed(batch_path, X=X_batch, y=y_batch, groups=groups_batch)
    return batch_path

def load_batch_from_disk(batch_path):
    data = np.load(batch_path)
    return data['X'], data['y'], data['groups']

def cleanup_temp_files(batch_paths):
    for batch_path in batch_paths:
        try:
            os.remove(batch_path)
        except:
            pass
    try:
        os.rmdir(cfg.TEMP_DIR)
    except:
        
        pass

# --- 3. ACTION CLASS DISCOVERY ---
def discover_action_classes(train_labeled):
    """Discover all unique action classes from annotation files"""
    print("ğŸ”� Discovering action classes...")
    action_set = set()
    
    for idx, meta_row in tqdm(train_labeled.iterrows(), total=len(train_labeled), desc="Scanning annotations"):
        _, annotation = load_tracking_annotation(meta_row)
        if not annotation.empty and 'action' in annotation.columns:
            valid_actions = [str(a) for a in annotation['action'].unique() if pd.notna(a) and str(a).strip() != '']
            action_set.update(valid_actions)
    
    actions = sorted(action_set)
    action_to_idx = {a: i for i, a in enumerate(actions)}
    idx_to_action = {i: a for i, a in enumerate(actions)}
    
    print(f"âœ… Detected {len(actions)} action classes: {actions}")
    return actions, action_to_idx, idx_to_action

# --- 4. ENHANCED FEATURE ENGINEERING ---
def get_enhanced_powerful_features():
    """Comprehensive feature set with HIGH-IMPACT additions"""
    
    feature_columns = []
    
    # 1. CORE COORDINATES (All body parts)
    for mouse in ['1', '2']:
        for part in ['body_center', 'nose', 'tail_base', 'left_ear', 'right_ear']:
            feature_columns.extend([
                f'x_{mouse}_{part}', 
                f'y_{mouse}_{part}'
            ])
    
    # 2. ENHANCED VELOCITY & MOVEMENT FEATURES
    for mouse in ['1', '2']:
        for w in [5, 10, 20, 30]:
            feature_columns.extend([
                f'velocity_{mouse}_{w}',
                f'acceleration_{mouse}_{w}',
                f'movement_angle_{mouse}_{w}',
                f'jerk_{mouse}_{w}',  # Rate of acceleration change
            ])
    
    # 3. ENHANCED ROLLING STATISTICS
    for mouse in ['1', '2']:
        for part in ['body_center', 'nose']:
            for w in [10, 20, 30, 50]:
                for stat in ['mean', 'std', 'min', 'max', 'median']:
                    feature_columns.extend([
                        f'x_{mouse}_{part}_{stat}_{w}',
                        f'y_{mouse}_{part}_{stat}_{w}',
                    ])
    
    # 4. ENHANCED SOCIAL INTERACTION FEATURES
    interaction_features = [
        'body_distance', 'nose_distance', 'tail_distance',
        'facing_angle', 'orientation_similarity',
        'velocity_correlation', 'acceleration_correlation',
        'approach_speed', 'relative_speed',
        'proximity_intensity', 'interaction_score',
        'chasing_m1_m2', 'chasing_m2_m1',
        'mutual_movement', 'social_engagement',
        'relative_position_x', 'relative_position_y',
        'heading_alignment', 'movement_synchronization',
        'acceleration_correlation_5', 'acceleration_correlation_10',
        'distance_velocity_ratio', 'angular_velocity_correlation'
    ]
    
    # Add distance-based interaction features
    for threshold in [50, 100, 150]:
        interaction_features.extend([
            f'close_proximity_{threshold}',
            f'interaction_zone_{threshold}'
        ])
    
    feature_columns.extend(interaction_features)
    
    # 5. ENHANCED INDIVIDUAL BEHAVIOR PATTERNS
    for mouse in ['1', '2']:
        feature_columns.extend([
            f'{mouse}_activity_level',
            f'{mouse}_freezing_duration', 
            f'{mouse}_movement_bouts',
            f'{mouse}_exploration_score',
            f'{mouse}_curiosity_index',
            f'{mouse}_movement_consistency',
            f'{mouse}_acceleration_variance',
            f'{mouse}_directional_persistence',
            f'{mouse}_speed_entropy'
        ])
    
    # 6. TEMPORAL PATTERN FEATURES
    for window in [10, 20, 30]:
        feature_columns.extend([
            f'behavior_bout_{window}',
            f'activity_transition_{window}',
            f'temporal_consistency_{window}'
        ])
    
    print(f"ğŸš€ ENHANCED features: {len(feature_columns)} comprehensive features")
    return feature_columns

def calculate_enhanced_features(wide, meta_row, feature_dict):
    """Calculate ENHANCED features with high-impact additions"""
    
    n_frames = len(wide)
    n_features = len(feature_dict)
    features = np.zeros((n_frames, n_features), dtype=np.float32)
    
    def safe_get(col, default=0):
        return wide[col].fillna(default) if col in wide.columns else pd.Series(default, index=wide.index)
    
    # Get all coordinates
    coords = {}
    for mouse in ['1', '2']:
        for part in ['body_center', 'nose', 'tail_base', 'left_ear', 'right_ear']:
            x_col = f'x_{mouse}_{part}'
            y_col = f'y_{mouse}_{part}'
            coords[x_col] = safe_get(x_col)
            coords[y_col] = safe_get(y_col)
    
    # 1. Fill basic coordinates
    for col, data in coords.items():
        if col in feature_dict:
            features[:, feature_dict[col]] = data.values
    
    # 2. ENHANCED Velocity & Movement features
    velocities = {}
    accelerations = {}
    
    for mouse in ['1', '2']:
        x_col = f'x_{mouse}_body_center'
        y_col = f'y_{mouse}_body_center'
        
        if x_col in coords and y_col in coords:
            x_data = coords[x_col]
            y_data = coords[y_col]
            
            for w in [5, 10, 20, 30]:
                # Velocity
                vel_x = x_data.diff().rolling(w, min_periods=1).mean()
                vel_y = y_data.diff().rolling(w, min_periods=1).mean()
                velocity = np.sqrt(vel_x**2 + vel_y**2)
                
                vel_col = f'velocity_{mouse}_{w}'
                velocities[vel_col] = velocity
                if vel_col in feature_dict:
                    features[:, feature_dict[vel_col]] = velocity.values
                
                # Acceleration
                acc_col = f'acceleration_{mouse}_{w}'
                acceleration = velocity.diff().rolling(w, min_periods=1).mean()
                accelerations[acc_col] = acceleration
                if acc_col in feature_dict:
                    features[:, feature_dict[acc_col]] = acceleration.values
                
                # Movement angle
                angle_col = f'movement_angle_{mouse}_{w}'
                if angle_col in feature_dict:
                    movement_angle = np.arctan2(vel_y, vel_x)
                    features[:, feature_dict[angle_col]] = movement_angle.values
    
    # 3. ENHANCED Rolling statistics with more windows
    for mouse in ['1', '2']:
        for part in ['body_center', 'nose']:
            x_col = f'x_{mouse}_{part}'
            y_col = f'y_{mouse}_{part}'
            
            if x_col in coords:
                x_data = coords[x_col]
                y_data = coords[y_col] if y_col in coords else pd.Series(0, index=wide.index)
                
                for w in [10, 20, 30, 50]:
                    for stat, func in [('mean', 'mean'), ('std', 'std'), ('min', 'min'), ('max', 'max'), ('median', 'median')]:
                        x_stat_col = f'{x_col}_{stat}_{w}'
                        y_stat_col = f'{y_col}_{stat}_{w}'
                        
                        if x_stat_col in feature_dict:
                            if func == 'mean':
                                features[:, feature_dict[x_stat_col]] = x_data.rolling(w, min_periods=1).mean().values
                            elif func == 'std':
                                features[:, feature_dict[x_stat_col]] = x_data.rolling(w, min_periods=1).std().fillna(0).values
                            elif func == 'min':
                                features[:, feature_dict[x_stat_col]] = x_data.rolling(w, min_periods=1).min().values
                            elif func == 'max':
                                features[:, feature_dict[x_stat_col]] = x_data.rolling(w, min_periods=1).max().values
                            elif func == 'median':
                                features[:, feature_dict[x_stat_col]] = x_data.rolling(w, min_periods=1).median().values
                        
                        if y_stat_col in feature_dict:
                            if func == 'mean':
                                features[:, feature_dict[y_stat_col]] = y_data.rolling(w, min_periods=1).mean().values
                            elif func == 'std':
                                features[:, feature_dict[y_stat_col]] = y_data.rolling(w, min_periods=1).std().fillna(0).values
                            elif func == 'min':
                                features[:, feature_dict[y_stat_col]] = y_data.rolling(w, min_periods=1).min().values
                            elif func == 'max':
                                features[:, feature_dict[y_stat_col]] = y_data.rolling(w, min_periods=1).max().values
                            elif func == 'median':
                                features[:, feature_dict[y_stat_col]] = y_data.rolling(w, min_periods=1).median().values
    
    # 4. ENHANCED Social interaction features
    # Distance features
    for part in ['body_center', 'nose', 'tail_base']:
        dist_col = f'{part}_distance'
        if dist_col in feature_dict:
            x1, y1 = f'x_1_{part}', f'y_1_{part}'
            x2, y2 = f'x_2_{part}', f'y_2_{part}'
            if x1 in coords and y1 in coords and x2 in coords and y2 in coords:
                distance = np.sqrt((coords[x1] - coords[x2])**2 + (coords[y1] - coords[y2])**2)
                features[:, feature_dict[dist_col]] = distance.values
    
    # ENHANCED: Relative position
    rel_x_col = 'relative_position_x'
    rel_y_col = 'relative_position_y'
    if rel_x_col in feature_dict and rel_y_col in feature_dict:
        body1_x = coords.get('x_1_body_center', 0)
        body2_x = coords.get('x_2_body_center', 0)
        body1_y = coords.get('y_1_body_center', 0)
        body2_y = coords.get('y_2_body_center', 0)
        features[:, feature_dict[rel_x_col]] = (body1_x - body2_x).values
        features[:, feature_dict[rel_y_col]] = (body1_y - body2_y).values
    
    # ENHANCED: Movement synchronization
    sync_col = 'movement_synchronization'
    if sync_col in feature_dict:
        vel1 = velocities.get('velocity_1_10', pd.Series(0, index=wide.index))
        vel2 = velocities.get('velocity_2_10', pd.Series(0, index=wide.index))
        sync = 1.0 - np.abs(vel1 - vel2) / (vel1 + vel2 + 1e-8)
        features[:, feature_dict[sync_col]] = sync.values
    
    # Final NaN handling
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def generate_powerful_features(tracking, annotation, meta_row, feature_columns, action_to_idx=None):
    """Generate powerful features with multi-class labels"""
    
    if tracking is None or len(tracking) == 0:
        return None, None
    
    # Only limit per video, not total
    if len(tracking) > cfg.MAX_FRAMES_PER_VIDEO:
        step = max(1, len(tracking) // cfg.MAX_FRAMES_PER_VIDEO)
        tracking = tracking.iloc[::step].reset_index(drop=True)
    
    # Create pivot table
    tracking["part"] = tracking["mouse_id"].astype(str) + "_" + tracking["bodypart"].astype(str)
    wide = tracking.pivot(index="video_frame", columns="part", values=["x", "y"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.sort_index().fillna(method='ffill').fillna(0)
    
    if wide.empty:
        return None, None
    
    n_frames = len(wide)
    feature_dict = {name: idx for idx, name in enumerate(feature_columns)}
    
    # Calculate powerful features
    features = calculate_enhanced_features(wide, meta_row, feature_dict)
    
    # Generate MULTI-CLASS labels
    labels = np.zeros(n_frames, dtype=int)  # Default to class 0 (no behavior)
    
    if not annotation.empty and 'start_frame' in annotation.columns and 'stop_frame' in annotation.columns and 'action' in annotation.columns:
        for _, row in annotation.iterrows():
            try:
                start, stop, action = int(row["start_frame"]), int(row["stop_frame"]), row["action"]
                if pd.notna(action) and action in action_to_idx:
                    action_idx = action_to_idx[action]
                    # Ensure we don't exceed frame bounds
                    start = min(start, n_frames - 1)
                    stop = min(stop, n_frames)
                    if start < stop:
                        labels[start:stop] = action_idx
            except Exception as e:
                continue
    
    return features, labels

# --- 5. NO-LIMITS DATA COLLECTION ---
def collect_nolimits_data(train_labeled, action_to_idx):
    """Collect ALL available data with smart batching"""
    
    print(f"ğŸš€ Processing ALL {len(train_labeled)} videos with NO LIMITS...")
    powerful_features = get_enhanced_powerful_features()
    
    batch_paths = []
    total_frames = 0
    processed_videos = 0
    batch_data = None
    
    # Process ALL videos
    for idx, meta_row in tqdm(train_labeled.iterrows(), total=len(train_labeled), desc="Processing ALL videos"):
        try:
            tracking, annotation = load_tracking_annotation(meta_row)
            
            if tracking is None or len(tracking) < cfg.MIN_VIDEO_LENGTH:
                continue
            
            features, labels = generate_powerful_features(tracking, annotation, meta_row, powerful_features, action_to_idx)
            
            if features is not None and len(features) > 0:
                processed_videos += 1
                total_frames += len(features)
                
                if batch_data is None:
                    batch_data = (features, labels, np.array([meta_row["video_id"]] * len(features)))
                else:
                    X_old, y_old, groups_old = batch_data
                    X_new = np.vstack([X_old, features])
                    y_new = np.concatenate([y_old, labels])
                    groups_new = np.concatenate([groups_old, [meta_row["video_id"]] * len(features)])
                    batch_data = (X_new, y_new, groups_new)
                
                # Save large batches to disk
                if batch_data[0].shape[0] >= cfg.BATCH_SIZE:
                    batch_num = len(batch_paths) + 1
                    batch_path = save_batch_to_disk(batch_data, batch_num)
                    batch_paths.append(batch_path)
                    print(f"ğŸ’¾ Saved batch {batch_num} with {batch_data[0].shape[0]:,} frames (Total: {total_frames:,})")
                    batch_data = None
                    gc.collect()
                    
        except Exception as e:
            continue
    
    # Save final batch
    if batch_data is not None:
        batch_num = len(batch_paths) + 1
        batch_path = save_batch_to_disk(batch_data, batch_num)
        batch_paths.append(batch_path)
        print(f"ğŸ’¾ Saved final batch {batch_num} with {batch_data[0].shape[0]:,} frames")
    
    print(f"âœ… Processed ALL {processed_videos} quality videos, {total_frames:,} total frames")
    return batch_paths, powerful_features

# --- 6. MEMORY-OPTIMIZED MULTI-CLASS TRAINING ---
def train_multiclass_batch_by_batch(batch_paths, feature_names, num_classes, samples_per_batch=50000):
    """Multi-class training with memory efficiency and ROBUST class handling"""
    
    print("ğŸ�¯ Training MULTI-CLASS BATCH-BY-BATCH")
    
    models = []
    batch_scores = []
    
    # Train on first few batches only (memory safe)
    train_batches = batch_paths[:3]  # Reduced to 3 batches for stability
    
    for batch_idx, batch_path in enumerate(train_batches):
        print(f"\nğŸ”„ Training on Batch {batch_idx + 1}/{len(train_batches)}")
        
        X_batch, y_batch, _ = load_batch_from_disk(batch_path)
        
        # Subsample if too large - reduced for stability
        if len(X_batch) > samples_per_batch:
            indices = np.random.choice(len(X_batch), samples_per_batch, replace=False)
            X_batch = X_batch[indices]
            y_batch = y_batch[indices]
        
        # Split for validation
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_batch, y_batch, test_size=0.2, random_state=cfg.SEED, stratify=y_batch
        )
        
        print(f"   Batch {batch_idx + 1}: Train {X_tr.shape}, Val {X_va.shape}, Classes: {len(np.unique(y_tr))}")
        
        # Feature scaling
        if cfg.USE_SCALING:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)
        
        # ROBUST MULTI-CLASS LightGBM with SIMPLE parameters
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            random_state=cfg.SEED + batch_idx,
            device='cpu',
            n_jobs=-1,
            objective='multiclass',
            num_class=num_classes,
            verbosity=-1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        # Simple training without class weights to avoid KeyError
        model.fit(X_tr, y_tr)
        
        # Quick validation
        y_pred = model.predict(X_va)
        score = f1_score(y_va, y_pred, average="macro")
        
        models.append(model)
        batch_scores.append(score)
        
        print(f"âœ… Batch {batch_idx + 1} Macro-F1: {score:.4f}")
        
        # Clean memory aggressively
        del X_batch, y_batch, X_tr, X_va, y_tr, y_va
        gc.collect()
    
    return models, batch_scores

def train_multiclass_memory_safe(batch_paths, feature_names, num_classes):
    """Memory-safe multi-class training wrapper"""
    
    print("ğŸ›¡ï¸�  Starting MULTI-CLASS MEMORY-SAFE training")
    print(f"ğŸ“¦ Using first {min(3, len(batch_paths))} batches, {num_classes} classes")
    
    try:
        models, scores = train_multiclass_batch_by_batch(batch_paths, feature_names, num_classes)
        
        # Create simple fold info for compatibility
        fold_info = []
        for i, score in enumerate(scores):
            fold_info.append({
                'fold': i + 1,
                'macro_f1': score,
                'num_classes': num_classes
            })
        
        return models, scores, pd.DataFrame(fold_info), feature_names
    except Exception as e:
        print(f"âš ï¸�  Training failed: {e}")
        print("ğŸ”„ Falling back to single batch training...")
        return train_single_batch_fallback(batch_paths, feature_names, num_classes)

def train_single_batch_fallback(batch_paths, feature_names, num_classes):
    """Fallback training with just one batch"""
    print("ğŸ”„ Using SINGLE BATCH FALLBACK...")
    
    if not batch_paths:
        print("â�Œ No batches available for fallback training")
        return [], [], pd.DataFrame(), feature_names
    
    # Use only the first batch
    batch_path = batch_paths[0]
    X_batch, y_batch, _ = load_batch_from_disk(batch_path)
    
    # Take smaller sample for stability
    if len(X_batch) > 20000:
        indices = np.random.choice(len(X_batch), 20000, replace=False)
        X_batch = X_batch[indices]
        y_batch = y_batch[indices]
    
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_batch, y_batch, test_size=0.2, random_state=cfg.SEED, stratify=y_batch
    )
    
    print(f"   Fallback: Train {X_tr.shape}, Val {X_va.shape}, Classes: {len(np.unique(y_tr))}")
    
    # Feature scaling
    if cfg.USE_SCALING:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
    
    # Very simple model for fallback
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=10,
        random_state=cfg.SEED,
        device='cpu',
        n_jobs=-1,
        objective='multiclass',
        num_class=num_classes,
        verbosity=-1,
        max_depth=5
    )
    
    model.fit(X_tr, y_tr)
    
    # Quick validation
    y_pred = model.predict(X_va)
    score = f1_score(y_va, y_pred, average="macro")
    
    print(f"âœ… Fallback training completed with Macro-F1: {score:.4f}")
    
    fold_info = [{'fold': 1, 'macro_f1': score, 'num_classes': num_classes}]
    
    return [model], [score], pd.DataFrame(fold_info), feature_names

# --- 7. TEST PIPELINE AND SUBMISSION GENERATION ---
def generate_test_features_with_mice(test, feature_columns):
    """Generate features for test videos with mouse information"""
    
    print("ğŸ”® Generating test features with mouse info...")
    test_features = []
    test_video_ids = []
    frame_indices = []
    agent_ids = []
    target_ids = []
    
    for idx, meta_row in tqdm(test.iterrows(), total=len(test), desc="Test feature extraction"):
        tracking, _ = load_tracking_annotation(meta_row, mode="test")
        
        if tracking is not None and len(tracking) > 0:
            feats, _ = generate_powerful_features(tracking, pd.DataFrame(), meta_row, feature_columns)
            
            if feats is not None:
                n_frames = feats.shape[0]
                test_features.append(feats)
                test_video_ids.extend([meta_row['video_id']] * n_frames)
                frame_indices.extend(np.arange(n_frames))
                
                # Extract mouse information from tracking data
                unique_mice = sorted(tracking['mouse_id'].unique())
                if len(unique_mice) >= 2:
                    # For social behaviors, use first two mice
                    agent_ids.extend([f'mouse{unique_mice[0]}'] * n_frames)
                    target_ids.extend([f'mouse{unique_mice[1]}'] * n_frames)
                elif len(unique_mice) == 1:
                    # For self behaviors, use same mouse
                    agent_ids.extend([f'mouse{unique_mice[0]}'] * n_frames)
                    target_ids.extend([f'mouse{unique_mice[0]}'] * n_frames)
                else:
                    # Default fallback
                    agent_ids.extend(['mouse1'] * n_frames)
                    target_ids.extend(['mouse2'] * n_frames)
    
    if test_features:
        X_test = np.vstack(test_features)
        print(f"âœ… Test features: {X_test.shape}")
        print(f"ğŸ“Š Mouse distribution - Agents: {np.unique(agent_ids)}, Targets: {np.unique(target_ids)}")
        return X_test, test_video_ids, frame_indices, agent_ids, target_ids
    else:
        print("â�Œ No test features generated")
        return None, None, None, None, None

def infer_agent_target_for_actions(action, default_agent, default_target):
    """Infer agent and target IDs based on the specific action types detected"""
    
    action_lower = action.lower()
    
    # Self-directed behaviors (same agent and target)
    self_behaviors = [
        'biteobject', 'climb', 'dig', 'escape', 'exploreobject', 
        'freeze', 'genitalgroom', 'rear', 'rest', 'run', 'selfgroom'
    ]
    
    # Social behaviors with same agent/target (self-directed social)
    self_social_behaviors = [
        'disengage', 'flinch', 'submit'
    ]
    
    # Social behaviors with different agent/target
    social_behaviors = [
        'allogroom', 'approach', 'attack', 'attemptmount', 'avoid',
        'chase', 'chaseattack', 'defend', 'dominance', 'dominancegroom',
        'dominancemount', 'ejaculate', 'follow', 'huddle', 'intromit',
        'mount', 'reciprocalsniff', 'shepherd', 'sniff', 'sniffbody',
        'sniffface', 'sniffgenital', 'tussle'
    ]
    
    # Check for self behaviors
    for self_behavior in self_behaviors:
        if self_behavior in action_lower:
            return default_agent, default_agent  # Same mouse
    
    # Check for self social behaviors
    for self_social_behavior in self_social_behaviors:
        if self_social_behavior in action_lower:
            return default_agent, default_agent  # Same mouse
    
    # Check for social behaviors  
    for social_behavior in social_behaviors:
        if social_behavior in action_lower:
            return default_agent, default_target  # Different mice
    
    # Default: assume social behavior
    return default_agent, default_target

def create_final_submission_correct(models, test_df, feature_columns, idx_to_action):
    """Create final submission with correct agent/target IDs for all actions"""
    
    print("ğŸ“� Creating final submission with correct agent/target mapping...")
    
    # Generate test features with mouse information
    X_test, test_video_ids, frame_indices, agent_ids, target_ids = generate_test_features_with_mice(test_df, feature_columns)
    
    if X_test is None:
        print("â�Œ No test data available for submission")
        return pd.DataFrame()
    
    # Ensemble predictions
    print("ğŸ¤– Generating ensemble predictions...")
    all_preds = []
    for i, model in enumerate(models):
        print(f"   Model {i+1}/{len(models)} predicting...")
        preds = model.predict(X_test)
        all_preds.append(preds)
    
    # Majority voting ensemble
    all_preds = np.stack(all_preds)
    y_test_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds
    )
    
    # Convert predictions to action names
    pred_actions = [idx_to_action.get(pred_idx, "unknown") for pred_idx in y_test_pred]
    
    # Create submission blocks with proper event detection
    submission = []
    row_id_counter = 0
    
    print("ğŸ“¦ Creating submission blocks with event detection...")
    
    # Group by video and detect continuous events
    unique_videos = np.unique(test_video_ids)
    
    for video_id in unique_videos:
        video_mask = np.array(test_video_ids) == video_id
        video_frames = np.array(frame_indices)[video_mask]
        video_actions = np.array(pred_actions)[video_mask]
        video_agents = np.array(agent_ids)[video_mask]
        video_targets = np.array(target_ids)[video_mask]
        
        if len(video_frames) == 0:
            continue
        
        # Sort by frame
        sort_idx = np.argsort(video_frames)
        video_frames = video_frames[sort_idx]
        video_actions = video_actions[sort_idx]
        video_agents = video_agents[sort_idx]
        video_targets = video_targets[sort_idx]
        
        # Detect continuous events
        current_action = None
        start_frame = None
        current_agent = None
        current_target = None
        
        for i, (frame, action, agent, target) in enumerate(zip(video_frames, video_actions, video_agents, video_targets)):
            if action != current_action:
                # End previous block if it exists
                if current_action is not None and start_frame is not None:
                    # Ensure minimum duration of 3 frames
                    if frame - start_frame >= 3:
                        submission.append({
                            'row_id': row_id_counter,
                            'video_id': video_id,
                            'agent_id': current_agent,
                            'target_id': current_target,
                            'action': current_action,
                            'start_frame': start_frame,
                            'stop_frame': frame - 1
                        })
                        row_id_counter += 1
                
                # Start new block
                current_action = action
                start_frame = frame
                current_agent = agent
                current_target = target
        
        # Final block for the video
        if current_action is not None and start_frame is not None:
            # Ensure minimum duration
            if video_frames[-1] - start_frame >= 3:
                submission.append({
                    'row_id': row_id_counter,
                    'video_id': video_id,
                    'agent_id': current_agent,
                    'target_id': current_target,
                    'action': current_action,
                    'start_frame': start_frame,
                    'stop_frame': video_frames[-1]
                })
                row_id_counter += 1
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(submission)
    
    if len(submission_df) == 0:
        print("â�Œ No events detected in submission")
        return pd.DataFrame()
    
    # Ensure correct column order
    required_columns = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    for col in required_columns:
        if col not in submission_df.columns:
            submission_df[col] = None
    
    submission_df = submission_df[required_columns]
    
    # Apply agent/target correction based on action types
    print("ğŸ”§ Applying agent/target correction based on action types...")
    for idx, row in submission_df.iterrows():
        action = row['action']
        current_agent = row['agent_id']
        current_target = row['target_id']
        
        # Get correct agent/target pairing for this action
        corrected_agent, corrected_target = infer_agent_target_for_actions(action, current_agent, current_target)
        
        submission_df.at[idx, 'agent_id'] = corrected_agent
        submission_df.at[idx, 'target_id'] = corrected_target
    
    # Save submission
    submission_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"âœ… Submission saved: {submission_path}")
    print(f"ğŸ“Š Submission shape: {submission_df.shape}")
    print(f"ğŸ�¯ Actions detected: {len(submission_df)} total events")
    
    # Show action distribution
    action_counts = submission_df['action'].value_counts()
    print(f"ğŸ“‹ Action distribution (top 10):")
    for action, count in action_counts.head(10).items():
        print(f"   {action}: {count} events")
    
    # Show agent/target distribution
    print(f"ğŸ”� Agent/Target distribution:")
    print(f"   Agents: {submission_df['agent_id'].value_counts().to_dict()}")
    print(f"   Targets: {submission_df['target_id'].value_counts().to_dict()}")
    
    # Show self vs social behavior breakdown
    self_behaviors = submission_df[submission_df['agent_id'] == submission_df['target_id']]
    social_behaviors = submission_df[submission_df['agent_id'] != submission_df['target_id']]
    
    print(f"ğŸ“ˆ Behavior breakdown:")
    print(f"   Self-behaviors: {len(self_behaviors)} events")
    print(f"   Social behaviors: {len(social_behaviors)} events")
    
    return submission_df

# --- 8. MAIN EXECUTION ---
def main_enhanced():
    print("ğŸš€ Starting ENHANCED MABe MULTI-CLASS Pipeline...")
    
    try:
        # Load data
        train, test = load_data()
        train = process_metadata(train)
        train, train_labeled = create_features(train)
        
        # Discover action classes
        actions, action_to_idx, idx_to_action = discover_action_classes(train_labeled)
        
        if len(actions) == 0:
            print("â�Œ No actions found. Using default behavior.")
            actions = ['behavior']
            action_to_idx = {'behavior': 0}
            idx_to_action = {0: 'behavior'}
        
        print(f"ğŸ�¯ Training for {len(actions)} actions")
        
        # Collect data
        print(f"\nğŸ”„ Collecting data from {len(train_labeled)} videos...")
        batch_paths, feature_columns = collect_nolimits_data(train_labeled, action_to_idx)
        
        if not batch_paths:
            print("â�Œ No data collected. Exiting.")
            return None, None, None, None, None, None
        
        # Train multi-class model
        print(f"\nğŸ�¯ Training MULTI-CLASS model with {len(actions)} classes...")
        models, scores, fold_df, selected_features = train_multiclass_memory_safe(
            batch_paths, feature_columns, len(actions)
        )
        
        print(f"\nğŸ�‰ MULTI-CLASS PIPELINE COMPLETED!")
        if scores:
            print(f"ğŸ�† Best Macro-F1: {np.max(scores):.4f}")
            print(f"ğŸ“ˆ Average Macro-F1: {np.mean(scores):.4f}")
            
            if np.mean(scores) >= 0.80:
                print("ğŸ�‰ SUCCESS! TARGET Macro-F1 > 0.80 ACHIEVED!")
            elif np.mean(scores) >= 0.70:
                print("ğŸ“ˆ Good progress! Close to target.")
            elif np.mean(scores) >= 0.60:
                print("ğŸ’ª Solid baseline - ready for competition!")
            else:
                print("ğŸ”„ Needs optimization - but working pipeline!")
        
        # Create submission with correct agent/target mapping
        print(f"\nğŸ“¤ Creating competition submission...")
        submission_df = create_final_submission_correct(models, test, feature_columns, idx_to_action)
        
        # Cleanup
        cleanup_temp_files(batch_paths)
        
        return models, scores, selected_features, fold_df, submission_df, actions
        
    except Exception as e:
        print(f"ğŸ’¥ Critical error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

# Run the enhanced pipeline
if __name__ == "__main__":
    print("=" * 60)
    print("ğŸš€ ENHANCED MABe MULTI-CLASS PIPELINE")
    print("=" * 60)
    
    models, scores, features, fold_df, submission_df, actions = main_enhanced()
    
    if scores and np.mean(scores) >= 0.80:
        print("\nğŸ�‰ğŸ�‰ğŸ�‰ CONGRATULATIONS! TARGET ACHIEVED! ğŸ�‰ğŸ�‰ğŸ�‰")
    elif scores:
        print(f"\nğŸ“Š Current Performance: Macro-F1 = {np.mean(scores):.4f}")
    
    if submission_df is not None:
        print(f"âœ… Submission ready with {len(submission_df)} detected events")
        print(f"ğŸ“‹ Actions in submission: {submission_df['action'].value_counts()}")  




