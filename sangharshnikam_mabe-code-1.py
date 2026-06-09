import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import json
import os
import warnings
import gc
warnings.filterwarnings('ignore')

# Machine Learning
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

print("✓ Libraries imported")

# Configuration
class Config:
    # Paths - ADAPTED TO YOUR DATA
    DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection'  # Update this!
    
    # Mode
    MODE = 'submit'  # 'submit'
    VERBOSE = True
    
    # Feature Engineering
    VELOCITY_WINDOWS = [5, 10, 20]
    ROLLING_WINDOWS = [10, 20, 30]
    
    # Model Parameters
    THRESHOLD = 0.25
    DOWNSAMPLE = 2  # Use every Nth frame
    
    # LightGBM
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 200,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # XGBoost
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 200,
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': 42
    }

config = Config()



print("="*60)
print("VERIFYING DATA STRUCTURE")
print("="*60)

# Check data path
print(f"\nChecking: {config.DATA_PATH}")
if os.path.exists(config.DATA_PATH):
    print("✓ Data path found!")
    
    # List contents
    contents = os.listdir(config.DATA_PATH)
    print(f"\nContents ({len(contents)} items):")
    for item in sorted(contents):
        print(f"  - {item}")
    
    # Check train_tracking labs
    train_tracking_path = f"{config.DATA_PATH}/train_tracking"
    if os.path.exists(train_tracking_path):
        labs = os.listdir(train_tracking_path)
        print(f"\n✓ Found {len(labs)} labs in train_tracking:")
        print(f"  Labs: {labs[:5]}...")  # Show first 5
    
    # Load metadata
    train = pd.read_csv(f'{config.DATA_PATH}/train.csv')
    test = pd.read_csv(f'{config.DATA_PATH}/test.csv')
    
    print(f"\n✓ train.csv loaded: {train.shape}")
    print(f"✓ test.csv loaded: {test.shape}")
    
    print("\nTrain columns:", train.columns.tolist())
    
else:
    print("❌ Data path not found!")
    print("Available inputs:")
    print(os.listdir('/kaggle/input/'))
    print("\n⚠️  Please update config.DATA_PATH to match your input folder name")


print("\n" + "="*60)
print("EXPLORING SAMPLE VIDEO")
print("="*60)

# Filter out MABe22 (no annotations)
train_clean = train[~train['lab_id'].str.startswith('MABe22')].reset_index(drop=True)
print(f"Train videos (clean): {len(train_clean)}")

# Load first video
sample_row = train_clean.iloc[0]
print(f"\nSample video:")
print(f"  Video ID: {sample_row['video_id']}")
print(f"  Lab: {sample_row['lab_id']}")
print(f"  Behaviors: {sample_row['behaviors_labeled']}")

# Load tracking data
lab_id = sample_row['lab_id']
video_id = sample_row['video_id']
tracking_path = f"{config.DATA_PATH}/train_tracking/{lab_id}/{video_id}.parquet"

if os.path.exists(tracking_path):
    vid = pd.read_parquet(tracking_path)
    print(f"\n✓ Tracking data loaded: {vid.shape}")
    print(f"  Columns: {vid.columns.tolist()}")
    print(f"\n  First few rows:")
    print(vid.head())
    
    print(f"\n  Unique mice: {vid['mouse_id'].unique()}")
    print(f"  Unique body parts: {vid['bodypart'].unique()}")
    print(f"  Frame range: {vid['video_frame'].min()} to {vid['video_frame'].max()}")
    
    # Check for missing values
    missing = vid[['x', 'y']].isna().sum()
    print(f"\n  Missing values: {missing.sum()} / {len(vid)*2} ({missing.sum()/(len(vid)*2)*100:.1f}%)")
else:
    print(f"❌ Tracking file not found at: {tracking_path}")

# Load annotation
annot_path = f"{config.DATA_PATH}/train_annotation/{lab_id}/{video_id}.parquet"
if os.path.exists(annot_path):
    annot = pd.read_parquet(annot_path)
    print(f"\n✓ Annotation loaded: {annot.shape}")
    print(f"  Columns: {annot.columns.tolist()}")
    print(f"\n  Sample annotations:")
    print(annot.head())
    print(f"\n  Unique actions: {annot['action'].unique()}")
else:
    print(f"❌ Annotation not found")

print("\n✓ Data exploration complete!")


print("\n" + "="*60)
print("DEFINING FEATURE ENGINEERING")
print("="*60)

class MotionFeatures:
    """Extract velocity and acceleration features"""
    
    @staticmethod
    def compute_velocity(coords, window=5):
        """Compute velocity magnitude"""
        velocity = np.zeros(len(coords))
        if len(coords) > window:
            velocity[window:] = np.linalg.norm(
                coords[window:] - coords[:-window], axis=1
            ) / window
        return velocity
    
    @staticmethod
    def compute_acceleration(coords, window=5):
        """Compute acceleration magnitude"""
        vel = MotionFeatures.compute_velocity(coords, window)
        accel = np.zeros_like(vel)
        if len(vel) > window:
            accel[window:] = (vel[window:] - vel[:-window]) / window
        return accel

class GeometricFeatures:
    """Extract geometric features"""
    
    @staticmethod
    def compute_angle(p1, p2, p3):
        """Compute angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        cos_angle = np.sum(v1 * v2, axis=1)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.arccos(cos_angle)
    
    @staticmethod
    def compute_distance(coords1, coords2):
        """Euclidean distance"""
        return np.sqrt(np.sum((coords1 - coords2)**2, axis=1))

def extract_single_mouse_features(mouse_df, config):
    """
    Extract features for single mouse
    
    Args:
        mouse_df: DataFrame with columns (bodypart, xy) for one mouse
        config: Configuration object
    
    Returns:
        DataFrame with features
    """
    features = {}
    body_parts = mouse_df.columns.get_level_values(0).unique().tolist()
    
    # 1. POSITION FEATURES (relative to center)
    if 'center' in body_parts:
        center = mouse_df['center'].values
    elif 'spine_middle' in body_parts:
        center = mouse_df['spine_middle'].values
    else:
        # Use mean of all parts
        center = mouse_df.values.reshape(len(mouse_df), -1, 2).mean(axis=1)
    
    for part in body_parts:
        if part in mouse_df.columns:
            coords = mouse_df[part].values
            rel_coords = coords - center
            features[f'{part}_rel_dist'] = np.linalg.norm(rel_coords, axis=1)
    
    # 2. MOTION FEATURES
    motion = MotionFeatures()
    for part in body_parts[:3]:  # Use top 3 parts to save memory
        if part in mouse_df.columns:
            coords = mouse_df[part].values
            for w in config.VELOCITY_WINDOWS:
                features[f'{part}_vel_{w}'] = motion.compute_velocity(coords, w)
    
    # 3. GEOMETRIC FEATURES
    geom = GeometricFeatures()
    
    # Body angle (if nose, spine_middle, tail_base exist)
    if all(p in body_parts for p in ['nose', 'spine_middle', 'tail_base']):
        angle = geom.compute_angle(
            mouse_df['nose'].values,
            mouse_df['spine_middle'].values,
            mouse_df['tail_base'].values
        )
        features['body_angle'] = angle
    
    # 4. DISTANCE FEATURES
    key_parts = [p for p in ['nose', 'spine_middle', 'tail_base'] if p in body_parts]
    for i, p1 in enumerate(key_parts):
        for p2 in key_parts[i+1:]:
            dist = geom.compute_distance(mouse_df[p1].values, mouse_df[p2].values)
            features[f'dist_{p1}_{p2}'] = dist
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features, index=mouse_df.index)
    
    # 5. ROLLING AGGREGATIONS
    for col in features_df.columns:
        for w in config.ROLLING_WINDOWS[:2]:  # Use 2 windows to save memory
            features_df[f'{col}_mean_{w}'] = features_df[col].rolling(
                window=w, min_periods=1, center=True
            ).mean()
    
    return features_df

def extract_pair_features(mouse_pair_df, config):
    """
    Extract features for mouse pair
    
    Args:
        mouse_pair_df: DataFrame with columns (A/B, bodypart, xy)
        config: Configuration object
    
    Returns:
        DataFrame with features
    """
    features = {}
    
    mouse_A = mouse_pair_df['A']
    mouse_B = mouse_pair_df['B']
    
    parts_A = mouse_A.columns.get_level_values(0).unique().tolist()
    parts_B = mouse_B.columns.get_level_values(0).unique().tolist()
    
    geom = GeometricFeatures()
    
    # 1. INTER-MOUSE DISTANCES
    key_parts = [p for p in ['nose', 'center', 'tail_base'] 
                 if p in parts_A and p in parts_B]
    
    for pa in key_parts:
        for pb in key_parts:
            dist = geom.compute_distance(mouse_A[pa].values, mouse_B[pb].values)
            features[f'dist_{pa}_{pb}'] = dist
    
    # 2. MINIMUM DISTANCE (closest approach)
    all_dists = []
    for pa in parts_A:
        if pa in mouse_A.columns:
            for pb in parts_B:
                if pb in mouse_B.columns:
                    dist = geom.compute_distance(mouse_A[pa].values, mouse_B[pb].values)
                    all_dists.append(dist)
    
    if all_dists:
        all_dists = np.array(all_dists).T
        features['min_dist'] = np.min(all_dists, axis=1)
        features['mean_dist'] = np.mean(all_dists, axis=1)
    
    # 3. RELATIVE MOTION
    if 'nose' in parts_A and 'nose' in parts_B:
        motion = MotionFeatures()
        for w in config.VELOCITY_WINDOWS[:2]:
            vel_A = motion.compute_velocity(mouse_A['nose'].values, w)
            vel_B = motion.compute_velocity(mouse_B['nose'].values, w)
            features[f'vel_diff_{w}'] = np.abs(vel_A - vel_B)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features, index=mouse_pair_df.index)
    
    # 4. ROLLING AGGREGATIONS
    for col in features_df.columns:
        for w in config.ROLLING_WINDOWS[:2]:
            features_df[f'{col}_mean_{w}'] = features_df[col].rolling(
                window=w, min_periods=1, center=True
            ).mean()
    
    return features_df

print("✓ Feature engineering functions defined")
print("  - Motion: velocity, acceleration")
print("  - Geometric: angles, distances")
print("  - Temporal: rolling aggregations")



print("\n" + "="*60)
print("DEFINING VIDEO PROCESSING")
print("="*60)

def load_and_pivot_video(lab_id, video_id, traintest='train'):
    """Load video and pivot to wide format"""
    path = f"{config.DATA_PATH}/{traintest}_tracking/{lab_id}/{video_id}.parquet"
    
    try:
        vid = pd.read_parquet(path)
        
        # Pivot: one row per frame
        pvid = vid.pivot(
            columns=['mouse_id', 'bodypart'],
            index='video_frame',
            values=['x', 'y']
        )
        
        # Reorder to (mouse_id, bodypart, xy)
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        
        return pvid
    except Exception as e:
        if config.VERBOSE:
            print(f"Error loading {video_id}: {e}")
        return None

def process_video(row, traintest='train'):
    """
    Process one video and extract all batches
    
    Returns:
        List of (switch, features, meta, labels/actions)
    """
    lab_id = row['lab_id']
    video_id = row['video_id']
    
    # Skip MABe22
    if str(lab_id).startswith('MABe22'):
        return []
    
    # Load video
    pvid = load_and_pivot_video(lab_id, video_id, traintest)
    if pvid is None or len(pvid) == 0:
        return []
    
    # Normalize to cm
    pvid = pvid / row['pix_per_cm_approx']
    
    # Parse behaviors
    try:
        behaviors = json.loads(row['behaviors_labeled'])
        behaviors = sorted(list({b.replace("'", "") for b in behaviors}))
        behaviors = [b.split(',') for b in behaviors]
        behaviors_df = pd.DataFrame(behaviors, columns=['agent', 'target', 'action'])
    except:
        return []
    
    results = []
    
    # ==== SINGLE MOUSE BEHAVIORS ====
    single_behaviors = behaviors_df[behaviors_df['target'] == 'self']
    
    for mouse_str in single_behaviors['agent'].unique():
        try:
            mouse_id = int(mouse_str.replace('mouse', ''))
            
            if mouse_id not in pvid.columns.get_level_values(0):
                continue
            
            mouse_data = pvid[mouse_id]
            
            # Extract features
            features = extract_single_mouse_features(mouse_data, config)
            
            # Downsample for training
            if traintest == 'train' and config.DOWNSAMPLE > 1:
                features = features.iloc[::config.DOWNSAMPLE]
            
            # Meta
            meta = pd.DataFrame({
                'video_id': video_id,
                'agent_id': mouse_str,
                'target_id': 'self',
                'video_frame': features.index
            })
            
            actions = single_behaviors[single_behaviors['agent'] == mouse_str]['action'].unique()
            
            if traintest == 'train':
                # Load annotations
                annot_path = f"{config.DATA_PATH}/train_annotation/{lab_id}/{video_id}.parquet"
                try:
                    annot = pd.read_parquet(annot_path)
                    
                    # Create labels
                    labels = pd.DataFrame(0, index=features.index, columns=actions)
                    
                    annot_sub = annot[(annot['agent_id'] == mouse_id) & (annot['target_id'] == mouse_id)]
                    for _, arow in annot_sub.iterrows():
                        if arow['action'] in actions:
                            start = arow['start_frame']
                            stop = arow['stop_frame']
                            mask = (labels.index >= start) & (labels.index <= stop)
                            labels.loc[mask, arow['action']] = 1
                    
                    results.append(('single', features, meta, labels))
                except:
                    pass
            else:
                results.append(('single', features, meta, actions))
        except:
            pass
    
    # ==== MOUSE PAIR BEHAVIORS ====
    pair_behaviors = behaviors_df[behaviors_df['target'] != 'self']
    
    if len(pair_behaviors) > 0:
        available_mice = pvid.columns.get_level_values(0).unique()
        
        for agent_id, target_id in [(a, b) for a in available_mice for b in available_mice if a != b]:
            agent_str = f'mouse{agent_id}'
            target_str = f'mouse{target_id}'
            
            actions = pair_behaviors[
                (pair_behaviors['agent'] == agent_str) & 
                (pair_behaviors['target'] == target_str)
            ]['action'].unique()
            
            if len(actions) == 0:
                continue
            
            try:
                mouse_pair = pd.concat([pvid[agent_id], pvid[target_id]], axis=1, keys=['A', 'B'])
                
                # Extract features
                features = extract_pair_features(mouse_pair, config)
                
                # Downsample
                if traintest == 'train' and config.DOWNSAMPLE > 1:
                    features = features.iloc[::config.DOWNSAMPLE]
                
                # Meta
                meta = pd.DataFrame({
                    'video_id': video_id,
                    'agent_id': agent_str,
                    'target_id': target_str,
                    'video_frame': features.index
                })
                
                if traintest == 'train':
                    annot_path = f"{config.DATA_PATH}/train_annotation/{lab_id}/{video_id}.parquet"
                    try:
                        annot = pd.read_parquet(annot_path)
                        
                        labels = pd.DataFrame(0, index=features.index, columns=actions)
                        
                        annot_sub = annot[(annot['agent_id'] == agent_id) & (annot['target_id'] == target_id)]
                        for _, arow in annot_sub.iterrows():
                            if arow['action'] in actions:
                                start = arow['start_frame']
                                stop = arow['stop_frame']
                                mask = (labels.index >= start) & (labels.index <= stop)
                                labels.loc[mask, arow['action']] = 1
                        
                        results.append(('pair', features, meta, labels))
                    except:
                        pass
                else:
                    results.append(('pair', features, meta, actions))
            except:
                pass
    
    return results

print("✓ Video processing functions defined")

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print("\nReady to process videos and train models!")
print("\nNext: Run training cells to build the model")


print("="*60)
print("PROCESSING TRAINING DATA")
print("="*60)

# Group by body parts tracked
body_parts_groups = train_clean['body_parts_tracked'].unique()
print(f"Found {len(body_parts_groups)} body part configurations")

# Storage for training data
train_data = {}

for bp_idx, bp_str in enumerate(body_parts_groups):
    print(f"\n[{bp_idx+1}/{len(body_parts_groups)}] Processing: {bp_str[:60]}...")
    
    subset = train_clean[train_clean['body_parts_tracked'] == bp_str]
    print(f"  Videos: {len(subset)}")
    
    single_data = {'features': [], 'meta': [], 'labels': []}
    pair_data = {'features': [], 'meta': [], 'labels': []}
    
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="  Processing", leave=False):
        results = process_video(row, 'train')
        
        for switch, feats, meta, labels in results:
            if switch == 'single':
                single_data['features'].append(feats)
                single_data['meta'].append(meta)
                single_data['labels'].append(labels)
            else:
                pair_data['features'].append(feats)
                pair_data['meta'].append(meta)
                pair_data['labels'].append(labels)
    
    train_data[bp_str] = {
        'single': single_data,
        'pair': pair_data
    }
    
    print(f"  ✓ Single batches: {len(single_data['features'])}")
    print(f"  ✓ Pair batches: {len(pair_data['features'])}")
    
    # Free memory
    gc.collect()

print(f"\n✓ All training data processed!")
print(f"Total body part groups: {len(train_data)}")


print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

trained_models = {}

for bp_idx, (bp_str, data) in enumerate(train_data.items()):
    print(f"\n[{bp_idx+1}/{len(train_data)}] Training for: {bp_str[:60]}...")
    
    # ==== TRAIN SINGLE MOUSE MODELS ====
    if len(data['single']['features']) > 0:
        print("\n  [SINGLE MOUSE]")
        
        # Concatenate
        X = pd.concat(data['single']['features'], ignore_index=True)
        meta = pd.concat(data['single']['meta'], ignore_index=True)
        labels = pd.concat(data['single']['labels'], ignore_index=True)
        
        print(f"    Shape: {X.shape}")
        print(f"    Actions: {list(labels.columns)}")
        
        single_models = {}
        
        for action in labels.columns:
            mask = ~labels[action].isna()
            
            if mask.sum() < 50:
                continue
            
            X_action = X[mask].fillna(0).values
            y_action = labels[action][mask].astype(int).values
            
            pos_rate = y_action.mean()
            
            # Train LightGBM
            model_lgbm = lgb.LGBMClassifier(**config.LGBM_PARAMS)
            model_lgbm.fit(X_action, y_action)
            
            # Train XGBoost
            model_xgb = xgb.XGBClassifier(**config.XGB_PARAMS)
            model_xgb.fit(X_action, y_action)
            
            single_models[action] = {
                'lgbm': model_lgbm,
                'xgb': model_xgb
            }
            
            print(f"      ✓ {action}: {mask.sum():,} samples, {pos_rate:.3f} pos")
        
        trained_models[f'{bp_str}__single'] = single_models
        
        # Free memory
        del X, meta, labels
        gc.collect()
    
    # ==== TRAIN PAIR MODELS ====
    if len(data['pair']['features']) > 0:
        print("\n  [MOUSE PAIRS]")
        
        # Concatenate
        X = pd.concat(data['pair']['features'], ignore_index=True)
        meta = pd.concat(data['pair']['meta'], ignore_index=True)
        labels = pd.concat(data['pair']['labels'], ignore_index=True)
        
        print(f"    Shape: {X.shape}")
        print(f"    Actions: {list(labels.columns)}")
        
        pair_models = {}
        
        for action in labels.columns:
            mask = ~labels[action].isna()
            
            if mask.sum() < 50:
                continue
            
            X_action = X[mask].fillna(0).values
            y_action = labels[action][mask].astype(int).values
            
            pos_rate = y_action.mean()
            
            # Train LightGBM
            model_lgbm = lgb.LGBMClassifier(**config.LGBM_PARAMS)
            model_lgbm.fit(X_action, y_action)
            
            # Train XGBoost
            model_xgb = xgb.XGBClassifier(**config.XGB_PARAMS)
            model_xgb.fit(X_action, y_action)
            
            pair_models[action] = {
                'lgbm': model_lgbm,
                'xgb': model_xgb
            }
            
            print(f"      ✓ {action}: {mask.sum():,} samples, {pos_rate:.3f} pos")
        
        trained_models[f'{bp_str}__pair'] = pair_models
        
        # Free memory
        del X, meta, labels
        gc.collect()

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Total model groups: {len(trained_models)}")


def predict_multiclass(preds_df, meta_df, threshold=0.25):
    """Convert probabilities to submission format"""
    if len(preds_df.columns) == 0:
        return pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    
    # Get most probable action per frame
    max_prob = preds_df.max(axis=1).values
    max_idx = preds_df.values.argmax(axis=1)
    
    # Apply threshold
    action_idx = np.where(max_prob >= threshold, max_idx, -1)
    action_series = pd.Series(action_idx, index=meta_df['video_frame'].values)
    
    # Find changes
    changes = (action_series != action_series.shift(1))
    change_frames = action_series[changes]
    change_meta = meta_df[changes.values].reset_index(drop=True)
    
    # Build submission
    submissions = []
    
    for i in range(len(change_frames) - 1):
        action_idx = change_frames.iloc[i]
        
        if action_idx >= 0:  # Valid action
            video_id = change_meta.iloc[i]['video_id']
            agent_id = change_meta.iloc[i]['agent_id']
            target_id = change_meta.iloc[i]['target_id']
            action = preds_df.columns[action_idx]
            start_frame = change_frames.index[i]
            
            # Check if next change is same video/agent/target
            next_video = change_meta.iloc[i + 1]['video_id']
            next_agent = change_meta.iloc[i + 1]['agent_id']
            next_target = change_meta.iloc[i + 1]['target_id']
            
            if video_id == next_video and agent_id == next_agent and target_id == next_target:
                stop_frame = change_frames.index[i + 1]
            else:
                # End of video/agent/target
                stop_frame = meta_df[meta_df['video_id'] == video_id]['video_frame'].max() + 1
            
            submissions.append({
                'video_id': video_id,
                'agent_id': agent_id,
                'target_id': target_id,
                'action': action,
                'start_frame': start_frame,
                'stop_frame': stop_frame
            })
    
    return pd.DataFrame(submissions)


print("\n" + "="*60)
if config.MODE == 'submit':
    print("PROCESSING TEST DATA & MAKING PREDICTIONS")
    dataset = test
else:
    print("VALIDATION MODE - Using last 20% of train")
    val_split = int(len(train_clean) * 0.8)
    dataset = train_clean.iloc[val_split:]
    print(f"Validation videos: {len(dataset)}")

print("="*60)

all_predictions = []

# Process by body parts
body_parts_groups = dataset['body_parts_tracked'].unique()

for bp_idx, bp_str in enumerate(body_parts_groups):
    print(f"\n[{bp_idx+1}/{len(body_parts_groups)}] Predicting: {bp_str[:60]}...")
    
    subset = dataset[dataset['body_parts_tracked'] == bp_str]
    print(f"  Videos: {len(subset)}")
    
    # Check if we have models for this body part configuration
    single_model_key = f'{bp_str}__single'
    pair_model_key = f'{bp_str}__pair'
    
    has_single = single_model_key in trained_models
    has_pair = pair_model_key in trained_models
    
    if not has_single and not has_pair:
        print(f"  ⚠️  No trained models for this configuration")
        continue
    
    # Process each video
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="  Videos", leave=False):
        
        if config.MODE == 'submit':
            results = process_video(row, 'test')
        else:
            results = process_video(row, 'train')
        
        for switch, features, meta, actions_or_labels in results:
            
            # Get actions to predict
            if config.MODE == 'submit':
                actions = actions_or_labels
            else:
                actions = actions_or_labels.columns.tolist()
            
            # Select models
            if switch == 'single' and has_single:
                models = trained_models[single_model_key]
            elif switch == 'pair' and has_pair:
                models = trained_models[pair_model_key]
            else:
                continue
            
            # Make predictions
            predictions = pd.DataFrame(index=features.index)
            
            for action in actions:
                if action in models:
                    # Ensemble: 50% LightGBM + 50% XGBoost
                    X = features.fillna(0).values
                    
                    try:
                        pred_lgbm = models[action]['lgbm'].predict_proba(X)[:, 1]
                    except:
                        pred_lgbm = np.zeros(len(X))
                    
                    try:
                        pred_xgb = models[action]['xgb'].predict_proba(X)[:, 1]
                    except:
                        pred_xgb = np.zeros(len(X))
                    
                    predictions[action] = 0.5 * pred_lgbm + 0.5 * pred_xgb
            
            # Convert to submission format
            if len(predictions.columns) > 0:
                submission_part = predict_multiclass(predictions, meta, config.THRESHOLD)
                all_predictions.append(submission_part)
    
    print(f"  ✓ Predictions: {sum(len(p) for p in all_predictions)}")

# Combine all predictions
if len(all_predictions) > 0:
    submission = pd.concat(all_predictions, ignore_index=True)
else:
    print("\n⚠️  No predictions generated!")
    submission = pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

print(f"\n✓ Total predictions: {len(submission)}")


print("\n" + "="*60)
print("POST-PROCESSING SUBMISSION")
print("="*60)

# Rule 1: Remove invalid predictions (start >= stop)
original_len = len(submission)
submission = submission[submission['start_frame'] < submission['stop_frame']]
if len(submission) < original_len:
    print(f"  Removed {original_len - len(submission)} invalid predictions")

# Rule 2: Remove overlapping predictions for same agent/target
cleaned = []
for key, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
    group = group.sort_values('start_frame').reset_index(drop=True)
    
    keep = []
    last_stop = -1
    
    for idx, row in group.iterrows():
        if row['start_frame'] >= last_stop:
            keep.append(idx)
            last_stop = row['stop_frame']
    
    cleaned.append(group.loc[keep])

submission = pd.concat(cleaned, ignore_index=True)

print(f"✓ After deduplication: {len(submission)} predictions")

# Rule 3: Fill missing videos
predicted_videos = set(submission['video_id'].unique())
all_videos = set(dataset['video_id'].unique())
missing_videos = all_videos - predicted_videos

if len(missing_videos) > 0:
    print(f"⚠️  Filling {len(missing_videos)} videos with no predictions")
    
    dummy_preds = []
    for video_id in missing_videos:
        row = dataset[dataset['video_id'] == video_id].iloc[0]
        
        try:
            behaviors = json.loads(row['behaviors_labeled'])
            if len(behaviors) > 0:
                first_behavior = behaviors[0].replace("'", "").split(',')
                dummy_preds.append({
                    'video_id': video_id,
                    'agent_id': first_behavior[0],
                    'target_id': first_behavior[1],
                    'action': first_behavior[2],
                    'start_frame': 0,
                    'stop_frame': 100
                })
        except:
            pass
    
    if dummy_preds:
        dummy_df = pd.DataFrame(dummy_preds)
        submission = pd.concat([submission, dummy_df], ignore_index=True)

print(f"✓ Final submission: {len(submission)} predictions")



print("\n" + "="*60)
print("SAVING SUBMISSION")
print("="*60)

# Add row_id and save
submission = submission.reset_index(drop=True)
submission.index.name = 'row_id'

submission.to_csv('submission.csv')

print("✓ Saved to submission.csv")
print(f"\nSubmission shape: {submission.shape}")
print(f"\nFirst 10 rows:")
print(submission.head(10))

print(f"\nAction distribution:")
action_counts = submission['action'].value_counts()
for action, count in action_counts.items():
    print(f"  {action}: {count}")

print(f"\nVideos covered: {submission['video_id'].nunique()} / {len(dataset)}")

