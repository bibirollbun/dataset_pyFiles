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
from sklearn.utils.class_weight import compute_class_weight

print("✓ Libraries imported")

class Config:
    # Paths
    DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection'
    
    # Mode
    MODE = 'submit'
    VERBOSE = True
    
    # Feature Engineering
    VELOCITY_WINDOWS = [5, 10, 20]
    ROLLING_WINDOWS = [10, 20]
    
    # Model Parameters
    THRESHOLD = 0.25
    DOWNSAMPLE = 1.25  # Use 80% of data
    MIN_SAMPLES_PER_ACTION = 50
    
    # LightGBM
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 4,
        'min_child_samples': 15,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 150,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # XGBoost
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 150,
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': 42
    }

config = Config()

print("\n" + "="*70)
print("⚡ PHASE 1 CONFIGURATION")
print("="*70)
print(f"Data path: {config.DATA_PATH}")
print(f"Mode: {config.MODE}")
print(f"Downsample: {config.DOWNSAMPLE} → Using {100/config.DOWNSAMPLE:.1f}% of data")
print(f"Min samples per action: {config.MIN_SAMPLES_PER_ACTION}")
print("="*70)



print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

train = pd.read_csv(f'{config.DATA_PATH}/train.csv')
test = pd.read_csv(f'{config.DATA_PATH}/test.csv')

print(f"✓ train.csv loaded: {train.shape}")
print(f"✓ test.csv loaded: {test.shape}")

# Filter out MABe22
train_clean = train[~train['lab_id'].str.startswith('MABe22')].reset_index(drop=True)
print(f"✓ Clean training videos: {len(train_clean)}")



print("\n" + "="*60)
print("DEFINING FEATURE ENGINEERING")
print("="*60)

class MotionFeatures:
    @staticmethod
    def compute_velocity(coords, window=5):
        velocity = np.zeros(len(coords))
        if len(coords) > window:
            velocity[window:] = np.linalg.norm(
                coords[window:] - coords[:-window], axis=1
            ) / window
        return velocity
    
    @staticmethod
    def compute_acceleration(coords, window=5):
        vel = MotionFeatures.compute_velocity(coords, window)
        accel = np.zeros_like(vel)
        if len(vel) > window:
            accel[window:] = (vel[window:] - vel[:-window]) / window
        return accel

class GeometricFeatures:
    @staticmethod
    def compute_angle(p1, p2, p3):
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
        return np.sqrt(np.sum((coords1 - coords2)**2, axis=1))

def extract_single_mouse_features(mouse_df, cfg):
    features = {}
    body_parts = mouse_df.columns.get_level_values(0).unique().tolist()
    
    # Center (using body_center which exists in data)
    if 'body_center' in body_parts:
        center = mouse_df['body_center'].values
    elif 'center' in body_parts:
        center = mouse_df['center'].values
    else:
        center = mouse_df.values.reshape(len(mouse_df), -1, 2).mean(axis=1)
    
    # Position features
    for part in body_parts[:3]:
        if part in mouse_df.columns:
            coords = mouse_df[part].values
            rel_coords = coords - center
            features[f'{part}_rel_dist'] = np.linalg.norm(rel_coords, axis=1)
    
    # Motion features
    motion = MotionFeatures()
    for part in body_parts[:2]:
        if part in mouse_df.columns:
            coords = mouse_df[part].values
            for w in cfg.VELOCITY_WINDOWS:
                features[f'{part}_vel_{w}'] = motion.compute_velocity(coords, w)
            features[f'{part}_accel'] = motion.compute_acceleration(coords, 5)
    
    # Angles (check for body_center, not spine_middle)
    geom = GeometricFeatures()
    if all(p in body_parts for p in ['nose', 'body_center', 'tail_base']):
        angle = geom.compute_angle(
            mouse_df['nose'].values,
            mouse_df['body_center'].values,
            mouse_df['tail_base'].values
        )
        features['body_angle'] = angle
        angle_vel = motion.compute_velocity(angle.reshape(-1, 1), 5).flatten()
        features['body_angle_vel'] = angle_vel
    
    # Distances
    key_parts = []
    for p in ['nose', 'body_center', 'tail_base', 'neck', 'ear_left']:
        if p in body_parts:
            key_parts.append(p)
            if len(key_parts) >= 3:
                break
    
    for i, p1 in enumerate(key_parts):
        for p2 in key_parts[i+1:]:
            dist = geom.compute_distance(mouse_df[p1].values, mouse_df[p2].values)
            features[f'dist_{p1}_{p2}'] = dist
    
    features_df = pd.DataFrame(features, index=mouse_df.index)
    
    # Rolling aggregations
    for col in features_df.columns:
        for w in cfg.ROLLING_WINDOWS:
            features_df[f'{col}_mean_{w}'] = features_df[col].rolling(
                window=w, min_periods=1, center=True
            ).mean()
    
    return features_df

def extract_pair_features(mouse_pair_df, cfg):
    features = {}
    
    mouse_A = mouse_pair_df['A']
    mouse_B = mouse_pair_df['B']
    
    parts_A = mouse_A.columns.get_level_values(0).unique().tolist()
    parts_B = mouse_B.columns.get_level_values(0).unique().tolist()
    
    geom = GeometricFeatures()
    motion = MotionFeatures()
    
    # Inter-mouse distances
    for pa in ['nose', 'body_center'][:2]:
        if pa in parts_A:
            for pb in ['nose', 'body_center'][:2]:
                if pb in parts_B:
                    if pa in mouse_A.columns and pb in mouse_B.columns:
                        dist = geom.compute_distance(mouse_A[pa].values, mouse_B[pb].values)
                        features[f'dist_{pa}_{pb}'] = dist
    
    # Minimum distance
    all_dists = []
    for pa in parts_A[:3]:
        if pa in mouse_A.columns:
            for pb in parts_B[:3]:
                if pb in mouse_B.columns:
                    dist = geom.compute_distance(mouse_A[pa].values, mouse_B[pb].values)
                    all_dists.append(dist)
    
    if all_dists:
        all_dists = np.array(all_dists).T
        min_dist = np.min(all_dists, axis=1)
        features['min_dist'] = min_dist
        features['mean_dist'] = np.mean(all_dists, axis=1)
        approach_rate = -np.gradient(min_dist)
        features['approach_rate'] = approach_rate
    
    # Relative motion
    if 'nose' in parts_A and 'nose' in parts_B:
        for w in cfg.VELOCITY_WINDOWS[:2]:
            vel_A = motion.compute_velocity(mouse_A['nose'].values, w)
            vel_B = motion.compute_velocity(mouse_B['nose'].values, w)
            features[f'vel_diff_{w}'] = np.abs(vel_A - vel_B)
    
    # Orientation
    if all(p in parts_A for p in ['nose', 'tail_base']) and 'nose' in parts_B:
        dir_A = mouse_A['nose'].values - mouse_A['tail_base'].values
        dir_A = dir_A / (np.linalg.norm(dir_A, axis=1, keepdims=True) + 1e-8)
        
        to_B = mouse_B['nose'].values - mouse_A['nose'].values
        to_B = to_B / (np.linalg.norm(to_B, axis=1, keepdims=True) + 1e-8)
        
        orientation_A_to_B = np.sum(dir_A * to_B, axis=1)
        features['A_faces_B'] = orientation_A_to_B
    
    features_df = pd.DataFrame(features, index=mouse_pair_df.index)
    
    # Rolling aggregations
    for col in features_df.columns:
        for w in cfg.ROLLING_WINDOWS:
            features_df[f'{col}_mean_{w}'] = features_df[col].rolling(
                window=w, min_periods=1, center=True
            ).mean()
    
    return features_df

print("✓ Feature engineering defined")


print("\n" + "="*60)
print("DEFINING VIDEO PROCESSING")
print("="*60)

def load_and_pivot_video(lab_id, video_id, traintest='train'):
    path = f"{config.DATA_PATH}/{traintest}_tracking/{lab_id}/{video_id}.parquet"
    
    try:
        vid = pd.read_parquet(path)
        pvid = vid.pivot(
            columns=['mouse_id', 'bodypart'],
            index='video_frame',
            values=['x', 'y']
        )
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        return pvid
    except Exception as e:
        if config.VERBOSE:
            print(f"Error loading {video_id}: {e}")
        return None

def process_video(row, traintest='train'):
    lab_id = row['lab_id']
    video_id = row['video_id']
    
    if str(lab_id).startswith('MABe22'):
        return []
    
    pvid = load_and_pivot_video(lab_id, video_id, traintest)
    if pvid is None or len(pvid) == 0:
        return []
    
    pvid = pvid / row['pix_per_cm_approx']
    
    try:
        behaviors = json.loads(row['behaviors_labeled'])
        behaviors = sorted(list({b.replace("'", "") for b in behaviors}))
        behaviors = [b.split(',') for b in behaviors]
        behaviors_df = pd.DataFrame(behaviors, columns=['agent', 'target', 'action'])
    except:
        return []
    
    results = []
    
    # Single mouse behaviors
    single_behaviors = behaviors_df[behaviors_df['target'] == 'self']
    
    for mouse_str in single_behaviors['agent'].unique():
        try:
            mouse_id = int(mouse_str.replace('mouse', ''))
            
            if mouse_id not in pvid.columns.get_level_values(0):
                continue
            
            mouse_data = pvid[mouse_id]
            features = extract_single_mouse_features(mouse_data, config)
            
            if traintest == 'train' and config.DOWNSAMPLE > 1:
                features = features.iloc[::int(config.DOWNSAMPLE)]
            
            meta = pd.DataFrame({
                'video_id': video_id,
                'agent_id': mouse_str,
                'target_id': 'self',
                'video_frame': features.index
            })
            
            actions = single_behaviors[single_behaviors['agent'] == mouse_str]['action'].unique()
            
            if traintest == 'train':
                annot_path = f"{config.DATA_PATH}/train_annotation/{lab_id}/{video_id}.parquet"
                try:
                    annot = pd.read_parquet(annot_path)
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
    
    # Pair behaviors
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
                features = extract_pair_features(mouse_pair, config)
                
                if traintest == 'train' and config.DOWNSAMPLE > 1:
                    features = features.iloc[::int(config.DOWNSAMPLE)]
                
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

print("✓ Video processing defined")




print("\n" + "="*60)
print("PROCESSING TRAINING DATA")
print("="*60)

body_parts_groups = train_clean['body_parts_tracked'].unique()
print(f"Found {len(body_parts_groups)} body part configurations")

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
    
    gc.collect()

print(f"\n✓ All training data processed!")



# Diagnostic cell - add this as a new cell
print("\n" + "="*60)
print("DIAGNOSTIC REPORT")
print("="*60)

total_single_batches = sum(len(data['single']['features']) for data in train_data.values())
total_pair_batches = sum(len(data['pair']['features']) for data in train_data.values())

print(f"Total single batches: {total_single_batches}")
print(f"Total pair batches: {total_pair_batches}")

if total_single_batches == 0 and total_pair_batches == 0:
    print("❌ CRITICAL: No data processed! Feature extraction failed!")
else:
    print("✓ Data processed successfully")


print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

trained_models = {}

for bp_idx, (bp_str, data) in enumerate(train_data.items()):
    print(f"\n[{bp_idx+1}/{len(train_data)}] Training for: {bp_str[:60]}...")
    
    # Single mouse
    if len(data['single']['features']) > 0:
        print("\n  [SINGLE MOUSE]")
        
        X = pd.concat(data['single']['features'], ignore_index=True)
        meta = pd.concat(data['single']['meta'], ignore_index=True)
        labels = pd.concat(data['single']['labels'], ignore_index=True)
        
        print(f"    Shape: {X.shape}")
        print(f"    Actions: {list(labels.columns)}")
        
        single_models = {}
        
        for action in labels.columns:
            mask = ~labels[action].isna()
            
            if mask.sum() < config.MIN_SAMPLES_PER_ACTION:
                continue
            
            X_action = X[mask].fillna(0).values
            y_action = labels[action][mask].astype(int).values
            
            pos_rate = y_action.mean()
            
            model_lgbm = lgb.LGBMClassifier(**config.LGBM_PARAMS)
            model_lgbm.fit(X_action, y_action)
            
            model_xgb = xgb.XGBClassifier(**config.XGB_PARAMS)
            model_xgb.fit(X_action, y_action)
            
            single_models[action] = {
                'lgbm': model_lgbm,
                'xgb': model_xgb
            }
            
            print(f"      ✓ {action}: {mask.sum():,} samples, {pos_rate:.3f} pos")
        
        if len(single_models) > 0:
            trained_models[f'{bp_str}__single'] = single_models
        
        del X, meta, labels
        gc.collect()
    
    # Pair
    if len(data['pair']['features']) > 0:
        print("\n  [MOUSE PAIRS]")
        
        X = pd.concat(data['pair']['features'], ignore_index=True)
        meta = pd.concat(data['pair']['meta'], ignore_index=True)
        labels = pd.concat(data['pair']['labels'], ignore_index=True)
        
        print(f"    Shape: {X.shape}")
        print(f"    Actions: {list(labels.columns)}")
        
        pair_models = {}
        
        for action in labels.columns:
            mask = ~labels[action].isna()
            
            if mask.sum() < config.MIN_SAMPLES_PER_ACTION:
                continue
            
            X_action = X[mask].fillna(0).values
            y_action = labels[action][mask].astype(int).values
            
            pos_rate = y_action.mean()
            
            model_lgbm = lgb.LGBMClassifier(**config.LGBM_PARAMS)
            model_lgbm.fit(X_action, y_action)
            
            model_xgb = xgb.XGBClassifier(**config.XGB_PARAMS)
            model_xgb.fit(X_action, y_action)
            
            pair_models[action] = {
                'lgbm': model_lgbm,
                'xgb': model_xgb
            }
            
            print(f"      ✓ {action}: {mask.sum():,} samples, {pos_rate:.3f} pos")
        
        if len(pair_models) > 0:
            trained_models[f'{bp_str}__pair'] = pair_models
        
        del X, meta, labels
        gc.collect()

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Total model groups: {len(trained_models)}")




def predict_multiclass(preds_df, meta_df, threshold=0.15):
    if len(preds_df.columns) == 0:
        return pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    
    action_idx_list = []
    
    for frame_idx in range(len(preds_df)):
        max_prob = 0
        max_action_idx = -1
        
        for action_idx, action in enumerate(preds_df.columns):
            prob = preds_df.iloc[frame_idx, action_idx]
            
            if prob >= threshold and prob > max_prob:
                max_prob = prob
                max_action_idx = action_idx
        
        action_idx_list.append(max_action_idx)
    
    action_series = pd.Series(action_idx_list, index=meta_df['video_frame'].values)
    
    changes = (action_series != action_series.shift(1))
    change_frames = action_series[changes]
    change_meta = meta_df[changes.values].reset_index(drop=True)
    
    submissions = []
    
    for i in range(len(change_frames) - 1):
        action_idx = change_frames.iloc[i]
        
        if action_idx >= 0:
            video_id = change_meta.iloc[i]['video_id']
            agent_id = change_meta.iloc[i]['agent_id']
            target_id = change_meta.iloc[i]['target_id']
            action = preds_df.columns[action_idx]
            start_frame = change_frames.index[i]
            
            next_video = change_meta.iloc[i + 1]['video_id']
            next_agent = change_meta.iloc[i + 1]['agent_id']
            next_target = change_meta.iloc[i + 1]['target_id']
            
            if video_id == next_video and agent_id == next_agent and target_id == next_target:
                stop_frame = change_frames.index[i + 1]
            else:
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

print("✓ Prediction function defined")


print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)

if config.MODE == 'submit':
    dataset = test
else:
    val_split = int(len(train_clean) * 0.8)
    dataset = train_clean.iloc[val_split:]

all_predictions = []

body_parts_groups = dataset['body_parts_tracked'].unique()

for bp_idx, bp_str in enumerate(body_parts_groups):
    print(f"\n[{bp_idx+1}/{len(body_parts_groups)}] Predicting: {bp_str[:60]}...")
    
    subset = dataset[dataset['body_parts_tracked'] == bp_str]
    print(f"  Videos: {len(subset)}")
    
    single_model_key = f'{bp_str}__single'
    pair_model_key = f'{bp_str}__pair'
    
    has_single = single_model_key in trained_models
    has_pair = pair_model_key in trained_models
    
    if not has_single and not has_pair:
        print(f"  ⚠️  No trained models")
        continue
    
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="  Videos", leave=False):
        
        if config.MODE == 'submit':
            results = process_video(row, 'test')
        else:
            results = process_video(row, 'train')
        
        for switch, features, meta, actions_or_labels in results:
            
            if config.MODE == 'submit':
                actions = actions_or_labels
            else:
                actions = actions_or_labels.columns.tolist()
            
            if switch == 'single' and has_single:
                models = trained_models[single_model_key]
            elif switch == 'pair' and has_pair:
                models = trained_models[pair_model_key]
            else:
                continue
            
            predictions = pd.DataFrame(index=features.index)
            
            for action in actions:
                if action in models:
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
            
            if len(predictions.columns) > 0:
                submission_part = predict_multiclass(predictions, meta, config.THRESHOLD)
                all_predictions.append(submission_part)
    
    print(f"  ✓ Predictions: {sum(len(p) for p in all_predictions)}")

if len(all_predictions) > 0:
    submission = pd.concat(all_predictions, ignore_index=True)
else:
    submission = pd.DataFrame(columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

print(f"\n✓ Total predictions: {len(submission)}")



print("\n" + "="*60)
print("POST-PROCESSING & SAVING")
print("="*60)

# Remove invalid
original_len = len(submission)
submission = submission[submission['start_frame'] < submission['stop_frame']]
print(f"✓ Removed {original_len - len(submission)} invalid predictions")

# Remove overlaps
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

# Fill missing videos
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

# Save
submission = submission.reset_index(drop=True)
submission.index.name = 'row_id'
submission.to_csv('submission.csv')

print("\n✓ Saved to submission.csv")
print(f"\nSubmission shape: {submission.shape}")
print(f"Videos covered: {submission['video_id'].nunique()} / {len(dataset)}")

print("\n" + "="*70)

