# Final version for commit

"""
NOTEBOOK A: PREPROCESSOR
- Processes all 8,789 videos with feature engineering
- Saves processed data for each video to separate files
- No training, just data preparation
- Runtime: ~4.5 hours on CPU
"""

import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from itertools import permutations
from collections import defaultdict
import warnings
from scipy.signal import welch

# Feature engineering functions
def add_olfactory_intent_features(df_full_flat, agent_id, target_id, fps):
    features = {}
    try:
        v_ba_x = df_full_flat[f'x_{agent_id}_nose'] - df_full_flat[f'x_{agent_id}_tail_base']
        v_ba_y = df_full_flat[f'y_{agent_id}_nose'] - df_full_flat[f'y_{agent_id}_tail_base']
        v_t_x = df_full_flat[f'x_{target_id}_tail_base'] - df_full_flat[f'x_{agent_id}_nose']
        v_t_y = df_full_flat[f'y_{target_id}_tail_base'] - df_full_flat[f'y_{agent_id}_nose']
        dot_product = v_ba_x * v_t_x + v_ba_y * v_t_y
        norm_ba = np.sqrt(v_ba_x**2 + v_ba_y**2)
        norm_t = np.sqrt(v_t_x**2 + v_t_y**2)
        cosine_angle = np.clip(dot_product / (norm_ba * norm_t + 1e-6), -1.0, 1.0)
        features['hae_genital_deg'] = np.degrees(np.arccos(cosine_angle))
        
        nose_speed = np.sqrt(df_full_flat[f'x_{agent_id}_nose_vel']**2 + df_full_flat[f'y_{agent_id}_nose_vel']**2)
        def calculate_psd_optimized(x):
            if len(x) < 8: return 0
            try:
                nperseg = min(len(x), max(8, len(x)//4))
                freqs, psd = welch(x, fs=fps, nperseg=nperseg, noverlap=nperseg//2)
                band_mask = (freqs >= 8.0) & (freqs <= 15.0)
                return np.sum(psd[band_mask]) if np.any(band_mask) else 0
            except: return 0
        
        window_size = max(8, int(fps / 4))
        features['p_sniff_agent'] = nose_speed.rolling(window=window_size, center=True, min_periods=8).apply(calculate_psd_optimized, raw=True).fillna(0)
    except KeyError: pass
    return pd.DataFrame(features, index=df_full_flat.index).fillna(0)

def add_lunge_features(df_full_flat, agent_id, fps):
    features = {}
    try:
        body_length = np.sqrt((df_full_flat[f'x_{agent_id}_nose'] - df_full_flat[f'x_{agent_id}_tail_base'])**2 + (df_full_flat[f'y_{agent_id}_nose'] - df_full_flat[f'y_{agent_id}_tail_base'])**2)
        reference_length = body_length.rolling(window=int(fps*5), min_periods=1, center=True).mean()
        features['dbcr_agent'] = body_length / (reference_length + 1e-6)
        
        center_bp = 'body_center' if f'x_{agent_id}_body_center' in df_full_flat.columns else 'neck'
        accel_x = df_full_flat[f'x_{agent_id}_{center_bp}_vel'].diff() * fps
        accel_y = df_full_flat[f'y_{agent_id}_{center_bp}_vel'].diff() * fps
        features['acom_agent'] = np.sqrt(accel_x**2 + accel_y**2)
        
        is_compressed = (features['dbcr_agent'].shift(int(fps * 0.05)) < 0.95)
        is_accelerating = (features['acom_agent'] > features['acom_agent'].mean() + 3 * features['acom_agent'].std())
        features['lunge_signature'] = (is_compressed & is_accelerating).astype(int)
    except KeyError: pass
    return pd.DataFrame(features, index=df_full_flat.index).fillna(0)

def add_withdrawal_features(df_full_flat, agent_id, fps):
    features = {}
    try:
        center_bp = 'body_center' if f'x_{agent_id}_body_center' in df_full_flat.columns else 'neck'
        vel_x = df_full_flat[f'x_{agent_id}_{center_bp}'].diff() * fps
        accel_x = vel_x.diff() * fps
        jerk_x = accel_x.diff() * fps
        vel_y = df_full_flat[f'y_{agent_id}_{center_bp}'].diff() * fps
        accel_y = vel_y.diff() * fps
        jerk_y = accel_y.diff() * fps
        features['jerk_magnitude_agent'] = np.sqrt(jerk_x**2 + jerk_y**2)
        
        if f'x_{agent_id}_hip_left' in df_full_flat.columns and f'x_{agent_id}_hip_right' in df_full_flat.columns:
            hip_width = np.sqrt((df_full_flat[f'x_{agent_id}_hip_left'] - df_full_flat[f'x_{agent_id}_hip_right'])**2 + (df_full_flat[f'y_{agent_id}_hip_left'] - df_full_flat[f'y_{agent_id}_hip_right'])**2)
            features['hip_width_agent'] = hip_width
        
        speed = np.sqrt(vel_x**2 + vel_y**2)
        is_stationary = speed < 2.0
        features['is_submitting_agent'] = is_stationary.rolling(window=int(fps)).apply(np.all, raw=True)
    except KeyError: pass
    return pd.DataFrame(features, index=df_full_flat.index).fillna(0)

def add_advanced_kinematic_features(df_flat, mouse_ids, pix_per_cm):
    for mid in mouse_ids:
        nose_x, nose_y = f'x_{mid}_nose', f'y_{mid}_nose'
        tail_x, tail_y = f'x_{mid}_tail_base', f'y_{mid}_tail_base'
        if nose_x in df_flat.columns and tail_x in df_flat.columns:
            body_length_px = np.sqrt((df_flat[nose_x] - df_flat[tail_x])**2 + (df_flat[nose_y] - df_flat[tail_y])**2)
            df_flat[f'body_length_cm_{mid}'] = body_length_px / pix_per_cm
            df_flat[f'body_length_vel_{mid}'] = df_flat[f'body_length_cm_{mid}'].diff()
            body_length_accel = df_flat[f'body_length_vel_{mid}'].diff()
            df_flat[f'body_flinch_signal_{mid}'] = body_length_accel.fillna(0)
        
        center_bp = 'body_center' if f'x_{mid}_body_center' in df_flat.columns else 'neck'
        center_bp_vel_x = f'x_{mid}_{center_bp}_vel'
        if center_bp_vel_x in df_flat.columns:
            accel_x = df_flat[f'x_{mid}_{center_bp}_vel'].diff()
            accel_y = df_flat[f'y_{mid}_{center_bp}_vel'].diff()
            window = 5
            accel_x_var = accel_x.rolling(window, center=True, min_periods=1).var()
            accel_y_var = accel_y.rolling(window, center=True, min_periods=1).var()
            df_flat[f'jitter_metric_{mid}'] = np.sqrt(accel_x_var**2 + accel_y_var**2).fillna(0)
    return df_flat

def add_sniff_detection_features(df_full_flat, mouse_ids, fps):
    for mid in mouse_ids:
        try:
            nose_vel_x_col = f'x_{mid}_nose_vel'
            nose_vel_y_col = f'y_{mid}_nose_vel'
            if (nose_vel_x_col in df_full_flat.columns and nose_vel_y_col in df_full_flat.columns and not df_full_flat[nose_vel_x_col].isna().all()):
                nose_speed = np.sqrt(df_full_flat[nose_vel_x_col]**2 + df_full_flat[nose_vel_y_col]**2)
                window_size = min(int(fps), 10)
                df_full_flat[f'sniff_intensity_{mid}'] = nose_speed.rolling(window=window_size, center=True, min_periods=1).var().fillna(0)
            else:
                df_full_flat[f'sniff_intensity_{mid}'] = 0
        except (KeyError, TypeError, ValueError):
            df_full_flat[f'sniff_intensity_{mid}'] = 0
    return df_full_flat

def add_submission_specific_features(df_full_flat, mouse_ids, pix_per_cm, fps):
    for mid in mouse_ids:
        lat_L = f'x_{mid}_lateral_left'
        lat_R = f'x_{mid}_lateral_right'
        if lat_L in df_full_flat.columns and lat_R in df_full_flat.columns:
            lateral_width_px = np.sqrt((df_full_flat[f'x_{mid}_lateral_left'] - df_full_flat[f'x_{mid}_lateral_right'])**2 + (df_full_flat[f'y_{mid}_lateral_left'] - df_full_flat[f'y_{mid}_lateral_right'])**2)
            df_full_flat[f'lateral_width_cm_{mid}'] = lateral_width_px / pix_per_cm
            
            nose_x, tail_x = f'x_{mid}_nose', f'x_{mid}_tail_base'
            if nose_x in df_full_flat.columns and tail_x in df_full_flat.columns:
                longitudinal_length_px = np.sqrt((df_full_flat[nose_x] - df_full_flat[tail_x])**2 + (df_full_flat[f'y_{mid}_nose'] - df_full_flat[f'y_{mid}_tail_base'])**2)
                df_full_flat[f'longitudinal_length_cm_{mid}'] = longitudinal_length_px / pix_per_cm
                df_full_flat[f'compression_index_{mid}'] = df_full_flat[f'lateral_width_cm_{mid}'] / (df_full_flat[f'longitudinal_length_cm_{mid}'] + 1e-6)
        
        center_bp = 'body_center' if f'x_{mid}_body_center' in df_full_flat.columns else 'neck'
        if f'x_{mid}_{center_bp}_vel' in df_full_flat.columns:
            speed_cm_s = np.sqrt(df_full_flat[f'x_{mid}_{center_bp}_vel']**2 + df_full_flat[f'y_{mid}_{center_bp}_vel']**2) * fps / pix_per_cm
            speed_3_frames_ago = speed_cm_s.shift(3)
            threshold_high_speed = 10.0
            threshold_low_speed = 1.0
            df_full_flat[f'velocity_collapse_{mid}'] = ((speed_3_frames_ago > threshold_high_speed) & (speed_cm_s < threshold_low_speed)).astype(int)
            
            if f'x_{mid}_nose' in df_full_flat.columns and f'x_{mid}_neck' in df_full_flat.columns:
                nose_neck_dist = np.sqrt((df_full_flat[f'x_{mid}_nose'] - df_full_flat[f'x_{mid}_neck'])**2 + (df_full_flat[f'y_{mid}_nose'] - df_full_flat[f'y_{mid}_neck'])**2)
                nose_tail_dist = np.sqrt((df_full_flat[f'x_{mid}_nose'] - df_full_flat[f'x_{mid}_tail_base'])**2 + (df_full_flat[f'y_{mid}_nose'] - df_full_flat[f'y_{mid}_tail_base'])**2)
                df_full_flat[f'head_tucking_ratio_{mid}'] = nose_neck_dist / (nose_tail_dist + 1e-6)
    return df_full_flat

def create_pair_specific_features_robust(df_full_flat, agent_id, target_id, fps, pix_per_cm, parts_agent, parts_target):
    features = {}
    center_bp_agent = 'body_center' if 'body_center' in parts_agent else 'neck'
    center_bp_target = 'body_center' if 'body_center' in parts_target else 'neck'
    
    try:
        if (f'x_{agent_id}_{center_bp_agent}' in df_full_flat.columns and f'x_{target_id}_{center_bp_target}' in df_full_flat.columns):
            dist_px = np.sqrt((df_full_flat[f'x_{agent_id}_{center_bp_agent}'] - df_full_flat[f'x_{target_id}_{center_bp_target}'])**2 + (df_full_flat[f'y_{agent_id}_{center_bp_agent}'] - df_full_flat[f'y_{target_id}_{center_bp_target}'])**2)
            features['dist_center_cm_agent_target'] = dist_px / pix_per_cm
    except (KeyError, TypeError): features['dist_center_cm_agent_target'] = 0
    
    try:
        if f'x_{agent_id}_{center_bp_agent}_vel' in df_full_flat.columns:
            agent_vel_x = df_full_flat[f'x_{agent_id}_{center_bp_agent}_vel']
            agent_vel_y = df_full_flat[f'y_{agent_id}_{center_bp_agent}_vel']
            features['speed_cm_s_agent'] = np.sqrt(agent_vel_x**2 + agent_vel_y**2) * fps / pix_per_cm
    except (KeyError, TypeError): features['speed_cm_s_agent'] = 0
    
    try:
        if f'x_{target_id}_{center_bp_target}_vel' in df_full_flat.columns:
            target_vel_x = df_full_flat[f'x_{target_id}_{center_bp_target}_vel']
            target_vel_y = df_full_flat[f'y_{target_id}_{center_bp_target}_vel']
            features['speed_cm_s_target'] = np.sqrt(target_vel_x**2 + target_vel_y**2) * fps / pix_per_cm
    except (KeyError, TypeError): features['speed_cm_s_target'] = 0
    
    if 'dist_center_cm_agent_target' in features:
        try: features['relative_speed_cm_s_agent_target'] = features['dist_center_cm_agent_target'].diff() * fps
        except (KeyError, TypeError): features['relative_speed_cm_s_agent_target'] = 0
    
    try:
        if (f'x_{agent_id}_nose' in df_full_flat.columns and f'x_{agent_id}_tail_base' in df_full_flat.columns and f'x_{target_id}_{center_bp_target}' in df_full_flat.columns):
            agent_angle = np.arctan2(df_full_flat[f'y_{agent_id}_nose'] - df_full_flat[f'y_{agent_id}_tail_base'], df_full_flat[f'x_{agent_id}_nose'] - df_full_flat[f'x_{agent_id}_tail_base'])
            target_angle = np.arctan2(df_full_flat[f'y_{target_id}_{center_bp_target}'] - df_full_flat[f'y_{agent_id}_{center_bp_agent}'], df_full_flat[f'x_{target_id}_{center_bp_target}'] - df_full_flat[f'x_{agent_id}_{center_bp_agent}'])
            relative_angle = target_angle - agent_angle
            features['relative_angle_agent_target'] = (relative_angle + np.pi) % (2 * np.pi) - np.pi
    except (KeyError, TypeError): features['relative_angle_agent_target'] = 0
    
    for mid, label in [(agent_id, 'agent'), (target_id, 'target')]:
        try:
            if f'body_length_cm_{mid}' in df_full_flat.columns:
                features[f'body_length_cm_{label}'] = df_full_flat[f'body_length_cm_{mid}']
            else: features[f'body_length_cm_{label}'] = 0
        except (KeyError, TypeError): features[f'body_length_cm_{label}'] = 0
        
        try:
            if f'sniff_intensity_{mid}' in df_full_flat.columns:
                features[f'sniff_intensity_{label}'] = df_full_flat[f'sniff_intensity_{mid}']
            else: features[f'sniff_intensity_{label}'] = 0
        except (KeyError, TypeError): features[f'sniff_intensity_{label}'] = 0
    
    return pd.DataFrame(features, index=df_full_flat.index).fillna(0)

def engineer_features_for_video(video_id, tracking_path, metadata_df):
    try:
        video_metadata = metadata_df.loc[int(video_id)]
        fps = video_metadata['frames_per_second']
        pix_per_cm = video_metadata['pix_per_cm_approx']
    except (KeyError, TypeError): fps, pix_per_cm = 30.0, 10.0
    
    try: df_long = pd.read_parquet(tracking_path)
    except Exception: return None
    
    df_wide = df_long.pivot_table(index='video_frame', columns=['mouse_id', 'bodypart'], values=['x', 'y'])
    df_wide_clean = df_wide.interpolate(method='linear', limit_direction='both', axis=0)
    df_flat = df_wide_clean.copy()
    df_flat.columns = ['_'.join(map(str, col)).replace("('", "").replace("', '", "_").replace("')", "") for col in df_flat.columns]
    
    mouse_ids = sorted(df_long['mouse_id'].unique())
    
    try: df_flat = add_advanced_kinematic_features(df_flat, mouse_ids, pix_per_cm)
    except Exception: return None
    
    try:
        velocity_df = df_flat.diff()
        velocity_df.columns = [f'{col}_vel' for col in df_flat.columns]
        df_full_flat = pd.concat([df_flat, velocity_df], axis=1)
    except Exception: return None
    
    try: df_full_flat = add_sniff_detection_features(df_full_flat, mouse_ids, fps)
    except Exception: return None
    
    try: df_full_flat = add_submission_specific_features(df_full_flat, mouse_ids, pix_per_cm, fps)
    except Exception: return None
    
    try:
        all_pair_features_list = []
        for agent_id, target_id in permutations(mouse_ids, 2):
            try:
                parts_agent = set(df_long[df_long['mouse_id'] == agent_id]['bodypart'])
                parts_target = set(df_long[df_long['mouse_id'] == target_id]['bodypart'])
                
                pair_features = create_pair_specific_features_robust(df_full_flat, agent_id, target_id, fps, pix_per_cm, parts_agent, parts_target)
                
                # Add rolling averages
                for window in [5, 15]:
                    for col in ['dist_center_cm_agent_target', 'relative_speed_cm_s_agent_target']:
                        if col in pair_features.columns and not pair_features[col].isna().all():
                            try: pair_features[f'{col}_mean_{window}f'] = pair_features[col].rolling(window=window, center=True, min_periods=1).mean()
                            except (KeyError, TypeError): pair_features[f'{col}_mean_{window}f'] = 0
                
                # Add advanced features
                try:
                    olfactory_features = add_olfactory_intent_features(df_full_flat, agent_id, target_id, fps)
                    lunge_features = add_lunge_features(df_full_flat, agent_id, fps)
                    withdrawal_features = add_withdrawal_features(df_full_flat, agent_id, fps)
                    
                    for df_feat in [olfactory_features, lunge_features, withdrawal_features]:
                        if not df_feat.empty:
                            for col in df_feat.columns:
                                pair_features[col] = df_feat[col]
                except Exception: pass
                
                pair_features['agent_id'] = agent_id
                pair_features['target_id'] = target_id
                pair_features['video_frame'] = pair_features.index
                all_pair_features_list.append(pair_features)
            except Exception: continue
        
        if not all_pair_features_list: return None
        
        final_video_features = pd.concat(all_pair_features_list, ignore_index=True)
        return final_video_features
    except Exception: return None

def create_training_examples(full_features_df, annotation_df, video_id, mouse_ids, metadata_df):
    try:
        video_metadata = metadata_df.loc[int(video_id)]
    except (KeyError, TypeError):
        video_metadata = {}
    
    mouse_info = {}
    for i in range(1, 5):
        if f'mouse{i}_strain' in video_metadata and pd.notna(video_metadata.get(f'mouse{i}_strain')):
            mouse_info[i] = {
                'strain': video_metadata.get(f'mouse{i}_strain', 'unknown'),
                'sex': video_metadata.get(f'mouse{i}_sex', 'unknown'),
            }
    
    X_pieces, y_pieces, groups_pieces = [], [], []
    
    for _, event in annotation_df.iterrows():
        agent, target, action = int(event['agent_id']), int(event['target_id']), event['action']
        start, end = int(event['start_frame']), int(event['stop_frame'])
        
        pair_mask = (full_features_df['agent_id'] == agent) & (full_features_df['target_id'] == target)
        frame_mask = (full_features_df['video_frame'] >= start) & (full_features_df['video_frame'] <= end)
        event_data = full_features_df[pair_mask & frame_mask].copy()
        
        if event_data.empty: continue
        
        event_data['lab_id'] = video_metadata.get('lab_id', 'unknown')
        event_data['agent_strain'] = mouse_info.get(agent, {}).get('strain', 'unknown')
        event_data['target_strain'] = mouse_info.get(target, {}).get('strain', 'unknown')
        event_data['agent_sex'] = mouse_info.get(agent, {}).get('sex', 'unknown')
        event_data['target_sex'] = mouse_info.get(target, {}).get('sex', 'unknown')
        event_data['arena_type'] = video_metadata.get('arena_type', 'unknown')
        event_data['tracking_method'] = video_metadata.get('tracking_method', 'unknown')
        event_data['lights_on'] = video_metadata.get('lights_on', 'unknown')
        
        X_pieces.append(event_data)
        y_pieces.extend([action] * len(event_data))
        groups_pieces.extend([video_id] * len(event_data))
    
    if not X_pieces: return None
    
    positive_frames = set()
    for _, event in annotation_df.iterrows():
        positive_frames.update(range(event['start_frame'], event['stop_frame'] + 1))
    
    all_frames = set(full_features_df['video_frame'].unique())
    background_frames = list(all_frames - positive_frames)
    
    num_pos_samples = sum(len(p) for p in X_pieces)
    num_neg_to_sample = min(num_pos_samples // 2, len(background_frames))
    
    if num_neg_to_sample > 0:
        sampled_frames = np.random.choice(background_frames, size=num_neg_to_sample, replace=False)
        
        for frame in sampled_frames:
            agent, target = np.random.choice(mouse_ids, 2, replace=False)
            
            pair_mask = (full_features_df['agent_id'] == agent) & (full_features_df['target_id'] == target)
            frame_mask = (full_features_df['video_frame'] == frame)
            neg_data = full_features_df[pair_mask & frame_mask].copy()
            
            if neg_data.empty: continue
            
            neg_data['lab_id'] = video_metadata.get('lab_id', 'unknown')
            neg_data['agent_strain'] = mouse_info.get(agent, {}).get('strain', 'unknown')
            neg_data['target_strain'] = mouse_info.get(target, {}).get('strain', 'unknown')
            neg_data['agent_sex'] = mouse_info.get(agent, {}).get('sex', 'unknown')
            neg_data['target_sex'] = mouse_info.get(target, {}).get('sex', 'unknown')
            neg_data['arena_type'] = video_metadata.get('arena_type', 'unknown')
            neg_data['tracking_method'] = video_metadata.get('tracking_method', 'unknown')
            neg_data['lights_on'] = video_metadata.get('lights_on', 'unknown')
            
            X_pieces.append(neg_data)
            y_pieces.extend(['background'] * len(neg_data))
            groups_pieces.extend([video_id] * len(neg_data))
    
    X_final = pd.concat(X_pieces, ignore_index=True)
    y_final = pd.Series(y_pieces)
    groups_final = pd.Series(groups_pieces)
    
    return X_final, y_final, groups_final

def check_files_exist(video_id, data_dir):
    tracking_base_dir = os.path.join(data_dir, "train_tracking")
    annotation_base_dir = os.path.join(data_dir, "train_annotation")
    all_subsets = [d for d in os.listdir(tracking_base_dir) if os.path.isdir(os.path.join(tracking_base_dir, d))]
    for subset in all_subsets:
        tracking_path = os.path.join(tracking_base_dir, subset, f"{video_id}.parquet")
        annotation_path = os.path.join(annotation_base_dir, subset, f"{video_id}.parquet")
        if os.path.exists(tracking_path) and os.path.exists(annotation_path):
            return tracking_path, annotation_path
    return None, None

# Configuration
CONFIG = {'DATA_DIR': "/kaggle/input/MABe-mouse-behavior-detection/"}
DATA_DIR = CONFIG['DATA_DIR']

print("ðŸš€ Starting Stage 1A: Data Preprocessing")
metadata_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).set_index("video_id")
print("Metadata loaded successfully.")

tracking_base_dir = os.path.join(DATA_DIR, "train_tracking")
all_subsets = [d for d in os.listdir(tracking_base_dir) if os.path.isdir(os.path.join(tracking_base_dir, d))]
VIDEO_IDS_TO_PROCESS = []
for subset in all_subsets:
    tracking_dir = os.path.join(tracking_base_dir, subset)
    subset_videos = [f.split(".")[0] for f in os.listdir(tracking_dir) if f.endswith('.parquet')]
    VIDEO_IDS_TO_PROCESS.extend(subset_videos)

print(f"Found {len(VIDEO_IDS_TO_PROCESS)} videos to process.")

# Main preprocessing work
print("\n=== Building and Saving Training Examples Video by Video ===")
os.makedirs('/kaggle/working/processed_videos', exist_ok=True)
processed_count = 0

for i, video_id in enumerate(tqdm(VIDEO_IDS_TO_PROCESS, desc="Processing Training Videos")):
    try:
        tracking_path, annotation_path = check_files_exist(video_id, DATA_DIR)
        if tracking_path is None: continue

        full_video_features = engineer_features_for_video(video_id, tracking_path, metadata_df)
        if full_video_features is None: continue

        annotation_df = pd.read_parquet(annotation_path)
        df_long = pd.read_parquet(tracking_path)
        mouse_ids = sorted(df_long['mouse_id'].unique())
        result = create_training_examples(full_video_features, annotation_df, video_id, mouse_ids, metadata_df)

        if result:
            X_video, y_video, groups_video = result
            temp_df = X_video.copy()
            temp_df['labels'] = y_video
            temp_df['groups'] = groups_video
            temp_df.to_parquet(f'/kaggle/working/processed_videos/video_{video_id}.parquet')
            processed_count += 1
            
            # Clean up memory
            del X_video, y_video, groups_video, temp_df, full_video_features, annotation_df, df_long
            if i % 50 == 0: gc.collect()

    except Exception as e:
        continue

print(f"\nâœ… Preprocessing Complete! Successfully processed and saved {processed_count} videos.")
print("The output of this notebook is the '/kaggle/working/processed_videos' directory.")
print("You can now commit this notebook and use its output as input for your training notebook.")

