import os
import gc
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
import lightgbm as lgb
from joblib import dump, load
import warnings

warnings.filterwarnings('ignore')

# Global constants for feature engineering
BODY_PARTS = ['nose', 'ear_left', 'ear_right', 'neck', 'hip_left', 'hip_right', 'tail_base']


class CFG:
    # --- Data Paths ---
    DATA_ROOT = "/kaggle/input/MABe-mouse-behavior-detection"
    TRAIN_TRACKING_DIR = os.path.join(DATA_ROOT, "train_tracking")
    TEST_TRACKING_DIR = os.path.join(DATA_ROOT, "test_tracking")
    TRAIN_ANNOTATION_DIR = os.path.join(DATA_ROOT, "train_annotation")
    
    # --- Behaviors to Model ---
    BEHAVIORS = ['attack', 'chase', 'investigation', 'mount', 'sniff', 'huddle', 'submit']
    
    # --- Post-Processing Parameters ---
    PROB_THRESHOLD = 0.5
    MIN_EVENT_LENGTH_FRAMES = 5
    
    # --- Training Parameters ---
    VIDEOS_PER_BEHAVIOR = 40 
    
    LGB_PARAMS = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31,
        'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    }


def create_features(df_tracking):
    """
    Transforms raw tracking data into a standardized egocentric feature set.
    Feature names are based on 'agent' and 'target' roles, not raw mouse IDs.
    """
    all_pair_features = []
    mice_present = sorted(df_tracking['mouse_id'].unique())
    if len(mice_present) < 2: return None

    try:
        df_wide = df_tracking.pivot_table(
            index='video_frame', columns=['mouse_id', 'bodypart'], values=['x', 'y']
        )
        df_wide.columns = [f'{val}_{mid}_{bp}' for val, mid, bp in df_wide.columns]
        df_wide = df_wide.sort_index().ffill().bfill()
    except Exception:
        return None

    for agent_id in mice_present:
        for target_id in mice_present:
            if agent_id == target_id: continue
            
            agent_cols = [f'x_{agent_id}_neck', f'y_{agent_id}_neck', f'x_{agent_id}_nose', f'y_{agent_id}_nose']
            if not all(c in df_wide.columns for c in agent_cols):
                continue

            agent_neck_x = df_wide[f'x_{agent_id}_neck'].values
            agent_neck_y = df_wide[f'y_{agent_id}_neck'].values
            agent_nose_x = df_wide[f'x_{agent_id}_nose'].values - agent_neck_x
            agent_nose_y = df_wide[f'y_{agent_id}_nose'].values - agent_neck_y
            
            angles = np.arctan2(agent_nose_y, agent_nose_x) - np.pi / 2
            cos_angles, sin_angles = np.cos(angles), np.sin(angles)
            
            pair_features = pd.DataFrame({'video_frame': df_wide.index})
            
            # Use STANDARDIZED names: 'agent' and 'target'
            for role, mouse_id in [('agent', agent_id), ('target', target_id)]:
                for bp in BODY_PARTS:
                    x_col, y_col = f'x_{mouse_id}_{bp}', f'y_{mouse_id}_{bp}'
                    if x_col in df_wide.columns and y_col in df_wide.columns:
                        orig_x = df_wide[x_col].values - agent_neck_x
                        orig_y = df_wide[y_col].values - agent_neck_y
                        rotated_x = orig_x * cos_angles + orig_y * sin_angles
                        rotated_y = -orig_x * sin_angles + orig_y * cos_angles
                    else:
                        rotated_x, rotated_y = np.zeros_like(agent_neck_x), np.zeros_like(agent_neck_y)
                    
                    pair_features[f'{role}_{bp}_x_ego'] = rotated_x
                    pair_features[f'{role}_{bp}_y_ego'] = rotated_y
                    pair_features[f'{role}_{bp}_vx_ego'] = np.diff(rotated_x, prepend=0)
                    pair_features[f'{role}_{bp}_vy_ego'] = np.diff(rotated_y, prepend=0)
            
            pair_features['agent_id'] = agent_id
            pair_features['target_id'] = target_id
            all_pair_features.append(pair_features)

    return pd.concat(all_pair_features) if all_pair_features else None


IS_SUBMISSION = len(glob(os.path.join(CFG.TEST_TRACKING_DIR, "*/*.parquet"))) > 100

if not IS_SUBMISSION:
    print("Detected Training Environment. Starting model training...")
    
    annotation_files = glob(os.path.join(CFG.TRAIN_ANNOTATION_DIR, "*/*.parquet"))
    all_labels_list = []
    for f in tqdm(annotation_files, desc="Reading annotations"):
        video_id = int(os.path.basename(f).split('.')[0])
        df = pd.read_parquet(f)
        df['video_id'] = video_id
        all_labels_list.append(df)
    all_labels_df = pd.concat(all_labels_list, ignore_index=True)

    for behavior in CFG.BEHAVIORS:
        print(f"--- Training model for: {behavior} ---")
        behavior_labels = all_labels_df[all_labels_df['action'] == behavior]
        if behavior_labels.empty:
            print(f"No annotations for {behavior}. Skipping.")
            continue

        X_train_all, y_train_all = [], []
        train_video_ids = behavior_labels['video_id'].unique()
        if len(train_video_ids) > CFG.VIDEOS_PER_BEHAVIOR:
            train_video_ids = np.random.choice(train_video_ids, CFG.VIDEOS_PER_BEHAVIOR, replace=False)
        
        for video_id in tqdm(train_video_ids, desc=f"Featurizing for {behavior}"):
            tracking_path = glob(os.path.join(CFG.TRAIN_TRACKING_DIR, f"*/{video_id}.parquet"))
            if not tracking_path: continue
            
            df_tracking = pd.read_parquet(tracking_path[0])
            df_features = create_features(df_tracking)
            if df_features is None: continue
            
            n_frames = df_tracking['video_frame'].max() + 1
            vid_labels = behavior_labels[behavior_labels['video_id'] == video_id]
            
            for (agent_id, target_id), pair_df in df_features.groupby(['agent_id', 'target_id']):
                target = np.zeros(n_frames)
                pair_annotations = vid_labels[(vid_labels['agent_id'] == agent_id) & (vid_labels['target_id'] == target_id)]
                for _, row in pair_annotations.iterrows():
                    target[row['start_frame']:row['stop_frame']] = 1
                
                pair_df['target'] = pair_df['video_frame'].map(lambda f: target[f] if f < n_frames else 0)
                
                positive_samples = pair_df[pair_df['target'] == 1]
                negative_samples = pair_df[pair_df['target'] == 0]
                neg_sample_size = min(len(positive_samples) * 2, len(negative_samples))
                if neg_sample_size > 0:
                    negative_samples = negative_samples.sample(n=neg_sample_size, random_state=42)
                
                train_sample = pd.concat([positive_samples, negative_samples])
                feature_cols = [c for c in train_sample.columns if c.startswith(('agent_', 'target_'))]
                
                X_train_all.append(train_sample[feature_cols])
                y_train_all.append(train_sample['target'])
            
        if X_train_all:
            X_train, y_train = pd.concat(X_train_all), pd.concat(y_train_all)
            model = lgb.LGBMClassifier(**CFG.LGB_PARAMS)
            model.fit(X_train, y_train)
            dump(model, f'{behavior}_model.joblib')
            print(f"Model for {behavior} trained and saved.")
else:
    print("Detected Submission Environment. Skipping training.")


print("Starting Inference...")

models = {}
for behavior in CFG.BEHAVIORS:
    MODEL_DIR = "."
    if IS_SUBMISSION:
        # Example path. Make sure your training notebook output folder is named 'mabe-final-train'
        # or update this path accordingly.
        MODEL_DIR = "/kaggle/input/mabe-final-train" 
        
    model_path = os.path.join(MODEL_DIR, f'{behavior}_model.joblib')
    if os.path.exists(model_path):
        models[behavior] = load(model_path)
    else:
        models[behavior] = None 
        print(f"Warning: Model for {behavior} not found at {model_path}.")

all_events = []
test_files = glob(os.path.join(CFG.TEST_TRACKING_DIR, "*/*.parquet"))

for f in tqdm(test_files, desc="Processing test files"):
    video_id = int(os.path.basename(f).split('.')[0])
    
    df_tracking = pd.read_parquet(f)
    if 'mouse_id' not in df_tracking.columns: continue
    df_tracking['mouse_id'] = df_tracking['mouse_id'].astype(str).str.replace('mouse', '').astype(int)
    
    df_features = create_features(df_tracking)
    if df_features is None: continue
        
    feature_cols = [c for c in df_features.columns if c.startswith(('agent_', 'target_'))]
    
    for (agent_id, target_id), pair_df in df_features.groupby(['agent_id', 'target_id']):
        if pair_df.empty: continue
        
        X_test = pair_df[feature_cols]

        for behavior in CFG.BEHAVIORS:
            if models.get(behavior) is None: continue
            model = models[behavior]
            
            preds = model.predict_proba(X_test)[:, 1]
            binary_preds = (preds > CFG.PROB_THRESHOLD).astype(int)
            
            if np.sum(binary_preds) > 0:
                diffs = np.diff(binary_preds, prepend=0, append=0)
                starts = pair_df['video_frame'].values[np.where(diffs == 1)[0]]
                stops = pair_df['video_frame'].values[np.where(diffs == -1)[0]]
                
                for start, stop in zip(starts, stops):
                    if stop - start >= CFG.MIN_EVENT_LENGTH_FRAMES:
                        all_events.append({
                            'video_id': video_id, 'agent_id': f'mouse{agent_id}', 'target_id': f'mouse{target_id}',
                            'action': behavior, 'start_frame': start, 'stop_frame': stop
                        })

# --- Create Submission File ---
submission_df = pd.DataFrame(all_events)
if not submission_df.empty:
    submission_df['row_id'] = submission_df.index
    submission_df = submission_df[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
else:
    sample_sub = pd.read_csv(os.path.join(CFG.DATA_ROOT, "sample_submission.csv"))
    submission_df = pd.DataFrame(columns=sample_sub.columns)

submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
print(f"Generated {len(submission_df)} events.")
print(submission_df.head())

