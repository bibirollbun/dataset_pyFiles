import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# Define paths
DATA_PATH = Path('/kaggle/input/MABe-mouse-behavior-detection')
TRAIN_CSV = DATA_PATH / 'train.csv'
TEST_CSV = DATA_PATH / 'test.csv'
TRAIN_TRACKING = DATA_PATH / 'train_tracking'
TEST_TRACKING = DATA_PATH / 'test_tracking'
TRAIN_ANNOTATION = DATA_PATH / 'train_annotation'

# Load metadata
train_meta = pd.read_csv(TRAIN_CSV)
test_meta = pd.read_csv(TEST_CSV)

print(f"Train videos: {len(train_meta)}")
print(f"Test videos: {len(test_meta)}")
print("\nTrain metadata columns:")
print(train_meta.columns.tolist())


def load_tracking_data(video_id, data_path, max_frames=None):
    """Load tracking data for a specific video"""
    tracking_file = data_path / f"{video_id}.parquet"
    
    if not tracking_file.exists():
        return None
    
    df = pd.read_parquet(tracking_file)
    
    if max_frames:
        df = df[df['video_frame'] < max_frames]
    
    return df

def load_annotation_data(video_id, annotation_path):
    """Load annotation data for a specific video"""
    annotation_file = annotation_path / f"{video_id}.parquet"
    
    if not annotation_file.exists():
        return None
    
    return pd.read_parquet(annotation_file)


def explore_data(train_meta):
    """Explore the training metadata"""

    
    # Lab distribution
    print("\nLab Distribution:")
    print(train_meta['lab_id'].value_counts())
    
    # Behaviors per lab
    print("\nUnique behaviors across all labs:")
    all_behaviors = set()
    for behaviors in train_meta['behaviors_labeled'].dropna():
        beh_list = eval(behaviors)
        
        for i in beh_list:
            buf_behavior = i.split(',')[2]
            all_behaviors.add(buf_behavior)
                
    print(f"Total unique behaviors: {len(all_behaviors)}")
    print(sorted(all_behaviors))
    
    # Tracking methods
    print("\nTracking Methods:")
    print(train_meta['tracking_method'].value_counts())
    
    # Arena types
    print("\nArena Types:")
    print(train_meta['arena_type'].value_counts())
    
    # Video statistics
    print("\nVideo Duration Statistics:")
    print(train_meta['video_duration_sec'].describe())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Lab distribution
    train_meta['lab_id'].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Videos per Lab')
    axes[0, 0].set_xlabel('Lab ID')
    axes[0, 0].set_ylabel('Count')
    
    # Video duration
    axes[0, 1].hist(train_meta['video_duration_sec'], bins=30, edgecolor='black')
    axes[0, 1].set_title('Video Duration Distribution')
    axes[0, 1].set_xlabel('Duration (seconds)')
    axes[0, 1].set_ylabel('Count')
    
    # FPS distribution
    train_meta['frames_per_second'].value_counts().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Frames Per Second Distribution')
    axes[1, 0].set_xlabel('FPS')
    axes[1, 0].set_ylabel('Count')
    
    # Tracking method
    train_meta['tracking_method'].value_counts().plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Tracking Methods')
    axes[1, 1].set_xlabel('Method')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

explore_data(train_meta)


# Sample: Load one video for exploration
sample = train_meta.iloc[3]
print(f"\nLoading sample video: {sample['video_id']}")

sample_tracking = load_tracking_data(sample['video_id'], TRAIN_TRACKING / sample['lab_id'])
sample_annotation = load_annotation_data(sample['video_id'], TRAIN_ANNOTATION / sample['lab_id'])

if sample_tracking is not None:
    print(f"\nTracking data shape: {sample_tracking.shape}")
    print(sample_tracking.head(10))
    
if sample_annotation is not None:
    print(f"\nAnnotation data shape: {sample_annotation.shape}")
    print(sample_annotation.head())


def compute_distance(x1, y1, x2, y2):
    """Compute Euclidean distance"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def compute_velocity(df, bodypart='centroid'):
    """Compute velocity for a body part"""
    df_part = df[df['bodypart'] == bodypart].sort_values('video_frame')
    
    dx = df_part['x'].diff()
    dy = df_part['y'].diff()
    velocity = np.sqrt(dx**2 + dy**2)
    
    return velocity

def extract_features_from_tracking(tracking_df, window_size=30):
    """
    Extract features from tracking data for behavior classification
    
    Features include:
    - Position statistics (mean, std, min, max)
    - Velocity and acceleration
    - Distance between mice
    - Body part angles and orientations
    - Movement patterns
    """
    
    features_list = []
    
    # Get unique mice and body parts
    mice = tracking_df['mouse_id'].unique()
    bodyparts = tracking_df['bodypart'].unique()
    frames = sorted(tracking_df['video_frame'].unique())
    
    # Process in windows
    for i in range(0, len(frames), window_size // 2):  # 50% overlap
        window_frames = frames[i:i + window_size]
        
        if len(window_frames) < window_size // 2:
            continue
        
        window_data = tracking_df[tracking_df['video_frame'].isin(window_frames)]
        
        feature_dict = {
            'start_frame': window_frames[0],
            'end_frame': window_frames[-1],
            'center_frame': (window_frames[0] + window_frames[-1]) // 2
        }
        
        # Features for each mouse
        for mouse in mice:
            mouse_data = window_data[window_data['mouse_id'] == mouse]
            
            for bodypart in bodyparts:
                part_data = mouse_data[mouse_data['bodypart'] == bodypart].sort_values('video_frame')
                
                if len(part_data) == 0:
                    continue
                
                # Position statistics
                feature_dict[f'mouse{mouse}_{bodypart}_x_mean'] = part_data['x'].mean()
                feature_dict[f'mouse{mouse}_{bodypart}_x_std'] = part_data['x'].std()
                feature_dict[f'mouse{mouse}_{bodypart}_y_mean'] = part_data['y'].mean()
                feature_dict[f'mouse{mouse}_{bodypart}_y_std'] = part_data['y'].std()
                
                # Velocity
                dx = part_data['x'].diff()
                dy = part_data['y'].diff()
                velocity = np.sqrt(dx**2 + dy**2)
                feature_dict[f'mouse{mouse}_{bodypart}_velocity_mean'] = velocity.mean()
                feature_dict[f'mouse{mouse}_{bodypart}_velocity_max'] = velocity.max()
                
                # Acceleration
                acceleration = velocity.diff()
                feature_dict[f'mouse{mouse}_{bodypart}_accel_mean'] = acceleration.mean()
        
        # Inter-mouse features
        if len(mice) >= 2:
            for i, mouse1 in enumerate(mice):
                for mouse2 in mice[i+1:]:
                    # Distance between centroids (if available)
                    m1_data = window_data[(window_data['mouse_id'] == mouse1) & 
                                         (window_data['bodypart'] == bodyparts[0])]
                    m2_data = window_data[(window_data['mouse_id'] == mouse2) & 
                                         (window_data['bodypart'] == bodyparts[0])]
                    
                    if len(m1_data) > 0 and len(m2_data) > 0:
                        # Merge on frame to compute distances
                        merged = pd.merge(m1_data[['video_frame', 'x', 'y']], 
                                        m2_data[['video_frame', 'x', 'y']], 
                                        on='video_frame', suffixes=('_1', '_2'))
                        
                        distances = compute_distance(merged['x_1'], merged['y_1'], 
                                                    merged['x_2'], merged['y_2'])
                        
                        feature_dict[f'mouse{mouse1}_mouse{mouse2}_dist_mean'] = distances.mean()
                        feature_dict[f'mouse{mouse1}_mouse{mouse2}_dist_min'] = distances.min()
                        feature_dict[f'mouse{mouse1}_mouse{mouse2}_dist_std'] = distances.std()
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)

# Test feature extraction on sample video
if sample_tracking is not None:
    print("\nExtracting features from sample video...")
    sample_features = extract_features_from_tracking(sample_tracking, window_size=30)
    print(f"Extracted {len(sample_features)} feature windows")
    print(f"Number of features: {len(sample_features.columns)}")
    print("\nSample features:")
    print(sample_features.head())


def prepare_training_data(train_meta, tracking_path, annotation_path, 
                         max_videos=None, window_size=30):
    """
    Prepare training data from videos
    
    Returns features (X) and labels (y)
    """

    
    all_features = []
    all_labels = []
    all_metadata = []
    
    if max_videos:
        train_meta = train_meta.head(max_videos)
    
    for idx, sample in train_meta.iterrows():
        video_id = sample['video_id']
        lab_id = sample['lab_id']
        if idx % 10 == 0:
            print(f"Processing video {idx+1}/{len(train_meta)}: {video_id}")
        
        # Load tracking and annotation
        tracking_path_iteration = tracking_path / f"{lab_id}"
        annotation_iteration = annotation_path / f"{lab_id}"
        tracking = load_tracking_data(video_id, tracking_path_iteration)
        annotation = load_annotation_data(video_id, annotation_iteration)
        
        if tracking is None or annotation is None:
            continue
        
        # Extract features
        features = extract_features_from_tracking(tracking, window_size=window_size)
        
        if len(features) == 0:
            continue
        
        # Match features with annotations
        for _, row in annotation.iterrows():
            start_frame = row['start_frame']
            end_frame = row['stop_frame']
            
            # Find feature windows that overlap with this annotation
            overlapping = features[
                ((features['start_frame'] >= start_frame) & (features['start_frame'] <= end_frame)) |
                ((features['end_frame'] >= start_frame) & (features['end_frame'] <= end_frame)) |
                ((features['start_frame'] <= start_frame) & (features['end_frame'] >= end_frame))
            ]
            
            for _, feat_row in overlapping.iterrows():
                all_features.append(feat_row)
                all_labels.append(row['action'])
                all_metadata.append({
                    'video_id': video_id,
                    'agent_id': row['agent_id'],
                    'target_id': row['target_id']
                })
    
    X = pd.DataFrame(all_features)
    y = pd.Series(all_labels)
    metadata = pd.DataFrame(all_metadata)
    
    return X, y, metadata

# Prepare training data

X_train, y_train, train_metadata = prepare_training_data(
     
    train_meta, 
    TRAIN_TRACKING, 
    TRAIN_ANNOTATION,
    max_videos=20,  
    window_size=30
)

print(f"\nTraining data prepared:")
print(f"Features shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")
print(f"\nBehavior distribution:")
print(y_train.value_counts())


# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Remove non-numeric columns for training
feature_cols = [col for col in X_train.columns if col not in ['start_frame', 'end_frame', 'center_frame']]
X_train_clean = X_train[feature_cols].fillna(0)

print(f"\nTraining with {len(feature_cols)} features")

# Split data
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_clean, y_train_encoded, 
    test_size=0.2, random_state=SEED, stratify=y_train_encoded
)

print(f"Train set: {len(X_tr)}, Validation set: {len(X_val)}")

# Train LightGBM model
print("\nTraining LightGBM model...")
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(label_encoder.classes_),
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': SEED
}

train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

# Evaluate
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

f1 = f1_score(y_val, y_val_pred_classes, average='weighted')
print(f"\nValidation F1 Score (weighted): {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_classes, 
                          target_names=label_encoder.classes_, zero_division=0))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 20 Important Features:")
print(feature_importance.head(20))

# Plot feature importance
plt.figure(figsize=(10, 8))
feature_importance.head(20).plot(x='feature', y='importance', kind='barh')
plt.title('Top 20 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


def generate_predictions(model, label_encoder, test_meta, tracking_path, 
                        feature_cols, window_size=30, threshold=0.5):
    """Generate predictions for test videos"""
    
    predictions = []
    row_id = 0
    
    for idx, row in test_meta.iterrows():
        video_id = row['video_id']
        
        if idx % 10 == 0:
            print(f"Predicting video {idx+1}/{len(test_meta)}: {video_id}")
        
        # Load tracking
        tracking = load_tracking_data(video_id, tracking_path / row['lab_id'])
        
        if tracking is None:
            continue
        
        # Extract features
        features = extract_features_from_tracking(tracking, window_size=window_size)
        
        if len(features) == 0:
            continue
        
        # Get features in correct order
        X_test = features[feature_cols].fillna(0)
        
        # Predict
        pred_probs = model.predict(X_test)
        pred_classes = np.argmax(pred_probs, axis=1)
        max_probs = np.max(pred_probs, axis=1)
        
        # Get behaviors labeled for this video
        behaviors_labeled = str(row.get('behaviors_labeled', ''))
        valid_behaviors = set()
        beh_list = eval(behaviors)
        for i in beh_list:
            buf_behavior = i.split(',')[2]
            valid_behaviors.add(buf_behavior)

        
        
        # Convert predictions to actions
        for i, (pred_class, prob) in enumerate(zip(pred_classes, max_probs)):
            if prob < threshold:
                continue
            
            action = label_encoder.inverse_transform([pred_class])[0]
            
            # Filter by valid behaviors if specified
            if valid_behaviors and action not in valid_behaviors:
                continue
            
            
            mice = tracking['mouse_id'].unique()
            agent_id = mice[0] if len(mice) > 0 else 1
            target_id = mice[1] if len(mice) > 1 else agent_id
            
            predictions.append({
                'row_id': row_id,
                'video_id': video_id,
                'agent_id': int(agent_id),
                'target_id': int(target_id),
                'action': action,
                'start_frame': int(features.iloc[i]['start_frame']),
                'stop_frame': int(features.iloc[i]['end_frame'])
            })
            
            row_id += 1
    
    return pd.DataFrame(predictions)

# Generate predictions
print("\nGenerating predictions for test set...")
test_predictions = generate_predictions(
    model, 
    label_encoder, 
    test_meta.head(10), 
    TEST_TRACKING,
    feature_cols,
    window_size=30,
    threshold=0.3
)

print(f"\nGenerated {len(test_predictions)} predictions")
print(test_predictions.head(10))


# Prepare submission
submission = test_predictions[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]

submission['agent_id'] = 'mouse'+submission['agent_id'].astype(str)
submission['target_id'] = 'mouse'+submission['target_id'].astype(str)

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"Total predictions: {len(submission)}")

