# System setup
import sys
import os
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from tqdm.auto import tqdm
from scipy.ndimage import binary_dilation

# Add model code to path
sys.path.append('../input/mabe-resnet1d-weights/src')

# Import custom modules
from feature_extractor import MouseFeatureExtractor
from resnet1d import ResNet1D

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load model weights
print("ğŸ“‚ Loading model...")

# Load behavior mapping
with open('../input/mabe-resnet1d-weights/models/behavior_mapping.json', 'r') as f:
    mapping_data = json.load(f)
    
behavior_to_id = mapping_data['behavior_to_id']
id_to_behavior = {v: k for k, v in behavior_to_id.items()}
num_classes = len(behavior_to_id)

print(f"   Behaviors: {num_classes}")

# Create model
model = ResNet1D(
    input_dim=52,
    embed_dim=256,
    num_behaviors=num_classes
)

# Load checkpoint
checkpoint = torch.load(
    '../input/mabe-resnet1d-weights/models/best_model.pt',
    map_location=device
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"   âœ… Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")


# Load normalization stats
print("ğŸ“Š Loading feature normalization stats...")
feature_stats = np.load('../input/mabe-resnet1d-weights/feature_stats.npz')
feature_mean = feature_stats['mean']
feature_std = feature_stats['std']
print(f"   Mean shape: {feature_mean.shape}")
print(f"   Std shape: {feature_std.shape}")


# Load test metadata
print("ğŸ“¹ Loading test data...")
test_df = pd.read_csv('../input/MABe-mouse-behavior-detection/test.csv')
print(f"   Test videos: {len(test_df)}")
print(test_df.head())


# Extract features for test videos
print("\n" + "="*80)
print("ğŸ”§ EXTRACTING TEST FEATURES")
print("="*80)

extractor = MouseFeatureExtractor()
test_features = {}

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting"):
    video_id = str(row['video_id'])
    lab_id = row['lab_id']
    
    # Load tracking data - FIXED PATH: MABe-mouse-behavior-detection (capital M and B)
    tracking_path = Path(f'../input/MABe-mouse-behavior-detection/test_tracking/{lab_id}/{video_id}.parquet')
    
    if not tracking_path.exists():
        print(f"   Warning: Tracking data not found for {video_id}")
        continue
    
    # Get frame count for reporting
    tracking_df = pd.read_parquet(tracking_path)
    n_frames = tracking_df['video_frame'].max() + 1
    print(f"\n   Video {video_id}: {n_frames:,} frames")
    
    # Extract features using the correct method
    features = extractor.extract_features_from_file(str(tracking_path), fps=30.0)
    
    # Normalize
    features = (features - feature_mean) / (feature_std + 1e-8)
    
    test_features[video_id] = features
    print(f"      Features shape: {features.shape}")

print(f"\nâœ… Feature extraction complete!")
print(f"   Total videos: {len(test_features)}")


# Prediction functions
def predict_video(model, features, device, window_size=200, stride=10, batch_size=512):
    """Run frame-level predictions on a video"""
    num_frames = len(features)
    num_classes = model.classifier[-1].out_features
    
    predictions = np.zeros((num_frames, num_classes), dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.float32)
    
    windows = []
    window_ranges = []
    
    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        
        window = features[start:end]
        windows.append(window)
        window_ranges.append((start, end))
        
        if len(windows) == batch_size:
            X = np.array(windows, dtype=np.float32)
            X = np.transpose(X, (0, 2, 1))  # (batch, 52, window_size)
            X = torch.from_numpy(X).to(device)
            
            with torch.no_grad():
                logits = model(X, mode='classify')
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Assign predictions to ALL frames in the window, not just center
            for prob, (start, end) in zip(probs, window_ranges):
                predictions[start:end] += prob
                counts[start:end] += 1
            
            windows = []
            window_ranges = []
    
    # Process remaining
    if windows:
        X = np.array(windows, dtype=np.float32)
        X = np.transpose(X, (0, 2, 1))
        X = torch.from_numpy(X).to(device)
        
        with torch.no_grad():
            logits = model(X, mode='classify')
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        for prob, (start, end) in zip(probs, window_ranges):
            predictions[start:end] += prob
            counts[start:end] += 1
    
    # Average predictions across overlapping windows
    counts[counts == 0] = 1
    predictions /= counts[:, np.newaxis]
    
    return predictions


def parse_behavior_label(behavior_str):
    """Parse behavior string into components"""
    parts = behavior_str.split(',')
    
    if len(parts) == 3:
        agent_str, target_str, action = parts
        agent_id = int(agent_str.replace('mouse', ''))
        target_id = int(target_str.replace('mouse', ''))
        return agent_id, target_id, action
    else:
        return None, None, behavior_str


def predictions_to_intervals(predictions, id_to_behavior, 
                            min_length=15, confidence_threshold=0.3):
    """Convert frame-level predictions to behavior intervals"""
    num_frames, num_classes = predictions.shape
    intervals = []
    
    for class_id in range(num_classes):
        behavior_str = id_to_behavior[class_id]
        probs = predictions[:, class_id]
        
        # Threshold
        binary = (probs > confidence_threshold).astype(np.uint8)
        
        # Smooth
        binary = binary_dilation(binary, structure=np.ones(5))
        
        # Find intervals
        in_interval = False
        start = 0
        
        for frame in range(num_frames):
            if binary[frame] and not in_interval:
                start = frame
                in_interval = True
            elif not binary[frame] and in_interval:
                end = frame - 1
                length = end - start + 1
                
                if length >= min_length:
                    agent_id, target_id, action = parse_behavior_label(behavior_str)
                    if agent_id is not None:
                        intervals.append((agent_id, target_id, action, start, end))
                
                in_interval = False
        
        # Handle interval at end
        if in_interval:
            end = num_frames - 1
            length = end - start + 1
            
            if length >= min_length:
                agent_id, target_id, action = parse_behavior_label(behavior_str)
                if agent_id is not None:
                    intervals.append((agent_id, target_id, action, start, end))
    
    return intervals

print("âœ… Prediction functions defined")


# Run predictions
print("\n" + "="*80)
print("ğŸ�¯ GENERATING PREDICTIONS")
print("="*80)

all_intervals = []

for video_id, features in tqdm(test_features.items(), desc="Predicting"):
    print(f"\n   Video {video_id}: {len(features):,} frames")
    
    # Predict
    predictions = predict_video(
        model, features, device,
        window_size=200,
        stride=10,
        batch_size=512
    )
    
    # Debug: Check prediction statistics
    max_probs = predictions.max(axis=1)
    print(f"      Prediction stats:")
    print(f"        Max prob per frame: min={max_probs.min():.4f}, max={max_probs.max():.4f}, mean={max_probs.mean():.4f}")
    print(f"        Frames above 0.20: {(max_probs > 0.20).sum()}")
    print(f"        Frames above 0.15: {(max_probs > 0.15).sum()}")
    print(f"        Frames above 0.10: {(max_probs > 0.10).sum()}")
    
    # Convert to intervals with MORE LENIENT parameters
    intervals = predictions_to_intervals(
        predictions, id_to_behavior,
        min_length=10,  # Reduced from 15 to 10 frames (0.33 seconds)
        confidence_threshold=0.15  # Lowered from 0.2 to 0.15
    )
    
    print(f"      Found {len(intervals)} behavior intervals")
    
    # Add video_id
    for agent_id, target_id, action, start, stop in intervals:
        all_intervals.append({
            'video_id': video_id,
            'agent_id': f'mouse{agent_id}',
            'target_id': f'mouse{target_id}',
            'action': action,
            'start_frame': start,
            'stop_frame': stop
        })

print(f"\nâœ… Predictions complete!")
print(f"   Total intervals: {len(all_intervals):,}")


# Create submission file
print("\n" + "="*80)
print("ğŸ“„ CREATING SUBMISSION FILE")
print("="*80)

if all_intervals:
    submission_df = pd.DataFrame(all_intervals)
    
    # Add row_id
    submission_df.insert(0, 'row_id', range(len(submission_df)))
    
    # Reorder columns
    submission_df = submission_df[[
        'row_id', 'video_id', 'agent_id', 'target_id',
        'action', 'start_frame', 'stop_frame'
    ]]
else:
    # Empty submission
    submission_df = pd.DataFrame(columns=[
        'row_id', 'video_id', 'agent_id', 'target_id',
        'action', 'start_frame', 'stop_frame'
    ])

# Save
submission_df.to_csv('submission.csv', index=False)

print(f"\nâœ… Submission file created!")
print(f"   Predictions: {len(submission_df):,}")

if len(submission_df) > 0:
    print("\nğŸ”� Sample predictions:")
    print(submission_df.head(10))
    
    print("\nğŸ“Š Action distribution:")
    action_counts = submission_df['action'].value_counts()
    for action, count in action_counts.items():
        print(f"   {action}: {count:,}")
else:
    print("\nâš ï¸�  No predictions generated!")


# Verify submission format
print("\n" + "="*80)
print("âœ… FINAL VALIDATION")
print("="*80)

# Check format
required_columns = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
assert list(submission_df.columns) == required_columns, "Columns mismatch!"
assert submission_df.isnull().sum().sum() == 0, "Missing values found!"

print("âœ… All validation checks passed!")
print(f"ğŸ“„ Submission ready: submission.csv ({len(submission_df):,} rows)")
print("\nğŸ�¯ Ready for Kaggle submission!")

