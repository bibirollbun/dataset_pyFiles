import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MABe Kaggle Submission - Inference Only")
print("="*60)


import torch.nn.functional as F

class Conv1DBiLSTM(nn.Module):
    """
    Conv1D + BiLSTM Model (matches checkpoint architecture)
    """

    def __init__(self, input_dim, num_classes,
                 conv_channels=[64, 128, 256],
                 lstm_hidden=256, lstm_layers=2,
                 dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Conv layers
        conv_layers = []
        in_channels = input_dim
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2)
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)
        self.pooling_factor = 2 ** len(conv_channels)

        # Feature projection
        self.feature_projection = nn.Linear(conv_channels[-1], lstm_hidden)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape

        # Conv: [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        conv_out = self.conv_layers(x)
        conv_out = conv_out.transpose(1, 2)

        # Feature projection
        lstm_in = self.feature_projection(conv_out)

        # BiLSTM
        lstm_out, _ = self.lstm(lstm_in)

        # Upsample to original seq_len
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = F.interpolate(lstm_out, size=seq_len, mode='linear', align_corners=False)
        lstm_out = lstm_out.transpose(1, 2)

        # Classify
        output = self.classifier(lstm_out)

        return output

print("âœ“ Conv1DBiLSTM model defined")


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Load checkpoint
MODEL_PATH = Path('/kaggle/input/mabe-submit/best_model.pth')
checkpoint = torch.load(MODEL_PATH, map_location=device)

print(f"âœ“ Loaded checkpoint")
print(f"  Epoch: {checkpoint['epoch'] + 1}")
if 'best_val_f1' in checkpoint:
    print(f"  Val F1: {checkpoint['best_val_f1']:.4f}")
print()

# Build model (exact config from checkpoint)
model = Conv1DBiLSTM(
    input_dim=142,  # 71 keypoints Ã— 2
    num_classes=4,
    conv_channels=[64, 128, 256],
    lstm_hidden=256,
    lstm_layers=2,
    dropout=0.3,
)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"âœ“ Model ready: Conv1DBiLSTM")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


# Try both possible dataset paths (case variations)
possible_paths = [
    Path('/kaggle/input/MABe-mouse-behavior-detection'),  # Official dataset
    Path('/kaggle/input/mabe-mouse-behavior-detection'),  # Lowercase
]

DATA_DIR = None
for path in possible_paths:
    if path.exists():
        DATA_DIR = path
        break

if DATA_DIR is None:
    raise FileNotFoundError("Cannot find MABe dataset. Please add it to notebook inputs.")

print(f"âœ“ Using dataset: {DATA_DIR}")

# Load test metadata
if (DATA_DIR / 'test.csv').exists():
    test_csv = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"  Loaded test.csv with {len(test_csv)} videos")
elif (DATA_DIR / 'sample_submission.csv').exists():
    # Use sample submission to get video list
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    test_videos = sample_sub['video_id'].unique()
    # Infer lab_id from test_tracking directory structure
    test_labs = []
    for lab_dir in (DATA_DIR / 'test_tracking').iterdir():
        if lab_dir.is_dir():
            test_labs.append(lab_dir.name)
    
    # Create test_csv from available videos in test_tracking
    test_data = []
    for lab_id in test_labs:
        lab_dir = DATA_DIR / 'test_tracking' / lab_id
        for video_file in lab_dir.glob('*.parquet'):
            video_id = video_file.stem
            test_data.append({'video_id': video_id, 'lab_id': lab_id})
    
    test_csv = pd.DataFrame(test_data)
    print(f"  Constructed test set from test_tracking: {len(test_csv)} videos")
else:
    raise FileNotFoundError("Cannot find test.csv or sample_submission.csv")

print(f"  Total test videos: {len(test_csv)}")


print("Generating predictions...")
print()

all_predictions = []
sequence_length = 100
stride = 25  # 75% overlap

with torch.no_grad():
    for idx, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Processing"):
        video_id = row['video_id']
        lab_id = row['lab_id']
        
        # Load tracking
        tracking_file = DATA_DIR / 'test_tracking' / lab_id / f'{video_id}.parquet'
        if not tracking_file.exists():
            continue
        
        try:
            tracking_df = pd.read_parquet(tracking_file)
            
            # Convert to wide format (142 features)
            tracking_pivot = tracking_df.pivot_table(
                index='video_frame',
                columns=['mouse_id', 'bodypart'],
                values=['x', 'y'],
                aggfunc='first'
            )
            tracking_pivot.columns = ['_'.join(map(str, col)).strip() 
                                      for col in tracking_pivot.columns.values]
            tracking_pivot = tracking_pivot.sort_index()
            
            keypoints = tracking_pivot.values.astype(np.float32)
            keypoints = np.nan_to_num(keypoints, nan=0.0)
            
            # Ensure exactly 142 features
            if keypoints.shape[1] != 142:
                if keypoints.shape[1] < 142:
                    # Pad with zeros
                    padding = np.zeros((keypoints.shape[0], 142 - keypoints.shape[1]), dtype=np.float32)
                    keypoints = np.concatenate([keypoints, padding], axis=1)
                else:
                    # Truncate
                    keypoints = keypoints[:, :142]
            
            num_frames = len(keypoints)
            
            # Sliding window predictions
            video_preds = np.zeros((num_frames, 4), dtype=np.float32)
            video_counts = np.zeros(num_frames, dtype=np.int32)
            
            for start_idx in range(0, max(1, num_frames - sequence_length + 1), stride):
                end_idx = min(start_idx + sequence_length, num_frames)
                
                # Handle last window
                if end_idx - start_idx < sequence_length:
                    start_idx = max(0, num_frames - sequence_length)
                    end_idx = num_frames
                
                window = keypoints[start_idx:end_idx]
                
                # Pad if needed
                if len(window) < sequence_length:
                    padding = np.zeros((sequence_length - len(window), 142), dtype=np.float32)
                    window = np.concatenate([window, padding], axis=0)
                
                # Predict
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
                output = model(window_tensor)
                probs = torch.softmax(output, dim=-1).squeeze(0).cpu().numpy()
                
                # Accumulate
                actual_length = min(sequence_length, end_idx - start_idx)
                video_preds[start_idx:start_idx + actual_length] += probs[:actual_length]
                video_counts[start_idx:start_idx + actual_length] += 1
            
            # Average overlapping predictions
            video_counts = np.maximum(video_counts, 1)
            video_preds = video_preds / video_counts[:, np.newaxis]
            final_preds = np.argmax(video_preds, axis=1)
            
            # Create submission rows
            for frame_idx, pred in enumerate(final_preds):
                all_predictions.append({
                    'video_id': video_id,
                    'frame': frame_idx,
                    'prediction': int(pred),
                })
        
        except Exception as e:
            print(f"Error: {video_id} - {e}")
            continue

print(f"\nâœ“ Generated {len(all_predictions):,} predictions")


submission_df = pd.DataFrame(all_predictions)
submission_df = submission_df.sort_values(['video_id', 'frame']).reset_index(drop=True)

print("="*60)
print("Converting to behavior intervals...")
print("="*60)

# Convert frame-level predictions to behavior intervals
class_names = {0: 'background', 1: 'social', 2: 'mating', 3: 'aggressive'}
action_mapping = {
    1: 'social',      # Social behaviors
    2: 'mating',      # Mating behaviors  
    3: 'aggressive',  # Aggressive behaviors
}

submission_rows = []
row_id = 0

for video_id in submission_df['video_id'].unique():
    video_preds = submission_df[submission_df['video_id'] == video_id].sort_values('frame')
    
    # Group consecutive frames with same prediction
    current_action = None
    start_frame = None
    
    for idx, row in video_preds.iterrows():
        frame = row['frame']
        pred = row['prediction']
        
        # Skip background (class 0)
        if pred == 0:
            if current_action is not None:
                # End previous action
                submission_rows.append({
                    'row_id': row_id,
                    'video_id': video_id,
                    'agent_id': 'mouse1',  # Default agent
                    'target_id': 'mouse2',  # Default target
                    'action': action_mapping.get(current_action, 'social'),
                    'start_frame': start_frame,
                    'stop_frame': frame - 1
                })
                row_id += 1
                current_action = None
            continue
        
        # Start new action or continue current
        if current_action != pred:
            if current_action is not None:
                # End previous action
                submission_rows.append({
                    'row_id': row_id,
                    'video_id': video_id,
                    'agent_id': 'mouse1',
                    'target_id': 'mouse2',
                    'action': action_mapping.get(current_action, 'social'),
                    'start_frame': start_frame,
                    'stop_frame': frame - 1
                })
                row_id += 1
            
            # Start new action
            current_action = pred
            start_frame = frame
    
    # Close any remaining action
    if current_action is not None and current_action != 0:
        submission_rows.append({
            'row_id': row_id,
            'video_id': video_id,
            'agent_id': 'mouse1',
            'target_id': 'mouse2',
            'action': action_mapping.get(current_action, 'social'),
            'start_frame': start_frame,
            'stop_frame': frame
        })
        row_id += 1

# Create final submission
final_submission = pd.DataFrame(submission_rows)

# Add one row per video if no behaviors detected
for video_id in submission_df['video_id'].unique():
    if video_id not in final_submission['video_id'].values:
        final_submission = pd.concat([final_submission, pd.DataFrame([{
            'row_id': row_id,
            'video_id': video_id,
            'agent_id': 'mouse1',
            'target_id': 'mouse2',
            'action': 'social',
            'start_frame': 0,
            'stop_frame': 1
        }])], ignore_index=True)
        row_id += 1

print(f"âœ“ Converted to {len(final_submission)} behavior intervals")
print(f"  Action distribution:")
for action, count in final_submission['action'].value_counts().items():
    print(f"    {action}: {count}")

# Save in correct format
final_submission = final_submission[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
final_submission.to_csv('/kaggle/working/submission.csv', index=False)

print(f"\nâœ“ Saved to /kaggle/working/submission.csv")
print(f"  Total intervals: {len(final_submission):,}")
print(f"  Unique videos: {final_submission['video_id'].nunique()}")
print("\nğŸ�¯ Submission ready!")


# Preview submission
submission_df.head(20)


# Preview submission
submission_df.head(20)

