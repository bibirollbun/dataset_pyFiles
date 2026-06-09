import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import os

# CNN Model (same as training)
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
        self._to_linear = None
        self._get_conv_output_size()
        
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 128, 126)  # Input shape [batch, channels, n_mels, time]
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            self._to_linear = x.numel() // x.shape[0]

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def process_audio_segment(audio_data, sample_rate, segment_duration, start_time):
    """Extract a segment from audio and process it into a mel-spectrogram."""
    try:
        start_samples = int(start_time * sample_rate)
        end_samples = start_samples + int(segment_duration * sample_rate)
        
        if end_samples > len(audio_data):
            audio_data = np.pad(audio_data, (0, end_samples - len(audio_data)), mode='constant')
        
        segment = audio_data[start_samples:end_samples]
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_db.astype(np.float32)
    except Exception as e:
        print(f"Error processing audio segment: {e}")
        return None

def predict_on_audio(audio_path, model, device, sample_rate, segment_duration, species_ids):
    """Run inference on a single audio file, segmenting into 5-second clips."""
    row_ids = []
    predictions = []
    soundscape_id = Path(audio_path).stem
    
    try:
        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=sample_rate)
        total_duration = len(audio_data) / sr
        total_segments = int(np.ceil(total_duration / segment_duration))
        
        model.eval()
        with torch.no_grad():
            for seg_idx in range(total_segments):
                start_time = seg_idx * segment_duration
                row_id = f"{soundscape_id}_{(seg_idx + 1) * segment_duration}"
                row_ids.append(row_id)
                
                # Process segment
                mel_spec = process_audio_segment(audio_data, sample_rate, segment_duration, start_time)
                if mel_spec is None:
                    # Fallback: uniform probabilities
                    predictions.append(np.full(len(species_ids), 1.0 / len(species_ids)))
                    print(f"Warning: Using uniform probabilities for {row_id} due to processing error")
                    continue
                
                mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict
                logits = model(mel_spec_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
                predictions.append(probs)
        
        return row_ids, predictions
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Fallback: generate row_ids with uniform probabilities
        total_segments = 3  # Assume 3 segments (5s, 10s, 15s) as in sample_submission.csv
        for seg_idx in range(total_segments):
            row_id = f"{soundscape_id}_{(seg_idx + 1) * segment_duration}"
            row_ids.append(row_id)
            predictions.append(np.full(len(species_ids), 1.0 / len(species_ids)))
        return row_ids, predictions

def create_submission(row_ids, predictions, species_ids, sample_submission_path, output_path):
    """Create submission DataFrame and save to CSV."""
    print("Creating submission dataframe...")
    
    try:
        # Create submission dictionary
        submission_dict = {'row_id': row_ids}
        for i, species in enumerate(species_ids):
            submission_dict[species] = [pred[i] for pred in predictions]
        
        # Create DataFrame
        submission_df = pd.DataFrame(submission_dict)
        
        # Load sample submission to ensure correct column order
        sample_sub = pd.read_csv(sample_submission_path)
        expected_cols = sample_sub.columns
        
        # Check for missing columns
        missing_cols = set(expected_cols) - set(submission_df.columns)
        if missing_cols:
            print(f"Warning: Missing {len(missing_cols)} species columns in submission")
            for col in missing_cols:
                if col != 'row_id':
                    submission_df[col] = 0.0
        
        # Ensure correct column order
        submission_df = submission_df[expected_cols]
        
        # Ensure all expected row_ids are present
        expected_row_ids = set(sample_sub['row_id'])
        missing_row_ids = expected_row_ids - set(submission_df['row_id'])
        if missing_row_ids:
            print(f"Warning: Missing {len(missing_row_ids)} row_ids in submission")
            for row_id in missing_row_ids:
                # Append missing row_id with uniform probabilities
                new_row = {'row_id': row_id}
                for col in expected_cols[1:]:  # Exclude 'row_id'
                    new_row[col] = 1.0 / len(species_ids)
                submission_df = pd.concat([submission_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    except Exception as e:
        print(f"Error creating submission: {e}")
        # Fallback: create a submission with uniform probabilities
        submission_df = pd.read_csv(sample_submission_path)
        for col in submission_df.columns[1:]:
            submission_df[col] = 1.0 / len(species_ids)
        submission_df.to_csv(output_path, index=False)
        print(f"Fallback submission saved to {output_path}")

def main():
    # Parameters
    test_dir = '/kaggle/input/birdclef-2025/test_soundscapes'  # Kaggle test path
    model_path = '/kaggle/input/bird-seeker-3/pytorch/default/1/audio_cnn_model_finetuned_10.pth'
    sample_submission_path = '/kaggle/input/birdclef-2025/sample_submission.csv'
    output_path = '/kaggle/working/submission.csv'  # Kaggle output path
    sample_rate = 16000
    segment_duration = 5  # 5-second segments to match row_id format
    
    # Load species from sample_submission.csv
    try:
        sample_submission = pd.read_csv(sample_submission_path)
        species_ids = sample_submission.columns[1:].tolist()  # Exclude 'row_id'
        num_classes = len(species_ids)
    except Exception as e:
        print(f"Error loading sample_submission.csv: {e}")
        return
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")
    
    # Initialize and load model
    try:
        model = AudioCNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        # Fallback: create a submission with uniform probabilities
        submission_df = pd.read_csv(sample_submission_path)
        for col in submission_df.columns[1:]:
            submission_df[col] = 1.0 / len(species_ids)
        submission_df.to_csv(output_path, index=False)
        print(f"Fallback submission saved to {output_path}")
        return
    
    # Find test audio files
    try:
        test_files = list(Path(test_dir).glob('*.ogg'))
         # Run inference
        all_row_ids = []
        all_predictions = []
        for audio_path in tqdm(test_files, desc="Processing audio files"):
            row_ids, predictions = predict_on_audio(audio_path, model, device, sample_rate, segment_duration, species_ids)
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
    except Exception as e:
        print(f"Error accessing test directory {test_dir}: {e}")
        return
    
   
    
    # Create and save submission
    create_submission(all_row_ids, all_predictions, species_ids, sample_submission_path, output_path)

if __name__ == "__main__":
    main()

