"""
# BirdCLEF 2025 - Fixed RegNet Inference Code

This notebook loads a trained RegNet model and creates predictions for the BirdCLEF 2025 test data.
It fixes the previous issues with the submission format to ensure correct evaluation.
"""

import os
import gc
import warnings
import logging
import time
import numpy as np
import pandas as pd
import librosa
import cv2
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Configuration class
class CFG:
    # Paths
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    sample_submission = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/regnet-008-15-epochs/pytorch/default/1/best_model.pth'
    
    # Audio parameters
    sample_rate = 32000
    duration = 5  # seconds
    
    # Mel spectrogram parameters - must match training settings
    n_fft = 1024
    hop_length = 512
    n_mels = 64
    fmin = 50
    fmax = 14000
    
    # Image parameters
    img_size = 224
    
    # Model parameters
    model_name = 'regnety_008'  # Must match training model
    
    # Inference parameters
    batch_size = 32
    threshold = 0.5  # Confidence threshold for positive predictions
    
    # Test-time augmentation
    use_tta = True  # Enable TTA for better performance
    tta_steps = 3
    
    # Debug
    debug_mode = False    # Set to True to process only a few soundscapes
    debug_count = 3       # Number of soundscapes to process in debug mode
    
    # Device
    device = "cpu"

cfg = CFG()

# RegNet model definition - must match training model architecture
class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=1):
        super().__init__()
        
        # Load the RegNet model
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,  # Not using pretrained weights for inference
            in_chans=in_channels,
            num_classes=0      # Remove classifier head
        )
        
        # Get feature dimension automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, cfg.img_size, cfg.img_size)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# Audio processing functions - identical to training
def audio_to_melspec(audio, cfg):
    """Convert audio data to mel spectrogram"""
    # Handle NaN values
    if np.isnan(audio).any():
        audio = np.nan_to_num(audio)
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_norm = (mel_spec_db + 80) / 80  # Typical dB range
    
    return np.clip(mel_spec_norm, 0, 1)  # Clip to [0, 1]

def apply_tta(mel_spec, step):
    """Apply test-time augmentation"""
    if step == 0:
        # Original spectrogram
        return mel_spec
    elif step == 1:
        # Horizontal flip (time axis)
        return np.flip(mel_spec, axis=1)
    elif step == 2:
        # Vertical flip (frequency axis)
        return np.flip(mel_spec, axis=0)
    else:
        return mel_spec

def load_model_and_species():
    """Load trained model from checkpoint and get species list"""
    print(f"Loading model from {cfg.model_path}")
    
    # First load the sample submission to get expected column names
    sample_sub = pd.read_csv(cfg.sample_submission)
    species_columns = [col for col in sample_sub.columns if col != 'row_id']
    num_species = len(species_columns)
    print(f"Sample submission has {num_species} species columns")
    
    # Load checkpoint
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device)
    
    # Create model with same number of outputs as species
    model = BirdCLEFModel(cfg.model_name, num_species, in_channels=1)
    
    # Try to load model state
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model state from 'model_state_dict'")
        else:
            for key in ['state_dict', 'model']:
                if key in checkpoint:
                    model.load_state_dict(checkpoint[key])
                    print(f"Loaded model state from '{key}'")
                    break
    except Exception as e:
        print(f"WARNING: Error loading model weights: {e}")
    
    model = model.to(cfg.device)
    model.eval()
    
    # Create direct mapping from model outputs to species column names
    species_map = {i: species for i, species in enumerate(species_columns)}
    
    return model, species_map, species_columns

def predict_on_soundscape(model, audio_path, species_map):
    """Process a soundscape file and generate predictions for each 5-second segment"""
    soundscape_id = Path(audio_path).stem
    
    try:
        # Load audio
        audio_data, _ = librosa.load(
            audio_path, 
            sr=cfg.sample_rate,
            res_type='kaiser_fast'  # Faster resampling
        )
        
        # Calculate total segments
        segment_samples = cfg.sample_rate * cfg.duration
        total_segments = int(len(audio_data) / segment_samples)
        
        # Initialize lists for results
        row_ids = []
        all_predictions = []
        
        # Process each segment
        for segment_idx in range(total_segments):
            # Extract segment
            start_sample = segment_idx * segment_samples
            end_sample = start_sample + segment_samples
            segment_audio = audio_data[start_sample:end_sample]
            
            # Create row ID in required format
            end_time_sec = (segment_idx + 1) * cfg.duration
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)
            
            if cfg.use_tta:
                # Apply test-time augmentation
                segment_preds = []
                
                for tta_step in range(cfg.tta_steps):
                    # Process audio to mel spectrogram
                    mel_spec = audio_to_melspec(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_step)
                    
                    # Resize
                    mel_spec = cv2.resize(mel_spec, (cfg.img_size, cfg.img_size))
                    
                    # Convert to tensor
                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = model(mel_spec)
                        probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                        segment_preds.append(probs)
                
                # Average predictions from all TTA steps
                final_preds = np.mean(segment_preds, axis=0)
            else:
                # Process audio without TTA
                mel_spec = audio_to_melspec(segment_audio, cfg)
                
                # Resize
                mel_spec = cv2.resize(mel_spec, (cfg.img_size, cfg.img_size))
                
                # Convert to tensor
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(mel_spec)
                    final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
            
            # Create a prediction dictionary
            pred_dict = {}
            
            # Handle case where final_preds might be a single value
            if np.isscalar(final_preds):
                # If model only has one output class
                if len(species_map) > 0:
                    species_id = list(species_map.values())[0]
                    pred_dict[species_id] = float(final_preds)
            else:
                # Map model outputs to species IDs
                for i, prob in enumerate(final_preds):
                    if i in species_map:
                        species_id = species_map[i]
                        pred_dict[species_id] = float(prob)
            
            all_predictions.append(pred_dict)
        
        return row_ids, all_predictions
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def create_submission_file(all_row_ids, all_predictions):
    """Create submission file in the required format"""
    print("Creating submission file...")
    
    # Load sample submission
    sample_sub = pd.read_csv(cfg.sample_submission)
    
    # Initialize submission with row_ids
    submission_df = pd.DataFrame({'row_id': all_row_ids})
    
    # Add each species column from sample submission
    for col in sample_sub.columns:
        if col != 'row_id':
            # Default all values to 0.0
            submission_df[col] = 0.0
            
            # Update with actual predictions if available
            for i, pred_dict in enumerate(all_predictions):
                if col in pred_dict and i < len(submission_df):
                    submission_df.loc[i, col] = pred_dict[col]
    
    # Ensure column order matches sample submission exactly
    submission_df = submission_df[sample_sub.columns]
    
    # Save to CSV
    submission_df.to_csv('submission.csv', index=False, float_format='%.6f')
    print(f"Submission file created with {len(submission_df)} predictions")

def run_inference():
    """Main inference function"""
    start_time = time.time()
    print(f"Starting inference using device: {cfg.device}")
    
    # Load model with species mapping
    model, species_map, _ = load_model_and_species()
    
    # Find test files
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
    if cfg.debug_mode:
        print(f"Debug mode: processing only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")
    
    # Process each soundscape
    all_row_ids = []
    all_predictions = []
    
    for audio_path in tqdm(test_files, desc="Processing test soundscapes"):
        row_ids, predictions = predict_on_soundscape(model, str(audio_path), species_map)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    
    # Create submission file
    create_submission_file(all_row_ids, all_predictions)
    
    print(f"Inference completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    try:
        run_inference()
        
        # Verify submission
        try:
            sample_sub = pd.read_csv(cfg.sample_submission)
            submission = pd.read_csv('submission.csv')
            
            print(f"Submission file: {submission.shape}, Sample: {sample_sub.shape}")
            if set(submission.columns) != set(sample_sub.columns):
                print("WARNING: Column mismatch!")
            else:
                print("Submission has correct columns ✓")
        except Exception as e:
            print(f"Error verifying submission: {e}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

