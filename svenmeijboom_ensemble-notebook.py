import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

#logging.basicConfig(level=logging.INFO)


#EFFICIENTNET MODEL

class CFG:
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/duration-with-original-data/pytorch/default/1'
    
    # Audio parameters
    FS = 32000  
    WINDOW_SIZE = 5  
    
    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (256, 256)
    
    model_name = 'efficientnet_b0'
    in_channels = 1
    device = 'cpu'  
    
    # Inference parameters
    batch_size = 16
    use_tta = False
    tta_count = 3   
    threshold = 0.5
    
    use_specific_folds = False  # If False, use all found models
    folds = [0, 1]  # Used only if use_specific_folds is True
    
    debug = False
    debug_count = 3

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg_eff, num_classes):
        super().__init__()
        self.cfg_eff = cfg_eff
        
        self.backbone = timm.create_model(
            cfg_eff.model_name,
            pretrained=False,  
            in_chans=cfg_eff.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
        )
        
        if 'efficientnet' in cfg_eff.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg_eff.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits

def audio2melspec(audio_data, cfg_eff):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg_eff.FS,
        n_fft=cfg_eff.N_FFT,
        hop_length=cfg_eff.HOP_LENGTH,
        n_mels=cfg_eff.N_MELS,
        fmin=cfg_eff.FMIN,
        fmax=cfg_eff.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_segment(audio_data, cfg_eff):
    """Process audio segment to get mel spectrogram"""
    if len(audio_data) < cfg_eff.FS * cfg_eff.WINDOW_SIZE:
        audio_data = np.pad(audio_data, 
                          (0, cfg_eff.FS * cfg_eff.WINDOW_SIZE - len(audio_data)), 
                          mode='constant')
    
    mel_spec = audio2melspec(audio_data, cfg_eff)
    
    # Resize if needed
    if mel_spec.shape != cfg_eff.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg_eff.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
    return mel_spec.astype(np.float32)

def find_model_files(cfg_eff):
    """
    Find all .pth model files in the specified model directory
    """
    model_files = []
    
    model_dir = Path(cfg_eff.model_path)
    
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    
    return model_files

def load_models(cfg_eff, num_classes):
    """
    Load all found model files and prepare them for ensemble
    """
    models = []
    
    model_files = find_model_files(cfg_eff)
    
    if not model_files:
        print(f"Warning: No model files found under {cfg_eff.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    if cfg_eff.use_specific_folds:
        filtered_files = []
        for fold in cfg_eff.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg_eff.folds}).")
    
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg_eff.device), weights_only=False)
            
            model = BirdCLEFModel(cfg_eff, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg_eff.device)
            model.eval()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models

def predict_on_spectrogram(audio_path, models, cfg_eff, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=cfg_eff.FS)
        
        total_segments = int(len(audio_data) / (cfg_eff.FS * cfg_eff.WINDOW_SIZE))
        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg_eff.FS * cfg_eff.WINDOW_SIZE
            end_sample = start_sample + cfg_eff.FS * cfg_eff.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            
            end_time_sec = (segment_idx + 1) * cfg_eff.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            if cfg_eff.use_tta:
                all_preds = []
                
                for tta_idx in range(cfg_eff.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg_eff)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg_eff.device)

                    if len(models) == 1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                final_preds = np.mean(all_preds, axis=0)
            else:
                mel_spec = process_audio_segment(segment_audio, cfg_eff)
                
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg_eff.device)
                
                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else:
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_preds = np.mean(segment_preds, axis=0)
                    
            predictions.append(final_preds)
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions

def apply_tta(spec, tta_idx):
    """Apply test-time augmentation"""
    if tta_idx == 0:
        # Original spectrogram
        return spec
    elif tta_idx == 1:
        # Time shift (horizontal flip)
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        # Frequency shift (vertical flip)
        return np.flip(spec, axis=0)
    else:
        return spec

def run_inference(cfg_eff, models, species_ids):
    """Run inference on all test soundscapes"""
    test_files = list(Path(cfg_eff.test_soundscapes).glob('*.ogg'))
    
    if cfg_eff.debug:
        print(f"Debug mode enabled, using only {cfg_eff.debug_count} files")
        test_files = test_files[:cfg_eff.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg_eff, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg_eff):
    """Create submission dataframe"""
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}
    
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg_eff.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()
    
    return submission_df

def run_efficientnet():
    print(f"Using device: {cfg_eff.device}")
    print(f"Loading taxonomy data...")
    taxonomy_df = pd.read_csv(cfg_eff.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    num_classes = len(species_ids)
    print(f"Number of classes: {num_classes}")

    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")
    print(f"TTA enabled: {cfg_eff.use_tta} (variations: {cfg_eff.tta_count if cfg_eff.use_tta else 0})")

    models = load_models(cfg_eff, num_classes)
    
    if not models:
        print("No models found! Please check model paths.")
        return
    
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg_eff, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg_eff)

    submission_path = 'submission_effnet.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")

cfg_eff = CFG()


run_efficientnet()


#REGNET MODEL
"""
# BirdCLEF 2025 - Fixed RegNet Inference Code

This notebook loads a trained RegNet model and creates predictions for the BirdCLEF 2025 test data.
It fixes the previous issues with the submission format to ensure correct evaluation.
"""
# Configuration class
class CFG_regnet:
    # Paths
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    sample_submission = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/regnetmodel/pytorch/default/1/best_model (1).pth'

    
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
            dummy_input = torch.zeros(1, in_channels, cfg_regnet.img_size, cfg_regnet.img_size)
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
def audio_to_melspec(audio, cfg_regnet):
    """Convert audio data to mel spectrogram"""
    # Handle NaN values
    if np.isnan(audio).any():
        audio = np.nan_to_num(audio)
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg_regnet.sample_rate,
        n_fft=cfg_regnet.n_fft,
        hop_length=cfg_regnet.hop_length,
        n_mels=cfg_regnet.n_mels,
        fmin=cfg_regnet.fmin,
        fmax=cfg_regnet.fmax,
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
    print(f"Loading model from {cfg_regnet.model_path}")
    
    # First load the sample submission to get expected column names
    sample_sub = pd.read_csv(cfg_regnet.sample_submission)
    species_columns = [col for col in sample_sub.columns if col != 'row_id']
    num_species = len(species_columns)
    print(f"Sample submission has {num_species} species columns")
    
    # Load checkpoint
    checkpoint = torch.load(cfg_regnet.model_path, map_location=cfg_regnet.device)
    
    # Create model with same number of outputs as species
    model = BirdCLEFModel(cfg_regnet.model_name, num_species, in_channels=1)
    
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
    
    model = model.to(cfg_regnet.device)
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
            sr=cfg_regnet.sample_rate,
            res_type='kaiser_fast'  # Faster resampling
        )
        
        # Calculate total segments
        segment_samples = cfg_regnet.sample_rate * cfg_regnet.duration
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
            end_time_sec = (segment_idx + 1) * cfg_regnet.duration
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)
            
            if cfg_regnet.use_tta:
                # Apply test-time augmentation
                segment_preds = []
                
                for tta_step in range(cfg_regnet.tta_steps):
                    # Process audio to mel spectrogram
                    mel_spec = audio_to_melspec(segment_audio, cfg_regnet)
                    mel_spec = apply_tta(mel_spec, tta_step)
                    
                    # Resize
                    mel_spec = cv2.resize(mel_spec, (cfg_regnet.img_size, cfg_regnet.img_size))
                    
                    # Convert to tensor
                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg_regnet.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = model(mel_spec)
                        probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                        segment_preds.append(probs)
                
                # Average predictions from all TTA steps
                final_preds = np.mean(segment_preds, axis=0)
            else:
                # Process audio without TTA
                mel_spec = audio_to_melspec(segment_audio, cfg_regnet)
                
                # Resize
                mel_spec = cv2.resize(mel_spec, (cfg_regnet.img_size, cfg_regnet.img_size))
                
                # Convert to tensor
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg_regnet.device)
                
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
    sample_sub = pd.read_csv(cfg_regnet.sample_submission)
    
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
    submission_df.to_csv('submission_regnet.csv', index=False, float_format='%.6f')
    print(f"Submission file created with {len(submission_df)} predictions")

def run_inference():
    """Main inference function"""
    start_time = time.time()
    print(f"Starting inference using device: {cfg_regnet.device}")
    
    # Load model with species mapping
    model, species_map, _ = load_model_and_species()
    
    # Find test files
    test_files = list(Path(cfg_regnet.test_soundscapes).glob('*.ogg'))
    
    if cfg_regnet.debug_mode:
        print(f"Debug mode: processing only {cfg_regnet.debug_count} files")
        test_files = test_files[:cfg_regnet.debug_count]
    
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

def run_regnet():
    try:
        run_inference()
        
        # Verify submission
        try:
            sample_sub = pd.read_csv(cfg_regnet.sample_submission)
            submission = pd.read_csv('submission_regnet.csv')
            
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
cfg_regnet = CFG_regnet()


run_regnet()


submission_effnet = pd.read_csv('/kaggle/working/submission_effnet.csv')
submission_regnet = pd.read_csv('/kaggle/working/submission_regnet.csv')

weight_effnet = 0.6826457183206767
weight_regnet = 1 - weight_effnet

if not submission_effnet['row_id'].equals(submission_regnet['row_id']):
    print("Warning: Row IDs of submissions do not match. Blending may be incorrect.")

submission_path_blended = 'submission.csv'
result = submission_effnet.set_index('row_id').multiply(weight_effnet).add(submission_regnet.set_index('row_id').multiply(weight_regnet), fill_value=0).reset_index()
result.to_csv(submission_path_blended, index=False)

