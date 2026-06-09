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
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm
from glob import glob
import random
import itertools
from typing import Union
import concurrent.futures

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


# Configuration
class CFG:
    seed = 42
    num_workers = 2
    
    # Path settings
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    
    # [Key] Modify this to your uploaded model path
    # Assuming you uploaded best_model.pth to a dataset named birdclef-my-model
    model_files = [
        '/kaggle/input/sed-baseline/pytorch/default/1/best_model.pth' 
    ]
 
    # Model parameters (Must match training)
    model_name = 'efficientnet_b0'  
    pretrained = False
    in_channels = 1
    
    # Audio parameters (Must match training)
    SR = 32000
    target_duration = 5 # 5-second slice
    
    # MelSpectrogram parameters
    n_fft = 1024
    hop_length = 512
    n_mels = 128
    f_min = 50
    f_max = 14000
    target_shape = (256, 256)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = CFG()

print(f"Using device: {cfg.device}")
print(f"Loading taxonomy data...")
# If taxonomy is missing locally (e.g., during testing with only sample_submission), provide compatibility
if os.path.exists(cfg.taxonomy_csv):
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
else:
    # Fallback: read from sample_submission
    ss = pd.read_csv(cfg.submission_csv)
    species_ids = [c for c in ss.columns if c != 'row_id']

num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")


# Utilities
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


# Model Definition (Matching your training)
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Backbone network
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False, # Cannot access internet during inference, set to False
            in_chans=cfg.in_channels
        )

        # Replace classification head
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            backbone_out = self.backbone.num_features
            self.backbone.reset_classifier(0, '')
            
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # num_classes needs to be obtained from global variables or cfg
        global num_classes
        self.classifier = nn.Linear(backbone_out, num_classes)

    def forward(self, x):
        # x: (batch, 1, freq, time)
        x = self.backbone(x)
        
        # Handle potential dict output from timm
        if isinstance(x, dict):
            x = x['features']
            
        # Global Average Pooling
        if len(x.shape) == 4:
            x = self.pooling(x)
            x = x.view(x.size(0), -1)
            
        logits = self.classifier(x)
        return logits


# Feature Extraction (Librosa)
class LogMelFeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, audio_data):
        # audio_data: numpy array (samples,)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.cfg.SR, 
            n_fft=self.cfg.n_fft, 
            hop_length=self.cfg.hop_length, 
            n_mels=self.cfg.n_mels, 
            fmin=self.cfg.f_min, 
            fmax=self.cfg.f_max, 
            power=2.0
        )
        # Log Scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Min-Max Normalization to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Resize to fixed shape (256, 256)
        if mel_spec_norm.shape != self.cfg.target_shape:
            mel_spec_norm = cv2.resize(mel_spec_norm, self.cfg.target_shape, interpolation=cv2.INTER_LINEAR)
            
        return mel_spec_norm.astype(np.float32)


# Audio Loading & Slicing
def load_and_slice_audio(path, cfg):
    """
    Load audio and slice into 5-second segments
    """
    try:
        audio, _ = librosa.load(path, sr=cfg.SR)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return [], []

    # Calculate required number of chunks
    chunk_len = int(cfg.target_duration * cfg.SR)
    # math.ceil for rounding up
    num_chunks = math.ceil(len(audio) / chunk_len)
    
    # Pad length
    target_len = num_chunks * chunk_len
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
        
    segments = []
    end_seconds = []
    
    for i in range(num_chunks):
        seg = audio[i*chunk_len : (i+1)*chunk_len]
        segments.append(seg)
        end_seconds.append((i+1) * cfg.target_duration)
        
    return segments, end_seconds


# Model Loading
def load_models(cfg):
    models = []
    model_files = cfg.model_files
    
    if not model_files:
        print(f"Warning: No model files found!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    for model_path in model_files:
        if not os.path.exists(model_path):
            print(f"Path does not exist: {model_path}")
            continue
            
        try:
            print(f"Loading model: {model_path}")
            # [Key] weights_only=False to avoid CFG class errors
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device), weights_only=False)
            
            # Initialize model
            model = BirdCLEFModel(cfg)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model = model.to(cfg.device)
            model.eval()
            
            # Half-precision acceleration (Optional, if GPU supports)
            # model.half() 
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    return models


# Inference Function (Per File)
def predict_on_file(audio_path, models, cfg, feature_extractor):
    """
    Process a single audio file
    """
    audio_path = str(audio_path)
    row_ids = []
    predictions = []
    soundscape_id = Path(audio_path).stem.split('.')[0] # Get filename part only

    # 1. Load and slice audio
    segments, seconds = load_and_slice_audio(audio_path, cfg)
    
    if len(segments) == 0:
        return [], []

    # 2. Pre-processing (Feature Extraction)
    # Batch processing for speedup (but serial might be safer on CPU)
    batch_imgs = []
    for seg in segments:
        spec = feature_extractor(seg) # (256, 256)
        batch_imgs.append(spec)
    
    # Convert to Tensor: (Batch, 1, 256, 256)
    batch_tensor = torch.tensor(np.array(batch_imgs), dtype=torch.float32).unsqueeze(1).to(cfg.device)
    
    # 3. Inference
    file_preds = []
    
    # models is a list here (even if only one model) to facilitate future Ensemble
    for model in models:
        with torch.no_grad():
            # (Batch, Num_Classes)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            file_preds.append(probs)
    
    # If multiple models, take the average
    final_preds = np.mean(file_preds, axis=0) # (Batch, Num_Classes)
    
    # 4. Generate Row IDs
    for i, sec in enumerate(seconds):
        row_id = f"{soundscape_id}_{int(sec)}"
        row_ids.append(row_id)
        predictions.append(final_preds[i])
        
    return row_ids, predictions

# Main Inference Loop
def run_inference(cfg, models):
    # 1. Find test files
    test_files = sorted(list(Path(cfg.test_soundscapes).glob('*.ogg')))
    
    # If no files in Commit stage, use dummy files or return directly
    if len(test_files) == 0:
        print("No test files found (Commit Stage). Using dummy files if available or skipping.")
        # To prevent empty errors, usually generate dummy submission
        return [], []
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []
    
    feature_extractor = LogMelFeatureExtractor(cfg)

    # 2. Parallel processing
    # Adjust max_workers based on CPU cores; audio processing is IO intensive + CPU intensive
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    predict_on_file,
                    test_files,
                    itertools.repeat(models),
                    itertools.repeat(cfg),
                    itertools.repeat(feature_extractor)
                ),
                total=len(test_files),
                desc="Inferencing"
            )
        )

    # 3. Aggregate results
    for rids, preds in results:
        all_row_ids.extend(rids)
        all_predictions.extend(preds)
    
    return all_row_ids, all_predictions


# Submission & Smoothing
def create_submission(row_ids, predictions, species_ids, cfg):
    print("Creating submission dataframe...")
    
    if len(row_ids) == 0:
        # Dummy submission creation
        print("Generating dummy submission structure.")
        sample_sub = pd.read_csv(cfg.submission_csv)
        sample_sub.to_csv("submission.csv", index=False)
        return

    # Build DataFrame
    # predictions is a list of arrays
    preds_np = np.array(predictions)
    
    submission_df = pd.DataFrame(preds_np, columns=species_ids)
    submission_df.insert(0, 'row_id', row_ids)

    # Align column names (just in case)
    if os.path.exists(cfg.submission_csv):
        sample_sub = pd.read_csv(cfg.submission_csv)
        # Fill missing columns
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        for col in missing_cols:
            submission_df[col] = 0.0
        # Sort
        submission_df = submission_df[sample_sub.columns]
    
    return submission_df

def smooth_submission(submission_path):
    """
    Post-processing smoothing: If there are bird calls in the previous and next second, 
    the middle one likely has them too.
    (0.8 * curr) + (0.2 * neighbor)
    """
    print("Smoothing submission predictions...")
    if not os.path.exists(submission_path): return

    sub = pd.read_csv(submission_path)
    if len(sub) == 0: return

    cols = sub.columns[1:]
    # Extract soundscape filename as group key
    groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        idx = np.where(groups == group)[0]
        # If file has only one slice, cannot smooth
        if len(idx) <= 1: continue

        sub_group = sub.iloc[idx].copy()
        predictions = sub_group[cols].values.astype(float)
        new_predictions = predictions.copy()
        
        # Special handling for start and end
        new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
        new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
        
        # Middle part processing
        # curr * 0.6 + prev * 0.2 + next * 0.2
        for i in range(1, predictions.shape[0]-1):
            new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
        
        # Write back
        sub.iloc[idx, 1:] = new_predictions
    
    sub.to_csv(submission_path, index=False)
    print(f"Smoothed submission saved to {submission_path}")


# Main Execution
def main():
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")

    # 1. Load models
    models = load_models(cfg)
    
    if not models:
        print("No models found! Please check model paths.")
        # Generate an empty one to prevent errors
        create_submission([], [], species_ids, cfg)
        return
    
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    # 2. Run inference
    row_ids, predictions = run_inference(cfg, models)

    # 3. Generate CSV
    submission_df = create_submission(row_ids, predictions, species_ids, cfg)
    
    if submission_df is not None:
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        # 4. Smooth results
        smooth_submission(submission_path)
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")
    
    # Print first few lines to check
    if os.path.exists("submission.csv"):
        print(pd.read_csv("submission.csv").head())

if __name__ == "__main__":
    main()


pd.read_csv("submission.csv")

