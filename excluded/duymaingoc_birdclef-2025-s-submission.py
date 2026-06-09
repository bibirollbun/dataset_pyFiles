import os
import gc
import time
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class CFG:
    seed = 42
    debug = False # Set to True for testing with fewer files
    num_workers = 2
    
    # Paths
    # !!! IMPORTANT: Update this path to where your trained model weights are located !!!
    model_weights_dir = '/kaggle/input/efficientnet-b0-pytorch-train-birdclef-25' # Example path, adjust as needed
    
    test_datadir = '/kaggle/input/birdclef-2025/test_soundscapes/'
    train_csv = '/kaggle/input/birdclef-2025/train.csv' # Needed for label mapping if not in taxonomy
    submission_csv_path = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    output_dir = '/kaggle/working/'
    
    # Model parameters (should match training)
    model_name = 'efficientnet_b0'
    pretrained = False # We load weights, not pre-trained from timm
    in_channels = 3
    num_classes = 206 # Will be updated based on taxonomy.csv
    
    # Audio parameters (should match training)
    FS = 32000
    TARGET_DURATION_SEC = 5.0
    TARGET_SAMPLES = int(TARGET_DURATION_SEC * FS)
    TARGET_SHAPE = (256, 256)
    
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 20
    FMAX = 16000
    
    # Inference parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16 # Adjust based on GPU memory
    num_folds = 5

cfg = CFG()

# Read taxonomy to get all species labels
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
cfg.species_ids = sorted(taxonomy_df['primary_label'].unique().tolist())
cfg.num_classes = len(cfg.species_ids)
print(f"Number of classes: {cfg.num_classes}")
print(f"Device: {cfg.device}")


def set_seed(seed=42):
    # ""Set seed for reproducibility""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # No need for deterministic settings in inference, prioritize speed
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


def audio2melspec(audio_data, cfg):
    # ""Convert audio data to mel spectrogram (identical to training)""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    # Resize if necessary (should match target shape)
    if mel_spec_norm.shape != cfg.TARGET_SHAPE:
       mel_spec_norm = cv2.resize(mel_spec_norm, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
    return mel_spec_norm.astype(np.float32)

def process_soundscape(audio_path, cfg):
    # ""Loads a soundscape, splits it into 5s chunks, and converts each to a mel spectrogram.""
    specs = []
    try:
        # Load the full soundscape
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS, mono=True)
        
        total_duration = librosa.get_duration(y=audio_data, sr=cfg.FS)
        num_segments = int(math.ceil(total_duration / cfg.TARGET_DURATION_SEC))
        
        for i in range(num_segments):
            start_sample = i * cfg.TARGET_SAMPLES
            end_sample = start_sample + cfg.TARGET_SAMPLES
            
            # Extract segment
            segment = audio_data[start_sample:end_sample]
            
            # Pad if segment is shorter than target duration (last segment)
            if len(segment) < cfg.TARGET_SAMPLES:
                segment = np.pad(segment, (0, cfg.TARGET_SAMPLES - len(segment)), mode='constant')
            
            # Convert to mel spectrogram
            mel_spec = audio2melspec(segment, cfg)
            specs.append(mel_spec)
            
        if not specs: # Handle potential errors or empty audio
             print(f"Warning: No segments processed for {audio_path}")
             # Return a list of zero spectrograms matching the expected output structure
             # Assuming 1 minute = 12 segments
             num_expected_segments = 12 
             return [np.zeros(cfg.TARGET_SHAPE, dtype=np.float32) for _ in range(num_expected_segments)]
            
        return specs
        
    except Exception as e:
        print(f"Error processing soundscape {audio_path}: {e}")
        # Return list of zero spectrograms if error occurs
        num_expected_segments = 12 # Assuming 1 minute test files
        return [np.zeros(cfg.TARGET_SHAPE, dtype=np.float32) for _ in range(num_expected_segments)]


class BirdCLEFInferenceDataset(Dataset):
    def __init__(self, soundscape_paths, cfg):
        self.soundscape_paths = soundscape_paths
        self.cfg = cfg

    def __len__(self):
        return len(self.soundscape_paths)

    def __getitem__(self, idx):
        audio_path = self.soundscape_paths[idx]
        soundscape_id = Path(audio_path).stem
        
        # Process the entire soundscape into a list of spectrograms
        list_of_specs = process_soundscape(audio_path, self.cfg)
        
        # Stack spectrograms into a single tensor for batching
        # Shape: (num_segments, channels, height, width)
        specs_tensor = torch.tensor(np.array(list_of_specs), dtype=torch.float32).unsqueeze(1)
        
        return {
            'soundscape_id': soundscape_id,
            'specs': specs_tensor # Tensor of shape [12, 1, H, W]
        }


class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 0.6,
            focal_weight: float = 1.4,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focal_loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss

def get_criterion(cfg):
    return FocalLossBCE()


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
            
        self.feat_dim = backbone_out
        
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha
            
    def forward(self, x, targets=None):
    
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits
    
    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


loaded_models = [] 
print(f"Loading {cfg.num_folds} models from {cfg.model_weights_dir}")
for fold in range(cfg.num_folds):
    model_path = os.path.join(cfg.model_weights_dir, f'model_fold{fold}.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model weight file not found at {model_path}")
        print("Please ensure the 'model_weights_dir' in CFG points to the correct dataset/directory.")
        raise FileNotFoundError(f"Model weight not found: {model_path}")

    # Create the model instance using the correct 'models' module
    model = BirdCLEFModel(cfg)
    try:
        # Load the state dict
        checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Optional: Adjust keys if needed
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.to(cfg.device)
        model.eval()
        loaded_models.append(model) # <<<--- Appending to the renamed list
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise e

# Check the renamed list
if len(loaded_models) != cfg.num_folds:
    print(f"Warning: Expected {cfg.num_folds} models, but only loaded {len(loaded_models)}. Check paths and files.")
    if len(loaded_models) == 0:
         raise RuntimeError("No models were loaded successfully. Cannot proceed with inference.")


def run_inference(models, cfg):
    all_predictions = []
    
    # Find test soundscape files
    test_files = []
    if os.path.exists(cfg.test_datadir):
        test_files = [os.path.join(cfg.test_datadir, f) for f in os.listdir(cfg.test_datadir) if f.endswith('.ogg')]
        print(f"Found {len(test_files)} test soundscapes.")
    else:
        print(f"Test directory {cfg.test_datadir} not found.")

    if cfg.debug:
        test_files = test_files[:3] # Limit files for debugging
        print(f"Debug mode: Processing only {len(test_files)} files.")
        
    # === Modified Condition ===
    # If no test files were found (directory non-existent or empty), generate sample submission
    if not test_files:
         print("Test directory empty or non-existent. Generating sample submission format.")
         try:
             sample_df = pd.read_csv(cfg.submission_csv_path)
         except FileNotFoundError:
              print(f"Error: Sample submission file not found at {cfg.submission_csv_path}")
              # Create a fallback structure if sample submission is missing
              return pd.DataFrame(columns=['row_id'] + cfg.species_ids)
         
         # Fill with a default value (e.g., 1/num_classes) or zeros
         default_prob = 1 / cfg.num_classes 
         for species in cfg.species_ids:
             if species in sample_df.columns:
                 sample_df[species] = default_prob
             else:
                 # Add missing species column if it's in taxonomy but not sample submission
                 print(f"Info: Species {species} from taxonomy added to submission columns.")
                 sample_df[species] = default_prob
                 
         # Ensure all required columns exist and are in order
         final_cols = ['row_id'] + cfg.species_ids
         # Select existing columns first, then add any missing ones
         sample_df = sample_df[[col for col in final_cols if col in sample_df.columns]]
         for col in final_cols:
             if col not in sample_df.columns:
                 sample_df[col] = default_prob # Should not happen often due to logic above
                 
         return sample_df[final_cols] # Return with columns in correct order

    # --- Proceed with inference if test files exist ---
    
    # Create dataset and dataloader
    inference_dataset = BirdCLEFInferenceDataset(test_files, cfg)
    # Batch size for dataloader is 1, as we process one full soundscape at a time
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Inference"):
            soundscape_id = batch['soundscape_id'][0] # Dataloader returns list for batch_size=1
            specs_batch = batch['specs'][0].to(cfg.device) # Shape [12, 1, H, W]
            
            num_segments = specs_batch.shape[0]
            fold_predictions = []
            
            # Get predictions from each fold's model
            for model in models:
                # Process segments in mini-batches if necessary (GPU memory)
                segment_preds = []
                for i in range(0, num_segments, cfg.batch_size):
                    mini_batch = specs_batch[i:i+cfg.batch_size]
                    logits = model(mini_batch)
                    probs = torch.sigmoid(logits) # Convert logits to probabilities
                    segment_preds.append(probs.cpu().numpy())
                
                # Concatenate predictions for the soundscape from this fold
                fold_soundscape_preds = np.concatenate(segment_preds, axis=0) # Shape [12, num_classes]
                fold_predictions.append(fold_soundscape_preds)
            
            # Ensemble predictions: Average probabilities across folds
            # Shape: (num_folds, num_segments, num_classes) -> (num_segments, num_classes)
            ensembled_preds = np.mean(np.stack(fold_predictions, axis=0), axis=0)
            
            # Store predictions for this soundscape
            for i in range(num_segments):
                end_time = (i + 1) * int(cfg.TARGET_DURATION_SEC)
                row_id = f"{soundscape_id}_{end_time}"
                
                prediction_dict = {'row_id': row_id}
                # Fill probabilities for each species
                for j, species_id in enumerate(cfg.species_ids):
                    prediction_dict[species_id] = ensembled_preds[i, j]
                    
                all_predictions.append(prediction_dict)
            
            # Clean up memory
            del specs_batch, fold_predictions, ensembled_preds
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
    # Create submission DataFrame from collected predictions
    if not all_predictions: # Should not happen if test_files was not empty
        print('Warning: No predictions were generated even though test files were found.')
        # Return empty df with correct columns as fallback
        return pd.DataFrame(columns=['row_id'] + cfg.species_ids) 
        
    submission_df = pd.DataFrame(all_predictions)
    
    # Ensure all required columns are present and in the correct order
    final_cols = ['row_id'] + cfg.species_ids
    # Add any missing species columns (e.g., if a species was in taxonomy but somehow missed in prediction dict)
    for col in final_cols:
        if col not in submission_df.columns:
            print(f"Warning: Column {col} missing in submission DataFrame, adding with zeros.")
            submission_df[col] = 0.0 # Or default_prob
            
    # Select and order columns - this should now work
    submission_df = submission_df[final_cols] 
    
    return submission_df


start_time = time.time()

# Ensure the renamed list 'loaded_models' is populated before calling run_inference
# Check if 'loaded_models' exists and is not empty
if 'loaded_models' not in locals() or not loaded_models:
    print("Error: Models were not loaded into 'loaded_models' list. Cannot run inference.")
    # As a fallback for Kaggle commit, create an empty submission
    # Consider raising an error if debugging locally: raise RuntimeError("Models not loaded.")
    submission_df = pd.DataFrame(columns=['row_id'] + cfg.species_ids)
else:
    # Pass the correct list to the inference function
    submission_df = run_inference(loaded_models, cfg) # <<<--- Passing the renamed list

# Save the submission file
output_path = os.path.join(cfg.output_dir, 'submission.csv')
submission_df.to_csv(output_path, index=False)

end_time = time.time()
print(f"Submission file created at: {output_path}")
print(f"Total inference time: {end_time - start_time:.2f} seconds")

# Display the first few rows of the submission
print("Submission DataFrame head:")
print(submission_df.head())




