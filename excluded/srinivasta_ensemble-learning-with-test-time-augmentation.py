import os, gc, warnings, logging, time, math, cv2
from pathlib import Path

import numpy as np, pandas as pd, librosa
import torch, torch.nn as nn, torch.nn.functional as F, timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


class CFG:
    test_soundscapes, submission_csv, taxonomy_csv = (
        '/kaggle/input/birdclef-2025/test_soundscapes',
        '/kaggle/input/birdclef-2025/sample_submission.csv',
        '/kaggle/input/birdclef-2025/taxonomy.csv',
    )
    model_path = '/kaggle/input/birdclef25-effnetb0-starter-weight'

    # Audio, Mel spectrogram, Model, Inference parameters
    FS, WINDOW_SIZE = 32000, 5
    N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX = 1024, 512, 128, 50, 14000
    TARGET_SHAPE = (256, 256)
    model_name, in_channels, device = 'efficientnet_b0', 1, 'cpu'
    batch_size, use_tta, tta_count, threshold = 16, False, 3, 0.5
    use_specific_folds, folds = False, [0, 1]
    debug, debug_count = False, 3

cfg = CFG()


print(f"Using device: {cfg.device}", 
      f"Loading taxonomy data...", sep='\n')  # Print on separate lines

taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)

print(f"Number of classes: {num_classes}")


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg, self.num_classes = cfg, num_classes 
        self.backbone = timm.create_model(
            cfg.model_name, pretrained=False, in_chans=cfg.in_channels,
            drop_rate=0, drop_path_rate=0  # Remove unnecessary decimals
        )

        # Determine backbone output features based on model architecture
        if 'efficientnet' in cfg.model_name:
            self.backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            self.backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            self.backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone_out, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = features['features'] if isinstance(features, dict) else features
        features = self.pooling(features).view(features.size(0), -1) if len(features.shape) == 4 else features
        return self.classifier(features)  # Directly return logits


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram."""
    audio_data = np.nan_to_num(audio_data, nan=np.nanmean(audio_data)) if np.isnan(audio_data).any() else audio_data

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=cfg.FS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX, power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

def process_audio_segment(audio_data, cfg):
    """Process audio segment to get mel spectrogram."""
    audio_data = np.pad(audio_data, (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)), mode='constant') if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE else audio_data
    mel_spec = audio2melspec(audio_data, cfg)
    return cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR).astype(np.float32) if mel_spec.shape != cfg.TARGET_SHAPE else mel_spec.astype(np.float32)


def find_model_files(cfg):
    """
    Find all .pth model files in the specified model directory
    """
    model_files = []
    
    model_dir = Path(cfg.model_path)
    
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    
    return model_files

def load_models(cfg, num_classes):
    """
    Load all found model files and prepare them for ensemble
    """
    models = []
    
    model_files = find_model_files(cfg)
    
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
    
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            if cfg.use_tta:
                all_preds = []
                
                for tta_idx in range(cfg.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)

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
                mel_spec = process_audio_segment(segment_audio, cfg)
                
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)
                
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
    """Apply test-time augmentation."""
    return {
        0: spec,
        1: np.flip(spec, axis=1),
        2: np.flip(spec, axis=0)
    }.get(tta_idx, spec)

def run_inference(cfg, models, species_ids):
    """Run inference on all test soundscapes."""
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    if cfg.debug:
        test_files = test_files[:cfg.debug_count]
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
    print(f"Found {len(test_files)} test soundscapes")
    
    all_row_ids, all_predictions = [], []
    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
        
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg):
    """Create submission dataframe."""
    print("Creating submission dataframe...")
    submission_dict = {'row_id': row_ids}
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]
    
    submission_df = pd.DataFrame(submission_dict).set_index('row_id')
    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')
    
    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        submission_df[list(missing_cols)] = 0.0  # Assigning 0 to missing columns directly

    return submission_df.reindex(columns=sample_sub.columns).reset_index()


def main():
    """Main function for BirdCLEF-2025 inference."""
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")
    print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")
    
    models = load_models(cfg, num_classes)
    if not models:
        print("No models found! Please check model paths.")
        return

    print(f"Model usage: {'Single' if len(models) == 1 else 'Ensemble'} model{'s' if len(models) > 1 else ''}")
    
    row_ids, predictions = run_inference(cfg, models, species_ids)
    submission_df = create_submission(row_ids, predictions, species_ids, cfg)
    
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    print(f"Inference completed in {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during inference: {e}")




