import os
import gc
import warnings
import logging
import time
import math
import cv2
import random
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


# ## Configuration

# %%
class CFG:
    # --- Paths ---
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/my-birdclef25-trained-models-v1' 


    # --- Model --- 
    model_name = 'efficientnet_b0'
    in_channels = 3

    # --- Audio / Mel / Resize --- 
    FS = 32000
    WINDOW_SIZE = 5
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048
    N_MELS = 128
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)

    # --- Inference ---
    device = 'cpu'
    batch_size = 12 
    use_tta = False
    tta_count = 3

    # --- Model Selection ---
    use_specific_folds = False

    # --- Debug ---
    debug = True  
    debug_limit = 1 

cfg = CFG()
if cfg.debug:
    print("!!!!!!!!!!!!!! DEBUG MODE IS ON !!!!!!!!!!!!!!")
    cfg.batch_size = 1


# ## Setup

# %%
print(f"--- Inference Configuration ---")
print(f"Device: {cfg.device}")
print(f"Model Path: {cfg.model_path}")
print(f"Using TTA: {cfg.use_tta} ({cfg.tta_count if cfg.use_tta else 0} variations)")
print(f"Using specific folds: {cfg.use_specific_folds} ({cfg.folds if cfg.use_specific_folds and hasattr(cfg, 'folds') else 'All found'})")
print(f"Mel Params: FS={cfg.FS}, N_FFT={cfg.N_FFT}, HOP={cfg.HOP_LENGTH}, N_MELS={cfg.N_MELS}")
print(f"Target Shape: {cfg.TARGET_SHAPE}")
print(f"Batch Size (for inference): {cfg.batch_size}")
print(f"-----------------------------")

print(f"Loading taxonomy data...")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
cfg.species_ids = taxonomy_df['primary_label'].tolist()
cfg.num_classes = len(cfg.species_ids)
print(f"Number of classes: {cfg.num_classes}")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ## Model Definition 

# %%
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.model_name, pretrained=False, in_chans=cfg.in_channels,
            drop_rate=0.0, drop_path_rate=0.0
        )
        if hasattr(self.backbone, 'get_classifier'):
             backbone_out = self.backbone.get_classifier().in_features
        elif hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'in_features'):
             backbone_out = self.backbone.head.in_features
        elif hasattr(self.backbone, 'fc') and hasattr(self.backbone.fc, 'in_features'):
            backbone_out = self.backbone.fc.in_features
        elif hasattr(self.backbone, 'classifier') and hasattr(self.backbone.classifier, 'in_features'):
            backbone_out = self.backbone.classifier.in_features
        else:
            try: backbone_out = self.backbone.num_features
            except AttributeError: raise ValueError(f"Cannot determine out feats for {cfg.model_name}")
        if hasattr(self.backbone, 'reset_classifier'): self.backbone.reset_classifier(0, '')
        elif hasattr(self.backbone, 'head'): self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'): self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'): self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3) # p không quan trọng khi eval
        self.classifier = nn.Linear(backbone_out, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
             features = features.get('features', features.get('head_output', next(iter(features.values()))))
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        # Dropout không áp dụng khi model.eval()
        logits = self.classifier(features)
        return logits


# ## Audio Processing and Inference Functions

# %%
def audio2melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal if not np.isnan(mean_signal) else 0.0)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=cfg.FS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX,
        power=2.0, center=True, pad_mode="reflect"
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.max() == mel_spec_db.min():
        target_time_steps = int(np.floor((cfg.WINDOW_SIZE * cfg.FS) / cfg.HOP_LENGTH) + 1)
        return np.zeros((cfg.N_MELS, target_time_steps), dtype=np.float32)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

def process_audio_segment(audio_data, cfg):
    target_samples = int(cfg.WINDOW_SIZE * cfg.FS)
    if len(audio_data) < target_samples:
        audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
    elif len(audio_data) > target_samples:
        audio_data = audio_data[:target_samples]
    mel_spec_norm = audio2melspec(audio_data, cfg)
    if mel_spec_norm.shape != cfg.TARGET_SHAPE:
         if mel_spec_norm.shape[1] == 0 or mel_spec_norm.shape[0] == 0 :
              final_spec = np.zeros(cfg.TARGET_SHAPE, dtype=np.float32)
              # ### DEBUG ###
              # print(f"    DEBUG (process_audio_segment): Zero-width/height spec encountered, returning zeros. Original shape: {mel_spec_norm.shape}")
         else:
              final_spec = cv2.resize(mel_spec_norm, (cfg.TARGET_SHAPE[1], cfg.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
    else:
        final_spec = mel_spec_norm
    return final_spec.astype(np.float32)

def apply_tta(spec, tta_idx):
    if tta_idx == 0: return spec
    elif tta_idx == 1: return np.flip(spec, axis=1).copy()
    elif tta_idx == 2: return np.flip(spec, axis=0).copy()
    else: return spec

# %%
def find_model_files(cfg):
    model_dir = Path(cfg.model_path)
    all_files = list(model_dir.glob('**/*.pth'))
    print(f"DEBUG (find_model_files): Found {len(all_files)} .pth files in {cfg.model_path}: {all_files}")

    if not all_files: return []
    if cfg.use_specific_folds and hasattr(cfg, 'folds'):
        model_files = [f for f in all_files if any(f"fold{fold}" in f.name for fold in cfg.folds)]
        print(f"DEBUG (find_model_files): Using {len(model_files)} models for folds: {cfg.folds}: {model_files}")
    else:
        model_files = all_files
        print(f"DEBUG (find_model_files): Using all {len(model_files)} found models: {model_files}")
    return [str(f) for f in model_files]

def load_models(cfg):
    models = []
    model_files = find_model_files(cfg)
    if not model_files:
        print(f"ERROR: No model files found or selected in {cfg.model_path}. Exiting.")
        return None
    for model_path_str in model_files:
        try:
            print(f"Loading model: {model_path_str}...")
            checkpoint = torch.load(model_path_str, map_location=torch.device(cfg.device))
            num_classes_model = cfg.num_classes
            if 'cfg' in checkpoint and 'num_classes' in checkpoint['cfg']:
                 loaded_cfg_num_classes = checkpoint['cfg']['num_classes']
                 if loaded_cfg_num_classes != cfg.num_classes:
                      print(f"  WARNING: num_classes in checkpoint ({loaded_cfg_num_classes}) differs from current taxonomy ({cfg.num_classes}). Using current taxonomy value.")
                 # Kiểm tra các tham số quan trọng khác từ checkpoint nếu có
                 if 'in_channels' in checkpoint['cfg'] and checkpoint['cfg']['in_channels'] != cfg.in_channels:
                     print(f"  ERROR: Mismatch in_channels! Checkpoint: {checkpoint['cfg']['in_channels']}, Current CFG: {cfg.in_channels}")
                     continue # Bỏ qua model này
            model = BirdCLEFModel(cfg, num_classes_model)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 model.load_state_dict(checkpoint)
                 print("  Warning: Loaded state_dict directly.")
            model = model.to(cfg.device)
            model.eval()
            models.append(model)
            print(f"  Model loaded successfully from {model_path_str}.")
        except Exception as e:
            print(f"  ERROR loading model {model_path_str}: {e}")
    if not models:
         print("ERROR: Failed to load ANY models.")
         return None
    return models

# %%
def predict_soundscape(audio_path, models, cfg):
    soundscape_id = Path(audio_path).stem
    # ### DEBUG ###
    print(f"DEBUG (predict_soundscape): Processing file: {soundscape_id}")

    try:
        audio_data, sr = librosa.load(audio_path, sr=cfg.FS)
        if sr != cfg.FS:
            print(f"  WARNING: Audio sr {sr} differs from cfg.FS {cfg.FS}. Resampling not done here, ensure consistency.")

        total_duration = len(audio_data) / cfg.FS
        num_segments = math.ceil(total_duration / cfg.WINDOW_SIZE)
        # ### DEBUG ###
        print(f"  DEBUG: Audio duration: {total_duration:.2f}s, Expected segments: {num_segments}")

        if num_segments == 0:
            print(f"  WARNING: No segments for {soundscape_id}, duration {total_duration}s too short.")
            return [], []

        segment_specs_list = []
        row_ids_list = []

        for i in range(num_segments):
            start_time = i * cfg.WINDOW_SIZE
            end_time = start_time + cfg.WINDOW_SIZE
            start_sample = int(start_time * cfg.FS)
            end_sample = int(end_time * cfg.FS)
            segment_audio = audio_data[start_sample:end_sample]

            spec = process_audio_segment(segment_audio, cfg)
            segment_specs_list.append(spec)
            row_ids_list.append(f"{soundscape_id}_{(i + 1) * cfg.WINDOW_SIZE}")
            if i < 2 and cfg.debug : # Chỉ in 2 segment đầu khi debug
                print(f"    DEBUG (segment {i}): Spec shape: {spec.shape}, min: {spec.min():.2f}, max: {spec.max():.2f}, mean: {spec.mean():.2f}")


        num_valid_segments = len(segment_specs_list)
        if num_valid_segments == 0:
            print(f"  WARNING: No valid spectrograms generated for {soundscape_id}")
            return [], []

        all_segment_preds = np.zeros((num_valid_segments, cfg.num_classes), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, num_valid_segments, cfg.batch_size):
                batch_start_idx = i
                batch_end_idx = min(i + cfg.batch_size, num_valid_segments)
                current_batch_size = batch_end_idx - batch_start_idx
                # ### DEBUG ###
                # print(f"    DEBUG: Batch {i//cfg.batch_size + 1}, indices [{batch_start_idx}:{batch_end_idx}], size: {current_batch_size}")

                batch_specs_np = np.array(segment_specs_list[batch_start_idx:batch_end_idx])
                batch_tensors = torch.tensor(batch_specs_np, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
                batch_tensors = normalize(batch_tensors)
                batch_tensors = batch_tensors.to(cfg.device)
                # ### DEBUG ###
                if i == 0 and cfg.debug:
                    print(f"      DEBUG (batch 0 tensor): Shape: {batch_tensors.shape}, min: {batch_tensors.min():.2f}, max: {batch_tensors.max():.2f}, mean: {batch_tensors.mean():.2f}")


                batch_ensemble_preds = []
                for model_idx, model in enumerate(models):
                    outputs = model(batch_tensors)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    batch_ensemble_preds.append(probs)
                    # ### DEBUG ###
                    if i == 0 and model_idx == 0 and cfg.debug:
                        print(f"        DEBUG (model {model_idx}, batch 0 logits): Shape: {outputs.shape}, min: {outputs.min():.2f}, max: {outputs.max():.2f}, mean: {outputs.mean():.2f}")
                        print(f"        DEBUG (model {model_idx}, batch 0 probs): Shape: {probs.shape}, min: {probs.min():.4f}, max: {probs.max():.4f}, mean: {probs.mean():.4f}, sum_per_sample_avg: {probs.sum(axis=1).mean():.2f}")


                if batch_ensemble_preds:
                    batch_avg_preds = np.mean(batch_ensemble_preds, axis=0)
                    all_segment_preds[batch_start_idx:batch_end_idx] = batch_avg_preds
                    # ### DEBUG ###
                    if i == 0 and cfg.debug:
                         print(f"      DEBUG (batch 0 ensemble_avg_preds): Shape: {batch_avg_preds.shape}, min: {batch_avg_preds.min():.4f}, max: {batch_avg_preds.max():.4f}, mean: {batch_avg_preds.mean():.4f}")
                else:
                    print(f"    WARNING: No predictions from ensemble for batch {i//cfg.batch_size + 1}")


            # TTA 
            if cfg.use_tta:
                pass

        predictions_list = list(all_segment_preds)
        return row_ids_list, predictions_list

    except Exception as e:
        print(f"ERROR processing {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], []


# %%
def run_inference(cfg, models):
    test_dir = Path(cfg.test_soundscapes)
    if not test_dir.exists():
         print(f"ERROR: Test soundscapes directory not found at {cfg.test_soundscapes}")
         return [], []
    test_files = sorted(list(test_dir.glob('*.ogg')))
    if not test_files:
         print(f"WARNING: No .ogg files found in {cfg.test_soundscapes}.")
         sample_sub = pd.read_csv(cfg.submission_csv)
         return sample_sub['row_id'].tolist(), [np.zeros(cfg.num_classes) for _ in range(len(sample_sub))]
    if cfg.debug:
        print(f"DEBUG mode: Processing first {cfg.debug_limit} files: {[f.name for f in test_files[:cfg.debug_limit]]}")
        test_files = test_files[:cfg.debug_limit]
    print(f"Found {len(test_files)} test soundscape files to process.")
    all_row_ids = []
    all_predictions = []
    for audio_path_obj in tqdm(test_files, desc="Inferencing Soundscapes"):
        row_ids, preds = predict_soundscape(str(audio_path_obj), models, cfg)
        all_row_ids.extend(row_ids)
        all_predictions.extend(preds)
    return all_row_ids, all_predictions

# %%
def create_submission(row_ids, predictions, species_ids, cfg):
    print("Creating submission dataframe...")
    # ### DEBUG ###
    print(f"  DEBUG (create_submission): len(row_ids): {len(row_ids)}, len(predictions): {len(predictions)}")
    if predictions:
        print(f"  DEBUG (create_submission): Shape of first prediction: {predictions[0].shape if len(predictions[0].shape) > 0 else 'scalar or empty'}")


    if not row_ids or not predictions or len(row_ids) != len(predictions):
        print("  WARNING: Mismatch in row_ids/predictions or empty. Creating submission based on sample.")
        submission_df = pd.read_csv(cfg.submission_csv)
        for col in submission_df.columns:
            if col != 'row_id': submission_df[col] = 0.0
        return submission_df

    submission_dict = {'row_id': row_ids}
    try:
        predictions_array = np.array(predictions)
        if predictions_array.ndim == 1 and predictions_array.shape[0] == len(row_ids) and len(row_ids)>0 and predictions_array.shape[0] > 0 and isinstance(predictions[0], (float, int)):
            print("  ERROR: Predictions seem to be a list of scalars, not arrays of probabilities per class.")
            sample_sub = pd.read_csv(cfg.submission_csv)
            sample_sub.iloc[:, 1:] = 0.0
            return sample_sub

        if predictions_array.shape[1] != len(species_ids):
             print(f"  ERROR: Num predicted columns ({predictions_array.shape[1]}) != num species ({len(species_ids)}).")
             sample_sub = pd.read_csv(cfg.submission_csv)
             sample_sub.iloc[:, 1:] = 0.0
             return sample_sub
        for i, species in enumerate(species_ids):
            submission_dict[species] = predictions_array[:, i]
    except Exception as e:
        print(f"  ERROR converting predictions to array or assigning columns: {e}")
        sample_sub = pd.read_csv(cfg.submission_csv)
        sample_sub.iloc[:, 1:] = 0.0
        return sample_sub


    submission_df = pd.DataFrame(submission_dict)
    # ### DEBUG ###
    print(f"  DEBUG (create_submission): Initial submission_df head:\n{submission_df.head()}")
    print(f"  DEBUG (create_submission): Initial submission_df describe:\n{submission_df.describe()}")


    try:
        sample_sub = pd.read_csv(cfg.submission_csv)
        # ### DEBUG ###
        print(f"  DEBUG (create_submission): sample_submission.csv head:\n{sample_sub.head()}")
        final_df = pd.merge(sample_sub[['row_id']], submission_df, on='row_id', how='left').fillna(0.0)
        final_df = final_df[['row_id'] + cfg.species_ids] 
        # ### DEBUG ###
        print(f"  DEBUG (create_submission): final_df (after merge) head:\n{final_df.head()}")
        print(f"  DEBUG (create_submission): final_df (after merge) describe:\n{final_df.describe()}")


    except Exception as e:
        print(f"  ERROR merging with sample submission: {e}. Returning raw predictions df.")
        final_df = submission_df

    print(f"Submission dataframe created with {len(final_df)} rows.")
    # ### DEBUG ###
    numeric_cols = final_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if final_df[col].min() < 0 or final_df[col].max() > 1:
            print(f"    WARNING: Column {col} has values outside [0,1]. Min: {final_df[col].min()}, Max: {final_df[col].max()}")
    if final_df[numeric_cols].isnull().any().any():
        print(f"    WARNING: Submission contains NaN values!")

    return final_df


# ## Main Execution

# %%
def main():
    overall_start_time = time.time()
    print("--- Starting BirdCLEF 2025 Inference (with DEBUG) ---")

    print("\n--- Loading Models ---")
    models = load_models(cfg)
    if models is None or not models:
        print("Failed to load models. Exiting.")
        # Tạo submission rỗng nếu không load được model
        sample_sub = pd.read_csv(cfg.submission_csv)
        sample_sub.iloc[:, 1:] = 0.0 # Điền 0 cho tất cả các loài
        sample_sub.to_csv('submission.csv', index=False)
        print("Created an empty submission.csv as no models were loaded.")
        return
    print(f"Successfully loaded {len(models)} model(s).")

    print("\n--- Running Inference ---")
    row_ids, predictions = run_inference(cfg, models)

    print("\n--- Creating Submission ---")
    submission_df = create_submission(row_ids, predictions, cfg.species_ids, cfg)

    submission_path = 'submission.csv'
    try:
        submission_df.to_csv(submission_path, index=False)
        print(f"\nSubmission file saved successfully to: {submission_path}")
        # ### DEBUG ###
        print(f"Final Submission head:\n{submission_df.head()}")
        if cfg.debug and not submission_df.empty:
            print(f"Final Submission describe:\n{submission_df.describe().transpose().head(10)}") # Chỉ in 10 loài đầu
            # Kiểm tra một vài row_id cụ thể nếu bạn biết
            # if 'soundscape_X_Y' in submission_df['row_id'].values:
            #     print(submission_df[submission_df['row_id'] == 'soundscape_X_Y'])

    except Exception as e:
        print(f"ERROR saving submission file: {e}")

    overall_end_time = time.time()
    print(f"\n--- Inference Finished ---")
    print(f"Total time: {(overall_end_time - overall_start_time)/60:.2f} minutes")

if __name__ == "__main__":
    gc.collect()
    if cfg.device == 'cuda': 
        torch.cuda.empty_cache()
    main()


