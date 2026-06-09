# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# import os
# import gc
# import warnings
# import logging
# import time
# import math
# import cv2
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import librosa
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from tqdm.auto import tqdm

# warnings.filterwarnings("ignore")
# logging.basicConfig(level=logging.ERROR)

# class CFG:
 
#     test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
#     submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
#     taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
#     model_path = '/kaggle/input/mobile-test/pytorch/default/1' 
    
#     # Audio parameters
#     FS = 32000  
#     WINDOW_SIZE = 5  
    
#     # Mel spectrogram parameters
#     N_FFT = 1024
#     HOP_LENGTH = 512
#     N_MELS = 148
#     FMIN = 50
#     FMAX = 14000
#     TARGET_SHAPE = (256, 256)
    
#     # model_name = 'resnet18'
#     # model_name = 'efficientnet_b0'
#     model_name = 'mobilenetv2_100'
#     in_channels = 1
#     device = 'cpu'  
    
#     # Inference parameters
#     batch_size = 16
#     use_tta = False  
#     tta_count = 3   
#     threshold = 0.5
    
#     use_specific_folds = False  # If False, use all found models
#     folds = [0, 1]  # Used only if use_specific_folds is True
    
#     debug = False
#     debug_count = 3

# cfg = CFG()

# taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
# species_ids = taxonomy_df['primary_label'].tolist()
# num_classes = len(species_ids)
# class BirdCLEFModel(nn.Module):
#     def __init__(self, cfg, num_classes):
#         super().__init__()
#         self.cfg = cfg
        
#         self.backbone = timm.create_model(
#             cfg.model_name,
#             pretrained=False,  
#             in_chans=cfg.in_channels,
#             drop_rate=0.0,    
#             drop_path_rate=0.0
#         )
        
#         if 'efficientnet' in cfg.model_name:
#             backbone_out = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Identity()
#         elif 'resnet' in cfg.model_name:
#             backbone_out = self.backbone.fc.in_features
#             self.backbone.fc = nn.Identity()
#         else:
#             backbone_out = self.backbone.get_classifier().in_features
#             self.backbone.reset_classifier(0, '')
        
#         self.pooling = nn.AdaptiveAvgPool2d(1)
#         self.feat_dim = backbone_out
#         self.classifier = nn.Linear(backbone_out, num_classes)
        
#     def forward(self, x):
#         features = self.backbone(x)
        
#         if isinstance(features, dict):
#             features = features['features']
            
#         if len(features.shape) == 4:
#             features = self.pooling(features)
#             features = features.view(features.size(0), -1)
        
#         logits = self.classifier(features)
#         return logits
        
# def audio2melspec(audio_data, cfg):
#     """Convert audio data to mel spectrogram"""
#     if np.isnan(audio_data).any():
#         mean_signal = np.nanmean(audio_data)
#         audio_data = np.nan_to_num(audio_data, nan=mean_signal)

#     mel_spec = librosa.feature.melspectrogram(
#         y=audio_data,
#         sr=cfg.FS,
#         n_fft=cfg.N_FFT,
#         hop_length=cfg.HOP_LENGTH,
#         n_mels=cfg.N_MELS,
#         fmin=cfg.FMIN,
#         fmax=cfg.FMAX,
#         power=2.0
#     )

#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
#     return mel_spec_norm

# def process_audio_segment(audio_data, cfg):
#     """Process audio segment to get mel spectrogram"""
#     if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
#         audio_data = np.pad(audio_data, 
#                           (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)), 
#                           mode='constant')
    
#     mel_spec = audio2melspec(audio_data, cfg)
    
#     # Resize if needed
#     if mel_spec.shape != cfg.TARGET_SHAPE:
#         mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
#     return mel_spec.astype(np.float32)
# def find_model_files(cfg):
#     """
#     Find all .pth model files in the specified model directory
#     """
#     model_files = []
    
#     model_dir = Path(cfg.model_path)
    
#     for path in model_dir.glob('**/*.pth'):
#         model_files.append(str(path))
    
#     return model_files

# def load_models(cfg, num_classes):
#     """
#     Load all found model files and prepare them for ensemble
#     """
#     models = []
    
#     model_files = find_model_files(cfg)
    
#     if not model_files:
#         print(f"Warning: No model files found under {cfg.model_path}!")
#         return models
    
#     print(f"Found a total of {len(model_files)} model files.")
    
#     if cfg.use_specific_folds:
#         filtered_files = []
#         for fold in cfg.folds:
#             fold_files = [f for f in model_files if f"fold{fold}" in f]
#             filtered_files.extend(fold_files)
#         model_files = filtered_files
#         print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
    
#     for model_path in model_files:
#         # try:
#         print(f"Loading model: {model_path}")
#         import os
#         print(os.path.isfile(model_path))
#         with open(model_path, "rb") as f:
#             checkpoint = torch.load(f, map_location="cpu")
#         from collections import OrderedDict
#         temp = OrderedDict()
#         for k, v in checkpoint["model_state_dict"].items():
#             temp[k.replace("module.", "")] = v
#         model = BirdCLEFModel(cfg, num_classes)
#         model.load_state_dict(temp)
#         model = model.to(cfg.device)
#         model.eval()
        
#         models.append(model)
#         # except Exception as e:
#         #     print(f"Error loading model {model_path}: {e}")
    
#     return models

# def predict_on_spectrogram(audio_path, models, cfg, species_ids):
#     """Process a single audio file and predict species presence for each 5-second segment"""
#     predictions = []
#     row_ids = []
#     soundscape_id = Path(audio_path).stem
    
#     try:
#         print(f"Processing {soundscape_id}")
#         audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
#         total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        
#         for segment_idx in range(total_segments):
#             start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
#             end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
#             segment_audio = audio_data[start_sample:end_sample]
            
#             end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
#             row_id = f"{soundscape_id}_{end_time_sec}"
#             row_ids.append(row_id)

#             if cfg.use_tta:
#                 all_preds = []
                
#                 for tta_idx in range(cfg.tta_count):
#                     mel_spec = process_audio_segment(segment_audio, cfg)
#                     mel_spec = apply_tta(mel_spec, tta_idx)

#                     mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#                     mel_spec = mel_spec.to(cfg.device)

#                     if len(models) == 1:
#                         with torch.no_grad():
#                             outputs = models[0](mel_spec)
#                             probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                             all_preds.append(probs)
#                     else:
#                         segment_preds = []
#                         for model in models:
#                             with torch.no_grad():
#                                 outputs = model(mel_spec)
#                                 probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                                 segment_preds.append(probs)
                        
#                         avg_preds = np.mean(segment_preds, axis=0)
#                         all_preds.append(avg_preds)

#                 final_preds = np.mean(all_preds, axis=0)
#             else:
#                 mel_spec = process_audio_segment(segment_audio, cfg)
                
#                 mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#                 mel_spec = mel_spec.to(cfg.device)
                
#                 if len(models) == 1:
#                     with torch.no_grad():
#                         outputs = models[0](mel_spec)
#                         final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                 else:
#                     segment_preds = []
#                     for model in models:
#                         with torch.no_grad():
#                             outputs = model(mel_spec)
#                             probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                             segment_preds.append(probs)

#                     final_preds = np.mean(segment_preds, axis=0)
                    
#             predictions.append(final_preds)
            
#     except Exception as e:
#         print(f"Error processing {audio_path}: {e}")
    
#     return row_ids, predictions
# def apply_tta(spec, tta_idx):
#     """Apply test-time augmentation"""
#     if tta_idx == 0:
#         # Original spectrogram
#         return spec
#     elif tta_idx == 1:
#         # Time shift (horizontal flip)
#         return np.flip(spec, axis=1)
#     elif tta_idx == 2:
#         # Frequency shift (vertical flip)
#         return np.flip(spec, axis=0)
#     else:
#         return spec

# def run_inference(cfg, models, species_ids):
#     """Run inference on all test soundscapes"""
#     test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
#     if cfg.debug:
#         print(f"Debug mode enabled, using only {cfg.debug_count} files")
#         test_files = test_files[:cfg.debug_count]
    
#     print(f"Found {len(test_files)} test soundscapes")

#     all_row_ids = []
#     all_predictions = []

#     for audio_path in tqdm(test_files):
#         row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
#         all_row_ids.extend(row_ids)
#         all_predictions.extend(predictions)
    
#     return all_row_ids, all_predictions

# def create_submission(row_ids, predictions, species_ids, cfg):
#     """Create submission dataframe"""
#     print("Creating submission dataframe...")

#     submission_dict = {'row_id': row_ids}
    
#     for i, species in enumerate(species_ids):
#         submission_dict[species] = [pred[i] for pred in predictions]

#     submission_df = pd.DataFrame(submission_dict)

#     submission_df.set_index('row_id', inplace=True)

#     sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

#     missing_cols = set(sample_sub.columns) - set(submission_df.columns)
#     if missing_cols:
#         print(f"Warning: Missing {len(missing_cols)} species columns in submission")
#         for col in missing_cols:
#             submission_df[col] = 0.0

#     submission_df = submission_df[sample_sub.columns]

#     submission_df = submission_df.reset_index()
    
#     return submission_df

# def main():
#     start_time = time.time()
#     print("Starting BirdCLEF-2025 inference...")
#     print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")

#     models = load_models(cfg, num_classes)
    
#     if not models:
#         print("No models found! Please check model paths.")
#         return
    
#     print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

#     row_ids, predictions = run_inference(cfg, models, species_ids)

#     submission_df = create_submission(row_ids, predictions, species_ids, cfg)

#     submission_path = 'submission.csv'
#     submission_df.to_csv(submission_path, index=False)
#     print(f"Submission saved to {submission_path}")
    
#     end_time = time.time()
#     print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")

# if __name__ == "__main__":
#     main()


# import os
# import gc
# import warnings
# import logging
# import time
# import math
# import cv2
# from pathlib import Path
# from collections import OrderedDict

# import numpy as np
# import pandas as pd
# import librosa
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from tqdm.auto import tqdm

# warnings.filterwarnings("ignore")
# logging.basicConfig(level=logging.ERROR)

# class CFG:
#     test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
#     submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
#     taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
#     model_path = '/kaggle/input/efficient-mfcc-test/pytorch/default/1' 

#     # Audio parameters
#     FS = 32000  
#     WINDOW_SIZE = 5  

#     # Mel spectrogram parameters
#     N_FFT = 1024
#     HOP_LENGTH = 512
#     N_MELS = 148
#     FMIN = 50
#     FMAX = 14000
#     TARGET_SHAPE = (256, 256)

#     model_name = 'efficientnet_b0'
#     # model_name = 'resnet18'
#     in_channels = 3  # 修改为 3 以适应多尺度特征
#     device = 'cpu'  

#     # Inference parameters
#     batch_size = 16
#     use_tta = False  
#     tta_count = 3   
#     threshold = 0.5

#     use_specific_folds = False  # If False, use all found models
#     folds = [0, 1]  # Used only if use_specific_folds is True

#     debug = False
#     debug_count = 3

# cfg = CFG()

# taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
# species_ids = taxonomy_df['primary_label'].tolist()
# num_classes = len(species_ids)

# class BirdCLEFModel(nn.Module):
#     def __init__(self, cfg, num_classes):
#         super().__init__()
#         self.cfg = cfg

#         self.backbone = timm.create_model(
#             cfg.model_name,
#             pretrained=False,  
#             in_chans=cfg.in_channels,
#             drop_rate=0.0,    
#             drop_path_rate=0.0
#         )

#         if 'efficientnet' in cfg.model_name:
#             backbone_out = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Identity()
#         elif 'resnet' in cfg.model_name:
#             backbone_out = self.backbone.fc.in_features
#             self.backbone.fc = nn.Identity()
#         else:
#             backbone_out = self.backbone.get_classifier().in_features
#             self.backbone.reset_classifier(0, '')

#         self.pooling = nn.AdaptiveAvgPool2d(1)
#         self.feat_dim = backbone_out
#         self.classifier = nn.Linear(backbone_out, num_classes)

#     def forward(self, x):
#         features = self.backbone(x)

#         if isinstance(features, dict):
#             features = features['features']

#         if len(features.shape) == 4:
#             features = self.pooling(features)
#             features = features.view(features.size(0), -1)

#         logits = self.classifier(features)
#         return logits


# def audio2melspec(audio_data, cfg):
#     """Convert audio data to mel spectrogram"""
#     if np.isnan(audio_data).any():
#         mean_signal = np.nanmean(audio_data)
#         audio_data = np.nan_to_num(audio_data, nan=mean_signal)

#     mel_spec = librosa.feature.melspectrogram(
#         y=audio_data,
#         sr=cfg.FS,
#         n_fft=cfg.N_FFT,
#         hop_length=cfg.HOP_LENGTH,
#         n_mels=cfg.N_MELS,
#         fmin=cfg.FMIN,
#         fmax=cfg.FMAX,
#         power=2.0
#     )

#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

#     return mel_spec_norm


# def audio2mfcc(audio_data, cfg):
#     """Convert audio data to MFCC"""
#     mfcc = librosa.feature.mfcc(y=audio_data, sr=cfg.FS, n_mfcc=cfg.N_MELS)
#     mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
#     return mfcc_norm


# def audio2chroma(audio_data, cfg):
#     """Convert audio data to chroma feature"""
#     chroma = librosa.feature.chroma_stft(y=audio_data, sr=cfg.FS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
#     chroma_norm = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-8)
#     return chroma_norm


# def process_audio_segment(audio_data, cfg):
#     """Process audio segment to get multi - scale features"""
#     if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
#         audio_data = np.pad(audio_data,
#                             (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)),
#                             mode='constant')

#     mel_spec = audio2melspec(audio_data, cfg)
#     mfcc = audio2mfcc(audio_data, cfg)
#     chroma = audio2chroma(audio_data, cfg)

#     # Resize if needed
#     if mel_spec.shape != cfg.TARGET_SHAPE:
#         mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
#     if mfcc.shape != cfg.TARGET_SHAPE:
#         mfcc = cv2.resize(mfcc, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
#     if chroma.shape != cfg.TARGET_SHAPE:
#         chroma = cv2.resize(chroma, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

#     multi_scale_features = np.stack([mel_spec, mfcc, chroma], axis=0)
#     return multi_scale_features.astype(np.float32)


# def find_model_files(cfg):
#     """
#     Find all .pth model files in the specified model directory
#     """
#     model_files = []

#     model_dir = Path(cfg.model_path)

#     for path in model_dir.glob('**/*.pth'):
#         model_files.append(str(path))

#     return model_files


# def load_models(cfg, num_classes):
#     """
#     Load all found model files and prepare them for ensemble
#     """
#     models = []

#     model_files = find_model_files(cfg)

#     if not model_files:
#         print(f"Warning: No model files found under {cfg.model_path}!")
#         return models

#     print(f"Found a total of {len(model_files)} model files.")

#     if cfg.use_specific_folds:
#         filtered_files = []
#         for fold in cfg.folds:
#             fold_files = [f for f in model_files if f"fold{fold}" in f]
#             filtered_files.extend(fold_files)
#         model_files = filtered_files
#         print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")

#     for model_path in model_files:
#         try:
#             print(f"Loading model: {model_path}")
#             checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))

#             new_ckpt = OrderedDict()
#             for k, v in checkpoint["model_state_dict"].items():
#                 new_ckpt[k.replace("module.", "")] = v

#             model = BirdCLEFModel(cfg, num_classes)
#             model.load_state_dict(new_ckpt)
#             model = model.to(cfg.device)
#             model.eval()

#             models.append(model)
#         except Exception as e:
#             print(f"Error loading model {model_path}: {e}")

#     return models


# def predict_on_spectrogram(audio_path, models, cfg, species_ids):
#     """Process a single audio file and predict species presence for each 5 - second segment"""
#     predictions = []
#     row_ids = []
#     soundscape_id = Path(audio_path).stem

#     try:
#         print(f"Processing {soundscape_id}")
#         audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

#         total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))

#         for segment_idx in range(total_segments):
#             start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
#             end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
#             segment_audio = audio_data[start_sample:end_sample]

#             end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
#             row_id = f"{soundscape_id}_{end_time_sec}"
#             row_ids.append(row_id)

#             if cfg.use_tta:
#                 all_preds = []

#                 for tta_idx in range(cfg.tta_count):
#                     multi_scale_features = process_audio_segment(segment_audio, cfg)
#                     multi_scale_features = apply_tta(multi_scale_features, tta_idx)

#                     multi_scale_features = torch.tensor(multi_scale_features, dtype=torch.float32).unsqueeze(0)
#                     multi_scale_features = multi_scale_features.to(cfg.device)

#                     if len(models) == 1:
#                         with torch.no_grad():
#                             outputs = models[0](multi_scale_features)
#                             probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                             all_preds.append(probs)
#                     else:
#                         segment_preds = []
#                         for model in models:
#                             with torch.no_grad():
#                                 outputs = model(multi_scale_features)
#                                 probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                                 segment_preds.append(probs)

#                         avg_preds = np.mean(segment_preds, axis=0)
#                         all_preds.append(avg_preds)

#                 final_preds = np.mean(all_preds, axis=0)
#             else:
#                 multi_scale_features = process_audio_segment(segment_audio, cfg)

#                 multi_scale_features = torch.tensor(multi_scale_features, dtype=torch.float32).unsqueeze(0)
#                 multi_scale_features = multi_scale_features.to(cfg.device)

#                 if len(models) == 1:
#                     with torch.no_grad():
#                         outputs = models[0](multi_scale_features)
#                         final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                 else:
#                     segment_preds = []
#                     for model in models:
#                         with torch.no_grad():
#                             outputs = model(multi_scale_features)
#                             probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
#                             segment_preds.append(probs)

#                     final_preds = np.mean(segment_preds, axis=0)

#             predictions.append(final_preds)

#     except Exception as e:
#         print(f"Error processing {audio_path}: {e}")

#     return row_ids, predictions


# def apply_tta(spec, tta_idx):
#     """Apply test - time augmentation"""
#     if tta_idx == 0:
#         # Original spectrogram
#         return spec
#     elif tta_idx == 1:
#         # Time shift (horizontal flip)
#         return np.flip(spec, axis=2)
#     elif tta_idx == 2:
#         # Frequency shift (vertical flip)
#         return np.flip(spec, axis=1)
#     else:
#         return spec


# def run_inference(cfg, models, species_ids):
#     """Run inference on all test soundscapes"""
#     test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))

#     if cfg.debug:
#         print(f"Debug mode enabled, using only {cfg.debug_count} files")
#         test_files = test_files[:cfg.debug_count]

#     print(f"Found {len(test_files)} test soundscapes")

#     all_row_ids = []
#     all_predictions = []

#     for audio_path in tqdm(test_files):
#         row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
#         all_row_ids.extend(row_ids)
#         all_predictions.extend(predictions)

#     return all_row_ids, all_predictions


# def create_submission(row_ids, predictions, species_ids, cfg):
#     """Create submission dataframe"""
#     print("Creating submission dataframe...")

#     submission_dict = {'row_id': row_ids}

#     for i, species in enumerate(species_ids):
#         submission_dict[species] = [pred[i] for pred in predictions]

#     submission_df = pd.DataFrame(submission_dict)

#     submission_df.set_index('row_id', inplace=True)

#     sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

#     missing_cols = set(sample_sub.columns) - set(submission_df.columns)
#     if missing_cols:
#         print(f"Warning: Missing {len(missing_cols)} species columns in submission")
#         for col in missing_cols:
#             submission_df[col] = 0.0

#     submission_df = submission_df[sample_sub.columns]

#     submission_df = submission_df.reset_index()

#     return submission_df


# def main():
#     start_time = time.time()
#     print("Starting BirdCLEF - 2025 inference...")
#     print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")

#     models = load_models(cfg, num_classes)

#     if not models:
#         print("No models found! Please check model paths.")
#         return

#     print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

#     row_ids, predictions = run_inference(cfg, models, species_ids)

#     submission_df = create_submission(row_ids, predictions, species_ids, cfg)

#     submission_path = 'submission.csv'
#     submission_df.to_csv(submission_path, index=False)
#     print(f"Submission saved to {submission_path}")

#     end_time = time.time()
#     print(f"Inference completed in {(end_time - start_time) / 60:.2f} minutes")


# if __name__ == "__main__":
#     main()


import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path
from collections import OrderedDict

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

class CFG:
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/noise-efficient-test/pytorch/default/1' 

    # Audio parameters
    FS = 32000  
    WINDOW_SIZE = 5  

    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 148
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (256, 256)

    # model_name = 'mobilenetv2_100'
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

cfg = CFG()

taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,  
            in_chans=cfg.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
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


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
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
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm


def process_audio_segment(audio_data, cfg):
    """Process audio segment to get mel spectrogram"""
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data,
                            (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)),
                            mode='constant')

    mel_spec = audio2melspec(audio_data, cfg)

    # Resize if needed
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

    return mel_spec.astype(np.float32)


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

            new_ckpt = OrderedDict()
            for k, v in checkpoint["model_state_dict"].items():
                new_ckpt[k.replace("module.", "")] = v

            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(new_ckpt)
            model = model.to(cfg.device)
            model.eval()

            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")

    return models


def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5 - second segment"""
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

            mel_spec = process_audio_segment(segment_audio, cfg)
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mel_spec = mel_spec.to(cfg.device)

            if len(models) == 1:
                with torch.no_grad():
                    outputs = models[0](mel_spec)
                    probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            else:
                segment_preds = []
                for model in models:
                    with torch.no_grad():
                        outputs = model(mel_spec)
                        probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                        segment_preds.append(probs)

                probs = np.mean(segment_preds, axis=0)

            predictions.append(probs)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

    return row_ids, predictions


def run_inference(cfg, models, species_ids):
    """Run inference on all test soundscapes"""
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))

    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]

    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)

    return all_row_ids, all_predictions


def create_submission(row_ids, predictions, species_ids, cfg):
    """Create submission dataframe"""
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}

    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()

    return submission_df


def main():
    start_time = time.time()
    print("Starting BirdCLEF - 2025 inference...")
    print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")

    models = load_models(cfg, num_classes)

    if not models:
        print("No models found! Please check model paths.")
        return

    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()

