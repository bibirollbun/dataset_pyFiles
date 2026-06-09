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

# Suppress warnings and limit logging output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/reg-model'  # 修改为训练模型路径
    
    FS = 32000
    WINDOW_SIZE = 5
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (256, 256)
    
    model_name = 'regnety_008'
    in_channels = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 启用 GPU
    
    use_tta = False # 启用 TTA
    tta_count = 3
    threshold = 0.7
    
    use_specific_folds = True
    folds = [2]  # 使用所有 5 个 fold 模型
    
    debug = False
    debug_count = 5

    if debug:
        test_soundscapes = '/kaggle/input/birdclef-2025/train_soundscapes'

class BirdCLEF2025Pipeline:
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

    def __init__(self, cfg):
        self.cfg = cfg
        self.taxonomy_df = None
        self.species_ids = []
        self.models = []
        self._load_taxonomy()

    def _load_taxonomy(self):
        print("Loading taxonomy data...")
        self.taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = self.taxonomy_df['primary_label'].tolist()
        print(f"Number of classes: {len(self.species_ids)}")

    def audio2melspec(self, audio_data):
        if np.isnan(audio_data).any():
            mean_signal = np.nanmean(audio_data)
            audio_data = np.nan_to_num(audio_data, nan=mean_signal)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.cfg.FS,
            n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH,
            n_mels=self.cfg.N_MELS,
            fmin=self.cfg.FMIN,
            fmax=self.cfg.FMAX,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        return mel_spec_norm

    def process_audio_segment(self, audio_data):
        if len(audio_data) < self.cfg.FS * self.cfg.WINDOW_SIZE:
            audio_data = np.pad(
                audio_data,
                (0, self.cfg.FS * self.cfg.WINDOW_SIZE - len(audio_data)),
                mode='constant'
            )
        
        mel_spec = self.audio2melspec(audio_data)
        if mel_spec.shape != self.cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            
        return mel_spec.astype(np.float32)

    def find_model_files(self):
        model_files = []
        model_dir = Path(self.cfg.model_path)
        for path in model_dir.glob('**/*.pth'):
            model_files.append(str(path))
        return model_files

    def load_models(self):
        self.models = []
        model_files = self.find_model_files()
        if not model_files:
            print(f"Warning: No model files found in {self.cfg.model_path}!")
            return self.models

        print(f"Found a total of {len(model_files)} model files.")
        
        if self.cfg.use_specific_folds:
            filtered_files = []
            for fold in self.cfg.folds:
                fold_files = [f for f in model_files if f"fold{fold}" in f]
                filtered_files.extend(fold_files)
            model_files = filtered_files
            print(f"Using {len(model_files)} model files for specified folds: {self.cfg.folds}")
        
        for model_path in model_files:
            try:
                print(f"Loading model: {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device(self.cfg.device))
                model = self.BirdCLEFModel(self.cfg, len(self.species_ids))
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.cfg.device)
                model.eval()
                self.models.append(model)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        
        return self.models

    def apply_tta(self, spec, tta_idx):
        if tta_idx == 0:
            return spec
        elif tta_idx == 1:
            return np.flip(spec, axis=1)
        elif tta_idx == 2:
            return np.flip(spec, axis=0)
        else:
            return spec

    def predict_on_spectrogram(self, audio_path):
        predictions = []
        row_ids = []
        soundscape_id = Path(audio_path).stem
        
        try:
            print(f"Processing {soundscape_id}...")
            audio_data, _ = librosa.load(audio_path, sr=self.cfg.FS)
            total_segments = int(len(audio_data) / (self.cfg.FS * self.cfg.WINDOW_SIZE))
            
            for segment_idx in range(total_segments):
                start_sample = segment_idx * self.cfg.FS * self.cfg.WINDOW_SIZE
                end_sample = start_sample + self.cfg.FS * self.cfg.WINDOW_SIZE
                segment_audio = audio_data[start_sample:end_sample]
                
                end_time_sec = (segment_idx + 1) * self.cfg.WINDOW_SIZE
                row_id = f"{soundscape_id}_{end_time_sec}"
                row_ids.append(row_id)

                if self.cfg.use_tta:
                    all_preds = []
                    for tta_idx in range(self.cfg.tta_count):
                        mel_spec = self.process_audio_segment(segment_audio)
                        mel_spec = self.apply_tta(mel_spec, tta_idx)
                        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        mel_spec_tensor = mel_spec_tensor.to(self.cfg.device)

                        if len(self.models) == 1:
                            with torch.no_grad():
                                outputs = self.models[0](mel_spec_tensor)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                all_preds.append(probs)
                        else:
                            segment_preds = []
                            for model in self.models:
                                with torch.no_grad():
                                    outputs = model(mel_spec_tensor)
                                    probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                    segment_preds.append(probs)
                            avg_preds = np.mean(segment_preds, axis=0)
                            all_preds.append(avg_preds)
                    final_preds = np.mean(all_preds, axis=0)
                else:
                    mel_spec = self.process_audio_segment(segment_audio)
                    mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec_tensor = mel_spec_tensor.to(self.cfg.device)
                    
                    if len(self.models) == 1:
                        with torch.no_grad():
                            outputs = self.models[0](mel_spec_tensor)
                            final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                    else:
                        segment_preds = []
                        for model in self.models:
                            with torch.no_grad():
                                outputs = model(mel_spec_tensor)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        final_preds = np.mean(segment_preds, axis=0)
                
                predictions.append(final_preds)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
        
        return row_ids, predictions

    def run_inference(self):
        test_files = list(Path(self.cfg.test_soundscapes).glob('*.ogg'))
        if self.cfg.debug:
            print(f"Debug mode is enabled. Using only {self.cfg.debug_count} files")
            test_files = test_files[:self.cfg.debug_count]
        print(f"Found {len(test_files)} test soundscape files")

        all_row_ids = []
        all_predictions = []

        for audio_path in tqdm(test_files):
            row_ids, predictions = self.predict_on_spectrogram(str(audio_path))
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
        
        return all_row_ids, all_predictions

    def create_submission(self, row_ids, predictions):
        print("Creating submission DataFrame...")
        submission_dict = {'row_id': row_ids}
        for i, species in enumerate(self.species_ids):
            submission_dict[species] = [pred[i] for pred in predictions]

        submission_df = pd.DataFrame(submission_dict)
        submission_df.set_index('row_id', inplace=True)

        sample_sub = pd.read_csv(self.cfg.submission_csv, index_col='row_id')
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} species are missing in the submission results")
            for col in missing_cols:
                submission_df[col] = 0.0

        submission_df = submission_df[sample_sub.columns]
        submission_df = submission_df.reset_index()
        
        return submission_df

    def smooth_submission(self, submission_path):
        print("Smoothing submission predictions...")
        sub = pd.read_csv(submission_path)
        cols = sub.columns[1:]
        groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            idx = np.where(groups == group)[0]
            sub_group = sub.iloc[idx].copy()
            predictions = sub_group[cols].values
            new_predictions = predictions.copy()
            
            if predictions.shape[0] > 1:
                new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
                new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
                for i in range(1, predictions.shape[0]-1):
                    new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
            sub.iloc[idx, 1:] = new_predictions
        
        sub.to_csv(submission_path, index=False)
        print(f"Smoothed submission saved to {submission_path}")

    def run(self):
        start_time = time.time()
        print("Starting BirdCLEF-2025 inference...")
        print(f"TTA enabled: {self.cfg.use_tta} (number of augmentations: {self.cfg.tta_count if self.cfg.use_tta else 0})")
    
        self.load_models()
        if not self.models:
            print("No models found! Please check the model path.")
            return
    
        print(f"Using model: {'single model' if len(self.models) == 1 else f'ensemble of {len(self.models)} models'}")
        row_ids, predictions = self.run_inference()
        submission_df = self.create_submission(row_ids, predictions)
    
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Initial submission file saved to {submission_path}")
    
        self.smooth_submission(submission_path)
    
        end_time = time.time()
        print(f"Inference completed (duration: {(end_time - start_time) / 60:.2f} minutes)")

if __name__ == "__main__":
    cfg = CFG()
    pipeline = BirdCLEF2025Pipeline(cfg)
    pipeline.run()




