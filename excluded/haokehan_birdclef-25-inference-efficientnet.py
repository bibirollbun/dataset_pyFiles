import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)



class CFG:
    """
    Configuration class holding all paths and parameters required for the inference pipeline.
    """
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/pth-birdclsefnet'
    
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
    threshold = 0.7
    
    use_specific_folds = False  # If False, use all found models
    folds = [0, 1]  # Used only if use_specific_folds is True
    
    debug = False
    debug_count = 3



class BirdCLEF2025Pipeline:
    """
    Pipeline for the BirdCLEF-2025 inference task.

    This class organizes the complete inference process:
      - Loading taxonomy data.
      - Loading and preparing the trained models.
      - Processing audio files into mel spectrograms.
      - Making predictions on each audio segment.
      - Creating the submission file.
      - Post-processing the submission to smooth predictions.
    """

    class BirdCLEFModel(nn.Module):
        """
        Custom neural network model for BirdCLEF-2025 that uses a timm backbone.
        """
        def __init__(self, cfg, num_classes):
            """
            Initialize the BirdCLEFModel.
            
            :param cfg: Configuration parameters.
            :param num_classes: Number of output classes.
            """
            super().__init__()
            self.cfg = cfg
            # Create backbone using timm with specified parameters.
            self.backbone = timm.create_model(
                cfg.model_name,
                pretrained=False,  
                in_chans=cfg.in_channels,
                drop_rate=0.0,    
                drop_path_rate=0.0
            )
            # Adjust final layers based on model type
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
            """
            Forward pass through the network.
            
            :param x: Input tensor.
            :return: Logits for each class.
            """
            features = self.backbone(x)
            if isinstance(features, dict):
                features = features['features']
            # If features are 4D, apply global average pooling.
            if len(features.shape) == 4:
                features = self.pooling(features)
                features = features.view(features.size(0), -1)
            logits = self.classifier(features)
            return logits

    def __init__(self, cfg):
        """
        Initialize the inference pipeline with the given configuration.
        
        :param cfg: Configuration object with paths and parameters.
        """
        self.cfg = cfg
        self.taxonomy_df = None
        self.species_ids = []
        self.models = []
        self._load_taxonomy()

    def _load_taxonomy(self):
        """
        Load taxonomy data from CSV and extract species identifiers.
        """
        print("Loading taxonomy data...")
        self.taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = self.taxonomy_df['primary_label'].tolist()
        print(f"Number of classes: {len(self.species_ids)}")

    def audio2melspec(self, audio_data):
        """
        Convert raw audio data to a normalized mel spectrogram.
        
        :param audio_data: 1D numpy array of audio samples.
        :return: Normalized mel spectrogram.
        """
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
        """
        Process an audio segment to obtain a mel spectrogram with the target shape.
        
        :param audio_data: 1D numpy array of audio samples.
        :return: Processed mel spectrogram as a float32 numpy array.
        """
        ##
        #add
        ##
        # Pad audio if it is shorter than the required window size.
        if len(audio_data) < self.cfg.FS * self.cfg.WINDOW_SIZE:
            audio_data = np.pad(
                audio_data,
                (0, self.cfg.FS * self.cfg.WINDOW_SIZE - len(audio_data)),
                mode='constant'
            )
        
        mel_spec = self.audio2melspec(audio_data)
        
        # Resize spectrogram to the target shape if necessary.
        if mel_spec.shape != self.cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            
        return mel_spec.astype(np.float32)

    def find_model_files(self):
        """
        Find all .pth model files in the specified model directory.
        
        :return: List of model file paths.
        """
        model_files = []
        model_dir = Path(self.cfg.model_path)
        for path in model_dir.glob('**/*.pth'):
            model_files.append(str(path))
        return model_files

    def load_models(self):
        """
        Load all found model files and prepare them for ensemble inference.
        
        :return: List of loaded PyTorch models.
        """
        self.models = []
        model_files = self.find_model_files()
        if not model_files:
            print(f"Warning: No model files found under {self.cfg.model_path}!")
            return self.models

        print(f"Found a total of {len(model_files)} model files.")
        
        # If specific folds are required, filter the model files.
        if self.cfg.use_specific_folds:
            filtered_files = []
            for fold in self.cfg.folds:
                fold_files = [f for f in model_files if f"fold{fold}" in f]
                filtered_files.extend(fold_files)
            model_files = filtered_files
            print(f"Using {len(model_files)} model files for the specified folds ({self.cfg.folds}).")
        
        # Load each model file.
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
        """
        Apply test-time augmentation (TTA) to the spectrogram.
        
        :param spec: Input mel spectrogram.
        :param tta_idx: Index indicating which TTA to apply.
        :return: Augmented spectrogram.
        """
        if tta_idx == 0:
            # No augmentation.
            return spec
        elif tta_idx == 1:
            # Time shift (horizontal flip).
            return np.flip(spec, axis=1)
        elif tta_idx == 2:
            # Frequency shift (vertical flip).
            return np.flip(spec, axis=0)
        else:
            return spec

    def predict_on_spectrogram(self, audio_path):
        """
        Process a single audio file and predict species presence for each 5-second segment.
        
        :param audio_path: Path to the audio file.
        :return: Tuple (row_ids, predictions) for each segment.
        """
        predictions = []
        row_ids = []
        soundscape_id = Path(audio_path).stem
        
        try:
            print(f"Processing {soundscape_id}")
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
        """
        Run inference on all test soundscape audio files.
        
        :return: Tuple (all_row_ids, all_predictions) aggregated from all files.
        """
        test_files = list(Path(self.cfg.test_soundscapes).glob('*.ogg'))
        if self.cfg.debug:
            print(f"Debug mode enabled, using only {self.cfg.debug_count} files")
            test_files = test_files[:self.cfg.debug_count]
        print(f"Found {len(test_files)} test soundscapes")

        all_row_ids = []
        all_predictions = []

        for audio_path in tqdm(test_files):
            row_ids, predictions = self.predict_on_spectrogram(str(audio_path))
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
        
        return all_row_ids, all_predictions

    def create_submission(self, row_ids, predictions):
        """
        Create the submission dataframe based on predictions.
        
        :param row_ids: List of row identifiers for each segment.
        :param predictions: List of prediction arrays.
        :return: A pandas DataFrame formatted for submission.
        """
        print("Creating submission dataframe...")
        submission_dict = {'row_id': row_ids}
        for i, species in enumerate(self.species_ids):
            submission_dict[species] = [pred[i] for pred in predictions]

        submission_df = pd.DataFrame(submission_dict)
        submission_df.set_index('row_id', inplace=True)

        sample_sub = pd.read_csv(self.cfg.submission_csv, index_col='row_id')
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        if missing_cols:
            print(f"Warning: Missing {len(missing_cols)} species columns in submission")
            for col in missing_cols:
                submission_df[col] = 0.0

        submission_df = submission_df[sample_sub.columns]
        submission_df = submission_df.reset_index()
        
        return submission_df

    def smooth_submission(self, submission_path):
        """
        Post-process the submission CSV by smoothing predictions to enforce temporal consistency.
        
        For each soundscape (grouped by the file name part of 'row_id'), each row's predictions
        are averaged with those of its neighbors using defined weights.
        
        :param submission_path: Path to the submission CSV file.
        """
        print("Smoothing submission predictions...")
        sub = pd.read_csv(submission_path)
        cols = sub.columns[1:]
        # Extract group names by splitting row_id on the last underscore
        groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            # Get indices for the current group
            idx = np.where(groups == group)[0]
            sub_group = sub.iloc[idx].copy()
            predictions = sub_group[cols].values
            new_predictions = predictions.copy()
            
            if predictions.shape[0] > 1:
                # Smooth the predictions using neighboring segments
                new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
                new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
                for i in range(1, predictions.shape[0]-1):
                    new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
            # Replace the smoothed values in the submission dataframe
            sub.iloc[idx, 1:] = new_predictions
        
        sub.to_csv(submission_path, index=False)
        print(f"Smoothed submission saved to {submission_path}")

    def run(self):
        """
        Main method to execute the complete inference pipeline.
        
        This method:
          - Loads the pre-trained models.
          - Processes test audio files and runs predictions.
          - Creates the submission CSV.
          - Applies smoothing to the predictions.
        """
        start_time = time.time()
        print("Starting BirdCLEF-2025 inference...")
        print(f"TTA enabled: {self.cfg.use_tta} (variations: {self.cfg.tta_count if self.cfg.use_tta else 0})")
        
        self.load_models()
        if not self.models:
            print("No models found! Please check model paths.")
            return
        
        print(f"Model usage: {'Single model' if len(self.models) == 1 else f'Ensemble of {len(self.models)} models'}")
        row_ids, predictions = self.run_inference()
        submission_df = self.create_submission(row_ids, predictions)
        
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Initial submission saved to {submission_path}")
        
        # Apply smoothing on the submission predictions.
        self.smooth_submission(submission_path)
        
        end_time = time.time()
        print(f"Inference completed in {(end_time - start_time) / 60:.2f} minutes")


# Run the BirdCLEF2025 Pipeline:
if __name__ == "__main__":
    cfg = CFG()
    print(f"Using device: {cfg.device}")
    pipeline = BirdCLEF2025Pipeline(cfg)
    pipeline.run()

















































