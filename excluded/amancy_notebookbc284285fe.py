# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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
from timm import create_model
from tqdm.auto import tqdm

# Suppress warnings and limit logging output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
"""
save all the params and config class used in the inference pipeline
"""
test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'  
submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'  
taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv' 
model_path = '/kaggle/input/linear/pytorch/default/1' 

# test_soundscapes = '/mnt/sda/lht/Kaggle/test_soundscapes'  
# submission_csv = '/mnt/sda/lht/Kaggle/sample_submission.csv'  
# taxonomy_csv = '/mnt/sda/lht/Kaggle/taxonomy.csv' 
# model_path = '/mnt/sda/lht/Kaggle/linear' 
    

# sound params
FS = 32000  # sampling frequency(32kHz)
WINDOW_SIZE = 5  # cutting window size (s)
    
# Mel
N_FFT = 1024  # FFT window size
HOP_LENGTH = 256
N_MELS = 256 # filter number
FMIN = 50  # minmum frequency #20
FMAX = 14000  # max freq　#16000
TARGET_SHAPE = (256, 256)  # Mel image size
    
# model params
model_name = 'regnety_008'  # model name
in_channels = 1  # input channel
device = 'cpu'
    
# 推論パラメータ
use_tta = False  # test time augmentation
tta_count = 3  # aug time
threshold = 0.7  
    
use_specific_folds = False  
folds = [0, 1] 
    
# debug setting
debug = False  
debug_count = 5  

if debug:
    test_soundscapes = '/kaggle/input/birdclef-2025/train_soundscapes'

class BirdCLEF2025Pipeline:

    class BirdCLEFModel(nn.Module):
        def __init__(self, feat_dim, num_classes):
            super().__init__()
            # self.encoder = encoder
            self.encoder = create_model("regnety_008", pretrained=False, num_classes=0)
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(feat_dim, num_classes)
    
        def forward(self, x):
            # with torch.no_grad():
            #     feat = self.encoder.forward_features(x)
            # print(x.shape)
            # feat = self.encoder(x)
            feat = self.encoder.forward_features(x)
            # feat = self.gap(feat).view(x.size(0), -1)
            # return self.classifier(feat)
            feat = self.gap(feat).view(x.size(0), -1)
            return self.classifier(feat)

    # class BirdCLEFModel(nn.Module):
    #     def __init__(self, num_classes):
    #         super().__init__()
    #         # self.cfg = cfg
    #         self.backbone = timm.create_model(
    #             model_name, # regnety_008
    #             pretrained=False,  
    #             in_chans=in_channels,   # 1 
    #             drop_rate=0.0,    
    #             drop_path_rate=0.0
    #         )

    #         if 'efficientnet' in model_name:
    #             backbone_out = self.backbone.classifier.in_features # channel
    #             self.backbone.classifier = nn.Identity()  # remove classifier
    #         elif 'resnet' in model_name:
    #             backbone_out = self.backbone.fc.in_features         # channel
    #             self.backbone.fc = nn.Identity()          # remove classifier
    #         else:
    #             backbone_out = self.backbone.get_classifier().in_features
    #             self.backbone.reset_classifier(0, '')
            
    #         self.pooling = nn.AdaptiveAvgPool2d(1)  
    #         self.feat_dim = backbone_out # backbone nerwork output dim
    #         self.classifier = nn.Linear(backbone_out, num_classes)  # classification head
            
    #     def forward(self, x):
    #         """
    #         input: [bs 1, h, w]
    #         output: [bs, num_classes] (logits)
    #         """
    #         features = self.backbone(x)   # feature extraction
    #         if isinstance(features, dict):
    #             features = features['features']
            
    #         if len(features.shape) == 4: # [batch, ch, W, H]
    #             features = self.pooling(features)
    #             features = features.view(features.size(0), -1) # [batch, ch]
    #         logits = self.classifier(features)  # [baatch, num_classes]
    #         return logits

    def __init__(self):
        """
        load 
        """
        # self.cfg = cfg
        self.taxonomy_df = None
        self.species_ids = []
        self.models = []
        self._load_taxonomy()  

    def _load_taxonomy(self):
        """
        load class info from taxonmy.csv
        self.species_ids: class id
        """
        print("loading class info...")
        self.taxonomy_df = pd.read_csv(taxonomy_csv)
        self.species_ids = self.taxonomy_df['primary_label'].tolist() 
        print(f"number of classes: {len(self.species_ids)}")

    def audio2melspec(self, audio_data):
        """
        audio file to mel sepc(image with 1 channel)
        """
        if np.isnan(audio_data).any():
            mean_signal = np.nanmean(audio_data)
            audio_data = np.nan_to_num(audio_data, nan=mean_signal)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=FS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        return mel_spec_norm

    def process_audio_segment(self, audio_data):
        
        # short then padding
        if len(audio_data) < FS * WINDOW_SIZE:
            audio_data = np.pad(
                audio_data,
                (0, FS * WINDOW_SIZE - len(audio_data)),
                mode='constant'
            )
        
        mel_spec = self.audio2melspec(audio_data)  # メルスペクトログラムに変換
        
        # reshape to fit model input shape
        if mel_spec.shape != TARGET_SHAPE:  # (256, 256)
            mel_spec = cv2.resize(mel_spec, TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            
        return mel_spec.astype(np.float32)

    def find_model_files(self):
        model_files = []
        model_dir = Path(model_path)
        for path in model_dir.glob('**/*.pth'):
            model_files.append(str(path))
        return model_files

    def load_models(self):
        self.models = []
        model_files = self.find_model_files() 
        if not model_files:
            print("model file not found!") 
            return self.models

        print(f" total {len(model_files)} models")
        
        if use_specific_folds:
            filtered_files = []
            for fold in folds:
                fold_files = [f for f in model_files if f"fold{fold}" in f]
                filtered_files.extend(fold_files)
            model_files = filtered_files
            print(f"specific file({folds}) for {len(model_files)}will utilize single model file")
        
        for model_path in model_files:
            try:
                print(f"loading model: {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only = False)
                model = self.BirdCLEFModel(768, len(self.species_ids))
                state_dict = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(state_dict)
                # state_dict = torch.load(path)
                # model.load_state_dict(state_dict)
                model = model.to(device)
                model.eval()  # eval mode
                self.models.append(model)
            except Exception as e:
                print(f"loading model {model_path} an error occurs: {e}")
        
        return self.models  # a list of models for ensemble inference??

    def apply_tta(self, spec, tta_idx):
        """
        augmentation on mel spec
        """
        if tta_idx == 0:
            # no aug
            return spec
        elif tta_idx == 1:
            # horizontal flip
            return np.flip(spec, axis=1)
        elif tta_idx == 2:
            # vertical flip
            return np.flip(spec, axis=0)
        else:
            return spec

    def predict_on_spectrogram(self, audio_path):
        predictions = []
        row_ids = []
        soundscape_id = Path(audio_path).stem
        
        try:
            print(f"{soundscape_id}processing...")
            audio_data, _ = librosa.load(audio_path, sr=FS)
            total_segments = int(len(audio_data) / (FS * WINDOW_SIZE))  # 5s a fragment
            
            for segment_idx in range(total_segments):
                start_sample = segment_idx * FS * WINDOW_SIZE
                end_sample = start_sample + FS * WINDOW_SIZE
                segment_audio = audio_data[start_sample:end_sample]
                
                end_time_sec = (segment_idx + 1) * WINDOW_SIZE
                row_id = f"{soundscape_id}_{end_time_sec}"
                row_ids.append(row_id)

                if use_tta:
                    all_preds = []
                    for tta_idx in range(tta_count):
                        mel_spec = self.process_audio_segment(segment_audio)
                        mel_spec = self.apply_tta(mel_spec, tta_idx)
                        # mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                        mel_spec_tensor = mel_spec_tensor.repeat(3, 1, 1).unsqueeze(0)
                        mel_spec_tensor = mel_spec_tensor.to(device)
                        print(f"mel_spec_tensor shape: {mel_spec_tensor.shape}")


                        if len(self.models) == 1:
                            with torch.no_grad():
                                outputs = self.models[0](mel_spec_tensor)   # forward processs for current fragment
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()  # l
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
                    # mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    # mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                    # mel_spec_tensor = mel_spec_tensor.repeat(3, 1, 1).unsqueeze(0)
                    mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
                    mel_spec_tensor = mel_spec_tensor.expand(3, -1, -1).unsqueeze(0) 
                    mel_spec_tensor = mel_spec_tensor.to(device)
                    # print(f"mel_spec_tensor shape: {mel_spec_tensor.shape}")
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
                        final_preds = np.mean(segment_preds, axis=0)  # ensemble prediction??
                
                predictions.append(final_preds)
        except Exception as e:
            print(f"processing {audio_path}an error occurs: {e}")
        
        return row_ids, predictions

    def run_inference(self):
        """
        uisng method predict_on_spectrogram to classify each audio file
        :return: all_row_ids & corresponding predictions
        """
        test_files = list(Path(test_soundscapes).glob('*.ogg'))  
        if debug:
            print(f"activate debug mode. Utilizing {debug_count} examples")
            test_files = test_files[:debug_count]
        print(f"{len(test_files)} number of test files is founded")

        all_row_ids = []
        all_predictions = []

        for audio_path in tqdm(test_files):
            row_ids, predictions = self.predict_on_spectrogram(str(audio_path))
            all_row_ids.extend(row_ids)
            all_predictions.extend(predictions)
        
        return all_row_ids, all_predictions

    def create_submission(self, row_ids, predictions):

        # print("提出用データフレームを作成中...")
        submission_dict = {'row_id': row_ids}
        for i, species in enumerate(self.species_ids):  # classes
            submission_dict[species] = [pred[i] for pred in predictions]

        submission_df = pd.DataFrame(submission_dict)
        submission_df.set_index('row_id', inplace=True)

        sample_sub = pd.read_csv(submission_csv, index_col='row_id')
        missing_cols = set(sample_sub.columns) - set(submission_df.columns)
        if missing_cols:
            print(f"{len(missing_cols)} kind of missing data in the results")
            for col in missing_cols:
                submission_df[col] = 0.0

        submission_df = submission_df[sample_sub.columns] 
        submission_df = submission_df.reset_index()
        
        return submission_df

    def smooth_submission(self, submission_path):
        
        # print("提出結果の予測を平滑化しています...")
        sub = pd.read_csv(submission_path)
        cols = sub.columns[1:]
        # 'row_id'を基にグループを抽出
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
        print(f"smoothed result is saved to {submission_path}")

    def run(self):

        start_time = time.time()
        print("BirdCLEF-2025 start inference...")
        # print(f"TTA有効: {self.cfg.use_tta} (変動数: {self.cfg.tta_count if self.cfg.use_tta else 0})")
    
        self.load_models()
        if not self.models:
            print("no model founded")
            return
    
        print(f"use model numbers: {len(self.models)}")
        row_ids, predictions = self.run_inference()
        submission_df = self.create_submission(row_ids, predictions)
    
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"submission saved at {submission_path}")
    
        self.smooth_submission(submission_path)
    
        end_time = time.time()
        print(f"inference finished (time needed {(end_time - start_time) / 60:.2f} minute)")


# cfg = CFG()
print(f"Using device: {device}")
pipeline = BirdCLEF2025Pipeline()
pipeline.run()  # Use the correct method name here


