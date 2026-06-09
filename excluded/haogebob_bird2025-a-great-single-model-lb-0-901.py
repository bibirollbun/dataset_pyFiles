import glob
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm.auto import tqdm
import glob
import concurrent.futures
import shutil
import albumentations as A
import torchaudio
from typing import Union
warnings.filterwarnings("ignore")


sub = pd.read_csv("../input/birdclef-2025/sample_submission.csv")
target_columns_ = sub.columns.tolist()
target_columns = sub.columns.tolist()[1:]
num_classes = len(target_columns)

TOTAL_SECONDS_CHUNKS = 12
test_path = "/kaggle/input/birdclef-2025/test_soundscapes/"
files = glob.glob(f'{test_path}*')
if len(files) == 1:
    TOTAL_SECONDS_CHUNKS = 2

seconds = [i for i in range(5, (TOTAL_SECONDS_CHUNKS*5) + 5, 5)]


test_path = "/kaggle/input/birdclef-2025/test_soundscapes/"

files = glob.glob(f'{test_path}*')
if len(files) == 1:
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1446779.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1442779.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1446779.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1446379.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1146779.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1426779.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1441779.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1446179.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1446719.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1446771.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1446789.ogg')
    shutil.copy('/kaggle/input/birdclef-2025/train_audio/bubcur1/XC10728.ogg', '/kaggle/working/soundscape_1448779.ogg')
    test_path = "/kaggle/working/"
    
print (test_path)


mel_spec_params = {
    "sample_rate": 32000,
    "n_mels": 128,
    "f_min": 20,
    "f_max": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "normalized": True,
    "center" : True,
    "pad_mode" : "constant",
    "norm" : "slaney",
    "onesided" : True,
    "mel_scale" : "slaney"
}
top_db = 80


def normalize_melspec(X, eps=1e-6):
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V


transforms_val = A.Compose([
    A.Resize(256, 256),
    A.Normalize()
])


class TestDataset(torchdata.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 clip: np.ndarray,
                ):
        
        self.df = df
        self.clip = clip
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)
        self.transform = transforms_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)
        
        wave = self.clip[:, 32000 * start_seconds : 32000 * end_seconds]
        
        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wave)))
        mel_spectrogram = mel_spectrogram * 255
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1).permute(1, 2, 0).numpy()
        
        res = self.transform(image=mel_spectrogram)
        spec = res['image'].astype(np.float32)
        spec = spec.transpose(2, 0, 1)
        
        return {
            "row_id": row_id,
            "wave": spec,
        }


def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = 30,
    exponent: Union[int, float] = 2,
    inplace: bool = True
) -> np.ndarray:
    if not inplace:
        p = p.copy()

    # Identify columns whose max value ranks below `top_k`
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]

    # Apply the power transformation to those columns
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p

def prediction_for_clip(audio_path):
    wav, org_sr = torchaudio.load(audio_path, normalize=True)
    clip = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=32000)
    
    name_ = os.path.basename(audio_path).split(".ogg")[0]
    row_ids_for_df = [f"{name_}_{s}" for s in seconds]

    test_df = pd.DataFrame({"row_id": row_ids_for_df, "seconds": seconds})
    
    dataset = TestDataset(df=test_df, clip=clip) # Uses global mel_spec_params, etc.
    loader = torchdata.DataLoader(dataset, batch_size=16, num_workers=os.cpu_count(), shuffle=False, pin_memory=True)

    # Store {row_id: averaged_probas_array}
    model_averaged_probas_dict = {}

    for batch_data in loader:
        batch_row_ids = batch_data['row_id']
        wave_tensor = batch_data['wave'].to(device)
        
        batch_model_outputs = []
        with torch.no_grad():
            for i, model_instance in enumerate(models):
                
                logits = torch.logit(model_instance.infer(wave_tensor))    

                p10 = logits.quantile(0.10, dim=1, keepdim=True)   # (B, 1)
                p90 = logits.quantile(0.90, dim=1, keepdim=True)   # (B, 1)
            
                clipped_logits = logits.clamp(min=p10, max=p90)
            
                probs_orig   = logits.sigmoid()           # Ïƒ(logits)
                probs_clip   = clipped_logits.sigmoid()   # Ïƒ(clipped_logits)
                probs_blend  = (0.5 * (probs_orig + probs_clip)).cpu().numpy()
            
                # probs_blend = apply_power_to_low_ranked_cols(probs_orig.cpu().numpy(), top_k=30, exponent=2)
                batch_model_outputs.append(probs_blend)
        
        # Average predictions across models for this batch
        # Shape: (num_models, batch_size, num_classes) -> (batch_size, num_classes)
        averaged_batch_probas = np.mean(batch_model_outputs, axis=0)
        
        for i, r_id in enumerate(batch_row_ids):
            model_averaged_probas_dict[str(r_id)] = averaged_batch_probas[i]

    final_predictions_for_clip = {}
    num_chunks = len(seconds)

    for idx, current_sec in enumerate(seconds):
        current_row_id = f"{name_}_{current_sec}"
        current_probas = model_averaged_probas_dict.get(current_row_id, np.zeros(len(target_columns)))

        # Left neighbor (boundary: use current if first)
        left_sec = seconds[max(0, idx - 1)]
        left_row_id = f"{name_}_{left_sec}"
        left_probas = model_averaged_probas_dict.get(left_row_id, current_probas) # Default to current if missing
        if idx == 0: # Explicit boundary for first element
            left_probas = current_probas

        # Right neighbor (boundary: use current if last)
        right_sec = seconds[min(num_chunks - 1, idx + 1)]
        right_row_id = f"{name_}_{right_sec}"
        right_probas = model_averaged_probas_dict.get(right_row_id, current_probas) # Default to current if missing
        if idx == num_chunks - 1: # Explicit boundary for last element
            right_probas = current_probas
        
        # Apply weights: 0.3 * left + 0.4 * current + 0.3 * right
        smoothed_probas = 0.2 * left_probas + 0.6 * current_probas + 0.2 * right_probas
        
        final_predictions_for_clip[current_row_id] = {}
        for i, label_name in enumerate(target_columns):
            final_predictions_for_clip[current_row_id][label_name] = smoothed_probas[i]
            
    return final_predictions_for_clip


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class TimmSED(nn.Module):
    def __init__(self, backbone, pretrained, in_chans=3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # BatchNorm for mel spectrograms
        self.bn0 = nn.BatchNorm2d(256)
        
        # Create backbone model
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=0.2,
            drop_path_rate=0.2,
            # cache_dir=cache_dir
        )
        
        # Remove classifier layers to get features
        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        
        # Get backbone output features
        if "efficientnet" in backbone:
            backbone_out = self.backbone.classifier.in_features
        elif "eca" in backbone:
            backbone_out = self.backbone.head.fc.in_features
        elif "res" in backbone:
            backbone_out = self.backbone.fc.in_features
        else:
            backbone_out = self.backbone.num_features
            
        self.fc1 = nn.Linear(backbone_out, backbone_out, bias=True)
        self.att_block = AttBlockV2(backbone_out, self.num_classes, activation="linear")
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def extract_feature(self, x):
        # x shape: (batch_size, channels, freq, time)
        x = x.permute((0, 1, 3, 2))  # (batch_size, channels, time, freq)
        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)  # (batch_size, freq, time, channels)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (batch_size, channels, time, freq)
        
        x = x.transpose(2, 3)  # (batch_size, channels, freq, time)
        
        # Encoder forward
        x = self.encoder(x)
        
        # Global average pooling on frequency dimension
        x = torch.mean(x, dim=2)  # (batch_size, channels, time)
        
        # Channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)  # (batch_size, time, channels)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)  # (batch_size, channels, time)
        x = F.dropout(x, p=0.5, training=self.training)
        
        return x, frames_num

    def forward(self, x):
        x, frames_num = self.extract_feature(x)
        
        clipwise_output, norm_att, segmentwise_output = self.att_block(x)
        
        return clipwise_output

    def infer(self, x, tta_delta=2, infer_duration=5, train_duration=5):
        
        x, _ = self.extract_feature(x)
        time_att = torch.tanh(self.att_block.att(x))
        feat_time = x.size(-1)
        
        start = feat_time / 2 - feat_time * (infer_duration / train_duration) / 2
        end = start + feat_time * (infer_duration / train_duration)
        start = int(start)
        end = int(end)
        
        pred = self.attention_infer(start, end, x, time_att)
        
        start_minus = max(0, start - tta_delta)
        end_minus = end - tta_delta
        pred_minus = self.attention_infer(start_minus, end_minus, x, time_att)
        
        start_plus = start + tta_delta
        end_plus = min(feat_time, end + tta_delta)
        pred_plus = self.attention_infer(start_plus, end_plus, x, time_att)
        
        pred = 0.5 * pred + 0.25 * pred_minus + 0.25 * pred_plus
        return pred
        
    def attention_infer(self, start, end, x, time_att):
        feat = x[:, :, start:end]
        framewise_pred = torch.sigmoid(self.att_block.cla(feat))
        framewise_pred_max = framewise_pred.max(dim=2)[0]
        return framewise_pred_max


device = "cpu"
model_paths = [
    "/kaggle/input/bird2025-great-models/exp8_modelsoup_1.bin",
    # "/kaggle/input/bird2025-great-models/exp8_modelsoup_2.bin",
    "/kaggle/input/bird2025-great-models/exp8_modelsoup_3.bin",
    
]
models = []
for i, model_path in enumerate(model_paths):
    try:
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        
        model = TimmSED(backbone='eca_nfnet_l0', pretrained=False)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        model.zero_grad()
        
        models.append(model)
    except Exception as e:
        print(f"Error loading model {path}: {e}")

if not models:
    raise ValueError("No models were loaded. Please check model_paths.")

print(f"Total models loaded: {len(models)}")


def main():
    
    all_audios = list(glob.glob(f'{test_path}*.ogg'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        dicts = list(executor.map(prediction_for_clip, all_audios))
    
    prediction_dicts = {}
    for d in dicts:
        prediction_dicts.update(d)
        
    submission = pd.DataFrame.from_dict(prediction_dicts, "index").rename_axis("row_id").reset_index()
    submission.to_csv("submission.csv", index=False)
    print ("Done")

if __name__ == "__main__":
    main()

