import os

import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram

from pathlib import Path


class BirdCLEFInferenceDataset(Dataset):
    """
    Dataset for 1-minute test soundscape files.
    Each file is split into 12 × 5-second segments.
    Returns Mel spectrogram and a row_id like 'soundscape_1234_5'.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        wave_sec: int = 5,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 512,
        f_min: int = 50,
        f_max: int = 16000,
        n_mels: int = 128
    ):
        self.audio_dir = Path(audio_dir)
        self.audio_files = sorted(list(self.audio_dir.glob("*.ogg")))  # or *.wav if needed
        self.wave_sec = wave_sec
        self.sample_rate = sample_rate
        self.chunk_samples = sample_rate * wave_sec

        self.spectrogram_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            f_min=f_min,
            f_max=f_max,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            n_mels=n_mels,
            mel_scale="htk",
        )

    @staticmethod
    def normalize_std(spec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = torch.mean(spec)
        std = torch.std(spec)
        return torch.where(std == 0, spec - mean, (spec - mean) / (std + eps))

    def __len__(self) -> int:
        return len(self.audio_files) * 12  # 12 chunks per file

    def __getitem__(self, idx: int):
        file_idx = idx // 12  # which file
        chunk_idx = idx % 12  # which 5s segment in the file

        filepath = self.audio_files[file_idx]
        filename_stem = filepath.stem  # 'soundscape_8358733'
        row_id = f"{filename_stem}_{(chunk_idx + 1) * 5}"

        waveform, _ = torchaudio.load(filepath, backend="soundfile")
        waveform = waveform[0, :].unsqueeze(0)  # mono (1, samples)

        start = chunk_idx * self.chunk_samples
        end = start + self.chunk_samples
        chunk = waveform[:, start:end]

        if chunk.shape[1] < self.chunk_samples:
            pad = self.chunk_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        melspec = self.spectrogram_transform(chunk)
        melspec = torch.log(melspec + 1e-6)
        melspec = self.normalize_std(melspec)

        return melspec, row_id


dataset = BirdCLEFInferenceDataset(
    audio_dir="/kaggle/input/birdclef-2025/test_soundscapes"
)

loader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2
)


# having a look
for mel, row_id in loader:
    print(mel.shape)
    print(row_id)
    break


import datetime
import tqdm

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Dict, Tuple, Any, Optional


def init_layer(layer: nn.Module): 
    """
    Applies Xavier uniform initialization to the weights.
    To have better variance for the initial weights which leads to avoiding
    e.g. sigmoids weaknesses of getting linear for weights close to 0.
    """
    nn.init.xavier_uniform_(layer.weight)

    # for beeing unbiased in the beginning if we have a bias
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
            

class AttBlockV2(nn.Module):
    """
    Implements a temporal attention mechanism for audio classification.

    This block takes a sequence of features (frame-level embeddings from a backbone CNN) and
    computes a weighted average of these features to produce a single, clip-level representation.
    The attention weights are learned, indicating which parts of the audio sequence are most
    relevant for each class. This allows the model to selectively focus on informative segments
    (e.g., where a bird call is present) rather than treating all time segments equally.
    This is crucial for sound event detection and classification where target sounds might be
    sparse within a long recording.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "sigmoid"
    ) -> None:
        """
        Initializes the AttBlockV2.

        Args:
            in_features: Number of input features per time step from the backbone.
            out_features: Number of output classes.
            activation: Activation function to apply to the classification branch ('linear' or 'sigmoid').
        """
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initializes weights of the attention and classification convolutional layers.
        """
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass through the attention block.

        Args:
            x: Input tensor of shape (batch_size, in_features, n_time).

        Returns:
            A tuple containing:
            - clipwise_output: (batch_size, out_features) - The clip-level aggregated features for each class.
            - norm_att: (batch_size, out_features, n_time) - Normalized attention weights across time.
            - cla_output: (batch_size, out_features, n_time) - Frame-level classification features (before weighted sum).
        """
        # to learn attention scores 
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)

        # frame-level class scores
        cla_output = self.nonlinear_transform(self.cla(x))

        # combine those
        clipwise_output = torch.sum(norm_att * cla_output, dim=2)

        return clipwise_output, norm_att, cla_output

    def nonlinear_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the specified activation function to the input.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying activation.
        """
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise NotImplementedError(f"Activation '{self.activation}' not supported.")


def image_delta(x: torch.Tensor) -> torch.Tensor:
    """
    Computes first and second order differences along the time axis of a spectrogram.
    This expands a single-channel spectrogram into a 3-channel (original, delta, delta-delta) input.
    delta --> difference in mel energy betwen time frames 

    Args:
        x: Input spectrogram tensor of shape (batch, 1, freq, time).

    Returns:
        A tensor of shape (batch, 3, freq, time) with original, delta, and delta-delta channels.
    """
    if x.shape[1] != 1:
        raise ValueError("image_delta expects input with 1 channel for this placeholder.")

    diff1_raw = x[:, :, :, 1:] - x[:, :, :, :-1]
    delta1 = F.pad(diff1_raw, (0, 1, 0, 0), 'constant', 0)
    
    diff2_raw = delta1[:, :, :, 1:] - delta1[:, :, :, :-1]
    delta2 = F.pad(diff2_raw, (0, 1, 0, 0), 'constant', 0)

    return torch.cat([x, delta1, delta2], dim=1)


class BirdCLEFModel(nn.Module):
    """
    A trainable PyTorch model for BirdCLEF audio classification.

    This model takes Mel spectrograms as input, processes them through a
    backbone CNN for feature extraction, and then uses an attention mechanism
    to aggregate temporal features for clip-level classification.
    """
    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Initializes the BirdCLEFModel.

        Args:
            cfg: A dictionary containing model configuration parameters, e.g.:
                - 'n_mels': Number of Mel bins (frequency dimension).
                - 'model_name': Name of the timm backbone model (e.g., 'efficientnet_b0').
                - 'SR': Sample rate (used for infer_duration calculation).
                - 'duration_train': Duration of training audio segments in seconds.
                - 'infer_duration': Duration of inference audio segments in seconds.
                - 'device': 'cuda' or 'cpu'.
        """
        super().__init__()
        self.cfg = cfg

        taxonomy_df = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
        self.num_classes = len(taxonomy_df) 

        self.bn0 = nn.BatchNorm2d(cfg['n_mels']) 

        self.backbone = timm.create_model(
            cfg['model_name'],
            pretrained=False, # was false but True for imagenet base
            in_chans=3,
            drop_rate=0.2,
            drop_path_rate=0.2,
        )

        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        # Determine the number of output features from the backbone's encoder
        if "efficientnet" in self.cfg['model_name']:
            backbone_out = self.backbone.classifier.in_features
        elif "eca" in self.cfg['model_name']:
            backbone_out = self.backbone.head.fc.in_features
        elif "res" in self.cfg['model_name']:
            backbone_out = self.backbone.fc.in_features
        else:
            raise NotImplementedError(f"Model: '{cfg['model_name']}' not supported.")

        self.fc1 = nn.Linear(backbone_out, backbone_out, bias=True)

        self.att_block = AttBlockV2(backbone_out, self.num_classes, activation="sigmoid")

    def extract_feature(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Extracts frame-level features from the spectrogram using the backbone encoder.

        Args:
            x: Input spectrogram tensor of shape (batch_size, channels, n_mels, n_frames).

        Returns:
            A tuple containing:
            - Features tensor: (batch_size, backbone_out_features, n_frames_reduced)
            - Original number of time frames.
        """
        original_frames_num = x.shape[3]

        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        # The pretrained model from timm without head
        x = self.encoder(x) 

        x = torch.mean(x, dim=2) 

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)

        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2) 

        x = F.dropout(x, p=0.5, training=self.training)

        return x, original_frames_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            x: Input Mel spectrogram tensor of shape (batch_size, channels, n_mels, n_frames).
               Channels will be 1 for mono, to apply the image detla after it.
               The model assumes normalization (e.g., to [0,1]) has already been applied.

        Returns:
            Logits for clip-wise classification output of shape (batch_size, num_classes).
        """
        if x.shape[1] != 1: 
            raise ValueError(f"Warning: Model expects 1 channel, but input has {x.shape[1]}.")
            
        x = image_delta(x)
        features, _ = self.extract_feature(x)

        (clipwise_output, _, _) = self.att_block(features)

        return torch.logit(clipwise_output)

    def attention_infer(self, start: int, end: int, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function for inference, computing framewise predictions from a slice of features.

        Args:
            start: Start frame index for the feature slice.
            end: End frame index for the feature slice.
            x: Feature tensor of shape (batch_size, features, n_time).

        Returns:
            Max framewise probabilities for each class (batch_size, num_classes).
        """
        feat_slice = x[:, :, start:end]
        # Get framewise probabilities from the classification branch
        framewise_pred = torch.sigmoid(self.att_block.cla(feat_slice))
        # Take the maximum probability across frames for each class
        framewise_pred_max = framewise_pred.max(dim=2)[0]
        return framewise_pred_max

    def infer(self, x: torch.Tensor, tta_delta: int = 2) -> torch.Tensor:
        """
        Performs inference with Test Time Augmentation (TTA).

        Args:
            x: Input Mel spectrogram tensor of shape (batch_size, channels, n_mels, n_frames).
            tta_delta: Number of frames to shift for TTA.

        Returns:
            Averaged clip-wise predictions (probabilities) of shape (batch_size, num_classes).
        """
        with torch.no_grad():
            if x.shape[1] != 1:
               raise ValueError(f"Warning: Model expects 1 channel, but input has {x.shape[1]}.")
                
            x = image_delta(x)
            features, _ = self.extract_feature(x)

            feat_time = features.size(-1)

            # Calculate central crop start and end points
            start = int(feat_time / 2 - feat_time * (self.cfg['infer_duration'] / self.cfg['duration_train']) / 2)
            end = int(start + feat_time * (self.cfg['infer_duration'] / self.cfg['duration_train']))

            # Base --> Get prediction for central crop
            pred = self.attention_infer(start, end, features)

            # TTA --> shifted
            start_minus = max(0, start - tta_delta)
            end_minus = end - tta_delta
            pred_minus = self.attention_infer(start_minus, end_minus, features)

            start_plus = start + tta_delta
            end_plus = min(feat_time, end + tta_delta)
            pred_plus = self.attention_infer(start_plus, end_plus, features)

            # combinbe it
            final_pred = 0.5 * pred + 0.25 * pred_minus + 0.25 * pred_plus
            return final_pred


import pandas as pd
import numpy as np

import time


cfg = {
    'csv_path': '/kaggle/input/birdclef-2025/train.csv',
    'audio_path': '/kaggle/input/birdclef-2025/train_audio',
    'SR': 32000,
    'target_duration': 10.0,
    'hop_length': 512, #320,
    'win_length': 1024,
    'n_mels': 128,
    'f_min': 20,
    'f_max': 16000,
    'n_fft': 1024, #2048,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'in_channels': 1,
    'model_name': 'efficientnet_b0',
    'duration_train': 5,
    'infer_duration': 5,
    'num_workers': 4,

    'model_weights_path': '/kaggle/input/model/pytorch/default/1/birdclef_model.pth'
}


print(f"Loading model: {cfg['model_name']}")
model = BirdCLEFModel(cfg).to(cfg['device'])

print("Class Labels ...")
class_labels = sorted(os.listdir('../input/birdclef-2025/train_audio/'))

#class_labels = loader.dataset.idx2label

print(f"Loading weights from: {cfg['model_weights_path']}")
model.load_state_dict(torch.load(cfg['model_weights_path'], map_location=cfg['device']))
print("Model weights loaded successfully.")

print("Create pred")
pred_dict = {'row_id': []}
for species_code in class_labels:
    pred_dict[species_code] = []

model.eval() 

start_time = time.time()

with torch.no_grad():
    for mel_specs, row_id in loader:
        mel_specs = mel_specs.to(cfg['device'])
        row_id = row_id[0]
        
        probs = model.infer(mel_specs)
        
        probs_np = probs.cpu().numpy()[0]
        
        pred_dict['row_id'].append(row_id)
        for class_idx, class_name in enumerate(class_labels):
            pred_dict[class_name].append(probs_np[class_idx])

end_time = time.time() 
duration = end_time - start_time 

print("Chunk inference complete.")
print(f"Inference duration: {duration:.2f} seconds") 


results_df = pd.DataFrame(pred_dict)
final_columns = ['row_id'] + class_labels
results_df = results_df[final_columns]

results_df.to_csv("submission.csv", index=False)




