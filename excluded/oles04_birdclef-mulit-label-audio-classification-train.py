import ast

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from pathlib import Path


class BirdCLEFDataset(Dataset):
    
    def __init__(
        self,
        csv_path: str | Path,          
        audio_dir: str | Path,
        sample_rate: int = 32_000,
        duration_sec: float = 5.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length = 1024,
        hop_length: int = 512,
        f_min: int = 50,
        f_max: int = 16_000,
    ):
        super().__init__()
        
        meta = (
            pd.read_csv(csv_path)
            .rename(columns={"filename": "filepath", "primary_label": "primary"})
        )
        meta["secondary"] = meta["secondary_labels"].apply(self._parse_secondary)
        self.meta = meta[["filepath", "primary", "secondary"]]

        # label mapping
        unique_labels = set(meta["primary"])
        for lst in meta["secondary"]:
            unique_labels.update(lst)

        self.label2idx = {
            lbl: idx for idx, lbl in enumerate(sorted(unique_labels))
        }
        self.idx2label = [lbl for lbl, _ in sorted(self.label2idx.items(), key=lambda x: x[1])]
        
        self.n_classes = len(self.label2idx)

        self.audio_dir = Path(audio_dir)
        self.sr = sample_rate
        self.samples_per_clip = int(sample_rate * duration_sec)

        # Pre-build transforms so they live on the CPU workers
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate    = sample_rate,
            n_fft          = n_fft,
            win_length     = win_length,
            hop_length     = hop_length,
            n_mels         = n_mels,
            f_min          = f_min,
            f_max          = f_max,
        )
        self.db  = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    @staticmethod
    def _parse_secondary(text: str) -> list[str]:
        """
        Convert the string form of secondary labels to a list.
    
           Args:
               text (str): List as string.
           Returns:
               list: Converted string.
        """
        if pd.isna(text):
            return []
        labels = ast.literal_eval(text)
        return [lbl for lbl in labels if lbl]  

    def _load_wave(self, wav_path: Path) -> torch.Tensor:
        """
        Read an audio file and return a mono 5 seconds, fixed-length waveform.
    
        Args:
            wav_path (Path): Absolute or relative path to an audio file
                readable by `torchaudio.load` (WAV, FLAC, Ogg, MP3 …).
        Returns:
            torch.Tensor: A 2-D tensor with shape  
            `[1, self.samples_per_clip]` and `dtype=torch.float32`.
    
                * First dim = 1 (mono channel).  
                * Second dim = timeline in **samples**.
        Raises:
            RuntimeError: If `torchaudio.load` cannot decode the file.   
        """
        wav, _ = torchaudio.load(wav_path)          # shape [chn, time]
        wav = torch.mean(wav, dim=0, keepdim=True)   # mono → [1, time]

        # Pad / trim to fixed length
        n = wav.shape[-1]
        if n < self.samples_per_clip:                      # pad
            pad_amt = self.samples_per_clip - n
            wav = torch.nn.functional.pad(wav, (0, pad_amt))
        elif n > self.samples_per_clip:                    # crop random (or centred)
            start = torch.randint(0, n - self.samples_per_clip + 1, (1,)).item()
            wav = wav[..., start : start+self.samples_per_clip]

        return wav

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        path = self.audio_dir / row["filepath"]
        wav = self._load_wave(path)               # [1, samples]
        spec = self.db(self.mel(wav))              # [1, n_mels, time]

        idxs = [self.label2idx[row.primary]] + [
            self.label2idx[s] for s in row.secondary
        ]
        target = torch.zeros(self.n_classes, dtype=torch.float32)
        target[idxs] = 1.0

        return spec, target




def make_loader(csv_path: str, 
                audio_dir: str,
                *,
                batch_size: int = 32,
                num_workers: int = 4,
                shuffle: bool = True,
                **dataset_kwargs) -> DataLoader:          
    """
    Factory that returns a fully configured ``DataLoader``.

    Args:
        csv_path (str): Metadata CSV (one row per recording).
        audio_dir (str): Root directory that contains the audio files.
        batch_size (int): Items per batch yielded by the loader, default 32. 
        num_workers (int): Number of background worker processes, default 4.
        shuffle (bool): Whether to reshuffle the dataset each epoch, default True.
        **dataset_kwargs
            Any additional arguments accepted by
            :class:`BirdCLEFDataset` (sample-rate, mel params, etc.).
    Returns:
        torch.utils.data.DataLoader
    """

    ds = BirdCLEFDataset(csv_path, audio_dir, **dataset_kwargs)

    return DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = True,
    )


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
    delta of delta --> acceleration

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
    def __init__(
        self, 
        model_name: str,
        taxonomy_path: str,
        n_mels: int,
        sr: int,
        duration_train: float,
        infer_duration: float,
        bias_for_linear_connection: bool,
        attention_activation: str,
        device,
    ) -> None:
        """
        Initializes the BirdCLEFModel.

        Args:
           model_name (str): Name of the base model we want to use
           taxonomy_path (str): Path which has the meta data for the labels
           n_mels (int): Number of Mel bins (frequency dimension).
           SR (float): Sample rate (used for infer_duration calculation).
           duration_train (float): Duration of training audio segments in seconds.
           infer_duration (float): Duration of inference audio segments in seconds.
           device (str): 'cuda' or 'cpu'.
        """
        super().__init__()

        taxonomy_df = pd.read_csv(taxonomy_path)
        self.num_classes = len(taxonomy_df) 

        self.bn0 = nn.BatchNorm2d(n_mels) 

        self.backbone = timm.create_model(
            model_name,
            pretrained=True, # false for inference 
            in_chans=3,
            drop_rate=0.2,
            drop_path_rate=0.2,
        )

        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        # Determine the number of output features from the backbone's encoder
        if "efficientnet" in model_name:
            backbone_out = self.backbone.classifier.in_features
        elif "eca" in model_name:
            backbone_out = self.backbone.head.fc.in_features
        elif "res" in model_name:
            backbone_out = self.backbone.fc.in_features
        else:
            raise NotImplementedError(f"Model: '{model_name}' not supported.")

        self.fc1 = nn.Linear(backbone_out, backbone_out, bias=True)

        self.att_block = AttBlockV2(
            backbone_out, 
            self.num_classes, 
            activation=attention_activation
        )

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


cfg = {
    'csv_path': '/kaggle/input/birdclef-2025/train.csv',
    'audio_path': '/kaggle/input/birdclef-2025/train_audio',
    'taxonomy_path': '/kaggle/input/birdclef-2025/taxonomy.csv'
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
    'duration_train': 10,
    'infer_duration': 5,
    'num_workers': 4,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 25
}


train_loader = make_loader(
    # Dataset kwargs
    csv_path=cfg['csv_path'],
    audio_dir=cfg['audio_path'],
    sample_rate=cfg['SR'],
    duration_sec=cfg['target_duration'],
    n_mels=cfg['n_mels'],
    n_fft=cfg['n_fft'],
    win_length=cfg['win_length'],
    hop_length=cfg['hop_length'],
    f_min=cfg['f_min'],
    f_max=cfg['f_max'],

    # DataLoder kwargs
    batch_size=cfg['batch_size'],
    num_workers=cfg['num_workers'],
    shuffle=True,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-label classification.
        alpha: Can be a scalar, or a tensor of shape (num_classes,) for per-class weighting.
               It often weights the positive class. A common starting point is 0.25 if positive class is rare.
               Given your case (2 positive vs 204 negative), you might want alpha to effectively
               upweight the positive examples. If alpha < 0.5, it downweights the class it applies to.
               Alternatively, for multi-label, it can be tricky. Some use alpha for the positive class
               and (1-alpha) for the negative class.
               A simpler approach if you handle the main imbalance (2 vs 204) with alpha might be
               to set alpha such that positive examples get higher weight (e.g. alpha_for_positives = 0.75).
               Or, ensure your 'targets' are balanced by pos_weight in BCE_loss first and then apply focal modulation.
               For simplicity here, let's assume alpha is applied to scale the loss of the positive samples.
               A common interpretation is alpha for the positive class and 1-alpha for the negative class.
               If you have few positives, an alpha > 0.5 for positive samples might be desired.
               Let's use a simpler formulation where alpha weights all loss contributions (can be 1 if not needed)
               and gamma is the main focusing parameter.
        gamma: Focusing parameter.
        reduction: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets): 
        
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        modulating_factor = (1.0 - pt).pow(self.gamma)
        
        focal_loss = alpha_factor * modulating_factor * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss


model = BirdCLEFModel(
    model_name = cfg['model_name'],
    taxonomy_path = cfg['taxonomy_path'],
    n_mels = cfg['n_mels'],
    sr = cfg['sr'],
    duration_train = cfg['duration_train'],
    infer_duration = cfg['infer_duration'],
    bias_for_linear_connection = cfg['bias_for_linear_connection'],
    attention_activation = cfg['attention_activation'],
    device= cfg['device'], 
).to(cfg['device'])

# Loss calc based on Binary Cross Entropy with Logits --> multi labes (softmax only single label)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])

# --- Training Loop ---
print("\nStarting training loop...")
for epoch in range(cfg['num_epochs']):
    model.train() 
    running_loss = 0.0

    for i, (spectrograms, labels) in enumerate(train_loader):
        spectrograms = spectrograms.to(cfg['device'])
        labels = labels.to(cfg['device'])

        optimizer.zero_grad() 

        outputs = model(spectrograms)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{cfg['num_epochs']}, Loss: {running_loss / len(train_loader):.4f}")


now = datetime.datetime.now()
date = now.strftime('%Y-%m-%d_%H-%M-%S')
model_save_path = f'birdclef_model_{date}.pth'
torch.save(model.state_dict(), model_save_path)
print(f"\nTraining complete. Model saved to {model_save_path}")




