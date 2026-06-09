import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import copy
import time
import os
import numpy as np
import gc
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from warmup_scheduler import GradualWarmupScheduler
from torch.optim import AdamW
import torch
import numpy as np
import sklearn
import torchaudio
import torchaudio.transforms as T
import warnings
import random
warnings.filterwarnings("ignore")
import audiomentations as AA
import albumentations as A
import librosa
import librosa.display
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import cv2
import numpy as np
import random
import timm
from utils.metrics import MetricMeter
from utils.data_utils import downsample_data, upsample_data
from torch.amp import autocast, GradScaler
import torchvision
from torchvision.transforms import v2
import json
import time


sub = pd.read_csv(f"../../../data/birdclef-2025/sample_submission.csv")
target_columns = sub.columns.tolist()[1:]


# Fix Warmup Bug
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
        

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

def map_score(solution, submission):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',
    )
    return score

def calculate_competition_metrics(gt, preds, target_columns):
    val_df = pd.DataFrame(gt, columns=target_columns)
    pred_df = pd.DataFrame(preds, columns=target_columns)
    cmAP_1 = padded_cmap(val_df, pred_df, padding_factor=1)
    cmAP_5 = padded_cmap(val_df, pred_df, padding_factor=5)
    mAP = map_score(val_df, pred_df)
    val_df['id'] = [f'id_{i}' for i in range(len(val_df))]
    pred_df['id'] = [f'id_{i}' for i in range(len(pred_df))]
    return {
        "cmAP_1": cmAP_1,
        "cmAP_5": cmAP_5,
        "mAP": mAP,
    }
def metrics_to_string(scores, key_word):
  log_info = ""
  for key in scores.keys():
      log_info = log_info + f"{key_word} {key} : {scores[key]:.4f}, "
  return log_info


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



import os
import copy
import torch
import numpy as np
from collections import OrderedDict

class ModelSoup:
    """
    Implementation of Model Soup technique that averages weights of models
    that improve validation performance.
    
    Reference: https://arxiv.org/abs/2203.05482
    """
    def __init__(self, base_model, save_dir="model_soup", exclude_nocall=False, max_trick=False):
        """
        Initialize Model Soup with a base model
        
        Args:
            base_model: Initial model to start with
            save_dir: Directory to save individual models
        """
        self.base_model = copy.deepcopy(base_model)
        self.best_models = OrderedDict()  # Maps epoch -> model state dict
        self.best_val_score = float('-inf')
        self.initial_val_score = None
        self.save_dir = save_dir
        self.max_trick = max_trick
        self.exclude_nocall = exclude_nocall
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def evaluate_model(self, unseen_index, model, val_loader, criterion, device):
        """
        Evaluate model on validation set
        
        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to run evaluation on
            
        Returns:
            float: Validation accuracy
        """
        if unseen_index is not None:
            scores = MetricMeter(indices_ignore=unseen_index)
        model.eval()
        val_loss = []
        PREDS = []
        TRUES = []
        bar = tqdm(val_loader)
        with torch.no_grad():
            for batch in bar:

                target = batch['target'].to(device)
                spec = batch['spec'].to(device)

                if self.max_trick:
                    BS, K, C, H, W = spec.shape
                    spec = spec.view(BS * K, C, H, W)

                
                predictions = model(spec)
                if self.max_trick:
                    predictions = predictions.view(BS, K, -1)
                    predictions = torch.max(predictions, dim=1).values

                
                if self.exclude_nocall:
                    target = target[:, :-1]
                    predictions = predictions[:, :-1]

                loss = criterion(predictions, target)
                
                val_loss.append(loss.detach().cpu().numpy())
                PREDS.append(predictions.detach().cpu())
                
                target = torch.round(target).long()

                TRUES.append(target.detach().cpu())
                

                if unseen_index is not None:
                    scores.update(target, predictions)
                
                loss_np = loss.detach().cpu().item()
                bar.set_description('loss: %.5f' % (loss_np))

                
        val_loss = np.mean(val_loss)
        P = np.concatenate(PREDS, axis=0)
        T = np.concatenate(TRUES, axis=0)
        metrics = calculate_competition_metrics(T, P, target_columns)
        metrics['score'] = scores.avg
        return metrics
    
    def add_model_if_improved(self, unseen_index, model, epoch, val_loader, device, criterion):
        """
        Add model to soup if it improves validation performance
        
        Args:
            model: Current model
            epoch: Current epoch
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to run evaluation on
            
        Returns:
            bool: Whether model was added to soup
        """
        scores = self.evaluate_model(unseen_index, model, val_loader, criterion, device)
        current_score = scores
        
        
        # Set initial validation score if not set
        if self.initial_val_score is None:
            self.initial_val_score = current_score
            self.best_val_score = current_score
            
            # Save initial model
            self.best_models[epoch] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"model_epoch_{epoch}.pt"))
            print(f"Initial model (Epoch {epoch}) - Validation Score: {json.dumps(current_score)}")
            return True
        print(f"Epoch {epoch} - Validation Scores: {json.dumps(current_score)} (Best: {self.best_val_score})")
        
        # Only add to soup if validation score improves
        is_better = False
        for each_key in current_score.keys():
            if current_score[each_key] > self.best_val_score[each_key]:
                is_better = True
                break
        


        if is_better:
            self.best_val_score = current_score
            self.best_models[epoch] = copy.deepcopy(model.state_dict())
            
            # Save individual model
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"model_epoch_{epoch}.pt"))
            
            print(f"Model from epoch {epoch} added to soup! New best scores: {json.dumps(current_score)}")
            return True
        
        return False
    
    def get_soup_model(self, soup_model_name = "model_soup_final.pt"):
        """
        Create model soup by averaging weights of best models
        
        Returns:
            torch.nn.Module: Model with averaged weights
        """
        if not self.best_models:
            print("No models in soup yet!")
            return self.base_model
        
        # Create a new model with the same architecture
        # Create a new model with the same architecture
        soup_model = copy.deepcopy(self.base_model)

        # Get the first model to initialize parameters
        first_model_state = next(iter(self.best_models.values()))

        # Initialize averaged state dict
        avg_state_dict = OrderedDict()
        for key in first_model_state.keys():
            avg_state_dict[key] = torch.zeros_like(first_model_state[key])

        # Sum all parameters
        for epoch, state_dict in self.best_models.items():
            for key in state_dict.keys():
                avg_state_dict[key] += state_dict[key]

        # Divide by number of models to get average
        num_models = len(self.best_models)
        for key in avg_state_dict.keys():
            avg_state_dict[key] = avg_state_dict[key] / int(num_models)

        # Load averaged parameters into the soup model
        soup_model.load_state_dict(avg_state_dict)
        
        # Save the final soup model
        torch.save(avg_state_dict, os.path.join(self.save_dir, soup_model_name))
        
        print(f"Created model soup from {num_models} models (epochs: {list(self.best_models.keys())})")
        return soup_model




import torch
import torchaudio
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F
import random
import numpy as np
from torch.nn import functional as F_nn


def compute_deltas(
    specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate"
) -> torch.Tensor:
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. math::
       d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(
        specgram.shape[1], 1, 1
    )

    output = (
        torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    )

    # unpack batch
    output = output.reshape(shape)

    return output


class SpecAugment(torch.nn.Module):
    def __init__(self,
                 freq_mask_param=15,
                 time_mask_param=35,
                 num_freq_masks=2,
                 num_time_masks=2):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec):
        for _ in range(self.num_freq_masks):
            spec = self.freq_mask(spec)
        for _ in range(self.num_time_masks):
            spec = self.time_mask(spec)
        return spec

class LocalGlobalStretch(torch.nn.Module):
    """
    Applies local and global stretching to melspectrograms in time and frequency dimensions.
    """
    def __init__(self, 
                 global_stretch_prob=0.5,
                 local_stretch_prob=0.5,
                 max_global_stretch=0.2,
                 max_local_stretch=0.3,
                 max_local_regions=3):
        super().__init__()
        self.global_stretch_prob = global_stretch_prob
        self.local_stretch_prob = local_stretch_prob
        self.max_global_stretch = max_global_stretch
        self.max_local_stretch = max_local_stretch
        self.max_local_regions = max_local_regions
        
    def _global_stretch(self, spec):
        """Apply global stretching in time or frequency dimension"""
        # Get current dimensions
        _, h, w = spec.shape
        
        # Randomly choose to stretch height (frequency) or width (time) or both
        stretch_dim = random.randint(0, 2)  # 0: freq, 1: time, 2: both
        
        # Calculate new dimensions with random stretching factor
        new_h = h
        new_w = w
        
        if stretch_dim in [0, 2]:  # Stretch frequency
            stretch_factor = 1.0 + random.uniform(-self.max_global_stretch, self.max_global_stretch)
            new_h = max(int(h * stretch_factor), 1)
            
        if stretch_dim in [1, 2]:  # Stretch time
            stretch_factor = 1.0 + random.uniform(-self.max_global_stretch, self.max_global_stretch)
            new_w = max(int(w * stretch_factor), 1)
        
        # Apply resize and then resize back to original
        stretched = F.resize(spec, [new_h, new_w], antialias=True)
        return F.resize(stretched, [h, w], antialias=True)
    
    def _local_stretch(self, spec):
        """Apply local stretching to random regions"""
        _, h, w = spec.shape
        spec_modified = spec.clone()
        
        # Apply random number of local stretches
        num_regions = random.randint(1, self.max_local_regions)
        
        for _ in range(num_regions):
            # Decide whether to stretch in frequency or time direction
            is_freq_stretch = random.random() < 0.5
            
            # Define local region
            if is_freq_stretch:
                # Frequency stretch (horizontal region)
                region_h = random.randint(max(1, int(h * 0.1)), max(2, int(h * 0.5)))
                start_h = random.randint(0, h - region_h)
                
                # Extract region
                region = spec[:, start_h:start_h+region_h, :]
                
                # Stretch factor
                stretch_factor = 1.0 + random.uniform(-self.max_local_stretch, self.max_local_stretch)
                new_h = max(int(region_h * stretch_factor), 1)
                
                # Stretch and resize back
                stretched = F.resize(region, [new_h, w], antialias=True)
                stretched = F.resize(stretched, [region_h, w], antialias=True)
                
                # Put back
                spec_modified[:, start_h:start_h+region_h, :] = stretched
            else:
                # Time stretch (vertical region)
                region_w = random.randint(max(1, int(w * 0.1)), max(2, int(w * 0.5)))
                start_w = random.randint(0, w - region_w)
                
                # Extract region
                region = spec[:, :, start_w:start_w+region_w]
                
                # Stretch factor
                stretch_factor = 1.0 + random.uniform(-self.max_local_stretch, self.max_local_stretch)
                new_w = max(int(region_w * stretch_factor), 1)
                
                # Stretch and resize back
                stretched = F.resize(region, [h, new_w], antialias=True)
                stretched = F.resize(stretched, [h, region_w], antialias=True)
                
                # Put back
                spec_modified[:, :, start_w:start_w+region_w] = stretched
                
        return spec_modified
    
    def forward(self, spec):
        # Apply global stretching with probability
        if random.random() < self.global_stretch_prob:
            spec = self._global_stretch(spec)
            
        # Apply local stretching with probability
        if random.random() < self.local_stretch_prob:
            spec = self._local_stretch(spec)
            
        return spec

class AddGaussianNoise(torch.nn.Module):
    """Add Gaussian noise to spectrogram"""
    def __init__(self, mean=0., std=0.01):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, spec):
        noise = torch.randn_like(spec) * self.std + self.mean
        return spec + noise

class TimeShift(torch.nn.Module):
    """Shift spectrogram in time dimension"""
    def __init__(self, max_shift_pct=0.1):
        super().__init__()
        self.max_shift_pct = max_shift_pct
        
    def forward(self, spec):
        _, _, width = spec.shape
        max_shift = int(width * self.max_shift_pct)
        if max_shift < 1:
            return spec
            
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return spec
            
        # Shift along time dimension (dim=2)
        return torch.roll(spec, shifts=shift, dims=2)

class FrequencyShift(torch.nn.Module):
    """Shift spectrogram in frequency dimension (within limit)"""
    def __init__(self, max_shift_pct=0.05):
        super().__init__()
        self.max_shift_pct = max_shift_pct
        
    def forward(self, spec):
        _, height, _ = spec.shape
        max_shift = int(height * self.max_shift_pct)
        if max_shift < 1:
            return spec
            
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return spec
            
        # Shift along frequency dimension (dim=1)
        return torch.roll(spec, shifts=shift, dims=1)


class AudioToSpec:
    def __init__(self, 
                 melSpecParams,
                 top_db=80,
                 image_size=None,
                 delta_stack = False
                 ):
        self.melSpec_transform = torchaudio.transforms.MelSpectrogram(**melSpecParams)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)


        if image_size is None:
            # freq_mask_param=15
            # time_mask_param=35
            freq_mask_param=18
            time_mask_param=50
        else: 
            freq_mask_param = int(image_size[0] * 0.04)
            time_mask_param = int(image_size[1] * 0.1)

        self.spec_aug = SpecAugment(freq_mask_param=freq_mask_param, time_mask_param=time_mask_param)
        self.stretch_aug = LocalGlobalStretch()
        self.top_db = top_db
        self.delta_stack = delta_stack

        print ("Top DB", top_db)

        if self.delta_stack:
            print ("Apply Delta Stack")


        if image_size is not None:
            print ("Using Image Size", image_size)
            self.resize_transforms = v2.Compose([
                v2.Resize(size=image_size),
            ])
        else:
            print ("No Resizing")
            self.resize_transforms = None

        self.train_transforms = v2.Compose([
            TimeShift(max_shift_pct=0.15),
            FrequencyShift(max_shift_pct=0.05),
            AddGaussianNoise(std=0.01),
            v2.RandomErasing(p=0.7, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        ])


    def __call__(self, audio, train): #Take Audio Tensor as input (1 x Audio)
        spec = self.db_transform(self.melSpec_transform(audio)) #1 x Mel_H x Mel_W
        spec = self.normalize_melspec(spec) #1 x Mel_H x Mel_W

        if self.delta_stack:
            delta_1 = compute_deltas(spec)
            delta_2 = compute_deltas(delta_1)
            spec = torch.cat([spec, delta_1, delta_2], dim=0)  # shape: (3, Mel_H, Mel_W)

        if train:
            spec = self.spec_aug(spec)
            spec = self.stretch_aug(spec)

        if not self.delta_stack:
            spec = spec.expand(3, -1, -1)

        if self.resize_transforms is not None:
            spec = self.resize_transforms(spec)
        if train:
            spec = self.train_transforms(spec)

        return spec
    
    def normalize_melspec(self, X):
        return torch.clamp((X + self.top_db) / self.top_db, 0.0, 1.0)




class TimmCNN(torch.nn.Module):
    def __init__(self, backbone, pretrained, num_classes):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.backbone(x)
        return x



import torch
import torch.nn as nn
import timm

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batch Normalization layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class AttBlockV2(nn.Module):
    """Attention block for SED tasks."""
    def __init__(self, in_features, out_features, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (batch_size, channels, time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)

        if self.activation == "sigmoid":
            eps = 1e-6
            x = torch.clamp(x, eps, 1 - eps)
        return x

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)

def interpolate(x, ratio):
    """Interpolate data in time domain."""
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames."""
    output = torch.nn.functional.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)
    return output

class SEDModel(nn.Module):
    def __init__(self, 
                 model_name, 
                 num_classes, 
                 n_mels,
                 in_chans=1, 
                 pretrained=True, 
                 drop_path_rate=0.2, 
                 drop_rate=0.5,
                 freq_wise = False
                 ):
        super().__init__()
        
        self.freq_wise = freq_wise
        self.num_classes = num_classes
        self.in_chans = in_chans
        
        # Batch normalization layer for mel spectrogram input
        self.bn0 = nn.BatchNorm2d(n_mels)
        
        # Create backbone model using timm
        base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
        
        # Extract all layers except the classification head
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        
        # Determine the feature dimension based on model architecture
        if "efficientnet" in model_name:
            in_features = base_model.classifier.in_features
        elif "eca" in model_name:
            in_features = base_model.head.fc.in_features
        elif "res" in model_name:
            in_features = base_model.fc.in_features
        else:
            # Default fallback for other architectures
            in_features = base_model.num_features
            
        # Add fully connected layer
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        
        # Add attention block for SED
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        
        
        # Initialize weights
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, time, freq)
                or (batch_size, channels, freq, time) depending on your data
        
        Returns:
            clipwise_output: Predicted probabilities for each class (batch_size, num_classes)
        """
        # Expected input shape: (batch_size, channels, time, freq)
        # Transpose to (batch_size, channels, freq, time) if needed


        if not self.freq_wise:
            x = x.permute(0, 1, 3, 2)
                
        frames_num = x.shape[2]
        
        # Apply batch normalization
        x = x.transpose(1, 3).contiguous()
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # Re-transpose for the CNN
        x = x.transpose(2, 3).contiguous()
        # Now shape: (batch_size, channels, freq, time)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Average pooling over frequency dimension
        x = torch.mean(x, dim=2)
        
        # Apply channel smoothing
        x1 = torch.nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = torch.nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

        # Apply FC layer
        x = x.transpose(1, 2).contiguous()
        x = torch.nn.functional.relu_(self.fc1(x))
        x = x.transpose(1, 2).contiguous()
        
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

        # # Get clipwise output through attention mechanism
        clipwise_output = self.att_block(x)

        clipwise_output = torch.logit(clipwise_output)
        
        return clipwise_output


def merge_close_segments(segments, max_gap=0.5):
    if not segments:
        return []

    merged = [segments[0]]

    for current in segments[1:]:
        last = merged[-1]
        if current['start'] - last['end'] < max_gap:
            # Merge with the previous segment
            last['end'] = current['end']
        else:
            merged.append(current)

    return merged



def get_human_non_human(waves, timestamp, max_gap=1.0, sample_rate=32000):

    if len(timestamp) == 0:
        return None, None

    speaking_segments = merge_close_segments(timestamp, max_gap=max_gap)

    speaking_indices = [(int(start * sample_rate), int(end * sample_rate)) for start, end in [(s['start'], s['end']) for s in speaking_segments]]
    human_audio_parts = [waves[start:end] for start, end in speaking_indices]
    non_human_audio_parts = []

    prev_end = 0
    for start, end in speaking_indices:
        if start > prev_end:
            non_human_audio_parts.append(waves[prev_end:start])
        prev_end = end

    if prev_end < len(waves):
        non_human_audio_parts.append(waves[prev_end:])

    human_audio = np.concatenate(human_audio_parts) if human_audio_parts else np.array([], dtype=waves.dtype)
    non_human_audio = np.concatenate(non_human_audio_parts) if non_human_audio_parts else np.array([], dtype=waves.dtype)
    return human_audio, non_human_audio




def unique_class_filtered_timestamps():
    human_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    df = pd.read_csv("../../../data/birdclef-2025/train.csv")
    unique, counts = np.unique(df['primary_label'], return_counts=True)
    unique_low = unique[counts < 30]

    xdf = df[df['primary_label'].isin(unique_low)]
    timestamps = {}
    for i, row in tqdm(xdf.iterrows(), total=xdf.shape[0]):
        wav = read_audio(f"../../../data/birdclef-2025/train_audio/{row['filename']}")
        speech_timestamps = get_speech_timestamps(
            wav,
            human_model,
            return_seconds=True,
            )
        if len(speech_timestamps) > 0:
            timestamps[row['filename']] = speech_timestamps
    return timestamps



def csa_filtered_timestamps():
    human_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    df = pd.read_csv("../../../data/birdclef-2025/train.csv")
    v = df['filename'].str.split("/", expand=True)
    v = v[v.columns[1]].values
    v = [each.startswith('CSA') for each in v]
    xdf = df[v]

    timestamps = {}
    for i, row in tqdm(xdf.iterrows(), total=xdf.shape[0]):
        wav = read_audio(f"../../../data/birdclef-2025/train_audio/{row['filename']}")
        speech_timestamps = get_speech_timestamps(
            wav,
            human_model,
            return_seconds=True,
            )
        if len(speech_timestamps) > 0:
            timestamps[row['filename']] = speech_timestamps
    return timestamps



class RawBirdClefMelSpec(torch.utils.data.Dataset):
    """ 
    This class is used to load the raw signal data from the data directory.
    """
    def __init__(self,
                 df,
                 mel_spec_params,
                 top_db,
                 num_classes = -1,
                 bird2id = None,
                 is_train=False,
                 wave_transforms=None,
                 use_secondary_labels=False,
                 image_size = None,
                 delta_stack = False,
                 human_filter_type = 'CSA', # Possible Options 'CSA', 'UNIQUE', None
                 human_filter_chance_threshold = 0.5,
                 add_human_noise = False,
                 max_trick = False, 
                 max_trick_factor = 1,
                 base_duration = 5,
                 combined_sampling = True,
                 random_shuffle = True
                 ):
        self.df = df
        self.human_filter_chance_threshold = human_filter_chance_threshold
        self.mel_spec_params = mel_spec_params
        self.bird2id = bird2id
        self.is_train = is_train
        self.num_classes = num_classes
        self.wave_transforms = wave_transforms
        self.use_secondary_labels = use_secondary_labels
        self.add_human_noise = add_human_noise
        self.max_trick = max_trick
        self.max_trick_factor = max_trick_factor
        self.base_duration = base_duration
        self.combined_sampling = combined_sampling
        self.random_shuffle = random_shuffle
        self.audio2spec = AudioToSpec(melSpecParams=mel_spec_params, top_db=top_db, image_size=image_size, delta_stack=delta_stack)
        if self.is_train:
            if human_filter_type == 'CSA':
                if os.path.exists('outputs/csa_timestamps.json'):
                    with open('outputs/csa_timestamps.json', 'r') as json_file:
                        self.timestamps = json.load(json_file)
                else:
                    self.timestamps = csa_filtered_timestamps()
                    with open('outputs/csa_timestamps.json', 'w') as json_file:
                        json.dump(self.timestamps, json_file)

                
            elif human_filter_type == 'UNIQUE':
                self.timestamps = unique_class_filtered_timestamps()
            else:
                print ("NO HUMAN AUDIO Filter Applied")
                self.timestamps = {}
        else:
            self.timestamps = {}
            


    def __len__(self):
        return len(self.df)
    
    def max_norm_wave(self, wave):
        max_v = np.max(np.abs(wave))
        if max_v > 1:
            wave = wave/max_v
        return wave

    def apply_max_stack(self, wave):
        if not self.max_trick:
            return wave
        
        new_wave = []
        for i in range(0, self.max_trick_factor):
            start = i * self.base_duration * 32000
            end = (i+1) * self.base_duration * 32000
            _wave = wave[start:end]
            new_wave.append(_wave)
        new_wave = np.stack(new_wave)
        return new_wave
    

    def combined_pick_rms_sample(self, waves, duration_samples):
        if len(waves) <= duration_samples:
            # Pad if too short
            pad_length = duration_samples - len(waves)
            waves = np.pad(waves, (0, pad_length), mode='constant')
            return waves

        stride = 32000
        max_rms = 0
        max_rms_start = 0

        for start in range(0, len(waves) - duration_samples + 1, stride):
            window = waves[start:start + duration_samples]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > max_rms:
                max_rms = rms
                max_rms_start = start

        wave = waves[max_rms_start:max_rms_start + duration_samples]
        return wave
    
    def ___pick_rms_sample(self, waves, duration_samples):
        if len(waves) <= duration_samples:
            # Pad if too short
            pad_length = duration_samples - len(waves)
            waves = np.pad(waves, (0, pad_length), mode='constant')
            return waves, np.array([])

        stride = 32000
        max_rms = 0
        max_rms_start = 0

        for start in range(0, len(waves) - duration_samples + 1, stride):
            window = waves[start:start + duration_samples]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > max_rms:
                max_rms = rms
                max_rms_start = start

        wave = waves[max_rms_start:max_rms_start + duration_samples]

        start_base_wave = waves[0:max_rms_start]
        next_base_wave = waves[max_rms_start + duration_samples:]
        base_wave = np.concatenate([start_base_wave, next_base_wave], axis=0)
        
        return wave, base_wave
    
    def chunked_pick_rms_sample(self, waves, duration_samples):
        total_waves = duration_samples // (self.base_duration * 32000)
        waves_ = []
        for i in range(total_waves):
            wave, waves = self.___pick_rms_sample(waves, self.base_duration * 32000)
            waves_.append(wave)
        waves_ = np.concatenate(waves_, axis=0).astype(np.float32)
        return waves_


    def pick_rms_sample(self, waves, duration_samples):
        if self.combined_sampling:
            return self.combined_pick_rms_sample(waves, duration_samples)
        return self.chunked_pick_rms_sample(waves, duration_samples)

    
    def load_filtered_sample(self, human_audio, non_human_audio):
        
        duration = self.base_duration
        if self.max_trick:
            duration = duration * self.max_trick_factor
            human_duration = duration // 3
            non_human_duration = duration - human_duration
        else:
            if duration == 5:
                human_duration = 2
                non_human_duration = 3
            elif duration == 15:
                human_duration = 5
                non_human_duration = 10


        target_length = 32000 * human_duration
        human_audio_sample = self.pick_rms_sample(human_audio, target_length)
        target_length = 32000 * non_human_duration
        non_human_audio_sample = self.pick_rms_sample(non_human_audio, target_length)

        if random.random() > 0.5:
            wave = np.concatenate([human_audio_sample, non_human_audio_sample], axis=0)
        else:
            wave = np.concatenate([non_human_audio_sample, human_audio_sample], axis=0)

        wave = self.apply_max_stack(wave)

        return wave
    
    def load_raw_sample(self, waves, pick_random):
        target_length = 32000 * self.base_duration

        if self.max_trick:
            target_length = target_length * self.max_trick_factor
        
        if pick_random:
            wave = self.pick_rms_sample(waves, target_length)
        else:
            pad_size = target_length - (waves.shape[0] % target_length)
            if pad_size != target_length:  # Avoid adding unnecessary padding if already aligned
                waves = np.pad(waves, (0, pad_size), mode='constant', constant_values=0)
            waves = np.reshape(waves, newshape=[-1, target_length])
            wave = waves[0]
        wave = self.apply_max_stack(wave)
        return wave

    def convert_to_spec(self, wave):
        if not self.max_trick:
            return self.audio2spec(wave.unsqueeze(0), train=self.is_train)
        
        specs = []
        for i in range(0, len(wave)):
            spec = self.audio2spec(wave[i].unsqueeze(0), train=self.is_train)
            specs.append(spec)
        specs = torch.stack(specs)
        return specs


    def __getitem__(self, idx):
        fn = self.df["path"].iloc[idx]
        primary_label = self.df["primary_label"].iloc[idx]
        secondary_labels = self.df["secondary_labels"].iloc[idx]
        waves, _ = sf.read(fn)
        if waves.ndim!=1:
            waves = waves.mean(1)
        waves = waves.astype(np.float32)

        duration = self.base_duration
        timstamp = self.timestamps.get(self.df["filename"].iloc[idx], {})
        human_audio, non_human_audio = get_human_non_human(waves, timstamp, max_gap=1.0, sample_rate=32000)
        if (len(timstamp) > 0) and (self.is_train == True) and (waves.shape[0] > 32000*duration) and (human_audio.shape[0] > (waves.shape[0] * 0.5)):
            if random.random() > 0.5 and self.human_filter_chance_threshold:
                wave = self.load_filtered_sample(human_audio=human_audio, non_human_audio=non_human_audio)
            else:
                wave = self.load_raw_sample(non_human_audio, pick_random=True)
        
        else:
            
            add_human_noise = False
            human_audio = None
            actual_waves = waves
            if self.is_train and self.add_human_noise:
                if random.random() > 0.2:
                    timstamp_keys = list(self.timestamps.keys())
                    timstamp_key = random.choice(timstamp_keys)
                    timstamp = self.timestamps.get(timstamp_key, {})
                    if len(timstamp) > 0:
                        fn = f"../../../data/birdclef-2025/train_audio/" + timstamp_key
                        noise_waves, _ = sf.read(fn)
                        if noise_waves.ndim!=1:
                            noise_waves = noise_waves.mean(1)
                        human_audio, _ = get_human_non_human(noise_waves, timstamp, max_gap=1.0, sample_rate=32000)
                        if human_audio.shape[0] > 0:
                            add_human_noise = True

            if add_human_noise:
                wave = self.load_filtered_sample(human_audio=human_audio, non_human_audio=actual_waves)
            else:
                wave = self.load_raw_sample(actual_waves, pick_random=self.is_train)

        if self.max_trick:
            for i in range(len(wave)):
                wave[i] = self.max_norm_wave(wave[i])
        else:
            wave = self.max_norm_wave(wave)
        
        if self.wave_transforms:
            wave = self.wave_transforms(wave, sr=32000)

        if self.use_secondary_labels:
            target = np.zeros(self.num_classes, dtype=np.float32)
            primary_label = self.bird2id[primary_label]
            target[primary_label] = 1.0
            
            secondary_labels = eval(secondary_labels)
            secondary_labels = [each for each in secondary_labels if each != '']
            coeff = len(secondary_labels)
            for label in secondary_labels:
                label = self.bird2id[label]
                target[label] = 1.0 #1.0 / coeff
        else:
            target = np.zeros(self.num_classes, dtype=np.float32)
            if primary_label != 'nocall':
                primary_label = self.bird2id[primary_label]
                target[primary_label] = 1.0
        if self.is_train and self.max_trick and self.random_shuffle:
            idx__ = np.array(list(range(0, self.max_trick_factor)))
            np.random.shuffle(idx__)
            wave = wave[idx__]

        wave = torch.from_numpy(wave)
        spec = self.convert_to_spec(wave)
        return {
            "spec": spec,
            "target": torch.tensor(target),
        }



class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss


def train_epoch(epoch, model, train_loader, optimizer, device, criterion, accumulation_steps=1, use_mixed_precision=False, scaler=None, max_trick = False, use_maximum_mix=False, use_mix_prob=0.5):
    """
    Unified training function supporting regular training, gradient accumulation, and mixed precision.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: The optimizer
        device: Device to run training on (cuda/cpu)
        criterion: Loss function
        accumulation_steps: Number of steps to accumulate gradients (default=1 means no accumulation)
        use_mixed_precision: Whether to use mixed precision training (default=False)
    
    Returns:
        List of training losses for the epoch
    """
    model.train()
    train_loss = []
    optimizer.zero_grad()
    if not use_mixed_precision:
        scaler = None
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for step, batch in bar:
        

        spec = batch['spec']
        target = batch['target']

        spec, target = mixup(spec, target, alpha=1.0)

        if max_trick:
            BS, K, C, H, W = spec.shape
            spec = spec.view(BS * K, C, H, W)

        spec = spec.to(device)
        target = target.to(device)
        
        # Forward pass with or without mixed precision
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                predictions = model(spec)
                if max_trick:
                    predictions = predictions.view(BS, K, -1)
                    predictions = torch.max(predictions, dim=1).values

                if use_maximum_mix and random.random() < use_mix_prob:
                    indices = torch.randperm(predictions.size(0)).to(device)
                    predictions = torch.maximum(predictions, predictions[indices])
                    target = torch.maximum(target, target[indices])

                loss = criterion(predictions, target)
                # Scale loss according to accumulation steps
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
        else:
            predictions = model(spec)
            if max_trick:
                predictions = predictions.view(BS, K, -1)
                predictions = torch.max(predictions, dim=1).values

            if use_maximum_mix and random.random() < use_mix_prob:
                indices = torch.randperm(predictions.size(0)).to(device)
                predictions = torch.maximum(predictions, predictions[indices])
                target = torch.maximum(target, target[indices])
            
            loss = criterion(predictions, target)
            # Scale loss according to accumulation steps
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
        
        # Backward pass with or without mixed precision
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Store unscaled loss for logging
        loss_value = loss.detach().cpu().item()
        if accumulation_steps > 1:
            loss_value *= accumulation_steps  # Un-normalize for logging
        train_loss.append(loss_value)
        
        # Update weights if needed
        update_weights = (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader)
        
        if update_weights:
            if use_mixed_precision:
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Update progress bar
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_value, smooth_loss))
    
    return train_loss


def val_epoch(model, valid_loader, device, target_columns, unseen_index=None, criterion=None, exclude_nocall = False, max_trick = False):
    if unseen_index is not None:
        scores = MetricMeter(indices_ignore=unseen_index)
        
    loss_function = criterion
    model.eval()
    val_loss = []
    PREDS = []
    TRUES = []
    bar = tqdm(valid_loader)
    with torch.no_grad():
        for batch in bar:
            target = batch['target'].to(device)
            spec = batch['spec'].to(device)
            
            if max_trick:
                BS, K, C, H, W = spec.shape
                spec = spec.view(BS * K, C, H, W)

            
            predictions = model(spec)
            if max_trick:
                predictions = predictions.view(BS, K, -1)
                predictions = torch.max(predictions, dim=1).values

            if exclude_nocall:
                target = target[:, :-1]
                predictions = predictions[:, :-1]

            loss = loss_function(predictions, target)
            
            val_loss.append(loss.detach().cpu().numpy())
            PREDS.append(predictions.detach().cpu())
            
            target = torch.round(target).long()

            TRUES.append(target.detach().cpu())
            

            if unseen_index is not None:
                scores.update(target, predictions)
            
            loss_np = loss.detach().cpu().item()
            bar.set_description('loss: %.5f' % (loss_np))

            
    val_loss = np.mean(val_loss)
    P = np.concatenate(PREDS, axis=0)
    T = np.concatenate(TRUES, axis=0)
    metrics = calculate_competition_metrics(T, P, target_columns)
    if unseen_index is not None:
        metrics['score'] = scores.avg
    log = metrics_to_string(metrics, "val")
    return val_loss, metrics, log


# better_mel_cfg = {
#     # keep your acquisition settings
#     "sample_rate": 32_000,          # 32 kHz
#     # frequency coverage & resolution
#     "f_min":      50,               # ignore infrasonic rumble
#     "f_max":      16_000,           # Nyquist
#     "n_fft":      4096,             # ≈128 ms window  →  fine 7.8 Hz bins
#     "hop_length": 320,              # 10 ms stride   →  tight time cues
#     # mel projection
#     "n_mels":     448,              # dense  →  good insect harmonics
#     "mel_scale":  "htk",            # slightly more linear high‑freq spacing
#     # dynamic‑range mapping
#     "power":      2.0,              # power‑spectrogram before log
#     "normalized": True,
# }

# tf_efficientnet_b0.aa_in1k
# tf_efficientnet_b0.ap_in1k
# tf_efficientnet_b0.in1k
# tf_efficientnet_b0.ns_jft_in1k


def prepare_mel_spec_params(image_size, duration = 5, get_default = False):
    if get_default:
        return {
            "sample_rate": 32000,
            "n_mels": 384,
            "f_min": 0,
            "f_max": 16000,
            "n_fft": 3072,
            "hop_length": 420,
            "normalized": True,
        }

    new_nfft = 4096 #int(image_size[0] * 8)
    new_mel_spec_params = {
        "sample_rate": 32000,
        "n_mels": image_size[0],
        "f_min": 50,
        "f_max": 16000,
        "n_fft": new_nfft,
        "normalized": True,
    }
    new_mel_spec_params['hop_length'] = int((duration * 32000 - new_mel_spec_params['n_fft'] + new_mel_spec_params['n_fft']) / (image_size[1])) + 1
    print (new_mel_spec_params)
    return new_mel_spec_params



RAW_TRAIN_CONFIG = {
    "seed": 42,
    "experiment_name": "stage0/b0_cnn_384_3_no_shuffle",
    "batch_size": 32,
    "accumulation_steps" : 1,
    "mixed_precision" : False,
    "num_workers": 12,
    "warmp_epoch": 3,
    "n_epochs": 15,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "minimum_learning_rate" : 1e-7,
    "fold_n": 5,
    "model_name": "tf_efficientnet_b0_ns",
    "drop_rate" : 0.5,
    "drop_path_rate" : 0.2,
    "pretrained": True,
    "base_path": "../../../data/birdclef-2025",
    "start_soup_epoch" : 12,
    "end_soup_epoch" : 15,
    "train_use_secondary_labels" : True,
    "val_use_secondary_labels" : True,
    "human_filter_type" : "CSA", #CSA, UNIQUE, NONE``   
    "human_filter_chance_threshold" : True,
    "add_human_noise" : False,
    "add_nocall" : False,
    "cap_nocall" : -1,
    'fold' : -1,
    "image_size" : (384, 384),
    "mel_spec_params" : prepare_mel_spec_params(image_size=(384, 381), duration=5, get_default=True),
    "delta_stack" : False,
    "multi_gpu" : True,
    "device": torch.device("cuda:1"),
    "max_trick" : True,
    "max_trick_factor" : 3,
    "use_maximum_mix" : False,
    "use_mix_prob" : 0.5,
    "base_duration" : 5,
    "save_unlabelled_predictions" : False,
    "top_db" : 80,
    "architecture_type" : "CNN", #SED or CNN
    "freq_wise_attn" : False,
    "combined_sampling" : True,
    "random_shuffle" : False,
}

mel_spec_params = RAW_TRAIN_CONFIG['mel_spec_params']
os.makedirs("outputs", exist_ok=True)
os.makedirs(RAW_TRAIN_CONFIG['experiment_name'], exist_ok=True)
os.makedirs(os.path.join(RAW_TRAIN_CONFIG['experiment_name'], "models"), exist_ok=True)


#### Checking Output Size of MelSpec
audio = torch.randn(1, 32000*5)
spec = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=RAW_TRAIN_CONFIG['top_db'])(torchaudio.transforms.MelSpectrogram(**mel_spec_params)(audio))
print (spec.size())



log_file = os.path.join(RAW_TRAIN_CONFIG['experiment_name'], f"train_log_fold_{RAW_TRAIN_CONFIG['fold']}.txt")
print(mel_spec_params)
with open(log_file, 'a') as f:
    f.write(str(mel_spec_params) + '\n')
print (RAW_TRAIN_CONFIG)
with open(log_file, 'a') as f:
    f.write(str(RAW_TRAIN_CONFIG) + '\n')



set_seed(RAW_TRAIN_CONFIG['seed'])
sub = pd.read_csv(f"{RAW_TRAIN_CONFIG['base_path']}/sample_submission.csv")
target_columns = sub.columns.tolist()[1:]
num_classes = len(target_columns)
bird2id = {b: i for i, b in enumerate(target_columns)}
df = pd.read_csv(f"{RAW_TRAIN_CONFIG['base_path']}/train.csv")
df["path"] = f"{RAW_TRAIN_CONFIG['base_path']}/train_audio/" + df["filename"]
df['filename_base'] = df['filename'].map(lambda x: x.split('/')[-1])
df = df.drop_duplicates(subset=['filename_base']).reset_index(drop=True)


####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################


""" 
Insecta: (0.994375, 896.57509375, 113.75990141129033, 99.41890625, 114.62581564261338)
Aves: (0.548, 1774.392, 35.03129723345455, 21.0024375, 49.89740423615133)
Mammalia: (1.018, 218.784, 33.827185393258425, 21.762921875, 35.44624156288265)
Amphibia: (0.54459375, 389.7730625, 30.358085173670666, 16.55584375, 43.96448730451026)
min(), _tmp.max(), _tmp.mean(), _tmp.median(), _tmp.std()
"""




gkf = StratifiedKFold(n_splits=RAW_TRAIN_CONFIG['fold_n'], shuffle=True, random_state=RAW_TRAIN_CONFIG['seed'])
df['fold'] = -1
for ifold, (train_idx, val_idx) in enumerate(gkf.split(df, y=df.primary_label.tolist())):
    df.loc[val_idx, 'fold'] = ifold

#Unseen Birds in Folds
bird_names = set(df.primary_label)
unseen_birds = []
for fold in range(RAW_TRAIN_CONFIG['fold_n']):
    bird_names_fold = set(df[df['fold']==fold].primary_label)
    unseen_bird = list(bird_names - bird_names_fold)
    print(f'fold{fold}: Unseen Birds - ', len(unseen_bird))
    unseen_birds.extend(unseen_bird)
unseen_birds = np.unique(unseen_birds)
print(f'Overall Unseen Birds - ', len(unseen_birds))
df.loc[df['primary_label'].isin(unseen_birds), 'fold'] = -1

unseen_index = []
for name in unseen_birds:
    unseen_index.append(target_columns.index(name))


####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

if RAW_TRAIN_CONFIG['fold'] != -1:
    trn_df = df[df['fold'] != RAW_TRAIN_CONFIG['fold']].reset_index(drop=True)
    val_df = df[df['fold'] == RAW_TRAIN_CONFIG['fold']].reset_index(drop=True)
else:
    trn_df = df.reset_index(drop=True)
    val_df = df[df['fold'] == 0].reset_index(drop=True)

# trn_df = downsample_data(trn_df, thr=500, seed=RAW_TRAIN_CONFIG['seed'])
trn_df = upsample_data(trn_df, thr=15, seed=RAW_TRAIN_CONFIG['seed'])


if RAW_TRAIN_CONFIG["add_nocall"]:
    nocall_path = os.path.join(RAW_TRAIN_CONFIG["base_path"], "ff1010bird_nocall")
    nocall_df = pd.read_csv(os.path.join(nocall_path, "ff1010bird_metadata_v1.csv"))
    nocall_df["filename"] = "nocall/" + nocall_df["filename"]
    nocall_df["path"] = nocall_path + "/" + nocall_df["filename"]
    nocall_df.pop("filename")
    nocall_df.pop("length")
    if RAW_TRAIN_CONFIG["cap_nocall"] != -1:
        nocall_df = nocall_df.sample(n=RAW_TRAIN_CONFIG["cap_nocall"])
    nocall_df_aligned = nocall_df.reindex(columns=trn_df.columns)
    print (trn_df.shape)
    trn_df = pd.concat([trn_df, nocall_df_aligned], ignore_index=True)
    print (trn_df.shape)
    bird2id["nocall"] = 206
    num_classes = num_classes + 1







trn_dataset = RawBirdClefMelSpec(df=trn_df.reset_index(drop=True), top_db = RAW_TRAIN_CONFIG['top_db'], delta_stack = RAW_TRAIN_CONFIG['delta_stack'],
                                bird2id=bird2id, is_train=True, num_classes=num_classes,
                                wave_transforms=None, use_secondary_labels=RAW_TRAIN_CONFIG['train_use_secondary_labels'],
                                mel_spec_params=mel_spec_params, image_size=RAW_TRAIN_CONFIG['image_size'], human_filter_type=RAW_TRAIN_CONFIG['human_filter_type'],
                                human_filter_chance_threshold = RAW_TRAIN_CONFIG['human_filter_chance_threshold'], add_human_noise = RAW_TRAIN_CONFIG['add_human_noise'],
                                max_trick = RAW_TRAIN_CONFIG['max_trick'], max_trick_factor = RAW_TRAIN_CONFIG["max_trick_factor"],
                                base_duration = RAW_TRAIN_CONFIG["base_duration"], combined_sampling = RAW_TRAIN_CONFIG['combined_sampling'],
                                random_shuffle = RAW_TRAIN_CONFIG['random_shuffle']
                                )

val_dataset = RawBirdClefMelSpec(df=val_df.reset_index(drop=True), top_db = RAW_TRAIN_CONFIG['top_db'], delta_stack = RAW_TRAIN_CONFIG['delta_stack'],
                                bird2id=bird2id, is_train=False, num_classes=num_classes,
                                wave_transforms=None, use_secondary_labels=RAW_TRAIN_CONFIG['val_use_secondary_labels'],
                                mel_spec_params=mel_spec_params, image_size=RAW_TRAIN_CONFIG['image_size'], human_filter_type='NONE',
                                human_filter_chance_threshold = RAW_TRAIN_CONFIG['human_filter_chance_threshold'], add_human_noise = RAW_TRAIN_CONFIG['add_human_noise'],
                                max_trick = RAW_TRAIN_CONFIG['max_trick'], max_trick_factor = RAW_TRAIN_CONFIG["max_trick_factor"],
                                base_duration = RAW_TRAIN_CONFIG["base_duration"], combined_sampling = RAW_TRAIN_CONFIG['combined_sampling'],
                                random_shuffle = False
                                )

train_loader = torch.utils.data.DataLoader(trn_dataset, shuffle=True, batch_size=RAW_TRAIN_CONFIG['batch_size'], 
                                        drop_last=True, num_workers=RAW_TRAIN_CONFIG['num_workers'], pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=RAW_TRAIN_CONFIG['batch_size'], 
                                            drop_last=False, num_workers=RAW_TRAIN_CONFIG['num_workers'], pin_memory=True)


if RAW_TRAIN_CONFIG['architecture_type'] == 'SED':

    if RAW_TRAIN_CONFIG['freq_wise_attn']:
        attn_dim = RAW_TRAIN_CONFIG['image_size'][1]
    else:
        attn_dim = RAW_TRAIN_CONFIG['image_size'][0]
    
    model = SEDModel(RAW_TRAIN_CONFIG['model_name'], 
            num_classes, 
            attn_dim,
            in_chans=3, 
            pretrained=True, 
            drop_rate=RAW_TRAIN_CONFIG['drop_rate'],
            drop_path_rate=RAW_TRAIN_CONFIG['drop_path_rate'],
            freq_wise=RAW_TRAIN_CONFIG['freq_wise_attn']
            )
    
else:
    model = TimmCNN(backbone=RAW_TRAIN_CONFIG['model_name'], pretrained=True, num_classes=num_classes)

if RAW_TRAIN_CONFIG['multi_gpu']:
    model = torch.nn.DataParallel(model, device_ids=[1, 0])

model = model.to(RAW_TRAIN_CONFIG['device'])


optimizer = AdamW(model.parameters(), 
                    lr=RAW_TRAIN_CONFIG['learning_rate'], 
                    weight_decay=RAW_TRAIN_CONFIG['weight_decay'])

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RAW_TRAIN_CONFIG['n_epochs'], 
                                                                eta_min=RAW_TRAIN_CONFIG['minimum_learning_rate'])
scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=RAW_TRAIN_CONFIG['warmp_epoch'], 
                                            after_scheduler=scheduler_cosine)


criterion = FocalLossBCE()
model_soup = ModelSoup(model, save_dir=os.path.join(RAW_TRAIN_CONFIG['experiment_name'], "model_soup"), exclude_nocall=RAW_TRAIN_CONFIG["add_nocall"], max_trick=RAW_TRAIN_CONFIG['max_trick'])


scaler = GradScaler(RAW_TRAIN_CONFIG['device'])
best_val_loss = float('inf')
best_map_score = 0
for epoch in range(0, RAW_TRAIN_CONFIG['n_epochs']):
    scheduler_warmup.step(epoch)
    train_loss = train_epoch(epoch + 1, model, train_loader, optimizer, RAW_TRAIN_CONFIG['device'], criterion, accumulation_steps=RAW_TRAIN_CONFIG['accumulation_steps'], 
                            use_mixed_precision=RAW_TRAIN_CONFIG['mixed_precision'], scaler=scaler, max_trick=RAW_TRAIN_CONFIG['max_trick'],
                            use_maximum_mix = RAW_TRAIN_CONFIG['use_maximum_mix'], use_mix_prob=RAW_TRAIN_CONFIG['use_mix_prob'])
    
    val_loss, metrics, log = val_epoch(model, val_loader, RAW_TRAIN_CONFIG['device'], target_columns, unseen_index, criterion=criterion, exclude_nocall=RAW_TRAIN_CONFIG["add_nocall"], 
                                    max_trick=RAW_TRAIN_CONFIG['max_trick'])
    current_lr = optimizer.param_groups[0]['lr']
    log = time.ctime() + ' ' + f"Epoch {epoch}, LR: {current_lr:.6f}, Train Loss: {np.mean(train_loss):.4f}, Val Loss: {val_loss:.4f}, {log}"
    
    if RAW_TRAIN_CONFIG['start_soup_epoch'] <= epoch <= RAW_TRAIN_CONFIG['end_soup_epoch']:
        model_soup.add_model_if_improved(unseen_index, model, epoch, val_loader, RAW_TRAIN_CONFIG['device'], criterion=criterion)
    
    print(log)
    with open(log_file, 'a') as f:
        f.write(log + '\n')
        
    if metrics['score'] > best_map_score:
        best_map_score = metrics['score']
        torch.save(model.state_dict(), f"{RAW_TRAIN_CONFIG['experiment_name']}/models/best_score_model_{RAW_TRAIN_CONFIG['fold']}.pt")
        print(f"New best score model saved with score: {best_map_score:.4f}")
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{RAW_TRAIN_CONFIG['experiment_name']}/models/best_val_loss_model_{RAW_TRAIN_CONFIG['fold']}.pt")
        print(f"New best Val Loss model saved with Loss: {best_val_loss:.4f}")
torch.save(model.state_dict(), f"{RAW_TRAIN_CONFIG['experiment_name']}/models/final_model{RAW_TRAIN_CONFIG['n_epochs']}_{RAW_TRAIN_CONFIG['fold']}.pt")


if epoch > RAW_TRAIN_CONFIG['start_soup_epoch']:
    experiment_name = RAW_TRAIN_CONFIG['experiment_name'].split('/')[-1]
    soup_model_name = f"{experiment_name}_model_soup_final.pt"
    final_soup_model = model_soup.get_soup_model(soup_model_name=soup_model_name)
    soup_accuracy = model_soup.evaluate_model(unseen_index, final_soup_model, val_loader, criterion=criterion, device=RAW_TRAIN_CONFIG['device'])
    soup_accuracy = soup_accuracy['score']
    log = f"Final Model Soup Validation Score: {soup_accuracy:.4f}\nImprovement over initial model: {soup_accuracy - model_soup.initial_val_score['score']:.4f}"
    print(log)
    with open(log_file, 'a') as f:  
        f.write(log + '\n')


import timm
list_models = timm.list_models(pretrained=True)
for each in list_models:
    print (each)



















