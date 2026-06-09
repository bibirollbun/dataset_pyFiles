import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
import torch
from torch import Tensor
import gc
from typing import Any, List, Tuple, Union

import albumentations as A
import os
import librosa
import pickle
import timm
from tqdm import tqdm
import matplotlib.pyplot as pl

import mne

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.signal import butter, lfilter


!cp /kaggle/input/torchvideo/x3d.py .
!cp /kaggle/input/torchvideo/hgnet.py .
from x3d import create_x3d
from hgnet import hgnetv2_b5


config = {
    'batch_size': 32,
    'num_worker': 4,
    'data': '/kaggle/input/hms-harmful-brain-activity-classification/test.csv',
    'weights_x3d': '/kaggle/input/hms-x3d',
    'weights_doubleheadbutterfilter': '/kaggle/input/hms-doublehead-butter-filter',
    'flip': True
}

keys_to_update = [
    'weights_x3d',
    'weights_doubleheadbutterfilter',
]

for key in keys_to_update:
    config[key] = [os.path.join(config[key], fname) for fname in sorted(os.listdir(config[key]))]


class DataProcessor:
    """
    Iterator for processing brain activity data, supporting EEG, spectrogram,
    and mixed modalities. It handles data augmentation (e.g., flipping), filtering,
    and channel differencing.
    """

    def __init__(self,
         dataframe: pd.DataFrame,
         training_flag: bool = False,
         shuffle: bool = False,
         use_spec: bool = False,
         use_eeg: bool = False,
         use_mix: bool = False,
         lower_cut: float = 0,
         upper_cut: float = 20,
         flip: bool = False,
         use_mne_filter: bool = True,
         use_18_lead: bool = False) -> None:
        """
        Initialize the data iterator with configuration options.

        Args:
            dataframe (pd.DataFrame): Dataframe with metadata for each sample.
            training_flag (bool): Set to True if used for training.
            shuffle (bool): Set to True to shuffle the data.
            use_spec (bool): Use spectrogram data if True.
            use_eeg (bool): Use EEG data if True.
            use_mix (bool): Use both EEG and spectrogram data if True.
            lower_cut (float): Lower frequency cutoff for filtering.
            upper_cut (float): Upper frequency cutoff for filtering.
            flip (bool): If True, apply mirroring to the data.
            use_mne_filter (bool): If True, use MNE filtering; otherwise use a Butterworth filter.
            use_18_lead (bool): Whether to use all 18 EEG leads.
        """
        self.flip_eeg: bool = flip
        self.lower_cut: float = lower_cut
        self.upper_cut: float = upper_cut
        self.use_18_lead: bool = use_18_lead

        print(self.lower_cut, self.upper_cut, 'with mne filter:', use_mne_filter, 'use 18 lead:', use_18_lead)

        self.training_flag: bool = training_flag
        self.shuffle: bool = shuffle
        self.dataframe: pd.DataFrame = dataframe

        # Mapping of brain activity classes to integer labels
        activity_to_label = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
        self.target_mapping: dict = activity_to_label
        self.target_mapping_inv: dict = {label: activity for activity, label in activity_to_label.items()}

        # List of EEG channel names (including an EKG channel)
        self.eeg_channel_names: List[str] = [
            'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
            'Fz', 'Cz', 'Pz',
            'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG'
        ]

        # Define channel groups for differential computations
        self.left_lateral: List[str] = ['Fp1', 'F7', 'T3', 'T5', 'O1']
        self.right_lateral: List[str] = ['Fp2', 'F8', 'T4', 'T6', 'O2']
        self.left_parietal: List[str] = ['Fp1', 'F3', 'C3', 'P3', 'O1']
        self.right_parietal: List[str] = ['Fp2', 'F4', 'C4', 'P4', 'O2']
        self.midline: List[str] = ['Fz', 'Cz', 'Pz']

        # Map channel names to their indices for quick lookup
        self.channel_index: dict = {name: idx for idx, name in enumerate(self.eeg_channel_names)}

        self.use_eeg: bool = use_eeg
        self.use_spec: bool = use_spec
        self.use_mix: bool = use_mix
        self.use_mne_filter: bool = use_mne_filter

    def __getitem__(self, index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve a processed data sample.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                Processed EEG or spectrogram data. In mixed mode, returns a tuple (EEG, spectrogram).
        """
        data_point = self.dataframe.iloc[index]
        return self._process_single_data(data_point, self.training_flag)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataframe)

    def compute_brain_leads(self, waves: np.ndarray) -> np.ndarray:
        """
        Compute differential signals (brain leads) from raw EEG data.

        Args:
            waves (np.ndarray): Raw EEG data with shape (channels, time).

        Returns:
            np.ndarray: Concatenated differential brain lead signals.
        """
        waves_copy = copy.deepcopy(waves)
        # Groups of channels used for differential calculation
        brain_lead_groups = [self.left_lateral, self.right_lateral, self.left_parietal, self.right_parietal]
        differential_leads: List[np.ndarray] = []

        for group in brain_lead_groups:
            for i in range(len(group) - 1):
                diff_signal = waves_copy[self.channel_index[group[i]]] - waves_copy[self.channel_index[group[i + 1]]]
                differential_leads.append(diff_signal)

        return np.stack(differential_leads, axis=0)

    def mirror_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply mirroring transformation to a spectrogram using a fixed index permutation.

        Args:
            spectrogram (np.ndarray): Spectrogram data.

        Returns:
            np.ndarray: Mirrored spectrogram.
        """
        index_order = [1, 0, 3, 2]
        return spectrogram[..., index_order]

    def mirror_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Mirror EEG data by swapping left-side channels with corresponding right-side channels.

        Args:
            eeg_data (np.ndarray): EEG data.

        Returns:
            np.ndarray: EEG data after applying mirror swap.
        """
        # Define indices for left and right channels (based on your data ordering)
        left_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        right_indices = [11, 12, 13, 14, 15, 16, 17, 18]
        eeg_data[left_indices, ...], eeg_data[right_indices, ...] = eeg_data[right_indices, ...], eeg_data[left_indices, ...]
        return eeg_data

    def butter_bandpass(self, lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Butterworth bandpass filter coefficients.

        Args:
            lowcut (float): Lower frequency cutoff.
            highcut (float): Upper frequency cutoff.
            fs (float): Sampling frequency.
            order (int): Filter order.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filter coefficients (b, a).
        """
        return butter(order, [lowcut, highcut], fs=fs, btype="band")

    def butter_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to the data.

        Args:
            data (np.ndarray): Data to filter.
            lowcut (float): Lower frequency cutoff.
            highcut (float): Upper frequency cutoff.
            fs (float): Sampling frequency.
            order (int): Filter order.

        Returns:
            np.ndarray: Filtered data.
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data)

    def load_eeg_data(self, data_point: pd.Series, is_training: bool, flip: bool = False) -> np.ndarray:
        """
        Load and process EEG data from a single data record.

        Args:
            data_point (pd.Series): A row from the dataframe containing metadata.
            is_training (bool): Flag indicating training mode.
            flip (bool): If True, apply mirroring to the EEG data.

        Returns:
            np.ndarray: Processed EEG data.
        """
        if is_training:
            eeg_file_path = f"/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/{data_point['eeg_id']}.parquet"
        else:
            eeg_file_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{data_point['eeg_id']}.parquet"
        eeg_df = pd.read_parquet(eeg_file_path)

        offset = 0
        eeg_df = eeg_df.iloc[int(offset * 200):int(offset * 200) + 10000]
        waves = eeg_df.values
        waves = np.transpose(waves, axes=[1, 0])

        # Handle NaN values for each channel
        for channel in range(waves.shape[0]):
            channel_mean = np.nanmean(waves[channel])
            if np.isnan(waves[channel]).mean() < 1:
                waves[channel] = np.nan_to_num(waves[channel], nan=channel_mean)
            else:
                waves[channel] = 0

        if flip:
            waves = self.mirror_eeg(waves)

        # Compute differential leads and apply filtering
        waves = self.compute_brain_leads(waves)
        waves = np.array(waves, dtype=np.float64)
        waves = np.clip(waves, -1024, 1024)
        if self.use_mne_filter:
            waves = mne.filter.filter_data(waves, 200, self.lower_cut, self.upper_cut, verbose=False)
        else:
            waves = self.butter_bandpass_filter(waves, 0.5, 20, 200, order=2)
        return waves

    def load_spectrogram(self, data_point: pd.Series, is_training: bool, flip: bool = False) -> np.ndarray:
        """
        Load and process spectrogram data from a single data record.

        Args:
            data_point (pd.Series): A row from the dataframe containing metadata.
            is_training (bool): Flag indicating training mode.
            flip (bool): If True, apply mirroring to the spectrogram.

        Returns:
            np.ndarray: Processed spectrogram data.
        """
        if is_training:
            spec_file_path = f"/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/{data_point['spectrogram_id']}.parquet"
        else:
            spec_file_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{data_point['spectrogram_id']}.parquet"
        spec_df = pd.read_parquet(spec_file_path)
        spec_values = spec_df.values[:, 1:]  # Exclude the first column

        spectrogram_images: List[np.ndarray] = []
        row_start = 0

        for region in range(4):
            # Extract and process region-specific image
            image = spec_values[row_start:row_start + 300, region * 100:(region + 1) * 100].T
            image = np.clip(image, np.exp(-4), np.exp(8))
            image = np.log(image)
            image = np.nan_to_num(image, nan=0.0)
            spectrogram_images.append(image)

        stacked_images = np.stack(spectrogram_images, axis=-1)
        if flip:
            stacked_images = self.mirror_spectrogram(stacked_images)
        processed_spec = np.transpose(stacked_images, [2, 0, 1])
        return processed_spec

    def load_mixed_data(self, data_point: pd.Series, is_training: bool, flip: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process both EEG and spectrogram data for a single data record.

        Args:
            data_point (pd.Series): A row from the dataframe containing metadata.
            is_training (bool): Flag indicating training mode.
            flip (bool): If True, apply mirroring to the data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing processed EEG data and spectrogram data.
        """
        if is_training:
            eeg_file_path = f"/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/{data_point['eeg_id']}.parquet"
        else:
            eeg_file_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{data_point['eeg_id']}.parquet"
        eeg_df = pd.read_parquet(eeg_file_path)
        offset = 0
        eeg_df = eeg_df.iloc[int(offset * 200):int(offset * 200) + 10000]
        waves = eeg_df.values
        waves = np.transpose(waves, axes=[1, 0])
        for channel in range(waves.shape[0]):
            channel_mean = np.nanmean(waves[channel])
            if np.isnan(waves[channel]).mean() < 1:
                waves[channel] = np.nan_to_num(waves[channel], nan=channel_mean)
            else:
                waves[channel] = 0
        if flip:
            waves = self.mirror_eeg(waves)
        waves = self.compute_brain_leads(waves)
        waves = np.array(waves, dtype=np.float64)
        waves = np.clip(waves, -1024, 1024)
        waves = mne.filter.filter_data(waves, 200, self.lower_cut, self.upper_cut, verbose=False)

        # Load spectrogram data
        row_start = 0
        spec_file_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{data_point['spectrogram_id']}.parquet"
        spec_df = pd.read_parquet(spec_file_path)
        spec_values = spec_df.values[:, 1:]
        spectrogram_images: List[np.ndarray] = []
        for region in range(4):
            image = spec_values[row_start:row_start + 300, region * 100:(region + 1) * 100].T
            image = np.clip(image, np.exp(-4), np.exp(8))
            image = np.log(image)
            image = np.nan_to_num(image, nan=0.0)
            spectrogram_images.append(image)
        stacked_images = np.stack(spectrogram_images, axis=-1)
        if flip:
            stacked_images = self.mirror_spectrogram(stacked_images)
        processed_spec = np.transpose(stacked_images, [2, 0, 1])

        return waves, processed_spec

    def _process_single_data(self, data_point: pd.Series, is_training: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Process a single data record into the desired modality (EEG, spectrogram, or mixed).

        Args:
            data_point (pd.Series): A row from the dataframe.
            is_training (bool): Flag indicating training mode.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                Processed EEG or spectrogram data, or both in mixed mode.
        """
        if self.use_eeg:
            eeg_data = self.load_eeg_data(data_point, is_training, self.flip_eeg)
            return eeg_data.astype(np.float32)
        elif self.use_spec:
            spec_data = self.load_spectrogram(data_point, is_training, self.flip_eeg)
            return spec_data.astype(np.float32)
        elif self.use_mix:
            eeg_data, spec_data = self.load_mixed_data(data_point, is_training, self.flip_eeg)
            return eeg_data.astype(np.float32), spec_data.astype(np.float32)
        
        # Default: return EEG data if no specific modality is selected
        default_data = self.load_eeg_data(data_point, is_training, self.flip_eeg)
        return default_data.astype(np.float32)



class Transform50s(nn.Module):
    """
    Applies a spectrogram transformation tailored for a 50s view of the EEG signal.

    This transform computes a spectrogram from the input signal using:
        - FFT size: 512
        - Window lencgth: 128
        - Hop length: 50
        - Power: 1 (magnitude spectrogram)
    It then clips the values, scales them down, and slices the frequency axis
    to select a specific region.
    """
    def __init__(self) -> None:
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=128,
            hop_length=50,
            power=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Transform50s.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time) or (batch_size, channels, time).

        Returns:
            Tensor: Processed spectrogram with shape (batch_size, channels, selected_freq, time).
        """
        # Compute spectrogram.
        image = self.wave_transform(x)
        # Clip values and scale down.
        image = torch.clip(image, min=0, max=10000) / 1000
        # Get dimensions and slice along frequency (height) dimension.
        n, c, h, w = image.size()
        image = image[:, :, :int(20 / 100 * h + 10), :]
        return image


class Transform10s(nn.Module):
    """
    Applies a spectrogram transformation tailored for a 10s slice of the EEG signal.

    This transform computes a spectrogram with:
        - FFT size: 512
        - Window length: 128
        - Hop length: 10
        - Power: 1
    Similar to Transform50s, it clips and scales the output and then selects a portion
    of the frequency axis.
    """
    def __init__(self) -> None:
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=128,
            hop_length=10,
            power=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Transform10s.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time) or (batch_size, channels, time).

        Returns:
            Tensor: Processed spectrogram with shape (batch_size, channels, selected_freq, time).
        """
        image = self.wave_transform(x)
        image = torch.clip(image, min=0, max=10000) / 1000
        n, c, h, w = image.size()
        image = image[:, :, :int(20 / 100 * h + 10), :]
        return image


class Modelx3d(nn.Module):
    """
    Wrapper for an X3D model configured for processing video-like data.

    The X3D model is created with:
        - Input clip length: 16 frames
        - Input crop size: 312
        - Depth factor: 5.0

    Certain layers in block 5 are replaced by identity modules to adjust the network's behavior.
    """
    def __init__(self) -> None:
        super().__init__()
        model_name = "x3d_l"
        self.net = create_x3d(
            input_clip_length=16,
            input_crop_size=312,
            depth_factor=5.0,
        )
        # Modify block 5: remove dropout, projection, activation, and output pooling.
        self.net.blocks[5].dropout = nn.Identity()
        self.net.blocks[5].proj = nn.Identity()
        self.net.blocks[5].activation = nn.Identity()
        self.net.blocks[5].output_pool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the X3D model.

        Args:
            x (Tensor): Input tensor of shape compatible with X3D (typically video-like).

        Returns:
            Tensor: Feature representation produced by the X3D model.
        """
        x = self.net(x)
        return x


class Netx3d(nn.Module):
    """
    Combines two spectrogram transforms with an X3D backbone for EEG classification.

    The network processes EEG data in two temporal resolutions:
        - A "50s" view using Transform50s.
        - A "10s" view (a slice from the EEG) using Transform10s.

    The outputs from these transforms are concatenated, converted to a 3-channel input,
    and then passed through the X3D model. Finally, a fully connected layer (with softmax)
    produces the class probabilities.
    """
    def __init__(self, num_classes: int = 6) -> None:
        """
        Initialize the Netx3d model.

        Args:
            num_classes (int): The number of output classes. Default is 6.
        """
        super().__init__()
        self.preprocess50s = Transform50s()
        self.preprocess10s = Transform10s()
        self.model = Modelx3d()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2048, num_classes, bias=True)
    
    def forward(self, eeg: Tensor) -> Tensor:
        """
        Forward pass of the Netx3d model.

        Args:
            eeg (Tensor): Input EEG data tensor. Expected shape is at least (batch_size, channels, time).
                          The full EEG is used for the 50s transform, and a temporal slice is used for the 10s transform.

        Returns:
            Tensor: The output probabilities for each class, with shape (batch_size, num_classes).
        """
        bs = eeg.size(0)

        # For the 50s transform, use the entire EEG.
        eeg_50s = eeg
        # For the 10s transform, select a temporal slice (e.g., from time index 4000 to 6000).
        eeg_10s = eeg[:, :, 4000:6000]
        x_50 = self.preprocess50s(eeg_50s)
        x_10 = self.preprocess10s(eeg_10s)
        # Concatenate along the channel dimension.
        x = torch.cat([x_10, x_50], dim=1)

        # The resulting tensor x is then unsqueezed to add a new dimension.
        x = torch.unsqueeze(x, dim=1)
        # Replicate the tensor across three channels to match the expected input for the X3D model.
        x = torch.cat([x, x, x], dim=1)
            
        # Pass through the X3D model.
        x = self.model(x)
        x = x.view(bs, -1)
        
        # Apply the fully connected layer.
        x = self.fc(x)
        # Apply softmax to produce probability distributions over classes.
        x = torch.softmax(x, dim=-1)
        return x

    def forward_ablate(self, eeg: torch.Tensor, ablate: str = None) -> torch.Tensor:
        """
        Forward pass that can "ablate" (zero-out) one of the branches.
        
        Args:
            eeg (torch.Tensor): Input EEG data, shape (batch_size, channels, time).
            ablate (str, optional): 
                - '10s' to ablate (zero out) the 10s branch,
                - '50s' to ablate the 50s branch,
                - None to use the full model.
        
        Returns:
            torch.Tensor: The output probabilities.
        """
        bs = eeg.size(0)
        eeg_50s = eeg
        eeg_10s = eeg[:, :, 4000:6000]
        x_50 = self.preprocess50s(eeg_50s)
        x_10 = self.preprocess10s(eeg_10s)
        
        # Ablate one branch if specified.
        if ablate == '10s':
            x_10 = torch.zeros_like(x_10)
        elif ablate == '50s':
            x_50 = torch.zeros_like(x_50)
        
        x = torch.cat([x_10, x_50], dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = torch.cat([x, x, x], dim=1)
        x = self.model(x)
        x = x.view(bs, -1)
        x = self.fc(x)
        x = torch.softmax(x, dim=-1)
        return x




class Transformdoubleheadbutterfilter(nn.Module):
    """
    Transforms raw EEG data into a spectrogram representation using a Butterworth filtering approach.
    
    This module computes a spectrogram via the STFT using torchaudio.transforms.Spectrogram, applies a logarithmic scaling,
    clips the values, and then slices the frequency axis to retain the lower frequencies (roughly 0-20 Hz).
    
    Expected input:
        Tensor of shape (batch_size, time)
    
    Returns:
        Tensor of shape (batch_size, channels, selected_freq, time_frames)
    """
    def __init__(self) -> None:
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512,
            hop_length=50,
            power=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the spectrogram transformation.
        
        Args:
            x (Tensor): Raw EEG signal of shape (batch_size, time).
            
        Returns:
            Tensor: Processed spectrogram of shape (batch_size, channels, selected_freq, time_frames).
        """
        bs: int = x.size(0)
        image: Tensor = self.wave_transform(x)  # Compute spectrogram
        image = torch.log10(image)              # Logarithmic scaling
        image = torch.clip(image, min=0)         # Clip negative values
        n, c, h, w = image.size()
        # Retain only the lower frequencies (approximately 0-20 Hz)
        image = image[:, :, :int(20 / 100 * h + 5), :]
        return image

class Modeleegbutterfilter(nn.Module):
    """
    Feature extraction branch for raw EEG data using EfficientNet_B5.
    
    This module reshapes a flattened EEG signal into a pseudo-image.
    The raw EEG (assumed to be a flattened tensor of shape (batch_size, 160000))
    is reshaped to (batch_size, 16, 1000, 10), then permuted and flattened into 
    (batch_size, 16*10, 1000). It then unsqueezes and replicates the single-channel 
    data into 3 channels to form a pseudo-RGB image for EfficientNet_B5.
    
    Expected input:
        Tensor of shape (batch_size, 160000)
    
    Returns:
        Tensor: Feature vector of shape (batch_size, feature_dim) extracted by EfficientNet_B5.
    """
    def __init__(self) -> None:
        super(Modeleegbutterfilter, self).__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=False, in_chans=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, out_features=6, bias=True)

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extract features using EfficientNet_B5's forward_features method.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, H, W).
            
        Returns:
            Tensor: Extracted feature map.
        """
        x = self.model.forward_features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the raw EEG branch.
        
        Steps:
          1. Reshape the flattened EEG vector from (batch_size, 160000) to (batch_size, 16, 1000, 10).
          2. Permute dimensions to (batch_size, 16, 10, 1000) and flatten the channel and segment dimensions.
          3. Unsqueeze to add a channel dimension and replicate to form a 3-channel image.
          4. Extract features using EfficientNet_B5 and apply adaptive pooling.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, 160000).
        
        Returns:
            Tensor: Extracted features of shape (batch_size, feature_dim).
        """
        bs: int = x.size(0)
        reshaped_tensor: Tensor = x.view(bs, 16, 1000, 10)          # (bs, 16, 1000, 10)
        reshaped_and_permuted_tensor: Tensor = reshaped_tensor.permute(0, 1, 3, 2)  # (bs, 16, 10, 1000)
        reshaped_and_permuted_tensor = reshaped_and_permuted_tensor.reshape(bs, 16 * 10, 1000)  # (bs, 160, 1000)
        x = torch.unsqueeze(reshaped_and_permuted_tensor, dim=1)       # (bs, 1, 160, 1000)
        x = torch.cat([x, x, x], dim=1)                                # (bs, 3, 160, 1000)
        bs = x.size(0)
        x = self.extract_features(x)
        x = self.pool(x)
        x = x.view(bs, -1)
        return x

class Modelspecbutterfilter(nn.Module):
    """
    Feature extraction branch for EEG spectrogram data using an X3D backbone.
    
    This branch processes the spectrogram output from Transformdoubleheadbutterfilter.
    The spectrogram is expanded to 3 channels and passed through an X3D model (with modifications
    in block 5, where dropout, projection, activation, and output pooling are replaced with identity).
    
    Expected input:
        Tensor of shape (batch_size, H, W) (i.e., the spectrogram).
    
    Returns:
        Tensor: Flattened feature vector of shape (batch_size, feature_dim).
    """
    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        model_name: str = "x3d_l"
        self.net = create_x3d(input_clip_length=16, input_crop_size=312, depth_factor=5.0)
        # Replace specific components in block 5 with Identity to modify network behavior
        self.net.blocks[5].dropout = nn.Identity()
        self.net.blocks[5].proj = nn.Identity()
        self.net.blocks[5].activation = nn.Identity()
        self.net.blocks[5].output_pool = nn.Identity()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the spectrogram branch.
        
        Args:
            x (Tensor): Input spectrogram of shape (batch_size, H, W).
            
        Returns:
            Tensor: Flattened feature vector.
        """
        # Add channel dimension and replicate to 3 channels: (bs, 3, H, W)
        x = torch.unsqueeze(x, dim=1)
        x = torch.cat([x, x, x], dim=1)
        x = self.net(x)
        bs: int = x.size(0)
        x = x.view(bs, -1)
        return x

class Netdoubleheadbutterfilter(nn.Module):
    """
    Double-head EEG classification model that fuses features from both the raw EEG branch and the
    spectrogram branch (using a Butterworth filter variant).
    
    The raw EEG branch (Modeleegbutterfilter) processes the flattened EEG data to extract features,
    while the spectrogram branch (Modelspecbutterfilter) processes the log-scaled, clipped spectrogram.
    Their outputs are concatenated and passed through a dropout layer and a fully connected layer to
    generate class probabilities over 6 classes.
    
    Expected input:
        Tensor of shape (batch_size, time) where time corresponds to the raw EEG signal length.
    
    Returns:
        Tensor: Class probabilities of shape (batch_size, 6).
    """
    def __init__(self) -> None:
        super(Netdoubleheadbutterfilter, self).__init__()
        self.transform: nn.Module = Transformdoubleheadbutterfilter()  # Converts raw EEG to spectrogram
        self.model_wave: nn.Module = Modeleegbutterfilter()             # Extracts features from raw EEG
        self.model_spec: nn.Module = Modelspecbutterfilter()            # Extracts features from the spectrogram
        self.pool: nn.Module = nn.AdaptiveAvgPool3d(1)
        self.fc: nn.Linear = nn.Linear(2048 * 2, 6, bias=True)
        self.droup: nn.Dropout = nn.Dropout(0.5)

    def forward(self, eeg: Tensor) -> Tensor:
        """
        Forward pass for the double-head model.
        
        Steps:
          1. Convert the raw EEG signal into a spectrogram using Transformdoubleheadbutterfilter.
          2. Extract features from the raw EEG branch (Modeleegbutterfilter).
          3. Extract features from the spectrogram branch (Modelspecbutterfilter).
          4. Concatenate the two feature vectors.
          5. Apply dropout and a fully connected layer.
          6. Apply softmax activation to produce class probabilities.
        
        Args:
            eeg (Tensor): Raw EEG data of shape (batch_size, time).
            
        Returns:
            Tensor: Class probability tensor of shape (batch_size, 6).
        """
        bs: int = eeg.size(0)
        # Generate spectrogram from raw EEG using the transform branch
        eeg_spec: Tensor = self.transform(eeg)
        # Extract features from raw EEG
        x: Tensor = self.model_wave(eeg)
        # Extract features from spectrogram
        y: Tensor = self.model_spec(eeg_spec)
        # Concatenate features from both branches
        x = torch.cat([x, y], dim=1)
        # Apply dropout for regularization
        x = self.droup(x)
        # Fully connected layer to map features to 6 classes
        x = self.fc(x)
        # Softmax activation to output probabilities
        x = torch.softmax(x, dim=-1)
        return x



def inference_function(
    test_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    double_input: bool = False
) -> dict[str, np.ndarray]:
    """
    Run inference on the provided DataLoader using the given model.
    
    Args:
        test_loader (DataLoader): DataLoader that provides test batches.
        model (torch.nn.Module): The model to run inference on.
        device (torch.device): Device (CPU or GPU) on which inference is performed.
        double_input (bool): If True, expects each batch to be a tuple (wave, spec).
                             Otherwise, expects a single tensor.
                             
    Returns:
        Dict[str, np.ndarray]: Dictionary with key "predictions" containing a numpy array
                               of concatenated predictions.
    """
    model.eval()
    preds = []
    
    with tqdm(test_loader, unit="test_batch", desc="Inference") as t_loader:
        for batch in t_loader:
            if double_input:
                wave, spec = batch
                wave = wave.to(device)
                spec = spec.to(device)
                with torch.no_grad():
                    y_preds = model(wave, spec)
            else:
                batch = batch.to(device)
                with torch.no_grad():
                    y_preds = model(batch)
            preds.append(y_preds.cpu().numpy())
    
    concatenated_preds = np.concatenate(preds, axis=0)
    return {"predictions": concatenated_preds}


def run_weight_x3d(data_frame: pd.DataFrame) -> np.ndarray:
    """
    Run inference using the Netx3d model with multiple weights on the given data.
    
    This function iterates over a list of model weight paths provided in the global config
    under 'weights_x3d'. For each weight, it:
    
      1. Creates a DataProcessor instance (without flipping) and a corresponding DataLoader.
      2. Loads the model weights into a Netx3d model, runs inference, and collects predictions.
      3. Repeats the above steps with flipping enabled.
    
    Finally, it averages the predictions from all runs.
    
    Args:
        data_frame (pd.DataFrame): DataFrame containing metadata for each EEG sample.
    
    Returns:
        np.ndarray: Averaged predictions from the ensemble of models.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Inference with weights_x3d")
    all_predictions = []
    
    # Iterate over each model weight in the global config
    for model_weight in config['weights_x3d']:
        # --- Inference without flipping ---
        test_dataset = DataProcessor(
            data_frame,
            training_flag=False,
            shuffle=False,
            use_eeg=True,
            lower_cut=0.5,
            upper_cut=20,
            use_mne_filter=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_worker'],
            shuffle=False
        )
        model = Netx3d()
        state_dict = torch.load(model_weight, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        pred_dict = inference_function(test_loader, model, device)
        all_predictions.append(pred_dict["predictions"])
        
        # --- Inference with flipping enabled ---
        test_dataset = DataProcessor(
            data_frame,
            training_flag=False,
            shuffle=False,
            use_eeg=True,
            flip=True,
            lower_cut=0.5,
            upper_cut=20,
            use_mne_filter=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_worker'],
            shuffle=False
        )
        model = Netx3d()
        state_dict = torch.load(model_weight, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        pred_dict = inference_function(test_loader, model, device)
        all_predictions.append(pred_dict["predictions"])
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Average predictions across all runs
    all_predictions = np.array(all_predictions)
    averaged_predictions = np.mean(all_predictions, axis=0)
    return averaged_predictions


def run_weight_double_headbutterfilter(data_frame: pd.DataFrame) -> np.ndarray:
    """
    Run inference using the Netdoubleheadbutterfilter model with multiple weight files on the given data.
    
    This function iterates over each model weight path provided in the global configuration
    under CFG['weights_doubleheadbutterfilter']. For each weight, it performs inference twice:
      1. Once with the default configuration (without flipping).
      2. Once with flipping enabled.
    
    In each case, an AlaskaDataIter instance is created with the appropriate parameters, a DataLoader
    is built, the model weights are loaded into a new Netdoubleheadbutterfilter instance, and inference
    is performed via the inference_function. The predictions from both runs for all weight files are collected
    and then averaged to produce the final output.
    
    Args:
        data_frame (pd.DataFrame): DataFrame containing metadata for each EEG sample.
    
    Returns:
        np.ndarray: Averaged predictions from the ensemble of model runs as a NumPy array.
    """
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Inference with weights_doubleheadbutterfilter")
    
    predictions: List[np.ndarray] = []
    
    # Iterate over each model weight in the configuration
    for model_weight in config['weights_doubleheadbutterfilter']:
        # --- Inference without flipping ---
        test_dataset = DataProcessor(
            data_frame,
            training_flag=False,
            shuffle=False,
            use_eeg=True,
            lower_cut=0.5,
            upper_cut=20,
            use_mne_filter=False
        )
        test_loader: DataLoader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'] // 2,
            num_workers=config['num_worker'],
            shuffle=False
        )
        
        model = Netdoubleheadbutterfilter()
        state_dict = torch.load(model_weight, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        pred_dict: Dict[str, np.ndarray] = inference_function(test_loader, model, device)
        predictions.append(pred_dict["predictions"])
        
        # --- Inference with flipping enabled ---
        test_dataset = DataProcessor(
            data_frame,
            training_flag=False,
            shuffle=False,
            flip=True,
            use_eeg=True,
            lower_cut=0.5,
            upper_cut=20,
            use_mne_filter=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'] // 2,
            num_workers=config['num_worker'],
            shuffle=False
        )
        
        model = Netdoubleheadbutterfilter()
        state_dict = torch.load(model_weight, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        pred_dict = inference_function(test_loader, model, device)
        predictions.append(pred_dict["predictions"])
        
        # Clear GPU memory after processing each weight
        torch.cuda.empty_cache()
        gc.collect()
    
    # Convert the list of predictions to a NumPy array and average across all runs
    predictions_np: np.ndarray = np.array(predictions)
    averaged_predictions: np.ndarray = np.mean(predictions_np, axis=0)
    
    return averaged_predictions


# Initialize a list to collect predictions from different ensemble branches.
predictions = []

# -------------------------------
# Inference using the X3D branch:
# -------------------------------
prediction_x3d = run_weight_x3d(test_df)
predictions.append(prediction_x3d)

# --------------------------------------------------------------
# Inference using the Double-Head Butter Filter branch:
# --------------------------------------------------------------
prediction_double_net = run_weight_double_headbutterfilter(test_df)
predictions.append(prediction_double_net)

# ---------------------------------------
# Ensemble Aggregation:
# ---------------------------------------
weights_by_score = np.array([0.2, 0.8])

# Combine the predictions from both models using a weighted sum.
final_predictions = predictions[0] * weights_by_score[0] + predictions[1] * weights_by_score[1]

# Output the final ensemble predictions.
final_predictions



# Prepare Submission
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv('submission.csv',index=False)
print(f'Submissionn shape: {sub.shape}')
sub.head()


!pip install zennit


from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient


# =============================================================================
# Utility Functions
# =============================================================================

def load_data_and_model(model, model_weight, device):
    """
    Loads training and test data, sets up the model with weights, and moves it to the specified device.
    
    Args:
        model (nn.Module): Model instance (e.g., Netx3d).
        model_weight (str): Path to the model weights file.
        device (torch.device): Device on which to load the model.
        config (dict): Dictionary containing data paths.
        
    Returns:
        tuple: (train_df, test_df, model)
    """
    # Load data.
    train_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(config['data'])
    
    # Load weights.
    state_dict = torch.load(model_weight, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    return train_df, test_df, model

def prepare_sample(test_df, sample_index, device):
    """
    Creates a test dataset instance, selects a sample, and converts it to a torch.Tensor.
    
    Args:
        test_df (pd.DataFrame): DataFrame with test metadata.
        sample_index (int): Index of the sample to explain.
        device (torch.device): Device for the tensor.
        
    Returns:
        torch.Tensor: Input tensor of shape (1, channels, time)
    """
    test_dataset = DataProcessor(
        test_df,
        training_flag=False,
        shuffle=False,
        use_eeg=True,
        lower_cut=0.5,
        upper_cut=20,
        use_mne_filter=False
    )
    
    sample_data = test_dataset[sample_index]
    if isinstance(sample_data, tuple):
        sample_data = sample_data[0]
    
    # Convert sample to tensor and add a batch dimension.
    input_tensor = torch.tensor(sample_data).unsqueeze(0).to(device)
    print("Input tensor shape:", input_tensor.shape)
    return input_tensor

def compute_lrp_attributions(model, input_tensor, composite):
    """
    Computes relevance scores on the input using Gradient-based LRP.
    
    Args:
        model (nn.Module): The model on which to compute attributions.
        input_tensor (torch.Tensor): Model input with shape (1, channels, time).
        composite: Composite rule for Zennit (e.g., EpsilonGammaBox).
        
    Returns:
        tuple: (output, relevance)
            - output: Model's output tensor.
            - relevance: Relevance tensor with the same shape as the input.
    """
    with Gradient(model=model, composite=composite) as attributor:
        output = model(input_tensor)
        target_class = output.argmax(dim=1)
        one_hot_target = torch.zeros_like(output)
        one_hot_target.scatter_(1, target_class.unsqueeze(1), 1.0)
        # Compute relevance.
        _, relevance = attributor(input_tensor, one_hot_target)
    return output, relevance

def visualize_relevance_overlay(input_np, relevance_np, channel_idx):
    """
    Overlays the relevance map on the raw EEG signal for a specified channel.
    
    Args:
        input_np (np.ndarray): Raw EEG input, shape (channels, time).
        relevance_np (np.ndarray): Relevance map, same shape as input_np.
        channel_idx (int): Index of the channel to visualize.
    """
    time_axis = np.arange(input_np.shape[1])
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, input_np[channel_idx], color='black', label='Raw EEG')
    plt.imshow(relevance_np[channel_idx][np.newaxis, :],
               aspect='auto', cmap='seismic', alpha=0.5,
               extent=[0, input_np.shape[1], input_np[channel_idx].min(), input_np[channel_idx].max()])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Channel {channel_idx} - Raw EEG with LRP Relevance Overlay')
    plt.legend()
    plt.show()

def plot_avg_relevance(avg_relevance_per_channel):
    """
    Plots a bar chart of the average relevance per channel.
    
    Args:
        avg_relevance_per_channel (np.ndarray): Array with average relevance for each channel.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(avg_relevance_per_channel)), avg_relevance_per_channel)
    plt.xlabel('Channel Index')
    plt.ylabel('Average Relevance')
    plt.title('Average LRP Relevance per Channel')
    plt.show()

def compute_class_relevance(model, input_tensor, composite, num_classes):
    """
    Computes LRP relevance scores for each class and aggregates mean relevance per channel.
    
    Args:
        model (nn.Module): The model on which to compute attributions.
        input_tensor (torch.Tensor): Input with shape (1, channels, time).
        composite: Composite rule for LRP (e.g., EpsilonGammaBox).
        num_classes (int): Total number of classes.
        
    Returns:
        dict: Mapping class index -> numpy array of mean relevance per channel.
    """
    class_relevance = {}
    # Get output for reference.
    with Gradient(model=model, composite=composite) as attributor:
        output = model(input_tensor)
    for cls in range(num_classes):
        one_hot = torch.zeros_like(output)
        one_hot[:, cls] = 1.0
        with Gradient(model=model, composite=composite) as attributor_cls:
            _, rel_cls = attributor_cls(input_tensor, one_hot)
        rel_cls_np = rel_cls.squeeze(0).cpu().detach().numpy()
        class_relevance[cls] = np.mean(rel_cls_np, axis=1)  # Average over time.
    return class_relevance

def plot_class_relevance_grouped(class_relevance, num_channels):
    """
    Plots a grouped bar chart of mean LRP relevance per channel across classes.
    
    Args:
        class_relevance (dict): Mapping class index -> per-channel relevance (numpy array).
        num_channels (int): Number of EEG channels.
    """
    channels = np.arange(num_channels)
    plt.figure(figsize=(10, 6))
    width = 0.1
    num_classes = len(class_relevance)
    for cls in range(num_classes):
        plt.bar(channels + cls * width, class_relevance[cls],
                width=width, label=f'Class {cls}')
    plt.xlabel('Channel Index')
    plt.ylabel('Mean Relevance')
    plt.title('Mean LRP Relevance per Channel across Classes')
    plt.legend()
    plt.show()

def compute_avg_class_relevance_over_dataset(model, dataset, composite, num_classes, device, sample_limit=None):
    """
    Iterates over a dataset and computes the average LRP relevance per channel for each class.
    
    Args:
        model (nn.Module): Model on which to compute LRP.
        dataset (iterable): Dataset instance (e.g., from DataProcessor) returning raw EEG samples.
        composite: Composite rule for LRP (e.g., EpsilonGammaBox).
        num_classes (int): Total number of classes.
        device (torch.device): Device for computation.
        sample_limit (int, optional): Optional limit on the number of samples to process.
        
    Returns:
        dict: Mapping class index -> average per-channel relevance (numpy array).
    """
    sums = {cls: None for cls in range(num_classes)}
    count = 0
    for idx, sample in enumerate(dataset):
        if sample_limit is not None and idx >= sample_limit:
            break
        if isinstance(sample, tuple):
            sample = sample[0]
        input_tensor = torch.tensor(sample).unsqueeze(0).to(device)
        with Gradient(model=model, composite=composite) as attributor:
            output = model(input_tensor)
            for cls in range(num_classes):
                one_hot = torch.zeros_like(output)
                one_hot[:, cls] = 1.0
                _, rel_cls = attributor(input_tensor, one_hot)
                rel_cls_np = rel_cls.squeeze(0).cpu().detach().numpy()
                sample_rel = np.mean(rel_cls_np, axis=1)  # Mean over time.
                if sums[cls] is None:
                    sums[cls] = sample_rel
                else:
                    sums[cls] += sample_rel
        count += 1
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} samples...")
    
    averages = {cls: sums[cls] / count for cls in range(num_classes)}
    return averages

def plot_dataset_class_relevance(avg_class_relevance, num_channels):
    """
    Plots grouped bar charts of average LRP relevance per channel across classes computed over a dataset.
    
    Args:
        avg_class_relevance (dict): Mapping class index -> average per-channel relevance.
        num_channels (int): Number of EEG channels.
    """
    channels = np.arange(num_channels)
    plt.figure(figsize=(10, 6))
    width = 0.1
    num_classes = len(avg_class_relevance)
    for cls in range(num_classes):
        plt.bar(channels + cls * width, avg_class_relevance[cls],
                width=width, label=f'Class {cls}')
    plt.xlabel('Channel Index')
    plt.ylabel('Average Relevance (Dataset)')
    plt.title('Average LRP Relevance per Channel across Classes (Dataset Aggregation)')
    plt.legend()
    plt.show()

# =============================================================================
# Main Workflow
# =============================================================================

def main():
    # Set up device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data and model.
    model_weight = config['weights_x3d'][0]
    model = Netx3d()  
    train_df, test_df, model = load_data_and_model(model, model_weight, device)
    
    # Prepare a test sample and compute single-sample LRP.
    sample_index = 0
    input_tensor = prepare_sample(test_df, sample_index, device)
    composite = EpsilonGammaBox(low=-3., high=3., canonizers=[SequentialMergeBatchNorm()])
    output, relevance = compute_lrp_attributions(model, input_tensor, composite)
    
    # Convert tensors to numpy arrays.
    relevance_np = relevance.squeeze(0).cpu().detach().numpy()  # (channels, time)
    input_np = input_tensor.squeeze(0).cpu().detach().numpy()     # (channels, time)
    
    # Visualize raw EEG overlay with relevance for a specified channel.
    for channel_index in range(len(relevance_np)):
        visualize_relevance_overlay(input_np, relevance_np, channel_index)
    
    # Plot average relevance per channel for the single sample.
    avg_relevance_per_channel = np.mean(relevance_np, axis=1)
    plot_avg_relevance(avg_relevance_per_channel)
    
    # Compute and plot per-class relevance for the single sample.
    num_classes = output.shape[1]
    class_relevance = compute_class_relevance(model, input_tensor, composite, num_classes)
    num_channels = input_np.shape[0]
    plot_class_relevance_grouped(class_relevance, num_channels)
    
    # =============================================================================
    # Dataset-Aggregated Analysis
    # =============================================================================
    sample_limit = 200  
    
    # Create a dataset instance from the training DataFrame.
    train_dataset = DataProcessor(
        test_df,
        training_flag=False,
        shuffle=False,
        use_eeg=True,
        lower_cut=0.5,
        upper_cut=20,
        use_mne_filter=False
    )
    
    avg_class_relevance = compute_avg_class_relevance_over_dataset(model, train_dataset, composite, num_classes, device, sample_limit)
    plot_dataset_class_relevance(avg_class_relevance, num_channels)
    
    print("XAI analysis and dataset-aggregated relevance complete.")

if __name__ == "__main__":
    main()



# ------------------------------
# Model Wrapper for Logits
# ------------------------------

class Netx3dLogits(Netx3d):
    """
    A wrapper for Netx3d that returns logits instead of probabilities.
    """
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        bs = eeg.size(0)
        # Process full EEG for 50s branch and a slice for 10s branch.
        eeg_50s = eeg
        eeg_10s = eeg[:, :, 4000:6000]
        x_50 = self.preprocess50s(eeg_50s)
        x_10 = self.preprocess10s(eeg_10s)
        x = torch.cat([x_10, x_50], dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = torch.cat([x, x, x], dim=1)
        x = self.model(x)
        x = x.view(bs, -1)
        logits = self.fc(x)  # do not apply softmax
        return logits

# ------------------------------
# Counterfactual Generation Function
# ------------------------------

def generate_counterfactual(model, original_input, target_class, 
                            num_iterations=100, learning_rate=0.01,
                            regularization_weight=0.1):
    """
    Generates a counterfactual input that forces the model to predict the target class.
    
    Optimization minimizes a loss combining cross-entropy (to match the target) and an L2 penalty
    (to keep the perturbation small).
    
    Args:
        model (nn.Module): Model that outputs logits.
        original_input (torch.Tensor): Input of shape (1, channels, time).
        target_class (int): Target class index.
        num_iterations (int): Maximum steps for optimization.
        learning_rate (float): Learning rate.
        regularization_weight (float): Weight for the perturbation loss.
        
    Returns:
        torch.Tensor: Counterfactual input (same shape as original_input).
    """
    cf_input = original_input.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([cf_input], lr=learning_rate)
    
    # Use a fixed target tensor.
    target = torch.tensor([target_class], device=original_input.device)
    
    for i in range(num_iterations):
        logits = model(cf_input)
        classification_loss = nn.functional.cross_entropy(logits, target)
        perturbation_loss = torch.norm(cf_input - original_input)
        loss = classification_loss + regularization_weight * perturbation_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clamp values to remain within valid EEG ranges (adjust as needed).
        with torch.no_grad():
            cf_input.clamp_(min=original_input.min(), max=original_input.max())
        
        if i % 10 == 0:
            pred_class = logits.argmax(dim=1).item()
            print(f"Iteration {i}: Loss = {loss.item():.4f}, Predicted = {pred_class}")
            # Optionally, break early if the prediction matches the target.
            if pred_class == target_class:
                print(f"Counterfactual achieved target class: {target_class}.")
                break
    
    return cf_input.detach()

# ------------------------------
# Visualization Functions for Counterfactuals
# ------------------------------

def plot_best_counterfactuals_side_by_side(original_np, counterfactuals, predictions, channels_to_plot):
    """
    For each channel in channels_to_plot, select the counterfactual signal that has the largest mean absolute
    difference from the original signal, and plot the original signal and that counterfactual side by side.
    
    Args:
        original_np (np.ndarray): Original EEG signal of shape (channels, time).
        counterfactuals (dict): Mapping from target class (int) to counterfactual numpy array (channels, time).
        predictions (dict): Mapping from target class (int) to final predicted class (int) for the counterfactual.
        channels_to_plot (list): List of channel indices to plot.
    """
    time_axis = np.arange(original_np.shape[1])
    num_channels = len(channels_to_plot)
    
    # Create a subplot with two columns: one for original and one for counterfactual
    fig, axs = plt.subplots(num_channels, 2, figsize=(16, num_channels * 4), sharex=True)
    # Ensure axs is 2D even for a single channel.
    if num_channels == 1:
        axs = np.expand_dims(axs, axis=0)
    
    for i, ch in enumerate(channels_to_plot):
        original_channel = original_np[ch]
        best_diff = -np.inf
        best_cf = None
        best_cls = None
        
        # Find the counterfactual with the highest mean absolute difference for this channel
        for cls, cf_np in counterfactuals.items():
            cf_channel = cf_np[ch]
            diff = np.mean(np.abs(cf_channel - original_channel))
            if diff > best_diff:
                best_diff = diff
                best_cf = cf_channel
                best_cls = cls
        
        # Left subplot: plot original signal
        axs[i, 0].plot(time_axis, original_channel, color='black', linewidth=2)
        axs[i, 0].set_title(f'Channel {ch} Original')
        axs[i, 0].set_ylabel('Amplitude')
        axs[i, 0].grid(True)
        
        # Right subplot: plot best counterfactual signal
        axs[i, 1].plot(time_axis, best_cf, color='red', linestyle='--', linewidth=2)
        axs[i, 1].set_title(f'Channel {ch} CF Target {best_cls} (Pred {predictions[best_cls]}), Diff={best_diff:.4f}')
        axs[i, 1].grid(True)
    
    # Set x-axis label on the bottom row of subplots.
    for ax in axs[-1, :]:
        ax.set_xlabel('Time')
    
    plt.suptitle('Original vs Best Counterfactual (Side-by-Side) per Channel', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ------------------------------
# Main Workflow
# ------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data and original model.
    model_weight = config['weights_x3d'][0]
    base_model = Netx3d()
    _, test_df, base_model = load_data_and_model(base_model, model_weight, device)
    
    # Create the logits wrapper model.
    logits_model = Netx3dLogits()
    logits_model.load_state_dict(base_model.state_dict(), strict=True)
    logits_model.to(device)
    logits_model.eval()
    
    # Prepare a sample from the test data.
    sample_index = 0  
    original_input = prepare_sample(test_df, sample_index, device)
    
    # Generate counterfactuals for each class.
    logits = logits_model(original_input)
    num_classes = logits.shape[1]
    
    counterfactuals = {}
    predictions = {}
    for cls in range(num_classes):
        print(f"\nGenerating counterfactual for target class {cls}:")
        cf_input = generate_counterfactual(logits_model, original_input, target_class=cls,
                                           num_iterations=100, learning_rate=0.01,
                                           regularization_weight=0.01)
        pred_logits = logits_model(cf_input)
        pred = pred_logits.argmax(dim=1).item()
        predictions[cls] = pred
        counterfactuals[cls] = cf_input.squeeze(0).cpu().detach().numpy()
    
    # Convert the original input to numpy.
    original_np = original_input.squeeze(0).cpu().detach().numpy()
    
    # Instead of plotting all counterfactuals per channel, select only the most different one for each channel.
    channels_to_plot = list(range(original_np.shape[0]))  # Plot all channels.
    plot_best_counterfactuals_side_by_side(original_np, counterfactuals, predictions, channels_to_plot)
    
    print("Counterfactual generation complete.")

if __name__ == "__main__":
    main()




# ------------------------------
# Sensitivity Analysis Functions
# ------------------------------

def add_noise(input_tensor, noise_std=0.01):
    """
    Adds Gaussian noise to the input tensor.
    """
    noise = torch.randn_like(input_tensor) * noise_std
    return input_tensor + noise

def occlude_region(input_tensor, mask):
    """
    Applies occlusion on the input_tensor using a mask.
    'mask' is a tensor with 1s for regions to keep and 0s for regions to occlude.
    
    Here, occluding a channel by setting it to a baseline value (e.g., zeros).
    """
    baseline = torch.zeros_like(input_tensor)  # or use the mean value of input_tensor
    return input_tensor * mask + baseline * (1 - mask)


def sensitivity_analysis_on_dataset(model, dataset, noise_std, device, sample_limit=100):
    """
    Performs sensitivity analysis over all samples in the dataset.
    
    For each sample:
      - Noise Analysis: Adds Gaussian noise and computes the average absolute difference in logits.
      - Occlusion Analysis: For each channel, occludes that channel and computes:
            (a) The average absolute difference in logits.
            (b) Whether the predicted class (argmax) changes.
            
    Returns:
        avg_noise_diff (float): Average difference in logits due to noise over samples.
        avg_occlusion_diffs (np.ndarray): Array of average differences (L1 norm on logits) per channel.
        decision_changes (np.ndarray): Array (per channel) of the fraction of samples where occlusion caused a decision change.
    """
    noise_diffs = []
    num_channels = None
    occlusion_diffs_sum = None
    decision_change_counts = None
    num_samples = 0
    
    for idx, sample in enumerate(dataset):
        if idx >= sample_limit:
            break
        # Assume sample is either raw EEG or (data, ...) tuple.
        if isinstance(sample, tuple):
            sample = sample[0]
        input_tensor = torch.tensor(sample).unsqueeze(0).to(device)
        original_pred = model(input_tensor)
        orig_class = original_pred.argmax(dim=1).item()
        
        # Noise analysis.
        noisy_input = add_noise(input_tensor, noise_std=noise_std)
        noisy_pred = model(noisy_input)
        noise_diff = torch.mean(torch.abs(original_pred - noisy_pred)).item()
        noise_diffs.append(noise_diff)
        
        # Initialize channel count and occlusion accumulators.
        if num_channels is None:
            num_channels = input_tensor.shape[1]
            occlusion_diffs_sum = np.zeros(num_channels)
            decision_change_counts = np.zeros(num_channels)
        
        # Occlusion analysis per channel.
        for ch in range(num_channels):
            mask = torch.ones_like(input_tensor)
            mask[:, ch, :] = 0  # occlude channel ch
            occluded_input = occlude_region(input_tensor, mask)
            occluded_pred = model(occluded_input)
            diff = torch.mean(torch.abs(original_pred - occluded_pred)).item()
            occlusion_diffs_sum[ch] += diff
            
            # Check if the occlusion changed the predicted class.
            occluded_class = occluded_pred.argmax(dim=1).item()
            if occluded_class != orig_class:
                decision_change_counts[ch] += 1
        
        num_samples += 1
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1} samples...")
    
    avg_noise_diff = np.mean(noise_diffs) if noise_diffs else 0
    avg_occlusion_diffs = occlusion_diffs_sum / num_samples if num_samples > 0 else None
    decision_changes = decision_change_counts / num_samples if num_samples > 0 else None
    
    return avg_noise_diff, avg_occlusion_diffs, decision_changes

def plot_occlusion_results(avg_occlusion_diffs, decision_changes):
    """
    Plots the occlusion sensitivity results.
    
    Two bar charts are produced:
      1. The average change in logits per channel.
      2. The fraction of samples where occluding a channel changed the predicted class.
    
    Args:
        avg_occlusion_diffs (np.ndarray): Array of average differences per channel.
        decision_changes (np.ndarray): Array of fraction (or percentage) of decision changes per channel.
    """
    channels = np.arange(len(avg_occlusion_diffs))
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot average change in logits.
    axs[0].bar(channels, avg_occlusion_diffs, color='salmon')
    axs[0].set_xlabel('Channel Index')
    axs[0].set_ylabel('Average Change in Logits')
    axs[0].set_title('Occlusion Sensitivity (Logits Difference)')
    axs[0].grid(True)
    
    # Plot decision change fractions.
    axs[1].bar(channels, decision_changes * 100, color='skyblue')
    axs[1].set_xlabel('Channel Index')
    axs[1].set_ylabel('Percentage of Decision Changes (%)')
    axs[1].set_title('Occlusion Sensitivity (Decision Changes)')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def temporal_occlusion_analysis(model, input_tensor, segments):
    """
    Performs temporal occlusion analysis on a single input_tensor over given segments.
    For each segment, occludes that time window and computes:
        - The average difference in logits.
        - Whether the predicted class changed.
    
    Args:
        model (nn.Module): The model (logits_model).
        input_tensor (torch.Tensor): Input of shape (1, channels, time).
        segments (list of tuple): Each tuple is (start, end) indices.
    
    Returns:
        dict: Mapping from segment (start, end) to a dict with keys:
            'diff', 'decision_change', 'orig_class', 'new_class'
    """
    original_pred = model(input_tensor)
    orig_class = original_pred.argmax(dim=1).item()
    results = {}
    for (start, end) in segments:
        occluded_input = occlude_temporal_segments(input_tensor, start, end)
        occluded_pred = model(occluded_input)
        diff = torch.mean(torch.abs(original_pred - occluded_pred)).item()
        new_class = occluded_pred.argmax(dim=1).item()
        decision_change = (new_class != orig_class)
        results[(start, end)] = {
            "diff": diff,
            "decision_change": decision_change,
            "orig_class": orig_class,
            "new_class": new_class
        }
    return results

def occlude_temporal_segments(input_tensor, start, end):
    """
    Occludes (zeros) a temporal segment from start to end.
    """
    mask = torch.ones_like(input_tensor)
    mask[:, :, start:end] = 0
    return occlude_region(input_tensor, mask)

def plot_temporal_occlusion_results(results):
    """
    Plots the differences for temporal occlusion analysis.
    
    Args:
        results (dict): Dictionary returned by temporal_occlusion_analysis.
    """
    segments = list(results.keys())
    diffs = [results[seg]["diff"] for seg in segments]
    decision_changes = [results[seg]["decision_change"] for seg in segments]
    
    # Plot the average difference per segment.
    labels = [f"{seg[0]}-{seg[1]}" for seg in segments]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, diffs, color='lightgreen')
    plt.xlabel('Time Segment (start-end)')
    plt.ylabel('Average Logit Difference')
    plt.title('Temporal Occlusion Analysis: Logit Differences')
    plt.grid(True)
    plt.show()
    
    # Print decision change results.
    for seg in segments:
        res = results[seg]
        print(f"Segment {seg}: Diff={res['diff']:.4f}, Decision changed: {res['decision_change']}, Orig class: {res['orig_class']}, New class: {res['new_class']}")


# ------------------------------
# Main Workflow
# ------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data and original model.
    model_weight = config['weights_x3d'][0]
    base_model = Netx3d()
    _, test_df, base_model = load_data_and_model(base_model, model_weight, device)
    
    # Create logits wrapper model.
    logits_model = Netx3dLogits()
    logits_model.load_state_dict(base_model.state_dict(), strict=True)
    logits_model.to(device)
    logits_model.eval()
    

    # ------------------------------
    # Sensitivity Analysis Over the Test Dataset
    # ------------------------------
    print("\nPerforming Sensitivity Analysis on Test Dataset:")
    test_dataset = DataProcessor(
        test_df, 
        training_flag=False,
        shuffle=False,
        use_eeg=True,
        lower_cut=0.5,
        upper_cut=20,
        use_mne_filter=False
    )
    
    # You can adjust sample_limit as needed.
    avg_noise_diff, avg_occlusion_diffs, decision_changes = sensitivity_analysis_on_dataset(
        logits_model, test_dataset, noise_std=0.01, device=device, sample_limit=100)
    
    print(f"Average noise-induced difference in logits: {avg_noise_diff:.4f}")
    print("Average occlusion-induced difference per channel:")
    for ch, diff in enumerate(avg_occlusion_diffs):
        print(f"  Channel {ch}: {diff:.4f}")
    print("Percentage of decision changes per channel due to occlusion:")
    for ch, change in enumerate(decision_changes):
        print(f"  Channel {ch}: {change*100:.2f}%")
    
    plot_occlusion_results(avg_occlusion_diffs, decision_changes)
    
    print("Sensitivity analysis complete.")

    # ------------------------------
    # Temporal Occlusion Analysis on a single sample
    # ------------------------------
    print("\nPerforming Temporal Occlusion Analysis on selected sample:")
    # Define segments for occlusion (e.g., segments of 100 time points).
    # Adjust these ranges to fit the time axis of your EEG data.
    segments = [(0, 100), (100, 200), (200, 300), (300, 400)]
    temporal_results = temporal_occlusion_analysis(logits_model, original_input, segments)
    
    print("Temporal Occlusion Analysis Results:")
    for seg, res in temporal_results.items():
        print(f"Segment {seg}: Diff={res['diff']:.4f}, Decision changed: {res['decision_change']}, Orig class: {res['orig_class']}, New class: {res['new_class']}")
    
    plot_temporal_occlusion_results(temporal_results)


if __name__ == "__main__":
    main()










