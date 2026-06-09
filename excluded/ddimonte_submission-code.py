#!pip install ruptures


import time
#loading_time = time.perf_counter() - start_loading
#print(f"Data loading timeC: {loading_time:.4f} sec")
#start_loading = time.perf_counter()


# from: https://www.kaggle.com/code/gordonyip/update-calibrating-and-binning-astronomical-data
import numpy as np
import pandas as pd
import itertools
import os
import glob 
from astropy.stats import sigma_clip
from tqdm import tqdm
import re
from skimage.restoration import inpaint_biharmonic
import torch
import matplotlib.pyplot as plt
from scipy.signal import medfilt
#import ruptures as rpt
from scipy.signal import savgol_filter


def ADC_convert(signal, gain, offset):
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead_slow(signal, dead, dark):
    hot = sigma_clip(
        dark, sigma=5, maxiters=5
    ).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

import numpy as np
from astropy.stats import sigma_clip

def mask_hot_dead(signal, dead, dark):
    # Compute hot mask only once, cache if multiple calls with same dark
    hot_mask_2d = sigma_clip(dark, sigma=5, maxiters=5).mask
    
    # Broadcast masks without np.tile to save memory/time
    # shape (time, x, y)
    hot_mask = np.broadcast_to(hot_mask_2d, signal.shape)
    dead_mask = np.broadcast_to(dead, signal.shape)
    
    combined_mask = np.logical_or(hot_mask, dead_mask)
    
    # Construct new masked array in one step
    masked_signal = np.ma.array(signal, mask=combined_mask)
    
    return masked_signal

def apply_linear_corr(linear_corr,clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark_old(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark* dt[:, np.newaxis, np.newaxis]
    return signal

def clean_dark(signal, dead, dark, dt):
    masked_dark = np.ma.masked_array(dark, mask=dead)  # mask dark where dead
    # Broadcast dark along time axis and multiply by dt, then subtract
    signal -= masked_dark * dt[:, np.newaxis, np.newaxis]
    return signal

def get_cds(signal):
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds

def correct_flat_field_old(flat,dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal
    
def correct_flat_field(flat,dead, signal):
    flat = flat.transpose(1, 0)  # shape (cols, rows)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = flat[np.newaxis, :, :]  # add axis for broadcast
    signal = signal / flat  # broadcasting instead of tile
    return signal


# from ruptures https://centre-borelli.github.io/ruptures-docs/code-reference/detection/binseg-reference/#ruptures.detection.binseg.Binseg
import abc
from functools import lru_cache

from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def sanity_check(n_samples, n_bkps, jump, min_size):
    """Check that segmentation parameters are valid.

    Args:
        n_samples (int): number of samples in the signal
        n_bkps (int): number of requested breakpoints
        jump (int): subsample jump size
        min_size (int): minimum segment size

    Returns:
        bool: True if parameters are consistent; False otherwise
    """
    if n_samples < 1:
        return False
    if n_bkps < 0:
        return False
    if jump < 1:
        return False
    if min_size < 1:
        return False
    # Check that at least one segment can be formed
    if n_samples < (n_bkps + 1) * min_size:
        return False
    return True


class BaseCost(object, metaclass=abc.ABCMeta):
    """Base class for all segment cost classes.

    Notes:
        All classes should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``).
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Set the parameters of the cost function, for instance the Gram
        matrix, etc."""
        pass

    @abc.abstractmethod
    def error(self, start, end):
        """Returns the cost on segment [start:end]."""
        pass

    def sum_of_costs(self, bkps):
        """Returns the sum of segments cost for the given segmentation.

        Args:
            bkps (list): list of change points. By convention, bkps[-1]==n_samples.

        Returns:
            float: sum of costs
        """
        soc = sum(self.error(start, end) for start, end in pairwise([0] + bkps))
        return soc

    @property
    @abc.abstractmethod
    def model(self):
        pass

def cost_factory(model="l2", **params):
    # Example placeholder. You must implement cost classes (like CostL2) yourself!
    if model == "l2":
        return CostL2(**params)
    elif model == "l1":
        return CostL1(**params)
    # Add other models as needed
    else:
        raise ValueError(f"Unknown model '{model}'")

class CostL2(BaseCost):
    r"""Least squared deviation."""

    model = "l2"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 1

    def fit(self, signal) -> "CostL2":
        """Set parameters of the instance.

        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        return self

    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        return self.signal[start:end].var(axis=0).sum() * (end - start)

class BaseEstimator(metaclass=abc.ABCMeta):
    """Base class for all change point detection estimators.

    Notes:
        All estimators should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``).
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """To call the segmentation algorithm."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """To call the segmentation algorithm."""
        pass

    @abc.abstractmethod
    def fit_predict(self, *args, **kwargs):
        """To call the segmentation algorithm."""
        pass

class Binseg(BaseEstimator):
    """Binary segmentation."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Binseg instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf",...]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5 samples.
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None
        self.signal = None

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        # initialization
        bkps = [self.n_samples]
        stop = False
        while not stop:
            stop = True
            new_bkps = [
                self.single_bkp(start, end) for start, end in pairwise([0] + bkps)
            ]
            bkp, gain = max(new_bkps, key=lambda x: x[1])

            if bkp is None:  # all possible configuration have been explored.
                break

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                error = self.cost.sum_of_costs(bkps)
                if error > epsilon:
                    stop = False

            if not stop:
                bkps.append(bkp)
                bkps.sort()
        partition = {
            (start, end): self.cost.error(start, end)
            for start, end in pairwise([0] + bkps)
        }
        return partition

    @lru_cache(maxsize=None)
    def single_bkp(self, start, end):
        """Return the optimal breakpoint of [start:end] (if it exists)."""
        segment_cost = self.cost.error(start, end)
        if np.isinf(segment_cost) and segment_cost < 0:  # if cost is -inf
            return None, 0
        gain_list = list()
        for bkp in range(start, end, self.jump):
            if bkp - start >= self.min_size and end - bkp >= self.min_size:
                gain = (
                    segment_cost
                    - self.cost.error(start, bkp)
                    - self.cost.error(bkp, end)
                )
                gain_list.append((gain, bkp))
        try:
            gain, bkp = max(gain_list)
        except ValueError:  # if empty sub_sampling
            return None, 0
        return bkp, gain

    def fit(self, signal) -> "Binseg":
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples, _ = self.signal.shape
        self.cost.fit(signal)
        self.single_bkp.cache_clear()

        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the
        signal passed to [`fit()`][ruptures.detection.binseg.Binseg.fit].
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            pen (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Raises:
            AssertionError: if none of `n_bkps`, `pen`, `epsilon` is set.
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.
            pen (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)


## we will start by getting the index of the training data:
def get_index(files,CHUNKS_SIZE ):
    index = []
    for file in files :
        file_name = file.split('/')[-1]
        if file_name.split('_')[0] == 'AIRS-CH0' and file_name.split('_')[-1] == '0.parquet':
            file_index = os.path.basename(os.path.dirname(file))
            index.append(int(file_index))
    index = np.array(index)
    index = np.sort(index) 
    # credit to DennisSakva
    index=np.array_split(index, len(index)//CHUNKS_SIZE)
    
    return index

def get_multiobs_index(files, CHUNKS_SIZE):
    """
    Extract (planet_id, obs_num) pairs from AIRS-CH0_signal_X.parquet files.
    Returns: list of (planet_id, obs_num) tuples in sorted order, split into chunks.
    """
    index = []
    # Regex: AIRS-CH0_signal_{obs}.parquet
    pattern = re.compile(r'^AIRS-CH0_signal_(\d+)\.parquet$')
    for file in files:
        file_name = os.path.basename(file)
        match = pattern.match(file_name)
        if match:
            planet_id = os.path.basename(os.path.dirname(file))
            obs_num = int(match.group(1))
            index.append((int(planet_id), obs_num))
    # Optional: sort by planet then obs number
    index.sort()
    # Remove duplicates in case of any
    index = list(dict.fromkeys(index))
    if len(index) >= CHUNKS_SIZE and CHUNKS_SIZE > 0:
        index_chunks = np.array_split(index, len(index)//CHUNKS_SIZE)
    else:
        index_chunks = [index]
    return index_chunks

def bin_obs(arr, binning, axis=1):
    # Ensure input is a masked array
    bin_size = binning
    arr = np.ma.masked_array(arr)
    shape = list(arr.shape)
    n_bins = shape[axis] // bin_size
    new_shape = shape[:axis] + [n_bins, bin_size] + shape[axis+1:]
    arr_reshaped = np.ma.reshape(arr, new_shape)
    # Now sum along the bin_size axis, which is axis=axis+1
    return np.ma.sum(arr_reshaped, axis=axis+1)

def median_filter_time(masked_arr, kernel_size=3):
    """Apply 1D median filter (default: size 3) along time axis (axis=1) for each batch.
    Ignores masked voxels; uses available neighbors at edges. Preserves masked array structure."""
    assert kernel_size % 2 == 1, "Kernel size must be odd!"
    batch_dim, time_dim, X, Y = masked_arr.shape
    pad = kernel_size // 2
    result = np.ma.masked_all(masked_arr.shape, dtype=masked_arr.dtype)
    arr_data = masked_arr.data
    arr_mask = masked_arr.mask

    for b in range(batch_dim):
        for t in range(time_dim):
            lo = max(0, t - pad)
            hi = min(time_dim, t + pad + 1)
            window = arr_data[b, lo:hi, :, :]
            window_mask = arr_mask[b, lo:hi, :, :]
            window_ma = np.ma.masked_array(window, mask=window_mask)
            # Use np.ma.median as a function for better compatibility
            median_vals = np.ma.median(window_ma, axis=0)
            result.data[b, t] = median_vals.data
            result.mask[b, t] = median_vals.mask

    return result

def already_saved(chunk_name, path_out):
    airs_file = os.path.join(path_out, f'AIRS_clean_train_{chunk_name}.pt')
    fgs1_file = os.path.join(path_out, f'FGS1_clean_train_{chunk_name}.pt')
    return os.path.exists(airs_file) and os.path.exists(fgs1_file)

def median_filter_and_downsample(
    signal,
    median_filter_window=101,
    stride=10,
    title='Median Filtered and Downsampled Signal',
    plot = True
):
    """
    Applies a median filter to a 1D signal, crops edges, downsamples by specified stride, 
    and plots the result. Returns the downsampled signal and its x coordinates.
    """
    # Apply median filter (pads internally)
    window_size = median_filter_window  # must be odd
    border = (window_size - 1) // 2

    median_filtered_full = medfilt(signal, kernel_size=window_size)

    # Crop edges to remove padding artifacts
    median_filtered_cropped = median_filtered_full[border:-border]
    x_cropped = np.arange(border, len(signal) - border)

    # Downsample
    downsampled_signal = median_filtered_cropped[::stride]
    x_downsampled = x_cropped[::stride]

    if plot:
        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(x_cropped, median_filtered_cropped, label=f'Median Filtered (window={window_size}, stride=1)', linewidth=2)
        plt.plot(x_downsampled, downsampled_signal, marker='o', linestyle='--',
                 label=f'Filtered & Downsampled (stride={stride})')
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return downsampled_signal, x_downsampled

def plot_transit_edges(
    signal,
    window_length=15,
    polyorder=2,
    window=5,
    percentile=30,
    min_size=10,
    title="Local Transit Edges Detection (Split at Min)",
    plot_raw=True,
    plot = True
):
    """
    Plots transit edges and detected change points for a 1D signal.
    Returns onset index (left edge), offset index (right edge).
    """
    # Smoothing
    smoothed_signal = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
    #smoothed_signal = signal
    
    # Find global minimum (likely transit midpoint)
    min_index = np.argmin(smoothed_signal)

    # Split signal at minimum
    signal_left = smoothed_signal[:min_index]
    signal_right = smoothed_signal[min_index:]

    # Detect on left half (before transit: drop)
    n_bkps = 1
    algo_left = Binseg(model="l2", min_size=min_size).fit(signal_left)
    bkps_left = algo_left.predict(n_bkps=n_bkps)
    change_left = bkps_left[0]

    onset_left = find_transit_edge_local(signal_left, change_left, find_onset=True, window=window, percentile=percentile)

    # Detect on right half (after transit: rise)
    algo_right = Binseg(model="l2", min_size=min_size).fit(signal_right)
    bkps_right = algo_right.predict(n_bkps=n_bkps)
    change_right = bkps_right[0]

    offset_right = find_transit_edge_local(signal_right, change_right, find_onset=False, window=window, percentile=percentile)
    offset_right_global = min_index + offset_right

    # (Optional) change points for info
    midpoints = [change_left, min_index + change_right]

    # Plot for confirmation
    if plot:
        plt.figure(figsize=(12, 6))
        if plot_raw:
            plt.plot(signal, label='Raw signal', color='gray', alpha=0.4)
        plt.plot(smoothed_signal, label='Smoothed signal', color='navy')
        plt.axvline(min_index, color='black', linestyle='--', label='Transit Min')

        plt.axvline(onset_left, color='green', linestyle='-', label='Onset (start drop)', lw=3)
        plt.scatter([onset_left], smoothed_signal[[onset_left]], color='green', s=80, zorder=10)

        plt.axvline(offset_right_global, color='red', linestyle='-', label='Offset (end rise)', lw=3)
        plt.scatter([offset_right_global], smoothed_signal[[offset_right_global]], color='red', s=80, zorder=10)

        plt.axvline(midpoints[0], color='purple', linestyle='--', label='Change point (start)')
        plt.axvline(midpoints[1], color='purple', linestyle='--', label='Change point (end)')

        plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Value')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    #print(f"Onset index (left side): {onset_left}")
    #print(f"Offset index (right side, global): {offset_right_global}")
    return onset_left, offset_right_global, min_index, np.min(smoothed_signal)

def find_transit_edge_local(signal, change_point, find_onset=True, window=5, percentile=80):
    if find_onset:
        region = signal[:change_point]
        threshold = np.percentile(region, percentile)
        for i in range(change_point, window, -1):
            if np.all(signal[i-window:i] >= threshold):
                return i
        return window
    else:
        region = signal[change_point:]
        threshold = np.percentile(region, percentile)
        for i in range(change_point, len(signal)-window):
            if np.all(signal[i:i+window] >= threshold):
                return i
        return len(signal)-window

def fit_and_plot_baseline(
    signal,
    onset_idx,
    offset_idx,
    delta=0,
    degree=2,
    planet_id=None,
    plot=True,
    title='Baseline Fit'
):
    """
    Fit a polynomial baseline curve to regions outside [onset_idx, offset_idx],
    with delta applied to edges. Plots result optionally.
    Returns: fitted_curve, coeffs, idx_baseline
    """
    # Adjust edges with delta
    phase1 = max(0, onset_idx - delta)
    phase2 = min(len(signal), offset_idx + delta)
    
    # Indices for left and right baseline regions
    idx_left = np.arange(0, phase1)
    idx_right = np.arange(phase2, len(signal))
    idx_baseline = np.concatenate([idx_left, idx_right])
    y_baseline = signal[idx_baseline]
    
    # Get a boolean mask for valid values (not NaN, not Inf)
    valid_mask = (~np.isnan(y_baseline)) & (~np.isinf(y_baseline))

    # Filter both arrays
    idx_baseline = idx_baseline[valid_mask]
    y_baseline = y_baseline[valid_mask]
    
    # Fit polynomial
    coeffs = np.polyfit(idx_baseline, y_baseline, deg=degree)
    poly = np.poly1d(coeffs)
    fitted_curve = poly(np.arange(len(signal)))
    
    # Plotting
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(signal, label='Signal')
        plt.plot(fitted_curve, '--', label='Fitted Baseline', color='orange')
        plt.scatter(idx_baseline, signal[idx_baseline], color='green', label='Baseline Points')
        plt.axvline(onset_idx, color='r', linestyle='--', label='Transit Onset', lw=2)
        plt.axvline(offset_idx, color='b', linestyle='--', label='Transit Offset', lw=2)
        plt.axvspan(phase1, min(len(signal), onset_idx + delta), color='r', alpha=0.2, label='Delta Onset')
        plt.axvspan(max(0, offset_idx - delta), phase2, color='b', alpha=0.2, label='Delta Offset')
        plt.legend()
        sub_id = f" for Planet ID {planet_id}" if planet_id is not None else ""
        plt.title(f'{title}{sub_id}')
        plt.xlabel('Sample Index')
        plt.ylabel('Signal')
        plt.tight_layout()
        plt.show()
    
    return fitted_curve, coeffs, idx_baseline


path_folder = '/kaggle/input/ariel-data-challenge-2025' # path to the folder containing the data
path_out = '/kaggle/working/processed_datak'
os.makedirs(path_out, exist_ok=True)
files = glob.glob(os.path.join(path_folder, 'test','*','*'))

CHUNKS_SIZE = 1
index_chunks = get_multiobs_index(files, CHUNKS_SIZE)

train_adc_info = pd.read_csv(os.path.join(path_folder, 'adc_info.csv'))
axis_info = pd.read_parquet(os.path.join(path_folder,'axis_info.parquet'))
DO_MASK = True
DO_THE_NL_CORR = False
DO_DARK = True
DO_FLAT = True
TIME_BINNING = True
FILT = False

cut_inf, cut_sup = 0, 356
l = cut_sup - cut_inf
count = 0


import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

constants = torch.load('/kaggle/input/norm05/normalization_constants05.pt')
constantMEANS = torch.tensor([constants["MRS"],constants["MMS"],constants['MTS'],constants['MMP'],constants['MP'],constants['MSMA'],constants['MI']]).to(device)
constantSTDS = torch.tensor([constants["SRS"],constants["SMS"],constants['STS'],constants['SMP'],constants['SP'],constants['SSMA'],constants['SI']]).to(device)
constantMMD = constants["MMD"].to(device)
constantSMD = constants["SMD"].to(device)
constantMSD = constants["MSD"].to(device)
constantSSD = constants["SSD"].to(device)

def get_planet_values(star_info_df, planet_id_list):
    columns = ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']
    filtered_df = star_info_df[star_info_df['planet_id'].isin(planet_id_list)]
    values_list = filtered_df[columns].values.tolist()
    return values_list

class MiniReducerSpatialBoth(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Collapse both spatial axes: (32,32) -> (1,32)
        self.conv_spatial = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(1, 27, 27),
            stride=(1, 1, 1),
            padding=0
        )
        # Linear projection of flattened spatial patch to 32 features
        self.lin = nn.Linear(36, 32)

    def forward(self, x):
        #print(f"Input: {x.shape}")                  # (B, C, T, 32, 32)
        x = self.conv_spatial(x)
        #print(f"After spatial collapse: {x.shape}") # (B, C, T, H', W')
        x = F.relu(x)
        B, C, T, H, W = x.shape
        x = x.view(B, C, T, H * W)                  # Flatten spatial dims
        x = x.reshape(-1, H * W)                    # Merge (B, C, T) for linear layer
        x = self.lin(x)                             # Project to 32
        x = x.view(B, C, T, 32)                     # Restore shape
        x = x.unsqueeze(3)                          # (B, C, T, 1, 32)
        #print(x.shape)
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel, pool_stride,
                 dilation=1, circular_pad_wavelength=False):
        super().__init__()
        # Allow tuple for dilation (pythonic for axis control)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        # Compute per-axis padding
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        pad = tuple(d * (k // 2) for d, k in zip(dilation, kernel_size))  # (D, H, W)
        self.pad = pad
        self.circular_pad_wavelength = circular_pad_wavelength

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=0 if circular_pad_wavelength else pad,  # all-zero padding applied manually if circular
            dilation=dilation
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)
        self.match_channels = None
        if in_channels != out_channels:
            self.match_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = x
        out = x
        # Apply circular padding ONLY to wavelength axis
        if self.circular_pad_wavelength:
            # self.pad = (pad_D, pad_H, pad_W)
            pad_D, pad_H, pad_W = self.pad
            # F.pad expects (W_left, W_right, H_top, H_bottom, D_front, D_back)
            # Wavelength=H axis (axis=3)
            out = F.pad(out, (0,0, pad_H, pad_H, 0,0), mode="circular")
            # add zero padding on depth and width if needed
            if pad_D > 0 or pad_W > 0:
                out = F.pad(out, (pad_W, pad_W, 0,0, pad_D, pad_D), mode="constant", value=0)
        out = self.conv(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pool(out)
        identity_pooled = self.pool(identity)
        if self.match_channels:
            identity_pooled = self.match_channels(identity_pooled)
        out = out + identity_pooled
        return out


class Custom2DCNN(nn.Module):
    def __init__(self, in_channels=2,
                 conv_kernel_sizes=[3,3,3,3,3,3],
                 conv_filters=[32,64,128,256,512,1028],
                 pool_kernel_sizes=[[16,1,4],[16,1,4],[16,1,4],[16,1,4],[16,1,4],[16,1,4]],
                 pool_strides=[[8,1,2],[8,1,2],[8,1,2],[8,1,2],[8,1,2],[8,1,2]],
                 dilation_rates=None,
                 circular_pad_wavelength_layers=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        chans = in_channels
        if dilation_rates is None:
            dilation_rates = [1] * len(conv_filters)
        if circular_pad_wavelength_layers is None:
            circular_pad_wavelength_layers = [False] * len(conv_filters)
        for idx in range(len(conv_filters)):
            self.blocks.append(
                ResidualBlock3D(
                    chans, conv_filters[idx],
                    kernel_size=conv_kernel_sizes[idx],
                    pool_kernel=pool_kernel_sizes[idx],
                    pool_stride=pool_strides[idx],
                    dilation=dilation_rates[idx],
                    circular_pad_wavelength=circular_pad_wavelength_layers[idx]
                )
            )
            chans = conv_filters[idx]
        self.fc1 = nn.Linear(chans+357+357+7, 512)
        self.fc_x = nn.Linear(512, 283)
        self.fc_y = nn.Linear(512, 283)
        self.fc_sigma = nn.Linear(512, 283)
        #self.fgs1_reducer = MiniReducerSpatialBoth(in_channels=2, out_channels=2)
    def forward(self, data, planet_data):
        xmean = torch.mean(data,dim=3).unsqueeze(3)
        xstd = torch.std(data,dim=3).unsqueeze(3)
        x = (data-xmean)/xstd
        xm = (xmean - constantMMD.view(1, 1, 357, 1))/constantSMD.view(1, 1, 357, 1)
        xs = (xstd - constantMSD.view(1, 1, 357, 1))/constantSSD.view(1, 1, 357, 1)
        
        planet_data = (planet_data - constantMEANS)/constantSTDS
        
        #x = torch.cat([airs, fgs1], dim=3)  # dim=3 is the 4th dimension (zero-based counting)
        #print(x.shape, xmean.shape,xstd.shape,xm.shape,xs.shape)  # Should show: torch.Size([1, 2, 5625, 357, 32])
        
        x = torch.transpose(x,2,3)
        x = x.unsqueeze(4)
        
        #print(f"Input: {x.shape}")
        for idx, block in enumerate(self.blocks):
            x = block(x)
            #print(f"After Residual Block {idx+1}: {x.shape}")
        x = nn.functional.adaptive_avg_pool3d(x, 1)
        #print(f"After global avg pool: {x.shape}")
        x = x.view(x.size(0), -1)

        #print("before cat",x.shape)
        ###concatenate params
        #print(x.shape,xm.shape,xs.shape,planet_data.shape)
        #x = torch.cat([x,xm.squeeze().unsqueeze(0),xs.squeeze().unsqueeze(0),planet_data.squeeze(1)], dim = 1)
        x = torch.cat([x,xm.squeeze(),xs.squeeze(),planet_data.squeeze(1)], dim = 1)

        #print("after cat",x.shape)

        
        x = F.relu(self.fc1(x))
        #print(f"After shared FC1: {x.shape}")
        indices = [0] + list(range(39, 321))
        xstd = xstd[:, 0, indices, 0]
        xmean = xmean[:, 0, indices, 0]
        pred_x = (self.fc_x(x)*xstd)+ xmean
        pred_y = (self.fc_y(x)*xstd)+ xmean
        #print(f"After prediction head: {pred_y.shape}")
        log_sigma = self.fc_sigma(x)
        #print(f"After sigma head: {log_sigma.shape}")
        return (pred_y-pred_x)/pred_y, log_sigma

star_info_df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/test_star_info.csv")


import pandas as pd

sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")  # or .txt if appropriate
columns = sample_submission.columns.tolist()


def weighted_avg_and_uncertainty(grp):
    preds = grp[wl_cols].to_numpy(dtype=float)
    sigmas = grp[sigma_cols].to_numpy(dtype=float)
    weights = 1.0 / (sigmas ** 2)
    weighted_preds = np.sum(preds * weights, axis=0) / np.sum(weights, axis=0)
    sigma_new = 1.0 / np.sqrt(np.sum(weights, axis=0))
    result = pd.Series(np.concatenate([weighted_preds, sigma_new]))
    return result

def mean_of_smallest_sigma_rows(group, wl_cols, sigma_cols, n=10):
    # Explicitly drop grouping columns to avoid warning
    group_values = group[wl_cols + sigma_cols].copy()
    group_values['row_sigma'] = group_values[sigma_cols].mean(axis=1)
    smallest = group_values.nsmallest(n, 'row_sigma')
    return smallest[wl_cols + sigma_cols].mean()

def weighted_avg_and_uncertainty_ii(grp):
    preds = grp[wl_cols].to_numpy(dtype=float)
    sigmas = grp[sigma_cols].to_numpy(dtype=float)

    weights = 1.0 / (sigmas ** 2)
    weighted_preds = np.sum(preds * weights, axis=0) / np.sum(weights, axis=0)

    # Basic average of sigmas, not weighted
    sigma_new = np.exp(np.mean(np.log(sigmas), axis=0))

    result = pd.Series(np.concatenate([weighted_preds, sigma_new]))
    return result



start_loading = time.perf_counter()

rows = []
for index_chunk in  index_chunks:
    AIRS_CH0_clean = np.ma.MaskedArray(np.zeros((CHUNKS_SIZE, 11250, 32, l)))
    FGS1_clean = np.ma.MaskedArray(np.zeros((CHUNKS_SIZE, 135000, 32, 32)))
    
    chunk_name = '__'.join([f"{pid}_{obs}" for pid, obs in index_chunk])
    
    if already_saved(chunk_name, path_out):
            print(f"Skipping {chunk_name} (already processed)")
            continue  # Go to next chunk
    print(chunk_name)
    
    for i in range (CHUNKS_SIZE) : 
        df = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/AIRS-CH0_signal_{index_chunk[i][1]}.parquet'))
        signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
        gain = train_adc_info['AIRS-CH0_adc_gain'][0]
        offset = train_adc_info['AIRS-CH0_adc_offset'][0]
        signal = ADC_convert(signal, gain, offset)
        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1
        chopped_signal = signal[:, :, cut_inf:cut_sup]
        del signal, df
        
        # CLEANING THE DATA: AIRS
        flat = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dark = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dead_airs = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        linear_corr = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
        
        if DO_MASK:
            chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
            AIRS_CH0_clean[i] = chopped_signal
        else:
            AIRS_CH0_clean[i] = chopped_signal
            
        if DO_THE_NL_CORR: 
            linear_corr_signal = apply_linear_corr(linear_corr,AIRS_CH0_clean[i])
            AIRS_CH0_clean[i,:, :, :] = linear_corr_signal
        del linear_corr
        
        if DO_DARK: 
            cleaned_signal = clean_dark(AIRS_CH0_clean[i], dead_airs, dark, dt_airs)
            AIRS_CH0_clean[i] = cleaned_signal
        else: 
            pass
        del dark
        
        df = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/FGS1_signal_{index_chunk[i][1]}.parquet'))
        fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))
        
        FGS1_gain = train_adc_info['FGS1_adc_gain'][0]
        FGS1_offset = train_adc_info['FGS1_adc_offset'][0]
        
        fgs_signal = ADC_convert(fgs_signal, FGS1_gain, FGS1_offset)
        dt_fgs1 = np.ones(len(fgs_signal))*0.1
        dt_fgs1[1::2] += 0.1
        chopped_FGS1 = fgs_signal
        del fgs_signal, df
        
        # CLEANING THE DATA: FGS1
        flat = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        dark = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/dark.parquet')).values.astype(np.float64).reshape((32, 32))
        dead_fgs1 = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/dead.parquet')).values.astype(np.float64).reshape((32, 32))
        linear_corr = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))
        
        if DO_MASK:
            chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
            FGS1_clean[i] = chopped_FGS1
        else:
            FGS1_clean[i] = chopped_FGS1

        if DO_THE_NL_CORR: 
            linear_corr_signal = apply_linear_corr(linear_corr,FGS1_clean[i])
            FGS1_clean[i,:, :, :] = linear_corr_signal
        del linear_corr
        
        if DO_DARK: 
            cleaned_signal = clean_dark(FGS1_clean[i], dead_fgs1, dark,dt_fgs1)
            FGS1_clean[i] = cleaned_signal
        else: 
            pass
        del dark
        
    # SAVE DATA AND FREE SPACE
    AIRS_cds = get_cds(AIRS_CH0_clean)
    FGS1_cds = get_cds(FGS1_clean)

    del AIRS_CH0_clean, FGS1_clean

    if FILT:
        AIRS_cds = median_filter_time(AIRS_cds)
        FGS1_cds = median_filter_time(FGS1_cds)
    
    ## (Optional) Time Binning to reduce space
    if TIME_BINNING:
        AIRS_cds_binned = bin_obs(AIRS_cds,binning=1)
        FGS1_cds_binned = bin_obs(FGS1_cds,binning=12*1)
    else:
        #AIRS_cds = AIRS_cds.transpose(0,1,3,2) ## this is important to make it consistent for flat fielding, but you can always change it
        AIRS_cds_binned = AIRS_cds
        #FGS1_cds = FGS1_cds.transpose(0,1,3,2)
        FGS1_cds_binned = FGS1_cds
    AIRS_cds_binned = AIRS_cds_binned.transpose(0,1,3,2)
    FGS1_cds_binned = FGS1_cds_binned.transpose(0,1,3,2)
    del AIRS_cds, FGS1_cds
    
    for i in range (CHUNKS_SIZE):
        flat_airs = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        flat_fgs = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        if DO_FLAT:
            corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[i])
            AIRS_cds_binned[i] = corrected_AIRS_cds_binned
            corrected_FGS1_cds_binned = correct_flat_field(flat_fgs,dead_fgs1, FGS1_cds_binned[i])
            FGS1_cds_binned[i] = corrected_FGS1_cds_binned
        else:
            pass

    AIRS_cds_binned = AIRS_cds_binned.transpose(0,1,3,2)
    FGS1_cds_binned = FGS1_cds_binned.transpose(0,1,3,2)
    
    # Example: FGS1_cds (shape: [time, x, y]) -- inpaint along time as channels
    # Suppose you have a masked array: FGS1_cds (time, x, y), with mask True for bad voxels
    
    # Convert to plain array and mask for inpainting
    data = FGS1_cds_binned[0,:,:,:].data         # shape: (time, x, y)
    mask = FGS1_cds_binned[0,0,:,:].mask         # shape: (x, y)
    data = data.transpose(1,2,0)                 # shape: (x, y, time)
    nan_mask = np.sum(np.isnan(data))
    if nan_mask:
        print("data contains nan NANANANANANANANANANA")
   
    # Inpaint, treating time as channels (axis=0)
    result_fgs1 = inpaint_biharmonic(data, mask, channel_axis=2)
    result_fgs1 = result_fgs1.transpose(2,0,1)

    data_airs = AIRS_cds_binned[0,:,:,:].data      # shape: (time, x, lambda)
    mask_airs = AIRS_cds_binned[0,0,:,:].mask      # shape: (x, lambda)
    data_airs = data_airs.transpose(1,2,0)          # shape: (x, lambda, time)
    # Inpaint, treating wavelength as channels (axis=0)
    result_airs = inpaint_biharmonic(data_airs, mask_airs, channel_axis=2)
    result_airs = result_airs.transpose(2,0,1)

    #data_3d_airs = torch.from_numpy(result_airs)      # shape: [frames, x, y]
    #mask_2d_airs = torch.from_numpy(AIRS_cds_binned.mask[0,0,:,:])      # shape: [x, y]
    #data_3d_fgs1 = torch.from_numpy(result_fgs1)      # shape: [frames, x, y]
    #mask_2d_fgs1 = torch.from_numpy(FGS1_cds_binned.mask[0,0,:,:])      # shape: [x, y]

    #sum spatial dimension
    result_fgs1 = np.sum(result_fgs1, axis=(1, 2))
    result_airs = np.sum(result_airs, axis=1)
    #print(result_fgs1.shape,result_airs.shape)

    xmins = []
    polys = []
    datas = []
    
    #median filter
    result_fgs1, xcrop = median_filter_and_downsample(result_fgs1, median_filter_window=101, stride=1, plot=False)
    #print(result_fgs1.shape,result_airs.shape)
    #find change points
    #try:
    onset, offset, mind, xmin = plot_transit_edges(result_fgs1, plot=False)
    linear = False
    #print('success')
    #fit P
    fitted_curve, coeffs, idx_baseline = fit_and_plot_baseline(result_fgs1,onset,offset,delta=10,degree=2,planet_id=None,plot=False)
    datas.append(result_fgs1)
    xmins.append(xmin)
    polys.append(fitted_curve)
    #except:
        #if not do linear
        #linear = True

    for wl in range(result_airs.shape[1]):
        signal = result_airs[:, wl]
        signal, xcrop = median_filter_and_downsample(signal, median_filter_window=101, stride=1, plot=False)
        #fitted_curve, coeffs, idx_baseline = fit_and_plot_baseline(
        #    signal,
        #    onset,
        #    offset,
        #    delta=10,
        #    degree=2,
        #    planet_id=None,
        #    plot=False
        #)
        #smoothed_signal = savgol_filter(signal, window_length=15, polyorder=2)
        datas.append(signal)
        #xmins.append(smoothed_signal[mind])
        #polys.append(fitted_curve)
    
    #save

    datas_tensor = torch.from_numpy(np.stack(datas))   # Shape: (num_arrays, array_length)
    polys_tensor = torch.from_numpy(np.stack(polys))   # Shape: (num_arrays, array_length)
    
    # Convert list of scalars to 1D tensor
    xmins_tensor = torch.tensor(xmins)                  # Shape: (num_scalars,)
    
    #torch.save({'data': datas_tensor, 'poly': polys_tensor, 'xmin': xmins_tensor, 'mind':torch.tensor(mind)}, os.path.join(path_out, f'clean_train_{chunk_name}.pt'))
    
    
    model = Custom2DCNN(
        in_channels=1,
        conv_kernel_sizes=[3,3,3,3,3,3],
        conv_filters=[16,32,64,128,256,512],
        pool_kernel_sizes=[[4,2,1],[4,2,1],[4,2,1],[4,2,1],[2,2,1],[2,2,1]],
        pool_strides=[[2,1,1],[2,1,1],[2,1,1],[2,1,1],[2,1,1],[2,1,1]],
        dilation_rates=[[1,1,1], [1,3,1], [1,9,1], [1,27,1], [1,27*3,1], [1,27*9,1]],
        circular_pad_wavelength_layers=[True, True, True, True, True, True]
    ).to(device)
    checkpoint = torch.load("/kaggle/input/runc1_0-best/pytorch/default/1/best_model (1).pth", weights_only=False).to(device)
    model = checkpoint

    offsets = torch.arange(5)  # tensor([0,1,2,3,4])
    stride = 10
    slices = [datas_tensor[:, offset::stride] for offset in offsets]  # list of 5 tensors, shape [wavelength, time_downsampled]
    # Get the flipped (time domain reversed) versions of these slices
    #slices_flipped = [torch.flip(s, dims=[1]) for s in slices]  # assume time dimension is dim=1
    
    #all_slices = slices + slices_flipped
    #print(all_slices)
    # Stack into batch dimension (dim=0)
    data = torch.stack(slices, dim=0) 
    
    planet_data = torch.tensor(get_planet_values(star_info_df, [index_chunk[i][0]]))
    #data = data.unsqueeze(0) 
    data = data.unsqueeze(1) 
    planet_data.unsqueeze(0)
    planet_data = planet_data.repeat(5, 1)
    model.eval()
    with torch.no_grad():
        y_pred, log_sigma = model(data.float().to(device), planet_data.float().to(device))
    #print(y_pred.shape,log_sigma.shape)

    y_pred[y_pred < 0] = 0

    batch_size = y_pred.shape[0]  # e.g., 5
    planet_id = index_chunk[i][0]  # Same planet_id for entire batch
    
    for batch_i in range(batch_size):
        row_dict = {}
        row_dict['planet_id'] = planet_id  # Same for all batch items
        sigma_exp = torch.exp(log_sigma)
        for j in range(1, 284):
            row_dict[f'wl_{j}'] = y_pred[batch_i, j-1].item()
            row_dict[f'sigma_{j}'] = sigma_exp[batch_i, j-1].item()
        #for k in range(7):
        rows.append(row_dict)

    checkpoint = torch.load("/kaggle/input/c1f3/pytorch/default/1/best_model.pth", weights_only=False).to(device)
    model = checkpoint
    model.eval()
    with torch.no_grad():
        y_pred, log_sigma = model(data.float().to(device), planet_data.float().to(device))
    #print(y_pred.shape,log_sigma.shape)

    y_pred[y_pred < 0] = 0

    batch_size = y_pred.shape[0]  # e.g., 5
    planet_id = index_chunk[i][0]  # Same planet_id for entire batch
    
    for batch_i in range(batch_size):
        row_dict = {}
        row_dict['planet_id'] = planet_id  # Same for all batch items
        sigma_exp = torch.exp(log_sigma)
        for j in range(1, 284):
            row_dict[f'wl_{j}'] = y_pred[batch_i, j-1].item()
            row_dict[f'sigma_{j}'] = sigma_exp[batch_i, j-1].item()
        rows.append(row_dict)
    
    
    #print(chunk_name, count)
    del AIRS_cds_binned
    del FGS1_cds_binned
    count +=1

submission_df = pd.DataFrame(rows, columns=columns)

### BEST SIGMA
# Find index of row with minimum average sigma for each planet_id
#submission_df['mean_sigma'] = submission_df[[f'sigma_{i}' for i in range(1, 284)]].mean(axis=1)
#df_best = submission_df.loc[submission_df.groupby('planet_id')['mean_sigma'].idxmin()].drop(columns='mean_sigma')

### BASIC AVERAGE
# List of columns for predictions and sigmas
wl_cols = [f'wl_{i}' for i in range(1, 284)]
sigma_cols = [f'sigma_{i}' for i in range(1, 284)]

# Take simple mean for each group (planet_id)
df_best = (
    submission_df.groupby('planet_id')[wl_cols + sigma_cols]
    .mean()
    .reset_index()
)

### BEST N AVERAGE
#df_best = (
#    submission_df.groupby('planet_id')
#    .apply(mean_of_smallest_sigma_rows, wl_cols=wl_cols, sigma_cols=sigma_cols, n=10)
#    .reset_index()
#)

### FANCY AVERAGE (II)
# Get column names for output
#output_cols = wl_cols + sigma_cols

#df_best = (
#    submission_df.groupby('planet_id')[wl_cols + sigma_cols].apply(weighted_avg_and_uncertainty_ii)
#    .reset_index()
#)
#df_best.columns = ['planet_id'] + output_cols


df_best.to_csv("submission.csv", index=False)

loading_time = time.perf_counter() - start_loading
print(f"Start processing data time: {loading_time:.4f} sec")


#torch.save({'data': torch.from_numpy(result_airs)}, os.path.join(path_out, f'AIRS_clean_train_{chunk_name}.pt'))
df_best


#submission_df

