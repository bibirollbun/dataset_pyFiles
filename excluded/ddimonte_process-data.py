!pip install ruptures
!pip install ldtk


pip install batman-package



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
import ruptures as rpt
from scipy.signal import savgol_filter
import seaborn as sns
import batman
from ldtk import LDPSetCreator, BoxcarFilter


def ADC_convert(signal, gain, offset):
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(
        dark, sigma=5, maxiters=5
    ).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def apply_linear_corr(linear_corr,clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark* dt[:, np.newaxis, np.newaxis]
    return signal

def get_cds(signal):
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds

def correct_flat_field(flat,dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal


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
    algo_left = rpt.Binseg(model="l2", min_size=min_size).fit(signal_left)
    bkps_left = algo_left.predict(n_bkps=n_bkps)
    change_left = bkps_left[0]

    onset_left = find_transit_edge_local(signal_left, change_left, find_onset=True, window=window, percentile=percentile)

    # Detect on right half (after transit: rise)
    algo_right = rpt.Binseg(model="l2", min_size=min_size).fit(signal_right)
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
files = glob.glob(os.path.join(path_folder, 'train','*','*'))

CHUNKS_SIZE = 1
index_chunks = get_multiobs_index(files, CHUNKS_SIZE)

midpoint = len(index_chunks) // 2
first_half_chunks = index_chunks[38:58]
second_half_chunks = index_chunks[midpoint:]

index_chunks = first_half_chunks

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
train_df = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
star_train_df = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')


ratios = []
stati = []
statTs = []
statsma = []
planets = []
for index_chunk in  index_chunks:
    AIRS_CH0_clean = np.ma.MaskedArray(np.zeros((CHUNKS_SIZE, 11250, 32, l)))
    FGS1_clean = np.ma.MaskedArray(np.zeros((CHUNKS_SIZE, 135000, 32, 32)))
    
    chunk_name = '__'.join([f"{pid}_{obs}" for pid, obs in index_chunk])
    
    if already_saved(chunk_name, path_out):
            print(f"Skipping {chunk_name} (already processed)")
            continue  # Go to next chunk
    print(chunk_name)
    
    for i in range (CHUNKS_SIZE) : 
        df = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/AIRS-CH0_signal_{index_chunk[i][1]}.parquet'))
        signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
        gain = train_adc_info['AIRS-CH0_adc_gain'][0]
        offset = train_adc_info['AIRS-CH0_adc_offset'][0]
        signal = ADC_convert(signal, gain, offset)
        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1
        chopped_signal = signal[:, :, cut_inf:cut_sup]
        del signal, df
        
        # CLEANING THE DATA: AIRS
        flat = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dark = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dead_airs = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        linear_corr = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
        
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
        
        df = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/FGS1_signal_{index_chunk[i][1]}.parquet'))
        fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))
        
        FGS1_gain = train_adc_info['FGS1_adc_gain'][0]
        FGS1_offset = train_adc_info['FGS1_adc_offset'][0]
        
        fgs_signal = ADC_convert(fgs_signal, FGS1_gain, FGS1_offset)
        dt_fgs1 = np.ones(len(fgs_signal))*0.1
        dt_fgs1[1::2] += 0.1
        chopped_FGS1 = fgs_signal
        del fgs_signal, df
        
        # CLEANING THE DATA: FGS1
        flat = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        dark = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/dark.parquet')).values.astype(np.float64).reshape((32, 32))
        dead_fgs1 = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/dead.parquet')).values.astype(np.float64).reshape((32, 32))
        linear_corr = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))
        
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
        flat_airs = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        flat_fgs = pd.read_parquet(os.path.join(path_folder,f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 32))
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
        print("data contains nan")
   
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
    print(result_fgs1.shape,result_airs.shape)

    xmins = []
    polys = []
    datas = []
    
    #median filter
    result_fgs1, xcrop = median_filter_and_downsample(result_fgs1, median_filter_window=101, stride=1, plot=False)
    print(result_fgs1.shape,result_airs.shape)
    #find change points
    #try:
    onset, offset, mind, xmin = plot_transit_edges(result_fgs1, plot=True)
    linear = False
    print('success')
    #fit P
    fitted_curve, coeffs, idx_baseline = fit_and_plot_baseline(result_fgs1,onset,offset,delta=10,degree=2,planet_id=None,plot=True)
    datas.append(result_fgs1)
    xmins.append(xmin)
    polys.append(fitted_curve)

    
    VAL = train_df.loc[train_df['planet_id'] == index_chunk[i][0], 'wl_1'].values
    if VAL.size == 0:
        print(f"No wl_1 value found for planet_id {index_chunk[i][0]}")
        VAL = 1.0  # Or set some default fallback
    else:
        VAL = VAL[0]

    # Calculate the modified curve
    modified_curve = fitted_curve - (fitted_curve * VAL)

    sval = np.max((fitted_curve-result_fgs1)/fitted_curve)
    ratio = VAL/sval
    ratios.append(ratio)

    stati.append(star_train_df.loc[star_train_df['planet_id']==index_chunk[i][0],'i'].values[0])
    statTs.append(star_train_df.loc[star_train_df['planet_id']==index_chunk[i][0],'Ts'].values[0])
    statsma.append(star_train_df.loc[star_train_df['planet_id']==index_chunk[i][0],'sma'].values[0])
    planets.append(index_chunk[i][0])

    # Create DataFrame
    df_stats = pd.DataFrame({
        'planet_id': planets,
        'ratio': ratios,
        'i': stati,
        'Ts': statTs,
        'sma': statsma
    })
       
    # result_fgs1: your observed light curve
    time = (np.arange(len(result_fgs1))-mind)*4*1.2/86400
    
    # Use pre-transit baseline to normalize, if enough points; otherwise, use median
    pre_transit = result_fgs1[:onset] if onset > 0 else result_fgs1
    flux = result_fgs1 / np.median(pre_transit)
    
    # Assign provided parameters for the current planet:
    period = star_train_df.loc[star_train_df['planet_id']==index_chunk[i][0],'P'].values[0]
    a_rs = star_train_df.loc[star_train_df['planet_id']==index_chunk[i][0],'sma'].values[0]
    inc = star_train_df.loc[star_train_df['planet_id']==index_chunk[i][0],'i'].values[0]
    ecc = star_train_df.loc[star_train_df['planet_id']==index_chunk[i][0],'e'].values[0]
    t0 = 0#(onset + offset) / 2  # index for mid-transit; if you have real time, use that!
    rp_rs = np.sqrt(sval)       # if you know R_p, rp_rs = R_p / Rs; else, sqrt(depth)
    
    # Quick setup of limb darkening coefficients (here, quadratic, dummy values)
    #u1, u2 = 0.2, 0.3  # You can use more physical values if you have tables by Ts

    
    # Retrieve Ts from your dataframe (already formatted per ID in your loop)
    Ts = float(star_train_df.loc[star_train_df['planet_id'] == index_chunk[i][0], 'Ts'].values[0])
    Ts_err = 100                # Set uncertainty, or use your own estimates if available
    logg = 4.5
    logg_err = 0.1
    z = 0.0                     # Metallicty
    z_err = 0.05
    
    # Define your filter, e.g. 500-600 nm (adjust to your passband as needed)
    filters = [BoxcarFilter('myband', 500, 600)]
    
    # Generate LD coefficients
    sc = LDPSetCreator(teff=(Ts, Ts_err), logg=(logg, logg_err), z=(z, z_err), filters=filters)
    ps = sc.create_profiles()
    u, uerr = ps.coeffs_qd(do_mc=True)  # This gets quadratic LD coeffs
    
    u1, u2 = u[0]
    print(f"Limb darkening for Ts={Ts}: u1={u1:.3f}, u2={u2:.3f}")
        
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = ecc
    params.w = 90.0  # argument of periastron; 90 deg is common for transits
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    
    m = batman.TransitModel(params, time)
    model_flux = m.light_curve(params)
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(time, flux, label='result_fgs1 (normalized)', color='blue')
    plt.plot(time, fitted_curve / np.median(pre_transit), label='fitted_curve (normalized)', color='orange')
    plt.plot(time, model_flux, label='BATMAN transit fit', color='green')
    plt.legend()
    plt.title(f'Transit Fit for Planet {index_chunk[i][0]}')
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Flux')
    plt.show()

    
    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(result_fgs1, label='result_fgs1')
    plt.plot(fitted_curve, label='fitted_curve')
    plt.plot(modified_curve, label='fitted_curve * VAL')
    plt.legend()
    plt.title(f'Transit Curve for Planet {index_chunk[i][0]}')
    plt.xlabel('Index or Time')
    plt.ylabel('Signal Strength')
    plt.show()

    for wl in range(result_airs.shape[1]):
        signal = result_airs[:, wl]
        signal, xcrop = median_filter_and_downsample(signal, median_filter_window=101, stride=1, plot=False)
        fitted_curve, coeffs, idx_baseline = fit_and_plot_baseline(
            signal,
            onset,
            offset,
            delta=10,
            degree=2,
            planet_id=None,
            plot=False
        )
        smoothed_signal = savgol_filter(signal, window_length=15, polyorder=2)
        datas.append(signal)
        xmins.append(smoothed_signal[mind])
        polys.append(fitted_curve)
    
    #save

    datas_tensor = torch.from_numpy(np.stack(datas))   # Shape: (num_arrays, array_length)
    polys_tensor = torch.from_numpy(np.stack(polys))   # Shape: (num_arrays, array_length)
    
    # Convert list of scalars to 1D tensor
    xmins_tensor = torch.tensor(xmins)                  # Shape: (num_scalars,)
    
    torch.save({'data': datas_tensor, 'poly': polys_tensor, 'xmin': xmins_tensor, 'mind':torch.tensor(mind)}, os.path.join(path_out, f'clean_train_{chunk_name}.pt'))
    #torch.save({'data': data_3d_fgs1, 'mask': mask_2d_fgs1}, os.path.join(path_out, f'FGS1_clean_train_{chunk_name}.pt'))
    
    print(chunk_name, count)
    del AIRS_cds_binned
    del FGS1_cds_binned
    count +=1


# Plotting
params = ['i', 'Ts', 'sma']
for param in params:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_stats, x=param, y='ratio')
    plt.title(f'Ratio vs {param}')
    plt.xlabel(param)
    plt.ylabel('Ratio (VAL / sval)')
    plt.grid(True)
    plt.show()

