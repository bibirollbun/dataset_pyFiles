import numpy as np
import pandas as pd
import itertools
import os
import glob 
from astropy.stats import sigma_clip
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import medfilt


path_folder = '/kaggle/input/ariel-data-challenge-2025/' # path to the folder containing the data
path_out = '/kaggle/tmp/data_light_raw/' # path to the folder to store the light data
out_dir = 'data/' # path for the output directory


# Useful functions
def ADC_convert(signal, gain=0.4369, offset=-1000):
    """The Analog-to-Digital Conversion (adc) is performed by the detector to convert
    the pixel voltage into an integer number. Since we are using the same conversion number 
    this year, we have simply hard-coded it inside. """
    signal /= gain
    signal += offset
    return signal

# Mask hot and dead pixels
def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(
        dark, sigma=5, maxiters=5
    ).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))

    # Combine masks
    combined_mask = np.logical_or(dead, hot)

    signal = np.ma.masked_where(combined_mask, signal).astype(signal.dtype, copy=False)
    return signal

def apply_linear_corr(linear_corr, signal):
    """
    Vectorized Horner evaluation.
    linear_corr: (K, Y, X) coefficients c0..c_{K-1} (ascending)
    clean_signal: (T, Y, X)
    Returns: (T, Y, X)
    """
    # xp = type(signal)  # works if you have xp in outer scope; else pass xp in

    # Optional: switch to float32 for speed (if accuracy allows)
    # signal = signal.astype(xp.float32, copy=False)
    # linear_corr = linear_corr.astype(xp.float32, copy=False)

    # Start from highest coefficient and fold down: v = c_{K-1}; v = v*s + c_{K-2}; ...
    c = linear_corr  # ascending
    s = signal

    # Broadcast c_{K-1} to (T,Y,X), then in-place Horner
    out = np.broadcast_to(c[-1][None, ...], s.shape).copy()
    for k in range(c.shape[0] - 2, -1, -1):
        out *= s           # v *= s
        out += c[k]        # v += c_k
    return out

# Dark current subtraction function
def clean_dark(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark* dt[:, np.newaxis, np.newaxis]
    return signal

# Co-related Double Sampling (CDS) function
def get_cds(signal):
    cds = signal[1::2,:,:] - signal[::2,:,:]
    return cds

# Flat field correction function
def correct_flat_field(flat, dead, signal):
    # flat = flat.transpose(1, 0)
    # dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal

# (Optional) Time Binning function (May need some modifications)
def bin_obs(cds_signal, binning):
    cds_transposed = cds_signal.transpose(0,1,3,2)
    cds_binned = np.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning, cds_transposed.shape[2], cds_transposed.shape[3]))
    for i in range(cds_transposed.shape[1]//binning):
        cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
    return cds_binned

# Cosmic ray removal function
# Credit: lordpatil
def mad_clip_data(data, window_size=51, sigma=3):
    """Simple sigma clipping function."""
    local_median = medfilt(data, kernel_size=window_size)
    residual = data - local_median
    mad = np.median(np.abs(residual))
    robust_std = mad * 1.4826
    outliers = np.abs(residual) > (sigma * robust_std)

    masked_data = np.ma.array(data, mask=outliers)
    return masked_data, outliers


# Code to process the signal from both detectors
def process_signal(path_folder, index, instrument, obs_count=0, x_slice=slice(None), y_slice=slice(None),
                   do_mask=True, do_nl_corr=True, do_dark=True, do_flat=True, binning_factor=1):
    """Process the signal for AIRS-CH0 or FGS1 with the given parameters.
    
    Args:
        path_folder (str): Base path to dataset folder.
        index (str): Planet ID (e.g., planet_id).
        instrument (str): 'AIRS-CH0' or 'FGS1'.
        obs_count (int): Observation count (default: 0).
        x_slice (slice): Slice for x-axis (default: instrument specific).
        y_slice (slice): Slice for y-axis (default: instrument specific).
        do_mask (bool): Apply hot/dead pixel masking.
        do_nl_corr (bool): Apply non-linearity correction.
        do_dark (bool): Apply dark subtraction.
        do_flat (bool): Apply flat-field correction.
        time_binning (bool): Apply time binning (default: False).
    
    Returns:
        times (np.ndarray): Time array for the observation.
        signal (np.ndarray): Processed 3D signal (n_times//2 x spatial x dispersion).
    """
    if instrument == 'AIRS-CH0':
        signal_file = f'{instrument}_signal_{obs_count}.parquet'
        calib_folder = f'{instrument}_calibration_{obs_count}'
        calib_shape = (32, 356)
        calib_linear_shape = (6, 32, 356)
        if x_slice == slice(None):
            x_slice = slice(39, 321)   # wavelength slice for AIRS-CH0
        if y_slice == slice(None):
            y_slice = slice(10, 22)  # spatial slice for AIRS-CH0
        axis_key = 'AIRS-CH0-axis0-h'
        dt_key = 'AIRS-CH0-integration_time'
    elif instrument == 'FGS1':
        signal_file = f'{instrument}_signal_{obs_count}.parquet'
        calib_folder = f'{instrument}_calibration_{obs_count}'
        calib_shape = (32, 32)
        calib_linear_shape = (6, 32, 32)
        if x_slice == slice(None):
            x_slice = slice(8, 24)
        if y_slice == slice(None):
            y_slice = slice(8, 24)
        axis_key = 'FGS1-axis0-h'
        # dt_key = 'FGS1-integration_time'
    else:
        raise ValueError("Instrument must be 'AIRS-CH0' or 'FGS1'")

    # Load signal data first
    df = pd.read_parquet(os.path.join(path_folder, f'train/{index}/{signal_file}'))
    reshape_dims = (df.shape[0], calib_shape[0], calib_shape[1])

    # Load calibrations
    flat = pd.read_parquet(os.path.join(path_folder, f'train/{index}/{calib_folder}/flat.parquet')).values.astype(np.float32).reshape(calib_shape)[y_slice, x_slice]
    dark = pd.read_parquet(os.path.join(path_folder, f'train/{index}/{calib_folder}/dark.parquet')).values.astype(np.float32).reshape(calib_shape)[y_slice, x_slice]
    dead = pd.read_parquet(os.path.join(path_folder, f'train/{index}/{calib_folder}/dead.parquet')).values.astype(np.float32).reshape(calib_shape)[y_slice, x_slice]
    linear_corr = pd.read_parquet(os.path.join(path_folder, f'train/{index}/{calib_folder}/linear_corr.parquet')).values.astype(np.float32).reshape(calib_linear_shape)[:, y_slice, x_slice]
    # Have not used the read data yet, probably used for handling uncertainty
    read = pd.read_parquet(os.path.join(path_folder, f'train/{index}/{calib_folder}/read.parquet')).values.astype(np.float32).reshape(calib_shape)[y_slice, x_slice]
    axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))
    if instrument == 'AIRS-CH0':
        dt = axis_info[dt_key].dropna().values
    else:
        dt = np.ones(df.shape[0]) * 0.1
    dt[1::2] += 0.1

    # Reshape and process signal
    signal = df.values.astype(np.float32).reshape(reshape_dims)[:, y_slice, x_slice]
    signal = ADC_convert(signal)

    if do_mask:
        signal = mask_hot_dead(signal, dead, dark)
    
    if do_nl_corr:
        signal = apply_linear_corr(linear_corr, signal)

    if do_dark:
        signal = clean_dark(signal, dead, dark, dt)
    
    # Compute time array
    times = axis_info[axis_key].dropna().values
    if len(times) % 2 != 0:   # Ensure even number of time points
        times = times[:-1]
    times = 0.5 * (times[1::2] + times[0::2])

    # Correlated Double Sampling (CDS)
    signal = get_cds(signal)

    if binning_factor > 1:
        signal = signal[:(signal.shape[0] // binning_factor) * binning_factor, :, :]
        signal = signal.reshape(signal.shape[0] // binning_factor, binning_factor, signal.shape[1], signal.shape[2]).mean(axis=1)
        times = times[:(len(times) // binning_factor) * binning_factor]
        times = times.reshape(-1, binning_factor).mean(axis=1)

    # Flat field correction was applied after time binning in the original code
    if do_flat:
        signal = correct_flat_field(flat, dead, signal).data

    # Returning read data for uncertainty estimation
    return times, signal, read


def get_index_obsCount(files):
    index_obsCount = []
    obs_count = 0
    for file in files:
        index = file.split('/')[-1]
        airs_files = glob.glob(os.path.join(file, 'AIRS-CH0_signal_*.parquet'))
        obs_count = len(airs_files)
        index_obsCount.append((int(index), obs_count))

    return index_obsCount

# Get list of planet IDs with their observation counts
files = glob.glob(os.path.join(path_folder + 'train/', '*'))

# planet IDs with their corresponding observation counts
indices_w_obsCount = get_index_obsCount(files)
indices_w_obsCount.sort(key=lambda x: x[0])


if not os.path.exists(path_out):
    os.makedirs(path_out)
    print(f"Directory {path_out} created.")
else:
    print(f"Directory {path_out} already exists.")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(f"Directory {out_dir} created.")
else:
    print(f"Directory {out_dir} already exists.")


train_star_info = pd.read_csv(os.path.join(path_folder, 'train_star_info.csv'))



# Create a new DataFrame to store the expanded metadata (repeating rows based on observation counts)
df_new = pd.DataFrame({'planet_id': pd.Series(dtype="int"), 
                       'Rs': pd.Series(dtype="float32"), 
                       'Ms': pd.Series(dtype="float32"), 
                       'Ts': pd.Series(dtype="float32"),
                       'Mp': pd.Series(dtype="float32"),
                       'e': pd.Series(dtype="float32"),
                       'P': pd.Series(dtype="float32"),
                       'sma': pd.Series(dtype="float32"),
                       'i': pd.Series(dtype="float32")})

for idx, obs_count in tqdm(indices_w_obsCount):
    for obs in range(obs_count):
        df_new.loc[len(df_new)] = train_star_info[train_star_info['planet_id'] == idx][['planet_id', 'Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i']].values[0]

df_new['planet_id'] = df_new['planet_id'].astype(int)

# Save the new DataFrame to a CSV file
df_new.to_csv(os.path.join(out_dir,"star_planet_metadata.csv"), index=False)


# WRAPPING THE WORK IN A FUNCTION
def process_obs(path_folder, index, obs, overall_binning, path_out=None, count=None):
    # AIRS
    times_airs, airs_cds, airs_read = process_signal(
        path_folder, index, 'AIRS-CH0', obs_count=obs,
        do_mask=True, do_nl_corr=True, do_dark=True,
        do_flat=True, binning_factor=overall_binning
    )

    # FGS
    times_fgs, fgs_cds, fgs_read = process_signal(
        path_folder, index, 'FGS1', obs_count=obs,
        do_mask=True, do_nl_corr=True, do_dark=True,
        do_flat=True, binning_factor=12 * overall_binning
    )

    # Optional saving
    if path_out is not None and count is not None:
        np.save(os.path.join(path_out, f'AIRS_train_{count}.npy'), airs_cds)
        np.save(os.path.join(path_out, f'FGS1_train_{count}.npy'), fgs_cds)

    # Clean memory
    del airs_cds, fgs_cds  

    return (index, obs, times_airs, times_fgs)  # return what you need


# BUILDING THE TASK LIST

# Counter for number of processed observations
count = 0
# A binning factor of 12 is used to match the time resolution of FGS1 with AIRS-CH0
# Additional overall binning is applied to both detectors to reduce noise
overall_binning = 10

tasks = []
for index, obs_count in indices_w_obsCount:
    for obs in range(obs_count):
        tasks.append((path_folder, index, obs, overall_binning, path_out, count))
        count += 1


n_workers = 4 # Number of workers

results = []
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = [executor.submit(process_obs, *task) for task in tasks]
    for fut in tqdm(as_completed(futures), total=len(futures)):
        results.append(fut.result())


# Concatenate all data into a single .npy file
def load_data(path, detector):
    files = glob.glob(os.path.join(path, f'{detector}_train_*.npy'))
    data_tmp = np.load(files[0])
    data_all = np.zeros((len(files), *data_tmp.shape))
    for i in range(len(files)):
        data_all[i] = np.load(os.path.join(path, f'{detector}_train_{i}.npy'))
    return data_all


!ls /kaggle/tmp/data_light_raw/


# Combined ARIS and FGS data
data_train_airs = load_data(path_out, 'AIRS')
np.save(os.path.join(out_dir, 'data_train_airs.npy'), data_train_airs)
del data_train_airs  # Free up memory
data_train_fgs = load_data(path_out, 'FGS1')
np.save(os.path.join(out_dir, 'data_train_fgs.npy'), data_train_fgs)
del data_train_fgs  # Free up memory


index, obs = indices_w_obsCount[0]
obs -= 1

times_airs, airs_cds, airs_read = process_signal(
        path_folder, index, 'AIRS-CH0', obs_count=obs,
        do_mask=True, do_nl_corr=True, do_dark=True,
        do_flat=True, binning_factor=overall_binning
    )

times_fgs, fgs_cds, fgs_read = process_signal(
        path_folder, index, 'FGS1', obs_count=obs,
        do_mask=True, do_nl_corr=True, do_dark=True,
        do_flat=True, binning_factor=12 * overall_binning
    )

np.save(os.path.join(out_dir, 'times_airs.npy'), times_airs)
np.save(os.path.join(out_dir, 'times_fgs.npy'), times_fgs)




