# preprocessing code borrowed from: https://www.kaggle.com/code/gordonyip/update-calibrating-and-binning-astronomical-data
import numpy as np
import pandas as pd
import itertools
import os
import glob 
from astropy.stats import sigma_clip
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re
from skimage.restoration import inpaint_biharmonic


# Load spectra and wavelengths
train = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
wavelengths = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')

# Columns for spectrum (excluding 'planet_id')
spectrum_col_names = train.columns.drop('planet_id')
wavelength_values = wavelengths.loc[0, spectrum_col_names].values.astype(float)  # Shape: (283,)

# Planet indices or planet_ids to compare (first 3 rows by default)
indices = [0, 1, 2]

# Plot: Original spectra
plt.figure(figsize=(10, 5))
for idx in indices:
    row = train.iloc[idx]
    planet_id = row['planet_id']
    spectrum_values = row[spectrum_col_names].values.astype(float)
    plt.plot(wavelength_values, spectrum_values, marker='o', label=f'Planet {planet_id}')
plt.xlabel('Wavelength (μm)')
plt.ylabel('rp/rs (planet-to-star radius ratio)')
plt.title('Transmission Spectra for 3 Exoplanets (Raw)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Z-normalized spectra
plt.figure(figsize=(10, 5))
for idx in indices:
    row = train.iloc[idx]
    planet_id = row['planet_id']
    spectrum_values = row[spectrum_col_names].values.astype(float)
    # Z-normalization: (value - mean) / std
    z_norm_spectrum = (spectrum_values - spectrum_values.mean()) / spectrum_values.std()
    plt.plot(wavelength_values, z_norm_spectrum, marker='o', label=f'Planet {planet_id}')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Z-normalized rp/rs')
plt.title('Transmission Spectra for 3 Exoplanets (Z-Normalized)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(spectrum_values.shape)


# Load ADC calibration parameters (identical for all planets)
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
airs_gain = adc['AIRS-CH0_adc_gain'].iloc[0]   # Gain for AIRS-CH0 detector
airs_offset = adc['AIRS-CH0_adc_offset'].iloc[0] # Offset for AIRS-CH0 detector
fgs_gain = adc['FGS1_adc_gain'].iloc[0]         # Gain for FGS1 detector
fgs_offset = adc['FGS1_adc_offset'].iloc[0]     # Offset for FGS1 detector

# --- AIRS-CH0 IMAGE (32x356), FIRST FRAME ---
# Load AIRS-CH0 signal data for a sample planet (first time frame)
airs_path = '/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_signal_0.parquet'
airs_signals = pd.read_parquet(airs_path)
airs_img = airs_signals.iloc[0].values.reshape(32, 356)  # Reshape to detector dimensions
# Calibrate raw pixel values using gain and offset
airs_img_cal = (airs_img / airs_gain) + airs_offset

# Visualize calibrated AIRS-CH0 image
plt.figure(figsize=(12,4))
plt.imshow(airs_img_cal, cmap='gray', aspect='auto')
plt.colorbar(label='Calibrated pixel value')
plt.title('AIRS-CH0 Calibrated Image (First Time Frame)')
plt.xlabel('Wavelength axis (column index)')
plt.ylabel('Spatial axis (row index)')
plt.show()

# --- FGS1 IMAGE (32x32), FIRST FRAME ---
# Load FGS1 signal data for the same planet (first time frame)
fgs_path = '/kaggle/input/ariel-data-challenge-2025/train/1010375142/FGS1_signal_0.parquet'
fgs_signals = pd.read_parquet(fgs_path)
fgs_img = fgs_signals.iloc[0].values.reshape(32, 32)  # Reshape to detector dimensions
# Calibrate raw pixel values using gain and offset
fgs_img_cal = (fgs_img / fgs_gain) + fgs_offset

# Visualize calibrated FGS1 image
plt.figure(figsize=(6,6))
plt.imshow(fgs_img_cal, cmap='gray')
plt.colorbar(label='Calibrated pixel value')
plt.title('FGS1 Calibrated Image (First Time Frame)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.show()



base_dir = '/kaggle/input/ariel-data-challenge-2025/train'

# List of sample planet IDs to check
sample_planets = ['1010375142', '1024292144', '1029552010']

for planet_id in sample_planets:
    # Construct full paths to AIRS-CH0 and FGS1 signal files for each planet
    airs_path = os.path.join(base_dir, planet_id, 'AIRS-CH0_signal_0.parquet')
    fgs1_path = os.path.join(base_dir, planet_id, 'FGS1_signal_0.parquet')
    
    # Check if AIRS-CH0 file exists, then load and print its shape
    if os.path.exists(airs_path):
        airs_data = pd.read_parquet(airs_path)
        print(f'Planet {planet_id} AIRS-CH0 shape:', airs_data.shape)
        # Expected shape: (num_time_frames, num_pixels)
    else:
        print(f'Planet {planet_id} AIRS-CH0 file missing.')
    
    # Check if FGS1 file exists, then load and print its shape
    if os.path.exists(fgs1_path):
        fgs1_data = pd.read_parquet(fgs1_path)
        print(f'Planet {planet_id} FGS1 shape:', fgs1_data.shape)
        # Expected shape: (num_time_frames, num_pixels)
    else:
        print(f'Planet {planet_id} FGS1 file missing.')
    
    print('---')  # Separator for readability


# Load ADC calibration parameters for AIRS-CH0 detector
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
airs_gain = adc['AIRS-CH0_adc_gain'].iloc[0]   # Gain for AIRS-CH0
airs_offset = adc['AIRS-CH0_adc_offset'].iloc[0] # Offset for AIRS-CH0

# Select a sample planet to visualize
planet_id = '1010375142'  # Replace with any valid planet_id as needed

# Load AIRS-CH0 signal data for the selected planet
path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_0.parquet'
df = pd.read_parquet(path)

frames_to_plot = 9  # Number of time frames to visualize (first 9)
plt.figure(figsize=(15, 10))
for idx in range(frames_to_plot):
    # Extract and reshape the signal for the current frame (32 rows x 356 columns)
    img = df.iloc[idx].values.reshape(32, 356)
    # Calibrate the raw pixel values using gain and offset
    cal_img = (img / airs_gain) + airs_offset
    # Plot the calibrated image in a 3x3 grid
    plt.subplot(3, 3, idx + 1)
    plt.imshow(cal_img, cmap='gray', aspect='auto')
    plt.title(f'AIRS Frame {idx}')
    plt.axis('off')  # Hide axis ticks for clarity
plt.tight_layout()
plt.show()


# Path to the folder containing the input data
path_folder = '/kaggle/input/ariel-data-challenge-2025'
# Path to the folder where processed light data will be stored
path_out = '/kaggle/working/data_light_raw/'
output_dir = '/kaggle/working/data_light_raw/'  # (redundant, but kept for clarity)

# Check if the output directory exists; create it if it does not
if not os.path.exists(path_out):
    os.makedirs(path_out)  # Creates all intermediate directories as needed
    print(f"Directory {path_out} created.")
else:
    print(f"Directory {path_out} already exists.")


CHUNKS_SIZE = 1


def ADC_convert(signal, gain, offset):
    """
    Convert raw detector signal to calibrated values using gain and offset.
    
    Parameters:
        signal (ndarray): Raw pixel values.
        gain (float): Calibration gain from ADC info.
        offset (float): Calibration offset from ADC info.
    
    Returns:
        ndarray: Calibrated signal as float64.
    """
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    """
    Mask hot and dead pixels in the signal array.
    Hot pixels are identified using sigma clipping on the dark frame.
    Masks are repeated along the time axis to match signal shape.
    
    Parameters:
        signal (ndarray): Detector data (time, x, y).
        dead (ndarray): Boolean mask of dead pixels (True = dead).
        dark (ndarray): Dark frame used to identify hot pixels.
    
    Returns:
        np.ma.MaskedArray: Signal with hot and dead pixels masked.
    """
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def apply_linear_corr(linear_corr, clean_signal):
    """
    Apply a per-pixel linearity correction to the signal using polynomial coefficients.
    
    Parameters:
        linear_corr (ndarray): Polynomial coefficients (degree, x, y).
        clean_signal (ndarray): Detector data (time, x, y).
    
    Returns:
        ndarray: Linearity-corrected signal.
    """
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark(signal, dead, dark, dt):
    """
    Subtract dark current from the signal, accounting for dead pixels and integration time.
    
    Parameters:
        signal (ndarray): Detector data (time, x, y).
        dead (ndarray): Boolean mask of dead pixels.
        dark (ndarray): Dark frame (x, y).
        dt (ndarray): Integration time per frame (time,).
    
    Returns:
        ndarray: Dark-corrected signal.
    """
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal

def get_cds(signal):
    """
    Compute correlated double sampling (CDS) difference along the time axis.
    
    Parameters:
        signal (ndarray): Detector data (time, ...).
    
    Returns:
        ndarray: CDS-differenced signal (half the time dimension).
    """
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds

def correct_flat_field(flat, dead, signal):
    """
    Apply flat field correction to the signal.
    Masks dead pixels in the flat field and repeats along the time axis.
    
    Parameters:
        flat (ndarray): Flat field frame (x, y).
        dead (ndarray): Boolean mask of dead pixels (x, y).
        signal (ndarray): Detector data (time, x, y).
    
    Returns:
        ndarray: Flat-field corrected signal.
    """
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal


def get_index(files, CHUNKS_SIZE):
    """
    Extract planet indices from AIRS-CH0_signal_0.parquet files and split into chunks.

    Parameters:
        files (list of str): List of file paths.
        CHUNKS_SIZE (int): Number of indices per chunk.

    Returns:
        list of np.ndarray: List of arrays, each containing planet indices for a chunk.
    """
    index = []
    for file in files:
        file_name = file.split('/')[-1]
        if file_name.split('_')[0] == 'AIRS-CH0' and file_name.split('_')[-1] == '0.parquet':
            file_index = os.path.basename(os.path.dirname(file))
            index.append(int(file_index))
    index = np.array(index)
    index = np.sort(index)
    # Split indices into chunks of size CHUNKS_SIZE
    index = np.array_split(index, len(index)//CHUNKS_SIZE)
    return index


def get_multiobs_index(files, CHUNKS_SIZE):
    """
    Extract (planet_id, obs_num) pairs from AIRS-CH0_signal_X.parquet files and split into chunks.

    Parameters:
        files (list of str): List of file paths.
        CHUNKS_SIZE (int): Number of pairs per chunk.

    Returns:
        list of list: List of (planet_id, obs_num) tuples, split into chunks.
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
    # Sort by planet_id then obs_num
    index.sort()
    # Remove duplicates
    index = list(dict.fromkeys(index))
    if len(index) >= CHUNKS_SIZE and CHUNKS_SIZE > 0:
        index_chunks = np.array_split(index, len(index)//CHUNKS_SIZE)
    else:
        index_chunks = [index]
    return index_chunks

def bin_obs(arr, binning, axis=1):
    """
    Bin an array along a specified axis by summing over non-overlapping bins.

    Parameters:
        arr (np.ndarray or np.ma.MaskedArray): Input array to bin.
        binning (int): Bin size (number of elements per bin).
        axis (int): Axis along which to bin (default: 1).

    Returns:
        np.ma.MaskedArray: Binned array with reduced size along the specified axis.
    """
    bin_size = binning
    arr = np.ma.masked_array(arr)
    shape = list(arr.shape)
    n_bins = shape[axis] // bin_size
    new_shape = shape[:axis] + [n_bins, bin_size] + shape[axis+1:]
    arr_reshaped = np.ma.reshape(arr, new_shape)
    # Sum along the bin_size axis (axis=axis+1)
    return np.ma.sum(arr_reshaped, axis=axis+1)


def median_filter_time(masked_arr, kernel_size=3):
    """
    Apply a 1D median filter along the time axis (axis=1) for each batch.
    Ignores masked voxels and preserves masked array structure.

    Parameters:
        masked_arr (np.ma.MaskedArray): Input array of shape (batch, time, X, Y).
        kernel_size (int): Size of the median filter window (must be odd, default: 3).

    Returns:
        np.ma.MaskedArray: Median-filtered array of the same shape as input.
    """
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
            # Compute median along the window (time) axis
            median_vals = np.ma.median(window_ma, axis=0)
            result.data[b, t] = median_vals.data
            result.mask[b, t] = median_vals.mask

    return result



# Gather all signal file paths for training data
files = glob.glob(os.path.join(path_folder, 'train','*','*'))
#print(files[:5])  # Print first 5 files for inspection
for file in files[:5]:
    file_name = file.split('/')[-1]
    #print(file_name)  # Print file names for quick check

# Get index chunks for batch processing (planet_id, obs_num pairs)
index_chunks = get_multiobs_index(files[:16], CHUNKS_SIZE)

# Load calibration and axis info
train_adc_info = pd.read_csv(os.path.join(path_folder, 'adc_info.csv'))
axis_info = pd.read_parquet(os.path.join(path_folder,'axis_info.parquet'))

# Flags to control cleaning steps
DO_MASK = True      # Mask hot/dead pixels
DO_THE_NL_CORR = False  # Apply non-linearity correction
DO_DARK = True      # Subtract dark current
DO_FLAT = True      # Apply flat field correction
TIME_BINNING = True # Bin along time axis
FILT = True         # Apply median filter

cut_inf, cut_sup = 0, 356  # Wavelength axis cut
l = cut_sup - cut_inf       # Length of wavelength axis

for index_chunk in index_chunks:
    # Initialize masked arrays for cleaned AIRS and FGS1 data
    AIRS_CH0_clean = np.ma.MaskedArray(np.zeros((CHUNKS_SIZE, 11250, 32, l)))
    FGS1_clean = np.ma.MaskedArray(np.zeros((CHUNKS_SIZE, 135000, 32, 32)))
    
    for i in range(CHUNKS_SIZE):
        # --- Load and calibrate AIRS-CH0 signal ---
        df = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/AIRS-CH0_signal_{index_chunk[i][1]}.parquet'))
        signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
        gain = train_adc_info['AIRS-CH0_adc_gain'][0]
        offset = train_adc_info['AIRS-CH0_adc_offset'][0]
        signal = ADC_convert(signal, gain, offset)
        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1  # Adjust integration time for odd frames
        chopped_signal = signal[:, :, cut_inf:cut_sup]  # Crop wavelength axis
        del signal, df
        
        # --- AIRS-CH0 cleaning steps ---
        flat = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dark = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dead_airs = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        linear_corr = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
        
        if DO_MASK:
            chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
            AIRS_CH0_clean[i] = chopped_signal
        else:
            AIRS_CH0_clean[i] = chopped_signal
        
        if DO_THE_NL_CORR:
            linear_corr_signal = apply_linear_corr(linear_corr, AIRS_CH0_clean[i])
            AIRS_CH0_clean[i, :, :, :] = linear_corr_signal
        del linear_corr
        
        if DO_DARK:
            cleaned_signal = clean_dark(AIRS_CH0_clean[i], dead_airs, dark, dt_airs)
            AIRS_CH0_clean[i] = cleaned_signal
        del dark
        
        # --- Load and calibrate FGS1 signal ---
        df = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/FGS1_signal_{index_chunk[i][1]}.parquet'))
        fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))
        FGS1_gain = train_adc_info['FGS1_adc_gain'][0]
        FGS1_offset = train_adc_info['FGS1_adc_offset'][0]
        fgs_signal = ADC_convert(fgs_signal, FGS1_gain, FGS1_offset)
        dt_fgs1 = np.ones(len(fgs_signal)) * 0.1
        dt_fgs1[1::2] += 0.1
        chopped_FGS1 = fgs_signal
        del fgs_signal, df
        
        # --- FGS1 cleaning steps ---
        flat = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        dark = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/dark.parquet')).values.astype(np.float64).reshape((32, 32))
        dead_fgs1 = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/dead.parquet')).values.astype(np.float64).reshape((32, 32))
        linear_corr = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))
        
        if DO_MASK:
            chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
            FGS1_clean[i] = chopped_FGS1
        else:
            FGS1_clean[i] = chopped_FGS1
        
        if DO_THE_NL_CORR:
            linear_corr_signal = apply_linear_corr(linear_corr, FGS1_clean[i])
            FGS1_clean[i, :, :, :] = linear_corr_signal
        del linear_corr
        
        if DO_DARK:
            cleaned_signal = clean_dark(FGS1_clean[i], dead_fgs1, dark, dt_fgs1)
            FGS1_clean[i] = cleaned_signal
        del dark
    
    # --- Post-processing: CDS, filtering, binning ---
    AIRS_cds = get_cds(AIRS_CH0_clean)
    FGS1_cds = get_cds(FGS1_clean)
    del AIRS_CH0_clean, FGS1_clean

    if FILT:
        AIRS_cds = median_filter_time(AIRS_cds)
        FGS1_cds = median_filter_time(FGS1_cds)
    
    # Optional: bin along time axis to reduce data size
    if TIME_BINNING:
        AIRS_cds_binned = bin_obs(AIRS_cds, binning=1)
        FGS1_cds_binned = bin_obs(FGS1_cds, binning=12)
    else:
        AIRS_cds_binned = AIRS_cds
        FGS1_cds_binned = FGS1_cds
    AIRS_cds_binned = AIRS_cds_binned.transpose(0, 1, 3, 2)
    FGS1_cds_binned = FGS1_cds_binned.transpose(0, 1, 3, 2)
    del AIRS_cds, FGS1_cds
    
    # --- Flat field correction ---
    for i in range(CHUNKS_SIZE):
        flat_airs = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/AIRS-CH0_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        flat_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{index_chunk[i][0]}/FGS1_calibration_{index_chunk[i][1]}/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        if DO_FLAT:
            corrected_AIRS_cds_binned = correct_flat_field(flat_airs, dead_airs, AIRS_cds_binned[i])
            AIRS_cds_binned[i] = corrected_AIRS_cds_binned
            corrected_FGS1_cds_binned = correct_flat_field(flat_fgs, dead_fgs1, FGS1_cds_binned[i])
            FGS1_cds_binned[i] = corrected_FGS1_cds_binned
    
    AIRS_cds_binned = AIRS_cds_binned.transpose(0, 1, 3, 2)
    FGS1_cds_binned = FGS1_cds_binned.transpose(0, 1, 3, 2)
    
    # --- Inpainting missing/bad pixels ---
    # FGS1: inpaint along time axis as channels
    data = FGS1_cds_binned[0, :, :, :].data         # shape: (time, x, y)
    mask = FGS1_cds_binned[0, 0, :, :].mask         # shape: (x, y)
    data = data.transpose(1, 2, 0)                 # shape: (x, y, time)
    result_fgs1 = inpaint_biharmonic(data, mask, channel_axis=2)
    result_fgs1 = result_fgs1.transpose(2, 0, 1)   # shape: (time, x, y)
    
    # AIRS: inpaint along wavelength axis as channels
    data_airs = AIRS_cds_binned[0, :, :, :].data      # shape: (time, x, lambda)
    mask_airs = AIRS_cds_binned[0, 0, :, :].mask      # shape: (x, lambda)
    data_airs = data_airs.transpose(1, 2, 0)          # shape: (x, lambda, time)
    result_airs = inpaint_biharmonic(data_airs, mask_airs, channel_axis=2)
    result_airs = result_airs.transpose(2, 0, 1)
    
    # --- Save cleaned and inpainted data ---
    chunk_name = '__'.join([f"{pid}_{obs}" for pid, obs in index_chunk])
    np.savez(os.path.join(path_out, f'AIRS_clean_train_{chunk_name}.npz'), data=AIRS_cds_binned.data, mask=AIRS_cds_binned.mask)
    np.savez(os.path.join(path_out, f'FGS1_clean_train_{chunk_name}.npz'), data=FGS1_cds_binned.data, mask=FGS1_cds_binned.mask)
    np.savez(os.path.join(path_out, f'AIRS_cleaninp_train_{chunk_name}.npz'), data=result_airs, mask=AIRS_cds_binned.mask[0, :, :, :])
    np.savez(os.path.join(path_out, f'FGS1_cleaninp_train_{chunk_name}.npz'), data=result_fgs1, mask=FGS1_cds_binned.mask[0, :, :, :])
    
    # Save index mapping as JSON for easy lookup
    import json
    index_chunk_serializable = [
        [int(pid), int(obs)] if (not isinstance(pid, str)) else [str(pid), int(obs)]
        for pid, obs in index_chunk
    ]
    with open(os.path.join(path_out, f'AIRS_clean_train_{chunk_name}_index.json'), 'w') as f_json:
        json.dump(index_chunk_serializable, f_json)
    
    del AIRS_cds_binned
    del FGS1_cds_binned


# Read ADC info
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
airs_gain = adc['AIRS-CH0_adc_gain'].iloc[0]
airs_offset = adc['AIRS-CH0_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 356)
orig_imgs_cal = (orig_imgs / airs_gain) + airs_offset

proc_path = '/kaggle/working/data_light_raw/AIRS_clean_train_1143471509_0.npz'
proc_imgs = np.load(proc_path)  # shape: (num_frames, 32, l), l = cut_sup - cut_inf
proc_imgs = np.ma.MaskedArray(proc_imgs['data'], proc_imgs['mask'])

proc_imgs=proc_imgs[0,:,:,:]
cut_inf, cut_sup = 0, 356     # as in your cleaning pipeline

# Crop the original to compare matching region
orig_imgs_cropped = orig_imgs_cal[:, :, cut_inf:cut_sup]

frames_to_plot = 9  # or any count you like
plt.figure(figsize=(10, 22))
for idx in range(frames_to_plot):
    # Plot original (calibrated, cropped) -- use odd frames only
    orig_idx = 2 * idx + 1
    plt.subplot(frames_to_plot, 2, 2*idx + 1)
    plt.imshow(orig_imgs_cropped[orig_idx], cmap='gray', aspect='auto')
    plt.title(f'Original Frame {orig_idx}')
    plt.axis('off')
    # Plot processed (already cleaned)
    plt.subplot(frames_to_plot, 2, 2*idx + 2)
    plt.imshow(proc_imgs[idx], cmap='gray', aspect='auto')
    plt.title(f'Processed Frame {idx}')
    plt.axis('off')
plt.tight_layout()
plt.show()




# Read ADC info
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
airs_gain = adc['AIRS-CH0_adc_gain'].iloc[0]
airs_offset = adc['AIRS-CH0_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0
cut_inf, cut_sup = 0, 356  # adjust if needed as in preprocessing

# Load original, processed, and inpainted data

# --- Original ---
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 356)
orig_imgs_cal = (orig_imgs / airs_gain) + airs_offset
orig_imgs_cropped = orig_imgs_cal[:, :, cut_inf:cut_sup]

# --- Processed (masked) ---
proc_path = '/kaggle/working/data_light_raw/AIRS_clean_train_1143471509_0.npz'
proc_npz = np.load(proc_path)
proc_imgs = np.ma.MaskedArray(proc_npz['data'], mask=proc_npz['mask'])
proc_imgs = proc_imgs[0, :, :, :]  # shape: [frames, 32, cropped_wavelength]

# --- Inpainted ---
inpaint_path = '/kaggle/working/data_light_raw/AIRS_cleaninp_train_1143471509_0.npz'
inp_npz = np.load(inpaint_path)
inp_imgs = inp_npz['data']  # This is fully filled, not masked
inp_mask = inp_npz['mask']  # Mask for reference if you want to display or highlight masked pixels

# Select frame range for comparison
frames_to_plot = 9  # or any count you like
plt.figure(figsize=(12, 22))
for idx in range(frames_to_plot):
    # Plot original (calibrated, cropped) -- use odd frames only
    orig_idx = 2 * idx + 1
    plt.subplot(frames_to_plot, 3, 3*idx + 1)
    plt.imshow(orig_imgs_cropped[orig_idx], cmap='gray', aspect='auto')
    plt.title(f'Original Frame {orig_idx}')
    plt.axis('off')
    
    # Plot processed (masked)
    plt.subplot(frames_to_plot, 3, 3*idx + 2)
    plt.imshow(np.ma.filled(proc_imgs[idx], fill_value=np.nan), cmap='gray', aspect='auto')
    plt.title(f'Processed Frame {idx}')
    plt.axis('off')
    
    # Plot inpainted (filled)
    plt.subplot(frames_to_plot, 3, 3*idx + 3)
    plt.imshow(inp_imgs[idx], cmap='gray', aspect='auto')
    plt.title(f'Inpainted Frame {idx}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# --- Load ADC info for FGS1 ---
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
fgs1_gain = adc['FGS1_adc_gain'].iloc[0]
fgs1_offset = adc['FGS1_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0

# --- ORIGINAL FGS1 SIGNAL ---
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 32)
orig_imgs_cal = (orig_imgs / fgs1_gain) + fgs1_offset

# --- PROCESSED FGS1 DATA ---
# Use your saved processed file, e.g., "/kaggle/working/data_light_raw/FGS1_clean_train_1143471509_0.npy"
proc_path = '/kaggle/working/data_light_raw/FGS1_clean_train_1143471509_0.npz'
proc_imgs = np.load(proc_path)  # shape: (num_frames, 32, 32) for a single planet/obs. If shape is e.g., (batch, num_frames, 32, 32): use proc_imgs = proc_imgs[0]
proc_imgs = np.ma.MaskedArray(proc_imgs['data'], proc_imgs['mask'])
proc_imgs = proc_imgs[0,:,:,:]
frames_to_plot = 9
plt.figure(figsize=(10, 22))
for idx in range(frames_to_plot):
    # Plot original (calibrated)
    plt.subplot(frames_to_plot, 2, 2*idx + 1)
    plt.imshow(orig_imgs_cal[idx], cmap='gray')
    plt.title(f'Original Frame {idx}')
    plt.axis('off')
    # Plot processed
    plt.subplot(frames_to_plot, 2, 2*idx + 2)
    plt.imshow(proc_imgs[idx], cmap='gray')
    plt.title(f'Processed Frame {idx}')
    plt.axis('off')
plt.tight_layout()
plt.show()



# --- Load ADC info for FGS1 ---
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
fgs1_gain = adc['FGS1_adc_gain'].iloc[0]
fgs1_offset = adc['FGS1_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0

# --- ORIGINAL FGS1 SIGNAL ---
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 32)
orig_imgs_cal = (orig_imgs / fgs1_gain) + fgs1_offset

# --- PROCESSED FGS1 DATA ---
proc_path = '/kaggle/working/data_light_raw/FGS1_clean_train_1143471509_0.npz'
proc_npz = np.load(proc_path)
proc_imgs = np.ma.MaskedArray(proc_npz['data'], mask=proc_npz['mask'])
proc_imgs = proc_imgs[0,:,:,:]  # [num_frames, 32, 32]

# --- INPAINTED FGS1 DATA ---
inpaint_path = '/kaggle/working/data_light_raw/FGS1_cleaninp_train_1143471509_0.npz'
inpaint_npz = np.load(inpaint_path)
inpaint_imgs = inpaint_npz['data']  # shape: [num_frames, 32, 32]
inpaint_mask = inpaint_npz['mask']  # [32, 32] or [num_frames, 32, 32] as reference

frames_to_plot = 9
plt.figure(figsize=(12, 22))
for idx in range(frames_to_plot):
    # Plot original
    plt.subplot(frames_to_plot, 3, 3*idx + 1)
    plt.imshow(orig_imgs_cal[idx], cmap='gray')
    plt.title(f'Original Frame {idx}')
    plt.axis('off')

    # Plot processed (masked)
    plt.subplot(frames_to_plot, 3, 3*idx + 2)
    plt.imshow(np.ma.filled(proc_imgs[idx], fill_value=np.nan), cmap='gray')
    plt.title(f'Processed Frame {idx}')
    plt.axis('off')

    # Plot inpainted (filled)
    plt.subplot(frames_to_plot, 3, 3*idx + 3)
    plt.imshow(inpaint_imgs[idx], cmap='gray')
    plt.title(f'Inpainted Frame {idx}')
    plt.axis('off')

plt.tight_layout()
plt.show()



# Load ADC info
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
airs_gain = adc['AIRS-CH0_adc_gain'].iloc[0]
airs_offset = adc['AIRS-CH0_adc_offset'].iloc[0]

# Identify your planet and observation number
planet_id = 1143471509
obs_num = 0

# Load original data, calibrate, crop
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 356)
orig_imgs_cal = (orig_imgs / airs_gain) + airs_offset
cut_inf, cut_sup = 0, 356
orig_imgs_cropped = orig_imgs_cal[:, :, cut_inf:cut_sup]

# Load processed data: shape (num_frames, 32, l)
proc_path = f'/kaggle/working/data_light_raw/AIRS_clean_train_{planet_id}_{obs_num}.npz'
proc_imgs = np.load(proc_path)
proc_imgs = np.ma.MaskedArray(proc_imgs['data'], proc_imgs['mask'])
if proc_imgs.ndim == 4:
    proc_imgs = proc_imgs[0]  # Select first batch if there is a chunk dimension

# Extract center slit (middle spatial row)
mid_idx = orig_imgs_cropped.shape[1] // 2
orig_center = orig_imgs_cropped[:, mid_idx, :]  # Shape: (time, wavelength)
proc_center = proc_imgs[:, mid_idx, :]          # Shape: (time, wavelength)
print(type(proc_center))
# Plot original vs processed (time x wavelength images)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(orig_center.T, aspect='auto', cmap='gray')
plt.title('Original Center-Slit (Time × Wavelength)')
plt.xlabel('Time Frame')
plt.ylabel('Wavelength Bin')
plt.colorbar(label='Calibrated Flux')

plt.subplot(1, 2, 2)
plt.imshow(proc_center.T, aspect='auto', cmap='gray')
plt.title('Processed Center-Slit (Time × Wavelength)')
plt.xlabel('Time Frame')
plt.ylabel('Wavelength Bin')
plt.colorbar(label='Processed Flux')

plt.tight_layout()
plt.show()



# --- Load ADC info for FGS1 ---
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
fgs1_gain = adc['FGS1_adc_gain'].iloc[0]
fgs1_offset = adc['FGS1_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0

# --- ORIGINAL FGS1 SIGNAL ---
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 32)
orig_imgs_cal = (orig_imgs / fgs1_gain) + fgs1_offset

# --- PROCESSED FGS1 DATA ---
proc_path = '/kaggle/working/data_light_raw/FGS1_clean_train_1143471509_0.npz'
proc_imgs = np.load(proc_path)
proc_imgs = np.ma.MaskedArray(proc_imgs['data'], proc_imgs['mask'])
if proc_imgs.ndim == 4:
    proc_imgs = proc_imgs[0]

# Calculate white light curves
lc_orig = orig_imgs_cal.sum(axis=(1,2))
lc_proc = proc_imgs.sum(axis=(1,2))

# Subsample original to match processed (odd and even indices)
lc_orig_odd = lc_orig[1::2]
lc_orig_even = lc_orig[0::2]

# Normalize all curves
lc_orig_odd_norm = lc_orig_odd / lc_orig_odd.mean()
lc_orig_even_norm = lc_orig_even / lc_orig_even.mean()
lc_proc_norm = lc_proc / lc_proc.mean()

plt.figure(figsize=(12,6))
plt.plot(lc_orig_even_norm, label='Original (even indices)', alpha=0.7)
plt.plot(lc_orig_odd_norm, label='Original (odd indices)', alpha=0.7)
plt.plot(lc_proc_norm, label='Processed', alpha=0.8, linewidth=2)
plt.xlabel('Time (binned frame index)')
plt.ylabel('Normalized flux in the frame')
plt.title(f'FGS1 White Light Curve: Even vs Odd Original Indices and Processed\nPlanet {planet_id}')
plt.legend()
plt.tight_layout()
plt.show()



# --- Load ADC info for FGS1 ---
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
fgs1_gain = adc['FGS1_adc_gain'].iloc[0]
fgs1_offset = adc['FGS1_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0

# --- ORIGINAL FGS1 SIGNAL ---
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 32)
orig_imgs_cal = (orig_imgs / fgs1_gain) + fgs1_offset

# --- PROCESSED FGS1 DATA ---
proc_path = '/kaggle/working/data_light_raw/FGS1_clean_train_1143471509_0.npz'
proc_imgs = np.load(proc_path)
proc_imgs = np.ma.MaskedArray(proc_imgs['data'], proc_imgs['mask'])
if proc_imgs.ndim == 4:
    proc_imgs = proc_imgs[0]

# Calculate white light curves
lc_orig = orig_imgs_cal.sum(axis=(1,2))
lc_proc = proc_imgs.sum(axis=(1,2))

# Subsample the original to match processed (even and odd indices)
lc_orig_even = lc_orig[0::2]
lc_orig_odd = lc_orig[1::2]

# Normalize all curves
lc_orig_even_norm = lc_orig_even / lc_orig_even.mean()
lc_orig_odd_norm = lc_orig_odd / lc_orig_odd.mean()
lc_proc_norm = lc_proc / lc_proc.mean()

# Plot three separate, aligned subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)

axes[0].plot(lc_orig_even_norm, color='C0')
axes[0].set_ylabel('Normalized Flux')
axes[0].set_title('Original (even indices)')

axes[1].plot(lc_orig_odd_norm, color='C1')
axes[1].set_ylabel('Normalized Flux')
axes[1].set_title('Original (odd indices)')

axes[2].plot(lc_proc_norm, color='C2')
axes[2].set_ylabel('Normalized Flux')
axes[2].set_title('Processed (binned/cleaned)')

axes[2].set_xlabel('Time (binned frame index)')
plt.tight_layout()
plt.show()



# Compute absolute difference between consecutive points
jumps = np.abs(np.diff(lc_orig_even_norm))

# Find index of largest jump
jump_idx = np.argmax(jumps)

print(f"Largest jump between frames {jump_idx} and {jump_idx+1}")
print(f"Jump value: {lc_orig_even_norm[jump_idx+1] - lc_orig_even_norm[jump_idx]:.4f}")

# Optionally, plot with a marker
plt.plot(lc_orig_even_norm)
plt.axvline(jump_idx, color='red', linestyle='--', label='Largest jump')
plt.xlabel('Frame Index')
plt.ylabel('Normalized White Light Flux')
plt.title('ORIGINAL EVEN White Light Curve With Largest Jump Marked')
plt.legend()
plt.show()



light_curve = lc_proc_norm
print(type(light_curve))

# Compute absolute difference between consecutive points
jumps = np.abs(np.diff(light_curve))

# Find index of largest jump
jump_idx = np.argmax(jumps)

print(f"Largest jump between frames {jump_idx} and {jump_idx+1}")
print(f"Jump value: {light_curve[jump_idx+1] - light_curve[jump_idx]:.4f}")

# Optionally, plot with a marker
plt.plot(light_curve)
plt.axvline(jump_idx, color='red', linestyle='--', label='Largest jump')
plt.xlabel('Frame Index')
plt.ylabel('Normalized White Light Flux')
plt.title('White Light Curve With Largest Jump Marked')
plt.legend()
plt.show()



# --- Load ADC info for AIRS-CH0 ---
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
airs_gain = adc['AIRS-CH0_adc_gain'].iloc[0]
airs_offset = adc['AIRS-CH0_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0

# --- ORIGINAL AIRS-CH0 SIGNAL ---
orig_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_{obs_num}.parquet'
orig_df = pd.read_parquet(orig_path)
orig_imgs = orig_df.values.reshape(-1, 32, 356)
orig_imgs_cal = (orig_imgs / airs_gain) + airs_offset

# --- PROCESSED AIRS-CH0 DATA ---
proc_path = f'/kaggle/working/data_light_raw/AIRS_clean_train_{planet_id}_{obs_num}.npz'
proc_imgs_npz = np.load(proc_path)
proc_imgs = np.ma.MaskedArray(proc_imgs_npz['data'], proc_imgs_npz['mask'])
if proc_imgs.ndim == 4:
    proc_imgs = proc_imgs[0]

# Optionally crop to match spectral region if necessary
cut_inf, cut_sup = 39, 321  # Set to your analysis region
orig_imgs_cropped = orig_imgs_cal[:, :, cut_inf:cut_sup]
proc_imgs_cropped = proc_imgs[:, :, :]  # If processed already cropped

# Calculate white light curves
lc_orig = orig_imgs_cropped.sum(axis=(1,2))
lc_proc = proc_imgs_cropped.sum(axis=(1,2))

# Subsample the original to match processed (even and odd indices)
lc_orig_even = lc_orig[0::2]
lc_orig_odd = lc_orig[1::2]

# Normalize all curves
lc_orig_even_norm = lc_orig_even / lc_orig_even.mean()
lc_orig_odd_norm = lc_orig_odd / lc_orig_odd.mean()
lc_proc_norm = lc_proc / lc_proc.mean()

# Plot three separate, aligned subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)

axes[0].plot(lc_orig_even_norm, color='C0')
axes[0].set_ylabel('Normalized Flux')
axes[0].set_title('AIRS Original (even indices)')

axes[1].plot(lc_orig_odd_norm, color='C1')
axes[1].set_ylabel('Normalized Flux')
axes[1].set_title('AIRS Original (odd indices)')

axes[2].plot(lc_proc_norm, color='C2')
axes[2].set_ylabel('Normalized Flux')
axes[2].set_title('AIRS Processed (binned/cleaned)')

axes[2].set_xlabel('Time (binned frame index)')
plt.tight_layout()
plt.show()



# -- LOAD ADC info --
adc = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
airs_gain = adc['AIRS-CH0_adc_gain'].iloc[0]
airs_offset = adc['AIRS-CH0_adc_offset'].iloc[0]
fgs1_gain = adc['FGS1_adc_gain'].iloc[0]
fgs1_offset = adc['FGS1_adc_offset'].iloc[0]

planet_id = 1143471509
obs_num = 0

# -- LOAD ORIGINAL AIRS-CH0 --
airs_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_{obs_num}.parquet'
airs_df = pd.read_parquet(airs_path)
orig_airs = airs_df.values.reshape(-1, 32, 356)
orig_airs_cal = (orig_airs / airs_gain) + airs_offset
cut_inf, cut_sup = 39, 321  # Adjust as needed
orig_airs_crop = orig_airs_cal[:, :, cut_inf:cut_sup]

# -- LOAD PROCESSED AIRS (masked .npz) --
airs_proc_path = f'/kaggle/working/data_light_raw/AIRS_clean_train_{planet_id}_{obs_num}.npz'
airs_proc_npz = np.load(airs_proc_path)
airs_proc = np.ma.MaskedArray(airs_proc_npz['data'], airs_proc_npz['mask'])
if airs_proc.ndim == 4:
    airs_proc = airs_proc[0]

# -- AIRS White Light Curves --
lc_airs_orig = orig_airs_crop.sum(axis=(1,2))
lc_airs_proc = airs_proc.sum(axis=(1,2))
lc_airs_even = lc_airs_orig[0::2]
lc_airs_odd = lc_airs_orig[1::2]
lc_airs_even_norm = lc_airs_even / lc_airs_even.mean()
lc_airs_odd_norm = lc_airs_odd / lc_airs_odd.mean()
lc_airs_proc_norm = lc_airs_proc / lc_airs_proc.mean()

# -- LOAD ORIGINAL FGS1 --
fgs1_path = f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_{obs_num}.parquet'
fgs1_df = pd.read_parquet(fgs1_path)
orig_fgs1 = fgs1_df.values.reshape(-1, 32, 32)
orig_fgs1_cal = (orig_fgs1 / fgs1_gain) + fgs1_offset

# -- LOAD PROCESSED FGS1 (masked .npz) --
fgs1_proc_path = f'/kaggle/working/data_light_raw/FGS1_clean_train_{planet_id}_{obs_num}.npz'
fgs1_proc_npz = np.load(fgs1_proc_path)
fgs1_proc = np.ma.MaskedArray(fgs1_proc_npz['data'], fgs1_proc_npz['mask'])
if fgs1_proc.ndim == 4:
    fgs1_proc = fgs1_proc[0]

# -- FGS1 White Light Curves --
lc_fgs1_orig = orig_fgs1_cal.sum(axis=(1,2))
lc_fgs1_proc = fgs1_proc.sum(axis=(1,2))
lc_fgs1_even = lc_fgs1_orig[0::2]
lc_fgs1_odd = lc_fgs1_orig[1::2]
lc_fgs1_even_norm = lc_fgs1_even / lc_fgs1_even.mean()
lc_fgs1_odd_norm = lc_fgs1_odd / lc_fgs1_odd.mean()
lc_fgs1_proc_norm = lc_fgs1_proc / lc_fgs1_proc.mean()

# -- PLOT 3x2 grid: left = AIRS, right = FGS1 --
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex='col', sharey='row')

# AIRS plots
axes[0,0].plot(lc_airs_even_norm, color='C0')
axes[0,0].set_title('AIRS Original (even)')
axes[0,0].set_ylabel('Normalized Flux')

axes[1,0].plot(lc_airs_odd_norm, color='C1')
axes[1,0].set_title('AIRS Original (odd)')
axes[1,0].set_ylabel('Normalized Flux')

axes[2,0].plot(lc_airs_proc_norm, color='C2')
axes[2,0].set_title('AIRS Processed')
axes[2,0].set_ylabel('Normalized Flux')
axes[2,0].set_xlabel('Time (binned frame index)')

# FGS1 plots
axes[0,1].plot(lc_fgs1_even_norm, color='C0')
axes[0,1].set_title('FGS1 Original (even)')

axes[1,1].plot(lc_fgs1_odd_norm, color='C1')
axes[1,1].set_title('FGS1 Original (odd)')

axes[2,1].plot(lc_fgs1_proc_norm, color='C2')
axes[2,1].set_title('FGS1 Processed')
axes[2,1].set_xlabel('Time (binned frame index)')

# Adjust layout
plt.tight_layout()
plt.show()



lc_airs_proc_norm.shape


lc_fgs1_proc_norm.shape

