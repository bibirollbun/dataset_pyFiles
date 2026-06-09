import numpy.ma as ma

import numba
import numpy as np
import pandas as pd
import itertools
import os
import glob 
from astropy.stats import sigma_clip

from tqdm import tqdm


path_folder = '/kaggle/input/ariel-data-challenge-2025/' # This path stays the same
path_out = '/kaggle/working/data_light_raw/' # Updated to save files permanently
output_dir = '/kaggle/working/data_light_raw/' # Updated to save files permanently


if not os.path.exists(path_out):
    os.makedirs(path_out)
    print(f"Directory {path_out} created.")
else:
    print(f"Directory {path_out} already exists.")


chunks_size = 5 #because if we load all the files together it might cause an memory overload



def correct_ADC_convert(signal, gain=0.4369, offset=-1000):
    """Restores the full dynamic range of the signal according to the competition rules."""
    # Convert to float for calculations
    signal = signal.astype(np.float64)
    
    # 1. Multiply by gain
    signal *= gain
    
    # 2. Add offset
    signal += offset
    
    return signal



def mask_hot_dead(signal, dead, dark):
    """
    Masks hot and dead pixels in a 4D signal array.
    """
    # --- THIS IS THE FIX ---
    # Ensure the input signal is a masked array so it has a .mask attribute
    signal = ma.asarray(signal)

    # Find hot pixels from the 2D dark frame
    hot_mask_2d = sigma_clip(dark, sigma=5, maxiters=5).mask
    
    # Combine the 2D hot and dead pixel maps
    combined_mask_2d = dead | hot_mask_2d
    
    # Directly update the signal's mask using broadcasting.
    # np.ma.mask_or safely combines the old and new masks.
    new_mask = combined_mask_2d[np.newaxis, np.newaxis, :, :]
    signal.mask = ma.mask_or(signal.mask, new_mask)
    
    return signal


@numba.njit
def apply_linear_corr(linear_corr, clean_signal):
    """
    Applies a unique polynomial linearity correction to each pixel.
    This version is optimized for speed using Numba.
    """
    # Create a copy to avoid changing the original input array
    corrected_signal = np.copy(clean_signal)

    # Loop through each pixel (Numba makes these loops extremely fast)
    for x in range(clean_signal.shape[1]):
        for y in range(clean_signal.shape[2]):
            # Loop through the time dimension for each pixel
            for t in range(clean_signal.shape[0]):
                
                # This manually evaluates the polynomial for the current pixel.
                # It's a fast implementation of what np.poly1d does.
                
                # Get the coefficients and the value for the current pixel
                coeffs = linear_corr[:, x, y]
                val = clean_signal[t, x, y]
                
                # Apply the polynomial
                result = coeffs[0]
                for i in range(1, len(coeffs)):
                    result = result * val + coeffs[i]
                
                clean_signal[t, x, y] = result
                
    return clean_signal



def clean_dark(signal, dead, dark, dt):
    """
    Subtracts a time-scaled dark frame from a signal using broadcasting.
    """
    # 1. Mask the dead pixels within the dark frame itself.
    dark_masked = np.ma.masked_where(dead, dark)

    # 2. Use broadcasting to subtract the scaled dark frame.
    #    NumPy automatically "stretches" the 2D dark_masked and 1D dt arrays
    #    to match the 3D signal's shape during the calculation.
    corrected_signal = signal - dark_masked[np.newaxis, :, :] * dt[:, np.newaxis, np.newaxis]

    return signal


def get_cds(signal):
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds


def bin_obs(cds_signal, binning):
    """
    Optimized time binning using reshape and sum.
    This version handles cases where the time axis is not perfectly divisible.
    """
    # Get the original shape
    num_planets, num_frames, height, width = cds_signal.shape
    
    # --- THIS IS THE FIX ---
    # Calculate the largest length that is perfectly divisible by the binning factor
    new_length = (num_frames // binning) * binning
    
    # Trim the array to that new length, discarding the extra frames at the end
    trimmed_signal = cds_signal[:, :new_length, :, :]

    # Calculate the new number of frames after binning
    new_num_frames = trimmed_signal.shape[1] // binning
    
    # Reshape the *trimmed* signal, which will now work correctly
    cds_binned = trimmed_signal.reshape(num_planets, 
                                           new_num_frames, 
                                           binning, 
                                           height, 
                                           width).sum(axis=2)
    
    return cds_binned


def correct_flat_field(signal, flat, dead):
    """
    Optimized flat-field correction using broadcasting.
    """
    # 1. Mask the dead pixels within the flat frame.
    # The .transpose() lines are removed.
    flat_masked = ma.masked_where(dead, flat)

    # 2. Use broadcasting to divide the signal by the 2D flat field.
    # This adds new axes to the flat_masked array to align it with the
    # 4D signal array shape: (chunk, time, height, width)
    return signal / flat_masked[np.newaxis, np.newaxis, :, :]


def get_index(files,chunks_size ):
    index = []
    for file in files:
        file_name = file.split('/')[-1]
        if file_name.split('_')[0] == 'AIRS-CH0' and file_name.split('_')[1] == 'signal' and file_name.split('_')[2] == '0.parquet':
            file_index = os.path.basename(os.path.dirname(file))
            index.append(int(file_index))
    index = np.array(index)
    index = np.sort(index) 
    # credit to DennisSakva
    index=np.array_split(index, len(index)//chunks_size)
    
    return index


files = glob.glob(os.path.join(path_folder, 'train/*/*'))
index = get_index(files, chunks_size) 
axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))

DO_MASK = True
DO_THE_NL_CORR = False # Linearity correction is slow and complex, skipped for this baseline
DO_DARK = True
DO_FLAT = True
TIME_BINNING = True

cut_inf, cut_sup = 39, 321
l = cut_sup - cut_inf

# =============================================================================
# 3. MAIN PROCESSING LOOP
# =============================================================================
for n, index_chunk in enumerate(tqdm(index)):
    
    # --- 1. EFFICIENTLY LOAD ALL DATA FOR THE CHUNK ---
    airs_signals = [pd.read_parquet(os.path.join(path_folder,f'train/{pid}/AIRS-CH0_signal_0.parquet')).values.reshape(-1, 32, 356) for pid in index_chunk]
    fgs1_signals = [pd.read_parquet(os.path.join(path_folder,f'train/{pid}/FGS1_signal_0.parquet')).values.reshape(-1, 32, 32) for pid in index_chunk]
    
    AIRS_CH0_clean = np.stack(airs_signals).astype(np.float64)[:, :, :, cut_inf:cut_sup]
    FGS1_clean = np.stack(fgs1_signals).astype(np.float64)
    
    # --- Load calibration files ONCE per chunk ---
    first_pid = index_chunk[0]
    flat_airs = pd.read_parquet(os.path.join(path_folder,f'train/{first_pid}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
    dark_airs = pd.read_parquet(os.path.join(path_folder,f'train/{first_pid}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
    
    # *** THIS IS THE FIX FOR THE TypeError ***
    dead_airs = pd.read_parquet(os.path.join(path_folder,f'train/{first_pid}/AIRS-CH0_calibration_0/dead.parquet')).values.reshape((32, 356))[:, cut_inf:cut_sup] > 0

    flat_fgs = pd.read_parquet(os.path.join(path_folder,f'train/{first_pid}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
    dark_fgs = pd.read_parquet(os.path.join(path_folder,f'train/{first_pid}/FGS1_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 32))
    
    # *** THIS IS THE FIX FOR THE TypeError ***
    dead_fgs = pd.read_parquet(os.path.join(path_folder,f'train/{first_pid}/FGS1_calibration_0/dead.parquet')).values.reshape((32, 32)) > 0

    # --- 2. APPLY VECTORIZED CALIBRATIONS TO THE ENTIRE CHUNK ---
    
    AIRS_CH0_clean = correct_ADC_convert(AIRS_CH0_clean)
    FGS1_clean = correct_ADC_convert(FGS1_clean)
    
    dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
    dt_airs[1::2] += 0.1
    dt_fgs1 = np.ones(FGS1_clean.shape[1]) * 0.1
    dt_fgs1[1::2] += 0.1

    if DO_MASK:
        AIRS_CH0_clean = mask_hot_dead(AIRS_CH0_clean, dead_airs, dark_airs)
        FGS1_clean = mask_hot_dead(FGS1_clean, dead_fgs, dark_fgs)
    
    if DO_DARK:
        # Note: Broadcasting dt requires careful reshaping for the 4D array
        dt_airs_b = dt_airs.reshape(1, -1, 1, 1)
        dt_fgs1_b = dt_fgs1.reshape(1, -1, 1, 1)
        AIRS_CH0_clean = AIRS_CH0_clean - dark_airs[np.newaxis, np.newaxis, :, :] * dt_airs_b
        FGS1_clean = FGS1_clean - dark_fgs[np.newaxis, np.newaxis, :, :] * dt_fgs1_b

    # --- 3. POST-CALIBRATION PROCESSING ---
    
    AIRS_cds = get_cds(AIRS_CH0_clean)
    FGS1_cds = get_cds(FGS1_clean)
    del AIRS_CH0_clean, FGS1_clean

    if TIME_BINNING:
        AIRS_cds_binned = bin_obs(AIRS_cds, binning=30)
        FGS1_cds_binned = bin_obs(FGS1_cds, binning=30*12)
    else:
        AIRS_cds_binned = AIRS_cds
        FGS1_cds_binned = FGS1_cds
    del AIRS_cds, FGS1_cds

    if DO_FLAT:
        AIRS_cds_binned = correct_flat_field(AIRS_cds_binned, flat_airs, dead_airs)
        FGS1_cds_binned = correct_flat_field(FGS1_cds_binned, flat_fgs, dead_fgs)
    
    # --- 4. SAVE THE PROCESSED CHUNK ---
  # Use the .filled() method to replace masked values with np.nan before saving
    np.save(os.path.join(path_out, f'AIRS_clean_train_{n}.npy'), AIRS_cds_binned.filled(np.nan))
    np.save(os.path.join(path_out, f'FGS1_train_{n}.npy'), FGS1_cds_binned.filled(np.nan))
    del AIRS_cds_binned, FGS1_cds_binned


def get_chunk_number(filepath):
    """Extracts the integer number from a chunked filename."""
    return int(filepath.split('_')[-1].split('.')[0])
def create_features(airs_npy_files, fgs1_npy_files, labels_df, chunk_size):
    """
    This function takes paths to calibrated AIRS and FGS1 .npy files
    and calculates features, processing in batches.
    """
    all_feature_chunks = []
    
    # A helper function to extract the number from the filename
    def get_chunk_number(filepath):
        return int(filepath.split('_')[-1].split('.')[0])
    
    # Sort the file lists numerically to ensure order
    airs_npy_files = sorted(airs_npy_files, key=get_chunk_number)
    fgs1_npy_files = sorted(fgs1_npy_files, key=get_chunk_number)
    
    print("Processing calibrated data to create features...")
    for i in tqdm(range(len(airs_npy_files))):
        airs_data_chunk = np.load(airs_npy_files[i])
        fgs1_data_chunk = np.load(fgs1_npy_files[i])

        # Create Light Curves
        airs_light_curves = np.nansum(airs_data_chunk, axis=(2, 3))
        fgs1_light_curves = np.nansum(fgs1_data_chunk, axis=(2, 3))

        # Calculate Dip Depth
        start_transit, end_transit = 75, 105
        
        fgs1_out = np.nanmean(np.concatenate([fgs1_light_curves[:, :start_transit], fgs1_light_curves[:, end_transit:]], axis=1), axis=1)
        fgs1_in = np.nanmean(fgs1_light_curves[:, start_transit:end_transit], axis=1)
        fgs1_dip_depth = (fgs1_out - fgs1_in) / fgs1_out

        airs_out = np.nanmean(np.concatenate([airs_light_curves[:, :start_transit], airs_light_curves[:, end_transit:]], axis=1), axis=1)
        airs_in = np.nanmean(airs_light_curves[:, start_transit:end_transit], axis=1)
        # Corrected this line to divide by 'out' of transit brightness
        airs_dip_depth = (airs_out - airs_in) / airs_out

        # Get the correct Planet IDs for this chunk
        start_index = i * chunk_size
        end_index = start_index + len(airs_data_chunk)
        chunk_planet_ids = labels_df.iloc[start_index:end_index]['planet_id']

        # Create a feature DataFrame for this chunk
        chunk_feature_df = pd.DataFrame({
            'planet_id': chunk_planet_ids,
            'fgs1_dip_depth': fgs1_dip_depth,
            'airs_dip_depth': airs_dip_depth
        })
        all_feature_chunks.append(chunk_feature_df)
            
    # Concatenate the list of DataFrames into one final DataFrame
    return pd.concat(all_feature_chunks).reset_index(drop=True)


# =============================================================================
# PART 1: SETUP (Assumes all helper functions are defined in the cells above)
# =============================================================================
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Paths and Parameters ---
# This path should point to the output of your FIRST notebook (the calibration one)
calibrated_train_path = '/kaggle/working/data_light_raw/' 
competition_data_path = '/kaggle/input/ariel-data-challenge-2025/'
CHUNK_SIZE_TRAIN = 5
CHUNK_SIZE_TEST = 5 # Using a larger chunk size for the test set is fine

# =============================================================================
# PART 2: FEATURE ENGINEERING AND MODEL TRAINING
# =============================================================================

# --- Create Training Features ---
print("--- Creating Training Features ---")
# (This assumes the create_features function is defined above)
train_airs_files = glob.glob(os.path.join(calibrated_train_path, 'AIRS_clean_train_*.npy'))
train_fgs1_files = glob.glob(os.path.join(calibrated_train_path, 'FGS1_train_*.npy'))
train_labels = pd.read_csv(os.path.join(competition_data_path, 'train.csv'))
train_labels = train_labels.sort_values(by='planet_id').reset_index(drop=True)
feature_df = create_features(train_airs_files, train_fgs1_files, train_labels, CHUNK_SIZE_TRAIN)
print("Training features created.")

# --- Train Models ---
X = feature_df.drop(columns=['planet_id'])
y = train_labels.drop(columns=['planet_id'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
models = []
uncertainties = []

print("\n--- Training Models ---")
for i in tqdm(range(y_train.shape[1]), desc="Training Models"):
    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model.fit(X_train, y_train.iloc[:, i])
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], val_preds))
    models.append(model)
    uncertainties.append(rmse)
uncertainties = np.array(uncertainties)
print("Model training complete.")

# =============================================================================
# PART 3: PREDICTION PIPELINE (for the hidden test set)
# =============================================================================

test_path = os.path.join(competition_data_path, 'test')
if os.path.exists(test_path):
    print("\n--- Test Data Found: Starting Full Inference Pipeline ---")
    
    # --- A: CALIBRATE RAW TEST DATA ---
    print("Step A: Calibrating raw test data...")
    path_folder = competition_data_path
    path_out = '/kaggle/working/calibrated_test/' # Temporary folder for calibrated test files
    if not os.path.exists(path_out): os.makedirs(path_out)

    sample_submission = pd.read_csv(os.path.join(path_folder, 'sample_submission.csv'))
    test_planet_ids = np.sort(sample_submission['planet_id'].unique())
    num_chunks = len(test_planet_ids) // CHUNK_SIZE_TEST
    if num_chunks == 0 and len(test_planet_ids) > 0: num_chunks = 1
    test_index_chunks = np.array_split(test_planet_ids, num_chunks)
    
    axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))
    DO_MASK, DO_DARK, DO_FLAT, TIME_BINNING = True, True, True, True
    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf

    for n, index_chunk in enumerate(tqdm(test_index_chunks, desc="Calibrating Test Chunks")):
        airs_signals = [pd.read_parquet(os.path.join(path_folder, f'test/{pid}/AIRS-CH0_signal_0.parquet')).values.reshape(-1, 32, 356) for pid in index_chunk]
        fgs1_signals = [pd.read_parquet(os.path.join(path_folder, f'test/{pid}/FGS1_signal_0.parquet')).values.reshape(-1, 32, 32) for pid in index_chunk]
        
        AIRS_CH0_clean = np.stack(airs_signals).astype(np.float64)[:, :, :, cut_inf:cut_sup]
        FGS1_clean = np.stack(fgs1_signals).astype(np.float64)
        
        first_pid = index_chunk[0]
        flat_airs = pd.read_parquet(os.path.join(path_folder, f'test/{first_pid}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dark_airs = pd.read_parquet(os.path.join(path_folder, f'test/{first_pid}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dead_airs = pd.read_parquet(os.path.join(path_folder, f'test/{first_pid}/AIRS-CH0_calibration_0/dead.parquet')).values.reshape((32, 356))[:, cut_inf:cut_sup] > 0
        flat_fgs = pd.read_parquet(os.path.join(path_folder, f'test/{first_pid}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        dark_fgs = pd.read_parquet(os.path.join(path_folder, f'test/{first_pid}/FGS1_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 32))
        dead_fgs = pd.read_parquet(os.path.join(path_folder, f'test/{first_pid}/FGS1_calibration_0/dead.parquet')).values.reshape((32, 32)) > 0
        
        AIRS_CH0_clean = correct_ADC_convert(AIRS_CH0_clean)
        FGS1_clean = correct_ADC_convert(FGS1_clean)
        
        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1
        dt_fgs1 = np.ones(FGS1_clean.shape[1]) * 0.1
        dt_fgs1[1::2] += 0.1

        if DO_MASK:
            AIRS_CH0_clean = mask_hot_dead(AIRS_CH0_clean, dead_airs, dark_airs)
            FGS1_clean = mask_hot_dead(FGS1_clean, dead_fgs, dark_fgs)
        if DO_DARK:
            AIRS_CH0_clean = clean_dark(AIRS_CH0_clean, dead_airs, dark_airs, dt_airs)
            FGS1_clean = clean_dark(FGS1_clean, dead_fgs, dark_fgs, dt_fgs1)
            
        AIRS_cds = get_cds(AIRS_CH0_clean)
        FGS1_cds = get_cds(FGS1_clean)
        del AIRS_CH0_clean, FGS1_clean

        if TIME_BINNING:
            AIRS_cds_binned = bin_obs(AIRS_cds, binning=30)
            FGS1_cds_binned = bin_obs(FGS1_cds, binning=30*12)
        else:
            AIRS_cds_binned = AIRS_cds
            FGS1_cds_binned = FGS1_cds
        del AIRS_cds, FGS1_cds

        if DO_FLAT:
            AIRS_cds_binned = correct_flat_field(AIRS_cds_binned, flat_airs, dead_airs)
            FGS1_cds_binned = correct_flat_field(FGS1_cds_binned, flat_fgs, dead_fgs)

        np.save(os.path.join(path_out, f'AIRS_clean_test_{n}.npy'), AIRS_cds_binned.filled(np.nan))
        np.save(os.path.join(path_out, f'FGS1_clean_test_{n}.npy'), FGS1_cds_binned.filled(np.nan))

    # --- B: CREATE TEST FEATURES ---
    print("\nStep B: Creating features for test data...")
    test_airs_files = glob.glob(os.path.join(path_out, 'AIRS_clean_test_*.npy'))
    test_fgs1_files = glob.glob(os.path.join(path_out, 'FGS1_clean_test_*.npy'))
    test_feature_df = create_features(test_airs_files, test_fgs1_files, sample_submission, CHUNK_SIZE_TEST)

    # --- C: MAKE PREDICTIONS & SUBMIT ---
    print("\nStep C: Making final predictions...")
    X_test = test_feature_df.drop(columns=['planet_id'])
    test_planet_ids = test_feature_df['planet_id']
    
    test_predictions = np.zeros((len(X_test), len(models)))
    for i, model in enumerate(tqdm(models, desc="Predicting")):
        test_predictions[:, i] = model.predict(X_test)

    test_uncertainties = np.tile(uncertainties, (len(X_test), 1))
    
    pred_df = pd.DataFrame(test_predictions, columns=[f'wl_{i+1}' for i in range(283)])
    unc_df = pd.DataFrame(test_uncertainties, columns=[f'sigma_{i+1}' for i in range(283)])
    submission_df = pd.DataFrame({'planet_id': test_planet_ids})
    submission_df = pd.concat([submission_df, pred_df, unc_df], axis=1)
    
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission file created successfully!")
    
else:
    print("\nTest data not found. This was likely a training run.")


pd.read_csv("submission.csv")










