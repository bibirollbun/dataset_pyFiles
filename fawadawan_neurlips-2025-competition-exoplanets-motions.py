import pandas as pd
from warnings import filterwarnings

filterwarnings('ignore')


adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
adc_info


BASE_DIR = '/kaggle/input/ariel-data-challenge-2025/'

train_csv = pd.read_csv(f'{BASE_DIR}/train.csv')
train_csv


train_star_info = pd.read_csv(f'{BASE_DIR}/train_star_info.csv')
train_star_info


train_star_info.columns


test_star_info = pd.read_csv(f'{BASE_DIR}/test_star_info.csv')
test_star_info


train_csv['planet_id'] == train_star_info['planet_id']


wl_csv = pd.read_csv(f'{BASE_DIR}/wavelengths.csv')
wl_csv


wl_csv.columns


axis_info = pd.read_parquet(f'{BASE_DIR}/axis_info.parquet')
axis_info


sample_submission = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')
sample_submission


sample_submission.columns


TRAIN_DIR = BASE_DIR + "/train"

airs_0_signal = pd.read_parquet(f'{TRAIN_DIR}/1010375142/AIRS-CH0_signal_0.parquet')
airs_0_signal


fgs1_0 = pd.read_parquet(f'{TRAIN_DIR}/1010375142/FGS1_signal_0.parquet')
fgs1_0


airs_ch0_calibration_dark = pd.read_parquet(f'{TRAIN_DIR}/1010375142/AIRS-CH0_calibration_0/dark.parquet')
airs_ch0_calibration_dark


airs_ch0_calib_dead = pd.read_parquet(f'{TRAIN_DIR}/1010375142/AIRS-CH0_calibration_0/dead.parquet')
airs_ch0_calib_dead


airs_ch0_flat = pd.read_parquet(f'{TRAIN_DIR}/1010375142/AIRS-CH0_calibration_0/flat.parquet')
airs_ch0_flat


airs_ch0_linear_corr = pd.read_parquet(f'{TRAIN_DIR}/1010375142/AIRS-CH0_calibration_0/linear_corr.parquet')
airs_ch0_linear_corr


airs_ch0_calib_read = pd.read_parquet(f'{TRAIN_DIR}/1010375142/AIRS-CH0_calibration_0/read.parquet')
airs_ch0_calib_read


fgs1_calibration_dark = pd.read_parquet(f'{TRAIN_DIR}/1010375142/FGS1_calibration_0/dark.parquet')
fgs1_calibration_dark


fgs1_calibration_dead = pd.read_parquet(f'{TRAIN_DIR}/1010375142/FGS1_calibration_0/dead.parquet')
fgs1_calibration_dead


fgs1_calibration_flat = pd.read_parquet(f'{TRAIN_DIR}/1010375142/FGS1_calibration_0/flat.parquet')
fgs1_calibration_flat


fgs1_calibration_0_lin_corr = pd.read_parquet(f'{TRAIN_DIR}/1010375142/FGS1_calibration_0/linear_corr.parquet')
fgs1_calibration_0_lin_corr


fgs1_calib0_read = pd.read_parquet(f'{TRAIN_DIR}/1010375142/FGS1_calibration_0/read.parquet')
fgs1_calib0_read


airs_ch0_signal_0 = pd.read_parquet(f'{TRAIN_DIR}/1010375142/AIRS-CH0_signal_0.parquet')
airs_ch0_signal_0


fgs1_signal_0 = pd.read_parquet(f'{TRAIN_DIR}/1010375142/FGS1_signal_0.parquet')
fgs1_signal_0


import torch
from torch import nn
from torch.optim import Adam

NUM_WAVELENGTH_OUTPUT_CHANNELS = 283 

# class SpectrumRegressor(torch.nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         # The network should map the 3 input features to a large output vector
#         # that is 2 * 283 = 566 elements long.
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, NUM_WAVELENGTH_OUTPUT_CHANNELS * 2) # <-- This is the critical line
#         )

#     def forward(self, x):
#         out = self.net(x)
#         # out has shape (batch_size, 566)
        
#         # Split the output into two equal chunks for mu and sigma
#         mu = out[..., :NUM_WAVELENGTH_OUTPUT_CHANNELS]
#         sigma = torch.nn.functional.softplus(out[..., NUM_WAVELENGTH_OUTPUT_CHANNELS:])
#         return mu, sigma

class SpectrumRegressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim): # Add output_dim as a parameter
        super().__init__()
        
        # The network should map input_dim features to output_dim * 2 elements (mu and sigma)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), # First layer takes input_dim (e.g., 364)
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim * 2) # Last layer produces output_dim * 2 (e.g., 283 * 2 = 566)
        )
        self.output_dim = output_dim # Store it for the forward pass

    def forward(self, x):
        out = self.net(x)
        
        # Split the output into two equal chunks for mu and sigma
        mu = out[..., :self.output_dim]
        # Ensure sigma has a minimum value to prevent numerical instability in loss
        sigma_raw = out[..., self.output_dim:]
        sigma = torch.nn.functional.softplus(sigma_raw).clamp(min=1e-6) 
        
        return mu, sigma


import numpy as np
import pandas as pd
from torch import tensor
import torch
import os
from warnings import filterwarnings

filterwarnings('ignore')

# --- 0. Configuration Constants ---
# These constants define the dimensions based on data formats
AIRS_CH0_SPECTRAL_DIM = 356 
AIRS_CH0_SPATIAL_DIM = 32

# Number of features from train_star_info.csv
NUM_STELLAR_FEATURES = 8 # ('Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i')

# Number of features derived from axis_info.parquet
# We'll use global means of: AIRS-CH0-axis0-h, AIRS-CH0-integration_time, FGS1-axis0-h
NUM_AXIS_INFO_FEATURES = 3

# Total input features for the model = (features from AIRS-CH0 signal) + (stellar features) + (axis info features)
# Here, AIRS-CH0 signal features are `AIRS_CH0_SPECTRAL_DIM` (356)
# TOTAL_MODEL_INPUT_DIM = AIRS_CH0_SPECTRAL_DIM + NUM_STELLAR_FEATURES + NUM_AXIS_INFO_FEATURES # 356 + 3 + 3 = 362
TOTAL_MODEL_INPUT_DIM = NUM_STELLAR_FEATURES  # The number of features we are using

OUTPUT_DIR = '/kaggle/working/'
DATA_ROOT = "/kaggle/input/ariel-data-challenge-2025"
TRAIN_DIR = f"{DATA_ROOT}/train"


def load_calibration(calib_dir, gain_val, offset_val):
    calib_data = {}
    
    # Iterate through files in the calibration directory
    for fname in os.listdir(calib_dir):
        file_path = os.path.join(calib_dir, fname)
        data = pd.read_parquet(file_path).values

        # Determining the key for the dictionary entry (e.g.,
        # 'dark', 'flat', 'dead', 'linear_corr'). Being a bit
        # more robust here, taking the part before the first '.'
        # and then splitting by '_' just in case (e.g.,
        # 'linear_corr.parquet' -> 'linear_corr')
        key = fname.split('.')[0] 
        # Specific reshaping logic for common calibration files like dark,
        #  flat, dead. These should typically be (spatial_dim, spectral_dim)
        if data.ndim == 1 and data.size == AIRS_CH0_SPATIAL_DIM * AIRS_CH0_SPECTRAL_DIM:
            data = data.reshape(AIRS_CH0_SPATIAL_DIM, AIRS_CH0_SPECTRAL_DIM)
        
        # For 'dead' mask, ensure it's boolean
        if key == 'dead':
            calib_data[key] = data.astype(bool)
        else:
            calib_data[key] = data

    # Explicitly add the gain and offset values provided as arguments
    calib_data['gain'] = gain_val
    calib_data['offset'] = offset_val
    
    return calib_data


def correct_signal(raw, calib):
    '''
    Takes raw 3D signal and 2D calibration data, corrects it, and averages it.
    Input `raw` shape: (num_timesteps, 32, 356)
    Output shape: (356,)
    '''
    # Applying a linear scaling to the raw sensor data using gain and offset.
    # This works because NumPy correctly broadcasts the scalar values.
    signal = raw * calib['gain'] + calib['offset']

    # Subtracting the dark frame.
    # NumPy correctly broadcasts the 2D calib['dark'] array to subtract
    # it from each time-step slice of the 3D 'signal' array.
    signal -= calib['dark']

    # Dividing by the flat field.
    # Again, NumPy correctly broadcasts the 2D calib['flat'] array.
    signal /= calib['flat']

    # Creating a mask based on dead pixels.
    # NumPy correctly broadcasts the 2D calib['dead'] mask.
    mask = calib['dead']
    signal = np.where(mask, np.nan, signal)

    # Now, after all the corrections have been applied,
    # perform the averaging over the time and spatial axes.
    corrected_and_averaged_signal = np.mean(signal, axis=(0, 1))

    # Replace any nan or inf values in the final, averaged result.
    return np.nan_to_num(corrected_and_averaged_signal)


import os

def load_planet_instrument_data(planet_root_dir, inst, adc_info_df):
    """
    Loads and corrects signal data for a single instrument of a given planet.
    """
    signal_file_path = os.path.join(planet_root_dir, f"{inst}_signal_0.parquet")
    raw_signal_flat = pd.read_parquet(signal_file_path).values

    calib_dir_path = os.path.join(planet_root_dir, f"{inst}_calibration_0")

    current_inst_gain = adc_info_df[f"{inst}_adc_gain"].iloc[0]
    current_inst_offset = adc_info_df[f"{inst}_adc_offset"].iloc[0]

    calib = load_calibration(calib_dir_path, current_inst_gain, current_inst_offset)

    if inst == 'AIRS-CH0':
        num_timesteps = raw_signal_flat.shape[0]
        AIRS_CH0_SPATIAL_DIM = 32
        AIRS_CH0_SPECTRAL_DIM = 356
        raw_signal_3d = raw_signal_flat.reshape(
            (num_timesteps, AIRS_CH0_SPATIAL_DIM, AIRS_CH0_SPECTRAL_DIM)
        )
        
        # Now, call correct_signal with the full 3D array.
        # This function will now return the final, averaged (356,) vector.
        return correct_signal(raw_signal_3d, calib)
        
    elif inst == 'FGS1':
        # ... FGS1 logic ...
        # Assuming FGS1 calibration is simpler, this part might need its own
        # averaging within correct_signal.
        return correct_signal(raw_signal_flat, calib)

    else:
        raise ValueError(f"Unknown instrument '{inst}'. Cannot determine expected shape.")


def load_ground_truth_spectrum(pid, train_labels_df):
    """
    Loads the true exoplanet transmission spectrum (ground truth, y) for a given planet ID.
    """
    if pid not in train_labels_df.index:
        raise ValueError(f"Ground truth spectrum for planet {pid} not found in train.csv!")
    
    # Select all columns that represent the spectrum (e.g., 'wl_1' to 'wl_283')
    # and convert to a NumPy array of float32.
    spectrum_values = train_labels_df.loc[pid].filter(like='wl_').values
    return spectrum_values.astype(np.float32)


def load_all_global_data(data_root):
    """
    Loads all global CSV/Parquet files (adc_info, train_star_info, wavelengths, train.csv, axis_info.parquet) once.
    """
    print("Loading global data files from:", data_root)
    
    adc_info_df = pd.read_csv(os.path.join(data_root, 'adc_info.csv'))
    train_star_info_df = pd.read_csv(os.path.join(data_root, 'train_star_info.csv'))
    wavelengths = pd.read_csv(os.path.join(data_root, 'wavelengths.csv')).iloc[0].values
    train_labels_df = pd.read_csv(os.path.join(data_root, 'train.csv'))

    train_star_info_df.set_index(['planet_id'], inplace=True)
    train_labels_df.set_index(['planet_id'], inplace=True)
    
    # --- NEW: Load axis_info.parquet ---
    axis_info_df = pd.read_parquet(os.path.join(data_root, 'axis_info.parquet'))
    # Fill NaNs as you indicated (using column means)
    axis_info_df = axis_info_df.fillna(axis_info_df.mean(numeric_only=True))

    # Derive global summary features from axis_info.parquet
    # We'll use the mean of a few relevant columns as features for every planet.
    # Note: If axis_info.parquet has specific rows per planet, you'd need
    # a more sophisticated mapping here. This approach uses global averages.
    global_axis_features = np.array([
        axis_info_df['AIRS-CH0-axis0-h'].mean(),
        axis_info_df['AIRS-CH0-integration_time'].mean(),
        axis_info_df['FGS1-axis0-h'].mean()
    ]).astype(np.float32)

    print(f" - ADC Info Shape: {adc_info_df.shape}")
    print(f" - Star Info Shape: {train_star_info_df.shape}")
    print(f" - Wavelengths Count: {len(wavelengths)}")
    print(f" - Train Labels (Ground Truth) Shape: {train_labels_df.shape}")
    print(f" - Axis Info Shape: {axis_info_df.shape}")
    print(f" - Derived Global Axis Features: {global_axis_features.shape} (e.g., {global_axis_features})")

    # Return all loaded dataframes and the derived global axis features
    return adc_info_df, train_star_info_df, wavelengths, train_labels_df, global_axis_features


# def processing_single_planet(pid, train_dir, global_data, is_train=True):
#     # This is a dummy function. Replace with your actual processing logic.
#     # It takes the planet ID as a string and the global data tuple.
#     adc_info_df, train_star_info_df, wavelengths, train_labels_df, global_axis_features = global_data
#     planet_root_dir = f"{train_dir}/{pid}"
    
#     # Check for ground truth and star info before attempting to process
#     if pid not in train_labels_df.index:
#         raise ValueError(f"Ground truth spectrum for planet {pid_int} not found in train_labels.csv!")
#     if pid not in train_star_info_df.index:
#         raise ValueError(f"Star info for planet {pid} not found in train_star_info.csv!")

#     # FGS1 is not loaded because it does only provide the stability
#     # AIRS-CH0 is loaded because it provides the spectrographical
#     # signals
#     airs_ch0_data = load_planet_instrument_data(
#         planet_root_dir, 'AIRS-CH0', adc_info_df
#     )
#     # Extract Features from AIRS-CH0 signal
#     signal_features = airs_ch0_data
    
#     star_row = train_star_info_df.loc[pid]
#     stellar_features = np.array([
#         star_row['Rs'],
#         star_row['Ms'],
#         star_row['Ts'],
#     ], dtype=np.float32)

#     # Combine all input features
#     combined_input_features = np.concatenate([signal_features, stellar_features, global_axis_features])

#     ground_truth_spectrum = None
#     if is_train:
#         # load_ground_truth_spectrum will raise ValueError if not found
#         ground_truth_spectrum = load_ground_truth_spectrum(pid, train_labels_df)

#     return combined_input_features, ground_truth_spectrum


import os
import numpy as np
from pathlib import Path
# from multiprocessing import Pool, cpu_count
# import functools

# --- Define a helper function to process a single planet's data ---
# This function does the work for one planet and can be parallelized.
def process_single_planet(pid, TRAIN_DIR, train_star_info_df, adc_info_df):
    """Processes all data for a single planet ID."""
    print(f"Processing planet: {pid}")
    planet_root_dir = f"{TRAIN_DIR}/{pid}"
    
    # Load and extract signal features
    airs_ch0_data = load_planet_instrument_data(
        planet_root_dir, 'AIRS-CH0', adc_info_df
    )
    signal_features = airs_ch0_data
    
    # Get stellar features
    star_row = train_star_info_df.loc[pid]
    stellar_features = np.array([
        star_row['Rs'], star_row['Ms'], star_row['Ts'], star_row['Mp'],
        star_row['e'], star_row['P'], star_row['sma'], star_row['i'],
    ], dtype=np.float32)
    
    combined_features = np.concatenate([stellar_features, signal_features])

    return combined_features


import os
import numpy as np
from pathlib import Path
import time
# import functools
# from tqdm.auto import tqdm

# train_star_info dataframe's features
# Index(
#  ['planet_id', 'Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i'],
#  dtype='object'
#)

def consolidate_data():
    print("------------- Starting path allocation -------------")

    start_path_alloc = time.perf_counter()
    
    feature_file = f'{OUTPUT_DIR}combined_input_features.npy'
    ground_truth_file = f'{OUTPUT_DIR}/ground_truth_spectrum.npy'
    global_axis_file = f"{OUTPUT_DIR}/global_axis_features.npy"
    # numpy_files_dir = os.path.join(OUTPUT_DIR, "consolidated_data")
    
    # os.makedirs(numpy_files_dir, exist_ok=True)
    
    # feature_dir = os.path.join(numpy_files_dir, feature_file)
    # ground_truth_dir = os.path.join(numpy_files_dir, ground_truth_file)
    # global_axis_features_dir = os.path.join(numpy_files_dir, "global_axis_features.npy")
    
    feature_path = Path(feature_file)
    ground_truth_path = Path(ground_truth_file)
    global_axis_features_path = Path(global_axis_file)

    if feature_path.is_file() and ground_truth_path.is_file() and global_axis_features_path.is_file():
        end_path_alloc = time.perf_counter()
        print(f"Time consumed for path allocation: {end_path_alloc - start_path_alloc:.2f}")
        
        print("Files of features and labels arrays already exist.")
        return feature_file, ground_truth_file, global_axis_features_file
    else:
        end_path_alloc = time.perf_counter()
        print(f"Time consumed for path allocation: {end_path_alloc - start_path_alloc:.2f}")
        
        print("------------- Starting arrays creation -------------")
        start_arr_creation = time.perf_counter()
        
        print("------------- Loading global data -------------")
        start_load_global = time.perf_counter()
        try:
            adc_info_df, train_star_info_df, _, train_labels_df, global_axis_features = load_all_global_data(DATA_ROOT)
        except FileNotFoundError as e:
            print(f"\nERROR: Required data file not found: {e}")
            return
        end_load_global = time.perf_counter()
        print(f"Global data loaded in {end_load_global - start_load_global:.2f} seconds.")
        
        dir_ids_set = set(
            [int(d) for d in os.listdir(TRAIN_DIR) if d.isdigit()]
        )
        labels_ids_set = set(train_labels_df.index.to_list())
        star_info_ids_set = set(train_star_info_df.index.to_list())
        training_pids_set = dir_ids_set \
                            .intersection(labels_ids_set) \
                            .intersection(star_info_ids_set)
        training_planet_ids = sorted(list(training_pids_set))

        if not training_planet_ids:
            print("\nERROR: No valid data for training.")
            return

        print(f"------------- Found {len(training_planet_ids)} valid planets. Consolidating data -------------")

        features = []
        labels = []

        i = 0
        start_training_time = time.perf_counter()
        for pid in training_planet_ids:
            print(f"Getting planet id at {i} index")
            combined_features = process_single_planet(
                pid,
                TRAIN_DIR,
                train_star_info_df,
                adc_info_df
            )
            features.append(combined_features)
            print(f"Getting the truth data of {pid}")
            true_label = train_labels_df.loc[pid].values.astype(np.float32)
            labels.append(true_label)
            i += 1
        end_training_time = time.perf_counter()
        print(f"Time taken for training {len(training_planet_ids)} planets: {end_training_time - start_training_time:.2f}")

        # --- Making arrays ---
        start_arr_time  = time.perf_counter()
        features_arr = np.array(features)
        ground_truth_arr = np.array(labels)
        end_arr_time = time.perf_counter()
        print(f"Making arrays took: {end_arr_time - start_arr_time:.2f}")
        
        # --- Save Files ---
        start_saving_time = time.perf_counter()
        np.save(feature_file, features_arr)
        np.save(ground_truth_file, ground_truth_arr)
        np.save(global_axis_file, global_axis_features)
        end_saving_time = time.perf_counter()
        print(f"Saving files took {end_saving_time - start_saving_time:.2f} seconds.")

        end_arr_creation = time.perf_counter()
        print(f"Total function runtime: {end_arr_creation - start_arr_creation:.2f} seconds.")
        
        print("Features and ground truth labels saved successfully!!")
        return feature_file, ground_truth_file, global_axis_file


# Uncomment and run this code if there're no .npy files in the datasets
# features_dir, ground_truth_dir, global_axis_file = consolidate_data()
# features_dir, ground_truth_dir, global_axis_file


features_path = '/kaggle/input/neurlips-2025-exoplanet-arrays/combined_input_features.npy'
labels_path = '/kaggle/input/neurlips-2025-exoplanet-arrays/ground_truth_spectrum.npy'
global_axis_path = '//kaggle/input/neurlips-2025-exoplanet-arrays/ground_truth_spectrum.npy'


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PlanetDataset(Dataset):
    def __init__(self, features_path, labels_path):
        # Load the entire consolidated dataset into memory once
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Simply retrieve the data from the pre-loaded arrays
        features_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return features_tensor, labels_tensor


def gll_loss(mu, sigma, y):
    """
    Calculates the negative log-likelihood of observing y given predicted mu and sigma.
    """
    # Use torch.mean to get the average loss over the batch
    return torch.mean(0.5 * torch.log(2 * np.pi * sigma**2) + (y - mu)**2 / (2 * sigma**2))


features_arr = np.load(features_path)
X_shape = features_arr.shape
X_shape


y_shape = np.load(labels_path).shape
y_shape


def run_training_pipeline_with_gpu():
    print("--- Starting Exoplanet Spectrum Training Pipeline (GPU-Optimized) ---")

    # Checking consolidated data 
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create the Dataset and DataLoader ---
    train_dataset = PlanetDataset(
        features_path,
        labels_path
    )
    
    # DataLoader handles batching and parallel data loading
    # Batch size: how many samples to process at once on the GPU
    # num_workers: how many CPU cores to use for data loading (set to 0 for a simple test)
    # pin_memory: faster data transfer to the GPU
    BATCH_SIZE = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # Shuffle the data for training
        num_workers=4, # Use multiple CPU cores to load data
        pin_memory=True
    )
    print(f"\nCreated DataLoader with {len(train_loader)} batches of size {BATCH_SIZE}.")

    # --- Step 4: Move Model to GPU ---
    model = SpectrumRegressor(X_shape[1], y_shape[1]).to(device)
    opt = Adam(model.parameters(), 1e-4)

    # --- Step 5: Training Loop using DataLoader ---
    print("\n--- Starting Model Training ---")
    num_epochs = 1000
    for epoch in range(num_epochs):
        print(f"Going in {epoch}")
        model.train()
        total_loss = 0
        
        # Iterate over the DataLoader, which yields batches of data
        for X_batch, y_batch in train_loader:
            # Move batch data to the GPU
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            opt.zero_grad()
            mu, sigma = model(X_batch)
            loss = gll_loss(mu, sigma, y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch + 1:02d}/{num_epochs} | Average Loss: {avg_loss:.6f}")

    print("\n--- Model Training Complete! ---")
    print(f"Final Average Loss: {avg_loss:.6f}")
    return model


# No need to run
# model = run_training_pipeline_with_gpu()


'''
I wrote this code to saved the model as a .pth file with its
entire learned weights. No need to run this code.
'''

# import torch

# torch.save(model.state_dict(), "/kaggle/working/spectrum_regressor_weights.pth")


# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader

# class PlanetTestDataset(Dataset):
#     def __init__(self, test_x, test_y):
#         self.test_x = test_x
#         self.test_y = test_y

#     def __len__(self):
#         return len(self.test_x)

#     def __getitem__(self, idx):
#         x = torch.tensor(self.test_x[idx], dtype=torch.float32)
#         y = torch.tensor(self.test_y[idx], dtype=torch.float32)

#         return x, y


'''
I wrote this code to create the planets_id.npy.
You don't need to run this!!
'''
# import numpy as np

# try:
#     _, train_star_info_df, _, train_labels_df, _ = load_all_global_data(DATA_ROOT)
# except FileNotFoundError as e:
#     print(f"\nERROR: Required data file not found: {e}")

# dir_ids_set = set(
#     [int(d) for d in os.listdir(TRAIN_DIR) if d.isdigit()]
# )
# labels_ids_set = set(train_labels_df.index.to_list())
# star_info_ids_set = set(train_star_info_df.index.to_list())
# training_pids_set = dir_ids_set \
#                     .intersection(labels_ids_set) \
#                     .intersection(star_info_ids_set)
# training_planet_ids = sorted(list(training_pids_set))

# planet_ids_arr = np.array(training_planet_ids)

# np.save('/kaggle/working/planet_ids.npy', planet_ids_arr)


sample_submission


train_csv.set_index('planet_id', inplace = True)
train_csv


train_csv.loc[4294092928]


import os
import numpy as np
import pandas as pd

def process_test_planet(pid, test_dir, train_star_info_df, adc_info_df):
    test_star_info_dir = '/kaggle/input/ariel-data-challenge-2025/test_star_info.csv'
    planet_dir = f"{test_dir}/{pid}"
    test_star_info_df = pd.read_csv(test_star_info_dir)
    test_star_info_df.set_index('planet_id', inplace=True)
    
    airs_ch0_data = load_planet_instrument_data(
        planet_dir, 'AIRS-CH0', adc_info_df
    )
    signal_features = airs_ch0_data
    
    test_planet_star_info = test_star_info_df.loc[pid]
    stellar_features = np.array([
        test_planet_star_info['Rs'], test_planet_star_info['Ms'], test_planet_star_info['Ts'], test_planet_star_info['Mp'],
        test_planet_star_info['e'], test_planet_star_info['P'], test_planet_star_info['sma'], test_planet_star_info['i']
    ], dtype=np.float32)
    combined_features = np.concatenate((signal_features, stellar_features))
    
    return combined_features


'''No need to run this!'''
# import os
# import pandas as pd
# import numpy as np

# test_feature_file = f'{OUTPUT_DIR}test_feature_file.npy'

# try:
#     adc_info_df, train_star_info_df, _, _, _ = load_all_global_data(DATA_ROOT)
# except FileNotFoundError as e:
#     print(f"\nERROR: Required data file not found: {e}")

# TEST_DIR = '/kaggle/input/ariel-data-challenge-2025/test'
# test_planets_ids = [int(pid) for pid in os.listdir(TEST_DIR)]

# if not test_planets_ids:
#     print("\nERROR: No valid data for testing.")

# print(f"------------- Found {len(test_planets_ids)} testing planets. Consolidating data -------------")

# features = []
# i = 0

# for pid in test_planets_ids:
#     print(f"Getting planet id at {i} index")
#     combined_features = process_test_planet(
#         pid,
#         TEST_DIR,
#         train_star_info_df,
#         adc_info_df
#     )
#     features.append(combined_features)
#     i += 1

# features_arr = np.array(features)
# np.save(test_feature_file, features_arr)

# print(f"------------- Saved the testing array as {test_feature_file} -------------")


import numpy as np

test_feature_dir = '/kaggle/input/neurlips-2025-exoplanet-arrays/test_feature_file.npy'
test_arr = np.load(test_feature_dir)
test_arr.shape


from torch.utils.data import Dataset, DataLoader
import torch

class PlanetRegressorTestSet(Dataset):
    def __init__(self):
        self.test_arr = test_arr

    def __len__(self):
        return len(self.test_arr)

    def __getitem__(self, idx):
        return torch.tensor(self.test_arr[idx], dtype=torch.float32)


test_planets = np.array([1103775])
test_planets.shape


import pandas as pd

output_dim = 283

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device at hand: ", device)
model_path = '/kaggle/input/spectrum-regressor/pytorch/default/1/spectrum_regressor_weights.pth'
loaded_dict = torch.load(model_path, map_location = device)
model = SpectrumRegressor(test_arr.shape[1], output_dim).to(device)

BATCH_SIZE = 32
test_dataset = PlanetRegressorTestSet()
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model.load_state_dict(loaded_dict)

all_mu_preds = []
all_sigma_preds = []

print("\n--- Generating Predictions for Submission ---")
i = 0
model.eval()

with torch.no_grad(): # Disable gradient calculations
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        mu, log_sigma = model(X_batch) # Get mu and log_sigma
        sigma = torch.exp(log_sigma) # Convert log_sigma to sigma

        all_mu_preds.append(mu.cpu().numpy())
        all_sigma_preds.append(sigma.cpu().numpy())
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} batches...")
        i += 1

mu_preds_array = np.vstack(all_mu_preds)
sigma_preds_array = np.vstack(all_sigma_preds)

print(f"Shape of aggregated mu predictions: {mu_preds_array.shape}")
print(f"Shape of aggregated sigma predictions: {sigma_preds_array.shape}")

# Create column names
output_dim = model.output_dim
wl_cols = [f"wl_{i}" for i in range(1, output_dim + 1)] 
sigma_cols = [f"sigma_{i}" for i in range(1, output_dim + 1)]
    
combined_preds_array = np.hstack((mu_preds_array, sigma_preds_array))

submission_df = pd.DataFrame(combined_preds_array, columns=wl_cols + sigma_cols)
submission_df.insert(0, 'planet_id', test_planets)

output_filename = f'{OUTPUT_DIR}/submission.csv'
submission_df.to_csv(output_filename, index=False)
    
print(f"\nSubmission file '{output_filename}' created successfully!")
print(f"Final submission file shape: {submission_df.shape}")

submission_df


sample_submission

