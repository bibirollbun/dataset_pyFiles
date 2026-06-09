!pip install tensorflow-model-optimization>=0.7.2 

!pip install astropy>=1.1.2 
!pip install tensorflow
!pip install colorama>=0.4.4 
!pip install imbalanced-learn==0.7.0
!pip install joblib>=1.2.0
!pip install optuna>=2.10.0
!pip install george 


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from scipy import stats, optimize as op
import george
from george import kernels
import warnings
from functools import partial
from typing import Dict, List, Union
from astropy.table import Table, vstack

# Suppress warnings
warnings.filterwarnings("ignore")



# Constants
NUM_CLASSES = 15  # Including 'others' class
TIME_STEPS = 30   # Sequence length for time series
FILTERS = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']
METADATA_FEATURES = ['hostgal_photoz', 'mwebv', 'ddf']
ALL_FEATURES = FILTERS + METADATA_FEATURES

# PLAsTiCC class mapping to sequential indices
PLASTICC_CLASS_MAPPING = {
    90: 0,   # SNIa
    67: 1,   # SNIa-91bg
    52: 2,   # SNIax
    42: 3,   # SNII
    62: 4,   # SNIbc
    95: 5,   # SLSN-I
    15: 6,   # TDE
    64: 7,   # KN
    88: 8,   # AGN
    92: 9,   # RRL
    65: 10,  # M-dwarf
    16: 11,  # EB
    53: 12,  # Mira
    6: 13,   # µ-Lens-Single
    99: 14   # Others (not in training set)
}

# Class names for reference
CLASS_NAMES = {
    0: 'SNIa',
    1: 'SNIa-91bg',
    2: 'SNIax',
    3: 'SNII',
    4: 'SNIbc',
    5: 'SLSN-I',
    6: 'TDE',
    7: 'KN',
    8: 'AGN',
    9: 'RRL',
    10: 'M-dwarf',
    11: 'EB',
    12: 'Mira',
    13: 'µ-Lens-Single',
    14: 'Others'
}

# LSST passband wavelengths
LSST_PB_WAVELENGTHS = {
    'lsstu': 3671.0,
    'lsstg': 4827.0,
    'lsstr': 6223.0,
    'lssti': 7546.0,
    'lsstz': 8691.0,
    'lssty': 9710.0
}



def load_and_merge_data(data_path, metadata_path):
    """Load and merge light curve data with metadata"""
    data = pd.read_csv(data_path)
    metadata = pd.read_csv(metadata_path)
    
    # Merge and clean data
    merged = pd.merge(data, metadata, on='object_id', how='left')
    
    # Handle missing values
    merged = merged.dropna(subset=['flux', 'flux_error'] + METADATA_FEATURES)
    
    return merged

def filter_dataframe_only_supernova(object_list_filename: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe that contains many classes to only Supernovae types."""
    plasticc_object_list = np.genfromtxt(object_list_filename, dtype="U")
    filtered_dataframe = dataframe[dataframe["object_id"].isin(plasticc_object_list)]
    return filtered_dataframe

def transient_trim(object_list: List[str], df: pd.DataFrame) -> tuple:
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps"""
    adf = pd.DataFrame(data=[], columns=df.columns)
    good_object_list = []
    for obj in object_list:
        obs = df[df["object_id"] == obj]
        obs_time = obs["mjd"]
        obs_detected_time = obs_time[obs["detected"] == 1]
        if len(obs_detected_time) == 0:
            print(f"Zero detected points for object:{object_list.index(obj)}")
            continue
        is_obs_transient = (obs_time > obs_detected_time.iat[0] - 50) & (
            obs_time < obs_detected_time.iat[-1] + 50
        )
        obs_transient = obs[is_obs_transient]
        if len(obs_transient["mjd"]) == 0:
            is_obs_transient = (obs_time > obs_detected_time.iat[0] - 1000) & (
                obs_time < obs_detected_time.iat[-1] + 1000
            )
            obs_transient = obs[is_obs_transient]
        obs_transient["mjd"] -= min(obs_transient["mjd"])  # so all transients start at time 0
        good_object_list.append(object_list.index(obj))
        adf = np.vstack((adf, obs_transient))

    obs_transient = pd.DataFrame(data=adf, columns=obs_transient.columns)
    filter_indices = good_object_list
    new_filtered_object_list = np.take(np.array(object_list), filter_indices, axis=0)
    return obs_transient, list(new_filtered_object_list)

def fit_2d_gp(obj_data: pd.DataFrame, return_kernel: bool = False, pb_wavelengths: Dict = LSST_PB_WAVELENGTHS, **kwargs):
    """Fit a 2D Gaussian process."""
    guess_length_scale = 20.0

    obj_times = obj_data.mjd.astype(float)
    obj_flux = obj_data.flux.astype(float)
    obj_flux_error = obj_data.flux_error.astype(float)
    obj_wavelengths = obj_data["filter"].map(pb_wavelengths)

    def neg_log_like(p):
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    signal_to_noises = np.abs(obj_flux) / np.sqrt(
        obj_flux_error**2 + (1e-2 * np.max(obj_flux)) ** 2
    )
    scale = np.abs(obj_flux[signal_to_noises.idxmax()])

    kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel(
        [guess_length_scale**2, 6000**2], ndim=2
    )
    kernel.freeze_parameter("k2:metric:log_M_1_1")

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000**2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(
        neg_log_like,
        gp.get_parameter_vector(),
        jac=grad_neg_log_like,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-6,
    )

    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        obj = obj_data["object_id"][0]
        print(f"GP fit failed for {obj}! Using guessed GP parameters.")
        gp.set_parameter_vector(default_gp_param)

    gp_predict = partial(gp.predict, obj_flux)

    if return_kernel:
        return kernel, gp_predict
    return gp_predict

def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
    """Outputs the predictions of a Gaussian Process."""
    unique_wavelengths = np.unique(gp_wavelengths)
    number_gp = len(gp_times)
    obj_gps = []
    for wavelength in unique_wavelengths:
        gp_wavelengths = np.ones(number_gp) * wavelength
        pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
        pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
        obj_gp_pb_array = np.column_stack((gp_times, pb_pred, np.sqrt(pb_pred_var)))
        obj_gp_pb = Table(
            [
                obj_gp_pb_array[:, 0],
                obj_gp_pb_array[:, 1],
                obj_gp_pb_array[:, 2],
                [wavelength] * number_gp,
            ],
            names=["mjd", "flux", "flux_error", "filter"],
        )
        if len(obj_gps) == 0:
            obj_gps = obj_gp_pb
        else:
            obj_gps = vstack((obj_gps, obj_gp_pb))

    return obj_gps.to_pandas()

def generate_gp_single_event(df: pd.DataFrame, timesteps: int = 100, pb_wavelengths: Dict = LSST_PB_WAVELENGTHS) -> pd.DataFrame:
    """Generate GP interpolation for a single event."""
    filters = list(np.unique(df["filter"]))
    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}

    gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), timesteps)
    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)

    return obj_gps

def generate_gp_all_objects(object_list, obs_transient, timesteps=100, pb_wavelengths=LSST_PB_WAVELENGTHS):
    """Generate Gaussian Process interpolation for all objects.
    
    Parameters
    ----------
    object_list : List[str]
        List of object IDs
    obs_transient : pd.DataFrame
        DataFrame containing the trimmed transient data
    timesteps : int
        Number of time steps for GP interpolation
    pb_wavelengths : Dict
        Dictionary mapping filter names to wavelengths
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the interpolated light curves for all objects
    """
    # Initialize with all possible columns
    columns = ['mjd'] + list(pb_wavelengths.keys()) + ['object_id']
    adf = pd.DataFrame(columns=columns)
    
    for object_id in object_list:
        print(f"OBJECT ID:{object_id} at INDEX:{object_list.index(object_id)}")
        df = obs_transient[obs_transient['object_id'] == object_id]
        
        # Generate GP predictions for this object
        obj_gps = generate_gp_single_event(df, timesteps, pb_wavelengths)
        
        # Pivot to get one row per time step
        obj_gps = pd.pivot_table(obj_gps, index='mjd', columns='filter', values='flux')
        obj_gps = obj_gps.reset_index()
        
        # Ensure all filter columns exist
        for filter_name in pb_wavelengths.keys():
            if filter_name not in obj_gps.columns:
                obj_gps[filter_name] = np.nan
        
        # Add object_id
        obj_gps['object_id'] = object_id
        
        # Reorder columns to match the expected order
        obj_gps = obj_gps[columns]
        
        # Append to the main DataFrame
        adf = pd.concat([adf, obj_gps], ignore_index=True)
    
    return adf

def remap_filters(df: pd.DataFrame, filter_map: Dict) -> pd.DataFrame:
    """Remap integer filters to the corresponding filters."""
    df.rename({"passband": "filter"}, axis="columns", inplace=True)
    df["filter"].replace(to_replace=filter_map, inplace=True)
    return df

def robust_scale(dataframe: pd.DataFrame, scale_columns: List[Union[str, int]]) -> pd.DataFrame:
    """Standardize a dataset along axis=0 (rows)"""
    scaler = RobustScaler()
    scaler = scaler.fit(dataframe[scale_columns])
    dataframe.loc[:, scale_columns] = scaler.transform(dataframe[scale_columns].to_numpy())
    return dataframe

def z_score_normalize(dataframe: pd.DataFrame, scale_columns: List[Union[str, int]]) -> pd.DataFrame:
    """Apply z-score normalization to specified columns.
    
    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe containing the data to normalize
    scale_columns: List[Union[str, int]]
        Columns to apply z-score normalization to
        
    Returns
    -------
    pd.DataFrame
        Dataframe with normalized columns
    """
    scaler = StandardScaler()
    scaler = scaler.fit(dataframe[scale_columns])
    dataframe.loc[:, scale_columns] = scaler.transform(dataframe[scale_columns].to_numpy())
    return dataframe

def map_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Map original target codes to 0-14 indices.
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the 'target' column with original class codes
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added 'mapped_target' column and invalid classes removed
    """
    df["mapped_target"] = df["target"].map(PLASTICC_CLASS_MAPPING)
    df = df[df["mapped_target"].notna()]  # Drop invalid/unmapped classes
    return df

def create_dataset(features, labels, time_steps, step):
    """Create time series sequences with specified time steps and step size.
    
    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing the features
    labels : pd.Series
        Series containing the labels
    time_steps : int
        Number of time steps in each sequence
    step : int
        Step size between sequences
        
    Returns
    -------
    tuple
        (sequences, labels) where sequences is a numpy array of shape (n_sequences, time_steps, n_features)
        and labels is a numpy array of shape (n_sequences,)
    """
    sequences, sequence_labels = [], []
    
    for i in range(0, len(features) - time_steps + 1, step):
        sequences.append(features[i:i + time_steps].values)
        sequence_labels.append(labels.iloc[i])
    
    return np.array(sequences), np.array(sequence_labels)

def extract_redshift(df):
    """Extract photometric redshift for each object (static value)."""
    # Group by object_id and take the first occurrence (same for all rows)
    redshift_df = df.groupby("object_id")["hostgal_photoz"].first().reset_index()
    return redshift_df["hostgal_photoz"].values.reshape(-1, 1)



# Path configuration
DATA_PATH = '/kaggle/input/PLAsTiCC-2018/training_set.csv'
METADATA_PATH ='/kaggle/input/PLAsTiCC-2018/training_set_metadata.csv'
SAVE_DIR = "/kaggle/working/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load and preprocess data
print("Loading data...")
data = pd.read_csv(DATA_PATH, sep=',')

# Remap filters
print("Remapping filters...")
data = remap_filters(data, {0: 'lsstu', 1: 'lsstg', 2: 'lsstr', 
                           3: 'lssti', 4: 'lsstz', 5: 'lssty'})
data.rename({'flux_err': 'flux_error'}, axis='columns', inplace=True)

# Get unique filters and objects
filters = list(np.unique(data['filter']))
object_list = list(np.unique(data['object_id']))

# Trim to transient period
print("Trimming transients...")
obs_transient, object_list = transient_trim(object_list, data)

# Generate GP predictions
print("Generating GP predictions...")
generated_gp_dataset = generate_gp_all_objects(object_list, obs_transient, timesteps=100, pb_wavelengths=LSST_PB_WAVELENGTHS)
    
# Load and merge metadata
print("Merging metadata...")
metadata_pd = pd.read_csv(METADATA_PATH, sep=',', index_col='object_id')
metadata_pd = metadata_pd.reset_index()
metadata_pd['object_id'] = metadata_pd['object_id'].astype(np.int32)
metadata_pd['target'] = metadata_pd['target'].astype(np.int32)

# Merge and clean data
df_combi = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')
columns_to_drop = [
"ra",         # Right Ascension (sky coordinate; not predictive)
"decl",       # Declination (sky coordinate; not predictive)
"gal_l",      # Galactic longitude (redundant with ra/decl)
"gal_b",      # Galactic latitude (redundant with ra/decl)
"hostgal_specz",  # Spectroscopic redshift (not available in test set)
"distmod",    # Distance modulus (derived from hostgal_photoz; redundant)
"hostgal_photoz_err",  # Photoz error (optional: keep if modeling uncertainty)
] 
df = df_combi.drop(columns=columns_to_drop)
 
# Map classes
print("Mapping classes...")
df = map_classes(df)

# Split data
print("Splitting data...")
obj_ids = df['object_id'].unique()
train_ids, test_ids = train_test_split(obj_ids, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.176, random_state=42)  # 85-15 split

train_data = df[df['object_id'].isin(train_ids)]
val_data = df[df['object_id'].isin(val_ids)]
test_data = df[df['object_id'].isin(test_ids)]

# Apply z-score normalization to features
print("Scaling features...")
z_scaler = StandardScaler()
z_scaler.fit(train_data[filters])
train_data[filters] = z_scaler.transform(train_data[filters].copy())
val_data[filters] = z_scaler.transform(val_data[filters].copy())
test_data[filters] = z_scaler.transform(test_data[filters].copy())

# Create sequences
print("Creating sequences...")
TIME_STEPS = 20
STEP = 20

X_train, y_train = create_dataset(
    train_data[filters],
    train_data['mapped_target'],
    TIME_STEPS,
    STEP
)

X_val, y_val = create_dataset(
    val_data[filters],
    val_data['mapped_target'],
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    test_data[filters],
    test_data['mapped_target'],
    TIME_STEPS,
    STEP
)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# Save data
print("Saving processed data...")
# Save features
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "X_val.npy"), X_val)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)

# Save labels
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_val.npy"), y_val)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)


# Save class mapping and names
np.save(os.path.join(SAVE_DIR, "class_mapping.npy"), PLASTICC_CLASS_MAPPING)
np.save(os.path.join(SAVE_DIR, "class_names.npy"), CLASS_NAMES)

print(f"Processing complete! Final shapes:")
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Validation: {X_val.shape}, {y_val.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Print class distribution
print("\nClass distribution in training set:")
class_dist = train_data['mapped_target'].value_counts().sort_index()
for idx, count in class_dist.items():
    print(f"{CLASS_NAMES[idx]}: {count}")



import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
import platform
import numpy as np
import psutil
import io
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# Set up logging
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    white = "\x1b[37;20m"

    FORMAT = "[%(asctime)s] "
    FORMAT += "{%(filename)s:%(lineno)d} "
    FORMAT += "%(levelname)s "
    FORMAT += "- %(message)s"

    FORMATS = {
        logging.DEBUG: grey + FORMAT + reset,
        logging.INFO: white + FORMAT + reset,
        logging.WARNING: yellow + FORMAT + reset,
        logging.ERROR: red + FORMAT + reset,
        logging.CRITICAL: bold_red + FORMAT + reset,
    }

    def format(self, record):
        DATEFORMAT = "%y-%m-%d %H:%M:%S"
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=DATEFORMAT)
        return formatter.format(record)

def powerpuffgirls_logger(name="kaggle_notebook", log_dir="/kaggle/working/logs"):
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    log_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

log = powerpuffgirls_logger()

log.info("Logger initialized successfully!")
log.debug("This is a debug message for deeper inspection.")
log.warning("Watch out for potential issues.")

# Visualization utilities
plt.rc("font", size=20)
plt.rc("figure", figsize=(15, 3))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def check_and_clean_nan_data(X, y):
    """Check for NaN values in the data and remove corresponding samples if found."""
    # Check for NaN in X
    nan_mask_X = np.isnan(X).any(axis=(1, 2))
    # Check for NaN in y
    nan_mask_y = np.isnan(y).any(axis=1)
    
    # Combine masks
    nan_mask = nan_mask_X | nan_mask_y
    
    if np.any(nan_mask):
        num_nans = np.sum(nan_mask)
        total_samples = X.shape[0]
        log.warning(f"Found {num_nans} samples with NaN values out of {total_samples} ({(num_nans/total_samples)*100:.2f}%)")
        
        # Keep only non-NaN samples
        X_clean = X[~nan_mask]
        y_clean = y[~nan_mask]
        
        log.info(f"Data shape after removing NaN samples: X={X_clean.shape}, y={y_clean.shape}")
        return X_clean, y_clean
    
    log.info("No NaN values found in the data")
    return X, y

def find_optimal_batch_size(training_set_length: int) -> int:
    """Determine optimal batch size to use. Ideally leave a large remainder such that the GPU is
    full for most of the time.
    """
    if training_set_length < 10000:
        batch_size_list = [16, 32, 64]
    else:
        batch_size_list = [2048, 4096]
    ratios = []
    for batch_size in batch_size_list:
        remainder = training_set_length % batch_size
        if remainder == 0:
            batch_size = remainder
        else:
            ratios.append(batch_size / remainder)

    index, ratio = min(enumerate(ratios), key=lambda x: abs(x[1] - 1))
    return batch_size_list[index]

def lazy_load_plasticc_noZ(X, y):
    """Create a TensorFlow dataset from numpy arrays without redshift information."""
    def generator():
        for x, L in zip(X, y):
            yield (x, L)

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            tf.type_spec_from_value(X[0]),
            tf.type_spec_from_value(y[0]),
        ),
    )
    return dataset.repeat()

class WeightedLogLoss(tf.keras.losses.Loss):
    # initialize instance attributes
    def __init__(self, name="weighted_log_loss"):
        super().__init__(name=name)

    # compute loss
    def call(self, y_true, y_pred):
        # Cast inputs to float32 first
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate weights
        wtable = tf.reduce_sum(y_true, axis=0) / tf.cast(tf.shape(y_true)[0], tf.float32)
        
        # Clip predictions to avoid numerical instability
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        
        # Calculate loss using same dtype throughout
        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )
        
        return loss

class DistributedWeightedLogLoss(tf.keras.losses.Loss):
    # initialize instance attributes
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="weighted_log_loss",
    ):
        super().__init__(reduction=reduction, name=name)

    # compute loss
    def call(self, y_true, y_pred):
        # Cast inputs to float32 first
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate weights using TensorFlow ops for better distributed support
        wtable = tf.reduce_sum(y_true, axis=0) / tf.cast(tf.shape(y_true)[0], tf.float32)
        
        # Clip predictions to avoid numerical instability
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        
        # Calculate loss using same dtype throughout
        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )
        
        return loss

class SGEBreakoutCallback(tf.keras.callbacks.Callback):
    """Callback to stop training if job runs too long."""
    def __init__(self, threshold=24):
        super(SGEBreakoutCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        try:
            # Check if we're in an SGE environment
            if not os.environ.get('JOB_ID'):
                return

            hrs = subprocess.run(
                f"qstat -j {os.environ.get('JOB_ID')} | grep 'cpu' | awk '{{print $3}}' | awk -F ':' '{{print $1}}' | awk -F  '=' '{{print $2}}'",
                check=True,
                capture_output=True,
                shell=True,
                text=True,
            ).stdout.strip()

            if hrs and int(hrs) > self.threshold:
                log.info("Stopping training...")
                self.model.stop_training = True
        except Exception as e:
            log.warning(f"Error in SGEBreakoutCallback: {e}. Continuing training.")

class Training(object):
    def __init__(
        self,
        architecture,
        dataset,
        fink=None,
        avocado=None,
        testset=None,
        redshift=False,
    ):
        self.architecture = architecture
        self.dataset = dataset
        self.fink = fink
        self.avocado = avocado
        self.testset = testset
        self.redshift = redshift

    def __call__(self):
        """Train a given architecture with, or without redshift, on either UGRIZY or GR passbands"""
        def build_label():
            UNIXTIMESTAMP = int(time.time())
            try:
                VERSION = (
                    subprocess.check_output(["git", "describe", "--always"])
                    .strip()
                    .decode()
                )
            except Exception:
                VERSION = "unknown"
            JOB_ID = os.environ.get("JOB_ID")
            LABEL = f"{UNIXTIMESTAMP}-{JOB_ID}-{VERSION}"
            return LABEL

        LABEL = build_label()
        checkpoint_path = Path(self.architecture) / "models" / self.dataset / "checkpoints" / f"checkpoint-{LABEL}.keras"
        csv_logger_file = Path("logs") / self.architecture / f"training-{LABEL}.log"

        # Create necessary directories
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger_file.parent.mkdir(parents=True, exist_ok=True)

        # Also create the final model save directory
        model_save_dir = Path(self.architecture) / "models" / self.dataset
        model_save_dir.mkdir(parents=True, exist_ok=True)
        weights_save_dir = model_save_dir / "weights"
        weights_save_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load data
        data_dir = Path("/kaggle/input/processed")
        X_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
        y_train = np.load(data_dir / "y_train.npy", mmap_mode="r")

        X_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
        y_test = np.load(data_dir / "y_test.npy", mmap_mode="r")

        # Check and clean NaN values
        X_train, y_train = check_and_clean_nan_data(X_train, y_train)
        X_test, y_test = check_and_clean_nan_data(X_test, y_test)

        num_classes = y_train.shape[1]

        if self.fink is not None:
            # Take only G, R bands
            X_train = X_train[:, :, 0:3:2]
            X_test = X_test[:, :, 0:3:2]

        log.info(f"{X_train.shape, y_train.shape}")

        num_samples, timesteps, num_features = X_train.shape

        BATCH_SIZE = find_optimal_batch_size(num_samples)
        log.info(f"BATCH_SIZE:{BATCH_SIZE}")

        input_shape = (BATCH_SIZE, timesteps, num_features)
        log.info(f"input_shape:{input_shape}")

        drop_remainder = False

        def get_compiled_model_and_data(loss, drop_remainder):
            hyper_results_file = "/kaggle/input/hyperparameters/hyperparameter/results.json"
            
            train_ds = (
                lazy_load_plasticc_noZ(X_train, y_train)
                .shuffle(1000, seed=RANDOM_SEED)
                .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                .prefetch(tf.data.AUTOTUNE)
                .cache()
            )
            test_ds = (
                lazy_load_plasticc_noZ(X_test, y_test)
                .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                .prefetch(tf.data.AUTOTUNE)
                .cache()
            )

            # Load hyperparameters
            try:
                with open(hyper_results_file, "r") as f:
                    hyper_results = json.load(f)
                    if not isinstance(hyper_results, list):
                        raise ValueError("Hyperparameter results must be a list")
                    if not hyper_results:
                        raise ValueError("Hyperparameter results list is empty")
            except Exception as e:
                log.warning(f"Error loading hyperparameters: {e}. Using default values.")
                hyper_results = [{
                    "name": "default_config",
                    "value": 0.0,
                    "hyperparameters": {
                        "units": 128,
                        "dropout": 0.2,
                        "learning_rate": 0.001
                    }
                }]

            # Get best hyperparameters
            try:
                best_trial = min(hyper_results, key=lambda x: x["value"])
                hyperparameters = best_trial["hyperparameters"]
            except Exception as e:
                log.warning(f"Error finding best trial: {e}. Using first trial.")
                hyperparameters = hyper_results[0]["hyperparameters"]

            # Build model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(timesteps, num_features)),
                tf.keras.layers.LSTM(hyperparameters["units"], return_sequences=True),
                tf.keras.layers.Dropout(hyperparameters["dropout"]),
                tf.keras.layers.LSTM(hyperparameters["units"]),
                tf.keras.layers.Dropout(hyperparameters["dropout"]),
                tf.keras.layers.Dense(num_classes, activation="softmax")
            ])

            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"]),
                loss=loss,
                metrics=["accuracy"]
            )

            # Initialize event dictionary
            event = {
                "name": "training_run",
                "value": 0.0,
                "hyperparameters": hyperparameters
            }

            return model, train_ds, test_ds, event, hyper_results_file

        # Set up distributed training if multiple GPUs available
        if len(tf.config.list_physical_devices("GPU")) > 1:
            strategy = tf.distribute.MirroredStrategy()
            log.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
            BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
            VALIDATION_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

            with strategy.scope():
                loss = WeightedLogLoss()
                model, train_ds, test_ds, event, hyper_results_file = get_compiled_model_and_data(loss, drop_remainder)
        else:
            # For single GPU or CPU, use the same batch size for validation
            VALIDATION_BATCH_SIZE = BATCH_SIZE
            loss = WeightedLogLoss()
            model, train_ds, test_ds, event, hyper_results_file = get_compiled_model_and_data(loss, drop_remainder)

        # Set up callbacks
        callbacks = [
            CSVLogger(csv_logger_file),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            SGEBreakoutCallback()
        ]

        # Calculate steps per epoch
        steps_per_epoch = 1100
        validation_steps = 350
        
        log.info(f"Steps per epoch: {steps_per_epoch}")
        log.info(f"Validation steps: {validation_steps}")
        log.info(f"Batch size: {BATCH_SIZE}")
        log.info(f"Validation batch size: {VALIDATION_BATCH_SIZE}")

        # Train model from scratch
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=10,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        log.info(f"PERCENT OF RAM USED: {psutil.virtual_memory().percent}")
        log.info(f"RAM USED: {psutil.virtual_memory().active / (1024*1024*1024)}")


        # Save model
        LABEL = "GR-" + LABEL if self.fink else "UGRIZY-" + LABEL
        # LABEL += f"-LL{WLOSS:.3f}"

        if platform.system() != "Darwin":
            # Create directories if they don't exist
            model_dir = Path(self.architecture) / "models" / self.dataset
            weights_dir = model_dir / "weights"
            model_dir.mkdir(parents=True, exist_ok=True)
            weights_dir.mkdir(parents=True, exist_ok=True)

            # Save model and weights with proper extensions
            model_path = model_dir / f"model-{LABEL}.keras"
            weights_path = weights_dir / f"weights-{LABEL}.weights.h5"
            
            log.info(f"Saving model to: {model_path}")
            log.info(f"Saving weights to: {weights_path}")
            
            model.save(str(model_path))
            model.save_weights(str(weights_path))

        return event


if __name__ == "__main__":
    import sys
    sys.argv = ['script.py', '--architecture', 't2', '--dataset', 'plasticc', '--fink']
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="t2", required=True)
    parser.add_argument("--dataset", type=str,default= "plasticc", required=True)
    parser.add_argument("--fink", action="store_true")
    parser.add_argument("--avocado", action="store_true")
    parser.add_argument("--testset", action="store_true")
    args = parser.parse_args()

    training = Training(
    architecture=args.architecture,
    dataset=args.dataset,
    fink=args.fink,
    avocado=args.avocado,
    testset=args.testset,
)
    training()



def handle_extra_class_and_normalize(y_pred_proba, le, test_features_df):
    """Handle the extra class (class 99) and create normalized probabilities for all 15 classes"""
    print("Processing predictions for all 15 classes...")
    
    all_classes = CLASSES  # [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
    trained_classes = list(le.classes_)
    
    results = pd.DataFrame()
    results['object_id'] = test_features_df['object_id']

    for class_id in all_classes:
        if class_id in trained_classes:
            class_idx = trained_classes.index(class_id)
            results[str(class_id)] = y_pred_proba[:, class_idx]
        else:
            # Initialize class 99 with zero for now
            results[str(class_id)] = 0.0

    if 99 not in trained_classes:
        # Assign class 99 as 1 - max(predicted probability across other classes)
        results['99'] = 1.0 - results[[str(c) for c in trained_classes]].max(axis=1)
        results['99'] = results['99'].clip(lower=0)  # Ensure no negatives

    # Normalize the probabilities so each row sums to 1
    prob_cols = [str(c) for c in all_classes]
    results[prob_cols] = results[prob_cols].div(results[prob_cols].sum(axis=1), axis=0)

    print("Final prediction probabilities normalized for all classes.")
    return results



X_test = np.load('/kaggle/input/processed/X_test.npy')
y_test = np.load('/kaggle/input/processed/y_test.npy')

print("\nOriginal data shapes:")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

n_samples = y_test.shape[0]
X_test = X_test[:n_samples]

print("\nAdjusted data shapes:")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


print("\nChecking for NaN values in test data...")
X_test, y_test = check_and_clean_nan_data(X_test, y_test)


custom_objects = {'WeightedLogLoss': WeightedLogLoss}
model = tf.keras.models.load_model(
    '/kaggle/input/hyperparameters/model-UGRIZY-1748131995-None-unknown.keras',
    custom_objects=custom_objects
)

print("\nModel Configuration:")
model.summary()

print("\nMaking predictions...")
y_pred = model.predict(X_test, verbose=1)

y_pred_normalized = y_pred / y_pred.sum(axis=1, keepdims=True)


print("\nCalculating metrics...")
wloss = WeightedLogLoss()
loss_value = wloss(y_test, y_pred_normalized).numpy()
print(f"\nWeighted Log Loss: {loss_value:.3f}")

y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred_normalized, axis=1)

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.title(f'Confusion Matrix (Normalized)\nWeighted Log Loss = {loss_value:.3f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("\nConfusion Matrix saved as 'confusion_matrix.png'")


TEST_DATA_PATH = '/kaggle/input/PLAsTiCC-2018/test_set_batch1.csv'
TEST_METADATA_PATH = '/kaggle/input/PLAsTiCC-2018/test_set_metadata.csv'
SAVE_DIR = "/kaggle/working/processed_test"
TRAIN_DATA_PATH = "/kaggle/input/xtrain/X_train.npy"

os.makedirs(SAVE_DIR, exist_ok=True)

# Load test data
print("Loading test data...")
data = pd.read_csv(TEST_DATA_PATH, sep=',')

# Remap filters
print("Remapping filters...")
data = remap_filters(data, {0: 'lsstu', 1: 'lsstg', 2: 'lsstr', 
                           3: 'lssti', 4: 'lsstz', 5: 'lssty'})
data.rename({'flux_err': 'flux_error'}, axis='columns', inplace=True)

# Get unique filters and objects
filters = list(np.unique(data['filter']))
object_list = list(np.unique(data['object_id']))

# Trim to transient period
print("Trimming transients...")
obs_transient, object_list = transient_trim(object_list, data)

# Generate GP predictions
print("Generating GP predictions...")
generated_gp_dataset = generate_gp_all_objects(object_list, obs_transient, timesteps=100, pb_wavelengths=LSST_PB_WAVELENGTHS)

# Load and merge metadata
print("Merging metadata...")
metadata_pd = pd.read_csv(TEST_METADATA_PATH, sep=',', index_col='object_id')
metadata_pd = metadata_pd.reset_index()
metadata_pd['object_id'] = metadata_pd['object_id'].astype(np.int32)

# Merge and clean data
df_combi = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')
columns_to_drop = [
    "ra",         # Right Ascension (sky coordinate; not predictive)
    "decl",       # Declination (sky coordinate; not predictive)
    "gal_l",      # Galactic longitude (redundant with ra/decl)
    "gal_b",      # Galactic latitude (redundant with ra/decl)
    "hostgal_specz",  # Spectroscopic redshift (not available in test set)
    "distmod",    # Distance modulus (derived from hostgal_photoz; redundant)
    "hostgal_photoz_err",  # Photoz error 
]
df = df_combi.drop(columns=columns_to_drop)

# Apply z-score normalization to features using training data statistics
print("Scaling features using training data statistics...")
if not os.path.exists(TRAIN_DATA_PATH):
    raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_PATH}. Please run training data preprocessing first.")

# Load training data to fit scaler
X_train = np.load(TRAIN_DATA_PATH)
# Reshape training data to 2D for fitting scaler
n_samples, n_timesteps, n_features = X_train.shape
X_train_2d = X_train.reshape(-1, n_features)

# Fit scaler on training data
z_scaler = StandardScaler()
z_scaler.fit(X_train_2d)

# Transform test data
test_data_2d = df[filters].values
df[filters] = z_scaler.transform(test_data_2d)

np.save(os.path.join(SAVE_DIR, "z_scaler.npy"), z_scaler)

# Create sequences
print("Creating sequences...")
STEP = 20  # Using same step size as training data

X_test, test_object_ids = create_sequences(df[filters + ['object_id']], TIME_STEPS, STEP)

# Save data
print("Saving processed test data...")
np.save(os.path.join(SAVE_DIR, "X_test1.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "test_object_ids.npy"), test_object_ids)

print(f"Test data processing complete! Final shape: {X_test.shape}")



# Load test data
X_test = np.load('/kaggle/input/processed/X_test1.npy')
test_features_df = pd.read_csv('/kaggle/input/processed/test_object_ids.npy')  # contains 'object_id'

print(f"X_test shape: {X_test.shape}")

# Remove rows with NaNs if any
if np.isnan(X_test).any():
    nan_mask = np.isnan(X_test).any(axis=1)
    X_test = X_test[~nan_mask]
    test_features_df = test_features_df.loc[~nan_mask].reset_index(drop=True)
    print(f"Removed {nan_mask.sum()} rows with NaNs. New shape: {X_test.shape}")

# Load the model
model = tf.keras.models.load_model('/kaggle/input/hyperparameters/model-UGRIZY-1748131995-None-unknown.keras')

# Make predictions (raw probabilities)
y_pred = model.predict(X_test, verbose=1)

# Load your label encoder
le = LabelEncoder()
le.classes_ = np.load('/kaggle/input/labelencoder/classes.npy', allow_pickle=True)  # adjust path as needed

# Your global classes list (must be defined)
CLASSES = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]

results_df = handle_extra_class_and_normalize(y_pred, le, test_features_df)

# Save results
results_df.to_csv('predictions_1.csv', index=False)
print("\nSaved predictions")
print(results_df.head())



def get_id(line):
    return line.strip().split(',')[0]


def split_time_series_file_into_three(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = []
    if lines[0].lower().startswith("object_id") or ',' in lines[0]:
        header = [lines[0]]
        lines = lines[1:]

    # Step 1: Find indices where a new ID starts
    id_indices = [0]
    for i in range(1, len(lines)):
        if get_id(lines[i]) != get_id(lines[i - 1]):
            id_indices.append(i)

    # Step 2: Split ID blocks into 3 parts
    total_ids = len(id_indices)
    cut1 = total_ids // 3
    cut2 = (2 * total_ids) // 3

    idx1 = id_indices[cut1]
    idx2 = id_indices[cut2] if cut2 < len(lines) else len(lines)

    # Step 3: Create parts
    part1 = header + lines[:idx1]
    part2 = header + lines[idx1:idx2]
    part3 = header + lines[idx2:]

    return part1, part2, part3


# Example usage:
file_path = '/kaggle/input/PLAsTiCC-2018/test_set_batch4.csv'
part1, part2, part3 = split_time_series_file_into_three(file_path)

# Save results
with open('/kaggle/working/batch4_1.csv', 'w') as f:
    f.writelines(part1)

with open('/kaggle/working/batch4_2.csv', 'w') as f:
    f.writelines(part2)

with open('/kaggle/working/batch4_3.csv', 'w') as f:
    f.writelines(part3)

print(f"Split into 3 parts on ID boundaries. Part1: {len(part1)} lines, Part2: {len(part2)} lines, Part3: {len(part3)} lines.")

