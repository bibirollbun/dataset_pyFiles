import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import os
from glob import glob
from collections import defaultdict
import gc
import time # Import time module for timing

# --- Configuration and Paths ---
DATA_DIR = '/kaggle/input/ariel-data-challenge-2025'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# --- Load Base Data ---
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
wavelengths_df = pd.read_csv(os.path.join(DATA_DIR, 'wavelengths.csv'))
train_star_info = pd.read_csv(os.path.join(DATA_DIR, 'train_star_info.csv'))
train_adc_info = pd.read_csv(os.path.join(DATA_DIR, 'adc_info.csv'))
train_labels = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='planet_id')

# Extract target column names (flux_0 to flux_355)
target_cols = [col for col in train_df.columns if col != 'planet_id']
NUM_WAVELENGTHS = len(target_cols) # This should be 356

# --- Define FGS1 and AIRS-Ch0 channel indices and weights ---
FGS1_CHANNEL_IDX = 0 # Corresponds to flux_0
AIRS_CH0_CHANNEL_INDICES = list(range(1, NUM_WAVELENGTHS)) # Corresponds to flux_1 to flux_355

# Weights based on competition description
WEIGHT_FGS1 = 2 * (0.2 / 1) # 0.4
WEIGHT_AIRS_CH0_PER_POINT = 1.95 / 282 # ~0.0069

# Ideal uncertainty values (from competition description)
IDEAL_UNC_FGS1_PPM = 1 # 1 PPM
IDEAL_UNC_AIRS_PPM = 10 # 10 PPM

print("Setup Complete. Data Loaded.")

# --- Custom GLL Metric Function (DEFINITIVELY CORRECTED) ---
def calculate_gll(y_true, mu_pred, sigma_pred, wavelengths_df_unused): # Renamed parameter for clarity
    """
    Calculates the Gaussian Log-likelihood (GLL) score for the Ariel Data Challenge.
    (MODIFIED to infer instrument from channel index, NOT wavelengths_df_for_gll)

    Parameters:
    y_true (pd.DataFrame or np.array): Ground truth flux values (shape: num_planets, num_wavelengths).
    mu_pred (pd.DataFrame or np.array): Predicted mean flux values (shape: num_planets, num_wavelengths).
    sigma_pred (pd.DataFrame or np.array): Predicted uncertainty values (shape: num_planets, num_wavelengths).
    wavelengths_df_unused (pd.DataFrame): The wavelengths.csv dataframe (this parameter is now truly UNUSED
                                          for instrument lookup to avoid KeyError).

    Returns:
    float: The total GLL value (L).
    """
    y_true = np.asarray(y_true)
    mu_pred = np.asarray(mu_pred)
    sigma_pred = np.asarray(sigma_pred)

    # Ensure all inputs have the same shape
    if not (y_true.shape == mu_pred.shape == sigma_pred.shape):
        raise ValueError("y_true, mu_pred, and sigma_pred must have the same shape.")

    # Ensure sigma_pred is strictly positive
    sigma_pred = np.maximum(sigma_pred, 1e-9) # Prevent log(0) or division by zero

    # Calculate log-likelihood for each point
    log_likelihood_per_point = -0.5 * np.log(2 * np.pi * sigma_pred**2) - 0.5 * ((y_true - mu_pred) / sigma_pred)**2

    total_gll = 0.0
    
    # Apply weights based on instrument, now determined by the column index (i)
    # The problematic line 'instrument = wavelengths_df_for_gll.loc[i, 'instrument']' IS REMOVED HERE.
    for i in range(y_true.shape[1]): # Iterate through each column (wavelength)
        
        # Determine instrument based on pre-defined global channel indices
        if i == FGS1_CHANNEL_IDX: # Check if it's the FGS1 channel (flux_0)
            weight = WEIGHT_FGS1
        elif i in AIRS_CH0_CHANNEL_INDICES: # Check if it's one of the AIRS-CH0 channels (flux_1 to flux_355)
            weight = WEIGHT_AIRS_CH0_PER_POINT
        else:
            # This case should ideally not be reached if NUM_WAVELENGTHS matches expected channels
            print(f"Warning: Unexpected wavelength index {i} found. Assigning weight 0.")
            weight = 0 

        # Sum likelihoods for this wavelength across all planets, then apply weight
        total_gll += np.sum(log_likelihood_per_point[:, i]) * weight

    return total_gll

# --- Function to calculate the final competition score ---
def calculate_competition_score(L_user, L_ideal, L_ref):
    """
    Calculates the final competition score.

    Parameters:
    L_user (float): GLL value from user's submission.
    L_ideal (float): GLL value for the ideal case.
    L_ref (float): GLL value for the reference case.

    Returns:
    float: The competition score, clipped to [0, 1].
    """
    score = (L_user - L_ref) / (L_ideal - L_ref)
    return np.clip(score, 0, 1)

print("Custom GLL Metric function defined.")

# --- Feature Engineering Functions (now defined within the main script) ---

def f_read_and_preprocess(dataset_path, planet_ids, band_name="FGS1"):
    """Read FGS1 files and extract comprehensive time series features."""
    extracted_features = []
    for planet_id in tqdm(planet_ids, desc=f"Processing {band_name} signals"):
        try:
            signal_files = glob(os.path.join(dataset_path, str(planet_id), f"{band_name}_signal_*.parquet"))
            
            if not signal_files:
                features_dict = {f'{band_name}_mean': np.nan, f'{band_name}_std': np.nan,
                                 f'{band_name}_skew': np.nan, f'{band_name}_kurt': np.nan,
                                 f'{band_name}_min': np.nan, f'{band_name}_max': np.nan}
                extracted_features.append({'planet_id': planet_id, **features_dict})
                continue
            
            f_signal = pl.read_parquet(signal_files[0])
            
            pixel_count = 32 * 32
                
            mean_signal = f_signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy().flatten() / pixel_count
            net_signal = mean_signal[1::2] - mean_signal[0::2]

            features_dict = {
                f'{band_name}_mean': np.mean(net_signal),
                f'{band_name}_std': np.std(net_signal),
                f'{band_name}_skew': scipy.stats.skew(net_signal),
                f'{band_name}_kurt': scipy.stats.kurtosis(net_signal),
                f'{band_name}_min': np.min(net_signal),
                f'{band_name}_max': np.max(net_signal),
            }
            extracted_features.append({'planet_id': planet_id, **features_dict})
        except Exception as e:
            print(f"Error processing planet_id {planet_id} for {band_name}: {e}")
            features_dict = {f'{band_name}_mean': np.nan, f'{band_name}_std': np.nan,
                             f'{band_name}_skew': np.nan, f'{band_name}_kurt': np.nan,
                             f'{band_name}_min': np.nan, f'{band_name}_max': np.nan}
            extracted_features.append({'planet_id': planet_id, **features_dict})
    return pd.DataFrame(extracted_features).set_index('planet_id')

def a_read_and_preprocess(dataset_path, planet_ids, band_name="AIRS-CH0"):
    """Read AIRS-CH0 files and extract comprehensive time series features."""
    extracted_features = []
    for planet_id in tqdm(planet_ids, desc=f"Processing {band_name} signals"):
        try:
            signal_files = glob(os.path.join(dataset_path, str(planet_id), f"{band_name}_signal_*.parquet"))
            
            if not signal_files:
                features_dict = {f'{band_name}_mean': np.nan, f'{band_name}_std': np.nan,
                                 f'{band_name}_skew': np.nan, f'{band_name}_kurt': np.nan,
                                 f'{band_name}_min': np.nan, f'{band_name}_max': np.nan}
                extracted_features.append({'planet_id': planet_id, **features_dict})
                continue

            a_signal = pl.read_parquet(signal_files[0])
            pixel_count = 32 * 356
            mean_signal = a_signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy().flatten() / pixel_count
            net_signal = mean_signal[1::2] - mean_signal[0::2]
            
            features_dict = {
                f'{band_name}_mean': np.mean(net_signal),
                f'{band_name}_std': np.std(net_signal),
                f'{band_name}_skew': scipy.stats.skew(net_signal),
                f'{band_name}_kurt': scipy.stats.kurtosis(net_signal),
                f'{band_name}_min': np.min(net_signal),
                f'{band_name}_max': np.max(net_signal),
            }
            extracted_features.append({'planet_id': planet_id, **features_dict})
        except Exception as e:
            print(f"Error processing planet_id {planet_id} for {band_name}: {e}")
            features_dict = {f'{band_name}_mean': np.nan, f'{band_name}_std': np.nan,
                             f'{band_name}_skew': np.nan, f'{band_name}_kurt': np.nan,
                             f'{band_name}_min': np.nan, f'{band_name}_max': np.nan}
            extracted_features.append({'planet_id': planet_id, **features_dict})
            
    return pd.DataFrame(extracted_features).set_index('planet_id')


# --- 1. Visualization & Exploratory Data Analysis (EDA) ---
print("\n--- 1. Visualization & Exploratory Data Analysis (EDA) ---")

# --- Basic Train Set Stats ---
print("ğŸª� Number of training planets:", train_df.shape[0])
print("ğŸ“ˆ Number of target labels (wavelengths):", train_df.shape[1] - 1)
print("ğŸ”¬ Length of wavelength grid:", wavelengths_df.shape[0])

# --- Target Stats (per flux column) ---
flux_summary = train_df[target_cols].agg(['min', 'max', 'mean', 'std']).T
print("\nğŸ“Š Flux value summary (first 5 rows):")
print(flux_summary.head())

# --- Unique Stars ---
if 'planet_id' in train_star_info.columns:
    num_stars = train_star_info.drop(columns='planet_id').drop_duplicates().shape[0]
else:
    num_stars = train_star_info.drop_duplicates().shape[0]
print("\nğŸŒŸ Number of unique stars in training:", num_stars)

# --- Planets with Multiple Observations ---
obs_counts = defaultdict(int)
train_planets_dirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
for pid_str in train_planets_dirs:
    air_obs = glob(os.path.join(TRAIN_DIR, pid_str, "AIRS-CH0_signal_*.parquet"))
    obs_counts[pid_str] = len(air_obs)
multi_obs = {pid: count for pid, count in obs_counts.items() if count > 1}
print("\nğŸ”� Planets with multiple observations:", len(multi_obs))

# --- Check Calibration File Coverage ---
missing_calibs = []
expected_calibs = {"dark", "dead", "flat", "linear_corr", "read"}
for pid_str in train_planets_dirs:
    for band in ["AIRS-CH0", "FGS1"]:
        calib_path = os.path.join(TRAIN_DIR, pid_str, f"{band}_calibration")
        calib_files = set()
        if os.path.exists(calib_path):
            calib_files = {os.path.splitext(f)[0] for f in os.listdir(calib_path)}
        missing = expected_calibs - calib_files
        if missing:
            missing_calibs.append((pid_str, band, missing))
print("\nğŸ§ª Planets missing calibration files:", len(missing_calibs))
if missing_calibs:
    print("   Example:", missing_calibs[0])

# --- Distribution of Observations Per Planet ---
obs_distribution = pd.Series(list(obs_counts.values())).value_counts().sort_index()
print("\nğŸ—‚ Observation count distribution per planet (AIR-CH0):")
print(obs_distribution)

# --- Planet-Star Uniqueness Check ---
merged_star_info = pd.merge(train_df[['planet_id']], train_star_info, on='planet_id', how='left')
unique_links = merged_star_info[['planet_id'] + [col for col in train_star_info.columns if col != 'planet_id']].drop_duplicates()
print("\nğŸ”— Unique planet-star mappings:", unique_links.shape[0])

# --- Visualizing FGS1 Images ---
print("\n--- Visualizing FGS1 Images ---")
planet_id_fgs = 1010375142 # Example ID
try:
    f_signal_ex_pl = pl.read_parquet(os.path.join(TRAIN_DIR, str(planet_id_fgs), 'FGS1_signal_0.parquet'))
    # Convert Polars DataFrame to Pandas DataFrame for .iloc access
    f_signal_ex = f_signal_ex_pl.to_pandas()

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(f_signal_ex.iloc[0].values.reshape(32, 32), ax=ax1, vmin=0, vmax=52000, cmap='viridis')
    ax1.set_aspect('equal')
    ax1.set_title(f'FGS1 Image Frame 0 for Planet {planet_id_fgs}')
    sns.heatmap(f_signal_ex.iloc[1].values.reshape(32, 32), ax=ax2, vmin=0, vmax=52000, cmap='viridis')
    ax2.set_aspect('equal')
    ax2.set_title(f'FGS1 Image Frame 1 for Planet {planet_id_fgs}')
    plt.suptitle('A pair of FGS1 Images')
    plt.show()
except Exception as e:
    print(f"Could not load FGS1 example for visualization: {e}")


# --- Visualizing FGS1 Time Series ---
print("\n--- Visualizing FGS1 Time Series ---")
planet_id_strong_signal = 1048114509 # Planet with strong signal
planet_id_weak_signal = 1240764363 # Planet with weak signal

_, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(14, 8))

# Strong signal planet
try:
    f_signal_strong_pl = pl.read_parquet(os.path.join(TRAIN_DIR, str(planet_id_strong_signal), 'FGS1_signal_0.parquet'))
    mean_signal_strong = f_signal_strong_pl.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy().flatten() / (32*32)
    net_signal_strong = mean_signal_strong[1::2] - mean_signal_strong[0::2]
    cum_signal_strong = net_signal_strong.cumsum()
    window=800 # Define window for smoothing
    smooth_signal_strong = (cum_signal_strong[window:] - cum_signal_strong[:-window]) / window

    ax1.set_title(f'FGS1: Raw Signal (Planet {planet_id_strong_signal})')
    ax1.plot(net_signal_strong, label='raw signal', alpha=0.7)
    ax1.legend()
    ax3.set_title(f'FGS1: Smoothed Signal (Planet {planet_id_strong_signal})')
    ax3.plot(smooth_signal_strong, color='c', label='smoothened signal')
    ax3.legend()
    ax3.set_xlabel('time step')
    for time_step in [20500, 23500, 44000, 47000]: # Example transit timings
        ax3.axvline(time_step, color='gray', linestyle='--', alpha=0.6)
except Exception as e:
    print(f"Could not load FGS1 strong signal example for visualization: {e}")

# Weak signal planet
try:
    f_signal_weak_pl = pl.read_parquet(os.path.join(TRAIN_DIR, str(planet_id_weak_signal), 'FGS1_signal_0.parquet'))
    mean_signal_weak = f_signal_weak_pl.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy().flatten() / (32*32)
    net_signal_weak = mean_signal_weak[1::2] - mean_signal_weak[0::2]
    cum_signal_weak = net_signal_weak.cumsum()
    window=800 # Define window for smoothing
    smooth_signal_weak = (cum_signal_weak[window:] - cum_signal_weak[:-window]) / window

    ax2.set_title(f'FGS1: Raw Signal (Planet {planet_id_weak_signal})')
    ax2.plot(net_signal_weak, label='raw signal', alpha=0.7)
    ax2.legend()
    ax4.set_title(f'FGS1: Smoothed Signal (Planet {planet_id_weak_signal})')
    ax4.plot(smooth_signal_weak, color='c', label='smoothened signal')
    ax4.legend()
    ax4.set_xlabel('time step')
    for time_step in [20500, 23500, 44000, 47000]: # Example transit timings
        ax4.axvline(time_step, color='gray', linestyle='--', alpha=0.6)
except Exception as e:
    print(f"Could not load FGS1 weak signal example for visualization: {e}")

plt.suptitle('FGS1 Time Series Analysis', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


# --- Visualizing AIRS-CH0 Data ---
print("\n--- Visualizing AIRS-CH0 Data ---")
planet_id_airs = 1240764363 # Example ID
try:
    a_signal_ex_pl = pl.read_parquet(os.path.join(TRAIN_DIR, str(planet_id_airs), 'AIRS-CH0_signal_0.parquet'))
    a_signal_np = a_signal_ex_pl.to_numpy().reshape(a_signal_ex_pl.shape[0], 32, 356)

    plt.figure(figsize=(12, 6))
    sns.heatmap(a_signal_np[100], cmap='viridis') # Displaying an arbitrary time slice (e.g., 100th frame)
    plt.ylabel('Spatial Dimension (Pixel Row)')
    plt.xlabel('Wavelength Dimension (Pixel Column)')
    plt.title(f'AIRS-CH0 Single Frame (Time Step 100) for Planet {planet_id_airs}')
    plt.show()

    mean_signal_airs = a_signal_ex_pl.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy().flatten() / (32*356)
    net_signal_airs = mean_signal_airs[1::2] - mean_signal_airs[0::2]
    cum_signal_airs = net_signal_airs.cumsum()
    window_airs=80 # Smaller window for AIRS due to fewer total steps
    smooth_signal_airs = (cum_signal_airs[window_airs:] - cum_signal_airs[:-window_airs]) / window_airs

    _, (ax1_airs, ax2_airs) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax1_airs.plot(net_signal_airs, label='raw net signal', alpha=0.7)
    ax1_airs.legend()
    ax1_airs.set_title(f'AIRS-CH0: Raw Net Signal (Planet {planet_id_airs})')
    ax2_airs.plot(smooth_signal_airs, color='c', label='smoothened net signal')
    ax2_airs.legend()
    ax2_airs.set_xlabel('Time Step (Paired Frames)')
    ax2_airs.set_title(f'AIRS-CH0: Smoothed Net Signal (Planet {planet_id_airs})')

    fgs1_total_frames_example = 135000
    airs0_total_frames_example = 11250
    scaling_factor = (airs0_total_frames_example / 2) / (fgs1_total_frames_example / 2) # Ratio of net steps
    for time_step in [20500, 23500, 44000, 47000]:
        ax2_airs.axvline(time_step * scaling_factor, color='gray', linestyle='--', alpha=0.6)
    plt.suptitle('AIRS-CH0 Time Series Analysis', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
except Exception as e:
    print(f"Could not load AIRS-CH0 example for visualization: {e}")

print("\nEDA and Visualization Complete.")

# --- 2. Feature Engineering & Preprocessing ---
print("\n--- 2. Feature Engineering & Preprocessing ---")

print("\n--- Running FGS1 Preprocessing for Training Data ---")
start_time = time.time()
f_train_features = f_read_and_preprocess(TRAIN_DIR, train_labels.index.tolist(), band_name="FGS1")
end_time = time.time()
print(f"FGS1 Preprocessing took: {end_time - start_time:.2f} seconds")

print("\n--- Running AIRS-CH0 Preprocessing for Training Data ---")
start_time = time.time()
a_train_features = a_read_and_preprocess(TRAIN_DIR, train_labels.index.tolist(), band_name="AIRS-CH0")
end_time = time.time()
print(f"AIRS-CH0 Preprocessing took: {end_time - start_time:.2f} seconds")

# Merge features and target labels
X_train = f_train_features.merge(a_train_features, left_index=True, right_index=True, how='left')
y_train = train_labels.copy()

# --- Additional Feature Engineering (from star_info and adc_info) ---
# Merge train_star_info using 'planet_id' (this is correct as star_info has planet_id)
X_train = X_train.merge(train_star_info.set_index('planet_id'), left_index=True, right_index=True, how='left')

# CORRECT WAY TO ADD ADC_INFO FEATURES:
# Since train_adc_info contains global ADC settings and NO 'planet_id' column,
# we add them as new columns to every row of X_train by broadcasting.
if not train_adc_info.empty:
    # Get the values from the first (and likely only) row of train_adc_info
    # This assumes adc_info has only one row of global settings.
    adc_features_values = train_adc_info.iloc[0].to_dict()

    # Add these values as new columns to X_train
    for col_name, value in adc_features_values.items():
        X_train[col_name] = value
    print("ADC info features added to X_train.")
else:
    print("Warning: train_adc_info is empty. ADC features will not be added to X_train.")


# Handle missing values (this should be done AFTER all merges/additions)
# This will fill NaNs that might arise from missing signal files or star_info.
X_train = X_train.fillna(0)

print("\n--- Processed Training Features (X_train) Head ---")
print(X_train.head())
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print("\nFeature Engineering and Preprocessing Complete.")

# --- 3. Model Training & Fitting ---
print("\n--- 3. Model Training & Fitting ---")

# Define models for MEAN prediction
ridge_mean_model = Ridge(random_state=42)
cat_mean_model = CatBoostRegressor(
    iterations=500, learning_rate=0.05, depth=6,
    loss_function='RMSE', eval_metric='RMSE',
    random_seed=42, verbose=0, thread_count=-1
)

# Define models for LOG-UNCERTAINTY prediction
ridge_log_sigma_model = Ridge(random_state=42)

# Pre-allocate OOF arrays for mean and sigma
mu_oof_preds = np.zeros(y_train.shape)
sigma_oof_preds = np.zeros(y_train.shape)

# Cross-validation setup
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print(f"\nTraining {N_SPLITS}-Fold Cross-Validation Models (Mean & Log-Sigma)...")

categorical_features_indices = np.where(X_train.dtypes == 'object')[0].tolist()
for col_idx in categorical_features_indices:
    col_name = X_train.columns[col_idx]
    X_train[col_name] = X_train[col_name].astype(str) # Ensure string type for CatBoost

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    for i, col in tqdm(enumerate(target_cols), total=len(target_cols), desc=f"Training for targets in Fold {fold+1}"):
        # --- Train Mean Models (mu_pred) ---
        ridge_mean_model.fit(X_train_fold, y_train_fold[col])
        
        cat_mean_model.fit(X_train_fold, y_train_fold[col], cat_features=categorical_features_indices)
        
        # Ensemble mean predictions for this fold
        fold_mu_pred = (ridge_mean_model.predict(X_val_fold) + cat_mean_model.predict(X_val_fold)) / 2
        fold_mu_pred[fold_mu_pred < 0] = 0 # Ensure non-negative flux

        mu_oof_preds[val_idx, i] = fold_mu_pred
        
        # --- Train Log-Sigma Models (sigma_pred) ---
        cat_mean_preds_on_train = cat_mean_model.predict(X_train_fold)
        residuals_sq = (y_train_fold[col].values - cat_mean_preds_on_train)**2
        
        target_log_var = np.log(residuals_sq + 1e-6) # Add a small epsilon to avoid log(0)

        ridge_log_sigma_model.fit(X_train_fold, target_log_var)
        
        log_var_val_pred = ridge_log_sigma_model.predict(X_val_fold)
        sigma_val_pred = np.sqrt(np.exp(log_var_val_pred))
        
        min_sigma_clamp = np.mean(y_train_fold[col].values) * (min(IDEAL_UNC_FGS1_PPM, IDEAL_UNC_AIRS_PPM) / 1e6)
        sigma_val_pred = np.maximum(sigma_val_pred, min_sigma_clamp)

        sigma_oof_preds[val_idx, i] = sigma_val_pred

    gc.collect()

print("\nModel Training & Fitting Complete.")

# --- 4. Computing GLL Score ---
print("\n--- 4. Computing GLL Score ---")

# Ensure predictions are in the correct format for GLL calculation
y_true_oof = y_train.values
mu_oof = mu_oof_preds
sigma_oof = sigma_oof_preds

# Calculate User's GLL (L_user)
# The calculate_gll function itself should now be using the channel index (i) for instrument type.
L_user = calculate_gll(y_true_oof, mu_oof, sigma_oof, wavelengths_df)
print(f"User's OOF GLL (L_user): {L_user:.6f}")

# --- Calculate L_ideal ---
y_ideal_mu = y_true_oof.copy() # Mu perfectly matches y_true

sigma_ideal = np.zeros_like(y_true_oof)
for i in range(NUM_WAVELENGTHS):
    # This is the line that needed modification:
    # Instead of looking up 'instrument' in wavelengths_df,
    # we determine it based on the channel index 'i' (as per problem description).
    if i == FGS1_CHANNEL_IDX: # Check if it's the FGS1 channel (flux_0)
        sigma_ideal[:, i] = y_ideal_mu[:, i] * (IDEAL_UNC_FGS1_PPM / 1e6)
    elif i in AIRS_CH0_CHANNEL_INDICES: # Check if it's one of the AIRS-CH0 channels (flux_1 to flux_355)
        sigma_ideal[:, i] = y_ideal_mu[:, i] * (IDEAL_UNC_AIRS_PPM / 1e6)
    # No 'else' needed here, as all columns should fall into one of these categories
    # if NUM_WAVELENGTHS is correctly defined as 356.

L_ideal = calculate_gll(y_true_oof, y_ideal_mu, sigma_ideal, wavelengths_df)
print(f"Ideal GLL (L_ideal): {L_ideal:.6f}")


# --- Calculate L_ref ---
mu_ref_base = y_train.mean(axis=0).values
mu_ref = np.tile(mu_ref_base, (y_true_oof.shape[0], 1))

sigma_ref_base = y_train.std(axis=0).values
sigma_ref = np.tile(sigma_ref_base, (y_true_oof.shape[0], 1))

L_ref = calculate_gll(y_true_oof, mu_ref, sigma_ref, wavelengths_df)
print(f"Reference GLL (L_ref): {L_ref:.6f}")


# --- Calculate Final Competition Score ---
competition_score = calculate_competition_score(L_user, L_ideal, L_ref)
print(f"\nFinal OOF Competition Score: {competition_score:.6f}")

print("\nGLL Score Computation Complete.")

# --- 5. Saving Submission File ---
print("\n--- 5. Saving Submission File ---")

# --- Load Test Data for Prediction ---
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
test_planet_ids = sample_submission_df['planet_id'].tolist()

print("\n--- Generating FGS1 Features for Test Set ---")
start_time = time.time()
f_test_features = f_read_and_preprocess(TEST_DIR, test_planet_ids, band_name="FGS1")
end_time = time.time()
print(f"FGS1 Test Feature Generation took: {end_time - start_time:.2f} seconds")


print("\n--- Generating AIRS-CH0 Features for Test Set ---")
start_time = time.time()
a_test_features = a_read_and_preprocess(TEST_DIR, test_planet_ids, band_name="AIRS-CH0")
end_time = time.time()
print(f"AIRS-CH0 Test Feature Generation took: {end_time - start_time:.2f} seconds")

# Merge features for test set
X_test = f_test_features.merge(a_test_features, left_index=True, right_index=True, how='left')

# --- Add adc_info features to X_test (same logic as X_train) ---
# Assuming train_adc_info (loaded at the top of the script) contains the global ADC settings.
if not train_adc_info.empty:
    adc_features_values = train_adc_info.iloc[0].to_dict()
    for col_name, value in adc_features_values.items():
        X_test[col_name] = value
    print("ADC info features added to X_test.")
else:
    print("Warning: train_adc_info is empty. ADC features will not be added to X_test.")


# --- Ensure Test Set Columns Match Training Set Columns ---
# This re-creates a dummy X_train to get column names if X_train is not in memory.
try:
    X_train_cols = X_train.columns
except NameError:
    print("X_train not found in memory. Re-generating dummy X_train to get column names for consistency.")
    
    train_labels_dummy = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='planet_id')
    train_star_info_dummy = pd.read_csv(os.path.join(DATA_DIR, 'train_star_info.csv'))
    train_adc_info_dummy = pd.read_csv(os.path.join(DATA_DIR, 'adc_info.csv')) # Load it again for dummy

    f_train_features_dummy = f_read_and_preprocess(TRAIN_DIR, train_labels_dummy.index.tolist(), band_name="FGS1")
    a_train_features_dummy = a_read_and_preprocess(TRAIN_DIR, train_labels_dummy.index.tolist(), band_name="AIRS-CH0")

    X_train_dummy = f_train_features_dummy.merge(a_train_features_dummy, left_index=True, right_index=True, how='left')
    X_train_dummy = X_train_dummy.merge(train_star_info_dummy.set_index('planet_id'), left_index=True, right_index=True, how='left')
    
    # CORRECTED: Add adc_info features to X_train_dummy using broadcasting
    if not train_adc_info_dummy.empty:
        adc_features_values_dummy = train_adc_info_dummy.iloc[0].to_dict()
        for col_name, value in adc_features_values_dummy.items():
            X_train_dummy[col_name] = value
    
    X_train_dummy = X_train_dummy.fillna(0)
    X_train_cols = X_train_dummy.columns
    del train_labels_dummy, train_star_info_dummy, train_adc_info_dummy, f_train_features_dummy, a_train_features_dummy, X_train_dummy
    gc.collect()

X_test = X_test.reindex(columns=X_train_cols, fill_value=0)
categorical_features_indices_test = np.where(X_test.dtypes == 'object')[0].tolist()
for col_idx in categorical_features_indices_test:
    col_name = X_test.columns[col_idx]
    X_test[col_name] = X_test[col_name].astype(str)

print(f"X_test shape after feature engineering and alignment: {X_test.shape}")
print("X_test head:\n", X_test.head())

# --- Train Final Models on Full Training Data ---
print("\nTraining final models on full training data (mean and log-sigma)...")
final_mean_models = {}
final_log_sigma_models = {}

# Re-define categorical_features_indices for the full X_train if it's not global
# This ensures consistency for CatBoost when training on the full dataset.
# It was defined in the cross-validation loop, but let's ensure it's accessible here.
categorical_features_indices_full_train = np.where(X_train.dtypes == 'object')[0].tolist()


for i, col in tqdm(enumerate(target_cols), total=len(target_cols), desc="Final Model Training"):
    # Train Mean Models (Ensemble of Ridge and CatBoost)
    ridge_m = Ridge(random_state=42)
    cat_m = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function='RMSE', eval_metric='RMSE',
        random_seed=42, verbose=0, thread_count=-1
    )
    ridge_m.fit(X_train, y_train[col])
    cat_m.fit(X_train, y_train[col], cat_features=categorical_features_indices_full_train) 
    
    final_mean_models[col] = (ridge_m, cat_m)

    # Train Log-Sigma Model (Ridge)
    log_sigma_m = Ridge(random_state=42)
    cat_mean_preds_on_full_train = cat_m.predict(X_train)
    residuals_sq_full_train = (y_train[col].values - cat_mean_preds_on_full_train)**2
    target_log_var_full_train = np.log(residuals_sq_full_train + 1e-6)

    log_sigma_m.fit(X_train, target_log_var_full_train)
    final_log_sigma_models[col] = log_sigma_m

# --- Generate Predictions on Test Set ---
print("\nGenerating mean (mu) and uncertainty (sigma) predictions for the test set...")
mu_test_predictions = np.zeros((len(test_planet_ids), NUM_WAVELENGTHS))
sigma_test_predictions = np.zeros((len(test_planet_ids), NUM_WAVELENGTHS))

for i, col in tqdm(enumerate(target_cols), total=len(target_cols), desc="Predicting for test targets"):
    ridge_m, cat_m = final_mean_models[col]
    log_sigma_m = final_log_sigma_models[col]

    # Predict mean
    ridge_mu_pred = ridge_m.predict(X_test)
    cat_mu_pred = cat_m.predict(X_test)
    
    ensemble_mu_pred = (ridge_mu_pred + cat_mu_pred) / 2
    ensemble_mu_pred[ensemble_mu_pred < 0] = 0

    mu_test_predictions[:, i] = ensemble_mu_pred

    # Predict log-variance and convert to sigma
    log_var_pred = log_sigma_m.predict(X_test)
    sigma_pred = np.sqrt(np.exp(log_var_pred))
    
    min_sigma_clamp_test = np.mean(ensemble_mu_pred) * (min(IDEAL_UNC_FGS1_PPM, IDEAL_UNC_AIRS_PPM) / 1e6)
    sigma_pred = np.maximum(sigma_pred, min_sigma_clamp_test)

    sigma_test_predictions[:, i] = sigma_pred





# --- 5. Saving Submission File ---
print("\n--- 5. Saving Submission File ---")

# --- Load Test Data for Prediction ---
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
test_planet_ids = sample_submission_df['planet_id'].tolist()

print("\n--- Generating FGS1 Features for Test Set ---")
start_time = time.time()
f_test_features = f_read_and_preprocess(TEST_DIR, test_planet_ids, band_name="FGS1")
end_time = time.time()
print(f"FGS1 Test Feature Generation took: {end_time - start_time:.2f} seconds")


print("\n--- Generating AIRS-CH0 Features for Test Set ---")
start_time = time.time()
a_test_features = a_read_and_preprocess(TEST_DIR, test_planet_ids, band_name="AIRS-CH0")
end_time = time.time()
print(f"AIRS-CH0 Test Feature Generation took: {end_time - start_time:.2f} seconds")

# Merge features for test set
X_test = f_test_features.merge(a_test_features, left_index=True, right_index=True, how='left')

# --- Add adc_info features to X_test (same logic as X_train) ---
# Assuming train_adc_info (loaded at the top of the script) contains the global ADC settings.
if not train_adc_info.empty:
    adc_features_values = train_adc_info.iloc[0].to_dict()
    for col_name, value in adc_features_values.items():
        X_test[col_name] = value
    print("ADC info features added to X_test.")
else:
    print("Warning: train_adc_info is empty. ADC features will not be added to X_test.")


# --- Ensure Test Set Columns Match Training Set Columns ---
# This re-creates a dummy X_train to get column names if X_train is not in memory.
try:
    X_train_cols = X_train.columns
except NameError:
    print("X_train not found in memory. Re-generating dummy X_train to get column names for consistency.")
    
    train_labels_dummy = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='planet_id')
    train_star_info_dummy = pd.read_csv(os.path.join(DATA_DIR, 'train_star_info.csv'))
    train_adc_info_dummy = pd.read_csv(os.path.join(DATA_DIR, 'adc_info.csv')) # Load it again for dummy

    f_train_features_dummy = f_read_and_preprocess(TRAIN_DIR, train_labels_dummy.index.tolist(), band_name="FGS1")
    a_train_features_dummy = a_read_and_preprocess(TRAIN_DIR, train_labels_dummy.index.tolist(), band_name="AIRS-CH0")

    X_train_dummy = f_train_features_dummy.merge(a_train_features_dummy, left_index=True, right_index=True, how='left')
    X_train_dummy = X_train_dummy.merge(train_star_info_dummy.set_index('planet_id'), left_index=True, right_index=True, how='left')
    
    # CORRECTED: Add adc_info features to X_train_dummy using broadcasting
    if not train_adc_info_dummy.empty:
        adc_features_values_dummy = train_adc_info_dummy.iloc[0].to_dict()
        for col_name, value in adc_features_values_dummy.items():
            X_train_dummy[col_name] = value
    
    X_train_dummy = X_train_dummy.fillna(0)
    X_train_cols = X_train_dummy.columns
    del train_labels_dummy, train_star_info_dummy, train_adc_info_dummy, f_train_features_dummy, a_train_features_dummy, X_train_dummy
    gc.collect()

X_test = X_test.reindex(columns=X_train_cols, fill_value=0)
categorical_features_indices_test = np.where(X_test.dtypes == 'object')[0].tolist()
for col_idx in categorical_features_indices_test:
    col_name = X_test.columns[col_idx]
    X_test[col_name] = X_test[col_name].astype(str)

print(f"X_test shape after feature engineering and alignment: {X_test.shape}")
print("X_test head:\n", X_test.head())

# --- Train Final Models on Full Training Data ---
print("\nTraining final models on full training data (mean and log-sigma)...")
final_mean_models = {}
final_log_sigma_models = {}

# Re-define categorical_features_indices for the full X_train if it's not global
# This ensures consistency for CatBoost when training on the full dataset.
# It was defined in the cross-validation loop, but let's ensure it's accessible here.
categorical_features_indices_full_train = np.where(X_train.dtypes == 'object')[0].tolist()


for i, col in tqdm(enumerate(target_cols), total=len(target_cols), desc="Final Model Training"):
    # Train Mean Models (Ensemble of Ridge and CatBoost)
    ridge_m = Ridge(random_state=42)
    cat_m = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function='RMSE', eval_metric='RMSE',
        random_seed=42, verbose=0, thread_count=-1
    )
    ridge_m.fit(X_train, y_train[col])
    cat_m.fit(X_train, y_train[col], cat_features=categorical_features_indices_full_train) 
    
    final_mean_models[col] = (ridge_m, cat_m)

    # Train Log-Sigma Model (Ridge)
    log_sigma_m = Ridge(random_state=42)
    cat_mean_preds_on_full_train = cat_m.predict(X_train)
    residuals_sq_full_train = (y_train[col].values - cat_mean_preds_on_full_train)**2
    target_log_var_full_train = np.log(residuals_sq_full_train + 1e-6)

    log_sigma_m.fit(X_train, target_log_var_full_train)
    final_log_sigma_models[col] = log_sigma_m

# --- Generate Predictions on Test Set ---
print("\nGenerating mean (mu) and uncertainty (sigma) predictions for the test set...")
mu_test_predictions = np.zeros((len(test_planet_ids), NUM_WAVELENGTHS))
sigma_test_predictions = np.zeros((len(test_planet_ids), NUM_WAVELENGTHS))

for i, col in tqdm(enumerate(target_cols), total=len(target_cols), desc="Predicting for test targets"):
    ridge_m, cat_m = final_mean_models[col]
    log_sigma_m = final_log_sigma_models[col]

    # Predict mean
    ridge_mu_pred = ridge_m.predict(X_test)
    cat_mu_pred = cat_m.predict(X_test)
    
    ensemble_mu_pred = (ridge_mu_pred + cat_mu_pred) / 2
    ensemble_mu_pred[ensemble_mu_pred < 0] = 0

    mu_test_predictions[:, i] = ensemble_mu_pred

    # Predict log-variance and convert to sigma
    log_var_pred = log_sigma_m.predict(X_test)
    sigma_pred = np.sqrt(np.exp(log_var_pred))
    
    min_sigma_clamp_test = np.mean(ensemble_mu_pred) * (min(IDEAL_UNC_FGS1_PPM, IDEAL_UNC_AIRS_PPM) / 1e6)
    sigma_pred = np.maximum(sigma_pred, min_sigma_clamp_test)

    sigma_test_predictions[:, i] = sigma_pred


# --- Create Submission DataFrame ---
mu_cols = [f'flux_{i}' for i in range(NUM_WAVELENGTHS)]
sigma_cols = [f'uncertainty_{i}' for i in range(NUM_WAVELENGTHS)]

submission_mu_df = pd.DataFrame(mu_test_predictions, columns=mu_cols)
submission_sigma_df = pd.DataFrame(sigma_test_predictions, columns=sigma_cols)

submission_df = pd.concat([submission_mu_df, submission_sigma_df], axis=1)
submission_df.insert(0, 'planet_id', test_planet_ids)

# --- ADDED DEBUGGING PRINTS ---
print("\n--- Final Submission DataFrame Info ---")
print(f"Submission DataFrame shape: {submission_df.shape}")
print(f"Submission DataFrame columns: {submission_df.columns.tolist()[:5]} ... {submission_df.columns.tolist()[-5:]}")
print("\nSubmission DataFrame Head (first 5 rows, first 10 columns):")
print(submission_df.iloc[:, :10].head())
print("\nSubmission DataFrame Head (first 5 rows, last 10 columns):")
print(submission_df.iloc[:, -10:].head())
print("\nChecking for any NaN values in submission_df:")
print(submission_df.isnull().sum().sum()) # Should be 0

# --- Save Submission File ---
submission_file_name = 'submission.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\n--- Submission file '{submission_file_name}' created successfully! ---")


# --- Verify the saved submission file ---
print("\n--- Verifying the saved 'submission.csv' file ---")
try:
    verified_submission_df = pd.read_csv(submission_file_name)
    print("Head of the saved submission.csv:")
    print(verified_submission_df.head())
    print(f"\nShape of the saved submission.csv: {verified_submission_df.shape}")
    print(f"Columns of the saved submission.csv (first 5 and last 5): {verified_submission_df.columns.tolist()[:5]} ... {verified_submission_df.columns.tolist()[-5:]}")
except Exception as e:
    print(f"Error reading the saved submission.csv: {e}")

