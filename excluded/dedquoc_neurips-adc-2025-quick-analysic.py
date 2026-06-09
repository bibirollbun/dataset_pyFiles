!pip install optuna
!pip install shap
# install pqdm for parallel processing
!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm


import pandas as pd
import numpy as np
import os
import glob

import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from tqdm import tqdm
from pqdm.threads import pqdm
import itertools
import pickle

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

import plotly.express as px

from astropy.stats import sigma_clip
from scipy.signal import savgol_filter   


# Load metadata CSV files
adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
axis_info = pq.read_table('/kaggle/input/ariel-data-challenge-2025/axis_info.parquet').to_pandas()
train_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')
test_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/test_star_info.csv')
wavelengths = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')
sample_submission = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv')

# Optional: Ground truth for training
train_df= pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')


adc_info.head(5)


train_star_info.head(5)


test_star_info.head(6)


wavelengths.head(5)


sample_submission.head(5)


train_df.head(5)


adc_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")

# Extract AIRS-CH0 gain and offset
instrument = "AIRS-CH0"
gain = adc_info[f"{instrument}_adc_gain"].iloc[0]
offset = adc_info[f"{instrument}_adc_offset"].iloc[0]

print(f"{instrument} gain = {gain}, offset = {offset}")


%%time
# Path to the raw detector data
read_path = "/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_calibration_0/read.parquet"

# Load the raw ADC signal
raw_df = pd.read_parquet(read_path)
print("Raw shape:", raw_df.shape)
print(raw_df.head())


%%time
import pandas as pd

# Set your dataset path (example: one PartQuest ID folder from train set)
dataset_path = "/kaggle/input/ariel-data-challenge-2025/train/1010375142/AIRS-CH0_calibration_0/"

# Load the read.parquet file
airs_data = pd.read_parquet(dataset_path + "read.parquet")

# Print basic info
print("Shape of airs_data:", airs_data.shape)
airs_data.head()


# Apply calibration
corrected_df = raw_df * gain + offset

# Display summary
print(corrected_df.describe())


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the star-planet system info
star_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train_star_info.csv")

# Select a system
row = star_info.iloc[0]
planet_id = row["planet_id"]

# Extract parameters
Rs = row["Rs"]                    # Star radius (solar radii)
sma = row["sma"]                  # Semi-major axis (AU)
e = row["e"]                      # Eccentricity
i_deg = row["i"]                  # Inclination (degrees)
i_rad = np.radians(i_deg)

# Convert sma from AU to solar radii
AU_TO_SOLAR_RADII = 215.032
sma_rsun = sma * AU_TO_SOLAR_RADII

# Orbital ellipse in polar coordinates
theta = np.linspace(0, 2 * np.pi, 500)
r = sma_rsun * (1 - e**2) / (1 + e * np.cos(theta))

# Coordinates in orbital plane
x_orb = r * np.cos(theta)
y_orb = r * np.sin(theta)
z_orb = np.zeros_like(theta)

# Rotate by inclination (around x-axis)
y_inc = y_orb * np.cos(i_rad)
z_inc = y_orb * np.sin(i_rad)

# 3D plot
fig = go.Figure()

# Star at origin
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=8, color='yellow'),
    name='Star'
))

# Planet orbit
fig.add_trace(go.Scatter3d(
    x=x_orb, y=y_inc, z=z_inc,
    mode='lines',
    line=dict(color='deepskyblue'),
    name='Orbit'
))

# Planet position (e.g., at periapsis)
fig.add_trace(go.Scatter3d(
    x=[x_orb[0]], y=[y_inc[0]], z=[z_inc[0]],
    mode='markers',
    marker=dict(size=4, color='blue'),
    name='Planet (start)'
))

# Layout
fig.update_layout(
    title=f"3D Orbit of Planet {planet_id} (Inclination: {i_deg:.1f}°)",
    scene=dict(
        xaxis_title='X (solar radii)',
        yaxis_title='Y (inclined)',
        zaxis_title='Z (inclined)',
        aspectmode='data',
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

fig.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Select a planet system (first row for demo)
planet = star_info.iloc[0]

# Parameters
sma = planet["sma"]        # semi-major axis in stellar radii
e = planet["e"]            # eccentricity
i = np.radians(planet["i"])  # inclination in radians
P = planet["P"]            # orbital period (not needed here unless time scaling)
Rs = planet["Rs"]          # star radius

# Compute orbit (Keplerian ellipse in x-y plane, rotate for inclination)
theta = np.linspace(0, 2 * np.pi, 200)
r = (sma * (1 - e**2)) / (1 + e * np.cos(theta))  # polar equation of ellipse

# Convert to 3D Cartesian
x = r * np.cos(theta)
y = r * np.sin(theta)
z = y * np.sin(i)
y = y * np.cos(i)  # adjust y after inclination

# Set up 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.set_xlim(-sma*1.2, sma*1.2)
ax.set_ylim(-sma*1.2, sma*1.2)
ax.set_zlim(-sma*1.2, sma*1.2)
ax.set_title("Animated Star-Planet System")

# Star
ax.scatter(0, 0, 0, color='yellow', s=300, label='Star')

# Static orbit
ax.plot(x, y, z, color='gray', linestyle='--', alpha=0.5)

# Planet dot (animated)
planet_dot, = ax.plot([], [], [], 'o', color='blue', markersize=10, label='Planet')

def update(frame):
    planet_dot.set_data(x[frame], y[frame])
    planet_dot.set_3d_properties(z[frame])
    return planet_dot,

ani = FuncAnimation(fig, update, frames=len(theta), interval=50, blit=True)

plt.legend()
plt.show()



trail_length = 30  # frames

def update(frame):
    start = max(0, frame - trail_length)
    planet_dot.set_data(x[frame], y[frame])
    planet_dot.set_3d_properties(z[frame])
    ax.plot(x[start:frame], y[start:frame], z[start:frame], color='blue', alpha=0.4)
    return planet_dot,


from matplotlib.cm import plasma
temp_norm = (planet["Ts"] - 3000) / (7000 - 3000)  # normalize temp to [0,1]
star_color = plasma(temp_norm)
ax.scatter(0, 0, 0, color=star_color, s=300)


# Plot a few pixel time series
num_pixels_to_plot = 5
plt.figure(figsize=(12, 5))

for i in range(num_pixels_to_plot):
    plt.plot(corrected_df.iloc[i], label=f'Pixel {i}')

plt.title(f'{instrument} Calibrated Signal (First {num_pixels_to_plot} Pixels)')
plt.xlabel('Time')
plt.ylabel('Calibrated Signal')
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load gain and offset for AIRS-CH0
adc_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")
instrument = "AIRS-CH0"
gain = adc_info[f"{instrument}_adc_gain"].iloc[0]
offset = adc_info[f"{instrument}_adc_offset"].iloc[0]
print(f"{instrument} gain = {gain}, offset = {offset}")

# 2. Load AIRS-CH0 signal file
signal_path = "/kaggle/input/ariel-data-challenge-2025/train/1253730513/AIRS-CH0_signal_0.parquet"
signal_df = pd.read_parquet(signal_path)

# 3. Apply gain and offset correction
signal_corrected = signal_df.values.astype(np.float32) * gain + offset

# 4. Reshape to (time, 32, 356)
signal_reshaped = signal_corrected.reshape(-1, 32, 356)

# 5. Plot a few frames
n_frames_to_plot = 3
fig, axes = plt.subplots(1, n_frames_to_plot, figsize=(15, 5))

for i in range(n_frames_to_plot):
    axes[i].imshow(signal_reshaped[i], cmap='viridis', aspect='auto')
    axes[i].set_title(f"Frame {i}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()



%%time
def feature_engineering(f_raw, a_raw):
    """Create a dataframe with two features from the raw data.
    
    Parameters:
    f_raw: ndarray of shape (n_planets, 67500)
    a_raw: ndarray of shape (n_planets, 5625)
    
    Return value:
    df: DataFrame of shape (n_planets, 2)
    """
    # Feature from f_raw: broadband flux
    f_obscured = f_raw[:, 23500:44000].mean(axis=1)  # Transit/eclipse in flux
    f_unobscured = (f_raw[:, :20500].mean(axis=1) + f_raw[:, 47000:].mean(axis=1)) / 2
    f_relative_reduction = (f_unobscured - f_obscured) / f_unobscured

    # Feature from a_raw: spectroscopic data
    a_obscured = a_raw[:, 1958:3666].mean(axis=1)  # Corresponding occultation window
    a_unobscured = (a_raw[:, :1708].mean(axis=1) + a_raw[:, 3916:].mean(axis=1)) / 2
    a_relative_reduction = (a_unobscured - a_obscured) / a_unobscured

    # Create DataFrame with both features
    df = pd.DataFrame({
        'a_relative_reduction': a_relative_reduction,
        'f_relative_reduction': f_relative_reduction
    })
    
    return df   


%%time
def build_feature_vector(planet_id):
    # Get spectrum from train.csv
    spec_row = train_df[train_df["planet_id"] == planet_id]
    if spec_row.empty:
        print(f"[!] planet_id {planet_id} not found in train.csv")
        return None

    # Get star/planet metadata
    phys_row = star_info[star_info["planet_id"] == planet_id]
    if phys_row.empty:
        print(f"[!] planet_id {planet_id} not found in star_info.csv")
        return None

    # Drop planet_id to avoid duplication
    spectrum = spec_row.drop(columns=["planet_id"]).reset_index(drop=True)
    phys = phys_row.drop(columns=["planet_id"]).reset_index(drop=True)

    # Combine
    combined = pd.concat([spectrum, phys], axis=1)
    combined["planet_id"] = planet_id  # Optional: keep ID

    return combined


%%time

def build_full_training_data():
    feature_rows = []
    for idx, row in train_df.iterrows():
        planet_id = row["planet_id"]
        features = build_feature_vector(planet_id)
        if features is not None:
            # Add label
            features["wl_276"] = row["wl_276"]
            feature_rows.append(features)

    return pd.concat(feature_rows, axis=0).reset_index(drop=True)


%%time
def build_full_training_data():
    feature_rows = []
    for idx, row in train_df.iterrows():
        planet_id = row["planet_id"]
        
        # Load f_raw and a_raw for this planet
        f_raw = np.load(f'/kaggle/input/ariel-data-challenge-2025/train/f/{planet_id}.npy')
        a_raw = np.load(f'/kaggle/input/ariel-data-challenge-2025/train/a/{planet_id}.npy')
        
        # Expand dimensions if needed (if single sample)
        f_raw = f_raw.reshape(1, -1)  # Shape: (1, 67500)
        a_raw = a_raw.reshape(1, -1)  # Shape: (1, 5625)
        
        # Use your feature_engineering function
        features_df = feature_engineering(f_raw, a_raw)
        
        # Add label
        features_df["wl_276"] = row["wl_276"]
        
        feature_rows.append(features_df)
    
    # Concatenate all feature rows
    return pd.concat(feature_rows, axis=0).reset_index(drop=True)   


%%time
class Config:
    # === Paths and Dataset Mode ===
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = 'test'  # Change to 'train' for training runs

    # === Signal Processing Parameters ===
    CUT_INF = 39
    CUT_SUP = 250
    N_JOBS = 4  # Parallel processing workers

    # === Model Scaling and Uncertainty ===
    SCALE = 0.95
    SIGMA = 0.0009  # Assumed constant uncertainty

    # === Transit Detection Settings ===
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 7   # Guard band around transit
    MODEL_POLYNOMIAL_DEGREE = 3    # Baseline fit degree

    # === Sensor Configuration (Computed Safely) ===
    @property
    def SENSOR_CONFIG(self):
        CUT_SUP = self.CUT_SUP
        CUT_INF = self.CUT_INF

        return {
            "AIRS-CH0": {
                "raw_shape": [11250, 32, 356],
                "calibrated_shape": [1, 32, CUT_SUP - CUT_INF],
                "linear_corr_shape": (6, 32, 356),
                "dt_pattern": (0.1, 4.5),
                "binning": 30
            },
            "FGS1": {
                "raw_shape": [135000, 32, 32],
                "calibrated_shape": [1, 32, 32],
                "linear_corr_shape": (6, 32, 32),
                "dt_pattern": (0.1, 0.1),
                "binning": 30 * 12  # 360 frames → 1 binned point
            }
        }

    # === Convenience Properties ===
    @property
    def is_training(self):
        """Check if current dataset is 'train'."""
        return self.DATASET.lower() == 'train'

    @property
    def sample_submission_path(self):
        return f"{self.DATA_PATH}/sample_submission.csv"

    @property
    def star_info_path(self):
        return f"{self.DATA_PATH}/{self.DATASET}_star_info.csv"

    # Optional: Add debug mode
    @property
    def DEBUG(self):
        return False  # Set to True for fast testing on small subset   


%%time

class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', index_col='planet_id').index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):
        linear_corr_flipped = np.flip(linear_corr, axis=0)
        corrected_signal = signal.copy()
        
        for x, y in itertools.product(range(signal.shape[1]), range(signal.shape[2])):
            poly = np.poly1d(linear_corr_flipped[:, x, y])
            corrected_signal[:, x, y] = poly(corrected_signal[:, x, y])
            
        return corrected_signal

    def _calibrate_single_signal(self, planet_id, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]

        signal = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_signal_0.parquet").to_numpy()
        dark = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/flat.parquet").to_numpy()
        linear_corr = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet").values.astype(np.float64).reshape(sensor_cfg["linear_corr_shape"])

        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask

        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
        
        base_dt, increment = sensor_cfg["dt_pattern"]
        dt = np.ones(len(signal)) * base_dt
        dt[1::2] += increment
        
        signal = signal.clip(0)
        signal = self._apply_linear_corr(linear_corr, signal)
        signal -= dark * dt[:, np.newaxis, np.newaxis]
        
        flat = flat.reshape(sensor_cfg["calibrated_shape"])
        flat[dead.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        flat[hot.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        
        signal = signal / flat
        
        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)
        
        mean_signal = np.nanmean(signal_roi, axis=1)

        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))
            
        return binned

    def _process_planet_sensor(self, args):
        planet_id, sensor = args['planet_id'], args['sensor']
        calibrated = self._calibrate_single_signal(planet_id, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self):
        args_fgs1 = [dict(planet_id=planet_id, sensor="FGS1") for planet_id in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        args_airs_ch0 = [dict(planet_id=planet_id, sensor="AIRS-CH0") for planet_id in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal 



%%time
class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', index_col='planet_id').index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):
        linear_corr_flipped = np.flip(linear_corr, axis=0)
        corrected_signal = signal.copy()
        
        for x, y in itertools.product(range(signal.shape[1]), range(signal.shape[2])):
            poly = np.poly1d(linear_corr_flipped[:, x, y])
            corrected_signal[:, x, y] = poly(corrected_signal[:, x, y])
            
        return corrected_signal

    def _calibrate_single_signal(self, planet_id, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]

        signal = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_signal_0.parquet").to_numpy()
        dark = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/flat.parquet").to_numpy()
        linear_corr = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet").values.astype(np.float64).reshape(sensor_cfg["linear_corr_shape"])

        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask

        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
        
        base_dt, increment = sensor_cfg["dt_pattern"]
        dt = np.ones(len(signal)) * base_dt
        dt[1::2] += increment
        
        signal = signal.clip(0)
        signal = self._apply_linear_corr(linear_corr, signal)
        signal -= dark * dt[:, np.newaxis, np.newaxis]
        
        flat = flat.reshape(sensor_cfg["calibrated_shape"])
        flat[dead.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        flat[hot.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        
        signal = signal / flat
        
        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)
        
        mean_signal = np.nanmean(signal_roi, axis=1)

        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))
            
        return binned

    def _process_planet_sensor(self, args):
        planet_id, sensor = args['planet_id'], args['sensor']
        calibrated = self._calibrate_single_signal(planet_id, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self):
        args_fgs1 = [dict(planet_id=planet_id, sensor="FGS1") for planet_id in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        args_airs_ch0 = [dict(planet_id=planet_id, sensor="AIRS-CH0") for planet_id in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal


%%time
class TransitModel:
    def __init__(self, config):
        self.cfg = config

    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]

        grad1 = np.gradient(signal1)
        grad1 /= grad1.max()
        
        grad2 = np.gradient(signal2)
        grad2 /= grad2.max()

        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index

        return phase1, phase2
    
    def _objective_function(self, s, signal, phase1, phase2):
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE

        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
            delta = 2

        y = np.concatenate([
            signal[: phase1 - delta],
            signal[phase1 + delta : phase2 - delta] * (1 + s),
            signal[phase2 + delta :]
        ])
        x = np.arange(len(y))

        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        
        return error

    def predict(self, single_preprocessed_signal):
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
        signal_1d = savgol_filter(signal_1d, 20, 2)
        
        phase1, phase2 = self._phase_detector(signal_1d)

        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    

        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead"
        )
        
        return result.x[0]

    def predict_all(self, preprocessed_signals):
        predictions = [
            self.predict(preprocessed_signal)
            for preprocessed_signal in tqdm(preprocessed_signals)
        ]
        return np.array(predictions) * self.cfg.SCALE


%%time

class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv(f"{self.cfg.DATA_PATH}/sample_submission.csv", index_col="planet_id")
        

    def create(self, predictions):
        planet_ids = self.sample_submission.index
        repeated_predictions = np.repeat(predictions, self.sample_submission.shape[1] // 2).reshape(len(predictions), -1)
        repeated_predictions = repeated_predictions.clip(0)
        
        sigmas = np.ones_like(repeated_predictions) * self.cfg.SIGMA

        submission_df = pd.DataFrame(
            np.concatenate([repeated_predictions, sigmas], axis=1),
            columns=self.sample_submission.columns,
            index=planet_ids
        )
        
        submission_df.to_csv("submission.csv")
        return submission_df



%%time
config = Config()
    
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)

submission_generator = SubmissionGenerator(config)
submission = submission_generator.create(predictions)

