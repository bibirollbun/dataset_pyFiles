


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq

# Set the data directory
DATA_DIR = Path("/kaggle/input/ariel-data-challenge-2025")


# 1. Load metadata
train_df = pd.read_csv(DATA_DIR / "train.csv")
wavelengths = pd.read_csv(DATA_DIR / "wavelengths.csv").values.flatten()
star_info = pd.read_csv(DATA_DIR / "train_star_info.csv")
adc_info = pd.read_csv(DATA_DIR / "adc_info.csv")

display(adc_info)

# Retrieve ADC parameters
gain = adc_info['FGS1_adc_gain'].values[0]
offset = adc_info['FGS1_adc_offset'].values[0]


# 2. Load and preprocess signal data
def load_signal_data(planet_id, instrument="AIRS-CH0", obs_count=0):
    """Load and preprocess signal data"""
    file_path = DATA_DIR / f"train/{planet_id}/{instrument}_signal_{obs_count}.parquet"
    data = pq.read_table(file_path).to_pandas().values

    # Reconstruct dynamic range via ADC conversion
    data = (data / gain) + offset
    data = data.astype(np.float64)

    # Reshape into image format (frames, height, width)
    if instrument == "AIRS-CH0":
        data = data.reshape(-1, 32, 356)
    else:  # FGS1
        data = data.reshape(-1, 32, 32)

    return data

# Example: Load AIRS-CH0 data for the first planet
planet_id = train_df['planet_id'].values[0]
airs_data = load_signal_data(planet_id, "AIRS-CH0")


# 3. Load calibration data
def load_calibration(planet_id, instrument="AIRS-CH0", calib_type="dark"):
    """Load calibration data"""
    file_path = DATA_DIR / f"train/{planet_id}/{instrument}_calibration_0/{calib_type}.parquet"
    return pq.read_table(file_path).to_pandas().values

# Load dark frames
dark_frames = load_calibration(planet_id, "AIRS-CH0", "dark")


# 4. Visualize data
def plot_sample_frames(data, title, n_samples=3):
    """Visualize sample frames"""
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(data[i], cmap='viridis')
        plt.title(f"{title} - Frame {i}")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

# Visualize the first 3 frames of AIRS-CH0 data
plot_sample_frames(airs_data, "AIRS-CH0 Raw Data")


# 5. Basic preprocessing pipeline
def preprocess_data(raw_data, dark_frames):
    """Basic preprocessing: dark subtraction and normalization"""
    # Subtract average dark frame
    avg_dark = np.mean(dark_frames, axis=0)
    processed = raw_data - avg_dark

    # Normalize along the time axis for each pixel
    processed = (processed - np.mean(processed, axis=0)) / np.std(processed, axis=0)

    return processed

# Apply preprocessing (first 1000 frames only to save memory)
processed_data = preprocess_data(airs_data[:1000], dark_frames)
plot_sample_frames(processed_data, "Processed Data")


# 6. Extract time series signal (light curve from center pixel region)
def extract_light_curve(data, x=None, y=None, radius=5):
    """Extract brightness variations from a specified region with NaN handling"""
    if x is None or y is None:
        # Automatically compute the center coordinates
        height, width = data.shape[1], data.shape[2]
        x, y = width // 2, height // 2

    y_slice = slice(max(0, y - radius), min(data.shape[1], y + radius + 1))
    x_slice = slice(max(0, x - radius), min(data.shape[2], x + radius + 1))

    # Compute the mean while ignoring NaN values
    light_curve = np.nanmean(data[:, y_slice, x_slice], axis=(1, 2))

    # If all values are NaN, fill with zeros
    if np.all(np.isnan(light_curve)):
        light_curve = np.zeros_like(light_curve)

    return light_curve

# Check statistics of the processed data
print("Processed data stats:")
print(f"Min: {np.nanmin(processed_data)}, Max: {np.nanmax(processed_data)}")
print(f"NaN ratio: {np.mean(np.isnan(processed_data)):.2%}")

# Re-extract light curve
lc = extract_light_curve(processed_data)
print("Light curve samples:", lc[:10])  # Show the first 10 points

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(lc)
plt.title("Light Curve (with NaN handling)")
plt.xlabel("Frame Number")
plt.ylabel("Normalized Flux")
plt.grid()
plt.show()



# 7. Visualize spectral data
def plot_spectrum(planet_id, train_df):
    """Visualize the spectrum of a given planet"""
    spectrum = train_df[train_df['planet_id'] == planet_id].iloc[:, 1:].values.flatten()

    plt.figure(figsize=(12, 4))
    plt.plot(wavelengths, spectrum, 'b-')
    plt.title(f"Spectrum of Planet {planet_id}")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Flux")
    plt.grid()
    plt.show()

plot_spectrum(planet_id, train_df)




