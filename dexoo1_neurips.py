# step 0

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os, glob
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


# step 1

train_df = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
print("Train shape:", train_df.shape)

wavelengths_df = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')
print("Wavelengths shape:", wavelengths_df.shape)

adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
print(adc_info)

train_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')
print("Star info shape:", train_star_info.shape)

axis_info = pq.read_table('/kaggle/input/ariel-data-challenge-2025/axis_info.parquet').to_pandas()
print("Axis Info shape:", axis_info.shape)


def process_fgs1_data(planet_id, adc_info):
    print(f'\n--- Processing Planet ID: {planet_id} ---')
    base_path = f"/kaggle/input/ariel-data-challenge-2025/train/{planet_id}"
    calib_path = f"{base_path}/FGS1_calibration_0"

    # Load raw signal
    fgs1_signal = pq.read_table(f"{base_path}/FGS1_signal_0.parquet").to_pandas()
    print(f"FGS1 raw signal Shape: {fgs1_signal.shape}")

    # Load calibration frames and convert to NumPy
    fgs1_dark = pq.read_table(f"{calib_path}/dark.parquet").to_pandas().to_numpy()
    fgs1_flat = pq.read_table(f"{calib_path}/flat.parquet").to_pandas().to_numpy()
    fgs1_dead = pq.read_table(f"{calib_path}/dead.parquet").to_pandas().to_numpy()
    fgs1_linear_corr = pq.read_table(f"{calib_path}/linear_corr.parquet").to_pandas()

    print(f"Shape of dark: {fgs1_dark.shape}")
    print(f"Shape of flat: {fgs1_flat.shape}")
    print(f"Shape of dead: {fgs1_dead.shape}")
    print(f"Shape of linear_corr: {fgs1_linear_corr.shape}")

    # Reshape raw signal from (135000, 1024) â†’ (135000, 32, 32)
    fgs1_signal_np = fgs1_signal.to_numpy().reshape((135000, 32, 32)).astype(np.float32)

    # Apply calibration
    signal = fgs1_signal_np - fgs1_dark[np.newaxis, :, :]
    signal = signal / fgs1_flat[np.newaxis, :, :]
    signal[:, fgs1_dead == 1] = np.nan

    # Apply ADC
    gain = adc_info['FGS1_adc_gain'].iloc[0]
    offset = adc_info['FGS1_adc_offset'].iloc[0]
    signal = signal * gain + offset

    return signal


def process_airs_data(planet_id, adc_info):
    print(f'\n--- Processing AIRS Data for Planet ID: {planet_id} ---')
    base_path = f"/kaggle/input/ariel-data-challenge-2025/train/{planet_id}"
    calib_path = f"{base_path}/AIRS-CH0_calibration_0"

    # Load raw signal
    airs_signal = pq.read_table(f"{base_path}/AIRS-CH0_signal_0.parquet").to_pandas()
    print(f"AIRS raw signal shape: {airs_signal.shape}")

    # Load calibration files and convert to NumPy
    airs_dark = pq.read_table(f"{calib_path}/dark.parquet").to_pandas().to_numpy()
    airs_flat = pq.read_table(f"{calib_path}/flat.parquet").to_pandas().to_numpy()
    airs_dead = pq.read_table(f"{calib_path}/dead.parquet").to_pandas().to_numpy()
    airs_linear_corr = pq.read_table(f"{calib_path}/linear_corr.parquet").to_pandas()

    print(f"Shape of dark: {airs_dark.shape}")
    print(f"Shape of flat: {airs_flat.shape}")
    print(f"Shape of dead: {airs_dead.shape}")
    print(f"Shape of linear_corr: {airs_linear_corr.shape}")

    # Reshape signal from (11250, 11392) â†’ (11250, 32, 356)
    airs_signal_np = airs_signal.to_numpy().reshape((11250, 32, 356)).astype(np.float32)

    # Calibration
    signal = airs_signal_np - airs_dark[np.newaxis, :, :]
    signal = signal / airs_flat[np.newaxis, :, :]
    signal[:, airs_dead == 1] = np.nan

    # Apply ADC
    gain = adc_info['AIRS-CH0_adc_gain'].iloc[0]
    offset = adc_info['AIRS-CH0_adc_offset'].iloc[0]
    signal = signal * gain + offset

    return signal


planet_id = 1010375142
fgs1_signal = process_fgs1_data(planet_id, adc_info)
airs_signal = process_airs_data(planet_id, adc_info)

print(fgs1_signal)
print("-------------------------------------------")
print(airs_signal)


# # step 2.2

# def process_airs_data(planet_id, adc_info):
#     print(f"\n--- Processing AIRS-CH0 for Planet ID: {planet_id} ---")

#     base_path = f"/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/"
#     calib_path = f"{base_path}/AIRS-CH0_calibration_0/"  # Fixed calibration path

#     # 1. Load AIRS-CH0 signal (CORRECTED FILENAME)
#     airs_ch0_signal = pq.read_table(f"{base_path}/AIRS-CH0_signal_0.parquet").to_pandas()
#     print(f"AIRS-CH0 raw signal shape: {airs_ch0_signal.shape}")

#     # 2. Load calibration data
#     print("Loading calibration files...")
#     airs_dark_calib = pq.read_table(f"{calib_path}/dark.parquet").to_pandas()
#     airs_flat_calib = pq.read_table(f"{calib_path}/flat.parquet").to_pandas()
#     airs_dead_calib = pq.read_table(f"{calib_path}/dead.parquet").to_pandas()
    
#     print(f"Dark calib shape: {airs_dark_calib.shape}")  # Should be (32, 356)
#     print(f"Flat calib shape: {airs_flat_calib.shape}")
#     print(f"Dead pixel map shape: {airs_dead_calib.shape}")

#     # 3. ADC correction
#     try:
#         airs_gain = adc_info.loc[adc_info['instrument'] == 'AIRS-CH0', 'gain'].values[0]
#         airs_offset = adc_info.loc[adc_info['instrument'] == 'AIRS-CH0', 'offset'].values[0]
#     except:
#         # Fallback if column names are different
#         airs_gain = adc_info['AIRS-CH0_adc_gain'].iloc[0]
#         airs_offset = adc_info['AIRS-CH0_adc_offset'].iloc[0]
        
#     print(f"AIRS ADC Gain: {airs_gain}")
#     print(f"AIRS ADC Offset: {airs_offset}")

#     # 4. Apply ADC correction and RESHAPE PROPERLY
#     airs_corrected = (airs_ch0_signal.values * airs_gain) + airs_offset
#     print(f"Corrected signal shape: {airs_corrected.shape}")
    
#     # Correct reshape: AIRS has 11250 timepoints Ã— (32 Ã— 356) pixels
#     try:
#         airs_images = airs_corrected.reshape(-1, 32, 356)
#     except ValueError:
#         # Calculate expected number of pixels
#         pixels = airs_corrected.shape[1]
#         frames = airs_corrected.shape[0]
#         print(f"WARNING: {pixels} pixels, unable to reshape to (X,32,356). Using reshape({frames},32,356))")
#         airs_images = airs_corrected.reshape(frames, 32, 356)
    
#     print(f"Reshaped AIRS images: {airs_images.shape}")

#     # 5. Calibration pipeline
#     # Dark subtraction
#     dark_frame = airs_dark_calib.values.squeeze()  # Handle possible extra dimensions
#     if dark_frame.ndim == 3:
#         dark_frame = np.median(dark_frame, axis=0)
#     calibrated_images = airs_images.copy()
#     calibrated_images -= dark_frame

#     # Dead pixel correction
#     dead_mask = (airs_dead_calib.values > 0).squeeze()  # Remove single-dimensional entries
#     dead_mask = dead_mask.astype(bool)
#     if dead_mask.shape != (32, 356):
#         dead_mask = dead_mask.reshape(32, 356)
#     print(f"Dead pixels: {np.sum(dead_mask)}/{dead_mask.size}")

#     # Vectorized dead pixel fix
#     from scipy.ndimage import generic_filter
#     dead_mask_3d = np.zeros_like(calibrated_images, dtype=bool)
#     dead_mask_3d[:] = dead_mask[np.newaxis, :, :]  # Broadcast mask to all frames
    
#     # Create filtered version
#     filtered_airs = np.stack([
#         generic_filter(calibrated_images[i], median_nan_fill, size=3, mode='mirror')
#         for i in range(calibrated_images.shape[0])
#     ])
    
#     calibrated_images[dead_mask_3d] = filtered_airs[dead_mask_3d]

#     # Flat field correction
#     flat_img = airs_flat_calib.values
#     if flat_img.ndim == 3:
#         flat_img = np.median(flat_img, axis=0)
#     normalized_flat = flat_img / np.median(flat_img)
#     calibrated_images /= normalized_flat

#     # 6. Extract flux per wavelength channel (sum over rows)
#     # This is CRITICAL for AIRS spectral separation
#     print("Extracting flux per spectral column...")
#     flux_per_wavelength = np.sum(calibrated_images, axis=1)  # Sum over rows (32 â†’ 1)
#     print(f"Flux per wavelength shape: {flux_per_wavelength.shape}")  # (11250, 356)

#     # 7. Visualization and return
#     # Plot light curve for first spectral channel
#     plt.figure(figsize=(12,4))
#     plt.plot(flux_per_wavelength[:, 0])
#     plt.title(f"AIRS-CH0 Spectra Channel 0 - Planet {planet_id}")
    
#     # Heatmap of first frame
#     plt.figure(figsize=(10,6))
#     plt.imshow(calibrated_images[0], aspect='auto', cmap='viridis')
#     plt.title("First Calibrated Frame")

#     return flux_per_wavelength  # Return (11250, 356) spectral flux array

# # Helper function for vectorized dead pixel replacement
# def median_nan_fill(window):
#     """Replace center pixel with median of neighbors if NaN"""
#     center = window[window.size//2]
#     neighbors = np.delete(window, window.size//2)
#     return np.nanmedian(neighbors) if np.isnan(center) else center



# def extract_spectrum_from_flux(fgs1_flux, airs_flux):
#     """
#     Reduce time series flux into 1D spectrum and corresponding uncertainty.
#     Returns:
#         - mu (length 283)
#         - sigma (length 283)
#     """
#     # FGS1 is a time series â†’ reduce to 1 value (mean + std)
#     mu_fgs1 = np.mean(fgs1_flux)
#     sigma_fgs1 = np.std(fgs1_flux)

#     # AIRS is (11250, 356) â†’ mean over time gives shape (356,)
#     mu_airs = np.mean(airs_flux, axis=0)[:282]
#     sigma_airs = np.std(airs_flux, axis=0)[:282]

#     # Combine into total 283
#     mu = np.concatenate(([mu_fgs1], mu_airs))  # 1 + 282 = 283
#     sigma = np.concatenate(([sigma_fgs1], sigma_airs))

#     return mu, sigma


# def format_submission_row(planet_id, mu, sigma):
#     return [planet_id] + list(mu) + list(sigma)


# def process_single_planet(planet_id, adc_info):
#     try:
#         print(f"\nğŸš€ Starting processing for planet: {planet_id}")

#         fgs1_flux, _ = process_fgs1_data(planet_id, adc_info)
#         airs_flux = process_airs_data(planet_id, adc_info)

#         mu, sigma = extract_spectrum_from_flux(fgs1_flux, airs_flux)
#         submission_row = format_submission_row(planet_id, mu, sigma)

#         # Header format fix
#         mu_cols = [f"wl_{i+1}" for i in range(283)]
#         sigma_cols = [f"sigma_{i+1}" for i in range(283)]
#         header = ["planet_id"] + mu_cols + sigma_cols

#         submission_df = pd.DataFrame([submission_row], columns=header)
#         submission_df.to_csv("submission.csv", index=False)

#         print("âœ… Done! Created submission.csv for planet:", planet_id)
#         return submission_df

#     except Exception as e:
#         print(f"â�Œ Failed for planet {planet_id}: {e}")
#         return None



# planet_id = 1010375142
# process_single_planet(planet_id, adc_info)

