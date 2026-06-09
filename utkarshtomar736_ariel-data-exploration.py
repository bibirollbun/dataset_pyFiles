import os
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("/kaggle/input/ariel-data-challenge-2025")

trn_fldr = DATA_DIR/'train'
tst_fldr = DATA_DIR/'test'

adc_info = pd.read_csv(DATA_DIR/'adc_info.csv')
axis_info = pd.read_parquet(DATA_DIR/'axis_info.parquet', engine='pyarrow')
smp_sub = pd.read_csv(DATA_DIR/'sample_submission.csv')
tst_str_info = pd.read_csv(DATA_DIR/'test_star_info.csv')
trn_csv = pd.read_csv(DATA_DIR/'train.csv')
trn_str_info = pd.read_csv(DATA_DIR/'train_star_info.csv')
waves = pd.read_csv(DATA_DIR/'wavelengths.csv')


# Check the shapes and basic info of metadata files
print("=== ADC Info ===")
print(f"Shape: {adc_info.shape}")
print(adc_info.head())
print("\n=== Axis Info ===")
print(f"Shape: {axis_info.shape}")
print(axis_info.head())
print("\n=== Training CSV ===")
print(f"Shape: {trn_csv.shape}")
print(trn_csv.head())


# Check wavelengths and star info
print("=== Wavelengths ===")
print(f"Shape: {waves.shape}")
print(waves.head())
print("\n=== Training Star Info ===")
print(f"Shape: {trn_str_info.shape}")
print(trn_str_info.head())
print("\n=== Test Star Info ===")
print(f"Shape: {tst_str_info.shape}")
print(tst_str_info.head())



# Get list of planet IDs in training data
train_planets = [p.name for p in trn_fldr.iterdir() if p.is_dir()]
test_planets = [p.name for p in tst_fldr.iterdir() if p.is_dir()]

print(f"Number of training planets: {len(train_planets)}")
print(f"Number of test planets: {len(test_planets)}")
print(f"First 5 training planets: {train_planets[:5]}")
print(f"First 5 test planets: {test_planets[:5]}")


# Check what files exist for the first training planet
first_planet = train_planets[0]
planet_path = trn_fldr / first_planet
print(f"Files for planet {first_planet}:")
for item in sorted(planet_path.iterdir()):
    print(f"  {item.name}")


# Load signal files for the first training planet
planet_id = train_planets[0]
planet_path = trn_fldr / planet_id

# Load AIRS-CH0 signal
airs_signal = pd.read_parquet(planet_path / 'AIRS-CH0_signal_0.parquet')
print(f"AIRS-CH0 signal shape: {airs_signal.shape}")
print(f"AIRS-CH0 data type: {airs_signal.dtypes.iloc[0]}")
print(f"AIRS-CH0 value range: {airs_signal.iloc[0].min()} to {airs_signal.iloc[0].max()}")

# Load FGS1 signal
fgs1_signal = pd.read_parquet(planet_path / 'FGS1_signal_0.parquet')
print(f"FGS1 signal shape: {fgs1_signal.shape}")
print(f"FGS1 data type: {fgs1_signal.dtypes.iloc[0]}")
print(f"FGS1 value range: {fgs1_signal.iloc[0].min()} to {fgs1_signal.iloc[0].max()}")


# Fixed ADC correction and reshape functions
def correct_and_reshape_airs(signal_data, adc_info):
    # Get gain and offset for AIRS-CH0
    gain = adc_info['AIRS-CH0_adc_gain'].iloc[0]
    offset = adc_info['AIRS-CH0_adc_offset'].iloc[0]
    
    # Apply correction: divide by gain, add offset, convert to float64
    corrected = (signal_data.values / gain + offset).astype(np.float64)
    
    # Reshape from flattened (11250, 11392) to (11250, 32, 356)
    reshaped = corrected.reshape(11250, 32, 356)
    
    return reshaped

def correct_and_reshape_fgs1(signal_data, adc_info):
    # Get gain and offset for FGS1
    gain = adc_info['FGS1_adc_gain'].iloc[0]
    offset = adc_info['FGS1_adc_offset'].iloc[0]
    
    # Apply correction: divide by gain, add offset, convert to float64
    corrected = (signal_data.values / gain + offset).astype(np.float64)
    
    # Reshape from flattened (135000, 1024) to (135000, 32, 32)
    reshaped = corrected.reshape(135000, 32, 32)
    
    return reshaped

# Apply corrections
airs_corrected = correct_and_reshape_airs(airs_signal, adc_info)
fgs1_corrected = correct_and_reshape_fgs1(fgs1_signal, adc_info)

print(f"AIRS-CH0 corrected shape: {airs_corrected.shape}")
print(f"FGS1 corrected shape: {fgs1_corrected.shape}")
print(f"AIRS-CH0 corrected range: {airs_corrected.min():.2f} to {airs_corrected.max():.2f}")
print(f"FGS1 corrected range: {fgs1_corrected.min():.2f} to {fgs1_corrected.max():.2f}")


# Let's see the effect of ADC correction
print("=== ADC Correction Effect ===")
print(f"AIRS-CH0 gain: {adc_info['AIRS-CH0_adc_gain'].iloc[0]}")
print(f"AIRS-CH0 offset: {adc_info['AIRS-CH0_adc_offset'].iloc[0]}")
print(f"FGS1 gain: {adc_info['FGS1_adc_gain'].iloc[0]}")
print(f"FGS1 offset: {adc_info['FGS1_adc_offset'].iloc[0]}")

# Example transformation for a few values
sample_raw = np.array([1000, 2000, 3000, 4000, 5000])
sample_corrected = (sample_raw / 0.4369 + (-1000)).astype(np.float64)
print(f"\nSample raw values: {sample_raw}")
print(f"Sample corrected values: {sample_corrected}")


import matplotlib.pyplot as plt

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot time series for AIRS-CH0 (sum over spatial dimensions)
airs_time_series = airs_corrected.sum(axis=(1, 2))
axes[0, 0].plot(airs_time_series)
axes[0, 0].set_title('AIRS-CH0 Time Series (Total Flux)')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Total Flux')

# Plot time series for FGS1 (sum over spatial dimensions)
fgs1_time_series = fgs1_corrected.sum(axis=(1, 2))
axes[1, 0].plot(fgs1_time_series)
axes[1, 0].set_title('FGS1 Time Series (Total Flux)')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Total Flux')

# Show a sample frame from AIRS-CH0
im1 = axes[0, 1].imshow(airs_corrected[0], aspect='auto')
axes[0, 1].set_title('AIRS-CH0 Frame 0')
plt.colorbar(im1, ax=axes[0, 1])

# Show a sample frame from FGS1
im2 = axes[1, 1].imshow(fgs1_corrected[0])
axes[1, 1].set_title('FGS1 Frame 0')
plt.colorbar(im2, ax=axes[1, 1])

# Show mean frame from AIRS-CH0
im3 = axes[0, 2].imshow(airs_corrected.mean(axis=0), aspect='auto')
axes[0, 2].set_title('AIRS-CH0 Mean Frame')
plt.colorbar(im3, ax=axes[0, 2])

# Show mean frame from FGS1
im4 = axes[1, 2].imshow(fgs1_corrected.mean(axis=0))
axes[1, 2].set_title('FGS1 Mean Frame')
plt.colorbar(im4, ax=axes[1, 2])

plt.tight_layout()
plt.show()




