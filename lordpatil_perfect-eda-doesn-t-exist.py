import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. Define the base path for the data ---
# This is the standard path in a Kaggle environment
BASE_PATH = Path("/kaggle/input/ariel-data-challenge-2025")

# --- 2. Load all the main metadata files ---
print("Loading all top-level metadata files...")
try:
    train_df = pd.read_csv(BASE_PATH / "train.csv")
    train_star_info_df = pd.read_csv(BASE_PATH / "train_star_info.csv")
    wavelengths_df = pd.read_csv(BASE_PATH / "wavelengths.csv")
    adc_info_df = pd.read_csv(BASE_PATH / "adc_info.csv")
    axis_info_df = pd.read_parquet(BASE_PATH / "axis_info.parquet")
    sample_submission_df = pd.read_csv(BASE_PATH / "sample_submission.csv")
    print("All files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure your notebook is connected to the competition dataset.")

# --- 3. Display the head of each dataframe ---

print("\n" + "="*50)
print("1. Ground Truth Spectra (train.csv)")
print("="*50)
# This file contains the target values (the true spectra) we need to predict.
# The columns are likely numbered 0 to 282, corresponding to different wavelengths.
display(train_df.head())

print("\n" + "="*50)
print("2. Star and Planet Physical Info (train_star_info.csv)")
print("="*50)
# These are the physical characteristics of each system. These will be our primary features.
display(train_star_info_df.head())

print("\n" + "="*50)
print("3. Wavelength Grid (wavelengths.csv)")
print("="*50)
# This file maps the column indices from train.csv to physical wavelengths (in µm).
# It's the 'x-axis' for our spectra.
display(wavelengths_df.head())

print("\n" + "="*50)
print("4. ADC Conversion Parameters (adc_info.csv)")
print("="*50)
# These are the gain and offset values to restore the raw signal data's dynamic range.
display(adc_info_df.head())

print("\n" + "="*50)
print("5. Axis Information (axis_info.parquet)")
print("="*50)
# This should contain timing or coordinate information for the raw signal data.
display(axis_info_df.head())

print("\n" + "="*50)
print("6. Sample Submission Format (sample_submission.csv)")
print("="*50)
# This shows us exactly how our final output file must be structured.
display(sample_submission_df.head())


# --- 1. Merge the training data ---
# We'll use an 'inner' merge, which is the default. This ensures that we only keep
# planets that appear in both files.
merged_train_df = pd.merge(train_df, train_star_info_df, on="planet_id")

print("\n" + "="*50)
print("1. Merged Training Data (Spectra + Star/Planet Info)")
print("="*50)
print(f"Shape of merged data: {merged_train_df.shape}")
print("This dataframe now contains our primary features (star/planet info) and targets (spectra).")
display(merged_train_df.head())


# --- 2. Reshape the wavelengths data ---
# The original format is wide (1 row, 283 columns). Let's make it long.
wavelengths_long_df = wavelengths_df.T.reset_index()
wavelengths_long_df.columns = ['wl_id', 'wavelength_um']

# Now, let's add the instrument based on the wavelength, as described in the report.
# FGS1 is a single point at 0.7 um. The rest are AIRS-CH0.
wavelengths_long_df['instrument'] = np.where(wavelengths_long_df['wavelength_um'] < 1.0, 'FGS1', 'AIRS-CH0')

print("\n" + "="*50)
print("2. Reshaped Wavelengths Data")
print("="*50)
print(f"Shape of reshaped wavelength data: {wavelengths_long_df.shape}")
print("This format is much easier to use for plotting and lookups.")
display(wavelengths_long_df.head()) # Shows the FGS1 point
display(wavelengths_long_df.tail()) # Shows the AIRS-CH0 points

# Let's double-check the counts per instrument
print("\nInstrument counts from reshaped wavelength data:")
print(wavelengths_long_df['instrument'].value_counts())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress the specific FutureWarning from Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set a nice theme for Seaborn plots
sns.set_theme(style="whitegrid")

# These are the columns we want to inspect
feature_cols = ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']

palette = sns.color_palette("viridis", len(feature_cols))

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    sns.histplot(
        data=merged_train_df,
        x=col,
        ax=ax,
        kde=True,
        color=palette[i] # Assign the i-th color from our new palette
    )
    ax.set_title(f'Distribution of {col}', fontsize=14)
    ax.set_yscale('log')
    ax.set_xlabel('')

if len(feature_cols) < len(axes):
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

fig.suptitle('Distributions of Star and Planet Physical Parameters', fontsize=20, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


# ==============================================================================
# PLOT 2: Spectrum of a Single Planet with custom colors
# ==============================================================================

print("\n--- 2. Visualizing a Single Spectrum with custom colors ---")

# --- Data Preparation (same as before) ---
planet_index = 0
planet_sample = merged_train_df.iloc[[planet_index]]
planet_id_sample = planet_sample['planet_id'].values[0]
wl_cols = [col for col in train_df.columns if col.startswith('wl_')]
spectrum_sample_long = planet_sample.melt(
    id_vars=['planet_id'], value_vars=wl_cols,
    var_name='wl_id', value_name='spectrum_value'
)
spectrum_to_plot = pd.merge(spectrum_sample_long, wavelengths_long_df, on='wl_id')


instrument_colors = {
    "FGS1": "#0077b6",  # A strong, clear blue
    "AIRS-CH0": "#d62728" # A classic, bold red
}

# --- Plotting with Seaborn ---
plt.figure(figsize=(14, 7))

sns.lineplot(
    data=spectrum_to_plot,
    x='wavelength_um',
    y='spectrum_value',
    hue='instrument',
    style='instrument',
    markers=True,
    dashes=False,
    palette=instrument_colors # Apply our custom color dictionary
)

plt.title(f'Ground Truth Spectrum for Planet ID: {planet_id_sample}', fontsize=18)
plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Transit Depth (unitless)', fontsize=12)
plt.legend(title='Instrument', fontsize=11)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()




planet_id_sample = 34983
fgs1_path = BASE_PATH / f"train/{planet_id_sample}/FGS1_signal_0.parquet"
airs_path = BASE_PATH / f"train/{planet_id_sample}/AIRS-CH0_signal_0.parquet"

print(f"Loading raw signal for planet {planet_id_sample}...")
fgs1_raw_df = pd.read_parquet(fgs1_path)
airs_raw_df = pd.read_parquet(airs_path)

gain = adc_info_df['FGS1_adc_gain'].iloc[0]
offset = adc_info_df['FGS1_adc_offset'].iloc[0]

fgs1_corrected = fgs1_raw_df * gain + offset
airs_corrected = airs_raw_df * gain + offset

fgs1_light_curve = fgs1_corrected.sum(axis=1)
airs_light_curve = airs_corrected.sum(axis=1)



print("Plotting light curves...")


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9))

# FGS1 Plot (Top)
sns.lineplot(x=fgs1_light_curve.index, y=fgs1_light_curve, ax=axes[0], color="#0077b6", lw=0.5)
axes[0].set_title('FGS1 Light Curve', fontsize=16)
axes[0].set_ylabel('Total Flux (Arbitrary Units)', fontsize=12)
# Add an x-label to the top plot for clarity
axes[0].set_xlabel('Frame Number (Time)', fontsize=12)


# AIRS-CH0 Plot (Bottom)
sns.lineplot(x=airs_light_curve.index, y=airs_light_curve, ax=axes[1], color="#d62728", lw=0.5)
axes[1].set_title('AIRS-CH0 Light Curve', fontsize=16)
axes[1].set_ylabel('Total Flux (Arbitrary Units)', fontsize=12)
axes[1].set_xlabel('Frame Number (Time)', fontsize=12)

fig.suptitle(f"Raw Light Curves for Planet ID: {planet_id_sample}", fontsize=20, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


fgs1_diff = fgs1_light_curve.iloc[1::2].values - fgs1_light_curve.iloc[0::2].values
airs_diff = airs_light_curve.iloc[1::2].values - airs_light_curve.iloc[0::2].values


print("Plotting 'Differential' Light Curves...")


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9))

# --- FGS1 Plot (Top subplot) ---
sns.lineplot(
    x=np.arange(len(fgs1_diff)), # Create an x-axis for the measurement number
    y=fgs1_diff,
    ax=axes[0],
    color="#0077b6", # Consistent blue color
    lw=1 # Use a thin line for clarity
)
axes[0].set_title('FGS1 Differential Light Curve', fontsize=16)
axes[0].set_ylabel('Differential Flux', fontsize=12)


# --- AIRS-CH0 Plot (Bottom subplot) ---
sns.lineplot(
    x=np.arange(len(airs_diff)),
    y=airs_diff,
    ax=axes[1],
    color="#d62728", # Consistent red color
    lw=1
)
axes[1].set_title('AIRS-CH0 Differential Light Curve', fontsize=16)
axes[1].set_ylabel('Differential Flux', fontsize=12)
axes[1].set_xlabel('Measurement Number (Time)', fontsize=12) # Label the shared x-axis



fig.suptitle(f"Differential Light Curves for Planet ID: {planet_id_sample}", fontsize=20, y=1.01)

# Improve layout to prevent titles/labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


from scipy.signal import medfilt


def sigma_clip(data, window_size=51, sigma=5):
    """Simple sigma clipping function."""
    local_median = medfilt(data, kernel_size=window_size)
    residual = data - local_median
    mad = np.median(np.abs(residual))
    robust_std = mad * 1.4826
    outliers = np.abs(residual) > (sigma * robust_std)
    clean_data = data.copy()
    clean_data[outliers] = np.nan
    return clean_data, outliers

print("--- 1. Performing Sigma Clipping on AIRS-CH0 Data ---")
airs_clipped, airs_outliers = sigma_clip(airs_diff)


# --- 2. Detrending and Normalization - This code is unchanged ---
out_of_transit_mask = np.ones_like(airs_clipped, dtype=bool)
out_of_transit_mask[1700:4200] = False
out_of_transit_mask[np.isnan(airs_clipped)] = False
x_axis = np.arange(len(airs_clipped))
poly_coeffs = np.polyfit(x_axis[out_of_transit_mask], airs_clipped[out_of_transit_mask], 2)
baseline_model = np.polyval(poly_coeffs, x_axis)
airs_normalized = airs_clipped / baseline_model


# --- 3. Visualize the Entire Cleaning Process (Seaborn/Matplotlib version) ---
print("\n--- 2. Visualizing the Cleaning Pipeline ---")

# Create a figure with 2 subplots, one on top of the other, sharing the x-axis
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

# --- Top Plot: Sigma Clipping and Baseline Fitting ---
ax_top = axes[0]
# Plot 1: The original differential data as a light background line
sns.lineplot(x=x_axis, y=airs_diff, ax=ax_top, color='lightgrey', label='Original Diff. Data')
# Plot 2: The baseline model as a thick orange line
sns.lineplot(x=x_axis, y=baseline_model, ax=ax_top, color='orange', linewidth=3, label='Baseline Fit')
# Plot 3: The clipped outliers as red markers
# Note: we plot the original 'airs_diff' values at the outlier indices
sns.scatterplot(x=x_axis[airs_outliers], y=airs_diff[airs_outliers], ax=ax_top, color='red', marker='o', s=50, label='Clipped Outliers', zorder=5)

ax_top.set_title('Step 1: Sigma Clipping and Baseline Fitting', fontsize=16)
ax_top.set_ylabel('Flux', fontsize=12)
ax_top.legend()


# --- Bottom Plot: Final Normalized Light Curve ---
ax_bottom = axes[1]
sns.lineplot(x=x_axis, y=airs_normalized, ax=ax_bottom, color='dodgerblue', label='Normalized Flux')
ax_bottom.set_title('Step 2: Final Normalized Light Curve', fontsize=16)
ax_bottom.set_ylabel('Normalized Flux', fontsize=12)
ax_bottom.set_xlabel('Measurement Number', fontsize=12)
# Set y-axis limits to better see the transit depth
ax_bottom.set_ylim(bottom=min(0.98, np.nanmin(airs_normalized) - 0.005), top=1.01)
ax_bottom.legend()


fig.suptitle(f"Data Cleaning Pipeline for Planet {planet_id_sample} (AIRS-CH0)", fontsize=20, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


print("--- Examining the Raw Data Structure ---")
print(f"Shape of FGS1 signal data: {fgs1_raw_df.shape}")
print(f"Expected pixel count for a 32x32 detector: 32 * 32 = {32*32}\n")

print(f"Shape of AIRS-CH0 signal data: {airs_raw_df.shape}")
print(f"Expected pixel count for a 32x356 detector: 32 * 356 = {32*356}\n")

gain = adc_info_df['FGS1_adc_gain'].iloc[0]
offset = adc_info_df['FGS1_adc_offset'].iloc[0]

# FGS1 Data
frame_index_fgs1 = 5000
fgs1_frame_flat = fgs1_raw_df.iloc[frame_index_fgs1]
fgs1_frame_corrected = fgs1_frame_flat * gain + offset
fgs1_frame_2d = fgs1_frame_corrected.values.reshape(32, 32)

# AIRS-CH0 Data
frame_index_airs = 5000
airs_frame_flat = airs_raw_df.iloc[frame_index_airs]
airs_frame_corrected = airs_frame_flat * gain + offset
airs_frame_2d = airs_frame_corrected.values.reshape(32, 356)



print("--- Visualizing Detector Frames with Seaborn ---")


# We make the figure wider to accommodate the long AIRS-CH0 detector
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

# --- FGS1 Visualization (Top Plot) ---
ax_fgs1 = axes[0]
sns.heatmap(
    fgs1_frame_2d,
    ax=ax_fgs1,
    cmap='viridis', # This is the same color scale as in Plotly
    cbar_kws={'label': 'Flux'} # Add a label to the color bar
)
ax_fgs1.set_title(f'FGS1 Detector Frame #{frame_index_fgs1}', fontsize=16)
ax_fgs1.set_xlabel('Pixel Column', fontsize=12)
ax_fgs1.set_ylabel('Pixel Row', fontsize=12)

# --- AIRS-CH0 Visualization (Bottom Plot) ---
ax_airs = axes[1]
sns.heatmap(
    airs_frame_2d,
    ax=ax_airs,
    cmap='viridis',
    cbar_kws={'label': 'Flux'}
)
ax_airs.set_title(f'AIRS-CH0 Detector Frame #{frame_index_airs}', fontsize=16)
ax_airs.set_xlabel('Wavelength Axis (Pixel Column)', fontsize=12)
ax_airs.set_ylabel('Spatial Axis (Pixel Row)', fontsize=12)


# --- Final Touches ---
fig.suptitle(f"Single Detector Frames for Planet {planet_id_sample}", fontsize=20, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()




