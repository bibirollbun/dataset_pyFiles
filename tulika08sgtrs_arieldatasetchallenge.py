# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from pathlib import Path

ROOT=Path('/kaggle/input/ariel-data-challenge-2025')
train_df=pd.read_csv(ROOT/'train.csv')
train_star_info=pd.read_csv(ROOT/'train_star_info.csv')
wavelengths=pd.read_csv(ROOT/'wavelengths.csv')
test_star_info=pd.read_csv(ROOT/'test_star_info.csv')
sample_submission=pd.read_csv(ROOT/'sample_submission.csv')
adc_info=pd.read_csv(ROOT/'adc_info.csv')
axis_info_df = pd.read_parquet(ROOT / "axis_info.parquet")




print("Train.csv")
display(train_df.head())
print("Train.csv shape")
print(train_df.shape)
print("Train star info.csv")
display(train_star_info.head())
print("Train_star_info.csv shape")
print(train_star_info.shape)
print("Test star info.csv")
display(test_star_info.head())
print("Wavelengths.csv")
display(wavelengths.head())
print("adc_info.csv")
display(adc_info.head())
print("sample_submission.csv")
display(sample_submission.head())
print("axis_info.csv")
display(axis_info_df.head())


#both the dataframes is of the same size therefore we can merge them.
merged_df=pd.merge(train_df,train_star_info,on='planet_id')

print("shape of the merged data:")
merged_df.shape


merged_df.columns


merged_df.head()


#Study about wavelength
print(wavelengths)


#we need to reshape the wavelength dataframe since its in the form of 1 row and 283 column.
#SO DIFFICULT TO READ.

#we can consider a dataframe as an array. when we take transpose of an array then the rows become column 
#and the column become the rows.

# .T is a shorthand for transpose.
#reset_index() function then moves indexes to their regular columns.

wavelength_shaped=wavelengths.T.reset_index()


#name the columns that hold some meaning.

wavelength_shaped.columns = ['wl_id','wavelength_um']


#from the contest overview we know that two instruments are used that are: FGS-1 AND AIRS-CH0. So we now 
#try to label the wavelength data by the type of instruments.

#for FGS-1 single point=0.7 and the rest are AIRS-CH0.

#we create a new column named instrument that stores the corresponding instrument for a reading.

wavelength_shaped['instrument']=np.where(wavelength_shaped['wavelength_um']<1.0,'FGS1','AIRS-CH0')


wavelength_shaped.shape


wavelength_shaped.head(15)


fgs1_count = wavelength_shaped['instrument'].value_counts()['FGS1']
print(fgs1_count)


airs_count = wavelength_shaped['instrument'].value_counts()['AIRS-CH0']
print(airs_count)


print(wavelength_shaped['instrument'].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress the specific FutureWarning from Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)


merged_df.head()


#from merged_df the numeric data is the one that we would like to study are the ones we merged from train_star_info.
feature_cols=['Rs','Ms','Ts','Mp','e','P','sma','i']


palette=sns.color_palette("deep",len(feature_cols)) 

#create 8 subplots for 8 features ['Rs','Ms','Ts','Mp','e','P','sma','i']
fig,axes=plt.subplots(nrows=2,ncols=4,figsize=(16, 8)) 

#axes initially is a 2D array (2x4) of subplot objects. The '.flatten()' function turns it into a 1D list for easy looping
axes=axes.flatten() 

# looping through all the features of the feature_cols.
# 'ax' is the i-th subplot to draw image on. With each iteration a new subplot is considered for each feature.
# the dataframe to be plotted=merged_df
# the x-axis of the histogram will be the column of the merged_df
# kde=True === adds a smooth Kernel Density Estimation over the histogram for shape visualization.
# color=pallete[i]===gives each feature its own distinct color

for i,col in enumerate(feature_cols):   
    ax=axes[i]
    sns.histplot(
        data=merged_df,
        x=col,
        ax=ax,
        kde=True,
        color=palette[i]
    )

    # Once the histograms are plotted a title is given to each column's histogram and the title is the name of the histogram itself.
    ax.set_title(f'Distribution of {col}', fontsize=14)
    #Using log scale because count span several orders of magnitude.
    ax.set_yscale('log')  
    ax.set_xlabel('')


fig.suptitle('Distributions of Star and Planet Physical Parameters', fontsize=20, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


#now lets study the plot of a single planet 

# the first planet from the merged_df is used to study the single planet
planet_index = 0

# selects the 'planet_index' row and stores it in the form of dataframe
planet_sample = merged_df.iloc[[planet_index]]

planet_id_sample = planet_sample['planet_id'].values[0]

#identify the spectral data columns in the merged_df like wl_1, wl_2..
wl_cols = [col for col in train_df.columns if col.startswith('wl_')]

#since the planet sample is in wide format with 1 row and 284 columns s 'melt()' function converts wide to long format
# the planet id is kept as it is.
# wl_id will hold the original wavelength column names.
# the 'spectrum_value' column holds the actual measyrement values.
spectrum_sample_long = planet_sample.melt(
    id_vars=['planet_id'], value_vars=wl_cols,
    var_name='wl_id', value_name='spectrum_value'
)

print("spectrum sample long")
print(spectrum_sample_long.head())

# to the spectrum_sample_long of planet at 0th index add actual wavelength values from the wavelength_haped dataframe

spectrum_to_plot = pd.merge(spectrum_sample_long, wavelength_shaped, on='wl_id')
print("spectrum to plot")
print(spectrum_to_plot.head())

#set the custom colors for instruments
instrument_colors = {
    "FGS1": "#0077b6",  # A strong, clear blue
    "AIRS-CH0": "#d62728" # A classic, bold red
}

# the spectrum_tp_plot dataframe is plotted since it contains planet_id with corresponding wl_id (wavelength_id), spectrum_value and wavelength_um and the corresponding instrument(FGS-1 or AIRS-CH0)
#x-axis ---> wavelength
# y-axis ---> spectrum_value

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


n_planets = 10 # Choose how many planets you want to plot
planet_indices = range(n_planets)  # First n planets

wl_cols = [col for col in train_df.columns if col.startswith('wl_')]

# Collect data for all selected planets
planet_samples = merged_df.iloc[planet_indices]

# Melt and prepare long-form data
spectrum_samples_long = planet_samples.melt(
    id_vars=['planet_id'], value_vars=wl_cols,
    var_name='wl_id', value_name='spectrum_value'
)

# Merge with wavelength and instrument info
spectrum_merged = pd.merge(spectrum_samples_long, wavelength_shaped, on='wl_id')

# Optional: set a color palette for planets
planet_palette = sns.color_palette("tab10", n_planets)  # Or "husl", "Dark2", etc.
planet_colors = dict(zip(planet_samples['planet_id'], planet_palette))

# --- Plot ---
plt.figure(figsize=(14, 7))

sns.lineplot(
    data=spectrum_merged,
    x='wavelength_um',
    y='spectrum_value',
    hue='planet_id',        # Different color per planet
    style='instrument',     # Different marker style per instrument
    markers=True,
    dashes=False,
    palette=planet_colors   # Use our custom planet color palette
)

plt.title(f'Spectrum of {n_planets} Planets', fontsize=18)
plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Transit Depth (unitless)', fontsize=12)
plt.legend(title='Planet ID / Instrument', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



#sample planet=planet_id(34983)
planet_id_sample = 34983

# paths for both the instruments 
# we could have studied FGS1_signal_1.parquet or AIRS_CH0_signal_1.parquet as well but this signal 
# does not exist for planet_id=34983
fgs1_path =ROOT/ f"train/{planet_id_sample}/FGS1_signal_0.parquet"
airs_path = ROOT/ f"train/{planet_id_sample}/AIRS-CH0_signal_0.parquet"

# load both the signals for the planet_id=34983
# each file is basically a 2D detector image per time frame.=> rows are time steps and columns are pixels.
# this is the raw that we are loading (straight from the detector type of raw data !!!!)
print(f"Loading raw signal for planet {planet_id_sample}...")
fgs1_raw_df = pd.read_parquet(fgs1_path)
airs_raw_df = pd.read_parquet(airs_path)


gain = adc_info['FGS1_adc_gain'].iloc[0]
offset = adc_info['FGS1_adc_offset'].iloc[0]


#Light curve of FGS1.
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


fgs1_corrected.shape


fgs1_light_curve.shape


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


# Base frames
base_frames = [0, len(fgs1_corrected)//4, len(fgs1_corrected)//2, 3*len(fgs1_corrected)//4, -1]

# Brightest and darkest indices
brightest = np.argmax(fgs1_light_curve)
darkest = np.argmin(fgs1_light_curve)

# Extra frames
extra_frames = []
for idx in [brightest, darkest]:
    if idx not in base_frames and (idx != -1 and idx != len(fgs1_corrected)-1):
        extra_frames.append(idx)
frames = base_frames + extra_frames

plt.style.use('default')
fig = plt.figure(figsize=(18, 10))

# ---- Top Row: Frames ----
for i, idx in enumerate(frames):
    ax = plt.subplot(2, len(frames), i + 1)
    # Reshape row into 2D frame
    frame = fgs1_corrected.iloc[idx].values.reshape((32, 32))  # change 32x32 if detector size differs

    vmin, vmax = np.percentile(frame, [2, 98])
    im = ax.imshow(frame, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)

    time_min = idx * 0.1 / 60
    if idx == brightest:
        title = f'Brightest\nT={time_min:.1f} min'
    elif idx == darkest:
        title = f'Darkest\nT={time_min:.1f} min'
    else:
        title = f'T={time_min:.1f} min'
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

# ---- Bottom: Light Curve ----
ax_light = plt.subplot(2, 1, 2)
time_hours = np.arange(len(fgs1_light_curve)) * 0.1 / 3600
sample = slice(None, None, max(1, len(fgs1_light_curve)//2000))

ax_light.plot(time_hours[sample], fgs1_light_curve[sample],
              color='lightsteelblue', alpha=0.4, linewidth=0.5, label='Raw flux')

window = 500
moving_avg = pd.Series(fgs1_light_curve).rolling(window, center=True).mean()
ax_light.plot(time_hours[sample], moving_avg.iloc[sample],
              color='darkblue', linewidth=3, label=f'{window}-frame average')

colors = ['red', 'orange', 'green', 'purple', 'brown', 'lime', 'black']
for i, idx in enumerate(frames):
    time_point = idx * 0.1 / 3600
    ax_light.axvline(time_point, color=colors[i % len(colors)], alpha=0.8,
                     linewidth=1.5, linestyle='--')

ax_light.set_xlabel('Time (hours)', fontsize=12)
ax_light.set_ylabel('Total Flux (counts)', fontsize=12)
ax_light.set_title(f'Transit Light Curve - Planet {planet_id_sample}', fontsize=14, pad=15)
ax_light.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax_light.legend(loc='upper right', framealpha=0.9)

smooth_flux = moving_avg.dropna()
flux_min, flux_max = smooth_flux.min(), smooth_flux.max()
flux_range = flux_max - flux_min
margin = flux_range * 0.1
ax_light.set_ylim(flux_min - margin, flux_max + margin)

plt.tight_layout(pad=2.0)
plt.show()

# ---- Summary ----
duration = time_hours[-1]
transit_depth = np.mean(fgs1_light_curve) - np.min(fgs1_light_curve)

print(f"🌟 Planet {planet_id_sample} Transit Observation")
print(f"   Duration: {duration:.2f} hours ({len(fgs1_corrected):,} frames)")
print(f"   Brightness: {np.min(fgs1_light_curve):,.0f} → {np.max(fgs1_light_curve):,.0f} counts")
print(f"   Brightest Frame: {brightest} | Darkest Frame: {darkest}")
print(f"   Transit depth: {transit_depth:,.0f} counts "
      f"({transit_depth/np.mean(fgs1_light_curve)*100:.3f}%)")



from scipy.stats import skew, kurtosis

def extract_global_flux_features(total_flux):
    features = {}
    
    # Basic statistics
    features['global_flux_mean'] = np.mean(total_flux)
    features['global_flux_std'] = np.std(total_flux)
    features['global_flux_min'] = np.min(total_flux)
    features['global_flux_max'] = np.max(total_flux)
    features['global_flux_range'] = features['global_flux_max'] - features['global_flux_min']
    features['global_flux_skew'] = skew(total_flux)
    features['global_flux_kurtosis'] = kurtosis(total_flux)
    features['global_flux_cv'] = features['global_flux_std'] / features['global_flux_mean']
    
    # Percentiles
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        features[f'global_flux_p{p}'] = np.percentile(total_flux, p)
    
    # Transit depth features
    features['global_flux_depth'] = features['global_flux_mean'] - features['global_flux_min']
    features['global_flux_depth_ratio'] = features['global_flux_depth'] / features['global_flux_mean']
    
    return features
    
# Demo: Global Flux Features
global_features = extract_global_flux_features(fgs1_light_curve)
print(f"Generated {len(global_features)} global flux features:")

# Print ALL features organized by category
print("\nBasic Statistics:")
basic_stats = ['global_flux_mean', 'global_flux_std', 'global_flux_min', 'global_flux_max', 
               'global_flux_range', 'global_flux_skew', 'global_flux_kurtosis', 'global_flux_cv']
for feature in basic_stats:
    print(f"  {feature}: {global_features[feature]:.4f}")

print("\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    feature = f'global_flux_p{p}'
    print(f"  {feature}: {global_features[feature]:.4f}")

print("\nTransit Depth:")
transit_features = ['global_flux_depth', 'global_flux_depth_ratio']
for feature in transit_features:
    print(f"  {feature}: {global_features[feature]:.6f}")                                                                                                              









from scipy.signal import medfilt

# performing sigma clipping

'''In a dataset, most points cluster around some expected value (like the median or mean).Points that are too far away (in terms of standard deviation, 
i.e. “sigma”) are likely outliers (noise, cosmic rays, bad sensor readings, etc.).'''

def sigma_clip(data, window_size=51, sigma=5):
    """Simple sigma clipping function."""
    #calculating the median data using medfilt
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

gain = adc_info['FGS1_adc_gain'].iloc[0]
offset = adc_info['FGS1_adc_offset'].iloc[0]

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


print(train_df.head())

