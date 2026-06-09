import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from scipy import signal, stats
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

import sys


base_path = '/kaggle/input/ariel-data-challenge-2025/'

train_df = pd.read_csv(f"{base_path}/train.csv")
print("Train")
display(train_df.head(2))

train_star_info_df = pd.read_csv(f"{base_path}/train_star_info.csv")
print("Train Star Info")
display(train_star_info_df.head(2))

test_star_info_df = pd.read_csv(f"{base_path}/test_star_info.csv")
print("Test")
display(test_star_info_df)


# for doing quick runs (not loading all data)
quick_run = False
quick_run_size = 200

# disable quick_run if submitting!
if len(test_star_info_df) > 1:
    quick_run = False

if quick_run:
    train_star_info_df=train_star_info_df.head(quick_run_size)
    print(f"Only loading {quick_run_size} records - this run is not a full training!")


%%time
# Load sample data for demonstrations
planet_id = int(train_star_info_df.iloc[0]['planet_id'])
file_path = f"/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_0.parquet"
df = pd.read_parquet(file_path)
signal_data = df.values.reshape(135000, 32, 32)

print(f"Demo data loaded for Planet ID: {planet_id}")
print(f"Signal data shape: {signal_data.shape}")

print("="*60)


def apply_adc_correction(signal_array, instrument, adc_info_path="/kaggle/input/ariel-data-challenge-2025/adc_info.csv"):
    """
    Applies ADC correction to raw signal data using known gain and offset.

    Args:
        signal_array (np.ndarray): Raw uint16 signal data.
        instrument (str): 'FGS1' or 'AIRS-CH0'.
        adc_info_path (str): Path to adc_info.csv file.

    Returns:
        np.ndarray: Calibrated float signal array.
    """
    # Load ADC values
    adc_df = pd.read_csv(adc_info_path)
    
    gain_col = f"{instrument}_adc_gain"
    offset_col = f"{instrument}_adc_offset"

    if gain_col not in adc_df.columns or offset_col not in adc_df.columns:
        raise ValueError(f"Missing columns: {gain_col} or {offset_col} in adc_info.csv")
    
    gain = adc_df.at[0, gain_col]
    offset = adc_df.at[0, offset_col]

    # Apply gain and offset
    calibrated_signal = signal_array.astype(np.float32) * gain + offset
    return calibrated_signal
    
signal_data = apply_adc_correction(signal_data, instrument='FGS1')
total_flux = np.sum(signal_data, axis=(1, 2))


%%time
def visualize_transit_data(signal_data, total_flux, planet_id):    
    # Key frame indices (quarter-points + first/last)
    base_frames = [0, len(signal_data)//4, len(signal_data)//2, 3*len(signal_data)//4, -1]

    # Find brightest and darkest frame indices
    brightest_idx = np.argmax(total_flux)
    darkest_idx = np.argmin(total_flux)

    # Add to the frame list, avoiding duplicates
    extra_frames = []
    for idx in [brightest_idx, darkest_idx]:
        if idx not in base_frames and (idx != -1 and idx != len(signal_data)-1):
            extra_frames.append(idx)
    frames = base_frames + extra_frames

    # Set up clean styling
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 10))
    
    # Top row: telescope frames
    for i, idx in enumerate(frames):
        ax = plt.subplot(2, len(frames), i + 1)
        frame = signal_data[idx]
        
        # Contrast scaling
        vmin, vmax = np.percentile(frame, [2, 98])
        im = ax.imshow(frame, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
        
        # Annotate
        time_min = idx * 0.1 / 60
        if idx == brightest_idx:
            title = f'Brightest\nT={time_min:.1f} min'
        elif idx == darkest_idx:
            title = f'Darkest\nT={time_min:.1f} min'
        else:
            title = f'T={time_min:.1f} min'
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Bottom: Light curve
    ax_light = plt.subplot(2, 1, 2)
    time_hours = np.arange(len(total_flux)) * 0.1 / 3600
    sample = slice(None, None, max(1, len(total_flux)//2000))

    ax_light.plot(time_hours[sample], total_flux[sample], 
                 color='lightsteelblue', alpha=0.4, linewidth=0.5, label='Raw flux')

    window = 500
    moving_avg = pd.Series(total_flux).rolling(window, center=True).mean()
    ax_light.plot(time_hours[sample], moving_avg.iloc[sample], 
                 color='darkblue', linewidth=3, label=f'{window}-frame average')

    colors = ['red', 'orange', 'green', 'purple', 'brown', 'lime', 'black']
    for i, idx in enumerate(frames):
        time_point = idx * 0.1 / 3600
        ax_light.axvline(time_point, color=colors[i % len(colors)], alpha=0.8, linewidth=1.5, linestyle='--')

    ax_light.set_xlabel('Time (hours)', fontsize=12)
    ax_light.set_ylabel('Total Flux (counts)', fontsize=12)
    ax_light.set_title(f'Transit Light Curve - Planet {planet_id}', fontsize=14, pad=15)
    ax_light.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_light.legend(loc='upper right', framealpha=0.9)

    smooth_flux = moving_avg.dropna()
    flux_min, flux_max = smooth_flux.min(), smooth_flux.max()
    flux_range = flux_max - flux_min
    margin = flux_range * 0.1
    ax_light.set_ylim(flux_min - margin, flux_max + margin)

    plt.tight_layout(pad=2.0)
    plt.show()

    # Enhanced summary
    duration = time_hours[-1]
    transit_depth = np.mean(total_flux) - np.min(total_flux)

    print(f"ğŸŒŸ Planet {planet_id} Transit Observation")
    print(f"   Duration: {duration:.2f} hours ({len(signal_data):,} frames)")
    print(f"   Brightness: {np.min(total_flux):,.0f} â†’ {np.max(total_flux):,.0f} counts")
    print(f"   Brightest Frame: {brightest_idx} | Darkest Frame: {darkest_idx}")
    print(f"   Transit depth: {transit_depth:,.0f} counts ({transit_depth/np.mean(total_flux)*100:.3f}%)")
    
visualize_transit_data(signal_data, total_flux, planet_id)


%%time
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
global_features = extract_global_flux_features(total_flux)
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

print(f"\nKey transit indicator - depth ratio: {global_features['global_flux_depth_ratio']:.6f}")


%%time
def extract_rolling_statistics_features(total_flux):
    features = {}
    window_sizes = [50, 100, 500, 1000, 2000, 5000]
    
    for window in window_sizes:
        if window < len(total_flux):
            # Calculate rolling statistics
            rolling_mean = pd.Series(total_flux).rolling(window=window, center=True).mean().dropna()
            rolling_std = pd.Series(total_flux).rolling(window=window, center=True).std().dropna()
            rolling_min = pd.Series(total_flux).rolling(window=window, center=True).min().dropna()
            rolling_max = pd.Series(total_flux).rolling(window=window, center=True).max().dropna()
            
            if len(rolling_mean) > 0:
                # Statistics of rolling statistics
                features[f'rolling{window}_mean_min'] = rolling_mean.min()
                features[f'rolling{window}_mean_max'] = rolling_mean.max()
                features[f'rolling{window}_mean_std'] = rolling_mean.std()
                features[f'rolling{window}_mean_range'] = rolling_mean.max() - rolling_mean.min()
                
                features[f'rolling{window}_std_mean'] = rolling_std.mean()
                features[f'rolling{window}_std_max'] = rolling_std.max()
                
                # Extreme values
                features[f'rolling{window}_deepest_dip'] = rolling_min.min()
                features[f'rolling{window}_highest_peak'] = rolling_max.max()
                features[f'rolling{window}_volatility'] = rolling_std.std()
    
    return features

# Demo: Rolling Statistics Features
rolling_features = extract_rolling_statistics_features(total_flux)
print(f"Generated {len(rolling_features)} rolling statistics features:")

# Print ALL features organized by window size
for window in [50, 100, 500, 1000, 2000, 5000]:
    window_features = {k: v for k, v in rolling_features.items() if f'rolling{window}_' in k}
    if window_features:
        print(f"\nWindow {window} features ({len(window_features)} total):")
        for feature_name, value in window_features.items():
            print(f"  {feature_name}: {value:.4f}")


%%time
def extract_transit_detection_features(total_flux):
    features = {}
    
    # Detrend the light curve
    baseline = pd.Series(total_flux).rolling(window=5000, center=True).median().fillna(method='bfill').fillna(method='ffill')
    detrended = total_flux - baseline
    
    # Detrended statistics
    features['detrended_min'] = np.min(detrended)
    features['detrended_std'] = np.std(detrended)
    features['detrended_skew'] = skew(detrended)
    features['detrended_neg_excursions'] = np.sum(detrended < -2 * np.std(detrended))
    features['detrended_deep_excursions'] = np.sum(detrended < -3 * np.std(detrended))
    
    # Transit period detection
    threshold = np.mean(total_flux) - 1.0 * np.std(total_flux)
    below_threshold = total_flux < threshold
    
    if np.any(below_threshold):
        # Duration analysis
        diff = np.diff(np.concatenate(([False], below_threshold, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        durations = ends - starts
        
        features['longest_dip_duration'] = np.max(durations) if len(durations) > 0 else 0
        features['num_dip_periods'] = len(durations)
        features['total_dip_time'] = np.sum(durations)
        features['avg_dip_duration'] = np.mean(durations) if len(durations) > 0 else 0
        
        # Timing analysis
        deepest_idx = np.argmin(total_flux)
        features['deepest_time_fraction'] = deepest_idx / len(total_flux)
        features['deepest_in_first_half'] = float(deepest_idx < len(total_flux) / 2)
        features['deepest_in_middle_third'] = float(len(total_flux) / 3 < deepest_idx < 2 * len(total_flux) / 3)
        
        # Transit shape
        transit_flux = total_flux[below_threshold]
        features['transit_depth_mean'] = np.mean(transit_flux)
        features['transit_depth_std'] = np.std(transit_flux)
        features['transit_assymetry'] = skew(transit_flux)
        features['transit_flatness'] = kurtosis(transit_flux)
    else:
        # No transit detected
        features.update({
            'longest_dip_duration': 0, 'num_dip_periods': 0, 'total_dip_time': 0, 'avg_dip_duration': 0,
            'deepest_time_fraction': 0.5, 'deepest_in_first_half': 0, 'deepest_in_middle_third': 0,
            'transit_depth_mean': np.mean(total_flux), 'transit_depth_std': 0, 
            'transit_assymetry': 0, 'transit_flatness': 0
        })
    
    # Observation structure
    first_quarter = total_flux[:len(total_flux)//4]
    last_quarter = total_flux[-len(total_flux)//4:]
    middle_half = total_flux[len(total_flux)//4:-len(total_flux)//4]
    
    features['first_quarter_mean'] = np.mean(first_quarter)
    features['last_quarter_mean'] = np.mean(last_quarter)
    features['middle_half_mean'] = np.mean(middle_half)
    features['middle_vs_edges'] = features['middle_half_mean'] - (features['first_quarter_mean'] + features['last_quarter_mean']) / 2
    
    return features

# Demo: Transit Detection Features
transit_features = extract_transit_detection_features(total_flux)
print(f"Generated {len(transit_features)} transit detection features:")

# Print ALL features organized by category
print("\nDetrended Statistics:")
detrended_features = ['detrended_min', 'detrended_std', 'detrended_skew', 
                     'detrended_neg_excursions', 'detrended_deep_excursions']
for feature in detrended_features:
    print(f"  {feature}: {transit_features[feature]:.4f}")

print("\nDuration Metrics:")
duration_features = ['longest_dip_duration', 'num_dip_periods', 'total_dip_time', 'avg_dip_duration']
for feature in duration_features:
    print(f"  {feature}: {transit_features[feature]:.4f}")

print("\nTiming Features:")
timing_features = ['deepest_time_fraction', 'deepest_in_first_half', 'deepest_in_middle_third']
for feature in timing_features:
    print(f"  {feature}: {transit_features[feature]:.4f}")

print("\nTransit Shape:")
shape_features = ['transit_depth_mean', 'transit_depth_std', 'transit_assymetry', 'transit_flatness']
for feature in shape_features:
    print(f"  {feature}: {transit_features[feature]:.4f}")

print("\nObservation Structure:")
structure_features = ['first_quarter_mean', 'last_quarter_mean', 'middle_half_mean', 'middle_vs_edges']
for feature in structure_features:
    print(f"  {feature}: {transit_features[feature]:.4f}")


%%time
def extract_frequency_features(total_flux):
    features = {}
    
    # FFT analysis
    fft_flux = np.fft.fft(total_flux - np.mean(total_flux))
    fft_power = np.abs(fft_flux)
    fft_freqs = np.fft.fftfreq(len(total_flux))
    
    # Power spectrum
    features['fft_peak_power'] = np.max(fft_power[1:len(fft_power)//2])
    features['fft_total_power'] = np.sum(fft_power[1:len(fft_power)//2])
    features['fft_mean_power'] = np.mean(fft_power[1:len(fft_power)//2])
    features['fft_std_power'] = np.std(fft_power[1:len(fft_power)//2])
    
    # Low frequency analysis
    low_freq_mask = np.abs(fft_freqs) < 0.01
    features['fft_low_freq_power'] = np.sum(fft_power[low_freq_mask])
    features['fft_low_freq_ratio'] = features['fft_low_freq_power'] / features['fft_total_power']
    
    # Spectral properties
    power_spectrum = fft_power[1:len(fft_power)//2]
    freqs = np.abs(fft_freqs[1:len(fft_freqs)//2])
    
    if np.sum(power_spectrum) > 0:
        features['spectral_centroid'] = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * power_spectrum) / np.sum(power_spectrum))
    else:
        features['spectral_centroid'] = 0
        features['spectral_bandwidth'] = 0
    
    # Autocorrelation
    autocorr = np.correlate(total_flux - np.mean(total_flux), total_flux - np.mean(total_flux), mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]
    
    # Lag features
    for lag in [10, 50, 100, 500, 1000]:
        if lag < len(autocorr):
            features[f'autocorr_lag{lag}'] = autocorr[lag]
    
    # Peak detection
    peaks, _ = signal.find_peaks(autocorr[1:1000], height=0.1)
    features['autocorr_num_peaks'] = len(peaks)
    features['autocorr_first_peak'] = peaks[0] if len(peaks) > 0 else 0
    features['autocorr_strongest_peak'] = np.max(autocorr[peaks]) if len(peaks) > 0 else 0
    
    return features

# Demo: Frequency Features
frequency_features = extract_frequency_features(total_flux)
print(f"Generated {len(frequency_features)} frequency domain features:")

# Print ALL features organized by category
print("\nPower Spectrum:")
power_features = ['fft_peak_power', 'fft_total_power', 'fft_mean_power', 'fft_std_power']
for feature in power_features:
    print(f"  {feature}: {frequency_features[feature]:.4f}")

print("\nLow Frequency Analysis:")
low_freq_features = ['fft_low_freq_power', 'fft_low_freq_ratio']
for feature in low_freq_features:
    print(f"  {feature}: {frequency_features[feature]:.4f}")

print("\nSpectral Properties:")
spectral_features = ['spectral_centroid', 'spectral_bandwidth']
for feature in spectral_features:
    print(f"  {feature}: {frequency_features[feature]:.6f}")

print("\nAutocorrelation Lags:")
for lag in [10, 50, 100, 500, 1000]:
    feature = f'autocorr_lag{lag}'
    if feature in frequency_features:
        print(f"  {feature}: {frequency_features[feature]:.6f}")

print("\nAutocorrelation Peaks:")
peak_features = ['autocorr_num_peaks', 'autocorr_first_peak', 'autocorr_strongest_peak']
for feature in peak_features:
    print(f"  {feature}: {frequency_features[feature]:.4f}")


%%time
def extract_spatial_features(signal_data):
    features = {}
    
    # Key time points
    key_frames = [0, len(signal_data) // 4, len(signal_data) // 2, 3 * len(signal_data) // 4, -1]
    centroids_x, centroids_y, concentrations = [], [], []
    
    for i, frame_idx in enumerate(key_frames):
        frame = signal_data[frame_idx]
        
        # Centroid calculation
        y_indices, x_indices = np.indices(frame.shape)
        total_spatial_flux = np.sum(frame)
        
        if total_spatial_flux > 0:
            centroid_x = np.sum(x_indices * frame) / total_spatial_flux
            centroid_y = np.sum(y_indices * frame) / total_spatial_flux
        else:
            centroid_x = centroid_y = 16
        
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)
        
        # Concentration
        center_region = frame[12:20, 12:20]
        concentration = np.sum(center_region) / total_spatial_flux if total_spatial_flux > 0 else 0
        concentrations.append(concentration)
        
        # Frame-specific features
        features[f'frame{i}_spatial_mean'] = np.mean(frame)
        features[f'frame{i}_spatial_std'] = np.std(frame)
        features[f'frame{i}_centroid_x'] = centroid_x
        features[f'frame{i}_centroid_y'] = centroid_y
        features[f'frame{i}_concentration'] = concentration
    
    # Movement analysis
    features['centroid_x_range'] = np.max(centroids_x) - np.min(centroids_x)
    features['centroid_y_range'] = np.max(centroids_y) - np.min(centroids_y)
    features['centroid_total_movement'] = np.sum(np.sqrt(np.diff(centroids_x)**2 + np.diff(centroids_y)**2))
    features['concentration_range'] = np.max(concentrations) - np.min(concentrations)
    features['concentration_std'] = np.std(concentrations)
    
    return features

# Demo: Spatial Features
spatial_features = extract_spatial_features(signal_data)
print(f"Generated {len(spatial_features)} spatial features:")

# Print ALL features organized by frame and analysis type
print("\nPer-Frame Spatial Statistics:")
for i in range(5):
    print(f"  Frame {i}:")
    print(f"    spatial_mean: {spatial_features[f'frame{i}_spatial_mean']:.4f}")
    print(f"    spatial_std: {spatial_features[f'frame{i}_spatial_std']:.4f}")
    print(f"    centroid_x: {spatial_features[f'frame{i}_centroid_x']:.4f}")
    print(f"    centroid_y: {spatial_features[f'frame{i}_centroid_y']:.4f}")
    print(f"    concentration: {spatial_features[f'frame{i}_concentration']:.6f}")

print("\nMovement Analysis:")
movement_features = ['centroid_x_range', 'centroid_y_range', 'centroid_total_movement', 
                    'concentration_range', 'concentration_std']
for feature in movement_features:
    print(f"  {feature}: {spatial_features[feature]:.6f}")


%%time
def extract_gradient_features(total_flux):
    features = {}
    
    # Derivatives
    total_flux = np.array(total_flux, dtype=np.float64)
    flux_diff1 = np.diff(total_flux)
    flux_diff2 = np.diff(flux_diff1)
    
    # First derivative statistics
    features['flux_diff1_mean'] = np.mean(flux_diff1)
    features['flux_diff1_std'] = np.std(flux_diff1)
    features['flux_diff1_min'] = np.min(flux_diff1)
    features['flux_diff1_max'] = np.max(flux_diff1)
    features['flux_diff1_range'] = features['flux_diff1_max'] - features['flux_diff1_min']
    features['flux_diff1_skew'] = skew(flux_diff1)
    features['flux_diff1_kurtosis'] = kurtosis(flux_diff1)
    
    # Second derivative statistics
    features['flux_diff2_mean'] = np.mean(flux_diff2)
    features['flux_diff2_std'] = np.std(flux_diff2)
    features['flux_diff2_extremes'] = np.sum(np.abs(flux_diff2) > 3 * np.std(flux_diff2))
    
    # Change point detection
    for window in [10, 50, 100]:
        if window < len(flux_diff1):
            rolling_diff_std = pd.Series(flux_diff1).rolling(window=window).std()
            features[f'diff_volatility_w{window}_max'] = rolling_diff_std.max()
            features[f'diff_volatility_w{window}_mean'] = rolling_diff_std.mean()
    
    # Segment analysis
    segments = 4
    segment_size = len(total_flux) // segments
    segment_means = []
    
    for i in range(segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < segments - 1 else len(total_flux)
        segment_flux = total_flux[start_idx:end_idx]
        
        if len(segment_flux) > 1:
            trend = np.polyfit(np.arange(len(segment_flux)), segment_flux, 1)[0]
            features[f'segment{i}_trend'] = trend
            features[f'segment{i}_mean'] = np.mean(segment_flux)
            features[f'segment{i}_std'] = np.std(segment_flux)
            features[f'segment{i}_range'] = np.max(segment_flux) - np.min(segment_flux)
            segment_means.append(features[f'segment{i}_mean'])
    
    # Cross-segment analysis
    features['segment_mean_range'] = np.max(segment_means) - np.min(segment_means)
    features['segment_mean_std'] = np.std(segment_means)
    
    # Transit segment detection
    global_mean = np.mean(total_flux)
    global_std = np.std(total_flux)
    transit_segments = sum(1 for mean in segment_means if mean < global_mean - global_std)
    
    features['num_transit_segments'] = transit_segments
    features['transit_segment_fraction'] = transit_segments / segments
    
    return features

# Demo: Gradient & Change Features
gradient_features = extract_gradient_features(total_flux)
print(f"Generated {len(gradient_features)} gradient and change features:")

# Print ALL features organized by category
print("\nFirst Derivative Statistics:")
diff1_features = ['flux_diff1_mean', 'flux_diff1_std', 'flux_diff1_min', 'flux_diff1_max', 
                 'flux_diff1_range', 'flux_diff1_skew', 'flux_diff1_kurtosis']
for feature in diff1_features:
    print(f"  {feature}: {gradient_features[feature]:.4f}")

print("\nSecond Derivative Statistics:")
diff2_features = ['flux_diff2_mean', 'flux_diff2_std', 'flux_diff2_extremes']
for feature in diff2_features:
    print(f"  {feature}: {gradient_features[feature]:.4f}")

print("\nChange Point Detection:")
for window in [10, 50, 100]:
    max_feature = f'diff_volatility_w{window}_max'
    mean_feature = f'diff_volatility_w{window}_mean'
    if max_feature in gradient_features:
        print(f"  {max_feature}: {gradient_features[max_feature]:.4f}")
        print(f"  {mean_feature}: {gradient_features[mean_feature]:.4f}")

print("\nSegment Analysis (4 quarters):")
for i in range(4):
    print(f"  Segment {i}:")
    print(f"    trend: {gradient_features[f'segment{i}_trend']:.4f}")
    print(f"    mean: {gradient_features[f'segment{i}_mean']:.4f}")
    print(f"    std: {gradient_features[f'segment{i}_std']:.4f}")
    print(f"    range: {gradient_features[f'segment{i}_range']:.4f}")

print("\nCross-Segment Analysis:")
cross_features = ['segment_mean_range', 'segment_mean_std', 'num_transit_segments', 'transit_segment_fraction']
for feature in cross_features:
    print(f"  {feature}: {gradient_features[feature]:.4f}")


%%time
def extract_enhanced_transit_features(signal_data, verbose=True):

    n_frames = signal_data.shape[0]
    signal_data = apply_adc_correction(signal_data, instrument='FGS1')
    total_flux = np.sum(signal_data, axis=(1, 2))
    
    if verbose:
        print(f"Extracting enhanced transit features from {n_frames} frames...")
    
    # Extract features by category (add or remove feature sets here)
    features = {}
    features.update(extract_global_flux_features(total_flux))
    features.update(extract_rolling_statistics_features(total_flux))
    features.update(extract_transit_detection_features(total_flux))
    features.update(extract_frequency_features(total_flux))
    features.update(extract_spatial_features(signal_data))
    features.update(extract_gradient_features(total_flux))
    
    if verbose:
        print(f"Generated {len(features)} enhanced transit features")
    
    return features

all_features = extract_enhanced_transit_features(signal_data, verbose=True)
print(f"\nFinal feature summary:")
print(f"  Total features: {len(all_features)}")
print(f"  Feature categories: 6")
print(f"  Ready for LGBM training!")

# Show final key features
print(f"\nKey Enhanced Transit Features:")
print(f"  Global flux depth: {all_features['global_flux_depth']:.2f}")
print(f"  Global flux depth ratio: {all_features['global_flux_depth_ratio']:.4f}")
print(f"  Longest dip duration: {all_features['longest_dip_duration']} frames")
print(f"  Deepest time fraction: {all_features['deepest_time_fraction']:.3f}")
print(f"  Rolling1000 deepest dip: {all_features['rolling1000_deepest_dip']:.2f}")
print(f"  Number of transit segments: {all_features['num_transit_segments']}")


%%time

import pandas as pd
import numpy as np
import glob

def prepare_data_with_enhanced_fgs1(train_df, star_info_df, data_path):
    """Enhanced FGS1 feature extractor â€” always processes multiple signals per planet with string IDs."""

    print(f"Extracting enhanced FGS1 features for {len(star_info_df)} planets...")
    fgs1_features = []

    # Ensure string-based planet IDs throughout
    star_info_df['planet_id'] = star_info_df['planet_id'].astype(str)

    if train_df is not None:
        train_df['planet_id'] = train_df['planet_id'].astype(str)

    # Extract features from all available FGS1 signals
    for i, row in star_info_df.iterrows():
        base_id = int(float(row['planet_id']))
        print(f"\rProcessing planet {i+1}/{len(star_info_df)} (ID: {base_id})", end='', flush=True)

        try:
            signal_paths = sorted(glob.glob(f"{data_path}{base_id}/FGS1_signal_*.parquet"))

            for j, path in enumerate(signal_paths):
                df = pd.read_parquet(path)
                signal = df.values.reshape(135000, 32, 32)
                features = extract_enhanced_transit_features(signal, verbose=False)
                signal_id = f"{base_id}_{j}"
                features['planet_id'] = signal_id
                fgs1_features.append(features)

        except Exception as e:
            print(f"\nâ�Œ Failed to process planet {base_id}: {e}")
            continue

    print("\nâœ… Feature extraction complete.")

    # Build features dataframe
    features_df = pd.DataFrame(fgs1_features)
    features_df['planet_id'] = features_df['planet_id'].astype(str)
    
    features_df = features_df.set_index('planet_id')
    print(f"â†’ Extracted features for {len(features_df)} entries with {features_df.shape[1]} columns")
    
    # Expand star metadata for each signal
    expanded_meta = []

    # Normalize IDs for safe comparison
    star_info_df['planet_id'] = star_info_df['planet_id'].apply(lambda x: str(int(float(x))))
    
    for pid in features_df.index:
        base_id = str(int(float(pid.split("_")[0])))
        row = star_info_df[star_info_df['planet_id'] == base_id].copy()
        if not row.empty:
            row['planet_id'] = pid
            expanded_meta.append(row)

    star_info_df = pd.concat(expanded_meta, ignore_index=True)

    # Merge metadata and features
    full_df = star_info_df.set_index('planet_id').join(features_df, how='left')
    X = full_df.select_dtypes(include=[np.number]).fillna(0).astype(np.float32)

    if train_df is not None:
        targets_df = train_df.set_index('planet_id')
        extended_targets = []
        for pid in X.index:
            base_id = pid.split("_")[0]
            if base_id in targets_df.index:
                y_row = targets_df.loc[base_id].copy()
                y_row.name = pid
                extended_targets.append(y_row)
        y = pd.DataFrame(extended_targets).astype(np.float32)

        print(f"âœ… Final shapes â€” X: {X.shape}, y: {y.shape}")
        return X, y

    else:
        print(f"âœ… Final shape â€” X_test: {X.shape}")
        return X
                
X, y = prepare_data_with_enhanced_fgs1(train_df, train_star_info_df, "/kaggle/input/ariel-data-challenge-2025/train/")


%%time

def train_validate_multioutput_cv(X, y, n_splits=5):
    print(f"Training with CV: {X.shape[1]} features â†’ {y.shape[1]} wavelengths")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_models = [[] for _ in range(y.shape[1])]
    all_fold_metrics = []

    oof_preds = np.zeros_like(y.values, dtype=float)
    oof_sigmas = np.zeros_like(y.values, dtype=float)
    residuals_all = [[] for _ in range(y.shape[1])]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nğŸ”� Fold {fold+1}/{n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for i in tqdm(range(y.shape[1]), desc=f"Training wavelength models for Fold {fold+1}"):
            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=127,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            )

            y_single = y_train.iloc[:, i]
            model.fit(X_train, y_single)
            all_models[i].append(model)

            # Predict on validation set (OOF)
            pred_val = model.predict(X_val)
            oof_preds[val_idx, i] = pred_val

            # Store residuals for uncertainty estimation
            pred_train = model.predict(X_train)
            residuals = y_single - pred_train
            residuals_all[i].extend(residuals.tolist())

            # Temporary sigma just for oof output
            sigma = np.std(residuals)
            oof_sigmas[val_idx, i] = max(sigma, 1e-6)

    # Compute final stddevs from full OOF residuals
    target_uncertainties = [max(np.std(res), 1e-6) for res in residuals_all]

    # Compute metrics
    r2_scores = [r2_score(y.values[:, i], oof_preds[:, i]) for i in range(y.shape[1])]
    rmses = [mean_squared_error(y.values[:, i], oof_preds[:, i], squared=False) for i in range(y.shape[1])]

    print("\nğŸ“Š CV Performance Summary:")
    print(f"â†’ Mean RÂ²:   {np.mean(r2_scores):.4f}")
    print(f"â†’ Mean RMSE: {np.mean(rmses):.4f}")

    return {
        'models': all_models,
        'feature_columns': X.columns.tolist(),
        'target_columns': y.columns.tolist(),
        'oof_predictions': oof_preds,
        'oof_uncertainties': oof_sigmas,
        'y_true': y.values,
        'cv_metrics': {
            'r2_per_target': r2_scores,
            'rmse_per_target': rmses,
            'mean_r2': np.mean(r2_scores),
            'mean_rmse': np.mean(rmses)
        },
        'target_uncertainties': target_uncertainties
    }

trained_models_cv = train_validate_multioutput_cv(X, y)


def analyze_feature_importance(trained_models, X, save_file=True):
    """Analyze feature importance across all trained models (supports CV version)"""
    
    print(f"\nğŸ”� Feature Importance Analysis:")
    
    all_models_nested = trained_models['models']  # list of lists: [n_targets][n_folds]
    all_models_flat = [m for models_per_target in all_models_nested for m in models_per_target]
    
    # Calculate average feature importance across all models
    feature_importance = pd.DataFrame({
        'feature': trained_models['feature_columns'],
        'importance': np.mean([model.feature_importances_ for model in all_models_flat], axis=0)
    }).sort_values('importance', ascending=False)
    
    # Display top features
    print("Most important features:")
    for i, row in feature_importance.head(25).iterrows():
        print(f"  {i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
    
    # Feature categories analysis
    categories = {}
    for _, row in feature_importance.iterrows():
        category = row['feature'].split('_')[0] if '_' in row['feature'] else 'other'
        categories.setdefault(category, []).append(row['importance'])
    
    print(f"\nFeature Category Performance:")
    for category, importances in sorted(categories.items(), key=lambda x: np.sum(x[1]), reverse=True):
        total_contrib = np.sum(importances) / feature_importance['importance'].sum() * 100
        print(f"  {category:<15}: {len(importances):>3d} features, {total_contrib:>5.1f}% contribution")
    
    # Quick stats
    zero_features = (feature_importance['importance'] == 0).sum()
    print(f"\nQuick Stats:")
    print(f"  Zero importance features: {zero_features}")
    print(f"  Top feature: {feature_importance.iloc[0]['feature']}")
    
    # Save to file
    if save_file:
        feature_importance.to_csv('feature_importance.csv', index=False)
        print(f"\nğŸ’¾ Saved feature importance to: feature_importance.csv")
    
    return feature_importance
    
feature_importance_df = analyze_feature_importance(trained_models_cv, X)


def predict_with_uncertainty(models, X, fixed_uncertainty=None, target_uncertainties=None):
    """
    Make predictions with uncertainty estimates using trained models.

    Args:
        models: List of trained models (flat or nested by CV)
        X: Features to predict on
        fixed_uncertainty: Use this fixed value for all targets
        target_uncertainties: Optional list of std values per target (from OOF residuals)

    Returns:
        tuple: (predictions, uncertainties) as numpy arrays
    """
    predictions = []
    uncertainties = []

    is_cv = isinstance(models[0], list)
    n_targets = len(models)

    for i in range(n_targets):
        model_group = models[i] if is_cv else [models[i]]
        preds = []

        for model in model_group:
            preds.append(model.predict(X))

        # Average predictions over folds
        pred_avg = np.mean(preds, axis=0)
        predictions.append(pred_avg)

        # Use appropriate uncertainty
        if fixed_uncertainty is not None:
            unc = fixed_uncertainty
        elif target_uncertainties is not None:
            unc = target_uncertainties[i]
        else:
            unc = 0.01  # fallback

        unc_array = np.full_like(pred_avg, max(unc, 1e-6))
        uncertainties.append(unc_array)

    y_pred = np.column_stack(predictions)
    sigma_pred = np.column_stack(uncertainties)

    # Clean predictions
    pred_df = pd.DataFrame(y_pred).apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)
    sigma_df = pd.DataFrame(sigma_pred).apply(pd.to_numeric, errors='coerce').fillna(1e-6).clip(lower=1e-15)

    return pred_df.values, sigma_df.values
    


sys.path.append('/kaggle/usr/lib/ariel-gaussian-log-likelihood')
from metric import score

def evaluate_with_gll_metric(y_true, y_pred, sigma_pred, naive_mean, naive_sigma):
    """
    Evaluate predictions using the official competition GLL metric.

    Args:
        y_true: Ground truth DataFrame or array (samples x wavelengths)
        y_pred: Predictions array (samples x wavelengths)
        sigma_pred: Uncertainty estimates array (samples x wavelengths)
        naive_mean: Scalar from training set (mean)
        naive_sigma: Scalar from training set (std)

    Returns:
        float: GLL score [0, 1]
    """
    # Ensure y_true is DataFrame
    if not isinstance(y_true, pd.DataFrame):
        y_true = pd.DataFrame(y_true)

    y_true = y_true.reset_index(drop=True)
    n_samples, n_waves = y_pred.shape

    # Ensure no negative or zero sigma
    sigma_pred = np.clip(sigma_pred, 1e-15, None)

    # Build solution DataFrame
    solution_df = y_true.copy()
    solution_df['row_id'] = np.arange(n_samples)

    # Build submission DataFrame in exact format
    submission_df = pd.DataFrame()
    for i in range(n_waves):
        submission_df[f'wavelength_{i}'] = y_pred[:, i]
    for i in range(n_waves):
        submission_df[f'wavelength_{i}_std'] = sigma_pred[:, i]
    submission_df['row_id'] = np.arange(n_samples)

    # Ensure non-negative values (required)
    submission_df = submission_df.clip(lower=1e-15)

    # Call official scorer
    gll_score = score(
        solution=solution_df,
        submission=submission_df,
        row_id_column_name='row_id',
        naive_mean=naive_mean,
        naive_sigma=naive_sigma,
        fsg_sigma_true=1e-6,
        airs_sigma_true=1e-5,
        fgs_weight=0.4
    )

    return gll_score

# use OOF predictions and uncertainties collected during training
y_true = trained_models_cv['y_true']
y_pred = trained_models_cv['oof_predictions']
sigma_pred = trained_models_cv['oof_uncertainties']

# Calculate baseline stats using full data
naive_mean = y_true.mean()
naive_sigma = y_true.std()

# Evaluate with GLL metric
gll_score = evaluate_with_gll_metric(y_true, y_pred, sigma_pred, naive_mean, naive_sigma)
print(f"ğŸ�¯ OOF Gaussian Log Likelihood Score: {gll_score:.6f}")


def fast_gll_score_numpy(y_true, y_pred, sigma_pred, naive_mean, naive_sigma,
                         fsg_sigma_true=1e-6, airs_sigma_true=1e-5, fgs_weight=0.4):
    """
    Fast NumPy-based GLL score computation (no DataFrames, optimized for speed).
    """
    sigma_pred = np.clip(sigma_pred, 1e-15, None)
    n_samples, n_waves = sigma_pred.shape

    sigma_true = np.append([fsg_sigma_true], np.full(n_waves - 1, airs_sigma_true))
    sigma_true = np.tile(sigma_true, (n_samples, 1))

    weights = np.append([fgs_weight], np.ones(n_waves - 1))
    weights = np.tile(weights, (n_samples, 1))

    gll_pred = norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    gll_true = norm.logpdf(y_true, loc=y_true, scale=sigma_true)
    gll_naive = norm.logpdf(y_true, loc=naive_mean, scale=naive_sigma)

    ind_scores = (gll_pred - gll_naive) / (gll_true - gll_naive + 1e-9)
    final_score = np.average(ind_scores, weights=weights)
    return float(np.clip(final_score, 0.0, 1.0))


def optimize_sigma_per_wavelength(y_true, y_pred, sigma_pred, naive_mean, naive_sigma,
                                       fsg_sigma_true=1e-6, airs_sigma_true=1e-5, fgs_weight=0.4):
    """
    Optimize sigma scaling per wavelength using fast GLL scoring.

    Returns:
        best_scales: np.ndarray of optimal scalers (shape: n_wavelengths,)
    """
    n_waves = sigma_pred.shape[1]
    best_scales = np.ones(n_waves)

    print(f"âš¡ Optimizing {n_waves} sigma scalers with fast NumPy GLL metric...\n")

    for i in tqdm(range(n_waves), desc="Optimizing wavelengths", unit="Î»"):
        def objective(scale):
            sigma_scaled = sigma_pred.copy()
            sigma_scaled[:, i] *= scale
            return -fast_gll_score_numpy(
                y_true, y_pred, sigma_scaled,
                naive_mean, naive_sigma,
                fsg_sigma_true=fsg_sigma_true,
                airs_sigma_true=airs_sigma_true,
                fgs_weight=fgs_weight
            )

        result = minimize_scalar(objective, bounds=(0.25, 4.0), method='bounded')
        best_scales[i] = result.x

    print("âœ… Optimization complete.")
    return best_scales
    

# Optimize per-dimension scale factors
best_scales = optimize_sigma_per_wavelength(
    y_true,
    y_pred,
    sigma_pred,
    naive_mean,
    naive_sigma
)

# Final GLL score - sigma_pred modified by best_scales
final_score = evaluate_with_gll_metric(
    y_true, y_pred, sigma_pred * best_scales,
    naive_mean, naive_sigma
)
print(f"ğŸ�¯ Calibrated GLL Score: {final_score:.6f}")


def create_submission(trained_models, test_star_info_df, output_path='submission.csv', sigma_scalers=None):
    """
    Generate a submission file from trained models and test data, averaging predictions across multiple signals per planet.
    """
    # Prepare test data using your custom feature engineering
    X_test = prepare_data_with_enhanced_fgs1(
        train_df=None,
        star_info_df=test_star_info_df,
        data_path="/kaggle/input/ariel-data-challenge-2025/test/"
    )

    # Align test data columns with training features
    X_test_aligned = X_test.reindex(columns=trained_models['feature_columns'], fill_value=0)

    # Predict with uncertainty
    y_pred, sigma_pred = predict_with_uncertainty(
        trained_models['models'],
        X_test_aligned,
        target_uncertainties=trained_models['target_uncertainties']
    )

    # Apply per-wavelength scaling to sigma_pred, if provided
    if sigma_scalers is not None:
        sigma_pred = sigma_pred * sigma_scalers

    # Parse base planet IDs from signal IDs like "12345_0"
    base_ids = [idx.split("_")[0] for idx in X_test_aligned.index]

    # Convert predictions to DataFrames with signal IDs
    wl_cols = trained_models['target_columns']
    sigma_cols = [f"sigma_{i+1}" for i in range(y_pred.shape[1])]
    
    pred_df = pd.DataFrame(y_pred, index=base_ids, columns=wl_cols)
    sigma_df = pd.DataFrame(sigma_pred, index=base_ids, columns=sigma_cols)

    # Average over signals per base planet
    pred_mean = pred_df.groupby(pred_df.index).mean()
    sigma_mean = sigma_df.groupby(sigma_df.index).mean()

    # Build submission DataFrame
    submission_df = pd.concat([pred_mean, sigma_mean], axis=1).reset_index()
    submission_df = submission_df.rename(columns={'index': 'planet_id'})
    submission_df['planet_id'] = submission_df['planet_id'].astype(int)

    # Save to file
    submission_df.to_csv(output_path, index=False, float_format='%.5f')
    print(f"âœ… Submission saved to: {output_path}")

    return submission_df
    
submission = create_submission(
    trained_models_cv,
    test_star_info_df,
    output_path='submission.csv',
    sigma_scalers=best_scales 
)


submission

