import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from scipy.spatial.transform import Rotation as R
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder
import joblib

# Configuration
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")  # Update this to your data directory
OUTPUT_DIR = Path("./features")
OUTPUT_DIR.mkdir(exist_ok=True)

print("ğŸ”§ BFRB Feature Extraction Pipeline for Tree-based Models")

#=============================================================================
# Core Feature Engineering Functions
#=============================================================================

def remove_gravity_from_acc(acc_data, rot_data):
    """Remove gravity component from accelerometer data"""
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data
    
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    
    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    
    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]
    
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    """Calculate angular velocity from quaternion data"""
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    
    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]
        
        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue
        
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    
    return angular_vel

def calculate_angular_distance(rot_data):
    """Calculate angular distance between consecutive quaternions"""
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    
    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)
    
    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]
        
        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
    
    return angular_dist

#=============================================================================
# Statistical Feature Extraction Functions
#=============================================================================

def calculate_statistical_features(series, prefix):
    """Calculate comprehensive statistical features for a time series"""
    features = {}
    
    # Basic statistics
    features[f'{prefix}_mean'] = series.mean()
    features[f'{prefix}_std'] = series.std()
    features[f'{prefix}_min'] = series.min()
    features[f'{prefix}_max'] = series.max()
    features[f'{prefix}_median'] = series.median()
    features[f'{prefix}_range'] = series.max() - series.min()
    features[f'{prefix}_iqr'] = series.quantile(0.75) - series.quantile(0.25)
    
    # Percentiles
    for q in [10, 25, 75, 90]:
        features[f'{prefix}_q{q}'] = series.quantile(q/100)
    
    # Shape statistics
    features[f'{prefix}_skew'] = skew(series.dropna())
    features[f'{prefix}_kurtosis'] = kurtosis(series.dropna())
    
    # Energy and power
    features[f'{prefix}_energy'] = np.sum(series**2)
    features[f'{prefix}_power'] = np.mean(series**2)
    features[f'{prefix}_rms'] = np.sqrt(np.mean(series**2))
    
    # Zero crossing rate
    zero_crossings = np.diff(np.signbit(series.fillna(0))).sum()
    features[f'{prefix}_zero_crossings'] = zero_crossings
    
    # Peak detection
    try:
        peaks, _ = find_peaks(series.fillna(0))
        features[f'{prefix}_num_peaks'] = len(peaks)
        features[f'{prefix}_peak_density'] = len(peaks) / len(series) if len(series) > 0 else 0
    except:
        features[f'{prefix}_num_peaks'] = 0
        features[f'{prefix}_peak_density'] = 0
    
    # Trend features
    if len(series) > 1:
        x = np.arange(len(series))
        slope = np.polyfit(x, series.fillna(method='ffill').fillna(0), 1)[0]
        features[f'{prefix}_trend_slope'] = slope
    else:
        features[f'{prefix}_trend_slope'] = 0
    
    # Variability features
    features[f'{prefix}_cv'] = series.std() / abs(series.mean()) if series.mean() != 0 else 0
    features[f'{prefix}_mad'] = (series - series.mean()).abs().mean()  # Mean absolute deviation
    
    return features

def calculate_derivative_features(series, prefix):
    """Calculate features based on derivatives (velocity, acceleration)"""
    features = {}
    
    # First derivative (velocity)
    velocity = series.diff().fillna(0)
    features.update(calculate_statistical_features(velocity, f'{prefix}_vel'))
    
    # Second derivative (acceleration) 
    acceleration = velocity.diff().fillna(0)
    features.update(calculate_statistical_features(acceleration, f'{prefix}_accel'))
    
    # Jerk (third derivative)
    jerk = acceleration.diff().fillna(0)
    features[f'{prefix}_jerk_mean'] = jerk.mean()
    features[f'{prefix}_jerk_std'] = jerk.std()
    features[f'{prefix}_jerk_max'] = jerk.max()
    
    return features

# def calculate_frequency_features(series, prefix, sampling_rate=200):
#     """Calculate frequency domain features using FFT"""
#     features = {}
    
#     try:
#         # Remove DC component and apply window
#         signal = series.fillna(0) - series.mean()
        
#         if len(signal) > 1:
#             # FFT
#             fft = np.fft.fft(signal)
#             freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
            
#             # Power spectral density
#             psd = np.abs(fft)**2
            
#             # Frequency domain statistics
#             features[f'{prefix}_spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
#             features[f'{prefix}_spectral_rolloff'] = freqs[np.where(np.cumsum(psd[:len(psd)//2]) >= 0.85 * np.sum(psd[:len(psd)//2]))[0][0]]
#             features[f'{prefix}_spectral_bandwidth'] = np.sqrt(np.sum(((freqs[:len(freqs)//2] - features[f'{prefix}_spectral_centroid'])**2) * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2]))
            
#             # Dominant frequency
#             dominant_freq_idx = np.argmax(psd[:len(psd)//2])
#             features[f'{prefix}_dominant_freq'] = freqs[dominant_freq_idx]
#             features[f'{prefix}_dominant_freq_power'] = psd[dominant_freq_idx]
            
#             # Energy in frequency bands
#             low_freq_mask = (freqs >= 0) & (freqs <= 5)
#             mid_freq_mask = (freqs > 5) & (freqs <= 15)
#             high_freq_mask = (freqs > 15) & (freqs <= 50)
            
#             total_energy = np.sum(psd[:len(psd)//2])
#             features[f'{prefix}_low_freq_energy'] = np.sum(psd[low_freq_mask]) / total_energy if total_energy > 0 else 0
#             features[f'{prefix}_mid_freq_energy'] = np.sum(psd[mid_freq_mask]) / total_energy if total_energy > 0 else 0
#             features[f'{prefix}_high_freq_energy'] = np.sum(psd[high_freq_mask]) / total_energy if total_energy > 0 else 0
        
#         else:
#             # Default values for short sequences
#             for feature_name in ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 
#                                'dominant_freq', 'dominant_freq_power', 'low_freq_energy', 
#                                'mid_freq_energy', 'high_freq_energy']:
#                 features[f'{prefix}_{feature_name}'] = 0
                
#     except Exception as e:
#         print(f"Frequency feature calculation failed for {prefix}: {e}")
#         for feature_name in ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 
#                            'dominant_freq', 'dominant_freq_power', 'low_freq_energy', 
#                            'mid_freq_energy', 'high_freq_energy']:
#             features[f'{prefix}_{feature_name}'] = 0
    
#     return features


def calculate_frequency_features(series, prefix, sampling_rate=200):
    """Calculate frequency domain features using FFT"""
    features = {}
    
    try:
        # Remove NaN values first - CRITICAL FIX
        series_clean = series.dropna()
        if len(series_clean) < 2:  # Need at least 2 points for FFT
            raise ValueError("Not enough data points after removing NaN")
        
        # Remove DC component and apply window
        signal = series_clean - series_clean.mean()
        
        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        
        # Power spectral density
        psd = np.abs(fft)**2
        
        # Only use positive frequencies (first half)
        positive_freq_mask = freqs >= 0
        positive_freqs = freqs[positive_freq_mask]
        positive_psd = psd[positive_freq_mask]
        
        if len(positive_freqs) == 0 or np.sum(positive_psd) == 0:
            raise ValueError("No valid frequency components")
        
        # Frequency domain statistics
        features[f'{prefix}_spectral_centroid'] = np.sum(positive_freqs * positive_psd) / np.sum(positive_psd)
        
        # Spectral rolloff (85%)
        cumulative_energy = np.cumsum(positive_psd)
        rolloff_index = np.where(cumulative_energy >= 0.85 * np.sum(positive_psd))[0]
        features[f'{prefix}_spectral_rolloff'] = positive_freqs[rolloff_index[0]] if len(rolloff_index) > 0 else 0
        
        # Spectral bandwidth
        features[f'{prefix}_spectral_bandwidth'] = np.sqrt(
            np.sum(((positive_freqs - features[f'{prefix}_spectral_centroid'])**2) * positive_psd) / np.sum(positive_psd)
        )
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(positive_psd)
        features[f'{prefix}_dominant_freq'] = positive_freqs[dominant_freq_idx]
        features[f'{prefix}_dominant_freq_power'] = positive_psd[dominant_freq_idx]
        
        # Energy in frequency bands
        low_freq_mask = (positive_freqs >= 0) & (positive_freqs <= 5)
        mid_freq_mask = (positive_freqs > 5) & (positive_freqs <= 15)
        high_freq_mask = (positive_freqs > 15) & (positive_freqs <= 50)
        
        total_energy = np.sum(positive_psd)
        features[f'{prefix}_low_freq_energy'] = np.sum(positive_psd[low_freq_mask]) / total_energy if total_energy > 0 else 0
        features[f'{prefix}_mid_freq_energy'] = np.sum(positive_psd[mid_freq_mask]) / total_energy if total_energy > 0 else 0
        features[f'{prefix}_high_freq_energy'] = np.sum(positive_psd[high_freq_mask]) / total_energy if total_energy > 0 else 0
        
    except Exception as e:
        # Set all frequency features to 0 when calculation fails
        for feature_name in ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 
                           'dominant_freq', 'dominant_freq_power', 'low_freq_energy', 
                           'mid_freq_energy', 'high_freq_energy']:
            features[f'{prefix}_{feature_name}'] = 0
    
    return features


def calculate_cross_signal_features(signal1, signal2, prefix1, prefix2):
    """Calculate features between two signals (correlations, etc.)"""
    features = {}
    
    # Cross-correlation
    if len(signal1) > 1 and len(signal2) > 1:
        correlation = np.corrcoef(signal1.fillna(0), signal2.fillna(0))[0, 1]
        features[f'{prefix1}_{prefix2}_correlation'] = correlation if not np.isnan(correlation) else 0
        
        # Cross-correlation lag features
        cross_corr = np.correlate(signal1.fillna(0), signal2.fillna(0), mode='full')
        max_corr_idx = np.argmax(np.abs(cross_corr))
        lag = max_corr_idx - len(signal1) + 1
        features[f'{prefix1}_{prefix2}_max_cross_corr'] = cross_corr[max_corr_idx]
        features[f'{prefix1}_{prefix2}_lag'] = lag
        
        # Coherence (simplified as correlation of power)
        power1 = signal1**2
        power2 = signal2**2
        power_corr = np.corrcoef(power1.fillna(0), power2.fillna(0))[0, 1]
        features[f'{prefix1}_{prefix2}_coherence'] = power_corr if not np.isnan(power_corr) else 0
    else:
        features[f'{prefix1}_{prefix2}_correlation'] = 0
        features[f'{prefix1}_{prefix2}_max_cross_corr'] = 0
        features[f'{prefix1}_{prefix2}_lag'] = 0
        features[f'{prefix1}_{prefix2}_coherence'] = 0
    
    return features

#=============================================================================
# Comprehensive Feature Extraction for Single Sequence
#=============================================================================

def extract_sequence_features(df_seq):
    """Extract comprehensive features from a single sequence"""
    features = {}
    
    # Basic sequence metadata
    features['sequence_length'] = len(df_seq)
    features['sequence_duration'] = features['sequence_length'] / 200  # Assuming 200Hz
    
    # Apply core feature engineering first
    df_seq = df_seq.copy()
    
    # Remove gravity from accelerometer data
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'] = linear_accel[:, 0]
    df_seq['linear_acc_y'] = linear_accel[:, 1] 
    df_seq['linear_acc_z'] = linear_accel[:, 2]
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + 
                                      df_seq['linear_acc_y']**2 + 
                                      df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
    
    # Calculate angular velocity
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'] = angular_vel[:, 0]
    df_seq['angular_vel_y'] = angular_vel[:, 1]
    df_seq['angular_vel_z'] = angular_vel[:, 2]
    df_seq['angular_vel_mag'] = np.sqrt(angular_vel[:, 0]**2 + angular_vel[:, 1]**2 + angular_vel[:, 2]**2)
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)
    
    # Original accelerometer magnitude
    df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
    
    # ===== IMU FEATURES =====
    imu_signals = {
        'acc_x': df_seq['acc_x'],
        'acc_y': df_seq['acc_y'], 
        'acc_z': df_seq['acc_z'],
        'acc_mag': df_seq['acc_mag'],
        'linear_acc_x': df_seq['linear_acc_x'],
        'linear_acc_y': df_seq['linear_acc_y'],
        'linear_acc_z': df_seq['linear_acc_z'],
        'linear_acc_mag': df_seq['linear_acc_mag'],
        'angular_vel_x': df_seq['angular_vel_x'],
        'angular_vel_y': df_seq['angular_vel_y'],
        'angular_vel_z': df_seq['angular_vel_z'],
        'angular_vel_mag': df_seq['angular_vel_mag'],
        'angular_distance': df_seq['angular_distance'],
    }
    
    # Extract statistical, derivative, and frequency features for each IMU signal
    for signal_name, signal_data in imu_signals.items():
        features.update(calculate_statistical_features(signal_data, signal_name))
        features.update(calculate_derivative_features(signal_data, signal_name))
        features.update(calculate_frequency_features(signal_data, signal_name))
    
    # ===== THERMAL FEATURES =====
    for i in range(1, 6):
        thm_col = f'thm_{i}'
        if thm_col in df_seq.columns:
            features.update(calculate_statistical_features(df_seq[thm_col], thm_col))
            features.update(calculate_derivative_features(df_seq[thm_col], thm_col))
            features.update(calculate_frequency_features(df_seq[thm_col], thm_col))
    
    # ===== TIME-OF-FLIGHT FEATURES =====
    for i in range(1, 6):
        # Individual pixel features (aggregated)
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        if all(col in df_seq.columns for col in pixel_cols):
            tof_data = df_seq[pixel_cols].replace(-1, np.nan)
            
            # Spatial aggregations over pixels at each timestep
            tof_mean_series = tof_data.mean(axis=1)
            tof_std_series = tof_data.std(axis=1)
            tof_min_series = tof_data.min(axis=1)
            tof_max_series = tof_data.max(axis=1)
            tof_range_series = tof_max_series - tof_min_series
            
            # Temporal features of spatial aggregations
            for agg_name, agg_series in [
                ('mean', tof_mean_series), ('std', tof_std_series),
                ('min', tof_min_series), ('max', tof_max_series), ('range', tof_range_series)
            ]:
                prefix = f'tof_{i}_{agg_name}'
                features.update(calculate_statistical_features(agg_series, prefix))
                features.update(calculate_derivative_features(agg_series, prefix))
            
            # Overall ToF sensor statistics
            all_tof_values = tof_data.values.flatten()
            all_tof_values = all_tof_values[~np.isnan(all_tof_values)]
            if len(all_tof_values) > 0:
                features[f'tof_{i}_overall_mean'] = np.mean(all_tof_values)
                features[f'tof_{i}_overall_std'] = np.std(all_tof_values)
                features[f'tof_{i}_overall_min'] = np.min(all_tof_values)
                features[f'tof_{i}_overall_max'] = np.max(all_tof_values)
                features[f'tof_{i}_coverage_ratio'] = len(all_tof_values) / (64 * len(df_seq))
            else:
                features[f'tof_{i}_overall_mean'] = 0
                features[f'tof_{i}_overall_std'] = 0
                features[f'tof_{i}_overall_min'] = 0
                features[f'tof_{i}_overall_max'] = 0
                features[f'tof_{i}_coverage_ratio'] = 0
    
    # ===== CROSS-SIGNAL FEATURES =====
    # Accelerometer cross-correlations
    features.update(calculate_cross_signal_features(df_seq['acc_x'], df_seq['acc_y'], 'acc_x', 'acc_y'))
    features.update(calculate_cross_signal_features(df_seq['acc_x'], df_seq['acc_z'], 'acc_x', 'acc_z'))
    features.update(calculate_cross_signal_features(df_seq['acc_y'], df_seq['acc_z'], 'acc_y', 'acc_z'))
    
    # Linear acceleration cross-correlations
    features.update(calculate_cross_signal_features(df_seq['linear_acc_x'], df_seq['linear_acc_y'], 'lin_acc_x', 'lin_acc_y'))
    features.update(calculate_cross_signal_features(df_seq['linear_acc_x'], df_seq['linear_acc_z'], 'lin_acc_x', 'lin_acc_z'))
    features.update(calculate_cross_signal_features(df_seq['linear_acc_y'], df_seq['linear_acc_z'], 'lin_acc_y', 'lin_acc_z'))
    
    # Angular velocity cross-correlations
    features.update(calculate_cross_signal_features(df_seq['angular_vel_x'], df_seq['angular_vel_y'], 'ang_vel_x', 'ang_vel_y'))
    features.update(calculate_cross_signal_features(df_seq['angular_vel_x'], df_seq['angular_vel_z'], 'ang_vel_x', 'ang_vel_z'))
    features.update(calculate_cross_signal_features(df_seq['angular_vel_y'], df_seq['angular_vel_z'], 'ang_vel_y', 'ang_vel_z'))
    
    # ===== BEHAVIOR-BASED FEATURES =====
    if 'behavior' in df_seq.columns:
        behavior_counts = df_seq['behavior'].value_counts()
        for behavior in ['Transition', 'Pause', 'Gesture']:
            features[f'behavior_{behavior}_count'] = behavior_counts.get(behavior, 0)
            features[f'behavior_{behavior}_ratio'] = behavior_counts.get(behavior, 0) / len(df_seq)
        
        # Behavior transitions
        behavior_changes = (df_seq['behavior'] != df_seq['behavior'].shift()).sum()
        features['behavior_transitions'] = behavior_changes
    
    return features

#=============================================================================
# Main Feature Extraction Pipeline
#=============================================================================

def extract_features_from_dataset(csv_path, output_path, demographics_path=None):
    """Extract features from entire dataset and save as DataFrame"""
    print(f"Loading data from {csv_path}")
    
    # Load data in chunks to manage memory
    chunk_size = 10000
    all_features = []
    
    # Get unique sequence IDs first
    sequence_ids = pd.read_csv(csv_path, usecols=['sequence_id'])['sequence_id'].unique()
    total_sequences = len(sequence_ids)
    
    print(f"Found {total_sequences} unique sequences")
    
    # Process sequences in batches
    batch_size = 100
    for batch_start in range(0, total_sequences, batch_size):
        batch_end = min(batch_start + batch_size, total_sequences)
        batch_seq_ids = sequence_ids[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_sequences-1)//batch_size + 1} "
              f"(sequences {batch_start+1}-{batch_end})")
        
        # Load data for this batch
        batch_data = pd.read_csv(csv_path)
        batch_data = batch_data[batch_data['sequence_id'].isin(batch_seq_ids)]
        
        batch_features = []
        for seq_id in batch_seq_ids:
            try:
                seq_data = batch_data[batch_data['sequence_id'] == seq_id].copy()
                
                if len(seq_data) == 0:
                    continue
                
                # Extract features
                seq_features = extract_sequence_features(seq_data)
                seq_features['sequence_id'] = seq_id
                
                # Add metadata
                if 'subject' in seq_data.columns:
                    seq_features['subject'] = seq_data['subject'].iloc[0]
                if 'gesture' in seq_data.columns:
                    seq_features['gesture'] = seq_data['gesture'].iloc[0]
                if 'orientation' in seq_data.columns:
                    seq_features['orientation'] = seq_data['orientation'].iloc[0]
                if 'sequence_type' in seq_data.columns:
                    seq_features['sequence_type'] = seq_data['sequence_type'].iloc[0]
                
                batch_features.append(seq_features)
                
            except Exception as e:
                print(f"Error processing sequence {seq_id}: {e}")
                continue
        
        all_features.extend(batch_features)
        
        # Free memory
        del batch_data
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(all_features)
    
    # Add demographic features if available
    if demographics_path and Path(demographics_path).exists():
        print("Adding demographic features...")
        demographics = pd.read_csv(demographics_path)
        feature_df = feature_df.merge(demographics, on='subject', how='left')
    
    # Save features
    print(f"Saving {len(feature_df)} feature vectors with {len(feature_df.columns)} features to {output_path}")
    feature_df.to_csv(output_path, index=False)
    
    # Save feature names for later use
    feature_names_path = output_path.parent / f"{output_path.stem}_feature_names.txt"
    feature_columns = [col for col in feature_df.columns 
                      if col not in ['sequence_id', 'subject', 'gesture', 'orientation', 'sequence_type']]
    
    with open(feature_names_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    
    print(f"Feature names saved to {feature_names_path}")
    print(f"Feature extraction completed! Shape: {feature_df.shape}")
    
    return feature_df

def create_tree_model_ready_dataset():
    """Create train/test datasets ready for tree-based models"""
    print("ğŸŒ³ Creating Tree Model Ready Dataset")
    
    # Extract training features
    train_features = extract_features_from_dataset(
        csv_path=RAW_DIR / "train.csv",
        output_path=OUTPUT_DIR / "train_features.csv",
        demographics_path=RAW_DIR / "train_demographics.csv"
    )
    
    # Extract test features (if available)
    test_path = RAW_DIR / "test.csv"
    if test_path.exists():
        test_features = extract_features_from_dataset(
            csv_path=test_path,
            output_path=OUTPUT_DIR / "test_features.csv",
            demographics_path=RAW_DIR / "test_demographics.csv"
        )
    
    # Create summary report
    with open(OUTPUT_DIR / "feature_extraction_report.txt", 'w') as f:
        f.write("BFRB Feature Extraction Report\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Training Features: {train_features.shape}\n")
        if 'test_features' in locals():
            f.write(f"Test Features: {test_features.shape}\n")
        
        f.write(f"\nFeature Categories:\n")
        f.write("- Statistical features (mean, std, min, max, percentiles, skew, kurtosis)\n")
        f.write("- Derivative features (velocity, acceleration, jerk)\n")
        f.write("- Frequency domain features (spectral centroid, rolloff, energy bands)\n")
        f.write("- Cross-signal features (correlations, coherence)\n")
        f.write("- Engineered features (linear acceleration, angular velocity)\n")
        f.write("- Sensor-specific features (ToF spatial aggregations, thermal)\n")
        f.write("- Behavioral features (transition counts, phase ratios)\n")
        
        if 'gesture' in train_features.columns:
            f.write(f"\nGesture Distribution:\n")
            gesture_counts = train_features['gesture'].value_counts()
            for gesture, count in gesture_counts.items():
                f.write(f"  {gesture}: {count}\n")
    
    print(f"\nâœ… Feature extraction completed!")
    print(f"ğŸ“� Files saved in {OUTPUT_DIR}/")
    print(f"ğŸ“Š Training features: train_features.csv ({train_features.shape})")
    if 'test_features' in locals():
        print(f"ğŸ“Š Test features: test_features.csv ({test_features.shape})")
    print(f"ğŸ“‹ Feature names: train_features_feature_names.txt")
    print(f"ğŸ“„ Report: feature_extraction_report.txt")
    
    return train_features



# Run the feature extraction pipeline
feature_df = create_tree_model_ready_dataset()

print("\nğŸš€ Ready for tree-based models!")
print("You can now use these features with:")
print("- XGBoost: xgb.XGBClassifier()")
print("- LightGBM: lgb.LGBMClassifier()")  
print("- CatBoost: cb.CatBoostClassifier()")
print("- Random Forest: sklearn.ensemble.RandomForestClassifier()")
print("\nExample usage:")
print("""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load features
df = pd.read_csv('./features/train_features.csv')

# Prepare data
feature_cols = [col for col in df.columns 
               if col not in ['sequence_id', 'subject', 'gesture', 'orientation', 'sequence_type']]
X = df[feature_cols]
y = df['gesture']

# Train model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
""")




