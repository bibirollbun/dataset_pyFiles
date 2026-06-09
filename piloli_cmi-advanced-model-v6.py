import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import gc
import sys
import warnings
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore")

# Add path for inference server
sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")

print("Advanced libraries imported successfully")


# Import inference server
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

print("CMI Inference Server imported")


def remove_gravity_from_acc(df_seq, alpha=0.8):
    """é‡�åŠ›é™¤å�»åŠ é€Ÿåº¦ã�®è¨ˆç®—ï¼ˆãƒˆãƒƒãƒ—è§£æ³•ã�®å¿…é ˆè¦�ç´ ï¼‰"""
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    acc_data = df_seq[acc_cols].values

    # Low-pass filter for gravity estimation
    gravity = np.zeros_like(acc_data)
    gravity[0] = acc_data[0]

    for i in range(1, len(acc_data)):
        gravity[i] = alpha * gravity[i - 1] + (1 - alpha) * acc_data[i]

    # Linear acceleration = total - gravity
    linear_acc = acc_data - gravity
    return linear_acc, gravity


def calculate_angular_velocity_from_quat(df_seq):
    """ã‚¯ã‚©ãƒ¼ã‚¿ãƒ‹ã‚ªãƒ³ã�‹ã‚‰è§’é€Ÿåº¦ã‚’è¨ˆç®—ï¼ˆ0.852ã‚¹ã‚³ã‚¢é�”æˆ�æ‰‹æ³•ï¼‰"""
    quat_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    quat = df_seq[quat_cols].values

    # Quaternion difference with proper normalization
    quat_norm = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-8)
    quat_diff = np.diff(quat_norm, axis=0)

    # Angular velocity approximation with improved calculation
    angular_vel = np.zeros((len(df_seq), 3))
    if len(quat_diff) > 0:
        angular_vel[1:, 0] = 2 * quat_diff[:, 0]
        angular_vel[1:, 1] = 2 * quat_diff[:, 1]
        angular_vel[1:, 2] = 2 * quat_diff[:, 2]

    return angular_vel


def calculate_jerk_and_snap(df_seq):
    """ã‚¸ãƒ£ãƒ¼ã‚¯ï¼ˆåŠ é€Ÿåº¦å¤‰åŒ–ç�‡ï¼‰ã�¨ã‚¹ãƒŠãƒƒãƒ—ï¼ˆã‚¸ãƒ£ãƒ¼ã‚¯å¤‰åŒ–ç�‡ï¼‰ã�®è¨ˆç®—"""
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    acc_data = df_seq[acc_cols].values

    # Jerk = derivative of acceleration
    jerk = np.zeros_like(acc_data)
    if len(acc_data) > 1:
        jerk[1:] = np.diff(acc_data, axis=0)

    # Snap = derivative of jerk
    snap = np.zeros_like(acc_data)
    if len(jerk) > 1:
        snap[1:] = np.diff(jerk, axis=0)

    return jerk, snap


def extract_frequency_features(signal_data, sampling_rate=50):
    """å‘¨æ³¢æ•°é ˜åŸŸç‰¹å¾´é‡�ã�®æŠ½å‡ºï¼ˆFFTã€�ãƒ‘ãƒ¯ãƒ¼ã‚¹ãƒšã‚¯ãƒˆãƒ«å¯†åº¦ï¼‰"""
    features = {}

    if len(signal_data) < 4:
        return {
            f"freq_{key}": 0
            for key in [
                "dominant",
                "power_total",
                "spectral_entropy",
                "spectral_centroid",
            ]
        }

    try:
        # FFT
        fft_values = np.abs(fft(signal_data))
        freqs = fftfreq(len(signal_data), 1 / sampling_rate)

        # Power spectral density
        psd = fft_values**2

        # Dominant frequency
        dominant_freq_idx = np.argmax(psd[1 : len(psd) // 2]) + 1
        features["freq_dominant"] = abs(freqs[dominant_freq_idx])

        # Total spectral power
        features["freq_power_total"] = np.sum(psd)

        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-8)
        features["freq_spectral_entropy"] = -np.sum(psd_norm * np.log(psd_norm + 1e-8))

        # Spectral centroid
        features["freq_spectral_centroid"] = np.sum(freqs[: len(psd)] * psd) / (
            np.sum(psd) + 1e-8
        )

    except Exception as e:
        # Fallback values
        features = {
            "freq_dominant": 0,
            "freq_power_total": 0,
            "freq_spectral_entropy": 0,
            "freq_spectral_centroid": 0,
        }

    return features


def extract_statistical_features(signal_data):
    """é«˜åº¦ã�ªçµ±è¨ˆçš„ç‰¹å¾´é‡�ï¼ˆæ­ªåº¦ã€�å°–åº¦ã€�ã‚¨ãƒ³ãƒˆãƒ­ãƒ”ãƒ¼ï¼‰"""
    features = {}

    if len(signal_data) == 0:
        return {
            f"stat_{key}": 0 for key in ["skewness", "kurtosis", "entropy", "iqr", "rms"]
        }

    try:
        # åŸºæœ¬çµ±è¨ˆé‡�
        features["stat_skewness"] = stats.skew(signal_data)
        features["stat_kurtosis"] = stats.kurtosis(signal_data)

        # ã‚¨ãƒ³ãƒˆãƒ­ãƒ”ãƒ¼
        hist, _ = np.histogram(signal_data, bins=10)
        hist_norm = hist / (np.sum(hist) + 1e-8)
        features["stat_entropy"] = -np.sum(hist_norm * np.log(hist_norm + 1e-8))

        # å››åˆ†ä½�ç¯„å›²
        features["stat_iqr"] = np.percentile(signal_data, 75) - np.percentile(
            signal_data, 25
        )

        # RMS (Root Mean Square)
        features["stat_rms"] = np.sqrt(np.mean(signal_data**2))

    except Exception as e:
        # Fallback values
        features = {
            "stat_skewness": 0,
            "stat_kurtosis": 0,
            "stat_entropy": 0,
            "stat_iqr": 0,
            "stat_rms": 0,
        }

    return features


def extract_time_series_features(signal_data, max_lag=10):
    """æ™‚ç³»åˆ—ç‰¹å¾´é‡�ï¼ˆè‡ªå·±ç›¸é–¢ã€�ãƒˆãƒ¬ãƒ³ãƒ‰ï¼‰"""
    features = {}

    if len(signal_data) < max_lag + 1:
        return {
            f"ts_{key}": 0 for key in ["autocorr_1", "autocorr_5", "trend", "seasonality"]
        }

    try:
        # Auto-correlation at lag 1 and 5
        features["ts_autocorr_1"] = (
            np.corrcoef(signal_data[:-1], signal_data[1:])[0, 1]
            if len(signal_data) > 1
            else 0
        )
        if len(signal_data) > 5:
            features["ts_autocorr_5"] = np.corrcoef(signal_data[:-5], signal_data[5:])[
                0, 1
            ]
        else:
            features["ts_autocorr_5"] = 0

        # Linear trend (slope of linear regression)
        x = np.arange(len(signal_data))
        slope, _, _, _, _ = stats.linregress(x, signal_data)
        features["ts_trend"] = slope

        # Seasonality detection (simplified)
        if len(signal_data) > 20:
            # Simple seasonality measure using autocorrelation at period/4
            period = min(len(signal_data) // 4, 10)
            if period > 0:
                features["ts_seasonality"] = np.corrcoef(
                    signal_data[:-period], signal_data[period:]
                )[0, 1]
            else:
                features["ts_seasonality"] = 0
        else:
            features["ts_seasonality"] = 0

    except Exception as e:
        features = {
            "ts_autocorr_1": 0,
            "ts_autocorr_5": 0,
            "ts_trend": 0,
            "ts_seasonality": 0,
        }

    return features


def extract_advanced_tof_features(df_seq):
    """ToFã‚»ãƒ³ã‚µãƒ¼ã�®é«˜åº¦ã�ªç‰¹å¾´é‡�æŠ½å‡º"""
    tof_features = {}

    for sensor_id in range(1, 6):
        pixel_cols = [f"tof_{sensor_id}_v{p}" for p in range(64)]
        if all(col in df_seq.columns for col in pixel_cols):
            tof_data = df_seq[pixel_cols].replace(-1, np.nan)

            # åŸºæœ¬çµ±è¨ˆé‡�
            tof_features[f"tof_{sensor_id}_mean"] = tof_data.mean(axis=1).mean()
            tof_features[f"tof_{sensor_id}_std"] = tof_data.std(axis=1).mean()
            tof_features[f"tof_{sensor_id}_min"] = tof_data.min(axis=1).mean()
            tof_features[f"tof_{sensor_id}_max"] = tof_data.max(axis=1).mean()

            # é«˜åº¦ã�ªçµ±è¨ˆé‡�
            tof_values = tof_data.values.flatten()
            tof_values = tof_values[~np.isnan(tof_values)]

            if len(tof_values) > 0:
                tof_features[f"tof_{sensor_id}_skew"] = stats.skew(tof_values)
                tof_features[f"tof_{sensor_id}_kurtosis"] = stats.kurtosis(tof_values)
                tof_features[f"tof_{sensor_id}_range"] = np.ptp(tof_values)

                # ç©ºé–“çš„ç‰¹å¾´ï¼ˆ8x8ã‚°ãƒªãƒƒãƒ‰ã�®ç‰¹æ€§ï¼‰
                tof_matrix = tof_data.mean(axis=0).values.reshape(8, 8)
                if not np.all(np.isnan(tof_matrix)):
                    # ä¸­å¿ƒã�¨ç«¯ã�®å·®
                    center_val = np.nanmean(tof_matrix[3:5, 3:5])
                    edge_val = np.nanmean(
                        [
                            tof_matrix[0, :],
                            tof_matrix[-1, :],
                            tof_matrix[:, 0],
                            tof_matrix[:, -1],
                        ]
                    )
                    tof_features[f"tof_{sensor_id}_center_edge_diff"] = (
                        center_val - edge_val
                    )
                else:
                    tof_features[f"tof_{sensor_id}_center_edge_diff"] = 0
            else:
                tof_features[f"tof_{sensor_id}_skew"] = 0
                tof_features[f"tof_{sensor_id}_kurtosis"] = 0
                tof_features[f"tof_{sensor_id}_range"] = 0
                tof_features[f"tof_{sensor_id}_center_edge_diff"] = 0
        else:
            # Default values when sensor data is missing
            for suffix in [
                "mean",
                "std",
                "min",
                "max",
                "skew",
                "kurtosis",
                "range",
                "center_edge_diff",
            ]:
                tof_features[f"tof_{sensor_id}_{suffix}"] = 0

    return tof_features


def extract_sensor_interaction_features(df_seq):
    """ã‚»ãƒ³ã‚µãƒ¼é–“ã�®ç›¸äº’ä½œç”¨ç‰¹å¾´é‡�"""
    features = {}

    # åŠ é€Ÿåº¦æˆ�åˆ†é–“ã�®ç›¸é–¢
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    if all(col in df_seq.columns for col in acc_cols):
        acc_data = df_seq[acc_cols]

        try:
            corr_matrix = acc_data.corr()
            features["acc_xy_corr"] = corr_matrix.loc["acc_x", "acc_y"]
            features["acc_xz_corr"] = corr_matrix.loc["acc_x", "acc_z"]
            features["acc_yz_corr"] = corr_matrix.loc["acc_y", "acc_z"]
        except (KeyError, ValueError):
            features["acc_xy_corr"] = 0
            features["acc_xz_corr"] = 0
            features["acc_yz_corr"] = 0

    # å›�è»¢ã�¨åŠ é€Ÿåº¦ã�®é–¢ä¿‚
    rot_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    if all(col in df_seq.columns for col in acc_cols + rot_cols):
        try:
            acc_mag = np.sqrt(
                df_seq["acc_x"] ** 2 + df_seq["acc_y"] ** 2 + df_seq["acc_z"] ** 2
            )
            rot_mag = np.sqrt(
                df_seq["rot_x"] ** 2 + df_seq["rot_y"] ** 2 + df_seq["rot_z"] ** 2
            )

            features["acc_rot_corr"] = (
                np.corrcoef(acc_mag, rot_mag)[0, 1] if len(acc_mag) > 1 else 0
            )
        except (ValueError, IndexError):
            features["acc_rot_corr"] = 0

    # æ¸©åº¦ã‚»ãƒ³ã‚µãƒ¼é–“ã�®é–¢ä¿‚
    temp_cols = [f"thm_{i}" for i in range(1, 6)]
    available_temp_cols = [col for col in temp_cols if col in df_seq.columns]

    if len(available_temp_cols) > 1:
        temp_data = df_seq[available_temp_cols]
        features["temp_range"] = temp_data.max().max() - temp_data.min().min()
        features["temp_var"] = temp_data.var(axis=1).mean()
    else:
        features["temp_range"] = 0
        features["temp_var"] = 0

    return features


def create_advanced_physics_features(df_seq):
    """åŒ…æ‹¬çš„ã�ªé«˜åº¦ç‰¹å¾´é‡�ç”Ÿæˆ�ï¼ˆ87%ç²¾åº¦é�”æˆ�ã�®ã�Ÿã‚�ã�®çµ±å�ˆç‰¹å¾´é‡�ï¼‰"""
    features = {}

    # 1. é‡�åŠ›é™¤å�»åŠ é€Ÿåº¦ã�®é«˜åº¦ã�ªè§£æ��
    linear_acc, gravity = remove_gravity_from_acc(df_seq)

    # Linear acceleration features
    for i, axis in enumerate(["x", "y", "z"]):
        axis_signal = linear_acc[:, i]

        # åŸºæœ¬çµ±è¨ˆé‡�
        features[f"linear_acc_{axis}_mean"] = np.mean(axis_signal)
        features[f"linear_acc_{axis}_std"] = np.std(axis_signal)
        features[f"linear_acc_{axis}_max"] = np.max(axis_signal)
        features[f"linear_acc_{axis}_min"] = np.min(axis_signal)

        # é«˜åº¦ã�ªçµ±è¨ˆç‰¹å¾´
        stat_features = extract_statistical_features(axis_signal)
        for key, val in stat_features.items():
            features[f"linear_acc_{axis}_{key}"] = val

        # å‘¨æ³¢æ•°ç‰¹å¾´
        freq_features = extract_frequency_features(axis_signal)
        for key, val in freq_features.items():
            features[f"linear_acc_{axis}_{key}"] = val

        # æ™‚ç³»åˆ—ç‰¹å¾´
        ts_features = extract_time_series_features(axis_signal)
        for key, val in ts_features.items():
            features[f"linear_acc_{axis}_{key}"] = val

    # Linear acceleration magnitude
    linear_acc_mag = np.sqrt(np.sum(linear_acc**2, axis=1))
    features["linear_acc_mag_mean"] = np.mean(linear_acc_mag)
    features["linear_acc_mag_std"] = np.std(linear_acc_mag)
    features["linear_acc_mag_max"] = np.max(linear_acc_mag)

    # Gravity features
    gravity_mag = np.sqrt(np.sum(gravity**2, axis=1))
    features["gravity_mag_mean"] = np.mean(gravity_mag)
    features["gravity_mag_std"] = np.std(gravity_mag)

    # 2. è§’é€Ÿåº¦ã�®é«˜åº¦ã�ªè§£æ��
    angular_vel = calculate_angular_velocity_from_quat(df_seq)

    for i, axis in enumerate(["x", "y", "z"]):
        ang_signal = angular_vel[:, i]
        features[f"angular_vel_{axis}_mean"] = np.mean(ang_signal)
        features[f"angular_vel_{axis}_std"] = np.std(ang_signal)

        # çµ±è¨ˆãƒ»å‘¨æ³¢æ•°ãƒ»æ™‚ç³»åˆ—ç‰¹å¾´ã‚’è¿½åŠ 
        for feat_dict, prefix in [
            (extract_statistical_features(ang_signal), "stat"),
            (extract_frequency_features(ang_signal), "freq"),
            (extract_time_series_features(ang_signal), "ts"),
        ]:
            for key, val in feat_dict.items():
                features[f"angular_vel_{axis}_{key}"] = val

    angular_vel_mag = np.sqrt(np.sum(angular_vel**2, axis=1))
    features["angular_vel_mag_mean"] = np.mean(angular_vel_mag)
    features["angular_vel_mag_std"] = np.std(angular_vel_mag)

    # 3. ã‚¸ãƒ£ãƒ¼ã‚¯ã�¨ã‚¹ãƒŠãƒƒãƒ—ã�®è§£æ��
    jerk, snap = calculate_jerk_and_snap(df_seq)

    # Jerk features
    jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
    features["jerk_mag_mean"] = np.mean(jerk_mag)
    features["jerk_mag_std"] = np.std(jerk_mag)
    features["jerk_mag_max"] = np.max(jerk_mag)

    # Snap features
    snap_mag = np.sqrt(np.sum(snap**2, axis=1))
    features["snap_mag_mean"] = np.mean(snap_mag)
    features["snap_mag_std"] = np.std(snap_mag)

    # 4. åŸºæœ¬åŠ é€Ÿåº¦ã�®å…¨ç‰¹å¾´
    acc_mag = np.sqrt(df_seq["acc_x"] ** 2 + df_seq["acc_y"] ** 2 + df_seq["acc_z"] ** 2)

    # åŸºæœ¬çµ±è¨ˆé‡�
    features["acc_mag_mean"] = np.mean(acc_mag)
    features["acc_mag_std"] = np.std(acc_mag)
    features["acc_mag_max"] = np.max(acc_mag)
    features["acc_mag_min"] = np.min(acc_mag)

    # é«˜åº¦ã�ªç‰¹å¾´
    for feat_dict, prefix in [
        (extract_statistical_features(acc_mag), "stat"),
        (extract_frequency_features(acc_mag), "freq"),
        (extract_time_series_features(acc_mag), "ts"),
    ]:
        for key, val in feat_dict.items():
            features[f"acc_mag_{key}"] = val

    # 5. å›�è»¢ç‰¹å¾´é‡�ã�®æ‹¡å¼µ
    rot_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    for col in rot_cols:
        if col in df_seq.columns:
            rot_signal = df_seq[col].values
            features[f"{col}_mean"] = np.mean(rot_signal)
            features[f"{col}_std"] = np.std(rot_signal)

            # çµ±è¨ˆç‰¹å¾´ã‚’è¿½åŠ 
            stat_features = extract_statistical_features(rot_signal)
            for key, val in stat_features.items():
                features[f"{col}_{key}"] = val

    # 6. æ¸©åº¦ã‚»ãƒ³ã‚µãƒ¼ã�®é«˜åº¦ã�ªè§£æ��
    temp_cols = [f"thm_{i}" for i in range(1, 6)]
    available_temp_cols = [col for col in temp_cols if col in df_seq.columns]

    if available_temp_cols:
        temp_data = df_seq[available_temp_cols]
        features["temp_mean"] = temp_data.mean().mean()
        features["temp_std"] = temp_data.std().mean()
        features["temp_max"] = temp_data.max().max()
        features["temp_min"] = temp_data.min().min()

        # æ¸©åº¦ã�®æ™‚ç³»åˆ—ç‰¹å¾´
        temp_mean_series = temp_data.mean(axis=1)
        temp_ts_features = extract_time_series_features(temp_mean_series)
        for key, val in temp_ts_features.items():
            features[f"temp_{key}"] = val
    else:
        # Default values
        for suffix in [
            "mean",
            "std",
            "max",
            "min",
            "ts_autocorr_1",
            "ts_autocorr_5",
            "ts_trend",
            "ts_seasonality",
        ]:
            features[f"temp_{suffix}"] = 0

    # 7. ToFã‚»ãƒ³ã‚µãƒ¼ã�®é«˜åº¦ã�ªç‰¹å¾´é‡�
    tof_features = extract_advanced_tof_features(df_seq)
    features.update(tof_features)

    # 8. ã‚»ãƒ³ã‚µãƒ¼é–“ç›¸äº’ä½œç”¨ç‰¹å¾´é‡�
    interaction_features = extract_sensor_interaction_features(df_seq)
    features.update(interaction_features)

    # 9. ç³»åˆ—ãƒ¬ãƒ™ãƒ«ç‰¹å¾´é‡�
    features["sequence_length"] = len(df_seq)

    # å‹•ä½œã�®è¤‡é›‘ã�•æŒ‡æ¨™
    features["motion_complexity"] = np.std(linear_acc_mag) + np.std(angular_vel_mag)

    # å§¿å‹¢å¤‰åŒ–ç�‡
    if len(df_seq) > 1:
        quat_change = np.diff(df_seq[rot_cols].values, axis=0)
        features["orientation_change_rate"] = np.mean(np.linalg.norm(quat_change, axis=1))
    else:
        features["orientation_change_rate"] = 0

    return features


print("Advanced feature engineering functions defined")


# Load training data
print("Loading training data...")
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demo = pd.read_csv(
    "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
)

# Get actual gesture labels from data
actual_gestures = sorted(train_df["gesture"].unique())
print(f"Actual gesture classes: {actual_gestures}")
print(f"Data loaded: {len(train_df)} samples")
print(f"Unique sequences: {train_df['sequence_id'].nunique()}")
print(f"Number of gesture classes: {len(actual_gestures)}")

# Check class distribution
class_counts = train_df["gesture"].value_counts()
print("\nClass distribution:")
for gesture, count in class_counts.items():
    print(f"{gesture}: {count} sequences")


# Advanced feature extraction with progress tracking
print("Extracting advanced features...")

X = []
y = []
sequence_ids = train_df["sequence_id"].unique()

# Use tqdm for progress tracking
for i, seq_id in enumerate(tqdm(sequence_ids, desc="Feature extraction")):
    try:
        # Get sequence data
        seq_data = train_df[train_df["sequence_id"] == seq_id].copy()

        # Extract advanced physics features
        features = create_advanced_physics_features(seq_data)

        # Add demographics using subject column
        subject_id = seq_data["subject"].iloc[0]
        demo_data = train_demo[train_demo["subject"] == subject_id].iloc[0]
        features["age"] = demo_data["age"]
        features["is_male"] = 1 if demo_data["sex"] == "M" else 0
        features["height"] = demo_data.get("height_cm", 170)  # Default height

        # Add subject as categorical feature (for GroupKFold later)
        features["subject_id"] = subject_id

        X.append(features)
        y.append(seq_data["gesture"].iloc[0])

        # Memory cleanup every 500 sequences
        if (i + 1) % 500 == 0:
            gc.collect()

    except Exception as e:
        print(f"Error processing sequence {seq_id}: {e}")
        continue

# Convert to DataFrame
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

print(f"Feature extraction complete: {X_df.shape}")
print(f"Total features: {len(X_df.columns)}")
print(f"Sample features: {list(X_df.columns)[:15]}...")

# Check for NaN values and fill them
nan_counts = X_df.isnull().sum().sum()
if nan_counts > 0:
    print(f"Found {nan_counts} NaN values, filling with 0")
    X_df = X_df.fillna(0)

print("Data preprocessing complete")


# Label encoding first (needed for all cases)
from sklearn.preprocessing import LabelEncoder
from collections import Counter

le = LabelEncoder()
le.fit(actual_gestures)
y_encoded = le.transform(y_series)

print("Original class distribution:")
original_dist = Counter(y_encoded)
for class_idx, count in original_dist.items():
    print(f"{le.inverse_transform([class_idx])[0]}: {count}")

# Data augmentation for minority classes using SMOTE (if available)
try:
    from imblearn.over_sampling import SMOTE

    # Prepare data for SMOTE (remove subject_id for SMOTE, add back later)
    X_for_smote = X_df.drop(["subject_id"], axis=1, errors="ignore")
    subjects = (
        X_df["subject_id"].values if "subject_id" in X_df.columns else np.zeros(len(X_df))
    )

    # Apply SMOTE with conservative parameters
    min_samples = min(original_dist.values())
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

    smote = SMOTE(
        sampling_strategy="auto",  # Balance all classes
        k_neighbors=k_neighbors,
        random_state=42,
    )

    print("Applying SMOTE for class balancing...")
    X_balanced, y_balanced = smote.fit_resample(X_for_smote, y_encoded)

    # Create subject IDs for new synthetic samples
    original_size = len(X_for_smote)
    new_size = len(X_balanced)

    # Keep original subjects, assign synthetic subjects with high IDs
    subjects_balanced = np.concatenate(
        [
            subjects,
            np.arange(max(subjects) + 1, max(subjects) + 1 + (new_size - original_size)),
        ]
    )

    # Add subject_id back to balanced data
    X_balanced = pd.DataFrame(X_balanced, columns=X_for_smote.columns)
    X_balanced["subject_id"] = subjects_balanced

    print("Balanced class distribution:")
    balanced_dist = Counter(y_balanced)
    for class_idx, count in balanced_dist.items():
        print(f"{le.inverse_transform([class_idx])[0]}: {count}")

    print(f"Data augmented: {X_for_smote.shape} -> {X_balanced.shape}")

    # Use balanced data for training
    X_final = X_balanced
    y_final = y_balanced

except (ImportError, Exception) as e:
    print(f"SMOTE not available or failed ({e}), using original data")
    X_final = X_df
    y_final = y_encoded

print(f"Final dataset size: {X_final.shape}")


import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, using LightGBM and XGBoost only")

from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, classification_report

# Prepare final training data
num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {le.classes_}")

# Extract subjects for GroupKFold
subjects_for_cv = X_final["subject_id"].values
X_features = X_final.drop(["subject_id"], axis=1)

print(f"Features for training: {X_features.shape}")
print(f"Unique subjects: {len(np.unique(subjects_for_cv))}")


# Hyperparameter optimization with Optuna
def objective_lgb(trial):
    """LightGBM hyperparameter optimization"""
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
        "verbose": -1,
        "random_state": 42,
        "num_threads": 2,
    }

    # Cross-validation with GroupKFold
    gkf = GroupKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in gkf.split(X_features, y_final, groups=subjects_for_cv):
        X_train_fold, X_val_fold = X_features.iloc[train_idx], X_features.iloc[val_idx]
        y_train_fold, y_val_fold = y_final[train_idx], y_final[val_idx]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        pred_class = np.argmax(pred, axis=1)
        score = accuracy_score(y_val_fold, pred_class)
        scores.append(score)

    return np.mean(scores)


# Run optimization for LightGBM
print("Optimizing LightGBM hyperparameters...")
study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=20, timeout=1800)  # 30 minutes max

best_lgb_params = study_lgb.best_params
best_lgb_score = study_lgb.best_value
print(f"Best LightGBM CV score: {best_lgb_score:.4f}")
print(f"Best LightGBM params: {best_lgb_params}")


# Hyperparameter optimization for XGBoost
def objective_xgb(trial):
    """XGBoost hyperparameter optimization"""
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": 42,
        "nthread": 2,
    }

    # Cross-validation with GroupKFold
    gkf = GroupKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in gkf.split(X_features, y_final, groups=subjects_for_cv):
        X_train_fold, X_val_fold = X_features.iloc[train_idx], X_features.iloc[val_idx]
        y_train_fold, y_val_fold = y_final[train_idx], y_final[val_idx]

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "validation")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )

        pred = model.predict(dval)
        pred_class = np.argmax(pred, axis=1)
        score = accuracy_score(y_val_fold, pred_class)
        scores.append(score)

    return np.mean(scores)


# Run optimization for XGBoost
print("Optimizing XGBoost hyperparameters...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=20, timeout=1800)  # 30 minutes max

best_xgb_params = study_xgb.best_params
best_xgb_score = study_xgb.best_value
print(f"Best XGBoost CV score: {best_xgb_score:.4f}")
print(f"Best XGBoost params: {best_xgb_params}")


# CatBoost optimization (if available)
if CATBOOST_AVAILABLE:

    def objective_cat(trial):
        """CatBoost hyperparameter optimization"""
        params = {
            "objective": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": 1000,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli"]
            ),
            "random_seed": 42,
            "verbose": False,
            "thread_count": 2,
            "early_stopping_rounds": 100,
        }

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 1
            )
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1)

        # Cross-validation with GroupKFold
        gkf = GroupKFold(n_splits=5)
        scores = []

        for train_idx, val_idx in gkf.split(X_features, y_final, groups=subjects_for_cv):
            X_train_fold, X_val_fold = (
                X_features.iloc[train_idx],
                X_features.iloc[val_idx],
            )
            y_train_fold, y_val_fold = y_final[train_idx], y_final[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))

            pred = model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, pred)
            scores.append(score)

        return np.mean(scores)

    print("Optimizing CatBoost hyperparameters...")
    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(objective_cat, n_trials=15, timeout=1800)  # 30 minutes max

    best_cat_params = study_cat.best_params
    best_cat_score = study_cat.best_value
    print(f"Best CatBoost CV score: {best_cat_score:.4f}")
    print(f"Best CatBoost params: {best_cat_params}")
else:
    best_cat_params = None
    best_cat_score = 0


from sklearn.model_selection import train_test_split

# Split data using GroupKFold logic for final training
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X_features, y_final, groups=subjects_for_cv))

X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
y_train, y_val = y_final[train_idx], y_final[val_idx]

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")


# Train optimized LightGBM model
print("Training optimized LightGBM model...")

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "verbose": 0,
    "random_state": 42,
    "num_threads": 2,
}
lgb_params.update(best_lgb_params)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val)

model_lgb = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
)

print("LightGBM training complete")


# Train optimized XGBoost model
print("Training optimized XGBoost model...")

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "nthread": 2,
}
xgb_params.update(best_xgb_params)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

model_xgb = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=2000,
    evals=[(dval, "validation")],
    early_stopping_rounds=200,
    verbose_eval=100,
)

print("XGBoost training complete")


# Train optimized CatBoost model (if available)
model_cat = None
if CATBOOST_AVAILABLE and best_cat_params:
    print("Training optimized CatBoost model...")

    cat_params = {
        "objective": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 2000,
        "random_seed": 42,
        "verbose": 100,
        "thread_count": 2,
        "early_stopping_rounds": 200,
    }
    cat_params.update(best_cat_params)

    model_cat = cb.CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    print("CatBoost training complete")
else:
    print("CatBoost not available, using LightGBM and XGBoost only")


# Validation performance with advanced ensemble
print("\n=== Validation Results ===")

# LightGBM predictions
pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
pred_lgb_class = np.argmax(pred_lgb, axis=1)
lgb_accuracy = accuracy_score(y_val, pred_lgb_class)

# XGBoost predictions
dval = xgb.DMatrix(X_val)
pred_xgb = model_xgb.predict(dval)
pred_xgb_class = np.argmax(pred_xgb, axis=1)
xgb_accuracy = accuracy_score(y_val, pred_xgb_class)

print(f"LightGBM Accuracy: {lgb_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# CatBoost predictions (if available)
if model_cat is not None:
    pred_cat = model_cat.predict_proba(X_val)
    pred_cat_class = np.argmax(pred_cat, axis=1)
    cat_accuracy = accuracy_score(y_val, pred_cat_class)
    print(f"CatBoost Accuracy: {cat_accuracy:.4f}")

    # Three-model ensemble
    ensemble_weights = [0.4, 0.35, 0.25]  # LightGBM, XGBoost, CatBoost
    pred_ensemble = (
        ensemble_weights[0] * pred_lgb
        + ensemble_weights[1] * pred_xgb
        + ensemble_weights[2] * pred_cat
    )
else:
    # Two-model ensemble
    ensemble_weights = [0.55, 0.45]  # LightGBM, XGBoost
    pred_ensemble = ensemble_weights[0] * pred_lgb + ensemble_weights[1] * pred_xgb

pred_ensemble_class = np.argmax(pred_ensemble, axis=1)
ensemble_accuracy = accuracy_score(y_val, pred_ensemble_class)

print(f"\nğŸ�¯ Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"Target: 87%+ ({'âœ… ACHIEVED' if ensemble_accuracy >= 0.87 else 'â�Œ NOT YET'})")

# Detailed classification report
print("\n=== Detailed Classification Report ===")
target_names = [str(cls) for cls in le.classes_]
print(classification_report(y_val, pred_ensemble_class, target_names=target_names))


# Save all models
print("Saving optimized models...")

# Save models
with open("advanced_model_lgb_v5.pkl", "wb") as f:
    pickle.dump(model_lgb, f)

with open("advanced_model_xgb_v5.pkl", "wb") as f:
    pickle.dump(model_xgb, f)

if model_cat is not None:
    model_cat.save_model("advanced_model_cat_v5.cbm")

# Save label encoder and parameters
with open("advanced_label_encoder_v5.pkl", "wb") as f:
    pickle.dump(le, f)

# Save ensemble weights and model info
model_info = {
    "ensemble_weights": ensemble_weights,
    "has_catboost": model_cat is not None,
    "lgb_params": lgb_params,
    "xgb_params": xgb_params,
    "validation_accuracy": ensemble_accuracy,
    "feature_names": list(X_features.columns),
}

if model_cat is not None:
    model_info["cat_params"] = cat_params

with open("advanced_model_info_v5.pkl", "wb") as f:
    pickle.dump(model_info, f)

print("Models saved successfully")
print(f"Final validation accuracy: {ensemble_accuracy:.4f}")


def predict(sequence, demographics):
    """
    Advanced physics-based prediction with optimized ensemble model.
    Target: 87%+ validation accuracy with comprehensive feature engineering.
    """
    try:
        # Convert to DataFrame
        df_seq = pd.DataFrame(sequence)

        # Extract comprehensive advanced features
        features = create_advanced_physics_features(df_seq)

        # Add demographics (using lowercase keys from CMI server)
        features["age"] = demographics.get("age", 30)
        features["is_male"] = 1 if demographics.get("sex", "M") == "M" else 0
        features["height"] = demographics.get("height_cm", 170)

        # Add dummy subject_id for inference (not used in prediction)
        features["subject_id"] = 0

        # Convert to DataFrame and ensure correct feature order
        X = pd.DataFrame([features])
        X_features_pred = X[model_info["feature_names"]]  # Ensure correct feature order

        # LightGBM prediction
        pred_lgb = model_lgb.predict(
            X_features_pred, num_iteration=model_lgb.best_iteration
        )

        # XGBoost prediction
        dtest = xgb.DMatrix(X_features_pred)
        pred_xgb = model_xgb.predict(dtest)

        # CatBoost prediction (if available)
        if model_info["has_catboost"] and model_cat is not None:
            pred_cat = model_cat.predict_proba(X_features_pred)

            # Three-model ensemble
            pred_ensemble = (
                model_info["ensemble_weights"][0] * pred_lgb
                + model_info["ensemble_weights"][1] * pred_xgb
                + model_info["ensemble_weights"][2] * pred_cat
            )
        else:
            # Two-model ensemble
            pred_ensemble = (
                model_info["ensemble_weights"][0] * pred_lgb
                + model_info["ensemble_weights"][1] * pred_xgb
            )

        # Get predicted class
        pred_class_idx = np.argmax(pred_ensemble[0])
        pred_gesture = le.inverse_transform([pred_class_idx])[0]

        # Memory cleanup
        del df_seq, features, X, X_features_pred, pred_lgb, pred_xgb, pred_ensemble
        if model_info["has_catboost"]:
            del pred_cat
        gc.collect()

        return pred_gesture

    except Exception as e:
        print(f"Advanced prediction error: {e}")
        # Fallback to most common gesture from training data
        return "Text on phone"  # Most common in training data


print("Advanced prediction function defined")
print(f"Expected validation accuracy: {ensemble_accuracy:.4f}")
print(
    f"Target achievement: {'âœ… SUCCESS' if ensemble_accuracy >= 0.87 else 'â�Œ NEEDS MORE WORK'}"
)


# Test the advanced prediction function
print("Testing advanced prediction function...")

# Get a sample sequence from validation set
test_seq_id = train_df["sequence_id"].unique()[val_idx[0]]  # Use validation sequence
test_seq = train_df[train_df["sequence_id"] == test_seq_id].drop(
    [
        "sequence_id",
        "behavior",
        "gesture",
        "subject",
        "orientation",
        "phase",
        "row_id",
        "sequence_type",
        "sequence_counter",
    ],
    axis=1,
    errors="ignore",
)

# Get demographics for test
test_subject = train_df[train_df["sequence_id"] == test_seq_id]["subject"].iloc[0]
test_demo_row = train_demo[train_demo["subject"] == test_subject].iloc[0]
test_demo = {
    "age": test_demo_row["age"],
    "sex": test_demo_row["sex"],
    "height_cm": test_demo_row.get("height_cm", 170),
}

# Test prediction
test_pred = predict(test_seq.to_dict("records"), test_demo)
actual_gesture = train_df[train_df["sequence_id"] == test_seq_id]["gesture"].iloc[0]

print(f"Test prediction: {test_pred}")
print(f"Actual gesture: {actual_gesture}")
print(f"Match: {test_pred == actual_gesture}")
print(f"\nAdvanced model ready with {ensemble_accuracy:.4f} validation accuracy")


# Start inference server
print("Starting Advanced CMI Inference Server...")
print(f"Model validation accuracy: {ensemble_accuracy:.4f}")
print(
    f"Target 87%+: {'âœ… ACHIEVED' if ensemble_accuracy >= 0.87 else 'â�Œ WORKING ON IT'}"
)
print("Server will handle predictions for test data with advanced feature engineering")

server = CMIInferenceServer(predict)
server.serve()

print("Advanced inference complete")

