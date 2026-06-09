# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: IMPORTS & ENHANCED FEATURE‐EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score

import lightgbm as lgb

from scipy.fftpack import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, butter, filtfilt

import warnings
warnings.filterwarnings('ignore')


def _apply_butterworth_filter(signal, sampling_rate=50.0, lowcut=0.5, highcut=20.0, order=4):
    """Apply Butterworth bandpass filter to remove noise."""
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0:
        high = 0.99
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def _compute_enhanced_basic_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced aggregation with additional statistical measures.
    """
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    rot_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
    thm_cols = [c for c in df.columns if c.startswith("thm_")]
    tof_cols_by_sensor = {
        i: [c for c in df.columns if c.startswith(f"tof_{i}_v")] for i in range(1, 6)
    }
    all_tof_cols = [c for sub in tof_cols_by_sensor.values() for c in sub]

    agg_dict = {}
    # Enhanced aggregations with more statistics
    for c in acc_cols + rot_cols + thm_cols + ["acc_mag"]:
        agg_dict[c] = [
            "mean", "std", "min", "max", "median", "skew",
            lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)
        ]
    # For each TOF pixel, keep fewer stats to avoid too‐many features
    for c in all_tof_cols:
        agg_dict[c] = ["mean", "std", "median"]

    grouped = df.groupby("sequence_id").agg(agg_dict)

    # Flatten multi‐level columns into names like “acc_x_mean”, “acc_x_q25”, etc.
    new_columns = []
    for lvl0, func in grouped.columns.to_list():
        if callable(func):
            fname = func.__name__ if hasattr(func, "__name__") else str(func)
            if "percentile" in fname:
                # distinguish 25 vs 75 based on lambda naming convention
                if "25" in fname:
                    new_columns.append(f"{lvl0}_q25")
                else:
                    new_columns.append(f"{lvl0}_q75")
            else:
                new_columns.append(f"{lvl0}_{fname}")
        else:
            new_columns.append(f"{lvl0}_{func}")
    grouped.columns = new_columns
    return grouped


def _compute_enhanced_time_series_features(df: pd.DataFrame, sampling_rate: float = 50.0) -> pd.DataFrame:
    """
    Enhanced time series features with filtering, peak detection, and more frequency‐domain features.
    """
    use_behavior = "behavior" in df.columns
    if use_behavior:
        df_gest = df[df["behavior"] == "Gesture"].copy()
    else:
        df_gest = df.copy()

    if "acc_mag" not in df_gest.columns:
        df_gest["acc_mag"] = np.sqrt(
            df_gest["acc_x"]**2 + df_gest["acc_y"]**2 + df_gest["acc_z"]**2
        )

    def _extract_enhanced_for_sequence(sub: pd.DataFrame) -> pd.Series:
        arr_x = sub["acc_x"].values
        arr_y = sub["acc_y"].values
        arr_z = sub["acc_z"].values
        arr_mag = sub["acc_mag"].values
        n = len(sub)

        # If too short, just return zeros for all enhanced features
        if n < 3:
            zero_keys = [
                "acc_x_rms", "acc_y_rms", "acc_z_rms",
                "acc_x_filtered_std", "acc_y_filtered_std", "acc_z_filtered_std",
                "acc_xy_corr", "acc_xz_corr", "acc_yz_corr",
                "acc_mag_energy_0_2Hz", "acc_mag_energy_2_5Hz",
                "acc_mag_energy_5_10Hz", "acc_mag_energy_10_20Hz",
                "acc_mag_spectral_centroid", "acc_mag_spectral_entropy",
                "acc_mag_peak_count", "acc_mag_zero_crossing_rate",
                "acc_mag_dominant_freq", "gesture_duration"
            ]
            return pd.Series({f: 0.0 for f in zero_keys})

        # 1) Apply Butterworth filter (if it fails, fall back to raw)
        try:
            arr_x_filt = _apply_butterworth_filter(arr_x, sampling_rate)
            arr_y_filt = _apply_butterworth_filter(arr_y, sampling_rate)
            arr_z_filt = _apply_butterworth_filter(arr_z, sampling_rate)
            arr_mag_filt = _apply_butterworth_filter(arr_mag, sampling_rate)
        except:
            arr_x_filt = arr_x
            arr_y_filt = arr_y
            arr_z_filt = arr_z
            arr_mag_filt = arr_mag

        # 2) RMS of filtered signals
        acc_x_rms = np.sqrt(np.mean(arr_x_filt**2))
        acc_y_rms = np.sqrt(np.mean(arr_y_filt**2))
        acc_z_rms = np.sqrt(np.mean(arr_z_filt**2))

        # 3) Filtered standard deviations
        acc_x_filtered_std = np.std(arr_x_filt)
        acc_y_filtered_std = np.std(arr_y_filt)
        acc_z_filtered_std = np.std(arr_z_filt)

        # 4) Safe correlations on filtered axes
        def safe_corr(a, b):
            if len(a) > 1 and a.std() > 1e-8 and b.std() > 1e-8:
                return np.corrcoef(a, b)[0, 1]
            return 0.0

        acc_xy_corr = safe_corr(arr_x_filt, arr_y_filt)
        acc_xz_corr = safe_corr(arr_x_filt, arr_z_filt)
        acc_yz_corr = safe_corr(arr_y_filt, arr_z_filt)

        # 5) FFT‐based features on filtered magnitude
        yf = rfft(arr_mag_filt)
        xf = rfftfreq(n, 1.0 / sampling_rate)
        magnitude_spectrum = np.abs(yf)

        band1_energy = np.sum(magnitude_spectrum[(xf >= 0) & (xf < 2)]**2)
        band2_energy = np.sum(magnitude_spectrum[(xf >= 2) & (xf < 5)]**2)
        band3_energy = np.sum(magnitude_spectrum[(xf >= 5) & (xf < 10)]**2)
        band4_energy = np.sum(magnitude_spectrum[(xf >= 10) & (xf < 20)]**2)

        if magnitude_spectrum.sum() > 0:
            centroid = np.sum(xf * magnitude_spectrum) / magnitude_spectrum.sum()
            dominant_freq = xf[np.argmax(magnitude_spectrum)]
        else:
            centroid = 0.0
            dominant_freq = 0.0

        psd = magnitude_spectrum**2
        psd_norm = psd / psd.sum() if psd.sum() > 0 else psd
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

        # 6) Peak detection on magnitude
        try:
            peaks, _ = find_peaks(arr_mag_filt, height=np.std(arr_mag_filt))
            peak_count = len(peaks)
        except:
            peak_count = 0

        # 7) Zero‐crossing rate
        zero_crossings = np.sum(np.diff(np.sign(arr_mag_filt - np.mean(arr_mag_filt))) != 0)
        zero_crossing_rate = zero_crossings / (n - 1) if n > 1 else 0

        # 8) Gesture duration
        gesture_duration = n / sampling_rate

        return pd.Series({
            "acc_x_rms":                  acc_x_rms,
            "acc_y_rms":                  acc_y_rms,
            "acc_z_rms":                  acc_z_rms,
            "acc_x_filtered_std":         acc_x_filtered_std,
            "acc_y_filtered_std":         acc_y_filtered_std,
            "acc_z_filtered_std":         acc_z_filtered_std,
            "acc_xy_corr":                acc_xy_corr,
            "acc_xz_corr":                acc_xz_corr,
            "acc_yz_corr":                acc_yz_corr,
            "acc_mag_energy_0_2Hz":       band1_energy,
            "acc_mag_energy_2_5Hz":       band2_energy,
            "acc_mag_energy_5_10Hz":      band3_energy,
            "acc_mag_energy_10_20Hz":     band4_energy,
            "acc_mag_spectral_centroid":  centroid,
            "acc_mag_spectral_entropy":   spectral_entropy,
            "acc_mag_peak_count":         peak_count,
            "acc_mag_zero_crossing_rate": zero_crossing_rate,
            "acc_mag_dominant_freq":      dominant_freq,
            "gesture_duration":           gesture_duration
        })

    ts_feats = df_gest.groupby("sequence_id").apply(_extract_enhanced_for_sequence)
    ts_feats.index.name = "sequence_id"
    return ts_feats


def _compute_enhanced_thermopile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced thermopile features with spatial patterns.
    """
    thm_cols = [c for c in df.columns if c.startswith("thm_")]

    def _extract_enhanced_thm_for_sequence(sub: pd.DataFrame) -> pd.Series:
        result = {}
        thm_data = sub[thm_cols].values  # shape: (time, n_thm_cols)
        for c in thm_cols:
            vals = sub[c].dropna().values
            if len(vals) >= 2:
                slope = (vals[-1] - vals[0]) / (len(vals) - 1)
                skewness = skew(vals)
                kurtv = kurtosis(vals)
                temp_range = vals.max() - vals.min()
                temp_gradient = np.std(np.diff(vals)) if len(vals) > 1 else 0.0
            else:
                slope = skewness = kurtv = temp_range = temp_gradient = 0.0
            result[f"{c}_slope"] = slope
            result[f"{c}_skew"]   = skewness
            result[f"{c}_kurtosis"] = kurtv
            result[f"{c}_range"]  = temp_range
            result[f"{c}_gradient"] = temp_gradient

        # If it's an 8×8 thermopile (64 columns), compute center of heat for first few frames
        if len(thm_cols) == 64:
            for t in range(min(len(thm_data), 5)):
                frame = thm_data[t].reshape(8, 8)
                y_idx, x_idx = np.mgrid[0:8, 0:8]
                total_heat = np.sum(frame)
                if total_heat > 0:
                    center_y = np.sum(y_idx * frame) / total_heat
                    center_x = np.sum(x_idx * frame) / total_heat
                    result[f"thm_center_y_t{t}"] = center_y
                    result[f"thm_center_x_t{t}"] = center_x
                else:
                    result[f"thm_center_y_t{t}"] = 0.0
                    result[f"thm_center_x_t{t}"] = 0.0

        return pd.Series(result)

    thm_feats = df.groupby("sequence_id").apply(_extract_enhanced_thm_for_sequence)
    thm_feats.index.name = "sequence_id"
    return thm_feats


def _compute_tof_block_time_trends(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    TOF block features + time‐trend for each sensor block.
    """
    tof_cols_by_sensor = {
        i: [c for c in df.columns if c.startswith(f"tof_{i}_v")] for i in range(1, 6)
    }

    df_masked = df.copy()
    for i, cols in tof_cols_by_sensor.items():
        df_masked[cols] = df_masked[cols].replace(-1, np.nan)

    def _extract_tof_for_sequence(sub: pd.DataFrame) -> pd.Series:
        out = {}
        for i, cols in tof_cols_by_sensor.items():
            block = sub[cols]
            valid_mask = ~block.isna()
            total = block.shape[0] * block.shape[1]
            valid_count = valid_mask.sum().sum()
            frac_valid = valid_count / total if total > 0 else 0.0

            vals = block.values.flatten()
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                block_min = float(vals.min())
                block_max = float(vals.max())
                count_lt_T = int((vals < threshold).sum())
            else:
                block_min = block_max = count_lt_T = 0

            # Time‐trend of average distance per frame:
            avg_dist_per_frame = block.mean(axis=1).values
            idxs = np.where(~np.isnan(avg_dist_per_frame))[0]
            if len(idxs) >= 2:
                x = idxs
                y = avg_dist_per_frame[idxs]
                slope = np.cov(x, y, bias=True)[0, 1] / np.var(x) if np.var(x) > 0 else 0.0
            else:
                slope = 0.0

            out[f"tof_{i}_frac_valid"]        = frac_valid
            out[f"tof_{i}_block_min"]         = block_min
            out[f"tof_{i}_block_max"]         = block_max
            out[f"tof_{i}_count_lt_{int(threshold)}"] = count_lt_T
            out[f"tof_{i}_time_trend"]        = slope
        return pd.Series(out)

    tof_feats = df_masked.groupby("sequence_id").apply(_extract_tof_for_sequence)
    tof_feats.index.name = "sequence_id"
    return tof_feats


def extract_enhanced_sequence_features(
    df: pd.DataFrame,
    demogs_df: pd.DataFrame = None,
    sampling_rate: float = 50.0
) -> pd.DataFrame:
    """
    (1) Compute acc_mag + jerk terms
    (2) Enhanced basic summary‐stat aggregates
    (3) Enhanced time‐series features (FFT, RMS, correlations, peaks, entropy)
    (4) Enhanced thermopile features
    (5) TOF‐block features + time trends
    (6) Join any {behavior, gesture, subject} labels
    (7) Merge demographics (age, sex, etc.) if provided
    """
    df = df.copy()
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    # Add “jerk” terms for acc and acc_mag
    for col in ["acc_x", "acc_y", "acc_z", "acc_mag"]:
        df[f"{col}_jerk"] = df.groupby("sequence_id")[col].diff().fillna(0)

    # 1) Enhanced basic aggregates
    basic_feats = _compute_enhanced_basic_aggregates(df)

    # 2) Enhanced time-series features
    ts_feats = _compute_enhanced_time_series_features(df, sampling_rate=sampling_rate)

    # 3) Enhanced thermopile features
    thm_feats = _compute_enhanced_thermopile_features(df)

    # 4) TOF block-level features + time trends
    tof_feats = _compute_tof_block_time_trends(df, threshold=10.0)

    # 5) Drop overlapping thermopile columns from basic_feats (if any)
    overlap_thm = set(basic_feats.columns).intersection(thm_feats.columns)
    if overlap_thm:
        basic_feats = basic_feats.drop(columns=list(overlap_thm), errors="ignore")

    # 6) Combine everything into one DataFrame
    features = (
        basic_feats
        .join(ts_feats, how="left")
        .join(thm_feats, how="left")
        .join(tof_feats, how="left")
    )

    # 7) Attach labels (if in training set)
    if {"behavior", "gesture", "subject"}.issubset(df.columns):
        seq_labels = (
            df[["sequence_id", "behavior", "gesture", "subject"]]
            .drop_duplicates(subset=["sequence_id"])
            .set_index("sequence_id")
        )
        for col in ["behavior", "gesture", "subject"]:
            if col in features.columns:
                features = features.drop(columns=[col])
        features = features.join(seq_labels, how="left")

    elif "subject" in df.columns:
        seq_subj = (
            df[["sequence_id", "subject"]]
            .drop_duplicates(subset=["sequence_id"])
            .set_index("sequence_id")
        )
        if "subject" in features.columns:
            features = features.drop(columns=["subject"])
        features = features.join(seq_subj, how="left")

    # 8) Merge demographics if provided
    if demogs_df is not None and "subject" in features.columns:
        overlap = set(features.columns).intersection(set(demogs_df.columns))
        if overlap:
            features = features.drop(columns=list(overlap), errors="ignore")
        features = features.join(demogs_df, on="subject", how="left")

    return features


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: LOAD TRAINED OBJECTS & DEFINE predict(...)
# ─────────────────────────────────────────────────────────────────────────────

import os
import pickle
import traceback

import polars as pl
import numpy as np
import pandas as pd

# (Assume Cell 1 has already done all imports for scikit‐learn, LightGBM, SciPy, etc.)

# ─────────────────────────────────────────────────────────────────────────────
# 2a) Load exactly the objects you saved during training
# ─────────────────────────────────────────────────────────────────────────────
PICKLE_PATH = "/kaggle/input/d/merolavtechnology/bfrb-dataset/trained_objects.pkl"
if not os.path.exists(PICKLE_PATH):
    raise FileNotFoundError(f"Cannot find '{PICKLE_PATH}'. Run training first.")

with open(PICKLE_PATH, "rb") as f:
    obj = pickle.load(f)

GLOBAL_BST       = obj["bst"]        # LightGBM booster (trained on selected features)
GLOBAL_LE        = obj["le"]         # LabelEncoder
GLOBAL_SCALER    = obj["scaler"]     # RobustScaler fitted on all pre‐selected features
GLOBAL_SELECTOR  = obj["selector"]   # SelectFromModel (no longer used at inference)
GLOBAL_FEAT_LIST = obj["feat_list"]  # The exact feature‐names (in order) that final_model was trained on

print("✅ Loaded model, encoder, scaler, selector, and feature list.")

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Called once per test sequence. Returns exactly one string "gesture".
    """
    try:
        # 1) Convert Polars → pandas
        seq_pd = sequence.to_pandas()
        demog_pd = demographics.to_pandas().set_index("subject") if demographics is not None else None

        # 2) Extract enhanced features (must match training)
        feats = extract_enhanced_sequence_features(seq_pd, demog_pd)

        # 3) Drop any "behavior"/"gesture"/"subject" columns if present
        drop_cols = [c for c in ("behavior", "gesture", "subject") if c in feats.columns]
        X_row = feats.drop(columns=drop_cols)

        # 4) Replace infinities / NaNs with 0
        X_row = X_row.replace([np.inf, -np.inf], np.nan).fillna(0)

        # ────────────────────────────────────────────────────────────────────
        # 5) PRE‐SCALING: reindex to exactly what the scaler was fit on
        #    (scaler.feature_names_in_ == the DataFrame.columns that RobustScaler.fit() saw in training)
        pre_cols = list(GLOBAL_SCALER.feature_names_in_)
        X_row = X_row.reindex(columns=pre_cols, fill_value=0)

        # 6) SCALE
        X_scaled = pd.DataFrame(
            GLOBAL_SCALER.transform(X_row),
            index=X_row.index,
            columns=pre_cols
        )

        # ────────────────────────────────────────────────────────────────────
        # 7) ADD exactly the same "interaction" columns that training used:
        #    - "acc_x_std_to_rms" if both "acc_x_std" & "acc_x_rms" exist
        #    - "acc_corr_abs_sum" if both "acc_xy_corr" & "acc_yz_corr" exist

        if {"acc_x_std", "acc_x_rms"}.issubset(X_scaled.columns):
            X_scaled["acc_x_std_to_rms"] = X_scaled["acc_x_std"] / (X_scaled["acc_x_rms"] + 1e-6)
        else:
            X_scaled["acc_x_std_to_rms"] = 0.0

        if {"acc_xy_corr", "acc_yz_corr"}.issubset(X_scaled.columns):
            X_scaled["acc_corr_abs_sum"] = X_scaled["acc_xy_corr"].abs() + X_scaled["acc_yz_corr"].abs()
        else:
            X_scaled["acc_corr_abs_sum"] = 0.0

        # ────────────────────────────────────────────────────────────────────
        # 8) POST‐SCALING: now reindex to exactly the final feature list
        #    that the LightGBM model was trained on (GLOBAL_FEAT_LIST).
        #
        #    Because GLOBAL_FEAT_LIST == selected_features in training, this
        #    drops any extra columns and orders them correctly.
        X_final = X_scaled.reindex(columns=GLOBAL_FEAT_LIST, fill_value=0)

        # ────────────────────────────────────────────────────────────────────
        # 9) Predict with LightGBM directly on X_final
        proba = GLOBAL_BST.predict(X_final, num_iteration=GLOBAL_BST.best_iteration)
        idx = int(np.argmax(proba, axis=1)[0])

        # 10) Decode integer back into string label
        return GLOBAL_LE.inverse_transform([idx])[0]

    except Exception:
        seq_id = "unknown"
        try:
            if "sequence_id" in sequence.columns:
                seq_id = sequence["sequence_id"][0]
        except:
            pass
        print(f"----- ERROR in predict() for sequence_id = {seq_id} -----")
        traceback.print_exc()
        raise



# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: RUN THE INFERENCE SERVER
# ─────────────────────────────────────────────────────────────────────────────

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
        )
    )


