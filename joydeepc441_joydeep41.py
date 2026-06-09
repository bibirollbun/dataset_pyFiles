%pip -q install pandas numpy scikit-learn polars scipy joblib grpcio
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import joblib
from scipy import stats
from scipy.spatial.transform import Rotation as R


# Paths configuration
IS_KAGGLE = Path('/kaggle/input').exists()

if IS_KAGGLE:
    ARTIFACT_PATH = Path('/kaggle/input/joydeep-gesture-model/joydeep_model_artifacts.joblib')
else:
    ARTIFACT_PATH = Path(r"C:\Users\Joydeep\Desktop\BigData\Final Project\cmi-detect-behavior-with-sensor-data\joydeep_model_artifacts.joblib")

print(f'Kaggle: {IS_KAGGLE}, Artifact: {ARTIFACT_PATH.name}')



# Inference functions and model artifacts (accuracy-oriented, still inference-only)

# Global state (lazy-loaded)
_artifacts_loaded = False
_artifacts = None

best_thr = None
fallback_non_target = None
le_gesture_classes = None

bin_models_all = None
bin_models_imu = None
multi_models_all = None
multi_models_imu = None

feature_columns_all = None
feature_columns_imu = None

use_stacking = None
stack_bin = None
stack_multi = None
calibrator = None
subject_norm = None

IMU_COLS = None
THM_COLS = None
TOF_COLS = None
DERIVED_COLS = None
DEMO_COLS = None
SENSOR_COLS_ALL = None
NEG1_MISSING_ALL = None

# Flags to avoid unnecessary compute when models don't use those features
NEED_SKEW_KURT = False
NEED_TEMP_SPECTRAL = False

# Defaults (used if not found in artifacts)
IMU_COLS_DEFAULT = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
THM_COLS_DEFAULT = [f"thm_{i}" for i in range(1, 6)]
TOF_COLS_DEFAULT = [f"tof_{s}_v{v}" for s in range(1, 6) for v in range(64)]
DEMO_COLS_DEFAULT = [
    "adult_child",
    "age",
    "sex",
    "handedness",
    "height_cm",
    "shoulder_to_wrist_cm",
    "elbow_to_wrist_cm",
]

TEMP_HEAD_N = 150
TEMP_TAIL_N = 150
TEMP_EPS = 1e-6

SPECTRAL_CONFIG = {
    "fs": 200.0,
    "bands": [(0, 20), (20, 40), (40, 60), (60, 100)],
}


def load_artifacts() -> None:
    global _artifacts_loaded, _artifacts
    global best_thr, fallback_non_target, le_gesture_classes
    global bin_models_all, bin_models_imu, multi_models_all, multi_models_imu
    global feature_columns_all, feature_columns_imu
    global use_stacking, stack_bin, stack_multi, calibrator, subject_norm
    global IMU_COLS, THM_COLS, TOF_COLS, DERIVED_COLS, DEMO_COLS, SENSOR_COLS_ALL, NEG1_MISSING_ALL
    global NEED_SKEW_KURT, NEED_TEMP_SPECTRAL

    if _artifacts_loaded:
        return

    _artifacts = joblib.load(ARTIFACT_PATH)

    best_thr = float(_artifacts.get("best_thr", 0.5))
    fallback_non_target = str(_artifacts.get("fallback_non_target", "Text on phone"))
    le_gesture_classes = [str(x) for x in list(_artifacts.get("le_gesture_classes", []))]

    bin_models_all = _artifacts.get("bin_models_all", [])
    bin_models_imu = _artifacts.get("bin_models_imu", [])
    multi_models_all = _artifacts.get("multi_models_all", [])
    multi_models_imu = _artifacts.get("multi_models_imu", [])

    feature_columns_all = list(_artifacts.get("feature_columns_all", []))
    feature_columns_imu = list(_artifacts.get("feature_columns_imu", []))

    use_stacking = bool(_artifacts.get("use_stacking", False))
    stack_bin = _artifacts.get("stack_bin", None)
    stack_multi = _artifacts.get("stack_multi", None)
    calibrator = _artifacts.get("calibrator", None)
    subject_norm = _artifacts.get("subject_norm", None)

    IMU_COLS = list(_artifacts.get("imu_cols", IMU_COLS_DEFAULT))
    THM_COLS = list(_artifacts.get("thm_cols", THM_COLS_DEFAULT))
    TOF_COLS = list(_artifacts.get("tof_cols", TOF_COLS_DEFAULT))
    DERIVED_COLS = list(_artifacts.get("derived_cols", []))
    DEMO_COLS = list(_artifacts.get("demo_cols", DEMO_COLS_DEFAULT))

    SENSOR_COLS_ALL = IMU_COLS + THM_COLS + TOF_COLS
    NEG1_MISSING_ALL = set(THM_COLS + TOF_COLS)

    # Decide whether to compute more expensive features
    all_feats = set(feature_columns_all) | set(feature_columns_imu)
    NEED_SKEW_KURT = any(f.endswith("__skew") or f.endswith("__kurt") for f in all_feats)

    spectral_markers = (
        "temp__acc_w_norm_entropy",
        "temp__acc_w_norm_centroid",
        "temp__acc_w_norm_bandwidth",
        "temp__acc_w_norm_dominant_freq",
        "temp__acc_w_norm_autocorr",
        "temp__ang_v_norm_entropy",
        "temp__ang_v_norm_centroid",
        "temp__ang_v_norm_bandwidth",
        "temp__ang_v_norm_dominant_freq",
        "temp__ang_v_norm_autocorr",
        "band_0_20",
        "band_20_40",
        "band_40_60",
        "band_60_100",
    )
    NEED_TEMP_SPECTRAL = any(any(m in f for m in spectral_markers) for f in all_feats)

    _artifacts_loaded = True


def _is_left_handed(handedness: object) -> bool:
    if handedness is None:
        return False
    s = str(handedness).strip().lower()
    return s.startswith("l")


def _select_numpy(sequence: pl.DataFrame, cols: list[str], fill_value: float = np.nan) -> np.ndarray:
    exprs = []
    for c in cols:
        if c in sequence.columns:
            exprs.append(pl.col(c))
        else:
            exprs.append(pl.lit(fill_value).alias(c))
    return sequence.select(exprs).to_numpy()


def handle_quaternion_missing_values(rot_wxyz: np.ndarray) -> np.ndarray:
    rot = rot_wxyz.astype(np.float64, copy=True)
    out = rot.copy()
    for i in range(rot.shape[0]):
        row = rot[i]
        missing = np.isnan(row)
        missing_count = int(missing.sum())
        if missing_count == 0:
            n = np.linalg.norm(row)
            out[i] = row / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])
        elif missing_count == 1:
            mi = int(np.where(missing)[0][0])
            valid = row[~missing]
            s2 = float(np.sum(valid * valid))
            if s2 <= 1.0:
                mv = float(np.sqrt(max(0.0, 1.0 - s2)))
                if i > 0 and np.isfinite(out[i - 1, mi]) and out[i - 1, mi] < 0:
                    mv = -mv
                out[i] = row
                out[i, mi] = mv
                n = np.linalg.norm(out[i])
                out[i] = out[i] / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])
            else:
                out[i] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            out[i] = np.array([1.0, 0.0, 0.0, 0.0])
    return out


def compute_world_acceleration(acc_xyz: np.ndarray, rot_wxyz: np.ndarray) -> np.ndarray:
    try:
        quat_xyzw = rot_wxyz[:, [1, 2, 3, 0]]
        r = R.from_quat(quat_xyzw)
        return r.apply(acc_xyz)
    except Exception:
        return acc_xyz.copy()


def compute_rot_step_features(rot_wxyz: np.ndarray, time_delta: float = 1 / 200) -> tuple[np.ndarray, np.ndarray]:
    T = rot_wxyz.shape[0]
    if T < 2:
        return np.zeros((T, 3), dtype=np.float64), np.zeros((T,), dtype=np.float64)
    try:
        quat_xyzw = rot_wxyz[:, [1, 2, 3, 0]]
        r = R.from_quat(quat_xyzw)
        delta = r[:-1].inv() * r[1:]
        rotvec = delta.as_rotvec()
        angle = np.linalg.norm(rotvec, axis=1)
        rotvec_full = np.vstack([np.zeros((1, 3)), rotvec])
        angle_full = np.concatenate([[0.0], angle])
        return rotvec_full, angle_full
    except Exception:
        return np.zeros((T, 3), dtype=np.float64), np.zeros((T,), dtype=np.float64)


def compute_angular_velocity(rot_wxyz: np.ndarray, time_delta: float = 1 / 200) -> np.ndarray:
    rotvec_step, _ = compute_rot_step_features(rot_wxyz, time_delta=time_delta)
    return rotvec_step / float(time_delta)


def mirror_acc_and_quaternion(acc_xyz: np.ndarray, rot_wxyz: np.ndarray, axis: str = "x") -> tuple[np.ndarray, np.ndarray]:
    if axis != "x":
        raise ValueError("Only axis='x' implemented")
    M = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
    acc_m = (acc_xyz @ M.T).astype(np.float64, copy=False)
    try:
        quat_xyzw = rot_wxyz[:, [1, 2, 3, 0]]
        r = R.from_quat(quat_xyzw)
        mats = r.as_matrix()
        mats_m = M[None, :, :] @ mats @ M[None, :, :]
        quat_xyzw_m = R.from_matrix(mats_m).as_quat()
        rot_wxyz_m = quat_xyzw_m[:, [3, 0, 1, 2]].astype(np.float64, copy=False)
        return acc_m, rot_wxyz_m
    except Exception:
        return acc_m, rot_wxyz.copy()


def compute_spectral_features(signal: np.ndarray, fs: float = 200.0) -> dict:
    out = {
        "entropy": 0.0,
        "centroid": 0.0,
        "bandwidth": 0.0,
        "dominant_freq": 0.0,
        "band_0_20": 0.0,
        "band_20_40": 0.0,
        "band_40_60": 0.0,
        "band_60_100": 0.0,
    }
    if signal.size < 8:
        return out
    try:
        sig = np.nan_to_num(signal - np.nanmean(signal), nan=0.0)
        fft_vals = np.fft.rfft(sig)
        power = (np.abs(fft_vals) ** 2).astype(np.float64, copy=False)
        freqs = np.fft.rfftfreq(sig.size, d=1.0 / fs).astype(np.float64, copy=False)

        total_power = float(np.sum(power))
        if total_power < 1e-12:
            return out
        pdf = power / total_power

        out["entropy"] = float(stats.entropy(pdf))

        centroid = float(np.sum(freqs * pdf))
        out["centroid"] = centroid

        out["bandwidth"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * pdf)))
        out["dominant_freq"] = float(freqs[int(np.argmax(power))])

        for low, high in SPECTRAL_CONFIG["bands"]:
            idx_low = int(np.searchsorted(freqs, low))
            idx_high = int(np.searchsorted(freqs, high))
            out[f"band_{low}_{high}"] = float(np.sum(power[idx_low:idx_high]) / total_power)

        return out
    except Exception:
        return out


def compute_autocorr_lag1(signal: np.ndarray) -> float:
    if signal.size < 3:
        return 0.0
    try:
        x = np.nan_to_num(signal, nan=0.0)
        x = x - float(np.mean(x))
        c0 = float(np.dot(x, x))
        if c0 < 1e-12:
            return 0.0
        c1 = float(np.dot(x[:-1], x[1:]))
        return float(c1 / c0)
    except Exception:
        return 0.0


def _compute_temporal_features_from_arrays(acc_w: np.ndarray, ang_v: np.ndarray, time_delta: float) -> dict:
    acc_w_norm = np.linalg.norm(acc_w, axis=1)
    ang_v_norm = np.linalg.norm(ang_v, axis=1)

    n = int(acc_w.shape[0])
    head_n = min(TEMP_HEAD_N, n)
    tail_n = min(TEMP_TAIL_N, n)

    def _mean_std(x: np.ndarray) -> tuple[float, float]:
        if x.size == 0:
            return 0.0, 0.0
        return float(np.mean(x)), float(np.std(x))

    h_acc_m, h_acc_s = _mean_std(acc_w_norm[:head_n])
    h_ang_m, h_ang_s = _mean_std(ang_v_norm[:head_n])
    t_acc_m, t_acc_s = _mean_std(acc_w_norm[-tail_n:])
    t_ang_m, t_ang_s = _mean_std(ang_v_norm[-tail_n:])

    # jerk
    if n >= 2:
        jerk = np.diff(acc_w, axis=0) / float(time_delta)
        jerk_norm = np.linalg.norm(jerk, axis=1)
        j_mean, j_std = _mean_std(jerk_norm)
        j_max = float(np.max(jerk_norm)) if jerk_norm.size else 0.0
        j_energy = float(np.mean(jerk_norm * jerk_norm)) if jerk_norm.size else 0.0
    else:
        j_mean = j_std = j_max = j_energy = 0.0

    # peaks
    def _peaks(x: np.ndarray) -> float:
        if x.size < 3:
            return 0.0
        return float(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))) / max(1.0, float(x.size))

    p0 = _peaks(acc_w_norm)
    p1 = _peaks(ang_v_norm)

    # zcr on axes
    axes = np.column_stack([acc_w[:, 0], acc_w[:, 1], acc_w[:, 2], ang_v[:, 0], ang_v[:, 1], ang_v[:, 2]])

    def _zcr(col: np.ndarray) -> float:
        if col.size < 2:
            return 0.0
        s = np.sign(col)
        s[np.abs(col) <= TEMP_EPS] = 0
        nz = s[s != 0]
        if nz.size < 2:
            return 0.0
        return float(np.sum(nz[1:] != nz[:-1])) / max(1.0, float(col.size))

    out = {
        "temp__acc_w_norm_head_mean": h_acc_m,
        "temp__acc_w_norm_head_std": h_acc_s,
        "temp__ang_v_norm_head_mean": h_ang_m,
        "temp__ang_v_norm_head_std": h_ang_s,
        "temp__acc_w_norm_tail_mean": t_acc_m,
        "temp__acc_w_norm_tail_std": t_acc_s,
        "temp__ang_v_norm_tail_mean": t_ang_m,
        "temp__ang_v_norm_tail_std": t_ang_s,
        "temp__jerk_mean": j_mean,
        "temp__jerk_std": j_std,
        "temp__jerk_max": j_max,
        "temp__jerk_energy": j_energy,
        "temp__acc_w_norm_peaks_per_step": p0,
        "temp__ang_v_norm_peaks_per_step": p1,
        "temp__corr_acc_ang": float(np.corrcoef(acc_w_norm, ang_v_norm)[0, 1])
        if n > 1 and np.std(acc_w_norm) > 1e-9 and np.std(ang_v_norm) > 1e-9
        else 0.0,
    }

    ax_names = ["acc_w_x", "acc_w_y", "acc_w_z", "ang_v_x", "ang_v_y", "ang_v_z"]
    for j, nm in enumerate(ax_names):
        out[f"temp__{nm}_zcr"] = _zcr(axes[:, j])

    # Spectral + autocorr (only if the models ask for it)
    if NEED_TEMP_SPECTRAL:
        spec_acc = compute_spectral_features(acc_w_norm, fs=float(SPECTRAL_CONFIG["fs"]))
        spec_ang = compute_spectral_features(ang_v_norm, fs=float(SPECTRAL_CONFIG["fs"]))

        for k, v in spec_acc.items():
            out[f"temp__acc_w_norm_{k}"] = float(v)
        for k, v in spec_ang.items():
            out[f"temp__ang_v_norm_{k}"] = float(v)

        out["temp__acc_w_norm_autocorr"] = float(compute_autocorr_lag1(acc_w_norm))
        out["temp__ang_v_norm_autocorr"] = float(compute_autocorr_lag1(ang_v_norm))

    return out


def featurize_sequence_df(
    sequence: pl.DataFrame,
    demos: pl.DataFrame,
    use_all_sensors: bool,
    time_delta: float = 1 / 200,
    mirror: bool = False,
) -> pd.DataFrame:
    load_artifacts()

    base_cols = SENSOR_COLS_ALL if use_all_sensors else IMU_COLS
    sensor_cols = list(base_cols) + list(DERIVED_COLS)

    n_rows = int(sequence.height)

    base_vals = _select_numpy(sequence, base_cols).astype(np.float64, copy=True)

    if use_all_sensors:
        base_is_missing = np.array([c in NEG1_MISSING_ALL for c in base_cols], dtype=bool)
        if base_is_missing.any():
            base_vals[:, base_is_missing] = np.where(base_vals[:, base_is_missing] <= -0.5, np.nan, base_vals[:, base_is_missing])

    acc = _select_numpy(sequence, ["acc_x", "acc_y", "acc_z"]).astype(np.float64, copy=True)
    rot = _select_numpy(sequence, ["rot_w", "rot_x", "rot_y", "rot_z"]).astype(np.float64, copy=True)
    rot = handle_quaternion_missing_values(rot)

    if mirror:
        acc, rot = mirror_acc_and_quaternion(acc, rot, axis="x")
        rot = handle_quaternion_missing_values(rot)

    quat_norm = np.linalg.norm(rot, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)

    acc_w = compute_world_acceleration(acc, rot)
    acc_w_norm = np.linalg.norm(acc_w, axis=1)

    ang_v = compute_angular_velocity(rot, time_delta=time_delta)
    ang_v_norm = np.linalg.norm(ang_v, axis=1)

    rotvec_step, rot_angle = compute_rot_step_features(rot, time_delta=time_delta)
    rotvec_step_norm = np.linalg.norm(rotvec_step, axis=1)

    derived = np.column_stack(
        [
            acc_norm,
            acc_w[:, 0],
            acc_w[:, 1],
            acc_w[:, 2],
            acc_w_norm,
            ang_v[:, 0],
            ang_v[:, 1],
            ang_v[:, 2],
            ang_v_norm,
            quat_norm,
            rot_angle,
            rotvec_step[:, 0],
            rotvec_step[:, 1],
            rotvec_step[:, 2],
            rotvec_step_norm,
        ]
    ).astype(np.float64, copy=False)

    vals = np.concatenate([base_vals, derived], axis=1)

    counts = (~np.isnan(vals)).sum(axis=0).astype(np.float64)
    means = np.divide(np.nansum(vals, axis=0), counts, out=np.zeros_like(counts), where=counts > 0)
    stds = np.nanstd(vals, axis=0)
    vmin = np.nanmin(vals, axis=0)
    vmax = np.nanmax(vals, axis=0)

    vmin = np.where(np.isnan(vmin), 0.0, vmin)
    vmax = np.where(np.isnan(vmax), 0.0, vmax)
    miss_frac = 1.0 - np.divide(counts, float(max(n_rows, 1)), out=np.zeros_like(counts), where=True)

    if n_rows >= 2:
        v1 = vals[:-1]
        v2 = vals[1:]
        valid_pair = (~np.isnan(v1)) & (~np.isnan(v2))
        diffs = np.abs(v2 - v1)
        diff_sum = np.nansum(np.where(valid_pair, diffs, np.nan), axis=0)
        diff_cnt = valid_pair.sum(axis=0).astype(np.float64)
        diff_mean = np.divide(diff_sum, diff_cnt, out=np.zeros_like(diff_sum), where=diff_cnt > 0)
    else:
        diff_mean = np.zeros(vals.shape[1], dtype=np.float64)

    skew_arr = None
    kurt_arr = None
    if NEED_SKEW_KURT:
        # Compute only when models actually contain __skew/__kurt columns
        try:
            skew_arr = stats.skew(vals, axis=0, nan_policy="omit", bias=False)
            kurt_arr = stats.kurtosis(vals, axis=0, nan_policy="omit", fisher=True, bias=False)
            skew_arr = np.where(np.isfinite(skew_arr), skew_arr, 0.0)
            kurt_arr = np.where(np.isfinite(kurt_arr), kurt_arr, 0.0)
        except Exception:
            skew_arr = None
            kurt_arr = None

    feats: dict[str, object] = {"sequence_len": int(n_rows)}
    for i, c in enumerate(sensor_cols):
        feats[f"{c}__mean"] = float(means[i]) if np.isfinite(means[i]) else 0.0
        feats[f"{c}__std"] = float(stds[i]) if np.isfinite(stds[i]) else 0.0
        feats[f"{c}__min"] = float(vmin[i]) if np.isfinite(vmin[i]) else 0.0
        feats[f"{c}__max"] = float(vmax[i]) if np.isfinite(vmax[i]) else 0.0
        feats[f"{c}__diff_mean"] = float(diff_mean[i]) if np.isfinite(diff_mean[i]) else 0.0
        feats[f"{c}__miss_frac"] = float(miss_frac[i]) if np.isfinite(miss_frac[i]) else 0.0
        if NEED_SKEW_KURT and skew_arr is not None and kurt_arr is not None:
            feats[f"{c}__skew"] = float(skew_arr[i])
            feats[f"{c}__kurt"] = float(kurt_arr[i])

    feats.update(_compute_temporal_features_from_arrays(acc_w=acc_w, ang_v=ang_v, time_delta=time_delta))

    # Demographics
    if demos is not None and demos.height:
        for c in DEMO_COLS:
            try:
                v = demos.select(c).item(0)
                feats[f"demo__{c}"] = float(v) if v is not None else 0.0
            except Exception:
                feats[f"demo__{c}"] = 0.0
    else:
        for c in DEMO_COLS:
            feats[f"demo__{c}"] = 0.0

    return pd.DataFrame([feats])


def _has_extra_sensors(sequence: pl.DataFrame) -> bool:
    load_artifacts()

    cols = set(sequence.columns)

    # THM: if any THM column exists and any value is present (>-0.5)
    thm_present = [c for c in THM_COLS if c in cols]
    if thm_present:
        thm_vals = sequence.select(thm_present).to_numpy()
        if np.any(thm_vals > -0.5):
            return True

    # TOF: probe subset for speed
    tof_present = [c for c in TOF_COLS[:64] if c in cols]
    if tof_present:
        tof_vals = sequence.select(tof_present).to_numpy()
        if np.any(tof_vals > -0.5):
            return True

    return False


def _avg_proba(models, X: pd.DataFrame) -> np.ndarray:
    probs = None
    for m in models:
        p = m.predict_proba(X)
        probs = p if probs is None else (probs + p)
    return probs / float(len(models))


def _apply_subject_norm_row(X: pd.DataFrame, subject: str, use_all: bool) -> pd.DataFrame:
    if subject_norm is None:
        return X

    Xn = X.copy()
    if use_all:
        cols = subject_norm.get("cols_all", [])
        mean_df = subject_norm.get("mean_all", None)
        std_df = subject_norm.get("std_all", None)
        gmean = subject_norm.get("gmean_all", None)
        gstd = subject_norm.get("gstd_all", None)
    else:
        cols = subject_norm.get("cols_imu", [])
        mean_df = subject_norm.get("mean_imu", None)
        std_df = subject_norm.get("std_imu", None)
        gmean = subject_norm.get("gmean_imu", None)
        gstd = subject_norm.get("gstd_imu", None)

    if mean_df is None or std_df is None or gmean is None or gstd is None or len(cols) == 0:
        return Xn

    cols = [c for c in cols if c in Xn.columns]
    if len(cols) == 0:
        return Xn

    if subject in mean_df.index:
        m = mean_df.loc[subject, cols].to_numpy(dtype=np.float64, copy=False).reshape(1, -1)
        s = std_df.loc[subject, cols].to_numpy(dtype=np.float64, copy=False).reshape(1, -1)
    else:
        m = gmean[cols].to_numpy(dtype=np.float64, copy=False).reshape(1, -1)
        s = gstd[cols].to_numpy(dtype=np.float64, copy=False).reshape(1, -1)

    s = np.where(np.isfinite(s) & (s > 1e-12), s, 1.0)
    Xv = Xn[cols].to_numpy(dtype=np.float64, copy=True)
    Xn[cols] = (Xv - m) / s
    return Xn


def _predict_proba_views(sequence: pl.DataFrame, demos: pl.DataFrame, time_delta: float, mirror: bool) -> tuple[float, np.ndarray]:
    load_artifacts()

    use_all_flag = _has_extra_sensors(sequence)

    X_all = featurize_sequence_df(sequence, demos, use_all_sensors=True, time_delta=time_delta, mirror=mirror)
    X_imu = featurize_sequence_df(sequence, demos, use_all_sensors=False, time_delta=time_delta, mirror=mirror)

    X_all = X_all.reindex(columns=feature_columns_all, fill_value=0.0)
    X_imu = X_imu.reindex(columns=feature_columns_imu, fill_value=0.0)

    subj = ""
    if "subject" in sequence.columns and sequence.height:
        try:
            subj = str(sequence.select("subject").to_series()[0])
        except Exception:
            subj = ""

    X_all = _apply_subject_norm_row(X_all, subj, use_all=True)
    X_imu = _apply_subject_norm_row(X_imu, subj, use_all=False)

    p_all = float(_avg_proba(bin_models_all, X_all)[:, 1][0])
    p_imu = float(_avg_proba(bin_models_imu, X_imu)[:, 1][0])

    if use_stacking and stack_bin is not None:
        p_bin = float(stack_bin.predict_proba([[p_all, p_imu, int(use_all_flag)]])[:, 1][0])
    else:
        p_bin = p_all if use_all_flag else p_imu

    if calibrator is not None:
        try:
            p_bin = float(calibrator.predict([p_bin])[0])
        except Exception:
            p_bin = float(p_bin)

    proba_all = _avg_proba(multi_models_all, X_all)
    proba_imu = _avg_proba(multi_models_imu, X_imu)

    if use_stacking and stack_multi is not None:
        stack_x = np.concatenate([proba_all, proba_imu, np.array([[float(use_all_flag)]])], axis=1)
        p_mul = stack_multi.predict_proba(stack_x)
    else:
        p_mul = proba_all if use_all_flag else proba_imu

    return p_bin, p_mul[0]


def predict_gesture(sequence: pl.DataFrame, demos: pl.DataFrame, time_delta: float = 1 / 200) -> str:
    load_artifacts()

    is_left = False
    if demos is not None and demos.height and ("handedness" in demos.columns):
        try:
            is_left = _is_left_handed(demos.select("handedness").to_series()[0])
        except Exception:
            is_left = False

    if is_left:
        p0, m0 = _predict_proba_views(sequence, demos, time_delta=time_delta, mirror=False)
        p1, m1 = _predict_proba_views(sequence, demos, time_delta=time_delta, mirror=True)
        p_bin = 0.5 * (p0 + p1)
        p_mul = 0.5 * (m0 + m1)
    else:
        p_bin, p_mul = _predict_proba_views(sequence, demos, time_delta=time_delta, mirror=False)

    if float(p_bin) < float(best_thr):
        return str(fallback_non_target)

    if not le_gesture_classes:
        return str(fallback_non_target)

    pred_idx = int(np.argmax(p_mul))
    pred_idx = max(0, min(pred_idx, len(le_gesture_classes) - 1))
    return str(le_gesture_classes[pred_idx])



# Kaggle Inference Server Setup
try:
    import kaggle_evaluation.cmi_inference_server
    
    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
        """
        Main prediction entry point for the evaluation server.
        Wraps the optimized predict_gesture function.
        """
        try:
            # Use the optimized prediction function
            return predict_gesture(sequence, demographics)
        except Exception as e:
            # Fallback in case of any error
            print(f"Prediction error: {e}")
            return "Text on phone"
    
    # Create inference server
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    
    # Run based on environment
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Competition evaluation mode
        inference_server.serve()
    else:
        # Kaggle notebook testing mode
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )
    
except ModuleNotFoundError:
    print("✅ Ready for Kaggle - upload 'joydeep_model_artifacts.joblib' as dataset 'joydeep-gesture-model'")

