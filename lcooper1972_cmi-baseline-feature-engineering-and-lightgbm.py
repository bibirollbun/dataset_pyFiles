import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy, kurtosis, skew
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Whether to generate a new model.
GEN_NEW_MODEL = False

RANDOM_SEED = 42
DATA_DIR = Path("../kaggle/input/cmi-detect-behavior-with-sensor-data").resolve()
MODELS_DIR = Path("../models").resolve()

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    DATA_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data").resolve()
    if GEN_NEW_MODEL:
        MODELS_DIR = Path("/kaggle/working/models").resolve()
    else:
        MODELS_DIR = Path("/kaggle/input/cmi-lightgbm-baseline").resolve()

MODELS_DIR.mkdir(exist_ok=True, parents=True)


train = pd.read_csv(DATA_DIR.joinpath("train.csv"))
train_demo = pd.read_csv(DATA_DIR.joinpath("train_demographics.csv"))
train = train.merge(train_demo, on="subject", how="left")

train.head()


def get_agg_stats_features(df: pd.DataFrame, segment_idx: int) -> dict[str, np.float32]:
    """Returns the time-domain statistical aggregates for each time-series feature in the data frame.

    Args:
        df (pd.DataFrame): Input data frame.
        segment_idx (int): Index of the segment of the data frame.

    Returns:
        dict[str, np.float32]: Mapping from all aggregate feature names to their values.
    """
    epsilon = 1e-8
    features: dict[str, np.float32] = {}

    for col in df.columns:
        data = df[col].to_numpy()
        features[f"{col}_{segment_idx}_mean"] = np.mean(data)
        features[f"{col}_{segment_idx}_std"] = np.std(data)
        features[f"{col}_{segment_idx}_min"] = np.min(data)
        features[f"{col}_{segment_idx}_max"] = np.max(data)
        features[f"{col}_{segment_idx}_median"] = np.median(data)
        features[f"{col}_{segment_idx}_range"] = (
            features[f"{col}_{segment_idx}_max"] - features[f"{col}_{segment_idx}_min"]
        )

        if np.std(data) > epsilon:
            features[f"{col}_{segment_idx}_skew"] = skew(data)
            features[f"{col}_{segment_idx}_kurtosis"] = kurtosis(data)
        else:
            features[f"{col}_{segment_idx}_skew"] = np.float32(0.0)
            features[f"{col}_{segment_idx}_kurtosis"] = np.float32(0.0)

    return features


def get_spectral_features(
    name: str, segment_idx: int, signal: np.ndarray, epsilon: float = 1e-12, min_freq: float = 0.1, num_bands: int = 4
) -> dict[str, np.float32]:
    """Gets the spectral features from a 1-D time-series signal of sensor data.

    Args:
        name (str): Name of the column.
        segment_idx (int): Index of the segment of the data frame.
        signal (np.ndarray): 1-D time-series signal of sensor data.
        epsilon (float, optional): Prevents division by zero. Defaults to 1e-12.
        min_freq (float, optional): Lower limit for the high-pass filter. Defaults to 0.1.
        num_bands (int, optional): Number of bands for which to split the frequencies when analyzing the energy per
            band. Defaults to 4.

    Returns:
        dict[str, np.float32]: Spectral features from a 1-D time-series signal of sensor data.
    """
    nyquist_frequency = 0.5
    full_name = f"{name}_{segment_idx}"

    if min_freq <= 0 or min_freq > nyquist_frequency:
        raise ValueError(
            f"Lower limit for the high-pass frequency filter {min_freq} is not greater than 0 or is above the Nyquist "
            f"frequency {nyquist_frequency}."
        )

    band_len = (nyquist_frequency - min_freq) / num_bands
    band_edges = [(min_freq + i * band_len, min_freq + (i + 1) * band_len) for i in range(num_bands)]
    bands = {f"{full_name}_energy_part{i}": edges for i, edges in enumerate(band_edges)}

    n = len(signal)
    if n < 2:
        return {
            f"{full_name}_spec_centroid": np.float32(0.0),
            f"{full_name}_spec_bandwidth": np.float32(0.0),
            f"{full_name}_spec_entropy": np.float32(0.0),
            f"{full_name}_dom_freq": np.float32(0.0),
            f"{full_name}_dom_freq_mag": np.float32(0.0),
            **{key: np.float32(0.0) for key in bands},
        }

    freqs = rfftfreq(n, d=1)
    fft_magnitudes = np.abs(np.array(rfft(signal)))
    power_spectrum = fft_magnitudes**2

    # Filter out frequencies below the lower limit of the high-pass filter.
    valid = freqs >= min_freq
    freqs = freqs[valid]
    power_spectrum = power_spectrum[valid]
    fft_magnitudes = fft_magnitudes[valid]

    total_power = np.sum(power_spectrum) + epsilon
    norm_power = power_spectrum / total_power
    centroid = np.sum(freqs * norm_power)
    bandwidth = np.sqrt(np.dot(norm_power, (freqs - centroid) ** 2))
    spec_entropy = np.float32(entropy(norm_power))
    dom_idx = np.argmax(power_spectrum)
    dom_freq = freqs[dom_idx]
    dom_mag = fft_magnitudes[dom_idx]

    band_energies = {}
    for key, (f_low, f_high) in bands.items():
        idxs = np.where((freqs >= f_low) & (freqs < f_high))[0]
        band_energy = np.sum(norm_power[idxs])
        band_energies[key] = band_energy

    return {
        f"{full_name}_spec_centroid": centroid,
        f"{full_name}_spec_bandwidth": bandwidth,
        f"{full_name}_spec_entropy": spec_entropy,
        f"{full_name}_dom_freq": dom_freq,
        f"{full_name}_dom_freq_mag": dom_mag,
        **band_energies,
    }


def preprocess_imu_cols(seq_df: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
    """Preprocess the IMU columns with interpolation and normalization.

    Args:
        seq_df (pd.DataFrame): Raw sequence data frame.
        epsilon (float, optional): Prevents division by zero. Defaults to 1e-8.

    Returns:
        pd.DataFrame: Preprocessed data fram.
    """
    imu_cols = [col for col in seq_df.columns if col.startswith(("acc_", "rot_"))]
    df = seq_df.copy().sort_values("sequence_counter").reset_index(drop=True)[imu_cols]
    df = df.interpolate(method="linear", limit_direction="both")
    return (df - df.mean()) / (df.std() + epsilon)


def split_df(df: pd.DataFrame, num_segments: int = 6) -> list[pd.DataFrame]:
    """Splits the data frame row-wise into segments of (approximately) equal length.

    Args:
        df (pd.DataFrame): Preprocessed data frame.
        num_segments (int, optional): Number of segments to split the data frame into. Defaults to 6.

    Returns:
        list[pd.DataFrame]: Data frame split into the specified number of segments.
    """
    segment_length = len(df) // num_segments
    segments = [
        df.iloc[i * segment_length : (i + 1) * segment_length].reset_index(drop=True) for i in range(num_segments - 1)
    ]
    segments.append(df.iloc[(num_segments - 1) * segment_length :].reset_index(drop=True))
    return segments


def get_processed_imu_features(seq_df: pd.DataFrame, epsilon: float = 1e-8, num_segments: int = 6) -> pd.DataFrame:
    """Returns the processed features for our baseline model given our raw sequence data frame.

    Args:
        seq_df (pd.DataFrame): Raw sequence data frame.
        epsilon (float, optional): Prevents division by zero. Defaults to 1e-8.
        num_segments (int, optional): Number of segments to split the data frame into. Defaults to 6.

    Returns:
        pd.DataFrame: Processed features.
    """
    df = preprocess_imu_cols(seq_df, epsilon)
    segments = split_df(df, num_segments)
    features: dict[str, np.float32] = {}

    for i, segment in enumerate(segments):
        features.update(get_agg_stats_features(segment, i))
        if i > num_segments // 2:
            for col in segment.columns:
                features.update(get_spectral_features(col, i, segment[col].to_numpy()))

    return pd.DataFrame([features])


if GEN_NEW_MODEL:
    # Adds a progress bar.
    tqdm.pandas()

    seq_groups = list(train.groupby("sequence_id"))

    # Parallelize processing the training data frame across all available cores.
    results = Parallel(n_jobs=-1)(
        delayed(lambda group: (get_processed_imu_features(group[1]), group[1].iloc[0]["gesture"]))(group)
        for group in tqdm(seq_groups)
    )

    all_features, all_labels = zip(*results, strict=False)
    X: pd.DataFrame = pd.concat(all_features, ignore_index=True)
    y = np.array(all_labels)

    X.to_csv(MODELS_DIR.joinpath("X_features.csv"), index=False)
    pd.DataFrame({"gesture": y}).to_csv(MODELS_DIR.joinpath("y_labels.csv"), index=False)
else:
    X = pd.read_csv(MODELS_DIR.joinpath("X_features.csv"))
    y = pd.read_csv(MODELS_DIR.joinpath("y_labels.csv"))["gesture"].to_numpy()

X.head()


le = LabelEncoder()
y_encoded = le.fit_transform(y)


def objective(trial: optuna.Trial) -> float:
    """Returns the mean F1 score from cross validation of the training data for tuning hyperparameters.

    Args:
        trial (optuna.Trial): Hyperparameter tuning trial.

    Returns:
        float: Mean F1 score from cross validation of the training data.
    """
    param = {
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
        "metric": "None",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "early_stopping_round": 30,
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    f1_scores = []
    y_arr = np.array(y_encoded)

    for train_idx, val_idx in skf.split(X, y_arr):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="multi_logloss")

        preds = np.array(model.predict(X_val))
        f1 = f1_score(y_val, preds, average="macro")
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


if GEN_NEW_MODEL:
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=5, show_progress_bar=True)
    MODELS_DIR.joinpath("best_lgbm_params.json").write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    best_params = study.best_params
else:
    best_params = json.loads(MODELS_DIR.joinpath("best_lgbm_params.json").read_text(encoding="utf-8"))


best_params.update(
    {
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
)

final_model = lgb.LGBMClassifier(**best_params)


if GEN_NEW_MODEL:
    feature_names = X.columns.tolist()
    final_model.fit(X, y_encoded)
    joblib.dump(final_model, MODELS_DIR.joinpath("baseline_model.pkl"))
    joblib.dump(le, MODELS_DIR.joinpath("label_encoder.pkl"))
    joblib.dump(feature_names, MODELS_DIR.joinpath("feature_names.pkl"))
else:
    final_model: lgb.LGBMClassifier = joblib.load(MODELS_DIR.joinpath("baseline_model.pkl"))
    feature_names: list[str] = joblib.load(MODELS_DIR.joinpath("feature_names.pkl"))
    le: LabelEncoder = joblib.load(MODELS_DIR.joinpath("label_encoder.pkl"))


def predict(sequence: pd.DataFrame) -> str:
    """Predicts the gesture based on a sequence of IMU data.

    Args:
        sequence (pd.DataFrame): Sequence of IMU data.

    Returns:
        str: Predicted gesture.
    """
    X = get_processed_imu_features(sequence)
    X = X.reindex(columns=feature_names, fill_value=0)
    pred = np.array(final_model.predict(X))[0]
    return le.inverse_transform([pred])[0]


test = pd.read_csv(DATA_DIR.joinpath("test.csv"))
preds: list[str] = []

for _, group in test.groupby("sequence_id"):
    preds.append(predict(group))

preds




