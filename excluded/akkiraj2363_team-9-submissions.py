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


# install pqdm for parallel processing
!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm

import pandas as pd
import numpy as np
from tqdm import tqdm
from pqdm.threads import pqdm
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the Ariel data processing pipeline."""
    DATA_PATH: str = '/kaggle/input/ariel-data-challenge-2025'
    DATASET: str = "test"
    SCALE: float = 0.95
    SIGMA: float = 0.0009
    CUT_INF: int = 39
    CUT_SUP: int = 321
    MODEL_PHASE_DETECTION_SLICE: slice = field(default_factory=lambda: slice(30, 140))
    MODEL_OPTIMIZATION_DELTA: int = 11
    MODEL_POLYNOMIAL_DEGREE: int = 3
    N_JOBS: int = 3
    LOG_HOT_STATS: bool = False
    
    # Sensor configurations
    SENSOR_CONFIG: Dict = field(default_factory=lambda: {
        "AIRS-CH0": {
            "raw_shape": [11250, 32, 356],
            "calibrated_shape": [1, 32, 321 - 39],  # CUT_SUP - CUT_INF
            "linear_corr_shape": (6, 32, 356),
            "dt_pattern": (0.1, 4.5), 
            "binning": 30,
            "roi_y": (10, 22),
            "roi_x": None
        },
        "FGS1": {
            "raw_shape": [135000, 32, 32],
            "calibrated_shape": [1, 32, 32],
            "linear_corr_shape": (6, 32, 32),
            "dt_pattern": (0.1, 0.1),
            "binning": 30 * 12,
            "roi_y": (10, 22),
            "roi_x": (10, 22)
        }
    })


class SignalProcessor:
    """Processes sensor signals for multiple planets."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(
            f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', 
            index_col='planet_id'
        ).index.astype(int)
        self.stats = []  # For logging statistics

    def _apply_linear_corr(self, linear_corr: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Apply linearity correction to the signal.
        
        Args:
            linear_corr: Coefficients for linearity correction
            signal: Input signal to correct
            
        Returns:
            Corrected signal
        """
        coeffs = np.flip(linear_corr, axis=0)  # shape: (D, X, Y), D - highest degree first
        x = signal.astype(np.float64, copy=False)
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]  # broadcast (X,Y) -> (T,X,Y)
        
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)  # in-place multiplication
            out += coeffs[k]  # broadcast (X,Y)

        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id: int, sensor: str) -> np.ndarray:
        """
        Calibrate a single sensor signal for a planet.
        
        Args:
            planet_id: ID of the planet to process
            sensor: Sensor name ('AIRS-CH0' or 'FGS1')
            
        Returns:
            Calibrated signal
        """
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        
        # Load data
        signal = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_signal_0.parquet"
        ).to_numpy()
        
        dark = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dark.parquet"
        ).to_numpy()
        
        dead = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dead.parquet"
        ).to_numpy()
        
        flat = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/flat.parquet"
        ).to_numpy()
        
        linear_corr = pd.read_parquet(
            f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet"
        ).values.astype(np.float64).reshape(sensor_cfg["linear_corr_shape"])

        # Reshape & apply ADC correction
        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        # Identify hot pixels (for monitoring only, not masking)
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask

        # Apply sensor-specific cropping
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]

        elif sensor == "FGS1":
            y0, y1 = sensor_cfg["roi_y"]
            x0, x1 = sensor_cfg["roi_x"]
            signal = signal[:, y0:y1, x0:x1]
            dark = dark[y0:y1, x0:x1]
            dead = dead[y0:y1, x0:x1]
            flat = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot = hot[y0:y1, x0:x1]

        # Non-negative clamp before linearity correction
        np.maximum(signal, 0, out=signal)

        # Apply linearity correction
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            y0, y1 = sensor_cfg["roi_y"]
            sl = (slice(None), slice(y0, y1), slice(None))  # T, Y, λ
            signal[sl] = self._apply_linear_corr(linear_corr[:, y0:y1, :], signal[sl])

        # Dark subtraction with integration pattern consideration
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale = base_dt + increment
        signal[::2] -= dark * even_scale
        signal[1::2] -= dark * odd_scale

        # Apply flat field correction (excluding hot pixels from mask)
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()
            bad = dead | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal /= flat_roi

        elif sensor == "AIRS-CH0":
            y0, y1 = sensor_cfg["roi_y"]
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()
            bad = dead[y0:y1, :] | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi

        # Log statistics if enabled
        if self.cfg.LOG_HOT_STATS:
            self.stats.append({
                "planet_id": int(planet_id),
                "sensor": sensor,
                "hot_frac": float(np.mean(hot)),
                "dead_frac": float(np.mean(dead)),
            })

        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal: np.ndarray, sensor: str) -> np.ndarray:
        """
        Preprocess calibrated signal by binning and applying weights.
        
        Args:
            calibrated_signal: Calibrated signal from sensor
            sensor: Sensor name
            
        Returns:
            Preprocessed signal
        """
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        # Extract region of interest
        if sensor == "AIRS-CH0":
            y0, y1 = sensor_cfg["roi_y"]
            signal_roi = calibrated_signal[:, y0:y1, :]
        elif sensor == "FGS1":
            y0, y1 = sensor_cfg["roi_y"]
            x0, x1 = sensor_cfg["roi_x"]
            signal_roi = calibrated_signal[:, y0:y1, x0:x1]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)

        # Calculate mean signal and CDS
        mean_signal = np.nanmean(signal_roi, axis=1)
        cds_signal = mean_signal[1::2] + mean_signal[0::2]

        # Bin the signal
        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        # Apply winsorization for AIRS
        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)
            np.clip(binned, q_lo, q_hi, out=binned)

        # Reshape FGS1 data
        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        # Apply inverse variance weighting for AIRS
        if sensor == "AIRS-CH0":
            var = np.nanvar(binned, axis=0, ddof=1)
            med = np.nanmedian(var)
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            # Clip weights to prevent dominance by single channel
            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            # Normalize weights
            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

            # Apply weights
            binned *= w[None, :]

        return binned

    def _process_planet_sensor(self, args: dict) -> np.ndarray:
        """Wrapper function for parallel processing."""
        planet_id, sensor = args['planet_id'], args['sensor']
        calibrated = self._calibrate_single_signal(planet_id, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self) -> np.ndarray:
        """Process all data for all planets and sensors."""
        logger.info("Processing FGS1 data...")
        args_fgs1 = [{"planet_id": planet_id, "sensor": "FGS1"} for planet_id in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS, desc="FGS1 Processing")

        logger.info("Processing AIRS-CH0 data...")
        args_airs_ch0 = [{"planet_id": planet_id, "sensor": "AIRS-CH0"} for planet_id in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS, desc="AIRS Processing")

        # Combine processed data
        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        
        return preprocessed_signal


class TransitModel:
    """Models transit signals to estimate depth."""
    
    def __init__(self, config: Config):
        self.cfg = config

    def _phase_detector(self, signal: np.ndarray) -> Tuple[int, int]:
        """
        Detect transit phases in the signal.
        
        Args:
            signal: Input signal
            
        Returns:
            Start and end indices of the transit
        """
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]

        grad1 = np.gradient(signal1)
        if grad1.max() != 0:
            grad1 /= grad1.max()
        
        grad2 = np.gradient(signal2)
        if grad2.max() != 0:
            grad2 /= grad2.max()

        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index

        return phase1, phase2
    
    def _objective_function(self, s: float, signal: np.ndarray, phase1: int, phase2: int) -> float:
        """
        Objective function for transit depth optimization.
        
        Args:
            s: Transit depth parameter
            signal: Input signal
            phase1: Start of transit
            phase2: End of transit
            
        Returns:
            Error metric
        """
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE

        # Adjust delta if phases are too close to boundaries
        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
            delta = min(2, phase1, len(signal) - phase2 - 1)

        # Create modified signal with transit applied
        y = np.concatenate([
            signal[: phase1 - delta],
            signal[phase1 + delta : phase2 - delta] * (1 + s),
            signal[phase2 + delta :]
        ])
        x = np.arange(len(y))

        # Fit polynomial and calculate error
        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        
        return error

    def predict(self, single_preprocessed_signal: np.ndarray) -> float:
        """
        Predict transit depth for a single signal.
        
        Args:
            single_preprocessed_signal: Preprocessed signal
            
        Returns:
            Estimated transit depth
        """
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
        signal_1d = savgol_filter(signal_1d, 23, 2)
        
        phase1, phase2 = self._phase_detector(signal_1d)

        # Ensure phases are within valid bounds
        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)

        # Optimize transit depth
        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead",
            options={'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        return result.x[0]

    def predict_all(self, preprocessed_signals: List[np.ndarray]) -> np.ndarray:
        """
        Predict transit depths for all signals.
        
        Args:
            preprocessed_signals: List of preprocessed signals
            
        Returns:
            Array of predicted transit depths
        """
        predictions = [
            self.predict(preprocessed_signal)
            for preprocessed_signal in tqdm(preprocessed_signals, desc="Transit Depth Prediction")
        ]
        return np.array(predictions) * self.cfg.SCALE


def estimate_sigma_fgs(preprocessed_data: List[np.ndarray], cfg: Config) -> np.ndarray:
    """
    Estimate sigma values for FGS1 sensor.
    
    Args:
        preprocessed_data: List of preprocessed signals
        cfg: Configuration object
        
    Returns:
        Array of sigma values for FGS1
    """
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12
    
    for single in preprocessed_data:
        # Use AIRS white curve for phase detection
        air_white = savgol_filter(single[:, 1:].mean(axis=1), 20, 2)
        p1, p2 = _phase_detector_signal(air_white, cfg)
        p1 = max(delta, p1)
        p2 = min(len(air_white) - delta - 1, p2)

        fgs = single[:, 0]
        oot = (fgs[: p1 - delta] if p1 - delta > 0 else np.empty(0, fgs.dtype))
        if p2 + delta < fgs.size:
            oot = np.concatenate([oot, fgs[p2 + delta :]])
        inn = fgs[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan)
            continue

        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(fgs))
        
        # Relative uncertainty of depth
        sigma_rel = np.sqrt(var_oot / max(n_oot, 1) + var_in / max(n_in, 1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # Soft multiplier with clipping
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)

    return k * cfg.SIGMA


def estimate_sigma_air(preprocessed_data: List[np.ndarray], cfg: Config) -> np.ndarray:
    """
    Estimate sigma values for AIRS sensor.
    
    Args:
        preprocessed_data: List of preprocessed signals
        cfg: Configuration object
        
    Returns:
        Array of sigma values for AIRS
    """
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        # White curve from AIRS binned data
        white = np.nanmean(single[:, 1:], axis=1)
        white_s = savgol_filter(white, 20, 2)

        p1, p2 = _phase_detector_signal(white_s, cfg)
        p1 = max(delta, p1)
        p2 = min(len(white) - delta - 1, p2)

        oot_left = white[: p1 - delta] if p1 - delta > 0 else np.empty(0, white.dtype)
        oot_right = white[p2 + delta :] if (p2 + delta) < white.size else np.empty(0, white.dtype)
        oot = np.concatenate([oot_left, oot_right]) if (oot_left.size + oot_right.size) else oot_left
        inn = white[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan)
            continue

        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(white))

        sigma_rel = np.sqrt(var_oot / max(n_oot, 1) + var_in / max(n_in, 1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # Soft multiplier with clipping
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.90, 1.20)

    return k * cfg.SIGMA


def _phase_detector_signal(signal: np.ndarray, cfg: Config) -> Tuple[int, int]:
    """
    Detect phases in a signal.
    
    Args:
        signal: Input signal
        cfg: Configuration object
        
    Returns:
        Start and end indices of detected feature
    """
    sl = cfg.MODEL_PHASE_DETECTION_SLICE
    min_idx = int(np.argmin(signal[sl])) + sl.start
    s1 = signal[:min_idx]
    s2 = signal[min_idx:]
    
    if s1.size < 3 or s2.size < 3:
        return 0, len(signal) - 1
        
    g1 = np.gradient(s1)
    g2 = np.gradient(s2)
    
    g1_max = np.max(g1) if np.size(g1) else 0.0
    g2_max = np.max(g2) if np.size(g2) else 0.0
    
    if g1_max != 0: 
        g1 /= g1_max
    if g2_max != 0: 
        g2 /= g2_max
        
    phase1 = int(np.argmin(g1))
    phase2 = int(np.argmax(g2)) + min_idx
    
    return phase1, phase2


class SubmissionGenerator:
    """Generates submission file from predictions."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.sample_submission = pd.read_csv(
            "/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", 
            index_col="planet_id"
        )

    def create(self, predictions: np.ndarray, sigma_fgs: Optional[np.ndarray] = None, 
               sigma_air: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create submission DataFrame.
        
        Args:
            predictions: Array of predicted transit depths
            sigma_fgs: Optional array of sigma values for FGS1
            sigma_air: Optional array of sigma values for AIRS
            
        Returns:
            Submission DataFrame
        """
        planet_ids = self.sample_submission.index
        n_mu = self.sample_submission.shape[1] // 2  # 283

        preds = np.asarray(predictions, dtype=float).reshape(-1)
        mu = np.tile(preds.reshape(-1, 1), (1, n_mu))
        mu = np.clip(mu, 0, None)

        sigmas = np.full_like(mu, self.cfg.SIGMA, dtype=float)
        
        if sigma_fgs is not None:
            sigma_fgs = np.asarray(sigma_fgs, dtype=float).reshape(-1)
            sigmas[:, 0] = np.clip(sigma_fgs, 1e-6, 0.1)
            
        if sigma_air is not None:
            sigma_air = np.asarray(sigma_air, dtype=float).reshape(-1, 1)
            sigmas[:, 1:] = np.clip(sigma_air, 1e-6, 0.1)

        submission_df = pd.DataFrame(
            np.concatenate([mu, sigmas], axis=1),
            columns=self.sample_submission.columns,
            index=planet_ids
        )
        submission_df.to_csv("submission.csv")
        return submission_df


def main():
    """Main execution function."""
    __t0 = time.perf_counter()
    
    # Initialize configuration and components
    config = Config()
    signal_processor = SignalProcessor(config)
    
    # Process data
    logger.info("Starting data processing...")
    preprocessed_data = signal_processor.process_all_data()
    
    # Model predictions
    logger.info("Starting transit modeling...")
    model = TransitModel(config)
    predictions = model.predict_all(preprocessed_data)
    
    # Estimate sigma values
    logger.info("Estimating sigma values...")
    sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
    sigma_air_vec = estimate_sigma_air(preprocessed_data, config)
    
    # Generate submission
    logger.info("Generating submission...")
    submission_generator = SubmissionGenerator(config)
    submission = submission_generator.create(predictions, sigma_fgs=sigma_fgs_vec, sigma_air=sigma_air_vec)
    
    # Print timing information
    __t1 = time.perf_counter()
    elapsed = __t1 - __t0
    logger.info(f"Total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    
    return submission


if __name__ == "__main__":
    submission = main()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import logging
import warnings

# --- New Helper Functions for Plotting ---

def plot_heatmap(data: np.ndarray, title: str, xlabel: str, ylabel: str, aspect: str = 'auto', cmap: str = 'viridis'):
    """Plots a 2D heatmap of sensor data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Ignore "matplotlib.pyplot.figure" warning
        plt.figure(figsize=(12, 6))
        
        # Calculate robust color limits using 1st and 99th percentiles
        v_min, v_max = np.nanpercentile(data, [1, 99])
        
        plt.imshow(data, aspect=aspect, cmap=cmap, interpolation='nearest', vmin=v_min, vmax=v_max)
        plt.colorbar(label='Signal / Weight')
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

def plot_light_curve(time: np.ndarray, flux: np.ndarray, title: str, ylabel: str, color: str = 'blue'):
    """Plots a 1D light curve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.figure(figsize=(12, 5))
        plt.plot(time, flux, 'o', markersize=3, alpha=0.6, color=color)
        plt.title(title, fontsize=16)
        plt.xlabel('Binned Time Step')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

def plot_weights(weights: np.ndarray, title: str):
    """Plots a bar chart of the AIRS channel weights."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(weights)), weights, color='cyan')
        plt.title(title, fontsize=16)
        plt.xlabel('AIRS Wavelength Channel Index')
        plt.ylabel('Calculated Weight (Trust)')
        plt.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        plt.show()

def plot_final_transit_fit(model: TransitModel, signal_1d: np.ndarray, title: str):
    """Re-creates and plots the final transit model fit."""
    cfg = model.cfg
    
    # 1. Get the smoothed signal
    signal_savgol = savgol_filter(signal_1d, 23, 2)
    
    # 2. Get the phases
    phase1, phase2 = model._phase_detector(signal_savgol)
    phase1 = max(cfg.MODEL_OPTIMIZATION_DELTA, phase1)
    phase2 = min(len(signal_savgol) - cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)

    # 3. Get the final optimized depth 's'
    result = minimize(
        fun=model._objective_function,
        x0=[0.0001],
        args=(signal_savgol, phase1, phase2),
        method="Nelder-Mead",
        options={'xatol': 1e-8, 'fatol': 1e-8}
    )
    final_s = result.x[0]

    # 4. Get the polynomial drift component
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    if phase1 - delta <= 0 or phase2 + delta >= len(signal_savgol) or phase2 - delta - (phase1 + delta) < 5:
        delta = min(2, phase1, len(signal_savgol) - phase2 - 1)
        
    y_for_poly = np.concatenate([
        signal_savgol[: phase1 - delta],
        signal_savgol[phase1 + delta : phase2 - delta] * (1 + final_s),
        signal_savgol[phase2 + delta :]
    ])
    x_for_poly = np.arange(len(y_for_poly))
    
    coeffs = np.polyfit(x_for_poly, y_for_poly, deg=cfg.MODEL_POLYNOMIAL_DEGREE)
    poly_drift = np.poly1d(coeffs)
    
    # 5. Create the full transit model
    time_steps = np.arange(len(signal_savgol))
    full_model_fit = poly_drift(time_steps)
    full_model_fit[phase1 + delta : phase2 - delta] /= (1 + final_s) # Create the "dip" in the poly

    # 6. Plot everything
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.figure(figsize=(14, 7))
    
    # Plot 1: The full, complex fit
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, signal_savgol, 'o', color='gray', markersize=3, alpha=0.5, label='Smoothed Data')
    plt.plot(time_steps, poly_drift(time_steps), 'r-', linewidth=2, label=f'Polynomial Drift (Deg {cfg.MODEL_POLYNOMIAL_DEGREE})')
    plt.plot(time_steps, full_model_fit, 'c-', linewidth=3, label=f'Full Transit Model (s={final_s:.6f})')
    plt.axvspan(phase1, phase2, color='blue', alpha=0.1, label='Detected Transit Phase')
    plt.title(title, fontsize=16)
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Plot 2: The de-trended light curve
    plt.subplot(2, 1, 2)
    detrended_flux = signal_savgol / poly_drift(time_steps)
    detrended_model = full_model_fit / poly_drift(time_steps)
    
    plt.plot(time_steps, detrended_flux, 'o', color='gray', markersize=3, alpha=0.7, label='De-trended Data')
    plt.plot(time_steps, detrended_model, 'c-', linewidth=3, label='Transit Model Shape')
    plt.axvspan(phase1, phase2, color='blue', alpha=0.1)
    plt.title("De-trended Light Curve", fontsize=16)
    plt.xlabel('Binned Time Step')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def get_airs_weights(binned_data: np.ndarray) -> np.ndarray:
    """Helper to recalculate and return the AIRS weights."""
    var = np.nanvar(binned_data, axis=0, ddof=1)
    med = np.nanmedian(var)
    safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
    w = 1.0 / safe_var

    lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        w = np.clip(w, lo, hi)

    M = binned_data.shape[1]
    s = np.nansum(w)
    if np.isfinite(s) and s > 0:
        w = w * (M / s)
    else:
        w = np.ones_like(w)
    return w

# --- Main Visualization Function ---

def visualize_full_pipeline(planet_id: int, config: Config):
    """
    Runs the full processing pipeline for a SINGLE planet and plots
    the results at each major step.
    """
    try:
        print(f"\n--- [VISUALIZER] Starting for Planet {planet_id} ---")
        logger.info(f"--- Visualizing Pipeline for Planet {planet_id} ---")
        processor = SignalProcessor(config)
        model = TransitModel(config)
        
        # --- (NEW) Step 0: Plot Calibration Files ---
        print("[VISUALIZER] Step 0: Loading and plotting calibration files...")
        dark_airs = pd.read_parquet(f"{config.DATA_PATH}/{config.DATASET}/{planet_id}/AIRS-CH0_calibration_0/dark.parquet").to_numpy()
        flat_airs = pd.read_parquet(f"{config.DATA_PATH}/{config.DATASET}/{planet_id}/AIRS-CH0_calibration_0/flat.parquet").to_numpy()
        
        # Crop them just as the pipeline does
        dark_airs = dark_airs[:, config.CUT_INF:config.CUT_SUP]
        flat_airs = flat_airs[:, config.CUT_INF:config.CUT_SUP]
        
        plot_heatmap(dark_airs, f"Planet {planet_id} - AIRS Dark Frame (What We Subtract)", "Wavelength", "Y-Pixel", aspect='auto', cmap='inferno')
        plot_heatmap(flat_airs, f"Planet {planet_id} - AIRS Flat Frame (What We Divide By)", "Wavelength", "Y-Pixel", aspect='auto', cmap='viridis')
        print("[VISUALIZER] Step 0 COMPLETE.")


        # --- Step 1: Calibrated Frame ---
        print("\n[VISUALIZER] Step 1: Calibrating AIRS signal...")
        logger.info("Visualizing Step 1: Calibration (AIRS-CH0)")
        calibrated_airs = processor._calibrate_single_signal(planet_id, "AIRS-CH0")
        plot_heatmap(calibrated_airs[0], f"Planet {planet_id} - Calibrated AIRS Frame (Time 0)", "Wavelength", "Y-Pixel", aspect='auto')
        print("[VISUALIZER] Step 1 COMPLETE.")

        # --- Step 2: Preprocessing & Light Curves ---
        print("\n[VISUALIZER] Step 2: Preprocessing and creating light curves...")
        logger.info("Visualizing Step 2: Preprocessing (FGS1 & AIRS-CH0)")
        
        # Process both sensors
        p_fgs1 = processor._process_planet_sensor({"planet_id": planet_id, "sensor": "FGS1"})
        p_airs_ch0 = processor._process_planet_sensor({"planet_id": planet_id, "sensor": "AIRS-CH0"})
        
        # (NEW) Plot the FGS1 light curve
        fgs_light_curve = p_fgs1.flatten() # It's just (N, 1)
        plot_light_curve(
            np.arange(len(fgs_light_curve)),
            fgs_light_curve,
            f"Planet {planet_id} - Binned FGS1 Light Curve",
            "Binned Flux",
            color='green'
        )
        
        # (NEW) Plot the AIRS 2D Spectroscopic Light Curve
        plot_heatmap(
            p_airs_ch0,
            f"Planet {planet_id} - Binned AIRS 2D Light Curve (Spectroscopic)",
            "Wavelength Channel",
            "Binned Time Step",
            aspect='auto',
            cmap='viridis'
        )
        
        # (NEW) Plot the AIRS weights
        # We need the *un-weighted* binned data to calculate the weights
        unweighted_binned_airs = processor._preprocess_calibrated_signal(calibrated_airs, "AIRS-CH0")
        airs_weights = get_airs_weights(unweighted_binned_airs)
        plot_weights(airs_weights, f"Planet {planet_id} - AIRS Channel Weights (Trust)")
        
        print("[VISUALIZER] Step 2 COMPLETE.")

        # --- Step 3: Modeling ---
        print("\n[VISUALIZER] Step 3: Fitting the final transit model...")
        logger.info("Visualizing Step 3: Transit Model Fit")
        
        # Create the combined signal as used in the main script
        combined_signal = np.concatenate([p_fgs1, p_airs_ch0], axis=1)
        signal_1d_for_model = combined_signal[:, 1:].mean(axis=1)
        
        plot_final_transit_fit(
            model,
            signal_1d_for_model,
            f"Planet {planet_id} - Final Transit Model Fit (on AIRS White Light)"
        )
        print("[VISUALIZER] Step 3 COMPLETE.")
        
        logger.info(f"--- Visualization Complete for Planet {planet_id} ---")
        print(f"\n--- [VISUALIZER] Finished for Planet {planet_id} ---")

    except Exception as e:
        print(f"\n--- [VISUALIZER] ERROR! ---")
        print(f"An error occurred: {e}")
        logger.error(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

# --- Run the Visualization ---
# (This assumes you have already run Block 1 so all classes are in memory)
try:
    config = Config()
    PLANET_TO_VISUALIZE = pd.read_csv(
        f'{config.DATA_PATH}/{config.DATASET}_star_info.csv'
    ).planet_id.iloc[0]
    
    visualize_full_pipeline(int(PLANET_TO_VISUALIZE), config)
except Exception as e:
    logger.error(f"Failed to start visualization: {e}")
    print(f"Failed to start visualization. Make sure Block 1 has been run. Error: {e}")

