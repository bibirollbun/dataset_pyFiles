!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm


import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm
from astropy.stats import sigma_clip
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip

import time
from typing import Tuple, Dict, List, Any, Optional

# Progress and parallel processing
from tqdm import tqdm
from pqdm.threads import pqdm
import itertools

# Optimization and metrics
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Signal processing and statistics
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter



# Performance timing
__t0 = time.perf_counter()

# Global constants
ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"


class Config:
    """
    Configuration class containing all pipeline parameters and sensor specifications.
    
    This class centralizes all configuration parameters including data paths,
    processing parameters, sensor configurations, and model hyperparameters.
    """
    
    # Data configuration
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"

    # Model parameters
    SCALE = 0.95                    # Final scaling factor for predictions
    SIGMA = 0.0006                  # Base uncertainty estimate
    
    # Spectral range configuration
    CUT_INF = 39                    # Lower wavelength cut
    CUT_SUP = 321                   # Upper wavelength cut
    
    # Sensor specifications
    SENSOR_CONFIG = {
        "AIRS-CH0": {
            "raw_shape": [11250, 32, 356],
            "calibrated_shape": [1, 32, CUT_SUP - CUT_INF],
            "linear_corr_shape": (6, 32, 356),
            "dt_pattern": (0.1, 4.5),      # Even/odd frame dark correction
            "binning": 30
        },
        "FGS1": {
            "raw_shape": [135000, 32, 32],
            "calibrated_shape": [1, 32, 32],
            "linear_corr_shape": (6, 32, 32),
            "dt_pattern": (0.1, 0.1),      # Even/odd frame dark correction
            "binning": 30 * 12
        }
    }
    
    # Transit model parameters
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)   # Search range for transit phases
    MODEL_OPTIMIZATION_DELTA = 7                   # Safety margin around transit phases
    MODEL_POLYNOMIAL_DEGREE = 3                    # Polynomial degree for baseline fitting
    
    # Processing configuration
    N_JOBS = 3                                     # Number of parallel jobs




def _phase_detector_signal(signal: np.ndarray, cfg: Config) -> Tuple[int, int]:
    """
    Detect transit ingress and egress phases from a light curve signal.
    
    This function identifies the two main phases of a transit by analyzing
    the gradient of the signal around the deepest point.
    
    Args:
        signal: 1D array representing the light curve
        cfg: Configuration object containing detection parameters
        
    Returns:
        Tuple of (phase1, phase2) representing ingress and egress indices
    """
    sl = cfg.MODEL_PHASE_DETECTION_SLICE
    min_idx = int(np.argmin(signal[sl])) + sl.start
    
    # Split signal at minimum point
    s1 = signal[:min_idx]
    s2 = signal[min_idx:]
    
    # Safety check for minimum signal length
    if s1.size < 3 or s2.size < 3:
        return 0, len(signal) - 1
    
    # Compute normalized gradients
    g1 = np.gradient(s1)
    g1_max = np.max(g1) if np.size(g1) else 0.0
    
    g2 = np.gradient(s2)
    g2_max = np.max(g2) if np.size(g2) else 0.0
    
    # Normalize gradients
    if g1_max != 0:
        g1 /= g1_max
    if g2_max != 0:
        g2 /= g2_max
    
    # Find phase boundaries
    phase1 = int(np.argmin(g1))
    phase2 = int(np.argmax(g2)) + min_idx
    
    return phase1, phase2


def estimate_sigma_fgs(preprocessed_data: List[np.ndarray], cfg: Config) -> np.ndarray:
    """
    Estimate adaptive uncertainty parameters for FGS1 sensor data.
    
    This function computes planet-specific uncertainty estimates by analyzing
    the variance in out-of-transit vs in-transit regions.
    
    Args:
        preprocessed_data: List of preprocessed data arrays for each planet
        cfg: Configuration object
        
    Returns:
        Array of uncertainty estimates, one per planet
    """
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12
    
    for single in preprocessed_data:
        # Extract white light curve from AIRS channels for phase detection
        air_white = savgol_filter(single[:, 1:].mean(axis=1), 20, 2)
        p1, p2 = _phase_detector_signal(air_white, cfg)
        
        # Apply safety margins
        p1 = max(delta, p1)
        p2 = min(len(air_white) - delta - 1, p2)

        # Extract FGS data and define regions
        fgs = single[:, 0]
        
        # Out-of-transit regions
        oot = (fgs[:p1 - delta] if p1 - delta > 0 else np.empty(0, fgs.dtype))
        if p2 + delta < fgs.size:
            oot = np.concatenate([oot, fgs[p2 + delta:]])
            
        # In-transit region
        inn = fgs[p1 + delta:max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan)
            continue

        # Compute relative uncertainty
        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(fgs))
        
        # Relative uncertainty of transit depth
        sigma_rel = np.sqrt(var_oot / max(n_oot, 1) + var_in / max(n_in, 1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    # Convert to array and compute adaptive scaling
    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # Apply soft scaling with conservative clipping
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)  # Â±20-25% from baseline Ïƒ

    return k * cfg.SIGMA


def estimate_sigma_air(preprocessed_data: List[np.ndarray], cfg: Config) -> np.ndarray:
    """
    Estimate adaptive uncertainty parameters for AIRS channels.
    
    Similar to estimate_sigma_fgs but for AIRS spectroscopic channels,
    using the white light curve for phase detection.
    
    Args:
        preprocessed_data: List of preprocessed data arrays for each planet
        cfg: Configuration object
        
    Returns:
        Array of uncertainty estimates, one per planet
    """
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        # Compute AIRS white light curve
        white = np.nanmean(single[:, 1:], axis=1)
        white_s = savgol_filter(white, 20, 2)

        # Detect transit phases
        p1, p2 = _phase_detector_signal(white_s, cfg)
        p1 = max(delta, p1)
        p2 = min(len(white) - delta - 1, p2)

        # Define out-of-transit and in-transit regions
        oot_left = white[:p1 - delta] if p1 - delta > 0 else np.empty(0, white.dtype)
        oot_right = white[p2 + delta:] if (p2 + delta) < white.size else np.empty(0, white.dtype)
        oot = np.concatenate([oot_left, oot_right]) if (oot_left.size + oot_right.size) else oot_left
        inn = white[p1 + delta:max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan)
            continue

        # Compute uncertainty metrics
        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(white))

        sigma_rel = np.sqrt(var_oot / max(n_oot, 1) + var_in / max(n_in, 1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    # Apply adaptive scaling with conservative bounds
    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.90, 1.20)  # Â±10-20% variation

    return k * cfg.SIGMA






class SignalProcessor:
    """
    Advanced signal processing pipeline for Ariel mission data.
    
    This class handles the complete signal processing workflow including:
    - Raw data calibration and correction
    - Linearity correction application
    - Dark current subtraction with time-dependent patterns
    - Signal preprocessing and binning
    - Outlier rejection and weighting
    """
    
    def __init__(self, config: Config):
        """
        Initialize the signal processor with configuration and metadata.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(
            f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', 
            index_col='planet_id'
        ).index.astype(int)

    def _apply_linear_corr(self, linear_corr: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Apply polynomial linearity correction to detector signals.
        
        This method applies a polynomial correction to account for non-linear
        detector response, using coefficients stored in reverse order.
        
        Args:
            linear_corr: Correction coefficients with shape (degree, x, y)
            signal: Input signal array with shape (time, x, y)
            
        Returns:
            Linearity-corrected signal array
        """
        # Flip coefficients for proper polynomial evaluation (highest degree first)
        coeffs = np.flip(linear_corr, axis=0)
        x = signal.astype(np.float64, copy=False)
        out = np.empty_like(x, dtype=np.float64)
        
        # Initialize with constant term
        out[...] = coeffs[0]
        
        # Apply Horner's method for stable polynomial evaluation
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)  # In-place multiplication for memory efficiency
            out += coeffs[k]
            
        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id: int, sensor: str) -> np.ndarray:
        """
        Perform complete calibration of a single sensor's raw data.
        
        This method applies the full calibration pipeline including:
        - ADC gain and offset correction
        - Hot pixel masking
        - Spectral/spatial cropping
        - Linearity correction
        - Time-dependent dark current subtraction
        
        Args:
            planet_id: Identifier for the target planet
            sensor: Sensor name ('AIRS-CH0' or 'FGS1')
            
        Returns:
            Calibrated signal array
        """
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]

        # Load raw data and calibration files
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

        # Reshape raw signal and apply ADC correction
        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        # Hot pixel detection using sigma clipping
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask

        # Sensor-specific cropping and processing
        if sensor == "AIRS-CH0":
            # Spectral cropping for AIRS
            signal = signal[:, :, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF:self.cfg.CUT_SUP]

        if sensor == "FGS1":
            # Spatial cropping for FGS1 (central region)
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark = dark[y0:y1, x0:x1]
            dead = dead[y0:y1, x0:x1]
            flat = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot = hot[y0:y1, x0:x1]

        # Ensure non-negative values
        np.maximum(signal, 0, out=signal)

        # Apply linearity correction
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            # Apply correction only to the central spatial region
            sl = (slice(None), slice(10, 22), slice(None))
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)

        # Time-dependent dark current subtraction
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale = base_dt + increment

        # Apply different dark correction for even/odd frames
        signal[::2] -= dark * even_scale    # Even frames: 0, 2, 4, ...
        signal[1::2] -= dark * odd_scale    # Odd frames: 1, 3, 5, ...
        
        return signal

    def _preprocess_calibrated_signal(self, calibrated_signal: np.ndarray, sensor: str) -> np.ndarray:
        """
        Preprocess calibrated signals for transit analysis.
        
        This method performs:
        - ROI extraction
        - Correlated double sampling (CDS)
        - Temporal binning
        - Outlier rejection (winsorization)
        - Inverse-variance weighting for AIRS channels
        
        Args:
            calibrated_signal: Calibrated signal array
            sensor: Sensor identifier
            
        Returns:
            Preprocessed signal ready for transit analysis
        """
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        # Extract region of interest
        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)
        
        # Spatial averaging
        mean_signal = np.nanmean(signal_roi, axis=1)

        # Correlated Double Sampling (CDS) - difference between consecutive reads
        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        # Temporal binning for noise reduction
        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning:(j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        # Winsorization (outlier clipping) for AIRS channels after binning
        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)
            np.clip(binned, q_lo, q_hi, out=binned)

        # Reshape FGS1 data
        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        # Apply inverse-variance weighting for AIRS channels
        if sensor == "AIRS-CH0":
            # Compute wavelength-dependent weights
            var = np.nanvar(binned, axis=0, ddof=1)
            med = np.nanmedian(var)
            
            # Replace invalid variances with median
            safe_var = np.where(
                ~np.isfinite(var) | (var <= 0), 
                med if (np.isfinite(med) and med > 0) else 1.0, 
                var
            )
            w = 1.0 / safe_var

            # Clip weights to prevent single-channel dominance
            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            # Normalize weights to preserve mean levels
            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

            # Apply weights
            binned *= w[None, :]

        return binned

    def _process_planet_sensor(self, args: Dict[str, Any]) -> np.ndarray:
        """
        Process a single planet-sensor combination.
        
        Args:
            args: Dictionary containing 'planet_id' and 'sensor' keys
            
        Returns:
            Preprocessed signal array
        """
        planet_id, sensor = args['planet_id'], args['sensor']
        calibrated = self._calibrate_single_signal(planet_id, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self) -> np.ndarray:
        """
        Process all planets and sensors in parallel.
        
        This method coordinates the processing of all available data,
        combining FGS1 and AIRS-CH0 sensors for each planet.
        
        Returns:
            Combined preprocessed signals with shape (n_planets, n_time, n_channels)
        """
        print("ğŸš€ Processing FGS1 sensor data...")
        args_fgs1 = [dict(planet_id=planet_id, sensor="FGS1") for planet_id in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        print("ğŸŒŸ Processing AIRS-CH0 sensor data...")
        args_airs_ch0 = [dict(planet_id=planet_id, sensor="AIRS-CH0") for planet_id in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        print("ğŸ”— Combining sensor data...")
        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal







class TransitModel:
    """
    Sophisticated transit detection and parameter estimation model.
    
    This class implements a robust transit detection algorithm using:
    - Gradient-based phase detection
    - Polynomial baseline modeling
    - Optimization-based depth estimation
    - Adaptive error handling
    """
    
    def __init__(self, config: Config):
        """
        Initialize the transit model with configuration parameters.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.cfg = config

    def _phase_detector(self, signal: np.ndarray) -> Tuple[int, int]:
        """
        Detect transit ingress and egress phases using gradient analysis.
        
        Args:
            signal: 1D light curve signal
            
        Returns:
            Tuple of (phase1, phase2) indices marking transit boundaries
        """
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        
        # Split signal at deepest point
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]

        # Compute and normalize gradients
        grad1 = np.gradient(signal1)
        grad1 /= grad1.max()
        
        grad2 = np.gradient(signal2)
        grad2 /= grad2.max()

        # Find phase boundaries
        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index

        return phase1, phase2
    
    def _objective_function(self, s: float, signal: np.ndarray, 
                          phase1: int, phase2: int) -> float:
        """
        Objective function for transit depth optimization.
        
        This function fits a polynomial baseline to the out-of-transit data
        and measures the error when applying a transit depth correction.
        
        Args:
            s: Transit depth parameter to optimize
            signal: Input light curve signal
            phase1: Ingress phase index
            phase2: Egress phase index
            
        Returns:
            Mean absolute error of the polynomial fit
        """
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE

        # Adaptive delta for edge cases
        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
            delta = 2

        # Construct corrected signal
        y = np.concatenate([
            signal[:phase1 - delta],                           # Pre-transit
            signal[phase1 + delta:phase2 - delta] * (1 + s),  # In-transit (corrected)
            signal[phase2 + delta:]                            # Post-transit
        ])
        x = np.arange(len(y))

        # Fit polynomial baseline and compute error
        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        
        return error

    def predict(self, single_preprocessed_signal: np.ndarray) -> float:
        """
        Predict transit depth for a single light curve.
        
        Args:
            single_preprocessed_signal: Preprocessed signal array
            
        Returns:
            Estimated transit depth
        """
        # Create white light curve from AIRS channels
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
        signal_1d = savgol_filter(signal_1d, 20, 2)  # Smooth for phase detection
        
        # Detect transit phases
        phase1, phase2 = self._phase_detector(signal_1d)

        # Apply safety margins
        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    

        # Optimize transit depth
        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],                    # Initial guess
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead"
        )
        
        return result.x[0]

    def predict_all(self, preprocessed_signals: np.ndarray) -> np.ndarray:
        """
        Predict transit depths for all input signals.
        
        Args:
            preprocessed_signals: Array of preprocessed signals
            
        Returns:
            Array of transit depth predictions
        """
        print("ğŸ”� Analyzing transit signals...")
        predictions = [
            self.predict(preprocessed_signal)
            for preprocessed_signal in tqdm(preprocessed_signals, desc="Transit Analysis")
        ]
        return np.array(predictions) * self.cfg.SCALE






class SubmissionGenerator:
    """
    Generate final submission file with predictions and uncertainties.
    
    This class handles the creation of the competition submission format,
    including proper formatting of predictions and uncertainty estimates.
    """
    
    def __init__(self, config: Config):
        """
        Initialize submission generator with template.
        
        Args:
            config: Configuration object
        """
        self.cfg = config
        self.sample_submission = pd.read_csv(
            "/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", 
            index_col="planet_id"
        )

    def create(self, predictions1: np.ndarray, predictions2: np.ndarray, 
               predictions: np.ndarray, sigma_fgs: Optional[np.ndarray] = None, 
               sigma_air: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create final submission DataFrame with predictions and uncertainties.
        
        Args:
            predictions1: FGS1 predictions
            predictions2: AIRS predictions  
            predictions: Combined predictions
            sigma_fgs: Optional FGS uncertainty estimates
            sigma_air: Optional AIRS uncertainty estimates
            
        Returns:
            Formatted submission DataFrame
        """
        print("ğŸ“� Generating submission file...")
        
        planet_ids = self.sample_submission.index
        n_mu = self.sample_submission.shape[1] // 2  # 283 channels

        # Prepare mean predictions
        preds = np.asarray(predictions, dtype=float).reshape(-1)
        mu = np.tile(preds.reshape(-1, 1), (1, n_mu))
        mu = np.clip(mu, 0, None)  # Ensure non-negative

        # Prepare uncertainty estimates
        sigmas = np.full_like(mu, self.cfg.SIGMA, dtype=float)
        
        if sigma_fgs is not None:
            sigma_fgs = np.asarray(sigma_fgs, dtype=float).reshape(-1)
            sigmas[:, 0] = np.clip(sigma_fgs, 1e-6, 0.1)
            
        if sigma_air is not None:
            sigma_air = np.asarray(sigma_air, dtype=float).reshape(-1, 1)
            sigmas[:, 1:] = np.clip(sigma_air, 1e-6, 0.1)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            np.concatenate([mu, sigmas], axis=1),
            columns=self.sample_submission.columns,
            index=planet_ids
        )
        
        # Apply specific predictions
        submission_df.iloc[:, 1:283] = predictions2  # AIRS channels
        submission_df.iloc[:, 0] = predictions1      # FGS1 channel

        # Save to file
        submission_df.to_csv("submission.csv")
        print("âœ… Submission file saved successfully!")
        return submission_df


class ResidualBlock(nn.Module):
    def __init__(self, dim, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)


class ResNetMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim = 282, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x
resnet = ResNetMLP(num_blocks=80, dropout_rate=0.3)
resnet.load_state_dict(torch.load("/kaggle/input/airs/pytorch/default/1/best_model_airs.pth"))
resnet.eval()


StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
PlanetIds = StarInfo["planet_id"].tolist()
StarInfo = StarInfo.set_index("planet_id")



config = Config()
    
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)
predictions


predictions_df = pd.DataFrame({ "planet_id": PlanetIds, "transit_depth": predictions })
predictions_df


input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")
input_df["transit_depth"] *= 10000
features = ['transit_depth','Rs','i']
X = input_df[features].values.astype(np.float32)
X


X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    predictions2 = resnet(X_tensor).numpy()
predictions2 /= 10000


predictions1 = predictions


submission_generator = SubmissionGenerator(config) 
submission = submission_generator.create(predictions1, predictions2, predictions, sigma_fgs=sigma_fgs_vec, sigma_air=sigma_air_vec)


pd.read_csv("submission.csv")

