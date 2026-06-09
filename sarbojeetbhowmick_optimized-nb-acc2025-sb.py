
# install pqdm library for parallel processing
!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm


import pandas as pd
import numpy as np

from tqdm import tqdm
from pqdm.threads import pqdm
import itertools

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
import time
__t0 = time.perf_counter()


class Config:
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"

    SCALE = 0.95
    SIGMA = 0.0009
    
    CUT_INF = 39
    CUT_SUP = 321
    
    SENSOR_CONFIG = {
        "AIRS-CH0": {
            "raw_shape": [11250, 32, 356],
            "calibrated_shape": [1, 32, CUT_SUP - CUT_INF],
            "linear_corr_shape": (6, 32, 356),
            "dt_pattern": (0.1, 4.5), 
            "binning": 30
        },
        "FGS1": {
            "raw_shape": [135000, 32, 32],
            "calibrated_shape": [1, 32, 32],
            "linear_corr_shape": (6, 32, 32),
            "dt_pattern": (0.1, 0.1),
            "binning": 30 * 12
        }
    }
    
    MODEL_PHASE_DETECTION_SLICE = slice(28, 145)
    MODEL_OPTIMIZATION_DELTA = 11 # 9
    MODEL_POLYNOMIAL_DEGREE = 2
    
    N_JOBS = 3

def _phase_detector_signal(signal, cfg):
    sl = cfg.MODEL_PHASE_DETECTION_SLICE
    min_idx = int(np.argmin(signal[sl])) + sl.start
    s1 = signal[:min_idx]; s2 = signal[min_idx:]
    if s1.size < 3 or s2.size < 3:
        return 0, len(signal) - 1
    g1 = np.gradient(s1); g1_max = np.max(g1) if np.size(g1) else 0.0
    g2 = np.gradient(s2); g2_max = np.max(g2) if np.size(g2) else 0.0
    if g1_max != 0: g1 /= g1_max
    if g2_max != 0: g2 /= g2_max
    phase1 = int(np.argmin(g1)); phase2 = int(np.argmax(g2)) + min_idx
    return phase1, phase2

def estimate_sigma_fgs(preprocessed_data, cfg):
    """Возвращает вектор sigma_1 (для FGS1) длиной N_planets — мягкий множитель к cfg.SIGMA."""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12
    for single in preprocessed_data:
        # phases @ AIRS white curve - similar to the model
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
            sig_rel.append(np.nan); continue

        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in  = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(fgs))
        # relative uncertainty of depth (in the same units as s)
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # soft multiplier: root, and narrow clip to play it safe
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)  # ±20–25% from the base σ

    return k * cfg.SIGMA

def estimate_sigma_air(preprocessed_data, cfg):
    """Returns a sigma_air vector of length N_planets - a soft multiplier to cfg.SIGMA for all AIRS channels."""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        # white AIRS curve on binned data (after all your scales λ)
        white = np.nanmean(single[:, 1:], axis=1)         # (n_bins,)
        white_s = savgol_filter(white, 20, 2)             # phases

        p1, p2 = _phase_detector_signal(white_s, cfg)
        p1 = max(delta, p1)
        p2 = min(len(white) - delta - 1, p2)

        oot_left = white[: p1 - delta] if p1 - delta > 0 else np.empty(0, white.dtype)
        oot_right = white[p2 + delta :] if (p2 + delta) < white.size else np.empty(0, white.dtype)
        oot = np.concatenate([oot_left, oot_right]) if (oot_left.size + oot_right.size) else oot_left
        inn = white[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan); continue

        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in  = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(white))

        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # Soft multiplier around the median
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.90, 1.20)  # ±10%–20%

    return k * cfg.SIGMA


class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', index_col='planet_id').index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):

        coeffs = np.flip(linear_corr, axis=0)      # shape: (D, X, Y), D — senior degree first
        x = signal.astype(np.float64, copy=False)  # we count in float64 for stability
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]  # broadcast (X,Y) -> (T,X,Y)
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)  # in-place multiplication
            out += coeffs[k]              # broadcast (X,Y)

        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id, sensor):
        """
        Calibration of single-node signal.
        DEAD — mask, HOT — DO NOT mask (leave in data).
        """
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
    
        # --- load ---
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
    
        # --- reshape & ADC ---
        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset 
    
        # HOT is for monitoring only, not for masking
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    
        # --- crop per sensor ---
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]  # logs
    
        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark   = dark[y0:y1, x0:x1]
            dead   = dead[y0:y1, x0:x1]
            flat   = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot    = hot[y0:y1, x0:x1]  # logs
    
        # --- non-neg clamp before linearity corr ---
        np.maximum(signal, 0, out=signal)
    
        # --- linearity correction ---
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))  # T, Y, λ
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)
    
        # --- dark subtraction considering the integration pattern ---
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale  = base_dt + increment
        signal[::2]  -= dark * even_scale
        signal[1::2] -= dark * odd_scale
    
        # --- APPLY FLAT (HOT-KEEP: don't turn on hot in the mask!) ---
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()      # (12,12)
            bad = (dead) | ~np.isfinite(flat_roi) | (flat_roi == 0)      # only invalid
            flat_roi[bad] = np.nan
            signal /= flat_roi
    
        elif sensor == "AIRS-CH0":
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()  # (12, λ)
            bad = (dead[y0:y1, :]) | ~np.isfinite(flat_roi) | (flat_roi == 0)  # only invalid
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi
    
        else:
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            bad2 = (dead) | ~np.isfinite(flat2) | (flat2 == 0)                  # only invalid
            flat2[bad2] = np.nan
            signal /= flat2
        # --- END FLAT ---
    
        # log metrics hot or dead
        if getattr(self.cfg, "LOG_HOT_STATS", False):
            if not hasattr(self, "stats"):
                self.stats = []
            self.stats.append({
                "planet_id": int(planet_id),
                "sensor": sensor,
                "hot_frac": float(np.mean(hot)),
                "dead_frac": float(np.mean(dead)),
            })
    
        return signal


    def _preprocess_calibrated_signal(self, calibrated_signal, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = sensor_cfg["binning"]

        if sensor == "AIRS-CH0":
            signal_roi = calibrated_signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_roi = calibrated_signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)
        
        mean_signal = np.nanmean(signal_roi, axis=1)

        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        # NEW: Winsorization AFTER binning (cheap), only for AIRS
        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)    # (n_bins, 1)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)   # (n_bins, 1)
            np.clip(binned, q_lo, q_hi, out=binned)

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        if sensor == "AIRS-CH0":
            # Inversely dispersed weights by λ on a binned series (n_bins, λ)
            var = np.nanvar(binned, axis=0, ddof=1)                 # (λ,0)
            med = np.nanmedian(var)
            # replace small variances with the median
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            # protective clip of scales so that one channel does not dominate
            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            # normalization: sum of weights = number of channels → mean == weighted mean
            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

            # apply weights to each time (broadcast on axis 0)
            binned *= w[None, :]


        return binned

    def _process_planet_sensor(self, args):
        planet_id, sensor = args['planet_id'], args['sensor']
        calibrated = self._calibrate_single_signal(planet_id, sensor)
        preprocessed = self._preprocess_calibrated_signal(calibrated, sensor)
        return preprocessed

    def process_all_data(self):
        args_fgs1 = [dict(planet_id=planet_id, sensor="FGS1") for planet_id in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        args_airs_ch0 = [dict(planet_id=planet_id, sensor="AIRS-CH0") for planet_id in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal
    

class TransitModel:
    def __init__(self, config):
        self.cfg = config

    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]

        grad1 = np.gradient(signal1)
        grad1 /= grad1.max()
        
        grad2 = np.gradient(signal2)
        grad2 /= grad2.max()

        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index

        return phase1, phase2
    
    def _objective_function(self, s, signal, phase1, phase2):
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE

        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
            delta = 2

        y = np.concatenate([
            signal[: phase1 - delta],
            signal[phase1 + delta : phase2 - delta] * (1 + s),
            signal[phase2 + delta :]
        ])
        x = np.arange(len(y))

        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        
        return error

    def predict(self, single_preprocessed_signal):
        air = single_preprocessed_signal[:, 1:].copy()
        q_lo = np.nanpercentile(air, 10.0, axis=1, keepdims=True)
        q_hi = np.nanpercentile(air, 90.0, axis=1, keepdims=True)
        np.clip(air, q_lo, q_hi, out=air)
        signal_1d = np.nanmean(air, axis=1)
        signal_1d = savgol_filter(signal_1d, 23, 2)
        
        phase1, phase2 = self._phase_detector(signal_1d)

        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    

        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead"
        )
        
        return result.x[0]

    def predict_all(self, preprocessed_signals):
        predictions = [
            self.predict(preprocessed_signal)
            for preprocessed_signal in tqdm(preprocessed_signals)
        ]
        return np.array(predictions) * self.cfg.SCALE
    
class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", index_col="planet_id")

    def create(self, predictions, sigma_fgs=None, sigma_air=None):
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



config = Config()
    
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)


submission_generator = SubmissionGenerator(config)
submission = submission_generator.create(predictions, sigma_fgs=sigma_fgs_vec, sigma_air=sigma_air_vec)


__t1 = time.perf_counter()
elapsed = __t1 - __t0
print(f"[TIMING] total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")

