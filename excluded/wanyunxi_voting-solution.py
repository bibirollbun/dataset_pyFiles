# install pqdm for parallel processing
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
ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"

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
    MODEL_OPTIMIZATION_DELTA = 11
    MODEL_POLYNOMIAL_DEGREE = 3
    
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
        # фазы по AIRS белой кривой — так же, как в модели
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
        # относительная неопределённость глубины (в тех же ед., что s)
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # мягкий множитель: корень, и узкий клип, чтобы не рисковать
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)  # ±20–25% от базовой σ

    return k * cfg.SIGMA

def estimate_sigma_air(preprocessed_data, cfg):
    """Возвращает вектор sigma_air длиной N_planets — мягкий множитель к cfg.SIGMA для всех AIRS-каналов."""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        # белая кривая AIRS на бинированных данных (после всех твоих весов по λ)
        white = np.nanmean(single[:, 1:], axis=1)         # (n_bins,)
        white_s = savgol_filter(white, 20, 2)             # для фаз

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

    # мягкий множитель вокруг медианы
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

        coeffs = np.flip(linear_corr, axis=0)      # shape: (D, X, Y), D — старшая степень сначала
        x = signal.astype(np.float64, copy=False)  # считаем в float64 для стабильности
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]  # broadcast (X,Y) -> (T,X,Y)
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)  # in-place умножение
            out += coeffs[k]              # broadcast (X,Y)

        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]

        signal = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_signal_0.parquet").to_numpy()
        dark = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/flat.parquet").to_numpy()
        linear_corr = pd.read_parquet(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}/{sensor}_calibration_0/linear_corr.parquet").values.astype(np.float64).reshape(sensor_cfg["linear_corr_shape"])

        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask

        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]


        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark   = dark[y0:y1, x0:x1]
            dead   = dead[y0:y1, x0:x1]
            flat   = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot    = hot[y0:y1, x0:x1]

        #np.maximum(signal, 0, out=signal)

        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))     # T, Y, λ (уже порезан по λ выше)
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)




        # === стало (in‑place, без 3D-временного буфера) ===
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale  = base_dt + increment

        signal[::2] -= dark * even_scale   # кадры 0,2,4,...
        signal[1::2] -= dark * odd_scale   # кадры 1,3,5,...


        # --- APPLY FLAT (ROI-aware, без reshape) ---
        if sensor == "FGS1":
            # после раннего ROI: signal, dark, dead, hot, flat уже (12,12)
            flat_roi = flat.astype(signal.dtype, copy=False).copy()   # (12,12)
            mask_roi = (dead | hot)                                   # (12,12)
            # защита от нулей/NaN в flat
            flat_roi[mask_roi | ~np.isfinite(flat_roi) | (flat_roi == 0)] = np.nan
            signal /= flat_roi                                        # (T,12,12) / (12,12)

        elif sensor == "AIRS-CH0":
            # делим только в рабочем ROI по Y (10:22), весь спектр по λ
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()   # (12, λ)
            mask_roi = (dead | hot)[y0:y1, :]                                   # (12, λ)
            flat_roi[mask_roi | ~np.isfinite(flat_roi) | (flat_roi == 0)] = np.nan
            signal[:, y0:y1, :] /= flat_roi                                     # (T,12,λ) / (12,λ)

        else:
            # запасной вариант для прочих сенсоров с «полным кадром»
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            mask2 = (dead | hot)
            flat2[mask2 | ~np.isfinite(flat2) | (flat2 == 0)] = np.nan
            signal /= flat2
        # --- END FLAT ---
        
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

        # >>> НОВОЕ: винсоризация ПОСЛЕ биннинга (дёшево), только для AIRS
        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)    # (n_bins, 1)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)   # (n_bins, 1)
            np.clip(binned, q_lo, q_hi, out=binned)

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        if sensor == "AIRS-CH0":
            # Инверсно-дисперсные веса по λ на бinned-рядe (n_bins, λ)
            var = np.nanvar(binned, axis=0, ddof=1)                 # (λ, )
            med = np.nanmedian(var)
            # заменим невалидные/слишком маленькие дисперсии на медиану
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            # защитный клип весов, чтобы один канал не доминировал
            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            # нормировка: сумма весов = числу каналов → mean == взвешенному mean
            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

            # применяем веса к каждому времени (broadcast по оси 0)
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
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
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
        #submission_df.to_csv("submission.csv")
        return submission_df



config = Config()
    
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)  # новый шаг
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)

submission_generator = SubmissionGenerator(config)


submission_solution1 = submission_generator.create(predictions, sigma_fgs=sigma_fgs_vec, sigma_air=sigma_air_vec)


submission_solution1.head()


from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"
import time
__t0 = time.perf_counter()

class Config:
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"

    SCALE = 0.9475
    SIGMA = 0.00055
    
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
    
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 11 # 9
    MODEL_POLYNOMIAL_DEGREE = 4
    
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
        # фазы по AIRS белой кривой — так же, как в модели
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
        # относительная неопределённость глубины (в тех же ед., что s)
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # мягкий множитель: корень, и узкий клип, чтобы не рисковать
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)  # ±20–25% от базовой σ

    return k * cfg.SIGMA * 1.042

def estimate_sigma_air(preprocessed_data, cfg):
    """Возвращает вектор sigma_air длиной N_planets — мягкий множитель к cfg.SIGMA для всех AIRS-каналов."""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        # белая кривая AIRS на бинированных данных (после всех твоих весов по λ)
        white = np.nanmean(single[:, 1:], axis=1)         # (n_bins,)
        white_s = savgol_filter(white, 20, 2)             # для фаз

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

    # мягкий множитель вокруг медианы
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.90, 1.20)  # ±10%–20%

    return k * cfg.SIGMA * 0.78


class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', index_col='planet_id').index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):

        coeffs = np.flip(linear_corr, axis=0)      # shape: (D, X, Y), D — старшая степень сначала
        x = signal.astype(np.float64, copy=False)  # считаем в float64 для стабильности
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]  # broadcast (X,Y) -> (T,X,Y)
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)  # in-place умножение
            out += coeffs[k]              # broadcast (X,Y)

        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id, sensor):
        """
        Калибровка single-node сигнала.
        Политика масок: DEAD — маскируем, HOT — НЕ маскируем (оставляем в данных).
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
        signal = signal / gain + offset  # сохраняем твою формулу
    
        # HOT только для мониторинга, не для маскирования
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    
        # --- crop per sensor ---
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]  # только для логов
    
        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark   = dark[y0:y1, x0:x1]
            dead   = dead[y0:y1, x0:x1]
            flat   = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot    = hot[y0:y1, x0:x1]  # только для логов
    
        # --- non-neg clamp before linearity corr (как у тебя) ---
        np.maximum(signal, 0, out=signal)
    
        # --- linearity correction ---
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))  # T, Y, λ
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)
    
        # --- dark subtraction с учётом паттерна интеграций ---
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale  = base_dt + increment
        signal[::2]  -= dark * even_scale
        signal[1::2] -= dark * odd_scale
    
        # --- APPLY FLAT (HOT-KEEP: не включаем hot в маску!) ---
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()      # (12,12)
            bad = (dead) | ~np.isfinite(flat_roi) | (flat_roi == 0)      # ← ТОЛЬКО dead/invalid
            flat_roi[bad] = np.nan
            signal /= flat_roi
    
        elif sensor == "AIRS-CH0":
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()  # (12, λ)
            bad = (dead[y0:y1, :]) | ~np.isfinite(flat_roi) | (flat_roi == 0)  # ← ТОЛЬКО dead/invalid
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi
    
        else:
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            bad2 = (dead) | ~np.isfinite(flat2) | (flat2 == 0)                  # ← ТОЛЬКО dead/invalid
            flat2[bad2] = np.nan
            signal /= flat2
        # --- END FLAT ---
    
        # (опционально) логируем метрики hot/dead
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

        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)    # (n_bins, 1)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)   # (n_bins, 1)
            np.clip(binned, q_lo, q_hi, out=binned)

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        if sensor == "AIRS-CH0":
            var = np.nanvar(binned, axis=0, ddof=1)                 # (λ, )
            med = np.nanmedian(var)
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

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
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
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
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1, num_blocks=3, dropout_rate=0.2):
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
resnet.load_state_dict(torch.load("/kaggle/input/fgs1/pytorch/default/1/best_model.pth"))
resnet.eval()
StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
PlanetIds = StarInfo["planet_id"].tolist()
StarInfo = StarInfo.set_index("planet_id")

class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", index_col="planet_id")

    def create(self, predictions1, predictions2, predictions, sigma_fgs=None, sigma_air=None):
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
        submission_df.iloc[:, 0] = predictions1
        submission_df.iloc[:, 1:283] = predictions2
        #submission_df.to_csv("submission.csv")
        
        return submission_df



config = Config()
    
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)
predictions
predictions_df = pd.DataFrame({
    "planet_id": PlanetIds,
    "transit_depth": predictions
})

input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")
input_df["transit_depth"] *= 10000
features = ['transit_depth','Rs','i']
X = input_df[features].values.astype(np.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    predictions1 = resnet(X_tensor).numpy()
predictions1 /= 10000

class ResidualBlock2(nn.Module):
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


class ResNetMLP2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=282, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock2(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x
resnet2 = ResNetMLP2(num_blocks=120, dropout_rate=0.3)
resnet2.load_state_dict(torch.load("/kaggle/input/airs-deeper/pytorch/default/1/best_model deep1.pth"))
resnet2.eval()

with torch.no_grad():
    predictions2 = resnet2(X_tensor).numpy()
predictions2 /= 10000



submission_generator = SubmissionGenerator(config)
submission_solution2 = submission_generator.create(predictions1, predictions2, predictions, sigma_fgs=sigma_fgs_vec, sigma_air=sigma_air_vec)

submission_solution2.head()


ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"

__t0 = time.perf_counter()

class Config:
    FEATURES = ['transit_depth', 'Rs', 'i']
    # FEATURES = ['transit_depth', 'Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i']
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"
    load_data = True  # Set to True to load from disk, False to generate
    DEBUG = False

    SCALE = 0.96
    SIGMA = 0.00056
    
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
    
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 11 # 9
    MODEL_POLYNOMIAL_DEGREE = 3
    
    N_JOBS = 3

class ParticipantVisibleError(Exception):
    pass

def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    naive_mean: float,
    naive_sigma: float,
    fsg_sigma_true: float = 1e-6,
    airs_sigma_true: float = 1e-5,
    fgs_weight: float = 1,
) -> float:
    """
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair of x_hat,
    x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian distributions, hence 283 values for each test spectrum,
    the GLL value for one spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        fsg_sigma_true: (float) standard deviation from the FSG1 instrument for the test set.
        airs_sigma_true: (float) standard deviation from the AIRS instrument for the test set.
        fgs_weight: (float) relative weight of the fgs channel
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError('Negative values in the submission')
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError('Wrong number of columns in the submission')

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None)
    sigma_true = np.append(
        np.array(
            [
                fsg_sigma_true,
            ]
        ),
        np.ones(n_wavelengths - 1) * airs_sigma_true,
    )
    y_true = solution.values

    GLL_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    GLL_true = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
    GLL_mean = scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true))

    # normalise the score, right now it becomes a matrix instead of a scalar.
    ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

    weights = np.append(np.array([fgs_weight]), np.ones(len(solution.columns) - 1))
    weights = weights * np.ones_like(ind_scores)
    submit_score = np.average(ind_scores, weights=weights)
    return float(np.clip(submit_score, 0.0, 1.0))

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
        # фазы по AIRS белой кривой — так же, как в модели
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
        # относительная неопределённость глубины (в тех же ед., что s)
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # мягкий множитель: корень, и узкий клип, чтобы не рисковать
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)  # ±20–25% от базовой σ

    return k * cfg.SIGMA

def estimate_sigma_air(preprocessed_data, cfg):
    """Возвращает вектор sigma_air длиной N_planets — мягкий множитель к cfg.SIGMA для всех AIRS-каналов."""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        # белая кривая AIRS на бинированных данных (после всех твоих весов по λ)
        white = np.nanmean(single[:, 1:], axis=1)         # (n_bins,)
        white_s = savgol_filter(white, 20, 2)             # для фаз

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

    # мягкий множитель вокруг медианы
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

        coeffs = np.flip(linear_corr, axis=0)      # shape: (D, X, Y), D — старшая степень сначала
        x = signal.astype(np.float64, copy=False)  # считаем в float64 для стабильности
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]  # broadcast (X,Y) -> (T,X,Y)
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)  # in-place умножение
            out += coeffs[k]              # broadcast (X,Y)

        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id, sensor):
        """
        Калибровка single-node сигнала.
        Политика масок: DEAD — маскируем, HOT — НЕ маскируем (оставляем в данных).
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
        signal = signal / gain + offset  # сохраняем твою формулу
    
        # HOT только для мониторинга, не для маскирования
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    
        # --- crop per sensor ---
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]  # только для логов
    
        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark   = dark[y0:y1, x0:x1]
            dead   = dead[y0:y1, x0:x1]
            flat   = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot    = hot[y0:y1, x0:x1]  # только для логов
    
        # --- non-neg clamp before linearity corr (как у тебя) ---
        np.maximum(signal, 0, out=signal)
    
        # --- linearity correction ---
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))  # T, Y, λ
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)
    
        # --- dark subtraction с учётом паттерна интеграций ---
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale  = base_dt + increment
        signal[::2]  -= dark * even_scale
        signal[1::2] -= dark * odd_scale
    
        # --- APPLY FLAT (HOT-KEEP: не включаем hot в маску!) ---
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()      # (12,12)
            bad = (dead) | ~np.isfinite(flat_roi) | (flat_roi == 0)      # ← ТОЛЬКО dead/invalid
            flat_roi[bad] = np.nan
            signal /= flat_roi
    
        elif sensor == "AIRS-CH0":
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()  # (12, λ)
            bad = (dead[y0:y1, :]) | ~np.isfinite(flat_roi) | (flat_roi == 0)  # ← ТОЛЬКО dead/invalid
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi
    
        else:
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            bad2 = (dead) | ~np.isfinite(flat2) | (flat2 == 0)                  # ← ТОЛЬКО dead/invalid
            flat2[bad2] = np.nan
            signal /= flat2
        # --- END FLAT ---
    
        # (опционально) логируем метрики hot/dead
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

        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)    # (n_bins, 1)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)   # (n_bins, 1)
            np.clip(binned, q_lo, q_hi, out=binned)

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        if sensor == "AIRS-CH0":
            var = np.nanvar(binned, axis=0, ddof=1)                 # (λ, )
            med = np.nanmedian(var)
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

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
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
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

StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
PlanetIds = StarInfo["planet_id"].tolist()
StarInfo = StarInfo.set_index("planet_id")
class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", index_col="planet_id")

    def create(self, predictions1, predictions, sigma_fgs=None, sigma_air=None):
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
        submission_df.iloc[:, 0] = predictions
        submission_df.iloc[:, 1:283] = predictions1
        submission_df.to_csv("submission.csv")
        
        return submission_df

class ResidualBlock2(nn.Module):
    def __init__(self, dim, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        identity = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + identity)

class ResNetMLP2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=282, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock2(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

# Execute the training
# Train and get scalers
# all_models, scaler_X, scaler_y = train_new_model()

def load_cv_models_and_scalers(directory):
    """
    Loads all cross-validation models and scalers from the specified directory.
    Args:
        directory (str): Path to the directory containing model and scaler files.
    Returns:
        all_models (list): List of loaded models.
        scaler_X: Loaded X scaler.
        scaler_y: Loaded y scaler.
    """
    import os
    import joblib
    # Load scalers
    scaler_X = joblib.load(os.path.join(directory, 'scaler_X.joblib'))
    scaler_y = joblib.load(os.path.join(directory, 'scaler_y.joblib'))

    # Load all CV models
    all_models = []
    model_params = {
        'input_dim': len(Config.FEATURES),  # Always use the current feature count
        'hidden_dim': 256,
        'output_dim': 282,
        'num_blocks': 80,
        'dropout_rate': 0.3
    }
    for fold in range(1, 11):
        model = ResNetMLP2(**model_params).double()
        model_path = os.path.join(directory, f'best_model_airs_cv_fold{fold}.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        all_models.append(model)
    return all_models, scaler_X, scaler_y
    
all_models, scaler_X, scaler_y = load_cv_models_and_scalers('/kaggle/input/ariel-2025-result/results_3_v25')

config = Config()
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)

predictions_df = pd.DataFrame({
    "planet_id": PlanetIds,
    "transit_depth": predictions
})

input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")
X = input_df[Config.FEATURES].values.astype(np.float64)
X_scaled = scaler_X.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float64)

# Generate average prediction from all CV models (on scaled X, then inverse transform)
with torch.no_grad():
    preds_scaled = [model(X_tensor).numpy() for model in all_models]
predictions1_scaled = np.mean(preds_scaled, axis=0)
predictions1 = scaler_y.inverse_transform(predictions1_scaled)

class Config:
    # FEATURES = ['transit_depth', 'Rs', 'i']
    FEATURES = ['transit_depth', 'Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i']
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"
    load_data = True  # Set to True to load from disk, False to generate
    DEBUG = False

    SCALE = 0.96
    SIGMA = 0.00055
    
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
    
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 11 # 9
    MODEL_POLYNOMIAL_DEGREE = 3
    
    N_JOBS = 3

def load_cv_models_and_scalers(directory):
    """
    Loads all cross-validation models and scalers from the specified directory.
    Args:
        directory (str): Path to the directory containing model and scaler files.
    Returns:
        all_models (list): List of loaded models.
        scaler_X: Loaded X scaler.
        scaler_y: Loaded y scaler.
    """
    import os
    import joblib
    # Load scalers
    scaler_X = joblib.load(os.path.join(directory, 'scaler_X.joblib'))
    scaler_y = joblib.load(os.path.join(directory, 'scaler_y.joblib'))

    # Load all CV models
    all_models = []
    model_params = {
        'input_dim': len(Config.FEATURES),  # Always use the current feature count
        'hidden_dim': 256,
        'output_dim': 282,
        'num_blocks': 80,
        'dropout_rate': 0.3
    }
    for fold in range(1, 11):
        model = ResNetMLP2(**model_params).double()
        model_path = os.path.join(directory, f'best_model_airs_cv_fold{fold}.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        all_models.append(model)
    return all_models, scaler_X, scaler_y
    
all_models, scaler_X, scaler_y = load_cv_models_and_scalers('/kaggle/input/ariel-2025-result/results_9_v26')
config = Config()
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)

predictions_df = pd.DataFrame({
    "planet_id": PlanetIds,
    "transit_depth": predictions
})

input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")
X = input_df[Config.FEATURES].values.astype(np.float64)
X_scaled = scaler_X.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float64)

# Generate average prediction from all CV models (on scaled X, then inverse transform)
with torch.no_grad():
    preds_scaled = [model(X_tensor).numpy() for model in all_models]

# Correctly average the predictions
predictions2_scaled = np.mean(preds_scaled, axis=0)

# Correctly inverse transform the *averaged* predictions
predictions2 = scaler_y.inverse_transform(predictions2_scaled)

# Combine the predictions into a single array
all_predictions = np.array([predictions1, predictions2])

# Calculate the mean across the models (axis=0)
final_predictions = np.mean(all_predictions, axis=0)


submission_generator = SubmissionGenerator(config)
submission_solution3 = submission_generator.create(final_predictions, predictions, sigma_fgs=sigma_fgs_vec, sigma_air=sigma_air_vec)
submission_solution3.head()


# ======== パスとモード ========
ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"
__t0 = time.perf_counter()


# ======== Config ========
class Config:
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"

    SCALE = 0.928          # ※ #9 は今回は変更しない
    SIGMA = 0.00056
    
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
    
    MODEL_PHASE_DETECTION_SLICE = slice(27, 143)
    MODEL_OPTIMIZATION_DELTA = 11
    MODEL_POLYNOMIAL_DEGREE = 4
    
    N_JOBS = 3


# ======== 位相補助 ========
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


# ======== σ推定（#4: クリップ幅＆仕上げ係数を拡張） ========
def estimate_sigma_fgs(preprocessed_data, cfg):
    """FGS1用 σ のソフト推定（保守拡張版）"""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12
    for single in preprocessed_data:
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
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)

    # --- #4: clipをやや緩める（0.8–1.25 → 0.85–1.30）---
    k = np.clip(k, 0.85, 1.30)

    sigma_fgs = k * cfg.SIGMA

    # --- #4: 仕上げの全体係数（わずかに拡張）---
    sigma_fgs *= 1.04
    return sigma_fgs


def estimate_sigma_air(preprocessed_data, cfg):
    """AIRS用 σ のソフト推定（保守拡張版）"""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
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

    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)

    # --- #4: clipをやや緩める（0.90–1.20 → 0.92–1.22）---
    k = np.clip(k, 0.92, 1.22)

    sigma_air = k * cfg.SIGMA

    # --- #4: 仕上げの全体係数（わずかに拡張）---
    #sigma_air *= 1.04
    return sigma_air


# ======== 前処理パイプライン ========
class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(f'{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv', index_col='planet_id').index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):
        coeffs = np.flip(linear_corr, axis=0)
        x = signal.astype(np.float64, copy=False)
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)
            out += coeffs[k]
        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id, sensor):
        sensor_cfg = self.cfg.SENSOR_CONFIG[sensor]
    
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
    
        signal = signal.reshape(sensor_cfg["raw_shape"])
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset
    
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
    
        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark   = dark[y0:y1, x0:x1]
            dead   = dead[y0:y1, x0:x1]
            flat   = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot    = hot[y0:y1, x0:x1]
    
        np.maximum(signal, 0, out=signal)
    
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)
    
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale  = base_dt + increment
        signal[::2]  -= dark * even_scale
        signal[1::2] -= dark * odd_scale
    
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()
            bad = (dead) | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal /= flat_roi
    
        elif sensor == "AIRS-CH0":
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()
            bad = (dead[y0:y1, :]) | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi
    
        else:
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            bad2 = (dead) | ~np.isfinite(flat2) | (flat2 == 0)
            flat2[bad2] = np.nan
            signal /= flat2
    
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

        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)
            np.clip(binned, q_lo, q_hi, out=binned)

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        if sensor == "AIRS-CH0":
            var = np.nanvar(binned, axis=0, ddof=1)
            med = np.nanmedian(var)
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

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
    

# ======== 1Dトランジット深さ推定 ========
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
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
        signal_1d = savgol_filter(signal_1d, 23, 2)   # #2 は今回は触らない
        
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


# ======== メタデータ ========
StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
PlanetIds = StarInfo["planet_id"].tolist()
StarInfo = StarInfo.set_index("planet_id")


# ======== AIRS 出力（282次元）モデル ========
class ResidualBlock2(nn.Module):
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


class ResNetMLP2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=282, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock2(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


# ======== 提出生成 ========
class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", index_col="planet_id")

    def create(self, predictions1, predictions, sigma_fgs=None, sigma_air=None):
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
        # μの1列目(白色FGS)と AIRS 1:282 を元の仕様どおり再配置
        submission_df.iloc[:, 0] = predictions
        submission_df.iloc[:, 1:283] = predictions1

        #submission_df.to_csv("submission.csv")
        return submission_df


# =========================
# 実行フロー
# =========================
if __name__ == "__main__":
    config = Config()
    
    # 前処理
    signal_processor = SignalProcessor(config)
    preprocessed_data = signal_processor.process_all_data()

    # 1D深さ
    model = TransitModel(config)
    predictions = model.predict_all(preprocessed_data)

    # --- #4: 保守拡張版の σ 推定 ---
    sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
    sigma_air_vec = estimate_sigma_air(preprocessed_data, config)

    # メタ特徴作成
    predictions_df = pd.DataFrame({
        "planet_id": PlanetIds,
        "transit_depth": predictions
    })
    input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")
    input_df["transit_depth"] *= 10000
    features = ['transit_depth','Rs','i']
    X = input_df[features].values.astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # AIRS 予測モデル読み込み
    resnet2 = ResNetMLP2(num_blocks=80, dropout_rate=0.3)
    resnet2.load_state_dict(torch.load("/kaggle/input/airs/pytorch/default/1/best_model_airs.pth", map_location="cpu"))
    resnet2.eval()

    with torch.no_grad():
        predictions1 = resnet2(X_tensor).numpy()
    predictions1 /= 10000

    # --- #5: AIRS 出力の軽スムージング・アンサンブル ---
    # 波長方向(軸=1)に Savitzky–Golay で極軽く平滑化し、0.7:0.3 でブレンド
    predictions1_smooth = savgol_filter(predictions1, window_length=13, polyorder=2, axis=1)
    predictions1 = 0.65 * predictions1 + 0.35 * predictions1_smooth

    submission_generator = SubmissionGenerator(config)
    submission_solution4 = submission_generator.create(
        predictions1,
        predictions,
        sigma_fgs=sigma_fgs_vec,
        sigma_air=sigma_air_vec
    )


submission_solution4.head()


ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"

__t0 = time.perf_counter()

class Config:
    FEATURES = ['transit_depth', 'Rs', 'i']
    # FEATURES = ['transit_depth', 'Rs', 'i', 'P', 'scaled_depth']
    # FEATURES = ['transit_depth', 'Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i']
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"
    load_data = False  # Set to True to load from disk, False to generate
    DEBUG = False

    SCALE = 0.96
    SIGMA = 0.00055
    
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
    
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 11 # 9
    MODEL_POLYNOMIAL_DEGREE = 3
    
    N_JOBS = 3

class ParticipantVisibleError(Exception):
    pass

def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    naive_mean: float,
    naive_sigma: float,
    fsg_sigma_true: float = 1e-6,
    airs_sigma_true: float = 1e-5,
    fgs_weight: float = 1,
) -> float:
    """
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair of x_hat,
    x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian distributions, hence 283 values for each test spectrum,
    the GLL value for one spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        fsg_sigma_true: (float) standard deviation from the FSG1 instrument for the test set.
        airs_sigma_true: (float) standard deviation from the AIRS instrument for the test set.
        fgs_weight: (float) relative weight of the fgs channel
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError('Negative values in the submission')
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError('Wrong number of columns in the submission')

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None)
    sigma_true = np.append(
        np.array(
            [
                fsg_sigma_true,
            ]
        ),
        np.ones(n_wavelengths - 1) * airs_sigma_true,
    )
    y_true = solution.values

    GLL_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    GLL_true = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
    GLL_mean = scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true))

    # normalise the score, right now it becomes a matrix instead of a scalar.
    ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

    weights = np.append(np.array([fgs_weight]), np.ones(len(solution.columns) - 1))
    weights = weights * np.ones_like(ind_scores)
    submit_score = np.average(ind_scores, weights=weights)
    return float(np.clip(submit_score, 0.0, 1.0))

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
        # фазы по AIRS белой кривой — так же, как в модели
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
        # относительная неопределённость глубины (в тех же ед., что s)
        sigma_rel = np.sqrt(var_oot / max(n_oot,1) + var_in / max(n_in,1)) / max(oot_mean, eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    # мягкий множитель: корень, и узкий клип, чтобы не рисковать
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)  # ±20–25% от базовой σ

    return k * cfg.SIGMA

def estimate_sigma_air(preprocessed_data, cfg):
    """Возвращает вектор sigma_air длиной N_planets — мягкий множитель к cfg.SIGMA для всех AIRS-каналов."""
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12

    for single in preprocessed_data:
        # белая кривая AIRS на бинированных данных (после всех твоих весов по λ)
        white = np.nanmean(single[:, 1:], axis=1)         # (n_bins,)
        white_s = savgol_filter(white, 20, 2)             # для фаз

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

    # мягкий множитель вокруг медианы
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

        coeffs = np.flip(linear_corr, axis=0)      # shape: (D, X, Y), D — старшая степень сначала
        x = signal.astype(np.float64, copy=False)  # считаем в float64 для стабильности
        out = np.empty_like(x, dtype=np.float64)
        out[...] = coeffs[0]  # broadcast (X,Y) -> (T,X,Y)
        for k in range(1, coeffs.shape[0]):
            np.multiply(out, x, out=out)  # in-place умножение
            out += coeffs[k]              # broadcast (X,Y)

        return out.astype(signal.dtype, copy=False)

    def _calibrate_single_signal(self, planet_id, sensor):
        """
        Калибровка single-node сигнала.
        Политика масок: DEAD — маскируем, HOT — НЕ маскируем (оставляем в данных).
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
        signal = signal / gain + offset  # сохраняем твою формулу
    
        # HOT только для мониторинга, не для маскирования
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    
        # --- crop per sensor ---
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            linear_corr = linear_corr[:, :, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dark = dark[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            dead = dead[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            flat = flat[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]
            hot = hot[:, self.cfg.CUT_INF : self.cfg.CUT_SUP]  # только для логов
    
        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark   = dark[y0:y1, x0:x1]
            dead   = dead[y0:y1, x0:x1]
            flat   = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot    = hot[y0:y1, x0:x1]  # только для логов
    
        # --- non-neg clamp before linearity corr (как у тебя) ---
        np.maximum(signal, 0, out=signal)
    
        # --- linearity correction ---
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))  # T, Y, λ
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)
    
        # --- dark subtraction с учётом паттерна интеграций ---
        base_dt, increment = sensor_cfg["dt_pattern"]
        even_scale = base_dt
        odd_scale  = base_dt + increment
        signal[::2]  -= dark * even_scale
        signal[1::2] -= dark * odd_scale
    
        # --- APPLY FLAT (HOT-KEEP: не включаем hot в маску!) ---
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()      # (12,12)
            bad = (dead) | ~np.isfinite(flat_roi) | (flat_roi == 0)      # ← ТОЛЬКО dead/invalid
            flat_roi[bad] = np.nan
            signal /= flat_roi
    
        elif sensor == "AIRS-CH0":
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()  # (12, λ)
            bad = (dead[y0:y1, :]) | ~np.isfinite(flat_roi) | (flat_roi == 0)  # ← ТОЛЬКО dead/invalid
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi
    
        else:
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            bad2 = (dead) | ~np.isfinite(flat2) | (flat2 == 0)                  # ← ТОЛЬКО dead/invalid
            flat2[bad2] = np.nan
            signal /= flat2
        # --- END FLAT ---
    
        # (опционально) логируем метрики hot/dead
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

        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)    # (n_bins, 1)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)   # (n_bins, 1)
            np.clip(binned, q_lo, q_hi, out=binned)

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))

        if sensor == "AIRS-CH0":
            var = np.nanvar(binned, axis=0, ddof=1)                 # (λ, )
            med = np.nanmedian(var)
            safe_var = np.where(~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var)
            w = 1.0 / safe_var

            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)

            M = binned.shape[1]
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (M / s)
            else:
                w = np.ones_like(w)

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
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
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

StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
PlanetIds = StarInfo["planet_id"].tolist()
StarInfo = StarInfo.set_index("planet_id")
class SubmissionGenerator:
    def __init__(self, config):
        self.cfg = config
        self.sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", index_col="planet_id")

    def create(self, predictions1, predictions, sigma_fgs=None, sigma_air=None):
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
        submission_df.iloc[:, 0] = predictions
        submission_df.iloc[:, 1:283] = predictions1
        submission_df.to_csv("submission.csv")
        
        return submission_df

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
        self.act = nn.SiLU()   # Or nn.ReLU()

    def forward(self, x):
        # Compute channel-wise attention
        w = x.mean(dim=0, keepdim=True)      # Global context (mean across batch)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w   # Rescale input

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expect shape [batch, seq_len, dim], so expand if only [batch, dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)   # -> [batch, 1, dim]
        attn_out, _ = self.attn(x, x, x)
        out = self.norm(x + self.dropout(attn_out))
        return out.squeeze(1)    # Back to [batch, dim]

class ResidualBlock2(nn.Module):
    def __init__(self, dim, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        identity = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + identity)

class ResNetMLP2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=282, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock2(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


def load_cv_models_and_scalers(directory):
    """
    Loads all cross-validation models and scalers from the specified directory.
    Args:
        directory (str): Path to the directory containing model and scaler files.
    Returns:
        all_models (list): List of loaded models.
        scaler_X: Loaded X scaler.
        scaler_y: Loaded y scaler.
    """
    import os
    import joblib
    # Load scalers
    scaler_X = joblib.load(os.path.join(directory, 'scaler_X.joblib'))
    scaler_y = joblib.load(os.path.join(directory, 'scaler_y.joblib'))

    # Load all CV models
    all_models = []
    model_params = {
        'input_dim': len(Config.FEATURES),  # Always use the current feature count
        'hidden_dim': 256,
        'output_dim': 282,
        'num_blocks': 35,
        'dropout_rate': 0.1
    }
    for fold in range(1, 11):
        model = ResNetMLP2(**model_params).double()
        model_path = os.path.join(directory, f'best_model_airs_cv_fold{fold}.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        all_models.append(model)
    return all_models, scaler_X, scaler_y
    
all_models, scaler_X, scaler_y = load_cv_models_and_scalers('/kaggle/input/d/pankajiitr/ariel-2025-result/results_v37')

config = Config()
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)

predictions_df = pd.DataFrame({
    "planet_id": PlanetIds,
    "transit_depth": predictions
})

input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")
input_df['scaled_depth'] = input_df['transit_depth'] / input_df['Rs']

X = input_df[Config.FEATURES].values.astype(np.float64)
X_scaled = scaler_X.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float64)

# Generate average prediction from all CV models (on scaled X, then inverse transform)
with torch.no_grad():
    preds_scaled = [model(X_tensor).numpy() for model in all_models]
predictions1_scaled = np.mean(preds_scaled, axis=0)
predictions1 = scaler_y.inverse_transform(predictions1_scaled)

class Config:
    FEATURES = ['transit_depth', 'Rs', 'i', 'P']

    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = "test"
    load_data = False  # Set to True to load from disk, False to generate
    DEBUG = False

    SCALE = 0.96
    SIGMA = 0.00055
    
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
    
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 11 # 9
    MODEL_POLYNOMIAL_DEGREE = 3
    
    N_JOBS = 3


def load_cv_models_and_scalers(directory):
    """
    Loads all cross-validation models and scalers from the specified directory.
    Args:
        directory (str): Path to the directory containing model and scaler files.
    Returns:
        all_models (list): List of loaded models.
        scaler_X: Loaded X scaler.
        scaler_y: Loaded y scaler.
    """
    import os
    import joblib
    # Load scalers
    scaler_X = joblib.load(os.path.join(directory, 'scaler_X.joblib'))
    scaler_y = joblib.load(os.path.join(directory, 'scaler_y.joblib'))

    # Load all CV models
    all_models = []
    model_params = {
        'input_dim': len(Config.FEATURES),  # Always use the current feature count
        'hidden_dim': 256,
        'output_dim': 282,
        'num_blocks': 35,
        'dropout_rate': 0.1
    }
    for fold in range(1, 11):
        model = ResNetMLP2(**model_params).double()
        model_path = os.path.join(directory, f'best_model_airs_cv_fold{fold}.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        all_models.append(model)
    return all_models, scaler_X, scaler_y
    
all_models, scaler_X, scaler_y = load_cv_models_and_scalers('/kaggle/input/d/pankajiitr/ariel-2025-result/results_sv44')
config = Config()
signal_processor = SignalProcessor(config)
preprocessed_data = signal_processor.process_all_data()

model = TransitModel(config)
predictions = model.predict_all(preprocessed_data)
sigma_fgs_vec = estimate_sigma_fgs(preprocessed_data, config)
sigma_air_vec = estimate_sigma_air(preprocessed_data, config)

predictions_df = pd.DataFrame({
    "planet_id": PlanetIds,
    "transit_depth": predictions
})

input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")
X = input_df[Config.FEATURES].values.astype(np.float64)
X_scaled = scaler_X.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float64)

# Generate average prediction from all CV models (on scaled X, then inverse transform)
with torch.no_grad():
    preds_scaled = [model(X_tensor).numpy() for model in all_models]

# Correctly average the predictions
predictions2_scaled = np.mean(preds_scaled, axis=0)

# Correctly inverse transform the *averaged* predictions
predictions2 = scaler_y.inverse_transform(predictions2_scaled)

# Combine the predictions into a single array
all_predictions = np.array([predictions1, predictions2])

# Calculate the mean across the models (axis=0)
final_predictions = np.mean(all_predictions, axis=0)

submission_generator = SubmissionGenerator(config)
submission_solution5 = submission_generator.create(final_predictions, predictions, sigma_fgs=sigma_fgs_vec, sigma_air=sigma_air_vec)


submission_solution5.head()





import numpy as np
import pandas as pd

weights = np.array([0.03, 0.40, 0.20, 0.20, 0.17], dtype=float)
solutions = [submission_solution1, submission_solution2, submission_solution3, submission_solution4, submission_solution5]

# 可选：权重归一化
#w_sum = weights.sum()
#if not np.isclose(w_sum, 1.0):
#    weights = weights / w_sum

# 先加权平均
weighted_avg = sum(w * df for w, df in zip(weights, solutions))

# 再裁剪
weighted_avg = weighted_avg.clip(lower=1e-7)

weighted_avg.head()


weighted_avg.to_csv('submission.csv')

