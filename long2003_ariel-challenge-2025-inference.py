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


print("Hello World")


!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm


import os
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pqdm.threads import pqdm
from astropy.stats import sigma_clip
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"


class Config:
    DATA_PATH = "/kaggle/input/ariel-data-challenge-2025"
    DATASET = "train"

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
    MODEL_OPTIMIZATION_DELTA = 11
    MODEL_POLYNOMIAL_DEGREE = 3
    
    N_JOBS = 6 


#Data for the FGS1 
class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = pd.read_csv(f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}_star_info.csv",
                                     index_col = "planet_id").index.astype(int)

    def _apply_linear_corr(self, linear_corr, signal):
        # Using the Homer's method for polynomial correction
        coeffs = np.flip(linear_corr, axis = 0)
        x = signal.astype(np.float64, copy = False)

        out = np.empty_like(x, dtype = np.float64)
        out[...] = coeffs[0]

        for k in range(1, coeffs.shape[0]):
            out = np.multiply(out, signal)
            out += coeffs[k]

        return out.astype(signal.dtype, copy = False)


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

        # ADC Conversion
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

        # Using linear correction 
        if sensor == "FGS1":
            signal = self._apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            # Use ROI to reduce the risk of noise amplification
            sl = (slice(None), slice(10, 22), slice(None))
            signal[sl] = self._apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = self._apply_linear_corr(linear_corr, signal)

        
        # Dark subtraction step
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
            signal_roi = calibrated_signal[:, :, :]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)

        # Averages the signal over the spatial dimension (axis=1) 
        # to produce a 1D time series per feature (wavelength for AIRS-CH0, flattened pixels for FGS1).
        mean_signal = np.nanmean(signal_roi, axis=1)
        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        # Removing the outliers
        if sensor == "AIRS-CH0":
            # Just keeping the values with low percentile (5%) and high percentile (95%)
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

        args_airs_ch0 = [dict(planet_id=planet_id, sensor="AIRS-CH0") for planet_id in self.planet_ids]
        preprocessed_airs_ch0 = pqdm(args_airs_ch0, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)
        print(preprocessed_airs_ch0[0].shape)
        
        args_fgs1 = [dict(planet_id=planet_id, sensor="FGS1") for planet_id in self.planet_ids]
        preprocessed_fgs1 = pqdm(args_fgs1, self._process_planet_sensor, n_jobs=self.cfg.N_JOBS)
        print(preprocessed_fgs1[0].shape)
        

        preprocessed_signal = np.concatenate(
            [np.stack(preprocessed_fgs1), np.stack(preprocessed_airs_ch0)], axis=2
        )
        return preprocessed_signal


config = Config()
print(config.SENSOR_CONFIG['FGS1']['binning'])

signal_process = SignalProcessor(config)
preprocess_data = signal_process.process_all_data()




preprocess_data.shape


data1 = preprocess_data[0]
fgs1 = data1[:, 0]
plt.plot(fgs1)


link_save = '/kaggle/working/preprocess_data.npy'

np.save(link_save, preprocess_data)




