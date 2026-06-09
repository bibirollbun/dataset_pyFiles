# install pqdm for parallel processing
!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm


# Standard imports
import os
import itertools
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from astropy.stats import sigma_clip

# ML & decomposition
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.decomposition import NMF
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pqdm.threads import pqdm
from sklearn.decomposition import PCA


# axis = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")
# print(axis)
# air_wl = axis['AIRS-CH0-axis2-um'][:356]
# air_wl[39]


# test_wl = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/wavelengths.csv")
# test_wl


import pandas as pd
import numpy as np


axis = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")

air_wl = axis['AIRS-CH0-axis2-um'][:356]

test_wl = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/wavelengths.csv")


submission_airs_df = test_wl.loc[0, 'wl_2':]



# This list will store our results
mapping_results = []


for channel_name, submission_wavelength in submission_airs_df.items():
    
    # For the current submission wavelength, find the index of the closest value
    # in the full 356-pixel raw wavelength grid.
    differences = np.abs(air_wl - submission_wavelength)
    closest_raw_pixel_index = differences.idxmin() # idxmin() gives the index of the minimum value

    # Get the corresponding wavelength value from the raw grid
    closest_raw_wavelength = air_wl.loc[closest_raw_pixel_index]

    # Store the mapping information
    mapping_results.append({
        'Submission_Channel': channel_name,
        'Submission_Wavelength': submission_wavelength,
        'Closest_Raw_Pixel_Index': closest_raw_pixel_index,
        'Closest_Raw_Wavelength': closest_raw_wavelength,
        'Wavelength_Difference': np.abs(submission_wavelength - closest_raw_wavelength)
    })

mapping_df = pd.DataFrame(mapping_results)


print(mapping_df.head())

print(mapping_df.tail())



class Config:
    DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
    DATASET = 'train'

    SCALE = 0.93960
    SIGMA = 0.0009

    CUT_INF = 39
    CUT_SUP = 321

    SENSOR_CONFIG = {
        'AIRS-CH0': {
            'raw_shape': [11250, 32, 356],
            'calibrated_shape': [1, 32, CUT_SUP - CUT_INF],
            'linear_corr_shape': (6, 32, 356),
            'dt_pattern': (0.1, 4.5),
            'binning': 30
        },
        'FGS1': {
            'raw_shape': [135000, 32, 32],
            'calibrated_shape': [1, 32, 32],
            'linear_corr_shape': (6, 32, 32),
            'dt_pattern': (0.1, 0.1),
            'binning': 30 * 12
        }
    }

    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 7
    MODEL_POLYNOMIAL_DEGREE = 3

    N_JOBS = 4

    # autoencoder / nmf
    AE_ENCODING_DIM = 4
    NMF_COMPONENTS = 5

    @classmethod
    def get_planet_ids(cls):
        df = pd.read_csv(f'{cls.DATA_PATH}/{cls.DATASET}_star_info.csv', index_col='planet_id')
        ids = df.index.astype(int)
        if cls.DATASET == 'train':
            # keep a small subset for debugging like origina
            return ids[:100]
        return ids


class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.adc_info = pd.read_csv(f"{self.cfg.DATA_PATH}/adc_info.csv")
        self.planet_ids = Config.get_planet_ids()

    def _apply_linear_corr(self, linear_corr, signal):
        linear_corr_flipped = np.flip(linear_corr, axis=0)
        corrected_signal = signal.copy()
        
        for x, y in itertools.product(range(signal.shape[1]), range(signal.shape[2])):
            poly = np.poly1d(linear_corr_flipped[:, x, y])
            corrected_signal[:, x, y] = poly(corrected_signal[:, x, y])
            
        return corrected_signal

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
        
        base_dt, increment = sensor_cfg["dt_pattern"]
        dt = np.ones(len(signal)) * base_dt
        dt[1::2] += increment
        
        signal = signal.clip(0)
        signal = self._apply_linear_corr(linear_corr, signal)
        signal -= dark * dt[:, np.newaxis, np.newaxis]
        
        flat = flat.reshape(sensor_cfg["calibrated_shape"])
        flat[dead.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        flat[hot.reshape(sensor_cfg["calibrated_shape"])] = np.nan
        
        signal = signal / flat
        
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

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))
            
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

    # --- Core logic methods remain the same ---
    def _phase_detector(self, signal_1d):
        s = self.cfg.MODEL_PHASE_DETECTION_SLICE
        if s.stop > len(signal_1d): s = slice(s.start, len(signal_1d))
        if len(signal_1d[s]) == 0: return 0, len(signal_1d) - 1
        min_idx = int(np.argmin(signal_1d[s]) + s.start)
        grad = np.gradient(signal_1d)
        left_candidates = grad[:min_idx] if min_idx > 0 else np.array([])
        right_candidates = grad[min_idx:] if min_idx < len(signal_1d) else np.array([])
        left = int(np.argmax(-left_candidates)) if left_candidates.size > 0 else max(0, min_idx - 50)
        right = int(min_idx + np.argmax(right_candidates)) if right_candidates.size > 0 else min(len(signal_1d) - 1, min_idx + 50)
        return left, right

    def _objective(self, s, signal, p1, p2):
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE
        if p1 - delta <= 0 or p2 + delta >= len(signal) or (p2 - p1) < 6: delta = 2
        y = np.concatenate([signal[:p1-delta], signal[p1+delta:p2-delta]*(1+s), signal[p2+delta:]])
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        return error

    # --- NEW: Fast per-wavelength fitting function WITHOUT bootstrap ---
    def _fit_single_wave_fast(self, args):
        """
        Fits a single light curve once and calculates sigma from residuals.
        Designed to be called in parallel.
        """
        lc, phases = args # Unpack arguments
        
        if np.all(np.isnan(lc)) or lc.size < 7:
            return 0.0, 0.0

        lc_filled = np.nan_to_num(lc, nan=np.nanmean(lc))
        
        # Smooth the data once
        win = int(max(5, min(71, (len(lc_filled)//9) | 1)))
        try:
            smoothed = savgol_filter(lc_filled, window_length=win, polyorder=2)
        except Exception:
            smoothed = lc_filled

        p1, p2 = phases
        p1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, p1)
        p2 = min(len(smoothed) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, p2)
        
        # --- Fit the dip only ONCE ---
        try:
            res = minimize(self._objective, x0=[0.0001], args=(smoothed, p1, p2), method='Nelder-Mead', options={'maxiter': 200})
            s_val = float(res.x[0])
            dip = 1 - 1.0 / (s_val + 1.0) if s_val > -0.999 else 0.0
            dip = np.clip(dip, 0.0, 0.999)
        except Exception:
            s_val, dip = 0.0, 0.0

        # --- Direct Sigma Calculation (Replaces Bootstrap) ---
        # 1. Reconstruct the full trend + transit model
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE
        if p1 - delta <= 0 or p2 + delta >= len(smoothed) or (p2 - p1) < 6: delta = 2
        y_model_coords = np.concatenate([smoothed[:p1-delta], smoothed[p1+delta:p2-delta]*(1+s_val), smoothed[p2+delta:]])
        x_model_coords = np.arange(len(y_model_coords))
        coeffs = np.polyfit(x_model_coords, y_model_coords, deg=power)
        poly_model = np.poly1d(coeffs)
        
        # 2. Calculate the residuals (difference between data and model)
        residuals = smoothed - poly_model(np.arange(len(smoothed)))
        
        # 3. The sigma is the standard deviation of these residuals
        sigma = np.std(residuals)

        return dip, sigma

    # --- Main prediction method, now using parallel processing ---
    def predict_per_wavelength(self, single_preprocessed_signal):
        if single_preprocessed_signal.size == 0:
            return np.zeros(283), np.ones(283) * self.cfg.SIGMA

        # --- Step 1: Find robust phase (fast, done once) ---
        white_light_curve = np.nanmean(single_preprocessed_signal, axis=1)
        if np.all(np.isnan(white_light_curve)) or white_light_curve.size < 7:
            return np.zeros(283), np.ones(283) * self.cfg.SIGMA
        win = int(max(5, min(71, (len(white_light_curve)//9) | 1)))
        smoothed_wlc = savgol_filter(white_light_curve, window_length=win, polyorder=2)
        robust_phases = self._phase_detector(smoothed_wlc)

        # --- Step 2: Prepare arguments for parallel processing ---
        n_waves = single_preprocessed_signal.shape[1]
        args_list = [(single_preprocessed_signal[:, w], robust_phases) for w in range(n_waves)]

        # --- Step 3: Run the fits in parallel ---
        # n_jobs comes from your Config class
        results = pqdm(args_list, self._fit_single_wave_fast, n_jobs=self.cfg.N_JOBS)

        # --- Step 4: Unpack results ---
        dips = np.array([res[0] for res in results])
        dips_sigma = np.array([res[1] for res in results])

        dips = dips * self.cfg.SCALE
        dips_sigma = np.maximum(dips_sigma * self.cfg.SCALE, 1e-9)
        return dips, dips_sigma

    def predict_all_per_wavelength(self, preprocessed_signals):
        results = []
        for i in range(preprocessed_signals.shape[0]):
            # The tqdm is now implicit inside pqdm, so we can just loop here
            dips, sigs = self.predict_per_wavelength(preprocessed_signals[i])
            results.append((dips, sigs))
        raw_dips = np.vstack([r[0] for r in results])
        raw_sigmas = np.vstack([r[1] for r in results])
        return raw_dips, raw_sigmas






cfg = Config()
print('Loading and preprocessing signals...')
sp = SignalProcessor(cfg)
preprocessed = sp.process_all_data()
print('Preprocessed shape:', preprocessed.shape)


# tm = TransitModel(cfg)

# # 1. Choose the planet to inspect
# planet_to_check = 3
# preprocessed_for_one_planet = preprocessed[planet_to_check, :, :] # Get ALL wavelengths for this planet

# # 2. <<< CRITICAL NEW STEP >>>
# #    Calculate the single robust phase for this planet, exactly like your main model does.
# white_light_curve = np.nanmean(preprocessed_for_one_planet, axis=1)
# win = int(max(5, min(71, (len(white_light_curve)//9) | 1))) # Use the same smoothing
# smoothed_wlc = savgol_filter(white_light_curve, window_length=win, polyorder=3)
# p1_robust, p2_robust = tm._phase_detector(smoothed_wlc)
# robust_phases_for_planet = (p1_robust, p2_robust)

# print(f"Calculated robust phase for Planet {planet_to_check} is: {robust_phases_for_planet}")

# # 3. Choose a specific (e.g., noisy) wavelength from that planet to visualize
# wavelength_to_visualize = 0 # The noisy one from your image example
# light_curve_to_visualize = preprocessed_for_one_planet[:, wavelength_to_visualize]

# # 4. Call the plotting function, now providing the robust phase as an argument
# tm.plot_phase_detection(
#     light_curve=light_curve_to_visualize,
#     robust_phases=robust_phases_for_planet,  # <-- This is the crucial addition
#     title=f"Verification for Planet {planet_to_check}, Wavelength {wavelength_to_visualize}"
# )


class TransitModelMean:
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
        signal_1d = savgol_filter(signal_1d, 30, 2)
        
        phase1, phase2 = self._phase_detector(signal_1d)

        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    

        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead"
        )
        s = result.x[0]
        return s * self.cfg.SCALE

    def predict_all(self, preprocessed_signals):
        predictions = [
            self.predict(preprocessed_signal)
            for preprocessed_signal in tqdm(preprocessed_signals)
        ]
        return np.array(predictions)


# Hybrid Model
class TransitModelHybrid:
    def __init__(self, config):
        self.cfg = config

    # <<< Using YOUR new, clean phase detector >>>
    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]

        # Normalize the gradient to prevent issues with scale
        grad1 = np.gradient(signal1)
        if grad1.max() != 0: grad1 /= grad1.max()
        
        grad2 = np.gradient(signal2)
        if grad2.max() != 0: grad2 /= grad2.max()

        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index
        return phase1, phase2
    
    # Using the objective function from our previous model
    def _objective(self, s, signal, p1, p2):
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE
        if p1 - delta <= 0 or p2 + delta >= len(signal) or (p2 - p1) < 6:
            delta = 2
        y = np.concatenate([
            signal[: p1 - delta],
            signal[p1 + delta : p2 - delta] * (1 + s),
            signal[p2 + delta :]
        ])
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        return error
    
    # Using the _fit_single_wave method that accepts robust phases
    def _fit_single_wave(self, lc, phases, n_bootstrap=20):
        # ... this method remains exactly the same as our final 'WLC' version ...
        # It takes a light curve and FIXED phases, and returns a dip + sigma
        if np.all(np.isnan(lc)) or lc.size == 0: return 0.0, 0.0
        lc_filled = np.nan_to_num(lc, nan=np.nanmean(lc))
        L = len(lc_filled)
        if L < 7: return 0.0, 0.0
        win = int(max(5, min(71, (L//9) | 1)))
        try: smoothed = savgol_filter(lc_filled, window_length=win, polyorder=2)
        except Exception: smoothed = lc_filled
        p1, p2 = phases
        p1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, p1)
        p2 = min(len(smoothed) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, p2)
        # ... (rest of the fitting and bootstrap logic) ...
        def fit_once(signal):
            try:
                res = minimize(self._objective, x0=[0.0001], args=(signal, p1, p2), method='Nelder-Mead', options={'maxiter':400})
                s_val = float(res.x[0]) if res.success else float(res.x[0])
            except Exception:
                s_val = 0.0
            pred = 1 - 1.0 / (s_val + 1.0) if s_val > -0.999 else 0.0
            return np.clip(pred, 0.0, 0.999)
        s0 = fit_once(smoothed)
        # Bootstrap logic would follow here...
        return s0, 0.0 # Placeholder for sigma


    # Using the main PREDICTION logic from our final 'WLC' version
    def predict(self, single_preprocessed_signal):
        # 1. Create White Light Curve
        white_light_curve = np.nanmean(single_preprocessed_signal, axis=1)
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        
        # 2. Find ONE robust phase using your new detector
        robust_phases = self._phase_detector(smoothed_wlc)
        
        # 3. Apply that phase to ALL 283 wavelengths
        dips = np.zeros(single_preprocessed_signal.shape[1])
        sigmas = np.zeros(single_preprocessed_signal.shape[1])
        
        for w in range(single_preprocessed_signal.shape[1]):
            lc = single_preprocessed_signal[:, w]
            dips[w], sigmas[w] = self._fit_single_wave(lc, phases=robust_phases)
            
        return dips * self.cfg.SCALE, sigmas * self.cfg.SCALE

    def predict_all(self, preprocessed_signals):
        results = [
            self.predict(preprocessed_signal)
            for preprocessed_signal in tqdm(preprocessed_signals)
        ]
        # Properly stack dips and sigmas
        raw_dips = np.vstack([r[0] for r in results])
        raw_sigmas = np.vstack([r[1] for r in results])
        return raw_dips, raw_sigmas


class AdvancedModel:
    def __init__(self, config):
        self.cfg = config
        # Add new config parameters based on the winning solution
        self.cfg.SPECTRAL_AVG_WINDOW_LOW_SNR = 8
        self.cfg.SPECTRAL_AVG_WINDOW_HIGH_SNR = 20
        self.cfg.HIGH_SNR_WAVELENGTH_THRESHOLD = 200
        self.cfg.PCA_COMPONENTS = 5

    # We can keep your clean phase detector
    def _phase_detector(self, signal):
        # ... (This function is identical to the one in your TransitModelMean) ...
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        signal1, signal2 = signal[:min_index], signal[min_index:]
        grad1, grad2 = np.gradient(signal1), np.gradient(signal2)
        if grad1.max() != 0: grad1 /= grad1.max()
        if grad2.max() != 0: grad2 /= grad2.max()
        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index
        return phase1, phase2

    # And your objective function
    def _objective_function(self, s, signal, phase1, phase2):
        # ... (This function is identical to the one in your TransitModelMean) ...
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE
        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or (phase2 - delta) - (phase1 + delta) < 5: delta = 2
        y = np.concatenate([signal[:phase1-delta], signal[phase1+delta:phase2-delta]*(1+s), signal[phase2+delta:]])
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        return error

    # This is a new fitting function that takes a light curve and fixed phases
    def _fit_single_dip(self, light_curve, phases):
        phase1, phase2 = phases
        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(light_curve) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)
        try:
            result = minimize(fun=self._objective_function, x0=[0.0001], args=(light_curve, phase1, phase2), method="Nelder-Mead")
            s = result.x[0]
            return s * self.cfg.SCALE
        except Exception:
            return 0.0

    # This is the main prediction logic for a single planet
    def predict_spectrum(self, preprocessed_signal):
        n_bins, n_waves = preprocessed_signal.shape
        
        # --- Stage 1: Get ONE robust phase from the White Light Curve ---
        white_light_curve = np.nanmean(preprocessed_signal, axis=1)
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        robust_phases = self._phase_detector(smoothed_wlc)

        raw_dips = np.zeros(n_waves)

        # --- Stage 2: Loop through each wavelength, using local spectral averaging ---
        for k in range(n_waves):
            # IDEA: Use different averaging windows for different parts of the spectrum
            if k < self.cfg.HIGH_SNR_WAVELENGTH_THRESHOLD:
                N = self.cfg.SPECTRAL_AVG_WINDOW_LOW_SNR
            else:
                N = self.cfg.SPECTRAL_AVG_WINDOW_HIGH_SNR
            
            start = max(0, k - N)
            end = min(n_waves, k + N + 1)
            
            # Create the clean, locally-averaged light curve
            averaged_lc = np.nanmean(preprocessed_signal[:, start:end], axis=1)
            
            if np.all(np.isnan(averaged_lc)):
                raw_dips[k] = 0.0
                continue
            
            # Fit the dip using the clean data but the single robust phase
            raw_dips[k] = self._fit_single_dip(averaged_lc, robust_phases)

        # --- Stage 3: Spectrum Post-processing (Conceptual) ---
        # The winning solution would now classify the spectrum and apply rules.
        # For now, we'll apply a simple Savitzky-Golay filter as a cleaning step.
        final_dips = savgol_filter(raw_dips, 21, 3) # Window=21, order=3 is a good start

        return final_dips

    def predict_all(self, preprocessed_signals):
        # --- This function now orchestrates the full, advanced pipeline ---
        
        # First, get the refined spectrum for every planet
        all_spectra = []
        for signal in tqdm(preprocessed_signals, desc="Stage 1/2: Predicting Raw Spectra"):
            spectrum = self.predict_spectrum(signal)
            all_spectra.append(spectrum)
        
        refined_spectra = np.array(all_spectra)

        # --- Stage 4: Final spectra refinement using PCA ---
        print("Stage 2/2: Refining all spectra with PCA...")
        pca = PCA(n_components=self.cfg.PCA_COMPONENTS)
        # We need to handle potential NaNs/Infs before PCA
        refined_spectra[~np.isfinite(refined_spectra)] = 0
        
        spectra_proj = pca.fit_transform(refined_spectra)
        final_predictions = pca.inverse_transform(spectra_proj)
        
        # --- Final Step: Sigma Estimation ---
        # The winning solution uses a complex, empirical sigma.
        # We will create a simple but effective one that mimics the "humbly wrong" strategy.
        # It's a combination of a baseline sigma and a term proportional to the variance of the prediction.
        baseline_sigma = 0.0001
        variance_sigma = np.std(final_predictions, axis=1, keepdims=True) * 0.5
        final_sigmas = baseline_sigma + variance_sigma
        
        return final_predictions, np.ones_like(final_predictions) * final_sigmas


class RobustSpectrumModel:
    def __init__(self, config):
        self.cfg = config
        # It uses your successful model as its foundation!
        self.mean_model = TransitModelMean(config)

    # A simplified function to get a noisy guess at the spectral shape.
    # It's fast because it uses a fixed phase for all wavelengths.
    def _get_noisy_spectrum_shape(self, preprocessed_signal, robust_phases):
        n_waves = preprocessed_signal.shape[1]
        noisy_dips = np.zeros(n_waves)

        for w in range(n_waves):
            lc = preprocessed_signal[:, w]
            if np.all(np.isnan(lc)): continue
            
            try:
                # Use the same objective function but with the robust phase
                res = minimize(
                    fun=self.mean_model._objective_function,
                    x0=[0.0001],
                    args=(lc, robust_phases[0], robust_phases[1]),
                    method="Nelder-Mead",
                    options={'maxiter': 200} # Faster, since it's just a guess
                )
                noisy_dips[w] = res.x[0]
            except Exception:
                noisy_dips[w] = 0.0
        
        return noisy_dips * self.cfg.SCALE

    def predict(self, preprocessed_signal):
        # --- Step 1: Get the ANCHOR value using your successful model ---
        # This is the most reliable piece of information we have.
        robust_mean_dip = self.mean_model.predict(preprocessed_signal)

        # --- Step 2: Get a noisy estimate of the spectral SHAPE ---
        # We need the robust phase to run this efficiently
        white_light_curve = np.nanmean(preprocessed_signal, axis=1)
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        robust_phases = self.mean_model._phase_detector(smoothed_wlc)
        
        noisy_spectrum = self._get_noisy_spectrum_shape(preprocessed_signal, robust_phases)
        
        # --- Step 3: Combine the ANCHOR and the SHAPE ---
        mean_of_noisy_spectrum = np.nanmean(noisy_spectrum)
        
        # Isolate the shape component by removing the noisy spectrum's own mean
        shape_component = noisy_spectrum - mean_of_noisy_spectrum
        
        # Add the shape to our trusted anchor value
        final_spectrum = robust_mean_dip + shape_component
        
        return final_spectrum

    def predict_all(self, preprocessed_signals):
        # First, get the initial "anchor + shape" predictions
        initial_predictions = []
        for signal in tqdm(preprocessed_signals, desc="Stage 1/2: Predicting with Robust Mean"):
            spectrum = self.predict(signal)
            initial_predictions.append(spectrum)
        
        initial_predictions = np.array(initial_predictions)
        
        # --- Step 4: Final Polish with PCA ---
        print("Stage 2/2: Refining all spectra with PCA...")
        pca = PCA(n_components=5)
        # Ensure data is clean for PCA
        initial_predictions[~np.isfinite(initial_predictions)] = 0
        
        spectra_proj = pca.fit_transform(initial_predictions)
        final_predictions = pca.inverse_transform(spectra_proj)
        
        # --- Final Step: A Score-Optimized Sigma ---
        # This sigma strategy is designed to get high scores.
        # It's a baseline plus a term that increases with the variance of the spectrum.
        baseline_sigma = 0.0001
        variance_sigma = np.std(final_predictions, axis=1, keepdims=True) * 0.5
        final_sigmas = baseline_sigma + variance_sigma
        
        return final_predictions, np.ones_like(final_predictions) * final_sigmas


# @Ensemble Model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.decomposition import NMF
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class EnsembleBuilder:
    def __init__(self, config):
        self.cfg = config

    def gp_smooth(self, dip, dip_err):
        # x = wavelength index
        x = np.arange(len(dip)).reshape(-1, 1)
        dip_mean = np.nanmean(dip)
        y = np.nan_to_num(dip - dip_mean, nan=0.0)

        # ensure positive uncertainties
        s = np.clip(dip_err, 1e-9, None)

        # choose kernel: RBF + Matern
        y_span = max(1e-9, (y.max() - y.min()))
        kernel = (
            C(y_span, (1e-9, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1, 1e5))
            + C(y_span, (1e-9, 1e3)) * Matern(length_scale=10, length_scale_bounds=(1, 1e5), nu=1.5)
        )

        gp = GaussianProcessRegressor(kernel=kernel, alpha=(s ** 2), normalize_y=False, n_restarts_optimizer=3)
        try:
            gp.fit(x, y)
            y_pred, y_std = gp.predict(x, return_std=True)
            return (y_pred + dip_mean), y_std
        except Exception:
            return dip, np.ones_like(dip) * np.nanstd(dip)

    def fit_autoencoder(self, all_dips):
        X = np.copy(all_dips)
        # moving median smoothing per-sample
        for i in range(X.shape[0]):
            X[i] = pd.Series(X[i]).rolling(window=5, min_periods=1, center=True).median().to_numpy()
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-9
        Xn = (X - mean) / std

        input_dim = Xn.shape[1]
        encoding_dim = getattr(self.cfg, "AE_ENCODING_DIM", 4)

        inp = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(inp)
        decoded = Dense(input_dim, activation='linear')(encoded)
        ae = Model(inp, decoded)
        ae.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ae.fit(Xn, Xn, epochs=500, batch_size=128, shuffle=True, validation_split=0.05, callbacks=[es], verbose=0)

        recon = ae.predict(Xn)
        recon_un = recon * std + mean
        return recon_un, ae

    def fit_nmf(self, all_dips):
        X = np.copy(all_dips)
        # shift to non-negative
        minv = X.min()
        shift = 0.0
        if minv < 0:
            shift = -minv + 1e-6
        Xp = X + shift
        nmf = NMF(n_components=self.cfg.NMF_COMPONENTS, init='random', random_state=0, max_iter=5000)
        W = nmf.fit_transform(Xp)
        H = nmf.components_
        recon = np.dot(W, H) - shift
        return recon, nmf

    def estimate_sigma(self, dip, gp_std=None):
        # local moving-average sigma
        window_size = 20
        if len(dip) < window_size:
            sigma_local = np.std(dip)
        else:
            kernel = np.ones(window_size) / window_size
            moving_average = np.convolve(dip, kernel, mode='valid')
            sigma_local = np.std(moving_average)
        if gp_std is None:
            return sigma_local
        # combine p-mean (p=1)
        coef = 0.6
        p = 1.0
        sigma = (sigma_local**p * 0.6 + 0.4 * gp_std**p) ** (1.0 / p)
        return sigma

    def build_ensemble(self, raw_dips, raw_sigmas):
        """
        raw_dips: (n_planets, n_waves)
        raw_sigmas: (n_planets, n_waves) bootstrap errors from TransitModel
        Returns final (n_planets, n_waves), sigmas (n_planets, n_waves)
        """
        n_planets, n_waves = raw_dips.shape
        all_gp = np.zeros_like(raw_dips)
        all_gp_std = np.zeros_like(raw_dips)
        for i in range(n_planets):
            gp_pred, gp_std = self.gp_smooth(raw_dips[i], raw_sigmas[i])
            all_gp[i] = gp_pred
            all_gp_std[i] = gp_std

        recon_ae, _ = self.fit_autoencoder(raw_dips)
        # recon_nmf, _ = self.fit_nmf(raw_dips)

        # final = 0.6 * all_gp + 0.4 * recon_ae
        final = all_gp
        final = np.clip(final, 0.0036543164187456, 0.088650254994159)

        # sigma combine: planet-level moving std, gp std (wavelength), constant floor
        sigmas = np.zeros_like(final)
        for i in range(n_planets):
            gp_std = all_gp_std[i]
            for w in range(n_waves):
                sigma_w = self.estimate_sigma(raw_dips[i], gp_std=gp_std[w])
                sigmas[i, w] = sigma_w

        # scale and floor
        offset_sigma = getattr(self.cfg, "SIGMA", 1e-3)
        sigmas = sigmas * 0.35 + offset_sigma
        return final, sigmas



class DetrendingModel:
    def __init__(self, config):
        self.cfg = config
        # Let's use a fixed polynomial degree for the detrending
        self.cfg.DETREND_POLY_DEGREE = 3
        self.cfg.PCA_COMPONENTS = 5

    # We still need a phase detector to know what is "in" and "out" of transit.
    # Your clean version is perfect for this.
    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        signal1, signal2 = signal[:min_index], signal[min_index:]
        grad1, grad2 = np.gradient(signal1), np.gradient(signal2)
        if grad1.max() != 0: grad1 /= grad1.max()
        if grad2.max() != 0: grad2 /= grad2.max()
        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index
        return phase1, phase2

    # This is the main prediction logic for a single planet, implementing YOUR new idea.
    def predict_spectrum(self, preprocessed_signal):
        n_bins, n_waves = preprocessed_signal.shape
        
        # --- Stage 1: Get ONE robust phase from the White Light Curve ---
        # This part is still a good idea, as it gives us the most reliable timing.
        white_light_curve = np.nanmean(preprocessed_signal, axis=1)
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        p1, p2 = self._phase_detector(smoothed_wlc)

        # This will hold our final predicted spectrum
        final_dips = np.zeros(n_waves)

        # --- Stage 2: Loop through each wavelength and apply your detrend-then-measure logic ---
        for w in range(n_waves):
            light_curve = preprocessed_signal[:, w]
            if np.all(np.isnan(light_curve)):
                continue

            time_axis = np.arange(n_bins)

            # --- Step A: Isolate the out-of-transit data (the "tails") ---
            out_of_transit_indices = np.concatenate([time_axis[:p1], time_axis[p2:]])
            out_of_transit_flux = np.concatenate([light_curve[:p1], light_curve[p2:]])
            
            # Remove any NaNs before fitting
            valid_oot_indices = ~np.isnan(out_of_transit_flux)
            x_oot = out_of_transit_indices[valid_oot_indices]
            y_oot = out_of_transit_flux[valid_oot_indices]

            if len(x_oot) < self.cfg.DETREND_POLY_DEGREE + 1:
                continue # Not enough points to fit the polynomial

            # --- Step B: Fit the "regressor" (a polynomial) to the tails ---
            coeffs = np.polyfit(x_oot, y_oot, self.cfg.DETREND_POLY_DEGREE)
            trend_model = np.poly1d(coeffs)

            # --- Step C: Normalize the ENTIRE light curve by this trend model ---
            # This creates a light curve that is flat at y=1.0 out of transit
            normalized_light_curve = light_curve / trend_model(time_axis)

            # --- Step D: Directly measure the depth of the normalized transit ---
            in_transit_flux = normalized_light_curve[p1:p2]
            
            # The transit depth is simply 1 minus the average flux during the transit
            measured_dip = 1.0 - np.nanmean(in_transit_flux)
            
            final_dips[w] = measured_dip

        # Apply a final scaling factor from your config
        return final_dips * self.cfg.SCALE

    def predict_all(self, preprocessed_signals):
        # First, get the initial predictions using your new method
        initial_predictions = []
        for signal in tqdm(preprocessed_signals, desc="Stage 1/2: Predicting with Detrending Model"):
            spectrum = self.predict_spectrum(signal)
            initial_predictions.append(spectrum)
        
        initial_predictions = np.array(initial_predictions)
        
        # --- Final Polish with PCA (still a good idea) ---
        print("Stage 2/2: Refining all spectra with PCA...")
        pca = PCA(n_components=self.cfg.PCA_COMPONENTS)
        initial_predictions[~np.isfinite(initial_predictions)] = 0
        
        spectra_proj = pca.fit_transform(initial_predictions)
        final_predictions = pca.inverse_transform(spectra_proj)
        
        # --- A Score-Optimized Sigma ---
        baseline_sigma = 0.0001
        variance_sigma = np.std(final_predictions, axis=1, keepdims=True) * 0.5
        final_sigmas = baseline_sigma + variance_sigma
        
        return final_predictions, np.ones_like(final_predictions) * final_sigmas


def polynomial_transit_model(time, transit_depth, c0, c1, c2, c3):
    """
    Defines the transit model for curve_fit.
    - A 3rd degree polynomial for the stellar trend: c0 + c1*t + c2*t^2 + c3*t^3
    - A transit mask (which we will pass in via a global or closure)
    """
    # The transit_mask needs to be accessible here. We'll use a closure.
    trend = np.poly1d([c3, c2, c1, c0])(time)
    return trend * (1.0 - transit_mask * transit_depth)


class StableFitModel:
    def __init__(self, config):
        self.cfg = config
        self.cfg.PCA_COMPONENTS = 5

    # We still need a phase detector to create the transit mask.
    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        signal1, signal2 = signal[:min_index], signal[min_index:]
        grad1, grad2 = np.gradient(signal1), np.gradient(signal2)
        if grad1.max() != 0: grad1 /= grad1.max()
        if grad2.max() != 0: grad2 /= grad2.max()
        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index
        return phase1, phase2

    def predict_spectrum(self, preprocessed_signal):
        global transit_mask # Use a global variable to pass the mask to the fit function
        n_bins, n_waves = preprocessed_signal.shape
        
        # --- Stage 1: Get ONE robust phase from the White Light Curve ---
        white_light_curve = np.nanmean(preprocessed_signal, axis=1)
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        p1, p2 = self._phase_detector(smoothed_wlc)

        # Create the transit mask based on the robust phase
        transit_mask = np.zeros(n_bins)
        transit_mask[p1:p2] = 1.0
        
        time_axis = np.arange(n_bins)
        final_dips = np.zeros(n_waves)

        # --- Stage 2: Loop through each wavelength and use curve_fit ---
        for w in range(n_waves):
            light_curve = preprocessed_signal[:, w]
            
            # Remove NaNs for fitting
            valid_indices = ~np.isnan(light_curve)
            if not np.any(valid_indices): continue
            
            x_data = time_axis[valid_indices]
            y_data = light_curve[valid_indices]

            # Provide a good initial guess for the parameters
            p0 = [
                0.001,              # transit_depth
                np.mean(y_data),    # c0 (constant term)
                0, 0, 0             # c1, c2, c3
            ]
            
            try:
                # This is the core of the new, stable method
                params, _ = curve_fit(
                    f=polynomial_transit_model,
                    xdata=x_data,
                    ydata=y_data,
                    p0=p0,
                    maxfev=5000 # Increase max iterations for better convergence
                )
                # The first returned parameter is our transit depth
                final_dips[w] = params[0]
            except RuntimeError:
                # If curve_fit fails, we record a zero dip
                final_dips[w] = 0.0

        return final_dips * self.cfg.SCALE

    def predict_all(self, preprocessed_signals):
        initial_predictions = []
        for signal in tqdm(preprocessed_signals, desc="Stage 1/2: Predicting with StableFit Model"):
            spectrum = self.predict_spectrum(signal)
            initial_predictions.append(spectrum)
        
        initial_predictions = np.array(initial_predictions)
        
        # --- Final Polish with PCA ---
        print("Stage 2/2: Refining all spectra with PCA...")
        pca = PCA(n_components=self.cfg.PCA_COMPONENTS)
        initial_predictions[~np.isfinite(initial_predictions)] = 0
        
        spectra_proj = pca.fit_transform(initial_predictions)
        final_predictions = pca.inverse_transform(spectra_proj)
        
        # --- A Score-Optimized Sigma ---
        baseline_sigma = 0.0001
        variance_sigma = np.std(final_predictions, axis=1, keepdims=True) * 0.5
        final_sigmas = baseline_sigma + variance_sigma
        
        return final_predictions, np.ones_like(final_predictions) * final_sigmas


class MeanBaselineModel:
    def __init__(self, config):
        self.cfg = config

    # Your phase detector remains unchanged. It's proven to work on the WLC.
    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]

        grad1 = np.gradient(signal1)
        if grad1.max() != 0: grad1 /= grad1.max()
        
        grad2 = np.gradient(signal2)
        if grad2.max() != 0: grad2 /= grad2.max()

        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index

        return phase1, phase2
    
    # Your objective function remains unchanged.
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

    # This function correctly predicts the single, robust mean dip. No changes needed.
    def predict_mean_dip(self, single_preprocessed_signal):
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
        signal_1d = savgol_filter(signal_1d, 30, 2)
        
        phase1, phase2 = self._phase_detector(signal_1d)

        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    

        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead"
        )
        s = result.x[0]
        return s * self.cfg.SCALE

    # <<< THIS IS THE UPGRADED PART >>>
    def predict_all(self, preprocessed_signals):
        """
        This function now produces a full spectrum and sigma prediction.
        """
        # 1. Get the single, robust mean dip for each planet.
        mean_predictions = [
            self.predict_mean_dip(preprocessed_signal)
            for preprocessed_signal in tqdm(preprocessed_signals, desc="Predicting Robust Mean Dip")
        ]
        mean_predictions = np.array(mean_predictions) # Shape: (n_planets,)

        # 2. Create the final spectrum by broadcasting the mean value.
        # We repeat the mean value 283 times for each planet.
        n_planets = len(mean_predictions)
        n_wavelengths = 283 # As required by the competition
        
        # Reshape to (n_planets, 1) then repeat along the new axis
        final_dips = np.repeat(mean_predictions[:, np.newaxis], n_wavelengths, axis=1)

        # 3. Create a stable, score-optimized sigma.
        # A constant, reasonably large sigma is a great "humbly wrong" strategy.
        # This avoids getting a score of 0 if our flat line is far from the truth.
        final_sigmas = np.ones_like(final_dips) * 0.0008

        return final_dips, final_sigmas


class DynamicTemplateModel:
    def __init__(self, config, train_file_path):
        self.cfg = config
        # It still uses your successful model as its core
        self.mean_model = TransitModelMean(config)
        self.cfg.PCA_COMPONENTS = 5
        
        # --- NEW: Pre-calculate the average shape template from training data ---
        print("Building the average spectral shape template from training data...")
        train_df = pd.read_csv(train_file_path, index_col='planet_id')
        # For each planet in the training set, subtract its own mean to isolate the "shape"
        train_shapes = train_df.values - train_df.values.mean(axis=1, keepdims=True)
        # Average all these shapes together to get one master template
        self.average_shape_template = np.mean(train_shapes, axis=0)
        print("Template built successfully.")

    # We still need a quick way to estimate the dynamics of a new planet
    def _get_noisy_spectrum_std(self, preprocessed_signal):
        # This is a simplified version of previous models, just to get a quick estimate
        white_light_curve = np.nanmean(preprocessed_signal, axis=1)
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        robust_phases = self.mean_model._phase_detector(smoothed_wlc)
        
        n_waves = preprocessed_signal.shape[1]
        noisy_dips = np.zeros(n_waves)
        for w in range(n_waves):
            lc = preprocessed_signal[:, w]
            if np.all(np.isnan(lc)): continue
            try:
                res = minimize(fun=self.mean_model._objective_function, x0=[0.0001],
                               args=(lc, robust_phases[0], robust_phases[1]),
                               method="Nelder-Mead", options={'maxiter': 100})
                noisy_dips[w] = res.x[0]
            except Exception:
                pass
        # The standard deviation of this noisy fit is a good proxy for "dynamics"
        return np.std(noisy_dips)

    def predict(self, preprocessed_signal):
        # --- Step 1: Get the robust mean dip (the anchor) ---
        robust_mean_dip = self.mean_model.predict(preprocessed_signal)

        # --- Step 2: Classify the planet as low or high dynamics ---
        dynamics_std = self._get_noisy_spectrum_std(preprocessed_signal)
        
        # This threshold is a hyperparameter you can tune
        DYNAMICS_THRESHOLD = 0.0005 

        # --- Step 3: Choose the appropriate template ---
        if dynamics_std < DYNAMICS_THRESHOLD:
            # LOW DYNAMICS: Use the flat line template
            final_spectrum = np.full(self.average_shape_template.shape, robust_mean_dip)
        else:
            # HIGH DYNAMICS: Use the bumpy template, centered on the robust mean
            final_spectrum = robust_mean_dip + self.average_shape_template
            
        return final_spectrum

    def predict_all(self, preprocessed_signals):
        initial_predictions = []
        for signal in tqdm(preprocessed_signals, desc="Stage 1/2: Predicting with Dynamic Templates"):
            spectrum = self.predict(signal)
            initial_predictions.append(spectrum)
        
        initial_predictions = np.array(initial_predictions)
        
        # --- Final Polish with PCA ---
        print("Stage 2/2: Refining all spectra with PCA...")
        pca = PCA(n_components=self.cfg.PCA_COMPONENTS)
        initial_predictions[~np.isfinite(initial_predictions)] = 0
        
        spectra_proj = pca.fit_transform(initial_predictions)
        final_predictions = pca.inverse_transform(spectra_proj)
        
        # --- A Score-Optimized Sigma ---
        baseline_sigma = 0.0001
        variance_sigma = np.std(final_predictions, axis=1, keepdims=True) * 0.5
        final_sigmas = baseline_sigma + variance_sigma
        
        return final_predictions, final_sigmas


class FinalInstrumentModel:
    def __init__(self, config):
        self.cfg = config
        # We use the logic from your successful model as a component
        self.mean_model_logic = TransitModelMean(config)
        self.cfg.PCA_COMPONENTS = 5

    # This helper function predicts the dip for the single FGS1 channel
    def _predict_fgs1_dip(self, fgs1_light_curve):
        # fgs1_light_curve has shape (n_bins, 1).
        
        # <<< FIX: The correct method name is 'predict', not 'predict_mean_dip' >>>
        return self.mean_model_logic.predict(fgs1_light_curve)

    # This helper function predicts the full 282-point spectrum for AIRS
    def _predict_airs_spectrum(self, airs_light_curves):
        # airs_light_curves has shape (n_bins, 282)
        
        white_light_curve = np.nanmean(airs_light_curves, axis=1)
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        robust_phases = self.mean_model_logic._phase_detector(smoothed_wlc)
        
        n_waves = airs_light_curves.shape[1]
        airs_dips = np.zeros(n_waves)

        for w in range(n_waves):
            light_curve = airs_light_curves[:, w]
            if np.all(np.isnan(light_curve)): continue
            
            try:
                res = minimize(
                    fun=self.mean_model_logic._objective_function,
                    x0=[0.0001],
                    args=(light_curve, robust_phases[0], robust_phases[1]),
                    method="Nelder-Mead", options={'maxiter': 200}
                )
                airs_dips[w] = res.x[0]
            except Exception:
                airs_dips[w] = 0.0

        smoothed_airs_dips = savgol_filter(airs_dips, 21, 3)
        return smoothed_airs_dips * self.cfg.SCALE

    # The main prediction function
    def predict_all(self, preprocessed_signals):
        # preprocessed_signals is the full 3D array: (n_planets, n_bins, 283)
        
        all_final_dips = []
        n_planets = preprocessed_signals.shape[0]
        
        for i in tqdm(range(n_planets), desc="Predicting with FinalInstrumentModel"):
            planet_signal = preprocessed_signals[i]
            
            fgs1_data = planet_signal[:, 0:1]
            airs_data = planet_signal[:, 1:]
            
            fgs1_dip = self._predict_fgs1_dip(fgs1_data)
            airs_spectrum = self._predict_airs_spectrum(airs_data)
            
            final_spectrum = np.concatenate([np.array([fgs1_dip]), airs_spectrum])
            all_final_dips.append(final_spectrum)
            
        initial_predictions = np.array(all_final_dips)
        
        # Final Polish with PCA and create sigma
        print("Refining all spectra with PCA...")
        pca = PCA(n_components=self.cfg.PCA_COMPONENTS)
        initial_predictions[~np.isfinite(initial_predictions)] = 0
        
        spectra_proj = pca.fit_transform(initial_predictions)
        final_predictions = pca.inverse_transform(spectra_proj)
        
        baseline_sigma = 0.0001
        variance_sigma = np.std(final_predictions, axis=1, keepdims=True) * 0.5
        final_sigmas = baseline_sigma + variance_sigma
        
        return final_predictions, final_sigmas


class RobustPerWavelengthModel:
    def __init__(self, config):
        self.cfg = config
        # We use the core logic from your successful model as a component
        self.mean_model_logic = TransitModel(config)

    # This is the main prediction function, implementing your final strategy
    def predict_all(self, preprocessed_signals):
        # preprocessed_signals is the full 3D array: (n_planets, n_bins, 283)
        
        all_final_dips = []
        n_planets = preprocessed_signals.shape[0]
        
        for i in tqdm(range(n_planets), desc="Predicting with RobustPerWavelengthModel"):
            # Get the (n_bins, 283) data for the current planet
            planet_signal = preprocessed_signals[i]
            
            # --- Step 1: Find the single, robust phase from the WLC ---
            white_light_curve = np.nanmean(planet_signal, axis=1)
            # Handle case where WLC is all NaNs
            if np.all(np.isnan(white_light_curve)):
                all_final_dips.append(np.zeros(planet_signal.shape[1]))
                continue
            
            white_light_curve = np.nan_to_num(white_light_curve, nan=np.nanmean(white_light_curve))
            smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
            robust_phases = self.mean_model_logic._phase_detector(smoothed_wlc)
            
            # --- Step 2: Loop through each wavelength and fit its dip using the robust phase ---
            n_waves = planet_signal.shape[1]
            planet_dips = np.zeros(n_waves)

            for w in range(n_waves):
                light_curve = planet_signal[:, w]
                # Handle NaNs in the specific light curve
                if np.all(np.isnan(light_curve)):
                    continue
                light_curve = np.nan_to_num(light_curve, nan=np.nanmean(light_curve))

                # This is the "overfit" step: we fit each wavelength individually
                try:
                    res = minimize(
                        fun=self.mean_model_logic._objective_function,
                        x0=[0.0001],
                        args=(light_curve, robust_phases[0], robust_phases[1]),
                        method="Nelder-Mead",
                        options={'maxiter': 200}
                    )
                    planet_dips[w] = res.x[0]
                except (ValueError, np.linalg.LinAlgError):
                    # If the fit fails for this noisy channel, just record a zero
                    planet_dips[w] = 0.0
            
            all_final_dips.append(planet_dips * self.cfg.SCALE)
            
        initial_predictions = np.array(all_final_dips)
        
        # --- Step 3: Apply a safe, local smoother (NO PCA) ---
        print("Applying final local smoothing to each spectrum...")
        final_predictions = np.zeros_like(initial_predictions)
        for i in range(len(initial_predictions)):
            # A gentle Savitzky-Golay filter is a good choice
            final_predictions[i] = savgol_filter(initial_predictions[i], 21, 3)

        # --- Step 4: Create a score-optimized sigma ---
        baseline_sigma = 0.0001
        # The variance is calculated on the final, smoothed predictions
        variance_sigma = np.std(final_predictions, axis=1, keepdims=True) * 0.5
        final_sigmas = baseline_sigma + variance_sigma
        
        return all_final_dips, np.ones_like(final_predictions) * final_sigmas


class GP_Hybrid_Model:
    def __init__(self, config):
        self.cfg = config
        # This model uses your proven mean model as a component
        self.mean_model = TransitModelMean(config)

    # A helper function to get the noisy first-guess spectrum
    def _get_noisy_spectrum(self, preprocessed_signal):
        white_light_curve = np.nanmean(preprocessed_signal, axis=1)
        if np.all(np.isnan(white_light_curve)): return np.zeros(preprocessed_signal.shape[1])
        white_light_curve = np.nan_to_num(white_light_curve, nan=np.nanmean(white_light_curve))
        smoothed_wlc = savgol_filter(white_light_curve, 31, 2)
        robust_phases = self.mean_model._phase_detector(smoothed_wlc)
        
        n_waves = preprocessed_signal.shape[1]
        noisy_dips = np.zeros(n_waves)
        for w in range(n_waves):
            lc = preprocessed_signal[:, w]
            if np.all(np.isnan(lc)): continue
            lc = np.nan_to_num(lc, nan=np.nanmean(lc))
            try:
                res = minimize(fun=self.mean_model._objective_function, x0=[0.0001],
                               args=(lc, robust_phases[0], robust_phases[1]),
                               method="Nelder-Mead", options={'maxiter': 200})
                noisy_dips[w] = res.x[0]
            except Exception:
                pass
        return noisy_dips * self.cfg.SCALE

    # A helper function to run the Gaussian Process smoother
    def _gp_smooth(self, noisy_spectrum):
        x = np.arange(len(noisy_spectrum)).reshape(-1, 1)
        y = noisy_spectrum
        
        # A good general-purpose kernel for this kind of data
        kernel = C(1.0, (1e-4, 1e4)) * RBF(length_scale=10, length_scale_bounds=(1, 1e3))
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=np.std(y)**2)
        
        try:
            gp.fit(x, y)
            gp_mean, gp_std = gp.predict(x, return_std=True)
            return gp_mean, gp_std
        except Exception:
            # If GP fails, return a simple smoothed version and a constant sigma
            smoothed = savgol_filter(y, 21, 3)
            return smoothed, np.full_like(y, np.std(y))

    def predict_all(self, preprocessed_signals):
        all_final_dips = []
        all_final_sigmas = []
        n_planets = preprocessed_signals.shape[0]

        for i in tqdm(range(n_planets), desc="Predicting with GP Hybrid Model"):
            planet_signal = preprocessed_signals[i]

            # --- Step 1: Get the ANCHOR ---
            robust_mean_dip = self.mean_model.predict(planet_signal)

            # --- Step 2: Get the NOISY SPECTRUM ---
            noisy_spectrum = self._get_noisy_spectrum(planet_signal)

            # --- Step 3: Run the GP to get SHAPE and SIGMA ---
            gp_mean_shape, gp_std = self._gp_smooth(noisy_spectrum)

            # --- Step 4: Combine everything ---
            # Re-center the GP's smooth shape using our robust anchor
            shape_component = gp_mean_shape - np.nanmean(gp_mean_shape)
            final_dips = robust_mean_dip + shape_component
            
            # Use the GP's standard deviation as our final sigma
            final_sigmas = gp_std + self.cfg.SIGMA # Add a baseline sigma floor

            all_final_dips.append(final_dips)
            all_final_sigmas.append(final_sigmas)

        return np.array(all_final_dips), np.array(all_final_sigmas)


class SubmissionGenerator:
    def __init__(self, config: Config):
        self.cfg = config
        self.sample_submission = pd.read_csv(f"{self.cfg.DATA_PATH}/sample_submission.csv", index_col='planet_id')

    def create(self, predictions, sigmas, outpath='submission.csv'):
        planet_ids = Config.get_planet_ids()
        # predictions shape: (n_planets, n_waves)
        # sample_submission has 566 columns: first 283 are values, next 283 sigmas
        cols = self.sample_submission.columns
        ncols = len(cols)
        assert ncols % 2 == 0
        n_waves = ncols // 2
        # make sure shapes match
        assert predictions.shape[1] == n_waves

        df_vals = pd.DataFrame(predictions, index=planet_ids, columns=cols[:n_waves])
        df_sig = pd.DataFrame(sigmas, index=planet_ids, columns=cols[n_waves:])
        submission_df = pd.concat([df_vals, df_sig], axis=1)[cols]
        submission_df.to_csv(outpath)
        return submission_df


tm = GP_Hybrid_Model(cfg)
print("Predicting per-wavelength dips and bootstrap sigmas (this may take a while)...")
raw_dips, raw_sigmas = tm.predict_all(preprocessed)
print("raw_dips shape:", raw_dips.shape, "raw_sigmas shape:", raw_sigmas.shape)


# TRAIN_FILE_PATH = "/kaggle/input/ariel-data-challenge-2025/train.csv"

# # Instantiate the new model
# model = DynamicTemplateModel(cfg, train_file_path=TRAIN_FILE_PATH)

# # Run the prediction
# final_dips, final_sigmas = model.predict_all(preprocessed)
# final_sigmas = np.ones_like(final_dips) * final_sigmas


# model = TransitModelMean(cfg)
# predictions = model.predict_all(preprocessed)
# repeated_predictions = np.repeat(predictions, 283).reshape(len(predictions), -1)
# mean_dips = repeated_predictions.clip(0)
# # sigmas = np.ones_like(repeated_predictions) * self.cfg.SIGMA

# # submission_df = pd.DataFrame(
# #     np.concatenate([repeated_predictions, sigmas], axis=1),
# #     columns=self.sample_submission.columns,
# #     index=planet_ids
# # )
# mean_dips.shape


eb = EnsembleBuilder(cfg)

if len(cfg.get_planet_ids()) > 1:
    print('Building ensemble (GP + AE + NMF)...')
    final_dips, final_sigmas = eb.build_ensemble(raw_dips, raw_sigmas)
else:
    # If only one planet, fallback to original prediction style, e.g. just raw dips repeated or averaged
    final_dips = raw_dips  # shape: (1, wavelengths) assumed
    final_sigmas = raw_sigmas



# pred = final_preds
# # 1. Calculate the mean for each row (axis=1) and keep the dimension
# # The output shape will be (3, 1) instead of just (3,)
# planet_means = pred.mean(axis=1, keepdims=True)

# print("\n--- Mean for Each Row (with keepdims=True) ---")
# print(planet_means)
# print(f"Shape of planet_means: {planet_means.shape}")

# # 2. NumPy's broadcasting automatically handles the rest
# # The (3, 1) array is automatically stretched across the (3, 4) shape
# broadcasted_per_planet = np.broadcast_to(planet_means, pred.shape)


# print("\n--- Broadcasted Per-Planet Mean ---")
# print(broadcasted_per_planet)


subg = SubmissionGenerator(cfg)
print('Creating submission.csv')
submission = subg.create(raw_dips, raw_sigmas, outpath='submission.csv')
print('Saved submission.csv')
submission


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    naive_mean=0.014689019532534073,
    naive_sigma=8.71680196893868e-05,
    fsg_sigma_true: float = 1e-6,
    airs_sigma_true: float = 1e-5,
    fgs_weight: float = 1,
) -> float:
    """
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x).
    """

    n_wavelengths = len(solution.columns)

    y_pred = submission.iloc[:, :n_wavelengths].values
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=1e-15, a_max=None)
    sigma_true = np.append(
        np.array([fsg_sigma_true]),
        np.ones(n_wavelengths - 1) * airs_sigma_true,
    )
    y_true = solution.values

    # Silence divide by zero warnings from the logpdf calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        GLL_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
        GLL_true = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
        GLL_mean = scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true))

    ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
    weights = np.append(np.array([fgs_weight]), np.ones(len(solution.columns) - 1))
    weights = weights * np.ones_like(ind_scores)
    submit_score = np.average(ind_scores, weights=weights)
    return float(np.clip(submit_score, 0.0, 1.0))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import warnings


# --- Configuration and Data Loading ---
# Using the test set ground truth as the "solution" for scoring the submission
SOLUTION_FILE_PATH = "/kaggle/input/ariel-data-challenge-2025/train.csv"
SUBMISSION_FILE_PATH = "/kaggle/working/submission.csv"
OUTPUT_PLOT_FILE = "local(no_PCA)+gp.png"

# --- Main Plotting Logic ---
print("Loading data for plotting...")
# Load the ground truth labels

label_df = pd.read_csv(SOLUTION_FILE_PATH, index_col='planet_id')

pred_df = pd.read_csv(SUBMISSION_FILE_PATH, index_col='planet_id')

# Get the list of planet IDs present in both dataframes
common_planet_ids = sorted(list(set(label_df.index) & set(pred_df.index)))

if not common_planet_ids:
    print("No common planet IDs found between solution and submission files.")
else:
    print(f"Found {len(common_planet_ids)} common planets. Plotting up to 5.")
    planets_to_plot = common_planet_ids[:100]

    # Extract the corresponding dataframes for the planets we want to plot
    y_true_df = label_df.loc[planets_to_plot]
    y_submission_df = pred_df.loc[planets_to_plot]
    
    n_wavelengths = y_true_df.shape[1]
    
    # Calculate the overall score for all plotted planets
    overall_score = score(solution=y_true_df, submission=y_submission_df)

    # --- Plotting ---
    num_plots = len(planets_to_plot)
    n_cols = min(num_plots, 3)
    n_rows = (num_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5), squeeze=False)
    axes = axes.flatten()
    
    planet_scores = []

    print("Generating plots...")
    for i, planet_id in enumerate(planets_to_plot):
        ax = axes[i]
        
        # Extract data for the current planet
        solution_series = y_true_df.loc[planet_id]
        submission_series = y_submission_df.loc[planet_id]
        
        predicted_mean = submission_series[:n_wavelengths].values
        predicted_sigma = submission_series[n_wavelengths:].values

        # Plot the ground truth and the predicted mean
        ax.plot(solution_series.values, label="Ground Truth", color="cornflowerblue", linewidth=2)
        ax.plot(predicted_mean, label="Predicted Mean", color="darkorange", linestyle='--', alpha=0.9)
        
        # Add a shaded region for the uncertainty (1-sigma confidence interval)
        # print(predicted_sigma)
        ax.fill_between(range(n_wavelengths),
                        predicted_mean - predicted_sigma,
                        predicted_mean + predicted_sigma,
                        color='darkorange', alpha=0.2, label='Predicted 1Ïƒ Uncertainty')

        # The score function requires DataFrame inputs, so we convert the Series back to a DataFrame
        solution_for_scoring = pd.DataFrame(solution_series).T
        submission_for_scoring = pd.DataFrame(submission_series).T
        
        current_score = score(solution=solution_for_scoring, submission=submission_for_scoring)
        planet_scores.append(current_score)
        
        # Set the title and labels for the subplot
        ax.set_title(f"Planet: {planet_id}\nOfficial Score: {current_score:.4f}", fontsize=12)
        ax.set_xlabel("Wavelength Bin", fontsize=10)
        ax.set_ylabel("Transit Depth", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout, add a main title with the average score, and save the figure
    fig.suptitle(f"ADC 2025: Comparison of Ground Truth and Predictions\nAverage Score for Plotted Planets: {np.mean(planet_scores):.4f}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    print(f"Plot saved to {OUTPUT_PLOT_FILE}")
    plt.show()

