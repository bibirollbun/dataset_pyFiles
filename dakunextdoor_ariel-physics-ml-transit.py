!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm


import os, glob, itertools, time
import numpy as np
import pandas as pd

from tqdm import tqdm
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
from scipy.optimize import minimize

# Optional parallel
try:
    from pqdm.threads import pqdm
    PQDM_AVAILABLE = True
except Exception:
    PQDM_AVAILABLE = False

import torch
import torch.nn as nn

# =========================================================
# Config
# =========================================================
class Config:
    # paths
    DATA_PATH  = '/kaggle/input/ariel-data-challenge-2025'
    DATASET    = 'test'      # 'train' or 'test'
    N_JOBS     = 4           # for parallel preprocessing (if pqdm available)

    # ADC fallback defaults
    ADC_GAIN_DEFAULT   = 0.4369
    ADC_OFFSET_DEFAULT = -1000.0

    # spectral crop (39..320 -> 282 bands)
    CUT_INF = 39
    CUT_SUP = 321  # exclusive

    # pipeline toggles
    DO_MASK     = True
    DO_NL_CORR  = False
    DO_DARK     = True
    DO_FLAT     = True

    # timing pattern
    AIRS_DT_BASE, AIRS_DT_INC = 0.1, 4.5
    FGS_DT_BASE,  FGS_DT_INC  = 0.1, 0.1

    # binning
    AIRS_BIN = 30
    FGS_BIN  = 30 * 12

    # shapes
    AIRS_RAW_SHAPE = (11250, 32, 356)
    FGS_RAW_SHAPE  = (135000, 32, 32)
    LIN_AIRS_SHAPE = (6, 32, 356)
    LIN_FGS_SHAPE  = (6, 32, 32)

    # ROI
    ROI_ROWS = slice(10, 22)     # 12 rows
    ROI_COLS = slice(10, 22)     # 12 cols for FGS

    # model knobs
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA    = 11
    MODEL_POLYNOMIAL_DEGREE     = 3
    SCALE = 0.96
    SIGMA = 0.00055

    # features for ML ensemble
    FEATURES = ['transit_depth', 'Rs', 'Ms', 'Ts', 'Mp', 'e', 'P', 'sma', 'i']



class UnifiedArielPreprocessor:
    """
    Returns (N, 187, 283): [FGS1(187,1) | AIRS(187,282)]
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        adc_csv = os.path.join(cfg.DATA_PATH, 'adc_info.csv')
        self.adc_info = pd.read_csv(adc_csv) if os.path.exists(adc_csv) else None
        ids_csv = os.path.join(cfg.DATA_PATH, f'{cfg.DATASET}_star_info.csv')
        self.planet_ids = pd.read_csv(ids_csv, index_col='planet_id').index.astype(int).tolist()

    @staticmethod
    def _adc_convert(signal, gain, offset):
        signal = np.asarray(signal, dtype=np.float64)
        return signal / gain + offset

    def _adc_for(self, sensor: str):
        if self.adc_info is not None:
            g = float(self.adc_info[f"{sensor}_adc_gain"].iloc[0])
            o = float(self.adc_info[f"{sensor}_adc_offset"].iloc[0])
            return g, o
        return self.cfg.ADC_GAIN_DEFAULT, self.cfg.ADC_OFFSET_DEFAULT

    def _dt_pattern(self, sensor: str, length: int):
        if sensor == 'AIRS-CH0':
            base, inc = self.cfg.AIRS_DT_BASE, self.cfg.AIRS_DT_INC
        else:
            base, inc = self.cfg.FGS_DT_BASE, self.cfg.FGS_DT_INC
        dt = np.ones(length, dtype=np.float64) * base
        dt[1::2] += inc
        return dt

    @staticmethod
    def _mask_hot_dead(signal, dead, dark):
        hot_mask = sigma_clip(dark, sigma=5, maxiters=5).mask
        hot  = np.tile(hot_mask,  (signal.shape[0], 1, 1))
        dead = np.tile(dead,      (signal.shape[0], 1, 1))
        sig = np.ma.masked_where(dead, signal)
        sig = np.ma.masked_where(hot,  sig)
        return sig

    @staticmethod
    def _apply_linear_corr(linear_corr, clean_signal):
        lin = np.flip(linear_corr, axis=0)
        out = np.asarray(clean_signal, dtype=np.float64).copy()
        T, X, W = out.shape
        for x in range(X):
            for w in range(W):
                poly = np.poly1d(lin[:, x, w])
                out[:, x, w] = poly(out[:, x, w])
        return out

    @staticmethod
    def _clean_dark(signal, dead, dark, dt):
        dark_m = np.ma.masked_where(dead, dark)
        dark_m = np.tile(dark_m, (signal.shape[0], 1, 1))
        return signal - dark_m * dt[:, np.newaxis, np.newaxis]

    @staticmethod
    def _cds(signal):
        return signal[1::2, :, :] - signal[0::2, :, :]

    @staticmethod
    def _time_bin_cds(cds_btxw, binning):
        B, T, W, X = cds_btxw.shape
        n_bins = T // binning
        out = np.empty((B, n_bins, W, X), dtype=cds_btxw.dtype)
        for j in range(n_bins):
            out[:, j, :, :] = cds_btxw[:, j*binning:(j+1)*binning, :, :].mean(axis=1)
        return out

    @staticmethod
    def _flat_field(signal_btxw, flat_wx, dead_wx):
        B, T, W, X = signal_btxw.shape
        flat2 = np.ma.masked_where(dead_wx, flat_wx)
        flat2 = np.tile(flat2, (B, T, 1, 1))
        return signal_btxw / flat2

    def _load_calibration(self, base, sensor):
        dark = pd.read_parquet(f"{base}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(f"{base}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(f"{base}/{sensor}_calibration_0/flat.parquet").to_numpy()
        lin  = pd.read_parquet(f"{base}/{sensor}_calibration_0/linear_corr.parquet").to_numpy()
        return dark, dead, flat, lin

    def _load_signal(self, base, sensor):
        sig = pd.read_parquet(f"{base}/{sensor}_signal_0.parquet").to_numpy()
        return sig.reshape(self.cfg.AIRS_RAW_SHAPE if sensor == 'AIRS-CH0' else self.cfg.FGS_RAW_SHAPE)

    def _calibrate_sensor_full(self, planet_id, sensor):
        base = f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}"
        signal = self._load_signal(base, sensor)
        dark, dead, flat, lin = self._load_calibration(base, sensor)

        if sensor == 'AIRS-CH0':
            dark = dark.reshape(self.cfg.AIRS_RAW_SHAPE[1:])[:, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            dead = dead.reshape(self.cfg.AIRS_RAW_SHAPE[1:])[:, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            flat = flat.reshape(self.cfg.AIRS_RAW_SHAPE[1:])[:, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            lin  = lin.reshape(self.cfg.LIN_AIRS_SHAPE)[:, :, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            signal = signal[:, :, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            binning = self.cfg.AIRS_BIN
        else:
            dark = dark.reshape(self.cfg.FGS_RAW_SHAPE[1:])
            dead = dead.reshape(self.cfg.FGS_RAW_SHAPE[1:])
            flat = flat.reshape(self.cfg.FGS_RAW_SHAPE[1:])
            lin  = lin.reshape(self.cfg.LIN_FGS_SHAPE)
            binning = self.cfg.FGS_BIN

        g, o = self._adc_for(sensor)
        signal = self._adc_convert(signal, g, o)
        signal = np.clip(signal, 0, None)
        dt = self._dt_pattern(sensor, len(signal))

        if self.cfg.DO_MASK: signal = self._mask_hot_dead(signal, dead, dark)
        if self.cfg.DO_NL_CORR: signal = self._apply_linear_corr(lin, signal)
        if self.cfg.DO_DARK: signal = self._clean_dark(signal, dead, dark, dt)

        signal = np.asarray(signal)
        cds = self._cds(signal)

        if sensor == 'AIRS-CH0':
            cds_btxw = cds[np.newaxis, ...].transpose(0,1,3,2)   # (1,T',W=282,X=32)
        else:
            cds_btxw = cds[np.newaxis, ...]                       # (1,T',W=32,X=32)

        binned = self._time_bin_cds(cds_btxw, binning)

        if self.cfg.DO_FLAT:
            flat_wx, dead_wx = (flat.T, dead.T) if sensor == 'AIRS-CH0' else (flat, dead)
            binned = self._flat_field(binned, flat_wx, dead_wx)

        return np.asarray(binned)

    def _roi_and_aggregate(self, binned, sensor):
        B, T, W, X = binned.shape
        if sensor == 'AIRS-CH0':
            x_roi = binned[:, :, :, self.cfg.ROI_ROWS]             # (1,187,282,12)
            return np.nanmean(x_roi, axis=3).reshape(1, T, W)      # (1,187,282)
        else:
            roi = binned[:, :, self.cfg.ROI_ROWS, self.cfg.ROI_COLS]   # (1,187,12,12)
            return np.nanmean(roi.reshape(B, T, -1), axis=2).reshape(1, T, 1)  # (1,187,1)

    def process_all(self):
        def run_one(pid, sensor):
            binned  = self._calibrate_sensor_full(pid, sensor)
            reduced = self._roi_and_aggregate(binned, sensor)
            arr = np.asarray(reduced[0])
            if arr.ndim == 1: arr = arr.reshape(-1, 1)
            return arr

        if PQDM_AVAILABLE:
            fgs_list  = pqdm([dict(pid=p, s='FGS1') for p in self.planet_ids],
                             lambda d: run_one(d['pid'], d['s']), n_jobs=self.cfg.N_JOBS)
            airs_list = pqdm([dict(pid=p, s='AIRS-CH0') for p in self.planet_ids],
                             lambda d: run_one(d['pid'], d['s']), n_jobs=self.cfg.N_JOBS)
        else:
            fgs_list  = [run_one(p, 'FGS1') for p in tqdm(self.planet_ids, desc='FGS1')]
            airs_list = [run_one(p, 'AIRS-CH0') for p in tqdm(self.planet_ids, desc='AIRS-CH0')]

        for name, lst in (('FGS1', fgs_list), ('AIRS-CH0', airs_list)):
            for idx, item in enumerate(lst):
                if isinstance(item, Exception):
                    pid = self.planet_ids[idx] if idx < len(self.planet_ids) else None
                    raise RuntimeError(f"{name} worker failed for planet_id={pid}") from item

        fgs_arr  = np.stack([a.reshape(187,1) for a in fgs_list])   # (N,187,1)
        airs_arr = np.stack([a for a in airs_list])                 # (N,187,282)
        return np.concatenate([fgs_arr, airs_arr], axis=2)          # (N,187,283)



class TransitModel:
    def __init__(self, config: Config):
        self.cfg = config

    def _phase_detector(self, signal_1d):
        sl = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_idx = int(np.argmin(signal_1d[sl]) + sl.start)
        pre, post = signal_1d[:min_idx], signal_1d[min_idx:]
        g1 = np.gradient(pre);  g1 /= max(g1.max(), 1e-12)
        g2 = np.gradient(post); g2 /= max(g2.max(), 1e-12)
        return int(np.argmin(g1)), int(np.argmax(g2) + min_idx)

    def _objective(self, s, signal, phase1, phase2):
        delta, power = self.cfg.MODEL_OPTIMIZATION_DELTA, self.cfg.MODEL_POLYNOMIAL_DEGREE
        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or (phase2 - delta) - (phase1 + delta) < 5:
            delta = 2
        y = np.concatenate([
            signal[:phase1 - delta],
            signal[phase1 + delta:phase2 - delta] * (1 + s),
            signal[phase2 + delta:]
        ])
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=power)
        return np.mean(np.abs(np.polyval(coeffs, x) - y))

    def predict(self, single_preprocessed_signal):
        signal_1d = savgol_filter(single_preprocessed_signal[:, 1:].mean(axis=1), 23, 2)
        p1, p2 = self._phase_detector(signal_1d)
        p1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, p1)
        p2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, p2)
        res = minimize(self._objective, [0.0001], args=(signal_1d, p1, p2), method='Nelder-Mead')
        return float(res.x[0])

    def predict_all(self, preprocessed_signals):
        preds = [self.predict(sp) for sp in tqdm(preprocessed_signals, desc='TransitModel')]
        return np.array(preds) * self.cfg.SCALE



def estimate_sigma_fgs(preprocessed_data, cfg: Config):
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12
    tm = TransitModel(cfg)
    for single in preprocessed_data:
        air_white = savgol_filter(single[:, 1:].mean(axis=1), 20, 2)
        p1, p2 = tm._phase_detector(air_white)
        p1 = max(delta, p1); p2 = min(len(air_white) - delta - 1, p2)

        fgs = single[:, 0]
        oot = (fgs[: p1 - delta] if p1 - delta > 0 else np.empty(0))
        if p2 + delta < fgs.size:
            oot = np.concatenate([oot, fgs[p2 + delta:]])
        inn = fgs[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan); continue

        sigma_rel = np.sqrt(np.nanvar(oot)/max(len(oot),1) + np.nanvar(inn)/max(len(inn),1)) / max(np.nanmean(oot), eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.8, 1.25)
    return k * cfg.SIGMA


def estimate_sigma_air(preprocessed_data, cfg: Config):
    sig_rel = []
    delta = cfg.MODEL_OPTIMIZATION_DELTA
    eps = 1e-12
    tm = TransitModel(cfg)

    for single in preprocessed_data:
        white = np.nanmean(single[:, 1:], axis=1)
        white_s = savgol_filter(white, 20, 2)
        p1, p2 = tm._phase_detector(white_s)
        p1 = max(delta, p1); p2 = min(len(white) - delta - 1, p2)

        oot_left = white[: p1 - delta] if p1 - delta > 0 else np.empty(0)
        oot_right = white[p2 + delta :] if (p2 + delta) < white.size else np.empty(0)
        oot = np.concatenate([oot_left, oot_right]) if (oot_left.size + oot_right.size) else oot_left
        inn = white[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan); continue

        sigma_rel = np.sqrt(np.nanvar(oot)/max(len(oot),1) + np.nanvar(inn)/max(len(inn),1)) / max(np.nanmean(oot), eps)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0
    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    k = np.clip(k, 0.90, 1.20)
    return k * cfg.SIGMA



import os
import torch
import torch.nn as nn

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
    def __init__(self, input_dim=9, hidden_dim=256, output_dim=282, num_blocks=80, dropout_rate=0.3):
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
    Load scalers + 10 CV models from `directory`, inferring input_dim from scaler_X.n_features_in_.
    Returns: models, scaler_X, scaler_y, input_dim
    """
    import joblib
    scaler_X = joblib.load(os.path.join(directory, 'scaler_X.joblib'))
    scaler_y = joblib.load(os.path.join(directory, 'scaler_y.joblib'))

    # Infer input_dim from scaler to match checkpoints trained with 3 or 9 features.
    input_dim = int(getattr(scaler_X, "n_features_in_", None) or getattr(scaler_X, "n_features_", None) or 9)

    models = []
    params = dict(input_dim=input_dim, hidden_dim=256, output_dim=282, num_blocks=80, dropout_rate=0.3)
    for fold in range(1, 11):
        model = ResNetMLP2(**params).double()
        model_path = os.path.join(directory, f'best_model_airs_cv_fold{fold}.pth')
        state = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    return models, scaler_X, scaler_y, input_dim



class SubmissionGenerator:
    def __init__(self, config: Config):
        self.cfg = config
        self.sample_submission = pd.read_csv(
            f"{self.cfg.DATA_PATH}/sample_submission.csv",
            index_col="planet_id"
        )

    def create(self, predictions_airs, predictions_fgs, sigma_fgs=None, sigma_air=None):
        """
        predictions_airs: (N, 282) model spectra (AIRS columns)
        predictions_fgs:  (N,)     transit depth (FGS column)
        """
        K = self.sample_submission.shape[1] // 2  # 283
        mu = np.tile(predictions_airs.reshape(-1, 282), (1, 1))  # (N, 282)
        mu = np.concatenate([np.zeros((mu.shape[0], 1)), mu], axis=1)  # prepend FGS placeholder
        mu[:, 0] = predictions_fgs                                   # set FGS to depths
        mu = np.clip(mu, 0, None)

        sig = np.ones_like(mu) * self.cfg.SIGMA
        if sigma_fgs is not None: sig[:, 0]  = np.clip(sigma_fgs, 1e-6, 0.1)
        if sigma_air is not None: sig[:, 1:] = np.clip(sigma_air.reshape(-1, 1), 1e-6, 0.1)

        df = pd.DataFrame(
            np.concatenate([mu, sig], axis=1),
            columns=self.sample_submission.columns,
            index=self.sample_submission.index
        )
        df.to_csv("submission.csv")
        return df



import pandas.api.types
import scipy.stats

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
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None)
    sigma_true = np.append(np.array([fsg_sigma_true]), np.ones(n_wavelengths - 1) * airs_sigma_true)
    y_true = solution.values

    GLL_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    GLL_true = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
    GLL_mean = scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true))

    ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

    weights = np.append(np.array([fgs_weight]), np.ones(len(solution.columns) - 1))
    weights = weights * np.ones_like(ind_scores)
    submit_score = np.average(ind_scores, weights=weights)
    return float(np.clip(submit_score, 0.0, 1.0))



__t0 = time.perf_counter()

cfg = Config()
prep = UnifiedArielPreprocessor(cfg)

# (N, 187, 283)
preprocessed = prep.process_all()
print('Preprocessed shape:', preprocessed.shape)

# Transit model â†’ depths (FGS column)
tm = TransitModel(cfg)
predictions_depth = tm.predict_all(preprocessed)  # (N,)

# Per-channel sigmas
sigma_fgs_vec = estimate_sigma_fgs(preprocessed, cfg)  # (N,)
sigma_air_vec = estimate_sigma_air(preprocessed, cfg)  # (N,)

# Star info and base feature table
StarInfo = pd.read_csv(f"{cfg.DATA_PATH}/{cfg.DATASET}_star_info.csv")
StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
StarInfo = StarInfo.set_index("planet_id")
PlanetIds = StarInfo.index.tolist()

predictions_df = pd.DataFrame({"planet_id": PlanetIds, "transit_depth": predictions_depth})
input_df = pd.merge(predictions_df, StarInfo, on="planet_id", how="left")

# Helper: canonical 9-feature order
canonical_feats9 = ['transit_depth','Rs','Ms','Ts','Mp','e','P','sma','i']


# Ensemble 1 (e.g., trained with 3 features)

dir1 = '/kaggle/input/ariel-2025-result/results_3_v25'
models1, scaler_X1, scaler_y1, in_dim1 = load_cv_models_and_scalers(dir1)
# choose exact columns for dir1
feats_dir1 = ['transit_depth', 'Rs', 'i'] if in_dim1 == 3 else canonical_feats9[:in_dim1]
print(f"[Dir1] expects {in_dim1} features â†’ using columns: {feats_dir1}")

X1 = input_df[feats_dir1].values.astype(np.float64)
X1_scaled = scaler_X1.transform(X1)
X1_tensor = torch.tensor(X1_scaled, dtype=torch.float64)
with torch.no_grad():
    preds1_scaled = [m(X1_tensor).numpy() for m in models1]
predictions1_scaled = np.mean(preds1_scaled, axis=0)             # (N, 282)
predictions1 = scaler_y1.inverse_transform(predictions1_scaled)   # (N, 282)


# Ensemble 2 (e.g., trained with 9 features)
dir2 = '/kaggle/input/ariel-2025-result/results_9_v26'
models2, scaler_X2, scaler_y2, in_dim2 = load_cv_models_and_scalers(dir2)
feats_dir2 = canonical_feats9 if in_dim2 >= 9 else canonical_feats9[:in_dim2]
print(f"[Dir2] expects {in_dim2} features â†’ using columns: {feats_dir2}")

X2 = input_df[feats_dir2].values.astype(np.float64)
X2_scaled = scaler_X2.transform(X2)
X2_tensor = torch.tensor(X2_scaled, dtype=torch.float64)
with torch.no_grad():
    preds2_scaled = [m(X2_tensor).numpy() for m in models2]
predictions2_scaled = np.mean(preds2_scaled, axis=0)             # (N, 282)
predictions2 = scaler_y2.inverse_transform(predictions2_scaled)   # (N, 282)


# Average the two ML ensembles â†’ final AIRS prediction (N, 282)
final_predictions_airs = np.mean([predictions1, predictions2], axis=0)

# Submission (FGS Î¼ = transit depth; AIRS Î¼ = ensemble spectra; Ïƒ per-channel)
submission = SubmissionGenerator(cfg).create(
    predictions_airs=final_predictions_airs,
    predictions_fgs=predictions_depth,
    sigma_fgs=sigma_fgs_vec,
    sigma_air=sigma_air_vec
)

__t1 = time.perf_counter()
elapsed = __t1 - __t0
print(f"[TIMING] total runtime: {elapsed:.2f} s ({elapsed/60:.2f} min)")
pd.read_csv("submission.csv").head()


