!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm


import os, glob, itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
from scipy.optimize import minimize

# ---- Optional parallel
try:
    from pqdm.threads import pqdm
    PQDM_AVAILABLE = True
except Exception:
    PQDM_AVAILABLE = False


class Config:
    # paths
    DATA_PATH  = '/kaggle/input/ariel-data-challenge-2025'
    DATASET    = 'test'      # 'train' or 'test'
    N_JOBS     = 4           # for parallel preprocessing if pqdm is available

    # ADC (fallback defaults if adc_info.csv missing)
    ADC_GAIN_DEFAULT   = 0.4369
    ADC_OFFSET_DEFAULT = -1000.0

    # spectral crop (keep 39..320 -> 282 bands)
    CUT_INF = 39
    CUT_SUP = 321  # exclusive

    # pipeline toggles
    DO_MASK     = True
    DO_NL_CORR  = False
    DO_DARK     = True
    DO_FLAT     = True

    # timing pattern for CDS pairs
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

    # ROI (center crop used before spatial aggregation)
    ROI_ROWS = slice(10, 22)     # 12 rows
    ROI_COLS = slice(10, 22)     # 12 cols for FGS

    # model knobs (kept same for your TransitModel)
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA    = 7
    MODEL_POLYNOMIAL_DEGREE     = 3
    SCALE = 0.95
    SIGMA = 0.0009



class UnifiedArielPreprocessor:
    """
    One-stop class that:
      - Reads per-planet AIRS-CH0 & FGS1
      - ADC invert
      - mask hot/dead
      - (optional) non-linearity inverse poly
      - dark subtraction scaled by integration time
      - CDS (end - start)
      - time binning
      - flat-field division (per-planet, safe masking)
      - spatial ROI + aggregation
      - Concatenate into (N, 187, 283): [FGS1(187,1) | AIRS(187,282)]
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # optional adc table
        adc_csv = os.path.join(cfg.DATA_PATH, 'adc_info.csv')
        self.adc_info = pd.read_csv(adc_csv) if os.path.exists(adc_csv) else None

        # planet IDs from star_info
        ids_csv = os.path.join(cfg.DATA_PATH, f'{cfg.DATASET}_star_info.csv')
        self.planet_ids = pd.read_csv(ids_csv, index_col='planet_id').index.astype(int).tolist()

    # ---------- utilities ----------
    @staticmethod
    def _adc_convert(signal, gain, offset):
        signal = np.asarray(signal, dtype=np.float64)
        signal /= gain
        signal += offset
        return signal

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
        """Dead & hot masking (hot from sigma-clip on dark)."""
        signal = np.asarray(signal)  # avoid masked-array surprises later
        hot_mask = sigma_clip(dark, sigma=5, maxiters=5).mask
        hot  = np.tile(hot_mask,  (signal.shape[0], 1, 1))
        dead = np.tile(dead,      (signal.shape[0], 1, 1))
        sig = np.ma.masked_where(dead, signal)
        sig = np.ma.masked_where(hot,  sig)
        return sig

    @staticmethod
    def _apply_linear_corr(linear_corr, clean_signal):
        """Inverse polynomial per pixel across time."""
        lin = np.flip(linear_corr, axis=0)  # highest degree first for np.poly1d
        out = np.asarray(clean_signal, dtype=np.float64).copy()
        T, X, W = out.shape
        for x in range(X):
            for w in range(W):
                poly = np.poly1d(lin[:, x, w])
                out[:, x, w] = poly(out[:, x, w])
        return out

    @staticmethod
    def _clean_dark(signal, dead, dark, dt):
        signal = np.asarray(signal)
        dark_m = np.ma.masked_where(dead, dark)
        dark_m = np.tile(dark_m, (signal.shape[0], 1, 1))
        return signal - dark_m * dt[:, np.newaxis, np.newaxis]

    @staticmethod
    def _cds(signal):
        """
        Correlated Double Sampling:
        - If signal is 4D: (B, T, W, X) â†’ return (B, T/2, W, X)
        - If signal is 3D: (T, W, X)     â†’ return (T/2, W, X)
        """
        if signal.ndim == 4:
            return signal[:, 1::2, :, :] - signal[:, ::2, :, :]
        elif signal.ndim == 3:
            return signal[1::2, :, :] - signal[0::2, :, :]
        else:
            raise ValueError(f"_cds expects 3D or 4D, got shape {signal.shape}")

    @staticmethod
    def _time_bin_cds(cds_btxw, binning):
        """Average over time windows; drop remainder. cds_btxw: (B, T, W, X)"""
        cds_btxw = np.asarray(cds_btxw)
        B, T, W, X = cds_btxw.shape
        n_bins = T // binning
        out = np.empty((B, n_bins, W, X), dtype=cds_btxw.dtype)
        for j in range(n_bins):
            out[:, j, :, :] = cds_btxw[:, j*binning:(j+1)*binning, :, :].mean(axis=1)
        return out

    @staticmethod
    def _flat_field(signal_btxw, flat_wx, dead_wx):
        """
        signal_btxw: (B, T, W, X)
        flat_wx, dead_wx: (W, X)
        """
        signal_btxw = np.asarray(signal_btxw)
        B, T, W, X = signal_btxw.shape
        flat2 = np.ma.masked_where(dead_wx, flat_wx)
        flat2 = np.tile(flat2, (B, T, 1, 1))  # (B,T,W,X)
        divided = signal_btxw / flat2
        return np.asarray(divided)  # back to plain ndarray

    # ---------- per-sensor processing ----------
    def _load_calibration(self, base, sensor):
        dark = pd.read_parquet(f"{base}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(f"{base}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(f"{base}/{sensor}_calibration_0/flat.parquet").to_numpy()
        lin  = pd.read_parquet(f"{base}/{sensor}_calibration_0/linear_corr.parquet").to_numpy()
        return dark, dead, flat, lin

    def _load_signal(self, base, sensor):
        sig = pd.read_parquet(f"{base}/{sensor}_signal_0.parquet").to_numpy()
        if sensor == 'AIRS-CH0':
            sig = sig.reshape(self.cfg.AIRS_RAW_SHAPE)  # (11250, 32, 356)
        else:
            sig = sig.reshape(self.cfg.FGS_RAW_SHAPE)   # (135000, 32, 32)
        return sig

    def _calibrate_sensor_full(self, planet_id, sensor):
        base = f"{self.cfg.DATA_PATH}/{self.cfg.DATASET}/{planet_id}"

        # raw signal & calibration
        signal = self._load_signal(base, sensor)              # (T, 32, Worig)
        dark, dead, flat, lin = self._load_calibration(base, sensor)

        # reshape cal maps
        if sensor == 'AIRS-CH0':
            # maps originally (32,356)
            dark = dark.reshape(self.cfg.AIRS_RAW_SHAPE[1:])
            dead = dead.reshape(self.cfg.AIRS_RAW_SHAPE[1:])
            flat = flat.reshape(self.cfg.AIRS_RAW_SHAPE[1:])
            lin  = lin.astype(np.float64).reshape(self.cfg.LIN_AIRS_SHAPE)

            # crop wavelengths to 39..320 (â†’ 282)
            signal = signal[:, :, self.cfg.CUT_INF:self.cfg.CUT_SUP]  # (T, 32, 282)
            dark   = dark[:,  self.cfg.CUT_INF:self.cfg.CUT_SUP]      # (32, 282)
            dead   = dead[:,  self.cfg.CUT_INF:self.cfg.CUT_SUP]
            flat   = flat[:,  self.cfg.CUT_INF:self.cfg.CUT_SUP]
            lin    = lin[:, :, self.cfg.CUT_INF:self.cfg.CUT_SUP]
            base_dt, inc = self.cfg.AIRS_DT_BASE, self.cfg.AIRS_DT_INC
        else:
            # FGS maps (32,32)
            dark = dark.reshape(self.cfg.FGS_RAW_SHAPE[1:])
            dead = dead.reshape(self.cfg.FGS_RAW_SHAPE[1:])
            flat = flat.reshape(self.cfg.FGS_RAW_SHAPE[1:])
            lin  = lin.astype(np.float64).reshape(self.cfg.LIN_FGS_SHAPE)
            base_dt, inc = self.cfg.FGS_DT_BASE, self.cfg.FGS_DT_INC

        # ADC invert
        g, o = self._adc_for(sensor)
        signal = self._adc_convert(signal, g, o)

        # dt pattern
        dt = self._dt_pattern(sensor, len(signal))

        # clip
        signal = np.clip(signal, 0, None)

        # mask (returns masked array; convert to ndarray right before CDS)
        if self.cfg.DO_MASK:
            signal = self._mask_hot_dead(signal, dead, dark)

        # NL corr
        if self.cfg.DO_NL_CORR:
            signal = self._apply_linear_corr(lin, signal)

        # dark subtraction
        if self.cfg.DO_DARK:
            signal = self._clean_dark(signal, dead, dark, dt)

        # ensure ndarray (not masked) for CDS
        signal = np.asarray(signal)

        # CDS: (T, 32, W) â†’ (T/2, 32, W)
        cds = self._cds(signal)

        # To (B, T, W, X) layout for both sensors
        # We define W = spectral axis; X = spatial rows
        if sensor == 'AIRS-CH0':          # cds: (T', 32, 282)
            cds_btxw = cds[np.newaxis, ...].transpose(0, 1, 3, 2)  # (1, T', W=282, X=32)
        else:                              # FGS1: (T', 32, 32)
            # treat W=32, X=32
            cds_btxw = cds[np.newaxis, ...]                           # (1, T', 32, 32)

        # bin time (keep (B,T,W,X))
        binning = self.cfg.AIRS_BIN if sensor == 'AIRS-CH0' else self.cfg.FGS_BIN
        binned = self._time_bin_cds(cds_btxw, binning)  # (1, 187, W, X)

        # flat-field at end (safer numerically after bin)
        if self.cfg.DO_FLAT:
            if sensor == 'AIRS-CH0':
                # flat is (32, 282); need (W=282, X=32)
                flat_wx = flat.T
                dead_wx = dead.T
            else:
                flat_wx = flat
                dead_wx = dead
            binned = self._flat_field(binned, flat_wx, dead_wx)  # (1,187,W,X)

        return np.asarray(binned)  # (1, 187, W, X)

    def _roi_and_aggregate(self, binned, sensor):
        """
        Reduce spatial dims to (1, 187, K):
          K = 282 for AIRS-CH0 (mean over spatial X ROI; keep spectral W)
          K = 1   for FGS1    (mean over 12x12 ROI)
        """
        binned = np.asarray(binned)
        B, T, W, X = binned.shape  # B=1
        if sensor == 'AIRS-CH0':
            # center rows in X (spatial), mean over them â†’ keep W=282
            x_roi = binned[:, :, :, self.cfg.ROI_ROWS]              # (1,187,282,12)
            mean_over_x = np.nanmean(x_roi, axis=3)                 # (1,187,282)
            return mean_over_x.reshape(1, T, W)
        else:
            # FGS1: 12x12 center then mean â†’ scalar per time
            roi = binned[:, :, self.cfg.ROI_ROWS, self.cfg.ROI_COLS]            # (1,187,12,12)
            mean_scalar = np.nanmean(roi.reshape(B, T, -1), axis=2)             # (1,187)
            return mean_scalar.reshape(1, T, 1)

    # ---------- public API ----------
    def process_all(self):
        """Return (N, 187, 283) by concatenating FGS1(187,1) and AIRS(187,282)."""

        def run_one(pid, sensor):
            binned  = self._calibrate_sensor_full(pid, sensor)   # (1, 187, W, X)
            reduced = self._roi_and_aggregate(binned, sensor)    # (1,187,K)
            arr = np.asarray(reduced[0])                         # (187,K) or (187,)
            if arr.ndim == 1:                                    # ensure (187,1)
                arr = arr.reshape(arr.shape[0], 1)
            return arr

        # parallel or sequential
        if PQDM_AVAILABLE:
            fgs_args  = [dict(pid=p, s='FGS1')     for p in self.planet_ids]
            airs_args = [dict(pid=p, s='AIRS-CH0') for p in self.planet_ids]
            fgs_list  = pqdm(fgs_args,  lambda d: run_one(d['pid'], d['s']), n_jobs=self.cfg.N_JOBS)
            airs_list = pqdm(airs_args, lambda d: run_one(d['pid'], d['s']), n_jobs=self.cfg.N_JOBS)
        else:
            fgs_list  = [run_one(p, 'FGS1')     for p in tqdm(self.planet_ids, desc='FGS1')]
            airs_list = [run_one(p, 'AIRS-CH0') for p in tqdm(self.planet_ids, desc='AIRS-CH0')]

        # detect and raise worker exceptions early (pqdm can return Exceptions)
        for name, lst in (('FGS1', fgs_list), ('AIRS-CH0', airs_list)):
            for idx, item in enumerate(lst):
                if isinstance(item, Exception):
                    pid = self.planet_ids[idx] if idx < len(self.planet_ids) else None
                    raise RuntimeError(f"{name} worker failed for planet_id={pid}") from item

        # force proper rank before stacking
        fgs_list  = [a if a.ndim == 2 else a.reshape(a.shape[0], 1) for a in fgs_list]   # (187,1)
        airs_list = [a if a.ndim == 2 else a.reshape(a.shape[0], 1) for a in airs_list]  # (187,282)

        fgs_arr  = np.stack(fgs_list)   # (N, 187, 1)
        airs_arr = np.stack(airs_list)  # (N, 187, 282)

        # sanity checks
        assert fgs_arr.ndim == 3 and fgs_arr.shape[1] == 187 and fgs_arr.shape[2] == 1, \
            f"FGS bad shape: {fgs_arr.shape}"
        assert airs_arr.ndim == 3 and airs_arr.shape[1] == 187 and airs_arr.shape[2] >= 100, \
            f"AIRS bad shape: {airs_arr.shape}"

        return np.concatenate([fgs_arr, airs_arr], axis=2)  # (N, 187, 283)



class TransitModel:
    def __init__(self, config: Config):
        self.cfg = config

    def _phase_detector(self, signal_1d):
        sl = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_idx = int(np.argmin(signal_1d[sl]) + sl.start)

        pre = signal_1d[:min_idx]
        post = signal_1d[min_idx:]

        g1 = np.gradient(pre);  g1 /= max(g1.max(), 1e-12)
        g2 = np.gradient(post); g2 /= max(g2.max(), 1e-12)

        phase1 = int(np.argmin(g1))
        phase2 = int(np.argmax(g2) + min_idx)
        return phase1, phase2

    def _objective(self, s, signal, phase1, phase2):
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE

        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or (phase2 - delta) - (phase1 + delta) < 5:
            delta = 2

        y = np.concatenate([
            signal[:phase1 - delta],
            signal[phase1 + delta:phase2 - delta] * (1 + s),
            signal[phase2 + delta:]
        ])
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        return np.mean(np.abs(poly(x) - y))

    def predict(self, single_preprocessed_signal):
        # drop FGS1 column, average over wavelengths â†’ (187,)
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
        signal_1d = savgol_filter(signal_1d, 20, 2)

        p1, p2 = self._phase_detector(signal_1d)
        p1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, p1)
        p2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, p2)

        res = minimize(
            fun=self._objective,
            x0=[0.0001],
            args=(signal_1d, p1, p2),
            method='Nelder-Mead'
        )
        return float(res.x[0])

    def predict_all(self, preprocessed_signals):
        preds = [self.predict(sp) for sp in tqdm(preprocessed_signals, desc='Model')]
        return np.array(preds) * self.cfg.SCALE



class SubmissionGenerator:
    def __init__(self, config: Config):
        self.cfg = config
        self.sample_submission = pd.read_csv(
            f"{self.cfg.DATA_PATH}/sample_submission.csv",
            index_col="planet_id"
        )

    def create(self, predictions):
        planet_ids = self.sample_submission.index
        K = self.sample_submission.shape[1] // 2  # half are sigma columns
        vals = np.repeat(predictions, K).reshape(len(predictions), -1)
        vals = np.clip(vals, 0, None)
        sigmas = np.ones_like(vals) * self.cfg.SIGMA

        df = pd.DataFrame(
            np.concatenate([vals, sigmas], axis=1),
            columns=self.sample_submission.columns,
            index=planet_ids
        )
        df.to_csv("submission.csv")
        return df



cfg = Config()
prep = UnifiedArielPreprocessor(cfg)

# unified preprocessing â†’ (N, 187, 283)
preprocessed = prep.process_all()
print('Preprocessed shape:', preprocessed.shape)

# model
model = TransitModel(cfg)
preds = model.predict_all(preprocessed)

# submission
sub = SubmissionGenerator(cfg).create(preds)
sub.head()





