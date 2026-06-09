# 1) Set Up Environment and Paths
import os, sys, math, json, random, platform, pathlib, gc, time
from pathlib import Path
NOTEBOOK_NAME = "NeurIPS_Ariel_2025_baseline"
print(f"Running: {NOTEBOOK_NAME}")

# Detect Kaggle environment and define input/output directories
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
LOCAL_ROOT = Path(".")
IN_KAGGLE = KAGGLE_INPUT.exists()

if IN_KAGGLE:
    INPUT_DIR = KAGGLE_INPUT
    WORK_DIR = KAGGLE_WORKING
else:
    INPUT_DIR = LOCAL_ROOT
    WORK_DIR = Path("./_working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detect dataset root inside /kaggle/input if present
if IN_KAGGLE:
    detected = None
    try:
        for d in KAGGLE_INPUT.iterdir():
            if d.is_dir() and (d / 'sample_submission.csv').exists():
                detected = d
                break
        if detected is not None:
            INPUT_DIR = detected
            print("Detected dataset root:", INPUT_DIR)
        else:
            print("Warning: Could not detect dataset folder; using /kaggle/input root.")
    except Exception as e:
        print("Dataset detection error:", e)

SUBMISSION_FILENAME = "submission.csv"
SUBMISSION_PATH = WORK_DIR / SUBMISSION_FILENAME
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CACHE_DIR = WORK_DIR / "cache"
for d in (CHECKPOINT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Determinism and threads
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
random.seed(42)
print("Env ready. Kaggle:", IN_KAGGLE, "Input:", str(INPUT_DIR), "Work:", str(WORK_DIR))


# 2) Import Libraries and Configure Precision/Seed
import numpy as np
np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)

import pandas as pd
pd.options.display.max_rows = 50
pd.options.display.width = 140

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    pa = None; pq = None; print("PyArrow not available, will fall back to pandas read_parquet if possible.")

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any, List

# Plotting
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_context("notebook")
    sns.set_style("whitegrid")
except Exception:
    sns = None

# SciPy/Statsmodels/Sklearn optional imports
try:
    from scipy import signal, optimize, stats, linalg
except Exception as e:
    signal = optimize = stats = linalg = None
try:
    import statsmodels.api as sm
except Exception as e:
    sm = None
try:
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.model_selection import GroupKFold
except Exception as e:
    Ridge = BayesianRidge = GroupKFold = None

# Optional acceleration
USE_NUMBA = False
try:
    import numba
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False
print("Numba:", USE_NUMBA)

FLOAT = np.float64


# 3) Load Competition Metadata (CSV/Parquet)
def try_read_csv(path: Path) -> Optional[pd.DataFrame]:
    f = path if path.exists() else None
    if f is None:
        return None
    try:
        return pd.read_csv(f)
    except Exception as e:
        print("Failed to read:", f, e)
        return None

def try_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    f = path if path.exists() else None
    if f is None:
        return None
    try:
        return pd.read_parquet(f)
    except Exception as e:
        print("Failed to read:", f, e)
        return None

meta = {}
meta['train'] = try_read_csv(INPUT_DIR / 'train.csv')
meta['wavelengths'] = try_read_csv(INPUT_DIR / 'wavelengths.csv')
_axis_pq = try_read_parquet(INPUT_DIR / 'axis_info.parquet')
meta['axis_info'] = _axis_pq if _axis_pq is not None else try_read_csv(INPUT_DIR / 'axis_info.csv')
meta['adc_info'] = try_read_csv(INPUT_DIR / 'adc_info.csv')
meta['train_star'] = try_read_csv(INPUT_DIR / 'train_star_info.csv')
meta['test_star'] = try_read_csv(INPUT_DIR / 'test_star_info.csv')
meta['sample_submission'] = try_read_csv(INPUT_DIR / 'sample_submission.csv')

for k,v in meta.items():
    if v is not None:
        print(k, v.shape)
    else:
        print(k, None)

# Basic validation (soft, do not assert to keep pipeline running)
if meta['wavelengths'] is not None:
    required_wl_cols = {'instrument','wavelength','index'}
    if not required_wl_cols.issubset(set(meta['wavelengths'].columns)):
        print("Warning: wavelengths.csv missing expected columns:", required_wl_cols, "found:", list(meta['wavelengths'].columns))
else:
    print("Warning: wavelengths.csv not found; proceeding without explicit wavelength grid.")

if meta['train'] is not None:
    if 'planet_id' not in meta['train'].columns:
        print("Warning: train.csv missing planet_id column; downstream CV disabled.")

if meta['sample_submission'] is not None:
    required_sub_cols = {'planet_id','instrument','index','mu','sigma'}
    if not required_sub_cols.issubset(set(meta['sample_submission'].columns)):
        print("Warning: sample_submission.csv missing expected columns:", required_sub_cols, "found:", list(meta['sample_submission'].columns))
else:
    print("Warning: sample_submission.csv not found; will build submission skeleton from predictions only.")


def compute_linear_coeffs(df_poly: pd.DataFrame, max_degree: int = 2) -> np.ndarray:
    # Try to parse polynomial coefficients from calibration if available.
    # Expected flexible formats:
    # - Rows = pixels, columns named c0,c1,c2,...
    # - Otherwise, fall back to identity (no correction)
    if df_poly is not None:
        cols = [c for c in df_poly.columns if isinstance(c, str) and c.startswith('c') and c[1:].isdigit()]
        if cols:
            # Sort columns by degree order c0,c1,c2...
            cols_sorted = sorted(cols, key=lambda x: int(x[1:]))
            coeffs = df_poly[cols_sorted].to_numpy(dtype=np.float64)
            # Ensure up to max_degree+1 columns
            if coeffs.shape[1] < (max_degree + 1):
                # Pad with zeros
                pad = np.zeros((coeffs.shape[0], (max_degree + 1) - coeffs.shape[1]), dtype=np.float64)
                coeffs = np.hstack([coeffs, pad])
            elif coeffs.shape[1] > (max_degree + 1):
                coeffs = coeffs[:, : (max_degree + 1)]
            # Sanity: if all-zero, use identity
            if np.allclose(coeffs, 0.0):
                n_pix = df_poly.shape[0]
                coeffs = np.zeros((n_pix, max_degree+1), dtype=np.float64)
                coeffs[:,1] = 1.0
            return coeffs
    # Fallback: identity transform y = x
    n_pix = df_poly.shape[1] if (df_poly is not None and df_poly.shape[1] > 0) else 1024
    coeffs = np.zeros((n_pix, max_degree+1), dtype=np.float64)
    coeffs[:,1] = 1.0
    return coeffs


def airs_unflatten(arr: np.ndarray) -> np.ndarray:
    # arr: (n_frames, 11392) -> (n_frames, 32, 356)
    return arr.reshape(arr.shape[0], 32, 356)

def median_collapsed_image(frames_3d: np.ndarray) -> np.ndarray:
    return np.nanmedian(frames_3d, axis=0)

def find_trace_center_y(med_img: np.ndarray) -> int:
    # crude approach: sum across dispersion (x) and pick max row
    prof = np.nansum(med_img, axis=1)  # shape (32,)
    return int(np.nanargmax(prof))

def crop_airs_x(frames_3d: np.ndarray, x0: int = 39, x1: int = 321) -> np.ndarray:
    # Crop dispersion axis to recommended region
    x0 = max(0, int(x0)); x1 = min(frames_3d.shape[2], int(x1))
    return frames_3d[:, :, x0:x1]

def optimal_extract_1d(frames_3d: np.ndarray, center_y: int, half_height: int = 3) -> np.ndarray:
    # Simple box/optimal hybrid extraction, assumes frames_3d already cropped on x if desired
    y0 = max(0, center_y - half_height)
    y1 = min(frames_3d.shape[1], center_y + half_height + 1)
    sub = frames_3d[:, y0:y1, :]  # (n, h, W)
    # Profile per column
    prof = np.nanmedian(sub, axis=0)  # (h, W)
    prof = prof / (np.nanmax(prof, axis=0, keepdims=True) + 1e-12)
    # Weighted sum
    w = prof[None, :, :]
    spec = np.nansum(sub * w, axis=1)  # (n, W)
    return spec

def extract_airs_spectra_batch(calibrated_batch: np.ndarray) -> np.ndarray:
    # Input (n_frames, 11392) -> output (n_frames, W_crop)
    frames = airs_unflatten(calibrated_batch)
    # Crop dispersion axis to reduce noise and edges
    frames = crop_airs_x(frames, 39, 321)
    med_img = median_collapsed_image(frames)
    cy = find_trace_center_y(med_img)
    spec = optimal_extract_1d(frames, cy, half_height=3)
    return spec


def build_design_matrix(lightcurve: np.ndarray, centroids: Optional[Tuple[np.ndarray,np.ndarray]]=None, background: Optional[np.ndarray]=None, order: int = 2, pld_pixels: Optional[np.ndarray]=None, pld_max_pixels: int = 50) -> np.ndarray:
    n = len(lightcurve)
    cols = [np.ones(n)]
    t = np.linspace(0, 1, n)
    for k in range(1, order+1):
        cols.append(t**k)
    if centroids is not None:
        cx, cy = centroids
        cols.extend([cx, cy, cx*cx, cy*cy, cx*cy])
    if background is not None:
        cols.append(background)
    # Pixel Level Decorrelation (PLD): use brightest pixels as regressors
    if pld_pixels is not None:
        # pld_pixels: (n, H, W) or (n, P)
        P = pld_pixels
        if P.ndim == 3:
            P = P.reshape(P.shape[0], -1)
        # Select top-variance pixels to limit regressors
        var = np.nanvar(P, axis=0)
        idx = np.argsort(var)[::-1][:pld_max_pixels]
        Psel = P[:, idx]
        # Normalize each frame to sum to 1 (standard PLD)
        denom = np.nansum(Psel, axis=1, keepdims=True) + 1e-12
        Ppld = (Psel / denom)
        cols.append(Ppld)
    X = np.vstack([c if c.ndim == 1 else c.T for c in cols]).T.astype(np.float64)
    return X


# 18) Build submission.csv Matching Sample Format
def build_submission(df_pred: pd.DataFrame) -> pd.DataFrame:
    cols = ['planet_id','instrument','index','mu','sigma']
    # If sample_submission exists and has required columns, use its ordering
    sub = meta.get('sample_submission')
    if sub is not None and {'planet_id','instrument','index'}.issubset(sub.columns):
        base = sub[['planet_id','instrument','index']].copy()
        merged = base.merge(df_pred, on=['planet_id','instrument','index'], how='left')
        merged['mu'] = merged['mu'].fillna(0.0).astype(np.float64)
        merged['sigma'] = merged['sigma'].fillna(1e-3).astype(np.float64)
        return merged[cols]
    # Otherwise, build from predictions and ensure required columns
    out = df_pred.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0 if c in ('mu','sigma') else 0
    # Sort for stability
    if set(['planet_id','instrument','index']).issubset(out.columns):
        out = out.sort_values(['planet_id','instrument','index']).reset_index(drop=True)
    return out[cols]


# 17) Hyperparameters: defaults + load/save helpers
HP_PATH = WORK_DIR / 'hparams.json'
DEFAULT_HPARAMS = {
    'fgs_aperture_r': 5.0,
    'poly_order': 2,
    'sigma_clip': 5.0,
    'airs_binned_len': 282,
}
def load_hparams(path: Path = HP_PATH, defaults: dict = DEFAULT_HPARAMS) -> dict:
    try:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            # Merge with defaults to ensure missing keys are filled
            out = defaults.copy()
            out.update({k: v for k, v in data.items() if k in defaults or True})
            return out
    except Exception as e:
        print('Warning: failed to load hparams:', e)
    return defaults.copy()
def save_hparams(hp: dict, path: Path = HP_PATH) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(hp, f, indent=2)
    except Exception as e:
        print('Warning: failed to save hparams:', e)


# 17b) Bootstrap helpers if running cells out of order
try:
    _ = load_adc_info
except NameError:
    def load_adc_info(meta: dict) -> tuple[float, float]:
        df = meta.get('adc_info')
        if df is None:
            return 1.0, 0.0
        gain = float(df.loc[0, 'gain']) if 'gain' in df.columns else 1.0
        offset = float(df.loc[0, 'offset']) if 'offset' in df.columns else 0.0
        return gain, offset
    print('Defined minimal load_adc_info() bootstrap.')


# 21a) Minimal run_inference stub (baseline)
# This placeholder mirrors sample_submission rows for the selected planets when possible,
# otherwise it generates a tiny default set of rows so the pipeline can complete end-to-end.
# Replace with the full extraction/detrending-based inference when ready.
def run_inference(split: str, planets: List[str]) -> pd.DataFrame:
    cols = ['planet_id','instrument','index','mu','sigma']
    sub = meta.get('sample_submission')
    # Case A: sample_submission has the expected key columns; mirror it for requested planets
    if sub is not None and {'planet_id','instrument','index'}.issubset(set(sub.columns)):
        planet_keys = [str(p) for p in planets]
        mask = sub['planet_id'].astype(str).isin(planet_keys)
        base = sub.loc[mask, ['planet_id','instrument','index']].copy()
        if base.empty:
            # If none of the requested planets are in sample, fallback to all
            base = sub[['planet_id','instrument','index']].copy()
        base['mu'] = 0.0
        base['sigma'] = 1e-3
        # Preserve dtypes of key columns where possible
        try:
            base['planet_id'] = base['planet_id'].astype(sub['planet_id'].dtype)
            base['instrument'] = base['instrument'].astype(sub['instrument'].dtype)
            base['index'] = base['index'].astype(sub['index'].dtype)
        except Exception:
            pass
        return base
    # Case B: sample_submission exists but lacks required columns OR sub is None
    # Build a minimal default skeleton: two instruments x one index per requested planet
    rows = []
    default_instruments = ['FGS1', 'AIRS-CH0']
    for pid in planets:
        for instr in default_instruments:
            rows.append({
                'planet_id': pid,
                'instrument': instr,
                'index': 0,
                'mu': 0.0,
                'sigma': 1e-3,
            })
    df = pd.DataFrame(rows, columns=cols)
    return df


# 20) Performance and Memory Controls (chunking, caching)
CONFIG = {
    'batch_rows_fgs': 5000,
    'batch_rows_airs': 1024,
    'use_cache': True
}
print("Config:", CONFIG)

# 21) Reproducibility and Logging + Main pipeline
def log_env():
    # Import locally to avoid issues if global names are shadowed by variables elsewhere
    import sys as _sys
    import platform as _platform
    import numpy as _np
    import pandas as _pd
    print("Python:", _sys.version)
    print("Platform:", _platform.platform())
    print("Numpy:", _np.__version__, "Pandas:", _pd.__version__)
    try:
        import pyarrow as _pa
        print("PyArrow:", _pa.__version__)
    except Exception:
        pass
log_env()

def list_planets(split: str) -> List[str]:
    base = INPUT_DIR / split
    if not base.exists(): return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])

# Main execution: try test set; if unavailable, do a small local dry run
hp = load_hparams()
print("Hyperparameters:", hp)

adc_gain, adc_offset = load_adc_info(meta)
print("ADC gain/offset:", adc_gain, adc_offset)

test_planets = list_planets('test')
if len(test_planets)==0:
    print("No test planets found; attempting train planets for smoke test...")
    test_planets = list_planets('train')[:1]  # limit to 1 for quick local check

if len(test_planets)>0:
    preds = run_inference('test' if (INPUT_DIR / 'test').exists() else 'train', test_planets)
    if len(preds)>0:
        sub = build_submission(preds)
        sub.to_csv(SUBMISSION_PATH, index=False)
        print("Wrote:", SUBMISSION_PATH)
        display(sub.head())
    else:
        print("No predictions generated.")
else:
    # fallback sample submission if available
    if meta['sample_submission'] is not None:
        meta['sample_submission'][['planet_id','instrument','index']].assign(mu=0.0, sigma=1e-3).to_csv(SUBMISSION_PATH, index=False)
        print("No data found. Wrote empty baseline submission:", SUBMISSION_PATH)
    else:
        print("No data or sample submission found — nothing to write.")


# Utilities: parquet iteration, FGS helpers, calibration stubs, GLS, time/wavelength
import numpy as _np
import pandas as _pd
from typing import Iterator as _Iterator, Optional as _Optional, Tuple as _Tuple

# Stream parquet rows in manageable chunks; uses PyArrow when available, else Pandas fallback
def iter_parquet_rows(path: Path, batch_rows: int = 2000) -> _Iterator[_np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    # Prefer PyArrow for large files
    if pq is not None:
        try:
            pf = pq.ParquetFile(str(path))
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg)
                df = tbl.to_pandas()
                arr = df.to_numpy()
                # If cells are arrays/series per-row, try to expand
                if arr.dtype == object and hasattr(arr[0, 0], '__len__'):
                    arr = _np.vstack(arr[:, 0]).astype(_np.float64)
                for i in range(0, len(arr), batch_rows):
                    yield _np.ascontiguousarray(arr[i:i+batch_rows], dtype=_np.float64)
            return
        except Exception:
            pass
    # Pandas fallback (loads whole file then chunks)
    df = _pd.read_parquet(str(path))
    arr = df.to_numpy()
    if arr.dtype == object and hasattr(arr[0, 0], '__len__'):
        arr = _np.vstack(arr[:, 0]).astype(_np.float64)
    for i in range(0, len(arr), batch_rows):
        yield _np.ascontiguousarray(arr[i:i+batch_rows], dtype=_np.float64)

# FGS reshape helper: (n, 1024) -> (n, 32, 32)
def fgs_unflatten(arr: _np.ndarray) -> _np.ndarray:
    arr = _np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError('Expected 2D array (n, P)')
    n, P = arr.shape
    if P == 1024:
        return arr.reshape(n, 32, 32)
    # Heuristic fallback: try square
    s = int(round(P ** 0.5))
    if s * s == P:
        return arr.reshape(n, s, s)
    # Last resort: assume (H=32, W=P//32)
    H = 32
    W = P // H
    if H * W != P:
        raise ValueError(f'Cannot reshape FGS array of length {P}')
    return arr.reshape(n, H, W)

# Simple centroid per frame (weighted by positive flux)
def compute_centroids(frames: _np.ndarray) -> _Tuple[_np.ndarray, _np.ndarray]:
    f3 = _np.asarray(frames, dtype=_np.float64)
    n, H, W = f3.shape
    yy, xx = _np.indices((H, W))
    # Shift to non-negative weights
    f = f3 - _np.nanmin(f3, axis=(1, 2), keepdims=True)
    f = _np.clip(f, 0, None)
    denom = _np.nansum(f, axis=(1, 2)) + 1e-12
    cx = _np.nansum(f * xx[None, :, :], axis=(1, 2)) / denom
    cy = _np.nansum(f * yy[None, :, :], axis=(1, 2)) / denom
    return cx, cy

# Aperture photometry around per-frame centroid
def aperture_photometry(frames: _np.ndarray, r: float = 5.0) -> _np.ndarray:
    f3 = _np.asarray(frames, dtype=_np.float64)
    n, H, W = f3.shape
    cx, cy = compute_centroids(f3)
    yy, xx = _np.indices((H, W))
    out = _np.empty(n, dtype=_np.float64)
    rr2 = None
    for i in range(n):
        if rr2 is None or True:
            rr2 = (xx - cx[i])**2 + (yy - cy[i])**2
        mask = rr2 <= (r*r)
        out[i] = _np.nansum(f3[i][mask])
    return out

# Minimal calibration pipeline: ADC restore only (gain/offset). Masters ignored for now.
def load_build_calibration(split: str, planet_id: str, instrument: str, gain: float, offset: float) -> dict:
    return {}

def calibrate_batch_frames(batch: _np.ndarray, masters: dict, gain: float, offset: float) -> _np.ndarray:
    arr = _np.asarray(batch, dtype=_np.float64)
    g = gain if gain not in (None, 0) else 1.0
    return (arr - (offset or 0.0)) / g

# Simple GLS (OLS) solver
def generalized_least_squares(y: _np.ndarray, X: _np.ndarray) -> _Tuple[_np.ndarray, _np.ndarray, float]:
    y = _np.asarray(y, dtype=_np.float64)
    X = _np.asarray(X, dtype=_np.float64)
    # Mask non-finite rows
    m = _np.isfinite(y) & _np.all(_np.isfinite(X), axis=1)
    if not _np.any(m):
        beta = _np.zeros(X.shape[1], dtype=_np.float64)
        yhat = _np.full_like(y, _np.nan, dtype=_np.float64)
        return beta, yhat, _np.nan
    beta, *_ = _np.linalg.lstsq(X[m], y[m], rcond=None)
    yhat = X @ beta
    resid = y[m] - (X[m] @ beta)
    s2 = float(_np.nanvar(resid))
    return beta, yhat, s2

# Wavelength/time helpers
def get_wavelength_grid(meta: dict, instrument: str) -> _Optional[_np.ndarray]:
    df = meta.get('wavelengths')
    if df is None:
        return None
    try:
        d = df[df['instrument'].astype(str) == str(instrument)].copy()
        if 'index' in d.columns:
            d = d.sort_values('index')
        if 'wavelength' in d.columns:
            return d['wavelength'].to_numpy(dtype=_np.float64)
    except Exception:
        pass
    return None

def build_time_axis(meta: dict, instrument: str, n: int) -> _np.ndarray:
    # If axis_info contains a suitable cadence, we could integrate it; for robustness, use 0..1.
    if n <= 1:
        return _np.zeros(n, dtype=_np.float64)
    return _np.linspace(0.0, 1.0, int(n))


# Visualization: AIRS median image and trace profile
try:
    planets = [p for p in (INPUT_DIR / ('test' if (INPUT_DIR / 'test').exists() else 'train')).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        p = signal_path('test' if (INPUT_DIR / 'test').exists() else 'train', pid, 'AIRS-CH0', 0)
        if p.exists():
            gain, offset = load_adc_info(meta)
            masters = load_build_calibration('test' if (INPUT_DIR / 'test').exists() else 'train', pid, 'AIRS-CH0', gain, offset)
            # take a small batch
            for batch in iter_parquet_rows(p, batch_rows=256):
                cal = calibrate_batch_frames(batch, masters, gain, offset)
                frames = airs_unflatten(cal)
                med = np.nanmedian(frames, axis=0)
                prof_y = np.nansum(med, axis=1)
                prof_x = np.nansum(med, axis=0)
                fig, axs = plt.subplots(1,3, figsize=(14,4))
                im = axs[0].imshow(med, aspect='auto', origin='lower')
                axs[0].set_title(f'AIRS median image (planet {pid})')
                plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
                axs[1].plot(prof_y)
                axs[1].set_title('Cross-dispersion profile (sum over x)')
                axs[2].plot(prof_x)
                axs[2].set_title('Dispersion profile (sum over y)')
                plt.tight_layout()
                break
except Exception as e:
    print('AIRS visualization skipped:', e)


# Visualization: FGS1 light curve preview
try:
    planets = [p for p in (INPUT_DIR / ('test' if (INPUT_DIR / 'test').exists() else 'train')).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        p = signal_path('test' if (INPUT_DIR / 'test').exists() else 'train', pid, 'FGS1', 0)
        if p.exists():
            gain, offset = load_adc_info(meta)
            masters = load_build_calibration('test' if (INPUT_DIR / 'test').exists() else 'train', pid, 'FGS1', gain, offset)
            vals = []
            for batch in iter_parquet_rows(p, batch_rows=5000):
                cal = calibrate_batch_frames(batch, masters, gain, offset)
                f3 = fgs_unflatten(cal)
                lc = aperture_photometry(f3, r=5.0)
                vals.append(lc)
                if len(np.concatenate(vals)) > 5000:
                    break
            if vals:
                lc_full = np.concatenate(vals)
                t = build_time_axis(meta, 'FGS1', len(lc_full))
                plt.figure(figsize=(12,3))
                plt.plot(t, lc_full/np.nanmedian(lc_full), '-', lw=0.5)
                plt.title(f'FGS1 light curve (planet {pid})')
                plt.xlabel('time')
                plt.ylabel('normalized flux')
                plt.tight_layout()
except Exception as e:
    print('FGS1 visualization skipped:', e)


# Visualization: Quick AIRS transmission spectrum (depth vs wavelength)
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        p = signal_path(split, pid, 'AIRS-CH0', 0)
        if p.exists():
            # Load helpers
            gain, offset = load_adc_info(meta)
            masters = load_build_calibration(split, pid, 'AIRS-CH0', gain, offset)
            # Stream a limited number of frames for speed
            frames = []
            max_frames = 1200
            for batch in iter_parquet_rows(p, batch_rows=2000):
                cal = calibrate_batch_frames(batch, masters, gain, offset)
                f3 = airs_unflatten(cal)  # (n, 32, 356)
                # Apply same crop used in extraction for cleaner spectra
                f3 = crop_airs_x(f3, 39, 321)
                frames.append(f3)
                if sum(x.shape[0] for x in frames) >= max_frames:
                    break
            if frames:
                cube = np.concatenate(frames, axis=0)  # (T, 32, W)
                T = cube.shape[0]
                # Collapse cross-dispersion to 1D spectra (simple sum; we already calibrated)
                spec = cube.sum(axis=1)  # (T, W)
                # Build time axis and choose a naive transit window (middle 20% as in, edges 40% as out)
                t = build_time_axis(meta, 'AIRS-CH0', T)
                i0, i1 = int(0.4*T), int(0.6*T)
                in_m = np.zeros(T, dtype=bool)
                in_m[i0:i1] = True
                out_m = ~in_m
                # Normalize each wavelength by its out-of-transit median
                out_med = np.nanmedian(spec[out_m], axis=0)
                out_med[out_med==0] = np.nan
                norm = spec / out_med[np.newaxis, :]
                # Depth per wavelength: 1 - mean_in
                mean_in = np.nanmean(norm[in_m], axis=0)
                mean_out = np.nanmean(norm[out_m], axis=0)
                var_in = np.nanvar(norm[in_m], axis=0)/(in_m.sum() + 1e-9)
                var_out = np.nanvar(norm[out_m], axis=0)/(out_m.sum() + 1e-9)
                depth = 1.0 - (mean_in/mean_out)
                sigma = np.sqrt(var_in + var_out)
                # Optional wavelength grid (crop-aware if full grid available)
                waves = None
                try:
                    w = get_wavelength_grid(meta, 'AIRS-CH0')
                    if w is not None:
                        w = w[39:321]
                        if len(w) == spec.shape[1]:
                            waves = w
                except Exception:
                    pass
                plt.figure(figsize=(12,4))
                x = np.arange(spec.shape[1]) if waves is None else waves
                plt.plot(x, depth, color='tab:blue', lw=1)
                if np.isfinite(sigma).any():
                    lo = depth - sigma
                    hi = depth + sigma
                    plt.fill_between(x, lo, hi, color='tab:blue', alpha=0.2, linewidth=0)
                plt.title(f'AIRS quick spectrum (planet {pid}, visit 0)')
                plt.xlabel('wavelength' if waves is not None else 'pixel column (cropped)')
                plt.ylabel('transit depth (arb)')
                plt.tight_layout()
                # Save figure
                fig_dir = WORK_DIR / 'figures'
                fig_dir.mkdir(parents=True, exist_ok=True)
                out_path = fig_dir / f'{pid}_AIRS-CH0_quick_spectrum.png'
                plt.savefig(out_path, dpi=150)
                print('Saved:', out_path)
except Exception as e:
    print('AIRS spectrum visualization skipped:', e)


# Visualization: FGS centroids + raw vs detrended, and save figures
from pathlib import Path
fig_dir = WORK_DIR / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        p = signal_path(split, pid, 'FGS1', 0)
        if p.exists():
            # Load calibration
            gain, offset = load_adc_info(meta)
            masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
            # Stream frames and compute time, flux, centroids, and keep small cube for PLD
            flux_chunks, cx_chunks, cy_chunks = [], [], []
            cubes = []
            cap_frames = 4000
            for batch in iter_parquet_rows(p, batch_rows=4000):
                cal = calibrate_batch_frames(batch, masters, gain, offset)
                f3 = fgs_unflatten(cal)
                flux_chunks.append(aperture_photometry(f3, r=5.0))
                cx, cy = compute_centroids(f3)
                cx_chunks.append(cx)
                cy_chunks.append(cy)
                if sum(len(a) for a in flux_chunks) < cap_frames:
                    cubes.append(f3)
                if sum(len(a) for a in flux_chunks) >= 8000:
                    break
            if flux_chunks:
                flux = np.concatenate(flux_chunks)
                cx = np.concatenate(cx_chunks)
                cy = np.concatenate(cy_chunks)
                t = build_time_axis(meta, 'FGS1', len(flux))
                # Baseline detrend with centroids + poly ramps
                X = build_design_matrix(flux, centroids=(cx, cy), order=2)
                beta, yhat, s2 = generalized_least_squares(flux, X)
                flux_detr = flux - yhat + np.nanmedian(flux)
                # Optional PLD detrend using early frames cube
                flux_pld = None
                if cubes:
                    cube_small = np.concatenate(cubes, axis=0)
                    n_use = min(len(flux), cube_small.shape[0])
                    X_pld = build_design_matrix(flux[:n_use], centroids=(cx[:n_use], cy[:n_use]), order=2, pld_pixels=cube_small[:n_use], pld_max_pixels=50)
                    _, yhat_pld, _ = generalized_least_squares(flux[:n_use], X_pld)
                    flux_pld = flux[:n_use] - yhat_pld + np.nanmedian(flux[:n_use])
                # Plot centroids vs time
                plt.figure(figsize=(12,3))
                plt.plot(t, cx, label='centroid_x', lw=0.6)
                plt.plot(t, cy, label='centroid_y', lw=0.6)
                plt.legend()
                plt.title(f'FGS1 centroids (planet {pid})')
                plt.xlabel('time'); plt.ylabel('pixels')
                plt.tight_layout()
                out_path1 = fig_dir / f'{pid}_FGS1_centroids.png'
                plt.savefig(out_path1, dpi=150)
                # Plot raw vs detrended flux
                plt.figure(figsize=(12,3))
                nf = flux/np.nanmedian(flux)
                nd = flux_detr/np.nanmedian(flux_detr)
                plt.plot(t, nf, label='raw', alpha=0.7, lw=0.6)
                plt.plot(t, nd, label='detrended', alpha=0.8, lw=0.8)
                if flux_pld is not None:
                    tt = t[:len(flux_pld)]
                    npd = flux_pld/np.nanmedian(flux_pld)
                    plt.plot(tt, npd, label='detrended+PLD', alpha=0.9, lw=0.8)
                plt.legend()
                plt.title(f'FGS1 raw vs detrended (planet {pid})')
                plt.xlabel('time'); plt.ylabel('normalized flux')
                plt.tight_layout()
                out_path2 = fig_dir / f'{pid}_FGS1_raw_vs_detrended.png'
                plt.savefig(out_path2, dpi=150)
                print('Saved:', out_path1)
                print('Saved:', out_path2)
except Exception as e:
    print('FGS centroids/detrend visualization skipped:', e)


# Visualization: AIRS spectrogram (time vs wavelength) and save
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        p = signal_path(split, pid, 'AIRS-CH0', 0)
        if p.exists():
            gain, offset = load_adc_info(meta)
            masters = load_build_calibration(split, pid, 'AIRS-CH0', gain, offset)
            frames = []
            max_frames = 1500
            for batch in iter_parquet_rows(p, batch_rows=1500):
                cal = calibrate_batch_frames(batch, masters, gain, offset)
                f3 = airs_unflatten(cal)
                f3 = crop_airs_x(f3, 39, 321)
                frames.append(f3)
                if sum(x.shape[0] for x in frames) >= max_frames:
                    break
            if frames:
                cube = np.concatenate(frames, axis=0)  # (T, 32, W)
                T, H, W = cube.shape
                spec = cube.sum(axis=1)  # (T, W)
                t = build_time_axis(meta, 'AIRS-CH0', T)
                # Naive in/out masks (middle 20% in-transit)
                i0, i1 = int(0.4*T), int(0.6*T)
                in_m = np.zeros(T, dtype=bool)
                in_m[i0:i1] = True
                out_m = ~in_m
                out_med = np.nanmedian(spec[out_m], axis=0)
                out_med[out_med==0] = np.nan
                norm = spec / out_med[np.newaxis, :]
                spect = norm - 1.0
                # Wavelength grid (cropped)
                waves = None
                try:
                    w = get_wavelength_grid(meta, 'AIRS-CH0')
                    if w is not None:
                        w = w[39:321]
                        if len(w) == W:
                            waves = w
                except Exception:
                    pass
                # Plot and save
                fig, ax = plt.subplots(1,1, figsize=(12,4))
                extent = [0, W-1, t[0], t[-1]] if waves is None else [waves[0], waves[-1], t[0], t[-1]]
                im = ax.imshow(spect, aspect='auto', origin='lower', extent=extent, cmap='coolwarm', vmin=-0.02, vmax=0.02)
                ax.set_xlabel('wavelength' if waves is not None else 'pixel column (cropped)')
                ax.set_ylabel('time')
                ax.set_title(f'AIRS spectrogram (planet {pid})')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='(flux/out) - 1')
                fig_dir = WORK_DIR / 'figures'
                fig_dir.mkdir(parents=True, exist_ok=True)
                out_path = fig_dir / f'{pid}_AIRS-CH0_spectrogram.png'
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                print('Saved:', out_path)
except Exception as e:
    print('AIRS spectrogram visualization skipped:', e)


# Visualization: FGS diagnostics — centroid scatter and PSD; save
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        p = signal_path(split, pid, 'FGS1', 0)
        if p.exists():
            gain, offset = load_adc_info(meta)
            masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
            flux_chunks, cx_chunks, cy_chunks = [], [], []
            for batch in iter_parquet_rows(p, batch_rows=4000):
                cal = calibrate_batch_frames(batch, masters, gain, offset)
                f3 = fgs_unflatten(cal)
                flux_chunks.append(aperture_photometry(f3, r=5.0))
                cx, cy = compute_centroids(f3)
                cx_chunks.append(cx)
                cy_chunks.append(cy)
                if sum(len(a) for a in flux_chunks) >= 8000:
                    break
            if flux_chunks:
                flux = np.concatenate(flux_chunks)
                cx = np.concatenate(cx_chunks)
                cy = np.concatenate(cy_chunks)
                t = build_time_axis(meta, 'FGS1', len(flux))
                X = build_design_matrix(flux, centroids=(cx, cy), order=2)
                _, yhat, _ = generalized_least_squares(flux, X)
                detr = flux - yhat + np.nanmedian(flux)
                # Centroid scatter
                plt.figure(figsize=(4,4))
                plt.scatter(cx, cy, s=1, alpha=0.5)
                plt.xlabel('centroid_x')
                plt.ylabel('centroid_y')
                plt.title(f'FGS centroid scatter (planet {pid})')
                fig_dir = WORK_DIR / 'figures'
                fig_dir.mkdir(parents=True, exist_ok=True)
                out1 = fig_dir / f'{pid}_FGS1_centroid_scatter.png'
                plt.tight_layout(); plt.savefig(out1, dpi=150)
                # PSD of raw vs detrended
                try:
                    from matplotlib import mlab
                    fs = 1.0/np.median(np.diff(t)) if len(t)>1 else 1.0
                    Pxx_raw, f_raw = mlab.psd((flux/np.nanmedian(flux))-1, NFFT=256, Fs=fs)
                    Pxx_det, f_det = mlab.psd((detr/np.nanmedian(detr))-1, NFFT=256, Fs=fs)
                    plt.figure(figsize=(6,3))
                    plt.semilogy(f_raw, Pxx_raw, label='raw')
                    plt.semilogy(f_det, Pxx_det, label='detrended')
                    plt.xlabel('frequency (1/time)'); plt.ylabel('PSD')
                    plt.title(f'FGS PSD (planet {pid})')
                    plt.legend(); plt.tight_layout()
                    out2 = fig_dir / f'{pid}_FGS1_psd.png'
                    plt.savefig(out2, dpi=150)
                    print('Saved:', out1)
                    print('Saved:', out2)
                except Exception:
                    pass
except Exception as e:
    print('FGS diagnostics visualization skipped:', e)


# Visualization: AIRS single-channel light curve with naive detrending; save
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        p = signal_path(split, pid, 'AIRS-CH0', 0)
        if p.exists():
            gain, offset = load_adc_info(meta)
            masters = load_build_calibration(split, pid, 'AIRS-CH0', gain, offset)
            frames = []
            for batch in iter_parquet_rows(p, batch_rows=1500):
                cal = calibrate_batch_frames(batch, masters, gain, offset)
                f3 = airs_unflatten(cal)
                f3 = crop_airs_x(f3, 39, 321)
                frames.append(f3)
                if sum(x.shape[0] for x in frames) >= 2000:
                    break
            if frames:
                cube = np.concatenate(frames, axis=0)  # (T, 32, W)
                spec = cube.sum(axis=1)  # (T, W)
                T, W = spec.shape
                j = W//2  # mid channel
                y = spec[:, j]
                t = build_time_axis(meta, 'AIRS-CH0', T)
                # Detrend with polynomial time only
                X = build_design_matrix(y, order=2)
                _, yhat, _ = generalized_least_squares(y, X)
                yd = y - yhat + np.nanmedian(y)
                plt.figure(figsize=(12,3))
                plt.plot(t, y/np.nanmedian(y), label='raw', lw=0.6)
                plt.plot(t, yd/np.nanmedian(yd), label='detrended', lw=0.8)
                plt.legend()
                plt.xlabel('time'); plt.ylabel('normalized flux')
                plt.title(f'AIRS single-channel LC (planet {pid}, col {j})')
                plt.tight_layout()
                fig_dir = WORK_DIR / 'figures'
                fig_dir.mkdir(parents=True, exist_ok=True)
                out = fig_dir / f'{pid}_AIRS-CH0_single_channel_lc.png'
                plt.savefig(out, dpi=150)
                print('Saved:', out)
except Exception as e:
    print('AIRS single-channel LC visualization skipped:', e)


# 17c) Path helpers (bootstrap) if running cells out of order
try:
    _ = signal_path  # type: ignore
except NameError:
    try:
        SIG_EXT_ = SIG_EXT  # use existing if defined
    except NameError:
        SIG_EXT_ = '.parquet'
    def planet_dir(split: str, planet_id: str) -> Path:
        return INPUT_DIR / split / str(planet_id)
    def signal_path(split: str, planet_id: str, instrument: str, visit_idx: int = 0) -> Path:
        base = planet_dir(split, planet_id)
        # Try a few common layouts; return first existing, else a sensible default
        candidates = [
            base / instrument / f"visit_{visit_idx:02d}{SIG_EXT_}",
            base / instrument / f"{visit_idx:02d}{SIG_EXT_}",
            base / instrument / f"visit_{visit_idx}{SIG_EXT_}",
            base / instrument / f"{visit_idx}{SIG_EXT_}",
            base / f"{instrument}_visit_{visit_idx:02d}{SIG_EXT_}",
            base / f"{instrument}_{visit_idx:02d}{SIG_EXT_}",
        ]
        for p in candidates:
            try:
                if p.exists():
                    return p
            except Exception:
                pass
        # Fallback: first parquet under instrument folder if any
        try:
            inst_dir = base / instrument
            if inst_dir.exists():
                for p in inst_dir.glob(f"*{SIG_EXT_}"):
                    return p
        except Exception:
            pass
        return candidates[0]
    print('Defined minimal signal_path() and planet_dir() bootstrap.')


# Visualization: Dataset overview — planets, instruments, and file counts
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    base = INPUT_DIR / split
    if not base.exists():
        raise FileNotFoundError(f"split folder not found: {base}")
    planets = [p for p in base.iterdir() if p.is_dir()]
    if not planets:
        raise RuntimeError('No planet directories found')
    # Collect counts per instrument for first N planets
    N = 8
    instruments = ['FGS1','FGS2','AIRS-CH0','AIRS-CH1']
    rows = []
    for pid_path in planets[:N]:
        pid = pid_path.name
        for inst in instruments:
            inst_dir = pid_path / inst
            cnt = 0
            if inst_dir.exists():
                try:
                    cnt = sum(1 for _ in inst_dir.glob('*.parquet'))
                except Exception:
                    cnt = 0
            rows.append({'planet_id': pid, 'instrument': inst, 'files': cnt})
    import pandas as _pd
    dfc = _pd.DataFrame(rows)
    if dfc['files'].sum() == 0:
        raise RuntimeError('No parquet files found for the first few planets')
    # Pivot to planet x instrument matrix
    piv = dfc.pivot(index='planet_id', columns='instrument', values='files').fillna(0)
    # Plot
    ax = piv.plot(kind='bar', figsize=(12,4))
    ax.set_title(f'Dataset overview: file counts per instrument (first {len(piv)} planets in {split})')
    ax.set_xlabel('planet_id'); ax.set_ylabel('# files')
    plt.tight_layout()
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'dataset_overview_{split}.png'
    plt.savefig(out, dpi=150)
    print('Saved:', out)
except Exception as e:
    print('Dataset overview visualization skipped:', e)


# Visualization: AIRS trace center drift and width over time
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        pth = signal_path(split, pid, 'AIRS-CH0', 0)
        if not pth.exists():
            raise FileNotFoundError(pth)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'AIRS-CH0', gain, offset)
        cys, widths = [], []
        Tcap = 2000
        seen = 0
        for batch in iter_parquet_rows(pth, batch_rows=512):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = airs_unflatten(cal)
            f3 = crop_airs_x(f3, 39, 321)
            med = np.nanmedian(f3, axis=2)  # collapse x -> shape (n, 32)
            # center and width per frame using simple weighted stats
            y = np.arange(32)
            w = med
            denom = np.nansum(w, axis=1) + 1e-12
            cy = np.nansum(w * y[None, :], axis=1) / denom
            var = np.nansum(w * (y[None, :] - cy[:, None])**2, axis=1) / denom
            cys.append(cy)
            widths.append(np.sqrt(np.maximum(var, 0)))
            seen += len(cy)
            if seen >= Tcap:
                break
        if cys:
            cy_all = np.concatenate(cys)
            wd_all = np.concatenate(widths)
            t = build_time_axis(meta, 'AIRS-CH0', len(cy_all))
            fig, ax = plt.subplots(2,1, figsize=(12,5), sharex=True)
            ax[0].plot(t, cy_all, lw=0.6)
            ax[0].set_ylabel('center_y [px]')
            ax[0].set_title(f'AIRS trace center drift (planet {pid})')
            ax[1].plot(t, wd_all, lw=0.6)
            ax[1].set_ylabel('width [px]'); ax[1].set_xlabel('time')
            plt.tight_layout()
            fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
            out = fig_dir / f'{pid}_AIRS-CH0_trace_center_width.png'
            plt.savefig(out, dpi=150)
            print('Saved:', out)
except Exception as e:
    print('AIRS center/width visualization skipped:', e)


# Visualization: FGS aperture growth curve (flux vs aperture radius)
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        pth = signal_path(split, pid, 'FGS1', 0)
        if not pth.exists():
            raise FileNotFoundError(pth)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
        # take one batch of frames
        batch0 = next(iter_parquet_rows(pth, batch_rows=512))
        cal = calibrate_batch_frames(batch0, masters, gain, offset)
        f3 = fgs_unflatten(cal)
        # median image to define center
        med = np.nanmedian(f3, axis=0)
        yy, xx = np.indices(med.shape)
        # rough centroid on median
        w = med - np.nanmin(med)
        w = np.clip(w, 0, None)
        denom = np.nansum(w) + 1e-12
        cx = np.nansum(w * xx) / denom
        cy = np.nansum(w * yy) / denom
        rs = np.linspace(2, 12, 11)
        fluxes = []
        for r in rs:
            rr = ((xx - cx)**2 + (yy - cy)**2)**0.5
            mask = rr <= r
            # integrate over mask for each frame then median across time
            vals = np.nansum(f3[:, mask], axis=1)
            fluxes.append(np.nanmedian(vals))
        plt.figure(figsize=(6,4))
        plt.plot(rs, fluxes, marker='o')
        plt.xlabel('aperture radius [px]'); plt.ylabel('median aperture flux')
        plt.title(f'FGS1 aperture growth curve (planet {pid})')
        plt.tight_layout()
        fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / f'{pid}_FGS1_aperture_growth.png'
        plt.savefig(out, dpi=150)
        print('Saved:', out)
except Exception as e:
    print('FGS aperture growth visualization skipped:', e)


# Visualization: FGS design-matrix correlations and residual histogram
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    planets = [p for p in (INPUT_DIR / split).iterdir() if p.is_dir()]
    if planets:
        pid = planets[0].name
        pth = signal_path(split, pid, 'FGS1', 0)
        if not pth.exists():
            raise FileNotFoundError(pth)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
        flux_chunks, cx_chunks, cy_chunks, cubes = [], [], [], []
        cap_frames = 3000
        for batch in iter_parquet_rows(pth, batch_rows=1500):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = fgs_unflatten(cal)
            flux_chunks.append(aperture_photometry(f3, r=5.0))
            cx, cy = compute_centroids(f3)
            cx_chunks.append(cx); cy_chunks.append(cy)
            if sum(len(a) for a in flux_chunks) < cap_frames:
                cubes.append(f3)
            if sum(len(a) for a in flux_chunks) >= cap_frames:
                break
        if flux_chunks:
            import numpy as _np
            flux = _np.concatenate(flux_chunks)
            cx = _np.concatenate(cx_chunks); cy = _np.concatenate(cy_chunks)
            t = build_time_axis(meta, 'FGS1', len(flux))
            # Build design matrix with centroids and PLD from a small cube
            P = None
            if cubes:
                cube_small = _np.concatenate(cubes, axis=0)
                n_use = min(len(flux), cube_small.shape[0], 2500)
                P = cube_small[:n_use]
            n_use = len(flux) if P is None else min(len(flux), P.shape[0])
            X = build_design_matrix(flux[:n_use], centroids=(cx[:n_use], cy[:n_use]), order=2, pld_pixels=P[:n_use] if P is not None else None, pld_max_pixels=40)
            # Correlation heatmap
            C = _np.corrcoef(X.T)
            plt.figure(figsize=(6,5))
            im = plt.imshow(C, vmin=-1, vmax=1, cmap='coolwarm')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f'FGS design-matrix correlation (planet {pid})')
            plt.tight_layout()
            fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
            out1 = fig_dir / f'{pid}_FGS1_design_corr.png'
            plt.savefig(out1, dpi=150)
            # Residual histogram
            beta, yhat, s2 = generalized_least_squares(flux[:n_use], X)
            resid = flux[:n_use] - yhat
            plt.figure(figsize=(6,3))
            plt.hist((resid - _np.nanmedian(resid))/_np.nanstd(resid), bins=60, alpha=0.8)
            plt.title(f'FGS residuals (z-scored), planet {pid}')
            plt.xlabel('z'); plt.ylabel('count'); plt.tight_layout()
            out2 = fig_dir / f'{pid}_FGS1_residual_hist.png'
            plt.savefig(out2, dpi=150)
            print('Saved:', out1)
            print('Saved:', out2)
except Exception as e:
    print('FGS design-matrix/residual visualization skipped:', e)


# Visualization: Global plotting style for publication-quality figures
try:
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'image.cmap': 'cividis',
    })
    if sns is not None:
        sns.set_theme(style='whitegrid', context='notebook')
    print('Plotting style configured.')
except Exception as e:
    print('Style configuration skipped:', e)


# Advanced: FGS diagnostics dashboard (raw/detrended, centroids, residuals, RMS)
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    base = INPUT_DIR / split
    planets = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    pid = planets[0].name if planets else 'SYNTH'
    have_data = bool(planets)
    if have_data:
        pth = signal_path(split, pid, 'FGS1', 0)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
        flux_chunks, cx_chunks, cy_chunks, cubes = [], [], [], []
        for batch in iter_parquet_rows(pth, batch_rows=4000):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = fgs_unflatten(cal)
            flux_chunks.append(aperture_photometry(f3, r=HP_PATH.exists() and load_hparams().get('fgs_aperture_r',5.0) or 5.0))
            cx, cy = compute_centroids(f3)
            cx_chunks.append(cx); cy_chunks.append(cy)
            cubes.append(f3)
            if sum(len(a) for a in flux_chunks) >= 12000:
                break
        flux = np.concatenate(flux_chunks) if flux_chunks else None
        cx = np.concatenate(cx_chunks) if cx_chunks else None
        cy = np.concatenate(cy_chunks) if cy_chunks else None
        cube_small = np.concatenate(cubes, axis=0) if cubes else None
    else:
        # Synthetic fallback
        n = 3000
        t = np.linspace(0, 1, n)
        cx = 16 + 0.1*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(n)
        cy = 16 + 0.1*np.cos(2*np.pi*2*t) + 0.02*np.random.randn(n)
        sys = 1 + 0.005*t + 0.002*np.sin(2*np.pi*5*t) + 0.001*cx + 0.001*cy
        transit = 1 - 0.005*(np.abs(t-0.5)<0.05)
        flux = sys * transit * (1 + 0.001*np.random.randn(n))
        cube_small = None
    if flux is None or len(flux) < 50:
        raise RuntimeError('Insufficient FGS data for dashboard')
    t = build_time_axis(meta, 'FGS1', len(flux)) if have_data else t
    # Build design matrix and detrend
    X = build_design_matrix(flux, centroids=(cx, cy), order=2, pld_pixels=cube_small[:len(flux)] if (cube_small is not None and cube_small.shape[0] >= len(flux)) else None, pld_max_pixels=60)
    beta, yhat, s2 = generalized_least_squares(flux, X)
    nf = flux/np.nanmedian(flux)
    nd = (flux - yhat + np.nanmedian(flux))/np.nanmedian(flux)
    # Time-averaging RMS (Allan-like)
    def time_avg_rms(y, max_bin=100):
        y = y - np.nanmedian(y)
        rms_x, rms_y = [], []
        for b in np.unique(np.logspace(0, np.log10(max_bin), 20).astype(int)):
            if b < 1: b = 1
            m = len(y)//b
            if m < 2: break
            yy = y[:m*b].reshape(m, b).mean(axis=1)
            rms_x.append(b)
            rms_y.append(np.nanstd(yy))
        return np.array(rms_x), np.array(rms_y)
    bx_raw, by_raw = time_avg_rms(nf-1)
    bx_det, by_det = time_avg_rms(nd-1)
    # Figure layout
    fig = plt.figure(figsize=(14,9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2,1,1], hspace=0.35, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    # Panel 1: raw vs detrended
    ax1.plot(t, nf, lw=0.5, alpha=0.8, label='raw')
    ax1.plot(t, nd, lw=0.8, alpha=0.9, label='detrended')
    ax1.set_title(f'FGS1 flux: raw vs detrended ({pid})')
    ax1.set_xlabel('time'); ax1.set_ylabel('normalized flux'); ax1.legend(loc='best')
    # Panel 2/3: centroids
    ax2.plot(t, cx, lw=0.6, color='tab:green')
    ax2.set_title('Centroid X'); ax2.set_xlabel('time'); ax2.set_ylabel('px')
    ax3.plot(t, cy, lw=0.6, color='tab:orange')
    ax3.set_title('Centroid Y'); ax3.set_xlabel('time'); ax3.set_ylabel('px')
    # Panel 4: residual histogram
    resid = (nf - nd)
    ax4.hist((resid - np.nanmedian(resid))/ (np.nanstd(resid)+1e-12), bins=60, color='tab:blue', alpha=0.8)
    ax4.set_title('Residuals (z-scored)'); ax4.set_xlabel('z'); ax4.set_ylabel('count')
    # Panel 5: time-averaging RMS
    ax5.loglog(bx_raw, by_raw, 'o-', label='raw')
    ax5.loglog(bx_det, by_det, 'o-', label='detrended')
    ax5.set_title('Time-averaging RMS'); ax5.set_xlabel('bin size'); ax5.set_ylabel('RMS')
    ax5.legend()
    plt.tight_layout()
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'{pid}_FGS1_dashboard.png'
    plt.savefig(out)
    print('Saved:', out)
except Exception as e:
    print('FGS diagnostics dashboard skipped:', e)


# Advanced: AIRS spectral dashboard (spectrogram, depth, rolling bands)
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    base = INPUT_DIR / split
    planets = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    pid = planets[0].name if planets else 'SYNTH'
    have_data = bool(planets)
    if have_data:
        pth = signal_path(split, pid, 'AIRS-CH0', 0)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'AIRS-CH0', gain, offset)
        frames = []
        cap = 2000
        for batch in iter_parquet_rows(pth, batch_rows=1000):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = airs_unflatten(cal)
            f3 = crop_airs_x(f3, 39, 321)
            frames.append(f3)
            if sum(x.shape[0] for x in frames) >= cap:
                break
        if frames:
            cube = np.concatenate(frames, axis=0)
        else:
            cube = None
    else:
        # Synthetic AIRS cube: transit-like dip in middle across modest wavelength band
        T, H, W = 800, 32, 260
        t = np.linspace(0, 1, T)
        waves = np.linspace(1.1, 1.9, W)
        band = np.exp(-0.5*((waves-1.5)/0.15)**2)
        sys = 1 + 0.01*np.sin(2*np.pi*3*t)[:,None,None]
        transit = 1 - (0.01*band[None,None,:]) * (np.abs(t-0.5)<0.06)[:,None,None]
        cube = sys*transit*(1 + 0.001*np.random.randn(T,H,W))
    if cube is None or cube.shape[0] < 50:
        raise RuntimeError('Insufficient AIRS data for dashboard')
    T, H, W = cube.shape
    spec = cube.sum(axis=1)  # (T, W)
    t = build_time_axis(meta, 'AIRS-CH0', T) if have_data else t
    # In/Out masks: middle 20% as in-transit
    i0, i1 = int(0.4*T), int(0.6*T)
    in_m = np.zeros(T, dtype=bool); in_m[i0:i1] = True
    out_m = ~in_m
    out_med = np.nanmedian(spec[out_m], axis=0)
    out_med[out_med==0] = np.nan
    norm = spec / out_med[np.newaxis, :]
    # Depth and uncertainty
    mean_in = np.nanmean(norm[in_m], axis=0)
    mean_out = np.nanmean(norm[out_m], axis=0)
    var_in = np.nanvar(norm[in_m], axis=0)/(in_m.sum() + 1e-9)
    var_out = np.nanvar(norm[out_m], axis=0)/(out_m.sum() + 1e-9)
    depth = 1.0 - (mean_in/mean_out)
    sigma = np.sqrt(np.maximum(var_in + var_out, 0))
    # Rolling bands time series: split W into 6 equal bands
    B = 6
    edges = np.linspace(0, W, B+1).astype(int)
    band_series = []
    for i in range(B):
        a,b = edges[i], edges[i+1]
        y = np.nanmean(norm[:, a:b], axis=1)
        band_series.append((a,b,y))
    # Figure layout
    fig = plt.figure(figsize=(14,9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2,1,1], hspace=0.35, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])
    # Panel 1: Spectrogram (normalized)
    im = ax1.imshow(norm-1.0, aspect='auto', origin='lower', extent=[0, W-1, t[0], t[-1]], vmin=-0.03, vmax=0.03)
    ax1.set_title(f'AIRS spectrogram (norm-1), {pid}')
    ax1.set_xlabel('pixel (cropped)'); ax1.set_ylabel('time')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='(flux/out)-1')
    # Panel 2: Depth vs wavelength
    x = np.arange(W)
    ax2.plot(x, depth, color='tab:blue', lw=1)
    ax2.fill_between(x, depth-sigma, depth+sigma, color='tab:blue', alpha=0.2, linewidth=0)
    ax2.set_title('Depth vs wavelength (arb units)'); ax2.set_xlabel('pixel (cropped)'); ax2.set_ylabel('depth')
    # Panel 3: Out-of-transit baseline (mean over time)
    ax3.plot(x, np.nanmean(norm[out_m], axis=0), color='tab:gray', lw=1)
    ax3.set_title('Out-of-transit mean spectrum'); ax3.set_xlabel('pixel (cropped)'); ax3.set_ylabel('mean flux')
    # Panel 4: Rolling band time series
    colors = plt.cm.tab10(np.linspace(0,1,B))
    for i,(a,b,y) in enumerate(band_series):
        ax4.plot(t, y/np.nanmedian(y), color=colors[i], lw=0.8, label=f'cols {a}-{b}')
    ax4.set_title('Band-averaged time series'); ax4.set_xlabel('time'); ax4.set_ylabel('normalized flux'); ax4.legend(ncol=3)
    plt.tight_layout()
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'{pid}_AIRS-CH0_dashboard.png'
    plt.savefig(out)
    print('Saved:', out)
except Exception as e:
    print('AIRS spectral dashboard skipped:', e)


# Visualization: Compact quality summary (text report)
try:
    report = []
    report.append(f"Env Kaggle={IN_KAGGLE} Input='{INPUT_DIR}' Work='{WORK_DIR}'")
    for name in ['train','wavelengths','axis_info','adc_info','train_star','test_star','sample_submission']:
        df = meta.get(name)
        shape = tuple(df.shape) if df is not None else None
        report.append(f"meta[{name}]: shape={shape}")
    # Check existence of split folders
    for split in ['train','test']:
        p = INPUT_DIR / split
        report.append(f"exists {split}: {p.exists()} path='{p}'")
    text = "\n".join(report)
    print(text)
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    with open(fig_dir / 'quality_summary.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Saved:', fig_dir / 'quality_summary.txt')
except Exception as e:
    print('Quality summary skipped:', e)


# Premium Viz: FGS light curve with shaded transit and zoom inset
try:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    base = INPUT_DIR / split
    planets = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    pid = planets[0].name if planets else 'SYNTH'
    have_data = bool(planets)
    if have_data:
        pth = signal_path(split, pid, 'FGS1', 0)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
        flux_chunks = []
        for batch in iter_parquet_rows(pth, batch_rows=6000):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = fgs_unflatten(cal)
            flux_chunks.append(aperture_photometry(f3, r=load_hparams().get('fgs_aperture_r',5.0)))
            if sum(len(a) for a in flux_chunks) >= 15000:
                break
        flux = np.concatenate(flux_chunks) if flux_chunks else None
    else:
        n = 6000
        t = np.linspace(0, 1, n)
        transit = 1 - 0.004*(np.abs(t-0.5)<0.06)
        sys = 1 + 0.003*np.sin(2*np.pi*2.5*t) + 0.001*np.random.randn(n)
        flux = transit*sys
    if flux is None or len(flux) < 50:
        raise RuntimeError('Insufficient FGS data')
    t = build_time_axis(meta, 'FGS1', len(flux)) if have_data else t
    # Detrend baseline for cleaner viewing
    X = build_design_matrix(flux, order=2)
    _, yhat, _ = generalized_least_squares(flux, X)
    nf = flux/np.nanmedian(flux)
    nd = (flux - yhat + np.nanmedian(flux))/np.nanmedian(flux)
    T = len(flux)
    t0, t1 = t[int(0.4*T)], t[int(0.6*T)]
    fig, ax = plt.subplots(figsize=(12,4), constrained_layout=True)
    ax.plot(t, nf, color='0.7', lw=0.4, label='raw')
    ax.plot(t, nd, color='tab:blue', lw=0.8, label='detrended')
    ax.axvspan(t0, t1, color='tab:blue', alpha=0.08, label='in-transit (naive)')
    ax.set_title(f'FGS1 light curve with shaded transit and zoom ({pid})')
    ax.set_xlabel('time'); ax.set_ylabel('normalized flux')
    ax.legend(loc='upper right', frameon=False)
    # Inset around the minimum in detrended flux (transit-like region)
    jmin = np.nanargmin(nd)
    w = max(20, int(0.02*T))
    x0 = max(t[0], t[jmin]-0.5*(t[w]-t[0]))
    x1 = min(t[-1], t[jmin]+0.5*(t[w]-t[0]))
    axins = inset_axes(ax, width="35%", height="60%", loc='lower left', bbox_to_anchor=(0.05,0.05,0.9,0.9), bbox_transform=ax.transAxes, borderpad=0)
    axins.plot(t, nd, color='tab:blue', lw=0.8)
    axins.set_xlim(x0, x1)
    axins.set_ylim(np.nanmin(nd[jmin-w:jmin+w])-0.001, np.nanmax(nd[jmin-w:jmin+w])+0.001)
    axins.set_xticks([]); axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'{pid}_FGS1_lc_zoom.png'
    plt.savefig(out)
    print('Saved:', out)
except Exception as e:
    print('FGS premium LC visualization skipped:', e)


# Premium Viz: AIRS spectrum with smoothing and zoom inset
try:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    base = INPUT_DIR / split
    planets = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    pid = planets[0].name if planets else 'SYNTH'
    have_data = bool(planets)
    if have_data:
        pth = signal_path(split, pid, 'AIRS-CH0', 0)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'AIRS-CH0', gain, offset)
        frames = []
        for batch in iter_parquet_rows(pth, batch_rows=1200):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = airs_unflatten(cal)
            f3 = crop_airs_x(f3, 39, 321)
            frames.append(f3)
            if sum(x.shape[0] for x in frames) >= 2000:
                break
        cube = np.concatenate(frames, axis=0) if frames else None
    else:
        # Synthetic cube similar to earlier
        T, H, W = 1000, 32, 260
        t = np.linspace(0, 1, T)
        waves = np.linspace(1.1, 1.9, W)
        band = np.exp(-0.5*((waves-1.5)/0.12)**2)
        sys = 1 + 0.01*np.sin(2*np.pi*2*t)[:,None,None]
        transit = 1 - (0.012*band[None,None,:]) * (np.abs(t-0.5)<0.08)[:,None,None]
        cube = sys*transit*(1 + 0.001*np.random.randn(T,H,W))
    if cube is None or cube.shape[0] < 50:
        raise RuntimeError('Insufficient AIRS data')
    T = cube.shape[0]
    spec = cube.sum(axis=1)  # (T, W)
    i0, i1 = int(0.4*T), int(0.6*T)
    in_m = np.zeros(T, dtype=bool); in_m[i0:i1] = True
    out_m = ~in_m
    out_med = np.nanmedian(spec[out_m], axis=0)
    out_med[out_med==0] = np.nan
    norm = spec / out_med[np.newaxis, :]
    mean_in = np.nanmean(norm[in_m], axis=0)
    mean_out = np.nanmean(norm[out_m], axis=0)
    depth = 1.0 - (mean_in/mean_out)
    x = np.arange(depth.shape[0])
    # Savitzky-Golay smoothing if scipy available
    d_smooth = depth
    try:
        from scipy.signal import savgol_filter
        win = max(7, (len(depth)//25)//2*2+1)
        d_smooth = savgol_filter(depth, window_length=win, polyorder=2, mode='interp')
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(12,4), constrained_layout=True)
    ax.plot(x, depth, color='0.7', lw=0.6, label='raw depth')
    ax.plot(x, d_smooth, color='tab:purple', lw=1.2, label='smoothed')
    ax.set_title(f'AIRS quick spectrum with smoothing ({pid})')
    ax.set_xlabel('pixel (cropped)'); ax.set_ylabel('depth (arb)')
    ax.legend(loc='best', frameon=False)
    # Inset on lowest (deepest) region
    j = int(np.nanargmax(d_smooth))
    w = max(20, len(d_smooth)//10)
    a = max(0, j-w//2); b = min(len(d_smooth), j+w//2)
    axins = inset_axes(ax, width="35%", height="60%", loc='upper right')
    axins.plot(x, depth, color='0.8', lw=0.6)
    axins.plot(x, d_smooth, color='tab:purple', lw=1.0)
    axins.set_xlim(a, b)
    ymin = np.nanmin(d_smooth[a:b]); ymax = np.nanmax(d_smooth[a:b])
    pad = 0.05*(ymax - ymin + 1e-9)
    axins.set_ylim(ymin - pad, ymax + pad)
    axins.set_xticks([]); axins.set_yticks([])
    mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5')
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'{pid}_AIRS-CH0_spectrum_zoom.png'
    plt.savefig(out)
    print('Saved:', out)
except Exception as e:
    print('AIRS premium spectrum visualization skipped:', e)


# Premium Viz: FGS centroid hexbin with marginal histograms
try:
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    base = INPUT_DIR / split
    planets = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    pid = planets[0].name if planets else 'SYNTH'
    have_data = bool(planets)
    if have_data:
        pth = signal_path(split, pid, 'FGS1', 0)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
        cx_chunks, cy_chunks = [], []
        for batch in iter_parquet_rows(pth, batch_rows=5000):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = fgs_unflatten(cal)
            cx, cy = compute_centroids(f3)
            cx_chunks.append(cx); cy_chunks.append(cy)
            if sum(len(a) for a in cx_chunks) >= 15000:
                break
        cx = np.concatenate(cx_chunks) if cx_chunks else None
        cy = np.concatenate(cy_chunks) if cy_chunks else None
    else:
        n = 15000
        cx = 16 + 0.2*np.sin(np.linspace(0, 25, n)) + 0.1*np.random.randn(n)
        cy = 16 + 0.2*np.cos(np.linspace(0, 25, n)) + 0.1*np.random.randn(n)
    if cx is None or len(cx) < 100:
        raise RuntimeError('Insufficient centroid data')
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(7,6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,4], wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1,0])
    ax_x = fig.add_subplot(gs[0,0], sharex=ax_main)
    ax_y = fig.add_subplot(gs[1,1], sharey=ax_main)
    hb = ax_main.hexbin(cx, cy, gridsize=60, cmap='viridis', mincnt=1)
    ax_main.set_xlabel('centroid_x'); ax_main.set_ylabel('centroid_y')
    cb = fig.colorbar(hb, ax=ax_main, fraction=0.046, pad=0.04)
    cb.set_label('counts')
    ax_x.hist(cx, bins=60, color='tab:green', alpha=0.8)
    ax_y.hist(cy, bins=60, orientation='horizontal', color='tab:orange', alpha=0.8)
    plt.setp(ax_x.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_yticklabels(), visible=False)
    ax_x.tick_params(axis='x', which='both', length=0)
    ax_y.tick_params(axis='y', which='both', length=0)
    ax_main.set_title(f'FGS centroid density with marginals ({pid})')
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'{pid}_FGS1_centroid_hexbin.png'
    plt.savefig(out)
    print('Saved:', out)
except Exception as e:
    print('Centroid hexbin visualization skipped:', e)


# Premium Viz: Correlation heatmap (clustered) + target-correlation bars
try:
    # 1) Load a small FGS dataset (or synth fallback)
    split = 'test' if (INPUT_DIR / 'test').exists() else 'train'
    base = INPUT_DIR / split
    planets = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    pid = planets[0].name if planets else 'SYNTH'
    have_data = bool(planets)

    if have_data:
        pth = signal_path(split, pid, 'FGS1', 0)
        gain, offset = load_adc_info(meta)
        masters = load_build_calibration(split, pid, 'FGS1', gain, offset)
        flux_chunks, cx_chunks, cy_chunks = [], [], []
        for batch in iter_parquet_rows(pth, batch_rows=4000):
            cal = calibrate_batch_frames(batch, masters, gain, offset)
            f3 = fgs_unflatten(cal)
            flux_chunks.append(aperture_photometry(f3, r=5.0))
            cx, cy = compute_centroids(f3)
            cx_chunks.append(cx); cy_chunks.append(cy)
            if sum(len(a) for a in flux_chunks) >= 8000:
                break
        flux = np.concatenate(flux_chunks) if flux_chunks else None
        cx = np.concatenate(cx_chunks) if cx_chunks else None
        cy = np.concatenate(cy_chunks) if cy_chunks else None
    else:
        # Synthetic sequence with centroid-driven systematics + shallow transit
        n = 3000
        t = np.linspace(0, 1, n)
        cx = 16 + 0.1*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(n)
        cy = 16 + 0.1*np.cos(2*np.pi*2*t) + 0.02*np.random.randn(n)
        sys = 1 + 0.005*t + 0.002*np.sin(2*np.pi*5*t) + 0.001*cx + 0.001*cy
        transit = 1 - 0.004*(np.abs(t-0.5)<0.06)
        flux = sys * transit * (1 + 0.001*np.random.randn(n))

    if flux is None or len(flux) < 50:
        raise RuntimeError('Insufficient data for correlation visualization')

    # 2) Build labeled design matrix (no PLD to keep matrix compact)
    order = load_hparams().get('poly_order', 2) if 'load_hparams' in globals() else 2
    X = build_design_matrix(flux, centroids=(cx, cy), order=order)
    n_cols = X.shape[1]
    # Labels: [1, t, t^2, ..., cx, cy, cx^2, cy^2, cx*cy]
    time_labels = ['t'] + [f't^{k}' for k in range(2, order+1)] if order >= 1 else []
    labels = ['1'] + time_labels + ['cx', 'cy', 'cx^2', 'cy^2', 'cx*cy']
    labels = labels[:n_cols]  # guard if order/layout differs

    # 3) Correlation matrix with robust handling of NaNs/const cols
    def safe_corrcoef(M):
        M = np.asarray(M, dtype=float)
        M = M - np.nanmean(M, axis=0, keepdims=True)
        std = np.nanstd(M, axis=0, ddof=0)
        std[std == 0] = 1.0
        Mz = M / std
        C = np.nan_to_num(np.corrcoef(Mz.T), nan=0.0, posinf=0.0, neginf=0.0)
        C = np.clip(C, -1, 1)
        return C
    C = safe_corrcoef(X)

    # 4) Column ordering: cluster by 1-|corr| if scipy available, else by similarity
    order_idx = np.arange(n_cols)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        D = 1 - np.abs(C)
        # Ensure proper condensed form input
        D = np.clip(D, 0, 2)
        Z = linkage(squareform(D, checks=False), method='average')
        order_idx = leaves_list(Z)
    except Exception:
        # Fallback: sort by total absolute correlation (group similar features)
        order_idx = np.argsort(-np.sum(np.abs(C), axis=0))

    C_ord = C[np.ix_(order_idx, order_idx)]
    labels_ord = [labels[i] if i < len(labels) else f'x{i}' for i in order_idx]

    # 5) Target correlation bars (|corr(y, regressor)|) — exclude intercept if present
    def corr_with_target(y, X):
        y = np.asarray(y, float)
        y = y - np.nanmean(y)
        ys = np.nanstd(y)
        ys = ys if ys != 0 else 1.0
        y /= ys
        out = []
        for j in range(X.shape[1]):
            x = X[:, j].astype(float)
            x = x - np.nanmean(x)
            xs = np.nanstd(x)
            xs = xs if xs != 0 else 1.0
            x /= xs
            out.append(float(np.nan_to_num(np.mean(x*y), nan=0.0)))
        return np.array(out)
    r = np.abs(corr_with_target(flux, X))
    # Drop intercept for bar chart if present at col 0
    if labels and labels[0] == '1':
        r_no_bias = r[1:]
        labels_no_bias = labels[1:]
    else:
        r_no_bias = r
        labels_no_bias = labels
    # Order bars by magnitude (top-K)
    K = min(10, len(labels_no_bias))
    bar_idx = np.argsort(-r_no_bias)[:K]

    # 6) Plot: heatmap (masked upper triangle) + side bar chart
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    axH = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # Heatmap (prefer seaborn if available)
    mask = np.triu(np.ones_like(C_ord, dtype=bool), k=1)
    try:
        import seaborn as sns
        sns.heatmap(
            C_ord,
            mask=mask,
            ax=axH,
            vmin=-1, vmax=1, center=0,
            cmap='coolwarm',
            square=True,
            cbar_kws={'label': 'correlation'},
            linewidths=0.5, linecolor='white'
        )
    except Exception:
        # Fallback to matplotlib
        im = axH.imshow(np.where(mask, np.nan, C_ord), vmin=-1, vmax=1, cmap='coolwarm')
        cb = fig.colorbar(im, ax=axH)
        cb.set_label('correlation')

    axH.set_xticks(np.arange(len(labels_ord)))
    axH.set_yticks(np.arange(len(labels_ord)))
    axH.set_xticklabels(labels_ord, rotation=45, ha='right')
    axH.set_yticklabels(labels_ord)
    axH.set_title(f'FGS1 design correlation (clustered) — {pid}')

    # Annotate numbers only for small matrices to avoid clutter
    if C_ord.shape[0] <= 18:
        for i in range(C_ord.shape[0]):
            for j in range(i+1):  # lower triangle incl. diagonal
                val = C_ord[i, j]
                axH.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7,
                         color=('white' if abs(val) > 0.75 else 'black'))

    # Bar chart: |corr(y, regressor)|
    axB.barh(range(K), r_no_bias[bar_idx][::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, K)))
    axB.set_yticks(range(K))
    axB.set_yticklabels([labels_no_bias[i] for i in bar_idx][::-1])
    axB.invert_yaxis()
    axB.set_xlim(0, 1)
    axB.set_xlabel('|corr with flux|')
    axB.set_title('Top drivers')

    # Figure title and save
    fig.suptitle('Design diagnostics: correlation structure and key drivers', fontsize=13)
    fig_dir = WORK_DIR / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'{pid}_FGS1_corr_annot.png'
    plt.savefig(out, dpi=200)
    # Optional SVG for vector clarity
    try:
        plt.savefig(out.with_suffix('.svg'))
    except Exception:
        pass
    print('Saved:', out)
except Exception as e:
    print('Annotated correlation heatmap skipped:', e)

