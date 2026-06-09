import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import energy_distance
from scipy import signal, stats
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

import sys


# =========================================================
# Ariel 2025 — FGS1 특징 (베이직셋) + LightGBM (CPU only)
#  + Post-Process Patch (튜닝 가능):
#    - 파장방향 Savitzky–Golay 스무딩 + α 블렌딩
#    - OOF 잔차 기반 σ(타깃 불확실도) 재보정(+ clip, scale)
#  + Template 정렬(Nearest Neighbors on PCA of AIRS) + PCA 서브스페이스 회귀 블렌드
#  + (옵션) 변동성 기반 클러스터링 + GMM 증강
#  + Wide 튜너로 후보정 하이퍼파라미터 탐색 + OOF GLL 스코어링
#  + 시각화/캐시/서브미션 유틸
# =========================================================

import os, re, glob, time, warnings, random, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis, norm
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors  # ★ 누락 보완
warnings.filterwarnings("ignore")
try:
    from scipy.fft import dct as _dct
except Exception:
    from scipy.fftpack import dct as _dct


# --- LightGBM 무음 로거 ---
class _SilentLogger:
    def info(self, msg: str):   pass
    def warning(self, msg: str): pass
try:
    lgb.register_logger(_SilentLogger())
except Exception as e:
    print("[LGBM] register_logger failed:", e)

# ----------------------- Config -----------------------
BASE_PATH = "/kaggle/input/ariel-data-challenge-2025"
WORK_DIR  = "/kaggle/working"
PLOT_DIR  = f"{WORK_DIR}/oof_plots"
os.makedirs(WORK_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)

USE_CACHED_XY = False
CACHED_TRAIN_XY = "/kaggle/input/fg1-train-xy/fgs1_train_xy.csv"

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ★ Train만 fraction 조절 (test는 항상 100%)
TRAIN_FRAC = 1
MAX_SIGNALS_PER_PLANET = 1

# ★ 밝은 픽셀 파생 (원하면 False로 끄기)
USE_BRIGHTPIXEL = True
BRIGHTPIX_RELAX = 0.75

# ★ Post-process 초기 하이퍼파라미터 (튜닝 전 기본값)
POST_SMOOTH_WINDOW = 13     # Savitzky–Golay window (odd)
POST_SMOOTH_POLY   = 2
POST_BLEND_ALPHA   = 0.65   # 최종 = alpha*raw + (1-alpha)*smooth
SIGMA_CLIP_FGS     = (0.85, 1.30)
SIGMA_CLIP_AIRS    = (0.92, 1.22)
SIGMA_FINAL_SCALE  = 1.04

# LightGBM (CPU only)
N_SPLITS = 5
EARLY_STOP = 200
BASE_LGB_PARAMS = dict(
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=3000,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1e-2,
    reg_lambda=1e-1,
    random_state=SEED,
    n_jobs=-1,
    num_threads=-1
)

# ----------------------- Load -----------------------
train_df     = pd.read_csv(f"{BASE_PATH}/train.csv")
train_star   = pd.read_csv(f"{BASE_PATH}/train_star_info.csv")
test_star    = pd.read_csv(f"{BASE_PATH}/test_star_info.csv")
sample_sub   = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

WL_COLS = sorted([c for c in train_df.columns if re.fullmatch(r"(wl|wavelength)_\d+", c)],
                 key=lambda x: int(re.findall(r"\d+", x)[0]))
assert len(WL_COLS) == 283

wldf = pd.read_csv(f"{BASE_PATH}/wavelengths.csv")
wavelengths = wldf[WL_COLS].iloc[0].to_numpy(float)

arr = train_df[WL_COLS].to_numpy(dtype=float)
NAIVE_MEAN  = float(np.nanmean(arr))
NAIVE_SIGMA = float(np.nanstd(arr) + 1e-12)

# ----------------------- Utils -----------------------
def add_interactions_to_X(X, small_feats, large_feats, eps=1e-9):
    new_cols = []
    for a, b in itertools.permutations(large_feats, 2):
        if a in X.columns and b in X.columns:
            c1 = f"{a}__minus__{b}"; X[c1] = X[a] - X[b]; new_cols.append(c1)
            c2 = f"{a}__div__{b}"  ; X[c2] = X[a] / (X[b].abs() + eps); new_cols.append(c2)
    for a, b in itertools.permutations(small_feats, 2):
        if a in X.columns and b in X.columns:
            c3 = f"{a}__mul__{b}"  ; X[c3] = X[a] * X[b]; new_cols.append(c3)
            c4 = f"{a}__div__{b}"  ; X[c4] = X[a] / (X[b].abs() + eps); new_cols.append(c4)
    for a in small_feats:
        for b in large_feats:
            if a in X.columns and b in X.columns:
                c5 = f"{a}__mul__{b}"; X[c5] = X[a] * X[b]; new_cols.append(c5)
    return new_cols

def get_mean_feature_importance(trained, kind="gain"):
    feats = trained["feature_columns"]
    imp = np.zeros(len(feats), dtype=np.float64); n = 0
    for fold_models in trained["models"]:
        for m in fold_models:
            try:
                arr = m.booster_.feature_importance(importance_type=kind)
            except Exception:
                arr = m.feature_importances_
            imp += np.asarray(arr, dtype=np.float64); n += 1
    imp = imp / max(n, 1)
    fi = pd.DataFrame({"feature": feats, "importance": imp}).sort_values(
        "importance", ascending=False).reset_index(drop=True)
    return fi

def build_derived_from_topk(X, fi_df, topk=20, mean_threshold=1.0, eps=1e-9):
    top = [f for f in fi_df["feature"].tolist() if f in X.columns][:topk]
    means = X[top].mean(numeric_only=True)
    small = [f for f in top if means[f] < mean_threshold]
    large = [f for f in top if means[f] >= mean_threshold]
    new_cols = add_interactions_to_X(X, small, large, eps=eps)
    info = {"small_feats": small, "large_feats": large, "eps": eps, "topk": top, "new_cols": new_cols}
    print(f"[derive] top{len(top)} -> small={len(small)}, large={len(large)}, new_cols={len(new_cols)}")
    return X, info, new_cols

def apply_derived_info_to_X(X, info):
    return add_interactions_to_X(X, info["small_feats"], info["large_feats"], eps=info.get("eps", 1e-9))

def extract_transit_shape_features(total_flux, baseline_win=5000, local_win=600):
    f = {}
    x = np.asarray(total_flux, dtype=np.float64); n = len(x)
    if n < 10:
        return {k:0.0 for k in [
            "shape_depth","shape_snr","shape_halfdur","shape_asym",
            "shape_curvature","shape_vu_ratio","shape_local_slope_l","shape_local_slope_r"
        ]}
    s = pd.Series(x)
    base = s.rolling(window=min(baseline_win, max(11, n//10)), center=True).median()
    base = base.fillna(method="bfill").fillna(method="ffill").to_numpy()
    d = x - base
    idx0 = int(np.argmin(d)); depth = float(-d[idx0]); sigma = float(np.nanstd(d))
    snr = depth / (sigma + 1e-12)
    w = int(min(local_win, n-1)); lo, hi = max(0, idx0 - w//2), min(n, idx0 + w//2)
    local = d[lo:hi]
    if local.size < 5:
        halfdur = asym = curv = vu = sl_l = sl_r = 0.0
    else:
        half = -depth/2.0; rel_idx0 = idx0 - lo
        left = local[:rel_idx0+1]; right = local[rel_idx0:]
        Li = np.where(left >= half)[0]; li = int(Li[-1]) if Li.size > 0 else 0
        Ri = np.where(right >= half)[0]; ri = int(Ri[0]) if Ri.size > 0 else (len(right)-1)
        halfdur = float((ri + (rel_idx0 - li)))
        left_w = float(rel_idx0 - li); right_w = float(ri)
        asym = float((right_w - left_w) / (left_w + right_w + 1e-12))
        rad = max(5, min(60, local.size//10))
        a_lo = max(0, rel_idx0 - rad); a_hi = min(local.size, rel_idx0 + rad + 1)
        yy = local[a_lo:a_hi]; xx = np.arange(len(yy), dtype=np.float64)
        if len(yy) >= 6:
            coef = np.polyfit(xx, yy, 2); curv = float(coef[0])
        else:
            curv = 0.0
        d1 = np.diff(local)
        vu = float(np.median(np.abs(d1)) / (depth + 1e-12))
        sl_win = max(3, min(40, local.size//20))
        sl_l = float(np.median(np.diff(local[max(0, rel_idx0-2*sl_win):rel_idx0])))
        sl_r = float(np.median(np.diff(local[rel_idx0:rel_idx0+2*sl_win])))
    f.update({
        "shape_depth": depth, "shape_snr": snr, "shape_halfdur": halfdur, "shape_asym": asym,
        "shape_curvature": curv, "shape_vu_ratio": vu, "shape_local_slope_l": sl_l, "shape_local_slope_r": sl_r,
    })
    return f

def extract_diff_volatility_features(total_flux):
    f = {}
    x = np.asarray(total_flux, dtype=np.float64)
    if x.size < 5:
        return {k:0.0 for k in [
            "d1_std","d1_medabs","d1_p95","d1_p99","d2_std",
            "d1_roll100_std_mean","d1_roll1000_std_max","d1_iqr",
            "vol_range_ratio"
        ]}
    d1 = np.diff(x); d2 = np.diff(d1)
    f["d1_std"] = float(np.std(d1)); f["d1_medabs"] = float(np.median(np.abs(d1)))
    f["d1_p95"] = float(np.percentile(np.abs(d1), 95)); f["d1_p99"] = float(np.percentile(np.abs(d1), 99))
    f["d2_std"] = float(np.std(d2)); f["d1_iqr"] = float(np.percentile(d1, 75) - np.percentile(d1, 25))
    s = pd.Series(d1); r100 = s.rolling(100, center=True).std().dropna(); r1000 = s.rolling(1000, center=True).std().dropna()
    f["d1_roll100_std_mean"] = float(r100.mean()) if len(r100) else 0.0
    f["d1_roll1000_std_max"] = float(r1000.max()) if len(r1000) else 0.0
    rng = float(np.max(x) - np.min(x)); mu = float(np.mean(x))
    f["vol_range_ratio"] = float(rng / (mu + 1e-12))
    return f

def _fft_autocorr(x):
    n = len(x); x = np.asarray(x, dtype=np.float64); x = x - np.mean(x)
    nfft = 1 << (n - 1).bit_length()
    f = np.fft.rfft(x, n=2*nfft)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= (ac[0] + 1e-12)
    return ac

def extract_autocorr_rich_features(total_flux, maxlag=4000):
    f = {}
    x = np.asarray(total_flux, dtype=np.float64)
    if x.size < 20:
        return {k: 0.0 for k in [
            "acf_first_zero_lag","acf_peak1_lag","acf_peak1_val",
            "acf_peak2_lag","acf_peak2_val","acf_int_time"
        ]}
    ac = _fft_autocorr(x); m = int(min(maxlag, len(ac)-1)); a = ac[1:m+1]
    zc = np.where(a <= 0)[0]; f["acf_first_zero_lag"] = float(zc[0]+1) if zc.size else float(m)
    peaks, _ = find_peaks(a, height=0.05, distance=5)
    if peaks.size > 0:
        p_sorted = peaks[np.argsort(a[peaks])[::-1]]
        p1 = int(p_sorted[0]); f["acf_peak1_lag"] = float(p1+1); f["acf_peak1_val"] = float(a[p1])
        if len(p_sorted) > 1:
            p2 = int(p_sorted[1]); f["acf_peak2_lag"] = float(p2+1); f["acf_peak2_val"] = float(a[p2])
        else:
            f["acf_peak2_lag"] = 0.0; f["acf_peak2_val"] = 0.0
    else:
        f["acf_peak1_lag"] = 0.0; f["acf_peak1_val"] = 0.0; f["acf_peak2_lag"] = 0.0; f["acf_peak2_val"] = 0.0
    pos = a[a > 0]; f["acf_int_time"] = float(np.sum(pos))
    return f

def extract_centroid_track_features(signal_3d, step=50, max_frames=20000):
    f = {}
    arr = np.asarray(signal_3d, dtype=np.float32); T, H, W = arr.shape if arr.ndim==3 else (0,0,0)
    if T == 0:
        return {k: 0.0 for k in [
            "centroid_x_std","centroid_y_std","centroid_drift",
            "centroid_speed_mean","centroid_speed_std","centroid_flux_corr"
        ]}
    idx = np.arange(0, T, step, dtype=int); 
    if idx.size > max_frames: idx = idx[:max_frames]
    xs = np.arange(W, dtype=np.float32)[None, :, None]
    ys = np.arange(H, dtype=np.float32)[:, None, None]
    cx, cy, tf = [], [], []
    for t in idx:
        frame = arr[t]; s = float(frame.sum()); tf.append(s)
        if s <= 0:
            if cx: cx.append(cx[-1]); cy.append(cy[-1])
            else:  cx.append(W/2.0);  cy.append(H/2.0)
            continue
        cx.append(float((frame * xs.squeeze(2)).sum() / s))
        cy.append(float((frame * ys.squeeze(2)).sum() / s))
    cx = np.asarray(cx); cy = np.asarray(cy); tf = np.asarray(tf)
    if cx.size < 3:
        return {k: 0.0 for k in [
            "centroid_x_std","centroid_y_std","centroid_drift",
            "centroid_speed_mean","centroid_speed_std","centroid_flux_corr"
        ]}
    f["centroid_x_std"] = float(np.std(cx)); f["centroid_y_std"] = float(np.std(cy))
    dx = cx - cx[0]; dy = cy - cy[0]; f["centroid_drift"] = float(np.hypot(dx[-1], dy[-1]))
    vx = np.diff(cx); vy = np.diff(cy); speed = np.hypot(vx, vy)
    f["centroid_speed_mean"] = float(np.mean(speed)); f["centroid_speed_std"]  = float(np.std(speed))
    if np.std(tf) > 0 and np.std(cx) > 0:
        f["centroid_flux_corr"] = float(np.corrcoef(tf, cx)[0,1])
    else:
        f["centroid_flux_corr"] = 0.0
    return f

def extract_segment_features(total_flux, n_segments=4):
    f = {}
    x = np.asarray(total_flux, dtype=np.float64); n = len(x)
    if n < n_segments:
        return {k: 0.0 for k in [
            "seg_mean_max","seg_mean_min","seg_std_mean","seg_min_min",
            "seg_range_max","seg_mean_slope","seg_mean_cv"
        ]}
    segs = np.array_split(x, n_segments)
    means = np.array([s.mean() for s in segs]); stds  = np.array([s.std()  for s in segs])
    mins  = np.array([s.min()  for s in segs]); maxs  = np.array([s.max()  for s in segs])
    rngs  = maxs - mins; xs = np.arange(n_segments, dtype=np.float64)
    A = np.vstack([xs, np.ones_like(xs)]).T; slope = float(np.linalg.lstsq(A, means, rcond=None)[0][0])
    f.update({
        "seg_mean_max": float(means.max()), "seg_mean_min": float(means.min()),
        "seg_std_mean": float(stds.mean()), "seg_min_min": float(mins.min()),
        "seg_range_max": float(rngs.max()), "seg_mean_slope": slope,
        "seg_mean_cv": float(stds.mean() / (means.mean() + 1e-12)),
    })
    return f

def _robust_std(x, axis=0):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return 1.4826 * mad + 1e-12

def load_cached_xy_csv(path, wl_cols=None, index_candidates=("planet_id","id","pid")):
    df = pd.read_csv(path)
    if wl_cols is None:
        y_cols = [c for c in df.columns if re.fullmatch(r"(wl|wavelength)_\d+", c)]
        if not y_cols: raise ValueError("Cached CSV에서 타깃 컬럼(wl_*)을 찾지 못했습니다.")
        wl_cols = sorted(y_cols, key=lambda x: int(re.findall(r"\d+", x)[0]))
    else:
        y_cols = list(wl_cols)
    idx_col = next((c for c in index_candidates if c in df.columns), None)
    if idx_col: df[idx_col] = df[idx_col].astype(str); df = df.set_index(idx_col)
    else:       df.index = df.index.astype(str)
    num_cols = df.select_dtypes(include=[np.number]).columns
    x_cols = [c for c in num_cols if c not in y_cols]
    X = df[x_cols].astype(np.float32).copy()
    y = df[y_cols].astype(np.float32).copy()
    return X, y, wl_cols

def adaptive_sg_blend(
    mu_raw,
    *,
    window=13,              # mid 구간 window
    poly=2,                 # SG poly
    alpha=0.65,             # mid 구간 α
    w_lo=9,  w_hi=25,       # oscillatory / flat window
    a_lo=0.85, a_hi=0.45    # oscillatory / flat α
):
    """
    스펙트럼 별 roughness = std(diff(mu))로 flat/mid/oscillatory 분기하여
    서로 다른 (W, α) 적용. 결과는 >=0 로 clip.
    """
    rough = np.std(np.diff(mu_raw, axis=1), axis=1)
    q1, q2 = np.percentile(rough, [33, 66])
    out = np.empty_like(mu_raw)
    for i, r in enumerate(rough):
        if r <= q1:            # flat → 크게/많이 스무딩
            W, A = w_hi, a_hi
        elif r <= q2:          # mid
            W, A = window, alpha
        else:                  # oscillatory → 작게/덜 스무딩
            W, A = w_lo, a_lo
        W = max(3, int(W) | 1); P = max(1, min(int(poly), W - 1))
        sm = savgol_filter(mu_raw[i], window_length=W, polyorder=P, mode="interp")
        out[i] = A * mu_raw[i] + (1.0 - A) * sm
    return np.clip(out, 0.0, None)

def _odd_window(n, prefer):
    w = int(prefer); w = max(3, min(w, (n if n%2==1 else n-1)))
    if w % 2 == 0: w -= 1
    w = max(3, w)
    return w

def apply_adc_correction(signal_array, instrument, adc_info_path=f"{BASE_PATH}/adc_info.csv"):
    adc_df = pd.read_csv(adc_info_path)
    gain = adc_df.at[0, f"{instrument}_adc_gain"]
    offset = adc_df.at[0, f"{instrument}_adc_offset"]
    return signal_array.astype(np.float32) / gain + offset

def extract_brightpixel_features(signal_3d, total_flux=None, relax=0.30):
    if total_flux is None:
        total_flux = np.sum(signal_3d, axis=(1, 2))
    bidx = int(np.argmax(total_flux))
    frame = signal_3d[bidx]
    max_val = float(np.max(frame))
    if max_val <= 0: mask = (frame == max_val)
    else:
        thr = max_val * (1.0 - relax)
        mask = frame >= thr
    vals = frame[mask]
    if vals.size == 0: vals = np.array([0.0], dtype=np.float32)
    return {
        "brightpix_npix": float(mask.sum()),
        "brightpix_brightframe_mean": float(np.mean(vals)),
        "brightpix_brightframe_max":  float(np.max(vals)),
        "brightpix_brightframe_min":  float(np.min(vals)),
    }

def _rolling_median(x, win):
    import pandas as pd, numpy as np
    w = int(max(11, min(win, len(x)//2)*1) )
    if w % 2 == 0: w += 1
    s = pd.Series(np.asarray(x, float))
    b = s.rolling(w, center=True).median()
    return b.fillna(method="bfill").fillna(method="ffill").to_numpy()

def _detrend_median(x, base_win=5000):
    x = np.asarray(x, float)
    base = _rolling_median(x, min(base_win, max(11, len(x)//10)))
    return x - base

def _transit_shape_physics(detrended):
    from scipy.signal import savgol_filter
    z = savgol_filter(detrended, max(31, (401 if len(detrended)>401 else (len(detrended)//5*2+1))), 3, mode="interp")
    i0 = int(np.argmin(z)); depth = float(-z[i0])
    half = -depth/2.0
    L = np.where(z[:i0] >= half)[0]; R = np.where(z[i0:] >= half)[0]
    l = int(L[-1]) if L.size else 0
    r = int(R[0])  if R.size else 0
    T14 = float(np.sum(z < 0))
    T23 = float(r + (i0 - l))
    d1 = np.gradient(z); d2 = np.gradient(d1)
    ingress = float(-np.min(d1[l:i0])) if i0>l else 0.0
    egress  = float(np.max(d1[i0:i0+r+1])) if r>0 else 0.0
    curv    = float(np.median(d2[max(0,i0-50):i0+51]))
    # 바깥 소음
    oot_std = float(np.std(np.concatenate([z[:max(1,l)], z[i0+r+1:]]))) + 1e-12
    return {
        "phys_depth": depth,
        "phys_T14": T14,
        "phys_T23": T23,
        "phys_T23_over_T14": T23 / (T14 + 1e-12),
        "phys_ingress_slope": ingress,
        "phys_egress_slope": egress,
        "phys_bottom_curvature": curv,
        "phys_snr_depth": depth / oot_std,
        "phys_energy": depth * T14
    }

def _rednoise_features(x, bin_sizes=(30,60,120,300)):
    x = np.asarray(x, float)
    z = x - np.median(x)
    out = {}
    base_std = np.std(z) + 1e-12
    eff_bins = [b for b in bin_sizes if len(z)//b >= 2]
    for B in eff_bins:
        n = len(z)//B
        xb = z[:n*B].reshape(n, B).mean(1)
        out[f"rn_bin{B}_std"]  = float(np.std(xb))
        out[f"rn_beta{B}"]     = float(np.std(xb) / (base_std/np.sqrt(B)))
    # PSD 기울기(저주파 지배 판단)
    fx  = np.fft.rfft(z); ps = (np.abs(fx)**2)[1:]
    fr  = np.fft.rfftfreq(len(z))[1:]
    m = (fr>0) & np.isfinite(ps)
    if m.sum()>10:
        X = np.vstack([np.log10(fr[m]), np.ones(m.sum())]).T
        a,_ = np.linalg.lstsq(X, np.log10(ps[m] + 1e-20), rcond=None)[0]
        out["rn_psd_slope"] = float(a)      # -1~-2면 적색 잡음 우세
    else:
        out["rn_psd_slope"] = 0.0
    return out

def _dct_lowfreq_coeffs(x, m=256, k=16):
    """균일 리샘플 후 DCT 저주파 k개(DC 포함)"""
    x = np.asarray(x, float)
    t = np.linspace(0, 1, len(x))
    grid = np.linspace(0, 1, m)
    y = np.interp(grid, t, x)
    y = (y - y.mean())/(y.std()+1e-12)
    c = _dct(y, type=2, norm="ortho")[:k]
    return {f"dct_lf_{i:02d}": float(c[i]) for i in range(k)}

def _wavelet_energy(x, wave="db4", levels=(2,3,4,5)):
    out = {}
    try:
        import pywt
        x = (np.asarray(x, float) - np.mean(x))/(np.std(x)+1e-12)
        L = max(levels) if len(x) > 2**max(levels) else int(np.floor(np.log2(len(x))) - 1)
        L = max(1, min(L, max(levels)))
        coeffs = pywt.wavedec(x, wave, level=L)
        for Lv in levels:
            if Lv <= L:
                cD = coeffs[Lv]
                out[f"wavE_L{Lv}"] = float(np.mean(cD**2))
        if L >= 2:
            lowE  = float(np.mean(coeffs[-1]**2))
            highE = float(np.mean(np.hstack(coeffs[1:-1])**2) + 1e-12)
            out["wavE_low_over_high"] = lowE / highE
        else:
            out["wavE_low_over_high"] = 0.0
    except Exception:
        # PyWavelets 없으면 생략
        out["wavE_low_over_high"] = 0.0
    return out

def _centroid_regression_residuals(signal_3d, step=100):
    """cx, cy, t, t^2 회귀 후 잔차 표준편차/딥"""
    arr = np.asarray(signal_3d, np.float32)
    T,H,W = arr.shape
    idx = np.arange(0, T, step, dtype=int)
    xs = np.arange(W, dtype=np.float32)[None, :, None]
    ys = np.arange(H, dtype=np.float32)[:, None, None]
    cx, cy, tf = [], [], []
    for t in idx:
        f = arr[t]; s = float(f.sum()); tf.append(s)
        if s <= 0:
            cx.append(cx[-1] if cx else W/2.0); cy.append(cy[-1] if cy else H/2.0)
        else:
            cx.append(float((f*xs.squeeze(2)).sum()/s))
            cy.append(float((f*ys.squeeze(2)).sum()/s))
    cx, cy, tf = np.asarray(cx), np.asarray(cy), np.asarray(tf)
    if cx.size < 5:
        return {"cr_res_std": 0.0, "cr_res_depth": 0.0}
    n = len(tf); t = np.linspace(-1,1,n)
    A = np.vstack([np.ones(n), cx, cy, t, t**2]).T
    coef = np.linalg.lstsq(A, tf, rcond=None)[0]
    res  = tf - A@coef
    return {"cr_res_std": float(np.std(res)), "cr_res_depth": float(np.max(np.median(tf)-res))}

def extract_global_flux_features(total_flux):
    f = {}
    f['global_flux_mean'] = np.mean(total_flux); f['global_flux_std']  = np.std(total_flux)
    f['global_flux_min']  = np.min(total_flux);  f['global_flux_max']  = np.max(total_flux)
    f['global_flux_range'] = f['global_flux_max'] - f['global_flux_min']
    f['global_flux_skew']   = skew(total_flux); f['global_flux_kurtosis'] = kurtosis(total_flux)
    f['global_flux_cv'] = f['global_flux_std'] / (f['global_flux_mean'] + 1e-12)
    for p in [1,5,10,25,50,75,90,95,99]:
        f[f'global_flux_p{p}'] = np.percentile(total_flux, p)
    f['global_flux_depth'] = f['global_flux_mean'] - f['global_flux_min']
    f['global_flux_depth_ratio'] = f['global_flux_depth'] / (f['global_flux_mean'] + 1e-12)
    return f

def extract_rolling_statistics_features(total_flux):
    f = {}
    s = pd.Series(total_flux)
    for window in [50,100,500,1000,2000,5000]:
        if window < len(total_flux):
            rm = s.rolling(window, center=True).mean().dropna()
            rs = s.rolling(window, center=True).std().dropna()
            rmin = s.rolling(window, center=True).min().dropna()
            rmax = s.rolling(window, center=True).max().dropna()
            f[f'rolling{window}_mean_min']  = float(rm.min())
            f[f'rolling{window}_mean_max']  = float(rm.max())
            f[f'rolling{window}_mean_std']  = float(rm.std())
            f[f'rolling{window}_std_mean']  = float(rs.mean())
            f[f'rolling{window}_std_max']   = float(rs.max())
            f[f'rolling{window}_deepest_dip'] = float(rmin.min())
            f[f'rolling{window}_highest_peak'] = float(rmax.max())
    return f

def extract_transit_detection_features(total_flux):
    f = {}
    baseline = pd.Series(total_flux).rolling(window=5000, center=True).median().fillna(method='bfill').fillna(method='ffill')
    detrended = total_flux - baseline
    f['detrended_min'] = float(np.min(detrended))
    f['detrended_std'] = float(np.std(detrended))
    f['detrended_skew'] = float(skew(detrended))
    f['detrended_neg_excursions']  = int(np.sum(detrended < -2*np.std(detrended)))
    f['detrended_deep_excursions'] = int(np.sum(detrended < -3*np.std(detrended)))
    thr = np.mean(total_flux) - 1.0*np.std(total_flux)
    below = total_flux < thr
    if np.any(below):
        diff = np.diff(np.concatenate(([False], below, [False])).astype(int))
        starts = np.where(diff == 1)[0]; ends = np.where(diff == -1)[0]
        durations = ends - starts
        f['longest_dip_duration'] = int(np.max(durations)) if len(durations)>0 else 0
        f['num_dip_periods']      = int(len(durations))
        f['total_dip_time']       = int(np.sum(durations))
        f['avg_dip_duration']     = float(np.mean(durations)) if len(durations)>0 else 0.0
        deepest_idx = int(np.argmin(total_flux))
        f['deepest_time_fraction'] = float(deepest_idx / len(total_flux))
    else:
        f.update(dict(longest_dip_duration=0, num_dip_periods=0, total_dip_time=0, avg_dip_duration=0.0, deepest_time_fraction=0.5))
    return f

def extract_frequency_features(total_flux):
    f = {}
    x = total_flux - np.mean(total_flux)
    fft_flux = np.fft.fft(x); fft_power = np.abs(fft_flux)
    freqs = np.fft.fftfreq(len(x))
    half = slice(1, len(x)//2)
    ps = np.abs(fft_power[half]); fr = np.abs(freqs[half])
    if len(ps)>0:
        f['fft_peak_power']  = float(np.max(ps))
        f['fft_total_power'] = float(np.sum(ps))
        f['fft_mean_power']  = float(np.mean(ps))
        f['fft_std_power']   = float(np.std(ps))
        low = np.abs(fr) < 0.01
        f['fft_low_freq_power']  = float(np.sum(ps[low]))
        f['fft_low_freq_ratio']  = float(f['fft_low_freq_power']/(f['fft_total_power']+1e-12))
    else:
        f.update(dict(fft_peak_power=0, fft_total_power=0, fft_mean_power=0, fft_std_power=0, fft_low_freq_power=0, fft_low_freq_ratio=0))
    return f

def extract_gradient_features(total_flux):
    f = {}
    d1 = np.diff(total_flux); d2 = np.diff(d1)
    f['flux_diff1_mean'] = float(np.mean(d1)); f['flux_diff1_std'] = float(np.std(d1))
    f['flux_diff1_min']  = float(np.min(d1));  f['flux_diff1_max'] = float(np.max(d1))
    f['flux_diff2_std']  = float(np.std(d2))
    return f

def extract_spatial_features(signal_data):
    f = {}
    idxs = [0, len(signal_data)//4, len(signal_data)//2, 3*len(signal_data)//4, -1]
    for i, idx in enumerate(idxs):
        frame = signal_data[idx]; tot = np.sum(frame)
        center = frame[12:20,12:20].sum() / (tot+1e-12)
        f[f'frame{i}_spatial_mean'] = float(np.mean(frame))
        f[f'frame{i}_spatial_std']  = float(np.std(frame))
        f[f'frame{i}_concentration'] = float(center)
    return f

def extract_sustained_slope_features(
    total_flux,
    base_win=5000,
    sg_win=401,
    k_mad=5.0,
    hysteresis=0.6,
    min_len=80
):
    """
    detrend → SG smoothing → |d1| 히스테리시스 run 추출
    """
    x = np.asarray(total_flux, dtype=np.float64)
    n = len(x)
    outs = {k:0.0 for k in [
        "ss_n_runs","ss_len_min","ss_len_median","ss_len_mean","ss_len_max",
        "ss_len_sum","ss_len_max_frac","ss_run_mean_absd1_mean","ss_run_peak_absd1_mean",
        "ss_len_at_dip","ss_dist_to_prev_run","ss_dist_to_next_run","ss_n_up","ss_n_down"
    ]}
    if n < 50:
        return outs
    # detrend + smoothing
    s = pd.Series(x)
    bw = min(base_win, max(11, n//10))
    base = s.rolling(bw, center=True).median().fillna(method="bfill").fillna(method="ffill").to_numpy()
    d = x - base
    W = int(sg_win) | 1
    if W >= n: W = n-1-(n%2==0)
    W = max(31, W)
    y = savgol_filter(d, window_length=W, polyorder=3, mode="interp")
    # thresholds
    d1 = np.diff(y)
    absd1 = np.abs(d1)
    mad = np.median(np.abs(d1 - np.median(d1))) + 1e-12
    thr_hi = 1.4826 * mad * k_mad
    thr_lo = thr_hi * float(hysteresis)
    mask_hi = absd1 >= thr_hi
    mask_lo = absd1 >= thr_lo
    # runs by hysteresis
    runs = []
    i, L = 0, len(d1)
    while i < L:
        if mask_hi[i]:
            s_idx = i; i += 1
            while i < L and mask_lo[i]: i += 1
            e_idx = i
            if e_idx - s_idx >= min_len:
                runs.append((s_idx, e_idx))
        else:
            i += 1
    if not runs:
        return outs
    lens = np.array([e - s for s, e in runs], dtype=np.float64)
    mean_abs = np.array([absd1[s:e].mean() for s, e in runs], dtype=np.float64)
    peak_abs = np.array([absd1[s:e].max()  for s, e in runs], dtype=np.float64)
    outs.update({
        "ss_n_runs": float(len(runs)),
        "ss_len_min": float(lens.min()),
        "ss_len_median": float(np.median(lens)),
        "ss_len_mean": float(lens.mean()),
        "ss_len_max": float(lens.max()),
        "ss_len_sum": float(lens.sum()),
        "ss_len_max_frac": float(lens.max() / (n + 1e-12)),
        "ss_run_mean_absd1_mean": float(mean_abs.mean()),
        "ss_run_peak_absd1_mean": float(peak_abs.mean()),
    })
    dip = int(np.argmin(y))
    len_at_dip = 0; prev_end = None; next_start = None
    for s_idx, e_idx in runs:
        if s_idx <= dip < e_idx: len_at_dip = e_idx - s_idx
        if e_idx <= dip: prev_end = e_idx
        if next_start is None and s_idx > dip: next_start = s_idx
    outs["ss_len_at_dip"] = float(len_at_dip)
    outs["ss_dist_to_prev_run"] = float(dip - prev_end) if prev_end is not None else 0.0
    outs["ss_dist_to_next_run"] = float(next_start - dip) if next_start is not None else 0.0
    up = sum(np.sign(d1[s:e]).mean() >= 0 for s, e in runs)
    downs = len(runs) - up
    outs["ss_n_up"] = float(up); outs["ss_n_down"] = float(downs)
    return outs

def extract_enhanced_fgs1_features(signal_3d, verbose=False):
    """
    기존 베이직셋 + (패치) 트랜짓물리/적색노이즈/DCT/웨이블릿/센트로이드회귀 잔차
    - 기존 코드의 apply_adc_correction, 기타 유틸이 같은 파일에 있다고 가정
    """
    # --- ADC 보정 & 총광도 ---
    signal_3d = apply_adc_correction(signal_3d, instrument='FGS1')
    total_flux = np.sum(signal_3d, axis=(1,2))

    feats = {}

    # ===== 기존 베이직 특징들 =====
    feats.update(extract_global_flux_features(total_flux))
    feats.update(extract_rolling_statistics_features(total_flux))
    feats.update(extract_transit_detection_features(total_flux))
    feats.update(extract_frequency_features(total_flux))
    feats.update(extract_gradient_features(total_flux))
    feats.update(extract_transit_shape_features(total_flux))
    feats.update(extract_sustained_slope_features(total_flux, base_win=5000, sg_win=401,
                                                  k_mad=5.0, hysteresis=0.6, min_len=80))
    feats.update(extract_diff_volatility_features(total_flux))
    feats.update(extract_autocorr_rich_features(total_flux))
    feats.update(extract_centroid_track_features(signal_3d, step=50, max_frames=20000))
    feats.update(extract_segment_features(total_flux, n_segments=4))
    feats.update(extract_spatial_features(signal_3d))
    if USE_BRIGHTPIXEL:
        feats.update(extract_brightpixel_features(signal_3d, total_flux=total_flux, relax=BRIGHTPIX_RELAX))

    # ====== (패치) 고효과 파생 ======
    detr = _detrend_median(total_flux, base_win=5000)
    feats.update(_transit_shape_physics(detr))
    feats.update(_rednoise_features(total_flux, bin_sizes=(30,60,120,300)))
    feats.update(_dct_lowfreq_coeffs(detr, m=256, k=16))
    feats.update(_wavelet_energy(total_flux, wave="db4", levels=(2,3,4,5)))
    feats.update(_centroid_regression_residuals(signal_3d, step=100))

    return feats

def filter_similar_derived(X, derived_cols, fi_df=None, thresh=0.90, max_check=2000):
    cols = [c for c in derived_cols if c in X.columns]
    const = [c for c in cols if float(np.std(X[c].values)) <= 1e-12]
    if const: X.drop(columns=const, inplace=True, errors="ignore")
    cols = [c for c in cols if c not in const]
    if not cols:
        print(f"[derive-dup] const={len(const)} -> no derived left")
        return []
    if fi_df is not None and "feature" in fi_df and "importance" in fi_df:
        imp = fi_df.set_index("feature")["importance"]
        cols = sorted(cols, key=lambda c: -float(imp.get(c, 0.0)))
    else:
        cols = sorted(cols, key=lambda c: -float(np.nanvar(X[c].values)))
    sub = cols[:min(len(cols), max_check)]
    C = np.abs(pd.DataFrame(X[sub], copy=False).corr().to_numpy())
    keep, dropped = [], set()
    for i, ci in enumerate(sub):
        if ci in dropped: continue
        keep.append(ci)
        high = np.where(C[i, i+1:] >= thresh)[0]
        for h in high:
            cj = sub[i+1+h]; dropped.add(cj)
    if dropped: X.drop(columns=list(dropped), inplace=True, errors="ignore")
    kept_full = [c for c in derived_cols if c in X.columns]
    print(f"[derive-dup] checked={len(sub)}, kept={len(keep)}, dropped≈{len(dropped)}, const={len(const)}")
    return kept_full

def load_fgs1_signals_for_planet(planet_id_str, split="train"):
    base = f"{BASE_PATH}/{split}/{int(planet_id_str)}"
    paths = sorted(glob.glob(f"{base}/FGS1_signal_*.parquet"))
    if MAX_SIGNALS_PER_PLANET is not None:
        paths = paths[:MAX_SIGNALS_PER_PLANET]
    return paths

def build_features(split="train", star_info_df=None, targets_df=None, frac=1.0):
    sid = star_info_df.copy()
    sid['planet_id'] = sid['planet_id'].astype(int).astype(str)
    if split=="train" and (0 < frac < 1.0):
        n = max(1, int(len(sid)*frac))
        sid = sid.sample(n=n, random_state=SEED).sort_values("planet_id")
    feats = []
    for _, row in tqdm(sid.iterrows(), total=len(sid), desc=f"Extract {split} FGS1 features"):
        pid = row['planet_id']
        for j, p in enumerate(load_fgs1_signals_for_planet(pid, split=split)):
            try:
                df = pd.read_parquet(p)
                signal_3d = df.values.reshape(135000, 32, 32)
                f = extract_enhanced_fgs1_features(signal_3d, verbose=False)
                f['planet_id'] = f"{pid}_{j}"
                feats.append(f)
            except Exception as e:
                print(f"[WARN] {split} {pid} signal{j} fail:", e)
    X = pd.DataFrame(feats).set_index("planet_id").sort_index()
    meta = star_info_df.copy(); meta['planet_id'] = meta['planet_id'].astype(int).astype(str)
    exp = []
    for pid in X.index:
        base_id = pid.split("_")[0]
        m = meta[meta['planet_id']==base_id].copy()
        if not m.empty:
            m['planet_id'] = pid; exp.append(m)
    if len(exp)>0:
        meta_expand = pd.concat(exp, ignore_index=True).set_index("planet_id").sort_index()
        X = X.join(meta_expand.select_dtypes(include=[np.number]).astype(np.float32), how="left")
    X = X.select_dtypes(include=[np.number]).fillna(0).astype(np.float32)
    if split=="train":
        tgt = targets_df.copy()
        tgt['planet_id'] = tgt['planet_id'].astype(int).astype(str)
        tgt = tgt.set_index("planet_id")[WL_COLS].astype(np.float32)
        y_rows = []
        for pid in X.index:
            base_id = pid.split("_")[0]
            if base_id in tgt.index:
                r = tgt.loc[base_id].copy(); r.name = pid; y_rows.append(r)
        y = pd.DataFrame(y_rows).sort_index().astype(np.float32)
        common = X.index.intersection(y.index)
        X = X.loc[common]; y = y.loc[common]
        print(f"[SHAPE] X:{X.shape}, y:{y.shape}")
        return X, y
    else:
        print(f"[SHAPE] X_test:{X.shape}")
        return X

# ----------------------- Train / Post-calibration / Predict -----------------------
def train_lgbm_cv(X, y):
    print("[LGBM] Device: CPU")
    params = BASE_LGB_PARAMS.copy()
    models = [[] for _ in range(y.shape[1])]
    oof_pred = np.zeros_like(y.values, dtype=np.float32)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for t_idx, target in enumerate(WL_COLS):
        y_t = y[target].values
        fold_models = []
        for fold, (tr, va) in enumerate(kf.split(X), 1):
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X.iloc[tr], y_t[tr],
                eval_set=[(X.iloc[va], y_t[va])],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
            )
            pred_va = m.predict(X.iloc[va]).astype(np.float32)
            oof_pred[va, t_idx] = pred_va
            fold_models.append(m)
        models[t_idx] = fold_models
        if (t_idx+1) % 25 == 0:
            print(f"[{t_idx+1}/{y.shape[1]}] targets trained")
    resid_base = y.values - oof_pred
    mad   = np.median(np.abs(resid_base - np.median(resid_base, axis=0, keepdims=True)), axis=0)
    target_unc = 1.4826 * mad
    return {
        "models": models,
        "feature_columns": list(X.columns),
        "target_columns": WL_COLS,
        "target_uncertainties": target_unc.astype(np.float32),
        "oof_predictions": oof_pred,
        "y_true": y.values,
        "index": list(y.index),
    }

def _apply_smooth_blend(mu, window=13, poly=2, alpha=0.65):
    W = _odd_window(mu.shape[1], window)
    smooth = savgol_filter(mu, window_length=W, polyorder=poly, axis=1, mode="interp")
    mu2 = alpha * mu + (1.0 - alpha) * smooth
    mu2 = np.clip(mu2, 0.0, None)
    return mu2

def fit_postcalibration(trained, y_true,
                        smooth_window=POST_SMOOTH_WINDOW,
                        smooth_poly=POST_SMOOTH_POLY,
                        blend_alpha=POST_BLEND_ALPHA,
                        clip_fgs=SIGMA_CLIP_FGS, clip_airs=SIGMA_CLIP_AIRS,
                        final_sigma_scale=SIGMA_FINAL_SCALE):
    oof = trained["oof_predictions"].astype(np.float64)
    base_unc = trained["target_uncertainties"].astype(np.float64)
    bias = np.mean(y_true - oof, axis=0)  # y - pred
    oof_corr = oof + bias
    oof_final = _apply_smooth_blend(oof_corr, window=smooth_window, poly=smooth_poly, alpha=blend_alpha)
    resid = y_true - oof_final
    mad = np.median(np.abs(resid - np.median(resid, axis=0, keepdims=True)), axis=0)
    robust_std = 1.4826 * mad + 1e-12
    scale = robust_std / (base_unc + 1e-12)
    if scale.size >= 1: scale[0]  = np.clip(scale[0],  clip_fgs[0],  clip_fgs[1])
    if scale.size >= 2: scale[1:] = np.clip(scale[1:], clip_airs[0], clip_airs[1])
    sigma_calib = base_unc * scale * final_sigma_scale
    trained["post_bias"] = bias.astype(np.float32)
    trained["post_alpha"] = float(blend_alpha)
    trained["post_smooth_window"] = int(smooth_window)
    trained["post_smooth_poly"] = int(smooth_poly)
    trained["calibrated_uncertainties"] = sigma_calib.astype(np.float32)
    trained["oof_predictions_post"] = oof_final.astype(np.float32)
    return trained

def apply_postprocess_to_preds(mu_raw, trained):
    bias = trained.get("post_bias", np.zeros(mu_raw.shape[1], dtype=np.float32))
    alpha = trained.get("post_alpha", POST_BLEND_ALPHA)
    window = trained.get("post_smooth_window", POST_SMOOTH_WINDOW)
    poly   = trained.get("post_smooth_poly", POST_SMOOTH_POLY)
    mu_corr = mu_raw + bias
    mu_post = _apply_smooth_blend(mu_corr, window=window, poly=poly, alpha=alpha)
    return mu_post

def gll_score_numpy(y_true, y_pred, sigma_pred,
                    naive_mean, naive_sigma,
                    fsg_sigma_true=1e-6, airs_sigma_true=1e-5, fgs_weight=0.4):
    sigma_pred = np.clip(sigma_pred, 1e-15, None)
    n_samples, n_waves = sigma_pred.shape
    sigma_true = np.append([fsg_sigma_true], np.full(n_waves-1, airs_sigma_true))
    sigma_true = np.tile(sigma_true, (n_samples, 1))
    weights = np.append([fgs_weight], np.ones(n_waves-1))
    weights = np.tile(weights, (n_samples, 1))
    gll_pred  = norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    gll_true  = norm.logpdf(y_true, loc=y_true, scale=sigma_true)
    gll_naive = norm.logpdf(y_true, loc=naive_mean, scale=naive_sigma)
    ind = (gll_pred - gll_naive) / (gll_true - gll_naive + 1e-12)
    return float(np.clip(np.average(ind, weights=weights), 0.0, 1.0))

def train_pca_coefficient_regressor(X_train: pd.DataFrame, Y_train: np.ndarray,
                                    k=24, alpha=1.0, seed=42):
    y_mean = Y_train.mean(0)
    y_std  = Y_train.std(0) + 1e-12
    Y_std = (Y_train - y_mean) / y_std
    k_eff = int(min(k, max(4, Y_std.shape[1]//4)))
    pca = PCA(n_components=k_eff, random_state=seed)
    C = pca.fit_transform(Y_std)
    reg = Ridge(alpha=alpha)
    reg.fit(X_train.values, C)
    return {"pca": pca, "reg": reg, "y_mean": y_mean.astype(np.float32), "y_std": y_std.astype(np.float32)}

def predict_with_pca_regressor(pack, X: pd.DataFrame):
    C_hat = pack["reg"].predict(X.values)
    Y_std_hat = pack["pca"].inverse_transform(C_hat)
    Y_hat = Y_std_hat * pack["y_std"] + pack["y_mean"]
    return np.clip(Y_hat, 0.0, None).astype(np.float32)

def _xcorr_shift(a, b, max_shift=3):
    best_d, best_corr = 0, -1e18
    for d in range(-max_shift, max_shift+1):
        if d < 0:   x, y = a[:len(a)+d], b[-d:]
        elif d > 0: x, y = a[d:], b[:len(b)-d]
        else:       x, y = a, b
        if len(x) < 10: continue
        c = np.corrcoef(x, y)[0, 1]
        if np.isfinite(c) and c > best_corr:
            best_corr, best_d = c, d
    return best_d

def _shift_apply_air(mu_row, d):
    out = mu_row.copy()
    if d == 0: return out
    a = out[1:]
    if d > 0:
        out[1+d:] = a[:-d]; out[1:1+d] = a[:1]
    else:
        d = -d
        out[1:-d] = a[d:]; out[-d:] = a[-1]
    return out

def build_template_library(Y_train, k_pca=12, n_neighbors=8, seed=42):
    pca = PCA(n_components=k_pca, random_state=seed)
    Z = pca.fit_transform(Y_train[:, 1:])  # AIRS만 사용
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(Z)
    return {"pca": pca, "nn": nn, "Y": Y_train.astype(np.float32)}

def align_with_templates(mu_pred_all, lib, max_shift=3):
    pca, nn, Y = lib["pca"], lib["nn"], lib["Y"]
    Zp = pca.transform(mu_pred_all[:, 1:])
    _, idx = nn.kneighbors(Zp, return_distance=True)
    out = np.empty_like(mu_pred_all)
    for i in range(mu_pred_all.shape[0]):
        tpl = Y[idx[i]].mean(axis=0)
        d = _xcorr_shift(mu_pred_all[i, 1:], tpl[1:], max_shift=max_shift)
        out[i] = _shift_apply_air(mu_pred_all[i], d)
    return out

def predict_with_uncertainty(trained, X):
    models = trained["models"]; feats = trained["feature_columns"]
    X = X.reindex(columns=feats, fill_value=0)
    n_targets = len(models)
    preds = np.zeros((len(X), n_targets), dtype=np.float32)
    for t in range(n_targets):
        fold_models = models[t]
        fold_preds = [m.predict(X).astype(np.float32) for m in fold_models]
        preds[:, t] = np.mean(fold_preds, axis=0)
    # 후보정
    mu = apply_postprocess_to_preds(preds, trained)
    # 템플릿 정렬
    if "template_lib" in trained:
        mu = align_with_templates(mu, trained["template_lib"], max_shift=trained.get("max_shift", 3))
    # PCA 서브스페이스 블렌드
    if "pca_pack" in trained:
        mu_pca = predict_with_pca_regressor(trained["pca_pack"], X)
        rough = np.std(np.diff(mu[:, 1:], axis=1), axis=1, keepdims=True)  # AIRS roughness
        q1, q2 = np.quantile(rough, [0.2, 0.8])
        w = np.clip((rough - q1) / (q2 - q1 + 1e-12), 0, 1) * (0.6 - 0.3) + 0.3  # 0.3~0.6
        mu = (1 - w) * mu + w * mu_pca
    # SG 한번 더(미세 계단 제거)
    mu = adaptive_sg_blend(mu, window=POST_SMOOTH_WINDOW, poly=POST_SMOOTH_POLY, alpha=POST_BLEND_ALPHA)
    # σ
    sigmas_base = trained.get("calibrated_uncertainties", trained["target_uncertainties"])
    sigmas = np.tile(sigmas_base, (len(X), 1))
    return mu.astype(np.float32), sigmas.astype(np.float32)

def make_submission(trained, test_star_info, out_path="submission.csv"):
    X_test = build_features(split="test", star_info_df=test_star_info, targets_df=None, frac=1.0)
    if "derived_info" in trained:
        apply_derived_info_to_X(X_test, trained["derived_info"])
    y_pred, sigma_pred = predict_with_uncertainty(trained, X_test)
    base_ids = [idx.split("_")[0] for idx in X_test.index]
    df_mu    = pd.DataFrame(y_pred, index=base_ids, columns=WL_COLS)
    df_sigma = pd.DataFrame(sigma_pred, index=base_ids, columns=[f"sigma_{i+1}" for i in range(y_pred.shape[1])])
    mu_mean    = df_mu.groupby(df_mu.index).mean()
    sigma_mean = df_sigma.groupby(df_sigma.index).mean()
    sub = pd.concat([mu_mean, sigma_mean], axis=1).reset_index().rename(columns={"index":"planet_id"})
    sub["planet_id"] = sub["planet_id"].astype(int)
    cols = list(sample_sub.columns)
    sub = sub.set_index("planet_id").reindex(sample_sub["planet_id"]).reset_index()
    sub = sub[["planet_id"] + cols[1:]].fillna(0)
    sub.to_csv(out_path, index=False)
    print(f"[SAVE] submission -> {out_path}  shape={sub.shape}")
    return sub

#--------------------cluster, GMM--------------------
def _select_volatility_columns(X: pd.DataFrame):
    cols = []
    prefixes = (
        "d1_", "d2_", "d3_", "diff_", "delta_", "flux_diff",
        "vol_", "std_", "var_", "mad_", "iqr_", "rms_", "zscore_",
        "acf_", "pacf_", "autocorr_", "lag_", "xcorr_",
        "fft_", "spec_", "psd_", "welch_", "stft_", "hilbert_", "envelope_", "band_",
        "seg_", "segment_", "cpd_", "changepoint_", "ruptures_", "bkpt_",
        "rolling", "roll_", "window_", "mov_", "ma_", "sma_", "ema_", "ewm_",
        "detrended_", "resid_", "noise_", "snr_",
        "shape_", "curv_", "curvature_", "asym_", "half", "fwhm_",
        "ss_", "centroid_",
    )
    extra_exact = {
        "global_flux_cv", "shape_vu_ratio", "shape_halfdur", "shape_asym", "shape_curvature",
        "d1_roll100_std_mean", "d1_roll1000_std_max", "d1_iqr",
        "global_flux_std", "global_flux_range", "global_flux_depth", "global_flux_depth_ratio",
        "global_flux_skew", "global_flux_kurtosis", "vol_range_ratio",
        "acf_int_time", "acf_peak1_val", "acf_peak2_val",
        "fft_peak_power", "fft_total_power", "fft_low_freq_ratio",
        "ss_n_runs", "ss_len_max", "ss_len_sum", "ss_len_max_frac",
        "ss_run_mean_absd1_mean", "ss_run_peak_absd1_mean",
    }
    for c in X.columns:
        if c in extra_exact: cols.append(c); continue
        if any(c.startswith(p) for p in prefixes): cols.append(c)
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 5: cols = X.columns.tolist()
    return sorted(list(dict.fromkeys(cols)))

def cluster_and_gmm_augment(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    k_clusters: int = 3,
    aug_mult: float = 0.5,
    gm_components: int = 3,
    y_pca_dim: int = 24,
    random_state: int = 42,
    min_cluster_size: int = 30
):
    rng = np.random.RandomState(random_state)
    X = X.copy(); y = y.copy()
    vol_cols = _select_volatility_columns(X)
    X_vol = X[vol_cols].values.astype(np.float32)
    vol_scaler = StandardScaler(); X_vol_scaled = vol_scaler.fit_transform(X_vol)
    y_scaler = StandardScaler(with_mean=True, with_std=True)
    y_std = y_scaler.fit_transform(y.values.astype(np.float32))
    y_pca_dim_eff = int(min(y_pca_dim, y_std.shape[1]-1, max(4, y_std.shape[1]//4)))
    pca = PCA(n_components=y_pca_dim_eff, random_state=random_state)
    y_pca = pca.fit_transform(y_std)
    y_pca_scaler = StandardScaler(); y_pca_scaled = y_pca_scaler.fit_transform(y_pca)
    gmm_cluster = GaussianMixture(
        n_components=k_clusters, covariance_type="full",
        random_state=random_state, reg_covar=1e-6
    )
    gmm_cluster.fit(X_vol_scaled)
    probs = gmm_cluster.predict_proba(X_vol_scaled)
    labels = probs.argmax(axis=1)
    X_synth_rows, y_synth_rows, synth_index = [], [], []
    all_cols = X.columns.tolist()
    other_cols = [c for c in all_cols if c not in vol_cols]
    X_df = X.copy()
    for c in range(k_clusters):
        mask = (labels == c); n_c = int(mask.sum())
        if n_c < min_cluster_size or aug_mult <= 0: continue
        n_aug = int(max(1, round(n_c * aug_mult)))
        Z = np.hstack([X_vol_scaled[mask], y_pca_scaled[mask]])
        cov_type = "diag" if n_c < 200 else "full"
        gm = GaussianMixture(
            n_components=min(gm_components, max(1, n_c // 20)),
            covariance_type=cov_type, random_state=random_state, reg_covar=1e-6
        )
        gm.fit(Z)
        Z_synth, _ = gm.sample(n_aug)
        x_vol_synth_scaled = Z_synth[:, :X_vol_scaled.shape[1]]
        y_pca_synth_scaled = Z_synth[:, X_vol_scaled.shape[1]:]
        x_vol_synth = vol_scaler.inverse_transform(x_vol_synth_scaled)
        y_pca_synth = y_pca_scaler.inverse_transform(y_pca_synth_scaled)
        y_std_synth = pca.inverse_transform(y_pca_synth)
        y_synth = y_scaler.inverse_transform(y_std_synth)
        y_synth = np.clip(y_synth, 0.0, None)
        cluster_stats = X_df[mask].agg(["mean", "std"])
        other_means = cluster_stats.loc["mean", other_cols].values.astype(np.float32)
        other_stds  = cluster_stats.loc["std",  other_cols].values.astype(np.float32)
        jitter = rng.normal(loc=0.0, scale=np.maximum(other_stds*0.05, 1e-6), size=(n_aug, len(other_cols)))
        for i in range(n_aug):
            row = np.empty(len(all_cols), dtype=np.float32)
            row[[all_cols.index(c) for c in vol_cols]] = x_vol_synth[i]
            row[[all_cols.index(c) for c in other_cols]] = (other_means + jitter[i])
            X_synth_rows.append(row)
        y_synth_rows.append(y_synth)
        synth_index += [f"synth_c{c}_{i}" for i in range(n_aug)]
    if not X_synth_rows:
        info = {
            "labels": labels, "cluster_sizes": [int((labels==i).sum()) for i in range(k_clusters)],
            "X_aug": X, "y_aug": y, "vol_cols": vol_cols
        }
        print("[AUG] No augmentation performed (clusters too small or aug_mult<=0).")
        return labels, gmm_cluster, info
    X_synth = pd.DataFrame(X_synth_rows, columns=all_cols, index=synth_index).astype(np.float32)
    y_synth = pd.DataFrame(np.vstack(y_synth_rows), columns=y.columns, index=synth_index).astype(np.float32)
    X_aug = pd.concat([X, X_synth], axis=0); y_aug = pd.concat([y, y_synth], axis=0)
    info = {
        "labels": labels, "cluster_sizes": [int((labels==i).sum()) for i in range(k_clusters)],
        "n_synth": len(X_synth), "X_aug": X_aug, "y_aug": y_aug, "vol_cols": vol_cols
    }
    print(f"[AUG] clusters={info['cluster_sizes']}, synth={info['n_synth']} rows")
    return labels, gmm_cluster, info

# ----------------------- Viz -----------------------
def plot_oof_lines(y_true, y_pred, wavelengths, ids, out_dir, batch_size=10, show=False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    N = len(ids); batches = (N + batch_size - 1)//batch_size
    for b in range(batches):
        idxs = slice(b*batch_size, min((b+1)*batch_size, N))
        n = idxs.stop - idxs.start; ncols=5; nrows=int(np.ceil(n/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2.8*nrows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for i, k in enumerate(range(idxs.start, idxs.stop)):
            ax = axes[i]
            yt = y_true[k]; yp = y_pred[k]
            ax.plot(wavelengths, yt, lw=1.5, label="truth")
            ax.plot(wavelengths, yp, lw=1.2, alpha=0.9, label="pred")
            ax.set_title(f"{ids[k]}", fontsize=9); ax.grid(True, alpha=0.3)
            if i//ncols == nrows-1: ax.set_xlabel("Wavelength (µm)")
            if i % ncols == 0:      ax.set_ylabel("Transit depth")
        for j in range(n, nrows*ncols): axes[j].set_visible(False)
        if b==0: axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"OOF spectra (batch {b+1}/{batches})", y=0.98, fontsize=12)
        fig.tight_layout()
        out = Path(out_dir)/f"oof_batch_{b+1:03d}.png"
        fig.savefig(out, dpi=150)
        if show: plt.show()
        plt.close(fig)

def plot_bias_sigma(trained, wavelengths, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    bias = trained.get("post_bias", np.zeros_like(wavelengths, dtype=np.float32))
    sig_base = trained["target_uncertainties"]
    sig_cal  = trained.get("calibrated_uncertainties", sig_base)
    scale = sig_cal / (sig_base + 1e-12)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(wavelengths, bias, lw=1.2)
    ax.axhline(0, ls="--", c="k", lw=0.8)
    ax.set_ylabel("bias (y - oof)")
    ax.set_title("λ-wise bias / uncertainty / scale")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(wavelengths, sig_base, lw=1.0, label="base_unc")
    ax.plot(wavelengths, sig_cal,  lw=1.0, label="calibrated_unc")
    ax.set_ylabel("σ"); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[2]
    ax.plot(wavelengths, scale, lw=1.0)
    ax.axhline(1.0, ls="--", c="k", lw=0.8)
    ax.set_ylabel("scale (=σ_cal/σ_base)"); ax.set_xlabel("Wavelength (µm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir)/"postcalib_bias_sigma.png"
    fig.savefig(out, dpi=150); plt.close(fig)

def plot_residual_std_vs_sigma(y_true, trained, wavelengths, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mu = trained.get("oof_predictions_post", trained["oof_predictions"])
    resid = y_true - mu
    med = np.median(resid, axis=0, keepdims=True)
    mad = np.median(np.abs(resid - med), axis=0)
    robstd = 1.4826 * mad
    sig_cal = trained.get("calibrated_uncertainties", trained["target_uncertainties"])
    fig = plt.figure(figsize=(12,4.2)); ax = plt.gca()
    ax.plot(wavelengths, robstd, lw=1.1, label="residual robust std")
    ax.plot(wavelengths, sig_cal, lw=1.1, label="pred σ (calibrated)")
    ax.set_title("Residual robust std vs predicted σ"); ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); out = Path(out_dir)/"resid_vs_sigma.png"
    fig.savefig(out, dpi=150); plt.close(fig)

def plot_scatter_and_residuals(y_true, y_pred, wavelengths, out_dir, picks=(0,50,150,250), show=False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for j in picks:
        fig = plt.figure(figsize=(12,4))
        ax1 = plt.subplot(1,2,1)
        ax1.scatter(y_true[:,j], y_pred[:,j], s=6, alpha=0.6)
        lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
        ax1.plot(lims, lims, 'k--', lw=1); ax1.set_xlim(lims); ax1.set_ylim(lims)
        ax1.set_title(f"y_true vs pred @λ{j+1} ({wavelengths[j]:.3f} µm)")
        ax1.set_xlabel("y_true"); ax1.set_ylabel("y_pred"); ax1.grid(True, alpha=0.3)
        ax2 = plt.subplot(1,2,2)
        res = y_true[:,j] - y_pred[:,j]
        ax2.hist(res, bins=40, alpha=0.8)
        ax2.set_title(f"Residuals @λ{j+1}  mean={res.mean():.3e}, std={res.std():.3e}")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        out = Path(out_dir)/f"oof_scatter_residuals_lambda_{j+1:03d}.png"
        fig.savefig(out, dpi=150)
        if show: plt.show()
        plt.close(fig)

def save_cached_xy_csv(X: pd.DataFrame, y: pd.DataFrame, path: str, wl_cols=None, index_name="planet_id"):
    dfX = X.copy(); dfY = y.copy() if y is not None else None
    idx = dfX.index.astype(str); dfX.index = idx
    if dfY is not None:
        dfY = dfY.copy(); dfY.index = dfY.index.astype(str)
        if wl_cols is not None: dfY = dfY[wl_cols]
        common = dfX.index.intersection(dfY.index); dfX = dfX.loc[common]; dfY = dfY.loc[common]
        out = pd.concat([dfX, dfY], axis=1)
    else:
        out = dfX
    out.to_csv(path, index=True)
    print(f"[CACHE] saved X,y -> {path}  shape={out.shape}")

# ----------------------- Postprocess 튜닝 -----------------------
def _cv_eval_postprocess(
    oof_raw, y_true,
    smooth_window, smooth_poly, alpha,
    clip_fgs, clip_airs, final_sigma_scale,
    naive_mean, naive_sigma, fgs_weight=0.4,
    n_splits=5, seed=42
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(oof_raw):
        oof_tr, y_tr = oof_raw[tr], y_true[tr]
        oof_va, y_va = oof_raw[va], y_true[va]
        bias = np.mean(y_tr - oof_tr, axis=0)
        resid_tr_base = y_tr - oof_tr
        base_unc = _robust_std(resid_tr_base, axis=0)
        oof_tr_post = _apply_bias_smooth_blend(oof_tr, bias, smooth_window, smooth_poly, alpha)
        resid_tr_post = y_tr - oof_tr_post
        post_unc = _robust_std(resid_tr_post, axis=0)
        scale = post_unc / (base_unc + 1e-12)
        if scale.size >= 1: scale[0]  = np.clip(scale[0],  clip_fgs[0],  clip_fgs[1])
        if scale.size >= 2: scale[1:] = np.clip(scale[1:], clip_airs[0], clip_airs[1])
        sigma_vec = base_unc * scale * final_sigma_scale
        oof_va_post = _apply_bias_smooth_blend(oof_va, bias, smooth_window, smooth_poly, alpha)
        sigma_val = np.tile(sigma_vec, (len(va), 1))
        sc = gll_score_numpy(y_va, oof_va_post, sigma_val, naive_mean, naive_sigma, fgs_weight=fgs_weight)
        scores.append(sc)
    return float(np.mean(scores)), float(np.std(scores))

def _apply_bias_smooth_blend(mu, bias, window, poly, alpha):
    mu_corr = mu + bias
    W = _odd_window(mu_corr.shape[1], window)
    poly = int(min(int(poly), W - 1))
    sm  = savgol_filter(mu_corr, window_length=W, polyorder=poly, axis=1, mode="interp")
    out = alpha * mu_corr + (1.0 - alpha) * sm
    return np.clip(out, 0.0, None)

def tune_postprocess_hparams(
    trained, y_true, naive_mean, naive_sigma,
    init=dict(window=13, poly=2, alpha=0.65,
              clip_fgs=(0.85,1.30), clip_airs=(0.92,1.22), sigma_scale=1.04),
    n_splits=5, seed=42, verbose=True
):
    oof_raw = trained["oof_predictions"].astype(np.float32)
    win_grid  = [7, 9, 11, 13, 15, 17]
    poly_grid = [2, 3]
    alp_grid  = [0.50, 0.60, 0.65, 0.70, 0.75]
    best = dict(**init); best_score = -1.0
    for w in win_grid:
        for p in poly_grid:
            if p >= w:  continue
            for a in alp_grid:
                sc, sd = _cv_eval_postprocess(
                    oof_raw, y_true, w, p, a,
                    best["clip_fgs"], best["clip_airs"], best["sigma_scale"],
                    naive_mean, naive_sigma, n_splits=n_splits, seed=seed
                )
                if sc > best_score:
                    best_score = sc; best.update(window=w, poly=p, alpha=a)
                    if verbose:
                        print(f"[Stage1] win={w}, poly={p}, alpha={a} -> GLL={sc:.6f} (±{sd:.6f})")
    fgs_lo_grid = [0.80, 0.85, 0.90]; fgs_hi_grid = [1.20, 1.25, 1.30, 1.35]
    for lo in fgs_lo_grid:
        for hi in fgs_hi_grid:
            if lo >= hi: continue
            sc, sd = _cv_eval_postprocess(
                oof_raw, y_true, best["window"], best["poly"], best["alpha"],
                (lo, hi), best["clip_airs"], best["sigma_scale"],
                naive_mean, naive_sigma, n_splits=n_splits, seed=seed
            )
            if sc > best_score:
                best_score = sc; best["clip_fgs"] = (lo, hi)
                if verbose: print(f"[Stage2a] FGS clip={best['clip_fgs']} -> GLL={sc:.6f} (±{sd:.6f})")
    air_lo_grid = [0.90, 0.92, 0.94]; air_hi_grid = [1.18, 1.20, 1.22, 1.24]
    for lo in air_lo_grid:
        for hi in air_hi_grid:
            if lo >= hi: continue
            sc, sd = _cv_eval_postprocess(
                oof_raw, y_true, best["window"], best["poly"], best["alpha"],
                best["clip_fgs"], (lo, hi), best["sigma_scale"],
                naive_mean, naive_sigma, n_splits=n_splits, seed=seed
            )
            if sc > best_score:
                best_score = sc; best["clip_airs"] = (lo, hi)
                if verbose: print(f"[Stage2b] AIRS clip={best['clip_airs']} -> GLL={sc:.6f} (±{sd:.6f})")
    for s in [1.00, 1.02, 1.04, 1.06]:
        sc, sd = _cv_eval_postprocess(
            oof_raw, y_true, best["window"], best["poly"], best["alpha"],
            best["clip_fgs"], best["clip_airs"], s,
            naive_mean, naive_sigma, n_splits=n_splits, seed=seed
        )
        if sc > best_score:
            best_score = sc; best["sigma_scale"] = s
            if verbose: print(f"[Stage2c] sigma_scale={s} -> GLL={sc:.6f} (±{sd:.6f})")
    if verbose:
        print("\n[BEST] ",
              f"win={best['window']}, poly={best['poly']}, alpha={best['alpha']}, ",
              f"FGS={best['clip_fgs']}, AIRS={best['clip_airs']}, scale={best['sigma_scale']}, ",
              f"GLL={best_score:.6f}")
    return best, best_score

def tune_postprocess_hparams_wide(
    trained, y_true, naive_mean, naive_sigma,
    n_splits=5, seed=42, verbose=True,
    extra_trials=300,
    clip_random_trials=800,
    grid_max_pairs=400,
    fgs_lo_range=(0.45, 1.20, 0.05),
    fgs_hi_range=(1.00, 2.00, 0.05),
    air_lo_range=(0.70, 1.05, 0.03),
    air_hi_range=(1.00, 1.70, 0.05),
    sigma_scale_range=(1.50, 3.0, 0.02),
):
    best, best_score = tune_postprocess_hparams(
        trained, y_true, naive_mean, naive_sigma,
        n_splits=n_splits, seed=seed, verbose=verbose
    )
    oof_raw = np.asarray(trained["oof_predictions"], dtype=np.float32)
    def _step_sample(rng):
        lo, hi, step = rng; n = int(np.floor((hi - lo) / step))
        k = np.random.randint(0, n + 1); return float(lo + k * step)
    # 1) 풀 랜덤
    for _ in range(int(extra_trials)):
        w = int(np.random.choice([7, 9, 11, 13, 15, 17, 19]))
        p = int(np.random.choice([2, 3])); 
        if p >= w:  continue
        a = float(np.random.uniform(0.50, 0.80))
        fgs_lo = _step_sample(fgs_lo_range); fgs_hi = _step_sample(fgs_hi_range)
        air_lo = _step_sample(air_lo_range); air_hi = _step_sample(air_hi_range)
        if not (fgs_lo < fgs_hi and air_lo < air_hi): continue
        s = _step_sample(sigma_scale_range)
        sc, sd = _cv_eval_postprocess(
            oof_raw, y_true, w, p, a,
            (fgs_lo, fgs_hi), (air_lo, air_hi), s,
            naive_mean, naive_sigma, n_splits=n_splits, seed=seed
        )
        if sc > best_score:
            best_score = sc
            best.update(window=w, poly=p, alpha=a,
                        clip_fgs=(fgs_lo, fgs_hi),
                        clip_airs=(air_lo, air_hi),
                        sigma_scale=s)
            if verbose:
                print(f"[WIDE-RAND] win={w}, poly={p}, alpha={a}, "
                      f"FGS={best['clip_fgs']}, AIRS={best['clip_airs']}, scale={s} "
                      f"-> GLL={sc:.6f} (±{sd:.6f})")
    # 1.5) 클립만 랜덤
    for _ in range(int(clip_random_trials)):
        f_lo = _step_sample(fgs_lo_range); f_hi = _step_sample(fgs_hi_range)
        a_lo = _step_sample(air_lo_range); a_hi = _step_sample(air_hi_range)
        if not (f_lo < f_hi and a_lo < a_hi): continue
        sc, sd = _cv_eval_postprocess(
            oof_raw, y_true, best["window"], best["poly"], best["alpha"],
            (f_lo, f_hi), (a_lo, a_hi), best["sigma_scale"],
            naive_mean, naive_sigma, n_splits=n_splits, seed=seed
        )
        if sc > best_score:
            best_score = sc; best["clip_fgs"] = (f_lo, f_hi); best["clip_airs"] = (a_lo, a_hi)
            if verbose: print(f"[WIDE-CLIP-RAND] FGS={best['clip_fgs']}, AIRS={best['clip_airs']} -> GLL={sc:.6f} (±{sd:.6f})")
    # 2) 코스 그리드
    def _linspace_with_step(lo, hi, step, max_points=12):
        num = int(np.floor((hi - lo) / step)) + 1
        if num <= max_points: return [lo + i * step for i in range(num)]
        stride = int(np.ceil(num / max_points))
        return [lo + i * step for i in range(0, num, stride)]
    fgs_lo_grid = _linspace_with_step(*fgs_lo_range, max_points=10)
    fgs_hi_grid = _linspace_with_step(*fgs_hi_range, max_points=10)
    air_lo_grid = _linspace_with_step(*air_lo_range, max_points=10)
    air_hi_grid = _linspace_with_step(*air_hi_range, max_points=10)
    tried = 0
    for _ in range(int(grid_max_pairs)):
        f_lo = float(np.random.choice(fgs_lo_grid)); f_hi = float(np.random.choice(fgs_hi_grid))
        a_lo = float(np.random.choice(air_lo_grid)); a_hi = float(np.random.choice(air_hi_grid))
        if not (f_lo < f_hi and a_lo < a_hi): continue
        tried += 1
        sc, sd = _cv_eval_postprocess(
            oof_raw, y_true, best["window"], best["poly"], best["alpha"],
            (f_lo, f_hi), (a_lo, a_hi), best["sigma_scale"],
            naive_mean, naive_sigma, n_splits=n_splits, seed=seed
        )
        if sc > best_score:
            best_score = sc; best["clip_fgs"] = (f_lo, f_hi); best["clip_airs"] = (a_lo, a_hi)
            if verbose: print(f"[WIDE-GRID] FGS={best['clip_fgs']}, AIRS={best['clip_airs']} -> GLL={sc:.6f} (±{sd:.6f})")
    if verbose:
        print("\n[WIDE-BEST] "
              f"win={best['window']}, poly={best['poly']}, alpha={best['alpha']}, "
              f"FGS={best['clip_fgs']}, AIRS={best['clip_airs']}, scale={best['sigma_scale']}, "
              f"GLL={best_score:.6f}")
    return best, best_score

# ----------------------- Main -----------------------
def _nrows(a):
    try: return a.shape[0]
    except Exception: return len(a)

def _take_rows(a, mask):
    import numpy as _np
    if hasattr(a, "iloc"): return a.iloc[mask]
    else: return a[_np.asarray(mask)]

if __name__ == "__main__":
    t0 = time.time()
    if USE_CACHED_XY:
        print("== Load TRAIN (cached X,y) ==")
        X, y, WL_COLS = load_cached_xy_csv(CACHED_TRAIN_XY)  # 자동 탐지
        print(f"[SHAPE] X:{X.shape}, y:{y.shape}")
        arr_y = y.values.astype(np.float32)
        NAIVE_MEAN  = float(np.nanmean(arr_y))
        NAIVE_SIGMA = float(np.nanstd(arr_y) + 1e-12)
    else:
        print("== Build TRAIN features (from images) ==")
        X, y = build_features(split="train", star_info_df=train_star, targets_df=train_df, frac=float(TRAIN_FRAC))
        arr_y = y.values.astype(np.float32)
        NAIVE_MEAN  = float(np.nanmean(arr_y))
        NAIVE_SIGMA = float(np.nanstd(arr_y) + 1e-12)

    # (A) Baseline 학습 → 중요도
    print("== Baseline train (for FI) ==")
    baseline_trained = train_lgbm_cv(X, y)
    fi = get_mean_feature_importance(baseline_trained, kind="gain")

    # (B) Top-K 기반 파생 생성
    TOPK_DERIVE = 50; MEAN_THRESH = 1.0
    X_derived = X.copy()
    X_derived, DERIVED_INFO, DERIVED_COLS = build_derived_from_topk(
        X_derived, fi, topk=TOPK_DERIVE, mean_threshold=MEAN_THRESH, eps=1e-9
    )
    X_derived.replace([np.inf, -np.inf], np.nan, inplace=True); X_derived.fillna(0, inplace=True)
    DERIVED_COLS = filter_similar_derived(
        X_derived, DERIVED_COLS, fi_df=fi, thresh=0.90, max_check=5000
    )
    DERIVED_INFO["new_cols"] = DERIVED_COLS

    # (B-1) (옵션) GMM 증강
    USE_GMM_AUG  = True
    K_CLUSTERS   = 4
    AUG_MULT     = 0.30
    GM_COMPONENTS= 2
    Y_PCA_DIM    = 8
    if USE_GMM_AUG:
        print("== Cluster by volatility & GMM-augment (on derived) ==")
        labels, clust_model, aug = cluster_and_gmm_augment(
            X_derived, y, k_clusters=K_CLUSTERS, aug_mult=AUG_MULT,
            gm_components=GM_COMPONENTS, y_pca_dim=Y_PCA_DIM, random_state=SEED
        )
        X_train, y_train = aug["X_aug"], aug["y_aug"]
    else:
        X_train, y_train = X_derived, y

    # (B-2) 파생(증강) 포함 재학습
    print("== Retrain with derived features ==")
    trained = train_lgbm_cv(X_train, y_train)
    trained["derived_info"] = DERIVED_INFO; trained["derived_cols"] = DERIVED_COLS

    # (C) 후보정(초기값) — 이후 WIDE 튜너로 갱신 예정
    trained = fit_postcalibration(
        trained, y_train.values,
        smooth_window=POST_SMOOTH_WINDOW, smooth_poly=POST_SMOOTH_POLY, blend_alpha=POST_BLEND_ALPHA,
        clip_fgs=SIGMA_CLIP_FGS, clip_airs=SIGMA_CLIP_AIRS, final_sigma_scale=SIGMA_FINAL_SCALE
    )

    # (D) Template 라이브러리 & PCA 서브스페이스 회귀
    print("== Build template library & PCA subspace regressor ==")
    trained["template_lib"] = build_template_library(y_train.values, k_pca=12, n_neighbors=8, seed=SEED)
    trained["pca_pack"]     = train_pca_coefficient_regressor(X_train, y_train.values, k=24, alpha=1.0, seed=SEED)
    trained["max_shift"]    = 3  # AIRS 정렬 허용 시프트

    # (E) 폭넓은 후보정 튜닝 → 최종 보정 반영
    print("== Tune postprocess (WIDE) ==")
    oof_base = trained["oof_predictions"]
    n_oof = _nrows(oof_base)
    orig_n = len(y); aug_n  = len(y_train)
    if n_oof == aug_n:
        y_for_trained = np.asarray(y_train.values if hasattr(y_train, "values") else y_train)
        mask_orig = np.arange(n_oof) < orig_n
    elif n_oof == orig_n:
        y_for_trained = np.asarray(y.values if hasattr(y, "values") else y)
        mask_orig = np.ones(n_oof, dtype=bool)
    else:
        raise ValueError(f"Length mismatch: OOF={n_oof}, y={orig_n}, y_train={aug_n}")

    best_params, best_cv = tune_postprocess_hparams_wide(
        trained, y_for_trained, NAIVE_MEAN, NAIVE_SIGMA,
        n_splits=N_SPLITS, seed=SEED, verbose=True,
        extra_trials=300, clip_random_trials=800, grid_max_pairs=400,
        fgs_lo_range=(0.45, 1.20, 0.05),
        fgs_hi_range=(1.00, 2.00, 0.05),
        air_lo_range=(0.70, 1.05, 0.03),
        air_hi_range=(1.00, 1.70, 0.05),
        sigma_scale_range=(1.50, 3.0, 0.02),
    )

    trained = fit_postcalibration(
        trained, y_for_trained,
        smooth_window=best_params["window"], smooth_poly=best_params["poly"], blend_alpha=best_params["alpha"],
        clip_fgs=best_params["clip_fgs"], clip_airs=best_params["clip_airs"],
        final_sigma_scale=best_params["sigma_scale"],
    )

    # (F) 원본 구간만 OOF 점수
    oof_post = trained["oof_predictions_post"]
    oof_mu_post = _take_rows(oof_post, mask_orig)
    calib_sig = np.asarray(trained["calibrated_uncertainties"]).reshape(-1)
    oof_sigma = np.tile(calib_sig, (mask_orig.sum(), 1))
    oof_score = gll_score_numpy(
        np.asarray(y.values), np.asarray(oof_mu_post), oof_sigma, NAIVE_MEAN, NAIVE_SIGMA, fgs_weight=0.4
    )
    print(f"[OOF/Post tuned] GLL: {oof_score:.6f}")

    # (G) 개선 정도 점검 + 플롯 저장
    mu_oof_improved, _ = predict_with_uncertainty(trained, X_train)
    delta = float(np.mean(np.abs(mu_oof_improved - trained["oof_predictions_post"])))
    print(f"[OOF] improved-vs-post mean|Δ| = {delta:.3e}")

    try:
        plot_oof_lines(y_for_trained, mu_oof_improved, wavelengths, trained["index"], PLOT_DIR, batch_size=12, show=False)
        plot_bias_sigma(trained, wavelengths, PLOT_DIR)
        plot_residual_std_vs_sigma(y_for_trained, trained, wavelengths, PLOT_DIR)
        plot_scatter_and_residuals(y_for_trained, mu_oof_improved, wavelengths, PLOT_DIR, picks=(0,50,150,250), show=False)
        print(f"[PLOTS] saved under {PLOT_DIR}")
    except Exception as e:
        print("[PLOTS] failed:", e)

    # (H) 캐시 저장
    cache_path = f"{WORK_DIR}/fgs1_train_xy.csv"
    save_cached_xy_csv(X_train, y_train, cache_path, wl_cols=WL_COLS)

    print(f"[DONE] total {time.time()-t0:.1f}s")



# ===== Imports (필수) =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# --- 커널 MMD (Gaussian RBF) ---
def _rbf_kernel(X, Y=None, gamma=None):
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    if gamma is None:
        # median heuristic (0 방지)
        D = cdist(X, Y, metric="euclidean")
        D = D[np.isfinite(D) & (D > 0)]
        med = np.median(D) if D.size else 1.0
        gamma = 1.0 / (2.0 * (med**2 + 1e-12))
    return np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

def mmd_rbf(X, Y, gamma=None):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0
    Kxx = _rbf_kernel(X, None, gamma)
    Kyy = _rbf_kernel(Y, None, gamma)
    Kxy = _rbf_kernel(X, Y,  gamma)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n*(n-1) + 1e-12) \
         + (Kyy.sum() - np.trace(Kyy)) / (m*(m-1) + 1e-12) \
         - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0))

# --- 증강 마스크 감지: 인덱스가 'synth_'로 시작하면 증강으로 간주 ---
def _is_aug_mask(df_like, n_all: int = None):
    """
    인덱스가 'synth_'로 시작하는지 보고 증강샘플 마스크를 만든다.
    - df_like: DataFrame/Series/ndarray 모두 허용
    - n_all  : 전체 길이(옵션). None이면 len(df_like)로 추정
    """
    try:
        if hasattr(df_like, "index"):
            idx = pd.Index(getattr(df_like, "index"))
            mask = pd.Series(idx.astype(str), dtype="string") \
                     .str.startswith("synth_") \
                     .fillna(False) \
                     .to_numpy(dtype=bool)
            return mask
    except Exception:
        pass
    if n_all is None:
        try:
            n_all = len(df_like)
        except Exception:
            n_all = 0
    return np.zeros(int(n_all), dtype=bool)

def diagnose_aug(
    X, y, X_train, y_train,
    *, use_y=True, robust=True, whiten=True, max_n=4000, random_state=42
):
    """
    - PCA 2D로 원본/증강 분포를 바로 띄움(plt.show)
    - 원본 vs 증강 분리 AUC, RBF-MMD^2, 최근접거리 히스토그램 표시
    """
    rng = np.random.default_rng(random_state)

    def _to_matrix(A):
        if hasattr(A, "values"): A = A.values
        A = np.asarray(A)
        if A.ndim == 1: A = A.reshape(-1, 1)
        return A.astype(np.float32)

    Y_all_df  = (y_train if use_y else X_train)
    Y_orig_df = (y if use_y else X)

    Y_all  = _to_matrix(Y_all_df)
    Y_orig = _to_matrix(Y_orig_df)

    is_aug = _is_aug_mask(Y_all_df, len(Y_all))
    has_aug = bool(is_aug.any())

    # ----- 스케일 & PCA -----
    scaler = RobustScaler() if robust else StandardScaler()
    Ys = scaler.fit_transform(Y_all)

    n_comp = int(max(2, min(10, Ys.shape[1], Ys.shape[0]-1)))
    pca = PCA(n_components=n_comp, whiten=whiten, random_state=random_state)
    Z = pca.fit_transform(Ys)
    evr = pca.explained_variance_ratio_
    print(f"[PCA] n_comp={pca.n_components_}, EVR top2={evr[:2].round(4)}, cum{n_comp}={evr[:n_comp].sum():.4f}")
    print(f"[diagnose_aug] n_all={len(Y_all)}, n_orig={len(Y_orig)}, n_synth={int(is_aug.sum())}")

    # ----- 2D 시각화 -----
    plt.figure(figsize=(6.8,5.2))
    idx = np.arange(len(Z))
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    if has_aug:
        base = idx[~is_aug[idx]]
        synth = idx[is_aug[idx]]
        plt.scatter(Z[base,0],  Z[base,1],  s=8, alpha=0.65, label="original")
        plt.scatter(Z[synth,0], Z[synth,1], s=10, alpha=0.75, marker='x', label="synthetic")
    else:
        plt.scatter(Z[idx,0], Z[idx,1], s=8, alpha=0.7, label="original (no synthetic)")
    plt.title(f"PCA 2D of {'y' if use_y else 'X'}  — robust={robust}, whiten={whiten}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- 원본/증강 분리 가능도 -----
    if has_aug and np.unique(is_aug.astype(int)).size >= 2:
        k = int(min(20, Z.shape[1]))
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Z[:,:k], is_aug.astype(int))
        auc = roc_auc_score(is_aug.astype(int), clf.predict_proba(Z[:,:k])[:,1])
        print(f"[AUC(original vs synthetic) @ PCA-{k}] = {auc:.3f}")
    else:
        print("[AUC] skipped (no synthetic or one class)")

    # ----- 분포 거리 & 최근접 거리 -----
    if has_aug:
        def _sub(A, n=max_n):
            if len(A) <= n: return A
            return A[rng.choice(len(A), n, replace=False)]
        A = _sub(Ys[~is_aug])
        B = _sub(Ys[is_aug])
        if len(A) > 10 and len(B) > 10:
            mmd = mmd_rbf(A, B)
            mean_dist = float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))
            print(f"[Distance] MMD^2={mmd:.6f} (↓유사), MeanVecDist={mean_dist:.6f}")
        else:
            print("[Distance] skipped (too few samples)")

        if (~is_aug).any():
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(Ys[~is_aug])
            dist, _ = nn.kneighbors(Ys[is_aug])
            plt.figure(figsize=(6.8,4.2))
            plt.hist(dist.ravel(), bins=40, alpha=0.85, density=True)
            plt.title("Nearest-original distance of synthetic samples")
            plt.xlabel("euclidean distance"); plt.ylabel("density")
            plt.tight_layout()
            plt.show()
        else:
            print("[NN] skipped (no original rows)")
    else:
        print("[NN/Distance] skipped (no synthetic)")
    
# ===== 사용 예시 =====
if len(X_train) > len(X):
    diagnose_aug(X, y, X_train, y_train, use_y=True,  robust=True,  whiten=True)
    diagnose_aug(X, y, X_train, y_train, use_y=False, robust=True,  whiten=True)
else:
    print("[diagnose_aug] skipped: no synthetic rows")


# === Visualization ===
try:
    ids_for_plot = trained.get("index", [f"pid_{i}" for i in range(oof_mu_post.shape[0])])

    # 개별 스펙트럼 라인 (배치 저장)
    plot_oof_lines(
        y_true=y.values,
        y_pred=oof_mu_post,
        wavelengths=wavelengths,
        ids=ids_for_plot,
        out_dir=PLOT_DIR,
        batch_size=12,
        show=True
    )

    # 몇 개 λ에서 산포/잔차 분포
    plot_scatter_and_residuals(
        y_true=y.values,
        y_pred=oof_mu_post,
        wavelengths=wavelengths,
        out_dir=PLOT_DIR,
        picks=(0, 50, 150, 250),
        show=True
    )

    # λ별 바이어스/σ/스케일 추적
    plot_bias_sigma(trained, wavelengths, PLOT_DIR)

    # λ별 잔차 robust std와 예측 σ 비교
    plot_residual_std_vs_sigma(y.values, trained, wavelengths, PLOT_DIR)

    print(f"[PLOTS] saved to: {PLOT_DIR}")
except Exception as e:
    print("[PLOTS] failed:", e)


# == Make submission ==
submission_path = f"{WORK_DIR}/submission.csv"  # Kaggle는 /kaggle/working/submission.csv 를 찾음
submission = make_submission(trained, test_star, out_path=submission_path)
print(f"[DONE] saved: {submission_path}  shape={submission.shape}")


submission

