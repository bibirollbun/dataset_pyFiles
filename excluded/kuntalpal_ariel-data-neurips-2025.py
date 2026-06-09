!pip install -q batman-package


import random
import time
import logging
from tqdm import tqdm
from pathlib import Path
import pathlib, json, gc, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.ndimage import median_filter
warnings.filterwarnings("ignore")


# ğŸ”�Â point this path to the folder that contains 'train.csv', etc.
DATA_DIR = pathlib.Path("/kaggle/input/ariel-data-challenge-2025/")
WORK_DIR    = pathlib.Path("/kaggle/working")
assert DATA_DIR.exists()


TRAIN_ROOT  = DATA_DIR/"train"
TEST_ROOT   = DATA_DIR/"test"

SAMPLE_COLS = pd.read_csv(DATA_DIR/"sample_submission.csv", nrows=0).columns


wav_df = pd.read_csv(DATA_DIR / "wavelengths.csv", header=None)
wav = wav_df.iloc[1].astype(float).values   # (283,)
adc = pd.read_csv(DATA_DIR / "adc_info.csv", index_col=0)

display(adc)

train_truth = (
    pd.read_csv(DATA_DIR / "train.csv")
      .set_index("planet_id")
      .astype(float)              # every cell is now a plain Python float
)

n_planets    = len(train_truth)
print(f"Number of training planets: {n_planets}")


star_info = pd.read_csv(DATA_DIR/"train_star_info.csv")
test_star_info   = pd.read_csv(DATA_DIR/"test_star_info.csv")
star_info.head()


train_planets = star_info["planet_id"].astype(str).tolist()

# â”€â”€â”€ 1) Seed the RNG for reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
random.seed(42)

# â”€â”€â”€ 2) Pick one planet (always the same) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pid = random.choice(train_planets)
# Or, to avoid randomness altogether, just do:
# pid = train_planets[0]

print("chosen planet :", pid)

# â”€â”€â”€ 3) Point at its data folder and list the first few files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
planet_dir = DATA_DIR / "train" / pid
print(list(planet_dir.glob("*"))[:10])


# The *row label* is the FGS1 offset; the real numbers sit in row 0
gain_fgs   = float(adc.iloc[0]["FGS1_adc_gain"])
offset_fgs = float(adc.index[0])                      # index label itself
gain_air   = float(adc.iloc[0]["AIRS-CH0_adc_gain"])
offset_air = float(adc.iloc[0]["AIRS-CH0_adc_offset"])

print("FGS1  gain / offset :", gain_fgs, offset_fgs)
print("AIRS  gain / offset :", gain_air, offset_air)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AIRS-CH0  (near-infrared spectrograph)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
airs_path = next(planet_dir.glob("AIRS-CH0_signal_*.parquet"))
airs_raw  = pd.read_parquet(airs_path).values.astype("float32")   # (11250, 11392)

airs_phys = airs_raw * gain_air + offset_air                      # counts â†’ electrons
cube_air  = airs_phys.reshape(-1, 32, 356)                        # (time, row, Î»)

print("AIRS cube:", cube_air.shape)                               # sanity: (11250, 32, 356)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FGS1  (optical broadband photometer)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fgs_path = next(planet_dir.glob("FGS1_signal_*.parquet"))
fgs_raw  = pd.read_parquet(fgs_path).values.astype("float32")     # (135000, 1024)

fgs_phys = fgs_raw * gain_fgs + offset_fgs
cube_fgs = fgs_phys.reshape(-1, 32, 32)                           # (time, row, col)

print("FGS cube :", cube_fgs.shape)                               # sanity: (135000, 32, 32)


import pyarrow.parquet as pq

def first_file(glob_pattern):
    """Return the first file that matches the given pattern."""
    files = list(DATA_DIR.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files match {glob_pattern}")
    return files[0]

# Grab one AIRS and one FGS Parquet file (any planet, any visit)
airs_file = first_file("train/*/AIRS-CH0_signal_*.parquet")
fgs_file  = first_file("train/*/FGS1_signal_*.parquet")

def parquet_shape(path):
    """Return (#rows, #columns) without loading the whole file."""
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, pf.metadata.num_columns

airs_rows, airs_cols = parquet_shape(airs_file)
fgs_rows,  fgs_cols  = parquet_shape(fgs_file)

# Derive the 2-D geometry that the columns flatten into
airs_geometry = f"32 Ã— {airs_cols // 32}"           # 32 spatial rows
fgs_side      = int(fgs_cols ** 0.5)                # expect 32 for 1 024
fgs_geometry  = f"{fgs_side} Ã— {fgs_side}"

summary = pd.DataFrame({
    "Instrument":          ["AIRS-CH0", "FGS1"],
    "Frames (rows)":       [airs_rows,   fgs_rows],
    "Flattened columns":   [airs_cols,   fgs_cols],
    "Un-flattened geometry": [airs_geometry, fgs_geometry],
    "Cadence":             ["0.8 s", "0.1 s"]  # fixed by instrument design
})

summary


# ---------- ORIGINAL DATA ----------
white_light = cube_fgs.sum(axis=(1, 2))          # integrate all pixels
time_s      = np.arange(len(white_light)) * 0.1  # cadence 0.1 s

# ---------- 1) BIN TO 2-SECOND CADENCE ----------
bin_size   = 20                # 20 Ã— 0.1 s  = 2 s
n_bins     = len(white_light) // bin_size
wl_binned  = white_light[: n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
time_b     = time_s[: n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)

# ---------- 2) RUNNING MEDIAN (Â±30 s window) ----------
wl_med = median_filter(wl_binned, size=15)       # 15 Ã— 2 s â‰ˆ 30 s

# ---------- 3) NORMALISE ----------
wl_norm = wl_binned / wl_binned.mean()
wl_med  = wl_med    / wl_binned.mean()

# ---------- 4) PLOT ----------
plt.figure(figsize=(11, 3.5))

# lightly-coloured line for every 2-s bin
plt.plot(time_b / 60, wl_norm,
         lw=0.6, alpha=0.3, label="2 s bins")

# bold line for the running median
plt.plot(time_b / 60, wl_med,
         lw=1.2, label="30 s running median")

# cosmetic tweaks
plt.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
plt.xlabel("Time [minutes]")
plt.ylabel("Relative flux")
plt.title(f"FGS1 white-light curve  (planet {pid})")
plt.ylim(0.985, 1.015)          # tight y-range shows ~1â€“2 % variations
plt.legend(frameon=False)
plt.tight_layout()
plt.show()



# â”€â”€â”€ 1) Cache all calibration maps once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CAL_DIR     = None
_DARK        = {}
_FLAT        = {}
_DEAD_MASK   = {}
_LIN_COEFFS  = {}
_READ_NOISE  = {}

def load_all_calibrations(sample_root: Path, instr="AIRS-CH0"):
    global _CAL_DIR
    # pick first planet folder
    sample_planet = next(p for p in sample_root.iterdir() if p.is_dir())
    cal_dir = sample_planet / f"{instr}_calibration_0"
    _CAL_DIR = cal_dir

    _DARK[instr]      = pd.read_parquet(cal_dir/"dark.parquet").to_numpy(np.float32).reshape(32,356)
    _FLAT[instr]      = pd.read_parquet(cal_dir/"flat.parquet").to_numpy(np.float32).reshape(32,356)
    _DEAD_MASK[instr] = pd.read_parquet(cal_dir/"dead.parquet").to_numpy().astype(bool).reshape(32,356)
    lin = pd.read_parquet(cal_dir/"linear_corr.parquet").to_numpy(np.float32)
    _LIN_COEFFS[instr] = lin.reshape(-1,32,356)[::-1]    # reverse for Hornerâ€™s
    _READ_NOISE[instr] = pd.read_parquet(cal_dir/"read.parquet").to_numpy(np.float32).reshape(32,356)

load_all_calibrations(TRAIN_ROOT, instr="AIRS-CH0")

# â”€â”€â”€ 2) Calibrate raw cube â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def calibrate_cube(planet_dir: Path, instr="AIRS-CH0"):
    """
    1) read raw â†’ apply ADC
    2) subtract dark, divide flat
    3) mask dead pixels
    4) Hornerâ€™s nonlinearity correction
    5) compute variance = shot + read_noise^2
    returns cube (n_frames,32,356) and var (same)
    """
    sig = next(planet_dir.glob(f"{instr}_signal_*.parquet"))
    raw = pd.read_parquet(sig).to_numpy(np.float32)
    arr = raw * gain_air + offset_air
    cube = arr.reshape(-1,32,356)

    # dark & flat
    cube -= _DARK[instr][None]
    safe_flat = np.where(_FLAT[instr]==0, np.nan, _FLAT[instr])
    cube /= safe_flat[None]

    # mask
    cube[:, _DEAD_MASK[instr]] = np.nan

    # nonlinearity via Hornerâ€™s
    corr = np.zeros_like(cube)
    x = cube
    for coeff in _LIN_COEFFS[instr]:
        corr = corr * x + coeff[None]
    cube = corr

    # variance
    var = np.abs(cube) + _READ_NOISE[instr][None]**2

    return cube, var

# â”€â”€â”€ 3) Optimal white-light extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def extract_white_light(cube: np.ndarray, var: np.ndarray):
    """
    PSF = median over time â†’ normalize â†’ optimal extraction:
    flux = sum( PSF * cube/var ) / sum( PSF^2/var )
    sigma = 1/sqrt(denominator)
    returns time_s (0.8s cadence), flux, sigma
    """
    psf = np.nanmedian(cube, axis=0)
    psf /= np.nansum(psf)

    num = np.nansum(psf[None,:,:] * cube / var, axis=(1,2))
    den = np.nansum(psf[None,:,:]**2 / var, axis=(1,2))

    flux  = num/den
    sigma = 1.0/np.sqrt(den)
    time_s = np.arange(cube.shape[0]) * 0.8
    return time_s, flux, sigma

# â”€â”€â”€ 4) Correlated-double sampling & binning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_cds(signal: np.ndarray) -> np.ndarray:
    """(end - start) reading pairs"""
    return signal[1::2] - signal[::2]

def bin_obs(cds_signal: np.ndarray, binning: int) -> np.ndarray:
    """sum every `binning` ramps into one frame"""
    n = cds_signal.shape[0] // binning
    b = np.zeros((n, *cds_signal.shape[1:]), dtype=cds_signal.dtype)
    for i in range(n):
        b[i] = np.nansum(cds_signal[i*binning:(i+1)*binning], axis=0)
    return b

# â”€â”€â”€ 5) Plot 10 random train white-light curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def plot_sample_train_curves(n_samples=20, binning=30):
    random.seed(0)
    planets = random.sample([d for d in TRAIN_ROOT.iterdir() if d.is_dir()], n_samples)

    plt.figure(figsize=(8,5))
    for pd_dir in tqdm(planets, desc="plotting"):
        cube, var        = calibrate_cube(pd_dir)
        time_s, flux_wl, _ = extract_white_light(cube, var)

        # build CDS+binned cube to sum pixels
        cds   = get_cds(cube)           # (n_ramps,32,356)
        binned= bin_obs(cds, binning)   # (n_bins,32,356)

        # white-light = sum over all pixels in each binned frame
        wl_curve = binned.sum(axis=(1,2))

        # normalize to first+last 10% OOT baseline
        n = len(wl_curve)
        oot = np.concatenate([wl_curve[:n//10], wl_curve[-n//10:]])
        baseline = np.nanmedian(oot)
        wl_norm  = wl_curve / baseline

        frames = np.arange(len(wl_norm))
        plt.plot(frames, wl_norm, alpha=0.5, lw=1)

    plt.xlabel("Time (binned frame index)")
    plt.ylabel("Normalized flux")
    plt.title(f"White-light curves from {n_samples} random training planets")
    plt.ylim(0.9, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# â”€â”€â”€ Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plot_sample_train_curves(n_samples=100, binning=30)


plt.figure(figsize=(8,3))
plt.imshow(cube_air[0], aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="eâ€‘ counts")
plt.xlabel("Dispersion (Î» bins)"); plt.ylabel("Spatial rows")
plt.title("AIRSâ€‘CH0 raw detector frame")
plt.show()



# â”€â”€â”€ Select 3 reproducible planets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
valid_planets = train_truth.index.intersection(star_info["planet_id"].astype(int))
random.seed(80)
pids = random.sample(list(valid_planets), 3)
print("Chosen planets:", pids)

# â”€â”€â”€ Prepare molecular features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
features = {
    2.7:  "Hâ‚‚O",
    2.97: "NHâ‚ƒ",
    3.3:  "CHâ‚„",
    3.4:  "Câ‚‚Hâ‚‚",
    3.5:  "HCN",
}

# â”€â”€â”€ Plot each spectrum in its own subplot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for ax, pi in zip(axes, pids):
    # Extract AIRS spectrum (skip bin 0)
    spec      = train_truth.loc[pi].values
    wav_airs  = wav[1:]
    depth_ppm = spec[1:] * 1e6

    # Plot spectrum and noise floor
    ax.plot(wav_airs, depth_ppm, "-o", ms=4, lw=1,
            color="navy", label="True spectrum")
    ax.fill_between(wav_airs,
                    depth_ppm - 50,
                    depth_ppm + 50,
                    color="gray", alpha=0.1,
                    label="Â±50 ppm noise")

    # Formatting
    ax.set_ylabel("Transit depth [ppm]", fontsize=12)
    ax.set_title(f"Planet {pid}", fontsize=14)
    ax.grid(alpha=0.3)

    # Mixed transform: x in data, y in axes fraction
    trans = ax.get_xaxis_transform()

    # Place labels a bit to the right of each dotted line
    for wl0, label in features.items():
        if wav_airs.min() < wl0 < wav_airs.max():
            # draw the dotted line
            ax.axvline(wl0,
                       color="firebrick", linestyle="--", alpha=0.7,
                       label="_nolegend_")
            # determine text position offset
            x_text = wl0 + 0.03          # shift right by 0.03 Î¼m
            # if near right edge, shift left instead
            if x_text > wav_airs.max():
                x_text = wl0 - 0.03
                ha = "right"
            else:
                ha = "left"
            # place text just inside the top (y=0.98)
            ax.text(x_text, 0.4,
                    f"{label} ({wl0:.2f} Âµm)",
                    color="firebrick",
                    fontsize=10,
                    rotation=90,
                    ha=ha, va="top",
                    transform=trans,
                    clip_on=True,
                    label="_nolegend_")

# â”€â”€â”€ Final formatting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
axes[-1].set_xlabel("Wavelength [Âµm]", fontsize=12)
axes[0].legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()


# â”€â”€â”€ 5) Spectral â€œboxâ€� extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def box_extract(cube: np.ndarray, center: int = 16, half: int = 3) -> np.ndarray:
    """
    Sum Â±half rows around `center` in the 32Ã—356 cube â†’ shape (n_frames, 356).
    """
    r0, r1 = center-half, center+half+1
    return cube[:, r0:r1, :].sum(axis=1)

# â”€â”€â”€ 6) Commonâ€�mode removal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def remove_common_mode(spec_cube: np.ndarray) -> np.ndarray:
    """
    spec_cube: (n_frames, Î»_bins)
    divide out the commonâ€�mode (perâ€�frame median) â†’ same shape
    """
    cm = np.nanmedian(spec_cube, axis=1)
    return spec_cube / cm[:, None]

# â”€â”€â”€ 7) Phaseâ€�fold & masks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def phase_fold_masks(time_s: np.ndarray, pid: int, dur_phase: float = 0.02):
    """
    time_s [s] â†’ days; look up P,t0 for this planet â†’ phase in (â€“0.5â€¦+0.5)
    returns (phase, in_mask, out_mask)
    """
    time_d = time_s / 86400.0
    row    = star_info.query("planet_id==@pid").iloc[0]
    P      = float(row["P"])
    t0     = float(row.get("t0", 0.5*P))
    phase  = ((time_d - t0 + 0.5*P) % P)/P - 0.5

    in_m = np.abs(phase) < dur_phase
    if not in_m.any():  # widen if no points
        in_m = np.abs(phase) < (dur_phase*2)
    return phase, in_m, ~in_m

# â”€â”€â”€ 8) Depth & Ïƒ measurement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def measure_depths(spec_cube, flux_wl, in_mask, out_mask, eps=1e-8):
    """
    Compute transit depth channel-by-channel as 1 - (mean in / mean out).
    Adds a tiny eps when dividing to avoid NaNs.
    """
    # make sure flux_wl never zero
    flux_safe = flux_wl.copy()
    flux_safe[np.abs(flux_safe) < eps] = np.nanmedian(flux_safe)

    # relative spectrum
    rel_cube = spec_cube / flux_safe[:, None]

    # compute in/out means
    mu_in  = np.nanmean(rel_cube[in_mask],  axis=0)
    mu_out = np.nanmean(rel_cube[out_mask], axis=0)

    # avoid zero out:
    mu_out_safe = mu_out.copy()
    mu_out_safe[np.abs(mu_out_safe) < eps] = eps

    depth = 1.0 - mu_in / mu_out_safe
    depth = np.clip(depth, 0.0, 1.0)

    sigma = np.nanstd(rel_cube[in_mask], axis=0) / np.sqrt(np.sum(in_mask))

    return depth, sigma

def build_submission(default_sigma=2e-3):
    sample = pd.read_csv(DATA_DIR/"sample_submission.csv", nrows=0).columns
    rows   = []
    star_info = pd.read_csv(DATA_DIR/"test_star_info.csv")

    for d in tqdm(sorted(TEST_ROOT.iterdir()), desc="planets"):
        pid = int(d.name)

        # 1) CALIBRATE & EXTRACT WHITE LIGHT
        cube, var        = calibrate_cube(d)
        t_s, flux_wl, _  = extract_white_light(cube, var)

        # 2) SPECTRAL BOX & NORMALIZE
        spec_cube = box_extract(cube)
        spec_rel  = spec_cube / flux_wl[:, None]

        # 3) COMMONâ€�MODE CORRECTION
        corr_cube = remove_common_mode(spec_rel)

        # 4) ORBITAL PARAMETERS
        rowp = star_info[star_info["planet_id"] == pid].iloc[0]
        P   = float(rowp["P"])
        dur = float(rowp.get("dur", 0.05 * P))
        t0  = float(rowp.get("t0", 0.5 * P))

        # 5) PHASE
        time_d = t_s / 86400.0
        phase  = ((time_d - t0 + 0.5*P) % P) / P - 0.5

        # 6) IN/OUT MASKS WITH FALLBACK
        half_w = 0.5 * dur / P
        in_m   = np.abs(phase) < half_w
        if in_m.sum() == 0:
            n = len(phase)
            lo, hi = int(0.35*n), int(0.65*n)
            in_m = np.zeros(n, bool)
            in_m[lo:hi] = True
        out_m = ~in_m

        # 7) MEASURE DEPTHS & UNCERTAINTIES
        depths, sigmas = measure_depths(corr_cube, phase, in_m, out_m)

        # 8) ASSEMBLE ROW WITH SANITYâ€�CHECK
        row = {"planet_id": pid, "wl_1": 0.0, "sigma_1": default_sigma}
        for idx, (Î´, Ïƒ) in enumerate(zip(depths, sigmas), start=2):
            # if Î´ is NaN or negative, set to zero
            if not np.isfinite(Î´) or Î´ < 0:
                Î´ = 0.0
            # if Ïƒ is NaN or non-positive, set to default_sigma
            if not np.isfinite(Ïƒ) or Ïƒ <= 0:
                Ïƒ = default_sigma

            row[f"wl_{idx}"]    = float(Î´)
            row[f"sigma_{idx}"] = float(Ïƒ)

        rows.append(row)

    submission = pd.DataFrame(rows)[sample]
    submission.to_csv(WORK_DIR/"submission.csv", index=False)
    print("âœ… Written submission.csv, shape:", submission.shape)

# â”€â”€â”€ run it! â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_submission()

