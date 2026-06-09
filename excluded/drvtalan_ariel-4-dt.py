import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
from scipy.signal import medfilt, savgol_filter
from astropy.stats import sigma_clip
import pandas as pd
!pip install --no-index --find-links=/kaggle/input/ariel-2024-pqdm pqdm
from pqdm.threads import pqdm
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


MODEE = 'test/'


def _plot_analysis(signal, analysis_results):
    """
    Helper function to visualize the results of the transit analysis.
    """
    ingress = analysis_results['ingress']
    egress = analysis_results['egress']
    sigma_rel = analysis_results.get('sigma_rel', None)
    debug = analysis_results['debug_info']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 9))

    # Plot data points
    ax.plot(signal, '.', color='gray', alpha=0.6, label='Signal Data')
    ax.plot(debug['initial_oot_indices'],
            signal[debug['initial_oot_indices']],
            'o', color='skyblue', markersize=4,
            label='Initial OOT Points')

    # Plot analysis lines
    ax.axhline(debug['global_baseline'], color='black', linestyle='--',
               label=f'Global Baseline ({debug["global_baseline"]:.4f})')
    ax.axhline(debug['detection_threshold'], color='red', linestyle=':',
               label=f'Detection Threshold ({debug["detection_threshold"]:.4f})')

    # Mark ingress/egress
    if ingress is not None:
        ax.axvline(ingress, color='green', linestyle='-', lw=2,
                   label=f'Ingress: {ingress}')
        ax.plot(ingress, signal[ingress], 'P', color='green',
                markersize=12, markeredgecolor='black')

    if egress is not None:
        ax.axvline(egress, color='purple', linestyle='-', lw=2,
                   label=f'Egress: {egress}')
        ax.plot(egress, signal[egress], 'P', color='purple',
                markersize=12, markeredgecolor='black')

    title = "Robust Transit Detection"
    if sigma_rel is not None and not np.isnan(sigma_rel):
        title += f" | σ_rel = {sigma_rel:.4e}"
    ax.set_title(title, fontsize=18)

    ax.set_xlabel("Time Step / Index", fontsize=12)
    ax.set_ylabel("Normalized Flux", fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def find_transit_points_robust(signal, initial_oot_window_size=60,
                               sigma_threshold=2.0, consecutive_points=3,
                               plot_results=False):
    """
    Finds ingress/egress, computes baseline and relative sigma (uncertainty).
    Returns NaN if sigma not computable.
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # --- 1. Global Stats ---
    min_idx = np.argmin(signal)
    half_window = initial_oot_window_size // 2
    win_start = max(0, min_idx - half_window)
    win_end = min(len(signal)-1, min_idx + half_window)

    oot_indices = np.concatenate([np.arange(0, win_start),
                                  np.arange(win_end, len(signal))])
    oot_indices = oot_indices[oot_indices < len(signal)]  # safety clamp

    if len(oot_indices) < 10:
        global_baseline = np.mean(signal)
        global_sigma = np.std(signal)
    else:
        global_baseline = np.mean(signal[oot_indices])
        global_sigma = np.std(signal[oot_indices])

    detection_threshold = global_baseline - sigma_threshold * global_sigma
    if detection_threshold < np.min(signal):
        detection_threshold = global_baseline

    # --- 2. Ingress & Egress detection ---
    ingress_idx = None
    for i in range(len(signal) - consecutive_points + 1):
        if np.all(signal[i:i+consecutive_points] < detection_threshold):
            ingress_idx = i
            break

    egress_idx = None
    for i in range(len(signal)-1, consecutive_points-2, -1):
        if np.all(signal[i-consecutive_points+1:i+1] < detection_threshold):
            egress_idx = i
            break

    if ingress_idx is not None:
        ingress_idx = min(max(0, ingress_idx), len(signal)-1)
    if egress_idx is not None:
        egress_idx = min(max(0, egress_idx), len(signal)-1)

    # --- 3. Build OOT/IN ---
    oot, inn = None, None
    if ingress_idx is not None and egress_idx is not None and ingress_idx < egress_idx:
        oot = np.concatenate((signal[:ingress_idx], signal[egress_idx:]))
        inn = signal[ingress_idx:egress_idx]

    # --- 4. Compute sigma ---
    sigma_rel = np.nan
    if oot is not None and inn is not None and len(oot) > 1 and len(inn) > 1:
        var_oot = np.nanvar(oot, ddof=1)
        var_in  = np.nanvar(inn, ddof=1)
        n_oot, n_in = len(oot), len(inn)
        oot_mean = np.nanmean(oot)

        if np.isfinite(var_oot) and np.isfinite(var_in) and np.isfinite(oot_mean) and oot_mean > 0:
            sigma_rel = np.sqrt(var_oot/n_oot + var_in/n_in) / oot_mean

    # --- Package results ---
    results = {
        'ingress': ingress_idx,
        'egress': egress_idx,
        'global_baseline': global_baseline,
        'sigma_rel': sigma_rel,
        'debug_info': {
            'min_idx': min_idx,
            'initial_oot_indices': oot_indices,
            'global_baseline': global_baseline,
            'global_sigma': global_sigma,
            'detection_threshold': detection_threshold,
        }
    }

    if plot_results:
        _plot_analysis(signal, results)

    return results



#out folder and axis info
OF =  '/kaggle/input/ariel-data-challenge-2025/'
AI = '/kaggle/input/ariel-data-challenge-2025/axis_info.parquet'

axis_info = pd.read_parquet(AI)


files  = glob.glob(os.path.join(OF,MODEE,'*')) #evrything inside the Folder means subfolder here



idlist = [] #index list
for f in files:
    index = int(f.split('/')[-1])
    idlist.append(index)

idlist.sort()
pids = idlist
pids = np.array(pids).reshape(-1,1)
pids = pd.DataFrame(pids,columns=['planet_id'])
pids.to_csv('planet_ids_info.csv',index=False)


# 2. SET CONFIGURATIONS
PROCESSING_FLAGS = {
    'DO_MASK': True, 'DO_THE_NL_CORR': False, 'DO_DARK': True, 
    'DO_FLAT': True, 'TIME_BINNING': True
}
INSTRUMENT_CONFIGS = {
    'AIRS': {
        'id': 'AIRS-CH0',
        'shape': (32, 356),
        'cut_inf': 39,
        'cut_sup': 321,
        'time_bin_factor': 30,
        'dt_calculator': lambda sig, ax_info: (d := ax_info['AIRS-CH0-integration_time'].dropna().values, d.__setitem__(slice(1, None, 2), d[1::2] + 0.1), d)[-1]
    },
    'FGS': {
        'id': 'FGS1',
        'shape': (32, 32),
        'time_bin_factor': 30 * 12,
        'dt_calculator': lambda sig, ax_info: (d := np.ones(len(sig)) * 0.1, d.__setitem__(slice(1, None, 2), d[1::2] + 0.1), d)[-1]
    }
}



# =================================================================================
# CORRECTED DATA PROCESSING FUNCTIONS
# =================================================================================

def ADC_convert(signal, gain=0.4369, offset=-1000):
    """The Analog-to-Digital Conversion (adc) is performed by the detector to convert
    the pixel voltage into an integer number. Since we are using the same conversion number
    this year, we have simply hard-coded it inside. """
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def apply_linear_corr(linear_corr,clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark(signal, dead, dark, dt):
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal

### --- FIX #1 APPLIED HERE --- ###
def get_cds(signal):
    """
    Performs Correlated Double Sampling on a 3D signal array.
    Shape is assumed to be (time, height, width).
    """
    # Correctly slices along the FIRST (time) axis using 3 indices
    cds = signal[1::2, :, :] - signal[::2, :, :]
    return cds

### --- FIX #2 APPLIED HERE --- ###
def bin_obs(cds_signal, binning):
    """
    Bins a 3D signal array along its first axis (time).
    Shape is assumed to be (frames, height, width).
    """
    n_frames, height, width = cds_signal.shape
    n_binned_frames = n_frames // binning
    # Trim the signal so its length is a multiple of the binning factor
    trimmed_signal = cds_signal[:n_binned_frames * binning]
    # Reshape and average over the new binning axis
    binned_signal = trimmed_signal.reshape(n_binned_frames, binning, height, width).mean(axis=1)
    return binned_signal

### --- FIX #3 APPLIED HERE --- ###
def correct_flat_field(flat, dead, signal):
    """
    Applies flat-field correction.
    This corrected version assumes all inputs use a consistent (height, width) shape.
    """
    flat_masked = np.ma.masked_where(dead, flat)
    flat_tiled = np.tile(flat_masked, (signal.shape[0], 1, 1))
    signal_corrected = signal / flat_tiled
    return signal_corrected


def process_instrument(idx, path_folder, config, flags, axis_info):
    """
    Loads, cleans, and processes all signal files for a specific instrument and index.
    """
    instrument_id = config['id']
    calib_path = os.path.join(path_folder, MODEE , f"{idx}/{instrument_id}_calibration_0")
    cut_slice = slice(config.get('cut_inf'), config.get('cut_sup'))

    # Load calibration files once per call
    flat = pd.read_parquet(os.path.join(calib_path, 'flat.parquet')).values.reshape(config['shape'])[..., cut_slice]
    dark = pd.read_parquet(os.path.join(calib_path, 'dark.parquet')).values.reshape(config['shape'])[..., cut_slice]
    dead = pd.read_parquet(os.path.join(calib_path, 'dead.parquet')).values.reshape(config['shape'])[..., cut_slice]

    signal_files = sorted(glob.glob(os.path.join(path_folder, MODEE ,f"{idx}/{instrument_id}_signal_*.parquet")))
    
    processed_signals = []
    for sig_path in signal_files:
        df = pd.read_parquet(sig_path)
        sig = ADC_convert(df.values.astype(np.float64).reshape((-1,) + config['shape']))
        sig = sig[..., cut_slice]

        if flags['DO_MASK']:
            sig = mask_hot_dead(sig, dead, dark)
        if flags['DO_THE_NL_CORR']:
            linear_corr = pd.read_parquet(os.path.join(calib_path, 'linear_corr.parquet')).values.reshape((6,) + config['shape'])[:, cut_slice]
            sig = apply_linear_corr(linear_corr, sig)
        if flags['DO_DARK']:
            dt = config['dt_calculator'](sig, axis_info)
            sig = clean_dark(sig, dead, dark, dt)

        cds = get_cds(sig)
        if flags['TIME_BINNING']:
            cds = bin_obs(cds, config['time_bin_factor'])
        if flags['DO_FLAT']:
            cds = correct_flat_field(flat, dead, cds)
            
        processed_signals.append(cds)
        
    return processed_signals



import numpy as np
from pqdm.threads import pqdm  # Use pqdm.threads if your task is I/O-bound


def process_single_id(indx):
    """
    Processes a single instrument ID to calculate:
      - White-light transit depth
      - White-light relative sigma (uncertainty)

    Returns a dictionary with keys:
        'white_light_depth'
        'white_light_sigma'
    """
    signal = process_instrument(indx, OF, INSTRUMENT_CONFIGS['AIRS'],
                                PROCESSING_FLAGS, axis_info)
    dc = {}

    # Case A: multiple sensor traces
    if len(signal) < 187:
        temp_depths, temp_sigmas = [], []
        for k in range(len(signal)):
            sig = np.mean(signal[k], axis=1)  # collapse to white-light curve
            sig = np.mean(sig,axis=1)
            results = find_transit_points_robust(sig)
            ing, eg = results['ingress'], results['egress']
            baseline, sigma_rel = results['global_baseline'], results['sigma_rel']

            if ing is not None and eg is not None and ing < eg:
                flat_bottom = np.mean(sig[ing:eg])
                depth = baseline - flat_bottom
                depth /= baseline
            else:
                depth = np.nan

            temp_depths.append(depth)
            temp_sigmas.append(sigma_rel)

        dc['white_light_depth'] = np.nanmean(temp_depths)
        dc['white_light_sigma'] = np.nanmean(temp_sigmas)

    # Case B: already averaged signal
    else:
        sig = np.mean(signal, axis=1)  # collapse to white-light curve
        sig = np.mean(signal, axis=1)  # collapse to white-light curve
        
        results = find_transit_points_robust(sig)
        ing, eg = results['ingress'], results['egress']
        baseline, sigma_rel = results['global_baseline'], results['sigma_rel']

        if ing is not None and eg is not None and ing < eg:
            flat_bottom = np.mean(sig[ing:eg])
            depth = baseline - flat_bottom
            depth /= baseline
        else:
            depth = np.nan

        dc['white_light_depth'] = depth
        dc['white_light_sigma'] = sigma_rel

    return dc


# --- Parallel execution ---
dt = pqdm(idlist[:], process_single_id, n_jobs=4)



dt = pd.DataFrame(dt)
sigma = dt[['white_light_sigma']]
dt = dt.drop(['white_light_sigma'],axis=1)


#note this for TRAIN
info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/test_star_info.csv")
info = info[['Rs','i']]
train = pd.concat([dt,info],axis=1)


train.to_csv('test_model.csv',index=False)
sigma.to_csv('sigma_planets.csv',index=False)


IN = pd.read_csv("/kaggle/working/test_model.csv")
IN = IN.fillna(0)


scaler = joblib.load("/kaggle/input/train-data-check-points-1/scaler_inp.pkl")


IN_scaled = scaler.transform(IN)


IN_scaled


# ---------------------------
# Residual Block
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        # Two fully connected layers, both keep dimension = dim
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        # Batch normalization helps stabilize training
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        # Save input for skip connection
        residual = x
        
        # First linear transformation + ReLU
        out = F.relu(self.bn1(self.fc1(x)))
        # Second linear transformation
        out = self.bn2(self.fc2(out))
        
        # Add skip connection (same dimension: dim)
        out = out + residual
        
        # Final ReLU
        out = F.relu(out)
        return out


# ---------------------------
# Residual MLP Model
# ---------------------------
class ResidualMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1, num_blocks=6):
        super(ResidualMLP, self).__init__()
        
        # Step 1: Project input (3 features) into hidden_dim (128)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        
        # Step 2: Stack of residual blocks, all working in hidden_dim space
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        # Step 3: Projection from hidden_dim (128) to smaller dimension (64)
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Step 4: Final output layer (64 → 1)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # Project 3 → 128
        x = F.relu(self.fc_in(x))
        
        # Pass through residual blocks (still 128)
        x = self.blocks(x)
        
        # Reduce dimension 128 → 64
        x = F.relu(self.fc_mid(x))
        
        # Final prediction 64 → 1
        x = self.fc_out(x)
        return x





# Recreate the same model structure
model = ResidualMLP(input_dim=3, hidden_dim=128, output_dim=1, num_blocks=6)

# Load weights
model.load_state_dict(torch.load("/kaggle/input/train-data-check-points-1/residual_mlp4.pth"))


model = model.double() #float 64


# Switch to evaluation mode
model.eval()



sample =  torch.tensor(IN_scaled,dtype=torch.float64)
with torch.no_grad():
    pred = model(sample)
pred/=100


cols = pred.repeat(1, 283)
temp_df = pd.DataFrame(cols.numpy(), columns=[f"wl_{i}" for i in range(283)])


#handle sigma 
fixed_sigma = 0.0026
sigma_cols = [f'sigma_{j+1}' for j in range(283)]

## 1. Create the Empty DataFrame

import pandas as pd
import numpy as np

# Define the dimensions
num_rows = len(temp_df)
num_cols = 283

# Create the empty DataFrame
sigma_df = pd.DataFrame(index=range(num_rows), columns=sigma_cols)
sigma_df[:] = fixed_sigma


pids = pd.read_csv("/kaggle/working/planet_ids_info.csv")


df = pd.concat([pids,temp_df,sigma_df],axis=1)


df


df.to_csv('submission.csv',index=False) #File create

