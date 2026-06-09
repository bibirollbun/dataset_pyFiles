import numpy as np
import pandas as pd
import os
import pickle
import scipy.stats as stats
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import glob 
from astropy.stats import sigma_clip
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load
from scipy.optimize import minimize

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import resample
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C


ariel_path = "/kaggle/input/ariel-data-challenge-2025/"
all_files = os.listdir(ariel_path)
path_out = "/kaggle/tmp/light_data_raw/"
output_dir = "/kaggle/tmp/light_data_raw/"

#training files
train_folder = "/kaggle/input/ariel-data-challenge-2025/train"
sample_path = "/kaggle/input/ariel-data-challenge-2025/sample_submission.csv"
train_df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train.csv")
train_star_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train_star_info.csv")

#test files
test_folder = "/kaggle/input/ariel-data-challenge-2025/test/"
test_file = os.listdir(test_folder)[0]
test_df = pd.read_csv(sample_path)

test_star_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/test_star_info.csv")

## other files
axis_info_df = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")
wavelengths_df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/wavelengths.csv")
adc_info_df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")


if not os.path.exists(path_out):
    os.makedirs(path_out)
    print(f"Directory {path_out} created.")
else:
    print(f"Directory {path_out} already exists.")


CHUNKS_SIZE=1


#Functions for preprocessing
# restore the original dynamic range of the data as outlined in the competition
def ADC_convert(signal, gain=0.4369, offset=-1000):
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(
        dark, sigma=5, maxiters=5
    ).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)

    return signal

#linearity correction
def apply_linear_corr(linear_corr, clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)

    for x, y in itertools.product(
        range(clean_signal.shape[1]), range(clean_signal.shape[2])
    ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[: x, y] = poli(clean_signal[:, x, y])

    return clean_signal

#dark current subtraction
def clean_dark(signal, dead, dark, dt):
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark*dt[:, np.newaxis, np.newaxis]

    return signal

#correlated double sampling
def get_cds(signal):
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]

    return cds

#time binning (observations binned together by frequency)
def bin_obs(cds_signal, binning):
    cds_transposed = cds_signal.transpose(0,1,3,2)
    cds_binned = np.zeros((cds_transposed.shape[0], 
                           cds_transposed.shape[1]//binning, 
                           cds_transposed.shape[2], 
                           cds_transposed.shape[3]))

    for i in range(cds_transposed.shape[1]//binning):
        cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:],
                                    axis=1)

    return cds_binned

#flat field correction
def correct_flat_field(flat, dead, signal):
    flat = flat.transpose(1,0)
    dead = dead.transpose(1,0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat

    return signal

#get index for training data
def get_index(files, CHUNKS_SIZE):
    index = []

    for file in files:
        file_name = file.split('/')[-1]

        if file_name.split('_')[0] == 'AIRS-CH0' and file_name.split('_')[1] == 'signal' and file_name.split('_')[2] == '0.parquet':
            file_index = os.path.basename(os.path.dirname(file))
            index.append(int(file_index))

    index = np.array(index)
    index = np.sort(index)

    index = np.array_split(index, len(index)//CHUNKS_SIZE)

    return index    


# files = glob.glob(os.path.join(ariel_path + 'train/', '*/*'))
# index = get_index(files, CHUNKS_SIZE)

# axis_info = pd.read_parquet(os.path.join(ariel_path, 'axis_info.parquet'))
# DO_MASK = True
# DO_THE_NL_CORR = False
# DO_DARK = True
# DO_FLAT = True
# TIME_BINNING = True

# cut_inf, cut_sup = 39, 321
# l = cut_sup - cut_inf

# for n, index_chunk in enumerate(tqdm(index)):
#     AIRS_CH0_clean = np.zeros((CHUNKS_SIZE, 11250, 32, l))
#     FGS1_clean = np.zeros((CHUNKS_SIZE, 135000, 32, 32))

#     for i in range(CHUNKS_SIZE):
#         df = pd.read_parquet(os.path.join(ariel_path, f'train/{index_chunk[i]}/AIRS-CH0_signal_0.parquet'))
#         signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
#         signal = ADC_convert(signal,)

#         dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
#         dt_airs[1::2] += 0.1
#         chopped_signal = signal[:, :, cut_inf:cut_sup]
        
#         del signal, df

#         flat = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
#         dark = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
#         dead_airs = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
#         linear_corr = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]

#         if DO_MASK:
#             chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
#             AIRS_CH0_clean[i] = chopped_signal
#         else:
#             AIRS_CH0_clean[i] = chopped_signal
            
#         if DO_THE_NL_CORR: 
#             linear_corr_signal = apply_linear_corr(linear_corr,AIRS_CH0_clean[i])
#             AIRS_CH0_clean[i,:, :, :] = linear_corr_signal
#         del linear_corr

#         if DO_DARK: 
#             cleaned_signal = clean_dark(AIRS_CH0_clean[i], dead_airs, dark, dt_airs)
#             AIRS_CH0_clean[i] = cleaned_signal
#         else: 
#             pass
#         del dark

#         df = pd.read_parquet(os.path.join(ariel_path, f'train/{index_chunk[i]}/FGS1_signal_0.parquet'))
#         fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))

#         fgs_signal = ADC_convert(fgs_signal, )
#         dt_fgs1 = np.ones(len(fgs_signal))*0.1
#         dt_fgs1[1::2] += 0.1
#         chopped_FGS1 = fgs_signal

#         del fgs_signal, df

# # cleaning fgs1
#         flat = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
#         dark = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/FGS1_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 32))
#         dead_fgs1 = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/FGS1_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 32))
#         linear_corr = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/FGS1_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))
        
#         if DO_MASK:
#             chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
#             FGS1_clean[i] = chopped_FGS1
#         else:
#             FGS1_clean[i] = chopped_FGS1

#         if DO_THE_NL_CORR: 
#             linear_corr_signal = apply_linear_corr(linear_corr,FGS1_clean[i])
#             FGS1_clean[i,:, :, :] = linear_corr_signal
#         del linear_corr
        
#         if DO_DARK: 
#             cleaned_signal = clean_dark(FGS1_clean[i], dead_fgs1, dark,dt_fgs1)
#             FGS1_clean[i] = cleaned_signal
#         else: 
#             pass
#         del dark
        
#     AIRS_cds = get_cds(AIRS_CH0_clean)
#     FGS1_cds = get_cds(FGS1_clean)
    
#     del AIRS_CH0_clean, FGS1_clean
    
#     if TIME_BINNING:
#         AIRS_cds_binned = bin_obs(AIRS_cds,binning=30)
#         FGS1_cds_binned = bin_obs(FGS1_cds,binning=30*12)
#     else:
#         AIRS_cds = AIRS_cds.transpose(0,1,3,2) ## this is important to make it consistent for flat fielding, but you can always change it
#         AIRS_cds_binned = AIRS_cds
#         FGS1_cds = FGS1_cds.transpose(0,1,3,2)
#         FGS1_cds_binned = FGS1_cds
    
#     del AIRS_cds, FGS1_cds
    
#     for i in range (CHUNKS_SIZE):
#         flat_airs = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
#         flat_fgs = pd.read_parquet(os.path.join(ariel_path,f'train/{index_chunk[i]}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
#         if DO_FLAT:
#             corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[i])
#             AIRS_cds_binned[i] = corrected_AIRS_cds_binned
#             corrected_FGS1_cds_binned = correct_flat_field(flat_fgs,dead_fgs1, FGS1_cds_binned[i])
#             FGS1_cds_binned[i] = corrected_FGS1_cds_binned
#         else:
#             pass

#     np.save(os.path.join(path_out, 'AIRS_clean_train_{}.npy'.format(n)), AIRS_cds_binned)
#     np.save(os.path.join(path_out, 'FGS1_train_{}.npy'.format(n)), FGS1_cds_binned)
#     del AIRS_cds_binned
#     del FGS1_cds_binned


#test files
files = glob.glob(os.path.join(ariel_path + 'test/', '*/*'))
index = get_index(files, CHUNKS_SIZE)

axis_info = pd.read_parquet(os.path.join(ariel_path, 'axis_info.parquet'))
DO_MASK = True
DO_THE_NL_CORR = False
DO_DARK = True
DO_FLAT = True
TIME_BINNING = True

cut_inf, cut_sup = 39, 321
l = cut_sup - cut_inf

for n, index_chunk in enumerate(tqdm(index)):
    AIRS_CH0_clean = np.zeros((CHUNKS_SIZE, 11250, 32, l))
    FGS1_clean = np.zeros((CHUNKS_SIZE, 135000, 32, 32))

    for i in range(CHUNKS_SIZE):
        df = pd.read_parquet(os.path.join(ariel_path, f'test/{index_chunk[i]}/AIRS-CH0_signal_0.parquet'))
        signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
        signal = ADC_convert(signal,)

        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1
        chopped_signal = signal[:, :, cut_inf:cut_sup]
        
        del signal, df

        flat = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dark = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dead_airs = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/AIRS-CH0_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        linear_corr = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/AIRS-CH0_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]

        if DO_MASK:
            chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
            AIRS_CH0_clean[i] = chopped_signal
        else:
            AIRS_CH0_clean[i] = chopped_signal
            
        if DO_THE_NL_CORR: 
            linear_corr_signal = apply_linear_corr(linear_corr,AIRS_CH0_clean[i])
            AIRS_CH0_clean[i,:, :, :] = linear_corr_signal
        del linear_corr

        if DO_DARK: 
            cleaned_signal = clean_dark(AIRS_CH0_clean[i], dead_airs, dark, dt_airs)
            AIRS_CH0_clean[i] = cleaned_signal
        else: 
            pass
        del dark

        df = pd.read_parquet(os.path.join(ariel_path, f'test/{index_chunk[i]}/FGS1_signal_0.parquet'))
        fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))

        fgs_signal = ADC_convert(fgs_signal, )
        dt_fgs1 = np.ones(len(fgs_signal))*0.1
        dt_fgs1[1::2] += 0.1
        chopped_FGS1 = fgs_signal

        del fgs_signal, df

# cleaning fgs1
        flat = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        dark = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/FGS1_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 32))
        dead_fgs1 = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/FGS1_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 32))
        linear_corr = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/FGS1_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))
        
        if DO_MASK:
            chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
            FGS1_clean[i] = chopped_FGS1
        else:
            FGS1_clean[i] = chopped_FGS1

        if DO_THE_NL_CORR: 
            linear_corr_signal = apply_linear_corr(linear_corr,FGS1_clean[i])
            FGS1_clean[i,:, :, :] = linear_corr_signal
        del linear_corr
        
        if DO_DARK: 
            cleaned_signal = clean_dark(FGS1_clean[i], dead_fgs1, dark,dt_fgs1)
            FGS1_clean[i] = cleaned_signal
        else: 
            pass
        del dark
        
    AIRS_cds = get_cds(AIRS_CH0_clean)
    FGS1_cds = get_cds(FGS1_clean)
    
    del AIRS_CH0_clean, FGS1_clean
    
    if TIME_BINNING:
        AIRS_cds_binned = bin_obs(AIRS_cds,binning=30)
        FGS1_cds_binned = bin_obs(FGS1_cds,binning=30*12)
    else:
        AIRS_cds = AIRS_cds.transpose(0,1,3,2) ## this is important to make it consistent for flat fielding, but you can always change it
        AIRS_cds_binned = AIRS_cds
        FGS1_cds = FGS1_cds.transpose(0,1,3,2)
        FGS1_cds_binned = FGS1_cds
    
    del AIRS_cds, FGS1_cds
    
    for i in range (CHUNKS_SIZE):
        flat_airs = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        flat_fgs = pd.read_parquet(os.path.join(ariel_path,f'test/{index_chunk[i]}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        if DO_FLAT:
            corrected_AIRS_cds_binned = correct_flat_field(flat_airs,dead_airs, AIRS_cds_binned[i])
            AIRS_cds_binned[i] = corrected_AIRS_cds_binned
            corrected_FGS1_cds_binned = correct_flat_field(flat_fgs,dead_fgs1, FGS1_cds_binned[i])
            FGS1_cds_binned[i] = corrected_FGS1_cds_binned
        else:
            pass

    #test files
    np.save(os.path.join("/kaggle/working/", 'AIRS_clean_test_{}.npy'.format(n)), AIRS_cds_binned)
    np.save(os.path.join("/kaggle/working/", 'FGS1_test_{}.npy'.format(n)), FGS1_cds_binned)
    del AIRS_cds_binned
    del FGS1_cds_binned


def load_data(file, chunk_size, nb_files):
    data0 = np.load(file + '_0.npy')
    data_all = np.zeros((nb_files*chunk_size,
                        data0.shape[1], data0.shape[2], 
                        data0.shape[3]))
    data_all[:chunk_size] = data0
    
    for i in range (1, nb_files):
        data_all[i*chunk_size:(i+1)*chunk_size] = np.load(file + '_{}.npy'.format(i))
    
    return data_all

#train files
# data_train = load_data(path_out + 'AIRS_clean_train', CHUNKS_SIZE, len(index))
# data_train_FGS = load_data(path_out + 'FGS1_train', CHUNKS_SIZE, len(index))

#test files
data_test = load_data("/kaggle/working/" + 'AIRS_clean_test', CHUNKS_SIZE, len(index))
data_test_FGS = load_data("/kaggle/working/" + 'FGS1_test', CHUNKS_SIZE, len(index))


# np.save('./' + 'data_train.npy', data_train)
# np.save('./' + 'data_train_FGS.npy', data_train_FGS)


# data_train = np.load("/kaggle/input/adc-preprocessed-data/data_train (1).npy")
# data_train_FGS = np.load("/kaggle/input/adc-preprocessed-data/data_train_FGS.npy")


planet_test_ids = [int(planet_id[0]) for planet_id in index]


def extract_physical_features(signal: np.ndarray, dt: float = 1.0):
    """
    extract physically motivated features from a detector time series cube (T,H,W).
    """
    T, H, W = signal.shape
    features = {}

    # total flux
    F_tot = signal.sum(axis=(1, 2))
    features["flux_mean"] = F_tot.mean()
    features["flux_std"] = F_tot.std()
    features["flux_skew"] = skew(F_tot)
    features["flux_kurt"] = kurtosis(F_tot)

    # centroid
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    X, Y = np.meshgrid(x_coords, y_coords)

    x_c = (signal * X[None, :, :]).sum(axis=(1, 2)) / F_tot
    y_c = (signal * Y[None, :, :]).sum(axis=(1, 2)) / F_tot
    features["centroid_x_std"] = x_c.std()
    features["centroid_y_std"] = y_c.std()

    # centroid scatter skew/kurt
    features["centroid_x_skew"] = skew(x_c)
    features["centroid_y_skew"] = skew(y_c)

    # background mean (outer rows/cols)
    outer_rows = np.r_[0, 1, H-2, H-1]
    outer_cols = np.r_[0, 1, W-2, W-1]
    background = np.concatenate([
        signal[:, outer_rows, :].reshape(T, -1),
        signal[:, :, outer_cols].reshape(T, -1)
    ], axis=1)
    features["background_mean"] = background.mean()
    features["background_std"] = background.std()

    # spatial RMS
    spatial_rms = np.sqrt(np.mean(
        (signal - signal.mean(axis=(1, 2), keepdims=True)) ** 2,
        axis=(1, 2)
    ))
    features["spatial_rms_mean"] = spatial_rms.mean()
    features["spatial_rms_std"] = spatial_rms.std()

    # flux derivative
    flux_deriv = np.diff(F_tot) / dt
    features["flux_deriv_mean"] = flux_deriv.mean()
    features["flux_deriv_std"] = flux_deriv.std()

    return features


def extract_frequency_features(signal: np.ndarray):
    """
    extract frequency domain features using FFT of total flux.
    """
    features = {}
    F_tot = signal.sum(axis=(1, 2))
    fft_vals = np.fft.fft(F_tot)
    fft_mag = np.abs(fft_vals[:len(fft_vals)//2])  # keep positive freqs
    freqs = np.fft.fftfreq(len(F_tot))[:len(fft_vals)//2]

    # dominant frequency
    dom_idx = np.argmax(fft_mag)
    features["dom_freq"] = freqs[dom_idx]
    features["dom_amp"] = fft_mag[dom_idx]

    # spectral centroid
    power = fft_mag ** 2
    if power.sum() > 0:
        features["spec_centroid"] = (freqs * power).sum() / power.sum()
        features["spec_spread"] = np.sqrt(((freqs - features["spec_centroid"]) ** 2 * power).sum() / power.sum())
    else:
        features["spec_centroid"] = 0
        features["spec_spread"] = 0

    return features



#all features utilizing above functions 

def extract_all_features(data_airs, data_fgs, planet_ids, star_info, spectra_df):
    all_features = []
    all_labels = []

    for i, planet_id in enumerate(planet_ids):
        airs_signal = data_airs[i]
        fgs_signal = data_fgs[i]

        # physics-inspired + FFT features
        airs_feats = {**extract_physical_features(airs_signal),
                      **extract_frequency_features(airs_signal)}
        fgs_feats = {**extract_physical_features(fgs_signal),
                     **extract_frequency_features(fgs_signal)}

        # metadata
        meta = star_info[star_info["planet_id"] == int(planet_id)].iloc[0]
        label = spectra_df[spectra_df["planet_id"] == int(planet_id)].drop(columns=["planet_id"]).iloc[0].to_numpy()

        combined = {
            "planet_id": planet_id,
            "Ts": meta["Ts"],
            "Mp": meta["Mp"],
            "P": meta["P"],
            **{f"AIRS_{k}": v for k, v in airs_feats.items()},
            **{f"FGS_{k}": v for k, v in fgs_feats.items()}
        }

        all_features.append(combined)
        all_labels.append(label)

    return pd.DataFrame(all_features), np.array(all_labels)


# X_df, y = extract_all_features(data_train, data_train_FGS, planet_ids, train_star_info, train_df)
# scaler = StandardScaler()
# X_numeric = X_df.drop(columns=['planet_id'])
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_numeric)
# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=1)


# def build_custom_kernel(y):
#     kernel1 = C(y.max() - y.min(), (1e-9, 1e3)) * RBF(10, (1, 1e5))
#     kernel2 = C(y.max() - y.min(), (1e-9, 1e3)) * Matern(length_scale=10, length_scale_bounds=(1, 1e5), nu=1.5)
#     return kernel1 + kernel2

# # Fit GPR
# def train_gpr(X_train, y_train, noise_level=1e-2):
#     kernel = build_custom_kernel(y_train)
#     gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_level**2, n_restarts_optimizer=10, normalize_y=True)
#     gpr.fit(X_train, y_train)
#     return gpr


# ridge = Ridge(alpha=1e-12)
# xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

# ridge.fit(X_train, y_train)
# xgb.fit(X_train, y_train)
# gpr = train_gpr(X_train, y_train)


ridge = load("/kaggle/input/ensemble-models/ridge.joblib")
xgb = load("/kaggle/input/ensemble-models/xgb.joblib")
gpr = load("/kaggle/input/ensemble-models/gpr.joblib")
scaler = load("/kaggle/input/ensemble-models/scaler.joblib")


# ## test data for competition submission
X_test_df, y_test_df = extract_all_features(data_test, data_test_FGS, planet_test_ids, test_star_info, test_df)
X_test_df = X_test_df.drop(columns=["planet_id"])
X_test_scaled = scaler.transform(X_test_df)


# Predict means
ridge_mean = ridge.predict(X_test_scaled)
xgb_mean = xgb.predict(X_test_scaled)
gpr_mean, gpr_std = gpr.predict(X_test_scaled, return_std=True)

#weighted mean
ensemble_mean = (gpr_mean*2 + xgb_mean*4 + ridge_mean) / 7

# weighted uncertainty (conservative aggregation)
ridge_std = np.std(ridge_mean) * np.ones_like(ridge_mean)
xgb_std = np.std(xgb_mean) * np.ones_like(xgb_mean)

ensemble_std = (gpr_std*2 + xgb_std*4 + ridge_std*2) / 8


X_train = load("/kaggle/input/ensemble-models/X_train.joblib")


ensemble_train_mean = load("/kaggle/input/ensemble-models/ensemble_train_mean2 (1).joblib")
ensemble_train_std = load("/kaggle/input/ensemble-models/ensemble_train_std2.joblib")
y_val = load("/kaggle/input/ensemble-models/y_val.joblib")


def gll_score(y, mu, sigma):
    # gaussian log-likelihood per sample
    return -0.5 * (np.log(2*np.pi) + 2*np.log(sigma) + ((y-mu)**2)/(sigma**2))

def gll_mean(y, mu, sigma):
    return np.mean(gll_score(y, mu, sigma))

# grid search on lambda, epsilon
def find_best_scale(y_val, mu_val, sigma_val, lambdas=None, epsilons=None):
    lambdas = np.concatenate([
        np.linspace(0.01, 0.1, 20),   # very conservative
        np.linspace(0.1, 1.0, 40),    # conservative  
        np.linspace(1.0, 5.0, 40),    # normal
        np.linspace(5.0, 50.0, 20)    # aggressive
    ])
    epsilons = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    best_grid = (-1e9, None, None)
    
    for lam in lambdas:
        for eps in epsilons:
            sigma_cal = np.maximum(eps, lam * sigma_val)
            score = gll_mean(y_val, mu_val, sigma_cal)
            if score > best_grid[0]:
                best_grid = (score, (lam, eps))

    return best_grid

# usage
best_score, (best_lambda, best_eps) = find_best_scale(y_val, ensemble_train_mean, ensemble_train_std)
sigma_test_cal = np.maximum(best_eps, best_lambda * ensemble_std)


def create_submission(predictions, uncertainties, planet_ids, sample_path="/kaggle/input/ariel-data-challenge-2025/sample_submission.csv", output_path="submission.csv"):
    sample_columns = pd.read_csv(sample_path, index_col="planet_id").columns

    submission_df = pd.DataFrame(
        np.concatenate([predictions, uncertainties], axis=1),
        columns=sample_columns,
        index=[int(pid) for pid in planet_ids]
    )

    submission_df.index.name = "planet_id"
    submission_df.reset_index(inplace=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return submission_df

create_submission(ensemble_mean, sigma_test_cal, planet_test_ids)


def competition_score_fixed(solution, submission, naive_mean=0.0025517145902829823, 
                           naive_sigma=0.0017261973793536417, sigma_true=1e-5):
    """
    naive, sigma true from last years winners
    """    
    n_wavelengths = 283
    
    # Remove planet_id columns if present
    solution_clean = solution.copy()
    print("Solution shape:", solution.shape)
    submission_clean = submission.copy()
    print("Submission shape:", submission.shape)
    
    if 'planet_id' in solution_clean.columns:
        solution_clean = solution_clean.drop('planet_id', axis=1)
    if 'planet_id' in submission_clean.columns:
        submission_clean = submission_clean.drop('planet_id', axis=1)
    
    # Extract predictions and uncertainties correctly
    y_pred = submission_clean.iloc[:, :n_wavelengths].values
    sigma_pred = submission_clean.iloc[:, n_wavelengths:].values  # FIX: Get uncertainties, not predictions
    
    # ensure non-zero sigma
    sigma_pred = np.clip(sigma_pred, a_min=1e-15, a_max=None)
    
    # ground truth
    #y_true = solution_clean.values
    y_true = solution_clean.values
    
    # calculate GLLs
    GLL_pred = np.sum(stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    GLL_true = np.sum(stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true))
    GLL_mean = np.sum(stats.norm.logpdf(y_true, loc=naive_mean, scale=naive_sigma))
    
    # normalize the score
    submit_score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
    
    print(f"GLL_pred: {GLL_pred:.2f}")
    print(f"GLL_true: {GLL_true:.2f}")
    print(f"GLL_mean: {GLL_mean:.2f}")
    print(f"Raw score: {submit_score:.6f}")
    print(GLL_pred > GLL_mean)
    
    return float(np.clip(submit_score, 0.0, 1.0))


#solution = train_df[:330]
solution = test_df.iloc[:, :284]
submission = pd.read_csv("/kaggle/working/submission.csv")
score = competition_score_fixed(solution, submission)
print(f"Final Score: {score:.6f}")

