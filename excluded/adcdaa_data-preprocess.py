import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
import polars as pl
from tqdm import tqdm
import pickle
import gc


import itertools
import torch
from time import time
from scipy.optimize import minimize
from functools import partial
import random, os
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

from catboost import CatBoostRegressor


test_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/test_star_info.csv')
train_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')
adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
axis_info = pd.read_parquet('/kaggle/input/ariel-data-challenge-2025/axis_info.parquet')
DEVICE = "cuda"
binn = 10



def apply_linear_corr(linear_corr, clean_signal):
    linear_corr = torch.flip(linear_corr, dims=[0])

    x_range = torch.arange(clean_signal.shape[1], device=clean_signal.device)
    y_range = torch.arange(clean_signal.shape[2], device=clean_signal.device)
    xx, yy = torch.meshgrid(x_range, y_range, indexing="ij")

    clean_signal = clean_signal.cuda()
    result_signal = torch.zeros_like(clean_signal)

    for i in range(linear_corr.shape[0]):
        result_signal += linear_corr[i, xx, yy] * clean_signal ** (linear_corr.shape[0] - 1 - i)

    return result_signal

def clean_dark(signal, dark, dt):
    dark_expanded = dark.unsqueeze(0)
    dt_expanded = dt.unsqueeze(1).unsqueeze(2)

    signal = signal - dark_expanded * dt_expanded

    return signal


def preproc(dataset, star_info, adc_info, sensor, binning = 15):
    cut_inf, cut_sup = 39, 321
    sensor_sizes_dict = {"AIRS-CH0":[[11250, 32, 356], [1, 32, cut_sup-cut_inf]], "FGS1":[[135000, 32, 32], [1, 32, 32]]}
    binned_dict = {"AIRS-CH0":[11250 // binning // 2, 282], "FGS1":[135000 // binning // 2]}
    linear_corr_dict = {"AIRS-CH0":(6, 32, 356), "FGS1":(6, 32, 32)}
    planet_ids = star_info['planet_id'].astype(int).tolist()
    DEVICE = "cuda:0"

    feats = []
    feats_center = []
    for i, planet_id in tqdm(list(enumerate(planet_ids)), desc=f"Processing {sensor}"):
        signal = torch.Tensor(pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/{sensor}_signal_0.parquet').to_numpy().astype(np.float32)).to(DEVICE)
        dark_frame = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration_0/dark.parquet').to_numpy()
        dead_frame = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration_0/dead.parquet').to_numpy()
        flat_frame = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration_0/flat.parquet').to_numpy()
        linear_corr = torch.Tensor(pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration_0/linear_corr.parquet').to_numpy().astype(np.float32).reshape(linear_corr_dict[sensor])).to(DEVICE)

        signal = signal.reshape(sensor_sizes_dict[sensor][0]) 
        gain = adc_info[f'{sensor}_adc_gain'].values[0]
        offset = adc_info[f'{sensor}_adc_offset'].values[0]
        signal = signal / gain + offset

        hot = sigma_clip(
            dark_frame, sigma=5, maxiters=5
        ).mask
        dark_frame = torch.Tensor(dark_frame).to(DEVICE)

        if sensor == "AIRS-CH0":
            signal = signal[:, :, cut_inf:cut_sup] #11250 * 32 * 282
            #dt = axis_info['AIRS-CH0-integration_time'].dropna().values
            dt = torch.ones(len(signal)).to(DEVICE)*0.1 
            dt[1::2] += 4.5 #@bilzard idea
            linear_corr = linear_corr[:, :, cut_inf:cut_sup]
            dark_frame = dark_frame[:, cut_inf:cut_sup]
            dead_frame = dead_frame[:, cut_inf:cut_sup]
            flat_frame = flat_frame[:, cut_inf:cut_sup]
            hot = hot[:, cut_inf:cut_sup]
        else:
            dt = torch.ones(len(signal)).to(DEVICE)*0.1
            dt[1::2] += 0.1

        signal = signal.clip(0) #@graySnow idea
        linear_corr_signal = apply_linear_corr(linear_corr, signal)
        signal = clean_dark(linear_corr_signal, dark_frame, dt).cpu().numpy()

        flat = flat_frame.reshape(sensor_sizes_dict[sensor][1])
        flat[dead_frame.reshape(sensor_sizes_dict[sensor][1])] = np.nan
        flat[hot.reshape(sensor_sizes_dict[sensor][1])] = np.nan
        signal = signal / flat
        
        if sensor == "AIRS-CH0":
            signal_center = signal[:, 10:22, :]
        elif sensor == "FGS1":
            signal_center = signal[:, 10:22, 10:22]
            signal_center = signal_center.reshape(
                signal_center.shape[0], signal_center.shape[1] * signal_center.shape[2]
            )
            
        if sensor == "FGS1":
            signal = signal.reshape((sensor_sizes_dict[sensor][0][0], sensor_sizes_dict[sensor][0][1]*sensor_sizes_dict[sensor][0][2]))

        mean_signal = np.nanmean(signal, axis=1) # mean over the 32*32(FGS1) or 32(CH0) pixels
        cds_signal = (mean_signal[1::2] - mean_signal[0::2])
        binned = np.zeros((binned_dict[sensor]))
        for j in range(cds_signal.shape[0] // binning):
            binned[j] = cds_signal[j*binning:j*binning+binning].mean(axis=0)
        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0],1))
        feats.append(binned)
        
        
        mean_signal = np.nanmean(signal_center, axis=1) # mean over the 32*32(FGS1) or 32(CH0) pixels
        cds_signal = (mean_signal[1::2] - mean_signal[0::2])

        binned = np.zeros((binned_dict[sensor]))
        for j in range(cds_signal.shape[0] // binning):
            binned[j] = cds_signal[j*binning:j*binning+binning].mean(axis=0)
       
        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0],1))

        feats_center.append(binned)
        
        

    return np.stack(feats), np.stack(feats_center)

fgs1_feats, fgs1_feats_center = preproc('train', train_star_info, adc_info, "FGS1", binn*12)
airsch0_feats, airsch0_feats_center = preproc('train', train_star_info, adc_info, "AIRS-CH0", binn)


train_binn = np.concatenate([fgs1_feats_center, airsch0_feats_center], axis=2)
train_binn_not_centered = np.concatenate([fgs1_feats, airsch0_feats], axis=2)

pickle.dump(train_binn_not_centered, open('train_binn_not_centered.pkl', 'wb'))

del fgs1_feats
del fgs1_feats_center
del airsch0_feats
del airsch0_feats_center
del train_binn_not_centered
gc.collect()
torch.cuda.empty_cache()


np.save('train_binn.npy', train_binn)


fgs1_feats, fgs1_feats_center = preproc('test', test_star_info, adc_info, "FGS1", binn*12)
airsch0_feats, airsch0_feats_center = preproc('test', test_star_info, adc_info, "AIRS-CH0", binn)


test_binn = np.concatenate([fgs1_feats_center, airsch0_feats_center], axis=2)
test_binn_not_centered = np.concatenate([fgs1_feats, airsch0_feats], axis=2)

pickle.dump(test_binn_not_centered, open('test_binn_not_centered.pkl', 'wb'))

del fgs1_feats
del fgs1_feats_center
del airsch0_feats
del airsch0_feats_center
del test_binn_not_centered
gc.collect()
torch.cuda.empty_cache()


np.save('test_binn.npy', test_binn)

