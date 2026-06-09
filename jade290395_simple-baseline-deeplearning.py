import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm
from astropy.stats import sigma_clip
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip

ROOT_PATH = "/kaggle/input/ariel-data-challenge-2025"
MODE = "test"

class Config:
    CUT_INF = 39
    CUT_SUP = 250

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


class SignalProcessor:
    def __init__(self, config):
        self.cfg = config
        self.planet_ids = pd.read_csv(f"{ROOT_PATH}/{MODE}_star_info.csv", index_col='planet_id').index.astype(int)

    def _calibrate_single_signal(self, planet_id, sensor):
        cfg = self.cfg.SENSOR_CONFIG[sensor]
        path = f"{ROOT_PATH}/{MODE}/{planet_id}/"

        signal = pd.read_parquet(path + f"{sensor}_signal_0.parquet").to_numpy().reshape(cfg["raw_shape"])
        dark = pd.read_parquet(path + f"{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(path + f"{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(path + f"{sensor}_calibration_0/flat.parquet").to_numpy()
        linear_corr = pd.read_parquet(path + f"{sensor}_calibration_0/linear_corr.parquet").to_numpy().astype(np.float64).reshape(cfg["linear_corr_shape"])

        gain, offset = 0.4369, -1000.0
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask

        if sensor == "AIRS-CH0":
            inf = self.cfg.CUT_INF
            sup = self.cfg.CUT_SUP
            signal = signal[:, :, inf:sup]
            linear_corr = linear_corr[:, :, inf:sup]
            dark = dark[:, inf:sup]
            dead = dead[:, inf:sup]
            flat = flat[:, inf:sup]
            hot = hot[:, inf:sup]

        base_dt, increment = cfg["dt_pattern"]
        dt = np.ones(len(signal)) * base_dt
        dt[1::2] += increment

        signal = signal.clip(0)
        linear_corr_flipped = np.flip(linear_corr, axis=0)
        corrected_signal = signal.copy()
        
        for x, y in itertools.product(range(signal.shape[1]), range(signal.shape[2])):
            poly = np.poly1d(linear_corr_flipped[:, x, y])
            corrected_signal[:, x, y] = poly(corrected_signal[:, x, y])
        signal = corrected_signal
        signal -= dark * dt[:, np.newaxis, np.newaxis]

        flat = flat.reshape(cfg["calibrated_shape"])
        dead_mask = dead.reshape(cfg["calibrated_shape"])
        hot_mask = hot.reshape(cfg["calibrated_shape"])

        flat[dead_mask] = np.nan
        flat[hot_mask] = np.nan

        signal = signal / flat

        return signal

    def _preprocess_calibrated_signal(self, signal, sensor):
        cfg = self.cfg.SENSOR_CONFIG[sensor]
        binning = cfg["binning"]

        if sensor == "AIRS-CH0":
            roi = signal[:, 10:22, :]
        else:
            roi = signal[:, 10:22, 10:22]
            roi = roi.reshape(roi.shape[0], -1)

        mean_signal = np.nanmean(roi, axis=1)
        
        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array([
            cds_signal[j*binning : (j+1)*binning].mean(axis=0) 
            for j in range(n_bins)
        ])

        if sensor == "FGS1":
            binned = binned.reshape((binned.shape[0], 1))
        return binned


    def process_all_data(self):
        combined_signals = []
    
        for planet_id in tqdm(self.planet_ids):
            signal_fgs = self._calibrate_single_signal(planet_id, "FGS1")
            processed_fgs = self._preprocess_calibrated_signal(signal_fgs, "FGS1")
    
            signal_airs = self._calibrate_single_signal(planet_id, "AIRS-CH0")
            processed_airs = self._preprocess_calibrated_signal(signal_airs, "AIRS-CH0")
    
            combined = np.concatenate([processed_fgs, processed_airs], axis=1)
            combined_signals.append(combined)
    
        result = np.stack(combined_signals)
        return result



processor = SignalProcessor(Config)
StarInfo = pd.read_csv(ROOT_PATH + f"/{MODE}_star_info.csv")
StarInfo["planet_id"] = StarInfo["planet_id"].astype(int)
PlanetIds = StarInfo["planet_id"].tolist()
StarInfo = StarInfo.set_index("planet_id")
signal_train = processor.process_all_data()



def phase_detector(signal):
    min_index = np.argmin(signal[30:140]) + 30
    signal1 = signal[:min_index]
    grad1 = np.gradient(signal1)
    grad1 /= grad1.max()
    signal2 = signal[min_index:]
    grad2 = np.gradient(signal2)
    grad2 /= grad2.max()

    phase1 = np.argmin(grad1)
    phase2 = np.argmax(grad2) + min_index

    return phase1, phase2



def compute_s_opt(signal, phase1, phase2, buffer_size = 7, degree = 3):
    delta = buffer_size
    if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
        delta = 2
    def objective(s):
        middle = signal[phase1 + delta : phase2 - delta] * (1 + s)
        y = np.concatenate([signal[:phase1 - delta], middle, signal[phase2 + delta:]])
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)        
        return np.abs(poly(x) - y).mean()

    res = minimize(objective, 0.0001, method="Nelder-Mead")
    return res.x[0]


y_shifts = []
for IDX in tqdm(range(len(signal_train))):
    data = signal_train[IDX]
    normalized_data = data[:, 1:].mean(1)
    normalized_data = savgol_filter(normalized_data, 30, 2)
    phase1, phase2 = phase_detector(normalized_data)
    buffer_size_poly = 7
    phase1 = max(7, phase1)
    phase2 = min(len(normalized_data) - 7 - 1, phase2) 
    opt = compute_s_opt(normalized_data, phase1, phase2, buffer_size_poly, degree = 3)
    y_shifts.append(opt)
y_shifts = np.array(y_shifts) * 0.93960
df = pd.DataFrame({"transit_depth":y_shifts},index=StarInfo.index,)


input_df = pd.merge(df, StarInfo, on="planet_id", how="left")


input_df["transit_depth"] *= 10000
features = ['transit_depth','Rs','i']
X = input_df[features].values.astype(np.float32)



class ResidualBlock(nn.Module):
    def __init__(self, dim, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)


class ResNetMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, p=dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


model = ResNetMLP(num_blocks=80, dropout_rate=0.5)
model.load_state_dict(torch.load("/kaggle/input/net/pytorch/default/1/best_artifacts/best_model.pth"))
model.eval()


X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_tensor).numpy()
y_pred /= 10000


sample = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")
sample_columns = sample.columns.drop("planet_id")
n_outputs = len(sample_columns) // 2
repeated_predictions = np.repeat(df.values, n_outputs).reshape(len(df), -1)
repeated_predictions[:,0] = (y_pred[:,0] * 0.8 + (X[:,0] * 0.2 )/ 10000)
repeated_predictions = repeated_predictions.clip(0)
sigmas = np.ones_like(repeated_predictions)
submission_values = np.concatenate([repeated_predictions, sigmas], axis=1)
submission_df = pd.DataFrame(submission_values, columns=sample_columns, index=df.index)
submission_df.insert(0, "planet_id", df.index)
submission_df.to_csv("submission.csv", index=False)
pd.read_csv("submission.csv")




