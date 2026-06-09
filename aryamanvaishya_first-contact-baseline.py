#### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np

# Load data
train = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train.csv")
sample_sub = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")

# Print shapes
print("Train shape:", train.shape)
print("Sample submission shape:", sample_sub.shape)

# Peek at train
train.head(2)


# Step 1: Compute mean spectrum from training set
spectrum_columns = [col for col in train.columns if col.startswith("wl_")]
mean_spectrum = train[spectrum_columns].mean().values

# Step 2: Create predictions for each test planet
# Extract test planet_ids from sample_submission
test_planet_ids = sample_sub["planet_id"]

# Step 3: Build the prediction DataFrame
pred_df = pd.DataFrame()
pred_df["planet_id"] = test_planet_ids

# Add 283 spectral predictions (mean_spectrum)
for i, wl in enumerate(spectrum_columns):
    pred_df[wl] = mean_spectrum[i]

# Add 283 uncertainty values — let's use a constant 0.01 for now
for i, wl in enumerate(spectrum_columns):
    pred_df[f"{wl}_uncertainty"] = 0.01

# Step 4: Save submission file
pred_df.to_csv("/kaggle/working/submission.csv", index=False)
pred_df.head()


# Ariel Data Challenge 2025 - End-to-End Notebook

import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from glob import glob
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



# Data

train_df = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv', index_col='planet_id')
wavelengths = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')
train_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')
train_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')

def f_read_and_preprocess(dataset, planet_ids):
    f_raw = np.full((len(planet_ids), 67500), np.nan, dtype=np.float32)
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        f_signal = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/FGS1_signal_0.parquet')
        mean_signal = f_signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy() / 1024
        net_signal = mean_signal[1::2] - mean_signal[0::2]
        f_raw[i] = net_signal
    return f_raw



def a_read_and_preprocess(dataset, planet_ids):
    a_raw = np.full((len(planet_ids), 5625), np.nan, dtype=np.float32)
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        signal = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/AIRS-CH0_signal_0.parquet')
        mean_signal = signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy() / (32*356)
        net_signal = mean_signal[1::2] - mean_signal[0::2]
        a_raw[i] = net_signal
    return a_raw

planet_ids = train_df.index
f_raw_train = f_read_and_preprocess('train', planet_ids)
a_raw_train = a_read_and_preprocess('train', planet_ids)


import pandas as pd

# Load training metadata and labels
train_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv', index_col='planet_id')
train_labels = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv', index_col='planet_id')


%%writefile f_read_and_preprocess.py
import numpy as np
import polars as pl
from tqdm import tqdm

def f_read_and_preprocess(dataset, adc_info, planet_ids):
    """Read the FGS1 files for all planet_ids and extract the time series.
    
    Parameters
    dataset: 'train' or 'test'
    adc_info: metadata dataframe
    planet_ids: list of planet ids
    
    Returns
    ndarray with one row per planet_id and 67500 values per row
    """
    f_raw = np.full((len(planet_ids), 67500), np.nan, dtype=np.float32)
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        f_signal = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/FGS1_signal_0.parquet')
        mean_signal = f_signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy() / 1024
        net_signal = mean_signal[1::2] - mean_signal[0::2]
        f_raw[i] = net_signal
    return f_raw


%%writefile a_read_and_preprocess.py
import numpy as np
import polars as pl
from tqdm import tqdm

def a_read_and_preprocess(dataset, adc_info, planet_ids):
    """Read the AIRS-CH0 files for all planet_ids and extract the time series.
    
    Parameters
    dataset: 'train' or 'test'
    adc_info: metadata dataframe
    planet_ids: list of planet ids
    
    Returns
    ndarray with one row per planet_id and 5625 values per row
    """
    a_raw = np.full((len(planet_ids), 5625), np.nan, dtype=np.float32)
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        signal = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/AIRS-CH0_signal_0.parquet')
        mean_signal = signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy() / (32 * 356)
        net_signal = mean_signal[1::2] - mean_signal[0::2]
        a_raw[i] = net_signal
    return a_raw


def feature_engineering(f_raw, a_raw, n_bins=75):
    f_feat = f_raw.reshape(f_raw.shape[0], n_bins, -1).mean(axis=2)
    a_feat = a_raw.reshape(a_raw.shape[0], n_bins, -1).mean(axis=2)
    return np.concatenate([f_feat, a_feat], axis=1)

exec(open('f_read_and_preprocess.py', 'r').read())
exec(open('a_read_and_preprocess.py', 'r').read())

f_raw_train = f_read_and_preprocess('train', train_adc_info, train_labels.index)
a_raw_train = a_read_and_preprocess('train', train_adc_info, train_labels.index)

X = feature_engineering(f_raw_train, a_raw_train)
y = train_df.values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=0.1)
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Train MSE: {mse:.6f} | R2: {r2:.6f}")

sigma_pred = 0.01  


# Save 

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)
with open('sigma_pred.pickle', 'wb') as f:
    pickle.dump(sigma_pred, f)


test_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/test_star_info.csv', index_col='planet_id')
sample_submission = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv', index_col='planet_id')

f_raw_test = f_read_and_preprocess('test', test_adc_info, sample_submission.index)
a_raw_test = a_read_and_preprocess('test', test_adc_info, sample_submission.index)

X_test = feature_engineering(f_raw_test, a_raw_test)
X_test_scaled = scaler.transform(X_test)
y_test_pred = model.predict(X_test_scaled)


def postprocessing(pred_array, index, sigma_pred):
    columns = [f"wl_{i+1}" for i in range(pred_array.shape[1])]
    df_pred = pd.DataFrame(pred_array.clip(0, None), index=index, columns=columns)
    if np.isscalar(sigma_pred):
        sigma_array = np.full_like(pred_array, sigma_pred)
    else:
        sigma_array = sigma_pred
    df_sigma = pd.DataFrame(sigma_array, index=index, columns=[f"{c}_uncertainty" for c in columns])
    return pd.concat([df_pred, df_sigma], axis=1)

submission = postprocessing(y_test_pred, sample_submission.index, sigma_pred)
submission.to_csv('submission.csv')
submission.head()




