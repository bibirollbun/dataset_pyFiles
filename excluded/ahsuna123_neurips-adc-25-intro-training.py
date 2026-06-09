import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
from tqdm import tqdm
import pickle

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


import os
import pandas as pd
from glob import glob
from collections import defaultdict
import numpy as np

# --- Load CSVs ---
train_df = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')
wavelengths_df = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')
train_star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')

# --- 1. Basic Train Set Stats ---
print("ğŸª� Number of training planets:", train_df.shape[0])
print("ğŸ“ˆ Number of target labels (wavelengths):", train_df.shape[1] - 1)
print("ğŸ”¬ Length of wavelength grid:", wavelengths_df.shape[0])

# --- 2. Target Stats (per flux column) ---
target_cols = [col for col in train_df.columns if col != 'planet_id']
flux_summary = train_df[target_cols].agg(['min', 'max', 'mean', 'std']).T
print("\nğŸ“Š Flux value summary (first 5 rows):")
print(flux_summary.head())

# --- 3. Unique Stars ---
if 'planet_id' in train_star_info.columns:
    num_stars = train_star_info.drop(columns='planet_id').drop_duplicates().shape[0]
else:
    num_stars = train_star_info.drop_duplicates().shape[0]
print("\nğŸŒŸ Number of unique stars in training:", num_stars)

# --- 4. Planets with Multiple Observations ---
obs_counts = defaultdict(int)
train_planets = os.listdir('/kaggle/input/ariel-data-challenge-2025/train')

for pid in train_planets:
    air_obs = glob(f"train/{pid}/AIRS-CH0_signal_*.parquet")
    obs_counts[pid] = len(air_obs)

multi_obs = {pid: count for pid, count in obs_counts.items() if count > 1}
print("\nğŸ”� Planets with multiple observations:", len(multi_obs))

# --- 5. Check Calibration File Coverage ---
missing_calibs = []
expected = {"dark", "dead", "flat", "linear_corr", "read"}

for pid in train_planets:
    for band in ["AIRS-CH0", "FGS1"]:
        calib_path = f"train/{pid}/{band}_calibration"
        calib_files = {os.path.splitext(f)[0] for f in os.listdir(calib_path)} if os.path.exists(calib_path) else set()
        missing = expected - calib_files
        if missing:
            missing_calibs.append((pid, band, missing))

print("\nğŸ§ª Planets missing calibration files:", len(missing_calibs))
if missing_calibs:
    print("   Example:", missing_calibs[0])

# --- 6. Optional: Distribution of Observations Per Planet ---
obs_distribution = pd.Series(list(obs_counts.values())).value_counts().sort_index()
print("\nğŸ—‚ Observation count distribution per planet (AIR-CH0):")
print(obs_distribution)

# --- 7. Planet-Star Uniqueness Check ---
merged = pd.merge(train_df[['planet_id']], train_star_info, on='planet_id', how='left')
unique_links = merged[['planet_id'] + [col for col in train_star_info.columns if col != 'planet_id']].drop_duplicates()
print("\nğŸ”— Unique planet-star mappings:", unique_links.shape[0])



train_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/adc_info.csv')
# test_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2024/test_adc_info.csv',
#                            index_col='planet_id')
train_labels = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv',
                           index_col='planet_id')
wavelengths = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')
axis_info = pd.read_parquet('/kaggle/input/ariel-data-challenge-2025/axis_info.parquet')



planet_id = 1010375142
f_signal = pd.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_0.parquet')
f_signal


_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(f_signal.iloc[0].values.reshape(32, 32), ax=ax1, vmin=0, vmax=52000)
ax1.set_aspect('equal')
sns.heatmap(f_signal.iloc[1].values.reshape(32, 32), ax=ax2, vmin=0, vmax=52000)
ax2.set_aspect('equal')
plt.suptitle('A pair of FGS1 images')
plt.show()


_, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(12, 4))

planet_id = 1048114509
f_signal = pd.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_0.parquet')

mean_signal = f_signal.values.mean(axis=1)
net_signal = mean_signal[1::2] - mean_signal[0::2]
cum_signal = net_signal.cumsum()
window=800
smooth_signal = (cum_signal[window:] - cum_signal[:-window]) / window

ax1.set_title('FGS1: time series of planet with strong signal')
ax1.plot(net_signal, label='raw signal')
ax1.legend()
ax3.plot(smooth_signal, color='c', label='smoothened signal')
ax3.legend()
ax3.set_xlabel('time step')
for time_step in [20500, 23500, 44000, 47000]:
    ax3.axvline(time_step, color='gray')

planet_id = 1240764363
f_signal = pd.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/FGS1_signal_0.parquet')

mean_signal = f_signal.values.mean(axis=1)
net_signal = mean_signal[1::2] - mean_signal[0::2]
cum_signal = net_signal.cumsum()
window=800
smooth_signal = (cum_signal[window:] - cum_signal[:-window]) / window

ax2.set_title('FGS1: time series of planet with weak signal')
ax2.plot(net_signal, label='raw signal')
ax2.legend()
ax4.plot(smooth_signal, color='c', label='smoothened signal')
ax4.legend()
ax4.set_xlabel('time step')
for time_step in [20500, 23500, 44000, 47000]:
    ax4.axvline(time_step, color='gray')

# plt.suptitle('FGS1 time series', y=0.96)
plt.show()


planet_id = 1240764363
a_signal = pd.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/train/{planet_id}/AIRS-CH0_signal_0.parquet')
a_signal


a_signal = a_signal.values.reshape(11250, 32, 356)

plt.figure(figsize=(10, 3))
sns.heatmap(a_signal[1])
plt.ylabel('spatial dimension')
plt.xlabel('wavelength dimension')
plt.show()


mean_signal = a_signal.mean(axis=2).mean(axis=1)
net_signal = mean_signal[1::2] - mean_signal[0::2]
cum_signal = net_signal.cumsum()
window=80
smooth_signal = (cum_signal[window:] - cum_signal[:-window]) / window

_, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(net_signal, label='raw net signal')
ax1.legend()
ax2.plot(smooth_signal, color='c', label='smoothened net signal')
ax2.legend()
ax2.set_xlabel('time')
for time_step in [20500, 23500, 44000, 47000]:
    ax2.axvline(time_step * 11250 // 135000, color='gray')
plt.suptitle('AIRS-CH0 time series', y=0.96)
plt.show()



%%writefile f_read_and_preprocess.py

def f_read_and_preprocess(dataset, adc_info, planet_ids):
    """Read the FGS1 files for all planet_ids and extract the time series.
    
    Parameters
    dataset: 'train' or 'test'
    adc_info: metadata dataframe, either train_adc_info or test_adc_info
    planet_ids: list of planet ids
    
    Returns
    dataframe with one row per planet_id and 67500 values per row
    
    """
    f_raw_train = np.full((len(planet_ids), 67500), np.nan, dtype=np.float32)
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        f_signal = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/FGS1_signal_0.parquet')
        mean_signal = f_signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy() / 1024 # mean over the 32*32 pixels
        net_signal = mean_signal[1::2] - mean_signal[0::2]
        f_raw_train[i] = net_signal
    return f_raw_train
    


%%time
exec(open('f_read_and_preprocess.py', 'r').read())
f_raw_train = f_read_and_preprocess('train', train_adc_info, train_labels.index)
with open('f_raw_train.pickle', 'wb') as f:
    pickle.dump(f_raw_train, f)



%%writefile a_read_and_preprocess.py
def a_read_and_preprocess(dataset, adc_info, planet_ids):
    """Read the AIRS-CH0 files for all planet_ids and extract the time series.
    
    Parameters
    dataset: 'train' or 'test'
    adc_info: metadata dataframe, either train_adc_info or test_adc_info
    planet_ids: list of planet ids
    
    Returns
    dataframe with one row per planet_id and 5625 values per row
    
    """
    a_raw_train = np.full((len(planet_ids), 5625), np.nan, dtype=np.float32)
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        signal = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/AIRS-CH0_signal_0.parquet')
        mean_signal = signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy() / (32*356) # mean over the 32*356 pixels
        net_signal = mean_signal[1::2] - mean_signal[0::2]
        a_raw_train[i] = net_signal
    return a_raw_train
    


%%time
exec(open('a_read_and_preprocess.py', 'r').read())
a_raw_train = a_read_and_preprocess('train', train_adc_info, train_labels.index)
with open('a_raw_train.pickle', 'wb') as f:
    pickle.dump(a_raw_train, f)



plt.figure(figsize=(6, 2))
plt.plot(f_raw_train.mean(axis=0))
for time_step in [20500, 23500, 44000, 47000]:
    plt.axvline(time_step, color='gray')
plt.xlabel('time step')
plt.title('FGS1: Overall mean')
plt.show()

plt.figure(figsize=(6, 2))
plt.plot(a_raw_train.mean(axis=0))
for time_step in [20500, 23500, 44000, 47000]:
    plt.axvline(time_step * 11250 // 135000, color='gray')
plt.xlabel('time step')
plt.title('AIRS-CH0: Overall mean')
plt.show()


%%writefile feature_engineering.py

def feature_engineering(f_raw, a_raw):
    """Create a dataframe with two features from the raw data.
    
    Parameters:
    f_raw: ndarray of shape (n_planets, 67500)
    a_raw: ndarray of shape (n_planets, 5625)
    
    Return value:
    df: DataFrame of shape (n_planets, 2)
    """
    obscured = f_raw[:, 23500:44000].mean(axis=1)
    unobscured = (f_raw[:, :20500].mean(axis=1) + f_raw[:, 47000:].mean(axis=1)) / 2
    f_relative_reduction = (unobscured - obscured) / unobscured
    obscured = a_raw[:, 1958:3666].mean(axis=1)
    unobscured = (a_raw[:, :1708].mean(axis=1) + a_raw[:, 3916:].mean(axis=1)) / 2
    a_relative_reduction = (unobscured - obscured) / unobscured

    df = pd.DataFrame({'a_relative_reduction': a_relative_reduction,
                       'f_relative_reduction': f_relative_reduction})
    
    return df



exec(open('feature_engineering.py', 'r').read())

train = feature_engineering(f_raw_train, a_raw_train)


model = Ridge(alpha=1e-12)

oof_pred = cross_val_predict(model, train, train_labels)

print(f"# R2 score: {r2_score(train_labels, oof_pred):.3f}")
sigma_pred = mean_squared_error(train_labels, oof_pred, squared=False)
print(f"# Root mean squared error: {sigma_pred:.6f}")
# R2 score: 0.971
# Root mean squared error: 0.000293


col = 1
plt.scatter(oof_pred[:,col], train_labels.iloc[:,col], s=15, c='lightgreen')
plt.gca().set_aspect('equal')
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.title('Comparing y_true and y_pred')
plt.show()


%%writefile competition_score.py
# Adapted from https://www.kaggle.com/code/metric/ariel-gaussian-log-likelihood

# Custom error for invalid submissions
class ParticipantVisibleError(Exception):
    pass

def competition_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    naive_mean: float,
    naive_sigma: float,
    sigma_true: float,
    row_id_column_name='planet_id'
) -> float:
    '''
    Computes a Gaussian Log Likelihood-based score.
    '''
    # Drop ID columns
    solution = solution.drop(columns=[row_id_column_name], errors='ignore')
    submission = submission.drop(columns=[row_id_column_name], errors='ignore')

    # Validation checks
    if submission.min().min() < 0:
        raise ParticipantVisibleError('Negative values in the submission')

    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be numeric: {col}')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != 2 * n_wavelengths:
        raise ParticipantVisibleError('Submission must have 2x columns of the solution')

    # Extract predictions and sigmas
    y_pred = submission.iloc[:, :n_wavelengths].values
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=1e-15, a_max=None)
    y_true = solution.values

    # Compute log likelihoods
    GLL_pred = np.sum(scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    GLL_true = np.sum(scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true))
    GLL_mean = np.sum(scipy.stats.norm.logpdf(y_true, loc=naive_mean, scale=naive_sigma))

    score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
    return float(np.clip(score, 0.0, 1.0))




def postprocessing(pred_array, index, sigma_pred, column_names=None):
    """
    Creates a submission DataFrame with mean predictions and uncertainties.

    Parameters:
    - pred_array: ndarray of shape (n_samples, 283)
    - index: pandas.Index of length n_samples
    - sigma_pred: float or ndarray of shape (n_samples, 283)
    - column_names: list of wavelength column names (optional)

    Returns:
    - df: DataFrame of shape (n_samples, 566)
    """
    n_samples, n_waves = pred_array.shape

    if column_names is None:
        column_names = [f"wl_{i+1}" for i in range(n_waves)]

    if np.isscalar(sigma_pred):
        sigma_pred = np.full_like(pred_array, sigma_pred)

    # Safety check
    assert sigma_pred.shape == pred_array.shape, "sigma_pred must match shape of pred_array"
    assert len(index) == n_samples, "Index length must match number of rows"

    df_mean = pd.DataFrame(pred_array.clip(0, None), index=index, columns=column_names)
    df_sigma = pd.DataFrame(sigma_pred, index=index, columns=[f"sigma_{i+1}" for i in range(n_waves)])

    return pd.concat([df_mean, df_sigma], axis=1)


exec(open('competition_score.py', 'r').read())
#exec(open('postprocessing.py', 'r').read())

oof_df = postprocessing(oof_pred, train_labels.index, sigma_pred)
display(oof_df)

gll_score = competition_score(train_labels.copy().reset_index(),
                              oof_df.copy().reset_index(),
                              naive_mean=train_labels.values.mean(),
                              naive_sigma=train_labels.values.std(),
                              sigma_true=0.000003)
print(f"# Estimated competition score: {gll_score:.3f}")
# Estimated competition score: 0.123


# Refit the model to the full dataset
model.fit(train, train_labels)
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
with open('sigma_pred.pickle', 'wb') as f:
    pickle.dump(sigma_pred, f)


import pandas as pd
import numpy as np
import pickle

# 1. Load required files
test_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/test_star_info.csv', index_col='planet_id')
sample_submission = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv', index_col='planet_id')
wavelengths = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/wavelengths.csv')

# 2. Load model and sigma
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('sigma_pred.pickle', 'rb') as f:
    sigma_pred = pickle.load(f)

# 3. Run your preprocessing + feature extraction on test set
# These must be implemented in your own code â€” you can adapt from training logic
f_raw_test = f_read_and_preprocess('test', test_adc_info, sample_submission.index)
a_raw_test = a_read_and_preprocess('test', test_adc_info, sample_submission.index)
test_features = feature_engineering(f_raw_test, a_raw_test)

# 4. Predict
test_pred = model.predict(test_features)

# 5. Postprocessing
def postprocessing(pred_array, index, sigma_pred, column_names):
    """
    Convert predictions and uncertainty into final submission DataFrame.
    """
    if np.isscalar(sigma_pred):
        sigma_array = np.full_like(pred_array, sigma_pred)
    else:
        sigma_array = sigma_pred
    df_pred = pd.DataFrame(pred_array.clip(0, None), index=index, columns=column_names)
    df_sigma = pd.DataFrame(sigma_array, index=index, columns=[f"sigma_{i}" for i in range(1, len(column_names)+1)])
    return pd.concat([df_pred, df_sigma], axis=1)

submission_df = postprocessing(
    pred_array=test_pred,
    index=sample_submission.index,
    sigma_pred=sigma_pred,
    column_names=wavelengths.columns
)

# 6. Save
submission_df.to_csv('submission.csv')

# 7. Preview
!head submission.csv


