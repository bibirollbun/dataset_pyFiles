# This Python 3 environment comes with many helpful analytics libraries installed
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
import cupy as cp
import gc
from sklearn.metrics import mean_squared_log_error

# Define version
VER = 1

# Define paths and lists
files = []
x_train = []
x_test = []
PATH = "/kaggle/input/may2025-playground-oofs/"

# List of models to include
models = ['xgb (1)', 'catboost', 'lgb', 'nn']

print("Loading files...")
for c in models:
    print(f"=> {c} ", end="")
    # Load OOF predictions
    oof_file = f"{PATH}oof_{c}.npy"
    try:
        oof = np.load(oof_file)
    except FileNotFoundError as e:
        print(f"\nError loading {oof_file}: {e}")
        raise
    # Check for NaNs, infinities, and clip values
    oof = np.nan_to_num(oof, nan=0, posinf=500, neginf=0)
    oof = np.clip(oof, 0, 500)
    # Apply log1p if not already in log space
    if oof.mean() > 10:
        oof = np.log1p(oof)
    x_train.append(oof)
    files.append(f"oof_{c}")
    
    # Load test predictions
    submission_file = f"{PATH}submission_{c}.csv"
    try:
        df = pd.read_csv(submission_file)
        pred = df['Calories'].values
    except FileNotFoundError as e:
        print(f"\nError loading {submission_file}: {e}")
        raise
    # Check for NaNs, infinities, and clip values
    pred = np.nan_to_num(pred, nan=0, posinf=500, neginf=0)
    pred = np.clip(pred, 0, 500)
    # Apply log1p to test predictions
    pred = np.log1p(pred)
    x_test.append(pred)
    print()

# Stack OOF and test predictions
x_train = np.stack(x_train).T
print("Our combined OOF have shape:", x_train.shape)

x_test = np.stack(x_test).T
print("Our combined PRED have shape:", x_test.shape)

# Load true values
try:
    train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
except FileNotFoundError as e:
    print(f"Error loading train data: {e}")
    raise
true = train['Calories'].values

# Define RMSLE metric (CPU)
def compute_metric_rmsle(p):
    p = np.clip(np.expm1(p), 0, 500)
    m = np.sqrt(mean_squared_log_error(true, p))
    return m

# Compute RMSLE for each OOF
best_score = float('inf')
best_index = -1

for k, name in enumerate(files):
    s = compute_metric_rmsle(x_train[:, k])
    if s < best_score:
        best_score = s
        best_index = k
    print(f'RMSLE {s:0.5f} {name}')
print()
print(f'Best single model is {files[best_index]} with RMSLE = {best_score:0.5f}')

# GPU-accelerated RMSLE computation
def multiple_rmsle_scores(actual, predicted):
    """
    Computes multiple RMSLE scores using GPU.
    
    Parameters:
    ----------
    actual : cupy.ndarray
        A 1D GPU array of shape (N,), true labels.
    predicted : cupy.ndarray
        A 2D GPU array of shape (N, K), predicted values in log space.
    
    Returns:
    -------
    cupy.ndarray
        A 1D GPU array of shape (K,) containing RMSLE scores.
    """
    if len(actual.shape) == 1:
        actual = actual[:, cp.newaxis]
    predicted_exp = cp.clip(cp.expm1(predicted), 0, 500)
    actual_exp = cp.clip(actual, 0, 500)
    m = cp.sqrt(cp.mean((cp.log1p(actual_exp) - cp.log1p(predicted_exp))**2.0, axis=0))
    return m

# Hill Climbing Parameters
USE_NEGATIVE_WGT = True
MAX_MODELS = 1000
TOL = 1e-5

indices = [best_index]
old_best_score = best_score
print(f'0 We begin with best single model RMSLE {best_score:0.5f} from "{files[best_index]}"')

# Move variables to GPU
x_train2 = cp.array(x_train)  # GPU
best_ensemble = x_train2[:, best_index]  # GPU
truth = cp.array(true)  # GPU
start = -0.50 if USE_NEGATIVE_WGT else 0.01
ww = cp.arange(start, 0.51, 0.01)  # GPU
nn = len(ww)

# Begin Hill Climbing
models = [best_index]
weights = []
metrics = [best_score]

for kk in range(1_000_000):
    best_score = float('inf')
    best_index = -1
    best_weight = 0

    # Try adding one more model
    for k, ff in enumerate(files):
        if k in models:
            continue
        new_model = x_train2[:, k]  # GPU
        m1 = cp.repeat(best_ensemble[:, cp.newaxis], nn, axis=1) * (1 - ww)  # GPU
        m2 = cp.repeat(new_model[:, cp.newaxis], nn, axis=1) * ww  # GPU
        mm = m1 + m2  # GPU
        new_scores = multiple_rmsle_scores(truth, mm)
        new_score = cp.min(new_scores).item()  # GPU -> CPU
        if new_score < best_score:
            best_score = new_score  # CPU
            best_index = k  # CPU
            ii = cp.argmin(new_scores).item()  # GPU -> CPU
            best_weight = ww[ii].item()  # GPU -> CPU
            potential_ensemble = mm[:, ii]  # GPU
    del new_model, m1, m2, mm, new_scores, new_score
    gc.collect()

    # Stopping criteria
    indices.append(best_index)
    indices = list(np.unique(indices))
    if len(indices) > MAX_MODELS:
        print(f'=> We reached {MAX_MODELS} models')
        indices = indices[:-1]
        break
    if -1 * (best_score - old_best_score) < TOL:
        print(f'=> We reached tolerance {TOL}')
        break

    # Record new result
    print(kk + 1, 'New best RMSLE', best_score, f'adding "{files[best_index]}"', 'with weight', f'{best_weight:0.3f}')
    models.append(best_index)
    weights.append(best_weight)
    metrics.append(best_score)
    best_ensemble = potential_ensemble
    old_best_score = best_score

# Compute final weights
wgt = np.array([1])
for w in weights:
    wgt = wgt * (1 - w)
    wgt = np.concatenate([wgt, np.array([w])])

# Create weight table
rows = []
for m, w, s in zip(models, wgt, metrics):
    name = files[m]
    dd = {'weight': w, 'model': name, 'rmsle': s}
    rows.append(dd)

# Display weights per model
df = pd.DataFrame(rows)
df = df.groupby('model').agg({'weight': 'sum', 'rmsle': 'first'}).reset_index().sort_values('weight', ascending=False)
df = df.reset_index(drop=True)
print("\nFinal weights:")
print(df)

# Combine OOF predictions (using weights from hill climbing)
x_map = {x: y for x, y in zip(files, np.arange(len(files)))}
x_train3 = x_train2.get()  # GPU -> CPU
ensemble = x_train3[:, x_map[df.model.iloc[0]]] * df.weight.iloc[0]
for k in range(1, len(df)):
    ensemble += x_train3[:, x_map[df.model.iloc[k]]] * df.weight.iloc[k]
m = compute_metric_rmsle(ensemble)
print(f'Overall Hill climbing RMSLE = {m:0.6f}')

# Save OOF ensemble
np.save(f'oof_hill_climb_v{VER}', ensemble)

# Combine test predictions (using weights from hill climbing)
pred = x_test[:, x_map[df.model.iloc[0]]] * df.weight.iloc[0]
for k in range(1, len(df)):
    pred += x_test[:, x_map[df.model.iloc[k]]] * df.weight.iloc[k]

# Write submission
sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
mn = train.Calories.min()
mx = train.Calories.max()
sub['Calories'] = np.clip(np.expm1(pred), mn, mx)

print("Test shape:", sub.shape)
print("Test target mean is:", sub.Calories.mean())
sub.to_csv(f"submission_hill_climb_v{VER}.csv", index=False)
print(sub.head())

