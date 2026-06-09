!pip install -q pybaselines


import pickle

with open('/kaggle/input/public-load-train-val-and-test-data/all_data.pkl', 'rb') as f:
    train_data, val_data, test_data = pickle.load(f)
    X_train, y_train, cv_folds, dataset_offsets = train_data
    X_val_raw, y_val, X_val = val_data
    X_test_raw, X_test = test_data


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

import pybaselines.whittaker

def apply_baseline_correction(spectra):
    corrected_spectra = np.zeros_like(spectra)
    baseline_fitter = pybaselines.Baseline()

    for i, spectrum in enumerate(spectra):
        baseline, params = baseline_fitter.snip(
            spectrum,
            max_half_window=40,
            decreasing=True,
            smooth_half_window=3
        )
        corrected_spectra[i] = spectrum - baseline
    return corrected_spectra

def apply_msc(spectra_to_transform, reference_spectrum=None):
    if reference_spectrum is None:
        ref_spec = np.mean(spectra_to_transform, axis=0)
    else:
        ref_spec = reference_spectrum

    corrected_spectra = np.zeros_like(spectra_to_transform, dtype=np.float64)

    for i, spectrum in enumerate(spectra_to_transform):
        coefficients = np.polyfit(ref_spec, spectrum, 1)
        slope = coefficients[0]
        intercept = coefficients[1]

        corrected_spectra[i] = (spectrum - intercept) / (slope + 1e-10)

    return corrected_spectra

def apply_smoothing(spectra):
    smoothed_spectra = np.zeros_like(spectra)
    for i, spectrum in enumerate(spectra):
        smoothed_spectra[i] = savgol_filter(spectrum, window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, deriv=DERIV)
    return smoothed_spectra

def apply_scaling(spectra):
    scaler = StandardScaler()
    return scaler.fit_transform(spectra)

def preprocess_spectra(spectra, reference_spectrum=None):
    msced = apply_msc(spectra, reference_spectrum)
    baseline_corrected = apply_baseline_correction(msced)
    smoothed = apply_smoothing(baseline_corrected)
    scaled = apply_scaling(smoothed)
    return scaled


def post_process(y_test_pred):
    target_names = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
    final_predictions = np.maximum(y_test_pred, 0)

    # Apply bounds based on training data
    for i, target in enumerate(target_names):
        lower_bound = np.percentile(y_val[:, i], 1)
        upper_bound = np.percentile(y_val[:, i], 99)
    
        # Add small margin
        margin = 0.1 * (upper_bound - lower_bound)
        lower_bound = max(0, lower_bound - margin)
        upper_bound = upper_bound + margin
    
        final_predictions[:, i] = np.clip(final_predictions[:, i], lower_bound, upper_bound)
        print(f"{target}: Clipped to [{lower_bound:.3f}, {upper_bound:.3f}]")
    return final_predictions

def fix_val_test_shape(X):
    lower_wns = 300
    upper_wns = 1942
    joint_wns = np.arange(lower_wns, upper_wns+1)
    spectral_values = np.linspace(65, 3350, 2048)

    spectra_selection = np.logical_and(
        lower_wns <= spectral_values, spectral_values <= upper_wns,
    )
    wns = spectral_values[spectra_selection]
    X = X[:, spectra_selection]
    X = np.array([np.interp(joint_wns, xp=wns, fp=spectrum,)for spectrum in X])
    return X


import numpy as np

def split_by_device(X_train, dataset_offsets):
    devices = []
    for i in range(len(dataset_offsets)):
        start = dataset_offsets[i]
        end = dataset_offsets[i+1] if i+1 < len(dataset_offsets) else len(X_train)
        devices.append(X_train[start:end])
    return devices

devices_X = split_by_device(X_train, dataset_offsets)
devices_X = devices_X[0:8]
devices_y = split_by_device(y_train, dataset_offsets)
devices_y = devices_y[0:8]


WINDOW_LENGTH = 7
POLY_ORDER = 2
DERIV = 0

devices_X_p = []

for device_X in devices_X:
    device_X_mean = device_X.mean(axis=0)
    device_X_p = preprocess_spectra(device_X, device_X_mean)
    devices_X_p.append(device_X_p)

X_val_p = preprocess_spectra(X_val, X_val.mean(axis=0))
X_test_p = preprocess_spectra(X_test, X_test.mean(axis=0))


from sklearn.model_selection import train_test_split

devices_models = []

for i, X_p in enumerate(devices_X_p):
    this_X, this_y = X_p, devices_y[i]
    X_tr, X_vl, y_tr, y_vl = train_test_split(this_X, this_y, test_size=0.2, random_state=42)
    
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, min_samples_leaf=20, random_state=42)
    multi_model = MultiOutputRegressor(model)
    multi_model.fit(X_tr, y_tr)
    y_vl_pred = multi_model.predict(X_vl)
    score = r2_score(y_vl, y_vl_pred)
    print(f"Device{i} Score: {score}")

    devices_models.append(multi_model)


model_scores = []
new_Xs = []

for i, device_model in enumerate(devices_models):
    y_vl_pred = device_model.predict(X_val_p)
    score = r2_score(y_val, y_vl_pred)
    print(f"Device{i} Score: {score}")
    model_scores.append(score)
    new_Xs.append(y_vl_pred)

top_3 = [index for score, index in sorted([(score, i) for i, score in enumerate(model_scores)], key=lambda x: x[0], reverse=True)[:3]]

print(top_3)
top_models = [devices_models[top_3[0]], devices_models[top_3[1]], devices_models[top_3[2]]]
top_Xs = [new_Xs[top_3[0]], new_Xs[top_3[1]], new_Xs[top_3[2]]]

tl_models = []

#CHANGED : Include all models instead of top_models as noone is good

for i, top_model in enumerate(devices_models): #top_models
    this_X, this_y = new_Xs[i], y_val #top_Xs[i], y_val
    X_tr, X_vl, y_tr, y_vl = train_test_split(this_X, this_y, test_size=0.2, random_state=42)
    
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, min_samples_leaf=20, random_state=42)
    multi_model = MultiOutputRegressor(model)
    multi_model.fit(X_tr, y_tr)
    y_vl_pred = multi_model.predict(X_vl)
    score = r2_score(y_vl, y_vl_pred)
    print(f"Device{i} Score: {score}")

    tl_models.append(multi_model)


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import numpy as np

#CHANGED : Include all models instead of top_models as noone is good

for i, top_model in enumerate(devices_models): #top_models
    this_X, this_y = new_Xs[i], y_val #top_Xs[i], y_val
    X_tr, X_vl, y_tr, y_vl = train_test_split(this_X, this_y, test_size=0.2, random_state=42)
    
    # --- Initial X vs Y Plot ---
    plt.figure(figsize=(10, 4))
    for f in range(min(3, this_X.shape[1])):
        plt.subplot(1, 3, f + 1)
        plt.scatter(this_X[:, f], this_y[:, 1] if this_y.ndim > 1 else this_y,
                    alpha=0.5, color="green", edgecolor="k")
        plt.xlabel(f"Feature {f}")
        plt.ylabel("Target")
        plt.title(f"Device {i} - Initial X{f} vs Y")
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # --- Train Model ---
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, min_samples_leaf=20, random_state=42)
    multi_model = MultiOutputRegressor(model)
    multi_model.fit(X_tr, y_tr)
    y_vl_pred = multi_model.predict(X_vl)
    score = r2_score(y_vl, y_vl_pred)
    print(f"Device {i} Score: {score:.4f}")

    # --- Predicted vs Actual Plot ---
    n_outputs = y_vl.shape[1] if y_vl.ndim > 1 else 1
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))

    if n_outputs == 1:
        axes = [axes]

    for j in range(n_outputs):
        ax = axes[j]
        actual = y_vl[:, j] if n_outputs > 1 else y_vl
        pred = y_vl_pred[:, j] if n_outputs > 1 else y_vl_pred
        ax.scatter(actual, pred, alpha=0.5, color="blue", edgecolor="k")
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2)
        ax.set_title(f"Target {j} - R²: {r2_score(actual, pred):.3f}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True)

    plt.suptitle(f"Device {i} - Predicted vs Actual")
    plt.show()


len(new_Xs)


model_scores = []
new_Xs = []

for i, device_model in enumerate(devices_models):
    y_vl_pred = device_model.predict(X_val_p)
    score = r2_score(y_val, y_vl_pred)
    print(f"Device{i} Score: {score}")
    model_scores.append(score)
    new_Xs.append(y_vl_pred)

print(" ")

#CHANGED : Combine all Xs
new_X = np.hstack(new_Xs)

this_X, this_y = new_X, y_val
X_tr, X_vl, y_tr, y_vl = train_test_split(this_X, this_y, test_size=0.2, random_state=42)
    
model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, min_samples_leaf=20, random_state=42)
multi_model = MultiOutputRegressor(model)
multi_model.fit(X_tr, y_tr)
y_vl_pred = multi_model.predict(X_vl)
score = r2_score(y_vl, y_vl_pred)
print(f"Score: {score}")

tl_model = model


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, r2_score
import numpy as np

# Assuming new_X and y_val are already defined
this_X, this_y = new_X, y_val

# Define the model
model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, min_samples_leaf=20, random_state=42)
multi_model = MultiOutputRegressor(model)

# Set up cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=21)

# Custom R2 score function for multi-output regression
def r2_multi(y_true, y_pred):
    return r2_score(y_true, y_pred, multioutput='uniform_average')

# Using cross-validation
cross_val_scores = cross_val_score(multi_model, this_X, this_y, cv=cv, scoring=make_scorer(r2_multi))

# Print out the cross-validation scores and the mean score
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean R2 score across all folds: {np.mean(cross_val_scores)}")





for i in [2, 3, 4, 6, 7]:
    this_X, this_y = new_Xs[i], y_val
    #print(this_X[:5], this_y[:5])
    #--- Initial X vs Y Plot ---
    plt.figure(figsize=(10, 4))
    for f in range(min(3, this_X.shape[1])):
        plt.subplot(1, 3, f + 1)
        plt.scatter(this_X[:, f], this_y[:, 1] if this_y.ndim > 1 else this_y,
                    alpha=0.5, color="green", edgecolor="k")
        plt.xlabel(f"Feature {f}")
        plt.ylabel("Target")
        plt.title(f"Device {i} - Initial X{f} vs Y")
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# predictions_list = []

# final_predictions = post_process(np.column_stack(predictions_list))

# submission_score = r2_score(y_val, final_predictions)
# print("Evaluation score on validation set: ", submission_score)


# import pickle

# with open('/kaggle/input/dig4bio-load-best-sub/best_sub.pkl', 'rb') as f:
#     my_best_sub = pickle.load(f)

# print("Unpickled variable:", my_best_sub[0], "...")

