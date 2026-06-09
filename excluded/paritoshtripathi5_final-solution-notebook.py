import numpy as np
import pandas as pd
import os


def load_comp_data(filepath, is_train=True):
    """Load and preprocess the Raman spectroscopy data"""
    if is_train:
        df = pd.read_csv(filepath)
        # Extract target variables
        target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        y = df[target_cols].dropna().values
        
        # Process spectral data
        X = df.iloc[:, :-4] # Remove last 4 columns (analyte info and targets)
    else:
        df = pd.read_csv(filepath, header=None)
        X = df
        y = None
    
    # Set column names
    X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1]-1)]
    
    # Fill sample_id using forward fill
    X['sample_id'] = X['sample_id'].ffill()
    
    # Clean sample_id
    if is_train:
        X['sample_id'] = X['sample_id'].str.strip()
    else:
        X['sample_id'] = X['sample_id'].str.strip().str.replace('sample', '').astype(int)
    
    # Clean spectral data (remove brackets)
    spectral_cols = X.columns[1:]
    for col in spectral_cols:
        X[col] = X[col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.drop('sample_id', axis=1).values.reshape(-1, 2, 2048).mean(axis=1)
    return X, y


!pip install -q pybaselines


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
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
            max_half_window=20,
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

def preprocess_spectra(spectra):
    msced = apply_msc(spectra)
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


WINDOW_LENGTH = 7
POLY_ORDER = 2
DERIV = 0

X_val, y_val = load_comp_data('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv', is_train=True)
X_test, _ = load_comp_data('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv', is_train=False)

X_val = fix_val_test_shape(X_val)
X_test = fix_val_test_shape(X_test)

X_val_p = preprocess_spectra(X_val)
X_test_p = preprocess_spectra(X_test)

# Train on val data with preprocessing
model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, min_samples_leaf=20, random_state=42)
multi_model = MultiOutputRegressor(model)
multi_model.fit(X_val_p, y_val)
y_test_pred = multi_model.predict(X_test_p)

final_predictions = post_process(y_test_pred)


print("\n\n CREATING SUBMISSION")
print("="*80)

# Load submission template
submission = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv')

# Fill with predictions
submission['Glucose'] = final_predictions[:, 0]
submission['Sodium Acetate'] = final_predictions[:, 1]
submission['Magnesium Sulfate'] = final_predictions[:, 2]  # Note: Using Magnesium Acetate predictions

# Validation
print("\nSubmission Validation:")
print(f"Shape: {submission.shape}")
print(f"Any NaN values: {submission.isna().sum().sum()}")
print(f"Any negative values: {(submission[['Glucose', 'Sodium Acetate', 'Magnesium Sulfate']] < 0).sum().sum()}")

# Save submission
submission.to_csv(f'submission_pp_hgb_{WINDOW_LENGTH}_{POLY_ORDER}_{DERIV}.csv', index=False)
print("\nSubmission saved successfully!")

print("\nFirst 10 rows of submission:")
print(submission.head(10))

print("\nLast 10 rows of submission:")
print(submission.tail(10))

