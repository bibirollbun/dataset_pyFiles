


import numpy as np
import joblib
import os
import glob
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import pandas as pd

# --- CONFIGURATION ---
# Define the input directory containing your models
input_dir = '/kaggle/input/how-to-load-train-and-sub-data'
models_pattern = 'saved_models/XGBoost_dataset*.pkl'


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from scipy import signal
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis
import warnings
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline



warnings.filterwarnings("ignore")

# --- Data Loading and Preprocessing Functions ---

def load_and_preprocess_data(filepath, is_train=True):
    """Load and preprocess the Raman spectroscopy data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    
    if is_train:
        df = pd.read_csv(filepath)
        target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        y = df[target_cols].dropna().values
        X = df.iloc[:, :-4]
    else:
        df = pd.read_csv(filepath, header=None)
        X = df
        y = None
    
    X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1]-1)]
    X['sample_id'] = X['sample_id'].ffill()
    
    if is_train:
        X['sample_id'] = X['sample_id'].str.strip()
    else:
        X['sample_id'] = X['sample_id'].astype(str).str.strip().str.replace('sample', '').astype(int)
    
    spectral_cols = X.columns[1:]
    for col in spectral_cols:
        X[col] = X[col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        X[col] = pd.to_numeric(X[col], errors='coerce')
    return X, y


def preprocess_spectra(X, method='baseline_snv', deriv_order=1):
    """Apply spectral preprocessing techniques."""
    X_processed = X.copy()
    if method == 'baseline_snv':
        for i in range(X.shape[0]):
            poly = np.polyfit(np.arange(X.shape[1]), X[i], 3)
            baseline = np.polyval(poly, np.arange(X.shape[1]))
            X_processed[i] = X[i] - baseline
            mean, std = X_processed[i].mean(), X_processed[i].std()
            if std > 0: X_processed[i] = (X_processed[i] - mean) / std
    elif method == 'derivative':
        X_processed = signal.savgol_filter(X, window_length=21, polyorder=2, deriv=deriv_order, axis=1)
    return X_processed


def extract_peak_features(spectra):
    """Extract features based on spectral peaks."""
    features = []
    for spec in spectra:
        peaks, _ = find_peaks(spec, height=np.percentile(spec, 90), prominence=1)
        widths, _, _, _ = peak_widths(spec, peaks, rel_height=0.5)
        features.append([
            len(peaks),
            np.sum(spec[peaks]) if len(peaks) > 0 else 0,
            np.mean(spec[peaks]) if len(peaks) > 0 else 0,
            np.mean(widths) if len(widths) > 0 else 0,
        ])
    return np.array(features)

def compute_statistical_features(spectra):
    """Compute basic statistical features from spectra."""
    return np.stack([
        np.mean(spectra, axis=1), np.std(spectra, axis=1),
        skew(spectra, axis=1), kurtosis(spectra, axis=1)
    ], axis=1)


def create_required_feature_sets(X_train_array, X_test_array):
    """Generate the feature sets required for the models."""
    X_train_mean = X_train_array.mean(axis=1)
    X_test_mean = X_test_array.mean(axis=1)
    X_mean_processed = preprocess_spectra(X_train_mean, 'baseline_snv')
    X_test_mean_processed = preprocess_spectra(X_test_mean, 'baseline_snv')
    X_derivative_1 = preprocess_spectra(X_train_mean, 'derivative', deriv_order=1)
    X_test_derivative_1 = preprocess_spectra(X_test_mean, 'derivative', deriv_order=1)
    
    X_combined = np.hstack([X_mean_processed, X_derivative_1])
    X_test_combined = np.hstack([X_test_mean_processed, X_test_derivative_1])
    peak_train = extract_peak_features(X_mean_processed)
    peak_test = extract_peak_features(X_test_mean_processed)
    stat_train = compute_statistical_features(X_mean_processed)
    stat_test = compute_statistical_features(X_test_mean_processed)
    combined_all_train = np.hstack([X_mean_processed, X_derivative_1, stat_train, peak_train])
    combined_all_test = np.hstack([X_test_mean_processed, X_test_derivative_1, stat_test, peak_test])
    
    scaler = StandardScaler()
    X_mean_processed_scaled = scaler.fit_transform(X_mean_processed)
    X_test_mean_processed_scaled = scaler.transform(X_test_mean_processed)
    
    feature_sets = {
        'Combined_All': (StandardScaler().fit_transform(combined_all_train), StandardScaler().fit_transform(combined_all_test)),
        'Combined_Processed': (StandardScaler().fit_transform(X_combined), StandardScaler().fit_transform(X_test_combined)),
        'Mean_Processed': (StandardScaler().fit_transform(X_mean_processed), StandardScaler().fit_transform(X_test_mean_processed)),
        "Mean_Processed_for_PCA": (X_mean_processed_scaled, X_test_mean_processed_scaled),
    }
    return feature_sets


train_filepath = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv'
test_filepath = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv'
    
try:
    print("1. Loading data...")
    X_train_raw, y_train = load_and_preprocess_data(train_filepath, True)
    X_test_raw, _ = load_and_preprocess_data(test_filepath, False)
except FileNotFoundError as e:
    print(f"\nERROR: {e}\nPlease ensure data files are in the correct directory.")


X_train_array = X_train_raw.drop('sample_id', axis=1).values.reshape(-1, 2, 2048)
X_test_array = X_test_raw.drop('sample_id', axis=1).values.reshape(-1, 2, 2048)
print(f"Train shape: {X_train_array.shape}, Test shape: {X_test_array.shape}")

print("\n2. Generating required feature sets...")
feature_sets = create_required_feature_sets(X_train_array, X_test_array)
for name, (X_feat, _) in feature_sets.items():
    print(f"  - {name}: train shape {X_feat.shape}")


import numpy as np
import pandas as pd
import joblib
import json
from scipy.interpolate import interp1d


def load_best_model_with_metadata(dataset_id, save_dir="/kaggle/input/how-to-load-train-and-sub-data/saved_models"):
    """
    Load the best model and its metadata for a given dataset ID.
    """
    # Construct file paths
    model_path = os.path.join(save_dir, f"best_model_dataset{dataset_id}.pkl")
    metadata_path = os.path.join(save_dir, f"best_model_dataset{dataset_id}_metadata.json")
    
    # Load model
    model = joblib.load(model_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata


def match_features_to_model(X_new_input, metadata):
    """
    Modify new dataset's features so they match the dimensions of the trained model.
    - Interpolates spectral data to match wavelength range
    - Ensures correct number of features
    """
    first_wl = metadata["first_wavelength"]
    last_wl = metadata["last_wavelength"]
    num_features = metadata["num_features"]

    # --- Ensure X_new is a NumPy array ---
    if isinstance(X_new_input, tuple):
        # Some pipelines return (array, other_info)
        X_new_input = X_new_input[0]

    if hasattr(X_new_input, "values"):  # pandas DataFrame
        new_wavelengths = X_new_input.columns.astype(float).values
        new_spectra = X_new_input.values
    elif isinstance(X_new_input, np.ndarray):
        new_spectra = X_new_input
        # Assume evenly spaced wavelengths across required range
        new_wavelengths = np.linspace(first_wl, last_wl, new_spectra.shape[1])
    else:
        raise TypeError(f"Unsupported type for X_new_input: {type(X_new_input)}")

    # --- If the feature count already matches, no interpolation needed ---
    if new_spectra.shape[1] == num_features:
        return new_spectra

    # --- Target wavelength grid ---
    target_wavelengths = np.linspace(first_wl, last_wl, num_features)

    # --- Interpolate each spectrum onto the target grid ---
    processed = []
    for spectrum in new_spectra:
        f = interp1d(new_wavelengths, spectrum, kind="linear", fill_value="extrapolate")
        processed.append(f(target_wavelengths))

    processed = np.array(processed)
    return processed



def predict_with_all_best_models(X_new_features_dict, save_dir="/kaggle/input/how-to-load-train-and-sub-data/saved_models", feature_set_name="Combined_All"):
    """
    Preprocess new features to match each model's requirements and make predictions.
    Returns a dict of predictions from all best models.
    """
    train_predictions_dict = {}
    test_predictions_dict = {}

    # Get new dataset features
    X_train, X_test = X_new_features_dict[feature_set_name]

    # Loop over all available models in the save directory
    for filename in os.listdir(save_dir):
        if filename.endswith("_metadata.json"):
            dataset_id = int(filename.split("dataset")[1].split("_")[0])
            # Load model + metadata
            model, metadata = load_best_model_with_metadata(dataset_id, save_dir)



            # Match dimensions to model requirements
            X_train_processed = match_features_to_model(X_train, metadata)
            X_test_processed = match_features_to_model(X_test, metadata)

            # Predict
            preds_train = model.predict(X_train_processed)
            preds_test = model.predict(X_test_processed)
            train_predictions_dict[dataset_id] = preds_train
            test_predictions_dict[dataset_id] = preds_test

    return train_predictions_dict,test_predictions_dict



train_predictions_dict, test_predictions_dict = predict_with_all_best_models(feature_sets, feature_set_name="Mean_Processed")
print(len(train_predictions_dict))


import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

# ----------------------
# Prepare meta features
# ----------------------
def prepare_meta_features(train_predictions_dict,test_predictions_dict, y_true):
    dataset_ids1 = sorted(train_predictions_dict.keys())
    dataset_ids2 = sorted(test_predictions_dict.keys())
    X_meta_train = np.column_stack([train_predictions_dict[ds_id] for ds_id in dataset_ids1])
    X_meta_test = np.column_stack([test_predictions_dict[ds_id] for ds_id in dataset_ids2])
    return X_meta_train, X_meta_test, np.array(y_true)

X_meta_train, X_meta_test, y = prepare_meta_features(train_predictions_dict,test_predictions_dict, y_train)

# Split meta data
X_meta_split_train, X_meta_split_test, y_train_split, y_test_split = train_test_split(
    X_meta_train, y, test_size=0.2, random_state=42
)

# ----------------------
# Define meta models
# ----------------------
meta_models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'Lasso Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.1))
    ]),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale')))
    ])
}

# Single-output models (trained separately per output)
single_output_models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# ----------------------
# Training & evaluation
# ----------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, multi_output=True):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if y_pred_train.ndim == 1:
        y_pred_train = y_pred_train.reshape(-1, 1)
        y_pred_test = y_pred_test.reshape(-1, 1)

    train_r2 = [r2_score(y_train[:, i], y_pred_train[:, i]) for i in range(y.shape[1])]
    test_r2 = [r2_score(y_test[:, i], y_pred_test[:, i]) for i in range(y.shape[1])]

    return {
        'model': model,
        'train_r2': np.mean(train_r2),
        'test_r2': np.mean(test_r2),
        'train_r2_per_output': train_r2,
        'test_r2_per_output': test_r2,
        'predictions_test': y_pred_test
    }

results = {}

# Multi-output capable models
for name, model in meta_models.items():
    results[name] = evaluate_model(model, X_meta_split_train, X_meta_split_test, y_train_split, y_test_split)
    print(f"{name}: Train R²={results[name]['train_r2']:.4f}, Test R²={results[name]['test_r2']:.4f}")

# Single-output models trained separately
for name, base_model in single_output_models.items():
    preds = []
    train_r2, test_r2, models = [], [], []

    for i in range(y.shape[1]):
        model = base_model.__class__(**base_model.get_params())
        model.fit(X_meta_split_train, y_train_split[:, i])
        preds.append(model.predict(X_meta_split_test))
        train_r2.append(r2_score(y_train_split[:, i], model.predict(X_meta_split_train)))
        test_r2.append(r2_score(y_test_split[:, i], model.predict(X_meta_split_test)))
        models.append(model)

    results[name] = {
        'models': models,
        'train_r2': np.mean(train_r2),
        'test_r2': np.mean(test_r2),
        'train_r2_per_output': train_r2,
        'test_r2_per_output': test_r2,
        'predictions_test': np.column_stack(preds)
    }
    print(f"{name}: Train R²={np.mean(train_r2):.4f}, Test R²={np.mean(test_r2):.4f}")

# ----------------------
# Best model selection & saving
# ----------------------
if results:
    best_model_name, best_result = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\nBest meta-model: {best_model_name} with Test R²={best_result['test_r2']:.4f}")

    # Save best model
    joblib.dump(best_result['model'] if 'model' in best_result else best_result['models'], "best_meta_model.pkl")




import joblib
import numpy as np

# Load the saved best meta-model
loaded_model = joblib.load("best_meta_model.pkl")

# Predict on test meta-features
if isinstance(loaded_model, list):  
    # Case: single-output models (one per target)
    y_test_pred = np.column_stack([m.predict(X_meta_test) for m in loaded_model])
else:  
    # Case: multi-output model (supports predict on full matrix)
    y_test_pred = loaded_model.predict(X_meat_test)

print("Predictions shape:", y_test_pred.shape)
print("First few predictions:\n", y_test_pred[:5])



import pandas as pd
import numpy as np

# Load the sample submission file to get the correct column names
sample_submission = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv')
print(sample_submission)
# Get the exact column names from the sample submission
column_names = list(sample_submission.columns)

# Create a DataFrame from your predictions using the exact same column names (excluding 'id')
submission_df = pd.DataFrame(y_test_pred, columns=column_names[1:])  # Skip the first column which is 'id'

# Add the id column from the sample submission (using the exact same values)
submission_df.insert(0, 'ID', sample_submission['ID'])

# Verify the shape matches
print(f"Submission shape: {submission_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Verify column names match exactly
print(f"Submission columns: {list(submission_df.columns)}")
print(f"Sample columns: {list(sample_submission.columns)}")

# Check if column names are identical
print(f"Column names match: {list(submission_df.columns) == list(sample_submission.columns)}")

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully with matching column names!")


