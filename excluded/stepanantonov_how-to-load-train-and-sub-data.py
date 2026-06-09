import numpy as np
import pandas as pd
import os
import warnings


warnings.filterwarnings("ignore")
file_paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_paths.append(os.path.join(dirname, filename))
print(file_paths)


train_path = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv"
train = pd.read_csv(train_path)
train


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_file_structure(df):
    """Check if file matches the matrix-style structure with last 5 metadata columns."""
    expected_meta = {"glucose", "Na_acetate", "Mg_SO4", "MSM_present", "fold_idx"}
    cols = set(df.columns)
    return expected_meta.issubset(cols)

def summarize_files(data_dir, file_pattern="*.csv"):
    paths = glob.glob(os.path.join(data_dir, file_pattern))
    
    good_files = []
    bad_files = []
    
    for path in paths:
        try:
            df = pd.read_csv(path)
            if check_file_structure(df):
                # Matrix-style file
                good_files.append({
                    "file": os.path.basename(path),
                    "n_samples": df.shape[0],
                    "n_features": df.shape[1] - 5,  # exclude metadata
                    "wavenumber_min": float(df.columns[0]),
                    "wavenumber_max": float(df.columns[-6]), # last before metadata
                    "metadata_cols": df.columns[-5:].tolist()
                })
            else:
                bad_files.append(os.path.basename(path))
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
            bad_files.append(os.path.basename(path))
    
    good_summary = pd.DataFrame(good_files)
    
    print("\n✅ Files matching matrix-style structure:\n", good_summary)
    print("\n❌ Files with a different structure:\n", bad_files)
    
    return good_summary, bad_files

data_dir = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"
good_summary, bad_files = summarize_files(data_dir)




import pandas as pd
import numpy as np
import os
from pathlib import Path

def process_csv_files(directory_path, target_cols=None, skip_files=None):
    """
    Process CSV files in a directory, extracting wavenumber data and target labels.
    
    Args:
        directory_path (str): Path to directory containing CSV files
        target_cols (list): List of target column names to extract
    
    Returns:
        tuple: (X_list, y_list, processed_files) where:
            X_list: List of DataFrames with wavenumber data
            y_list: List of DataFrames with target labels
            processed_files: List of successfully processed file names
    """
    if target_cols is None:
        target_cols = ['glucose', 'Na_acetate', 'Mg_SO4']
    
    if skip_files is None:
        skip_files = []
    
    directory = Path(directory_path)
    csv_files = list(directory.glob('*.csv'))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in directory: {directory_path}")

    X_list = []
    y_list = []
    processed_files = []
    
    files_to_process = len(csv_files)  # Process up to 8 files

    for i, file_path in enumerate(csv_files[:files_to_process]):
        # Skip files based on filename patterns
        skip_this_file = False
        for pattern in skip_files:
            if pattern in file_path.name:
                print(f"Skipping {file_path.name} (matches pattern: '{pattern}')")
                skip_this_file = True
                break

        if skip_this_file:
            continue
        try:
            print(f"Processing file {i+1}/{files_to_process}: {file_path.name}")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if file has at least 6 columns (wavenumbers + 5 target columns)
            if len(df.columns) < 6:
                print(f"  Skipping {file_path.name}: Not enough columns ({len(df.columns)})")
                continue
            
            # Identify the split point (last 5 columns are targets)
            split_idx = len(df.columns) - 5
            
            # Extract wavenumber columns (first n-5 columns)
            wavenumber_cols = df.columns[:split_idx]
            
            # Check if wavenumber columns contain numeric data
            try:
                # Convert wavenumber columns to numeric, coercing errors
                wavenumber_data = df[wavenumber_cols].apply(pd.to_numeric, errors='coerce')
                
                # Check for any non-numeric values that couldn't be converted
                if wavenumber_data.isna().any().any():
                    print(f"  Warning: {file_path.name} contains non-numeric values in wavenumber columns")
                    continue
                
            except Exception as e:
                print(f"  Skipping {file_path.name}: Error converting wavenumber columns to numeric - {e}")
                continue
            
            # Check wavenumber range (approximately 200-2000)
            try:
                # Convert column names to numeric for range checking
                wavenumber_values = pd.to_numeric(wavenumber_cols, errors='coerce')
                
                # Check if most wavenumbers are in the expected range
                valid_wavenumbers = wavenumber_values[(wavenumber_values >= 150) & (wavenumber_values <= 2100)]
                if len(valid_wavenumbers) / len(wavenumber_values) < 0.8:  # At least 80% in range
                    print(f"  Warning: {file_path.name} wavenumbers may not be in expected range (200-2000)")
                
            except:
                print(f"  Warning: Could not validate wavenumber range for {file_path.name}")
            
            # Extract target columns (last 5 columns)
            target_data = df.iloc[:, split_idx:]
            
            # Check if our expected target columns exist in the last 5 columns
            available_targets = [col for col in target_cols if col in target_data.columns]
            
            if len(available_targets) == 0:
                print(f"  Skipping {file_path.name}: None of the target columns found in last 5 columns")
                print(f"  Available columns: {list(target_data.columns)}")
                continue
            
            # Extract only the available target columns
            y = target_data[available_targets]
            
            # Store the data
            X_list.append(wavenumber_data)
            y_list.append(y)
            processed_files.append(file_path.name)
            
            print(f"  Successfully processed: {file_path.name}")
            print(f"  Wavenumber columns: {len(wavenumber_cols)}")
            print(f"  Target columns found: {available_targets}")
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    print(f"\nProcessing complete. Successfully processed {len(X_list)} files.")
    
    return X_list, y_list, processed_files


directory_path = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"
skip_files = ['sample_submission', '96_samples', 'transfer_plate']
X_list, y_list, processed_files = process_csv_files(directory_path, skip_files = skip_files)



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

column_headings_list = [X_df.columns.tolist() for X_df in X_list]

X_arrays = [X_df.values for X_df in X_list]  # Convert to numpy arrays
X_processed_arrays = [preprocess_spectra(X_arr, method='baseline_snv') for X_arr in X_arrays]

X_processed_dfs = []
for i, (X_array, original_columns) in enumerate(zip(X_processed_arrays, column_headings_list)):
    X_processed_df = pd.DataFrame(X_array, columns=original_columns)
    X_processed_dfs.append(X_processed_df)
    print(f"Array {i}: Converted back to DataFrame with shape {X_processed_df.shape}")


from scipy import signal
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from pprint import pprint

def extract_peak_features_with_wavenumbers(spectra_df):
    """
    Extract peak features using actual wavenumber values.
    """
    spectra_array = spectra_df.values
    wavenumbers = spectra_df.columns.astype(float).values  # Get actual wavenumbers
    
    features = []
    for spec in spectra_array:
        peaks, _ = find_peaks(spec, height=np.percentile(spec, 90), prominence=1)
        
        if len(peaks) > 0:
            # Use actual wavenumbers for peak positions and widths
            peak_wavenumbers = wavenumbers[peaks]
            peak_intensities = spec[peaks]
            
            # Calculate widths using actual wavenumber scale
            widths, width_heights, left_ips, right_ips = peak_widths(
                spec, peaks, rel_height=0.5
            )
            # Convert width from index units to wavenumber units
            width_wavenumbers = wavenumbers[np.round(right_ips).astype(int)] - wavenumbers[np.round(left_ips).astype(int)]
            
            features.append([
                len(peaks),
                np.sum(peak_intensities),
                np.mean(peak_intensities),
                np.mean(width_wavenumbers),  # Actual wavenumber width
                np.mean(peak_wavenumbers),   # Mean peak position
                np.std(peak_wavenumbers)     # Spread of peak positions
            ])
        else:
            features.append([0, 0, 0, 0, 0, 0])
    
    return np.array(features)

def compute_statistical_features(spectra_df):
    """
    Compute basic statistical features from spectral DataFrame.
    Already correctly handles non-uniform wavenumber spacing.
    """
    spectra_array = spectra_df.values
    
    return np.stack([
        np.mean(spectra_array, axis=1),
        np.std(spectra_array, axis=1),
        skew(spectra_array, axis=1),
        kurtosis(spectra_array, axis=1)
    ], axis=1)

def create_required_feature_sets_from_list(X_train_list):
    """
    Generate feature sets required for the models from a list of DataFrames.
    Each DataFrame should have rows = spectra, cols = wavenumbers, values = intensities.
    Returns a list of feature set dictionaries.
    """
    feature_sets_list = []
    
    for spectra_df in X_train_list:
        # Convert to NumPy array
        spectra_array = spectra_df.values
        
        # --- Step 1: Mean spectra ---
        X_mean = spectra_array
        X_mean_processed = preprocess_spectra(X_mean, 'baseline_snv')
        X_derivative_1 = preprocess_spectra(X_mean, 'derivative', deriv_order=1)
        
        # --- Step 2: Combined versions ---
        X_combined = np.hstack([X_mean_processed, X_derivative_1])
        
        # --- Step 3: Peak + Statistical features ---
        peak_features = extract_peak_features_with_wavenumbers(spectra_df)
        stat_features = compute_statistical_features(spectra_df)
        
        combined_all = np.hstack([X_mean_processed, X_derivative_1, stat_features, peak_features])
        
        # --- Step 4: Scaling ---
        scaler = StandardScaler()
        X_mean_processed_scaled = scaler.fit_transform(X_mean_processed)
        
        # Build dictionary for this DataFrame
        feature_sets = {
            'Combined_All': StandardScaler().fit_transform(combined_all),
            'Combined_Processed': StandardScaler().fit_transform(X_combined),
            'Mean_Processed': StandardScaler().fit_transform(X_mean_processed),
            "Mean_Processed_for_PCA": X_mean_processed_scaled,
        }
        
        feature_sets_list.append(feature_sets)
    
    return feature_sets_list

feature_sets = create_required_feature_sets_from_list(X_list)
for feature_set in feature_sets:
    # Number of key-value pairs
    print(f"Number of items: {len(feature_set['Mean_Processed'][0])}")
    # Keys in the dictionary
    print(f"Keys: {list(feature_set.keys())}")


import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor


def train_and_save_best_models(X_features_list, Y_list, feature_set_name="Combined_All", save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    results = []
    best_models = {}

    for i, (feature_dict, y_df) in enumerate(zip(X_features_list, Y_list)):
        print(f"\n--- Dataset {i+1} ---")

        # Select feature set
        X = feature_dict[feature_set_name]
        y = y_df.values

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define models
        models = {
            "RandomForest": MultiOutputRegressor(RandomForestRegressor(
                n_estimators=300, random_state=42, n_jobs=-1
            )),
            "PLS": PLSRegression(n_components=min(20, X_train.shape[1])),
            "XGBoost": MultiOutputRegressor(XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                tree_method="gpu_hist", predictor="gpu_predictor", random_state=42
            ))
        }

        best_model = None
        best_r2 = -np.inf
        best_name = None

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred, multioutput="uniform_average")
            print(f"{name} R²: {r2:.3f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

        # Save only the best model
        best_model_path = os.path.join(save_dir, f"best_model_dataset{i+1}.pkl")
        joblib.dump(best_model, best_model_path)

        # Handle wavelength info depending on X type
        if hasattr(X, "columns"):  # pandas DataFrame
            try:
                wavelengths = X.columns.astype(float).values
            except Exception:
                wavelengths = np.arange(X.shape[1])
        else:  # numpy array
            wavelengths = np.arange(X.shape[1])

        metadata = {
            "dataset": i + 1,
            "best_model_name": best_name,
            "best_r2": best_r2,
            "first_wavelength": float(wavelengths[0]),
            "last_wavelength": float(wavelengths[-1]),
            "num_features": X.shape[1],
            "model_path": best_model_path
        }

        metadata_path = os.path.join(save_dir, f"best_model_dataset{i+1}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Best model for dataset {i+1}: {best_name} (R²={best_r2:.3f}) saved to {best_model_path}")
        print(f"   Metadata saved to {metadata_path}")

        results.append(metadata)
        best_models[i + 1] = {"model": best_model, "metadata": metadata}

    return pd.DataFrame(results), best_models


results_df, best_models = train_and_save_best_models(feature_sets, y_list, feature_set_name="Combined_All")
print(results_df)

