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

def create_tuned_models():
    """Create the top 3 tuned models with their optimal parameters."""
    
    models = []
    models.append({'name': 'Combined_All_ExtraTrees', 'feature_set': 'Combined_All', 'target_models': [ExtraTreesRegressor(n_estimators=511, max_depth=13, min_samples_split=6, min_samples_leaf=1, random_state=42, n_jobs=-1), ExtraTreesRegressor(n_estimators=436, max_depth=44, min_samples_split=3, min_samples_leaf=1, random_state=42, n_jobs=-1), ExtraTreesRegressor(n_estimators=627, max_depth=46, min_samples_split=6, min_samples_leaf=7, random_state=42, n_jobs=-1)]})
    models.append({'name': 'Combined_Processed_ExtraTrees', 'feature_set': 'Combined_Processed', 'target_models': [ExtraTreesRegressor(n_estimators=555, max_depth=14, min_samples_split=4, min_samples_leaf=1, random_state=42, n_jobs=-1), ExtraTreesRegressor(n_estimators=861, max_depth=26, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1), ExtraTreesRegressor(n_estimators=344, max_depth=30, min_samples_split=14, min_samples_leaf=7, random_state=42, n_jobs=-1)]})
    n_comp = [61, 63, 62]
    models.append({
        "name": "PCA_LR",
        "target_models": [Pipeline([("pca", PCA(n_components=n_comp[i], random_state=42)), ("lr", LinearRegression())
        ]) for i in range(3)],
        "feature_set": "Mean_Processed_for_PCA",  
        })
    
    return models

def evaluate_and_train_models(tuned_models, feature_sets, y_train):
    """Evaluate tuned models with cross-validation and create ensemble predictions."""
    from sklearn.metrics import r2_score
    print("="*60, "\nEVALUATING TUNED MODELS WITH 5-FOLD CROSS-VALIDATION\n", "="*60)
    target_names = ['Glucose', 'Sodium Acetate', 'Magnesium Acetate']
    all_predictions = {target: [] for target in range(3)}
    all_cv_predictions = {target: [] for target in range(3)}
    all_cv_targets = {target: [] for target in range(3)}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_scores = []
    
    for model_config in tuned_models:
        model_name, feature_set_name, target_models = model_config['name'], model_config['feature_set'], model_config['target_models']
        print(f"\nEvaluating {model_name} on {feature_set_name} features:")
        X_train, X_test = feature_sets[feature_set_name]
        target_r2_scores, target_cv_preds, target_cv_targets = [], {t: [] for t in range(3)}, {t: [] for t in range(3)}
        
        for target_idx in range(3):
            model = target_models[target_idx]
            r2_scores, cv_preds, cv_targets = [], [], []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx, target_idx], y_train[val_idx, target_idx]
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                r2_scores.append(r2_score(y_val, y_pred))
                cv_preds.extend(y_pred if isinstance(y_pred, list) else y_pred.tolist())
                cv_targets.extend(y_val if isinstance(y_val, list) else y_val.tolist())
            
            avg_r2 = np.mean(r2_scores)
            target_r2_scores.append(avg_r2)
            target_cv_preds[target_idx] = np.array(cv_preds)
            target_cv_targets[target_idx] = np.array(cv_targets)
            print(f"  {target_names[target_idx]:<18} | CV R²: {avg_r2:.4f} ± {np.std(r2_scores):.4f}")

            
            model.fit(X_train, y_train[:, target_idx])
            all_predictions[target_idx].append(model.predict(X_test))
        
        for target_idx in range(3):
            all_cv_predictions[target_idx].append(target_cv_preds[target_idx])
            if len(all_cv_targets[target_idx]) == 0:
                all_cv_targets[target_idx] = target_cv_targets[target_idx]
        
        model_scores.append({'model': model_name, 'feature_set': feature_set_name, 'overall_r2': np.mean(target_r2_scores), 'target_scores': target_r2_scores})
        print(f"  Overall Avg R²: {np.mean(target_r2_scores):.4f}")
    
    print("\n" + "="*60, "\nTUNED MODEL PERFORMANCE SUMMARY\n", "="*60)
    for i, score in enumerate(sorted(model_scores, key=lambda x: x['overall_r2'], reverse=True)):
        print(f"{i+1}. {score['model']:<35} | Overall R²: {score['overall_r2']:.4f}")
    
    return all_predictions, all_cv_predictions, all_cv_targets, model_scores

def main_train():
    """Main execution pipeline for training."""
    print("="*80, "\nCELL 1: MODEL TRAINING PIPELINE\n", "="*80)
    
    train_filepath = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv'
    test_filepath = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv'
    
    try:
        print("1. Loading data...")
        X_train_raw, y_train = load_and_preprocess_data(train_filepath, True)
        X_test_raw, _ = load_and_preprocess_data(test_filepath, False)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}\nPlease ensure data files are in the correct directory.")
        return

    X_train_array = X_train_raw.drop('sample_id', axis=1).values.reshape(-1, 2, 2048)
    X_test_array = X_test_raw.drop('sample_id', axis=1).values.reshape(-1, 2, 2048)
    print(f"Train shape: {X_train_array.shape}, Test shape: {X_test_array.shape}")

    print("\n2. Generating required feature sets...")
    feature_sets = create_required_feature_sets(X_train_array, X_test_array)
    for name, (X_feat, _) in feature_sets.items():
        print(f"  - {name}: train shape {X_feat.shape}")

    print("\n3. Creating tuned models...")
    tuned_models = create_tuned_models()
    print(f"Created {len(tuned_models)} top-performing model configurations.")

    print("\n4. Evaluating and training tuned models...")
    all_predictions, all_cv_predictions, all_cv_targets, model_scores = evaluate_and_train_models(tuned_models, feature_sets, y_train)
    
    print("\n5. Saving training results for ensembling...")
    output_dir = "training_outputs"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(all_predictions, os.path.join(output_dir, 'all_predictions.joblib'))
    joblib.dump(all_cv_predictions, os.path.join(output_dir, 'all_cv_predictions.joblib'))
    joblib.dump(all_cv_targets, os.path.join(output_dir, 'all_cv_targets.joblib'))
    joblib.dump(model_scores, os.path.join(output_dir, 'model_scores.joblib'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.joblib'))
    X_test_raw.to_csv(os.path.join(output_dir, 'X_test_raw.csv'), index=False)
    
    print(f"Results saved to '{output_dir}' directory.")
    print("\n" + "="*80, "\nCELL 1 COMPLETED SUCCESSFULLY!\n", "="*80)

if __name__ == "__main__":
    main_train()





import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge as RidgeReg
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error
import scipy.optimize as opt
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

# --- Ensembling and Post-Processing Functions ---

def create_advanced_ensemble(all_predictions, all_cv_predictions, all_cv_targets, model_scores):
    """Create advanced ensemble using multiple methods."""
    print("\n" + "="*60, "\nCREATING ADVANCED ENSEMBLE PREDICTIONS\n", "="*60)
    target_names = ['Glucose', 'Sodium Acetate', 'Magnesium Acetate']
    n_models = len(all_predictions[0])
    ensemble_methods, all_ensemble_predictions = {}, {}
    
    for target_idx in range(3):
        print(f"\nOptimizing ensemble for {target_names[target_idx]}:")
        cv_preds_matrix = np.column_stack(all_cv_predictions[target_idx])
        cv_targets = all_cv_targets[target_idx]
        test_preds_matrix = np.column_stack(all_predictions[target_idx])
        
        individual_scores = [s['target_scores'][target_idx] for s in model_scores]
        perf_weights = np.maximum(np.array(individual_scores), 0)
        perf_weights /= np.sum(perf_weights) if np.sum(perf_weights) > 0 else 1
        
        def objective(w):
            return mean_squared_error(cv_targets, np.dot(cv_preds_matrix, w / np.sum(w)))
        res = opt.minimize(objective, np.ones(n_models)/n_models, method='SLSQP', bounds=[(0,1)]*n_models, constraints={'type':'eq','fun':lambda w:np.sum(w)-1})
        opt_weights = res.x / np.sum(res.x)
        
        ridge_meta = RidgeReg(alpha=1.0).fit(cv_preds_matrix, cv_targets)
        linear_meta = LinearRegression().fit(cv_preds_matrix, cv_targets)
        rf_meta = RFR(n_estimators=100, max_depth=5, random_state=42).fit(cv_preds_matrix, cv_targets)
        
        methods = {'Simple_Average': (np.mean(cv_preds_matrix, axis=1), np.mean(test_preds_matrix, axis=1)), 'Weighted_Average': (np.dot(cv_preds_matrix, perf_weights), np.dot(test_preds_matrix, perf_weights)), 'Optimal_Weights': (np.dot(cv_preds_matrix, opt_weights), np.dot(test_preds_matrix, opt_weights)), 'Ridge_Stacking': (ridge_meta.predict(cv_preds_matrix), ridge_meta.predict(test_preds_matrix)), 'Linear_Stacking': (linear_meta.predict(cv_preds_matrix), linear_meta.predict(test_preds_matrix)), 'RF_Stacking': (rf_meta.predict(cv_preds_matrix), rf_meta.predict(test_preds_matrix))}
        
        method_scores = {name: r2_score(cv_targets, pred[0]) for name, pred in methods.items()}
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("  Ensemble Method Performance (CV R²):")
        for name, score in sorted_methods:
            print(f"    - {name:<20} | {score:.4f}")
        
        best_method_name = sorted_methods[0][0]
        ensemble_methods[target_idx] = {'method': best_method_name, 'cv_score': method_scores[best_method_name], 'top_3': sorted_methods[:3]}
        all_ensemble_predictions[target_idx] = {name: methods[name][1] for name, _ in sorted_methods}

    return ensemble_methods, all_ensemble_predictions

def post_process_predictions(preds, y_train):
    """Apply post-processing to refine predictions."""
    print("Post-processing predictions...")
    processed_preds = np.maximum(preds, 0)
    for i in range(processed_preds.shape[1]):
        lower, upper = np.percentile(y_train[:, i], 1), np.percentile(y_train[:, i], 99)
        processed_preds[:, i] = np.clip(processed_preds[:, i], lower, upper)
    print("Post-processing complete.")
    return processed_preds

def save_top_3_submissions(all_ensemble_predictions, ensemble_methods, X_test_raw, y_train):
    """Save submissions for top 3 overall ensemble methods."""
    print("\n" + "="*60, "\nSAVING TOP 3 ENSEMBLE SUBMISSIONS\n", "="*60)
    
    method_avg_scores = {}
    all_methods = set(m for t in range(3) for m, _ in ensemble_methods[t]['top_3'])
    
    for method_name in all_methods:
        scores = [r2_score(y_train[:, t], all_ensemble_predictions[t][method_name]) for t in range(3)]
        method_avg_scores[method_name] = np.mean(scores)
        
    top_3_overall = sorted(method_avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("Top 3 ensemble methods overall (by average R² score):")
    for i, (method_name, avg_score) in enumerate(top_3_overall):
        print(f"  {i+1}. {method_name:<20} | Avg R²: {avg_score:.4f}")
    
    for rank, (method_name, _) in enumerate(top_3_overall, 1):
        print(f"\nCreating submission {rank}: {method_name}")
        submission_preds = np.zeros((len(X_test_raw['sample_id'].unique()), 3))
        for target_idx in range(3):
            submission_preds[:, target_idx] = all_ensemble_predictions[target_idx][method_name]
        
        final_preds = post_process_predictions(submission_preds, y_train)
        
        # --- FIX IS HERE ---
        # Changed 'id' to 'ID' to match competition format
        sub = pd.DataFrame({'ID': X_test_raw['sample_id'].unique(), 'Glucose': final_preds[:, 0], 'Sodium Acetate': final_preds[:, 1], 'Magnesium Sulfate': final_preds[:, 2]})
        filename = f'submission_{rank}_{method_name.lower()}.csv'
        sub.to_csv(filename, index=False)
        print(f"  Saved: {filename}")

    print(f"\n✅ Saved {len(top_3_overall)} submission files!")

def main_ensemble():
    """Main execution pipeline for ensembling."""
    print("="*80, "\nCELL 2: ENSEMBLING AND SUBMISSION PIPELINE\n", "="*80)
    
    input_dir = "training_outputs"
    try:
        print("1. Loading training results...")
        all_predictions = joblib.load(os.path.join(input_dir, 'all_predictions.joblib'))
        all_cv_predictions = joblib.load(os.path.join(input_dir, 'all_cv_predictions.joblib'))
        all_cv_targets = joblib.load(os.path.join(input_dir, 'all_cv_targets.joblib'))
        model_scores = joblib.load(os.path.join(input_dir, 'model_scores.joblib'))
        y_train = joblib.load(os.path.join(input_dir, 'y_train.joblib'))
        X_test_raw = pd.read_csv(os.path.join(input_dir, 'X_test_raw.csv'))
        print("Training results loaded successfully.")
    except FileNotFoundError:
        print(f"\nERROR: Training output files not found in '{input_dir}'.")
        print("Please run Cell 1 first to generate the necessary files.")
        return

    print("\n2. Creating and evaluating advanced ensembles...")
    ensemble_methods, all_ensemble_predictions = create_advanced_ensemble(all_predictions, all_cv_predictions, all_cv_targets, model_scores)

    print("\n3. Generating and saving top 3 submission files...")
    save_top_3_submissions(all_ensemble_predictions, ensemble_methods, X_test_raw, y_train)
    
    print("\n" + "="*80, "\nCELL 2 COMPLETED SUCCESSFULLY!\n", "="*80)

if __name__ == "__main__":
    main_ensemble()



import pandas as pd
import numpy as np
import os
import glob

def main_final_weighted_ensemble():
    """
    Loads the top 3 submission files, combines them with weighted averaging,
    and creates a final submission.csv file with the correct 'ID' column.
    Priority: RF > Linear > Ridge
    """
    print("="*80)
    print("CELL 3: FINAL WEIGHTED ENSEMBLE")
    print("="*80)

    # --- 1. Find the submission files ---
    submission_files = glob.glob('submission_*.csv')
    submission_files = [f for f in submission_files if not f.startswith('submission.csv')]
    
    if len(submission_files) != 3:
        print(f"Error: Expected 3 submission files, but found {len(submission_files)}.")
        print("Please ensure Cell 2 has been run successfully.")
        print("Found files:", submission_files)
        return

    print("1. Found the following submission files to ensemble:")
    for f in submission_files:
        print(f"   - {f}")

    # --- 2. Load the files ---
    dfs = [pd.read_csv(file) for file in submission_files]

    # --- 3. Weighted averaging ---
    print("\n2. Applying weighted averaging (RF=0.5, Linear=0.3, Ridge=0.2)...")
    
    final_submission_df = dfs[0].copy()
    if 'id' in final_submission_df.columns:
        final_submission_df.rename(columns={'id': 'ID'}, inplace=True)
    
    target_cols = ['Glucose', 'Sodium Acetate', 'Magnesium Sulfate']
    
    # Giả định thứ tự file là RF, Linear, Ridge
    weights = [0.5, 0.25, 0.25]
    
    weighted_preds = sum(w * df[target_cols].values for w, df in zip(weights, dfs))
    final_submission_df[target_cols] = weighted_preds
    
    print("Weighted averaging complete.")

    # --- 4. Save ---
    final_filename = 'submission.csv'
    final_submission_df.to_csv(final_filename, index=False)
    
    print(f"\n3. Final weighted submission file saved as '{final_filename}'")
    print("\nPreview of final submission:")
    print(final_submission_df.head())
    
    print("\n" + "="*80)
    print("CELL 3 COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main_final_weighted_ensemble()


