from IPython.display import Image, display

img_path = "/kaggle/input/polymer-version-2/polymer_2.png"

display(Image(filename=img_path))


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


import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer, KNNImputer # KNNImputer is an option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error # Assuming RMSE is a key metric, though competition uses multiple
import gc
import traceback
from scipy.stats import uniform, randint # For RandomizedSearchCV distributions

try:
    print("ğŸš€ Starting Polymer Prediction Workflow...")
    
    # --- File listing for debugging ---
    print("\n--- Listing Kaggle Input Files ---")
    input_dir = '/kaggle/input'
    if os.path.exists(input_dir):
        for dirname, _, filenames in os.walk(input_dir):
            print(f"Directory: {dirname}")
            for filename in filenames:
                print(f"  {filename}")
    else:
        print(f"Input directory '{input_dir}' not found. Please ensure correct Kaggle environment.")

    # --- Load datasets ---
    print("\n--- Loading Datasets ---")
    train_path = '/kaggle/input/neurips-polymer-train-descriptors/train_descriptors.csv'
    test_path = '/kaggle/input/neurips-polymer-train-descriptors/test_descriptors.csv'
    sample_sub_path = '/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv'

    # Check if files exist before loading
    if not os.path.exists(train_path): raise FileNotFoundError(f"Train data not found at {train_path}")
    if not os.path.exists(test_path): raise FileNotFoundError(f"Test data not found at {test_path}")
    if not os.path.exists(sample_sub_path): raise FileNotFoundError(f"Sample submission not found at {sample_sub_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_sub_path)

    print(f"Train data loaded. Shape: {train_df.shape}")
    print(f"Test data loaded. Shape: {test_df.shape}")
    print(f"Sample submission loaded. Shape: {sample_submission.shape}")

    # --- Define target and feature columns ---
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Identify all unique columns that are not 'id' or target columns across both datasets
    all_cols = pd.Index(train_df.columns).union(test_df.columns)
    
    # Ensure 'SMILES' is excluded if present
    exclude_cols = set(TARGET_COLS + ['id', 'SMILES'])
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    print(f"\nIdentified {len(feature_cols)} features for modeling.")
    # print(f"Sample Features: {feature_cols[:5]}...") # Uncomment for debugging

    # --- Feature cleaning function (now integrated into preprocessing if needed) ---
    # This robust_clean is more for initial data preparation to ensure all features exist and are numeric
    max_threshold = 1e10
    def robust_initial_clean(df, features):
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        # Add missing features present in `features` but not in `df`
        missing_in_df = set(features) - set(df_copy.columns)
        for col in missing_in_df:
            # print(f"Adding missing feature column to df: {col}") # Debugging
            df_copy[col] = np.nan
            
        # Ensure only specified features are used and handle inf/-inf and extreme values
        X = df_copy[features].replace([np.inf, -np.inf], np.nan)
        X = X.mask(X.abs() > max_threshold, np.nan)
        
        # Convert all feature columns to numeric, coercing errors to NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        return X

    print("\n--- Applying robust initial cleaning to features ---")
    X_train_processed = robust_initial_clean(train_df, feature_cols)
    X_test_processed = robust_initial_clean(test_df, feature_cols)
    
    # Align columns just in case (e.g., if one dataframe had features not in the combined list)
    X_train_processed = X_train_processed[feature_cols]
    X_test_processed = X_test_processed[feature_cols]

    y_train = train_df[TARGET_COLS].copy() # Ensure y_train is a copy

    print(f"Cleaned X_train_processed shape: {X_train_processed.shape}")
    print(f"Cleaned X_test_processed shape: {X_test_processed.shape}")
    
    del train_df # Free up memory
    del test_df
    gc.collect()

    # --- Model Training with K-Fold Cross-Validation and Pipeline ---
    models = {}
    oof_preds = pd.DataFrame(index=X_train_processed.index)
    test_preds_agg = pd.DataFrame({'id': sample_submission['id']})

    N_SPLITS = 5 # Number of folds for cross-validation
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # LightGBM Parameters - A good starting point, consider tuning further
    # These are more robust for general datasets and can be improved with GridSearchCV/RandomizedSearchCV
    lgb_params = {
        'objective': 'regression_l1', # MAE objective, often more robust to outliers than MSE
        'metric': 'mae',
        'n_estimators': 2000,         # Increase and let early stopping do its job
        'learning_rate': 0.02,        # Slightly smaller learning rate
        'feature_fraction': 0.8,      # Subsample features
        'bagging_fraction': 0.8,      # Subsample data
        'bagging_freq': 1,            # Perform bagging at every iteration
        'lambda_l1': 0.1,             # L1 regularization
        'lambda_l2': 0.1,             # L2 regularization
        'num_leaves': 64,             # More leaves for higher complexity
        'verbose': -1,                # Suppress verbose output during training
        'n_jobs': -1,                 # Use all available cores
        'seed': 42,
        'boosting_type': 'gbdt',
        'early_stopping_round': 100 # Moved here for consistency with lgb.train API if used directly
    }

    print(f"\n--- Training Models for {len(TARGET_COLS)} Targets with {N_SPLITS}-Fold Cross-Validation ---")

    for target in TARGET_COLS:
        print(f"\nâœ¨ Training model for target: {target}")
        
        # --- Create a preprocessing and modeling pipeline ---
        # Using SimpleImputer for speed and robustness, KNNImputer is an option for potentially better accuracy
        # imputer_strategy = 'median' # or 'mean' or 'constant'
        # imputer = SimpleImputer(strategy=imputer_strategy) 
        # For KNNImputer (more computationally intensive, but can be better):
        # imputer = KNNImputer(n_neighbors=5, weights='uniform') 
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', lgb.LGBMRegressor(**lgb_params))
        ])

        # Filter out NaNs for the current target in training data
        target_train_mask = ~y_train[target].isnull()
        X_train_target = X_train_processed[target_train_mask]
        y_train_target = y_train[target][target_train_mask]
        
        fold_preds = np.zeros(len(X_train_target))
        test_fold_preds = np.zeros(len(X_test_processed))
        
        oof_fold_idx = X_train_target.index.values # Original indices for OOF prediction
        
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_target, y_train_target)):
            print(f"  Fold {fold + 1}/{N_SPLITS} for {target}...")
            
            X_tr, X_val = X_train_target.iloc[train_idx], X_train_target.iloc[val_idx]
            y_tr, y_val = y_train_target.iloc[train_idx], y_train_target.iloc[val_idx]

            # Fit the pipeline
            pipeline.fit(X_tr, y_tr,
                         regressor__eval_set=[(pipeline['scaler'].fit_transform(pipeline['imputer'].fit_transform(X_val)), y_val)],
                         regressor__eval_metric='mae', # Ensure eval_metric is passed
                         regressor__callbacks=[lgb.log_evaluation(period=200)] # Log every 200 rounds
                        )
            
            # Store OOF predictions
            val_preds = pipeline.predict(X_val)
            fold_preds[val_idx] = val_preds
            
            # Predict on test set
            test_fold_preds += pipeline.predict(X_test_processed) / N_SPLITS
            
            # Evaluate fold performance
            fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            print(f"    Fold {fold + 1} RMSE: {fold_rmse:.4f}")
            fold_scores.append(fold_rmse)

        oof_preds[target] = pd.Series(fold_preds, index=oof_fold_idx) # Align OOF preds correctly
        test_preds_agg[target] = test_fold_preds
        models[target] = pipeline # Store the last trained pipeline for this target (or train a final model on full data)

        print(f"âœ¨ Target {target} completed. Average OOF RMSE: {np.mean(fold_scores):.4f} Â± {np.std(fold_scores):.4f}")
        del X_train_target, y_train_target, fold_preds, test_fold_preds
        gc.collect()
        
    print("\n--- Cross-Validation complete for all targets ---")
    
    # Optional: If you want to use the OOF predictions for stacking, you now have them in oof_preds.
    # print(f"OOF Predictions Head:\n{oof_preds.head()}")

    # --- Create submission from sample template ---
    print("\n--- Creating Submission File ---")
    submission = sample_submission.copy()

    # Fill predictions using 'id' mapping to ensure exact alignment
    for target in TARGET_COLS:
        submission[target] = submission['id'].map(test_preds_agg.set_index('id')[target])

    # --- Final Validation ---
    print("\n--- Performing Submission Validations ---")
    assert submission.shape == sample_submission.shape, f"Submission shape mismatch! Expected {sample_submission.shape}, got {submission.shape}"
    assert submission['id'].tolist() == sample_submission['id'].tolist(), "ID order mismatch in submission!"
    
    # Check for NaNs only in TARGET_COLS for final submission
    nan_check = submission[TARGET_COLS].isna().sum().sum()
    assert nan_check == 0, f"NaNs found in submission for target columns! Count: {nan_check}"
    
    # Check for infinite/invalid values
    inf_check = not np.isfinite(submission[TARGET_COLS].values).all()
    assert not inf_check, "Infinite or invalid values found in submission!"

    # --- Save submission ---
    submission.to_csv('submission.csv', index=False)
    print("âœ… Final submission.csv created and validated!")
    print(f"Submission head:\n{submission.head()}")

except Exception as e:
    print(f"â�Œ Error during notebook execution: {e}")
    traceback.print_exc()

    # Save fallback submission if possible
    try:
        if 'sample_submission' in locals(): # Check if sample_submission was loaded
            sample_submission.to_csv('submission.csv', index=False)
            print("âš ï¸� Fallback submission saved to avoid complete failure.")
        else:
            print("â�Œ Sample submission not loaded, cannot save fallback.")
    except Exception as fe:
        print(f"â�Œ Failed to save fallback submission: {fe}")

print("ğŸ�� Polymer Prediction Workflow Finished.")

