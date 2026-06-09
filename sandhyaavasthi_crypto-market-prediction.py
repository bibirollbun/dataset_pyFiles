import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
import gc

# Suppress common warnings for a cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    """Holds all major configuration parameters for the pipeline."""
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SAMPLE_SUB_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    LABEL_COLUMN = "label"
    RANDOM_STATE = 42
    
    # Bootstrap settings for stability analysis
    N_BOOTSTRAPS = 15         # Number of bootstrap models to train for a stable ensemble.
    BOOTSTRAP_RATIO = 0.8     # Proportion of data to sample in each bootstrap.
    
    # A single, powerful, well-regularized XGBoost configuration
    XGB_PARAMS = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.02,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5,
        'reg_lambda': 0.5, 'min_child_weight': 10, 'gamma': 0.1,
        'random_state': RANDOM_STATE, 'n_jobs': -1, 
        'verbosity': 0, 'device': 'gpu', 'tree_method': 'hist'
    }




# ==============================================================================
# Feature Engineering
# ==============================================================================
def feature_engineering(df):
    """ Creates a rich set of features from the raw data. """
    df = df.copy()
    
    # 1. Basic Microstructure Ratios
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['bid_ask_spread'] = (df['ask_qty'] - df['bid_qty'])
    
    # 2. Rolling Window Features for context
    windows = [50, 200]
    for window in windows:
        df[f'ofi_mean_{window}'] = df['order_flow_imbalance'].rolling(window=window, min_periods=1).mean()
        df[f'spread_std_{window}'] = df['bid_ask_spread'].rolling(window=window, min_periods=1).std()

    # Handle any potential infinite values or NaNs created during engineering
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df




# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
if __name__ == "__main__":
    
    # --- STAGE 1: Load and Process Data Sequentially for Memory Efficiency ---
    print("--- Stage 1: Loading and Processing Data ---")
    all_feature_cols = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"] + [f"X{i}" for i in range(1, 781)]
    
    # Process training data first
    print("  Processing training data...")
    train = feature_engineering(pd.read_parquet(Config.TRAIN_PATH, columns=all_feature_cols + ['label']))
    if '__index_level_0__' in train.columns:
        train = train.rename(columns={'__index_level_0__': 'timestamp'}).sort_values('timestamp').reset_index(drop=True)
    X, y = train.drop(columns=['label', '__index_level_0__', 'timestamp'], errors='ignore'), train[Config.LABEL_COLUMN]
    
    # Free up memory
    del train
    gc.collect()
    
    # Process test data second
    print("  Processing test data...")
    test = feature_engineering(pd.read_parquet(Config.TEST_PATH, columns=all_feature_cols))
    X_test = test[X.columns] # Ensure column order matches
    
    # Free up memory
    del test
    gc.collect()

    # --- STAGE 2: Bootstrap Ensemble Training ---
    print(f"\n--- Stage 2: Training Bootstrap Ensemble ({Config.N_BOOTSTRAPS} iterations) ---")
    
    all_test_predictions = []
    
    for i in range(Config.N_BOOTSTRAPS):
        print(f"  > Running Bootstrap Iteration {i+1}/{Config.N_BOOTSTRAPS}...")
        
        # Create a bootstrap sample using integer positions for .iloc
        bootstrap_indices = np.random.choice(len(X), size=int(len(X) * Config.BOOTSTRAP_RATIO), replace=True)
        X_boot, y_boot = X.iloc[bootstrap_indices], y.iloc[bootstrap_indices]
        
        # Train model on this bootstrap sample
        params = Config.XGB_PARAMS.copy()
        params['random_state'] = Config.RANDOM_STATE + i # Vary seed for each model
        
        model = XGBRegressor(**params)
        model.fit(X_boot, y_boot)
        
        # Store predictions for the test set
        all_test_predictions.append(model.predict(X_test))
        
        del model, X_boot, y_boot, bootstrap_indices
        gc.collect()

    # --- STAGE 3: Generate Final Submission ---
    print("\n--- Stage 3: Averaging Bootstrap Predictions and Saving Submission ---")
    
    # The final prediction is the average of all predictions from all bootstrap runs
    final_prediction = np.mean(all_test_predictions, axis=0)
    
    # Post-processing: Clip predictions to a reasonable range based on training labels
    p01, p99 = np.percentile(y, [1, 99])
    final_prediction_clipped = np.clip(final_prediction, p01, p99)
    
    # Create submission file
    submission_df = pd.read_csv(Config.SAMPLE_SUB_PATH)
    submission_df["prediction"] = final_prediction_clipped
    submission_df.to_csv("submission.csv", index=False)
    
    print("\nSubmission file 'submission.csv' created successfully.")
    print("This script is the final, recommended model for the competition.")

