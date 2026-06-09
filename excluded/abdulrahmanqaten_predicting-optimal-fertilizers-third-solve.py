import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =================================================================================
# ===== Final Strategy: Data Augmentation + CV + Feature Engineering + Regression =====
# =================================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- 1. Configuration Class ---
class CFG:
    TRAIN_PATH = "/kaggle/input/playground-series-s5e6/train.csv"
    TEST_PATH = "/kaggle/input/playground-series-s5e6/test.csv"
    ORIGINAL_PATH = "/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"
    
    TARGET_COL = 'Fertilizer Name'
    ID_COL = 'id'
    
    SEED = 42
    N_SPLITS = 5  # Using 5 folds for Cross-Validation
    N_JOBS = -1

# --- 2. Data Loading and Preparation Function ---
def prepare_data(cfg):
    """Loads train, test, and original data, then combines them."""
    try:
        train = pd.read_csv(cfg.TRAIN_PATH)
        test = pd.read_csv(cfg.TEST_PATH)
        original = pd.read_csv(cfg.ORIGINAL_PATH)
    except FileNotFoundError:
        print("Error: Could not find all necessary data files.")
        return None, None, None

    original = original.rename(columns={
        "Temparature ": "Temparature", "Humidity ": "Humidity", "Moisture ": "Moisture",
        "Soil Type": "Soil Type", "Crop Type": "Crop Type", "Nitrogen": "Nitrogen",
        "Potassium": "Potassium", "Phosphorous": "Phosphorous", "Fertilizer Name": "Fertilizer Name"
    })
    
    train_df = pd.concat([train, original], ignore_index=True)
    test_df = test
    
    return train_df, test_df

# --- 3. Preprocessing and Feature Engineering Function ---
def preprocess(train_df, test_df, cfg):
    """Prepares the data for modeling (feature engineering, encoding)."""
    
    # === Feature Engineering Function ===
    def create_features(df):
        df['N_div_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
        df['N_div_K'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
        df['P_div_K'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
        df['NPK_total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
        return df

    # Apply Feature Engineering to both train and test data
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    X = train_df.drop(columns=[cfg.TARGET_COL, cfg.ID_COL], errors='ignore')
    y_raw = train_df[cfg.TARGET_COL]
    X_test = test_df.drop(columns=[cfg.ID_COL], errors='ignore')
    
    # Encode categorical features
    categorical_features = ['Soil Type', 'Crop Type']
    for col in categorical_features:
        le = LabelEncoder()
        combined_series = pd.concat([X[col], X_test[col]]).astype(str)
        le.fit(combined_series)
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
        
    return X, y_raw, X_test

# --- 4. Main Training Function with Cross-Validation ---
def train_and_predict(X, y_raw, X_test, cfg):
    """Trains 7 XGBRegressor models using 5-Fold CV and returns predictions."""
    unique_labels = y_raw.unique()
    
    # DataFrame to store test predictions, initialized to zeros
    test_predictions_agg = pd.DataFrame(index=X_test.index, columns=unique_labels, dtype=float).fillna(0)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 2000,  # Increased, but early stopping will find the optimal number
        'learning_rate': 0.02,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': cfg.SEED,
        'n_jobs': cfg.N_JOBS
    }

    # Initialize StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)

    print(f"\nStarting training with {len(unique_labels)} regressors using {cfg.N_SPLITS}-Fold CV...")
    
    # Loop through each fertilizer type (label)
    for label in unique_labels:
        print(f"\n--> Training for label: {label}")
        y_binary = (y_raw == label).astype(int)

        # Loop through each fold of the cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_raw)):
            print(f"  -- Fold {fold+1}/{cfg.N_SPLITS}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_binary, y_val_binary = y_binary.iloc[train_idx], y_binary.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            
            # Fit the model with EARLY STOPPING
            model.fit(X_train, y_train_binary,
                      eval_set=[(X_val, y_val_binary)],
                      early_stopping_rounds=50,  # Stops if performance doesn't improve for 50 rounds
                      verbose=False)
            
            # Add this fold's predictions to the aggregate, dividing by number of splits
            test_predictions_agg[label] += model.predict(X_test) / cfg.N_SPLITS
            
    print("\nTraining finished successfully!")
    return test_predictions_agg

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    
    cfg = CFG()
    
    print("Step 1: Loading and combining data...")
    train_df, test_df_original = prepare_data(cfg)
    
    if train_df is not None:
        print("\nStep 2: Preprocessing and creating new features...")
        X, y_raw, X_test = preprocess(train_df, test_df_original.copy(), cfg)
        
        print("\nStep 3: Training models with Cross-Validation...")
        preds_df = train_and_predict(X, y_raw, X_test, cfg)
        
        print("\nStep 4: Assembling final submission file...")
        predictions = []
        for index, row in preds_df.iterrows():
            top_3_labels = row.sort_values(ascending=False).head(3).index.tolist()
            predictions.append(" ".join(top_3_labels))
            
        submission_df = pd.DataFrame({'id': test_df_original[cfg.ID_COL], cfg.TARGET_COL: predictions})
        submission_df.to_csv('submission.csv', index=False)
        
        print("\n'submission.csv' created successfully!")
        print("This version uses Feature Engineering and Cross-Validation to improve accuracy.")
        print("\nTop 5 rows of the submission file:")
        print(submission_df.head())




