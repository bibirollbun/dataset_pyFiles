import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =================================================================
# ===== Ultra-Fast Baseline: LightGBM Only =====
# =================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import time

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
    TEST_SIZE = 0.2 
    
# --- 2. Data Loading and Preparation Function ---
def load_data(cfg):
    """Loads and combines all data sources."""
    train = pd.read_csv(cfg.TRAIN_PATH)
    test = pd.read_csv(cfg.TEST_PATH)
    original = pd.read_csv(cfg.ORIGINAL_PATH)
    
    original = original.rename(columns={
        "Temparature ": "Temparature", "Humidity ": "Humidity", "Moisture ": "Moisture",
        "Soil Type": "Soil Type", "Crop Type": "Crop Type", "Nitrogen": "Nitrogen",
        "Potassium": "Potassium", "Phosphorous": "Phosphorous", "Fertilizer Name": "Fertilizer Name"
    })
    
    train_df = pd.concat([train, original], ignore_index=True).drop_duplicates()
    return train_df, test

# --- 3. Preprocessing and Feature Engineering ---
def preprocess(df, test_df):
    """Encodes features and target."""
    
    for frame in [df, test_df]:
        frame['N_div_P'] = frame['Nitrogen'] / (frame['Phosphorous'] + 1e-6)
        frame['N_div_K'] = frame['Nitrogen'] / (frame['Potassium'] + 1e-6)
        frame['P_div_K'] = frame['Phosphorous'] / (frame['Potassium'] + 1e-6)
        frame['NPK_total'] = frame['Nitrogen'] + frame['Phosphorous'] + frame['Potassium']

    X = df.drop(columns=[CFG.TARGET_COL, CFG.ID_COL], errors='ignore')
    y_raw = df[CFG.TARGET_COL]
    X_test = test_df.drop(columns=[CFG.ID_COL], errors='ignore')

    train_cols = X.columns
    test_cols = X_test.columns
    shared_cols = list(set(train_cols) & set(test_cols))
    X = X[shared_cols]
    X_test = X_test[shared_cols]

    for col in ['Soil Type', 'Crop Type']:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]]).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_raw)
    
    return X, y_encoded, X_test, target_encoder

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    start_time = time.time()
    
    # --- Load and Preprocess Data ---
    print("Step 1: Loading and preprocessing data...")
    train_df, test_df = load_data(CFG())
    X, y, X_test, target_encoder = preprocess(train_df, test_df)
    
    # --- Create a single train/validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CFG.TEST_SIZE, random_state=CFG.SEED, stratify=y
    )
    
    # --- Train LightGBM Model ---
    print("\nStep 2: Training LightGBM model (Fast Mode)...")
    lgb_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(target_encoder.classes_),
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7, # Slightly deeper can be good for single models
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': CFG.SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    
    lgb_preds_proba = lgb_model.predict_proba(X_test)
    print("LightGBM training finished.")
    
    # --- Create Submission File ---
    print("\nStep 3: Creating submission file...")
    # Predictions are based only on the single LightGBM model
    top_3_indices = np.argsort(lgb_preds_proba, axis=1)[:, ::-1][:, :3]
    top_3_labels = target_encoder.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)
    predictions = [" ".join(row) for row in top_3_labels]
    
    submission_df = pd.DataFrame({'id': test_df[CFG.ID_COL], CFG.TARGET_COL: predictions})
    submission_df.to_csv('submission_lgbm_fast.csv', index=False)
    
    end_time = time.time()
    print(f"\n'submission_lgbm_fast.csv' created successfully!")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    print("\nTop 5 rows of the submission file:")
    print(submission_df.head())




