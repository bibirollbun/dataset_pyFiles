import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================================
# ===== Final Version: Very Fast & Smart LightGBM =====
# =====================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import warnings

warnings.filterwarnings("ignore")

# --- 1. Configuration Class ---
class CFG:
    # Using ONLY the main competition data for maximum speed
    TRAIN_PATH = "/kaggle/input/playground-series-s5e6/train.csv"
    TEST_PATH = "/kaggle/input/playground-series-s5e6/test.csv"
    
    TARGET_COL = 'Fertilizer Name'
    ID_COL = 'id'
    
    SEED = 42
    # A validation split is still needed for Early Stopping to work
    TEST_SIZE = 0.2 
    
# --- 2. Data Loading and Preparation Function (Simplified for Speed) ---
def load_data(cfg):
    """Loads only the essential competition data."""
    train = pd.read_csv(cfg.TRAIN_PATH)
    test = pd.read_csv(cfg.TEST_PATH)
    
    # No external data is used, ensuring maximum speed
    return train, test

# --- 3. Preprocessing with Feature Engineering ---
def preprocess(df, test_df):
    """Encodes features and adds simple, high-impact new features."""
    
    # Feature Engineering is very fast and adds a lot of value
    for frame in [df, test_df]:
        frame['N_div_P'] = frame['Nitrogen'] / (frame['Phosphorous'] + 1e-6)
        frame['N_div_K'] = frame['Nitrogen'] / (frame['Potassium'] + 1e-6)
        frame['P_div_K'] = frame['Phosphorous'] / (frame['Potassium'] + 1e-6)
        frame['NPK_total'] = frame['Nitrogen'] + frame['Phosphorous'] + frame['Potassium']

    X = df.drop(columns=[CFG.TARGET_COL, CFG.ID_COL], errors='ignore')
    y_raw = df[CFG.TARGET_COL]
    X_test = test_df.drop(columns=[CFG.ID_COL], errors='ignore')

    # Align columns to ensure consistency
    shared_cols = list(set(X.columns) & set(X_test.columns))
    X = X[shared_cols]
    X_test = X_test[shared_cols]

    # Label Encoding for categorical features
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
    
    print("Step 1: Loading main data and creating features...")
    train_df, test_df = load_data(CFG())
    X, y, X_test, target_encoder = preprocess(train_df, test_df)
    
    # Create a train/validation split to use for Early Stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CFG.TEST_SIZE, random_state=CFG.SEED, stratify=y
    )
    
    print("\nStep 2: Training a smart LightGBM model with Early Stopping...")
    lgb_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(target_encoder.classes_),
        'n_estimators': 1000, # Set high, early stopping will find the best
        'learning_rate': 0.05,
        'seed': CFG.SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    # Fit the model with the Early Stopping callback
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)]) 
    
    print("Training complete. Best iteration found by early stopping.")
    
    # Predict on the test set
    predictions_proba = lgb_model.predict_proba(X_test)
    
    print("\nStep 3: Creating submission file...")
    top_3_indices = np.argsort(predictions_proba, axis=1)[:, ::-1][:, :3]
    top_3_labels = target_encoder.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)
    predictions = [" ".join(row) for row in top_3_labels]
    
    submission_df = pd.DataFrame({'id': test_df[CFG.ID_COL], CFG.TARGET_COL: predictions})
    submission_df.to_csv('submission_final_fast.csv', index=False)
    
    end_time = time.time()
    print(f"\n'submission_final_fast.csv' created successfully!")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

