import os, joblib, polars as pl, pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import kaggle_evaluation.cmi_inference_server
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Load Pre-trained Artifacts ---
INPUT_DIR = '/kaggle/input/cmi2025-single-lgbm-train' 

MODEL_PATH = os.path.join(INPUT_DIR, 'imu_advanced_model.pkl')
LE_PATH = os.path.join(INPUT_DIR, 'imu_advanced_label_encoder.pkl')
FEATURES_PATH = os.path.join(INPUT_DIR, 'imu_advanced_feature_list.txt')

print("Loading pre-trained model and assets for IMU-Only model...")
final_model = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)
with open(FEATURES_PATH, 'r') as f:
    feature_list = [line.strip() for line in f.readlines()]
print("Assets loaded successfully.")


# --- 2. Define Feature Engineering Function (must be identical to training) ---
def create_advanced_imu_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates advanced statistical and phase-approximated features 
    from IMU data ONLY.
    """
    imu_cols = [col for col in df.columns if 'acc_' in col or 'rot_' in col]
    aggs = []
    
    # 1. Whole-sequence statistical features
    for col in imu_cols:
        aggs.extend([
            pl.mean(col).alias(f'{col}_mean'),
            pl.std(col).alias(f'{col}_std'),
            pl.max(col).alias(f'{col}_max'),
            pl.min(col).alias(f'{col}_min'),
            pl.quantile(col, 0.25).alias(f'{col}_q25'),
            pl.quantile(col, 0.75).alias(f'{col}_q75'),
        ])

    # 2. Difference features
    for col in imu_cols:
        aggs.extend([
            (pl.col(col).diff().fill_null(0)).mean().alias(f'{col}_diff_mean'),
            (pl.col(col).diff().fill_null(0)).std().alias(f'{col}_diff_std'),
        ])
        
    # 3. Phase-approximated features
    for part_name, part_expr in [
        ('first_30pct', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.3),
        ('middle_40pct', (pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.3) & (pl.col('sequence_counter') < pl.max('sequence_counter') * 0.7)),
        ('last_30pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.7),
    ]:
        for col in imu_cols:
            aggs.extend([
                (pl.when(part_expr).then(pl.col(col))).mean().alias(f'{col}_mean_{part_name}'),
                (pl.when(part_expr).then(pl.col(col))).std().alias(f'{col}_std_{part_name}'),
            ])

    feature_df = df.group_by('sequence_id').agg(aggs).fill_null(0)
    return feature_df

# --- 3. Define Predict Function ---
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Receives a single test sequence and predicts the gesture.
    """
    # Create features from the input sequence using the advanced function
    feature_df = create_advanced_imu_features(sequence)
    
    # Convert to pandas and ensure column order matches the training data
    feature_df_pd = feature_df.drop('sequence_id').to_pandas()
    X_test = feature_df_pd[feature_list] 
    
    # Predict the class index
    pred_idx = final_model.predict(X_test)
    
    # Convert the index back to the original string label
    pred_label = le.inverse_transform(pred_idx)[0]
    
    return pred_label

# --- 4. Start the Inference Server ---
print("Starting inference server...")
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # This branch runs when you submit the notebook
    inference_server.serve()
else:
    # This branch runs for local testing in the notebook editor
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




