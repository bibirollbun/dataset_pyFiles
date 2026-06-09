import kaggle_evaluation.cmi_inference_server
from collections import Counter
import pandas as pd
import polars as pl
import numpy as np
import joblib
import os

final_model = joblib.load('/kaggle/input/cmi-xgboost-train/model.pkl')
print('Loaded Model')
rev_target_map = {
    0:"Above ear - pull hair",
    1:"Cheek - pinch skin",
    2:"Eyebrow - pull hair",
    3:"Eyelash - pull hair", 
    4:"Forehead - pull hairline",
    5:"Forehead - scratch",
    6:"Neck - pinch skin", 
    7:"Neck - scratch",
    
    8:"Drink from bottle/cup",
    9:"Feel around in tray and pull out an object",
    10:"Glasses on/off",
    11:"Pinch knee/leg skin", 
    12:"Pull air toward your face",
    13:"Scratch knee/leg skin",
    14:"Text on phone",
    15:"Wave hello",
    16:"Write name in air",
    17:"Write name on leg",
}

def create_advanced_imu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates advanced statistical and phase-approximated features 
    from IMU data ONLY, returning one row per sequence_id.
    """
    # Identify IMU columns
    imu_cols = [c for c in df.columns if c.startswith('acc_') or c.startswith('rot_')]

    def extract_features(group: pd.DataFrame) -> pd.Series:
        feats = {}
        # Whole-sequence statistical features
        for col in imu_cols:
            data = group[col]
            diffs = data.diff().fillna(0)

            # Basic stats
            feats[f'{col}_mean']       = data.mean()
            feats[f'{col}_std']        = data.std(ddof=0)
            feats[f'{col}_min']        = data.min()
            feats[f'{col}_max']        = data.max()
            feats[f'{col}_median']     = data.median()
            feats[f'{col}_range']      = data.max() - data.min()
            feats[f'{col}_q25']        = data.quantile(0.25)
            feats[f'{col}_q75']        = data.quantile(0.75)
            feats[f'{col}_iqr']        = data.quantile(0.75) - data.quantile(0.25)
            feats[f'{col}_mad']        = (data - data.mean()).abs().mean() # mean absolute deviation
            feats[f'{col}_mean_abs']   = data.abs().mean()
            feats[f'{col}_rms']        = np.sqrt((data**2).mean())      # root mean square
            feats[f'{col}_energy']     = (data**2).sum()                # signal energy

            # Optional: skewness & kurtosis (requires pandas >= 1.1 or scipy)
            feats[f'{col}_skew']       = data.skew()
            feats[f'{col}_kurtosis']   = data.kurtosis()

            # Difference‐based stats
            feats[f'{col}_diff_mean']  = diffs.mean()
            feats[f'{col}_diff_std']   = diffs.std(ddof=0)
            feats[f'{col}_diff_median']= diffs.median()
            feats[f'{col}_diff_q25']   = diffs.quantile(0.25)
            feats[f'{col}_diff_q75']   = diffs.quantile(0.75)
            feats[f'{col}_diff_iqr']   = diffs.quantile(0.75) - diffs.quantile(0.25)
            feats[f'{col}_diff_rms']   = np.sqrt((diffs**2).mean())
            feats[f'{col}_diff_mad']   = diffs.abs().mean()

            # Zero‐crossing count (how many times signal changes sign)
            zc = ((data.shift(1) * data) < 0).sum()
            feats[f'{col}_zero_crossings'] = zc
        # Phase-approximated features
        seq      = group['sequence_counter']
        max_seq  = seq.max()
        first    = seq <  max_seq * 0.3
        middle   = (seq >= max_seq * 0.3) & (seq < max_seq * 0.7)
        last     = seq >= max_seq * 0.7

        for col in imu_cols:
            # precompute diffs for this column
            diffs = group[col].diff().fillna(0)

            for mask, name in [
                (first,  'first_30pct'),
                (middle, 'middle_40pct'),
                (last,   'last_30pct')
            ]:
                sub       = group.loc[mask, col]

                # basic stats
                feats[f'{col}_mean_{name}']      = sub.mean() if not sub.empty else 0
                feats[f'{col}_std_{name}']       = sub.std(ddof=0) if not sub.empty else 0

        return pd.Series(feats)

    # Apply to each sequence_id
    feature_df = df.groupby('sequence_id').apply(extract_features).fillna(0)
    df = df.merge(feature_df, on = 'sequence_id', how = 'left')
    #diffs = df[imu_cols + ['subject']].groupby('subject').diff(periods = 2).fillna(0)
    #diffs.columns = diffs.columns + '_diff'
    #df = pd.concat([df, diffs], axis = 1)
    # Fill any remaining NaNs (e.g. empty phases) with 0
    return df.fillna(0)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Receives a single test sequence and predicts the gesture.
    """
    # Create features from the input sequence using the advanced function
    sequence = sequence.to_pandas()
    demographics = demographics.to_pandas()
    sequence = sequence.merge(demographics, on = 'subject', how = 'left')
    sequence = create_advanced_imu_features(sequence)
    sequence = sequence.drop(columns = ['row_id','sequence_id','sequence_counter','subject']).astype('float32')
    pred = final_model.predict(sequence)

    pred = [rev_target_map[p] for p in pred]
    
    return Counter(pred).most_common(1)[0][0]

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

