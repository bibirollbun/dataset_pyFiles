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


"""
MEDICAL MODEL - GUARANTEED SAVE
Full medical features + location knowledge + guaranteed to save
"""

import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import os

DATA_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection'
SERIES_PATH = Path(DATA_PATH) / 'series'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# MEDICAL KNOWLEDGE from your research
LOCATION_PREVALENCE = {
    'Anterior Communicating Artery': 0.325,  # 30-35% most common
    'Left Posterior Communicating Artery': 0.125,
    'Right Posterior Communicating Artery': 0.125,
    'Left Middle Cerebral Artery': 0.10,
    'Right Middle Cerebral Artery': 0.10,
    'Basilar Tip': 0.06,
    'Left Supraclinoid Internal Carotid Artery': 0.04,
    'Right Supraclinoid Internal Carotid Artery': 0.04,
    'Left Anterior Cerebral Artery': 0.025,
    'Right Anterior Cerebral Artery': 0.025,
    'Other Posterior Circulation': 0.05,
    'Left Infraclinoid Internal Carotid Artery': 0.015,
    'Right Infraclinoid Internal Carotid Artery': 0.015,
}

# Visibility scores (higher = better visibility on that modality)
CT_VISIBILITY = {
    'Anterior Communicating Artery': 0.9,
    'Left Middle Cerebral Artery': 0.9,
    'Right Middle Cerebral Artery': 0.9,
    'Left Posterior Communicating Artery': 0.85,
    'Right Posterior Communicating Artery': 0.85,
    'Left Supraclinoid Internal Carotid Artery': 0.85,
    'Right Supraclinoid Internal Carotid Artery': 0.85,
    'Basilar Tip': 0.8,
    'Left Anterior Cerebral Artery': 0.75,
    'Right Anterior Cerebral Artery': 0.75,
    'Other Posterior Circulation': 0.6,
    'Left Infraclinoid Internal Carotid Artery': 0.5,
    'Right Infraclinoid Internal Carotid Artery': 0.5,
}

def extract_medical_features(series_id):
    """Extract full medical features."""
    
    series_path = SERIES_PATH / series_id
    dcm_files = sorted(list(series_path.glob('*.dcm')))
    
    if not dcm_files:
        return None
    
    features = {'SeriesInstanceUID': series_id}
    
    try:
        ds = pydicom.dcmread(dcm_files[0], force=True)
        
        # === MODALITY (Critical for visibility) ===
        modality = str(getattr(ds, 'Modality', 'Unknown'))
        features['is_ct'] = 1.0 if modality == 'CT' else 0.0
        features['is_mr'] = 1.0 if modality == 'MR' else 0.0
        
        # === PROTOCOL QUALITY ===
        features['num_slices'] = float(len(dcm_files))
        features['num_slices_log'] = np.log(len(dcm_files) + 1)
        
        slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
        features['slice_thickness'] = slice_thickness
        features['is_thin'] = 1.0 if slice_thickness < 0.7 else 0.0
        features['is_good'] = 1.0 if slice_thickness < 1.0 else 0.0
        
        pixel_spacing = getattr(ds, 'PixelSpacing', [0.5, 0.5])
        features['pixel_spacing'] = float(pixel_spacing[0])
        features['is_high_res'] = 1.0 if float(pixel_spacing[0]) < 0.4 else 0.0
        
        # Protocol quality score
        features['protocol_quality'] = (
            (1.0 if features['is_thin'] else 0.5) *
            (1.0 if features['is_high_res'] else 0.7) *
            (1.0 if features['num_slices'] > 100 else 0.8)
        )
        
        # === INTENSITY ANALYSIS (Sample 3 regions) ===
        n_slices = len(dcm_files)
        sample_indices = [
            int(n_slices * 0.3),
            int(n_slices * 0.5),
            int(n_slices * 0.7),
        ]
        
        intensities = []
        for idx in sample_indices:
            if 0 <= idx < n_slices:
                ds_sample = pydicom.dcmread(dcm_files[idx], force=True)
                if hasattr(ds_sample, 'pixel_array'):
                    pixels = ds_sample.pixel_array.astype(float)
                    intercept = getattr(ds_sample, 'RescaleIntercept', 0)
                    slope = getattr(ds_sample, 'RescaleSlope', 1)
                    pixels = pixels * slope + intercept
                    intensities.append(pixels)
        
        if intensities:
            all_pixels = np.concatenate([img.flatten() for img in intensities])
            
            features['intensity_mean'] = np.mean(all_pixels)
            features['intensity_std'] = np.std(all_pixels)
            features['intensity_max'] = np.max(all_pixels)
            features['intensity_p95'] = np.percentile(all_pixels, 95)
            features['intensity_p99'] = np.percentile(all_pixels, 99)
            
            # CRITICAL: Enhancement ratio (aneurysms are bright!)
            features['enhancement_ratio'] = features['intensity_p99'] / (features['intensity_mean'] + 1.0)
            features['contrast_quality'] = (features['intensity_p95'] - features['intensity_mean']) / (features['intensity_mean'] + 1.0)
            features['has_strong_enhancement'] = 1.0 if features['enhancement_ratio'] > 3.0 else 0.0
            
        else:
            features.update({
                'intensity_mean': 0.0, 'intensity_std': 0.0, 'intensity_max': 0.0,
                'intensity_p95': 0.0, 'intensity_p99': 0.0,
                'enhancement_ratio': 1.0, 'contrast_quality': 0.0,
                'has_strong_enhancement': 0.0,
            })
        
        # === MEDICAL INTERACTION FEATURES ===
        features['ct_with_enhancement'] = features['is_ct'] * features['enhancement_ratio']
        features['quality_coverage'] = features['protocol_quality'] * features['num_slices_log']
        features['optimal_for_small'] = features['is_thin'] * features['has_strong_enhancement']
        
    except Exception as e:
        return None
    
    return features

def train_medical_model():
    """Train with medical knowledge + guaranteed save."""
    
    print("="*80)
    print("MEDICAL MODEL TRAINING - WITH GUARANTEED SAVE")
    print("="*80)
    
    # Load
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
    print(f"   âœ… {len(train_df):,} samples")
    
    # Save checkpoint
    with open('checkpoint_loaded.pkl', 'wb') as f:
        pickle.dump({'n_samples': len(train_df)}, f)
    
    # Extract
    print("\n[2/5] Extracting medical features...")
    feature_list = []
    
    for series_id in tqdm(train_df['SeriesInstanceUID'], desc="   Progress"):
        feats = extract_medical_features(series_id)
        if feats:
            feature_list.append(feats)
        if len(feature_list) % 100 == 0:
            gc.collect()
    
    features_df = pd.DataFrame(feature_list)
    train_data = train_df.merge(features_df, on='SeriesInstanceUID', how='inner')
    
    print(f"\n   âœ… {len(features_df)} samples, {len(features_df.columns)-1} features")
    print(f"   Features include:")
    print(f"      - Modality (CT/MR)")
    print(f"      - Protocol quality (slice thickness, resolution)")
    print(f"      - Intensity analysis (enhancement detection)")
    print(f"      - Medical interactions (optimal for small aneurysms)")
    
    feature_cols = [col for col in features_df.columns if col != 'SeriesInstanceUID']
    
    # Save checkpoint
    with open('checkpoint_features.pkl', 'wb') as f:
        pickle.dump({'feature_cols': feature_cols, 'n_features': len(feature_cols)}, f)
    
    # Train
    print("\n[3/5] Training location-specific models...")
    
    models = {}
    cv_scores = {}
    n_trained = 0
    n_skipped = 0
    
    for target in LABEL_COLS:
        X = train_data[feature_cols].fillna(0).values
        y = train_data[target].values
        
        n_pos = y.sum()
        
        if n_pos < 10:
            print(f"   {target:50s}: SKIP ({n_pos} pos)")
            n_skipped += 1
            continue
        
        # Get medical info
        expected_prev = LOCATION_PREVALENCE.get(target, 0.1)
        
        # Train with 3-fold (faster)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []
        fold_models = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            n_neg = len(y_train) - y_train.sum()
            scale_pos_weight = n_neg / (y_train.sum() + 1e-6)
            
            lgbm = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1
            )
            
            lgbm.fit(X_train_scaled, y_train)
            
            if len(np.unique(y_val)) > 1:
                pred = lgbm.predict_proba(X_val_scaled)[:, 1]
                auc = roc_auc_score(y_val, pred)
                fold_scores.append(auc)
                fold_models.append({'model': lgbm, 'scaler': scaler})
        
        if fold_scores:
            mean_auc = np.mean(fold_scores)
            print(f"   {target:50s}: {mean_auc:.4f} (prev:{expected_prev:.1%})")
            cv_scores[target] = mean_auc
            models[target] = fold_models
            n_trained += 1
    
    print(f"\n   âœ… Trained {n_trained} models, Skipped {n_skipped}")
    
    # Save checkpoint
    with open('checkpoint_trained.pkl', 'wb') as f:
        pickle.dump({'n_trained': n_trained, 'cv_scores': cv_scores}, f)
    
    # Calculate
    print("\n[4/5] Results...")
    
    aneurysm_score = cv_scores.get('Aneurysm Present', 0.5)
    location_scores = [cv_scores[col] for col in LABEL_COLS[:-1] if col in cv_scores]
    
    if location_scores:
        location_avg = np.mean(location_scores)
        comp_score = (aneurysm_score + location_avg) / 2
        print(f"\n   Aneurysm Present: {aneurysm_score:.4f}")
        print(f"   Location Average: {location_avg:.4f}")
        print(f"   ğŸ�† COMPETITION: {comp_score:.4f}")
    else:
        comp_score = 0.5
        print(f"   âš ï¸�  No scores")
    
    # GUARANTEED SAVE
    print("\n[5/5] SAVING (trying multiple methods)...")
    
    model_data = {
        'models': models,
        'feature_cols': feature_cols,
        'cv_scores': cv_scores,
        'prevalence': LOCATION_PREVALENCE,
        'ct_visibility': CT_VISIBILITY,
    }
    
    saved_files = []
    
    # Method 1
    try:
        with open('medical_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        if os.path.exists('medical_model.pkl'):
            size = os.path.getsize('medical_model.pkl') / (1024*1024)
            print(f"   âœ… medical_model.pkl ({size:.2f} MB)")
            saved_files.append('medical_model.pkl')
    except Exception as e:
        print(f"   â�Œ Method 1: {e}")
    
    # Method 2
    try:
        with open('medical_model_v2.pkl', 'wb') as f:
            pickle.dump(model_data, f, protocol=4)
        if os.path.exists('medical_model_v2.pkl'):
            size = os.path.getsize('medical_model_v2.pkl') / (1024*1024)
            print(f"   âœ… medical_model_v2.pkl ({size:.2f} MB)")
            saved_files.append('medical_model_v2.pkl')
    except Exception as e:
        print(f"   â�Œ Method 2: {e}")
    
    # List all
    print("\nğŸ“� All .pkl files:")
    for file in sorted(os.listdir('.')):
        if file.endswith('.pkl'):
            size = os.path.getsize(file) / 1024
            print(f"   {file:40s} {size:8.2f} KB")
    
    if saved_files:
        print(f"\nâœ… SUCCESS! Saved: {saved_files[0]}")
        print(f"\nğŸ�¯ Next steps:")
        print(f"   1. Download {saved_files[0]}")
        print(f"   2. Create Kaggle dataset")
        print(f"   3. Use inference code")
    else:
        print(f"\nâ�Œ Save failed! But you have checkpoints")
    
    return model_data

if __name__ == "__main__":
    train_medical_model()

