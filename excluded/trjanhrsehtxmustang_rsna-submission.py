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
MEDICAL INFERENCE - Matches guaranteed training
Full medical features for submission
"""

import os
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import polars as pl
import pydicom
import pickle
import gc

import kaggle_evaluation.rsna_inference_server

ID_COL = 'SeriesInstanceUID'

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

class MedicalInference:
    def __init__(self):
        self.models = None
        self.feature_cols = None
        self.prevalence = None
        self._load_models()
    
    def _load_models(self):
        """Load medical models."""
        
        paths = [
            'medical_model.pkl',
            'medical_model_v2.pkl',
            '/kaggle/input/rsna-medical-aneurysm/medical_model.pkl',
            '/kaggle/input/rsna-medical-aneurysm/medical_model_v2.pkl',
        ]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                        self.models = data['models']
                        self.feature_cols = data['feature_cols']
                        self.prevalence = data.get('prevalence', {})
                    print(f"✅ Models loaded: {path}")
                    print(f"   Features: {len(self.feature_cols)}")
                    print(f"   Models: {len(self.models)}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load {path}: {e}")
        
        print("⚠️  No models found, using medical priors")
        self._set_priors()
    
    def _set_priors(self):
        """Medical priors from literature."""
        self.prevalence = {
            'Anterior Communicating Artery': 0.325,
            'Left Posterior Communicating Artery': 0.125,
            'Right Posterior Communicating Artery': 0.125,
            'Left Middle Cerebral Artery': 0.10,
            'Right Middle Cerebral Artery': 0.10,
            'Basilar Tip': 0.06,
            'Left Supraclinoid Internal Carotid Artery': 0.04,
            'Right Supraclinoid Internal Carotid Artery': 0.04,
            'Other Posterior Circulation': 0.05,
            'Left Anterior Cerebral Artery': 0.025,
            'Right Anterior Cerebral Artery': 0.025,
            'Left Infraclinoid Internal Carotid Artery': 0.015,
            'Right Infraclinoid Internal Carotid Artery': 0.015,
            'Aneurysm Present': 0.43,
        }
        self.feature_cols = []
    
    def extract_features(self, series_path, tags):
        """Extract same medical features."""
        
        features = {}
        filepaths = tags.get('filepath', [])
        
        if not filepaths:
            return self._defaults()
        
        try:
            ds = pydicom.dcmread(filepaths[0], force=True)
            
            # Modality
            modality = str(getattr(ds, 'Modality', 'Unknown'))
            features['is_ct'] = 1.0 if modality == 'CT' else 0.0
            features['is_mr'] = 1.0 if modality == 'MR' else 0.0
            
            # Protocol
            features['num_slices'] = float(len(filepaths))
            features['num_slices_log'] = np.log(len(filepaths) + 1)
            
            slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))
            features['slice_thickness'] = slice_thickness
            features['is_thin'] = 1.0 if slice_thickness < 0.7 else 0.0
            features['is_good'] = 1.0 if slice_thickness < 1.0 else 0.0
            
            pixel_spacing = getattr(ds, 'PixelSpacing', [0.5, 0.5])
            features['pixel_spacing'] = float(pixel_spacing[0])
            features['is_high_res'] = 1.0 if float(pixel_spacing[0]) < 0.4 else 0.0
            
            features['protocol_quality'] = (
                (1.0 if features['is_thin'] else 0.5) *
                (1.0 if features['is_high_res'] else 0.7) *
                (1.0 if features['num_slices'] > 100 else 0.8)
            )
            
            # Intensity
            n_slices = len(filepaths)
            sample_indices = [
                int(n_slices * 0.3),
                int(n_slices * 0.5),
                int(n_slices * 0.7),
            ]
            
            intensities = []
            for idx in sample_indices:
                if 0 <= idx < n_slices:
                    ds_sample = pydicom.dcmread(filepaths[idx], force=True)
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
            
            # Interactions
            features['ct_with_enhancement'] = features['is_ct'] * features['enhancement_ratio']
            features['quality_coverage'] = features['protocol_quality'] * features['num_slices_log']
            features['optimal_for_small'] = features['is_thin'] * features['has_strong_enhancement']
            
        except:
            return self._defaults()
        
        return features
    
    def _defaults(self):
        return {
            'is_ct': 0.0, 'is_mr': 0.0, 'num_slices': 100.0, 'num_slices_log': 4.6,
            'slice_thickness': 1.0, 'is_thin': 0.0, 'is_good': 0.0,
            'pixel_spacing': 0.5, 'is_high_res': 0.0, 'protocol_quality': 0.5,
            'intensity_mean': 0.0, 'intensity_std': 0.0, 'intensity_max': 0.0,
            'intensity_p95': 0.0, 'intensity_p99': 0.0,
            'enhancement_ratio': 1.0, 'contrast_quality': 0.0,
            'has_strong_enhancement': 0.0,
            'ct_with_enhancement': 0.0, 'quality_coverage': 0.0,
            'optimal_for_small': 0.0,
        }
    
    def predict(self, series_path, tags):
        """Predict with medical model."""
        
        features = self.extract_features(series_path, tags)
        
        if not self.models:
            return {col: self.prevalence.get(col, 0.35) for col in LABEL_COLS}
        
        predictions = {}
        feature_vector = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
        
        for target in LABEL_COLS:
            if target not in self.models:
                predictions[target] = self.prevalence.get(target, 0.35)
                continue
            
            fold_preds = []
            for fold_model in self.models[target]:
                scaler = fold_model['scaler']
                model = fold_model['model']
                
                X_scaled = scaler.transform(feature_vector)
                pred = model.predict_proba(X_scaled)[0, 1]
                fold_preds.append(pred)
            
            predictions[target] = np.clip(np.mean(fold_preds), 0.01, 0.99)
        
        return predictions

def predict(series_path: str) -> pl.DataFrame:
    """Main predict function."""
    
    if not hasattr(predict, 'predictor'):
        predict.predictor = MedicalInference()
    
    predictor = predict.predictor
    series_id = os.path.basename(series_path)
    
    # Collect files
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    tags = {'filepath': all_filepaths}
    
    # Predict
    predictions_dict = predictor.predict(series_path, tags)
    
    # Format
    row = [series_id] + [predictions_dict[col] for col in LABEL_COLS]
    predictions = pl.DataFrame(data=[row], schema=[ID_COL, *LABEL_COLS], orient='row')
    
    # Cleanup
    gc.collect()
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return predictions.drop(ID_COL)

# Server
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("COMPETITION MODE - Medical Model")
    inference_server.serve()
else:
    print("TEST MODE - Medical Model")
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

