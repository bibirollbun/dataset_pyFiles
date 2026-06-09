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


!pip install autogluon


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
import os
import gc
import logging
from typing import Tuple, Dict, List, Optional
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for results
output_dir = '/kaggle/working'
os.makedirs(output_dir, exist_ok=True)

# Enhanced logging setup
def setup_logging(output_dir: str) -> logging.Logger:
    """Configure comprehensive logging for the pipeline."""
    log_file = os.path.join(output_dir, f'horse_health_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging(output_dir)
logger.info("Starting Horse Health Prediction Pipeline - Alternative Implementation")

# Memory optimization decorator
def memory_optimize(func):
    """Decorator to optimize memory usage for functions."""
    def wrapper(*args, **kwargs):
        gc.collect()
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper

# First, let's try to use AutoGluon with proper version handling
try:
    # Try importing AutoGluon
    import subprocess
    import sys
    
    # Install specific compatible versions
    logger.info("Installing compatible AutoGluon and dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.4.2"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon==1.1.1", "--no-deps"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon.tabular==1.1.1", "--no-deps"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon.core==1.1.1", "--no-deps"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon.common==1.1.1", "--no-deps"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon.features==1.1.1", "--no-deps"])
    
    import autogluon
    from autogluon.tabular import TabularPredictor
    USE_AUTOGLUON = True
    logger.info("AutoGluon successfully imported")
    
except Exception as e:
    logger.warning(f"Could not import AutoGluon: {str(e)}")
    logger.info("Falling back to manual ensemble implementation")
    USE_AUTOGLUON = False

# Load the data with validation
def load_and_validate_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data with comprehensive validation and error handling."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logger.info(f"Train data shape: {train_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")
        
        # Validate data integrity
        assert 'outcome' in train_data.columns, "Target column 'outcome' not found in training data"
        assert 'id' in test_data.columns, "ID column not found in test data"
        
        # Check for duplicate IDs
        if train_data['id'].duplicated().any():
            logger.warning(f"Found {train_data['id'].duplicated().sum()} duplicate IDs in training data")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Enhanced feature engineering (same as before)
@memory_optimize
def engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Create comprehensive domain-specific features with memory optimization."""
    logger.info(f"Starting feature engineering for {'training' if is_training else 'test'} data")
    data = df.copy()
    
    # Smart missing value imputation
    def smart_impute(data: pd.DataFrame) -> pd.DataFrame:
        """Intelligent imputation based on feature type and distribution."""
        for col in data.columns:
            if col in ['id', 'outcome']:
                continue
                
            missing_pct = data[col].isnull().sum() / len(data)
            
            if missing_pct > 0:
                if data[col].dtype == 'object':
                    # For categorical with high missing %, create separate category
                    if missing_pct > 0.1:
                        data[col] = data[col].fillna('missing')
                    else:
                        # Use mode for low missing %
                        mode_val = data[col].mode()[0] if not data[col].mode().empty else 'unknown'
                        data[col] = data[col].fillna(mode_val)
                else:
                    # For numerical features
                    if col in ['lesion_1', 'lesion_2', 'lesion_3']:
                        data[col] = data[col].fillna(0)  # No lesion
                    elif missing_pct > 0.3:
                        # High missing - use indicator variable
                        data[f'{col}_was_missing'] = data[col].isnull().astype(int)
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        # Low missing - use median
                        data[col] = data[col].fillna(data[col].median())
        
        return data
    
    data = smart_impute(data)
    
    # 1. ENHANCED VITAL SIGNS ANALYSIS
    if all(col in data.columns for col in ['rectal_temp', 'pulse', 'respiratory_rate']):
        # Temperature features
        data['temp_low'] = (data['rectal_temp'] < 37.5).astype(int)
        data['temp_high'] = (data['rectal_temp'] > 38.5).astype(int)
        data['temp_critical_low'] = (data['rectal_temp'] < 36.5).astype(int)
        data['temp_critical_high'] = (data['rectal_temp'] > 39.5).astype(int)
        data['temp_deviation'] = abs(data['rectal_temp'] - 38.0)
        data['temp_squared_deviation'] = data['temp_deviation'] ** 2
        
        # Pulse features
        data['pulse_low'] = (data['pulse'] < 28).astype(int)
        data['pulse_high'] = (data['pulse'] > 44).astype(int)
        data['pulse_very_high'] = (data['pulse'] > 80).astype(int)
        data['pulse_critical'] = (data['pulse'] > 120).astype(int)
        data['pulse_deviation'] = abs(data['pulse'] - 36)
        
        # Respiratory features
        data['resp_low'] = (data['respiratory_rate'] < 8).astype(int)
        data['resp_high'] = (data['respiratory_rate'] > 16).astype(int)
        data['resp_very_high'] = (data['respiratory_rate'] > 30).astype(int)
        data['resp_critical'] = (data['respiratory_rate'] > 50).astype(int)
        
        # Advanced vital sign ratios
        data['pulse_temp_ratio'] = data['pulse'] / data['rectal_temp'].replace(0, 0.1)
        data['resp_pulse_ratio'] = data['respiratory_rate'] / data['pulse'].replace(0, 0.1)
        data['resp_temp_ratio'] = data['respiratory_rate'] / data['rectal_temp'].replace(0, 0.1)
        
        # Shock index (pulse/systolic BP proxy)
        data['shock_index_proxy'] = data['pulse'] / 110  # Using normal systolic as proxy
        
        # Combined severity scores with weighting
        data['vital_signs_severity'] = (
            data['temp_low'] * 1 + data['temp_high'] * 1 + 
            data['temp_critical_low'] * 3 + data['temp_critical_high'] * 3 +
            data['pulse_low'] * 1 + data['pulse_high'] * 1 + 
            data['pulse_very_high'] * 2 + data['pulse_critical'] * 4 +
            data['resp_low'] * 1 + data['resp_high'] * 1 + 
            data['resp_very_high'] * 2 + data['resp_critical'] * 4
        )
        
        # Create vital signs risk categories
        data['vital_risk_category'] = pd.cut(
            data['vital_signs_severity'], 
            bins=[-1, 2, 5, 10, 100], 
            labels=['low', 'moderate', 'high', 'critical']
        )
    
    # 2. ADVANCED LESION ANALYSIS
    if all(col in data.columns for col in ['lesion_1', 'lesion_2', 'lesion_3']):
        # Basic lesion features
        data['has_lesion'] = ((data['lesion_1'] > 0) | (data['lesion_2'] > 0) | (data['lesion_3'] > 0)).astype(int)
        data['lesion_count'] = ((data['lesion_1'] > 0).astype(int) + 
                               (data['lesion_2'] > 0).astype(int) + 
                               (data['lesion_3'] > 0).astype(int))
        
        # Lesion type analysis (based on common veterinary lesion codes)
        # Gastrointestinal lesions
        gi_lesion_codes = [2208, 2209, 5124, 5125, 2112, 2113]
        data['lesion_gi_tract'] = (
            data['lesion_1'].isin(gi_lesion_codes) | 
            data['lesion_2'].isin(gi_lesion_codes) | 
            data['lesion_3'].isin(gi_lesion_codes)
        ).astype(int)
        
        # Strangulating lesions (more severe)
        strangulating_codes = [2208, 2209, 3111, 3112]
        data['lesion_strangulating'] = (
            data['lesion_1'].isin(strangulating_codes) | 
            data['lesion_2'].isin(strangulating_codes) | 
            data['lesion_3'].isin(strangulating_codes)
        ).astype(int)
        
        # Non-strangulating lesions
        non_strangulating_codes = [5124, 5125, 7111, 7112]
        data['lesion_non_strangulating'] = (
            data['lesion_1'].isin(non_strangulating_codes) | 
            data['lesion_2'].isin(non_strangulating_codes) | 
            data['lesion_3'].isin(non_strangulating_codes)
        ).astype(int)
        
        # Multiple site involvement
        data['multiple_lesion_sites'] = (data['lesion_count'] > 1).astype(int)
        data['all_lesion_sites'] = (data['lesion_count'] == 3).astype(int)
        
        # Lesion severity scoring
        data['lesion_severity'] = (
            data['lesion_count'] * 1 + 
            data['lesion_gi_tract'] * 2 + 
            data['lesion_strangulating'] * 3 +
            data['multiple_lesion_sites'] * 2
        )
        
        # Create unique lesion combinations
        data['lesion_pattern'] = (
            data['lesion_1'].astype(str) + '_' + 
            data['lesion_2'].astype(str) + '_' + 
            data['lesion_3'].astype(str)
        )
    
    # 3. SURGICAL INDICATORS AND INTERACTIONS
    if 'surgery' in data.columns:
        # Handle different encodings of surgery column
        if data['surgery'].dtype == 'object':
            data['surgery_performed'] = (data['surgery'].str.lower() == 'yes').astype(int)
        elif data['surgery'].dtype == bool:
            data['surgery_performed'] = data['surgery'].astype(int)
        else:
            data['surgery_performed'] = data['surgery']
        
        if 'surgical_lesion' in data.columns:
            if data['surgical_lesion'].dtype == 'object':
                data['surgical_lesion_identified'] = (data['surgical_lesion'].str.lower() == 'yes').astype(int)
            elif data['surgical_lesion'].dtype == bool:
                data['surgical_lesion_identified'] = data['surgical_lesion'].astype(int)
            else:
                data['surgical_lesion_identified'] = data['surgical_lesion']
            
            # Surgical accuracy indicators
            data['surgical_intervention_match'] = (
                (data['surgery_performed'] == 1) & (data['surgical_lesion_identified'] == 1)
            ).astype(int)
            
            data['surgical_mismatch'] = (
                ((data['surgery_performed'] == 1) & (data['surgical_lesion_identified'] == 0)) |
                ((data['surgery_performed'] == 0) & (data['surgical_lesion_identified'] == 1))
            ).astype(int)
            
            # Unnecessary surgery indicator
            data['unnecessary_surgery'] = (
                (data['surgery_performed'] == 1) & (data['surgical_lesion_identified'] == 0)
            ).astype(int)
        
        # Surgery-lesion interactions
        if 'has_lesion' in data.columns:
            data['surgery_with_lesion'] = (data['surgery_performed'] & data['has_lesion']).astype(int)
            data['no_surgery_with_lesion'] = ((data['surgery_performed'] == 0) & data['has_lesion']).astype(int)
        
        # Surgery-vital signs interactions
        if 'vital_signs_severity' in data.columns:
            data['surgery_high_vitals'] = (
                (data['surgery_performed'] == 1) & (data['vital_signs_severity'] > 5)
            ).astype(int)
    
    # 4. COMPREHENSIVE ABDOMINAL ASSESSMENT
    abdominal_columns = ['abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 
                        'nasogastric_reflux_ph', 'rectal_exam_feces', 'abdomen', 'abdomo_appearance',
                        'abdomo_protein', 'abdominocentesis_appearance', 'abdominocentesis_total_protein']
    
    present_abdominal_columns = [col for col in abdominal_columns if col in data.columns]
    
    if present_abdominal_columns:
        abdominal_score = pd.Series(0, index=data.index)
        
        # Abdominal distention scoring
        if 'abdominal_distention' in data.columns:
            distention_map = {'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3}
            abdominal_score += data['abdominal_distention'].map(distention_map).fillna(0)
        
        # Nasogastric reflux scoring
        if 'nasogastric_reflux' in data.columns:
            reflux_map = {'none': 0, 'slight': 1, 'significant': 2, '> 1 liter': 3}
            abdominal_score += data['nasogastric_reflux'].map(reflux_map).fillna(0)
        
        # pH scoring (lower pH indicates more severe condition)
        if 'nasogastric_reflux_ph' in data.columns:
            ph_score = pd.Series(0, index=data.index)
            ph_score[data['nasogastric_reflux_ph'] < 3] = 3
            ph_score[(data['nasogastric_reflux_ph'] >= 3) & (data['nasogastric_reflux_ph'] < 4)] = 2
            ph_score[(data['nasogastric_reflux_ph'] >= 4) & (data['nasogastric_reflux_ph'] < 5)] = 1
            abdominal_score += ph_score
        
        # Abdominocentesis appearance
        if 'abdomo_appearance' in data.columns:
            appearance_map = {'clear': 0, 'cloudy': 1, 'serosanguious': 2}
            abdominal_score += data['abdomo_appearance'].map(appearance_map).fillna(0)
        
        # Rectal exam findings
        if 'rectal_exam_feces' in data.columns:
            feces_map = {'normal': 0, 'increased': 0, 'decreased': 1, 'absent': 2}
            abdominal_score += data['rectal_exam_feces'].map(feces_map).fillna(0)
        
        data['abdominal_severity'] = abdominal_score
        
        # Create abdominal risk categories
        data['abdominal_risk_category'] = pd.cut(
            data['abdominal_severity'], 
            bins=[-1, 2, 5, 8, 100], 
            labels=['minimal', 'mild', 'moderate', 'severe']
        )
        
        # Specific abdominal conditions
        if 'nasogastric_tube' in data.columns:
            if data['nasogastric_tube'].dtype == 'object':
                data['ng_tube_placed'] = (data['nasogastric_tube'].str.lower() == 'yes').astype(int)
            else:
                data['ng_tube_placed'] = data['nasogastric_tube'].astype(int)
    
    # 5. PAIN AND NEUROLOGICAL ASSESSMENT
    pain_columns = ['pain', 'peristalsis', 'capillary_refill_time', 'mucous_membrane']
    if any(col in data.columns for col in pain_columns):
        pain_score = pd.Series(0, index=data.index)
        
        # Pain level scoring
        if 'pain' in data.columns:
            pain_map = {'none': 0, 'depressed': 1, 'mild': 1, 'moderate': 2, 
                       'severe': 3, 'extreme': 4}
            pain_score += data['pain'].map(pain_map).fillna(0)
        
        # Peristalsis scoring
        if 'peristalsis' in data.columns:
            peristalsis_map = {'hypermotile': 1, 'normal': 0, 'hypomotile': 2, 'absent': 3}
            pain_score += data['peristalsis'].map(peristalsis_map).fillna(0)
        
        # Capillary refill time
        if 'capillary_refill_time' in data.columns:
            crt_map = {'< 3': 0, '3': 1, '> 3': 2}
            pain_score += data['capillary_refill_time'].map(crt_map).fillna(0)
        
        # Mucous membrane assessment
        if 'mucous_membrane' in data.columns:
            mm_map = {'normal_pink': 0, 'pale_pink': 1, 'pale_cyanotic': 2, 
                     'bright_pink': 1, 'bright_red': 2, 'dark_cyanotic': 3}
            pain_score += data['mucous_membrane'].map(mm_map).fillna(0)
        
        data['pain_severity'] = pain_score
        
        # Create comprehensive pain categories
        data['pain_category'] = pd.cut(
            data['pain_severity'], 
            bins=[-1, 1, 3, 6, 100], 
            labels=['minimal', 'mild', 'moderate', 'severe']
        )
    
    # 6. ENHANCED BLOOD VALUES ANALYSIS
    if all(col in data.columns for col in ['packed_cell_volume', 'total_protein']):
        # PCV (Hematocrit) analysis
        data['pcv_low'] = (data['packed_cell_volume'] < 30).astype(int)  # Severe anemia
        data['pcv_mild_low'] = ((data['packed_cell_volume'] >= 30) & 
                               (data['packed_cell_volume'] < 35)).astype(int)
        data['pcv_high'] = (data['packed_cell_volume'] > 48).astype(int)  # Dehydration
        data['pcv_very_high'] = (data['packed_cell_volume'] > 55).astype(int)  # Severe dehydration
        
        # Total protein analysis
        data['protein_low'] = (data['total_protein'] < 5.5).astype(int)  # Severe hypoproteinemia
        data['protein_mild_low'] = ((data['total_protein'] >= 5.5) & 
                                   (data['total_protein'] < 6.5)).astype(int)
        data['protein_high'] = (data['total_protein'] > 8.5).astype(int)
        data['protein_very_high'] = (data['total_protein'] > 10).astype(int)
        
        # Clinical ratios
        data['protein_pcv_ratio'] = data['total_protein'] / data['packed_cell_volume'].replace(0, 0.1)
        
        # Dehydration index
        data['dehydration_index'] = (
            data['pcv_high'] * 1 + data['pcv_very_high'] * 2 +
            data['protein_high'] * 1 + data['protein_very_high'] * 2
        )
        
        # Blood loss indicator (low PCV with normal/low protein)
        data['blood_loss_indicator'] = (
            (data['packed_cell_volume'] < 35) & (data['total_protein'] < 7)
        ).astype(int)
        
        # Create blood values severity score
        data['blood_values_abnormality'] = (
            data['pcv_low'] * 2 + data['pcv_mild_low'] * 1 + 
            data['pcv_high'] * 1 + data['pcv_very_high'] * 2 +
            data['protein_low'] * 2 + data['protein_mild_low'] * 1 + 
            data['protein_high'] * 1 + data['protein_very_high'] * 2
        )
    
    # 7. PERIPHERAL PERFUSION ASSESSMENT
    perfusion_columns = ['peripheral_pulse', 'temp_of_extremities', 'capillary_refill_time']
    if any(col in data.columns for col in perfusion_columns):
        perfusion_score = pd.Series(0, index=data.index)
        
        if 'peripheral_pulse' in data.columns:
            pulse_map = {'normal': 0, 'increased': 1, 'reduced': 2, 'absent': 3}
            perfusion_score += data['peripheral_pulse'].map(pulse_map).fillna(0)
        
        if 'temp_of_extremities' in data.columns:
            temp_map = {'normal': 0, 'warm': 1, 'cool': 2, 'cold': 3}
            perfusion_score += data['temp_of_extremities'].map(temp_map).fillna(0)
        
        data['perfusion_deficit'] = perfusion_score
        
        # Shock indicators
        data['shock_signs'] = (
            (perfusion_score > 3) | 
            ((data['pulse'] > 80) if 'pulse' in data.columns else False) |
            ((data['capillary_refill_time'] == '> 3') if 'capillary_refill_time' in data.columns else False)
        ).astype(int)
    
    # 8. AGE-SPECIFIC RISK FACTORS
    if 'age' in data.columns:
        data['is_young'] = (data['age'] == 'young').astype(int)
        data['is_adult'] = (data['age'] == 'adult').astype(int)
        
        # Age-vital signs interactions
        if 'vital_signs_severity' in data.columns:
            data['young_high_vitals'] = (data['is_young'] & (data['vital_signs_severity'] > 5)).astype(int)
        
        # Age-surgery interactions
        if 'surgery_performed' in data.columns:
            data['young_surgery'] = (data['is_young'] & data['surgery_performed']).astype(int)
    
    # 9. COMPREHENSIVE RISK SCORES
    # Calculate multi-system organ dysfunction score
    risk_components = []
    
    if 'vital_signs_severity' in data.columns:
        risk_components.append(data['vital_signs_severity'])
    if 'lesion_severity' in data.columns:
        risk_components.append(data['lesion_severity'])
    if 'abdominal_severity' in data.columns:
        risk_components.append(data['abdominal_severity'])
    if 'pain_severity' in data.columns:
        risk_components.append(data['pain_severity'])
    if 'blood_values_abnormality' in data.columns:
        risk_components.append(data['blood_values_abnormality'])
    if 'perfusion_deficit' in data.columns:
        risk_components.append(data['perfusion_deficit'])
    
    if risk_components:
        data['clinical_severity_score'] = pd.concat(risk_components, axis=1).sum(axis=1)
        
        # Create overall risk stratification
        data['risk_stratification'] = pd.cut(
            data['clinical_severity_score'],
            bins=[-1, 5, 10, 20, 100],
            labels=['low_risk', 'moderate_risk', 'high_risk', 'critical']
        )
        
        # Calculate number of systems affected
        system_affected = []
        for col in ['vital_signs_severity', 'lesion_severity', 'abdominal_severity', 
                   'pain_severity', 'blood_values_abnormality', 'perfusion_deficit']:
            if col in data.columns:
                system_affected.append((data[col] > 0).astype(int))
        
        if system_affected:
            data['multi_system_involvement'] = pd.concat(system_affected, axis=1).sum(axis=1)
            data['severe_multi_system'] = (data['multi_system_involvement'] >= 4).astype(int)
    
    # 10. TIME-BASED FEATURES (if hospital number can be used as proxy for admission order)
    if 'hospital_number' in data.columns:
        # Normalize hospital number to get temporal patterns
        data['normalized_hospital_number'] = (
            data['hospital_number'] - data['hospital_number'].min()
        ) / (data['hospital_number'].max() - data['hospital_number'].min())
        
        # Create time-based bins
        data['admission_period'] = pd.qcut(
            data['normalized_hospital_number'], 
            q=4, 
            labels=['early', 'mid_early', 'mid_late', 'late']
        )
    
    logger.info(f"Feature engineering completed. Total features: {len(data.columns)}")
    return data

# Feature selection with multiple methods
@memory_optimize
def select_features(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
                   n_features: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Select most important features using multiple methods."""
    logger.info(f"Starting feature selection. Initial features: {X_train.shape[1]}")
    
    # Remove constant features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_train_var = variance_selector.fit_transform(X_train)
    X_test_var = variance_selector.transform(X_test)
    
    # Get remaining feature names
    remaining_features = X_train.columns[variance_selector.get_support()].tolist()
    X_train_var = pd.DataFrame(X_train_var, columns=remaining_features, index=X_train.index)
    X_test_var = pd.DataFrame(X_test_var, columns=remaining_features, index=X_test.index)
    
    logger.info(f"Features after variance threshold: {len(remaining_features)}")
    
    # Remove highly correlated features
    correlation_matrix = X_train_var.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > 0.95)]
    
    X_train_corr = X_train_var.drop(columns=to_drop)
    X_test_corr = X_test_var.drop(columns=to_drop)
    
    logger.info(f"Features after correlation filter: {X_train_corr.shape[1]}")
    
    # Use SelectKBest for final selection if still too many features
    if X_train_corr.shape[1] > n_features:
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train_corr, y_train)
        X_test_selected = selector.transform(X_test_corr)
        
        # Get selected feature names
        selected_features = X_train_corr.columns[selector.get_support()].tolist()
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    else:
        X_train_selected = X_train_corr
        X_test_selected = X_test_corr
        selected_features = X_train_corr.columns.tolist()
    
    logger.info(f"Final selected features: {len(selected_features)}")
    return X_train_selected, X_test_selected, selected_features

# Manual ensemble implementation
class ManualEnsemble:
    """Manual ensemble implementation as fallback for AutoGluon"""
    
    def __init__(self, eval_metric='f1_micro'):
        self.eval_metric = eval_metric
        self.models = {}
        self.label_encoder = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train multiple models for ensemble"""
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train_encoded)
        self.models['rf'] = rf
        
        # Train Extra Trees
        logger.info("Training Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        et.fit(X_train, y_train_encoded)
        self.models['et'] = et
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(
            X_train, y_train_encoded,
            eval_set=[(X_val, y_val_encoded)] if X_val is not None else None,
            early_stopping_rounds=50 if X_val is not None else None,
            verbose=False
        )
        self.models['xgb'] = xgb_model
        
        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            num_leaves=64,
            learning_rate=0.01,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            min_data_in_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        lgb_model.fit(
            X_train, y_train_encoded,
            eval_set=[(X_val, y_val_encoded)] if X_val is not None else None,
            callbacks=[lgb.early_stopping(50)] if X_val is not None else None,
            eval_metric='multi_logloss'
        )
        self.models['lgb'] = lgb_model
        
        # Create voting ensemble
        logger.info("Creating voting ensemble...")
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['rf']),
                ('et', self.models['et']),
                ('xgb', self.models['xgb']),
                ('lgb', self.models['lgb'])
            ],
            voting='soft',
            n_jobs=-1
        )
        self.ensemble.fit(X_train, y_train_encoded)
        
    def predict(self, X_test):
        """Make predictions using ensemble"""
        predictions_encoded = self.ensemble.predict(X_test)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, X_test):
        """Get probability predictions"""
        proba = self.ensemble.predict_proba(X_test)
        # Convert to DataFrame with class labels
        proba_df = pd.DataFrame(
            proba,
            columns=[f'p_{cls}' for cls in self.label_encoder.classes_]
        )
        return proba_df
    
    def feature_importance(self, X, feature_names):
        """Calculate average feature importance across models"""
        importances = {}
        
        # Get importances from tree-based models
        for name, model in [('rf', self.models['rf']), ('et', self.models['et']), 
                           ('xgb', self.models['xgb']), ('lgb', self.models['lgb'])]:
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        
        # Average importances
        avg_importance = np.mean(list(importances.values()), axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

# Cross-validation function
def cross_validate_model(train_data: pd.DataFrame, model_class, n_folds: int = 5) -> Dict[str, float]:
    """Perform cross-validation with given model"""
    logger.info(f"Starting {n_folds}-fold cross-validation")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = {'f1_micro': [], 'f1_macro': [], 'f1_weighted': []}
    
    X = train_data.drop(columns=['outcome'])
    y = train_data['outcome']
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Processing fold {fold + 1}/{n_folds}")
        
        # Split data
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = model_class()
        model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        
        # Evaluate
        val_predictions = model.predict(X_fold_val)
        
        cv_scores['f1_micro'].append(f1_score(y_fold_val, val_predictions, average='micro'))
        cv_scores['f1_macro'].append(f1_score(y_fold_val, val_predictions, average='macro'))
        cv_scores['f1_weighted'].append(f1_score(y_fold_val, val_predictions, average='weighted'))
        
        # Clean up
        del model
        gc.collect()
    
    # Calculate mean and std
    results = {}
    for metric, scores in cv_scores.items():
        results[f'{metric}_mean'] = np.mean(scores)
        results[f'{metric}_std'] = np.std(scores)
        logger.info(f"{metric}: {results[f'{metric}_mean']:.4f} (+/- {results[f'{metric}_std']:.4f})")
    
    return results

# Enhanced polynomial features
def add_polynomial_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          columns: List[str], degree: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add polynomial and interaction features for specified columns."""
    logger.info(f"Adding polynomial features of degree {degree}")
    
    # Filter to available columns
    available_cols = [col for col in columns if col in train_df.columns]
    if not available_cols:
        return train_df, test_df
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    
    # Fit and transform
    poly_train = poly.fit_transform(train_df[available_cols])
    poly_test = poly.transform(test_df[available_cols])
    
    # Get feature names
    feature_names = poly.get_feature_names_out(available_cols)
    
    # Create dataframes
    poly_train_df = pd.DataFrame(poly_train, columns=feature_names, index=train_df.index)
    poly_test_df = pd.DataFrame(poly_test, columns=feature_names, index=test_df.index)
    
    # Remove original columns to avoid duplication
    poly_train_df = poly_train_df.drop(columns=available_cols)
    poly_test_df = poly_test_df.drop(columns=available_cols)
    
    # Combine with original data
    train_combined = pd.concat([train_df, poly_train_df], axis=1)
    test_combined = pd.concat([test_df, poly_test_df], axis=1)
    
    logger.info(f"Added {poly_train_df.shape[1]} polynomial features")
    
    return train_combined, test_combined

# Main pipeline
def main():
    """Execute the complete horse health prediction pipeline."""
    try:
        # Load data
        train_data, test_data = load_and_validate_data(
            '/kaggle/input/playground-series-s3e22/train.csv',
            '/kaggle/input/playground-series-s3e22/test.csv'
        )
        
        # Display target distribution
        logger.info("\nTarget distribution:")
        logger.info(f"\n{train_data['outcome'].value_counts()}")
        logger.info(f"\n{train_data['outcome'].value_counts(normalize=True).map(lambda x: f'{x:.2%}')}")
        
        # Apply feature engineering
        train_processed = engineer_features(train_data, is_training=True)
        test_processed = engineer_features(test_data, is_training=False)
        
        # Add polynomial features for key numerical variables
        numerical_columns = ['rectal_temp', 'pulse', 'respiratory_rate', 'packed_cell_volume', 
                           'total_protein', 'temp_deviation', 'pulse_deviation']
        
        train_processed, test_processed = add_polynomial_features(
            train_processed, test_processed, numerical_columns, degree=2
        )
        
        # Prepare features for selection
        feature_cols = [col for col in train_processed.columns 
                       if col not in ['id', 'outcome', 'lesion_pattern', 'admission_period']]
        
        X_train = train_processed[feature_cols]
        y_train = train_processed['outcome']
        X_test = test_processed[feature_cols]
        
        # Convert categorical features to numeric
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        if len(categorical_features) > 0:
            logger.info(f"Converting {len(categorical_features)} categorical features")
            
            # Use pandas get_dummies for one-hot encoding
            X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
            
            # Align columns
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        # Feature selection
        X_train_selected, X_test_selected, selected_features = select_features(
            X_train, y_train, X_test, n_features=150
        )
        
        # Save selected features
        with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
            f.write('\n'.join(selected_features))
        
        # Create final training data
        train_final = pd.concat([train_processed[['outcome']], X_train_selected], axis=1)
        
        # Split for validation
        train_split, val_data = train_test_split(
            train_final, test_size=0.15, random_state=42, stratify=train_final['outcome']
        )
        
        # Handle class imbalance
        logger.info("\nHandling class imbalance...")
        class_counts = train_split['outcome'].value_counts()
        
        if USE_AUTOGLUON:
            logger.info("\nUsing AutoGluon for model training...")
            
            # Calculate sample weights for AutoGluon
            class_weights = {}
            total_samples = len(train_split)
            n_classes = len(class_counts)
            
            for outcome_class, count in class_counts.items():
                class_weights[outcome_class] = total_samples / (n_classes * count)
            
            # Create sample weights
            sample_weights = train_split['outcome'].map(class_weights)
            
            # Configure AutoGluon predictor
            predictor = TabularPredictor(
                label='outcome',
                path=os.path.join(output_dir, 'autogluon_models_final'),
                eval_metric='f1_micro',
                problem_type='multiclass'
            )
            
            # Train with AutoGluon
            start_time = time.time()
            predictor.fit(
                train_split,
                sample_weight=sample_weights,
                time_limit=1200,  # 20 minutes
                presets='best_quality',
                verbosity=2
            )
            training_time = time.time() - start_time
            
            # Get feature importance
            feature_importance = predictor.feature_importance(val_data)
            
        else:
            logger.info("\nUsing manual ensemble for model training...")
            
            # Train manual ensemble
            start_time = time.time()
            predictor = ManualEnsemble(eval_metric='f1_micro')
            
            X_train_split = train_split.drop(columns=['outcome'])
            y_train_split = train_split['outcome']
            X_val = val_data.drop(columns=['outcome'])
            y_val = val_data['outcome']
            
            predictor.fit(X_train_split, y_train_split, X_val, y_val)
            training_time = time.time() - start_time
            
            # Get feature importance
            feature_importance = predictor.feature_importance(
                X_val, selected_features
            )
        
        logger.info(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Model evaluation
        logger.info("\nEvaluating model performance...")
        
        # Validation performance
        if USE_AUTOGLUON:
            val_predictions = predictor.predict(val_data)
            val_proba = predictor.predict_proba(val_data)
        else:
            val_predictions = predictor.predict(val_data.drop(columns=['outcome']))
            val_proba = predictor.predict_proba(val_data.drop(columns=['outcome']))
        
        # Calculate metrics
        val_f1_micro = f1_score(val_data['outcome'], val_predictions, average='micro')
        val_f1_macro = f1_score(val_data['outcome'], val_predictions, average='macro')
        val_f1_weighted = f1_score(val_data['outcome'], val_predictions, average='weighted')
        
        logger.info(f"\nValidation Performance:")
        logger.info(f"F1 Score (Micro): {val_f1_micro:.4f}")
        logger.info(f"F1 Score (Macro): {val_f1_macro:.4f}")
        logger.info(f"F1 Score (Weighted): {val_f1_weighted:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(val_data['outcome'], val_predictions))
        
        # Save feature importance
        logger.info(f"\nTop 20 Important Features:\n{feature_importance.head(20)}")
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'))
        
        # Plot confusion matrices
        plot_confusion_matrices(val_data['outcome'], val_predictions, train_data['outcome'].unique())
        
        # Generate test predictions
        logger.info("\nGenerating test predictions...")
        
        # Prepare test data
        test_final = pd.concat([test_data[['id']], X_test_selected], axis=1)
        
        # Make predictions
        if USE_AUTOGLUON:
            test_predictions = predictor.predict(test_final.drop(columns=['id']))
            test_proba = predictor.predict_proba(test_final.drop(columns=['id']))
        else:
            test_predictions = predictor.predict(test_final.drop(columns=['id']))
            test_proba = predictor.predict_proba(test_final.drop(columns=['id']))
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'outcome': test_predictions
        })
        
        submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
        logger.info(f"\nSubmission saved to: {os.path.join(output_dir, 'submission.csv')}")
        
        # Save probability predictions
        test_proba['id'] = test_data['id']
        test_proba.to_csv(os.path.join(output_dir, 'test_probabilities.csv'), index=False)
        
        # Generate comprehensive report
        generate_performance_report(
            training_time, val_f1_micro, val_f1_macro, 
            val_f1_weighted, len(selected_features), len(train_final.columns) - 1,
            USE_AUTOGLUON
        )
        
        logger.info("\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

def plot_confusion_matrices(y_true, y_pred, classes):
    """Create enhanced confusion matrix visualizations."""
    plt.figure(figsize=(16, 7))
    
    # Raw counts
    plt.subplot(1, 2, 1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(classes), yticklabels=sorted(classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Counts)')
    
    # Percentages
    plt.subplot(1, 2, 2)
    conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_matrix_pct, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=sorted(classes), yticklabels=sorted(classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentages)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()

def generate_performance_report(training_time, val_f1_micro, val_f1_macro, 
                              val_f1_weighted, n_selected_features, 
                              n_total_features, used_autogluon):
    """Generate comprehensive performance report."""
    with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
        f.write("HORSE HEALTH PREDICTION MODEL REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {'AutoGluon' if used_autogluon else 'Manual Ensemble'}\n\n")
        
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Total Features Created: {n_total_features}\n")
        f.write(f"Selected Features: {n_selected_features}\n\n")
        
        f.write("VALIDATION SET PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"F1 Score (Micro): {val_f1_micro:.4f}\n")
        f.write(f"F1 Score (Macro): {val_f1_macro:.4f}\n")
        f.write(f"F1 Score (Weighted): {val_f1_weighted:.4f}\n\n")
        
        f.write("KEY FEATURES\n")
        f.write("-" * 30 + "\n")
        f.write("- Comprehensive domain-specific feature engineering\n")
        f.write("- Multi-method feature selection (variance, correlation, statistical)\n")
        f.write("- Enhanced class balancing with sample weights\n")
        f.write("- Robust error handling and logging\n")
        f.write("- Memory optimization throughout pipeline\n")
        f.write("- Clinical risk stratification features\n")
        f.write("- Multi-system organ dysfunction scoring\n")
        
        if not used_autogluon:
            f.write("\nMANUAL ENSEMBLE COMPONENTS\n")
            f.write("-" * 30 + "\n")
            f.write("- Random Forest (300 trees)\n")
            f.write("- Extra Trees (300 trees)\n")
            f.write("- XGBoost (1000 rounds with early stopping)\n")
            f.write("- LightGBM (1000 rounds with early stopping)\n")
            f.write("- Soft voting ensemble combination\n")

if __name__ == "__main__":
    main()

