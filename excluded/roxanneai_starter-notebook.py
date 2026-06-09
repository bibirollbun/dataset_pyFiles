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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install required packages (run this first in a new environment)
# !pip install pandas numpy scikit-learn lightgbm scipy warnings

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# ===================== DATA LOADING AND CLEANING =====================

def clean_extended_quantity(value):
    """Clean ExtendedQuantity and QtyShipped columns."""
    if pd.isna(value) or value == '' or value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convert to string for processing
    value = str(value)
    
    # Handle newlines
    if '\n' in value:
        return 3.0
    
    # Remove commas and trailing dashes
    value = value.replace(',', '')
    while value.endswith('-'):
        value = value[:-1]
    
    # Handle EA suffix
    if value.endswith(' EA'):
        value = value[:-3].strip()
    
    # Try to convert to float
    try:
        return float(value)
    except:
        return np.nan

def load_and_clean_data():
    """Load and clean the dataset."""
    print("Loading data...")
    
    # Load with date parsing
    train = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/train.csv',
                       parse_dates=['CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE', 'invoiceDate'],
                       date_format={'CONSTRUCTION_START_DATE': '%m/%d/%Y %H:%M',
                                   'SUBSTANTIAL_COMPLETION_DATE': '%m/%d/%Y %H:%M',
                                   'invoiceDate': 'mixed'})
    
    test = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/test.csv',
                      parse_dates=['CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE', 'invoiceDate'],
                      date_format={'CONSTRUCTION_START_DATE': '%m/%d/%Y %H:%M',
                                  'SUBSTANTIAL_COMPLETION_DATE': '%m/%d/%Y %H:%M',
                                  'invoiceDate': '%m/%d/%Y'})
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    # Clean numeric columns
    print("Cleaning numeric columns...")
    for df in [train, test]:
        # Clean ExtendedQuantity
        if 'ExtendedQuantity' in df.columns:
            df['ExtendedQuantity'] = df['ExtendedQuantity'].apply(clean_extended_quantity)
        
        # Clean other numeric columns that might have string formatting
        numeric_cols = ['invoiceTotal', 'UnitPrice', 'ExtendedPrice']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean QtyShipped in train
    if 'QtyShipped' in train.columns:
        train['QtyShipped'] = train['QtyShipped'].apply(clean_extended_quantity)
    
    return train, test

# ===================== FEATURE ENGINEERING =====================

def handle_infinity_and_nan(df):
    """Replace infinity values and handle NaN properly."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Replace infinity with NaN first
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median (calculate median excluding NaN)
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)
    
    return df

def engineer_features(df, label_encoders=None, fit_encoders=True):
    """Engineer features from the dataset."""
    df = df.copy()
    
    # Extract date features
    date_cols = ['CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE', 'invoiceDate']
    for col in date_cols:
        if col in df.columns:
            df[f'{col}_year'] = df[col].dt.year.fillna(0)
            df[f'{col}_month'] = df[col].dt.month.fillna(0)
            df[f'{col}_quarter'] = df[col].dt.quarter.fillna(0)
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek.fillna(0)
    
    # Calculate project duration
    if 'CONSTRUCTION_START_DATE' in df.columns and 'SUBSTANTIAL_COMPLETION_DATE' in df.columns:
        df['project_duration_days'] = (df['SUBSTANTIAL_COMPLETION_DATE'] - df['CONSTRUCTION_START_DATE']).dt.days
        df['project_duration_days'] = df['project_duration_days'].fillna(df['project_duration_days'].median())
    
    # Size-based features - handle division by zero
    df['size_per_floor'] = np.where(df['NUMFLOORS'] > 0, 
                                    df['SIZE_BUILDINGSIZE'] / df['NUMFLOORS'], 
                                    df['SIZE_BUILDINGSIZE'])
    df['rooms_per_floor'] = np.where(df['NUMFLOORS'] > 0, 
                                     df['NUMROOMS'] / df['NUMFLOORS'], 
                                     df['NUMROOMS'])
    df['beds_per_room'] = np.where(df['NUMROOMS'] > 0, 
                                   df['NUMBEDS'] / df['NUMROOMS'], 
                                   df['NUMBEDS'])
    df['mw_per_sqft'] = np.where(df['SIZE_BUILDINGSIZE'] > 0, 
                                 df['MW'] / df['SIZE_BUILDINGSIZE'], 
                                 0)
    
    # Project indicators
    df['is_large_project'] = (df['SIZE_BUILDINGSIZE'] > df['SIZE_BUILDINGSIZE'].quantile(0.75)).astype(int)
    df['is_multi_floor'] = (df['NUMFLOORS'] > 1).astype(int)
    df['has_many_rooms'] = (df['NUMROOMS'] > df['NUMROOMS'].quantile(0.75)).astype(int)
    
    # Price ratios - handle division by zero
    df['price_per_qty'] = np.where(df['ExtendedQuantity'] > 0, 
                                   df['ExtendedPrice'] / df['ExtendedQuantity'], 
                                   0)
    df['unit_to_extended_ratio'] = np.where(df['ExtendedPrice'] > 0, 
                                           df['UnitPrice'] / df['ExtendedPrice'], 
                                           0)
    
    # Log transformations for skewed features
    for col in ['SIZE_BUILDINGSIZE', 'invoiceTotal', 'ExtendedPrice', 'REVISED_ESTIMATE']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0).clip(lower=0))
    
    # Encode categorical variables
    categorical_cols = ['PROJECT_CITY', 'STATE', 'CORE_MARKET', 'PROJECT_TYPE', 'UOM', 'PriceUOM']
    
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            
            if fit_encoders:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                label_encoders[col] = le
            else:
                if col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen categories
                    df[col + '_encoded'] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    df[col + '_encoded'] = -1
    
    # Create text features from ItemDescription
    if 'ItemDescription' in df.columns:
        df['item_desc_length'] = df['ItemDescription'].fillna('').str.len()
        df['item_desc_word_count'] = df['ItemDescription'].fillna('').str.split().str.len()
        
        # Common keywords
        keywords = ['cable', 'pipe', 'panel', 'wire', 'steel', 'concrete', 'door', 'window']
        for keyword in keywords:
            df[f'has_{keyword}'] = df['ItemDescription'].fillna('').str.lower().str.contains(keyword).astype(int)
    
    # Handle infinity and NaN values
    df = handle_infinity_and_nan(df)
    
    # Drop original date columns
    for col in date_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Drop original categorical columns (keeping encoded versions)
    for col in categorical_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df, label_encoders

# ===================== MODEL TRAINING =====================

def prepare_features(train_df, test_df):
    """Prepare feature matrices for training."""
    
    # Engineer features
    print("Engineering features...")
    train_featured, label_encoders = engineer_features(train_df, fit_encoders=True)
    test_featured, _ = engineer_features(test_df, label_encoders=label_encoders, fit_encoders=False)
    
    # Prepare classification target - MasterItemNo is categorical!
    print("Encoding MasterItemNo...")
    target_encoder = LabelEncoder()
    
    # Handle MasterItemNo - it's a categorical string column
    y_class_labels = train_df['MasterItemNo'].fillna('UNKNOWN').astype(str)
    y_class = target_encoder.fit_transform(y_class_labels)
    
    # Prepare regression target
    y_reg = train_df['QtyShipped'].fillna(train_df['QtyShipped'].median())
    
    # Select feature columns (exclude targets and IDs)
    feature_cols = [col for col in train_featured.columns 
                   if col not in ['id', 'MasterItemNo', 'QtyShipped', 'PROJECTNUMBER', 
                                'invoiceId', 'ItemDescription', 'PROJECT_COUNTRY']]
    
    # Ensure common columns between train and test
    common_cols = [col for col in feature_cols if col in test_featured.columns]
    
    X_train = train_featured[common_cols].values
    X_test = test_featured[common_cols].values
    
    # Final check for infinity values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features shape - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Targets - Classes: {len(target_encoder.classes_)}, Regression range: [{y_reg.min():.0f}, {y_reg.max():.0f}]")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_class, y_reg, target_encoder, train_featured, test_featured

def train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_class, y_reg, train_df, test_df):
    """Train ensemble of models."""
    print("\nTraining models...")
    
    predictions = {}
    
    # Remove samples with invalid targets for training
    valid_class = ~pd.isna(y_class)
    valid_reg = ~pd.isna(y_reg)
    
    # ========== Classification Models ==========
    
    # 1. Simple ItemDescription-based classifier (baseline)
    print("Training ItemDescription classifier...")
    if 'ItemDescription' in train_df.columns:
        item_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        item_clf = make_pipeline(item_encoder, DecisionTreeClassifier(max_depth=20, min_samples_leaf=2, random_state=42))
        
        train_items = train_df.loc[valid_class, 'ItemDescription'].fillna('Unknown').values.reshape(-1, 1)
        test_items = test_df['ItemDescription'].fillna('Unknown').values.reshape(-1, 1)
        
        item_clf.fit(train_items, y_class[valid_class])
        predictions['clf_item'] = item_clf.predict(test_items)
    
    # 2. Random Forest Classifier
    print("Training Random Forest classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=3, 
                                   random_state=42, n_jobs=-1)
    rf_clf.fit(X_train[valid_class], y_class[valid_class])
    predictions['clf_rf'] = rf_clf.predict(X_test)
    
    # 3. LightGBM Classifier
    print("Training LightGBM classifier...")
    lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, 
                                num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    lgb_clf.fit(X_train[valid_class], y_class[valid_class])
    predictions['clf_lgb'] = lgb_clf.predict(X_test)
    
    # ========== Regression Models ==========
    
    # 1. Simple ExtendedQuantity-based regressor (baseline)
    print("Training ExtendedQuantity regressor...")
    if 'ExtendedQuantity' in train_df.columns:
        ext_reg = make_pipeline(SimpleImputer(strategy='median'), 
                              DecisionTreeRegressor(max_depth=15, min_samples_leaf=3, random_state=42))
        
        train_ext = train_df.loc[valid_reg, 'ExtendedQuantity'].values.reshape(-1, 1)
        test_ext = test_df['ExtendedQuantity'].values.reshape(-1, 1)
        
        # Handle any remaining infinity values
        train_ext = np.nan_to_num(train_ext, nan=0.0, posinf=1e10, neginf=-1e10)
        test_ext = np.nan_to_num(test_ext, nan=0.0, posinf=1e10, neginf=-1e10)
        
        ext_reg.fit(train_ext, y_reg[valid_reg])
        predictions['reg_ext'] = ext_reg.predict(test_ext)
    
    # 2. Random Forest Regressor
    print("Training Random Forest regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1)
    rf_reg.fit(X_train[valid_reg], y_reg[valid_reg])
    predictions['reg_rf'] = rf_reg.predict(X_test)
    
    # 3. Extra Trees Regressor
    print("Training Extra Trees regressor...")
    et_reg = ExtraTreesRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3,
                                random_state=42, n_jobs=-1)
    et_reg.fit(X_train_scaled[valid_reg], y_reg[valid_reg])
    predictions['reg_et'] = et_reg.predict(X_test_scaled)
    
    # 4. LightGBM Regressor
    print("Training LightGBM regressor...")
    lgb_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=10, learning_rate=0.1,
                               num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    lgb_reg.fit(X_train[valid_reg], y_reg[valid_reg])
    predictions['reg_lgb'] = lgb_reg.predict(X_test)
    
    return predictions

def create_ensemble_predictions(predictions):
    """Create ensemble predictions."""
    print("\nCreating ensemble predictions...")
    
    # Classification ensemble - majority voting
    clf_preds = [predictions[k] for k in predictions if k.startswith('clf_')]
    
    # Use mode for each sample
    clf_ensemble = np.zeros(len(clf_preds[0]))
    for i in range(len(clf_preds[0])):
        votes = [int(pred[i]) for pred in clf_preds]
        clf_ensemble[i] = stats.mode(votes, keepdims=False)[0]
    
    # Regression ensemble - weighted average
    weights = {
        'reg_ext': 0.35,   # ExtendedQuantity is highly correlated with QtyShipped
        'reg_rf': 0.25,
        'reg_et': 0.25,
        'reg_lgb': 0.15
    }
    
    reg_preds = []
    reg_weights = []
    for k in predictions:
        if k.startswith('reg_'):
            if k in weights:
                reg_preds.append(predictions[k])
                reg_weights.append(weights[k])
    
    # Normalize weights
    reg_weights = np.array(reg_weights) / sum(reg_weights)
    reg_ensemble = np.average(reg_preds, axis=0, weights=reg_weights)
    
    # Ensure non-negative predictions
    reg_ensemble = np.maximum(reg_ensemble, 0)
    
    return clf_ensemble.astype(int), reg_ensemble

# ===================== MAIN EXECUTION =====================

if __name__ == "__main__":
    print("="*70)
    print("CTAI Material Prediction - Complete Solution")
    print("="*70)
    
    # Load and clean data
    train_df, test_df = load_and_clean_data()
    
    # Keep only samples with valid targets for training
    train_df_clean = train_df.dropna(subset=['QtyShipped'])
    
    # Prepare features
    X_train, X_test, X_train_scaled, X_test_scaled, y_class, y_reg, target_encoder, train_featured, test_featured = prepare_features(
        train_df_clean, test_df
    )
    
    # Train models
    predictions = train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_class, y_reg, train_df_clean, test_df)
    
    # Create ensemble
    final_class_encoded, final_reg = create_ensemble_predictions(predictions)
    
    # Decode classification predictions back to original MasterItemNo values
    print("\nDecoding predictions...")
    final_class = target_encoder.inverse_transform(final_class_encoded.astype(int))
    
    # Create submission
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'MasterItemNo': final_class,
        'QtyShipped': np.round(final_reg, 2)
    })
    
    # Validate submission
    assert submission['id'].notna().all(), "id has NaN values"
    assert submission['MasterItemNo'].notna().all(), "MasterItemNo has NaN values"
    assert submission['QtyShipped'].notna().all(), "QtyShipped has NaN values"
    assert (submission['QtyShipped'] >= 0).all(), "QtyShipped has negative values"
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    
    print("\nSubmission Summary:")
    print(f"Shape: {submission.shape}")
    print(f"MasterItemNo unique values: {submission['MasterItemNo'].nunique()}")
    print(f"QtyShipped range: [{submission['QtyShipped'].min():.2f}, {submission['QtyShipped'].max():.2f}]")
    print(f"QtyShipped mean: {submission['QtyShipped'].mean():.2f}")
    print(f"QtyShipped median: {submission['QtyShipped'].median():.2f}")
    
    print("\nFirst 10 rows:")
    print(submission.head(10))
    
    print("\nMasterItemNo value counts (top 10):")
    print(submission['MasterItemNo'].value_counts().head(10))
    
    print("\n" + "="*70)
    print("Submission file 'submission.csv' created successfully!")
    print("="*70)

