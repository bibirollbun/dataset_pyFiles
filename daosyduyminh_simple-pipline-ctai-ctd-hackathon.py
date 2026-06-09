# -*- coding: utf-8 -*-
"""CTAI Construction Material Prediction Pipeline

Complete ensemble approach for construction material forecasting.
Predicts MasterItemNo (classification) and QtyShipped (regression) from project metadata.

Data path: /kaggle/input/ctai-ctd-hackathon/
"""

import numpy as np
import pandas as pd
import warnings
import os
import joblib
from datetime import datetime
import re

# ML Libraries
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# --- Data Loading and Preprocessing Functions ---

def load_and_preprocess_data():
    """Load and preprocess the construction project data."""
    train_path = '/kaggle/input/ctai-ctd-hackathon/train.csv'
    test_path = '/kaggle/input/ctai-ctd-hackathon/test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found. Please check paths.")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train columns: {train_df.columns.tolist()}")
    
    return train_df, test_df

def clean_numeric_columns(df):
    """Clean numeric columns that may contain string formatting."""
    numeric_cols = ['invoiceTotal', 'QtyShipped', 'ExtendedQuantity', 'PriceUOM', 
                   'UnitPrice', 'ExtendedPrice', 'REVISED_ESTIMATE']
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove currency symbols, commas, and convert to numeric
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def extract_date_features(df):
    """Extract features from date columns."""
    date_cols = ['invoiceDate', 'CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE']
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Extract date components
            df[f'{col}_year'] = df[col].dt.year.fillna(0).astype(int)
            df[f'{col}_month'] = df[col].dt.month.fillna(0).astype(int)
            df[f'{col}_day'] = df[col].dt.day.fillna(0).astype(int)
            df[f'{col}_weekday'] = df[col].dt.weekday.fillna(0).astype(int)
            df[f'{col}_quarter'] = df[col].dt.quarter.fillna(0).astype(int)
    
    # Calculate project duration if both dates exist
    if 'CONSTRUCTION_START_DATE' in df.columns and 'SUBSTANTIAL_COMPLETION_DATE' in df.columns:
        duration = (df['SUBSTANTIAL_COMPLETION_DATE'] - df['CONSTRUCTION_START_DATE']).dt.days
        df['project_duration_days'] = duration.fillna(0).astype(float)
    
    # Drop original datetime columns to avoid dtype issues
    for col in date_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df

def engineer_features(df, is_train=True):
    """Engineer features from project metadata."""
    df = df.copy()
    
    # Clean numeric columns
    df = clean_numeric_columns(df)
    
    # Extract date features
    df = extract_date_features(df)
    
    # Engineering derived features
    # Size ratios and densities
    df['rooms_per_floor'] = df['NUMROOMS'] / (df['NUMFLOORS'] + 1)  # +1 to avoid division by zero
    df['beds_per_room'] = df['NUMBEDS'] / (df['NUMROOMS'] + 1)
    df['size_per_floor'] = df['SIZE_BUILDINGSIZE'] / (df['NUMFLOORS'] + 1)
    df['mw_per_sqft'] = df['MW'] / (df['SIZE_BUILDINGSIZE'] + 1)
    
    # Project complexity indicators
    df['is_large_project'] = (df['SIZE_BUILDINGSIZE'] > df['SIZE_BUILDINGSIZE'].median()).astype(int)
    df['is_high_capacity'] = (df['MW'] > df['MW'].median()).astype(int)
    df['is_multi_floor'] = (df['NUMFLOORS'] > 1).astype(int)
    
    # Text feature engineering
    # Project type complexity
    complexity_mapping = {
        'Workspace': 1, 'Health Center': 2, 'Learning Hub': 2, 
        'Ambulatory Care': 3, 'Critical Ops': 4, 'R&D Laboratories': 4,
        'Hospitality Hall': 3, 'Misc Build': 2, 'Pharma Synth': 4
    }
    df['project_complexity'] = df['PROJECT_TYPE'].map(complexity_mapping).fillna(2)
    
    # Market importance
    market_importance = {
        'Enterprise': 3, 'Future Tech': 4, 'Bio Innovation': 4,
        'Wellness': 2, 'Tertiary Learning': 2, 'Misc Market': 1
    }
    df['market_importance'] = df['CORE_MARKET'].map(market_importance).fillna(2)
    
    # State economic indicators (simplified)
    economic_tiers = {
        'Maharashtra': 3, 'Gujarat': 3, 'Karnataka': 3, 'Tamil Nadu': 3,
        'Rajasthan': 2, 'Uttar Pradesh': 2, 'Madhya Pradesh': 2,
        'West Bengal': 2, 'Kerala': 2, 'Punjab': 2
    }
    df['state_economic_tier'] = df['STATE'].map(economic_tiers).fillna(1)
    
    # Fill missing values with appropriate defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def prepare_features(train_df, test_df):
    """Prepare feature matrices for training."""
    
    # Separate target variables (only in train)
    if 'MasterItemNo' in train_df.columns and 'QtyShipped' in train_df.columns:
        # Clean and encode target variables
        y_class_raw = train_df['MasterItemNo'].fillna('Unknown').astype(str)
        y_reg_raw = train_df['QtyShipped']
        
        # For classification, reduce the number of classes by grouping rare classes
        class_counts = y_class_raw.value_counts()
        min_samples = 5  # Minimum samples per class
        frequent_classes = class_counts[class_counts >= min_samples].index.tolist()
        
        # Group rare classes into "Other"
        y_class_grouped = y_class_raw.copy()
        y_class_grouped[~y_class_grouped.isin(frequent_classes)] = 'OTHER_RARE'
        
        # Encode classification target
        target_encoder = LabelEncoder()
        y_class = target_encoder.fit_transform(y_class_grouped)
        
        # Clean regression target
        if y_reg_raw.dtype == 'object':
            y_reg_raw = pd.to_numeric(y_reg_raw.astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        y_reg = y_reg_raw.fillna(y_reg_raw.median()).values
        
        print(f"Target encoding - Classes: {len(target_encoder.classes_)} (reduced from {len(class_counts)})")
        print(f"Frequent classes: {len(frequent_classes)}, Rare classes grouped: {len(class_counts) - len(frequent_classes)}")
        print(f"QtyShipped range: {y_reg.min():.2f} to {y_reg.max():.2f}")
        
        # Store original class mapping for submission
        class_mapping = {'encoder': target_encoder, 'original_classes': y_class_raw.values}
        
        # Drop target columns from features
        feature_cols = [col for col in train_df.columns if col not in ['MasterItemNo', 'QtyShipped', 'id']]
    else:
        y_class, y_reg, class_mapping = None, None, None
        feature_cols = [col for col in train_df.columns if col not in ['id']]
    
    # Select common feature columns between train and test
    common_cols = [col for col in feature_cols if col in test_df.columns]
    
    X_train = train_df[common_cols].copy()
    X_test = test_df[common_cols].copy()
    
    # Engineer features
    X_train = engineer_features(X_train, is_train=True)
    X_test = engineer_features(X_test, is_train=False)
    
    # Get common columns after feature engineering
    common_engineered_cols = [col for col in X_train.columns if col in X_test.columns]
    X_train = X_train[common_engineered_cols]
    X_test = X_test[common_engineered_cols]
    
    # Ensure all columns are numeric
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            # Try to convert to numeric first
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            
            # If still object type, use label encoding
            if X_train[col].dtype == 'object':
                le = LabelEncoder()
                # Fit on combined data to ensure consistent encoding
                combined_values = pd.concat([X_train[col], X_test[col]]).astype(str).fillna('Unknown')
                le.fit(combined_values)
                
                X_train[col] = le.transform(X_train[col].astype(str).fillna('Unknown'))
                X_test[col] = le.transform(X_test[col].astype(str).fillna('Unknown'))
    
    # Fill any remaining NaN values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Ensure all columns are numeric types
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    
    feature_sets = {
        'original': (X_train.values, X_test.values),
        'scaled': (X_train_scaled, X_test_scaled),
    }
    
    print(f"Feature matrix shape: {X_train_scaled.shape}")
    print(f"Final feature columns: {len(X_train.columns)}")
    
    return feature_sets, y_class, y_reg, X_train.columns.tolist(), class_mapping

def create_base_models():
    """Create base models for both classification and regression."""
    
    # Classification models
    clf_models = [
        {
            'name': 'RandomForest_Clf',
            'model': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1),
            'feature_set': 'original'
        },
        {
            'name': 'ExtraTrees_Clf', 
            'model': ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=42, n_jobs=-1),
            'feature_set': 'scaled'
        },
        {
            'name': 'LogisticRegression_Clf',
            'model': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'feature_set': 'scaled'
        },
        {
            'name': 'LightGBM_Clf',
            'model': lgb.LGBMClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1),
            'feature_set': 'original'
        }
    ]
    
    # Regression models  
    reg_models = [
        {
            'name': 'RandomForest_Reg',
            'model': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1),
            'feature_set': 'original'
        },
        {
            'name': 'ExtraTrees_Reg',
            'model': ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_split=10, random_state=42, n_jobs=-1), 
            'feature_set': 'scaled'
        },
        {
            'name': 'LinearRegression_Reg',
            'model': LinearRegression(),
            'feature_set': 'scaled'
        },
        {
            'name': 'LightGBM_Reg',
            'model': lgb.LGBMRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1),
            'feature_set': 'original'
        }
    ]
    
    return clf_models, reg_models

def evaluate_models(clf_models, reg_models, feature_sets, y_class, y_reg):
    """Evaluate models with cross-validation."""
    print("\n" + "="*60)
    print("EVALUATING BASE MODELS WITH 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store predictions and scores
    clf_predictions = {}
    reg_predictions = {}
    clf_cv_preds = {}
    reg_cv_preds = {}
    model_scores = []
    
    # Classification models
    print("\nClassification Models:")
    print("-" * 40)
    
    for model_config in clf_models:
        model_name = model_config['name']
        model = model_config['model']
        feature_set_name = model_config['feature_set']
        
        X_train, X_test = feature_sets[feature_set_name]
        
        # Cross-validation
        cv_scores_acc = []
        cv_scores_f1 = []
        cv_preds = np.zeros(len(y_class))
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_class[train_idx], y_class[val_idx]
            
            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                cv_scores_acc.append(acc)
                cv_scores_f1.append(f1)
                cv_preds[val_idx] = y_pred
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                cv_scores_acc.append(0)
                cv_scores_f1.append(0)
                cv_preds[val_idx] = 0
        
        avg_acc = np.mean(cv_scores_acc)
        avg_f1 = np.mean(cv_scores_f1)
        
        print(f"{model_name:<25} | Accuracy: {avg_acc:.4f} ± {np.std(cv_scores_acc):.4f} | F1: {avg_f1:.4f} ± {np.std(cv_scores_f1):.4f}")
        
        # Train on full data and predict test
        try:
            model.fit(X_train, y_class)
            test_pred = model.predict(X_test)
        except Exception as e:
            print(f"Error training {model_name} on full data: {e}")
            test_pred = np.zeros(X_test.shape[0])
        
        clf_predictions[model_name] = test_pred
        clf_cv_preds[model_name] = cv_preds
        
        model_scores.append({
            'model': model_name,
            'type': 'classification',
            'accuracy': avg_acc,
            'f1': avg_f1
        })
    
    # Regression models
    print("\nRegression Models:")
    print("-" * 40)
    
    for model_config in reg_models:
        model_name = model_config['name'] 
        model = model_config['model']
        feature_set_name = model_config['feature_set']
        
        X_train, X_test = feature_sets[feature_set_name]
        
        # Cross-validation
        cv_scores_mae = []
        cv_preds = np.zeros(len(y_reg))
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_reg[train_idx], y_reg[val_idx]
            
            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores_mae.append(mae)
                cv_preds[val_idx] = y_pred
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                cv_scores_mae.append(y_reg.std())
                cv_preds[val_idx] = y_reg.mean()
        
        avg_mae = np.mean(cv_scores_mae)
        
        # Convert MAE to score (for comparison)
        mae_range = y_reg.max() - y_reg.min()
        norm_mae = avg_mae / mae_range if mae_range > 0 else 0
        reg_score = max(0, 1 - norm_mae)
        
        print(f"{model_name:<25} | MAE: {avg_mae:.4f} ± {np.std(cv_scores_mae):.4f} | Reg Score: {reg_score:.4f}")
        
        # Train on full data and predict test
        try:
            model.fit(X_train, y_reg)
            test_pred = model.predict(X_test)
            # Ensure non-negative predictions
            test_pred = np.maximum(test_pred, 0)
        except Exception as e:
            print(f"Error training {model_name} on full data: {e}")
            test_pred = np.full(X_test.shape[0], y_reg.mean())
        
        reg_predictions[model_name] = test_pred
        reg_cv_preds[model_name] = cv_preds
        
        model_scores.append({
            'model': model_name,
            'type': 'regression', 
            'mae': avg_mae,
            'reg_score': reg_score
        })
    
    return clf_predictions, reg_predictions, clf_cv_preds, reg_cv_preds, model_scores, y_class, y_reg

def main_train():
    """Main training pipeline."""
    print("="*80)
    print("CELL 1: CTAI CONSTRUCTION MATERIAL PREDICTION TRAINING")
    print("="*80)
    
    try:
        # Load data
        print("1. Loading data...")
        train_df, test_df = load_and_preprocess_data()
        
        # Prepare features
        print("\n2. Engineering features...")
        feature_sets, y_class, y_reg, feature_names, class_mapping = prepare_features(train_df, test_df)
        
        # Create models
        print("\n3. Creating base models...")
        clf_models, reg_models = create_base_models()
        print(f"Created {len(clf_models)} classification and {len(reg_models)} regression models.")
        
        # Evaluate models
        print("\n4. Training and evaluating models...")
        clf_preds, reg_preds, clf_cv_preds, reg_cv_preds, model_scores, y_class, y_reg = evaluate_models(
            clf_models, reg_models, feature_sets, y_class, y_reg
        )
        
        # Save results
        print("\n5. Saving training results...")
        output_dir = "training_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(clf_preds, os.path.join(output_dir, 'clf_predictions.joblib'))
        joblib.dump(reg_preds, os.path.join(output_dir, 'reg_predictions.joblib'))
        joblib.dump(clf_cv_preds, os.path.join(output_dir, 'clf_cv_preds.joblib'))
        joblib.dump(reg_cv_preds, os.path.join(output_dir, 'reg_cv_preds.joblib'))
        joblib.dump(model_scores, os.path.join(output_dir, 'model_scores.joblib'))
        joblib.dump(y_class, os.path.join(output_dir, 'y_class.joblib'))
        joblib.dump(y_reg, os.path.join(output_dir, 'y_reg.joblib'))
        joblib.dump(test_df['id'].values, os.path.join(output_dir, 'test_ids.joblib'))
        joblib.dump(class_mapping, os.path.join(output_dir, 'class_mapping.joblib'))
        
        print(f"Results saved to '{output_dir}' directory.")
        print("\n" + "="*80)
        print("CELL 1 COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main_train()

# ============================================================================
# Cell 2: Advanced Ensembling and Model Selection
# ============================================================================

def composite_score(y_true_class, y_pred_class, y_true_reg, y_pred_reg):
    """Calculate the composite score as per competition rules."""
    # Classification metrics
    acc = accuracy_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    
    # Regression metrics
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    reg_range = y_true_reg.max() - y_true_reg.min() if y_true_reg.max() != y_true_reg.min() else 1
    norm_mae = mae / reg_range
    reg_score = max(0, 1 - norm_mae)
    
    # Composite score: 0.25*accuracy + 0.25*F1 + 0.5*reg_score
    composite = 0.25 * acc + 0.25 * f1 + 0.5 * reg_score
    
    return composite, acc, f1, reg_score

def create_ensemble_predictions(clf_preds, reg_preds, clf_cv_preds, reg_cv_preds, y_class, y_reg):
    """Create ensemble predictions using various methods."""
    print("\n" + "="*60)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*60)
    
    from scipy import stats
    
    ensemble_methods = {}
    
    # Simple ensemble methods
    print("\nEvaluating ensemble methods...")
    
    # 1. Voting/Averaging ensemble
    clf_names = list(clf_preds.keys())
    reg_names = list(reg_preds.keys())
    
    # Classification: majority voting
    clf_ensemble_cv = np.zeros(len(y_class))
    clf_ensemble_test = np.zeros(len(list(clf_preds.values())[0]))
    
    for i in range(len(y_class)):
        votes = [clf_cv_preds[name][i] for name in clf_names]
        clf_ensemble_cv[i] = stats.mode(votes, keepdims=True)[0][0]
    
    for i in range(len(list(clf_preds.values())[0])):
        votes = [clf_preds[name][i] for name in clf_names]
        clf_ensemble_test[i] = stats.mode(votes, keepdims=True)[0][0]
    
    # Regression: simple average
    reg_ensemble_cv = np.mean([reg_cv_preds[name] for name in reg_names], axis=0)
    reg_ensemble_test = np.mean([reg_preds[name] for name in reg_names], axis=0)
    
    # Evaluate ensemble
    composite, acc, f1, reg_score = composite_score(y_class, clf_ensemble_cv, y_reg, reg_ensemble_cv)
    
    ensemble_methods['simple_ensemble'] = {
        'clf_test': clf_ensemble_test,
        'reg_test': reg_ensemble_test,
        'cv_score': composite,
        'accuracy': acc,
        'f1': f1,
        'reg_score': reg_score
    }
    
    print(f"Simple Ensemble | Composite: {composite:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Reg: {reg_score:.4f}")
    
    # 2. Weighted ensemble based on individual model performance
    from collections import defaultdict
    model_performance = defaultdict(dict)
    
    # Calculate individual model CV scores
    for clf_name in clf_names:
        clf_cv = clf_cv_preds[clf_name] 
        acc_cv = accuracy_score(y_class, clf_cv)
        f1_cv = f1_score(y_class, clf_cv, average='weighted', zero_division=0)
        model_performance[clf_name]['acc'] = acc_cv
        model_performance[clf_name]['f1'] = f1_cv
        model_performance[clf_name]['clf_weight'] = (acc_cv + f1_cv) / 2
    
    for reg_name in reg_names:
        reg_cv = reg_cv_preds[reg_name]
        mae_cv = mean_absolute_error(y_reg, reg_cv)
        reg_range = y_reg.max() - y_reg.min() if y_reg.max() != y_reg.min() else 1
        reg_score_cv = max(0, 1 - mae_cv/reg_range) 
        model_performance[reg_name]['reg_score'] = reg_score_cv
        model_performance[reg_name]['reg_weight'] = reg_score_cv
    
    # Weighted classification ensemble (use simple voting for now)
    clf_weighted_cv = clf_ensemble_cv.copy()
    clf_weighted_test = clf_ensemble_test.copy()
    
    # Weighted regression ensemble
    reg_weights = np.array([model_performance[name]['reg_weight'] for name in reg_names])
    reg_weights = reg_weights / reg_weights.sum() if reg_weights.sum() > 0 else np.ones(len(reg_names)) / len(reg_names)
    
    reg_weighted_cv = np.average([reg_cv_preds[name] for name in reg_names], axis=0, weights=reg_weights)
    reg_weighted_test = np.average([reg_preds[name] for name in reg_names], axis=0, weights=reg_weights)
    
    # Evaluate weighted ensemble
    composite, acc, f1, reg_score = composite_score(y_class, clf_weighted_cv, y_reg, reg_weighted_cv)
    
    ensemble_methods['weighted_ensemble'] = {
        'clf_test': clf_weighted_test,
        'reg_test': reg_weighted_test,
        'cv_score': composite,
        'accuracy': acc, 
        'f1': f1,
        'reg_score': reg_score
    }
    
    print(f"Weighted Ensemble | Composite: {composite:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Reg: {reg_score:.4f}")
    
    # 3. Best individual models combination
    best_clf_name = max(clf_names, key=lambda x: model_performance[x]['clf_weight'])
    best_reg_name = max(reg_names, key=lambda x: model_performance[x]['reg_weight'])
    
    composite, acc, f1, reg_score = composite_score(
        y_class, clf_cv_preds[best_clf_name], 
        y_reg, reg_cv_preds[best_reg_name]
    )
    
    ensemble_methods['best_individual'] = {
        'clf_test': clf_preds[best_clf_name],
        'reg_test': reg_preds[best_reg_name], 
        'cv_score': composite,
        'accuracy': acc,
        'f1': f1,
        'reg_score': reg_score
    }
    
    print(f"Best Individual | Composite: {composite:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Reg: {reg_score:.4f}")
    print(f"  (Classification: {best_clf_name}, Regression: {best_reg_name})")
    
    return ensemble_methods

def decode_predictions(encoded_preds, class_mapping):
    """Decode predictions back to original class labels."""
    encoder = class_mapping['encoder']
    original_classes = class_mapping['original_classes']
    
    # Decode the grouped predictions
    decoded_grouped = encoder.inverse_transform(encoded_preds.astype(int))
    
    # For predictions that were grouped as 'OTHER_RARE', we could:
    # 1. Keep them as 'OTHER_RARE' 
    # 2. Replace with most frequent rare class
    # 3. Use some other strategy
    
    # For now, let's replace 'OTHER_RARE' with the most frequent class overall
    most_frequent_class = pd.Series(original_classes).value_counts().index[0]
    decoded_final = np.where(decoded_grouped == 'OTHER_RARE', most_frequent_class, decoded_grouped)
    
    return decoded_final

def save_ensemble_submissions(ensemble_methods, test_ids, class_mapping):
    """Save ensemble submissions."""
    print("\n" + "="*60)
    print("SAVING ENSEMBLE SUBMISSIONS")
    print("="*60)
    
    # Sort methods by CV score
    sorted_methods = sorted(ensemble_methods.items(), key=lambda x: x[1]['cv_score'], reverse=True)
    
    print("\nEnsemble method rankings:")
    for i, (method_name, method_data) in enumerate(sorted_methods, 1):
        score = method_data['cv_score']
        print(f"{i}. {method_name:<20} | Composite Score: {score:.4f}")
    
    # Save top 3 methods
    for rank, (method_name, method_data) in enumerate(sorted_methods[:3], 1):
        clf_pred_encoded = method_data['clf_test'].astype(int)
        reg_pred = np.maximum(method_data['reg_test'], 0)  # Ensure non-negative
        
        # Decode classification predictions back to original labels
        clf_pred_original = decode_predictions(clf_pred_encoded, class_mapping)
        
        submission_df = pd.DataFrame({
            'id': test_ids,
            'MasterItemNo': clf_pred_original,
            'QtyShipped': reg_pred
        })
        
        filename = f'ensemble_submission_{rank}_{method_name}.csv'
        submission_df.to_csv(filename, index=False)
        
        print(f"Saved: {filename}")
        print(f"  - MasterItemNo samples: {clf_pred_original[:5]}")
        print(f"  - QtyShipped range: {reg_pred.min():.2f} to {reg_pred.max():.2f}")

def main_ensemble():
    """Main ensembling pipeline."""
    print("="*80)
    print("CELL 2: ENSEMBLE CREATION AND EVALUATION")
    print("="*80)
    
    input_dir = "training_outputs"
    
    try:
        # Load training results
        print("1. Loading training results...")
        clf_preds = joblib.load(os.path.join(input_dir, 'clf_predictions.joblib'))
        reg_preds = joblib.load(os.path.join(input_dir, 'reg_predictions.joblib'))
        clf_cv_preds = joblib.load(os.path.join(input_dir, 'clf_cv_preds.joblib'))
        reg_cv_preds = joblib.load(os.path.join(input_dir, 'reg_cv_preds.joblib'))
        y_class = joblib.load(os.path.join(input_dir, 'y_class.joblib'))
        y_reg = joblib.load(os.path.join(input_dir, 'y_reg.joblib'))
        test_ids = joblib.load(os.path.join(input_dir, 'test_ids.joblib'))
        class_mapping = joblib.load(os.path.join(input_dir, 'class_mapping.joblib'))
        
        print("Training results loaded successfully.")
        
        # Create ensemble predictions
        print("\n2. Creating ensemble predictions...")
        ensemble_methods = create_ensemble_predictions(
            clf_preds, reg_preds, clf_cv_preds, reg_cv_preds, y_class, y_reg
        )
        
        # Save ensemble submissions
        print("\n3. Saving ensemble submissions...")
        save_ensemble_submissions(ensemble_methods, test_ids, class_mapping)
        
        print("\n" + "="*80)
        print("CELL 2 COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"Error in ensemble pipeline: {str(e)}")
        raise

# ============================================================================  
# Cell 3: Final Submission Creation
# ============================================================================

def create_final_submission():
    """Create final weighted submission from ensemble results."""
    print("="*80)
    print("CELL 3: FINAL SUBMISSION CREATION")
    print("="*80)
    
    # Find ensemble submission files
    import glob
    ensemble_files = glob.glob('ensemble_submission_*.csv')
    
    if len(ensemble_files) == 0:
        print("No ensemble submission files found. Please run Cell 2 first.")
        return
    
    print(f"Found {len(ensemble_files)} ensemble submission files:")
    for f in ensemble_files:
        print(f"  - {f}")
    
    # Load submissions
    submissions = []
    for file in ensemble_files[:3]:  # Top 3
        df = pd.read_csv(file)
        submissions.append(df)
        print(f"Loaded {file} with shape {df.shape}")
    
    if len(submissions) == 0:
        print("No valid submission files to process.")
        return
    
    # Create weighted final submission
    print("\nCreating weighted final submission...")
    
    # Use the best performing ensemble for final submission
    final_df = submissions[0].copy()
    
    # For classification, use the best ensemble result
    final_df['MasterItemNo'] = submissions[0]['MasterItemNo']
    
    # For regression, use weighted average of top performers
    if len(submissions) > 1:
        weights = [0.5, 0.3, 0.2][:len(submissions)]
        weights = np.array(weights) / sum(weights)  # Normalize
        
        # Weighted average for regression
        all_reg_preds = np.array([sub['QtyShipped'].values for sub in submissions])
        final_reg_preds = np.average(all_reg_preds, axis=0, weights=weights)
        final_df['QtyShipped'] = np.maximum(final_reg_preds, 0)  # Ensure non-negative
    
    # Save final submission
    final_df.to_csv('submission.csv', index=False)
    
    print("Final submission created!")
    print(f"Shape: {final_df.shape}")
    print(f"MasterItemNo unique values: {final_df['MasterItemNo'].nunique()}")
    print(f"QtyShipped range: {final_df['QtyShipped'].min():.2f} to {final_df['QtyShipped'].max():.2f}")
    
    # Preview
    print("\nSubmission preview:")
    print(final_df.head(10))
    
    print("\n" + "="*80)
    print("CELL 3 COMPLETED SUCCESSFULLY!")
    print("Final submission saved as 'submission.csv'")
    print("="*80)

if __name__ == "__main__":
    # Uncomment the function you want to run:
    # main_train()      # Run Cell 1
    # main_ensemble()   # Run Cell 2  
    # create_final_submission()  # Run Cell 3
    
    # For full pipeline, run all three:
    main_train()
    main_ensemble()
    create_final_submission()

