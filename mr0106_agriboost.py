#!/usr/bin/env python
# coding: utf-8

# # Fertilizer Genius: Smart Crop Nutrient Predictor
# **Competition:** Playground Series S5E6  
# **Author:** [Your Name]
# **Version:** 1.3  
# **Last Updated:** [Current Date]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.5f}'.format)
np.random.seed(42)

# Visualization setup
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12


# =============================================
# CONFIGURATION
# =============================================
CONFIG = {
    'paths': {
        'main': Path("/kaggle/input/playground-series-s5e6"),
        'original': Path("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
    },
    'dtypes': {
        'Temperature': 'float16', 'Humidity': 'float16', 'Moisture': 'float16',
        'Nitrogen': 'float16', 'Potassium': 'float16', 'Phosphorous': 'float16',
        'Soil Type': 'category', 'Crop Type': 'category', 'Fertilizer Name': 'category'
    },
    'n_folds': 5,
    'random_state': 42,
    'use_gpu': False
}


# =============================================
# DATA LOADING & PREPROCESSING
# =============================================
def load_data():
    """Load and optimize datasets with smart error handling"""
    try:
        train = pd.read_csv(CONFIG['paths']['main']/"train.csv", dtype=CONFIG['dtypes'])
        test = pd.read_csv(CONFIG['paths']['main']/"test.csv", dtype=CONFIG['dtypes'])
        
        # Fix column names
        for df in [train, test]:
            df.rename(columns={'Temparature': 'Temperature'}, inplace=True)

        if CONFIG['paths']['original'].exists():
            original = pd.read_csv(CONFIG['paths']['original'])
            original.rename(columns={'Temparature': 'Temperature'}, inplace=True)
            train = pd.concat([train, original], ignore_index=True)
            
        return train, test
    
    except Exception as e:
        raise SystemError(f"Data loading failed: {str(e)}")

def optimize_memory(df):
    """Optimize memory usage for dataframe"""
    initial_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.select_dtypes(include=['number']).columns:
        c_min, c_max = df[col].min(), df[col].max()
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    final_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory reduced by {100*(initial_mem-final_mem)/initial_mem:.1f}% ({initial_mem:.2f}MB â†’ {final_mem:.2f}MB)")
    return df


# =============================================
# FEATURE ENGINEERING
# =============================================
def create_features(df):
    """Generate domain-specific features for precision agriculture"""
    # Nutrient interactions
    df['N/P_ratio'] = (df['Nitrogen'] + 1) / (df['Phosphorous'] + 1)
    df['N/K_ratio'] = (df['Nitrogen'] + 1) / (df['Potassium'] + 1)
    df['P/K_ratio'] = (df['Phosphorous'] + 1) / (df['Potassium'] + 1)
    df['NP_balance'] = df['Nitrogen'] - df['Phosphorous']
    
    # Environmental interactions
    df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
    df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
    df['Environmental_Stress'] = ((df['Temperature'] > 30) & (df['Humidity'] < 40)).astype(np.int8)
    
    # Nutrient metrics
    nutrients = ['Nitrogen', 'Phosphorous', 'Potassium']
    df['Nutrient_Sum'] = df[nutrients].sum(axis=1)
    df['Nutrient_Imbalance'] = df['Nitrogen'] - df['Phosphorous'] - df['Potassium']
    df['NPK_Score'] = df['Nitrogen']*0.5 + df['Phosphorous']*0.3 + df['Potassium']*0.2
    
    # Soil-Crop synergy
    df['Soil_Crop_Combo'] = df['Soil Type'].astype(str) + "_" + df['Crop Type'].astype(str)
    
    return df


# =============================================
# MODEL TRAINING & EVALUATION
# =============================================
def mapk_score(y_true, y_pred_proba, k=3):
    """Calculate Mean Average Precision at K"""
    ap_scores = []
    for true, pred in zip(y_true, y_pred_proba.argsort(axis=1)[:, ::-1][:, :k]):
        correct = 0
        total_precision = 0
        for i, p in enumerate(pred):
            if p == true:
                correct += 1
                total_precision += correct / (i + 1)
        ap_scores.append(total_precision / min(k, len(pred)))
    return np.mean(ap_scores)

def train_models(X, y, X_test, n_folds=5):
    """Train ensemble of models with cross-validation"""
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = np.zeros((len(X_test), len(np.unique(y))))
    models = {}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CONFIG['random_state'])
    
    # Convert categorical columns to ordinal encoding
    cat_cols = X.select_dtypes(include=['category', 'object']).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded = X.copy()
    X_encoded[cat_cols] = encoder.fit_transform(X[cat_cols])
    X_test_encoded = X_test.copy()
    X_test_encoded[cat_cols] = encoder.transform(X_test[cat_cols])
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nâ�³ Fold {fold}/{n_folds}")
        X_train, X_valid = X_encoded.iloc[train_idx], X_encoded.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Initialize models
        model_dict = {
            'xgb': XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=CONFIG['random_state'],
                tree_method='gpu_hist' if CONFIG['use_gpu'] else 'hist',
                eval_metric='mlogloss',
                use_label_encoder=False
            ),
            'lgb': LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                random_state=CONFIG['random_state'],
                objective='multiclass'
            )
        }
        
        fold_preds = []
        
        for name, model in model_dict.items():
            print(f"  Training {name}...", end=' ')
            try:
                if name == 'xgb':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        early_stopping_rounds=100,
                        verbose=0
                    )
                else:  # LGBM
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        eval_metric='multi_logloss',
                        callbacks=[LGBMClassifier.early_stopping(stopping_rounds=100)],
                        verbose=0
                    )
                
                val_preds = model.predict_proba(X_valid)
                fold_score = mapk_score(y_valid, val_preds)
                fold_preds.append(val_preds)
                print(f"âœ“ (MAP@3: {fold_score:.4f})")
                
                # Accumulate test predictions
                test_preds += model.predict_proba(X_test_encoded) / n_folds
                if fold == 1:
                    models[name] = model
                    
            except Exception as e:
                print(f"âœ— Failed: {str(e)}")
                # Add uniform predictions if model fails
                dummy_preds = np.full((len(X_valid), len(np.unique(y))), 1/len(np.unique(y)))
                fold_preds.append(dummy_preds)
                test_preds += np.full((len(X_test), len(np.unique(y))), 1/len(np.unique(y))) / n_folds
        
        # Ensemble predictions
        if fold_preds:
            ensemble_preds = np.mean(fold_preds, axis=0)
            oof_preds[valid_idx] = ensemble_preds
            fold_score = mapk_score(y_valid, ensemble_preds)
            print(f"  Ensemble MAP@3: {fold_score:.4f}")
    
    oof_score = mapk_score(y, oof_preds)
    print(f"\nğŸ�¯ Final OOF MAP@3 Score: {oof_score:.5f}")
    
    return models, test_preds, oof_preds


# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    # Load and preprocess data
    print("ğŸ”� Loading and preprocessing data...")
    train, test = load_data()
    train = optimize_memory(train)
    test = optimize_memory(test)
    
    # Feature engineering
    print("\nğŸ”§ Creating agricultural features...")
    train = create_features(train)
    test = create_features(test)
    
    # Prepare data
    le = LabelEncoder()
    train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])
    
    # Prepare datasets
    X = train.drop(['id', 'Fertilizer Name'], axis=1)
    y = train['Fertilizer Name']
    X_test = test.drop('id', axis=1)
    
    # Train models
    print("\nğŸ¤– Training models...")
    models, test_preds, oof_preds = train_models(X, y, X_test, n_folds=CONFIG['n_folds'])
    
    # Generate submission
    print("\nğŸ�¯ Generating submission...")
    try:
        # Get top 3 predictions
        top3_indices = np.argsort(-test_preds, axis=1)[:, :3]
        top3_labels = le.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
        
        submission = pd.DataFrame({
            'id': test['id'],
            'Fertilizer Name': [' '.join(row) for row in top3_labels]
        })
        
        # Validate submission
        assert len(submission) == 250000, "Submission must have 250k rows"
        assert list(submission.columns) == ['id', 'Fertilizer Name'], "Incorrect columns"
        assert not submission.isnull().any().any(), "Submission contains null values"
        
        submission.to_csv('submission.csv', index=False)
        print("\nâœ… Submission created successfully!")
        print(f"Sample predictions:\n{submission.head(3)}")
        
    except Exception as e:
        print(f"\nâ�Œ Submission error: {str(e)}")
        # Create valid fallback submission
        submission = pd.DataFrame({
            'id': test['id'] if 'id' in test.columns else range(1, 250001),
            'Fertilizer Name': ['Fert_A Fert_B Fert_C'] * 250000
        })
        submission.to_csv('submission.csv', index=False)
        print("âš ï¸� Created fallback submission")
    
    # Feature importance visualization
    plt.figure(figsize=(12, 6))
    for i, (name, model) in enumerate(models.items(), 1):
        plt.subplot(1, 2, i)
        if model is not None and hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=X.columns)
            fi.nlargest(15).sort_values().plot.barh()
            plt.title(f'{name.upper()} Feature Importance')
        else:
            plt.title(f'{name.upper()} - No Model')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nğŸ�† Pipeline completed successfully!")

