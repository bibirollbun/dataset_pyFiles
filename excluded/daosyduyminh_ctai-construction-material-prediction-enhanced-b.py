# -*- coding: utf-8 -*-
"""
CTAI Construction Material Prediction - Enhanced Baseline
Building on the 0.84722 baseline with targeted improvements
Focus: Keep the winning strategy, enhance the weak points
"""

import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

def clean_quantity_column(series):
    """Enhanced cleaning function based on baseline approach."""
    def clean_value(o):
        if pd.isna(o) or o == '':
            return np.nan
        if not isinstance(o, str):
            try:
                return float(o)
            except:
                return np.nan
        
        # Handle newlines (from baseline)
        if '\n' in o:
            return 3.0
            
        # Clean the string
        o = str(o).replace(',', '').strip()
        
        # Remove trailing dashes (from baseline)
        while len(o) > 0 and o[-1] == '-':
            o = o[:-1]
            
        # Handle ' EA' suffix (from baseline)
        if len(o) >= 5 and o.endswith(' EA'):
            o = o[:-3]
            
        # Additional cleaning patterns
        o = o.replace('$', '').replace('%', '')
        
        try:
            return float(o)
        except:
            return np.nan
    
    return series.apply(clean_value)

def load_and_clean_data():
    """Load and clean data using baseline approach + enhancements."""
    
    # Load with proper date parsing (from baseline)
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
    
    # Clean quantity columns (baseline approach)
    for df in [train, test]:
        df['ExtendedQuantity'] = clean_quantity_column(df['ExtendedQuantity'])
        
    train['QtyShipped'] = clean_quantity_column(train['QtyShipped'])
    
    # Remove rows without target (from baseline)
    train_clean = train.dropna(subset=['QtyShipped']).copy()
    
    print(f"Data loaded: Train {train_clean.shape}, Test {test.shape}")
    print(f"Removed {len(train) - len(train_clean)} rows without QtyShipped target")
    
    return train_clean, test

def create_enhanced_features(train_df, test_df):
    """Create enhanced features while keeping baseline simplicity."""
    
    # Core features that work (from baseline)
    base_features = {
        'classification': ['ItemDescription'],
        'regression': ['ExtendedQuantity']
    }
    
    # Enhancement 1: Add complementary features
    enhanced_features = {
        'classification': ['ItemDescription', 'UOM', 'PROJECT_TYPE', 'CORE_MARKET'],
        'regression': ['ExtendedQuantity', 'UnitPrice', 'ExtendedPrice']
    }
    
    # Enhancement 2: Create interaction features
    for df in [train_df, test_df]:
        # Price-based features (if available)
        if 'UnitPrice' in df.columns and 'ExtendedQuantity' in df.columns:
            df['calculated_total'] = df['UnitPrice'] * df['ExtendedQuantity']
            enhanced_features['regression'].append('calculated_total')
        
        # Text-based features from ItemDescription
        if 'ItemDescription' in df.columns:
            desc = df['ItemDescription'].fillna('unknown').str.lower()
            
            # Material type indicators
            df['is_electrical'] = desc.str.contains('electric|wire|cable|power', na=False).astype(int)
            df['is_network'] = desc.str.contains('network|fiber|copper|cat6', na=False).astype(int)
            df['is_structural'] = desc.str.contains('steel|concrete|beam|frame', na=False).astype(int)
            
            # Size indicators
            df['has_dimensions'] = desc.str.contains(r'\d+ft|\d+mm|\d+inch', na=False).astype(int)
            df['desc_length'] = df['ItemDescription'].fillna('').str.len()
            
            enhanced_features['classification'].extend(['is_electrical', 'is_network', 'is_structural', 'has_dimensions'])
    
    return base_features, enhanced_features

def create_models():
    """Create model configurations from simple to enhanced."""
    
    models = {
        'classification': {
            'baseline': make_pipeline(
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10000), 
                DecisionTreeClassifier(random_state=42)
            ),
            'enhanced_tree': make_pipeline(
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10000),
                DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42)
            ),
            'random_forest': make_pipeline(
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10000),
                RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
            ),
            'lightgbm': make_pipeline(
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10000),
                lgb.LGBMClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, verbose=-1)
            )
        },
        'regression': {
            'baseline': make_pipeline(
                SimpleImputer(), 
                DecisionTreeRegressor(random_state=42)
            ),
            'enhanced_tree': make_pipeline(
                SimpleImputer(),
                DecisionTreeRegressor(max_depth=20, min_samples_split=5, random_state=42)
            ),
            'random_forest': make_pipeline(
                SimpleImputer(),
                RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            ),
            'lightgbm': make_pipeline(
                SimpleImputer(),
                lgb.LGBMRegressor(n_estimators=200, max_depth=8, random_state=42, verbose=-1)
            )
        }
    }
    
    return models

def evaluate_models(train_df, base_features, enhanced_features, models):
    """Evaluate different model configurations."""
    
    results = {}
    
    # Prepare targets
    y_class = train_df['MasterItemNo']
    y_reg = train_df['QtyShipped']
    
    print("="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Classification evaluation
    print("\nClassification Models:")
    print("-" * 40)
    
    feature_sets = {'base': base_features['classification'], 'enhanced': enhanced_features['classification']}
    
    for feat_name, features in feature_sets.items():
        print(f"\n{feat_name.upper()} FEATURES: {features}")
        
        for model_name, model in models['classification'].items():
            try:
                # Cross-validation predictions
                oof_pred = cross_val_predict(model, train_df[features], y_class, cv=5)
                
                acc = accuracy_score(y_class, oof_pred)
                f1 = f1_score(y_class, oof_pred, average='weighted', zero_division=0)
                
                results[f'clf_{feat_name}_{model_name}'] = {
                    'model': model, 'features': features, 
                    'accuracy': acc, 'f1': f1, 'clf_score': (acc + f1) / 2
                }
                
                print(f"  {model_name:<15} | Acc: {acc:.4f} | F1: {f1:.4f} | Avg: {(acc+f1)/2:.4f}")
                
            except Exception as e:
                print(f"  {model_name:<15} | ERROR: {e}")
    
    # Regression evaluation
    print("\nRegression Models:")
    print("-" * 40)
    
    feature_sets = {'base': base_features['regression'], 'enhanced': enhanced_features['regression']}
    
    for feat_name, features in feature_sets.items():
        print(f"\n{feat_name.upper()} FEATURES: {features}")
        
        for model_name, model in models['regression'].items():
            try:
                # Cross-validation predictions
                oof_pred = cross_val_predict(model, train_df[features], y_reg, cv=5)
                
                mae = mean_absolute_error(y_reg, oof_pred)
                norm_mae = mae / (y_reg.max() - y_reg.min()) if y_reg.max() != y_reg.min() else 0
                reg_score = max(0, 1 - norm_mae)
                
                results[f'reg_{feat_name}_{model_name}'] = {
                    'model': model, 'features': features,
                    'mae': mae, 'reg_score': reg_score
                }
                
                print(f"  {model_name:<15} | MAE: {mae:.1f} | RegScore: {reg_score:.5f}")
                
            except Exception as e:
                print(f"  {model_name:<15} | ERROR: {e}")
    
    return results

def create_submissions(results, train_df, test_df):
    """Create submissions for top model combinations."""
    
    # Find best models
    clf_results = {k: v for k, v in results.items() if k.startswith('clf_')}
    reg_results = {k: v for k, v in results.items() if k.startswith('reg_')}
    
    best_clf = max(clf_results.items(), key=lambda x: x[1]['clf_score'])
    best_reg = max(reg_results.items(), key=lambda x: x[1]['reg_score'])
    
    print("\n" + "="*60)
    print("CREATING SUBMISSIONS")
    print("="*60)
    
    print(f"Best Classification: {best_clf[0]} (Score: {best_clf[1]['clf_score']:.4f})")
    print(f"Best Regression: {best_reg[0]} (Score: {best_reg[1]['reg_score']:.5f})")
    
    # Prepare best models
    clf_model = best_clf[1]['model']
    clf_features = best_clf[1]['features']
    reg_model = best_reg[1]['model'] 
    reg_features = best_reg[1]['features']
    
    # Train on full data
    clf_model.fit(train_df[clf_features], train_df['MasterItemNo'])
    reg_model.fit(train_df[reg_features], train_df['QtyShipped'])
    
    # Create predictions
    clf_pred = clf_model.predict(test_df[clf_features])
    reg_pred = reg_model.predict(test_df[reg_features])
    
    # Ensure non-negative quantities
    reg_pred = np.maximum(reg_pred, 0)
    
    # Calculate composite score estimate
    clf_acc = best_clf[1]['accuracy']
    clf_f1 = best_clf[1]['f1']
    reg_score = best_reg[1]['reg_score']
    composite_estimate = 0.25 * clf_acc + 0.25 * clf_f1 + 0.5 * reg_score
    
    print(f"Estimated composite score: {composite_estimate:.5f}")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'MasterItemNo': clf_pred,
        'QtyShipped': reg_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"\nSubmission created:")
    print(f"  Shape: {submission.shape}")
    print(f"  Unique MasterItemNo: {submission['MasterItemNo'].nunique()}")
    print(f"  QtyShipped range: {reg_pred.min():.1f} to {reg_pred.max():.1f}")
    
    print("\nSubmission preview:")
    print(submission.head(10))
    
    return submission

def main():
    """Main pipeline based on enhanced baseline approach."""
    
    print("="*80)
    print("CTAI ENHANCED BASELINE - Building on 0.84722 Success")
    print("="*80)
    
    try:
        # Step 1: Load and clean data
        print("1. Loading and cleaning data...")
        train_df, test_df = load_and_clean_data()
        
        # Step 2: Create features
        print("\n2. Creating enhanced features...")
        base_features, enhanced_features = create_enhanced_features(train_df, test_df)
        
        # Step 3: Create models
        print("\n3. Creating model configurations...")
        models = create_models()
        
        # Step 4: Evaluate models
        print("\n4. Evaluating models...")
        results = evaluate_models(train_df, base_features, enhanced_features, models)
        
        # Step 5: Create submission
        print("\n5. Creating submission...")
        submission = create_submissions(results, train_df, test_df)
        
        print("\n" + "="*80)
        print("ENHANCED BASELINE COMPLETED!")
        print("Target: Beat 0.84722 through targeted improvements")
        print("="*80)
        
        return submission
        
    except Exception as e:
        print(f"Error in enhanced baseline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

