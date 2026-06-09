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
Tugas 1 Statistical Machine Learning A 2025 - Restaurant Cost Prediction
=========================================================================
Fault-tolerant solution with extensive error handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import re
import ast

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# Model Selection
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ================================================================================
# CONFIGURATION
# ================================================================================

INPUT_DIR = Path("/kaggle/input/tugas-1-statistical-machine-learning-a-2025-regression/")
OUTPUT_FILE = "submission.csv"
RANDOM_STATE = 42
N_FOLDS = 5
TARGET = 'approx_cost(for two people)'

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def safe_parse_rating(rate_str):
    """Safely parse rating string to numeric"""
    try:
        if pd.isna(rate_str) or rate_str in ['NEW', '-', 'nan', None]:
            return 3.5  # Default rating
        if '/' in str(rate_str):
            return float(str(rate_str).split('/')[0])
        return float(rate_str)
    except:
        return 3.5  # Default rating

def safe_clean_cost(cost_str):
    """Safely clean cost string to numeric"""
    try:
        if pd.isna(cost_str):
            return 500.0  # Default cost
        cost_clean = str(cost_str).replace(',', '')
        return float(cost_clean)
    except:
        return 500.0  # Default cost

def safe_str_len(x):
    """Safely get string length"""
    try:
        return len(str(x)) if pd.notna(x) else 0
    except:
        return 0

def safe_str_split_len(x):
    """Safely get split string length"""
    try:
        return len(str(x).split(',')) if pd.notna(x) else 1
    except:
        return 1

def safe_contains(text, pattern):
    """Safely check if text contains pattern"""
    try:
        if pd.isna(text):
            return 0
        return int(pattern.lower() in str(text).lower())
    except:
        return 0

# ================================================================================
# FEATURE ENGINEERING
# ================================================================================

def engineer_features(df, is_train=True):
    """Comprehensive feature engineering with fault tolerance"""
    print("ğŸ”§ Engineering features...")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # 1. HANDLE RATE COLUMN
    if 'rate' in df.columns:
        df['rate_numeric'] = df['rate'].apply(safe_parse_rating)
    else:
        df['rate_numeric'] = 3.5  # Default if column doesn't exist
    
    # 2. HANDLE TARGET COLUMN (for training)
    if is_train and TARGET in df.columns:
        df[TARGET] = df[TARGET].apply(safe_clean_cost)
        # Remove rows with invalid target
        df = df[df[TARGET] > 0]
    
    # 3. HANDLE BINARY COLUMNS
    if 'online_order' in df.columns:
        df['online_order'] = (df['online_order'] == 'Yes').astype(int)
    else:
        df['online_order'] = 0
    
    if 'book_table' in df.columns:
        df['book_table'] = (df['book_table'] == 'Yes').astype(int)
    else:
        df['book_table'] = 0
    
    # 4. HANDLE VOTES
    if 'votes' in df.columns:
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)
    else:
        df['votes'] = 0
    
    # 5. BASIC FEATURES
    if 'name' in df.columns:
        df['name_length'] = df['name'].apply(safe_str_len)
        df['name_word_count'] = df['name'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 1)
    else:
        df['name_length'] = 10
        df['name_word_count'] = 2
    
    if 'address' in df.columns:
        df['address_length'] = df['address'].apply(safe_str_len)
    else:
        df['address_length'] = 50
    
    # 6. RATING FEATURES
    df['has_rating'] = (df['rate_numeric'] > 0).astype(int)
    df['rating_x_votes'] = df['rate_numeric'] * df['votes']
    df['log_votes'] = np.log1p(df['votes'])
    
    # 7. CUISINE FEATURES
    if 'cuisines' in df.columns:
        df['num_cuisines'] = df['cuisines'].apply(safe_str_split_len)
        
        # Popular cuisines
        popular_cuisines = ['North Indian', 'Chinese', 'Continental', 'South Indian', 'Fast Food']
        for cuisine in popular_cuisines:
            col_name = f'has_{cuisine.lower().replace(" ", "_")}'
            df[col_name] = df['cuisines'].apply(lambda x: safe_contains(x, cuisine))
    else:
        df['num_cuisines'] = 2
        for cuisine in ['North Indian', 'Chinese', 'Continental', 'South Indian', 'Fast Food']:
            df[f'has_{cuisine.lower().replace(" ", "_")}'] = 0
    
    # 8. RESTAURANT TYPE FEATURES
    if 'rest_type' in df.columns:
        df['num_rest_types'] = df['rest_type'].apply(safe_str_split_len)
        
        rest_types = ['Casual Dining', 'Quick Bites', 'Cafe', 'Delivery', 'Fine Dining']
        for rest_type in rest_types:
            col_name = f'is_{rest_type.lower().replace(" ", "_")}'
            df[col_name] = df['rest_type'].apply(lambda x: safe_contains(x, rest_type))
    else:
        df['num_rest_types'] = 1
        for rest_type in ['Casual Dining', 'Quick Bites', 'Cafe', 'Delivery', 'Fine Dining']:
            df[f'is_{rest_type.lower().replace(" ", "_")}'] = 0
    
    # 9. LOCATION FEATURES
    if 'location' in df.columns:
        location_counts = df['location'].value_counts().to_dict()
        df['location_popularity'] = df['location'].map(location_counts).fillna(1)
        
        premium_areas = ['Koramangala', 'Indiranagar', 'Whitefield', 'MG Road']
        df['in_premium_area'] = df['location'].apply(
            lambda x: int(any(area in str(x) for area in premium_areas)) if pd.notna(x) else 0
        )
    else:
        df['location_popularity'] = 100
        df['in_premium_area'] = 0
    
    # 10. LISTED IN TYPE FEATURES
    if 'listed_in(type)' in df.columns:
        listing_types = ['Buffet', 'Cafes', 'Delivery', 'Dine-out']
        for listing in listing_types:
            col_name = f'listed_in_{listing.lower().replace(" ", "_").replace("-", "_")}'
            df[col_name] = (df['listed_in(type)'] == listing).astype(int)
    else:
        for listing in ['Buffet', 'Cafes', 'Delivery', 'Dine-out']:
            df[f'listed_in_{listing.lower().replace(" ", "_").replace("-", "_")}'] = 0
    
    # 11. INTERACTION FEATURES
    df['rating_popularity'] = df['rate_numeric'] * np.log1p(df['location_popularity'])
    df['full_service'] = df['online_order'] * df['book_table']
    df['is_premium'] = ((df['rate_numeric'] > 4.0) & 
                        (df['in_premium_area'] == 1) & 
                        (df['book_table'] == 1)).astype(int)
    df['cuisine_location_score'] = df['num_cuisines'] * np.log1p(df['location_popularity'])
    
    # 12. ENCODE HIGH CARDINALITY CATEGORICALS
    for col in ['location', 'rest_type', 'listed_in(city)']:
        if col in df.columns:
            # Simple frequency encoding
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_freq_encoded'] = df[col].map(freq_map).fillna(1)
    
    return df

def get_feature_columns(df):
    """Get safe list of feature columns"""
    # Columns to exclude
    exclude = [
        TARGET, 'url', 'address', 'name', 'phone', 'dish_liked',
        'cuisines', 'reviews_list', 'menu_item', 'location',
        'rest_type', 'listed_in(city)', 'rate', 'listed_in(type)'
    ]
    
    # Get numeric columns only
    feature_cols = []
    for col in df.columns:
        if col not in exclude:
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
                feature_cols.append(col)
            except:
                pass
    
    return feature_cols

# ================================================================================
# MODEL TRAINING
# ================================================================================

def train_models_robust(X_train, y_train, X_test):
    """Train models with robust error handling"""
    print("ğŸ�¯ Training models...")
    
    # Initialize models
    models = {
        'ridge': Ridge(alpha=10, random_state=RANDOM_STATE),
        'rf': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=20,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    }
    
    # Try to add XGBoost and CatBoost if available
    try:
        models['xgb'] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    except:
        print("  XGBoost not available, skipping...")
    
    try:
        models['cat'] = CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.1,
            random_seed=RANDOM_STATE, verbose=False
        )
    except:
        print("  CatBoost not available, skipping...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train each model
    predictions = {}
    weights = {}
    
    for name, model in models.items():
        try:
            print(f"  Training {name}...")
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            predictions[name] = pred
            
            # Simple validation
            kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            for train_idx, val_idx in kf.split(X_train_scaled):
                X_t, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                m = model.__class__(**model.get_params())
                m.fit(X_t, y_t)
                val_pred = m.predict(X_v)
                score = np.sqrt(mean_squared_error(y_v, val_pred))
                scores.append(score)
            
            avg_score = np.mean(scores)
            print(f"    {name} CV RMSE: {avg_score:.2f}")
            
            # Weight inversely proportional to error
            weights[name] = 1.0 / (avg_score + 1)
            
        except Exception as e:
            print(f"    Error training {name}: {e}")
            continue
    
    # Ensemble predictions
    if len(predictions) == 0:
        print("  WARNING: No models trained successfully, using default prediction")
        return np.full(len(X_test), y_train.mean())
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Weighted average
    final_pred = np.zeros(len(X_test))
    for name, pred in predictions.items():
        final_pred += pred * weights[name]
    
    return final_pred

# ================================================================================
# MAIN PIPELINE
# ================================================================================

def main():
    """Main execution pipeline with comprehensive error handling"""
    
    print("=" * 80)
    print("RESTAURANT COST PREDICTION - FAULT TOLERANT VERSION")
    print("=" * 80)
    
    try:
        # Load data
        print("\nğŸ“‚ Loading data...")
        train = pd.read_csv(INPUT_DIR / "train.csv")
        test = pd.read_csv(INPUT_DIR / "test.csv")
        print(f"  Train shape: {train.shape}")
        print(f"  Test shape: {test.shape}")
        
        # Engineer features
        train = engineer_features(train, is_train=True)
        test = engineer_features(test, is_train=False)
        
        # Get feature columns
        feature_cols = get_feature_columns(train)
        
        # Ensure test has same columns
        for col in feature_cols:
            if col not in test.columns:
                test[col] = 0
        
        print(f"\nğŸ“Š Using {len(feature_cols)} features")
        
        # Prepare data
        X_train = train[feature_cols]
        y_train = train[TARGET]
        X_test = test[feature_cols]
        
        # Remove extreme outliers
        print("\nğŸ”� Handling outliers...")
        q1, q99 = y_train.quantile(0.05), y_train.quantile(0.95)
        mask = (y_train >= q1) & (y_train <= q99)
        X_train = X_train[mask]
        y_train = y_train[mask]
        print(f"  Training samples: {len(X_train)}")
        
        # Train models
        predictions = train_models_robust(X_train, y_train, X_test)
        
        # Create submission
        submission = pd.DataFrame({
            'index': test.index,
            TARGET: predictions
        })
        
        # Ensure reasonable predictions
        submission[TARGET] = submission[TARGET].clip(lower=50, upper=5000)
        
        # Save
        submission.to_csv(OUTPUT_FILE, index=False)
        print(f"\nâœ… Submission saved to {OUTPUT_FILE}")
        print(f"\nğŸ“ˆ Stats: Mean={predictions.mean():.0f}, Std={predictions.std():.0f}")
        
    except Exception as e:
        print(f"\nâ�Œ Fatal error: {e}")
        print("Creating default submission...")
        
        # Create default submission
        test = pd.read_csv(INPUT_DIR / "test.csv")
        submission = pd.DataFrame({
            'index': test.index,
            TARGET: 500  # Default prediction
        })
        submission.to_csv(OUTPUT_FILE, index=False)
        print("Default submission created")

if __name__ == "__main__":
    main()

