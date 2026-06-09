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


# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')



# ===================== DATA LOADING & PREPROCESSING =====================
def load_and_preprocess_data():
    """Load and preprocess the data."""
    train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
    submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
    
    # Basic preprocessing
    train = train.drop_duplicates().reset_index(drop=True)
    train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
    test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
    
    return train, test, submission


# ===================== OPTIMIZED FEATURE ENGINEERING =====================
def create_features(df):
    """Create only the most important features."""
    # Basic derived features
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    
    # Only the most important interactions
    df['Age_x_Weight'] = df['Age'] * df['Weight']
    df['Heart_Rate_x_Duration'] = df['Heart_Rate'] * df['Duration']
    
    return df



# ===================== OPTIMIZED MODEL TRAINING =====================
def train_and_predict(X_train, y_train, X_test, model_type='catboost', n_folds=5):
    """Optimized training function with progress tracking."""
    test_preds = np.zeros(len(X_test))
    
    # Duration-based stratified folds
    bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    duration_bins = bins.fit_transform(X_train[['Duration']]).flatten()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, duration_bins)):
        print(f"Processing fold {fold+1}/{n_folds}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if model_type == 'catboost':
            model = CatBoostRegressor(
                iterations=1000,  # Reduced from 2000
                learning_rate=0.05,  # Increased from 0.02
                depth=8,  # Reduced from 10
                loss_function='RMSE',
                eval_metric='RMSE',
                early_stopping_rounds=50,  # Reduced from 100
                random_seed=42,
                verbose=100  # Added progress tracking
            )
            cat_features = [X_train.columns.get_loc("Sex")] if 'Sex' in X_train.columns else None
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features)
            
        elif model_type == 'xgboost':
            model = XGBRegressor(
                n_estimators=1000,  # Reduced from 2000
                learning_rate=0.05,  # Increased from 0.02
                max_depth=8,  # Reduced from 10
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.01,
                eval_metric='rmse',
                early_stopping_rounds=50,  # Reduced from 100
                random_state=42,
                tree_method="hist",
                enable_categorical=True  # For better handling of 'Sex'
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
        
        test_preds += model.predict(X_test) / n_folds
    
    return test_preds



# ===================== SIMPLIFIED MODELING APPROACH =====================
def get_predictions(train, test):
    """Single optimized modeling approach."""
    # Feature engineering
    df = create_features(train.copy())
    test_df = create_features(test.copy())
    
    # Prepare data
    X = df.drop(columns=['id', 'Calories'])
    y = np.log1p(df['Calories'])
    X_test = test_df.drop(columns=['id'])
    
    # Get predictions from both models
    print("\nTraining CatBoost...")
    cat_preds = train_and_predict(X, y, X_test, model_type='catboost', n_folds=5)
    
    print("\nTraining XGBoost...")
    xgb_preds = train_and_predict(X, y, X_test, model_type='xgboost', n_folds=5)
    
    # Ensemble predictions
    final_preds = 0.5 * cat_preds + 0.5 * xgb_preds
    return np.clip(np.expm1(final_preds), 1, 314)



# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train, test, submission = load_and_preprocess_data()
    
    # Get predictions (single optimized approach)
    print("\nStarting model training...")
    submission['Calories'] = get_predictions(train, test)
    
    # Save results
    submission.to_csv("submission.csv", index=False)
    print("\n✅ Submission file saved successfully!")

