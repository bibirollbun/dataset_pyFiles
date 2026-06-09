import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import re
import gc
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Limit CPU usage
os.environ["OMP_NUM_THREADS"] = "1"

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TARGET = 'Listening_Time_minutes'
N_FOLDS = 5

# Feature groups (based on the notebook analysis)
CATS = ['Podcast_Name', 'Episode_Num', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
NUMS = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

def load_data():
    """Load training and test data with Kaggle paths"""
    train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
    
    # Also load the original dataset if available
    try:
        original = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
        # Clean and combine with training data
        original_clean = original.dropna(subset=[TARGET]).drop_duplicates()
        train = pd.concat([train, original_clean], axis=0, ignore_index=True)
        print(f"Combined training data: {train.shape[0]} rows")
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        print("Using only competition data")
    
    return train, test

def engineer_features(train_df, test_df):
    """Engineer features based on the successful strategies in the notebook"""
    # Make copies to avoid modifying originals
    train = train_df.copy()
    test = test_df.copy()
    
    # Extract episode number
    train['Episode_Num'] = train['Episode_Title'].str.extract(r'Episode (\d+)', expand=False).astype(float)
    test['Episode_Num'] = test['Episode_Title'].str.extract(r'Episode (\d+)', expand=False).astype(float)
    
    # Fill missing episode numbers with median
    ep_median = train['Episode_Num'].median()
    train['Episode_Num'] = train['Episode_Num'].fillna(ep_median)
    test['Episode_Num'] = test['Episode_Num'].fillna(ep_median)
    
    # Weekend feature
    train['is_weekend'] = train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    test['is_weekend'] = test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Episode length rounding features (captures threshold effects)
    for k in range(3):
        col_name = f'ELm_r{k}'
        train[col_name] = train['Episode_Length_minutes'].round(k)
        test[col_name] = test['Episode_Length_minutes'].round(k)
    
    # Create numerical interactions instead of categorical string combinations
    # This avoids the categorical feature issues with XGBoost
    train['ep_len_host_pop'] = train['Episode_Length_minutes'] * train['Host_Popularity_percentage']
    test['ep_len_host_pop'] = test['Episode_Length_minutes'] * test['Host_Popularity_percentage']
    
    train['ep_len_guest_pop'] = train['Episode_Length_minutes'] * train['Guest_Popularity_percentage']
    test['ep_len_guest_pop'] = test['Episode_Length_minutes'] * test['Guest_Popularity_percentage']
    
    train['ep_len_ads'] = train['Episode_Length_minutes'] * train['Number_of_Ads']
    test['ep_len_ads'] = test['Episode_Length_minutes'] * test['Number_of_Ads']
    
    train['ep_num_host_pop'] = train['Episode_Num'] * train['Host_Popularity_percentage']
    test['ep_num_host_pop'] = test['Episode_Num'] * test['Host_Popularity_percentage']
    
    train['ep_num_guest_pop'] = train['Episode_Num'] * train['Guest_Popularity_percentage']
    test['ep_num_guest_pop'] = test['Episode_Num'] * test['Guest_Popularity_percentage']
    
    train['host_guest_pop'] = train['Host_Popularity_percentage'] * train['Guest_Popularity_percentage']
    test['host_guest_pop'] = test['Host_Popularity_percentage'] * test['Guest_Popularity_percentage']
    
    # Create 3-way interactions as numeric features
    train['ep_len_ep_num_host'] = train['Episode_Length_minutes'] * train['Episode_Num'] * train['Host_Popularity_percentage']
    test['ep_len_ep_num_host'] = test['Episode_Length_minutes'] * test['Episode_Num'] * test['Host_Popularity_percentage']
    
    train['ep_len_host_guest'] = train['Episode_Length_minutes'] * train['Host_Popularity_percentage'] * train['Guest_Popularity_percentage']
    test['ep_len_host_guest'] = test['Episode_Length_minutes'] * test['Host_Popularity_percentage'] * test['Guest_Popularity_percentage']
    
    train['ep_len_ep_num_guest'] = train['Episode_Length_minutes'] * train['Episode_Num'] * train['Guest_Popularity_percentage']
    test['ep_len_ep_num_guest'] = test['Episode_Length_minutes'] * test['Episode_Num'] * test['Guest_Popularity_percentage']
    
    # Convert object columns to category type explicitly
    for col in CATS:
        if col in train.columns and train[col].dtype == 'object':
            train[col] = train[col].astype('category')
        if col in test.columns and test[col].dtype == 'object':
            test[col] = test[col].astype('category')
    
    # Fill missing numeric values
    for col in train.select_dtypes(include=['number']).columns:
        if col != TARGET:  # Don't fill target
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    # Drop unnecessary columns
    train = train.drop(['Episode_Title'], axis=1)
    test = test.drop(['Episode_Title'], axis=1)
    
    return train, test

def target_encode(df_train, df_val, col, target, stats='mean', prefix='TE'):
    """Perform target encoding for a single column"""
    df_val = df_val.copy()
    agg = df_train.groupby(col)[target].agg(stats)    
    
    if isinstance(stats, (list, tuple)):
        for s in stats:
            colname = f"{prefix}_{col}_{s}"
            df_val[colname] = df_val[col].map(agg[s]).astype(float)
            df_val[colname].fillna(agg[s].mean(), inplace=True)
    else:
        suffix = stats if isinstance(stats, str) else stats.__name__
        colname = f"{prefix}_{col}_{suffix}"
        df_val[colname] = df_val[col].map(agg).astype(float)
        df_val[colname].fillna(agg.mean(), inplace=True)
    
    return df_val

def add_target_encodings(train_df, test_df, encode_cols, target, n_folds=5, encode_stats=['mean']):
    """Add target encoding features using cross-validation to prevent leakage"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # For training data: use out-of-fold target encoding
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        tr_data = train_df.iloc[tr_idx]
        val_data = train_df.iloc[val_idx]
        
        for col in encode_cols:
            for stat in encode_stats:
                # Add target encoding using only training data from this fold
                encoded_val = target_encode(tr_data, val_data, col, target, stats=stat, prefix='TE')
                te_col = f"TE_{col}_{stat}"
                train_df.loc[val_idx, te_col] = encoded_val[te_col].values
    
    # For test data: use target encoding based on all training data
    for col in encode_cols:
        for stat in encode_stats:
            test_df = target_encode(train_df, test_df, col, target, stats=stat, prefix='TE')
    
    return train_df, test_df

def train_model(X_train, y_train, X_test):
    """Train XGBoost model and make predictions"""
    # Initialize KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Initialize arrays for predictions
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    # Print data types for debugging
    print("\nData types check:")
    print(X_train.dtypes.value_counts())
    
    # Handle categorical columns correctly
    cat_features = X_train.select_dtypes(include=['category']).columns.tolist()
    print(f"Number of categorical features: {len(cat_features)}")
    
    # Train model for each fold
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold+1}/{N_FOLDS}")
        
        # Split data
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        # Initialize model with parameters from notebook
        # Make sure enable_categorical is True for XGBoost to handle categorical features
        model = xgb.XGBRegressor(
            tree_method='hist',
            max_depth=14,
            colsample_bytree=0.5,
            subsample=0.9,
            n_estimators=1000,  # Reduced from original 50,000 for performance
            learning_rate=0.02,
            min_child_weight=10,
            early_stopping_rounds=100,
            random_state=RANDOM_STATE,
            enable_categorical=True  # Critical parameter for handling categorical features
        )
        
        # Train model
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=100  # Reduced verbosity
        )
        
        # Make predictions
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / N_FOLDS
        
        # Print fold score
        fold_score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} RMSE: {fold_score:.5f}")
        
        # Clean up memory
        if fold < N_FOLDS - 1:  # Keep the last model for feature importance
            del model
            gc.collect()
    
    # Calculate overall score
    final_score = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"Overall RMSE: {final_score:.5f}")
    
    # Get feature importance from the last model
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return oof_preds, test_preds, final_score, feature_importance, model

def main():
    start_time = time.time()
    print("Loading data...")
    train, test = load_data()
    
    print("\nEngineering features...")
    train, test = engineer_features(train, test)
    
    print("\nAdding target encodings...")
    # Only use core categorical columns for target encoding
    # This avoids issues with newly created categorical features
    train, test = add_target_encodings(train, test, CATS, TARGET, n_folds=N_FOLDS)
    
    # Save IDs for submission
    test_ids = test['id'].copy() if 'id' in test.columns else test.index.copy()
    
    # Prepare data for modeling
    X_train = train.drop([TARGET, 'id'], axis=1, errors='ignore')
    y_train = train[TARGET]
    X_test = test.drop(['id'], axis=1, errors='ignore')
    
    # Print column summary for debugging
    print(f"\nTrain columns: {X_train.shape[1]}")
    print(f"Train rows: {X_train.shape[0]}")
    print(f"Test columns: {X_test.shape[1]}")
    print(f"Test rows: {X_test.shape[0]}")
    
    print("\nTraining model...")
    oof_preds, test_preds, final_score, feature_importance, model = train_model(X_train, y_train, X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_ids,
        TARGET: test_preds
    })
    
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print(f"\nSubmission saved to: /kaggle/working/submission.csv")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(30))
    plt.title('Top 30 Feature Importance')
    plt.tight_layout()
    plt.savefig('/kaggle/working/feature_importance.png')
    
    # Print execution time
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return {
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'final_score': final_score,
        'feature_importance': feature_importance,
        'model': model
    }

if __name__ == "__main__":
    try:
        print("Starting fixed podcast listening time prediction model...")
        results = main()
        print(f"\nFinal RMSE: {results['final_score']:.5f}")
        
        # Display top features
        print("\nTop 20 important features:")
        for i, (feature, importance) in enumerate(zip(results['feature_importance']['feature'].values[:20], 
                                                     results['feature_importance']['importance'].values[:20])):
            print(f"{i+1}. {feature}: {importance:.6f}")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

