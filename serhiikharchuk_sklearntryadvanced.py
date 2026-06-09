
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Target variable
TARGET = 'Listening_Time_minutes'

def load_data():
    """Load data without any transformations"""
    # Load the datasets
    train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
    
    try:
        # Load and merge original dataset if available
        original = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
        # Clean and combine with training data
        original_clean = original.dropna(subset=[TARGET]).drop_duplicates()
        train = pd.concat([train, original_clean], axis=0, ignore_index=True)
        print(f"Combined training data: {train.shape[0]} rows")
    except Exception as e:
        print(f"Could not load original dataset: {e}")
    
    return train, test

def safe_feature_engineering(train_df, test_df):
    """Create only the safest features with no conversion issues"""
    print("Creating basic features...")
    
    # Make copies to avoid modifying originals
    train = train_df.copy()
    test = test_df.copy()
    
    # Extract episode number (simplified approach)
    train['Episode_Num'] = train['Episode_Title'].str.extract(r'Episode (\d+)').astype(float)
    test['Episode_Num'] = test['Episode_Title'].str.extract(r'Episode (\d+)').astype(float)
    
    # Fill missing episode numbers with mean
    mean_episode = train['Episode_Num'].mean()
    train['Episode_Num'] = train['Episode_Num'].fillna(mean_episode)
    test['Episode_Num'] = test['Episode_Num'].fillna(mean_episode)
    
    # Round episode length (key predictor)
    for k in range(3):
        col_name = f'ELm_r{k}'
        train[col_name] = train['Episode_Length_minutes'].round(k)
        test[col_name] = test['Episode_Length_minutes'].round(k)
    
    # Log transform of episode length
    train['log_length'] = np.log1p(train['Episode_Length_minutes'])
    test['log_length'] = np.log1p(test['Episode_Length_minutes'])
    
    # Simple binary features (no conversion issues)
    train['is_weekend'] = train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(float)
    test['is_weekend'] = test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(float)
    
    # Basic numeric interactions
    train['length_x_host'] = train['Episode_Length_minutes'] * train['Host_Popularity_percentage']
    test['length_x_host'] = test['Episode_Length_minutes'] * test['Host_Popularity_percentage']
    
    train['length_x_guest'] = train['Episode_Length_minutes'] * train['Guest_Popularity_percentage']
    test['length_x_guest'] = test['Episode_Length_minutes'] * test['Guest_Popularity_percentage']
    
    train['length_x_ads'] = train['Episode_Length_minutes'] * train['Number_of_Ads']
    test['length_x_ads'] = test['Episode_Length_minutes'] * test['Number_of_Ads']
    
    train['host_x_guest'] = train['Host_Popularity_percentage'] * train['Guest_Popularity_percentage']
    test['host_x_guest'] = test['Host_Popularity_percentage'] * test['Guest_Popularity_percentage']
    
    # Fill NaN values in numeric columns
    numeric_cols = train.select_dtypes(include=['float', 'int']).columns
    for col in numeric_cols:
        if col != TARGET:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    # Drop unnecessary columns
    train = train.drop(['Episode_Title'], axis=1, errors='ignore')
    test = test.drop(['Episode_Title'], axis=1, errors='ignore')
    
    return train, test

def simple_target_encoding(train_df, test_df):
    """Simple target encoding with no conversion issues"""
    print("Performing target encoding...")
    
    # Make copies
    train = train_df.copy()
    test = test_df.copy()
    
    # Fill missing values in categorical columns
    cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    for col in cat_cols:
        train[col] = train[col].fillna('Unknown')
        test[col] = test[col].fillna('Unknown')
    
    # Use KFold for target encoding
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Target encode each categorical column
    for col in cat_cols:
        # Create column name for encoded feature
        encoded_name = f'te_{col}'
        
        # Initialize with zeros
        train[encoded_name] = 0
        test[encoded_name] = 0
        
        # Train/validation splits for encoding
        for train_idx, val_idx in kf.split(train):
            # Get the split
            train_part = train.iloc[train_idx]
            val_part = train.iloc[val_idx]
            
            # Calculate target mean for each category
            target_means = train_part.groupby(col)[TARGET].mean()
            
            # Apply to validation fold
            train.loc[val_idx, encoded_name] = val_part[col].map(target_means).fillna(train_part[TARGET].mean())
        
        # Encode test data using all training data
        full_target_means = train.groupby(col)[TARGET].mean()
        test[encoded_name] = test[col].map(full_target_means).fillna(train[TARGET].mean())
    
    return train, test

def train_xgboost_model(X_train, y_train, X_test):
    """Train a simple XGBoost model with no categorical handling"""
    print("Training XGBoost model...")
    
    # Only use numeric columns to avoid any issues
    numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns
    X_train_safe = X_train[numeric_cols]
    X_test_safe = X_test[numeric_cols]
    
    print(f"Using {len(numeric_cols)} numeric features")
    
    # Setup cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Initialize arrays for predictions
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    # Train model for each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold+1}/5")
        
        # Split data
        X_tr, X_val = X_train_safe.iloc[train_idx], X_train_safe.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create model with safe parameters
        model = XGBRegressor(
            tree_method='hist',
            max_depth=10,
            learning_rate=0.01,
            n_estimators=3000,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=100,
            random_state=SEED
        )
        
        # Train model
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # Make predictions
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test_safe) / 5
        
        # Calculate score
        fold_score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} RMSE: {fold_score:.5f}")
    
    # Calculate overall score
    overall_score = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"Overall RMSE: {overall_score:.5f}")
    
    # Create one final model on all data for feature importance
    final_model = XGBRegressor(
        tree_method='hist',
        max_depth=10,
        learning_rate=0.01,
        n_estimators=1000,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED
    )
    
    final_model.fit(X_train_safe, y_train)
    
    return oof_preds, test_preds, overall_score, final_model, numeric_cols

def main():
    # Load data
    print("Loading data...")
    train, test = load_data()
    
    # Save IDs for submission
    test_ids = test['id'].copy()
    
    # Basic feature engineering
    train, test = safe_feature_engineering(train, test)
    
    # Target encoding
    train, test = simple_target_encoding(train, test)
    
    # Prepare data for modeling
    X_train = train.drop([TARGET, 'id'], axis=1, errors='ignore')
    y_train = train[TARGET]
    X_test = test.drop(['id'], axis=1, errors='ignore')
    
    # Train model
    oof_preds, test_preds, rmse, model, numeric_cols = train_xgboost_model(X_train, y_train, X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        TARGET: test_preds
    })
    
    # Save submission
    submission_path = '/kaggle/working/minimal_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    
    # Plot feature importance
    try:
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 30 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(30))
        plt.title('Top 30 Feature Importance')
        plt.tight_layout()
        plt.savefig('/kaggle/working/feature_importance.png')
        print("Feature importance plot saved")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
    
    print(f"Final RMSE: {rmse:.5f}")
    
    return {
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'rmse': rmse,
        'model': model
    }

# Run the main function
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    try:
        print("Starting minimal robust podcast model...")
        results = main()
        
        # Print execution time
        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        print(f"\nFinal RMSE: {results['rmse']:.5f}")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

