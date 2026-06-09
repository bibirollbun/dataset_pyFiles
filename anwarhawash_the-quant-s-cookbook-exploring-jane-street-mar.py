import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
import os
import multiprocessing
from numba import njit, prange
import gc


# Disable warnings and set performance-related settings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning
os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())

# Global configuration
CONFIG = {
    'use_gpu': True,           # Set to False if GPU not available
    'sample_size': 200000,       # Set to an integer for sampling
    'max_partitions': 10,      # Number of data partitions to load
    'use_dask': True,         # Set to True for distributed computing
    'target_column': 'responder_6',  # Target variable
    'weight_column': 'weight', # Weight column
    'random_seed': 42,         # Random seed for reproducibility
    'data_path': '/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet', 
    'memory_efficient': True,  # Enable memory optimizations
    'verbose': False           # Enable detailed logging
}

def log(message, force=False):
    """Utility function for controlled logging"""
    if CONFIG['verbose'] or force:
        print(message)


def get_optimal_dtypes(df):
    """
    Optimize dataframe memory usage by selecting appropriate dtypes
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    dict: Optimized dtypes for columns
    """
    dtypes = {}
    
    for col in df.columns:
        # Categoricals
        if col in ['symbol_id'] or df[col].nunique() < 100:
            dtypes[col] = 'category'
        # Integers
        elif 'id' in col or col.endswith('_id'):
            dtypes[col] = 'int32'
        # Floats
        elif df[col].dtype == np.float64:
            dtypes[col] = 'float32'
    
    return dtypes

def memory_efficient_load_data(file_path, sample_size=None):
    """
    Load data with minimal memory usage
    
    Parameters:
    file_path (str): Path to the parquet file
    sample_size (int, optional): Number of rows to sample
    
    Returns:
    pandas.DataFrame: Loaded and optimized dataframe
    """
    # First load a small sample to determine dtypes
    sample_df = pd.read_parquet(file_path, engine='pyarrow')
    if sample_size is not None:
        sample_df = sample_df.sample(min(100000, sample_size), random_state=CONFIG['random_seed'])
    
    # Get optimized dtypes
    dtypes = get_optimal_dtypes(sample_df)
    del sample_df
    gc.collect()
    
    # Read parquet without dtypes
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    # Sample if needed
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=CONFIG['random_seed'])
    
    # Apply optimized dtypes
    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)
    
    # Force garbage collection
    gc.collect()
    
    return df
def load_train_data(sample_size=None, max_partitions=10, use_dask=False):
    """
    Optimized data loading with optional Dask support for large datasets
    
    Parameters:
    sample_size (int, optional): Number of rows to sample from each partition
    max_partitions (int): Maximum number of partitions to load
    use_dask (bool): Use Dask for distributed loading
    
    Returns:
    pandas.DataFrame: Combined training data
    """
    log("Loading training data...", force=True)
    
    if use_dask:
        try:
            import dask.dataframe as dd
            dfs = []
            for partition_id in range(max_partitions):
                partition_path = f'{CONFIG["data_path"]}/partition_id={partition_id}/part-0.parquet'
                try:
                    ddf = dd.read_parquet(partition_path)
                    if sample_size:
                        ddf = ddf.sample(frac=sample_size/len(ddf))
                    dfs.append(ddf)
                    log(f"Loaded partition {partition_id}")
                except Exception as e:
                    log(f"Error loading partition {partition_id}: {e}")
            
            log("Computing combined dataframe...")
            combined_df = dd.concat(dfs).compute()
            
            # Apply memory optimizations
            if CONFIG['memory_efficient']:
                dtypes = get_optimal_dtypes(combined_df)
                for col, dtype in dtypes.items():
                    combined_df[col] = combined_df[col].astype(dtype)
            
            log(f"Combined data shape: {combined_df.shape}", force=True)
            return combined_df
        
        except ImportError:
            log("Dask not available, falling back to pandas")
    
    # Pandas-based loading
    train_dfs = []
    
    for partition_id in range(max_partitions):
        partition_path = f'{CONFIG["data_path"]}/partition_id={partition_id}/part-0.parquet'
        
        try:
            if CONFIG['memory_efficient']:
                df = memory_efficient_load_data(partition_path, sample_size)
            else:
                df = pd.read_parquet(partition_path, engine='pyarrow')
                if sample_size:
                    df = df.sample(sample_size, random_state=CONFIG['random_seed'])
            
            train_dfs.append(df)
            log(f"Loaded partition {partition_id} with shape {df.shape}")
            
            # Free memory
            gc.collect()
            
        except Exception as e:
            log(f"Error loading partition {partition_id}: {e}")
    
    # Combine all partitions
    combined_df = pd.concat(train_dfs, ignore_index=True)
    log(f"Combined training data shape: {combined_df.shape}", force=True)
    
    # Clear memory
    del train_dfs
    gc.collect()
    
    return combined_df


@njit
def compute_rolling_features(col_data, window_sizes):
    """
    Compute rolling features for a single column
    
    Parameters:
    col_data (np.ndarray): Input column data
    window_sizes (tuple): Window sizes for rolling computations
    
    Returns:
    np.ndarray: Computed rolling features
    """
    rows = len(col_data)
    result = np.zeros((rows, len(window_sizes) * 2))
    
    for idx, window in enumerate(window_sizes):
        # Rolling mean
        rolling_mean = np.zeros_like(col_data, dtype=np.float32)
        for i in range(rows):
            start = max(0, i - window + 1)
            if i >= start:
                rolling_mean[i] = np.mean(col_data[start:i+1])
        
        # Rolling standard deviation
        rolling_std = np.zeros_like(col_data, dtype=np.float32)
        for i in range(rows):
            start = max(0, i - window + 1)
            if i >= start and i - start > 1:  # Need at least 2 points for std
                rolling_std[i] = np.std(col_data[start:i+1])
        
        result[:, idx * 2] = rolling_mean
        result[:, idx * 2 + 1] = rolling_std
    
    return result

@njit(parallel=True)
def fast_rolling_features(data, window_sizes=(3, 5)):
    """
    Numba-accelerated rolling window feature computation
    
    Parameters:
    data (np.ndarray): Input data array
    window_sizes (tuple): Window sizes for rolling computations
    
    Returns:
    np.ndarray: Computed rolling features
    """
    rows, cols = data.shape
    
    # Preallocate result array
    result = np.zeros((rows, cols * len(window_sizes) * 2), dtype=np.float32)
    
    # Compute rolling features for each column
    for col in prange(cols):
        col_features = compute_rolling_features(data[:, col], window_sizes)
        
        # Copy computed features to result array
        for w_idx in range(len(window_sizes)):
            result[:, col * len(window_sizes) * 2 + w_idx * 2] = col_features[:, w_idx * 2]
            result[:, col * len(window_sizes) * 2 + w_idx * 2 + 1] = col_features[:, w_idx * 2 + 1]
    
    return result

def advanced_feature_engineering(df, memory_efficient=True):
    """
    Optimized feature engineering with Numba and vectorized operations
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    memory_efficient (bool): Use memory-efficient operations
    
    Returns:
    pandas.DataFrame: Dataframe with engineered features
    """
    log("Performing feature engineering...", force=True)
    
    # Create a copy or work with input dataframe
    if memory_efficient:
        result = df  # Work with original to save memory
    else:
        result = df.copy()
    
    # Identify feature columns
    feature_cols = [col for col in result.columns if col.startswith('feature_')]
    log(f"Found {len(feature_cols)} feature columns")
    
    # 1. Time-based Features (Vectorized)
    result['day_of_week'] = result['date_id'] % 7
    result['time_of_day'] = result['time_id'] % 24
    
    # 2. Rolling Window Features (Vectorized + Numba)
    # Process in chunks to manage memory
    chunk_size = 10 if memory_efficient else len(feature_cols)
    
    for i in range(0, len(feature_cols), chunk_size):
        chunk_cols = feature_cols[i:i+chunk_size]
        log(f"Processing chunk {i//chunk_size + 1}/{(len(feature_cols)-1)//chunk_size + 1}")
        
        feature_matrix = result[chunk_cols].values
        rolling_features = fast_rolling_features(feature_matrix)
        
        # Add rolling features back to dataframe
        for j, col in enumerate(chunk_cols):
            for k, suffix in enumerate(['_roll_mean_3', '_roll_std_3', '_roll_mean_5', '_roll_std_5']):
                result[f'{col}{suffix}'] = rolling_features[:, j * 4 + k]
        
        # Clean up to free memory
        del feature_matrix, rolling_features
        gc.collect()
    
    # 3. Focus on top features for other transformations to save memory
    top_features = feature_cols[:min(20, len(feature_cols))]
    
    # 4. Momentum Features (Vectorized)
    for feature in top_features[:min(10, len(top_features))]:
        result[f'{feature}_momentum'] = result.groupby('symbol_id')[feature].pct_change()
    
    # 5. Cross-sectional Z-score (Vectorized)
    for feature in top_features[:min(5, len(top_features))]:
        result[f'{feature}_zscore'] = result.groupby('date_id')[feature].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    
    # 6. Feature Interactions (Vectorized)
    for i, feat1 in enumerate(top_features[:3]):
        for feat2 in top_features[i+1:i+3]:  # Limit interactions to save memory
            # Multiplicative interactions
            result[f'{feat1}_{feat2}_interaction'] = result[feat1] * result[feat2]
    
    # 7. Target Encoding (Vectorized)
    result['symbol_target_mean'] = result.groupby('symbol_id')[CONFIG['target_column']].transform('mean')
    result['day_target_mean'] = result.groupby('day_of_week')[CONFIG['target_column']].transform('mean')
    
    # Fill remaining NaNs
    result = result.fillna(0)
    
    log("Feature engineering completed")
    log(f"Final dataframe shape: {result.shape}", force=True)
    
    return result


def get_lgb_params(use_gpu=False):
    """
    Get optimized LightGBM parameters
    
    Parameters:
    use_gpu (bool): Enable GPU acceleration
    
    Returns:
    dict: LightGBM parameters
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart',  # More robust to overfitting
        'learning_rate': 0.05,
        'num_leaves': 128,
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 3000,
        'early_stopping_rounds': 150,
        'min_child_samples': 30,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'n_jobs': -1,  # Use all available cores
        'verbose': -1
    }
    
    # Add GPU parameters if enabled
    if use_gpu:
        try:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        except:
            log("GPU acceleration not available, falling back to CPU")
    
    return params

def train_enhanced_model(train_df, feature_cols, target_col='responder_6', weight_col='weight'):
    """
    Optimized LightGBM model training with advanced configurations
    
    Parameters:
    train_df (pandas.DataFrame): Training dataframe
    feature_cols (list): List of feature columns
    target_col (str): Target column name
    weight_col (str): Weight column name
    
    Returns:
    tuple: (model, validation_metrics)
    """
    log("Training model...", force=True)
    
    # Get model parameters
    params = get_lgb_params(CONFIG['use_gpu'])
    
    # Time-based train-validation split
    date_ids = sorted(train_df['date_id'].unique())
    split_idx = int(len(date_ids) * 0.8)
    
    train_dates = date_ids[:split_idx]
    valid_dates = date_ids[split_idx:]
    
    train_mask = train_df['date_id'].isin(train_dates)
    valid_mask = train_df['date_id'].isin(valid_dates)
    
    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, target_col]
    w_train = train_df.loc[train_mask, weight_col]
    
    X_valid = train_df.loc[valid_mask, feature_cols]
    y_valid = train_df.loc[valid_mask, target_col]
    w_valid = train_df.loc[valid_mask, weight_col]
    
    log(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")
    
    # Create LightGBM datasets with categorical feature support
    categorical_features = [
        col for col in feature_cols 
        if train_df[col].dtype.name == 'category' or 
        (train_df[col].nunique() < 100 and 'float' not in str(train_df[col].dtype))
    ]
    
    train_dataset = lgb.Dataset(
        X_train, 
        y_train, 
        weight=w_train, 
        categorical_feature=categorical_features
    )
    valid_dataset = lgb.Dataset(
        X_valid, 
        y_valid, 
        weight=w_valid, 
        categorical_feature=categorical_features
    )
    
    # Train model
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[train_dataset, valid_dataset],
        callbacks=[
            lgb.early_stopping(150),
            lgb.log_evaluation(100)
        ]
    )
    
    # Predictions and metrics
    y_pred = model.predict(X_valid)
    
    # Custom weighted metrics
    mse = mean_squared_error(y_valid, y_pred, sample_weight=w_valid)
    weighted_r2 = 1 - (np.sum(w_valid * (y_valid - y_pred)**2) / 
                       np.sum(w_valid * (y_valid - y_valid.mean())**2))
    
    log(f"Model training completed. Best iteration: {model.best_iteration}", force=True)
    log(f"MSE: {mse:.6f}, Weighted RÂ²: {weighted_r2:.6f}", force=True)
    
    # Free memory
    del X_train, X_valid, y_train, y_valid, w_train, w_valid
    gc.collect()
    
    return model, {
        'mse': mse,
        'weighted_r2': weighted_r2,
        'best_iteration': model.best_iteration
    }


def predict_batch(model, data, feature_cols, batch_size=100000):
    """
    Make predictions in batches to manage memory
    
    Parameters:
    model (LGBMModel): Trained LightGBM model
    data (pandas.DataFrame): Data to predict on
    feature_cols (list): Feature columns
    batch_size (int): Batch size for prediction
    
    Returns:
    numpy.ndarray: Predictions
    """
    n_samples = len(data)
    predictions = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_data = data.iloc[i:end]
        batch_preds = model.predict(batch_data[feature_cols])
        predictions[i:end] = batch_preds
    
    return predictions


def prepare_data():
    """
    Load and prepare data
    
    Returns:
    pandas.DataFrame: Prepared dataframe
    """
    # Load training data
    train_df = load_train_data(
        sample_size=CONFIG['sample_size'],
        max_partitions=CONFIG['max_partitions'],
        use_dask=CONFIG['use_dask']
    )
    
    # Basic exploratory analysis
    log("\nðŸ“Š Basic Data Overview:", force=True)
    log(f"Total Rows: {len(train_df)}", force=True)
    log(f"Unique Symbols: {train_df['symbol_id'].nunique()}", force=True)
    log(f"Date Range: {train_df['date_id'].min()} - {train_df['date_id'].max()}", force=True)
    
    return train_df

def remove_correlated_features(train_df, feature_cols, threshold=0.9):
    """
    Remove highly correlated features based on a threshold.
    
    Parameters:
    train_df (pd.DataFrame): The training dataframe.
    feature_cols (list): List of feature columns to consider.
    threshold (float): Correlation threshold (default: 0.9).
    
    Returns:
    list: List of selected features after removing correlated ones.
    """
    # Compute the correlation matrix
    corr_matrix = train_df[feature_cols].corr().abs()
    
    # Select upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Keep only the features that are not highly correlated
    selected_features = [col for col in feature_cols if col not in to_drop]
    
    log(f"Removed {len(to_drop)} highly correlated features", force=True)
    log(f"Selected {len(selected_features)} features after correlation filtering", force=True)
    
    return selected_features

def engineer_features(train_df):
    """
    Perform feature engineering and correlation-based feature selection.
    """
    # Feature engineering
    train_df_engineered = advanced_feature_engineering(
        train_df, 
        memory_efficient=CONFIG['memory_efficient']
    )
    
    # Feature selection with comprehensive filtering
    feature_cols = [
        col for col in train_df_engineered.columns 
        if (col.startswith('feature_') or 
            any(suffix in col for suffix in [
                '_roll_mean', '_roll_std', 
                '_momentum', '_zscore', 
                '_interaction', '_ratio', 
                '_target_mean', 'day_of_week', 'time_of_day'
            ])) 
        and col not in [CONFIG['target_column'], CONFIG['weight_column']]
    ]
    
    log(f"Selected {len(feature_cols)} features", force=True)
    
    # Remove highly correlated features
    selected_features = remove_correlated_features(train_df_engineered, feature_cols, threshold=0.9)
    
    log(f"Remaining {len(selected_features)} features after correlation filtering", force=True)
    
    return train_df_engineered, selected_features  


def train_model(train_df_engineered, feature_cols):
    """
    Train prediction model with correlation-based feature selection.
    """
    # Train model with selected features
    model, metrics = train_enhanced_model(
        train_df_engineered, 
        feature_cols,
        target_col=CONFIG['target_column'],
        weight_col=CONFIG['weight_column']
    )
    
    # Print results
    log("\nðŸ“ˆ Model Performance with Selected Features:", force=True)
    log(f"Mean Squared Error: {metrics['mse']:.6f}", force=True)
    log(f"Weighted R-squared: {metrics['weighted_r2']:.6f}", force=True)
    log(f"Best Iteration: {metrics['best_iteration']}", force=True)
    
    return model, metrics, feature_cols


def save_model(model, feature_cols, output_path='model.txt'):
    """
    Save trained model and feature columns
    
    Parameters:
    model (LGBMModel): Trained model
    feature_cols (list): Feature columns
    output_path (str): Output path for model
    """
    # Save model
    model.save_model(output_path)
    log(f"Model saved to {output_path}", force=True)
    
    # Save feature columns
    import json
    with open('feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    log("Feature columns saved to feature_cols.json", force=True)


def main():
    """
    Main execution function with correlation-based feature selection.
    """
    # Load and prepare data
    train_df = prepare_data()
    
    # Feature engineering and correlation-based feature selection
    train_df_engineered, selected_features = engineer_features(train_df)
    
    # Free memory
    del train_df
    gc.collect()
    
    # Train model with selected features
    model, metrics, selected_features = train_model(train_df_engineered, selected_features)
    
    # Save model and selected features
    save_model(model, selected_features)
    
    return model, metrics, selected_features


if __name__ == "__main__":
    # Set configuration options here
    CONFIG.update({
        'use_gpu': True,
        'sample_size': 200000,
        'max_partitions': 5,
        'use_dask': True,
        'memory_efficient': False,
        'verbose': True
    })
    
    # Run full pipeline
    model, metrics, selected_feature_cols = main()

