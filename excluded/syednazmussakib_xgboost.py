!pip install pycaret


import pandas as pd
import numpy as np
from pycaret.time_series import *
from sklearn.metrics import mean_absolute_percentage_error


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


def calculate_mape(y_true, y_pred):
    """Calculate MAPE with handling for edge cases"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Handle zeros in y_true
    mask = y_true != 0
    if not mask.any():
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'])
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    df['is_quarter_start'] = df['date'].dt.is_quarter_start
    df['is_quarter_end'] = df['date'].dt.is_quarter_end
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    if 'num_sold' in df.columns:
        df = df.dropna(subset=['num_sold'])
    
    return df

def create_xgboost_model(ts_data, seasonal_period='auto'):
    """Create XGBoost model optimized for MAPE"""
    exp = TSForecastingExperiment()
    exp.setup(
        data=ts_data,
        target='num_sold',
        fold=5,
        numeric_imputation_target='mean',
        transform_target='log',  # Log transform helps with MAPE
        seasonal_period=seasonal_period
    )
    
    # Create XGBoost model with MAPE-optimized parameters
    xgboost_model = create_model('xgboost_cds_dt', 
                                fold=5,
                                seasonal_period=seasonal_period,
                                estimator_kwargs={
                                    'n_estimators': 1000,  # Increased for better accuracy
                                    'max_depth': 8,
                                    'learning_rate': 0.005,  # Smaller learning rate
                                    'subsample': 0.8,
                                    'colsample_bytree': 0.8,
                                    'min_child_weight': 3,
                                    'gamma': 0.1,
                                    'reg_alpha': 0.1,
                                    'reg_lambda': 1,
                                    'objective': 'reg:squarederror',  # Good for MAPE
                                    'random_state': 42
                                })
    
    return xgboost_model

def train_models(df):
    models = {}
    groups = df.groupby(['country', 'store', 'product'])
    
    for (country, store, product), group_data in groups:
        key = f"{country}_{store}_{product}"
        print(f"Training model for {key}")
        
        # Prepare time series data
        ts_data = group_data.set_index('date')[['num_sold']].resample('D').mean()
        ts_data = ts_data.fillna(method='ffill').fillna(method='bfill')
        
        # Handle zeros and small values for MAPE
        ts_data['num_sold'] = ts_data['num_sold'].clip(lower=0.1)
        
        try:
            # Detect seasonality
            seasonal_period = detect_seasonality(ts_data)
            
            # Train XGBoost model
            model = create_xgboost_model(ts_data, seasonal_period)
            
            # Fine-tune the model specifically for MAPE
            tuned_model = tune_model(model, 
                                   optimize='MAPE',  # Optimize for MAPE
                                   search_algorithm='random',
                                   n_iter=15,  # Increased iterations
                                   choose_better=True)
            
            # Finalize the model
            final_model = finalize_model(tuned_model)
            models[key] = final_model
            
        except Exception as e:
            print(f"Error training model for {key}: {str(e)}")
            # Fallback to weighted moving average
            ts_data['WMA'] = ts_data['num_sold'].rolling(
                window=14, min_periods=1
            ).mean()
            models[key] = ts_data['WMA'].mean()
    
    return models

def detect_seasonality(ts_data):
    """Detect appropriate seasonal period"""
    try:
        from statsmodels.tsa.stattools import acf
        
        # Calculate ACF
        acf_values = acf(ts_data['num_sold'], nlags=365)
        
        # Find peaks in ACF
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acf_values, distance=2)
        
        if len(peaks) > 0:
            # Get the first significant peak
            first_peak = peaks[0]
            if first_peak > 1:
                return first_peak
        
        return 7  # Default to weekly seasonality
    except:
        return 'auto'



def make_predictions(models, test_df):
    # Initialize predictions array with the same length as test_df
    predictions = np.zeros(len(test_df))
    
    # Create a mapping of row indices for each group
    groups = test_df.groupby(['country', 'store', 'product'])
    
    current_idx = 0
    for (country, store, product), group_data in groups:
        key = f"{country}_{store}_{product}"
        group_size = len(group_data)
        
        try:
            if key in models:
                if isinstance(models[key], float):
                    # If using fallback mean prediction
                    predictions[current_idx:current_idx + group_size] = models[key]
                else:
                    # Regular model prediction
                    forecast = predict_model(models[key], fh=group_size)
                    preds = np.exp(forecast['y_pred'].values) - 1
                    preds = np.maximum(preds, 0)  # Ensure non-negative
                    predictions[current_idx:current_idx + group_size] = preds
            else:
                # If no model exists for this group, use global mean
                predictions[current_idx:current_idx + group_size] = train_df['num_sold'].mean()
                
        except Exception as e:
            print(f"Error predicting for {key}: {str(e)}")
            # Use fallback prediction
            fallback_value = models[key] if isinstance(models[key], float) else train_df['num_sold'].mean()
            predictions[current_idx:current_idx + group_size] = fallback_value
            
        current_idx += group_size
    
    return predictions

def validate_predictions(submission):
    # Handle invalid values
    submission['num_sold'] = submission['num_sold'].fillna(0)
    submission['num_sold'] = submission['num_sold'].clip(lower=0)
    
    # Round predictions to nearest integer
    submission['num_sold'] = submission['num_sold'].round()
    
    return submission

def run_forecast(train_df, test_df):
    print("Preparing data...")
    train_df = prepare_data(train_df)
    test_df = prepare_data(test_df)
    
    print("Training XGBoost models...")
    models = train_models(train_df)
    
    print("Generating predictions...")
    predictions = make_predictions(models, test_df)
    
    # Verify prediction length matches test_df length
    assert len(predictions) == len(test_df), f"Predictions length {len(predictions)} doesn't match test_df length {len(test_df)}"
    
    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'num_sold': predictions
    })
    
    return validate_predictions(submission)


# Run forecast
submission = run_forecast(train_df, test_df)

# Save results
submission.to_csv('submission.csv', index=False)




