!pip install pycaret


import pandas as pd
import numpy as np
from pycaret.time_series import *
from datetime import datetime


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


import pandas as pd
import numpy as np
from pycaret.time_series import *

def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

    # Drop rows where the target variable is missing, only if it exists
    if 'num_sold' in df.columns:
        df = df.dropna(subset=['num_sold'])
    
    return df


def train_models(df):
    models = {}
    groups = df.groupby(['country', 'store', 'product'])
    
    for (country, store, product), group_data in groups:
        key = f"{country}_{store}_{product}"
        
        # Prepare time series data
        ts_data = group_data.set_index('date')[['num_sold']].resample('D').mean()
        ts_data = ts_data.fillna(method='ffill').fillna(method='bfill')
        
        # Add 1 to handle zeros before log transform
        ts_data['num_sold'] = ts_data['num_sold'] + 1
        
        # Dynamically infer seasonal period if possible
        inferred_freq = pd.infer_freq(ts_data.index)
        seasonal_period = 'D'  # Default to daily frequency
        if inferred_freq:
            valid_freqs = ['B', 'D', 'W', 'M', 'Q', 'A', 'Y', 'H', 'T', 'S']
            seasonal_period = inferred_freq if inferred_freq in valid_freqs else 'D'
        
        # Setup with minimal transformations
        exp = TSForecastingExperiment()
        exp.setup(
            data=ts_data,
            target='num_sold',
            fold=3,
            numeric_imputation_target='mean',
            transform_target='log',  
            seasonal_period=seasonal_period,
            remove_harmonics=True
        )
        
        # Train model with error handling
        try:
            best = exp.compare_models(n_select=1)
            models[key] = best
        except Exception as e:
            print(f"Error training model for {key}: {str(e)}")
            # Use simple moving average as fallback
            ts_data['MA7'] = ts_data['num_sold'].rolling(window=7).mean()
            models[key] = ts_data['MA7'].mean()
    
    return models


def make_predictions(models, test_df):
    predictions = []
    groups = test_df.groupby(['country', 'store', 'product'])
    
    for (country, store, product), group_data in groups:
        key = f"{country}_{store}_{product}"
        try:
            if key in models:
                if isinstance(models[key], float):
                    # If using fallback mean prediction
                    predictions.extend([models[key]] * len(group_data))
                else:
                    # Regular model prediction
                    horizon = len(group_data)
                    forecast = predict_model(models[key], fh=horizon)
                    predictions.extend(forecast['y_pred'].values)
        except Exception as e:
            print(f"Error predicting for {key}: {str(e)}")
            # Use last known value as fallback
            predictions.extend([models[key] if isinstance(models[key], float) else models[key].mean()] * len(group_data))
    
    return predictions

def run_forecast(train_df, test_df):
    train_df = prepare_data(train_df)
    test_df = prepare_data(test_df)
    
    print("Training models...")
    models = train_models(train_df)
    
    print("Generating predictions...")
    predictions = make_predictions(models, test_df)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'num_sold': predictions
    })
    
    return validate_predictions(submission)

def validate_predictions(submission):
    submission['num_sold'] = submission['num_sold'].fillna(0)
    submission['num_sold'] = submission['num_sold'].clip(lower=0)
    return submission



run_forecast(train_df, test_df)

