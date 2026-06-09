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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def simple_prophet_outlier_detection(df, feature_col, timestamp_col='timestamp'):
    """
    Simple example of using Prophet to detect outliers in a single feature.
    
    The idea: Prophet models the expected behavior of the feature over time.
    Points that fall far outside Prophet's confidence interval are outliers.
    """
    print(f"Detecting outliers in {feature_col} using Prophet...")
    
    # 1. Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': df[timestamp_col],
        'y': df[feature_col]
    })
    
    # Remove obvious bad values
    prophet_df = prophet_df[np.isfinite(prophet_df['y'])]
    
    # Resample to hourly for efficiency (adjust based on your needs)
    prophet_hourly = prophet_df.set_index('ds').resample('1H').mean().reset_index()
    prophet_hourly = prophet_hourly.dropna()
    
    # 2. Fit Prophet model
    model = Prophet(
        changepoint_prior_scale=0.05,  # Low value = less sensitive to outliers
        interval_width=0.95,  # 95% confidence interval
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    
    model.fit(prophet_hourly)
    
    # 3. Generate predictions
    forecast = model.predict(prophet_hourly)
    
    # 4. Identify outliers
    # Method 1: Points outside confidence interval
    outliers_ci = (
        (prophet_hourly['y'] < forecast['yhat_lower']) | 
        (prophet_hourly['y'] > forecast['yhat_upper'])
    )
    
    # Method 2: Points with large standardized residuals
    residuals = prophet_hourly['y'] - forecast['yhat']
    residual_std = residuals.std()
    outliers_residual = np.abs(residuals) > 3 * residual_std
    
    # Combine both methods
    outliers = outliers_ci & outliers_residual
    
    # 5. Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Time series with outliers
    ax1.plot(prophet_hourly['ds'], prophet_hourly['y'], 'b.', alpha=0.5, label='Actual')
    ax1.plot(forecast['ds'], forecast['yhat'], 'g-', linewidth=2, label='Prophet Fit')
    ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     alpha=0.2, color='green', label='95% CI')
    
    # Highlight outliers
    outlier_points = prophet_hourly[outliers]
    ax1.scatter(outlier_points['ds'], outlier_points['y'], 
               color='red', s=100, edgecolor='darkred', linewidth=2, 
               label=f'Outliers ({outliers.sum()})', zorder=10)
    
    ax1.set_title(f'Prophet Outlier Detection: {feature_col}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual distribution
    ax2.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=-3*residual_std, color='red', linestyle='--', label='±3σ threshold')
    ax2.axvline(x=3*residual_std, color='red', linestyle='--')
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residual (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Return outlier information
    outlier_info = {
        'n_outliers': outliers.sum(),
        'outlier_pct': outliers.sum() / len(prophet_hourly) * 100,
        'outlier_timestamps': outlier_points['ds'].tolist(),
        'outlier_values': outlier_points['y'].tolist(),
        'residual_std': residual_std
    }
    
    print(f"Found {outlier_info['n_outliers']} outliers ({outlier_info['outlier_pct']:.2f}%)")
    
    return outlier_info, model, forecast


# Quick example for multiple features
def detect_outliers_multiple_features(data_path, features_to_check=None):
    """
    Run outlier detection on multiple features.
    """
    # Load data
    if features_to_check is None:
        # Default to some common features
        features_to_check = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
    
    df = pd.read_parquet(data_path, columns=['timestamp'] + features_to_check)
    
    # Ensure timestamp
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range('2023-03-01', periods=len(df), freq='T')
    
    # Analyze each feature
    outlier_summary = {}
    
    for feature in features_to_check:
        if feature in df.columns:
            print(f"\n{'='*60}")
            outlier_info, model, forecast = simple_prophet_outlier_detection(df, feature)
            outlier_summary[feature] = outlier_info
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(outlier_summary.keys())
    outlier_pcts = [outlier_summary[f]['outlier_pct'] for f in features]
    
    bars = ax.bar(features, outlier_pcts, color='coral', edgecolor='darkred', linewidth=2)
    
    # Highlight high outlier features
    for i, pct in enumerate(outlier_pcts):
        if pct > 1.0:  # More than 1% outliers
            bars[i].set_color('red')
    
    ax.set_title('Outlier Percentage by Feature', fontsize=14, fontweight='bold')
    ax.set_ylabel('Outlier %')
    ax.set_xlabel('Feature')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, pct in zip(bars, outlier_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return outlier_summary


# Practical example: Using outlier detection for data cleaning
def clean_feature_outliers(df, feature_col, outlier_info, method='cap'):
    """
    Clean outliers from a feature using different methods.
    """
    # Get outlier timestamps
    outlier_timestamps = outlier_info['outlier_timestamps']
    
    # Create copy
    df_clean = df.copy()
    
    if method == 'remove':
        # Remove rows with outliers
        mask = ~df_clean['timestamp'].isin(outlier_timestamps)
        df_clean = df_clean[mask]
        print(f"Removed {len(outlier_timestamps)} rows with outliers")
        
    elif method == 'cap':
        # Cap outliers at 99th percentile
        lower_cap = df_clean[feature_col].quantile(0.01)
        upper_cap = df_clean[feature_col].quantile(0.99)
        
        df_clean[feature_col] = df_clean[feature_col].clip(lower=lower_cap, upper=upper_cap)
        print(f"Capped {feature_col} to range [{lower_cap:.2f}, {upper_cap:.2f}]")
        
    elif method == 'interpolate':
        # Replace outliers with interpolated values
        outlier_mask = df_clean['timestamp'].isin(outlier_timestamps)
        df_clean.loc[outlier_mask, feature_col] = np.nan
        df_clean[feature_col] = df_clean[feature_col].interpolate(method='linear')
        print(f"Interpolated {len(outlier_timestamps)} outlier values")
    
    return df_clean


# Example usage
if __name__ == "__main__":
    # Example 1: Single feature analysis
    data_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
    df = pd.read_parquet(data_path, columns=['timestamp', 'volume'])
    
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range('2023-03-01', periods=len(df), freq='T')
    
    # Detect outliers in volume
    outlier_info, model, forecast = simple_prophet_outlier_detection(df, 'volume')
    
    # Example 2: Multiple features
    outlier_summary = detect_outliers_multiple_features(
        data_path,
        features_to_check=['volume', 'bid_qty', 'ask_qty']
    )
    
    # Example 3: Clean the data
    df_clean = clean_feature_outliers(df, 'volume', outlier_info, method='cap')
    
    print("\nOutlier detection complete!")

