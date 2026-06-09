import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib as plt
from datetime import datetime, timedelta
import re


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data_fpath =  "/kaggle/input/hill-of-towie-wind-turbine-power-prediction/training_dataset.parquet"
submission_data_fpath =  "/kaggle/input/hill-of-towie-wind-turbine-power-prediction/submission_dataset.parquet"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
input_df = pd.read_parquet(train_data_fpath)
test_df = pd.read_parquet(submission_data_fpath)
input_df.head(3)


def add_lagged_ActPower(df, timestamp_col='TimeStamp_StartFormat', turbine_prefix=';'):
    """
    Add time lagged Actual Power for each turbine (10min and 1hr into the past)
    
    Args:
        df (pd.DataFrame): Must contain:
            - timestamp_col: DateTime column
            - f'Wind_speed_{turbine_prefix}N' for each turbine
        turbine_prefix (str): Prefix used in turbine column names
        
    Returns:
        pd.DataFrame: Original df + new lagged production columns
    """
    df = df.copy()
    
    # Sort by time to ensure proper shifting
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Time deltas for lags
    lag_10min = timedelta(minutes=10)
    lag_1hr = timedelta(minutes=60)
    
    for turbine_num in [2,3,4,5,7]:  # For turbines 2,3,4,5,7 (6 being predicted)
        wind_col = f'wtc_ActPower_mean{turbine_prefix}{turbine_num}'
        
        # Create temporary DataFrame for merging
        lagged = df[[timestamp_col, wind_col]].copy()
        
        # Calculate 10min lag
        lagged['temp_10min'] = lagged[timestamp_col] + lag_10min
        lagged = lagged.rename(columns={wind_col: f'{wind_col}_lag10min'})
        df = pd.merge_asof(
            df.sort_values(timestamp_col),
            lagged[['temp_10min', f'{wind_col}_lag10min']].sort_values('temp_10min'),
            left_on=timestamp_col,
            right_on='temp_10min',
            direction='backward'
        ).drop(columns='temp_10min')
        
        # Calculate 1hr lag
        lagged['temp_1hr'] = lagged[timestamp_col] + lag_1hr
        lagged = lagged.rename(columns={f'{wind_col}_lag10min': f'{wind_col}_lag1hr'})
        df = pd.merge_asof(
            df.sort_values(timestamp_col),
            lagged[['temp_1hr', f'{wind_col}_lag1hr']].sort_values('temp_1hr'),
            left_on=timestamp_col,
            right_on='temp_1hr',
            direction='backward'
        ).drop(columns='temp_1hr')
    
    return df


def rel_dir(df):
    for turbine_num in [2, 3, 4, 5, 7]:
        yaw_col = f'wtc_ScYawPos_mean;{turbine_num}'
        wind_dir_col = 'ERA5_wind_direction_10m'  # or 100m depending on your needs
        
        # Calculate relative angle
        df[f'relative_angle_{turbine_num}'] = (df[wind_dir_col] - df[yaw_col]).abs()
        
        # Normalize to [-180, 180]
        df[f'relative_angle_{turbine_num}'] = df[f'relative_angle_{turbine_num}'].apply(
            lambda x: x if x <= 180 else 360 - x
        )
    return df


def day_feature_adder(df):
    day_of_year = df['TimeStamp_StartFormat'].dt.dayofyear
    df['daylight_hours'] = 8 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    df['is_peak_wind'] = np.where(df['TimeStamp_StartFormat'].dt.month.isin([12, 1, 2]),1,0)



    df['minute_sin'] = np.sin(2 * np.pi * df['TimeStamp_StartFormat'].dt.minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['TimeStamp_StartFormat'].dt.minute / 60)

    df['hour_sin'] = np.sin(2 * np.pi * df['TimeStamp_StartFormat'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['TimeStamp_StartFormat'].dt.hour / 24)

    df['dayofyear_sin'] = np.sin(2 * np.pi * df['TimeStamp_StartFormat'].dt.dayofyear / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['TimeStamp_StartFormat'].dt.dayofyear / 365)

    df['month_sin'] = np.sin(2 * np.pi * df['TimeStamp_StartFormat'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['TimeStamp_StartFormat'].dt.month / 12)
    return df


def add_trig_transforms(df, substrings=('Pos', 'direction'), radians=True):
    """
    Adds sin and cos transformations for columns containing any of the specified substrings which indicate angles
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    substrings : tuple of str (default=('Pos', 'direction'))
        Substrings to identify columns to transform
    radians : bool (default=True)
        If False, converts values from degrees to radians first
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with new sin/cos columns added
    """
    df_transformed = df.copy()
    
    for col in df.columns:
        if any(sub in col for sub in substrings):
            values = df[col]
            
            if not radians:
                values = np.radians(values)
                
            df_transformed[f'{col}_sin'] = np.sin(values)
            df_transformed[f'{col}_cos'] = np.cos(values)
    
    return df_transformed


#remove training fields without a value for the target
input_df= input_df[~input_df.target.isna()]
#add lagged ActualPower
input_df = add_lagged_ActPower(input_df)
test_df= add_lagged_ActPower(test_df)


y = input_df["target"]

X_raw = input_df[test_df.columns].fillna(0).drop(["TimeStamp_StartFormat"],axis=1)
scaler = StandardScaler()
model = scaler.fit(X_raw)
X_scaled = model.transform(X_raw)

test_raw = test_df.fillna(0).drop(["TimeStamp_StartFormat"],axis=1)
X_test = model.transform(test_raw)


X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.25, random_state=123
)


xgb_model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10
    ,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

# Train the model
xgb_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    
    verbose=False
)

# Evaluate on validation set
val_pred = xgb_model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE: {val_mae:.4f}")


# Make predictions
test_pred = xgb_model.predict(X_test)



model = xgb.XGBRegressor().fit(X_train, y_train)


model.get_booster().feature_names = test_raw.columns.values.tolist()
# Visualize feature importance
xgb.plot_importance(model.get_booster(), max_num_features=20)
#plt.show()

