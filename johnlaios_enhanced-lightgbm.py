# ===========================================
# Wind Speed Prediction Pipeline with LightGBM
# ===========================================

# ---------------------------
# 1. Import Necessary Libraries
# ---------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pywt

# ---------------------------
# 2. Data Loading and Initial Inspection
# ---------------------------

def load_data(train_path: str, test_path: str, timestamp_col: str = 'Timestamp') -> tuple:
    """
    Load train and test datasets, parse the timestamp column.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.
        timestamp_col (str): Name of the timestamp column.

    Returns:
        tuple: (train_data, test_data) as pandas DataFrames.
    """
    train = pd.read_csv(train_path, parse_dates=[timestamp_col])
    test = pd.read_csv(test_path, parse_dates=[timestamp_col])
    train = train.drop(columns=['training'], errors='ignore')

    # Ensure correct datetime format
    for df in [train, test]:
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        for col in df.columns:
            if col != 'Timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')

    
    return train, test

def check_missing_values(df: pd.DataFrame, dataset_name: str):
    """
    Print the count of missing values for each column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        dataset_name (str): Name of the dataset (for printing purposes).
    """
    print(f"Missing values in {dataset_name} before interpolation:")
    print(df.isna().sum())
    print("\n")

# ---------------------------
# 3. Data Preprocessing
# ---------------------------

def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values using linear interpolation.

    Args:
        df (pd.DataFrame): The DataFrame to interpolate.

    Returns:
        pd.DataFrame: DataFrame with interpolated values.
    """
    return df.interpolate(method='linear', limit_direction='both', axis=0)

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time-based features to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame with a 'Timestamp' column.

    Returns:
        pd.DataFrame: DataFrame with added cyclic features.
    """
    df['minute_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.minute / 60)

    df['hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)

    df['day_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.dayofyear / 365)

    df['month_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.month / 12)

    return df

def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sine and cosine transformations for wind direction features.

    Args:
        df (pd.DataFrame): The DataFrame containing wind direction columns.

    Returns:
        pd.DataFrame: DataFrame with added wind direction sine and cosine features.
    """
    wind_cols = ['Wind direction (°)', 'Wind direction (°).1', 'Wind direction (°).2',
                'Wind direction (°).3', 'Wind direction (°).4','Vane position 1+2 (°)','Wind speed (m/s)','Wind speed (m/s).1','Wind speed (m/s).2','Wind speed (m/s).3','Wind speed (m/s).4']

    for col in wind_cols:
        if col in df.columns:
            df[f'{col}_sin'] = np.sin(np.radians(df[col]))
            df[f'{col}_cos'] = np.cos(np.radians(df[col]))
    
    return df

def preprocess_data(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline: datetime parsing, feature engineering.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        target_feature (str): The name of the target variable.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Ensure 'Timestamp' is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Add cyclic time features
    df = add_cyclic_features(df)

    # Add wind direction sine and cosine features
    df = add_wind_direction_features(df)

    return df

# ---------------------------
# 4. Feature Engineering: Filtering
# ---------------------------

def butter_lowpass(cutoff, fs=1.0, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs=1.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    return filtfilt(b, a, data)

def add_low_pass_filter(df: pd.DataFrame, columns: list, cutoff: float, target_feature: str) -> pd.DataFrame:
    """
    Apply a low-pass Butterworth filter to specified columns.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        columns (list): List of column names to apply the filter on.
        cutoff (float): Cutoff frequency for the low-pass filter.
        target_feature (str): Name of the target variable to exclude.

    Returns:
        pd.DataFrame: DataFrame with added low-pass filtered features.
    """
    b, a = butter_lowpass(cutoff)
    numeric_columns = [col for col in columns if col not in ['Timestamp', target_feature]]

    for col in numeric_columns:
        if col in df.columns:
            df[f'{col}_low_pass'] = apply_filter(df[col].values, b, a)
    
    return df
    

def apply_wavelet_transform(df: pd.DataFrame, columns: list, wavelet: str = 'db4', level: int = 3) -> pd.DataFrame:
    """
    Apply wavelet transformation to specified columns, handling length mismatch.

    Args:
        df (pd.DataFrame): The DataFrame to transform.
        columns (list): List of column names to apply wavelet transform on.
        wavelet (str): The wavelet type (e.g., 'db4').
        level (int): Decomposition level.

    Returns:
        pd.DataFrame: DataFrame with wavelet-transformed features.
    """
    new_columns = {}  # Store new features as a dictionary

    for col in columns:
        if col in df.columns:
            # Get the signal from the column
            signal = df[col].fillna(0).values

            # Pad the signal to ensure proper decomposition
            original_length = len(signal)
            pad_length = (2**level - (original_length % (2**level))) % (2**level)
            padded_signal = np.pad(signal, (0, pad_length), mode='constant')

            # Perform wavelet decomposition
            coeffs = pywt.wavedec(padded_signal, wavelet=wavelet, level=level)

            # Reconstruct the signal from approximation coefficients
            reconstructed_signal = pywt.waverec(coeffs[:1] + [None]*(len(coeffs)-1), wavelet)

            # Trim the reconstructed signal to the original length
            reconstructed_signal = reconstructed_signal[:original_length]

            # Store the new feature
            new_columns[f'{col}_wavelet'] = reconstructed_signal

    # Add all new columns to the DataFrame at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df

# ---------------------------
# 5. Model Training and Evaluation
# ---------------------------

def prepare_features(train_df, test_df, target_feature):
    # Exclude non-numeric columns such as 'Timestamp'
    numeric_features = train_df.select_dtypes(include=['number']).columns.tolist()
    features = [col for col in numeric_features if col != target_feature]
    
    # Separate features and target
    X = train_df[features]
    y = train_df[target_feature]
    X_test = test_df[features]
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, scaler



def train_lightgbm(params: dict, X_train, y_train, X_val, y_val) -> lgb.Booster:
    """
    Train a LightGBM model with early stopping.

    Args:
        params (dict): LightGBM parameters.
        X_train: Scaled training features.
        y_train: Training target.
        X_val: Scaled validation features.
        y_val: Validation target.

    Returns:
        lgb.Booster: Trained LightGBM model.
    """
    # Create LightGBM datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # Train the model
    model = lgb.train(
        params,
        train_set,
        num_boost_round=params.get('n_estimators', 9000),
        valid_sets=[train_set, val_set],
        valid_names=['train', 'validation'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(500)
        ],
    )

    return model

def evaluate_model(model: lgb.Booster, X_val, y_val):
    """
    Evaluate the model on the validation set and print metrics.

    Args:
        model (lgb.Booster): Trained LightGBM model.
        X_val: Scaled validation features.
        y_val: Validation target.
    """
    # Predict on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(14, 7))
    plt.plot(y_val.index, y_val, label="Actual", color="blue")
    plt.plot(y_val.index, y_pred, label="Predicted", color="red", alpha=0.7)
    plt.xlabel("Timestamp")
    plt.ylabel('target_feature')
    plt.title("LightGBM - Actual vs Predicted Values on Validation Set")
    plt.legend()
    plt.show()

def make_predictions(model: lgb.Booster, X_test, scaler: MinMaxScaler) -> np.ndarray:
    """
    Make predictions on the test set.

    Args:
        model (lgb.Booster): Trained LightGBM model.
        X_test: Scaled test features.
        scaler (MinMaxScaler): Scaler used for feature scaling.

    Returns:
        np.ndarray: Predicted values.
    """
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return y_test_pred

def save_predictions(predictions: np.ndarray, output_path: str, start_id: int = 2016):
    """
    Save predictions to a CSV file with sequential IDs.

    Args:
        predictions (np.ndarray): Predicted values.
        output_path (str): Path to save the predictions CSV.
        start_id (int): Starting ID value.
    """
    id_values = list(range(start_id, start_id + len(predictions)))
    submission_df = pd.DataFrame({
        'id': id_values,
        'target_feature': predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to '{output_path}'")

# ---------------------------
# 6. Main Execution Pipeline
# ---------------------------

def main():
    # File paths
    train_path = '/kaggle/input/predict-the-wind-speed-at-a-wind-turbine/train.csv'
    test_path = '/kaggle/input/predict-the-wind-speed-at-a-wind-turbine/test.csv'
    sample_submission_path = '/kaggle/input/predict-the-wind-speed-at-a-wind-turbine/sample_submission.csv'

    # Target feature
    target_feature = 'target_feature'

    # Load data
    train_df, test_df = load_data(train_path, test_path)

    # Initial missing value check
    check_missing_values(train_df, 'train_data')
    check_missing_values(test_df, 'test_data')

    # Interpolate missing values
    train_df = interpolate_missing_values(train_df)
    test_df = interpolate_missing_values(test_df)

    # Missing value check after interpolation
    print("After interpolation:")
    check_missing_values(train_df, 'train_data')
    check_missing_values(test_df, 'test_data')

    # Preprocess data
    train_df = preprocess_data(train_df, target_feature)
    test_df = preprocess_data(test_df, target_feature)

    


    # Apply low-pass and band-pass filters
    filter_columns = [
        'Wind speed (m/s)', 'Wind speed (m/s).2', 'Wind direction (°)', 
        'Vane position 1+2 (°)', 'Wind direction (°).2', 'Wind speed (m/s).3',
        'Wind direction (°).3','Wind direction (°).4','Wind speed, Standard deviation (m/s)', 'Wind speed (m/s).3', 'Wind speed (m/s).4','Wind speed (m/s)_cos','Wind speed (m/s)_sin',
        'Wind speed (m/s).1_cos'

        
    ]

    train_df = add_low_pass_filter(
        train_df.copy(), 
        columns=filter_columns, 
        cutoff=0.16, 
        target_feature=target_feature
    )
    

    test_df = add_low_pass_filter(
        test_df.copy(), 
        columns=filter_columns, 
        cutoff=0.16, 
        target_feature=target_feature
    )
    train_df = apply_wavelet_transform(train_df, filter_columns, wavelet='db6', level=3)
    test_df = apply_wavelet_transform(test_df, filter_columns, wavelet='db6', level=3)

    X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, scaler = prepare_features(
    train_df=train_df,
    test_df=test_df,
    target_feature=target_feature
)



    # Define LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 115, 
        'learning_rate': 0.008807141546383906, 
        'min_data_in_leaf': 14,
        'max_depth': 17, 
        'feature_fraction': 0.33881320499801776, 
        'bagging_fraction': 0.916352734185957,
        'bagging_freq': 1,
        'lambda_l1': 0.0521953803412947, 
        'lambda_l2': 0.017128694323983472,
        'verbose': -1,
        'devide':'gpu'
    }

    # Train LightGBM model
    model = train_lightgbm(
        params=lgb_params,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val
    )

    # Evaluate the model
    evaluate_model(model, X_val_scaled, y_val)

    # Make predictions on the test set
    y_test_pred = make_predictions(model, X_test_scaled, scaler)

    # Plot predictions (optional)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_pred, label="Predicted", color="red", alpha=0.7)
    plt.xlabel("Timestamp")
    plt.ylabel(target_feature)
    plt.title("Predicted Values on Test Set")
    plt.legend()
    plt.show()


# ---------------------------
# 7. Execute the Pipeline
# ---------------------------
if __name__ == "__main__":
    main()


