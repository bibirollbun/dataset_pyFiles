# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def feature_engineering(df):
    """
    Create new features based on meteorological understanding and data analysis,
    with 'day' representing day of the year (1-365).
    Ensures no data leakage by avoiding use of the target variable (rainfall).
    """
    # Make a copy to avoid modifying the original dataframe
    enhanced_df = df.copy()
    
    # 1. temparature range (difference between max and min temparatures)
    enhanced_df['temp_range'] = enhanced_df['maxtemp'] - enhanced_df['mintemp']
    
    # 2. Dew point depression (difference between temparature and dew point)
    enhanced_df['dewpoint_depression'] = enhanced_df['temparature'] - enhanced_df['dewpoint']
    
    # 3. Pressure change from previous day
    enhanced_df['pressure_change'] = enhanced_df['pressure'].diff().fillna(0)
    
    # 4. Humidity to dew point ratio
    enhanced_df['humidity_dewpoint_ratio'] = enhanced_df['humidity'] / enhanced_df['dewpoint'].clip(lower=0.1)
    
    # 5. Cloud coverage to sunshine ratio (inverse relationship)
    enhanced_df['cloud_sunshine_ratio'] = enhanced_df['cloud'] / enhanced_df['sunshine'].clip(lower=0.1)
    
    # 6. Wind intensity factor (combination of speed and humidity)
    enhanced_df['wind_humidity_factor'] = enhanced_df['windspeed'] * (enhanced_df['humidity'] / 100)
    
    # 7. temparature-humidity index (simple version of heat index)
    enhanced_df['temp_humidity_index'] = (0.8 * enhanced_df['temparature']) + \
                                        ((enhanced_df['humidity'] / 100) * \
                                        (enhanced_df['temparature'] - 14.3)) + 46.4
    
    # 8. Pressure change rate (acceleration)
    enhanced_df['pressure_acceleration'] = enhanced_df['pressure_change'].diff().fillna(0)
    
    # 9. Seasonal features (based on day of year)
    # Convert day to month (1-365 to 1-12)
    enhanced_df['month'] = ((enhanced_df['day'] - 1) // 30) + 1
    enhanced_df['month'] = enhanced_df['month'].clip(upper=12)  # Ensure month doesn't exceed 12
    
    # 10. Convert day to season (1-365 to 1-4)
    enhanced_df['season'] = ((enhanced_df['month'] - 1) // 3) + 1

    # 11. Sine and cosine transformations to capture cyclical nature of days in a year
    enhanced_df['day_of_year_sin'] = np.sin(2 * np.pi * enhanced_df['day'] / 365)
    enhanced_df['day_of_year_cos'] = np.cos(2 * np.pi * enhanced_df['day'] / 365)
    
    # 12-14. Rolling averages for temperature (3 features)
    enhanced_df['temparature_rolling_3d'] = enhanced_df['temparature'].rolling(window=3, min_periods=1).mean()
    enhanced_df['temparature_rolling_7d'] = enhanced_df['temparature'].rolling(window=7, min_periods=1).mean()
    enhanced_df['temparature_rolling_14d'] = enhanced_df['temparature'].rolling(window=14, min_periods=1).mean()
    
    # 15-17. Rolling averages for pressure (3 features)
    enhanced_df['pressure_rolling_3d'] = enhanced_df['pressure'].rolling(window=3, min_periods=1).mean()
    enhanced_df['pressure_rolling_7d'] = enhanced_df['pressure'].rolling(window=7, min_periods=1).mean()
    enhanced_df['pressure_rolling_14d'] = enhanced_df['pressure'].rolling(window=14, min_periods=1).mean()
    
    # 18-20. Rolling averages for humidity (3 features)
    enhanced_df['humidity_rolling_3d'] = enhanced_df['humidity'].rolling(window=3, min_periods=1).mean()
    enhanced_df['humidity_rolling_7d'] = enhanced_df['humidity'].rolling(window=7, min_periods=1).mean()
    enhanced_df['humidity_rolling_14d'] = enhanced_df['humidity'].rolling(window=14, min_periods=1).mean()
    
    # 21-23. Rolling averages for cloud (3 features)
    enhanced_df['cloud_rolling_3d'] = enhanced_df['cloud'].rolling(window=3, min_periods=1).mean()
    enhanced_df['cloud_rolling_7d'] = enhanced_df['cloud'].rolling(window=7, min_periods=1).mean()
    enhanced_df['cloud_rolling_14d'] = enhanced_df['cloud'].rolling(window=14, min_periods=1).mean()

    # 24-26. Rolling averages for windspeed (3 features)
    enhanced_df['windspeed_rolling_3d'] = enhanced_df['windspeed'].rolling(window=3, min_periods=1).mean()
    enhanced_df['windspeed_rolling_7d'] = enhanced_df['windspeed'].rolling(window=7, min_periods=1).mean()
    enhanced_df['windspeed_rolling_14d'] = enhanced_df['windspeed'].rolling(window=14, min_periods=1).mean()
    
    # 27. Weather pattern change features - temperature trend
    enhanced_df['temp_trend_3d'] = enhanced_df['temparature'].diff(3).fillna(0)
    
    # 28. Pressure trend
    enhanced_df['pressure_trend_3d'] = enhanced_df['pressure'].diff(3).fillna(0)
    
    # 29. Humidity trend
    enhanced_df['humidity_trend_3d'] = enhanced_df['humidity'].diff(3).fillna(0)
    
    # 30. Extreme weather indicators - temperature
    enhanced_df['extreme_temp'] = (enhanced_df['temparature'] > enhanced_df['temparature'].quantile(0.95)) | \
                                 (enhanced_df['temparature'] < enhanced_df['temparature'].quantile(0.05))
    enhanced_df['extreme_temp'] = enhanced_df['extreme_temp'].astype(int)
    
    # 31. Extreme humidity indicator
    enhanced_df['extreme_humidity'] = (enhanced_df['humidity'] > enhanced_df['humidity'].quantile(0.95)) | \
                                     (enhanced_df['humidity'] < enhanced_df['humidity'].quantile(0.05))
    enhanced_df['extreme_humidity'] = enhanced_df['extreme_humidity'].astype(int)
    
    # 32. Extreme pressure indicator
    enhanced_df['extreme_pressure'] = (enhanced_df['pressure'] > enhanced_df['pressure'].quantile(0.95)) | \
                                     (enhanced_df['pressure'] < enhanced_df['pressure'].quantile(0.05))
    enhanced_df['extreme_pressure'] = enhanced_df['extreme_pressure'].astype(int)
    
    # 33. Interaction terms between temperature and humidity
    enhanced_df['temp_humidity_interaction'] = enhanced_df['temparature'] * enhanced_df['humidity']
    
    # 34. Pressure and wind interaction
    enhanced_df['pressure_wind_interaction'] = enhanced_df['pressure'] * enhanced_df['windspeed']
    
    # 35. Cloud and sunshine interaction
    enhanced_df['cloud_sunshine_interaction'] = enhanced_df['cloud'] * enhanced_df['sunshine']
    
    # 36. Dewpoint and humidity interaction
    enhanced_df['dewpoint_humidity_interaction'] = enhanced_df['dewpoint'] * enhanced_df['humidity']

# 37-38. Moving standard deviations for temperature (2 features)
    enhanced_df['temp_std_7d'] = enhanced_df['temparature'].rolling(window=7, min_periods=4).std().fillna(0)
    enhanced_df['temp_std_14d'] = enhanced_df['temparature'].rolling(window=14, min_periods=4).std().fillna(0)
    
    # 39-40. Moving standard deviations for pressure (2 features)
    enhanced_df['pressure_std_7d'] = enhanced_df['pressure'].rolling(window=7, min_periods=4).std().fillna(0)
    enhanced_df['pressure_std_14d'] = enhanced_df['pressure'].rolling(window=14, min_periods=4).std().fillna(0)
    
    # 41-42. Moving standard deviations for humidity (2 features)
    enhanced_df['humidity_std_7d'] = enhanced_df['humidity'].rolling(window=7, min_periods=4).std().fillna(0)
    enhanced_df['humidity_std_14d'] = enhanced_df['humidity'].rolling(window=14, min_periods=4).std().fillna(0)
    
    # 43. Average temperature
    enhanced_df['avg_temp'] = (enhanced_df['maxtemp'] + enhanced_df['mintemp']) / 2
    
    # 44. Temperature deviation from average
    enhanced_df['temp_deviation'] = enhanced_df['temparature'] - enhanced_df['avg_temp']
    
    # 45-46. Hot and cold indicators
    enhanced_df['is_hot'] = (enhanced_df['temparature'] > enhanced_df['temparature'].quantile(0.75)).astype(int)
    enhanced_df['is_cold'] = (enhanced_df['temparature'] < enhanced_df['temparature'].quantile(0.25)).astype(int)
    
    # 47. Temperature volatility
    enhanced_df['temp_volatility'] = abs(enhanced_df['temparature'] - enhanced_df['avg_temp'])
    
    # 48. Dewpoint depression squared
    enhanced_df['dewpoint_depression_squared'] = enhanced_df['dewpoint_depression'] ** 2
    
    # 49. Approximate relative humidity from dewpoint
    enhanced_df['approx_relative_humidity'] = 100 - 5 * enhanced_df['dewpoint_depression']
    enhanced_df['approx_relative_humidity'] = enhanced_df['approx_relative_humidity'].clip(0, 100)
    
    # 50. Precipitation probability based on dewpoint
    enhanced_df['precip_probability'] = 1 / (1 + np.exp(-(10 - enhanced_df['dewpoint_depression'])))

    # 51-52. Wind direction transformation (if available)
    if 'winddirection' in enhanced_df.columns:
        enhanced_df['wind_dir_sin'] = np.sin(np.deg2rad(enhanced_df['winddirection']))
        enhanced_df['wind_dir_cos'] = np.cos(np.deg2rad(enhanced_df['winddirection']))
    else:
        enhanced_df['wind_dir_sin'] = 0
        enhanced_df['wind_dir_cos'] = 0
    
    # 53-54. Additional wind direction harmonics
    enhanced_df['wind_dir_sin_2x'] = np.sin(2 * np.deg2rad(enhanced_df['winddirection'])) if 'winddirection' in enhanced_df.columns else 0
    enhanced_df['wind_dir_cos_2x'] = np.cos(2 * np.deg2rad(enhanced_df['winddirection'])) if 'winddirection' in enhanced_df.columns else 0
    
    # 55. Pressure trend indicator
    enhanced_df['pressure_trend'] = enhanced_df['pressure_diff1'] = enhanced_df['pressure'].diff(1).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # 56-57. Pressure extremes
    enhanced_df['pressure_high'] = (enhanced_df['pressure'] > enhanced_df['pressure'].quantile(0.75)).astype(int)
    enhanced_df['pressure_low'] = (enhanced_df['pressure'] < enhanced_df['pressure'].quantile(0.25)).astype(int)
    
    # 58. Second-order pressure change
    enhanced_df['pressure_diff2'] = enhanced_df['pressure_diff1'].diff(1).fillna(0)
    
    # 59-63. Cloud interaction terms
    enhanced_df['cloud_humidity_mul'] = enhanced_df['cloud'] * enhanced_df['humidity']
    enhanced_df['cloud_pressure_mul'] = enhanced_df['cloud'] * enhanced_df['pressure']
    enhanced_df['cloud_dewpoint_mul'] = enhanced_df['cloud'] * enhanced_df['dewpoint']
    enhanced_df['cloud_maxtemp_mul'] = enhanced_df['cloud'] * enhanced_df['maxtemp']
    enhanced_df['cloud_temparature_mul'] = enhanced_df['cloud'] * enhanced_df['temparature']

# 64-68. Squared interaction terms
    enhanced_df['cloud_humidity_squared'] = enhanced_df['cloud'] * (enhanced_df['humidity'] ** 2)
    enhanced_df['cloud_pressure_squared'] = enhanced_df['cloud'] * (enhanced_df['pressure'] ** 2)
    enhanced_df['cloud_dewpoint_squared'] = enhanced_df['cloud'] * (enhanced_df['dewpoint'] ** 2)
    enhanced_df['cloud_maxtemp_squared'] = enhanced_df['cloud'] * (enhanced_df['maxtemp'] ** 2)
    enhanced_df['cloud_temparature_squared'] = enhanced_df['cloud'] * (enhanced_df['temparature'] ** 2)
    
    # 69. Composite rain signal
    rain_predictors = [
        enhanced_df['humidity'] / 100,
        enhanced_df['cloud'] / 100,
        1 - enhanced_df['dewpoint_depression'] / 10,  # Inverse relationship
        (enhanced_df['pressure_diff1'] < 0).astype(float) * 0.5  # Falling pressure
    ]
    rain_signals = np.mean(rain_predictors, axis=0)
    enhanced_df['composite_rain_signal'] = rain_signals
    
    # 70. Rain probability index
    enhanced_df['rain_probability_index'] = 1 / (1 + np.exp(-10 * (rain_signals - 0.5)))
    
    # 71-74. Temperature shifts and diffs
    enhanced_df['temparature_shift1'] = enhanced_df['temparature'].shift(1).fillna(enhanced_df['temparature'].mean())
    enhanced_df['temparature_shift2'] = enhanced_df['temparature'].shift(2).fillna(enhanced_df['temparature'].mean())
    enhanced_df['temparature_diff1'] = enhanced_df['temparature'].diff(1).fillna(0)
    enhanced_df['temparature_diff2'] = enhanced_df['temparature'].diff(2).fillna(0)
    
    # 75-78. Pressure shifts and diffs
    enhanced_df['pressure_shift1'] = enhanced_df['pressure'].shift(1).fillna(enhanced_df['pressure'].mean())
    enhanced_df['pressure_shift2'] = enhanced_df['pressure'].shift(2).fillna(enhanced_df['pressure'].mean())
    enhanced_df['pressure_diff1'] = enhanced_df['pressure'].diff(1).fillna(0)
    enhanced_df['pressure_diff2'] = enhanced_df['pressure'].diff(2).fillna(0)

    # 79-82. Humidity shifts and diffs
    enhanced_df['humidity_shift1'] = enhanced_df['humidity'].shift(1).fillna(enhanced_df['humidity'].mean())
    enhanced_df['humidity_shift2'] = enhanced_df['humidity'].shift(2).fillna(enhanced_df['humidity'].mean())
    enhanced_df['humidity_diff1'] = enhanced_df['humidity'].diff(1).fillna(0)
    enhanced_df['humidity_diff2'] = enhanced_df['humidity'].diff(2).fillna(0)
    
    # 83-86. Cloud shifts and diffs
    enhanced_df['cloud_shift1'] = enhanced_df['cloud'].shift(1).fillna(enhanced_df['cloud'].mean())
    enhanced_df['cloud_shift2'] = enhanced_df['cloud'].shift(2).fillna(enhanced_df['cloud'].mean())
    enhanced_df['cloud_diff1'] = enhanced_df['cloud'].diff(1).fillna(0)
    enhanced_df['cloud_diff2'] = enhanced_df['cloud'].diff(2).fillna(0)
    
    # 87-90. Windspeed shifts and diffs
    enhanced_df['windspeed_shift1'] = enhanced_df['windspeed'].shift(1).fillna(enhanced_df['windspeed'].mean())
    enhanced_df['windspeed_shift2'] = enhanced_df['windspeed'].shift(2).fillna(enhanced_df['windspeed'].mean())
    enhanced_df['windspeed_diff1'] = enhanced_df['windspeed'].diff(1).fillna(0)
    enhanced_df['windspeed_diff2'] = enhanced_df['windspeed'].diff(2).fillna(0)
    
    # 91-92. Detrended temperature sinusoidal components (if day is available)
    if 'day_of_year_sin' in enhanced_df.columns:
        temp_mean = enhanced_df['temparature'].mean()
        enhanced_df['temp_detrended_sin'] = (enhanced_df['temparature'] - temp_mean) * enhanced_df['day_of_year_sin']
        enhanced_df['temp_detrended_cos'] = (enhanced_df['temparature'] - temp_mean) * enhanced_df['day_of_year_cos']
    else:
        enhanced_df['temp_detrended_sin'] = 0
        enhanced_df['temp_detrended_cos'] = 0
    
    # 93-94. Semi-annual temperature cycle
    enhanced_df['temp_semiannual_sin'] = np.sin(4 * np.pi * enhanced_df['day'] / 365) if 'day' in enhanced_df.columns else 0
    enhanced_df['temp_semiannual_cos'] = np.cos(4 * np.pi * enhanced_df['day'] / 365) if 'day' in enhanced_df.columns else 0
    
    # 95. Week of year (if day is available)
    enhanced_df['week_of_year'] = (enhanced_df['day'] // 7) + 1 if 'day' in enhanced_df.columns else 0
    
    # 96-97. Weekly cyclical features
    enhanced_df['week_sin'] = np.sin(2 * np.pi * enhanced_df['week_of_year'] / 52) if 'week_of_year' in enhanced_df.columns else 0
    enhanced_df['week_cos'] = np.cos(2 * np.pi * enhanced_df['week_of_year'] / 52) if 'week_of_year' in enhanced_df.columns else 0
    
    # Fill any NaN values
    enhanced_df = enhanced_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return enhanced_df


# Function to select top features
def select_top_features(X_train, y_train, X_test, num_features=19, method='mutual_info'):
    print(f"Selecting top {num_features} features using {method}...")
    
    # Create copies
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    if method == 'mutual_info':
        # Use Mutual Information for feature selection
        selector = SelectKBest(mutual_info_classif, k=num_features)
        selector.fit(X_train_copy, y_train)
        
        # Get mask of selected features
        selected_mask = selector.get_support()
        selected_features = X_train_copy.columns[selected_mask]
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Print selected features
    print(f"Selected features: {', '.join(selected_features)}")
    print(f"Total features selected: {len(selected_features)}")
    
    # Apply selection to train and test data
    X_train_selected = X_train_copy[selected_features]
    X_test_selected = X_test_copy[selected_features]
    
    return X_train_selected, X_test_selected, selected_features

# Function to evaluate model
def evaluate_model(model, X_val, y_val):
    # Get predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_proba)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc_score
    }


# Define purged cross-validation function
def purged_cross_validation(X, y, model, n_splits=5, purge_length=1):
    # Create TimeSeriesSplit
    ts_split = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    fold = 1
    
    for train_idx, val_idx in ts_split.split(X):
        # Apply purging to avoid leakage
        val_idx = val_idx[val_idx >= train_idx[-purge_length]]
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Convert pandas DataFrames to NumPy arrays for better compatibility
        X_train_array = np.ascontiguousarray(X_train.values)
        X_val_array = np.ascontiguousarray(X_val.values)
        y_train_array = np.ascontiguousarray(y_train.values)
        y_val_array = np.ascontiguousarray(y_val.values)
        
        # Train model using arrays
        model.fit(X_train_array, y_train_array)
        
        # Evaluate using arrays
        fold_scores = evaluate_model(model, X_val_array, y_val_array)
        fold_scores['fold'] = fold
        cv_scores.append(fold_scores)
        
        print(f"Fold {fold} - Accuracy: {fold_scores['accuracy']:.4f}, "
              f"F1 Score: {fold_scores['f1_score']:.4f}, AUC: {fold_scores['auc']:.4f}")
        
        fold += 1
    
    # Calculate average scores
    avg_scores = {
        'accuracy': np.mean([score['accuracy'] for score in cv_scores]),
        'f1_score': np.mean([score['f1_score'] for score in cv_scores]),
        'auc': np.mean([score['auc'] for score in cv_scores])
    }
    
    print(f"Average - Accuracy: {avg_scores['accuracy']:.4f}, "
          f"F1 Score: {avg_scores['f1_score']:.4f}, AUC: {avg_scores['auc']:.4f}")
    
    return cv_scores, avg_scores

# Function to evaluate model on known test samples
def evaluate_on_known_test(model, X_test, known_test_df, test_ids):
    """Evaluate model performance on known test samples"""
    print("\n=== Evaluating model on known test samples ===")
    
    # Get predictions for test data
    if isinstance(X_test, pd.DataFrame):
        X_test_array = np.ascontiguousarray(X_test.values)
    else:
        X_test_array = X_test
        
    test_predictions = model.predict_proba(X_test_array)[:, 1]
    
    # Create a dataframe with predictions
    predictions_df = pd.DataFrame({
        'id': test_ids,
        'target': test_predictions
    })
    
    # Extract predictions for known test samples
    known_predictions = predictions_df[predictions_df['id'].isin(known_test_df['id'])]
    
    # Merge with known test values
    evaluation_df = pd.merge(known_predictions, known_test_df, on='id')
    
    # Calculate AUC
    known_auc = roc_auc_score(evaluation_df['rainfall'], evaluation_df['target'])
    print(f"AUC Score on known samples: {known_auc:.4f}")
    
    # Convert probabilities to binary predictions with 0.5 threshold
    evaluation_df['predicted_class'] = (evaluation_df['target'] >= 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(evaluation_df['rainfall'], evaluation_df['predicted_class'])
    
    # Get metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return known_auc, predictions_df


# Main function
def main():
    # Load and preprocess the data
    print("Loading datasets...")
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
    known_test_df = pd.read_csv('/kaggle/input/146-rows/test_146.csv')  # Load known test values
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Known test shape: {known_test_df.shape}")
    
    # Extract IDs before preprocessing
    test_ids = test_df['id'].copy()
    
    # Drop ID column for processing (but keep a copy of IDs)
    train_df.drop(columns=['id'], inplace=True, errors='ignore')
    test_df.drop(columns=['id'], inplace=True, errors='ignore')
    
    # Fill missing values
    train_df = train_df.fillna(train_df.mean())
    test_df = test_df.fillna(test_df.mean())
    
    # Apply feature engineering
    print("Applying feature engineering...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Separate features and target
    y = train_df['rainfall']
    X = train_df.drop(columns=['rainfall'])
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    
    # Set default number of features
    num_features = 19
    
    # Feature selection
    print("Performing feature selection...")
    X_selected, test_selected, selected_features = select_top_features(
        X_scaled, y, test_scaled, num_features=num_features, method='mutual_info'
    )
    # Define the MLP model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.0075,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=15,
        random_state=SEED,
        verbose=False,
        warm_start=True
    )
    
    print("\nTraining MLP model...")
    
    # Convert to NumPy arrays for cross-validation
    X_selected_array = np.ascontiguousarray(X_selected.values)
    y_array = np.ascontiguousarray(y.values)
        
    # Run cross-validation
    _, avg_scores = purged_cross_validation(X_selected, y, mlp_model)
    
    # Train on full dataset
    print("\nTraining model on full dataset...")
    mlp_model.fit(X_selected_array, y_array)
    
    # Evaluate on known test samples
    known_auc, predictions_df = evaluate_on_known_test(
        mlp_model, test_selected, known_test_df, test_ids
    )
    
    # Create submission file
    submission_path = "submission.csv"
    predictions_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    
    # Display summary information
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE SUMMARY:")
    print("=" * 50)
    print(f"Number of features used: {num_features}")
    print(f"Cross-validation AUC: {avg_scores['auc']:.6f}")
    print(f"Cross-validation Accuracy: {avg_scores['accuracy']:.6f}")
    print(f"Cross-validation F1 Score: {avg_scores['f1_score']:.6f}")
    print("-" * 50)
    print(f"AUC on known test data: {known_auc:.6f}")
    print("=" * 50)
    print(f"Selected features: {', '.join(selected_features)}")
    print("=" * 50)
    
    return predictions_df

# Execute the script
if __name__ == "__main__":
    print("=" * 80)
    print("Enhanced Rainfall Prediction with MLP Model")
    print("=" * 80)
    submission = main()

