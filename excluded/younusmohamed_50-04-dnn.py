# Check GPU availability
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))


!pip install scikeras --quiet


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, time

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Install scikeras if needed
!pip install scikeras --quiet
from scikeras.wrappers import KerasClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline


# Define memory optimization function
def reduce_memory_usage(df, verbose=True):
    """Downcasts numeric columns to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


# Load and optimize data
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

print("Before memory optimization:")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

train = reduce_memory_usage(train)
test = reduce_memory_usage(test)

# Separate features and target from training data
X = train.drop(columns=['id', 'rainfall'])
y = train['rainfall']

# For test data, drop the id column and save the IDs for submission
X_test = test.drop(columns=['id'])
test_ids = test['id']

print("After optimization:")
print("X shape:", X.shape)
print("X_test shape:", X_test.shape)


def advanced_features(df, is_train=False, target_series=None):
    """
    Create advanced features.
    Assumes:
      - 'day' column exists.
      - For training data (is_train=True), target_series (rainfall) is provided.
    """
    df = df.copy()
    
    # Build date: day=1 corresponds to 2024-01-01
    base_date = pd.to_datetime('2024-01-01')
    df['date'] = base_date + pd.to_timedelta(df['day'] - 1, unit='D')
    
    # Extract month (crucial for consistency)
    df['month'] = df['date'].dt.month
    
    # Date-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    
    # Periodic features
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Temperature features
    if 'maxtemp' in df.columns and 'mintemp' in df.columns:
        df['temp_range'] = df['maxtemp'] - df['mintemp']
    if 'temparature' in df.columns and 'dewpoint' in df.columns:
        df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    
    # Interaction/Ratio features
    if 'humidity' in df.columns and 'cloud' in df.columns:
        df['humidity_cloud_ratio'] = df['humidity'] / (df['cloud'] + 1e-3)
    if 'sunshine' in df.columns and 'cloud' in df.columns:
        df['sunshine_cloud_ratio'] = df['sunshine'] / (df['cloud'] + 1e-3)
    if 'pressure' in df.columns and 'winddirection' in df.columns:
        df['pressure_wind_interaction'] = df['pressure'] * df['winddirection']
    if 'temparature' in df.columns and 'pressure' in df.columns:
        df['temp_pressure_ratio'] = df['temparature'] / (df['pressure'] + 1e-3)
    if 'windspeed' in df.columns and 'pressure' in df.columns:
        df['wind_pressure_ratio'] = df['windspeed'] / (df['pressure'] + 1e-3)
    
    # Lag features (for training data)
    # if is_train:
    #     if target_series is not None:
    #         df['rainfall'] = target_series.values
    #     df = df.sort_values('date').reset_index(drop=True)
    #     df['rain_prev_day'] = df['rainfall'].shift(1).fillna(0)
    #     df['rain_next_day'] = df['rainfall'].shift(-1).fillna(0)
    #     df['gap_before_rain'] = df.groupby((df['rain_prev_day'] != df['rainfall']).cumsum()).cumcount()
    #     df['gap_after_rain'] = df[::-1].groupby((df['rain_next_day'] != df['rainfall']).cumsum()).cumcount()
    #     df.drop(['rain_prev_day', 'rain_next_day'], axis=1, inplace=True)
    # else:
    #     # For test data, set gap features to 0
    #     df['gap_before_rain'] = 0
    #     df['gap_after_rain'] = 0
    
    # Drop the date column to avoid leakage
    df.drop(['date'], axis=1, inplace=True, errors='ignore')
    
    return df

# Apply advanced features to train and test
X = advanced_features(pd.concat([X, y], axis=1), is_train=True, target_series=y)
y = X.pop('rainfall')
X_test = advanced_features(X_test, is_train=False)

print("Enhanced Train Shape:", X.shape)
print("Enhanced Test Shape:", X_test.shape)


# Define preprocessing pipelines
preprocessors = {
    'standard': StandardScaler(),
    # 'minmax': MinMaxScaler(),
    # 'robust': RobustScaler(),
    # 'raw': None
}


# Define the expanded Keras model function
def create_dnn_model(optimizer='adam', dropout_rate=0.2, learning_rate=0.001,
                     num_layers=3, units1=64, units2=32, units3=16):
    model = Sequential()
    # First hidden layer using units1
    model.add(Dense(units1, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(dropout_rate))
    
    # Second hidden layer using units2 (if at least 2 layers)
    if num_layers >= 2:
        model.add(Dense(units2, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Third hidden layer using units3 (if at least 3 layers)
    if num_layers >= 3:
        model.add(Dense(units3, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['AUC'])
    return model


# Wrap the Keras model and define an expanded hyperparameter grid
dnn_wrapper = KerasClassifier(
    model=create_dnn_model,
    epochs=20,           # Default epochs; to be tuned
    batch_size=32,       # Default batch size; to be tuned
    verbose=0
)

# Expanded hyperparameter grid
param_dist = {
    'dnn__model__optimizer': ['adam', 'rmsprop'],
    'dnn__model__dropout_rate': [0.2, 0.3, 0.4],
    'dnn__model__learning_rate': [1e-3, 1e-4, 5e-4],
    'dnn__epochs': [20, 30, 50],
    'dnn__batch_size': [16, 32, 64],
    'dnn__model__num_layers': [2, 3, 4],
    'dnn__model__units1': [32, 64, 128],
    'dnn__model__units2': [32, 64, 128],
    'dnn__model__units3': [16, 32, 64]
}


# Run hyperparameter tuning and generate submissions for all preprocessing methods
from joblib import parallel_backend

results = []  # List to store tuning results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs("submissions", exist_ok=True)

for prep_name, scaler in preprocessors.items():
    print(f"\n--- Preprocessing: {prep_name} ---")
    
    # Build pipeline: imputer -> (scaler if provided) -> DNN
    steps = []
    steps.append(('imputer', SimpleImputer(strategy="median")))
    if scaler is not None:
        steps.append((prep_name, scaler))
    steps.append(('dnn', dnn_wrapper))
    pipeline = Pipeline(steps)
    
    with parallel_backend('threading', n_jobs=-1):
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=100,  # Increase iterations for a more thorough search
            scoring='roc_auc',
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X, y)
    
    best_score = search.best_score_
    best_params = search.best_params_
    
    print(f"Best CV ROC AUC for {prep_name}: {best_score:.4f}")
    print("Best parameters:", best_params)
    
    # Re-fit best estimator on the full training data
    best_estimator = search.best_estimator_
    best_estimator.fit(X, y)
    
    # Manually transform X_test using all steps except the final DNN step, then predict probabilities.
    X_test_transformed = best_estimator[:-1].transform(X_test)
    test_preds = best_estimator.named_steps['dnn'].predict_proba(X_test_transformed)[:, 1]
    
    # Save the submission file for this preprocessing method
    sub_filename = f"submissions/{prep_name}_dnn_submission.csv"
    sub_df = pd.DataFrame({'id': test_ids, 'rainfall': test_preds})
    sub_df.to_csv(sub_filename, index=False)
    print(f"Submission file saved: {sub_filename}\n")
    
    results.append({
        'preprocessing': prep_name,
        'best_auc': best_score,
        'best_params': best_params,
        'submission_file': sub_filename
    })


# Create a bar plot to compare best CV ROC AUC scores
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='best_auc', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='preprocessing', y='best_auc', data=results_df, palette='viridis')
plt.title("Best CV ROC AUC by Preprocessing Method")
plt.xlabel("Preprocessing Method")
plt.ylabel("Best CV ROC AUC")
plt.ylim(0.0, 1.0)
plt.show()




