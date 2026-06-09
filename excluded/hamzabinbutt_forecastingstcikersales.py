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


# Core libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn (ML and preprocessing)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

# TensorFlow & Keras (Deep learning)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2



# Load and preprocess the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
# Remove the 'id' column in-place
train_data.drop(columns=['id'], inplace=True)
# Display the updated DataFrame
train_data.head()


# Calculate the percentage of missing values for each column
nan_percentage = train_data.isna().sum() / len(train_data) * 100
# Display the percentages
nan_percentage


# Remove rows with NaN values
train_data = train_data.dropna()
# Display the first few rows of the updated dataset
train_data.head()


import numpy as np

# Ensure 'date' column is in datetime format
train_data['date'] = pd.to_datetime(train_data['date'])

# Extract basic date-related features
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['day_of_week'] = train_data['date'].dt.dayofweek  # 0=Monday, 6=Sunday
train_data['week_of_year'] = train_data['date'].dt.isocalendar().week

# Create cyclic features using sine and cosine transformations
train_data['month_sin'] = np.sin(2 * np.pi * train_data['month'] / 12)  # 12 months in a year
train_data['month_cos'] = np.cos(2 * np.pi * train_data['month'] / 12)

train_data['day_sin'] = np.sin(2 * np.pi * train_data['day'] / 31)  # Assume max 31 days in a month
train_data['day_cos'] = np.cos(2 * np.pi * train_data['day'] / 31)

train_data['day_of_week_sin'] = np.sin(2 * np.pi * train_data['day_of_week'] / 7)  # 7 days in a week
train_data['day_of_week_cos'] = np.cos(2 * np.pi * train_data['day_of_week'] / 7)

train_data['week_of_year_sin'] = np.sin(2 * np.pi * train_data['week_of_year'] / 52)  # 52 weeks in a year
train_data['week_of_year_cos'] = np.cos(2 * np.pi * train_data['week_of_year'] / 52)
# Drop the 'date' column if no longer needed
train_data.drop(columns=['date'], inplace=True)
# Display the first few rows with new features
train_data.head()


#Initialize LabelEncoder
label_encoder = LabelEncoder()
# Identify categorical columns
categorical_columns = train_data.select_dtypes(include=['object']).columns
# Apply LabelEncoder to each categorical column
for column in categorical_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column])
train_data.head()


# Apply ln transformation on the `num_sold` column
train_data['log_num_sold'] = np.log(train_data['num_sold'])
# Display the updated dataframe
train_data.head()


# Define your features (independent variables) and target (dependent variable)
X = train_data.drop(columns=['log_num_sold', 'num_sold'])
y = train_data['log_num_sold']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Custom MAPE loss function
def mape_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store evaluation metrics across folds
mape_scores = []
history_list = []

# Run cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_scaled), 1):
    print(f"Training fold {fold}...")
    
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Build the neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)  # Single output for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss=mape_loss, metrics=[tf.keras.metrics.MeanAbsolutePercentageError(name="mape")])

    # Define EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=32,
                        verbose=1,
                        callbacks=[early_stopping])

    # Append history
    history_list.append(history)

    # Predict on validation set
    y_pred = model.predict(X_val).flatten()
    
    # Calculate MAPE for this fold
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    mape_scores.append(mape)

    print(f"MAPE for fold {fold}: {mape:.3f}%")

# Mean MAPE across all folds
print(f"\nMean MAPE across 5 folds: {np.mean(mape_scores):.3f}%")

# Plot training history for all folds
def plot_history(history_list):
    """Plots training and validation MAPE for all folds."""
    plt.figure(figsize=(12, 6))
    for history in history_list:
        plt.plot(history.history['loss'], label='Train Loss', alpha=0.5)
        plt.plot(history.history['val_loss'], label='Val Loss', alpha=0.5)
    
    plt.title('Training and Validation Loss Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAPE)')
    plt.legend(loc='best')
    plt.show()

# Plot training history across folds
plot_history(history_list)


