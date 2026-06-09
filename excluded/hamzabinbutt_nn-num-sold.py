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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# File paths in the Kaggle environment
train_file_path = '/kaggle/input/playground-series-s5e1/train.csv'
# Read the train dataset
train_data = pd.read_csv(train_file_path)
# Count the number of missing (NA) values in each column
na_counts = train_data.isna().sum()
# Calculate the percentage of missing values for each column
na_percentages = (na_counts / len(train_data)) * 100
# Remove rows with missing values
train_data = train_data.dropna()
# Drop the 'id' column
train_data = train_data.drop(columns=['id'])
# Ensure 'date' is in datetime format
train_data['date'] = pd.to_datetime(train_data['date'])
# Create new features from the 'date' column
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['day_of_week'] = train_data['date'].dt.dayofweek  # Monday=0, Sunday=6
train_data['week_of_year'] = train_data['date'].dt.isocalendar().week
train_data['quarter'] = train_data['date'].dt.quarter  # 1 = Jan-Mar, 2 = Apr-Jun, etc.
# Drop the original 'date' column if it's no longer needed
train_data = train_data.drop(columns=['date'])
from sklearn.preprocessing import LabelEncoder
#Initialize LabelEncoder
label_encoder = LabelEncoder()
# Identify categorical columns
categorical_columns = train_data.select_dtypes(include=['object']).columns
# Apply LabelEncoder to each categorical column
for column in categorical_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column])
# Display the updated dataframe
train_data.head()


# Apply the natural log transformation to the `num_sold` column
train_data['log_num_sold'] = np.log(train_data['num_sold'])

# Plot histograms for both the original and transformed data
plt.figure(figsize=(12, 6))

# Histogram for `num_sold`
plt.subplot(1, 2, 1)
plt.hist(train_data['num_sold'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of num_sold (Original Scale)')
plt.xlabel('num_sold')
plt.ylabel('Frequency')

# Histogram for `log_num_sold`
plt.subplot(1, 2, 2)
plt.hist(train_data['log_num_sold'], bins=30, color='orange', edgecolor='black')
plt.title('Histogram of log_num_sold (Log Scale)')
plt.xlabel('log_num_sold')
plt.ylabel('Frequency')

# Adjust layout and display
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
# Assume 'train_data' contains the 'num_sold' column (raw variable)
original_data = np.exp(train_data['log_num_sold'])  # Convert log back to original if necessary
# Define transformations
transformed_data = {
    "Original (num_sold)": original_data,
    "Logarithm Transformation": np.log(original_data + 1),  # +1 to handle zeroes
    "Square Root Transformation": np.sqrt(original_data),
    "Reciprocal Transformation": 1 / (original_data + 1),  # Avoid division by zero
    "Box-Cox Transformation": boxcox(original_data + 1)[0]  # Box-Cox requires strictly positive values
}
# Plot histograms for each transformation
plt.figure(figsize=(15, 10))
for i, (key, data) in enumerate(transformed_data.items(), 1):
    plt.subplot(3, 2, i)
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(key)
    plt.xlabel('Transformed Values')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Define your features (independent variables) and target (dependent variable)
X = train_data.drop(columns=['log_num_sold','num_sold'])
y = train_data['log_num_sold']
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Custom MAPE loss function
def mape_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100
# Build the neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1)  # Single output for regression
    ])
# Compile the model
model.compile(optimizer='adam',loss=mape_loss,metrics=[tf.keras.metrics.MeanAbsolutePercentageError(name="mape")])
# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
# Train the model
history = model.fit(X_train, y_train,validation_split=0.2,epochs=50,batch_size=32,verbose=1,callbacks=[early_stopping])
# Predict on the test set
y_pred = model.predict(X_test).flatten()
# Calculate additional metrics
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# Print evaluation metrics
print(f"Mean Absolute Percentage Error (MAPE): {mape:.3f}")
# Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MAPE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MAPE)')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MAPE Loss (%)')
plt.legend()
# Plot MSE, MAE, MAPE metrics
plt.subplot(1, 2, 2)
plt.plot(history.history['mape'], label='Training MAPE')
plt.plot(history.history['val_mape'], label='Validation MAPE')
plt.title('Metrics Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.tight_layout()
plt.show()


train_data.head()


from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Separate features and target
X = train_data.drop(columns=['num_sold','log_num_sold'])
y = train_data['num_sold']
# Apply Box-Cox Transformation to the dependent variable
# Note: Box-Cox only works for strictly positive data, ensure no zero values
y_boxcox, fitted_lambda = boxcox(y + 1)  # Add 1 to avoid issues with log(0)
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_boxcox, test_size=0.2, random_state=42)
# Define custom MAPE loss function
def mape_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100
# Build the neural network
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
# EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)
# Predict on the test set
y_pred = model.predict(X_test).flatten()
# Inverse Box-Cox Transformation on Predictions
y_pred_original = np.exp(np.log(y_pred * fitted_lambda + 1) / fitted_lambda - 1)
y_test_original = np.exp(np.log(y_test * fitted_lambda + 1) / fitted_lambda - 1)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.3f}%")
# Plot training and validation losses
plt.figure(figsize=(12, 4))
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MAPE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MAPE)')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MAPE Loss (%)')
plt.legend()
# Plot Metrics
plt.subplot(1, 2, 2)
plt.plot(history.history['mape'], label='Training MAPE')
plt.plot(history.history['val_mape'], label='Validation MAPE')
plt.title('Training and Validation Metrics Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metrics (%)')
plt.legend()
plt.tight_layout()
plt.show()

