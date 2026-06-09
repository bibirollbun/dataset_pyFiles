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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs before modifying test_data
test_ids = test_data['id']

# Handle missing values - Fill NaN with median (for numerical) or mode (for categorical)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Preprocess the data
X = train_data.drop(columns=['rainfall'])  # Features
y = train_data['rainfall']  # Target variable

# One-hot encoding for categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure feature consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Reshape data for CNN (we're treating it as a 1D sequence)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # 3D shape (samples, features, 1)
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))  # 3D shape
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))  # 3D shape for test

# Build the CNN model
model = Sequential()

# 1D Convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

# Add more layers if needed
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Flatten the output to connect it to fully connected layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (Rain or No Rain)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Validate the model
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1 (as integers)

# Calculate accuracy
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

# Predict on the test set
test_predictions = model.predict(test_data)
test_predictions = (test_predictions > 0.5).astype(int)  # Convert probabilities to 0 or 1 (as integers)

# Ensure ID and Predictions have the same length
assert len(test_ids) == len(test_predictions), f"Mismatch: {len(test_ids)} IDs vs {len(test_predictions)} predictions"

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,  # Ensure 'id' column matches expected format
    'RainTomorrow': test_predictions.flatten()  # Flatten in case it's a 2D array
})
submission.to_csv('submission_final_cnn2.csv', index=False)

print("Submission file saved as 'submission_final_cnn.csv'")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs before modifying test_data
test_ids = test_data['id']

# Handle missing values - Fill NaN with median (for numerical) or mode (for categorical)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Preprocess the data
X = train_data.drop(columns=['rainfall'])  # Features
y = train_data['rainfall']  # Target variable

# One-hot encoding for categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure feature consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Reshape data for CNN (we're treating it as a 1D sequence)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # 3D shape (samples, features, 1)
X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))  # 3D shape
test_data_cnn = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))  # 3D shape for test

# Build the CNN model
def build_cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))  # Dropout layer to avoid overfitting
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))  # Another dropout layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (Rain or No Rain)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the CNN model
cnn_model = build_cnn_model()
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_data=(X_val_cnn, y_val), callbacks=[early_stopping])

# Validate the CNN model
y_pred_cnn = cnn_model.predict(X_val_cnn)
y_pred_cnn = (y_pred_cnn > 0.5).astype(int)

# Calculate accuracy for CNN
cnn_accuracy = accuracy_score(y_val, y_pred_cnn)
print(f'CNN Validation Accuracy: {cnn_accuracy}')

# Flatten the CNN-preprocessed data for RandomForestClassifier
X_train_rf = X_train.reshape(X_train.shape[0], -1)  # Flatten to 2D (samples, features)
X_val_rf = X_val.reshape(X_val.shape[0], -1)  # Flatten to 2D
test_data_rf = test_data.reshape(test_data.shape[0], -1)  # Flatten to 2D for test data

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf, y_train)

# Predict using Random Forest
rf_predictions = rf_model.predict(X_val_rf)
print(f'Random Forest Validation Accuracy: {accuracy_score(y_val, rf_predictions)}')

# Predict on the test set with Random Forest
rf_test_predictions = rf_model.predict(test_data_rf)

# Ensemble Model: Average the predictions from CNN and Random Forest
# (Taking the majority vote between the models)
test_pred_cnn = cnn_model.predict(test_data_cnn)
test_pred_cnn = (test_pred_cnn > 0.5).astype(int)

# Ensemble Test Predictions (majority vote)
test_pred_ensemble = (test_pred_cnn + rf_test_predictions.reshape(-1, 1)) // 2

# Ensure ID and Predictions have the same length
assert len(test_ids) == len(test_pred_ensemble), f"Mismatch: {len(test_ids)} IDs vs {len(test_pred_ensemble)} predictions"

# Create submission file for the Ensemble model
submission_ensemble = pd.DataFrame({
    'id': test_ids,  # Ensure 'id' column matches expected format
    'RainTomorrow': test_pred_ensemble.flatten()  # Flatten to 1D array
})
submission_ensemble.to_csv('submission_final_cnn_rf_ensemble.csv', index=False)

print("Submission file saved as 'submission_final_cnn_rf_ensemble.csv'")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs before modifying test_data
test_ids = test_data['id']

# Handle missing values - Fill NaN with median (for numerical) or mode (for categorical)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Preprocess the data
X = train_data.drop(columns=['rainfall'])  # Features
y = train_data['rainfall']  # Target variable

# One-hot encoding for categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure feature consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Reshape data for CNN (we're treating it as a 1D sequence)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # 3D shape (samples, features, 1)
X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))  # 3D shape
test_data_cnn = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))  # 3D shape for test

# Build the CNN model
def build_cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))  # Dropout layer to avoid overfitting
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))  # Another dropout layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (Rain or No Rain)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the CNN model
cnn_model = build_cnn_model()
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_data=(X_val_cnn, y_val))

# Validate the CNN model
y_pred_cnn = cnn_model.predict(X_val_cnn)
y_pred_cnn = (y_pred_cnn > 0.5).astype(int)

# Calculate accuracy for CNN
cnn_accuracy = accuracy_score(y_val, y_pred_cnn)
print(f'CNN Validation Accuracy: {cnn_accuracy}')

# Predict on the test set with CNN
test_pred_cnn = cnn_model.predict(test_data_cnn)
test_pred_cnn = (test_pred_cnn > 0.5).astype(int)

# Ensure ID and Predictions have the same length
assert len(test_ids) == len(test_pred_cnn), f"Mismatch: {len(test_ids)} IDs vs {len(test_pred_cnn)} predictions"

# Create submission file for the CNN model
submission_cnn = pd.DataFrame({
    'id': test_ids,  # Ensure 'id' column matches expected format
    'RainTomorrow': test_pred_cnn.flatten()  # Flatten to 1D array
})
submission_cnn.to_csv('submission_final_cnn.csv', index=False)

print("Submission file saved as 'submission_final_cnn3.csv'")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, GlobalAveragePooling1D
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Store test IDs before modifying test_data
test_ids = test_data['id']

# Handle missing values - Fill NaN with median (for numerical) or mode (for categorical)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Preprocess the data
X = train_data.drop(columns=['rainfall'])  # Features
y = train_data['rainfall']  # Target variable

# One-hot encoding for categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure feature consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Reshape data for CNN-LSTM (we're treating it as a 1D sequence)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # 3D shape (samples, features, 1)
X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))  # 3D shape
test_data_cnn = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))  # 3D shape for test

def build_cnn_lstm_model_2():
    model = Sequential()

    # First convolutional layer
    model.add(Conv1D(filters=512, kernel_size=7, activation='relu', input_shape=(X_train_cnn.shape[1], 1), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # Second convolutional layer
    model.add(Conv1D(filters=1024, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # Adding more convolutional layers for higher complexity
    model.add(Conv1D(filters=2048, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.6))

    # Adding LSTM Layer
    model.add(LSTM(256, activation='relu', return_sequences=True))  # Added a second LSTM layer
    model.add(LSTM(128, activation='relu', return_sequences=False))  # Reduced units for complexity

    # Fully connected layers
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.7))  # Heavier dropout for more regularization
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Callbacks for early stopping and learning rate adjustment
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the improved CNN-LSTM model with the learning rate scheduler
cnn_lstm_model_2 = build_cnn_lstm_model_2()
cnn_lstm_model_2.fit(X_train_cnn, y_train, epochs=500, batch_size=64, validation_data=(X_val_cnn, y_val), 
                     callbacks=[early_stopping, lr_scheduler])

# Validate the improved CNN-LSTM model
y_pred_cnn_lstm_2 = cnn_lstm_model_2.predict(X_val_cnn)
y_pred_cnn_lstm_2 = (y_pred_cnn_lstm_2 > 0.5).astype(int)

# Calculate accuracy for the improved CNN-LSTM
cnn_lstm_accuracy_2 = accuracy_score(y_val, y_pred_cnn_lstm_2)
print(f'Improved CNN-LSTM Validation Accuracy: {cnn_lstm_accuracy_2}')

# Predict on the test set with the improved CNN-LSTM
test_pred_cnn_lstm_2 = cnn_lstm_model_2.predict(test_data_cnn)
test_pred_cnn_lstm_2 = (test_pred_cnn_lstm_2 > 0.5).astype(int)

# Ensure ID and Predictions have the same length
assert len(test_ids) == len(test_pred_cnn_lstm_2), f"Mismatch: {len(test_ids)} IDs vs {len(test_pred_cnn_lstm_2)} predictions"

# Create submission file for the improved CNN-LSTM model
submission_cnn_lstm_2 = pd.DataFrame({
    'id': test_ids,  # Ensure 'id' column matches expected format
    'RainTomorrow': test_pred_cnn_lstm_2.flatten()  # Flatten to 1D array
})
submission_cnn_lstm_2.to_csv('submission_cnn_lstm_2.csv', index=False)

print("Submission file saved as 'submission_cnn_lstm_2.csv'")





