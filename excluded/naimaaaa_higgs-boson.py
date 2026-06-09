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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb






# Assuming 'train.csv' is in the specified path.
try:
    # Read the CSV file from the specified path, letting pandas infer the header
    df = pd.read_csv('/content/drive/MyDrive/train.csv') # Removed header=None and names=column_names

    print("Initial DataFrame head:")
    print(df.head())
    print("\nInitial DataFrame info:")
    df.info()

    # --- 1. Data Preparation ---

    # Convert 'Prediction' to numerical (0 for background 'b', 1 for signal 's')
    # If there are any missing labels, they will be converted to NaN by map and then to -1 by fillna
    # Changed 'Label' to 'Prediction'
    df['Prediction'] = df['Prediction'].map({'b': 0, 's': 1}).fillna(-1).astype(int)

    # Filter out rows where Prediction is -1 (originally NaN or other unexpected values)
    # This ensures we only use data with valid labels for training
    # Changed 'Label' to 'Prediction'
    df = df[df['Prediction'] != -1].copy()


    # Drop 'Id' and 'Weight' as they are not features for classification
    # Keep 'Weight' for potential use in weighted metrics if needed later, but drop it for now for standard training
    # Changed 'Label' to 'Prediction'
    X = df.drop(['Id', 'Prediction', 'Weight'], axis=1, errors='ignore')
    y = df['Prediction'] # Changed 'Label' to 'Prediction']

    # Handle missing or zero-filled values:
    # The dataset description mentions -999.0 as a placeholder for missing values.
    # Let's replace these with NaN and then impute.
    X = X.replace(-999.0, np.nan)

    # Impute missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0: # Check if there are any NaNs in the column
            if X[col].count() > 0: # Check if there are any non-NaN values to calculate a meaningful mean
                X[col] = X[col].fillna(X[col].mean())
            else:
                 # If a column is entirely NaN after replacement, fill with 0
                 X[col] = X[col].fillna(0)


    print("\nMissing values after imputation:")
    print(X.isnull().sum().sum()) # Should be 0

    # Normalize continuous features

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Train/validation/test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_df, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")

    # --- 2. Model Building ---

    # Define architecture
    def build_model(input_dim, optimizer_name='Adam', learning_rate=0.001):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid') # Output layer for binary classification
        ])

        if optimizer_name == 'Adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'AdamW': # AdamW is not directly available in core Keras/TF before 2.9
                                       # Requires tensorflow_addons or a custom implementation.
                                       # For simplicity, sticking to Adam and RMSprop.
            print("AdamW not directly supported without tensorflow_addons. Using Adam.")
            optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported optimizer name")

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    input_dim = X_train.shape[1]
    model = build_model(input_dim, optimizer_name='Adam', learning_rate=0.001)
    model.summary()

    # Callbacks
    #early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) # Removed EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)

    # Train the model
    epochs = 100 # Can be increased, EarlyStopping will manage it
    batch_size = 64

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[reduce_lr], # Removed early_stopping
                        verbose=1)

    # The remaining code for model evaluation is in the next cell (U7qLfOTR4XUW)

except FileNotFoundError:
    print(f"Error: The file '/content/drive/MyDrive/train.csv' was not found.")
    print("Please check the file path and try again.")
    # The rest of the code for data preparation, model building, and evaluation
    # cannot be executed without the dataset file.
    # The script will stop here and inform the user about the missing file.

