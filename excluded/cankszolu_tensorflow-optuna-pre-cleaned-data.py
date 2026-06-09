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


from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import optuna
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam


# Load the dataset
data = pd.read_csv('/kaggle/input/forecasting-sticker-sales-cleaned-data/train_data_1.csv')


# Define the target variable and features
X = data.drop(columns=["num_sold"])
y = data["num_sold"]


# Train-test split (95% training, 5% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)


# Check the dimensions of the data
X_train_shape = X_train.shape
X_test_shape = X_test.shape
y_train_shape = y_train.shape
y_test_shape = y_test.shape

(X_train_shape, X_test_shape, y_train_shape, y_test_shape)


# Define Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),  # 1. Hidden Layer
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),  # 2. Hidden Layer
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)  # Output Layer (Lineer Activation)
])


# Compile Model
model.compile(optimizer='adam', loss='mape', metrics=['mape'])


# Model summary
model.summary()


# Model training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    verbose=1
)


# Visualize training and validation losses
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    
    # MAPE plot
    plt.plot(history.history['mape'], label='Training MAPE')
    plt.plot(history.history['val_mape'], label='Validation MAPE')
    plt.title('Training and Validation MAPE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Percentage Error (MAPE)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAPE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the training history
plot_training_history(history)


def objective(trial):
    # Hyperparameters suggested by Optuna
    units_1 = trial.suggest_int('units_1', 64, 256)  # Number of neurons in the first layer
    units_2 = trial.suggest_int('units_2', 32, 128)  # Number of neurons in the second layer
    units_3 = trial.suggest_int('units_3', 16, 64)   # Number of neurons in the third layer
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)  # Dropout rate
    activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'swish', 'tanh'])  # Activation function
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Learning rate
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)  # L2 regularization
    n_layers = trial.suggest_int('n_layers', 2, 4)  # Number of layers
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])  # Batch size

    # Model creation
    model = Sequential()
    model.add(Dense(units_1, activation=activation_fn, input_shape=(X_train.shape[1],),
                    kernel_regularizer=l2(l2_reg)))  # First layer
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Dynamic layer addition (adds Dense layers according to n_layers)
    for i in range(n_layers - 1):
        units = trial.suggest_int(f'units_l{i}', 32, 128)
        model.add(Dense(units, activation=activation_fn, kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))  # Linear activation for MAPE prediction

    # Model compilation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mape', metrics=['mape'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_mape', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_mape', factor=0.5, patience=5, min_lr=1e-6)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_mape', mode='min')

    # Model training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=0
    )
    
    # Return the validation MAPE score
    val_mape = min(history.history['val_mape'])
    return val_mape


# Create an Optuna study
study = optuna.create_study(direction='minimize')  # We aim to minimize MAPE
study.optimize(objective, n_trials=100)  # Run 100 trials

# Display the best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")


# Manually setting the best hyperparameters
best_params = {
    'units_1': 135,
    'units_2': 60,
    'units_3': 59,
    'dropout_rate': 0.1386233261993522,
    'activation_fn': 'tanh',
    'lr': 0.0021820242700000205,
    'l2_reg': 0.0006194000028441383,
    'n_layers': 3,
    'batch_size': 256,
    'units_l0': 108,
    'units_l1': 54
}


# Model creation using best hyperparameters
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Input layer
model.add(Dense(best_params['units_1'], activation=best_params['activation_fn'],
                kernel_regularizer=l2(best_params['l2_reg'])))
model.add(BatchNormalization())
model.add(Dropout(best_params['dropout_rate']))

# Additional dense layers
model.add(Dense(best_params['units_l0'], activation=best_params['activation_fn'],
                kernel_regularizer=l2(best_params['l2_reg'])))
model.add(BatchNormalization())
model.add(Dropout(best_params['dropout_rate']))

model.add(Dense(best_params['units_l1'], activation=best_params['activation_fn'],
                kernel_regularizer=l2(best_params['l2_reg'])))
model.add(BatchNormalization())
model.add(Dropout(best_params['dropout_rate']))

# Output layer
model.add(Dense(1))  # Linear activation for regression (MAPE prediction)

# Model compilation
model.compile(optimizer=Adam(learning_rate=best_params['lr']),
              loss='mape', metrics=['mape'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_mape', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_mape', factor=0.5, patience=5, min_lr=1e-6)

# Model training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=best_params['batch_size'],
    callbacks=[early_stopping, reduce_lr],
    verbose=1  
)


# Training and validation MAPE plot
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    
    # MAPE plot
    plt.plot(history.history['mape'], label='Training MAPE', linestyle='-', marker='o')
    plt.plot(history.history['val_mape'], label='Validation MAPE', linestyle='--', marker='x')
    plt.title('Training and Validation MAPE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Percentage Error (MAPE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linestyle='-', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAPE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the result of the model training
plot_training_history(history)


# Load the test dataset
test_data = pd.read_csv('/kaggle/input/forecasting-sticker-sales-cleaned-data/test_data_1.csv')


# Extract the 'id' column from the test dataset
test_ids = test_data['id']


# Drop the 'id' column from the test dataset
test_data = test_data.drop(columns=['id'])


# Make predictions on the test set
print("\nMaking predictions on the test set...")
predictions = model.predict(test_data)


# Ensure predictions are 1-dimensional
predictions = predictions.flatten()  # Converts predictions to 1D array


# Save the results
submission = pd.DataFrame({
    'id': test_ids,
    'num_sold': (predictions > 0.5).astype(int)
submission.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'.")




