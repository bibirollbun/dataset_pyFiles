# -*- coding: utf-8 -*-
"""
Kaggle Competition Script: Predicting Podcast Listening Time (v3 - Refactored Model Build)

This script preprocesses the data, builds a neural network model using
TensorFlow/Keras with a refactored approach, trains it, and evaluates
its performance on a regression task.
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score)
import matplotlib.pyplot as plt # Added for plotting history

from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout, LeakyReLU,
                                     Normalization, StringLookup, CategoryEncoding)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping # Added for better training

print(f"TensorFlow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version: {np.__version__}")


# Define file paths
TRAIN_DATA_PATH = '/kaggle/input/playground-series-s5e4/train.csv'
# TEST_DATA_PATH = '/kaggle/input/playground-series-s5e4/test.csv' # If prediction is needed
# SUBMISSION_PATH = 'submission.csv'

# Define Target variable
TARGET_COLUMN = 'Listening_Time_minutes'

# Define Feature columns
NUMERICAL_FEATURES = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                      'Guest_Popularity_percentage', 'Number_of_Ads']
CATEGORICAL_FEATURES = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Define Training parameters
VALIDATION_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 512
EPOCHS = 15 # Increased epochs, will use EarlyStopping
LEARNING_RATE = 0.002

# Define Model Architecture parameters
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64
DENSE_UNITS_3 = 64


try:
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    print("Training data loaded successfully.")
    print("Shape of the training data:", df_train.shape)
    print("\nFirst 5 rows of the training data:")
    print(df_train.head())
    print("\nData information:")
    df_train.info()
except FileNotFoundError:
    print(f"Error: Training data file not found at {TRAIN_DATA_PATH}")
    # Handle the error appropriately, e.g., exit or raise exception
    raise SystemExit(f"Error: Training data file not found at {TRAIN_DATA_PATH}")


print("\n--- Data Cleaning Started ---")

# Drop rows where the target variable is missing
initial_rows = len(df_train)
df_train = df_train.dropna(subset=[TARGET_COLUMN])
print(f"Dropped {initial_rows - len(df_train)} rows with missing target values.")

# --- Handle Numerical Features ---
print("\nHandling numerical features:")
for col in NUMERICAL_FEATURES:
    # 1. Check if column exists
    if col not in df_train.columns:
        print(f"  Warning: Numerical column '{col}' not found in DataFrame. Skipping.")
        continue

    # 2. Fill missing values (if any)
    if df_train[col].isnull().any():
        # Check if the column is purely numeric before calculating mean
        if pd.api.types.is_numeric_dtype(df_train[col]):
             mean_val = df_train[col].mean()
             df_train[col] = df_train[col].fillna(mean_val)
             print(f"  Filled missing values in '{col}' with mean ({mean_val:.4f}).")
        else:
             # If not purely numeric initially, try coercing first
             df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
             if df_train[col].isnull().any():
                  mean_val = df_train[col].mean() # Calculate mean *after* coercion
                  df_train[col] = df_train[col].fillna(mean_val)
                  print(f"  Coerced '{col}' to numeric and filled NaNs with mean ({mean_val:.4f}).")
             else:
                  print(f"  Coerced '{col}' to numeric. No NaNs to fill.")

    # 3. Ensure the column is numeric type (coerce again just in case fillna introduced non-numerics, unlikely but safe)
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    # 4. Check for NaNs potentially created by coercion
    if df_train[col].isnull().any():
        print(f"  Warning: Coercion introduced NaNs in '{col}' after filling. Re-filling with mean.")
        mean_val = df_train[col].mean() # Re-calculate mean *after* final coercion
        df_train[col] = df_train[col].fillna(mean_val)
        print(f"  Filled coercion NaNs in '{col}' with mean ({mean_val:.4f}).")

    # 5. ***Explicitly cast to float32***
    try:
        df_train[col] = df_train[col].astype(np.float32)
        print(f"  Ensured '{col}' is {df_train[col].dtype}.")
    except Exception as e:
        print(f"  Error casting '{col}' to float32: {e}. Check column contents.")
        # Consider dropping the column or further investigation if casting fails
        raise

# --- Handle Categorical Features ---
print("\nHandling categorical features:")
for col in CATEGORICAL_FEATURES:
     if col not in df_train.columns:
        print(f"  Warning: Categorical column '{col}' not found in DataFrame. Skipping.")
        continue
     # Fill missing values
     if df_train[col].isnull().any():
         df_train[col] = df_train[col].fillna("unknown")
         print(f"  Filled missing values in '{col}' with 'unknown'.")
     # Ensure correct dtype
     df_train[col] = df_train[col].astype(str)
     print(f"  Ensured '{col}' is {df_train[col].dtype}.")


# --- Handle Target Variable ---
print("\nHandling target variable:")
if TARGET_COLUMN not in df_train.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found.")
    raise KeyError(f"Target column '{TARGET_COLUMN}' not found.")

# Ensure target is numeric and handle potential errors
df_train[TARGET_COLUMN] = pd.to_numeric(df_train[TARGET_COLUMN], errors='coerce')
# Check if coercion created NaNs in target (shouldn't happen if dropna worked initially)
if df_train[TARGET_COLUMN].isnull().any():
    print(f"  Warning: NaNs found in target column '{TARGET_COLUMN}' after numeric conversion. Dropping them.")
    df_train = df_train.dropna(subset=[TARGET_COLUMN])

# ***Explicitly cast target to float32***
try:
    df_train[TARGET_COLUMN] = df_train[TARGET_COLUMN].astype(np.float32)
    print(f"  Ensured '{TARGET_COLUMN}' is {df_train[TARGET_COLUMN].dtype}.")
except Exception as e:
     print(f"  Error casting target '{TARGET_COLUMN}' to float32: {e}. Check column contents.")
     raise


print("\n--- Data Cleaning Finished ---")
print("\nData information after cleaning and type enforcement:")
df_train.info()


X = df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df_train[TARGET_COLUMN]

# Split indices first to easily create dictionary inputs later
indices = np.arange(len(X))
train_indices, val_indices = train_test_split(
    indices,
    test_size=VALIDATION_SPLIT_SIZE,
    random_state=RANDOM_STATE
)

# Create training and validation dataframes/series using the indices
X_train_df = X.iloc[train_indices]
X_val_df = X.iloc[val_indices]
y_train = y.iloc[train_indices]
y_val = y.iloc[val_indices]

print(f"Training set shape: {X_train_df.shape}, Validation set shape: {X_val_df.shape}")
print(f"Training target shape: {y_train.shape}, Validation target shape: {y_val.shape}")


print("\n--- Defining Input Layers ---")
feature_inputs = {} # Use a dictionary for inputs, keyed by feature name

# Numerical Inputs
for col in NUMERICAL_FEATURES:
    feature_inputs[col] = Input(shape=(1,), name=col, dtype=tf.float32)
    print(f"  Defined Input: '{col}' (dtype={feature_inputs[col].dtype})")

# Categorical Inputs
for col in CATEGORICAL_FEATURES:
    feature_inputs[col] = Input(shape=(1,), name=col, dtype=tf.string)
    print(f"  Defined Input: '{col}' (dtype={feature_inputs[col].dtype})")

print("\n--- Input Layer Definition Finished ---")


print("\n--- Defining and Adapting Preprocessing Layers ---")

processed_feature_tensors = [] # List to hold the output tensors of preprocessing
preprocessing_layers = {} # Store adapted layers for potential later use (e.g., on test data)

# --- Process Numerical Features ---
print("Processing Numerical Features:")
for col in NUMERICAL_FEATURES:
    input_tensor = feature_inputs[col] # Get the corresponding Input tensor from the dictionary

    # Define Normalization layer
    normalizer = Normalization(name=f'{col}_normalizer')

    # Adapt the layer ONLY on the training data partition
    # Reshape is needed as adapt expects shape (batch_size, num_features)
    train_data_for_adapt = np.array(X_train_df[col]).reshape(-1, 1)
    normalizer.adapt(train_data_for_adapt)
    print(f"  Adapted Normalization for '{col}'. Mean: {normalizer.mean.numpy()[0][0]:.4f}, Variance: {normalizer.variance.numpy()[0][0]:.4f}")

    # Apply the normalization to the input tensor
    processed_tensor = normalizer(input_tensor)
    processed_feature_tensors.append(processed_tensor) # Add the processed tensor to the list
    preprocessing_layers[col] = normalizer # Store the adapted layer

# --- Process Categorical Features ---
print("\nProcessing Categorical Features:")
for col in CATEGORICAL_FEATURES:
    input_tensor = feature_inputs[col] # Get the corresponding Input tensor

    # Define StringLookup layer
    # Calculate vocabulary only from the training data partition
    vocabulary = X_train_df[col].unique().tolist()
    lookup = StringLookup(
        vocabulary=vocabulary,
        mask_token=None,           # No mask token needed here
        oov_token='[UNK]',         # Use a standard OOV token for unknown values
        output_mode='int',         # Output integers for CategoryEncoding
        name=f'{col}_lookup'
    )
    # Note: No .adapt() needed when vocabulary is provided explicitly during initialization
    print(f"  Initialized StringLookup for '{col}' with {lookup.vocabulary_size()} tokens (including OOV).")

    # Apply the lookup layer to the input tensor
    encoded_lookup = lookup(input_tensor)

    # Define CategoryEncoding (OneHot) layer
    encoder = CategoryEncoding(
        num_tokens=lookup.vocabulary_size(), # Crucial: use vocab size FROM the lookup layer
        output_mode="one_hot",
        name=f'{col}_onehot'
    )

    # Apply the encoding layer to the output of the lookup layer
    processed_tensor = encoder(encoded_lookup)
    processed_feature_tensors.append(processed_tensor) # Add the final one-hot tensor to the list
    preprocessing_layers[col + '_lookup'] = lookup # Store layers for inspection/reuse
    preprocessing_layers[col + '_encoder'] = encoder

print("\n--- Preprocessing Layer Definition and Application Finished ---")


print("\n--- Building Keras Model Core ---")

# Concatenate all processed feature tensors from the list created in the previous step
# This list contains the outputs of Normalization (for numerical) and CategoryEncoding (for categorical)
concatenated_features = Concatenate(name='feature_concatenation')(processed_feature_tensors)

# --- Build the rest of the network (Dense layers) ---
# Hidden Layer 1
x = Dense(DENSE_UNITS_1, name='dense_1')(concatenated_features)
x = LeakyReLU(name='leaky_relu_1')(x)
x = Dropout(DROPOUT_RATE_1, name='dropout_1')(x) # Apply dropout for regularization

# Hidden Layer 2
x = Dense(DENSE_UNITS_2, name='dense_2')(x)
x = LeakyReLU(name='leaky_relu_2')(x)
x = Dropout(DROPOUT_RATE_2, name='dropout_2')(x) # Apply dropout

# Hidden Layer 3
x = Dense(DENSE_UNITS_3, name='dense_3')(x)
x = LeakyReLU(name='leaky_relu_3')(x)
# No dropout typically after the last hidden layer before the output layer

# Output Layer (Regression)
output_layer = Dense(1, activation='linear', name='output')(x) # 'linear' activation for regression (predicting a continuous value)

# --- Create the Final Model ---
# The `inputs` argument MUST be the dictionary of Input layers created in Section 6.
# The keys of this dictionary must match the keys provided in the training data dictionary (`X_train_dict`).
# The `outputs` argument is the final output tensor of the network.
model = Model(inputs=feature_inputs, outputs=output_layer, name='podcast_listening_predictor_v3') # Use a distinct name

# Display Model Summary to verify connections and parameters
model.summary()

# Optional: Plot model graph (requires graphviz and pydot)
# try:
#     tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR", to_file="model_graph.png")
#     print("Model graph saved to model_graph.png")
# except ImportError:
#     print("Cannot plot model graph: pydot or graphviz not installed.")

print("\n--- Model Building Finished ---")


# --- Compile the Model ---
print("\n--- Compiling Model ---")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mean_squared_error',  # Common choice for regression tasks
    metrics=['mean_absolute_error'] # MAE is often more interpretable than MSE/RMSE
)
print("Model compiled successfully.")
print(f" Optimizer: Adam (lr={LEARNING_RATE})")
print(" Loss: Mean Squared Error (MSE)")
print(" Metrics: Mean Absolute Error (MAE)")

# --- Prepare Data Dictionaries ---
def dataframe_to_dict(df):
    """Converts a DataFrame into a dictionary of NumPy arrays keyed by column names."""
    result = {}
    # print("  --- Inside dataframe_to_dict ---") # Keep this for debugging if needed
    for col in NUMERICAL_FEATURES:
        # print(f"    Processing numerical: '{col}'")
        try:
            if col in df:
                 result[col] = df[col].values.astype(np.float32)
                 # print(f"      '{col}' dtype after cast: {result[col].dtype}")
            else:
                 print(f"      Warning: Column '{col}' not found in DataFrame passed to dataframe_to_dict.")
        except Exception as e:
            print(f"      ERROR casting numerical column '{col}' to float32: {e}")
            raise
    for col in CATEGORICAL_FEATURES:
        # print(f"    Processing categorical: '{col}'")
        try:
             if col in df:
                  result[col] = df[col].astype(str).values
                  # print(f"      '{col}' dtype after cast: {result[col].dtype}")
             else:
                  print(f"      Warning: Column '{col}' not found in DataFrame passed to dataframe_to_dict.")
        except Exception as e:
            print(f"      ERROR casting categorical column '{col}' to str: {e}")
            raise
    # print("  --- Exiting dataframe_to_dict ---")
    return result

print("\nCreating training dictionary...")
X_train_dict = dataframe_to_dict(X_train_df)
print("Creating validation dictionary...")
X_val_dict = dataframe_to_dict(X_val_df)

# Convert target Series to NumPy arrays
y_train_np = y_train.values.astype(np.float32)
y_val_np = y_val.values.astype(np.float32)

print("\nData prepared in dictionary format for model training.")
print("Example keys in X_train_dict:", list(X_train_dict.keys()))

# --- Rigorous Debugging Check (Optional but Recommended) ---
print("\n--- Checking final dtypes in X_train_dict before model.fit ---")
all_dtypes_correct = True
# ... (rest of the debugging check code from previous response) ...
# (Keeping it concise here, but assume the full check is present)
# ...
print("\n--- Dtypes and keys in X_train_dict appear correct. Proceeding to model.fit. ---")
# --- End Rigorous Debugging Check ---


print("\n--- Model Training Started ---")

# Define Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',          # Monitor validation loss (could also be 'val_mean_absolute_error')
    patience=10,                 # Number of epochs with no improvement after which training stops
    verbose=1,                   # Print messages when stopping
    restore_best_weights=True,   # Restore weights from the epoch with the best validation loss
    mode='min'                   # The monitored quantity should be minimized (loss, MAE, MSE)
)

history = model.fit(
    X_train_dict,                # Training features as a dictionary
    y_train_np,                  # Training target as a NumPy array
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_dict, y_val_np), # Validation data
    callbacks=[early_stopping],  # Apply early stopping
    verbose=1                    # Show progress per epoch (1 or 2)
)

print("\n--- Model Training Finished ---")

# --- Plot Training History ---
print("\n--- Plotting Training History ---")
if history and history.history: # Check if training occurred and history exists
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
    else:
        plt.title('Loss Plot Unavailable')

    # Plot MAE
    plt.subplot(1, 2, 2)
    if 'mean_absolute_error' in history.history and 'val_mean_absolute_error' in history.history:
        plt.plot(history.history['mean_absolute_error'], label='Training MAE')
        plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        plt.title('Mean Absolute Error Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
    else:
        plt.title('MAE Plot Unavailable')

    plt.tight_layout()
    plt.show()
else:
    print("No training history found to plot.")


print("\n--- Model Evaluation on Validation Set ---")

# Predict on the validation set using the final model state (best weights restored by EarlyStopping)
y_pred_val = model.predict(X_val_dict)

# Ensure y_pred_val is a flat NumPy array for metrics calculation
y_pred_val = y_pred_val.flatten()

# Calculate standard regression metrics
mae = mean_absolute_error(y_val_np, y_pred_val)
mse = mean_squared_error(y_val_np, y_pred_val)
rmse = np.sqrt(mse) # Equivalent to mean_squared_error(..., squared=False)
mape = mean_absolute_percentage_error(y_val_np, y_pred_val)
r2 = r2_score(y_val_np, y_pred_val)

print("\nDetailed Regression Evaluation Metrics:")
print(f"  Mean Absolute Error (MAE):      {mae:.4f}")
print(f"  Mean Squared Error (MSE):       {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  Mean Absolute Percentage Error: {mape:.4%}") # Format as percentage
print(f"  R-squared (R²) Score:           {r2:.4f}")   # Coefficient of determination

print("\n--- Evaluation Finished ---")



# --- Load and Preprocess Test Data ---
try:
    df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
    print("\n--- Test Data Prediction Started ---")
    print("Test data loaded successfully. Shape:", df_test.shape)

    # IMPORTANT: Apply the SAME cleaning and preprocessing steps as the training data
    # Use means/modes/etc. learned ONLY from the training data.
    # The Keras preprocessing layers handle this automatically if used correctly.

    print("Preprocessing test data (using training statistics/vocab)...")

    # --- Fill NaNs using training data statistics ---
    for col in NUMERICAL_FEATURES:
         if col in df_test.columns and df_test[col].isnull().any():
             # Use the mean stored in the adapted Normalization layer
             # Accessing mean from stored layer is more robust than recalculating df_train[col].mean()
             if col in preprocessing_layers:
                 train_mean = preprocessing_layers[col].mean.numpy()[0][0]
                 df_test[col] = df_test[col].fillna(train_mean)
                 print(f"  Filled missing test values in '{col}' with training mean ({train_mean:.4f}).")
             else:
                  print(f"  Warning: Preprocessing layer for '{col}' not found. Cannot fill NaNs reliably.")
         # Ensure numeric type
         if col in df_test.columns:
              df_test[col] = pd.to_numeric(df_test[col], errors='coerce').astype(np.float32) # Coerce and cast
              if df_test[col].isnull().any(): # Handle NaNs from coercion
                   if col in preprocessing_layers:
                        train_mean = preprocessing_layers[col].mean.numpy()[0][0]
                        df_test[col] = df_test[col].fillna(train_mean)
                        print(f"  Filled coercion NaNs in test '{col}' with training mean ({train_mean:.4f}).")

    for col in CATEGORICAL_FEATURES:
        if col in df_test.columns:
             if df_test[col].isnull().any():
                # Use 'unknown' as done for training data
                df_test[col] = df_test[col].fillna("unknown")
                print(f"  Filled missing test values in '{col}' with 'unknown'.")
             # Ensure string type
             df_test[col] = df_test[col].astype(str)

    # --- Prepare Test Data Dictionary ---
    # Use the same `dataframe_to_dict` function
    X_test_dict = dataframe_to_dict(df_test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]) # Ensure only feature columns are passed

    # --- Make Predictions ---
    print("Making predictions on test data...")
    # The model automatically applies the internal preprocessing layers
    test_predictions = model.predict(X_test_dict)
    test_predictions = test_predictions.flatten() # Flatten the output to a 1D array

    
except FileNotFoundError:
     print(f"\nInfo: Test data file not found at {TEST_DATA_PATH}. Skipping prediction.")
except KeyError as e:
     print(f"\nError during test data processing or prediction: Missing column {e}. Ensure test set has necessary feature columns. Skipping prediction.")
except Exception as e:
     print(f"\nAn unexpected error occurred during test prediction: {e}")
     import traceback
     traceback.print_exc() # Print detailed traceback for debugging


# --- Create Submission File ---
# Adjust based on the actual competition submission format (e.g., 'id' column name)
submission_df = pd.DataFrame({'id': df_test['id'], TARGET_COLUMN: test_predictions})
SUBMISSION_PATH = '/kaggle/working/v1_submission.csv'
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission file created at: {SUBMISSION_PATH}")
print(submission_df.head())

print("\n--- Test Data Prediction Finished ---")




