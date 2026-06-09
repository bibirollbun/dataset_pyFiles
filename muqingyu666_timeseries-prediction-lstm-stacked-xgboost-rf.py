import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

RANDOM_SEED = 0
TARGET_COLUMN = "rainfall"

# File paths
# Kaggle paths
TRAIN_FILE = "/kaggle/input/playground-series-s5e3/train.csv"
TEST_FILE = "/kaggle/input/playground-series-s5e3/test.csv"
SAMPLE_SUBMISSION_FILE = (
    "/kaggle/input/playground-series-s5e3/sample_submission.csv"
)
# Local paths
# TRAIN_FILE = "data/train.csv"
# TEST_FILE = "data/test.csv"
# SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

# Load the data
train_df = pd.read_csv(TRAIN_FILE)
train_df = train_df.drop(columns=["id"])  # Drop the 'id' column

# Local path
# extra_data = pd.read_csv("data/Rainfall.csv")  # Extra dataset
extra_data = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv") # Extra dataset
extra_data.columns = (
    extra_data.columns.str.strip()
)  # Strip whitespace from column names
extra_data["rainfall"] = extra_data["rainfall"].apply(
    lambda x: 1 if x == "yes" else 0
)  # Convert rainfall to binary

# Combine train and extra data for preprocessing
train_df = pd.concat([train_df, extra_data])

test_df = pd.read_csv(TEST_FILE)
test_df = test_df.drop(columns=["id"])  # Drop the 'id' column

sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)

# Display the first few rows to understand the data
print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Sample submission shape:", sample_submission_df.shape)


def basic_preprocessing(train_df, test_df, target_col):
    """Performs basic preprocessing on train and test sets (missing value handling, boolean column conversion)."""
    print("\n=== [2] Data Preprocessing ===")

    # Check missing values
    print("\nMissing values in training set:")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
    print("\nMissing values in test set:")
    print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

    # Fill missing values in numeric columns with median
    numeric_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    for col in numeric_cols:
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(median_value)

    # Fill missing values in categorical columns with mode
    categorical_cols = train_df.select_dtypes(
        include=["object", "category"]
    ).columns
    for col in categorical_cols:
        mode_value = train_df[col].mode(dropna=True)[
            0
        ]  # Get the first mode
        train_df[col] = train_df[col].fillna(mode_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(mode_value)

    # Convert boolean columns
    bool_mapping = {
        "Yes": True,
        "No": False,
        "yes": True,
        "no": False,
        "TRUE": True,
        "FALSE": False,
        "true": True,
        "false": False,
    }
    for col in categorical_cols:
        unique_values = train_df[col].dropna().unique()
        if all(
            str(val).lower() in bool_mapping for val in unique_values
        ):
            print(f"Detected boolean column {col}, converting...")
            train_df[col] = train_df[col].map(bool_mapping)
            if col in test_df.columns:
                test_df[col] = test_df[col].map(bool_mapping)

    print("Basic preprocessing completed!")
    return train_df, test_df


train_df, test_df = basic_preprocessing(
    train_df, test_df, TARGET_COLUMN
)


def feature_engineering(train_df, test_df):
    """Performs feature engineering on train and test sets, generating new features."""
    print("\n=== [2.5] Feature Engineering ===")
    combined_data = pd.concat(
        [train_df, test_df], ignore_index=True, sort=False
    )

    # Log transformation for windspeed (avoiding zero values)
    combined_data["windspeed_log"] = np.log1p(
        combined_data["windspeed"]
    )  # log1p = log(1 + x)

    # Temperature-related features
    combined_data["temp_range"] = (
        combined_data["maxtemp"] - combined_data["mintemp"]
    )
    combined_data["avg_temp"] = (
        combined_data["maxtemp"] + combined_data["mintemp"]
    ) / 2

    # Humidity and dew point interaction feature
    combined_data["humidity_dew_diff"] = (
        combined_data["humidity"] - combined_data["dewpoint"]
    )

    # Cloud-related features
    combined_data["cloud_sun_ratio"] = combined_data["cloud"] / (
        combined_data["sunshine"] + 1e-6
    )
    combined_data["cloud_humidity_interaction"] = (
        combined_data["cloud"] * combined_data["humidity"]
    )
    combined_data["cloud_temp_interaction"] = (
        combined_data["cloud"] * combined_data["avg_temp"]
    )
    combined_data["cloud_wind_interaction"] = (
        combined_data["cloud"] * combined_data["windspeed_log"]
    )
    combined_data["cloud_pressure_ratio"] = combined_data["cloud"] / (
        combined_data["pressure"] + 1e-6
    )
    combined_data["cloud_temp_diff"] = (
        combined_data["cloud"] - combined_data["avg_temp"]
    )

    # Dew point extended feature
    combined_data["dew_point_spread"] = (
        combined_data["temparature"] - combined_data["dewpoint"]
    )

    # # Wind speed intensity category (using log-transformed wind speed)
    # combined_data["wind_speed_intensity"] = pd.cut(
    #     combined_data["windspeed_log"],
    #     bins=[0, np.log1p(10), np.log1p(25), np.log1p(60)],
    #     labels=["Calm", "Breezy", "Windy"],
    # )

    # # Wind direction quadrant (4 quadrants)
    # combined_data["wind_quadrant"] = pd.cut(
    #     combined_data["winddirection"],
    #     bins=[0, 90, 180, 270, 360],
    #     labels=["NE", "SE", "SW", "NW"],
    #     include_lowest=True,
    # )

    # Interaction features
    combined_data["pressure_humidity_interaction"] = (
        combined_data["pressure"] * combined_data["humidity"]
    )
    combined_data["wind_cloud_interaction"] = (
        combined_data["windspeed_log"] * combined_data["cloud"]
    )

    # Temperature ratio
    combined_data["temp_ratio"] = (
        combined_data["temparature"] / combined_data["maxtemp"].max()
    )

    # Split back into training and test sets
    train_df = combined_data.iloc[: len(train_df)].copy()
    test_df = (
        combined_data.iloc[len(train_df) :]
        .copy()
        .drop(columns=["rainfall"], errors="ignore")
    )

    # Handle missing values in new features
    numeric_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    for col in numeric_cols:
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(median_value)

    print(
        f"Training set dimensions after feature engineering: {train_df.shape}, Test set dimensions: {test_df.shape}"
    )
    return train_df, test_df


# Apply feature engineering
train_df, test_df = feature_engineering(train_df, test_df)

# Separate features and target (example: predicting 'Pastry')
X = train_df.drop(columns=TARGET_COLUMN, axis=1)
y = train_df[TARGET_COLUMN]


from sklearn.model_selection import train_test_split

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")


from sklearn.preprocessing import RobustScaler, StandardScaler

# Standardize the data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Reshape for LSTM (samples, timesteps, features)
timesteps = 1
n_features = X_train_scaled.shape[1]


def reshape_data_for_lstm(X, timesteps):
    n_samples = (X.shape[0] // timesteps) * timesteps
    X = X[:n_samples]
    return X.reshape(-1, timesteps, n_features)


X_train_reshaped = reshape_data_for_lstm(X_train_scaled, timesteps)
X_val_reshaped = reshape_data_for_lstm(X_val_scaled, timesteps)

# Adjust target variable length
y_train = y_train.iloc[: X_train_reshaped.shape[0]]
y_val = y_val.iloc[: X_val_reshaped.shape[0]]

print(f"Reshaped training data shape: {X_train_reshaped.shape}")


# ! pip install keras_tuner

import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import AUC
import tensorflow as tf


# Function to build the model for tuning
def build_model(hp):
    inputs = Input(shape=(timesteps, n_features))

    # First LSTM Layer
    x = LSTM(
        hp.Choice("lstm_units_1", [32, 64, 128, 256]),
        return_sequences=True,
    )(inputs)
    x = Dropout(hp.Choice("dropout_1", [0.1, 0.2, 0.3, 0.4, 0.5]))(x)

    # Second LSTM Layer
    x = LSTM(
        hp.Choice("lstm_units_2", [32, 64, 128, 256]),
        return_sequences=False,
    )(x)
    x = Dropout(hp.Choice("dropout_2", [0.1, 0.2, 0.3, 0.4, 0.5]))(x)

    # Batch Normalization to stabilize learning
    x = BatchNormalization()(x)

    # Fully Connected Layer
    x = Dense(
        hp.Choice("dense_units", [16, 32, 64, 128]), activation="relu"
    )(x)
    x = Dropout(hp.Choice("dropout_3", [0.1, 0.2, 0.3, 0.4, 0.5]))(x)

    # Output Layer
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Choose Optimizer
    optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
    learning_rate = hp.Choice(
        "learning_rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    )

    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[AUC(name="auc"), "accuracy"],
    )

    return model


# Define KerasTuner search method (Bayesian Optimization)
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_auc",  # Optimize for AUC
    max_trials=40,  # Number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of times each model is trained
    directory="lstm_tuning",
    project_name="lstm_rainfall",
)

# Perform the hyperparameter search
tuner.search(
    X_train_reshaped,
    y_train,
    validation_data=(X_val_reshaped, y_val),
    epochs=50,
    batch_size=512,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5
        )
    ],
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Rebuild the best model and train
best_model = tuner.hypermodel.build(best_hps)
batch_size = (
    best_hps.get("batch_size")
    if "batch_size" in best_hps.values
    else 512
)

# Define callbacks
callbacks = [
    ModelCheckpoint(
        "best_model.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
        verbose=1,
        restore_best_weights=True,
    ),
]


history = best_model.fit(
    X_train_reshaped,
    y_train,
    validation_data=(X_val_reshaped, y_val),
    epochs=50,
    batch_size=batch_size,  # Use fixed batch size
    callbacks=callbacks,
)


# Print the best hyperparameters
print("Best hyperparameters:")
for key, value in best_hps.values.items():
    print(f"{key}: {value}")


from sklearn.metrics import roc_auc_score, accuracy_score

# Load the best model
best_model.load_weights("best_model.keras")

# Predict on validation set
y_pred_prob = best_model.predict(X_val_reshaped).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
auc = roc_auc_score(y_val, y_pred_prob)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation AUC: {auc:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")


# Prepare test data
X_test = test_df

# Standardize and reshape
X_test_scaled = scaler.transform(X_test)
X_test_reshaped = reshape_data_for_lstm(X_test_scaled, timesteps)

# Predict
test_preds = best_model.predict(X_test_reshaped).flatten()

# Handle NaN values
test_preds = np.nan_to_num(test_preds)

# Create submission DataFrame
submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
submission_df["rainfall"] = test_preds

# Save to CSV
submission_df.to_csv("submission.csv", index=False)

# Display the first few rows
print("-" * 50)
print("Submission file created successfully!")
print("Submission looks like this:")
print(submission_df.head())
print("-" * 50)
print(f"Submission shape: {submission_df.shape}")




