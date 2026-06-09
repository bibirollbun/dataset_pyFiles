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


# Import essential libraries
import numpy as np
import pandas as pd
import os # Essential for handling file paths
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # For progress bars

# --- Define Paths (Kaggle Environment) ---
# This is the standard path in Kaggle competitions
BASE_DIR = '/kaggle/input/ariel-data-challenge-2025/'

# Check if the directory exists to provide a helpful error
if not os.path.exists(BASE_DIR):
    # If not in Kaggle, update this path to where you saved the data
    print("Kaggle directory not found. Trying local 'ariel-data-challenge-2025/'...")
    BASE_DIR = 'ariel-data-challenge-2025/'
    if not os.path.exists(BASE_DIR):
        print("Error: Data directory not found. Please download the data and check the BASE_DIR path.")
        # Stop execution if data is not found
        # In a real script, you might raise an exception here.
        exit()

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# --- Load the Metadata CSVs ---
try:
    # This file contains the planet IDs and their ground truth target fluxes
    train_labels_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    
    # This file contains information about the wavelengths for the target fluxes
    wavelengths_df = pd.read_csv(os.path.join(BASE_DIR, 'wavelengths.csv'))
    
    # The sample submission tells us the required format for our predictions
    sample_submission_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))

    print("Metadata loaded successfully!")
    print("\nTraining labels shape:", train_labels_df.shape)
    print(train_labels_df.head())
    
except FileNotFoundError as e:
    print(f"Error loading metadata: {e}")
    print("Please ensure your data directory is structured correctly.")
    exit()


def load_and_process_planet_data(planet_id, data_dir):
    """
    Loads the time-series signal data for a single planet and engineers basic features.
    """
    try:
        # Construct the path to the planet's signal file.
        # NOTE: The file might have a name like 'FGS1_signal_0.parquet' or similar.
        # We use a wildcard (*) to find the parquet file.
        # This example assumes we're using the FGS1 instrument data.
        # You should explore using AIRS data as well.
        
        # A more robust way is to find the file explicitly
        planet_folder = os.path.join(data_dir, str(planet_id))
        signal_file_path = None
        for file in os.listdir(planet_folder):
            if "FGS1_signal" in file and file.endswith('.parquet'):
                signal_file_path = os.path.join(planet_folder, file)
                break
        
        if signal_file_path is None:
            # print(f"Warning: No FGS1 signal file found for planet {planet_id}")
            return None

        # Load the parquet file
        signal_df = pd.read_parquet(signal_file_path)
        
        # --- Baseline Feature Engineering ---
        # The signal_df contains time-series data. Each column is a detector pixel (or similar).
        # For a simple baseline, let's compute the mean and std dev for each column.
        # This flattens the time-series into a single feature vector.
        features_mean = signal_df.mean().values
        features_std = signal_df.std().values
        
        # Combine the features into a single array
        features = np.concatenate([features_mean, features_std])
        
        return features
        
    except Exception as e:
        # print(f"Error processing planet {planet_id}: {e}")
        return None

# --- Process all planets in the training set ---
# Use tqdm for a nice progress bar
tqdm.pandas(desc="Processing Planets")

# Get a list of planet IDs from our labels file
planet_ids = train_labels_df['planet_id'].unique()

# Create features for all planets
all_features = []
corresponding_planet_ids = []

for planet_id in tqdm(planet_ids, desc="Creating Training Features"):
    features = load_and_process_planet_data(planet_id, TRAIN_DIR)
    if features is not None:
        all_features.append(features)
        corresponding_planet_ids.append(planet_id)

# Create our final feature matrix (X)
X_train_processed = pd.DataFrame(all_features)
X_train_processed['planet_id'] = corresponding_planet_ids

# Merge with labels to ensure correct alignment
# Set index to planet_id for easy merging
train_labels_indexed = train_labels_df.set_index('planet_id')
X_train_final = X_train_processed.set_index('planet_id').join(train_labels_indexed, how='inner')

# Separate features (X) and targets (y)
y_train_final = X_train_final[train_labels_df.columns.drop('planet_id')]
X_train_final = X_train_final.drop(columns=y_train_final.columns)

print("\nFeature matrix created.")
print("Shape of X_train:", X_train_final.shape)
print("Shape of y_train:", y_train_final.shape)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# --- Data Preprocessing ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_final)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_train_final.values, test_size=0.2, random_state=42
)

# --- Build the Model (Multi-output) ---
def build_multi_output_model(input_shape, output_shape):
    """Builds a neural network for multi-output regression."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=[input_shape]),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # The output layer must have one neuron for each target wavelength
        tf.keras.layers.Dense(output_shape)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Create the model
input_shape = X_train.shape[1]
output_shape = y_train.shape[1] # Number of target wavelengths
model = build_multi_output_model(input_shape, output_shape)
model.summary()


# --- Train the Model ---
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping]
)


# --- Process the Test Set ---
# Get a list of test planet IDs from the sample submission file
test_planet_ids = sample_submission_df['planet_id'].unique()

all_test_features = []
corresponding_test_ids = []

for planet_id in tqdm(test_planet_ids, desc="Creating Test Features"):
    features = load_and_process_planet_data(planet_id, TEST_DIR)
    if features is not None:
        all_test_features.append(features)
        corresponding_test_ids.append(planet_id)

# Create the final test feature matrix
X_test_processed = pd.DataFrame(all_test_features)

# Scale the test features using the *same scaler* fitted on the training data
X_test_scaled = scaler.transform(X_test_processed)

# --- Generate Predictions ---
print("Generating test predictions...")
predictions = model.predict(X_test_scaled)

# --- Create Submission File ---
# Create a DataFrame with the predictions
pred_df = pd.DataFrame(predictions, columns=y_train_final.columns)
pred_df['planet_id'] = corresponding_test_ids

# Reorder columns to match submission format if necessary
pred_df = pred_df[['planet_id'] + list(y_train_final.columns)]

# Save to csv
pred_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(pred_df.head())

