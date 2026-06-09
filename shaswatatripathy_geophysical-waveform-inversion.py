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


# Cell 1: Setup, Configuration, and Imports (No change needed, comments added)
print("--- Cell 1: Setup ---")
import numpy as np
import pandas as pd
import os
import gc # Garbage Collector
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # Progress bars

# Deep Learning Framework - TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow Version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    # Added explicit warning about training time
    print("WARNING: GPU not available. Training will be very slow, especially with image-based models.")

# --- Paths ---
BASE_PATH = '/kaggle/input/waveform-inversion/'
TRAIN_PATH = os.path.join(BASE_PATH, 'train_samples/')
TEST_PATH = os.path.join(BASE_PATH, 'test/')
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, 'sample_submission.csv')
OUTPUT_DIR = '/kaggle/working/' # Directory to save model weights and submission file
MODEL_WEIGHTS_FILE = os.path.join(OUTPUT_DIR, 'best_unet_model.keras')
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, 'submission.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
# Data dimensions
DEPTH = 70
WIDTH = 70
TIME_STEPS = 1000
N_RECEIVERS = 70
N_SHOTS = 5

# Model Input/Output Shapes (with padding for U-Net)
PAD_H = 1024
PAD_W = 80
INPUT_SHAPE = (PAD_H, PAD_W, N_SHOTS) # (1024, 80, 5)
OUTPUT_SHAPE = (DEPTH, WIDTH)         # (70, 70)

# Submission format details
SUBMISSION_Y_POS = DEPTH
SUBMISSION_X_INDICES = np.arange(1, WIDTH, 2)
SUBMISSION_X_COLS = [f'x_{i}' for i in SUBMISSION_X_INDICES]
SUBMISSION_ID_COL = 'oid_ypos'

# Training Hyperparameters
# NOTE: BATCH_SIZE adjusted for validation below.
#       EPOCHS might need significant tuning based on overfitting.
BATCH_SIZE_TRAIN = 8    # Keep train batch size reasonable
EPOCHS = 30           # Starting point, monitor validation loss closely
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1 # Using 10% (2 samples) for validation

# Velocity normalization range
V_MIN = 1500.0
V_MAX = 5000.0

# --- End Configuration ---

print(f"Target Velocity Map Shape: {OUTPUT_SHAPE}")
print(f"Seismic Data Shape (Shots, Time, Receivers): ({N_SHOTS}, {TIME_STEPS}, {N_RECEIVERS})")
print(f"Padded Model Input Shape: {INPUT_SHAPE}")
print(f"Submission ID Column: '{SUBMISSION_ID_COL}'")
print(f"Submission X Columns ({len(SUBMISSION_X_COLS)}): {SUBMISSION_X_COLS}")
print(f"Model weights will be saved to: {MODEL_WEIGHTS_FILE}")
print(f"Submission file will be saved to: {SUBMISSION_FILE}")
print("--- Cell 1: Done ---")


# Cell 2: Load Sample Submission and Define Test Set Scope (No change needed)
print("\n--- Cell 2: Load Sample Submission ---")
test_oids_from_sub = []
expected_submission_rows = 0
sample_submission = None # Initialize

try:
    sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
    print("Sample Submission Info:")
    # sample_submission.info() # Reduce verbosity
    print(f"Loaded sample submission with {len(sample_submission)} rows.")
    print("Sample Submission Head:")
    print(sample_submission.head())
    expected_submission_rows = len(sample_submission)
    print(f"\nExpected number of rows in final submission: {expected_submission_rows}")

    if SUBMISSION_ID_COL not in sample_submission.columns:
         raise KeyError(f"Submission ID column '{SUBMISSION_ID_COL}' not found in sample_submission.csv")

    sample_submission['oid'] = sample_submission[SUBMISSION_ID_COL].apply(lambda x: x.split('_y_')[0])
    test_oids_from_sub = sorted(sample_submission['oid'].unique())
    print(f"\nFound {len(test_oids_from_sub)} unique test file IDs (oids) required for submission.")
    # print("Example required oids:", test_oids_from_sub[:5]) # Reduce verbosity

    # Structure verification... (already confirmed OK)
    # first_oid_rows = sample_submission[sample_submission['oid'] == test_oids_from_sub[0]]
    # print(f"\nNumber of rows for first oid '{test_oids_from_sub[0]}': {len(first_oid_rows)}")
    # if len(first_oid_rows) == DEPTH: print(f"Row count per oid matches DEPTH ({DEPTH}). OK.")
    # else: print(f"WARNING: Row count per oid ({len(first_oid_rows)}) does NOT match DEPTH ({DEPTH}).")
    # Column name verification... (already confirmed OK)
    # expected_columns = [SUBMISSION_ID_COL] + SUBMISSION_X_COLS
    # if list(sample_submission.columns.drop('oid')) == expected_columns: print(f"\nSubmission columns match expected format. OK.")
    # else: print(f"\nWARNING: Submission columns mismatch!")

except FileNotFoundError:
    print(f"Error: Sample submission file not found at {SAMPLE_SUB_PATH}")
except KeyError as e:
     print(f"Error processing sample submission: {e}. Please check column names.")
except Exception as e:
    print(f"An unexpected error occurred loading or processing sample submission: {e}")

# --- Get all actual test files ---
try:
    actual_test_files_npy = sorted([f for f in os.listdir(TEST_PATH) if f.endswith('.npy')])
    actual_test_oids = sorted([f.split('.')[0] for f in actual_test_files_npy])
    print(f"\nFound {len(actual_test_oids)} actual .npy files in test directory.")
    if not actual_test_oids: print("Warning: No files found in test directory!")

    # Compare required oids vs actual oids (already confirmed OK)
    # if test_oids_from_sub and actual_test_oids:
    #     required_set = set(test_oids_from_sub)
    #     actual_set = set(actual_test_oids)
    #     if required_set == actual_set: print("Required test oids match actual files in test folder.")
    #     # ... (rest of comparison logic omitted for brevity as it passed)
    # elif not test_oids_from_sub:
    #      print("Warning: Could not determine required test oids...")
    #      test_oids_from_sub = actual_test_oids

except FileNotFoundError:
     print(f"Error: Test directory not found at {TEST_PATH}")
     actual_test_oids = []

print("--- Cell 2: Done ---")


# Cell 3: Data Loading and Preprocessing Utilities (No change needed)
print("\n--- Cell 3: Preprocessing Utilities ---")

def load_npy(path):
    """Loads a .npy file safely."""
    try:
        return np.load(path)
    except Exception as e: return None

def preprocess_input(seismic_data_raw, target_h=PAD_H, target_w=PAD_W, input_shape=INPUT_SHAPE):
    """Extracts, transposes, normalizes, and pads seismic data."""
    if seismic_data_raw is None: return None
    if seismic_data_raw.shape != (500, N_SHOTS, TIME_STEPS, N_RECEIVERS): return None
    seismic_data = seismic_data_raw[0, :, :, :]
    seismic_data = np.transpose(seismic_data, (1, 2, 0))
    min_val = np.min(seismic_data); max_val = np.max(seismic_data)
    if max_val > min_val: seismic_data = (seismic_data - min_val) / (max_val - min_val)
    else: seismic_data = np.zeros_like(seismic_data)
    seismic_data_padded = tf.image.resize_with_pad(
        tf.cast(seismic_data, tf.float32), target_h, target_w, method=tf.image.ResizeMethod.BILINEAR
    ).numpy()
    if seismic_data_padded.shape == input_shape: return seismic_data_padded
    else: return None

def preprocess_output(velocity_data_raw, output_shape=OUTPUT_SHAPE, v_min=V_MIN, v_max=V_MAX):
    """Extracts and normalizes velocity data."""
    if velocity_data_raw is None: return None
    if velocity_data_raw.shape != (500, 1, DEPTH, WIDTH): return None
    velocity_data = velocity_data_raw[0, 0, :, :]
    velocity_data = np.clip(velocity_data, v_min, v_max)
    velocity_data = (velocity_data - v_min) / (v_max - v_min)
    if velocity_data.shape == output_shape: return velocity_data
    else: return None

def postprocess_prediction(pred_normalized, output_shape=OUTPUT_SHAPE, v_min=V_MIN, v_max=V_MAX):
    """De-normalizes model output back to original velocity range."""
    if pred_normalized is None: return None
    pred_velocity = pred_normalized * (v_max - v_min) + v_min
    if pred_velocity.shape != output_shape:
        pred_velocity = tf.image.resize(
            tf.expand_dims(tf.cast(pred_velocity, tf.float32), axis=-1),
            [output_shape[0], output_shape[1]]
        ).numpy()
        pred_velocity = np.squeeze(pred_velocity)
    return pred_velocity

def load_train_pairs(train_path):
    """Loads all input/output training file pairs."""
    pairs = []
    sample_types = os.listdir(train_path)
    print(f"Scanning sample types: {sample_types}")
    for s_type in tqdm(sample_types, desc="Scanning Training Samples"):
        s_type_path = os.path.join(train_path, s_type)
        if os.path.isdir(s_type_path):
            files_in_sample = os.listdir(s_type_path)
            # Handle 'seis'/'vel' structure
            if any(f.startswith('seis') for f in files_in_sample):
                 seis_files = sorted([f for f in files_in_sample if f.startswith('seis') and f.endswith('.npy')])
                 for seis_file in seis_files:
                      vel_file = seis_file.replace('seis', 'vel', 1);
                      if vel_file in files_in_sample: pairs.append((os.path.join(s_type_path, seis_file), os.path.join(s_type_path, vel_file)))
            # Handle 'data'/'model' structure
            elif 'data' in files_in_sample and 'model' in files_in_sample:
                 data_dir = os.path.join(s_type_path, 'data'); model_dir = os.path.join(s_type_path, 'model')
                 if os.path.isdir(data_dir) and os.path.isdir(model_dir):
                     data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
                     for data_file in data_files:
                          model_file = data_file.replace('data', 'model', 1); model_file_path = os.path.join(model_dir, model_file)
                          if os.path.exists(model_file_path): pairs.append((os.path.join(data_dir, data_file), model_file_path))
    return pairs

# --- Load Pairs ---
train_pairs = load_train_pairs(TRAIN_PATH)
print(f"\nFound {len(train_pairs)} training pairs.")
if train_pairs: print("Example pair:", train_pairs[0])
else: print("Warning: No training pairs found!")
# --- DATASET SIZE WARNING ---
if len(train_pairs) < 100: # Arbitrary threshold
     print(f"\nWARNING: The training dataset size ({len(train_pairs)}) is very small.")
     print("         This significantly increases the risk of overfitting and may limit model performance.")
     print("         Consider data augmentation or acquiring more data if possible.")

print("--- Cell 3: Done ---")


# Cell 4: Data Generator (Keras Sequence - Adjusted Validation Batch Size)
print("\n--- Cell 4: Data Generator ---")

class DataGenerator(keras.utils.Sequence):
    def __init__(self, pairs, batch_size, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, shuffle=True, is_validation=False):
        self.pairs = pairs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.shuffle = shuffle
        self.is_validation = is_validation # Flag for potential different handling
        self.n = len(self.pairs)
        if self.n == 0:
             print("Warning: DataGenerator created with 0 samples.")
        self.on_epoch_end()
        print(f"DataGenerator created: {self.n} samples, Batch size: {self.batch_size}, Input: {self.input_shape}, Output: {self.output_shape}")

    def __len__(self):
        # Number of batches per epoch
        if self.n == 0: return 0
        return int(np.ceil(self.n / self.batch_size)) # Use ceil to ensure all samples are seen

    def __getitem__(self, index):
        # Generate one batch of data
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n) # Handle last batch size
        actual_batch_size = end_idx - start_idx

        indexes = self.indexes[start_idx:end_idx]
        batch_pairs = [self.pairs[k] for k in indexes]

        # Initialize arrays for the actual batch size
        X = np.empty((actual_batch_size, *self.input_shape))
        y = np.empty((actual_batch_size, *self.output_shape))

        valid_samples_in_batch = 0
        for i, (seis_path, vel_path) in enumerate(batch_pairs):
            seis_data_raw = load_npy(seis_path)
            vel_data_raw = load_npy(vel_path)
            processed_seis = preprocess_input(seis_data_raw)
            processed_vel = preprocess_output(vel_data_raw)

            if processed_seis is not None and processed_vel is not None:
                X[i,] = processed_seis; y[i,] = processed_vel
                valid_samples_in_batch += 1
            else:
                X[i,] = np.zeros(self.input_shape); y[i,] = np.zeros(self.output_shape)
        del seis_data_raw, vel_data_raw, processed_seis, processed_vel
        gc.collect()
        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.n)
        if self.shuffle: np.random.shuffle(self.indexes)

# --- Create Train/Validation Generators ---
train_generator = None
val_generator = None
num_train_samples = 0
num_val_samples = 0

if train_pairs:
    np.random.seed(42)
    np.random.shuffle(train_pairs)
    split_idx = int((1.0 - VALIDATION_SPLIT) * len(train_pairs))
    train_gen_pairs = train_pairs[:split_idx]
    val_gen_pairs = train_pairs[split_idx:]
    num_train_samples = len(train_gen_pairs)
    num_val_samples = len(val_gen_pairs)

    print(f"\nSplitting data: {num_train_samples} training pairs, {num_val_samples} validation pairs.")

    if num_train_samples > 0:
         train_generator = DataGenerator(train_gen_pairs, BATCH_SIZE_TRAIN, shuffle=True)
         if len(train_generator) == 0:
              print("Warning: Training generator has length 0.")
    else:
         print("Warning: No samples available for training generator.")


    if num_val_samples > 0:
         # --- Adjust Validation Batch Size ---
         # Ensure batch size is <= number of validation samples to avoid steps=0
         BATCH_SIZE_VAL = min(num_val_samples, BATCH_SIZE_TRAIN)
         print(f"Using validation batch size: {BATCH_SIZE_VAL}")
         val_generator = DataGenerator(val_gen_pairs, BATCH_SIZE_VAL, shuffle=False, is_validation=True)
         if len(val_generator) == 0:
              print("Warning: Validation generator has length 0, even after batch size adjustment.")
    else:
         print("Warning: No samples available for validation generator. Validation will be skipped.")

    # --- Test and Visualize Generator Output (if possible) ---
    if train_generator and len(train_generator) > 0:
        print("\nTesting generator...")
        try:
            X_batch, y_batch = train_generator[0]
            print(f"Generator test SUCCESS: Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
            # Visualization code... (omitted for brevity, already confirmed working)
            # plt.figure... show()... close()
            del X_batch, y_batch; gc.collect()
        except Exception as e:
            print(f"Error testing generator: {e}")
    else:
         print("Skipping generator test as train_generator is empty.")

else:
    print("Cannot create generators: No training pairs found.")

print("--- Cell 4: Done ---")


# Cell 5: Model Definition (2D U-Net) (No change needed)
print("\n--- Cell 5: Model Definition ---")

def conv_block(input_tensor, num_filters, kernel_size=(3, 3)):
    x = layers.Conv2D(num_filters, kernel_size, padding="same")(input_tensor); x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, kernel_size, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    return x
def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters); p = layers.MaxPooling2D((2, 2))(x); return x, p
def decoder_block(input_tensor, skip_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    x = layers.Concatenate()([x, skip_tensor]); x = conv_block(x, num_filters); return x

def build_unet(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE):
    print(f"Building U-Net -- Input: {input_shape}, Target Output: {output_shape}")
    inputs = keras.Input(shape=input_shape)
    s1, p1 = encoder_block(inputs, 64); s2, p2 = encoder_block(p1, 128); s3, p3 = encoder_block(p2, 256); s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512); d2 = decoder_block(d1, s3, 256); d3 = decoder_block(d2, s2, 128); d4 = decoder_block(d3, s1, 64)
    outputs_padded = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    def resize_layer(x, target_shape=output_shape): return tf.image.resize(x, [target_shape[0], target_shape[1]], method=tf.image.ResizeMethod.BILINEAR)
    outputs_resized = layers.Lambda(resize_layer, name='resize_to_output')(outputs_padded)
    final_outputs = layers.Reshape(output_shape, name='final_reshape')(outputs_resized)
    model = keras.Model(inputs=inputs, outputs=final_outputs, name="UNet_FWI_Padded_v3") # Incremented version name
    return model

# --- Build and Compile ---
tf.keras.backend.clear_session(); gc.collect()
model = build_unet()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
print("\nModel Summary:")
model.summary(line_length=120)

print("--- Cell 5: Done ---")


# Cell 6: Model Training (Adjusted Validation Handling)
print("\n--- Cell 6: Model Training ---")

# Check if generators are valid before attempting training
if train_generator is not None and len(train_generator) > 0:
    print(f"Starting training for up to {EPOCHS} epochs...")
    print(f"Training samples: {num_train_samples}, Validation samples: {num_val_samples}")
    start_time = time.time()

    # --- Callbacks ---
    callbacks = []
    # Save the best model
    # Note: If val_generator is None or empty, monitor='loss' (training loss) instead.
    monitor_metric = 'val_mae' if val_generator and len(val_generator) > 0 else 'mae'
    print(f"ModelCheckpoint and EarlyStopping will monitor: '{monitor_metric}'")

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_WEIGHTS_FILE, monitor=monitor_metric, save_best_only=True,
        save_weights_only=False, mode='min', verbose=1
    )
    callbacks.append(model_checkpoint)

    # Stop training early if no improvement
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor_metric, patience=10, # Increased patience slightly more
        restore_best_weights=True, mode='min', verbose=1
    )
    callbacks.append(early_stopping)

    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric, factor=0.2, patience=4, # Adjusted patience
        min_lr=1e-7, mode='min', verbose=1
    )
    callbacks.append(reduce_lr)

    # --- Determine Validation Data Argument ---
    validation_args = {}
    if val_generator is not None and len(val_generator) > 0:
         validation_args['validation_data'] = val_generator
    else:
         print("Validation generator is empty or None. Training without validation.")

    # --- Start Training ---
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        **validation_args # Pass validation_data only if available
    )

    # --- Post-Training ---
    end_time = time.time(); training_time = end_time - start_time
    print(f"\nTraining finished in {training_time / 60:.2f} minutes.")

    # --- Plot Training History ---
    if history and history.history:
        print("\nPlotting training history...")
        try:
            plt.figure(figsize=(12, 5))
            # Plot MAE/Loss
            plt.plot(history.history['mae'], label='Training MAE')
            if 'val_mae' in history.history:
                 plt.plot(history.history['val_mae'], label='Validation MAE')
                 best_epoch = np.argmin(history.history['val_mae'])
                 best_val_mae = np.min(history.history['val_mae'])
                 print(f"Best validation MAE = {best_val_mae:.5f} at epoch {best_epoch + 1}")
            else:
                 best_epoch = np.argmin(history.history['mae'])
                 best_train_mae = np.min(history.history['mae'])
                 print(f"Best training MAE = {best_train_mae:.5f} at epoch {best_epoch + 1} (No validation data)")

            plt.title('Mean Absolute Error (MAE)')
            plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.show(); plt.close()
        except Exception as plot_err: print(f"Error plotting history: {plot_err}")
    else: print("No training history available to plot.")

    # Best weights are restored by EarlyStopping if restore_best_weights=True

else:
    print("Skipping training: Training generator is empty or could not be created.")

gc.collect()
print("--- Cell 6: Done ---")


# Cell 7: Prediction on Test Set and Submission Generation (No change needed)
print("\n--- Cell 7: Prediction and Submission ---")

# --- Load the Best Trained Model ---
model_loaded = False
if os.path.exists(MODEL_WEIGHTS_FILE):
     print(f"Loading best model from {MODEL_WEIGHTS_FILE}...")
     try:
         # Load the entire model (including architecture and optimizer state)
         model = keras.models.load_model(MODEL_WEIGHTS_FILE)
         print("Model loaded successfully.")
         model_loaded = True
     except Exception as e:
         print(f"Error loading model: {e}. Cannot generate predictions.")
         # If model wasn't loaded from training cell, set it to None
         if 'model' not in locals() or model is None: model = None
elif 'model' in locals() and model is not None:
     print("Using model from the training cell (best weights should be restored by EarlyStopping).")
     model_loaded = True
else:
     print("Error: No trained model available (neither in memory nor saved file).")
     model = None

if model_loaded and actual_test_oids:
    print(f"Generating predictions for {len(actual_test_oids)} actual test files...")
    all_preds_dict = {}
    start_pred_time = time.time()

    for oid in tqdm(actual_test_oids, desc="Predicting Test Set"):
        test_file_path = os.path.join(TEST_PATH, f"{oid}.npy")
        pred_velocity = None # Ensure variable is defined
        try:
            test_data_raw = load_npy(test_file_path)
            preprocessed_test = preprocess_input(test_data_raw)
            if preprocessed_test is not None:
                pred_normalized = model.predict(np.expand_dims(preprocessed_test, axis=0), verbose=0)[0]
                pred_velocity = postprocess_prediction(pred_normalized)
            del test_data_raw, preprocessed_test, pred_normalized # Memory clean
        except Exception as e:
            print(f"Error during prediction for {oid}: {e}")
            pred_velocity = None # Ensure it's None on error

        # Store result (or None if failed)
        all_preds_dict[oid] = pred_velocity.astype(np.float32) if pred_velocity is not None and pred_velocity.shape == OUTPUT_SHAPE else None
        gc.collect() # Collect garbage more frequently during prediction

    end_pred_time = time.time()
    print(f"Prediction loop finished in {(end_pred_time - start_pred_time) / 60:.2f} minutes.")

    # --- Format predictions for submission ---
    print(f"\nFormatting predictions for the {len(test_oids_from_sub)} required submission oids...")
    submission_rows = []; default_value = 3000.0

    for oid in tqdm(test_oids_from_sub, desc="Formatting Submission"):
        velocity_map = all_preds_dict.get(oid, None)
        if velocity_map is not None:
            selected_velocity = velocity_map[:, SUBMISSION_X_INDICES]
            for y_pos in range(DEPTH):
                submission_rows.append([f"{oid}_y_{y_pos}"] + selected_velocity[y_pos, :].tolist())
        else: # Handle missing/failed predictions
            # print(f"Warning: Using default value for required oid {oid}.") # Reduce verbosity
            num_x_cols = len(SUBMISSION_X_INDICES)
            for y_pos in range(DEPTH):
                submission_rows.append([f"{oid}_y_{y_pos}"] + [default_value] * num_x_cols)

    # --- Create and Save Submission DataFrame ---
    if submission_rows:
        submission_df = pd.DataFrame(submission_rows, columns=[SUBMISSION_ID_COL] + SUBMISSION_X_COLS)
        print("\nFinal Submission DataFrame Info:")
        submission_df.info() # Check size and types
        # print("\nSubmission DataFrame Head:") # Reduce verbosity
        # print(submission_df.head())
        print(f"\nExpected submission rows: {expected_submission_rows}")
        print(f"Generated submission rows: {len(submission_df)}")
        if len(submission_df) == expected_submission_rows: print("Row count matches expected. OK.")
        else: print(f"CRITICAL WARNING: Row count ({len(submission_df)}) != expected ({expected_submission_rows}).")
        try:
            submission_df.to_csv(SUBMISSION_FILE, index=False)
            print(f"\nSubmission file saved successfully to: {SUBMISSION_FILE}")
        except Exception as e: print(f"Error saving submission file: {e}")
    else: print("Error: No submission rows generated.")
    del all_preds_dict; gc.collect()

elif not model_loaded: print("Skipping prediction: Model not available.")
elif not actual_test_oids: print("Skipping prediction: No actual test files found.")

print("--- Cell 7: Done ---")
print("\n--- End of Script ---")

