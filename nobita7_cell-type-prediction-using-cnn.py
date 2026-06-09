import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# train_test_split is not used here due to slide-based split
# from sklearn.model_selection import train_test_split
# StandardScaler is optional, uncomment if needed
# from sklearn.preprocessing import StandardScaler
import cv2 # OpenCV for patch extraction
import os
import gc
from tqdm import tqdm
import time # Added for timestamp in model saving
import traceback # For detailed error printing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration & Hyperparameters ---
H5_FILE_PATH = '/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5'
OUTPUT_SUBMISSION_PATH = 'submission.csv'
# Add timestamp to model path to avoid overwriting during experiments
MODEL_SAVE_PATH = f'best_custom_cnn_model_{time.strftime("%Y%m%d_%H%M%S")}.keras'

# Model Hyperparameters (STARTING POINTS - REQUIRES TUNING)
PATCH_SIZE = 128       # Size of the square image patch (e.g., 64, 128, 256)
BATCH_SIZE = 64
EPOCHS = 200          # Adjust based on convergence and early stopping
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3     # Regularization strength
N_CELL_TYPES = 35     # From problem description

# Validation Strategy (Using Leave-One-Slide-Out)
VALIDATION_SLIDE = 'S_6'
# Assuming S_1 to S_6 are the training slides as per original description
ALL_TRAIN_SLIDES = ['S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6']
TRAIN_SLIDES = [s for s in ALL_TRAIN_SLIDES if s != VALIDATION_SLIDE]

TEST_SLIDE = 'S_7' # Assuming S_7 is the single test slide

# --- Data Loading Functions ---

def load_data_from_h5(h5_path):
    """
    Loads images and spot data from the HDF5 file using specified paths.
    Correctly handles spot data stored as a single HDF5 Dataset per slide.
    """
    data = {'Train': {'images': {}, 'spots': {}},
            'Test': {'images': {}, 'spots': {}}}
    try:
        with h5py.File(h5_path, 'r') as f:
            # Define base paths
            train_img_base = 'images/Train'
            train_spots_base = 'spots/Train'
            test_img_base = 'images/Test'
            test_spots_base = 'spots/Test'

            # Check if base paths exist
            if train_img_base not in f or train_spots_base not in f or \
               test_img_base not in f or test_spots_base not in f:
                raise ValueError(f"HDF5 file structure does not match expected paths. Missing one of: {train_img_base}, {train_spots_base}, {test_img_base}, {test_spots_base}")

            # Load Training Data
            train_image_group = f[train_img_base]
            train_spots_group = f[train_spots_base]
            train_slide_ids = list(train_image_group.keys()) # Get slide IDs from images
            print(f"Found training slides: {train_slide_ids}")
            for slide_id in train_slide_ids:
                if slide_id in train_spots_group and slide_id in train_image_group: # Check both image and spots exist
                    print(f"Loading train slide: {slide_id}")
                    data['Train']['images'][slide_id] = np.array(train_image_group[slide_id])

                    # --- FIX for Spot Data Loading ---
                    # Load the entire dataset for spots
                    spots_dataset = train_spots_group[slide_id]
                    # Convert the HDF5 dataset directly to a pandas DataFrame
                    # This works if the dataset has a compound dtype (structured array)
                    try:
                         spots_array = np.array(spots_dataset)
                         # Check if it's a structured array, pandas handles this directly
                         if spots_array.dtype.names:
                              data['Train']['spots'][slide_id] = pd.DataFrame(spots_array)
                              print(f"  Loaded spots for {slide_id} as structured array. Columns: {list(data['Train']['spots'][slide_id].columns)}")
                         else:
                              # If not structured, assume columns are in order [x, y, cell_type_0, ..., cell_type_34]
                              print(f"  Warning: spots for {slide_id} is not a structured array. Assuming column order.")
                              num_cols = spots_array.shape[1]
                              if num_cols == 2 + N_CELL_TYPES: # Check if number of columns matches expectation
                                   col_names = ['x', 'y'] + [f'cell_type_{i}' for i in range(N_CELL_TYPES)]
                                   data['Train']['spots'][slide_id] = pd.DataFrame(spots_array, columns=col_names)
                              else:
                                   raise ValueError(f"spots dataset for {slide_id} has unexpected shape {spots_array.shape} and is not structured.")

                    except Exception as e_spots:
                        print(f"  Error processing spots dataset for {slide_id}: {e_spots}. Skipping slide.")
                        # Remove potentially partially loaded data for this slide
                        if slide_id in data['Train']['images']: del data['Train']['images'][slide_id]
                        if slide_id in data['Train']['spots']: del data['Train']['spots'][slide_id]
                        continue # Skip to next slide
                    # --- End FIX ---

                else:
                    print(f"Warning: Image or spots data not found for training slide {slide_id}. Skipping.")


            # Load Test Data
            test_image_group = f[test_img_base]
            test_spots_group = f[test_spots_base]
            test_slide_ids = list(test_image_group.keys()) # Get slide IDs from images
            print(f"Found test slides: {test_slide_ids}")
            for slide_id in test_slide_ids:
                 if slide_id == TEST_SLIDE: # Process only the expected test slide
                    if slide_id in test_spots_group and slide_id in test_image_group:
                        print(f"Loading test slide: {slide_id}")
                        data['Test']['images'][slide_id] = np.array(test_image_group[slide_id])

                        # --- FIX for Spot Data Loading ---
                        spots_dataset = test_spots_group[slide_id]
                        try:
                             spots_array = np.array(spots_dataset)
                             if spots_array.dtype.names:
                                  data['Test']['spots'][slide_id] = pd.DataFrame(spots_array)
                                  print(f"  Loaded spots for {slide_id} as structured array. Columns: {list(data['Test']['spots'][slide_id].columns)}")
                             else:
                                 print(f"  Warning: spots for {slide_id} is not a structured array. Assuming column order (only x, y expected).")
                                 num_cols = spots_array.shape[1]
                                 if num_cols >= 2: # Test set might only have x, y
                                       col_names = ['x', 'y'] + [f'unknown_{i}' for i in range(num_cols - 2)] # Placeholder names
                                       data['Test']['spots'][slide_id] = pd.DataFrame(spots_array, columns=col_names[:num_cols])
                                       # Ensure 'x' and 'y' exist, regardless of structured array or not
                                       if 'x' not in data['Test']['spots'][slide_id].columns or 'y' not in data['Test']['spots'][slide_id].columns:
                                           raise ValueError(f"Could not find 'x' or 'y' columns in test spots for {slide_id}")
                                 else:
                                       raise ValueError(f"Test spots dataset for {slide_id} has unexpected shape {spots_array.shape}.")

                        except Exception as e_spots:
                            print(f"  Error processing spots dataset for {slide_id}: {e_spots}. Cannot proceed with test data.")
                            return None # Fatal error if test data cannot be loaded
                        # --- End FIX ---
                    else:
                        print(f"Warning: Image or spots data not found for test slide {slide_id}. Cannot proceed.")
                        return None # Fatal error if test data cannot be loaded
                 elif slide_id in test_image_group.keys(): # Check if it's just an unexpected slide ID
                      print(f"Skipping unexpected test slide found in HDF5: {slide_id}")

            # Final check if essential data is present
            if not data['Train']['images'] or not data['Train']['spots']:
                 print("Error: No training data loaded successfully.")
                 return None
            if TEST_SLIDE not in data['Test']['images'] or TEST_SLIDE not in data['Test']['spots']:
                 print(f"Error: Test data for slide {TEST_SLIDE} not loaded successfully.")
                 return None


            print("Data loaded successfully.")
            return data # Return the populated dictionary

    except Exception as e:
        print(f"Error loading data from HDF5 file: {e}")
        traceback.print_exc() # Print detailed traceback
        return None # Return None on failure

def extract_patch(image, x, y, patch_size):
    """Extracts a patch centered at (x, y) from the image."""
    # Ensure coordinates are finite numbers
    if not np.isfinite(x) or not np.isfinite(y):
        # print(f"Warning: Invalid coordinates (x={x}, y={y}). Returning None.")
        return None # Indicate failure

    img_h, img_w = image.shape[:2]
    half_patch = patch_size // 2

    # Calculate patch boundaries (adjusting for integer coordinates)
    y_center, x_center = int(round(y)), int(round(x))

    # Ensure center coordinates are within reasonable bounds (slightly flexible)
    if not (-patch_size < y_center < img_h + patch_size and -patch_size < x_center < img_w + patch_size):
        # print(f"Warning: Coordinates (x={x_center}, y={y_center}) seem too far outside image bounds ({img_w}x{img_h}). Returning None.")
        return None # Indicate failure if coordinates are wildly off

    y_start = max(0, y_center - half_patch)
    y_end = min(img_h, y_center + (patch_size - half_patch)) # Adjusted end calculation
    x_start = max(0, x_center - half_patch)
    x_end = min(img_w, x_center + (patch_size - half_patch)) # Adjusted end calculation

    patch = image[y_start:y_end, x_start:x_end]

    # Check if patch extraction resulted in an empty array (e.g., coordinates way outside)
    if patch.size == 0:
        # print(f"Warning: Patch extraction resulted in empty array for coords (x={x}, y={y}). Returning None.")
        return None # Indicate failure

    # Calculate padding needed based on desired size vs actual extracted size
    current_h, current_w = patch.shape[:2]
    pad_top_needed = max(0, (patch_size - current_h) // 2)
    pad_bottom_needed = max(0, patch_size - current_h - pad_top_needed)
    pad_left_needed = max(0, (patch_size - current_w) // 2)
    pad_right_needed = max(0, patch_size - current_w - pad_left_needed)

    if pad_top_needed > 0 or pad_bottom_needed > 0 or pad_left_needed > 0 or pad_right_needed > 0:
        # Using edge padding (reflect) is often better than constant padding for natural images
        patch = cv2.copyMakeBorder(patch, pad_top_needed, pad_bottom_needed, pad_left_needed, pad_right_needed, cv2.BORDER_REFLECT_101)

    # Final check to ensure the patch is exactly the target size (resize as fallback for rare issues)
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
         # print(f"Warning: Patch resize needed for coords (x={x}, y={y}). Original patch shape: ({current_h}, {current_w}), Padded shape: {patch.shape[:2]}")
         patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    return patch

def create_dataset(slide_ids, data_dict, patch_size, is_train=True):
    """
    Creates dataset of patches and labels (if training) for the given slide IDs.
    Uses data loaded into the dictionary structure. Handles potential patch extraction failures.
    """
    all_patches = []
    all_labels = [] if is_train else None
    valid_indices = [] # Store original indices of spots that were successfully processed

    data_source = 'Train' if is_train else 'Test'
    print(f"Processing {data_source} slides: {slide_ids}")

    for slide_id in tqdm(slide_ids):
        if slide_id not in data_dict['images'] or slide_id not in data_dict['spots']:
            print(f"Warning: Missing image or spot data for {data_source} slide {slide_id}. Skipping.")
            continue

        image = data_dict['images'][slide_id]
        spots_df = data_dict['spots'][slide_id]

        # Identify label columns if training
        label_cols = []
        if is_train:
            # Use lowercase comparison for robustness, check if required cols exist
            if 'x' not in spots_df.columns or 'y' not in spots_df.columns:
                 print(f"Error: Missing 'x' or 'y' column in training spots for slide {slide_id}. Skipping slide.")
                 continue
            potential_labels = [col for col in spots_df.columns if col.lower() not in ['x', 'y']]
            if len(potential_labels) != N_CELL_TYPES:
                 print(f"Warning: Found {len(potential_labels)} potential label columns for train slide {slide_id}, expected {N_CELL_TYPES}. Using them anyway.")
                 # You might want to add stricter checking here depending on expected column names
            label_cols = potential_labels
            if not label_cols:
                 print(f"Error: Could not identify label columns for training slide {slide_id}. Skipping slide.")
                 continue
        else:
             # For test, ensure 'x' and 'y' exist
             if 'x' not in spots_df.columns or 'y' not in spots_df.columns:
                  print(f"Error: Missing 'x' or 'y' column in test spots for slide {slide_id}. Cannot proceed.")
                  return None, None # Fatal error for test set


        for index, spot_row in spots_df.iterrows():
            x, y = spot_row['x'], spot_row['y']

            # Extract patch, handle potential failure (returns None)
            patch = extract_patch(image, x, y, patch_size)
            if patch is None:
                print(f"Warning: Failed to extract patch for spot index {index} in slide {slide_id} (coords x={x}, y={y}). Skipping spot.")
                continue # Skip this spot

            # Check for NaN labels BEFORE adding patch/label
            if is_train:
                labels = spot_row[label_cols].values.astype(np.float32)
                if np.isnan(labels).any():
                    print(f"Warning: Skipping spot index {index} in train slide {slide_id} due to NaN labels.")
                    continue # Skip this spot
                all_labels.append(labels)

            # If all checks passed, add the patch and store the original index
            all_patches.append(patch)
            valid_indices.append(index) # Store original index of the spot we successfully processed


        # Memory optimization
        del image
        del spots_df
        gc.collect()

    if not all_patches: # Check if any patches were successfully created
         print(f"Error: No valid patches could be extracted for slides: {slide_ids}. Cannot create dataset.")
         return None, None if is_train else None # Return None based on train/test context


    print("Converting data to numpy arrays...")
    # Ensure patches have 3 channels even if grayscale (unlikely for H&E)
    temp_patches = np.array(all_patches, dtype=np.uint8) # Load as uint8 first
    if temp_patches.ndim == 3: # Grayscale patches (N, H, W)
         X = np.stack([temp_patches]*3, axis=-1) # Repeat channel to make (N, H, W, 3)
    elif temp_patches.ndim == 4 and temp_patches.shape[-1] == 1: # Grayscale with channel dim (N, H, W, 1)
         X = np.repeat(temp_patches, 3, axis=-1)
    elif temp_patches.ndim == 4 and temp_patches.shape[-1] == 3: # Already RGB
         X = temp_patches
    else:
         raise ValueError(f"Unexpected patch array shape: {temp_patches.shape}")

    X = X.astype(np.float32) / 255.0 # Normalize pixels to [0, 1]


    if is_train:
        if not all_labels: # Check if any labels were collected (handles case where all spots had NaN labels)
            print(f"Error: No valid labels found for training slides: {slide_ids}.")
            return None, None
        y = np.array(all_labels, dtype=np.float32)
        print(f"Created dataset: X shape={X.shape}, y shape={y.shape}")
        # Sanity check shapes
        if X.shape[0] != y.shape[0]:
             raise ValueError(f"Mismatch between number of patches ({X.shape[0]}) and labels ({y.shape[0]})")
        return X, y # Return valid indices as well? Not currently needed for training.
    else:
        print(f"Created test dataset: X shape={X.shape}")
        # Return original indices corresponding to the patches in X
        # These indices refer to the original DataFrame for the processed test slide
        return X, valid_indices


# --- Model Definition ---
def build_custom_cnn(input_shape, num_classes, dropout_rate):
    """Builds the custom CNN model."""
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 4 (Optional deeper block)
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 4 (Optional deeper block)
    model.add(layers.Conv2D(512, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 4 (Optional deeper block)
    model.add(layers.Conv2D(1024, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.GlobalAveragePooling2D())
    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(512)) # Another dense layer
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax')) # Linear activation for regression

    return model





if __name__ == "__main__":
    # Configure GPU memory growth if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU memory growth could not be set: {e}")
    else:
        print("No GPU detected, using CPU.")


    # 1. Load Data
    print("\n--- Loading Data ---")
    all_data = load_data_from_h5(H5_FILE_PATH)
    # --- FIX: Check if all_data is None after loading ---
    if all_data is None:
        print("Failed to load data. Exiting.")
        exit()
    # --- End FIX ---

    # Check if slides needed for train/val exist (Corrected 'images' to 'images')
    # --- FIX: Check presence in the actual loaded data dict ---
    available_train_slides = list(all_data['Train']['images'].keys())
    required_train_slides_present = all(s in available_train_slides for s in TRAIN_SLIDES)
    validation_slide_present = VALIDATION_SLIDE in available_train_slides

    if not required_train_slides_present or not validation_slide_present:
           print("Error: Not all required training/validation slides found in loaded data.")
           print(f"Required train slides: {TRAIN_SLIDES}")
           print(f"Required validation slide: {VALIDATION_SLIDE}")
           print(f"Available train slides in HDF5: {available_train_slides}")
           exit()
    
    # --- Training Phase ---
    print("\n--- Preparing Training Data ---")
    X_train_data, y_train_data = create_dataset(TRAIN_SLIDES, all_data['Train'], PATCH_SIZE, is_train=True)

    print("\n--- Preparing Validation Data ---")
    X_val_data, y_val_data = create_dataset([VALIDATION_SLIDE], all_data['Train'], PATCH_SIZE, is_train=True)

    # --- FIX: Check if dataset creation failed ---
    if X_train_data is None or y_train_data is None or X_val_data is None or y_val_data is None:
        print("Failed to create training or validation datasets (check warnings above). Exiting.")
        exit()
    # --- End FIX ---
    # --- Main Execution ---
    #with tf.name_scope("custom_cnn_scope"):
       # model = build_custom_cnn(input_shape, num_classes, dropout_rate)
    # --- End FIX ---
    datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

    # Fit on your training data
    datagen.fit(X_train_data)
    print("\n--- Building Model ---")
    input_shape = (PATCH_SIZE, PATCH_SIZE, 3)
    model = build_custom_cnn(input_shape, N_CELL_TYPES, DROPOUT_RATE)
    model.summary()

    print("\n--- Compiling Model ---")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae']) # Mean Squared Error loss

    print("\n--- Setting up Callbacks ---")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )

    print("\n--- Starting Training ---")
    # Clean up memory before training
    # Keep all_data for test phase if memory allows, otherwise reload later
    # Let's try keeping it for now, assuming moderate memory usage after patch creation
    # del all_data['Train'] # Optionally delete only train part
    gc.collect()

    history = model.fit(
        X_train_data, y_train_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_data, y_val_data),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )

    print("\n--- Training Complete ---")

    # --- Prediction Phase ---
    # Optional: Reload data if deleted earlier for memory saving
    # if 'Test' not in all_data:
    #     print("\n--- Reloading Data for Test Set ---")
    #     all_data = load_data_from_h5(H5_FILE_PATH)
    #     if all_data is None:
    #         print("Failed to reload data for test set. Exiting.")
    #         exit()

    # Check if test slide exists in the currently loaded data
    if TEST_SLIDE not in all_data['Test']['images'] or TEST_SLIDE not in all_data['Test']['spots']:
         print(f"Error: Test slide {TEST_SLIDE} not found in loaded data for prediction.")
         exit()

    print("\n--- Preparing Test Data ---")
    # Pass the 'Test' part of the dictionary
    X_test, test_valid_indices = create_dataset([TEST_SLIDE], all_data['Test'], PATCH_SIZE, is_train=False)

    # --- FIX: Check if test dataset creation failimagesed ---
    if X_test is None:
        print("Failed to create test dataset (check warnings above). Exiting.")
        exit()
    # --- End FIX ---

    print("\n--- Making Predictions ---")
    predictions = model.predict(X_test, batch_size=BATCH_SIZE)

    print(f"Predictions shape: {predictions.shape}") # Should be (N_valid_test_spots, 35)

    # --- Submission File Generation ---
    print("\n--- Generating Submission File ---")
    if predictions.shape[0] == 0:
         print("Error: No predictions were generated (likely no valid test spots found). Cannot create submission file.")
    else:
        # Using the exact code provided by the user:
        submission_df = pd.DataFrame(predictions, columns=[f'cell_type_{i}' for i in range(N_CELL_TYPES)])
        # Generate simple sequential IDs starting from 0, matching the number of predictions made
        submission_df.insert(0, 'ID', range(len(submission_df)))
        submission_df.to_csv(OUTPUT_SUBMISSION_PATH, index=False)

        # Add a check for the expected number of rows (2088) based on original prompt info
        # Note: The number of rows will now be the number of *successfully processed* test spots.
        print(f"Submission file generated with {len(submission_df)} rows.")
        expected_rows = 2088 # From original description
        original_test_spot_count = len(all_data['Test']['spots'][TEST_SLIDE])
        if len(submission_df) != original_test_spot_count:
             print(f"Warning: Original test slide {TEST_SLIDE} had {original_test_spot_count} spots, but submission has {len(submission_df)} rows (due to potential patch extraction failures or NaN coordinates).")
        if len(submission_df) != expected_rows:
             print(f"Warning: Submission file has {len(submission_df)} rows, but the original description mentioned {expected_rows} were expected. This might be due to excluded spots or different test set size.")

        print(f"Submission file saved to {OUTPUT_SUBMISSION_PATH}")

    print("\n--- Script Finished ---")

