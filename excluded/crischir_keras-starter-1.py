import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path to the specific .npy file
# Ensure this path is correct for your Kaggle environment where the dataset is mounted.
file_path = '/kaggle/input/open-wfi-test/test/000039dca2.npy'

try:
    # Load the .npy file
    data = np.load(file_path)
    
    # Print the shape of the loaded data
    print(f"Shape of the sample data from {os.path.basename(file_path)}: {data.shape}")
    
    # Determine the image slice to plot
    img_to_plot = None
    if data.ndim == 3:
        # If the data is (channels, height, width) or (time_steps, depth, width),
        # take the first channel/slice for plotting.
        # This assumes the first dimension is features/channels/time-steps.
        img_to_plot = data[0, :, :]
        print(f"Plotting the first slice (index 0) from the 3D data. Slice shape: {img_to_plot.shape}")
    elif data.ndim == 2:
        # If the data is already 2-dimensional (height, width), plot it directly.
        img_to_plot = data
        print(f"Data is 2-dimensional. Plotting directly. Shape: {img_to_plot.shape}")
    else:
        print(f"Cannot plot data with {data.ndim} dimensions. Expected 2 or 3 dimensions for image visualization.")

    # Plot the image if a valid slice was extracted
    if img_to_plot is not None:
        plt.figure(figsize=(10, 6))
        # Use 'gray' colormap for intensity data, 'aspect='auto' to prevent stretching if dimensions are very different
        plt.imshow(img_to_plot, cmap='gray', aspect='auto') 
        plt.title(f"Sample from {os.path.basename(file_path)}\n(Slice 0, Shape: {img_to_plot.shape})")
        plt.colorbar(label='Amplitude')
        plt.xlabel('X (Width)')
        plt.ylabel('Y (Time/Depth)')
        plt.tight_layout()
        plt.show()
    else:
        print("No image could be generated for plotting.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the path is correct and the dataset is mounted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Define the path to the specific .npy file
# Ensure this path is correct for your Kaggle environment where the dataset is mounted.
file_path = '/kaggle/input/waveform-inversion/test/0000fd8ec8.npy'

try:
    # Load the .npy file
    data = np.load(file_path)
    
    # Print the shape of the loaded data
    print(f"Shape of the sample data from {os.path.basename(file_path)}: {data.shape}")
    
    # Determine the image slice to plot
    img_to_plot = None
    if data.ndim == 3:
        # If the data is (channels, height, width) or (time_steps, depth, width),
        # take the first channel/slice for plotting.
        # This assumes the first dimension is features/channels/time-steps.
        img_to_plot = data[0, :, :]
        print(f"Plotting the first slice (index 0) from the 3D data. Slice shape: {img_to_plot.shape}")
    elif data.ndim == 2:
        # If the data is already 2-dimensional (height, width), plot it directly.
        img_to_plot = data
        print(f"Data is 2-dimensional. Plotting directly. Shape: {img_to_plot.shape}")
    else:
        print(f"Cannot plot data with {data.ndim} dimensions. Expected 2 or 3 dimensions for image visualization.")

    # Plot the image if a valid slice was extracted
    if img_to_plot is not None:
        plt.figure(figsize=(10, 6))
        # Use 'gray' colormap for intensity data, 'aspect='auto' to prevent stretching if dimensions are very different
        plt.imshow(img_to_plot, cmap='gray', aspect='auto') 
        plt.title(f"Sample from {os.path.basename(file_path)}\n(Slice 0, Shape: {img_to_plot.shape})")
        plt.colorbar(label='Amplitude')
        plt.xlabel('X (Width)')
        plt.ylabel('Y (Time/Depth)')
        plt.tight_layout()
        plt.show()
    else:
        print("No image could be generated for plotting.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the path is correct and the dataset is mounted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm # Used for progress bar during initial metadata loading
import glob # For finding test files
import csv # For writing submission.csv
import time # For timing inference
import matplotlib.pyplot as plt # For plotting predictions

# --- Weights & Biases (WandB) Imports and Setup ---
import wandb
from kaggle_secrets import UserSecretsClient # Required for Kaggle environments to access secrets
from wandb.integration.keras import WandbCallback # Recommended for Keras integration with WandB

# --- Global Flags to Control Script Execution ---
# Set these to True or False to control which parts of the script run.
RUN_TRAIN = True  # Set to True to initialize and train the model on the training dataset.
RUN_VALID = True  # Set to True to initialize and evaluate the model on the validation dataset.
RUN_TEST  = True  # Set to True to initialize the test dataset, run inference, and plot predictions.

# --- Experimental Mode ---
# If True, a limited number of samples will be used for training, validation, and prediction
# to quickly check if the script is running correctly without processing the entire dataset.
EXPERIMENTAL_MODE = True
EXPERIMENTAL_SUBSAMPLE_LIMIT = 200 # Number of records/files to use in experimental mode

# --- Configuration Class ---
# A simple configuration class to hold dataset parameters.
# This mimics the 'cfg' object used in your original PyTorch dataset.
class Cfg:
    def __init__(self, data_dir, subsample=None, local_rank=0, samples_per_record=500, batch_size_val=4):
        """
        Initializes the configuration for the dataset.

        Args:
            data_dir (str): Path to the directory containing the actual .npy data and label files.
                            This should be the base directory where data_fpath/label_fpath in folds.csv are relative to.
            subsample (int, optional): If specified, limits the number of records (files) in the dataset.
                                       Defaults to None (no subsampling).
            local_rank (int): Used to control tqdm's progress bar visibility (only show for rank 0).
            samples_per_record (int): The number of individual samples contained within each .npy file.
                                      Derived from your original __len__ and __getitem__ logic (500).
            batch_size_val (int): Batch size for validation/test datasets.
        """
        self.data_dir = data_dir
        self.subsample = subsample
        self.local_rank = local_rank
        self.samples_per_record = samples_per_record
        self.batch_size_val = batch_size_val
        # Define a dummy device for compatibility, TF handles device placement
        self.device = tf.config.list_physical_devices('GPU')[0] if tf.config.list_physical_devices('GPU') else tf.config.list_physical_devices('CPU')[0]


# --- TensorFlow Training/Evaluation Dataset Adapter Class ---
class CustomTFDataset:
    def __init__(self, cfg, mode="train"):
        """
        Initializes the custom TensorFlow dataset adapter.

        Args:
            cfg (Cfg): Configuration object containing data_dir, subsample, local_rank, samples_per_record.
            mode (str): "train" or "eval". Determines data split and augmentation logic.
        """
        self.cfg = cfg
        self.mode = mode
        # Load file paths and other metadata (but not the actual numpy arrays)
        self.metadata_list = self._load_metadata()
        # Calculate the total number of samples across all selected files
        self.total_samples = len(self.metadata_list) * self.cfg.samples_per_record

        # Crucially, load one dummy sample to infer the expected shapes and data types
        # of the output tensors. This information is required by tf.py_function.
        print(f"Loading a dummy sample to determine data types and shapes for {self.mode} dataset...")
        # tf.constant(0, dtype=tf.int64) ensures the input type matches what tf.data.Dataset.range yields
        # Ensure that total_samples is at least 1 before trying to load a dummy sample
        if self.total_samples == 0:
            raise ValueError(f"No samples found in the {self.mode} dataset based on the provided configuration and folds.csv. "
                             "Please check your data_dir, folds.csv, and mode settings.")
        
        dummy_x, dummy_y = self._load_and_process_single_sample(tf.constant(0, dtype=tf.int64))
        self.x_dtype = dummy_x.dtype
        self.y_dtype = dummy_y.dtype
        self.x_shape = dummy_x.shape
        self.y_shape = dummy_y.shape
        print(f"Determined X shape: {self.x_shape}, X dtype: {self.x_dtype}")
        print(f"Determined Y shape: {self.y_shape}, Y dtype: {self.y_dtype}")

    def _load_metadata(self):
        """
        Loads the paths to data and label files from the folds.csv.
        This function performs the initial file system scan and filtering based on mode and subsample,
        but it does NOT load the actual numpy array data into memory.
        """
        # Explicitly set the full path to folds.csv as per your data organization
        folds_csv_path = "/kaggle/input/openfwi-preprocessed-72x72/folds.csv"
        if not os.path.exists(folds_csv_path):
            raise FileNotFoundError(f"Error: folds.csv not found at {folds_csv_path}. "
                                    f"Please ensure the path is correct and the file exists.")

        df = pd.read_csv(folds_csv_path)

        # Filter dataframe rows based on the dataset mode ("train" or "eval")
        if self.mode == "train":
            df = df[df["fold"] != 0] # Use folds other than 0 for training
        else:
            df = df[df["fold"] == 0] # Use fold 0 for evaluation/testing

        # Apply subsampling if specified in the config (limits total files after mode filter)
        if self.cfg.subsample is not None:
            df = df.head(self.cfg.subsample) # Take the first N files directly

        metadata_list = []
        # Use tqdm for a progress bar during initial metadata loading.
        # Disable tqdm if local_rank is not 0 (e.g., in a distributed training setup).
        for _, row in tqdm(df.iterrows(), total=len(df), disable=(self.cfg.local_rank != 0),
                           desc=f"Loading {self.mode} metadata"):
            row_dict = row.to_dict()
            metadata_list.append({
                # Construct full paths to data and label files using the base data_dir
                "data_fpath": os.path.join(self.cfg.data_dir, row_dict["data_fpath"]),
                "label_fpath": os.path.join(self.cfg.data_dir, row_dict["label_fpath"]),
                "dataset_name": row_dict["dataset"]
            })
        return metadata_list

    def _load_and_process_single_sample(self, global_idx_tensor):
        """
        Loads and processes a single sample given its global index.
        This function is intended to be wrapped by tf.py_function.
        It performs actual numpy file loading, slicing, and augmentation.

        Args:
            global_idx_tensor (tf.Tensor): The global index of the sample, as a TensorFlow tensor.

        Returns:
            tuple: A tuple containing the processed data (x) and label (y) as tf.Tensor.
        """
        # Convert the TensorFlow tensor to a numpy integer for Python indexing
        global_idx = global_idx_tensor.numpy().item() # .item() extracts scalar from 0-dim array

        # Calculate the row (file) and column (sample within file) index
        row_idx = global_idx // self.cfg.samples_per_record
        col_idx = global_idx % self.cfg.samples_per_record

        # Retrieve file paths from the pre-loaded metadata list
        record_metadata = self.metadata_list[row_idx]
        data_fpath = record_metadata["data_fpath"]
        label_fpath = record_metadata["label_fpath"]

        # Determine the memory map mode based on the dataset mode.
        # 'r' is read-only memory map; None means load fully into memory (often for smaller files).
        mmap_mode = "r" if self.mode == "train" else None
        
        try:
            # Load the full numpy arrays. Using mmap_mode efficiently handles large files
            # by mapping them directly into memory without reading the entire content.
            arr = np.load(data_fpath, mmap_mode=mmap_mode)
            lbl = np.load(label_fpath, mmap_mode=mmap_mode)
        except Exception as e:
            # Handle potential errors during file loading (e.g., file not found, corruption)
            print(f"Error loading numpy file: {e}. Data file: {data_fpath}, Label file: {label_fpath}")
            # In a real scenario, you might log this, skip the sample, or return dummy data.
            # Raising the exception here will stop the dataset pipeline if a file is missing/corrupt.
            raise e

        # Extract the specific sample using the column index
        # '...' ensures that all other dimensions are included
        x = arr[col_idx, ...]
        y = lbl[col_idx, ...]

        # Apply augmentations only when in training mode
        if self.mode == "train":
            # Temporal flip augmentation:
            # Flips the first dimension (time) and the last spatial dimension (width) for x.
            # Flips the last spatial dimension (width) for y.
            if np.random.random() < 0.5:
                x = x[::-1, :, ::-1] # Example: if x is (Channels, Time, Width) or (Time, Height, Width)
                y = y[..., ::-1]     # Example: if y is (Height, Width) or (Height, Width, Channels)

        # It's crucial to make a copy of the numpy arrays.
        # Memory-mapped arrays are views, and if not copied, subsequent TensorFlow operations
        # or Python garbage collection might cause issues.
        x = x.copy()
        y = y.copy()
        
        # Convert the numpy arrays to TensorFlow Tensors and ensure they have the desired dtype.
        # Assuming data and labels are floating point. If they are integers, adjust dtype accordingly.
        return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)

    def create_tf_dataset(self):
        """
        Creates and returns a tf.data.Dataset object for the custom dataset.

        Returns:
            tf.data.Dataset: A TensorFlow dataset ready for training or evaluation.
        """
        # Create a dataset that yields sequential global indices from 0 to total_samples - 1
        indices_ds = tf.data.Dataset.range(self.total_samples)

        # Map the Python processing function (_load_and_process_single_sample) to each index.
        # tf.py_function allows arbitrary Python code to run within the TF graph.
        # Tout specifies the output types of the Python function (inferred during initialization).
        dataset = indices_ds.map(
            lambda idx: tf.py_function(
                self._load_and_process_single_sample, # The Python function to execute
                inp=[idx],                            # Input tensors to the Python function
                Tout=[self.x_dtype, self.y_dtype]     # Expected output types
            ),
            num_parallel_calls=tf.data.AUTOTUNE # Automatically determines optimal number of parallel calls
        )

        # Re-enabled tf.ensure_shape to provide static shape information for graph compilation.
        dataset = dataset.map(
            lambda x, y: (tf.ensure_shape(x, self.x_shape), tf.ensure_shape(y, self.y_shape)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle the dataset only for training mode to ensure randomness in batches
        if self.mode == "train":
            # The buffer size determines how many elements from the dataset are buffered
            # for shuffling. A larger buffer leads to better shuffling but uses more memory.
            # Common heuristic: a few thousand or min(self.total_samples, 10000).
            shuffle_buffer_size = min(self.total_samples, 10000) 
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
        
        # As explicitly requested: "the dataset is huge so no cache is needed".
        # The .cache() operation is intentionally omitted.

        # Prefetch data to overlap data preprocessing with model execution (GPU/TPU training).
        # This significantly improves pipeline throughput.
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

# --- TensorFlow Test (Prediction) Dataset Adapter Class ---
class CustomTFTestDataset:
    def __init__(self, cfg, test_files):
        """
        Initializes the custom TensorFlow test dataset adapter for prediction.

        Args:
            cfg (Cfg): Configuration object.
            test_files (list): List of paths to individual test .npy files.
        """
        self.cfg = cfg
        self.test_files = test_files
        self.total_samples = len(self.test_files)

        print(f"Loading a dummy sample to determine data types and shapes for test dataset...")
        if self.total_samples == 0:
            raise ValueError("No test files found. Please check `test_files` path.")
        
        # Load one dummy sample to infer the expected shapes and data types
        # Input to _load_single_test_sample is a tf.string tensor (file path)
        dummy_x, dummy_oid = self._load_single_test_sample(tf.constant(self.test_files[0], dtype=tf.string))
        self.x_dtype = dummy_x.dtype
        # IMPORTANT FIX: The output shape of _preprocess_test_sample is (channels, 72, 72)
        # So, self.x_shape must reflect this.
        self.x_shape = dummy_x.shape 
        # OID is a string (stem), its shape will be scalar
        self.oid_dtype = dummy_oid.dtype 
        self.oid_shape = tf.TensorShape([]) 
        print(f"Determined X shape (after preprocessing in dataset): {self.x_shape}, X dtype: {self.x_dtype}")
        print(f"Determined OID shape: {self.oid_shape}, OID dtype: {self.oid_dtype}")


    def _preprocess_test_sample(self, x):
        """
        Applies preprocessing similar to your PyTorch _preprocess function,
        but using TensorFlow operations.
        Input x is expected to be (channels, height, width).
        """
        # Permute to (height, width, channels) for tf.image.resize
        x_permuted = tf.transpose(x, perm=[1, 2, 0]) # (H, W, C)

        # Interpolate to (70, 70) using AREA method
        # tf.image.resize expects (height, width)
        x_interpolated = tf.image.resize(
            x_permuted, size=(70, 70), method=tf.image.ResizeMethod.AREA
        )

        # Pad with 1 pixel on all sides, replicate mode
        # tf.pad expects [[top, bottom], [left, right], [channel_pad_top, channel_pad_bottom]]
        # For replicate mode, we use 'REFLECT' which is often a good substitute for 'replicate' at edges in TF.
        # This padding will make the 70x70 image into a 72x72 image.
        x_padded = tf.pad(x_interpolated, [[1, 1], [1, 1], [0, 0]], mode='REFLECT') # (70+1+1, 70+1+1, C) = (72, 72, C)

        # Permute back to (channels, height, width) as the model input expects channels-first
        x_final = tf.transpose(x_padded, perm=[2, 0, 1]) # (C, 72, 72)
        
        return x_final

    def _load_single_test_sample(self, file_path_tensor):
        """
        Loads a single test sample and its OID (stem) given its file path.
        This function is intended to be wrapped by tf.py_function.
        It now includes the preprocessing step.

        Args:
            file_path_tensor (tf.Tensor): The path to the test .npy file, as a TensorFlow string tensor.

        Returns:
            tuple: A tuple containing the processed data (x) as tf.Tensor and the OID (stem) as tf.Tensor (string).
        """
        # Convert the TensorFlow string tensor to a Python string
        file_path = file_path_tensor.numpy().decode('utf-8')

        try:
            # Load the numpy array from the file
            x_np = np.load(file_path)
            # Convert to TensorFlow tensor for preprocessing
            x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
        except Exception as e:
            print(f"Error loading test numpy file: {e}. File: {file_path}")
            raise e

        # Apply preprocessing
        x_processed = self._preprocess_test_sample(x_tf)

        # Extract the stem (OID) from the file path
        test_stem = os.path.basename(file_path).split(".")[0]

        # Convert stem to TensorFlow Tensor
        return x_processed, tf.convert_to_tensor(test_stem, dtype=tf.string)

    def create_tf_dataset(self):
        """
        Creates and returns a tf.data.Dataset object for the test dataset.

        Returns:
            tf.data.Dataset: A TensorFlow dataset ready for prediction.
        """
        # Create a dataset from the list of test file paths
        file_paths_ds = tf.data.Dataset.from_tensor_slices(self.test_files)

        # Map the Python processing function (_load_single_test_sample) to each file path.
        dataset = file_paths_ds.map(
            lambda fp: tf.py_function(
                self._load_single_test_sample, # The Python function to execute
                inp=[fp],                     # Input tensor (file path)
                Tout=[self.x_dtype, self.oid_dtype] # Expected output types (x, oid)
            ),
            num_parallel_calls=tf.data.AUTOTUNE # Automatically determines optimal number of parallel calls
        )

        # Re-enabled tf.ensure_shape to provide static shape information for graph compilation.
        dataset = dataset.map(
            lambda x, oid: (tf.ensure_shape(x, self.x_shape), tf.ensure_shape(oid, self.oid_shape)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # No shuffling for test/prediction dataset
        # Prefetch data to overlap data preprocessing with model execution
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

# --- Helper function for formatting time ---
def format_time(seconds):
    """Formats a duration in seconds into H:MM:SS string."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# --- Custom Keras Layer for Noise Band Augmentation ---
class AddNoiseBands(tf.keras.layers.Layer):
    """
    A Keras layer to add horizontal and/or vertical noise bands to input images.
    Intended for data augmentation during training.
    """
    def __init__(self, horizontal_bands_prob=0.3, vertical_bands_prob=0.3,
                 max_band_width_h_ratio=0.05, max_band_width_v_ratio=0.05,
                 max_noise_intensity=0.1, name=None, **kwargs):
        super(AddNoiseBands, self).__init__(name=name, **kwargs)
        self.horizontal_bands_prob = tf.constant(horizontal_bands_prob, dtype=tf.float32)
        self.vertical_bands_prob = tf.constant(vertical_bands_prob, dtype=tf.float32)
        self.max_band_width_h_ratio = tf.constant(max_band_width_h_ratio, dtype=tf.float32)
        self.max_band_width_v_ratio = tf.constant(max_band_width_v_ratio, dtype=tf.float32)
        self.max_noise_intensity = tf.constant(max_noise_intensity, dtype=tf.float32)

    @tf.function
    def call(self, inputs, training=None):
        # Only apply augmentation during training
        if training is False:
            return inputs
        if training is None:
            # If `training` is None, assume training mode.
            # This is common when the layer is used directly in a `model.call` or Sequential model
            # where the `training` argument is propagated from `model.fit`.
            pass 

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3] # Assuming channels-last format here

        current_output = inputs

        def add_horizontal_band_fn():
            # Calculate band width, cast height to float32 first for ratio calculation
            band_width_h = tf.cast(tf.cast(height, tf.float32) * self.max_band_width_h_ratio, dtype=tf.int32)
            band_width_h = tf.maximum(1, tf.minimum(band_width_h, height - 1)) # Ensure min 1 pixel and not exceed height

            start_row_h = tf.random.uniform([], minval=0, maxval=height - band_width_h + 1, dtype=tf.int32)

            noise_value_h_per_channel = tf.random.uniform((1, 1, 1, channels), # Noise applies per channel
                                                           minval=-self.max_noise_intensity,
                                                           maxval=self.max_noise_intensity)

            noise_tensor = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=1.0, dtype=inputs.dtype) * noise_value_h_per_channel
            noise_tensor = tf.clip_by_value(noise_tensor, -self.max_noise_intensity, self.max_noise_intensity)

            # XLA-COMPATIBLE MASKING FOR HORIZONTAL BAND
            row_indices = tf.range(height, dtype=tf.int32)
            # Create a boolean mask for rows within the band
            is_in_band_h = tf.logical_and(row_indices >= start_row_h,
                                        row_indices < start_row_h + band_width_h)
            # Reshape and tile to match input tensor dimensions (batch, height, width, channels)
            is_in_band_h = tf.reshape(is_in_band_h, (1, height, 1, 1))
            is_in_band_h = tf.tile(is_in_band_h, [batch_size, 1, width, channels])

            # Use tf.where to apply ones where the condition is true, zeros otherwise
            band_mask_applied = tf.where(is_in_band_h, tf.ones_like(inputs, dtype=inputs.dtype), tf.zeros_like(inputs, dtype=inputs.dtype))

            return current_output + noise_tensor * band_mask_applied

        def add_vertical_band_fn():
            # Calculate band width, cast width to float32 first for ratio calculation
            band_width_v = tf.cast(tf.cast(width, tf.float32) * self.max_band_width_v_ratio, dtype=tf.int32)
            band_width_v = tf.maximum(1, tf.minimum(band_width_v, width - 1)) # Ensure min 1 pixel and not exceed width

            start_col_v = tf.random.uniform([], minval=0, maxval=width - band_width_v + 1, dtype=tf.int32)

            noise_value_v_per_channel = tf.random.uniform((1, 1, 1, channels), # Noise applies per channel
                                                           minval=-self.max_noise_intensity,
                                                           maxval=self.max_noise_intensity)

            noise_tensor = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=1.0, dtype=inputs.dtype) * noise_value_v_per_channel
            noise_tensor = tf.clip_by_value(noise_tensor, -self.max_noise_intensity, self.max_noise_intensity)

            # XLA-COMPATIBLE MASKING FOR VERTICAL BAND
            col_indices = tf.range(width, dtype=tf.int32)
            # Create a boolean mask for columns within the band
            is_in_band_v = tf.logical_and(col_indices >= start_col_v,
                                        col_indices < start_col_v + band_width_v)
            # Reshape and tile to match input tensor dimensions (batch, height, width, channels)
            is_in_band_v = tf.reshape(is_in_band_v, (1, 1, width, 1))
            is_in_band_v = tf.tile(is_in_band_v, [batch_size, height, 1, channels])

            # Use tf.where to apply ones where the condition is true, zeros otherwise
            band_mask_applied = tf.where(is_in_band_v, tf.ones_like(inputs, dtype=inputs.dtype), tf.zeros_like(inputs, dtype=inputs.dtype))

            return current_output + noise_tensor * band_mask_applied

        # Randomly apply horizontal and vertical bands with specified probabilities
        current_output = tf.cond(tf.random.uniform(()) < self.horizontal_bands_prob,
                                 add_horizontal_band_fn,
                                 lambda: current_output)

        current_output = tf.cond(tf.random.uniform(()) < self.vertical_bands_prob,
                                 add_vertical_band_fn,
                                 lambda: current_output)

        return current_output

    def get_config(self):
        config = super(AddNoiseBands, self).get_config()
        config.update({
            'horizontal_bands_prob': self.horizontal_bands_prob.numpy(),
            'vertical_bands_prob': self.vertical_bands_prob.numpy(),
            'max_band_width_h_ratio': self.max_band_width_h_ratio.numpy(),
            'max_band_width_v_ratio': self.max_band_width_v_ratio.numpy(),
            'max_noise_intensity': self.max_noise_intensity.numpy(),
        })
        return config

# --- Model Definition Function ---
def build_model(input_shape, output_shape, model_name="seismic_model"):
    """
    Builds the TensorFlow Keras model based on the provided architecture,
    adapting to the specific input_shape.

    Args:
        input_shape (tuple): Expected input shape for the model (e.g., (5, 1000, 70) or (5, 72, 72)).
        output_shape (tuple): Expected output shape for the model (e.g., (1, 70, 70)).
        model_name (str): Name for the Keras model.

    Returns:
        tf.keras.Model: Compiled TensorFlow Keras model.
    """
    from tensorflow.keras.applications import EfficientNetB2, MobileNetV2
    from tensorflow.keras import layers, models

    input_channels = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]
    
    print(f"\n--- Building Model '{model_name}' with Input Shape: {input_shape} ---")

    # Input layer: raw seismic data in channels-first format.
    inputs = layers.Input(shape=input_shape, name='seismic_input') # (None, C, H, W)

    # --- BRANCH 1: First 3 channels with EfficientNetB2 ---
    print(f"--- Building Branch 1 (Channels 0, 1, 2) with EfficientNetB2 for {model_name} ---")
    x1 = layers.Lambda(lambda tensor: tensor[:, :3, :, :], name='select_3_channels_b1')(inputs) # (None, 3, H, W)
    x1 = layers.Permute((2, 3, 1), name='permute_input_b1')(x1) # (None, H, W, 3)
    
    # Adaptive Pooling and Cropping based on input_height
    if input_height == 1000 and input_width == 70:
        # Original logic for (5, 1000, 70) data
        print(f"  Branch 1: Applying pooling (14,1) and cropping ((1,0), (0,0)) for 1000-height input.")
        x1 = layers.AveragePooling2D(pool_size=(14, 1), padding='valid', name='time_pooling_b1')(x1) # (None, 71, 70, 3)
        x1 = layers.Cropping2D(cropping=((1, 0), (0, 0)), name='crop_time_b1')(x1) # (None, 70, 70, 3)
    elif input_height == 72 and input_width == 72:
        # Logic for (5, 72, 72) data - directly crop to 70x70
        print(f"  Branch 1: Applying cropping ((1,1), (1,1)) for 72x72 input.")
        x1 = layers.Cropping2D(cropping=((1, 1), (1, 1)), name='crop_72x72_b1')(x1) # (None, 70, 70, 3)
    else:
        raise ValueError(f"Unsupported input spatial dimensions for Branch 1 in {model_name}: ({input_height}, {input_width}). "
                         "Model expects (5, 1000, 70) or (5, 72, 72).")

    x1 = layers.Resizing(224, 224, name='resize_for_cnn_encoder_b1')(x1) # (None, 224, 224, 3)
    
    # Add Noise Bands for augmentation during training
    x1 = AddNoiseBands(
        horizontal_bands_prob=0.5, vertical_bands_prob=0.2,
        max_band_width_h_ratio=0.01, max_band_width_v_ratio=0.01,
        max_noise_intensity=0.2, name='noise_augmentation_b1')(x1)
    
    efficientnet_b2_model = EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3), 
        pooling=None # We'll add our own pooling layer later
    )
    efficientnet_b2_model.trainable = True # Fine-tune the EfficientNet model
    x1 = efficientnet_b2_model(x1)
    x1 = layers.GlobalAveragePooling2D(name='global_avg_pooling_b1')(x1)

    # --- BRANCH 2: Remaining 2 channels with MobileNetV2 ---
    print(f"\n--- Building Branch 2 (Channels 3, 4) with MobileNetV2 for {model_name} ---")
    x2 = layers.Lambda(lambda tensor: tensor[:, 3:, :, :], name='select_2_channels_b2')(inputs) # (None, 2, H, W)
    
    # MobileNetV2 expects 3 channels. Duplicate the last channel to make it 3-channel.
    x2 = layers.Lambda(lambda t: tf.concat([t, t[:, -1:, :, :]], axis=1), name='duplicate_channel_to_3d_b2')(x2) # (None, 3, H, W)
    
    x2 = layers.Permute((2, 3, 1), name='permute_input_b2')(x2) # (None, H, W, 3)
    
    # Adaptive Pooling and Cropping based on input_height
    if input_height == 1000 and input_width == 70:
        # Logic for (5, 1000, 70) data
        print(f"  Branch 2: Applying pooling (14,1) and cropping ((1,0), (0,0)) for 1000-height input.")
        x2 = layers.AveragePooling2D(pool_size=(14, 1), padding='valid', name='time_pooling_b2')(x2) # (None, 71, 70, 3)
        x2 = layers.Cropping2D(cropping=((1, 0), (0, 0)), name='crop_time_b2')(x2) # (None, 70, 70, 3)
    elif input_height == 72 and input_width == 72:
        # Logic for (5, 72, 72) data - directly crop to 70x70
        print(f"  Branch 2: Applying cropping ((1,1), (1,1)) for 72x72 input.")
        x2 = layers.Cropping2D(cropping=((1, 1), (1, 1)), name='crop_72x72_b2')(x2) # (None, 70, 70, 3)
    else:
        # This error should have been caught for Branch 1 already, but included for robustness
        raise ValueError(f"Unsupported input spatial dimensions for Branch 2 in {model_name}: ({input_height}, {input_width}). "
                         "Model expects (5, 1000, 70) or (5, 72, 72).")

    x2 = layers.Resizing(224, 224, name='resize_for_cnn_encoder_b2')(x2) # (None, 224, 224, 3)
    
    # Add Noise Bands for augmentation during training
    x2 = AddNoiseBands(
        horizontal_bands_prob=0.5, vertical_bands_prob=0.2,
        max_band_width_h_ratio=0.01, max_band_width_v_ratio=0.01,
        max_noise_intensity=0.2, name='noise_augmentation_b2')(x2)
    
    mobilenet_v2_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling=None # We'll add our own pooling layer later
    )
    mobilenet_v2_model.trainable = True # Fine-tune the MobileNetV2 model
    x2 = mobilenet_v2_model(x2)
    x2 = layers.GlobalAveragePooling2D(name='global_avg_pooling_b2')(x2)

    # --- Feature Assembly / Combination ---
    print("\n--- Assembling Features ---")
    # Concatenate the features from both branches
    combined_features = layers.Concatenate(axis=-1, name='combine_features')([x1, x2])

    # --- Custom Head (now taking combined features) ---
    print("\n--- Building Custom Head ---")
    x = layers.Dense(512, activation='gelu', name='dense_head_1')(combined_features)
    x = layers.Dropout(0.5, name='dropout_head_1')(x)
    # x = layers.Dense(256, activation='gelu', name='dense_head_2')(x) # Optional layer, commented out as in your code
    # x = layers.Dropout(0.5, name='dropout_head_2')(x) # Optional layer, commented out as in your code
    
    # The Dense layer's output units must match the total number of elements
    # in the target output_shape (C * H * W).
    x = layers.Dense(np.prod(output_shape), activation='linear', name='dense_output')(x)
    outputs = layers.Reshape(output_shape, name='reshape_output')(x)

    # Build the model.
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # --- Compilation ---
    # For seismic velocity prediction (regression), Mean Absolute Error (MAE) or
    # Mean Squared Error (MSE) are common choices. MAE is more robust to outliers.
    # MSE penalizes larger errors more heavily.
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    
    model.summary()
    return model


if __name__ == '__main__':
    # --- WandB Initialization ---
    # This block retrieves your WandB API key from Kaggle secrets
    # and initializes a new WandB run.
    try:
        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_api")
        wandb.login(key=secret_value_0)
        
        # Define default config for WandB, can be overridden by sweep or command line
        wandb_config_defaults = {
            "learning_rate": 0.001,
            "epochs": 7,
            "batch_size_per_replica": 8, # Changed to per_replica batch size
            "batch_size_val_per_replica": 4, # Changed to per_replica batch size
            "subsample": None # Default to no subsampling
        }
        if EXPERIMENTAL_MODE:
            wandb_config_defaults["subsample"] = EXPERIMENTAL_SUBSAMPLE_LIMIT

        wandb.init(project="seismic-velocity-adapted-dataset", entity="crischir", 
                   config=wandb_config_defaults)
        print("WandB initialized successfully!")
    except Exception as e:
        print(f"Error initializing WandB: {e}. Please ensure 'wandb_api' secret is set on Kaggle.")
        wandb.init(mode="disabled") # Disable wandb if login fails to allow script to continue
        print("WandB is disabled. Script will continue without WandB logging.")

    # --- Setup for Multi-GPU Distribution Strategy ---
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        print(f"Detected {len(gpus)} GPUs. Using MirroredStrategy for distributed training.")
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("Detected 1 or no GPU. Using default strategy.")
        # If no GPUs, it will use CPU. If 1 GPU, it will use OneDeviceStrategy on that GPU.
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0" if gpus else "/cpu:0")
    
    num_replicas = strategy.num_replicas_in_sync
    print(f"Number of replicas (devices in sync): {num_replicas}")

    # --- Configuration for your real data ---
    # Set data_dir to the base directory where your .npy files are located.
    # The paths in folds.csv (e.g., "data/data_A_f0.npy") will be joined with this data_dir.
    REAL_DATA_DIR = "/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72/"
    # Corrected path to the directory containing test .npy files
    REAL_TEST_DIR = "/kaggle/input/waveform-inversion/test/" 

    # Initialize configuration object using WandB's config if available
    cfg = Cfg(
        data_dir=REAL_DATA_DIR,
        subsample=wandb.config.subsample if wandb.run and "subsample" in wandb.config else None, 
        local_rank=0,        # Show tqdm progress bar
        samples_per_record=500, # This must match the number of samples stored in each .npy file
        # Batch sizes are per-replica from WandB, but cfg stores the global batch size.
        # This will be overridden later with the global batch size.
        batch_size_val=wandb.config.batch_size_val_per_replica if wandb.run and "batch_size_val_per_replica" in wandb.config else 4,
    )
    # Override subsample if in experimental mode (explicitly setting to EXPERIMENTAL_SUBSAMPLE_LIMIT)
    if EXPERIMENTAL_MODE:
        cfg.subsample = EXPERIMENTAL_SUBSAMPLE_LIMIT

    # Calculate global batch sizes
    GLOBAL_BATCH_SIZE_TRAIN = (wandb.config.batch_size_per_replica if wandb.run and "batch_size_per_replica" in wandb.config else 8) * num_replicas
    GLOBAL_BATCH_SIZE_VAL = (wandb.config.batch_size_val_per_replica if wandb.run and "batch_size_val_per_replica" in wandb.config else 4) * num_replicas

    # --- Infer Dataset Shapes for Model Building ---
    # Infer shapes from the training dataset. This shape will be the input to the 'train_val_model'.
    print("\n--- Inferring Dataset Shapes for Model Building (using training data) ---")
    inferred_x_shape_train = None
    inferred_y_shape_train = None

    try:
        # Use a minimal subsample to infer shapes quickly without loading too much data
        temp_config_train_infer = Cfg(data_dir=REAL_DATA_DIR, subsample=1, local_rank=0, samples_per_record=500)
        temp_train_dataset_adapter = CustomTFDataset(temp_config_train_infer, mode="train")
        inferred_x_shape_train = temp_train_dataset_adapter.x_shape
        inferred_y_shape_train = temp_train_dataset_adapter.y_shape
        print(f"Inferred input (X) shape from TRAIN dataset: {inferred_x_shape_train}")
        print(f"Inferred output (Y) shape from TRAIN dataset: {inferred_y_shape_train}")
        
        # Infer shapes for test data after preprocessing to get the input shape for 'test_model'.
        # The test files themselves have varying shapes, but _preprocess_test_sample transforms them to (5, 72, 72).
        temp_test_files_infer = sorted(glob.glob(os.path.join(REAL_TEST_DIR, "*.npy")))
        if not temp_test_files_infer:
            print(f"Warning: No test files found for shape inference in {REAL_TEST_DIR}. Cannot infer test model shape.")
            inferred_x_shape_test = None 
        else:
            temp_test_config_infer = Cfg(data_dir=REAL_DATA_DIR, subsample=1, local_rank=0, samples_per_record=500) 
            temp_test_dataset_adapter_infer = CustomTFTestDataset(temp_test_config_infer, [temp_test_files_infer[0]])
            inferred_x_shape_test = temp_test_dataset_adapter_infer.x_shape # This should be (5, 72, 72)
            print(f"Inferred input (X) shape from TEST dataset (after preprocessing): {inferred_x_shape_test}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not infer dataset shapes. {e}")
        if wandb.run:
            wandb.finish()
        exit()

    train_val_model = None
    test_model = None

    # Build and compile models within the distribution strategy scope
    with strategy.scope():
        try:
            if inferred_x_shape_train and inferred_y_shape_train:
                print("\n--- Building Train/Validation Model ---")
                train_val_model = build_model(input_shape=inferred_x_shape_train, output_shape=inferred_y_shape_train, model_name="train_val_model")
            else:
                print("Skipping train/validation model build due to missing shape inference.")

            if inferred_x_shape_test and inferred_y_shape_train: # Test model output is same as train Y
                print("\n--- Building Test Model ---")
                test_model = build_model(input_shape=inferred_x_shape_test, output_shape=inferred_y_shape_train, model_name="test_model")
            else:
                print("Skipping test model build due to missing shape inference.")

        except Exception as e:
            print(f"\nCRITICAL ERROR: Could not build model(s). {e}")
            if wandb.run:
                wandb.finish()
            exit()


    # 1. Create the training dataset and run training
    if RUN_TRAIN and train_val_model:
        print("\n--- Initializing Training Dataset ---")
        try:
            train_dataset_adapter = CustomTFDataset(cfg, mode="train")
            tf_train_dataset = train_dataset_adapter.create_tf_dataset()

            # Distribute the dataset across replicas
            dist_train_dataset = strategy.experimental_distribute_dataset(tf_train_dataset.batch(GLOBAL_BATCH_SIZE_TRAIN))

            print(f"\n--- Running Training Loop with WandB Callback (Global Batch Size: {GLOBAL_BATCH_SIZE_TRAIN}) ---")
            history = train_val_model.fit( # Use the train_val_model for training
                dist_train_dataset,
                epochs=wandb.config.epochs if wandb.run and "epochs" in wandb.config else 5, # Use epochs from wandb.config
                callbacks=[WandbCallback(save_graph=False, save_model=False)], # Integrate WandB callback
                verbose=1 # Show training progress
            )
            print("Training complete. Check WandB dashboard for logs.")

            # Save the trained model for later use
            model_save_path = "seismic_velocity_model.keras" # Recommended Keras format for TF 2.x
            print(f"Saving the model to: {model_save_path}")
            train_val_model.save(model_save_path) # Save the training model
            print("Model saved successfully.")

        except Exception as e:
            print(f"\nError initializing or training model: {e}")
            print("Please ensure the training data paths, model, and setup are correct.")


    # 2. Create the evaluation dataset and run evaluation
    if RUN_VALID and train_val_model:
        print("\n--- Initializing Evaluation Dataset ---")
        try:
            eval_dataset_adapter = CustomTFDataset(cfg, mode="eval")
            tf_eval_dataset = eval_dataset_adapter.create_tf_dataset()

            # Distribute the dataset across replicas
            dist_eval_dataset = strategy.experimental_distribute_dataset(tf_eval_dataset.batch(GLOBAL_BATCH_SIZE_VAL))

            print(f"\n--- Running Evaluation on Validation Dataset (Global Batch Size: {GLOBAL_BATCH_SIZE_VAL}) ---")
            # Evaluate the trained model
            evaluation_results = train_val_model.evaluate(dist_eval_dataset, verbose=1) # Use train_val_model for eval
            print(f"Validation Loss: {evaluation_results[0]:.4f}, Validation MAE: {evaluation_results[1]:.4f}")
            
            # Log evaluation metrics to WandB
            if wandb.run:
                wandb.log({"val_loss": evaluation_results[0], "val_mae": evaluation_results[1]})
                print("Validation metrics logged to WandB.")

        except Exception as e:
            print(f"\nError initializing or evaluating model: {e}")
            print("Please ensure the evaluation data paths and file contents are correct.")


    # 3. Create the Test (Prediction) Dataset and run inference
    if RUN_TEST and test_model: # Ensure test_model is built
        print("\n--- Initializing Test Dataset and Running Inference ---")
        row_count = 0
        t0 = time.time()
        
        # Find all .npy files in the test directory
        test_files_full = sorted(glob.glob(os.path.join(REAL_TEST_DIR, "*.npy")))
        
        # Apply experimental subsample limit if enabled
        if EXPERIMENTAL_MODE and EXPERIMENTAL_SUBSAMPLE_LIMIT is not None:
            test_files = test_files_full[:EXPERIMENTAL_SUBSAMPLE_LIMIT]
            print(f"EXPERIMENTAL_MODE: Limiting test files to {len(test_files)} out of {len(test_files_full)}")
        else:
            test_files = test_files_full

        if not test_files:
            print(f"Warning: No test files found in {REAL_TEST_DIR} after applying filters. Skipping inference.")
        else:
            # Column names for the submission CSV, adapted from your original
            x_cols = [f"x_{i}" for i in range(1, 70, 2)]
            # Store integer indices corresponding to x_cols for easier numpy slicing
            x_col_indices = [int(col.split('_')[1]) for col in x_cols] 
            fieldnames = ["oid_ypos"] + x_cols
            
            try:
                # The CustomTFTestDataset will now preprocess the input 'x' to (C, 72, 72)
                # This aligns with the model's expected input shape (C, H, W) where H=72, W=72 after its internal permute.
                test_dataset_adapter = CustomTFTestDataset(cfg, test_files)
                # Batching test dataset using GLOBAL_BATCH_SIZE_VAL (same as validation)
                # Also distribute the test dataset for consistency with distributed strategy
                tf_test_dataset = test_dataset_adapter.create_tf_dataset()
                dist_test_dataset = strategy.experimental_distribute_dataset(tf_test_dataset.batch(GLOBAL_BATCH_SIZE_VAL))

                # Open the submission CSV file for writing
                with open("submission.csv", "wt", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    print(f"Starting inference on {len(test_files)} test files...")
                    
                    # Collect all OIDs upfront to ensure proper pairing with predictions
                    all_oids = []
                    # Create a new non-batched dataset just for collecting OIDs in order
                    # Need to explicitly call create_tf_dataset again to get a fresh iterator
                    for _, oid_tensor in tqdm(test_dataset_adapter.create_tf_dataset(), total=test_dataset_adapter.total_samples, desc="Collecting OIDs"):
                        all_oids.append(oid_tensor.numpy().decode('utf-8'))
                    
                    # Perform prediction with the TensorFlow model directly on the distributed dataset
                    # model.predict handles the distribution and gathering internally.
                    all_outputs = test_model.predict(dist_test_dataset, verbose=1) # verbose=1 to see progress

                    # Ensure all_outputs and all_oids have the same length
                    if len(all_outputs) != len(all_oids):
                        raise ValueError(f"Mismatch between number of predictions ({len(all_outputs)}) and OIDs collected ({len(all_oids)}).")

                    # Store a few outputs and oids for later plotting
                    plot_y_preds = []
                    plot_oids_test = []
                    max_plots = 15 # Max number of plots (3x5 grid)

                    # Iterate through the predictions and OIDs
                    for i in tqdm(range(len(all_outputs)), desc="Writing Submission and Collecting Plots"):
                        y_pred_single_batched = all_outputs[i] # This is (1, 70, 70)
                        oid_test = all_oids[i]

                        # Collect samples for plotting from the first few
                        if len(plot_y_preds) < max_plots:
                            plot_y_preds.append(y_pred_single_batched[0, :, :]) # Squeeze channel dim for plotting
                            plot_oids_test.append(oid_test)

                        # Iterate through y_pos (0 to 69) and select specific x_pos for CSV
                        y_pred_single = np.squeeze(y_pred_single_batched, axis=0) # (70, 70)
                        for y_pos in range(70):
                            row_values = y_pred_single[y_pos, x_col_indices] 
                            row = dict(zip(x_cols, row_values))
                            row["oid_ypos"] = f"{oid_test}_y_{y_pos}"
                    
                            writer.writerow(row)
                            row_count += 1

                            # Clear buffer periodically
                            if row_count % 100_000 == 0:
                                csvfile.flush()
                
                t1 = format_time(time.time() - t0)
                print(f"Inference complete. Total rows written: {row_count}")
                print(f"Inference Time: {t1}")

                # --- Plotting predictions ---
                if plot_y_preds: # Only plot if we collected any samples
                    print("\n--- Plotting a few predicted samples ---")
                    fig, axes = plt.subplots(3, 5, figsize=(15, 9)) # Adjust figsize for better view
                    axes= axes.flatten()

                    # Use the collected samples for plotting
                    n = min(len(plot_y_preds), len(axes)) 
                    
                    for i in range(n):
                        img = plot_y_preds[i] # y_preds is (70, 70)
                        idx = plot_oids_test[i] # Get the OID for the title
                    
                        # Plot
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(idx, fontsize=8) # Reduce font size if titles are long
                        axes[i].axis('off')

                    # Turn off any unused subplots
                    for i in range(n, len(axes)):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                else:
                    print("No samples collected for plotting. Ensure there are test files and RUN_TEST is True.")

            except Exception as e:
                print(f"\nError during test dataset initialization or inference: {e}")
                print("Please ensure the test data paths, model, and output logic are correct.")
    
    # Finish the WandB run if it was initialized
    if wandb.run:
        wandb.finish()






