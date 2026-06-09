# CODE CELL 1: Imports and Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import random
import gc
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim # Import SSIM

# Set seeds for reproducibility - crucial for academic reporting
def set_seed(seed=42):
    """Sets random seeds for major libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Optional: Configure TensorFlow to be deterministic (may impact performance)
    # tf.config.experimental.enable_op_determinism() 

set_seed(42)

# Enable mixed precision for potentially faster training on compatible GPUs
# Note: May slightly affect numerical precision, monitor validation loss closely.
try:
    mixed_precision = tf.keras.mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled.")
except Exception as e:
    print(f"Could not enable mixed precision: {e}")

# Data directories (adjust path if necessary)
INPUT_DIR = '/kaggle/input/waveform-inversion'
TRAIN_DIR = os.path.join(INPUT_DIR, 'train_samples')
TEST_DIR = os.path.join(INPUT_DIR, 'test')

# Display available training datasets
print("\nAvailable training datasets:")
if os.path.exists(TRAIN_DIR):
    for item in sorted(os.listdir(TRAIN_DIR)):
        if os.path.isdir(os.path.join(TRAIN_DIR, item)):
            print(f"- {item}")
else:
    print(f"Training directory not found: {TRAIN_DIR}")


# CODE CELL 2: Data Loading Functions
def load_training_data(dataset_family, sample_limit=None):
    """Loads training data (seismic gathers and velocity models) for a specific family."""
    data_dir = os.path.join(TRAIN_DIR, dataset_family)
    data_path = os.path.join(data_dir, 'data', '*.npy')
    model_path = os.path.join(data_dir, 'model', '*.npy')
    
    data_files = sorted(glob.glob(data_path))
    model_files = sorted(glob.glob(model_path))
    
    if not data_files or not model_files:
        print(f"Warning: No data or model files found in {dataset_family} under expected paths.")
        return None, None
        
    if len(data_files) != len(model_files):
        print(f"Warning: Mismatch in number of data ({len(data_files)}) and model ({len(model_files)}) files in {dataset_family}. Skipping.")
        return None, None
        
    X_list = []
    y_list = []
    
    file_pairs = list(zip(data_files, model_files))
    if sample_limit and sample_limit < len(file_pairs):
        # Select a deterministic subset if sample_limit is used
        indices = np.linspace(0, len(file_pairs) - 1, sample_limit, dtype=int)
        file_pairs = [file_pairs[i] for i in indices]
        print(f"Limiting to {len(file_pairs)} samples for {dataset_family}.")

    print(f"Loading data from {len(file_pairs)} file pairs in {dataset_family}...")
    for data_file, model_file in tqdm(file_pairs, desc=f"Loading {dataset_family}", leave=False):
        try:
            # Consider adding error handling for corrupted files
            X = np.load(data_file)
            y = np.load(model_file)
            
            # Basic validation of shapes (optional but recommended)
            if X.ndim != 4 or y.ndim != 3:
                 print(f"Warning: Unexpected dimensions in {data_file} (X shape: {X.shape}) or {model_file} (y shape: {y.shape}). Skipping file pair.")
                 continue

            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            print(f"Error loading file pair: {data_file}, {model_file}. Error: {e}")

    if not X_list or not y_list:
        print(f"No valid data loaded for {dataset_family}.")
        return None, None

    X = np.concatenate(X_list, axis=0).astype(np.float32) # Ensure float32 for TF
    y = np.concatenate(y_list, axis=0).astype(np.float32) # Ensure float32 for TF
    
    # Free memory
    del X_list, y_list
    gc.collect()
    
    return X, y

def load_test_data():
    """Loads test seismic data keyed by object ID (oid)."""
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*.npy')))
    if not test_files:
        print(f"Warning: No test files found in {TEST_DIR}")
        return {}, []
        
    test_data = {}
    oids = []
    
    print(f"Loading {len(test_files)} test files...")
    for test_file in tqdm(test_files, desc="Loading test data", leave=False):
        try:
            oid = os.path.basename(test_file).split('.')[0]
            data = np.load(test_file).astype(np.float32) # Ensure float32
             # Basic validation
            if data.ndim != 4:
                print(f"Warning: Unexpected dimensions in test file {test_file} (shape: {data.shape}). Skipping.")
                continue
            test_data[oid] = data
            oids.append(oid)
        except Exception as e:
            print(f"Error loading test file: {test_file}. Error: {e}")
            
    return test_data, oids


# CODE CELL 3: Visualization and Preprocessing Functions
def plot_seismic_and_velocity(seismic_data, velocity_map, sample_idx=0, title_prefix=""):
    """Plots a seismic shot gather and the corresponding velocity map."""
    if sample_idx >= seismic_data.shape[0]:
        print(f"Error: sample_idx {sample_idx} out of bounds for data with shape {seismic_data.shape}")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Seismic Data Plot ---
    num_sources = seismic_data.shape[1]
    source_idx = num_sources // 2  # Visualize the middle source gather
    seismic_gather = seismic_data[sample_idx, source_idx]
    
    # Determine appropriate color limits for seismic data
    clim_abs = np.percentile(np.abs(seismic_gather), 98) # Use 98th percentile for robust limits
    
    im_seismic = axes[0].imshow(seismic_gather, aspect='auto', cmap='seismic', 
                                vmin=-clim_abs, vmax=clim_abs)
    axes[0].set_title(f'{title_prefix}Seismic Data (Sample {sample_idx}, Source {source_idx})')
    axes[0].set_xlabel('Receiver Index')
    axes[0].set_ylabel('Time Sample Index')
    plt.colorbar(im_seismic, ax=axes[0], label='Amplitude')
    
    # --- Velocity Map Plot ---
    vel_map_sample = velocity_map[sample_idx]
    im_velocity = axes[1].imshow(vel_map_sample, cmap='viridis', aspect='auto',
                                 vmin=np.min(vel_map_sample), vmax=np.max(vel_map_sample)) # Use actual range
    axes[1].set_title(f'{title_prefix}Ground Truth Velocity Map (Sample {sample_idx})')
    axes[1].set_xlabel('Horizontal Position Index (X)')
    axes[1].set_ylabel('Depth Position Index (Y)')
    plt.colorbar(im_velocity, ax=axes[1], label='Velocity (m/s)')
    
    plt.tight_layout()
    plt.show()

def preprocess_seismic_batch(seismic_batch):
    """
    Applies sample-wise normalization to a batch of seismic data.
    Input shape: (batch, num_sources, time_steps, num_receivers)
    Output shape: (batch, num_sources, time_steps, num_receivers)
    """
    batch_size, num_sources, time_steps, num_receivers = seismic_batch.shape
    # Process in float32 for precision during normalization
    processed_batch = seismic_batch.astype(np.float32) 
    
    epsilon = 1e-8 # Small constant for numerical stability

    for i in range(batch_size):
        for j in range(num_sources):
            data_slice = processed_batch[i, j] # Shape (time_steps, num_receivers)
            mean = np.mean(data_slice)
            std = np.std(data_slice)
            if std > epsilon:
                processed_batch[i, j] = (data_slice - mean) / std
            else:
                # Handle constant or near-constant slices (avoid division by zero)
                processed_batch[i, j] = data_slice - mean # Just center it
                
    return processed_batch

def apply_output_constraints(velocity_maps, min_velocity=1500.0, smoothing_sigma=0.5):
    """
    Applies physics-based constraints to predicted velocity maps:
    1. Enforces minimum velocity.
    2. Applies gentle Gaussian smoothing.
    Input shape: (batch, height, width)
    Output shape: (batch, height, width)
    """
    constrained_maps = velocity_maps.copy()
    
    # 1. Minimum Velocity Constraint
    constrained_maps = np.maximum(constrained_maps, min_velocity)
    
    # 2. Gaussian Smoothing (applied per map in the batch)
    if smoothing_sigma is not None and smoothing_sigma > 0:
        for i in range(constrained_maps.shape[0]):
            constrained_maps[i] = gaussian_filter(constrained_maps[i], sigma=smoothing_sigma)
            
    return constrained_maps


# CODE CELL 4: Physics-Guided U-Net Implementation
def build_physics_guided_unet(input_shape, output_shape, base_filters=64, depth=4, 
                                kernel_size=3, pool_size=(2, 2), 
                                use_batch_norm=True, dropout_rate=0.0, # Added dropout option
                                smoothness_weight=0.05, min_velocity=1500.0):
    """Builds a U-Net model with configurable depth and physics-guided components."""
    
    # --- Loss Function Definition ---
    def physics_guided_loss(y_true, y_pred):
        """Custom loss: MAE + Smoothness Regularization."""
        # Ensure calculations are in float32 for stability if using mixed precision
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Calculate spatial gradients (difference between adjacent pixels)
        # Note: tf.image.image_gradients returns derivatives in y, x order
        dy, dx = tf.image.image_gradients(y_pred) 
        
        # L1 norm of gradients encourages sparsity (sharp edges) while promoting smoothness
        smoothness_loss = tf.reduce_mean(tf.abs(dy)) + tf.reduce_mean(tf.abs(dx))
        
        total_loss = mae_loss + smoothness_weight * smoothness_loss
        return total_loss

    # --- U-Net Building Blocks ---
    def conv_block(inputs, filters, kernel_size=kernel_size, padding='same', 
                   use_batch_norm=use_batch_norm, activation='leaky_relu', dropout=dropout_rate):
        """Standard convolutional block for U-Net."""
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(inputs)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        if activation == 'leaky_relu':
             x = layers.LeakyReLU(alpha=0.2)(x)
        else:
             x = layers.Activation(activation)(x)
        if dropout > 0:
             x = layers.Dropout(dropout)(x)

        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        if activation == 'leaky_relu':
             x = layers.LeakyReLU(alpha=0.2)(x)
        else:
             x = layers.Activation(activation)(x)
        if dropout > 0:
             x = layers.Dropout(dropout)(x)
        return x

    def encoder_block(inputs, filters):
        """Encoder block: ConvBlock + MaxPooling."""
        conv = conv_block(inputs, filters)
        pool = layers.MaxPooling2D(pool_size=pool_size)(conv)
        return conv, pool # Return conv output for skip connection

    def decoder_block(inputs, skip_connection, filters):
        """Decoder block: Upsample -> Concatenate -> ConvBlock."""
        # Upsampling using Transposed Convolution
        up = layers.Conv2DTranspose(filters, kernel_size=pool_size, strides=pool_size, padding='same')(inputs)
        
        # Concatenate skip connection
        # Ensure skip connection shape matches upsampled shape if padding='valid' was used in encoder
        concat = layers.Concatenate()([up, skip_connection])
        
        conv = conv_block(concat, filters)
        return conv

    # --- Model Construction ---
    inputs = keras.Input(shape=input_shape)
    
    # Placeholder for potential future physics-informed input layers
    current_layer = inputs 
    
    skip_connections = []
    filters = base_filters

    # Encoder Path
    print("Building Encoder...")
    for _ in range(depth):
        print(f"  Depth {_ + 1}, Filters: {filters}")
        conv, pool = encoder_block(current_layer, filters)
        skip_connections.append(conv)
        current_layer = pool
        filters *= 2 
        
    # Bottleneck
    print(f"Building Bottleneck, Filters: {filters}")
    bridge = conv_block(current_layer, filters)
    current_layer = bridge
    
    # Decoder Path
    print("Building Decoder...")
    for i in range(depth):
        filters //= 2
        print(f"  Depth {depth - i}, Filters: {filters}")
        skip = skip_connections[depth - 1 - i]
        current_layer = decoder_block(current_layer, skip, filters)

    # Output Layer
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='linear')(current_layer) 
    # Reshape to match target velocity map shape (H, W)
    # The Conv2D output might have shape (H, W, 1), so Reshape removes the channel dim.
    outputs = layers.Reshape(output_shape, name="raw_output")(outputs) 
    
    # Apply Minimum Velocity Constraint
    # Using a Lambda layer ensures this constraint is part of the model graph
    outputs = layers.Lambda(lambda x: tf.maximum(x, min_velocity), name="constrained_output")(outputs)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"PhysicsGuided_UNet_Depth{depth}")
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=1e-3) # Initial learning rate
    # If using mixed precision, wrap the optimizer
    if mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
    model.compile(optimizer=optimizer, 
                  loss=physics_guided_loss, 
                  metrics=[keras.metrics.MeanAbsoluteError(name='mae')]) # Track standard MAE

    return model


# CODE CELL 5: ResNet-style Model Implementation
def build_resnet_style_model(input_shape, output_shape, base_filters=64, num_blocks_per_stage=[2, 2, 2, 2],
                               kernel_size=3, use_batch_norm=True, min_velocity=1500.0):
    """Builds a ResNet-style encoder-decoder model for inversion."""

    def residual_block(x, filters, kernel_size=kernel_size, stride=1, 
                       use_batch_norm=use_batch_norm, activation='leaky_relu'):
        """A standard residual block."""
        shortcut = x # Store the input for the shortcut connection
        
        # First convolutional layer in the block
        conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(x)
        if use_batch_norm:
            conv1 = layers.BatchNormalization()(conv1)
        if activation == 'leaky_relu':
             conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
        else:
             conv1 = layers.Activation(activation)(conv1)
             
        # Second convolutional layer in the block
        conv2 = layers.Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal')(conv1)
        if use_batch_norm:
            conv2 = layers.BatchNormalization()(conv2)
            
        # Shortcut connection: Add input to the output of the conv layers
        # If dimensions change (due to stride > 1 or different number of filters), 
        # apply a projection (1x1 conv) to the shortcut.
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
            if use_batch_norm: # Apply BN to shortcut as well if used elsewhere
                shortcut = layers.BatchNormalization()(shortcut)

        output = layers.add([conv2, shortcut])
        
        # Final activation after merging
        if activation == 'leaky_relu':
             output = layers.LeakyReLU(alpha=0.2)(output)
        else:
             output = layers.Activation(activation)(output)
        return output

    inputs = keras.Input(shape=input_shape)
    
    # --- Initial Convolution ---
    # Using a larger kernel initially can capture broader features
    x = layers.Conv2D(base_filters, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs) 
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x) # Initial pooling

    # --- Encoder Stages ---
    filters = base_filters
    encoder_stages = []
    print("Building ResNet Encoder...")
    for i, num_blocks in enumerate(num_blocks_per_stage):
        print(f"  Stage {i+1}, Filters: {filters}, Blocks: {num_blocks}")
        # Downsample at the start of each stage (except the first, already done)
        stride = 2 if i > 0 else 1 
        x = residual_block(x, filters, stride=stride) 
        for _ in range(num_blocks - 1):
            x = residual_block(x, filters, stride=1)
        encoder_stages.append(x) # Store for potential skip connections later if needed
        filters *= 2

    # --- Decoder Stages (Simplified: No skip connections here, focuses on upsampling + residual blocks) ---
    print("Building ResNet Decoder...")
    for i in range(len(num_blocks_per_stage) - 1, -1, -1): # Iterate backward through stages
        filters //= 2
        num_blocks = num_blocks_per_stage[i]
        print(f"  Stage {i+1}, Filters: {filters}, Blocks: {num_blocks}")
        # Upsample using Conv2DTranspose
        x = layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same')(x)
        # Apply residual blocks after upsampling
        for _ in range(num_blocks):
            x = residual_block(x, filters, stride=1)
            
    # --- Final Upsampling and Output ---
    # Add potentially more ConvTranspose layers if needed to match output size
    # This depends heavily on the strides used in the encoder
    # For simplicity, assume final stage output needs one more upsample + final conv
    
    # Example: One more transpose conv to potentially restore resolution before final 1x1
    x = layers.Conv2DTranspose(base_filters // 2, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = residual_block(x, base_filters // 2, stride=1) # Final residual block

    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='linear')(x)
    # Adjust output shape if needed, ensure it matches y_train's HxW
    # This might require careful calculation of padding/strides or an adaptive pooling/upsampling layer
    
    # Example: Ensure output has the target spatial dimensions. If not exact, could use resizing.
    # This is a common challenge in designing encoder-decoders precisely.
    # We will rely on Reshape, assuming the network learns to produce the correct HxW spatially.
    outputs = layers.Reshape(output_shape, name="raw_output")(outputs) 

    outputs = layers.Lambda(lambda x: tf.maximum(x, min_velocity), name="constrained_output")(outputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNetStyle_Inverter")
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    if mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
    # Using standard MAE loss for this model variant
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae']) 
    
    return model


# CODE CELL 6: Custom Data Generator with Augmentation
class FWIDataGenerator(keras.utils.Sequence):
    """
    Custom Keras data generator for FWI training.
    Handles batching, shuffling, preprocessing, and data augmentation.
    """
    def __init__(self, X, y, batch_size=8, input_shape=(751, 70, 10), output_shape=(101,101), 
                 shuffle=True, augment=True, noise_level_range=(0.01, 0.05), flip_prob=0.5):
        """
        Initializes the data generator.
        Args:
            X: Input seismic data (num_samples, num_sources, time_steps, num_receivers)
            y: Target velocity models (num_samples, height, width)
            batch_size: Number of samples per batch
            input_shape: Expected model input shape (time_steps, num_receivers, num_sources)
            output_shape: Expected model output shape (height, width) - used for verification
            shuffle: Whether to shuffle data indices at the end of each epoch
            augment: Whether to apply data augmentation
            noise_level_range: Tuple (min, max) for the std deviation of Gaussian noise relative to data range
            flip_prob: Probability of applying horizontal flip augmentation
        """
        if X.shape[0] != y.shape[0]:
             raise ValueError("X and y must have the same number of samples.")
        if y.shape[1:] != output_shape:
             print(f"Warning: y shape {y.shape[1:]} doesn't match expected output_shape {output_shape}")
             
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.input_shape = input_shape # (time, receivers, sources)
        self.output_shape = output_shape # (height, width)
        self.shuffle = shuffle
        self.augment = augment
        self.noise_level_range = noise_level_range
        self.flip_prob = flip_prob
        
        self.num_samples = len(self.X)
        self.indexes = np.arange(self.num_samples)
        self.on_epoch_end() # Initial shuffle if needed

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Generate indexes of the batch
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        indexes = self.indexes[start_idx:end_idx]

        # Find list of IDs
        X_batch = self.X[indexes]
        y_batch = self.y[indexes]

        # Preprocess the seismic data (normalization)
        X_batch_processed = self.preprocess_batch(X_batch)
        
        # Apply augmentation if enabled
        if self.augment:
            X_batch_processed, y_batch = self.augment_batch(X_batch_processed, y_batch)

        # Reshape X for model input: (batch, T, R, S)
        # Original X shape: (batch, S, T, R)
        # Target shape: (batch, T, R, S) based on input_shape
        # Need to transpose: axes (0, 2, 3, 1)
        try:
             X_batch_final = np.transpose(X_batch_processed, (0, 2, 3, 1))
             # Verify against self.input_shape
             if X_batch_final.shape[1:] != self.input_shape:
                 raise ValueError(f"Processed X shape {X_batch_final.shape[1:]} != expected input shape {self.input_shape}")
        except Exception as e:
             print(f"Error during final reshape/transpose of X_batch: {e}")
             print(f"  X_batch_processed shape was: {X_batch_processed.shape}")
             # Return empty arrays or re-raise? For now, print and return potentially incorrect shape
             # This indicates an issue in shape definitions or loading
             return np.zeros((self.batch_size, *self.input_shape)), np.zeros((self.batch_size, *self.output_shape))


        # Ensure y_batch has the correct shape as well
        if y_batch.shape[1:] != self.output_shape:
             print(f"Warning: y_batch shape {y_batch.shape[1:]} != expected output shape {self.output_shape}")

        return X_batch_final, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_batch(self, X_batch_raw):
        """Applies sample-wise normalization to the batch."""
        # Uses the standalone function defined earlier
        return preprocess_seismic_batch(X_batch_raw)

    def augment_batch(self, X_batch, y_batch):
        """Applies random augmentation to the batch."""
        augmented_X = X_batch.copy()
        augmented_y = y_batch.copy()
        
        batch_size = X_batch.shape[0]
        num_sources = X_batch.shape[1]
        
        for i in range(batch_size):
            # 1. Add Noise (per source)
            if np.random.rand() > 0.5: # Apply noise ~50% of the time
                noise_std_factor = np.random.uniform(self.noise_level_range[0], self.noise_level_range[1])
                for j in range(num_sources):
                    data_slice = augmented_X[i, j]
                    signal_std = np.std(data_slice)
                    noise_std = signal_std * noise_std_factor 
                    noise = np.random.normal(0, noise_std, data_slice.shape).astype(data_slice.dtype)
                    augmented_X[i, j] += noise

            # 2. Horizontal Flip
            if np.random.rand() < self.flip_prob:
                # Flip receivers (last dimension of X before transpose)
                augmented_X[i] = augmented_X[i, :, :, ::-1] 
                # Flip velocity map horizontally (last dimension)
                augmented_y[i] = augmented_y[i, :, ::-1]
                
        return augmented_X, augmented_y


# CODE CELL 7: Ensemble Model Definition (Conceptual)
def define_ensemble_models(input_shape, output_shape):
    """
    Defines a list of models to be potentially used in an ensemble.
    Note: This function only *builds* the models; it doesn't train or combine them.
    """
    models = []
    
    print("Defining Model 1: Physics-Guided U-Net")
    model1 = build_physics_guided_unet(input_shape, output_shape, 
                                       base_filters=64, depth=4, smoothness_weight=0.05) 
    models.append(("UNet_Physics_D4", model1))
    
    print("\nDefining Model 2: ResNet-style Model")
    model2 = build_resnet_style_model(input_shape, output_shape,
                                      base_filters=64, num_blocks_per_stage=[2, 2, 2, 2])
    models.append(("ResNet_2222", model2))
    
    # --- Placeholder for potential additional models ---
    # print("\nDefining Model 3: Deeper U-Net")
    # model3 = build_physics_guided_unet(input_shape, output_shape, 
    #                                    base_filters=32, depth=5, smoothness_weight=0.03) 
    # models.append(("UNet_Physics_D5_F32", model3))

    # print("\nDefining Model 4: U-Net without Physics Loss (for comparison)")
    # model4 = build_physics_guided_unet(input_shape, output_shape, 
    #                                    base_filters=64, depth=4, smoothness_weight=0.0) # No smoothness term
    # # Need to recompile model4 with standard MAE loss if smoothness_weight=0 in builder doesn't handle it
    # # optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    # # if mixed_precision.global_policy().name == 'mixed_float16':
    # #     optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    # # model4.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    # models.append(("UNet_StandardMAE_D4", model4))
    
    print(f"\nDefined {len(models)} candidate models for ensemble.")
    return models

# Note: Global variables X_train, y_train are no longer needed here, shapes are passed explicitly.


# CODE CELL 8: Model Training Function
def train_model(model, X_train, y_train, X_val, y_val, 
                input_shape, output_shape, # Pass shapes explicitly
                batch_size=8, epochs=30, model_save_path='best_fwi_model.h5'):
    """
    Trains the FWI model using custom data generators and callbacks.
    
    Args:
        model: Compiled Keras model to train.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        input_shape: Expected model input shape tuple (T, R, S).
        output_shape: Expected model output shape tuple (H, W).
        batch_size: Training batch size.
        epochs: Maximum number of training epochs.
        model_save_path: Path to save the best model weights.
        
    Returns:
        model: Trained model (best weights restored).
        history: Keras training history object.
    """
    print(f"\n--- Starting Model Training ---")
    print(f"Model: {model.name}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}, Max Epochs: {epochs}")
    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
    
    # Create Data Generators
    train_generator = FWIDataGenerator(X_train, y_train, batch_size=batch_size, 
                                     input_shape=input_shape, output_shape=output_shape,
                                     shuffle=True, augment=True)
    val_generator = FWIDataGenerator(X_val, y_val, batch_size=batch_size, 
                                   input_shape=input_shape, output_shape=output_shape,
                                   shuffle=False, augment=False) # No augmentation/shuffle for validation

    # Define Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, 
                                        monitor='val_loss', mode='min', verbose=1,
                                        save_weights_only=True), # Save only weights is usually sufficient and faster
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, 
                                      restore_best_weights=True) # Automatically restores best weights
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 # Set to 1 for progress bar, 2 for one line per epoch, 0 for silent
    )
    
    # Note: If EarlyStopping restored best weights, no need to load manually.
    # If save_weights_only=False in ModelCheckpoint, or if not using EarlyStopping's restore_best_weights,
    # you might need to load the best model explicitly:
    # print(f"Loading best weights from {model_save_path}")
    # model.load_weights(model_save_path) # Load the best weights saved by ModelCheckpoint
    
    print("--- Model Training Finished ---")
    return model, history


# CODE CELL 9: Model Evaluation Function
def evaluate_model(model, X_eval, y_eval, input_shape, 
                   num_samples_to_plot=3, batch_size_eval=16):
    """
    Evaluates the model on evaluation data (e.g., validation set) and visualizes results.
    
    Args:
        model: Trained Keras model.
        X_eval, y_eval: Evaluation data and labels.
        input_shape: Model's expected input shape (T, R, S).
        num_samples_to_plot: Number of random samples to visualize.
        batch_size_eval: Batch size for prediction to manage memory.
    """
    print("\n--- Starting Model Evaluation ---")
    if len(X_eval) == 0:
        print("Evaluation dataset is empty. Skipping evaluation.")
        return

    # --- Predict on Evaluation Data ---
    # Reshape X_eval for prediction: (batch, T, R, S)
    eval_samples = len(X_eval)
    try:
        X_eval_reshaped = np.transpose(X_eval, (0, 2, 3, 1))
        if X_eval_reshaped.shape[1:] != input_shape:
             raise ValueError(f"Evaluation X shape {X_eval_reshaped.shape[1:]} != expected input shape {input_shape}")
    except Exception as e:
         print(f"Error reshaping X_eval for evaluation: {e}. Aborting evaluation.")
         return

    print(f"Predicting on {eval_samples} evaluation samples...")
    y_pred_raw = model.predict(X_eval_reshaped, batch_size=batch_size_eval, verbose=0)
    
    # Apply physics constraints (min velocity + smoothing) post-prediction
    print("Applying physics constraints to predictions...")
    y_pred_constrained = apply_output_constraints(y_pred_raw, min_velocity=1500.0, smoothing_sigma=0.5)
    
    # --- Calculate Metrics ---
    print("Calculating metrics...")
    # Ensure y_eval is float32 for consistent calculations
    y_eval_f32 = y_eval.astype(np.float32) 
    
    mae_raw = np.mean(np.abs(y_eval_f32 - y_pred_raw))
    mae_constrained = np.mean(np.abs(y_eval_f32 - y_pred_constrained))
    
    print(f"  Mean Absolute Error (Raw Prediction):       {mae_raw:.4f}")
    print(f"  Mean Absolute Error (Constrained Prediction): {mae_constrained:.4f}")
    
    # Calculate SSIM (using constrained predictions as the final output)
    ssim_scores = []
    # Determine data range for SSIM (e.g., min/max velocity in ground truth)
    data_range = np.max(y_eval_f32) - np.min(y_eval_f32)
    if data_range == 0: data_range = 1.0 # Avoid division by zero if data is constant
    
    for i in range(eval_samples):
        score = ssim(y_eval_f32[i], y_pred_constrained[i], data_range=data_range)
        ssim_scores.append(score)
    avg_ssim = np.mean(ssim_scores)
    print(f"  Average Structural Similarity Index (SSIM) (Constrained): {avg_ssim:.4f}")

    # --- Visualize Results for Selected Samples ---
    print(f"\nVisualizing results for {num_samples_to_plot} random samples...")
    if eval_samples < num_samples_to_plot:
         print(f"  (Requested {num_samples_to_plot} samples, but only {eval_samples} available)")
         num_samples_to_plot = eval_samples
         
    indices = np.random.choice(eval_samples, num_samples_to_plot, replace=False)
    
    for i, idx in enumerate(indices):
        print(f"\n--- Sample {i+1} (Index {idx}) ---")
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        
        vmin = np.min(y_eval_f32[idx])
        vmax = np.max(y_eval_f32[idx])
        
        # Ground Truth
        im0 = axes[0].imshow(y_eval_f32[idx], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Ground Truth (Index {idx})')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        plt.colorbar(im0, ax=axes[0], label='Velocity (m/s)')
        
        # Raw Model Prediction
        im1 = axes[1].imshow(y_pred_raw[idx], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Raw Model Prediction')
        axes[1].set_xlabel('X Position'); axes[1].set_ylabel('Y Position')
        plt.colorbar(im1, ax=axes[1], label='Velocity (m/s)')
        
        # Physics-Constrained Prediction
        im2 = axes[2].imshow(y_pred_constrained[idx], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title('Physics-Constrained Prediction')
        axes[2].set_xlabel('X Position'); axes[2].set_ylabel('Y Position')
        plt.colorbar(im2, ax=axes[2], label='Velocity (m/s)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.suptitle(f"Velocity Model Comparison - Sample {idx}", fontsize=16)
        plt.show()
        
        # Error Maps
        fig_err, axes_err = plt.subplots(1, 2, figsize=(16, 6))
        
        error_raw = np.abs(y_eval_f32[idx] - y_pred_raw[idx])
        error_constrained = np.abs(y_eval_f32[idx] - y_pred_constrained[idx])
        err_max = np.max([np.max(error_raw), np.max(error_constrained)]) # Consistent color scale
        
        im_err1 = axes_err[0].imshow(error_raw, cmap='hot', vmin=0, vmax=err_max)
        axes_err[0].set_title('Absolute Error Map (Raw Prediction)')
        axes_err[0].set_xlabel('X Position'); axes_err[0].set_ylabel('Y Position')
        plt.colorbar(im_err1, ax=axes_err[0], label='Velocity Error (m/s)')
        
        im_err2 = axes_err[1].imshow(error_constrained, cmap='hot', vmin=0, vmax=err_max)
        axes_err[1].set_title('Absolute Error Map (Constrained Prediction)')
        axes_err[1].set_xlabel('X Position'); axes_err[1].set_ylabel('Y Position')
        plt.colorbar(im_err2, ax=axes_err[1], label='Velocity Error (m/s)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"Prediction Error Comparison - Sample {idx}", fontsize=16)
        plt.show()
        
    print("--- Model Evaluation Finished ---")


# CODE CELL 10: Submission Preparation Function
def prepare_submission(model, test_data, oids, input_shape, 
                       batch_size_pred=16, submission_file='submission.csv'):
    """
    Generates predictions for test data and prepares the Kaggle submission file.
    
    Args:
        model: Trained Keras model.
        test_data: Dictionary mapping oid to test seismic data arrays.
        oids: List of object IDs (keys in test_data) in desired order.
        input_shape: Model's expected input shape (T, R, S).
        batch_size_pred: Batch size for prediction on test data.
        submission_file: Name of the output CSV file.
        
    Returns:
        submission_df: Pandas DataFrame containing the submission.
    """
    print("\n--- Preparing Kaggle Submission ---")
    if not test_data:
        print("Test data dictionary is empty. Cannot generate submission.")
        return pd.DataFrame()
        
    all_predictions_list = []
    
    print(f"Generating predictions for {len(oids)} test samples...")
    for oid in tqdm(oids, desc="Processing test samples"):
        data_raw = test_data[oid] # Shape (batch=1, S, T, R)
        
        # Preprocess (normalize)
        # Note: preprocess_seismic_batch expects (batch, S, T, R)
        data_preprocessed = preprocess_seismic_batch(data_raw)
        
        # Reshape for model input (batch, T, R, S)
        try:
            data_reshaped = np.transpose(data_preprocessed, (0, 2, 3, 1))
            if data_reshaped.shape[1:] != input_shape:
                 raise ValueError(f"Test data oid {oid} shape {data_reshaped.shape[1:]} != expected input {input_shape}")
        except Exception as e:
             print(f"Error reshaping test data for oid {oid}: {e}. Skipping this sample.")
             continue # Skip this sample if reshaping fails

        # Predict (model expects batch dimension)
        predictions_raw = model.predict(data_reshaped, batch_size=batch_size_pred, verbose=0)
        
        # Apply physics constraints
        predictions_constrained = apply_output_constraints(predictions_raw, min_velocity=1500.0, smoothing_sigma=0.5)
        
        # Extract the single predicted map (output shape is likely (1, H, W))
        if predictions_constrained.shape[0] != 1:
            print(f"Warning: Unexpected batch dimension in prediction for oid {oid}. Shape: {predictions_constrained.shape}. Using first element.")
        vel_map = predictions_constrained[0] # Shape (H, W)
        height, width = vel_map.shape
        
        # Extract required values for submission format
        for y_pos in range(height):
            # Get values at odd horizontal indices (x=1, 3, 5, ...)
            odd_indices = np.arange(1, width, 2) 
            if len(odd_indices) == 0: continue # Skip if width is 0 or 1

            values = vel_map[y_pos, odd_indices]
            
            row_id = f"{oid}_y_{y_pos}"
            row_dict = {"oid_ypos": row_id}
            
            # Populate the dictionary with x_i columns
            for i, val in enumerate(values):
                col_name = f"x_{2*i + 1}" # x_1, x_3, x_5, ...
                row_dict[col_name] = val
            
            all_predictions_list.append(row_dict)
            
    # Create DataFrame
    submission_df = pd.DataFrame(all_predictions_list)
    
    # Ensure columns are in the expected order (oid_ypos, x_1, x_3, ...)
    if not submission_df.empty:
        first_row_keys = list(all_predictions_list[0].keys())
        # Find max x index from column names like 'x_i'
        x_cols = [col for col in first_row_keys if col.startswith('x_')]
        if x_cols:
             max_x_index = max([int(col.split('_')[1]) for col in x_cols])
             expected_x_cols = [f"x_{i}" for i in range(1, max_x_index + 1, 2)]
             column_order = ["oid_ypos"] + expected_x_cols
             # Reorder df columns, handling potential missing columns if width varies?
             submission_df = submission_df.reindex(columns=column_order) 
        else:
             column_order = ["oid_ypos"] # Case where no x columns were generated
             submission_df = submission_df.reindex(columns=column_order)

    # Save to CSV
    try:
        submission_df.to_csv(submission_file, index=False)
        print(f"Submission file saved to '{submission_file}' with {len(submission_df)} rows and {len(submission_df.columns)} columns.")
        print("\nSubmission Sample (first 5 rows):")
        print(submission_df.head())
    except Exception as e:
        print(f"Error saving submission file: {e}")

    print("--- Submission Preparation Finished ---")
    return submission_df


# CODE CELL 11: Main Execution Function
def main():
    """Main function to orchestrate the FWI pipeline."""
    print("==============================================================")
    print(" Starting Yale/UNC-CH Geophysical Waveform Inversion Pipeline ")
    print("==============================================================")
    print(f"Timestamp: {pd.Timestamp.now()}")
    
    # --- Configuration ---
    # Set a sample limit for faster testing/debugging, None to use all data
    TRAINING_SAMPLE_LIMIT = None # e.g., 100 or 500. Set to None for full run.
    EPOCHS = 30 # Adjust as needed
    BATCH_SIZE = 8 # Adjust based on GPU memory
    MODEL_SAVE_NAME = 'best_physics_guided_unet.h5' # Specific name for the saved model
    SUBMISSION_FILENAME = 'submission.csv'
    
    # --- 1. Load Training Data ---
    print("\n=== 1. Loading Training Data ===")
    train_data = {}
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Training directory '{TRAIN_DIR}' not found. Exiting.")
        return
        
    available_families = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    if not available_families:
        print(f"ERROR: No dataset families found in '{TRAIN_DIR}'. Exiting.")
        return
        
    print(f"Found families: {available_families}")
    
    X_all_list, y_all_list = [], []
    for family in available_families:
        print(f"\n--- Processing Family: {family} ---")
        X, y = load_training_data(family, sample_limit=TRAINING_SAMPLE_LIMIT)
        if X is not None and y is not None:
            print(f"Loaded {family}: X shape={X.shape}, y shape={y.shape}")
            X_all_list.append(X)
            y_all_list.append(y)
            # Optional: Visualize one sample per family
            # plot_seismic_and_velocity(X, y, sample_idx=0, title_prefix=f"{family} - ")
        else:
            print(f"Skipping family {family} due to loading issues.")
        gc.collect() # Clean up memory after loading each family

    if not X_all_list:
        print("ERROR: No training data could be loaded. Exiting.")
        return

    X_all = np.concatenate(X_all_list, axis=0)
    y_all = np.concatenate(y_all_list, axis=0)
    del X_all_list, y_all_list # Free memory
    gc.collect()
    print(f"\nCombined Training Data: X shape={X_all.shape}, y shape={y_all.shape}")
    
    # --- 2. Data Exploration (Combined Data) ---
    print("\n=== 2. Data Visualization (Combined Sample) ===")
    num_samples_to_plot = min(2, len(X_all))
    if num_samples_to_plot > 0:
         for i in range(num_samples_to_plot):
             plot_seismic_and_velocity(X_all, y_all, sample_idx=i, title_prefix="Combined ")
    else:
         print("No combined data available to plot.")

    # --- 3. Data Splitting ---
    print("\n=== 3. Splitting Data into Training/Validation ===")
    if len(X_all) < 2:
         print("ERROR: Not enough data to split into training and validation sets. Need at least 2 samples.")
         # Decide how to handle: exit, or use all data for training (no validation)?
         # For now, we'll exit if we can't validate.
         return 
         
    val_size = 0.2 # 20% for validation
    try:
        X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=val_size, random_state=42)
        del X_all, y_all # Free memory
        gc.collect()
        print(f"Training set:   X shape={X_train.shape}, y shape={y_train.shape}")
        print(f"Validation set: X shape={X_val.shape}, y shape={y_val.shape}")
    except Exception as e:
        print(f"Error during train/test split: {e}")
        return

    # --- 4. Determine Shapes ---
    print("\n=== 4. Determining Model Input/Output Shapes ===")
    try:
        # X shape: (batch, S, T, R) -> Model Input (T, R, S)
        _, num_sources, time_steps, num_receivers = X_train.shape 
        # y shape: (batch, H, W) -> Model Output (H, W)
        _, height, width = y_train.shape 
        
        input_shape = (time_steps, num_receivers, num_sources)
        output_shape = (height, width)
        print(f"Deduced Input Shape (T, R, S): {input_shape}")
        print(f"Deduced Output Shape (H, W): {output_shape}")
    except Exception as e:
        print(f"Error determining shapes from training data: {e}")
        return

    # --- 5. Build Model ---
    print("\n=== 5. Building Neural Network Model ===")
    # Choose which model to build here
    # model = build_resnet_style_model(input_shape, output_shape)
    model = build_physics_guided_unet(input_shape, output_shape, 
                                      base_filters=64, depth=4, # Example parameters
                                      smoothness_weight=0.05, min_velocity=1500.0)
    model.summary(line_length=120)
    
    # --- 6. Train Model ---
    print("\n=== 6. Training Model ===")
    trained_model, history = train_model(model, X_train, y_train, X_val, y_val, 
                                         input_shape=input_shape, output_shape=output_shape,
                                         batch_size=BATCH_SIZE, epochs=EPOCHS, 
                                         model_save_path=MODEL_SAVE_NAME)
    
    # --- 7. Plot Training History ---
    print("\n=== 7. Plotting Training History ===")
    if history and history.history:
        try:
            plt.figure(figsize=(14, 6))
            
            # Loss Plot
            plt.subplot(1, 2, 1)
            if 'loss' in history.history: plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history: plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss Value')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # MAE Plot (or other primary metric)
            plt.subplot(1, 2, 2)
            primary_metric = 'mae' # Or 'mean_absolute_error' depending on tf version/naming
            val_primary_metric = f'val_{primary_metric}'
            if primary_metric in history.history: plt.plot(history.history[primary_metric], label=f'Training {primary_metric.upper()}')
            if val_primary_metric in history.history: plt.plot(history.history[val_primary_metric], label=f'Validation {primary_metric.upper()}')
            plt.title('Model Mean Absolute Error (MAE)')
            plt.ylabel('MAE Value')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot training history: {e}")
    else:
        print("No training history available to plot.")

    # --- 8. Evaluate Model ---
    print("\n=== 8. Evaluating Model on Validation Set ===")
    evaluate_model(trained_model, X_val, y_val, input_shape=input_shape, 
                   num_samples_to_plot=min(3, len(X_val)), # Plot up to 3 samples
                   batch_size_eval=BATCH_SIZE) # Use same batch size or adjust for memory
                   
    # Optional: Clean up validation data if no longer needed
    # del X_val, y_val 
    # gc.collect()

    # --- 9. Load Test Data ---
    print("\n=== 9. Loading Test Data ===")
    if not os.path.exists(TEST_DIR):
         print(f"Warning: Test directory '{TEST_DIR}' not found. Skipping submission generation.")
         test_data, oids = {}, []
    else:
         test_data, oids = load_test_data()
         if test_data:
             print(f"Loaded {len(oids)} test samples. Example OID: {oids[0] if oids else 'N/A'}")
             # print(f"  Example test data shape: {test_data[oids[0]].shape if oids else 'N/A'}")
         else:
             print("No test data loaded.")

    # --- 10. Prepare Submission ---
    print("\n=== 10. Preparing Submission File ===")
    if test_data and oids:
        submission_df = prepare_submission(trained_model, test_data, oids, 
                                           input_shape=input_shape, 
                                           batch_size_pred=BATCH_SIZE, # Adjust if needed for test inference
                                           submission_file=SUBMISSION_FILENAME)
        # submission_df now holds the result, already saved to CSV
    else:
        print("Skipping submission file generation as no test data was loaded.")

    print("\n==============================================================")
    print(" Pipeline Execution Finished")
    print("==============================================================")

# --- Entry Point Check ---
if __name__ == "__main__":
    # This ensures the main function runs only when the script is executed directly
    # (not when imported as a module)
    main()
    # Optional: Explicitly clear session if running multiple times in one environment
    # tf.keras.backend.clear_session() 
    # gc.collect()

