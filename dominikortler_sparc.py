import os
import json # Needed to test loading

# --- Dataset Path Verification ---

def detect_base_path():
    # Priority: env var -> Kaggle path -> local kaggle mirror -> ./data -> .
    env = os.getenv('ARC_DATA_DIR')
    if env and os.path.exists(env):
        return env
    kaggle_path = '/kaggle/input/arc-prize-2025/'
    if os.path.exists(kaggle_path):
        return kaggle_path
    local_kaggle = os.path.join('.', 'kaggle', 'input', 'arc-prize-2025')
    if os.path.exists(local_kaggle):
        return local_kaggle
    data_dir = os.path.join('.', 'data')
    if os.path.exists(data_dir):
        return data_dir
    return '.'

BASE_PATH = detect_base_path()
# Assuming these are now direct paths to the JSON files
TRAINING_JSON_PATH = os.path.join(BASE_PATH, 'arc-agi_training_challenges.json')
EVALUATION_JSON_PATH = os.path.join(BASE_PATH, 'arc-agi_evaluation_challenges.json')
TEST_JSON_PATH = os.path.join(BASE_PATH, 'arc-agi_test_challenges.json') # Optional
TRAINING_SOLUTIONS_JSON_PATH = os.path.join(BASE_PATH, 'arc-agi_training_solutions.json')
EVALUATION_SOLUTIONS_JSON_PATH = os.path.join(BASE_PATH, 'arc-agi_evaluation_solutions.json')

print("--- Verifying ARC Dataset Paths ---")
print(f"Using Base Path: {BASE_PATH}")
print(f"Expected Training JSON: {TRAINING_JSON_PATH}")
print(f"Expected Evaluation JSON: {EVALUATION_JSON_PATH}")
print(f"Expected Test JSON: {TEST_JSON_PATH}") # Optional
print(f"Expected Training Solutions JSON: {TRAINING_SOLUTIONS_JSON_PATH}")
print(f"Expected Evaluation Solutions JSON: {EVALUATION_SOLUTIONS_JSON_PATH}")

paths_exist = True
files_found = True

if not os.path.exists(BASE_PATH):
    print(f"ERROR: Base path does not exist: {BASE_PATH}")
    print("Please ensure the dataset is correctly added via '+ Add Data'.")
    paths_exist = False
    files_found = False
else:
    print(f"Base path found.")

    # Check for EVALUATION file
    if not os.path.isfile(EVALUATION_JSON_PATH):
        print(f"ERROR: Evaluation file not found: {EVALUATION_JSON_PATH}")
        files_found = False
    else:
        print(f"Evaluation file found.")
        # Optional: Try loading a small part to check format
        try:
            with open(EVALUATION_JSON_PATH, 'r') as f:
                data = json.load(f)
                print(f"  Successfully loaded evaluation JSON. Found {len(data)} tasks.")
                # Check if it's a dictionary as expected
                if not isinstance(data, dict):
                     print(f"  WARNING: Evaluation file content is not a dictionary.")
        except Exception as e:
            print(f"  ERROR reading evaluation JSON: {e}")
            files_found = False


    # Check for TRAINING file
    if not os.path.isfile(TRAINING_JSON_PATH):
        print(f"WARNING: Training file not found: {TRAINING_JSON_PATH}")
        # If only doing evaluation, this might be okay.
    else:
        print(f"Training file found.")
        # Optional: Try loading
        try:
            with open(TRAINING_JSON_PATH, 'r') as f:
                 data = json.load(f)
                 print(f"  Successfully loaded training JSON. Found {len(data)} tasks.")
                 if not isinstance(data, dict):
                      print(f"  WARNING: Training file content is not a dictionary.")
        except Exception as e:
            print(f"  ERROR reading training JSON: {e}")
            # files_found = False # Don't fail if only training is broken
# Add checks for these files within the verification block in Cell 1
if paths_exist:
    # Check for TRAINING SOLUTIONS file
    if not os.path.isfile(TRAINING_SOLUTIONS_JSON_PATH):
        print(f"ERROR: Training Solutions file not found: {TRAINING_SOLUTIONS_JSON_PATH}")
        files_found = False # Make this essential for training
    else:
        print(f"Training Solutions file found.")
        try:
            with open(TRAINING_SOLUTIONS_JSON_PATH, 'r') as f:
                 data = json.load(f)
                 print(f"  Successfully loaded training solutions JSON. Found solutions for {len(data)} tasks.")
                 if not isinstance(data, dict):
                      print(f"  WARNING: Training solutions file content is not a dictionary.")
        except Exception as e:
            print(f"  ERROR reading training solutions JSON: {e}")
            files_found = False

    # Check for EVALUATION SOLUTIONS file
    if not os.path.isfile(EVALUATION_SOLUTIONS_JSON_PATH):
        print(f"WARNING: Evaluation Solutions file not found: {EVALUATION_SOLUTIONS_JSON_PATH}")
        # Might be okay if only training, but needed for proper evaluation
    else:
        print(f"Evaluation Solutions file found.")
        try:
            with open(EVALUATION_SOLUTIONS_JSON_PATH, 'r') as f:
                 data = json.load(f)
                 print(f"  Successfully loaded evaluation solutions JSON. Found solutions for {len(data)} tasks.")
                 if not isinstance(data, dict):
                      print(f"  WARNING: Evaluation solutions file content is not a dictionary.")
        except Exception as e:
            print(f"  ERROR reading evaluation solutions JSON: {e}")
            # Depending on goal, might set files_found = False

if paths_exist and files_found:
    print("Dataset files seem correct and loadable.")
else:
    print("!!! Problem detected with dataset paths or file contents. Please check. !!!")
print("--- Verification Complete ---")

# Stop execution if essential files are missing
if not files_found:
   raise FileNotFoundError("Required ARC JSON dataset files not found or failed to load. Stopping execution.")






# --- Model Hyperparameters ---
MAX_GRID_SIZE = 30 # <<< THIS LINE IS NEEDED
NUM_COLORS = 10
WAVELET = 'coif2'
WAVELET_LEVELS = 2
# KAN/Encoder Params
KAN_HIDDEN_UNITS = 128 # Hidden units *within* KAN layers
ENCODER_DIM = 128  # Dimension *after* Wavelet->KAN->Dense projection

# VQ-VAE Parameters
VQ_EMBEDDING_DIM = ENCODER_DIM # Dimension of embeddings
VQ_NUM_EMBEDDINGS = 512        # Size of the codebook
VQ_COMMITMENT_COST = 0.25      # Beta term for VQ loss

# Transformer Parameters
D_MODEL = ENCODER_DIM # Core dimension, should match Encoder output
N_HEADS = 4        # Transformer heads
N_TRANSFORMER_LAYERS = 4 # Transformer layers
FFN_DIM = D_MODEL * 2    # Transformer FeedForward internal dim
DROPOUT = 0.1

# --- Recurrent Decoder Parameters (NEEDED FOR CELL 5) ---
RECURRENT_STEPS = 5       # Number of refinement steps
CONV_LSTM_FILTERS = D_MODEL # Filters in ConvLSTM cell (uses D_MODEL)

# --- Training Hyperparameters ---
EPOCHS = 15
LEARNING_RATE = 3e-5
CLIP_NORM = 1.0
BATCH_SIZE = 1

# --- Loss Weights ---
CE_LOSS_WEIGHT = 1.0
L1_LOSS_WEIGHT = 0.05
L2_LOSS_WEIGHT = 0.05
LINF_LOSS_WEIGHT = 0.0
EQUI_LOSS_WEIGHT = 0.1
VQ_LOSS_WEIGHT = 1.0
SIZE_LOSS_WEIGHT = 10

# ... (all other loss weights) ...

# --- Calculated Constants ---
import pywt
import numpy as np

try:
   _approx_coeffs_shape = pywt.wavedec2(np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE)), WAVELET, level=WAVELET_LEVELS)
   _arr_temp, _ = pywt.coeffs_to_array(_approx_coeffs_shape)
   MAX_COEFF_LEN = _arr_temp.flatten().shape[0]
   print(f"Calculated MAX_COEFF_LEN: {MAX_COEFF_LEN}")
except ValueError:
   MAX_COEFF_LEN = MAX_GRID_SIZE * MAX_GRID_SIZE
   print(f"Using fallback MAX_COEFF_LEN: {MAX_COEFF_LEN}")


# --- Standard Libraries ---
import numpy as np
import json
import os
import traceback
import math
import time
from collections import Counter # Useful for aggregating losses

# --- Core ML Libraries ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print("TensorFlow Version:", tf.__version__)

# --- Specialised Libraries ---
# Install if needed (already in your original script)
# !pip install pywavelets --quiet
# !wget https://raw.githubusercontent.com/Mattral/Kolmogorov-Arnold-Networks/main/KANtf.py -O KANtf.py --quiet

import pywt # For Wavelet Transform
print("PyWavelets Version:", pywt.__version__)


# --- Matplotlib for Visualization ---
import matplotlib.pyplot as plt


# Cell 2.5: Custom KAN Layer Implementation

class KANLayer(layers.Layer):
    """
    Custom KAN Layer implementation based on Kolmogorov-Arnold Networks.
    Uses learnable piecewise linear functions (B-splines of order 2)
    on the edges, combined with a base linear function.

    y_j = sum_{i} ( base_weight_{ij} * activation_base(x_i) +
                    spline_scaler_{ij} * spline(x_i | coeffs_{ij}, grid) )
    """
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=1,
                 grid_range=[-1, 1], base_activation=None, spline_activation=None,
                 l1_spline_reg=0.0, # Optional L1 regularization on spline coefficients
                 name="kan_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order # Currently only supports 1 (piecewise linear)
        self.grid_range = grid_range
        self.base_activation = keras.activations.get(base_activation)
        self.spline_activation = keras.activations.get(spline_activation)
        self.l1_spline_reg = l1_spline_reg

        if self.spline_order != 1:
             print("Warning: KANLayer currently only supports spline_order=1 (piecewise linear)."
                   " Ignoring specified order.")
             self.spline_order = 1

        # --- Define Grid (Non-trainable) ---
        # Shape: (1, 1, grid_size) for broadcasting
        grid = tf.cast(tf.linspace(grid_range[0], grid_range[1], grid_size), dtype=self.dtype)
        self.grid = tf.reshape(grid, (1, 1, grid_size)) # Make it broadcastable

        # --- Learnable Weights ---
        # Base function weights (like a standard linear layer)
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.input_dim, self.output_dim),
            initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal'),
            trainable=True,
        )

        # Spline function coefficients (control points on the grid)
        # Shape: (input_dim, output_dim, grid_size)
        self.spline_coeffs = self.add_weight(
            name="spline_coeffs",
            shape=(self.input_dim, self.output_dim, self.grid_size),
            initializer=keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), # Smaller init for splines
            regularizer=keras.regularizers.L1(l1_spline_reg) if l1_spline_reg > 0 else None,
            trainable=True,
        )

        # Scaler for the spline component's contribution
        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(self.input_dim, self.output_dim),
            initializer=keras.initializers.Constant(value=1.0), # Start by adding spline directly
            trainable=True,
        )

        # Add bias term per output neuron (optional, but common)
        self.bias = self.add_weight(
            name='bias',
            shape=(output_dim,),
            initializer='zeros',
            trainable=True
        )

    def build(self, input_shape):
        # Keras convention - weights defined in __init__ here
        super().build(input_shape)

    def call(self, x, training=None):
        # x shape: (batch_size, input_dim)

        # --- 1. Base Function ---
        # Standard linear transformation
        # Output shape: (batch_size, output_dim)
        base_output = tf.matmul(x, self.base_weight)
        if self.base_activation is not None:
            base_output = self.base_activation(base_output)

        # --- 2. Spline Function (Piecewise Linear Interpolation) ---
        # Reshape/expand inputs for broadcasting
        # x: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        x_expanded = tf.expand_dims(x, axis=-1)

        # grid: (1, 1, grid_size)
        # spline_coeffs: (input_dim, output_dim, grid_size) -> (1, input_dim, output_dim, grid_size) for broadcast
        spline_coeffs_b = tf.expand_dims(self.spline_coeffs, axis=0)

        # Find the interval index for each input value on the grid
        # Ensure values outside the grid range are handled (clip or use edge values)
        x_clipped = tf.clip_by_value(x_expanded, self.grid_range[0], self.grid_range[1])

        # Find the indices of the grid points *before* each x value
        # `tf.searchsorted` returns the index `k` such that grid[k-1] < x <= grid[k]
        # We need indices `k` such that grid[k] <= x < grid[k+1]
        # Let's use a slightly safer approach: find nearest grid points
        # Shape: (batch_size, input_dim, grid_size)
        distances = tf.abs(x_clipped - self.grid)
        # Indices of the two nearest grid points
        # Using top_k with k=2 and inverting distances might work, but searchsorted is more direct for intervals
        # Let's stick to the interpolation idea based on interval indices

        # Indices `k` such that grid[k] is the largest grid value <= x
          # --- MODIFICATION START ---
        # Get input shapes
        batch_size = tf.shape(x_clipped)[0]
        input_dim = tf.shape(x_clipped)[1] # Should be self.input_dim

        # Prepare grid sequence (1D)
        grid_sequence = self.grid[0, 0, :] # Shape (G,)

        # Prepare values tensor (needs to be searched)
        values_to_search = x_clipped[:, :, 0] # Shape (B, I)

        # Reshape values to 1D for searchsorted
        values_to_search_flat = tf.reshape(values_to_search, [-1]) # Shape (B*I,)

        # Perform searchsorted on flattened data
        k_flat = tf.searchsorted(grid_sequence, values_to_search_flat, side='right') - 1 # Shape (B*I,)

        # Reshape k back to (B, I)
        k = tf.reshape(k_flat, [batch_size, input_dim]) # Shape (B, I)
        # --- MODIFICATION END ---
        # Clip k to be within valid range [0, grid_size - 2] for k and k+1 access
        k = tf.clip_by_value(k, 0, self.grid_size - 2)

        # Gather corresponding grid points and coefficients
        # Indices need to be shaped for gather_nd.
        # Create multi-dimensional indices for batch and input dimensions
        batch_indices = tf.range(tf.shape(x)[0])
        input_indices = tf.range(self.input_dim)
        batch_indices_mesh, input_indices_mesh = tf.meshgrid(batch_indices, input_indices, indexing='ij')

        # Indices for coefficients at k and k+1
        # Shape (batch_size, input_dim, 3) -> (batch, input_idx, grid_idx)
        indices_k = tf.stack([input_indices_mesh, k], axis=-1) # (batch, input_dim, 2) -> need output dim
        indices_kp1 = tf.stack([input_indices_mesh, k + 1], axis=-1) # (batch, input_dim, 2)

        # We need to gather across output dims too. Coeffs are (input_dim, output_dim, grid_size)
        # Let's compute spline values per input-output pair, then sum later.
        # spline_coeffs: (input_dim, output_dim, grid_size)
        # We need coeffs for C(i, j, k) and C(i, j, k+1)

        # Gather requires careful index construction or map_fn/loops (less efficient)
        # Alternative: Calculate weights and use direct indexing/multiplication if possible

        # Let's use the direct interpolation formula:
        # y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        # Grid points g_k = grid[k], g_kp1 = grid[k+1]
        # Shape: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        g_k = tf.gather(grid_sequence, k)
        g_kp1 = tf.gather(grid_sequence, k + 1)
        g_k = tf.expand_dims(g_k, axis=-1)
        g_kp1 = tf.expand_dims(g_kp1, axis=-1)

        # Coefficients c_k = spline_coeffs[i, :, k], c_kp1 = spline_coeffs[i, :, k+1]
        # Need to gather coeffs for all output dims j.
        # Coeffs shape (input_dim, output_dim, grid_size)

        # Create indices for gather_nd that include batch, input_dim, output_dim, grid_dim
        # This becomes complex. Let's simplify: Compute spline per input feature first, then project.

        # Simpler approach: Compute spline value for each input independently,
        # then use linear combination (base_weight handles this projection).
        # Let spline_coeffs be (input_dim, grid_size) - only one spline per input
        # --> NO, KAN requires spline *per connection* (i, j).

        # Let's try the interpolation again, carefully:
        # Target: Calculate `spline_values` of shape (batch, input_dim, output_dim)
        # where spline_values[b, i, j] = spline_ij(x[b, i])

        # Broadcast x_clipped to (batch, input_dim, 1, 1)
        x_bc = tf.expand_dims(x_clipped, axis=-1)
        # Broadcast grid to (1, 1, grid_size, 1)
        grid_bc = tf.expand_dims(self.grid, axis=-1)
        # Broadcast coeffs to (1, input_dim, output_dim, grid_size)
        coeffs_bc = tf.expand_dims(self.spline_coeffs, axis=0)

        # Find indices k and k+1
        # `k` shape is (batch_size, input_dim)
        k_exp = tf.expand_dims(tf.expand_dims(k, axis=-1), axis=-1) # (batch, input_dim, 1, 1)

        # Create indices for gathering coefficients
        # Need indices for [input_dim, output_dim, k] and [input_dim, output_dim, k+1]
        # Indices must align with the batch dimension as well, but coeffs don't have batch dim.
        # We can use tf.gather with batch_dims=0 on coeffs.

        # Let's compute the normalized position within the interval:
        # t = (x - g_k) / (g_kp1 - g_k)
        # Avoid division by zero if grid points are identical
        grid_diff = g_kp1 - g_k
        # Add epsilon to prevent NaN gradients where grid_diff is zero
        t = (x_clipped - g_k) / (grid_diff + tf.keras.backend.epsilon())
        t = tf.clip_by_value(t, 0.0, 1.0) # Ensure t is within [0, 1]
        # t shape: (batch_size, input_dim, 1)

        # Gather coefficients C_k and C_k+1 for all output dims
        # Need to create indices for tf.gather_nd for C(i, j, k)
        input_idx_mesh, output_idx_mesh = tf.meshgrid(tf.range(self.input_dim),
                                                      tf.range(self.output_dim), indexing='ij')
        # Mesh shape: (input_dim, output_dim)
        # Need to repeat/tile for batch dimension
        batch_size = tf.shape(x)[0]
        input_idx_mesh_b = tf.tile(tf.expand_dims(input_idx_mesh, axis=0), [batch_size, 1, 1])
        output_idx_mesh_b = tf.tile(tf.expand_dims(output_idx_mesh, axis=0), [batch_size, 1, 1])
        # k has shape (batch, input_dim), needs expansion for output dim
        k_b = tf.tile(tf.expand_dims(k, axis=-1), [1, 1, self.output_dim]) # (batch, input, output)

        indices_k = tf.stack([input_idx_mesh_b, output_idx_mesh_b, k_b], axis=-1)
        indices_kp1 = tf.stack([input_idx_mesh_b, output_idx_mesh_b, k_b + 1], axis=-1)

        # Gather coeffs: spline_coeffs shape is (input_dim, output_dim, grid_size)
        coeff_k = tf.gather_nd(self.spline_coeffs, indices_k) # (batch, input, output)
        coeff_kp1 = tf.gather_nd(self.spline_coeffs, indices_kp1) # (batch, input, output)

        # Linear interpolation: y = (1-t)*y0 + t*y1
        # t shape: (batch, input, 1) - needs broadcast against (batch, input, output)
        t_bc = tf.expand_dims(t, axis=-1)
        # spline_val shape: (batch, input_dim, output_dim)
        spline_val = (1.0 - t) * coeff_k + t * coeff_kp1

        # Apply scaler (broadcast scaler over batch dim)
        # spline_scaler shape: (input_dim, output_dim) -> (1, input_dim, output_dim)
        spline_val_scaled = spline_val * tf.expand_dims(self.spline_scaler, axis=0)

        # Sum over the input dimension
        # spline_output shape: (batch_size, output_dim)
        spline_output = tf.reduce_sum(spline_val_scaled, axis=1)

        if self.spline_activation is not None:
            spline_output = self.spline_activation(spline_output)

        # --- 3. Combine Base and Spline ---
        # y shape: (batch_size, output_dim)
        y = base_output + spline_output + self.bias

        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "base_activation": keras.activations.serialize(self.base_activation),
            "spline_activation": keras.activations.serialize(self.spline_activation),
            "l1_spline_reg": self.l1_spline_reg,
        })
        return config


# --- Grid Utilities ---
def pad_grid(grid, max_h, max_w, pad_value=0):
    # ... implementation ...
    h, w = grid.shape
    pad_h = max_h - h
    pad_w = max_w - w
    if pad_h < 0 or pad_w < 0:
        grid = grid[:max_h, :max_w]
        pad_h, pad_w = 0, 0
    # Ensure non-negative padding dimensions before calling np.pad
    pad_h = max(0, pad_h)
    pad_w = max(0, pad_w)
    return np.pad(grid, ((0, pad_h), (0, pad_w)), constant_values=pad_value)


TRANSFORMATIONS = ['identity', 'rotate90', 'rotate180', 'rotate270', 'flip_lr', 'flip_ud']
def augment_grid(grid, transformation):
    # ... implementation ...
    if transformation == 'rotate90': return np.rot90(grid, k=1)
    if transformation == 'rotate180': return np.rot90(grid, k=2)
    if transformation == 'rotate270': return np.rot90(grid, k=3)
    if transformation == 'flip_lr': return np.fliplr(grid)
    if transformation == 'flip_ud': return np.flipud(grid)
    return grid # Identity

# --- Wavelet Transform ---
def np_dwt2(data, wavelet_arg, level_arg, max_coeff_len): # Use different names for args
    """Applies 2D DWT using PyWavelets. Pads coeffs to a fixed size."""
    batch_coeffs = []

    # --- START: Explicit Type Conversion ---
    # Decode wavelet name if it arrives as bytes (TF sometimes does this)
    if isinstance(wavelet_arg, bytes):
        wavelet = wavelet_arg.decode('utf-8')
    else:
        wavelet = str(wavelet_arg) # Ensure it's definitely a string

    # Ensure level is an integer
    level = int(level_arg)
    # --- END: Explicit Type Conversion ---

    for i in range(data.shape[0]): # Iterate over batch
        grid = data[i, :, :, 0]
        h, w = grid.shape

        # Ensure grid is large enough for the level of decomposition
        try:
            min_req_dim = 0
            wavelet_obj = pywt.Wavelet(wavelet) # Use the converted string
            min_req_dim = wavelet_obj.dec_len * (2**(level-1)) if level > 0 else 0
        except:
            min_req_dim = 2**level

        current_level = level
        if min(h, w) < min_req_dim and level > 0:
             max_level_h = pywt.dwt_max_level(h, wavelet) if h > 0 else 0 # Use converted string
             max_level_w = pywt.dwt_max_level(w, wavelet) if w > 0 else 0 # Use converted string
             current_level = max(0, min(max_level_h, max_level_w))

        # Perform DWT if possible
        if current_level > 0:
             # Use the converted string and int for pywt call
             coeffs = pywt.wavedec2(grid, wavelet, level=current_level)
             arr, coeff_slices = pywt.coeffs_to_array(coeffs)
             flat_coeffs = arr.flatten()
        else:
             flat_coeffs = grid.flatten()

        # Padding/Truncating logic
        if len(flat_coeffs) >= max_coeff_len:
             padded_coeffs = flat_coeffs[:max_coeff_len]
        else:
             padded_coeffs = np.pad(flat_coeffs, (0, max_coeff_len - len(flat_coeffs)))
        batch_coeffs.append(padded_coeffs)

    return np.array(batch_coeffs).astype(np.float32)


@tf.function(input_signature=[tf.TensorSpec(shape=(None, MAX_GRID_SIZE, MAX_GRID_SIZE, 1), dtype=tf.float32)])
def tf_dwt2_wrapper(data):
    """TF wrapper for NumPy DWT. Ensures output shape."""
    # MAX_COEFF_LEN is available from Cell 2
    output = tf.numpy_function(
        func=np_dwt2,
        # Pass MAX_COEFF_LEN explicitly
        inp=[data, WAVELET, WAVELET_LEVELS, MAX_COEFF_LEN],
        Tout=tf.float32
    )
    # Use the global MAX_COEFF_LEN for setting shape
    output.set_shape((None, MAX_COEFF_LEN))
    return output

# --- Visualization Helper ---
def plot_grid(ax, grid, title=""):
    """Plots a single ARC grid using matplotlib."""
    cmap = plt.get_cmap('tab10', NUM_COLORS)
    norm = plt.Normalize(vmin=-0.5, vmax=NUM_COLORS - 0.5) # Center colors
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def plot_task(task_data):
    """Plots the training and test pairs for an ARC task."""
    num_train = len(task_data['train'])
    num_test = len(task_data['test'])
    total_pairs = num_train + num_test
    fig, axs = plt.subplots(2, total_pairs, figsize=(3 * total_pairs, 6))
    fig.suptitle(f"Task Visualization", fontsize=16)

    for i, pair in enumerate(task_data['train']):
        plot_grid(axs[0, i], np.array(pair['input']), f"Train {i} Input")
        plot_grid(axs[1, i], np.array(pair['output']), f"Train {i} Output")

    for i, pair in enumerate(task_data['test']):
        plot_grid(axs[0, num_train + i], np.array(pair['input']), f"Test {i} Input")
        if 'output' in pair:
            plot_grid(axs[1, num_train + i], np.array(pair['output']), f"Test {i} Output")
        else:
             axs[1, num_train + i].set_visible(False) # Hide if no output

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.show()


# --- Test pad_grid ---
print("Testing pad_grid...")
test_grid = np.arange(6).reshape(2, 3)
padded = pad_grid(test_grid, 5, 5, pad_value=-1)
print("Original:\n", test_grid)
print("Padded (5x5):\n", padded)
padded_smaller = pad_grid(test_grid, 1, 2, pad_value=-1)
print("Padded (1x2 - should crop):\n", padded_smaller)

# --- Test augment_grid ---
print("\nTesting augment_grid...")
fig, axs = plt.subplots(1, len(TRANSFORMATIONS), figsize=(15, 3))
for i, trans in enumerate(TRANSFORMATIONS):
    augmented = augment_grid(test_grid, trans)
    plot_grid(axs[i], augmented, title=trans)
plt.show()

# --- Test DWT Wrapper (Requires TF context) ---
print("\nTesting DWT Wrapper...")
try:
    # Create a dummy batch of grids (use float32 for wrapper)
    dummy_grids = np.random.rand(2, MAX_GRID_SIZE, MAX_GRID_SIZE, 1).astype(np.float32)
    dwt_coeffs = tf_dwt2_wrapper(tf.constant(dummy_grids))
    print("DWT output shape:", dwt_coeffs.shape)
    # Check if shape matches expected (Batch, MAX_COEFF_LEN)
    assert dwt_coeffs.shape == (2, MAX_COEFF_LEN)
    print("DWT Wrapper test successful.")
except Exception as e:
    print(f"DWT Wrapper test failed: {e}")
    traceback.print_exc()

# --- Test Task Loading and Plotting ---
print("\nLoading and plotting first training task...")
try:
    # Use the path to the single JSON file defined in Cell 2
    if os.path.exists(TRAINING_JSON_PATH):
        with open(TRAINING_JSON_PATH, 'r') as f:
            all_training_tasks_for_test = json.load(f)
        # Get the first task ID (assuming it's a dictionary)
        first_task_id = list(all_training_tasks_for_test.keys())[0]
        sample_task_data = all_training_tasks_for_test[first_task_id]
        if sample_task_data:
            plot_task(sample_task_data)
            print(f"Successfully plotted test task: {first_task_id}")
        else:
            print("Could not get data for the first task.")
    else:
        print(f"Skipping task plot test: Training file not found at {TRAINING_JSON_PATH}")
except Exception as e:
    print(f"Error loading/plotting task: {e}")
    traceback.print_exc()


# --- VQ-VAE Layer ---
# --- Replace the VQEmbedding class in Cell 5 with this ---
class VQEmbedding(layers.Layer):
    """Vector Quantizer Layer."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, name="vq_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost # Beta

        # Initialize the codebook (embeddings)
        initializer = tf.keras.initializers.RandomUniform(-1./num_embeddings, 1./num_embeddings)
        self.embeddings = tf.Variable(
            initial_value=initializer(shape=(embedding_dim, num_embeddings)),
            trainable=True, name="codebook") # Shape [D, K]

    def call(self, inputs):
        # inputs shape: (Batch, Sequence, Dimension) or (Batch, Dimension)
        # Ensure input has rank 3 for broadcasting, even if sequence len is 1
        input_shape = tf.shape(inputs)
        # Flatten input to (Batch * Sequence, Dimension)
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        # --- Calculate distances ---
        # distances = ||z_e(x) - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2*z_e.T*e_k
        # Use broadcasting for efficiency
        # inputs are [N, D], embeddings are [D, K]
        term1 = tf.reduce_sum(tf.square(flat_inputs), axis=1, keepdims=True) # [N, 1]
        term2 = tf.reduce_sum(tf.square(self.embeddings), axis=0, keepdims=True) # [1, K]
        term3 = 2 * tf.matmul(flat_inputs, self.embeddings) # [N, K]
        distances = term1 + term2 - term3 # [N, K]

        # --- Find nearest neighbours ---
        encoding_indices = tf.argmin(distances, axis=1) # [N,]
        # Convert indices to one-hot encodings (optional, sometimes useful)
        # encodings = tf.one_hot(encoding_indices, self.num_embeddings) # [N, K]

        # --- Quantize ---
        # Use indices to get vectors from codebook
        # Need to transpose embeddings to [K, D] for gather
        quantized = tf.gather(tf.transpose(self.embeddings), encoding_indices) # [N, D]
        # Reshape back to original input shape (if rank 3)
        quantized = tf.reshape(quantized, input_shape)

        # --- Calculate VQ Loss ---
        # Commitment Loss (beta * ||sg[e_k] - z_e(x)||^2)
        e_latent_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized) - inputs))
        # Codebook Loss (||e_k - sg[z_e(x)]||^2) - Note: This is optimized via the EMA update usually,
        # but for direct loss calculation, sometimes included. Here we focus on commitment loss.
        # q_latent_loss = tf.reduce_mean(tf.square(quantized - tf.stop_gradient(inputs)))

        # The loss added to the model encourages the encoder output (inputs) to commit to an embedding
        vq_loss = self.commitment_cost * e_latent_loss
        self.add_loss(vq_loss * VQ_LOSS_WEIGHT) # Apply weight here

        # --- Straight-Through Estimator ---
        # Pass gradients from decoder input back to encoder output
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        # Optional: Return indices for analysis if needed
        # return quantized, encoding_indices
        return quantized

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "commitment_cost": self.commitment_cost,
        })
        return config
# --- End Replacement ---

# --- Wavelet KAN Encoder ---
class WaveletKANEncoder(layers.Layer):
    # ... implementation ...
    def __init__(self, kan_hidden_units, output_dim, vq_layer, name="wavelet_kan_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.kan_hidden_units = kan_hidden_units
        self.output_dim = output_dim
        self.vq_layer = vq_layer # Pass the VQ layer instance
        # Use global MAX_COEFF_LEN defined in Cell 2
        self.coeff_len = MAX_COEFF_LEN
        self.kan_layer1 = KANLayer(self.coeff_len, kan_hidden_units, name="kan_layer1")
        self.dense_post_kan = layers.Dense(output_dim, activation='relu', name="dense_post_kan")
        self.layer_norm = layers.LayerNormalization(name="layer_norm_post_kan")

    def call(self, inputs, training=None):
        inputs_float = tf.cast(inputs, tf.float32)
        # tf_dwt2_wrapper uses the globally defined WAVELET, WAVELET_LEVELS
        wavelet_coeffs = tf_dwt2_wrapper(inputs_float)
        x = self.kan_layer1(wavelet_coeffs)
        encoded = self.dense_post_kan(x)
        encoded = self.layer_norm(encoded)
        quantized_encoded = self.vq_layer(encoded) # Pass through VQ
        return quantized_encoded

# --- Size Predictor Head ---
class SizePredictorHead(layers.Layer):
    # ... implementation ...
    def __init__(self, name="size_predictor_head", **kwargs):
        super().__init__(name=name, **kwargs)
        # Use global D_MODEL, MAX_GRID_SIZE
        self.dense1 = layers.Dense(D_MODEL // 2, activation='relu')
        self.dense_h = layers.Dense(1)
        self.dense_w = layers.Dense(1)

    def call(self, transformer_output):
        # Pass the input through the intermediate dense layer
        x = self.dense1(transformer_output) # Shape: (Batch, D_MODEL // 2)

        # Calculate the raw predictions using the specific dense layers
        pred_h_raw = self.dense_h(x) # Shape: (Batch, 1)
        pred_w_raw = self.dense_w(x) # Shape: (Batch, 1)

        # Now use pred_h_raw and pred_w_raw
        # Apply sigmoid, scale to [1, MAX_GRID_SIZE] range
        pred_h_float = tf.nn.sigmoid(pred_h_raw) * (MAX_GRID_SIZE - 1.0) + 1.0 # Use float literal
        pred_w_float = tf.nn.sigmoid(pred_w_raw) * (MAX_GRID_SIZE - 1.0) + 1.0 # Use float literal

        # Get integer predictions (round and clip)
        pred_h_int = tf.cast(tf.round(pred_h_float), dtype=tf.int32)
        pred_w_int = tf.cast(tf.round(pred_w_float), dtype=tf.int32)
        pred_h_int = tf.clip_by_value(pred_h_int, 1, MAX_GRID_SIZE)
        pred_w_int = tf.clip_by_value(pred_w_int, 1, MAX_GRID_SIZE)

        # Return all four values
        return pred_h_float, pred_w_float, pred_h_int, pred_w_int

class RecurrentGridDecoder(layers.Layer):
    def __init__(self, num_steps=RECURRENT_STEPS, lstm_filters=CONV_LSTM_FILTERS, num_colors=NUM_COLORS, name="recurrent_grid_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_steps = num_steps # Note: Simplified implementation below might not use multiple steps effectively
        self.lstm_filters = lstm_filters
        self.num_colors = num_colors
        self.start_h, self.start_w = 4, 4 # Example starting size
        self.init_filters = lstm_filters

        self.dense_init = layers.Dense(self.start_h * self.start_w * self.init_filters, activation='relu', name="dec_dense_init")
        self.reshape = layers.Reshape((self.start_h, self.start_w, self.init_filters), name="dec_reshape")

        # ConvLSTM layer - return_sequences=False to get only the last output state
        self.conv_lstm = layers.ConvLSTM2D(filters=self.lstm_filters, kernel_size=(3, 3),
                                          padding='same', return_sequences=False,
                                          return_state=True, name="conv_lstm") # return_state=True gives final h, c

        # Upsampling layers
        self.deconv1 = layers.Conv2DTranspose(lstm_filters // 2, 3, strides=2, padding='same', activation='relu', name="dec_deconv1")
        self.norm1 = layers.LayerNormalization(name="dec_norm1")
        self.deconv2 = layers.Conv2DTranspose(lstm_filters // 4, 3, strides=2, padding='same', activation='relu', name="dec_deconv2")
        self.norm2 = layers.LayerNormalization(name="dec_norm2")
        self.deconv3 = layers.Conv2DTranspose(lstm_filters // 8, 3, strides=2, padding='same', activation='relu', name="dec_deconv3")
        self.norm3 = layers.LayerNormalization(name="dec_norm3")

        # Final convolution to get color logits
        self.conv_final = layers.Conv2D(num_colors, 1, padding='valid', activation=None, name="dec_conv_final")

    def call(self, inputs, training=None):
        transformer_vector, target_h_int, target_w_int = inputs
        batch_size = tf.shape(transformer_vector)[0]

        # 1. Initial Projection and Reshape
        initial_state_flat = self.dense_init(transformer_vector)
        initial_state_spatial = self.reshape(initial_state_flat) # Shape: (B, start_h, start_w, init_filters)

        # 2. ConvLSTM Step
        # Create a dummy time sequence of length 1 using the initial spatial state
        lstm_input_sequence = tf.expand_dims(initial_state_spatial, axis=1) # Shape: (B, 1, start_h, start_w, init_filters)

        # Get the final hidden state (h) and carry state (c)
        # We only need the hidden state (lstm_output_h) to feed into the upsampling path
        _, lstm_output_h, _ = self.conv_lstm(lstm_input_sequence, training=training)
        # lstm_output_h shape: (B, start_h, start_w, lstm_filters)

        # 3. Upsampling Deconvolutions
        x = self.deconv1(lstm_output_h)
        x = self.norm1(x, training=training)
        x = self.deconv2(x)
        x = self.norm2(x, training=training)
        x = self.deconv3(x)
        x = self.norm3(x, training=training)

        # 4. Final Color Logit Prediction
        # THIS IS WHERE 'logits' IS DEFINED
        logits = self.conv_final(x) # Shape: (B, upsampled_h, upsampled_w, num_colors)

        # 5. --- Dynamic Cropping/Padding Logic ---
        current_h, current_w = tf.shape(logits)[1], tf.shape(logits)[2]

        # --- MODIFICATION START: Use tf.cond ---
        # Define functions for tf.cond
        true_fn_h = lambda: target_h_int[0, 0] # Case if rank is 2
        false_fn_h = lambda: target_h_int[0]   # Case if rank is not 2 (e.g., 1)
        # Use tf.cond to select the correct slicing based on rank
        th = tf.cond(tf.equal(tf.rank(target_h_int), 2), true_fn_h, false_fn_h)

        # Repeat for width
        true_fn_w = lambda: target_w_int[0, 0]
        false_fn_w = lambda: target_w_int[0]
        tw = tf.cond(tf.equal(tf.rank(target_w_int), 2), true_fn_w, false_fn_w)

        # Ensure dtype after tf.cond
        th = tf.cast(th, tf.int32)
        tw = tf.cast(tw, tf.int32)
        # --- MODIFICATION END ---


        # Calculate cropping size
        crop_h_size = tf.minimum(current_h, th)
        crop_w_size = tf.minimum(current_w, tw)

        # Crop the logits tensor
        logits_maybe_cropped = tf.slice(logits,
                                        [0, 0, 0, 0],
                                        tf.stack([batch_size, crop_h_size, crop_w_size, self.num_colors]))

        # Calculate padding amounts
        pad_h = tf.maximum(0, th - crop_h_size)
        pad_w = tf.maximum(0, tw - crop_w_size)

        # Apply padding
        paddings = [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        final_logits = tf.pad(logits_maybe_cropped, paddings, constant_values=0) # Pad logits with 0

        return final_logits


# Add this class definition to Cell 5

# --- Transformer Encoder Block ---
class TransformerEncoderBlock(layers.Layer):
    """Implements a standard Transformer Encoder block."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, name="transformer_encoder_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Multi-Head Attention sub-layer
        # Note: key_dim is often embed_dim // num_heads
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=rate # Added dropout within MHA
            )

        # Feed Forward Network sub-layer
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )

        # Layer Normalization sub-layers
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout sub-layers
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        # --- Multi-Head Attention Path ---
        # Self-attention: query, value, key are all the same input
        attn_output = self.att(query=inputs, value=inputs, key=inputs, training=training) # Pass training flag
        # Apply dropout after attention
        attn_output = self.dropout1(attn_output, training=training) # Pass training flag
        # Add & Norm (Residual connection 1)
        out1 = self.layernorm1(inputs + attn_output)

        # --- Feed Forward Path ---
        ffn_output = self.ffn(out1)
        # Apply dropout after feed forward
        ffn_output = self.dropout2(ffn_output, training=training) # Pass training flag
        # Add & Norm (Residual connection 2)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# --- Categorical Size Predictor Head ---
class CategoricalSizePredictorHead(layers.Layer):
    def __init__(self, max_dim=MAX_GRID_SIZE, name="categorical_size_predictor_head", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_dim = max_dim
        # Use global D_MODEL
        self.dense1 = layers.Dense(D_MODEL, activation='relu', kernel_initializer='glorot_uniform')
        self.dense_h = layers.Dense(max_dim, kernel_initializer='glorot_uniform') # Output logits for each possible height (1 to max_dim)
        self.dense_w = layers.Dense(max_dim, kernel_initializer='glorot_uniform') # Output logits for each possible width (1 to max_dim)

    def call(self, transformer_output):
        # Input shape: (Batch, D_MODEL)
        x = self.dense1(transformer_output)
        height_logits = self.dense_h(x) # Shape: (Batch, max_dim)
        width_logits = self.dense_w(x) # Shape: (Batch, max_dim)
        return height_logits, width_logits


# --- Main ARC Research Model (v2 - Categorical Size) ---
class ArcResearchModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use global hyperparameters
        self.vq_layer = VQEmbedding(VQ_NUM_EMBEDDINGS, VQ_EMBEDDING_DIM, VQ_COMMITMENT_COST)
        self.grid_encoder = WaveletKANEncoder(KAN_HIDDEN_UNITS, ENCODER_DIM, self.vq_layer)
        self.type_embedding = layers.Embedding(3, D_MODEL, name="type_emb")
        self.transformer_blocks = [
            TransformerEncoderBlock(embed_dim=D_MODEL, num_heads=N_HEADS, ff_dim=FFN_DIM, rate=DROPOUT, name=f"transformer_block_{i}")
            for i in range(N_TRANSFORMER_LAYERS)
        ]
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-6, name="final_norm")
        # --- Use new Categorical Head ---
        self.size_predictor = CategoricalSizePredictorHead(max_dim=MAX_GRID_SIZE)
        # ---
        self.grid_decoder = RecurrentGridDecoder() # Decoder needs integer size input

    def call(self, inputs, training=None):
        all_grids, type_ids = inputs
        B = tf.shape(all_grids)[0]
        num_grids = tf.shape(all_grids)[1]
        H, W = tf.shape(all_grids)[2], tf.shape(all_grids)[3]

        reshaped_grids = tf.reshape(all_grids, [-1, H, W, 1])
        encoded_vectors = self.grid_encoder(reshaped_grids, training=training)
        sequence_vectors = tf.reshape(encoded_vectors, [B, num_grids, D_MODEL])

        type_emb = self.type_embedding(type_ids)
        transformer_input = sequence_vectors + type_emb

        h = transformer_input
        for transformer_block in self.transformer_blocks:
             h = transformer_block(h, training=training)
        transformer_output = self.final_layer_norm(h)

        test_vector = transformer_output[:, -1, :]

        # --- Get Size Logits ---
        height_logits, width_logits = self.size_predictor(test_vector)
        # ---

        # --- Get Integer Size Prediction for Decoder ---
        # Argmax gives 0-based index, add 1 for 1-based size
        pred_h_int = tf.argmax(height_logits, axis=-1, output_type=tf.int32) + 1
        pred_w_int = tf.argmax(width_logits, axis=-1, output_type=tf.int32) + 1
        # Add batch dimension back for decoder compatibility if needed (decoder expects [B, 1]?)
        # Check decoder's input requirement, assuming it needs rank 2:
        pred_h_int = tf.expand_dims(pred_h_int, axis=-1) # Shape: (B, 1)
        pred_w_int = tf.expand_dims(pred_w_int, axis=-1) # Shape: (B, 1)
        # Ensure clipping
        pred_h_int = tf.clip_by_value(pred_h_int, 1, MAX_GRID_SIZE)
        pred_w_int = tf.clip_by_value(pred_w_int, 1, MAX_GRID_SIZE)
        # ---

        decoder_inputs = (test_vector, pred_h_int, pred_w_int)
        output_logits = self.grid_decoder(decoder_inputs, training=training)

        # Return grid logits and size logits
        return output_logits, height_logits, width_logits


# --- Helper: compute size logits without decoder ---
def model_size_logits_only(model, inputs, training=False):
    all_grids, type_ids = inputs
    B = tf.shape(all_grids)[0]
    num_grids = tf.shape(all_grids)[1]
    H, W = tf.shape(all_grids)[2], tf.shape(all_grids)[3]

    reshaped_grids = tf.reshape(all_grids, [-1, H, W, 1])
    encoded_vectors = model.grid_encoder(reshaped_grids, training=training)
    sequence_vectors = tf.reshape(encoded_vectors, [B, num_grids, D_MODEL])

    type_emb = model.type_embedding(type_ids)
    h = sequence_vectors + type_emb
    for block in model.transformer_blocks:
        h = block(h, training=training)
    transformer_output = model.final_layer_norm(h)
    test_vector = transformer_output[:, -1, :]
    return model.size_predictor(test_vector)  # height_logits, width_logits



# --- Data Loading & Preprocessing ---
def load_arc_task(filepath):
    # ... implementation ...
    try:
        with open(filepath, 'r') as f: task_data = json.load(f)
        return task_data
    except Exception as e:
        # print(f"Error loading task {filepath}: {e}") # Keep errors quiet during train loop maybe
        return None
# Add this helper function at the start of Cell 7 or Cell 10
import numpy as np
import traceback

def extract_true_output_grid(solution_data, task_id="unknown"):
    """
    Accepts solution_data as dict or list and extracts the first output grid as a numpy array.
    Attempts to reshape 1D arrays to (N, 1). Raises informative errors otherwise.
    """
    raw_grid = None
    try:
        if isinstance(solution_data, dict):
            og = (solution_data.get('output_grids')
                  or solution_data.get('output_grid')
                  or solution_data.get('grids'))
            if og is None or not isinstance(og, list) or not og:
                raise KeyError(f"Task {task_id}: Could not find non-empty 'output_grids' list in dict solution_data.")
            raw_grid = og[0]

        elif isinstance(solution_data, list) and solution_data:
            first = solution_data[0]
            if isinstance(first, dict):
                og = (first.get('output_grids')
                      or first.get('output_grid')
                      or first.get('grids'))
                if og is None:
                    for item in solution_data:
                        if isinstance(item, dict):
                             og_item = (item.get('output_grids') or item.get('output_grid') or item.get('grids'))
                             if og_item is not None and isinstance(og_item, list) and og_item:
                                 og = og_item
                                 break
                    if og is None:
                         raise KeyError(f"Task {task_id}: No 'output_grids' key found in any item of list solution_data.")

                if not isinstance(og, list) or not og:
                     raise ValueError(f"Task {task_id}: Found 'output_grids' but it's not a non-empty list.")
                raw_grid = og[0]

            elif isinstance(first, (list, tuple, np.ndarray)):
                raw_grid = first
            else:
                 raise TypeError(f"Task {task_id}: First item in list solution_data is not a dict or grid. Type: {type(first)}")
        else:
             raise TypeError(f"Task {task_id}: Unexpected solution_data structure or empty list. Type: {type(solution_data)}")

        # Convert to numpy array
        grid_np = np.asarray(raw_grid, dtype=np.int32)

        # --- ADD 1D Reshape Logic ---
        if grid_np.ndim == 1:
            #print(f"  INFO (Task {task_id}): Reshaping 1D grid with shape {grid_np.shape} to ({grid_np.shape[0]}, 1)")
            grid_np = grid_np.reshape(-1, 1) # Reshape to N x 1
        # --- END 1D Reshape Logic ---

        # Final check must be 2D now
        if grid_np.ndim != 2:
             raise ValueError(f"Task {task_id}: Grid is not 1D or 2D after processing. Final shape: {grid_np.shape}")

        return grid_np

    except Exception as e:
        print(f"!!! ERROR extracting true output grid for task {task_id} !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        # traceback.print_exc() # Optional: full traceback
        raise # Re-raise to stop processing this task
# --- REFACTORED preprocess_task ---
# --- REFACTORED preprocess_task (v2 - with scaling factors) ---
def preprocess_task(task_data, task_id="unknown", mode='train', augment=False, test_output_grid_true=None):
    # Uses global MAX_GRID_SIZE, TRANSFORMATIONS

    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', [])

    # --- Calculate Scaling Factors (using original, unaugmented train pairs) ---
    h_scales = []
    w_scales = []
    if train_pairs: # Avoid division by zero if no train pairs
        for pair in train_pairs:
            try:
                in_grid_np = np.array(pair['input'], dtype=np.int32)
                out_grid_np = np.array(pair['output'], dtype=np.int32)
                in_h, in_w = in_grid_np.shape
                out_h, out_w = out_grid_np.shape
                if in_h > 0: h_scales.append(out_h / in_h)
                if in_w > 0: w_scales.append(out_w / in_w)
            except Exception:
                pass # Ignore errors in individual pairs for scaling calc
    # Use median to be robust to outliers, default to 1.0 if no valid scales
    mean_h_scale = np.median(h_scales) if h_scales else 1.0
    mean_w_scale = np.median(w_scales) if w_scales else 1.0
    scaling_factors = tf.constant([[mean_h_scale, mean_w_scale]], dtype=tf.float32)
    # --- End Scaling Factor Calculation ---


    # --- 1. Fixed Augmentation Logic ---
    transformation = 'identity'
    # Apply random transformation only if augment is True AND mode is train
    if augment and mode == 'train':
        transformation = np.random.choice(TRANSFORMATIONS)

    all_grids_list = []
    type_ids_list = []
    pad_h_target = MAX_GRID_SIZE
    pad_w_target = MAX_GRID_SIZE

    # --- 2. Process Train Pairs with Error Handling ---
    for i, pair in enumerate(train_pairs):
        try:
            in_grid_np = np.array(pair['input'], dtype=np.int32)
            out_grid_np = np.array(pair['output'], dtype=np.int32)
            # Apply augmentation *after* scaling factors are calculated
            in_grid_aug = augment_grid(in_grid_np, transformation)
            out_grid_aug = augment_grid(out_grid_np, transformation)

            all_grids_list.append(pad_grid(in_grid_aug, pad_h_target, pad_w_target))
            type_ids_list.append(0)
            all_grids_list.append(pad_grid(out_grid_aug, pad_h_target, pad_w_target))
            type_ids_list.append(1)
        except Exception as e:
            print(f"!!! ERROR processing training pair {i} in task {task_id} !!!")
            print(f"Transformation: {transformation}, Error: {e}")
            return None, None, None, None, None, None # Consistent error return (now 6 items)

    # --- 3. Process Test Input ---
    if not test_pairs:
        print(f"WARNING: Task {task_id} has no test pairs.")
        return None, None, None, None, None, None

    try:
        test_in_grid_np = np.array(test_pairs[0]['input'], dtype=np.int32)
        # Store original test input shape for heuristic calculation later
        original_test_input_shape = test_in_grid_np.shape
        test_in_grid_aug = augment_grid(test_in_grid_np, transformation)
        all_grids_list.append(pad_grid(test_in_grid_aug, pad_h_target, pad_w_target))
        type_ids_list.append(2)
    except Exception as e:
        print(f"!!! ERROR processing test pair 0 input in task {task_id} !!!")
        print(f"Transformation: {transformation}, Error: {e}")
        return None, None, None, None, None, None

    # --- 4. Prepare Target Tensors ---
    target_tensor = None
    original_output_shape = None
    target_size_tensor = None # This will be Int32

    if mode == 'train' or mode == 'evaluate':
        if test_output_grid_true is not None and isinstance(test_output_grid_true, np.ndarray):
            try:
                test_out_grid_orig = test_output_grid_true # Already extracted np array
                if test_out_grid_orig.ndim != 2:
                    raise ValueError(f"Grid for task {task_id} is not 2D (shape: {test_out_grid_orig.shape}).")
                original_output_shape = test_out_grid_orig.shape

                test_out_grid_aug = augment_grid(test_out_grid_orig, transformation)
                if test_out_grid_aug.ndim != 2:
                     raise ValueError(f"Grid for task {task_id} became non-2D after augmentation.")
                target_h_aug, target_w_aug = test_out_grid_aug.shape

                padded_test_out = pad_grid(test_out_grid_aug, pad_h_target, pad_w_target, pad_value=-1)
                target_tensor = tf.constant(padded_test_out, dtype=tf.int32)[tf.newaxis, ...]

                # Use Integer Target Size for categorical loss
                target_size_tensor = tf.constant([[target_h_aug, target_w_aug]], dtype=tf.int32) # [1, 2]

            except Exception as e:
                print(f"!!! ERROR during target tensor creation in task {task_id} !!!")
                print(f"Mode: {mode}, Transformation: {transformation}, Orig Shape: {original_output_shape}")
                print(f"Error Type: {type(e).__name__}, Error: {e}")
                target_tensor, target_size_tensor, original_output_shape = None, None, None
        # If test_output_grid_true is None, tensors remain None

    # --- 5. Stack Inputs ---
    model_inputs = None
    try:
        if not all_grids_list: raise ValueError("all_grids_list is empty.")
        all_grids_stacked = np.stack(all_grids_list, axis=0)
        all_grids_tensor = tf.constant(all_grids_stacked, dtype=tf.int32)[tf.newaxis, ...]
        type_ids_tensor = tf.constant(type_ids_list, dtype=tf.int32)[tf.newaxis, ...]
        model_inputs = (all_grids_tensor, type_ids_tensor)
    except ValueError as e:
        print(f"!!! ERROR stacking input grids for task {task_id} !!! Shapes:")
        for i, grid in enumerate(all_grids_list): print(f"  Grid {i} shape: {grid.shape}")
        print(f"Error: {e}")
        return None, None, None, None, None, None

    # --- 6. Consistent Return Structure (7 items) ---
    return (model_inputs, target_tensor, target_size_tensor, transformation,
            original_output_shape, scaling_factors, original_test_input_shape)


# --- Loss Components ---
def apply_tf_transform(grid_batch, transformation_str):
    # ... implementation ...
    # Assumes input B, H, W, C -> output B, H', W', C
    k = 0; flip_lr = False; flip_ud = False
    if transformation_str == 'rotate90': k = 1
    elif transformation_str == 'rotate180': k = 2
    elif transformation_str == 'rotate270': k = 3
    elif transformation_str == 'flip_lr': flip_lr = True
    elif transformation_str == 'flip_ud': flip_ud = True

    # Assume B, H, W, C format
    if k > 0: grid_batch = tf.image.rot90(grid_batch, k=k)
    if flip_lr: grid_batch = tf.image.flip_left_right(grid_batch)
    if flip_ud: grid_batch = tf.image.flip_up_down(grid_batch)
    return grid_batch

# Uses global loss weights (CE_LOSS_WEIGHT, SIZE_LOSS_WEIGHT, etc.)

# --- REFACTORED calculate_losses (v2 - Categorical Size Loss) ---

# NO @tf.function DECORATOR HERE
def calculate_losses(y_true_grid, y_pred_grid_logits, # Grid prediction
                     y_true_size_int, y_pred_height_logits, y_pred_width_logits, # Size prediction (logits + true int)

                     # Equivariance inputs
                     transformation, identity_grid_logits,

                     # Explicit losses from model pass
                     main_vq_loss, main_reg_loss):
    """Calculates all losses using categorical size loss."""

    total_loss = 0.0
    losses_dict = {}
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # --- Grid CE Loss ---
    # (Keep the existing CE loss calculation with masking from previous fixes)
    y_true_grid = tf.cast(y_true_grid, dtype=tf.int32)
    pred_shape = tf.shape(y_pred_grid_logits)
    B, pred_h, pred_w, C = pred_shape[0], pred_shape[1], pred_shape[2], pred_shape[3]
    y_true_sliced = y_true_grid[:, :pred_h, :pred_w]
    y_pred_logits_flat = tf.reshape(y_pred_grid_logits, [-1, C])
    y_true_flat = tf.reshape(y_true_sliced, [-1])
    valid_mask_flat = tf.math.greater_equal(y_true_flat, 0)
    valid_logits = tf.boolean_mask(y_pred_logits_flat, valid_mask_flat)
    valid_targets = tf.boolean_mask(y_true_flat, valid_mask_flat)
    ce_loss = tf.constant(0.0, dtype=tf.float32)
    valid_pixel_count = tf.shape(valid_targets)[0]
    if valid_pixel_count > 0:
        ce_loss_per_pixel = scce(valid_targets, valid_logits) # Use SCCE here too
        ce_loss = tf.reduce_mean(ce_loss_per_pixel) # Reduce mean if SCCE doesn't
    mask = tf.not_equal(y_true_sliced, -1)
    mask_float = tf.cast(mask, dtype=tf.float32)
    valid_pixels = tf.reduce_sum(mask_float) + 1e-8
    total_loss += ce_loss * CE_LOSS_WEIGHT
    losses_dict['ce_loss'] = ce_loss

    # --- Size Prediction Loss (Sparse Categorical Cross Entropy) ---
    # y_true_size_int has shape [1, 2] containing [true_h, true_w] (1-based)
    # y_pred_height_logits / y_pred_width_logits have shape [1, MAX_GRID_SIZE]
    # SCCE expects 0-based labels
    true_h_label = y_true_size_int[:, 0] - 1 # Shape [1]
    true_w_label = y_true_size_int[:, 1] - 1 # Shape [1]

    # Calculate loss - ensure labels are correct shape for SCCE ([Batch,])
    size_loss_h = scce(true_h_label, y_pred_height_logits)
    size_loss_w = scce(true_w_label, y_pred_width_logits)

    size_loss = (size_loss_h + size_loss_w) / 2.0
    total_loss += size_loss * SIZE_LOSS_WEIGHT
    losses_dict['size_loss'] = size_loss
    # --- End Size Loss ---


    # --- Lp Losses (Placeholder) ---
    # (Keep as is)
    losses_dict['l1_loss'] = 0.0
    losses_dict['l2_loss'] = 0.0
    losses_dict['linf_loss'] = 0.0

    # --- VQ Loss ---
    # (Keep as is)
    total_loss += main_vq_loss
    losses_dict['vq_loss'] = main_vq_loss / VQ_LOSS_WEIGHT if VQ_LOSS_WEIGHT != 0 else 0.0

    # --- Equivariance Loss ---
    # (Keep as is, but ensure 'identity_grid_logits' is used correctly)
    equiv_loss = tf.constant(0.0, dtype=tf.float32)
    if EQUI_LOSS_WEIGHT > 0 and transformation != 'identity' and valid_pixels > 1e-7:
        try:
            # T(f(x)) using identity_grid_logits
            identity_probs = tf.nn.softmax(identity_grid_logits, axis=-1)
            transformed_identity_probs = apply_tf_transform(identity_probs, transformation)
            # f(T(x)) using y_pred_grid_logits
            transformed_probs = tf.nn.softmax(y_pred_grid_logits, axis=-1)

            shape_transformed = tf.shape(transformed_probs)
            shape_id_transformed = tf.shape(transformed_identity_probs)
            if tf.reduce_all(tf.equal(shape_transformed, shape_id_transformed)):
                 equiv_diff_probs = tf.square(transformed_probs - transformed_identity_probs)
                 equiv_loss_unreduced = tf.reduce_mean(equiv_diff_probs, axis=-1)
                 mask_equiv = mask_float
                 valid_pixels_equiv = valid_pixels
                 masked_equiv_loss = equiv_loss_unreduced * mask_equiv
                 equiv_loss = tf.reduce_sum(masked_equiv_loss) / valid_pixels_equiv
        except Exception as e:
            tf.print("Warning: Equivariance loss calculation failed:", e)
            equiv_loss = tf.constant(0.0, dtype=tf.float32)
    total_loss += equiv_loss * EQUI_LOSS_WEIGHT
    losses_dict['equiv_loss'] = equiv_loss

    # --- Model Regularization Losses ---
    # (Keep as is)
    total_loss += main_reg_loss
    losses_dict['internal_reg_loss'] = main_reg_loss

    return total_loss, losses_dict


# --- REFACTORED tf_train_step (v2 - Categorical Size) ---

@tf.function
def tf_train_step(model, optimizer, model_inputs_transformed, target_grid_aug, target_size_aug_int, # Pass INT size now
                  model_inputs_identity, transformation):
    """Performs the forward/backward pass with categorical size prediction."""

    with tf.GradientTape() as tape:
        # --- Main Forward pass: f(T(x)) ---
        # Returns grid_logits, height_logits, width_logits
        output_grid_logits, pred_height_logits, pred_width_logits = model(model_inputs_transformed, training=True)

        # --- CAPTURE LOSSES from *this* pass ---
        main_vq_losses = model.vq_layer.losses
        main_vq_loss = tf.add_n(main_vq_losses) if main_vq_losses else tf.constant(0.0)
        main_reg_losses = [l for l in model.losses if l not in main_vq_losses]
        main_reg_loss = tf.add_n(main_reg_losses) if main_reg_losses else tf.constant(0.0)

        # --- Equivariance Forward Pass: f(x) ---
        # Returns grid_logits, height_logits, width_logits - we only need grid_logits
        identity_grid_logits, _, _ = model(model_inputs_identity, training=False)

        # --- Calculate all losses ---
        final_loss, losses_dict = calculate_losses(
            y_true_grid=target_grid_aug,
            y_pred_grid_logits=output_grid_logits,        # Grid logits f(T(x))

            y_true_size_int=target_size_aug_int,          # True size (Int32)
            y_pred_height_logits=pred_height_logits,      # Height logits f(T(x))
            y_pred_width_logits=pred_width_logits,        # Width logits f(T(x))

            transformation=transformation,
            identity_grid_logits=identity_grid_logits,    # Grid logits f(x)

            main_vq_loss=main_vq_loss,
            main_reg_loss=main_reg_loss
        )

        losses_dict['total_loss_calc'] = final_loss

    # --- Compute and Apply Gradients ---
    # (Keep gradient calculation and application logic as is)
    trainable_vars = model.trainable_variables
    grads = tape.gradient(final_loss, trainable_vars)
    filtered_grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    if not filtered_grads_and_vars:
         tf.print("Warning: No valid gradients found for task. Skipping optimizer step.")
         return final_loss, losses_dict
    filtered_grads = [g for g, v in filtered_grads_and_vars]
    filtered_vars = [v for g, v in filtered_grads_and_vars]
    if CLIP_NORM > 0:
        filtered_grads, _ = tf.clip_by_global_norm(filtered_grads, CLIP_NORM)
    optimizer.apply_gradients(zip(filtered_grads, filtered_vars))

    return final_loss, losses_dict





# --- REFACTORED evaluate_model (v2 - Heuristic + Categorical Size) ---

def evaluate_model(model, all_evaluation_tasks, all_evaluation_solutions, epoch_num=-1):
    print(f"--- RUNNING NEW EVALUATE_MODEL v3 (Heuristic + Categorical) ---") # Update print message
    eval_task_ids = list(all_evaluation_tasks.keys())
    # ... (initialize counters) ...
    total_tasks = 0
    correct_tasks_exact = 0
    correct_tasks_size_only = 0
    total_pixel_acc = 0.0
    total_size_err = 0.0
    heuristic_matches_categorical = 0
    size_correct_count = 0 # Use this instead of correct_tasks_size_only initially

    print(f"\nStarting evaluation on {len(eval_task_ids)} tasks...")
    start_time = time.time()
    task_results = []

    for i, task_id in enumerate(eval_task_ids):
        task_data = all_evaluation_tasks.get(task_id)
        solution_data = all_evaluation_solutions.get(task_id)

        if task_data is None or solution_data is None:
             print(f"  Skipping eval task {task_id} (data or solution is None).")
             task_results.append({'file': task_id, 'status': 'load_error'})
             continue

        try:
            true_output_grid_np = extract_true_output_grid(solution_data, task_id)
            original_output_shape = true_output_grid_np.shape
            true_h, true_w = original_output_shape
        except Exception as e:
            print(f"  Skipping {task_id} due to error processing solution grid.")
            task_results.append({'file': task_id, 'status': 'solution_data_error'})
            continue

        total_tasks += 1
        task_status = 'fail'
        pixel_acc = 0.0
        size_correct = False
        size_err = 0.0

        try:
            # --- Preprocess (returns 7 items now) ---
            (model_inputs, target_grid, target_size,       # Vals 1, 2, 3
             transformation, _,                            # Val 4, ignore 5
             scaling_factors, original_test_input_shape    # Vals 6, 7
            ) = preprocess_task(
                 task_data,
                 task_id=task_id,
                 mode='evaluate',
                 augment=False,
                 test_output_grid_true=true_output_grid_np
            )
            if model_inputs is None: # Handle preprocessing failure
                 print(f"  Skipping task {task_id} (preprocess error).")
                 task_results.append({'file': task_id, 'status': 'preprocess_error'})
                 continue

            # --- Predict (Model returns grid_logits, height_logits, width_logits) ---
            output_grid_logits, height_logits, width_logits = model(model_inputs, training=False)

            # --- Calculate Heuristic Size ---
            test_in_h, test_in_w = original_test_input_shape
            h_scale, w_scale = scaling_factors.numpy()[0]
            h_heuristic = tf.clip_by_value(tf.cast(tf.round(test_in_h * h_scale), tf.int32), 1, MAX_GRID_SIZE).numpy().item()
            w_heuristic = tf.clip_by_value(tf.cast(tf.round(test_in_w * w_scale), tf.int32), 1, MAX_GRID_SIZE).numpy().item()

            # --- Get Categorical Size Prediction ---
            h_categorical = (tf.argmax(height_logits, axis=-1, output_type=tf.int32) + 1).numpy()[0] # [0] to get scalar from batch 1
            w_categorical = (tf.argmax(width_logits, axis=-1, output_type=tf.int32) + 1).numpy()[0] # [0] to get scalar from batch 1
            h_categorical = np.clip(h_categorical, 1, MAX_GRID_SIZE) # Clip numpy int
            w_categorical = np.clip(w_categorical, 1, MAX_GRID_SIZE) # Clip numpy int

            # --- Log Mismatch ---
            if h_heuristic == h_categorical and w_heuristic == w_categorical:
                heuristic_matches_categorical += 1
            else:
                 print(f"  Task {task_id}: Size mismatch! Heuristic ({h_heuristic},{w_heuristic}) vs Categorical ({h_categorical},{w_categorical})")

            # --- Use CATEGORICAL prediction for evaluation ---
            pred_h, pred_w = h_categorical, w_categorical

            # --- Compare Size vs Ground Truth ---
            size_err = abs(pred_h - true_h) + abs(pred_w - true_w)
            total_size_err += size_err
            if pred_h == true_h and pred_w == true_w:
                size_correct = True
                size_correct_count += 1 # Use separate counter

            # --- Extract Predicted Grid Indices ---
            pred_grid_indices = tf.argmax(output_grid_logits, axis=-1, output_type=tf.int32)[0] # Shape [pred_decoder_h, pred_decoder_w]

            # --- Compare Grid Content (using robust method) ---
            exact_match = False
            pixel_acc = 0.0
            if size_correct:
                target_np_full = target_grid.numpy()[0]
                target_unpadded = target_np_full[:true_h, :true_w]

                pred_raw_np = pred_grid_indices.numpy()
                pred_raw_h, pred_raw_w = pred_raw_np.shape
                pred_unpadded = np.full((true_h, true_w), 0, dtype=np.int32)
                copy_h = min(pred_raw_h, true_h)
                copy_w = min(pred_raw_w, true_w)
                if copy_h > 0 and copy_w > 0:
                   pred_unpadded[:copy_h, :copy_w] = pred_raw_np[:copy_h, :copy_w]

                valid_mask = target_unpadded != -1
                total_valid_pixels = np.sum(valid_mask)
                if total_valid_pixels > 0:
                    correct_pixels = np.sum(pred_unpadded[valid_mask] == target_unpadded[valid_mask])
                    pixel_acc = correct_pixels / total_valid_pixels
                    if np.array_equal(pred_unpadded[valid_mask], target_unpadded[valid_mask]):
                           exact_match = True
                elif pred_unpadded.size == 0 and target_unpadded.size == 0:
                    exact_match = True
                    pixel_acc = 1.0

                if exact_match:
                    correct_tasks_exact += 1
                    task_status = 'success'
                else:
                    task_status = 'size_correct_pixels_wrong'
            else:
                task_status = 'size_wrong'
                pixel_acc = 0.0

            total_pixel_acc += pixel_acc
            task_results.append({
                'file': task_id, 'status': task_status,
                'size_pred_cat': (int(h_categorical), int(w_categorical)), # Log both predictions
                'size_pred_heu': (int(h_heuristic), int(w_heuristic)),
                'size_true': (int(true_h), int(true_w)),
                'pixel_acc': float(pixel_acc)})

        except Exception as e:
            print(f"\n--- Error evaluating task {task_id} ---")
            print(f"Error Type: {type(e).__name__}, Error: {e}")
            traceback.print_exc()
            task_results.append({'file': task_id, 'status': 'prediction_error'})

    # --- Evaluation Summary ---
    end_time = time.time()
    avg_pixel_acc = (total_pixel_acc / total_tasks) if total_tasks > 0 else 0
    task_success_rate = (correct_tasks_exact / total_tasks) if total_tasks > 0 else 0
    size_success_rate = (size_correct_count / total_tasks) if total_tasks > 0 else 0 # Use the counter
    avg_size_err = (total_size_err / total_tasks) if total_tasks > 0 else 0
    match_rate = (heuristic_matches_categorical / total_tasks) if total_tasks > 0 else 0

    print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")
    print(f"Average Pixel Accuracy (on tasks where prediction ran): {avg_pixel_acc:.4f}")
    print(f"Average Size Error (Manhattan distance, using categorical pred): {avg_size_err:.4f}")
    print(f"Size Prediction Accuracy (Exact Match, using categorical pred): {size_success_rate:.4f} ({size_correct_count}/{total_tasks})")
    print(f"Task Success Rate (Exact Grid Match): {task_success_rate:.4f} ({correct_tasks_exact}/{total_tasks})")
    print(f"Heuristic vs Categorical Size Match Rate: {match_rate:.4f} ({heuristic_matches_categorical}/{total_tasks})") # Log match rate

    # (Keep results saving logic)
    results_filename = f"/kaggle/working/eval_results_epoch_{epoch_num}.json"
    try:
        with open(results_filename, 'w') as f: json.dump(task_results, f, indent=2)
        print(f"Saved detailed evaluation results to {results_filename}")
    except Exception as e: print(f"Error saving detailed eval results: {e}")

    return task_success_rate


# --- Load Full Datasets ---
print("Loading full training and evaluation JSON files...")
try:
    # Use the correct JSON file paths defined in Cell 2
    with open(TRAINING_JSON_PATH, 'r') as f:
        all_training_tasks = json.load(f)
    print(f"Loaded {len(all_training_tasks)} training tasks.")

    with open(EVALUATION_JSON_PATH, 'r') as f:
        all_evaluation_tasks = json.load(f)
    print(f"Loaded {len(all_evaluation_tasks)} evaluation tasks.")
    # --- ADD: Load Solutions ---
    with open(TRAINING_SOLUTIONS_JSON_PATH, 'r') as f:
        all_training_solutions = json.load(f)
    print(f"Loaded {len(all_training_solutions)} training solutions.")

    with open(EVALUATION_SOLUTIONS_JSON_PATH, 'r') as f:
        all_evaluation_solutions = json.load(f)
    print(f"Loaded {len(all_evaluation_solutions)} evaluation solutions.")
    # --- END ADD ---

except Exception as e:
    print(f"FATAL ERROR: Could not load JSON datasets: {e}")
    traceback.print_exc()
    # Consider stopping execution if loading fails
    raise  # Re-raise the exception to stop the notebook

# --- Get Task IDs ---
train_task_ids = list(all_training_tasks.keys())
eval_task_ids = list(all_evaluation_tasks.keys())
# Optional: Check if solution keys match challenge keys
train_solution_ids = set(all_training_solutions.keys())
eval_solution_ids = set(all_evaluation_solutions.keys())
if set(train_task_ids) != train_solution_ids:
    print("WARNING: Mismatch between training task IDs and training solution IDs!")
if set(eval_task_ids) != eval_solution_ids:
    print("WARNING: Mismatch between evaluation task IDs and evaluation solution IDs!")
print(f"Found {len(train_task_ids)} training tasks and {len(eval_task_ids)} evaluation tasks (with corresponding solutions).")


# --- Instantiate Model and Optimizer ---
# Uses LEARNING_RATE from Cell 2
model = ArcResearchModel()
optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE)

# --- Build the Model ---
# Use the first task ID from the loaded training data
print("Building model (can take a moment)...")
build_success = False
if train_task_ids: # Check if there are any training tasks
    try:
        dummy_task_id = train_task_ids[0] # Get the first ID
        dummy_task_data = all_training_tasks[dummy_task_id] # Get data using the ID

        # --- ADD: Get the solution for the dummy task ---
        # Assuming tasks usually have one test case, get the first solution grid
        if dummy_task_id in all_training_solutions:
             dummy_solution_grid = all_training_solutions[dummy_task_id][0] # Get the grid for test pair 0
             # --- END ADD ---

             if dummy_task_data and dummy_solution_grid:
                 # --- MODIFY: Pass solution grid AND task_id to preprocess_task ---
                # Extract the grid first using the robust extractor
                dummy_solution_grid_np = extract_true_output_grid(dummy_solution_grid, dummy_task_id)

                d_inputs, d_target_g, d_target_s, _, _,_, _ = preprocess_task(
                    dummy_task_data,
                    task_id=dummy_task_id, # <-- Pass task_id
                    mode='train', # Build using train mode logic
                    augment=False, # No augmentation needed for build
                    test_output_grid_true=dummy_solution_grid_np # Pass the extracted grid
                )
                # --- END MODIFY ---
                if d_inputs and d_target_g is not None and d_target_s is not None:
                    _ = model(d_inputs, training=False)
                    model.summary(expand_nested=True)
                    print("Model built successfully.")
                    build_success = True
                else:
                     print(f"ERROR: Failed to preprocess dummy task {dummy_task_id} even with solution provided.")
             else:
                  print(f"ERROR: Could not retrieve data or solution for the first training task ID ({dummy_task_id}) for build.")
        else:
            print(f"ERROR: Solution not found for dummy task ID ({dummy_task_id}) in training solutions file.")
    except Exception as e:
        print(f"Error building model: {e}")
        traceback.print_exc()
else:
    print("ERROR: No training tasks found in the JSON file to build the model.")

if not build_success:
     print("!!! MODEL BUILD FAILED. Cannot proceed with training. !!!")
     # Optionally raise an error: raise RuntimeError("Model build failed")


# --- Warmup: train size head only (no decoder) ---
print("Starting size-head warmup (no decoder)...")
warmup_lr = 1e-3
warmup_epochs = 2
size_vars = model.size_predictor.trainable_variables + model.final_layer_norm.trainable_variables
opt_size = keras.optimizers.Adam(learning_rate=warmup_lr)

train_ids = list(all_training_tasks.keys())
for e in range(warmup_epochs):
    np.random.shuffle(train_ids)
    correct, total = 0, 0
    for i, task_id in enumerate(train_ids):
        try:
            if task_id not in all_training_solutions:
                continue
            sol_grid = all_training_solutions[task_id][0]
            sol_np = extract_true_output_grid(sol_grid, task_id)
            (model_inputs, _, target_size, _, _, _, _) = preprocess_task(
                all_training_tasks[task_id], task_id=task_id,
                mode='train', augment=True, test_output_grid_true=sol_np)
            if model_inputs is None or target_size is None:
                continue

            with tf.GradientTape() as tape:
                h_logits, w_logits = model_size_logits_only(model, model_inputs, training=True)
                scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                y_h = target_size[:, 0] - 1  # 0-based
                y_w = target_size[:, 1] - 1
                loss_h = scce(y_h, h_logits)
                loss_w = scce(y_w, w_logits)
                loss = (loss_h + loss_w) * 0.5

            grads = tape.gradient(loss, size_vars)
            opt_size.apply_gradients(list(zip(grads, size_vars)))

            pred_h = int((tf.argmax(h_logits, -1) + 1).numpy()[0])
            pred_w = int((tf.argmax(w_logits, -1) + 1).numpy()[0])
            th = int(target_size.numpy()[0,0]); tw = int(target_size.numpy()[0,1])
            if pred_h == th and pred_w == tw:
                correct += 1
            total += 1
            if (i+1) % 200 == 0:
                acc = correct/max(1,total)
                print(f"  Warmup e{e+1} {i+1}/{len(train_ids)} loss={float(loss):.4f} acc={acc:.3f}")
        except Exception:
            pass
    print(f"Warmup epoch {e+1}: size acc={correct/max(1,total):.3f} ({correct}/{total})")
print("Warmup complete.")

# Temporarily boost size loss weight for early epochs
SIZE_LOSS_WEIGHT = 50



# --- Training Setup ---
# Uses EPOCHS from Cell 2
if build_success: # Only proceed if model built successfully
    print("\nStarting training...")
    best_eval_success_rate = -1.0
    # No need for tf.data.Dataset.from_tensor_slices anymore

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        epoch_losses = Counter() # Use Counter to sum losses by name
        tasks_processed = 0

        # Shuffle the task IDs each epoch
        shuffled_train_ids = np.random.permutation(train_task_ids)

        # Iterate over the shuffled task IDs
        for i, task_id in enumerate(shuffled_train_ids):
            step_start = time.time()
            try:
                # 1. Get task data from the pre-loaded dictionary
                task_data = all_training_tasks[task_id]
                 # --- ADD: Get the corresponding training solution ---
                if task_id not in all_training_solutions:
                    print(f"  Skipping task {task_id} (solution not found).")
                    continue
                train_solution_grid = all_training_solutions[task_id][0] # Get grid for test pair 0
                # --- END ADD ---
                if task_data is None:
                    print(f"  Skipping task {task_id} (data is None).")
                    continue

                # 2. Preprocess *outside* the tf_train_step function
                # --- MODIFY: Pass solution grid AND task_id ---
                # Extract grid first
                train_solution_grid_np = extract_true_output_grid(train_solution_grid, task_id)
 
                model_inputs_transformed, target_grid_aug, target_size_aug, transformation, _, _, _ = \
                    preprocess_task(task_data,
                                    task_id=task_id, # <-- Pass task_id
                                    mode='train',
                                    augment=True,
                                    test_output_grid_true=train_solution_grid_np) # Pass extracted grid
 
                # Need identity inputs AND the identity target grid+size for equiv loss
                model_inputs_identity, target_grid_identity, target_size_identity, _, _, _, _ = \
                    preprocess_task(task_data,
                                    task_id=task_id, # <-- Pass task_id
                                    mode='train',
                                    augment=False,
                                    test_output_grid_true=train_solution_grid_np) # Pass extracted grid
                # --- END MODIFY ---
                 # Check for preprocessing errors
                if model_inputs_transformed is None or target_grid_aug is None or \
                   target_size_aug is None or model_inputs_identity is None or \
                   target_grid_identity is None or target_size_identity is None: # Check identity targets too
                    print(f"  Skipping task {task_id} (preprocess error).")
                    continue

                # 3. Call the @tf.function decorated training step
                # Note: Equivariance loss inside tf_train_step needs access to the
                # *identity* target grid/size if it recalculates the identity output.
                # However, calculate_losses currently doesn't need identity target,
                # it just needs the identity *inputs* to re-run the model.
                final_loss_tensor,  losses_dict_tensor = tf_train_step(
                if (i + 1) % 300 == 0:
                    # Telemetry: compare predicted size vs true (identity pass)
                    _, h_logits_dbg, w_logits_dbg = model(model_inputs_identity, training=False)
                    ph = int((tf.argmax(h_logits_dbg, -1) + 1).numpy()[0])
                    pw = int((tf.argmax(w_logits_dbg, -1) + 1).numpy()[0])
                    th = int(target_size_identity.numpy()[0,0]); tw = int(target_size_identity.numpy()[0,1])
                    print(f"    size pred {ph}x{pw} vs true {th}x{tw}")

                    model, optimizer,
                    model_inputs_transformed, target_grid_aug, target_size_aug,
                    model_inputs_identity, tf.constant(transformation) # Pass transformation as tensor
                )

                # Check if step returned a valid dictionary (signalling success)
                if isinstance(losses_dict_tensor, dict) and losses_dict_tensor:
                     tasks_processed += 1
                     # Convert tensor values in dict to numpy for accumulation
                     losses_numpy = {k: v.numpy() for k, v in losses_dict_tensor.items()}
                     epoch_losses.update(losses_numpy) # Add values to counter

                     # Optional: Print progress
                     if (i + 1) % 50 == 0:
                         avg_total_loss = epoch_losses.get('total_loss_calc', 0) / tasks_processed if tasks_processed else 0
                         avg_ce_loss = epoch_losses.get('ce_loss', 0) / tasks_processed if tasks_processed else 0
                         step_time = time.time() - step_start
                         print(f"  Epoch {epoch+1}, Task {i+1}/{len(train_task_ids)}, "
                               f"Avg Total Loss: {avg_total_loss:.4f}, Avg CE Loss: {avg_ce_loss:.4f} "
                               f"(step time: {step_time:.2f}s)")

                elif isinstance(losses_dict_tensor, dict) and not losses_dict_tensor:
                     # Task was skipped within tf_train_step (e.g., no valid gradients)
                     print(f"  Skipping task {task_id} (internal step failure).")
                     pass

            except Exception as e:
                # Catch errors during data loading, preprocessing, or the tf_train_step call
                print(f"\n--- Training Error on Task: {task_id} ---")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}")
                print("Traceback:")
                traceback.print_exc()
                print("--- End Error Report ---")
                # continue # Decide whether to continue training

        # --- Epoch Summary ---
        epoch_end_time = time.time()
        print(f"\nEpoch {epoch+1} finished in {epoch_end_time - epoch_start_time:.2f} seconds.")
        if tasks_processed > 0:
            print("Average Training Losses for Epoch:")
            for loss_name, total_value in sorted(epoch_losses.items()):
                avg_loss = total_value / tasks_processed
                print(f"  {loss_name}: {avg_loss:.4f}")
        else:
            print("No tasks successfully processed in this epoch.")

        # --- Evaluation ---
        # Pass the pre-loaded evaluation dictionary
        
        current_eval_success_rate = evaluate_model(model, all_evaluation_tasks, all_evaluation_solutions, epoch_num=epoch+1)      
        # --- Save Best Model ---
        # (Saving logic remains the same)
        if current_eval_success_rate > best_eval_success_rate:
             print(f"New best evaluation success rate: {current_eval_success_rate:.4f}. Saving model weights...")
             best_eval_success_rate = current_eval_success_rate
             try:
                  model.save_weights('/kaggle/working/arc_research_best_model.weights.h5')
                  print("Model weights saved successfully.")
             except Exception as e:
                  print(f"ERROR saving model weights: {e}")
        else:
             print(f"Evaluation success rate ({current_eval_success_rate:.4f}) did not improve from best ({best_eval_success_rate:.4f}).")


    print("\nTraining complete.")
    print(f"Best evaluation task success rate achieved: {best_eval_success_rate:.4f}")

else: # if build_success was False
    print("Skipping training because model build failed.")

# --- Final Evaluation (Cell 13) ---
# Needs to pass the loaded dictionary `all_evaluation_tasks` instead of `eval_files`
# Example:
# final_success_rate = evaluate_model(model, all_evaluation_tasks, epoch_num='final')



# --- Optional: Load best weights and run final evaluation ---
best_model_path = '/kaggle/working/arc_research_best_model.weights.h5'
if build_success and os.path.exists(best_model_path):
    print(f"\nLoading best weights from {best_model_path} for final evaluation...")
    try:
        # Re-instantiate model or ensure current model structure matches saved weights
        # If structure hasn't changed, loading weights into the existing model is fine.
        model.load_weights(best_model_path)
        print("Weights loaded successfully.")
        # Run evaluation with epoch_num=-1 or similar to indicate final eval
        final_success_rate = evaluate_model(model, eval_files, epoch_num='final')
        print(f"Final evaluation success rate with best weights: {final_success_rate:.4f}")
    except Exception as e:
        print(f"Error loading weights or running final evaluation: {e}")
        traceback.print_exc()
elif not build_success:
    print("Skipping final evaluation because model build failed.")
else:
    print("Skipping final evaluation because best model weights file not found.")

