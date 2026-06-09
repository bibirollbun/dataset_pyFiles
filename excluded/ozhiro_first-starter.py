import os
import sys
import json
import glob

import torch
import numpy as np
import pandas as pd
from torchvision.transforms import Compose

sys.path.append('/kaggle/input/openfwi-pretrainedmodel/OpenFWI')

import network
import transforms as T
from dataset import FWIDataset


DEBUG = False # Set to False to process all files


#! wget --no-check-certificate 'https://zenodo.org/record/7293942/files/cva_l1.pth'


WEIGHTS_PATH = '/kaggle/input/gwi-pretrainedmodel/PretrainedModel/ffb_l2.pth'


MODEL_NAME = 'InversionNet' 

TEST_DATA_DIR = '/kaggle/input/waveform-inversion/test' 
OUTPUT_CSV = 'submission.csv'

DATASET_CONFIG = '/kaggle/input/openfwi-pretrainedmodel/OpenFWI/dataset_config.json'
DATASET_NAME = 'flatfault-b' # Adjust based on the dataset used for training the weights, needed for normalization params

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_TRANSFORM = 1 # k value for LogTransform, adjust if needed (from test.py --k)
BATCH_SIZE = 16 # Process one file (oid) at a time

# Sample spatial/temporal might be needed depending on the model architecture used for fva_l1.pth
SAMPLE_SPATIAL = 1.0 # Adjust if needed (from test.py --sample-spatial)
SAMPLE_TEMPORAL = 1  # Adjust if needed (from test.py --sample-temporal)
NORM_LAYER = 'bn' # Adjust if needed (from test.py --norm)
UP_MODE = None # Adjust if needed (from test.py --up-mode)


def load_dataset_config(config_path, dataset_name):
    """Loads normalization parameters from dataset_config.json."""
    try:
        with open(config_path) as f:
            ctx = json.load(f)[dataset_name]
        print(f"Loaded config for dataset: {dataset_name}")
        return ctx
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Dataset '{dataset_name}' not found in {config_path}.")
        sys.exit(1)

def get_transforms(ctx, k):
    """Gets the transformations for data and label based on test.py."""
    log_data_min = T.log_transform(ctx['data_min'], k=k)
    log_data_max = T.log_transform(ctx['data_max'], k=k)
    transform_data = Compose([
        T.LogTransform(k=k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    return transform_data


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"Loading weights from: {WEIGHTS_PATH}")
    print(f"Test data directory: {TEST_DATA_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")

    # Load dataset configuration for normalization parameters
    ctx = load_dataset_config(DATASET_CONFIG, DATASET_NAME)

    # Initialize Model
    if MODEL_NAME not in network.model_dict:
        print(f"Error: Unsupported model '{MODEL_NAME}'. Check network.py.")
        sys.exit(1)

    model = network.model_dict[MODEL_NAME](
        upsample_mode=UP_MODE,
        sample_spatial=SAMPLE_SPATIAL,
        sample_temporal=SAMPLE_TEMPORAL,
        norm=NORM_LAYER
    ).to(DEVICE)

    # Load Weights
    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        # Handle potential legacy keys or different saving structures
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # Apply legacy replacement if needed (adapt from test.py if necessary)
        # state_dict = network.replace_legacy(state_dict) # Uncomment/adapt if needed
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
        if 'epoch' in checkpoint and 'step' in checkpoint:
             print(f"Weights from Epoch {checkpoint['epoch']} / Step {checkpoint['step']}.")

    except FileNotFoundError:
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    model.eval()

    # Get data transform
    transform_data = get_transforms(ctx, K_TRANSFORM)

    # Find test files
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, '*.npy'))
    if not test_files:
        print(f"Error: No .npy files found in {TEST_DATA_DIR}")
        sys.exit(1)

    # --- Debugging: Process only the first file if DEBUG is True ---
    if DEBUG:
        print("*** DEBUG MODE: Processing only the first file ***")
        test_files = test_files[:1]
    # --- End Debugging ---

    print(f"Found {len(test_files)} test file(s) to process.")

    results = []
    with torch.no_grad():
        for i, file_path in enumerate(test_files):
            oid = os.path.splitext(os.path.basename(file_path))[0]
            print(f"Processing ({i+1}/{len(test_files)}): {oid}")

            try:
                # Load seismic data
                seismic_data = np.load(file_path)

                if seismic_data.ndim == 3:
                     # Add batch dimension
                    seismic_data = seismic_data[np.newaxis, :, :, :]
                    # If model expects single channel, select one source, e.g., seismic_data = seismic_data[:, 0:1, :, :]
                elif seismic_data.ndim != 4:
                     print(f"Warning: Unexpected data dimension {seismic_data.ndim} for {oid}. Skipping.")
                     continue

                # Convert to tensor and move to device
                data_tensor = torch.from_numpy(seismic_data).type(torch.FloatTensor)

                # Manual Transformation (Example - adapt based on actual T implementations)
                data_tensor = T.log_transform(data_tensor, k=K_TRANSFORM)
                log_data_min = T.log_transform(ctx['data_min'], k=K_TRANSFORM)
                log_data_max = T.log_transform(ctx['data_max'], k=K_TRANSFORM)
                data_tensor = T.minmax_normalize(data_tensor, log_data_min, log_data_max)


                data_tensor = data_tensor.to(DEVICE)


                # Perform inference
                pred_tensor = model(data_tensor)

                # Denormalize prediction
                # pred_tensor shape is likely (batch, 1, height, width)
                pred_np = T.tonumpy_denormalize(pred_tensor, ctx['label_min'], ctx['label_max'], exp=False)

                # pred_np shape should now be (batch, 1, height, width) as numpy array
                velocity_map = pred_np[0, 0] # Get the (height, width) map

                # Format for submission
                height, width = velocity_map.shape
                # Ensure width allows for indexing up to x_69 (i.e., width >= 70)
                if width < 70:
                    print(f"Warning: Predicted width {width} for {oid} is less than 70. Padding or check model output.")


                for y_pos in range(height):
                    row_data = {'oid_ypos': f"{oid}_y_{y_pos}"}
                    odd_indices = range(1, min(width, 70), 2) # Generate indices 1, 3, ..., 69 (or less if width is smaller)
                    for x_idx in odd_indices:
                        col_name = f"x_{x_idx}"
                        row_data[col_name] = velocity_map[y_pos, x_idx]

                    # Handle missing columns if width < 70 by filling with a default (e.g., last valid value or 0)
                    last_valid_x = odd_indices[-1] if odd_indices else -1 # Find the last index added
                    for x_idx_req in range(1, 70, 2): # Required indices 1, 3, ..., 69
                         col_name_req = f"x_{x_idx_req}"
                         if x_idx_req > last_valid_x:
                             # Fill missing required columns, e.g., with the value of the last valid odd column
                             fill_value = velocity_map[y_pos, last_valid_x] if last_valid_x >= 0 else 3000.0 # Or use a constant like 3000
                             row_data[col_name_req] = fill_value


                    results.append(row_data)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue # Skip to next file

    # Create DataFrame and save to CSV
    if not results:
        print("No results generated. Exiting.")
        sys.exit(1)

    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    cols = ['oid_ypos'] + [f'x_{i}' for i in range(1, 70, 2)]
    submission_df = submission_df[cols]

    submission_df.to_csv(OUTPUT_CSV, index=False, float_format='%.4f') # Format float precision if needed
    print(f"Submission file saved to {OUTPUT_CSV}")

    # --- Debugging: Print head of submission if DEBUG is True ---
    if DEBUG:
        print("\n--- Submission DataFrame Head (DEBUG) ---")
        print(submission_df.head(5))
        print("-----------------------------------------")
    # --- End Debugging ---



if __name__ == '__main__':
    main()




