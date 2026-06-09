import os
import sys
import json
import glob

import torch
import numpy as np
import pandas as pd
from torchvision.transforms import Compose
sys.path.append('/kaggle/input/openfwi/OpenFWI-main')
import network
import transforms as T
from dataset import FWIDataset


DEBUG = False
WEIGHTS_PATH = '/kaggle/input/fwi-pretrained-weight/PretrainedModel/'


MODEL_NAME = 'InversionNet'

TEST_DATA_DIR = '/kaggle/input/waveform-inversion/test'
OUTPUT_CSV = '/kaggle/working/submission.csv'
DATASET_CONFIG = '/kaggle/input/openfwi/OpenFWI-main/dataset_config.json'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_TRANSFORM = 1 # k value for LogTransform, adjust if needed (from test.py --k)

# Sample spatial/temporal might be needed depending on the model architecture used for fva_l1.pth
SAMPLE_SPATIAL = 1.0 # Adjust if needed (from test.py --sample-spatial)
SAMPLE_TEMPORAL = 1  # Adjust if needed (from test.py --sample-temporal)
NORM_LAYER = 'bn' # Adjust if needed (from test.py --norm)
UP_MODE = None # Adjust if needed (from test.py --up-mode)


def load_dataset_config(config_path, dataset_name):
    try:
        with open(config_path) as f:
            ctx = json.load(f)[dataset_name]
        print(f"Loaded config for dataset: {dataset_name}")
        return ctx

    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: {dataset_name} not found in {config_path}.")
        sys.exit(1)


LabelsMap = {
    0: "curvefault-a",
    1: "curvefault-b",
    2: "curvevel-a",
    3: "curvevel-b",
    4: "flatfault-a",
    5: "flatfault-b",
    6: "flatvel-a",
    7: "flatvel-b",
    8: "style-a",
    9: "style-b",
}

LabelToNum = {v: k for k, v in LabelsMap.items()}

WeightsMap = {
    "curvefault-a":"cfa_l1.pth",
    "curvefault-b":"cfb_l1.pth",
    "curvevel-a":"cva_l1.pth",
    "curvevel-b":"cvb_l1.pth",
    "flatfault-a":"ffa_l1.pth",
    "flatfault-b":"ffb_l1.pth",
    "flatvel-a":"fva_l1.pth",
    "flatvel-b":"fvb_l1.pth",
    "style-a":"sta_l1_new.pth",
    "style-b":"stb_l1.pth",
}


def main():
    print(f"Using device: {DEVICE}")
    print(f"Test data directory: {TEST_DATA_DIR}")
    print(f"Output CSV {OUTPUT_CSV}")
    print(f"Loading model: {MODEL_NAME}")

    # Initialize Model
    if MODEL_NAME not in network.model_dict:
        print(f"Error: Unsupported mode '{MODEL_NAME}'. Check network.py")
        sys.exit(1)

    model = network.model_dict[MODEL_NAME](
        upsample_mode=UP_MODE,
        sample_temporal=SAMPLE_TEMPORAL,
        norm=NORM_LAYER
    ).to(DEVICE)

    # Find test files
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, '*.npy'))
    if not test_files:
        print(f"Error: No .npy files found in {TEST_DATA_DIR}")
        sys.exit(1)

    result = [None]*len(test_files)

    # Read the list of styles of test data
    test_types = [int(line.strip()) for line in open("/kaggle/input/test-type-fwi/test_type.txt", 'r', encoding='utf-8')]
    
    print("Loop all styles")
    for style in LabelsMap.values():
        W_PATH = WEIGHTS_PATH + WeightsMap[style]
        print(f"Style {style}")
        print(f"Loading weights from {W_PATH}")

        ctx = load_dataset_config(DATASET_CONFIG, style)

        #Load Weights
        try:
            checkpoint = torch.load(W_PATH, map_location=DEVICE, weights_only=False)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")

            if 'epoch' in checkpoint and 'step' in checkpoint:
                print(f"Weights from Epoch {checkpoint['epoch']}/Step {checkpoint['step']}.")

        except FileNotFoundError:
            print(f"Error: Weights file not found at {W_PATH}")
            sys.exit(1)

        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)

        model.eval()

        # Inference
        with torch.no_grad():
            for i, (file_path, t_style) in enumerate(zip(test_files, test_types)):
                # If style isn't match next case.
                if LabelsMap[t_style] != style:
                    continue

                else:
                    oid = os.path.splitext(os.path.basename(file_path))[0]
                    
                    try:
                        seismic_data = np.load(file_path)
                        if seismic_data.ndim == 3:
                            seismic_data = seismic_data[np.newaxis, :, :, :]
                        elif seismic_data.ndim != 4:
                            print(f"Warning: Unexpected data dimension {seismic_data.ndim} for {oid}. Skipping")
                            continue

                        # Convert to tensor and move to device
                        data_tensor = torch.from_numpy(seismic_data).type(torch.FloatTensor)
                        data_tensor = T.log_transform(data_tensor, k=K_TRANSFORM)
                        log_data_min = T.log_transform(ctx['data_min'], k=K_TRANSFORM)
                        log_data_max = T.log_transform(ctx['data_max'], k=K_TRANSFORM)
                        data_tensor = T.minmax_normalize(data_tensor, log_data_min, log_data_max)

                        data_tensor = data_tensor.to(DEVICE)

                        # Prediction
                        pred_tensor = model(data_tensor)

                        # Denormalize prediction
                        pred_np = T.tonumpy_denormalize(pred_tensor, ctx['label_min'], ctx['label_max'], exp=False)
                        velo_map = pred_np[0,0]

                        # Format for submission
                        height, width = velo_map.shape
                        if width < 70:
                            print(f"Warning: Predicted width {width} for {oid} is less than 70. Padding or check model output.")

                        tmp_result = []
                        for y_pos in range(height):
                            # Processing each row
                            row_data = {'oid_ypos': f"{oid}_y_{y_pos}"}
                            odd_indices = range(1, min(width, 70), 2)

                            for x_idx in odd_indices:
                                col_name = f"x_{x_idx}"
                                row_data[col_name]=velo_map[y_pos, x_idx]

                            # Handle missing velue
                            last_valid_x = odd_indices[-1]
                            for x_idx_req in range(1, 70, 2):
                                col_name_req = f"x_{x_idx_req}"
                                if x_idx_req > last_valid_x:
                                    fill_value = velo_map[y_pos, last_valid_x] if last_valid_x >= 0 else 3000
                                    row_data[col_name_req] = fill_value


                            tmp_result.append(row_data)
                        result[i] = tmp_result
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue

    if not result:
        print("No results generated. Exiting.")
        sys.exit(1)


    # Make submission
    submission = []
    for res in result:
        for row in res:
            submission.append(row)

    submission_df = pd.DataFrame(submission)
            
    # Ensure correct column order
    cols = ['oid_ypos'] + [f'x_{i}' for i in range(1, 70, 2)]
    submission_df = submission_df[cols]
    submission_df.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')

    print(f"Submission file save to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()

