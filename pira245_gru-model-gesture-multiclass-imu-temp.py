import importlib.util
import sys
import os 
import polars as pl
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path


from tqdm.notebook import tqdm
import polars.selectors as cs


# import kagglehub
# from kagglehub import KaggleDatasetAdapter


# Load external module dynamically
def load_my_utils(filedir, module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    
    spec = importlib.util.spec_from_file_location(module_name, filedir)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


filedir = '/kaggle/usr/lib/cmi_sensor_data_utility_functions/cmi_sensor_data_utility_functions.py'
module_name = "my_utils"
my_utils = load_my_utils(filedir, module_name)


filedir = '/kaggle/input/cmi-gesture-classification-gru-models/pytorch/imu-temp-data/9/gru_model_module.py'
module_name = "gru_utils"
gru_utils = load_my_utils(filedir, module_name)


print(my_utils.notebook_folder) 
data_folders_dictionary = my_utils.data_folder(my_utils.notebook_folder)


def move_column_to_end(df, col_name):
    cols = [col for col in df.columns if col != col_name] + [col_name]
    return df[cols]


from collections import defaultdict
def add_dicts_with_padding(dict1, dict2):
    result = {}
    all_keys = set(dict1.keys()).union(dict2.keys())

    for key in all_keys:
        list1 = dict1.get(key, [])
        list2 = dict2.get(key, [])

        max_len = max(len(list1), len(list2))
        
        # Pad shorter list with zeros
        padded1 = list1 + [0] * (max_len - len(list1))
        padded2 = list2 + [0] * (max_len - len(list2))

        result[key] = [a + b for a, b in zip(padded1, padded2)]

    return result


def check_quaternion_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a quaternion DataFrame with columns ['rot_x', 'rot_y', 'rot_z', 'rot_w'].
    - Returns the input quaternion if all values are non-null and norm > 0.
    - Returns the default identity quaternion [0, 0, 0, 1] otherwise.

    Args:
        df (pd.DataFrame): Single-row DataFrame with quaternion components.

    Returns:
        pd.DataFrame: Validated quaternion as a single-row DataFrame.
    """

    # Define identity quaternion [0, 0, 0, 1]
    identity_q = pd.DataFrame({
        'rot_x': [0.0],
        'rot_y': [0.0],
        'rot_z': [0.0],
        'rot_w': [1.0]
    })

    # If any value is NaN, return identity quaternion
    if df.isnull().values.any():
        return identity_q

    # Convert DataFrame to numpy array to check norm
    q_values = df.to_numpy(dtype='float')
    norm = np.linalg.norm(q_values)

    # If norm is zero or negative (invalid quaternion), return identity
    if norm > 0:
        return df.copy()
    else:
        return identity_q   


# Valid quaternion
df_valid = pd.DataFrame({'rot_x': [0.0], 'rot_y': [0.0], 'rot_z': [0.0], 'rot_w': [1.0]})
print(check_quaternion_values(df_valid))

# Quaternion with NaN
df_nan = pd.DataFrame({'rot_x': [np.nan], 'rot_y': [0.0], 'rot_z': [0.0], 'rot_w': [1.0]})
print(check_quaternion_values(df_nan))

# Zero quaternion
df_zero = pd.DataFrame({'rot_x': [0.0], 'rot_y': [0.0], 'rot_z': [0.0], 'rot_w': [0.0]})
print(check_quaternion_values(df_zero))


# - Function to find key by unique value
def find_key_by_unique_value(my_dict, target_value):
    """
    :param my_dict:
    :param target_value:
    :return:
    """

    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None


def build_imu_subset(sequence_cat_subset: pd.DataFrame,
                     sequence_acc_subset: pd.DataFrame,
                     sequence_rot_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Processes IMU data by validating quaternion data and computing motion features:
    world-frame acceleration, motion-frame acceleration, velocity, and position.

    Parameters:
        sequence_cat_subset (pd.DataFrame): Metadata including timestamps (e.g., 'sequence_counter')
        sequence_acc_subset (pd.DataFrame): Accelerometer data (e.g., x, y, z)
        sequence_rot_subset (pd.DataFrame): Quaternion rotation data (e.g., rot_x, rot_y, rot_z, rot_w)

    Returns:
        pd.DataFrame: Combined DataFrame with all raw and computed features.
    """
    # Validate quaternion data row by row
    checked_rot_subset = sequence_rot_subset.copy()
    for i in tqdm(range(len(sequence_rot_subset)), desc="Checking quaternions"):
        rot_row = sequence_rot_subset.iloc[i]
        checked_rot_subset.iloc[i] = check_quaternion_values(rot_row)

    # Convert all components to NumPy arrays
    tdata = sequence_cat_subset['sequence_counter'].to_numpy(dtype='float')
    acc_data = sequence_acc_subset.to_numpy(dtype='float')
    q = checked_rot_subset.to_numpy(dtype='float')

    # Process IMU data using utility function (assumes pre-imported my_utils module)
    acc_world, acc_motion, velocity, position = my_utils.process_imu_data(
        acc_data=acc_data,
        quaternions=q,
        timestamps=tdata
    )

    # Convert computed outputs into DataFrames
    acc_world_df = pd.DataFrame(acc_world, columns=['acc_world_x', 'acc_world_y', 'acc_world_z'])
    acc_motion_df = pd.DataFrame(acc_motion, columns=['acc_motion_x', 'acc_motion_y', 'acc_motion_z'])
    velocity_df = pd.DataFrame(velocity, columns=['velocity_x', 'velocity_y', 'velocity_z'])
    position_df = pd.DataFrame(position, columns=['position_x', 'position_y', 'position_z'])

    # Combine everything into a single IMU dataset
    imu_merged_subset = pd.concat([
        sequence_cat_subset[['sequence_counter']].reset_index(drop=True),
        sequence_acc_subset.reset_index(drop=True),
        checked_rot_subset.reset_index(drop=True),
        acc_world_df,
        acc_motion_df,
        velocity_df,
        position_df
    ], axis=1)

    # orientation invariant df
    imu_merged_df = my_utils.compute_orientation_invariant_features(imu_merged_subset)

    return imu_merged_df


# Backup real my_utils
real_my_utils = my_utils
# Create a mock wrapper that delegates everything else to the original
class MockMyUtils:
    @staticmethod
    def process_imu_data(acc_data, quaternions, timestamps):
        # Mock behavior only for process_imu_data
        return acc_data, acc_data * 0.5, acc_data * 0.1, acc_data * 0.01
    
    def __getattr__(self, name):
        # Delegate all other attribute lookups to real my_utils
        return getattr(real_my_utils, name)

# Override my_utils for testing
my_utils = MockMyUtils()

# Sample data
sequence_cat_subset = pd.DataFrame({'sequence_counter': np.arange(5)})
sequence_acc_subset = pd.DataFrame({
    'acc_x': np.random.rand(5),
    'acc_y': np.random.rand(5),
    'acc_z': np.random.rand(5)
})
sequence_rot_subset = pd.DataFrame({
    'rot_x': [0, 0, 0, 0, np.nan],
    'rot_y': [0, 0, 0, 0, np.nan],
    'rot_z': [0, 0, 0, 0, np.nan],
    'rot_w': [1, 1, 1, 1, np.nan]
})

# Run your test
result = build_imu_subset(sequence_cat_subset, sequence_acc_subset, sequence_rot_subset)
print(result.head())

# Restore original after test if needed
my_utils = real_my_utils


def build_imu_temp_subset(sequence_temp_subset: pd.DataFrame,
                          imu_merged_subset: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """
    Merges temperature sensor readings with IMU data,
    handling missing values by substituting predefined defaults.

    Parameters:
        sequence_temp_subset (pd.DataFrame): Temperature sensor readings.
        imu_merged_subset (pd.DataFrame): Processed IMU data.

    Returns:
        Tuple:
            - all_values_null (bool): True if all rows have completely missing temperature data.
            - imu_temp_merged_subset (pd.DataFrame): Combined IMU and temperature DataFrame.
    """

    # Default substitution values for each thermometer sensor
    missing_temp_substitution_values = {
        'thm_1': 26.982324,
        'thm_2': 26.354338,
        'thm_3': 26.956276,
        'thm_4': 27.742224,
        'thm_5': 29.500000
    }

    # Define which columns to check for nulls
    temp_cols = list(missing_temp_substitution_values.keys())
    checked_temp_subset = sequence_temp_subset.copy()

    # Determine if all values are NaN across all temperature columns
    all_values_null = checked_temp_subset[temp_cols].isnull().all(axis=1).all()

    # Only fill missing values in rows with at least one NaN
    any_null_indices = checked_temp_subset[temp_cols].isnull().any(axis=1)
    for col, default_val in missing_temp_substitution_values.items():
        checked_temp_subset.loc[any_null_indices, col] = (
            checked_temp_subset.loc[any_null_indices, col].fillna(default_val)
        )

    # Concatenate IMU data with cleaned temperature data
    imu_temp_merged_subset = pd.concat([
        imu_merged_subset.reset_index(drop=True),
        checked_temp_subset.reset_index(drop=True)
    ], axis=1)

    return all_values_null, imu_temp_merged_subset


# --- Mock IMU Data ---
imu_merged_subset = pd.DataFrame({
    'acc_x': [0.1, 0.2],
    'acc_y': [0.3, 0.4],
    'rot_w': [1.0, 1.0]
})

# --- Temperature Data with Missing Values ---
sequence_temp_subset = pd.DataFrame({
    'thm_1': [np.nan, 27.0],
    'thm_2': [26.3, np.nan],
    'thm_3': [np.nan, 26.9],
    'thm_4': [np.nan, np.nan],
    'thm_5': [29.5, np.nan]
})

# Call the function
all_null, imu_temp_merged = build_imu_temp_subset(sequence_temp_subset, imu_merged_subset)

# Show result
print("All values null in temperature input:", all_null)
print("Merged DataFrame:")
print(imu_temp_merged)


#load imu only trained model: Gesture
import torch
import torch.nn as nn
# ----------------------------
# âœ… Load best training params
# ----------------------------
# Load the trained models from a pickle file located in the predictions folder
folder = '/kaggle/input/cmi-gesture-classification-gru-models/pytorch/imu-temp-data/9/'

imu_trained_models = my_utils.handle_pickle_dict(
        folder=folder,
        pickle_filename='imu_trained_models.pkl'
    )

imu_temp_trained_models = my_utils.handle_pickle_dict(
        folder=folder,
        pickle_filename='imu_temp_trained_models.pkl'
    )

# ----------------------------
# âœ… Load best training params for gesture
# ----------------------------
best_params = imu_trained_models['gesture']['best_params']
input_dim = imu_trained_models['gesture']['input_dim']
output_dim = imu_trained_models['gesture']['output_dim']

# ----------------------------
# âœ… Recreate model architecture for gesture
# ----------------------------
gesture_model = gru_utils.GRUMultiClassModel(
    # input_size=input_dim,
    input_size=input_dim,
    hidden_size=best_params['hidden_size'],
    output_size=output_dim,
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],
    bidirectional=best_params['bidirectional']
)

# ----------------------------
# ğŸ”� Load model weights safely (GPU/CPU compatible)
# ----------------------------
gesture_model_weights = Path(folder) / Path(imu_trained_models['gesture']['model_name']) 
gesture_adapter_state = Path(folder) / Path(imu_trained_models['gesture']['model_adapter']) 

assert gesture_model_weights.exists(), f"â�Œ Model file not found: {gesture_model_weights}"
assert gesture_adapter_state.exists(), f"â�Œ Model file not found: {gesture_adapter_state}"



# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ğŸ–¥ï¸� Loading model on: {device}")

# Load with correct map_location
gesture_model.load_state_dict(torch.load(gesture_model_weights, map_location=device))
adapter_state = torch.load(gesture_adapter_state, map_location=device)



# --- 2) Rebuild the adapter wrapper exactly as during adapter training ---
model_adapter = gru_utils.make_adapter_from_pretrained(
    pretrained_backbone=gesture_model,
    num_classes=output_dim,   # same out_features as the original head
    bottleneck_dim=32,
    bn_momentum=0.05,
    lora_rank=8,
    lora_alpha=8.0,
    p_drop=0.3
)
# --- 3) Load the adapter weights into the wrapper ---
model_adapter.load_state_dict(adapter_state)  # loads SafeBN, Bottleneck, and LoRA weights



# ----------------------------
# âœ… Load best training params for gesture
# ----------------------------
best_params = imu_temp_trained_models['gesture']['best_params']
input_dim = imu_temp_trained_models['gesture']['input_dim']
output_dim = imu_temp_trained_models['gesture']['output_dim']

# ----------------------------
# âœ… Recreate model architecture for gesture
# ----------------------------
gesture_model_temp = gru_utils.GRUMultiClassModel(
    input_size=input_dim,
    hidden_size=best_params['hidden_size'],
    output_size=output_dim,
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],
    bidirectional=best_params['bidirectional'])

# + TEMP 
gesture_temp_model_weights = Path(folder) / Path(imu_temp_trained_models['gesture']['model_name']) 
gesture_temp_adapter_state = Path(folder) / Path(imu_temp_trained_models['gesture']['model_adapter']) 

assert gesture_temp_model_weights.exists(), f"â�Œ Model file not found: {gesture_temp_model_weights}"
assert gesture_temp_adapter_state.exists(), f"â�Œ Model file not found: {gesture_temp_adapter_state}"

# + TEMP 
gesture_model_temp.load_state_dict(torch.load(gesture_temp_model_weights, map_location=device))
adapter_temp_state = torch.load(gesture_temp_adapter_state, map_location=device)

# + TEMP 
# --- 2) Rebuild the adapter wrapper exactly as during adapter training ---
model_temp_adapter = gru_utils.make_adapter_from_pretrained(
    pretrained_backbone=gesture_model_temp,
    num_classes=output_dim,   # same out_features as the original head
    bottleneck_dim=32,
    bn_momentum=0.05,
    lora_rank=8,
    lora_alpha=8.0,
    p_drop=0.3
)
# --- 3) Load the adapter weights into the wrapper ---
model_temp_adapter.load_state_dict(adapter_temp_state)  # loads SafeBN, Bottleneck, and LoRA weights


@torch.no_grad()
def gru_model_prediction(imu_merged_subset, imu_trained_models, model, key):
    
    # Scale features with training scaler
    scaler = imu_trained_models[key]['scaler']
    scaled_x_test = scaler.transform(imu_merged_subset.to_numpy())
    # Sequence lengths per sequence
    sequence_length = imu_merged_subset["sequence_counter"].max() + 1
    sequence_lengths = np.array([sequence_length]) 
    # ----------------------------
    # ğŸ“¦ Convert to tensors
    # ----------------------------
    x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test, sequence_lengths) 
    lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
    # ----------------------------
    # ğŸ”® Make predictions
    # ----------------------------
    # ----- Gesture classification -----
    # Set model to eval mode and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():

        logits = model(x_tensor, lengths_tensor, return_logits=True) # shape (B, C)
        probs = torch.softmax(logits, dim=1)
        probs_list = probs.squeeze().tolist()
        # shape (B,)
        preds = torch.argmax(probs, dim=1)
        

    if key == 'gesture':
        # the gesture model was trained on zero based gesture indexes [0, 1, 2, ...]
        return preds + 1, probs_list

    else:
        return preds, probs_list


@torch.no_grad()
def adapter_prediction(imu_merged_subset, imu_trained_models, model_adapter: nn.Module, key: str):
    """
    Drop-in replacement for your gru_model_prediction() that calls the adapter model.
    It reuses your scaler and reshaper exactly the same way.
    """
    # 1) Scale with training scaler
    scaler = imu_trained_models[key]['scaler']
    scaled_x_test = scaler.transform(imu_merged_subset.to_numpy())

    # 2) Sequence lengths
    sequence_length = imu_merged_subset["sequence_counter"].max() + 1
    sequence_lengths = np.array([sequence_length])

    # 3) Convert to tensors
    x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test, sequence_lengths)
    lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)

    # 4) Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_adapter.to(device).eval()

    logits = model_adapter(x_tensor, lengths_tensor, return_logits=True)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    probs_list = probs.squeeze().tolist()
    return (preds + 1, probs_list) if key == 'gesture' else (preds, probs_list)


# calibration_100_670 = {
#     'imu': {},
#     'imu_temp': {}
# }


# imu_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"pira245/cmi-dataset","full_rot_imu_merged_v3_df.csv",)
# imu_df = imu_df.drop(['row_id.1' , 'sequence_counter.1'], axis=1)


# gesture_columns = ['gesture']
# n_splits = 5 # 80% train / 20% val
# split_index  = 0
# df = imu_df.drop(['row_id'], axis=1)
# imu_balanced_datasets = my_utils.prepare_balanced_gesture_datasets(df, gesture_columns,n_splits, split_index)


# gesture_val = imu_balanced_datasets['gesture']['train']


# # main_gesture_df = gesture_val.copy()
# main_gesture_df = imu_df.drop(['row_id'], axis=1).copy()


# # ğŸ”§ Set the threshold for what you consider a "long" sequence
# sequence_length_threshold = 100  # Default value (can be changed as needed)

# # Step 1: Compute sequence-level statistics for 'sequence_counter'
# seq_stats = main_gesture_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])

# # Step 2: Identify sequence_ids where length exceeds the threshold
# long_sequence_ids = seq_stats[seq_stats["count"] > sequence_length_threshold].index

# # Step 3: Filter the DataFrame to rows from long sequences
# long_sequences_df = main_gesture_df[main_gesture_df["sequence_id"].isin(long_sequence_ids)]

# # Step 4: Count gestures associated with long sequences (based on first label per sequence)
# long_gesture_counts = (
#     long_sequences_df
#     .groupby("sequence_id")["gesture"]
#     .first()
#     .value_counts()
#     .reset_index(name="count")
#     .rename(columns={"index": "gesture"})
# )

# # Step 5: Print the result
# print(f"ğŸ�¯ Gestures with sequences longer than {sequence_length_threshold} timesteps:")
# print(long_gesture_counts)

# # Step 6: Store the long sequence_ids for future removal
# long_sequence_ids_list = list(long_sequence_ids)
# print(f"\nğŸ§¹ Sequence IDs to consider removing ({len(long_sequence_ids_list)} total):")
# print(long_sequence_ids_list[:10], "...")  # Show sample

# # Step 7: Optionally remove long sequences from the original dataframe
# main_gesture_filtered_df = main_gesture_df[~main_gesture_df["sequence_id"].isin(long_sequence_ids)]
# special_case_df = main_gesture_df[main_gesture_df["sequence_id"].isin(long_sequence_ids)]

# # Step 8: Confirm shape of the filtered dataset
# print(f"\nâœ… New DataFrame shape after removal: {main_gesture_filtered_df.shape}")


# # Sequence statistics (min, max, length of each sequence)
# seq_stats = main_gesture_filtered_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])
# num_sequences = main_gesture_filtered_df["sequence_id"].nunique()
# num_sequences, seq_stats.describe()


# # Sequence statistics (min, max, length of each sequence)
# seq_stats = special_case_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])
# num_sequences = special_case_df["sequence_id"].nunique()
# num_sequences, seq_stats.describe()


# gesture_val = main_gesture_filtered_df.copy()


# from sklearn.metrics import classification_report, f1_score, confusion_matrix

# # ----------------------------
# # ğŸ“Š Prepare features
# # ----------------------------
# x_test_gesture = gesture_val.drop([
#     'sequence_type', 'sequence_id', 'subject', 'orientation',
#     'behavior', 'phase', 'gesture', 'bfrb_gesture', 'nbfrb_gesture'
# ], axis=1).to_numpy()

# # One label per sequence (likely 1-based: 1..18)
# gesture_seq_labels = gesture_val.groupby("sequence_id")["gesture"].first().values
# print(f"ğŸ§© # Test Sequences: {len(gesture_seq_labels)}")
# zero_based_labels = np.array(gesture_seq_labels) - 1


# # Scale with training scaler
# scaler = imu_trained_models['gesture']['scaler']
# scaled_x_test_gesture = scaler.transform(x_test_gesture)

# # Sequence lengths per sequence
# sequence_lengths_df = gesture_val.groupby("sequence_id")["sequence_counter"].max() + 1
# sequence_lengths_gesture = sequence_lengths_df.to_numpy()

# # ----------------------------
# # ğŸ“¦ Convert to tensors
# # ----------------------------
# x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test_gesture, sequence_lengths_gesture)  # (B, T, F)
# lengths_tensor = torch.tensor(sequence_lengths_gesture, dtype=torch.long)
# y_val_zero_based = torch.tensor(zero_based_labels, dtype=torch.long)

# # Ensure same device for inputs and models
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x_tensor = x_tensor.to(device)
# lengths_tensor = lengths_tensor.to(device)
# y_val_zero_based = y_val_zero_based.to(device)


# imu_base_T, imu_adpt_T, imu_p_base_val, imu_p_adpt_val = gru_utils.calibrate_two_models(
#                         gesture_model,
#                          model_adapter,
#                          x_tensor,          # (N, T, F) CPU
#                          lengths_tensor,        # (N,) CPU
#                          y_val_zero_based,   # (N,) 0..C-1
#                          device,
#                          batch_size = 128,
#                          amp = True)


# calibration_100_670['imu'].setdefault(100, {'model': [imu_base_T, imu_p_base_val], 'adapter': [imu_adpt_T, imu_p_adpt_val]})


# imu_w_base_history =  gru_utils.search_soft_voting_weight(
#     calibration_100_670['imu'][100]['model'][1],               # np.ndarray or torch.Tensor, shape (N, C)
#     calibration_100_670['imu'][100]['adapter'][1],             # same shape (N, C)
#     y_val_zero_based.cpu(),         # array-like labels, shape (N,), 0..C-1
#     weight_grid=None,          # defaults np.linspace(0,1,21)
#     max_epochs=None,           # defaults len(weight_grid)
#     verbose=True)


# imu_w_base_100 = imu_w_base_history['best_w']
# calibration_100_670['imu'][100].setdefault('best_ens_weight', [imu_w_base_100])
# print(f"Best weight: {imu_w_base_100}")


# gesture_val = special_case_df.copy()


# # from sklearn.metrics import classification_report, f1_score, confusion_matrix

# # ----------------------------
# # ğŸ“Š Prepare features
# # ----------------------------
# x_test_gesture = gesture_val.drop([
#     'sequence_type', 'sequence_id', 'subject', 'orientation',
#     'behavior', 'phase', 'gesture', 'bfrb_gesture', 'nbfrb_gesture'
# ], axis=1).to_numpy()

# # One label per sequence (likely 1-based: 1..18)
# gesture_seq_labels = gesture_val.groupby("sequence_id")["gesture"].first().values
# print(f"ğŸ§© # Test Sequences: {len(gesture_seq_labels)}")
# zero_based_labels = np.array(gesture_seq_labels) - 1


# # Scale with training scaler
# scaler = imu_trained_models['gesture']['scaler']
# scaled_x_test_gesture = scaler.transform(x_test_gesture)

# # Sequence lengths per sequence
# sequence_lengths_df = gesture_val.groupby("sequence_id")["sequence_counter"].max() + 1
# sequence_lengths_gesture = sequence_lengths_df.to_numpy()

# # ----------------------------
# # ğŸ“¦ Convert to tensors
# # ----------------------------
# x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test_gesture, sequence_lengths_gesture)  # (B, T, F)
# lengths_tensor = torch.tensor(sequence_lengths_gesture, dtype=torch.long)
# y_val_zero_based = torch.tensor(zero_based_labels, dtype=torch.long)

# # Ensure same device for inputs and models
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x_tensor = x_tensor.to(device)
# lengths_tensor = lengths_tensor.to(device)
# y_val_zero_based = y_val_zero_based.to(device)


# imu_base_T, imu_adpt_T, imu_p_base_val, imu_p_adpt_val = gru_utils.calibrate_two_models(
#                         gesture_model,
#                          model_adapter,
#                          x_tensor,          # (N, T, F) CPU
#                          lengths_tensor,        # (N,) CPU
#                          y_val_zero_based,   # (N,) 0..C-1
#                          device,
#                          batch_size = 128,
#                          amp = True)


# calibration_100_670['imu'].setdefault(670, {'model': [imu_base_T, imu_p_base_val], 'adapter': [imu_adpt_T, imu_p_adpt_val]})


# imu_w_base_history =  gru_utils.search_soft_voting_weight(
#     calibration_100_670['imu'][670]['model'][1],               # np.ndarray or torch.Tensor, shape (N, C)
#     calibration_100_670['imu'][670]['adapter'][1],             # same shape (N, C)
#     y_val_zero_based.cpu(),         # array-like labels, shape (N,), 0..C-1
#     weight_grid=None,          # defaults np.linspace(0,1,21)
#     max_epochs=None,           # defaults len(weight_grid)
#     verbose=True)


# imu_w_base_670 = imu_w_base_history['best_w']
# calibration_100_670['imu'][670].setdefault('best_ens_weight', [imu_w_base_670])
# print(f"Best weight: {imu_w_base_670}")


# imu_temp_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"pira245/cmi-dataset","full_rot_imu_temp_merged_v3_df.csv",)
# imu_temp_df = imu_temp_df.drop(['row_id.1' , 'row_id.2', 'sequence_counter.1'], axis=1)


# gesture_columns = ['gesture']
# n_splits = 5 # 80% train / 20% val
# split_index  = 0
# df = imu_temp_df.drop(['row_id'], axis=1)
# imu_balanced_datasets = my_utils.prepare_balanced_gesture_datasets(df, gesture_columns,n_splits, split_index)


# gesture_val = imu_balanced_datasets['gesture']['train']


# # main_gesture_df = gesture_val.copy()
# main_gesture_df = imu_temp_df.drop(['row_id'], axis=1).copy()


# # ğŸ”§ Set the threshold for what you consider a "long" sequence
# sequence_length_threshold = 100  # Default value (can be changed as needed)

# # Step 1: Compute sequence-level statistics for 'sequence_counter'
# seq_stats = main_gesture_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])

# # Step 2: Identify sequence_ids where length exceeds the threshold
# long_sequence_ids = seq_stats[seq_stats["count"] > sequence_length_threshold].index

# # Step 3: Filter the DataFrame to rows from long sequences
# long_sequences_df = main_gesture_df[main_gesture_df["sequence_id"].isin(long_sequence_ids)]

# # Step 4: Count gestures associated with long sequences (based on first label per sequence)
# long_gesture_counts = (
#     long_sequences_df
#     .groupby("sequence_id")["gesture"]
#     .first()
#     .value_counts()
#     .reset_index(name="count")
#     .rename(columns={"index": "gesture"})
# )

# # Step 5: Print the result
# print(f"ğŸ�¯ Gestures with sequences longer than {sequence_length_threshold} timesteps:")
# print(long_gesture_counts)

# # Step 6: Store the long sequence_ids for future removal
# long_sequence_ids_list = list(long_sequence_ids)
# print(f"\nğŸ§¹ Sequence IDs to consider removing ({len(long_sequence_ids_list)} total):")
# print(long_sequence_ids_list[:10], "...")  # Show sample

# # Step 7: Optionally remove long sequences from the original dataframe
# main_gesture_filtered_df = main_gesture_df[~main_gesture_df["sequence_id"].isin(long_sequence_ids)]
# special_case_df = main_gesture_df[main_gesture_df["sequence_id"].isin(long_sequence_ids)]

# # Step 8: Confirm shape of the filtered dataset
# print(f"\nâœ… New DataFrame shape after removal: {main_gesture_filtered_df.shape}")


# # Sequence statistics (min, max, length of each sequence)
# seq_stats = main_gesture_filtered_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])
# num_sequences = main_gesture_filtered_df["sequence_id"].nunique()
# num_sequences, seq_stats.describe()


# # Sequence statistics (min, max, length of each sequence)
# seq_stats = special_case_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])
# num_sequences = special_case_df["sequence_id"].nunique()
# num_sequences, seq_stats.describe()


# gesture_val = main_gesture_filtered_df.copy()


# # from sklearn.metrics import classification_report, f1_score, confusion_matrix

# # ----------------------------
# # ğŸ“Š Prepare features
# # ----------------------------
# x_test_gesture = gesture_val.drop([
#     'sequence_type', 'sequence_id', 'subject', 'orientation',
#     'behavior', 'phase', 'gesture', 'bfrb_gesture', 'nbfrb_gesture'
# ], axis=1).to_numpy()

# # One label per sequence (likely 1-based: 1..18)
# gesture_seq_labels = gesture_val.groupby("sequence_id")["gesture"].first().values
# print(f"ğŸ§© # Test Sequences: {len(gesture_seq_labels)}")
# zero_based_labels = np.array(gesture_seq_labels) - 1


# # Scale with training scaler
# scaler = imu_temp_trained_models['gesture']['scaler']
# scaled_x_test_gesture = scaler.transform(x_test_gesture)

# # Sequence lengths per sequence
# sequence_lengths_df = gesture_val.groupby("sequence_id")["sequence_counter"].max() + 1
# sequence_lengths_gesture = sequence_lengths_df.to_numpy()

# # ----------------------------
# # ğŸ“¦ Convert to tensors
# # ----------------------------
# x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test_gesture, sequence_lengths_gesture)  # (B, T, F)
# lengths_tensor = torch.tensor(sequence_lengths_gesture, dtype=torch.long)
# y_val_zero_based = torch.tensor(zero_based_labels, dtype=torch.long)

# # Ensure same device for inputs and models
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x_tensor = x_tensor.to(device)
# lengths_tensor = lengths_tensor.to(device)
# y_val_zero_based = y_val_zero_based.to(device)


# imu_temp_base_T, imu_temp_adpt_T, imu_temp_p_base_val, imu_temp_p_adpt_val = gru_utils.calibrate_two_models(
#                         gesture_model_temp,
#                          model_temp_adapter,
#                          x_tensor,          # (N, T, F) CPU
#                          lengths_tensor,        # (N,) CPU
#                          y_val_zero_based,   # (N,) 0..C-1
#                          device,
#                          batch_size = 128,
#                          amp = True)


# calibration_100_670['imu_temp'].setdefault(100, {'model': [imu_temp_base_T, imu_temp_p_base_val], 'adapter': [imu_temp_adpt_T, imu_temp_p_adpt_val]})


# imu_temp_w_base_history =  gru_utils.search_soft_voting_weight(
#     calibration_100_670['imu_temp'][100]['model'][1],               # np.ndarray or torch.Tensor, shape (N, C)
#     calibration_100_670['imu_temp'][100]['adapter'][1],             # same shape (N, C)
#     y_val_zero_based.cpu(),         # array-like labels, shape (N,), 0..C-1
#     weight_grid=None,          # defaults np.linspace(0,1,21)
#     max_epochs=None,           # defaults len(weight_grid)
#     verbose=True)


# imu_temp_w_base_100 = imu_temp_w_base_history['best_w']
# calibration_100_670['imu_temp'][100].setdefault('best_ens_weight', [imu_temp_w_base_100])
# print(f"Best weight: {imu_temp_w_base_100}")


# gesture_val = special_case_df.copy()


# # from sklearn.metrics import classification_report, f1_score, confusion_matrix

# # ----------------------------
# # ğŸ“Š Prepare features
# # ----------------------------
# x_test_gesture = gesture_val.drop([
#     'sequence_type', 'sequence_id', 'subject', 'orientation',
#     'behavior', 'phase', 'gesture', 'bfrb_gesture', 'nbfrb_gesture'
# ], axis=1).to_numpy()

# # One label per sequence (likely 1-based: 1..18)
# gesture_seq_labels = gesture_val.groupby("sequence_id")["gesture"].first().values
# print(f"ğŸ§© # Test Sequences: {len(gesture_seq_labels)}")
# zero_based_labels = np.array(gesture_seq_labels) - 1


# # Scale with training scaler
# scaler = imu_temp_trained_models['gesture']['scaler']
# scaled_x_test_gesture = scaler.transform(x_test_gesture)

# # Sequence lengths per sequence
# sequence_lengths_df = gesture_val.groupby("sequence_id")["sequence_counter"].max() + 1
# sequence_lengths_gesture = sequence_lengths_df.to_numpy()

# # ----------------------------
# # ğŸ“¦ Convert to tensors
# # ----------------------------
# x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test_gesture, sequence_lengths_gesture)  # (B, T, F)
# lengths_tensor = torch.tensor(sequence_lengths_gesture, dtype=torch.long)
# y_val_zero_based = torch.tensor(zero_based_labels, dtype=torch.long)

# # Ensure same device for inputs and models
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x_tensor = x_tensor.to(device)
# lengths_tensor = lengths_tensor.to(device)
# y_val_zero_based = y_val_zero_based.to(device)


# imu_temp_base_T, imu_temp_adpt_T, imu_temp_p_base_val, imu_temp_p_adpt_val = gru_utils.calibrate_two_models(
#                         gesture_model_temp,
#                          model_temp_adapter,
#                          x_tensor,          # (N, T, F) CPU
#                          lengths_tensor,        # (N,) CPU
#                          y_val_zero_based,   # (N,) 0..C-1
#                          device,
#                          batch_size = 128,
#                          amp = True)


# calibration_100_670['imu_temp'].setdefault(670, {'model': [imu_temp_base_T, imu_temp_p_base_val], 'adapter': [imu_temp_adpt_T, imu_temp_p_adpt_val]})


# imu_temp_w_base_history =  gru_utils.search_soft_voting_weight(
#     calibration_100_670['imu_temp'][670]['model'][1],               # np.ndarray or torch.Tensor, shape (N, C)
#     calibration_100_670['imu_temp'][670]['adapter'][1],             # same shape (N, C)
#     y_val_zero_based.cpu(),         # array-like labels, shape (N,), 0..C-1
#     weight_grid=None,          # defaults np.linspace(0,1,21)
#     max_epochs=None,           # defaults len(weight_grid)
#     verbose=True)


# imu_temp_w_base_670 = imu_temp_w_base_history['best_w']
# calibration_100_670['imu_temp'][670].setdefault('best_ens_weight', [imu_temp_w_base_670])
# print(f"Best weight: {imu_temp_w_base_670}")


# # save calibration dictionary :
# folder = data_folders_dictionary['predictions_data']
# my_utils.handle_pickle_dict(folder=folder, pickle_filename ='calibration_100_670.pkl', data_dict=calibration_100_670)


# load calibration dictionary :
folder = '/kaggle/input/cmi-gesture-classification-gru-models/pytorch/imu-temp-data/9/'
loaded_calibration_100_670 = my_utils.handle_pickle_dict(folder=folder, pickle_filename ='calibration_100_670_v2.pkl')


def imu_only_prediction(
    imu_merged_subset: pd.DataFrame,
    task_key: str = 'gesture'
):
    """
    Performs prediction for a specified gesture-related task (e.g., 'gesture', 'bfrb_gesture')
    using IMU-only features and a pre-trained GRU model.

    Parameters:
        imu_merged_subset (pd.DataFrame): The input DataFrame containing one full sequence.
        task_key (str): The key indicating the task or output label, e.g., 'gesture', 'bfrb_gesture'.

    Returns:
        tuple: 
            - predictions (dict): {task_key: predicted label (1-based)}
            - probs_list (list): List of predicted class probabilities.
    """

    # ----------------------------
    # ğŸ”§ Setup
    # ----------------------------

    # Global models and configs (loaded externally)
    global imu_trained_models
    global model_adapter
    global gesture_model
    global loaded_calibration_100_670

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to the correct device
    gesture_model.to(device)
    model_adapter.to(device)

    # Initialize the predictions dictionary for this task
    predictions = {task_key: []}

    # Make a copy of the input to avoid modifying it in-place
    new_test_df = imu_merged_subset.copy()

    # ----------------------------
    # 1ï¸�âƒ£ Preprocessing
    # ----------------------------

    # Apply the saved scaler from training to standardize input features
    scaler = imu_trained_models[task_key]['scaler']
    scaled_x_test = scaler.transform(new_test_df.to_numpy())

    # Compute the sequence length (assumes a single sequence per input)
    sequence_length = new_test_df["sequence_counter"].max() + 1
    sequence_lengths = np.array([sequence_length])

    # Convert scaled data to padded GRU-compatible tensor
    x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test, sequence_lengths)
    lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)

    # Move tensors to the same device as the model
    x_tensor = x_tensor.to(device)
    lengths_tensor = lengths_tensor.to(device)

    # ----------------------------
    # 2ï¸�âƒ£ Model Selection: Based on sequence length
    # ----------------------------
    if sequence_length <= 100:
        model_key = 100
    else:
        model_key = 670

    # Extract appropriate model, adapter, and ensemble weight
    best_model = loaded_calibration_100_670['imu'][model_key]['model'][0]
    best_adapter = loaded_calibration_100_670['imu'][model_key]['adapter'][0]
    best_weight = loaded_calibration_100_670['imu'][model_key]['best_ens_weight'][0]

    # ----------------------------
    # 3ï¸�âƒ£ Make Predictions (with soft voting)
    # ----------------------------

    predicted_label, gesture_probs = gru_utils.soft_vote_predict(
        gesture_model,
        model_adapter,
        best_model,
        best_adapter,
        x_tensor,
        lengths_tensor,
        device,
        best_weight,
        batch_size=128,
        amp=True
    )

    # Convert to 1-based index as required by some pipelines
    predictions[task_key] = predicted_label + 1

    # Flatten probabilities to a Python list
    probs_list = gesture_probs.squeeze().tolist()

    return predictions, probs_list


def imu_temp_prediction(
    imu_temp_merged_subset: pd.DataFrame,
    task_key: str = 'gesture'
):
    """
    Performs prediction for a specified gesture-related task (e.g., 'gesture', 'bfrb_gesture')
    using IMU+Temperature features and a pre-trained GRU model.

    Parameters:
        imu_temp_merged_subset (pd.DataFrame): The input DataFrame containing one full sequence.
        task_key (str): The key indicating the task or output label, e.g., 'gesture', 'bfrb_gesture'.

    Returns:
        tuple: 
            - predictions (dict): {task_key: predicted label (1-based)}
            - probs_list (list): List of predicted class probabilities.
    """

    # ----------------------------
    # ğŸ”§ Setup
    # ----------------------------

    # Global models and configs (loaded externally)
    global imu_temp_trained_models
    global model_temp_adapter
    global gesture_model_temp
    global loaded_calibration_100_670

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to the correct device
    gesture_model_temp.to(device)
    model_temp_adapter.to(device)

    # Initialize the predictions dictionary for this task
    predictions = {task_key: []}

    # Make a copy of the input to avoid modifying it in-place
    new_test_df = imu_temp_merged_subset.copy()

    # ----------------------------
    # 1ï¸�âƒ£ Preprocessing
    # ----------------------------

    # Apply the saved scaler from training to standardize input features
    scaler = imu_temp_trained_models[task_key]['scaler']
    scaled_x_test = scaler.transform(new_test_df.to_numpy())

    # Compute the sequence length (assumes a single sequence per input)
    sequence_length = new_test_df["sequence_counter"].max() + 1
    sequence_lengths = np.array([sequence_length])

    # Convert scaled data to padded GRU-compatible tensor
    x_tensor = gru_utils.reshape_to_padded_tensor(scaled_x_test, sequence_lengths)
    lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)

    # Move tensors to the same device as the model
    x_tensor = x_tensor.to(device)
    lengths_tensor = lengths_tensor.to(device)

    # ----------------------------
    # 2ï¸�âƒ£ Model Selection: Based on sequence length
    # ----------------------------
    if sequence_length <= 100:
        model_key = 100
    else:
        model_key = 670

    # Extract appropriate model, adapter, and ensemble weight
    best_model = loaded_calibration_100_670['imu_temp'][model_key]['model'][0]
    best_adapter = loaded_calibration_100_670['imu_temp'][model_key]['adapter'][0]
    best_weight = loaded_calibration_100_670['imu_temp'][model_key]['best_ens_weight'][0]

    # ----------------------------
    # 3ï¸�âƒ£ Make Predictions (with soft voting)
    # ----------------------------

    predicted_label, gesture_probs = gru_utils.soft_vote_predict(
        gesture_model_temp,
        model_temp_adapter,
        best_model,
        best_adapter,
        x_tensor,
        lengths_tensor,
        device,
        best_weight,
        batch_size=128,
        amp=True
    )

    # Convert to 1-based index as required by some pipelines
    predictions[task_key] = predicted_label + 1

    # Flatten probabilities to a Python list
    probs_list = gesture_probs.squeeze().tolist()

    return predictions, probs_list


# load gesture types dictionaries:
folder = '/kaggle/usr/lib/cmi_detect_gesture_with_a_wrist_worm_device/competition-data/process-data'
loaded_gesture_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='gesture_type_dictionary.pkl')


def map_probs_to_dataframe(gesture_type_dict: dict, probs: list) -> pd.DataFrame:
    """
    Maps a list of probabilities to the corresponding gesture dictionary.

    Parameters:
        gesture_type_dict (dict): Dictionary containing gesture mapping.
                                  e.g., loaded_gesture_type_dictionary['gesture_type']
        probs (list): List of probabilities (same length and order as the values in the dictionary).

    Returns:
        pd.DataFrame: Sorted DataFrame with columns ['key', 'value', 'probability'].
    """

    # Sanity check
    
    if len(gesture_type_dict) != len(probs):
        raise ValueError("Length of probabilities list does not match dictionary length.")

    # Convert dict to list of tuples
    keys = list(gesture_type_dict.keys())
    values = list(gesture_type_dict.values())

    # Create dataframe
    df = pd.DataFrame({
        'key': keys,
        'value': values,
        'probability': probs
    })

    # Sort by probability (ascending)
    #df = df.sort_values(by='probability', ascending=True).reset_index(drop=True)

    return df


def merge_probability_dataframes(df_base: pd.DataFrame, df_add: pd.DataFrame) -> pd.DataFrame:
    """
    Adds probability values from df_add to df_base based on matching 'key'.

    Parameters:
        df_base (pd.DataFrame): The base dataframe (always 18 rows) with columns ['key', 'value', 'probability'].
        df_add (pd.DataFrame): The variable-length dataframe with same columns.

    Returns:
        pd.DataFrame: Updated df_base with summed probabilities.
    """

    # Merge on 'key' to align rows
    merged_df = df_base.merge(df_add[['key', 'probability']], on='key', how='left', suffixes=('', '_add'))

    # Fill missing probabilities with 0 (in case df_add has fewer keys)
    merged_df['probability_add'] = merged_df['probability_add'].fillna(0)

    # Sum original and added probabilities
    merged_df['probability'] = merged_df['probability'] + merged_df['probability_add']

    # Drop the temporary column
    merged_df = merged_df.drop(columns=['probability_add'])

    return merged_df


def gru_eval_predictions(gesture_probs):
    """
    Evaluates GRU output probabilities and generates mapped probability dataframes.

    Parameters:
        gesture_probs (list): List of 18 probabilities for the main gesture classifier.

    Returns:
        - If only `gesture_probs` is provided:
            gesture_probs_df
    """

    # Use global gesture type dictionaries
    global loaded_gesture_type_dictionary

    # === Case 1: Only gesture_probs provided ===
    
    gesture_probs_df = map_probs_to_dataframe(
            loaded_gesture_type_dictionary['gesture_type'],
            gesture_probs
        )
    
    return gesture_probs_df


def sequence_sampling_300(df_seq: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a long sequence to exactly 300 observations using a structured 3-part sampling strategy:

    Args:
        df_seq (pd.DataFrame): A mini dataframe for a single sequence_id

    Returns:
        pd.DataFrame: Sampled dataframe with exactly 300 rows and a reset `sequence_counter`
    """
    # Step 1: Ensure data is sorted by time
    df_seq = df_seq.sort_values(by="sequence_counter").reset_index(drop=True)
    seq_len = len(df_seq)
    
    # Step 2: Define key index boundaries
    start_index = 0
    last_index = 300

    
    if  seq_len < 300: 
        return df_seq
    else:
        return df_seq.iloc[start_index:last_index]


import kaggle_evaluation.cmi_inference_server

counter = 0

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Main prediction entry point for Kaggle evaluation.

    Args:
        sequence (pl.DataFrame): Sensor data for a single sample (acc, rot, temp, etc.)
        demographics (pl.DataFrame): Subject metadata (not used currently)

    Returns:
        str: Predicted gesture label (for leaderboard submission interface)
    """
    global counter

    # Convert to pandas
    sequence_pd = sequence.to_pandas()
    # sequence_pd = sequence_sampling_300(sequence_pd)
    demographics_pd = demographics.to_pandas()  # Currently unused
    sequence_counter_length = len(sequence_pd)
    # print(sequence_counter_length)
    
    # === Step 1: Subset sensor columns ===
    CAT_COLUMNS = ['row_id', 'sequence_id', 'sequence_counter', 'subject']
    IMU_ACC_COLUMNS = ['acc_x', 'acc_y', 'acc_z']
    IMU_ROT_COLUMNS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    TEMP_COLUMNS = [f"thm_{i}" for i in range(1, 6)]

    sequence_cat_subset = sequence_pd[CAT_COLUMNS]
    sequence_acc_subset = sequence_pd[IMU_ACC_COLUMNS]
    sequence_rot_subset = sequence_pd[IMU_ROT_COLUMNS]
    sequence_rot_subset = move_column_to_end(sequence_rot_subset, 'rot_w')  # Ensure rot_w last
    sequence_temp_subset = sequence_pd[TEMP_COLUMNS]

    # === Step 2: Process IMU and temperature data ===
    imu_merged_subset = build_imu_subset(sequence_cat_subset, sequence_acc_subset, sequence_rot_subset)
    # invariant orientation:
    _, imu_temp_merged_subset = build_imu_temp_subset(sequence_temp_subset, imu_merged_subset)
    
    # # === Step 3: IMU+TEMP predictions ===
    temp_cols = TEMP_COLUMNS
    
    for step in tqdm(range(1), desc="IMU+Temp Model Prediction"):
        
        if _ == False:
            #print('imu+temp prediction') 
            predictions, gesture_probs  = imu_temp_prediction(imu_temp_merged_subset)
            # Evaluate predictions:
            gesture_df = gru_eval_predictions(gesture_probs)
            #print(gesture_df)
            gesture_most_probable = gesture_df.loc[gesture_df['probability'].idxmax(), 'key']

        else:
            #print('only imu prediction')
            predictions, gesture_probs = imu_only_prediction(imu_merged_subset)
            # Evaluate predictions:
            gesture_df = gru_eval_predictions(gesture_probs)
            #print(gesture_df)
            gesture_most_probable = gesture_df.loc[gesture_df['probability'].idxmax(), 'key']

        result = gesture_most_probable

    # === Step 5: Optional debug print every 50 calls ===
    if counter % 50 == 0:
        row_id = sequence_cat_subset['row_id'].iloc[0]
        subject = sequence_cat_subset['subject'].iloc[0]
        print(f"\nPREDICTION #{counter}")
        print(f"ğŸ“Œ SUBJECT: {subject}   |   ğŸ†” Row ID: {row_id}")
        print("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"Gesture Prediction:\n  ğŸ‘‰ Base Gesture Model = {result}")
        print("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€\n")

    counter += 1
    #print(result)
    return result


# Create mock sequence with minimal required structure
mock_sequence = pl.DataFrame({
    'row_id': [1],
    'sequence_id': [100],
    'sequence_counter': [0.0],
    'subject': [42],
    'acc_x': [0.1], 'acc_y': [0.2], 'acc_z': [0.3],
    'rot_x': [0.0], 'rot_y': [0.0], 'rot_z': [0.0], 'rot_w': [1.0],
    'thm_1': [np.nan], 'thm_2': [np.nan], 'thm_3': [np.nan], 'thm_4': [np.nan], 'thm_5': [np.nan]
})
mock_demo = pl.DataFrame({'subject': [42]})

# Required helpers must be defined: move_column_to_end, build_imu_subset, build_imu_temp_subset,
# imu_only_prediction, imu_temp_prediction

# Call predict
gesture_prediction = predict(mock_sequence, mock_demo)
print("Returned prediction:", gesture_prediction)


%%time 
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            #data_folders_dictionary['process_data'] / Path('test.csv'),
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            #data_folders_dictionary['process_data'] / Path('test_demographics.csv'),
        ))
print()

