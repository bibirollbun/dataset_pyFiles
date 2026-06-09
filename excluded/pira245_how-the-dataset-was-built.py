import numpy as np
import pandas as pd
import pathlib
from pathlib import Path


import kagglehub
from kagglehub import KaggleDatasetAdapter


#%run annex-notebook.ipynb
import cmi_sensor_data_utility_functions as my_utils


print(my_utils.notebook_folder) 
data_folders_dictionary = my_utils.data_folder(my_utils.notebook_folder)


# filepath = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
# raw_train_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


# CAT_COLUMNS = [
#     'row_id', 
#     'sequence_type', 	
#     'sequence_id', 	
#     'sequence_counter', 	
#     'subject', 	
#     'orientation', 	
#     'behavior', 	
#     'phase', 	
#     'gesture'
# ]
# gesture_cat_df = raw_train_df[CAT_COLUMNS]


# gesture_cat_df.head(5)


folder = '/kaggle/usr/lib/cmi-detect-gesture-with-a-wrist-worm-device/competition-data/process-data'
loaded_sequence_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='sequence_type_dictionary.pkl')
loaded_sequence_ids_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='sequence_ids_dictionary.pkl')
loaded_subject_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='subjects_dictionary.pkl')
loaded_orientation_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='orientation_type_dictionary.pkl')
loaded_behavior_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='behavior_type_dictionary.pkl')
loaded_phase_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='phase_type_dictionary.pkl')
loaded_gesture_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='gesture_type_dictionary.pkl')
loaded_bfrb_gesture_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='bfrb_type_dictionary.pkl')
loaded_nbfrb_gesture_type_dictionary = my_utils.handle_pickle_dict(folder=folder, pickle_filename='nbfrb_type_dictionary.pkl')


print(loaded_bfrb_gesture_type_dictionary)


# def update_bfrb_columns(df, update_bfrb_gesture_dictionary, update_nbfrb_gesture_dictionary):
#     # Initialize the new columns with zeros
#     df['bfrb_gesture'] = 0
#     df['nbfrb_gesture'] = 0

#     # Get gesture lists
#     bfrb_keys = update_bfrb_gesture_dictionary['bfrb_gesture']
#     nbfrb_keys = update_nbfrb_gesture_dictionary['nbfrb_gesture']

#     # Apply BFRB gesture mapping
#     df.loc[df['gesture'].isin(bfrb_keys.keys()), 'bfrb_gesture'] = df['gesture'].map(bfrb_keys)

#     # Apply non-BFRB gesture mapping
#     df.loc[df['gesture'].isin(nbfrb_keys.keys()), 'nbfrb_gesture'] = df['gesture'].map(nbfrb_keys)

#     return df


# updated_df = update_bfrb_columns(gesture_cat_df, loaded_bfrb_gesture_type_dictionary, loaded_nbfrb_gesture_type_dictionary)


# # Number of random rows
# n = 10
# # Subset with n random rows
# df_random_subset = updated_df.sample(n=n, random_state=42)  # Set random_state for reproducibility
# df_random_subset.head(10)


# folder = '/kaggle/usr/lib/cmi-detect-gesture-with-a-wrist-worm-device/competition-data/final-data'
# loaded_cat_value_map = my_utils.handle_pickle_dict(folder=folder, pickle_filename='cat_value_mapping_dictionary.pkl')
# gesture_cat_df = my_utils.update_dataframe_values(updated_df, columns_to_update=CAT_COLUMNS, value_map=loaded_cat_value_map)


# # Number of random rows
# n = 10
# # Subset with n random rows
# df_random_subset = gesture_cat_df.sample(n=n, random_state=42)  # Set random_state for reproducibility
# df_random_subset.head(10)


# # Save to CSV
# filepath = data_folders_dictionary['final_data'] / Path('cat_dataset.csv')
# gesture_cat_df.to_csv(filepath, index=False)


# filepath = Path('/kaggle/usr/lib/cmi-detect-gesture-with-a-wrist-worm-device/competition-data/process-data/full_rot_imu_merged_dataset.csv')
# imu_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


# imu_orientation_invariant_df = my_utils.compute_orientation_invariant_features(imu_df)


# imu_orientation_invariant_df.head(5)


# print(len(imu_orientation_invariant_df))


# # Number of random rows
# n = 10
# # Subset with n random rows
# df_random_subset = imu_orientation_invariant_df.sample(n=n, random_state=42)  # Set random_state for reproducibility
# df_random_subset.head(10)


# gesture_cat_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"pira245/cmi-dataset","mapped_categorical_dataset.csv",)
# gesture_cat_df.head(5)


# # Example: check shape and index
# for df in [gesture_cat_df, imu_orientation_invariant_df]:
#     print(len(df), df.index.is_unique)


# # Reset index before concatenation to align rows by position, not by index label
# full_imu_merged_df = pd.concat([
#     gesture_cat_df.reset_index(drop=True),
#     imu_orientation_invariant_df.reset_index(drop=True),
# ], axis=1)
# print(len(full_imu_merged_df))


# full_imu_merged_df.head(5)


# # # Save to CSV
# filepath = data_folders_dictionary['final_data'] / Path('full_rot_imu_merged_df.csv')
# full_imu_merged_df.to_csv(filepath, index=False)


# filepath = Path('/kaggle/usr/lib/cmi-detect-gesture-with-a-wrist-worm-device/competition-data/process-data/corrected_temperature_data.csv')
# temp_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


# # Number of random rows
# n = 10
# # Subset with n random rows
# df_random_subset = temp_df.sample(n=n, random_state=42)  # Set random_state for reproducibility
# df_random_subset.head(10)


# # Example: check shape and index
# for df in [full_imu_merged_df, temp_df]:
#     print(len(df), df.index.is_unique)


# # Reset index before concatenation to align rows by position, not by index label
# full_imu_temp_merged_df = pd.concat([
#     full_imu_merged_df.reset_index(drop=True),
#     temp_df.reset_index(drop=True),
# ], axis=1)
# print(len(full_imu_temp_merged_df))


# full_imu_temp_merged_df.head(5)


# # # Save to CSV
# filepath = data_folders_dictionary['final_data'] / Path('full_rot_imu_temp_merged_df.csv')
# full_imu_temp_merged_df.to_csv(filepath, index=False)

