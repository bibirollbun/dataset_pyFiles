import numpy as np
import pandas as pd
from scipy.stats import uniform
import pathlib
from pathlib import Path
# visualisation library
import IPython.display
from IPython.display import Image, display
import matplotlib
from matplotlib import pyplot as plt
import itertools
from itertools import cycle
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
sns.set(rc={'figure.figsize':(12,14)})
sns.set_theme()


import kagglehub
from kagglehub import KaggleDatasetAdapter


import cmi_sensor_data_utility_functions as my_utils


print(my_utils.notebook_folder) 
data_folders_dictionary = my_utils.data_folder(my_utils.notebook_folder)


filepath = my_utils.Path('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
raw_train_demographics_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


raw_train_demographics_df.shape


raw_train_demographics_df.head(10)


train_shape = raw_train_demographics_df.shape
train_dtypes = raw_train_demographics_df.dtypes
missing_counts = raw_train_demographics_df.isna().sum()
missing_percent = ((raw_train_demographics_df.isnull().sum() / len(raw_train_demographics_df)) * 100).sort_values()

summary = f"""
Train Demographics Dataset Summary:
- Shape: {train_shape[0]} rows, {train_shape[1]} columns
- Data types:
{train_dtypes.to_string()}
- Missing values (count):
{missing_counts.to_string()}
- Missing values (percentage):
{missing_percent.to_string()}

"""
print(summary)


DEMO_COLUMNS = [
    'subject',
    'adult_child',       
    'age',    
    'sex',     
    'handedness',     
    'height_cm',        
    'shoulder_to_wrist_cm', 
    'elbow_to_wrist_cm', 
]
DEMO_COLUMNS


sns.pairplot(raw_train_demographics_df[DEMO_COLUMNS], corner=True)


# Plot histograms
my_utils.display_3x_histoplot(raw_train_demographics_df, ['sex', 'adult_child', 'handedness'], figsize=(18, 5))


sns.histplot(data=raw_train_demographics_df, x="age", hue="sex", stat="percent")


sns.histplot(data=raw_train_demographics_df, x="height_cm", hue="sex", stat="percent")


# Scatter plot: shoulder_to_wrist_cm vs elbow_to_wrist_cm
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=raw_train_demographics_df,
    x="shoulder_to_wrist_cm",
    y="elbow_to_wrist_cm"
)
plt.title("Scatter Plot: Shoulder to Wrist vs Elbow to Wrist")
plt.xlabel("Shoulder to Wrist (cm)")
plt.ylabel("Elbow to Wrist (cm)")
plt.show()

# Boxplot for both variables
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(y=raw_train_demographics_df["shoulder_to_wrist_cm"], ax=axes[0])
axes[0].set_title("Boxplot: Shoulder to Wrist (cm)")
sns.boxplot(y=raw_train_demographics_df["elbow_to_wrist_cm"], ax=axes[1])
axes[1].set_title("Boxplot: Elbow to Wrist (cm)")
plt.tight_layout()
plt.show()


# Calculate IQR for 'shoulder_to_wrist_cm'

q1 = raw_train_demographics_df['shoulder_to_wrist_cm'].quantile(0.25)
q3 = raw_train_demographics_df['shoulder_to_wrist_cm'].quantile(0.75)
iqr = q3 - q1

# Define outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Find outlier rows
outliers = raw_train_demographics_df[
    (raw_train_demographics_df['shoulder_to_wrist_cm'] < lower_bound) |
    (raw_train_demographics_df['shoulder_to_wrist_cm'] > upper_bound)
]

outliers


# Calculate IQR for 'elbow_to_wrist_cm'

q1 = raw_train_demographics_df['elbow_to_wrist_cm'].quantile(0.25)
q3 = raw_train_demographics_df['elbow_to_wrist_cm'].quantile(0.75)
iqr = q3 - q1

# Define outlier bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Find outlier rows
outliers = raw_train_demographics_df[
    (raw_train_demographics_df['elbow_to_wrist_cm'] < lower_bound) |
    (raw_train_demographics_df['elbow_to_wrist_cm'] > upper_bound)
]

outliers


filepath = my_utils.Path('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')
raw_test_demographics_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


raw_test_demographics_df.shape


raw_test_demographics_df.head(10)


filepath = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
raw_train_df = pd.read_csv(filepath, sep=',', encoding='utf-8', na_filter=True)


raw_train_df.head(5)


train_shape = raw_train_df.shape
train_shape


train_shape = raw_train_df.shape
train_dtypes = raw_train_df.dtypes
missing_counts = raw_train_df.isna().sum()
missing_percent = ((raw_train_df.isnull().sum() / len(raw_train_df)) * 100).sort_values()

summary = f"""
Train Dataset Summary:
- Shape: {train_shape[0]} rows, {train_shape[1]} columns
- Data types:
{train_dtypes.to_string()}
- Missing values (count):
{missing_counts.to_string()}
- Missing values (percentage):
{missing_percent.to_string()}
"""
#print(summary)


column = 'subject'
my_utils.count_unique_values(raw_train_df, column)


column = 'sequence_id'
my_utils.count_unique_values(raw_train_df, column)


# Group by 'subject' and count the number of unique 'sequence_id' per subject
unique_sequences_id_df = (
    raw_train_df.groupby("subject")["sequence_id"]
    .nunique()  # Count unique sequence IDs
    .reset_index(name="unique_sequences")  # Convert index to column and rename
    .sort_values("unique_sequences", ascending=False)  # Sort by count descending
)

# Plot the results as a bar chart
unique_sequences_id_df.plot(
    kind="bar",  # Bar chart
    x="subject",  # X-axis: subjects
    y="unique_sequences",  # Y-axis: number of unique sequences
    title="Number of Unique Sequences per Subject",
    xlabel="Subject",
    ylabel="Unique Sequences",
    figsize=(16, 5)  # Wide but short figure
)

plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()  # Display the plot


column = 'sequence_counter'
my_utils.count_unique_values(raw_train_df, column)


raw_train_df["sequence_counter"].value_counts()


value_map = {} #  DICTIONARY TO MAP THE VALUES OF ALL CATEGORICAL DATA


# create and save a subject dictionary
dic = my_utils.extract_unique_values_as_dict(raw_train_df, column='subject', dict_key='subjects')
del dic['subjects']
data = {'subject':dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='subjects_dictionary.pkl', data_dict = data);
# Bulding value map dictionary:
value_map['subject'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


# create and save a sequence_id dictionary
dic = my_utils.extract_unique_values_as_dict(raw_train_df, column='sequence_id', dict_key='sequence_ids')
del dic['sequence_ids']
data = {'sequence_id': dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='sequence_ids_dictionary.pkl', data_dict = data);
# Bulding value map dictionary:
value_map['sequence_id'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


# Group by 'sequence_type' and count unique 'sequence_counter'
unique_sequences_type_df = (
    raw_train_df.groupby("sequence_type")["sequence_counter"]
    .nunique()
    .reset_index(name="unique_sequences")
    .sort_values("unique_sequences", ascending=False)
)

# Plot using seaborn to allow color grouping (hue)
plt.figure(figsize=(16, 4))
sns.barplot(
    data=unique_sequences_type_df,
    x="sequence_type",
    y="unique_sequences",
    hue="sequence_type",  # Hue differentiates colors per sequence_type
    dodge=False,  # Avoid offsetting bars (since x and hue are the same)
    palette="Set2",
)

# Formatting the plot
plt.title("Number of Unique Sequence Counters per Sequence Type")
plt.xlabel("Sequence Type")
plt.ylabel("Unique Sequence Counters")
plt.legend(title="Sequence Type")
plt.tight_layout()
plt.show()


my_utils.plot_donut_distribution(raw_train_df, ["sequence_type"], "Unique Sequences by Sequence Type")


dic = my_utils.extract_unique_values_as_dict(raw_train_df, column='sequence_type', dict_key='sequence-type')
# {'sequence-type': ['Target', 'Non-Target'], 'Target': 1, 'Non-Target': 2}
del dic['sequence-type']
data = {'sequence_type':dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='sequence_type_dictionary.pkl', data_dict = data)
# Bulding value map dictionary:
value_map['sequence_type'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


# Group by 'phase-type' and count unique 'sequence_counter'
unique_sequences_df = (
    raw_train_df.groupby("phase")["sequence_counter"]
    .nunique()
    .reset_index(name="unique_sequences")
    .sort_values("unique_sequences", ascending=False)
)

# Plot with seaborn to distinguish phase-types by color
plt.figure(figsize=(12, 4))
sns.barplot(
    data=unique_sequences_df,
    x="phase",
    y="unique_sequences",
    hue="phase",    # Color bars by phase-type
    dodge=False,
    palette="Set2"
)

# Customize chart
plt.title("Number of Unique Sequence Counters per Phase Type")
plt.xlabel("Phase Type")
plt.ylabel("Unique Sequence Counters")
plt.legend(title="Phase Type")
plt.tight_layout()
plt.show()


my_utils.plot_donut_distribution(raw_train_df, ["sequence_type", "phase"], "Unique Sequences by Sequence Type and Phase")


dic = my_utils.extract_unique_values_as_dict(raw_train_df, column='phase', dict_key='phase-type')
# {'phase-type': ['Transition', 'Gesture'], 'Transition': 1, 'Gesture': 2}
del dic['phase-type']
data = {'phase':dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='phase_type_dictionary.pkl', data_dict = data)
# Bulding value map dictionary:
value_map['phase'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


column = 'gesture'
my_utils.count_unique_values(raw_train_df, column)


my_utils.plot_donut_distribution(raw_train_df, ["sequence_type", "gesture"], "Unique Sequences by Sequence Type and Gesture")


dic = my_utils.extract_unique_values_as_dict(raw_train_df, column=column, dict_key='gesture-type')
del dic['gesture-type']
data = {'gesture_type':dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='gesture_type_dictionary.pkl', data_dict = data)
# Bulding value map dictionary:
value_map['gesture'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


# Filter rows where sequence_type is 'Target'
bfrb_filtered_df = raw_train_df[raw_train_df['sequence_type'] == 'Target']
dic = my_utils.extract_unique_values_as_dict(bfrb_filtered_df, column=column, dict_key='bfrb-type')
del dic['bfrb-type']
data = {'bfrb_gesture':dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='bfrb_type_dictionary.pkl', data_dict = data)
# Bulding value map dictionary:
value_map['bfrb_gesture'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


# Group by 'gesture' and count unique 'sequence_counter'
unique_sequences_bfrb_gesture_df = (
    bfrb_filtered_df.groupby("gesture")["sequence_counter"]
    .nunique()
    .reset_index(name="unique_sequences")
    .sort_values("unique_sequences", ascending=False)
)

# Set plot size and style
plt.figure(figsize=(16, 6))
ax = sns.barplot(
    data=unique_sequences_bfrb_gesture_df,
    x="gesture",
    y="unique_sequences",
    palette="Set2"
)

# Rotate x-axis labels for better spacing
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


# Chart titles and labels
plt.title("Number of Unique Sequence Counters per BFRB Gestures")
plt.xlabel("Gesture")
plt.ylabel("Unique Sequence Counters")
plt.tight_layout()
plt.show()


my_utils.plot_donut_distribution(bfrb_filtered_df, ["sequence_type", "gesture"], "Unique Sequences by Sequence Type and BFRB Gestures")


# Filter rows where sequence_type is 'Non-Target'
nbfrb_filtered_df = raw_train_df[raw_train_df['sequence_type'] == 'Non-Target']
dic = my_utils.extract_unique_values_as_dict(nbfrb_filtered_df, column=column, dict_key='nbfrb-type')
del dic['nbfrb-type']
data = {'nbfrb_gesture': dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='nbfrb_type_dictionary.pkl', data_dict = data)
# Bulding value map dictionary:
value_map['nbfrb_gesture'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


# Group by 'gesture' and count unique 'sequence_counter'
unique_sequences_nbfrb_gesture_df = (
    nbfrb_filtered_df.groupby("gesture")["sequence_counter"]
    .nunique()
    .reset_index(name="unique_sequences")
    .sort_values("unique_sequences", ascending=False)
)

# Set plot size and style
plt.figure(figsize=(16, 6))
ax = sns.barplot(
    data=unique_sequences_nbfrb_gesture_df,
    x="gesture",
    y="unique_sequences",
    palette="Set2"
)

# Rotate x-axis labels for better spacing
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# Chart titles and labels
plt.title("Number of Unique Sequence Counters per Non-BFRB Gestures")
plt.xlabel("Gesture")
plt.ylabel("Unique Sequence Counters")
plt.tight_layout()
plt.show()


my_utils.plot_donut_distribution(nbfrb_filtered_df, ["sequence_type", "gesture"], "Unique Sequences by Sequence Type and Non-BFRB Gesture")


my_utils.plot_donut_distribution(raw_train_df, ["sequence_type", "behavior"], "Unique Sequences by Sequence type and Behavior")


# behavior
dic = my_utils.extract_unique_values_as_dict(raw_train_df, column='behavior', dict_key='behavior-type')
del dic['behavior-type']
data =  {'behavior': dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='behavior_type_dictionary.pkl', data_dict = data)
# Bulding value map dictionary:
value_map['behavior'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


my_utils.plot_donut_distribution(raw_train_df, ["sequence_type", "orientation"], "Unique Sequences by Sequence Type and Orientation")


#Orientation
dic = my_utils.extract_unique_values_as_dict(raw_train_df, column='orientation', dict_key='orientation-type')
del dic['orientation-type']
data = {'orientation':dic}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='orientation_type_dictionary.pkl', data_dict = data)
# Bulding value map dictionary:
value_map['orientation'] = dic
my_utils.handle_pickle_dict(folder=data_folders_dictionary['final_data'], pickle_filename='cat_value_mapping_dictionary.pkl', data_dict = value_map);


TEMP_COLUMNS = [
    'row_id',
    *[f"thm_{i}" for i in range(1, 6)]
]
TEMP_COLUMNS


temp_raw_df =  raw_train_df[TEMP_COLUMNS]


_shape = temp_raw_df.shape
_dtypes = temp_raw_df.dtypes
missing_counts = temp_raw_df.isna().sum()
missing_percent = ((temp_raw_df.isnull().sum() / len(temp_raw_df)) * 100).sort_values()

summary = f"""
Train Dataset Summary:
- Shape: {_shape[0]} rows, {_shape[1]} columns
- Data types:
{_dtypes.to_string()}
- Missing values (count):
{missing_counts.to_string()}
- Missing values (percentage):
{missing_percent.to_string()}
"""
print(summary)


# 1. Find rows with at least one NaN
nan_rows = temp_raw_df[temp_raw_df.isna().any(axis=1)]
# 2. Get index values of those rows
nan_indices = nan_rows.index.tolist()
# Display results
print("Rows with missing values:\n", nan_rows)
#print("Indices of rows with NaNs:", nan_indices)


# 3. Get the dicitonary of nan_indices 
dic_nan_indices = my_utils.get_nan_indices(temp_raw_df)
#dic_nan_indices


# rows where the original values are missing: Investigate original values
temp_raw_missing_df = temp_raw_df.loc[nan_indices]


# Display descriptive statistics
print(temp_raw_missing_df.describe())


temp_stats_df = temp_raw_df.describe()
temp_stats_df.head(10)


# Plot temperature sensor 1 (TMP1) histogram
my_utils.display_histoplots(raw_train_df, ['thm_1'], base_width=6, height=5)


# Plot temperature sensor 2&3 (TMP2-3) histograms
my_utils.display_histoplots(raw_train_df, ['thm_2', 'thm_3'], base_width=6, height=5)


# Plot temperature sensor 4&5 (TMP4-5) histograms
my_utils.display_histoplots(raw_train_df, ['thm_4', 'thm_5'], base_width=6, height=5)


missing_temp_substitution_values = {
    'thm_1': 26.982324,
    'thm_2': 26.354338,
    'thm_3': 26.956276,
    'thm_4': 27.742224,
    'thm_5': 29.500000
}
nan_indices = dic_nan_indices.copy()
substitution_dict = missing_temp_substitution_values.copy()
temp_filled_df = my_utils.replace_missing_values(temp_raw_df, nan_indices, substitution_dict)


# 1. Find rows with at least one NaN
nan_rows = temp_filled_df[temp_filled_df.isna().any(axis=1)]
# 2. Get index values of those rows
nan_indices = nan_rows.index.tolist()
# Display results
print("Rows with missing values:\n", nan_rows)
print("Indices of rows with NaNs:", nan_indices)


# Plot temperature sensor 1 (TMP1) histogram
my_utils.display_histoplots(temp_filled_df, ['thm_5'], base_width=6, height=5)


# Save to CSV
filepath = data_folders_dictionary['process_data'] / Path('corrected_temperature_data.csv')
temp_filled_df.to_csv(filepath, index=False)


T_COLUMN = [
    'sequence_counter'
]
T_COLUMN


tdata = raw_train_df[T_COLUMN].to_numpy(dtype='float')


tdata.shape


print(tdata)


IMU_ACC_COLUMNS = [
    'row_id',
    'acc_x',
    'acc_y',
    'acc_z',
]
IMU_ACC_COLUMNS
ACC_COLUMNS = [
    'acc_x',
    'acc_y',
    'acc_z',
]
ACC_COLUMNS


imu_raw_acc_df =  raw_train_df[IMU_ACC_COLUMNS]


# 1. Find rows with at least one NaN
nan_rows = imu_raw_acc_df[imu_raw_acc_df.isna().any(axis=1)]
# 2. Get index values of those rows
nan_indices = nan_rows.index.tolist()
# Display results
print("Rows with missing values:\n", nan_rows)
print("Indices of rows with NaNs:", nan_indices)


# check on original values
imu_raw_acc_df.loc[nan_indices]


imu_raw_acc_df.head(10)


acc_data = imu_raw_acc_df[ACC_COLUMNS].to_numpy(dtype='float')


acc_data.shape


IMU_ROT_COLUMNS = [
    'row_id',
    'rot_w',
    'rot_x',
    'rot_y',
    'rot_z'
]
IMU_ROT_COLUMNS


imu_raw_rot_df =  raw_train_df[IMU_ROT_COLUMNS]


imu_raw_rot_df.head(10)


# 1. Find rows with at least one NaN
nan_rows = imu_raw_rot_df[imu_raw_rot_df.isna().any(axis=1)]
# 2. Get index values of those rows
nan_indices = nan_rows.index.tolist()
# Display results
print("Rows with missing values:\n", nan_rows)
#print("Indices of rows with NaNs:", nan_indices)


# 3. Get the dicitonary of nan_indices 
dic_nan_indices = my_utils.get_nan_indices(imu_raw_rot_df)
#dic_nan_indices


missing_counts = imu_raw_rot_df.isna().sum()
print(missing_counts)


missing_percent = ((imu_raw_rot_df.isnull().sum() / len(imu_raw_rot_df)) * 100).sort_values()
print(missing_percent)


# For example: Investigate original values
imu_raw_rot_df.loc[nan_indices]

raw_train_imu_missing_value_df = raw_train_df.loc[nan_indices]


my_utils.plot_donut_distribution(raw_train_imu_missing_value_df, ["sequence_type"], "Base 1: Unique Sequences by Sequence Type")


# Group by 'subject' and count the number of unique 'sequence_id' per subject
unique_sequences_id_df = (
    raw_train_imu_missing_value_df.groupby("subject")["sequence_id"]
    .nunique()  # Count unique sequence IDs
    .reset_index(name="unique_sequences")  # Convert index to column and rename
    .sort_values("unique_sequences", ascending=False)  # Sort by count descending
)

# Plot the results as a bar chart
unique_sequences_id_df.plot(
    kind="bar",  # Bar chart
    x="subject",  # X-axis: subjects
    y="unique_sequences",  # Y-axis: number of unique sequences
    title="Number of Unique Sequences per Subject",
    xlabel="Subject",
    ylabel="Unique Sequences",
    figsize=(16, 5)  # Wide but short figure
)

plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()  # Display the plot


# Define quaternion columns
quat_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
# 1. Find rows with at least one NaN
nan_rows = imu_raw_rot_df[quat_cols].isna().any(axis=1)
# 2. Get index values of those rows
nan_indices = imu_raw_rot_df[nan_rows].index.tolist()
# 3. Replace only quaternion columns in those rows
imu_raw_rot_df.loc[nan_indices, quat_cols] = [1.0, 0.0, 0.0, 0.0]
# 4. Compute quaternion norms row-wise
quat_norm = np.linalg.norm(imu_raw_rot_df[quat_cols].values, axis=1)
# 5. Identify zero-norm quaternions
mask_zero_norm = quat_norm == 0
# 6. Replace those with the identity quaternion
imu_raw_rot_df.loc[mask_zero_norm, quat_cols] = [1.0, 0.0, 0.0, 0.0]
# 7. Optional: make a copy
cleaned_imu_raw_rot_df = imu_raw_rot_df.copy()


cleaned_imu_raw_rot_df.shape


missing_percent = ((cleaned_imu_raw_rot_df.isnull().sum() / len(cleaned_imu_raw_rot_df)) * 100).sort_values()
print(missing_percent)


dic_rot_nan_indices_to_be_drop = {"rot_nan_indices": nan_indices}
my_utils.handle_pickle_dict(folder=data_folders_dictionary['process_data'], pickle_filename='rot_nan_indices_dictionary.pkl', data_dict = dic_rot_nan_indices_to_be_drop);


col_name = "rot_w"
scipy_cleaned_imu_raw_rot_df = my_utils.move_column_to_end(cleaned_imu_raw_rot_df, col_name)


scipy_cleaned_imu_raw_rot_df.head(5)


# Drop nan rows: in timestamp data
tdata = raw_train_df[T_COLUMN].to_numpy(dtype='float')


tdata.shape


acc_data.shape


ROT_COLUMNS = [
    'rot_x',
    'rot_y',
    'rot_z',
    'rot_w'
]
ROT_COLUMNS
q = scipy_cleaned_imu_raw_rot_df[ROT_COLUMNS].to_numpy(dtype='float')


q.shape


[acc_world, acc_motion, velocity, position] = my_utils.process_all_imu_sequences(
    tdata=tdata,
    acc_data=acc_data,
    q=q
)


acc_world.shape


# Column names list
columns = ['acc_world_x', 'acc_world_y', 'acc_world_z']
# Create DataFrame
acc_world_df = pd.DataFrame(acc_world, columns=columns)
# Display result
acc_world_df.head(5)


acc_motion.shape


# Column names list
columns = ['acc_motion_x', 'acc_motion_y', 'acc_motion_z']
# Create DataFrame
acc_motion_df = pd.DataFrame(acc_motion, columns=columns)
# Display result
acc_motion_df.head(5)


velocity.shape


# Column names list
columns = ['velocity_x', 'velocity_y', 'velocity_z']
# Create DataFrame
velocity_df = pd.DataFrame(velocity, columns=columns)
# Display result
velocity_df.head(5)


position.shape


# Column names list
columns = ['position_x', 'position_y', 'position_z']
# Create DataFrame
position_df = pd.DataFrame(position, columns=columns)
# Display result
position_df.head(5)


ID_COLUMNS = [
    'row_id',
    'sequence_counter'
]
ID_COLUMNS


# Identification dataframe :
id_df = raw_train_df[ID_COLUMNS]


# Example: check shape and index
for df in [id_df, imu_raw_acc_df, scipy_cleaned_imu_raw_rot_df, acc_world_df, acc_motion_df, velocity_df, position_df]:
    print(len(df), df.index.is_unique)


# Reset index before concatenation to align rows by position, not by index label
imu_merged_df = pd.concat([
    id_df.reset_index(drop=True),
    imu_raw_acc_df[ACC_COLUMNS].reset_index(drop=True),
    scipy_cleaned_imu_raw_rot_df[ROT_COLUMNS].reset_index(drop=True),
    acc_world_df.reset_index(drop=True),
    acc_motion_df.reset_index(drop=True),
    velocity_df.reset_index(drop=True),
    position_df.reset_index(drop=True)
], axis=1)
print(len(imu_merged_df))


imu_merged_df.head(5)


# Save merged DataFrame to CSV
filepath = data_folders_dictionary['process_data'] / Path('full_rot_imu_merged_dataset.csv')
imu_merged_df.to_csv(filepath, index=False)


ToF_COLUMNS = [
    'row_id',
    *[f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)],
]
#ToF_COLUMNS


# Filter only Tof data with row id column: 
tof_raw_sensor_df =  raw_train_df[ToF_COLUMNS]


tof_raw_sensor_df.head(10)


_shape = tof_raw_sensor_df.shape
missing_percent = ((tof_raw_sensor_df.isnull().sum() / len(tof_raw_sensor_df)) * 100).sort_values()
no_response_percent = ((tof_raw_sensor_df == -1).sum() / len(tof_raw_sensor_df) * 100).sort_values()

summary = f"""
- Shape: {_shape[0]} rows, {_shape[1]} columns
- Missing values (percentage):
{missing_percent.to_string()}
- No response values (-1) (percentage):
{no_response_percent.to_string()}
"""
#print(summary)


# === Sensor Positions in mm (relative to the origin) ===
tof_sensor_positions_mm = {
    1: (0, 12.5),
    2: (0, 22.5),
    3: (15.5, 0),
    4: (0, -22.5),
    5: (-15.5, 0)
}


# === Visualization of Sensor Layout in 2D ===
layout_fig, layout_ax = plt.subplots(figsize=(10, 10))
layout_ax.set_title("2D Sensor Layout (mm)")
layout_ax.set_xlabel("X (mm)")
layout_ax.set_ylabel("Y (mm)")
layout_ax.set_aspect('equal')

# Draw each sensor with label
for sid, (x, y) in tof_sensor_positions_mm.items():
    layout_ax.plot(x, y, 'bo')
    layout_ax.text(x + 1.5, y, f"Sensor {sid}", fontsize=9)
layout_ax.plot(0, 0, 'r^', markersize=10, label="Eye (origin)")
layout_ax.legend()

# Save sensor layout image
sensor_layout_filepath = data_folders_dictionary['media_data'] / Path('tof_sensor_layout.png')
layout_fig.savefig(sensor_layout_filepath)
plt.close(layout_fig)


display(Image(filename=sensor_layout_filepath))


sensor_reading_df = pd.DataFrame([tof_raw_sensor_df.iloc[2]]).reset_index(drop=True)
sensor_reading_df


sensor_readings = sensor_reading_df.drop('row_id',axis=1)
row_id = sensor_reading_df['row_id'].iloc[0]


dir = data_folders_dictionary['process_data']
[depth_points, normalized_image, sensor_histograms] = my_utils.process_tof_sensors_depth_image(tof_sensor_positions_mm, sensor_readings, output_resolution='option_b', output_dir=dir)


#my_utils.display_point_cloud_3d_plotly(points=depth_points, title="3D Points Cloud", color_by="z", point_size=3, axis_unit="m")


my_utils.display_point_cloud_3d(points=depth_points, title="3D Points Cloud", elev=40, azim=45, point_size=3, color_by="z")


#plt.imshow(normalized_image)


filepath = data_folders_dictionary['media_data'] / Path('histogram_path.png')
my_utils.process_tof_sensors_depth_image_histograms(normalized_image=normalized_image, sensor_histograms_mm=sensor_histograms, histogram_path=filepath)


display(Image(filename=filepath))


imu_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"pira245/cmi-dataset","full_imu_temp_tof_merged_v2_df.csv",)
imu_df = imu_df.drop(['row_id.1' , 'row_id.2', 'row_id.3', 'sequence_counter.1'], axis=1)


main_gesture_df = imu_df.copy()


# Set the threshold for the sequence lenght
sequence_length_threshold = 300  # Default value (can be changed as needed)

# Step 1: Compute sequence-level statistics for 'sequence_counter'
seq_stats = main_gesture_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])

# Step 2: Identify sequence_ids where length exceeds the threshold
long_sequence_ids = seq_stats[seq_stats["count"] > sequence_length_threshold].index

# Step 3: Filter the DataFrame to rows from long sequences
long_sequences_df = main_gesture_df[main_gesture_df["sequence_id"].isin(long_sequence_ids)]

# Step 4: Count gestures associated with long sequences (based on first label per sequence)
long_gesture_counts = (
    long_sequences_df
    .groupby("sequence_id")["gesture"]
    .first()
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"index": "gesture"})
)

# Step 5: Print the result
print(f"Gestures with sequences longer than {sequence_length_threshold} timesteps:")
print(long_gesture_counts)

# Step 6: Store the long sequence_ids for future removal
long_sequence_ids_list = list(long_sequence_ids)
print(f"\n Sequence IDs to consider removing ({len(long_sequence_ids_list)} total):")
print(long_sequence_ids_list[:10], "...")  # Show sample

# Step 7: Optionally remove long sequences from the original dataframe
main_gesture_filtered_df = main_gesture_df[~main_gesture_df["sequence_id"].isin(long_sequence_ids)]
special_case_df = main_gesture_df[main_gesture_df["sequence_id"].isin(long_sequence_ids)]

# Step 8: Confirm shape of the filtered dataset
print(f"\n✅ New DataFrame shape after removal: {main_gesture_filtered_df.shape}")


# Get the first row per sequence_id and count the gesture values
gesture_counts = (
    main_gesture_filtered_df
    .groupby("sequence_id")           # Group by sequence
    .first()["gesture"]          # Take the first gesture of each group
    .value_counts()                   # Count each unique gesture
    .reset_index(name="count")        # Reset index and name the count column
    .rename(columns={"index": "gesture"})  # Rename index column for clarity
)

# Display result
print(gesture_counts)


# Sequence statistics (min, max, length of each sequence)
seq_stats = main_gesture_filtered_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])
num_sequences = main_gesture_filtered_df["sequence_id"].nunique()
num_sequences, seq_stats.describe()


# Get the first row per sequence_id and count the gesture values
gesture_counts = (
    special_case_df
    .groupby("sequence_id")           # Group by sequence
    .first()["gesture"]          # Take the first gesture of each group
    .value_counts()                   # Count each unique gesture
    .reset_index(name="count")        # Reset index and name the count column
    .rename(columns={"index": "gesture"})  # Rename index column for clarity
)

# Display result
print(gesture_counts)


# Sequence statistics (min, max, length of each sequence)
seq_stats = special_case_df.groupby("sequence_id")["sequence_counter"].agg(["min", "max", "count"])
num_sequences = special_case_df["sequence_id"].nunique()
num_sequences, seq_stats.describe()

