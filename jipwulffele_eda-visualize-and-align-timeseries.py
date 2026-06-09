import os

import pandas as pd
pd.options.display.max_columns = 100

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
sys.path.append('../Data/cmi-detect-behavior-with-sensor-data')

import kaggle_evaluation.cmi_inference_server



!pip install /kaggle/input/pip-install-dependencies-cmi/scikit_base-0.12.3-py3-none-any.whl
!pip install /kaggle/input/pip-install-dependencies-cmi/scikit_learn-1.7.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/pip-install-dependencies-cmi/sktime-0.38.1-py3-none-any.whl


# Training data
df_train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
df_train_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

# Save copy to use for model training only
df_train_copy = df_train.copy()
df_train_demo_copy = df_train_demo.copy()

# Testing data
df_test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
df_test_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")


df_train.head()


cmap = sns.color_palette("inferno", 4)


# Prepare a reference table 

df_ref = df_train[['sequence_id', 'subject', 'gesture', 'sequence_type']].drop_duplicates()
df_ref.head()


# Extract unique gestures
unique_gestures = df_ref.gesture.unique()


import random

def subplot_timeseries(axes, x_data, y_data, ax, x_1, x_2, x_3, x_4, title=None, x_label=None, y_label=None):
    
    # Background
    axes[ax].fill_between(x=x_1, y1=max(y_data)+1, y2=min(y_data)-1,
                        color=cmap[0], alpha=0.3)
    axes[ax].fill_between(x=x_2, y1=max(y_data)+1, y2=min(y_data)-1,
                        color=cmap[1], alpha=0.3)
    axes[ax].fill_between(x=x_3, y1=max(y_data)+1, y2=min(y_data)-1,
                        color=cmap[2], alpha=0.3)
    axes[ax].fill_between(x=x_4, y1=max(y_data)+1, y2=min(y_data)-1,
                        color=cmap[3], alpha=0.3)

    # Line plot
    sns.lineplot(x=x_data, y=y_data,
                 color="black",
                 ax=axes[ax])

    # Axes
    axes[ax].set_title(title)
    axes[ax].set_xlabel(x_label)
    axes[ax].set_ylabel(y_label)
    

def plot_acceleration(df, seq_id):

    # Extract gesture phase
    x_1 = df.sequence_counter[df.behavior == "Relaxes and moves hand to target location"]
    x_2 = df.sequence_counter[df.behavior == "Moves hand to target location"]
    x_3 = df.sequence_counter[df.behavior == "Hand at target location"]
    x_4 = df.sequence_counter[df.behavior == "Performs gesture"]
    
    # Set-up plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    subplot_timeseries(axes, df.sequence_counter, df.acc_x, 0, 
                       x_1, x_2, x_3, x_4,
                      "Acceleration in x", "Sequence counter", "Acceleration")
    subplot_timeseries(axes, df.sequence_counter, df.acc_y, 1, 
                       x_1, x_2, x_3, x_4,
                      "Acceleration in y", "Sequence counter", "Acceleration")
    subplot_timeseries(axes, df.sequence_counter, df.acc_z, 2, 
                       x_1, x_2, x_3, x_4,
                      "Acceleration in z", "Sequence counter", "Acceleration")

    fig.suptitle(f"Acceleration Data for Sequence {seq_id}: {df.reset_index().gesture[0]}", fontsize=12)
    
    plt.tight_layout()
    plt.show()


for gesture in unique_gestures:
    df_gesture = df_train[df_train.gesture == gesture].reset_index()
    rand_n = random.randint(0, df_gesture.shape[0])
    seq_id = df_gesture.sequence_id[rand_n]
    df_id = df_gesture[df_gesture.sequence_id == seq_id]
    plot_acceleration(df_id, seq_id)


from scipy.interpolate import interp1d

def norm_serie(seq_id, group, col, len_total, start_p2, start_p3):

    # Extract values
    times = group['sequence_counter'].values
    values = group[col].values
    
    # Start of phase 2 and 3
    t_start_p2 = group.sequence_counter[(group['behavior'] == 'Hand at target location')].iloc[0]
    t_start_p3 = group.sequence_counter[(group['behavior'] == 'Performs gesture')].iloc[0]

    # Time min and max
    t_min = times.min()
    t_max = times.max()
    
    # Normalize time in 3 segments
    norm_time = []
    for t in times:
        if t < t_start_p2: # Moves hand to position
            norm = (t - t_min) / (t_start_p2 - t_min) * start_p2
        elif t <= t_start_p3: # Hand at position
            norm = ((t - t_start_p2) / (t_start_p3 - t_start_p2)) * (start_p3 - start_p2) + start_p2
        else: # Performs action
            norm = ((t - t_start_p3) / (t_max - t_start_p3)) * (1 - start_p3) + start_p3
        norm_time.append(norm)

    # Interpolate to fixed-length time axis
    f = interp1d(norm_time, values, kind='linear', bounds_error=False, fill_value="extrapolate")
    x_out = np.linspace(0, 1, len_total)
    y_interp = f(x_out)
    return y_interp


def allign_series(df, col):

    # Get general statistics (median phase length (moves, relaxes, perfoms gesture etc))
    len_p1 = round(np.median(df_gesture[(df_gesture.behavior == 'Relaxes and moves hand to target location') | (df_gesture.behavior == 'Moves hand to target location')].groupby(['sequence_id', 'behavior']).size().values))
    len_p2 = round(np.median(df_gesture[(df_gesture.behavior == 'Hand at target location')].groupby(['sequence_id', 'behavior']).size().values))
    len_p3 = round(np.median(df_gesture[(df_gesture.behavior == 'Performs gesture')].groupby(['sequence_id', 'behavior']).size().values))
    len_total = len_p1 + len_p2 + len_p3

    # Calculate relative starting points of p2 and p3
    start_p2 = len_p2 / len_total
    start_p3 = (len_p2 + len_p3) / len_total

    # Initialize alligned series
    aligned_series = []

    for seq_id, group in df.groupby("sequence_id"):
        aligned_series.append(norm_serie(seq_id, group, col, len_total, start_p2, start_p3))

    return aligned_series, start_p2, start_p3


def subplot_alligned(aligned_series, start_p2, start_p3, axes, ax, title, x_label, y_label, spagetti=True):

    # Prepare data
    aligned_array = np.vstack(aligned_series)
    mean_vals = aligned_array.mean(axis=0)
    std_vals = aligned_array.std(axis=0)
    x = np.linspace(0, 1, aligned_array.shape[1])

    # Plot background
    axes[ax].axvline(start_p2, linestyle='--', color=[0.5, 0.5, 0.5])
    axes[ax].axvline(start_p3, linestyle='--', color=[0.5, 0.5, 0.5])

    if spagetti:
        # Plot data
        for i, series in enumerate(aligned_array):
            axes[ax].plot(x, series, color=cmap[3], alpha=0.5)
        
        axes[ax].plot(x, mean_vals, color='black')
    else: 
        # Plot data
        axes[ax].plot(x, mean_vals, color='black')
        axes[ax].fill_between(x, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, color='gray')

    # Axes
    axes[ax].set_title(title)
    axes[ax].set_xlabel(x_label)
    axes[ax].set_ylabel(y_label)
    

def plot_alligned_acc(df, gesture):

    # Get alligned series
    aligned_acc_x, start_p2, start_p3 = allign_series(df_gesture, 'acc_x')
    aligned_acc_y, _, _ = allign_series(df_gesture, 'acc_y')
    aligned_acc_z, _, _ = allign_series(df_gesture, 'acc_z')
    
    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    subplot_alligned(aligned_acc_x, start_p2, start_p3, axes, 0,
                    "Acceleration in x", "Sequence counter", "Acceleration")
    subplot_alligned(aligned_acc_y, start_p2, start_p3, axes, 1,
                    "Acceleration in y", "Sequence counter", "Acceleration")
    subplot_alligned(aligned_acc_z, start_p2, start_p3, axes, 2,
                    "Acceleration in z", "Sequence counter", "Acceleration")
    
    fig.suptitle(f"Acceleration Data: {gesture}", fontsize=12)
    
    plt.tight_layout()
    plt.show()


random.seed(2025)

for gesture in unique_gestures:

    df_gesture = df_train[df_train.gesture == gesture].reset_index()
    seq_ids = list(df_gesture.sequence_id.unique())
    
    N = 10
    seq_ids_sampled = random.sample(seq_ids, N)
    df_gesture = df_gesture[df_gesture.sequence_id.isin(seq_ids_sampled)]
    
    plot_alligned_acc(df_gesture, gesture)


def plot_alligned_rot(df, gesture):

    # Get alligned series
    aligned_rot_x, start_p2, start_p3 = allign_series(df_gesture, 'rot_x')
    aligned_rot_y, _, _ = allign_series(df_gesture, 'rot_y')
    aligned_rot_z, _, _ = allign_series(df_gesture, 'rot_z')
    aligned_rot_w, _, _ = allign_series(df_gesture, 'rot_w')
    
    # Set up figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    subplot_alligned(aligned_rot_x, start_p2, start_p3, axes, 0,
                    "Rotation in x", "Sequence counter", "Acceleration")
    subplot_alligned(aligned_rot_y, start_p2, start_p3, axes, 1,
                    "Rotation in y", "Sequence counter", "Acceleration")
    subplot_alligned(aligned_rot_z, start_p2, start_p3, axes, 2,
                    "Rotation in z", "Sequence counter", "Acceleration")
    subplot_alligned(aligned_rot_w, start_p2, start_p3, axes, 3,
                    "Rotation in w", "Sequence counter", "Acceleration")
    
    fig.suptitle(f"Rotation Data: {gesture}", fontsize=12)
    
    plt.tight_layout()
    plt.show()


for gesture in unique_gestures:

    df_gesture = df_train[df_train.gesture == gesture].reset_index()
    seq_ids = list(df_gesture.sequence_id.unique())
    
    N = 10
    seq_ids_sampled = random.sample(seq_ids, N)
    df_gesture = df_gesture[df_gesture.sequence_id.isin(seq_ids_sampled)]
    
    plot_alligned_rot(df_gesture, gesture)


thm_cols = [col for col in df_train.columns if 'thm' in col]
df_temp = df_train[['sequence_id', 'sequence_counter', 'subject', 'behavior', 'gesture', 'sequence_type'] + thm_cols]
df_temp = df_temp.replace(-1, np.nan)
df_temp['thm_mean'] = df_temp[thm_cols].mean(axis=1)


cmap = sns.color_palette("inferno", 5)

def plot_temperature(group, gesture, ax):

    # Get breakpoints between phases
    t_start_p2 = group.sequence_counter[(group['behavior'] == 'Hand at target location')].iloc[0]
    t_start_p3 = group.sequence_counter[(group['behavior'] == 'Performs gesture')].iloc[0]

    # Plot background
    ax.axvline(t_start_p2, linestyle='--', color=[0.5, 0.5, 0.5])
    ax.axvline(t_start_p3, linestyle='--', color=[0.5, 0.5, 0.5])

    # Plot data
    for i, series in enumerate(['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']):
        ax.plot(group['sequence_counter'], group[series], color=cmap[i], alpha=0.5)
    
    ax.plot(group['sequence_counter'], group['thm_mean'], color='black', linewidth=2)

    # Axes
    ax.set_title(gesture)
    ax.set_xlabel('Time (sequence counter)')
    ax.set_ylabel('Temperature')


fig, axes = plt.subplots(3, 6, figsize=(16, 9))
axes = axes.flatten()

for i, gesture in enumerate(unique_gestures):

    df_gesture = df_temp[df_temp.gesture == gesture].reset_index()
    seq_ids = list(df_gesture.sequence_id.unique())

    seq_ids_sampled = random.sample(seq_ids, 1)
    df_gesture = df_gesture[df_gesture.sequence_id.isin(seq_ids_sampled)]
    
    plot_temperature(df_gesture, gesture, axes[i])

plt.tight_layout()
plt.show()


tof_cols = [col for col in df_train.columns if 'tof' in col]
df_tof = df_train[['sequence_id', 'sequence_counter', 'subject', 'behavior', 'gesture', 'sequence_type'] + tof_cols]
df_tof = df_tof.replace(-1, np.nan)
df_tof['tof_mean'] = df_tof[tof_cols].mean(axis=1)


cmap = sns.color_palette("inferno", 321)

def plot_tof(group, gesture, ax):

    tof_cols = [col for col in group.columns if 'tof' in col]
    
    # Get breakpoints between phases
    t_start_p2 = group.sequence_counter[(group['behavior'] == 'Hand at target location')].iloc[0]
    t_start_p3 = group.sequence_counter[(group['behavior'] == 'Performs gesture')].iloc[0]

    # Plot background
    ax.axvline(t_start_p2, linestyle='--', color=[0.5, 0.5, 0.5])
    ax.axvline(t_start_p3, linestyle='--', color=[0.5, 0.5, 0.5])

    # Plot data
    for i, series in enumerate(tof_cols):
        ax.plot(group['sequence_counter'], group[series], color=cmap[i], alpha=0.5)
    
    ax.plot(group['sequence_counter'], group['tof_mean'], color='black', linewidth=2)

    # Axes
    ax.set_title(gesture)
    ax.set_xlabel('Time (sequence counter)')
    ax.set_ylabel('Time of flight')


random.seed(2025)

fig, axes = plt.subplots(3, 6, figsize=(16, 9))
axes = axes.flatten()

for i, gesture in enumerate(unique_gestures):

    df_gesture = df_tof[df_tof.gesture == gesture].reset_index()
    seq_ids = list(df_gesture.sequence_id.unique())

    seq_ids_sampled = random.sample(seq_ids, 1)
    df_gesture = df_gesture[df_gesture.sequence_id.isin(seq_ids_sampled)]
    
    plot_tof(df_gesture, gesture, axes[i])

plt.tight_layout()
plt.show()


cmap = sns.color_palette("inferno", 321)

def plot_tof_by_sensor(group, gesture):

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    tof_cols_1 = [col for col in group.columns if 'tof_1' in col]
    tof_cols_2 = [col for col in group.columns if 'tof_2' in col]
    tof_cols_3 = [col for col in group.columns if 'tof_3' in col]
    tof_cols_4 = [col for col in group.columns if 'tof_4' in col]
    tof_cols_5 = [col for col in group.columns if 'tof_5' in col]
    
    # Get breakpoints between phases
    t_start_p2 = group.sequence_counter[(group['behavior'] == 'Hand at target location')].iloc[0]
    t_start_p3 = group.sequence_counter[(group['behavior'] == 'Performs gesture')].iloc[0]

    # Plot background
    for i, tof_cols in enumerate([tof_cols_1, tof_cols_2, tof_cols_3, tof_cols_4, tof_cols_5]):
        axes[i].axvline(t_start_p2, linestyle='--', color=[0.5, 0.5, 0.5])
        axes[i].axvline(t_start_p3, linestyle='--', color=[0.5, 0.5, 0.5])
    
        # Plot data
        for j, series in enumerate(tof_cols):
            axes[i].plot(group['sequence_counter'], group[series], color=cmap[(i*64)+j], alpha=0.5)    
        # Axes
        axes[i].set_title(f'Sensor {i+1}')
        axes[i].set_xlabel('Time (sequence counter)')
        axes[i].set_ylabel('Time of flight')
    
    fig.suptitle(f"TOF Data: {gesture}", fontsize=12)
    plt.tight_layout()
    plt.show()


random.seed(2025)


for i, gesture in enumerate(unique_gestures):

    df_gesture = df_tof[df_tof.gesture == gesture].reset_index()
    seq_ids = list(df_gesture.sequence_id.unique())

    seq_ids_sampled = random.sample(seq_ids, 1)
    df_gesture = df_gesture[df_gesture.sequence_id.isin(seq_ids_sampled)]
    
    plot_tof_by_sensor(df_gesture, gesture)


# Clean up the df (only keep mean for thm and tof data)

# Extract col names
tof_cols = [col for col in df_train.columns if 'tof_' in col]
thm_cols = [col for col in df_train.columns if 'thm_' in col]

# replace -1 with NaN
df_train = df_train.replace(-1, np.nan)

# Calculate mean
df_train['mean_tof'] = df_train[tof_cols].mean(axis=1)
df_train['mean_thm'] = df_train[thm_cols].mean(axis=1)

# Drop original columns
df_train_clean = df_train.drop(labels=tof_cols+thm_cols, axis=1)

# Convert to nested format for feature extraction
ts_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w', 'mean_tof', 'mean_thm']

# Convert to nested format using groupby 'id'
X_nested = (
    df_train_clean[ts_cols + ['sequence_id']]
    .groupby('sequence_id', group_keys=False)[ts_cols]
    .apply(lambda group: pd.Series({col: pd.Series(group[col].values) for col in ts_cols}))
)

X_nested.head()


from sktime.transformations.series.summarize import SummaryTransformer

transformer = SummaryTransformer(summary_function=('mean', 'std', 'min', 'max', 'skew', 'kurt', 'mad'), 
                                 quantiles=None)
X_nested_transformed = transformer.fit_transform(X_nested)

X_nested_transformed


df_transformed = df_ref.join(X_nested_transformed, on='sequence_id')
df_transformed['gesture_encoded'] = pd.factorize(df_transformed['gesture'])[0]
df_transformed.head()


ts_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w', 'mean_tof', 'mean_thm']

for col in ts_cols:
    cols = [c for c in df_transformed.columns if col in c]

    fig, axes = plt.subplots(1, 7, figsize=(20, 3))
    for i, c in enumerate(cols):
        sns.boxplot(data=df_transformed, x='gesture_encoded', y=c,
                   ax=axes[i])
        axes[i].set_title(c)
    
    plt.tight_layout()
    plt.show()
        


# Sample the training data to speed-up
def balanced_sample(df, target_col, total_samples, random_state=42):
    classes = df[target_col].unique()
    n_classes = len(classes)
    samples_per_class = total_samples // n_classes
    
    balanced_parts = []
    for cls in classes:
        subset = df[df[target_col] == cls]
        
        if len(subset) < samples_per_class:
            raise ValueError(f"Not enough samples in class '{cls}' to take {samples_per_class}")
        
        balanced_sample = subset.sample(n=samples_per_class, random_state=random_state)
        balanced_parts.append(balanced_sample)
    
    return pd.concat(balanced_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

# Select sequence ids (balance gesture class)
df_seq_id = df_train.groupby("sequence_id").first().reset_index()
df_seq_id_balanced = balanced_sample(df_seq_id, target_col="gesture", total_samples=90)
seq_ids_to_keep = df_seq_id_balanced["sequence_id"]
subjects_to_keep = df_seq_id_balanced["subject"]


# Sample subset for plotting
df_plot = df_train[df_train.sequence_id.isin(seq_ids_to_keep)]

# Drop original columns
df_plot_clean = df_plot.drop(labels=tof_cols+thm_cols, axis=1)

# Convert to nested format for feature extraction
ts_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w', 'mean_tof', 'mean_thm']

# Convert to nested format using groupby 'id'
X_nested = (
    df_plot_clean[ts_cols + ['sequence_id']]
    .groupby('sequence_id', group_keys=False)[ts_cols]
    .apply(lambda group: pd.Series({col: pd.Series(group[col].values) for col in ts_cols}))
)


from sktime.transformations.panel.catch22 import Catch22

class SafeCatch22(Catch22):
    def _transform_case(self, X, f_idx):
        try:
            return super()._transform_case(X, f_idx)
        except Exception:
            # Return array of NaNs if Catch22 fails on a series
            return np.full((1, len(f_idx)), np.nan)

# Use SafeCatch22
catch22 = SafeCatch22()
X_catch22 = catch22.fit_transform(X_nested)

X_catch22


df_catch22 = df_ref.join(X_catch22, on='sequence_id', how='right')
df_catch22['gesture_encoded'] = pd.factorize(df_catch22['gesture'])[0]
df_catch22.head()


for col in ts_cols:
    cols = [c for c in df_catch22.columns if col in c]

    fig, axes = plt.subplots(11, 2, figsize=(20,22))
    axes = axes.flatten()
    
    for i, c in enumerate(cols):
        sns.boxplot(data=df_catch22, x='gesture_encoded', y=c,
                   ax=axes[i])
        axes[i].set_title(c)
    
    plt.tight_layout()
    plt.show()




