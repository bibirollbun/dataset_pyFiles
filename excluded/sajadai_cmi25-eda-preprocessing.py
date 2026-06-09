import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plot style
sns.set() 


DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data"


df_dmgc = pd.read_csv(f"{DIR}/train_demographics.csv")
df_dmgc.head()


df_dmgc.shape


df_dmgc.info()


df_dmgc['adult_child'] = df_dmgc['adult_child'].astype('object')
df_dmgc['sex'] = df_dmgc['sex'].astype('object')
df_dmgc['handedness'] = df_dmgc['handedness'].astype('object')


df_dmgc.info()


df_dmgc.describe(include='all')


df_dmgc['adult_child'] = df_dmgc['adult_child'].astype('int64')
df_dmgc['sex'] = df_dmgc['sex'].astype('int64')
df_dmgc['handedness'] = df_dmgc['handedness'].astype('int64')


# Select only numeric columns
numeric_cols = df_dmgc.select_dtypes(include='number').columns

# Set number of rows and columns for subplots
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

# Plot histogram for each numeric column
for i, col in enumerate(numeric_cols):
    axes[i].hist(df_dmgc[col], bins=5, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Remove unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


df_train = pd.read_csv(f"{DIR}/train.csv")
df_train.head()


print(f"ğŸ”¹ Number of Rows: {df_train.shape[0]} \nğŸ”¹ Number of Cols: {df_train.shape[1]}")


df_train.describe(include='all')


df_train.behavior.value_counts()


# Count the number of unique sequences per gesture
gesture_counts = df_train[['gesture', 'sequence_id']].drop_duplicates().gesture.value_counts()

plt.figure(figsize=(12, 6))
gesture_counts.plot(kind='bar')
plt.title("Gesture Class Distribution")
plt.xlabel("Gesture")
plt.ylabel("Number of Unique Sequences")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()


# Count the number of unique sequences per subject
subject_counts = df_train[['subject', 'sequence_id']].drop_duplicates().subject.value_counts()

plt.figure(figsize=(20, 6))
subject_counts.sort_index().plot(kind='bar')
plt.title("Number of Sequences per Subject")
plt.xlabel("Subject")
plt.ylabel("Number of Unique Sequences")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()


18*4


# Choose the subject ID you want to analyze
subject_id = "SUBJ_059520"  # â†� Replace with any other subject

# Filter the DataFrame for this subject
df_subj = df_train[df_train['subject'] == subject_id]

# Drop duplicates to get one row per sequence
unique_sequences = df_subj[['gesture', 'orientation', 'sequence_id']].drop_duplicates()

# Count number of unique sequences per (gesture, orientation)
gesture_orientation_seq_counts = unique_sequences.groupby(['gesture', 'orientation']).size().unstack(fill_value=0)

# Display as table
gesture_orientation_seq_counts


gesture_orientation_seq_counts.sum()#.sum()


# Count the number of rows in each sequence (i.e., sequence length)
sequence_lengths = df_train.groupby('sequence_id').size()

# Count how many sequences have each unique length
length_counts = sequence_lengths.value_counts().sort_index()

plt.figure(figsize=(20, 5))
length_counts.plot(kind='bar')
plt.title("Distribution of Sequence Lengths")
plt.xlabel("Sequence Length (Number of Rows)")
plt.ylabel("Number of Sequences")
plt.grid(True)
plt.tight_layout()
plt.show()


length_counts.sort_values(ascending=False)


# Identify sensor columns (acc, rot, thm, tof)
sensor_cols = [col for col in df_train.columns if (
    col.startswith('acc_') or col.startswith('rot_') or 
    col.startswith('thm_') or col.startswith('tof_')
)]

# Count number of NaNs per sensor column
missing_values = df_train[sensor_cols].isna().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

missing_values[['rot_x', 'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5', 'tof_1_v0', 'tof_2_v0', 'tof_3_v0', 'tof_4_v0', 'tof_5_v0']]


# Extract only TOF columns
tof_cols = [col for col in df_train.columns if col.startswith('tof_')]

# Count number of -1 values per TOF column
tof_negative_ones = (df_train[tof_cols] == -1).sum()

# Show only columns with at least one -1
tof_negative_ones = tof_negative_ones[tof_negative_ones > 0]

tof_negative_ones


sequence_id = "SEQ_000007"
df_seq = df_train[df_train["sequence_id"] == sequence_id]

plt.plot(df_seq["acc_x"])
plt.plot(df_seq["acc_y"])
plt.plot(df_seq["acc_z"])


# Choose a specific sequence_id to analyze
sequence_id = "SEQ_000007"
sequence_id = random.choice(df_train["sequence_id"].unique())
df_seq = df_train[df_train["sequence_id"] == sequence_id]

# Plot accelerometer signals over time (sequence_counter)
plt.figure(figsize=(20, 10))
plt.plot(df_seq["sequence_counter"], df_seq["acc_x"], label="acc_x")
plt.plot(df_seq["sequence_counter"], df_seq["acc_y"], label="acc_y")
plt.plot(df_seq["sequence_counter"], df_seq["acc_z"], label="acc_z")

plt.title(f"Accelerometer data over time for sequence {sequence_id}")
plt.xlabel("Sequence Counter")
plt.ylabel("Acceleration (m/sÂ²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Choose a specific sequence_id
sequence_id = random.choice(df_train["sequence_id"].unique())
df_seq = df_train[df_train["sequence_id"] == sequence_id]

# Extract gesture and orientation info
gesture = df_seq["gesture"].iloc[0]
orientation = df_seq["orientation"].iloc[0]

# Find end index of the 'Transition' phase
transition_mask = df_seq["phase"] == "Transition"
if transition_mask.any():
    transition_end_idx = df_seq[transition_mask]["sequence_counter"].max()
else:
    transition_end_idx = None  # Fallback in case 'Transition' phase is missing

# Define accelerometer columns
acc_cols = ["acc_x", "acc_y", "acc_z"]

# Create subplots
fig, axs = plt.subplots(len(acc_cols), 1, figsize=(20, 10), sharex=True)

for i, col in enumerate(acc_cols):
    axs[i].plot(df_seq["sequence_counter"], df_seq[col], label=col)

    # Mark transition boundary with vertical dashed line
    if transition_end_idx is not None:
        axs[i].axvline(x=transition_end_idx, color='red', linestyle='--',
                       label='Transition â†’ Gesture' if i == 0 else "")
    
    axs[i].set_ylabel(col)
    axs[i].legend(loc='upper right')

axs[-1].set_xlabel("Sequence Counter")

# Main title with orientation included
fig.suptitle(
    f"Sensor: Accelerometer | Gesture: {gesture} | Orientation: {orientation} "
    f"{'| Transition â†’ Gesture @ ' + str(transition_end_idx) if transition_end_idx is not None else ''}",
    fontsize=16
)

plt.tight_layout()
plt.show()


# Choose a specific sequence_id
sequence_id = "SEQ_000007"
df_seq = df_train[df_train["sequence_id"] == sequence_id]

# Extract gesture and orientation info
gesture = df_seq["gesture"].iloc[0]
orientation = df_seq["orientation"].iloc[0]

# Define accelerometer columns
acc_cols = ["acc_x", "acc_y", "acc_z"]

# Create subplots
fig, axs = plt.subplots(len(acc_cols), 1, figsize=(20, 10), sharex=True)

for i, col in enumerate(acc_cols):
    axs[i].plot(df_seq["sequence_counter"], df_seq[col], label=col)
    
    # Highlight behavior regions
    for behavior in df_seq["behavior"].unique():
        mask = df_seq["behavior"] == behavior
        xmin = df_seq[mask]["sequence_counter"].min()
        xmax = df_seq[mask]["sequence_counter"].max()
        xmid = (xmin + xmax) / 2

        # Add shaded region
        axs[i].axvspan(xmin, xmax, color='gray', alpha=0.1)

        # Add behavior label only on top subplot
        if i == 0:
            ymax = df_seq[col].max()
            axs[i].text(xmid, ymax + 0.5, behavior, ha='center', va='bottom', fontsize=10, color='black')

    axs[i].set_ylabel(col)
    axs[i].legend(loc='upper right')

axs[-1].set_xlabel("Sequence Counter")

# Main title with orientation and gesture
fig.suptitle(
    f"Sensor: Accelerometer | Gesture: {gesture} | Orientation: {orientation}",
    fontsize=16
)

plt.tight_layout()
plt.show()


# Choose a specific sequence_id
sequence_id = "SEQ_000007"
df_seq = df_train[df_train["sequence_id"] == sequence_id]

# Extract gesture and orientation info
gesture = df_seq["gesture"].iloc[0]
orientation = df_seq["orientation"].iloc[0]

# Define rotation columns
rot_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]

# Create subplots for each rotation component
fig, axs = plt.subplots(len(rot_cols), 1, figsize=(20, 12), sharex=True)

for i, col in enumerate(rot_cols):
    axs[i].plot(df_seq["sequence_counter"], df_seq[col], label=col)
    
    # Highlight behavior regions with shading and label on top subplot
    for behavior in df_seq["behavior"].unique():
        mask = df_seq["behavior"] == behavior
        xmin = df_seq[mask]["sequence_counter"].min()
        xmax = df_seq[mask]["sequence_counter"].max()
        xmid = (xmin + xmax) / 2

        axs[i].axvspan(xmin, xmax, color='gray', alpha=0.1)
        
        if i == 0:
            ymax = df_seq[col].max()
            axs[i].text(xmid, ymax + 0.02, behavior, ha='center', va='bottom', fontsize=10, color='black')

    axs[i].set_ylabel(col)
    axs[i].legend(loc='upper right')

axs[-1].set_xlabel("Sequence Counter")

# Main title with context
fig.suptitle(
    f"Sensor: Rotation | Gesture: {gesture} | Orientation: {orientation}",
    fontsize=16
)

plt.tight_layout()
plt.show()


from scipy.spatial.transform import Rotation as R

# Choose a specific sequence_id
sequence_id = "SEQ_000007"
# sequence_id = random.choice(df_train["sequence_id"].unique())
df_seq = df_train[df_train["sequence_id"] == sequence_id]

# Extract gesture and orientation info
gesture = df_seq["gesture"].iloc[0]
orientation = df_seq["orientation"].iloc[0]

# Convert quaternion to Euler angles (roll, pitch, yaw) in degrees
quats = df_seq[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
euler = R.from_quat(quats).as_euler('xyz', degrees=True)  # 'xyz' = roll, pitch, yaw

# Create subplots for Roll, Pitch, Yaw
fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']

for i in range(3):
    axs[i].plot(df_seq["sequence_counter"], euler[:, i], label=labels[i], color='tab:blue')
    
    # Highlight behavior regions
    for behavior in df_seq["behavior"].unique():
        mask = df_seq["behavior"] == behavior
        xmin = df_seq[mask]["sequence_counter"].min()
        xmax = df_seq[mask]["sequence_counter"].max()
        xmid = (xmin + xmax) / 2
        axs[i].axvspan(xmin, xmax, color='gray', alpha=0.1)
        if i == 0:
            ymax = euler[:, i].max()
            axs[i].text(xmid, ymax + 7, behavior, ha='center', va='bottom', fontsize=10)

    axs[i].set_ylabel(labels[i])
    axs[i].legend(loc='upper right')

axs[-1].set_xlabel("Sequence Counter")

# Main title with context
fig.suptitle(
    f"Sensor: Orientation (Euler angles) | Gesture: {gesture} | Orientation: {orientation}",
    fontsize=15
)
plt.tight_layout()
plt.show()


def plot_sequence_signals(df_seq, signal_cols, title_prefix="Sensor Signals"):
    """
    Plot multiple time-series sensor signals from a single sequence with behavior overlays.

    Parameters:
    - df_seq: filtered DataFrame for one sequence_id
    - signal_cols: list of column names (e.g., ["acc_x", "acc_y", "acc_z"])
    - title_prefix: custom string for title (e.g., "Rotation" or "ToF")
    """
    gesture = df_seq["gesture"].iloc[0] if "gesture" in df_seq else "Unknown Gesture"
    orientation = df_seq["orientation"].iloc[0] if "orientation" in df_seq else "Unknown Orientation"
    
    fig, axs = plt.subplots(len(signal_cols), 1, figsize=(20, 2.5 * len(signal_cols)), sharex=True)

    # Handle single subplot case
    if len(signal_cols) == 1:
        axs = [axs]

    for i, col in enumerate(signal_cols):
        axs[i].plot(df_seq["sequence_counter"], df_seq[col], label=col, color='tab:blue')

        # Shade behavior regions
        for behavior in df_seq["behavior"].unique():
            mask = df_seq["behavior"] == behavior
            xmin = df_seq[mask]["sequence_counter"].min()
            xmax = df_seq[mask]["sequence_counter"].max()
            xmid = (xmin + xmax) / 2
            axs[i].axvspan(xmin, xmax, color='gray', alpha=0.1)

            if i == 0:
                ymax = df_seq[col].max()
                axs[i].text(xmid, ymax, behavior, ha='center', va='bottom', fontsize=11, color='black')

        axs[i].set_ylabel(col)
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel("Sequence Counter")

    fig.suptitle(f"{title_prefix} | Gesture: {gesture} | Orientation: {orientation}", fontsize=14)
    plt.tight_layout()
    plt.show()


# Choose a specific sequence_id
sequence_id = random.choice(df_train["sequence_id"].unique())
df_seq = df_train[df_train["sequence_id"] == sequence_id]


plot_sequence_signals(df_seq, ["acc_x", "acc_y", "acc_z"], title_prefix="Accelerometer")

plot_sequence_signals(df_seq, ["rot_x", "rot_y", "rot_z", "rot_w"], title_prefix="Rotation (Quaternion)")

plot_sequence_signals(df_seq, ["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"], title_prefix="Thermal Sensors")

pixel_index = random.randint(0, 63)
tof_cols = [f"tof_{s}_v{pixel_index}" for s in range(1, 6)]
plot_sequence_signals(df_seq, tof_cols, title_prefix="ToF Sensors (partial)")


# Randomly select a gesture from available gestures
gesture = random.choice(df_train["gesture"].dropna().unique())

# Randomly select an orientation from available orientations
orientation = random.choice(df_train["orientation"].dropna().unique())

# Filter the dataframe for rows with the same randomly selected gesture and orientation
df_filtered = df_train[(df_train["gesture"] == gesture) & (df_train["orientation"] == orientation)]

# Get the first two distinct subjects that performed this gesture in this orientation
subjects = df_filtered["subject"].unique()

# If fewer than two subjects exist for that combination, print a warning and exit
if len(subjects) < 2:
    print(f"âš ï¸� Not enough subjects found for gesture '{gesture}' and orientation '{orientation}'.")
else:
    subj1, subj2 = random.sample(list(subjects), 2)  # Pick 2 distinct subjects at random

    # Select one sequence from each subject
    seq1 = df_filtered[df_filtered["subject"] == subj1]["sequence_id"].unique()[0]
    seq2 = df_filtered[df_filtered["subject"] == subj2]["sequence_id"].unique()[0]

    # Filter the dataframe for each selected sequence
    df_seq1 = df_train[df_train["sequence_id"] == seq1]
    df_seq2 = df_train[df_train["sequence_id"] == seq2]

    # Choose the signal columns to compare (can be any sensor)
    signal_cols = ["acc_x", "acc_y", "acc_z"]

    # Plot both sequences using the same function
    plot_sequence_signals(df_seq1, signal_cols, title_prefix=f"Subject: {subj1} | {gesture} | {orientation}")
    plot_sequence_signals(df_seq2, signal_cols, title_prefix=f"Subject: {subj2} | {gesture} | {orientation}")


# Randomly select a gesture
gesture = random.choice(df_train["gesture"].dropna().unique())

# Filter rows with that gesture
df_gesture = df_train[df_train["gesture"] == gesture]

# Pick a subject that has done this gesture in at least two orientations
subject_counts = df_gesture.groupby("subject")["orientation"].nunique()
valid_subjects = subject_counts[subject_counts >= 2].index.tolist()

if not valid_subjects:
    print(f"âš ï¸� No subject has performed the gesture '{gesture}' in multiple orientations.")
else:
    # Select one valid subject randomly
    subject = random.choice(valid_subjects)

    # Get all orientations this subject performed this gesture in
    orientations = df_gesture[df_gesture["subject"] == subject]["orientation"].unique()
    orientation1, orientation2 = random.sample(list(orientations), 2)

    # Pick one sequence per orientation
    seq1 = df_gesture[(df_gesture["subject"] == subject) & (df_gesture["orientation"] == orientation1)]["sequence_id"].unique()[0]
    seq2 = df_gesture[(df_gesture["subject"] == subject) & (df_gesture["orientation"] == orientation2)]["sequence_id"].unique()[0]

    # Extract sequences
    df_seq1 = df_train[df_train["sequence_id"] == seq1]
    df_seq2 = df_train[df_train["sequence_id"] == seq2]

    # Choose signals to visualize
    signal_cols = ["acc_x", "acc_y", "acc_z"]
    signal_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]

    # Plot each sequence using the same function
    plot_sequence_signals(df_seq1, signal_cols, title_prefix=f"Subject: {subject} | Orientation: {orientation1} | Gesture: {gesture}")
    plot_sequence_signals(df_seq2, signal_cols, title_prefix=f"Subject: {subject} | Orientation: {orientation2} | Gesture: {gesture}")


def plot_tof_grid(df_seq, row_index, sensor_id=1):
    """
    Plot 8x8 ToF sensor grid for a given row and sensor.

    Parameters:
    - df_seq: filtered DataFrame for a specific sequence
    - row_index: index within df_seq (not global df_train)
    - sensor_id: from 1 to 5
    """
    # Generate column names for this sensor
    tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(64)]
    
    # Convert to float explicitly to avoid type issues
    values = df_seq.iloc[row_index][tof_cols].astype(float).to_numpy()
    grid = values.reshape((8, 8)) / 254.
    
    # Replace -1 (no reflection) with np.nan to make heatmap clearer
    grid[grid == -1] = np.nan
    
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="ToF Value")
    plt.title(f"ToF Sensor {sensor_id} | Frame {row_index} | Sequence: {df_seq['sequence_id'].iloc[0]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_tof_grid(df_seq, row_index=10, sensor_id=3)


df_train


df_train.iloc[:, 8:].isna().sum()


df_train.describe()


df_train[df_train['tof_5_v1'] > -1].tof_5_v1.median()


# Define feature groups
acc_cols = [col for col in df_train.columns if col.startswith("acc_")]
rot_cols = [col for col in df_train.columns if col.startswith("rot_")]
thm_cols = [col for col in df_train.columns if col.startswith("thm_")]
tof_cols = [f"tof_{i}_v{0}" for i in range(1, 6)]


def plot_combined_box_violin(df, columns, title):
    data = [df[col].dropna().values for col in columns]

    fig, ax = plt.subplots(figsize=(20, 10))

    # Horizontal boxplot on top of violin plot
    ax.boxplot(data, notch=False, patch_artist=True, widths=0.1, vert=False)

    # Horizontal violin plot
    ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, vert=False)

    # Axis settings
    ax.set_title(title)
    ax.set_yticks(range(1, len(columns) + 1))
    ax.set_yticklabels(columns)
    ax.grid(True, axis='x')  # Grid on x-axis since horizontal plot

    plt.tight_layout()
    plt.show()


# Run plots
plot_combined_box_violin(df_train, acc_cols, "Accelerometer Sensors")
plot_combined_box_violin(df_train, rot_cols, "Rotation Sensors")
plot_combined_box_violin(df_train, thm_cols, "Thermal Sensors")
plot_combined_box_violin(df_train, tof_cols, "ToF Sensors (first pixel per group)")


# Find outliers in thm_1
outliers_thm1 = df_train[df_train['thm_1'] < 10]
display(outliers_thm1[thm_cols])

# Find outliers in thm_3
outliers_thm3 = df_train[df_train['thm_3'] == 0]
display(outliers_thm3[thm_cols])


thm_target = 'thm_3'

# Identify all sequence_ids where thm_3 is zero in at least one row
sequences_with_zero_thm3 = df_train[df_train[thm_target] == 0]['sequence_id'].unique()

# Create a dataframe only for those sequences
subset = df_train[df_train['sequence_id'].isin(sequences_with_zero_thm3)]

# For each sequence, check if *all* thm_3 values are zero or just some
sequence_zero_stats = subset.groupby('sequence_id')[thm_target].agg(
    total_frames='count',
    zero_count=lambda x: (x == 0).sum()
)

# Step 4: Add a column showing if the entire sequence is zero or partial
sequence_zero_stats['zero_type'] = sequence_zero_stats.apply(
    lambda row: 'All Zero' if row['total_frames'] == row['zero_count'] else 'Partial Zero',
    axis=1
)

# Display results
print(sequence_zero_stats['zero_type'].value_counts())  # Count how many are "All Zero" vs "Partial Zero"
display(sequence_zero_stats)


thm_target = 'thm_1'

# Identify all sequence_ids where thm_3 is zero in at least one row
sequences_with_zero_thm3 = df_train[df_train[thm_target] < 10]['sequence_id'].unique()

# Create a dataframe only for those sequences
subset = df_train[df_train['sequence_id'].isin(sequences_with_zero_thm3)]

# For each sequence, check if *all* thm_3 values are zero or just some
sequence_zero_stats = subset.groupby('sequence_id')[thm_target].agg(
    total_frames='count',
    zero_count=lambda x: (x < 10).sum()
)

# Step 4: Add a column showing if the entire sequence is zero or partial
sequence_zero_stats['zero_type'] = sequence_zero_stats.apply(
    lambda row: 'All Zero' if row['total_frames'] == row['zero_count'] else 'Partial Zero',
    axis=1
)

# Display results
print(sequence_zero_stats['zero_type'].value_counts())  # Count how many are "All Zero" vs "Partial Zero"
display(sequence_zero_stats)


def impute_low_thermal_values(df, thm_columns, threshold=10.0):
    # Create a mask of low values
    low_mask = df[thm_columns] < threshold

    # For each row, replace low values with the mean of other valid (>= threshold) thermal values
    for col in thm_columns:
        low_rows = low_mask[col]

        # Compute row-wise means excluding the low value itself
        row_means = df.loc[low_rows, thm_columns].mask(low_mask, np.nan).mean(axis=1)

        # Impute
        df.loc[low_rows, col] = row_means

    return df


df_train = impute_low_thermal_values(df_train, thm_cols, threshold=10.0)


plot_combined_box_violin(df_train, thm_cols, "Thermal Sensors")


df_train.iloc[:, 9:].isna().sum()


# Total number of rows in the dataset
total_rows = len(df_train)

# Define sensor groups (using one column per ToF sensor)
sensor_groups = {
    "Accelerometer": ["acc_x", "acc_y", "acc_z"],
    "Rotation": ["rot_w", "rot_x", "rot_y", "rot_z"],
    "Thermal thm_1": ["thm_1"],
    "Thermal thm_2": ["thm_2"],
    "Thermal thm_3": ["thm_3"],
    "Thermal thm_4": ["thm_4"],
    "Thermal thm_5": ["thm_5"],
    "ToF 1 (64 cols)": ["tof_1_v0"],
    "ToF 2 (64 cols)": ["tof_2_v0"],
    "ToF 3 (64 cols)": ["tof_3_v0"],
    "ToF 4 (64 cols)": ["tof_4_v0"],
    "ToF 5 (64 cols)": ["tof_5_v0"],
}

# Initialize list to collect results
missing_data = []

# Loop over each sensor group and calculate missing values
for sensor_name, columns in sensor_groups.items():
    # Count missing values from the representative column(s)
    total_missing = df_train[columns].isna().sum().sum()
    # Compute missing percentage
    missing_percentage = (total_missing / total_rows) * 100
    # Append result
    missing_data.append({
        "Sensor Type": sensor_name,
        "Missing Count": int(total_missing),
        "Missing Percentage": f"{missing_percentage:.2f}%"
    })

# Create summary DataFrame
df_missing_summary = pd.DataFrame(missing_data)

# Display result
df_missing_summary


# Sensor groups
acc_cols = ["acc_x", "acc_y", "acc_z"]
rot_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
tof_sensors = [f"tof_{i}_" for i in range(1, 6)]
thm_sensors = [f"thm_{i}" for i in range(1, 6)]

# Result storage
sensor_results = []


# Function to analyze missingness for a sensor group
def analyze_sensor_missingness(df, feature_cols, sensor_name):
    nan_stats = (
        df.groupby("sequence_id")[feature_cols]
        .apply(lambda x: x.isna().mean())
        .reset_index()
    )

    nan_stats["max_nan_ratio"] = nan_stats[feature_cols].max(axis=1)
    nan_stats["min_nan_ratio"] = nan_stats[feature_cols].min(axis=1)

    sequences_with_nan = nan_stats[nan_stats["max_nan_ratio"] > 0]
    fully_missing = nan_stats[nan_stats["max_nan_ratio"] == 1.0]
    partially_missing = nan_stats[
        (nan_stats["max_nan_ratio"] > 0) & (nan_stats["max_nan_ratio"] < 1.0)
    ]
    nan_stats["nan_overlap"] = nan_stats["max_nan_ratio"] == nan_stats["min_nan_ratio"]
    overlapping = nan_stats[
        (nan_stats["max_nan_ratio"] > 0) & (nan_stats["nan_overlap"] == True)
    ]

    sensor_results.append({
        "Sensor": sensor_name,
        "Any Missing": len(sequences_with_nan),
        "Fully Missing": len(fully_missing),
        "Partially Missing": len(partially_missing),
        "Overlap Missing": len(overlapping),
        "Independent Missing": len(sequences_with_nan) - len(overlapping)
    })


# Analyze accelerometer
analyze_sensor_missingness(df_train, acc_cols, "acc")

# Analyze rotation
analyze_sensor_missingness(df_train, rot_cols, "rot")

# Analyze thermal sensors
for thm in thm_sensors:
    analyze_sensor_missingness(df_train, [thm], thm)

# Analyze ToF sensors
for i, base in enumerate(tof_sensors, 1):
    tof_cols = [f"{base}v{j}" for j in range(64)]
    analyze_sensor_missingness(df_train, tof_cols, f"tof_{i}")

# Create and display result table
df_sensor_missing = pd.DataFrame(sensor_results)
df_sensor_missing


# Define sensor groups
acc_cols = ["acc_x", "acc_y", "acc_z"]
rot_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
thm_sensors = [f"thm_{i}" for i in range(1, 6)]
tof_sensors = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]

# Organize sensor groups into a dictionary
sensor_groups = {
    "acc": acc_cols,
    "rot": rot_cols,
    "thm": thm_sensors,
    "tof": tof_sensors
}

# Compute NaN ratio per column for each sequence
nan_ratio_per_seq = df_train.groupby("sequence_id").apply(lambda x: x.isna().mean())

# For each sequence, count how many columns (sensors) in each group have missing values
sequence_missing_counts = pd.DataFrame(index=nan_ratio_per_seq.index)

for group_name, group_cols in sensor_groups.items():
    group_nan = nan_ratio_per_seq[group_cols]
    # Count how many columns in the group have any NaNs (ratio > 0)
    sequence_missing_counts[f"{group_name}_missing_sensors"] = (group_nan > 0).sum(axis=1)

# Summarize: For each group, how many sequences have N sensors missing (N = 1, 2, ..., 320)
summary = {}
for group in ["thm", "tof"]:
    counts = sequence_missing_counts[f"{group}_missing_sensors"].value_counts().sort_index()
    summary[group] = counts

# Convert summary into a DataFrame
summary_df = pd.DataFrame(summary).fillna(0).astype(int)
summary_df.index.name = "Num sensors with NaN"
summary_df


# Copy the sequence-level missing sensor counts
combo_df = sequence_missing_counts.copy()

# Convert counts to boolean: True if any sensor in the group is missing
combo_df_binary = combo_df > 0

# Analyze sensor group combinations with missing values
combo_df_binary["thm_and_rot"] = combo_df_binary["thm_missing_sensors"] & combo_df_binary["rot_missing_sensors"]
combo_df_binary["thm_and_tof"] = combo_df_binary["thm_missing_sensors"] & combo_df_binary["tof_missing_sensors"]
combo_df_binary["rot_and_tof"] = combo_df_binary["rot_missing_sensors"] & combo_df_binary["tof_missing_sensors"]

# All three sensor groups missing
combo_df_binary["all_three"] = (
    combo_df_binary["thm_missing_sensors"]
    & combo_df_binary["rot_missing_sensors"]
    & combo_df_binary["tof_missing_sensors"]
)

# Count how many sequences fall into each combination
combo_stats = combo_df_binary[["thm_and_rot", "thm_and_tof", "rot_and_tof", "all_three"]].sum()
combo_stats



# Get all ToF columns (those that start with 'tof_')
tof_columns = [col for col in df_train.columns if col.startswith("tof_")]

# Total number of rows
total_rows = len(df_train)

# Create list to store results
tof_neg1_stats = []

# Loop through each ToF column and count -1 values
for col in tof_columns:
    count_neg1 = (df_train[col] == -1).sum()
    percent_neg1 = (count_neg1 / total_rows) * 100
    tof_neg1_stats.append({
        "ToF Column": col,
        "Count of -1": int(count_neg1),
        "Percentage of -1": f"{percent_neg1:.2f}%"
    })

# Convert list to DataFrame
df_neg1_summary = pd.DataFrame(tof_neg1_stats)

# Display summary
df_neg1_summary


# df_seq.ffill().bfill().fillna(0)


df_train.describe()


from sklearn.base import BaseEstimator, TransformerMixin


class SensorImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='mean'):
        self.method = method  # 'mean' or 'median'
        self.acc_columns = []
        self.rot_columns = []
        self.thm_columns = []
        self.tof_columns = []
        self.column_stats = {}

    def fit(self, X, y=None):
        # Identify sensor columns
        self.acc_columns = [col for col in X.columns if col.startswith("acc_")]
        self.rot_columns = [col for col in X.columns if col.startswith("rot_")]
        self.thm_columns = [col for col in X.columns if col.startswith("thm_")]
        self.tof_columns = [col for col in X.columns if col.startswith("tof_")]

        # Calculate statistics for acc, rot, and thm columns
        sensor_cols = self.acc_columns + self.rot_columns + self.thm_columns
        if self.method == 'mean':
            self.column_stats.update({col: X[col].mean() for col in sensor_cols})
        elif self.method == 'median':
            self.column_stats.update({col: X[col].median() for col in sensor_cols})
        else:
            raise ValueError("Method must be either 'mean' or 'median'")

        # Calculate statistics for tof columns (ignoring -1 values)
        for col in self.tof_columns:
            valid_values = X[col][X[col] != -1]
            if self.method == 'mean':
                self.column_stats[col] = valid_values.mean()
            elif self.method == 'median':
                self.column_stats[col] = valid_values.median()

        return self

    def transform(self, X):
        X = X.copy()

        # Fill missing values in acc, rot, and thm columns using precomputed stats
        for col in self.acc_columns + self.rot_columns + self.thm_columns:
            X[col] = X[col].fillna(self.column_stats[col])

        # Replace -1 with NaN in tof columns, then fill with precomputed stats
        for col in self.tof_columns:
            X[col] = X[col].replace(-1, np.nan)
            X[col] = X[col].fillna(self.column_stats[col])  # 255

        return X


imputer = SensorImputer(method='median')
imputer


df_train_imputed = imputer.fit_transform(df_train)


df_train_imputed.isna().sum()


plot_combined_box_violin(df_train, acc_cols, "Accelerometer Sensors")

plot_combined_box_violin(df_train, rot_cols, "Rotation Sensors")

plot_combined_box_violin(df_train, thm_cols, "Thermal Sensors")

tof_cols = [f"tof_{i}_v{0}" for i in range(1, 6)]
plot_combined_box_violin(df_train, tof_cols, "ToF Sensors (first pixel per group)")


df_train.describe()


from sklearn.preprocessing import StandardScaler

# Select sensor columns (excluding rotation)
feature_cols = [col for col in df_train.columns if col.startswith(("acc_", "thm_", "tof_"))]

# Initialize scaler
scaler = StandardScaler()

# Apply Z-score normalization
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])

df_train.describe()


df_train_imputed.describe()


# from sklearn.preprocessing import StandardScaler

# Select sensor columns (excluding rotation)
feature_cols = [col for col in df_train.columns if col.startswith(("acc_", "thm_", "tof_"))]

# Initialize scaler
scaler = StandardScaler()

# Apply Z-score normalization
df_train_imputed[feature_cols] = scaler.fit_transform(df_train_imputed[feature_cols])

df_train_imputed.describe()


df_train_imputed.head()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train_imputed['gesture'] = le.fit_transform(df_train_imputed['gesture'])
df_train_imputed.head()


# df_train_imputed.to_csv("train-preprocessed.csv", index=False)
df_train_imputed.sample(100).to_csv("train-preprocessed.csv", index=False)


# /kaggle/input/cmi25-train-preprocessed/train-preprocessed.csv

