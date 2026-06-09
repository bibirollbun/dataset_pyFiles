import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
print(f"Train rows: {len(train_df):,}  Test rows: {len(test_df)}")


train_dem_df.info()


train_dem_df.isnull().sum().sum(), train_dem_df.duplicated().sum()


train_dem_df.head()


train_dem_df.shape


print('unique subject values:')
print(train_dem_df['subject'].unique()), print(f"number of unique subject values: {len(train_dem_df['subject'].unique())}")
print()
print('unique adult_child values:')
print(train_dem_df['adult_child'].unique()), print(f"number of unique adult_child values: {len(train_dem_df['adult_child'].unique())}")
print()
print('unique age values:')
print(train_dem_df['age'].unique()), print(f"number of unique age values: {len(train_dem_df['age'].unique())}")
print()
print('unique sex values:')
print(train_dem_df['sex'].unique()), print(f"number of unique sex values: {len(train_dem_df['sex'].unique())}")
print()
print('unique handedness values:')
print(train_dem_df['handedness'].unique()), print(f"number of unique handedness values: {len(train_dem_df['handedness'].unique())}")
print()
print('unique height_cm values:')
print(train_dem_df['height_cm'].unique()), print(f"number of unique height_cm values: {len(train_dem_df['height_cm'].unique())}")
print()
print('unique shoulder_to_wrist_cm values:')
print(train_dem_df['shoulder_to_wrist_cm'].unique()), print(f"number of unique shoulder_to_wrist_cm values: {len(train_dem_df['shoulder_to_wrist_cm'].unique())}")
print()
print('unique elbow_to_wrist_cm values:')
print(train_dem_df['elbow_to_wrist_cm'].unique()), print(f"number of unique elbow_to_wrist_cm values: {len(train_dem_df['elbow_to_wrist_cm'].unique())}");


train_dem_df


cat_cols = ['sex', 'adult_child', 'handedness']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    counts = train_dem_df[col].value_counts().sort_index()

    if col == 'sex':
        labels = ['female' if i == 0 else 'male' for i in counts.index]
    elif col == 'adult_child':
        labels = ['child' if i == 0 else 'adult' for i in counts.index]
    elif col == 'handedness':
        labels = ['left-handed' if i == 0 else 'right-handed' for i in counts.index]

    plot_df = counts.reset_index()
    plot_df.columns = [col, 'count']
    plot_df[col] = labels

    sns.barplot(data=plot_df, x=col, y='count', palette='Set2', ax=axes[i])
    axes[i].set_title(f'Count of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].grid(True, axis='y')

# fig.delaxes(axes[3])
plt.tight_layout()
plt.show()


numeric_cols = ['height_cm', 'age', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(train_dem_df[col], bins=20, kde=True, color='skyblue', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=train_dem_df, x='sex', y='height_cm', ax=axes[0])
axes[0].set_title('Height by Sex')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['female', 'male'])

sns.boxplot(data=train_dem_df, x='adult_child', y='age', ax=axes[1])
axes[1].set_title('Age by Adult/Child')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['child', 'adult'])

plt.tight_layout()
plt.show()


ax = sns.scatterplot(data=train_dem_df, x='age', y='height_cm', hue='sex')
plt.title('Age vs Height')

handles, labels = ax.get_legend_handles_labels()
new_labels = ['female', 'male']
ax.legend(handles=handles, labels=new_labels, title='sex')
plt.show()


numeric_cols = ['height_cm', 'age', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(x=train_dem_df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
    
plt.tight_layout()
plt.show()


train_dem_df.groupby('sex')['height_cm'].mean()


train_dem_df.groupby('sex')['age'].mean()


train_dem_df.groupby('adult_child')['height_cm'].mean()


train_dem_df.groupby('adult_child')['age'].mean()


corr = train_dem_df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


train_df.info()
train_df.describe()


train_temp = train_df[['row_id', 'sequence_type', 'sequence_id', 'sequence_counter',
                       'subject', 'orientation', 'behavior', 'phase', 'gesture']]
train_temp


train_temp.isna().any().any()


train_df.head()


print(train_df.columns)


print('unique row_id values:')
print(train_df['row_id'].unique()), print(f"number of unique row_id values: {len(train_df['row_id'].unique())}")
print()
print('unique sequence_type values:')
print(train_df['sequence_type'].unique()), print(f"number of unique sequence_type values: {len(train_df['sequence_type'].unique())}")
print()
print('unique sequence_id values:')
print(train_df['sequence_id'].unique()), print(f"number of unique sequence_id values: {len(train_df['sequence_id'].unique())}")
print()
print('unique orientation values:')
print(train_df['orientation'].unique()), print(f"number of unique orientation values: {len(train_df['orientation'].unique())}")
print()
print('unique behavior values:')
print(train_df['behavior'].unique()), print(f"number of unique behavior values: {len(train_df['behavior'].unique())}")
print()
print('unique phase values:')
print(train_df['phase'].unique()), print(f"number of unique phase values: {len(train_df['phase'].unique())}")
print()
print('unique gesture values:')
print(train_df['gesture'].unique()), print(f"number of unique gesture values: {len(train_df['gesture'].unique())}")
print()
print('unique subject values:')
print(train_df['subject'].unique()), print(f"number of unique subject values: {len(train_df['subject'].unique())}")
print()
print('unique sequence_counter values:')
print(train_df['sequence_counter'].unique()), print(f"number of unique sequence_counter values: {len(train_df['sequence_counter'].unique())}");


train_df_unique_pair = train_df.drop_duplicates(subset=['sequence_id', 'gesture', 'orientation'], keep='first')
train_df_unique_pair


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

train_df_unique_with_subject = train_df.drop_duplicates(subset=['gesture', 'orientation', 'subject'], keep='first')
train_df_unique_with_subject


# Count the number of unique sequence_id values for each orientation
gesture_to_seq_count = train_df.groupby('orientation')['sequence_id'].nunique().sort_values(ascending=False)
print(gesture_to_seq_count)


# Count sequence_type in both DataFrames
type_counts_full = train_df['sequence_type'].value_counts()
type_counts_unique = train_df_unique_pair['sequence_type'].value_counts()

# Create side-by-side bar plots (independent Y axes)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # removed sharey=True

# Plot for full dataset
type_counts_full.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[0])
axes[0].set_title('All Data')
axes[0].set_xlabel('Sequence Type')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(type_counts_full.index, rotation=0)

# Plot for unique dataset
type_counts_unique.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[1])
axes[1].set_title('Unique Sequence_id-Gesture-Orientation')
axes[1].set_xlabel('Sequence Type')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(type_counts_unique.index, rotation=0)

# Adjust layout
plt.tight_layout()
plt.show()


# Count sequence_type in both DataFrames
type_counts_full = train_df['phase'].value_counts()
type_counts_unique = train_df_unique_pair['phase'].value_counts()
train_df_unique_with_phase = train_df.drop_duplicates(subset=['sequence_id', 'gesture', 'orientation', 'phase'], keep='first')
type_counts_unique_with_phase = train_df_unique_with_phase['phase'].value_counts()

# Create side-by-side bar plots (independent Y axes)
fig, axes = plt.subplots(1, 3, figsize=(14, 7))  # removed sharey=True

# Plot for full dataset
type_counts_full.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[0])
axes[0].set_title('All Data')
axes[0].set_xlabel('Phase')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(type_counts_full.index, rotation=90)

# Plot for unique dataset
type_counts_unique.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[1])
axes[1].set_title('Unique Sequence_id-Gesture-Orientation')
axes[1].set_xlabel('Phase')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(type_counts_unique.index, rotation=90)

# Plot for unique dataset
type_counts_unique_with_phase.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[2])
axes[2].set_title('Unique Sequence_id-Gesture-Orientation-Phase')
axes[2].set_xlabel('Phase')
axes[2].set_ylabel('Count')
axes[2].set_xticklabels(type_counts_unique_with_phase.index, rotation=90)

# Adjust layout
plt.tight_layout()
plt.show()


# Count the number of unique sequence_id values for each gesture
sequence_id_counts = train_df.groupby('gesture')['sequence_id'].nunique().sort_values(ascending=False)
print(sequence_id_counts)


# Count sequence_type in both DataFrames
type_counts_full = train_df['gesture'].value_counts()
type_counts_unique = train_df_unique_pair['gesture'].value_counts()
type_counts_unique_with_subject = train_df_unique_with_subject['gesture'].value_counts()

# Create side-by-side bar plots (independent Y axes)
fig, axes = plt.subplots(1, 3, figsize=(14, 7))  # removed sharey=True

# Plot for full dataset
type_counts_full.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[0])
axes[0].set_title('All Data')
axes[0].set_xlabel('Gesture')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(type_counts_full.index, rotation=90)

# Plot for unique dataset
type_counts_unique.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[1])
axes[1].set_title('Unique Sequence_id-Gesture-Orientation')
axes[1].set_xlabel('Gesture')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(type_counts_unique.index, rotation=90)

# Plot for unique dataset
type_counts_unique_with_subject.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[2])
axes[2].set_title('Unique Subject-Gesture-Orientation')
axes[2].set_xlabel('Gesture')
axes[2].set_ylabel('Count')
axes[2].set_xticklabels(type_counts_unique_with_subject.index, rotation=90)

# Adjust layout
plt.tight_layout()
plt.show()


# Count sequence_type in both DataFrames
type_counts_full = train_df['behavior'].value_counts()
type_counts_unique = train_df_unique_pair['behavior'].value_counts()

# Create side-by-side bar plots (independent Y axes)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # removed sharey=True

# Plot for full dataset
type_counts_full.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[0])
axes[0].set_title('All Data')
axes[0].set_xlabel('Behavior')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(type_counts_full.index, rotation=90)

# Plot for unique dataset
type_counts_unique.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[1])
axes[1].set_title('Unique Sequence_id-Gesture-Orientation')
axes[1].set_xlabel('Behavior')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(type_counts_unique.index, rotation=90)

# Adjust layout
plt.tight_layout()
plt.show()


gesture_counts = train_df.groupby('behavior')['sequence_id'].nunique().sort_values(ascending=False)

plt.figure(figsize=(16, 8))
sns.barplot(x=gesture_counts.index, y=gesture_counts.values, width=0.5,  color='salmon')
plt.title('Unique Behavior-Sequence_id')
plt.xlabel('Behavior')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()


gesture_counts = train_df.groupby('subject')['gesture'].nunique().sort_values(ascending=False)

plt.figure(figsize=(16, 6))
sns.barplot(x=gesture_counts.index, y=gesture_counts.values, width=0.5)
plt.title('Number of Unique Gesture per Subject')
plt.xlabel('Subject')
plt.ylabel('Unique Gesture')
plt.xticks(rotation=90)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()


gesture_counts = train_df.groupby('subject')['gesture'].nunique()

subjects_with_less_than_18 = gesture_counts[gesture_counts < 18]

print("Subjects with fewer than 18 unique gestures:")
print(subjects_with_less_than_18)


gesture_counts = train_df.groupby('subject')['orientation'].nunique().sort_values(ascending=False)

plt.figure(figsize=(16, 5))
sns.barplot(x=gesture_counts.index, y=gesture_counts.values, width=0.4)
plt.title('Number of Unique Orientation per Subject')
plt.xlabel('Subject')
plt.ylabel('Unique Orientation')
plt.xticks(rotation=90)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()


train_df[train_df['subject'] == 'SUBJ_053217']['orientation'].unique()


sequence_id_counts = train_df.groupby('subject')['sequence_id'].nunique().sort_values(ascending=False)
print(sequence_id_counts)

plt.figure(figsize=(16, 4))
sns.barplot(x=sequence_id_counts.index, y=sequence_id_counts.values, width=0.5)
plt.title('Number of Unique sequence_id per Subject')
plt.xlabel('Subject')
plt.ylabel('Unique sequence_id Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


print(train_df[train_df['subject'] == 'SUBJ_036405']['gesture'].unique())
print()
print(train_df[train_df['subject'] == 'SUBJ_036405']['orientation'].unique())


4*18


# Keep only one row per sequence_id (e.g., the first occurrence)
unique_sequences_df = train_df.drop_duplicates(subset='sequence_id')

# Filter for the two specific subjects
subset_df = unique_sequences_df[unique_sequences_df['subject'].isin(['SUBJ_053217', 'SUBJ_036405'])]

# Count the number of times each gesture was performed by each subject
gesture_counts = subset_df.groupby(['gesture', 'subject']).size().unstack(fill_value=0)

print(gesture_counts)


# Keep only one row per sequence_id (e.g., the first occurrence)
unique_sequences_df = train_df.drop_duplicates(subset='sequence_id')

# Filter for the two specific subjects
subset_df = unique_sequences_df[unique_sequences_df['subject'].isin(['SUBJ_053217', 'SUBJ_036405', 'SUBJ_011323', 'SUBJ_016552'])]

# Count the number of times each orientation was performed by each subject
orientation_counts = subset_df.groupby(['orientation', 'subject']).size().unstack(fill_value=0)

print(orientation_counts)


# Step 1: Unique sequences only (one per sequence_id)
unique_sequences_df = train_df.drop_duplicates(subset='sequence_id')

# Step 2: Subjects to check
subjects_to_check = ['SUBJ_016552', 'SUBJ_038023', 'SUBJ_053217', 'SUBJ_036405']

# Step 3: All possible gesture–orientation combinations in the dataset
all_gestures = train_df['gesture'].unique()
all_orientations = train_df['orientation'].unique()

from itertools import product

# Create full combination set
full_combinations = set(product(all_gestures, all_orientations))

# Step 4: Check missing combinations for each subject
for subject in subjects_to_check:
    subject_df = unique_sequences_df[unique_sequences_df['subject'] == subject]
    subject_combinations = set(zip(subject_df['gesture'], subject_df['orientation']))

    missing = full_combinations - subject_combinations

    print(f"\n🔍 Missing gesture-orientation combinations for {subject} ({len(missing)} missing):")
    for gesture, orientation in sorted(missing):
        print(f" - {gesture} / {orientation}")


# dedup_df = train_df.drop_duplicates(subset=['sequence_id', 'sequence_counter'])

counter_dist = train_df.groupby('sequence_id')['sequence_counter'].nunique()

sns.histplot(counter_dist, bins=50, color='steelblue')
plt.title('Histogram of Number of sequence_counter per sequence_id')
plt.xlabel('Number of sequence_counter')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


sequence_counter_counts = train_df.groupby('phase')['sequence_counter'].nunique().sort_values(ascending=False)

sns.barplot(x=sequence_counter_counts.index, y=sequence_counter_counts.values, width=0.5)
plt.title('Number of Unique sequence_counter per Phase')
plt.xlabel('Phase')
plt.ylabel('Unique sequence_counter')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


print(train_df.columns)


columns_to_drop = ['row_id', 'sequence_type', 'sequence_id', 'sequence_counter',
                   'subject', 'orientation', 'behavior', 'phase', 'gesture']

train_temp = train_df.drop(columns=columns_to_drop)
train_temp


train_temp.isna().any().any()


train_temp.isna().sum().sum()


train_acc = train_df[['acc_x', 'acc_y', 'acc_z']]
train_acc


train_acc.isna().any().any()


# Choose a random sequence
random_seq_id = np.random.choice(train_df['sequence_id'].unique())
train_temp = train_df[train_df['sequence_id'] == random_seq_id].copy()
train_temp = train_temp.sort_values('sequence_counter').reset_index(drop=True)
# Get handedness
handedness = train_dem_df[train_dem_df['subject']== train_temp['subject'].iloc[0]]['handedness'].iloc[0]
label_handedness = 'Left-handed' if handedness == 0 else 'Right-handed'

# Acceleration Magnitude
train_temp['acc_mag'] = np.sqrt(train_temp['acc_x']**2 + train_temp['acc_y']**2 + train_temp['acc_z']**2)

behavior_colors = {
    'Relaxes and moves hand to target location': 'lightgray',
    'Moves hand to target location': 'lightgray',
    'Hand at target location': 'gold',
    'Performs gesture': 'lightcoral'
}

plt.figure(figsize=(15, 7))
plt.plot(train_temp['sequence_counter'], train_temp['acc_mag'], label='Acceleration Magnitude', color='black', linewidth=2.5)
plt.plot(train_temp['sequence_counter'], train_temp['acc_x'], label='acc_x', color='blue')
plt.plot(train_temp['sequence_counter'], train_temp['acc_y'], label='acc_y', color='orange')
plt.plot(train_temp['sequence_counter'], train_temp['acc_z'], label='acc_z', color='green')


behavior_series = train_temp['behavior']
counter_series = train_temp['sequence_counter']
change_points = (behavior_series != behavior_series.shift()).cumsum()
grouped = list(train_temp.groupby(change_points))

used_labels = set()
for i, (_, group) in enumerate(grouped):
    behavior = group['behavior'].iloc[0]
    color = behavior_colors.get(behavior, 'white')
    t0 = group['sequence_counter'].iloc[0]

    if i < len(grouped) - 1:
        t1 = grouped[i + 1][1]['sequence_counter'].iloc[0]
    else:
        t1 = group['sequence_counter'].iloc[-1]

    label_arg = behavior if behavior not in used_labels else None
    plt.axvspan(t0, t1, color=color, alpha=0.3, label=label_arg)
    used_labels.add(behavior)

subject_name = train_temp['subject'].iloc[0]
gesture_name = train_temp['gesture'].iloc[0]
seq_type = train_temp['sequence_type'].iloc[0]

plt.title(
f"$\\bf{{Subject:}}${subject_name}   $\\bf{{Sequence:}}${random_seq_id}   $\\bf{{Gesture:}}${gesture_name}   $\\bf{{Seq-type:}}${seq_type}   $\\bf{{Handedness:}}${label_handedness}"
)
# plt.title(
# f"Subject: $\\bf{{{subject_name}}}$     Sequence: $\\bf{{{random_seq_id}}}$     Gesture: $\\bf{{{gesture_name}}}$   ({seq_type}) - ({label_handedness})"
# )
plt.xlabel('Time Step')
plt.ylabel('Acceleration')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


train_rot = train_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
train_rot


train_rot.isna().any().any()


train_rot.isna().sum()


random_seq_id = np.random.choice(train_df['sequence_id'].unique())
train_temp = train_df[train_df['sequence_id'] == random_seq_id].copy()
train_temp = train_temp.sort_values('sequence_counter').reset_index(drop=True)

# Label handedness
handedness = train_dem_df[train_dem_df['subject']== train_temp['subject'].iloc[0]]['handedness'].iloc[0]
label_handedness = 'Left-handed' if handedness == 0 else 'Right-handed'

# Rotation Angle (rad)
train_temp['rot_w_clipped'] = train_temp['rot_w'].clip(-1, 1)
train_temp['rot_angle'] = 2 * np.arccos(train_temp['rot_w_clipped'])

behavior_colors = {
    'Relaxes and moves hand to target location': 'lightgray',
    'Moves hand to target location': 'lightgray',
    'Hand at target location': 'gold',
    'Performs gesture': 'lightcoral'
}

plt.figure(figsize=(15, 7))
plt.plot(train_temp['sequence_counter'], train_temp['rot_angle'], label='Rotation Angle (rad)', color='black', linewidth=1.5)
plt.plot(train_temp['sequence_counter'], train_temp['rot_x'], label='rot_x', color='blue')
plt.plot(train_temp['sequence_counter'], train_temp['rot_y'], label='rot_y', color='orange')
plt.plot(train_temp['sequence_counter'], train_temp['rot_z'], label='rot_z', color='green')
plt.plot(train_temp['sequence_counter'], train_temp['rot_w'], label='rot_w', color='purple')


behavior_series = train_temp['behavior']
counter_series = train_temp['sequence_counter']
change_points = (behavior_series != behavior_series.shift()).cumsum()
grouped = list(train_temp.groupby(change_points))

used_labels = set()
for i, (_, group) in enumerate(grouped):
    behavior = group['behavior'].iloc[0]
    color = behavior_colors.get(behavior, 'white')
    t0 = group['sequence_counter'].iloc[0]

    if i < len(grouped) - 1:
        t1 = grouped[i + 1][1]['sequence_counter'].iloc[0]
    else:
        t1 = group['sequence_counter'].iloc[-1]

    label_arg = behavior if behavior not in used_labels else None
    plt.axvspan(t0, t1, color=color, alpha=0.3, label=label_arg)
    used_labels.add(behavior)

subject_name = train_temp['subject'].iloc[0]
gesture_name = train_temp['gesture'].iloc[0]
seq_type = train_temp['sequence_type'].iloc[0]

plt.title(
f"$\\bf{{Subject:}}${subject_name}   $\\bf{{Sequence:}}${random_seq_id}   $\\bf{{Gesture:}}${gesture_name}   $\\bf{{Seq-type:}}${seq_type}   $\\bf{{Handedness:}}${label_handedness}"
)
# plt.title(
# f"Subject: $\\bf{{{subject_name}}}$     Sequence: $\\bf{{{random_seq_id}}}$     Gesture: $\\bf{{{gesture_name}}}$   ({seq_type}) - ({label_handedness})"
# )
plt.xlabel('Time Step')
plt.ylabel('Orientation data')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


from scipy.spatial.transform import Rotation as R

random_seq_id = np.random.choice(train_df['sequence_id'].unique())
train_temp = train_df[train_df['sequence_id'] == random_seq_id].copy()
train_temp = train_temp.sort_values('sequence_counter').reset_index(drop=True)

# Label handedness
handedness = train_dem_df[train_dem_df['subject']== train_temp['subject'].iloc[0]]['handedness'].iloc[0]
label_handedness = 'Left-handed' if handedness == 0 else 'Right-handed'

# Convert quaternion (rot_x, rot_y, rot_z, rot_w) to Euler angles: roll, pitch, yaw
quats = train_temp[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
r = R.from_quat(quats)  # Format: (x, y, z, w)
euler_angles = r.as_euler('xyz', degrees=True)  # Convert to degrees: roll (x), pitch (y), yaw (z)

# Add Euler angles to DataFrame
train_temp['roll'] = euler_angles[:, 0]
train_temp['pitch'] = euler_angles[:, 1]
train_temp['yaw'] = euler_angles[:, 2]

behavior_colors = {
    'Relaxes and moves hand to target location': 'lightgray',
    'Moves hand to target location': 'lightgray',
    'Hand at target location': 'gold',
    'Performs gesture': 'lightcoral'
}

# Plot roll, pitch, yaw over time
plt.figure(figsize=(15, 7))
plt.plot(train_temp['sequence_counter'], train_temp['roll'], label='Roll (°)', color='red')
plt.plot(train_temp['sequence_counter'], train_temp['pitch'], label='Pitch (°)', color='blue')
plt.plot(train_temp['sequence_counter'], train_temp['yaw'], label='Yaw (°)', color='green')

behavior_series = train_temp['behavior']
counter_series = train_temp['sequence_counter']
change_points = (behavior_series != behavior_series.shift()).cumsum()
grouped = list(train_temp.groupby(change_points))

used_labels = set()
for i, (_, group) in enumerate(grouped):
    behavior = group['behavior'].iloc[0]
    color = behavior_colors.get(behavior, 'white')
    t0 = group['sequence_counter'].iloc[0]

    if i < len(grouped) - 1:
        t1 = grouped[i + 1][1]['sequence_counter'].iloc[0]
    else:
        t1 = group['sequence_counter'].iloc[-1]

    label_arg = behavior if behavior not in used_labels else None
    plt.axvspan(t0, t1, color=color, alpha=0.3, label=label_arg)
    used_labels.add(behavior)

subject_name = train_temp['subject'].iloc[0]
gesture_name = train_temp['gesture'].iloc[0]
seq_type = train_temp['sequence_type'].iloc[0]

plt.title(
f"$\\bf{{Subject:}}${subject_name}   $\\bf{{Sequence:}}${random_seq_id}   $\\bf{{Gesture:}}${gesture_name}   $\\bf{{Seq-type:}}${seq_type}   $\\bf{{Handedness:}}${label_handedness}"
)
plt.xlabel('Time Step')
plt.ylabel('Orientation data')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


train_thm = train_df[['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']]
train_thm


train_thm.isna().any().any()


train_thm.isna().sum()


random_seq_id = np.random.choice(train_df['sequence_id'].unique())
train_temp = train_df[train_df['sequence_id'] == random_seq_id].copy()
handedness = train_dem_df[train_dem_df['subject']== train_temp['subject'].iloc[0]]['handedness'].iloc[0]
label_handedness = 'Left-handed' if handedness == 0 else 'Right-handed'
# train_temp = train_temp.sort_values('sequence_counter').reset_index(drop=True)

# Gather thermopile columns and compute their per-timestamp mean
thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
train_temp['thm_mean'] = train_temp[thm_cols].mean(axis=1)

behavior_colors = {
    'Relaxes and moves hand to target location': 'lightgray',
    'Moves hand to target location': 'lightgray',
    'Hand at target location': 'gold',
    'Performs gesture': 'lightcoral'
}

plt.figure(figsize=(15, 7))
plt.plot(train_temp['sequence_counter'], train_temp['thm_mean'], label='Average Temperature', color='black', linewidth=2.5)
plt.plot(train_temp['sequence_counter'], train_temp['thm_1'], label='thm_1', color='blue')
plt.plot(train_temp['sequence_counter'], train_temp['thm_2'], label='thm_2', color='orange')
plt.plot(train_temp['sequence_counter'], train_temp['thm_3'], label='thm_3', color='green')
plt.plot(train_temp['sequence_counter'], train_temp['thm_4'], label='thm_4', color='purple')
plt.plot(train_temp['sequence_counter'], train_temp['thm_5'], label='thm_5', color='red')


behavior_series = train_temp['behavior']
counter_series = train_temp['sequence_counter']
change_points = (behavior_series != behavior_series.shift()).cumsum()
grouped = list(train_temp.groupby(change_points))

used_labels = set()
for i, (_, group) in enumerate(grouped):
    behavior = group['behavior'].iloc[0]
    color = behavior_colors.get(behavior, 'white')
    t0 = group['sequence_counter'].iloc[0]

    if i < len(grouped) - 1:
        t1 = grouped[i + 1][1]['sequence_counter'].iloc[0]
    else:
        t1 = group['sequence_counter'].iloc[-1]

    label_arg = behavior if behavior not in used_labels else None
    plt.axvspan(t0, t1, color=color, alpha=0.3, label=label_arg)
    used_labels.add(behavior)

subject_name = train_temp['subject'].iloc[0]
gesture_name = train_temp['gesture'].iloc[0]
seq_type = train_temp['sequence_type'].iloc[0]

plt.title(
f"$\\bf{{Subject:}}${subject_name}   $\\bf{{Sequence:}}${random_seq_id}   $\\bf{{Gesture:}}${gesture_name}   $\\bf{{Seq-type:}}${seq_type}   $\\bf{{Handedness:}}${label_handedness}"
)
# plt.title(
# f"Subject: $\\bf{{{subject_name}}}$     Sequence: $\\bf{{{random_seq_id}}}$     Gesture: $\\bf{{{gesture_name}}}$   ({seq_type}) - ({label_handedness})"
# )

plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


columns_TOF = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
train_TOF = train_df[columns_TOF]
train_TOF


train_thm.isna().any().any()


train_thm.isna().sum()


random_seq_id = np.random.choice(train_df['sequence_id'].unique())
train_temp = train_df[train_df['sequence_id'] == random_seq_id].copy()
train_temp = train_temp.sort_values('sequence_counter').reset_index(drop=True)
handedness = train_dem_df[train_dem_df['subject']== train_temp['subject'].iloc[0]]['handedness'].iloc[0]
label_handedness = 'Left-handed' if handedness == 0 else 'Right-handed'


# Identify the 64 columns for each ToF sensor and compute their per-timestamp mean
for i_sensor in range(1, 6):
    pixel_cols = [f'tof_{i_sensor}_v{pix}' for pix in range(64)]
    train_temp[f'tof_{i_sensor}_mean'] = train_temp[pixel_cols].replace(-1, np.nan).mean(axis=1)

behavior_colors = {
    'Relaxes and moves hand to target location': 'lightgray',
    'Moves hand to target location': 'lightgray',
    'Hand at target location': 'gold',
    'Performs gesture': 'lightcoral'
}

plt.figure(figsize=(15, 7))
plt.plot(train_temp['sequence_counter'], train_temp['tof_1_mean'], label='tof_1_mean', color='blue')
plt.plot(train_temp['sequence_counter'], train_temp['tof_2_mean'], label='tof_2_mean', color='orange')
plt.plot(train_temp['sequence_counter'], train_temp['tof_3_mean'], label='tof_3_mean', color='green')
plt.plot(train_temp['sequence_counter'], train_temp['tof_4_mean'], label='tof_4_mean', color='purple')
plt.plot(train_temp['sequence_counter'], train_temp['tof_5_mean'], label='tof_5_mean', color='red')


behavior_series = train_temp['behavior']
counter_series = train_temp['sequence_counter']
change_points = (behavior_series != behavior_series.shift()).cumsum()
grouped = list(train_temp.groupby(change_points))

used_labels = set()
for i, (_, group) in enumerate(grouped):
    behavior = group['behavior'].iloc[0]
    color = behavior_colors.get(behavior, 'white')
    t0 = group['sequence_counter'].iloc[0]

    if i < len(grouped) - 1:
        t1 = grouped[i + 1][1]['sequence_counter'].iloc[0]
    else:
        t1 = group['sequence_counter'].iloc[-1]

    label_arg = behavior if behavior not in used_labels else None
    plt.axvspan(t0, t1, color=color, alpha=0.3, label=label_arg)
    used_labels.add(behavior)

subject_name = train_temp['subject'].iloc[0]
gesture_name = train_temp['gesture'].iloc[0]
seq_type = train_temp['sequence_type'].iloc[0]

plt.title(
f"$\\bf{{Subject:}}$ {subject_name}    $\\bf{{Sequence:}}$ {random_seq_id}    $\\bf{{Gesture:}}$ {gesture_name}     ({seq_type}) - ({label_handedness})"
)
# plt.title(
# f"Subject: $\\bf{{{subject_name}}}$     Sequence: $\\bf{{{random_seq_id}}}$     Gesture: $\\bf{{{gesture_name}}}$   ({seq_type}) - ({label_handedness})"
# )

plt.xlabel('Time Step')
plt.ylabel('Mean ToF Distance')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()


import imageio.v2 as imageio  # for saving GIF

# Choose ToF sensor number (1 to 5)
tof_sensor_num = 5  # Change to desired sensor number

# Extract the 64 sensor values from the selected ToF sensor
sensor_cols = [f'tof_{tof_sensor_num}_v{i}' for i in range(64)]
tof_sensor_data = train_temp[sensor_cols].replace(-1, np.nan).values  # Replace -1 with NaN for better visualization

# Determine the number of frames available and limit to a maximum (e.g., 100)
num_frames = min(100, len(tof_sensor_data))  # Prevent index out of bounds

# Create a temporary directory to store individual frame images
os.makedirs("temp_frames", exist_ok=True)
filenames = []

# Generate a plot for each frame and save it as an image
for i in range(num_frames):
    frame = tof_sensor_data[i].reshape(8, 8)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(frame, cmap='inferno', vmin=0, vmax=np.nanmax(tof_sensor_data), interpolation='nearest')
    ax.set_title(f'ToF Sensor {tof_sensor_num} - Frame {i}')
    ax.axis('off')
    
    filename = f"temp_frames/frame_{i:03d}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    filenames.append(filename)
    plt.close()

# Combine all saved images into a single animated GIF
gif_filename = f'tof_sensor_{tof_sensor_num}_frames.gif'
with imageio.get_writer(gif_filename, mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"✅ GIF saved as: {gif_filename}")


train_df


train_df.iloc[:, 8:].isna().sum()


train_df.describe()


train_df[train_df['tof_5_v1'] > -1].tof_5_v1.median()


# Define feature groups
acc_cols = [col for col in train_df.columns if col.startswith("acc_")]
rot_cols = [col for col in train_df.columns if col.startswith("rot_")]
thm_cols = [col for col in train_df.columns if col.startswith("thm_")]
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
plot_combined_box_violin(train_df, acc_cols, "Accelerometer Sensors")
plot_combined_box_violin(train_df, rot_cols, "Rotation Sensors")
plot_combined_box_violin(train_df, thm_cols, "Thermal Sensors")
plot_combined_box_violin(train_df, tof_cols, "ToF Sensors (first pixel per group)")


# Find outliers in thm_1
outliers_thm1 = train_df[train_df['thm_1'] < 10]
display(outliers_thm1[thm_cols])

# Find outliers in thm_3
outliers_thm3 = train_df[train_df['thm_3'] == 0]
display(outliers_thm3[thm_cols])


thm_target = 'thm_3'

# Identify all sequence_ids where thm_3 is zero in at least one row
sequences_with_zero_thm3 = train_df[train_df[thm_target] == 0]['sequence_id'].unique()

# Create a dataframe only for those sequences
subset = train_df[train_df['sequence_id'].isin(sequences_with_zero_thm3)]

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
sequences_with_zero_thm3 = train_df[train_df[thm_target] < 10]['sequence_id'].unique()

# Create a dataframe only for those sequences
subset = train_df[train_df['sequence_id'].isin(sequences_with_zero_thm3)]

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


train_df = impute_low_thermal_values(train_df, thm_cols, threshold=10.0)


plot_combined_box_violin(train_df, thm_cols, "Thermal Sensors")


train_df.iloc[:, 9:].isna().sum()


# Total number of rows in the dataset
total_rows = len(train_df)

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
    total_missing = train_df[columns].isna().sum().sum()
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
analyze_sensor_missingness(train_df, acc_cols, "acc")

# Analyze rotation
analyze_sensor_missingness(train_df, rot_cols, "rot")

# Analyze thermal sensors
for thm in thm_sensors:
    analyze_sensor_missingness(train_df, [thm], thm)

# Analyze ToF sensors
for i, base in enumerate(tof_sensors, 1):
    tof_cols = [f"{base}v{j}" for j in range(64)]
    analyze_sensor_missingness(train_df, tof_cols, f"tof_{i}")

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
nan_ratio_per_seq = train_df.groupby("sequence_id").apply(lambda x: x.isna().mean())

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
tof_columns = [col for col in train_df.columns if col.startswith("tof_")]

# Total number of rows
total_rows = len(train_df)

# Create list to store results
tof_neg1_stats = []

# Loop through each ToF column and count -1 values
for col in tof_columns:
    count_neg1 = (train_df[col] == -1).sum()
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


train_df.describe()


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


df_train_imputed = imputer.fit_transform(train_df)


df_train_imputed.isna().sum().sum()


plot_combined_box_violin(train_df, acc_cols, "Accelerometer Sensors")

plot_combined_box_violin(train_df, rot_cols, "Rotation Sensors")

plot_combined_box_violin(train_df, thm_cols, "Thermal Sensors")

tof_cols = [f"tof_{i}_v{0}" for i in range(1, 6)]
plot_combined_box_violin(train_df, tof_cols, "ToF Sensors (first pixel per group)")


train_df.describe()


from sklearn.preprocessing import StandardScaler

# Select sensor columns (excluding rotation)
feature_cols = [col for col in train_df.columns if col.startswith(("acc_", "thm_", "tof_"))]

# Initialize scaler
scaler = StandardScaler()

# Apply Z-score normalization
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

train_df.describe()


df_train_imputed.describe()


# from sklearn.preprocessing import StandardScaler

# Select sensor columns (excluding rotation)
feature_cols = [col for col in train_df.columns if col.startswith(("acc_", "thm_", "tof_"))]

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

