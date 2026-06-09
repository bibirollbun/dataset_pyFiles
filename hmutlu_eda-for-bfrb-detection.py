import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
%matplotlib inline


# File paths
data_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/"

# Load datasets
train_df      = pd.read_csv(data_path + "train.csv")
test_df       = pd.read_csv(data_path + "test.csv")
train_dem_df  = pd.read_csv(data_path + "train_demographics.csv")
test_dem_df   = pd.read_csv(data_path + "test_demographics.csv")


# Display dataset shapes in a DataFrame 
shapes_df = pd.DataFrame({
    "Dataset": ["Train Sensor Data", "Test Sensor Data", "Train Demographics", "Test Demographics"],
    "Rows": [train_df.shape[0], test_df.shape[0], train_dem_df.shape[0], test_dem_df.shape[0]],
    "Columns": [train_df.shape[1], test_df.shape[1], train_dem_df.shape[1], test_dem_df.shape[1]]
})

shapes_df.style.set_caption("ğŸ“¦ Dataset Shapes Overview").background_gradient(cmap='Blues', subset=["Rows", "Columns"])


print("Sample rows from train.csv:")
display(train_df.head(2))

print("\nSample rows from train_demographics.csv:")
display(train_dem_df.head(2))


print("Sample rows from test.csv:")
display(test_df.head(2))

print("\nSample rows from test_demographics.csv:")
display(test_dem_df.head(2))


# Select numeric columns only
numeric_cols = train_df.select_dtypes(include=[np.number]).columns

# Generate summary statistics
summary_stats = train_df[numeric_cols].describe().transpose()
summary_stats = summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
summary_stats


unique_sequences = train_df['sequence_id'].nunique()
unique_gestures = train_df['gesture'].nunique()
unique_subjects = train_df['subject'].nunique()

summary_df = pd.DataFrame({
    "Metric": ["Unique sequences", "Unique gestures", "Unique subjects"],
    "Count": [unique_sequences, unique_gestures, unique_subjects]
})

summary_df.style.set_caption("ğŸ”¢ Unique Values Summary").background_gradient(cmap='Greens', subset=["Count"])


imu_cols = [col for col in train_df.columns if col.startswith('acc_') or col.startswith('rot_')]
thm_cols = [col for col in train_df.columns if col.startswith('thm_')]
tof_cols = [col for col in train_df.columns if col.startswith('tof_')]


# Fraction of missing values per sensor group
imu_missing_frac = train_df[imu_cols].isnull().mean().mean()
thm_missing_frac = train_df[thm_cols].isnull().mean().mean()
tof_missing_frac = train_df[tof_cols].isnull().mean().mean()

missing_summary = pd.DataFrame({
    'Sensor Group': ['IMU (acc + rot)', 'Thermopiles (thm_1 to thm_5)', 'ToF sensors (tof_1_v* to tof_5_v*)'],
    'Fraction Missing': [imu_missing_frac, thm_missing_frac, tof_missing_frac]
})

missing_summary.style.format({"Fraction Missing": "{:.4f}"}).set_caption("Missing Values Fraction by Sensor Group")


# Calculate % of -1 per ToF sensor (grouped by tof_1, tof_2, ..., tof_5)

tof_sensor_ids = [f"tof_{i}" for i in range(1,6)]
tof_neg1_percent = {}

for sensor_id in tof_sensor_ids:
    sensor_cols = [col for col in tof_cols if col.startswith(sensor_id)]
    # Flatten data for all these columns and calculate percentage of -1
    neg1_count = (train_df[sensor_cols] == -1).sum().sum()
    total_count = train_df[sensor_cols].size
    tof_neg1_percent[sensor_id] = neg1_count / total_count

tof_neg1_df = pd.DataFrame.from_dict(tof_neg1_percent, orient='index', columns=['% No Reflection (-1)'])
tof_neg1_df.style.format({"% No Reflection (-1)": "{:.4%}"}).set_caption("Percentage of -1 (No Reflection) Values in ToF Sensors")


tof_sensor_ids = [f"tof_{i}" for i in range(1,6)]
tof_neg1_percent = {}
tof_nan_percent = {}

for sensor_id in tof_sensor_ids:
    sensor_cols = [col for col in tof_cols if col.startswith(sensor_id)]
    
    total_count = train_df[sensor_cols].size
    neg1_count = (train_df[sensor_cols] == -1).sum().sum()
    nan_count = train_df[sensor_cols].isna().sum().sum()
    
    tof_neg1_percent[sensor_id] = neg1_count / total_count * 100
    tof_nan_percent[sensor_id] = nan_count / total_count * 100

# Combine into a single DataFrame
tof_quality_df = pd.DataFrame({
    'No Reflection (-1) %': tof_neg1_percent,
    'Missing (NaN) %': tof_nan_percent
}).T

# Plot
x = np.arange(len(tof_sensor_ids))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, list(tof_nan_percent.values()), width, label='% NaN', color='skyblue')
bars2 = ax.bar(x + width/2, list(tof_neg1_percent.values()), width, label='% -1', color='salmon')

# Labels and formatting
ax.set_ylabel('Percentage')
ax.set_title('ToF Sensor Groups: Missing (NaN) vs. No Reflection (-1)')
ax.set_xticks(x)
ax.set_xticklabels(tof_sensor_ids)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Annotate bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()



# Count gestures grouped by sequence type (Target or Non-Target)
gesture_counts = train_df.groupby(['gesture', 'sequence_type'])['sequence_id'].nunique().reset_index()
gesture_counts.columns = ['Gesture', 'Sequence Type', 'Unique Sequence Count']
# Define custom palette matching your colors
custom_palette = {
    'Target': '#ff9999',      # soft red
    'Non-Target': '#66b3ff'   # soft blue
}
# Sort by total frequency for consistent ordering
gesture_order = gesture_counts.groupby('Gesture')['Unique Sequence Count'].sum().sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.barplot(data=gesture_counts, x='Gesture', y='Unique Sequence Count', hue='Sequence Type', order=gesture_order, palette=custom_palette)
plt.xticks(rotation=45, ha='right')
plt.title('Unique Gesture Sequence Counts by Type')
plt.xlabel('Gesture')
plt.ylabel('Number of Unique Sequences')
plt.legend(title='Sequence Type')
plt.tight_layout()
plt.show()


# Count total number of sequences by type
sequence_type_counts = train_df[['sequence_id', 'sequence_type']].drop_duplicates()['sequence_type'].value_counts()

# Plot
plt.figure(figsize=(6, 4))
sequence_type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
plt.title('Proportion of Target vs. Non-Target Sequences')
plt.ylabel('')
plt.tight_layout()
plt.show()



# Count unique sequences per gesture only
gesture_seq_counts = train_df.groupby('gesture')['sequence_id'].nunique().sort_values(ascending=False)

# Display as styled DataFrame
gesture_seq_counts_df = gesture_seq_counts.reset_index()
gesture_seq_counts_df.columns = ['Gesture', 'Unique Sequences']
gesture_seq_counts_df.style.set_caption(" Unique Sequences per Gesture").background_gradient(cmap='Purples', subset=["Unique Sequences"])


# Select a few example sequence_ids for visualization (stratified by gesture type)
np.random.seed(42)
example_sequences = []
for gesture_type in ['Target', 'Non-Target']:
    seqs = train_df[train_df['sequence_type'] == gesture_type]['sequence_id'].unique()
    example_sequences.extend(np.random.choice(seqs, size=2, replace=False))
example_sequences


for seq_id in example_sequences:
    seq_data = train_df[train_df['sequence_id'] == seq_id]
    plt.figure(figsize=(12, 4))
    plt.plot(seq_data['sequence_counter'], seq_data['acc_x'], label='acc_x')
    plt.plot(seq_data['sequence_counter'], seq_data['acc_y'], label='acc_y')
    plt.plot(seq_data['sequence_counter'], seq_data['acc_z'], label='acc_z')
    plt.title(f'IMU Acceleration for Sequence {seq_id} ({seq_data["gesture"].iloc[0]})')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration (m/sÂ²)')
    plt.legend()
    plt.tight_layout()
    plt.show()


thm_cols = [f'thm_{i}' for i in range(1, 6)]
for seq_id in example_sequences:
    seq_data = train_df[train_df['sequence_id'] == seq_id]
    plt.figure(figsize=(12, 4))
    for col in thm_cols:
        plt.plot(seq_data['sequence_counter'], seq_data[col], label=col)
    plt.title(f'Thermopile Sensor Readings for Sequence {seq_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (Â°C)')
    plt.legend()
    plt.show()


tof_cols = [col for col in train_df.columns if col.startswith('tof_')]
for seq_id in example_sequences:
    seq_data = train_df[train_df['sequence_id'] == seq_id]
    plt.figure(figsize=(12, 4))
    for i in range(1, 6):
        tof_sensor_cols = [f'tof_{i}_v{j}' for j in range(64)]
        mean_dist = seq_data[tof_sensor_cols].replace(-1, np.nan).mean(axis=1)
        pct_neg1 = (seq_data[tof_sensor_cols] == -1).mean(axis=1) * 100
        plt.plot(seq_data['sequence_counter'], mean_dist, label=f'ToF {i} Mean Dist')
        plt.plot(seq_data['sequence_counter'], pct_neg1, '--', label=f'ToF {i} % -1')
    plt.title(f'ToF Sensor Mean Distance and Missingness for Sequence {seq_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Distance / % -1')
    plt.legend()
    plt.show()


seq_id = example_sequences[0]
seq_data = train_df[train_df['sequence_id'] == seq_id]

time_step = 10
if time_step >= len(seq_data):
    time_step = len(seq_data) // 2  # fallback if index is out of bounds

tof_sensor = 1
tof_grid_cols = [f'tof_{tof_sensor}_v{j}' for j in range(64)]

# Extract the row 
row = seq_data.iloc[time_step][tof_grid_cols]

# Convert to numeric (coerce errors to NaN)
tof_values = pd.to_numeric(row, errors='coerce').values.astype(float)

#  Reshape into 8x8 grid
tof_grid = tof_values.reshape(8, 8)

#  - mask -1 values as NaN for better visualization
tof_grid[tof_grid == -1] = np.nan

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(tof_grid, cmap='viridis', annot=False, cbar=True)
plt.title(f'ToF Sensor {tof_sensor} Heatmap at Time Step {time_step} (Sequence {seq_id})')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(data=train_df, x='orientation', order=train_df['orientation'].value_counts().index, palette='Set2')
plt.title('Orientation Distribution')
plt.xticks(rotation=45, ha='right')
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(data=train_df, x='behavior', order=train_df['behavior'].value_counts().index, palette='Pastel1')
plt.title('Behavior Phase Distribution')
plt.xticks(rotation=45, ha='right')
plt.show()


phase_counts = train_df.groupby(['sequence_id', 'behavior']).size().unstack(fill_value=0)
phase_counts.describe()


train_merged = train_df.merge(train_dem_df, on='subject', how='left')


plt.figure(figsize=(6, 4))
sns.histplot(train_merged['age'], bins=15, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.tight_layout()
plt.show()


train_merged['sex_label'] = train_merged['sex'].map({0: 'Female', 1: 'Male'})

plt.figure(figsize=(4, 4))
sns.countplot(data=train_merged, x='sex_label', palette=['#f4a582', '#92c5de'])  # salmon & sky blue
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(
    data=train_merged,
    x='gesture',
    hue='adult_child',
    order=train_merged['gesture'].value_counts().index,
    palette={0: '#8dd3c7', 1: '#fb8072'}  # teal for children, red-orange for adults
)
plt.title('Gesture Counts by Age Group')
plt.xlabel('Gesture')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Age Group', labels=['Child', 'Adult'])
plt.tight_layout()
plt.show()


train_df['imu_magnitude'] = np.sqrt(train_df['acc_x']**2 + train_df['acc_y']**2 + train_df['acc_z']**2)


train_df['quat_norm'] = np.sqrt(train_df['rot_w']**2 + train_df['rot_x']**2 + train_df['rot_y']**2 + train_df['rot_z']**2)


tof_neg1_pct_row = (train_df[tof_cols] == -1).mean(axis=1)


train_df['thm_range'] = train_df[thm_cols].max(axis=1) - train_df[thm_cols].min(axis=1)


plt.figure(figsize=(10, 4))
sns.histplot(train_df['imu_magnitude'], bins=50, kde=True)
plt.title('IMU Signal Magnitude Distribution')
plt.show()

