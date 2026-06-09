%%time
import pandas as pd
import numpy as np

# Load data
path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/'

train = pd.read_csv(path + 'train.csv')
num_rows = len(train)
print(f"Number of rows: {num_rows}")

train = pd.read_csv(path + 'train.csv', nrows=200)
train_demo = pd.read_csv(path + 'train_demographics.csv')


# Merge demographics
train = train.merge(train_demo, on='subject', how='left')

# Replace ToF '-1' values with NaN
tof_cols = [col for col in train.columns if col.startswith('tof_')]
train[tof_cols] = train[tof_cols].replace(-1, np.nan)

# Basic info
print("Train shape:", train.shape)
print("Number of sequences:", train['sequence_id'].nunique())
print("Gesture classes:", train['gesture'].nunique())
print("Missing values per sensor type:")
print(train[['acc_x', 'rot_w'] + tof_cols[:5] + ['thm_1']].isna().mean())

# Optional: downcast to reduce memory
float_cols = train.select_dtypes(include='float').columns
train[float_cols] = train[float_cols].astype(np.float32)


print(train.shape)
print(train_demo.shape)
train.head()
train_demo.head()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in `train` Dataset")
plt.show()

plt.figure(figsize=(8, 3))
sns.heatmap(train_demo.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in `train_demo` Dataset")
plt.show()


import seaborn as sns
sns.set(style="whitegrid")

# Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(train_demo['age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.show()

# Sex Distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=train_demo, x='sex')
plt.title('Sex Distribution')
plt.show()

# Handedness
plt.figure(figsize=(5, 4))
sns.countplot(data=train_demo, x='handedness')
plt.title('Handedness Distribution')
plt.show()


# Behavior
plt.figure(figsize=(10, 4))
sns.countplot(data=train, y='behavior', order=train['behavior'].value_counts().index)
plt.title('Behavior Distribution')
plt.show()

# Gesture
plt.figure(figsize=(10, 6))
sns.countplot(data=train, y='gesture', order=train['gesture'].value_counts().index)
plt.title('Gesture Distribution')
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='sequence_type')
plt.title('Sequence Type Distribution')
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=train, x='orientation')
plt.title('Orientation Distribution')
plt.xticks(rotation=45)
plt.show()


sensor_prefixes = ['acc_', 'gyro_', 'mag_', 'tof_']
sensor_counts = [len([col for col in train.columns if col.startswith(prefix)]) for prefix in sensor_prefixes]

plt.figure(figsize=(6, 4))
sns.barplot(x=sensor_prefixes, y=sensor_counts)
plt.title('Sensor Column Counts by Type')
plt.ylabel('Number of Features')
plt.show()

