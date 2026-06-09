# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)



print("Loading data files...")
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Training demographics shape: {train_demographics.shape}")
print(f"Test demographics shape: {test_demographics.shape}")


print("\n" + "="*50)
print("BASIC DATA EXPLORATION")
print("="*50)

imu_cols = [col for col in train_df.columns if col.startswith(('acc_', 'rot_'))]
thm_cols = [col for col in train_df.columns if col.startswith('thm_')]
tof_cols = [col for col in train_df.columns if col.startswith('tof_')]
meta_cols = ['row_id', 'sequence_id', 'sequence_type', 'sequence_counter', 
             'subject', 'gesture', 'orientation', 'behavior']

print(f"\nColumn groups:")
print(f"- Metadata columns: {len(meta_cols)}")
print(f"- IMU columns: {len(imu_cols)}")
print(f"- Thermopile columns: {len(thm_cols)}")
print(f"- Time-of-Flight columns: {len(tof_cols)}")
print(f"- Total columns: {len(train_df.columns)}")

print("\n\nUnique values in categorical columns:")
for col in ['sequence_type', 'gesture', 'orientation', 'behavior']:
    if col in train_df.columns:
        unique_vals = train_df[col].nunique()
        print(f"- {col}: {unique_vals} unique values")
        if unique_vals < 20:
            print(f"  Values: {sorted(train_df[col].unique())}")


print("\n" + "="*50)
print("SEQUENCE ANALYSIS")
print("="*50)

sequence_info = train_df.groupby('sequence_id').agg({
    'sequence_counter': ['count', 'max'],
    'subject': 'first',
    'gesture': 'first',
    'sequence_type': 'first'
}).reset_index()

sequence_info.columns = ['sequence_id', 'length', 'max_counter', 'subject', 'gesture', 'sequence_type']

print(f"\nTotal sequences: {len(sequence_info)}")
print(f"Sequence length statistics:")
print(sequence_info['length'].describe())


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(sequence_info['length'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Distribution of Sequence Lengths')

plt.subplot(1, 2, 2)
sequence_info.boxplot(column='length', by='sequence_type', ax=plt.gca())
plt.title('Sequence Length by Type')
plt.suptitle('')
plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("GESTURE ANALYSIS")
print("="*50)

gesture_counts = sequence_info['gesture'].value_counts()
print(f"\nGesture distribution:")
print(gesture_counts)

plt.figure(figsize=(12, 6))
gesture_counts.plot(kind='bar', color='#ff69b4')
plt.title('Gesture Distribution in Training Data', fontsize=14)
plt.xlabel('Gesture', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.subplot(2, 1, 2)
gesture_type_df = pd.crosstab(sequence_info['gesture'], sequence_info['sequence_type'])
gesture_type_df.plot(kind='bar', stacked=True)
plt.title('Gesture Distribution by Sequence Type')
plt.xlabel('Gesture')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sequence Type')
plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("DEMOGRAPHIC ANALYSIS")
print("="*50)

demo_sequence = sequence_info.merge(train_demographics, on='subject', how='left')

print("\nDemographic statistics:")
print(train_demographics.describe())

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].hist(train_demographics['age'], bins=20, edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')

adult_child_counts = train_demographics['adult_child'].value_counts()
axes[0, 1].pie(adult_child_counts.values, labels=['Child', 'Adult'], autopct='%1.1f%%')
axes[0, 1].set_title('Adult vs Child Distribution')

sex_counts = train_demographics['sex'].value_counts()
axes[0, 2].pie(sex_counts.values, labels=['Female', 'Male'], autopct='%1.1f%%')
axes[0, 2].set_title('Sex Distribution')

hand_counts = train_demographics['handedness'].value_counts()
axes[1, 0].pie(hand_counts.values, labels=['Left', 'Right'], autopct='%1.1f%%')
axes[1, 0].set_title('Handedness Distribution')

axes[1, 1].hist(train_demographics['height_cm'], bins=20, edgecolor='black')
axes[1, 1].set_title('Height Distribution')
axes[1, 1].set_xlabel('Height (cm)')
axes[1, 1].set_ylabel('Count')

axes[1, 2].scatter(train_demographics['shoulder_to_wrist_cm'], 
                   train_demographics['elbow_to_wrist_cm'], alpha=0.6)
axes[1, 2].set_xlabel('Shoulder to Wrist (cm)')
axes[1, 2].set_ylabel('Elbow to Wrist (cm)')
axes[1, 2].set_title('Arm Measurements')

plt.tight_layout()
plt.show()



print("\n" + "="*50)
print("SENSOR DATA VISUALIZATION")
print("="*50)

sample_sequences = sequence_info.sample(min(3, len(sequence_info)))['sequence_id'].values

for seq_id in sample_sequences[:2]:  # Visualize first 2
    seq_data = train_df[train_df['sequence_id'] == seq_id].sort_values('sequence_counter')
    gesture = seq_data['gesture'].iloc[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Sensor Data for Sequence {seq_id} - Gesture: {gesture}')
    
    axes[0].plot(seq_data['sequence_counter'], seq_data['acc_x'], label='Acc X')
    axes[0].plot(seq_data['sequence_counter'], seq_data['acc_y'], label='Acc Y')
    axes[0].plot(seq_data['sequence_counter'], seq_data['acc_z'], label='Acc Z')
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Accelerometer Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for i in range(1, 6):
        axes[1].plot(seq_data['sequence_counter'], seq_data[f'thm_{i}'], label=f'Thermopile {i}')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Thermopile Sensor Data')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    behavior_map = {'Transition': 0, 'Pause': 1, 'Gesture': 2}
    behavior_numeric = seq_data['behavior'].map(behavior_map)
    axes[2].plot(seq_data['sequence_counter'], behavior_numeric, 'o-')
    axes[2].set_ylabel('Behavior Phase')
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(['Transition', 'Pause', 'Gesture'])
    axes[2].set_xlabel('Sequence Counter')
    axes[2].set_title('Behavior Phases')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()




