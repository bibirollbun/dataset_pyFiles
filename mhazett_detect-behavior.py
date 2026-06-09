import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from scipy.stats import skew, kurtosis
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
valid_sequences = df.groupby('sequence_id')[imu_cols].apply(lambda x: x.notna().any().all())
valid_sequence_ids = valid_sequences[valid_sequences].index
df = df[df['sequence_id'].isin(valid_sequence_ids)].copy()

print("Label distribution before training:")
print(df['sequence_type'].value_counts())
print(df.info())
print(df.describe())


sns.boxplot(data=df[['acc_x', 'acc_y', 'acc_z']])
plt.title("Accelerometer Value Distribution")


sample_id = df['sequence_id'].iloc[0]
sample = df[df['sequence_id'] == sample_id]

tof_cols = [f"tof_1_v{i}" for i in range(64)]
tof_grid = sample[tof_cols].iloc[0].values.reshape(8, 8)

plt.imshow(tof_grid, cmap='viridis')
plt.title('ToF Sensor 1 Heatmap')
plt.colorbar(label='Distance')


tof_frames = sample[tof_cols].values.reshape(-1, 8, 8)

fig, ax = plt.subplots()
cax = ax.imshow(tof_frames[0], cmap='viridis', vmin=0, vmax=255)
plt.title(f"ToF Sensor 1 Heatmap – Sequence {sample_id}")
fig.colorbar(cax, label='Distance')

def update(frame):
    cax.set_data(tof_frames[frame])
    ax.set_title("ToF Sensor 1 – Frame {} (Sequence {})".format(frame, sample_id))
    return [cax]
ani = animation.FuncAnimation(fig, update, frames=len(tof_frames), interval=200, blit=True)
ani.save('tof_sensor1_sequence.gif', writer='pillow', fps=5)
plt.tight_layout()


df_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
df_demo = df_demo.dropna()
df_demo = df_demo.rename(columns={'sex':'gender'})
print(df_demo.info())
print('----------------------------------------------')
print(df_demo.describe())
print('----------------------------------------------')
print("Null data:\n",df_demo.isnull().sum())


plt.figure(figsize=(10,6))
correlation_matrix = df_demo.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation of Age, Height, and Body Dimensions (Training Set)')
plt.show()


plt.figure(figsize=(10,7))
df_demo['gender_label'] = df_demo['gender'].map({0: 'Female', 1: 'Male'})

sns.boxplot(data=df_demo, x='gender', y='age', palette='Reds', hue='gender_label')
plt.title("Age distribution by gender")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Age")
plt.grid(True)


sns.histplot(data=df_demo, x='height_cm', kde=True)
plt.title("Height distribution")


print('Preview of Raw Data (First 5 Rows):')
print(df[['sequence_id', 'sequence_counter', 'gesture', 'behavior']].head())

group = df.groupby('sequence_id')

sequence = group.get_group(df['sequence_id'].iloc[0])
print("\nFirst Sequence Group (sequence_id = {}):".format(df['sequence_id'].iloc[0]))
print(sequence.head())


plt.figure(figsize=(12, 5))
plt.plot(sequence['sequence_counter'], sequence['acc_x'], label='acc_x')
plt.plot(sequence['sequence_counter'], sequence['acc_y'], label='acc_y')
plt.plot(sequence['sequence_counter'], sequence['acc_z'], label='acc_z')
plt.title("Accelerometer Signal – Sequence ID: {}".format(df['sequence_id'].iloc[0]))
plt.xlabel("Sensor reading step")
plt.ylabel("Motion intensity")
plt.legend()
plt.grid(True)


tof_cols = [f'tof_1_v{i}' for i in range(64)]
tof_frame = sequence[tof_cols].iloc[0].values.reshape(8, 8)

sns.heatmap(tof_frame, cmap='viridis')
plt.title("ToF sensor 1 – Frame 0")


sensor_cols = []

for col in df.columns:
    if col.startswith(('acc_', 'rot_', 'thm_', 'tof_')):
        sensor_cols.append(col)

print('Sensor columns to be standardized:')
print(sensor_cols)

scaler = StandardScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])


df['sequence_type'].value_counts().plot.pie(autopct='%1.1f%%', labels=['non_target', 'target'], colors=['#8fd9b6','#ff9999'])
plt.title("Class Proportion: BFRB-like vs Non-BFRB")
plt.ylabel("")

