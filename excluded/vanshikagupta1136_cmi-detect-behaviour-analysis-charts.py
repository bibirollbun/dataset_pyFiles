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


import numpy as np # linear algebra
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '{:,.3f}'.format(x))


df_train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
df_train_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
df_test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
df_test_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')


print(list(df_train.columns))


sns.countplot(x='gesture', data=df_train)
plt.title('Distribution of Gesture Labels')
plt.xticks(rotation=90)
plt.show()


seq_lengths = df_train.groupby('sequence_id')['sequence_counter'].count()
sns.histplot(seq_lengths, bins=30, kde=True)
plt.title("Sequence Length Distribution")
plt.xlabel("Number of Rows in a Sequence")
plt.ylabel("Frequency")
plt.show()



acc_cols = ['acc_x', 'acc_y', 'acc_z']
df_grouped = df_train.groupby('sequence_counter')[acc_cols].mean()

df_grouped.plot(figsize=(10, 5), title='Mean Acceleration over Sequence Step')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)
plt.show()



# Convert all thm_ columns to a heatmap-like matrix
thm_cols = [col for col in df_train.columns if col.startswith('thm_')]
mean_thm = df_train[thm_cols].mean().values.reshape(1, -1)

sns.heatmap(mean_thm, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=thm_cols)
plt.title("Mean Thermopile Sensor Values")
plt.yticks([])
plt.show()



tof_cols = [col for col in df_train.columns if col.startswith('tof_1_v')]
tof_grid = df_train[tof_cols].iloc[0].values.reshape(8, 8)

sns.heatmap(tof_grid, cmap='viridis', cbar=True)
plt.title("TOF Sensor 1 (Sample Frame)")
plt.show()



plt.figure(figsize=(12, 5))
sns.countplot(x='behavior', data=df_train, order=df_train['behavior'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Behavior Distribution")

plt.figure(figsize=(12, 5))
sns.countplot(x='orientation', data=df_train, order=df_train['orientation'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Orientation Distribution")
plt.show()



sensor_cols = [col for col in df_train.columns if col.startswith(('acc_', 'rot_', 'thm_'))]
corr_matrix = df_train[sensor_cols].corr()

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmax=1, vmin=-1)
plt.title("Sensor Correlation Matrix")
plt.show()



fig = px.scatter_3d(df_train, x='acc_x', y='acc_y', z='acc_z', color='gesture')
fig.update_layout(title="3D Acceleration by Gesture")
fig.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Summary stats for accelerometer, rotation, and thermopile
sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 
               'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_train[sensor_cols])
plt.xticks(rotation=45)
plt.title("Sensor Value Distribution (acc, rot, thm)")
plt.tight_layout()
plt.show()



import numpy as np

for i in range(1, 6):
    tof_cols = [f'tof_{i}_v{j}' for j in range(64)]
    sample = df_train[tof_cols].iloc[0].values.reshape(8, 8)

    plt.figure(figsize=(4, 4))
    sns.heatmap(sample, cmap="viridis", cbar=True)
    plt.title(f"TOF Sensor {i} - Sample Grid")
    plt.axis("off")
    plt.show()



# Include only continuous columns
selected_cols = sensor_cols + [f'thm_{i}' for i in range(1, 6)]
corr = df_train[selected_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
plt.title("Sensor Correlation Matrix")
plt.show()



acc_cols = ['acc_x', 'acc_y', 'acc_z']
df_grouped = df_train.groupby('gesture')[acc_cols].mean().reset_index()

df_grouped.plot(x='gesture', kind='bar', figsize=(10, 5))
plt.title("Mean Acceleration per Gesture")
plt.xticks(rotation=45)
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.tight_layout()
plt.show()



sns.countplot(x='sequence_type', data=df_train)
plt.title("Sequence Type Distribution")
plt.show()


