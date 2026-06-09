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


df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
df.head()



df.info()


df.describe()


df.shape


import pandas as pd



# Calculate average acceleration values
avg_acc = df[['acc_x', 'acc_y', 'acc_z']].mean()
print("Average Acceleration:")
print(avg_acc)



tof_cols = [col for col in df.columns if col.startswith('tof_5_')]
missing_counts = (df[tof_cols] == -1.0).sum()
print("Missing Values in tof_5_* columns:")
print(missing_counts)


most_missing = missing_counts.idxmax()
print(f"Column with most missing values: {most_missing} ({missing_counts[most_missing]})")


unique_sequences = df['sequence_id'].nunique()
unique_subjects = df['subject'].nunique()
print(f"Unique sequences: {unique_sequences}")
print(f"Unique subjects: {unique_subjects}")


import matplotlib.pyplot as plt

# Filter one sequence
sequence_df = df[df['sequence_id'] == 'SEQ_000001']

# Plot
plt.figure(figsize=(10, 5))
plt.plot(sequence_df['sequence_counter'], sequence_df['acc_x'], label='acc_x')
plt.plot(sequence_df['sequence_counter'], sequence_df['acc_y'], label='acc_y')
plt.plot(sequence_df['sequence_counter'], sequence_df['acc_z'], label='acc_z')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration')
plt.title('Acceleration over Time for SEQ_000001')
plt.legend()
plt.grid(True)
plt.show()




