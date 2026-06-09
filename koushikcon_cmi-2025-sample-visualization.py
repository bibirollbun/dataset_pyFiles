# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter)


base_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

train = pd.read_csv(f"{base_path}/train.csv")
train_demo = pd.read_csv(f"{base_path}/train_demographics.csv")


# Display basic information
print("Train Shape:", train.shape)
print("Train Demographics Shape:", train_demo.shape)
print(train.info())
print(train_demo.info())


# Unique values in key columns
print("Unique gestures:", train['gesture'].unique())


print("Unique behaviors:", train['behavior'].unique())


print("Unique orientations:", train['orientation'].unique())


import matplotlib.pyplot as plt

# Select a sequence
example_seq_id = train['sequence_id'].unique()[0]
example_seq = train[train['sequence_id'] == example_seq_id]

# Plot accelerometer data
plt.plot(example_seq['sequence_counter'], example_seq['acc_x'], label='acc_x')
plt.plot(example_seq['sequence_counter'], example_seq['acc_y'], label='acc_y')
plt.plot(example_seq['sequence_counter'], example_seq['acc_z'], label='acc_z')
plt.title(f"Accelerometer Data for Sequence {example_seq_id}")
plt.xlabel("Sequence Step")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.show()


import seaborn as sns

# Identify ToF columns
tof_cols = [col for col in train.columns if col.startswith('tof_')]

# Calculate percentage of missing values
missing_tof = (train[tof_cols] == -1).mean().sort_values(ascending=False) * 100

# Plot missing data
missing_tof.head(30).plot(kind='bar', title="Top ToF Pixels with Missing Signals (-1)")
plt.ylabel("% Missing")
plt.show()


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler

# Aggregate features per sequence
agg_features = train.groupby("sequence_id").agg({
    'acc_x': 'mean',
    'acc_y': 'mean',
    'acc_z': 'mean',
    'rot_x': 'mean',
    'rot_y': 'mean',
    'rot_z': 'mean',
    'thm_1': 'mean',
    'thm_2': 'mean',
    'thm_3': 'mean',
    'thm_4': 'mean',
    'thm_5': 'mean'
}).reset_index()

# Add gesture labels
gesture_labels = train.groupby("sequence_id")["gesture"].first().reset_index()
agg_features = agg_features.merge(gesture_labels, on="sequence_id")

# Drop rows with any NaN values
agg_features_clean = agg_features.dropna()

# Dimensionality reduction with UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_umap = reducer.fit_transform(agg_features_clean.drop(["sequence_id", "gesture"], axis=1))

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=embedding_umap[:, 0],
    y=embedding_umap[:, 1],
    hue=agg_features_clean["gesture"],
    palette="tab20",
    s=50
)
plt.title("UMAP projection of aggregated sensor features by gesture")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


