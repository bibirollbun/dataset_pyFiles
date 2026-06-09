import pandas as pd

# Load the training labels
train_labels = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')

# Count unique tomograms
unique_tomograms = train_labels['tomo_id'].nunique()

# Motors per tomogram
motors_per_tomo = train_labels.groupby('tomo_id')['Number of motors'].first()
motors_per_tomo.describe()


import matplotlib.pyplot as plt

# Plot histograms for motor coordinates
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(train_labels['Motor axis 0'], bins=50, color='blue', alpha=0.7)
axes[0].set_title('Distribution of Z-coordinates')
axes[0].set_xlabel('Z-coordinate')
axes[0].set_ylabel('Frequency')

axes[1].hist(train_labels['Motor axis 1'], bins=50, color='green', alpha=0.7)
axes[1].set_title('Distribution of Y-coordinates')
axes[1].set_xlabel('Y-coordinate')

axes[2].hist(train_labels['Motor axis 2'], bins=50, color='red', alpha=0.7)
axes[2].set_title('Distribution of X-coordinates')
axes[2].set_xlabel('X-coordinate')

plt.tight_layout()
plt.show()


# Plot histogram of voxel spacing
plt.hist(train_labels['Voxel spacing'], bins=50, color='purple', alpha=0.7)
plt.title('Distribution of Voxel Spacing')
plt.xlabel('Voxel Spacing (angstroms per voxel)')
plt.ylabel('Frequency')
plt.show()


# Plot histograms for tomogram dimensions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(train_labels['Array shape (axis 0)'], bins=50, color='orange', alpha=0.7)
axes[0].set_title('Distribution of Z-axis Length')
axes[0].set_xlabel('Z-axis Length')

axes[1].hist(train_labels['Array shape (axis 1)'], bins=50, color='cyan', alpha=0.7)
axes[1].set_title('Distribution of Y-axis Length')
axes[1].set_xlabel('Y-axis Length')

axes[2].hist(train_labels['Array shape (axis 2)'], bins=50, color='magenta', alpha=0.7)
axes[2].set_title('Distribution of X-axis Length')
axes[2].set_xlabel('X-axis Length')

plt.tight_layout()
plt.show()

