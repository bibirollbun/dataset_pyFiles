import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
import torch
import warnings
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
from scipy.ndimage import gaussian_filter


INPUT_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
TRAIN_JPG = os.path.join(INPUT_DIR, "train")
TRAIN_LABELS = os.path.join(INPUT_DIR, "train_labels.csv")
TEST_JPG = os.path.join(INPUT_DIR, "test")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file containing motor labels
labels_df = pd.read_csv(TRAIN_LABELS)
display(labels_df.head())

# Plot: Number of motors per tomogram
motor_counts = labels_df.groupby('tomo_id').size().reset_index(name='motor_count')
plt.figure(figsize=(10, 4))
sns.histplot(motor_counts['motor_count'], bins=20, kde=False)
plt.title("Distribution of motors per tomogram")
plt.xlabel("Number of motors")
plt.ylabel("Number of tomograms")
plt.show()

# Load a limited number of slices from a given tomogram as a 3D volume
def load_volume(tomo_id, max_slices=200):
    folder = os.path.join(TRAIN_JPG, tomo_id)
    slices = sorted(os.listdir(folder))[:max_slices]
    volume = [imageio.imread(os.path.join(folder, s)) for s in slices]
    return np.stack(volume)

# Select a sample tomogram with at least one motor
tomo_sample = labels_df[labels_df['Number of motors'] > 0]['tomo_id'].iloc[0]
volume = load_volume(tomo_sample)
print(f"Loaded volume shape: {volume.shape}")

# Display the middle slice of the volume
plt.imshow(volume[volume.shape[0] // 2], cmap='gray')
plt.title(f"Middle slice of {tomo_sample}")
plt.axis('off')
plt.show()

# Extract and plot motor coordinates in 3D
coords = labels_df[labels_df['tomo_id'] == tomo_sample][['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
coords = np.array([c for c in coords if np.all(np.isfinite(c)) and np.all(np.array(c) >= 0)])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:,2], coords[:,1], coords[:,0], c='r', marker='o')
ax.set_title(f"Motor positions in {tomo_sample}")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis (slice index)')
plt.show()

# Generate 3D heatmap from motor positions
def create_heatmap(volume_shape, coordinates, sigma=3):
    heatmap = np.zeros(volume_shape, dtype=np.float32)
    for z, y, x in coordinates:
        z, y, x = int(z), int(y), int(x)
        if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
            heatmap[z, y, x] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap

heatmap = create_heatmap(volume.shape, coords)

# Visualize several slices of the heatmap around the strongest activation
max_z = np.argmax(np.max(np.max(heatmap, axis=1), axis=1))
slice_range = range(max(0, max_z - 3), min(volume.shape[0], max_z + 3))

fig, axs = plt.subplots(2, len(slice_range), figsize=(18, 6))
for i, z in enumerate(slice_range):
    axs[0, i].imshow(volume[z], cmap='gray')
    axs[0, i].set_title(f"Volume Slice z={z}")
    axs[0, i].axis('off')

    axs[1, i].imshow(heatmap[z], cmap='hot')
    axs[1, i].set_title(f"Heatmap Slice z={z}")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()

# Summary statistics of the dataset
data_summary = labels_df.copy()
data_summary['has_motor'] = data_summary['Motor axis 0'] >= 0
summary = data_summary.groupby('tomo_id').agg(
    number_of_motors=('has_motor', 'sum'),
    z_slices=('Array shape (axis 0)', 'first'),
    height=('Array shape (axis 1)', 'first'),
    width=('Array shape (axis 2)', 'first'),
    voxel_spacing=('Voxel spacing', 'first')
).reset_index()

print("Summary of Tomograms:")
display(summary.head())

# Additional analysis: pixel size statistics
summary['pixel_area'] = summary['height'] * summary['width']
mean_pixel_size = summary['pixel_area'].mean()
min_pixel_size = summary['pixel_area'].min()
max_pixel_size = summary['pixel_area'].max()

min_shape = summary.loc[summary['pixel_area'].idxmin(), ['height', 'width']].values
max_shape = summary.loc[summary['pixel_area'].idxmax(), ['height', 'width']].values

print("\nPixel size analysis:")
print(f"Average pixel area: {mean_pixel_size:.2f} (in pixels)")
print(f"Smallest image: {min_shape[0]}x{min_shape[1]} (area={min_pixel_size})")
print(f"Largest image: {max_shape[0]}x{max_shape[1]} (area={max_pixel_size})")

# Zoom-in images around motor positions
sample_tomos = labels_df[labels_df['Number of motors'] > 0]['tomo_id'].unique()[:6]
fig, axs = plt.subplots(1, 6, figsize=(20, 4))
for i, tomo_id in enumerate(sample_tomos):
    volume = load_volume(tomo_id)
    motor_pos = labels_df[labels_df['tomo_id'] == tomo_id][['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values[0]
    z, y, x = int(motor_pos[0]), int(motor_pos[1]), int(motor_pos[2])
    z = np.clip(z, 0, volume.shape[0]-1)
    y1, y2 = max(0, y-32), min(volume.shape[1], y+32)
    x1, x2 = max(0, x-32), min(volume.shape[2], x+32)
    zoom = volume[z, y1:y2, x1:x2]
    axs[i].imshow(zoom, cmap='gray')
    axs[i].set_title(f"{tomo_id}\nz={z}")
    axs[i].axis('off')

plt.tight_layout()
plt.show()


# Balance analysis: tomograms with vs. without motors
labels_df['has_motor'] = labels_df['Motor axis 0'] >= 0
balance_df = labels_df.groupby('tomo_id')['has_motor'].any().value_counts().rename(index={True: 'Has Motor', False: 'No Motor'}).reset_index()
balance_df.columns = ['Class', 'Count']
display(balance_df)

plt.figure(figsize=(6, 4))
sns.barplot(x='Class', y='Count', data=balance_df)
plt.title("Balance of Tomograms (with vs. without motors)")
plt.ylabel("Number of tomograms")
plt.xlabel("Class")
plt.show()

# Voxel spacing distribution and implications
plt.figure(figsize=(6, 4))
sns.histplot(summary['voxel_spacing'], bins=30)
plt.title("Voxel spacing distribution")
plt.xlabel("Angstrom per voxel")
plt.ylabel("Count")
plt.show()

print("Voxel spacing range:")
print(f"Min: {summary['voxel_spacing'].min()} Å, Max: {summary['voxel_spacing'].max()} Å")

# Volume size statistics (for memory planning)
summary['volume_size'] = summary['z_slices'] * summary['height'] * summary['width']
plt.figure(figsize=(6, 4))
sns.histplot(summary['volume_size'] / 1e6, bins=30)
plt.title("Distribution of volume sizes")
plt.xlabel("Volume size (in millions of voxels)")
plt.ylabel("Number of tomograms")
plt.show()

print("Volume size stats (in voxels):")
print(f"Min: {summary['volume_size'].min()} | Max: {summary['volume_size'].max()} | Avg: {summary['volume_size'].mean():.2f}")
print("Voxel spacing range:")
print(f"Min: {summary['voxel_spacing'].min()} Å, Max: {summary['voxel_spacing'].max()} Å")

# Volume size statistics (for memory planning)
summary['volume_size'] = summary['z_slices'] * summary['height'] * summary['width']
plt.figure(figsize=(6, 4))
sns.histplot(summary['volume_size'] / 1e6, bins=30)
plt.title("Distribution of volume sizes")
plt.xlabel("Volume size (in millions of voxels)")
plt.ylabel("Number of tomograms")
plt.show()

print("Volume size stats (in voxels):")
print(f"Min: {summary['volume_size'].min()} | Max: {summary['volume_size'].max()} | Avg: {summary['volume_size'].mean():.2f}")

...

# 3D Volume Preview using Plotly
import plotly.graph_objects as go

fig = go.Figure(data=go.Volume(
    x=np.repeat(np.arange(volume.shape[2]), volume.shape[0] * volume.shape[1]),
    y=np.tile(np.repeat(np.arange(volume.shape[1]), volume.shape[2]), volume.shape[0]),
    z=np.tile(np.arange(volume.shape[0]), volume.shape[1] * volume.shape[2]),
    value=volume.flatten(),
    opacity=0.1,
    surface_count=15,
    colorscale='Gray'
))
fig.update_layout(title='3D Volume Preview (Plotly)', scene=dict(zaxis_title='Z', yaxis_title='Y', xaxis_title='X'))
fig.show()

# Distribution of motors along Z axis
motor_z = labels_df[labels_df['Motor axis 0'] >= 0]['Motor axis 0']
plt.figure(figsize=(8, 4))
sns.histplot(motor_z, bins=50)
plt.title("Distribution of motors along Z-axis")
plt.xlabel("Z (Slice index)")
plt.ylabel("Count")
plt.show()

# Analysis: Number of motors vs tomogram size
summary['motors_per_million_voxels'] = summary['number_of_motors'] / (summary['volume_size'] / 1e6)
plt.figure(figsize=(6, 4))
sns.scatterplot(x='volume_size', y='number_of_motors', data=summary)
plt.title("Number of motors vs. volume size")
plt.xlabel("Volume size (voxels)")
plt.ylabel("Number of motors")
plt.show()


