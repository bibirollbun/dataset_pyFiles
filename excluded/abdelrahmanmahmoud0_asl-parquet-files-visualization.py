import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

warnings.filterwarnings("ignore")


def extract_xyz_magnitude(data, columns):
    coord_groups = {}
    for col in columns:
        match = re.match(r"(x|y|z)_(.*)_(\d+)", col)
        if match:
            axis, _, idx = match.groups()
            idx = int(idx)
            coord_groups.setdefault(idx, {})[axis] = col

    magnitudes = []
    for idx in sorted(coord_groups):
        group = coord_groups[idx]
        if all(k in group for k in ['x', 'y', 'z']):
            x = data[group['x']].fillna(0).values
            y = data[group['y']].fillna(0).values
            z = data[group['z']].fillna(0).values
            mag = np.sqrt(x**2 + y**2 + z**2)
            magnitudes.append(mag)

    return np.array(magnitudes) if magnitudes else None

def normalize_matrix(mat):
    norm_mat = mat.copy()
    for i in range(norm_mat.shape[0]):
        row = norm_mat[i]
        if np.max(row) > 0:
            norm_mat[i] = (row - np.min(row)) / (np.max(row) - np.min(row))
    return norm_mat

def plot_component(name, matrix):
    print(name)
    plt.figure(figsize=(8, 3))
    plt.imshow(matrix, aspect='auto', cmap='inferno', vmin=0, vmax=np.percentile(matrix, 99))
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    plt.savefig(f'{name}_figure.png')
    plt.show()


base = '/kaggle/input/asl-fingerspelling/train_landmarks'
pq_paths = os.listdir(base)
idx = np.random.randint(0, len(pq_paths))

path = os.path.join(base, pq_paths[idx])
data = pd.read_parquet(path)



face_columns = [col for col in data.columns if 'face' in col]
pose_columns = [col for col in data.columns if 'pose' in col]
left_hand_columns = [col for col in data.columns if 'left_hand' in col]
right_hand_columns = [col for col in data.columns if 'right_hand' in col]

components = {
    "face": face_columns,
    "pose": pose_columns,
    "left_hand": left_hand_columns,
    "right_hand": right_hand_columns,
}


component_matrices = {}
for name, cols in components.items():
    mag = extract_xyz_magnitude(data, cols)
    if mag is not None and mag.size > 0:
        norm_mag = normalize_matrix(mag)
        component_matrices[name] = norm_mag
        plot_component(name, norm_mag)
    else:
        print(f"⚠️ Skipping '{name}': no valid landmarks found.")

gap_size = 5
block_offsets = {}
current_offset = 0
total_rows = sum(mat.shape[0] + gap_size for mat in component_matrices.values())

num_frames = next(iter(component_matrices.values())).shape[1]
combined_matrix = np.zeros((total_rows, num_frames))

for name, mat in component_matrices.items():
    block_offsets[name] = current_offset
    combined_matrix[current_offset:current_offset + mat.shape[0], :] = mat
    current_offset += mat.shape[0] + gap_size

plt.figure(figsize=(8, 8))
plt.imshow(combined_matrix, aspect='auto', cmap='inferno', vmin=0, vmax=np.percentile(combined_matrix, 99))
plt.xlabel("Time")
plt.ylabel("Landmark idx")

for name, offset in block_offsets.items():
    plt.text(5, offset + 5, name, color='white', fontsize=10, va='top')

plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('combined_components_figure.png')
plt.show()

