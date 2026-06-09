import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


train=pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_labels.csv')
display(train)


train2=train[train['dataset']=='imc2023_haiper'][train['scene']=='fountain']
display(train2)



# Convert a string to a list of floats
def parse_vector(s):
    return list(map(float, s.split(';')))

# Convert the 'translation_vector' column to a numerical array
vec_array = train2['translation_vector'].apply(parse_vector).tolist()
vec_array = np.array(vec_array)

# Extract each component along the axes
tx, ty, tz = vec_array[:, 0], vec_array[:, 1], vec_array[:, 2]

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tx, ty, tz, color='red', s=20)  # Plot points

# Axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.grid(True)
plt.title('Translation Vector Endpoints')
plt.show()




# Convert a string to a list of floats
def parse_vector(s):
    return list(map(float, s.split(';')))

# Convert columns to arrays
positions = train2['translation_vector'].apply(parse_vector).tolist()
rotations = train2['rotation_matrix'].apply(parse_vector).tolist()

positions = np.array(positions)
rotations = np.array(rotations)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(positions)):
    pos = positions[i]
    R = np.array(rotations[i]).reshape(3, 3)  # 3x3 rotation matrix

    # Combine the x, y, z direction vectors
    combined_vector = R[:, 0] + R[:, 1] + R[:, 2]
    combined_vector = combined_vector / np.linalg.norm(combined_vector)  # Normalize (optional)
    scale = 0.3
    ax.quiver(*pos, *(combined_vector * scale), color='purple')

    # Plot the camera position
    ax.scatter(*pos, color='black', s=10)

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)
plt.title('Combined Rotation Vectors from Camera Poses')
plt.show()




paths=[]
for dirname, _, filenames in os.walk('/kaggle/input/image-matching-challenge-2025/train/imc2023_haiper'):
    for filename in filenames:
        if filename.split('_')[0]=='fountain':
            paths+=[(os.path.join(dirname, filename))]


num_images = len(paths)
cols = 3
rows = math.ceil(num_images / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

for i, path in enumerate(paths):
    img = Image.open(path)
    row = i // cols
    col = i % cols
    axes[row, col].imshow(img)
    axes[row, col].axis('off')

for j in range(num_images, rows * cols):
    row = j // cols
    col = j % cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()





