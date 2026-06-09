import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.patches as patches
import numpy as np
import math

labels_df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv', index_col=0)
labels_df = labels_df[labels_df['Motor axis 0'] > 0] # filtering out only cells with motors

labels_df['file_name'] = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/' + \
                        labels_df['tomo_id'] + '/slice_' + \
                        labels_df['Motor axis 0'].astype(int).astype(str).str.zfill(4) + '.jpg'

num_images = 450
images_per_row = 8

num_rows = math.ceil(num_images / images_per_row)
fig, axes = plt.subplots(num_rows, images_per_row, figsize=(12, 1.5 * num_rows))
axes = axes.flatten() if num_rows > 1 else [axes] if num_images == 1 else axes

for i, (idx, row) in enumerate(labels_df[:num_images].iterrows()):
    if i < len(axes):
        path = row['file_name']
        x_coord = row['Motor axis 2']
        y_coord = row['Motor axis 1']

        img = mpimg.imread(path)
        axes[i].imshow(img)

        circle = patches.Circle((x_coord, y_coord), radius=40, 
                               edgecolor='red', facecolor='none', linewidth=1)
        axes[i].add_patch(circle)
        
        #axes[i].set_title(f"{row['tomo_id']}\nSlice {int(row['Motor axis 0'])}", fontsize=8) 

        axes[i].set_xticks([])
        axes[i].set_yticks([])

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('motor_locations_grid.png', dpi=50, bbox_inches='tight')
plt.show()

