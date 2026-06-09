%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_flagellar_motors(base_path, tomo_id=None, alpha=0.3, downscale=2):
    tomograms = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    target_dir = os.path.join(base_path, tomo_id if tomo_id else tomograms[0])

    slices = sorted([f for f in os.listdir(target_dir) if f.endswith('.jpg')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))  # 解析slice_0000格式

    first_slice = Image.open(os.path.join(target_dir, slices[0]))
    original_size = first_slice.size  # (width, height)
    first_slice.close()
    
    new_size = (original_size[0]//downscale, original_size[1]//downscale)
    
    z_center_idx = len(slices) // 2
    z_center_slice = np.array(Image.open(os.path.join(target_dir, slices[z_center_idx]))
                             .convert('L').resize(new_size))
    
    max_projection = np.zeros(new_size[::-1], dtype=np.uint8)  # (height, width)
    sum_projection = np.zeros(new_size[::-1], dtype=np.float32)
    
    for i, f in enumerate(slices):
        img = Image.open(os.path.join(target_dir, f)).convert('L').resize(new_size)
        slice_data = np.array(img)
        img.close()
        
        np.maximum(max_projection, slice_data, out=max_projection)
        sum_projection += slice_data.astype(np.float32)
        
        del slice_data
    
    avg_projection = (sum_projection / len(slices)).astype(np.uint8)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(z_center_slice, cmap='gray')
    axs[0].set_title("Center Slice")
    
    axs[1].imshow(max_projection, cmap='gray')
    axs[1].set_title("Max Projection")
    
    axs[2].imshow(avg_projection, cmap='gray')
    axs[2].set_title("Average Projection")
    
    plt.tight_layout()
    #plt.close()
    return #fig

visualize_flagellar_motors('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/', tomo_id='tomo_00e047', downscale=4)


import os
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def interactive_3d_visualization(base_path, tomo_id, label_path='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv', downscale=8):
    labels = pd.read_csv(label_path)
    tomo_labels = labels[labels['tomo_id'] == tomo_id]
    
    target_dir = os.path.join(base_path, tomo_id)
    slices = sorted([f for f in os.listdir(target_dir) if f.endswith('.jpg')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    first_slice = Image.open(os.path.join(target_dir, slices[0]))
    original_size = first_slice.size
    new_size = (original_size[0]//downscale, original_size[1]//downscale)
    first_slice.close()
    
    volume = np.zeros((len(slices), new_size[1], new_size[0]), dtype=np.uint8)
    
    for i, f in enumerate(slices):
        img = Image.open(os.path.join(target_dir, f)).convert('L').resize(new_size)
        volume[i] = np.array(img)
        img.close()
    
    fig = go.Figure()
    
    fig.add_trace(go.Volume(
        x=volume.shape[2]*np.ones(volume.size),
        y=np.repeat(np.arange(volume.shape[1]), volume.shape[0]*volume.shape[2]),
        z=np.tile(np.repeat(np.arange(volume.shape[0]), volume.shape[2]), volume.shape[1]),
        value=volume.flatten(),
        isomin=np.percentile(volume, 50),
        isomax=np.percentile(volume, 99),
        opacity=0.05,
        surface_count=15,
        colorscale='gray'
    ))
    
    if not tomo_labels.empty:
        for _, row in tomo_labels.iterrows():
            z = row['Motor axis 0'] / downscale
            y = row['Motor axis 1'] / downscale
            x = row['Motor axis 2'] / downscale
            
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    opacity=0.8
                ),
                name=f'Motor ({x:.1f}, {y:.1f}, {z:.1f})'
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=640
    )
    
    return fig

tomo_id = 'tomo_00e047'
fig = interactive_3d_visualization('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/', tomo_id)
fig.show()


import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_motor_slices(base_path, tomo_id, label_path='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv', z_range=5, box_size=20):
    labels = pd.read_csv(label_path)
    tomo_labels = labels[labels['tomo_id'] == tomo_id]

    target_dir = os.path.join(base_path, tomo_id)
    slices = sorted([f for f in os.listdir(target_dir) if f.endswith('.jpg')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    sample_slice = np.array(Image.open(os.path.join(target_dir, slices[0])))
    slice_shape = sample_slice.shape
    num_slices = len(slices)
    
    combined_image = np.zeros(slice_shape, dtype=np.float32)
    motor_boxes = []

    for _, motor in tomo_labels.iterrows():
        z_center = int(motor['Motor axis 0'])
        y_center = int(motor['Motor axis 1'])
        x_center = int(motor['Motor axis 2'])
        
        z_start = max(0, z_center - z_range)
        z_end = min(num_slices, z_center + z_range + 1)
        
        slice_count = 0
        for z in range(z_start, z_end):
            img = np.array(Image.open(os.path.join(target_dir, slices[z])).convert('L'))
            combined_image += img.astype(np.float32)
            slice_count += 1
        
        motor_boxes.append((
            x_center - box_size//2,  # x_min
            y_center - box_size//2,  # y_min
            box_size,                # width
            box_size,                # height
            f"Motor@{z_center}"      # label
        ))

        if slice_count > 0:
            combined_image /= slice_count
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(combined_image, cmap='gray')
    
    for box in motor_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(
            box[0], box[1]-5, box[4],
            color='red', fontsize=10, weight='bold'
        )
    
    plt.title(f"Tomogram {tomo_id}\nMotor Positions (Red Boxes)")
    plt.axis('off')
    plt.show()

visualize_motor_slices('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/', 'tomo_00e047', z_range=3, box_size=30)




