import pandas as pd
import os
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import math
from matplotlib import animation, rc
import cv2


df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
print('Number of tomograms in train set = ', len(set(df['tomo_id'])))
df


df.describe()


# Unique shapes of tomograms available in the train set
df[['Array shape (axis 0)',	'Array shape (axis 1)',	'Array shape (axis 2)']].drop_duplicates().reset_index(drop=True)


def plot_numeric_histograms(df):
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_plots = len(numeric_cols)
    rows = math.ceil(num_plots / 2)
    
    fig = sp.make_subplots(rows=rows, cols=2, subplot_titles=numeric_cols)
    
    for i, col in enumerate(numeric_cols):
        row = (i // 2) + 1
        col_pos = (i % 2) + 1
        fig.add_trace(go.Histogram(x=df[col], name=col, nbinsx=30), row=row, col=col_pos)
    
    fig.update_layout(title_text="Histograms of Numeric Columns", height=rows * 300, showlegend=False)
    fig.show(renderer='iframe')

plot_numeric_histograms(df)


print('Total number of motors in train set = ',len(df[df['Motor axis 0']!=-1]))
print('Total number of tomograms which have motors = ', len(set(df[df['Motor axis 0']!=-1]['tomo_id'])))


def get_tomogram(directory_path, experiment_name, scaling='None'):
    
    path = os.path.join(directory_path, experiment_name)
    file_list = os.listdir(path)
    file_list = sorted(file_list, key=lambda x: x)
    arrays = []
    
    for file in file_list:
        arr = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        arrays.append(arr)

    tomogram = np.stack(arrays, axis=0)
    scaling = str.lower(scaling)

    if scaling == 'z':
        tomogram = (tomogram-np.mean(tomogram))/(np.maximum(np.std(tomogram),1e-6))

    elif scaling == 'max':
        tomogram = ((tomogram-tomogram.min())/(tomogram.max()-tomogram.min()))

    return tomogram
    
def view_slice(tomogram, z):
    
    image = tomogram[z]
    plt.imshow(image, cmap="gray")
    plt.show()

def view_slice_circle(tomogram, z, y, x, voxel_spacing, thickness=2, ax=None):
    
    x, y, z = int(x), int(y), int(z)
    center = (x, y)
    radius = int(1000 / voxel_spacing)

    if all(v >= 0 for v in [x, y, z]):
        image = tomogram[z].copy()
        color = int(np.max(image))
        cv2.rectangle(image, (x-radius,y-radius), (x+radius,y+radius), color, thickness)  
        cv2.putText(image, f'z={z} y={y} x={x}', (15,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness, cv2.LINE_AA)
        
        if ax is None:
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap="gray")
            plt.show()
        else:
            ax.imshow(image, cmap="gray")
    else:
        print('There is no flagellum in this tomogram. Use "view_slice" if you want to see a specific slice of this tomogram.')

def plot_all_flagellum(tomogram, experiment, data):
    
    exp_data = data[data['tomo_id'] == experiment].copy()
    num_images = len(exp_data)
    cols = 3
    rows = (num_images + cols - 1) // cols 
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(rows, cols) 
    
    for idx, (i, row) in enumerate(exp_data.iterrows()):
        z, y, x, spacing = row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2'], row['Voxel spacing']
        ax = axes[idx // cols, idx % cols]
        view_slice_circle(tomogram, z, y, x, spacing, thickness=2, ax=ax)
        ax.set_title(f"Flagellum {idx}")

    for idx in range(num_images, rows * cols):
        fig.delaxes(axes[idx // cols, idx % cols])

    plt.tight_layout()
    plt.show()


directory = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'

for exp in sorted(set(df[df['Motor axis 0']!=-1]['tomo_id'])):
    
    tom = get_tomogram(directory, exp, scaling='None')
    print('*'*80)
    print(exp, tom.shape)
    plot_all_flagellum(tom, exp, df)




