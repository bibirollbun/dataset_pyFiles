import PIL
PIL.Image.open("/kaggle/input/rsna-2024-poster-overview/lumparspine-posteroverview.png")


import warnings
import pydicom
import glob, os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import re
import matplotlib.patches as patches
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import seaborn as sns


path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'


dfm = pd.read_csv(f'{path}/train.csv')
dfm.head()


category_order = ['Normal/Mild', 'Moderate', 'Severe']

figure, axis = plt.subplots(1, 3, figsize=(20, 5)) 
for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):
    diagnosis = list(filter(lambda x: x.find(d) > -1, dfm.columns))
    dff = dfm[diagnosis]
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        value_counts = dff.apply(pd.value_counts).fillna(0).T

    value_counts = value_counts[category_order]
   
    value_counts.plot(kind='bar', stacked=True, ax=axis[idx])
    axis[idx].set_title(f'{d} distribution')


dfc = pd.read_csv(f'{path}/train_label_coordinates.csv')
dfc.head()


folder_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/100206310/1012284084'
dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]

study_id = folder_path.split('/')[-2]
study_label_coordinates = dfc[dfc['study_id'] == int(study_id)]
filtered_dicom_files = []
filtered_label_coordinates = []

for dicom_file in dicom_files:
    instance_number = int(dicom_file.split('.')[0])
    corresponding_coordinates = study_label_coordinates[study_label_coordinates['instance_number'] == instance_number]
    if not corresponding_coordinates.empty:
        filtered_dicom_files.append(dicom_file)
        filtered_label_coordinates.append(corresponding_coordinates)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
second_row_index = 1
second_row_images = filtered_dicom_files[second_row_index : second_row_index + 4]
second_row_coordinates = filtered_label_coordinates[second_row_index : second_row_index + 4]

for i, (dicom_file, label_coordinates) in enumerate(zip(second_row_images, second_row_coordinates)):
    if label_coordinates['condition'].values[0] == 'Spinal Canal Stenosis':
        path_temp = folder_path.replace('1012284084', str(label_coordinates['series_id'].values[0]))
        dicom_file_path = os.path.join(path_temp, dicom_file)
    else:
        dicom_file_path = os.path.join(folder_path, dicom_file)
    dicom_data = pydicom.dcmread(dicom_file_path)
    image = dicom_data.pixel_array   
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(f'DICOM Image - {dicom_file}')
    axs[i].axis('off')   
    for _, row in label_coordinates.iterrows():
        axs[i].plot(row['x'], row['y'], 'ro', markersize=5) 
        
plt.tight_layout()


dfd = pd.read_csv(f'{path}/train_series_descriptions.csv')
dfd.head()


description_count = dfd['series_description'].value_counts().reset_index()
description_count.columns = ['series_description', 'count']

plt.figure(figsize=(8, 6))
sns.barplot(x='count', y='series_description', data=description_count, palette='Set2')
plt.title('Number of IDs by Series Description', fontsize=16)
plt.xlabel('Number of IDs', fontsize=12)
plt.ylabel('Series Description', fontsize=12)
plt.show()


dfc_dfm = pd.merge(left=dfc, right=dfm, how='left', on='study_id').reset_index(drop=True)
merge_df = pd.merge(left=dfc_dfm, right=dfd, how='left', on=['study_id', 'series_id']).reset_index(drop=True)
merge_df.study_id = merge_df.study_id.astype('category')
merge_df.series_id = merge_df.series_id.astype('category')
merge_df.head()


rand_data = np.random.randint(0, 255, (100, 100))
fig, axes = plt.subplots(1, 8, figsize=(15, 3))

axes[0].imshow(rand_data, cmap ='viridis')
axes[1].imshow(rand_data, cmap ='CMRmap')
axes[2].imshow(rand_data, cmap ='binary')
axes[3].imshow(rand_data, cmap ='plasma')
axes[4].imshow(rand_data, cmap ='jet')
axes[5].imshow(rand_data, cmap ='cividis')
axes[6].imshow(rand_data, cmap ='inferno')
axes[7].imshow(rand_data, cmap ='coolwarm')

plt.show()


dfc['series_description'] = merge_df.series_description


def mrt(id, ser, inst):
    lag=20
    path2 = path+'/train_images/' + str(id) +'/' + str(ser)+'/' + str(inst) + '.dcm'

    ds = pydicom.dcmread(path2)
    fig, ax = plt.subplots(figsize=(16, 8))
    from matplotlib.colors import LogNorm 

    ax.imshow(ds.pixel_array, cmap ='inferno')     # Display the image

    # Create a legend
    legend_elements = []

    # Plot the coordinates for the current condition
    ab = dfc[(dfc.study_id==id) & (dfc.instance_number==inst) & (dfc.series_id==ser)]

    a = 25 * max(ds.pixel_array.shape)/640
    for _, row in ab.iterrows():
        x, y = row['x'], row['y']

        rect2 = patches.Rectangle((x - a, y - a), 2*a, 2*a, linewidth=2, edgecolor='white', facecolor='none')
        rect1 = patches.Rectangle((x - a, y - a), 2*a, 2*a, linewidth=2, facecolor='white', alpha = 0.25)

        ax.add_patch(rect2)
        ax.add_patch(rect1)

        # Add the condition to the legend
        legend_elements.append(patches.Patch(facecolor='none', edgecolor='r', ))

    # Add title
    title = f"{ab.series_description.unique()}, Study: {id}, Series: {ser}, Instance: {inst}"
    ax.set_title(title, fontsize=20)

    # Display additional columns:
    for _, row in ab.iterrows():
        text = f"level {row['level']}, {row['condition']}"
        ax.text(row['x'] + lag, row['y']+np.random.randint(-15, 15), text, fontsize=10, color='white', verticalalignment='center_baseline')
    
    plt.show() 
    
def MR3d(id, ser):
    
    path_to_folder = path + "/train_images/"+str(id)+'/'+str(ser)
    def load_dicom(path):
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return data

    rc('animation', html='jshtml')

    def load_dicom(filename):
        ds = pydicom.dcmread(filename)
        return ds.pixel_array

    def load_dicom_line(path):
        t_paths = sorted(
            glob.glob(os.path.join(path, "*")), 
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]),
        )
        images = []
        for filename in t_paths:
            data = load_dicom(filename)
            if data.max() == 0:
                continue
            images.append(data)
        return images

    def create_animation(ims):
        fig = plt.figure(figsize=(6, 6))
        plt.axis('off')
        im = plt.imshow(ims[0], cmap="inferno")
        text = plt.text(0.05, 0.05, f'Slide {1}', transform=fig.transFigure, fontsize=16, color='darkblue')

        def animate_func(i):
            im.set_array(ims[i])
            text.set_text(f'Slide {i+1}')  
            return [im]
        plt.title(f'id = {id}, series = {ser}')
        
        plt.close()  

        return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000//10) #24

    images = load_dicom_line(path_to_folder)
    
    return create_animation(images)


id = 4003253
ser= 702807833
inst=8

mrt(id, ser, inst)


id = 4003253
ser= 2448190387
inst=11

mrt(id, ser, inst)


id_ = 4003253
ser = 702807833

MR3d(id_, ser)


id_ = 100206310
ser = 1012284084

MR3d(id_, ser)

