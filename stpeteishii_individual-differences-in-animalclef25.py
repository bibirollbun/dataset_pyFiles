import numpy as np
import pandas as pd 
import random
import os
import cv2
import matplotlib.pyplot as plt


df=pd.read_csv('/kaggle/input/animal-clef-2025/metadata.csv')
print(df.columns.tolist())
display(df[0:2].T)
print(df['dataset'].unique().tolist())
cols=['identity', 'path',  'dataset']
df=df[cols]
dir0='/kaggle/input/animal-clef-2025'


def dataset2show16(df0,dataset):
    
    df=df0[df0['dataset']==dataset]
    names0 = df['identity'].value_counts()
    valid_names = names0[names0 >= 4].index.tolist()
    names = random.sample(valid_names, 8)
    
    selected_images = []
    grouped_images = {}  # Dictionary to store images grouped by name
    for name in names:
        paths = df[df['identity'] == name]['path'].tolist()
        paths2 = random.sample(paths, 4)  # Select 4 random images per individual
        grouped_images[name] = [os.path.join(dir0, p) for p in paths2]
    
    fig, axes = plt.subplots(8, 4, figsize=(10, 20))
    
    for row, (name, images) in enumerate(grouped_images.items()):
        # Set row title
        axes[row, 0].set_title(name, fontsize=12, loc='left', pad=10)
        
        for col, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
                axes[row, col].imshow(img)
            axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.show()



dataset2show16(df,'LynxID2025')


dataset2show16(df,'SalamanderID2025')


dataset2show16(df,'SeaTurtleID2022')

