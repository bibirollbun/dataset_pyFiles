import numpy as np
import pandas as pd 
import random
import os
import cv2
import matplotlib.pyplot as plt
import requests


df=pd.read_csv('/kaggle/input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv',sep=',')
print(df.columns.tolist())
display(df[0:2].T)


dir0='/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/train/720p'
cols=['species', 'filename']
df=df[cols]


def dataset2show16(df):

    names0 = df['species'].value_counts()
    valid_names = names0[names0 >= 4].index.tolist()
    names = random.sample(valid_names,16)
    
    selected_images = []
    grouped_images = {}  # Dictionary to store images grouped by name
    for name in names:
        paths = df[df['species'] == name]['filename'].tolist()
        paths2 = random.sample(paths, 4)  # Select 4 random images per individual
        grouped_images[name] = [os.path.join(dir0,p) for p in paths2]
    
    fig, axes = plt.subplots(16, 4, figsize=(10, 40))
    
    for row, (name, images) in enumerate(grouped_images.items()):
        # Set row title
        #axes[row, 0].set_title(name, fontsize=12, loc='left', pad=10)
        axes[row, 0].annotate(name, xy=(0, 1), xycoords="axes fraction",fontsize=12, ha="left", va="bottom")
        
        for col, path in enumerate(images):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
                axes[row, col].imshow(img)
            axes[row, col].axis("off")
    
    #plt.tight_layout()
    plt.show()



dataset2show16(df)




