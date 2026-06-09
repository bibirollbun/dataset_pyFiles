import numpy as np
import pandas as pd 
import random
import os
import cv2
import matplotlib.pyplot as plt
import requests


df=pd.read_csv('/kaggle/input/image-matching-challenge-2025/train_labels.csv')
print(df.columns.tolist())
display(df[0:2].T)
print(len(df))


cols=['dataset','scene','image']
df=df[cols]
df


df['dataset'].value_counts()


def dataset2show16(df):

    names0 = df['dataset'].value_counts()
    valid_names = names0[names0 >= 5].index.tolist()
    names = valid_names
    dir0='/kaggle/input/image-matching-challenge-2025/train'
    selected_images = []
    grouped_images = {}  # Dictionary to store images grouped by name
    for name in names:
        paths = df[df['dataset'] == name]['image'].tolist()

        paths2 = random.sample(paths, 5)  # Select 5 random images per individual
        grouped_images[name] = [os.path.join(dir0,name,p) for p in paths2]
    
    fig, axes = plt.subplots(13, 5, figsize=(14,60))
    
    for row, (nameid, images) in enumerate(grouped_images.items()):
        # Set row title
        axes[row, 0].annotate(nameid, xy=(0, 1), xycoords="axes fraction",fontsize=12, ha="left", va="bottom")
        
        for col, path in enumerate(images):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
                axes[row, col].imshow(img)
            axes[row, col].axis("off")
    
    #plt.tight_layout()
    plt.show()



dataset2show16(df)




