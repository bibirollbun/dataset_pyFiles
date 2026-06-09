import numpy as np
import pandas as pd 
import random
import os
import cv2
import matplotlib.pyplot as plt
import requests


df=pd.read_csv('/kaggle/input/plantclef-2025/PlantCLEF2024_single_plant_training_metadata.csv',sep=';')
print(df.columns.tolist())
display(df[0:2].T)


cols=['species_id','species', 'image_backup_url']
df=df[cols]
df2=df[['species_id','species']].drop_duplicates()
id2name = df2.set_index('species_id')['species'].to_dict()


def dataset2show16(df):

    names0 = df['species_id'].value_counts()
    valid_names = names0[names0 >= 4].index.tolist()
    names = random.sample(valid_names,16)
    
    selected_images = []
    grouped_images = {}  # Dictionary to store images grouped by name
    for name in names:
        paths = df[df['species_id'] == name]['image_backup_url'].tolist()
        paths2 = random.sample(paths, 4)  # Select 4 random images per individual
        grouped_images[name] = [p for p in paths2]
    
    fig, axes = plt.subplots(16, 4, figsize=(10, 40))
    
    for row, (nameid, images) in enumerate(grouped_images.items()):
        # Set row title
        #axes[row, 0].set_title(str(nameid)+' '+id2name[nameid], fontsize=12, loc='left', pad=10)
        axes[row, 0].annotate(id2name[nameid], xy=(0, 1), xycoords="axes fraction",fontsize=12, ha="left", va="bottom")
        
        for col, img_url in enumerate(images):
            response = requests.get(img_url)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
                axes[row, col].imshow(img)
            axes[row, col].axis("off")
    
    #plt.tight_layout()
    plt.show()



dataset2show16(df)




