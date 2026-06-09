from PIL import Image
import glob
import pandas as pd
import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt
!mkdir train
import shutil
from sklearn.model_selection import train_test_split


df=pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
print(df.columns.tolist())
df=df[df['Number of motors']!=0]
display(df)
unique_names = df['tomo_id'].unique().tolist()
print(unique_names[0:3])


df = df.rename(columns={"Array shape (axis 1)": "W", "Array shape (axis 2)": "H",
                        "Motor axis 1":"Yc", "Motor axis 2":"Xc"})
df['xc']=df['Xc']/df['H']
df['yc']=df['Yc']/df['W']
df['width']=10/df['W']
df['height']=10/df['H']
#yolo txt shows class_id x_center y_center width height in order.

df['txt0']=df[['xc','yc','width','height']].apply(lambda row: " ".join(map(str,row)), axis=1)
df['txt']=['0 ']+df['txt0'].astype(str)
display(df[0:2].T)


df.loc[1,"txt"]


dir0='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'
train_dir = "dataset/train/"
val_dir = "dataset/valid/"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)



all_files = []
for name in unique_names:
    dfi = df[df["tomo_id"] == name]
    z_axs = dfi["Motor axis 0"].tolist()
    for zi in z_axs:
        z = int(zi)
        dfii = dfi[dfi["Motor axis 0"] == z]
        label_txt = dfii["txt"]
        pre_path = os.path.join(dir0, name, f"slice_{z:04d}.jpg")
        post_name = f"{name}_{z}"
        all_files.append((pre_path, post_name, label_txt))


train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

for pre_path, post_name, label_txt in train_files:
    if os.path.exists(pre_path):  
        shutil.copy(pre_path, f"{train_dir}{post_name}.jpg")
        label_txt.to_csv(f"{train_dir}{post_name}.txt", index=False, header=False)

for pre_path, post_name, label_txt in val_files:
    if os.path.exists(pre_path):  
        shutil.copy(pre_path, f"{val_dir}{post_name}.jpg")
        label_txt.to_csv(f"{val_dir}{post_name}.txt", index=False, header=False)








