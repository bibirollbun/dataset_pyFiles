# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#data viz
import matplotlib.pyplot as plt
import seaborn as sns 

# avoiding warnings
import warnings 
warnings.filterwarnings('ignore')

from PIL import Image
import cv2 
import tensorflow 
from tensorflow import keras 
from keras import Sequential
from keras.layers import Conv2D , Dense , Softmax , Flatten , MaxPooling2D


metadata = pd.read_csv("/kaggle/input/animal-clef-2025/metadata.csv")
submission = pd.read_csv("/kaggle/input/animal-clef-2025/sample_submission.csv")


data = {
    "metadata": metadata,
    "submission_sample": submission
}

for name , df in data.items():
    print(f"--{name}--")
    print(f"shape : {df.shape}")
    print(f"features : {df.columns}")
    display(df.head())
    print("\n" + "__"*42)


# missing info 
metadata.isnull().sum()


print("__DataBase__")
display(metadata[metadata["split"] == "database"].isna().sum())
print("\n")
print("__Querry__")
display(metadata[metadata["split"] == "query"].isna().sum())


#By Jocelyn Dumlao
# Directories for each animal category
data_dirs = {    
    "SeaTurtlesD": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/database/turtles-data/data/images/t001",
    "SeaTurtlesQ": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/query/images",
    "LynxsD": "/kaggle/input/animal-clef-2025/images/LynxID2025/database",
    "LynxsQ": "/kaggle/input/animal-clef-2025/images/LynxID2025/query",
    "SalamandersD": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/database/images",
    "SalamandersQ": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/query/images"       
}


# image preprocessing 

def show_sample_images():
    for label, dir_path in data_dirs.items():
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        # Load the first few images from the directory
        sample_images = []
        for root, _, files in os.walk(dir_path):
            for img_name in files[:5]:  # Load 5 images for display
                img_path = os.path.join(root, img_name)
                if img_path.lower().endswith(('.JPG', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (150, 150))
                        sample_images.append(img)
                    if len(sample_images) == 5:
                        break
            if len(sample_images) == 5:
                break
                
        # Plot the images
        plt.figure(figsize=(10, 10))
        for i, img in enumerate(sample_images):
            plt.subplot(1, 5, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"{label}")
        plt.show()

# Call the function to display images
show_sample_images()










