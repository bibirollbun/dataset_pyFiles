!pip install -q mediapy



%cd /kaggle/working/
!rm -rf /kaggle/working/Hierarchical-Localization
!git clone --quiet --recursive https://github.com/cvg/Hierarchical-Localization/
%cd /kaggle/working/Hierarchical-Localization
!pip install -e .

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

%cd /kaggle/working/


from pathlib import Path

import cv2
import mediapy
import pandas as pd
import plotly.express as px
import pycolmap
import os
import random
from PIL import Image
import matplotlib.pyplot as plt


train_labels = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_labels.csv")
train_labels


train_labels.groupby("dataset")["scene"].nunique()



dataset_counts = train_labels["dataset"].value_counts()

plt.figure(figsize=(12, 7))
dataset_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Images Across Datasets', fontsize=16, fontweight='bold')
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=12)  
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


def plot_random_images(scene_name, base_path="/kaggle/input/image-matching-challenge-2025/train"):
    scene_path = os.path.join(base_path, scene_name)
    image_filenames = [f for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random_images = random.sample(image_filenames, min(5, len(image_filenames)))
    fig, axes = plt.subplots(1, len(random_images), figsize=(15, 5))
    if len(random_images) == 1:
        axes = [axes]

    for ax, img_filename in zip(axes, random_images):
        img_path = os.path.join(scene_path, img_filename)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(img_filename, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_random_images("stairs")



plot_random_images("pt_stpeters_stpauls")



plot_random_images("pt_sacrecoeur_trevi_tajmahal")



plot_random_images("pt_piazzasanmarco_grandplace")



plot_random_images("pt_brandenburg_british_buckingham")



plot_random_images("imc2024_lizard_pond")



plot_random_images("imc2024_dioscuri_baalshamin")



plot_random_images("imc2023_theather_imc2024_church")



plot_random_images("imc2023_heritage")



plot_random_images("imc2023_haiper")



plot_random_images("fbk_vineyard")



plot_random_images("amy_gardens")



plot_random_images("ETs")





