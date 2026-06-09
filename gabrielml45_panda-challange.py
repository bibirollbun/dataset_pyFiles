!pip install openslide-python --quiet

import os
import openslide
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

BASE_PATH = "/kaggle/input/prostate-cancer-grade-assessment/train_images"
MASK_PATH = "/kaggle/input/prostate-cancer-grade-assessment/train_label_masks"

# Lista todos os arquivos e ordena pelo nome
all_files = sorted(os.listdir(BASE_PATH))

# Seleciona as imagens do índice 100 até 105
image_files = all_files[100:106]  # 106 porque o slice não inclui o último índice

def show_image_and_mask(file_name, size=(512,512)):
    # Abre a imagem
    img_path = os.path.join(BASE_PATH, file_name)
    slide = openslide.OpenSlide(img_path)
    thumbnail = slide.get_thumbnail(size).convert("RGB")
    
    # Inicializa máscara vazia
    mask_rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Abre a máscara, se existir
    mask_file = os.path.join(MASK_PATH, file_name.replace(".tiff","_mask.tiff"))
    if os.path.exists(mask_file):
        mask_slide = openslide.OpenSlide(mask_file)
        mask_img = mask_slide.get_thumbnail(size).convert("L")
        mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img)
        mask_slide.close()

        colors = {
            1: [0,255,0],     # verde
            2: [0,0,255],     # azul
            3: [255,0,0],     # vermelho
            4: [255,165,0],   # laranja
            5: [128,0,128]    # roxo
        }
        for k, v in colors.items():
            for c in range(3):
                mask_rgb[:,:,c] = np.where(mask_img==k, v[c], mask_rgb[:,:,c])
    
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].imshow(thumbnail)
    axes[0].set_title("Imagem")
    axes[0].axis('off')
    
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Máscara")
    axes[1].axis('off')
    
    plt.suptitle(file_name)
    plt.show()
    
    slide.close()

for file_name in image_files:
    show_image_and_mask(file_name)

