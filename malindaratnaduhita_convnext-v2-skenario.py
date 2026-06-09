import os
import copy
import timm
import random
import time
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.optim.optimizer
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.cuda import amp
from torchvision import transforms as T
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from tqdm import tqdm

print(torch.__version__)


def seed_everything(seed):
    """
    Sets seeds for reproducibility in training.

    Args:
        seed (int): Seed value to ensure determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # Seed for hash-based operations
    np.random.seed(seed)  # Seed for NumPy
    torch.manual_seed(seed)  # Seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Seed for PyTorch (GPU)
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Enable benchmark mode for CuDNN


seed_everything(42)


data = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')


print('Number of samples: ', data.shape[0])
display(data.head())


data['diagnosis'].value_counts()


f, ax = plt.subplots(figsize=(14, 8.7))
ax = sns.countplot(x="diagnosis", data=data, palette="GnBu_d")
sns.despine()
plt.savefig('Distruption class', dpi=300, transparent=True)
plt.show()


# Setting the style for the plot
sns.set_style("white")

# Mapping class labels to their corresponding categories
level_to_category = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# Plotting the first 15 images along with their labels
count = 1
plt.figure(figsize=[20, 20])

for img_name in data['id_code'][:15]:  # Assuming 'train' contains the dataset
    img = cv2.imread(f"../input/aptos2019-blindness-detection/train_images/{img_name}.png")[..., [2, 1, 0]]  # Reading the image
    
    # Getting the label (class) for the image
    label = data[data['id_code'] == img_name]['diagnosis'].values[0]  # Assuming 'diagnosis' is the label column
    
    # Setting up the subplot with image and label
    plt.subplot(5, 5, count)
    plt.imshow(img)
    plt.title(f"Image {count}: {level_to_category[label]}")  # Display the class label
    count += 1
    
# Display the plot
plt.savefig('/kaggle/working/imagebeforepreprecssing.png')
plt.show()


# Function to crop the image based on grayscale threshold
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark so that we crop out everything
            return img  # Return original image
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


# Set input and output directories
input_dir = '/kaggle/input/aptos2019-blindness-detection/train_images/'
output_dir = '/kaggle/working/processed_images/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the CSV containing image names and labels
csv_path = '/kaggle/input/aptos2019-blindness-detection/train.csv'
df = pd.read_csv(csv_path)


def process_image(row, sigmaX=10):
    sample_image_id = row['id_code']
    sample_image_file = sample_image_id + '.png'
    sample_image_path = os.path.join(input_dir, sample_image_file)
    
    if os.path.exists(sample_image_path):
        # Ben Graham's preprocessing
        image = cv2.imread(sample_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (384, 384))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        
        # Save the processed image to the output directory
        output_path = os.path.join(output_dir, sample_image_file)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# Using ThreadPoolExecutor to process images in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(process_image, [row for _, row in df.iterrows()]), total=df.shape[0], desc="Processing images", unit="image"))

print("Processing complete for all images.")


# Setting the style for the plot
sns.set_style("white")

# Mapping class labels to their corresponding categories`
level_to_category = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# Plotting the first 15 images along with their labels
count = 1
plt.figure(figsize=[20, 20])

for img_name in data['id_code'][:15]:  # Assuming 'train' contains the dataset
    img = cv2.imread(f"/kaggle/working/processed_images/{img_name}.png")[..., [2, 1, 0]]  # Reading the image
    
    # Getting the label (class) for the image
    label = data[data['id_code'] == img_name]['diagnosis'].values[0]  # Assuming 'diagnosis' is the label column
    
    # Setting up the subplot with image and label
    plt.subplot(5, 5, count)
    plt.imshow(img)
    plt.title(f"Image {count}: {level_to_category[label]}")  # Display the class label
    count += 1

# Display the plot
plt.show()


# Setelah print("Processing complete for all images.")

# ---------------------------------------------------
# Tampilkan & simpan Before & After untuk 5 sampel
# ---------------------------------------------------

# Buat direktori untuk menyimpan sampel jika belum ada
sample_dir = '/kaggle/working/samples_again/'
os.makedirs(sample_dir, exist_ok=True)

# Ambil 5 sample pertama
num_samples = 15
sample_ids = data['id_code'][:num_samples]

plt.figure(figsize=(12, num_samples * 4))

for idx, img_name in enumerate(sample_ids):
    # ---- Before preprocessing ----
    orig = cv2.imread(os.path.join(input_dir, img_name + '.png'))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    # Crop abu-abu lalu resize ke 384×384 agar match After
    orig_cropped = crop_image_from_gray(orig)
    orig_resized = cv2.resize(orig_cropped, (384, 384))
    
    # ---- After preprocessing (Ben Graham) ----
    proc = cv2.imread(os.path.join(output_dir, img_name + '.png'))[..., [2,1,0]]  # BGR->RGB
    # (Assume proc sudah 384×384 dari proses sebelumnya)

    # Subplot: dua kolom (Before | After)
    ax_before = plt.subplot(num_samples, 2, idx * 2 + 1)
    ax_before.imshow(orig_resized)
    ax_before.set_title(f"{img_name} — Before", fontsize=10)
    ax_before.axis('off')

    ax_after = plt.subplot(num_samples, 2, idx * 2 + 2)
    ax_after.imshow(proc)
    ax_after.set_title(f"{img_name} — After", fontsize=10)
    ax_after.axis('off')

    # Simpan masing-masing sebagai PNG
    # Konversi kembali ke BGR untuk imwrite
    orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)
    proc_bgr = cv2.cvtColor(proc, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(sample_dir, f"{img_name}_before.png"), orig_bgr)
    cv2.imwrite(os.path.join(sample_dir, f"{img_name}_after.png"),  proc_bgr)

plt.tight_layout()
plt.show()

print(f"Sampel saved in {sample_dir}")



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ——————————————
# Definisi Ben Graham preprocessing
# ——————————————
def ben_graham_preprocessing(img, radius=300, sigmaX=10, crop_ratio=0.9):
    """
    1) Resize so that min-distance-from-center = radius
    2) Subtract local average (map to 50% gray)
    3) Center-crop crop_ratio (e.g. 0.9) to remove boundary effects
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r0 = min(cx, cy)
    scale = radius / r0
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # subtract local mean → high-pass + 50% gray
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX)
    img = cv2.addWeighted(img, 4, blur, -4, 128)

    # center-crop
    h2, w2 = img.shape[:2]
    crop_size = int(min(h2, w2) * crop_ratio)
    x0 = (w2 - crop_size) // 2
    y0 = (h2 - crop_size) // 2
    img = img[y0:y0+crop_size, x0:x0+crop_size]

    return img

# ——————————————
# Setelah print("Processing complete for all images.")
# ——————————————

# ---------------------------------------------------
# Tampilkan & simpan Before & After untuk 5 sampel
# ---------------------------------------------------

# Buat direktori untuk menyimpan sampel jika belum ada
sample_dir = '/kaggle/working/samples_again/'
os.makedirs(sample_dir, exist_ok=True)

# Ambil 5 sample pertama
num_samples = 15
sample_ids = data['id_code'][:num_samples]

plt.figure(figsize=(12, num_samples * 4))

for idx, img_name in enumerate(sample_ids):
    # ---- Before preprocessing ----
    orig = cv2.imread(os.path.join(input_dir, img_name + '.png'))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    #orig_cropped = crop_image_from_gray(orig)            # biar background gelap ter-crop
    #orig_resized = cv2.resize(orig_cropped, (384, 384))  # ukuran match After

    # ---- After preprocessing (Ben Graham) ----
    # langsung terapkan Graham ke gambar RGB asli
    proc = ben_graham_preprocessing(orig, radius=300, sigmaX=10, crop_ratio=0.9)
    proc = cv2.resize(proc, (384, 384))                  # resize output akhir

    # ---- Plot ----
    ax_before = plt.subplot(num_samples, 2, idx * 2 + 1)
    ax_before.imshow(orig_resized)
    ax_before.set_title(f"{img_name} — Before", fontsize=10)
    ax_before.axis('off')

    ax_after = plt.subplot(num_samples, 2, idx * 2 + 2)
    ax_after.imshow(proc)
    ax_after.set_title(f"{img_name} — After", fontsize=10)
    ax_after.axis('off')

    # ---- Save PNGs ----
    orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)
    proc_bgr = cv2.cvtColor(proc,       cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(sample_dir, f"{img_name}_before.png"), orig_bgr)
    cv2.imwrite(os.path.join(sample_dir, f"{img_name}_after.png"),  proc_bgr)

plt.tight_layout()
plt.show()

print(f"Sampel saved in {sample_dir}")


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Fungsi crop background abu-abu
# Function to crop the image based on grayscale threshold
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark so that we crop out everything
            return img  # Return original image
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

# 2) Fungsi resize by radius
def resize_by_radius(img, target_radius=300):
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    r0 = min(cx, cy)
    scale = target_radius / r0
    return cv2.resize(img, (384, 384))

# 3) Fungsi subtract local mean (high-pass + map mean→128)
def highpass_graymapping(img, sigmaX=10):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX)
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    
# 4) Fungsi center-crop
def center_crop(img, crop_ratio=0.9):
    h, w = img.shape[:2]
    c = int(min(h, w) * crop_ratio)
    x0 = (w - c)//2
    y0 = (h - c)//2
    return img[y0:y0+c, x0:x0+c]

# ————————————————————————————————————————————
# Demo satu gambar
# ————————————————————————————————————————————
input_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
fname     = "0083ee8054ee.png"

# baca & konversi ke RGB
img_bgr = cv2.imread(os.path.join(input_dir, fname))
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Langkah 1: Crop abu-abu
step1 = crop_image_from_gray(img)

# Langkah 2: Resize by radius
step2 = resize_by_radius(step1, target_radius=384)

# Langkah 3: Subtract local mean
step3 = highpass_graymapping(step2, sigmaX=10)





# Langkah 2: Resize by radius
step4 = resize_by_radius(step1, target_radius=384)


# Plot hasil setiap langkah
titles = ["Original", 
          "1) Cropping Gray Background", 
          "2) Resize to 384px",
          "3) Subtractive Normalization"
         ]
images = [img
          , step1, step2, step3
         ]

plt.figure(figsize=(15, 4))
for i, (im, t) in enumerate(zip(images, titles), 1):
    ax = plt.subplot(1, 5, i)
    ax.imshow(im.astype(np.uint8))
    ax.set_title(t, fontsize=10, fontweight='bold')
    ax.axis("off")

plt.tight_layout()
plt.savefig("preprocessing_steps.png", dpi=500)
plt.show()


import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def scaleRadius(img, scale):
    """
    1) Hitung intensitas baris tengah → proyeksi gray (sum channel)
    2) Tentukan r = (jumlah pixel > mean(gray)/10) / 2
    3) s = scale / r
    4) resize img dengan faktor s
    """
    # ambil baris tengah, sum jika color
    mid = img.shape[0] // 2
    if img.ndim == 3:
        proj = img[mid, :, :].sum(axis=1)
    else:
        proj = img[mid, :]
    thresh = proj.mean() / 10.0
    mask  = proj > thresh
    r = mask.sum() / 2.0
    if r == 0:
        return img
    s = scale / r
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

def ben_graham_preprocess(img, scale=300):
    """
    1) scaleRadius
    2) subtract local mean (high-pass + map to 50% gray)
    3) mask outer 10% → isi 50% gray
    """
    # 1) Resize by radius
    a = scaleRadius(img, scale)

    # 2) Subtract lokal mean
    sigma = scale / 30.0
    blur  = cv2.GaussianBlur(a, (0, 0), sigma)
    a     = cv2.addWeighted(a, 4, blur, -4, 128)

    # 3) Remove outer 10% via circular mask
    h, w = a.shape[:2]
    radius = int(min(h, w) * 0.9 / 2)
    mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), radius, 1, -1)

    # terapkan mask, di luar lingkaran diisi 128
    if a.ndim == 3:
        for c in range(3):
            a[:,:,c] = a[:,:,c] * mask + 128 * (1-mask)
    else:
        a = a * mask + 128 * (1-mask)

    return a

# ————————————————————————————————————————————
# Demo satu gambar & plot tiap langkah
# ————————————————————————————————————————————
input_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
fname     = "000c1434d8d7.png"

# Load BGR → RGB
img_bgr = cv2.imread(os.path.join(input_dir, fname))
img     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Langkah‑langkah Graham
step1 = scaleRadius(img, scale=300)
step2 = highpass = cv2.GaussianBlur(step1, (0,0), 300/30.0)
step2 = cv2.addWeighted(step1, 4, step2, -4, 128)
step3 = ben_graham_preprocess(img, scale=300)  # sudah termasuk semua langkah

# Plot
titles = ["Original",
          "1) scaleRadius",
          "2) high-pass + graymap",
          "3) remove outer 10%"]
images = [img, step1, step2, step3]

plt.figure(figsize=(12, 4))
for i, (im, t) in enumerate(zip(images, titles), 1):
    ax = plt.subplot(1, 4, i)
    ax.imshow(im.astype(np.uint8))
    ax.set_title(t, fontsize=10, fontweight='bold')
    ax.axis("off")

plt.tight_layout()
plt.show()





