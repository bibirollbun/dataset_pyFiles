import os
import pydicom
import random
from tqdm import tqdm  # pastikan tqdm sudah terinstall

# Path ke folder train
train_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/'

# Ambil semua folder pasien
all_patients = [p for p in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, p))]

# Acak dan ambil 10% folder pasien
sample_size = max(1, int(0.3 * len(all_patients)))
sampled_patients = random.sample(all_patients, sample_size)

# List untuk menyimpan data DICOM FLAIR
dicom_data_flair = []

# Iterasi tiap pasien dengan progress bar
for patient_id in tqdm(sampled_patients, desc="Memuat data DICOM FLAIR"):
    flair_path = os.path.join(train_path, patient_id, 'FLAIR')
    if os.path.isdir(flair_path):
        for dcm_file in os.listdir(flair_path):
            if dcm_file.endswith('.dcm'):
                dcm_path = os.path.join(flair_path, dcm_file)
                try:
                    dicom = pydicom.dcmread(dcm_path)
                    dicom_data_flair.append(dicom)
                except Exception as e:
                    print(f"Gagal membaca {dcm_path}: {e}")

print(f"\nTotal pasien diambil: {len(sampled_patients)}")
print(f"Total file DICOM FLAIR dimuat: {len(dicom_data_flair)}")



dicom_data_flair[0]


import pandas as pd

# Simpan hasil ke dalam DataFrame
data = []

for dicom in dicom_data_flair:
    try:
        image_array = dicom.pixel_array  # Ambil data citra (2D array)
        data.append({
            "file_path": dicom.filename,
            "image": image_array
        })
    except Exception as e:
        print(f"Gagal mengambil pixel_array dari {dicom.filename}: {e}")

df_dicom_flair = pd.DataFrame(data)

print(f"\nTotal file disimpan dalam DataFrame: {len(df_dicom_flair)}")
df_dicom_flair.head()



import matplotlib.pyplot as plt

# Menampilkan 1 gambar pertama
plt.imshow(df_dicom_flair.iloc[1000]["image"], cmap='gray')
plt.title(df_dicom_flair.iloc[1000]["file_path"])
plt.axis('off')
plt.show()


import cv2
import numpy as np

def resize_images(df, image_column='image', size=(256, 256)):
    resized_images = []
    for img in df[image_column]:
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    df['resized'] = resized_images
    return df

df_dicom_flair = resize_images(df_dicom_flair, image_column='image', size=(256, 256))


import matplotlib.pyplot as plt

def show_images_side_by_side(df, col1, col2, index=0):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(df.iloc[index][col1], cmap='gray')
    axs[0].set_title(col1)
    axs[0].axis('off')

    axs[1].imshow(df.iloc[index][col2], cmap='gray')
    axs[1].set_title(col2)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

show_images_side_by_side(df_dicom_flair, 'image', 'resized', index=1000)


import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Melakukan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    untuk memperjelas citra medis grayscale.
    
    Parameter:
    - clip_limit: batas kontras untuk mencegah noise
    - tile_grid_size: ukuran patch untuk pengolahan lokal
    """
    if image.dtype != np.uint8:
        # Normalize ke rentang 0-255 dan konversi ke uint8
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced

# Terapkan CLAHE untuk setiap gambar pada kolom 'resized'
df_dicom_flair['enhancement'] = df_dicom_flair['resized'].apply(apply_clahe)


show_images_side_by_side(df_dicom_flair, 'resized', 'enhancement', index=1000)


def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """
    Terapkan unsharp masking ke citra grayscale.
    
    Parameters:
    - kernel_size: ukuran kernel Gaussian blur
    - sigma: deviasi standar untuk Gaussian blur
    - amount: seberapa banyak detail ditambahkan kembali (penguatan tepi)
    - threshold: nilai minimum perubahan untuk diterapkan (hindari noise)
    """
    # Pastikan gambar dalam format uint8
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Blur citra
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Hitung mask (detail edges)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, 0)
    sharpened = np.minimum(sharpened, 255)
    sharpened = sharpened.round().astype(np.uint8)

    if threshold > 0:
        # Terapkan thresholding agar hanya bagian signifikan yang diasah
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened

df_dicom_flair['unsharp_masking'] = df_dicom_flair['enhancement'].apply(
    lambda img: apply_unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0)
)


show_images_side_by_side(df_dicom_flair, 'enhancement', 'unsharp_masking', index=1000)


# def simple_brain_segmentation(image):
#     """
#     Segmentasi sederhana gambar CT scan otak dengan thresholding dan operasi morfologi.
    
#     image: gambar grayscale (uint8)
#     return: gambar hasil segmentasi (mask biner)
#     """
#     # Normalisasi dan pastikan uint8
#     if image.dtype != np.uint8:
#         image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     # Blur untuk mengurangi noise
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)

#     # Otsu Thresholding (otomatis)
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Invers jika latar belakang lebih terang
#     if np.mean(image[thresh == 255]) > np.mean(image[thresh == 0]):
#         thresh = cv2.bitwise_not(thresh)

#     # Operasi morfologi untuk mengisi lubang dan menghilangkan noise kecil
#     kernel = np.ones((3, 3), np.uint8)
#     cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#     cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

#     return cleaned

# df_dicom_flair['segmentation'] = df_dicom_flair['unsharp_masking'].apply(simple_brain_segmentation)

# def crop_and_resize_brain(image, mask, output_size=(64, 64)):
#     """
#     Crop bagian otak dari image berdasarkan mask, lalu resize ke ukuran output_size.

#     Parameters:
#     - image: numpy array gambar grayscale (2D)
#     - mask: binary mask (2D), hasil segmentasi
#     - output_size: tuple ukuran (width, height)

#     Returns:
#     - image hasil crop & resize
#     """
#     # Cari koordinat non-zero (bagian otak)
#     coords = cv2.findNonZero(mask)
#     if coords is None:
#         # Jika tidak ada area otak terdeteksi, kembalikan hasil resize langsung
#         resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
#         return resized
    
#     # Hitung bounding rectangle dari area otak
#     x, y, w, h = cv2.boundingRect(coords)
    
#     # Crop area penting
#     cropped = image[y:y+h, x:x+w]

#     # Resize ke ukuran tetap
#     resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)

#     return resized

# df_dicom_flair['cropped'] = df_dicom_flair.apply(
#     lambda row: crop_and_resize_brain(row['unsharp_masking'], row['segmentation']), axis=1
# )


# show_images_side_by_side(df_dicom_flair, 'unsharp_masking', 'cropped', index=1000)


import cv2
import numpy as np

def apply_interpolation(image, target_size, method=cv2.INTER_LANCZOS4):
    """
    Interpolasi gambar ke ukuran target menggunakan metode tertentu.
    
    Parameters:
    - image: array gambar input (grayscale)
    - target_size: tuple (width, height)
    - method: metode interpolasi dari OpenCV
    """
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    resized = cv2.resize(image, target_size, interpolation=method)
    return resized

df_dicom_flair['interpolation'] = df_dicom_flair['unsharp_masking'].apply(
    lambda img: apply_interpolation(img, target_size=(256, 256), method=cv2.INTER_LANCZOS4)
)



show_images_side_by_side(df_dicom_flair, 'unsharp_masking', 'interpolation', index=1000)


import numpy as np

def normalize_image(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image)  # menghindari pembagian nol
    return (image - min_val) / (max_val - min_val)

df_dicom_flair['normalized'] = df_dicom_flair['interpolation'].apply(normalize_image)


show_images_side_by_side(df_dicom_flair, 'interpolation', 'normalized', index=1000)


df_dicom_flair

