import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report 
import joblib
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')


# Function to crop the image based on grayscale threshold
def top_bottom_hat_filtering(path):
    image = cv2.imread(path)

    cropped_img = crop_image_from_gray(image)
   
   # Elemen struktural (kernel) untuk operasi morfologi
    kernel_size = 15  # Ukuran kernel harus disesuaikan dengan fitur yang ingin diperjelas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Top-Hat transform (menyoroti fitur terang)
    top_hat = cv2.morphologyEx(cropped_img, cv2.MORPH_TOPHAT, kernel)
    
    # Bottom-Hat transform (menyoroti fitur gelap)
    bottom_hat = cv2.morphologyEx(cropped_img, cv2.MORPH_BLACKHAT, kernel)
    
    # Hasil akhir: Menambahkan Top-Hat dan mengurangi Bottom-Hat untuk meningkatkan kontras
    enhanced_image = cv2.add(cropped_img, top_hat)  # Menonjolkan area terang
    # enhanced_image = cv2.subtract(enhanced_image, bottom_hat)  # Menghilangkan bayangan gelap

    
    return image


image = cv2.imread('/kaggle/input/resized-dataset-aptos/Severe/14e3f84445f7.png')

# Elemen struktural (kernel) untuk operasi morfologi
kernel_size = 15  # Ukuran kernel harus disesuaikan dengan fitur yang ingin diperjelas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# Top-Hat transform (menyoroti fitur terang)
top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# Bottom-Hat transform (menyoroti fitur gelap)
bottom_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

# Hasil akhir: Menambahkan Top-Hat dan mengurangi Bottom-Hat untuk meningkatkan kontras
enhanced_image = cv2.add(image, top_hat)  # Menonjolkan area terang
enhanced_image2 = cv2.subtract(image, bottom_hat)  # Menghilangkan bayangan gelap
enhanced_image3 = cv2.subtract(enhanced_image, bottom_hat)  # Menghilangkan bayangan gelap

# Menampilkan hasil
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(top_hat, cmap='gray'), plt.title('Top-Hat Transform'), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(bottom_hat, cmap='gray'), plt.title('Bottom-Hat Transform'), plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(1,4,1), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(1,4,2), plt.imshow(enhanced_image, cmap='gray'), plt.title('Top-Hat '), plt.axis('off')
plt.subplot(1,4,3), plt.imshow(enhanced_image2, cmap='gray'), plt.title('Bottom-Hat '), plt.axis('off')
plt.subplot(1,4,4), plt.imshow(enhanced_image3, cmap='gray'), plt.title('Top-Bottom '), plt.axis('off')
plt.show()




# import os
# import cv2
# import pandas as pd
# import shutil

# # Dictionary mapping dari int ke label string
# label_dict = {
#     0: "No DR",
#     1: "Mild",
#     2: "Moderate",
#     3: "Severe",
#     4: "Proliferative DR"
# }

# # Path ke file CSV dan folder gambar input
# csv_path = '/kaggle/input/aptos2019-blindness-detection/train.csv'
# input_folder = '/kaggle/input/aptos2019-blindness-detection/train_images'  # Pastikan folder ini ada di Kaggle working directory

# # Folder output yang akan menyimpan gambar yang sudah diproses
# output_folder = '/kaggle/working/resized_aptos_512'

# # Buat folder output jika belum ada
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Baca file CSV
# df = pd.read_csv(csv_path)

# # Loop setiap baris dalam CSV
# for index, row in df.iterrows():
#     filename = row['id_code']
    
#     # Tambahkan ekstensi .png jika belum ada
#     if not os.path.splitext(filename)[1]:
#         filename += ".png"
    
#     # Ubah label dari int ke string menggunakan dictionary
#     int_label = int(row['diagnosis'])
#     label_str = label_dict.get(int_label, "Unknown")
    
#     # Buat folder untuk label jika belum ada
#     label_folder = os.path.join(output_folder, label_str)
#     if not os.path.exists(label_folder):
#         os.makedirs(label_folder)
    
#     # Path lengkap ke file gambar input
#     image_path = os.path.join(input_folder, filename)
    
#     # Baca gambar menggunakan OpenCV
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Warning: Gambar {filename} tidak dapat dibaca.")
#         continue
    
#     # Resize gambar ke 512x512 piksel
#     resized_image = cv2.resize(image, (512, 512))
    
#     # Tentukan path output untuk gambar yang sudah diproses
#     output_path = os.path.join(label_folder, filename)
    
#     # Simpan gambar yang sudah di-resize
#     cv2.imwrite(output_path, resized_image)

# print("Proses pembuatan dataset selesai!")

# # Setelah dataset selesai dibuat, zip folder output
# archive_name = "aptos_512"  # nama file zip tanpa ekstensi
# shutil.make_archive(archive_name, 'zip', output_folder)
# print(f"Folder {output_folder} telah di-zip menjadi {archive_name}.zip")



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



def preprocessing_clahe_rgb(path):
    image = cv2.imread(path)

    # Crop the image based on gray threshold
    image_cropped = crop_image_from_gray(image)
    
    image_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)

    # Split the image into its channels (BGR format)
    blue, green, red = cv2.split(image_rgb)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

    # Apply CLAHE to all three channels
    blue_clahe = clahe.apply(blue)
    green_clahe = clahe.apply(green)
    red_clahe = clahe.apply(red)
    
    # Merge the CLAHE-enhanced channels back together
    result_image = cv2.merge([red, green_clahe, blue])

    return result_image


import cv2
import numpy as np

def preprocessing_clahe(path, clip_limit=4.0, grid_size=(4, 4)):
    image = cv2.imread(path)

    image_cropped = crop_image_from_gray(image)

    # Konversi ke LAB
    lab_image = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2Lab)
    
    # Pisahkan channel L, A, dan B
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Terapkan CLAHE pada L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    l_channel_clahe = clahe.apply(l_channel)

    # Gabungkan kembali L-channel yang telah diproses dengan A dan B yang asli
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    
    # Konversi kembali ke RGB
    image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_Lab2RGB)

    
    return image



def preprocessing_clahe_grayscale(path, clip_limit=4.0, grid_size=(4, 4)):
    # Baca gambar dalam mode grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Buat objek CLAHE dengan clip limit dan ukuran grid yang diberikan
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    # Terapkan CLAHE pada gambar grayscale
    image_clahe = clahe.apply(image)
    
    return image_clahe


import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Misal, data_dir adalah folder dataset asli
data_dir = "/kaggle/input/resized-dataset-aptos"  
output_dir = "/kaggle/working/preprocessed"

# Jika folder output sudah ada, hapus terlebih dahulu
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

# Fungsi untuk menyimpan gambar hasil preprocessing ke folder output berdasarkan kelas baru
def save_image(image, new_label, filename):
    # Pastikan folder untuk new_label sudah ada di output_dir
    label_dir = os.path.join(output_dir, new_label)
    os.makedirs(label_dir, exist_ok=True)  # Buat folder jika belum ada
    image_path = os.path.join(label_dir, filename)
    cv2.imwrite(image_path, image)

# Looping melalui setiap folder di dataset
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    
    # Tentukan kelas baru: jika label == "No_DR", maka new_label = "0"; selain itu new_label = "1"
    new_label = "0" if label == "No_DR" else "1"
    
    if os.path.isdir(label_dir):
        for filename in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
            # Tambahkan ekstensi .png jika belum ada
            if not os.path.splitext(filename)[1]:
                filename += ".png"
                
            image_path = os.path.join(label_dir, filename)
            processed_image = top_bottom_hat_filtering(image_path)
            if processed_image is not None:
                save_image(processed_image, new_label, filename)

# Mengompres folder hasil preprocessing menjadi file ZIP
shutil.make_archive("/kaggle/working/preprocessed_resized", "zip", output_dir)

print("Preprocessing selesai dan file zip sudah dibuat!")



# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# import shutil

# # data_dir = "/kaggle/input/resized-dataset-aptos"  
# output_dir = "/kaggle/working/preprocessed"

# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)

# os.makedirs(output_dir, exist_ok=True)

# # Fungsi untuk menyimpan gambar hasil preprocessing
# def save_image(image, label, filename):
#     # Pastikan folder untuk label sudah ada di output_dir
#     label_dir = os.path.join(output_dir, label)
#     os.makedirs(label_dir, exist_ok=True)  # Buat folder label jika belum ada
#     image_path = os.path.join(label_dir, filename)
#     cv2.imwrite(image_path, image)

# # Looping melalui setiap folder label dalam dataset
# for label in os.listdir(data_dir):
#     label_dir = os.path.join(data_dir, label)
    
#     if os.path.isdir(label_dir):  # Hanya proses folder, bukan file
#         for filename in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
#             image_path = os.path.join(label_dir, filename)

#             # Preprocess gambar
#             processed_image = preprocessing_clahe(image_path)

#             if processed_image is not None:
#                 save_image(processed_image, label, filename)

# # Mengompres folder hasil preprocessing menjadi file ZIP
# shutil.make_archive("/kaggle/working/preprocessed_resized", "zip", output_dir)

# print("Preprocessing selesai dan file zip sudah dibuat!")



import tensorflow as tf

# Membaca dataset dan membagi menjadi train dan validation
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    output_dir,
    validation_split=0.2,  # 80% untuk train, 20% untuk validasi
    subset="training",
    seed=12,
    image_size=(224,224),
    batch_size=32
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    output_dir,
    validation_split=0.2,  # Sesuaikan dengan split yang sama
    subset="validation",
    seed=12,
    image_size=(224, 224),
    batch_size=32
)


class_names = train_dataset.class_names

# Menghitung jumlah gambar per kelas
class_counts = {class_name: 0 for class_name in class_names}
for images, labels in train_dataset:
    for label in labels.numpy():
        class_counts[class_names[label]] += 1

print("Jumlah gambar per kelas:", class_counts)
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Membuat diagram batang (bar chart)
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.title('Jumlah Gambar per Kelas di train_dataset')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Gambar')
plt.xticks(rotation=45)
plt.show()


# Data augmentation layer
data_augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Horizontal & Vertical Flip
    tf.keras.layers.RandomRotation(0.2, fill_mode="constant"),  # Rotasi dengan fill_mode constant
    tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),  # Brightness augmentation
    tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, lower=0.8, upper=1.2))  # Contrast augmentation
])

# Terapkan augmentation pada dataset training
augmented_train = train_dataset.map(lambda x, y: (data_augmentation_layer(x), y))



# import tensorflow as tf
# import cv2
# import numpy as np

# # Fungsi kustom untuk mengaplikasikan blur atau sharpen secara acak pada batch gambar
# def apply_random_blur_sharpen(images):
#     # images: numpy array dengan shape (batch, height, width, channels)
#     out_images = []
#     # Pastikan nilai gambar berada pada rentang 0-255 (dtype uint8) untuk operasi cv2
#     images = images.astype(np.uint8)
#     for img in images:
#         # Pilih secara acak apakah akan blur, sharpen, atau tidak mengubah
#         choice = np.random.choice(['blur', 'sharpen', 'none'])
#         if choice == 'blur':
#             # Gunakan Gaussian Blur dengan kernel 5x5
#             out_img = cv2.GaussianBlur(img, (5,5), 0)
#         elif choice == 'sharpen':
#             # Kernel sharpening standar
#             kernel = np.array([[0, -1, 0],
#                                [-1, 5, -1],
#                                [0, -1, 0]])
#             out_img = cv2.filter2D(img, -1, kernel)
#         else:
#             out_img = img
#         out_images.append(out_img)
#     return np.array(out_images)

# # Data augmentation layer yang mengintegrasikan parameter:
# # contrast_range=0.2, brightness_range=20., hue_range=10.,
# # saturation_range=20., blur_and_sharpen=True, rotate_range=180.,
# # scale_range=0.2, shear_range=0.2, shift_range=0.2, do_mirror=True.
# data_augmentation_layer = tf.keras.Sequential([
#     # Mirror: do_mirror=True
#     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#     # Rotation: rotate_range=180 deg -> factor 0.5 (180/360)
#     tf.keras.layers.RandomRotation(0.5, fill_mode="constant"),
#     # Scale: scale_range=0.2 -> RandomZoom, factor negatif berarti zoom out
#     tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="constant"),
#     # Shift: shift_range=0.2 -> RandomTranslation
#     tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="constant"),
#     # Contrast: contrast_range=0.2
#     tf.keras.layers.RandomContrast(0.2),
#     # Brightness: brightness_range=20. (asumsi gambar dalam [0,1], 20/255 ~ 0.078)
#     tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=20./255.0)),
#     # Hue: hue_range=10 deg -> 10/360 ~ 0.0278
#     tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, max_delta=10./360.0)),
#     # Saturation: saturation_range=20% -> lower=0.8, upper=1.2
#     tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, lower=0.8, upper=1.2)),
#     # Shear: shear_range=0.2, menggunakan tfa.image.transform
#     tf.keras.layers.Lambda(lambda x: tfa.image.transform(
#         x,
#         transforms=tf.random.uniform((tf.shape(x)[0], 8), minval=-0.2, maxval=0.2),
#         interpolation='BILINEAR',
#         fill_mode='CONSTANT'
#     )),
#     # Blur and sharpen: menggunakan fungsi kustom dengan tf.numpy_function
#     tf.keras.layers.Lambda(lambda x: tf.numpy_function(
#         func=apply_random_blur_sharpen,
#         inp=[x],
#         Tout=x.dtype
#     ))
# ])

# # Terapkan augmentation pada dataset training
# augmented_train = train_dataset.map(lambda x, y: (data_augmentation_layer(x), y))

# # Contoh: Jika ingin melihat output augmentasi
# # Untuk satu batch dari augmented_train, misalnya:
# for batch_images, batch_labels in augmented_train.take(1):
#     # Tampilkan gambar pertama dalam batch
#     import matplotlib.pyplot as plt
#     plt.imshow(tf.cast(batch_images[0], tf.uint8).numpy())
#     plt.title("Augmented Image Example")
#     plt.axis("off")
#     plt.show()



import matplotlib.pyplot as plt

# Ambil satu batch gambar dan label menggunakan iterator
train_iterator = iter(train_dataset)
images, labels = next(train_iterator)

# Pastikan gambar ada dalam range [0,1], lalu ubah ke [0,255] untuk ditampilkan
fig, axes = plt.subplots(2, 4, figsize=(8, 8))  # Menyesuaikan figsize lebih kecil agar gambar lebih rapat
for i in range(8):
    row = i // 4  # Menentukan baris (0 hingga 3)
    col = i % 4   # Menentukan kolom (0 hingga 3)
    
    axes[row, col].imshow(images[i] / 255)  # Skala ulang agar bisa ditampilkan
    axes[row, col].axis("off")

# Menyesuaikan layout agar gambar tidak tumpang tindih
plt.tight_layout(pad=0.5)
plt.show()



for images, labels in augmented_train.take(2):  # Taking one batch
    plt.figure(figsize=(10, 10))

    # Loop over the first 16 images in the batch
    for i in range(8):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i] / 255 )
        plt.axis("off")
    
    plt.show()


# from sklearn.utils.class_weight import compute_class_weight
# # Ambil label dari train_dataset
# y_train = np.concatenate([y for x, y in train_dataset], axis=0)  # Gabungkan semua label

# # Jika label one-hot encoded, konversi ke integer
# if y_train.ndim > 1:
#     y_train = np.argmax(y_train, axis=1)

# # Hitung class weight
# class_weights = compute_class_weight(
#     class_weight='balanced',  # Menghitung bobot secara otomatis
#     classes=np.unique(y_train),  # Kelas unik dalam dataset
#     y=y_train  # Label training
# )

# class_weights_dict = dict(enumerate(class_weights))
# print("Class Weights:", class_weights_dict)


# val_batches = tf.data.experimental.cardinality(val_normalization_dataset)
val_batches = tf.data.experimental.cardinality(val_dataset)
test_batches = val_batches // 2 
val_batches = val_batches - test_batches  

test_dataset = val_dataset.take(test_batches)
val_dataset = val_dataset.skip(test_batches)
# test_normalization_dataset = val_normalization_dataset.take(test_batches)
# val_normalization_dataset = val_normalization_dataset.skip(test_batches)



# Prefetch untuk performa lebih baik
AUTOTUNE = tf.data.AUTOTUNE
train_gen = augmented_train.prefetch(buffer_size=AUTOTUNE)
val_gen = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_gen = test_dataset.prefetch(buffer_size=AUTOTUNE)


import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adamax, Adam

rescale_layer = tf.keras.layers.Rescaling(1./255)

# Define the model using Functional API
inputs = Input(shape=(224, 224, 3))

# x = preprocess_input(inputs)
# x = rescale_layer(inputs)

base_model = tf.keras.applications.ConvNeXtTiny(include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = True
    
x = base_model(inputs)

x = Dropout(0.5)(x)

x = GlobalAveragePooling2D()(x)

# # # Fully connected layer with regularization
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)

x = Dropout(0.5)(x)

# Output layer (4 classes)
outputs = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

# Model summary
model.summary()


# import tensorflow as tf
# from tensorflow.keras import layers, Model

# def ConvNeXtBlock(x, filters, drop_path_rate=0.0):
#     """
#     ConvNeXt Block: Combines depthwise convolution, layer normalization, and GELU activation.
#     """
#     # Save the input for residual connection
#     residual = x

#     # Depthwise Convolution
#     x = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)

#     # Pointwise Convolution (1x1 Conv to expand/reduce channels)
#     x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)
#     x = layers.Activation("gelu")(x)

#     # Pointwise Convolution (1x1 Conv to restore channels)
#     x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)

#     # Drop Path (Stochastic Depth)
#     if drop_path_rate > 0.0:
#         x = layers.Dropout(drop_path_rate)(x)

#     # Add residual connection
#     x = layers.Add()([residual, x])

#     return x

# def ConvNeXtStem(x, filters):
#     """
#     Stem block for ConvNeXt: Initial downsampling and feature extraction.
#     """
#     x = layers.Conv2D(filters=filters, kernel_size=4, strides=4, padding="same")(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     return x

# def ConvNeXtStage(x, filters, num_blocks, drop_path_rate=0.0):
#     """
#     ConvNeXt Stage: A sequence of ConvNeXt blocks.
#     """
#     for _ in range(num_blocks):
#         x = ConvNeXtBlock(x, filters, drop_path_rate)
#     return x

# def ConvNeXt(input_shape=(224, 224, 3), depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0):
#     """
#     ConvNeXt Model: Full architecture with multiple stages.
#     """
#     inputs = layers.Input(shape=input_shape)

#     # Stem
#     x = ConvNeXtStem(inputs, dims[0])

#     # Stages
#     for i, (depth, dim) in enumerate(zip(depths, dims)):
#         x = ConvNeXtStage(x, dim, depth, drop_path_rate)
#         if i < len(depths) - 1:  # Downsample between stages
#             x = layers.Conv2D(filters=dims[i + 1], kernel_size=2, strides=2, padding="same")(x)
#             x = layers.LayerNormalization(epsilon=1e-6)(x)

#     # Global Average Pooling and Classifier
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     outputs = layers.Dense(1, activation="sigmoid")(x)

#     # Create model
#     model = Model(inputs, outputs, name="ConvNeXt")
#     return model

# # Create the ConvNeXt model
# model = ConvNeXt(input_shape=(224, 224, 3))
# model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

# model.summary()


# import tensorflow as tf
# from tensorflow.keras import layers, models

# def build_cnn(input_shape=(224, 224, 3)):
#     """
#     Membangun model CNN sederhana untuk klasifikasi gambar.
    
#     Args:
#         input_shape (tuple): Bentuk input gambar (height, width, channels).
#         num_classes (int): Jumlah kelas output.
    
#     Returns:
#         model: Model CNN.
#     """
#     # Input layer
#     inputs = layers.Input(shape=input_shape)

#     # Convolutional Block 1
#     x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
#     x = layers.MaxPooling2D((2, 2))(x)

#     # Convolutional Block 2
#     x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     # Convolutional Block 3
#     x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     # Fully Connected Layers
#     x = layers.Flatten()(x)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)  # Dropout untuk mengurangi overfitting
#     outputs = layers.Dense(1, activation="sigmoid")(x)

#     # Membuat model
#     model = models.Model(inputs, outputs, name="CNN")
#     return model

# # Membangun model
# model = build_cnn(input_shape=(224, 224, 3))
# model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

# model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

# ReduceLROnPlateau: Mengurangi learning rate jika val_loss tidak membaik
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Memantau 'val_loss'
    factor=0.5,          # Mengurangi learning rate sebesar 50%
    patience=3,          # Menunggu 2 epoch sebelum mengurangi learning rate
    min_lr=1e-5,         # Nilai learning rate terendah
    verbose=1            # Menampilkan pesan ketika learning rate diubah
)

history=model.fit(train_gen,epochs=50,
                  validation_data=val_gen,shuffle=True,
                  callbacks=[reduce_lr,early_stopping],
                 )


import matplotlib.pyplot as plt
import numpy as np

# Data dari history training
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Mencari epoch terbaik
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

# Label untuk plot
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# Plotting
plt.figure(figsize=(20, 8))
plt.style.use('fivethirtyeight')

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, max(max(tr_loss), max(val_loss)) * 1.1)  # Set y-axis mulai dari 0
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1)  # Set y-axis mulai dari 0 hingga 1 (karena akurasi antara 0 dan 1)
plt.legend()

# Simpan plot
plt.savefig('/kaggle/working/training_plot.png')

# Tampilkan plot
plt.tight_layout()
plt.show()


test_score = model.evaluate(test_gen)
    
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Ambil nama kelas dari dataset (misal: ["No DR", "DR"])
class_names = train_dataset.class_names  

# Buat daftar untuk menyimpan label asli, prediksi, dan probabilitas prediksi
y_true = []
y_pred = []
y_pred_proba = []  # Menyimpan probabilitas prediksi untuk ROC curve

# Pastikan model hanya mendukung binary classification (misalnya, No DR vs DR)
for images, labels in test_gen:
    y_true.extend(labels.numpy())  # Label asli dalam format integer
    
    # Prediksi menggunakan model
    predictions = model.predict(images)
    
    # Karena pakai sigmoid, output hanya satu neuron -> langsung ambil nilai
    y_pred_proba.extend(predictions.flatten())  # Ubah ke 1D array
    y_pred.extend((predictions > 0.5).astype(int).flatten())  # Konversi ke label biner

# Ubah y_true ke binary jika dataset masih multi-class
y_true = np.array(y_true)
if len(set(y_true)) > 2:
    y_true = (y_true > 0).astype(int)  # Pastikan labelnya hanya 0 dan 1

# Menghitung confusion matrix dengan nilai hitungan asli
cm = confusion_matrix(y_true, y_pred)

# Menghitung akurasi
accuracy = accuracy_score(y_true, y_pred)

# Menampilkan confusion matrix dengan seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues", linewidths=.5)
plt.xlabel('\nPredicted Label', fontsize=13)
plt.ylabel('Actual Label\n', fontsize=13)
plt.title('Confusion Matrix - Binary', fontsize=15)
plt.savefig('/kaggle/working/confusion_matrix_binary.png')
plt.show()

# Menampilkan classification report
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_true, y_pred, target_names=class_names))

# Menghitung ROC curve dan AUC untuk binary classification
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Garis diagonal acak

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Binary Classification')
plt.legend(loc='lower right')
plt.grid(True)

# Simpan gambar
plt.savefig('/kaggle/working/AUC_ROC_Binary.png')
plt.show()
print(y_true[:10], y_pred[:10], y_pred_proba[:10])



model.save('/kaggle/working/DenseNet121+clahe+dataaug+finetune.h5')

