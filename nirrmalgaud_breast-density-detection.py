import pandas as pd
import numpy as np
import os


base_path = '/kaggle/input/breast-density-prediction/train/train'

images_dir = os.path.join(base_path, 'images')
dense_masks_dir = os.path.join(base_path, 'dense_masks')
breast_masks_dir = os.path.join(base_path, 'breast_masks')

data = []

for img_file in os.listdir(images_dir):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(images_dir, img_file)
        dense_mask_path = os.path.join(dense_masks_dir, img_file)
        breast_mask_path = os.path.join(breast_masks_dir, img_file)
        
        if os.path.exists(dense_mask_path) and os.path.exists(breast_mask_path):
            data.append({
                'image_path': image_path,
                'dense_mask_path': dense_mask_path,
                'breast_mask_path': breast_mask_path
            })

df = pd.DataFrame(data)


df


meta_df = pd.read_csv('/kaggle/input/breast-density-prediction/train.csv')

df['Filename'] = df['image_path'].apply(lambda x: os.path.basename(x))

df = df.merge(meta_df, on='Filename', how='left')

df.drop(columns=['Filename'], inplace=True)


df


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

plt.figure(figsize=(20, 12))

for i in range(5):
    
    img = np.array(Image.open(df['image_path'].iloc[i]))
    dense_mask = np.array(Image.open(df['dense_mask_path'].iloc[i]))
    breast_mask = np.array(Image.open(df['breast_mask_path'].iloc[i]))
    
    plt.subplot(3, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Image {i+1}')
    
    plt.subplot(3, 5, i + 6)
    plt.imshow(dense_mask, cmap='gray')
    plt.axis('off')
    plt.title('Dense Mask')
    
    plt.subplot(3, 5, i + 11)
    plt.imshow(breast_mask, cmap='gray')
    plt.axis('off')
    plt.title('Breast Mask')

plt.tight_layout()
plt.show()


import cv2
import matplotlib.pyplot as plt

sample = df.iloc[0]
image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
breast_mask = cv2.imread(sample['breast_mask_path'], cv2.IMREAD_GRAYSCALE)
dense_mask = cv2.imread(sample['dense_mask_path'], cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Mammogram")
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Breast Mask")
plt.imshow(breast_mask, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Dense Mask")
plt.imshow(dense_mask, cmap='gray')
plt.show()


df.columns


import tensorflow as tf
# Configure memory growth for all GPUs
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for all GPUs")
    except RuntimeError as e:
        print(f"Warning: Could not set memory growth: {e}")

from tensorflow.keras import layers, models, backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import os
import matplotlib.pyplot as plt

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def load_data(df, img_size=(128, 128)):
    images = []
    masks = []
    density = []
    for idx, row in df.iterrows():
        try:
            img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            mask = cv2.imread(row['dense_mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, img_size)
            mask = (mask > 127).astype(np.float32)  # Binarize mask
            images.append(img)
            masks.append(mask)
            density.append(float(row['Density']))
        except Exception as e:
            print(f"Error loading image/mask at index {idx}: {e}")
            continue
    if not images:
        raise ValueError("No valid images/masks loaded")
    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    masks = np.array(masks).reshape(-1, img_size[0], img_size[1], 1)
    density = np.array(density)
    return images, masks, density

def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)
    seg_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='seg_output')(c7)
    flat = layers.Flatten()(c7)
    dense1 = layers.Dense(64, activation='relu')(flat)
    dense2 = layers.Dense(32, activation='relu')(dense1)
    dense_output = layers.Dense(1, name='dense_output')(dense2)
    model = models.Model(inputs=inputs, outputs=[seg_output, dense_output])
    return model

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Total Loss')
    plt.plot(history.history['val_loss'], label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history.history['seg_output_dice_coefficient'], label='Dice Score')
    plt.plot(history.history['val_seg_output_dice_coefficient'], label='Val Dice Score')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history.history['seg_output_iou'], label='IoU')
    plt.plot(history.history['val_seg_output_iou'], label='Val IoU')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_metrics.png')
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['seg_output_accuracy'], label='Seg Accuracy')
    plt.plot(history.history['val_seg_output_accuracy'], label='Val Seg Accuracy')
    plt.title('Segmentation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dense_output_mae'], label='Density MAE')
    plt.plot(history.history['val_dense_output_mae'], label='Val Density MAE')
    plt.title('Density MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/kaggle/working/accuracy_mae.png')
    plt.close()

images, masks, density = load_data(df)
X_train, X_test, y_train_mask, y_test_mask, y_train_dense, y_test_dense = train_test_split(
    images, masks, density, test_size=0.2, random_state=42
)
scaler = StandardScaler()
y_train_dense = scaler.fit_transform(y_train_dense.reshape(-1, 1))
y_test_dense = scaler.transform(y_test_dense.reshape(-1, 1))
model = build_unet()
model.compile(
    optimizer='adam',
    loss={'seg_output': 'binary_crossentropy', 'dense_output': 'mean_squared_error'},
    loss_weights={'seg_output': 1.0, 'dense_output': 0.1},  # Weight segmentation higher
    metrics={
        'seg_output': ['accuracy', dice_coefficient, iou],
        'dense_output': 'mae'
    }
)
history = model.fit(
    X_train,
    {'seg_output': y_train_mask, 'dense_output': y_train_dense},
    validation_data=(X_test, {'seg_output': y_test_mask, 'dense_output': y_test_dense}),
    epochs=50,  # Increased epochs for better segmentation
    batch_size=8
)
model.save('/kaggle/working/unet_density_model.h5')
plot_training_history(history)

