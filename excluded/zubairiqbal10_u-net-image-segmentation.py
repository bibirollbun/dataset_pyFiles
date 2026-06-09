import json
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Load COCO JSON file
json_path = "/kaggle/working/_annotations.coco.json"  # Update this path
images_dir = "/kaggle/working/"  # Update this to the directory of images
masks_dir = "/kaggle/working/masks/"  # Directory where masks will be saved

# Create masks directory if it doesn't exist
os.makedirs(masks_dir, exist_ok=True)

# Load JSON annotations
with open(json_path, "r") as f:
    data = json.load(f)

# Map image IDs to metadata
image_id_to_metadata = {img["id"]: img for img in data["images"]}

# Group annotations by image_id
image_annotations = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    if image_id not in image_annotations:
        image_annotations[image_id] = []
    image_annotations[image_id].append(ann)

# Process each image
for image_id, annotations in image_annotations.items():
    metadata = image_id_to_metadata.get(image_id, {})
    filename = metadata.get("file_name", "")
    width, height = metadata.get("width", 0), metadata.get("height", 0)

    if not filename:
        continue

    # Define mask path
    mask_filename = filename.replace(".jpg", "_mask.png").replace(".jpeg", "_mask.png").replace(".png", "_mask.png")
    mask_path = os.path.join(masks_dir, mask_filename)

    # Create blank mask for the entire image
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Draw all segmentations for this image
    for ann in annotations:
        segmentation = ann["segmentation"]
        for seg in segmentation:
            poly = [(int(seg[i]), int(seg[i+1])) for i in range(0, len(seg), 2)]
            draw.polygon(poly, outline=255, fill=255)

    # Save the mask
    mask.save(mask_path)

print("Mask images have been successfully generated and saved in:", masks_dir)



# Unzip data
import zipfile
with zipfile.ZipFile('/kaggle/input/data-science-bowl-2018/'+ 'stage1_train.zip', 'r') as zip_ref:
    zip_ref.extractall('./train')
    
with zipfile.ZipFile('/kaggle/input/data-science-bowl-2018/' + 'stage1_test.zip', 'r') as zip_ref:
    zip_ref.extractall('./test')


from PIL import Image
import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import tensorflow as tf

img_width = 128
img_height = 128
img_channels = 3
Train_Path = '/kaggle/working/train/'
Test_Path = '/kaggle/working/test/'

train_Ids = next(os.walk(Train_Path))[1]
test_Ids = next(os.walk(Test_Path))[1]

X_train = np.zeros((len(train_Ids), img_width, img_height, img_channels), dtype=np.uint8)
Y_train = np.zeros((len(train_Ids), img_width, img_height, 1), dtype=np.bool_)

print('ðŸ“¦ Resizing training images and masks...')
for n, id_ in tqdm(enumerate(train_Ids), total=len(train_Ids)):   
    path = os.path.join(Train_Path, id_)
    img_path = os.path.join(path, 'images', os.listdir(os.path.join(path, 'images'))[0])
    img = imread(img_path)[:,:,:img_channels]  
    img = resize(img, (img_width, img_height), mode='constant', preserve_range=True)
    X_train[n] = img

    mask = np.zeros((img_height, img_width, 1), dtype=np.bool_)
    for mask_file in os.listdir(os.path.join(path, 'masks')):
        mask_ = imread(os.path.join(path, 'masks', mask_file))
        mask_ = resize(mask_, (img_height, img_width), mode='constant', preserve_range=True)
        mask_ = np.expand_dims(mask_, axis=-1)
        mask_ = np.clip(mask_, 0, 1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask   

# Process test images
X_test = np.zeros((len(test_Ids), img_height, img_width, img_channels), dtype=np.uint8)
sizes_test = []
print('ðŸ§ª Resizing test images...')
for n, id_ in tqdm(enumerate(test_Ids), total=len(test_Ids)):
    path = os.path.join(Test_Path, id_)
    img_path = os.path.join(path, 'images', os.listdir(os.path.join(path, 'images'))[0])
    img = imread(img_path)[:,:,:img_channels]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    X_test[n] = img

print('âœ… Done!')

# Show a random image + mask
ix = random.randint(0, len(train_Ids)-1)
imshow(X_train[ix])
plt.title("Image")
plt.show()

imshow(np.squeeze(Y_train[ix]), cmap='gray')
plt.title("Mask")
plt.show()



#Build the model
inputs = tf.keras.layers.Input((img_height, img_width, img_channels))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


tf.keras.utils.plot_model(model, "model.png")


#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)


idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)
print("Prediction shape:", preds_test.shape)
 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


import random
import matplotlib.pyplot as plt
import numpy as np

# === Sanity check on training sample ===
ix = random.randint(0, len(preds_train_t) - 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(X_train[ix])
plt.title("Train Image")

plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(Y_train[ix]), cmap='gray')
plt.title("Ground Truth Mask")

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(preds_train_t[ix]), cmap='gray')
plt.title("Predicted Mask (Train)")

plt.show()


# === Sanity check on validation sample ===
val_start_idx = int(X_train.shape[0] * 0.9)
ix = random.randint(0, len(preds_val_t) - 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(X_train[val_start_idx + ix])
plt.title("Validation Image")

plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(Y_train[val_start_idx + ix]), cmap='gray')
plt.title("Ground Truth Mask")

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(preds_val_t[ix]), cmap='gray')
plt.title("Predicted Mask (Val)")

plt.show()


