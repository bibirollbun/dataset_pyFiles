import tensorflow as tf
import os
import numpy as np
import random
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


import zipfile
TRAIN_ZIP = '/kaggle/input/data-science-bowl-2018/stage1_train.zip'
TEST_ZIP = '/kaggle/input/data-science-bowl-2018/stage1_test.zip'
# Directory where to extract
TRAIN_PATH = '/kaggle/working/stage1_train/'
TEST_PATH = '/kaggle/working/stage1_test/'

# Unzip the train data
with zipfile.ZipFile(TRAIN_ZIP, 'r') as zip_ref:
    zip_ref.extractall(TRAIN_PATH)

# Unzip the test data
with zipfile.ZipFile(TEST_ZIP, 'r') as zip_ref:
    zip_ref.extractall(TEST_PATH)

print("Extraction complete!")


train_ids = next(os.walk(TRAIN_PATH))[1] #gives the folder names
test_ids = next(os.walk(TEST_PATH))[1]


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)


X_train.shape, Y_train.shape


seed = 42
np.random.seed = seed


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # After resizing, ensure the data is in uint8 format and the range [0, 255]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)  # Clip values
        img = img.astype(np.uint8)  # Convert to uint8
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask_ = mask_.astype(np.uint8)
        
        mask = np.maximum(mask, mask_)  
    # Convert the mask to boolean
    mask = (mask > 0).astype(bool)    
    Y_train[n] = mask   


X_train[0].shape, Y_train[0].shape


imshow(X_train[0])
plt.show()
imshow(Y_train[0])
plt.show()


# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # print(img.dtype)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)  # Clip values
        img = img.astype(np.uint8)  # Convert to uint8
    X_test[n] = img

print('Done!')


sizes_test[1]
imshow(X_test[1])
X_test[1].max()


image_x = random.randint(0, len(train_ids))
print(image_x)

imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
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



import datetime


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Defining callbacks
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.keras', verbose=1, save_best_only=True)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model_for_nuclei.keras', verbose=1, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
#     tf.keras.callbacks.TensorBoard(log_dir='logs')
    tensorboard_callback
]


# Clear any logs from previous runs
!rm -rf /kaggle/working/logs


results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=35, callbacks=callbacks)


# type(results)
results.history


history = results.history
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


history = results.history
plt.plot(history['accuracy'], label='Accuracy')
plt.show()


model.input_shape


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input image', 'True mask', 'Predicted mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


idx = random.randint(0, len(X_train))
true_image = X_train[idx]
true_mask = np.squeeze(Y_train[idx])
# Add batch dimension and then predict
prediction = model.predict(true_image[tf.newaxis, ...])[0]

predicted_mask = (prediction > 0.5).astype(np.uint8)
print(predicted_mask.shape)
display([true_image, true_mask,predicted_mask])


model.save('/kaggle/working/unet.h5')


predicted_masks = model.predict(X_train)
# For binary segmentation, we usually apply a threshold of 0.5
predicted_masks = (predicted_masks > 0.5).astype(np.uint8)

# Now `predicted_masks` is an array of shape (670, 128, 128, 1), similar to `y_train`
print(predicted_masks.shape) 


# Create directories to save the true and predicted masks
true_masks_folder = '/kaggle/working/true_masks'
predicted_masks_folder = '/kaggle/working/predicted_masks'
os.makedirs(true_masks_folder, exist_ok=True)
os.makedirs(predicted_masks_folder, exist_ok=True)



import shutil
import imageio
# Assuming `true_masks` and `predicted_masks` are numpy arrays of shape (670, 128, 128, 1)
# Iterate through all masks and save them as PNG images
for i in range(len(X_train)):  # Adjust the range based on the number of masks
    # Get the i-th true and predicted mask
    true_mask = np.squeeze(Y_train[i])  # Squeeze to remove the single channel dimension (128, 128)
    predicted_mask = np.squeeze(predicted_masks[i])

    # Create the paths to save the images
    true_mask_save_path = os.path.join(true_masks_folder, f'true_mask_{i}.png')
    predicted_mask_save_path = os.path.join(predicted_masks_folder, f'predicted_mask_{i}.png')

    # Save the true and predicted masks as images
    imageio.imwrite(true_mask_save_path, (true_mask * 255).astype(np.uint8))  # Scale if necessary
    imageio.imwrite(predicted_mask_save_path, (predicted_mask * 255).astype(np.uint8))

print("True and Predicted masks saved in their respective folders.")


# Now create zip archives for both directories
shutil.make_archive('/kaggle/working/true_masks_archive', 'zip', true_masks_folder)
shutil.make_archive('/kaggle/working/predicted_masks_archive', 'zip', predicted_masks_folder)

print("Masks folders zipped successfully.")































