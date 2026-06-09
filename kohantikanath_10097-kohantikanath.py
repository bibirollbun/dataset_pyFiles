# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import cv2
from PIL import Image
import os
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import tensorflow as tf






input_folder = "/kaggle/input/petfinder-pawpularity-score/train"
output_folder = "processed_images"
target_size = (224, 224)  # Change to (224,224) if using ResNet/other pretrained

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        img = cv2.imread(os.path.join(input_folder, file))
        img_resized = cv2.resize(img, target_size)
        cv2.imwrite(os.path.join(output_folder, file), img_resized)


input_folder = "/kaggle/input/petfinder-pawpularity-score/test"
output_folder = "test_processed_images"
target_size = (224, 224)  # Change to (224,224) if using ResNet/other pretrained

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        img = cv2.imread(os.path.join(input_folder, file))
        img_resized = cv2.resize(img, target_size)
        cv2.imwrite(os.path.join(output_folder, file), img_resized)


df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
df


df_test = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
df_test


df['Pawpularity'] = df['Pawpularity'] / 100.0

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)


metadata_cols = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]


X_metadata = train_df[metadata_cols].values
y = train_df['Pawpularity'].values
Xval_metadata = valid_df[metadata_cols].values
y_val = valid_df['Pawpularity'].values



Xtest_metadata = df_test[metadata_cols].values


train_df['Pawpularity'].values


IMG_DIR = "/kaggle/input/petfinder-pawpularity-score/train"  # folder with images
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


image_dir = "/kaggle/working/processed_images/"

X_images = np.array([
    image.img_to_array(
        image.load_img(f"{image_dir}/{img_id}.jpg", target_size=(224, 224))
    ) / 255.0
    for img_id in train_df['Id']
])


image_dir = "/kaggle/working/processed_images/"

Xval_images = np.array([
    image.img_to_array(
        image.load_img(f"{image_dir}/{img_id}.jpg", target_size=(224, 224))
    ) / 255.0
    for img_id in valid_df['Id']
])


test_image_dir = "/kaggle/working/test_processed_images/"
Xtest_images = np.array([
    image.img_to_array(
        image.load_img(f"{test_image_dir}/{img_id}.jpg", target_size=(224, 224))
    ) / 255.0
    for img_id in df_test['Id']
])


X_images = preprocess_input(X_images)
Xval_images = preprocess_input(Xval_images)


y


datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)


def multi_input_generator(X_images, X_metadata, y, batch_size, datagen):
    n = len(X_images)
    i = 0
    while True:
        batch_images = []
        batch_metadata = []
        batch_y = []
        
        for _ in range(batch_size):
            if i >= n:
                i = 0
            img = X_images[i]
            meta = X_metadata[i]
            label = y[i]
            i += 1

            # Apply augmentation to the image
            img = datagen.random_transform(img)
            img = datagen.standardize(img)

            batch_images.append(img)
            batch_metadata.append(meta)
            batch_y.append(label)

        yield ((np.array(batch_images, dtype=np.float32), 
                np.array(batch_metadata, dtype=np.float32)),
               np.array(batch_y, dtype=np.float32))



from tensorflow.keras.applications import EfficientNetB3



from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import EfficientNetB0  # ✅ use EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3


num_metadata_features = len(metadata_cols)

# Image branch
image_input = Input(shape=(224, 224, 3))
# model = EfficientNetB3(include_top=False, weights='imagenet')
base_model = EfficientNetB3(
    include_top=False,
    weights=None,
    input_tensor=image_input,
    pooling='avg'
)
base_model.load_weights("/kaggle/input/weights-efficientnetb3/efficientnetb3_weights.weights.h5")

for layer in base_model.layers:
    layer.trainable = False


image_features = base_model.output

# Metadata branch
metadata_input = Input(shape=(num_metadata_features,))
meta_features = layers.Dense(128, activation='relu')(metadata_input)
meta_features = layers.Dense(64, activation='relu')(meta_features)

# Combine both branches
combined = layers.Concatenate()([image_features, meta_features])

# Final regression head
x = layers.Dense(256, activation='relu')(combined)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation='linear')(x)

# Final model
model = Model(inputs=[image_input, metadata_input], outputs=output)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)


early_stop = EarlyStopping(monitor='val_rmse', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_rmse', factor=0.5, patience=3)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_rmse', mode='min')


batch_size = 32

train_gen = tf.data.Dataset.from_generator(
    lambda: multi_input_generator(X_images, X_metadata, y, batch_size, datagen),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_metadata_features), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)
train_gen = train_gen.prefetch(tf.data.AUTOTUNE)


train_gen = train_gen.prefetch(tf.data.AUTOTUNE)
steps_per_epoch = len(X_images) // batch_size



history = model.fit(
    train_gen,
    validation_data=([Xval_images, Xval_metadata], y_val),
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)


for layer in base_model.layers[-50:]:
    layer.trainable = True


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)



history_finetune = model.fit(
    train_gen,
    validation_data=([Xval_images, Xval_metadata], y_val),
    epochs=18,
    steps_per_epoch=steps_per_epoch,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)


preds = model.predict([Xtest_images, Xtest_metadata]) * 100  # scale back to [0,100]
print(preds)



submission = pd.DataFrame({
    'Id': df_test['Id'],  
    'Pawpularity': preds.flatten()
})


submission.to_csv('submission.csv', index=False)

