# 1) IMPORTS & CONSTANTS
import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

IMG_SIZE    = 224
BATCH_SIZE  = 32
BASE_LR     = 1e-4
EPOCHS      = 5

INPUT_DIR   = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'
WORK_DIR    = '/kaggle/working/data'

# 2) UNZIP (only if you haven’t already)
!unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip  -d {WORK_DIR}
!unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip   -d {WORK_DIR}

# 3) SET PATHS
TRAIN_DIR = f'{WORK_DIR}/train'
TEST_DIR  = f'{WORK_DIR}/test'

print("Train JPG count:", len([f for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')]))
print("Test  JPG count:", len([f for f in os.listdir(TEST_DIR)  if f.endswith('.jpg')]))


# Display sample images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 4) CREATE DF
filenames = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')]
labels    = [fn.split('.')[0] for fn in filenames]   # 'dog' or 'cat'
df = pd.DataFrame({'filename': filenames, 'label': labels})

# Plot label distribution
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Label Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(False)
plt.show()

sample_files = np.random.choice(filenames, size=8, replace=False)

plt.figure(figsize=(14, 6))
for i, file in enumerate(sample_files):
    img_path = os.path.join(TRAIN_DIR, file)
    img = mpimg.imread(img_path)
    label = file.split('.')[0]
    
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    
plt.tight_layout()
plt.suptitle("Sample Images from Training Set", fontsize=16, y=1.05)
plt.show()


# 4) BUILD DF
filenames = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')]
labels    = [fn.split('.')[0] for fn in filenames]

df = pd.DataFrame({'filename': filenames, 'label': labels})
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 5) DATA GENERATORS
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect'
)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_dataframe(
    train_df, TRAIN_DIR,
    x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    seed=42
)
val_gen = val_aug.flow_from_dataframe(
    val_df, TRAIN_DIR,
    x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE,
    seed=42
)


# 6) BUILD & COMPILE MODEL (PHASE 1: HEAD ONLY)
from tensorflow.keras.applications import ResNet50V2

base = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x   = base(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.Dropout(0.5)(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inp, out)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)




# 7) CALLBACKS
cb = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

# 8) TRAIN HEAD ONLY
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=cb
)


# 9) FINE-TUNE BASE MODEL
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    callbacks=cb
)


# 10) PREPARE TEST DATA
test_fns = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')])
test_df  = pd.DataFrame({'filename': test_fns})
test_aug = ImageDataGenerator(rescale=1./255)

test_gen = test_aug.flow_from_dataframe(
    test_df, TEST_DIR,
    x_col='filename', y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 11) PREDICT & SUBMIT
preds = model.predict(test_gen, verbose=1)

submission = pd.DataFrame({
    'id':    [int(fn.split('.')[0]) for fn in test_fns],
    'label': preds.ravel()
})
submission.to_csv('submission.csv', index=False)


!ls -lh /kaggle/working




