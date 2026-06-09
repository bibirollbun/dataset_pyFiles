
!pip install -q py7zr tqdm

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)






from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='reflect'
)
datagen.fit(x_train)



from tensorflow.keras import layers, models, regularizers

def build_cifar_model():
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',
                            kernel_initializer='he_normal', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    # Block 2
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # Block 3
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    return model

model = build_cifar_model()
model.summary()



from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-5),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    steps_per_epoch=x_train.shape[0] // 64,
    epochs=60,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=2
)



print("Step 1: Imports running...")

import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tqdm import tqdm

print("Step 1: Imports complete âœ…")



import py7zr
import os
import glob

print("â�³ Extracting Kaggle test set (300,000 images)...")

test_archive = "/kaggle/input/cifar-10/test.7z"

with py7zr.SevenZipFile(test_archive, mode='r') as archive:
    archive.extractall(path="./test")

print("âœ… Extraction complete!")

# Recursively check all PNG files
num_files = len(glob.glob('./test/**/*.png', recursive=True))
print("ğŸ–¼ï¸� Total extracted images:", num_files)




print("â�³ Step 1: Imports running...")

# Safe installation for version saving
try:
    import py7zr
except ModuleNotFoundError:
    !pip install -q py7zr tqdm
    import py7zr

import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

print("âœ… Step 1: Imports complete")


import os
import cv2
import numpy as np
from tqdm import tqdm

print("â�³ Preprocessing Kaggle test images...")

test_path = "./test/"  

# Recursively search for PNG files in all subfolders
file_names = sorted(
    [os.path.join(dp, f) for dp, dn, fn in os.walk(test_path) for f in fn if f.endswith('.png')],
    key=lambda x: int(os.path.basename(x).split('.')[0])
)

X_test = []
for f in tqdm(file_names, desc="Processing images"):
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0
    X_test.append(img)

X_test = np.array(X_test)

print("âœ… Preprocessing complete! Final test shape:", X_test.shape)



print("â�³ Running predictions on Kaggle test set...")

# Predict probabilities
pred_probs = model.predict(X_test, batch_size=128, verbose=1)

# Convert to class labels
pred_classes = np.argmax(pred_probs, axis=1)

print("âœ… Predictions complete! Sample output:", pred_classes[:10])



# CIFAR-10 label mapping
label_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# Convert numeric predictions to text labels
pred_labels = [label_map[i] for i in pred_classes]

# Build submission DataFrame with numeric IDs
submission = pd.DataFrame({
    "id": range(1, len(pred_labels) + 1),   # strictly 1,2,3...300000
    "label": pred_labels
})

# Save correctly formatted CSV
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created with correct ID format!")
submission.head()


