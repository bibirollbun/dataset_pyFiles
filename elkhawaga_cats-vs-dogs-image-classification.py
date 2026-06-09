#importsssssss
import zipfile
import os, cv2, re, random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras


import shutil

# Paths
train_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
train_dir = "/kaggle/working/train"

# 1. Clean old extraction if exists
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

# 2. Extract train.zip into /kaggle/working
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working")

# After extraction, images are in /kaggle/working/train/
print("Total extracted files:", len(os.listdir(train_dir)))
print("First 10 files:", os.listdir(train_dir)[:10])
print("Last 10 files:", os.listdir(train_dir)[-10:])




for root, dirs, files in os.walk("/kaggle/working"):
    level = root.replace("/kaggle/working", "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 4 * (level + 1)
    for f in files[:10]:  # show up to 10 files per folder
        print(f"{subindent}{f}")


train_image_paths = []
train_labels = []

for fname in os.listdir(train_dir):
    fpath = os.path.join(train_dir, fname)
    
    if os.path.getsize(fpath) <= 0:
        print(fname + " has not enough pixels, seems corrupted, ignoring.")
        continue  # skip corrupted file
    if ".jpg" not in fpath:
        print("not image: ", fname)
        continue
    if fname.startswith("cat"):
        train_labels.append(0)
        train_image_paths.append(fpath)
    elif fname.startswith("dog"):
        train_labels.append(1)
        train_image_paths.append(fpath)

print(len(train_image_paths), len(train_labels))
print("Cats:", train_labels.count(0))
print("Dogs:", train_labels.count(1))



# preprocessing and Convert to numpy arrays
train_images = []
for path in train_image_paths:
    img = cv2.imread(path)                      # read image (BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = cv2.resize(img, (150, 150))   # resize
    train_images.append(img)



train_images = np.array(train_images, dtype="float32")/255.0
train_labels = np.array(train_labels, dtype="int32")


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# Data Augmentation (Train)
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,          # normalize
    rotation_range=30,       # random rotation
    width_shift_range=0.1,   # horizontal shift
    height_shift_range=0.1,  # vertical shift
    zoom_range = 0.2,        # zoom in
    horizontal_flip=True     # random flip
)


# ---------------------------
# Validation Data (No Augmentation, only rescale)
# ---------------------------
val_datagen = ImageDataGenerator(rescale=1./255)


train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()


train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)


#checking images with its labels

import matplotlib.pyplot as plt

# pick 16 random indices
idx = np.random.choice(len(train_images), 16, replace=False)

plt.figure(figsize=(10, 10))
for i, index in enumerate(idx):
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[index].squeeze(), cmap="gray")  # squeeze in case of (128,128,1)
    plt.title("Dog" if train_labels[index] == 1 else "Cat")
    plt.axis("off")

plt.tight_layout()
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Build CNN model
# ---------------------------
model = Sequential([
            Conv2D(16, (3, 3),activation='relu',input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])


#combiling model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ---------------------------
# Early stopping
# ---------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# ---------------------------
# Train the model
# ---------------------------
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=16, shuffle=True),
    validation_data=(X_val, y_val),   # (numpy arrays for validation set)
    epochs=20,
    callbacks=[early_stop]
)


import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
test_extract_path = "working/"

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_extract_path)

print("Extracted to:", test_extract_path)
print(os.listdir("working/"))  # should show 'train' and 'test'


test_dir = "working/test"
img_size = (150, 150)  # same size used in training

# Image generator (only rescaling, no augmentation for test)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow from directory (no labels since it's test data)
test_generator = test_datagen.flow_from_directory(
    directory="working",         # parent dir
    classes=["test"],            # only the test folder
    target_size=(150, 150),      # use same size as training
    batch_size=32,
    class_mode=None,             # no labels
    shuffle=False                # IMPORTANT to keep order
)


# Predict probabilities
preds = model.predict(test_generator, verbose=1)

# Flatten if needed
preds = preds.ravel()

# Extract ids from filenames
ids = [os.path.basename(f).split(".")[0] for f in test_generator.filenames]



# Create dataframe
submission = pd.DataFrame({"id": ids, "label": preds})




submission["id"] = submission["id"].astype(int)  # make sure ids are int
submission = submission.sort_values("id")  # ensure correct order
submission.to_csv("submission.csv", index=False)


print(submission.head(10))
print(submission.tail(10))




