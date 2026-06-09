!pip install -q py7zr

import os, py7zr, pandas as pd, numpy as np
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# 1️⃣ Extract train.7z if you haven’t already
py7zr.SevenZipFile("/kaggle/input/cifar-10/train.7z", mode='r')\
     .extractall(path="/kaggle/working/train")

# 2️⃣ Read labels and convert class names to integers
labels_df = pd.read_csv("/kaggle/input/cifar-10/trainLabels.csv")
le = LabelEncoder()
labels_df["label"] = le.fit_transform(labels_df["label"])  # Fix applied here

# 3️⃣ Load images and labels
def load_images(df, folder):
    X, y = [], []
    for img_id, lbl in tqdm(df.values, desc="Loading images"):
        img = Image.open(f"{folder}/{img_id}.png").resize((32, 32))
        X.append(np.array(img))
        y.append(lbl)
    return np.stack(X), np.array(y, dtype='int32')

x, y = load_images(labels_df, "/kaggle/working/train/train")
x = x.astype('float32') / 255.0
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.1,
                                            stratify=y, random_state=42)

# 4️⃣ Build & compile model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

es = callbacks.EarlyStopping(monitor='val_loss', patience=3,
                             restore_best_weights=True)

# 5️⃣ Train
print("GPU available:", tf.config.list_physical_devices('GPU'))
history = model.fit(x_tr, y_tr,
                    validation_data=(x_val, y_val),
                    epochs=20, batch_size=64,
                    callbacks=[es], verbose=2)

# 6️⃣ Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend()
plt.show()




import py7zr
import os

# Only extract if not already done
if not os.path.exists("/kaggle/working/test"):
    with py7zr.SevenZipFile("/kaggle/input/cifar-10/test.7z", mode='r') as archive:
        archive.extractall(path="/kaggle/working/test")



import os
from PIL import Image
import numpy as np

test_dir = "/kaggle/working/test/test"
test_images = []

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"{test_dir} does not exist. Check extraction path.")

image_files = [f for f in os.listdir(test_dir) if f.endswith(".png") and f.split('.')[0].isdigit()]
image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))

# Load and preprocess
for img_file in image_files:
    img_path = os.path.join(test_dir, img_file)
    img = Image.open(img_path).resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    test_images.append(img_array)

X_test = np.array(test_images)
print(f"Loaded {len(X_test)} test images.")





predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)



import pandas as pd

file_names = [f"{i}.png" for i in range(1, len(predicted_classes) + 1)]

# Create DataFrame
submission_df = pd.DataFrame({
    "Id": file_names,
    "Label": predicted_classes
})

submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully.")


