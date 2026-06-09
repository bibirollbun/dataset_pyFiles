!pip install -q -U efficientnet tensorflow_addons==0.20.0 typeguard==2.13.3

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import efficientnet.tfkeras as efn
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_dir = '/kaggle/input/rsna-pneumonia-detection-challenge/'
labels_df = pd.read_csv(os.path.join(data_dir, 'stage_2_train_labels.csv'))
labels_df = labels_df.drop_duplicates('patientId')
labels_df['Target'] = labels_df['Target'].astype(str)


image_dir = os.path.join(data_dir, 'stage_2_train_images')
labels_df['path'] = labels_df['patientId'].apply(lambda x: os.path.join(image_dir, f"{x}.dcm"))


import pydicom

def load_dicom_image(path, resize=(224, 224)):
    dcm = pydicom.dcmread(path)
    image = dcm.pixel_array
    image = cv2.resize(image, resize)
    image = np.stack((image,) * 3, axis=-1)  # Convert to 3 channels
    image = image / 255.0
    return image


class PneumoniaDataset(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=16, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

        
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        df_batch = self.df.iloc[indexes]
        
        X = np.array([load_dicom_image(path) for path in df_batch['path']])
        y = df_batch['Target'].astype(int).values
        
        if self.augment:
            for i in range(len(X)):
                if np.random.rand() < 0.5:
                    X[i] = tf.image.flip_left_right(X[i])
        
        return X, y


train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['Target'], random_state=42)
train_gen = PneumoniaDataset(train_df, batch_size=16, augment=True)
val_gen = PneumoniaDataset(val_df, batch_size=16, augment=False)


from tensorflow.keras.applications import EfficientNetV2M

def build_model():
    base_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.summary()


history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=10)


# Save model
model.save("pneumonia_model.h5")


from tensorflow.keras.models import load_model
model = load_model("pneumonia_model.h5")


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axs[0].plot(history.history['accuracy'], label='Train Accuracy')
axs[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axs[0].set_title('Accuracy over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')
axs[0].legend()

# Loss plot
axs[1].plot(history.history['loss'], label='Train Loss')
axs[1].plot(history.history['val_loss'], label='Val Loss')
axs[1].set_title('Loss over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()

plt.tight_layout()
plt.show()



# Class Distribution
labels_df['Target'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Data Distribution (0 = Normal, 1 = Pneumonia)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
import os
import numpy as np
import pandas as pd
import cv2
import pydicom
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Load labels and paths
data_dir = '/kaggle/input/rsna-pneumonia-detection-challenge/'
labels_df = pd.read_csv(os.path.join(data_dir, 'stage_2_train_labels.csv'))
labels_df = labels_df.drop_duplicates('patientId')
labels_df['Target'] = labels_df['Target'].astype(int)
image_dir = os.path.join(data_dir, 'stage_2_train_images')
labels_df['path'] = labels_df['patientId'].apply(lambda x: os.path.join(image_dir, f"{x}.dcm"))

# Function to load and preprocess DICOM images
def load_dicom_image(path, resize=(224, 224)):
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        img = cv2.resize(img, resize)
        img = np.stack((img,) * 3, axis=-1)
        img = img / 255.0
        return img
    except:
        return None

# Load a small sample to avoid memory overload
sample_df = labels_df.sample(n=1000, random_state=42)
sample_df['image'] = sample_df['path'].apply(load_dicom_image)
sample_df = sample_df.dropna(subset=['image'])

# Prepare X and y
X = np.stack(sample_df['image'].values)
y = sample_df['Target'].values

# Split and load model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = load_model("pneumonia_model.h5")

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

# Accuracy
acc = accuracy_score(y_test, y_pred_labels)
print("âœ… Overall Accuracy on Test Set:", acc)


from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Predict
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)

# Output metrics
print(f"âœ… Accuracy:  {acc:.4f}")


# Classification Report
print("\nğŸ“‹ Classification Report:\n")
print(classification_report(y_test, y_pred_labels))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
disp.plot(cmap=plt.cm.Blues)
plt.title("ğŸ§  Confusion Matrix")
plt.show()


import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import pydicom

# ====== Paths ======
model_path = '/kaggle/working/pneumonia_model.h5'
image_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images/0005d3cc-3c3f-40b9-93c3-46231c3eb813.dcm'

# ====== Load Model ======
model = load_model(model_path)

# ====== Load and Preprocess ======
def load_image(img_path, target_size=(224, 224)):
    ext = os.path.splitext(img_path)[1].lower()

    if ext == '.dcm':
        dicom = pydicom.dcmread(img_path)
        img = dicom.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGB)
    else:
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

# ====== Predict and Show ======
def predict_and_show(img_path):
    img_array = load_image(img_path)
    input_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(input_array)[0][0]

    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

    display_img = (img_array * 255).astype(np.uint8)
    green = (0, 255, 0)
    display_img = cv2.putText(display_img.copy(), label, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)

    plt.imshow(display_img)
    plt.axis('off')
    plt.show()

# ğŸ”� Run prediction
predict_and_show(image_path)



import shutil
from IPython.display import FileLink

# ====== Zip the model ======
shutil.make_archive('pneumonia_model', 'zip', '/kaggle/working', 'pneumonia_model.h5')

# ====== Create Download Link ======
FileLink('pneumonia_model.zip')

