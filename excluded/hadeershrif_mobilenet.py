# Cell 1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



import os

# Example: replace this with the actual shared folder path
shared_folder = '/kaggle/input/state-farm-distracted-driver-detection/imgs/train'

# Make sure the folder exists
assert os.path.exists(shared_folder), "❌ The path doesn't exist. Double-check the shared folder path."




img_size = 224  # MobileNetV2 input size
batch_size = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)




from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

base_model = MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()




from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.2)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)



import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.grid()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.grid()
plt.show()




import os
import numpy as np
import cv2

test_dir = "/content/drive/Shareddrives/YOUR_SHARED_DRIVE_NAME/test"
test_images = []
test_image_names = []

for img_name in sorted(os.listdir(test_dir)):
    img_path = os.path.join(test_dir, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype("float32") / 255.0
        test_images.append(img)
        test_image_names.append(img_name)

test_images = np.array(test_images)

# Predict
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Map back to labels
class_labels = list(train_generator.class_indices.keys())
for i in range(min(10, len(test_image_names))):
    print(f"Image: {test_image_names[i]}, Predicted Class: {class_labels[predicted_classes[i]]}")




import cv2
import numpy as np
import matplotlib.pyplot as plt

class_labels = [
    'safe driving', 'texting - right', 'talking on the phone - right', 'texting - left',
    'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
    'hair and makeup', 'talking to passenger'
]

test_image_path = "/kaggle/input/mmmmmm/WhatsApp Image 2025-05-11 at 7.31.17 PM.jpeg"
img_size = 224  # adjust as per model requirement

# Load and preprocess image
img = cv2.imread(test_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for displaying with correct colors
img_resized = cv2.resize(img, (img_size, img_size))
img_input = img_resized.astype("float32") / 255.0
img_input = np.expand_dims(img_input, axis=0)

# Predict
prediction = model.predict(img_input)
predicted_class = np.argmax(prediction)
predicted_label = class_labels[predicted_class]

# Display image with label
plt.imshow(img_rgb)
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()



