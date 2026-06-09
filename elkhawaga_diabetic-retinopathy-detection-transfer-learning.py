# Basic libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight



# 2. Set directories and labels
data_dir = "/kaggle/input/diabetic-retinopathy-224x224-2019-data/colored_images"
labels = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
label_to_int = {label: idx for idx, label in enumerate(labels)}


# Count images per class
class_counts = {}
for cls in os.listdir(data_dir):  # <- use os.listdir here
    cls_path = os.path.join(data_dir, cls)
    count = len(os.listdir(cls_path))
    class_counts[cls] = count

print("Class distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")


# 3. Preview images using cv2
plt.figure(figsize=(20, 12))
images_per_label = 5
for i, label in enumerate(labels):
    folder_path = os.path.join(data_dir, label)
    for j in range(images_per_label):
        img_file = os.listdir(folder_path)[j]
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)                 # BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        plt.subplot(len(labels), images_per_label, i*images_per_label + j + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
plt.show()


# =========================
# 4. Compute class weights
# =========================
all_labels = []
for idx, cls in enumerate(labels):
    all_labels.extend([idx]*class_counts[cls])
all_labels = np.array(all_labels)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)


# 4. Create ImageDataGenerator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)




# Validation generator (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


# Training generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)



# Validation generator
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


# Load ResNet50 without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False


# Build Sequential model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(len(labels), activation='softmax')  # 5 classes
])


# Compile
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights

)




