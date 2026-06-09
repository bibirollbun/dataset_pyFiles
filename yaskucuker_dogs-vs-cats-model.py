
import os
import zipfile
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------
# 1ï¸�âƒ£ Veri YÃ¼kleme
# --------------------------

input_dir = '/kaggle/input/dogs-vs-cats/'
train_zip = os.path.join(input_dir, 'train.zip')
test_zip = os.path.join(input_dir, 'test1.zip')

# Zipleri aÃ§ma
with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train')
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')

original_train_dir = '/kaggle/working/train/train/'

# --------------------------
# 2ï¸�âƒ£ Train / Validation Split
# --------------------------

# Split klasÃ¶rÃ¼
base_dir = '/kaggle/working/train_split_clean/'

train_cats_dir = os.path.join(base_dir, 'train/cats')
train_dogs_dir = os.path.join(base_dir, 'train/dogs')
val_cats_dir = os.path.join(base_dir, 'validation/cats')
val_dogs_dir = os.path.join(base_dir, 'validation/dogs')

for d in [train_cats_dir, train_dogs_dir, val_cats_dir, val_dogs_dir]:
    os.makedirs(d, exist_ok=True)

# DosyalarÄ± karÄ±ÅŸtÄ±r ve ayÄ±r
all_cats = [f for f in os.listdir(original_train_dir) if f.startswith('cat')]
all_dogs = [f for f in os.listdir(original_train_dir) if f.startswith('dog')]
random.shuffle(all_cats)
random.shuffle(all_dogs)

split_ratio = 0.2
split_cats = int(len(all_cats)*split_ratio)
split_dogs = int(len(all_dogs)*split_ratio)

train_cats_files = all_cats[split_cats:]
val_cats_files = all_cats[:split_cats]
train_dogs_files = all_dogs[split_dogs:]
val_dogs_files = all_dogs[:split_dogs]

# DosyalarÄ± kopyala
for f in train_cats_files:
    shutil.copy(os.path.join(original_train_dir,f), train_cats_dir)
for f in val_cats_files:
    shutil.copy(os.path.join(original_train_dir,f), val_cats_dir)
for f in train_dogs_files:
    shutil.copy(os.path.join(original_train_dir,f), train_dogs_dir)
for f in val_dogs_files:
    shutil.copy(os.path.join(original_train_dir,f), val_dogs_dir)

print("âœ… Train / Validation klasÃ¶rleri hazÄ±r.")
print("Train Cats:", len(os.listdir(train_cats_dir)))
print("Train Dogs:", len(os.listdir(train_dogs_dir)))
print("Val Cats:", len(os.listdir(val_cats_dir)))
print("Val Dogs:", len(os.listdir(val_dogs_dir)))

# --------------------------
# 3ï¸�âƒ£ Data Generators
# --------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir,'train'),
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir,'validation'),
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

# Ã–rnek gÃ¶rseller
sample_imgs, _ = next(train_generator)
plt.figure(figsize=(12,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(sample_imgs[i])
    plt.axis('off')
plt.show()

# --------------------------
# 4ï¸�âƒ£ CNN Modeli
# --------------------------

model = Sequential([
    Input(shape=(150,150,3)),
    Conv2D(16,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------
# 5ï¸�âƒ£ Modeli EÄŸit
# --------------------------

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)

# --------------------------
# 6ï¸�âƒ£ DeÄŸerlendirme
# --------------------------

# Confusion Matrix
y_pred_val = (model.predict(validation_generator) > 0.5).astype(int).reshape(-1)
true_labels = validation_generator.classes
labels = list(validation_generator.class_indices.keys())

cm = confusion_matrix(true_labels, y_pred_val)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (Validation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("ğŸ“Œ Classification Report (Validation)")
print(classification_report(true_labels, y_pred_val, target_names=labels))

# Ã–rnek tahminler
sample_images, _ = next(validation_generator)
plt.figure(figsize=(12,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(sample_images[i])
    pred = "Dog" if model.predict(sample_images[i][np.newaxis,...])[0][0] > 0.5 else "Cat"
    plt.title(f"Tahmin: {pred}")
    plt.axis('off')
plt.show()



