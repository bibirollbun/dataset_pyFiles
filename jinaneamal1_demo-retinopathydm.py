# copy the weights and configurations for the pre-trained models 
!mkdir ~/.keras
!mkdir ~/.keras/models7
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import math



labels_path = "/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip"
train_images_path = "/kaggle/input/diabetic-retinopathy-train-unzipped/train/"
test_images_path = "/kaggle/input/diabetic-retinopathy-test-unzipped/test/"

labels_df = pd.read_csv(labels_path)
labels_df['image_path'] = labels_df['image'].apply(lambda name: os.path.join(train_images_path, f"{name}.jpeg"))
labels_df['is_present'] = labels_df['image_path'].apply(os.path.exists)
labels_df = labels_df[labels_df['is_present']]
labels_df.dropna(inplace=True)



unique_patients_df = labels_df[['image', 'level']].drop_duplicates()
train_patients, val_patients = train_test_split(
    unique_patients_df['image'],
    test_size=0.25,
    stratify=unique_patients_df['level'],
    random_state=42
)

train_df = labels_df[labels_df['image'].isin(train_patients)]
val_df = labels_df[labels_df['image'].isin(val_patients)]

# Avoid over-sampling class 0
target_size = train_df['level'].value_counts().median()

balanced_parts = []
for level in train_df['level'].unique():
    class_subset = train_df[train_df['level'] == level]
    if level == 0:
        balanced = class_subset.sample(int(target_size), random_state=42)
    else:
        balanced = resample(class_subset, replace=True, n_samples=int(target_size), random_state=42)
    balanced_parts.append(balanced)

balanced_train_df = pd.concat(balanced_parts)



train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

def generate_from_dataframe(df, datagen, batch_size=32, target_size=(224, 224), num_classes=5, shuffle=True):
    while True:
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        for start in range(0, len(df), batch_size):
            end = start + batch_size
            batch = df.iloc[start:end]
            imgs, labels = [], []
            for _, row in batch.iterrows():
                img = load_img(row['image_path'], target_size=target_size)
                imgs.append(img_to_array(img))
                labels.append(row['level'])
            X = np.array(imgs, dtype=np.float32)
            y = to_categorical(np.array(labels), num_classes=num_classes)
            for aug_X, aug_y in datagen.flow(X, y, batch_size=batch_size, shuffle=False):
                yield aug_X, aug_y
                break

num_classes = labels_df['level'].nunique()
train_generator = generate_from_dataframe(balanced_train_df, train_datagen, num_classes=num_classes)
valid_generator = generate_from_dataframe(val_df, valid_datagen, num_classes=num_classes, shuffle=False)



base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
preds = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])



callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

steps_per_epoch = len(balanced_train_df) // 32
validation_steps = len(val_df) // 32

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=callbacks
)



all_predictions = []
all_true_labels = []
steps = math.ceil(len(val_df) / 32)

for _ in range(steps):
    x_batch, y_batch = next(valid_generator)
    preds = model.predict(x_batch, verbose=0)
    all_predictions.extend(np.argmax(preds, axis=1))
    all_true_labels.extend(np.argmax(y_batch, axis=1))

cm = confusion_matrix(all_true_labels, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(all_true_labels, all_predictions, digits=4))



# === Load and Predict on Test Set ===
test_paths = glob("/kaggle/input/diabetic-retinopathy-test-unzipped/test/*.jpeg")
test_df = pd.DataFrame({"image": [os.path.basename(p) for p in test_paths], "image_path": test_paths})

def generate_test_batches(df, datagen, batch_size=32, target_size=(224, 224)):
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        imgs = []
        for path in batch['image_path']:
            img = load_img(path, target_size=target_size)
            imgs.append(img_to_array(img))
        X = np.array(imgs, dtype=np.float32)
        X /= 255.
        yield X
        
batch_size = 32
test_preds = []
test_steps = math.ceil(len(test_df) / batch_size)
test_gen = generate_test_batches(test_df, test_datagen, batch_size=batch_size)

for _ in range(test_steps):
    x_batch = next(test_gen)
    preds = model.predict(x_batch, verbose=0)
    test_preds.extend(np.argmax(preds, axis=1))

submission_df = pd.DataFrame({"image": test_df['image'], "level": test_preds})
submission_df.to_csv("submission.csv", index=False)


