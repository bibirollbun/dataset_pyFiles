# ---- Imports ----
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # optional, for nicer confusion matrix plots


DATA = Path('/kaggle/input/cassava-leaf-disease-classification')
train_csv = pd.read_csv(DATA/'train.csv')         # columns: image_id, label
with open(DATA/'label_num_to_disease_map.json') as f:
    label_map = json.load(f)

train_csv['filepath'] = train_csv['image_id'].apply(lambda x: str(DATA/'train_images'/x))
num_classes = train_csv['label'].nunique()
train_csv.head(), label_map, num_classes



train_csv['label'].value_counts().sort_index().plot(kind='bar')
plt.title('Class counts'); plt.show()

def show_samples(df, n=10):
    sample = df.sample(n)
    plt.figure(figsize=(12,8))
    for i,(fp,lab) in enumerate(zip(sample['filepath'], sample['label'])):
        plt.subplot(2, n//2, i+1); plt.imshow(plt.imread(fp)); plt.axis('off')
        plt.title(f"{lab}: {label_map[str(lab)]}")
    plt.tight_layout(); plt.show()

show_samples(train_csv, n=8)



#Create Data Generators
IMG_SIZE = (150, 200)
BATCH = 32
SEED = 2025

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    train_csv, x_col='filepath', y_col='label',
    target_size=IMG_SIZE, class_mode='raw',   # ← raw returns y as-is
    batch_size=BATCH, shuffle=True, seed=SEED, subset='training'
)

val_gen = datagen.flow_from_dataframe(
    train_csv, x_col='filepath', y_col='label',
    target_size=IMG_SIZE, class_mode='raw',
    batch_size=BATCH, shuffle=False, seed=SEED, subset='validation'
)


model_cnn2 = models.Sequential([
    layers.Input(shape=(150, 200, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    # Block 4
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)   # L2 regularization
    ),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model_cnn2.summary()



model_cnn2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cnn2 = keras.callbacks.ModelCheckpoint(
    "stronger_cnn_cassava.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_cnn2 = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

reduce_lr_cnn2 = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

history_cnn2 = model_cnn2.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,                
    callbacks=[checkpoint_cnn2, early_cnn2, reduce_lr_cnn2]
)

plt.plot(history_cnn2.history['accuracy'], label='train acc (cnn2)')
plt.plot(history_cnn2.history['val_accuracy'], label='val acc (cnn2)')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Stronger CNN Accuracy')
plt.show()


IMG_TL = (224, 224)
BATCH_TL = 32
SEED = 2025  # reuse if you like

# Train generator with augmentation + EfficientNet preprocessing
train_datagen_tl = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation generator (no augmentation, only preprocessing)
val_datagen_tl = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen_tl = train_datagen_tl.flow_from_dataframe(
    train_csv,
    x_col='filepath',
    y_col='label',
    target_size=IMG_TL,
    class_mode='raw',          # integer labels
    batch_size=BATCH_TL,
    shuffle=True,
    seed=SEED,
    subset='training'
)

val_gen_tl = val_datagen_tl.flow_from_dataframe(
    train_csv,
    x_col='filepath',
    y_col='label',
    target_size=IMG_TL,
    class_mode='raw',
    batch_size=BATCH_TL,
    shuffle=False,
    seed=SEED,
    subset='validation'
)



# ---- Build EfficientNetB0 model ----
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)

# Stage 1: freeze backbone
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model_eff = keras.Model(inputs, outputs)

model_eff.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ ONE checkpoint only
checkpoint_eff = keras.callbacks.ModelCheckpoint(
    "efficientnetb0_cassava.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_eff = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

reduce_lr_eff = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# ---------------- Stage 1: frozen backbone ----------------
history_eff = model_eff.fit(
    train_gen_tl,
    validation_data=val_gen_tl,
    epochs=15,
    callbacks=[checkpoint_eff, early_eff, reduce_lr_eff]
)

# ---------------- Stage 2: fine-tune top layers ----------------
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model_eff.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_eff_ft = model_eff.fit(
    train_gen_tl,
    validation_data=val_gen_tl,
    epochs=10,
    callbacks=[checkpoint_eff, early_eff, reduce_lr_eff]  # ✅ reuse same checkpoint
)



# Load the best EfficientNetB0 model
best_eff = tf.keras.models.load_model("efficientnetb0_cassava.h5")

# Predict on validation set
val_gen_tl.reset()
y_true = val_gen_tl.labels.astype(int)
y_prob = best_eff.predict(val_gen_tl)
y_pred = np.argmax(y_prob, axis=1)

print("Classification report (EfficientNetB0):")
print(classification_report(y_true, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=False, fmt='d')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix – EfficientNetB0')
plt.show()

# Misclassified samples
mis_idx = np.where(y_true != y_pred)[0][:16]
mis_images, mis_true, mis_pred = [], [], []

for i in mis_idx:
    img, label = val_gen_tl[i]
    mis_images.append(img[0])
    mis_true.append(label[0])
    mis_pred.append(y_pred[i])

plt.figure(figsize=(12, 12))
for i, (img, t, p) in enumerate(zip(mis_images, mis_true, mis_pred)):
    plt.subplot(4, 4, i + 1)
    img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imshow(img_disp)
    plt.axis('off')
    plt.title(f"True: {t}\nPred: {p}")

plt.suptitle('Misclassified Validation Samples – EfficientNetB0', y=0.92)
plt.show()



# ---------------- SAVE BEST + FINAL MODEL + HISTORY ----------------

# Best model already saved by checkpoint: efficientnetb0_cassava.h5

# Save final model after training
model_eff.save("efficientnetb0_cassava_final.keras")
print("Final model saved as efficientnetb0_cassava_final.keras")

# Save combined training history
combined_history = {
    "stage1": history_eff.history,
    "stage2": history_eff_ft.history
}

with open("efficientnetb0_cassava_history.pkl", "wb") as f:
    pickle.dump(combined_history, f)

print("History saved as efficientnetb0_cassava_history.pkl")

# Show files in directory
import os
print(os.listdir("."))


