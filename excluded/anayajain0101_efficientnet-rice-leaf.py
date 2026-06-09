# fast_train_script_15epochs.py
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Config / reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SOURCE_DIR = '/kaggle/input/paddy-disease-classification/train_images'
TRAIN_DIR = '/kaggle/working/train'
TEST_DIR = '/kaggle/working/test'
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# -------------------------
# Data split (20% test)
# -------------------------
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=SEED, shuffle=True)

    train_cdir = os.path.join(TRAIN_DIR, class_name)
    test_cdir = os.path.join(TEST_DIR, class_name)
    os.makedirs(train_cdir, exist_ok=True)
    os.makedirs(test_cdir, exist_ok=True)

    for f in train_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(train_cdir, f)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    for f in test_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(test_cdir, f)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

print("Data split done.")
print("Train images:", sum(len(os.listdir(os.path.join(TRAIN_DIR, d))) for d in os.listdir(TRAIN_DIR)))
print("Test images:", sum(len(os.listdir(os.path.join(TEST_DIR, d))) for d in os.listdir(TEST_DIR)))

# -------------------------
# Hyperparameters
# -------------------------
BATCH_SIZE = 32
IMG_SIZE = (300, 300)
INITIAL_LR = 1e-4
EPOCHS = 15             # fixed to 15

# -------------------------
# Generators (lighter augmentation)
# -------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)


eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED
)

test_ds = eval_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

NUM_CLASSES = len(train_ds.class_indices)
STEPS_PER_EPOCH = len(train_ds)
VALIDATION_STEPS = len(test_ds)
print(f"Classes: {NUM_CLASSES}, steps_per_epoch: {STEPS_PER_EPOCH}, val_steps: {VALIDATION_STEPS}")

# -------------------------
# Model
# -------------------------
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base.trainable = False  # freeze base

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(NUM_CLASSES, activation='softmax')
])

# -------------------------
# Compile & fit
# -------------------------
opt = Adam(learning_rate=INITIAL_LR)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stop],
    verbose=2
)

# -------------------------
# Fine-tune EfficientNet
# -------------------------

base.trainable = True   # unfreeze EfficientNet

from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(1e-5),   # smaller LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,     # 8–12 fine-tuning epochs
    verbose=2
)

# -------------------------
# Plots: accuracy / loss
# -------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -------------------------
# Confusion matrix & overall metrics
# -------------------------
test_ds.reset()
pred_probs = model.predict(test_ds, steps=VALIDATION_STEPS, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_ds.classes
idx_to_class = {v:k for k,v in test_ds.class_indices.items()}
labels_ordered = [idx_to_class[i] for i in range(NUM_CLASSES)]

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_ordered)
plt.figure(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=plt.gca())
plt.title('Confusion Matrix - Test Set')
plt.show()

# Overall metrics
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='weighted')
rec = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')

print("\nOverall Test Metrics:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

# -------------------------
# Save model
# -------------------------
model.save('efficientnetb0_model.h5')
print("Saved model -> efficientnetb0_model.h5")



# -------------------------
# Evaluation (Train & Test Accuracy)
# -------------------------
print("\n--- Final Evaluation ---")
train_loss, train_acc = model.evaluate(train_ds, verbose=0)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")


