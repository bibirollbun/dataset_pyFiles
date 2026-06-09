

import os
import random
import shutil
import zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import itertools
import seaborn as sns

# ----------------------------- Reproducibility -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ----------------------------- Paths & params ------------------------------
TRAIN_ZIP = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
TEST_ZIP  = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
WORK_DIR = Path("/kaggle/working/dvc_scratch")
DATA_DIR = WORK_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 50
AUTOTUNE = tf.data.AUTOTUNE

# L2 regularization weight
L2 = 1e-4

# Create working dirs
for d in [WORK_DIR, DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------- Helper functions ----------------------------

def extract_zip(zip_path, extract_to):
    """Extract zip if not already extracted."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(path=extract_to)


def organize_train_images(extracted_train_dir, train_out_dir, val_out_dir, val_split=0.15):
    """
    Move images to class subfolders and split into train/val.
    expected filenames: cat.<id>.jpg and dog.<id>.jpg
    """
    extracted_train_dir = Path(extracted_train_dir)
    train_out_dir = Path(train_out_dir)
    val_out_dir = Path(val_out_dir)

    # Ensure class folders exist
    for cls in ["cats", "dogs"]:
        (train_out_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_out_dir / cls).mkdir(parents=True, exist_ok=True)

    # List files
    files = list(extracted_train_dir.glob('*.jpg'))
    print(f"Found {len(files)} training images in extracted folder")

    # Separate by class
    cat_files = [f for f in files if f.name.startswith('cat')]
    dog_files = [f for f in files if f.name.startswith('dog')]
    print(f"Cats: {len(cat_files)}, Dogs: {len(dog_files)}")

    def split_and_copy(file_list, cls_name):
        random.shuffle(file_list)
        n_val = int(len(file_list) * val_split)
        val_files = file_list[:n_val]
        train_files = file_list[n_val:]
        for f in train_files:
            shutil.copy(f, train_out_dir / cls_name / f.name)
        for f in val_files:
            shutil.copy(f, val_out_dir / cls_name / f.name)
        print(f"Class {cls_name}: train={len(train_files)}, val={len(val_files)}")

    split_and_copy(cat_files, 'cats')
    split_and_copy(dog_files, 'dogs')


# ----------------------------- Extract & prepare ---------------------------
# 1) Extract
print("Extracting zips (this may take a while)...")
EXTRACTED_TRAIN_DIR = WORK_DIR / "extracted_train"
EXTRACTED_TEST_DIR = WORK_DIR / "extracted_test"
EXTRACTED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_TEST_DIR.mkdir(parents=True, exist_ok=True)

# Only extract if empty to avoid re-extracting in reruns
if not any(EXTRACTED_TRAIN_DIR.iterdir()):
    extract_zip(TRAIN_ZIP, EXTRACTED_TRAIN_DIR)
else:
    print("Train already extracted.")

if not any(EXTRACTED_TEST_DIR.iterdir()):
    extract_zip(TEST_ZIP, EXTRACTED_TEST_DIR)
else:
    print("Test already extracted.")

# 2) Organize training and validation sets
# The train.zip in this competition contains a folder named 'train' with images inside.
# Sometimes extraction produces a top-level folder 'train' or directly images. We find the folder with images.

def find_image_dir(root):
    root = Path(root)
    # Prefer direct jpg files at root
    jpgs = list(root.glob('*.jpg'))
    if jpgs:
        return root
    # Otherwise search immediate children
    for child in root.iterdir():
        if child.is_dir():
            jpgs = list(child.glob('*.jpg'))
            if jpgs:
                return child
    raise FileNotFoundError(f"No jpg images found under {root}")

EX_TRAIN_IMG_DIR = find_image_dir(EXTRACTED_TRAIN_DIR)
EX_TEST_IMG_DIR = find_image_dir(EXTRACTED_TEST_DIR)
print("Images dir:", EX_TRAIN_IMG_DIR, EX_TEST_IMG_DIR)

# Only organize once (copies) to save time on re-runs
if not any((TRAIN_DIR).iterdir()):
    organize_train_images(EX_TRAIN_IMG_DIR, TRAIN_DIR, VAL_DIR, val_split=0.15)
else:
    print("Train/val already prepared.")

# Copy test images to test dir (keep original filenames)
if not any(TEST_DIR.iterdir()):
    for f in EX_TEST_IMG_DIR.glob('*.jpg'):
        shutil.copy(f, TEST_DIR / f.name)
    print(f"Copied {len(list(TEST_DIR.glob('*.jpg')))} test images")
else:
    print("Test already prepared.")

# ----------------------------- Data generators -----------------------------
print("Preparing data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Optional: compute class weights (in case of imbalance)
from sklearn.utils.class_weight import compute_class_weight
classes = train_gen.classes
class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ----------------------------- Build model --------------------------------
print("Building model...")

def conv_block(x, filters, kernel_size=3, pool=True, dropout_rate=0.2):
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=initializers.HeNormal(seed=SEED),
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_initializer=initializers.HeNormal(seed=SEED),
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if pool:
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, seed=SEED)(x)
    return x

input_shape = IMG_SIZE + (3,)
inputs = layers.Input(shape=input_shape)

x = conv_block(inputs, 32, dropout_rate=0.15)
x = conv_block(x, 64, dropout_rate=0.2)
x = conv_block(x, 128, dropout_rate=0.25)
x = conv_block(x, 256, dropout_rate=0.3)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, kernel_regularizer=regularizers.l2(L2),
                 kernel_initializer=initializers.HeNormal(seed=SEED))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5, seed=SEED)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------- Callbacks ----------------------------------
checkpoint_path = WORK_DIR / 'best_model.h5'
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_loss', save_best_only=True, save_weights_only=False)
]

# ----------------------------- Training -----------------------------------
steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
validation_steps = max(1, val_gen.samples // BATCH_SIZE)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weights
)

# ----------------------------- Plots -------------------------------------
print("Plotting training history...")
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.show()

# ----------------------------- Evaluation --------------------------------
print("Evaluating on validation set...")
# Predict on validation set
val_gen.reset()
preds = model.predict(val_gen, steps=validation_steps, verbose=1)
# Because we used flow_from_directory with shuffle=False, we can get labels in order
y_true = val_gen.classes[:validation_steps * BATCH_SIZE]
# Truncate preds to same length
preds = preds.ravel()[:len(y_true)]

y_pred = (preds >= 0.5).astype(int)

print("Classification report:")
print(classification_report(y_true, y_pred, target_names=['cats','dogs']))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticklabels(['cats','dogs'])
ax.set_yticklabels(['cats','dogs'])
plt.title('Confusion Matrix')
plt.show()

prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
acc = (y_true == y_pred).mean()

print(f"Validation accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

# ----------------------------- Test predictions (optional) -----------------
# The competition test set is unlabeled - below code shows how to predict and save a CSV
# If you only want validation metrics, you can skip this.

make_test_preds = True
if make_test_preds:
    print("Predicting on test set (no labels expected)...")
    test_files = sorted(TEST_DIR.glob('*.jpg'))
    out_preds = []
    batch = []
    names = []
    for i, f in enumerate(test_files):
        img = tf.keras.preprocessing.image.load_img(f, target_size=IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        batch.append(arr)
        names.append(f.name)
        # Predict in batches
        if len(batch) == BATCH_SIZE or i == len(test_files)-1:
            batch_arr = np.stack(batch, axis=0)
            p = model.predict(batch_arr)
            out_preds.extend(p.ravel().tolist())
            batch = []
    # Example of saving predictions
    import pandas as pd
    submission = pd.DataFrame({
        'id': [n.split('.')[0] for n in names],
        'label': out_preds
    })
    submission.to_csv(WORK_DIR / 'test_predictions.csv', index=False)
    print(f"Saved test predictions to {WORK_DIR / 'test_predictions.csv'}")

print("Done. Best model saved to:", checkpoint_path)
print("You can adjust architecture, augmentations, batch size, and regularization to push accuracy higher.")



# ----------------------------- Test Set Evaluation (fast) -----------------------------
print("=== Test Set Evaluation (using organized folders) ===")

test_generator = val_datagen.flow_from_directory(
    test_dir,                # path to your test_organized folder
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Predict labels
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification report
print("\nTest Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Cat','Dog']))

# Confusion matrix
test_conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Cat','Dog'], yticklabels=['Cat','Dog'])
plt.title('Test Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


