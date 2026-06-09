import os, re, zipfile, shutil, pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
TRAIN_ZIP = '/kaggle/input/dogs-vs-cats/train.zip'
TEST_ZIP  = '/kaggle/input/dogs-vs-cats/test1.zip'
WORK_DIR  = '/kaggle/working/dogs-vs-cats'
BATCH_SIZE = 32
IMG_SIZE = (224,224)
EPOCHS_HEAD = 8
EPOCHS_FINE = 6
AUTOTUNE = tf.data.AUTOTUNE
MODEL_DIR = '/kaggle/working/model'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- UNZIP safely ---
os.makedirs(WORK_DIR, exist_ok=True)
def safe_unzip(zip_path, extract_to):
    print('Unzipping', zip_path, '->', extract_to)
    with zipfile.ZipFile(zip_path,'r') as z:
        z.extractall(extract_to)
if os.path.exists(TRAIN_ZIP):
    safe_unzip(TRAIN_ZIP, WORK_DIR)
else:
    print('Train zip not found at', TRAIN_ZIP)
if os.path.exists(TEST_ZIP):
    safe_unzip(TEST_ZIP, WORK_DIR)
else:
    print('Test zip not found at', TEST_ZIP)

# --- Locate train images (some zips contain a 'train' folder or images directly) ---
# After unzipping Kaggle common structure: WORK_DIR/train/.jpg  and WORK_DIR/test1/.jpg
possible_train_dirs = [
    os.path.join(WORK_DIR, 'train'),
    WORK_DIR
]
train_found = None
for d in possible_train_dirs:
    if os.path.isdir(d) and any(p.suffix.lower() == '.jpg' for p in pathlib.Path(d).glob('*.jpg')):
        train_found = d
        break

if train_found is None:
    # maybe images are in WORK_DIR/train/*.jpg in nested folder
    for root, dirs, files in os.walk(WORK_DIR):
        if any(f.lower().endswith('.jpg') for f in files):
            train_found = root
            break

if train_found is None:
    raise FileNotFoundError('Could not find train images after unzipping. Looked in: ' + str(possible_train_dirs))
print('Train images found in:', train_found)

# --- Prepare structured folder: data/train/cat and data/train/dog ---
data_dir = '/kaggle/working/data'
train_dir = os.path.join(data_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
cat_dir = os.path.join(train_dir, 'cat')
dog_dir = os.path.join(train_dir, 'dog')
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

# Move (or copy) images into class folders based on filename prefix (cat. / dog.)
moved = 0
for p in pathlib.Path(train_found).glob('*.jpg'):
    name = p.name.lower()
    if name.startswith('cat'):
        target = os.path.join(cat_dir, p.name)
    elif name.startswith('dog'):
        target = os.path.join(dog_dir, p.name)
    else:
        # fallback: try to parse 'cat' or 'dog' anywhere in filename
        if 'cat' in name:
            target = os.path.join(cat_dir, p.name)
        elif 'dog' in name:
            target = os.path.join(dog_dir, p.name)
        else:
            continue
    # Use copy to avoid modifying input location
    shutil.copy(str(p), target)
    moved += 1
print('Copied', moved, 'train images into', cat_dir, 'and', dog_dir)


# --- Create training and validation datasets using image_dataset_from_directory ---
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Prefetch
train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# --- Data augmentation layer (applied only in training) ---
data_augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.06),
])

# --- Build model with EfficientNetB0 backbone ---
base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(*IMG_SIZE,3), weights='imagenet')
base_model.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE,3))
x = data_augment(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)  # use correct preprocessing
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()




# --- Callbacks ---
# Note: best_model.keras is correct here
callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.keras'), 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
]

# --- Train only the head first ---
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS_HEAD, callbacks=callbacks)

# --- Fine-tune: unfreeze top layers of base_model ---
base_model.trainable = True

# Freeze most layers, fine-tune last ~50 layers
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a low learning rate for fine-tuning
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train (Fine-tune)
history_fine = model.fit(train_dataset, 
                         validation_data=val_dataset, 
                         epochs=EPOCHS_HEAD + EPOCHS_FINE,
                         initial_epoch=history.epoch[-1] + 1,
                         callbacks=callbacks)

# --- FIX 1: Add .keras extension here ---
model.save(os.path.join(MODEL_DIR, 'final_model.keras'))

# --- Prepare test predictions ---
test_folder = None
# Check common Kaggle paths
possible_paths = [
    os.path.join(WORK_DIR, 'test1'), 
    os.path.join(WORK_DIR, 'test'), 
    os.path.join(WORK_DIR, 'test1', 'test1')
]

for candidate in possible_paths:
    if os.path.isdir(candidate):
        test_folder = candidate
        break

if test_folder is None:
    # Fallback search
    for root, dirs, files in os.walk(WORK_DIR):
        if any(f.lower().endswith('.jpg') for f in files):
            if 'train' not in root.lower():
                test_folder = root
                break

if test_folder:
    # --- FIX 2: Numeric Sorting ---
    # Standard sorted() treats '10.jpg' as coming before '2.jpg'. 
    # This lambda function tries to sort by the number inside the filename.
    paths = pathlib.Path(test_folder).glob('*.jpg')
    
    def try_int_sort(p):
        # Extracts numbers from filename for sorting (e.g. 10.jpg vs 2.jpg)
        parts = re.split(r'(\d+)', str(p))
        return [int(part) if part.isdigit() else part for part in parts]

    test_images = sorted([str(p) for p in paths], key=try_int_sort)
    
    print('Found', len(test_images), 'test images in', test_folder)

    # Build dataset for prediction
    def prep(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        # Ensure efficientnet preprocess is what you want (it scales differently than ResNet)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_images)\
        .map(prep, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.AUTOTUNE)

    preds = model.predict(test_ds, verbose=1)

    # Generate Submission
    filenames = [pathlib.Path(p).name for p in test_images]
    df = pd.DataFrame({
        'id': [os.path.splitext(f)[0] for f in filenames],
        'label': [float(p[0]) for p in preds]  # Probability of class 1
    })

    out_path = '/kaggle/working/submission.csv'
    df.to_csv(out_path, index=False)
    print('Wrote submission to', out_path)

else:
    print('No test folder found; skip prediction.')

print('Done')


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- BAGIAN 1: Visualisasi Grafik Training (Loss & Accuracy) ---
# Menggabungkan history dari tahap "Head Training" dan "Fine Tuning"
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 6))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([EPOCHS_HEAD-1, EPOCHS_HEAD-1], plt.ylim(), label='Start Fine Tuning', linestyle='--')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([EPOCHS_HEAD-1, EPOCHS_HEAD-1], plt.ylim(), label='Start Fine Tuning', linestyle='--')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- BAGIAN 2: Evaluasi Detail pada Data Validasi ---
print("Sedang memproses prediksi pada data validasi...")

# Kita perlu mengambil label asli (y_true) dan prediksi (y_pred) dari dataset
y_true = []
y_pred_probs = []

# Loop dataset validasi (batch demi batch)
for images, labels in val_dataset:
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs).flatten() # Ubah ke 1D array
y_pred_binary = (y_pred_probs > 0.5).astype(int) # Ambang batas 0.5

# 1. Tampilkan Classification Report (Precision, Recall, F1-Score)
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_binary, target_names=['Class 0', 'Class 1']))

# 2. Tampilkan Confusion Matrix
cm = confusion_matrix(y_true, y_pred_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# --- BAGIAN 3: Visualisasi Hasil Prediksi (Gambar) ---
# Menampilkan beberapa gambar sampel beserta prediksinya
def visualize_predictions(dataset, num_images=9):
    plt.figure(figsize=(10, 10))
    # Ambil 1 batch
    for images, labels in dataset.take(1):
        preds = model.predict(images, verbose=0)
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            img = images[i].numpy().astype("uint8") # Pastikan tipe data gambar benar untuk ditampilkan
            # Jika preprocessing menggunakan efisiennet, gambar mungkin perlu di-rescale balik agar warnanya normal
            # Jika warnanya aneh, coba hilangkan komentar baris bawah:
            # img = (img + 1) * 127.5 
            
            plt.imshow(img)
            
            prob = preds[i][0]
            predicted_label = 1 if prob > 0.5 else 0
            actual_label = int(labels[i])
            
            color = 'green' if predicted_label == actual_label else 'red'
            
            plt.title(f"True: {actual_label} | Pred: {predicted_label}\nConf: {prob:.2f}", color=color)
            plt.axis("off")
    plt.show()

print("\n--- Contoh Hasil Prediksi Visual ---")
visualize_predictions(val_dataset)




