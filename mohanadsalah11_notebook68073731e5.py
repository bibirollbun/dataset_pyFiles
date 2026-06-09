# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === CONFIGURATION ===
INPUT_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TARGET_SIZE = (224, 224)
SCALE_TYPE = '[0,1]'  # '[0,1]' or '[-1,1]'
NUM_CLASSES = 10
SHOW_TEST_IMAGES = True
MAX_IMAGES = 10000  # limit for faster load

# === LABELS ===
LABELS = {
    'c0': "safe_driving", 'c1': "texting_right", 'c2': "talking_on_the_phone_right",
    'c3': "texting_left", 'c4': "talking_on_the_phone_left", 'c5': "operating_the_radio",
    'c6': "drinking", 'c7': "reaching_behind", 'c8': "hair_and_makeup", 'c9': "talking_to_passenger"
}
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# === LIGHTING ENHANCEMENT WITH PROBABILITY ===
def enhance_lighting(img, prob=0.5):
    """Randomly apply CLAHE, gamma correction, and brightness/contrast normalization."""
    if random.random() > prob:
        return img  # Skip enhancement

    # CLAHE in LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Dynamic clipLimit based on image stats
    brightness = np.mean(l)
    contrast = np.std(l)
    if brightness < 90 and contrast < 40:
        clip_limit = 3.0
    elif contrast > 60:
        clip_limit = 1.2
    else:
        clip_limit = 2.0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Smart gamma correction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    gamma = np.interp(mean_brightness, [50, 200], [0.6, 1.6])
    img = np.power(img / 255.0, gamma)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Brightness/contrast normalization
    target_mean, target_std = 128, 64
    mean, std = cv2.meanStdDev(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    mean, std = mean[0][0], std[0][0]
    if abs(mean - target_mean) > 10 or abs(std - target_std) > 5:
        alpha = target_std / (std + 1e-6)
        beta = target_mean - mean * alpha
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

# === PREPROCESS IMAGE ===
def preprocess_image(img_path, training=True):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if training:
        img = enhance_lighting(img, prob=0.5)  # Apply only for training images
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32)
    if SCALE_TYPE == '[0,1]':
        img /= 255.0
    elif SCALE_TYPE == '[-1,1]':
        img = (img / 127.5) - 1.0
    return img

# === LOAD DATASET ===
def load_dataset(input_dir, max_images=None, training=True):
    X, y = [], []
    print(f"ğŸ“¦ Loading {'training' if training else 'test'} images...")
    count = 0
    for label in tqdm(os.listdir(input_dir), desc="Classes"):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if max_images and count >= max_images:
                break
            img_path = os.path.join(label_dir, fname)
            img = preprocess_image(img_path, training=training)
            if img is not None:
                X.append(img)
                y.append(LABEL_TO_IDX[label])
                count += 1
    return np.array(X), np.array(y)

# === VISUALIZE ENHANCEMENT ===
def visualize_lighting_effect():
    sample_dir = os.path.join(INPUT_DIR, 'c0')  # one class folder
    sample_files = os.listdir(sample_dir)[:5]   # first 5 images

    plt.figure(figsize=(10, 6))
    for i, fname in enumerate(sample_files):
        img_path = os.path.join(sample_dir, fname)
        original = cv2.imread(img_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        enhanced = enhance_lighting(original.copy(), prob=1.0)  # force enhancement

        # Original
        plt.subplot(2, len(sample_files), i+1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis('off')

        # Enhanced
        plt.subplot(2, len(sample_files), i+1+len(sample_files))
        plt.imshow(enhanced)
        plt.title("Enhanced")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Preview enhancement before training
visualize_lighting_effect()

# === LOAD DATA ===
X, y = load_dataset(INPUT_DIR, max_images=MAX_IMAGES, training=True)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

# === DATA GENERATORS ===
train_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
).flow(X_train, y_train, batch_size=32, shuffle=True)

val_gen = ImageDataGenerator().flow(X_test, y_test, batch_size=32, shuffle=False)

# === MODEL ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === CALLBACKS ===
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

# === TRAIN ===
print("ğŸš€ Training...")
model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[reduce_lr])

# === EVALUATE ===
print("\nğŸ“Š Evaluating on test set...")
loss, acc = model.evaluate(val_gen)
print(f"âœ… Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

# === SHOW PREDICTIONS ===
if SHOW_TEST_IMAGES:
    print("\nğŸ–¼ï¸� Showing test predictions...")
    X_vis = X_test[:10]
    y_true = np.argmax(y_test[:10], axis=1)
    y_pred = np.argmax(model.predict(X_vis), axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        axes[i//5, i%5].imshow(X_vis[i])
        true_label = LABELS[IDX_TO_LABEL[y_true[i]]]
        pred_label = LABELS[IDX_TO_LABEL[y_pred[i]]]
        axes[i//5, i%5].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    plt.show()



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === CONFIGURATION ===
INPUT_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TARGET_SIZE = (224, 224)
SCALE_TYPE = '[0,1]'  # '[0,1]' or '[-1,1]'
NUM_CLASSES = 10
SHOW_TEST_IMAGES = True
MAX_IMAGES = 2000  # reduce for faster run

# === LABELS ===
LABELS = {
    'c0': "safe_driving", 'c1': "texting_right", 'c2': "talking_on_the_phone_right",
    'c3': "texting_left", 'c4': "talking_on_the_phone_left", 'c5': "operating_the_radio",
    'c6': "drinking", 'c7': "reaching_behind", 'c8': "hair_and_makeup", 'c9': "talking_to_passenger"
}
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# === INDIVIDUAL STEPS FOR VISUALIZATION ===
def apply_CLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    brightness = np.mean(l)
    contrast = np.std(l)
    if brightness < 90 and contrast < 40:
        clip_limit = 3.0
    elif contrast > 60:
        clip_limit = 1.2
    else:
        clip_limit = 2.0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_gamma(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    gamma = np.interp(mean_brightness, [50, 200], [0.6, 1.6])
    img = np.power(img / 255.0, gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def apply_brightness_contrast(img):
    target_mean, target_std = 128, 64
    mean, std = cv2.meanStdDev(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    mean, std = mean[0][0], std[0][0]
    if abs(mean - target_mean) > 10 or abs(std - target_std) > 5:
        alpha = target_std / (std + 1e-6)
        beta = target_mean - mean * alpha
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# === PROBABILISTIC ENHANCEMENT FOR TRAINING ===
def enhance_lighting(img, prob=0.5):
    if random.random() > prob:
        return img
    img = apply_CLAHE(img)
    img = apply_gamma(img)
    img = apply_brightness_contrast(img)
    return img

# === PREPROCESSING ===
def preprocess_image(img_path, training=True):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if training:
        img = enhance_lighting(img, prob=0.5)
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32)
    if SCALE_TYPE == '[0,1]':
        img /= 255.0
    elif SCALE_TYPE == '[-1,1]':
        img = (img / 127.5) - 1.0
    return img

# === DATA LOADING ===
def load_dataset(input_dir, max_images=None, training=True):
    X, y = [], []
    print(f"ğŸ“¦ Loading {'training' if training else 'test'} images...")
    count = 0
    for label in tqdm(os.listdir(input_dir), desc="Classes"):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if max_images and count >= max_images:
                break
            img_path = os.path.join(label_dir, fname)
            img = preprocess_image(img_path, training=training)
            if img is not None:
                X.append(img)
                y.append(LABEL_TO_IDX[label])
                count += 1
    return np.array(X), np.array(y)

# === VISUALIZE STEPS ===
def visualize_lighting_steps():
    sample_dir = os.path.join(INPUT_DIR, "c0")
    sample_img_path = os.path.join(sample_dir, os.listdir(sample_dir)[0])
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clahe_img = apply_CLAHE(img)
    clahe_gamma_img = apply_gamma(clahe_img)
    final_img = apply_brightness_contrast(clahe_gamma_img)

    titles = [
        "Original",
        "CLAHE only",
        "CLAHE + Gamma",
        "CLAHE + Gamma + Bright/Contrast"
    ]
    images = [img, clahe_img, clahe_gamma_img, final_img]

    plt.figure(figsize=(15, 5))
    for i, (title, im) in enumerate(zip(titles, images)):
        plt.subplot(1, 4, i+1)
        plt.imshow(im)
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# === RUN VISUALIZATION ===
visualize_lighting_steps()

# === LOAD DATA ===
X, y = load_dataset(INPUT_DIR, max_images=MAX_IMAGES, training=True)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

# === DATA GENERATORS ===
train_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
).flow(X_train, y_train, batch_size=32, shuffle=True)

val_gen = ImageDataGenerator().flow(X_test, y_test, batch_size=32, shuffle=False)

# === MODEL ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === CALLBACKS ===
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

# === TRAIN ===
print("ğŸš€ Training...")
model.fit(train_gen, epochs=5, validation_data=val_gen, callbacks=[reduce_lr])

# === EVALUATE ===
print("\nğŸ“Š Evaluating on test set...")
loss, acc = model.evaluate(val_gen)
print(f"âœ… Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

# === SHOW PREDICTIONS ===
if SHOW_TEST_IMAGES:
    print("\nğŸ–¼ï¸� Showing test predictions...")
    X_vis = X_test[:10]
    y_true = np.argmax(y_test[:10], axis=1)
    y_pred = np.argmax(model.predict(X_vis), axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        axes[i//5, i%5].imshow(X_vis[i])
        true_label = LABELS[IDX_TO_LABEL[y_true[i]]]
        pred_label = LABELS[IDX_TO_LABEL[y_pred[i]]]
        axes[i//5, i%5].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    plt.show()



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# === CONFIGURATION ===
INPUT_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TARGET_SIZE = (224, 224)
SCALE_TYPE = '[0,1]'  # or '[-1,1]'
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# === LABELS ===
LABELS = {
    'c0': "safe_driving",
    'c1': "texting_right",
    'c2': "talking_on_the_phone_right",
    'c3': "texting_left",
    'c4': "talking_on_the_phone_left",
    'c5': "operating_the_radio",
    'c6': "drinking",
    'c7': "reaching_behind",
    'c8': "hair_and_makeup",
    'c9': "talking_to_passenger"
}

# === 1. STANDARD IMAGE PROCESSING ===
def resize_and_scale(img, target_size=TARGET_SIZE, scale_type=SCALE_TYPE):
    """
    Resize image with appropriate interpolation and scale pixel values.
    """
    h, w = img.shape[:2]
    if h > target_size[0] or w > target_size[1]:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32)

    if scale_type == '[0,1]':
        img /= 255.0
    elif scale_type == '[-1,1]':
        img = (img / 127.5) - 1.0

    return img

# === 2. LIGHTING CONDITION ENHANCEMENT ===
def adaptive_CLAHE(img):
    """Apply adaptive CLAHE to L channel of LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    brightness = np.mean(l)
    contrast = np.std(l)

    if brightness < 90 and contrast < 40:
        clip_limit = 3.0
    elif contrast > 60:
        clip_limit = 1.2
    else:
        clip_limit = 2.0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def adaptive_gamma(img):
    """Adjust gamma based on mean brightness."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    gamma = np.interp(mean_brightness, [50, 200], [0.6, 1.6])
    img = np.power(img / 255.0, gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def adaptive_brightness_contrast(img):
    """Normalize brightness and contrast."""
    target_mean, target_std = 128, 64
    mean, std = cv2.meanStdDev(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    mean, std = mean[0][0], std[0][0]

    alpha = target_std / (std + 1e-6)
    beta = target_mean - mean * alpha

    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# === 3. FULL PIPELINE FUNCTION ===
def preprocess_pipeline(img, apply_resize=True, apply_scale=True):
    """
    Apply full preprocessing: resize, scale, CLAHE, gamma correction, brightness/contrast normalization.
    """
    if apply_resize or apply_scale:
        img = resize_and_scale(img)

    clahe_img = adaptive_CLAHE(img.astype(np.uint8))
    clahe_gamma_img = adaptive_gamma(clahe_img)
    final_img = adaptive_brightness_contrast(clahe_gamma_img)

    return clahe_img, clahe_gamma_img, final_img

# === 4. VISUALIZATION ===
def visualize_pipeline_all_classes():
    sample_images = []

    for cls in LABELS.keys():
        sample_dir = os.path.join(INPUT_DIR, cls)
        all_files = os.listdir(sample_dir)
        if not all_files:
            continue

        fname = random.choice(all_files)
        img_path = os.path.join(sample_dir, fname)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        clahe_img, clahe_gamma_img, final_img = preprocess_pipeline(img, apply_resize=False, apply_scale=False)

        sample_images.append((cls, img, clahe_img, clahe_gamma_img, final_img))

    titles = ["Original", "CLAHE only", "CLAHE + Gamma", "CLAHE + Gamma + Bright/Contrast"]
    fig, axes = plt.subplots(len(sample_images), 4, figsize=(12, len(sample_images) * 3))

    for row_idx, (cls, orig, clahe, gamma, final) in enumerate(sample_images):
        for col_idx, (title, im) in enumerate(zip(titles, [orig, clahe, gamma, final])):
            axes[row_idx, col_idx].imshow(im)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(title, fontsize=9)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(LABELS[cls], fontsize=8)
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()

# === RUN VISUALIZATION ===
visualize_pipeline_all_classes()



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# === CONFIGURATION ===
INPUT_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TARGET_SIZE = (224, 224)
SCALE_TYPE = '[0,1]'  # or '[-1,1]'
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# === LABELS ===
LABELS = {
    'c0': "safe_driving",
    'c1': "texting_right",
    'c2': "talking_on_the_phone_right",
    'c3': "texting_left",
    'c4': "talking_on_the_phone_left",
    'c5': "operating_the_radio",
    'c6': "drinking",
    'c7': "reaching_behind",
    'c8': "hair_and_makeup",
    'c9': "talking_to_passenger"
}

class ImagePreprocessor:
    """
    A class to handle a full image preprocessing pipeline for computer vision tasks.
    """
    def __init__(self, target_size=TARGET_SIZE, scale_type=SCALE_TYPE):
        self.target_size = target_size
        self.scale_type = scale_type

    def resize_and_scale(self, img):
        """
        Resize image with appropriate interpolation and scale pixel values.
        """
        h, w = img.shape[:2]
        if h > self.target_size[0] or w > self.target_size[1]:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)

        img = img.astype(np.float32)

        if self.scale_type == '[0,1]':
            img /= 255.0
        elif self.scale_type == '[-1,1]':
            img = (img / 127.5) - 1.0

        return img

    def adaptive_CLAHE(self, img):
        """Apply adaptive CLAHE to L channel of LAB space."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        brightness = np.mean(l)
        contrast = np.std(l)

        if brightness < 90 and contrast < 40:
            clip_limit = 3.0
        elif contrast > 60:
            clip_limit = 1.2
        else:
            clip_limit = 2.0

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def adaptive_gamma(self, img):
        """Adjust gamma based on mean brightness."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        gamma = np.interp(mean_brightness, [50, 200], [0.6, 1.6])
        img = np.power(img / 255.0, gamma)
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def adaptive_brightness_contrast(self, img):
        """Normalize brightness and contrast."""
        target_mean, target_std = 128, 64
        mean, std = cv2.meanStdDev(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        mean, std = mean[0][0], std[0][0]

        alpha = target_std / (std + 1e-6)
        beta = target_mean - mean * alpha

        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    def apply_pipeline(self, img, apply_resize=True, apply_scale=True):
        """
        Apply the full preprocessing pipeline to a single image.
        """
        if apply_resize or apply_scale:
            img = self.resize_and_scale(img)

        # The lighting enhancement functions should be applied to images with uint8 pixel values
        clahe_img = self.adaptive_CLAHE(img.astype(np.uint8))
        clahe_gamma_img = self.adaptive_gamma(clahe_img)
        final_img = self.adaptive_brightness_contrast(clahe_gamma_img)

        return clahe_img, clahe_gamma_img, final_img

# === 4. VISUALIZATION ===
def visualize_pipeline_all_classes():
    preprocessor = ImagePreprocessor()
    sample_images = []

    for cls in LABELS.keys():
        sample_dir = os.path.join(INPUT_DIR, cls)
        all_files = os.listdir(sample_dir)
        if not all_files:
            continue

        fname = random.choice(all_files)
        img_path = os.path.join(sample_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # The visualization pipeline will now use the new class method
        clahe_img, clahe_gamma_img, final_img = preprocessor.apply_pipeline(
            img, apply_resize=False, apply_scale=False
        )
        sample_images.append((cls, img, clahe_img, clahe_gamma_img, final_img))

    titles = ["Original", "CLAHE only", "CLAHE + Gamma", "CLAHE + Gamma + Bright/Contrast"]
    num_classes = len(sample_images)
    if num_classes == 0:
        print("No images found for visualization.")
        return

    fig, axes = plt.subplots(num_classes, 4, figsize=(12, num_classes * 3))
    
    # Handle the case where there's only one row (single class)
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (cls, orig, clahe, gamma, final) in enumerate(sample_images):
        for col_idx, (title, im) in enumerate(zip(titles, [orig, clahe, gamma, final])):
            axes[row_idx, col_idx].imshow(im)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(title, fontsize=9)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(LABELS[cls], fontsize=8)
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()

# === RUN VISUALIZATION ===
if __name__ == "__main__":
    visualize_pipeline_all_classes()

