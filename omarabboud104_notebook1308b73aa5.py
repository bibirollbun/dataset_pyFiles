# Full end-to-end script (Kaggle-ready)
# - 5 preprocessing pipelines
# - For each pipeline: load 1000 images/class, train/validate split (80/20)
# - Train Dense NN (BatchNorm + Dropout)
# - Show validation confusion matrix + training curves
# - Draw prediction distribution on 3000 random test images + show 20 sample test predictions
#
# Paths are set for the Kaggle "State Farm Distracted Driver Detection" dataset.

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------------------------
# Config (Kaggle dataset paths)
# ------------------------------
TRAIN_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TEST_DIR  = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"

IMG_SIZE = 64                # resize: 64x64
TRAIN_PER_CLASS = 1000       # take up to 1000 images per class from train/
MAX_TEST_IMAGES = 3000       # number of test images to sample for predictions
EPOCHS = 25
BATCH_SIZE = 64
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------
# Class names (folder names c0..c9)
# ------------------------------
# If your train folders are named differently, update this accordingly.
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)
print("Detected classes:", class_names)

# ------------------------------
# Preprocessing functions (5 pipelines)
# ------------------------------
def preprocess_standard(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return img.astype(np.float32) / 255.0

def preprocess_lighting(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gamma = 1.15
    img = np.power(img/255.0, 1.0/gamma)
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def preprocess_noise(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.medianBlur(img, 3)
    return img.astype(np.float32) / 255.0

def preprocess_feature(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(img.astype(np.float32), 0.75, edges_rgb.astype(np.float32), 0.25, 0.0)
    return np.clip(blended / 255.0, 0.0, 1.0).astype(np.float32)

def preprocess_aug(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    alpha = 1.0 + (np.random.rand() - 0.5) * 0.3
    beta = int((np.random.rand() - 0.5) * 50)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img.astype(np.float32) / 255.0

PIPELINES = {
    "Standard": preprocess_standard,
    "Lighting": preprocess_lighting,
    "NoiseReduction": preprocess_noise,
    "FeatureEnhancement": preprocess_feature,
    "Augmentation": preprocess_aug
}

# ------------------------------
# Dense model builder (flatten input)
# ------------------------------
def build_dense(input_dim, num_classes, lr=1e-4):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------
# Utilities: load exactly TRAIN_PER_CLASS per class
# ------------------------------
def load_train_sampled(train_dir, preprocess_fn, per_class=TRAIN_PER_CLASS):
    X, y = [], []
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls)
        files = sorted(os.listdir(cls_dir))[:per_class]
        for fname in files:
            p = os.path.join(cls_dir, fname)
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(preprocess_fn(img))
            y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, classes

# ------------------------------
# Utilities: load N random test images (no labels)
# ------------------------------
def load_test_sample(test_dir, preprocess_fn, max_images=MAX_TEST_IMAGES):
    all_files = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
    # sample randomly but reproducibly:
    rng = random.Random(RANDOM_SEED)
    sample_files = rng.sample(all_files, min(len(all_files), max_images))
    X_test_raw = []        # raw resized images for display (RGB uint8)
    X_test_input = []      # preprocessed normalized arrays
    for fname in tqdm(sample_files, desc="Loading test images"):
        p = os.path.join(test_dir, fname)
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        X_test_raw.append(img_resized)                    # uint8 RGB for display
        X_test_input.append(preprocess_fn(img_resized))   # normalized float32
    X_test_raw = np.array(X_test_raw, dtype=np.uint8)
    X_test_input = np.array(X_test_input, dtype=np.float32)
    return X_test_raw, X_test_input

# ------------------------------
# Plot helpers
# ------------------------------
def plot_confusion_matrix_labels(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(8,8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca())
    plt.title(title)
    plt.show()

def plot_training_curves(history, pipeline_name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{pipeline_name} Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{pipeline_name} Loss"); plt.legend()
    plt.show()

def plot_test_distribution(preds, classes, pipeline_name):
    plt.figure(figsize=(9,4))
    sns.countplot(x=preds, order=range(len(classes)))
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.title(f"Prediction distribution on test (pipeline={pipeline_name})")
    plt.xlabel("Predicted class"); plt.ylabel("Count")
    plt.show()

def show_test_samples(X_raw, preds, classes, pipeline_name, n_samples=20):
    n = min(n_samples, len(X_raw))
    rng = random.Random(RANDOM_SEED)
    idxs = rng.sample(range(len(X_raw)), n)
    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X_raw[idx])
        plt.title(f"P: {classes[preds[idx]]}")
        plt.axis("off")
    plt.suptitle(f"Sample Test Predictions - {pipeline_name}", fontsize=16)
    plt.show()

# ------------------------------
# Main loop: iterate pipelines
# ------------------------------
results_summary = {}

for pipeline_name, preprocess_fn in PIPELINES.items():
    print("\n" + "="*80)
    print(f"PIPELINE: {pipeline_name}")
    print("="*80)

    # 1) Load train (1000 per class) & make train/val split
    X_all, y_all, classes = load_train_sampled(TRAIN_DIR, preprocess_fn, per_class=TRAIN_PER_CLASS)
    if X_all.shape[0] == 0:
        raise RuntimeError(f"No training images found in {TRAIN_DIR}. Check path.")
    print(f"Loaded train images: {X_all.shape}, labels: {np.unique(y_all).size} classes")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
    )
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # 2) Prepare dense inputs (flatten)
    X_tr_input = X_train.reshape(X_train.shape[0], -1)
    X_val_input = X_val.reshape(X_val.shape[0], -1)
    input_dim = X_tr_input.shape[1]
    print("Dense input dimension:", input_dim)

    # 3) Build + train model
    model = build_dense(input_dim=input_dim, num_classes=num_classes, lr=1e-4)
    model.summary()

    cb_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        X_tr_input, y_train,
        validation_data=(X_val_input, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1
    )

    # 4) Validation evaluation: predictions, conf matrix, acc
    y_val_pred = np.argmax(model.predict(X_val_input, verbose=0), axis=1)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"[{pipeline_name}] Validation accuracy = {val_acc:.4f}")

    plot_confusion_matrix_labels(y_val, y_val_pred, classes, title=f"Validation Confusion Matrix - {pipeline_name}")
    plot_training_curves(history, pipeline_name)

    # 5) Test predictions on sampled 3000 images
    print("Loading & predicting on up to", MAX_TEST_IMAGES, "test images (random sample)...")
    X_test_raw, X_test_input = load_test_sample(TEST_DIR, preprocess_fn, max_images=MAX_TEST_IMAGES)
    if X_test_input.shape[0] == 0:
        print("No test images loaded (check TEST_DIR). Skipping test predictions for this pipeline.")
        results_summary[pipeline_name] = {
            "val_acc": val_acc,
            "num_test": 0,
            "test_preds": None
        }
        continue

    # flatten test input for dense model
    X_test_flat = X_test_input.reshape(X_test_input.shape[0], -1)

    # Predict in batches
    preds_prob = model.predict(X_test_flat, batch_size=128, verbose=0)
    preds = np.argmax(preds_prob, axis=1)

    # 6) Show distribution and sample images
    plot_test_distribution(preds, classes, pipeline_name)
    show_test_samples(X_test_raw, preds, classes, pipeline_name, n_samples=20)

    # Save summary
    results_summary[pipeline_name] = {
        "val_acc": val_acc,
        "num_test": X_test_flat.shape[0],
        "test_preds": preds  # array of predicted class indices for test sample
    }

# ------------------------------
# Final summary: validation accuracies bar chart
# ------------------------------
pipelines_done = list(results_summary.keys())
val_accs = [results_summary[p]["val_acc"] for p in pipelines_done]

plt.figure(figsize=(9,5))
sns.barplot(x=pipelines_done, y=val_accs, palette="mako")
plt.ylim(0,1)
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Preprocessing Pipeline")
plt.xticks(rotation=20)
plt.show()

# Print numeric summary
print("\nNumeric summary:")
for p in pipelines_done:
    info = results_summary[p]
    print(f"{p:20s} | val_acc = {info['val_acc']:.4f} | test_images = {info['num_test']}")



import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------------------------
# Config (Kaggle dataset paths)
# ------------------------------
TRAIN_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TEST_DIR  = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"

IMG_SIZE = 64                # resize: 64x64
TRAIN_PER_CLASS = 1000       # take up to 1000 images per class from train/
MAX_TEST_IMAGES = 3000       # number of test images to sample for predictions
EPOCHS = 25
BATCH_SIZE = 64
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------
# Class names (folder names c0..c9)
# ------------------------------
# If your train folders are named differently, update this accordingly.
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)
print("Detected classes:", class_names)

# Optional: full readable class names
class_full_names = {
    "c0": "safe driving",
    "c1": "texting right",
    "c2": "talking on phone right",
    "c3": "texting left",
    "c4": "talking on phone left",
    "c5": "operating radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair & makeup",
    "c9": "talking to passenger"
}

# ------------------------------
# Preprocessing functions (5 pipelines)
# ------------------------------
def preprocess_standard(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return img.astype(np.float32) / 255.0

def preprocess_lighting(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gamma = 1.15
    img = np.power(img/255.0, 1.0/gamma)
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def preprocess_noise(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.medianBlur(img, 3)
    return img.astype(np.float32) / 255.0

def preprocess_feature(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(img.astype(np.float32), 0.75, edges_rgb.astype(np.float32), 0.25, 0.0)
    return np.clip(blended / 255.0, 0.0, 1.0).astype(np.float32)

def preprocess_aug(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    alpha = 1.0 + (np.random.rand() - 0.5) * 0.3
    beta = int((np.random.rand() - 0.5) * 50)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img.astype(np.float32) / 255.0

PIPELINES = {
    "Standard": preprocess_standard,
    "Lighting": preprocess_lighting,
    "NoiseReduction": preprocess_noise,
    "FeatureEnhancement": preprocess_feature,
    "Augmentation": preprocess_aug
}

# ------------------------------
# Dense model builder (flatten input)
# ------------------------------
def build_dense(input_dim, num_classes, lr=1e-4):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------
# Utilities: load exactly TRAIN_PER_CLASS per class
# ------------------------------
def load_train_sampled(train_dir, preprocess_fn, per_class=TRAIN_PER_CLASS):
    X, y = [], []
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls)
        files = sorted(os.listdir(cls_dir))[:per_class]
        for fname in files:
            p = os.path.join(cls_dir, fname)
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(preprocess_fn(img))
            y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, classes

# ------------------------------
# Utilities: load N random test images (no labels)
# ------------------------------
def load_test_sample(test_dir, preprocess_fn, max_images=MAX_TEST_IMAGES):
    all_files = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
    # sample randomly but reproducibly:
    rng = random.Random(RANDOM_SEED)
    sample_files = rng.sample(all_files, min(len(all_files), max_images))
    X_test_raw = []        # raw resized images for display (RGB uint8)
    X_test_input = []      # preprocessed normalized arrays
    for fname in tqdm(sample_files, desc="Loading test images"):
        p = os.path.join(test_dir, fname)
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        X_test_raw.append(img_resized)                    # uint8 RGB for display
        X_test_input.append(preprocess_fn(img_resized))   # normalized float32
    X_test_raw = np.array(X_test_raw, dtype=np.uint8)
    X_test_input = np.array(X_test_input, dtype=np.float32)
    return X_test_raw, X_test_input

# ------------------------------
# Plot helpers
# ------------------------------
def plot_confusion_matrix_labels(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(8,8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca())
    plt.title(title)
    plt.show()

def plot_training_curves(history, pipeline_name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{pipeline_name} Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{pipeline_name} Loss"); plt.legend()
    plt.show()

def plot_test_distribution(preds, classes, pipeline_name):
    plt.figure(figsize=(9,4))
    sns.countplot(x=preds, order=range(len(classes)))
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.title(f"Prediction distribution on test (pipeline={pipeline_name})")
    plt.xlabel("Predicted class"); plt.ylabel("Count")
    plt.show()

def show_test_samples(X_raw, preds, classes, pipeline_name, n_samples=20):
    n = min(n_samples, len(X_raw))
    rng = random.Random(RANDOM_SEED)
    idxs = rng.sample(range(len(X_raw)), n)
    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X_raw[idx])
        # ====== تعديل العنوان: يكتب الكلاس + الاسم الكامل ======
        class_id = preds[idx]
        class_code = classes[class_id]
        class_name_full = class_full_names.get(class_code, class_code)
        plt.title(f"{class_code} - {class_name_full}", fontsize=9)
        plt.axis("off")
    plt.suptitle(f"Sample Test Predictions - {pipeline_name}", fontsize=16)
    plt.show()

# ------------------------------
# Main loop: iterate pipelines
# ------------------------------
results_summary = {}

for pipeline_name, preprocess_fn in PIPELINES.items():
    print("\n" + "="*80)
    print(f"PIPELINE: {pipeline_name}")
    print("="*80)

    # 1) Load train (1000 per class) & make train/val split
    X_all, y_all, classes = load_train_sampled(TRAIN_DIR, preprocess_fn, per_class=TRAIN_PER_CLASS)
    if X_all.shape[0] == 0:
        raise RuntimeError(f"No training images found in {TRAIN_DIR}. Check path.")
    print(f"Loaded train images: {X_all.shape}, labels: {np.unique(y_all).size} classes")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
    )
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # 2) Prepare dense inputs (flatten)
    X_tr_input = X_train.reshape(X_train.shape[0], -1)
    X_val_input = X_val.reshape(X_val.shape[0], -1)
    input_dim = X_tr_input.shape[1]
    print("Dense input dimension:", input_dim)

    # 3) Build + train model
    model = build_dense(input_dim=input_dim, num_classes=num_classes, lr=1e-4)
    model.summary()

    cb_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        X_tr_input, y_train,
        validation_data=(X_val_input, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1
    )

    # 4) Validation evaluation: predictions, conf matrix, acc
    y_val_pred = np.argmax(model.predict(X_val_input, verbose=0), axis=1)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"[{pipeline_name}] Validation accuracy = {val_acc:.4f}")

    plot_confusion_matrix_labels(y_val, y_val_pred, classes, title=f"Validation Confusion Matrix - {pipeline_name}")
    plot_training_curves(history, pipeline_name)

    # 5) Test predictions on sampled 3000 images
    print("Loading & predicting on up to", MAX_TEST_IMAGES, "test images (random sample)...")
    X_test_raw, X_test_input = load_test_sample(TEST_DIR, preprocess_fn, max_images=MAX_TEST_IMAGES)
    if X_test_input.shape[0] == 0:
        print("No test images loaded (check TEST_DIR). Skipping test predictions for this pipeline.")
        results_summary[pipeline_name] = {
            "val_acc": val_acc,
            "num_test": 0,
            "test_preds": None
        }
        continue

    # flatten test input for dense model
    X_test_flat = X_test_input.reshape(X_test_input.shape[0], -1)

    # Predict in batches
    preds_prob = model.predict(X_test_flat, batch_size=128, verbose=0)
    preds = np.argmax(preds_prob, axis=1)

    # 6) Show distribution and sample images
    plot_test_distribution(preds, classes, pipeline_name)
    show_test_samples(X_test_raw, preds, classes, pipeline_name, n_samples=20)

    # Save summary
    results_summary[pipeline_name] = {
        "val_acc": val_acc,
        "num_test": X_test_flat.shape[0],
        "test_preds": preds  # array of predicted class indices for test sample
    }

# ------------------------------
# Final summary: validation accuracies bar chart
# ------------------------------
pipelines_done = list(results_summary.keys())
val_accs = [results_summary[p]["val_acc"] for p in pipelines_done]

plt.figure(figsize=(9,5))
sns.barplot(x=pipelines_done, y=val_accs, palette="mako")
plt.ylim(0,1)
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Preprocessing Pipeline")
plt.xticks(rotation=20)
plt.show()

# Print numeric summary
print("\nNumeric summary:")
for p in pipelines_done:
    info = results_summary[p]
    print(f"{p:20s} | val_acc = {info['val_acc']:.4f} | test_images = {info['num_test']}")



import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

# ------------------------------
# Config (Kaggle dataset paths)
# ------------------------------
TRAIN_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TEST_DIR  = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"

IMG_SIZE = 64                # resize: 64x64
TRAIN_PER_CLASS = 1500       # take up to 1000 images per class from train/
MAX_TEST_IMAGES = 50000       # number of test images to sample for predictions
EPOCHS = 30                   # lower for Kaggle runtime
BATCH_SIZE = 32
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------
# Class names (folder names c0..c9)
# ------------------------------
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)
print("Detected classes:", class_names)

# ------------------------------
# Preprocessing functions (5 pipelines)
# ------------------------------
def preprocess_standard(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return preprocess_input(img.astype(np.float32))

def preprocess_lighting(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gamma = 1.15
    img = np.power(img/255.0, 1.0/gamma) * 255.0
    return preprocess_input(img.astype(np.float32))

def preprocess_noise(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.medianBlur(img, 3)
    return preprocess_input(img.astype(np.float32))

def preprocess_feature(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(img.astype(np.float32), 0.75, edges_rgb.astype(np.float32), 0.25, 0.0)
    return preprocess_input(np.clip(blended, 0.0, 255.0).astype(np.float32))

def preprocess_aug(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    alpha = 1.0 + (np.random.rand() - 0.5) * 0.3
    beta = int((np.random.rand() - 0.5) * 50)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return preprocess_input(img.astype(np.float32))

PIPELINES = {
    "Standard": preprocess_standard,
    "Lighting": preprocess_lighting,
    "NoiseReduction": preprocess_noise,
    "FeatureEnhancement": preprocess_feature,
    "Augmentation": preprocess_aug
}

# ------------------------------
# Transfer Learning: ResNet50
# ------------------------------
def build_resnet50_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes, lr=1e-4):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------
# Utilities: load exactly TRAIN_PER_CLASS per class
# ------------------------------
def load_train_sampled(train_dir, preprocess_fn, per_class=TRAIN_PER_CLASS):
    X, y = [], []
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls)
        files = sorted(os.listdir(cls_dir))[:per_class]
        for fname in files:
            p = os.path.join(cls_dir, fname)
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(preprocess_fn(img))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), classes

# ------------------------------
# Utilities: load N random test images (no labels)
# ------------------------------
def load_test_sample(test_dir, preprocess_fn, max_images=MAX_TEST_IMAGES):
    all_files = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
    rng = random.Random(RANDOM_SEED)
    sample_files = rng.sample(all_files, min(len(all_files), max_images))
    X_test_raw, X_test_input = [], []
    for fname in tqdm(sample_files, desc="Loading test images"):
        p = os.path.join(test_dir, fname)
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        X_test_raw.append(img_resized)
        X_test_input.append(preprocess_fn(img_resized))
    return np.array(X_test_raw, dtype=np.uint8), np.array(X_test_input, dtype=np.float32)

# ------------------------------
# Plot helpers
# ------------------------------
def plot_confusion_matrix_labels(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(8,8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca())
    plt.title(title)
    plt.show()

def plot_training_curves(history, pipeline_name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{pipeline_name} Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{pipeline_name} Loss"); plt.legend()
    plt.show()

def plot_test_distribution(preds, classes, pipeline_name):
    plt.figure(figsize=(9,4))
    sns.countplot(x=preds, order=range(len(classes)))
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.title(f"Prediction distribution on test (pipeline={pipeline_name})")
    plt.xlabel("Predicted class"); plt.ylabel("Count")
    plt.show()

def show_test_samples(X_raw, preds, classes, pipeline_name, n_samples=20):
    n = min(n_samples, len(X_raw))
    rng = random.Random(RANDOM_SEED)
    idxs = rng.sample(range(len(X_raw)), n)
    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X_raw[idx])
        plt.title(f"{classes[preds[idx]]} ({preds[idx]})")
        plt.axis("off")
    plt.suptitle(f"Sample Test Predictions - {pipeline_name}", fontsize=16)
    plt.show()

# ------------------------------
# Main loop: iterate pipelines
# ------------------------------
results_summary = {}

for pipeline_name, preprocess_fn in PIPELINES.items():
    print("\n" + "="*80)
    print(f"PIPELINE: {pipeline_name}")
    print("="*80)

    X_all, y_all, classes = load_train_sampled(TRAIN_DIR, preprocess_fn, per_class=TRAIN_PER_CLASS)
    if X_all.shape[0] == 0:
        raise RuntimeError(f"No training images found in {TRAIN_DIR}. Check path.")
    print(f"Loaded train images: {X_all.shape}, labels: {np.unique(y_all).size} classes")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
    )
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    model = build_resnet50_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes, lr=1e-4)
    model.summary()

    cb_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1
    )

    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"[{pipeline_name}] Validation accuracy = {val_acc:.4f}")

    plot_confusion_matrix_labels(y_val, y_val_pred, classes, title=f"Validation Confusion Matrix - {pipeline_name}")
    plot_training_curves(history, pipeline_name)

    print("Loading & predicting on up to", MAX_TEST_IMAGES, "test images (random sample)...")
    X_test_raw, X_test_input = load_test_sample(TEST_DIR, preprocess_fn, max_images=MAX_TEST_IMAGES)
    if X_test_input.shape[0] == 0:
        print("No test images loaded (check TEST_DIR). Skipping test predictions.")
        results_summary[pipeline_name] = {"val_acc": val_acc, "num_test": 0, "test_preds": None}
        continue

    preds_prob = model.predict(X_test_input, batch_size=128, verbose=0)
    preds = np.argmax(preds_prob, axis=1)

    plot_test_distribution(preds, classes, pipeline_name)
    show_test_samples(X_test_raw, preds, classes, pipeline_name, n_samples=20)

    results_summary[pipeline_name] = {"val_acc": val_acc, "num_test": X_test_input.shape[0], "test_preds": preds}

# ------------------------------
# Final summary: validation accuracies bar chart
# ------------------------------
pipelines_done = list(results_summary.keys())
val_accs = [results_summary[p]["val_acc"] for p in pipelines_done]

plt.figure(figsize=(9,5))
sns.barplot(x=pipelines_done, y=val_accs, palette="mako")
plt.ylim(0,1)
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Preprocessing Pipeline")
plt.xticks(rotation=20)
plt.show()

print("\nNumeric summary:")
for p in pipelines_done:
    info = results_summary[p]
    print(f"{p:20s} | val_acc = {info['val_acc']:.4f} | test_images = {info['num_test']}")



# ==============================
# Full end-to-end script (Kaggle-ready)
# State Farm Distracted Driver — 5 Preprocessing Pipelines + ResNet50 (frozen)
# ==============================

import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from tensorflow.keras import callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import tensorflow as tf

# ------------------------------
# Config (Kaggle dataset paths)
# ------------------------------
TRAIN_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TEST_DIR  = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"

IMG_SIZE = 224                 # ResNet50 input size
PER_CLASS_LIMIT = 1500         # 1500 images per class from train/
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train/val/test
MAX_TEST_IMAGES = 10000        # unlabeled test sample size
EPOCHS = 10                    # زدها لو عندك وقت على كاجل
BATCH_SIZE = 32
RANDOM_SEED = 42
LR = 1e-4

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------
# Classes (folder names c0..c9) + readable names
# ------------------------------
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)
print("Detected classes:", class_names)

class_full_names = {
    "c0": "safe driving",
    "c1": "texting right",
    "c2": "talking on phone right",
    "c3": "texting left",
    "c4": "talking on phone left",
    "c5": "operating radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair & makeup",
    "c9": "talking to passenger"
}

# ------------------------------
# Preprocessing pipelines (return RGB float32 in [0..255]; preprocess_input handles normalization)
# ------------------------------
def preprocess_standard(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return img.astype(np.float32)

def preprocess_lighting(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gamma = 1.15
    img = np.power(np.clip(img, 0, 255)/255.0, 1.0/gamma) * 255.0
    return np.clip(img, 0, 255).astype(np.float32)

def preprocess_noise(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.medianBlur(img, 3)
    return img.astype(np.float32)

def preprocess_feature(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(img.astype(np.float32), 0.75, edges_rgb.astype(np.float32), 0.25, 0.0)
    return np.clip(blended, 0, 255).astype(np.float32)

def preprocess_aug(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    alpha = 1.0 + (np.random.rand() - 0.5) * 0.3
    beta = int((np.random.rand() - 0.5) * 50)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img.astype(np.float32)

PIPELINES = {
    "Standard": preprocess_standard,
    "Lighting": preprocess_lighting,
    "NoiseReduction": preprocess_noise,
    "FeatureEnhancement": preprocess_feature,
    "Augmentation": preprocess_aug
}

# ------------------------------
# Build ResNet50 (Frozen) classifier
# ------------------------------
def build_resnet50_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes, lr=LR):
    base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------
# Utilities: collect & split paths (per class)
# ------------------------------
def collect_paths_per_class(train_dir, per_class=PER_CLASS_LIMIT, seed=RANDOM_SEED):
    rng = random.Random(seed)
    class_to_paths = {}
    for cls in sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]):
        cls_dir = os.path.join(train_dir, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        files.sort()
        if len(files) > per_class:
            rng.shuffle(files)
            files = files[:per_class]
        class_to_paths[cls] = files
    return class_to_paths

def split_70_15_15(class_to_paths, ratios=SPLIT_RATIOS, seed=RANDOM_SEED):
    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    rng = random.Random(seed)
    for label, cls in enumerate(sorted(class_to_paths.keys())):
        paths = class_to_paths[cls]
        idxs = list(range(len(paths)))
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * ratios[0]))
        n_val   = int(round(n * ratios[1]))
        # ensure total == n
        n_test  = n - n_train - n_val

        tr = idxs[:n_train]
        va = idxs[n_train:n_train+n_val]
        te = idxs[n_train+n_val:]

        for i in tr:
            train_paths.append(paths[i]); train_labels.append(label)
        for i in va:
            val_paths.append(paths[i]);   val_labels.append(label)
        for i in te:
            test_paths.append(paths[i]);  test_labels.append(label)

    return (train_paths, np.array(train_labels, dtype=np.int32),
            val_paths,   np.array(val_labels,   dtype=np.int32),
            test_paths,  np.array(test_labels,  dtype=np.int32))

# ------------------------------
# Generator reading from disk + preprocessing + ResNet50 preprocess_input
# ------------------------------
def make_generator(paths, labels, preprocess_fn, batch_size=BATCH_SIZE, shuffle=True):
    n = len(paths)
    idxs = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = idxs[start:end]
            X_batch = np.zeros((len(batch_idx), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            y_batch = None if labels is None else labels[batch_idx]
            for j, k in enumerate(batch_idx):
                p = paths[k]
                img_bgr = cv2.imread(p)
                if img_bgr is None:
                    # fallback empty if read fails
                    X_batch[j] = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                proc = preprocess_fn(img_rgb)        # float32 RGB [0..255]
                X_batch[j] = preprocess_input(proc)  # ResNet50 preprocess
            yield (X_batch, y_batch) if labels is not None else X_batch

# ------------------------------
# Plot helpers
# ------------------------------
def plot_training_curves(history, pipeline_name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{pipeline_name} Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{pipeline_name} Loss"); plt.legend()
    plt.show()

def plot_confusion_matrix_labels(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(8,8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=plt.gca(), colorbar=False)
    plt.title(title)
    plt.show()

def plot_test_distribution(preds, classes, pipeline_name):
    plt.figure(figsize=(9,4))
    sns.countplot(x=preds, order=range(len(classes)))
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.title(f"Prediction distribution on 10k test (pipeline={pipeline_name})")
    plt.xlabel("Predicted class"); plt.ylabel("Count")
    plt.show()

def show_test_samples(paths, preds, classes, pipeline_name, n_samples=20):
    n = min(n_samples, len(paths))
    rng = random.Random(RANDOM_SEED)
    idxs = rng.sample(range(len(paths)), n)
    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(idxs):
        img_bgr = cv2.imread(paths[idx])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        class_id = preds[idx]
        class_code = classes[class_id]
        class_name_full = class_full_names.get(class_code, class_code)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_resized)
        plt.title(f"{class_code} - {class_name_full}", fontsize=9)
        plt.axis("off")
    plt.suptitle(f"Sample Test Predictions - {pipeline_name}", fontsize=16)
    plt.show()

# ------------------------------
# Train/Eval per pipeline
# ------------------------------
results_summary = {}

for pipeline_name, preprocess_fn in PIPELINES.items():
    print("\n" + "="*100)
    print(f"PIPELINE: {pipeline_name}")
    print("="*100)

    # 1) Collect & split paths 70/15/15
    class_to_paths = collect_paths_per_class(TRAIN_DIR, per_class=PER_CLASS_LIMIT, seed=RANDOM_SEED)
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = split_70_15_15(class_to_paths, ratios=SPLIT_RATIOS, seed=RANDOM_SEED)

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # 2) Build model
    model = build_resnet50_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes, lr=LR)
    model.summary()

    steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
    val_steps       = math.ceil(len(val_paths)   / BATCH_SIZE)
    test_steps      = math.ceil(len(test_paths)  / BATCH_SIZE)

    # 3) Generators
    train_gen = make_generator(train_paths, train_labels, preprocess_fn, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = make_generator(val_paths,   val_labels,   preprocess_fn, batch_size=BATCH_SIZE, shuffle=False)
    test_gen  = make_generator(test_paths,  test_labels,  preprocess_fn, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Callbacks
    cb_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
    ]

    # 5) Train
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1
    )

    # 6) Validation predictions (Confusion Matrix + Val Accuracy)
    val_pred_prob = model.predict(val_gen, steps=val_steps, verbose=0)
    y_val_pred = np.argmax(val_pred_prob, axis=1)
    # trim to exact length (last partial batch)
    y_val_pred = y_val_pred[:len(val_paths)]
    y_val_true = val_labels[:len(val_paths)]

    val_acc = accuracy_score(y_val_true, y_val_pred)
    print(f"[{pipeline_name}] Validation accuracy = {val_acc:.4f}")

    plot_confusion_matrix_labels(y_val_true, y_val_pred, class_names,
                                 title=f"Validation Confusion Matrix - {pipeline_name}")
    plot_training_curves(history, pipeline_name)

    # 7) Evaluate on internal 15% test (optional metrics)
    test_pred_prob = model.predict(test_gen, steps=test_steps, verbose=0)
    y_test_pred = np.argmax(test_pred_prob, axis=1)
    y_test_pred = y_test_pred[:len(test_paths)]
    y_test_true = test_labels[:len(test_paths)]
    test_acc = accuracy_score(y_test_true, y_test_pred)
    print(f"[{pipeline_name}] Internal 15% test accuracy = {test_acc:.4f}")

    # 8) Unlabeled Kaggle test predictions (10,000 random)
    all_test_files = [f for f in os.listdir(TEST_DIR) if os.path.isfile(os.path.join(TEST_DIR, f))]
    rng = random.Random(RANDOM_SEED)
    sample_files = rng.sample(all_test_files, min(len(all_test_files), MAX_TEST_IMAGES))
    test_unlabeled_paths = [os.path.join(TEST_DIR, f) for f in sample_files]

    # Batch predict
    preds_all = []
    batch = []
    for p in tqdm(test_unlabeled_paths, desc=f"Predicting 10k test ({pipeline_name})"):
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        proc = preprocess_fn(img_rgb)
        x = preprocess_input(proc)
        batch.append(x)
        if len(batch) == 128:
            batch_np = np.stack(batch, axis=0)
            prob = model.predict(batch_np, verbose=0)
            preds_all.extend(np.argmax(prob, axis=1))
            batch = []
    if len(batch) > 0:
        batch_np = np.stack(batch, axis=0)
        prob = model.predict(batch_np, verbose=0)
        preds_all.extend(np.argmax(prob, axis=1))

    preds_all = np.array(preds_all, dtype=np.int32)
    # Trim paths to preds length in case of read failures
    test_unlabeled_paths = test_unlabeled_paths[:len(preds_all)]

    # 9) Show prediction distribution + 20 samples with class code + full name
    plot_test_distribution(preds_all, class_names, pipeline_name)
    show_test_samples(test_unlabeled_paths, preds_all, class_names, pipeline_name, n_samples=20)

    # Save summary
    results_summary[pipeline_name] = {
        "val_acc": val_acc,
        "test_acc_internal": test_acc,
        "num_val": len(val_paths),
        "num_test_internal": len(test_paths),
        "num_test_unlabeled_pred": len(test_unlabeled_paths)
    }

# ------------------------------
# Final summary: compare validation accuracies across 5 pipelines
# ------------------------------
pipelines_done = list(results_summary.keys())
val_accs = [results_summary[p]["val_acc"] for p in pipelines_done]

plt.figure(figsize=(9,5))
sns.barplot(x=pipelines_done, y=val_accs)
plt.ylim(0,1)
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Preprocessing Pipeline (ResNet50 Frozen)")
plt.xticks(rotation=20)
plt.show()

print("\nNumeric summary:")
for p in pipelines_done:
    info = results_summary[p]
    print(f"{p:20s} | val_acc = {info['val_acc']:.4f} | internal_test_acc = {info['test_acc_internal']:.4f} | "
          f"val_n = {info['num_val']:5d} | test_n = {info['num_test_internal']:5d} | unlabeled_test_pred_n = {info['num_test_unlabeled_pred']:5d}")


