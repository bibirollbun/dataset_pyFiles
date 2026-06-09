# ============================================================
# Grand X-Ray Slam Division A — Train, Evaluate, and Predict
# ============================================================
# Produces: /kaggle/working/submission.csv
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_fscore_support, confusion_matrix
)
import itertools

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "/kaggle/input/grand-xray-slam-division-a"
TRAIN_DIR = os.path.join(DATA_DIR, "train1")
TEST_DIR  = os.path.join(DATA_DIR, "test1")
TRAIN_CSV = os.path.join(DATA_DIR, "train1.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission_1.csv")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

IMG_SIZE = (96, 96)
BATCH_SIZE = 128
EPOCHS = 8
RANDOM_STATE = 42
THRESHOLD = 0.5

# -----------------------------
# Labels
# -----------------------------
targets = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema',
    'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
    'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
    'Pneumothorax','Support Devices'
]

# -----------------------------
# Load dataset
# -----------------------------
print("Loading training CSV...")
df = pd.read_csv(TRAIN_CSV)
df['path'] = df['Image_name'].apply(lambda x: os.path.join(TRAIN_DIR, x))
labels = df[targets].values.astype(np.int32)

print(f"Training set shape: {df.shape}")
missing = (~df['path'].apply(os.path.exists)).sum()
print(f"Missing image files: {missing}")

# Display dataset summary
print("\n=== Dataset Summary ===")
print(f"Total images: {len(df)}")
class_counts = labels.sum(axis=0)
for i, t in enumerate(targets):
    print(f"{t:30s}: {class_counts[i]:6d} positives ({class_counts[i]/len(df):.2%})")
print("========================\n")

# -----------------------------
# Train / Validation Split
# -----------------------------
stratify_col = (labels.sum(axis=1) > 0).astype(int)
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.1, stratify=stratify_col, random_state=RANDOM_STATE
)
train_paths = df.loc[train_idx, 'path'].values
val_paths   = df.loc[val_idx, 'path'].values
y_train = labels[train_idx]
y_val   = labels[val_idx]

print(f"Train size: {len(train_paths)}, Val size: {len(val_paths)}")

# -----------------------------
# TF Datasets
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32)/255.0, label

def decode_test_img(path, name):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32)/255.0, name

def make_train_ds(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_img, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(4096, seed=RANDOM_STATE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_train_ds(train_paths, y_train, shuffle=True)
val_ds   = make_train_ds(val_paths, y_val)

# -----------------------------
# Mixed precision
# -----------------------------
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled.")
except Exception as e:
    print("Could not enable mixed precision:", e)

# -----------------------------
# CNN Model
# -----------------------------
from tensorflow.keras import layers, models

def build_model():
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(targets), activation='sigmoid', dtype='float32')
    ])
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=len(targets))
    ]
)
model.summary()

# -----------------------------
# Train
# -----------------------------
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True, verbose=1)
]

print("\nTraining model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# -----------------------------
# Evaluation: Metrics
# -----------------------------
print("\nEvaluating on validation set...")
y_true, y_pred = [], []
for x, y in val_ds:
    preds = model.predict(x, verbose=0)
    y_true.append(y.numpy())
    y_pred.append(preds)
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

print(f"Validation set predictions shape: {y_pred.shape}")

# ROC AUC
print("\nPer-class ROC AUC:")
aucs = []
for i, t in enumerate(targets):
    try:
        score = roc_auc_score(y_true[:,i], y_pred[:,i])
    except ValueError:
        score = np.nan
    aucs.append(score)
    print(f"  {t:30s}: {score:.4f}")
print(f"\nMean AUC: {np.nanmean(aucs):.4f}")

# Threshold predictions for F1/Precision/Recall
y_pred_bin = (y_pred >= THRESHOLD).astype(int)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_bin, average='micro', zero_division=0)
print(f"\nMicro Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

# -----------------------------
# Confusion Matrices
# -----------------------------
def plot_cm(cm, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0','1'])
    plt.yticks(tick_marks, ['0','1'])
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()

print("\nPlotting per-class confusion matrices...")
plt.figure(figsize=(10, 20))
for i, t in enumerate(targets):
    cm = confusion_matrix(y_true[:,i], y_pred_bin[:,i])
    plt.subplot(7, 2, i+1)
    plot_cm(cm, t)
plt.tight_layout()
plt.show()

# -----------------------------
# ROC Curves
# -----------------------------
plt.figure(figsize=(10,10))
for i, t in enumerate(targets):
    try:
        fpr, tpr, _ = roc_curve(y_true[:,i], y_pred[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{t} (AUC={roc_auc:.2f})")
    except:
        continue
plt.plot([0,1],[0,1],'--',color='gray')
plt.legend(fontsize=8, ncol=2)
plt.title("Validation ROC Curves")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.show()

# -----------------------------
# Predict on test1
# -----------------------------
print("\nPredicting on test set...")

sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_rows = []
for img_name in sample_sub['Image_name']:
    path = os.path.join(TEST_DIR, img_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test image: {path}")
    test_rows.append((path, img_name))

def make_test_ds(rows):
    paths, names = zip(*rows)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(names)))
    ds = ds.map(lambda p, n: decode_test_img(p, n), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = make_test_ds(test_rows)

probs, names = [], []
for imgs, names_batch in test_ds:
    preds = model.predict(imgs, verbose=0)
    probs.append(preds)
    names.extend([n.numpy().decode() for n in names_batch])
probs = np.vstack(probs)

# -----------------------------
# Build Submission (Probabilities)
# -----------------------------
submission = pd.DataFrame(probs, columns=targets)
submission.insert(0, "Image_name", names)
submission = submission.set_index("Image_name").reindex(sample_sub["Image_name"]).reset_index()

# Clip to [0,1]
submission[targets] = submission[targets].clip(0.0, 1.0)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSaved submission with probabilities to: {SUBMISSION_PATH}")
print(submission.head())
print(f"Total rows: {len(submission)}")





