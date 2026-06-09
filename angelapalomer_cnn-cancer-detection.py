# -----------------------------
# Import libraries
# -----------------------------

# Silence Warnings
import warnings
warnings.filterwarnings("ignore")

# General libraries
import os
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display, HTML, Markdown

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.patches import Patch, Rectangle

# Image Processing
import cv2 as cv
from skimage import io
from skimage.transform import rotate
from tifffile import imread

# TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import image_dataset_from_directory

# Common Keras Layers
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, RandomZoom,
    Conv2D, MaxPooling2D, AveragePooling2D,
    Flatten, Dense, Dropout, BatchNormalization
)

# Profiling
import pandas_profiling as pp

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


# Set random_state for reproducibility
random_state = 86


# -----------------------------
# Load files
# -----------------------------
sample_submission = pd.read_csv("../input/histopathologic-cancer-detection/sample_submission.csv")
train_raw = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv")

train_path = "../input/histopathologic-cancer-detection/train/"
test_path = "../input/histopathologic-cancer-detection/test/"


# -----------------------------
# Initial look at train_raw
# -----------------------------
train_raw.head()


# -----------------------------
# Check for null data in train_raw
# -----------------------------
train_raw.info()


# -----------------------------
# Get number of images in train set and test set
# -----------------------------
n_train_raw = len(os.listdir("../input/histopathologic-cancer-detection/train"))
display(HTML(f"<strong>Number of images in Train Set:</strong> {n_train_raw}"))

n_test = len(os.listdir("../input/histopathologic-cancer-detection/test"))
display(HTML(f"<strong>Number of images in Test Set:</strong>  {n_test}"))


# -----------------------------
# Get label counts for train_raw
# -----------------------------
display(pd.DataFrame(data={"Counts": train_raw["label"].value_counts()}))


# -----------------------------
# Pie chart
# -----------------------------
colors = sns.color_palette("seismic", 2).as_hex()

fig = px.pie(
    train_raw, 
    values=train_raw["label"].value_counts().values,
    names=train_raw["label"].unique(),
    color_discrete_sequence=colors
)

fig.update_layout(
    title={
        "text": "Label Distribution (Pie Chart)",
        "y": 0.95,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": dict(size=18, weight="bold")
    },
    legend_title="Label Meaning",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.95,
        xanchor="left",
        x=1.02,
        bordercolor="Gainsboro",
        borderwidth=2
    )
)


# -----------------------------
# Histogram
# -----------------------------
colors = sns.color_palette("seismic", 2).as_hex()
ax = sns.countplot(
    x=train_raw["label"],
    palette="seismic"
)
ax.set(title="Label Distribution (Histogram)")
ax.title.set_fontweight("bold")

legend_patches = [
    Patch(color=colors[0], label="0 = Negative for Cancer"),
    Patch(color=colors[1], label="1 = Positive for Cancer")
]

plt.legend(handles=legend_patches, title="Label Meaning", loc="upper right")
plt.show()


# -----------------------------
# Visualize some images from train_raw
# -----------------------------
label_colors = sns.color_palette("seismic", 2)

fig, ax = plt.subplots(5, 5, figsize=(15, 15))

for i, axis in enumerate(ax.flat):
    file = str(train_path + train_raw.id[i] + ".tif")
    image = io.imread(file)
    axis.imshow(image,
                #cmap="gray"
               )
    label = train_raw.label[i]
    color = label_colors[label]
    box = Rectangle((32, 32), 32, 32,
                     linewidth=2,
                     edgecolor=color,
                     facecolor="none")
    axis.add_patch(box)
    
    # Set label below image
    axis.set(
        xlabel=f"{label} ({'Negative' if label==0 else 'Positive'})",
        xticks=[], yticks=[]
    )

plt.show()


# -----------------------------
# Inital Hyperparameters
# -----------------------------
img_size = (96, 96)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


# -----------------------------
# File paths
# -----------------------------
train_files = [os.path.join(train_path, f"{fid}.tif") for fid in train_raw.id]
train_targets = train_raw.label.values

# Grab all TIFF files in the folder
test_files = sorted([
    os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith(".tif")
])


# -----------------------------
# Dataset generator functions
# -----------------------------
def load_img_train(path, label):
    path = path.numpy().decode("utf-8")
    img = io.imread(path)
    # Convert to tensor early to avoid dtype mismatch
    img = tf.convert_to_tensor(img)
    
    if img.ndim == 2:                               # Grayscale
        img = tf.stack([img, img, img], axis=-1)    # Force RGB
    elif img.shape[-1] == 4:                        # RGBA
        img = img[..., :3]

    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def set_shape_train(img, label):
    img.set_shape((*img_size, 3))
    label.set_shape(())
    return img, label


def load_img_test(path):
    path = path.numpy().decode("utf-8")
    img = io.imread(path)

    img = tf.convert_to_tensor(img)

    # Grayscale
    if img.ndim == 2:
        img = tf.stack([img, img, img], axis=-1)

    # Weird TIFFs shaped (H, 3)
    if img.ndim == 2 and img.shape[-1] == 3:
        img = tf.expand_dims(img, axis=1)

    # RGBA
    if img.shape[-1] == 4:
        img = img[..., :3]

    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_img_test_wrapper(path):
    img = tf.py_function(load_img_test, [path], tf.float32)
    img.set_shape((*img_size, 3))
    return img


# -----------------------------
# Train + Val Set
# -----------------------------
train_raw_data = tf.data.Dataset.from_tensor_slices((train_files, train_targets))
train_raw_data = train_raw_data.map(
    lambda x, y: tf.py_function(load_img_train, [x, y], [tf.float32, tf.int64]),
    num_parallel_calls=AUTOTUNE
)
train_raw_data = train_raw_data.map(set_shape_train)
train_raw_data = train_raw_data.shuffle(1000)

val_split = 0.2
n_val = int(len(train_files) * val_split)

# Final split: Take/skip AFTER batching
val_set   = train_raw_data.take(n_val).batch(batch_size).cache().prefetch(AUTOTUNE)
train_set = train_raw_data.skip(n_val).batch(batch_size).cache().prefetch(AUTOTUNE)


# -----------------------------
# Test Set
# -----------------------------
test_set = tf.data.Dataset.from_tensor_slices(test_files)
test_set = test_set.map(load_img_test_wrapper, num_parallel_calls=AUTOTUNE)
test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)


# -----------------------------
# Sequential CNN Model
# -----------------------------
def seq_cnn(dropout_rate=0.2, lr=0.001, input_shape=(*img_size, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics = ["accuracy", tf.keras.metrics.AUC(name="roc_auc", curve="ROC")]
    )
    return model


# -----------------------------
# VGGNet Model
# -----------------------------
def vgg_net(dropout_rate=0.2, lr=0.001, input_shape=(*img_size, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics = ["accuracy", tf.keras.metrics.AUC(name="roc_auc", curve="ROC")]
    )
    return model


# -----------------------------
# Tuning and Metrics
# -----------------------------
hyperparams = [
    {"dropout_rate": 0.2, "lr": 0.001},
    {"dropout_rate": 0.2, "lr": 0.0005},
]

best_val_auc = 0
best_model_seq_cnn = None
seq_cnn_summary = []

def seq_cnn_predict(model, dataset, labeled=True):
    all_probs = []
    all_labels = []
    for batch in tqdm(dataset, desc="Predicting"):
        if labeled:
            imgs, labels = batch
            all_labels.append(labels.numpy())
        else:
            imgs = batch
        probs = model.predict(imgs, verbose=0)
        all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0).flatten()
    if labeled:
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        return all_probs, all_labels
    else:
        return all_probs

for params in hyperparams:
    print(f"\nTraining model with params: {params}")
    model = seq_cnn(**params)
    
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )]
    )
    
    # Predict train + val
    train_probs, train_labels = seq_cnn_predict(model, train_set, labeled=True)
    val_probs, val_labels     = seq_cnn_predict(model, val_set, labeled=True)

    # Convert probabilities to binary predictions
    train_preds = (train_probs > 0.5).astype(int)
    val_preds   = (val_probs > 0.5).astype(int)

    # Compute metrics
    train_auc = roc_auc_score(train_labels, train_probs)
    val_auc   = roc_auc_score(val_labels, val_probs)
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc   = accuracy_score(val_labels, val_preds)

    print(f"Train Acc: {train_acc:.4f} | Train ROC-AUC: {train_auc:.4f}")
    print(f"Val Acc: {val_acc:.4f} | Val   ROC-AUC: {val_auc:.4f}")
    
    # Save metrics for seq_cnn_summary
    seq_cnn_summary.append({
        "dropout_rate": params["dropout_rate"],
        "lr": params["lr"],
        "train_acc": train_acc,
        "train_auc": train_auc,
        "val_acc": val_acc,
        "val_auc": val_auc
    })
    
    # Track best model by val ROC-AUC
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_seq_cnn = model
        print("Best model updated!")


# -----------------------------
# Metrics summary table
# -----------------------------
seq_cnn_summary = pd.DataFrame(seq_cnn_summary)
print("\nHyperparameter tuning summary:")
print(seq_cnn_summary)


# -----------------------------
# Best model metrics
# -----------------------------
best_row = seq_cnn_summary.loc[seq_cnn_summary["val_auc"].idxmax()]

print("\nBest model metrics (based on val ROC-AUC):")
print(f"Dropout rate: {best_row['dropout_rate']}")
print(f"Learning rate: {best_row['lr']}")
print(f"Train Accuracy: {best_row['train_acc']:.4f}")
print(f"Train ROC-AUC: {best_row['train_auc']:.4f}")
print(f"Val Accuracy:   {best_row['val_acc']:.4f}")
print(f"Val ROC-AUC:   {best_row['val_auc']:.4f}")


# ------------------------
# Plot Results
# ------------------------
seq_cnn_summary["label"] = seq_cnn_summary.apply(
    lambda row: f"drop={row['dropout_rate']}, lr={row['lr']}", axis=1
)

x = range(len(seq_cnn_summary))
plt.figure(figsize=(10,6))
plt.plot(x, seq_cnn_summary["train_auc"], marker="o", linestyle="-", 
         color="RoyalBlue", label="Train ROC-AUC")
plt.plot(x, seq_cnn_summary["train_acc"], marker="s", linestyle="--", 
         color="RoyalBlue", label="Train Accuracy")

plt.plot(x, seq_cnn_summary["val_auc"], marker="o", linestyle="-",
         color="Crimson", label="Val ROC-AUC")
plt.plot(x, seq_cnn_summary["val_acc"], marker="s", linestyle="--",
         color="Crimson", label="Val Accuracy")

# Highlight best model
highlight_x = len(seq_cnn_summary) - 1
plt.axvspan(
    highlight_x - 0.1,  # left boundary
    highlight_x + 0.1,  # right boundary
    color="Plum",
    alpha=0.2,
    label="Best Model"
)

# Labels & style
plt.xticks(x, seq_cnn_summary["label"], rotation=45)
plt.title("Sequential CNN Model: Hyperparameter Tuning", weight="bold")
plt.ylabel("Score")
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# Sequential CNN on test set
# -----------------------------
test_probs_seq_cnn = seq_cnn_predict(best_model_seq_cnn, test_set, labeled=False)
test_files_sorted = sorted([f for f in os.listdir(test_path) if f.endswith(".tif")])
test_ids = [os.path.splitext(f)[0] for f in test_files_sorted]

submission_seq_cnn = pd.DataFrame({
    "id": test_ids,
    "label": test_probs_seq_cnn
})
#submission_seq_cnn.to_csv("/kaggle/working/submission_seq_cnn.csv", index=False)
#print("Submission CSV saved for Sequential CNN model.")


# -----------------------------
# Tuning and Metrics
# -----------------------------
hyperparams = [
    {"dropout_rate": 0.2, "lr": 0.001},
    {"dropout_rate": 0.2, "lr": 0.0005},
]

best_val_auc = 0
best_model_vgg = None
vgg_summary = []

def vgg_net_predict(model, dataset, labeled=True):
    all_probs = []
    all_labels = []
    for batch in tqdm(dataset, desc="Predicting"):
        if labeled:
            imgs, labels = batch
            all_labels.append(labels.numpy())
        else:
            imgs = batch
        probs = model.predict(imgs, verbose=0)
        all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0).flatten()
    if labeled:
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        return all_probs, all_labels
    else:
        return all_probs

for params in hyperparams:
    print(f"\nTraining model with params: {params}")
    model = vgg_net(**params)
    
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )]
    )
    
    # Predict train + val
    train_probs, train_labels = vgg_net_predict(model, train_set, labeled=True)
    val_probs, val_labels     = vgg_net_predict(model, val_set, labeled=True)

    # Convert probabilities to binary predictions
    train_preds = (train_probs > 0.5).astype(int)
    val_preds   = (val_probs > 0.5).astype(int)

    # Compute metrics
    train_auc = roc_auc_score(train_labels, train_probs)
    val_auc   = roc_auc_score(val_labels, val_probs)
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc   = accuracy_score(val_labels, val_preds)

    print(f"Train Acc: {train_acc:.4f} | Train ROC-AUC: {train_auc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}   | Val   ROC-AUC: {val_auc:.4f}")
    
    # Save metrics for vgg_summary
    vgg_summary.append({
        "dropout_rate": params["dropout_rate"],
        "lr": params["lr"],
        "train_acc": train_acc,
        "train_auc": train_auc,
        "val_acc": val_acc,
        "val_auc": val_auc
    })
    
    # Track best model by val ROC-AUC
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_vgg = model
        print("Best model updated!")
        

# -----------------------------
# Metrics summary table
# -----------------------------
vgg_summary = pd.DataFrame(vgg_summary)
print("\nHyperparameter tuning vgg_summary:")
print(vgg_summary)


# -----------------------------
# Best model metrics
# -----------------------------
best_row = vgg_summary.loc[vgg_summary["val_auc"].idxmax()]

print("\nBest model metrics (based on val ROC-AUC):")
print(f"Dropout rate: {best_row['dropout_rate']}")
print(f"Learning rate: {best_row['lr']}")
print(f"Train Accuracy: {best_row['train_acc']:.4f}")
print(f"Train ROC-AUC: {best_row['train_auc']:.4f}")
print(f"Val Accuracy:   {best_row['val_acc']:.4f}")
print(f"Val ROC-AUC:   {best_row['val_auc']:.4f}")


# ------------------------
# Plot Results
# ------------------------
vgg_summary["label"] = vgg_summary.apply(
    lambda row: f"drop={row['dropout_rate']}, lr={row['lr']}", axis=1
)

x = range(len(vgg_summary))
plt.figure(figsize=(10,6))
plt.plot(x, vgg_summary["train_auc"], marker="o", linestyle="-",
         color="RoyalBlue", label="Train ROC-AUC")
plt.plot(x, vgg_summary["train_acc"], marker="s", linestyle="--",
         color="RoyalBlue", label="Train Accuracy")

plt.plot(x, vgg_summary["val_auc"], marker="o", linestyle="-",
         color="Crimson", label="Val ROC-AUC")
plt.plot(x, vgg_summary["val_acc"], marker="s", linestyle="--",
         color="Crimson", label="Val Accuracy")

# Highlight best model
highlight_x = len(vgg_summary) - 1
plt.axvspan(
    highlight_x - 0.1,  # left boundary
    highlight_x + 0.1,  # right boundary
    color="Plum",
    alpha=0.2,
    label="Best Model"
)

# Labels & style
plt.xticks(x, vgg_summary["label"], rotation=45)
plt.title("VGGNet Model: Hyperparameter Tuning", weight="bold")
plt.ylabel("Score")
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# VGGNet on test set
# -----------------------------
test_probs_vgg = vgg_net_predict(best_model_vgg, test_set, labeled=False)
test_files_sorted = sorted([f for f in os.listdir(test_path) if f.endswith(".tif")])
test_ids = [os.path.splitext(f)[0] for f in test_files_sorted]

submission_vgg = pd.DataFrame({
    "id": test_ids,
    "label": test_probs_vgg
})
submission_vgg.to_csv("/kaggle/working/submission.csv", index=False)
print("CSV saved for VGGNet model.")


# -----------------------------
# Submission file
# -----------------------------
print(f"CSV saved for VGGNet model at {submission_file}")

