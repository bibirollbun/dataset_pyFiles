import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# TensorFlow / Keras imports 
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.callbacks import EarlyStopping


# Hugging Face
from transformers import TFViTModel, TFAutoModel


# ---- CONFIG ----
IMG_SIZE = 224
SAVE_DIR = "/kaggle/working/processed_data"
VINDR_PATH = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
NIH_PATH = "/kaggle/input/data"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---- LOAD DATA ----
vindr_df = pd.read_csv(os.path.join(VINDR_PATH, "train.csv"))
nih_df = pd.read_csv(os.path.join(NIH_PATH, "Data_Entry_2017.csv"))


# ---- BUILD NIH IMAGE MAP ----
def build_nih_path_map(base_path):
    path_map = {}
    for subdir in os.listdir(base_path):
        if subdir.startswith("images_"):
            img_dir = os.path.join(base_path, subdir, "images")
            for file in os.listdir(img_dir):
                path_map[file] = os.path.join(img_dir, file)
    return path_map


nih_image_map = build_nih_path_map(NIH_PATH)


# ---- NIH BINARY LABELS ----
nih_pos = nih_df[nih_df["Finding Labels"].str.contains("Nodule|Mass")]
nih_neg = nih_df[nih_df["Finding Labels"] == "No Finding"]
nih_neg_sampled = nih_neg.sample(n=len(nih_pos), random_state=42)


nih_binary_df = pd.concat([nih_pos, nih_neg_sampled])
nih_binary_df["label"] = nih_binary_df["Finding Labels"].apply(lambda x: 1 if "Nodule" in x or "Mass" in x else 0)
nih_binary_df = nih_binary_df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---- BUILD VINDR IMAGE MAP ----
def build_vindr_path_map(base_path):
    img_dir = os.path.join(base_path, "train")
    return {file.replace(".dicom", ""): os.path.join(img_dir, file)
            for file in os.listdir(img_dir) if file.endswith(".dicom")}


vindr_image_map = build_vindr_path_map(VINDR_PATH)


# ---- VINDR BINARY LABELS ----
vindr_binary_df = vindr_df[vindr_df["class_name"].isin(["Nodule/Mass", "No finding"])].copy()
vindr_binary_df["label"] = vindr_binary_df["class_name"].apply(lambda x: 1 if x == "Nodule/Mass" else 0)
vindr_binary_df = vindr_binary_df.drop_duplicates("image_id")


pos = vindr_binary_df[vindr_binary_df["label"] == 1]
neg = vindr_binary_df[vindr_binary_df["label"] == 0].sample(n=len(pos), random_state=42)
vindr_binary_df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)


# Safety checks (optional)
assert {'Image Index','Finding Labels'}.issubset(nih_binary_df.columns)
assert {'image_id','class_name','label'}.issubset(vindr_binary_df.columns)


# ---- CONFIG ----
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 30  # bump if you have more time/TPU
SEED = 42
AUTO = tf.data.AUTOTUNE


def stratified_split(df, label_col='label', test_size=0.2, seed=SEED):
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _read_nih_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = (img - np.mean(img)) / (np.std(img) + 1e-6)   # z-score normalization
    return np.stack([img, img, img], axis=-1)


def nih_generator(df, image_map, file_col='Image Index', label_col='label'):
    for _, row in df.iterrows():
        yield _read_nih_image(image_map[row[file_col]]), np.array(row[label_col], dtype=np.int32)


def _read_vindr_dicom(path):
    arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)   # z-score normalization
    return np.stack([arr, arr, arr], axis=-1)


def vindr_generator(df, image_map, file_col='image_id', label_col='label'):
    for _, row in df.iterrows():
        yield _read_vindr_dicom(image_map[row[file_col]]), np.array(row[label_col], dtype=np.int32)


def make_tfdata_from_generator(gen_fn, df, **kwargs):
    sig = (tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32), tf.TensorSpec((), tf.int32))
    return tf.data.Dataset.from_generator(lambda: gen_fn(df, **kwargs), output_signature=sig)


def prepare_dataset(ds, training=True):
    if training:
        ds = ds.shuffle(2048, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTO)


# -----------------------------
# Visualization
# -----------------------------

def visualize_samples(df, image_map, read_fn, title, file_col):
    samples = df.sample(5, random_state=SEED)
    plt.figure(figsize=(15, 3))
    for i, (_, row) in enumerate(samples.iterrows()):
        img = read_fn(image_map[row[file_col]])
        plt.subplot(1, 5, i+1)
        plt.imshow(img[..., 0], cmap='gray')
        plt.title(f"Label: {row['label']}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


nih_train_df, nih_val_df = stratified_split(nih_binary_df)
visualize_samples(nih_train_df, nih_image_map, _read_nih_image, 'NIH Training Samples', 'Image Index')

vindr_train_df, vindr_val_df = stratified_split(vindr_binary_df)
visualize_samples(vindr_train_df, vindr_image_map, _read_vindr_dicom, 'VinDr Training Samples', 'image_id')


def build_densenet_model_transfer(img_size=224, 
                                  unfreeze_layers=30, 
                                  dropout=0.2, 
                                  learning_rate=1e-5):
    inp = keras.Input((img_size, img_size, 3))

    base = keras.applications.DenseNet169(
        include_top=False,
        weights='imagenet',
        input_tensor=inp,
        pooling='avg'
    )

    # Freeze all layers first
    base.trainable = False

    # Unfreeze last N layers (skip BatchNorm for stability)
    for layer in base.layers[-unfreeze_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    x = layers.Dropout(dropout)(base.output)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out, name='DenseNet169_binary')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='pr_auc', curve='PR'),
            'accuracy'
        ]
    )
    return model


# def build_vit_model_transfer(hf_model_name="google/vit-base-patch16-224-in21k",
#                              img_size=224,
#                              unfreeze_layers=4,
#                              dropout=0.2,
#                              base_lr=1e-4,
#                              weight_decay=1e-4,
#                              warmup_epochs=2,
#                              total_epochs=20,
#                              steps_per_epoch=100):

#     inp = keras.Input((img_size, img_size, 3), dtype=tf.float32, name="pixel_values")

#     vit_backbone = TFAutoModel.from_pretrained(hf_model_name)
#     vit_backbone.trainable = False

#     if hasattr(vit_backbone, "vit"):
#         encoder_layers = vit_backbone.vit.encoder.layer
#         for layer in encoder_layers[-unfreeze_layers:]:
#             layer.trainable = True

#     hidden_size = vit_backbone.config.hidden_size

#     def vit_forward(tensor):
#         tensor = tf.transpose(tensor, perm=[0, 3, 1, 2])  # NHWC → NCHW
#         outputs = vit_backbone(pixel_values=tensor)
#         if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
#             return outputs.pooler_output
#         else:
#             return tf.reduce_mean(outputs.last_hidden_state, axis=1)

#     pooled = layers.Lambda(vit_forward, name="vit_backbone", output_shape=(hidden_size,))(inp)

#     h = layers.Dropout(dropout)(pooled)
#     h = layers.Dense(256, activation=tf.nn.gelu)(h)
#     h = layers.Dropout(dropout)(h)
#     out = layers.Dense(1, activation="sigmoid", name="out")(h)

#     model = keras.Model(inputs=inp, outputs=out, name="ViT_transfer_medical")

#     # ---- Native TF Cosine Decay ----
#     total_steps = steps_per_epoch * total_epochs
#     lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#         initial_learning_rate=base_lr,
#         decay_steps=total_steps
#     )

#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr_schedule
#     )

#     model.compile(
#         optimizer=optimizer,
#         loss="binary_crossentropy",
#         metrics=[
#             keras.metrics.Recall(name="recall"),
#             keras.metrics.AUC(name="auc"),
#             keras.metrics.AUC(name="pr_auc", curve="PR"),
#             "accuracy"
#         ]
#     )

#     return model


# =========================
# 5. Plotting helpers
# =========================
def plot_training_curves(history, metrics=None):
    hist = history.history
    if metrics is None:
        metrics = [m for m in hist.keys() if not m.startswith('val_') and m != 'loss']

    plt.figure(figsize=(15, 4))
    plt.subplot(1, len(metrics) + 1, 1)
    plt.plot(hist['loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()

    for i, metric in enumerate(metrics, start=2):
        plt.subplot(1, len(metrics) + 1, i)
        plt.plot(hist[metric], label=f'Train {metric}')
        plt.plot(hist[f'val_{metric}'], label=f'Val {metric}')
        plt.xlabel('Epoch'); plt.ylabel(metric)
        plt.title(metric.capitalize()); plt.legend()

    plt.tight_layout()
    plt.show()

def predict_on_tf_dataset(model, ds):
    y_true, y_pred = [], []
    for batch_x, batch_y in ds.unbatch().batch(256):
        preds = model.predict(batch_x, verbose=0)
        y_true.append(batch_y.numpy().ravel())
        y_pred.append(preds.ravel())
    return np.concatenate(y_true), np.concatenate(y_pred)

def plot_roc_pr_curves(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    print(f"Validation ROC AUC: {roc_auc:.4f}")
    print(f"Validation PR-AUC: {pr_auc:.4f}")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], '--', alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend()

    plt.tight_layout(); plt.show()


def evaluate_model(y_true, y_pred, threshold=0.5):
    """
    Extended evaluation: Classification report,
    specificity, confusion matrix, calibration curve
    """

    # --- Thresholding ---
    y_pred_bin = (y_pred >= threshold).astype(int)

    # --- Classification report (precision, recall, F1) ---
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_bin, digits=4))

    # --- Specificity ---
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.4f}")

    # --- Confusion Matrix ---
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_bin,
        display_labels=["No Cancer", "Cancer"],
        cmap="Blues"
    )
    disp.ax_.set_title("Confusion Matrix")
    plt.show()



# ======= DenseNet on VinDr =======
# print("\n=== Training DenseNet on VinDr ===")
train_df, val_df = stratified_split(vindr_binary_df, test_size=0.2)
train_ds = make_tfdata_from_generator(vindr_generator, train_df, image_map=vindr_image_map)
val_ds = make_tfdata_from_generator(vindr_generator, val_df, image_map=vindr_image_map)
train_ds = prepare_dataset(train_ds, training=True)
val_ds = prepare_dataset(val_ds, training=False)

# densenet_vindr = build_densenet_model_transfer(unfreeze_layers=100, learning_rate=0.00001)

# # Early stopping callback
# early_stop = EarlyStopping(
#     monitor="val_loss", 
#     patience=3, 
#     restore_best_weights=True,
#     verbose=1
# )

# history_dn_vindr = densenet_vindr.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS,
#     callbacks=[early_stop]
# )

# plot_training_curves(history_dn_vindr, metrics=['accuracy','recall','auc','pr_auc'])

# # Save the trained model
# MODEL_PATH = "densenet_vindr_model.h5"
# densenet_vindr.save(MODEL_PATH)
# MODEL_PATH = "densenet_vindr_model"
# densenet_vindr.save(MODEL_PATH)  # Saves in TensorFlow SavedModel format

# print(f"Model saved to {MODEL_PATH}")

# Load the model back
MODEL_PATH = "/kaggle/input/densenet_vindr_model/tensorflow2/default/1/densenet_vindr_model.h5"
# Rebuild model from code
densenet_vindr = build_densenet_model_transfer(unfreeze_layers=100, learning_rate=1e-5)

# Load only weights
densenet_vindr.load_weights(MODEL_PATH)
print("Model reloaded successfully.")

y_true, y_pred = predict_on_tf_dataset(densenet_vindr, val_ds)

# ROC & PR curves
plot_roc_pr_curves(y_true, y_pred)

# Evaluate across thresholds
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"\nThreshold {thr}")
    evaluate_model(y_true, y_pred, threshold=thr)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    img_array: np.array of shape (1, H, W, 3), preprocessed image
    model: trained keras model
    last_conv_layer_name: name of last conv layer
    """
    # Build a sub-model that maps input -> (conv outputs, predictions)
    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_class = predictions[:, 0]  # binary classification

    # Compute gradients of class score wrt conv feature maps
    grads = tape.gradient(pred_class, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, cmap='jet'):
    heatmap = np.uint8(255 * heatmap)
    cmap = cm.get_cmap(cmap)
    colored = cmap(np.arange(256))[:, :3]
    heatmap_colored = colored[heatmap]

    heatmap_colored = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).resize(img.size)
    overlayed = Image.blend(img.convert("RGBA"), heatmap_colored.convert("RGBA"), alpha)
    return overlayed



print(type(densenet_vindr.input))
print(densenet_vindr.input)
print(densenet_vindr.layers[0])



def plot_gradcam_samples(model, val_ds, last_conv_layer_name, n_samples=6, threshold=0.5):
    """
    Plot Original / Grad-CAM / Overlay for n_samples from val_ds,
    including True label and Predicted label (with probability).

    Args:
        model: trained Keras model (binary output with sigmoid).
        val_ds: tf.data.Dataset yielding (image, label).
        last_conv_layer_name: str, name of last conv layer for Grad-CAM.
        n_samples: int, number of samples to visualize.
        threshold: float, decision threshold for binary class (default 0.5).
    """
    # Collect exactly n_samples regardless of dataset batch size
    sample_imgs, sample_labels = [], []
    for img, label in val_ds.unbatch().take(n_samples):
        sample_imgs.append(img.numpy())
        sample_labels.append(int(label.numpy()))

    rows, cols = len(sample_imgs), 3  # one row per sample: Original | Heatmap | Overlay
    plt.figure(figsize=(5 * cols, 4 * rows))

    for idx, (img, true_label) in enumerate(zip(sample_imgs, sample_labels)):
        # Prepare input
        x_in = np.expand_dims(img, axis=0)

        # Predict probability and class
        prob = model(x_in, training=False).numpy().squeeze()
        prob = float(np.array(prob).squeeze())  # robust squeeze
        pred_label = int(prob >= threshold)

        # Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(x_in, model, last_conv_layer_name)

        # Overlay
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        overlayed_img = overlay_heatmap(img_pil, heatmap)

        # --- Plotting ---
        # Original
        ax = plt.subplot(rows, cols, idx * 3 + 1)
        ax.imshow(img)
        ax.set_title(f"Original\nTrue={true_label}")
        ax.axis("off")

        # Heatmap
        ax = plt.subplot(rows, cols, idx * 3 + 2)
        ax.imshow(heatmap, cmap="jet")
        ax.set_title("Grad-CAM Heatmap")
        ax.axis("off")

        # Overlay with labels
        ax = plt.subplot(rows, cols, idx * 3 + 3)
        ax.imshow(overlayed_img)
        ax.set_title(f"Overlay\nTrue={true_label} | Pred={pred_label} (p={prob:.2f})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


plot_gradcam_samples(densenet_vindr, val_ds, "conv5_block32_concat", n_samples=6, threshold=0.5)


def plot_gradcam_misclassified(model, val_ds, last_conv_layer_name, max_samples=10, threshold=0.5):
    """
    Plot Grad-CAM for only misclassified samples from val_ds.
    
    Args:
        model: trained Keras model (binary output with sigmoid).
        val_ds: tf.data.Dataset yielding (image, label).
        last_conv_layer_name: str, name of last conv layer for Grad-CAM.
        max_samples: int, maximum number of misclassified samples to show.
        threshold: float, decision threshold for binary class (default 0.5).
    """
    sample_imgs, true_labels, pred_labels, probs = [], [], [], []

    # Go through dataset and collect misclassified samples
    for img, label in val_ds.unbatch():
        x_in = np.expand_dims(img.numpy(), axis=0)

        # Predict
        prob = model(x_in, training=False).numpy().squeeze()
        prob = float(np.array(prob).squeeze())  # robust squeeze
        pred_label = int(prob >= threshold)
        true_label = int(label.numpy())

        if pred_label != true_label:  # misclassified
            sample_imgs.append(img.numpy())
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            probs.append(prob)

        if len(sample_imgs) >= max_samples:
            break

    if len(sample_imgs) == 0:
        print("No misclassified samples found in the given dataset.")
        return

    rows, cols = len(sample_imgs), 3
    plt.figure(figsize=(5 * cols, 4 * rows))

    for idx, (img, t, p, prob) in enumerate(zip(sample_imgs, true_labels, pred_labels, probs)):
        x_in = np.expand_dims(img, axis=0)
        heatmap = make_gradcam_heatmap(x_in, model, last_conv_layer_name)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        overlayed_img = overlay_heatmap(img_pil, heatmap)

        # Original
        ax = plt.subplot(rows, cols, idx * 3 + 1)
        ax.imshow(img)
        ax.set_title(f"Original\nTrue={t}")
        ax.axis("off")

        # Heatmap
        ax = plt.subplot(rows, cols, idx * 3 + 2)
        ax.imshow(heatmap, cmap="jet")
        ax.set_title("Grad-CAM Heatmap")
        ax.axis("off")

        # Overlay with misclassification info
        ax = plt.subplot(rows, cols, idx * 3 + 3)
        ax.imshow(overlayed_img)
        ax.set_title(f"Overlay\nTrue={t} | Pred={p} (p={prob:.2f})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()




# Show 5 misclassified samples with Grad-CAM
plot_gradcam_misclassified(densenet_vindr, val_ds, "conv5_block32_concat", max_samples=5, threshold=0.5)








