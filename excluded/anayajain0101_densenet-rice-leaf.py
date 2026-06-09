import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
IMG_SIZE = (256, 256)
BATCH = 32

SOURCE = "/kaggle/input/paddy-disease-classification/train_images"
TRAIN = "/kaggle/working/train"
VAL   = "/kaggle/working/val"

# ------------------------------------
# 1. Train/Validation Split (80/20)
# ------------------------------------
from sklearn.model_selection import train_test_split
import shutil

os.makedirs(TRAIN, exist_ok=True)
os.makedirs(VAL, exist_ok=True)

for cls in os.listdir(SOURCE):
    cls_path = os.path.join(SOURCE, cls)
    if not os.path.isdir(cls_path): continue

    files = os.listdir(cls_path)
    train_f, val_f = train_test_split(files, test_size=0.2, random_state=SEED)

    os.makedirs(os.path.join(TRAIN, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL, cls), exist_ok=True)

    for f in train_f:
        shutil.copy(os.path.join(cls_path, f), os.path.join(TRAIN, cls, f))
    for f in val_f:
        shutil.copy(os.path.join(cls_path, f), os.path.join(VAL, cls, f))

print("Split done.")

# ------------------------------------
# 2. Image Generators
# ------------------------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = val_gen.flow_from_directory(
    VAL,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = len(train_ds.class_indices)

# ------------------------------------
# 3. DenseNet121 Model
# ------------------------------------
base = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3)
)
base.trainable = False   # freeze base for stability in JEI setting

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------
# 4. Train Model
# ------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop],
    verbose=2
)

model.save("/kaggle/working/densenet_bio_tool.h5")
print("DenseNet model saved!")



model.save("/kaggle/working/densenet_bio_tool.h5")
print("Model saved successfully!")


# ============================================================
# FINAL, FULLY-CORRECTED SEVERITY EXTRACTION PIPELINE
# ============================================================
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt

# --------------------------
# 1. LOAD MODEL
# --------------------------
model_path = "/kaggle/working/densenet_bio_tool.h5"
model = load_model(model_path)
print("Model loaded!")

# Force graph build
_ = model.predict(np.zeros((1,256,256,3)))
print("Model graph initialized!")

# Split into backbone + head
backbone = model.layers[0]       # DenseNet121
gap_layer = model.layers[1]
bn1       = model.layers[2]
dense1    = model.layers[3]
bn2       = model.layers[4]
dense2    = model.layers[5]

# --------------------------
# 2. LOAD METADATA
# --------------------------
meta = pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")

# --------------------------
# 3. LOAD VAL IMAGES
# --------------------------
IMG_SIZE = (256,256)
VAL_DIR = "/kaggle/working/val"

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    class_mode=None,
    shuffle=False,
    batch_size=1
)

index_to_class = {v:k for k,v in val_ds.class_indices.items()}

# --------------------------
# 4. FULLY-WORKING GRAD-CAM
# --------------------------
def get_gradcam(img_array, layer_name="conv5_block16_concat"):

    # Build model: DenseNet input -> conv layer + backbone output
    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer(layer_name).output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_out, backbone_feats = grad_model(img_array)

        # Manually pass through classification head
        x = gap_layer(backbone_feats)
        x = bn1(x, training=False)
        x = dense1(x)
        x = bn2(x, training=False)
        preds = dense2(x)

        class_idx = tf.argmax(preds[0])
        class_score = preds[:, class_idx]

    # Compute gradients
    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]

    # Weighted sum of activation maps
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-10)

    return heatmap   # <- FIXED

# --------------------------
# 5. SEVERITY EXTRACTION LOOP
# --------------------------
records = []

for i in range(len(val_ds)):
    img = val_ds[i]  # shape (1,256,256,3)

    # prediction
    preds = model.predict(img, verbose=0)
    pred_idx = np.argmax(preds)
    pred_label = index_to_class[pred_idx]

    # filename → metadata mapping
    filename = val_ds.filenames[i]
    img_id = filename.split("/")[-1]

    row = meta[meta["image_id"] == img_id]
    if row.empty:
        continue

    age = int(row["age"].values[0])
    true_label = row["label"].values[0]

    # Grad-CAM heatmap
    heatmap = get_gradcam(img)

    # Convert to mask
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    mask = (heatmap_resized > 0.4).astype(np.uint8)
    severity = mask.sum() / mask.size

    records.append([img_id, age, true_label, pred_label, severity])

    if i % 200 == 0:
        print(f"Processed {i} images...")

# --------------------------
# 6. SAVE CSV
# --------------------------
df = pd.DataFrame(records, columns=["image_id", "age", "true_label", "pred_label", "severity"])
df.to_csv("/kaggle/working/severity_scores.csv", index=False)

print("\nSeverity extraction complete!")
df.head()


# ================================
# JEI ANALYSIS + PLOTS CELL
# ================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------
# 1. LOAD SEVERITY CSV
# ---------------------------------------
csv_path = "/kaggle/working/severity_scores.csv"
assert os.path.exists(csv_path), "CSV not found! Did you run severity extraction?"
df = pd.read_csv(csv_path)

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['severity'] = pd.to_numeric(df['severity'], errors='coerce')
df = df.dropna(subset=['age','severity'])

print("Rows:", len(df))
print(df.head())

# ---------------------------------------
# 2. SUMMARY STATISTICS
# ---------------------------------------
print("\n=== SUMMARY ===")
print(df[['age','severity']].describe())

print("\nAge range:", df['age'].min(), "-", df['age'].max())
print("Severity range:", df['severity'].min(), "-", df['severity'].max())

# ---------------------------------------
# 3. CORRELATIONS (JEI-SAFE)
# ---------------------------------------
pearson_r, pearson_p = stats.pearsonr(df['age'], df['severity'])
spearman_r, spearman_p = stats.spearmanr(df['age'], df['severity'])

print(f"\nPearson r = {pearson_r:.4f}, p = {pearson_p:.4g}")
print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.4g}")

# ---------------------------------------
# 4. SIMPLE LINEAR REGRESSION
# ---------------------------------------
slope, intercept, r_val, p_val, std_err = stats.linregress(df['age'], df['severity'])
print(f"\nLinear regression: severity = {slope:.6f}*age + {intercept:.6f}")
print(f"R² = {r_val**2:.4f}, p = {p_val:.4g}")

# ---------------------------------------
# 5. KRUSKAL–WALLIS (JEI-friendly ANOVA)
# ---------------------------------------
groups = [grp['severity'].values for name, grp in df.groupby('true_label')]
kw_stat, kw_p = stats.kruskal(*groups)
print(f"\nKruskal–Wallis H = {kw_stat:.4f}, p = {kw_p:.4g}")

# ---------------------------------------
# 6. AGE BINS
# ---------------------------------------
bins = [0,20,40,60,80,200]
labels = ['0–20','21–40','41–60','61–80','81+']
df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)

print("\nSeverity by age bin:")
print(df.groupby('age_bin').severity.agg(['count','mean','median','std']))

# ---------------------------------------
# 7. PLOT DIRECTORY
# ---------------------------------------
out_dir = "/kaggle/working/jei_plots"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------
# 8. FIGURE 1: Age vs Severity (scatter + line)
# ---------------------------------------
plt.figure(figsize=(6,5))
plt.scatter(df['age'], df['severity'], alpha=0.35)
x = np.linspace(df['age'].min(), df['age'].max(), 300)
y = intercept + slope*x
plt.plot(x, y)
plt.xlabel("Plant Age (days)")
plt.ylabel("Severity (fraction of leaf area)")
plt.title("Age vs Severity (scatter + trendline)")
plt.tight_layout()
plt.savefig(f"{out_dir}/figure_age_vs_severity.png", dpi=150)
plt.close()

# ---------------------------------------
# 9. FIGURE 2: Severity by disease (top diseases)
# ---------------------------------------
counts = df['true_label'].value_counts()
diseases = counts[counts >= 20].index.tolist()  # only stable categories

if len(diseases) > 0:
    data = [df[df['true_label']==d]['severity'].values for d in diseases]
    plt.figure(figsize=(10,4))
    plt.boxplot(data, labels=diseases, vert=True, showfliers=False)
    plt.xticks(rotation=60)
    plt.ylabel("Severity")
    plt.title("Severity distribution across disease types")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/figure_severity_by_disease.png", dpi=150)
    plt.close()

# ---------------------------------------
# 10. FIGURE 3: Mean severity per disease
# ---------------------------------------
mean_sev = df.groupby('true_label').severity.mean().sort_values(ascending=False)
plt.figure(figsize=(9,4))
plt.bar(mean_sev.index, mean_sev.values)
plt.xticks(rotation=60)
plt.ylabel("Mean Severity")
plt.title("Mean severity per disease type")
plt.tight_layout()
plt.savefig(f"{out_dir}/figure_mean_severity.png", dpi=150)
plt.close()

# ---------------------------------------
# 11. FIGURE 4: Age-bin severity
# ---------------------------------------
agebin_stats = df.groupby('age_bin').severity.mean()
plt.figure(figsize=(6,4))
plt.bar(agebin_stats.index.astype(str), agebin_stats.values)
plt.xlabel("Age Bin (days)")
plt.ylabel("Mean Severity")
plt.title("Severity vs Plant Age Group")
plt.tight_layout()
plt.savefig(f"{out_dir}/figure_agebin_severity.png", dpi=150)
plt.close()

print("\nAll plots saved in:", out_dir)
print("Done!")



model.save("/kaggle/working/densenet_bio_tool.h5")
print("Saved!")


# ============================================================
# JEI FINAL FIX CELL (PASTE AT END ONLY)
# Fixes: val_ds + model reload + Grad-CAM + visualization
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input

# --------------------------
# 1. RELOAD VALIDATION DATA (SAFE)
# --------------------------
IMG_SIZE = (256, 256)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    "/kaggle/input/paddy-disease-classification/train_images",
    target_size=IMG_SIZE,
    class_mode=None,
    shuffle=False,
    batch_size=1
)

print("val_ds loaded:", len(val_ds))

# --------------------------
# 2. RELOAD MODEL + LAYERS (CRITICAL)
# --------------------------
model = tf.keras.models.load_model("/kaggle/working/densenet_bio_tool.h5")

backbone = model.layers[0]
gap_layer = model.layers[1]
bn1 = model.layers[2]
dense1 = model.layers[3]
bn2 = model.layers[4]
dense2 = model.layers[5]

# --------------------------
# 3. REBUILD GRAD-CAM MODEL
# --------------------------
grad_model = tf.keras.models.Model(
    inputs=backbone.input,
    outputs=[
        backbone.get_layer("conv5_block16_concat").output,
        backbone.output
    ]
)

print("Grad-CAM model ready!")

# --------------------------
# 4. GRAD-CAM FUNCTION
# --------------------------
def get_gradcam(img_array):
    with tf.GradientTape() as tape:
        conv_out, backbone_feats = grad_model(img_array)

        x = gap_layer(backbone_feats)
        x = bn1(x, training=False)
        x = dense1(x)
        x = bn2(x, training=False)
        preds = dense2(x)

        class_idx = tf.argmax(preds[0])
        class_score = preds[:, class_idx]

    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

# --------------------------
# 5. VISUALIZATION FUNCTION
# --------------------------
def show_gradcam(img_tensor, heatmap):
    img = img_tensor[0]

    img = img - img.min()
    img = img / (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

# --------------------------
# 6. RUN SAMPLE VISUALS (JEI FIGURE)
# --------------------------
for i in [0, 50, 100, 150]:
    img = val_ds[i]
    heatmap = get_gradcam(img)
    show_gradcam(img, heatmap)

plt.savefig("/kaggle/working/gradcam_example.png", dpi=200)


import os
print(os.listdir("/kaggle/working"))


model.save("/kaggle/working/densenet_bio_tool.h5")
print("Model saved")


df.to_csv("/kaggle/working/severity_scores.csv", index=False)
print("CSV saved")


import os
print(os.listdir("/kaggle/working/jei_plots"))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get true labels
y_true = val_ds.classes

# Get predictions
y_pred_probs = model.predict(val_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Class names
class_names = list(val_ds.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Confusion Matrix - DenseNet121 Model")
plt.tight_layout()
plt.show()

plt.savefig("/kaggle/working/confusion_matrix.png", dpi=200)


from sklearn.metrics import classification_report

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("Classification Report:\n")
print(report)


import pandas as pd

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

df_report


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

n_classes = len(class_names)

# Binarize labels
y_true_bin = label_binarize(y_true, classes=range(n_classes))

# Model probabilities
y_score = model.predict(val_ds, verbose=1)

# ROC per class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot
plt.figure(figsize=(10,8))
colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "purple", "brown", "pink", "gray", "olive"])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Multi-class)")
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("/kaggle/working/roc.png", dpi=200)


model.save("/kaggle/working/densenet_bio_tool.h5")
print("Model saved")


import matplotlib.pyplot as plt

# Accuracy
plt.figure(figsize=(6,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/jei_plots/training_validation_accuracy.png", dpi=200)
plt.show()

# Loss
plt.figure(figsize=(6,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/jei_plots/training_validation_loss.png", dpi=200)
plt.show()




# Post-hoc Dunn test
!pip install scikit-posthocs

import scikit_posthocs as sp

dunn = sp.posthoc_dunn(
    df,
    val_col='severity',
    group_col='true_label',
    p_adjust='bonferroni'
)

dunn


# =========================================
# LOAD DATA (IMPORTANT - fixes df error)
# =========================================
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

df = pd.read_csv("/kaggle/working/severity_scores.csv")

print("Data loaded:", df.shape)
df.head()


# =========================================
# 1. CLASSIFICATION REPORT (per disease)
# =========================================
y_true = df['true_label']
y_pred = df['pred_label']

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred))


# =========================================
# 2. CONFUSION MATRIX (optional but useful)
# =========================================
cm = confusion_matrix(y_true, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(cm)


# =========================================
# 3. ACCURACY BY SEVERITY (VERY IMPORTANT)
# =========================================

# Low severity (<0.1)
low_sev = df[df['severity'] < 0.1]
acc_low = (low_sev['true_label'] == low_sev['pred_label']).mean()

# Medium severity (0.1–0.4)
mid_sev = df[(df['severity'] >= 0.1) & (df['severity'] < 0.4)]
acc_mid = (mid_sev['true_label'] == mid_sev['pred_label']).mean()

# High severity (>0.4)
high_sev = df[df['severity'] >= 0.4]
acc_high = (high_sev['true_label'] == high_sev['pred_label']).mean()

print("\n=== ACCURACY BY SEVERITY ===")
print(f"Low severity (<0.1): {acc_low:.2f}")
print(f"Medium severity (0.1–0.4): {acc_mid:.2f}")
print(f"High severity (>0.4): {acc_high:.2f}")


# =========================================
# 4. SAVE RESULTS (optional but good)
# =========================================
results = pd.DataFrame({
    "severity_range": ["<0.1", "0.1–0.4", ">0.4"],
    "accuracy": [acc_low, acc_mid, acc_high]
})

results.to_csv("/kaggle/working/jei_plots/accuracy_by_severity.csv", index=False)

print("\nSaved accuracy_by_severity.csv")


from tensorflow.keras.models import load_model

model = load_model("/kaggle/working/densenet_bio_tool.h5")
print("Model loaded successfully!")

