import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import numpy as np
import pandas as pd
import tensorflow as tf
import h5py

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Giảm lỗi VRAM / CUDA lặt vặt
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set memory growth:", e)

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))


DATA_DIR = "/kaggle/input/isic-2024-challenge"  # đổi nếu chạy local
META_PATH = f"{DATA_DIR}/train-metadata.csv"
HDF5_PATH = f"{DATA_DIR}/train-image.hdf5"

df = pd.read_csv(META_PATH, low_memory=False)

assert "isic_id" in df.columns and "target" in df.columns, "Thiếu cột isic_id hoặc target"

print("Total rows:", len(df))
print("Label counts:\n", df["target"].value_counts())


NEG_SAMPLE = 19650

pos_df = df[df["target"] == 1].copy()
neg_df = df[df["target"] == 0].sample(n=NEG_SAMPLE, random_state=SEED)

sample_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)

print("After sampling:\n", sample_df["target"].value_counts())


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    sample_df,
    test_size=0.2,
    random_state=SEED,
    stratify=sample_df["target"]
)

print("Train split:\n", train_df["target"].value_counts())
print("Val split:\n", val_df["target"].value_counts())


h5f = h5py.File(HDF5_PATH, "r")
print("HDF5 keys sample:", list(h5f.keys())[:5])

test_id = train_df.iloc[0]["isic_id"]
raw = h5f[test_id][()]

print("Test id:", test_id)
print("raw type:", type(raw))
print("raw dtype:", getattr(raw, "dtype", None))
print("raw shape:", getattr(raw, "shape", None))


# Loader chuẩn ISIC 2024
import matplotlib.pyplot as plt

def load_img_uint8(isic_id: str) -> np.ndarray:
    """
    Load ảnh từ HDF5 theo isic_id.
    Hỗ trợ:
      - ảnh array (H,W,3)/(H,W,4)
      - JPEG/PNG bytes (np.void/bytes/1D ndarray)
    Return: uint8 RGB (H,W,3)
    """
    data = h5f[isic_id][()]

    # Case 1: đã là ảnh array
    if isinstance(data, np.ndarray) and data.ndim == 3:
        if data.shape[-1] == 3:
            return data.astype(np.uint8)
        if data.shape[-1] == 4:
            return data[..., :3].astype(np.uint8)

    # Case 2: bytes -> decode
    if isinstance(data, np.void):
        img_bytes = data.tobytes()
    elif isinstance(data, (bytes, bytearray)):
        img_bytes = bytes(data)
    elif isinstance(data, np.ndarray):
        img_bytes = data.tobytes()
    else:
        img_bytes = bytes(data)

    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    return img.numpy().astype(np.uint8)

# Test loader 1 ảnh
img = load_img_uint8(test_id)
print("Loaded image:", img.shape, img.dtype)

plt.figure(figsize=(3,3))
plt.imshow(img)
plt.axis("off")
plt.show()


# Preprocess cho ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 224

def preprocess_resnet(img_uint8: np.ndarray) -> np.ndarray:
    img = tf.convert_to_tensor(img_uint8, dtype=tf.uint8)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img.numpy()


# Build ResNet50 (trích feature, không train)
from tensorflow.keras.applications import ResNet50

feat_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
feat_model.trainable = False

print("Feature output shape:", feat_model.output_shape)  # (None, 2048)


# Hàm extract_features theo batch
BATCH_SIZE = 64

def extract_features(df_in: pd.DataFrame, batch_size=64) -> np.ndarray:
    ids = df_in["isic_id"].astype(str).tolist()

    feats_list = []
    batch_imgs = []

    for i, isic_id in enumerate(ids):
        img = load_img_uint8(isic_id)
        img = preprocess_resnet(img)
        batch_imgs.append(img)

        # chạy khi đủ batch hoặc tới cuối
        if len(batch_imgs) >= batch_size or i == len(ids) - 1:
            x = np.stack(batch_imgs, axis=0)        # (B,224,224,3)
            f = feat_model.predict(x, verbose=0)    # (B,2048)
            feats_list.append(f)
            batch_imgs.clear()

        if (i + 1) % 2000 == 0:
            print(f"Extracted {i+1}/{len(ids)}")

    return np.concatenate(feats_list, axis=0)


# Test nhanh 128 ảnh
X_tmp = extract_features(train_df.iloc[:128], batch_size=64)
print("X_tmp shape:", X_tmp.shape)  # (128, 2048)


# Extract full features train/val
print("Extract train features...")
X_train = extract_features(train_df, batch_size=BATCH_SIZE)
y_train = train_df["target"].values.astype(int)

print("Extract val features...")
X_val = extract_features(val_df, batch_size=BATCH_SIZE)
y_val = val_df["target"].values.astype(int)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)


# Train SVM
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Scale feature (rất quan trọng cho SVM)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# Train SVM
svm = SVC(
    kernel="rbf",            # nếu chậm -> đổi "linear"
    C=2.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

print("Training SVM...")
svm.fit(X_train_s, y_train)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Predict probability + label
val_prob = svm.predict_proba(X_val_s)[:, 1]
val_pred = (val_prob >= 0.5).astype(int)

# 2. Metrics
acc = accuracy_score(y_val, val_pred)
precision = precision_score(y_val, val_pred, pos_label=1)
recall = recall_score(y_val, val_pred, pos_label=1)
f1 = f1_score(y_val, val_pred, pos_label=1)

print("===== MODEL EVALUATION =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Tính FPR, TPR
fpr, tpr, thresholds = roc_curve(y_val, val_prob)

# Tính AUC
auc_score = roc_auc_score(y_val, val_prob)

# Vẽ ROC Curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - Skin Cancer Classification (SVM + ResNet50)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("AUC:", auc_score)


# So sánh metrics khi đổi threshold (0.3 / 0.4 / 0.5)
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5]
results = []

for th in thresholds:
    val_pred_th = (val_prob >= th).astype(int)
    
    acc = accuracy_score(y_val, val_pred_th)
    precision = precision_score(y_val, val_pred_th, pos_label=1)
    recall = recall_score(y_val, val_pred_th, pos_label=1)
    f1 = f1_score(y_val, val_pred_th, pos_label=1)
    
    results.append({
        "Threshold": th,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

results_df = pd.DataFrame(results)
results_df


from sklearn.feature_selection import SelectKBest, f_classif

K = 50  # thử 10, 20, 50, 100 (10 thường hơi ít)
selector = SelectKBest(score_func=f_classif, k=K)

X_train_k = selector.fit_transform(X_train_s, y_train)
X_val_k   = selector.transform(X_val_s)

top_idx = selector.get_support(indices=True)
print("Selected K =", K)
print("Top feature indices:", top_idx[:20], "...")
print("Shapes:", X_train_k.shape, X_val_k.shape)


# Train trên các feature đã chọn
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    C=2.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=SEED
)
svm.fit(X_train_k, y_train)

val_prob = svm.predict_proba(X_val_k)[:, 1]


# Threshold + Metrics (Accuracy/Precision/Recall/F1)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

th = 0.3  # bạn có thể đổi 0.1-0.5
val_pred = (val_prob >= th).astype(int)

print("Threshold:", th)
print("Accuracy :", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred, zero_division=0))
print("Recall   :", recall_score(y_val, val_pred, zero_division=0))
print("F1-score :", f1_score(y_val, val_pred, zero_division=0))


for th in [0.05, 0.1, 0.2, 0.3]:
    pred = (val_prob >= th).astype(int)
    print(f"\nThreshold = {th}")
    print("Predicted positives:", np.sum(pred == 1))
    print("Recall:", recall_score(y_val, pred, zero_division=0))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

print("Scaled shapes:", X_train_s.shape, X_val_s.shape)


from sklearn.linear_model import LogisticRegression

print("Training Logistic Regression...")

lr = LogisticRegression(
    max_iter=3000,          # đủ lớn cho hội tụ
    class_weight="balanced",# xử lý mất cân bằng
    solver="liblinear"      # ổn định cho binary classification
)

lr.fit(X_train_s, y_train)


val_prob = lr.predict_proba(X_val_s)[:, 1]

print("val_prob range:", val_prob.min(), "→", val_prob.max())


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

th = 0.3   # thử 0.05 – 0.5
val_pred = (val_prob >= th).astype(int)

print("Threshold:", th)
print("Accuracy :", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred, zero_division=0))
print("Recall   :", recall_score(y_val, val_pred, zero_division=0))
print("F1-score :", f1_score(y_val, val_pred, zero_division=0))
print("Predicted positives:", (val_pred == 1).sum())


import pandas as pd
import numpy as np

thresholds = np.round(np.linspace(0.01, 0.5, 50), 3)
rows = []

for th in thresholds:
    pred = (val_prob >= th).astype(int)
    rows.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "Pred_Pos": int((pred == 1).sum())
    })

df_lr = pd.DataFrame(rows)
df_lr.sort_values("Recall", ascending=False).head(10)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_val, val_prob)
auc = roc_auc_score(y_val, val_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"LR (AUC={auc:.4f})")
plt.plot([0,1],[0,1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Logistic Regression (ResNet50 features)")
plt.legend()
plt.grid(True)
plt.show()

print("ROC-AUC:", auc)


# Train Random Forest
from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,          # số cây
    max_depth=None,            # để cây tự học
    min_samples_leaf=5,        # tránh overfit
    class_weight="balanced",   # CỰC KỲ quan trọng
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)


val_prob = rf.predict_proba(X_val)[:, 1]

print("val_prob range:", val_prob.min(), "→", val_prob.max())


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

thresholds = [0.01, 0.03, 0.05, 0.1]
rows = []

for th in thresholds:
    val_pred = (val_prob >= th).astype(int)
    rows.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_val, val_pred),
        "Precision": precision_score(y_val, val_pred, zero_division=0),
        "Recall": recall_score(y_val, val_pred, zero_division=0),
        "F1": f1_score(y_val, val_pred, zero_division=0),
        "Pred_Pos": int((val_pred == 1).sum())
    })

import pandas as pd
pd.DataFrame(rows)


ths = np.round(np.linspace(0.01, 0.5, 50), 3)
rows = []

for th in ths:
    pred = (val_prob >= th).astype(int)
    rows.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
        "Pred_Pos": int((pred == 1).sum())
    })

df_rf = pd.DataFrame(rows)
df_rf.sort_values("Recall", ascending=False).head(10)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_val, val_prob)
auc = roc_auc_score(y_val, val_prob)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"RF (AUC={auc:.4f})")
plt.plot([0,1],[0,1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.show()

print("ROC-AUC:", auc)


from sklearn.metrics import precision_recall_curve, average_precision_score

prec, rec, _ = precision_recall_curve(y_val, val_prob)
ap = average_precision_score(y_val, val_prob)

plt.figure(figsize=(7,6))
plt.plot(rec, prec, label=f"PR (AP={ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.show()

print("PR-AUC (AP):", ap)


# Feature importance
importances = rf.feature_importances_
top10_idx = np.argsort(importances)[-10:]

print("Top 10 important feature indices:", top10_idx)
print("Importance scores:", importances[top10_idx])


import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# tránh TF ăn full VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except:
            pass


df = pd.read_csv(META_PATH, low_memory=False)
df = df[["isic_id", "target"]].copy()
print(df.shape)
print(df["target"].value_counts())


# Split train/val
from sklearn.model_selection import train_test_split

SEED = 42

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["target"]
)

print("Train:", train_df["target"].value_counts())
print("Val:", val_df["target"].value_counts())


IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(img, label, training=False):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    label = tf.cast(label, tf.float32)
    return img, label


# --- HDF5 global handle ---
_h5 = None

def _get_h5():
    global _h5
    if _h5 is None:
        _h5 = h5py.File(HDF5_PATH, "r")
    return _h5

def _load_image_py(isic_id):
    isic_id = isic_id.numpy().decode("utf-8")
    f = _get_h5()
    img = f[isic_id][()]   # uint8 (H,W,3)
    return img


def load_and_preprocess(isic_id, label, training=False):
    img = tf.py_function(_load_image_py, [isic_id], tf.uint8)

    # BẮT BUỘC: báo shape để TF không bị mù
    img.set_shape([None, None, 3])

    img, label = preprocess(img, label, training=training)
    return img, label


def make_dataset(df_in, training=False):
    ids = df_in["isic_id"].astype(str).values
    labels = df_in["target"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((ids, labels))
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(lambda i, y: load_and_preprocess(i, y, training=training),
                num_parallel_calls=1)   # ⭐ HDF5-friendly

    ds = ds.batch(BATCH_SIZE).prefetch(1)
    return ds

train_ds = make_dataset(train_df, training=True)
val_ds   = make_dataset(val_df, training=False)

