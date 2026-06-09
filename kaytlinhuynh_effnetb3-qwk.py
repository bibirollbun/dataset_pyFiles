!pip install albumentations


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print('✅')


# Standard dependencies
import cv2
import time
import scipy as sp
import numpy as np
import random as rn
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

# Machine Learning
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Rotate, Resize, RandomGamma

import tensorflow as tf
import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import backend as K
from tensorflow.keras.activations import elu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
print('✅')


# Path specifications
KAGGLE_DIR = '../input/aptos2019-blindness-detection/'
df = pd.read_csv(os.path.join(KAGGLE_DIR + "train.csv"))
#test_df_path = KAGGLE_DIR + 'test.csv'
img_dir = KAGGLE_DIR + "train_images/"
df['image_path'] = df['id_code'].apply(lambda x: os.path.join(img_dir, f"{x}.png"))
#test_img_path = KAGGLE_DIR + 'test_images/'
#SAVED_MODEL_NAME = '/kaggle/working/efficientnetb3_best_kappa_model.keras'

save_img_path = '/kaggle/working/images'
os.makedirs(save_img_path, exist_ok=True)
# aug_img_path = "/kaggle/working/aug_images/"
# os.makedirs(aug_img_path, exist_ok=True)

print('✅')


for i in range(5):
    df[f"ovr_{i}"] = df['diagnosis'].apply(lambda x: 1 if x == i else 0)
print('✅')


from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Bước 1: Chia train và val_test (train khoảng 70%)
train_df, val_test_df = train_test_split(
    df,
    test_size=0.30,  # val + test chiếm 30%
    stratify=df['diagnosis'],
    shuffle=True,
    random_state=42
)

# Bước 2: Chia val và test từ val_test_df theo tỉ lệ 1:2 (val = 10%, test = 20%)
val_df, test_df = train_test_split(
    val_test_df,
    test_size=2/3,  # test = 2 phần, val = 1 phần → test ~20%, val ~10%
    stratify=val_test_df['diagnosis'],
    shuffle=True,
    random_state=42
)

max_count = train_df["diagnosis"].value_counts().max()
oversampled_dfs = []

for i in range(5):
    cls_df = train_df[train_df["diagnosis"] == i]
    if len(cls_df) < max_count:
        cls_df = resample(cls_df, replace=True, n_samples=max_count, random_state=42)
    oversampled_dfs.append(cls_df)

oversampled_train_df = pd.concat(oversampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
print('✅')


print("Image IDs and Labels (TRAIN)")
# Add extension to id_code
#train_val_df['id_code'] = train_val_df['id_code'] + ".png"
print(f"Training images: {train_df.shape[0]}")
display(train_df.head())

print("Image IDs (TEST)")
# Add extension to id_code
#test_df['id_code'] = test_df['id_code'] + ".png"
print(f"Testing Images: {test_df.shape[0]}")
display(test_df.head())
print('✅')


import matplotlib.pyplot as plt

# Đếm số lượng ảnh từng lớp trước và sau
before_counts = train_df['diagnosis'].value_counts().sort_index()
after_counts = oversampled_train_df['diagnosis'].value_counts().sort_index()

# Vẽ biểu đồ cột trên cùng một hàng
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Trước khi oversample
axes[0].bar(before_counts.index.astype(str), before_counts.values, color='skyblue')
axes[0].set_title("Before Oversampling")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Image Count")

# Thêm số trên đỉnh cột
for i, v in enumerate(before_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', va='bottom', fontsize=10)

# Sau khi oversample
axes[1].bar(after_counts.index.astype(str), after_counts.values, color='salmon')
axes[1].set_title("After Oversampling")
axes[1].set_xlabel("Class")

# Thêm số trên đỉnh cột
for i, v in enumerate(after_counts.values):
    axes[1].text(i, v + 20, str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig('/kaggle/working/comparison.png')


# Specify image size
IMG_WIDTH = 224
IMG_HEIGHT = 224
CHANNELS = 3
print('✅')


def check_missing(df, name):
    missing = df[~df['image_path'].apply(os.path.exists)]
    print(f"{name}: {len(missing)} ảnh bị thiếu.")
    if not missing.empty:
        print(missing[['id_code', 'image_path']].head(5))
    return missing

missing_train = check_missing(train_df, "Train")
missing_val = check_missing(val_df, "Validation")
missing_oversample = check_missing(oversampled_train_df, "Oversampled Train")
print('✅')


from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
from albumentations import (
    Compose, Resize, HorizontalFlip, CLAHE,
    RandomBrightnessContrast, Rotate, Normalize
)
from albumentations.core.composition import OneOf

# Augment mạnh cho training
train_transform = Compose([
    Resize(224, 224),
    HorizontalFlip(p=0.5),
    CLAHE(p=0.3),
    RandomBrightnessContrast(p=0.5),
    Rotate(limit=10, p=0.5),
    Normalize()
])

# Augment nhẹ hoặc chỉ resize cho validation/test
val_transform = Compose([
    Resize(224, 224),
    Normalize()
])
print('✅')


from tensorflow.keras.utils import Sequence
import cv2
import numpy as np

class DR_OvR_Generator(Sequence):
    def __init__(self, df, batch_size=32, transform=None, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        X = np.empty((len(batch_df), 224, 224, 3), dtype=np.float32)
        y = np.empty((len(batch_df), 5), dtype=np.float32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            path = row['image_path']
            image = cv2.imread(path)

            if image is None:
                print(f"❗ Không thể load ảnh: {path}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)

            elif len(image.shape) != 3 or image.shape[2] != 3:
                print(f"❗ Ảnh không đúng định dạng RGB: {path}, shape: {image.shape}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)

            else:
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"❗ Lỗi chuyển màu RGB ảnh {path}: {e}")
                    image = np.zeros((224, 224, 3), dtype=np.uint8)

            if self.transform:
                try:
                    image = self.transform(image=image)['image']
                except Exception as e:
                    print(f"❗ Lỗi augment ảnh {path}: {e}")
                    image = np.zeros((224, 224, 3), dtype=np.uint8)

            X[i] = image
            y[i] = [row[f"ovr_{j}"] for j in range(5)]

        return X, {f"ovr_{j}": y[:, j] for j in range(5)}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
print('✅')


from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_resnet_ovr(input_shape=(224,224,3), fine_tune_at=60):
    base = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-fine_tune_at:]:
        layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = [layers.Dense(1, activation='sigmoid', name=f"ovr_{i}")(x) for i in range(5)]
    return models.Model(inputs=base.input, outputs=outputs)


from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy, axis=-1)
    return focal_loss


losses = {f"ovr_{i}": binary_focal_loss() for i in range(5)}
metrics = {f"ovr_{i}": 'AUC' for i in range(5)}  # ✅ mỗi output cần metric riêng

model = build_resnet_ovr()
model.compile(optimizer='adam', loss=losses, metrics=metrics)

model.summary()


# Prior terms for QWK evaluation
orig_counts = train_df['diagnosis'].value_counts().sort_index().to_numpy().astype(np.float32)
orig_prior = orig_counts / orig_counts.sum()
os_counts = oversampled_train_df['diagnosis'].value_counts().sort_index().to_numpy().astype(np.float32)
os_prior = os_counts / os_counts.sum()
ratio_prior = os_prior / orig_prior

def prior_correct_and_normalize(p_ovr, ratio):
    p = p_ovr / ratio
    p = p / p.sum(axis=1, keepdims=True)
    return p


# QWK callback on validation
class QWKCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, y_true, ratio_prior):
        super().__init__()
        self.val_gen = val_gen
        self.y_true = y_true
        self.ratio_prior = ratio_prior
        self.best_qwk = -1.0

    def on_epoch_end(self, epoch, logs=None):
        p_list = self.model.predict(self.val_gen, verbose=0)
        p_val = np.concatenate(p_list, axis=1)
        p_val = np.clip(p_val, 1e-6, 1-1e-6)
        p_val = prior_correct_and_normalize(p_val, self.ratio_prior)
        y_hat = p_val.argmax(axis=1)
        qwk = cohen_kappa_score(self.y_true, y_hat, weights='quadratic')
        print(f"\nval_qwk: {qwk:.4f}")
        logs = logs or {}
        logs['val_qwk'] = qwk
        if qwk > self.best_qwk:
            self.best_qwk = qwk

qwk_cb = QWKCallback(
    val_gen=DR_OvR_Generator(val_df, 32, val_transform, shuffle=False),
    y_true=val_df["diagnosis"].values,
    ratio_prior=ratio_prior
)


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-4),
]


import os

missing_files = []
for path in df['image_path']:
    if not os.path.exists(path):
        missing_files.append(path)

print(f"Tổng số ảnh bị thiếu: {len(missing_files)}")
print("Ví dụ các ảnh thiếu:", missing_files[:5])



train_gen = DR_OvR_Generator(oversampled_train_df, batch_size=32, transform=train_transform)
val_gen = DR_OvR_Generator(val_df, batch_size=32, transform=val_transform, shuffle=False)


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)


import json
import os

# Tạo thư mục output nếu chưa tồn tại
output_dir = '/kaggle/working/output/ResNet'
os.makedirs(output_dir, exist_ok=True)

# Lưu history vào file JSON
with open(output_dir + 'history_1.json', 'w') as f:
    json.dump(history.history, f)


# Lưu trọng số mô hình
model.save_weights(os.path.join(output+dir,'/my_model.weights.h5'))


import json
import matplotlib.pyplot as plt

# Đọc history từ file JSON
with open(output_dir + 'history_1.json', 'r') as f:
    history = json.load(f)


import numpy as np
best_epoch = np.argmin(history['val_loss']) + 1
print(f"Best epoch selected by EarlyStopping: {best_epoch}")



from sklearn.metrics import cohen_kappa_score, confusion_matrix

y_true = val_df["diagnosis"].values
y_pred_probs = model.predict(DR_OvR_Generator(val_df, 32, val_transform, shuffle=False))
y_pred_probs = np.concatenate(y_pred_probs, axis=1)
y_pred = np.argmax(y_pred_probs, axis=1)

qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f"QWK = {qwk:.4f}")
print(confusion_matrix(y_true, y_pred))



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def make_gradcam_heatmap(img_array, model, class_index, conv_layer_name="conv5_block3_out"):
    """
    Args:
        img_array: shape (1, H, W, 3)
        model: trained model with multiple outputs (OvR)
        class_index: int in [0, 4] → which OvR output to use
        conv_layer_name: usually "top_conv" for EfficientNet
    Returns:
        heatmap: np.array (H, W)
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.outputs[class_index]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weighted sum of channels
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def plot_gradcam(img_path, model, class_index, conv_layer_name="conv5_block3_out", alpha=0.4):
    """
    Args:
        img_path: path to original .png image
        class_index: OvR index (0-4)
    """
    # Load & preprocess ảnh
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    input_array = tf.keras.applications.efficientnet.preprocess_input(img_resized.astype(np.float32))
    input_array = np.expand_dims(input_array, axis=0)

    # GradCAM
    heatmap = make_gradcam_heatmap(input_array, model, class_index, conv_layer_name)

    # Resize heatmap về đúng size ảnh gốc
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"GradCAM - OvR_{class_index}")
    plt.axis(False)
    plt.tight_layout()
    plt.show()


import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

img_path = test_df.iloc[0]['image_path']

# Load and preprocess the image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224))  # ResNet50 dùng 224x224
img_array = preprocess_input(img_resized.astype(np.float32))

# Dự đoán
y_pred = model.predict(np.expand_dims(img_array, axis=0))  # trả về list 5 nhánh sigmoid

# Plot Grad-CAM (giữ nguyên class_index nếu muốn highlight nhánh số 2)
plot_gradcam(img_path, model, class_index=2)


import pandas as pd

# Lấy xác suất trên val
p_list = model.predict(DR_OvR_Generator(val_df, 32, val_transform, shuffle=False),
                       workers=4, use_multiprocessing=True, verbose=0)
p_val = np.concatenate(p_list, axis=1)  # shape (N,5)
p_val = np.clip(p_val, 1e-6, 1-1e-6)
y_true = val_df["diagnosis"].values

# Các hàm suy luận và chỉ số
def pred_argmax_raw(p): return p.argmax(axis=1)
def pred_argmax_prior(p, ratio): return prior_correct_and_normalize(p, ratio).argmax(axis=1)

def pred_expected_grade_round(p, ratio=None):
    if ratio is not None:
        p = prior_correct_and_normalize(p, ratio)
    ks = np.arange(p.shape[1], dtype=np.float32)
    score = (p * ks).sum(axis=1)
    return np.rint(score).astype(int).clip(0, p.shape[1]-1)

def off_by_one_acc(y, yhat): return np.mean(np.abs(y - yhat) <= 1)
def label_mae(y, yhat): return np.mean(np.abs(y - yhat))

def weighted_error_qwk_style(y, yhat, K=5):
    from sklearn.metrics import confusion_matrix
    W = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            W[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)
    cm = confusion_matrix(y, yhat, labels=np.arange(K)).astype(np.float32)
    return (W * cm).sum() / cm.sum()

methods = {
    "argmax_raw":            lambda p: pred_argmax_raw(p),
    "argmax_prior":          lambda p: pred_argmax_prior(p, ratio_prior),
    "exp_grade_round":       lambda p: pred_expected_grade_round(p, ratio=None),
    "exp_grade_round_prior": lambda p: pred_expected_grade_round(p, ratio=ratio_prior),
}

rows = []
for name, fn in methods.items():
    y_hat = fn(p_val)
    rows.append({
        "Method": name,
        "QWK": cohen_kappa_score(y_true, y_hat, weights='quadratic'),
        "Accuracy": (y_true == y_hat).mean(),
        "OffByOne": off_by_one_acc(y_true, y_hat),
        "MAE": label_mae(y_true, y_hat),
        "WeightedError": weighted_error_qwk_style(y_true, y_hat, K=5)
    })

df_summary = pd.DataFrame(rows).sort_values("QWK", ascending=False)
print(df_summary)

# Lưu bảng
os.makedirs(output_dir, exist_ok=True)
summary_path = os.path.join(output_dir, "summary_metrics_qwk.csv")
df_summary.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

