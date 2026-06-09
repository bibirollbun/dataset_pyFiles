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
from glob import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical


class_names = [str(c) for c in np.unique(Y)]



# Optional augmentation library
try:
    import albumentations as A
    ALBU = True
except ImportError:
    ALBU = False


# CONFIG

INPUT_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TARGET_SIZE = (128,128)
SCALE_MODE = "[0,1]"
MAX_PER_CLASS = 100  # small for quick runs
USE_PCA = True
PCA_DIM = 300
EPOCHS = 25
BATCH_SIZE = 64
RANDOM_SEED = 42
IMG_SIZE = 128

dataset_path = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"  # ضع مسار الداتا هنا
image_paths = []
labels = []
class_names = sorted(os.listdir(dataset_path))
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        image_paths.append(img_path)
        labels.append(class_to_idx[class_name])


labels = np.array(labels)


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 1. Standard Image Processing

def stage1(img):
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32)
    return img / 255.0 if SCALE_MODE == "[0,1]" else (img / 127.5) - 1.0


# 2. Lighting Enhancement

def stage2(img):
    img_u8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
    img_cl = cv2.cvtColor(cv2.merge((cl,a,b)),cv2.COLOR_LAB2RGB)/255.0
    gray = cv2.cvtColor((img_cl*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gamma = np.interp(np.mean(gray), [50,200], [0.8,1.2])
    img_g = np.clip(img_cl ** gamma, 0, 1)
    yuv = cv2.cvtColor((img_g*255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    y_eq = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = y_eq
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)/255.0

# 3. Noise Reduction

def stage3(img):
    img_u8 = (img*255).astype(np.uint8)
    m = cv2.medianBlur(img_u8,3)
    b = cv2.bilateralFilter(m,9,75,75)
    g = cv2.GaussianBlur(b,(5,5),0)
    gray = cv2.cvtColor(g,cv2.COLOR_RGB2GRAY)
    _,mask = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    out = cv2.bitwise_and(g,g,mask=mask)
    return out.astype(np.float32)/255.0

# 4. Feature Enhancement

def stage4(img):
    img_u8 = (img*255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    sobx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    soby = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    sob = np.uint8(np.clip(np.sqrt(sobx**2+soby**2)/(np.sqrt((sobx**2+soby**2).max())+1e-8)*255,0,255))
    lbp = local_binary_pattern(gray,8,1,'uniform')
    lbp = ((lbp-lbp.min())/(lbp.max()-lbp.min()+1e-8)*255).astype(np.uint8)
    return edges, sob, lbp


# 5. Data Augmentation

if ALBU:
    aug = A.Compose([
        A.Rotate(limit=15,p=0.5), A.HorizontalFlip(p=0.5),
        A.RandomCrop(TARGET_SIZE[0]-20, TARGET_SIZE[1]-20,p=0.5),
        A.Perspective(scale=(0.05,0.1),p=0.4),
        A.GaussNoise(p=0.3),
        A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,p=0.5),
        A.PadIfNeeded(TARGET_SIZE[0],TARGET_SIZE[1],border_mode=cv2.BORDER_REFLECT,p=1.0)
    ])

    def stage5(img):
        img_u8 = (img*255).astype(np.uint8)
        img_t = aug(image=img_u8)['image']
        return img_t.astype(np.float32)/255.0
else:
    def stage5(img):
        return img  # no augmentation if albumentations not available

# Load sample images

sample_paths = glob(os.path.join(INPUT_DIR, 'c0', '*.jpg'))[:5]
samples = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in sample_paths]

# Visualize all stages
for i,orig in enumerate(samples):
    s1 = stage1(orig)
    s2 = stage2(s1)
    s3 = stage3(s2)
    e, sob, lb = stage4(s3)
    s5 = stage5(s3)

    fig,ax = plt.subplots(1,6,figsize=(20,4))
    ax[0].imshow(orig); ax[0].set_title('Original')
    ax[1].imshow(s1); ax[1].set_title('1. Std Proc')
    ax[2].imshow(s2); ax[2].set_title('2. Lighting')
    ax[3].imshow(s3); ax[3].set_title('3. Denoised')
    ax[4].imshow(e, cmap='gray'); ax[4].set_title('4. Edges')
    ax[5].imshow(s5); ax[5].set_title('5. Augmented')
    for a in ax: a.axis('off')
    plt.show()


# Prepare full dataset for DNN

fps, labs = [], []
for cls in sorted(os.listdir(INPUT_DIR)):
    p = os.path.join(INPUT_DIR, cls)
    files = glob(os.path.join(p, '*.jpg'))
    if MAX_PER_CLASS:
        files = files[:MAX_PER_CLASS]
    for f in files:
        fps.append(f)
        labs.append(cls)

X_feat, y , paths= [], [] ,[]
for f, l in tqdm(zip(fps, labs),total=len(fps),desc="Processing"):
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    img = stage1(img)
    img = stage2(img)
    img = stage3(img)
    e, sob, lbp = stage4(img)
    feat = np.stack([e.astype(np.float32)/255.0, lbp.astype(np.float32)/255.0], axis=-1)
    feat = cv2.resize(feat, TARGET_SIZE)
    X_feat.append(feat.flatten())
    y.append(l)
    paths.append(f)


X = np.array(X_feat)
y = np.array(y)
paths = np.array(paths) 

# PCA 

if USE_PCA and PCA_DIM < X.shape[1]:
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    X = pca.fit_transform(X)

lb = LabelBinarizer()
Y = lb.fit_transform(y)

# spliting data

X_train, X_temp, Y_train, Y_temp, paths_train, paths_temp = train_test_split(
    X, Y, paths, test_size=0.3, random_state=RANDOM_SEED, stratify=Y
)

X_val, X_test, Y_val, Y_test, paths_val, paths_test = train_test_split(
    X_temp, Y_temp, paths_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=Y_temp
)

# Build and train DNN

input_dim = X_train.shape[1]
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(Y_train.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Plot results
plt.plot(hist.history['accuracy'], label='train_acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title("Accuracy")
plt.show()

plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss")
plt.show()

y_pred = model.predict(X_test)  # probabilities
y_pred_classes = np.argmax(y_pred, axis=1)  # predicted labels
y_true_classes = np.argmax(Y_test, axis=1)  # true labels

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap=plt.cm.Reds, xticks_rotation=45)
plt.title("Confusion Matrix (test)")
plt.tight_layout()
plt.show()
def show_predictions_with_originals(model, X_data, Y_data, paths, class_names, num_images=12):
    idxs = np.random.choice(len(X_data), num_images, replace=False)
    X_sample = X_data[idxs]
    Y_true = Y_data[idxs]
    path_sample = [paths[int(i)] for i in idxs]  # safe indexing

    preds = model.predict(X_sample)
    Y_pred = np.argmax(preds, axis=1)

    # if Y_true is one-hot → convert
    if len(Y_true.shape) > 1 and Y_true.shape[1] > 1:
        Y_true = np.argmax(Y_true, axis=1)

    plt.figure(figsize=(15, 8))
    for i in range(num_images):
        plt.subplot(3, 4, i+1)

        # load original image
        img = cv2.imread(path_sample[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.axis("off")

        true_label = class_names[Y_true[i]]
        pred_label = class_names[Y_pred[i]]
        color = "green" if Y_true[i] == Y_pred[i] else "red"

        plt.title(f"P: {pred_label}\nT: {true_label}", color=color)

    plt.tight_layout()
    plt.show()


# =====================
# Run visualization
# =====================
show_predictions_with_originals(model, np.array(X_test), np.array(Y_test), paths_test, class_names, num_images=12)



# class_names = [str(c) for c in np.unique(Y)]

# Show predictions on the test set
#show_predictions(model, np.array(X_test), np.array(Y_test), class_names, num_images=12)


# Final evaluation
loss, acc = model.evaluate(X_val, Y_val, verbose=0)
print(f"Validation Accuracy: {acc:.4f}")



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------
# CONFIG
# ------------------------------
INPUT_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TARGET_SIZE = (128,128)
MAX_PER_CLASS = 100   # limit per class for quick runs
EPOCHS = 250
BATCH_SIZE = 32
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------
# Preprocessing Stages
# ------------------------------
def stage1(img):
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

def stage2(img):
    img_u8 = (img*255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
    img_cl = cv2.cvtColor(cv2.merge((cl,a,b)),cv2.COLOR_LAB2RGB)/255.0
    return img_cl

def stage3(img):
    img_u8 = (img*255).astype(np.uint8)
    g = cv2.GaussianBlur(img_u8,(5,5),0)
    return g.astype(np.float32)/255.0

def preprocess(img):
    img = stage1(img)
    img = stage2(img)
    img = stage3(img)
    return img

# ------------------------------
# Load Data
# ------------------------------
X, y, paths = [], [], []
class_names = sorted(os.listdir(INPUT_DIR))
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

for cls in class_names:
    cls_path = os.path.join(INPUT_DIR, cls)
    files = glob(os.path.join(cls_path, "*.jpg"))
    files = files[:MAX_PER_CLASS]
    for f in tqdm(files, desc=f"Class {cls}"):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        img = preprocess(img)
        X.append(img)
        y.append(class_to_idx[cls])
        paths.append(f)

X = np.array(X, dtype=np.float32)
y = np.array(y)
Y = to_categorical(y, num_classes=len(class_names))
paths = np.array(paths)

# Split dataset
X_train, X_temp, Y_train, Y_temp, paths_train, paths_temp = train_test_split(
    X, Y, paths, test_size=0.3, random_state=RANDOM_SEED, stratify=Y
)
X_val, X_test, Y_val, Y_test, paths_val, paths_test = train_test_split(
    X_temp, Y_temp, paths_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=Y_temp
)

# ------------------------------
# CNN Model (4 Conv Layers)
# ------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((3,3)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((3,3)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((3,3)),

   
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# Train
# ------------------------------
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                 epochs=EPOCHS, batch_size=BATCH_SIZE)

# ------------------------------
# Plot Training Curves
# ------------------------------
plt.plot(hist.history['accuracy'], label='train_acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title("Accuracy")
plt.show()

plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss")
plt.show()

# ------------------------------
# Evaluate & Confusion Matrix
# ------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(Y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix (test)")
plt.tight_layout()
plt.show()

# ------------------------------
# Show Predictions with Originals
# ------------------------------
def show_predictions_with_originals(model, X_data, Y_data, paths, class_names, num_images=12):
    idxs = np.random.choice(len(X_data), num_images, replace=False)
    X_sample = X_data[idxs]
    Y_true = Y_data[idxs]
    path_sample = [paths[int(i)] for i in idxs]

    preds = model.predict(X_sample)
    Y_pred = np.argmax(preds, axis=1)

    if len(Y_true.shape) > 1 and Y_true.shape[1] > 1:
        Y_true = np.argmax(Y_true, axis=1)

    plt.figure(figsize=(15, 8))
    for i in range(num_images):
        plt.subplot(3, 4, i+1)
        img = cv2.imread(path_sample[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")

        true_label = class_names[Y_true[i]]
        pred_label = class_names[Y_pred[i]]
        color = "green" if Y_true[i] == Y_pred[i] else "red"

        plt.title(f"P: {pred_label}\nT: {true_label}", color=color)

    plt.tight_layout()
    plt.show()

# Run visualization
show_predictions_with_originals(model, X_test, Y_test, paths_test, class_names, num_images=12)

# Final evaluation
loss, acc = model.evaluate(X_val, Y_val, verbose=0)
print(f"Validation Accuracy: {acc:.4f}")



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
import albumentations as A

# ------------------------------
# CONFIG
# ------------------------------
INPUT_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TARGET_SIZE = (128,128)
SCALE_MODE = "[0,1]"
MAX_PER_CLASS = 100
EPOCHS = 100
BATCH_SIZE = 64
RANDOM_SEED = 42
IMG_SIZE = 128

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------
# Stages
# ------------------------------
def stage1(img):
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32)
    return img / 255.0 if SCALE_MODE == "[0,1]" else (img / 127.5) - 1.0

def stage2(img):
    img_u8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
    img_cl = cv2.cvtColor(cv2.merge((cl,a,b)),cv2.COLOR_LAB2RGB)/255.0
    gray = cv2.cvtColor((img_cl*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gamma = np.interp(np.mean(gray), [50,200], [0.8,1.2])
    img_g = np.clip(img_cl ** gamma, 0, 1)
    yuv = cv2.cvtColor((img_g*255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    y_eq = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = y_eq
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)/255.0

def stage3(img):
    img_u8 = (img*255).astype(np.uint8)
    m = cv2.medianBlur(img_u8,3)
    b = cv2.bilateralFilter(m,9,75,75)
    g = cv2.GaussianBlur(b,(5,5),0)
    gray = cv2.cvtColor(g,cv2.COLOR_RGB2GRAY)
    _,mask = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    out = cv2.bitwise_and(g,g,mask=mask)
    return out.astype(np.float32)/255.0

def stage4(img):
    img_u8 = (img*255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    sobx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    soby = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    sob = np.uint8(np.clip(np.sqrt(sobx**2+soby**2)/(np.sqrt((sobx**2+soby**2).max())+1e-8)*255,0,255))
    lbp = local_binary_pattern(gray,8,1,'uniform')
    lbp = ((lbp-lbp.min())/(lbp.max()-lbp.min()+1e-8)*255).astype(np.uint8)
    return edges, sob, lbp

# Augmentation
aug = A.Compose([
    A.Rotate(limit=15,p=0.5), A.HorizontalFlip(p=0.5),
    A.RandomCrop(TARGET_SIZE[0]-20, TARGET_SIZE[1]-20,p=0.5),
    A.Perspective(scale=(0.05,0.1),p=0.4),
    A.GaussNoise(p=0.3),
    A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,p=0.5),
    A.PadIfNeeded(TARGET_SIZE[0],TARGET_SIZE[1],border_mode=cv2.BORDER_REFLECT,p=1.0)
])

def stage5(img):
    img_u8 = (img*255).astype(np.uint8)
    img_t = aug(image=img_u8)['image']
    return img_t.astype(np.float32)/255.0

# ------------------------------
# LOAD DATA
# ------------------------------
fps, labs = [], []
class_names = sorted(os.listdir(INPUT_DIR))
for cls in class_names:
    p = os.path.join(INPUT_DIR, cls)
    files = glob(os.path.join(p, '*.jpg'))
    if MAX_PER_CLASS:
        files = files[:MAX_PER_CLASS]
    for f in files:
        fps.append(f)
        labs.append(cls)

X_data, y_data, paths = [], [], []
for f, l in tqdm(zip(fps, labs), total=len(fps)):
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    img = stage1(img)
    img = stage2(img)
    img = stage3(img)
    e, sob, lbp = stage4(img)
    feat = np.stack([e, sob, lbp], axis=-1).astype(np.float32) / 255.0  # 3 channels
    feat = cv2.resize(feat, TARGET_SIZE)
    X_data.append(feat)
    y_data.append(l)
    paths.append(f)

X_data = np.array(X_data)
y_data = np.array(y_data)

lb = LabelBinarizer()
Y = lb.fit_transform(y_data)

# Train/Val/Test split 70/15/15
X_train, X_temp, Y_train, Y_temp, paths_train, paths_temp = train_test_split(
    X_data, Y, paths, test_size=0.3, random_state=RANDOM_SEED, stratify=Y
)
X_val, X_test, Y_val, Y_test, paths_val, paths_test = train_test_split(
    X_temp, Y_temp, paths_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=Y_temp
)

# ------------------------------
# MobileNetV2 MODEL
# ------------------------------
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # freeze backbone first

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=True)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(len(lb.classes_), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# TRAIN
# ------------------------------
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                 epochs=EPOCHS, batch_size=BATCH_SIZE)

# ------------------------------
# EVALUATION
# ------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(Y_test, axis=1)

plt.plot(hist.history['accuracy'], label='train_acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title("Accuracy")
plt.show()

plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss")
plt.show()

# print(classification_report(y_true_classes, y_pred_classes, target_names=lb.classes_))

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap=plt.cm.Greens, xticks_rotation=45)
plt.show()

def show_predictions_with_originals(model, X_data, Y_data, paths, class_names, num_images=12):
    idxs = np.random.choice(len(X_data), num_images, replace=False)
    X_sample = X_data[idxs]
    Y_true = Y_data[idxs]
    path_sample = [paths[int(i)] for i in idxs]

    preds = model.predict(X_sample)
    Y_pred = np.argmax(preds, axis=1)

    if len(Y_true.shape) > 1 and Y_true.shape[1] > 1:
        Y_true = np.argmax(Y_true, axis=1)

    plt.figure(figsize=(15, 8))
    for i in range(num_images):
        plt.subplot(3, 4, i+1)
        img = cv2.imread(path_sample[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")

        true_label = class_names[Y_true[i]]
        pred_label = class_names[Y_pred[i]]
        color = "green" if Y_true[i] == Y_pred[i] else "red"

        plt.title(f"P: {pred_label}\nT: {true_label}", color=color)

    plt.tight_layout()
    plt.show()

# Run visualization
show_predictions_with_originals(model, X_test, Y_test, paths_test, class_names, num_images=12)

# Final evaluation
loss, acc = model.evaluate(X_val, Y_val, verbose=0)
print(f"Validation Accuracy: {acc:.4f}")




import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# Parameters
# =========================
IMG_SIZE = 128
num_classes = 10
DATASET_PATH = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
radius = 3
n_points = 8 * radius

# =========================
# Feature extraction
# =========================
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Edge detection
    edges = cv2.Canny(img, 100, 200)

    # Local Binary Pattern
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    lbp = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))

    # Stack as 2 channels
    stacked = np.stack([edges, lbp], axis=-1)
    return stacked

# =========================
# Load dataset
# =========================
X = []
y = []
paths = []
class_names = sorted(os.listdir(DATASET_PATH))

for label, class_name in enumerate(class_names):
    img_paths = glob(os.path.join(DATASET_PATH, class_name, "*.jpg"))
    for path in tqdm(img_paths, desc=f"Loading {class_name}"):
        X.append(preprocess_image(path))
        y.append(label)
        paths.append(path)

X = np.array(X, dtype=np.uint8)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)

# Train/test split
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, paths, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Augmentation
# =========================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    fill_mode="nearest"
)
datagen.fit(X_train)

# =========================
# Build ResNet50 model
# =========================
base_model = ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 2),  # 2 channels (edges + lbp)
    include_top=False,
    weights=None
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Train (reduced epochs)
# =========================
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs = 25  
)

# =========================
# Evaluate
# =========================
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss")
plt.show()

# =========================
# Confusion Matrix
# =========================
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Greens, xticks_rotation=45)
plt.show()

# =========================
# Show Predictions with Originals
# =========================
def show_predictions_with_originals(model, X_data, Y_data, paths, class_names, num_images=12):
    idxs = np.random.choice(len(X_data), num_images, replace=False)
    X_sample = X_data[idxs]
    Y_true = Y_data[idxs]
    path_sample = [paths[int(i)] for i in idxs]

    preds = model.predict(X_sample)
    Y_pred = np.argmax(preds, axis=1)

    plt.figure(figsize=(15, 8))
    for i in range(num_images):
        plt.subplot(3, 4, i+1)
        img = cv2.imread(path_sample[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")

        true_label = class_names[Y_true[i]]
        pred_label = class_names[Y_pred[i]]
        color = "green" if Y_true[i] == Y_pred[i] else "red"

        plt.title(f"P: {pred_label}\nT: {true_label}", color=color)

    plt.tight_layout()
    plt.show()

# Run visualization
show_predictions_with_originals(model, X_test, y_test, paths_test, class_names, num_images=12)



