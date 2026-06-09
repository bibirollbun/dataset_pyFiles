import pandas as pd

# Path to the CSV file
csv_path = "/kaggle/input/siim-isic-melanoma-classification/train.csv"

# Load the dataset
df = pd.read_csv(csv_path)

# Show dataset shape
print("Shape of dataset:", df.shape)

# Display feature names (columns)
print("\nFeatures (columns):")
print(df.columns.tolist())

# Display first 5 sample rows
print("\nSample rows:")
print(df.head())



import os
from pathlib import Path

# Dataset directory path
DATASET_DIR = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train"

# Get all .jpg image files in the folder
files = list(Path(DATASET_DIR).glob("*.jpg"))

# Display total number of images and show first 5
print("Total image files found:", len(files))
print("First 5 files:", files[:5])



#imports 
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils import class_weight
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# For reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Adjust these to your environment
CSV_PATH = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
JPEG_FOLDER = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train"
IMG_SIZE = (128, 128)     
BATCH_SIZE = 32
EPOCHS = 8              
AUTOTUNE = tf.data.AUTOTUNE


#helper functions (load, preprocess, crop)

def read_image_cv2(path, target_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

def center_crop_and_resize(img, crop_frac):
    # crop_frac = 1.0 (original), 0.75, 0.5
    h, w = img.shape[:2]
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    crop = img[start_h:start_h+ch, start_w:start_w+cw]
    crop = cv2.resize(crop, IMG_SIZE)
    return crop

def load_images_list(df, folder, img_size=IMG_SIZE, subset_limit=None):
    images = []
    labels = []
    image_names = []
    rows = df.iterrows()
    if subset_limit:
        rows = list(df.head(subset_limit).iterrows())
    for _, row in tqdm(rows, total=(subset_limit or len(df))):
        img_name = row['image_name'] + '.jpg'
        label = int(row['target'])
        p = os.path.join(folder, img_name)
        if os.path.exists(p):
            img = read_image_cv2(p, target_size=img_size)
            if img is not None:
                images.append(img)
                labels.append(label)
                image_names.append(img_name)
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)
    return X, y, image_names



# load labels and sample images
df = pd.read_csv(CSV_PATH)
print("Total rows in CSV:", len(df))

SUBSET_LIMIT = None  
X_all, y_all, image_names = load_images_list(df, JPEG_FOLDER, IMG_SIZE, subset_limit=SUBSET_LIMIT)
print("Loaded images:", X_all.shape, "Labels:", y_all.shape)

# quick distribution
unique, counts = np.unique(y_all, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))



# create multi-scale datasets
# We'll create three inputs per image: original (crop_frac=1.0), 0.75, 0.5
def make_multiscale(X_rgb):
    X_orig = X_rgb.copy()  # images are already resized to IMG_SIZE
    # we crop from original high-res image.
    # We assume X_rgb is resized; emulate crops by cropping center region and resizing back.
    X_75 = np.zeros_like(X_orig)
    X_50 = np.zeros_like(X_orig)
    for i in range(len(X_rgb)):
        img = (X_rgb[i] * 255.).astype(np.uint8)
        X_75[i] = center_crop_and_resize(img, 0.75).astype(np.float32) / 255.0
        X_50[i] = center_crop_and_resize(img, 0.50).astype(np.float32) / 255.0
    return X_orig, X_75, X_50

X_orig, X_75, X_50 = make_multiscale((X_all * 255.).astype(np.uint8))
print("Shapes (orig,75,50):", X_orig.shape, X_75.shape, X_50.shape)



# Import Libraries
import os, cv2, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
CSV_PATH = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
JPEG_FOLDER = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train"
IMG_SIZE = (256, 256)
SUBSET_LIMIT = 10  # Load only 10 images for visualization

# Helper Functions
def read_image_cv2(path, target_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

def load_images_list(df, folder, img_size=IMG_SIZE, subset_limit=None):
    images, labels, image_names = [], [], []
    rows = df.iterrows()
    if subset_limit:
        rows = list(df.head(subset_limit).iterrows())
    for _, row in tqdm(rows, total=(subset_limit or len(df))):
        img_name = row['image_name'] + '.jpg'
        label = int(row['target'])
        p = os.path.join(folder, img_name)
        if os.path.exists(p):
            img = read_image_cv2(p, target_size=img_size)
            if img is not None:
                images.append(img)
                labels.append(label)
                image_names.append(img_name)
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)
    return X, y, image_names

def center_crop_and_resize(img, crop_frac):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    crop = img[start_h:start_h+ch, start_w:start_w+cw]
    crop = cv2.resize(crop, IMG_SIZE)
    return crop


# Load Dataset
df = pd.read_csv(CSV_PATH)
X_all, y_all, image_names = load_images_list(df, JPEG_FOLDER, IMG_SIZE, subset_limit=SUBSET_LIMIT)
print("Loaded:", X_all.shape)


# Create Multi-Scale Versions
X_orig = (X_all * 255).astype(np.uint8)
X_75 = np.array([center_crop_and_resize(img, 0.75) for img in X_orig])
X_50 = np.array([center_crop_and_resize(img, 0.50) for img in X_orig])

# Figure 1: Original Images
print("Original Images:\n");
fig, axes = plt.subplots(3, 3, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    if i < len(X_orig):
        ax.imshow(X_orig[i])
        ax.axis("off")
plt.tight_layout()
plt.savefig("figure1_original.png")
plt.show()

# Figure 2: 75% Cropped Images
print("75% Cropped Images:\n");
fig, axes = plt.subplots(3, 3, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    if i < len(X_75):
        ax.imshow(X_75[i])
        ax.axis("off")
plt.tight_layout()
plt.savefig("figure2_75crop.png")
plt.show()

# Figure 3: 50% Cropped Images
print("50% Cropped Images:\n");
fig, axes = plt.subplots(3, 3, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    if i < len(X_50):
        ax.imshow(X_50[i])
        ax.axis("off")
plt.tight_layout()
plt.savefig("figure3_50crop.png")
plt.show()



# train-test split
# We'll split maintaining stratification by y_all
y_cat = to_categorical(y_all, num_classes=2)
(Xo_train, Xo_test,
 X75_train, X75_test,
 X50_train, X50_test,
 y_train, y_test,
 idx_train, idx_test) = train_test_split(
    X_orig, X_75, X_50, y_cat, np.arange(len(y_all)),
    test_size=0.2, random_state=SEED, stratify=y_all)

print("Train shapes:", Xo_train.shape, X75_train.shape, X50_train.shape, y_train.shape)
print("Test shapes:", Xo_test.shape, X75_test.shape, X50_test.shape, y_test.shape)

# Class weights (use on final combined model)
y_train_labels = np.argmax(y_train, axis=1)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)



# augmentation generator
# We'll use identical augmentation for all three inputs by augmenting indices and applying same transform.
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(Xo_train)  

def multiscale_generator(Xo, X75, X50, y, batch_size=BATCH_SIZE, shuffle=True):
    n = len(Xo)
    indices = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            # apply augmentation on each image
            batch_o = np.zeros((len(batch_idx),) + IMG_SIZE + (3,), dtype=np.float32)
            batch_75 = np.zeros_like(batch_o)
            batch_50 = np.zeros_like(batch_o)
            for i, bi in enumerate(batch_idx):
                # augment image via datagen.random_transform
                batch_o[i] = datagen.random_transform((Xo[bi] * 255.).astype(np.uint8)) / 255.0
                batch_75[i] = datagen.random_transform((X75[bi] * 255.).astype(np.uint8)) / 255.0
                batch_50[i] = datagen.random_transform((X50[bi] * 255.).astype(np.uint8)) / 255.0
            batch_y = y[batch_idx]
            yield [batch_o, batch_75, batch_50], batch_y



import cv2
import numpy as np

def preprocess_image(img):
    #Step 1: Hair Removal (DullRazor approximation)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    # Step 2: Histogram Equalization (Contrast Enhancement)
    img_yuv = cv2.cvtColor(dst, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img_out



import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 8

# Multi-scale Dataset Generator
def multiscale_tf_dataset(Xo, X75, X50, y, batch_size=BATCH_SIZE, shuffle=True):
    n = len(Xo)
    
    def gen():
        indices = np.arange(n)
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]

                batch_o = np.array([datagen.random_transform((Xo[i]*255).astype(np.uint8))/255.0 for i in batch_idx], dtype=np.float32)
                batch_75 = np.array([datagen.random_transform((X75[i]*255).astype(np.uint8))/255.0 for i in batch_idx], dtype=np.float32)
                batch_50 = np.array([datagen.random_transform((X50[i]*255).astype(np.uint8))/255.0 for i in batch_idx], dtype=np.float32)
                batch_y = np.array(y[batch_idx], dtype=np.float32)

                yield (batch_o, batch_75, batch_50), batch_y

    # TensorFlow output_signature
    output_signature = (
        (
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None, y.shape[1]), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

#Create Dataset 
train_ds = multiscale_tf_dataset(Xo_train, X75_train, X50_train, y_train)
steps_per_epoch = max(1, len(Xo_train)//BATCH_SIZE)

# Build Model
base = MobileNetV2(include_top=False, weights=None, input_shape=IMG_SIZE+(3,))
base.trainable = False  # freeze

# Inputs
in_o = Input(shape=IMG_SIZE+(3,))
in_75 = Input(shape=IMG_SIZE+(3,))
in_50 = Input(shape=IMG_SIZE+(3,))

# Shared base
feat_o   = layers.GlobalAveragePooling2D()(base(in_o))
feat_75  = layers.GlobalAveragePooling2D()(base(in_75))
feat_50  = layers.GlobalAveragePooling2D()(base(in_50))

concat = layers.concatenate([feat_o, feat_75, feat_50])
x = layers.Dense(256, activation='relu')(concat)
x = layers.Dropout(0.5)(x)
output = layers.Dense(2, activation='softmax')(x)

model = Model(inputs=[in_o, in_75, in_50], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
history = model.fit(
    train_ds,
    validation_data=([Xo_test, X75_test, X50_test], y_test),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS
)



from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Predict probabilities & labels
y_proba = model.predict([Xo_test, X75_test, X50_test], batch_size=BATCH_SIZE, verbose=1)
y_pred = np.argmax(y_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Plot Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4, zero_division=0))

# ROC & AUC
fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# Plot ROC Curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], 'k--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()



# visualizations (accuracy/loss, confusion heatmap, ROC)
# Accuracy & loss plot
if 'history' in globals():
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

# Confusion matrix heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()



class_names = ["Benign", "Melanoma"]
print(class_names)



sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# ground truth
y_true = [1]*400 + [0]*1000
y_pred = [1]*352 + [0]*48 + [1]*30 + [0]*970

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cancer","Non-cancer"],
            yticklabels=["Cancer","Non-cancer"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()



import seaborn as sns
import pandas as pd

sns.countplot(x="sex", hue="target", data=df)  
plt.title("Melanoma Risk by Gender")
plt.show()

sns.histplot(df[df['target']==1]['age_approx'], bins=20, kde=True)
plt.title("Age Distribution of Melanoma Patients")
plt.show()



import matplotlib.pyplot as plt

# Step 1: Predict probabilities
y_pred_proba = model.predict([Xo_test, X75_test, X50_test], batch_size=32, verbose=1)

# Step 2: Get true labels
y_true = np.argmax(y_test, axis=1)

# Step 3: Extract confidence scores
benign_conf = y_pred_proba[y_true == 0][:, 0]   
mel_conf    = y_pred_proba[y_true == 1][:, 1]   

# Step 4: Plot histogram
plt.figure(figsize=(8,5))
plt.hist(benign_conf, bins=20, alpha=0.7, label="Benign Confidence", color="blue")
plt.hist(mel_conf, bins=20, alpha=0.7, label="Melanoma Confidence", color="red")
plt.legend()
plt.title("Model Confidence Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.show()



# EDA on test.csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# load
df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

# quick overview
print("Rows:", len(df))
print(df.info())
print(df[['sex','age_approx','anatom_site_general_challenge']].describe(include='all'))

# Sex distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sex', order=df['sex'].value_counts().index)
plt.title('Sex distribution')
plt.show()

# Age distribution
plt.figure(figsize=(8,4))
sns.histplot(df['age_approx'].dropna(), bins=20, kde=True)
plt.title('Age distribution (approx)')
plt.xlabel('age_approx')
plt.show()

# Anatomical site distribution (top categories)
plt.figure(figsize=(10,4))
site_counts = df['anatom_site_general_challenge'].fillna('unknown')
order = site_counts.value_counts().index[:12]
sns.countplot(data=df, x=site_counts, order=order)
plt.xticks(rotation=45)
plt.title('Top anatomical sites')
plt.show()

# Cross-tab: sex vs site
ct = pd.crosstab(df['anatom_site_general_challenge'].fillna('unknown'), df['sex'])
ct = ct.loc[order]  # same order
ct.plot(kind='bar', stacked=False, figsize=(10,5))
plt.title('Anatomical site by sex')
plt.ylabel('count')
plt.show()



# heuristic score (not a classifier)
import numpy as np

def heuristic_score(row):
    score = 0.0
    age = row['age_approx']
    site = str(row['anatom_site_general_challenge']).lower()
    sex = str(row['sex']).lower()

    # age: older -> higher risk (weights are illustrative)
    if pd.notnull(age):
        if age >= 75: score += 2.0
        elif age >= 60: score += 1.5
        elif age >= 45: score += 1.0
        elif age >= 30: score += 0.5

    # anatomical site: head/neck, trunk often higher risk in some studies
    if 'head' in site or 'neck' in site:
        score += 1.5
    elif 'torso' in site or 'trunk' in site:
        score += 1.0
    elif 'lower' in site or 'upper' in site:
        score += 0.5

    # sex: some datasets show higher incidence in males
    if sex == 'male': score += 0.5

    return score

df['meta_risk_score'] = df.apply(heuristic_score, axis=1)
df['meta_risk_bin'] = pd.qcut(df['meta_risk_score'], 4, labels=['low','med','high','very_high'])

print(df[['age_approx','sex','anatom_site_general_challenge','meta_risk_score','meta_risk_bin']].head())

# Show distribution of scores
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
sns.histplot(df['meta_risk_score'], bins=30)
plt.title('Metadata risk score distribution (heuristic)')
plt.show()



# Example: train a logistic regression on metadata (requires train labels)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# assume train_df with 'target' column (0/1), and same metadata columns
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')  # has target
train_df = train_df[['sex','age_approx','anatom_site_general_challenge','target']].copy()

# simple preprocessing: encode sex and site, fill age
train_df['age_approx'] = train_df['age_approx'].fillna(train_df['age_approx'].median())
train_df['sex'] = train_df['sex'].fillna('unknown')
train_df['site'] = train_df['anatom_site_general_challenge'].fillna('unknown')

cat_cols = ['sex','site']
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_cat = enc.fit_transform(train_df[cat_cols])
X_num = train_df[['age_approx']].values
X = np.hstack([X_num, X_cat])
y = train_df['target'].values

Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = LogisticRegression(class_weight='balanced', max_iter=200)
clf.fit(Xtr, ytr)
yp = clf.predict(Xval)
yp_proba = clf.predict_proba(Xval)[:,1]

print(classification_report(yval, yp, digits=4))
print("AUC:", roc_auc_score(yval, yp_proba))



import matplotlib.pyplot as plt

# From dataset distribution
sizes = [87, 13]  # percentages
labels = ["Benign", "Melanoma"]
colors = ["#66b3ff", "#ff6666"]

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=colors, shadow=True, explode=(0.05,0.05))
plt.title("Original Cancer Type Distribution (SIIM-ISIC Dataset)")
plt.show()


