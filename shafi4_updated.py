!pip uninstall -y scikit-learn imbalanced-learn
#after doing this do the next


!pip install -U scikit-learn==1.3.2
!pip install -U imbalanced-learn==0.11.0
#after doing this click on restart and clear cell output


from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN


import os
import random
import shutil
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from glob import glob
import gc
from tensorflow.keras import mixed_precision
import math  # Side note: Added import for math, required by get_pad_width function
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential


CSV_PATH = '/kaggle/input/aptos2019-blindness-detection/train.csv'
IMAGE_DIR = '/kaggle/input/aptos2019-blindness-detection/train_images'
#c
import gc
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

# =====================
# STEP 1: Load Data
# =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = '/kaggle/input/aptos2019-blindness-detection/train.csv'
IMAGE_DIR = '/kaggle/input/aptos2019-blindness-detection/train_images'

df = pd.read_csv(CSV_PATH)
df['image'] = df['id_code'] + '.png'
df.rename(columns={'diagnosis': 'level'}, inplace=True)

# Count total images
total_images = len(df)
print(f"Total images: {total_images}")

# Count images per class
class_counts = df['level'].value_counts().sort_index()
print("\nImages per class:")
print(class_counts)

# =====================
# Plot Histogram of Class Distribution
# =====================

plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='black')
plt.xticks(class_counts.index)
plt.xlabel('Class Label')
plt.ylabel('Number of Images')
plt.title('Histogram of Images per Class')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# =====================
#  Plot 2 Images from Each Class
# =====================
print("Showing two raw images from each class")
plt.figure(figsize=(15, 10))

for class_label in range(5):  # Classes 0 to 4
    class_images = df[df['level'] == class_label]['image'].values
    selected_images = np.random.choice(class_images, 2, replace=False)
    
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(5, 2, class_label * 2 + i + 1)
        plt.imshow(img)
        plt.title(f'Class {class_label}')
        plt.axis('off')

plt.tight_layout()
plt.show()




# =====================
# STEP 2: Preprocessing + Augmentation
# =====================
IMG_SIZE = 224
NB_CHANNELS = 3

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def standardize(x):
    x = x.astype(np.float32)
    x = x / np.max(x)
    return (x - np.mean(x)) / (np.std(x))

def normalize(img):
    img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255
    return img.astype(np.uint8)

def crop_image(img, tol=10):
    def crop_image_1(img):
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    
    if img.ndim == 2:
        return crop_image_1(img)
    
    elif img.ndim == 3:
        try:
            img_cpy = img.copy()
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))

            img[:,:,0] = img1
            img[:,:,1] = img2
            img[:,:,2] = img3
            return img
        except:
            return img_cpy

def preprocess_image(img_name, label=None, base_dir=IMAGE_DIR):
    img_path = os.path.join(base_dir, img_name)
    im = cv2.imread(img_path)
    if im is None:
        print(f"Failed to load {img_path}")
        return None

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = normalize(im)
    im = crop_image(im)
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))

    

    # Note: Applying CLAHE after resize to enhance contrast while preserving color
    im_lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(im_lab)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(2, 2))
    l_channel = clahe.apply(l_channel)
    im_lab = cv2.merge([l_channel, a_channel, b_channel])
    im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)
    
    im = cv2.addWeighted(im, 4, cv2.GaussianBlur(im, (0, 0), IMG_SIZE / 10), -4, 128)

     # Mask background to black using circular ROI
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)  # ← new
    cv2.circle(mask, (IMG_SIZE // 2, IMG_SIZE // 2), IMG_SIZE // 2, 255, -1)  # ← new
    for c in range(3):  # ← new
        im[:, :, c] = np.where(mask == 255, im[:, :, c], 0)  # ← new

   
    
    return im.astype(np.uint8)

def augment_image(img):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True
    )
    img = img.reshape((1,) + img.shape)
    return next(datagen.flow(img, batch_size=1))[0].astype(np.uint8).squeeze()

print("Applying preprocessing and augmentation...")

PREPROCESSED_DIR = 'processed_images'
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
updated_rows = []

##for idx, row in df.iterrows():
##    img_name = row['image']
##    label = row['level']
##    img = preprocess_image(img_name, label, IMAGE_DIR)
##    if img is None:
##        continue
##    for i in range(2):  # 2 augmentations per image
##        aug_img = augment_image(img)
##        new_name = f"{row['image'].split('.')[0]}_aug{i}.png"
##        cv2.imwrite(os.path.join(PREPROCESSED_DIR, new_name), aug_img)
##        updated_rows.append({'image': new_name, 'level': row['level']})

for idx, row in df.iterrows():
    img_name = row['image']
    label = row['level']
    img = preprocess_image(img_name, label, IMAGE_DIR)
    if img is not None:
        new_name = row['image'].replace('.jpg', '.png').replace('.jpeg', '.png')
        cv2.imwrite(os.path.join(PREPROCESSED_DIR, new_name), img)
        updated_rows.append({'image': new_name, 'level': row['level']})

aug_df = pd.DataFrame(updated_rows)

print("Done...")




# Count total images
total_images = len(aug_df)
print(f"Total images: {total_images}")

# Count images per class
class_counts = aug_df['level'].value_counts().sort_index()
print("\nImages per class:")
print(class_counts)


# =====================
# Plot Histogram of Class Distribution
# =====================

plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='black')
plt.xticks(class_counts.index)
plt.xlabel('Class Label')
plt.ylabel('Number of Images')
plt.title('Images per class after preprocessing and augmentation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# =====================
#  Plot 2 Images from Each Class
# =====================
print("\nShowing 2 preprocessed images from each class:")
plt.figure(figsize=(15, 10))

for class_label in range(5):  # Classes 0 to 4
    class_images = aug_df[aug_df['level'] == class_label]['image'].values
    selected_images = np.random.choice(class_images, 2, replace=False)
    
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(PREPROCESSED_DIR, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(5, 2, class_label * 2 + i + 1)
        plt.imshow(img)
        plt.title(f'Class {class_label}')
        plt.axis('off')

plt.tight_layout()
plt.show()


# ignore this cell

# =====================
#  Show first 10 preprocessed images from "PREPROCESSED_DIR"
# =====================
#print("\nShowing first 10 images from PREPROCESSED_DIR:")
#plt.figure(figsize=(15, 5))

#for i, row in enumerate(aug_df.head(10).itertuples(), 1):
#    img_path = os.path.join(PREPROCESSED_DIR, row.image)
#    img = cv2.imread(img_path)      #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
#    if img is not None:
#        plt.subplot(2, 5, i)
#        plt.imshow(img)      #plt.imshow(img, cmap='gray')
#        plt.title(f"Label: {row.level}")
#        plt.axis('off')

#plt.tight_layout()
#plt.show()


# =====================
# STEP 3: Feature Extraction using ResNet50 (without classification head)
# =====================

from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB7, VGG16, VGG19, EfficientNetV2M
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess


from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def fine_tune_inceptionV3_head(img_size, aug_df, preprocessed_dir):
    # Load and preprocess images
    img_paths = [os.path.join(preprocessed_dir, fname) for fname in aug_df['image']]
    labels = aug_df['level'].values
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_cat = to_categorical(labels_encoded, num_classes=5)

    X = []
    for path in img_paths:
        img = load_img(path, target_size=(img_size, img_size))
        img = img_to_array(img)
        img = inception_preprocess(img)
        X.append(img)
    X = np.array(X)

    # Build model
    base_input = Input(shape=(img_size, img_size, 3))
    base_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_tensor=base_input)
    x = base_model.output
    x = Dropout(0.5)(x)
    output = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Train head
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training classification head...")
    model.fit(X, labels_cat, batch_size=32, epochs=10, validation_split=0.1, verbose=1)

    # Fine-tune last layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Fine-tuning last 30 layers...")
    model.fit(X, labels_cat, batch_size=32, epochs=10, validation_split=0.1, verbose=1)

    # Return the base model for feature extraction
    return Model(inputs=base_model.input, outputs=base_model.output)



FEATURES_DIR = 'features'
os.makedirs(FEATURES_DIR, exist_ok=True)

InceptionV3_model = fine_tune_inceptionV3_head(IMG_SIZE, aug_df, PREPROCESSED_DIR)


feature_rows = []

for idx, row in aug_df.iterrows():
    img_path = os.path.join(PREPROCESSED_DIR, row['image'])
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_preprocess(img)
    feature = InceptionV3_model.predict(img, verbose=0)[0]

    feature_name = row['image'].replace('.png', '.npy')
    np.save(os.path.join(FEATURES_DIR, feature_name), feature)

    feature_rows.append({'feature': feature_name, 'level': row['level']})

features_df = pd.DataFrame(feature_rows)
features_df.to_csv('features.csv', index=False)

print("Feature extraction complete...")

#give seperate for each model what i have to change to make it work with inceptionv3, yolov8, efficientnetb7, vgg16, vgg19,efficientnetv2M. 


feature_list = []
# Iterate through the feature_df
for idx, row in features_df.iterrows():
    file_name = row['feature']
    file_path = os.path.join(FEATURES_DIR, file_name)
    data = np.load(file_path)
    feature_list.append(data)

feature_list = np.array(feature_list)


# =====================
# STEP 4: Plot 2 Features from Each Class
# =====================

import seaborn as sns

features_per_class = {i: [] for i in range(5)}

for idx, row in features_df.iterrows():
    if len(features_per_class[row['level']]) < 2:
        feat = np.load(os.path.join(FEATURES_DIR, row['feature']))
        features_per_class[row['level']].append(feat)

plt.figure(figsize=(20, 10))

for cls in range(5):
    for i, feat in enumerate(features_per_class[cls]):
        plt.subplot(5, 2, cls*2+i+1)
        sns.histplot(feat, bins=30, kde=True)
        plt.title(f'Class {cls} Feature {i+1}')

plt.tight_layout()
plt.show()


# First split: 80% train, 20% temp (val + test)
train_dt, temp_dt, train_df, temp_df = train_test_split(feature_list, features_df, test_size=0.3, random_state=SEED, stratify=features_df['level'])

# Second split: 50% of temp = 10% of total for val and test each
val_dt, test_dt, val_df, test_df = train_test_split(temp_dt, temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['level'])

# Save CSVs
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)

# Print summary
print("Train / Validation / Test splits saved as CSVs:")
print(f"Train: {len(train_df)} samples")
print(f"Validation: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

# Class distribution in each split
print("\nClass distribution in Train:")
print(train_df['level'].value_counts().sort_index())
print("\nClass distribution in Validation:")
print(val_df['level'].value_counts().sort_index())
print("\nClass distribution in Test:")
print(test_df['level'].value_counts().sort_index())


# =====================
# STEP 5: Apply SMOTE for classes 1,2,3,4
# =====================

selected_classes = [ 1, 2, 3, 4]
selected_features = []
selected_labels = []

for idx, row in train_df.iterrows():
    if row['level'] in selected_classes:
        feat = np.load(os.path.join(FEATURES_DIR, row['feature']))
        selected_features.append(feat)
        selected_labels.append(row['level'])

selected_features = np.array(selected_features)
selected_labels = np.array(selected_labels)

# Find the class with the highest count
class_counts = pd.Series(selected_labels).value_counts()
highest_count = class_counts.max()

print("Class counts before SMOTE:")
print(class_counts)

# Apply SMOTE
#smote = SMOTE(random_state=SEED)
#X_resampled, y_resampled = smote.fit_resample(selected_features, selected_labels)

#adasyn
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

# Step 1: Apply ADASYN
adasyn = ADASYN(random_state=SEED, n_neighbors=5)
X_adasyn, y_adasyn = adasyn.fit_resample(selected_features, selected_labels)      #X_resampled, y_resampled = adasyn.fit_resample(selected_features, selected_labels)                                        

# Step 2: Apply ENN to clean noisy points
enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X_adasyn, y_adasyn)
#adasyn


# Save synthetic features
SYNTHETIC_DIR = 'synthetic_features'
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

synthetic_rows = []

count_original = len(selected_labels)
count_generated = len(X_resampled) - count_original

# Saving only the synthetic samples
synthetic_start_idx = count_original

for i in range(synthetic_start_idx, len(X_resampled)):
    feature = X_resampled[i]
    label = y_resampled[i]
    name = f'synthetic_{i}.npy'
    np.save(os.path.join(SYNTHETIC_DIR, name), feature)
    synthetic_rows.append({'feature': name, 'level': label})

synthetic_df = pd.DataFrame(synthetic_rows)
synthetic_df.to_csv('synthetic_features.csv', index=False)

print("Synthetic features generated for each class:")
print(synthetic_df['level'].value_counts())


# =====================
# STEP 6: Plot 2 Synthetic Features from Each Class
# =====================

synthetic_per_class = {i: [] for i in selected_classes}

for idx, row in synthetic_df.iterrows():
    if len(synthetic_per_class[row['level']]) < 2:
        feat = np.load(os.path.join(SYNTHETIC_DIR, row['feature']))
        synthetic_per_class[row['level']].append(feat)

plt.figure(figsize=(20, 10))

for cls in selected_classes:
    for i, feat in enumerate(synthetic_per_class[cls]):
        plt.subplot(4, 2, (cls-1)*2+i+1)
        sns.histplot(feat, bins=30, kde=True)
        plt.title(f'Synthetic Class {cls} Feature {i+1}')

plt.tight_layout()
plt.show()


# =====================
# STEP 7: Prepare Final trainig Dataset
# =====================

final_features = []
final_labels = []

# Add original features
for idx, row in train_df.iterrows():
    feat = np.load(os.path.join(FEATURES_DIR, row['feature']))
    final_features.append(feat)
    final_labels.append(row['level'])

# Add synthetic features
for idx, row in synthetic_df.iterrows():
    feat = np.load(os.path.join(SYNTHETIC_DIR, row['feature']))
    final_features.append(feat)
    final_labels.append(row['level'])

final_features = np.array(final_features)
final_labels = np.array(final_labels)

print(f"Train: {len(final_features)}, Val: {len(val_dt)}, Test: {len(test_dt)}")


# =====================
# Final data Distribution 
# =====================

# Count final class distribution
final_class_counts = pd.Series(final_labels).value_counts().sort_index()

print("\nFinal number of samples in each class:")
for cls, count in final_class_counts.items():
    print(f"Class {cls}: {count} samples")

# Plot histogram of final class distribution
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']  # Different colors for each class

plt.figure(figsize=(8, 6))
plt.bar(final_class_counts.index, final_class_counts.values, color=colors, edgecolor='black')
plt.xticks(final_class_counts.index)
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.title('Final Class Distribution After Augmentation and SMOTE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# STEP 8: Train on sequential
# =====================

y_train = train_df['level'].values.astype(np.int32)
y_val = val_df['level'].values.astype(np.int32)


# Simple classifier on top of extracted features
model = Sequential([
    layers.Input(shape=(2048,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(
    #train_dt, y_train,
    final_features, final_labels,
    validation_data=(val_dt, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

"""# STEP 8: Train on sequential
# =====================

y_train = train_df['level'].values.astype(np.int32)
y_val = val_df['level'].values.astype(np.int32)


# Simple classifier on top of extracted features
model = Sequential([
    layers.Input(shape=(2048,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(
    final_features, final_labels,
    validation_data=(val_dt, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)"""


# =====================
# STEP 9: Plot Accuracy and Loss
# =====================

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()


# =====================
# STEP 10: Evaluate on Test Set
# =====================

y_pred = model.predict(test_dt)
y_pred_labels = np.argmax(y_pred, axis=1)

print(classification_report(test_df['level'], y_pred_labels))

# Confusion Matrix
cm = confusion_matrix(test_df['level'], y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Assuming `model` is your final trained model
# and `test_df['level']` are the true labels

# Predict probabilities for the test set
y_true = test_df['level'].values
y_score = model.predict(test_dt)

# Binarize the output labels for multi-class AUC-ROC
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
for i in range(5):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()



# =========================================
# STEP 4: GradCAM for 2 images per class
# =========================================

IMG_SIZE = 224  
PREPROCESSED_DIR = 'processed_images'

base_model = ResNet50(weights='imagenet', include_top=False, pooling=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
last_conv_layer_name = 'conv5_block3_out'

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def load_preprocessed_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

# Read your features dataframe (to know levels)
features_df = pd.read_csv('features.csv')

# Pick 2 random images from each class
selected_images = []

for level in sorted(features_df['level'].unique()):
    level_df = features_df[features_df['level'] == level]
    samples = level_df.sample(n=2, random_state=42)
    selected_images.extend(samples['feature'].str.replace('.npy', '.png').tolist())

# Apply GradCAM
for img_name in selected_images:
    img_path = os.path.join(PREPROCESSED_DIR, img_name)
    
    # Preprocess
    img_array = load_preprocessed_image(img_path)
    
    # Generate GradCAM heatmap
    heatmap = make_gradcam_heatmap(img_array, base_model, last_conv_layer_name)
    
    # Resize heatmap
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Load original image
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    
    # Superimpose
    superimposed_img = heatmap * 0.4 + original_img

    # Plot
    plt.figure(figsize=(8,4))
    plt.suptitle(f"GradCAM: {img_name}", fontsize=14)
    
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.title('GradCAM Heatmap')
    plt.axis('off')
    
    plt.show()


