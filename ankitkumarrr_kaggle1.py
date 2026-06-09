import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import NASNetMobile
from tensorflow import keras
from tensorflow.keras import layers , models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation , MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np
import shutil
import pandas as pd
import random
import matplotlib.pyplot as plt


# Load first CSV (target is already 0/1)
df1 = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")  # has columns 'image_name', 'target'
df1['filepath'] = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/" + df1['image_name'] + ".jpg"

# Upsample malignant (target=1) 7x
df1_pos = df1[df1['target'] == 1]
df1 = pd.concat([df1, pd.concat([df1_pos] * 7, ignore_index=True)], ignore_index=True)

# Load second CSV (with diagnosis)
df2_raw = pd.read_csv("/kaggle/input/isic-2019-training-groundtruth/ISIC_2019_Training_GroundTruth (2).csv")  # has columns 'image_name', 'diagnosis'


# Convert one-hot to single label
diagnosis_columns = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK']
df2_raw['diagnosis'] = df2_raw[diagnosis_columns].idxmax(axis=1)

# Map diagnosis to binary target
benign_labels = ['NV', 'BKL', 'DF', 'VASC', 'UNK']
df2_raw['target'] = df2_raw['diagnosis'].apply(lambda x: 0 if x in benign_labels else 1)

# Final columns + filepath
df2_raw['filepath'] = "/kaggle/input/jpeg-isic2019-512x512/train/" + df2_raw['image'] + ".jpg"
df2 = df2_raw[['image', 'target', 'filepath']].rename(columns={'image': 'image_name'})


# Combine datasets
df_all = pd.concat([df1[['image_name', 'target', 'filepath']], df2], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Split into train, val, test (70/15/15)
train_df, temp_df = train_test_split(df_all, test_size=0.3, stratify=df_all['target'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42)




print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


df_all.head()


import os

filepath = df_all.iloc[14]["filepath"]  # Or 'filepath' if the column name is that
if os.path.exists(filepath):
    print("File exists:", filepath)
else:
    print("File does NOT exist:", filepath)


# Parameters
IMAGE_SIZE = (224,224)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE
AUG_COPIES = 7  # Number of augmented copies for target==1

# ------------------------------------------
# Step 1: Duplicate class 1 rows in the DataFrame
df_pos = train_df[train_df['target'] == 1]
df_pos_aug = pd.concat([df_pos] * AUG_COPIES, ignore_index=True)
train_df_augmented = pd.concat([train_df, df_pos_aug], ignore_index=True)
train_df_augmented = train_df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------
# Step 2: Shades of Gray implementation
# Shades of Gray implementation using TensorFlow operations
def shades_of_gray_tf(img, power=6, gamma=None):
    img = tf.cast(img, tf.float32) # Use tf.cast instead of .astype
    if gamma is not None:
        img = tf.pow(img, (1.0 / gamma)) # Use tf.pow
    img_power = tf.pow(img, power) # Use tf.pow
    # Use tf.reduce_mean for mean calculation
    mean_per_channel = tf.pow(tf.reduce_mean(img_power, axis=[0, 1]), 1 / power)
    # Use tf.square and tf.reduce_sum for norm calculation
    norm = tf.sqrt(tf.reduce_sum(tf.square(mean_per_channel)))
    scaling_factors = norm / mean_per_channel
    img = img * scaling_factors
    img = tf.clip_by_value(img, 0, 255) # Use tf.clip_by_value
    return tf.cast(img, tf.uint8) # Use tf.cast



# TensorFlow wrapper
def tf_shades_of_gray(image):
    image = tf.py_function(func=shades_of_gray, inp=[image], Tout=tf.uint8)
    image.set_shape([None, None, 3])
    return image

# ------------------------------------------
# Step 3: Preprocessing
def preprocess_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    # Apply Shades of Gray here using the TensorFlow version
    #image = shades_of_gray_tf(image)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label


# ------------------------------------------
# Step 4: Augmentation (only for target == 1)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)

    def augment_if_positive():
        img = tf.image.random_brightness(image, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.image.random_hue(img, max_delta=0.05)
        return img

    def no_extra_augment():
        return image

    image = tf.cond(tf.equal(label, 1),
                    true_fn=augment_if_positive,
                    false_fn=no_extra_augment)

    return image, label

# ------------------------------------------
# Step 5: Dataset builder
def df_to_dataset(df, augment_data=False):
    filepaths = df['filepath'].values
    labels = df['target'].values
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(1024)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ------------------------------------------
# Step 6: Final datasets
train_ds = df_to_dataset(train_df_augmented, augment_data=True)
val_ds   = df_to_dataset(val_df)
test_ds  = df_to_dataset(test_df)



num_samples = len(train_df)
num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
print("Total samples:", num_samples)
print("Approximate batches:", num_batches)


from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


import tensorflow as tf
from tensorflow.keras import layers, models, Model

# Recommended input size for EfficientNetB6 is typically 528x528
# Ensure IMAGE_SIZE is defined and set accordingly, e.g., IMAGE_SIZE = (528, 528)
BATCH_SIZE = 16
NUM_CLASSES = 1 # Binary classification


base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.trainable = False # Freeze base initially


from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
lr_schedule = CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=1.0,
    alpha=1e-5
)


# Assuming base_model is loaded and frozen as shown above

# --- Build Model using Functional API ---
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x) # Add dropout layer as in your sequential model
outputs = layers.Dense(NUM_CLASSES, activation='sigmoid')(x) # Use NUM_CLASSES variable

model = Model(inputs=base_model.input, outputs=outputs)
#model.summary()

metrics=[
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=metrics
)



from sklearn.utils import class_weight
y_original = df_all['target']
classes = np.unique(y_original)
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=classes,
                                                  y=y_original)

class_weight_dict = dict(zip(classes, class_weights))

print("Calculated Class Weights (based on original data):", class_weight_dict)


# Callbacks: early stopping + model checkpoint
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_auc', save_best_only=True, mode='max')
]

# Training
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
    class_weight=None  # imbalance handling
)



model.save("/kaggle/working/final_model.keras")


import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt

# Get true labels and predictions
y_true = []
y_pred_probs = []

for images, labels in train_ds:
    preds = model.predict(images).ravel()
    y_pred_probs.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# Convert predicted probabilities to class labels (threshold = 0.5)
y_pred = (y_pred_probs >= 0.5).astype(int)
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Validation Data')
plt.legend(loc='lower right')
plt.grid()
plt.show()
# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report includes precision, recall, f1
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# F1 score separately if needed
f1 = f1_score(y_true, y_pred)
print("\nF1 Score:", f1)



# --- Evaluation ---
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(preds.flatten())
    y_true.extend(labels.numpy().flatten())

y_pred_binary = np.array(y_pred) > 0.5

print("\nClassification Report:")
print(classification_report(y_true, y_pred_binary))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_binary))
print("\nAUC:", roc_auc_score(y_true, y_pred))


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve, 
                            average_precision_score)

def plot_confusion_matrix(y_true, y_pred, classes, 
                          normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return ax

def plot_metrics(y_true, y_pred_probs, y_pred_binary):
    """Plot ROC curve and Precision-Recall curve"""
    # Calculate metrics
    report = classification_report(y_true, y_pred_binary, output_dict=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    avg_precision = average_precision_score(y_true, y_pred_probs)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall (AP = {avg_precision:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    
    # Print metrics
    print("\nDetailed Classification Metrics:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision (Class 1): {report['1']['precision']:.4f}")
    print(f"Recall (Class 1): {report['1']['recall']:.4f}")
    print(f"F1-Score (Class 1): {report['1']['f1-score']:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    
    return fig

# Example usage
# Assuming you have:
# y_true - true labels (0 or 1)
# y_pred_probs - predicted probabilities (continuous between 0-1)
# y_pred_binary - binary predictions (0 or 1)

# Generate metrics and plots
plot_confusion_matrix(y_true, y_pred_binary, classes=['Negative', 'Positive'], 
                     normalize=True, title='Normalized Confusion Matrix')
plt.show()

plot_metrics(y_true, y_pred, y_pred_binary)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve, 
                            average_precision_score)

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix with actual counts.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),  # 'd' means integer format
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return ax

def plot_metrics(y_true, y_pred_probs, y_pred_binary):
    """Plot ROC curve and Precision-Recall curve with metrics"""
    # Calculate metrics
    report = classification_report(y_true, y_pred_binary, output_dict=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    avg_precision = average_precision_score(y_true, y_pred_probs)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall (AP = {avg_precision:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    
    # Print metrics
    print("\nDetailed Classification Metrics:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision (Positive Class): {report['1']['precision']:.4f}")
    print(f"Recall (Positive Class): {report['1']['recall']:.4f}")
    print(f"F1-Score (Positive Class): {report['1']['f1-score']:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    
    return fig

# Example usage
# Assuming you have:
# y_true = true labels (0 or 1)
# y_pred_probs = predicted probabilities (continuous between 0-1)
# y_pred_binary = binary predictions (0 or 1, threshold=0.5)

# Generate confusion matrix with actual counts
plot_confusion_matrix(y_true, y_pred_binary, 
                     classes=['Negative', 'Positive'], 
                     title='Confusion Matrix (Actual Counts)')
plt.show()

# Generate ROC and Precision-Recall curves with metrics
plot_metrics(y_true, y_pred, y_pred_binary)
plt.show()

