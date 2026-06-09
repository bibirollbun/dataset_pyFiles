import cv2
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from functools import partial
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Parameters for extracting features
IMG_RESIZE = (128, 128)  # Resize images for faster processing
GCS_PATH = '/kaggle/input/cassava-leaf-disease-classification'


def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = example['image']
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum


# Reuse decode_image but remove preprocess_input for classic ML
def decode_for_sklearn(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_RESIZE)
    image = tf.cast(image, tf.uint8)
    return image.numpy()


# Extract color histogram features
def extract_hist_features(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def load_features_from_tfrecords(tfrecord_files, max_items=None):
    features, labels = [], []
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(partial(read_tfrecord, labeled=True))

    for i, (image_bytes, label) in enumerate(tqdm(parsed_dataset, desc="Extracting features")):
        image_np = decode_for_sklearn(image_bytes.numpy())  # decode and convert to NumPy
        feat = extract_hist_features(image_np)
        features.append(feat)
        labels.append(label.numpy())
        if max_items and i >= max_items:
            break
    return np.array(features), np.array(labels)


TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
    tf.io.gfile.glob(GCS_PATH + '/train_tfrecords/ld_train*.tfrec'),
    test_size=0.2, random_state=5
)

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test_tfrecords/ld_test*.tfrec')


# Extract features from training and validation sets (you can reduce the max_items if needed)
X_train, y_train = load_features_from_tfrecords(TRAINING_FILENAMES, max_items=3000)
X_val, y_val = load_features_from_tfrecords(VALID_FILENAMES, max_items=1000)


# Train the Decision Tree Classifier
clf = RandomForestClassifier(
    n_estimators=100,         # Number of trees in the forest
    max_depth=30,             # You can adjust this
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1                 # Use all available CPU cores
)
clf.fit(X_train, y_train)



# Evaluate
y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


def load_test_features(tfrecord_files, max_items=None):
    features, image_ids = [], []
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(partial(read_tfrecord, labeled=False))

    for i, (image_bytes, image_id_bytes) in enumerate(tqdm(parsed_dataset, desc="Extracting test features")):
        image_np = decode_for_sklearn(image_bytes.numpy())  # decode from bytes to NumPy array
        feat = extract_hist_features(image_np)
        features.append(feat)
        image_id = image_id_bytes.numpy().decode('utf-8')
        image_ids.append(image_id)
        if max_items and i >= max_items:
            break
    return np.array(features), image_ids


# For Kaggle Submission
# Load test data
X_test, test_image_ids = load_test_features(TEST_FILENAMES)

# Predict
test_preds = clf.predict(X_test)

# Prepare submission
submission_df = pd.DataFrame({
    'image_id': test_image_ids,
    'label': test_preds
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

