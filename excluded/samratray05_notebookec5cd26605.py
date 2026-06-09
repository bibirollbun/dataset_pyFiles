import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import pandas as pd
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Layer, Permute, Multiply, Softmax)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# Constants
IMG_HEIGHT, IMG_WIDTH = 64, 256
BATCH_SIZE = 64
MAX_LABEL_LENGTH = 6
EPOCHS = 50

# Paths
TRAIN_DATASET_PATH = '/kaggle/input/opencode-24-geek-haven/train/train'
TEST_DATASET_PATH = '/kaggle/input/opencode-24-geek-haven/test/test'
MODEL_SAVE_PATH = './checkpoints/captcha_model.h5'


# Load data paths and labels
train_paths = [os.path.join(TRAIN_DATASET_PATH, fname) for fname in os.listdir(TRAIN_DATASET_PATH)]
train_labels = [fname.split('.')[0] for fname in os.listdir(TRAIN_DATASET_PATH)]

val_paths = train_paths[-100:]  # Use the last 100 samples for validation
val_labels = train_labels[-100:]
train_paths = train_paths[:-100]
train_labels = train_labels[:-100]

test_image_paths = [os.path.join(TEST_DATASET_PATH, fname) for fname in os.listdir(TEST_DATASET_PATH)]


# Character Encoding with Blank Class
all_characters = sorted(set("".join(train_labels)))
vocab = [''] + all_characters  # Blank class is index 0
num_classes = len(vocab)

# StringLookup Layer
char_lookup = tf.keras.layers.StringLookup(
    vocabulary=vocab,
    mask_token=None,
    num_oov_indices=0
)

# Label Encoder
def label_encoder(label):
    encoded_label = char_lookup(tf.strings.unicode_split(label, 'UTF-8'))
    encoded_label = tf.pad(encoded_label, [[0, MAX_LABEL_LENGTH - tf.shape(encoded_label)[0]]], constant_values=0)
    return encoded_label

# CTC Loss Function
def ctc_loss(y_true, y_pred):
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    y_true_sparse = tf.cast(y_true, tf.int32)
    y_true_sparse = tf.sparse.from_dense(y_true_sparse)
    logit_length = tf.fill([tf.shape(y_pred)[1]], tf.shape(y_pred)[0])
    loss = tf.nn.ctc_loss(
        labels=y_true_sparse,
        logits=y_pred,
        label_length=None,
        logit_length=logit_length,
        blank_index=0,
        logits_time_major=True
    )
    return tf.reduce_mean(loss)

# Data Loading Function
def load_image(path, label=None):
    img = tf.io.read_file(path)
    
    def decode_png():
        return tf.io.decode_png(img, channels=1)
    
    def decode_jpeg():
        return tf.io.decode_jpeg(img, channels=1)
    
    def decode_default():
        return tf.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.uint8)
    
    png_condition = tf.strings.regex_full_match(path, ".*\\.png")
    jpeg_condition = tf.strings.regex_full_match(path, ".*\\.jpg")
    
    img = tf.cond(png_condition, decode_png, lambda: tf.cond(jpeg_condition, decode_jpeg, decode_default))
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.image.grayscale_to_rgb(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    
    if label is not None:
        encoded_label = label_encoder(label)
        return img, encoded_label
    return img



# Create Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(len(train_paths))
train_dataset = train_dataset.map(lambda path, label: load_image(path, label), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(lambda path, label: load_image(path, label), num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# Clear any previous session to avoid conflicting strategies
from tensorflow.keras import backend as K
K.clear_session()

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Bidirectional, Dropout

def build_model(input_shape, num_classes):
    # Use ResNet50 as the base model for better feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    # Reshape the output to feed into LSTM layers
    x = Reshape((-1, x.shape[-1]))(x)
    
    # Add Bidirectional LSTM layers for better sequence modeling
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.1)(x)  # Add dropout for regularization
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    
    # Output layer with num_classes units and linear activation
    predictions = Dense(num_classes, activation='linear')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Define the Mirrored Strategy
strategy = tf.distribute.MirroredStrategy()

# Print the number of devices
print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

# Model Compilation and Training
with strategy.scope():
    model = build_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=ctc_loss)



model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)


import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Prediction and Decoding
def decode_predictions(predictions, char_lookup):
    predictions_time_major = tf.transpose(predictions, perm=[1, 0, 2])
    logit_length = tf.fill([tf.shape(predictions)[0]], tf.shape(predictions_time_major)[0])
    decoded_sparse, _ = tf.nn.ctc_greedy_decoder(
        predictions_time_major,
        logit_length
    )
    decoded_dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1)
    vocab = char_lookup.get_vocabulary()
    decoded_labels = []
    for row in decoded_dense.numpy():
        label = ''.join([vocab[i] for i in row if i != -1 and i != 0])
        decoded_labels.append(label)
    return decoded_labels

# Collect predictions and indices
# Prepare test_dataset with image paths and their indices
test_image_indices = list(range(len(test_image_paths)))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_indices))
test_dataset = test_dataset.map(lambda path, idx: (load_image(path), idx), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Collect predictions and indices
predictions = []
indices = []
for (batch_images, batch_indices) in test_dataset:
    batch_preds = model.predict(batch_images)
    predictions.append(batch_preds)
    indices.extend(batch_indices.numpy())

# Concatenate predictions and sort by original indices
predictions = np.concatenate(predictions, axis=0)
sorted_order = np.argsort(indices)
predictions = predictions[sorted_order]

# Decode predictions
decoded_labels = decode_predictions(predictions, char_lookup)

# Visualization
def plot_images_with_predictions(image_paths, predictions, num_images=10):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        img = load_image(image_paths[i])
        img = (img.numpy() + 1) / 2  # Normalize image for display
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {predictions[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_images_with_predictions(test_image_paths[:10], decoded_labels[:10])

# Generate submission file
test_ids = [os.path.basename(path).split('.')[0] + '.jpg' for path in test_image_paths]  # Add '.jpg'

# Align test IDs with sample_submission.csv
sample_submission = pd.read_csv('/kaggle/input/opencode-24-geek-haven/sample_submission.csv')

# Initialize aligned_labels with empty strings
aligned_labels = []
for id in sample_submission['ID']:
    if id in test_ids:
        aligned_labels.append(decoded_labels[test_ids.index(id)])
    else:
        aligned_labels.append("")  # Add an empty string for missing IDs

# Generate aligned submission file
output_df = pd.DataFrame({'ID': sample_submission['ID'], 'Label': aligned_labels})
output_df.to_csv('./test_predictions_aligned.csv', index=False)

# Debug assertions
assert len(decoded_labels) == len(test_ids), "Mismatch in number of labels and IDs."

# Debug prints to verify correctness
for i in range(10):
    print(f"Image {test_image_paths[i]} -> Predicted Label: {decoded_labels[i]}")

# Check for missing or extra IDs
extra_ids = set(test_ids) - set(sample_submission['ID'])
missing_ids = set(sample_submission['ID']) - set(test_ids)
print(f"Extra IDs in test_predictions: {extra_ids}")
print(f"Missing IDs in test_predictions: {missing_ids}")


# Count the number of files in TEST_DATASET_PATH
num_test_files = len(os.listdir(TEST_DATASET_PATH))
print(f"Number of files in the test dataset: {num_test_files}")


# Count the number of predictions made by the model
num_predictions = len(decoded_labels)
print(f"Number of predictions made by the model: {num_predictions}")


# Check if the numbers match
if num_test_files == num_predictions:
    print("The number of files matches the number of predictions.")
else:
    print("Mismatch: Check the dataset or predictions pipeline!")


# Save the model
model.save('./Captcha2.h5')  # You can change the filename or path as needed





