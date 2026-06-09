import warnings
warnings.filterwarnings("ignore")
!pip install --upgrade pip
!pip install -q efficientnet


!pip install --upgrade plotly


import os
import gc
import re

import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd

import tensorflow as tf
from IPython.display import SVG
import efficientnet.tfkeras as efn
from keras.utils import plot_model
import tensorflow.keras.layers as L
from keras.utils import model_to_dot
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import DenseNet121

import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tqdm.pandas()
import plotly.io as pio
pio.renderers.default = 'notebook'
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

np.random.seed(0)
tf.random.set_seed(0)

import warnings
warnings.filterwarnings("ignore")


EPOCHS = 25
SAMPLE_LEN = 100
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "../input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "../input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)


train_data.head()


test_data.head()


def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(IMAGE_PATH + file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

train_images = train_data["image_id"][:SAMPLE_LEN].progress_apply(load_image)


fig = px.imshow(cv2.resize(train_images[0], (205, 136)))
fig.show()


AUTO = tf.data.experimental.AUTOTUNE
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
GCS_DS_PATH = KaggleDatasets().get_gcs_path()


print(BATCH_SIZE)


def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'

test_paths = test_data.image_id.apply(format_path).values
train_paths = train_data.image_id.apply(format_path).values

train_labels = np.float32(train_data.loc[:, 'healthy':'scab'].values)
train_paths, valid_paths, train_labels, valid_labels =\
train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)





def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


lrfn = build_lrfn()
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)


with strategy.scope():
    model1 = tf.keras.Sequential([DenseNet121(input_shape=(512, 512, 3),
                                             weights='imagenet',
                                             include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels.shape[1],
                                         activation='softmax')])
        
    model1.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model1.summary()


history = model1.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


import matplotlib.pyplot as plt
import numpy as np

def display_training_curves(training, validation, yaxis, epochs):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    # Plot training and validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), training, marker='o', color='dodgerblue', label='Train', linestyle='-')
    plt.plot(np.arange(1, epochs + 1), validation, marker='o', color='darkorange', label='Validation', linestyle='--')
    
    # Add labels, title, and legend
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage:
display_training_curves(
    history.history['categorical_accuracy'], 
    history.history['val_categorical_accuracy'], 
    'accuracy', 
    epochs=len(history.history['categorical_accuracy'])
)



import matplotlib.pyplot as plt
import numpy as np
import cv2

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img):
    return model1.layers[2](model1.layers[1](model1.layers[0](process(img)))).numpy()[0]

def plot_predictions(image, preds, pred_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Image
    axes[0].imshow(cv2.resize(image, (205, 136)))
    axes[0].axis('off')
    axes[0].set_title(f"Prediction: {pred_label}")
    
    # Plot Bar Graph
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    colors = ['seagreen' if label == pred_label else 'dodgerblue' for label in labels]
    axes[1].bar(labels, preds, color=colors)
    axes[1].set_title("Prediction Probabilities")
    
    plt.tight_layout()
    plt.show()

# Example predictions for a few images
train_images = [train_images[2], train_images[0], train_images[3], train_images[1]]  # Adjust the indices accordingly
for img in train_images:
    preds = predict(img)
    pred_idx = np.argmax(preds)  # Index of max value
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    pred_label = labels[pred_idx]
    plot_predictions(img, preds, pred_label)



import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Make predictions on the validation/test dataset
y_true = np.argmax(valid_labels, axis=1)  # Actual labels
y_pred = np.argmax(model1.predict(valid_dataset), axis=1)  # Predicted labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Seaborn Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC Curve
# Binarize the labels for multiclass ROC
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(valid_labels)  # One-hot encoded true labels
y_pred_bin = model1.predict(valid_dataset)  # Predicted probabilities

fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC curve and ROC AUC for each class
for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=len(fpr))  # Attractive color palette
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Each Class', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# If you want to print overall AUC score
overall_auc = np.mean(list(roc_auc.values()))
print(f"Overall AUC: {overall_auc:.4f}")






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Prepare the DataFrame
EPOCHS = len(history.history['categorical_accuracy'])
acc_df = pd.DataFrame({
    "Epochs": [*np.arange(1, EPOCHS + 1).tolist() * 3],
    "Stage": ["Train"] * EPOCHS + ["Val"] * EPOCHS + ["Benchmark"] * EPOCHS,
    "Accuracy": history.history['categorical_accuracy'] + history.history['val_categorical_accuracy'] + [1.0] * EPOCHS
})

# Function to create a frame for each epoch
def update(frame):
    plt.cla()  # Clear the axes
    epoch_data = acc_df[acc_df["Epochs"] == frame + 1]  # Filter data for the current frame
    colors = {"Train": "dodgerblue", "Val": "darkorange", "Benchmark": "seagreen"}
    
    plt.barh(epoch_data["Stage"], epoch_data["Accuracy"], color=epoch_data["Stage"].map(colors))
    plt.xlim(0, 1)
    plt.xlabel("Accuracy", fontsize=12)
    plt.ylabel("Stage", fontsize=12)
    plt.title(f"Accuracy vs. Epochs (Epoch {frame + 1})", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

# Create the animation
fig, ax = plt.subplots(figsize=(8, 6))
ani = FuncAnimation(fig, update, frames=EPOCHS, repeat=False)

# Save or display the animation
ani.save('accuracy_vs_epochs.mp4', writer='ffmpeg', fps=2)  # Save as video file
plt.show()  # Display the animation in a notebook (if supported)



probs_dnn = model1.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_dnn
sub.to_csv('submission_dnn.csv', index=False)
sub.head()


with strategy.scope():
    model2 = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels.shape[1],
                                         activation='softmax')])
    
    
        
    model2.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model2.summary()


history2 = model2.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


import matplotlib.pyplot as plt
import numpy as np

def display_training_curves(training, validation, yaxis, epochs):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    # Plot training and validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), training, marker='o', color='dodgerblue', label='Train', linestyle='-')
    plt.plot(np.arange(1, epochs + 1), validation, marker='o', color='darkorange', label='Validation', linestyle='--')
    
    # Add labels, title, and legend
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage:
display_training_curves(
    history2.history['categorical_accuracy'], 
    history2.history['val_categorical_accuracy'], 
    'accuracy', 
    epochs=len(history2.history['categorical_accuracy'])
)



import matplotlib.pyplot as plt
import numpy as np
import cv2

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img):
    return model2.layers[2](model2.layers[1](model2.layers[0](process(img)))).numpy()[0]

def plot_predictions(image, preds, pred_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Image
    axes[0].imshow(cv2.resize(image, (205, 136)))
    axes[0].axis('off')
    axes[0].set_title(f"Prediction: {pred_label}")
    
    # Plot Bar Graph
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    colors = ['seagreen' if label == pred_label else 'dodgerblue' for label in labels]
    axes[1].bar(labels, preds, color=colors)
    axes[1].set_title("Prediction Probabilities")
    
    plt.tight_layout()
    plt.show()

# Example predictions for a few images
train_images = [train_images[2], train_images[0], train_images[3], train_images[1]]  # Adjust the indices accordingly
for img in train_images:
    preds = predict(img)
    pred_idx = np.argmax(preds)  # Index of max value
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    pred_label = labels[pred_idx]
    plot_predictions(img, preds, pred_label)



import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Make predictions on the validation/test dataset
y_true = np.argmax(valid_labels, axis=1)  # Actual labels
y_pred = np.argmax(model2.predict(valid_dataset), axis=1)  # Predicted labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Seaborn Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC Curve
# Binarize the labels for multiclass ROC
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(valid_labels)  # One-hot encoded true labels
y_pred_bin = model2.predict(valid_dataset)  # Predicted probabilities

fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC curve and ROC AUC for each class
for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=len(fpr))  # Attractive color palette
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Each Class', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# If you want to print overall AUC score
overall_auc = np.mean(list(roc_auc.values()))
print(f"Overall AUC: {overall_auc:.4f}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Prepare the DataFrame
EPOCHS = len(history2.history['categorical_accuracy'])
acc_df = pd.DataFrame({
    "Epochs": [*np.arange(1, EPOCHS + 1).tolist() * 3],
    "Stage": ["Train"] * EPOCHS + ["Val"] * EPOCHS + ["Benchmark"] * EPOCHS,
    "Accuracy": history2.history['categorical_accuracy'] + history2.history['val_categorical_accuracy'] + [1.0] * EPOCHS
})

# Function to create a frame for each epoch
def update(frame):
    plt.cla()  # Clear the axes
    epoch_data = acc_df[acc_df["Epochs"] == frame + 1]  # Filter data for the current frame
    colors = {"Train": "dodgerblue", "Val": "darkorange", "Benchmark": "seagreen"}
    
    plt.barh(epoch_data["Stage"], epoch_data["Accuracy"], color=epoch_data["Stage"].map(colors))
    plt.xlim(0, 1)
    plt.xlabel("Accuracy", fontsize=12)
    plt.ylabel("Stage", fontsize=12)
    plt.title(f"Accuracy vs. Epochs (Epoch {frame + 1})", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

# Create the animation
fig, ax = plt.subplots(figsize=(8, 6))
ani = FuncAnimation(fig, update, frames=EPOCHS, repeat=False)

# Save or display the animation
ani.save('accuracy_vs_epochs.mp4', writer='ffmpeg', fps=2)  # Save as video file
plt.show()  # Display the animation in a notebook (if supported)



probs_efn = model2.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_efn
sub.to_csv('submission_efn.csv', index=False)
sub.head()


with strategy.scope():
    model3 = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                    weights='noisy-student',
                                                    include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels.shape[1],
                                         activation='softmax')])
    
    
        
    model3.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model3.summary()


history3 = model3.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


import matplotlib.pyplot as plt
import numpy as np

def display_training_curves(training, validation, yaxis, epochs):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    # Plot training and validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), training, marker='o', color='dodgerblue', label='Train', linestyle='-')
    plt.plot(np.arange(1, epochs + 1), validation, marker='o', color='darkorange', label='Validation', linestyle='--')
    
    # Add labels, title, and legend
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage:
display_training_curves(
    history3.history['categorical_accuracy'], 
    history3.history['val_categorical_accuracy'], 
    'accuracy', 
    epochs=len(history3.history['categorical_accuracy'])
)



import matplotlib.pyplot as plt
import numpy as np
import cv2

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img):
    return model3.layers[2](model3.layers[1](model3.layers[0](process(img)))).numpy()[0]

def plot_predictions(image, preds, pred_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Image
    axes[0].imshow(cv2.resize(image, (205, 136)))
    axes[0].axis('off')
    axes[0].set_title(f"Prediction: {pred_label}")
    
    # Plot Bar Graph
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    colors = ['seagreen' if label == pred_label else 'dodgerblue' for label in labels]
    axes[1].bar(labels, preds, color=colors)
    axes[1].set_title("Prediction Probabilities")
    
    plt.tight_layout()
    plt.show()

# Example predictions for a few images
train_images = [train_images[2], train_images[0], train_images[3], train_images[1]]  # Adjust the indices accordingly
for img in train_images:
    preds = predict(img)
    pred_idx = np.argmax(preds)  # Index of max value
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    pred_label = labels[pred_idx]
    plot_predictions(img, preds, pred_label)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Prepare the DataFrame
EPOCHS = len(history3.history['categorical_accuracy'])
acc_df = pd.DataFrame({
    "Epochs": [*np.arange(1, EPOCHS + 1).tolist() * 3],
    "Stage": ["Train"] * EPOCHS + ["Val"] * EPOCHS + ["Benchmark"] * EPOCHS,
    "Accuracy": history3.history['categorical_accuracy'] + history3.history['val_categorical_accuracy'] + [1.0] * EPOCHS
})

# Function to create a frame for each epoch
def update(frame):
    plt.cla()  # Clear the axes
    epoch_data = acc_df[acc_df["Epochs"] == frame + 1]  # Filter data for the current frame
    colors = {"Train": "dodgerblue", "Val": "darkorange", "Benchmark": "seagreen"}
    
    plt.barh(epoch_data["Stage"], epoch_data["Accuracy"], color=epoch_data["Stage"].map(colors))
    plt.xlim(0, 1)
    plt.xlabel("Accuracy", fontsize=12)
    plt.ylabel("Stage", fontsize=12)
    plt.title(f"Accuracy vs. Epochs (Epoch {frame + 1})", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

# Create the animation
fig, ax = plt.subplots(figsize=(8, 6))
ani = FuncAnimation(fig, update, frames=EPOCHS, repeat=False)

# Save or display the animation
ani.save('accuracy_vs_epochs.mp4', writer='ffmpeg', fps=2)  # Save as video file
plt.show()  # Display the animation in a notebook (if supported)



import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Make predictions on the validation/test dataset
y_true = np.argmax(valid_labels, axis=1)  # Actual labels
y_pred = np.argmax(model3.predict(valid_dataset), axis=1)  # Predicted labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Seaborn Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC Curve
# Binarize the labels for multiclass ROC
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(valid_labels)  # One-hot encoded true labels
y_pred_bin = model3.predict(valid_dataset)  # Predicted probabilities

fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC curve and ROC AUC for each class
for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=len(fpr))  # Attractive color palette
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Each Class', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# If you want to print overall AUC score
overall_auc = np.mean(list(roc_auc.values()))
print(f"Overall AUC: {overall_auc:.4f}")



probs_efnns = model3.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_efnns
sub.to_csv('submission_efnns.csv', index=False)
sub.head()


with strategy.scope():
    model4 = tf.keras.Sequential([tf.keras.applications.InceptionV3(input_shape=(512, 512, 3),
                                                                             weights='imagenet',
                                                                             include_top=False),
                                          L.GlobalAveragePooling2D(),
                                          L.Dense(train_labels.shape[1],
                                                  activation='softmax')])

    model4.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
    model4.summary()



SVG(tf.keras.utils.model_to_dot(model_inception, dpi=70).create(prog='dot', format='svg'))



history4 = model4.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


import matplotlib.pyplot as plt
import numpy as np

def display_training_curves(training, validation, yaxis, epochs):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    # Plot training and validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), training, marker='o', color='dodgerblue', label='Train', linestyle='-')
    plt.plot(np.arange(1, epochs + 1), validation, marker='o', color='darkorange', label='Validation', linestyle='--')
    
    # Add labels, title, and legend
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage:
display_training_curves(
    history4.history['categorical_accuracy'], 
    history4.history['val_categorical_accuracy'], 
    'accuracy', 
    epochs=len(history4.history['categorical_accuracy'])
)



import matplotlib.pyplot as plt
import numpy as np
import cv2

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img):
    return model4.layers[2](model4.layers[1](model4.layers[0](process(img)))).numpy()[0]

def plot_predictions(image, preds, pred_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Image
    axes[0].imshow(cv2.resize(image, (205, 136)))
    axes[0].axis('off')
    axes[0].set_title(f"Prediction: {pred_label}")
    
    # Plot Bar Graph
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    colors = ['seagreen' if label == pred_label else 'dodgerblue' for label in labels]
    axes[1].bar(labels, preds, color=colors)
    axes[1].set_title("Prediction Probabilities")
    
    plt.tight_layout()
    plt.show()

# Example predictions for a few images
train_images = [train_images[2], train_images[0], train_images[3], train_images[1]]  # Adjust the indices accordingly
for img in train_images:
    preds = predict(img)
    pred_idx = np.argmax(preds)  # Index of max value
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    pred_label = labels[pred_idx]
    plot_predictions(img, preds, pred_label)



import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Make predictions on the validation/test dataset
y_true = np.argmax(valid_labels, axis=1)  # Actual labels
y_pred = np.argmax(model4.predict(valid_dataset), axis=1)  # Predicted labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Seaborn Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC Curve
# Binarize the labels for multiclass ROC
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(valid_labels)  # One-hot encoded true labels
y_pred_bin = model4.predict(valid_dataset)  # Predicted probabilities

fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC curve and ROC AUC for each class
for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=len(fpr))  # Attractive color palette
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Each Class', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# If you want to print overall AUC score
overall_auc = np.mean(list(roc_auc.values()))
print(f"Overall AUC: {overall_auc:.4f}")



probs_gnn = model4.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_gnn
sub.to_csv('submission_gnn.csv', index=False)
sub.head()


with strategy.scope():
    model5 = tf.keras.Sequential([tf.keras.applications.MobileNetV2(input_shape=(512, 512, 3),
                                                                            weights='imagenet',
                                                                            include_top=False),
                                          L.GlobalAveragePooling2D(),
                                          L.Dense(train_labels.shape[1],
                                                  activation='softmax')])

    model5.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
    model5.summary()



history5 = model5.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


import matplotlib.pyplot as plt
import numpy as np

def display_training_curves(training, validation, yaxis, epochs):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    # Plot training and validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, epochs + 1), training, marker='o', color='dodgerblue', label='Train', linestyle='-')
    plt.plot(np.arange(1, epochs + 1), validation, marker='o', color='darkorange', label='Validation', linestyle='--')
    
    # Add labels, title, and legend
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage:
display_training_curves(
    history5.history['categorical_accuracy'], 
    history5.history['val_categorical_accuracy'], 
    'accuracy', 
    epochs=len(history5.history['categorical_accuracy'])
)



import matplotlib.pyplot as plt
import numpy as np
import cv2

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img):
    return model5.layers[2](model5.layers[1](model5.layers[0](process(img)))).numpy()[0]

def plot_predictions(image, preds, pred_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Image
    axes[0].imshow(cv2.resize(image, (205, 136)))
    axes[0].axis('off')
    axes[0].set_title(f"Prediction: {pred_label}")
    
    # Plot Bar Graph
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    colors = ['seagreen' if label == pred_label else 'dodgerblue' for label in labels]
    axes[1].bar(labels, preds, color=colors)
    axes[1].set_title("Prediction Probabilities")
    
    plt.tight_layout()
    plt.show()

# Example predictions for a few images
train_images = [train_images[2], train_images[0], train_images[3], train_images[1]]  # Adjust the indices accordingly
for img in train_images:
    preds = predict(img)
    pred_idx = np.argmax(preds)  # Index of max value
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    pred_label = labels[pred_idx]
    plot_predictions(img, preds, pred_label)



import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Make predictions on the validation/test dataset
y_true = np.argmax(valid_labels, axis=1)  # Actual labels
y_pred = np.argmax(model5.predict(valid_dataset), axis=1)  # Predicted labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Seaborn Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC Curve
# Binarize the labels for multiclass ROC
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(valid_labels)  # One-hot encoded true labels
y_pred_bin = model5.predict(valid_dataset)  # Predicted probabilities

fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC curve and ROC AUC for each class
for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=len(fpr))  # Attractive color palette
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Each Class', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# If you want to print overall AUC score
overall_auc = np.mean(list(roc_auc.values()))
print(f"Overall AUC: {overall_auc:.4f}")



probs_mbn = model5.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_mbn
sub.to_csv('submission_mbn.csv', index=False)
sub.head()


with strategy.scope():
    model6 = tf.keras.Sequential([
        tf.keras.applications.ResNet50(input_shape=(512, 512, 3),
                                       weights='imagenet',
                                       include_top=False),  # Use pretrained ResNet50
        L.GlobalAveragePooling2D(),  # Pooling layer to reduce feature dimensions
        L.Dense(train_labels.shape[1], activation='softmax')  # Output layer for classification
    ])

    model6.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])

    model6.summary()


history6 = model6.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Make predictions on the validation/test dataset
y_true = np.argmax(valid_labels, axis=1)  # Actual labels
y_pred = np.argmax(model6.predict(valid_dataset), axis=1)  # Predicted labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Seaborn Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC Curve
# Binarize the labels for multiclass ROC
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(valid_labels)  # One-hot encoded true labels
y_pred_bin = model6.predict(valid_dataset)  # Predicted probabilities

fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC curve and ROC AUC for each class
for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=len(fpr))  # Attractive color palette
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Each Class', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# If you want to print overall AUC score
overall_auc = np.mean(list(roc_auc.values()))
print(f"Overall AUC: {overall_auc:.4f}")




def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img):
    return model6.layers[2](model6.layers[1](model6.layers[0](process(img)))).numpy()[0]

def plot_predictions(image, preds, pred_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Image
    axes[0].imshow(cv2.resize(image, (205, 136)))
    axes[0].axis('off')
    axes[0].set_title(f"Prediction: {pred_label}")
    
    # Plot Bar Graph
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    colors = ['seagreen' if label == pred_label else 'dodgerblue' for label in labels]
    axes[1].bar(labels, preds, color=colors)
    axes[1].set_title("Prediction Probabilities")
    
    plt.tight_layout()
    plt.show()

# Example predictions for a few images
train_images = [train_images[2], train_images[0], train_images[3], train_images[1]]  # Adjust the indices accordingly
for img in train_images:
    preds = predict(img)
    pred_idx = np.argmax(preds)  # Index of max value
    labels = ["Healthy", "Multiple diseases", "Rust", "Scab"]
    pred_label = labels[pred_idx]
    plot_predictions(img, preds, pred_label)



probs_rsn = model6.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_rsn
sub.to_csv('submission_rsn.csv', index=False)
sub.head()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example Data: Replace these lists with your actual model history data
epochs = list(range(1, EPOCHS + 1))  # Convert range to a list

# Replace these with the validation accuracies from your model histories
val_acc_model1 = history.history['val_categorical_accuracy']  # DenseNet121
val_acc_model2 = history2.history['val_categorical_accuracy']  # EfficientNetB7
val_acc_model3 = history3.history['val_categorical_accuracy']  # NoisyStudent EfficientNetB7
val_acc_model4 = history4.history['val_categorical_accuracy']  # InceptionV3
val_acc_model5 = history5.history['val_categorical_accuracy']  # MobileNetV2
val_acc_model6 = history6.history['val_categorical_accuracy']  # ResNet50


# Prepare DataFrame for Seaborn
model_performance = pd.DataFrame({
    'Epochs': epochs * 6,  # Multiply the list of epochs by the number of models
    'Validation Accuracy': (
        val_acc_model1 + val_acc_model2 + val_acc_model3 + val_acc_model4 + val_acc_model5+val_acc_model6
    ),  # Concatenate accuracies for all models
    'Model': (
        ['DenseNet121'] * len(epochs) +
        ['EfficientNetB7'] * len(epochs) +
        ['NoisyStudent'] * len(epochs) +
        ['InceptionV3'] * len(epochs) +
        ['MobileNetV2'] * len(epochs)+
        ['RESNET50'] * len(epochs)
    )  # Extend model labels accordingly
})

# Plot Validation Accuracy vs. Epochs
plt.figure(figsize=(14, 8))
sns.lineplot(data=model_performance, x='Epochs', y='Validation Accuracy', hue='Model', marker='o', linewidth=2)

# Customize plot
plt.title('Model Comparison: Validation Accuracy Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Validation Accuracy', fontsize=14)
plt.legend(title='Model', fontsize=12)
plt.grid(alpha=0.3)
plt.show()



ensemble_1, ensemble_2, ensemble_3 = [sub]*3

ensemble_1.loc[:, 'healthy':] = 0.50*probs_dnn + 0.50*probs_efn
ensemble_2.loc[:, 'healthy':] = 0.25*probs_dnn + 0.75*probs_efn
ensemble_3.loc[:, 'healthy':] = 0.75*probs_dnn + 0.25*probs_efn

ensemble_1.to_csv('submission_ensemble_1.csv', index=False)
ensemble_2.to_csv('submission_ensemble_2.csv', index=False)
ensemble_3.to_csv('submission_ensemble_3.csv', index=False)
ensemble_1.head()


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

# Probabilities from the top-performing models
probs_densenet = model1.predict(valid_dataset)  # DenseNet121
probs_efficientnet = model2.predict(valid_dataset)  # EfficientNetB7
probs_inception = model4.predict(valid_dataset)  # InceptionV3 or GoogLeNet

# Weighted averaging of probabilities (weights can be adjusted based on individual model performance)
weights = {
    'densenet': 0.4,
    'efficientnet': 0.1,
    'inception': 0.5
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['densenet'] * probs_densenet +
    weights['efficientnet'] * probs_efficientnet +
    weights['inception'] * probs_inception
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Save ensemble predictions to a CSV (if needed for submission)
submission = pd.DataFrame(ensemble_probs, columns=['healthy', 'disease1', 'disease2', ...])  # Replace with your class names
submission.to_csv('submission_ensemble_top_models.csv', index=False)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Ensemble Model: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC Curve and AUC for Ensemble Model
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(valid_labels)  # One-hot encoded true labels
fpr = {}
tpr = {}
roc_auc = {}

# Compute ROC curve and AUC for each class
for i in range(y_true_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], ensemble_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=len(fpr))
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Ensemble Model: ROC Curve', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Overall AUC
overall_auc = np.mean(list(roc_auc.values()))
print(f"Overall AUC: {overall_auc:.4f}")



# Probabilities from DenseNet and EfficientNet models
probs_densenet = model1.predict(valid_dataset)  # DenseNet121
probs_efficientnet = model2.predict(valid_dataset)  # EfficientNetB7

# Weighted averaging of probabilities (weights can be adjusted)
weights = {
    'densenet': 0.5,
    'efficientnet': 0.5
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['densenet'] * probs_densenet +
    weights['efficientnet'] * probs_efficientnet
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('DenseNet + EfficientNet: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"DenseNet + EfficientNet Accuracy: {accuracy:.4f}")
print(f"DenseNet + EfficientNet Precision: {precision:.4f}")
print(f"DenseNet + EfficientNet Recall: {recall:.4f}")
print(f"DenseNet + EfficientNet F1 Score: {f1:.4f}")



# Probabilities from DenseNet and EfficientNet models
probs_densenet = model1.predict(valid_dataset)  # DenseNet121
probs_efficientnet = model2.predict(valid_dataset)  # EfficientNetB7

# Weighted averaging of probabilities (weights can be adjusted)
weights = {
    'densenet': 0.75,
    'efficientnet': 0.25
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['densenet'] * probs_densenet +
    weights['efficientnet'] * probs_efficientnet
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('DenseNet + EfficientNet: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"DenseNet + EfficientNet Accuracy: {accuracy:.4f}")
print(f"DenseNet + EfficientNet Precision: {precision:.4f}")
print(f"DenseNet + EfficientNet Recall: {recall:.4f}")
print(f"DenseNet + EfficientNet F1 Score: {f1:.4f}")



# Probabilities from DenseNet and EfficientNet models
probs_densenet = model1.predict(valid_dataset)  # DenseNet121
probs_efficientnet = model2.predict(valid_dataset)  # EfficientNetB7

# Weighted averaging of probabilities (weights can be adjusted)
weights = {
    'densenet': 0.25,
    'efficientnet': 0.75
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['densenet'] * probs_densenet +
    weights['efficientnet'] * probs_efficientnet
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('DenseNet + EfficientNet: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"DenseNet + EfficientNet Accuracy: {accuracy:.4f}")
print(f"DenseNet + EfficientNet Precision: {precision:.4f}")
print(f"DenseNet + EfficientNet Recall: {recall:.4f}")
print(f"DenseNet + EfficientNet F1 Score: {f1:.4f}")



# Probabilities from DenseNet and GoogLeNet (InceptionV3) models
probs_densenet = model1.predict(valid_dataset)  # DenseNet121
probs_inception = model4.predict(valid_dataset)  # InceptionV3 (GoogLeNet)

# Weighted averaging of probabilities (weights can be adjusted)
weights = {
    'densenet': 0.5,
    'inception': 0.5
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['densenet'] * probs_densenet +
    weights['inception'] * probs_inception
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('DenseNet + GoogLeNet: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"DenseNet + GoogLeNet Accuracy: {accuracy:.4f}")
print(f"DenseNet + GoogLeNet Precision: {precision:.4f}")
print(f"DenseNet + GoogLeNet Recall: {recall:.4f}")
print(f"DenseNet + GoogLeNet F1 Score: {f1:.4f}")



# Probabilities from DenseNet and GoogLeNet (InceptionV3) models
probs_densenet = model1.predict(valid_dataset)  # DenseNet121
probs_inception = model4.predict(valid_dataset)  # InceptionV3 (GoogLeNet)

# Weighted averaging of probabilities (weights can be adjusted)
weights = {
    'densenet': 0.25,
    'inception': 0.75
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['densenet'] * probs_densenet +
    weights['inception'] * probs_inception
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('DenseNet + GoogLeNet: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"DenseNet + GoogLeNet Accuracy: {accuracy:.4f}")
print(f"DenseNet + GoogLeNet Precision: {precision:.4f}")
print(f"DenseNet + GoogLeNet Recall: {recall:.4f}")
print(f"DenseNet + GoogLeNet F1 Score: {f1:.4f}")



# Probabilities from DenseNet and GoogLeNet (InceptionV3) models
probs_densenet = model1.predict(valid_dataset)  # DenseNet121
probs_inception = model4.predict(valid_dataset)  # InceptionV3 (GoogLeNet)

# Weighted averaging of probabilities (weights can be adjusted)
weights = {
    'densenet': 0.75,
    'inception': 0.25
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['densenet'] * probs_densenet +
    weights['inception'] * probs_inception
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('DenseNet + GoogLeNet: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"DenseNet + GoogLeNet Accuracy: {accuracy:.4f}")
print(f"DenseNet + GoogLeNet Precision: {precision:.4f}")
print(f"DenseNet + GoogLeNet Recall: {recall:.4f}")
print(f"DenseNet + GoogLeNet F1 Score: {f1:.4f}")



# Probabilities from EfficientNet and GoogLeNet (InceptionV3) models
probs_efficientnet = model2.predict(valid_dataset)  # EfficientNet
probs_inception = model4.predict(valid_dataset)  # InceptionV3 (GoogLeNet)

# Weighted averaging of probabilities (weights can be adjusted)
weights = {
    'efficientnet': 0.5,
    'inception': 0.5
}

# Compute the ensemble probabilities
ensemble_probs = (
    weights['efficientnet'] * probs_efficientnet +
    weights['inception'] * probs_inception
)

# Convert ensemble probabilities to predictions
y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(valid_labels, axis=1)  # True labels

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(conf_matrix.shape[1]), yticklabels=range(conf_matrix.shape[0]))
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('EfficientNet + GoogLeNet: Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred_ensemble)
precision = precision_score(y_true, y_pred_ensemble, average='weighted')
recall = recall_score(y_true, y_pred_ensemble, average='weighted')
f1 = f1_score(y_true, y_pred_ensemble, average='weighted')

print(f"EfficientNet + GoogLeNet Accuracy: {accuracy:.4f}")
print(f"EfficientNet + GoogLeNet Precision: {precision:.4f}")
print(f"EfficientNet + GoogLeNet Recall: {recall:.4f}")
print(f"EfficientNet + GoogLeNet F1 Score: {f1:.4f}")


