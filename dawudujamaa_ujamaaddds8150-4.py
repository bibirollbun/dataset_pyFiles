# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models 
from tensorflow.keras.layers import Dense 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Needed for reproducibility
import random
seed = 42 # Set Python's hash seed
os.environ['PYTHONHASHSEED'] = str(seed)
# Set random seeds for reproducibility
tf.random.set_seed(seed)  # For TensorFlow operations
np.random.seed(seed)      # For NumPy operations
random.seed(seed)         # For Python's random module


# load the dataset containing lables and include a column for filename and numeric label
labels_df = pd.read_csv('../input/cifar-10/trainLabels.csv')
labels_df['filename'] = labels_df['id'].astype(str) + '.png'

class_vals = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
labels_df['label_idx'] = labels_df['label'].map(class_vals)

labels_df


# Unpack image files
!7z x /kaggle/input/cifar-10/train.7z -o/kaggle/working/train -bso0 -bsp0 -mtc=off


TRAIN_DIR = r"../working/train/train"

# build dataset from image files
images = []
labels = []
img_size=(32, 32) # CIFAR-10 image size

for _, row in labels_df.iterrows():
    img_path = os.path.join(TRAIN_DIR, row["filename"])

    # Load image
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)

    images.append(np.array(img))
    labels.append(row["label_idx"])

X_TRAIN = np.array(images, dtype=np.float32)
Y_TRAIN = np.array(labels, dtype=np.int64)

x_train, x_test0, y_train, y_test = train_test_split(X_TRAIN, Y_TRAIN, test_size=0.2, random_state=42, stratify=Y_TRAIN)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Check shapes
print(x_train.shape, y_train.shape)
print(x_test0.shape, y_test.shape)


# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(12, 2))

for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[i] / 255.0)
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')

plt.tight_layout()
plt.show()


# Normalize the pixel values to range between 0 and 1 
x_train = x_train / 255.0
x_test = x_test0 / 255.0

# Add a channel dimension (required for CNN) 
x_train = np.expand_dims(x_train, axis=-1) 
x_test = np.expand_dims(x_test, axis=-1) 

# Split training data into 80% training and 20% validation 
train_images, val_images, train_labels, val_labels = train_test_split(x_train, y_train, test_size=0.2, random_state=42) 


# Define a CNN model 
model = models.Sequential([
    # Data augmentation
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),

    # low-level convolutional layers
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),

    # mid-level convolutional layers
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),

    # high-level convolutional layers
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),    
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),

    # Classification and output     
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# Train the model 
history = model.fit(train_images, train_labels, epochs=45, validation_data=(val_images, val_labels), verbose=1)


fig, axs = plt.subplots(1,2,figsize=(12,4)) 

# loss
axs[0].plot(history.history['loss'], label='Training Loss') 
axs[0].plot(history.history['val_loss'], label='Validation Loss') 
axs[0].set_title('Model Loss')
axs[0].set_ylabel('Loss') 
axs[0].set_xlabel('Epoch')
axs[0].legend(['train', 'validate'], loc='upper left')

# accuracy
axs[1].plot(history.history['accuracy'], label='Training Accuracy') 
axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy') 
axs[1].set_title('Model Accuracy')
axs[1].set_ylabel('Accuracy') 
axs[1].set_xlabel('Epoch')
axs[1].legend(['train', 'validate'], loc='upper left')

plt.show()


# Evaluate the model on the test set 
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) 
print(f'\nTest accuracy: {test_acc:.2f}')


# Classification Reports 

# Predictions
y_pred_train = model.predict(x_train).argmax(axis=1)
y_pred_test = model.predict(x_test).argmax(axis=1)

# generate Classification report on training and test set
print( 'Classification Report on Training set')
print(classification_report(y_train,y_pred_train, digits=4, target_names=class_names))

print( 'Classification Report on Test set')
print(classification_report(y_test,y_pred_test, digits=4, target_names=class_names))


# Confusion matrix on Training set

conf_matrix_train = metrics.confusion_matrix(y_train, y_pred_train)

plt.figure(figsize=(8, 6)) 
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.title('Confusion Matrix for CIFAR-10 Classification on Training Set') 
plt.show() 


# Confusion matrix on Test set

conf_matrix_test = metrics.confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(8, 6)) 
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.title('Confusion Matrix for CIFAR-10 Classification on test set.') 
plt.show() 


# First 30 images in CIFAR-10 test dataset with predicted labels
plt.figure(figsize=(12, 3))

for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(x_test0[i] / 255.0)
    plt.title(f"Actual: {class_names[y_test[i][0]]}\nPred: {class_names[y_pred_test[i]]}", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()


# load test data
!7z x /kaggle/input/cifar-10/test.7z -o/kaggle/working/test -bso0 -bsp0 -mtc=off


TEST_DIR = r"../working/test/test"
test_files = os.listdir(TEST_DIR)
df_test_files = pd.DataFrame(test_files, columns=["filename"])
df_test_files['id'] = df_test_files["filename"].str.replace(".png", "")
df_test_files['id'] = df_test_files['id'].astype(int)

id_test = df_test_files['id']

df_test_files


test_images = []

for _, row in df_test_files.iterrows():
    img_path = os.path.join(TEST_DIR, row["filename"])

    # Load image
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)

    test_images.append(np.array(img))

X_TEST0 = np.array(test_images, dtype=np.float32)

# Check shapes
print(X_TEST0.shape)


# Normalize the pixel values to range between 0 and 1 
X_TEST = X_TEST0 / 255.0
X_TRAIN = X_TRAIN / 255.0

# Add a channel dimension (required for CNN) 
X_TEST = np.expand_dims(X_TEST, axis=-1) 
X_TRAIN = np.expand_dims(X_TRAIN, axis=-1) 

Y_TRAIN = Y_TRAIN.reshape(-1, 1)

# Check shapes
print(X_TRAIN.shape, Y_TRAIN.shape)


# train model on full training data
history = model.fit(X_TRAIN, Y_TRAIN, epochs=45, verbose=1)


# make predictions on competition test dataset
Y_TEST_PRED = model.predict(X_TEST).argmax(axis=1)


submission = pd.DataFrame(id_test)
submission['labelnum'] = Y_TEST_PRED

# add a column with numeric values for class
class_vals = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
submission['label']=submission['labelnum'].map(class_vals)
submission.drop('labelnum', axis=1, inplace=True)
submission.sort_values(by='id', inplace=True)
submission.to_csv(r'./submission.csv', index = False)
submission


# First 30 images in CIFAR-10 TEST Competition dataset with predicted labels
plt.figure(figsize=(12, 3))

for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(X_TEST0[i] / 255.0)
    plt.title(class_names[Y_TEST_PRED[i]])
    plt.axis('off')

plt.tight_layout()
plt.show()


# One-vs-Rest ROC curve

# Get softmax predictions
y_pred_probs = model.predict(x_test)

# Binarize the true labels
num_classes = len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))

# Compute ROC and AUC for each class
fpr = {}   # false positive rate
tpr = {}   # true positive rate
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(
    y_test_bin.ravel(), 
    y_pred_probs.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve
# Aggregate all FPR points
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Interpolate TPR for each class at these FPR points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Plot the One-vs-Rest ROC curves

plt.figure(figsize=(10, 5))

# Plot for each class
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], linewidth=2,
             label=f"Class: Rating: {class_names[i]} (AUC = {roc_auc[i]:.3f})")

# Micro & Macro
plt.plot(fpr["micro"], tpr["micro"],
         linestyle='--', linewidth=2,
         label=f"micro-average (AUC = {roc_auc['micro']:.3f})")

plt.plot(fpr["macro"], tpr["macro"],
         linestyle='--', linewidth=2,
         label=f"macro-average (AUC = {roc_auc['macro']:.3f})")

# Diagonal
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

