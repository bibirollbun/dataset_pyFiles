import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0


train_labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
test_labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')
print(train_labels.head)


print(train_labels.shape)


plt.hist(train_labels['label'], bins=2, edgecolor='black')  # bins=2 for binary data
plt.title('Histogram of Train Labels')
plt.xlabel('Label Value')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non Cancer', 'Cancer'])
plt.show()


# removing the rows who label value is non 0 or 1
train_labels = train_labels[train_labels['label'].isin([0, 1])]


non_cancer_data = train_labels[train_labels['label'] == 0]
cancer_data = train_labels[train_labels['label'] == 1]

non_cancer_samples = non_cancer_data.sample(n=25000, random_state=42, replace=False)
cancer_samples = cancer_data.sample(n=25000, random_state=42, replace=False)

train_data = pd.concat([non_cancer_samples, cancer_samples])

train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)


train_dir = '/kaggle/input/histopathologic-cancer-detection/train/'
test_dir = '/kaggle/input/histopathologic-cancer-detection/test/'
def load_image_from_id(image_id, image_dir=train_dir):
    image_path = image_dir + image_id + '.tif'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


images = np.array([load_image_from_id(i) for i in train_data['id']])
y = train_data['label'].values


images.shape



cnn_model1 = keras.Sequential([
    layers.Input(shape=(96,96,3)),
    layers.Rescaling(1./255), # normalizing the values from 0 to 1
    layers.Cropping2D(cropping=32),
    layers.Conv2D(20, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(60, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    
    # Output layer
    layers.Dense(1, activation='sigmoid')    
])

cnn_model1.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='binary_crossentropy',
    metrics=['accuracy', 'auc']
)
cnn_model1.summary()


es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cnn_model1 = cnn_model1.fit(images, y, validation_split=0.2, epochs=50, callbacks=[es])


# Adding one more layer in between with 40 nodes
cnn_model2 = keras.Sequential([
    layers.Input(shape=(96,96,3)),
    layers.Rescaling(1./255), # normalizing the values from 0 to 1
    layers.Cropping2D(cropping=32),
    layers.Conv2D(20, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(60, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    
    # Output layer
    layers.Dense(1, activation='sigmoid')    
])

cnn_model2.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='binary_crossentropy',
    metrics=['accuracy', 'auc']
)
cnn_model2.summary()


es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cnn_model2 = cnn_model2.fit(images, y, validation_split=0.2, epochs=50, callbacks=[es])


from tensorflow.keras.applications import EfficientNetB0

# Local path to the uploaded weights
weights_path = '/kaggle/input/efficient_net/keras/default/1/efficientnetb0_notop.h5'
base_model = EfficientNetB0(weights=weights_path, include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False  # freezes during initial training.

effnet_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
effnet_model.summary()


effnet_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'auc']
)
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model1 = effnet_model.fit(images, y, validation_split=0.20, epochs=15, callbacks=[es])


effnet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'auc']
)
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model2 = effnet_model.fit(images, y, validation_split=0.20, epochs=15, callbacks=[es])


def plot_validation_accuracy_condensed(model, model_name=None):
    
    val_accuracy = model.history['val_accuracy']

    epochs = range(1, len(val_accuracy) + 1)

    plt.figure(figsize=(7, 4)) # Slightly smaller figure
    plt.plot(epochs, val_accuracy, marker='.', linestyle='-', label='Val Accuracy') # Smaller marker
    plt.title(f'{model_name} - Val Accuracy' if model_name else 'Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    if len(epochs) <= 15: plt.xticks(epochs)
    else: plt.locator_params(axis='x', integer=True)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Grid on y-axis only, lighter
    plt.tight_layout() # Adjust layout
    plt.show()



plot_validation_accuracy_condensed(cnn_model1, "efficient net cnn_model1")
plot_validation_accuracy_condensed(cnn_model2, "efficient net cnn_model2")
plot_validation_accuracy_condensed(model1, "efficient net cnn_model1")
plot_validation_accuracy_condensed(model2, "efficient net cnn_model2")


test_images = np.array([load_image_from_id(i, image_dir=test_dir) for i in test_labels['id']])



y_pred_output = effnet_model.predict(test_images)
y_pred_output = y_pred_output.ravel()


!rm -f /kaggle/working/submission_cnn.npy


submission_cnn_df = pd.DataFrame({
            'id':test_labels["id"],
            'label':y_pred_output })
submission_cnn_df.to_csv('submission.csv', index=False)

