import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight


class TrainingDataset:
    def __init__(self, title, images_npz_path, labels_csv_path = "none"):

        self.images = self.get_images_from_npz(images_npz_path)
        self.n = len(list(self.images))
        self.labels = self.get_labels_from_csv(self.n, labels_csv_path)

        print(f"Loaded '{title}' data, containing {self.n} samples")

    def get_images_from_npz(self, filename):

        data = np.load(filename)
        images = data['arr_0']
        return images

    def get_labels_from_csv(self, n, filename):

        if not os.path.exists(filename):
            return ["Unknown"] * n

        train_y = pd.read_csv(filename)
        train_y = train_y["Predicted"].values
        return train_y


DATASET_DIRECTORY = "/kaggle/input/cnn-face-recognition-25"
TRAIN_IMAGES_NPZ = f"{DATASET_DIRECTORY}/faces_train_x.npz"
TRAIN_LABELS_CSV = f"{DATASET_DIRECTORY}/faces_train_y.csv"
TEST_IMAGES_NPZ = f"{DATASET_DIRECTORY}/faces_test_x.npz"


TRAINING_DATASET = TrainingDataset("train", TRAIN_IMAGES_NPZ, TRAIN_LABELS_CSV)
TESTING_DATASET = TrainingDataset("test", TEST_IMAGES_NPZ)


train_data = TRAINING_DATASET

X = TRAINING_DATASET.images.astype("float32") / 255.0
X = np.expand_dims(X, axis=-1)
y = np.array(TRAINING_DATASET.labels)


input_shape = X.shape[1:]
NUM_CLASSES = 8

NUM_CONV_FILTERS = 64
NUM_DENSE_NODES = 128

model = Sequential([
    Conv2D(NUM_CONV_FILTERS, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(NUM_CONV_FILTERS, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(NUM_CONV_FILTERS, (3, 3), activation='relu', padding='same'),
    Conv2D(NUM_CONV_FILTERS, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(NUM_DENSE_NODES, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X, y,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([-0.05, 1.05])
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()


X_test = TESTING_DATASET.images.astype("float32") / 255.0
X_test = np.expand_dims(X_test, axis=-1)

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels[:10])


SAMPLE_SUBMISSION_CSV = f"{DATASET_DIRECTORY}/faces_test_sample_solution.csv"
output_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
output_df["Predicted"] = predicted_labels

output_df.to_csv("submission.csv", index=False)
print(f"submission saved.")

output_df = pd.read_csv("submission.csv")
output_df.head()

