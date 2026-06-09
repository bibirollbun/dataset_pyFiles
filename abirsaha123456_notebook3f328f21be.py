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


# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

# Load CSV and encode labels
file_path = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv"
df = pd.read_csv(file_path)
df['label'] = df['label'].map({'real': 1, 'editada': 0})
df.to_csv('train_modified.csv', index=False)

# Custom Data Generator Class
class ImageLabelGenerator(Sequence):
    def __init__(self, dataframe, batch_size, image_size, image_folder, shuffle=True, augmentation_params=None, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_folder = image_folder
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.augmentation_params = augmentation_params
        if self.augmentation_params:
            self.datagen = ImageDataGenerator(**self.augmentation_params)
        else:
            self.datagen = ImageDataGenerator()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indexes]
        images, labels = [], []
        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.image_folder, row['image'])
            img = image.load_img(img_path, target_size=self.image_size)
            img = image.img_to_array(img)
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / std
            images.append(img)
            labels.append(row['label'])
        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Prepare data splits
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# Parameters
image_folder = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/'
batch_size = 32
image_size = (224, 224)
augmentation_params = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Generators
train_generator = ImageLabelGenerator(train_df, batch_size, image_size, image_folder, shuffle=True, augmentation_params=augmentation_params)
val_generator = ImageLabelGenerator(val_df, batch_size, image_size, image_folder, shuffle=False)

# Custom CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Prediction on test data
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 1 if prediction[0] > 0.5 else 0

test_folder = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'
image_filenames = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

results = []
for img_name in image_filenames:
    img_path = os.path.join(test_folder, img_name)
    prediction = predict_image(img_path)
    results.append({'image': img_name, 'label': prediction})

# Save submission
submission_df = pd.DataFrame(results)
submission_csv_path = 'submission_cnn.csv'
submission_df.to_csv(submission_csv_path, index=False)
print(f"Predictions saved to {submission_csv_path}")


