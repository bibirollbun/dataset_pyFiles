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


file_path = "/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Map the 'label' column to binary values
df['label'] = df['label'].map({'real': 1, 'editada': 0})
df.to_csv('train_modified.csv', index=False)
df.head()


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageLabelGenerator(Sequence):
    def __init__(self, dataframe, batch_size, image_size, image_folder, shuffle=True,augmentation_params=None, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_folder = image_folder  # Folder where images are stored
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))  # Track indexes for shuffling

        self.augmentation_params = augmentation_params
        if self.augmentation_params:
            self.datagen = ImageDataGenerator(**self.augmentation_params)
        else:
            self.datagen = ImageDataGenerator()  

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))  # Number of batches per epoch

    def __getitem__(self, index):
        # Get the indexes of the images for this batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get the image filenames and labels for this batch
        batch_data = self.dataframe.iloc[batch_indexes]
        
        # Load images and their labels
        images = []
        labels = []
        for _, row in batch_data.iterrows():
            img_filename = row['image']  # Filename column
            label = row['label']  # Label column
            img_path = self.image_folder + img_filename  # Construct the full image path
            img = image.load_img(img_path, target_size=self.image_size)
            img = image.img_to_array(img) / 255.0  # Normalize the image
            images.append(img)
            labels.append(label)
        
        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            # Shuffle the indexes at the end of each epoch
            np.random.shuffle(self.indexes)



from sklearn.model_selection import train_test_split

# Load the CSV file
# df = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')

# Split the dataframe into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the folder where your images are stored
image_folder = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/'  # Update this with the actual path to your images

# Define batch size and image size
batch_size = 16
image_size = (224, 224)  # ResNet50 expects images of size 224x224

augmentation_params = {
    'rotation_range': 40,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}
# Create the data generators
train_generator = ImageLabelGenerator(train_df, batch_size, image_size, image_folder,shuffle=True,
    augmentation_params=augmentation_params)
val_generator = ImageLabelGenerator(val_df, batch_size, image_size, image_folder,shuffle=False)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Build a custom CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(2048, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# Set up learning rate scheduler to adjust the learning rate if validation loss plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=12,
    validation_data=val_generator,
    callbacks=[lr_scheduler]
)



import os
import pandas as pd
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Load and resize the image
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)  # Make prediction
    return 1 if prediction[0] > 0.5 else 0

# Define the test folder path
test_folder = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'  # Replace with the actual path to your test folder

# Get a list of all image filenames in the test folder
image_filenames = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Prepare a list to store results
results = []

# Iterate over each image and make predictions
for img_name in image_filenames:
    img_path = os.path.join(test_folder, img_name)  # Construct the full image path
    prediction = predict_image(img_path)  # Predict if the image is real or fake
    results.append({'image': img_name, 'label': prediction})  # Append result

# Convert results to a DataFrame
submission_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
submission_csv_path = 'submission2.csv'
submission_df.to_csv(submission_csv_path, index=False)

print(f"Predictions saved to {submission_csv_path}")



sub_data = pd.read_csv("submission2.csv")
sub_data.head()

