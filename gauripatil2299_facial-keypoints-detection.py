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


import zipfile

# Open the zip file and extract the CSV
with zipfile.ZipFile('/kaggle/input/facial-keypoints-detection/training.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')  # Extract to a working folder

# Now load the CSV file
train_df = pd.read_csv('/kaggle/working/training.csv')

# Display first few rows
train_df.head()



# Convert image strings to numpy arrays
train_df['Image'] = train_df['Image'].apply(lambda img: np.fromstring(img, sep=' ') if isinstance(img, str) else np.nan)

# Drop rows with missing data (optional but often needed)
train_df.dropna(inplace=True)

# Convert image column to 96x96 array
X = np.stack(train_df['Image'].values)  # Shape: (num_samples, 9216)
X = X.reshape(-1, 96, 96, 1).astype(np.float32)  # Add channel dimension

# Normalize pixel values to [0, 1]
X /= 255.0

# Extract target keypoints
y = train_df.drop('Image', axis=1).values  # Shape: (num_samples, 30)



import matplotlib.pyplot as plt

# Function to plot a single image with keypoints
def plot_image_with_keypoints(image, keypoints):
    plt.imshow(image.reshape(96, 96), cmap='gray')
    plt.scatter(keypoints[0::2], keypoints[1::2], marker='x', s=20, color='red')  # x: even indices, y: odd
    plt.axis('off')

# Plot 5 random images
plt.figure(figsize=(15, 5))
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    plot_image_with_keypoints(X[i], y[i])
plt.suptitle('Sample Facial Keypoints', fontsize=16)
plt.show()



import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(30)  # 15 keypoints × 2 (x, y)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Print model summary
model.summary()



# Train the model
history = model.fit(X, y, epochs=50, batch_size=64, validation_split=0.1)



plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.show()


# Predict on some images
preds = model.predict(X[:5])

# Visualize predictions
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plot_image_with_keypoints(X[i], preds[i])
plt.suptitle('Predicted Facial Keypoints')
plt.show()



model.save('/kaggle/working/facial_keypoints_model.h5')



lookup_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')


test_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')
test_df['Image'] = test_df['Image'].apply(lambda im: np.fromstring(im, sep=' ').reshape(96, 96) / 255.0)
X_test = np.stack(test_df['Image'].values).reshape(-1, 96, 96, 1)


from tensorflow.keras.models import load_model

model = load_model('/kaggle/working/facial_keypoints_model.h5')  # Or the path where your model is saved
preds = model.predict(X_test)


preds *= 96


# Load the features used in training
features = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip').dropna().columns[:-1]  # 30 keypoints

# Create a dictionary for each test image's predicted keypoints
pred_df = pd.DataFrame(preds, columns=features)

locations = []
for idx, row in lookup_df.iterrows():
    feature_name = row['FeatureName']
    image_id = row['ImageId'] - 1  # ImageId starts from 1
    value = pred_df.loc[image_id, feature_name]
    locations.append(value)

lookup_df['Location'] = locations
lookup_df[['RowId', 'Location']].to_csv('submission.csv', index=False)


