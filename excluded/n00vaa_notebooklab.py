import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(file_path):
    img = load_img(file_path, target_size=(64, 64))
    img = img_to_array(img) / 255.0
    return img


train_data = pd.read_csv('../input/petfinder-pawpularity-score/train.csv') 
test_data = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')

train_data['Pawpularity'] = train_data['Pawpularity'] / 100


train_images = np.array([preprocess_image(f'../input/petfinder-pawpularity-score/train/{img_id}.jpg') for img_id in train_data['Id']])

test_images = np.array([preprocess_image(f'../input/petfinder-pawpularity-score/test/{img_id}.jpg') for img_id in test_data['Id']])

y_train = train_data['Pawpularity'].values


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False


model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


model.fit(train_images, y_train, validation_data=(train_images, train_data['Pawpularity'] / 100), epochs=10, batch_size=16)


test_predictions = model.predict(test_images)
test_predictions = test_predictions * 100

submission = pd.DataFrame({
    'Id': test_data['Id'],
    'Pawpularity': test_predictions.flatten()
})

submission.to_csv('submission.csv', index=False)

print('Submission file created successfully.')
print(submission)

