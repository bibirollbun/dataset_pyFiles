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


import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import cv2
import pandas as pd
import os 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D,Dropout,BatchNormalization,Reshape
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D,Dropout,BatchNormalization,Reshape
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import math
from tensorflow.keras.models import load_model


train_df=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-fall-2023/train.csv")
test_df=pd.read_csv('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/test.csv')


train_df['file']="/kaggle/input/applications-of-deep-learning-wustl-fall-2023/"+train_df['file']


test_df['file']="/kaggle/input/applications-of-deep-learning-wustl-fall-2023/"+test_df['file']


train_df.head()


train_df.shape


test_df.head()


train_df['glasses'].value_counts()


train_df.isnull().sum()


# Count the occurrences of each label
label_counts = train_df['glasses'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.title('Distribution of Glasses')
plt.xlabel('Glasses')
plt.ylabel('Count')
plt.xticks(ticks=label_counts.index, labels=['0', '1'], rotation=0)
plt.show()



def show_sample_images(df, label, num_images=5):
    sample_images = train_df[train_df['glasses'] == label]['file'].sample(num_images).values
    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Label: {label}')
    plt.show()

# Show sample images for both classes
show_sample_images(train_df, 0)
show_sample_images(train_df, 1)


X_train, X_test, y_train, y_test = train_test_split(train_df['file'], train_df['glasses'], test_size=0.2, random_state=42)


def load_and_preprocess_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))  # Resize to a fixed size
    img = img / 255.0  # Normalize pixel values
    return img


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Load and preprocess images for training
train_images = np.array([load_and_preprocess_image(img_path) for img_path in X_train])
test_images = np.array([load_and_preprocess_image(img_path) for img_path in X_test])

model.fit(train_images, y_train, epochs=10, batch_size=32, validation_data=(test_images, y_test))


model.save('cnn_model.h5')


test_loss, test_accuracy = model.evaluate(test_images, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')


# Make predictions on the test set
predictions = model.predict(test_images)
predictions = (predictions > 0.5).astype(int)


cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Glass', 'Glass'], yticklabels=['No Glass', 'Glass'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()


def preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        img = load_and_preprocess_image(img_path)  # Ensure this function resizes and normalizes the image
        images.append(img)
    return np.array(images)
test_images = preprocess_images(test_df['file'].values)


# Make predictions
predictions = model.predict(test_images)
predictions = (predictions > 0.5).astype(int)


predictions


predictions = predictions.flatten() 


submission=pd.DataFrame({
    'id':test_df['id'],
    'glasses':predictions
})


submission.head()


submission.to_csv('submission.csv',index=False)




