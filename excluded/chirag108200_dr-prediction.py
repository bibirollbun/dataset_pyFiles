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


import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



# Load the dataset
df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')

# Visualize class distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='diagnosis', data=df)
plt.title('Distribution of Diabetic Retinopathy Classes')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()



# Define image size
IMG_SIZE = 224

# Initialize lists to store images and labels
X = []
y = []

# Loop through each row in the dataframe
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    image_id = row['id_code']
    label = row['diagnosis']
    image_path = os.path.join('/kaggle/input/aptos2019-blindness-detection/train_images', f'{image_id}.png')
    
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0  # Normalize pixel values
        X.append(image)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)



# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)



# Initialize the model
model = Sequential()

# Add convolutional and pooling layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 5 classes for DR stages

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)



# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')



import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Define image size
IMG_SIZE = 224

# Load the sample submission file
test_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/sample_submission.csv')

# Initialize list to store processed test images
X_test = []

# Loop through each image in the test set
for image_id in tqdm(test_df['id_code']):
    image_path = os.path.join('/kaggle/input/aptos2019-blindness-detection/test_images', f'{image_id}.png')
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0  # Normalize pixel values
        X_test.append(image)
    else:
        # If image is not found or cannot be read, append a zero array
        X_test.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

# Convert list to NumPy array
X_test = np.array(X_test)



# Use the trained model to make predictions on the test set
predictions = model.predict(X_test)

# For each prediction, select the class with the highest probability
predicted_classes = np.argmax(predictions, axis=1)



# Display the first few rows of the submission file
print(test_df.head())



model.save('dr_prediction.h5')


