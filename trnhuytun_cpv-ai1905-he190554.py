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


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
from tqdm import tqdm
import cv2


# Load datasets
train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

# Set image path
IMAGE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/images/"


# Prepare image data
def load_and_preprocess_image(image_id):
    path = os.path.join(IMAGE_PATH, image_id + ".jpg")
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    return img

# Load train images
X_train = np.array([load_and_preprocess_image(img_id) for img_id in train_df["image_id"]])

# Prepare labels
y_train = train_df.drop(columns=["image_id"]).values

# Define model
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(4, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Load test images
X_test = np.array([load_and_preprocess_image(img_id) for img_id in test_df["image_id"]])

# Make predictions
test_preds = model.predict(X_test)


# Create submission file
submission = pd.DataFrame(test_preds, columns=["healthy", "multiple_diseases", "rust", "scab"])
submission.insert(0, "image_id", test_df["image_id"])
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")

