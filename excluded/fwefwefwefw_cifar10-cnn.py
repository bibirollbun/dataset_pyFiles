import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install py7zr


import py7zr

os.makedirs('/kaggle/temp/train', exist_ok=True)
os.makedirs('/kaggle/temp/test', exist_ok=True)

# Extract TRAIN data
with py7zr.SevenZipFile('../input/cifar-10/train.7z', mode='r') as archive_train:
    archive_train.extractall(path='/kaggle/temp/train')

# Extract TEST data
with py7zr.SevenZipFile('../input/cifar-10/test.7z', mode='r') as archive_test:
    archive_test.extractall(path='/kaggle/temp/test')

print("Extraction complete!")


print("Train files:", len(os.listdir('/kaggle/temp/train/train')))
print("Test files:", len(os.listdir('/kaggle/temp/test/test'))) 


import pandas as pd

labels_df = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
print(labels_df.head())


import numpy as np
from PIL import Image
import os
from tqdm import tqdm

train_dir = '/kaggle/temp/train/train'
x_train = []
y_train = []

for idx, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0]):
    img_id = row['id']
    label = row['label']
    img_path = os.path.join(train_dir, f"{img_id}.png")
    img = Image.open(img_path)
    img = np.array(img)
    x_train.append(img)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape, y_train.shape)


import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

i = 56
plt.imshow(x_train[i])
plt.title(f"Label: {y_train[i]}")
plt.axis('off')
plt.show()
x_train = x_train.astype('float32') / 255.0
print(x_train[1])

le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_train_onehot = to_categorical(y_train_int, num_classes=10)
print(y_train_onehot.shape)


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),

    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    x_train, y_train_onehot,
    batch_size=64,
    epochs=20,
    validation_split=0.2 
)


test_dir = '/kaggle/temp/test/test'
test_files = sorted(
    [f for f in os.listdir(test_dir) if f.endswith('.png')],
    key=lambda x: int(x.split('.')[0])  # Sort by numeric ID
)

x_test = []
for fname in tqdm(test_files):
    img_path = os.path.join(test_dir, fname)
    img = Image.open(img_path)
    img = np.array(img).astype('float32') / 255.0  # Normalize here
    x_test.append(img)

x_test = np.array(x_test)


# Predict
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)  # Shape: (300000,)


import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Make predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
predicted_class_names = le.inverse_transform(predicted_labels)  # Decode labels using LabelEncoder

# Display multiple test images with predicted labels in a grid
num_images_to_display = 25  # Adjust this number as needed
rows, cols = 5, 5  # 5x5 grid
plt.figure(figsize=(15, 15))  # Adjust figure size for better visibility
for i in range(min(num_images_to_display, len(x_test))):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Predicted: {predicted_class_names[i]}", fontsize=10)
    plt.axis('off')
plt.tight_layout()  # Adjust spacing between subplots
plt.show()

# Count predictions for each class
pred_counts = Counter(predicted_class_names)
class_names = list(pred_counts.keys())
counts = list(pred_counts.values())

# Create a bar chart for prediction distribution
{
  "type": "bar",
  "data": {
    "labels": class_names,
    "datasets": [{
      "label": "Prediction Distribution",
      "data": counts,
      "backgroundColor": ["#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1", "#955251", "#B565A7", "#009B77", "#DD4124", "#D65076"],
      "borderColor": ["#D64550", "#5A4A78", "#618A3D", "#D9A7B0", "#7A8EB1", "#7A3E3E", "#8B4A8E", "#007B5F", "#B3311F", "#B03A5E"],
      "borderWidth": 1
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": True,
        "title": {
          "display": True,
          "text": "Number of Predictions"
        }
      },
      "x": {
        "title": {
          "display": True,
          "text": "Class"
        }
      }
    },
    "plugins": {
      "legend": {
        "display": False
      },
      "title": {
        "display": True,
        "text": "Distribution of Predicted Classes"
      }
    }
  }
}

