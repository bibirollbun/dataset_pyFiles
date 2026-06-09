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


!pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras opencv-python


import pandas as pd

df = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
print(df.head())
print(df['label'].value_counts())


import matplotlib.pyplot as plt
import cv2

sample = df.sample(5)
for idx, row in sample.iterrows():
    img = cv2.imread(f"/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/{row['filename']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(row['label'])
    plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df['label'])


label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_map)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, 
    horizontal_flip=True, 
    rotation_range=20,
    zoom_range=0.2
)

train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train',
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    subset='training',
    batch_size=32
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train',
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    subset='validation',
    batch_size=32
)



from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)


import os
test_df = pd.DataFrame({'filename': os.listdir('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test')})

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=test_df,
    directory='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test',
    x_col='filename',
    y_col=None,
    target_size=(224, 224),
    class_mode=None,
    batch_size=32,
    shuffle=False
)

preds = model.predict(test_gen)
pred_classes = le.inverse_transform(preds.argmax(axis=1))


test_df['label'] = pred_classes
test_df[['filename', 'label']].to_csv('submission.csv', index=False)


# Assuming test_df has a 'filename' column and you predicted labels into pred_classes
test_df['label'] = pred_classes

# Create submission DataFrame
submission = test_df[['filename', 'label']]

# Save to CSV without index
submission.to_csv('submission.csv', index=False)

# Optional: Show first few rows to verify
print(submission.head())


