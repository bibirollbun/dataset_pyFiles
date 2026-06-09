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


!unzip -q "/kaggle/input/dogs-vs-cats/train.zip"
!unzip -q "/kaggle/input/dogs-vs-cats/test1.zip"


import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split

def prepare_data(train_path, val_size=0.2, random_state=42):
    train_filenames = os.listdir(train_path)
    train_categories = ['dog' if filename.split(".")[0] == 'dog' else 'cat' for filename in train_filenames]

    df = pd.DataFrame({
        'filename': train_filenames,
        'category': train_categories
    })

    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df["category"], random_state=random_state)
    return train_df, val_df


train_path = "/kaggle/working/train"
train_df, val_df = prepare_data(train_path)
print(f"Total Training Images: {len(train_df)}")
print(f"Total Validation Images: {len(val_df)}")


train_df.head()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_path,
    x_col='filename',
    y_col='category',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32,
    subset='training'
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_path,
    x_col='filename',
    y_col='category',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32,
    subset='validation'
)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout

IMAGE_SIZE = (128, 128, 3)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=IMAGE_SIZE),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    verbose=1
)


from tensorflow.keras.utils import load_img, img_to_array

test_dir = "/kaggle/working/test1"

def load_test_images(test_dir, image_size=(128, 128)):
    image_ids = []
    images = []
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(test_dir, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            image_ids.append(int(filename.split(".")[0]))
    return np.array(images), image_ids

test_images, test_ids = load_test_images(test_dir)
predictions = model.predict(test_images, batch_size=32, verbose=1)
predictions = (predictions > 0.5).astype(int).flatten()

submission_df = pd.DataFrame({"id": test_ids, "label": predictions})
submission_df = submission_df.sort_values(by="id")
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

