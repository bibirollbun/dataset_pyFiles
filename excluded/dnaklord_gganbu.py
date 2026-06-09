from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
import tensorflow as tf
import os


def safe_load(path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        return img.convert('RGB')
    except (UnidentifiedImageError, OSError) as e:
        return None


TRAIN_DIR = '/kaggle/input/logical-rhythm-2k21-gganbu/Training/Training'
images = []
labels = []
for denom in os.listdir(TRAIN_DIR):
    if denom.startswith('India') or denom.startswith('Thai'):
        for img_name in os.listdir(os.path.join(TRAIN_DIR, denom)):
            img = safe_load(os.path.join(TRAIN_DIR, denom, img_name))
            if img is not None:
                img = img.resize((128, 128))
                images.append(np.array(img))
                labels.append('_'.join(denom.split('_')[:2]))


X = np.array(images) / 255.0
y = np.array(labels)


le = LabelEncoder()
y = le.fit_transform(y) 


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


num_classes = len(set(y))
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)


class_names = le.classes_
def predict_image(path):
    img = safe_load(path)
    if img is None:
        return "Invalid image"
    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    return class_names[predicted_class_idx]  # decode back to string


TEST_DIR = '/kaggle/input/logical-rhythm-2k21-gganbu/Test/Test'
df = pd.DataFrame(columns=['image_number', 'country_denomination'])
for img_name in os.listdir(TEST_DIR):
    denom = predict_image(os.path.join(TEST_DIR, img_name))
    df.loc[len(df)] = [img_name.split('.')[0], denom.lower()]
df.head()


df.to_csv('submission.csv', index=False)

