import cv2
import os
import pandas as pd
import numpy as np
import math
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


#creating file paths and defining folder structure
img_path = '/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age'

df=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/train.csv")


df.head()


df.shape


#age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=10, kde=True, color='skyblue', stat='density')
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(axis='y')
plt.show()


#10 random images
sample_df = df.sample(10, random_state=42)
image_files = sample_df['filename']
ages = sample_df['age']

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.subplots_adjust(hspace=0.5)
for idx, (filename, age) in enumerate(zip(image_files, ages)):
    img = cv2.imread(os.path.join(img_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[idx // 5, idx % 5].imshow(img)
    axes[idx // 5, idx % 5].set_title(f"Age: {age}", fontsize=10)
    axes[idx // 5, idx % 5].axis("off") 
plt.tight_layout()
plt.show()


#preprocessing function for images
def preprocess_image(img_path, img_size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return img


#creating arrays for images and ages
images = []
ages = df['age'].values

for filename in df['filename']:
    full_path = os.path.join(img_path, filename)
    images.append(preprocess_image(full_path))

images = np.array(images)
ages = np.array(ages)

print("Images shape:", images.shape)
print("Ages shape:", ages.shape)



#data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


#EfficientNetB0 as a base model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1)
])

#compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error", metrics=["mae"])


x_train, x_val, y_train, y_val = train_test_split(images, ages, test_size=0.2, random_state=42)


#training the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),validation_data=(x_val, y_val),epochs=12, verbose=0)


#predicting ages and computing RMSE
y_pred = model.predict(x_val).flatten()

rmse = math.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")


#actual vs predicted ages
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, color='royalblue')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='dashed', linewidth=2)
plt.title('Actual vs Predicted Ages', fontsize=16)
plt.xlabel('Actual Ages', fontsize=14)
plt.ylabel('Predicted Ages', fontsize=14)
plt.grid()
plt.show()


#distribution of actual and predicted ages
plt.figure(figsize=(10, 6))
sns.histplot(y_val, color='green', label='Actual Ages', kde=True, stat='density', bins=8)
sns.histplot(y_pred, color='orange', label='Predicted Ages', kde=True, stat='density', bins=8)
plt.title('Distribution of Actual and Predicted Ages', fontsize=16)
plt.xlabel('Ages', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid()
plt.show()

