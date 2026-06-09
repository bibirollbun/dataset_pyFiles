#pip install opencv-python


# !pip install -U scikit-learn==1.3.2 imbalanced-learn==0.11.0 --quiet


# !pip install --upgrade tensorflow


import os
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import re
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Reshape, GlobalAveragePooling2D,Activation
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


warnings.filterwarnings("ignore")


df=pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
df.head()


img_path = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
df['image_id'] = df['image_id'].apply(lambda x: img_path + x)


df.rename(columns={'image_id': 'img'}, inplace=True)


df.img[0]


df.label.value_counts()


selected_images = df.groupby('label', as_index=False).apply(lambda x: x.sample(n=3, random_state=1)).reset_index(drop=True)

sns.set(style='whitegrid')
fig, axes = plt.subplots(5, 3, figsize=(15, 8))

axes = axes.flatten()

for ax, (img_path, label) in zip(axes, zip(selected_images['img'], selected_images['label'])):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label, fontsize=14)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df)

# Add titles and labels
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')

# Display the plot
plt.show()


img_width, img_height = 64, 64


x = []
y = []

for img_path, label in zip(df['img'], df['label']):
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    x.append(img)
    y.append(label)


x = np.array(x)
y=df[["label"]]


x_reshaped = x.reshape(x.shape[0], -1)  
y_reshaped = y['label'].values 

smote = SMOTE(random_state=42)

x_resampled, y_resampled = smote.fit_resample(x_reshaped, y_reshaped)
x_resampled_images = x_resampled.reshape(-1, img_width, img_height, 3)


plt.figure(figsize=(20,5))
y_resampled_series = pd.Series(y_resampled)
ax=sns.countplot(x=y_resampled_series)


x_train,x_test,y_train,y_test=train_test_split(x_resampled_images,y_resampled,test_size=.2,random_state=42)


model = Sequential()
model.add(Input(shape=(img_width, img_height, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(df.label.unique()),activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

history = model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])


model.summary()


history_df = pd.DataFrame(history.history)
history_df


ax = history_df[['accuracy','val_accuracy']].plot(title = "Train and Validation Accuracies" , marker='o')
ax.set(xlabel ="Epochs", ylabel = "Accuracy")
plt.show()


model.save('leaf_disease.keras')


img_size = (img_width, img_height)
test_images = df.sample(n=12)

images = []
predicted_classes = []
actual_classes = []

for index, row in test_images.iterrows():
    img_path = row['img']
    actual_class = row['label']
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    images.append(img)
    predicted_classes.append(predicted_class[0])
    actual_classes.append(actual_class)

class_labels = {
    0: "CBB",
    1: "CBSD",
    2: "CGM",
    3: "CMD",
    4: "Healthy"
}


num_images = len(images)
cols = 3
rows = (num_images // cols) + (num_images % cols > 0)

plt.figure(figsize=(15, rows * 5))
for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    plt.title(f'Actual: {class_labels[actual_classes[i]]}\nPredicted: {class_labels[predicted_classes[i]]}')
    plt.axis('off')

plt.tight_layout()
plt.show()


img_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
img_list=[]
img_id=[]
for img in os.listdir(img_path):
    img_list.append(img_path+"/"+img)
    img_id.append(img)


df_test=pd.DataFrame({
    'img_id':img_id,
    'img_file':img_list
})


x_test=[]
for img in df_test['img_file']:
    img=cv2.imread(img)
    img=cv2.resize(img,(img_width,img_height))
    img=img/255.0 
    x_test.append(img)
x_test=np.array(x_test)
x_test = x_test.reshape((-1, img_width, img_height, 3))


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


df_test['label']=predictions


df_test.to_csv('submission.csv',index=False)

