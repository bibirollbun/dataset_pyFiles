#pip install opencv-python


#!pip install scikit-learn


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
from sklearn.model_selection import train_test_split
#from IPython.display import Image

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Reshape, GlobalAveragePooling2D,Activation
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

warnings.filterwarnings("ignore", category=DeprecationWarning)


img_path= '/content/data/train'

zip_path = "/kaggle/input/dogs-vs-cats/train.zip"
extract_path = "/kaggle/working/train"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

img_path = extract_path + "/train"
label_list=[]
img_list=[]

for f in os.listdir(img_path):
    label_list.append(f[:3])
    img_list.append(img_path+'/'+f)
	


df = pd.DataFrame({'img':img_list,'label':label_list})
df.head()


selected_images = df.groupby('label', as_index=False).apply(lambda x: x.sample(n=3, random_state=1)).reset_index(drop=True)

sns.set(style='whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(5, 5))

axes = axes.flatten()

for ax, (img_path, label) in zip(axes, zip(selected_images['img'], selected_images['label'])):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label, fontsize=14)

plt.tight_layout()
plt.show()


label_values = df['label'].value_counts()
plt.figure(figsize=(3, 3))
plt.pie(label_values, labels=label_values.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.show()


labelEncode = {'cat':0,'dog':1}
df['encoded_label'] = df['label'].map(labelEncode)


df.head()


img_width, img_height = 170, 170


x = []
y = []

for img_path, label in zip(df['img'], df['encoded_label']):
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    x.append(img)
    y.append(label)


x = np.array(x)
y = np.array(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = Sequential()
model.add(Input(shape=(img_width,img_height, 3)))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']);


early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])


model.summary()


history_df = pd.DataFrame(history.history)
history_df


ax = history_df[['accuracy','val_accuracy']].plot(title = "Train and Validation Accuracies" , marker='o')
ax.set(xlabel ="Epochs", ylabel = "Accuracy")
plt.show()


model.save('Cat_Dog.keras')


img_size = (img_width, img_height)
test_images = df.sample(n=12)

images = []
predicted_classes = []
actual_classes = []

for index, row in test_images.iterrows():
    img_path = row['img']
    actual_class = row['encoded_label']
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    images.append(img)
    predicted_classes.append(predicted_class[0])
    actual_classes.append(actual_class)

class_labels = {0: 'Cat', 1: 'Dog'}

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


img_path= '/content/data/test1'

zip_path = "/kaggle/input/dogs-vs-cats/test1.zip"
extract_path = "/kaggle/working/test1"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

img_path = extract_path + "/test1"
id_list=[]
img_list=[]

for f in os.listdir(img_path):
    match = re.search(r"(\d+)", f)
    if match:
        id_list.append(int(match.group(1)))
    img_list.append(img_path+'/'+f)




df_test = pd.DataFrame({'img':img_list,'id':id_list})
df_test.head()

df_test= df_test.sort_values(by='id', ascending=True)


img_width, img_height = 170, 170


x = []

for img_path in df_test['img']:
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    x.append(img)


x_test = np.array(x)


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


submission=pd.DataFrame({
    'id':df_test['id'],
    'label':predictions
})


submission.head()


submission.to_csv('submission.csv',index=False)

