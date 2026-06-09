import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import cv2
import pandas as pd
import os
import re
import zipfile
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


zip_path = "/kaggle/input/dogs-vs-cats/train.zip"
extract_path = "/kaggle/working/train"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


folder_path = extract_path + "/train"
labels=[]
img=[]
for f in os.listdir(folder_path):
    labels.append(f[:3])
    img.append(folder_path+'/'+f)


df=pd.DataFrame({
    'img':img,
    'label':labels
})


df.sample(10)


sample_size = 10  

images = []
for label in labels[:sample_size]:  
    img_path = df[df['label'] == label]['img'].iloc[0]
    image = Image.open(img_path).resize((128, 128))  
    images.append((image, label))

images_per_row = 5
num_rows = math.ceil(len(images) / images_per_row)

plt.figure(figsize=(12, num_rows * 3)) 
for i, (image, label) in enumerate(images):
    plt.subplot(num_rows, images_per_row, i + 1) 
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.show()


label_encoding={'dog':1,'cat':0}
df['label_encoding']=df['label'].map(label_encoding)


df.sample(5)


plt.figure(figsize=(23,8))
ax=sns.countplot(x=df["label"],palette="viridis",order=df['label'].value_counts().index)
for p in ax.containers:
    ax.bar_label(p, fontsize=12, color='black', padding=5)
plt.xticks(rotation=90);


x=[]
for img in df['img']:
    img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(32,32))
    img=img/255.0 
    x.append(img)
x=np.array(x)


y=df[["label_encoding"]]


x.shape,y.shape


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


model=Sequential()
model.add(Input(shape=(32,32,3)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=36,verbose=1)


model.save("cnn_model.h5")


history.history['accuracy'][-1]


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['val_accuracy'],label='Val_Accuracy')
plt.legend();


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


accuracy_score(predictions,y_test)


cm = confusion_matrix(y_test, predictions)

label_name = ['Dog', 'Cat'] 

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='g',
    cmap='Blues',
    xticklabels=label_name,
    yticklabels=label_name,
    cbar=False
)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix');


random_indices = np.random.choice(len(x_test), size=min(10, len(x_test)), replace=False)

x_test_resized = x_test[random_indices]
y_test_array = np.array(y_test)
actual_labels = y_test_array[random_indices].flatten()

predictions = model.predict(x_test_resized)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx])  
    plt.title(f"Actual: {labels[actual_labels[i]]}\nPredicted: {labels[predicted_labels[i]]}", fontsize=10)
    plt.axis('off')  
plt.tight_layout()
plt.show()


zip_path = "/kaggle/input/dogs-vs-cats/test1.zip"
extract_path = "/kaggle/working/test1"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


folder_path = extract_path + "/test1"
img=[]
ids=[]
for f in os.listdir(folder_path):
    img.append(folder_path+'/'+f)


folder_path = extract_path + "/test1"
img = []
ids = []

for f in os.listdir(folder_path):
    img.append(folder_path + '/' + f)  

    match = re.search(r"(\d+)", f)  
    if match:
        ids.append(int(match.group(1)))


df_test=pd.DataFrame({
    'id':ids,
    'img':img
})


df_test.head()


df_test.info()


df_test= df_test.sort_values(by='id', ascending=True)


df_test.head()


x_test=[]
for img in df_test['img']:
    img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(32,32))
    img=img/255.0 
    x_test.append(img)
x_test=np.array(x_test)


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


submission=pd.DataFrame({
    'id':df_test['id'],
    'label':predictions
})


submission.head()


submission.to_csv('submission.csv',index=False)

