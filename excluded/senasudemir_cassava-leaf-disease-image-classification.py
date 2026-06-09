import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import cv2
import pandas as pd
import os 
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


df=pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')


df.shape


df.head()


df['image_id']='/kaggle/input/cassava-leaf-disease-classification/train_images/'+df['image_id']


df['label'].value_counts()


plt.figure(figsize=(25,8))
ax=sns.countplot(x=df["label"],palette="viridis",order=df['label'].value_counts().index)
for p in ax.containers:
    ax.bar_label(p, fontsize=20, color='black', padding=5);


x=[]
for img in df['image_id']:
    img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(32,32))
    img=img/255.0 
    x.append(img)
x=np.array(x)


y=df[["label"]]


x.shape,y.shape


from imblearn.over_sampling import SMOTE

x_flat = x.reshape(x.shape[0], -1)  
y_flat = y['label'].values 

smote = SMOTE(random_state=42)

x_resampled, y_resampled = smote.fit_resample(x_flat, y_flat)
x_resampled_images = x_resampled.reshape(-1, 32, 32, 3)


x_resampled_images.shape,y_resampled.shape


plt.figure(figsize=(25,8))
y_resampled_series = pd.Series(y_resampled)
ax=sns.countplot(x=y_resampled_series,palette="viridis")
for p in ax.containers:
    ax.bar_label(p, fontsize=12, color='black', padding=5);


x_train,x_test,y_train,y_test=train_test_split(x_resampled_images,y_resampled,test_size=.2,random_state=42)


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
model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=36,verbose=1)


model.save("cnn_model.h5")


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


accuracy_score(predictions,y_test)


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['val_accuracy'],label='Val_Accuracy')
plt.legend();


label_name = {"0":"Cassava Bacterial Blight (CBB)",
"1":"Cassava Brown Streak Disease (CBSD)",
"2":"Cassava Green Mottle (CGM)",
"3":"Cassava Mosaic Disease (CMD)",
"4":"Healthy"}

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(20, 15))
sns.heatmap(
    cm,
    annot=True,
    fmt='g',
    cmap='Blues',
    xticklabels=label_name.values(),
    yticklabels=label_name.values(),
    cbar=False
)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.show()



import random

unique_labels = np.unique(np.concatenate((y_test, predictions)))
label_name = {label: f'Class {label}' for label in unique_labels}
num_samples = 5 
sample_indices = random.sample(range(len(y_test)), num_samples)
plt.figure(figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_test[idx])  
    plt.axis('off')  
    plt.title(f"Actual: {label_name[y_test[idx]]}\nPredicted: {label_name[predictions[idx]]}")

plt.show()


image_path="/kaggle/input/cassava-leaf-disease-classification/test_images/"
img_list=[]
img_id=[]
for img in os.listdir(image_path):
    img_list.append(image_path+"/"+img)
    img_id.append(img)


df_test=pd.DataFrame({
    'img_id':img_id,
    'img_file':img_list
})


x_test=[]
for img in df_test['img_file']:
    img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(32,32))
    img=img/255.0 
    x_test.append(img)
x_test=np.array(x_test)


x_test = x_test.reshape((-1, 32, 32, 3))


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


df_test['label']=predictions


df_test.to_csv('submission.csv',index=False)

