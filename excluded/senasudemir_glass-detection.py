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


df=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-fall-2023/train.csv")


df.head()


df['file']="/kaggle/input/applications-of-deep-learning-wustl-fall-2023/"+df['file']


df.head()


df.shape


df['glasses'].value_counts()


plt.figure(figsize=(15, 5))
labels=[0,1]
for i, label in enumerate(labels):
    sample_img_path = df[df['glasses'] == label].iloc[0]['file'] 
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(label)

plt.tight_layout()
plt.show()


plt.figure(figsize=(23,8))
ax=sns.countplot(x=df["glasses"],palette="viridis",order=df['glasses'].value_counts().index)
for p in ax.containers:
    ax.bar_label(p, fontsize=12, color='black', padding=5);


x=[]
for img in df["file"]:
    img=cv2.imread(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(64,64))
    img=img/255.0
    x.append(img)


x=np.array(x)


y=df[["glasses"]]


x.shape,y.shape


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


model=Sequential()
model.add(Input(shape=(64,64,3)))
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


model.save('cnn_model.h5')


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


accuracy_score(predictions,y_test)


cm = confusion_matrix(y_test, predictions)  
label_encoding_dict = {'No Glass':0,'Glass':1}

plt.figure(figsize=(5, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='g',
    cmap='Blues',
    xticklabels=label_encoding_dict.keys(),
    yticklabels=label_encoding_dict.keys(),
    cbar=False
)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix');


label_decoding_dict = {0: 'No Glass', 1: 'Glass'}

y_test = np.array(y_test)

y_test_names = [label_decoding_dict[label.item()] for label in y_test]
predictions_names = [label_decoding_dict[label.item()] for label in predictions]

plt.figure(figsize=(15, 5))

num_samples = 10
indices = np.random.choice(len(x_test), num_samples, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx])  
    plt.axis('off')
    plt.title(f"Actual: {y_test_names[idx]}\nPredicted: {predictions_names[idx]}")

plt.tight_layout()
plt.show()



history.history['accuracy'][-1]


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['val_accuracy'],label='Val_Accuracy')
plt.legend();


df_test=pd.read_csv('/kaggle/input/applications-of-deep-learning-wustl-fall-2023/test.csv')


df_test.head()


df_test['file']="/kaggle/input/applications-of-deep-learning-wustl-fall-2023/"+df_test['file']


x_test=[]
for img in df_test["file"]:
    img=cv2.imread(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(64,64))
    img=img/255.0
    x_test.append(img)


x_test=np.array(x_test)


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


submission=pd.DataFrame({
    'id':df_test['id'],
    'glasses':predictions
})


submission.to_csv('submission.csv',index=False)

