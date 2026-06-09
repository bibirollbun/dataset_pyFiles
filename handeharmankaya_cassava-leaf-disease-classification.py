import pandas as pd
import numpy as np
import os
import cv2
import json
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D,Dropout,BatchNormalization,Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = '/kaggle/input/cassava-leaf-disease-classification'
TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train_images')
TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'train.csv')
JSON_PATH = os.path.join(BASE_DIR, 'label_num_to_disease_map.json')


df = pd.read_csv(TRAIN_CSV_PATH)


df['image_id']='/kaggle/input/cassava-leaf-disease-classification/train_images/'+df['image_id']


with open(JSON_PATH, 'r') as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}


df['class_name'] = df['label'].map(label_map)


df.head()


plt.figure(figsize=(8, 6))
sns.countplot(x=df['class_name'], palette='Spectral')
plt.xticks(rotation=45);


x = []
for img in df['image_id']:
    img=cv2.imread(str(img))
    if img is None:
        print(f"Resim yüklenemedi: {img}") 
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img, (128,128))
    img=img / 255.0
    x.append(img)


x = np.array(x)


y=df[["label"]]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


model=Sequential()
model.add(Input(shape=(128,128,3)))
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


early_stop=EarlyStopping(monitor='val_loss',patience=20)


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=36,
                  callbacks=[early_stop],verbose=1)


model.save("cnn_leaf.h5")


image_path="/kaggle/input/cassava-leaf-disease-classification/test_images/"
img_list=[]
img_id=[]
for img in os.listdir(image_path):
    img_list.append(image_path+"/"+img)
    img_id.append(img)


df_test=pd.DataFrame({'img_id':img_id,'img_file':img_list})


x_test=[]
for img in df_test['img_file']:
    img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(128,128))
    img=img/255.0 
    x_test.append(img)
x_test=np.array(x_test)


x_test = x_test.reshape((-1, 128,128, 3))


predictions=model.predict(x_test)
predictions=predictions.argmax(axis=-1)
predictions=np.array(predictions)


df_test['label']=predictions
df_test.to_csv('submission.csv',index=False)


df_test['label'] = predictions

final_submission = df_test[['img_id', 'label']]
final_submission.columns = ['image_id', 'label']  

final_submission.to_csv('submission.csv', index=False)

print(final_submission.head())

