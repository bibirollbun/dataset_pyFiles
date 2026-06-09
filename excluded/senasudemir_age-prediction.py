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
from sklearn.metrics import r2_score
from tensorflow.keras import regularizers


df_train=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/train.csv")


df_train.head()


df_train.isnull().sum()


df_train.shape


df_train['filename']="/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/"+df_train['filename']


df_train.head()


df_train['age'].value_counts()


plt.figure(figsize=(15, 5))

for i in range(10):
    sample_img_path = df_train.iloc[i]['filename'] 
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(df_train.iloc[i]['age'])

plt.tight_layout()
plt.show()


plt.figure(figsize=(35,15))
ax=sns.countplot(x=df_train["age"],palette="viridis",order=df_train['age'].value_counts().index)
for p in ax.containers:
    ax.bar_label(p, fontsize=15, color='black', padding=5)
plt.xticks(rotation=90);


x=[]
for img in df_train['filename']:
    img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(128,128))
    img=img/255.0
    x.append(img)
x=np.array(x)


y=df_train[["age"]]


x.shape,y.shape


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)


model = Sequential()

model.add(Input(shape=(128, 128, 3)))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(1) ) 

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])


model.summary()



history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=36,verbose=1)


model.save("cnn_model.h5")


history.history['mae'][-1] 


plt.figure(figsize=(10, 5))

# Plot Mean Squared Error (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()

# Plot Mean Absolute Error (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Training vs Validation MAE")
plt.legend()

plt.tight_layout()
plt.show()


df_test=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/test.csv")


df_test['filename']="/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/"+df_test['filename']


df_test.head()


x_test=[]
for img in df_test['filename']:
    img=cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(128,128))
    img=img/255.0
    x_test.append(img)
x_test=np.array(x_test)


predictions = model.predict(x_test)  
predictions = np.round(predictions).astype(int)  
predictions = predictions.flatten()


predictions


submission=pd.DataFrame({
    'id':df_test['id'],
    'age':predictions
})


submission.head()


submission.to_csv('submission.csv',index=False)

