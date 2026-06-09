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



import os
import random
import warnings
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam

# Added for InceptionV3 
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model\

# Set seed for reproducibility 
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)


warnings.filterwarnings("ignore")
sns.set_style(style="darkgrid")

def apply_clahe_lab(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return final


# **Read Data**"""

df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
print(df.shape)
df.head()

## Data consist of two columns, first column have name of image and second column have classify of diagnosis of image**


## **Visualize Distribution**


data = df.replace({"diagnosis":{0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative DR"}})
diagnosis_count = data.diagnosis.value_counts()
sns.countplot(data=data,x="diagnosis",order=diagnosis_count.index)
plt.xlabel("Diagnosis",weight="bold",size=15)
plt.ylabel("Freq",weight="bold",size=15)
for i,v in enumerate(diagnosis_count.values,0):
    text = f"{v*100/len(data):0.2f}%"
    plt.text(s=text,x=i,y=v+10,ha="center",weight="bold")
plt.show()
del data


## **Read Image & preprocessing**

X = []
y = []
for target in range(5):
    for image in df[df["diagnosis"] == target]["id_code"]:
        path = os.path.join("/kaggle/input/aptos2019-blindness-detection/train_images", f"{image}.png")
        img = cv.imread(path, 1)
        img = apply_clahe_lab(img)  # Apply CLAHE before resizing
        img = cv.resize(img, (299, 299))  # Resize after enhancement
        img = preprocess_input(img)  
        X.append(img)
        y.append(target)
        if target in (3, 4):
            flipped = cv.flip(img, 1)
            X.append(flipped)
            y.append(target)

X = np.asarray(X)
y = np.asarray(y)

data = pd.Series(y)
data = data.replace({0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative DR"})
diagnosis_count = data.value_counts()
sns.countplot(x=data,order=diagnosis_count.index)
plt.xlabel("Diagnosis",weight="bold",size=15)
plt.ylabel("Freq",weight="bold",size=15)
for i,v in enumerate(diagnosis_count.values,0):
    text = f"{v*100/len(data):0.2f}%"
    plt.text(s=text,x=i,y=v+10,ha="center",weight="bold")
plt.show()
del data

## **Splitting**"""

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0,shuffle=True)
x_test,x_val,y_test,y_val = train_test_split(x_test,y_test,test_size=0.5,random_state=0,shuffle=True)
del X,y

# Print dataset
print("after increasing data of class 3,4")
print(f"Total samples: {len(x_train) + len(x_val) + len(x_test)}")  # ✅ Added total count
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")






# **Workin On Model**

## **Building Model**


# Replaced custom CNN with InceptionV3
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

# Freeze base model (optional: change to True for fine-tuning)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Summary for model
model.summary()

# model.add(keras.layers.BatchNormalization())


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=360,
    
    horizontal_flip=True,
    fill_mode='nearest'
)


## **Model Training**"""

es = EarlyStopping(monitor='val_accuracy',
                   mode="max",
                   verbose=1,
                   patience=6,
                   restore_best_weights=True)


history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=25,
    validation_data=(x_val, y_val),
    callbacks=[es]
)


plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.title("Accuracy")
plt.plot(history.history["val_accuracy"],label="val accuracy")
plt.plot(history.history["accuracy"],label="accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.title("Loss")
plt.plot(history.history["val_loss"],label="val loss")
plt.plot(history.history["loss"],label="loss")
plt.legend()
plt.show()

## **Model Evaluation**"""

y_predicted = model.predict(x_test)
y_predicted = np.argmax(y_predicted,axis=1)
cm = confusion_matrix(y_test,y_predicted)
cmd = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["No DR","Mild","Moderate","Severe","Proliferative DR"])
cmd.plot(cmap=plt.cm.Blues, values_format='d',xticks_rotation="vertical")
plt.show()

acc = accuracy_score(y_test,y_predicted)
print(f"Test score: {acc*100:.2f}")

## **Random Test**"""

decode = {0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative DR"}
plt.figure(figsize=(15,5))
for i,test in enumerate(np.random.randint(0,len(df)-1,8),1):
    plt.subplot(2,4,i)
    image = df.loc[test,"id_code"]
    path = os.path.join("/kaggle/input/aptos2019-blindness-detection/train_images",f"{image}.png")
    img = cv.imread(path,1)
    img = cv.resize(img, (299, 299))  # Resize after enhancement
        
    plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    res = model.predict(img)
    res = f"Prediction is {decode[np.argmax(res)]}"
    true = f"Real is {decode[df.loc[test,'diagnosis']]}"
    plt.axis("off")
    plt.text(x=10,y=20,s=true,color="Blue",weight="bold",backgroundcolor="Gray",size=8)
    plt.text(x=10,y=215,s=res,color="Blue",weight="bold",backgroundcolor="Gray",size=8)
plt.show()

## **Save Model**"""

model.save("Blind_DetectionV1.keras")

del x_train,x_test,x_val,y_train,y_test,y_val,history,df


# **Submit file**"""

df2 = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")
df2.head()

diagnosis = []
for j,i in enumerate(df2["id_code"],1):
    path=os.path.join("/kaggle/input/aptos2019-blindness-detection/test_images",f"{i}.png")
    img = cv.imread(path,1)
    img = apply_clahe_lab(img)
    img = cv.resize(img,(299,299))
    img = preprocess_input(img)
    diagnosis.append(img)
    if j%500==0:
        print("Done of 500 images")
diagnosis = np.asarray(diagnosis)

diagnosis = model.predict(diagnosis)
diagnosis = np.argmax(diagnosis,axis=1)
df2["diagnosis"]=diagnosis
df2.head()

df2.to_csv(path_or_buf="submission.csv",index=False)

