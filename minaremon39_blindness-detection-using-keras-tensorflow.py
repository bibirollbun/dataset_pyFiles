import os
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
warnings.filterwarnings("ignore")
sns.set_style(style="darkgrid")


df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
print(df.shape)
df.head()


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


X = []
y = []
for target in range(5):
    for image in df[df["diagnosis"]==target]["id_code"]:
        path = os.path.join("/kaggle/input/aptos2019-blindness-detection/train_images",f"{image}.png")
        img = cv.imread(path,1)
        img = cv.resize(img,(224,224))
        img = img/255.0
        X.append(img)
        y.append(target)
        if target in (3,4) :
            X.append(cv.flip(img,1))
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


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0,shuffle=True)
x_test,x_val,y_test,y_val = train_test_split(x_test,y_test,test_size=0.5,random_state=0,shuffle=True)
del X,y


model = Sequential()

# Input Layer
model.add(keras.layers.InputLayer(shape=(224,224,3)))


# CNN Layer1   
model.add(keras.layers.Conv2D(32,(3,3),padding="valid",activation="relu")) 
model.add(keras.layers.MaxPool2D((2,2),strides=2)) 

# CNN Layer2
model.add(keras.layers.Conv2D(64,(3,3),padding="valid",activation="relu"))
model.add(keras.layers.MaxPool2D((3,3),strides=3))

# CNN Layer3
model.add(keras.layers.Conv2D(128,(3,3),padding="valid",activation="relu"))
model.add(keras.layers.MaxPool2D((3,3),strides=3))

# CNN Layer4
model.add(keras.layers.Conv2D(256,(3,3),padding="valid",activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.MaxPool2D((3,3),strides=3))

# Flatten Layer
model.add(keras.layers.Flatten())

# Fully Connected layer
model.add(keras.layers.Dense(256,activation="relu"))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dense(5,activation="softmax"))

# Compile
model.compile(loss=SparseCategoricalCrossentropy(),metrics=["accuracy"],optimizer="adam")

# Summary for model
model.summary()


es = EarlyStopping(monitor='val_accuracy',
                   mode="max",
                   verbose=1,
                   patience=6,
                   restore_best_weights=True)


history = model.fit(x_train,y_train,
                    epochs=25,batch_size=190,
                    validation_data=[x_val,y_val],
                    callbacks=[es])


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


y_predicted = model.predict(x_test)
y_predicted = np.argmax(y_predicted,axis=1)
cm = confusion_matrix(y_test,y_predicted)
cmd = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["No DR","Mild","Moderate","Severe","Proliferative DR"])
cmd.plot(cmap=plt.cm.Blues, values_format='d',xticks_rotation="vertical")
plt.show()


acc = accuracy_score(y_test,y_predicted)
print(f"Test score: {acc*100:.2f}")


decode = {0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative DR"}
plt.figure(figsize=(15,5))
for i,test in enumerate(np.random.randint(0,len(df)-1,8),1):
    plt.subplot(2,4,i)
    image = df.loc[test,"id_code"]
    path = os.path.join("/kaggle/input/aptos2019-blindness-detection/train_images",f"{image}.png")
    img = cv.imread(path,1)
    img = cv.resize(img,(224,224))
    plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    res = model.predict(img)
    res = f"Prediction is {decode[np.argmax(res)]}"
    true = f"Real is {decode[df.loc[test,'diagnosis']]}"
    plt.axis("off")
    plt.text(x=10,y=20,s=true,color="Blue",weight="bold",backgroundcolor="Gray",size=8)
    plt.text(x=10,y=215,s=res,color="Blue",weight="bold",backgroundcolor="Gray",size=8)
plt.show()


model.save("Blind_DetectionV1.keras")


del x_train,x_test,x_val,y_train,y_test,y_val,history,df


df2 = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")
df2.head()


diagnosis = []
for j,i in enumerate(df2["id_code"],1):
    path=os.path.join("/kaggle/input/aptos2019-blindness-detection/test_images",f"{i}.png")
    img = cv.imread(path,1)
    img = cv.resize(img,(224,224))
    img = img/255.0
    diagnosis.append(img)
    if j%500==0:
        print("Done of 500 images")
diagnosis = np.asarray(diagnosis)


diagnosis = model.predict(diagnosis)
diagnosis = np.argmax(diagnosis,axis=1)
df2["diagnosis"]=diagnosis
df2.head()


df2.to_csv(path_or_buf="submission.csv",index=False)

