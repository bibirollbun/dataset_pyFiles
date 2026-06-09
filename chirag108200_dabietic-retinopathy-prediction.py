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


import os
import cv2 as cv
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Load the CSV file containing image IDs and their corresponding diagnoses
df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')

# Define the base path to the training images
base_path = '/kaggle/input/aptos2019-blindness-detection/train_images'

# Helper function to process each image
def process_image(row):
    image_id = row['id_code']
    target = row['diagnosis']
    path = os.path.join(base_path, f"{image_id}.png")
    img = cv.imread(path, 1)
    if img is None:
        return []  # Skip if image is not found
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    img = img / 255.0  # Normalize pixel values
    results = [(img, target)]
    if target in (3, 4):
        flipped = cv.flip(img, 1)
        results.append((flipped, target))
    return results

# Prepare the list of image processing tasks
rows = df.to_dict('records')

# Process images in parallel using ThreadPoolExecutor
X = []
y = []
with ThreadPoolExecutor() as executor:
    for result in executor.map(process_image, rows):
        for img, label in result:
            X.append(img)
            y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)



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


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# EarlyStopping callback
early_stop = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau callback to reduce LR when val_accuracy plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.3,
    patience=3,
    verbose=1,
    mode='max',
    min_lr=1e-6
)

# Fit the model
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_val, y_val),  # Always use tuple for val data
    callbacks=[early_stop, reduce_lr],
    verbose=1
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


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    roc_curve,
    auc
)
import seaborn as sns

# Make predictions
y_pred = model.predict(X_test)

# If predictions are probabilities, convert to class labels
if hasattr(y_pred[0], '__len__'):  # i.e., if it's a 2D array from predict_proba
    y_pred_class = np.argmax(y_pred, axis=1)
else:
    y_pred_class = y_pred

# Metrics
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class)
rec = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
rmse = mean_squared_error(y_test, y_pred_class, squared=False)

# ROC-AUC
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:,1]
    auc_roc = roc_auc_score(y_test, y_proba)
else:
    auc_roc = "Not available (no predict_proba)"

# Display results
print("ðŸ“Š Evaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"AUC-ROC:   {auc_roc if isinstance(auc_roc, str) else round(auc_roc,4)}")

print("\nðŸ“‹ Classification Report:\n")
print(classification_report(y_test, y_pred_class))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
if isinstance(auc_roc, float):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


