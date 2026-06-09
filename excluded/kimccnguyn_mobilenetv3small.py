import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV3Small



image_size = 224
batch_size = 8

train_ds = tf.keras.utils.image_dataset_from_directory(
  '/kaggle/input/paddy-disease-classification/train_images',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(image_size, image_size),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  '/kaggle/input/paddy-disease-classification/train_images',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(image_size, image_size),
  batch_size=batch_size)


class_names = train_ds.class_names


val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


input_tensor_shape = (224, 224, 3)
input_tensor = Input(shape=input_tensor_shape)
pre_model=MobileNetV3Small(include_top=False,pooling='avg',input_shape = (224,224,3),weights = 'imagenet')
for layer in pre_model.layers:
    layer.trainable = False
model= Sequential()
model.add(pre_model)
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
history = model.fit(train_ds,
                    epochs=100,
                    validation_data=val_ds,
                   callbacks=[callback])


import matplotlib.pyplot as plt

true_labels = []
predicted_labels = []
misclassified_images = []
all_images = []
for image, label in test_ds:
    for l in label:
        true_labels.append(l.numpy())
    for p in model.predict(image, verbose=0):
        predicted_labels.append(np.argmax(p)) 
    for img in image:
        all_images.append(img)
        
for i in range(len(all_images)):
    if true_labels[i] != predicted_labels[i]:
        misclassified_images.append(all_images[i])

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax)
plt.xticks(rotation=90)
plt.show()



from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
def evaluate_model(test):
    results = model.evaluate(test, batch_size=32)
    return results

def Accuracy(y_test , y_pred):
    AccScore = accuracy_score(y_test, y_pred, normalize=True)
    return AccScore

def macro_precision(y_test , y_pred):
    PrecisionScore = precision_score(y_test, y_pred, average='macro') 
    return PrecisionScore

def macro_recall(y_test , y_pred):
    RecallScore = recall_score(y_test, y_pred, average='macro') 
    return  RecallScore

def macro_F1Score(y_test , y_pred):
    F1Score = f1_score(y_test, y_pred, average='macro') 
    return F1Score


def Acc_Loss_Graph(history):
    pd.DataFrame(history.history)
    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
#Calculating Area Under the Curve :  
fprValue2, tprValue2, thresholdsValue2 = roc_curve(true_labels , predicted_labels , pos_label=2 )
AUCValue = auc(fprValue2, tprValue2)



print("***** Model Evaluations Details ******** \n")
print("Test loss, Test acc : " , evaluate_model(test_ds)) 
print("******************************")
print("Accuracy  : " , Accuracy(true_labels , predicted_labels)) 
print("******************************")
print("Precision Score is : " , macro_precision(true_labels , predicted_labels))
print("******************************")
print("Recall Scores is : " , macro_recall(true_labels , predicted_labels))
print("******************************")
print('F1 Score is : ', macro_F1Score(true_labels , predicted_labels))
print("******************************")
print('AUC Value  : ', AUCValue)


Acc_Loss_Graph(history)


import numpy as np
import matplotlib.pyplot as plt
true_labels = []
predicted_labels = []
scores = []
misclassified_images = []
all_images = []
n_images = 30
for i, (image, label) in enumerate(test_ds):
    if i == n_images:
        break
    for l in label:
        true_labels.append(l.numpy())
    for p in model.predict(image, verbose=0):
        predicted_labels.append(np.argmax(p))
        scores.append(round(100 * (np.max(p)), 2))
    for img in image:
        all_images.append(img)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
scores = np.array(scores)

for i in range(len(all_images)):
    if true_labels[i] != predicted_labels[i]:
        misclassified_images.append(all_images[i])
        
for i in range(0, n_images, 2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    for j, ax in enumerate(axes):
        ax.imshow(tf.keras.preprocessing.image.array_to_img(all_images[i+j]))
        ax.set_title('True label: {}, \nPredicted label: {} \nprobability: {}%'.format(
            class_names[true_labels[i+j]], class_names[predicted_labels[i+j]], scores[i+j]))
        ax.axis('off')
    plt.show()




model.save('paddy_MobileNetV3Large.h5')

