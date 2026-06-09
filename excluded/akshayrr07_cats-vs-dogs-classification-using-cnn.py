from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras import layers
from keras import models 
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")


print(os.listdir("/kaggle/input/dogs-vs-cats"))


import zipfile
zip_files = ['test1','train']
for zip_file in zip_files:
    with zipfile.ZipFile(f'/kaggle/input/dogs-vs-cats/{zip_file}.zip','r') as file:
        file.extractall(".")
        print("{} unzipped successfully".format(zip_file))


print(os.listdir("."))
train_path="./train"
file_names=os.listdir(train_path)
print("train samples ",len(file_names))
print(file_names[:3])


test_path="./test1"
file_names=os.listdir(test_path)
print("test samples ",len(file_names))
print(file_names[:3])


def path_to_frame(files):
    paths,labels=[],[]
    path=f"./{files}"
    file_names=os.listdir(path)
    for name in file_names:
        paths.append(name)
        labels.append(name[:3])
    return pd.DataFrame({"path":paths,"target":labels})


train_df=path_to_frame("train")
test_df=path_to_frame("test1")
print("train shape",train_df.shape)
print("test shape", test_df.shape)


print("train: ")
print(train_df.head(2))
print()
print("test: ")
print(test_df.head(2))


import matplotlib.pyplot as plt
import seaborn as sns 

sns.countplot(x=train_df["target"],alpha=0.2,color="g")
plt.show()


from sklearn.model_selection import train_test_split

train_df,valid_df=train_test_split(train_df,test_size=0.2,stratify=train_df["target"],random_state=42)


fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
sns.countplot(x=train_df["target"],alpha=0.2,color="g",ax=ax[0])
ax[0].set_title("Train data")
sns.countplot(x=valid_df["target"],alpha=0.2,color="g",ax=ax[1])
ax[1].set_title("Validition data")
plt.show()


train_datagen=ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
training_data = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='./train',
    x_col = 'path',
    y_col='target',
    target_size=(150,150),
    class_mode='binary',
    batch_size=50) 


valid_datagen=ImageDataGenerator(rescale=1/255)

validation_data = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory='./train',
    x_col = 'path',
    y_col='target',
    target_size=(150,150),
    class_mode='binary',
    batch_size=50) 


test_df=test_df.drop("target",axis=1)
test_datagen=ImageDataGenerator(rescale=1/255)

test_data = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='./test1',
    x_col = 'path',
    y_col=None,
    target_size=(150,150),
    class_mode=None,
    batch_size=50) 


counts, idx = 5, 0
print("Training samples",end="\n\n")
fig, ax = plt.subplots(1, counts, figsize=(15, 5))
for img_batch, _ in training_data:  # Unpack batch to separate images and labels
    # Display one sample image from the batch
    ax[idx].imshow(img_batch[0])
    ax[idx].axis("off") 
    idx += 1
    if idx == counts:
        break

plt.show()


counts, idx = 5, 0
print("Validation samples",end="\n\n")
fig, ax = plt.subplots(1, counts, figsize=(15, 5))
for img_batch, _ in validation_data:  # Unpack batch to separate images and labels
    # Display one sample image from the batch
    ax[idx].imshow(img_batch[0])
    ax[idx].axis("off") 
    idx += 1
    if idx == counts:
        break

plt.show()


counts, idx = 5, 0
print("Test samples",end="\n\n")
fig, ax = plt.subplots(1, counts, figsize=(15, 5))
for img_batch in test_data:  # Unpack batch to separate images and labels
    # Display one sample image from the batch
    ax[idx].imshow(img_batch[0])
    ax[idx].axis("off") 
    idx += 1
    if idx == counts:
        break

plt.show()


model=models.Sequential(
[
    layers.Conv2D(32,(3,3),input_shape=(150,150,3,),activation="relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Dropout((0.5)),

    layers.Conv2D(128,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dropout((0.5)),
    layers.Dense(512,activation="relu"),
    layers.Dense(1,activation="sigmoid")
])
model.summary()


model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
keras.utils.plot_model(model,show_layer_activations=True,show_layer_names=True,show_shapes=True,show_dtype=True)


history=model.fit(
    training_data,
    steps_per_epoch=train_df.shape[0]//32,
    epochs=10,
    validation_data= validation_data, 
    validation_steps= valid_df.shape[0]// 32,
)


def plot_model_parameters(history):
    acc=history["accuracy"]
    val_acc=history["val_accuracy"]
    loss=history["loss"]
    val_loss=history["val_loss"]
    fig, ax = plt.subplots(2, 1, figsize = (8, 10))

    epochs = range(1, len(acc) + 1)
    ax[0].plot(epochs, acc, '--', label='Training acc')
    ax[0].plot(epochs, val_acc, label='Validation acc')
    ax[0].set_title('Training and validation accuracy')
    ax[0].set_ylim([0.0, 1.0])
    ax[0].legend()

    ax[1].plot(epochs, loss, '--', label='Training loss')
    ax[1].plot(epochs, val_loss, label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    
    plt.show()
    plot_model_parameters(history.history)


from keras.models import Sequential
from keras.applications.vgg16 import VGG16

model2 = Sequential([
    VGG16(include_top=False, pooling='max', input_shape=(150, 150, 3), weights='imagenet'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Freeze the VGG16 model layers to prevent training
model2.layers[0].trainable = False

model2.summary()


model2.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


history=model2.fit(
    training_data,
    steps_per_epoch=train_df.shape[0]//32,
    epochs=10,
    validation_data= validation_data, 
    validation_steps= valid_df.shape[0]// 32,
)


import matplotlib.pyplot as plt

# Function to plot model training parameters
def plot_model_parameters(history_dict):
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Correct usage
plot_model_parameters(history.history)

