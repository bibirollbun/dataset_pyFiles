import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
from sklearn.datasets import load_files
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import layers
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image                  
from tqdm import tqdm

import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")


data_dir = "../input/state-farm-distracted-driver-detection/imgs"
train_dir = os.path.join(data_dir,"train")
test_dir = os.path.join(data_dir,"test")
model1_path = os.path.join(os.getcwd(),"model1","self_trained")
pickle_dir = os.path.join(os.getcwd(),"pickle_files")
csv_dir = os.path.join(os.getcwd(),"csv_files")


img_size = (224, 224) 
batch_size = 32
seed = 42

train_datagen = ImageDataGenerator(
    rescale=1./255,         
    validation_split=0.2,   
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    seed=seed
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    seed=seed
)

# test_generator = test_val_datagen.flow_from_directory(
#     directory=os.path.dirname(test_dir),                # parent dir
#     classes=['test'],         # only test folder
#     target_size=(224,224),
#     batch_size=32,
#     class_mode=None,          # no labels
#     shuffle=False
# )
#print("\nClass indices:", train_generator.class_indices)


test_dir = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"


test_files = os.listdir(test_dir)
test_df = pd.DataFrame({"filename": test_files})

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_dir,
    x_col="filename",
    y_col=None,          # no labels in test set
    target_size=(224,224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)



activity_map = {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}


images, labels = next(train_generator)
plt.figure(figsize=(14, 14))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    label_index = np.argmax(labels[i])
    class_code = list(train_generator.class_indices.keys())[label_index]
    activity_name = activity_map[class_code]
    plt.title(f"{class_code}: {activity_name}", fontsize=10)
    plt.axis("off")
plt.show()


class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) 
                for cls in os.listdir(train_dir)}
df_counts = pd.DataFrame.from_dict(class_counts, orient='index', columns=['count'])
df_counts.plot(kind='bar', figsize=(10,5), legend=False)
plt.title("Number of Images per Class in Training Set")
plt.xlabel("Class")
plt.ylabel("Image Count")
plt.show()


from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

conv_base.trainable = True


conv_base.summary()


set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


conv_base.summary()


model_1 = models.Sequential()
model_1.add(conv_base)
model_1.add(layers.Flatten())
model_1.add(layers.Dense(512, activation='relu'))
model_1.add(layers.Dense(64, activation='relu'))
model_1.add(Dense(10, activation='softmax'))


model_1.summary()


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
checkpoint = ModelCheckpoint(
    filepath="model1/best_model.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,          
    verbose=1
)


from tensorflow.keras import optimizers
model_1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=1e-5),
              metrics=['acc'])


history_1 = model_1.fit(
    train_generator,
    steps_per_epoch=len(train_generator), 
    validation_data=val_generator,
    validation_steps=len(val_generator),         
    epochs=30,
    callbacks=[early_stop, checkpoint]
)


acc = history_1.history['acc']
val_acc = history_1.history['val_acc']
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


model_1.load_weights("model1/best_model.weights.h5")
preds=model_1.predict(test_generator, verbose=1)  # shape: (79726, 10)



filenames = test_generator.filenames
filenames = [f.split('/')[-1] for f in filenames] 


submission = pd.DataFrame(preds, columns=[f'c{i}' for i in range(10)])
submission.insert(0, 'img', filenames)

submission.to_csv("submission.csv", index=False)


