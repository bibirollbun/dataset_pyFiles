import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os


# !pip install -U -q PyDrive

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

# # 1. Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)


# from google.colab import drive
# drive.mount('/content/drive')


# ! mkdir ~/.kaggle


# !cp /content/drive/MyDrive/CollabData/kaggle_API/kaggle.json ~/.kaggle/kaggle.json


# ! chmod 600 ~/.kaggle/kaggle.json


# ! kaggle competitions download challenges-in-representation-learning-facial-expression-recognition-challenge


# !unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip


data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')


data.shape


data.head()


np_data = np.array(data)


np_data.shape


np_data[0]


test_image = np_data[0][2].split()
print(type(test_image))
print(len(test_image))


test_image = np.array(test_image).reshape(48, 48)
test_image.shape


test_image = test_image.astype('float')


test_image


plt.imshow(test_image)


import os
import numpy as np
from PIL import Image

for i in range(len(data)):
    directory = 'data/' + np_data[i][1] + '/' + str(np_data[i][0])
    os.makedirs(directory, exist_ok=True)

    test_image = np_data[i][2].split(" ")
    test_image = np.array(test_image).reshape(48, 48)
    test_image = test_image.astype('float')

    # Use Pillow to save as JPG
    image = Image.fromarray(test_image.astype(np.uint8))
    image.save(directory + '/' + str(i) + '.jpg')


!ls data/


!ls data/Training


from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import losses, optimizers, metrics


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = 'data/Training'
validation_dir = 'data/PublicTest'
test_dir ='data/PrivateTest'


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (48, 48),
    batch_size=32
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (48, 48),
    batch_size=32
)


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.summary()


model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.categorical_crossentropy,
              metrics=['accuracy']
             )

history = model.fit(
    train_generator,
    steps_per_epoch = 900,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=115
)


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training losses')
plt.plot(epochs, val_loss, 'b', label='Validation losses')
plt.title('Training and validation loss')
plt.legend()

plt.show()


from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(48, 48, 3))


model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.summary()


conv_base.trainable = False
print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

model.compile(
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]
)

history = model.fit(train_generator,
                              steps_per_epoch=900,
                              epochs=15,
                              validation_data=validation_generator,
                              validation_steps=115)


conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


model.compile(
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]
)

history = model.fit(train_generator,
                              steps_per_epoch=900,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=115)


import matplotlib.pyplot as plt

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training losses')
plt.plot(epochs, val_loss, 'b', label='Validation losses')
plt.title('Training and validation loss')
plt.legend()

plt.show()




