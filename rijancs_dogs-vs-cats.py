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


import shutil, pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd


import os 

data_dir = '/kaggle/input/dogs-vs-cats'
print(os.listdir(data_dir))


!unzip -qq /kaggle/input/dogs-vs-cats/train.zip


import matplotlib.pyplot as plt
from PIL import Image
plt.figure(figsize = (15,15))

image =  os.listdir('/kaggle/working/train')

for i in range(36):
    plt.subplot(6,6,i+1)
    img = Image.open(os.path.join('/kaggle/working/train',image[i]))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image {i+1}")
plt.tight_layout()
plt.show()


original_dir = pathlib.Path('train')
new_base_dir = pathlib.Path('cats_vs_dogs_small')

def make_subset(subset_name, start_index, end_index):
    for category in ('cat', 'dog'):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir, exist_ok = True)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src = original_dir/fname, dst = dir / fname)

make_subset("train", start_index = 0, end_index = 1000)
make_subset("validation", start_index = 1000, end_index = 1500)
make_subset("test", start_index = 1500, end_index = 2500)


#Understanding tf.data.Dataset

random_numbers = np.random.normal(size = (1000, 16))
dataset = tf.data.Dataset.from_tensor_slices(random_numbers)

dataset = dataset.shuffle(100).batch(32)
dataset = dataset.map(lambda x: tf.reshape(x, (16,32)))
for i, element in enumerate(dataset):
    print(element.shape)
    if i >=2:
        break


#Creating the dataset from the directory
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

root = pathlib.Path('cats_vs_dogs_small')

train_dataset = image_dataset_from_directory(
    root /'train',
    image_size = (180,180),
    batch_size = 32,
    seed = 125
)

plt.figure(figsize = (15, 15))
for i, element in enumerate(train_dataset):
    element = element[0].numpy()
    plt.subplot(6,6,i+1)
    plt.imshow(element[i].astype('uint8'))
    plt.axis('off')
    plt.title(f"Image {i+1}")
    if i>=24:
        break
plt.tight_layout()
plt.show()


validation_dataset = image_dataset_from_directory(
    root / 'validation',
    image_size = (180,180),
    batch_size = 32
)

test_dataset = image_dataset_from_directory(
    root / "test",
    image_size = (180,180),
    batch_size = 32
)


inputs = keras.Input(shape = (180,180,3), name = "Input for Classification of Image")
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu')(x)
flatten = layers.Flatten()(x)
outputs = layers.Dense(1, activation = 'sigmoid')(flatten)
model = keras.Model(inputs, outputs)


model.summary()


model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)


modelCheckPoint = keras.callbacks.ModelCheckpoint(
    filepath = 'convnet_from_scratch.keras',
    save_best_only = True, 
    monitor = 'val_loss'
)

history = model.fit(
    train_dataset, 
    epochs = 30,
    validation_data = validation_dataset,
    callbacks = [modelCheckPoint]
)

test_loss, test_acc = model.evaluate(test_dataset)
print("The accuracy on test_images is ", test_acc )


history = history.history
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, "bo", label = "Training Accuracy")
plt.plot(epochs, val_accuracy, "b", label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()


loss = history['loss']
val_loss = history['val_loss']
plt.plot(epochs, loss, "bo", label = "Training Loss")
plt.plot(epochs, val_loss, "b", label = "Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()


loaded_model = keras.models.load_model('convnet_from_scratch.keras')
predictions = loaded_model.evaluate(test_dataset)

loaded_model.summary()


data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


plt.figure(figsize=(10, 10)) 
for images, _ in train_dataset.take(1):                           
    for i in range(9):
        augmented_images = data_augmentation(images)              
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))   
        plt.axis("off")


inputs = keras.Input(shape = (180,180,3), name ="Dogs_vs_Cats Input")
augmented_data = data_augmentation(inputs)
x = layers.Rescaling(1./255)(augmented_data)
x = layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu')(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)


#Training the model
modelCheckPoint = keras.callbacks.ModelCheckpoint(
    filepath = 'convnet_with_aug_from_scratch.keras',
    save_best_only = True,
    monitor = 'val_loss'
)

history = model.fit(
    train_dataset, 
    epochs = 100, 
    validation_data = validation_dataset,
    callbacks =  [modelCheckPoint]
)


accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, "bo", label = "Training Accuracy")
plt.plot(epochs, val_accuracy, "b", label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, "bo", label = "Training loss")
plt.plot(epochs, val_loss, "b", label = "Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation loss")
plt.legend()
plt.show()


loaded_model = keras.models.load_model(
    'convnet_with_aug_from_scratch.keras'
)
test_loss, test_acc = loaded_model.evaluate(test_dataset)
print("Test Accuracy: ", test_acc)


import tensorflow as tf
from tensorflow import keras

conv_base = keras.applications.vgg16.VGG16(
    weights = 'imagenet',
    include_top = False, 
    input_shape = (180,180,3)
)

conv_base.summary()


import numpy as np

def get_features_and_labels(dataset):
    all_features = []
    all_labels = []

    for images, labels in dataset:

        #Showing the difference between the preprocessed image and the original image
        # print("Images", images.shape)
        # plt.imshow(images[1].numpy() / 255)
        # plt.axis('off')
        # plt.show()

        
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        
        
        # print(preprocessed_images[0].numpy().min(), preprocessed_images[0].numpy().max())
        # plt.imshow(preprocessed_images[1] /255)
        # plt.axis('off')
        # plt.show()

        # plt.figure(figsize = (15,15))

        
        features = conv_base.predict(preprocessed_images)

        
        # print("The shape of features is ", features.shape)
        # #Here, we have the feature map from the convolutional base. Let's visualize some of them
        # f = features[0]
        # for i in range(32):
        #     plt.subplot(6,6,i+1)
        #     plt.imshow(f[:,:,i].astype('uint8'), cmap='gray')
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        #Showing different depths or channels or rgb of a image
        # plt.figure(figsize = (15,15))
        # for i in range(3):
        #     plt.subplot(6,6,i+1)
        #     plt.imshow(images[1][:,:,i].numpy().astype('uint8'))
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)
train_features.shape


#Now that we have extracted the featurs, we can run the model with the dense classifier

inputs = keras.Input(shape = (5,5,512),name = "Pre-trained model input")
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(
    loss = 'binary_crossentropy',
    metrics = ['accuracy'],
    optimizer = 'rmsprop'
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "feature_extraction.keras",
        save_best_only = True, 
        monitor = 'val_loss'
    )
]

history = model.fit(train_features, train_labels, epochs = 20, 
                    validation_data =(val_features, val_labels),
                   callbacks = callbacks
                   )


#Let's plot the model
history = history.history
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, "bo", label = "Training Accuracy")
plt.plot(epochs, val_accuracy, "b", label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, "bo", label = "Training loss")
plt.plot(epochs, val_loss, "b", label = "Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation loss")
plt.legend()
plt.show()


loaded_model = keras.models.load_model(
    'feature_extraction.keras'
)

test_loss, test_accuracy = loaded_model.evaluate(test_features, test_labels)
print("The test accuracy is ", test_accuracy)


conv_base = keras.applications.vgg16.VGG16(
    weights = 'imagenet',
    include_top = False, 
)




#Let's see the difference between the number of trainable weights before and after freezing

conv_base.trainable = True
print("Before freezing", len(conv_base.trainable_weights))

#Freezing the convolutional base
conv_base.trainable = False
print("After freezing", len(conv_base.trainable_weights))


data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.1)
])

inputs = keras.Input(shape = (180,180,3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)


#Fitting the model for training
modelCheckPoint = keras.callbacks.ModelCheckpoint(
    filepath = 'feature_with_aug.keras',
    save_best_only = True, 
    monitor = 'val_loss'
)

history = model.fit(train_dataset,
                   epochs = 50, 
                   callbacks = [modelCheckPoint],
                   validation_data = validation_dataset)


#Let's Plot the model

history = history
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, "bo", label = "Training Accuracy")
plt.plot(epochs, val_accuracy, "b", label = "Validation Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.figure()
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, "bo", label = "Training Loss")
plt.plot(epochs, val_loss, "b", label = "Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.show()


#Let's check the accuracy of the model in test data

loaded_model = keras.models.load_model('feature_with_aug.keras')

test_loss, test_accuracy = loaded_model.evaluate(test_dataset)
print("The accuracy of the model in test dataset is ", test_accuracy)


# We will only fine tune the layers that are specialized because that's where we need repurposoning
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False


#Compiling the model
model.compile(
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.RMSprop(learning_rate = 1e-5),
    metrics = ['accuracy']
)

modelCheckPoint = keras.callbacks.ModelCheckpoint(
    filepath = 'fine_tuned_model.keras',
    save_best_only = True, 
    monitor = 'val_loss'
)

history = model.fit(
    train_dataset, 
    validation_data = validation_dataset, 
    epochs = 30, 
    callbacks = [modelCheckPoint]
)


loaded_model = keras.models.load_model('fine_tuned_model.keras')
test_loss, test_accuracy = loaded_model.evaluate(test_dataset)
print("The accuracy of the model on test dataset is ", round(test_accuracy * 100, 4))

