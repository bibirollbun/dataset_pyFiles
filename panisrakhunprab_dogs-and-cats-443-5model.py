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
print(os.listdir("../input"))

import zipfile

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")


# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'train/'
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'dog.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()


# plot cat photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'train/'
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'cat.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()


pip install --upgrade 'numpy<1.23.0'


# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)


# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		copyfile(src, dst)


!ls -l dataset_dogs_vs_cats/train/dogs | wc -l


!ls -l dataset_dogs_vs_cats/train/cats | wc -l


!pip -q install livelossplot==0.5.5


from livelossplot import PlotLossesKeras


# save the final model to file
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


from keras.preprocessing.image import ImageDataGenerator
# create data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# prepare iterator
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=(224, 224))
test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
    class_mode='binary', batch_size=64, target_size=(224, 224))


# load model
model = VGG16(
    include_top=False,
    input_shape=(224, 224, 3)
    )
print(model.summary())
#model = Xception(include_top=False, input_shape=(299, 299, 3))




# mark loaded layers as not trainable
for index, layer in enumerate(model.layers[:15]):
    print(index, layer.name, layer.output_shape)
    layer.trainable = False
    
for index, layer in enumerate(model.layers[15:]):
    print(index, layer.name, layer.output_shape)
    layer.trainable = True


model.layers[-1].output


# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
# define new model
my_model = Model(inputs=model.inputs, outputs=output)


# compile model
opt = SGD(lr=0.0001, momentum=0.9)
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
my_model.summary()


# fit model
my_model.reset_states()
history = my_model.fit(train_it, 
                    steps_per_epoch=len(train_it),
                    validation_data=test_it, 
                    validation_steps=len(test_it), 
                    epochs=3, verbose=1,
                    callbacks=[PlotLossesKeras()])


# evaluate model
_, acc = my_model.evaluate(test_it, steps=len(test_it), verbose=1)
print('> %.3f' % (acc * 100.0))


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.tight_layout()
    pyplot.show()
    pyplot.close()


# learning curves
summarize_diagnostics(history)


# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img


# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sample_image.jpg')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict(img)
	print(result[0])


load_image(filename)


from keras.preprocessing import image
import requests
from io import BytesIO

url = 'https://machinelearningmastery.com/wp-content/uploads/2019/03/sample_image.jpg'
response = requests.get(url)
img = image.load_img(BytesIO(response.content), target_size=(224, 224))


import os
print(os.getcwd())


# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/'] #test==>validation
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir       
        makedirs(newdir, exist_ok=True)
        print(newdir)


ls -l


ls -l dataset_dogs_vs_cats/test


# seed random number generator
seed(1337)
# define ratio of pictures to use for validation
val_ratio = 0.25 #changable
# copy training dataset images into subdirectories
src_directory = 'train/'

for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		copyfile(src, dst)


import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

# à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¸ªà¸³à¸«à¸£à¸±à¸šà¸�à¸²à¸£à¸ªà¸¸à¹ˆà¸¡à¸£à¸¹à¸›
def random_image_from_folder(folder, num_images=5):
    # à¸ªà¸¸à¹ˆà¸¡à¹„à¸Ÿà¸¥à¹Œà¸ˆà¸²à¸�à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œà¸—à¸µà¹ˆà¸£à¸°à¸šà¸¸
    image_files = random.sample(os.listdir(folder), num_images)
    images = []
    for file in image_files:
        img_path = os.path.join(folder, file)
        img = image.load_img(img_path, target_size=(224, 224))  # Resize à¸ à¸²à¸�à¹ƒà¸«à¹‰à¸‚à¸™à¸²à¸”à¸—à¸µà¹ˆà¸•à¹‰à¸­à¸‡à¸�à¸²à¸£
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # à¹€à¸�à¸´à¹ˆà¸¡à¸¡à¸´à¸•à¸´
        images.append((img_array, file, img))  # à¹€à¸�à¹‡à¸šà¸—à¸±à¹‰à¸‡à¸ à¸²à¸�à¹�à¸¥à¸°à¸Šà¸·à¹ˆà¸­à¹„à¸Ÿà¸¥à¹Œ
    return images

# à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¸—à¸³à¸™à¸²à¸¢à¸£à¸¹à¸›
def predict_image(model, img_array):
    prediction = model.predict(img_array)
    return "Dog" if prediction[0] > 0.5 else "Cat"

# à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¸ªà¸³à¸«à¸£à¸±à¸šà¹�à¸ªà¸”à¸‡à¸£à¸¹à¸›à¸�à¸£à¹‰à¸­à¸¡à¸œà¸¥à¸�à¸²à¸£à¸—à¸³à¸™à¸²à¸¢à¹�à¸¥à¸°à¹€à¸‰à¸¥à¸¢
def plot_images_with_predictions(model, folder_dog, folder_cat, num_images=5):
    # à¸ªà¸¸à¹ˆà¸¡à¸ à¸²à¸�à¸«à¸¡à¸²à¹�à¸¥à¸°à¹�à¸¡à¸§
    dog_images = random_image_from_folder(folder_dog, num_images)
    cat_images = random_image_from_folder(folder_cat, num_images)

    # à¸ªà¸£à¹‰à¸²à¸‡à¸�à¸£à¸²à¸Ÿ
    plt.figure(figsize=(12, 6))  # à¸�à¸³à¸«à¸™à¸”à¸‚à¸™à¸²à¸”à¸‚à¸­à¸‡à¸�à¸£à¸²à¸Ÿ

    # à¹�à¸ªà¸”à¸‡à¸œà¸¥à¸�à¸²à¸£à¸—à¸³à¸™à¸²à¸¢à¸«à¸¡à¸² (à¹�à¸–à¸§à¹�à¸£à¸�)
    for i, (img_array, img_name, img) in enumerate(dog_images):
        plt.subplot(2, 5, i + 1)  # à¸�à¸²à¸£à¸ˆà¸±à¸”à¸£à¸¹à¸›à¸ à¸²à¸�à¹ƒà¸™à¹�à¸–à¸§à¹�à¸£à¸�
        plt.imshow(img)  # à¹ƒà¸Šà¹‰à¸ à¸²à¸�à¹€à¸”à¸´à¸¡à¸—à¸µà¹ˆà¹„à¸¡à¹ˆà¸œà¹ˆà¸²à¸™à¸�à¸²à¸£ preprocess
        predicted_label = predict_image(model, img_array)
        true_label = "Dog"  # à¹€à¸™à¸·à¹ˆà¸­à¸‡à¸ˆà¸²à¸�à¹€à¸›à¹‡à¸™à¸ à¸²à¸�à¸«à¸¡à¸²
        plt.title(f"Prediction: {predicted_label}\nTrue: {true_label}")
        plt.axis('off')

    # à¹�à¸ªà¸”à¸‡à¸œà¸¥à¸�à¸²à¸£à¸—à¸³à¸™à¸²à¸¢à¹�à¸¡à¸§ (à¹�à¸–à¸§à¸—à¸µà¹ˆà¸ªà¸­à¸‡)
    for i, (img_array, img_name, img) in enumerate(cat_images):
        plt.subplot(2, 5, i + 6)  # à¸�à¸²à¸£à¸ˆà¸±à¸”à¸£à¸¹à¸›à¸ à¸²à¸�à¹ƒà¸™à¹�à¸–à¸§à¸—à¸µà¹ˆà¸ªà¸­à¸‡
        plt.imshow(img)  # à¹ƒà¸Šà¹‰à¸ à¸²à¸�à¹€à¸”à¸´à¸¡à¸—à¸µà¹ˆà¹„à¸¡à¹ˆà¸œà¹ˆà¸²à¸™à¸�à¸²à¸£ preprocess
        predicted_label = predict_image(model, img_array)
        true_label = "Cat"  # à¹€à¸™à¸·à¹ˆà¸­à¸‡à¸ˆà¸²à¸�à¹€à¸›à¹‡à¸™à¸ à¸²à¸�à¹�à¸¡à¸§
        plt.title(f"Prediction: {predicted_label}\nTrue: {true_label}")
        plt.axis('off')

    plt.tight_layout()  # à¸—à¸³à¹ƒà¸«à¹‰à¸�à¸£à¸²à¸Ÿà¹„à¸¡à¹ˆà¹€à¸šà¸µà¸¢à¸”à¸�à¸±à¸™
    plt.show()

# à¹ƒà¸Šà¹‰à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆà¸„à¸¸à¸“à¸�à¸¶à¸�à¹„à¸§à¹‰
# à¸ªà¸¡à¸¡à¸¸à¸•à¸´à¸§à¹ˆà¸²à¹‚à¸¡à¹€à¸”à¸¥à¸–à¸¹à¸�à¸�à¸¶à¸�à¹�à¸¥à¸°à¸šà¸±à¸™à¸—à¸¶à¸�à¹�à¸¥à¹‰à¸§
# à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¹ƒà¸«à¹‰à¹‚à¸¡à¹€à¸”à¸¥à¹€à¸›à¹‡à¸™à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆà¸„à¸¸à¸“à¹ƒà¸Šà¹‰
model = my_model  # à¸«à¸£à¸·à¸­à¹‚à¸¡à¹€à¸”à¸¥à¸‚à¸­à¸‡à¸„à¸¸à¸“à¸—à¸µà¹ˆà¸�à¸¶à¸�à¸¡à¸²

# à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œà¸ªà¸³à¸«à¸£à¸±à¸šà¸«à¸¡à¸²à¹�à¸¥à¸°à¹�à¸¡à¸§
folder_dog = 'dataset_dogs_vs_cats/test/dogs'
folder_cat = 'dataset_dogs_vs_cats/test/cats'

# à¹€à¸£à¸µà¸¢à¸�à¹ƒà¸Šà¹‰à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¹ƒà¸™à¸�à¸²à¸£à¹�à¸ªà¸”à¸‡à¸œà¸¥
plot_images_with_predictions(model, folder_dog, folder_cat)


import os
import pandas as pd
from glob import glob

# Define the image folder path
image_folder = '/kaggle/working/test1'

# List all .jpg files in the directory (adjust the pattern if needed)
image_files = glob(os.path.join(image_folder, '*.jpg'))

# Check if images were found
if not image_files:
    print("No images found. Please verify the directory and file extensions.")
    
# Create a DataFrame with the file paths
df = pd.DataFrame({'filename': image_files})

# If your use-case doesn't require labels (for example, during testing), you can leave out the label column.
# Otherwise, if you need a dummy label, you can add one:
# df['class'] = 'dummy_label'


IMAGE_SIZE = (299, 299)

datagen = ImageDataGenerator(rescale=1.0/255.0,
                            horizontal_flip=True,)

# prepare iterators
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=IMAGE_SIZE )

val_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
    class_mode='binary', batch_size=64, target_size=IMAGE_SIZE )

# Prediction
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize your ImageDataGenerator (modify parameters as needed)
datagen_test = ImageDataGenerator(rescale=1./255,
                                 horizontal_flip=True)

# Create the data generator using the DataFrame
test_it = datagen_test.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col=None,            # No labels for testing, so set to None
    target_size=IMAGE_SIZE,
    batch_size=64,
    class_mode=None,       # Since there are no labels
    # shuffle=True,         # Usually testing data should not be shuffled
    shuffle=False,
)


from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD

def define_model_inceptionv3():
    # load model
    model = InceptionV3(include_top=False, input_shape=(299, 299, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')(flat1)
    drop1 = Dropout(0.5)(class1)
    class2 = Dense(512, activation='relu', kernel_initializer='he_uniform')(drop1)
    drop2 = Dropout(0.5)(class2)
    class3 = Dense(256, activation='relu', kernel_initializer='he_uniform')(drop2)
    drop3 = Dropout(0.5)(class3)
    class4 = Dense(128, activation='relu', kernel_initializer='he_uniform')(drop3)
    drop4 = Dropout(0.5)(class4)
    
    output = Dense(1, activation='sigmoid')(drop4)  # à¹ƒà¸Šà¹‰ drop4 à¹�à¸—à¸™ class4

    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)  # à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™ lr à¹€à¸›à¹‡à¸™ learning_rate
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


import matplotlib.pyplot as plt
def plot_exmaple(images):
    # Assuming test_images is a numpy array with shape (64, height, width, channels)
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    axes = axes.flatten()  # Flatten the 8x8 grid into a list for easy iteration
    
    for i, ax in enumerate(axes):
        # Ensure there are enough images in the batch before plotting
        if i < test_images.shape[0]:
            ax.imshow(test_images[i])
        ax.axis('off')  # Turn off axis ticks/labels for clarity
    
    plt.tight_layout()
plt.show()


model_inceptionv3 = define_model_inceptionv3()


model_inceptionv3.summary()


from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


# fit model
model_inceptionv3.reset_states()
history_inceptionv3 = model_inceptionv3.fit(
    train_it, 
    steps_per_epoch=len(train_it),
    validation_data=val_it,
    validation_steps=len(val_it),
    epochs=3, verbose=1,
    callbacks = [PlotLossesKeras()])


import matplotlib.pyplot as plt

def plot_images_with_labels(images, labels, train_log, set_label, save=False):
    """
    Plots a grid of images with their corresponding labels and saves the figure.
    
    Args:
    - images (list or np.array): List of images to be displayed.
    - labels (np.array): Array of one-hot encoded labels.
    - train_log (str): The directory or log name to save the plot.
    - set_label (str): The specific set label (e.g., "train", "validation", etc.).
    - config (dict): Configuration dictionary containing image dimensions.
    
    Returns:
    - None: Displays and saves the plot.
    """
    # Convert one-hot encoded labels to class indices
    class_indices = labels

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=120)
    axes = axes.flatten()

    for img, label_idx, ax in zip(images, class_indices, axes):
        # Undo any preprocessing (like rescaling for pre-trained models)
        if np.max(img) == 1:
            img = img * 255

        # Display the image
        img_shape = (224, 224, 3)
        ax.imshow(img.astype("uint8"))
        # ax.imshow(img)

        # Set the title to the corresponding class label
        ax.set_title(f"Label: {label_idx}", fontsize=18)
        ax.axis('on')

    # Set the overall title for the figure
    plot_title = f"{train_log}-{set_label}"
    plt.suptitle(plot_title, fontsize=18)

    # Show the plot
    plt.show()

    # Save the figure
    if save:
        plot_file = os.path.join(train_log, f"{plot_title}.png")
        fig.savefig(plot_file, dpi=120)
        print(f'Saved {plot_file}')


# evaluate model
_, acc = model_inceptionv3.evaluate(val_it, steps=len(val_it), verbose=1)
print('> %.3f' % (acc * 100.0))


len(test_it)


next(iter(test_it))


# learning curves
summarize_diagnostics(history_inceptionv3)


import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input  # à¹ƒà¸Šà¹‰ preprocess à¸‚à¸­à¸‡ InceptionV3
from keras.models import load_model

# à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¸ªà¸³à¸«à¸£à¸±à¸šà¸�à¸²à¸£à¸ªà¸¸à¹ˆà¸¡à¸£à¸¹à¸›
def random_image_from_folder(folder, num_images=5):
    image_files = random.sample(os.listdir(folder), num_images)
    images = []
    for file in image_files:
        img_path = os.path.join(folder, file)
        img = image.load_img(img_path, target_size=(299, 299))  # à¹ƒà¸Šà¹‰ 299x299
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # à¹€à¸�à¸´à¹ˆà¸¡à¸¡à¸´à¸•à¸´
        img_array = preprocess_input(img_array)  # ğŸ”¹ à¹ƒà¸Šà¹‰ preprocess_input
        images.append((img_array, file, img))
    return images

# à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¸—à¸³à¸™à¸²à¸¢à¸£à¸¹à¸›
def predict_image(model, img_array):
    prediction = model.predict(img_array, batch_size=1)[0][0]  # ğŸ”¹ à¸�à¸³à¸«à¸™à¸” batch_size=1
    print(f"Confidence Score: {prediction:.4f}")  # ğŸ”¹ à¹€à¸�à¸´à¹ˆà¸¡à¸�à¸²à¸£à¸�à¸´à¸¡à¸�à¹Œà¸„à¹ˆà¸² Score
    return "Dog" if prediction > 0.5 else "Cat"  # ğŸ”¹ à¸›à¸£à¸±à¸š Threshold à¸�à¸¥à¸±à¸šà¹„à¸› 0.5

# à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¸ªà¸³à¸«à¸£à¸±à¸šà¹�à¸ªà¸”à¸‡à¸£à¸¹à¸›à¸�à¸£à¹‰à¸­à¸¡à¸œà¸¥à¸�à¸²à¸£à¸—à¸³à¸™à¸²à¸¢à¹�à¸¥à¸°à¹€à¸‰à¸¥à¸¢
def plot_images_with_predictions(model, folder_dog, folder_cat, num_images=5):
    dog_images = random_image_from_folder(folder_dog, num_images)
    cat_images = random_image_from_folder(folder_cat, num_images)

    plt.figure(figsize=(12, 6))

    for i, (img_array, img_name, img) in enumerate(dog_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        predicted_label = predict_image(model, img_array)
        plt.title(f"Prediction: {predicted_label}\nTrue: Dog")
        plt.axis('off')

    for i, (img_array, img_name, img) in enumerate(cat_images):
        plt.subplot(2, 5, i + 6)
        plt.imshow(img)
        predicted_label = predict_image(model, img_array)
        plt.title(f"Prediction: {predicted_label}\nTrue: Cat")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆà¸�à¸¶à¸�à¹„à¸§à¹‰
model = model_inceptionv3  # à¹�à¸—à¸™à¸”à¹‰à¸§à¸¢à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆà¸„à¸¸à¸“à¸�à¸¶à¸�

# à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œà¸ªà¸³à¸«à¸£à¸±à¸šà¸«à¸¡à¸²à¹�à¸¥à¸°à¹�à¸¡à¸§
folder_dog = 'dataset_dogs_vs_cats/test/dogs'
folder_cat = 'dataset_dogs_vs_cats/test/cats'

# à¹€à¸£à¸µà¸¢à¸�à¹ƒà¸Šà¹‰à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¹ƒà¸™à¸�à¸²à¸£à¹�à¸ªà¸”à¸‡à¸œà¸¥
plot_images_with_predictions(model, folder_dog, folder_cat)


print("Training set:")
print(train_it.class_indices)
print(train_it.samples)


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_it.classes)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

model_inceptionv3.fit(train_it, epochs=3, class_weight=class_weight_dict)


from keras.layers import GlobalAveragePooling2D

def create_model(base_model, input_shape=(224, 224, 3), num_classes=1):
    base_model = base_model(include_top=False, input_shape=input_shape, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    # x = GlobalAveragePooling2D()(base_model.output)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)

    flat1 = GlobalAveragePooling2D()(base_model.output)
    class1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')(flat1)
    drop1 = Dropout(0.5)(class1)
    class2 = Dense(512, activation='relu', kernel_initializer='he_uniform')(drop1)
    drop2 = Dropout(0.5)(class2)
    class3 = Dense(256, activation='relu', kernel_initializer='he_uniform')(drop2)
    drop3 = Dropout(0.5)(class3)
    class4 = Dense(128, activation='relu', kernel_initializer='he_uniform')(drop3)
    drop4 = Dropout(0.5)(class4)
    output = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(drop4)

    return model


from tensorflow.keras.applications import ResNet50
resnet_model = create_model(ResNet50, input_shape=(224, 224, 3))


from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


# fit model
resnet_model.reset_states()
history_resnet_model = resnet_model.fit(
    train_it, 
    steps_per_epoch=len(train_it),
    validation_data=val_it,
    validation_steps=len(val_it),
    epochs=3, verbose=1,
    callbacks = [PlotLossesKeras()])


# evaluate model
_, acc = resnet_model.evaluate(val_it, steps=len(val_it), verbose=1)
print('> %.3f' % (acc * 100.0))


# learning curves
summarize_diagnostics(history_inceptionv3)


plot_images_with_predictions(resnet_model, folder_dog, folder_cat)


from tensorflow.keras.applications import MobileNetV2
mobilenet_model = create_model(MobileNetV2, input_shape=(224, 224, 3))


from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


mobilenet_model.summary()


# fit model
resnet_model.reset_states()
history_mobilenet_model = mobilenet_model.fit(
    train_it, 
    steps_per_epoch=len(train_it),
    validation_data=val_it,
    validation_steps=len(val_it),
    epochs=3, verbose=1,
    callbacks = [PlotLossesKeras()])


# evaluate model
_, acc = mobilenet_model.evaluate(val_it, steps=len(val_it), verbose=1)
print('> %.3f' % (acc * 100.0))


# learning curves
summarize_diagnostics(history_mobilenet_model)


plot_images_with_predictions(mobilenet_model, folder_dog, folder_cat)


from tensorflow.keras.applications import EfficientNetB0
efficientnet_model = create_model(EfficientNetB0, input_shape=(224, 224, 3))


from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


# fit model
efficientnet_model.reset_states()
history_efficientnet_model = efficientnet_model.fit(
    train_it, 
    steps_per_epoch=len(train_it),
    validation_data=val_it,
    validation_steps=len(val_it),
    epochs=3, verbose=1,
    callbacks = [PlotLossesKeras()])


# evaluate model
_, acc = efficientnet_model.evaluate(val_it, steps=len(val_it), verbose=1)
print('> %.3f' % (acc * 100.0))


# learning curves
summarize_diagnostics(history_efficientnet_model)


plot_images_with_predictions(efficientnet_model, folder_dog, folder_cat)


from tensorflow.keras.applications import NASNetMobile
nasnet_model = create_model(NASNetMobile, input_shape=(224, 224, 3))


from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


# fit model
nasnet_model.reset_states()
history_nasnet_model = nasnet_model.fit(
    train_it, 
    steps_per_epoch=len(train_it),
    validation_data=val_it,
    validation_steps=len(val_it),
    epochs=3, verbose=1,
    callbacks = [PlotLossesKeras()])


# evaluate model
_, acc = nasnet_model.evaluate(val_it, steps=len(val_it), verbose=1)
print('> %.3f' % (acc * 100.0))


# learning curves
summarize_diagnostics(history_nasnet_model)


plot_images_with_predictions(nasnet_model, folder_dog, folder_cat)




