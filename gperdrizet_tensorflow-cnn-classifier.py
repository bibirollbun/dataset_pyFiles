# Handle imports up-front
import os
import sys
import glob
import random

# Silence logging messages from TensorFlow, except errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# PyPI imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Silence logging messages from TensorFlow, except errors
tf.get_logger().setLevel('ERROR')

# Import image from keras correctly based on the TensorFlow version
tf_version = float('.'.join(tf.__version__.split('.')[0:2]))
print(f'Tensorflow version {tf_version}')

if tf_version > 2.8:
    import keras.utils as image

else:
    from keras.preprocessing import image

# Figure out if we are running on Kaggle or not, if so
# add the location of utils.py to path so we can import
path_list = os.getcwd().split(os.sep)

if path_list[1] == 'kaggle':
    sys.path.append('/kaggle/usr/lib/image_classification_functions')

# Import custom helper functions from utils.py
from image_classification_functions import prep_data
from image_classification_functions import single_training_run
from image_classification_functions import plot_single_training_run
from image_classification_functions import hyperparameter_optimization_run
from image_classification_functions import plot_hyperparameter_optimization_run


# Decompress and organize the images
training_data_path, validation_data_path, testing_data_path = prep_data()

# Get lists of training and validation dog and cat images
training_dogs = glob.glob(f'{training_data_path}/dogs/dog.*')
training_cats = glob.glob(f'{training_data_path}/cats/cat.*')
validation_dogs = glob.glob(f'{validation_data_path}/dogs/dog.*')
validation_cats = glob.glob(f'{validation_data_path}/cats/cat.*')


# Plot some of the cat and dog images
fig, axs = plt.subplots(3,2,figsize=(6, 4))

for cat, dog, row in zip(training_cats, training_dogs, axs):
    for animal, ax in zip([cat, dog], row):
        animal = image.load_img(animal)
        animal = image.img_to_array(animal)
        animal /= 255.0
        ax.imshow(animal)
        ax.axis('off')

fig.tight_layout()


# Load one of the dogs
dog = image.load_img(training_dogs[0])

# And convert it to an array - this is how TensorFlow will handel the data
dog = image.img_to_array(dog)

# Take a look at some properties of the object
print(f'Image data is: {type(dog)}')
print(f'Image data shape: {dog.shape}')


plt.hist(dog[:,:,0].flatten(), bins=100, color='red', alpha=0.5, label='Red channel')
plt.hist(dog[:,:,1].flatten(), bins=100, color='green', alpha=0.5, label='Green channel')
plt.hist(dog[:,:,2].flatten(), bins=100, color='blue', alpha=0.5, label='Blue channel')
plt.xlabel('Pixel value')
plt.ylabel('Count')
plt.legend(loc='best')
plt.show()


# Get a random sample of images, half cats and half dogs
sample_size = 500
sample = random.sample(training_dogs, sample_size//2)
sample += random.sample(training_cats, sample_size//2)

# Collectors for data
heights = []
widths = []

# Loop on the sample images
for sample_image in sample:

    # Load the image and convert it to an array
    sample_image = image.load_img(sample_image)
    sample_image = image.img_to_array(sample_image)

    # Get the width and height and add to collections
    heights.append(sample_image.shape[0])
    widths.append(sample_image.shape[1])

# Plot results as a histogram
plt.hist(heights, bins=50, alpha=0.5, label='Image heights')
plt.hist(widths, bins=50, alpha=0.5, label='Image widths')
plt.xlabel('Image dimension')
plt.ylabel('Count')
plt.legend(loc='best')
plt.show()


# Calculate the sample image aspect ratios
aspect_ratios = np.array(widths)/np.array(heights)

# Plot as histogram
plt.hist(aspect_ratios, bins=50, color='black')
plt.xlabel('Image aspect ratio')
plt.ylabel('Count')
plt.show()


image_width = 64
aspect_ratio = 4/3
image_height=int(image_width / aspect_ratio)

print(f'Input image dimensions: {image_width} x {image_height}')


epochs = 20
steps_per_epoch = 50
validation_steps = 50


%%time

# Do a single training run with default settings
training_results = single_training_run(
    training_data_path,
    validation_data_path,
    image_height=image_width,
    image_width=image_height,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    print_model_summary=True
)

# Plot the results
plot_single_training_run(training_results).show()

print()


%%time

# Define hyperparameters
hyperparameters = {
    'batch_sizes': [32, 64, 128],
    'learning_rates': [0.01, 0.001, 0.0001],
    'image_widths': [image_width],
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Train the model with each set of hyperparameters
hyperparameter_optimization_results = hyperparameter_optimization_run(
    training_data_path,
    validation_data_path,
    **hyperparameters
)

# Specify which hyperparameters to include in the plot labels
plot_labels = ['batch_sizes', 'learning_rates']

# Plot the learning curves
plot_hyperparameter_optimization_run(
    hyperparameter_optimization_results,
    hyperparameters,
    plot_labels,
    accuracy_ylims=[45,90],
    entropy_ylims=[0.4,0.8]
).show()

print()


batch_size = 64
learning_rate = 0.001
epochs = 150


%%time

# Set some hyperparameters for the run
hyperparameters={
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'image_width': image_width,
    'image_height': image_height,
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Do a single training run
training_results=single_training_run(
    training_data_path,
    validation_data_path,
    **hyperparameters
)

# Plot the results
plot_single_training_run(training_results).show()

print()


epochs = 150


%%time

# Define hyperparameters
hyperparameters = {
    'l1_penalties': [0.001, 0.0001],
    'l2_penalties': [0.001, 0.0001],
    'batch_sizes': [batch_size],
    'learning_rates': [learning_rate],
    'image_widths': [image_width],
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Train the model with each set of hyperparameters
hyperparameter_optimization_results = hyperparameter_optimization_run(
    training_data_path,
    validation_data_path,
    **hyperparameters
)

# Specify which hyperparameters to include in the plot labels
plot_labels = ['l1_penalties', 'l2_penalties']

# Plot the learning curves
plot_hyperparameter_optimization_run(
    hyperparameter_optimization_results,
    hyperparameters,
    plot_labels,
    accuracy_ylims=[45,90],
    entropy_ylims=[0.2,1.25]
).show()

print()


l1_penalty = 0.0001
l2_penalty = 0.001
epochs = 200


%%time

# Set some hyperparameters for the run
hyperparameters = {
    'l1_penalty': l1_penalty,
    'l2_penalty': l2_penalty,
    'image_width': image_width,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Do a single training run
training_results = single_training_run(
    training_data_path,
    validation_data_path,
    **hyperparameters,
)

# Plot the results
plot_single_training_run(training_results).show()

print()


l1_penalty = 0.0002
l2_penalty = 0.002


epochs = 200


%%time

# Define hyperparameters
hyperparameters = {
    'l1_penalties': [l1_penalty],
    'l2_penalties': [l2_penalty],
    'batch_sizes': [batch_size],
    'learning_rates': [learning_rate],
    'image_widths': [32, 64, 128],
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Train the model with each combination of hyperparameters
hyperparameter_optimization_results = hyperparameter_optimization_run(
    training_data_path,
    validation_data_path,
    **hyperparameters
)

# Specify which hyperparameters to include in the plot labels
plot_labels = ['image_widths']

# Plot the training curves
plot_hyperparameter_optimization_run(
    hyperparameter_optimization_results,
    hyperparameters,
    plot_labels,
    accuracy_ylims=[45,90],
    entropy_ylims=[0.4,1.4]
).show()

print()


image_width = 128
image_height = int(image_width/aspect_ratio)
epochs = 1000


%%time

# Set some hyperparameters for the run
hyperparameters = {
    'l1_penalty': l1_penalty,
    'l2_penalty': l2_penalty,
    'image_height': image_height,
    'image_width': image_width,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Do a single training run
training_results = single_training_run(
    training_data_path,
    validation_data_path,
    **hyperparameters
)

# Plot the results
plot_single_training_run(training_results).show()

print()


l1_penalty = 0.0006
l2_penalty = 0.006


epochs = 200


%%time

# Define hyperparameters
hyperparameters = {
    'filter_nums_list': [[16,32,64],[32,64,128],[64,128,256]],
    'filter_sizes': [3,4,5],
    'l1_penalties': [l1_penalty],
    'l2_penalties': [l2_penalty],
    'batch_sizes': [batch_size],
    'learning_rates': [learning_rate],
    'image_widths': [image_width],
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Train the model with each combination of hyperparameters
hyperparameter_optimization_results = hyperparameter_optimization_run(
    training_data_path,
    validation_data_path,
    **hyperparameters
)

# Specify which hyperparameters to include in the plot labels
plot_labels = ['filter_nums_list', 'filter_sizes']

# Plot the training curves
plot_hyperparameter_optimization_run(
    hyperparameter_optimization_results,
    hyperparameters,
    plot_labels,
    accuracy_ylims=[45,95],
    entropy_ylims=[0,2]
).show()

print()


filter_nums = [64,128,256]
filter_size = 4

epochs = 1000


%%time

# Set some hyperparameters for the run
hyperparameters = {
    'filter_nums': filter_nums,
    'filter_size': filter_size,
    'l1_penalty': l1_penalty,
    'l2_penalty': l2_penalty,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'image_height': image_height,
    'image_width': image_width,
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

# Do a single training run
training_results = single_training_run(
    training_data_path,
    validation_data_path,
    **hyperparameters
)

# Plot the results
plot_single_training_run(training_results).show()

print()


hyperparameters = {
    'filter_nums': filter_nums,
    'filter_size': filter_size,
    'l1_penalty': l1_penalty,
    'l2_penalty': l2_penalty,
    'image_height': image_height,
    'image_width': image_width,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'epochs': epochs
}

for key, value in hyperparameters.items():
    print(f'{key}: {value}')


# Get lists of testing dog and cat images
testing_dogs = glob.glob(f'{testing_data_path}/dogs/dog.*')
testing_cats = glob.glob(f'{testing_data_path}/cats/cat.*')

# Plot some of the cat and dog images
fig, axs = plt.subplots(3,2,figsize=(6, 4))

for cat, dog, row in zip(testing_cats, testing_dogs, axs):
    for animal, ax in zip([cat, dog], row):
        animal = image.load_img(animal)
        animal = image.img_to_array(animal)
        animal /= 255.0
        ax.imshow(animal)
        ax.axis('off')

fig.tight_layout()


testing_dataset = tf.keras.utils.image_dataset_from_directory(
    testing_data_path,
    image_size = (image_height, image_width),
    batch_size = batch_size,
    shuffle = False
)

images = np.concatenate([x for x, y in testing_dataset], axis=0)
labels = np.concatenate([y for x, y in testing_dataset], axis=0)


logits = training_results.model.predict(testing_dataset).flatten()

plt.hist(logits, color='black', bins=30)
plt.show()


threshold = 0.5
predictions = np.array([1 if p > threshold else 0 for p in logits])
labels = np.concatenate([y for x, y in testing_dataset], axis=0)
accuracy = accuracy_score(labels, predictions) * 100

# Plot the confusion matrix
cm = confusion_matrix(labels, predictions, normalize='true')
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
_ = cm_disp.plot()

plt.title(f'Test set performance\noverall accuracy: {accuracy:.1f}%')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()

