#import necessary tools
import tensorflow as tf
import tensorflow_hub as hub
print(tf.__version__)

#Check for GPU availability
print(tf.config.list_physical_devices("GPU"))


#checkout the labels of our data
import pandas as pd

labels_csv = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
print(labels_csv.describe())
labels_csv.head()


labels_csv["breed"].value_counts().plot.bar(figsize=(20,5))


#Let's view an image
from IPython.display import Image
Image("/kaggle/input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg")


# Create pathnames from image Id
filename = ["/kaggle/input/dog-breed-identification/train/" + fname + ".jpg" for fname in labels_csv["id"]]
filename[:10]


#Lets' check whether filenames matches actual amount of files
import os
if len(os.listdir("/kaggle/input/dog-breed-identification/train")) == len(filename):
  print("Success")
else:
  print("Check again")


Image(filename[200])


import numpy as np
labels = labels_csv["breed"].to_numpy()
len(labels)


# See if number of labels matches the number of filenames
if len(labels) == len(filename):
  print("Number of labels matches number of filenames!")
else:
  print("Number of labels does not match number of filenames, check data directories!")


# Find the unique label values
unique_breeds = np.unique(labels)
len(unique_breeds)


unique_breeds


# Turn every labels into a boolean array
boolean_labels = [labels == unique_breeds for labels in labels]
boolean_labels[2]


len(labels)


boolean_labels = [label == unique_breeds for label in labels]
print(boolean_labels[0].astype(int))


boolean_labels[:2]


len(boolean_labels)


# Example: Turning boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds == labels[0])) # index where label occurs
print(boolean_labels[0].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs


print(labels[2])
print(boolean_labels[2].astype(int))


boolean_labels[:3]


filename[:10]


#Setup X & y variables
X = filename
y = boolean_labels


len(filename)


# Set number of images to use for experimenting
NUM_IMAGES = 1000 #@param {type:"slider", min:1000, max:10000}


#Let's split the data into train and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)


len(X_train), len(y_train), len(X_val), len(y_val)


#Define Image size
IMG_SIZE = 224

#Creating a functon to preprocess the images
def process_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

  return image


#Create a function to return a tuple (image, label)
def get_image_label(image_path, label):
  image = process_image(image_path)
  return image, label


# Define the batch size, 32 is a good default
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch

  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    # If the data is a training dataset, we shuffle it
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels

    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))

    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch


#Create training and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)


train_data.element_spec, val_data.element_spec


import matplotlib.pyplot as plt

def show_25_images(images, labels):
  plt.figure(figsize=(10,10))
  for i in range (25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.title(unique_breeds[labels[i]])
    plt.axis("off")



train_images, train_lables = next(train_data.as_numpy_iterator())
train_images, train_lables


#Let's visualize the data
show_25_images(train_images, train_lables)


val_images, val_lables = next(val_data.as_numpy_iterator())
val_images, val_lables


show_25_images(val_images, val_lables)


import tf_keras as keras                # <-- use tf-keras API
from tensorflow_hub import KerasLayer

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
OUTPUT_SHAPE = len(unique_breeds)
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"  # use feature_vector model that accepts 224x224 input

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print("Building with:", model_url)
    model = keras.Sequential([
        KerasLayer(model_url, input_shape=input_shape, trainable=False),  # freeze base
        keras.layers.Dense(output_shape, activation="softmax")
    ])
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return model

model = create_model()
model.summary()


import datetime
def create_tensorboard_callback():
  logdir = os.path.join("/content/drive/MyDrive/Dog Breed Identificattion/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  return tf.keras.callbacks.TensorBoard(logdir)


#creating early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience=3)


NUM_EPOCHS = 100 #@param {type:"slider", min:10, max:100, step:10}


# Build a function to train and return a trained model
def train_model():
  """
  Trains a given model and returns the trained version.
  """
  # Create a model
  model = create_model()

  # Create new TensorBoard session everytime we train a model
  tensorboard = create_tensorboard_callback()

  # Fit the model to the data passing it the callbacks we created
  from tf_keras.callbacks import EarlyStopping, TensorBoard

# recreate BOTH callbacks from tf_keras
  early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
  tensorboard = TensorBoard(log_dir="/content/drive/MyDrive/Dog Breed Identificattion/logs/run") # Update log_dir

  model.fit(train_data, epochs=NUM_EPOCHS, validation_data=val_data,
          callbacks=[tensorboard, early_stopping])


  # Return the fitted model
  return model


model = train_model()


predictions = model.predict(val_data, verbose=1)
predictions


# Turn prediction probabilities into label

def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

pred_label = get_pred_label(predictions[81])
pred_label


# Create a function to unbatch a dataset
def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels"""
  images = []
  labels = []
  for image, label in data.unbatch():
    images.append(image.numpy())
    labels.append(unique_breeds[label.numpy().argmax()])
  return images, labels

val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]


def plot_pred(prediction_probabilities, labels, images, n=1):
  """
  View the prediction"""
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
  pred_label = get_pred_label(pred_prob)

  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  plt.title(f"Prediction: {pred_label}, {np.max(pred_prob)*100:.1f}% | GT: {true_label}", color=color)
  plt.axis(False)


def plot_pred_conf(prediction_probabilities, labels, n=1):
  pred_prob, true_label = prediction_probabilities[n], labels[n]
  pred_label = get_pred_label(pred_prob)
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]

  # Create a list of colors for the bars
  colors = ["blue"] * 10  # Start with all blue bars
  if true_label in top_10_pred_labels:
    # Find the index of the true label in the top 10 predictions
    true_label_index = np.where(top_10_pred_labels == true_label)[0][0]
    colors[true_label_index] = "green" # Set the color of the true label bar to green


  plt.bar(range(10), top_10_pred_values, tick_label=top_10_pred_labels, color=colors)
  plt.xticks(rotation="vertical")


plot_pred_conf(predictions, val_labels, n=60)


full_data = create_data_batches(X, y)
full_data


full_model = create_model()


from tf_keras.callbacks import EarlyStopping, TensorBoard
full_model_tensorboard = TensorBoard(log_dir="/content/drive/MyDrive/Dog Breed Identificattion/logs/full_model_run")
full_model_early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)


full_model.fit(full_data, epochs=NUM_EPOCHS, callbacks=[full_model_tensorboard])


# Load test image file names
test_path = "/kaggle/input/dog-breed-identification/test"
test_filenames = [test_path + "/" + fname for fname in os.listdir(test_path)]
test_filenames[:10]


test_data = create_data_batches(test_filenames, test_data=True)


test_predictions = model.predict(test_data, verbose=1)


# Create pandas dataframe
preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()


test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids
preds_df.head()


preds_df[list(unique_breeds)] = test_predictions
preds_df.head()


preds_df.to_csv("submission.csv", index=False)

