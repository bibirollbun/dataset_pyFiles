# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
    # for filename in filenames:
        # print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


# Import neccessary tools

import tensorflow as tf
import tensorflow_hub as hub
print(tf.__version__)
print(hub.__version__)

# Check for GPU availability
print("GPU", "available" if tf.config.list_physical_devices("GPU") else "not available")


import os

# Define dataset directory
dataset_dir = "/kaggle/input/dog-breed-identification"

# List all files and folders inside the dataset directory
files = os.listdir(dataset_dir)
print(files)


# check out the labels of our data
import pandas as pd

# Load labels.csv (contains image filenames and their corresponding breeds)
labels_df = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")

# Display first few rows
print(labels_df.head())


labels_df["breed"].value_counts().plot.bar(figsize=(20, 10))


# create pathnames from image's ID's
filenames = ["/kaggle/input/dog-breed-identification/train/" + fname + ".jpg" for fname in labels_df["id"]]
filenames[:10]


# check whether number of filenames matches number of actual image files

import os
if len(os.listdir("/kaggle/input/dog-breed-identification/train/")) == len(filenames):
  print("Filenames match actual amount of files")
else:
  print("Filenames do not match actual amount of files")


labels_df["breed"][9000]


import numpy as np
labels = labels_df["breed"].to_numpy()
labels


len(labels)


# see if number of labels matches the number of filenames
if len(labels) == len(filenames):
  print("Number of labels matches number of filenames")
else:
  print("Number of labels does not match number of filenames")


# Find the unique label values
unique_breeds = np.unique(labels)
unique_breeds


# Turn a single label into an array of booleans
print(labels[0])
labels[0] == unique_breeds


# Turn every label into a boolean array
boolean_labels = [label == unique_breeds for label in labels]
boolean_labels[:2]


# example turning boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds == labels[0])) # index where label occurs
print(boolean_labels[0].argmax()) #index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be 1 in the index where the label occurs


X = filenames
y = boolean_labels


# Set number of images to use for experimentation
NUM_IMAGES = 1000 #@param (type:"slider", min:1000, max:1000, step:1000)


# split into train and validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                 y[:NUM_IMAGES],
                                                 test_size=0.2,
                                                 random_state=42)
len(X_train), len(y_train)


# convert image to numpy
from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape


tf.constant(image)[:2]


# Define image size
IMG_SIZE = 224

# Create a functon for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
    """
    Takes an image file path and turns the image into a Tensor
    """
    # Read in an image file
    image = tf.io.read_file(image_path)
    # Turn the jpeg into numerical Tensor with 3 color channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # convert the color channel values from 0-225 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired value (224, 224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


# Create a simple function to return a tuple (image, label)

def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label.
    """

    image = process_image(image_path)
    return image, label


# demo of the above
(process_image(X[42]), y[42])


# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# create a function to turn data into batches
def create_data_batches(X,y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but does not shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    # If the data is a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), # filepaths
                                                  tf.constant(y))) # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                                  tf.constant(y)))
        # shuffling pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))

        # create (image, label) tuples (this also turns the image path into a preprocessed image)
            
        data = data.map(get_image_label)

        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)
    return data_batch


# Create training and validating data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)


# Check out the differernt attributes of our data batches

train_data.element_spec, val_data.element_spec


import matplotlib.pyplot as plt

# create a function for viewing images in a data batch
def show_25_images(images, labels):
    """
    Displays a plot of 25 images and their labels from
    a data batch
    """
    # setup the figure
    plt.figure(figsize=(10, 10))
    # loop through 25 (for displaying 25 images)
    for i in range(25):
        # create subplots (5 rows, 5 columns)
        ax = plt.subplot(5, 5, i+1)
        # Displays an image
        plt.imshow(images[i])
        # Add the image label as the title
        plt.title(unique_breeds[labels[i].argmax()])
        # Turn the grid lines off
        plt.axis("off")


train_images, train_labels = next(train_data.as_numpy_iterator())
train_images, train_labels


# Now let's visualize the data in a training batch
show_25_images(train_images, train_labels)



val_images, val_labels = next(val_data.as_numpy_iterator())
show_25_images(val_images, val_labels)


# # Setup input shape to the model
# INPUT_SHAPE = [IMG_SIZE, IMG_SIZE, 3] #batch, height, width, color channels

# # setup output shape of our model
# OUTPUT_SHAPE = len(unique_breeds)

# # setup model URL from TensorFlow Hub
# MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
# # MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


import tf_keras as keras



# # Create a function which builds a Keras model
# def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
#     print("Building model with:", model_url)

#     # Setup the model layers
#     model = keras.Sequential([
#         hub.KerasLayer(model_url, input_shape=input_shape, trainable=False),  # Feature extractor layer
#         keras.layers.Dense(units=output_shape, activation="softmax")  # Output layer
#     ])

#     # Compile the model
#     model.compile(
#         loss=keras.losses.CategoricalCrossentropy(),
#         optimizer=keras.optimizers.Adam(),
#         metrics=["accuracy"]
#     )

#     return model

# # Build and summarize the model
# model = create_model()
# model.summary()


# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] #batch, height, width, color channels

# setup output shape of our model
OUTPUT_SHAPE = len(unique_breeds)

# setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)

  # Setup the model layers
  model = keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
    keras.layers.Dense(units=OUTPUT_SHAPE, 
                          activation="softmax") # Layer 2 (output layer)
  ])

  # Compile the model
  model.compile(
      loss=keras.losses.CategoricalCrossentropy(), # Our model wants to reduce this (how wrong its guesses are)
      optimizer=keras.optimizers.Adam(), # A friend telling our model how to improve its guesses
      metrics=["accuracy"] # We'd like this to go up
  )

  # Build the model
  model.build(INPUT_SHAPE) # Let the model know what kind of inputs it'll be getting
  
  return model


# Create a model and check its details
model = create_model()
model.summary()


# Load TensorBoard notebook extension
import os
os.makedirs("/kaggle/working/logs", exist_ok=True)  # Create a log directory
%load_ext tensorboard


!ls /kaggle/working/logs



import os
import datetime
from tf_keras.callbacks import TensorBoard

def create_tensorboard_callback():
    # Create a new log directory with a timestamp
    logdir = os.path.join("/kaggle/working/logs", 
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Return a TensorBoard callback that saves logs to this directory
    return TensorBoard(log_dir=logdir)



# Create early stopping call back

early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                              patience=2,
                                               min_delta=0.01,
                                              restore_best_weights=True)


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

    # Fit the model to the data passing in the callbacks we created
    model.fit(x=train_data,
             epochs=NUM_EPOCHS,
             validation_data=val_data,
             validation_freq=1,
             callbacks=[tensorboard, early_stopping])

    # Return fitted model
    return model

# Fit the model to the data
model = train_model()


!ls -lh /kaggle/working/logs


%reload_ext tensorboard
%tensorboard --logdir /kaggle/working/logs


!tensorboard dev upload --logdir /kaggle/working/logs \
    --name "My Kaggle Experiment" \
    --description "Training logs for my model"


# Make predictions on the validation data (not used to train on)
predictions = model.predict(val_data, verbose=1)
predictions


predictions.shape


# First prediction
index = 0
print(predictions[index])
print(f"Max value(probability of predictions): {np.max(predictions[index])}")
print(f"Sum: {np.sum(predictions[index])}")
print(f"Max index: {np.argmax(predictions[index])}")
print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")


# Turn prediction probability into their respective label(easier to understand)

def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]

# Get a predicted label based on an array of prediction probabilities
pred_label = get_pred_label(predictions[11])
pred_label


val_data


# Create a function to unbatch a batch dataset
def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and
    returns separate arrays of image and labels.
    """
    images = []
    labels = []
    # Loop through unbatched data
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique_breeds[np.argmax(label)])

    return images, labels
# Unbathify the validation data
val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]



def plot_pred(prediction_probabilities, labels, images, n=1):
    """
    View the prediction, ground truth and image for sample n
    """
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

    # Get the pred label
    pred_label = get_pred_label(pred_prob)

    # Plot image & remove ticks
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    # Change the color of the title depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # Change plot title to be predicted, probablity of prediction and truth label
    plt.title("{} {:2.0f}% {}".format(pred_label,
                                     np.max(pred_prob)*100,
                                     true_label),
                                     color=color)


plot_pred(prediction_probabilities=predictions,
         labels=val_labels,
         images=val_images,
         n=100)


def plot_pred_conf(prediction_probabilities, labels, n=1):
  """
  Plots the top 10 highest prediction confidences along with
  the truth label for sample n.
  """
  pred_prob, true_label = prediction_probabilities[n], labels[n]

  # Get the predicted label
  pred_label = get_pred_label(pred_prob)

  # Find the top 10 prediction confidence indexes
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  # Find the top 10 prediction confidence values
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  # Find the top 10 prediction labels
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]

  # Setup plot
  top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                     top_10_pred_values, 
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  # Change color of true label
  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass


plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)


# Let's check a few predictions and their different values
i_multiplier = 0
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(5*2*num_cols, 5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            n=i+i_multiplier)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                labels=val_labels,
                n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()


# # Create a function to save a trained model
# def save_model(model, suffix=None):
#     """
#     Saves a given model in a model's directory and appends a suffix (string).
#     """
#     # crate a model directory pathname with current time
#     modeldir = os.path.join("/kaggle/working/models",
#                            datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
#     model_path = modeldir + "-" + suffix + ".keras" # Save format of model
#     print(f"Saving model to: {model_path}...")
#     model.save(model_path)
#     return model_path


# Create a function to save a trained model
def save_model(model, suffix=None):
    """
    Saves a given model in a model's directory and appends a suffix (string).
    """
    # Define the model directory
    modeldir = os.path.join("/kaggle/working/models")
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # Create a unique model file path with timestamp and suffix
    model_path = os.path.join(modeldir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + suffix + ".keras")
    
    print(f"Saving model to: {model_path}...")
    model.save(model_path)

    return model_path

# Save the model trained on 1000 images
save_model(model, suffix="1000-images-mobilenetv2-Adam")



def load_model(model_path):
    """
    Loads a saved model from a specified path
    """
    print(f"Loading saved model from: {model_path}")
    model = keras.models.load_model(model_path,
                                   custom_objects={"KerasLayer":hub.KerasLayer})
    return model


# Load trained model
load_image_model = load_model("/kaggle/working/models/20250215-083149-1000-images-mobilenetv2-Adam.keras")


# Evaluate a pre-saved model
model.evaluate(val_data)


len(X), len(y)


X[:10]


# Create a data batch with the full dataset
full_data = create_data_batches(X, y)


full_data


# Create a model for full_model
full_model = create_model()


# Create full_model callbacks
full_model_tensorboard = create_tensorboard_callback()
# No validation set when training on all the data, so we can't monitor validation accuracy
full_model_early_stopping = keras.callbacks.EarlyStopping(monitor="accuracy",
                                                         patience=2,
                                                         min_delta=0.01,
                                                         restore_best_weights=True)


# Fit the full model to the full data
full_model.fit(x=full_data,
              epochs=NUM_EPOCHS,
              callbacks=[full_model_tensorboard, full_model_early_stopping])


save_model(full_model, suffix="full-image-set-mobilenetv2-Adam")


# Load in the full model
loaded_full_model = load_model("/kaggle/working/models/20250215-083820-full-image-set-mobilenetv2-Adam.keras")


# Load test image filenames
test_path = "/kaggle/input/dog-breed-identification/test/"
test_filename = [test_path + fname for fname in os.listdir(test_path)]
test_filename[:10]


len(test_filename)


# Create test data batch
test_data = create_data_batches(test_filename,test_data=True )


test_data


# Make predictions on test data batch using the loaded full model
test_predictions = loaded_full_model.predict(test_data,
                                            verbose=1)


# Save predictions (Numpy Array) to csv file (for access later)
np.savetxt("/kaggle/working/preds_array.csv", test_predictions, delimiter=",")


# Load predictions (Numpy array) from csv file
test_predictions = np.loadtxt("/kaggle/working/preds_array.csv", delimiter=",")


test_predictions[:10]


test_predictions.shape


["id"] + list(unique_breeds)


# Create a pandas DataFrame with empty columns
preds_df = pd.DataFrame(columns=["id"] + list(unique_breeds))
preds_df.head()


# append test image ID's to predictions dataframe
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids


preds_df.head()


# Add the prediction probabilities to each dog breed column
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()


# Save preds_df to CSV in the correct directory
preds_df.to_csv("/kaggle/working/submission.csv", index=False)

print("Submission file saved successfully!")




