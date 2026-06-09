import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os 

import tensorflow as tf


dir_path = "../input/dog-breed-identification/"
os.listdir(dir_path)



  for dirpath, dirnames, filenames in os.walk(dir_path ):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


train_images_path = "../input/dog-breed-identification/train/"
test_images_path = "../input/dog-breed-identification/test/"


#Lets read labels.csv file and check whats in that 
labels_df = pd.read_csv(dir_path + 'labels.csv')
print(labels_df.head())
print(labels_df.describe())


#Lets check if images in labels.csv is equal to images in our train folder 

def is_equal_images(target_dir, target_df): 
    """
    This function will check if target_dir images are equal to image list in target_df
    """
    len_target_dir = len(os.listdir(target_dir))
    len_target_df = len(target_df)
    if len_target_dir == len_target_df: 
        print(f"Both are having same no of images:{len_target_dir}")
    else: 
        print(f"Target dir having {len_target_dir} images while Target DF having {len_target_df}")
        
is_equal_images(target_dir = train_images_path, target_df = labels_df)


#check one image from training data
from IPython.display import Image, display
Image(train_images_path + '000bec180eb18c7604dcecc8fe0dba07.jpg')


#Check how many images per breed of dog. 

labels_df['breed'].value_counts().plot.bar(figsize=(20,10))
print(f"Average Images per breed:{int(labels_df['breed'].value_counts().sum()/len(labels_df['breed'].unique()))}")
print(f"Total no of breeds:{len(labels_df['breed'].unique())}")


#Create an array of train images 
filenames = [train_images_path + fname + '.jpg' for fname in labels_df['id']]
filenames[:10]


# Create class names array 
class_names = labels_df['breed'].unique()
class_names[:10]


target_labels = [breed for breed in labels_df['breed']]
target_labels[:10]


# Example: Turn one label into array of boolean 
print(target_labels[0])
target_labels[0] == class_names


#Lets do for all the labels 

target_labels_encoded = [label == np.array(class_names) for label in target_labels]
target_labels_encoded[:2]


# Example: Turning a boolean array into integers
print(target_labels[0]) # original label
print(np.where(class_names == target_labels[0])[0][0]) # index where label occurs
print(target_labels_encoded[0].argmax()) # index where label occurs in boolean array
print(target_labels_encoded[0].astype(int)) # there will be a 1 where the sample label occurs


#Import train test split from sklearn 

from sklearn.model_selection import train_test_split 

#Experiement with small data 1000 images 
NUM_IMAGES = 2000

#Split data into training & validation 
X_train, X_val, Y_train, Y_val = train_test_split(filenames[:NUM_IMAGES], target_labels_encoded[:NUM_IMAGES], test_size=0.2, random_state=42)

len(X_train), len(X_val), len(Y_train), len(Y_val)


#Check our the training data 

X_train[0], Y_train[0]


#Random image and its shape 
from matplotlib.pyplot import imread

img = imread(X_train[0])
plt.imshow(img)
print(f"Image Shape: {img.shape}")


tf.constant(img)


tf.image.convert_image_dtype(img, tf.float32)


IMAGE_SIZE = 224

# Lets write our preprocessing function
def process_image(image_path): 
    """
    This function will read image, resize the image and return into TF format. 
    Arguments: 
        image_path(str): Path of image
    Returns: 
        img: Tensor image
    """
    img = tf.io.read_file(image_path)
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    img = tf.io.decode_image(img, channels =3)
    # Convert the colour channel values from 0-225 values to 0-1 values
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to our desired size (224, 244)
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    return img



# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    image = process_image(image_path)
    return image, label

get_image_label(X_train[10], Y_train[10])


BATCH_SIZE = 32 

#Create function to create dataset batches 
def create_data_batches(X, y=None, batch_size = BATCH_SIZE, valid_data= False, test_data=False): 
    """
    This function will help to accept Train Images (X) and labels (y). 
    Also Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
    Also accepts test data as input (no labels).
    """
    if test_data: 
        print("Creating Test data")
        test_data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        test_data = test_data.map(process_image).batch(BATCH_SIZE) 
        return test_data 
    
    #Create validation data
    if valid_data: 
        print("Creating Validation data")
        valid_data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        valid_data = valid_data.map(get_image_label).batch(BATCH_SIZE)
        return valid_data
    
    #Shuffle and create training data
    else: 
        print("Creating Training Data") 
        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y))).shuffle(buffer_size = len(X))
        train_data = train_data.map(get_image_label).batch(BATCH_SIZE) 
        return train_data 
        


train_data = create_data_batches(X_train, Y_train)
valid_data = create_data_batches(X_val, Y_val, valid_data= True)


sample =next(iter(train_data))
sample[0][0]


import matplotlib.pyplot as plt 

def show_images(images, label): 
    """
    Display 25 Images with labels. 
    """
    #Setup the figure 
    plt.figure(figsize = (12,12)) 
    for i in range(0,25): 
        ax = plt.subplot(5, 5, i+1)
        
        plt.imshow(images[i])
        
        plt.title(class_names[tf.argmax(label[i])])
        
        plt.axis("off")
        


# Visualize training images from the training data batch
train_images, train_labels = next(train_data.as_numpy_iterator())
show_images(train_images, train_labels)


# Visualize validation images from the validation data batch
val_images, val_labels = next(valid_data.as_numpy_iterator())
show_images(val_images, val_labels)


import tensorflow as tf 
from tensorflow.keras import layers 

def create_model():
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top = False, 
                                                     classes = len(class_names)) 
    base_model.trainable = False 

    inputs = layers.Input(shape = (224,224,3))
    x = base_model(inputs, training = False) 
    x = tf.keras.layers.GlobalAveragePooling2D(name= "global_average_pooling")(x)
    x = layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)


    ModelDogBreed = tf.keras.Model(inputs, outputs) 

    ModelDogBreed.compile(loss = "categorical_crossentropy", 
                         optimizer = tf.keras.optimizers.Adam(), 
                         metrics=["accuracy"]) 

    return ModelDogBreed


model = create_model()

# Callbacks 

EarlyStoppingCallbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, baseline=None, restore_best_weights=True
)


ModelDogBreed_History = model.fit(train_data, 
                                         steps_per_epoch = len(train_data),
                                         epochs = 5, 
                                         validation_data= valid_data, 
                                         validation_steps = len(valid_data),
                                         callbacks = [EarlyStoppingCallbacks])


model.evaluate(valid_data)


def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


plot_loss_curves(ModelDogBreed_History)


predictions = model.predict(valid_data)
predictions


predictions.shape


# First prediction
print(predictions[0])
print(f"Max value (probability of prediction): {np.max(predictions[0])}") # the max probability value predicted by the model
print(f"Sum: {np.sum(predictions[0])}") # because we used softmax activation in our model, this will be close to 1
print(f"Max index: {np.argmax(predictions[0])}") # the index of where the max value in predictions[0] occurs
print(f"Predicted label: {class_names[np.argmax(predictions[0])]}") # the predicted label


# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return class_names[np.argmax(prediction_probabilities)]

# Get a predicted label based on an array of prediction probabilities
pred_label = get_pred_label(predictions[0])
pred_label


# Create a function to unbatch a batched dataset
def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  # Loop through unbatched data
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(class_names[np.argmax(label)])
  return images, labels

# Unbatchify the validation data
val_images, val_labels = unbatchify(valid_data)
val_images[0], val_labels[0]


def plot_pred(prediction_probabilities, labels, images, n=1):
  """
  View the prediction, ground truth label and image for sample n.
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

  plt.title("{} {:2.0f}% ({})".format(pred_label,
                                      np.max(pred_prob)*100,
                                      true_label),
                                      color=color)


# View an example prediction, original image and truth label
plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images)


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
  top_10_pred_labels = class_names[top_10_pred_indexes]

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


# Remind ourselves of the size of the full dataset
len(filenames), len(target_labels_encoded)


#Create training batch of data 

full_data = create_data_batches(filenames, target_labels_encoded)


ModelDogBreed_FullData = create_model()  #Create model
ModelDogBreed_FullData.summary()  


#Train our final model 

FinalModelDogBreed_FullData_History = ModelDogBreed_FullData.fit(full_data, 
                                         steps_per_epoch = len(full_data),
                                         epochs = 10,
                                         callbacks = [EarlyStoppingCallbacks])


# Load test image filenames (since we're using os.listdir(), these already have .jpg)
test_path = "../input/dog-breed-identification/test/"
test_filenames = [test_images_path + fname for fname in os.listdir(test_images_path)]

test_filenames[:10]


#View some test images 
img = imread(test_filenames[44])
plt.imshow(img)


# How many test images are there?
len(test_filenames)


# Create test data batch
test_data = create_data_batches(X=test_filenames, test_data=True)


# Make predictions on test data batch using the loaded full model
test_predictions = ModelDogBreed_FullData.predict(test_data,
                                      verbose=1)


# Create pandas DataFrame with empty columns
preds_df = pd.DataFrame(columns=["id"] + list(class_names))
preds_df.head()


# Append test image ID's to predictions DataFrame
preds_df["id"] = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df.head()


# Add the prediction probabilities to each dog breed column
preds_df[list(class_names)] = test_predictions
preds_df.head()


preds_df.to_csv("submission_with_mobilienetV2_1.csv",
                 index=False)

