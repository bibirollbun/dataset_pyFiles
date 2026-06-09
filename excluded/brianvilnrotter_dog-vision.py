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


#!unzip '/content/drive/MyDrive/Colab Notebooks/Zero-to-Mastery/machinelearning-ai/dog-vision-deep-learning-project/dog-breed-identification.zip' -d '/content/drive/MyDrive/Colab Notebooks/Zero-to-Mastery/machinelearning-ai/dog-vision-deep-learning-project/dog-breed-identification'


import zipfile,h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image,ImageFilter

# Consider making a class that pulls the data from the zip, allows for EDA, then download sets into h5 datasets.
# Taken from a class project I completed recently
class ZipImageDatasetBuilder:
  def __init__(self,zip_path,ext,label_map=None,split='train'):
    self.split = split
    self.zip_path = zip_path
    self.zf = zipfile.ZipFile(zip_path,'r')
    self.label_map = self._read_label_map(label_map) if label_map else None
    self.image_index = self._index_images(ext)
    self.label_set = set()
    print("Zipfile open, be sure to close once done")

  def _read_label_map(self,label_map):
    with self.zf.open(label_map) as f:
      return pd.read_csv(f)

  def _index_images(self,ext):
    if self.split == 'train':
      image_files = [f for f in self.zf.namelist() if f.lower().endswith(ext) and f.split('/')[0] == 'train']
      return[(
          f,
          f.split('/')[0],
          self.label_map.loc[self.label_map['id'] == (f.split('/')[1]).replace(f'.{ext}',''),'breed'].tolist()[0],
          f.split('/')[1]) for f in image_files]


  def iterate_images(self):
    for filepath,split,label,image_key in self.image_index:
      yield filepath,split,label,image_key

  def extract_metadata_row(self,filepath,label,split,image_key):
    try:
      with self.zf.open(filepath) as file:
        img = Image.open(file)
        width,height = img.size
        channels = len(img.getbands())
        size_bytes = self.zf.getinfo(filepath).file_size
      return {
          "image_key":image_key,
          "filepath":filepath,
          "split":split,
          "label":label,
          "width":width,
          "height":height,
          "channels":channels,
          "size_bytes":size_bytes
      }
    except Exception as e:
      print(f"Error processing {filepath}: {e}")
      return None

  def extract_metadata(self,metadata_output_path):
    metadata = []
    for (filepath,split,label,image_key) in tqdm(self.iterate_images(),desc="Extracting metadata"):
      self.label_set.add(label)
      row = self.extract_metadata_row(filepath,label,split,image_key)
      if row:
        metadata.append(row)

    df = pd.DataFrame(metadata)
    df.to_csv(metadata_output_path,index=False)
    print(f"{metadata_output_path} exported.")

  def width_height_kde_plot(self,metadata_output_path):
    # read the data and setup the plots
    df = pd.read_csv(pwd+'/metadata.csv')
    fig = plt.figure(figsize=(12,5))

    # draw kde 2d plot for image analysis
    sns.kdeplot(data=df,x='width',y='height')
    H,xedges,yedges = np.histogram2d(df['width'],df['height'],bins=50)
    for x,y in np.argwhere(H==H.max()):
      xi,yi = np.average(xedges[x:x+2]),np.average(yedges[y:y+2])
      plt.scatter(x=xi,y=yi,color='red')
      plt.annotate(f'({xi:.2f},{yi:.2f})',xy=(xi,yi),
                   xytext=(xi+0.5,yi+0.5),
                   fontsize=12,fontweight='bold',color='red')

    plt.tight_layout()
    plt.show()

  def build_h5_dataset(self,hd5_path,conver_string,resize):
    image_data = {'train':{'X':[],'y':[]},'test':{'X':[],'y':[]}}

    for filepath,split,label,_ in tqdm(self.iterate_images(),desc="Processing images"):
      self.label_set.add(label)

      try:
        with self.zf.open(filepath) as file:
          img = Image.open(file)
          img = img.convert(conver_string).resize(resize)
          img_arr = np.array(img,dtype=np.uint8)
      except Exception as e:
        print(f"Error processing {filepath}: {e}")
        continue

      image_data[split]['X'].append(img_arr)
      image_data[split]['y'].append(label)

    label_list = sorted(self.label_set)
    label_to_index = {label:idx for idx,label in enumerate(label_list)}

    with h5py.File(hd5_path,'w') as f:
      for split in ['train','test']:
        X = np.array(image_data[split]['X'],dtype=np.uint8)
        y = np.array([label_to_index[label] for label in image_data[split]['y']],dtype=np.uint8)

        f.create_dataset(f"{split}_X",data=X,compression="gzip")
        f.create_dataset(f"{split}_y",data=y)

      f.create_dataset("label_list",data=np.array(label_list,dtype=h5py.string_dtype(encoding='utf-8')))
      print(f"{hd5_path} exported.")

  def close(self):
    self.zf.close()
    print("Zipfile closed.")


# define the directory variable
pwd = '/kaggle/input/dog-breed-identification/'


#z = ZipImageDatasetBuilder(pwd+'/dog-breed-identification.zip',ext='jpg',label_map='labels.csv')
#z.extract_metadata(pwd+'/metadata.csv')


#z.width_height_kde_plot(pwd+'/metadata.csv')


#z.build_h5_dataset(pwd+'/dataset_224.h5',conver_string='RGB',resize=(128,128))
#z.close()


!pip install tensorflow==2.15.0 tensorflow-hub keras==2.15.0
import tensorflow as tf
import tensorflow_hub as hub
print("TF version:",tf.__version__)
print("TF-Hub version:",hub.__version__)

# Check for GPU availability
print("GPU", "available" if tf.config.list_physical_devices("GPU") else "not available")


# Checkout the labels.csv file
labels_csv = pd.read_csv(pwd+'labels.csv')
print(labels_csv.describe())
print(labels_csv.head())


# How many images are there of each breed
labels_csv['breed'].value_counts().plot.bar(figsize=(20,10))


labels_csv['breed'].value_counts().median(),labels_csv['breed'].value_counts().mean()


# Let's view an image
from IPython.display import Image
Image(pwd+'train/001513dfcb2ffafc82cccf4d8bbaba97.jpg')


filenames = [f'{pwd}train/{fname}.jpg' for fname in labels_csv['id']]
filenames[:10]


# Check whether number of filenames matches number of actual image files
import os
if len(os.listdir(pwd+'train/')) == len(filenames):
  print("Number of filenames matches the number of image files.")


Image(filenames[9000])


labels_csv['breed'][9000]


labels = labels_csv['breed'].to_numpy()
labels


len(labels)


# See if number of labels matches the number of filenames
if len(labels) == len(filenames):
  print("Number of labels matches the number of filenames.")


# Find the unique label values
unique_breeds = np.unique(labels)
len(unique_breeds)


# Turn a single label into an array of booleans
print(labels[0])
labels[0] == unique_breeds


# Turn every label into a boolean array
boolean_labels = [label == unique_breeds for label in labels]
boolean_labels[:2]


len(boolean_labels)


# Example: Turning boolean array into integers
print(labels[0])
print(np.where(unique_breeds == labels[0]))
print(boolean_labels[0].argmax())
boolean_labels[0].argmax()


# Setup X and y variables
X = filenames
y = boolean_labels


len(filenames)


# Set number of images to use for experimenting
NUM_IMAGES = 1000 #@param {type:"slider",min:1000,max:10000}


# Let's split our data into train and validation sets
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X[:NUM_IMAGES],y[:NUM_IMAGES],test_size=0.2,random_state=42)

len(X_train),len(y_train),len(X_val),len(y_val)


# Let's have a geez at the training data
X_train[:5],y_train[:2]


from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape


image


tf.constant(image)


# Define image size
IMG_SIZE = 224 #@param {type:"slider",min:128,max:512,step:64}

#Create a function for preprocessing images
def process_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image,channels=3)
  image = tf.image.convert_image_dtype(image,tf.float32)
  image = tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
  return image


# Create a simple function that returns a tuple of tensors
def get_image_label(image_path,label):
  return process_image(image_path),label


# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(X,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
  """
  Creates batches of data out of image (X) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """

  if test_data:
    print("Create test data batches...")
    data = tf.data.Dataset.from_tensor_slices(tf.constant(X)) # Only filepaths, no labels
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch

  elif valid_data:
    print("Create validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  print("Create training data batches...")
  # Turn filepaths and labels into Tensors
  data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
  data = data.shuffle(buffer_size=len(X))
  data_batch = data.map(get_image_label).batch(BATCH_SIZE)
  return data_batch


# Create training and validation data batches
train_data = create_data_batches(X_train,y_train)
val_data = create_data_batches(X_val,y_val,valid_data=True)


# Check out the different attributes of our data batches
train_data.element_spec,val_data.element_spec


# Create a function for viewing images in a data batch
def show_25_images(images,labels):
  """
  Displays 25 images from a data batch.
  """
  # Setup the figure
  plt.figure(figsize=(10,10))
  for i in range(25):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(images[i])
    plt.axis("off")
    plt.title(unique_breeds[labels[i].argmax()])

train_images,train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images,train_labels)


# Setup input shape to the model
INPUT_SHAPE = [None,IMG_SIZE,IMG_SIZE,3] # Batch, Height, Width, Colour channels

# Setup output shape of our model
OUTPUT_SHAPE = len(unique_breeds)

# Setup model URL
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model_url=MODEL_URL):
  print("Building model with:",MODEL_URL)

  # Setup the model layers
  model = tf.keras.Sequential([
      hub.KerasLayer(MODEL_URL),
      tf.keras.layers.Dense(units=OUTPUT_SHAPE,activation="softmax")
  ])

  # Compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"]
  )

  # Build the model
  model.build(INPUT_SHAPE)
  return model


model = create_model()
model.summary()


# Load Tensorboard notebook extension
#%load_ext tensorboard


import datetime

# Create a function to build a Tensorboard callback
def create_tensorboard_callback():
  logdir = os.path.join("logs",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  print(f"Saving TensorBoard log files to: {pwd+logdir}")
  return tf.keras.callbacks.TensorBoard(pwd+logdir)


# Create early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)


NUM_EPOCHS = 100 #@param {type:"slider",min:10,max:100}


# Check to make sure we'restill running on a GPU
print("GPU","available" if tf.config.list_physical_devices("GPU") else "not available")


# BUild a function to train and return a trained model
def train_model():
  model = create_model()
  tensorboard = create_tensorboard_callback()
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[early_stopping])
  return model


# Fit the model to the data
model = train_model()


#%tensorboard --logdir '/content/drive/MyDrive/Colab Notebooks/Zero-to-Mastery/machinelearning-ai/dog-vision-deep-learning-project/logs'


# Make predictions on the validation data (not used to train on)
predictions = model.predict(val_data,verbose=1)
predictions


predictions.shape


index = 0
print(predictions[index])
print(f"Max value (probability of prediction): {np.max(predictions[index])}")
print(f"Sum: {np.sum(predictions[index])}")
print(f"Max index: {np.argmax(predictions[index])}")
print(f"Predicted label: {unique_breeds[np.argmax(predictions[index])]}")


# Turn prediction probabilities into thei rrespective label (easier to understand)
def get_pred_label(prediction_probabilities):
  return unique_breeds[np.argmax(prediction_probabilities)]

# Get a predicted label based on array of prediction probabilities
pred_label = get_pred_label(predictions[0])
pred_label


# Create a function to unbatch a batch dataset
def unbatchify(data):
  images = []
  labels = []
  for image,label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])
  return images,labels

val_images,val_labels = unbatchify(val_data)
val_images[0],val_labels[0]


get_pred_label(val_labels[0])


def plot_pred(prediction_probabilities,labels,images,n=1):
  pred_prob,true_label,image = prediction_probabilities[n],labels[n],images[n]
  pred_label = get_pred_label(pred_prob)

  plt.imshow(image)
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"
  plt.title(f"Prediction: {pred_label} \n Probability: {np.max(pred_prob):.2f} \n True label: {true_label}",color=color)
  plt.axis(False)


plot_pred(predictions,val_labels,val_images,n=0)


def plot_pred_conf(prediction_probabilities,labels,n=1):
  pred_prob,true_label = prediction_probabilities[n],labels[n]
  pred_label = get_pred_label(pred_prob)
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]

  top_plot = plt.bar(np.arange(len(top_10_pred_labels)),top_10_pred_values,color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),labels=top_10_pred_labels,rotation='vertical')

  if np.isin(true_label,top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")


plot_pred_conf(prediction_probabilities=predictions,labels=labels,n=64)


# Let's check out a few prediction and their different values
i_multiplier = 0
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(10*num_cols,5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows,2*num_cols,2*i+1)
  plot_pred(prediction_probabilities=predictions,labels=val_labels,images=val_images,n=i+i_multiplier)
  plt.subplot(num_rows,2*num_cols,2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,labels=val_labels,n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()


# Create a function to save a model
def save_model(model,suffix=None):
    modeldir = os.path.join(pwd+'models',datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
    model_path = modeldir+"-"+suffix+".h5"
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path


# Create a function to load a saved mode
def load_model(model_path):
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                      custom_objects={"KerasLayer":hub.KerasLayer})
    return model


#save_model(model,suffix="1000-images-mobilenetv2-Adam")


# Load atrained model
#loaded_1000_image_model = load_model()


model.evaluate(val_data)


len(X), len(y)


# Create a data batch with the full data set
full_data = create_data_batches(X,y)


full_data


# Create a model for full model
full_model = create_model()


# Create full model callbacks
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=3)


# Check for GPU availability
print("GPU", "available" if tf.config.list_physical_devices("GPU") else "not available")


full_model.fit(x=full_data,
               epochs=NUM_EPOCHS,
               callbacks=[full_model_early_stopping])


# Save model


# Load test image filenames
test_path = pwd+"test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
test_filenames[:10]


len(test_filenames)


# Create test data batch
test_data = create_data_batches(test_filenames,test_data=True)


test_data


# Make predictions on test data batch using the loaded full model
test_predictions = full_model.predict(test_data,verbose=1)


# Save predictions (NumPy array) to csv file (for access later)
np.savetxt(pwd+'preds_array.csv',test_predictions,delimiter=",")


test_predictions = np.loadtxt(pwd+'preds_array.csv',delimiter=',')


test_predictions[:10]


test_predictions.shape


# Create a pandas DataFrame with empty columns
preds_df = pd.DataFrame(columns=['id'] + list(unique_breeds))
preds_df.head()


# Append test image ID's to predictions DataFrame
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df['id'] = test_ids
preds_df.head()


# Add the prediction probabilities to each dog breed column
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()


preds_df.to_csv(pwd+'full_model_predictions_submission_.csv',index=False)




