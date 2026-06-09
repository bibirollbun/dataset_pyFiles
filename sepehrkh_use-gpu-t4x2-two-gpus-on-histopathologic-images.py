import warnings
warnings.filterwarnings("ignore")
from numpy import asarray
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import glob
import os 
import random
import subprocess

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.io import imread
from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2

import gc
from keras import backend as K # to clear the previous session and memory



# tf.keras.mixed_precision.set_global_policy('mixed_float16')


# GPU or CPU is using

gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 1:
    print('Two GPUs are using') 
elif len(gpu_devices) == 1:
    print('GPU is using') 
else: 
    print('CPU is using')


train_path = '/kaggle/input/histopathologic-cancer-detection/train'
train_label_path = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'
test_path = '/kaggle/input/histopathologic-cancer-detection/test.csv'
sample_path = '/kaggle/input/histopathologic-cancer-detection/sample_submission.csv'


df = pd.read_csv(train_label_path)
print(df.head().to_markdown())


df.isnull().sum()


df_sample = pd.read_csv(sample_path)
print(df_sample.head().to_markdown())


df_sample.isnull().sum()


df.shape


df.info()


df['label'].value_counts()


print('Number of image : ', len(df))
print('Ratio labels : ', sum(df['label'].values)/len(df))


# Malignant (positive) and Normal (negative) Images

positive = df.loc[df['label']==1]['id'].values    # the ids of positive (malignant) cases
negative = df.loc[df['label']==0]['id'].values      
label_percent = df.value_counts(normalize=True)

print('Malignant (positive):')
print ('    Number of images:', len(positive),'=',round(len(positive)/len(df),2),'of all images')
print(positive[0:3],'\n')
      
print('Normal (negative):')
print ('    Number of images:', len(negative),'=',round(len(negative)/len(df),2),'of all images')
print(negative[0:3],'\n')


def plot_fig(ids,title,nrows=5,ncols=15):

    fig,ax = plt.subplots(nrows,ncols,figsize=(18,6))
    plt.subplots_adjust(wspace=0, hspace=0) 
    for i,j in enumerate(ids[:nrows*ncols]):
        fname = os.path.join(train_path ,j +'.tif')
        #fname = os.path.join(train_path ,j)
        img = Image.open(fname)
        idcol = ImageDraw.Draw(img)
        idcol.rectangle(((0,0),(95,95)),outline='white')
        plt.subplot(nrows, ncols, i+1) 
        plt.imshow(np.array(img))
        plt.axis('off')

    plt.suptitle(title, y=0.94)


plot_fig(positive,'Malignant (positive)')


plot_fig(negative,'Normal (negative):')


# Input Images into the List

train_file_path = '/kaggle/input/histopathologic-cancer-detection/train'
test_file_path = '/kaggle/input/histopathologic-cancer-detection/test'
labels_csv_path = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'# labels in CSV file

# Read the CSV file include labels
df_labels = pd.read_csv(labels_csv_path)

# Convert the dataframe to a dictionary for faster access to label
label_dict = dict(zip(df_labels['id'], df_labels['label']))

# Two-dimensional list for train
image_list_train = []
for train in os.listdir(train_file_path): 
    if train.endswith(".tif"): 
        file_path = os.path.join(train_file_path, train)
        image_id = train.split(".")[0]  # Remove the .tif extension to find the label
        label = label_dict.get(image_id, "Unknown")  # Get the label from the dictionary (Unknown if doesn't exist)
        image_list_train.append([file_path, label])

# Two-dimensional list for test (No labels)
image_list_test = []
for test in os.listdir(test_file_path): 
    if test.endswith(".tif"): 
        file_path = os.path.join(test_file_path, test)
        # label = "Test"  # Replace the label
        # image_list_test.append([file_path, label])
        image_list_test.append([file_path])


print(image_list_train[:5])



# The Number of Images
train_len = len(image_list_train)
print ("The Number of Histopathologic images in the train dataset: ", train_len)
test_len = len(image_list_test)
print ("The Number of Histopathologic images in the test dataset: ", test_len)


# Show Some Image
for i in range(5):
    img = plt.imread('/kaggle/input/histopathologic-cancer-detection/train/'+df.iloc[i]['id']+'.tif')
    print(df.iloc[i]['label'])
    plt.imshow(img)
    plt.show()


random_img = random.choice(df['id'])
random_img


# Calculate the Ratio of Images

# img = '/kaggle/input/histopathologic-cancer-detection/train/'+ df.iloc[0]['id']+'.tif'
img = '/kaggle/input/histopathologic-cancer-detection/train/'+ random_img +'.tif'

image= cv2.imread(img)
height, width= image.shape[:2]
print("The height is ", height)
print("The width is ", width)


Img_height = 96
Img_width = 96
Batch_size = 64 # Since overfitting occurs, and use GPU T4x2 and ReduceLR to adjust the dynamic learning rate.

epochs=10


LenTrain = 0.85
LenValid = 0.15


train_size = (int(LenTrain * len(image_list_train)) // Batch_size) * Batch_size
valid_size = (int(LenValid * len(image_list_train)) // Batch_size) * Batch_size
test_size = (int(len(image_list_test) // Batch_size) * Batch_size) # Because I'll use "drop_remainder=True"!
print("Split images in train to: ", "Train =", train_size , " Valid =", valid_size)
print("Also images in test: ", "Test =", test_size)


data_train = image_list_train[:train_size]
data_valid = image_list_train[train_size:train_size + valid_size]
data_test = image_list_test[:test_size]


# Image Augmentation and Rescale the "Train Dataset"
trainGenerator = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=15,       # Low rotation (15 degrees)
    width_shift_range=0.1,   # shift (up to 10% of the image)
    height_shift_range=0.1,  
    zoom_range=0.1,          # Low zoom (max 10% resizing)
    horizontal_flip=True,    # Horizontal mirroring
    fill_mode='nearest'      # How to do new pixels
)

# Only Rescale the "Validation Dataset" and "Test Dataset"
valGenerator = ImageDataGenerator(rescale=1./255.)
testGenerator = ImageDataGenerator(rescale=1./255.)


data_train[:10]


data_valid[:10]


data_test[:10]


# Create dataframes for train
df_train = pd.DataFrame(data_train,columns = ['id', 'label'])

# Create dataframes for validation
df_valid = pd.DataFrame(data_valid,columns = ['id', 'label'])

# Create dataframes for test
df_test = pd.DataFrame(data_test,columns = ['id'])


print(type(df_valid))


df_train.head()


df_valid.head()


df_train.shape


df_valid.shape


df_test.shape


df_test.head()


df_train.info()


# Convert numeric values of label column to string
df_train['label'] = df_train['label'].astype(str)
df_valid['label'] = df_valid['label'].astype(str)


df_train.head()


df_valid.info()


trainDataset = trainGenerator.flow_from_dataframe(
  dataframe = df_train,
  class_mode= "binary",  # class_mode set to 'binary'
  x_col = "id",
  y_col = "label",
  batch_size = Batch_size,
  seed = 42,
  shuffle = True,
  target_size = (Img_height,Img_width), #set the height and width of the images
  #drop_remainder=True  # The flow_from_dataframe or flow_from_directory functions do not support this parameter.
)

valDataset = valGenerator.flow_from_dataframe(
  dataframe = df_valid,
  class_mode = "binary", # class_mode set to 'binary'
  x_col = "id",
  y_col = "label",
  batch_size = Batch_size,
  seed = 42,
  shuffle = True,
  target_size = (Img_height,Img_width),
  #drop_remainder=True
)

testDataset = testGenerator.flow_from_dataframe(
  dataframe = df_test,
  class_mode = None, # Because it doesn't have any label
  x_col = "id",
  y_col = None, # Because it doesn't have any label
  batch_size = Batch_size,
  seed = 42,
  shuffle = False, # No need to shuffle test data
  target_size = (Img_height,Img_width),
)



# A function that converts the Generator to a tf.data.Dataset [Usable for training, validation, and testing datasets]
def dataset_from_generator(generator, has_labels=True):
    if has_labels:
        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=(None, Img_height, Img_width, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=tf.TensorSpec(shape=(None, Img_height, Img_width, 3), dtype=tf.float32)
        )
    # Since we are using multiple GPUs, never forget 'drop_remainder=True'! Otherwise the second GPU may not receive data in the last batch.
    # Safely unbatch and re-batch
    return dataset.unbatch().batch(Batch_size, drop_remainder=True)


# Load
def load_training_history(history_file):
    if os.path.exists(history_file):
        try:
            data = np.load(history_file)
            print(">>> Load Previous history ...")
            return (list(data['acc']) if 'acc' in data else [], 
                    list(data['val_acc']) if 'val_acc' in data else [],
                    list(data['loss']) if 'loss' in data else [],
                    list(data['val_loss']) if 'val_loss' in data else [])
        except Exception as e:
            print(f"âš ï¸� Error loading history: {e}")
            return [], [], [], []
    else:
        print("*** First run: Model training starts from scratch ... ")
        return [], [], [], []
# Save
def save_training_history(history_file, full_acc, full_val_acc, full_loss, full_val_loss):
    try:
        np.savez(history_file, acc=full_acc, val_acc=full_val_acc, loss=full_loss, val_loss=full_val_loss)
        print("Training history saved!")
    except Exception as e:
        print(f"âš ï¸� Error save history: {e}")


# Define MirroredStrategy only once, outside the `train_model` function

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"âœ… GPUs detected: {[gpu.name for gpu in gpus]}")
    strategy = tf.distribute.MirroredStrategy()
else:
    print("âš  No GPU detected! Running on CPU.")
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")



def train_model(model_name, model_fn, trainDataset, valDataset, testDataset, epochs):
    
    # 0) Clear previous session to avoid shape mismatch or residual graph issues
    K.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()
    
    # 1) History management
    history_file = f"{model_name}_history.npz"
    prev_acc, prev_val_acc, prev_loss, prev_val_loss = load_training_history(history_file)

    # 2) Using MirroredStrategy: Build or load model inside strategy scope
        # - Strategy already defined outside; use it here
    with strategy.scope():
        if os.path.exists(model_name):
            print(f">>> Loading the model from '{model_name}' to continue training...")
            model = tf.keras.models.load_model(model_name)  
        else:
            print(f"*** Building a new model and starting training ...")
            model = model_fn()  
            model.compile(loss=BinaryCrossentropy(),
                          optimizer=Adam(learning_rate=0.001), # The Best Result [Change from learning_rate=0.002 and 0.0005]
                          metrics=['accuracy'])        

    if not os.path.exists(model_name):
        model.summary()

    print(f"[INFO] Dataset sizes -> train: {len(trainDataset)}, valid: {len(valDataset)}, test: {len(testDataset)}")
    assert len(trainDataset) % strategy.num_replicas_in_sync == 0, "Train dataset not divisible by number of GPUs"

    # 3) Convert DataGenerator to tf.data.Dataset and preprocess
        # - Optimizing processing and preventing slowdowns can be achieved by "adding prefetch".
    trainDataset_tf = dataset_from_generator(trainDataset, has_labels=True).prefetch(tf.data.AUTOTUNE)
    valDataset_tf   = dataset_from_generator(valDataset, has_labels=True).prefetch(tf.data.AUTOTUNE)
    testDataset_tf  = dataset_from_generator(testDataset, has_labels=False).prefetch(tf.data.AUTOTUNE)


    # 4) Set Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    callbacks = [reduce_lr, early_stopping, checkpoint]

    # 5) Check batch shape of input data before training
    
    for sample_batch in trainDataset_tf.take(1):
        if isinstance(sample_batch, tuple):
            print("Batch shape (X):", sample_batch[0].shape)
            print("Batch shape (Y):", sample_batch[1].shape)
        else:
            print("Batch shape:", sample_batch.shape)


    # 6) Model training
        # - We must define steps to limit the number of execution steps; otherwise, execution may continue indefinitely.
        # - Note: Since flow_from_dataframe has been used before, its output works in batches, and the length (len(generator)) is the number of batches. 
          # [Therefore, there is no need for "// Batch_size" on trainDataset or valDataset unless we want to consider a smaller amount of data!]
    steps_per_epoch = len(trainDataset) ##// Batch_size # [Batch_size = 32]
    print("steps_per_epoch =", steps_per_epoch )
    validation_steps = len(valDataset) ##// Batch_size
    test_steps = df_test.shape[0] // Batch_size  # Only the full batches will be used  
    valid_test_count = test_steps * Batch_size
    
    history = model.fit(
        trainDataset_tf,
        epochs=epochs,
        validation_data=valDataset_tf,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )


    # 7) Merge previous history with new data
    full_acc = prev_acc + history.history['accuracy']
    full_val_acc = prev_val_acc + history.history['val_accuracy']
    full_loss = prev_loss + history.history['loss']
    full_val_loss = prev_val_loss + history.history['val_loss']
    print('Merged')
        # Save new history
    save_training_history(history_file, full_acc, full_val_acc, full_loss, full_val_loss)
    print('saved')

    # 8) Evaluate and predict the saved model
        # Verifying that the model has been successfully loaded [Note: If this code is not written, loading and execution may stop]
    if os.path.exists(model_name):
        print(f"âœ… Model '{model_name}' exists. Loading...")
        best_model = tf.keras.models.load_model(model_name)
    else:
        print(f"â�Œ Model '{model_name}' NOT found!")
        return 
        # - Evaluating the validation dataset
    print('Evaluated')
    loss, acc = best_model.evaluate(valDataset, verbose=1, steps=validation_steps) ## valDataset_tf
    print(f"âœ… Model '{model_name}' -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
        # - Predicting the test dataset
    print("ğŸ”� Predicting on test dataset...")
    # testDataset_tf = testDataset_tf.cache()
    predictions = best_model.predict(testDataset, verbose=1, steps=test_steps) ## testDataset_tf
    binary_predictions = (predictions > 0.5).astype(int).flatten() # Convert [0,1] predictions to 0 or 1
    print(binary_predictions[:20])

    # 9) Create a DataFrame from the model's prediction results and save it to a CSV file.
    ids = df_test['id'].values[:valid_test_count] # Extracting ids from df_test [Match with only the IDs that correspond to used test samples]
    
    output_df = pd.DataFrame({
        'id': ids,
        'label': binary_predictions
    })
    output_filename = f"{model_name.replace('.keras', '')}_predictions.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"ğŸ“� Predictions saved to '{output_filename}'")

    return best_model 



# History loading function and visualization "Accuracy" plot
def plot_accuracy(model_name):
    history_file = f"{model_name}_history.npz"
    if not os.path.exists(history_file):
        print("âš ï¸�No history was found for this model.")
        return

    data = np.load(history_file)
    acc = data['acc']
    val_acc = data['val_acc']

    plt.figure(figsize=(8, 6))
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Number of epochs & Accuracy in - {model_name}')
    plt.legend()
    plt.grid()
    plt.show()

# History loading function and visualization "Loss" plot
def plot_loss(model_name):
    history_file = f"{model_name}_history.npz"
    if not os.path.exists(history_file):
        print("âš ï¸�No history was found for this model.")
        return

    data = np.load(history_file)
    loss = data['loss']
    val_loss = data['val_loss']

    plt.figure(figsize=(8, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Number of epochs & Loss in - {model_name}')
    plt.legend()
    plt.grid()
    plt.show()

# Function to compare the "Accuracy" of multiple models together
def compare_models_accuracy(models):
    plt.figure(figsize=(10, 6))
    
    for model_name in models:
        history_file = f"{model_name}_history.npz"
        if os.path.exists(history_file):
            data = np.load(history_file)
            plt.plot(data['acc'], label=f"{model_name} - Train")
            plt.plot(data['val_acc'], linestyle='dashed', label=f"{model_name} - Val")
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Models - Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

# Function to compare the "Loss" of multiple models together
def compare_models_loss(models):
    plt.figure(figsize=(10, 6))
    
    for model_name in models:
        history_file = f"{model_name}_history.npz"
        if os.path.exists(history_file):
            data = np.load(history_file)
            plt.plot(data['loss'], label=f"{model_name} - Train")
            plt.plot(data['val_loss'], linestyle='dashed', label=f"{model_name} - Val")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Comparison of Models - Loss')
    plt.legend()
    plt.grid()
    plt.show()



print ('Remeber That;')
print ('The Size of input Images in the Modle are: ',Img_height,'*', Img_width)

print ('The Batch Size is: ', Batch_size)



##
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs are ready:", gpus)
else:
    print("No GPU detected!")


def AlexNet():
    inp = layers.Input((Img_height,Img_width, 3)) 

    x = layers.Conv2D(96, 7, 2, activation='relu', padding='same', kernel_regularizer=l2(0.0001))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2, padding='same')(x)

    x = layers.Conv2D(256, 5, 1, activation='relu', padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2, padding='same')(x)

    x = layers.Conv2D(384, 3, 1, activation='relu', padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.Conv2D(384, 3, 1, activation='relu', padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.Conv2D(256, 3, 1, activation='relu', padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.MaxPooling2D(3, 2, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.0001))(x) # Since overfitting occurs (Add L2) 
    x = layers.Dropout(0.5)(x) # Since overfitting occurs
    x = layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.Dropout(0.5)(x) # Since overfitting occurs

    x = layers.Dense(1, activation='sigmoid')(x)  # One-dimensional output for binary classification
   
    model_Alex = models.Model(inputs=inp, outputs=x)

    return model_Alex



# Ù�Shape of a random Input of images
sample_img, _ = next(iter(trainDataset)) 
print("Image shape from dataset:", sample_img.shape)


# Utilize train_model Function to Model (epochs=10)

train_model("AlexNet.keras", AlexNet, trainDataset, valDataset, testDataset, epochs)


# Visualize Accuracy history
plot_accuracy("AlexNet.keras")

# Visualize loss history
plot_loss("AlexNet.keras")


#VGGNet

def VGGNet():
    inp = layers.Input((Img_height,Img_width, 3))
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(inp) # Since overfitting occurs (Add L2)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)  # (48, 48, 64)

    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)  # (24, 24, 128)

    x = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)  # (12, 12, 256)

    x = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)  # (6, 6, 512)

    """
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.0005))(x)  # Reduce the number of nodes from 2048
    x = layers.Dropout(0.5)(x) # Since overfitting occurs
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)  # Reduce the number of nodes from 1024
    x = layers.Dropout(0.5)(x) # Since overfitting occurs
    x = layers.Dense(1, activation='sigmoid')(x)  # Binary output

    model_VGG = models.Model(inputs=inp, outputs=x)
    """

    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x) # Reduce the number of nodes
    x = layers.Dropout(0.4)(x) # Since overfitting occurs
    x = layers.Dense(1, activation='sigmoid')(x) # Binary output

    model_VGG = models.Model(inputs=inp, outputs=x)

    return model_VGG



# Utilize train_model Function to Model (epochs=10)

train_model("VGGNet.keras", VGGNet, trainDataset, valDataset, testDataset, epochs)


# Visualize Accuracy history
plot_accuracy("VGGNet.keras")

# Visualize Loss history
plot_loss("VGGNet.keras")


# ResNet

# Skip Connection
def residual_block(x, filters):
    shortcut = x 
    
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x) # Since overfitting occurs (Add L2)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation=None, kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    
    # Add Skip Connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x
    

def ResNet34():
    inp = layers.Input((Img_height,Img_width, 3))
    
    # The initial layer
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # ResNet blocks
    for _ in range(3):   # 3 blocks with 64 filters
        x = residual_block(x, 64)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x) 
    x = layers.BatchNormalization()(x)
    
    for _ in range(4):
        x = residual_block(x, 128)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    
    for _ in range(6):
        x = residual_block(x, 256)
    
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    
    for _ in range(3):
        x = residual_block(x, 512)
    
    # The final layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = layers.Dropout(0.5)(x) # Since overfitting occurs
    x = layers.Dense(1, activation='sigmoid')(x)  
    
    model = models.Model(inputs=inp, outputs=x)
    return model



# Utilize train_model Function to Model (epochs=10)

train_model("ResNet34.keras", ResNet34, trainDataset, valDataset, testDataset, epochs)


# Visualize Accuracy history
plot_accuracy("ResNet34.keras")

# Visualize Loss history
plot_loss("ResNet34.keras")


# Comparison of "AlexNet", "VGGNet", "ResNet34" models [Accuracy & Loss]
compare_models_accuracy(["AlexNet.keras", "VGGNet.keras", "ResNet34.keras"])
compare_models_loss(["AlexNet.keras", "VGGNet.keras", "ResNet34.keras"])
compare_models_accuracy(["VGGNet.keras", "ResNet34.keras"])
compare_models_loss(["VGGNet.keras", "ResNet34.keras"])


# Using weights="imagenet" for pre-trained weights
def efficientnet(model_type, input_shape=(Img_height,Img_width, 3), num_classes=1):
    if model_type == "B0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_type == "B1":
        base_model = EfficientNetB1(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_type == "B2":
        base_model = EfficientNetB2(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Invalid model type. Choose 'B0', 'B1' or 'B2'")
    
    # Freeze base model
    base_model.trainable = False  # Set False to use pre-trained weights
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x) # Since overfitting occurs (Add L2)
    x = layers.Dropout(0.4)(x) # Since overfitting occurs
    output = layers.Dense(num_classes, activation='sigmoid')(x)  # Binary classification
    
    model = models.Model(inputs=base_model.input, outputs=output)
    
    return model


# Utilize train_model Function to Model (epochs=10)

train_model("EfficientNetB0.keras", lambda: efficientnet("B0"), trainDataset, valDataset, testDataset, epochs)


# Visualize Accuracy history
plot_accuracy("EfficientNetB0.keras")

# Visualize Loss history
plot_loss("EfficientNetB0.keras")


# Utilize train_model Function to Model (epochs=10)

train_model("EfficientNetB1.keras", lambda: efficientnet("B1"), trainDataset, valDataset, testDataset, epochs)


# Visualize Accuracy history
plot_accuracy("EfficientNetB1.keras")

# Visualize Loss history
plot_loss("EfficientNetB1.keras")


# Utilize train_model Function to Model (epochs=10)

train_model("EfficientNetB2.keras", lambda: efficientnet("B2"), trainDataset, valDataset, testDataset, epochs)


# Visualize Accuracy history
plot_accuracy("EfficientNetB2.keras")

# Visualize Loss history
plot_loss("EfficientNetB2.keras")


# Comparison of "EfficientNetB2", "EfficientNetB1", "EfficientNetB0" models [Accuracy & Loss]
compare_models_accuracy(["EfficientNetB2.keras", "EfficientNetB1.keras", "EfficientNetB0.keras"])
compare_models_loss(["EfficientNetB2.keras", "EfficientNetB1.keras", "EfficientNetB0.keras"])
compare_models_accuracy(["EfficientNetB1.keras", "EfficientNetB0.keras"])
compare_models_loss(["EfficientNetB1.keras", "EfficientNetB0.keras"])


# Comparison of All models [Accuracy & Loss]
compare_models_accuracy(["AlexNet.keras", "VGGNet.keras", "ResNet34.keras","EfficientNetB2.keras", "EfficientNetB1.keras", "EfficientNetB0.keras"])
compare_models_loss(["AlexNet.keras", "VGGNet.keras", "ResNet34.keras","EfficientNetB2.keras", "EfficientNetB1.keras", "EfficientNetB0.keras"])
compare_models_accuracy(["VGGNet.keras", "ResNet34.keras","EfficientNetB1.keras", "EfficientNetB0.keras"])
compare_models_loss(["VGGNet.keras", "ResNet34.keras","EfficientNetB1.keras", "EfficientNetB0.keras"])




