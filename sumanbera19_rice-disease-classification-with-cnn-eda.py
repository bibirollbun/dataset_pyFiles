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
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Import Os and Basis Libraries
import cv2
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
#Matplot Images
import matplotlib.image as mpimg
# Tensflor and Keras Layer and Model and Optimize and Loss
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
#PreTrained Model
from tensorflow.keras.applications import *
#Image Generator DataAugmentation
#Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
# Warnings Remove 
import warnings 
warnings.filterwarnings("ignore")
#Splitting Data 
# import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
import random
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


df=pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")


df.head()


df["variety"].value_counts()


plt.figure(figsize=(15,10))
sns.countplot(x="variety" ,data=df)
plt.title("variety countplot")
plt.show()


plt.figure(figsize=(15,10))
sns.histplot(x="variety",data=df,kde=True)
plt.title("variety Histogram")
plt.show()


colors = sns.color_palette('pastel')
plt.figure(figsize=(15,10))
plt.pie(df["variety"].value_counts(), labels=df["variety"].value_counts().index, colors=colors, autopct='%.2f%%')

plt.title("variety Distribution")
plt.show()



plt.figure(figsize=(15,8))
sns.barplot(x="variety",y="age",data=df)
plt.title("Variety vs Age")
plt.show()


# Directory containing the "Train" folder
directory = "/kaggle/input/paddy-disease-classification/train_images"

filepath =[]
label = []

folds = os.listdir(directory)

for fold in folds:
    f_path = os.path.join(directory , fold)
    
    imgs = os.listdir(f_path)
    
    for img in imgs:
        
        img_path = os.path.join(f_path , img)
        filepath.append(img_path)
        label.append(fold)
        
#Concat data paths with labels
file_path_series = pd.Series(filepath , name= 'path')
Label_path_series = pd.Series(label , name = 'label')
df_train = pd.concat([file_path_series ,Label_path_series ] , axis = 1)


df_train


df_train['label'].value_counts()


plt.figure(figsize=(15, 10))
sns.countplot(x="label", data=df_train)
plt.title("Label Countplot (Diseases)")
plt.xticks(rotation=45)
plt.tight_layout()       
plt.show()


plt.figure(figsize=(15, 10))
sns.histplot(x="label",data=df_train,kde=True)
plt.title("Label histogram (Diseases)")
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.tight_layout()       # Prevent label cutoff
plt.show()


colors = sns.color_palette('pastel')
plt.figure(figsize=(15,10))
plt.pie(df_train["label"].value_counts(), labels=df_train["label"].value_counts().index, colors=colors, autopct='%.2f%%')

plt.title("variety Distribution")
plt.show()


def visualize_images(path, num_images=5):
    # Get a list of image filenames in the specified path
    image_filenames = os.listdir(path)
    
    # Limit the number of images to visualize if there are more than num_images
    num_images = min(num_images, len(image_filenames))
    
    # Create a figure and axis object to display images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3),facecolor='white')
    
    # Iterate over the selected images and display them
    for i, image_filename in enumerate(image_filenames[:num_images]):
        # Load the image using Matplotlib
        image_path = os.path.join(path, image_filename)
        image = mpimg.imread(image_path)
        
        # Display the image
        axes[i].imshow(image)
        axes[i].axis('off')  # Turn off axis
        axes[i].set_title(image_filename)  # Set image filename as title
    
    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()



path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_blight"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_streak"
# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/bacterial_panicle_blight"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/blast"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/brown_spot"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/dead_heart"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/downy_mildew"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/hispa"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/normal"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/tungro"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=8)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = "/kaggle/input/paddy-disease-classification/train_images"
test_dir  = "/kaggle/input/paddy-disease-classification/test_images"

# âœ… IMPORTANT: Make sure all test images are inside a subfolder like 'unknown/'
# Structure: /test_images/unknown/img1.jpg

# Train and Validation Generator from train_dir
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # Split 80% train, 20% validation
)

# Training data (80%)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset="training",
    shuffle=True,
    seed=42
)

# Validation data (20%)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset="validation",
    shuffle=False,
    seed=42
)





model = Sequential([

    # Block 1
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
  

    # Block 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),


    # Block 4
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    

  
    # Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.52),

    Dense(train_generator.num_classes, activation='softmax')
])


model.summary()


# 6. Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,  # Set to the value of val_loss at the desired epoch
    restore_best_weights=False,
)


# 7. Train Model using validation data
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,callbacks=[early_stopping]
)


#Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()



plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


from tensorflow.keras.preprocessing import image_dataset_from_directory

test_dir = '/kaggle/input/paddy-disease-classification/test_images'
test_ds = image_dataset_from_directory(
    test_dir,
    label_mode=None,
    shuffle=False,
    image_size=(128,128),
    batch_size=32,
)

# batch_size=32,
#     image_size=(256, 256),

def process(image):
    image = tf.cast(image / 255, tf.float32)
    return image

test_ds = test_ds.map(process)


# Assuming label_names contains the class names in the correct order
label_names = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast',
               'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

# Predict labels for test images
predicted_labels_all = []
for images in test_ds:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels_all.extend(predicted_classes)

# Map predicted class indices to class names
predicted_labels_names_all = [label_names[prediction] for prediction in predicted_labels_all]

# Print predicted labels for the first few images
num_predictions = 5  # Number of predictions to print
for idx, label in enumerate(predicted_labels_names_all[:num_predictions]):
    print(f"Image {idx + 1}: Predicted Label: {label}")


# Assuming predicted_labels_names_all contains all predicted labels
predicted_labels_df = pd.DataFrame({'label': predicted_labels_names_all})

# Load the sample_submission.csv file
submission_df = pd.read_csv('/kaggle/input/paddy-disease-classification/sample_submission.csv')

# Add the predicted labels to the submission dataframe
submission_df['label'] = predicted_labels_df['label']

# Save the updated dataframe back to the sample_submission.csv file
submission_df.to_csv('sample_submission.csv', index=False)


submission_df




