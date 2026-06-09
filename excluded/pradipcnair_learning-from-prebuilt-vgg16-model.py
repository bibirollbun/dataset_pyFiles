# Importing all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import VGG16


# Import the zipfile module to work with zip files
import zipfile

# Define a function to extract a zip file to a specified directory
def extract_zip(input_zip, output_dir="."):
    with zipfile.ZipFile(input_zip, "r") as z:
        z.extractall(output_dir)

# Paths to the zip files for training and testing datasets
train_zip_path = "/kaggle/input/dogs-vs-cats/test1.zip"
test_zip_path = "/kaggle/input/dogs-vs-cats/train.zip"

extract_zip(train_zip_path)
extract_zip(test_zip_path)



#Import necessary libraries
import os
import pandas as pd

#Get the list of file names in the "train" folder
filenames = os.listdir("/kaggle/working/train")

# Initialize an empty list to store labels
labels = []

# Loop through each filename to determine its label based on its name
for filename in filenames:
    if "cat" in filename:
        labels.append('cat')  # Append 'cat' if the filename contains "cat"
    elif "dog" in filename:
        labels.append('dog')  # Append 'dog' if the filename contains "dog"
    
# Create a DataFrame with filenames and their corresponding labels
df = pd.DataFrame({
    'Filename': filenames,
    'Label': labels
})

# Display the first five rows of the DataFrame
df.head()


# Define image dimensions and batch size
IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150
BATCH_SIZE = 32

# Create an image data generator with rescaling and validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generate training data from DataFrame with specified parameters
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='train/',
    x_col='Filename',
    y_col='Label',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Generate validation data from DataFrame with specified parameters
validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='train/',
    x_col='Filename',
    y_col='Label',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


# Load VGG16 without the top layer (fully connected layers)
model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))


# Freeze the convolutional base so that it's not trained again
model.trainable = False

# Summary of the VGG16 base model
model.summary()


# Create a custom model on top of the pretrained VGG16 model

# Flatten the output of the base model
new_top = Flatten()(model.output)  

# Add a fully connected layer with 512 units and ReLU activation
new_top = Dense(512, activation='relu')(new_top)  

# Apply dropout regularization to prevent overfitting
new_top = Dropout(0.5)(new_top)  

# Add a final output layer with sigmoid activation for binary classification
predictions = Dense(1, activation='sigmoid')(new_top) 

 # Define the input and output of the custom model
model = Model(inputs=model.input, outputs=predictions)  

# Summary of the final model
model.summary()


# Suppress warnings for a cleaner output
import warnings
warnings.filterwarnings('ignore')

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])


history = model.fit(train_generator, steps_per_epoch=137, 
                      validation_data= validation_generator,  validation_steps=35, 
                      epochs=15,
                      verbose=2)


# Visualize model training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])  # Set y-axis limits for clarity
plt.legend(loc='lower right')
plt.title('Model Accuracy over Epochs')
plt.show()


# Unfreeze the last 10 layers of the base model for fine-tuning
for layer in model.layers[-10:]:
    layer.trainable = True

# Compile the model with Adam optimizer, binary crossentropy loss, and accuracy metric
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Continue training the model with fine-tuning
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=10 
)


# Visualize the accuracy of the fine-tuned model
plt.plot(history_fine.history['accuracy'], label='accuracy')
plt.plot(history_fine.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.92, 1])
plt.legend(loc='lower right')


# Extracting the test data from the ZIP file
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as test1_zip:
    test1_zip.extractall('.') 

# Define the directory path for the test data
test_dir = "test1/"

# List the filenames of the test images
filenames = os.listdir(test_dir)

# Create a DataFrame to store the filenames of the test images
test_data = pd.DataFrame({"Filename": filenames})

# Display the first few rows of the test data DataFrame
test_data.head()


# Create an ImageDataGenerator for test data preprocessing
test_gen = ImageDataGenerator(rescale=1./255)

# Generate batches of test data from the DataFrame
test_generator = test_gen.flow_from_dataframe(
    dataframe=test_data,
    directory='test1/',
    x_col='Filename',
    y_col=None,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
)


# Perform predictions on the test data using the trained model
test1_predict = model.predict(test_generator)
test1_predict


# Set the threshold for binary classification
threshold = 0.5

# Assign labels to the test data based on the predicted probabilities
test_data['Label'] = np.where(test1_predict > threshold, 'dog', 'cat')


# Image processing
import matplotlib.image as mpimg

# Display a sample of test images with predicted labels
sample_test = test_data.sample(n=6)
plt.figure(figsize=(24, 8))

subplot_index = 1
for _, row in sample_test.iterrows():
    image_path = os.path.join(test_dir, row['Filename'])
    img = mpimg.imread(image_path)
    plt.subplot(2, 3, subplot_index)
    plt.imshow(img)
    plt.title(row['Label'])
    plt.axis('off')
    subplot_index += 1

plt.show()

