import pandas as pd
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB4
import os
import matplotlib.pyplot as plt
import random
# encode both columns label and variety
from sklearn.preprocessing import LabelEncoder
# ignore warnings   
import warnings
warnings.filterwarnings('ignore')


# import train.csv file
train_df = pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")
train_df.head()


# Check the value counts of the label column
train_df['label'].value_counts()


# Check the number of unique values
train_df['label'].nunique()


rescale = tf.keras.layers.Rescaling(1./255)


train_ds = keras.utils.image_dataset_from_directory(
    directory = '/kaggle/input/paddy-disease-classification/train_images',
    batch_size = 32,
    image_size = (224, 224),
    validation_split=0.2,
    subset="training",
    seed=123  
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='/kaggle/input/paddy-disease-classification/train_images',
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="validation",
    seed=123 
)


test_ds = keras.utils.image_dataset_from_directory(
    directory = '/kaggle/input/paddy-disease-classification/test_images',
    batch_size = 32,
    image_size = (224, 224),
    label_mode = None,
    shuffle=False
)


fig = px.scatter(train_df, x="age", y= "variety",color = "label")
fig.show()


fig = px.bar(train_df, x='label' , y='age', color='label')
fig.show()


# Create a sunburst plot
fig = px.sunburst(train_df, 
                  path=['label', 'variety'], 
                  values='age' , color='label')
# Show the plot
fig.show()


def visualize_images(path, num_images=5):

    # Get a list of image filenames
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if not image_filenames:
        raise ValueError("No images found in the specified path")

    # Select random images
    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))

    # Create a figure and axes
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3), facecolor='white')

    # Display each image
    for i, image_filename in enumerate(selected_images):
        # Load image
        image_path = os.path.join(path, image_filename)
        image = plt.imread(image_path)

        # Display image
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)  # Set image filename as title

    # Adjust layout and display
    plt.tight_layout()
    plt.show()



# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_blight"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_streak"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/bacterial_panicle_blight"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/blast"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/brown_spot"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/dead_heart"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/downy_mildew"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/hispa"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/normal"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


# Specify the path containing the images to visualize
path_to_visualize = "/kaggle/input/paddy-disease-classification/train_images/tungro"

# Visualize 5 random images
visualize_images(path_to_visualize, num_images=5)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Load the pre-trained EfficientNetB4 model without the top classification layer
efficientnet_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained base model layers
efficientnet_base.trainable = False


from keras.models import Sequential
# Build the model
model = Sequential()

# Add the pre-trained Xception base
model.add(efficientnet_base)

# Add global average pooling layer to reduce spatial dimensions
model.add(AveragePooling2D())

# Flatten the feature maps
model.add(Flatten())

# Add a dense layer with 120 units and ReLU activation function
model.add(Dense(220, activation='relu'))

# Dropout Layer
model.add(Dropout(0.25)) 

# Add the output layer with 1 unit and sigmoid activation function for binary classification
model.add(Dense(10, activation='softmax'))


model.summary()


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
 loss='sparse_categorical_crossentropy', metrics=['accuracy'])


%%time
# Define the callback function
early_stopping = EarlyStopping(patience=10)

history= model.fit(train_ds,
          validation_data=val_ds,
          epochs=100,
          callbacks=[early_stopping])


# evaluat the model
loss = model.evaluate(val_ds)

# Plotting the training and testing loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# plot the accuracy of training and validation

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


model.save('paddy_class')

