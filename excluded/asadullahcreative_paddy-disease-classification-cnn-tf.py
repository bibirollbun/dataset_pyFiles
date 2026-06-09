# Importing Libararies
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# Remove Warnings
import warnings
warnings.filterwarnings('ignore')


# Lets Load the Training Data
data=pd.read_csv('/kaggle/input/paddy-disease-classification/train.csv')
data.head()


# Lets print the shape of the Training Data
print('The shape of the Training Data is:', {data.shape})
print('The Number of Rows  in the Training Data is:', data.shape[0])
print('The Number of Column  in the Training Data is:', data.shape[1])


# Lets check the unique value list of the label column
unique_values = data['label'].nunique()
print("Total number of unique values in label column are :", unique_values)
print('-----------------------------------------')
print('Unique values of label column are:')
data['label'].unique().tolist()


# Lets check the unique value list of the label column
unique_values_variety = data['variety'].nunique()
print("Total number of unique values in variety column are :", unique_values_variety)
print('-----------------------------------------')
print('Unique values of variety column are:')
data['variety'].unique().tolist()


fig, axes = plt.subplots(1, 1, figsize=(12, 6))
sns.countplot(data=data, x='variety', ax=axes, palette='magma')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Variety', fontsize=12)
plt.title('Count plot of the Variety column', fontsize=14)
plt.show()


fig, ax = plt.subplots(figsize=(20, 5))

sns.countplot(x='label', data=data, ax=ax, palette='Dark2')
ax.set_title('Count plot of the Label column', fontsize=15)
ax.set_xlabel('Label', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
plt.show()


data['age'].describe()


# Lets check Which Variety of RICE plant have Maximum age
max_age_variety = data[data['age'] == 82]['variety'].iloc[0]
print(f"The variety with the maximum age of 82 is **{max_age_variety}**.")


# Replace 'data' with your DataFrame name
fig = px.histogram(data_frame=data, x='age', color='variety', nbins=20, color_discrete_sequence=px.colors.qualitative.Dark24)
fig.show()


# Lets Filter the label column based on Normal
normal=data[data['label']=='normal']
# Lets Filter the variety column based on ADT45
normal=normal[normal['variety']=='ADT45']
five_normal=normal.image_id[:5].values
five_normal.tolist()


# Lets Filter the label column based on dead_heart
dead=data[data['label']=='dead_heart']
# Lets Filter the variety column based on ADT45
dead=dead[dead['variety']=='ADT45']
five_dead=dead.image_id[:5].values
five_dead.tolist()


# make plot of images just to have an idea 
plt.figure(figsize=(20,10))
columns=5
path='/kaggle/input/paddy-disease-classification/train_images/'

for i,image_loc in enumerate(np.concatenate((five_normal,five_dead))):
    plt.subplot(10//columns+1,columns,i+1)

    if(i<5):
        image = plt.imread(path+"normal/"+image_loc)
        plt.title('normal')
    else:
        plt.title('dead heart disease')
        image = plt.imread(path+"dead_heart/"+image_loc)
    plt.imshow(image)


data.head()


# Encode the two categorical columns : label and variety
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['label']=label_encoder.fit_transform(data['label'])
data['variety']=label_encoder.fit_transform(data['variety'])
data.head()


# Define Parameters 
batch_size=16 # 16 images to proceed in one batch 
img_height=224 # height of the image
img_width=224 # width of the image 


train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
)


# Get the class names from the dataset
class_name=train_ds.class_names
print('class Names:\n',class_name)


# --- 1. Grab a batch ---------------------------------------------------------
images, labels = next(iter(train_ds))          # images: (B, H, W, C)

# If labels are oneâ€‘hot, squeeze them down to integer IDs
if len(labels.shape) > 1:                      # e.g. (B, 10)
    labels = tf.argmax(labels, axis=1)

labels = labels.numpy()                        # -> NumPy array
batch_size = images.shape[0]

# --- 2. Build a grid big enough for the batch (cap at 25) --------------------
max_show = min(batch_size, 25)
rows = cols = int(np.ceil(np.sqrt(max_show)))  # 1Ã—1 â€¦ 5Ã—5

fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
axes = axes.flatten()                          # easy 1â€‘D indexing

# --- 3. Plot -----------------------------------------------------------------
for i in range(rows * cols):
    ax = axes[i]
    if i < max_show:
        ax.imshow(images[i].numpy().astype("uint8"))
        idx = int(labels[i])                   # safe Python int
        title = class_name[idx] if idx < len(class_name) else "Unknown"
        ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()



# Get the first image and its label from the dataset
for image_batch,label_batch in train_ds:
    print("Shape of the Image Batch : ",image_batch.shape)
    print("Shape of the Label Batch : ",label_batch.shape)
    break


# For this dataset, the pixel values are in `[0, 255]`.
# We need to rescale them to `[0, 1]`
# Perform the rescaling by using the `Rescaling` layer
normalization_layer = tf.keras.layers.Rescaling(1./255)


# Apply the layer to the dataset by calling the `map` method which returns a batched dataset by applying the layer on each element of the dataset in parallel 
normalized_ds = train_ds.map(lambda x,y: (normalization_layer(x),y))

# Get the first image and its label from the normalized dataset
image_batch,label_batch = next(iter(normalized_ds))
# Look at the first image in the batch
first_image = image_batch[0]

# Notice the pixel values are now in `[0,1]`.
# min and max values are now 0 and 1 respectively
print(np.min(first_image),np.max(first_image))


# AUTOTUNE is a constant that will be passed to the `prefetch` method and used to tune the performance of the dataset
AUTOTUNE = tf.data.AUTOTUNE
# Configure the dataset for performance by caching and prefetching the images and labels in parallel using the `prefetch` method 
# This is used to improve the performance of the dataset
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = len(class_name)

# Build the sequential model which is a linear stack of layers
model = tf.keras.Sequential([
    # Add the normalization layer
    normalization_layer,
    # Add the convolutional layers and max pooling layers
    # 32 is the number of filters and 3 is the kernel size
    layers.Conv2D(32, 3,  activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3,  activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    # Flatten the output of the convolutional layers
    layers.Flatten(),
    # Dropout is used to prevent overfitting when there are many parameters
    layers.Dropout(0.25),
    # Dense layer with 128 units is the fully connected layer
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
# compile model means the model is ready for training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# summary of the model tells us the architecture of the model
model.summary()


%%time
early_stopping=EarlyStopping(patience=15)
# Fit the model
history=model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stopping])
# Evaluate the model
loss, accuracy=model.evaluate(val_ds)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


# plot the Training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Model loss')
plt.show()
# plot the Training  and validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.title('Model accuracy')
plt.show()


# Load and preprocess the test images
test_image_paths = ['/kaggle/input/paddy-disease-classification/test_images/200001.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200002.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200003.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200004.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200005.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200006.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200007.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200008.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200009.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200010.jpg']# List of test image paths

# Initialize an empty list to store the test images
test_images = []
# Loop over the test image paths
for image_path in test_image_paths:
    # Load the image for preprocessing and prediction using the load_img function
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    # Convert the image to a numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Rescale pixel values to [0, 1]
    # Add an extra dimension to match the input shape of the model
    test_images.append(image)
# Convert the list of test images to a numpy array
test_images = np.array(test_images)

# Predict on the test images
predictions = model.predict(test_images)


# Save the trained model to a file named 'fashion_mnist_cnn.h5'
# This allows you to reuse the model later without retraining
model.save('Rice_Disease_Prediction.h5')

# Print a message to confirm the model was saved
print("Model saved â¬†ï¸� successfully")



# Save the predictions to a CSV file
# argmax returns the index of the maximum value in each row which means the index of the predicted class that have the highest probability
class_labels = np.argmax(predictions, axis=1)

# Create a DataFrame with the test image paths and the predicted class labels and save it to a CSV file
submission_df = pd.DataFrame({'image_id': test_image_paths, 'Label': class_labels})
submission_df.to_csv('submission.csv', index=False)

# Print a message to confirm the predictions were saved
print("Predictions saved â¬†ï¸� successfully")

