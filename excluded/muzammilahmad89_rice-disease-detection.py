# importing the important libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf 
from tensorflow import keras 
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout


# Setting a seed value ensures that the random splitting of the dataset into 
# training and validation sets is reproducible, meaning the same split will be
# obtained each time the code is run.
seed = 123  

# The image_dataset_from_directory function is used to create training and
# validation datasets from image files, allowing for easy loading and preprocessing 
# of image data for our model.

train_ds = keras.utils.image_dataset_from_directory(
    directory='/kaggle/input/paddy-disease-classification/train_images',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="training",
    seed=seed  # Add seed argument
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='/kaggle/input/paddy-disease-classification/train_images',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="validation",
    seed=seed  # Add seed argument
)

# Specifying labels="inferred" indicates that the class labels are inferred from the directory structure,
# where each subdirectory represents a different class.
# Using label_mode="int" specifies that the class labels are represented as integers,
# which is suitable for classification tasks.
# The datasets are split into training and validation subsets using validation_split=0.2,
# where 20% of the data is reserved for validation to evaluate the model's performance.



# Normalizing the data 
def process(image,label):
    image = tf.cast(image/255,tf.float32)
    return image,label
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)


import os

# Get the directory path from the DirectoryIterator object
dataset_path = '/kaggle/input/paddy-disease-classification/train_images'

# Count the number of subdirectories (classes)
num_classes = len([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])

print("Number of classes:", num_classes)




# Create CNN model 
# create CNN model

model = Sequential()


model.add(Conv2D(128,kernel_size=(3,3),padding='valid',strides = 1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(3,3),padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',strides = 1,activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),padding='valid'))

model.add(Conv2D(32,kernel_size=(3,3),padding='same',strides = 1,activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),padding='valid'))

model.add(Conv2D(16,kernel_size=(3,3),padding='same',strides = 1,activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes,activation='softmax'))


model.summary()


from tensorflow.keras.optimizers.legacy import Adam

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,  # Set to the value of val_loss at the desired epoch
    restore_best_weights=False,
)




# real waly images use ni hongy jo images transform han wo use hongy
history = model.fit(train_ds, validation_data=validation_ds, epochs=500,callbacks=[early_stopping] )




# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
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
    image_size=(256, 256),
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



submission_df.head()

