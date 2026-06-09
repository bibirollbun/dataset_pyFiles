# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


# import train.csv file
data = pd.read_csv('/kaggle/input/paddy-disease-classification/train.csv')
data.head()


data.shape


data['label'].unique().tolist()


data['variety'].unique().tolist()


data['age'].describe()


# plot the data count

fig, axes = plt.subplots(1, 1,figsize=(21, 7))
sns.countplot(data, x='variety', ax=axes)
plt.title('Variety Distribution in the dataset')
plt.show()


# plot the data count

fig, axes = plt.subplots(1, 1,figsize=(21, 7))
sns.countplot(data, x='label', ax=axes)
plt.title('Disease Distribution in the dataset')
plt.show()


normal = data[data['label'] =='normal']
normal = normal[normal['variety'] =='ADT45']
five_normals = normal.image_id[:5].values
five_normals.tolist()


dead = data[data['label'] =='dead_heart']
dead = dead[dead['variety'] =='ADT45']
five_deads= dead.image_id[:5].values
five_deads.tolist()


import os
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20, 10))
columns = 5
path = '/kaggle/input/paddy-disease-classification/train_images/'

# Assuming five_normals and five_deads are lists of image filenames
for i, image_loc in enumerate(np.concatenate((five_normals, five_deads))):
    plt.subplot(10 // columns + 1, columns, i + 1)
    
    if i < 5:
        image_path = os.path.join(path, "normal", image_loc)
        plt.title("Normal")
    else:
        image_path = os.path.join(path, "dead_heart", image_loc)
        plt.title("Dead Heart Disease")
        
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis('off')  # Hide axes for better visualization

plt.show()


data.head()


# encode both column for label and variety
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
data['variety'] = le.fit_transform(data['variety'])

data.head()


# define parameters
batch_size = 16
img_height = 224
img_width = 224


train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    seed = 123,
    validation_split=0.2,
    subset="training",
    image_size = (img_height, img_width),
    batch_size = batch_size,
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    seed = 123,
    validation_split=0.2,
    subset="training",
    image_size = (img_height, img_width),
    batch_size = batch_size,
)


class_name = train_ds.class_names
print(class_name)


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


normalization_layer = tf.keras.layers.Rescaling(1./255)


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image))


num_classes = len(class_name)
num_classes

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])


model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])



%%time
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.callbacks import EarlyStopping
# Define the callback function
early_stopping = EarlyStopping(patience=5)

history= model.fit(train_ds,
          validation_data=val_ds,
          epochs=5,
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





import os
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20, 10))
columns = 5
path = '/kaggle/input/paddy-disease-classification/train_images/'

# Assuming five_normals and five_deads are lists of image filenames
for i, image_loc in enumerate(np.concatenate((five_normals, five_deads))):
    plt.subplot(10 // columns + 1, columns, i + 1)
    
    if i < 5:
        image_path = os.path.join(path, "normal", image_loc)
        plt.title("Normal")
    else:
        image_path = os.path.join(path, "dead_heart", image_loc)
        plt.title("Dead Heart Disease")
        
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis('off')  # Hide axes for better visualization

plt.show()

