


# Data Cleaning or Data Working Libraries 
import pandas as pd
import numpy as np

# Visualizing Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Data Preprocessing Library
from sklearn.preprocessing import LabelEncoder


# Deep Learning or Neural Network Libraries

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


df_train = pd.read_csv('/kaggle/input/paddy-disease-classification/train.csv')
df_submission = pd.read_csv('/kaggle/input/paddy-disease-classification/sample_submission.csv')


# Creating the copy of the original dataset
df = df_train.copy()


print(f"No of rows this data contain is {df.shape[0]} and no of columns are {df.shape[1]}");


df.columns.tolist()


df['label'].value_counts()


plt.figure(figsize=(15,5))
sns.countplot(df,x='label',color='#450693')
plt.title("Distribution of Diseases Based on Paddy or leaves")
plt.xlabel("Disease Name")
plt.ylabel("Count of Diseases")
plt.show()


df['variety'].value_counts()


plt.figure(figsize=(15,5))
sns.countplot(df,x='variety',color='#450693')
plt.title("Distribution of Paddy")
plt.xlabel("Paddy Name")
plt.ylabel("Count of paddy Leafs")
plt.show()


df['age'].describe()


normal = df[df['label']=='normal']
normal = normal[normal['variety']=='ADT45']
five_normal = normal.image_id[:5].values
five_normal.tolist()


dead = df[df['label']=='dead_heart']
dead = dead[dead['variety']=='ADT45']
five_dead = dead.image_id[:5].values
five_dead.tolist()


plt.figure(figsize=(20,20))
columns = 5
path = '/kaggle/input/paddy-disease-classification/train_images/'

for i,image_loc in enumerate(np.concatenate((five_normal,five_dead))):
    plt.subplot((10//columns + 1),columns,i+1)

    if i < 5:
        image = plt.imread(path + 'normal/' + image_loc)
        plt.title("Normal")
    else:
        image = plt.imread(path + 'dead_heart/' + image_loc)
        plt.title("Dead Heart Disease")
        
    plt.imshow(image)


images = ['/kaggle/input/paddy-disease-classification/train_images/hispa/106590.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/tungro/109629.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_blight/109372.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/downy_mildew/102350.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/blast/110243.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_streak/101104.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/normal/109760.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/brown_spot/104675.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/dead_heart/105159.jpg',\
          '/kaggle/input/paddy-disease-classification/train_images/bacterial_panicle_blight/101351.jpg',\
         ]
diseases = ['hispa','tungro','bacterial_leaf_blight','downy_mildew','blast','bacterial_leaf_streak',\
           'normal','brown_spot','dead_heart','bacterial_panicle_blight']
diseases = [disease + ' image' for disease in diseases]
plt.figure(figsize=(20,10))
columns = 5
for i, image_loc in enumerate(images):
    plt.subplot(len(images)//columns + 1, columns, i + 1)
    image=plt.imread(image_loc)
    plt.title(diseases[i])
    plt.imshow(image)


le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
df['variety'] = le.fit_transform(df['variety'])


batch_size = 32
img_width = 224
img_height = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    image_size = (img_width,img_height),
    batch_size = batch_size
)



validation_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    image_size = (img_width,img_height),
    batch_size = batch_size
)    


class_name = train_ds.class_names
class_name


for image_batch , label_batch in train_ds:
    print(image_batch.shape)
    print(label_batch.shape)
    break



normalization_layer = tf.keras.layers.Rescaling(1./255)
normalization_ds = train_ds.map(lambda x,y: (normalization_layer(x),y))
image_batch, labels_batch = next(iter(normalization_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


Autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=Autotune)
val_ds = validation_ds.cache().prefetch(buffer_size=Autotune)


num_classes = len(class_name)


# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     tf.keras.layers.Conv2D(128,3,activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(64,3,activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32,3,activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32,3,activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Dense(128,activation='relu'),
#     tf.keras.layers.Dense(num_classes,activation='softmax')
# ])

# model.compile(
#   optimizer='adam',
#   loss=tf.losses.SparseCategoricalCrossentropy(),
#   metrics=['accuracy'])

# # Define the callback function
# early_stopping = EarlyStopping(patience=30)

# history= model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=10,
#           callbacks=[early_stopping])

# # evaluat the model
# loss = model.evaluate(val_ds)

# # Plotting the training and testing los
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()

# # plot the accuracy of training and validation

# # Plotting the training and validation accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='lower right')
# plt.show()




