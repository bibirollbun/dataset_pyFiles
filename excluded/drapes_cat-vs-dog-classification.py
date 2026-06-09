

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))






import zipfile
zip_files = ['test1','train']
for zip_file in zip_files:
    with zipfile.ZipFile(f'/kaggle/input/dogs-vs-cats/{zip_file}.zip','r') as file:
        file.extractall(".")
        print("{} unzipped successfully".format(zip_file))


print(os.listdir("."))
train_path="./train"
file_names=os.listdir(train_path)
print("train samples ",len(file_names))
print(file_names[:3])


os.mkdir("/kaggle/working/final_train")
os.mkdir("/kaggle/working/final_train/dog")
os.mkdir("/kaggle/working/final_train/cat")
dog_dest="/kaggle/working/final_train/dog"
cat_dest="/kaggle/working/final_train/cat"




def make_folder(train_path):
    for file in os.listdir(train_path):
        if file.split(".")[0]=='dog':
            src_path = os.path.join(train_path, file)
            dst_path = os.path.join(dog_dest, file)
            os.rename(src_path, dst_path)
        elif file.split(".")[0]=='cat':
            src_path = os.path.join(train_path, file)
            dst_path = os.path.join(cat_dest, file)
            os.rename(src_path, dst_path)


make_folder(train_path)


# tf.keras.preprocessing.image_dataset_from_directory is used to take data from a directory,
# where data is divided into categories by folder
dataset=tf.keras.preprocessing.image_dataset_from_directory('/kaggle/working/final_train',batch_size=256, image_size=(128, 128),)


# Get basic information
for images, labels in dataset.take(1):
    print("Batch shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Number of classes:", len(dataset.class_names))
    print("Class names:", dataset.class_names)

# Count total number of samples
total_samples = 0
for images, labels in dataset:
    total_samples += len(images)
print("Total number of images:", total_samples)

# Get batch size from the first batch
batch_size = next(iter(dataset))[0].shape[0]
print("Batch size:", batch_size)


# This os done for normailzation of the pixels 0-white 255 represents black
dataset = dataset.map(lambda x, y: (x/255.0, y))



data_iterator = dataset.as_numpy_iterator() # Creates numpy array from the data as batch
batch = data_iterator.next() # Takes a batch from the data


import cv2
# 1 is dog 0 is cat


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# Dividing the train, test, and val data
train_size = int(len(dataset)*.7)
val_size = int(len(dataset)*.2)
test_size = int(len(dataset)*.1)
train = dataset.take(train_size)
val = dataset.skip(train_size).take(val_size)
test = dataset.skip(train_size+val_size).take(test_size)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D


model=Sequential()


model.add(Conv2D(32,(3,3), 1, activation='relu', input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), 1, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


model.summary()


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


logdir='log'


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


os.listdir('/kaggle/working/test1')


import cv2
import matplotlib.pyplot as plt

# Read the image using OpenCV
# img = cv2.imread('/kaggle/working/test1/11878.jpg')
img = cv2.imread('/kaggle/input/image-single/images.jpg')


# Display the image using Matplotlib
plt.imshow(img)
plt.show()



resize = tf.image.resize(img, (128,128))
yhat = model.predict(np.expand_dims(resize/255, 0))
# model.predict()


if yhat > 0.5: 
    print(f'Predicted class is dog')
else:
    print(f'Predicted class is cat')


print(yhat)


for file in os.listdir("/kaggle/working/test1"):
    img = cv2.imread('/kaggle/input/image-single/images.jpg')


# Display the image using Matplotlib
plt.imshow(img)
plt.show()


#  Save model
model.save(os.path.join('models','imageclassifier.h5'))


# import os
# import tensorflow as tf
# import pandas as pd
# import numpy as np

# # Define the function to create CSV of predictions
# def create_prediction_csv(model, folder_path, csv_filename):
#     # Initialize a list to store results
#     predictions = []

#     # Loop through the images in the folder
#     for img_name in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_name)
        
#         # Check if the file is an image (you can add other checks for supported formats)
#         if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # Load and preprocess the image
#             img = tf.keras.preprocessing.image.load_img(img_path)
#             img = tf.keras.preprocessing.image.img_to_array(img)

#             # Resize image using tf.image.resize as you mentioned
#             img_resized = tf.image.resize(img, (128, 128))

#             # Make the image ready for prediction (normalize and expand dimensions)
#             img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
#             img_resized = img_resized / 255.0  # Normalize the image

#             # Get model prediction
#             prediction = model.predict(img_resized)
#             predicted_class = np.argmax(prediction)  # Assuming it's a classification task

#             # Append the result to the list
#             predictions.append([img_name, predicted_class])

#     # Create a DataFrame from the predictions
#     df = pd.DataFrame(predictions, columns=['Image', 'Predicted_Class'])

#     # Save the DataFrame as a CSV file
#     df.to_csv(csv_filename, index=False)

#     print(f'CSV file created: {csv_filename}')

# # Usage example:
# # Assuming you have a trained model and an image folder path
# # model = tf.keras.models.load_model('your_model.h5')
# create_prediction_csv(model, '/kaggle/working/test1', 'predictions.csv')





