import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import cv2

from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, BatchNormalization, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
print("Completed")


TEST_LABELS_RUTE = '/kaggle/input/street-view-getting-started-with-julia/trainLabels.csv'
TRAIN_IMG_DIR = '/kaggle/output/street-view-getting-started-with-julia/trainResized/'
TEST_IMG_DIR = '/kaggle/output/street-view-getting-started-with-julia/testResized/'
TRAIN_IMG_ZIP = '/kaggle/input/street-view-getting-started-with-julia/trainResized.zip'
TEST_IMG_ZIP = '/kaggle/input/street-view-getting-started-with-julia/testResized.zip'
BASE_DIR = '/kaggle/output/street-view-getting-started-with-julia/'


def data_ext(Dir, Zip):
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    zip_ref = zipfile.ZipFile(Zip)
    zip_ref.extractall(BASE_DIR)
    zip_ref.close()


def data_description(df):
    print("Data description")
    print(f"Total number of records {df.shape[0]}")
    print(f'number of features {df.shape[1]}\n\n')
    columns = df.columns
    data_type = []
    
    # Get the datatype of features
    for col in df.columns:
        data_type.append(df[col].dtype)
        
    n_uni = df.nunique()
    # Number of NaN values
    n_miss = df.isna().sum()
    
    names = list(zip(columns, data_type, n_uni, n_miss))
    variable_desc = pd.DataFrame(names, columns=["Name","Type","Unique levels","Missing"])
    print(variable_desc)


data_ext(TRAIN_IMG_DIR, TRAIN_IMG_ZIP) # Train data
data_ext(TEST_IMG_DIR, TEST_IMG_ZIP) # Test data

train_data = pd.read_csv(TEST_LABELS_RUTE)


train_data.head(15)


data_description(train_data)


train_data['img'] = [TRAIN_IMG_DIR + str(id) + '.Bmp' for id in train_data['ID'].values]


lab=[]
test=[]

for i in os.listdir(TEST_IMG_DIR):
    test.append(io.imread(TEST_IMG_DIR+i,as_gray=True))
    lab.append(i.split('.')[0])


test_img=np.array([cv2.resize(image,(28,28)) for image in test])
test_img=test_img[:,:,:,np.newaxis]
test_img.shape


X = train_data.drop("Class", axis = 1)
y = train_data["Class"]
print("Completed")


label = train_data['Class']
unique_labels = list(set(label))

label_id = [unique_labels.index(l) for l in label]
label_id = np.array(label_id, dtype=np.float32)
train_data['label'] = label_id


train_data.head()


y.value_counts()


result = y.value_counts()
plt.figure(figsize = (20, 5))
plt.title("Number of items b in each category")
plt.xlabel("Characters")
plt.ylabel("Number of characters")
result.plot.bar(width = 0.8)


X = X.drop("ID", axis = 1)


print(f'Number of images: {X.shape[0]}\nNumer of pixels per image {X.shape[1]}')


X.head()


train_data_img = []
for img_path in train_data["img"]:
    img = image.imread(img_path)
    data = np.asarray(img)
    if data.shape != (20, 20, 3):
        data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    train_data_img.append(data)
    
    
img_data = np.asarray(train_data_img, dtype=np.uint8)
print("Completed")


# final_train=[cv2.resize(image,(28,28)).flatten() for image in train_data_img]


img_data.shape


# See if the number of images its the same in X and in img_data
print(f"X: {X.shape[0]} size \nimg_data: {img_data.shape[0]} size")


for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(img_data[i])


datagen = ImageDataGenerator (
    zoom_range = 0.2,
    rescale = 1./255,
    rotation_range = 5.0,
    shear_range = 3.0,
    brightness_range = [0.0, 3.0]
)


X = img_data
X = np.array(X).astype(float) / 255 # Normalize the data

y = np.array(train_data["label"])


datagen.fit(X)
plt.figure(figsize = (20, 8))
datagen.flow(X, y, batch_size = 10, shuffle = False)


for img, label in datagen.flow(X, y, batch_size = 10, shuffle = False):
    for i in range(10):
        plt.subplot(2,5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img[i])
    break


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 0) # Test size = 25% of the data


data_gen_train = datagen.flow(X_train, y_train, batch_size = 20)


num_pos = len(train_data["Class"].value_counts()) # Number of total posible results


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape=(20,20,3)),
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation = "relu"),
   tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation = "relu"),
    
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(125, activation = "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(250, activation = "relu"),
    tf.keras.layers.Dense(500, activation = "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_pos, activation='softmax')
])
print("Model created")


model.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])
print("Model compiled")


es_callbacks = [tf.keras.callbacks.EarlyStopping(patience = 10, monitor = "accuracy")]


history = model.fit(data_gen_train,
             validation_data = (X_val, y_val),
             epochs = 100, batch_size = 50, callbacks = es_callbacks, # Epochs number of times to repet the proces
             validation_steps = 5)


history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()


test_imgs = []
names = []
for dirname, _, filenames in os.walk(TEST_IMG_DIR):
    for filename in filenames:
        test_imgs.append(os.path.join(dirname, filename))
        names.append(os.path.splitext(filename)[0])
test_imgs = np.array(test_imgs)
names = np.array(names)


test_data_img_list = []
for img_path in test_imgs:
    img = image.imread(img_path)
    data = np.asarray(img)
    if data.shape != (20,20,3):
        data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    data = data / 255.
    test_data_img_list.append(data)
    
test_data_img = np.asarray(test_data_img_list)


pred = model.predict(test_data_img)


test = []
res = []
for i in range(0, len(pred)):
    res.append(unique_labels[np.argmax(pred[i])])
    test.append(np.argmax(pred[i]))


results = []
test = []
for i in range(0, len(pred)):
    results.append(unique_labels[np.argmax(pred[i])])
    test.append(np.argmax(pred[i]))

submission = pd.DataFrame(names, columns=['ID'])
submission['Class'] = results



submission.head(20)


submission.to_csv('submission.csv', index = False)

