!pip install keras-tuner --upgrade


# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import datasets, layers, models
from tensorflow.math import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import cv2
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
train_dir = "/kaggle/input/dog-breed-identification/train"
test_dir = "/kaggle/input/dog-breed-identification/test"


def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    feature_maps = feature_extractor.predict(data, batch_size=64, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


dog_breeds = sorted(list(set(train['breed'])))
n_classes = len(dog_breeds)
print(n_classes)
dog_breeds[:5]


class_to_num = dict(zip(dog_breeds, range(n_classes)))



def images_to_array(data_dir, labels_dataframe, img_size = (224,224,3)):
    images_names = labels_dataframe['id']
    images_labels = labels_dataframe['breed']
    data_size = len(images_names)
    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    y = np.zeros([data_size,1], dtype=np.uint8)

    for i in tqdm(range(data_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name+'.jpg')
        img_pixels = load_img(img_dir, target_size=img_size)
        X[i] = img_pixels

        image_breed = images_labels[i]
        y[i] = class_to_num[image_breed]

    y = to_categorical(y)

    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]
    print('Output Data Size: ', X.shape)
    print('Output Label Size: ', y.shape)
    return X, y


from tensorflow.keras.preprocessing.image import load_img
from tqdm import tqdm
from keras.utils import to_categorical


img_size = (224,224, 3)
X, y = images_to_array(train_dir, train, img_size)


from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  img_size, X)



from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 img_size, X)


from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
mobilenet_v2_preprocessor = preprocess_input
mobilenet_v2_features = get_features(MobileNetV2,
                                   mobilenet_v2_preprocessor,
                                   img_size, X)


del X
final_features = np.concatenate([inception_features,
                                 xception_features,
                                 mobilenet_v2_features], axis=-1)
print('Final feature maps shape', final_features.shape)


from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(final_features, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



from keras.callbacks import EarlyStopping
EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
my_callback=[EarlyStop_callback]
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


model = models.Sequential([
    InputLayer(X_train.shape[1:]),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(n_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=my_callback, batch_size=128)



test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()




