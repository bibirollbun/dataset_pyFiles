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
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB7

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


TRAIN_IMG_PATH = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train'
TEST_IMG_PATH = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test'


train_metadata = pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv')
test_metadata =  pd.read_csv('/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv')


class_weights = compute_class_weight(
    'balanced',  
    classes=np.unique(train_metadata.label),  
    y=train_metadata.label 
)

class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


def custom_preprocessing(image):
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (image - mean) / std



train_datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    preprocessing_function=custom_preprocessing,
    validation_split=0.2
)

train_img = train_datagen.flow_from_dataframe(
    dataframe=train_metadata,
    directory=TRAIN_IMG_PATH,
    x_col='image',
    y_col='label',
    target_size=(512, 512),
    batch_size=8,
    class_mode='binary',
    subset='training'
)



datagen = ImageDataGenerator(rescale=1./255)



val_img = train_datagen.flow_from_dataframe(
    dataframe=train_metadata,
    directory=TRAIN_IMG_PATH,
    x_col='image',
    y_col='label',
    target_size=(512, 512),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)


test_img = datagen.flow_from_dataframe(
    dataframe=test_metadata,
    directory=TEST_IMG_PATH,
    x_col='image',
    target_size=(512, 512),
    batch_size=8,
    class_mode=None,
    shuffle=False
)


test_img = datagen.flow_from_dataframe(
    dataframe=test_metadata,
    directory=TEST_IMG_PATH,
    x_col='image',
    target_size=(512, 512),
    batch_size=8,
    class_mode=None,
    shuffle=False
)


class_names = {v: k for k, v in train_img.class_indices.items()}


input_shape = (512,512,3)
num_classes = 1

image_input = layers.Input(shape=input_shape)
effnet = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
x = effnet(image_input)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)  
x = layers.Dense(1024, activation='relu')(x) 
x = layers.Dense(num_classes,activation='sigmoid')(x)
model =  models.Model(inputs=image_input, outputs=x)
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy','AUC']) 

chkpnt_loss = ModelCheckpoint(
    'best_model_loss.keras',           
    monitor='val_loss',          
    verbose=1,                  
    save_best_only=True,        
    mode='min',                 
    save_weights_only=False,     
)

chkpnt_auc = ModelCheckpoint(
    'best_model_auc.keras',            
    monitor='val_AUC',         
    verbose=1,                  
    save_best_only=True,       
    mode='max',                
    save_weights_only=False,    
)


history = model.fit(train_img,
                    validation_data=val_img,
                    epochs=64,
                    class_weight=class_weight_dict,
                    callbacks=[chkpnt_loss,chkpnt_auc])


plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


filenames = test_img.filenames
predictions = model.predict(test_img)
predicted_labels = (predictions > 0.5).astype(int)
submission = pd.DataFrame({'image': filenames, 'label': predicted_labels.flatten()})
submission.to_csv('submission.csv', index=False)


best_model = tf.keras.models.load_model('best_model_auc.keras')
predictions = best_model.predict(test_img)
submission = pd.DataFrame({'image': test_metadata['image'], 'label': predictions.flatten()})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

