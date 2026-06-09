import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


facialexpression_df = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
facialexpression_df.head()


facialexpression_df[' pixels'][0] 


def string2array(x):
  return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')


facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: string2array(x))


facialexpression_df.head()


facialexpression_df.isnull().sum()


label_to_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sad', 5: 'surprise', 6: 'neutral'}


num_rows = 2
num_cols = 4

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
axes = axes.flatten()

emotions = [0, 1, 2, 3, 4, 5, 6]

for i in emotions:
    data = facialexpression_df[facialexpression_df['emotion'] == i][:1]
    img = data[' pixels'].item()
    img = img.reshape(48, 48) 
    ax = axes[i]
    ax.set_title(label_to_text[i])
    ax.imshow(img, cmap='gray')

for i in range(len(emotions), num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout()

plt.show()


facialexpression_df.emotion.value_counts().index


facialexpression_df.emotion.value_counts()


plt.figure(figsize = (10,3))
sns.barplot(x = facialexpression_df.emotion.value_counts().index, y = facialexpression_df.emotion.value_counts())


# Separate data based on 'Usage' column
train_data = facialexpression_df[facialexpression_df[' Usage'] == 'Training']
test_data = facialexpression_df[facialexpression_df[' Usage'] == 'PublicTest']
val_data = facialexpression_df[facialexpression_df[' Usage'] == 'PrivateTest']

#Training Data
X_train = train_data[' pixels']
y_train = train_data['emotion']

#Testing Data
X_test = test_data[' pixels']
y_test = test_data['emotion']

#Validation Data
X_val = val_data[' pixels']
y_val = val_data['emotion']


X_train = train_data[' pixels']
y_train = to_categorical(train_data['emotion'])

X_test = test_data[' pixels']
y_test = to_categorical(test_data['emotion'])

X_val = val_data[' pixels']
y_val = to_categorical(val_data['emotion'])


X_train = np.stack(X_train, axis = 0)
X_train = X_train.reshape(28709 , 48, 48, 1)

X_test = np.stack(X_test, axis = 0)
X_test = X_test.reshape(3589 , 48, 48, 1)

X_val = np.stack(X_val, axis = 0)
X_val = X_val.reshape(3589 , 48, 48, 1)


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)


X_train = X_train/255
X_val   = X_val /255
X_Test  = X_test/255


train_datagen = ImageDataGenerator(
rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = "nearest")


# BUILD DEEP RESIDUAL NEURAL NETWORK FOR FACIAL POINTS DETECTION MODEL
def res_block(X, filter, stage):

  # Convolutional_block
  X_copy = X

  f1 , f2, f3 = filter

  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = MaxPool2D((2,2))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
  X = Activation('relu')(X)
  # stride equal to 1 , that means they're going to be shifting and scanning only by one pixel
  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
  X = Activation('relu')(X)

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


  # Short path
  X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
  X_copy = MaxPool2D((2,2))(X_copy)
  X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 1
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
  X = Activation('relu')(X)

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
  X = Activation('relu')(X)

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 2
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
  X = Activation('relu')(X)

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
  X = Activation('relu')(X)

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X


input_shape = (48, 48, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides= (2, 2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides= (2, 2))(X)

# 2 - stage
X = res_block(X, filter= [64, 64, 256], stage= 2)

# 3 - stage
X = res_block(X, filter= [128, 128, 512], stage= 3)

# Average Pooling
X = AveragePooling2D((2, 2), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(7, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)

model_emotion = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model_emotion.summary()


# train the network
model_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


earlystopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath = "Emotion_weights.keras", verbose = 1, save_best_only=True)


history = model_emotion.fit(train_datagen.flow(X_train, y_train, batch_size=64),
	validation_data=(X_val, y_val), steps_per_epoch=len(X_train) // 64,
	epochs= 100, callbacks=[checkpointer, earlystopping])


# saving the model architecture to json file

model_json = model_emotion.to_json()
with open("FacialExpression-model.json","w") as json_file:
  json_file.write(model_json)


score = model_emotion.evaluate(X_Test, y_test)
print('Test Accuracy: {}'.format(score[1]))


predicted_classes = np.argmax(model_emotion.predict(X_Test), axis=-1)
y_true = np.argmax(y_test, axis=-1)


y_true.shape


cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, cbar = False)


print(classification_report(y_true, predicted_classes))


from sklearn.utils import resample

# Calculate the average class count
average_class_count = facialexpression_df['emotion'].value_counts().mean()

# Separate data for each class
class_data = [facialexpression_df[facialexpression_df['emotion'] == i] for i in range(7)]

# Oversample to balance classes based on average class count
oversampled_data = [resample(class_df, replace=True, n_samples=int(average_class_count), random_state=42) for class_df in class_data]
balanced_data = pd.concat(oversampled_data)


balanced_data.head()


plt.figure(figsize = (10,3))
sns.barplot(x = balanced_data.emotion.value_counts().index, y = balanced_data.emotion.value_counts())


X_train, X_temp, y_train, y_temp = train_test_split(balanced_data[' pixels'], balanced_data['emotion'], test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)


print("Train set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])


y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_val = to_categorical(y_val)


X_train = np.stack(X_train, axis = 0)
X_train = X_train.reshape(28705 , 48, 48, 1)

X_test = np.stack(X_test, axis = 0)
X_test = X_test.reshape(3589 , 48, 48, 1)

X_val = np.stack(X_val, axis = 0)
X_val = X_val.reshape(3588 , 48, 48, 1)


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)


X_train = X_train/255
X_val   = X_val /255
X_Test  = X_test/255


model_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


earlystopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)

checkpointer = ModelCheckpoint(filepath = "EmotionExpression_weights.keras", verbose = 1, save_best_only=True)


history = model_emotion.fit(X_train, y_train, batch_size=64,
                            validation_data=(X_val, y_val),
                            steps_per_epoch=len(X_train) // 64,
                            epochs=30,
                            callbacks=[checkpointer, earlystopping])


model_json = model_emotion.to_json()
with open("FacialExpression-balanced-model.json","w") as json_file:
  json_file.write(model_json)


score = model_emotion.evaluate(X_Test, y_test)
print('Test Accuracy: {}'.format(score[1]))


predicted_classes = np.argmax(model_emotion.predict(X_Test), axis=-1)
y_true = np.argmax(y_test, axis=-1)


y_true.shape


cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, cbar = False)


L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize=(40, 40))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(48, 48), cmap='gray')
    title = 'Prediction = {}\nTrue = {}'.format(label_to_text[predicted_classes[i]], label_to_text[y_true[i]])
    axes[i].set_title(title, fontsize=30)
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.3)
plt.show()

