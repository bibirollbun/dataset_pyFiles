import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import cv2
import os


# Fixed for our Cats & Dogs classes
NUM_CLASSES = 2

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
resnet_base = ResNet50(
    include_top=False,
    weights='imagenet',  # هنا نستخدم weights الجاهزة
    pooling='avg',
    input_shape=(224, 224, 3)
)
resnet_base.trainable = False
#Still not talking about our train/test data or any pre-processing.
model = Sequential()

# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(resnet_base)
model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False


model.summary()


from tensorflow.keras.optimizers import SGD

sgd = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    optimizer=sgd,
    loss=OBJECTIVE_FUNCTION,
    metrics=LOSS_METRICS
)



import os

print(os.listdir('../input'))



import zipfile
import os

zip_path = '../input/dogs-vs-cats-redux-kernels-edition/train.zip'
extract_path = '/kaggle/working/train'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(os.listdir(extract_path))



import os
import pandas as pd

train_dir = '/kaggle/working/train/train'

filenames = os.listdir(train_dir)
categories = []

for fname in filenames:
    if fname.startswith('cat'):
        categories.append('cat')
    else:
        categories.append('dog')

df = pd.DataFrame({
    'filename': filenames,
    'class': categories
})



from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)



train_generator = datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col='filename',
    y_col='class',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_TRAINING,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_dataframe(
    valid_df,
    directory=train_dir,
    x_col='filename',
    y_col='class',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical'
)



# Max number of steps that these generator will have opportunity to process their source content
# len(train_generator) should be 'no. of available train images / BATCH_SIZE_TRAINING'
# len(valid_generator) should be 'no. of available train images / BATCH_SIZE_VALIDATION'
(BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(validation_generator))


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

cb_checkpointer = ModelCheckpoint(
    filepath='best_model.keras',   # الأفضل .keras
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

cb_early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)



# Grid Search is an ideal candidate for distributed machine learning
# Pseudo code for hyperparameters Grid Search

'''
from sklearn.grid_search import ParameterGrid
param_grid = {'epochs': [5, 10, 15], 'steps_per_epoch' : [10, 20, 50]}

grid = ParameterGrid(param_grid)

# Accumulate history of all permutations (may be for viewing trend) and keep watching for lowest val_loss as final model
for params in grid:
    print(params)
'''


fit_history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=STEPS_PER_EPOCH_VALIDATION,
    callbacks=[cb_checkpointer, cb_early_stopper]
)



print(fit_history.history.keys())


plt.figure(1, figsize=(15, 8))

plt.subplot(221)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.show()

