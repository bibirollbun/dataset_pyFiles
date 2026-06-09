import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import shuffle


TRAIN_DIR = '/kaggle/input/facial-keypoints/training.csv'
TEST_DIR = '/kaggle/input/facial-keypoints/test.csv'
LOOK_ID_DIR = '/kaggle/input/facial-keypoints-detection/IdLookupTable.csv'

TRAIN_DATA = pd.read_csv(TRAIN_DIR)  
TEST_DATA = pd.read_csv(TEST_DIR)
LOOK_ID_DATA = pd.read_csv(LOOK_ID_DIR)


print(TRAIN_DATA.shape)
print(TEST_DATA.shape)


TRAIN_DATA.head()


TRAIN_DATA['Image'] = TRAIN_DATA['Image'].apply(lambda im: np.fromstring(im, sep=' '))
TRAIN_DATA.dropna(inplace=True)
TRAIN_DATA.isnull().any().value_counts()


def load_data(dirname, test=False):
    data = pd.read_csv(dirname)

    if not test:
        data = data.dropna()

    data['Image'] = data['Image'].apply(lambda img: np.fromstring(img, sep=' '))
    
    imgs = np.vstack(data['Image'].values) / 255.0
    imgs = imgs.reshape(-1, 96, 96, 1).astype(np.float32)

    if not test:
        points = data[data.columns[:-1]].values
        points = (points - 48) / 48.0 
        points = points.astype(np.float32)
        imgs, points = shuffle(imgs, points, random_state=42)
        return imgs, points
    
    else:
        return imgs

X_train, y_train = load_data(TRAIN_DIR)
X_test = load_data(TEST_DIR, test=True)


X_train.shape, y_train.shape


plt.imshow(np.squeeze(X_train[1]), cmap='gray')


def plotKeyPoints(img, points):
    plt.imshow(img,cmap='gray')

    for i in range(0,30,2):
        plt.scatter((points[i] + 1) * 48, (points[i+1] + 1) * 48, color='red')


        
plotKeyPoints(X_train[1],y_train[1])


def augment_data(img, points):
    new_img = np.fliplr(img)
    new_points = np.copy(points)
    new_points[::2] = -new_points[::2]
    return new_img, new_points

flip_img, flip_points = augment_data(X_train[1], y_train[1])
plotKeyPoints(flip_img, flip_points)


final_X_train = []
final_y_train = []

# apply flipping operation to each example in the training set
for i in range(0,X_train.shape[0]):
    aug_img, aug_point = augment_data(X_train[i], y_train[i])
     # append the original data first
    final_X_train.append(X_train[i])
    final_y_train.append(y_train[i]) 
    
    # then the augmented data
    final_X_train.append(aug_img)
    final_y_train.append(aug_point) 

# convert to numpy
final_X_train = np.array(final_X_train)   
final_y_train = np.copy(final_y_train)

print(final_X_train.shape)
print(final_y_train.shape)


plotKeyPoints(final_X_train[2], final_y_train[2])


plotKeyPoints(final_X_train[3], final_y_train[3])


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError


final_X_train = final_X_train.reshape(final_X_train.shape[0], 96, 96, 1)


model = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(96, 96, 1)),
    # BatchNormalization(),
    MaxPooling2D(2),
    
    Conv2D(32, 3, activation='relu'),
    # BatchNormalization(),
    MaxPooling2D(2),
    
    Conv2D(64, 3, activation='relu'),
    # BatchNormalization(),
    MaxPooling2D(2),
    
    Conv2D(128, 3, activation='relu'),
    # BatchNormalization(),
    MaxPooling2D(2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    # BatchNormalization(),
    
    Dense(512, activation='relu'),
    Dropout(0.2),
    # BatchNormalization(),
    
    Dense(30, activation='tanh')  
])

model.compile(loss='mse', optimizer='adam', metrics=[MeanAbsoluteError()])
model.summary()


checkpoint = ModelCheckpoint(
    'best_model_full.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False
)


callbacks = [checkpoint]

hist = model.fit(
    final_X_train,
    final_y_train,
    validation_split=0.2,
    batch_size=64,
    shuffle=True,
    epochs=150,
    verbose=1,
    callbacks=callbacks
)


plt.figure(figsize=(16, 5))

# Loss Plot (MSE)
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.plot(hist.history['loss'], color='b', label='Training Loss')
plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')

# MAE Plot
plt.subplot(1, 2, 2)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
plt.plot(hist.history['mean_absolute_error'], color='b', label='Training MAE')
plt.plot(hist.history['val_mean_absolute_error'], color='r', label='Validation MAE')
plt.legend(loc='upper right')

plt.show()



fig = plt.figure(figsize=(20,20))
# make test images keypoints prediction
points_test = model.predict(X_test.reshape(X_test.shape[0], 96, 96, 1))

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    plotKeyPoints(X_test[i], np.squeeze(points_test[i]))




