# ignore warnings
import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# unzipping the zip file containing the training data
!unzip "/kaggle/input/facial-keypoints-detection/training.zip"


# unzipping the zip file containing the testing data
!unzip "/kaggle/input/facial-keypoints-detection/test.zip"


# Loaind the train and test CSVs as dataframes
train_df = pd.read_csv('/kaggle/working/training.csv')
test_df = pd.read_csv('/kaggle/working/test.csv')
lookup_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')


# Size of training data
train_df.shape


# Displaying first 5 rows of training data
train_df.head().T


# descriptive statistics
train_df.describe().T


# measure data correlation

CorrelationMatrix = train_df.drop('Image', axis=1).corr()

plt.figure(figsize=(17,10))

mask = np.triu(np.ones_like(CorrelationMatrix, dtype=bool))

sns.heatmap(CorrelationMatrix,
            cmap='RdBu_r',
            annot=True,
            fmt='.2f',
            vmin=-1, vmax=1)

plt.show()


# check for missing values
train_df.isnull().sum()


plt.figure(figsize=(7, 6))
train_df.isnull().sum().plot(kind='barh',color = 'orange')
plt.title("Missing Data Concentration")
plt.show()


# check for duplicated values
train_df.duplicated().any()


# Fill all missing values by mean value in each column
for i in train_df.columns[:-1]:
    train_df[i] = train_df[i].fillna(train_df[i].mean())


# Sanity check (kiểm tra lại các giá trị còn thiếu)
train_df.isnull().sum().any()


import albumentations as A


img_height, img_width = 96, 96


X = []
for row in train_df['Image']:
    face_pixel = np.array(row.split(' '), dtype='float')
    face_pixel = np.reshape(face_pixel, (img_height,img_width))
    X.append(face_pixel)

X = np.array(X, dtype = 'float')


features = train_df.drop('Image',axis = 1)

y = []
for i in range(len(features)):
    y.append(features.iloc[i,:])

y = np.array(y, dtype = 'float')


augmentations = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=1/img_width, scale_limit=0, rotate_limit=1, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0), 
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
        ], p=1.0),
    ], 
    keypoint_params=A.KeypointParams(format='xy')
)


def augment_image(image, keypoints):
    augmented = augmentations(image=image, keypoints=keypoints)
    augmented_image = augmented['image']
    augmented_keypoints = augmented['keypoints']
    return augmented_image, augmented_keypoints

def kpts_to_tuples(keypoints):
    return [(keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 2)]

def tuples_to_kpts(tuples):
    return [coord for point in tuples for coord in point]

def augment_data(X, y):
    X_aug = []
    y_aug = []
    for img, kpt in zip(X, y):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        kpt = kpts_to_tuples(kpt)
        aug_img, aug_kpt = augment_image(img, kpt)
        X_aug.append(aug_img)
        y_aug.append(tuples_to_kpts(aug_kpt))
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    return X_aug, y_aug


X_aug, y_aug = augment_data(X, y)


X_new = np.concatenate((X, X_aug), axis=0)
y_new = np.concatenate((y, y_aug), axis=0)


print('X shape', X.shape)
print('y shape', y.shape)


print('New X shape', X_new.shape)
print('New y shape', y_new.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X_new, y_new, 
    test_size=0.2, 
    random_state=42
)

X_train, X_val, y_train, y_val =  train_test_split(
    X_train, y_train, 
    test_size=0.125, 
    random_state=42
)


print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_val shape:', X_val.shape)
print('y_val shape:', y_val.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


def display_grid(rows, cols, X, y):
    plt.figure(figsize=(10, 4))
    for i in range(rows*cols):
        random_index=np.random.choice(len(X))
        plt.subplot(rows,cols,i+1)
        plt.imshow(X[random_index], cmap='gray')
        plt.scatter(y[random_index,:,0],y[random_index,:,1], marker='.', color='red')
    plt.show()

y_reshaped = y.reshape(y.shape[0], 15, 2)
display_grid(2, 4, X, y_reshaped)


# plot a sample image with different color for each key point
def plot_features(image, feature_names, features_reshaped):
    NUM_COLORS = 15
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    ax.imshow(image, cmap='gray')
    for n, name in enumerate(feature_names):
        ax.scatter(features_reshaped[0,n,0],features_reshaped[0,n,1],marker='.',label=name)
    ax.legend(bbox_to_anchor=(1,1))
    plt.show()

feature_names=[i.replace('_x','') for n, i in enumerate(features) if n%2==0 ]

plot_features(X[np.random.choice(len(X))], feature_names, y_reshaped)


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, layers
from keras.regularizers import l2
from keras.utils import plot_model

from PIL import Image

from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE


def preprocess_image(image, target_size=(299, 299)):
    image = Image.fromarray(image, mode='L') # mode L means grayscale
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    return image

def preprocess_dataset(X, target_size=(299, 299)):
    X_processed = np.array([preprocess_image(img, target_size) for img in X])
    return X_processed


def create_data_generators(batch_size = 32, target_size = (299, 299), preprocessing=None, scale=1):
    X_train_processed = preprocess_dataset(X_train, target_size)
    X_val_processed = preprocess_dataset(X_val, target_size)
    X_test_processed = preprocess_dataset(X_test, target_size)
    
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        rescale=1.0 / scale,
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        rescale=1.0 / scale,
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        rescale=1.0 / scale,
    )
    
    train_generator = train_datagen.flow(
        X_train_processed, 
        y_train, 
        batch_size=batch_size, 
        shuffle=True,
    )
    
    val_generator = val_datagen.flow(
        X_val_processed, 
        y_val, 
        batch_size=batch_size,
    )
    
    test_generator = test_datagen.flow(
        X_test_processed, 
        y_test, 
        batch_size=batch_size,
    )
    
    return train_generator, val_generator, test_generator


def plot_training_process(history):
    # Check if history object contains validation data
    has_val = 'val_loss' in history.history

    # Plot Loss
    plt.figure(figsize=(14, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if has_val:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    if has_val and 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluation(model):
    # Evaluate the model on the test data
    test_loss = model.evaluate(
        test_generator,
        steps=test_generator.n // test_generator.batch_size
    )
    
    predictions = []
    true_labels = []
    for _ in range(len(test_generator)):
        X_batch, y_batch = next(test_generator)
        predictions_batch = model.predict_on_batch(X_batch)
        predictions.append(predictions_batch)
        true_labels.append(y_batch)

    # Convert lists to numpy arrays
    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)

    # Calculate regression metrics
    mae = MAE(true_labels, predictions)
    mse = MSE(true_labels, predictions)
    rmse = mse**(1/2)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)


def feature_extraction(backbone, generator):
    return backbone.predict(generator)


target_size = (96, 96)
batch_size = 32
train_generator, val_generator, test_generator = create_data_generators(
    batch_size = batch_size, 
    target_size = target_size,
    preprocessing = inception_preprocess
)


# check the first item in train generator
X0, y0 = train_generator[0]

print('X shape', X0.shape)
print('y shape', y0.shape)

print(f'Image range: from {np.min(X0)} to  {np.max(X0)}')


# define backbone model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg') # pooling = avg to reduce dimension for ft purpose
inception_model.trainable = False

# feature extraction
train_features = feature_extraction(inception_model, train_generator)
val_features = feature_extraction(inception_model, val_generator)
test_features = feature_extraction(inception_model, test_generator)


# design model architecture

model2 = models.Sequential([
    layers.Reshape((1, 1, train_features.shape[1]), input_shape=(train_features.shape[1],)),
    
    layers.Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)),
    layers.LeakyReLU(alpha = 0.1),
    layers.BatchNormalization(),
    
    layers.Convolution2D(96, (3,3), padding='same', use_bias=False),
    layers.LeakyReLU(alpha = 0.1),
    layers.BatchNormalization(),
    
    layers.Convolution2D(96, (3,3), padding='same', use_bias=False),
    layers.LeakyReLU(alpha = 0.1),
    layers.BatchNormalization(),

    layers.Convolution2D(128, (3,3), padding='same', use_bias=False),
    layers.LeakyReLU(alpha = 0.1),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    
    layers.Convolution2D(256, (3,3), padding='same', use_bias=False),
    layers.LeakyReLU(alpha = 0.1),
    layers.BatchNormalization(),
    
    layers.Convolution2D(256, (3,3), padding='same', use_bias=False),
    layers.LeakyReLU(alpha = 0.1),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(1, 1)),

    layers.Convolution2D(512, (3,3), padding='same', use_bias=False),
    layers.LeakyReLU(alpha = 0.1),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(30), 
])

model2.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])


# training
history = model2.fit(
    train_features, y_train,
    validation_data = (val_features, y_val),
    epochs = 100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, 
            min_delta=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            patience=3,  
            factor=.2,
            min_lr=.00001
        )
    ]
)
plot_training_process(history)
        


# evaluation
model2.evaluate(test_features, y_test)
y_pred = model2.predict(test_features)

mse = MSE(y_test, y_pred)
rmse = mse**(1/2)
mae = MAE(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)


model2.summary()


pretrained_model = InceptionV3(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
pretrained_model.trainable = False

model1 = models.Sequential([
    layers.Convolution2D(3, (1, 1), padding='same', input_shape=(96,96,3)),
    layers.LeakyReLU(alpha = 0.1),
    pretrained_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.1),
    layers.Dense(30),
])

model1.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae']
)


history = model1.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3, 
            min_delta=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            patience=3,  
            factor=.2,
            min_lr=.00001
        )
    ]
)
plot_training_process(history)


model1.evaluate(test_generator)


model1.summary()


best_model = model2


test_df.head()


test_data = []
for row in train_df['Image']:
    face_pixel = np.array(row.split(' '), dtype='float')
    face_pixel = np.reshape(face_pixel, (img_height,img_width))
    test_data.append(face_pixel)

test_data = np.array(test_data, dtype = 'float')
test_data.shape


X_test = preprocess_dataset(test_data, target_size)
X_test.shape


test_features = feature_extraction(inception_model, X_test)


pred = best_model.predict(test_features)


lookup_df.head()


lookid_list = list(lookup_df['FeatureName'])
imageID = list(lookup_df['ImageId']-1)
pre_list = list(pred)
rowid = lookup_df['RowId']
rowid = list(rowid)


feature = []
for f in list(lookup_df['FeatureName']):
    feature.append(lookid_list.index(f))


pred_coord = []
for x, y in zip(imageID, feature):
    pred_coord.append(pre_list[x][y])


rowid = pd.Series(rowid, name = 'RowId')
loc = pd.Series(pred_coord, name = 'Location')


submission = pd.concat([rowid, loc],axis = 1)


submission.to_csv('submission.csv', index = False)

