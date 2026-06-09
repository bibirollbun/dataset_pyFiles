# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os
import zipfile

import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


os.listdir('/kaggle/input/facial-keypoints-detection')


# unzipfile 
with zipfile.ZipFile('/kaggle/input/facial-keypoints-detection/training.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

with zipfile.ZipFile('/kaggle/input/facial-keypoints-detection/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')


train_data = pd.read_csv('/kaggle/working/training.csv')
test_data = pd.read_csv('/kaggle/working/test.csv')
id_lookup = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')


train_data.head().T


print("Check missing values in the data")
train_data.isnull().any().value_counts()


# filling the null values with their previous values
train_data.ffill(inplace=True)


train_data.info()


imag = []
for i in range(0,7049):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imag.append(img)


# convert grayscale to RGB
image_list = np.array(imag).astype('float32') /255.0 # Normalizing iamge 

# Expand
image_gray = np.expand_dims(image_list.reshape(-1, 96, 96), axis=-1)

image_rgb = np.repeat(image_gray, 3, axis=-1)


image_resized = np.array([cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA) for img in image_rgb])
print("Image reshaped size : ",image_resized.shape)
# Preprocess 
X = preprocess_input(image_resized)


# Drop the 'Image' column and convert coordinates to float32 array
y = train_data.drop(['Image'], axis=1).values
y = y.astype('float32')


# spliting the data into training and validation 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# load the model 
base_model = MobileNetV2(include_top=False,weights='imagenet',input_shape=(224,224,3))

# Freez the base model so we don't retrain it initially
base_model.trainable = False

# Build custom model head 
inputs = base_model.input
x = base_model.output
x = GlobalAveragePooling2D()(x) # feature map 
x = Dropout(0.2)(x)             # Help prevent overfitting
x = Dense(512,activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(30)(x)

# combine into a model 
model = Model(inputs = inputs , outputs = outputs)


# lets define loss 
def rmse(y_true,y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred-y_true)))


# Compile model 
model.compile(
    optimizer = 'adam',
    loss = rmse,
    metrics = ['mae',rmse]
)


# # let's use early stopping 

# early_stop = EarlyStopping(
#     monitor='val_loss',         
#     patience=5,                 
#     restore_best_weights=True,  
#     verbose=1                   
# )

# # Checkpoints model when validation loss improve 
# checkpoint = ModelCheckpoint(
#     'best_model.h5',
#     monitor='val_loss',
#     save_best_only=True,
#     verbose=1
# )



# training the model 
EPOCHS = 30
BATCH_SIZE = 32
history = model.fit(
    X_train,y_train,
    validation_data = (X_val,y_val),
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    # callbacks = [early_stop,checkpoint]
)


# Plot RMSE (custom loss used)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train RMSE')
plt.plot(history.history['val_loss'], label='Val RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Training vs Validation RMSE')
plt.legend()
plt.grid(True)
plt.show()

# Plot Mean Absolute Error
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training vs Validation MAE')
plt.legend()
plt.grid(True)
plt.show()


# calculate accuracy 

def calculate_keypoint_accuracy(y_true, y_pred, threshold=5.0):
    
    # Number of samples Ã— keypoints = total keypoints
    total_keypoints = y_true.shape[0] * (y_true.shape[1] // 2)
    
    # Compute distance per keypoint (x, y => Euclidean norm)
    distances = np.sqrt((y_true - y_pred) ** 2)

    # Group x,y pairs
    distances_pair = distances.reshape(distances.shape[0], -1, 2)  # shape: [n, 15, 2]
    euclidean_distances = np.sqrt(np.sum(distances_pair ** 2, axis=2))  # shape: [n, 15]

    # Count number of distances under threshold
    accurate_points = (euclidean_distances < threshold).sum()
    accuracy = (accurate_points / total_keypoints) * 100

    return round(accuracy, 2)


# Predict on validation set
y_pred = model.predict(X_val)

# Calculate keypoint accuracy with 5-pixel threshold
accuracy = calculate_keypoint_accuracy(y_val, y_pred, threshold=5.0)

print(f"Keypoint Accuracy (â‰¤ 5px error): {accuracy}%")


# Prepaing test data for submission
test_data.head(5)


# get names of 30 coordinate columns
feature_names = list(train_data.columns[:-1]) 


# Preprocess test images 
test_images = test_data['Image'].apply(lambda i: np.array(i.split(), dtype='float32') if isinstance(i, str) else np.zeros((96*96,), dtype='float32'))
test_images = np.stack(test_images.values) / 255.0                       # normalize
test_images = test_images.reshape(-1, 96, 96, 1)                         # reshape
test_rgb = np.repeat(test_images, 3, axis=-1)                           # convert to RGB
test_resized = np.array([cv2.resize(img, (224, 224)) for img in test_rgb])  # resize
X_test = tf.keras.applications.mobilenet_v2.preprocess_input(test_resized)  # final preprocess



# load trained model 
from tensorflow.keras.models import load_model
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




predictions = model.predict(X_test)
                            
# prepare submission file 
row_ids = id_lookup['RowId']
image_ids = id_lookup['ImageId'] - 1   # 0-based index
features = id_lookup['FeatureName']


# Collect the predicted location for each row
locations = []
for img_id, feature in zip(image_ids, features):
    feature_idx = feature_names.index(feature)   # find the column position
    loc = predictions[img_id][feature_idx]       # fetch the prediction
    locations.append(loc)



# builld dataframe and save 
submission = pd.DataFrame({
    'RowId': row_ids,
    'Location': locations
})



submission.to_csv('submission.csv', index=False)
submission.head(5)




