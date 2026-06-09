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


# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


# read the CSV files

train_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip')
test_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')
idlookup = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')

#print(train_df.head())
#print(test_df.head())
#print(idlookup.head())
print(test_df.columns)


# Split data into test data and training data
X = train_df['Image']
y = train_df.drop('Image', axis=1)


# Convert all image strings to arrays
num_images = len(X)
image_list = []

for i in range(num_images):
    img_str = X.iloc[i]
    img_array = np.fromstring(img_str, sep=' ')
    if img_array.shape[0] != 9216:
        print(f"Skipping row {i}, wrong image shape")
        continue
    image_list.append(img_array)

# Convert to NumPy array and reshape
X_np_all = np.array(image_list).reshape(-1, 96, 96, 1)
X_np_all = X_np_all / 255.0



X_train, X_valid, y_train, y_valid =  train_test_split(X_np_all, y, random_state=0)

print(X_train.shape)


# Visualize 5 training images with keypoints as crosses
plt.figure(figsize=(15, 3))
for i in range(5):
    image = X_train[i].reshape(96, 96)
    label = y_train.iloc[i]

    plt.subplot(1, 5, i + 1)
    plt.imshow(image, cmap='gray')
    for j in range(0, len(label), 2):
        x, y = label.iloc[j], label.iloc[j + 1]
        if not np.isnan(x) and not np.isnan(y):
            plt.plot(x, y, 'rx')  # red cross
    plt.axis('off')
plt.suptitle("Training images with true keypoints", fontsize=14)
plt.tight_layout()
plt.show()


# Data Pipeline

# Normalise pixel values between [0, 1]
X_train = X_train / 255.0
X_valid = X_valid / 255.0

# Convert from pandas dataframe to tensor flow dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train.reshape(-1, 96, 96, 1), y_train.values))
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid.reshape(-1, 96, 96, 1), y_valid.values))

# Apply batching, shuffling, caching
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = (
    train_ds
    .shuffle(buffer_size=1024)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

valid_ds = (
    valid_ds
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)



# Improvement:
# apply augmentation to improve dataset variability
# when using flips, need to make sure that the label also flips, so center of left eye becomes center of right eye

#augment = keras.Sequential([
    #layers.RandomContrast(factor=0.5),
    #layers.RandomFlip('horizontal'),
    #layers.Resizing(28, 28)
#])

# apply augmentation to training dataset only:
# train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

# ex = next(iter(train_ds.unbatch().map(lambda x, y: x).batch(1)))


# visualise the augmented dataset
# plt.figure(figsize=(10,10))
# for i in range(16):
    # image = augment(ex, training=True)
    # plt.subplot(4, 4, i+1)
    # plt.imshow(tf.squeeze(image))
    # plt.axis('off')
# plt.show()


# Many of the rows include missing data in at least 1 column.
# So better to train individual models for each feature and only exclude rows that do not exist for that feature
mod_1_features = ['left_eye_center_x', 'left_eye_center_y','right_eye_center_x', 'right_eye_center_y',]
mod_2_features = ['nose_tip_x','nose_tip_y']
mod_3_features = ['mouth_left_corner_x', 'mouth_left_corner_y','mouth_right_corner_x', 'mouth_right_corner_y','mouth_center_top_lip_x', 'mouth_center_top_lip_y',]
mod_4_features = ['mouth_center_bottom_lip_x','mouth_center_bottom_lip_y',]
mod_5_features = ['left_eye_inner_corner_x', 'left_eye_inner_corner_y','right_eye_inner_corner_x', 'right_eye_inner_corner_y','left_eye_outer_corner_x', 'left_eye_outer_corner_y','right_eye_outer_corner_x', 'right_eye_outer_corner_y',]
mod_6_features = ['left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y','right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y','left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y','right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',]


# model 1
# drop rows where any feature in this model is missing values
valid_rows_1 = train_df[mod_1_features + ['Image']].dropna()

# Extract images and labels for this model
X_mod_1 = np.stack(valid_rows_1['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values)
X_mod_1 = X_mod_1.reshape(-1, 96, 96, 1) / 255.0
y_mod_1 = valid_rows_1[mod_1_features].values

# model 2
# drop rows where any feature in this model is missing values
valid_rows_2 = train_df[mod_2_features + ['Image']].dropna()

# Extract images and labels for this model
X_mod_2 = np.stack(valid_rows_2['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values)
X_mod_2 = X_mod_2.reshape(-1, 96, 96, 1) / 255.0
y_mod_2 = valid_rows_2[mod_2_features].values


# model 3
# drop rows where any feature in this model is missing values
valid_rows_3 = train_df[mod_3_features + ['Image']].dropna()

# Extract images and labels for this model
X_mod_3 = np.stack(valid_rows_3['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values)
X_mod_3 = X_mod_3.reshape(-1, 96, 96, 1) / 255.0
y_mod_3 = valid_rows_3[mod_3_features].values


# model 4
# drop rows where any feature in this model is missing values
valid_rows_4 = train_df[mod_4_features + ['Image']].dropna()

# Extract images and labels for this model
X_mod_4 = np.stack(valid_rows_4['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values)
X_mod_4 = X_mod_4.reshape(-1, 96, 96, 1) / 255.0
y_mod_4 = valid_rows_4[mod_4_features].values


# model 5
# drop rows where any feature in this model is missing values
valid_rows_5 = train_df[mod_5_features + ['Image']].dropna()

# Extract images and labels for this model
X_mod_5 = np.stack(valid_rows_5['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values)
X_mod_5 = X_mod_5.reshape(-1, 96, 96, 1) / 255.0
y_mod_5 = valid_rows_5[mod_5_features].values


# model 6
# drop rows where any feature in this model is missing values
valid_rows_6 = train_df[mod_6_features + ['Image']].dropna()

# Extract images and labels for this model
X_mod_6 = np.stack(valid_rows_6['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values)
X_mod_6 = X_mod_6.reshape(-1, 96, 96, 1) / 255.0
y_mod_6 = valid_rows_6[mod_6_features].values


# Build the model - needs to output a different number of outputs for each model number 
def build_model(num_outputs):
    model = keras.Sequential([
        layers.InputLayer(shape=[96, 96, 1]),
    
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
    
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
    
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
    
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(num_outputs, activation='linear'),
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model



def plot_history(history):
    df = pd.DataFrame(history.history)
    df.loc[:, ['loss']].plot(title="Loss")
    df.loc[:, ['mae']].plot(title="MAE")


# Train the models
early_stop_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # for regression, monitor validation loss
    patience=8,              # stop after 8 epochs without improvement
    restore_best_weights=True
)

# model 1
model_1 = build_model(num_outputs=4)

history_1 = model_1.fit(
    X_mod_1, y_mod_1,
    # validation_split=0.2,
    epochs=50,
    #callbacks=[early_stop_cb],
    batch_size=64
)

# model 2
model_2 = build_model(num_outputs=2)

history_2 = model_2.fit(
    X_mod_2, y_mod_2,
    # validation_split=0.2,
    epochs=50,
    #callbacks=[early_stop_cb],
    batch_size=64
)


# model 3
model_3 = build_model(num_outputs=6)

history_3 = model_3.fit(
    X_mod_3, y_mod_3,
    # validation_split=0.2,
    epochs=50,
    #callbacks=[early_stop_cb],
    batch_size=64
)


# model 4
model_4 = build_model(num_outputs=2)

history_4 = model_4.fit(
    X_mod_4, y_mod_4,
    #validation_split=0.2,
    epochs=50,
    #callbacks=[early_stop_cb],
    batch_size=64
)


# model 5
model_5 = build_model(num_outputs=8)

history_5 = model_5.fit(
    X_mod_5, y_mod_5,
    # validation_split=0.2,
    epochs=50,
    #callbacks=[early_stop_cb],
    batch_size=64
)


# model 6
model_6 = build_model(num_outputs=8)

history_6 = model_6.fit(
    X_mod_6, y_mod_6,
    # validation_split=0.2,
    epochs=50,
    #callbacks=[early_stop_cb],
    batch_size=64
)

plot_history(history_1)
plot_history(history_2)
plot_history(history_3)
plot_history(history_4)
plot_history(history_5)
plot_history(history_6)


def preprocess_image(row):
    pixels = np.array(row.split(), dtype=np.float32).reshape(96, 96, 1)
    pixels = pixels / 255.0
    return pixels

processed_images = np.array([preprocess_image(img) for img in test_df['Image']])
print("Processed Image Shape:", processed_images.shape)



# Run predictions
pred_1 = model_1.predict(processed_images)
pred_2 = model_2.predict(processed_images)
pred_3 = model_3.predict(processed_images)
pred_4 = model_4.predict(processed_images)
pred_5 = model_5.predict(processed_images)
pred_6 = model_6.predict(processed_images)

# Combine predictions
predictions_dict = {
    'left_eye_center_x': pred_1[:, 0],
    'left_eye_center_y': pred_1[:, 1],
    'right_eye_center_x': pred_1[:, 2],
    'right_eye_center_y': pred_1[:, 3],
    
    'nose_tip_x': pred_2[:, 0],
    'nose_tip_y': pred_2[:, 1],

    'mouth_left_corner_x': pred_3[:, 0],
    'mouth_left_corner_y': pred_3[:, 1],
    'mouth_right_corner_x': pred_3[:, 2],
    'mouth_right_corner_y': pred_3[:, 3],
    'mouth_center_top_lip_x': pred_3[:, 4],
    'mouth_center_top_lip_y': pred_3[:, 5],

    'mouth_center_bottom_lip_x': pred_4[:, 0],
    'mouth_center_bottom_lip_y': pred_4[:, 1],

    'left_eye_inner_corner_x': pred_5[:, 0],
    'left_eye_inner_corner_y': pred_5[:, 1],
    'right_eye_inner_corner_x': pred_5[:, 2],
    'right_eye_inner_corner_y': pred_5[:, 3],
    'left_eye_outer_corner_x': pred_5[:, 4],
    'left_eye_outer_corner_y': pred_5[:, 5],
    'right_eye_outer_corner_x': pred_5[:, 6],
    'right_eye_outer_corner_y': pred_5[:, 7],

    'left_eyebrow_inner_end_x': pred_6[:, 0],
    'left_eyebrow_inner_end_y': pred_6[:, 1],
    'right_eyebrow_inner_end_x': pred_6[:, 2],
    'right_eyebrow_inner_end_y': pred_6[:, 3],
    'left_eyebrow_outer_end_x': pred_6[:, 4],
    'left_eyebrow_outer_end_y': pred_6[:, 5],
    'right_eyebrow_outer_end_x': pred_6[:, 6],
    'right_eyebrow_outer_end_y': pred_6[:, 7],
}


# Create submission
# idlookup table has columns: RowId, ImageId, FeatureName
row_ids = idlookup['RowId']
image_ids = idlookup['ImageId'] - 1  # Adjust to zero-based indexing
features = idlookup['FeatureName']

# Build the 'Location' list
locations = []
for img_id, feature in zip(image_ids, features):
    if feature in predictions_dict:
        locations.append(predictions_dict[feature][img_id])
    else:
        locations.append(np.nan)  # in case something is missing

# Create submission DataFrame
submission = pd.DataFrame({
    'RowId': row_ids,
    'Location': locations
})

# Save submission
submission.to_csv('facial_keypoints_detection_submission.csv', index=False)
print("Submission file saved successfully!")



# Show 5 test images with predicted keypoints
plt.figure(figsize=(15, 3))
for i in range(5):
    image = processed_images[i].reshape(96, 96)

    plt.subplot(1, 5, i + 1)
    plt.imshow(image, cmap='gray')

    # For each feature, plot predicted x and y
    for feature_name in predictions_dict.keys():
        if predictions_dict[feature_name] is not None:
            coord = predictions_dict[feature_name][i]
            if 'x' in feature_name:
                x = coord
                y_name = feature_name.replace('_x', '_y')
                if y_name in predictions_dict:
                    y = predictions_dict[y_name][i]
                    if not np.isnan(x) and not np.isnan(y):
                        plt.plot(x, y, 'rx')  # red cross
    plt.axis('off')
plt.suptitle("Test images with predicted keypoints", fontsize=14)
plt.tight_layout()
plt.show()

