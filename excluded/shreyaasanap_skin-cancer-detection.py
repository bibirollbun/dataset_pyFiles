import os
os.environ["KERAS_BACKEND"] = "tensorflow" # other options: tensorflow or torch

import keras_cv
import keras
from keras import ops
import tensorflow as tf
import cv2

import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 


tf.__version__
keras_cv.__version__


class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    neg_sample = 0.01 # Downsample negative calss
    pos_sample = 5.0  # Upsample positive class
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [128, 128]  # Input image size
    epochs = 8 # Training epochs
    batch_size = 256  # Batch size
    lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    class_names = ['target']
    num_classes = 1


keras.utils.set_random_seed(CFG.seed)


train=pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv")
train.ffill()
train.sample(5)




test=pd.read_csv("/kaggle/input/isic-2024-challenge/test-metadata.csv")
test.ffill()



train.target.value_counts()


negative_df=train[train["target"]== 0].sample(frac=CFG.neg_sample,random_state=CFG.seed)
positive_df=train[train["target"]== 1].sample(frac=CFG.pos_sample,random_state=CFG.seed,replace=True)
df=pd.concat([negative_df,positive_df])
df.target.value_counts()


import h5py

train_h5py=h5py.File("/kaggle/input/isic-2024-challenge/train-image.hdf5","r")
testing_hdf5 = h5py.File("/kaggle/input/isic-2024-challenge/test-image.hdf5","r")


isic_id = test.isic_id.iloc[1]

# Image as Byte String
byte_string = testing_hdf5[isic_id][()]
print(f"Byte String: {byte_string[:20]}....")

# Convert byte string to numpy array
nparr = np.frombuffer(byte_string, np.uint8)

print("Image:")
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # reverse last axis for bgr -> rgb
plt.imshow(image);


isic_id = df.isic_id.iloc[7]

# Image as Byte String
byte_string = train_h5py[isic_id][()]
print(f"Byte String: {byte_string[:20]}....")

# Convert byte string to numpy array
nparr = np.frombuffer(byte_string, np.uint8)

print("Image:")
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # reverse last axis for bgr -> rgb
plt.imshow(image);


def Image_nparray(id1):
    byte_string = train_h5py[id1][()]
    nparr = np.frombuffer(byte_string, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (128, 128))
    image_vector = image_resized.astype(np.float32) / 255.0
    return image_vector


images = []

for i in range(len(df)):
    isic_id = df.isic_id.iloc[i]  # Use the current index
    image_array = Image_nparray(isic_id)
    images.append(image_array)
df["image"] = images


# Categorical features which will be one hot encoded
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general",
            "tbp_tile_type","tbp_lv_location", ]
Primary_key=["isic_id","image","target"]

# Numeraical features which will be normalized
NUMERIC_COLUMNS = ["age_approx", "tbp_lv_nevi_confidence", "clin_size_long_diam_mm",
           "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
           "tbp_lv_deltaLBnorm", "tbp_lv_minorAxisMM", ]

FEAT_COLS = Primary_key + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
FEAT_COLS


df=df[FEAT_COLS]


df["target"].value_counts()


df["image"].iloc[0].shape


df.info()


df=df.dropna()


df=df.drop(columns=["isic_id"])


from sklearn.preprocessing import LabelEncoder
# Identify object columns
object_columns = df.select_dtypes(include=['object']).columns

# Create a LabelEncoder instance
labelencoder = LabelEncoder()

# Encode object columns
for column in object_columns:
    if column!= "image":
        df[column] = labelencoder.fit_transform(df[column])
df


# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess tabular data
tabular_columns = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'age_approx', 
                   'tbp_lv_nevi_confidence', 'clin_size_long_diam_mm', 'tbp_lv_areaMM2', 
                   'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaLBnorm', 
                   'tbp_lv_minorAxisMM']

X_train_tabular = X_train[tabular_columns]
X_test_tabular = X_test[tabular_columns]

# Standardize the tabular data
scaler = StandardScaler()
X_train_tabular = scaler.fit_transform(X_train_tabular)
X_test_tabular = scaler.transform(X_test_tabular)

# # Preprocess image data
# 
X_train_images =np.stack(X_train['image'].apply(lambda x: np.array(x)/255))
# 
X_test_images = np.stack(X_test['image'].apply(lambda x: np.array(x)/255 ))

# # Resize images to 128x128x3
X_train_images = tf.image.resize(X_train_images, [128, 128])
X_test_images = tf.image.resize(X_test_images, [128, 128])


df["image"]


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform


# Input layers
image_input = Input(shape=(128, 128, 3), name='image_input')
tabular_input = Input(shape=(X_train_tabular.shape[1],), name='tabular_input')

# CNN for image data
effnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
effnet.trainable = True  # Unfreeze the model

# Freeze only first 100 layers (keep some layers frozen)
for layer in effnet.layers[:100]:
    layer.trainable = False
x = effnet(image_input)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# ANN for tabular data
y = Dense(256, activation='relu', kernel_initializer=GlorotNormal())(tabular_input)
y = Dropout(0.5)(y)
y = Dense(128, activation='relu', kernel_initializer=GlorotNormal())(y)
y = Dropout(0.3)(y)
y = Dense(64, activation='relu', kernel_initializer=GlorotNormal())(y)


# Concatenate both models
combined = Concatenate()([x, y])

# Final layers
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
output = Dense(1, activation='sigmoid')(z)

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
# Create the model
model = Model(inputs=[image_input, tabular_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()


# Train the model
history = model.fit(
    [X_train_images, X_train_tabular], y_train,
    validation_data=([X_test_images, X_test_tabular], y_test),
    epochs=30,
    batch_size=16,
    callbacks=[callback]
)


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0

# Assuming X_train_tabular is already defined
# Input layers
image_input = Input(shape=(128, 128, 3), name='image_input')
tabular_input = Input(shape=(X_train_tabular.shape[1],), name='tabular_input')

# CNN for image data
effnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
effnet.trainable = True  # Unfreeze the model

# Freeze only first 100 layers (keep some layers frozen)
for layer in effnet.layers[:100]:
    layer.trainable = False
x = effnet(image_input)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# ANN for tabular data
y = Dense(256, activation='relu', kernel_initializer=GlorotNormal())(tabular_input)
y = Dropout(0.5)(y)
y = Dense(128, activation='relu', kernel_initializer=GlorotNormal())(y)
y = Dropout(0.3)(y)
y = Dense(64, activation='relu', kernel_initializer=GlorotNormal())(y)


# Concatenate both models
combined = Concatenate()([x, y])

# Final layers
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
output = Dense(1, activation='sigmoid')(z)

# Create the model
model = Model(inputs=[image_input, tabular_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Build the model explicitly (VERY IMPORTANT)
model([tf.zeros((1, 128, 128, 3)), tf.zeros((1, X_train_tabular.shape[1]))])  # Pass dummy data

# Summary of the model
model.summary()

callback = EarlyStopping(monitor='loss', patience=5)

# Train the model
history = model.fit(
    [X_train_images, X_train_tabular], y_train,
    validation_data=([X_test_images, X_test_tabular], y_test),
    epochs=30,
    batch_size=16,
    callbacks=[callback]
)



# Evaluate the model
loss, accuracy = model.evaluate([X_test_images, X_test_tabular], y_test,batch_size=16)
print(f'Test Accuracy: {accuracy:.4f}')


# Save the model
model.save('skin_cancer_model.h5')


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming you have already loaded your model, X_test_images, X_test_tabular, and y_test

# 1. Evaluate the model to get loss and accuracy
loss, accuracy = model.evaluate([X_test_images, X_test_tabular], y_test, verbose=0,batch_size=16)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# 2. Get the model's predictions
y_pred_probs = model.predict([X_test_images, X_test_tabular])

# Convert probabilities to predicted labels (binary classification)
y_pred = (y_pred_probs > 0.5).astype(int)  # Threshold at 0.5

# Handle the case where y_test might be one-hot encoded (unlikely in binary, but checking)
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test

# 3. Calculate precision, recall, and F1 score (binary classification)
precision = precision_score(y_test_labels, y_pred)
recall = recall_score(y_test_labels, y_pred)
f1score = f1_score(y_test_labels, y_pred)

# 4. Print the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1score:.4f}')





