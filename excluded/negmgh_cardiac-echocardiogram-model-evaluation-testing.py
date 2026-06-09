import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm    #  tqdm library provides a progress bar for loops in Python. It is especially useful when working with large datasets, training deep learning models, or processing large .npy files.
import tensorflow as tf


# Load the dataset
loaded_data_example = np.load("/kaggle/input/echo2022/train_data/train_data/2CH/patient076_2CH_sequence.npy")

# Print dataset shape
print("Loaded data shape:", loaded_data_example.shape)
loaded_data_example


train_data = pd.read_csv("../input/echo2022/train_data.csv")
train_data


sample_sub = pd.read_csv("../input/echo2022/sample_submission.csv")


train_2CH_dir = "../input/echo2022/train_data/train_data/2CH/"
train_4CH_dir = "../input/echo2022/train_data/train_data/4CH/"
test_2CH_dir = "../input/echo2022/test_data/test_data/2CH/"
test_4CH_dir = "../input/echo2022/test_data/test_data/4CH/"


# Import metadata about the echo images
sequence_number = []
train_img_w = []
train_img_h = []
for i in tqdm(os.listdir(train_2CH_dir)):
    if i.endswith(".npy"):
        number, width, height = np.load(train_2CH_dir + i).shape
        sequence_number.append(number)
        train_img_w.append(width)
        train_img_h.append(height)


# Because of large filesize we may need to import the image data into Python in batches. 
# This can be done using batch generators.

def batch_generator(batch_size, gen_x): 
    batch_features = np.zeros((batch_size,10, 256, 256))
    batch_labels = np.zeros((batch_size,1)) 
    while True:
        for i in range(batch_size):
            batch_features[i] , batch_labels[i] = next(gen_x)
        yield np.expand_dims(batch_features,4), batch_labels

def generate_data(filelist, img_path, gt_df):
    while True:
        for i in filelist:
            if i.endswith(".npy"):
                img = np.load(img_path + i)
                img = img[:10]
                resized_img = np.zeros((10,256,256))
                for j,k in enumerate(img):
                    resized_img[j,:,:] = cv2.resize(k, (256,256), interpolation= cv2.INTER_LINEAR )
                y = float(gt_df.LV_ef[np.where(gt_df.Patient_number == i.split("_")[0])[0]])

                yield resized_img, y


test = batch_generator(5,generate_data(os.listdir(train_4CH_dir),train_4CH_dir,train_data))

print("The shape of one batch of images (batch size = 5):")
print(next(test)[0].shape)


import tensorflow as tf

model = tf.keras.models.Sequential()

# CNN feature extraction
model.add(
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'), 
        input_shape=(10, 256, 256, 1)
    )
)
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2,2))))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2,2))))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2,2))))

# Flatten before LSTM
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()))

# LSTM for temporal dependencies
model.add(tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))

# Fully connected output
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compile with AdamW optimizer
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)
model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Model summary
model.summary()



batch_size = 5
num_epoch = 5
steps = len(os.listdir(train_4CH_dir))//batch_size
history = model.fit(x=batch_generator(batch_size,generate_data(os.listdir(train_4CH_dir),train_4CH_dir,train_data)), epochs=num_epoch, 
                            steps_per_epoch=steps, verbose=0)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(10)
fig.set_figwidth(30)
ax1.plot(history.history["loss"])
ax1.set_title("Loss")
ax2.plot(history.history["root_mean_squared_error"])
ax2.set_title("RMSE")
plt.show()


y_pred=[]
for i in tqdm(sorted(os.listdir(test_4CH_dir))):
    if i.endswith(".npy"):
        img = np.load(test_4CH_dir + i)
        img = img[:10]
        resized_img = np.zeros((10,256,256))
        for j,k in enumerate(img):
            resized_img[j,:,:] = cv2.resize(k, (256,256), interpolation= cv2.INTER_LINEAR )
        y_pred.append(model.predict(np.expand_dims(np.expand_dims(resized_img,3),0)))


sample_sub.LV_ef = np.asarray(y_pred).ravel()
sample_sub


sample_sub.to_csv("submission.csv",index=False)

