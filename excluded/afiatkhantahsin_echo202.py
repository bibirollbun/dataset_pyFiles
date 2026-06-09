import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf


train_data = pd.read_csv("../input/echo2022/train_data.csv")


sample_sub = pd.read_csv("../input/echo2022/sample_submission.csv")


train_2CH_dir = "../input/echo2022/train_data/train_data/2CH/"
train_4CH_dir = "../input/echo2022/train_data/train_data/4CH/"
test_2CH_dir = "../input/echo2022/test_data/test_data/2CH/"
test_4CH_dir = "../input/echo2022/test_data/test_data/4CH/"


file_path = os.path.join(train_4CH_dir, "patient001_4CH_sequence.npy")

# Load the .npy file
data = np.load(file_path)

# Check the shape of the data to decide how to plot
print(f"Shape of the data: {data.shape}")

# Assuming the data is 2D or 3D (if 3D, the third dimension might represent channels or time points)
# Plot a single slice or the first 2D array if it's 3D
if data.ndim == 2:
    plt.imshow(data, cmap='gray')
    plt.title('2D Sequence')
elif data.ndim == 3:
    # If it's 3D, we can plot the first channel or time step
    plt.imshow(data[0, :, :], cmap='gray')  # Adjust indexing if needed
    plt.title('First Channel of 3D Sequence')
else:
    print("Data has an unsupported number of dimensions for plotting.")
plt.show()






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


plt.hist(np.asarray(sequence_number), bins=10)
plt.title("Distribution of the number of image across all patients in the training set")
plt.xlabel("Number of images in the echo loop")
plt.ylabel("Number of patients")
plt.show()


plt.hist(np.asarray(train_img_w), bins=10)
plt.title("Distribution of image widths across all patients in the training set")
plt.xlabel("Image widths")
plt.ylabel("Number of patients")
plt.show()


plt.hist(np.asarray(train_img_h), bins=10)
plt.title("Distribution of image heights across all patients in the training set")
plt.xlabel("Image heights")
plt.ylabel("Number of patients")
plt.show()


plt.hist(np.asarray(train_data.LV_ef), bins=10)
plt.title("Distribution of ejection fraction across all patients in the training set")
plt.xlabel("Ejection fraction (%)")
plt.ylabel("Number of patients")
plt.xlim(0,100)
plt.show()


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


model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'), 
        input_shape=(10, 256, 256, 1) # 5 images...
    )
)
model.add(
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.GlobalAveragePooling2D() # Or Flatten()
    )
)
model.add(
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=False)
)
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile('adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])


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
sample_sub.head()


sample_sub.to_csv("submission.csv",index=False)

