# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np

# Example for loading seismic data and corresponding velocity map
seismic_data = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy')
velocity_map = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model1.npy')

# You can load other samples similarly



import os

data_dir = '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/'
model_dir = '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/'

# List the files in the data and model directories
data_files = os.listdir(data_dir)
model_files = os.listdir(model_dir)

print(f"Available data files: {data_files}")
print(f"Available model files: {model_files}")



import numpy as np
import os

X = []  # Store seismic data
y = []  # Store velocity maps (labels)

# Directory paths
data_dir = '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/'
model_dir = '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/'

# Loop through available files and load them
data_files = sorted(os.listdir(data_dir))
model_files = sorted(os.listdir(model_dir))

# Ensure the files match and we have a corresponding data and model pair
for data_file, model_file in zip(data_files, model_files):
    if data_file.endswith('.npy') and model_file.endswith('.npy'):
        seismic_data = np.load(os.path.join(data_dir, data_file))
        velocity_map = np.load(os.path.join(model_dir, model_file))
        
        X.append(seismic_data)
        y.append(velocity_map)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Seismic Data Shape: {X.shape}")
print(f"Velocity Maps Shape: {y.shape}")

# Proceed to split the dataset
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")



import numpy as np
import matplotlib.pyplot as plt

data = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy')
model = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model1.npy')

print("Data shape:", data.shape)    # e.g., (500, 32, 1000, 100)
print("Model shape:", model.shape)  # e.g., (500, 100, 70)

#plt.imshow(model[0], cmap='jet')
#plt.imshow(model[0].squeeze(), cmap='jet')
plt.imshow(model[0][0], cmap='jet')

plt.colorbar(label='Velocity (m/s)')
plt.title("Sample velocity map")
plt.show()



train_data = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy')
print(train_data.shape)  # Check the shape of the original data


train_data = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy')
print(train_data.shape)  # Output should be (500, 5, 1000, 70)

# Reshape it to (500, 1000, 70, 5)
train_data_reshaped = train_data.transpose(0, 2, 3, 1)  # (500, 1000, 70, 5)
print(train_data_reshaped.shape)


train_data = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy')
print(train_data.shape)  # Output should be (500, 5, 1000, 70)

# Flatten the sources dimension
train_data_reshaped = train_data.reshape(500, 1000, 70 * 5)  # Flatten sources (5) into the last dimension
print(train_data_reshaped.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build the CNN model
model = Sequential([
    # Convolutional layer with 32 filters, kernel size (3, 3), and input shape (1000, 70, 5)
    Conv2D(32, (3, 3), activation='relu', input_shape=(1000, 70, 5)),
    MaxPooling2D((2, 2)),
    
    # Add another convolutional layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten the output from the convolutional layers
    Flatten(),
    
    # Dense layer for classification or regression
    Dense(128, activation='relu'),
    Dense(1)  # Or change the number of units based on your output requirements
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Change the loss function depending on the task

# Summary of the model
model.summary()



# Build the CNN model for the flattened source-receiver case
model = Sequential([
    # Convolutional layer with 32 filters, kernel size (3, 3), and input shape (1000, 350)
    Conv2D(32, (3, 3), activation='relu', input_shape=(1000, 350, 1)),
    MaxPooling2D((2, 2)),
    
    # Add another convolutional layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten the output from the convolutional layers
    Flatten(),
    
    # Dense layer for classification or regression
    Dense(128, activation='relu'),
    Dense(1)  # Or change the number of units based on your output requirements
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Change the loss function depending on the task

# Summary of the model
model.summary()



import numpy as np

# Load the seismic data (X_train)
X_train = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy')

# Load the corresponding velocity map data (y_train)
y_train = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model1.npy')

# Check shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Optionally, load and check validation set (X_val, y_val)
X_val = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data2.npy')
y_val = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model2.npy')

print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")




model.layers  # list of layers
model.summary()  # prints model architecture


# First, check the shape
print(X_train.shape)  # Likely (500, 5, 1000, 70, 1, 1, 1, 1, 1, 1, 1)

# Remove extra singleton dimensions
X_train = np.squeeze(X_train)
X_val = np.squeeze(X_val)

# Then expand ONLY the last channel dimension
X_train = np.expand_dims(X_train, axis=-1)  # Now shape: (500, 5, 1000, 70, 1)
X_val = np.expand_dims(X_val, axis=-1)



y_train = np.squeeze(y_train)
y_val = np.squeeze(y_val)
y_train = np.expand_dims(y_train, axis=-1)  # (500, 70, 70, 1)
y_val = np.expand_dims(y_val, axis=-1)


# Remove all singleton dimensions
X_train = np.squeeze(X_train)
X_val = np.squeeze(X_val)

# Add back a single channel dimension at the end (for Conv3D)
X_train = np.expand_dims(X_train, axis=-1)  # shape: (500, 5, 1000, 70, 1)
X_val = np.expand_dims(X_val, axis=-1)

# Do the same for y
y_train = np.squeeze(y_train)
y_val = np.squeeze(y_val)

y_train = np.expand_dims(y_train, axis=-1)  # shape: (500, 70, 70, 1)
y_val = np.expand_dims(y_val, axis=-1)

# Print to verify
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)



from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=X_train.shape[1:], padding='same'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(np.prod(y_train.shape[1:]), activation='linear'),
    layers.Reshape(y_train.shape[1:])
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,
    verbose=1
)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training Curve")
plt.show()


import matplotlib.pyplot as plt

# Pick a few samples
num_samples = 5
indices = np.random.choice(len(X_val), num_samples, replace=False)

for idx in indices:
    pred = model.predict(X_val[idx:idx+1])[0, 0]
    true = y_val[idx, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(true, cmap='jet')
    axes[0].set_title("Ground Truth")

    axes[1].imshow(pred, cmap='jet')
    axes[1].set_title("Prediction")

    plt.suptitle(f"Sample {idx}")
    plt.show()



preds = model.predict(X_val)
np.save("predicted_velocity_maps.npy", preds)

