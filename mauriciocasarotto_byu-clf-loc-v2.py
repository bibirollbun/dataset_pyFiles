import cv2
from functools import partial
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import random
import re
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm


os.environ["PYTHONHASHSEED"] = "0"  # Asegura el hash fijo de strings
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def path2id_slice(file_path):
    """
    Extracts identifiers from a given file path of the format '.../tomo_XXXacc/slice_YYYY.jpg'.
    
    Args:
        file_path (tf.Tensor or str): Full path to the image file, e.g., '.../tomo_003acc/slice_0007.jpg'.
    
    Returns:
        tuple: A tuple (tomo, slice_num) where:
            - tomo (str): The name of the tomography folder, e.g., 'tomo_003acc'.
            - slice_num (str): The slice number without leading zeros, e.g., '7'.
    """
    # Split the file path into parts
    parts = tf.strings.split(file_path, '/')

    # Extract the tomography folder name (e.g., 'tomo_003acc')
    tomo = parts[-2].numpy().decode("utf-8")
    # Extract the file name (e.g., 'slice_0007.jpg')
    slice_file = parts[-1]
    
    # Remove the '.jpg' extension
    slice_name = tf.strings.regex_replace(slice_file, r"\.jpg$", "")
    
    # Extract the numeric part from the file name and remove leading zeros: 'slice_0007' → '7'
    slice_num = tf.strings.regex_replace(slice_name, r"slice_0*(\d+)", r"\1").numpy().decode("utf-8")
    
    # Return the tuple (e.g., ('tomo_003acc', '7'))
    combined = (tomo,slice_num)
    
    return combined


def expand_labels(dataset,labels):
        """
        Processes a dataset of image paths and expands the associated motor label information 
        into a structured DataFrame containing slice-level annotations.
    
        Args:
            dataset (list or iterable): Collection of file paths to image slices.
            labels (pd.DataFrame): DataFrame containing motor annotations with columns:
                - 'tomo_id'
                - 'Motor axis 0' (slice number)
                - 'Motor axis 1' (Y coordinate)
                - 'Motor axis 2' (X coordinate)
    
        Returns:
            pd.DataFrame: A new DataFrame with columns:
                - 'tomo_id': Tomography ID.
                - 'slice': Slice number.
                - 'Number of motors': Number of motors present in the slice.
                - 'x': X coordinate of the motor (0 if none).
                - 'y': Y coordinate of the motor (0 if none).
        """
    
        tomo_ids = []
        slices = []
        n_motors = []
        bbox = []
    
        for file in tqdm(dataset,desc='Processing'):
            # Extract tomography ID and slice number from path
            tomo,slc = path2id_slice(file)
            slc = float(slc)
            tomo_ids.append(tomo)
            slices.append(slc)
    
            # Filter labels for the given tomo and slice
            motors = labels[
                             (labels['tomo_id']==tomo) & 
                             (labels['Motor axis 0']==slc)
                             ]
            # Default coordinates (no motor)
            x,y = 0.,0.
            # If there is exactly one motor in this slice, get its coordinates
            if motors == 1:
                x = labels[(labels['tomo_id']==tomo) & 
                            (labels['Motor axis 0']==slc)
                            ]['Motor axis 2']
                y = labels[(labels['tomo_id']==tomo) & 
                            (labels['Motor axis 0']==slc)
                            ]['Motor axis 1']
    
            # Number of motors (rows) in this slice
            n_motors.append(motors)
            # Bounding box coordinates
            bbox.append((x,y))
    
        # Build expanded DataFrame
        labels_expanded =pd.DataFrame({
                'tomo_id':tomo_ids,
                'slice':slices,
                'Number of motors':n_motors,
                'x':[i[0] for i in bbox],
                'y':[i[1] for i in bbox]
            })
        
         # Optional: Add binary label
        # labels_expanded['has_motors'] = (labels_expanded['Number of motors'] > 0).astype(int)
    
        return labels_expanded


def load_image_from_path(path,channels=1):
        """
        Loads and preprocesses an image from a given file path.
    
        Args:
            path (str or tf.Tensor): Path to the image file.
            channels (int): Number of channels to decode (1 for grayscale, 3 for RGB). Default is 1.
    
        Returns:
            tf.Tensor: The preprocessed image tensor with shape [256, 256, channels] 
                       and dtype tf.float32, scaled to the [0, 1] range.
        """
        # Read the image file as a byte string
        image = tf.io.read_file(path)         
        # Decode the JPEG image with the specified number of channels
        image = tf.image.decode_jpeg(image, channels=channels)     
        # Convert image to float32 in the [0, 1] range
        image = tf.image.convert_image_dtype(image, tf.float32) 
        # Resize the image to 256x256
        image = tf.image.resize(image, [256, 256])
        return image


def train_test_split(dataset,ds_size, split=0.8):

    train_size = int(split * ds_size)
    valid_size = ds_size - train_size
    
    # Split the dataset into training and validation sets
    train_ds = dataset.take(train_size)
    valid_ds = dataset.skip(train_size)
    
    print(train_size)
    print(f'Train size: {train_size}')
    print(f'Validation size: {valid_size}')

    BATCH_SIZE = 32
    # Apply batching, repeating and prefetching on the training dataset
    train_ds = train_ds.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
     # Apply batching and prefetching on the validation dataset (no repeat)
    valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds


IMG_SIZE = 256

def rotate_coords(x, y, k):
    # k is an integer in [0, 3], representing the number of 90° counter-clockwise rotations.
    # Rotates the point (x, y) according to k:
    # 0: (x, y)
    # 1: (y, IMG_SIZE - x)
    # 2: (IMG_SIZE - x, IMG_SIZE - y)
    # 3: (IMG_SIZE - y, x)

    def rot0():
        return x, y
    def rot1():
        return y, IMG_SIZE - x
    def rot2():
        return IMG_SIZE - x, IMG_SIZE - y
    def rot3():
        return IMG_SIZE - y, x

    return tf.switch_case(k, branch_fns={0: rot0, 1: rot1, 2: rot2, 3: rot3})

def augment_sample(image, label):
    # label: [presence, x, y]
    presence = label[0]
    x = label[1]
    y = label[2]

    # Convert image to float32 for safe processing
    image = tf.image.convert_image_dtype(image, tf.float32)

    # 1. Random horizontal flip with 50% probability
    flip_lr = tf.random.uniform([]) > 0.5
    image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)

    # 2. Random vertical flip with 50% probability
    flip_ud = tf.random.uniform([]) > 0.5
    image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)

    # 3. Random rotation by multiple of 90 degrees
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    # --- Update coordinates only if presence > 0 ---
    def transform_coords():
        # Flip coordinates horizontally if flip_lr is True
        x_flipped = tf.cond(flip_lr, lambda: IMG_SIZE - x, lambda: x)
        # Flip coordinates vertically if flip_ud is True
        y_flipped = tf.cond(flip_ud, lambda: IMG_SIZE - y, lambda: y)
        # Rotate coordinates
        x_rot, y_rot = rotate_coords(x_flipped, y_flipped, k)
        return x_rot, y_rot

    x_new, y_new = tf.cond(presence > 0,
                           true_fn=transform_coords,
                           false_fn=lambda: (x, y))

    # Rebuild label tensor
    label_new = tf.stack([presence, x_new, y_new])

    return image, label_new


def build_U(inputs):
    """
    Builds a custom convolutional neural network with an encoder-like structure,
    hierarchical feature fusion, and two parallel outputs:
    - 'presence': binary classification (presence of an object).
    - 'coords': regression of 2D coordinates (e.g., bounding box center).

    Args:
        inputs (tf.Tensor): Input tensor (e.g., image batch).

    Returns:
        tf.keras.Model: A compiled Keras model with two outputs:
            - 'presence' (1 unit, sigmoid activation)
            - 'coords' (2 units, sigmoid activation)
    """
    

    relu = tf.keras.activations.relu

    # ----- Level 0 -----
    conv0_0 = tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=1,
                                 padding="same",kernel_initializer="he_normal")(inputs)
    x = relu(tf.keras.layers.BatchNormalization()(conv0_0))
    
    conv0_1 = tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv0_1))
    
    conv0_2 = tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv0_2))
    pooling0 = tf.keras.layers.MaxPooling2D(padding="same")(x)
    
    # ----- Level 1 -----
    conv1_0 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(pooling0)
    x = relu(tf.keras.layers.BatchNormalization()(conv1_0))
    
    conv1_1 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv1_1))
    
    conv1_2 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x) 
    x = relu(tf.keras.layers.BatchNormalization()(conv1_2))
    pooling1 = tf.keras.layers.MaxPooling2D(padding="same")(x)
    
    # ----- Level 2 -----
    conv2_0 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(pooling1)
    x = relu(tf.keras.layers.BatchNormalization()(conv2_0))
    
    conv2_1 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv2_1))
       
    conv2_2 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv2_2))
    pooling2 = tf.keras.layers.MaxPooling2D(padding="same")(x)
    
    # ----- Decoder & Feature Fusion -----

    # Upsample and concatenate with Level 1
    upsamp2 = tf.keras.layers.UpSampling2D(size=(2, 2), 
                                           interpolation="bilinear"
                                           )(pooling2)
    concat2_1 = tf.keras.layers.Concatenate()([pooling1,upsamp2])
    conv1_3 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(concat2_1)
    x = relu(tf.keras.layers.BatchNormalization()(conv1_3))
    
    conv1_4 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv1_4))
       
    conv1_5 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv1_5))
    pooling1_5 = tf.keras.layers.MaxPooling2D(padding="same")(x)

    # Upsample and concatenate with Level 0
    upsamp1 = tf.keras.layers.UpSampling2D(size=(4, 4), 
                                           interpolation="bilinear"
                                           )(pooling1_5)
    concat1_0 = tf.keras.layers.Concatenate()([pooling0,upsamp1])
    conv0_3 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(concat1_0)
    x = relu(tf.keras.layers.BatchNormalization()(conv0_3))
         
    conv0_4 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv0_4))
    
    conv0_5 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,
                                 padding="same",
                                kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(conv0_5))
    pooling0_5 = tf.keras.layers.MaxPooling2D(padding="same")(x)

    # ----- Dense Layers -----
    flatt = tf.keras.layers.Flatten()(pooling0_5)
    dense1 = tf.keras.layers.Dense(units=128,
        kernel_initializer="he_normal")(flatt)
    
    # Branch X - for 'presence' output
    x = relu(tf.keras.layers.BatchNormalization()(dense1))
    x = tf.keras.layers.Dropout(0.5, seed=42)(x)
    x = tf.keras.layers.Dense(units=20,
        kernel_initializer="he_normal")(x)
    x = relu(tf.keras.layers.BatchNormalization()(x))
    x = tf.keras.layers.Dropout(0.5, seed=42)(x)

    # Branch Y - for 'coords' output
    y = relu(tf.keras.layers.BatchNormalization()(dense1))
    y = tf.keras.layers.Dense(units=50,
        kernel_initializer="he_normal")(y)
    y = relu(tf.keras.layers.BatchNormalization()(y))
    
    # ----- Output Layers -----
    presence_output = tf.keras.layers.Dense(units=1, activation="sigmoid",name='presence')(x) 
    coords_output = tf.keras.layers.Dense(units=2,activation='sigmoid',name='coords')(y)
    
    return tf.keras.Model(inputs=inputs,outputs=[presence_output,coords_output])


# Set base paths for dataset directories
base_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
train_path = base_path + '/train'
test_path = base_path + '/test'

# Create TensorFlow datasets with image file paths for training and testing
training_paths_dataset = tf.data.Dataset.list_files(train_path +'/*/*.jpg',seed=42,shuffle=False)
test_paths_dataset = tf.data.Dataset.list_files(test_path +'/*/*.jpg',seed=42,shuffle=False)

# Load CSV file containing training labels
train_labels = pd.read_csv(base_path +"/"+"train_labels.csv")


# Filter out all samples with more than one motor
train_labels = train_labels[train_labels['Number of motors']<=1]

train_labels.describe()


# Count how many samples have 0 or 1 motor
train_labels['Number of motors'].value_counts()


# Filter labels where a motor is present (slice index != -1)
labels_one_motor = train_labels[train_labels['Motor axis 0']!=-1]

# Visualize the first 5 annotated motor positions on their corresponding images
for item in labels_one_motor.iloc[:5].iterrows():
    # Construct full image path based on tomo ID and slice number
    path = train_path + "/" + item[1]['tomo_id'] + f"/slice_{int(item[1]['Motor axis 0']):04d}.jpg"

    # Load the image using OpenCV
    imagen = cv2.imread(path)

    # Optional binarization: normalize and round pixel values
    imagen = np.array([np.round(x/225.0) for x in imagen])

    # Plot image with motor location highlighted
    plt.figure(figsize=(12,8))
    plt.imshow(imagen)
    plt.axis('off')
    # Overlay the motor position (x = axis 2, y = axis 1)
    plt.scatter(item[1]['Motor axis 2'],item[1]['Motor axis 1'],
                800,facecolors='none',edgecolors='r')
    plt.show()


train_labels.columns


# Initialize a list to store all training images
full_train_images_set = []

for label in train_labels.iterrows():
    tomo = label[1]['tomo_id']                                  # Get tomo ID
    slc = int(label[1]['Motor axis 0'])                         # Get slice index

    # If the slice index is -1 (no motor), select a random slice from the tomo folder
    if (slc == -1):
        slice_list = sorted(os.listdir(train_path + "/" + tomo))
        slc = int(np.random.rand()*len(slice_list))
    # Load the image from the constructed file path and append to the list
    full_train_images_set.append(load_image_from_path(f'{train_path}/{tomo}/slice_{slc:04d}.jpg'))


# Extract relevant columns from train_labels to create the full_train_labels dataframe
# These columns represent the number of motors and their coordinates (axis 2 and axis 1)
full_train_labels = train_labels[['Number of motors','Motor axis 2','Motor axis 1']]


train_labels.head()


# Normalize motor coordinates to the resized image scale (256x256)
for label in train_labels.iterrows():
    # Calculate scaling factors based on the original array shape for width and height
    width_factor = label[1]['Array shape (axis 2)'] /256.0
    height_factor = label[1]['Array shape (axis 1)'] /256.0

    # Adjust motor axis by dividing by the appropriate scaling factor
    full_train_labels.loc[label[0],'Motor axis 2'] = full_train_labels.loc[label[0],'Motor axis 2'] / width_factor 
    full_train_labels.loc[label[0],'Motor axis 1'] = full_train_labels.loc[label[0],'Motor axis 1'] / height_factor 


full_train_labels


# Combine the normalized labels with their corresponding images into a single DataFrame
full_training = full_train_labels
full_training.loc[:,'image'] = full_train_images_set


# Create a TensorFlow dataset of labels, where each element is a tuple:
# (Number of motors, Motor axis 2 coordinate, Motor axis 1 coordinate)
full_training_labels =  tf.data.Dataset.from_tensor_slices([(row['Number of motors'], row['Motor axis 2'], row['Motor axis 1'])
                                                            for _,row in full_train_labels.iterrows()])
# Create a TensorFlow dataset from the list of image tensors
full_training_dataset = tf.data.Dataset.from_tensor_slices(full_train_images_set)

# Zip the image dataset and the label dataset together into one dataset
# Each element is a tuple: (image, (Number of motors, x_coord, y_coord))
full_training_dataset = tf.data.Dataset.zip(full_training_dataset,full_training_labels)


training_ds, validation_ds = train_test_split(full_training_dataset,len(full_training_dataset))


# Create 8 augmented datasets with different map calls
augmented_datasets = [full_training_dataset.map(augment_sample) for _ in range(8)]

# Concatenate all augmented datasets into one big dataset
augmented_dataset = augmented_datasets[0]
for ds in augmented_datasets[1:]:
    augmented_dataset = augmented_dataset.concatenate(ds)


X_train_full = []
y_clf_full = []
y_loc_full = []
            #full_training_dataset
for x, y in augmented_dataset.as_numpy_iterator():
    # Append each image tensor to X_train_full list
    X_train_full.append(x)
    # Append classification label (presence) to y_clf_full list
    y_clf_full.append(y[0])
    # Append localization labels (x, y coordinates) as a tuple to y_loc_full list
    y_loc_full.append((y[1],y[2]))

# Combine the three lists into one list of tuples to maintain correspondence
combine = list(zip(X_train_full, y_clf_full, y_loc_full))

# Shuffle the combined list to randomize the order while keeping data-label pairs
random.shuffle(combine)

# Unzip the combined list back into separate lists
X_train_full, y_clf_full, y_loc_full = zip(*combine)

# Convert tuples back to lists if needed
X_train_full = list(X_train_full)
y_clf_full = list(y_clf_full)
y_loc_full = list(y_loc_full)


len(X_train_full)*.8


# Split the full training data into training and validation sets at 80%
X_train = X_train_full[:3833]            # Training images
X_validation = X_train_full[3833:]       # Validation images

# Split the classification labels into training and validation sets
y_clf_train = y_clf_full[:3833]          # Training classification labels
y_clf_validation = y_clf_full[3833:]     # Validation classification labels

# Split the localization labels into training and validation sets
y_loc_train = y_loc_full[:3833]          # Training localization labels (coordinates)
y_loc_validation = y_loc_full[3833:]     # Validation localization labels (coordinates)


# Convert training images and labels to NumPy arrays
X_train = np.array(X_train)
y_clf_train = np.array(y_clf_train)
y_loc_train = np.array(y_loc_train)

# Convert validation images and labels to NumPy arrays
X_validation = np.array(X_validation)
y_clf_validation = np.array(y_clf_validation)
y_loc_validation = np.array(y_loc_validation)


# Calculate mean of the training images
mean = np.mean(X_train)
# Calculate standard deviation of the training images
stddev = np.std(X_train)
# Standardize training images using the training mean and std
X_train_std = (X_train - mean) / stddev
# Standardize validation images using the same training mean and std
X_validation_std = (X_validation - mean) / stddev


# Normalize location labels by dividing by image size (256)
y_loc_train_std = y_loc_train /256.0
y_loc_validation_std = y_loc_validation /256.0
# Check if the normalized labels have the same type and shape as the original
type(y_loc_train_std) == type(y_loc_train), y_loc_train_std.shape == y_loc_train.shape


# List available physical GPU devices
gpus = tf.config.list_physical_devices('GPU')
# Print how many GPUs are available
print(f'Available GPUs: {len(gpus)}')
# Create a MirroredStrategy for distributed training across GPUs
strategy = tf.distribute.MirroredStrategy()
# Print the number of devices used in the strategy (replicas in sync)
print(f"Number of devices: {strategy.num_replicas_in_sync}")


# Replace coordinates with [-1, -1] if any coordinate is negative in y_loc_train_std
y_loc_train_std = np.array([
    np.array([-1,-1]) if x[0]<0 or x[1]<0 else x
    for x in y_loc_train_std])
# Replace coordinates with [-1, -1] if any coordinate is negative in y_loc_validation_std
y_loc_validation_std = np.array([
    np.array([-1,-1]) if x[0]<0 or x[1]<0 else x
    for x in y_loc_validation_std])


y_loc_train_std


# Convert lists to tf.data.Dataset
def create_dataset(X, y_clf, y_loc, batch_size=32, shuffle=True):
     # Combine labels into a dictionary for multi-output
    y = {'presence': y_clf, 'coords': y_loc}

    # Create a dataset from the features and labels
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    # Batch the dataset and prefetch for performance optimization
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

batch_size = 32
# Create training dataset with shuffling enabled
train_dataset = create_dataset(X_train, y_clf_train, y_loc_train_std, batch_size=batch_size, shuffle=True)
# Create validation dataset without shuffling
val_dataset = create_dataset(X_validation, y_clf_validation, y_loc_validation_std, batch_size=batch_size, shuffle=False)



for batch in train_dataset.take(1):
    images, targets = batch
    print("Images shape:", images.shape)  # Print the shape of the image batch

    print("\nKeys in targets:", targets.keys())  # Print the keys in the targets dictionary
    print("Presence shape:", targets['presence'].shape)  # Shape of the presence labels batch
    print("Coords shape:", targets['coords'].shape)  # Shape of the coordinates labels batch
    print("\nExample presence label:", targets['presence'][0].numpy())  # Print an example presence label
    print("Example coords label:", targets['coords'][0].numpy())  # Print an example coordinates label
    break  # Only process one batch for this inspection


early_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                           restore_best_weights=True) # Stop training early if no improvement for 10 epochs, restore best model weights
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'clf_model.keras',
    monitor="val_loss",
    verbose=0,
    save_best_only=True,         # Save only the best model based on validation loss
    save_weights_only=False,
    mode="min",
    save_freq="epoch"
) # Checkpoint callback to save model after each epoch if improved

with strategy.scope(): # Use MirroredStrategy for distributed training on GPUs

    @tf.keras.utils.register_keras_serializable()
    class ConditionalMAEMetric(tf.keras.metrics.Metric):
        def __init__(self, name="conditional_mae", **kwargs):
            super().__init__(name=name, **kwargs)
            self.total_error = self.add_weight(name="total_error", initializer="zeros") # Accumulated total error
            self.count = self.add_weight(name="count", initializer="zeros")  # Count of valid samples
    
        def update_state(self, y_true, y_pred, sample_weight=None):
            # Mask: 1 if both coordinates >= 0, else 0 to ignore invalid samples
            mask = tf.cast(tf.reduce_all(y_true >= 0, axis=1), tf.float32)  # shape: (batch_size,)
            y_true_cast = tf.cast(y_true,tf.float32)
            y_pred_cast = tf.cast(y_pred,tf.float32)
            
            # Absolute error summed over x and y for each sample
            error_per_sample = tf.reduce_sum(tf.abs(y_true_cast - y_pred_cast), axis=1)  # shape: (batch_size,)
            
            # Apply mask to ignore invalid samples
            masked_error = error_per_sample * mask  # shape: (batch_size,)
            
            # Accumulate total error and count valid samples
            self.total_error.assign_add(tf.reduce_sum(masked_error))
            self.count.assign_add(tf.reduce_sum(mask))
    
        def result(self):
            # Return mean absolute error over valid samples, avoiding division by zero
            return tf.math.divide_no_nan(self.total_error, self.count)
    
        def reset_states(self):
            # Reset metric state at the start of each epoch
            self.total_error.assign(0.0)
            self.count.assign(0.0)

    @tf.keras.utils.register_keras_serializable()
    def conditional_mae_loss(y_true, y_pred):
        """
        Custom loss that computes MAE ignoring samples where both true coords are negative.
        y_true and y_pred have shape (batch_size, 2).
        """
        # Create mask: 1 if both coordinates are >= 0 (object present), else 0
        mask = K.cast(K.all(y_true >= 0.0, axis=1, keepdims=True), dtype='float32')  # shape (batch_size, 1)
    
        # Calculate absolute error and apply mask
        absolute_error = K.abs(y_true - y_pred)
        masked_error = mask * absolute_error
    
    
        # Return mean error only over valid samples, avoiding division by zero
        return K.sum(masked_error) / (K.sum(mask) + K.epsilon())

    # Define learning rate schedule with exponential decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1200,
        decay_rate=0.96,
        staircase=True)

    # Use SGD optimizer with momentum and the learning rate schedule
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,momentum=0.85)

     # Input layer for grayscale images 256x256x1
    input_layer = tf.keras.Input(shape=(256, 256,1))  
    # Build model with the U-Net function
    model = build_U(input_layer)
    # Compile model with binary crossentropy for classification and conditional MAE loss for coords
    model.compile(optimizer=optimizer,
                  loss={'presence': 'binary_crossentropy', 'coords': conditional_mae_loss},
                  #loss_weights={'presence': 1.0, 'coords': 1.0},
                  metrics={'presence': 'accuracy','coords': ConditionalMAEMetric()})

    # Calculate steps per epoch and validation steps for training
    steps_per_epoch = len(train_dataset) * 8 // 32
    validation_steps = int((1 - 0.8) * len(train_dataset)) // 32

    # Train the model with the datasets, steps, epochs, and checkpoint callback
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=100,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=[checkpoint_cb])

# Load the best saved model after training finishes
model = tf.keras.models.load_model('clf_model.keras', compile=True, safe_mode=True)


paths = [path.numpy().decode() for path in test_paths_dataset]

data = []
for path in paths:
    parts = path.split(os.sep)
    tomo_id = parts[-2]  # "tomo_003acc"
    slice_name = os.path.splitext(parts[-1])[0]  # "slice_0000"
    
    # Extract slice number as a float from the slice name using regex
    match = re.search(r'slice_(\d+)', slice_name)
    slice_id = float(match.group(1)) if match else -1  # Use -1 if no match found
    
    data.append({
        "tomo_id": tomo_id,
        "slice" : slice_id,
        "Motor axis 0": None,
        "Motor axis 1": None,
        "Motor axis 2": None
    })
test_df = pd.DataFrame(data)


test_df


test_images = []
for img in test_df.iterrows():
    tomo = img[1]['tomo_id']         # Get tomo folder ID from dataframe row
    slc = int(img[1]['slice'])       # Get slice number and convert to int
    # Load image from constructed path with zero-padded slice number (4 digits)
    test_images.append(load_image_from_path(f'{test_path}/{tomo}/slice_{slc:04d}.jpg'))


# Predict outputs for the test images by stacking them into a tensor and feeding into the model
y_preds = model.predict(tf.stack(test_images))


# Assign predictions to the corresponding columns in test_df:
# - 'Motor axis 0' gets the classification prediction (presence) from the first output y_preds[0]
# - 'Motor axis 1' gets the second coordinate (y) from the location predictions y_preds[1]
# - 'Motor axis 2' gets the first coordinate (x) from the location predictions y_preds[1]
test_df['Motor axis 0'] = [x[0] for x in y_preds[0]]
test_df['Motor axis 1'] = [x for x in y_preds[1][:,1]]
test_df['Motor axis 2'] = [x for x in y_preds[1][:,0]]


# For each 'tomo_id' group, find the index of the row with the maximum 'Motor axis 0' value (presence score)
idx = test_df.groupby('tomo_id')['Motor axis 0'].idxmax()
# Select only those rows with the highest presence per tomo_id and reset the index
test_df = test_df.loc[idx].reset_index(drop=True)


idx


test_df


# Round the 'Motor axis 0' values to nearest integer (0 or 1)
test_df['Motor axis 0'] = np.round(test_df['Motor axis 0'])

# For rows where 'Motor axis 0' is 0, set it to -1 (indicating absence)
test_df.loc[test_df['Motor axis 0'] == 0,'Motor axis 0'] = -1

# For rows where 'Motor axis 0' is 1, replace its value with the corresponding 'slice' number
test_df.loc[test_df['Motor axis 0'] == 1,'Motor axis 0'] = test_df.loc[test_df['Motor axis 0'] == 1,'slice']

# For rows where 'Motor axis 0' is -1 (absence), set 'Motor axis 1' and 'Motor axis 2' also to -1
test_df.loc[test_df['Motor axis 0'] == -1,'Motor axis 1'] = -1
test_df.loc[test_df['Motor axis 0'] == -1,'Motor axis 2'] = -1
# For rows where 'Motor axis 0' is positive (presence), replace its value again with the 'slice' number 
test_df.loc[test_df['Motor axis 0'] >0,'Motor axis 0'] = test_df.loc[test_df['Motor axis 0'] >0,'slice']

# Remove the now unnecessary 'slice' column
test_df.drop('slice',axis=1,inplace=True)


test_df


for item in test_df.iterrows():
    # Check if 'Motor axis 0' value is positive (indicating presence)
    if item[1]['Motor axis 0'] > 0:
        # Build the path to the image file for the corresponding tomo and slice
        path = tf.constant(f"{test_path}/{item[1]['tomo_id']}/slice_{int(item[1]['Motor axis 0']):04d}.jpg")

        # Read the image file as a binary string
        image_data = tf.io.read_file(path)

        # Decode the JPEG image to a tensor with 1 channel (grayscale)
        image = tf.image.decode_jpeg(image_data, channels=1)
        
        height = tf.shape(image)[0].numpy()
        width = tf.shape(image)[1].numpy()
        
        # Scale 'Motor axis 1' by image height and update the DataFrame in column index 2 (Motor axis 1)
        test_df.iloc[item[0],2] = np.float32(item[1]['Motor axis 1']*height)

        # Scale 'Motor axis 1' by image width and update the DataFrame in column index 3 (Motor axis 2)
        test_df.iloc[item[0],3] = np.float32(item[1]['Motor axis 1']*width)
        
test_df


# Convert the columns 'Motor axis 0', 'Motor axis 1', and 'Motor axis 2' to integer type
test_df[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] = test_df[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].astype(int)

test_df


# Save the DataFrame as a CSV file named 'submission.csv' without including the index column
test_df.to_csv('submission.csv', index=False)




