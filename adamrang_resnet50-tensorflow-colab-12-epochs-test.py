import pandas as pd
import os

def csvToDataframe(csv_path: str) -> pd.DataFrame:
    """
    Return a panda dataframe from a CSV file path. Print errors if reading is unsuccessful.

    Parameters:
    csv path (str): File path of a CSV file.

    Returns:
    df - Panda Dataframe containing CSV file
    """

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        df = pd.DataFrame()  # To not break the following code

    return df

def displayCsv(csv_path: str) -> None:
    """
    Display the name and shape of the CSV file.
    Display the labels and first rows of CSV file.

    Parameters:
    csv path (str): File path of a CSV file.

    Returns:
    None - prints file name, shape, labels and sample rows.

    """

    df = csvToDataframe(csv_path)
    file_name = os.path.basename(csv_path)
    print(f"\n{file_name} has shape: {df.shape}")
    display(df.head())




train_csv = ("/kaggle/input/ai-vs-human-generated-dataset/train.csv")
test_csv = ("/kaggle/input/ai-vs-human-generated-dataset/test.csv")
df_train = csvToDataframe(train_csv)    # save panda dataframe
df_test = csvToDataframe(test_csv)    # save panda dataframe
displayCsv(train_csv)
displayCsv(test_csv)




# Create a dictionary of img path - label pairs
train_dict = dict(zip(df_train["file_name"], df_train["label"]))



img_paths = list(train_dict.keys())
img_labels = list(train_dict.values())

print(f"First 2 filepaths: \n {img_paths[:2]}")
print(f"First 2 labels: \n {img_labels[:2]}")


import random

# Constants
TRAIN_SPLIT = 0.9
VALID_SPLIT = 0.2
SEED = 42    # For reproducibility

# Create labeled list of images
labeled_img_files = [list(pair) for pair in zip(df_train["file_name"], df_train["label"])]
print(f"First 2 pairs: \n {labeled_img_files[:2]}")


# Build directories for organizing train-valid data
# 0 label - Human
# 1 label - AI

directories = [
    "/kaggle/working/ai-vs-human",
    "/kaggle/working/ai-vs-human/dataset",
    "/kaggle/working/ai-vs-human/dataset/train",
    "/kaggle/working/ai-vs-human/dataset/valid",
    "/kaggle/working/ai-vs-human/dataset/train/0",
    "/kaggle/working/ai-vs-human/dataset/train/1",
    "/kaggle/working/ai-vs-human/dataset/valid/0",
    "/kaggle/working/ai-vs-human/dataset/valid/1"
]

# Create directories from list
for directory in directories:
    os.makedirs(directory, exist_ok=True)


# 1st image - AI (1), 2nd image - Human (0), and so on
img_pair_paths = []    # Target structure of labels -> [[1,0], [1,0], [1,0]...]
for i in range(0, len(img_paths), 2):
  img_pair_paths.append([img_paths[i], img_paths[i+1]])



import shutil
import numpy as np


data_dir = "/kaggle/input/ai-vs-human-generated-dataset"
output_dir = "/kaggle/working/ai-vs-human/dataset"
train_ratio = TRAIN_SPLIT
np.random.seed(SEED)
random.seed(SEED)

# Create new shuffled paired list
shuffled_pair_paths = random.sample(img_pair_paths, len(img_pair_paths))

indeces_train = int(len(shuffled_pair_paths) * train_ratio)
train_pairs = shuffled_pair_paths[:indeces_train]
valid_pairs = shuffled_pair_paths[indeces_train:]



# Function to copy files
def copy_files(pairs, target_folder):
  count = 0
  for ai_img, human_img in pairs:
    shutil.copy(os.path.join(data_dir, ai_img), os.path.join(output_dir, target_folder, "1", os.path.basename(ai_img)))
    shutil.copy(os.path.join(data_dir,human_img), os.path.join(output_dir, target_folder, "0", os.path.basename(human_img)))

    # Monitor Progress
    count += 1
    if count % 1000 == 0:
      print(f"Copied {count}/{len(pairs)} files to {target_folder}")


# Copy valid files to respective folders
copy_files(valid_pairs, "valid")

print("Dataset successfully split and shuffled!")


# Copy train files to respective folde'rs
copy_files(train_pairs, "train")

print("Dataset successfully split and shuffled!")


dataset_dir = "/kaggle/working/ai-vs-human/dataset"
for dirpath, dirnames, filenames in os.walk(dataset_dir):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


import glob

def get_all_image_paths(dir_path):

    # Retrieve all the images paths.
    image_paths = glob.glob(f"{dir_path}/train/*/*", recursive=True)
    return image_paths


import matplotlib.pyplot as plt

def display_dataset_samples(image_paths):

    plt.figure(figsize=(18, 12))
    num_rows = 2
    num_cols = 2
    for i in range(num_rows*num_cols):
        plt.subplot(num_rows, num_cols, i+1)

        # Generate a random index.
        random_idx = random.choice(list(range(0, len(image_paths))))
        image = plt.imread(image_paths[random_idx])
        label = image_paths[random_idx].split('/')[-2]
        plt.imshow(image)
        plt.axis('off')
        plt.title(label)

    plt.show(block=True)



# Get all the image paths.
image_paths = get_all_image_paths('/kaggle/working/ai-vs-human/dataset')
print(f"There are {len(image_paths)} images in the dataset.")
print(image_paths[0])


# Display several random images from the dataset.
display_dataset_samples(image_paths)


import tensorflow as tf
# Set the global policy to mixed_float16 to improve training speed
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Create directory for saved models
os.makedirs("/kaggle/working/ai-vs-human/saved_models", exist_ok=True)


# Define constants
from dataclasses import dataclass
@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 2
    IMG_HEIGHT:  int = 640
    IMG_WIDTH:   int = 640
    CHANNELS:    int = 3
    BATCH_SIZE:  int = 16
    DATA_ROOT:   str = '/kaggle/working/ai-vs-human/dataset'

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE:     int   = 16
    EPOCHS:         int   = 2
    LEARNING_RATE:  float = 0.0001
    CHECKPOINT_DIR: str   = '/kaggle/working/ai-vs-human/saved_models.keras'




import tensorflow as tf

input_shape = (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH, DatasetConfig.CHANNELS)

print('Loading model with ImageNet weights...')
ResNet50_conv_base = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape,
                                                    include_top=False, # We will supply our own top.
                                                    weights='/kaggle/input/resnet50/keras/default/1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                                   )
# Set the `trainable` attribute of base to False to keep pre-trained weights.
ResNet50_conv_base.trainable = False

print('All weights trainable, fine tuning...')


print(ResNet50_conv_base.summary())


from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import Layer, Flatten, Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomCrop, RandomZoom, Rescaling
from tensorflow.keras.models import Sequential

class CustomDataAugmentation(Layer):
    """
    Custom data augmentation layer that combines multiple augmentation techniques.
    This layer applies random rotation and random zoom to input images.
    """
    
    def __init__(
        self, 
        rotation_factor=0.2, 
        zoom_height_factor=0.2, 
        zoom_width_factor=0.2,
        seed=None,
        fill_mode='reflect',
        interpolation='bilinear',
        fill_value=0.0,
        name="custom_augmentation",
        **kwargs
    ):
        """
        Initialize the data augmentation layer.
        
        Args:
            rotation_factor: Maximum rotation angle in radians or degrees.
                             If float, interpreted as radians; if int, interpreted as degrees.
            zoom_height_factor: Range for random height zoom. 
                                For a factor of 0.2, the zoomed area will be between 80% to 120% of original size.
            zoom_width_factor: Range for random width zoom.
                               If None, same as height_factor.
            seed: Random seed for reproducibility.
            fill_mode: Points outside boundaries are filled according to the given mode
                       (one of {'constant', 'reflect', 'wrap', 'nearest'}).
            interpolation: Interpolation method used to fill in new pixels.
                           One of {'nearest', 'bilinear'}.
            fill_value: Value used for points outside boundaries when fill_mode='constant'.
            name: Name of the layer.
        """
        super(CustomDataAugmentation, self).__init__(name=name, **kwargs)
        
        # Create the individual augmentation layers
        self.random_rotation = RandomRotation(
            rotation_factor,
            fill_mode=fill_mode,
            interpolation=interpolation,
            seed=seed,
            fill_value=fill_value
        )
        
        self.random_zoom = RandomZoom(
            height_factor=zoom_height_factor,
            width_factor=zoom_width_factor,
            fill_mode=fill_mode,
            interpolation=interpolation,
            seed=seed,
            fill_value=fill_value
        )
    
    def call(self, inputs, training=None):
        """
        Apply the augmentation to input images.
        
        Args:
            inputs: Input tensor, usually images.
            training: Boolean indicating whether the layer should behave in
                      training mode (applying augmentation) or inference mode (identity).
        
        Returns:
            Augmented images if training=True, original images otherwise.
        """
        if training is None:
            training = tf.keras.backend.learning_phase()
            
        if training:
            x = self.random_rotation(inputs)
            x = self.random_zoom(x)
            return x
        else:
            return inputs
    
    def compute_output_shape(self, input_shape):
        """The output shape is the same as the input shape."""
        return input_shape
    
    def get_config(self):
        """Return the configuration for serialization."""
        config = super(CustomDataAugmentation, self).get_config()
        # Add the configuration of the individual augmentation layers
        config.update({
            "rotation_factor": self.random_rotation.factor,
            "zoom_height_factor": self.random_zoom.height_factor,
            "zoom_width_factor": self.random_zoom.width_factor,
            "fill_mode": self.random_rotation.fill_mode,
            "interpolation": self.random_rotation.interpolation,
            "fill_value": self.random_rotation.fill_value,
            "seed": self.random_rotation.seed,
        })
        return config




# Custom preprocess Layer for easy serialization
class ResNetPreprocessingLayer(layers.Layer):
    """Custom layer that applies ResNet preprocessing to input images."""
    
    def __init__(self, name="resnet_preprocessing", **kwargs):
        super(ResNetPreprocessingLayer, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
        """Apply ResNet preprocessing to the input tensor."""
        return preprocess_input(inputs)
    
    def compute_output_shape(self, input_shape):
        """The output shape is the same as the input shape."""
        return input_shape
    
    def get_config(self):
        """Return the config dictionary for serialization."""
        config = super(ResNetPreprocessingLayer, self).get_config()
        return config


# Image input shape
INPUT_SHAPE = (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH, DatasetConfig.CHANNELS)

# CNN Model
model = Sequential([

  # Input Layer
  Input(shape=INPUT_SHAPE),

  # Data Augmentation
  CustomDataAugmentation(rotation_factor=0.2, zoom_height_factor=0.2),

  # ResNet Preprocessing (applied during training & inference)
  ResNetPreprocessingLayer(),

  ResNet50(weights='imagenet', include_top=False, pooling='avg'),

  Flatten(),

  
  Dense(100, activation='relu'),
  Dropout(0.5),

  Dense(50, activation='relu'),
  Dropout(0.5),

  # The final `Dense` layer with the number of classes.
  Dense(1, activation ='sigmoid')

])

print(model.summary())


from tensorflow.keras.utils import image_dataset_from_directory

input_shape = (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH)
batch_size = DatasetConfig.BATCH_SIZE
train_path = DatasetConfig.DATA_ROOT + '/train'
valid_path = DatasetConfig.DATA_ROOT + '/valid'

train_dataset = image_dataset_from_directory(directory=train_path,
                                             image_size=input_shape,
                                             batch_size=batch_size,
                                             seed=SEED,
                                             label_mode='binary',
                                            )

valid_dataset = image_dataset_from_directory(directory=valid_path,
                                             image_size=input_shape,
                                             batch_size=batch_size,
                                             seed=SEED,
                                             label_mode='binary'
                                            )





# Print the shape of the data and the aassociated labels.
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break


valid_dataset.class_names


import numpy as np
class_names = train_dataset.class_names
print(class_names)
plt.figure(figsize=(18, 10))

# Assumes dataset batch_size is at least 32.
num_rows = 2
num_cols = 2

# Here we use the take() method to retrieve just the first batch of data from the training portion of the dataset.
for data_batch, labels_batch in train_dataset.take(1):
    # For the batch of images and the associated (one-hot encoded) labels,
    # plot each of the images in the batch and the associated ground truth labels.
    for i in range(num_rows*num_cols):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(data_batch[i].numpy().astype("uint8"))
        truth_idx = int(labels_batch[i].numpy())
        plt.title(class_names[truth_idx])
        plt.axis("off")


# Compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Create a model checkpoint callback to save the "best" model based on highest validation_accuracy.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=TrainingConfig.CHECKPOINT_DIR,
                                                               save_weights_only=False,
                                                               monitor="val_accuracy",
                                                               mode="max",
                                                               save_best_only=True,
                                                              )
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=2,
    min_lr=1e-6)

# Train the model.
training_results = model.fit(train_dataset,
                             epochs=TrainingConfig.EPOCHS,
                             validation_data=valid_dataset,
                             callbacks= [model_checkpoint_callback,
                                         early_stopping, lr_reducer]
                            )


custom_objects = {
      "CustomDataAugmentation": CustomDataAugmentation,
      "ResNetPreprocessingLayer": ResNetPreprocessingLayer
    }
model.save("/kaggle/working/my_model_custom_class.h5")
model.save("/kaggle/working/my_model_custom_class.keras")


from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

def plot_results(metrics, ylabel=None, ylim=None, metric_name=None, color=None):

    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]

    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xlim([0, TrainingConfig.EPOCHS-1])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show(block=True)
    plt.close()


# Retrieve training results.
train_loss = training_results.history["loss"]
train_acc  = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc  = training_results.history["val_accuracy"]

plot_results([ train_acc, valid_acc ],
            ylabel="Accuracy",
            metric_name=["Training Accuracy", "Validation Accuracy"],
            color=["g", "b"])

plot_results([ train_loss, valid_loss ],
            ylabel="Loss",
            metric_name=["Training Loss", "Validation Loss"],
            color=["g", "b"]);


def evaluate_model(dataset, checkpoint_dir=None):

    if not checkpoint_dir:
        checkpoint_dir = os.path.join(os.getcwd(),TrainingConfig.CHECKPOINT_DIR)

    print(f"Checkpoint_dir: {checkpoint_dir}")

    # Load saved model.
    model = tf.keras.models.load_model(checkpoint_dir, custom_objects=custom_objects)

    num_matches = 0
    plt.figure(figsize=(17, 12))
    num_rows = 5
    num_cols = 3
    class_names = dataset.class_names

    # Retrieve a single batch.
    for data_batch, labels_batch in dataset.take(1):

        predictions = model.predict(data_batch)

        for idx in range(num_rows*num_cols):
            ax = plt.subplot(num_rows, num_cols, idx + 1)
            plt.axis("off")
            plt.imshow(data_batch[idx].numpy().astype("uint8"))

            if predictions[idx] > 0.5:
                pred_idx = 1
            else:
                pred_idx = 0

            truth_idx = int(labels_batch[idx].numpy())

            title = str("Truth: " + class_names[truth_idx]) + " vs Pred:  " + str(class_names[pred_idx])
            title_obj = plt.title(title, fontdict={'fontsize':11})

            if pred_idx == truth_idx:
                num_matches += 1
                plt.setp(title_obj, color='g')
            else:
                plt.setp(title_obj, color='r')

            acc = num_matches/(idx+1)
        print("Prediction accuracy: ", int(100*acc)/100)


evaluate_model(valid_dataset, TrainingConfig.CHECKPOINT_DIR)



val_dir = '/kaggle/working/ai-vs-human/dataset/valid'
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),  # Resize to match input size
    batch_size=16,
    label_mode='binary'
)


from sklearn.metrics import f1_score

# Predict on the validation data
y_true = []
y_pred = []

checkpoint_dir = os.path.join(os.getcwd(),TrainingConfig.CHECKPOINT_DIR)
model = tf.keras.models.load_model(checkpoint_dir, custom_objects=custom_objects)

for images, labels in val_ds:
    predictions = model.predict(images)  # Model's predictions
    y_true.extend(labels.numpy())        # True labels
    y_pred.extend((predictions >= 0.5).astype(int))  # Predicted labels (binary)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

dataset_path = '/kaggle/input/ai-vs-human-generated-dataset'
test_path = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2'
input_csv = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'
checkpoint_dir = TrainingConfig.CHECKPOINT_DIR

# Load your CSV file with image paths
df = pd.read_csv(input_csv, skiprows=1, header=None, names=['id'])

# Load the model
model = tf.keras.models.load_model(checkpoint_dir,custom_objects=custom_objects)

# Initialize an empty list to store the predictions
predictions = []

# Set up a batch size
batch_size = 16  # Adjust based on available memory

# Initialize a list to store images for batch prediction
batch_images = []
batch_img_paths = []

# Loop through each image path
for idx, img_path in enumerate(df['id']):
    img = image.load_img(os.path.join(dataset_path, img_path), target_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH))
    img_array = img_to_array(img)
    batch_images.append(img_array)
    batch_img_paths.append(img_path)

    # Process in batches
    if len(batch_images) == batch_size or idx == len(df) - 1:
        batch_images = np.array(batch_images)

        # Get batch predictions
        batch_preds = model.predict(batch_images)

        for i, pred in enumerate(batch_preds):
            label = 1 if pred >= 0.5 else 0
            predictions.append([batch_img_paths[i], label])

        # Reset for next batch
        batch_images = []
        batch_img_paths = []

    if idx % 1000 == 0:
        print(f"Processed {idx}/{len(df)} images")



id = []
label = []
for item in predictions:
  id.append(item[0])
  label.append(item[1])


results = pd.DataFrame({
    'id': id,
    'label': label
})

results[:5]


# Save results to CSV
csv_filename = "/kaggle/working/submission_custom.csv"
results.to_csv(csv_filename, index=False)


print(f"Model saved")



from google.colab import files
files.download(csv_filename)    
files.download("/kaggle/working/my_model_custom_class.h5")
files.download("/kaggle/working/my_model_custom_class.keras")

