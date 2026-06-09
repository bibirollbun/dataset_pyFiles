# Downgrading protobuf to avoid: "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'" error
!pip install protobuf==3.20.3 -q --no-warn-conflicts


# For filtering warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Python Libraries
import os
import zipfile
import datetime
from tqdm import tqdm

# Data Handling & Visualisation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Deep Learning
import tensorflow as tf


# Configurations
config = {

    # Data
    "CLASS_INT_LABELS": None,
    "CLASS_NAMES": None,
    "TARGET_COLUMN": "encoded_label",
    "FEATURE_COLUMN": "image_path",

    # File paths
    "DATA_DIR": "/kaggle/input/dogs-vs-cats",
    "SAVE_MODEL_TO": "/kaggle/working/saved-models/",
    "EXTRACT_TO": "/kaggle/working/",
    "PROJECT_NAME": "dogs-vs-cats-classifier",

    # Dataset
    "TRAIN_RATIO": 0.80,
    "RANDOM_STATE": 7,
    "N_SAMPLES_TO_USE": 8000,

    # Model Architecture
    "IMG_SIZE": 224,
    "N_CHANNELS": 3,

    # Regularization
    "DROPOUT_RATE": 0.4,
    "L2_REG": 1e-4,
    "OUTPUT_ACTIVATION_FUNC": "sigmoid",

    # CNN Config
    "INITIAL_N_FILTERS": 32,
    "KERNEL_SIZE": (3, 3),
    "PADDING": "same",
    "KERNEL_INIT": "he_normal",

    # Dense Layer
    "DENSE_UNITS": 128,

    # Training
    "BATCH_SIZE": 32,
    "BUFFER_SIZE": 500,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 100,

    # Callback Parameters
    "METRIC_TO_MONITOR": "val_accuracy",
    "ES_PATIENCE": 10,
    "LR_PATIENCE": 3,
    "MIN_DELTA": 0.001,
    "LR_FACTOR": 0.5,
    "MIN_LR": 1e-7,
    "RESTORE_BEST_WEIGHTS": True

}


# Function to extract all files from a ZIP file into the target directory
def extract_zip_files(zip_file_name, config=config):
    """
    Extract all files from the given ZIP file into a specified directory.

    Parameters:
    - zip_file_name: Name of the ZIP file.
    - config: Configuration dictionary.

    Returns: None
    """

    # Creating the output directory if it doesn't exist
    os.makedirs(config.get("EXTRACT_TO", ""), exist_ok=True)

    # Getting the full path to the ZIP file
    zip_path = os.path.join(config.get("DATA_DIR", ""), zip_file_name)

    # Opening and reading the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        # List of all files inside the ZIP
        file_list = zip_ref.infolist()

        # Extracting files with a progress bar
        for file in tqdm(file_list, desc=f"Extracting {zip_file_name.title()}", unit="file"):
            zip_ref.extract(member=file, path=config.get("EXTRACT_TO", ""))



# Unzipping the data
extract_zip_files("train.zip")
extract_zip_files("test1.zip")


# Function to load data from folders and generate file paths for images
def load_data(dataset_type, config):
    """
    Loads image file paths for the given dataset type.

    Parameters:
    - dataset_type : One of 'train', 'training', 'test', 'testing'.
    - config : Configuration dictionary.

    Returns:
    - pd.DataFrame: DataFrame with image paths and labels (for training data).
    """

    # Normalizing dataset type
    ds_type = dataset_type.lower()

    # Validating dataset_type
    if ds_type not in ("train", "training", "test", "testing"):
        raise ValueError("Invalid 'dataset_type' provided. It should be one of ('train', 'training', 'test', 'testing').")

    # Getting the full path to the dataset folder
    base_dir = config.get("EXTRACT_TO", "")
    folder_path = next(
        (os.path.join(base_dir, folder) for folder in [ds_type, ds_type + "1"] if os.path.exists(os.path.join(base_dir, folder))),
        None
    )

    # Checking if a valid dataset folder was found
    if folder_path is None:
        raise FileNotFoundError(f"No valid folder found for dataset_type='{dataset_type}' under '{base_dir}'.")

    # List of all image filenames (ignoring hidden/system files)
    filenames = [f for f in os.listdir(folder_path) if not f.startswith(".")]

    # Building DataFrame with image file paths
    feature_col = config.get("FEATURE_COLUMN", "path")
    image_paths = [os.path.join(folder_path, f) for f in filenames]
    df = pd.DataFrame({feature_col: image_paths})

    # Adding labels for training data
    if ds_type.startswith("train"):
        df["label"] = [fname.split(".")[0] for fname in filenames]

    return df


# Loading training data
train_df = load_data("train", config)
train_df.head()


# Loading testing data
test_df = load_data("test", config)
test_df.head()


# Checking for duplicate values in the datasets
print(f"{'Training Dataset':>16}: {train_df.duplicated().sum()} duplicates")
print(f"{'Testing Dataset':>16}: {test_df.duplicated().sum()} duplicates")


# Checking the distribution of dog and cat images in the training dataset
train_df["label"].value_counts(normalize = True)


# Checking if the number of image paths matches the number of actual image files
if len(os.listdir(os.path.join(config.get("EXTRACT_TO", ""), "train"))) == len(train_df["image_path"]):
    print("File check passed: Number of filenames matches the number of image files. Good to proceed.")
else:
    print("File check failed: Mismatch between filenames and image files. Please verify.")


# Function to display the first 25 images
def display_25_images(images, labels):
    """
    Displays the first 25 images from the provided list of image file paths.

    Parameters:
    - images: List of image file paths.
    - labels: List of image labels.

    Returns: None
    """

    # Setting figure size
    plt.figure(figsize=(15, 15))

    # Looping through the images
    for i in range(25):

        # Plotting the image in a 5x5 grid
        plt.subplot(5, 5, i + 1)
        plt.imshow(plt.imread(images[i]) / 255.0)
        plt.title(f"{labels[i].title()}")

        # Hiding axis and adjusting layout
        plt.axis("off")
        plt.tight_layout()



# Displaying the first 25 images from the training dataset
display_25_images(train_df["image_path"], train_df["label"])


# Creating an instance of LabelEncoder
encoder = LabelEncoder()


# Encoding output classes for training
train_df[config.get("TARGET_COLUMN", "")] = encoder.fit_transform(train_df["label"])
train_df.head()


# Setting "CLASS_NAMES" & "CLASS_INT_LABELS"
config["CLASS_INT_LABELS"] = train_df[config.get("TARGET_COLUMN", "")].sort_values().unique()
config["CLASS_NAMES"] = encoder.inverse_transform(config.get("CLASS_INT_LABELS", []))


# Function to preprocess images
def preprocess_images(filepath, label = None, dataset_type = "train"):
    """
    Preprocesses (read, decode, and resize) an image.

    Parameters:
    - filepath: Path to the image file.
    - label: Label corresponding to the image.
    - dataset_type: The dataset type ("train", "validation", etc.).

    Returns:
    - tf.Tensor: Preprocessed image tensor, and the label if dataset_type is "train" or "validation".
    """

    # Reading the image file
    img = tf.io.read_file(filepath)

    # Decoding the JPEG image
    img = tf.image.decode_jpeg(img, channels=config.get("N_CHANNELS", 3))

    # Resizing the image
    img = tf.image.resize(img, [config.get("IMG_SIZE", 224), config.get("IMG_SIZE", 224)])

    # Returning image-label pair for training/validation datasets
    if dataset_type.startswith(("train", "training", "val", "validation")):
        if label is None:
            raise ValueError(
                f"For dataset type '{dataset_type}', labels must be provided."
            )
        return (img, label)

    elif dataset_type.startswith(("test", "testing")):
        if label is not None:
            raise ValueError(
                f"For dataset type '{dataset_type}', no labels should be provided."
            )
        return img

    else:
        raise ValueError(
            f"Invalid datasety_type='{dataset_type}'. Expected 'train', 'val', or 'test'."
        )



# Splitting data into X and y
X = train_df[config.get("FEATURE_COLUMN", "")]
y = train_df[config.get("TARGET_COLUMN", "")]


# Function to create dataset
def create_dataset(X, y=None, config=config, dataset_type="train", shuffle=True):
    """
    Creates a TensorFlow dataset for training, validation, or testing.

    Parameters:
    - X: List or array of image file paths.
    - y: Labels corresponding to X.
    - config: Configuration dictionary.
    - dataset_type: One of "train", "val", or "test".
    - shuffle: Whether to shuffle the dataset (only applies to training).

    Returns:
    - tf.data.Dataset: A TensorFlow dataset object.
    """

    # Normalzing dataset_type
    ds_type = dataset_type.lower()
    print(f"Creating {ds_type.title()} Dataset...")

    # Training & Validation Dataset
    if ds_type.startswith(("train", "training", "val", "validation")):
        if y is None:
            raise ValueError(
                f"For dataset type '{ds_type}', labels must be provided."
            )

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(
            lambda img_path, label: preprocess_images(img_path, label, ds_type),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        # Shuffling the dataset for training
        if shuffle and ds_type.startswith(("train", "training")):
            dataset = dataset.shuffle(
                buffer_size=config.get("BUFFER_SIZE", 500),
                reshuffle_each_iteration=True,
                seed=config.get("RANDOM_STATE", 7)
            )

    # Testing Dataset
    elif ds_type.startswith(("test", "testing")):
        if y is not None:
            raise ValueError(
                f"For dataset type '{ds_type}', no label should be provided."
            )
        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.map(
            lambda img_path: preprocess_images(img_path, dataset_type=ds_type),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

    else:
        raise ValueError(
            f"Invalid dataset_type='{ds_type}'. Expected 'train', 'validation', or 'test'."
        )

    # Batching, and Prefetching
    dataset = dataset.batch(batch_size=config.get("BATCH_SIZE", 32)).prefetch(tf.data.AUTOTUNE)

    return dataset


# Function to split the data into train and validation sets to create tf.data.Dataset
def prepare_train_val_datasets(X, y, config=config):
    """
    Splits the data into training and validation sets, then creates TensorFlow datasets for model training and evaluation.

    Parameters:
    - X: Array of image file paths.
    - y: Array of corresponding labels for each image.
    - config: Configuration dictionary.

    Returns:
    - tuple: (train_dataset, val_dataset)
    """

    # Train-Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size = config.get("TRAIN_RATIO", 0.80),
        random_state = config.get("RANDOM_STATE", 7),
        stratify = y  # To ensure the class distribution
    )

    # Print dataset sizes
    print(f"# Dataset sizes:")
    print(f"1. {'Training':<12}: {X_train.shape, y_train.shape}")
    print(f"2. {'Validation':<12}: {X_val.shape, y_val.shape}")

    # Creating TensorFlow dataset for the training set
    train_dataset = create_dataset(
        X = X_train,
        y = y_train,
        config = config,
        dataset_type = "train",
        shuffle = True
    )

    # Create TensorFlow dataset for the validation set
    val_dataset = create_dataset(
        X = X_val,
        y = y_val,
        config = config,
        dataset_type = "val",
        shuffle = False
    )

    return train_dataset, val_dataset


# Function to build a custom VGG model
def build_vgg_model(inputs):
    """
    Builds a VGG-Style feature extraction layers.

    Parameters:
    - inputs: Input tensor.

    Returns:
    - Tensor after feature extraction.
    """

    # Conv Block: 1 (Stride = 2 for downsampling)
    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 1,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
        strides=2,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 1,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPooling Layer: 1
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    
    # Conv Block: 2
    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 2,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 2,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPooling Layer: 2
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)


    # Conv Block: 3
    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 4,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 4,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPooling Layer: 3
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    
    # Conv Block: 4
    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 8,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters=config.get("INITIAL_N_FILTERS", 32) * 8,
        kernel_size=config.get("KERNEL_SIZE", (3, 3)),
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPooling Layer: 4
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Dropout Layer
    x = tf.keras.layers.Dropout(rate = config.get("DROPOUT_RATE", 0.4))(x)

    return x



# Function to create residual block for ResNet model
def residual_block(x, filters, stride=1, config=config):
    """
    Builds a residual block with optional downsampling.

    Parameters:
    - x: Input tensor.
    - filters: Number of filters for conv layers.
    - stride: Stride for downsampling.
    - config: Configuration dictionary.

    Returns:
    - Output tensor of the residual block.
    """

    # Saving input for shortcut
    shortcut = x

    # Conv Block: 1
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.get("KERNEL_SIZE", (3,3)),
        strides=stride,
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Conv Block: 2
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=config.get("KERNEL_SIZE", (3,3)),
        strides=1,
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Matching Shortcut Dimensions in case of shape mismatch
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=stride,
            padding=config.get("PADDING", "same"),
            kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Adding & Activating
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x


# Function to build a ResNet-Style model
def build_resnet_model(inputs):
    """
    Builds a ResNet-style feature extractor.

    Parameters:
    - inputs: Input tensor.

    Returns:
    - Output tensor after the ResNet feature extractor.

    """

    # Stem Block: Initial Conv, BatchNormalisation, ReLU, and Downsampling
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=2,
        padding=config.get("PADDING", "same"),
        kernel_initializer=config.get("KERNEL_INIT", "he_normal"),
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding=config.get("PADDING", "same"))(x)

    # Residual Stack
    filter_sizes = [64, 128, 256, 512]
    num_blocks = [2, 2, 2, 2]
    
    for i, (n_filters, n_blocks) in enumerate(zip(filter_sizes, num_blocks)):
        for block_idx in range(n_blocks):
            # Downsampling at the first block of each stage (except the first stage)
            stride = 2 if (i > 0 and block_idx == 0) else 1
            x = residual_block(x, filters=n_filters, stride=stride)

    return x


# Function to build a classification head
def build_classification_head(features, config=config):
    """
    Builds the classification head on top of extracted features.

    Parameters:
    - features: Output tensor from the feature extractor.
    - config: Configuration dictionary.

    Returns:
    - Final prediction tensor.
    """

    # Getting the features
    x = features

    # Applying global pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(config.get("DROPOUT_RATE", 0.4))(x)


    # Dense Block
    x = tf.keras.layers.Dense(
        units = config.get("DENSE_UNITS", 128),
        kernel_initializer = config.get("KERNEL_INIT", "he_normal"),
        kernel_regularizer = tf.keras.regularizers.l2(config.get("L2_REG", 1e-4))
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(config.get("DROPOUT_RATE", 0.4))(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        units = 1,
        activation = config.get("OUTPUT_ACTIVATION_FUNC", "sigmoid"),
        name = "output"
    )(x)

    return outputs


# Function to create the final model
def create_model(model_name="vgg", config=config):
    """
    Builds and compiles a complete model.

    Parameters:
    - model_name: The base model to use ("vgg" or "resnet").
    - config: Configuration dictionary.

    Returns:
    - tf.keras.Model: Compiled model ready for training.
    """

    print(f"\n{'='*60}")
    print(f"CREATING {model_name.upper()} MODEL")
    print(f"{'='*60}")

    # Normalizing model_name
    model_name = model_name.lower()

    # Input shape
    input_shape = (
        config.get("IMG_SIZE", 224),
        config.get("IMG_SIZE", 224),
        config.get("N_CHANNELS", 3)
    )

    # Input layer
    inputs = tf.keras.layers.Input(shape = input_shape)

    # Data augmentation
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed = config.get("RANDOM_STATE", 7)),          # Random horizontal flip
        tf.keras.layers.RandomRotation(0.15, seed = config.get("RANDOM_STATE", 7)),              # Random rotation
        tf.keras.layers.RandomZoom(0.1, seed = config.get("RANDOM_STATE", 7)),                   # Random zoom
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=config.get("RANDOM_STATE", 7)),         # Random translation
        tf.keras.layers.RandomContrast(0.15, seed=config.get("RANDOM_STATE", 7)),                # Random contrast
        tf.keras.layers.RandomBrightness(0.15, seed=config.get("RANDOM_STATE", 7)),              # Random brightness
    ], name="augmentation")
    x = augmentation(inputs)

    # Rescale pixel values
    x = tf.keras.layers.Rescaling(1./255)(x)

    # Building the base model
    if model_name.startswith(("vgg")):
        base_output = build_vgg_model(inputs = x)

    elif model_name.startswith("resnet"):
        base_output = build_resnet_model(inputs = x)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}. Expected one of ('vgg', or 'resnet').")

    # Adding the classification head
    outputs = build_classification_head(features = base_output, config = config)

    # Creating the complete model
    model = tf.keras.Model(
        inputs = inputs,
        outputs = outputs,
        name = f"{model_name}_classifier"
    )

    # Compiling the model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = config.get("LEARNING_RATE", 0.001)
        ),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name = "precision"),
            tf.keras.metrics.Recall(name = "recall"),
            tf.keras.metrics.AUC(name = "auc")
        ]
    )

    return model


# Function to get the list of callbacks
def get_callbacks(config=config):

    """
    Create and return a list of `tf.keras.callbacks` configured using the provided values.

    Parameters
    - config : Configuration dictionary.

    Returns
    - list: List of `tf.keras.callbacks` instances to use during model training.
    """

    # List to store callbacks
    callback_list = []

    # EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = config.get("METRIC_TO_MONITOR", "val_accuracy"),
        min_delta = config.get("MIN_DELTA", 0.001),
        patience = config.get("ES_PATIENCE", 10),
        verbose = 1,
        restore_best_weights = config.get("RESTORE_BEST_WEIGHTS", True),
        mode='max'
    )
    callback_list.append(early_stopping)

    # ReduceLROnPlateau callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = config.get("METRIC_TO_MONITOR", "val_accuracy"),
        factor = config.get("LR_FACTOR", 0.5),
        patience = config.get("LR_PATIENCE", 3),
        verbose = 1,
        min_delta = config.get("MIN_DELTA", 0.001),
        min_lr = config.get("MIN_LR", 1e-7),
        mode='max'
    )
    callback_list.append(reduce_lr)

    return callback_list


# Function to train the model
def train_model(model_name, training_dataset, num_epochs, validation_dataset, callbacks):
    """
    Creates and trains a model using the provided datasets and callbacks.

    Parameters:
    - model_name:  The base model to use ("vgg" or "resnet").
    - training_dataset: Dataset for training the model.
    - num_epochs: Number of epochs to train the model.
    - validation_dataset: Dataset for validating the model during training.
    - callbacks: List of callbacks to use during training.

    Returns:
    - tuple: (model, history)
    """

    # Creating the full model
    model = create_model(model_name = model_name)

    # Training the model
    history = model.fit(
        training_dataset,
        epochs = num_epochs,
        validation_data = validation_dataset,
        callbacks = callbacks,
        verbose=1,
    )

    # Return the trained model
    return model, history


# Creating training and validation datasets for VGG model
train_dataset_vgg, val_dataset_vgg = prepare_train_val_datasets(
    X = X[:config.get("N_SAMPLES_TO_USE", 8000)],
    y = y[:config.get("N_SAMPLES_TO_USE", 8000)],
)


# Training VGG model
vgg_model, vgg_model_history = train_model(
    model_name = "vgg",
    training_dataset = train_dataset_vgg,
    num_epochs = config.get("N_EPOCHS", 100),
    validation_dataset = val_dataset_vgg,
    callbacks = get_callbacks()
)


# Creating training and validation datasets for ResNet model
train_dataset_resnet, val_dataset_resnet = prepare_train_val_datasets(
    X = X[:config.get("N_SAMPLES_TO_USE", 8000)],
    y = y[:config.get("N_SAMPLES_TO_USE", 8000)],
)


# Training ResNet model
resnet_model, resnet_model_history = train_model(
    model_name = "resnet",
    training_dataset = train_dataset_resnet,
    num_epochs = config.get("N_EPOCHS", 100),
    validation_dataset = val_dataset_resnet,
    callbacks = get_callbacks()
)


# VGG Model
vgg_probabilities = vgg_model.predict(val_dataset_vgg)
vgg_predicted_labels = (vgg_probabilities >= 0.5).astype(int)


# ResNet Model
resnet_probabilities = resnet_model.predict(val_dataset_resnet)
resnet_predicted_labels = (resnet_probabilities >= 0.5).astype(int)


# Function to unbatch the data and extract images and labels from it
def extract_images_and_labels(dataset):
    """
    Unbatches a TensorFlow dataset and extracts images and labels into lists.

    Parameters:
    - dataset: A batched TensorFlow dataset containing image and label pairs.

    Returns:
    - Tuple containing:
        - images: List of image arrays (NumPy).
        - labels: List of label arrays/values (NumPy).
    """
    images = []
    labels = []
    
    # Unbatching the dataset and creating an iterator to loop through it
    iterator = dataset.unbatch().as_numpy_iterator()

    for image, label in iterator:
        images.append(image)
        labels.append(label)

    return images, labels


# Separating images and labels
images_vgg, labels_vgg = extract_images_and_labels(val_dataset_vgg)
images_resnet, labels_resnet = extract_images_and_labels(val_dataset_resnet)


# Function to visualise training and validation performance of the trained model
def plot_training_history(history, figsize = (12, 5)):
    """
    Plots training and validation accuracy and loss.

    Parameters:
    - history: The .history dictionary from a trained Keras model.
    - figsize: Tuple defining the figure size (width, height).
    """
    
    # Getting training history
    hist = history.history if hasattr(history, "history") else history   

    # Getting training and validation accuracy values
    acc = hist.get("accuracy")
    val_acc = hist.get("val_accuracy")

    # Getting training and validation loss values
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    # No of epochs for Xticks
    epochs = range(1, len(acc) + 1)

    # Setting figure size
    plt.figure(figsize=figsize, dpi=350)

    # For Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Acc", color="orange", marker='o', markersize=4)
    plt.plot(epochs, val_acc, label="Validation Acc", color="blue", marker='o', markersize=4)
    plt.title("Training & Validation Accuracy", fontweight="bold")
    plt.xlabel("Epochs", fontweight="bold")
    plt.ylabel("Accuracy", fontweight="bold")
    plt.legend()
    
    # For Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss", color="orange", marker="o", markersize=4)
    plt.plot(epochs, val_loss, label="Validation Loss", color="blue", marker="o", markersize=4)
    plt.title("Training & Validation Loss", fontweight="bold")
    plt.xlabel("Epochs", fontweight="bold")
    plt.ylabel("Loss", fontweight="bold")
    plt.legend()
    plt.show()


# Function to plot confusion matrix of a trained model
def show_confusion_matrix(y_true, y_pred, class_names = config.get("CLASS_NAMES", []), figsize = (12, 5)):
    """
    Plots the confusion matrix.

    Parameters:
    - y_true: List of true class labels.
    - y_pred: List of predicted class labels.
    - class_names: List of custom class names for axis labels.
    - figsize: Tuple defining the figure size (width, height).

    Returns: None
    """
    
    # Setting the figure size
    plt.figure(figsize=figsize, dpi = 350)
    
    # Computing the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting the confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False, 
        xticklabels=class_names, 
        yticklabels=class_names    
    )

    # Adding title and labels to the plot
    plt.title("Confusion Matrix", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.show()



# Function to plot an image with its predicted and true labels along with confidence
def plot_image_with_prediction(prediction_probs, images, true_labels, class_names=config.get("CLASS_NAMES", []), sample_index=7, threshold=0.5):
    """
    Plot a single image with its predicted and true label for a binary classifier.

    Parameters:
    - prediction_probs: Array of prediction probabilities for each sample.
    - images: Array of images corresponding to the predictions.
    - true_labels: Array of true labels (0 or 1).
    - class_names: List of class names.
    - sample_index: Index of the sample to visualize.
    - threshold: Probability threshold for determining predicted class.

    Returns: None.
    """
    
    # Validating sample index
    if sample_index >= len(images):
        raise ValueError(f"Sample index {sample_index} is out of bounds for image list of size {len(images)}.")

    # Extracting probability and image
    prob = prediction_probs[sample_index][0]
    img = images[sample_index]

    # Converting true label to integer
    true_label_idx = int(true_labels[sample_index])

    # Validating true label index
    if true_label_idx >= len(class_names):
        raise ValueError(f"Label index {true_label_idx} is out of bounds for class_names: {class_names}")

    # Determining predicted class and confidence
    if prob >= threshold:
        predicted_idx = 1
        confidence_score = prob * 100
    else:
        predicted_idx = 0
        confidence_score = (1 - prob) * 100

    # Normalizing image if needed
    if img.max() > 1.0:
        img = img / 255.0

    # Setting title color
    color = "green" if predicted_idx == true_label_idx else "red"

    # Plotting the image with labels
    plt.imshow(img)
    plt.axis("off")

    # Setting the plot title
    plt.title(
        f"Pred: {class_names[predicted_idx].title()} \nTrue: {class_names[true_label_idx].title()} \nConfidence: {confidence_score:.1f}%",
        color=color,
        fontsize=12
    )


# Function to display a grid of images with predicted labels and confidence scores
def display_prediction_grid(prediction_probs, images, true_labels, class_names=config.get("CLASS_NAMES", []), start_index=0, num_rows=3, num_cols=2, threshold=0.5):
    """
    Display a grid of images with predicted and true labels with confidence.

    Parameters:
    - prediction_probs: Array of prediction probabilities for each sample.
    - images: Array of images to display.
    - true_labels: Array of true labels.
    - class_names: List of class names.
    - start_index: Index to start displaying from.
    - num_rows: Number of rows in the grid.
    - num_cols: Number of columns in the grid.
    - threshold: Probability threshold for determining predicted class.

    Returns: None.
    """

    # Calculating total number of images to display
    num_images = num_rows * num_cols

    # Setting figure size based on grid dimensions
    plt.figure(figsize=(num_cols * 4, num_rows * 4))

    # Looping through each grid slot
    for i in range(num_images):
        current_index = start_index + i

        # Stop if we run out of images
        if current_index >= len(images):
            break

        # Creating subplot for current image
        plt.subplot(num_rows, num_cols, i + 1)

        try:
            # Plotting the image using the helper function
            plot_image_with_prediction(
                prediction_probs=prediction_probs,
                images=images,
                true_labels=true_labels,
                class_names=class_names,
                sample_index=current_index,
                threshold=threshold
            )

        except ValueError as e:
            # Displaying error inside the grid cell
            plt.text(0.5, 0.5, f"Error:\n{str(e)}", ha="center", va="center")
            plt.axis("off")

    # Adjusting layout to avoid overlap
    plt.tight_layout()



# Visualising model performance of the trained VGG model
plot_training_history(vgg_model_history)


# Visualising confusion matrix of the trained VGG model
show_confusion_matrix(
    y_true = labels_vgg,
    y_pred = vgg_predicted_labels
)


# Displaying predictions from a VGG-Style model
display_prediction_grid(
    prediction_probs=vgg_probabilities,
    images=images_vgg,
    true_labels=labels_vgg,
    start_index = 0,
    num_rows = 5,
    num_cols = 5
)


# Visualising model performance of the trained ResNet model
plot_training_history(resnet_model_history)


# Visualising confusion matrix of the trained ResNet model
show_confusion_matrix(
    y_true = labels_resnet,
    y_pred = resnet_predicted_labels
)


# Displaying predictions from a ResNet-Style model
display_prediction_grid(
    prediction_probs=resnet_probabilities,
    images=images_resnet,
    true_labels=labels_resnet,
    start_index = 0,
    num_rows = 5,
    num_cols = 5
)


# Function to save the trained model
def save_model_with_timestamp(model, suffix="", base_dir=config.get("SAVE_MODEL_TO", "")):
    """
    Save a trained Keras model with a timestamped filename.

    Parameters:
    - model: Trained TensorFlow/Keras model.
    - suffix: Optional text appended to the filename.
    - base_dir: Directory where the model will be saved. Created if missing.

    Returns:
    - str: Full path of the saved model file.
    """

    # Using default directory if base directory is none
    if base_dir == "":
        base_dir = "./saved-models/"

    # Creating base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Creating filename with current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"model_{timestamp}"

    # Appending suffix if provided
    if suffix:
        filename += f"_{suffix}"

    filename += ".keras"

    # Createing full model path
    model_filepath = os.path.join(base_dir, filename)

    # Saving the model
    print(f"Saving model to: {model_filepath}...")
    try:
        
        model.save(model_filepath)
        print("Model saved successfully")
        return model_filepath
    except Exception as e:
        print(f"Failed to save model: {e}")
        raise e



# Function to load a saved Keras model
def load_model_from_filepath(model_filepath, mode=True):
    """
    Load a trained Keras model from a file path.

    Parameters:
    - model_filepath: Full path to the saved .keras or .h5 file.
    - safe_mode: If True, prevents loading models with potentially unsafe Lambda layers.

    Returns:
    - tf.keras.Model: Loaded Keras model.
    """
    # Check if the saved model file exists
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"No model found at {model_filepath}")

    print(f"Loading saved model from: {model_filepath}...")

    try:
        
        # Loading the Keras model
        model = tf.keras.models.load_model(model_filepath, safe_mode=mode)
        print("Model loaded successfully")
        return model
    
    except Exception as e:
    
        # Handling loading errors
        print(f"Failed to load model. Reason: {e}")
        raise e



# Creating training and validation datasets for VGG-Style model
train_dataset_full, val_dataset_full = prepare_train_val_datasets(
    X = X,
    y = y,
)


# Training VGG model on full dataset
model, model_history = train_model(
    model_name = "vgg",
    training_dataset = train_dataset_full,
    num_epochs = config.get("N_EPOCHS", 100),
    validation_dataset = val_dataset_full,
    callbacks = get_callbacks()
)


# Function to make predictions on a dataset and visualize them in a grid
def predict_and_visualize_model(model, dataset, class_names=config.get("CLASS_NAMES", []), num_rows=3, num_cols=2):
    """
    Run predictions on a dataset and displays results in a grid.

    Parameters:
    - model: Trained Keras model.
    - dataset: tf.data.Dataset to predict on.
    - class_names: List of class names.
    - num_rows: Number of rows in the grid.
    - num_cols: Number of columns in the grid.

    Returns:
    - prediction_probs: Model's prediction probabilities.
    - images: Images from the dataset.
    - true_labels: True labels corresponding to images.
    """
    
    print("Making predictions...")
    
    # Making predictions
    prediction_probs = model.predict(dataset)

    print("Extracting images and labels...")

    # Extracting images and true labels from dataset
    images, true_labels = extract_images_and_labels(dataset)

    # Validating number of samples
    num_samples = num_rows * num_cols
    if len(images) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but dataset only has {len(images)}.")

    print("Generating visualization grid...")
    
    # Displaying the prediction grid
    display_prediction_grid(
        prediction_probs=prediction_probs,
        images=images,
        true_labels=true_labels,
        class_names=class_names,
        num_rows=num_rows,
        num_cols=num_cols,
        start_index=0
    )

    return prediction_probs, images, true_labels


# Predicting and visualizing them 
preds, images, labels = predict_and_visualize_model(
    model=model,
    dataset=val_dataset_full,
    class_names=config.get("CLASS_NAMES", []),
    num_rows=5,
    num_cols=5
)


# Visualising model performance of the trained model
plot_training_history(model_history)


# Converting probabilities to binary class labels
pred_labels = (preds >= 0.5).astype(int)


# Visualising confusion matrix of the trained model
show_confusion_matrix(
    y_true = labels,
    y_pred = pred_labels
)


# Saving the fully trained model
suffix = "VggCNN_CatsVsDog_e100_b32"
saved_model_filepath = save_model_with_timestamp(model, suffix = suffix)


# Loading saved model
loaded_model = load_model_from_filepath(saved_model_filepath, mode = True)


# Using the loaded model to make predictions and visualizing the results
preds, images, labels = predict_and_visualize_model(
    model=loaded_model,
    dataset=val_dataset_full,
    class_names=config.get("CLASS_NAMES", []),
    num_rows=5,
    num_cols=5
)


def generate_submission_file(model, test_df, output_filename="submission.csv", threshold=0.5):
    """
    Generate a submission file using model predictions on test images.

    Parameters:
    - model: Trained model used for generating predictions.
    - test_df: DataFrame containing test image paths.
    - output_filename: Name of the output CSV file.
    - threshold: Probability threshold for determining predicted class.

    Returns: None.
    """

    print("Preparing test data...")

    # Getting image IDs from file names
    test_df["id"] = test_df["image_path"].apply(lambda x: int(os.path.basename(x).split(".")[0]))

    # Building test dataset
    print(f"Processing {len(test_df)} images...")
    test_dataset = create_dataset(
        X=test_df["image_path"],
        dataset_type="test",
        shuffle=False
    )

    print("Predicting...")
    raw_predictions = model.predict(test_dataset, verbose=1)

    # Flattening predictions if needed
    predictions = raw_predictions.flatten() if raw_predictions.ndim > 1 else raw_predictions

    # Creating submission DataFrame
    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "preds": predictions
    })

    # Converting predicted probabilities to binary class labels
    submission_df["label"] = submission_df["preds"].apply(lambda x: 1 if x >= threshold else 0)

    # Saving submission file
    submission_df.drop("preds", axis = 1).to_csv(output_filename, index=False)
    print(f"Submission saved to '{output_filename}'")



# Generating submission file
generate_submission_file(
    model = loaded_model,
    test_df = test_df
)




