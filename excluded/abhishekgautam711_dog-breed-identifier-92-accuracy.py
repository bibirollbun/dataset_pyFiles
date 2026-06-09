# For filtering warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Python Libraries
import os
import datetime

# Data Handling & Visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Deep Learning
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB3


# Configurations
config = {

    # Data
    "CLASS_INT_LABELS": None,
    "CLASS_NAMES": None,
    "TARGET_COLUMN": "encoded_breed",
    "FEATURE_COLUMN": "image_path",

    # File paths
    "DATA_DIR": "/kaggle/input/dog-breed-identification/",
    "SAVE_MODEL_TO": "/kaggle/working/saved-models/",
    "PROJECT_NAME": "dob-breed-identification-classifier",

    # Split ratio
    "TRAIN_RATIO": 0.80,
    "RANDOM_STATE": 7,
    "N_SAMPLES_TO_USE": 3000,

    # Model Architecture
    "MOBILE_NET_IMG_SIZE": 224,
    "EFFICIENT_NET_IMG_SIZE": 300,
    "N_CHANNELS": 3,
    "ACTIVATION_FUNC": "relu",
    "DROPOUT_RATE_1": 0.3,
    "DROPOUT_RATE_2": 0.5,
    "DENSE_UNITS": 256,
    "L2_REGULARISATION": 0.01,
    "OUTPUT_ACTIVATION_FUNC": "softmax",

    # Training Parameters
    "BATCH_SIZE": 32,
    "BUFFER_SIZE": 500,
    "LEARNING_RATE": 0.001,
    "N_EPOCHS": 100,
    "LABEL_SMOOTHING": 0.15,

    # Callback Parameters
    "METRIC_TO_MONITOR": "val_loss",
    "ES_PATIENCE": 5,
    "LR_PATIENCE": 2,
    "MIN_DELTA": 0.001,
    "LR_FACTOR": 0.5,
    "MIN_LR": 1e-7,
    "RESTORE_BEST_WEIGHTS": True

}


# Function to load data from a csv file
def load_data(filename, config):
    """
    Reads a CSV file and generates full file paths for training images.

    Parameters:
    - filename: Name of the CSV file.
    - config: Configuration dictionary.

    Returns:
    - DataFrame with image file paths.
    """

    # Full path to the CSV file
    filepath = config.get("DATA_DIR", "") + filename

    # Reading data from the file
    df = pd.read_csv(filepath)

    # Generating full file path for each training image
    df[config.get("FEATURE_COLUMN", "")] = df["id"].apply(
        lambda id: os.path.join(config.get("DATA_DIR", ""), "train", f"{id}.jpg")
    )

    return df


# Loading data from the file: "labels.csv"
labels = load_data("labels.csv", config)
labels.head()


# Checking for duplicate values in the dataset
labels.duplicated().sum()


# Checking the top 10 dog breeds
labels["breed"].value_counts().nlargest(10)


# Checking the bottom 10 dog breeds
labels["breed"].value_counts().nsmallest(10)


# Checking what is the average number of images for each class
labels["breed"].value_counts().median()


# Checking if the number of image_path matches the number of actual image files
if len(os.listdir(os.path.join(config.get("DATA_DIR", ""), "train"))) == len(labels["image_path"]):
    print("File check passed: Number of filenames matches the number of image files. Good to proceed.")
else:
    print("File check failed: Mismatch between filenames and image files. Please verify.")


# Setting figure size
plt.figure(figsize = (12, 5))
plt.imshow(
    # Reading an image (normalized)
    plt.imread(labels.loc[9000, "image_path"]) / 255.0
)
plt.title(" ".join(labels.loc[9000, "breed"].split("_")).title())
plt.axis("off");


# Creating an instance of LabelEncoder
encoder = LabelEncoder()


# Encoding dog breeds
labels[config.get("TARGET_COLUMN", "")] = encoder.fit_transform(labels["breed"])
labels.head()


# Setting "CLASS_NAMES", and "CLASS_INT_LABELS"
config["CLASS_INT_LABELS"] = labels["encoded_breed"].sort_values().unique()
config["CLASS_NAMES"] = encoder.inverse_transform(config.get("CLASS_INT_LABELS", []))


# Function to preprocess images
def preprocess_images(filepath, img_size):
    """
    Reads and preprocesses an image based on the model requirements.

    Parameters:
    - filepath: Path to the image file.
    - img_size: Target size to resize the image, e.g., (224, 224) or (300, 300).

    Returns:
    - Preprocessed image tensor.
    """

    # Reading image
    img = tf.io.read_file(filepath)

    # Decoding JPEG image
    img = tf.image.decode_jpeg(img, channels=config.get("N_CHANNELS", 3))

    # Resizing image
    img = tf.image.resize(img, [img_size, img_size])

    return img


# Function to preprocess image and convert label to one-hot encoding
def get_processed_image_and_label(image_path, label, img_size, total_classes=len(config.get("CLASS_NAMES", []))):
    """
    Loads and preprocesses an image, and converts its label to one-hot encoding.

    Parameters:
    - image_path: Path to the image file.
    - label: Integer label for the image.
    - img_size: Target size to resize the image, e.g., (224, 224) or (300, 300).
    - total_classes: Total number of classes for one-hot encoding.

    Returns:
    - Tuple of (preprocessed image tensor, one-hot encoded label).
    """

    # Preprocessing the image
    image = preprocess_images(image_path, img_size)

    # Converting label to one-hot encoding
    label = tf.one_hot(label, depth = total_classes)
    return (image, label)


# Splitting data into X and y
X = labels[config.get("FEATURE_COLUMN", "")]
y = labels[config.get("TARGET_COLUMN", "")]


# Function to create a TensorFlow dataset for training, validation, or testing
def create_dataset(X, y=None, model_name=None, config=config, dataset_type="train", shuffle=True):
    """
    Creates a TensorFlow dataset for training, validation, or testing.

    Parameters:
    - X: List or array of image file paths.
    - y: Labels corresponding to X.
    - model_name: Name of the model to load (e.g., 'EfficientNetB0', 'MobileNetV2').
    - config: Configuration dictionary.
    - dataset_type: One of "train", "val", or "test".
    - shuffle: Whether to shuffle the dataset (only applies to training).

    Returns:
    - A tf.data.Dataset object.
    """

    if "efficientnet" in model_name.lower():
        # Input shape for EfficientNetV2
        img_size = config.get("EFFICIENT_NET_IMG_SIZE", 300)

    elif "mobilenet" in model_name.lower():
        # Input shape for MobileNetV2
        img_size = config.get("MOBILE_NET_IMG_SIZE", 224)
    
    # Training & Validation Dataset
    if dataset_type.lower() in ["train", "training", "val", "validation"] and y is not None:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(
            lambda img_path, label: get_processed_image_and_label(img_path, label, img_size),
            num_parallel_calls = tf.data.AUTOTUNE,
            deterministic=True
        )

        # For training dataset
        if shuffle and dataset_type.lower() in ["train", "training"]:
            dataset = dataset.shuffle(
                buffer_size = config.get("BUFFER_SIZE", 500),
                reshuffle_each_iteration = True,
                seed = config.get("RANDOM_STATE", 7)
            )

    # Testing Dataset
    elif dataset_type.lower() in ["test", "testing"] and y is None:
        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.map(
            lambda img_path: preprocess_images(img_path, img_size),
            num_parallel_calls = tf.data.AUTOTUNE,
            deterministic=True
        )

    else:
        raise ValueError(
            "Invalid dataset_type or label configuration. Please verify your inputs."
        )

    # Batching, Caching and Prefetching
    dataset = dataset.batch(batch_size = config.get("BATCH_SIZE", 32)).prefetch(tf.data.AUTOTUNE)

    return dataset


# Function to prepare training and validation datasets
def prepare_train_val_datasets(X, y, model_name, config=config):
    """
    Splits the data into training and validation sets, then creates TensorFlow datasets for model training and evaluation.

    Parameters:
    - X: Array of image file paths.
    - y: Array of corresponding labels for each image.
    - model_name: Name of the model to load (e.g., 'EfficientNetB0', 'MobileNetV2').
    - config: Configuration dictionary.

    Returns:
    - tuple: (train_dataset, val_dataset).
    """

    # Train-Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size = config.get("TRAIN_RATIO", 0.80),
        random_state = config.get("RANDOM_STATE", 7),
        stratify=y  # To ensure the class distribution
    )

    # Printing dataset sizes
    print(f"# Dataset sizes:")
    print(f"1. {'Training':<12}: {X_train.shape, y_train.shape}")
    print(f"2. {'Validation':<12}: {X_val.shape, y_val.shape}")

    # Creating TensorFlow dataset for the training set with optional shuffling
    train_dataset = create_dataset(
        X = X_train,
        y = y_train,
        model_name = model_name,
        config = config,
        dataset_type = "train",
        shuffle = True
    )

    # Create TensorFlow dataset for the validation set without shuffling
    val_dataset = create_dataset(
        X = X_val,
        y = y_val,
        model_name = model_name,
        config = config,
        dataset_type = "val",
        shuffle = False
    )

    return train_dataset, val_dataset



# Function to load the base model
def load_base_model(model_name, input_shape, config=config):

    """
    Load a pre-trained base model without the top classification layer.

    Parameters:
    - model_name: Name of the model to load (e.g., 'EfficientNetB0', 'MobileNetV2').
    - input_shape: The shape of input images.
    - config: Configuration dictionary.

    Returns:
    - tf.keras.Model: Base model ready for fine-tuning.
    """

    # Output shape
    output_shape = len(config.get("CLASS_NAMES", []))

    # MobileNetV2
    if "mobilenet" in model_name.lower():
        base_model = MobileNetV2(
            input_shape = input_shape,
            include_top = False,
            weights = "imagenet"
        )

    # EfficientNetB3
    elif "efficientnet" in model_name.lower():
        base_model = EfficientNetB3(
            input_shape = input_shape,
            include_top = False,
            weights = "imagenet"
        )

    else:
        # Raise error if model name is not correct
        raise ValueError(f"Unknown model name: {model_name.lower()}. Expected one of ('mobilenet', or 'efficientnet'.")

    # Freezing the base model layers
    base_model.trainable = False

    return base_model


# Function to create and compile a model
def create_model(base_model, input_shape, config=config):
    """
    Creates and compiles a model using a base model.

    Parameters:
    - base_model: The base pre-trained model.
    - input_shape: The shape of input images.
    - config: Configuration dictionary.

    Returns:
    - tf.keras.Model: Compiled model ready for training.
    """
    print(f"Building model with base from: {base_model.name}")

    # Number of output units
    output_units = len(config.get("CLASS_NAMES", []))

    # Input Layer
    input = tf.keras.layers.Input(shape = input_shape)

    # Data augmentation
    x = tf.keras.layers.RandomFlip("horizontal", seed= config.get("RANDOM_STATE", 7))(input, training = True)     # Random horizontal flip
    x = tf.keras.layers.RandomRotation(0.15, seed=config.get("RANDOM_STATE", 7))(x, training = True)              # Random rotation
    x = tf.keras.layers.RandomZoom(0.15, seed=config.get("RANDOM_STATE", 7))(x, training = True)                  # Random zoom
    x = tf.keras.layers.RandomContrast(0.15, seed=config.get("RANDOM_STATE", 7))(x, training = True)              # Random contrast
    x = tf.keras.layers.RandomBrightness(0.15, seed=config.get("RANDOM_STATE", 7))(x, training = True)            # Random brightness


    # Rescale pixel values if the model is not EfficientNet
    if "efficientnet" not in base_model.name.lower():
        x = tf.keras.layers.Rescaling(1./255)(x)

    # Pre-trained base model
    x = base_model(x, training = False)

    # Global Average Pooling Layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dropout Layer
    x = tf.keras.layers.Dropout(
        rate = config.get("DROPOUT_RATE_1", 0.3),
        seed = config.get("RANDOM_STATE", 7)
    )(x)

    # Dense Layer
    x = tf.keras.layers.Dense(
        units = config.get("DENSE_UNITS", 256) // 2,
        activation = config.get("ACTIVATION_FUNC", "relu"),
        kernel_regularizer = tf.keras.regularizers.l2(config.get("L2_REGULARISATION", 0.01))
    )(x)

    # Dropout Layer
    x = tf.keras.layers.Dropout(
        rate = config.get("DROPOUT_RATE_2", 0.5),
        seed = config.get("RANDOM_STATE", 7)
    )(x)

    # Output Layer
    output = tf.keras.layers.Dense(
        units = output_units,
        activation = "softmax",
    )(x)

    # Creating model
    model = tf.keras.Model(
        inputs = input,
        outputs = output,
        name = f"{base_model.name}_dog_classifier"
    )

    # Compiling the model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = config.get("LEARNING_RATE", 0.001)
        ),
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing = config.get("LABEL_SMOOTHING", 0.15)
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k = 3, name = "top_3_accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k = 5, name = "top_5_accuracy")
        ]
    )

    return model


# EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = config.get("METRIC_TO_MONITOR", "val_loss"),
    min_delta = config.get("MIN_DELTA", 0.005),
    patience = config.get("ES_PATIENCE", 5),
    verbose = 1,
    restore_best_weights = config.get("RESTORE_BEST_WEIGHTS", True)
)


# ReduceLROnPlateau callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = config.get("METRIC_TO_MONITOR", "val_loss"),
    factor = config.get("LR_FACTOR", 0.5),
    patience = config.get("LR_PATIENCE", 2),
    verbose = 1,
    min_delta = config.get("MIN_DELTA", 0.005),
    min_lr = config.get("MIN_LR", 1e-7),
)


# Function to train the model
def train_model(base_model, training_dataset, num_epochs, validation_dataset, callbacks):
    """
    Creates and trains a model using the provided datasets and callbacks.

    Parameters:
    - base_model: The base pre-trained model.
    - training_dataset: Dataset for training the model.
    - num_epochs: Number of epochs to train.
    - validation_dataset: Dataset for validation during training.
    - callbacks: List of callbacks to use during training.

    Returns:
    - model: The trained model and training history.
    """

    if "efficientnet" in base_model.lower():
        # Input shape for EfficientNetV2
        input_shape = (
            config.get("EFFICIENT_NET_IMG_SIZE", 300),
            config.get("EFFICIENT_NET_IMG_SIZE", 300),
            config.get("N_CHANNELS", 3)
        )

    elif "mobilenet" in base_model.lower():
        # Input shape for MobileNetV2
        input_shape = (
            config.get("MOBILE_NET_IMG_SIZE", 224),
            config.get("MOBILE_NET_IMG_SIZE", 224),
            config.get("N_CHANNELS", 3)
        )


    # Loading the base model and creating the full model
    base_model = load_base_model(base_model, input_shape)
    model = create_model(base_model, input_shape)

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


# Creating training and validation datasets for MobileNetV2 model
train_dataset_mbnetv2, val_dataset_mbnetv2 = prepare_train_val_datasets(
    X = X[:config.get("N_SAMPLES_TO_USE", 3000)],
    y = y[:config.get("N_SAMPLES_TO_USE", 3000)],
    model_name = "MobileNetV2"
)


# Training MobileNetV2 model
mbnetv2_model, mbnetv2_model_history = train_model(
    base_model = "MobileNetV2",
    training_dataset = train_dataset_mbnetv2,
    num_epochs = config.get("N_EPOCHS", 100),
    validation_dataset = val_dataset_mbnetv2,
    callbacks = [early_stopping, reduce_lr]
)


# Creating training and validation datasets for EfficientNetB3 model
train_dataset_efnetb3, val_dataset_efnetb3 = prepare_train_val_datasets(
    X = X[:config.get("N_SAMPLES_TO_USE", 3000)],
    y = y[:config.get("N_SAMPLES_TO_USE", 3000)],
    model_name = "EfficientNetB3"
)


# Training EfficientNetB3 model
efnetb3_model, efnetb3_model_history = train_model(
    base_model = "EfficientNetB3",
    training_dataset = train_dataset_efnetb3,
    num_epochs = config.get("N_EPOCHS", 100),
    validation_dataset = val_dataset_efnetb3,
    callbacks = [early_stopping, reduce_lr]
)


# Function to get the predicted label from probabilities
def get_predicted_class_label(prediction_probabilities, config = config):
    """
    Converts prediction probabilities into their corresponding class label.

    Parameters:
    - prediction_probabilities: Array of prediction probabilities.
    - config: Configuration dictionary.

    Returns:
    - str: The formatted label corresponding to the highest prediction probability.
    """

    # Index of the highest probability
    predicted_index = prediction_probabilities.argmax()

    # Label associated with the predicted index
    label = " ".join((config.get("CLASS_NAMES", [])[predicted_index]).split("_")).title()

    return label


# Making predictions on the validation dataset
mbnetv2_predictions = mbnetv2_model.predict(val_dataset_mbnetv2)
efnetb3_predictions = efnetb3_model.predict(val_dataset_efnetb3)


# Function to unbatch the data
def unbatchify(data):
    """
    Unbatches a dataset and extracts images and label names.

    Parameters:
    - data: A batched TensorFlow dataset containing image and label pairs.

    Returns:
    - Tuple of two lists:
        - images: List of unbatched image tensors.
        - labels: List of label strings.
    """

    # Lists to store the images and labels
    images = []
    labels = []

    # Looping through the images and labels
    for image, label in data.unbatch():

        # Appending the data
        images.append(image)
        labels.append(" ".join(config.get("CLASS_NAMES", [])[np.argmax(label)].split("_")).title())

    return images, labels


# Separating images and labels
images_mbnetv2, labels_mbnetv2 = unbatchify(val_dataset_mbnetv2)
images_efnetb3, labels_efnetb3 = unbatchify(val_dataset_efnetb3)


# Function to plot the image along with predicted and true labels
def plot_image_with_prediction(prediction_probs, images, true_labels, sample_index=7):
    """
    Plot a single image along with its predicted and true labels.

    Parameters:
    - prediction_probabilities: Array of prediction probabilities for all images.
    - images: List of image tensors.
    - true_labels: List of true class label strings.
    - sample_index: Index of the sample to plot.

    Returns: None
    """

    # Extracting prediction, image, and true label for the specified index
    sample_prediction = prediction_probs[sample_index]
    sample_image = images[sample_index]
    sample_true_label = true_labels[sample_index]

    # Getting predicted class label
    predicted_label = get_predicted_class_label(sample_prediction)

    # Normalising the sample image if pixel values are not in [0,1] range
    if np.max(sample_image) > 1:
        sample_image = sample_image/255.0

    # Setting title color
    title_color = "green" if predicted_label == sample_true_label else "red"

    # Plotting the image
    plt.imshow(sample_image)
    plt.axis("off")

    # Prediction confidence
    prediction_confidence = np.max(sample_prediction) * 100

    # Set the plot title
    plt.title(
        f"Predicted: {predicted_label} \nTrue: {sample_true_label} \nConfidence:{prediction_confidence:.2f}%",
        color = title_color,
        fontsize = 12
    )



# Function to plot the top 10 highest prediction probabilties along with the true label
def plot_top_prediction_confidences(prediction_probs, true_labels, n, sample_index=7):
    """
    Plots the top 10 predicted class probabilities for a given sample index, highlighting the true label in green.

    Parameters:
    - prediction_probs: Array of prediction probabilities (batch_size x num_classes).
    - true_labels: List or array of true class labels.
    - n: Number of top predictions to display.
    - sample_index: Index of the sample to plot.

    Returns: None
    """

    # Extracting prediction probabilities and true label for the specified sample
    sample_prediction = prediction_probs[sample_index]
    sample_true_label = true_labels[sample_index]

    # Getting indices of the top 10 predicted class probabilities
    top_n_indices = sample_prediction.argsort()[-n:][::-1]
    top_n_probabilities = sample_prediction[top_n_indices]
    top_n_class_names = [" ".join(label.split("_")).title() for label in config.get("CLASS_NAMES", [])[top_n_indices]]

    # Plotting the top 10 class probabilities as a bar chart
    bars = plt.bar(range(len(top_n_indices)), top_n_probabilities, color = "grey")
    plt.xticks(range(len(top_n_class_names)), labels = top_n_class_names, rotation = "vertical")

    # Highlighting the bar for the true label
    if sample_true_label in top_n_class_names:
        true_label_index = top_n_class_names.index(sample_true_label)
        bars[true_label_index].set_color("green")



# Function to display a grid of images with predicted labels and their top prediction confidence score.
def display_prediction_grid(n, prediction_probs, images, true_labels, start_index=0, num_rows=3, num_cols=2):
    """
    Displays a grid of images with their predicted labels and top prediction confidences.

    Parameters:
    - n: Number of top predictions to display.
    - prediction_probs: Array of prediction probabilities.
    - images: List of image tensors.
    - true_labels: List of true label strings.
    - start_index: Index to start displaying from.
    - num_rows: Number of rows in the grid.
    - num_cols: Number of columns in the grid.

    Returns: None
    """

    # Total number of images to display
    num_images = num_rows * num_cols

    # Setting the overall figure size
    plt.figure(figsize=(12 * num_cols, 5 * num_rows))

    # Looping through the images
    for i in range(num_images):
        sample_index = start_index + i

        # Plotting the image with its predicted and true label
        plt.subplot(num_rows, num_cols * 2, 2 * i + 1)
        plot_image_with_prediction(prediction_probs, images, true_labels, sample_index)

        # Plotting the bar chart showing top 10 prediction confidences
        plt.subplot(num_rows, num_cols * 2, 2 * i + 2)
        plot_top_prediction_confidences(prediction_probs, true_labels, n, sample_index)

    # Adjusting layout to prevent overlap
    plt.tight_layout()



# Showing prediction grid of MobileNetV2 model
display_prediction_grid(
    7,
    mbnetv2_predictions,
    images_mbnetv2,
    labels_mbnetv2,
    start_index = 0,
    num_rows = 3,
    num_cols = 3
)


# Showing prediction grid of EfficientNetB3 model
display_prediction_grid(
    7,
    efnetb3_predictions,
    images_efnetb3,
    labels_efnetb3,
    start_index = 0,
    num_rows = 3,
    num_cols = 3
)


# Function to save the trained model
def save_model_with_timestamp(model, suffix=None, base_dir=config.get("SAVE_MODEL_TO", "")):
    """
    Saves a trained model to a timestamped file in the specified directory.

    Parameters:
    - model: Trained model to save.
    - suffix: Optional suffix to add to the filename.
    - base_dir: Base directory to save the model.

    Returns:
    - str: Full filepath of the saved model.
    """

    # If base_dir is not specified
    if base_dir is None:
        base_dir = "/kaggle/working/"

    # Creating base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Creating filename with current timestamp and optional suffix
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"model-{timestamp}"

    # Adding suffix, if provided
    if suffix:
        filename += f"-{suffix}"

    # Creating model filepath
    filename += ".keras"
    model_filepath = os.path.join(base_dir , filename)

    # Saving the model
    print(f"Saving model to: {model_filepath}...")
    model.save(model_filepath)

    return model_filepath


# Function to load a saved Keras model
def load_model_from_filepath(model_filepath, mode = True):
    """
    Loads a trained Keras model from the given file path.

    Parameters:
    - model_filepath: Path to the saved model file.
    - mode: Specifies the 'safe_mode' parameter when loading the model.

    Returns:
        tf.keras.Model: Loaded Keras model.
    """
    print(f"Loading saved model from: {model_filepath}")
    model = tf.keras.models.load_model(model_filepath, safe_mode = mode)
    return model



# Creating full training and validation datasets
train_dataset_full, val_dataset_full = prepare_train_val_datasets(
    X = X,
    y = y,
    model_name = "EfficientNetB3"
)


# Training EfficientNetB3 model on full dataset
model, model_history = train_model(
    base_model = "EfficientNetB3",
    training_dataset = train_dataset_full,
    num_epochs = config.get("N_EPOCHS", 100),
    validation_dataset = val_dataset_full,
    callbacks = [early_stopping, reduce_lr]
)


# Making prediction
predictions = model.predict(val_dataset_full)


# Separating images and labels
images, labels = unbatchify(val_dataset_full)


# Showing prediction grid
display_prediction_grid(
    7,
    predictions,
    images,
    labels,
    start_index = 77,
    num_rows = 3,
    num_cols = 3
)


# Saving the fully trained model
suffix = "full-efficientnetb3-adam"
saved_model_filepath = save_model_with_timestamp(model, suffix = suffix)


# Loading saved model
loaded_model = load_model_from_filepath(saved_model_filepath, mode = True)


# Making prediction using the loaded model
loaded_model_predictions = loaded_model.predict(val_dataset_full)


# Showing first 6 predictions of EfficientNetB3 model
display_prediction_grid(
    7,
    efnetb3_predictions,
    images_efnetb3,
    labels_efnetb3,
    start_index = 0,
    num_rows = 3,
    num_cols = 2
)


# Path to the test images directory
test_dataset_path = os.path.join(config.get("DATA_DIR", ""), "test")

# List of all image file paths in the test directory
test_filenames = [os.path.join(test_dataset_path, fname) for fname in os.listdir(test_dataset_path)]
print("Sample test files:", *test_filenames[:5], sep = "\n")


# Creating test dataset
test_dataset = create_dataset(
    X = test_filenames,
    model_name = "efficientnet",
    config = config,
    dataset_type = "test",
    shuffle = False
)


# Displaying images from test dataset

# Setting figure size
plt.figure(figsize=(15, 15))

for img in test_dataset.take(1):

    # Looping through the images
    for i in range(25):
        
        # Plotting image in a 5x5 grid
        plt.subplot(5, 5, i + 1)
        plt.imshow(img[i] / 255.0)

        # Hiding axes and adjust layout
        plt.axis("off")
        plt.tight_layout()


# Making predictions on test data
test_predictions = loaded_model.predict(test_dataset, verbose=1)


# Extracting test image IDs
test_ids = [os.path.splitext(os.path.basename(path))[0] for path in test_filenames]
print("Sample test ids:", *test_ids[:5], sep = "\n")


# Creating a DataFrame with image IDs and prediction probabilities
preds_df = pd.DataFrame(test_predictions, columns = config.get("CLASS_NAMES", []))
preds_df.insert(0, "id", test_ids)
preds_df.head()


# Saving predictions to a CSV file 
output_csv_path = "/kaggle/working/submission.csv"
preds_df.to_csv(output_csv_path, index=False)
print(f"Submission file saved to: {output_csv_path}")




