# Required libraries
import os
import random
import numpy as np
import cv2
import zipfile
import matplotlib.pyplot as plt
import pandas as pd


from imutils import paths
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Define paths
zip_train_path = "/kaggle/input/dogs-vs-cats/train.zip"
zip_test_path = "/kaggle/input/dogs-vs-cats/test1.zip"
extract_train_path = "/kaggle/working/train"
extract_test_path = "/kaggle/working/test"

# Extract training data
with zipfile.ZipFile(zip_train_path, 'r') as zip_ref:
    zip_ref.extractall(extract_train_path)

# Extract test data
with zipfile.ZipFile(zip_test_path, 'r') as zip_ref:
    zip_ref.extractall(extract_test_path)

print("Train and test zip files extracted successfully.")


# Data loading function
class_labels = {'cat': 0, 'dog': 1}  # Map class names to numeric labels

def load_dataset(path, target_size=(100, 100), max_samples=None):
    """
    Load images from the given directory, preprocess them, and return image arrays with labels.

    Parameters:
        path (str): Directory containing the images
        target_size (tuple): Desired image size (width, height) after resizing
        max_samples (int or None): Maximum number of samples to load

    Returns:
        x (np.array): Array of preprocessed images normalized to [0, 1]
        labels (np.array): Corresponding numeric labels
    """
    x = []           # List to store images
    labels = []      # List to store labels

    # Get all image file paths and shuffle them randomly
    image_paths = list(paths.list_images(path))
    random.shuffle(image_paths)
    
    # Limit number of samples to 2000 for faster processing
    image_paths = image_paths[:2000]
    if max_samples:
        image_paths = image_paths[:max_samples]

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: {image_path} could not be read, skipping.")
            continue

        img = cv2.resize(img, target_size)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x.append(img)

        # Determine label from filename prefix
        file_name = os.path.basename(image_path)
        if file_name.startswith('cat'):
            label_name = 'cat'
        elif file_name.startswith('dog'):
            label_name = 'dog'
        else:
            print(f"Warning: Unknown file format {file_name}, skipping.")
            continue

        labels.append(class_labels[label_name])

    # Convert lists to numpy arrays and normalize pixel values to range [0, 1]
    x = np.array(x, dtype="float32") / 255.0
    labels = np.array(labels)

    print(f"Loaded data shape: {x.shape}, Number of labels: {len(labels)}")
    return x, labels


def visualize_img(image_batch, label_batch, class_labels=class_labels):
    """
    Displays a sample of images with their corresponding labels.
    
    Parameters:
        image_batch (array-like): Batch of images (e.g., x_train)
        label_batch (array-like): Corresponding labels (e.g., y_train)
        class_labels (dict): Dictionary mapping class names to numeric labels
    """

    plt.figure(figsize=(12, 6))  # Set the figure size
    class_names = list(class_labels.keys())  # Get class names (e.g., ['cat', 'dog'])

    # Select 10 random indices (or less if fewer images are available)
    sample_size = min(10, len(image_batch))
    idxs = np.random.choice(len(image_batch), size=sample_size, replace=False)

    for i, idx in enumerate(idxs):
        ax = plt.subplot(2, 5, i + 1)  # Arrange in 2 rows and 5 columns
        plt.imshow(image_batch[idx])  # Show the image
        plt.title(class_names[label_batch[idx]].title())  # Display the label
        plt.axis('off')  # Hide axis ticks

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot

# Load the dataset
x_train, y_train = load_dataset(extract_train_path)

# Visualize the dataset
visualize_img(x_train, y_train)



visualize_img(x_train, y_train)


X, y = load_dataset(extract_train_path, max_samples=2000)


#  Load test dataset (without labels)
test_path = os.path.join(extract_test_path, "test1")
test_image_paths = sorted(list(paths.list_images(test_path)), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
X_test = []

# Load and preprocess each test image
for img_path in test_image_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_test.append(img)

X_test = np.array(X_test, dtype="float32") / 255.0


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Build the model
IMG_SIZE = 100

#Define a simple Convolutional Neural Network using the Sequential API
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # First convolutional layer
    MaxPooling2D(2, 2),  # First max pooling layer

    Conv2D(16, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D(2, 2),  # Second max pooling layer

    Flatten(),  # Flatten the 3D feature maps to 1D
    Dense(64, activation='relu'),  # Fully connected layer with 64 units
    Dropout(0.5),  # Dropout layer to prevent overfitting

    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model with appropriate loss function and optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,
    verbose=2
)


# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()# Training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))


# Training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))


def predict_image(image_path, model, target_size=(100, 100)):
    """
    Load an image, preprocess it, and predict whether it's a cat or dog using the trained model.
    
    Parameters:
        image_path (str): Path to the image file.
        model (keras.Model): Trained CNN model for prediction.
        target_size (tuple): Desired image size for the model input.
    """

    # Load the image and resize it to the target size
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, IMG_SIZE, IMG_SIZE, 3)

    # Make prediction using the model
    prediction = model.predict(img_array)[0][0]

    # Classification threshold set at 0.75
    label = "Dog" if prediction > 0.75 else "Cat"
    confidence = prediction if prediction > 0.75 else 1 - prediction

    # Print the prediction and confidence percentage
    print(f"Prediction: {label} ({confidence * 100:.2f}%)")

    # Display the image with the predicted label as the title
    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis('off')
    plt.show()


# Predict probabilities on test set
predictions = model.predict(X_test)


# Convert probabilities to binary labels (0 = cat, 1 = dog)
labels = (predictions > 0.5).astype(int).flatten()

# Extract image IDs from filenames
image_ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in test_image_paths]

# Prepare submission DataFrame
submission = pd.DataFrame({
    "id": image_ids,
    "label": labels
})

# Sort by ID
submission = submission.sort_values(by="id")

# Save to CSV
submission.to_csv("submission.csv", index=False)
print(" submission.csv file created successfully!")

