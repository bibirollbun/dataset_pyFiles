import os
import random
import sys
import cv2
import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Recall
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Database
train_dir = "/kaggle/input/breast-cancer-detection/train"
test_dir = "/kaggle/input/breast-cancer-detection/test"
val_dir = "/kaggle/input/breast-cancer-detection/valid"


# Train data preparation
classes = ["0", "1"]
train_dataset = []
for class_label in classes:
    class_path = os.path.join(train_dir, class_label)
    label_index = classes.index(class_label)
    for img_file in tqdm.tqdm(os.listdir(class_path)):
        img_full_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_full_path)
        img = cv2.resize(img, (224, 224))
        train_dataset.append([img, label_index])


# Test data preparation
test_dataset = []
for class_label in classes:
    class_path = os.path.join(test_dir, class_label)
    label_index = classes.index(class_label)
    for img_file in tqdm.tqdm(os.listdir(class_path)):
        img_full_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_full_path)
        img = cv2.resize(img, (224, 224))
        test_dataset.append([img, label_index])

# Validation data Preparation
val_dataset = []
for class_label in classes:
    class_path = os.path.join(val_dir, class_label)
    label_index = classes.index(class_label)
    for img_file in tqdm.tqdm(os.listdir(class_path)):
        img_full_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_full_path)
        img = cv2.resize(img, (224, 224))
        val_dataset.append([img, label_index])


# Validation data Preparation
val_dataset = []
for class_label in classes:
    class_path = os.path.join(val_dir, class_label)  
    label_index = classes.index(class_label)
    for img_file in tqdm.tqdm(os.listdir(class_path)):
        img_full_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_full_path)
        img = cv2.resize(img, (224, 224))
        val_dataset.append([img, label_index])


# Data shuffling
random.shuffle(train_dataset)
random.shuffle(test_dataset)
random.shuffle(val_dataset)


# List to numpy array conversion
# Train data
X_train_images = []
y_train_labels = []
for img, label in train_dataset:
    X_train_images.append(img)
    y_train_labels.append(label)

X_train = np.array(X_train_images)/255
y_train = np.array(y_train_labels)

# Test data
X_test_images = []
y_test_labels = []
for img, label in test_dataset:
    X_test_images.append(img)
    y_test_labels.append(label)

X_test = np.array(X_test_images)/255
y_test = np.array(y_test_labels)

# Validation data
X_val_images = []
y_val_labels = []
for img, label in val_dataset:
    X_val_images.append(img)
    y_val_labels.append(label)

X_val = np.array(X_val_images)/255
y_val = np.array(y_val_labels)


# Load a sample image (replace with the actual path of an image in your dataset)
# Update to the correct image path
sample_image_path = '/kaggle/input/breast-cancer-detection/train/0/105_1232990271_png.rf.0d15468a4bec2ad2147f0616b6821681.jpg'

try:
    sample_image = Image.open(sample_image_path).resize((640, 640))
    plt.imshow(sample_image)
    plt.axis('off')  # Turn off the axes for a cleaner visualization
    plt.show()
except FileNotFoundError:
    print("Please replace 'sample_image_path.jpg' with the correct image path on your system.")

# Load a sample image (replace with the actual path of an image in your dataset)
sample_image_path = '/kaggle/input/breast-cancer-detection/train/0/106_1160585918_png.rf.ddf71d5395cefb701d0b9ec2b0d81176.jpg'  # Update to the correct image path

try:
    sample_image = Image.open(sample_image_path).resize((640, 640))
    plt.imshow(sample_image)
    plt.axis('off')  # Turn off the axes for a cleaner visualization
    plt.show()
except FileNotFoundError:
    print("Please replace 'sample_image_path.jpg' with the correct image path on your system.")


# Sample Visualization
num_samples = 6
random_samples = np.random.choice(len(X_train), num_samples, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i, idx in enumerate(random_samples):
    ax = axes[i // 3, i % 3]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f"Label: {y_train[idx]}")
    ax.axis('off')

plt.tight_layout()
plt.show()


# Initialize a Sequential model
model = Sequential()

# First convolutional layer with 100 filters of size 3x3, using ReLU activation
# The input shape is set to 224x224 pixels with 3 color channels (RGB)
model.add(Conv2D(100, (3, 3), activation="relu", input_shape=(224, 224, 3)))

# First max pooling layer to reduce spatial dimensions (down-sampling by half)
model.add(MaxPooling2D(2, 2))

# Second convolutional layer with 100 filters of size 3x3, using ReLU activation
model.add(Conv2D(100, (3, 3), activation="relu"))

# Second max pooling layer for further down-sampling
model.add(MaxPooling2D(2, 2))

# Third convolutional layer with 64 filters of size 3x3, using ReLU activation
model.add(Conv2D(64, (3, 3), activation="relu"))

# Fourth convolutional layer, also with 64 filters of size 3x3, using ReLU activation
model.add(Conv2D(64, (3, 3), activation="relu"))

# Third max pooling layer to further reduce spatial dimensions
model.add(MaxPooling2D(2, 2))

# Flatten the feature maps into a 1D vector for the fully connected layers
model.add(Flatten())

# First fully connected layer with 64 units and ReLU activation
model.add(Dense(64, activation="relu"))

# Dropout layer to randomly set 20% of the inputs to zero, preventing overfitting
model.add(Dropout(.2))

# Second fully connected layer with 32 units and ReLU activation
model.add(Dense(32, activation="relu"))

# Dropout layer to randomly set 30% of the inputs to zero, for further regularization
model.add(Dropout(.3))

# Third fully connected layer with 32 units and ReLU activation
model.add(Dense(32, activation="relu"))

# Output layer with 1 unit and sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model with Adam optimizer and binary cross-entropy loss for binary classification
# Use accuracy as the evaluation metric
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary for an overview of the layers and parameters
model.summary()


# Training the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), verbose=1)


# Graphs and Metrics
plt.figure(figsize=(30.5, 10))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Confusion Matrix
threshold = 0.5
y_pred = (model.predict(X_test) >= threshold).astype(int)

model.evaluate(X_test, y_test)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with custom labels
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-malignant', 'Malignant cancer'])
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Non-malignant', 'Malignant cancer'])

# Print the classification report
print("Classification Report:")
print(report)


from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a DataFrame to store the accuracy
accuracy_df = pd.DataFrame({'Metric': ['Accuracy'],
                            'Score': [accuracy]})

# Display the DataFrame
accuracy_df


# Save the model after training
model.save('/kaggle/working/breast_cancer_model.h5')


# Assuming X_test and y_test are your test data and labels
predictions = model.predict(X_test)

# Convert probabilities to binary predictions (0 or 1)
predictions = (predictions > 0.5).astype(int)  

# Separate indices of predicted classes
class_0_indices = np.where(predictions.flatten() == 0)[0]
class_1_indices = np.where(predictions.flatten() == 1)[0]

# Display up to 5 examples for each class
num_examples = 5
fig, axes = plt.subplots(2, num_examples, figsize=(20, 6))

# Plot class 0 predictions
for i, idx in enumerate(class_0_indices[:num_examples]):
    axes[0, i].imshow(X_test[idx])
    axes[0, i].set_title(f"Pred: 0 | Actual: {y_test[idx]}")
    axes[0, i].axis('off')

# Plot class 1 predictions
for i, idx in enumerate(class_1_indices[:num_examples]):
    axes[1, i].imshow(X_test[idx])
    axes[1, i].set_title(f"Pred: 1 | Actual: {y_test[idx]}")
    axes[1, i].axis('off')

# Set row titles
axes[0, 0].set_ylabel("Class 0")
axes[1, 0].set_ylabel("Class 1")

plt.suptitle("CNN Model Predictions for Classes 0 and 1")
plt.show()


import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import glob


# Load the trained model
model = load_model('/kaggle/working/breast_cancer_model.h5')

# Path to the image to predict
img_path = '/kaggle/input/breast-cancer-detection/test/0/1890_712521570_png.rf.6fca80e8b87142cc12b129e85e664eb7.jpg'

# Test loading the image
img_test = cv2.imread(img_path)


# Check if the image loaded correctly
if img_test is None:
    print("Error loading image. Check the path.")
else:
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Prediction and display function with a rectangle
def predict_and_display(model, img_path, threshold=0.5):
    # Load the image using OpenCV
    img_cv = cv2.imread(img_path)

    # Make a copy for display
    img_copy = img_cv.copy()

    # Resize image to 224x224 as required by the CNN model
    img_resized = cv2.resize(img_cv, (224, 224))

    # Preprocess the image for the model
    img_array = image.img_to_array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(img_array)

    # Define the label and color based on the prediction
    if prediction[0] >= threshold:
        label = "Cancer: Yes"
        color = (0, 0, 255)  # Red for cancer
    else:
        label = "Cancer: No"
        color = (0, 255, 0)  # Green for no cancer

    # Draw a rectangle around the original image
    height, width, _ = img_copy.shape
    padding = 100  # Adjust for square size
    start_point = (padding, padding)
    end_point = (width - padding, height - padding)

    # Draw the rectangle
    img_with_rectangle = cv2.rectangle(img_copy, start_point, end_point, color, 5)

    # Put the label text on the image
    img_with_text = cv2.putText(img_with_rectangle, label, (50, height - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB)
    
    # Display the image with matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Predict and display
predict_and_display(model, img_path)


# Load the trained model
model = load_model('/kaggle/working/breast_cancer_model.h5')

# Define the directory containing the validation images
img_dir = '/kaggle/input/breast-cancer-detection/valid'


# Function to predict and display each image with bounding box and label
def predict_and_display(model, img_path, threshold=0.5):
    # Load the image using OpenCV
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"Error loading image: {img_path}")
        return

    # Make a copy of the original image for modification
    img_copy = img_cv.copy()

    # Resize the image to 224x224 pixels as required by the CNN model
    img_resized = cv2.resize(img_cv, (224, 224))

    # Preprocess the image for the model
    img_array = image.img_to_array(img_resized) / 255.0  # Normalize the image (scale pixel values to [0,1])
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension for the model input

    # Make the prediction using the CNN model
    prediction = model.predict(img_array)

    # Define the label and color based on the prediction
    if prediction[0] >= threshold:
        label = "Cancer: Yes"
        color = (0, 0, 255)  # Red color for positive prediction (Cancer: Yes)
    else:
        label = "Cancer: No"
        color = (0, 255, 0)  # Green color for negative prediction (Cancer: No)

    # Draw a bounding box around the original image
    height, width, _ = img_copy.shape
    padding = 50  # Adjust the padding for the box size
    start_point = (padding, padding)  # Start point of the bounding box
    end_point = (width - padding, height - padding)  # End point of the bounding box

    # Draw the rectangle and label on the image
    img_with_rectangle = cv2.rectangle(img_copy, start_point, end_point, color, 5)  # Draw rectangle
    img_with_text = cv2.putText(img_with_rectangle, label, (50, height - 60),  # Put label text
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format) for display
    img_rgb = cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB)

    # Display the image with Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis('off')  # Turn off axes for a clean display
    plt.show()


# Loop through each image in the validation directory and apply prediction
for img_path in glob.glob(os.path.join(img_dir, '*/*.jpg')):  # Adjust file extension if needed (e.g., .png)
    print(f"Processing: {img_path}")
    predict_and_display(model, img_path)




