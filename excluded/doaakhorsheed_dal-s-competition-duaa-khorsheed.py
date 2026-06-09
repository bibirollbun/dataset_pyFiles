import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50, InceptionV3, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import accuracy_score


# Define directories
data_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/train"
test_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/test"
labels_file = "/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv"
submission_file = "/kaggle/input/open-data-day-2025-dates-types-classification/sample_submission.csv"


# Load labels
df_labels = pd.read_csv(labels_file)
df_labels["label"] = df_labels["label"].astype(str)


# Get unique classes
classes = df_labels["label"].unique()
num_classes = len(classes)
class_mapping = {label: i for i, label in enumerate(classes)}


# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image


# Load images and labels
x_data = []
y_data = []

for idx, row in df_labels.iterrows():
    image_path = os.path.join(data_dir, row["filename"])
    if os.path.exists(image_path):
        x_data.append(preprocess_image(image_path))
        y_data.append(class_mapping[row["label"]])


# Convert to NumPy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)


x_data



y_data


# Split data
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)


# Display sample images
plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i])
    plt.title("Sample of training set")
    plt.axis("off")
plt.show()


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)


# Load dataset (Assuming x_data, y_data are already preprocessed)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

# Image augmentation for better generalization
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(x_train)


# Function to build models
def build_model(base_model):
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Define models
resnet_model = build_model(ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3)))
inception_model = build_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3)))
densenet_model = build_model(DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3)))


# Train and evaluate models
def train_and_evaluate(model, name):
    print(f"Training {name}...")
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=100, validation_data=(x_val, y_val))
    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"{name} Validation Accuracy: {val_acc:.4f}")
    return history

# Train models
hist_resnet = train_and_evaluate(resnet_model, "ResNet-50")
hist_inception = train_and_evaluate(inception_model, "InceptionV3")
hist_densenet = train_and_evaluate(densenet_model, "DenseNet")



import matplotlib.pyplot as plt

# Function to plot training history
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Call the function for each model after training
plot_training_history(hist_resnet, "ResNet-50")
plot_training_history(hist_inception, "InceptionV3")
plot_training_history(hist_densenet, "DenseNet")



# Train best model 
history = densenet_model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_val, y_val), epochs=10)

# Evaluate on validation data
y_pred = np.argmax(resnet_model.predict(x_val), axis=1)
y_true = np.argmax(y_val, axis=1)



import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train best model
history = densenet_model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_val, y_val), epochs=10)

# Evaluate on validation data using DenseNet model
y_pred = np.argmax(densenet_model.predict(x_val), axis=1)  # Get predicted labels
y_true = np.argmax(y_val, axis=1)  # Get true labels

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Load test images
def load_test_images(test_dir):
    test_images = []
    filenames = []
    for file in os.listdir(test_dir):
        image_path = os.path.join(test_dir, file)
        if os.path.isfile(image_path):
            test_images.append(preprocess_image(image_path))
            filenames.append(file)
    return np.array(test_images), filenames

test_images, filenames = load_test_images(test_dir)


# Predict on test images
test_preds = np.argmax(densenet_model.predict(test_images), axis=1)
test_labels = [classes[i] for i in test_preds]



# Number of images to display
num_images = 10
plt.figure(figsize=(12, 6))

for i in range(num_images):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i])  # Assuming test_images is preprocessed and normalized
    plt.axis("off")
    plt.title(f"Predicted: {test_labels[i]}", fontsize=10)

plt.tight_layout()
plt.show()


# Create submission file
submission_df = pd.DataFrame({"filename": filenames, "label": test_labels})
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
submission_df

