from google.colab import files
files.upload()  # Upload kaggle.json



!mv "kaggle (1).json" kaggle.json 2>/dev/null
!mv "kaggle .json" kaggle.json 2>/dev/null



import re
import numpy as np
from matplotlib import pyplot as plt


!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json



!kaggle competitions download -c open-data-day-2025-dates-types-classification



#Extract the Data
!unzip -q open-data-day-2025-dates-types-classification.zip -d /content/dataset




import os

dataset_path = "/content/dataset"  # Change this if you extracted elsewhere
for root, dirs, files in os.walk(dataset_path):
    print(root)
    for file in files[:5]:  # Show only the first 5 files per folder
        print("  ├──", file)



import matplotlib.pyplot as plt
import cv2
import random

# Set path to images (modify if needed)
image_folder = "/content/dataset/train"  # Change this based on your structure

# Get a random image
image_files = os.listdir(image_folder)
sample_image = random.choice(image_files)

# Read and display the image
img = cv2.imread(os.path.join(image_folder, sample_image))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors

plt.imshow(img)
plt.title(sample_image)
plt.axis("off")
plt.show()



import pandas as pd

df_labels = "/content/dataset/train_labels.csv"
df = pd.read_csv(df_labels)
print(df.head())  # Show first few rows



df = pd.read_csv("/content/dataset/sample_submission.csv")
df.head()


import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Path to the CSV file with labels
labels_path = '/content/dataset/train_labels.csv'

# Load the CSV file
df_labels = pd.read_csv(labels_path)

# Path to the images directory
train_dir = '/content/dataset/train'  # Adjust this as needed

# Initialize lists for images and labels
images = []
labels = []

# Loop through the DataFrame to load images and labels
for _, row in df_labels.iterrows():
    img_path = os.path.join(train_dir, row['filename'])

    # Open the image, convert it to RGB (to ensure all images have 3 channels), resize it to 224x224
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img = np.array(img)

    # Normalize the image (rescale pixel values to [0, 1])
    img = img / 255.0

    # Append the image to the images list
    images.append(img)

    # Append the label to the labels list
    labels.append(row['label'])

# Convert lists to numpy arrays
X_train = np.array(images)
y_train = np.array(labels)

# Check the shape of the data
print(X_train.shape)



from sklearn.model_selection import train_test_split

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Encode labels into integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# One-hot encode labels if multi-class classification
num_classes = len(label_encoder.classes_)

if num_classes > 2:
    y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
else:
    y_train_one_hot = y_train_encoded.reshape(-1, 1)  # Keep shape (num_samples, 1) for binary classification

# Print to verify
print("X_train shape:", X_train.shape)  # Expected: (num_samples, 224, 224, 3)
print("y_train_one_hot shape:", y_train_one_hot.shape)  # (num_samples, num_classes) if multi-class


from sklearn.preprocessing import LabelEncoder

# Encode labels (e.g., 'Ajwa', 'Madjool', etc.)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train_encoded)

# Check the shape of labels
print(y_train_one_hot.shape)  # Should print (num_images, num_classes)



import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the pre-trained VGG16 model without the top layer
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers to prevent them from being trained
for layer in vgg16_base.layers:
    layer.trainable = False

# Build the model with the VGG16 base + custom classifier
model = Sequential([
    vgg16_base,  # Add the VGG16 base (convolutional layers)
    Flatten(),   # Flatten the output from the VGG16 base
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(512, activation='relu'),  # Fully connected layer
    Dense(7, activation='softmax')  # Output layer for 7 classes (softmax for multi-class classification)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),  # Learning rate can be adjusted
              loss='categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])



model.summary()



from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the training labels
y_train_encoded = label_encoder.fit_transform(y_train)

# Convert integer labels to one-hot encoding
num_classes = len(label_encoder.classes_)  # Get number of unique classes
y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)

# Print class mapping
print("Class Mapping:", dict(enumerate(label_encoder.classes_)))




# Early stopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)



# Convert y_val to class labels if they are in one-hot encoding
if len(y_val.shape) > 1 and y_val.shape[1] > 1:
    y_val = np.argmax(y_val, axis=1)

# Apply label encoding to the validation labels
y_val_encoded = label_encoder.transform(y_val)  # Use transform() to maintain consistent mapping

# One-hot encode the validation labels
y_val_one_hot = to_categorical(y_val_encoded, num_classes=7)


# Train the model
history = model.fit(X_train, y_train_one_hot, epochs=20, validation_data=(X_val, y_val_one_hot), callbacks=[early_stopping])



# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val_one_hot)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")


import matplotlib.pyplot as plt

# Plot training accuracy
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()




# Save the model to an H5 file
model.save('/content/my_model.h5')
print("Model saved successfully!")


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model
model = load_model('/content/my_model.h5')

# Path to the test dataset
test_dir = '/content/dataset/test'
test_files = os.listdir(test_dir)

# Initialize lists for filenames and predictions
filenames = []
predictions = []

# Process each test image
for file in test_files:
    if file.endswith('.jpg') or file.endswith('.png'):
        img_path = os.path.join(test_dir, file)

        # Load, resize, and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Predict the class using the model
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Map to class labels
        class_labels = ['Ajwa', 'Medjool', 'Nabtat Ali', 'Sokari', 'Sukkary', 'Barhi', 'Khalas']
        predicted_label = class_labels[predicted_class_index]

        filenames.append(file)
        predictions.append(predicted_label)

# Create DataFrame for submission
submission_df = pd.DataFrame({
    'filename': filenames,
    'label': predictions
})

# Save the DataFrame to CSV
submission_df.to_csv('/content/dataset/sample_submission.csv', index=False)



submission_df

