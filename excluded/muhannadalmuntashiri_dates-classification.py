import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os

# Load labels
labels_df = pd.read_csv('/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv')
print(labels_df.head())

# Define image processing function
def load_and_process_images(df, image_folder):
    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        with Image.open(img_path) as img:
            img = img.convert('RGB').resize((224, 224))
            images.append(np.array(img))
            labels.append(row['label'])
    return np.array(images), np.array(labels)

# Load and split data
train_image_folder = '/kaggle/input/open-data-day-2025-dates-types-classification/train/'  # Update path if needed
images, labels = load_and_process_images(labels_df, train_image_folder)
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)



from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(np.unique(labels)), activation='softmax')(x)

# Compile model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

# Convert labels to one-hot
train_labels_one_hot = to_categorical(train_labels_encoded)
val_labels_one_hot = to_categorical(val_labels_encoded)

# Train model
history = model.fit(train_images, train_labels_one_hot, epochs=10, validation_data=(val_images, val_labels_one_hot))





import os
from PIL import Image
import numpy as np

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            images.append(np.array(img))
            filenames.append(filename)
    return np.array(images), filenames

    

# Load test images
test_images, image_filenames = load_images_from_folder('/kaggle/input/open-data-day-2025-dates-types-classification/test')

# Predict labels
predicted_labels = model.predict(test_images)
predicted_labels = np.argmax(predicted_labels, axis=1)
predicted_label_names = label_encoder.inverse_transform(predicted_labels)

# Create and save submission file
submission_df = pd.DataFrame({'filename': image_filenames, 'label': predicted_label_names})
submission_file_path = 'submission.csv'
submission_df.to_csv(submission_file_path, index=False)
print("Submission file saved successfully at:", submission_file_path)



import matplotlib.pyplot as plt

# Select random images to display
num_images = 9
indices = np.random.choice(range(len(test_images)), num_images, replace=False)

# Set up the plot
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
axes = axes.flatten()

for i, idx in enumerate(indices):
    ax = axes[i]
    img = test_images[idx].astype('uint8')
    label = predicted_label_names[idx]

    ax.imshow(img)
    ax.set_title(f"Predicted: {label}")
    ax.axis('off')

plt.tight_layout()
plt.show()




# Display accuracy and loss plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


