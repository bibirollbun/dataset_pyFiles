path="/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"


import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from tensorflow.keras.applications import VGG16,ResNet50,MobileNet, DenseNet201, InceptionV3,VGG19, NASNetLarge, InceptionResNetV2, NASNetMobile ,EfficientNetB0
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
# Changed import: ImageDataGenerator is now in tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools

%matplotlib inline


import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Example paths (change as needed)

train_dir = os.path.join(path, "train")
test_dir = os.path.join(path, "test")
csv_path = os.path.join(path, "train_labels.csv")
# Load CSV
df = pd.read_csv(csv_path)

# Preview the data
print(df.head())





import matplotlib.pyplot as plt

label_column_name = 'label'

if label_column_name in df.columns:
    label_distribution = df[label_column_name].value_counts()
    print("Distribution of labels:")
    print(label_distribution)

    # You can also
    plt.figure(figsize=(8, 6))
    label_distribution.plot(kind='bar')
    plt.title('Distribution of Labels')
    plt.xlabel(label_column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()
else:
    print(f"Column '{label_column_name}' not found in the DataFrame.")





print("Checking for null or NaN values:")
print(df.isnull().sum())


# Read one train image
first_train_img = df['filename'].iloc[0]
img_path = os.path.join(train_dir, first_train_img)

# Open and display the image
img = Image.open(img_path)
plt.imshow(img)
plt.title(f"Label: {df['label'].iloc[0]}")
plt.axis('off')
plt.show()

# Optionally read a test image
test_images = sorted(os.listdir(test_dir))
test_img_path = os.path.join(test_dir, test_images[0])
img_test = Image.open(test_img_path)
plt.imshow(img_test)
plt.title("Test Image Example")
plt.axis('off')
plt.show()




import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
for i in range(10):
    image_name = df['filename'].iloc[i]
    image_path = os.path.join(train_dir, image_name)
    img = Image.open(image_path)
    label = df['label'].iloc[i]
    plt.subplot(5, 2, i + 1)
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()



train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))

# Display the first few rows of the training and validation sets
print("\nTraining set head:")
print(train_df.head())

print("\nValidation set head:")
print(val_df.head())

# Check the distribution of labels in the split sets
print("\nLabel distribution in Training set:")
print(train_df['label'].value_counts().sort_index())

print("\nLabel distribution in Validation set:")
print(val_df['label'].value_counts().sort_index())



print("\nDistribution of classes in training set:")
print(train_df['label'].value_counts().sort_index())

print("\nDistribution of classes in validation set:")
print(val_df['label'].value_counts().sort_index())



print("\nNumber of images in training set:", len(train_df))
print("Number of images in validation set:", len(val_df))



import matplotlib.pyplot as plt
IMG_SIZE = 224 # Or the size appropriate for your chosen model
BATCH_SIZE = 32 # Or adjust as needed

train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,       # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,   # Randomly shift width by up to 20%
    height_shift_range=0.2,  # Randomly shift height by up to 20%
    shear_range=0.15,        # Apply shearing transformations
    zoom_range=0.2,          # Apply zooming
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Fill points outside the boundaries of the input
)

val_datagen = ImageDataGenerator(rescale=1./255) # Only rescale for validation

# Create generators from DataFrames
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Or 'binary' depending on your number of classes
    subset='training' # Specify 'training' as this is the training generator
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir, # Validation images are also in the training directory
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Specify 'validation'
)


print("\nTraining Generator:")
print("Found", train_generator.samples, "images belonging to", len(train_generator.class_indices), "classes.")

print("\nValidation Generator:")
print("Found", validation_generator.samples, "images belonging to", len(validation_generator.class_indices), "classes.")



validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir, # Validation images are also in the training directory
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Ensure this matches your number of classes and label format
)




print(f"\nTotal images in train set: {train_generator.samples}")
print(f"Total images in validation set: {validation_generator.samples}")


# prompt: total image befor augmentation

# Count total images before augmentation
total_train_images = len(train_df)
total_val_images = len(val_df)
print(f"\nTotal images in the original training set before augmentation: {total_train_images}")
print(f"Total images in the original validation set before augmentation: {total_val_images}")
print(f"Total images for training data generator before augmentation: {train_generator.samples}")
print(f"Total images for validation data generator before augmentation: {validation_generator.samples}")




import matplotlib.pyplot as plt
import numpy as np

augmented_batch = next(train_generator)
images, labels = augmented_batch

# Plot a few augmented images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])

    plt.title(f"Augmented Image")
    plt.axis('off')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import random
import os

# Ø£Ø®Ø° ØµÙˆØ±Ø© Ø¹Ø´ÙˆØ§Ø¦ÙŠØ© Ù…Ù† Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„ØªØ¯Ø±ÙŠØ¨
random_row = train_df.sample(n=1).iloc[0]
filename = random_row['filename']
label = random_row['label']
image_path = os.path.join(train_dir, filename)

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„ØµÙˆØ±Ø© Ø§Ù„Ø£ØµÙ„ÙŠØ© ÙˆØªØ¹Ø¯ÙŠÙ„ Ø§Ù„Ø­Ø¬Ù…
original_img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(original_img)
img_array = np.expand_dims(img_array, axis=0)  # (1, h, w, 3)

# ØªÙˆÙ„ÙŠØ¯ Ù†Ø³Ø®Ø© Ù…Ø¹Ø¯Ù„Ø© (Augmented)
augmented_iter = train_datagen.flow(img_array, batch_size=1, shuffle=False)
augmented_img = next(augmented_iter)[0]  # Ù„Ø§ ØªÙ‚Ø³Ù‘Ù…Ù‡Ø§ Ù…Ø±Ø© Ø£Ø®Ø±Ù‰

# Ø¹Ø±Ø¶ Ø§Ù„ØµÙˆØ±ØªÙŠÙ† Ø¬Ù†Ø¨Ù‹Ø§ Ù„Ø¬Ù†Ø¨
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title(f"ğŸŸ¦ Original ({label})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(augmented_img.clip(0, 1))  # Ù„Ø¶Ù…Ø§Ù† Ø¹Ø±Ø¶ ØµØ­ÙŠØ­
plt.title("ğŸ› ï¸� Augmented")
plt.axis('off')

plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import numpy as np

class_indices = train_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])

    # Decode the one-hot encoded label
    label_index = np.argmax(labels[i])
    label_name = idx_to_class[label_index]

    plt.title(f"Augmented Image\nLabel: {label_name}")
    plt.axis('off')
plt.tight_layout()
plt.show()



print(f"Count of images in train set: {train_generator.samples}")
print(f"Count of images in validation set: {validation_generator.samples}")



def build_model(img_height, img_width, num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False # Freeze the base model initially

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax') # Use softmax for multi-class classification
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', # Use categorical_crossentropy for multi-class
                  metrics=['accuracy'])

    return model

num_classes = len(train_generator.class_indices) # Get the number of classes from the generator
efficientnet_model = build_model(224, 224, num_classes)
efficientnet_model.summary()




checkpoint = ModelCheckpoint('InceptionV3_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00001, mode='max', verbose=1)

callbacks_list = [checkpoint, reduce_lr] # Add tensorboard_callback if you set up log_dir

EPOCHS = 20

history = efficientnet_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks_list
)




import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


predict_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col='filename',
    y_col='label', # Keep y_col to easily get true labels later
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


true_labels_df = pd.get_dummies(val_df['label'])
# Get the class labels (the column names of the one-hot encoded DataFrame)
class_labels = true_labels_df.columns.tolist()
# Convert to numpy array
true_labels = true_labels_df.values

# Method 2: Get true labels and predictions from the generator (more robust)
# Predict method will yield predictions in the same order as the generator
predictions = efficientnet_model.predict(predict_generator, steps=len(predict_generator))


predict_generator.reset()


all_true_labels = []
i = 0

while i < len(predict_generator):
    batch_images, batch_labels = next(predict_generator)
    all_true_labels.extend(batch_labels)
    i += 1

# Convert the list of batches to a numpy array
all_true_labels = np.array(all_true_labels)

# Now, predictions and all_true_labels should be in the same order.

# Convert predictions from probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# Convert true labels from one-hot encoding to class indices
true_classes = np.argmax(all_true_labels, axis=1)

# Get the class names from the generator's class_indices
class_names = list(predict_generator.class_indices.keys())

# Generate the classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names)

print("\nClassification Report:")
print(report)

# Optional: Display confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



overall_accuracy = accuracy_score(true_classes, predicted_classes)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")



import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

img_height, img_width = 224, 224

index_to_label = {v: k for k, v in class_indices.items()}


test_filenames = sorted(os.listdir(test_dir))
test_images = []

for fname in test_filenames:
    path = os.path.join(test_dir, fname)
    img = Image.open(path).resize((img_height, img_width)).convert('RGB')
    img_array = img_to_array(img) / 255.0
    test_images.append(img_array)

test_images = np.array(test_images)

pred_probs = efficientnet_model.predict(test_images, verbose=1)
pred_indices = np.argmax(pred_probs, axis=1)
pred_labels = [index_to_label[i] for i in pred_indices]


submission_df = pd.DataFrame({
    'filename': test_filenames,
    'label': pred_labels
})
submission_df.to_csv('submission.csv', index=False)

print("âœ… ØªÙ…Øª Ø§Ù„ØªÙ†Ø¨Ø¤Ø§ØªØŒ ÙˆØªÙ… Ø¥Ù†Ø´Ø§Ø¡ Ø§Ù„Ù…Ù„Ù� submission.csv Ø¨Ù†Ø¬Ø§Ø­.")



from collections import defaultdict

grouped = defaultdict(list)
for fname, label in zip(test_filenames, pred_labels):
    grouped[label].append(fname)


for label, files in grouped.items():
    print(f"\n Predicted label: {label}")
    for fname in files[:3]:
        path = os.path.join(test_dir, fname)
        img = Image.open(path)
        plt.imshow(img)
        plt.title(f"{label} - {fname}")
        plt.axis('off')
        plt.show()





print(submission_df.head())
submission_df.columns



print(f"\nTotal number of rows in submission.csv: {len(submission_df)}")

