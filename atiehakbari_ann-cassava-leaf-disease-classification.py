# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix





# Step 1: Load the dataset metadata
# The train.csv contains image_id and label
train_df = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')

# Add the full path to the images
train_df['image_path'] = '/kaggle/input/cassava-leaf-disease-classification/train_images/' + train_df['image_id']

# Convert label to string for categorical classification
train_df['label'] = train_df['label'].astype(str)

# Split the data into training and validation sets (80-20 split)
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])





# Step 2: Data Augmentation and Preprocessing
# Use ImageDataGenerator for data augmentation to improve generalization
# Rescale images, apply rotations, flips, etc.
IMG_SIZE = 224  # MobileNetV2 input size
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)





# Step 3: Build the Model using Transfer Learning
# Use MobileNetV2 as base model (pre-trained on ImageNet), suitable for mobile-quality images

# دانلود دستی و مسیر بده
weights_path = '/kaggle/input/mobilenet-v2-weights-h5/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'  # تغییر بده

base_model = MobileNetV2(weights=weights_path,include_top=False,input_shape=(224, 224, 3))

# Freeze the base model layers to prevent updating pre-trained weights initially
base_model.trainable = False

# Add custom layers on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),  # Batch Normalization to stabilize training
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(5, activation='softmax')  # 5 classes: 4 diseases + healthy
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()



# Step 4: Train the Model
# Use callbacks for early stopping and learning rate reduction

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
EPOCHS = 20  # You can adjust this based on performance

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Optional: Unfreeze some layers for fine-tuning
# After initial training, unfreeze the base model and fine-tune with lower learning rate
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune for a few more epochs
history_fine = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=10,  # Fewer epochs for fine-tuning
    callbacks=[early_stopping, reduce_lr]
)



# Step 5: Evaluate the Model
# Plot training history
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Generate predictions on validation set for confusion matrix
val_predictions = model.predict(val_generator)
val_pred_classes = np.argmax(val_predictions, axis=1)
val_true_classes = val_generator.classes

print(classification_report(val_true_classes, val_pred_classes))
print(confusion_matrix(val_true_classes, val_pred_classes))




# Step 6: Prepare Submission
# Load test images
test_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images/'
test_images = os.listdir(test_dir)

# Create a DataFrame for test images
test_df = pd.DataFrame(test_images, columns=['image_id'])
test_df['image_path'] = test_dir + test_df['image_id']

# Test data generator (no augmentation, just rescale)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

# Predict on test set
test_predictions = model.predict(test_generator)
test_pred_classes = np.argmax(test_predictions, axis=1)

# Create submission file
submission = pd.DataFrame({
    'image_id': test_df['image_id'],
    'label': test_pred_classes
})

submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


