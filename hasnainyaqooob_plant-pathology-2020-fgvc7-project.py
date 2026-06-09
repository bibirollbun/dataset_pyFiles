import pandas as pd
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random



# Load CSV
df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

# Find class from one-hot columns
df['label'] = df[['healthy', 'scab', 'rust', 'multiple_diseases']].idxmax(axis=1)

# Create a full path to image
df['image'] = df['image_id'].apply(lambda x: f"/kaggle/input/plant-pathology-2020-fgvc7/images/{x}.jpg")



# Classes list
classes = df['label'].unique()

# Plot
plt.figure(figsize=(12, 6))
i = 1
for label in classes:
    sample = df[df['label'] == label].sample(2)
    for row in sample.itertuples():
        img = mpimg.imread(row.image)
        plt.subplot(2, 4, i)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        i += 1

plt.tight_layout()
plt.show()



# Randomly select 9 images
sample_df = df.sample(12)

# Plotting
plt.figure(figsize=(10, 10))
for i, row in enumerate(sample_df.itertuples()):
    img = mpimg.imread(row.image)
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(row.label)
    plt.axis('off')
plt.tight_layout()
plt.show()



# Data augmentation for training
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

# Train generator
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='image',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    subset='training',
    batch_size=32,
    shuffle=True
)

# Validation generator
val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='image',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    subset='validation',
    batch_size=32,
    shuffle=False
)


# Load MobileNetV2 base
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # Freeze base model

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=output)


# Compile the Model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',    # or 'val_accuracy'
    patience=3,            # stop after 3 epochs of no improvement
    restore_best_weights=True
)

# Train model with early stopping
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)


# Evaluate on validation data
val_loss, val_accuracy = model.evaluate(val_generator)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")


import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


from sklearn.metrics import classification_report
import numpy as np

# Get true labels
val_generator.reset()
Y_true = val_generator.classes

# Get predictions
Y_pred = model.predict(val_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Class indices mapping
labels = list(val_generator.class_indices.keys())

# Classification report
print(classification_report(Y_true, Y_pred_classes, target_names=labels))



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



# Load test.csv
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

# Create full image path
test_df['image'] = test_df['image_id'].apply(lambda x: f"/kaggle/input/plant-pathology-2020-fgvc7/images/{x}.jpg")



# Reuse the same rescaling used for training/validation
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image',
    y_col=None,
    target_size=(224, 224),
    class_mode=None,
    shuffle=False
)



# Predict probabilities
predictions = model.predict(test_generator)

# Create DataFrame with proper column names
submission = pd.DataFrame(predictions, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])

# Add image_id column
submission.insert(0, 'image_id', test_df['image_id'])


# Save to CSV
submission.to_csv("submission.csv", index=False)

