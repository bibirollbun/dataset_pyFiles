import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import os
import numpy as np

# Load labels
df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
df['label'].value_counts().plot(kind='bar', title='Label Distribution')
df['label'] = df['label'].astype(str)
plt.show()

# Show a few example images
sample_ids = df.sample(6)['id'].values
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, img_id in zip(axes.flatten(), sample_ids):
    img = Image.open(f'/kaggle/input/histopathologic-cancer-detection/train/{img_id}.tif')
    ax.imshow(img)
    ax.set_title(f'ID: {img_id}')
    ax.axis('off')
plt.tight_layout()
plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_dir = "/kaggle/input/histopathologic-cancer-detection/train"
df["id"] = df["id"].astype(str) + ".tif"
df["path"] = df["id"].apply(lambda x: os.path.join(image_dir, x))

# Parameters
IMG_SIZE = 96
BATCH_SIZE = 64

# Data Generators
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='/kaggle/input/histopathologic-cancer-detection/train',
    x_col='path',
    y_col='label',
    subset='training',
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='/kaggle/input/histopathologic-cancer-detection/train',
    x_col='path',
    y_col='label',
    subset='validation',
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb'
)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("baseline_model.keras", save_best_only=True)

baseline_history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)


from tensorflow.keras.optimizers import RMSprop, SGD

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

checkpoint = ModelCheckpoint("optimized_model.keras", save_best_only=True)
# Update model with RMSprop
print("Compiling model")
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

print("Training optimized model")
optimized_history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)


plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(baseline_history.history['accuracy'], label='Train Accuracy')
plt.plot(baseline_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Baseline Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(optimized_history.history['accuracy'], label='Train Accuracy')
plt.plot(optimized_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Optimized Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


baseline_res = baseline_history.history
optimized_res = optimized_history.history

results_df = pd.DataFrame({
    'Metric': ['Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss'],
    'Baseline': [baseline_res['accuracy'][-1], baseline_res['val_accuracy'][-1], baseline_res['loss'][-1], baseline_res['val_loss'][-1]],
    'Optimized': [optimized_res['accuracy'][-1], optimized_res['val_accuracy'][-1], optimized_res['loss'][-1], optimized_res['val_loss'][-1]]
})
display(results_df)


# test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# test file names DF
test_files = os.listdir('/kaggle/input/histopathologic-cancer-detection/test')
test_df = pd.DataFrame({
    'id': [os.path.splitext(file)[0] for file in test_files],
    'filename': test_files
})

# make test data gen
submission_batch_size = BATCH_SIZE * 2 

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='/kaggle/input/histopathologic-cancer-detection/test',
    x_col='filename',
    y_col=None,  # no labels for test data
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)


# predict
print("\nGenerating predictions for submission...")
optimized_model = tf.keras.models.load_model('/kaggle/working/baseline_model.keras')
predictions = optimized_model.predict(
    test_generator,
    verbose=1
)
predicted_classes = (predictions > 0.5).astype(int).flatten()


# save
submission_df = pd.DataFrame({
    'id': test_df['id'][:len(predicted_classes)],
    'label': predicted_classes
})

submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Sample of submission file:\n{submission_df.head()}")

