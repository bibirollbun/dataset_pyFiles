import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(os.path.join(dirname))



BASE_DIR = "/kaggle/input/histopathologic-cancer-detection"



# ==========================================
# STEP 1: Import Libraries
# ==========================================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================================
# STEP 2: Set Paths for Dataset
# ==========================================
# Dataset is already mounted in Kaggle
BASE_DIR = "/kaggle/input/histopathologic-cancer-detection"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")

# ==========================================
# STEP 3: Load Labels
# ==========================================
labels_df = pd.read_csv(LABELS_PATH)
labels_df['id'] = labels_df['id'] + ".tif"  # Add file extension
labels_df['label'] = labels_df['label'].astype(str)  # Convert to string for keras
print("Dataset shape:", labels_df.shape)
print(labels_df.head())

# ==========================================
# STEP 4: Data Generators
# ==========================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="label",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="binary",
    target_size=(96, 96)
)

val_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="label",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="binary",
    target_size=(96, 96)
)

# ==========================================
# STEP 5: Build Lightweight CNN Model
# ==========================================
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(96,96,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# STEP 6: Train Model
# ==========================================
epochs = 3  # Keep epochs low for faster Kaggle runtime
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ==========================================
# STEP 7: Predict on Test Data
# ==========================================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=BASE_DIR,
    classes=["test"],
    class_mode=None,
    shuffle=False,
    target_size=(96, 96),
    batch_size=32
)

preds = model.predict(test_generator, verbose=1)

# ==========================================
# STEP 8: Create Submission File
# ==========================================
test_filenames = test_generator.filenames
test_ids = [fname.split("/")[1].replace(".tif", "") for fname in test_filenames]

submission = pd.DataFrame({
    "id": test_ids,
    "label": preds.flatten()
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv created at /kaggle/working/")

# ==========================================
# Plot Predictions
# ==========================================
for i in range(9):
    img_path = os.path.join(TEST_DIR, test_filenames[i].split("/")[1])
    img = tf.keras.utils.load_img(img_path, target_size=(96, 96))
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"Pred: {preds[i][0]:.2f}")
    plt.axis('off')
plt.show()


