import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("TensorFlow version:", tf.__version__)



BASE_DIR = "/kaggle/input/solidworks-ai-hackathon"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train/train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test/test")

LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")

print("Files in base dir:", os.listdir(BASE_DIR))
print("Train images:", len(os.listdir(TRAIN_IMG_DIR)))
print("Test images:", len(os.listdir(TEST_IMG_DIR)))



labels_df = pd.read_csv(LABELS_PATH)
labels_df.head()



IMG_SIZE = (128, 128)   # small size = faster
NUM_CLASSES = 4      # bolt, locatingpin, nut, washer



X = []
y = []

for _, row in labels_df.iterrows():
    img_path = os.path.join(TRAIN_IMG_DIR, row["image_name"])
    
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0  # normalize
    
    X.append(img)
    y.append(row[["bolt", "locatingpin", "nut", "washer"]].values)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)



from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.05),
    RandomZoom(0.1),
])



model = Sequential([
    Input(shape=(128, 128, 3)),
    data_augmentation,

    
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dense(4, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()



# Convert labels to numeric float
y = y.astype(np.float32)

# Convert images also to float32 (safe practice)
X = X.astype(np.float32)

print(X.dtype, y.dtype)



history = model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)



test_images = []
test_names = []

for img_name in os.listdir(TEST_IMG_DIR):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    
    test_images.append(img)
    test_names.append(img_name)

test_images = np.array(test_images)

print("Test images loaded:", test_images.shape)



predictions = model.predict(test_images)
predictions = np.round(predictions).astype(int)
predictions[predictions < 0] = 0



submission = pd.DataFrame({
    "image_name": test_names,
    "bolt": predictions[:,0],
    "locatingpin": predictions[:,1],
    "nut": predictions[:,2],
    "washer": predictions[:,3]
})

submission.to_csv("submission.csv", index=False)
submission.head()





