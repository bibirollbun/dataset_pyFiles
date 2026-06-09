# pip install opencv-python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


IMG_SIZE = 128
# 64 of fast and 128 for better accuracy


head_images = []
tail_images = []
path="/content/drive/MyDrive/Colab Notebooks/heads-or-tails-image-classification/train/"
for i in range(1, 411):
    filename = f"{path}heads/heads_{i:03}.jpg"
    img = cv2.imread(filename)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        head_images.append(img)
    else:
        print(f"Image not found or cannot load: {filename}")
for i in range(1, 391):
    filename = f"{path}tails/tails_{i:03}.jpg"
    img = cv2.imread(filename)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        tail_images.append(img)
    else:
        print(f"Image not found or cannot load: {filename}")


# from google.colab import drive
# drive.mount('/content/drive')


# Combine images
X = np.array(head_images + tail_images)  # Shape: (num_samples, IMG_SIZE, IMG_SIZE, 3)
y = np.array([0]*len(head_images) + [1]*len(tail_images))  # 0 = head, 1 = tail

# Normalize pixel values to [0, 1]
X = X / 255.0  # Already float after cv2

print(f"Total samples: {X.shape[0]}")
print(f"Image shape: {X.shape[1:]}")
print(f"Labels: {np.unique(y, return_counts=True)}")


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")


model = Sequential()

# Layer 1 - Convolution + MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2 - Deeper convolution
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Fully Connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Prevent overfitting
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Summary of the model
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=10,                      # Start with 10; increase if needed
    batch_size=32,                  # Small batches help generalization
    validation_data=(X_val, y_val) # Monitor model performance
)


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_path = "/content/drive/MyDrive/Colab Notebooks/heads-or-tails-image-classification/test/"
test_images = []
test_filenames = []
for i in range(1, 201):
    filename = f"unknown_{i:03}.jpg"
    filepath = test_path + filename
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        test_images.append(img)
        test_filenames.append(filename)
    else:
        print(f"Image not found or couldn't load: {filename}")

X_test = np.array(test_images)
print(f"Loaded {len(X_test)} test images.")


# Predict probabilities of class 1 (tail)
pred_probs = model.predict(X_test).reshape(-1)
# Probability of HEAD = 1 - probability of TAIL
prob_heads = 1 - pred_probs
submission = pd.DataFrame({
    "prediction_id": list(range(1, 201)),
    "probability_of_heads": prob_heads
})
submission.to_csv("submission.csv", index=False)




