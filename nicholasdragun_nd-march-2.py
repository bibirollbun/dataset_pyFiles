# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
print(os.listdir("/kaggle/working/"))  # List all files in /kaggle/working/


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)


import os

# Check all available datasets in /kaggle/input/
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  # Print full file paths


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Load dataset (Make sure the file is uploaded in Kaggle dataset section)
file_path = "/kaggle/input/Kannada-MNIST/Dig-MNIST.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Inspect the dataset
print(df.head())

# Separate labels and features
y = df.iloc[:, 0].values  # First column is the label
X = df.iloc[:, 1:].values  # The rest are pixel values

# Normalize pixel values to [0, 1]
X = X / 255.0

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=10)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model (Fully Connected Neural Network)
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    keras.layers.Dense(256, activation='relu'),  # Hidden layer 1
    keras.layers.Dense(128, activation='relu'),  # Hidden layer 2
    keras.layers.Dense(10, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=20, batch_size=64, verbose=1)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()

# Prepare predictions for Kaggle submission
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Create submission DataFrame
submission_df = pd.DataFrame({'Id': np.arange(len(predicted_labels)), 'Label': predicted_labels})

# Save submission file (ensure the correct Kaggle output path)
submission_filename = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_filename, index=False)
print(f"Submission file saved as: {submission_filename}")


submission_filename = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_filename, index=False)
print(f"✅ Submission file saved as: {submission_filename}")


# Ensure predictions are generated correctly
submission_filename = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_filename, index=False)

# Confirm file creation
print(f"✅ Submission file saved as: {submission_filename}")
print(os.listdir("/kaggle/working/"))  # Verify file exists


# Ensure predictions are generated correctly
submission_filename = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_filename, index=False)

# Confirm file creation
print(f"✅ Submission file saved as: {submission_filename}")
print(os.listdir("/kaggle/working/"))  # Verify file exists


from IPython.display import FileLink
FileLink("/kaggle/working/submission.csv")


FileLink("/kaggle/working/submission.csv")


import shutil

new_path = "/kaggle/working/submission.csv"
shutil.copy(file_path, new_path)

print(f"✅ File copied to: {new_path}")


from IPython.display import FileLink
FileLink("/kaggle/working/submission.csv")

