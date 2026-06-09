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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the train and test data
train = pd.read_csv('/kaggle/input/digit-recognizer-without-Extra-Training-Data/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer-without-Extra-Training-Data/test.csv')

# Display the first few rows of the train dataset
train.head()

# Display the first few rows of the test dataset
test.head()



# Visualize some sample images from the training data
def visualize_images(images, labels):
    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    for i in range(5):
        axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plt.show()

# Extract image and label data
X_train = train.drop('label', axis=1).values
y_train = train['label'].values

# Visualize the first few images and their labels
visualize_images(X_train, y_train)



# Convert X_train to DataFrame
X_train_df = pd.DataFrame(X_train)

# Display statistical summary for pixel values
print(X_train_df.describe())



# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0

# Normalize the test data
X_test = test.values / 255.0



from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=48)



# Check class distribution in the train dataset
import seaborn as sns

# Get the class distribution
class_distribution = train['label'].value_counts()

# Plot the class distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title("Class Distribution in Training Data")
plt.xlabel("Digit")
plt.ylabel("Number of Instances")
plt.show()



import tensorflow as tf
from tensorflow.keras import layers, models

# Build a  neural network
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Print the model summary
model.summary()



# Train the model
history = model.fit(X_train_split, y_train_split, epochs=10, batch_size=32, validation_data=(X_val_split, y_val_split))

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()



# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(X_val_split, y_val_split)
print(f'Validation accuracy: {val_acc}')



import seaborn as sns
from sklearn.metrics import confusion_matrix
# Make predictions on the validation data
y_val_pred = model.predict(X_val_split)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_val_split, y_val_pred_classes)
# Plot the confusion matrix using heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10), cbar=False)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Make predictions on the test dataset
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)



# Save the trained model in .h5 format
model.save('digit_recognizer_model.h5')

# Make predictions on the test dataset
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'ImageId': np.arange(1, len(y_pred_classes) + 1),
    'Label': y_pred_classes
})

# Save to CSV file
submission.to_csv('submission.csv', index=False)




