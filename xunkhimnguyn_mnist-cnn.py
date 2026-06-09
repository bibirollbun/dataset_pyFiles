# Ignore this again
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


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape to (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

# One-hot encoding
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_labels)
y_test = to_categorical(y_test, num_classes=num_labels)

# CNN model
model = Sequential()

# Block 1
model.add(Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Fully connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("\n✅ Test accuracy: %.2f%%" % (acc * 100))

# Plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Training/Validation Accuracy')
plt.show()


# Load test data
test_df = pd.read_csv('/kaggle/input/mnist-dataset-number-classification/test_mnist.csv')
display(test_df.head())

# Preprocess test data
test_data = test_df.drop(['Unnamed: 0', 'id'], axis=1).values

# Reshape to (samples, height, width, channels) and normalize
test_data = test_data.reshape(-1, 28, 28, 1).astype('float32') / 255.

display(test_data.shape)


# Make predictions
predictions = model.predict(test_data)
display(predictions.shape)


# Format submission file
# Get the predicted class for each sample
predicted_labels = np.argmax(predictions, axis=1)

# Create a submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'label': predicted_labels})

display(submission_df.head())


# Save submission file
submission_df.to_csv('submission.csv', index=False)


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get the true labels for the test set
true_labels = np.argmax(y_test, axis=1)

# Get the predicted labels for the test set
predicted_test_labels = np.argmax(model.predict(x_test), axis=1)

# Generate the confusion matrix
cm = confusion_matrix(true_labels, predicted_test_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

