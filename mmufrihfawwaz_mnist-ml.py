# Install and import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical





# Load Dataset
train_data = pd.read_csv('/kaggle/input/xpc-team-digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/xpc-team-digit-recognizer/test.csv')

train_data.head()




test_data.head()


# Check for missing values
print("Missing values in train_data:")
print(train_data.isnull().sum().sum())

print("\nMissing values in test_data:")
print(test_data.isnull().sum().sum())

# Check for duplicate rows
print("\nDuplicate rows in train_data:")
print(train_data.duplicated().sum())

print("\nDuplicate rows in test_data:")
print(test_data.duplicated().sum())

# Display data types
print("\nData types in train_data:")
print(train_data.info())

print("\nData types in test_data:")
print(test_data.info())


# Normalize data and separate features and labels
X = train_data.drop('label', axis=1)  # Drop the 'label' column to get input features
y = train_data['label']              # Extract the 'label' column as the target

# Normalize pixel values to range [0, 1]
X = X / 255.0
test_data = test_data / 255.0

# Check the maximum and minimum pixel values in the training data
print("Max pixel value in X:", X.max().max())
print("Min pixel value in X:", X.min().min())

# Also check the test data if needed
print("Max pixel value in test_data:", test_data.max().max())


# Normalize data and separate features and labels
X = train_data.drop('label', axis=1)  # Drop the 'label' column to get input features
y = train_data['label']              # Extract the 'label' column as the target

# Normalize pixel values to range [0, 1]
X = X / 255.0
test_data = test_data / 255.0

# Convert the labels into one-hot encoded vectors
y = to_categorical(y, num_classes=10)



# Visualize sample images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(y[i])}")
    plt.axis('off')
plt.show()



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# One-Hot Encoding the labels
y_cat = to_categorical(y)

# Display the first few labels in one-hot encoding (horizontally)
print("Example of one-hot encoded labels (horizontal view):")
for i in range(5):
    print(f"Label {i}: {y_cat[i].tolist()}")





from tensorflow.keras.layers import Input

# Build the model
model = Sequential()
model.add(Input(shape=(784,))) # Use Input layer to define the input shape
model.add(Dense(128, activation='relu'))  # First hidden layer with 128 neurons
model.add(Dense(64, activation='relu'))                      # Second hidden layer with 64 neurons
model.add(Dense(10, activation='softmax'))                   # Output layer with 10 classes (for classification)


# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Training Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# Predict test data
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)



# Visualize some test images with predictions
plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)

    # Display the image (change this line if X_test is not a DataFrame)
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap='gray')

    # Set the title color: green if prediction is correct, red if wrong
    color = 'green' if predicted_labels[i] == np.argmax(y_test[i]) else 'red'

    # Display the true label and the predicted label
    plt.title(f"True: {np.argmax(y_test[i])}\nPred: {predicted_labels[i]}", color=color)

    # Hide axis lines and ticks
    plt.axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()



# Submit the results
submission = pd.DataFrame({'ImageId': range(1, len(predicted_labels) + 1), 'Label': predicted_labels})
submission.to_csv('submission.csv', index=False)


