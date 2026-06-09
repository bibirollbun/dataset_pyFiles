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


# Import packages and data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')


test_df= pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')


df.head()


#Image preprocessing 
labels = df['label'].values
images = df.drop(columns=['label'], axis=1).values

print("Images Shape: ", images.shape)
print("Label Shape: ", labels.shape)


#Test Image preprocessing 
test_id = test_df['id'].values
test_images = test_df.drop(columns=['id'], axis=1).values

print("Test Images Shape: ", test_images.shape)
print("Test ids Shape: ", test_id.shape)


#Normalizing to range [0, 1] - training data
X = np.array(images) / 255 
y = np.array(labels)


# Convert labels to categorical (one-hot encoding)
def onehot(y):
    return np.eye(10)[y]

y = onehot(y)


print("Images Shape: ", X.shape)
print("Label Shape: ", y.shape)


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Build the neural network model
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu')) # 10 neurons, input dimension = number of features (pixels)
model.add(Dense(10, activation='relu')) # Hidden layer with 10 neurons
model.add(Dense(10, activation='softmax')) # Output layer with 10 neurons (for 10 digits)


#compile the model
model.compile(
    optimizer='adam',            # or SGD, RMSprop, etc.
    loss='categorical_crossentropy',  # or 'binary_crossentropy', depending on your task
    metrics=['accuracy']
)


#need to train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))



# Evaluate the model on the test set 
loss, accuracy = model.evaluate(X_test, y_test, verbose=0) 
print(f"Test accuracy: {accuracy:.2f}") 


# Make predictions on the test set 
predictions = model.predict(X_test) 
predicted_classes = np.argmax(predictions, axis=1) 
true_classes = np.argmax(y_test, axis=1) 


# Confusion matrix for test set
conf_matrix = confusion_matrix(true_classes, predicted_classes) 


# Plot the confusion matrix of test data
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Classification report for test data
from sklearn.metrics import classification_report

print(classification_report(true_classes, predicted_classes, target_names=[str(i) for i in range(10)]))


# Make predictions on the test set for submission 
test_predictions = model.predict(test_images) 
predicted_classes = np.argmax(test_predictions, axis=1) 


test_df["label"] = predicted_classes


test_df_new = test_df[["id", "label"]]


test_df_new.head()


# Save to CSV
test_df_new.to_csv("/kaggle/working/submission.csv", index=False)

print("Predictions saved to submission.csv")

