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



import numpy as np
from keras.datasets import fashion_mnist

# -----------------
# 1. Load & Preprocess Data
# -----------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# One-hot encode labels
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# -----------------
# 2. Initialize Parameters
# -----------------
input_dim = 784
hidden_dim = 128
output_dim = 10

np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0/input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0/hidden_dim)
b2 = np.zeros((1, output_dim))

# -----------------
# 3. Activations & Helpers
# -----------------
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

# -----------------
# 4. Training Loop
# -----------------
learning_rate = 0.01
batch_size = 64
epochs = 20

for epoch in range(epochs):
    # Shuffle training data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train_oh = y_train_oh[indices]

    for i in range(0, x_train.shape[0], batch_size):
        X_batch = x_train[i:i+batch_size]
        Y_batch = y_train_oh[i:i+batch_size]

        # Forward pass
        Z1 = X_batch.dot(W1) + b1
        A1 = relu(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = softmax(Z2)

        # Loss
        loss = cross_entropy(Y_batch, A2)

        # Backward pass
        m = X_batch.shape[0]
        dZ2 = (A2 - Y_batch) / m
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * relu_deriv(Z1)
        dW1 = X_batch.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Accuracy on training set per epoch
    Z1 = x_train.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    preds = np.argmax(softmax(Z2), axis=1)
    acc = np.mean(preds == y_train)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Train Acc: {acc:.4f}")

# -----------------
# 5. Evaluate on Test Set
# -----------------
Z1 = x_test.dot(W1) + b1
A1 = relu(Z1)
Z2 = A1.dot(W2) + b2
preds_test = np.argmax(softmax(Z2), axis=1)
acc_test = np.mean(preds_test == y_test)
print("Final Test Accuracy:", acc_test)

# -----------------
# 6. Prepare Submission File
# -----------------
import pandas as pd
submission = pd.DataFrame({"id": np.arange(len(preds_test)), "target_feature": preds_test})
submission.to_csv("submission.csv", index=False)


