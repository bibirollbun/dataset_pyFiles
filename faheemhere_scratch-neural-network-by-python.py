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


train='/kaggle/input/playground-series-s4e6/train.csv'
test='/kaggle/input/playground-series-s4e6/test.csv'
sub=pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')


# scratch_nn.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Neural Network components
def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return parameters

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e_Z / np.sum(e_Z, axis=0, keepdims=True)

def forward_propagation(X, parameters):
    caches = {"A0": X.T}
    L = len(parameters) // 2
    for l in range(1, L):
        Z = parameters[f"W{l}"] @ caches[f"A{l-1}"] + parameters[f"b{l}"]
        A = relu(Z)
        caches[f"Z{l}"], caches[f"A{l}"] = Z, A
    ZL = parameters[f"W{L}"] @ caches[f"A{L-1}"] + parameters[f"b{L}"]
    AL = softmax(ZL)
    caches[f"Z{L}"], caches[f"A{L}"] = ZL, AL
    return AL, caches

def compute_loss(Y_hat, Y):
    m = Y.shape[0]
    return -np.sum(Y.T * np.log(Y_hat + 1e-9)) / m

def backward_propagation(Y_hat, Y, parameters, caches):
    grads = {}
    L = len(parameters) // 2
    m = Y.shape[0]
    Y = Y.T

    dZ = Y_hat - Y
    grads[f"dW{L}"] = (dZ @ caches[f"A{L-1}"].T) / m
    grads[f"db{L}"] = np.sum(dZ, axis=1, keepdims=True) / m

    for l in reversed(range(1, L)):
        dA = parameters[f"W{l+1}"].T @ dZ
        dZ = dA * relu_derivative(caches[f"Z{l}"])
        grads[f"dW{l}"] = (dZ @ caches[f"A{l-1}"].T) / m
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

def model(X, Y, layer_dims, iterations=1000, learning_rate=0.01):
    parameters = initialize_parameters(layer_dims)
    for i in range(iterations):
        Y_hat, caches = forward_propagation(X, parameters)
        loss = compute_loss(Y_hat, Y)
        grads = backward_propagation(Y_hat, Y, parameters, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")
    return parameters

def predict(X, parameters):
    Y_hat, _ = forward_propagation(X, parameters)
    return np.argmax(Y_hat, axis=0)

def load_train_test(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Save test IDs before dropping
    test_ids = test_df["id"].copy()

    train_df.drop(columns=["id"], inplace=True)
    test_df.drop(columns=["id"], inplace=True)

    label_encoder = LabelEncoder()
    train_df["Target"] = label_encoder.fit_transform(train_df["Target"])

    X_train = pd.get_dummies(train_df.drop(columns=["Target"]))
    y_train = train_df["Target"].values.reshape(-1, 1)

    X_test = pd.get_dummies(test_df)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ohe = OneHotEncoder(sparse=False)
    y_train_ohe = ohe.fit_transform(y_train)

    return X_train_scaled, y_train_ohe, X_test_scaled, test_ids, label_encoder


def main():
    

    X_train, y_train, X_test, test_ids, label_encoder = load_train_test(train, test)

    layer_dims = [X_train.shape[1], 64, 32, y_train.shape[1]]

    params = model(X_train, y_train, layer_dims, iterations=1000, learning_rate=0.05)

    preds = predict(X_test, params)

    inv_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    pred_labels = [inv_map[p] for p in preds]

    submission = pd.DataFrame({
        "id": test_ids,
        "Target": pred_labels
    })
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission file saved as submission.csv")




if __name__ == "__main__":
    main()





