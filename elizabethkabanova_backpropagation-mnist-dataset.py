# import scipy.io
# import mat73
import h5py
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os
import cv2
# from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time

import warnings
warnings.filterwarnings("ignore")


!kaggle competitions download -c Kannada-MNIST -p C:\Users\eliza\.kaggle\


!unzip C:\Users\eliza\.kaggle\Kannada-MNIST.zip -d C:\Users\eliza\.kaggle\KannadaMnist


!ls C:\Users\eliza\.kaggle\KannadaMnist


train_df = pd.read_csv(r"C:\Users\eliza\.kaggle\KannadaMnist\train.csv")
train_df.head()


labels = train_df['label'].values
images = train_df.drop(columns=['label'], axis=1).values

print("Images Shape: ", images.shape)
print("Label Shape: ", labels.shape)


# if images contains file paths, we need to read each image using cv2.imread()


# preprocessed_images = []

# for img in images:
#     # Making each image 28 x 28 flattened vector
#     img = img.reshape(28, 28)
    
#     # Resizing
#     img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

#     # Converting images to gray
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
#     # Normalizing to* range [0, 1]
#     # img = img / 255

#     # Reshaping Convert to Tensor Format (H, W, C) to (Batch, H, W, Channels (rgb))
#     img = img.reshape(28, 28, 1)

#     preprocessed_images.append(img)

# X = np.array(preprocessed_images)
# y = np.array(labels)

# X = X / 255


X = np.array(images) / 255 
y = np.array(labels)


# OneHotEncoder expects a 2D array
# encoder = OneHotEncoder(sparse_output=False)
# y_onehot = encoder.fit_transform(y.reshape(-1,1))


# One Hot Encoding Labels
def onehot(y):
    return np.eye(10)[y]

y = onehot(y)


print("Images Shape: ", X.shape)
print("Label Shape: ", y.shape)


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
ax = ax.flatten()

for i in range(6):
    ax[i].imshow(X[i].reshape(28,28), cmap="gray")
    ax[i].set_title(f'True: {np.argmax(y[i])}')   #y is one-hot encoded, so take argmax
    ax[i].axis("off")
    
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.astype(np.float32) 
X_test = X_test.astype(np.float32)


print(f"Train Data: {X_train.shape}, {y_train.shape}")
print(f"Test Data: {X_test.shape}, {y_test.shape}")


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha * np.exp(x))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation):
        self.input_size = input_size # img_size[0] * img_size[1]
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.activation_function = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)

        # Initialize weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * 0.01

        # Initialize the biases
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def _get_activation_function(self, activation):
        if activation == 'tanh':
            return tanh
        elif activation == 'relu':
            return relu
        elif activation == 'leaky_relu':
            return leaky_relu
        elif activation == 'elu':
            return elu
        else:
            return relu # by default

    def _get_activation_derivative(self, activation):
        if activation == 'tanh':
            return tanh_derivative
        elif activation == 'relu':
            return relu_derivative
        elif activation == 'leaky_relu':
            return leaky_relu_derivative
        elif activation == 'elu':
            return elu_derivative
        else:
            return relu_derivative # by default

    # forward propagation
    # zj = (wij * xi) + bi - activation ; a = F(zj) - activation function
    def forward(self, X):
        # Input to hidden layers (x -> h) h = x * W1 + b1
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.activation_function(self.z1)

        # Hidden to output layers (h -> o) o = h * W2 + b2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    # Backpropagation
    # Δwij = δj × Oj × η 
    def backward(self, X, y):
        m = y.shape[0]
        # error = self.a2 - y
        dz2 = self.a2 - y     
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.w2.T) * self.activation_derivative(self.a1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.w2 -= dw2 * self.learning_rate
        self.b2 -= db2 * self.learning_rate  
        self.w1 -= dw1 * self.learning_rate
        self.b1 -= db1 * self.learning_rate
        
    def cross_entropy_loss(self, y, output):
        m = y.shape[0]
        loss = -np.sum(y * np.log(output + 1e-9)) / m
        return loss         

    def accuracy(self, y, output):
        m = y.shape[0]
        pred_label = np.argmax(output, axis=1)
        true_label = np.argmax(y, axis=1)
        accuracy = np.sum(pred_label == true_label) / m 
        return accuracy

    def train(self, X_train, y_train, X_test, y_test, epochs=500):
        for epoch in range(epochs):
            # train data 
            output_train = self.forward(X_train)
            self.backward(X_train, y_train)

            train_loss = self.cross_entropy_loss(y_train, output_train)
            train_accuracy = self.accuracy(y_train, output_train)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # test data
            output_test = self.forward(X_test)

            test_loss = self.cross_entropy_loss(y_test, output_test)
            test_accuracy = self.accuracy(y_test, output_test)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)

            if epoch % 499 == 0:
                print(f"\n|For Epoch {epoch}|")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

    def predict(self, X):
        output = self.forward(X)                
        return np.argmax(output, axis=1)


def plot_performance(model):
    fig, ax = plt.subplots(2,1, figsize=(6,6))
    ax[0].plot(model.train_losses, label = "Train Loss")
    ax[0].plot(model.test_losses, label = "Test Loss")
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid()
    
    ax[1].plot(model.train_accuracies, label="Train Accuracy")
    ax[1].plot(model.test_accuracies, label="Test Accuracy")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid()
    
    plt.tight_layout()
    plt.show()


def plot_predictions(X_test, y_test, y_pred, num_samples=6):
    inc_idx = [i for i in range(len(y_test)) if np.argmax(y_test[i]) != y_pred[i]]

    if len(inc_idx) == 0:
        print("No Incorrect Labels: ")
        return
    else:
        print("Num of Incorrect Labels: ", len(inc_idx))
    
    fig, ax = plt.subplots(2, 3, figsize=(6, 8))  
    ax = ax.flatten()

    for i, idx in enumerate(inc_idx[:num_samples]):
        image = X_test[idx].reshape(28, 28)  
        true_label = np.argmax(y_test[idx])  # Якщо y_test у One-Hot Encoding
        predicted_label = y_pred[idx]
                
        ax[i].imshow(image, cmap='gray')
        ax[i].set_title(f"True: {true_label} | Pred: {predicted_label}")
        ax[i].axis('off')
    
    plt.tight_layout()
    plt.show()


activations = ['tanh', 'relu'] #, "leaky_relu", 'elu']
learning_rates = [0.001, 0.01, 0.1]
statistics = []

for activation in activations:
    print(f"\n-- Activation: {activation} --")
    for learning_rate in learning_rates:
        print(f"-- Learning Rate: {learning_rate} --")
        
        nn = NeuralNetwork(input_size=X_train.shape[1], 
                           hidden_size=64, 
                           output_size=y_train.shape[1], 
                           learning_rate=learning_rate, 
                           activation=activation)
        
        start = time.time()
        nn.train(X_train, y_train, X_test, y_test)
        time_train = time.time() - start

        start = time.time()
        y_pred = nn.predict(X_test)
        time_pred = time.time() - start
        
        test_accuracy = nn.accuracy(y_test, onehot(y_pred)) * 100
        print(f"\nPrediction Accuracy: {test_accuracy:.2f}%")
        
        statistics.append([activation, learning_rate, time_train, time_pred, test_accuracy])

        plot_performance(nn)
        plot_predictions(X_test, y_test, y_pred)

# statistics_df = pd.DataFrame(statistics, columns=["Activation", "Learning Rate" "Time Train", "Time Pred", "Final Accuracy"]).set_index(["Activation","Learning Rate"])
# statistics_df


statistics_df = pd.DataFrame(statistics, columns=["Activation", "Learning Rate", "Time Train", "Time Pred", "Final Accuracy"]).set_index(["Activation","Learning Rate"])
statistics_df


def model_building(activations, learning_rates):
    for activation in activations:
        print(f"\n-- Activation: {activation} --")
        for learning_rate in learning_rates:
            print(f"-- Learning Rate: {learning_rate} --")
            
            nn = NeuralNetwork(input_size=X_train.shape[1], 
                               hidden_size=64, 
                               output_size=y_train.shape[1], 
                               learning_rate=learning_rate, 
                               activation=activation)
            
            start = time.time()
            nn.train(X_train, y_train, X_test, y_test)
            time_train = time.time() - start
    
            start = time.time()
            y_pred = nn.predict(X_test)
            time_pred = time.time() - start
            
            test_accuracy = nn.accuracy(y_test, onehot(y_pred)) * 100
            print(f"\nPrediction Accuracy: {test_accuracy:.2f}%")
            
            statistics.append([activation, learning_rate, time_train, time_pred, test_accuracy])
    
            plot_performance(nn)
            plot_predictions(X_test, y_test, y_pred)


statistics = []
activations = ['relu', "leaky_relu", 'elu']
learning_rates = [0.7, 1.4]

model_building(activations, learning_rates)

statistics_df = pd.DataFrame(statistics, columns=["Activation", "Learning Rate", "Time Train", "Time Pred", "Final Accuracy"]).set_index(["Activation","Learning Rate"])
statistics_df


statistics_df = pd.DataFrame(statistics, columns=["Activation", "Learning Rate", "Time Train", "Time Pred", "Final Accuracy"]).set_index(["Activation","Learning Rate"])
statistics_df


test_df = pd.read_csv(r"C:\Users\eliza\.kaggle\KannadaMnist\test.csv")
test_df.head(5)


X_data_predict = test_df.drop(columns='id', axis=1).values


X_data_predict.shape


activation = 'elu'
learning_rate = 1.4

nn = NeuralNetwork(input_size=X_data_predict.shape[1], 
                   hidden_size=64, 
                   output_size=10, 
                   learning_rate=learning_rate, 
                   activation=activation)

nn.train(X_train, y_train, X_test, y_test)

y_data_predict = nn.predict(X_data_predict)


y_data_predict.size


plt.figure(figsize=(6,6))
for i in range(10):
    idx = np.random.randint(0, X_data_predict.shape[0])
    image = X_data_predict[idx].reshape(28, 28) 
    predicted_label = y_data_predict[idx]
    
    plt.subplot(5, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Pred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()


test_df['label'] = y_data_predict
test_df.head()


result_df = test_df[['id', 'label']]
result_df.head(10)


result_df.to_csv('kannada-mnist-results.csv', index=False)


!ls


!kaggle competitions submit -c Kannada-MNIST -f kannada-mnist-results.csv -m "Kannada-MNIST submission"




