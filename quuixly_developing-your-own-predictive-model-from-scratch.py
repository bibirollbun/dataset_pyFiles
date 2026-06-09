import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

# Drop all string columns for simplicity in the code
data = data.select_dtypes(exclude=["object"])

# Split a dataset into training and testing sets
X = data["waist_to_hip_ratio"]
y = data["bmi"]

print(X.shape)

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=420
)

# Train for a future reference simple linear regression model
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = LinearRegression()
model.fit(X_train, y_train)

X_line = np.array([X.min(), X.max()])
y_line = model.coef_[0] * X_line + model.intercept_

fig, ax = plt.subplots()

ax.scatter(X, y, color='royalblue', alpha=0.5)
ax.plot(X_line, y_line, color='red', linewidth=3)

# Calculate accuracy
threshold = 25 # Overweight is defined as a BMI over 25

total = X_test.shape[0]
good_classified = 0
for i in range(total):
    y_true = 1 if y_test[i] > threshold else 0
    pred = 1 if model.predict(X_test[i].reshape(1, -1)) > threshold else 0

    if pred == y_true:
        good_classified += 1

print(f"Accuracy of LinearRegression: {good_classified / total}")


import numpy as np


class MostUselessAlgorithm:
    def __init__(self):
        self.alpha = 0
        self.max_x = 0
    
    def fit(self, X, y):
        angles = []
        # Needed to adjust our vector
        self.max_x = np.max(X)
        
        for j in range(X.shape[0]):
            loss = self._get_loss(X[j], y[j])
            angle = np.arccos(loss)
            angles.append(angle)

        self.alpha = np.mean(angles)
            
    def predict(self, X):
        return np.clip((X * self.get_vector())[1], 0, 1)

    def _get_loss(self, X, y):
        a = np.hstack([X, y])
        b = X * self.get_vector()

        return self._get_cosine_simlarity(a, b)

    def get_vector(self):
        return np.array([np.cos(self.alpha), np.sin(self.alpha)]) * (self.max_x / np.cos(self.alpha))

    def _get_cosine_simlarity(self, a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot / (norm_a * norm_b)

# Train
most_useless_algorithm = MostUselessAlgorithm()
most_useless_algorithm.fit(X_train, y_train)

# Visualize
alpha = most_useless_algorithm.alpha
fig, ax = plt.subplots()

vector = most_useless_algorithm.get_vector()

ax.scatter(X_train, y_train, color='royalblue', alpha=0.5)
plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='red')


# Overweight is defined as a BMI over 25
threshold = 25

total = X_test.shape[0]
good_classified = 0
for i in range(total):
    y_true = 1 if y_test[i] > threshold else 0
    pred = 1 if most_useless_algorithm.predict(X_test[i]) > threshold else 0

    if pred == y_true:
        good_classified += 1

print(f"Accuracy of MostUselessAlgorithm (without rotating): {good_classified / total}")


from sklearn.preprocessing import MinMaxScaler

# Scaler X
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scaler y
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Train
most_useless_algorithm = MostUselessAlgorithm()
most_useless_algorithm.fit(X_train_scaled, y_train_scaled)

# Visualize
alpha = most_useless_algorithm.alpha
fig, ax = plt.subplots()

vector = most_useless_algorithm.get_vector()

ax.scatter(X_train_scaled, y_train_scaled, color='royalblue', alpha=0.5)
plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='red')


# Overweight is defined as a BMI over 25
print(scaler_y.transform(np.array(25).reshape(-1, 1)))
threshold = 0.4248927

total = X_test_scaled.shape[0]
good_classified = 0
for i in range(total):
    y_true = 1 if y_test_scaled[i] > threshold else 0
    pred = 1 if most_useless_algorithm.predict(X_test_scaled[i]) > threshold else 0

    if pred == y_true:
        good_classified += 1

print(f"Accuracy of MostUselessAlgorithm: {good_classified / total}")


import numpy as np


class MostUselessAlgorithmOptimized(MostUselessAlgorithm):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self.max_x = np.max(X)
        angles = self._get_angles(X, y)
        self.alpha = np.mean(angles)

    def predict(self, X):
        return (X * (self.get_vector()[1]))

    def _get_angles(self, X, y):
        angles = np.arctan2(y, X)
        return angles

# Train
most_useless_algorithm_optimized = MostUselessAlgorithmOptimized()
most_useless_algorithm_optimized.fit(X_train_scaled, y_train_scaled)

# Visualize
alpha = most_useless_algorithm_optimized.alpha
fig, ax = plt.subplots()

vector = most_useless_algorithm_optimized.get_vector()

ax.scatter(X_train_scaled, y_train_scaled, color='royalblue', alpha=0.5)
plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='red')


# Overweight is defined as a BMI over 25
print(scaler_y.transform(np.array(25).reshape(-1, 1)))
threshold = 0.4248927

total = X_test_scaled.shape[0]
good_classified = 0
for i in range(total):
    y_true = 1 if y_test_scaled[i] > threshold else 0
    pred = 1 if most_useless_algorithm_optimized.predict(X_test_scaled[i]) > threshold else 0

    if pred == y_true:
        good_classified += 1

print(f"Accuracy of MostUselessAlgorithm (optimized): {good_classified / total}")


import time
import numpy as np

def benchmark(func, n_runs=10, *args, **kwargs):
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        
        func(*args, **kwargs)
        
        end = time.perf_counter()
        times.append(end - start)
    
    median_time = np.median(times)
    return median_time


median_time = benchmark(most_useless_algorithm.fit, 10, X_train_scaled, y_train_scaled)
print(f"Median time (fit, most_useless_algorithm): {median_time:.6f} s")

median_time = benchmark(most_useless_algorithm.predict, 10, X_test_scaled)
print(f"Median time (predict, most_useless_algorithm): {median_time:.6f} s")

median_time = benchmark(most_useless_algorithm_optimized.fit, 10, X_train_scaled, y_train_scaled)
print(f"Median time (fit, most_useless_algorithm_optimized): {median_time:.6f} s")

median_time = benchmark(most_useless_algorithm_optimized.predict, 10, X_test_scaled)
print(f"Median time (predict, most_useless_algorithm_optimized): {median_time:.6f} s")




