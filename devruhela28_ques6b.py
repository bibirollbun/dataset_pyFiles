import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('/kaggle/input/iiita-iml-fall-2025-lab-1/train.csv')
test_df  = pd.read_csv('/kaggle/input/iiita-iml-fall-2025-lab-1/test.csv')

features = ['lotsize', 'bedrooms', 'bathrms']
X_full = train_df[features].values
y_full = train_df['price'].values

split_idx = int(0.7 * len(X_full))
X_train, X_val = X_full[:split_idx], X_full[split_idx:]
y_train, y_val = y_full[:split_idx], y_full[split_idx:]


def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]

def feature_scaling(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu) / sigma, mu, sigma

def unscale_theta(theta_scaled, mu, sigma):
    theta_unscaled = np.zeros_like(theta_scaled)
    theta_unscaled[1:] = theta_scaled[1:] / sigma
    theta_unscaled[0] = theta_scaled[0] - np.sum((mu / sigma) * theta_scaled[1:])
    return theta_unscaled

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def percent_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def batch_gradient_descent(X, y, alpha=0.01, iterations=1000):
    m = len(y)
    X_b = add_bias(X)
    theta = np.zeros(X_b.shape[1])
    history = []
    for _ in range(iterations):
        gradient = (1/m) * (X_b.T @ (X_b @ theta - y))
        theta -= alpha * gradient
        history.append(rmse(y, X_b @ theta))
    return theta, history

def stochastic_gradient_descent(X, y, alpha=0.01, epochs=50):
    m = len(y)
    X_b = add_bias(X)
    theta = np.zeros(X_b.shape[1])
    history = []
    for _ in range(epochs):
        for i in range(m):
            rand_i = np.random.randint(m)
            xi = X_b[rand_i:rand_i+1]
            yi = y[rand_i:rand_i+1]
            gradient = xi.T @ (xi @ theta - yi)
            theta -= alpha * gradient
        history.append(rmse(y, X_b @ theta))
    return theta, history

def mini_batch_gradient_descent(X, y, alpha=0.01, iterations=1000, batch_size=20):
    m = len(y)
    X_b = add_bias(X)
    theta = np.zeros(X_b.shape[1])
    history = []
    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_b_shuffled = X_b[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            xi = X_b_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradient = (xi.T @ (xi @ theta - yi)) / len(xi)
            theta -= alpha * gradient
        history.append(rmse(y, X_b @ theta))
    return theta, history


X_train_scaled, mu, sigma = feature_scaling(X_train)
X_val_scaled = (X_val - mu) / sigma

results = {}
histories = {}

theta_bgd_ns, hist_bgd_ns = batch_gradient_descent(X_train, y_train, alpha=1e-7, iterations=200)
y_val_pred = add_bias(X_val) @ theta_bgd_ns
results["Batch GD (No Scaling)"] = (mse(y_val, y_val_pred), percent_error(y_val, y_val_pred))
histories["Batch GD (No Scaling)"] = hist_bgd_ns

theta_bgd_s, hist_bgd_s = batch_gradient_descent(X_train_scaled, y_train, alpha=0.01, iterations=200)
theta_bgd_s_unscaled = unscale_theta(theta_bgd_s, mu, sigma)
y_val_pred = add_bias(X_val) @ theta_bgd_s_unscaled
results["Batch GD (Scaling)"] = (mse(y_val, y_val_pred), percent_error(y_val, y_val_pred))
histories["Batch GD (Scaling)"] = hist_bgd_s

theta_sgd_ns, hist_sgd_ns = stochastic_gradient_descent(X_train, y_train, alpha=1e-7, epochs=50)
y_val_pred = add_bias(X_val) @ theta_sgd_ns
results["SGD (No Scaling)"] = (mse(y_val, y_val_pred), percent_error(y_val, y_val_pred))
histories["SGD (No Scaling)"] = hist_sgd_ns

theta_sgd_s, hist_sgd_s = stochastic_gradient_descent(X_train_scaled, y_train, alpha=0.01, epochs=50)
theta_sgd_s_unscaled = unscale_theta(theta_sgd_s, mu, sigma)
y_val_pred = add_bias(X_val) @ theta_sgd_s_unscaled
results["SGD (Scaling)"] = (mse(y_val, y_val_pred), percent_error(y_val, y_val_pred))
histories["SGD (Scaling)"] = hist_sgd_s

# Mini-Batch GD
theta_mbgd_ns, hist_mbgd_ns = mini_batch_gradient_descent(X_train, y_train, alpha=1e-8, iterations=200, batch_size=20)
y_val_pred = add_bias(X_val) @ theta_mbgd_ns
results["Mini-Batch GD (No Scaling)"] = (mse(y_val, y_val_pred), percent_error(y_val, y_val_pred))
histories["Mini-Batch GD (No Scaling)"] = hist_mbgd_ns

theta_mbgd_s, hist_mbgd_s = mini_batch_gradient_descent(X_train_scaled, y_train, alpha=0.01, iterations=200, batch_size=20)
theta_mbgd_s_unscaled = unscale_theta(theta_mbgd_s, mu, sigma)
y_val_pred = add_bias(X_val) @ theta_mbgd_s_unscaled
results["Mini-Batch GD (Scaling)"] = (mse(y_val, y_val_pred), percent_error(y_val, y_val_pred))
histories["Mini-Batch GD (Scaling)"] = hist_mbgd_s


for method, (mse_val, err) in results.items():
    print(f"{method}: MSE={mse_val:.2f}, %Error={err:.2f}")


methods = list(results.keys())
mse_vals = [results[m][0] for m in methods]
err_vals = [results[m][1] for m in methods]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.barh(methods, mse_vals, color='skyblue')
plt.xlabel("MSE")
plt.xscale("log")  
plt.title("MSE Comparison")

plt.subplot(1,2,2)
plt.barh(methods, err_vals, color='salmon')
plt.xlabel("% Error")
plt.xscale("log")  
plt.title("Percentage Error Comparison")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
for method, hist in histories.items():
    plt.plot(hist, label=method)
plt.xlabel("Iterations / Epochs")
plt.ylabel("RMSE (Training)")
plt.title("Learning Curves of Different Gradient Descent Variants")
plt.legend()
plt.grid(True)
plt.show()





