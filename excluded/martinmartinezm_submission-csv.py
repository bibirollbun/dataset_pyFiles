import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Efficient One-Hot Encoding
def one_hot_encode(sequences, vocab_size=5):
    return np.eye(vocab_size)[sequences]

# Custom Dataset Class
class RNADataset:
    def __init__(self, sequences, structures):
        self.sequences = one_hot_encode(sequences, vocab_size=5).reshape(sequences.shape[0], -1)
        self.structures = np.array(structures).reshape(-1, 3)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.structures[idx]

# He Initialization for Better Gradient Flow
def initialize_weights(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim):
    weights = {
        'W1': np.random.randn(input_dim, hidden_dim1) * np.sqrt(2.0 / input_dim),
        'b1': np.zeros((hidden_dim1,)),
        'W2': np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(2.0 / hidden_dim1),
        'b2': np.zeros((hidden_dim2,)),
        'W3': np.random.randn(hidden_dim2, hidden_dim3) * np.sqrt(2.0 / hidden_dim2),
        'b3': np.zeros((hidden_dim3,)),
        'W4': np.random.randn(hidden_dim3, hidden_dim4) * np.sqrt(2.0 / hidden_dim3),
        'b4': np.zeros((hidden_dim4,)),
        'W5': np.random.randn(hidden_dim4, output_dim) * np.sqrt(2.0 / hidden_dim4),
        'b5': np.zeros((output_dim,))
    }
    return weights

# Forward Pass with Batch Normalization & Leaky ReLU
def forward_pass(x, weights, dropout_rate=0.15):
    def batch_norm(z):
        return (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-5)
    
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    z1 = np.dot(x, weights['W1']) + weights['b1']
    a1 = leaky_relu(batch_norm(z1))
    a1 *= np.random.binomial(1, 1 - dropout_rate, size=a1.shape) / (1 - dropout_rate)
    
    z2 = np.dot(a1, weights['W2']) + weights['b2']
    a2 = leaky_relu(batch_norm(z2))
    a2 *= np.random.binomial(1, 1 - dropout_rate, size=a2.shape) / (1 - dropout_rate)
    
    z3 = np.dot(a2, weights['W3']) + weights['b3']
    a3 = leaky_relu(batch_norm(z3))
    
    z4 = np.dot(a3, weights['W4']) + weights['b4']
    a4 = leaky_relu(batch_norm(z4))
    
    z5 = np.dot(a4, weights['W5']) + weights['b5']
    y_pred = z5
    
    return y_pred, a1, a2, a3, a4

# Compute Loss with L2 Regularization
def compute_loss(y_pred, y_true, weights, lambda_l2=0.00005):
    mse_loss = np.mean((y_pred - y_true) ** 2)
    l2_penalty = lambda_l2 * sum(np.sum(w ** 2) for w in weights.values() if 'W' in w)
    return mse_loss + l2_penalty

# Backward Pass with Adam Optimizer
def backward_pass(x, y_true, y_pred, a1, a2, a3, a4, weights, momentums, velocities, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    dz5 = 2 * (y_pred - y_true)
    dW5 = np.dot(a4.T, dz5)
    db5 = np.sum(dz5, axis=0)
    
    da4 = np.dot(dz5, weights['W5'].T)
    dW4 = np.dot(a3.T, da4)
    db4 = np.sum(da4, axis=0)
    
    da3 = np.dot(da4, weights['W4'].T)
    dW3 = np.dot(a2.T, da3)
    db3 = np.sum(da3, axis=0)
    
    da2 = np.dot(da3, weights['W3'].T)
    dW2 = np.dot(a1.T, da2)
    db2 = np.sum(da2, axis=0)
    
    da1 = np.dot(da2, weights['W2'].T)
    dW1 = np.dot(x.T, da1)
    db1 = np.sum(da1, axis=0)
    
    for i, (dW, db) in enumerate([(dW1, db1), (dW2, db2), (dW3, db3), (dW4, db4), (dW5, db5)]):
        momentums[i] = beta1 * momentums[i] + (1 - beta1) * dW
        velocities[i] = beta2 * velocities[i] + (1 - beta2) * (dW ** 2)
        corrected_momentum = momentums[i] / (1 - beta1)
        corrected_velocity = velocities[i] / (1 - beta2)
        weights[f'W{i+1}'] -= learning_rate * corrected_momentum / (np.sqrt(corrected_velocity) + epsilon)
        weights[f'b{i+1}'] -= learning_rate * db

# Train Model with Increased Epochs
def train_model(dataset, weights, epochs=100, batch_size=32, learning_rate=0.0015, decay=0.98, dropout_rate=0.15):
    loss_history = []
    accuracy_history = []
    momentums = [np.zeros_like(weights[f'W{i+1}']) for i in range(5)]
    velocities = [np.zeros_like(weights[f'W{i+1}']) for i in range(5)]
    
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            x_batch, y_batch = zip(*batch)
            x_batch, y_batch = np.array(x_batch), np.array(y_batch).reshape(len(batch), -1)
            y_pred, a1, a2, a3, a4 = forward_pass(x_batch, weights, dropout_rate)
            
            loss = compute_loss(y_pred, y_batch, weights)
            backward_pass(x_batch, y_batch, y_pred, a1, a2, a3, a4, weights, momentums, velocities, learning_rate)
            total_loss += loss
            total_correct += np.sum(np.round(y_pred) == np.round(y_batch))
            total_samples += y_batch.size
        
        avg_loss = total_loss / (len(dataset) / batch_size)
        accuracy = total_correct / total_samples
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        learning_rate *= decay

    plt.plot(loss_history, label='Loss')
    plt.plot(accuracy_history, label='Accuracy')
    plt.legend()
    plt.show()

train_model(RNADataset(np.random.randint(0, 5, (1000, 100)), np.random.rand(1000, 3)), initialize_weights(500, 512, 256, 128, 64, 3))


