# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- 1. Mock Data Generation (for demonstration) ---
# In a real scenario, you would load data from a file (e.g., CSV).
def generate_mock_data(num_samples=1000, sequence_length=50, num_sensors=5):
    """
    Generates a mock dataset of time-series sensor data.
    Simulates two distinct behaviors (labels 0 and 1).
    """
    X = np.random.randn(num_samples, sequence_length, num_sensors)
    y = np.zeros(num_samples, dtype=int)

    # Introduce a pattern for a specific behavior (e.g., label 1)
    for i in range(num_samples // 2, num_samples):
        # Add a sine wave pattern to a few sensors for behavior 1
        t = np.linspace(0, 2 * np.pi, sequence_length)
        pattern = np.sin(t) * 2 + np.cos(t * 3)
        # Add the pattern with some additional random noise to make the problem harder
        X[i, :, 0] += pattern + np.random.randn(sequence_length) * 0.5
        y[i] = 1

    return X, y



# --- 2. PyTorch Dataset and DataLoader ---
class SensorDataset(Dataset):
    """
    Custom PyTorch Dataset for our sensor data.
    """
    def __init__(self, features, labels):
        # Convert numpy arrays to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 3. Neural Network Model Definition ---
class BehaviorDetector(nn.Module):
    """
    A simple neural network for classifying time-series sensor data.
    This uses a combination of a simple Recurrent Neural Network (RNN) layer
    to process the sequence and a fully-connected layer for classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer to process the time-series data
        # batch_first=True means the input tensor has shape (batch, sequence, features)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully-connected layer for final classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass the input through the RNN layer
        out, _ = self.rnn(x, h0)
        
        # Get the output of the last time step for classification
        out = self.fc(out[:, -1, :])
        return out





# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print("Generating mock sensor data...")
    X, y = generate_mock_data()
    
    # --- Data Visualization: Plotting mock data for different behaviors ---
    # This helps us understand what the model needs to learn.
    plt.figure(figsize=(12, 6))
    
    # Find indices of a few samples for each behavior
    behavior_0_idx = np.where(y == 0)[0][:3]
    behavior_1_idx = np.where(y == 1)[0][:3]

    # Plot the first sensor's data for Behavior 0 samples
    for i, idx in enumerate(behavior_0_idx):
        plt.plot(X[idx, :, 0], label=f'Behavior 0 Sample {i+1}', linestyle='--')
    
    # Plot the first sensor's data for Behavior 1 samples
    for i, idx in enumerate(behavior_1_idx):
        plt.plot(X[idx, :, 0], label=f'Behavior 1 Sample {i+1}', linestyle='-')
    
    plt.title('Sample Sensor Data for Two Different Behaviors')
    plt.xlabel('Time Step')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    # This is a critical step for many ML models, especially neural networks.
    # We fit the scaler on the training data only to avoid data leakage.
    scaler = StandardScaler()
    
    # Reshape the data for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Create datasets and dataloaders
    train_dataset = SensorDataset(X_train_scaled, y_train)
    test_dataset = SensorDataset(X_test_scaled, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # Hyperparameters
    input_size = X.shape[2]
    hidden_size = 64
    num_layers = 2
    num_classes = 2 # Behavior 0 or Behavior 1
    learning_rate = 0.001
    num_epochs = 10
    
    # Initialize a list to store loss for plotting
    loss_history = []

    # Initialize model, loss function, and optimizer
    model = BehaviorDetector(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting model training...")
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Store the final loss for the epoch for visualization
        loss_history.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("\nTraining complete. Evaluating model on test data...")
    # Evaluation loop
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    # Calculate and print final accuracy and classification report
    final_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # --- Data Visualization: Plotting training loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show() 

    # --- Data Visualization: Plotting the Confusion Matrix ---
    # This shows the breakdown of correct and incorrect predictions for each class.
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Behavior 0', 'Behavior 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
 




