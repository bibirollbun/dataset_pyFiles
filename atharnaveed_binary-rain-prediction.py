import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df


df.isnull().sum()


df = df.drop(columns=["id"])


x = df.loc[:,df.columns != "rainfall"]
y = df["rainfall"]


print(f"x:{x}")
print(f"y: {y}")


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


# Create DataLoader for batch training
batch_size = 128
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class RainPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2,dropout=0.3):
        super(RainPredictionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # Multiply by 2 because of bidirectional LSTM
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Add sequence dimension (LSTM expects 3D input: batch, sequence, features)
        x = x.unsqueeze(1)  # Shape: (batch, sequence_length=1, features)

        lstm_out, _ = self.lstm(x)  # LSTM output
        last_time_step_output = lstm_out[:, -1, :]  # Extract last output
        
        return self.fc_layers(last_time_step_output)



# Initialize model
input_size = x_train.shape[1]  # Number of features
hidden_size = 50
num_layers = 4

model = RainPredictionLSTM(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.AdamW(model.parameters(), lr=0.001)



num_epochs = 150  # You can adjust this

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Reset gradients
        
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")



from sklearn.metrics import roc_auc_score

# Set model to evaluation mode
model.eval()
y_probs = []  # Stores probability scores
y_true = []  # Stores actual labels

with torch.no_grad():  # No need to calculate gradients
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Get predicted probabilities
        y_probs.extend(outputs.numpy())  # Convert to numpy and store
        y_true.extend(labels.numpy())  # Store actual labels

# Convert to numpy arrays
y_probs = np.array(y_probs)
y_true = np.array(y_true)

# Compute AUC-ROC Score
auc_score = roc_auc_score(y_true, y_probs)
print(f"AUC-ROC Score: {auc_score:.4f}")



test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test_df


test_df.isnull().sum()


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mode()[0])
test_df['winddirection'].unique()


test_ids = test_df["id"]
test_df = test_df.drop(columns=['id'])


# Scale the test data
test_df_scaled = scaler.transform(test_df)
test_tensor = torch.tensor(test_df_scaled, dtype=torch.float32)  # Shape: (num_test_samples, input_size)


test_dataset = TensorDataset(test_tensor)  # No labels needed for prediction
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Ensure the model is in evaluation mode
model.eval()

predictions = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch[0]  # Shape: (batch_size, input_size)
        outputs = model(inputs)  # Forward will add sequence dimension
        predictions.extend(outputs.squeeze().numpy())

# Convert predictions to a numpy array
predictions = np.array(predictions)


# Ensure predictions is a NumPy array and flatten it
predictions = predictions.flatten()

# Apply threshold (0.5) to convert probabilities to binary labels
binary_predictions = (predictions >= 0.5).astype(int)



# Create a DataFrame with 'id' and 'rainfall'
submission = pd.DataFrame({
    "id": test_ids,  # Ensure test_df contains the 'id' column
    "rainfall": predictions  # Use probabilities or binary_predictions based on requirement
})

# Save as CSV
submission.to_csv("submission.csv", index=False)

print("Submission file saved!")





