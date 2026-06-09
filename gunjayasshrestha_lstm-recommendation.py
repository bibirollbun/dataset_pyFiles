import numpy as np
import pandas as pd
import os

# List all files under the input directory, but ignore image files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.csv'):
            print(os.path.join(dirname, filename))


import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import pandas as pd
from pathlib import Path


# Define the dataset path
base_path = '../input/h-and-m-personalized-fashion-recommendations/'
transactions_path = f'{base_path}transactions_train.csv'
csv_sub = f'{base_path}sample_submission.csv'


# Load and Clean Data
def load_and_filter_data(transactions_path, start_date="2020-01-01", min_purchases=40):
    """
    Load transaction data, filter users with at least `min_purchases` and start_date, 
    and sort the data by customer_id and transaction date.
    """
    # Load data
    transactions = pd.read_csv(
        transactions_path,
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"]
    )
    
    # Filter transactions based on start date
    transactions = transactions[transactions["t_dat"] >= pd.Timestamp(start_date)]
    
    # Filter users with at least `min_purchases`
    user_purchase_counts = transactions.groupby("customer_id").size()
    filtered_users = user_purchase_counts[user_purchase_counts >= min_purchases].index
    filtered_data = transactions[transactions["customer_id"].isin(filtered_users)]
    
    # Sort data by customer_id and transaction date
    filtered_data = filtered_data.sort_values(by=["customer_id", "t_dat"])
    
    return filtered_data



# Group and Preprocess Purchase Histories
def preprocess_purchase_histories(filtered_data):
    """
    Group purchase histories by user and prepare sequences.
    """
    # Group article_ids by customer_id (sorted by t_dat)
    user_histories = filtered_data.groupby("customer_id")["article_id"].apply(list).tolist()
    
    # Ensure no empty histories
    user_histories = [history for history in user_histories if len(history) > 1]
    
    return user_histories


# Generate Sequences
def generate_sequences(histories, sequence_length):
    """
    Generate sliding window sequences from purchase histories.
    """
    input_sequences, target_items = [], []
    for history in histories:
        if len(history) > sequence_length:  # Ensure the sequence can be formed
            for i in range(len(history) - sequence_length):
                input_sequences.append(history[i:i + sequence_length])
                target_items.append(history[i + sequence_length])
    return input_sequences, target_items


# Encode Items to Indices
def encode_items(input_sequences, target_items, unique_items):
    """
    Convert article_ids to indices using a mapping.
    """
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    # Convert sequences and targets to indices
    input_sequences = [
        [item_to_idx[item] for item in seq if item in item_to_idx]
        for seq in input_sequences
    ]
    target_items = [item_to_idx[item] for item in target_items if item in item_to_idx]
    
    return input_sequences, target_items, item_to_idx, idx_to_item


# Split Data
def split_data(input_sequences, target_items, validation_ratio=0.2):
    """
    Split sequences and target items into training and validation sets.
    """
    dataset_size = len(input_sequences)
    validation_size = int(dataset_size * validation_ratio)

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)  # Shuffle indices to ensure randomness

    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train_sequences = [input_sequences[i] for i in train_indices]
    train_targets = [target_items[i] for i in train_indices]
    val_sequences = [input_sequences[i] for i in val_indices]
    val_targets = [target_items[i] for i in val_indices]

    return train_sequences, train_targets, val_sequences, val_targets


# Create PyTorch Dataset
class PurchaseDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }



# Define the Model
class LSTMRecommendationModel(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim):
        super(LSTMRecommendationModel, self).__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)  # Embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, num_items)  # Fully connected layer to predict items

    def forward(self, x):
        embeddings = self.embedding(x)  # Convert items to embeddings
        lstm_out, _ = self.lstm(embeddings)  # Process through LSTM
        output = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output for prediction
        return output


# Predict Function
def predict_next_item(model, sequence, idx_to_item):
    """
    Given a sequence of items, predict the next item.
    """
    model.eval()
    with torch.no_grad():
        sequence = torch.tensor([sequence], dtype=torch.long)  # Add batch dimension
        output = model(sequence)  # Get predictions
        predicted_idx = output.argmax(dim=1).item()  # Index with highest probability
        return idx_to_item[predicted_idx]


# Paths and hyperparameters
transactions_path = f'{base_path}transactions_train.csv'
sequence_length = 5
batch_size = 128
embedding_dim = 50
hidden_dim = 100
epochs = 10
validation_ratio = 0.2
early_stop_threshold = 0.5  # Stop training if validation loss drops below this value
learning_rate = 0.005

print("Step 1: Loading and preprocessing data...")
# Load and preprocess data
filtered_data = load_and_filter_data(
    transactions_path,
    start_date="2020-07-02", 
    min_purchases=40
)
print(f"Filtered data contains {len(filtered_data)} transactions from {filtered_data['customer_id'].nunique()} users.")

user_histories = preprocess_purchase_histories(filtered_data)
print(f"Generated purchase histories for {len(user_histories)} users.")

print("Step 2: Generating sequences...")
input_sequences, target_items = generate_sequences(user_histories, sequence_length)
print(f"Generated {len(input_sequences)} input sequences.")

print("Step 3: Encoding items...")
unique_items = set(filtered_data["article_id"])
input_sequences, target_items, item_to_idx, idx_to_item = encode_items(
    input_sequences, target_items, unique_items
)
print(f"Encoded {len(unique_items)} unique items.")

print("Step 4: Splitting data into training and validation sets...")
train_sequences, train_targets, val_sequences, val_targets = split_data(
    input_sequences, target_items, validation_ratio
)
print(f"Training set: {len(train_sequences)} sequences, Validation set: {len(val_sequences)} sequences.")

print("Step 5: Creating datasets and dataloaders...")
train_dataset = PurchaseDataset(train_sequences, train_targets)
val_dataset = PurchaseDataset(val_sequences, val_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Step 6: Initializing model, criterion, and optimizer...")
model = LSTMRecommendationModel(len(unique_items), embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)


print("Step 7: Starting training...")
train_losses = []
val_losses = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        batch_inputs = batch['sequence']
        batch_targets = batch['target']

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

        # Print progress for every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_inputs = batch['sequence']
            batch_targets = batch['target']

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

    # Reduce learning rate if no improvement
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < early_stop_threshold:
        print(f"Early stopping triggered. Validation loss {avg_val_loss:.4f} is below the threshold {early_stop_threshold}.")
        break

print("Training complete.")


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()


def recommend_items(model, user_sequence, idx_to_item, top_k=5):
    """
    Recommend top-k items based on the user's purchase sequence.
    
    Parameters:
    - model: Trained LSTM model
    - user_sequence: List of item indices representing the user's recent history
    - idx_to_item: Dictionary mapping item indices back to article IDs
    - top_k: Number of recommendations to generate
    
    Returns:
    - List of recommended article IDs
    """
    model.eval()
    with torch.no_grad():
        # Convert the user sequence to a tensor and ensure correct length
        input_seq = torch.tensor([user_sequence[-sequence_length:]], dtype=torch.long)
        
        # Get the model's predictions
        output = model(input_seq)
        
        # Get the top-k predictions
        top_indices = torch.topk(output, top_k).indices.squeeze().tolist()
        
        # Convert indices back to article IDs
        recommended_items = [idx_to_item[idx] for idx in top_indices]
    
    return recommended_items

# Example usage
user_history = train_sequences[0]  # Take one user's purchase history as an example
recommended_items = recommend_items(model, user_history, idx_to_item, top_k=5)

print("Recommended items:", recommended_items)


