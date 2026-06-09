import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import csv



# Define the dataset path
data_path = "/kaggle/input/stanford-rna-3d-folding"

# Load datasets
train_sequences = pd.read_csv(data_path + "/train_sequences.csv")
train_labels = pd.read_csv(data_path + "/train_labels.csv")
validation_sequences = pd.read_csv(data_path + "/validation_sequences.csv")
validation_labels = pd.read_csv(data_path + "/validation_labels.csv")
test_sequences = pd.read_csv(data_path + "/test_sequences.csv")

print("Train Sequences:")
print(train_sequences.head())

print("\nTrain Labels:")
print(train_labels.head())

print("\nValidation Sequences:")
print(validation_sequences.head())

print("\nValidation Labels:")
print(validation_labels.head())

print("\nTest Sequences:")
print(test_sequences.head())



# Map nucleotides to integers (A=0, C=1, G=2, U=3)
nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

def encode_sequence(sequence):
    # Map valid nucleotides; assign -1 for invalid characters (if any)
    return [nucleotide_map.get(n, -1) for n in sequence]

# Encode RNA sequences for training, validation, and test datasets
train_sequences['encoded_seq'] = train_sequences['sequence'].apply(encode_sequence)
validation_sequences['encoded_seq'] = validation_sequences['sequence'].apply(encode_sequence)
test_sequences['encoded_seq'] = test_sequences['sequence'].apply(encode_sequence)

print("\nEncoded Train Sequences:")
print(train_sequences.head())



# Drop non-numerical columns from train_labels and validation_labels
train_labels_numerical = train_labels.drop(columns=['ID', 'resname', 'resid'])
validation_labels_numerical = validation_labels.drop(columns=['ID', 'resname', 'resid'])

# Ensure all remaining columns are numeric
train_labels_numerical = train_labels_numerical.apply(pd.to_numeric, errors='coerce')
validation_labels_numerical = validation_labels_numerical.apply(pd.to_numeric, errors='coerce')

# Fill missing values (if any) with a default value (e.g., 0.0)
train_labels_numerical = train_labels_numerical.fillna(0.0)
validation_labels_numerical = validation_labels_numerical.fillna(0.0)

print("\nTrain Labels (Numerical):")
print(train_labels_numerical.head())

print("\nValidation Labels (Numerical):")
print(validation_labels_numerical.head())



class RNADataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Get encoded sequence and its length
        seq = torch.tensor(self.sequences.iloc[idx]['encoded_seq'], dtype=torch.long)
        seq_len = len(seq)
        
        if self.labels is not None:
            # Use only numerical columns for labels
            coords = torch.tensor(self.labels.iloc[idx].values.astype(float), dtype=torch.float32)
            return seq, coords, seq_len
        else:
            return seq, seq_len

# Create datasets for training and validation using numerical labels
train_dataset = RNADataset(train_sequences, train_labels_numerical)
val_dataset = RNADataset(validation_sequences, validation_labels_numerical)

# Create data loaders with custom collate function for padding sequences
def collate_fn(batch):
    sequences, coords, lengths = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  # Padding with 0
    coords = torch.stack(coords)  # Stack coordinates (no need for padding)
    lengths = torch.tensor(lengths)  # Tensor of sequence lengths
    return padded_sequences, coords, lengths

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

print("Data loaders created successfully!")



class RNA3DModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super(RNA3DModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)  # Embed nucleotide sequences
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # Predict x, y, z coordinates

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Initialize the model
model = RNA3DModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for sequences, coords, lengths in train_loader:
            sequences, coords = sequences.to(device), coords.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)

            # Mask out padded positions using sequence lengths
            max_len = outputs.size(1)  # Maximum sequence length in batch
            mask = torch.arange(max_len).unsqueeze(0).to(device) < lengths.unsqueeze(1)

            outputs_masked = outputs[mask]
            coords_masked = coords.view(-1, 3)[mask.view(-1)]

            loss = criterion(outputs_masked.view(-1, 3), coords_masked.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = validate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

def validate_model(model_, val_loader_):
    model_.eval()
    val_loss_agg_=[]



def generate_predictions(model, test_sequences):
    model.eval()
    predictions = []

    with torch.no_grad():
        for _, row in test_sequences.iterrows():
            seq_id = row['target_id']
            sequence_encoded = torch.tensor(row['encoded_seq'], dtype=torch.long).unsqueeze(0).to(device)

            pred_coords_list = []
            for _ in range(5):  # Generate five predictions per sequence
                pred_coords_list.append(model(sequence_encoded).cpu().numpy().flatten())

            predictions.append([seq_id] + np.concatenate(pred_coords_list).tolist())

    return predictions

predictions = generate_predictions(model, test_sequences)



submission_file = "submission.csv"

# Define header fields based on competition requirements
header = ["ID", "resname", "resid"] + [f"x_{i},y_{i},z_{i}" for i in range(1, 6)]

# Generate predictions and save them to submission.csv
with open(submission_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(header)
    
    # Write predictions (ensure each row matches header format)
    for pred in predictions:
        # Flatten predictions into a single row (ensure correct number of fields)
        writer.writerow(pred)

print(f"Submission file saved to {submission_file}")



import csv

# Define expected header format
expected_header = ["ID", "resname", "resid"] + [f"x_{i},y_{i},z_{i}" for i in range(1, 6)]

# Read and validate submission file
with open("submission.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read header
    rows = list(reader)  # Read all rows

# Validate header
header_matches = header == expected_header

# Validate row lengths
correct_row_lengths = all(len(row) == len(expected_header) for row in rows)

print("Header Matches Expected Format:", header_matches)
print("All Rows Have Correct Number of Fields:", correct_row_lengths)
print("Number of Rows in Submission File:", len(rows))


