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


# This is a placeholder - replace with actual download/access code
import os

data_dir = '/kaggle/input/stanford-rna-3d-folding/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")
    print("Please download the Stanford RNA 3D Folding dataset and place it in:", data_dir)
else:
    print(f"Data directory exists: {data_dir}")


def extract_sequence(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(lines[1:]).strip().upper().replace('T', 'U')
    return sequence

# Example usage (assuming a file named 'RNA_001.fasta' exists)
sequence_file = os.path.join(data_dir, 'RNA_001.fasta')
if os.path.exists(sequence_file):
    rna_sequence = extract_sequence(sequence_file)
    print(f"Extracted sequence: {rna_sequence}")


def parse_pdb(filepath):
    coords = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[17:20].strip() in ['A', 'U', 'G', 'C'] and line[12:16].strip() == "C3'":
                residue_number = int(line[22:26].strip())
                residue_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if residue_number not in coords:
                    coords[residue_number] = {'res_name': residue_name, 'coords': (x, y, z)}
    # Ensure coordinates are ordered by residue number
    sorted_coords = [coords[i]['coords'] for i in sorted(coords.keys())]
    return sorted_coords

# Example usage (assuming a file named 'RNA_001.pdb' exists)
pdb_file = os.path.join(data_dir, 'RNA_001.pdb')
if os.path.exists(pdb_file):
    c3_prime_coordinates = parse_pdb(pdb_file)
    print(f"Extracted C3' coordinates (first 5): {c3_prime_coordinates[:5]}")


import numpy as np
import torch

def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return torch.tensor(dist_matrix, dtype=torch.float32)

if os.path.exists(pdb_file):
    c3_prime_coords = parse_pdb(pdb_file)
    if c3_prime_coords:
        distance_matrix = calculate_distance_matrix(c3_prime_coords)
        print(f"Distance matrix shape: {distance_matrix.shape}")


def one_hot_encode_sequence(sequence):
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    encoded_sequence = [mapping[char] for char in sequence]
    encoded_sequence = torch.nn.functional.one_hot(torch.tensor(encoded_sequence), num_classes=4).float()
    return encoded_sequence

if os.path.exists(sequence_file):
    rna_sequence = extract_sequence(sequence_file)
    encoded_seq = one_hot_encode_sequence(rna_sequence)
    print(f"One-hot encoded sequence shape: {encoded_seq.shape}")


import os
from torch.utils.data import Dataset

class RNA3DDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.rna_ids = self._filter_complete_data(data_dir) # Filter in the constructor
        self.transform = transform

    def _filter_complete_data(self, data_dir):
        complete_ids = []
        for f in os.listdir(data_dir):
            rna_id_base = f.split('.')[0]
            if f.endswith('.fasta') or f.endswith('.seq'):
                seq_file = os.path.join(data_dir, f"{rna_id_base}.{'fasta' if os.path.exists(os.path.join(data_dir, f'{rna_id_base}.fasta')) else 'seq'}")
                pdb_file = os.path.join(data_dir, f"{rna_id_base}.pdb")
                if os.path.exists(seq_file) and os.path.exists(pdb_file):
                    complete_ids.append(rna_id_base)
        return list(set(complete_ids)) # Ensure unique IDs

    def __len__(self):
        return len(self.rna_ids)

    def __getitem__(self, idx):
        rna_id = self.rna_ids[idx]
        seq_file_ext = 'fasta' if os.path.exists(os.path.join(self.data_dir, f'{rna_id}.fasta')) else 'seq'
        seq_file = os.path.join(self.data_dir, f"{rna_id}.{seq_file_ext}")
        pdb_file = os.path.join(self.data_dir, f"{rna_id}.pdb")

        try:
            sequence = self._extract_sequence(seq_file)
            structure_data = self._parse_pdb(pdb_file)
            c3_prime_coords = structure_data.get('C3\'')
            if c3_prime_coords is None or len(sequence) != len(c3_prime_coords):
                print(f"Warning: Sequence/structure mismatch or missing C3' for {rna_id}")
                return None

            distance_matrix = self._calculate_distance_matrix(c3_prime_coords)
            sample = {'sequence': sequence, 'distance_matrix': distance_matrix}
            if self.transform:
                sample = self.transform(sample)
            return sample
        except FileNotFoundError:
            print(f"Warning: File not found for {rna_id}")
            return None

    def _extract_sequence(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            sequence = ''.join(lines[1:]).strip().upper().replace('T', 'U')
        return sequence

    def _parse_pdb(self, filepath):
        coords = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("ATOM") and line[17:20].strip() in ['A', 'U', 'G', 'C']:
                    atom_name = line[12:16].strip()
                    residue_number = int(line[22:26].strip())
                    residue_name = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if residue_number not in coords:
                        coords[residue_number] = {'res_name': residue_name, 'coords': {}}
                    coords[residue_number]['coords'][atom_name] = (x, y, z)
        # Reformat to get C3' coordinates in order
        c3_prime_coords_list = []
        sorted_residues = sorted(coords.keys())
        for res_num in sorted_residues:
            if 'C3\'' in coords[res_num]['coords']:
                c3_prime_coords_list.append(coords[res_num]['coords']['C3\''])
        return {'C3\'': c3_prime_coords_list}

    def _calculate_distance_matrix(self, coordinates):
        n = len(coordinates)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return torch.tensor(dist_matrix, dtype=torch.float32)


def _extract_sequence(self, filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(lines[1:]).strip().upper().replace('T', 'U')
        if not all(char in ['A', 'U', 'G', 'C'] for char in sequence):
            print(f"Warning: Sequence contains unusual characters in {filepath}. Skipping.")
            return None # Indicate invalid sequence
        return sequence

# Update __getitem__ to handle None return from _extract_sequence
def __getitem__(self, idx):
    if idx < 0 or idx >= len(self.file_list):
        raise IndexError(f"Index {idx} out of range")

    filename = self.file_list[idx]
    seq_file = os.path.join(self.data_dir, filename + self.seq_suffix)
    label_file = os.path.join(self.data_dir, filename + self.label_suffix)

    try:
        sequence = self._extract_sequence(seq_file)
        if sequence is None:
            return None # Skip this item if the sequence is invalid

        with open(label_file, 'r') as f:
            label = int(f.readline().strip())

        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            label = self.target_transform(label)

        return sequence, label

    except FileNotFoundError:
        print(f"Error: Sequence or label file not found for {filename}. Skipping.")
        return None


def _parse_pdb(self, filepath):
    coords = {}
    models = []
    current_model = None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model = []
            elif line.startswith("ENDMDL"):
                if current_model:
                    models.append(current_model)
                current_model = None
            elif current_model is not None and line.startswith("ATOM") and line[17:20].strip() in ['A', 'U', 'G', 'C']:
                current_model.append(line)
            elif current_model is None and line.startswith("ATOM") and line[17:20].strip() in ['A', 'U', 'G', 'C']:
                models.append([line]) # Handle single model PDBs

    if not models:
        return {'C3\'': None} # No valid ATOM records found

    # Use the first model
    atom_lines = models[0]
    residue_coords = {}
    for line in atom_lines:
        atom_name = line[12:16].strip()
        residue_number = int(line[22:26].strip())
        residue_name = line[17:20].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        if residue_number not in residue_coords:
            residue_coords[residue_number] = {'res_name': residue_name, 'coords': {}}
        residue_coords[residue_number]['coords'][atom_name] = (x, y, z)

    c3_prime_coords_list = []
    sorted_residues = sorted(residue_coords.keys())
    for res_num in sorted_residues:
        if 'C3\'' in residue_coords[res_num]['coords']:
            c3_prime_coords_list.append(residue_coords[res_num]['coords']['C3\''])
        else:
            print(f"Warning: Missing C3' atom in residue {res_num} of {filepath}")
            return {'C3\'': None} # Indicate missing C3'

    if len(c3_prime_coords_list) == 0:
        print(f"Warning: No C3' coordinates found in {filepath}")
        return {'C3\'': None}

    return {'C3\'': c3_prime_coords_list}

# Update __getitem__ to handle None return from _parse_pdb
def __getitem__(self, idx):
    rna_id = self.ids[idx]
    seq_file = os.path.join(self.seq_dir, f"{rna_id}.fasta")
    pdb_file = os.path.join(self.pdb_dir, f"{rna_id}.pdb")

    try:
        sequence = self._extract_sequence(seq_file)
        if sequence is None:
            return None
        structure_data = self._parse_pdb(pdb_file)
        c3_prime_coords = structure_data.get('C3\'')
        if c3_prime_coords is None or len(sequence) != len(c3_prime_coords):
            print(f"Warning: Sequence/structure mismatch or missing C3' for {rna_id}")
            return None

        # Create a list of coordinate tuples
        coords_list = [coord for coord in c3_prime_coords]
        # Convert sequence to numerical representation (if needed)
        numerical_sequence = [self.letter_to_int[base] for base in sequence]

        if self.transform:
            coords_list = self.transform(coords_list)
            numerical_sequence = self.transform(numerical_sequence) # Apply same transform if applicable

        return numerical_sequence, torch.tensor(np.array(coords_list), dtype=torch.float32)

    except FileNotFoundError:
        print(f"Warning: Sequence or PDB file not found for {rna_id}")
        return None
    except Exception as e:
        print(f"Error processing {rna_id}: {e}")
        return None


class RNA3DDataset(Dataset):
    def __init__(self, data_dir, min_len=10, max_len=500, transform=None):
        self.data_dir = data_dir
        self.min_len = min_len
        self.max_len = max_len
        self.rna_ids = self._filter_data_by_length(data_dir)
        self.transform = transform

    def _filter_data_by_length(self, data_dir):
        valid_ids = []
        for f in os.listdir(data_dir):
            rna_id_base = f.split('.')[0]
            if f.endswith('.fasta') or f.endswith('.seq'):
                seq_file = os.path.join(data_dir, f"{rna_id_base}.{'fasta' if os.path.exists(os.path.join(data_dir, f'{rna_id_base}.fasta')) else 'seq'}")
                pdb_file = os.path.join(data_dir, f"{rna_id_base}.pdb")
                if os.path.exists(seq_file) and os.path.exists(pdb_file):
                    sequence = self._extract_sequence_for_length_check(seq_file)
                    if sequence and self.min_len <= len(sequence) <= self.max_len:
                        valid_ids.append(rna_id_base)
        return list(set(valid_ids))

    def _extract_sequence_for_length_check(self, filepath):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                sequence = ''.join(lines[1:]).strip().upper().replace('T', 'U')
                return sequence
        except Exception:
            return None


import os
from sklearn.model_selection import train_test_split
import random

data_dir = "/kaggle/input/stanford-rna-3d-folding/MSA/"

# Assuming you have a list of RNA identifiers (e.g., filenames without extensions)
all_rna_ids = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.fasta')]
random.shuffle(all_rna_ids)

train_ids, temp_ids = train_test_split(all_rna_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

print(f"Number of training samples: {len(train_ids)}")
print(f"Number of validation samples: {len(val_ids)}")
print(f"Number of test samples: {len(test_ids)}")


import torch.nn as nn

class DistancePredictorCNN(nn.Module):
    def __init__(self, seq_len, num_filters=32, kernel_size=3):
        super(DistancePredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size, padding='same') # Input channels = 4 (one-hot)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding='same')
        # ... more convolutional layers ...
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * seq_len * seq_len, seq_len * seq_len) # Output is flattened distance matrix
        self.seq_len = seq_len

    def forward(self, x):
        x = self.relu(self.conv1(x.transpose(1, 2))) # Transpose for Conv1D
        x = self.relu(self.conv2(x))
        # ... more convolutional layers ...
        x = self.flatten(x.unsqueeze(-1).unsqueeze(-1)) # Prepare for FC layer
        x = self.fc(x)
        return x.view(self.seq_len, self.seq_len) # Reshape to distance matrix

# Example instantiation
example_seq_len = 100
distance_model = DistancePredictorCNN(example_seq_len)
print(distance_model)


class TorsionAnglePredictorRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4): # Example: 4 torsion angles per residue
        super(TorsionAnglePredictorRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Example instantiation
example_seq_len = 100
torsion_model = TorsionAnglePredictorRNN(input_size=4, hidden_size=64, output_size=4, num_layers=2)
print(torsion_model)


distance_criterion = nn.MSELoss()
torsion_criterion = nn.MSELoss() # Or other suitable regression loss


import torch.optim as optim

distance_optimizer = optim.Adam(distance_model.parameters(), lr=0.001)
torsion_optimizer = optim.Adam(torsion_model.parameters(), lr=0.001)


def train_distance_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            if batch:
                sequences = torch.stack([item['sequence'] for item in batch])
                distance_matrices = torch.stack([item['distance_matrix'] for item in batch])

                optimizer.zero_grad()
                predictions = model(sequences)
                loss = criterion(predictions, distance_matrices)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Example training loop (assuming you have a DataLoader named 'train_loader')
# train_distance_model(distance_model, train_loader, distance_criterion, distance_optimizer)


from torch.utils.data import DataLoader

def evaluate_distance_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch:
                sequences = torch.stack([item['sequence'] for item in batch])
                distance_matrices = torch.stack([item['distance_matrix'] for item in batch])
                predictions = model(sequences)
                loss = criterion(predictions, distance_matrices)
                total_loss += loss.item()
    return total_loss / len(dataloader)

# Example of a simple hyperparameter search (you'd likely use a more systematic approach)
learning_rates = [0.001, 0.0005]
num_filters_options = [16, 32]

best_val_loss = float('inf')
best_params = None


from typing import List
class RNADistanceDataset(Dataset):
    def __init__(self, sequences: List[str], distance_matrices: List[torch.Tensor]):
        self.sequences = sequences
        self.distance_matrices = distance_matrices
        self.mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.seq_len = max(len(seq) for seq in sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        distance_matrix = self.distance_matrices[idx]
        encoded_sequence = torch.zeros(self.seq_len, 4)
        for i, base in enumerate(sequence):
            encoded_sequence[i, self.mapping[base]] = 1
        return {'sequence': encoded_sequence, 'distance_matrix': distance_matrix}


def calculate_rmsd(predicted_coords, true_coords):
    # Implementation of RMSD calculation
    # This requires aligning the predicted and true structures
    pass # Replace with actual RMSD calculation

def evaluate_structure_prediction(model, dataloader):
    model.eval()
    all_rmsds = []
    with torch.no_grad():
        for batch in dataloader:
            if batch:
                sequences = [item['sequence'] for item in batch]
                # Assuming your model predicts coordinates directly or something from which coords can be derived
                # true_coords_list = [item['coordinates'] for item in batch]
                predicted_outputs = model(torch.stack(sequences))
                for i in range(len(batch)):
                     predicted_coords = [] # Derive coordinates from model output
                     for box in raw_output:
                            x_min, y_min, x_max, y_max = box
                            center_x = (x_min + x_max) / 2
                            center_y = (y_min + y_max) / 2
                            predicted_coords.append((center_x, center_y))

                     true_coords = true_coords_list[i]
                     if predicted_coords is not None and len(predicted_coords) == len(true_coords):
                         rmsd = calculate_rmsd(predicted_coords, true_coords)
                         all_rmsds.append(rmsd)
    if all_rmsds:
        print(f"Average RMSD on test set: {np.mean(all_rmsds)}")
    else:
        print("No valid predictions for RMSD calculation.")

# Assuming you have a test DataLoader named 'test_loader'
# evaluate_structure_prediction(best_distance_model, test_loader) # If your model predicts something from which structure can be derived


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # Example for distance matrix

def analyze_errors_distance_matrix(model, dataloader, device="cpu", num_samples=5):
    model.eval()
    error_examples = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            if batch:
                sequences = torch.stack([item['sequence'] for item in batch]).to(device)
                true_distance_matrices = torch.stack([item['distance_matrix'] for item in batch]).to(device)
                rna_ids = [item['id'] for item in batch] # Assuming you included IDs in your dataset

                predicted_distance_matrices = model(sequences)

                for j in range(len(batch)):
                    true_dm = true_distance_matrices[j].cpu().numpy()
                    pred_dm = predicted_distance_matrices[j].cpu().numpy()
                    mse = mean_squared_error(true_dm.flatten(), pred_dm.flatten())
                    error_examples.append({'id': rna_ids[j], 'true': true_dm, 'predicted': pred_dm, 'mse': mse})

    # Sort by error for visualization
    error_examples.sort(key=lambda x: x['mse'], reverse=True)

    for example in error_examples:
        print(f"RNA ID: {example['id']}, MSE: {example['mse']:.4f}")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(example['true'], cmap='viridis')
        plt.title('True Distance Matrix')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(example['predicted'], cmap='viridis')
        plt.title('Predicted Distance Matrix')
        plt.colorbar()
        plt.show()

# Assuming you have a test_loader and your model is 'best_distance_model'
# and your dataset returns 'id' in each item
# analyze_errors_distance_matrix(best_distance_model, test_loader, device)


pip install biopython


from Bio import SeqIO
# Install RNAfold if not already: conda install -c bioconda rnafold

def predict_secondary_structure(sequence):
    import subprocess
    try:
        process = subprocess.run(['RNAfold', '-i'], input=sequence.encode('utf-8'), capture_output=True, text=True, check=True)
        output = process.stdout.strip().split('\n')[1] # Assuming dot-bracket notation is on the second line
        return output.split()[0]
    except subprocess.CalledProcessError as e:
        print(f"RNAfold error: {e}")
        return None

class EnhancedRNA3DDataset(RNA3DDataset): # Inherit from your previous dataset class
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if sample is None:
            return None

        sequence = sample['sequence']
        # Predict secondary structure
        secondary_structure = predict_secondary_structure(sequence)
        if secondary_structure:
            # Encode secondary structure (e.g., one-hot for each position: paired, unpaired)
            secondary_features = self._encode_secondary_structure(secondary_structure)
            sample['secondary_structure'] = secondary_features
        else:
            sample['secondary_structure'] = torch.zeros(len(sequence), 2) # Example: all unpaired if prediction fails

        # Add other features here if needed
        return sample

    def _encode_secondary_structure(self, ss):
        encoding = []
        for char in ss:
            if char == '.':
                encoding.append([1, 0]) # Unpaired
            elif char in '()[]{}:<>': # Paired (can distinguish different types if needed)
                encoding.append([0, 1]) # Paired
            else:
                encoding.append([0.5, 0.5]) # Unknown
        return torch.tensor(encoding, dtype=torch.float32)

# Example usage:
# enhanced_dataset = EnhancedRNA3DDataset(data_dir, transform=YourExistingTransform)
# enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: [item for item in x if item is not None])


import torch.nn as nn

# Example: Adding a Bidirectional LSTM layer to a CNN-based model
class HybridCNNLSTM(nn.Module):
    def __init__(self, seq_len, num_filters=32, kernel_size=3, lstm_hidden=64, lstm_layers=2):
        super(HybridCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size, padding='same')
        self.relu = nn.ReLU()
        # ... more CNN layers ...
        self.lstm = nn.LSTM(num_filters, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2 * seq_len, seq_len * seq_len) # Adjust output size

    def forward(self, x):
        x = self.relu(self.conv1(x.transpose(1, 2)))
        # ... more CNN layers ...
        x = x.transpose(1, 2) # Prepare for LSTM (batch, seq, features)
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1) # Flatten for FC
        out = self.fc(out)
        return out.view(out.size(0), self.seq_len, self.seq_len)

# Example: Using a Transformer Encoder
class TransformerDistancePredictor(nn.Module):
    def __init__(self, seq_len, num_heads=4, num_layers=2, d_model=64):
        super(TransformerDistancePredictor, self).__init__()
        self.embedding = nn.Linear(4, d_model) # Project one-hot to d_model
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * seq_len, seq_len * seq_len)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded.transpose(0, 1)).transpose(0, 1) # (batch, seq, d_model)
        flattened = encoded.reshape(encoded.size(0), -1)
        out = self.fc(flattened)
        return out.view(out.size(0), x.size(1), x.size(1))

# ... instantiate and train the new models ...


import torch
import torch.nn as nn

# Example: Using a custom loss that penalizes long-range distance errors more
class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, predicted, target):
        mse = (predicted - target)**2
        # Create a weight matrix where long-range distances have higher weights
        weights = torch.ones_like(target)
        n = target.size(-1)
        for i in range(n):
            for j in range(i + 1, n):
                distance = abs(i - j)
                if distance > n // 2: # Example threshold for "long-range"
                    weights[:, i, j] *= self.alpha
                    weights[:, j, i] *= self.alpha
        return (mse * weights).mean()

# Example: Using a loss that encourages specific contact patterns (requires defining contacts)
class ContactMapLoss(nn.Module):
    def __init__(self, threshold=8.0): # Distance threshold for contact
        super(ContactMapLoss, self).__init__()
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss() # Binary Cross-Entropy for contact prediction

    def forward(self, predicted_distance_matrix):
        # Convert distance matrix to contact probability (sigmoid or similar)
        contact_probabilities = torch.sigmoid(-predicted_distance_matrix) # Closer = higher prob

        # Generate "ground truth" contact map from true coordinates (if available in the batch)
        # This part is complex and depends on how your data is structured
        # true_contact_map = ...

        # if true_contact_map is not None:
        #     return self.bce(contact_probabilities, true_contact_map.float())
        # else:
        return torch.tensor(0.0, requires_grad=True) # Placeholder if no true contacts

# ... replace your loss function in the training loop ...
# distance_criterion = WeightedMSELoss(alpha=2.0)
# distance_criterion = ContactMapLoss(threshold=8.0)


import numpy as np
import torch

class CoordinatePerturbation(object):
    def __init__(self, max_translation=0.1):
        self.max_translation = max_translation

    def __call__(self, sample):
        if 'coordinates' in sample:
            coords = np.array(sample['coordinates'])
            translation = np.random.uniform(-self.max_translation, self.max_translation, size=coords.shape)
            perturbed_coords = coords + translation
            sample['coordinates'] = perturbed_coords.tolist()
            # Recalculate distance matrix if you are predicting that
            if 'distance_matrix' in sample:
                n = len(perturbed_coords)
                dist_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.linalg.norm(perturbed_coords[i] - perturbed_coords[j])
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist
                sample['distance_matrix'] = torch.tensor(dist_matrix, dtype=torch.float32)
        return sample

# Apply this transform during training data loading
# train_dataset = RNA3DDataset(train_data_dir, transform=transforms.Compose([OneHotEncode(), CoordinatePerturbation()]))
# train_loader = DataLoader(train_dataset, ...)

