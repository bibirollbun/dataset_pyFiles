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


!pip install biopython


import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import numpy as np

# File Paths
train_seq_path = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
val_seq_path = "/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv"
val_labels_path = "/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
msa_dir = "/kaggle/input/stanford-rna-3d-folding/MSA"

# Load sequences
train_seqs = pd.read_csv(train_seq_path)
train_labels = pd.read_csv(train_labels_path)
val_seqs = pd.read_csv(val_seq_path)
val_labels = pd.read_csv(val_labels_path)

# Map RNA bases to numerical values
rna_vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '-': 4}  # Adding '-' as a special case

def encode_sequence(seq):
    return [rna_vocab.get(nt, 4) for nt in seq]  # Use `.get()` to avoid KeyError

# Convert sequences to numerical format
train_seqs['encoded_seq'] = train_seqs['sequence'].apply(encode_sequence)
val_seqs['encoded_seq'] = val_seqs['sequence'].apply(encode_sequence)

print("Loaded Training & Validation Sequences!")


import glob

def load_msa(msa_path):
    sequences = []
    with open(msa_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequences.append(str(record.seq))
    return sequences

# Load MSA features
msa_files = glob.glob(os.path.join(msa_dir, "*.fasta"))
msa_dict = {os.path.basename(f): load_msa(f) for f in msa_files}

print(f"Loaded {len(msa_files)} MSA files!")


class RNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences.iloc[idx]['encoded_seq'], dtype=torch.long)
        label = torch.tensor(self.labels.iloc[idx, 1:].values, dtype=torch.float)  # Exclude ID
        return seq, label

# Create dataset objects
train_dataset = RNADataset(train_seqs, train_labels)
val_dataset = RNADataset(val_seqs, val_labels)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Dataloaders Ready!")


import torch
import torch.nn as nn
import torch.nn.functional as F

class RNA3DModel(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=256, num_heads=8, num_layers=6, dropout=0.3, hidden_dim=512):
        """
        Initialize the RNA3DModel.

        Parameters:
            vocab_size (int): The size of the vocabulary (RNA bases, typically 4: A, C, G, U)
            embed_dim (int): The dimension of the embedding for each RNA base
            num_heads (int): Number of heads in the multi-head attention mechanism
            num_layers (int): Number of transformer encoder layers
            dropout (float): Dropout probability to avoid overfitting
            hidden_dim (int): The dimension of the hidden layer in the feed-forward networks of the transformer
        """
        super(RNA3DModel, self).__init__()

        # Embedding Layer for RNA sequence (transforming nucleotides to vectors)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,  # Dimension of model (embedding size)
                nhead=num_heads,  # Number of attention heads
                dim_feedforward=hidden_dim,  # Feed-forward hidden dimension
                dropout=dropout,  # Dropout probability
                batch_first=True  # Set batch_first=True for easier batch handling
            ),
            num_layers=num_layers  # Number of transformer layers
        )

        # Graph Convolution-like Fully Connected Layers
        self.gcn1 = nn.Linear(embed_dim, 128)  # First fully connected layer
        self.gcn2 = nn.Linear(128, 64)  # Second fully connected layer

        # Output Layer (predicts 3D coordinates)
        self.output_layer = nn.Linear(64, 3)  # Output dimension is 3 for x, y, z coordinates

    def forward(self, src):
        """
        The forward pass of the model.

        Parameters:
            src (Tensor): The input RNA sequence tensor with shape (batch_size, sequence_length)

        Returns:
            Tensor: Predicted 3D coordinates of shape (batch_size, sequence_length, 3)
        """
        # Convert sequence into embeddings
        x = self.embedding(src)

        # Pass through transformer layers
        x = self.encoder(x)

        # Apply graph convolution layers (fully connected layers)
        x = F.relu(self.gcn1(x))  # First graph layer with ReLU activation
        x = F.relu(self.gcn2(x))  # Second graph layer with ReLU activation

        # Predict 3D coordinates (x, y, z) for each sequence
        xyz = self.output_layer(x)

        return xyz

# Instantiate model and move to GPU (if available)
model = RNA3DModel().cuda() if torch.cuda.is_available() else RNA3DModel()
print(model)

