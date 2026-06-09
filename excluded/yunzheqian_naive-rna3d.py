# Import libraries and set up paths
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm.notebook import tqdm
import yaml
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Define paths
COMPETITION_PATH = "/kaggle/input/stanford-rna-3d-folding"
WORKSPACE_PATH = "/kaggle/input/ribonanzanet3d/ribonanzanet3D"
MODEL_DIR = f"{WORKSPACE_PATH}/models"
OUTPUT_DIR = f"{WORKSPACE_PATH}/output"

# Add model path to Python path
sys.path.append(f"{MODEL_DIR}/ribonanzanet2d-final")

# Show directory structure
print("Working directory contents:")
!ls -la {WORKSPACE_PATH}
print("\nModel directory contents:")
!ls -la {MODEL_DIR}
print("\nOutput directory contents:")
!ls -la {OUTPUT_DIR}


# Define model architecture
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

# Check if Network.py exists
if not os.path.exists(f"{MODEL_DIR}/ribonanzanet2d-final/Network.py"):
    print("Network.py not found. Please create it using the %%writefile magic command or upload it.")
else:
    try:
        from Network import RibonanzaNet
        print("Successfully imported RibonanzaNet!")
        
        class finetuned_RibonanzaNet(RibonanzaNet):
            def __init__(self, config, pretrained=False):
                config.dropout = 0.1
                super(finetuned_RibonanzaNet, self).__init__(config)
                self.dropout = nn.Dropout(0.0)
                self.xyz_predictor = nn.Linear(256, 3)
            
            def forward(self, src):
                sequence_features, pairwise_features = self.get_embeddings(
                    src, torch.ones_like(src).long().to(src.device))
                xyz = self.xyz_predictor(sequence_features)
                return xyz
    except Exception as e:
        print(f"Error importing RibonanzaNet: {e}")


# Load test data
test_data = pd.read_csv(f"{COMPETITION_PATH}/test_sequences.csv")
print(f"Loaded {len(test_data)} test sequences")

# Define dataset class
class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokens = {nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = [self.tokens[nt] for nt in (self.data.loc[idx, 'sequence'])]
        sequence = np.array(sequence)
        sequence = torch.tensor(sequence)
        return {'sequence': sequence}

# Create test dataset
test_dataset = RNADataset(test_data)
print(f"Example sequence length: {len(test_dataset[0]['sequence'])}")


# Load model
# Check if config file exists
config_file = f"{MODEL_DIR}/ribonanzanet2d-final/configs/pairwise.yaml"
if not os.path.exists(config_file):
    print(f"Config file not found at {config_file}")
    # Create a basic config if needed
    basic_config = {
        "n_tokens": 4,
        "d_model": 256,
        "d_ff": 1024,
        "n_layers": 9,
        "n_heads": 8,
        "dropout": 0.1,
        "max_len": 384
    }
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        yaml.dump(basic_config, f)
    print(f"Created basic config file at {config_file}")

# Check if model weights exist
model_file = f"{OUTPUT_DIR}/RibonanzaNet-3D.pt"
if not os.path.exists(model_file):
    print(f"Model file not found at {model_file}")
    print("Please upload the model weights file")
else:
    print(f"Found model weights at {model_file}")

try:
    # Initialize model
    config = load_config_from_yaml(config_file)
    model = finetuned_RibonanzaNet(config, pretrained=False)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    # Load pre-trained weights
    model.load_state_dict(torch.load(model_file, 
                                    map_location="cuda" if torch.cuda.is_available() else "cpu"))
    print("Model loaded successfully!")

    # Generate predictions
    model.eval()
    preds = []
    print("Generating predictions...")

    for i in tqdm(range(len(test_dataset))):
        src = test_dataset[i]['sequence'].long()
        src = src.unsqueeze(0)
        if torch.cuda.is_available():
            src = src.cuda()
        
        # Generate 5 predictions per sequence
        tmp = []
        # First 4 with dropout enabled (stochastic)
        model.train()
        for j in range(4):
            with torch.no_grad():
                xyz = model(src).squeeze()
            tmp.append(xyz.cpu().numpy())
        
        # Last one without dropout
        model.eval()
        with torch.no_grad():
            xyz = model(src).squeeze()
        tmp.append(xyz.cpu().numpy())
        
        tmp = np.stack(tmp, 0)
        preds.append(tmp)

    print(f"Generated predictions for {len(preds)} sequences")

    # Format submission
    print("Formatting submission...")
    data = []
    
    for i in range(len(test_data)):
        for j in range(len(test_data.loc[i, 'sequence'])):
            row = [
                test_data.loc[i, 'target_id'] + f"_{j+1}",
                test_data.loc[i, 'sequence'][j],
                j+1  # 1-indexed
            ]
            
            for k in range(5):  # 5 predictions
                for coord in range(3):  # x, y, z
                    row.append(preds[i][k][j][coord])
            
            data.append(row)
    
    columns = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        columns += [f"x_{i}", f"y_{i}", f"z_{i}"]
    
    submission = pd.DataFrame(data, columns=columns)
    submission.to_csv('submission.csv', index=False)
    print("Submission saved!")
    
    # Display first few rows
    submission.head()
    
except Exception as e:
    print(f"Error in model loading or inference: {e}")

