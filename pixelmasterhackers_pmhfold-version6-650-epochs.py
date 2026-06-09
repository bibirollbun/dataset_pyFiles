# RNADataLoader Class
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class RNADataLoader:
    def __init__(self):
        self.nucleotide_encoder = LabelEncoder()
        self.nucleotide_encoder.fit(['A', 'U', 'G', 'C'])
        
    def one_hot_encode(self, sequence):
        """Convert RNA sequence to one-hot encoding"""
        encoded = np.zeros((len(sequence), 4))
        for i, nt in enumerate(sequence):
            if nt in self.nucleotide_encoder.classes_:
                encoded[i, self.nucleotide_encoder.transform([nt])[0]] = 1
            else:
                # Handle unexpected characters (e.g., '-', 'N') by skipping or encoding as zeros
                print(f"Warning: Unexpected nucleotide '{nt}' encountered. Encoding as zeros.")
        return encoded
        
    def load_train_data(self, seq_file, labels_file):
        """Load training data from CSV files"""
        sequences_df = pd.read_csv(seq_file)
        labels_df = pd.read_csv(labels_file)
        
        X = []
        y = []
        
        for _, row in sequences_df.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']
            
            # Get coordinates for this sequence
            seq_labels = labels_df[labels_df['ID'].str.startswith(target_id)]  # Use "ID" for labels
            
            if len(seq_labels) > 0:
                # Extract coordinates
                coords = seq_labels[['x_1', 'y_1', 'z_1']].values
                # Mask invalid values
                coords[np.abs(coords) > 1e6] = 0.0  # or np.nan, but 0.0 is safe for masking
                # Mask NaNs
                coords[np.isnan(coords)] = 0.0
                # One-hot encode the sequence
                encoded_seq = self.one_hot_encode(sequence)
                
                X.append(encoded_seq)
                y.append(coords)
        
        # Convert to arrays with padding to ensure all sequences have same length
        max_length = max(len(x) for x in X)
        X_padded = np.zeros((len(X), max_length, 4))
        y_padded = np.zeros((len(y), max_length, 3))  # Ensure same max_length

        for i, (x, coords) in enumerate(zip(X, y)):
            X_padded[i, :len(x)] = x
            y_padded[i, :len(coords)] = coords
            
        print(f"X shape: {X_padded.shape}, y shape: {y_padded.shape}")
        return X_padded, y_padded, sequences_df['target_id'].values
    
    def load_test_data(self, seq_file):
        """Load test data from CSV file"""
        sequences_df = pd.read_csv(seq_file)
        
        X = []
        sequence_lengths = []
        
        for _, row in sequences_df.iterrows():
            sequence = row['sequence']
            encoded_seq = self.one_hot_encode(sequence)
            X.append(encoded_seq)
            sequence_lengths.append(len(sequence))
        
        # Pad sequences to max length
        max_length = max(len(x) for x in X)
        X_padded = np.zeros((len(X), max_length, 4))
        
        for i, x in enumerate(X):
            X_padded[i, :len(x)] = x
            
        return X_padded, sequence_lengths, sequences_df['target_id'].values


import torch
import torch.nn as nn
import torch.nn.functional as F

def create_base_pairing_mask(x):
    """Create a mask for valid RNA base pairs following established folding rules
    References:
    - Mathews et al. (2004) "Incorporating chemical modification constraints into a dynamic programming algorithm for prediction of RNA secondary structure"
    - Zuker & Sankoff (1984) "RNA secondary structures and their prediction"
    """
    batch_size, seq_len, _ = x.shape
    
    # Extract base type positions: [A,C,G,U]
    A = x[:, :, 0].unsqueeze(2)
    C = x[:, :, 1].unsqueeze(2)
    G = x[:, :, 2].unsqueeze(2)
    U = x[:, :, 3].unsqueeze(2)
    
    # Create pairing masks with different strengths
    # Strong canonical pairs
    A_U = torch.matmul(A, U.transpose(1,2)) * 0.9  # A-U pairs (strong)
    U_A = torch.matmul(U, A.transpose(1,2)) * 0.9  # U-A pairs (strong)
    G_C = torch.matmul(G, C.transpose(1,2))  # G-C pairs (strongest)
    C_G = torch.matmul(C, G.transpose(1,2))  # C-G pairs (strongest)
    
    # Weak wobble pairs (G-U)
    G_U = torch.matmul(G, U.transpose(1,2)) * 0.7  # G-U pairs (weaker)
    U_G = torch.matmul(U, G.transpose(1,2)) * 0.7  # U-G pairs (weaker)
    
    # Combine all valid pairs with their respective strengths
    pairing_mask = G_C + C_G + A_U + U_A + G_U + U_G
    
    # Create smooth distance penalty based on loop size constraints
    positions = torch.arange(seq_len, device=x.device)
    distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
    
    # Smooth penalty function:
    # - Distance < 3: prohibited (physical constraint)
    # - Distance 3-4: suboptimal but allowed
    # - Distance 4-7: optimal hairpin size
    # - Distance > 7: allowed but slightly penalized for long-range interactions
    distance_penalty = torch.zeros_like(distances, dtype=torch.float)
    distance_penalty = torch.where(distances < 3, torch.zeros_like(distances, dtype=torch.float), distance_penalty)
    distance_penalty = torch.where((distances >= 3) & (distances < 4), 0.5 * torch.ones_like(distances, dtype=torch.float), distance_penalty)
    distance_penalty = torch.where((distances >= 4) & (distances <= 7), torch.ones_like(distances, dtype=torch.float), distance_penalty)
    distance_penalty = torch.where(distances > 7, 0.8 * torch.ones_like(distances, dtype=torch.float), distance_penalty)
    
    # Apply distance penalty and add small baseline for non-pairs
    pairing_mask = pairing_mask * distance_penalty + 0.1
    
    return pairing_mask

def compute_distance_violations(coords, pairing_mask, min_distance=3.0, 
                             wc_pair_distance=5.9, backbone_distance=6.0):
    """
    Compute distance violation penalties based on RNA structural constraints
    References:
    - Watson-Crick pair distance ~5.9Å (Regions et al. 2011)
    - P-P backbone distance ~6.0Å (Richardson et al. 2008)
    - Minimum allowed distance 3.0Å (van der Waals + water shell)
    """
    batch_size, seq_length, _ = coords.size()
    
    # Compute all pairwise distances for each sequence in batch
    # Shape: (batch_size, seq_length, seq_length)
    distances = torch.cdist(coords, coords, p=2)
    
    # 1. Minimum distance violation (steric clash)
    min_dist_violation = torch.relu(min_distance - distances)
    identity_mask = torch.eye(seq_length, device=coords.device).unsqueeze(0)
    min_dist_violation = min_dist_violation * (1 - identity_mask)  # Exclude self-distances
    
    # 2. Base pair distance violation (for paired bases)
    # Ensure pairing_mask is expanded if needed
    if len(pairing_mask.shape) == 2:
        pairing_mask = pairing_mask.unsqueeze(0).expand(batch_size, -1, -1)
    pair_dist_violation = torch.abs(distances - wc_pair_distance) * (pairing_mask > 0.5)
    
    # 3. Sequential backbone distance violation
    # Create a mask for consecutive residues using diagonal shift
    backbone_mask = torch.zeros(seq_length, seq_length, device=coords.device)
    idx = torch.arange(seq_length-1, device=coords.device)
    backbone_mask[idx, idx+1] = 1.0  # Set 1s on first diagonal above main diagonal
    backbone_mask = backbone_mask.unsqueeze(0).expand(batch_size, -1, -1)
    backbone_violation = torch.abs(distances - backbone_distance) * backbone_mask
    
    # Combine violations with weights
    total_violation = (min_dist_violation * 2.0 +  # Stronger penalty for steric clashes
                      pair_dist_violation * 1.0 +  # Base pair distance penalty
                      backbone_violation * 1.0)    # Backbone distance penalty
    
    return total_violation.mean()  # Average over all violations

class RNAStructureModel(nn.Module):
    def __init__(self, seq_length, feature_dim=4):
        super(RNAStructureModel, self).__init__()
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        
        # Feature extraction blocks with residual connections
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128)
            )
        ])
        
        # Projection layers for residual connections
        self.project1 = nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=1)
        self.project2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        
        # Batch normalization after residual connections
        self.bn_after_res1 = nn.BatchNorm1d(64)
        self.bn_after_res2 = nn.BatchNorm1d(128)
        
        # LSTM layers organized using LSTMBlock
        self.lstm1 = LSTMBlock(input_size=128, hidden_size=128, bidirectional=True, layer_norm=True)
        self.lstm2 = LSTMBlock(input_size=256, hidden_size=128, bidirectional=True, layer_norm=True)
        
        # Self-attention layer
        self.base_pair_attention = BaseAwareAttention(embed_dim=256, num_heads=4)
        
        # Output layer for 3D coordinates (x, y, z) for each position
        self.output_layer = nn.Linear(256, 3)

    def forward(self, x, return_distance_loss=False):
        """Forward pass of the model"""
        pairing_mask = create_base_pairing_mask(x)

        batch_size, seq_len, feat_dim = x.shape
        
        # Transpose for Conv1d: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Save input for residual connection
        identity1 = self.project1(x)
        
        # First conv block
        x = self.features[0](x)
        x = self.features[1](x)
        x = x + identity1
        x = self.bn_after_res1(x)
        x = F.relu(x)
        
        # Second conv block
        identity2 = self.project2(x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = x + identity2
        x = self.bn_after_res2(x)
        x = F.relu(x)
        
        # Transpose back for sequence processing
        x = x.transpose(1, 2)
        
        # LSTM layers
        x = self.lstm1(x)
        x = self.lstm2(x)
        
        # Self-attention with memory optimization
        x = self.base_pair_attention(x, pairing_mask)
        
        # Final output layer
        coords = self.output_layer(x)
        
        if return_distance_loss:
            distance_loss = compute_distance_violations(coords, pairing_mask)
            return coords, distance_loss
            
        return coords
    
    def predict(self, x, sequence_lengths):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            # Forward pass
            output = self.forward(x)
            
            # If the forward pass returns a tuple, extract the predictions
            if isinstance(output, tuple):
                predictions = output[0]  # Extract the first element (coords)
            else:
                predictions = output  # If it's already a tensor

            # Trim predictions based on sequence lengths
            trimmed_predictions = []
            for i, length in enumerate(sequence_lengths):
                trimmed_predictions.append(predictions[i, :length].cpu().numpy())
            
            return trimmed_predictions
    
    def save(self, filepath):
        """Save the model weights"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model weights"""
        self.load_state_dict(torch.load(filepath))

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, layer_norm=False):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            batch_first=True, bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm(hidden_size * 2) if bidirectional and layer_norm else None

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM returns (output, (h_n, c_n))
        if self.layer_norm:
            x = self.layer_norm(x)  # Apply layer normalization if enabled
        return x

class BaseAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x, base_pair_mask, padding_mask=None):
        # Get the batch size and sequence length
        batch_size, seq_length, _ = x.shape
        
        # Repeat the base_pair_mask for the number of attention heads
        base_pair_mask = base_pair_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # Shape: (batch_size, num_heads, seq_length, seq_length)
        base_pair_mask = base_pair_mask.view(-1, seq_length, seq_length)  # Flatten to match attention batch size
        
        # Ensure base_pair_mask is of type float (required for attn_mask)
        base_pair_mask = base_pair_mask.to(dtype=torch.float32)
        
        # Ensure padding_mask is of type float and repeat it for the number of attention heads
        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.float32)
            padding_mask = padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1)  # Shape: (batch_size, num_heads, seq_length)
            padding_mask = padding_mask.view(-1, seq_length)  # Flatten to match the batch size of base_pair_mask
        
        # Combine base_pair_mask and padding_mask
        if padding_mask is not None:
            combined_mask = base_pair_mask + padding_mask.unsqueeze(1).to(x.device)  # Combine masks
        else:
            combined_mask = base_pair_mask
        
        # Apply the combined mask during attention
        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=None,  # Set to None since we're using attn_mask
            attn_mask=combined_mask.to(x.device)  # Use combined mask for attn_mask
        )
        return attn_output



# Prediction and evaluation utilities
import os
import numpy as np
import pandas as pd
import torch

# --- Prediction and evaluation functions from predict.py ---
def predict_structures(model_path, test_seq_file, output_path):
    """Predict RNA 3D structures for test sequences"""
    data_loader = RNADataLoader()
    print("Loading test data...")
    X, sequence_lengths, ids = data_loader.load_test_data(test_seq_file)
    print(f"Loaded {len(X)} sequences for prediction")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = X_tensor.to(device)
    print("Loading model...")
    model = RNAStructureModel(seq_length=X.shape[1], feature_dim=X.shape[2])
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        print("Loading model state_dict")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Predicting structures...")
    with torch.no_grad():
        predictions = model.predict(X_tensor, sequence_lengths)

    # Add 4 noisy versions for each prediction
    all_rows = []
    for seq_id, seq_len, pred in zip(ids, sequence_lengths, predictions):
        if not hasattr(predict_structures, '_seq_cache'):
            seq_df = pd.read_csv(test_seq_file)
            predict_structures._seq_cache = dict(zip(seq_df['target_id'], seq_df['sequence']))
        sequence = predict_structures._seq_cache.get(seq_id, '')
        for pos in range(seq_len):
            resname = sequence[pos] if pos < len(sequence) else ''
            resid = pos + 1
            # ID should be target_id_position (e.g., R1107_1)
            id_with_pos = f"{seq_id}_{resid}"
            row = {
                'ID': id_with_pos,
                'resname': resname,
                'resid': resid,
                'x_1': pred[pos][0],
                'y_1': pred[pos][1],
                'z_1': pred[pos][2],
            }
            for i in range(4):
                noisy = pred + np.random.normal(0, 0.1, pred.shape)
                row[f'x_{i+2}'] = noisy[pos][0]
                row[f'y_{i+2}'] = noisy[pos][1]
                row[f'z_{i+2}'] = noisy[pos][2]
            all_rows.append(row)
    output_df = pd.DataFrame(all_rows)
    cols = ['ID', 'resname', 'resid'] + [f'{c}_{i}' for i in range(1,6) for c in ['x','y','z']]
    output_df = output_df[[c for c in cols if c in output_df.columns]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return output_df

def evaluate_predictions(predictions_file, ground_truth_file):
    """Evaluate predictions against ground truth"""
    predictions = pd.read_csv(predictions_file)
    ground_truth = pd.read_csv(ground_truth_file)
    merged = predictions.merge(
        ground_truth,
        left_on=['target_id', 'position'],
        right_on=['ID', 'position'],
        suffixes=('_pred', '')
    )
    merged['squared_diff_x'] = (merged['x_1_pred'] - merged['x_1'])**2
    merged['squared_diff_y'] = (merged['y_1_pred'] - merged['y_1'])**2
    merged['squared_diff_z'] = (merged['z_1_pred'] - merged['z_1'])**2
    merged['rmsd'] = np.sqrt(
        merged['squared_diff_x'] + 
        merged['squared_diff_y'] + 
        merged['squared_diff_z']
    )
    overall_rmsd = np.sqrt(merged[['squared_diff_x', 'squared_diff_y', 'squared_diff_z']].sum().sum() / len(merged))
    print(f"Overall RMSD: {overall_rmsd:.4f}Å")
    sequence_rmsd = merged.groupby('target_id').apply(
        lambda x: np.sqrt(
            (x['squared_diff_x'].sum() + 
             x['squared_diff_y'].sum() + 
             x['squared_diff_z'].sum()) / len(x)
        )
    )
    print(f"Mean sequence RMSD: {sequence_rmsd.mean():.4f}Å")
    print(f"Min sequence RMSD: {sequence_rmsd.min():.4f}Å")
    print(f"Max sequence RMSD: {sequence_rmsd.max():.4f}Å")
    return overall_rmsd, sequence_rmsd

def get_sequence_labels(labels_df, target_id):
    """Get sequence labels for a specific target ID"""
    seq_labels = labels_df[labels_df['ID'] == target_id]
    return seq_labels

# --- Run prediction and output submission.csv ---
# Set your model and test file paths here
model_path = "/kaggle/input/phmfold650/pytorch/default/1/rna_structure_model_l5_lr005_epoch_650.pth"  # Update as needed
test_seq_file = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"  # Example: use train_sequences.csv as test input
output_path = "/kaggle/working/submission.csv"

# Run prediction
df_pred = predict_structures(model_path, test_seq_file, output_path)
print(df_pred.head())

