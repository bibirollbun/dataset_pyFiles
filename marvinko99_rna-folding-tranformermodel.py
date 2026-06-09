import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Define constants
MAX_SEQ_LENGTH = 512  # Maximum sequence length
EMBEDDING_DIM = 256   # Dimension of embeddings
NUM_HEADS = 4         # Number of attention heads
NUM_LAYERS = 4        # Number of transformer layers
HIDDEN_DIM = 512      # Hidden dimension for feed-forward layers
BATCH_SIZE = 4       # Batch size
LEARNING_RATE = 3e-4  # Learning rate
NUM_EPOCHS = 20       # Number of training epochs
NUM_PREDICTIONS = 5   # Number of structure predictions required
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # Load data
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


#set Max_SEQ_length
max_len = max(train_sequences['sequence'].str.len().max(), 
              validation_sequences['sequence'].str.len().max(),
              test_sequences['sequence'].str.len().max())
print(f"Maximum sequence length in dataset: {max_len}")
MAX_SEQ_LENGTH = max_len


# RNA Dataset class
class RNADataset(Dataset):
    def __init__(self, sequences, labels=None, is_test=False):
        self.sequences = sequences
        self.labels = labels
        self.is_test = is_test
        self.nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'T': 4}
        
        if not is_test and labels is not None:
            # Extract target_id from ID in labels
            self.labels['target_id'] = self.labels['ID'].apply(
                lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x
            )
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_row = self.sequences.iloc[idx]
        target_id = sequence_row['target_id']
        sequence = sequence_row['sequence']
        
        # Encode sequence
        encoded_seq = np.zeros(MAX_SEQ_LENGTH, dtype=np.int64)
        for i, nuc in enumerate(sequence[:MAX_SEQ_LENGTH]):
            encoded_seq[i] = self.nucleotide_map.get(nuc, 0)
        
        # Create attention mask
        seq_length = min(len(sequence), MAX_SEQ_LENGTH)
        attention_mask = np.zeros(MAX_SEQ_LENGTH, dtype=np.int64)
        attention_mask[:seq_length] = 1
        
        result = {
            'target_id': target_id,
            'sequence': sequence,
            'encoded_seq': torch.tensor(encoded_seq),
            'attention_mask': torch.tensor(attention_mask),
            'seq_length': seq_length
        }
        
        if not self.is_test and self.labels is not None:
            # Find all labels for this target_id
            target_labels = self.labels[self.labels['target_id'] == target_id]
            
            if len(target_labels) > 0:
                # Extract coordinates
                coords = np.zeros((MAX_SEQ_LENGTH, 3), dtype=np.float32)
                mask = np.zeros(MAX_SEQ_LENGTH, dtype=np.float32)
                
                for i in range(1, seq_length + 1):
                    # Find row with matching resid
                    label_row = target_labels[target_labels['resid'] == i]
                    
                    if len(label_row) > 0:
                        x = label_row['x_1'].values[0]
                        y = label_row['y_1'].values[0]
                        z = label_row['z_1'].values[0]
                        
                        if not (pd.isna(x) or pd.isna(y) or pd.isna(z)):
                            coords[i-1] = [x, y, z]
                            mask[i-1] = 1.0
                
                result['coords'] = torch.tensor(coords)
                result['mask'] = torch.tensor(mask)
        
        return result



# Transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=HIDDEN_DIM, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
    
    def forward(self, src, src_mask=None):
        # Self attention with residual connection and layer norm
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feed forward with residual connection and layer norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src



# RhoFold+ model (simplified)
class RhoFoldPlus(nn.Module):
    def __init__(self, vocab_size=5):
        super(RhoFoldPlus, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(MAX_SEQ_LENGTH, EMBEDDING_DIM)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(EMBEDDING_DIM, NUM_HEADS)
            for _ in range(NUM_LAYERS)
        ])
        
        # Output heads for coordinate prediction (multiple heads for diverse predictions)
        self.coordinate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(EMBEDDING_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM // 2, 3)  # x, y, z coordinates
            )
            for _ in range(NUM_PREDICTIONS)
        ])
    
    def forward(self, input_ids, attention_mask):
        # Create padding mask for attention (True for padding positions)
        padding_mask = (attention_mask == 0)
        
        # Create position indices
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embedding lookup
        x = self.embedding(input_ids)
        pos_emb = self.pos_encoder(positions)
        
        # Add positional embeddings
        x = x + pos_emb
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, padding_mask)
        
        # Predict coordinates using different heads for diverse predictions
        coordinates = [head(x) for head in self.coordinate_heads]
        
        return coordinates



# Loss function with masking for valid positions
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, pred, target, mask):
        # Expand mask to match dimensions
        mask = mask.unsqueeze(-1).expand_as(pred)
        
        # Calculate squared error
        squared_error = (pred - target) ** 2
        
        # Apply mask and calculate mean
        masked_error = squared_error * mask
        loss = masked_error.sum() / (mask.sum() + 1e-8)
        
        return loss



# Training function
def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MaskedMSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['encoded_seq'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            target_coords = batch['coords'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            # Forward pass
            pred_coords_list = model(input_ids, attention_mask)
            
            # Calculate loss for all prediction heads
            loss = 0.0
            for pred_coords in pred_coords_list:
                loss += criterion(pred_coords, target_coords, mask)
            loss /= len(pred_coords_list)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['encoded_seq'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                target_coords = batch['coords'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                
                # Forward pass
                pred_coords_list = model(input_ids, attention_mask)
                
                # Calculate loss
                batch_loss = 0.0
                for pred_coords in pred_coords_list:
                    batch_loss += criterion(pred_coords, target_coords, mask)
                batch_loss /= len(pred_coords_list)
                
                val_loss += batch_loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f'New best model saved with val loss: {val_loss:.6f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model



# Generate test predictions
def generate_predictions(model, test_loader, test_sequences, sample_submission):
    model.eval()
    
    # Create test_clean dataframe (similar to original code)
    def parse_target(tmp_ID, tmp_sequence):
        seq_length = len(tmp_sequence)
        tmp_df = pd.DataFrame(columns=['ID', 'resname', 'resid'], index=range(seq_length))
        tmp_df['resname'] = list(tmp_sequence)
        tmp_df['ID'] = tmp_ID
        tmp_df['resid'] = range(1, seq_length + 1)
        return tmp_df
    
    test_id_seq = test_sequences[['target_id', 'sequence']]
    test_clean = pd.DataFrame(columns=['ID', 'resname', 'resid'])
    
    for index, row in test_id_seq.iterrows():
        tmp_df = parse_target(row['target_id'], row['sequence'])
        test_clean = pd.concat([test_clean, tmp_df], ignore_index=True)
    
    # Generate predictions
    predictions = {}
    
    with torch.no_grad():
        for batch in test_loader:
            target_ids = batch['target_id']
            sequences = batch['sequence']
            input_ids = batch['encoded_seq'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            seq_lengths = batch['seq_length']
            
            # Get predictions
            pred_coords_list = model(input_ids, attention_mask)
            
            # Process each sequence
            for i, (target_id, seq_len) in enumerate(zip(target_ids, seq_lengths)):
                for j in range(seq_len):
                    key = f"{target_id}_{j+1}"
                    predictions[key] = {}
                    
                    # Save all 5 predictions
                    for k, pred_coords in enumerate(pred_coords_list):
                        predictions[key][f'x_{k+1}'] = pred_coords[i, j, 0].item()
                        predictions[key][f'y_{k+1}'] = pred_coords[i, j, 1].item()
                        predictions[key][f'z_{k+1}'] = pred_coords[i, j, 2].item()
    
    # Create submission dataframe
    for idx, row in test_clean.iterrows():
        key = f"{row['ID']}_{row['resid']}"
        if key in predictions:
            for col, value in predictions[key].items():
                test_clean.loc[idx, col] = value
    
    # Format submission to match sample submission
    submission = test_clean.copy()
    submission['ID'] = submission['ID'] + '_' + submission['resid'].astype(str)
    
    # Store original columns before adding sort_order
    original_columns = sample_submission.columns.tolist()
    
    # Add sort_order for sorting
    sample_submission['sort_order'] = range(len(sample_submission))
    
    # Use only the original columns for the merge
    submission = pd.merge(
        submission[original_columns], 
        sample_submission[['ID', 'sort_order']], 
        on='ID', 
        how='left'
    ).sort_values('sort_order').drop(columns=['sort_order'])
    
    return submission



# Main function
print(f"Using device: {DEVICE}")
    

    # Create datasets
train_dataset = RNADataset(train_sequences, train_labels)
val_dataset = RNADataset(validation_sequences, validation_labels)
test_dataset = RNADataset(test_sequences, is_test=True)
    
    # Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
model = RhoFoldPlus().to(DEVICE)
print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
model = train_model(model, train_loader, val_loader)
    
    # Generate predictions
submission = generate_predictions(model, test_loader, test_sequences, sample_submission)
    
    # Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")





import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


subs = pd.read_csv("/kaggle/working/submission.csv")
subs


# After generating submission but before saving
print(submission[submission['ID'].str.startswith('R1138_')].head())


