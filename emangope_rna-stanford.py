import os
import warnings
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # For interactive plots
import plotly.graph_objects as go # For interactive plots

# Deep Learning 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


print("Loading datasets...")
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


# Fill missing coordinate values to avoid NaNs.
train_labels.fillna(0, inplace=True)
validation_labels.fillna(0, inplace=True)


print("\nBasic dataset information:")
print(f"Training sequences: {train_sequences.shape}")
print(f"Training labels: {train_labels.shape}")
print(f"Validation sequences: {validation_sequences.shape}")
print(f"Validation labels: {validation_labels.shape}")
print(f"Test sequences: {test_sequences.shape}")
print(f"Sample submission: {sample_submission.shape}")


# 2. Data Analysis (optional visualizations)
print("\nAnalyzing RNA sequence lengths...")
train_sequences['length'] = train_sequences['sequence'].str.len()
plt.figure(figsize=(12, 6))
sns.histplot(train_sequences['length'], bins=50)
plt.title('Distribution of RNA Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.savefig('sequence_length_distribution.png')
plt.show()


train_sequences.head()


sample_submission.head(3)


# Get the first 10 rows
#ten_sequences = train_sequences.head(10)

# Download as CSV
#ten_sequences.to_csv('first_10_sequences.csv', index=False)

#print("first_10_sequences.csv has been created.")


# Get the first 10 rows
#sample_submission = sample_submission.head(10)

# Download as CSV
#sample_submission.to_csv('first_10_sample_sub.csv', index=False)

#print("first_10_sample_sub.csv has been created.")


# Get the first 10 rows
train_labels= train_labels.head(10)

# Download as CSV
train_labels.to_csv('first_10_train_labels.csv', index=False)

print("first_10_sequences.csv has been created.")


import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Data Loading
print("Starting Stanford RNA 3D Folding notebook...")
print("Loading datasets...")
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')

# Fill missing coordinate values to avoid NaNs.
train_labels.fillna(0, inplace=True)
validation_labels.fillna(0, inplace=True)

print("\nBasic dataset information:")
print(f"Training sequences: {train_sequences.shape}")
print(f"Training labels: {train_labels.shape}")
print(f"Validation sequences: {validation_sequences.shape}")
print(f"Validation labels: {validation_labels.shape}")
print(f"Test sequences: {test_sequences.shape}")
print(f"Sample submission: {sample_submission.shape}")

# 2. Analyze sequence lengths (optional)
train_sequences['length'] = train_sequences['sequence'].str.len()
plt.figure(figsize=(12, 6))
sns.histplot(train_sequences['length'], bins=50)
plt.title('Distribution of RNA Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.savefig('sequence_length_distribution.png')
plt.close()

# 3. Data Preprocessing
def preprocess_sequence_data(sequences_df, labels_df=None, is_train=True):
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    processed_data = []
    for idx, row in sequences_df.iterrows():
        seq_id = row['target_id']
        sequence = row['sequence']
        numerical_seq = [nucleotide_map.get(nuc, 4) for nuc in sequence]
        structures = None
        if is_train and labels_df is not None:
            sequence_labels = labels_df[labels_df['ID'].str.startswith(seq_id + '_')]
            if not sequence_labels.empty:
                coords = []
                for _, label_row in sequence_labels.iterrows():
                    x = label_row['x_1']
                    y = label_row['y_1']
                    z = label_row['z_1']
                    coords.append([x, y, z])
                coords = np.array(coords)
                mean = np.mean(coords, axis=0)
                std = np.std(coords, axis=0) + 1e-8
                coords_norm = (coords - mean) / std
                structures = [coords_norm]
        processed_data.append({
            'id': seq_id,
            'sequence': numerical_seq,
            'structures': structures
        })
    return processed_data

print("Preprocessing training data...")
train_data = preprocess_sequence_data(train_sequences, train_labels, is_train=True)
print("Preprocessing validation data...")
validation_data = preprocess_sequence_data(validation_sequences, validation_labels, is_train=True)
print("Preprocessing test data...")
test_data = preprocess_sequence_data(test_sequences, is_train=False)

# 4. Feature Engineering: one-hot encoding, positional info, GC-content
def extract_sequence_features(sequence):
    one_hot = np.zeros((len(sequence), 5))
    for i, nuc in enumerate(sequence):
        one_hot[i, nuc] = 1
    gc_content = []
    window_size = 5
    for i in range(len(sequence)):
        start = max(0, i - window_size // 2)
        end = min(len(sequence), i + window_size // 2 + 1)
        window = sequence[start:end]
        gc_count = sum(1 for n in window if n in [1, 2])
        gc_content.append(gc_count / len(window))
    positions = np.array([[i / len(sequence)] for i in range(len(sequence))])
    features = np.hstack((one_hot, positions, np.array(gc_content).reshape(-1, 1)))
    return features

print("Extracting sequence features...")
for i, data in enumerate(train_data):
    train_data[i]['features'] = extract_sequence_features(data['sequence'])
for i, data in enumerate(validation_data):
    validation_data[i]['features'] = extract_sequence_features(data['sequence'])
for i, data in enumerate(test_data):
    test_data[i]['features'] = extract_sequence_features(data['sequence'])

# 5. Enhance features with RNA Secondary Structure information
def predict_rna_secondary_structure(sequence):
    nucleotide_map_inv = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'X'}
    seq_chars = [nucleotide_map_inv[n] for n in sequence]
    structure = ['.' for _ in range(len(seq_chars))]
    complementary = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'X': None}
    for i in range(len(seq_chars)):
        if structure[i] != '.':
            continue
        for j in range(len(seq_chars) - 1, i + 3, -1):
            if structure[j] != '.':
                continue
            if complementary.get(seq_chars[i]) == seq_chars[j]:
                structure[i] = '('
                structure[j] = ')'
                break
    return ''.join(structure)

def enhance_features_with_ss(data):
    for i, item in enumerate(data):
        seq = item['sequence']
        ss = predict_rna_secondary_structure(seq)
        ss_features = np.zeros((len(ss), 3))
        for j, char in enumerate(ss):
            if char == '.':
                ss_features[j, 0] = 1
            elif char == '(':
                ss_features[j, 1] = 1
            elif char == ')':
                ss_features[j, 2] = 1
        data[i]['features'] = np.hstack((item['features'], ss_features))
    return data

print("Enhancing features with secondary structure information...")
train_data = enhance_features_with_ss(train_data)
validation_data = enhance_features_with_ss(validation_data)
test_data = enhance_features_with_ss(test_data)

# 6. Custom Dataset and DataLoader
class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        if item['structures'] is not None:
            target = torch.tensor(item['structures'][0], dtype=torch.float32)
            return features, target, item['id']
        else:
            return features, None, item['id']

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    max_length = features[0].shape[0]
    feature_dim = features[0].shape[1]
    padded_features = []
    padded_targets = []
    lengths = []
    for i, feat in enumerate(features):
        length = feat.shape[0]
        lengths.append(length)
        pad_feat = torch.cat([feat, torch.zeros((max_length - length, feature_dim))], dim=0)
        padded_features.append(pad_feat)
        if targets[i] is not None:
            pad_target = torch.cat([targets[i], torch.zeros((max_length - length, 3))], dim=0)
            padded_targets.append(pad_target)
    features_tensor = torch.stack(padded_features)
    if all(t is not None for t in targets):
        targets_tensor = torch.stack(padded_targets)
        return features_tensor, targets_tensor, ids, lengths
    else:
        return features_tensor, None, ids, lengths

train_dataset = RNADataset(train_data)
validation_dataset = RNADataset(validation_data)
test_dataset = RNADataset(test_data)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 7. Define a Transformer-based Model (improved capacity)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RNAFoldingTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.3):
        super(RNAFoldingTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, 3)
    
    def forward(self, x, seq_lengths=None):
        # x: (B, L, input_dim)
        x = self.input_linear(x)  # (B, L, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (L, B, d_model)
        x = self.transformer_encoder(x)  # (L, B, d_model)
        x = x.transpose(0, 1)  # (B, L, d_model)
        x = self.dropout(x)
        x = self.fc_out(x)  # (B, L, 3)
        return x

# 8. Loss Function: Smooth L1 Loss over valid timesteps
def smooth_l1_loss(output, target, seq_lengths):
    mask = torch.zeros_like(target, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        mask[i, :length, :] = True
    loss = nn.SmoothL1Loss(reduction='none')(output, target)
    masked_loss = loss * mask.float()
    return masked_loss.sum() / mask.sum() if mask.sum() > 0 else 0

# 9. Training and Evaluation Functions
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0
    batches = 0
    for features, targets, _, seq_lengths in dataloader:
        if targets is None:
            continue
        optimizer.zero_grad()
        features = features.to(device)
        targets = targets.to(device)
        outputs = model(features, seq_lengths)
        loss = smooth_l1_loss(outputs, targets, seq_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1
    return epoch_loss / batches if batches > 0 else float('inf')

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    batches = 0
    with torch.no_grad():
        for features, targets, _, seq_lengths in dataloader:
            if targets is None:
                continue
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features, seq_lengths)
            loss = smooth_l1_loss(outputs, targets, seq_lengths)
            val_loss += loss.item()
            batches += 1
    return val_loss / batches if batches > 0 else float('inf')

def train_model(model, train_loader, val_loader, num_epochs=40, lr=0.0002, device='cpu'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    val_losses = []
    best_model_state = model.state_dict()
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print("  New best model saved.")
    
    model.load_state_dict(best_model_state)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training History")
    plt.savefig("training_history.png")
    plt.close()
    
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    return model

# 10. Ensemble: Train multiple models and average predictions
class ModelEnsemble:
    def __init__(self, models):
        self.models = models
    def predict(self, features, seq_lengths=None, device='cpu'):
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(features.to(device), seq_lengths)
                preds.append(out)
        # Average predictions
        return torch.mean(torch.stack(preds), dim=0)

def generate_ensemble_predictions(ensemble, dataloader, device, num_predictions=5):
    ensemble.models[0].eval()
    all_predictions = {}
    with torch.no_grad():
        for features, _, ids, seq_lengths in dataloader:
            features = features.to(device)
            for i, (seq_id, length) in enumerate(zip(ids, seq_lengths)):
                seq_features = features[i, :length, :].unsqueeze(0)
                # Get ensemble prediction (averaged output)
                avg_pred = ensemble.predict(seq_features, [length], device=device)
                # Generate diverse predictions via noise perturbation
                preds = []
                for _ in range(num_predictions):
                    noise = torch.randn_like(seq_features) * 0.03
                    noisy_features = seq_features + noise
                    noisy_pred = ensemble.predict(noisy_features, [length], device=device)
                    preds.append(noisy_pred[0].cpu().numpy())
                all_predictions[seq_id] = preds
    return all_predictions

# 11. Submission File Generation (same as before)
def create_submission_file(predictions, test_sequences_df, output_file='submission.csv'):
    submission_rows = []
    for _, row in test_sequences_df.iterrows():
        seq_id = row['target_id']
        sequence = row['sequence']
        if seq_id in predictions:
            pred_structures = predictions[seq_id]
            for i in range(len(sequence)):
                submission_row = {
                    'ID': f"{seq_id}_{i+1}",
                    'resname': sequence[i],
                    'resid': i+1
                }
                for j in range(5):
                    coords = pred_structures[j][i]
                    submission_row[f'x_{j+1}'] = coords[0]
                    submission_row[f'y_{j+1}'] = coords[1]
                    submission_row[f'z_{j+1}'] = coords[2]
                submission_rows.append(submission_row)
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_file, index=False)
    return submission_df

# 12. Visualization Function for 3D Structures (same as before)
def visualize_3d_structure(coords, title="RNA 3D Structure"):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', s=30, label="C1' atoms")
    for i in range(len(coords)-1):
        ax.plot([coords[i, 0], coords[i+1, 0]], 
                [coords[i, 1], coords[i+1, 1]], 
                [coords[i, 2], coords[i+1, 2]], 'k-', lw=1)
    ax.set_title(title)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# 13. Main Execution: Train two transformer models and ensemble them
def main():
    print("\n--- Main execution ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = train_data[0]['features'].shape[1]
    # Train two transformer models with different random seeds
    models = []
    for seed in [42, 2023]:
        torch.manual_seed(seed)
        model = RNAFoldingTransformer(input_dim=input_dim, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.3).to(device)
        print(f"Training model with seed {seed}...")
        model = train_model(model, train_loader, validation_loader, num_epochs=40, lr=0.0002, device=device)
        models.append(model)
    
    ensemble = ModelEnsemble(models)
    print("Ensemble created.")
    
    print("Generating ensemble predictions on test data...")
    test_predictions = generate_ensemble_predictions(ensemble, test_loader, device, num_predictions=5)
    print("Predictions generated.")
    
    print("Creating submission file...")
    submission_df = create_submission_file(test_predictions, test_sequences, output_file='submission.csv')
    print("Submission file created. Sample:")
    print(submission_df.head())
    
    print("Visualizing a sample prediction (first test sequence)...")
    sample_seq_id = test_sequences['target_id'].iloc[0]
    if sample_seq_id in test_predictions:
        sample_prediction = test_predictions[sample_seq_id][0]
        visualize_3d_structure(sample_prediction, title=f"Predicted 3D Structure - {sample_seq_id}")
        print(f"Visualization saved for {sample_seq_id}.")
    else:
        print("No prediction found for the first test sequence for visualization.")
    
    print("\n--- Main execution completed ---")

if __name__ == '__main__':
    main()

print("\nNotebook execution finished.")



sub_df = pd.read_csv("/kaggle/working/submission.csv")
sub_df.head()




