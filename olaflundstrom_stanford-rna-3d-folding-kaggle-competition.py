# Stanford RNA 3D Folding Competition
# Revised Notebook for RNA 3D Structure Prediction (Improved V3)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import warnings
warnings.filterwarnings('ignore')

print("Starting Stanford RNA 3D Folding notebook...")

# 1. Data Loading and Exploration
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
plt.close()

# 3. Data Preprocessing
def preprocess_sequence_data(sequences_df, labels_df=None, is_train=True):
    """
    Preprocess RNA sequence data.
    Convert sequences to numerical form and normalize coordinate targets per sequence.
    """
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
                num_structures = (len(sequence_labels.columns) - 3) // 3
                structures = []
                for i in range(1, num_structures + 1):
                    coords = []
                    for _, label_row in sequence_labels.iterrows():
                        x = label_row[f'x_{i}']
                        y = label_row[f'y_{i}']
                        z = label_row[f'z_{i}']
                        coords.append([x, y, z])
                    coords = np.array(coords)
                    # Normalize coordinates per sequence (center and scale)
                    mean = np.mean(coords, axis=0)
                    std = np.std(coords, axis=0) + 1e-8
                    coords_norm = (coords - mean) / std
                    structures.append(coords_norm)
        processed_data.append({
            'id': seq_id,
            'sequence': numerical_seq,
            'structures': structures
        })
    return processed_data

print("Preprocessing training data...")
train_data = preprocess_sequence_data(train_sequences, train_labels)
print("Preprocessing validation data...")
validation_data = preprocess_sequence_data(validation_sequences, validation_labels)
print("Preprocessing test data...")
test_data = preprocess_sequence_data(test_sequences, is_train=False)

# 4. Feature Engineering
def extract_sequence_features(sequence):
    """
    Extract one-hot encoding, positional encoding, and GC-content as features.
    """
    one_hot = np.zeros((len(sequence), 5))
    for i, nucleotide in enumerate(sequence):
        one_hot[i, nucleotide] = 1
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

# 5. RNA Secondary Structure Prediction (simple rule-based)
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
            if complementary[seq_chars[i]] == seq_chars[j]:
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
    for i, feature in enumerate(features):
        length = feature.shape[0]
        padding = torch.zeros((max_length - length, feature_dim), dtype=torch.float32)
        padded_feature = torch.cat([feature, padding], dim=0)
        padded_features.append(padded_feature)
        if targets[i] is not None:
            target_padding = torch.zeros((max_length - length, 3), dtype=torch.float32)
            padded_target = torch.cat([targets[i], target_padding], dim=0)
            padded_targets.append(padded_target)
    features_tensor = torch.stack(padded_features)
    if all(target is not None for target in targets):
        targets_tensor = torch.stack(padded_targets)
        return features_tensor, targets_tensor, ids, [f.shape[0] for f in features]
    else:
        return features_tensor, None, ids, [f.shape[0] for f in features]

train_dataset = RNADataset(train_data)
validation_dataset = RNADataset(validation_data)
test_dataset = RNADataset(test_data)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 7. Model Architecture with BatchNorm and increased capacity
class RNAFoldingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3):
        super(RNAFoldingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 3)  # 3D coordinates
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x, seq_lengths=None):
        batch_size, seq_len, _ = x.size()
        if seq_lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=True)
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        context_vector = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        combined = lstm_out + context_vector
        x = self.relu(self.fc1(combined))
        # BatchNorm expects input as (B, C, L), so transpose, apply, then transpose back
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 8. Training Functions using Smooth L1 Loss
def smooth_l1_loss(output, target, seq_lengths):
    mask = torch.zeros_like(target, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        mask[i, :length, :] = True
    loss = nn.SmoothL1Loss(reduction='none')(output, target)
    masked_loss = loss * mask.float()
    return masked_loss.sum() / mask.sum() if mask.sum() > 0 else 0

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

def calculate_tm_score(predicted, reference):
    l_ref = len(reference)
    if l_ref >= 30:
        d0 = 0.6 * (l_ref - 0.5) ** 0.5 - 2.5
    elif l_ref >= 24:
        d0 = 0.7
    elif l_ref >= 20:
        d0 = 0.6
    elif l_ref >= 16:
        d0 = 0.5
    elif l_ref >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    tm_score = 0
    for i in range(min(len(predicted), l_ref)):
        di = np.linalg.norm(predicted[i] - reference[i])
        tm_score += 1 / (1 + (di/d0)**2)
    return tm_score / l_ref

def evaluate_model(model, dataloader, device):
    model.eval()
    tm_scores = []
    with torch.no_grad():
        for features, targets, _, seq_lengths in dataloader:
            if targets is None:
                continue
            features = features.to(device)
            outputs = model(features, seq_lengths)
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            for i, length in enumerate(seq_lengths):
                pred_coords = outputs[i, :length, :]
                target_coords = targets[i, :length, :]
                tm_score = calculate_tm_score(pred_coords, target_coords)
                tm_scores.append(tm_score)
    return np.mean(tm_scores) if tm_scores else 0

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.0005, device='cpu'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    tm_scores = []
    
    best_model_state = model.state_dict().copy()
    best_val_loss = float('inf')
    best_tm_score = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        tm_score = evaluate_model(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        tm_scores.append(tm_score)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  TM-Score: {tm_score:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f'  New best model saved (Val Loss: {val_loss:.4f})')
        if tm_score > best_tm_score:
            best_tm_score = tm_score
            print(f'  New best TM-score: {tm_score:.4f}')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: Best model state not found; using current parameters.")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(tm_scores, label='TM-Score')
    plt.title('TM-Score Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('TM-Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}, Best TM-Score: {best_tm_score:.4f}")
    return model

# 9. Model Inference and Multiple Structure Generation
def generate_diverse_structures(model, features, seq_length, num_structures=5, noise_scale=0.05, device='cpu'):
    model.eval()
    structures = []
    for i in range(num_structures):
        with torch.no_grad():
            if i > 0:
                noise = torch.randn_like(features) * noise_scale
                features_with_noise = features + noise
            else:
                features_with_noise = features
            output = model(features_with_noise.unsqueeze(0))
            coords = output[0, :seq_length, :].cpu().numpy()
            structures.append(coords)
    return structures

def generate_predictions(model, dataloader, device, num_predictions=5):
    model.eval()
    all_predictions = {}
    for features, _, ids, seq_lengths in dataloader:
        features = features.to(device)
        for i, (seq_id, length) in enumerate(zip(ids, seq_lengths)):
            seq_features = features[i, :length, :]
            predictions = generate_diverse_structures(
                model, 
                seq_features, 
                length, 
                num_structures=num_predictions,
                device=device
            )
            all_predictions[seq_id] = predictions
    return all_predictions

# 10. Submission File Generation
def create_submission_file(predictions, test_sequences_df, output_file='submission.csv'):
    submission_rows = []
    for _, row in test_sequences_df.iterrows():
        seq_id = row['target_id']
        sequence = row['sequence']
        if seq_id in predictions:
            pred_structures = predictions[seq_id]
            num_structures = len(pred_structures)
            for i in range(len(sequence)):
                submission_row = {
                    'ID': f"{seq_id}_{i+1}",
                    'resname': sequence[i],
                    'resid': i+1
                }
                for j in range(5):
                    if j < num_structures:
                        coords = pred_structures[j][i]
                        submission_row[f'x_{j+1}'] = coords[0]
                        submission_row[f'y_{j+1}'] = coords[1]
                        submission_row[f'z_{j+1}'] = coords[2]
                    else:
                        submission_row[f'x_{j+1}'] = submission_row[f'x_{j}']
                        submission_row[f'y_{j+1}'] = submission_row[f'y_{j}']
                        submission_row[f'z_{j+1}'] = submission_row[f'z_{j}']
                submission_rows.append(submission_row)
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_file, index=False)
    return submission_df

# 11. Visualization Functions
def visualize_3d_structure(coords, title="RNA 3D Structure"):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', marker='o', s=30, label="C1' atoms")
    for i in range(len(coords) - 1):
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

# 12. (Optional) Ensemble Modeling
class ModelEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
    def predict(self, features, seq_lengths=None):
        all_predictions = []
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                output = model(features, seq_lengths)
                all_predictions.append(output * self.weights[i])
        return sum(all_predictions)

# 13. Main Execution
def main():
    print("\n--- Main execution ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = train_data[0]['features'].shape[1]
    model = RNAFoldingModel(input_dim=input_dim).to(device)
    print("\nModel instantiated.")
    
    print("\nStarting model training...")
    trained_model = train_model(
        model,
        train_loader,
        validation_loader,
        num_epochs=20,  # Increased epochs
        lr=0.0005,     # Lower learning rate for stability
        device=device
    )
    print("\nModel training finished.")
    
    print("\nGenerating predictions on test data...")
    test_predictions = generate_predictions(trained_model, test_loader, device, num_predictions=5)
    print("\nPredictions generated.")
    
    print("\nCreating submission file...")
    submission_file = create_submission_file(test_predictions, test_sequences)
    print(f"\nSubmission file created: submission.csv")
    print(submission_file.head())
    
    print("\nVisualizing a sample prediction (first test sequence)...")
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

