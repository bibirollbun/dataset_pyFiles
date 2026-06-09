import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# train_seq = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
# train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
# train_seq.head(1)
# print(train_labels.head(1))


# print(f'Shape: {train_labels.shape}')
# train_labels.isna().sum()


# train_seq[train_seq['all_sequences'].isna()]
# print(train_labels[train_labels['x_1'].isna()])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================
# 1. Data Preprocessing
# =====================
def load_and_preprocess_data(train_seq_file, train_labels_file):
    # Load train sequences
    train_seq = pd.read_csv(train_seq_file)
    train_labels = pd.read_csv(train_labels_file)

    # Extract target_id from ID column
    train_labels["target_id"] = train_labels["ID"].apply(lambda x: "_".join(x.split("_")[:2]))

    # Create sequence features
    sequence_features = create_sequence_features(train_seq)

    # Merge sequence features with labels
    df_merged = pd.merge(train_labels, sequence_features, on="target_id", how="left")

    # Normalize coordinate labels
    scaler = StandardScaler()
    coord_cols = [col for col in df_merged.columns if col.startswith(("x_", "y_", "z_"))]
    df_merged[coord_cols] = scaler.fit_transform(df_merged[coord_cols])

    # Split into train and validation sets
    train_df, val_df = train_test_split(df_merged, test_size=0.2, random_state=42)

    return train_df, val_df, scaler

def create_sequence_features(train_seq):
    """Convert RNA sequences into numerical features"""
    sequence_features = {}

    nucleotide_map = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 
                      'C': [0, 0, 1, 0], 'U': [0, 0, 0, 1],
                      'T': [0, 0, 0, 1]}  # Treat T as U

    for _, row in train_seq.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        
        seq_length = len(sequence)
        nt_counts = {nt: sequence.count(nt) for nt in "AGCU"}
        
        features = {
            'target_id': target_id,
            'seq_length': seq_length,
            'A_count': nt_counts['A'],
            'G_count': nt_counts['G'],
            'C_count': nt_counts['C'],
            'U_count': nt_counts['U'],
        }

        # One-hot encoding per position (optional)
        for i, nt in enumerate(sequence[:100]):  # Limit sequence length
            for j, val in enumerate(nucleotide_map.get(nt, [0, 0, 0, 0])):
                features[f'pos_{i}_{j}'] = val
        
        sequence_features[target_id] = features
    
    return pd.DataFrame.from_dict(sequence_features, orient="index")



# =====================
# 2. Custom Dataset
# =====================
class RNADataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = [col for col in dataframe.columns if col.startswith(("seq_length", "A_count", "G_count", "C_count", "U_count", "pos_"))]
        self.targets = [col for col in dataframe.columns if col.startswith(("x_", "y_", "z_"))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    # Ensure features are numeric and fill NaN values
        features = self.data.iloc[idx][self.features].astype(float).fillna(0).values
        targets = self.data.iloc[idx][self.targets].astype(float).fillna(0).values
        return {"features": torch.tensor(features, dtype=torch.float32),
                "targets": torch.tensor(targets, dtype=torch.float32)}



# =====================
# 3. Model Definition
# =====================
class RNAFoldingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(RNAFoldingModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Predict (x_1, y_1, z_1)
        )

    def forward(self, x):
        return self.model(x)


# =====================
# 4. Training Function
# =====================
def train_model(train_df, val_df, epochs=20, batch_size=64, lr=0.001):
    # Create datasets
    train_dataset = RNADataset(train_df)
    val_dataset = RNADataset(val_df)

    # Input dimension
    input_dim = len(train_dataset.features)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = RNAFoldingModel(input_dim).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "rna_model.pth")
    print("Model saved successfully!")

    return model


# =====================
# 5. Generate Submission
# =====================
def generate_submission(model, scaler, test_seq_file, sample_submission_file):
    test_seq = pd.read_csv(test_seq_file)
    sample_submission = pd.read_csv(sample_submission_file)

    test_features = create_sequence_features(test_seq)

    submission_df = sample_submission.copy()
    submission_df["target_id"] = submission_df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
    submission_df = pd.merge(submission_df, test_features, on="target_id", how="left")

    test_dataset = RNADataset(submission_df)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            features = batch["features"].to(device)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_predictions = scaler.inverse_transform(all_predictions)

    submission_df[["x_1", "y_1", "z_1"]] = all_predictions
    submission_df.to_csv("submission.csv", index=False)
    print("Submission generated successfully!")


# =====================
# 6. Main Execution
# =====================
if __name__ == "__main__":
    train_df, val_df, scaler = load_and_preprocess_data("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv", "/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
    model = train_model(train_df, val_df)
    generate_submission(model, scaler, "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv", "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Extract residue IDs and coordinates
residues = 3
x = -21
y = 5
z = 11

# Create 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot points in 3D
ax.scatter(x, y, z, c=residues, cmap='viridis', s=50, label="RNA Residues")
ax.plot(x, y, z, linestyle='-', linewidth=2, color='blue', label="RNA Backbone")

# Labels
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("Predicted RNA 3D Folding Structure")

plt.legend()
plt.show()





