# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


# Define dataset class
class DiabetesDataset(Dataset):
    # Initialize with features and labels
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)

    # Return length
    def __len__(self):
        return len(self.features)

    # Return sample
    def __getitem__(self, idx):
        if self.labels is None:
            return self.features[idx]
        return self.features[idx], self.labels[idx]


# Define residual block
class ResidualBlock(nn.Module):
    # Initialize layers
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    # Forward pass with skip connection
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return self.relu(out + x)

# Define main model
class ResidualMLP(nn.Module):
    # Initialize layers
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.block3 = ResidualBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    # Forward pass
    def forward(self, x):
        x = self.input_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_layer(x)
        return self.sigmoid(x)


# Define function to train for one epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_features).squeeze()

        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Define function to evaluate model
def evaluate(model, loader, criterion, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            eval_loss += loss.item()

    return eval_loss / len(loader)


# Main function
def main():
    # Load training data
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

    # Extract labels
    labels = train_df["diagnosed_diabetes"].values

    # Identify categorical and numeric columns
    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Remove label and id from numeric columns
    if "diagnosed_diabetes" in numeric_cols:
        numeric_cols.remove("diagnosed_diabetes")
    if "id" in numeric_cols:
        numeric_cols.remove("id")

    # Convert categorical columns to ordinal integer codes
    for col in categorical_cols:
        train_df[col] = pd.Categorical(train_df[col]).codes.astype("int8")

    # Extract feature matrix
    features_raw = train_df[numeric_cols + categorical_cols].values

    # Standardize full feature matrix
    scaler = StandardScaler()
    features = scaler.fit_transform(features_raw)

    # Train-validation split
    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.1, random_state=42
    )

    # Build datasets
    train_dataset = DiabetesDataset(x_train, y_train)
    val_dataset = DiabetesDataset(x_val, y_val)

    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = ResidualMLP(input_dim=features.shape[1]).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train loop
    epochs = 20
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Load test data
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
    test_ids = test_df["id"].values

    # Convert categorical test columns using same ordinal mapping
    for col in categorical_cols:
        test_df[col] = pd.Categorical(test_df[col]).codes.astype("int8")

    # Extract test features
    test_features_raw = test_df[numeric_cols + categorical_cols].values

    # Apply scaling
    test_features = scaler.transform(test_features_raw)

    # Build test dataset and loader
    test_dataset = DiabetesDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Predict test probabilities
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch).squeeze()
            predictions.extend(outputs.cpu().numpy())

    # Create submission dataframe
    submission = pd.DataFrame({
        "id": test_ids,
        "diagnosed_diabetes": predictions
    })

    # Save to CSV
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


# Execute main
if __name__ == "__main__":
    main()

