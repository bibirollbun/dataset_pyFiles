import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/kaggle/input/playground-series-s5e11'  # Update this to your dataset directory
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
HIDDEN_DIM = 512
DROPOUT_RATE = 0.3



def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()
print(f"Using device: {DEVICE}")


def load_and_preprocess_data():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    
    target_col = 'loan_paid_back'
    id_col = 'id'
    
    test_ids = test_df[id_col]
    
    y = train_df[target_col].values
    train_df = train_df.drop(columns=[id_col, target_col])
    test_df = test_df.drop(columns=[id_col])

    all_data = pd.concat([train_df, test_df], axis=0)

    cat_cols = [c for c in all_data.columns if all_data[c].dtype == 'object']
    num_cols = [c for c in all_data.columns if all_data[c].dtype in ['int64', 'float64']]

    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        all_data[num_cols] = num_imputer.fit_transform(all_data[num_cols])
    
    for col in cat_cols:
        all_data[col] = all_data[col].fillna("MISSING")
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))

    if num_cols:
        scaler = StandardScaler()
        all_data[num_cols] = scaler.fit_transform(all_data[num_cols])

    X = all_data.iloc[:len(train_df)].values
    X_test = all_data.iloc[len(train_df):].values

    return X, y, X_test, test_ids

X, y, X_test, test_ids = load_and_preprocess_data()


# --- Dataset Class ---
class LoanDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# --- Model Architecture ---
class LoanPaybackModel(nn.Module):
    def __init__(self, input_dim):
        super(LoanPaybackModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.BatchNorm1d(HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 4),
            nn.BatchNorm1d(HIDDEN_DIM // 4),
            nn.ReLU(),
            
            nn.Linear(HIDDEN_DIM // 4, 1) # Output raw logits
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # Apply Sigmoid for ROC-AUC calculation
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    auc = roc_auc_score(all_targets, all_preds)
    return running_loss / len(loader), auc


# Split Train/Validation (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Datasets and Loaders
train_dataset = LoanDataset(X_train, y_train)
val_dataset = LoanDataset(X_val, y_val)
test_dataset = LoanDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model
model = LoanPaybackModel(input_dim=X.shape[1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# --- FIX: Removed 'verbose=True' ---
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# History for plotting
history = {
    'train_loss': [],
    'val_loss': [],
    'val_auc': []
}

print("--- Starting Training ---")
best_auc = 0
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_auc = validate(model, val_loader, criterion)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)
    
    scheduler.step(val_auc)
    
    # Get current LR manually for logging
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}")
    
    # Early Stopping
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

# Restore best model
if best_model_state:
    model.load_state_dict(best_model_state)
print(f"Training Complete. Best Validation AUC: {best_auc:.4f}")


plt.figure(figsize=(12, 5))

# Plot Losses
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='.')
plt.plot(history['val_loss'], label='Val Loss', marker='.')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot AUC
plt.subplot(1, 2, 2)
plt.plot(history['val_auc'], label='Val AUC', color='green', marker='.')
plt.title('Validation AUC Score')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("Generating Predictions...")
model.eval()
test_preds = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        test_preds.extend(preds)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': np.array(test_preds).flatten()
})

# Save to current directory
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")

# VERIFICATION: List files to ensure it exists
print("\n--- FILE CHECK ---")
files = os.listdir('.')
print(f"Files in current directory: {files}")

if 'submission.csv' in files:
    print("SUCCESS: submission.csv exists. You can now Commit and Submit.")
else:
    print("ERROR: File was not created.")

