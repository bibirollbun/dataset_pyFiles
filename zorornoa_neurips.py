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


train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")


import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch


train["SMILES"] = train["SMILES"].fillna("UNKNOWN").astype(str)
test["SMILES"] = test["SMILES"].fillna("UNKNOWN").astype(str)


target_columns = ["Tg", "FFV", "Tc", "Density", "Rg"]
train[target_columns] = train[target_columns].fillna(train[target_columns].median())


from transformers import AutoTokenizer, AutoModel
model_dir = "/kaggle/input/chemberta-zinc-base-v1.0/transformers/default/1"


tokenizer = AutoTokenizer.from_pretrained(model_dir)
bert = AutoModel.from_pretrained(model_dir)


class PolymerDataset(Dataset):
    def __init__(self, smiles, targets=None, max_length=256):
        self.smiles = smiles
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        
        # Tokenize the SMILES string
        inputs = tokenizer(smiles, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        
        if self.targets is not None:
            target = self.targets[idx]
            return inputs, torch.tensor(target, dtype=torch.float32)
        else:
            return inputs


X_train = train["SMILES"].values
y_train = train[target_columns].values
X_test = test["SMILES"].values

train_dataset = PolymerDataset(X_train, y_train)
test_dataset = PolymerDataset(X_test, targets=None)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train), 5))  # Out-of-fold predictions
test_preds = np.zeros((len(X_test), 5))  # Test predictions


for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nTraining for Fold {fold + 1}/5")

    # Split data for training and validation
    X_train_fold = X_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_train_fold = y_train[train_idx]
    y_val_fold = y_train[val_idx]

    train_dataset = PolymerDataset(X_train_fold, y_train_fold)
    val_dataset = PolymerDataset(X_val_fold, y_val_fold)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize the best validation loss and patience counter
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop for the fold
    for epoch in range(10):  # Set a maximum number of epochs
        model.train()
        train_loss = 0

        # Training step
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training"):
            inputs, targets = batch
            inputs = {key: value.to(device) for key, value in inputs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs).logits.squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
                inputs, targets = batch
                inputs = {key: value.to(device) for key, value in inputs.items()}
                targets = targets.to(device)

                outputs = model(**inputs).logits.squeeze(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping: Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 3:  # Patience for early stopping
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Store out-of-fold predictions
    oof_preds[val_idx, :] = np.array(val_preds)

    # Make predictions on the test set for the current fold
    model.eval()
    test_fold_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold + 1} - Test Predictions"):
            inputs = batch
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs).logits.squeeze(-1)
            test_fold_preds.extend(outputs.cpu().numpy())

    test_preds[:, :] += np.array(test_fold_preds) / 5  # Average predictions across folds


for target_idx, target in enumerate(target_columns):
    mae = mean_absolute_error(y_train[:, target_idx], oof_preds[:, target_idx])
    print(f"MAE for {target}: {mae:.4f}")


submission = pd.DataFrame(test_preds, columns=target_columns)
submission["id"] = test["id"]
submission.to_csv("submission.csv", index=False)
print("Submission file created.")




