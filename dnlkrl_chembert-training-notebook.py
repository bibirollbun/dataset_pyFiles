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

!pip install transformers scikit-learn pandas numpy --no-index --find-links /kaggle/input/downloadedlibs/
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'

from transformers import AutoTokenizer, AutoModel

import torch
from torch import nn

from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from torch import nn
from torch.optim import AdamW
import matplotlib.pyplot as plt

import os
import joblib
import gc

os.environ['XLA_USE_GPU'] = '0'

df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
local_model_path = "/kaggle/input/openpoly-chembert-dataset-1/openpoly_chembert_dataset/model_files"
absolute_path = os.path.abspath(local_model_path)



class ChemBERTRegressor(nn.Module):
    def __init__(self, model_name, dropout_p=0.1):
        super(ChemBERTRegressor, self).__init__()
        self.chembert = AutoModel.from_pretrained(absolute_path)
        # Get the size of the ChemBERT output embeddings
        embedding_dim = self.chembert.config.hidden_size
        # The regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim, 1) # Maps the embedding to a single output value
        )

    def forward(self, input_ids, attention_mask):
        # Pass the tokenized input to ChemBERT
        outputs = self.chembert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the embedding of the [CLS] token (the first token)
        cls_embedding = outputs.last_hidden_state[:, 0]
        # Pass it through the regression head to get the final prediction
        prediction = self.regressor(cls_embedding)
        return prediction

class PolymerDataset(Dataset):
    def __init__(self, dataframe, tokenizer, target_name):
        self.tokenizer = tokenizer
        self.smiles = dataframe['SMILES'].tolist()
        self.targets = dataframe[target_name].tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_str = self.smiles[idx]
        target_val = self.targets[idx]
        
        # Tokenize the SMILES string
        inputs = self.tokenizer(
            smiles_str,
            padding='max_length',
            truncation=True,
            max_length=512, # A reasonable max length for polymers
            return_tensors='pt'
        )
        
        # The tokenizer returns a dictionary. We need to squeeze the tensors
        # to remove the batch dimension, as DataLoader will add it back.
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': torch.tensor(target_val, dtype=torch.float)
        }


# Set environment variables for offline use
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'

# Define paths and device once
absolute_path = "/kaggle/input/openpoly-chembert-dataset-1/openpoly_chembert_dataset/model_files" #<-- Make sure this path is correct
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(absolute_path)

df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
#target_name_list = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
target_name_list = ["Density"]

# --- MAIN TRAINING LOOP ---
for target_col in target_name_list:
    print(f"--- NOW TRAINING FOR TARGET: {target_col} ---")
    
    # 1. Prepare data for the current target
    df_clean = df.dropna(subset=[target_col]).copy()
    df_train, df_val = train_test_split(df_clean, test_size=0.1, random_state=42)
    
    # 2. Create, fit, and save the scaler for this specific target
    scaler = StandardScaler()
    df_train[target_col + "_scaled"] = scaler.fit_transform(df_train[[target_col]])
    df_val[target_col + "_scaled"] = scaler.transform(df_val[[target_col]])
    joblib.dump(scaler, f'scaler_{target_col}.pkl')
    print(f"Scaler for {target_col} saved to scaler_{target_col}.pkl")

    # 3. Create datasets and dataloaders
    train_dataset = PolymerDataset(df_train, tokenizer, target_name=target_col + "_scaled")
    val_dataset = PolymerDataset(df_val, tokenizer, target_name=target_col + "_scaled")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers = 0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers = 0)

    # 4. Initialize a new model and optimizer for this target
    model = ChemBERTRegressor(absolute_path).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    
    # 5. Training loop with early stopping
    num_epochs = 200
    patience = 30
    patience_counter = 0
    best_val_loss = float('inf')

    train_mae_history = []
    val_mae_history = []


    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            # (Training batch logic...)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)

        train_mae_history.append(avg_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # (Validation batch logic...)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['target'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.squeeze(), targets)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        val_mae_history.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the model with a UNIQUE name for this target
            torch.save(model.state_dict(), f'best_model_{target_col}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered for {target_col}.")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(train_mae_history, label='Train MAE')
    plt.plot(val_mae_history, label='Val MAE')
    title = 'MAE over Epochs ' + target_col 
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)
    
print("\n--- All models trained and saved successfully! ---")




