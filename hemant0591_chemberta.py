import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import os
import pickle
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


class SMILESDataset(Dataset):
    def __init__(self, df, target_col=None):
        self.smiles = df["SMILES"].tolist()
        self.has_target = target_col is not None
        if self.has_target:
            self.targets = df[target_col].values.astype("float32")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        item = {"smiles": self.smiles[idx]}
        if self.has_target:
            item["target"] = self.targets[idx]
        return item


def chemberta_collate_fn(batch, tokenizer):
    smiles = [item["smiles"] for item in batch]
    encoding = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")

    result = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    }
    if "target" in batch[0]:
        targets = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
        result["targets"] = targets
    return result


class chemBERTaModel(nn.Module):
    def __init__(self, base_model, out_dim=1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(base_model.config.hidden_size, out_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_tokens = outputs.last_hidden_state[:,0]
        return self.regressor(self.dropout(cls_tokens)).squeeze(1)


def train(model, dl, loss_fn, opt, sched):
    model.train()
    total_loss = 0
    n = 0
    for batch in dl:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        preds = model(input_ids, attention_mask)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n

def eval(model, dl, loss_fn):
    model.eval()
    total_loss, n = 0, 0
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, targets)
            total_loss += loss.item() * len(targets)
            n += len(targets)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targs.extend(targets.detach().cpu().numpy())
    return total_loss / n, all_preds, all_targs


def train_chemberta(df, target, base_model, tokenizer, n_epochs=30, save_dir = 'saved_models_chemberta', patience=5):
    os.makedirs(save_dir, exist_ok=True)
    
    df_clean = df[['SMILES', target]].dropna()
        
    # scale target
    y_scaler = StandardScaler()
    df_clean[target] = y_scaler.fit_transform(df_clean[[target]])
    
    # save scaler
    with open(os.path.join(save_dir, f"{target}_scaler.pkl"), 'wb') as f:
        pickle.dump(y_scaler, f)
        
    train_data, val_data = train_test_split(df_clean, test_size=0.2, random_state=42)
    train_ds = SMILESDataset(train_data, target)
    val_ds = SMILESDataset(val_data, target)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=lambda x: chemberta_collate_fn(x, tokenizer))
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=lambda x: chemberta_collate_fn(x, tokenizer))
    
    model = chemBERTaModel(base_model).to(device)
    opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.1 * len(train_dl) * n_epochs),
        num_training_steps=len(train_dl) * n_epochs
    )
    loss_fn = nn.MSELoss()
    
    best_val_loss = float("inf")
    epoch_no_improve = 0
    
    for epoch in range(n_epochs):
        train_loss = train(model, train_dl, loss_fn, opt, scheduler)
        val_loss, preds, targs = eval(model, val_dl, loss_fn)
        print(f"Epoch {epoch+1}/{n_epochs} | Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}")
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"{target}_model.pt"))
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= patience:
                print("Early stopping triggered!")
                break
        
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{target}_model.pt")))
    
    # Free up memory
    # del train_dl, val_dl, opt, scheduler
    # torch.cuda.empty_cache()
    # gc.collect()
    
    return model, tokenizer, y_scaler


def predict_chemberta(df, target, model, tokenizer, scaler):
    test_ds = SMILESDataset(df)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=lambda x: chemberta_collate_fn(x, tokenizer))
    
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.detach().cpu().numpy())
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


MODEL_NAME = "/kaggle/input/chemberta-zinc-base-v1/transformers/seyonec-chemberta-zinc-base-v1/1/models/ChemBERTa-zinc-base-v1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

for target in targets:
    print(f'Working on predicting: {target}')
    
    base_model = RobertaModel.from_pretrained(MODEL_NAME)
    
    model, tokenizer, y_scaler = train_chemberta(
        df=train_df, 
        target=target, 
        base_model=base_model, 
        tokenizer=tokenizer)
    
    test_df[target] = predict_chemberta(
        df=test_df, 
        target=target, 
        model=model, 
        tokenizer=tokenizer, 
        scaler=y_scaler
    )
    
    # Clean up to be safe
    del model, base_model, y_scaler
    gc.collect()
    torch.cuda.empty_cache()


submission = test_df[['id'] + targets]
submission.head


submission.to_csv('submission.csv', index=False)

