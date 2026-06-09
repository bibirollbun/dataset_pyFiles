import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

MODELS_CONFIG = [
    {
        'name': 'deberta-v3-base',
        'path': '/kaggle/input/deberta-v3/other/deberta-v3/1/deberta-v3-base-offline',
        'max_length': 128,
        'batch_size': 16,
        'lr': 2e-5
    },
    {
        'name': 'roberta-base',
        'path': '/kaggle/input/roberta-base/other/roberta-base/1/roberta-base',
        'max_length': 128,
        'batch_size': 32,
        'lr': 3e-5
    },
    {
        'name': 'bert-base-uncased',
        'path': '/kaggle/input/bert-base/other/bert-base/1/bert-base-uncased',
        'max_length': 128,
        'batch_size': 32,
        'lr': 2e-5
    }
]

class PatentDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'target': torch.tensor(self.scores[idx], dtype=torch.float)
        }

def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
            targets = batch['target'].to(device)
            
            outputs = model(**inputs)
            loss = torch.nn.MSELoss()(outputs.logits.view(-1), targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model_config, train_loader, val_loader, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['path'])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['path'], 
        num_labels=1
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=model_config['lr'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader)*epochs
    )
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}') as progress:
            for batch in progress:
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
                targets = batch['target'].to(device)
                
                outputs = model(**inputs)
                loss = torch.nn.MSELoss()(outputs.logits.view(-1), targets)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, device)
        
        print(f"{model_config['name']} Epoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            torch.save(model.state_dict(), f"best_{model_config['name']}.pth")
            best_loss = val_loss
    
    return model

def predict(model, tokenizer, texts, max_length, batch_size, device):
    dataset = PatentDataset(texts, np.zeros(len(texts)), tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
            outputs = model(**inputs)
            preds = outputs.logits.view(-1).cpu().numpy()
            predictions.extend(preds)
    
    return np.array(predictions)

def main():
    train_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
    test_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')
    
    train_df['text'] = train_df['context'] + ' [SEP] ' + train_df['anchor'] + ' [SEP] ' + train_df['target']
    test_df['text'] = test_df['context'] + ' [SEP] ' + test_df['anchor'] + ' [SEP] ' + test_df['target']
    
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['text'].values,
        train_df['score'].values,
        test_size=0.2,
        random_state=42
    )
    
    all_predictions = []
    for config in MODELS_CONFIG:
        print(f"\nTraining {config['name']}")
        
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        train_dataset = PatentDataset(X_train, y_train, tokenizer, config['max_length'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        val_dataset = PatentDataset(X_val, y_val, tokenizer, config['max_length'])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        model = train_model(config, train_loader, val_loader)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(f"best_{config['name']}.pth"))
        test_preds = predict(
            model,
            tokenizer,
            test_df['text'].values,
            config['max_length'],
            config['batch_size'],
            device
        )
        all_predictions.append(test_preds)
    
    weights = [0.4, 0.3, 0.3]
    ensemble_preds = np.average(all_predictions, axis=0, weights=weights)
    ensemble_preds = np.clip(ensemble_preds, 0, 1)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'score': ensemble_preds
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()




