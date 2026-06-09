!pip install torch transformers pandas numpy scikit-learn


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RedditCommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create enhanced text with rule context
        text = f"Rule: {row['rule']}. Subreddit: {row['subreddit']}. Comment: {row['body']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if self.is_test:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(row['rule_violation'], dtype=torch.float)
            }

def create_augmented_data(df):
    """Create augmented data using positive and negative examples"""
    augmented_rows = []
    
    for _, row in df.iterrows():
        # Add original row
        augmented_rows.append(row.to_dict())
        
        # Add positive examples as additional training data
        for i in [1, 2]:
            pos_example = row.get(f'positive_example_{i}')
            if pd.notna(pos_example) and pos_example != '':
                aug_row = row.copy()
                aug_row['body'] = pos_example
                aug_row['rule_violation'] = 1
                augmented_rows.append(aug_row.to_dict())
            
            # Add negative examples
            neg_example = row.get(f'negative_example_{i}')
            if pd.notna(neg_example) and neg_example != '':
                aug_row = row.copy()
                aug_row['body'] = neg_example
                aug_row['rule_violation'] = 0
                augmented_rows.append(aug_row.to_dict())
    
    return pd.DataFrame(augmented_rows)

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    best_val_auc = 0
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Validation
        val_auc = evaluate_model(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Validation AUC: {val_auc:.4f}')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            
            predictions.extend(probs.flatten())
            true_labels.extend(labels)
    
    return roc_auc_score(true_labels, predictions)

def predict(model, data_loader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            
            predictions.extend(probs.flatten())
    
    return predictions

def main():
    # Load data
    train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    
    # Create augmented training data
    print("Creating augmented data...")
    augmented_train_df = create_augmented_data(train_df)
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        augmented_train_df, test_size=0.2, random_state=42, stratify=augmented_train_df['rule_violation']
    )
    
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"  # You can try other models like roberta-base, distilbert-base-uncased
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = RedditCommentDataset(train_data, tokenizer)
    val_dataset = RedditCommentDataset(val_data, tokenizer)
    test_dataset = RedditCommentDataset(test_df, tokenizer, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Train model
    print("Training model...")
    model = train_model(model, train_loader, val_loader, epochs=3)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Predict on test set
    print("Making predictions...")
    predictions = predict(model, test_loader)
    
    # Create submission
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': predictions
    })
    
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print("Submission file created: submission.csv")

if __name__ == "__main__":
    main()


import os
if os.path.exists('/kaggle/working/submission.csv'):
    submission_check = pd.read_csv('/kaggle/working/submission.csv')
    print("Submission file created successfully!")
    print(f"Shape: {submission_check.shape}")
    print("\nFirst few rows:")
    print(submission_check.head())
else:
    print("Submission file not found!")




