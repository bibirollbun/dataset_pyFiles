# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import os
import gc
from collections import Counter
from tqdm.notebook import tqdm

# Machine learning libraries
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Deep learning libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# Use PyTorch AdamW
from torch.optim import AdamW

# Visualization settings
sns.set(style='whitegrid')
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


# Read the data
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_submission = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Let's check the data
print("\nTrain data info:")
train_df.info()

print("\nTest data info:")
test_df.info()


# Display a few examples from the training data
train_df.sample(3)


# Fill NaN values in Misconception column with 'NA'
train_df['Misconception'] = train_df['Misconception'].fillna('NA')

# Check the distribution of Category values
plt.figure(figsize=(10, 6))
sns.countplot(y='Category', data=train_df, order=train_df['Category'].value_counts().index)
plt.title('Distribution of Categories')
plt.tight_layout()
plt.show()


# Check the distribution of misconceptions (excluding 'NA')
misconception_counts = train_df[train_df['Misconception'] != 'NA']['Misconception'].value_counts()

plt.figure(figsize=(14, 10))
sns.barplot(x=misconception_counts.values[:20], y=misconception_counts.index[:20])
plt.title('Top 20 Misconceptions')
plt.xlabel('Count')
plt.tight_layout()
plt.show()


# Create Category:Misconception pairs
train_df['Category_Misconception'] = train_df['Category'] + ':' + train_df['Misconception']

# Check the distribution of Category:Misconception pairs
category_misconception_counts = train_df['Category_Misconception'].value_counts()

plt.figure(figsize=(14, 10))
sns.barplot(x=category_misconception_counts.values[:20], y=category_misconception_counts.index[:20])
plt.title('Top 20 Category:Misconception Pairs')
plt.xlabel('Count')
plt.tight_layout()
plt.show()


# Examine the length of student explanations
train_df['explanation_length'] = train_df['StudentExplanation'].apply(len)

plt.figure(figsize=(10, 6))
sns.histplot(data=train_df, x='explanation_length', bins=50)
plt.title('Distribution of Student Explanation Lengths')
plt.xlabel('Length (characters)')
plt.axvline(x=train_df['explanation_length'].median(), color='r', linestyle='--', label=f"Median: {train_df['explanation_length'].median():.1f}")
plt.legend()
plt.show()


# Examine the relation between explanation length and Category
plt.figure(figsize=(12, 8))
sns.boxplot(x='Category', y='explanation_length', data=train_df)
plt.title('Explanation Length by Category')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep mathematical symbols
        text = re.sub(r'[^\w\s+\-*/=()<>\[\]{}.,;:!?%$#@&|~^\'\"\\\_]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ''

# Apply text cleaning to StudentExplanation and QuestionText columns
train_df['cleaned_explanation'] = train_df['StudentExplanation'].apply(clean_text)
train_df['cleaned_question'] = train_df['QuestionText'].apply(clean_text)

test_df['cleaned_explanation'] = test_df['StudentExplanation'].apply(clean_text)
test_df['cleaned_question'] = test_df['QuestionText'].apply(clean_text)


# Create combined features
train_df['combined_text'] = 'Question: ' + train_df['cleaned_question'] + ' Answer: ' + train_df['MC_Answer'] + ' Explanation: ' + train_df['cleaned_explanation']
test_df['combined_text'] = 'Question: ' + test_df['cleaned_question'] + ' Answer: ' + test_df['MC_Answer'] + ' Explanation: ' + test_df['cleaned_explanation']

# Display a few examples of the combined text
train_df[['combined_text', 'Category', 'Misconception']].sample(2)


# Create a list of all unique Category:Misconception combinations
all_labels = train_df['Category_Misconception'].unique().tolist()
label2id = {label: idx for idx, label in enumerate(all_labels)}
id2label = {idx: label for idx, label in enumerate(all_labels)}

print(f"Number of unique labels: {len(all_labels)}")


# Split the data into training and validation sets
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['Category']
)

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")


# Custom dataset class for transformers
class MisconceptionDataset(Dataset):
    def __init__(self, texts, labels=None, label2id=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove batch dimension added by the tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        if self.labels is not None:
            label = self.labels[idx]
            label_id = self.label2id[label]
            encoding['labels'] = torch.tensor(label_id)
            
        return encoding


# Define function to calculate MAP@3
def map_at_3(predictions, labels):
    """Calculate the Mean Average Precision @ 3 for predictions."""
    aps = []
    for pred, true_label in zip(predictions, labels):
        ap = 0
        hits = 0
        for i, p in enumerate(pred[:3]):
            if p == true_label and p not in pred[:i]:
                hits += 1
                ap += hits / (i + 1)
                break
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0


# Define the model architecture
class MisconceptionClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super(MisconceptionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        # Use the [CLS] token for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


# Initialize tokenizer and model
model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create datasets
train_dataset = MisconceptionDataset(
    texts=train_data['combined_text'].tolist(),
    labels=train_data['Category_Misconception'].tolist(),
    label2id=label2id,
    tokenizer=tokenizer,
    max_length=384
)

val_dataset = MisconceptionDataset(
    texts=val_data['combined_text'].tolist(),
    labels=val_data['Category_Misconception'].tolist(),
    label2id=label2id,
    tokenizer=tokenizer,
    max_length=384
)

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MisconceptionClassifier(model_name, num_labels=len(all_labels))
model.to(device)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# Training function
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return epoch_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, device, id2label):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
                
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item()
            
            # Get top 3 predictions for each sample
            batch_preds = torch.topk(outputs, k=3, dim=1).indices.cpu().numpy()
            batch_preds = [[id2label[idx] for idx in pred_list] for pred_list in batch_preds]
            
            all_preds.extend(batch_preds)
            all_labels.extend([id2label[idx.item()] for idx in labels.cpu()])
    
    # Calculate MAP@3
    map3 = map_at_3(all_preds, all_labels)
    
    return val_loss / len(data_loader), map3, all_preds


# Training loop
epochs = 3
best_map3 = 0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Training loss: {train_loss:.4f}")
    
    # Evaluate
    val_loss, val_map3, _ = evaluate(model, val_loader, device, id2label)
    print(f"Validation loss: {val_loss:.4f}, MAP@3: {val_map3:.4f}")
    
    # Save the best model
    if val_map3 > best_map3:
        best_map3 = val_map3
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Saved best model with MAP@3: {best_map3:.4f}")

print(f"\nTraining complete! Best MAP@3: {best_map3:.4f}")


# Load the best model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Create test dataset
test_dataset = MisconceptionDataset(
    texts=test_df['combined_text'].tolist(),
    tokenizer=tokenizer,
    max_length=384
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Make predictions
all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        outputs = model(input_ids, attention_mask, token_type_ids)
        
        # Get top 3 predictions for each sample
        batch_preds = torch.topk(outputs, k=3, dim=1).indices.cpu().numpy()
        batch_preds = [[id2label[idx] for idx in pred_list] for pred_list in batch_preds]
        
        all_predictions.extend(batch_preds)


# Create submission file
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': [' '.join(pred) for pred in all_predictions]
})

submission.to_csv('submission.csv', index=False)
submission.head()


# Define models for ensemble
model_names = [
    'microsoft/deberta-v3-base',  # Main model
    'roberta-base',              # Secondary model
]

# We would train each model separately and save their outputs
# For brevity, we'll simulate the ensemble process

def ensemble_predictions(model_outputs, id2label, top_k=3):
    """Ensemble predictions from multiple models by averaging logits"""
    # Average the logits from all models
    ensemble_logits = sum(model_outputs) / len(model_outputs)
    
    # Get top k predictions
    top_indices = torch.topk(ensemble_logits, k=top_k, dim=1).indices.cpu().numpy()
    top_predictions = [[id2label[idx] for idx in pred_list] for pred_list in top_indices]
    
    return top_predictions

# In a real implementation, we would:
# 1. Train each model separately
# 2. Save their logits on the test set
# 3. Ensemble the logits to get final predictions

