import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split

# Load all competition files
train_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    return text

# Apply preprocessing
train_df['processed_text'] = train_df['comment_text'].apply(preprocess_text)
test_df['processed_text'] = test_df['comment_text'].apply(preprocess_text)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dataset class
class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )
        
        item = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item

# Prepare data
TARGETS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X_train, X_val, y_train, y_val = train_test_split(
    train_df['processed_text'].values,
    train_df[TARGETS].values,
    test_size=0.2,
    random_state=42
)

# Create datasets
train_dataset = ToxicCommentDataset(X_train, y_train)
test_dataset = ToxicCommentDataset(test_df['processed_text'].values)

# Model definition
class ToxicRoBERTa(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, 6)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Initialize model
model = ToxicRoBERTa()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Class weights
pos_counts = train_df[TARGETS].sum()
neg_counts = len(train_df) - pos_counts
pos_weights = (neg_counts / pos_counts).values
pos_weights = torch.tensor(pos_weights, device=device)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = AdamW(model.parameters(), lr=2e-5)

# DataLoaders with reduced batch size for Kaggle memory
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training loop with progress bar
from tqdm import tqdm

for epoch in range(3):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss/(progress_bar.n+1)})

# Generate predictions aligned with sample submission
model.eval()
test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Predicting'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask)
        probas = torch.sigmoid(outputs).cpu().numpy()
        test_preds.extend(probas)

# Create submission dataframe using sample submission format
submission = sample_sub.copy()
submission[TARGETS] = np.array(test_preds)

# Save to Kaggle working directory
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved to /kaggle/working/submission.csv")


# Correct paths for Kaggle
!ls /kaggle/input/jigsaw-toxic-comment-classification-challenge/
train_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

