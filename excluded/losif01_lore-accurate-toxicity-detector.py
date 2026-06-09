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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')


print("Train shape:", train.shape);print("Test shape:", test.shape)
print(test_labels.head())
print(sample_submission.head())




train.head(100)


test.head()


sample_submission.head()


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train[label_cols].sum().sort_values(ascending=False).plot(kind='bar', figsize=(8,5), color='blue')
plt.title("Number of Comments per Toxic Class")
plt.ylabel("Count")
plt.xticks(rotation=60)
plt.show()


train['label_sum'] = train[label_cols].sum(axis=1)

train['label_sum'].value_counts().sort_index().plot(kind='bar', color='g')
plt.title("Multi-Label Distribution")
plt.xlabel("Number of Toxic Tags per Comment")
plt.ylabel("Number of Comments")
plt.show()


import random
for label in label_cols:
    print(f"\n\n Example of '{label}':\n")
    example = train[train[label] == 1]['comment_text'].iloc[random.randint(0,100)]
    print(example)


example = train[train[label] == 1]['comment_text']
example.head(100)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


train['clean_text'] = train['comment_text'].apply(clean_text)
test['clean_text'] = test['comment_text'].apply(clean_text)



!pip install transformers




import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Prepare data
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Tokenizer setup
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 128
BATCH_SIZE = 32


# PyTorch Dataset
class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Create datasets
train_dataset = ToxicCommentsDataset(
    train_df['clean_text'].values,
    train_df[label_cols].values,
    tokenizer,
    MAX_LEN
)

val_dataset = ToxicCommentsDataset(
    val_df['clean_text'].values,
    val_df[label_cols].values,
    tokenizer,
    MAX_LEN
)


# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model setup
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)




# Training function
def train_model(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
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
        optimizer.step()
    
    return total_loss / len(data_loader)

# Validation function
def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
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
    
    return total_loss / len(data_loader)




# Training loop
EPOCHS = 2
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss = train_model(model, train_loader, optimizer, device)
    val_loss = eval_model(model, val_loader, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")



# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/loss_curve.png')
plt.show()


# Generate predictions for heatmap
def get_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)




# Get predictions
val_predictions, val_true = get_predictions(model, val_loader, device)

# Generate confusion matrix for 'toxic' label
toxic_idx = label_cols.index('toxic')
toxic_preds = (val_predictions[:, toxic_idx] > 0.5).astype(int)
toxic_true = val_true[:, toxic_idx].astype(int)

cm = confusion_matrix(toxic_true, toxic_preds)




# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Toxic', 'Toxic'], 
            yticklabels=['Not Toxic', 'Toxic'])
plt.title('Confusion Matrix for Toxic Label')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('/kaggle/working/confusion_matrix.png')
plt.show()

print("Training complete!")

