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

# Load training data
df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")  # update path as needed

# Rename column for consistency
df.rename(columns={"generated": "label"}, inplace=True)

# Preview the data
print("ğŸ“„ Data Preview:")
print(df.head())

# Check for nulls
print("\nğŸ§¼ Missing values:")
print(df.isnull().sum())

# Class balance (0 = human, 1 = AI)
print("\nğŸ“Š Label distribution:")
print(df['label'].value_counts())

# Plot distribution
df['label'].value_counts().plot(kind='bar', title='Class Distribution')
plt.xticks([0, 1], ['Human', 'AI'])
plt.ylabel('Count')
plt.show()



# Load prompt metadata
prompts_df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_prompts.csv")  # update path

# Preview prompt file
print("ğŸ“Œ Prompt Metadata:")
print(prompts_df.head())

# Merge into training data
df_merged = df.merge(prompts_df, on="prompt_id", how="left")

# Show result
print("\nğŸ“� Merged Dataset Sample:")
print(df_merged[['text', 'prompt_name', 'instructions', 'label']].head(3))

# Check for nulls again
print("\nâ�“ Nulls in merged data:")
print(df_merged.isnull().sum())




# Build the final input text by combining prompt + essay
df_merged['input_text'] = "[PROMPT] " + df_merged['instructions'] + " [ESSAY] " + df_merged['text']

# Keep only what we need
df_final = df_merged[['input_text', 'label']]

# Check class counts again
print("ğŸ“Š Label counts before split:")
print(df_final['label'].value_counts())

# Split into training and validation sets
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_final['input_text'].tolist(),
    df_final['label'].tolist(),
    test_size=0.1,
    stratify=df_final['label'],  # keeps label balance
    random_state=42
)

print(f"âœ… Train size: {len(train_texts)}, Val size: {len(val_texts)}")



from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Load a Huggingface tokenizer (you can swap for BERT, ALBERT, DeBERTa, etc.)
model_name = "albert-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom PyTorch Dataset class
class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Create datasets
train_ds = EssayDataset(train_texts, train_labels, tokenizer)
val_ds = EssayDataset(val_texts, val_labels, tokenizer)

# Create dataloaders
batch_size = 4
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

print("âœ… Tokenization complete. Dataloaders are ready.")



import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AdamW

# Set device (MPS for Mac or fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"ğŸ“¡ Using device: {device}")

# Define the model class
class BERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]  # Take the [CLS] token
        return self.classifier(cls_token)

# Initialize model
model = BERTClassifier(model_name).to(device)

# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)



from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Run 1 training epoch
print("ğŸš€ Starting training...")
model.train()
for epoch in range(1):
    loop = tqdm(train_loader, total=len(train_loader))
    for step, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask).view(-1)
        loss = loss_fn(logits, labels)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
        
        # Optional logging
        if (step + 1) % 10 == 0:
            print(f"ğŸ§  Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

print("âœ… Training complete!")



print("ğŸ§ª Evaluating on validation set...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()

        logits = model(input_ids, attention_mask).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels)

# Final accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"âœ… Validation Accuracy: {acc:.4f}")



# ğŸ“„ Load test file
test_df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")  # update path
test_texts = test_df['text'].tolist()
ids = test_df['id'].tolist()

# If you want to include prompt context (optional, depending on test format)
# For now we just use text alone
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

# Create test loader
test_ds = TestDataset(test_texts, tokenizer)
test_loader = DataLoader(test_ds, batch_size=4)

# ğŸ”� Make predictions
model.eval()
pred_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        pred_probs.extend(probs)

# âœ… Save predictions to CSV
import pandas as pd
sub_df = pd.DataFrame({'id': ids, 'generated': pred_probs})
sub_path = "submission_bbt1.csv"
sub_df.to_csv(sub_path, index=False)
print(f"ğŸ“¦ Submission saved to: {sub_path}")





