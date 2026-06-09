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

# Train dataset load
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# Label creation: Category + Misconception
def combine_label(row):
    if "Misconception" in row['Category']:
        return f"{row['Category']}:{row['Misconception']}"
    else:
        return f"{row['Category']}:NA"

train["Label"] = train.apply(combine_label, axis=1)

# Keep only necessary columns
train = train[["QuestionText", "StudentExplanation", "Label"]]

# Print number of unique labels
print("Unique Labels:", train["Label"].nunique())
train.head()



from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

le = LabelEncoder()
train['LabelEnc'] = le.fit_transform(train['Label'])

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

def tokenize_function(examples):
    return tokenizer(
        examples["QuestionText"] + " " + examples["StudentExplanation"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )



import torch
from torch.utils.data import Dataset, DataLoader

class MathMisconceptionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.iloc[idx]['QuestionText']
        explanation = self.df.iloc[idx]['StudentExplanation']
        inputs = self.tokenizer(
            question + " " + explanation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        label = torch.tensor(self.df.iloc[idx]['LabelEnc'], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

# Dataset and Make DataLoader 
train_dataset = MathMisconceptionDataset(train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)



import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small",
    num_labels=train['LabelEnc'].nunique()
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()



train_losses = []
train_accuracies = []

model.train()
for epoch in range(5):
    loop = tqdm(train_loader, leave=True)
    running_loss = 0
    correct = 0
    total = 0
    
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")



model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)

from sklearn.metrics import classification_report

true_labels = train['LabelEnc'].values
print(classification_report(true_labels, predictions, target_names=le.classes_))



test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.df.iloc[idx]['QuestionText']
        explanation = self.df.iloc[idx]['StudentExplanation']
        inputs = self.tokenizer(
            question + " " + explanation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

test_dataset = TestDataset(test, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)



model.eval()
test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)



import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Loss Plot
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')

# Accuracy Plot
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Epochs')

plt.show()



model.eval()
test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)



import pandas as pd
import os

submission = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')
submission['prediction'] = le.inverse_transform(test_preds)
submission.to_csv('submission.csv', index=False)

submission_file = 'submission.csv'

if os.path.exists(submission_file):
    print(f"{submission_file} has been created successfully. You can now submit it to Kaggle.")
else:
    print(f"{submission_file} was not found. Please check your submission code.")


