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


import os
import gc
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()


# Config
MODEL_NAME ="/kaggle/input/deberta-v3-large/transformers/default/1/deberta-v3-large"
MAX_LEN = 384
BATCH_SIZE = 4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load data
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


# Combine text fields for better context
def create_input(row):
    return (
        row['rule'] + ' [SEP] ' + row['body'] +
        ' [SEP] POS1: ' + str(row['positive_example_1']) +
        ' [SEP] POS2: ' + str(row['positive_example_2']) +
        ' [SEP] NEG1: ' + str(row['negative_example_1']) +
        ' [SEP] NEG2: ' + str(row['negative_example_2'])
    )
df['text'] = df.apply(create_input, axis=1)
df_test['text'] = df_test.apply(create_input, axis=1)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x='rule_violation', data=df)
plt.title('Label Distribution')
plt.show()

plt.figure(figsize=(8, 4))
df['rule_text_length'] = df['rule'].apply(lambda x: len(str(x).split()))
df['comment_text_length'] = df['body'].apply(lambda x: len(str(x).split()))
sns.histplot(df['comment_text_length'], bins=50, kde=True)
plt.title('Comment Length Distribution')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='rule_violation', y='comment_text_length', data=df)
plt.title('Comment Length by Label')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='rule_violation', y='rule_text_length', data=df)
plt.title('Rule Text Length by Label')
plt.show()


print(df.columns.tolist())


# Dataset
class JigsawDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_scheduler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Define model
class JigsawModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.backbone.gradient_checkpointing_enable()
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0])  # [CLS] token
        return logits.squeeze(-1)

# Stratified K-Fold using rule + label
df['rule_label'] = df['rule'].astype(str) + '_' + df['rule_violation'].astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# For storing predictions
all_val_preds = np.zeros(len(df))
all_test_preds = np.zeros(len(df_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df['rule_label'])):
    print(f"\n===== Fold {fold + 1} =====")

    # Create datasets
    train_dataset = JigsawDataset(
        df.iloc[tr_idx]['text'].tolist(),
        df.iloc[tr_idx]['rule_violation'].tolist()
    )
    val_dataset = JigsawDataset(
        df.iloc[val_idx]['text'].tolist(),
        df.iloc[val_idx]['rule_violation'].tolist()
    )
    test_dataset = JigsawDataset(df_test['text'].tolist())

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model, optimizer, scheduler
    model = JigsawModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler()

    best_auc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = nn.BCEWithLogitsLoss()(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE).float()

                logits = model(input_ids, attention_mask)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        score = roc_auc_score(val_labels, val_preds)
        print(f"Fold {fold + 1} Epoch {epoch + 1} AUC: {score:.4f}")

        if score > best_auc:
            best_auc = score
            torch.save(model.state_dict(), f"best_model_fold{fold}.pt")

    # Load best model for inference
    model.load_state_dict(torch.load(f"best_model_fold{fold}.pt"))
    model.eval()

    # OOF predictions
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model(input_ids, attention_mask)
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())

    all_val_preds[val_idx] = val_preds[:len(val_idx)]

    # Test predictions
    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            fold_test_preds.extend(preds)

    all_test_preds += np.array(fold_test_preds) / skf.n_splits

    del model
    torch.cuda.empty_cache()
    gc.collect()


# Save OOF predictions
np.save("oof_preds.npy", all_val_preds)
np.save("test_preds.npy", all_test_preds)

# Compute final AUC using the correct label column
true_labels = df['rule_violation'].values.astype(float)
overall_auc = roc_auc_score(true_labels, all_val_preds)
print(f"Overall AUC: {overall_auc:.4f}")



import matplotlib.pyplot as plt

plt.hist(all_val_preds, bins=50, alpha=0.7, label="OOF Predictions")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("OOF Prediction Distribution")
plt.legend()
plt.show()



print(df_test.columns)


submission = pd.DataFrame({
    'row_id': df_test['row_id'],
    'prediction': all_test_preds
})
submission.to_csv("submission.csv", index=False)
print("saved csv file for submission")

