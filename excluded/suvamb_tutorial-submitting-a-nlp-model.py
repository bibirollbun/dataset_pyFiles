import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if not iskaggle:
    !pip install -q transformers datasets scikit-learn fsspec 



from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm


path = Path('../input/us-patent-phrase-to-phrase-matching')
df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')

# Load full training set
df = pd.read_csv(path/'train.csv')  # contains anchor, target, context, score

# Do train/val split yourself
anchors = df['anchor'].unique()
np.random.seed(42)
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = df['anchor'].isin(val_anchors)
train_df = df[~is_val].reset_index(drop=True)
val_df   = df[is_val].reset_index(drop=True)



# Load model/tokenizer once to cache
model_name = '/kaggle/input/cached/mv1'
tokz = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
model = model.cuda()
sep = tokz.sep_token

#model_name = '/kaggle/working/hf/models--microsoft--deberta-v3-small'
#tokz = AutoTokenizer.from_pretrained(model_name, cache_dir='/kaggle/working/hf')
#model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, cache_dir='/kaggle/working/hf')


# Construct the input text
df['inputs'] = df.context + sep + df.anchor + sep + df.target

# Optional: convert labels to float if it's a regression task
df['label'] = df['score'].astype(np.float32)

# Train/val split based on anchor
anchors = df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = df.anchor.isin(val_anchors)
train_df = df[~is_val].reset_index(drop=True)
val_df = df[is_val].reset_index(drop=True)

print("Train/Val size:", len(train_df), len(val_df))
print("Train/Val score means:", train_df.label.mean(), val_df.label.mean())


class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['inputs'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Flatten from (1, seq_len) to (seq_len,)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['label'] = torch.tensor(row['label'], dtype=torch.float)
        return item



BATCH_SIZE = 16 

train_ds = TextPairDataset(train_df, tokz)
val_ds = TextPairDataset(val_df, tokz)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)


train_ds = TextPairDataset(train_df, tokz)
val_ds = TextPairDataset(val_df, tokz)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)



lr = 8e-5
bs = 32
wd = 0.01
epochs = 1
warmup_ratio = 0.1



optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

num_training_steps = len(train_dl) * epochs
num_warmup_steps = int(num_training_steps * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

loss_fn = nn.MSELoss()



def pearsonr(x, y):
    return np.corrcoef(x, y)[0, 1]



for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1} Training"):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].unsqueeze(1).cuda()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dl)

    # --- Validation ---
    model.eval()
    val_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Epoch {epoch+1} Validation"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].unsqueeze(1).cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits

            loss = loss_fn(preds, labels)
            val_loss += loss.item()

            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    pearson = pearsonr(preds_all.flatten(), labels_all.flatten())
    avg_val_loss = val_loss / len(val_dl)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Pearson = {pearson:.4f}")



test_df['inputs'] = test_df.context + sep + test_df.anchor + sep + test_df.target



class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['inputs'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['id'] = row['id']
        return item

test_ds = TestDataset(test_df, tokz)
test_dl = DataLoader(test_ds, batch_size=128)



model.eval()
pred_ids = []
pred_scores = []

with torch.no_grad():
    for batch in tqdm(test_dl, desc="Predicting"):
        ids = batch.pop('id')
        inputs = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1).cpu().numpy()

        pred_ids.extend(ids)
        pred_scores.extend(preds)



submission = pd.DataFrame({'id': pred_ids, 'score': pred_scores})
submission.to_csv("submission.csv", index=False)



!cat submission.csv


!cat /kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv




