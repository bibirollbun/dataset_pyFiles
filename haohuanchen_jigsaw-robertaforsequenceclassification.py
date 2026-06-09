import os
from transformers import logging
logging.set_verbosity_error()

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# train_df = train_df[:200]


from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("/kaggle/input/m/haohuanchen/roberta-base/transformers/default/1")


import matplotlib.pyplot as plt

rule_lens = []
body_lens = []
token_lens = []

for rule, body in zip(train_df["rule"], train_df["body"]):
    rule_lens.append(len(rule))
    body_lens.append(len(body))
    encoded = tokenizer(rule, body, truncation=False)
    token_lens.append(len(encoded["input_ids"]))


plt.hist(rule_lens, bins=50)
plt.xlabel("Rule Length")
plt.ylabel("Frequency")
plt.title("Rule Length Distribution")
plt.show()

print(f"Rule最大长度: {max(rule_lens)}")
print(f"95%分位数: {np.percentile(rule_lens, 95)}")
print(f"99%分位数: {np.percentile(rule_lens, 99)}")


plt.hist(body_lens, bins=50)
plt.xlabel("Body Length")
plt.ylabel("Frequency")
plt.title("Body Length Distribution")
plt.show()

print(f"Body最大长度: {max(body_lens)}")
print(f"95%分位数: {np.percentile(body_lens, 95)}")
print(f"99%分位数: {np.percentile(body_lens, 99)}")


plt.hist(token_lens, bins=50)
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.title("Token Length Distribution")
plt.show()

print(f"Token最大长度: {max(token_lens)}")
print(f"95%分位数: {np.percentile(token_lens, 95)}")
print(f"99%分位数: {np.percentile(token_lens, 99)}")


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)


# print(train_df.head())


# print(train_df.isnull().sum())
# print(test_df.isnull().sum())


from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = list(df['rule'])
        self.bodies = list(df['body'])
        self.labels = list(df['rule_violation'])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        rule = str(self.texts[idx])
        body = str(self.bodies[idx])
        label = int(self.labels[idx])

        encoded = self.tokenizer(rule, body, padding='max_length', truncation='longest_first', max_length=self.max_len, return_tensors="pt")
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.rules = list(df['rule'])
        self.bodies = list(df['body'])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        rule = str(self.rules[idx])
        body = str(self.bodies[idx])

        encoded = self.tokenizer(rule, body, padding='max_length', truncation='longest_first', max_length=self.max_len, return_tensors="pt")
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


max_len = 256

train_ds = TrainDataset(train_df, tokenizer, max_len=max_len)
val_ds = TrainDataset(val_df, tokenizer, max_len=max_len)
test_ds = TestDataset(test_df, tokenizer, max_len=max_len)


from torch.utils.data import DataLoader

batch_size = 32

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


from transformers import RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained("/kaggle/input/m/haohuanchen/roberta-base/transformers/default/1", num_labels=1)
model = model.to(device)


from torch.optim import AdamW
from transformers import get_scheduler

epochs = 5

optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


from tqdm import tqdm

def train(model, dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


from sklearn.metrics import roc_auc_score

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy().squeeze()
            preds.extend(logits)
            labels.extend(labels_batch)
    preds = torch.sigmoid(torch.tensor(preds)).numpy()
    return roc_auc_score(labels, preds)


for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_auc = evaluate(model, val_loader)
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")


model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze().cpu()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.numpy())


submission = pd.DataFrame({'row_id': test_df['row_id'],'rule_violation': all_preds})
submission.to_csv("submission.csv", index=False)

print("success!")


plt.hist(all_preds, bins=50)
plt.xlabel("Preds")
plt.ylabel("Frequency")
plt.title("Preds Distribution")
plt.show()

print(f"最大值: {np.max(all_preds)}")
print(f"最小值: {np.min(all_preds)}")
print(f"中位数: {np.median(all_preds)}")

