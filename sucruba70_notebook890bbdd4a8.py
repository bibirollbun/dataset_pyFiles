import os
import numpy as np
import pandas as pd
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import get_cosine_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from torchinfo import summary
from tqdm import tqdm

import wandb
from kaggle_secrets import UserSecretsClient


from torch.amp import autocast, GradScaler


root_dir = '/kaggle/input/jigsaw-agile-community-rules'
train_dir = os.path.join(root_dir, 'train.csv')
test_dir = os.path.join(root_dir, 'test.csv')


### 데이터 로드
train_df = pd.read_csv(train_dir)
test_df = pd.read_csv(test_dir)

train_df.info()


### 데이터 전처리
def preprocessing(text):
    text = text.lower().strip()  # 소문자화 및 strip
    # url/email/handle etc
    # markdown
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text) # bold, __text__ 제거
    text = text = re.sub(r'(\*|_)(.*?)\1', r'\2', text) # italic 제거
    text = re.sub(r'~~(.*?)~~', r'\1', text) # strike 제거
    text = re.sub(r'([-=>|<])\1{1,}', ' ', text) # # 2개 이상 반복되는 -, =, >, <, | 등은 공백으로 대체
    text = re.sub(r'([.,!?;:]){2,}', r'\1', text) # 연속된 구두점 한개로 변경
    text = re.sub(r'\s+', ' ', text) # 연속 공백 제거
    text = text.strip()

    return text

def apply_preprocessing_to_all_cols(df):
    for col in [
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
        "body"
    ]:
        df[f"{col}"] = df[col].apply(preprocessing)
    return df


def make_prompt(row):
    prompt = (
        f"Rule: {row['rule']}\n"
        f"Positive Example 1: {row['positive_example_1']}\n"
        f"Positive Example 2: {row['positive_example_2']}\n"
        f"Negative Example 1: {row['negative_example_1']}\n"
        f"Negative Example 2: {row['negative_example_2']}\n"
        f"Reddit Comment: {row['body']}\n"
    )
    return prompt


X = train_df.drop(columns=['row_id','rule_violation'])
y = train_df['rule_violation']
X = apply_preprocessing_to_all_cols(X)
X_prompt = X.apply(make_prompt, axis=1)


### 데이터셋 구성
class CustomDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length = 512):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length= self.max_length,
            return_tensors = 'pt' # 이후에 만듬.
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        
        return item


### 토크나이저&model 로드
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


collator = DataCollatorWithPadding(
    tokenizer,
    padding = 'longest',
    max_length=512
)

X_train, X_valid, y_train, y_valid = train_test_split(X_prompt, y, test_size=0.2, stratify=y, random_state=100)
X_train = X_train.reset_index(drop=True).tolist()
X_valid = X_valid.reset_index(drop=True).tolist()
y_train = y_train.reset_index(drop=True).tolist()
y_valid = y_valid.reset_index(drop=True).tolist()


train_dataset = CustomDataset(X_train, y_train, tokenizer)
train_loader = DataLoader(
    train_dataset,
    collate_fn=collator,
    shuffle=True,
    batch_size=32,
    pin_memory=True,
    num_workers=2
)

valid_dataset = CustomDataset(X_valid, y_valid, tokenizer)
valid_loader = DataLoader(
    valid_dataset,
    collate_fn=collator,
    shuffle=False,
    batch_size=32,
    pin_memory=True,
    num_workers=2
)


for i, batch in enumerate(train_loader):
    print(f"[Batch {i}]")
    for k, v in batch.items():
        print(k, v.shape)
    if i > 2:  # 3개만 보면 break
        break


batch = next(iter(train_loader))
summary(model, input_data={'input_ids':batch['input_ids'],'attention_mask':batch['attention_mask']})


### GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)    

model = model.cuda()


optimizer = AdamW(
    #filter(lambda p: p.requires_grad, model.parameters()),
    model.parameters(),
    lr=2e-5,
    weight_decay=1e-3
)

num_epochs = 5

# scheduler
training_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * training_steps)


scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=training_steps,
    num_cycles=0.5
)


user_secrets = UserSecretsClient()
secret_value = user_secrets.get_secret("wandb")
os.environ["WANDB_API_KEY"] = secret_value


wandb.login()


wandb.init(project="reddit", name="0801_1")


!nvidia-smi



scaler = GradScaler("cuda")

for epoch in range(1,num_epochs+1):
    model.train()
    train_iterator = tqdm(train_loader,
                          desc=f"[Training] {epoch}/{num_epochs}",
                          leave=False)
    
    train_loss = 0.0
    all_preds, all_labels = [], []
    for batch in train_iterator:
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(non_blocking=True)
        labels = batch['labels'].cuda(non_blocking=True)
        
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits, loss = outputs.logits, outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()          

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds) 
    train_f1 = f1_score(all_labels, all_preds, average='macro')

    wandb.log({
        "train/loss": avg_train_loss,
        "train/acc": train_acc,
        "train/f1": train_f1,
        "epoch": epoch
    })

    print(f"[Train] Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} F1: {train_f1:.4f}")

    model.eval()
    valid_iterator = tqdm(valid_loader,
                          desc=f"[Validation] {epoch}/{num_epochs}",
                          leave=False)
    valid_loss = 0.0
    all_preds_val, all_labels_val = [], []
    with torch.no_grad():
        for batch in valid_iterator:
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(non_blocking=True)
            labels = batch['labels'].cuda(non_blocking=True)

            with autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                if loss.dim() > 0:
                    loss = loss.mean()
                    
            preds = torch.argmax(logits, dim=1)
            valid_loss += loss.item()
            all_preds_val.extend(preds.cpu().tolist())
            all_labels_val.extend(labels.cpu().tolist())
        
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_acc = accuracy_score(all_labels_val, all_preds_val) 
        valid_f1 = f1_score(all_labels_val, all_preds_val, average='macro')

    wandb.log({
        "valid/loss": avg_valid_loss,
        "valid/acc": valid_acc,
        "valid/f1": valid_f1,
        "epoch": epoch
    })

    print(f"[Validation] Loss: {avg_valid_loss:.4f}, Acc: {valid_acc:.4f} F1: {valid_f1:.4f}")


test_X = test_df.drop(columns=['row_id'])
test_X = apply_preprocessing_to_all_cols(test_X)
test_X_prompt = test_X.apply(make_prompt, axis=1)

### 데이터셋 구성
class TestDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length = 512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self,idx):
        prompt = self.prompts[idx]
        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length= self.max_length,
            return_tensors = 'pt' # 이후에 만듬.
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        
        return item


test_dataset = TestDataset(test_X_prompt, tokenizer)
test_loader = DataLoader(
    test_dataset,
    collate_fn=collator,
    shuffle=False,
    batch_size=32,
    pin_memory=True,
    num_workers=2
)


model.eval()
responses = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        probs = torch.softmax(logits,dim=1)[:,-1]
    responses.extend(probs.cpu().tolist())


responses


test_df['row_id']


my_submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': responses
})


my_submission.to_csv('submission.csv', index=False)




