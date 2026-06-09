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


import torch
import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertModel,AutoTokenizer,AutoModelForMaskedLM,get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm

train_df=pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
test_df=pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')
submission=pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv')


### Check all datas
print(f'train_df.head:\n{train_df.head}')
print(f'test_df.head:\n{test_df.head}')
print(f'submission.head:\n{submission.head}')


### Cheak data types and missing values in the meantime
print(pd.concat([train_df.dtypes,train_df.isnull().sum()],axis=1,
          keys=['train_data-type','train_data-null_values']).sort_values(by='train_data-null_values',ascending=False))

print(pd.concat([test_df.dtypes,test_df.isnull().sum()],axis=1,
          keys=['test_data-type','test_data-null_values']).sort_values(by='test_data-null_values',ascending=False))

### There are no missing values in the training data/ testing data. Let's go on!


### Cheak the questions lables of training/testing data
train_labels = train_df.groupby('label').size().reset_index(name='Count')
train_labels['Percentage(%)'] = (train_labels['Count'] / len(train_df) * 100).round(2)
print(train_labels)

# print(train.groupby('label').size())
# print(train.groupby('label').size().reset_index(name='Count'))


### Use bert-base-uncased to compute tokens
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
print("successfully")


### Compute tokens of each text
train_df['token_count'] = train_df['Question'].apply(
  lambda x: len(tokenizer.encode(str(x), add_special_tokens=True))
  )

# save training data with tokens
train_df.to_csv("trian_with_tokens.csv",index=False)

#Select all data rows whrer tokens>512. 
train_tokens_greater_than_512=train_df[train_df['token_count']>512]

### Check details 
print("Number and proportion of training data with tokens greater than 512:")
print(f"number:{len(train_tokens_greater_than_512)}")
print(f'proportion:{(len(train_tokens_greater_than_512)/len(train_df)*100)}%')

## We can see the proportion with tokens greater than 512 is little small. 
## Set the maximum input length to 512 in subsequent data modeling;
## For text with tokens greater than 512, the part greater than 512 will be automatically truncated. 
## Due to the small proportion of this part, impact by this data truncation can be ignored


test_df['token_count'] = test_df['Question'].apply(
  lambda x: len(tokenizer.encode(str(x), add_special_tokens=True))
  )

### save testing data with tokens
test_df.to_csv("test_with_tokens.csv",index=False)

### Select all data rows whrer tokens>512. 
test_tokens_greater_than_512=test_df[test_df['token_count']>512]

### Check details 
print("Number and proportion of testing data with tokens greater than 512:")
print(f"number:{len(test_tokens_greater_than_512)}")
print(f'proportion:{(len(test_tokens_greater_than_512)/len(test_df)*100)}%')

### Similarly, on the testing data, the proportion of data with tokens greater than 512 is also very small.


### 1.Check if there are available GPU/CUDA devices
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


### 2.Load tokenizer and base_model
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
# base_model = AutoModelForMaskedLM.from_pretrained("tbs17/MathBERT").to(device)

tokenizer = BertTokenizer.from_pretrained("tbs17/MathBERT")
base_model = BertModel.from_pretrained("tbs17/MathBERT").to(device)

### Recommend BertTokenizer and BertModel rather than AutoTokenizer and AutoModelForMaskedLM from huggingface


### 3. Set custom classification model for classifying math problems (8 labels (0 - 7))
class MathClassifier(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(MathClassifier, self).__init__()
        self.bert = base_model
        self.dropout = torch.nn.Dropout(0.3) 
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  #[CLS]
        cls_output = self.dropout(cls_output) #setting dropout
        logits = self.classifier(cls_output)
        return logits

model = MathClassifier().to(device)


### 4.Data Encoding function
def encode_data(texts, labels=None, tokenizer=None, max_length=512):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    if labels is not None:
        labels_tensor = torch.tensor([label if label in range(8) else -1 for label in labels])
        dataset = torch.utils.data.TensorDataset(
          encodings['input_ids'], 
          encodings['attention_mask'], 
          labels_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'], 
            encodings['attention_mask']
        )
    return dataset

### 5.Split training set and validation set
## Due to the limited samples of some categories(such as label 6,label 7), 
## stratified sampling is used to ensure that samples from all categories exist on both the training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["Question"], 
    train_df["label"],
    test_size=0.15, 
    random_state=42,
    stratify=train_df["label"]
)

### Encode datas
train_dataset = encode_data(train_texts, train_labels, tokenizer)
val_dataset = encode_data(val_texts, val_labels, tokenizer)
test_dataset = encode_data(test_df["Question"], tokenizer=tokenizer)  

### Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



### 6.Define optimizer and lr_scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 10
num_training_steps = num_epochs * len(train_loader)

num_warmup_steps = int(0.1 * num_training_steps)  # set warmup
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


### Storage metrics for plotting
history = {
    'train_loss': [], 'val_loss': [],
    'train_f1_micro': [], 'val_f1_micro': []
}

best_f1_micro = 0
patience_counter = 0
early_stop_patience = 3

### 7.Start trainingï¼ˆwith progress bar and time displayï¼‰
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    ### -----training----- ###
    model.train()
    total_train_loss = 0
    all_train_preds = []
    all_train_labels = []

    start_time = time.time()
    train_pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in train_pbar:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    train_duration = time.time() - start_time
    print(f"Training finished in {train_duration:.2f}s | Avg Loss: {avg_train_loss:.4f}")

    ### -----validation----- ###
    model.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []

    start_time = time.time()
    val_pbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in val_pbar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            logits = model(input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

            val_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss = total_val_loss / len(val_loader)
    val_duration = time.time() - start_time
    print(f"Validation finished in {val_duration:.2f}s | Avg Loss: {avg_val_loss:.4f}")

    ### -----compute metrics----- ###
    def compute_metrics(y_true, y_pred):
        f1_micro = f1_score(y_true, y_pred, average='micro')
        return f1_micro

    train_f1_micro = compute_metrics(all_train_labels, all_train_preds)
    val_f1_micro = compute_metrics(all_val_labels, all_val_preds)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_f1_micro'].append(train_f1_micro)
    history['val_f1_micro'].append(val_f1_micro)

    print(f"[Train] F1-Micro: {train_f1_micro:.4f}")
    print(f"[Val]   F1-Micro: {val_f1_micro:.4f}")

    ### -----Add early_stopping and save best model----- ###
    if val_f1_micro > best_f1_micro:
        best_f1_micro = val_f1_micro
        patience_counter = 0
        torch.save(model.state_dict(), "best_mathbert_model.pth")
        print("Best model saved.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{early_stop_patience}")

    if patience_counter >= early_stop_patience:
        print("Early stopping triggered.")
        break


### 8.Plot Loss and F1-Micro of training and validation sets
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_f1_micro'], label='Train F1-Micro')
plt.plot(history['val_f1_micro'], label='Val F1-Micro')
plt.xlabel('Epoch')
plt.ylabel('F1-Micro')
plt.title('Training and Validation F1-Micro')
plt.legend()
plt.tight_layout()
plt.show()


### 9.Predict labels of testing set 
## Define predict function
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    start_time = time.time()
    test_pbar = tqdm(data_loader, desc="Predicting", leave=False)

    with torch.no_grad():
        for batch in test_pbar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask = batch
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    duration = time.time() - start_time
    print(f"Prediction finished in {duration:.2f}s")
    return predictions

## Load final_model and predict
model.load_state_dict(torch.load("best_mathbert_model.pth"))
test_predictions = predict(model, test_loader, device)

## Save the predicted values of the testing set labels to the submission file
submission['label']=test_predictions
submission.to_csv("submission_new.csv",index=False)

