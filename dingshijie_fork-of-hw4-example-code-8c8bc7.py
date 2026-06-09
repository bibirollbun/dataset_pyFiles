# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

!pip install evaluate

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datasets
from datasets import Dataset, DatasetDict
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ast
from transformers import AutoTokenizer,BertTokenizer, BertForTokenClassification, get_scheduler
from torch.optim import AdamW

from tqdm import tqdm
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(42)


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


#df = pd.read_csv('/kaggle/input/2025-hw/modifyClassifier_train.csv', encoding='utf-8')
df = pd.read_csv('/kaggle/input/new-mix-1/merged_md_labeler_train_clean.csv', encoding='utf-8')

# Convert the string representations of lists in 'Label_tag' column to actual Python lists.
df['Label_tag'] = df['label'].apply(ast.literal_eval)

#print(df.head())

train_data, val_data = train_test_split(df, test_size = 0.2, random_state=42)


print(val_data)


label2id = {
    "O": 0,
    "B-Modify": 1,
    "B-Filling": 2
}
id2label = {v: k for k, v in label2id.items()}


def create_dataset(data, is_test=False):
    dataset = []
    
    for _, row in data.iterrows():
        text = row['text']

        # Tokenize and align labels
        tokenized_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512)
        
        # Include labels only for training data
        if not is_test:
            label = row['Label_tag']
            word_ids = tokenized_inputs.word_ids()  # Get word indices for each token
            
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                     # Set special tokens to -100
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                     # Assign labels to the first subword token only
                    label_ids.append(label2id[label[word_idx]])
                else:
                     # Set subsequent subword tokens to -100
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            tokenized_inputs['labels'] = label_ids
        
        # Add to the dataset
        dataset.append({
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': tokenized_inputs.get('labels') if not is_test else None
        })

    return dataset



train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data)

print(train_dataset[0])


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.dataset[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.dataset[idx]['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.dataset[idx]['labels'], dtype=torch.long)
        }

train_dataset = CustomDataset(train_dataset)
val_dataset = CustomDataset(val_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)


# You can modify your own model structure

class TokenClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(TokenClassification, self).__init__()
        # Pretrain Model
        self.bert = BertForTokenClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)
        
        # You can modfiy this part as you want
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get outptuts from your model
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        # compute the loss in training & validation
        if labels is not None:
            # CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss()
            # Need to flatten logits and labels, ignoring padding positions
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return {"loss": loss, "logits": logits}
        else:
            # Test mode
            return {"logits": logits}


# Load the pretrain model from huggingface
pretrained_model_name = "google-bert/bert-base-multilingual-cased"
num_labels = 3
model = TokenClassification(pretrained_model_name=pretrained_model_name, num_labels=num_labels)


optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Set up learning rate scheduler
num_training_steps = len(train_dataset) * 3  # Total steps = steps per epoch * total number of epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Load evaluation metrics
metric = evaluate.load("accuracy", trust_remote_code=True)

# Training and evaluation loop
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Training mode
    model.train()
    train_loop = tqdm(train_dataloader, desc="Training")
    for batch in train_loop:
        # Move batch data to device
        inputs = {key: val.to(device) for key, val in batch.items()}
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.get("loss")  # Retrive loss

        if loss is not None:
            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
            # Update progress bar to show loss
            train_loop.set_postfix(loss=loss.item())
    
    # Evaluation mode
    model.eval()
    for batch in val_dataloader:
        with torch.no_grad():
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
        
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        references = inputs["labels"]
        
        # Filter positions with -100
        active_indices = references != -100
        filtered_predictions = predictions[active_indices]
        filtered_references = references[active_indices]
        
        # Add batch data for evaluation
        metric.add_batch(
            predictions=filtered_predictions.cpu(),
            references=filtered_references.cpu()
        )
    
     # Compute evaluation results
    eval_result = metric.compute()
    print(f"Validation Results: {eval_result}")


torch.save(model.state_dict(), "model_weights.pt")


model.eval()

all_preds = []
all_labels = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


with torch.no_grad():
    for batch in val_dataloader:
        
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        references = inputs["labels"]
        
        active_indices = references != -100
        filtered_predictions = predictions[active_indices].cpu().tolist()  
        filtered_references = references[active_indices].cpu().tolist()

        all_preds.extend(filtered_predictions)
        all_labels.extend(filtered_references)


# Generate classification report
report = classification_report(all_labels, all_preds, digits=4)  # digits=4 controls the number of decimal places
print(report)

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print(conf_matrix)


cm_display = metrics.ConfusionMatrixDisplay(
                confusion_matrix = conf_matrix, 
                display_labels = [0, 1,2])
cm_display.plot()
plt.show()


#test_df = pd.read_csv('/kaggle/input/2025-hw/modifyClassifier_test.csv',encoding= 'utf-8')
test_df = pd.read_csv('/kaggle/input/new-mix-1/merged_md_labeler_test.csv',encoding= 'utf-8')
test_df['Label_tag'] = test_df['label'].apply(ast.literal_eval)



test_dataset = create_dataset(test_df)

test_dataset = CustomDataset(test_dataset)

test_dataloader = DataLoader(test_dataset, batch_size=8)


model.eval()

all_preds = []
all_labels = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


with torch.no_grad():
    for batch in test_dataloader:
        
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        references = inputs["labels"]
        
        active_indices = references != -100
        filtered_predictions = predictions[active_indices].cpu().tolist()  
        filtered_references = references[active_indices].cpu().tolist()

        all_preds.extend(filtered_predictions)
        all_labels.extend(filtered_references)


perfect_matches = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        references = inputs["labels"]

        # 逐筆資料判斷
        for pred_seq, ref_seq in zip(predictions, references):
            # 過濾 padding (-100)
            active_indices = ref_seq != -100
            filtered_pred = pred_seq[active_indices].cpu().tolist()
            filtered_ref = ref_seq[active_indices].cpu().tolist()

            # 判斷是否完全 match
            perfect_matches.append(filtered_pred == filtered_ref)

# perfect_matches 是一個布林列表，每個元素代表一筆資料是否完全正確
num_perfect = sum(perfect_matches)
total = len(perfect_matches)
print(f"{num_perfect}/{total} 完全 match")


all_preds_str = [id2label[p] for p in all_preds]
all_labels_str = [id2label[l] for l in all_labels]  # 同理

print(all_preds_str)


from sklearn.metrics import classification_report, f1_score

print(classification_report(all_labels_str, all_preds_str))
print("F1:", f1_score(all_labels_str, all_preds_str, average="macro"))


conf_matrix = confusion_matrix(all_labels_str, all_preds_str)
print(conf_matrix)


class CustomTestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.dataset[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.dataset[idx]['attention_mask'], dtype=torch.long)
        }


test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=lambda x: {
    'input_ids': torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in x], batch_first=True),
    'attention_mask': torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in x], batch_first=True),
})

all_preds = []
# Define Special tokens & padding parts
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Get model outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()
        input_ids_cpu = input_ids.cpu().numpy()  # 轉成 numpy 才能處理

        for i in range(len(preds)):
            valid_preds = []
            for j, p in enumerate(preds[i]):
                if attention_mask[i][j] == 1 and input_ids_cpu[i][j] not in [cls_token_id, pad_token_id]:
                    # 如果是 SEP，且位於最後一個 token，就跳過
                    if input_ids_cpu[i][j] == sep_token_id and j == (attention_mask[i].sum() - 1):
                        continue
                    valid_preds.append(id2label[p])
            all_preds.append(valid_preds)



output_df = pd.DataFrame({"ID": range(1, len(all_preds) + 1), "Label": all_preds})
output_df['Label'] = output_df['Label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

output_df.to_csv('/kaggle/working/output.csv',index=False, encoding="utf-8")

