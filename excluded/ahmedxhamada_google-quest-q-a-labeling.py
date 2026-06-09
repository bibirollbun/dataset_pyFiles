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


pip install torch transformers pandas scikit-learn tqdm


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm


df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')


df.head()


TARGET_COLUMNS = [
    'question_asker_intent_understanding',
    'question_body_critical',
    'question_conversational',
    'question_expect_short_answer',
    'question_fact_seeking',
    'question_has_commonly_accepted_answer',
    'question_interestingness_others',
    'question_interestingness_self',
    'question_multi_intent',
    'question_not_really_a_question',
    'question_opinion_seeking',
    'question_type_choice',
    'question_type_compare',
    'question_type_consequence',
    'question_type_definition',
    'question_type_entity',
    'question_type_instructions',
    'question_type_procedure',
    'question_type_reason_explanation',
    'question_type_spelling',
    'question_well_written',
    'answer_helpful',
    'answer_level_of_information',
    'answer_plausible',
    'answer_relevance',
    'answer_satisfaction',
    'answer_type_instructions',
    'answer_type_procedure',
    'answer_type_reason_explanation',
    'answer_well_written'
]



TARGET_COLUMNS


class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['question_title'] + ' ' + row['question_body'] + ' ' + row['answer']
        inputs = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(row[TARGET_COLUMNS].values.astype(float), dtype=torch.float)
        return item


tokenizer = BertTokenizer.from_pretrained('/kaggle/input/quest-bert-base-tf2-0')
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = QuestDataset(train_df, tokenizer)
val_dataset = QuestDataset(val_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)



class QuestModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.out(pooled_output)
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits

model = QuestModel(num_labels=len(TARGET_COLUMNS)).cuda()



optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        token_type_ids = batch['token_type_ids'].cuda()
        labels = batch['labels'].cuda()
        loss, _ = model(input_ids, attention_mask, token_type_ids, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            labels = batch['labels'].cuda()
            loss, _ = model(input_ids, attention_mask, token_type_ids, labels)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss/len(val_loader)}")



torch.save(model.state_dict(), 'quest_model.bin')


test_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

class QuestTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['question_title']) + ' ' + str(row['question_body']) + ' ' + str(row['answer'])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        return item


test_dataset = QuestTestDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


model = QuestModel(num_labels=len(TARGET_COLUMNS))
model.load_state_dict(torch.load('/kaggle/working/quest_model.bin', map_location='cpu'))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')


all_preds = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
all_preds = np.concatenate(all_preds, axis=0)


submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
submission[TARGET_COLUMNS] = all_preds
submission.to_csv('submission.csv', index=False)
print('submission.csv created!')




