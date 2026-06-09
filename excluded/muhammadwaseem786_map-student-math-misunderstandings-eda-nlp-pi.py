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


# Install Required Dependencies and Library
!pip install transformers accelerate datasets scikit-learn torch pytorch-lightning



import pandas as pd

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")


import matplotlib.pyplot as plt

train['Category'].value_counts().plot(kind='bar', title='Category Distribution')
plt.show()

train['Misconception'].value_counts().head(10).plot(kind='barh', title='Top Misconceptions')
plt.show()



def create_text_input(df):
    df['input'] = "Question: " + df['QuestionText'] + \
                  " Answer: " + df['MC_Answer'] + \
                  " Explanation: " + df['StudentExplanation']
    return df

train = create_text_input(train)
test = create_text_input(test)



from sklearn.preprocessing import LabelEncoder

category_enc = LabelEncoder()
train['cat_label'] = category_enc.fit_transform(train['Category'])

misconception_enc = LabelEncoder()
train['mis_label'] = misconception_enc.fit_transform(train['Misconception'])



from torch.utils.data import Dataset

class MAPDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.input = df['input'].tolist()
        self.cat_labels = df['cat_label'].tolist()
        self.mis_labels = df['mis_label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        encoded = self.tokenizer(self.input[index],
                                 truncation=True,
                                 padding='max_length',
                                 max_length=self.max_len,
                                 return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'cat_label': torch.tensor(self.cat_labels[index]),
            'mis_label': torch.tensor(self.mis_labels[index])
        }



import torch.nn as nn
from transformers import AutoModel

class DualHeadModel(nn.Module):
    def __init__(self, base_model, num_cat, num_mis):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.3)
        self.cat_head = nn.Linear(self.encoder.config.hidden_size, num_cat)
        self.mis_head = nn.Linear(self.encoder.config.hidden_size, num_mis)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return self.cat_head(pooled), self.mis_head(pooled)



import pytorch_lightning as pl
import torch.nn.functional as F

class MAPLitModel(pl.LightningModule):
    def __init__(self, base_model, num_cat, num_mis, lr=2e-5):
        super().__init__()
        self.model = DualHeadModel(base_model, num_cat, num_mis)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        cat_out, mis_out = self.forward(batch['input_ids'], batch['attention_mask'])
        cat_loss = F.cross_entropy(cat_out, batch['cat_label'])
        mis_loss = F.cross_entropy(mis_out, batch['mis_label'])
        loss = cat_loss + mis_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        cat_out, mis_out = self.forward(batch['input_ids'], batch['attention_mask'])
        cat_loss = F.cross_entropy(cat_out, batch['cat_label'])
        mis_loss = F.cross_entropy(mis_out, batch['mis_label'])
        val_loss = cat_loss + mis_loss
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


