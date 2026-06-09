class Config:
    n_fold = 5
    batch_size = 32
    train_path = '/kaggle/input/commonlitreadabilityprize/train.csv'
    test_path = '/kaggle/input/commonlitreadabilityprize/test.csv'
    bins = 64
    pad = 16

cfg = Config


import pandas as pd
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import math
import spacy


train_df = pd.read_csv(cfg.train_path)
test_df = pd.read_csv(cfg.test_path)


class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }



class BertClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, cfg.bins+2*cfg.pad+2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        out = self.dropout(out)
        out = self.classifier(out)
        out[:,-2:] = self.sigmoid(out[:,-2:])
        return out

from transformers import AutoTokenizer, AutoModel

class AutoBertModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", output_dim=cfg.bins+2*cfg.pad+2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.bert.config.hidden_size, cfg.bins+2*cfg.pad+2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_out = torch.mean(out.last_hidden_state, dim=1)
        out = self.classifier(pooled_out)
        out[:,-2:] = self.sigmoid(out[:,-2:])
        return out


outputs = []


models = {'/kaggle/input/commonlit-bert/roberta-base1':'/kaggle/input/d/lime1123/transformers-bert/roberta-base/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b',
          '/kaggle/input/commonlit-bert/roberta-base':'/kaggle/input/d/lime1123/transformers-bert/roberta-base/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b',
          '/kaggle/input/commonlit-bert/deberta-ce-mse-lb=0.515':'/kaggle/input/d/lime1123/transformers-bert/deberta-base/models--microsoft--deberta-base/snapshots/0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195',
          '/kaggle/input/commonlit-bert/deberta-base':'/kaggle/input/d/lime1123/transformers-bert/deberta-base/models--microsoft--deberta-base/snapshots/0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195',
          }


for model_path, model_name in models.items():
    for fold in range(cfg.n_fold):
        outputs.append([])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        test_dataset = TestDataset(test_df['excerpt'], tokenizer, max_len=320)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        model = AutoBertModel(model_name=model_name)
        model.load_state_dict(torch.load(f"{model_path}/bert_finetune_fold{fold}.pth"))
        
        if torch.cuda.is_available():
            device = "cuda"
            model = nn.DataParallel(model)
        else:
            device = "cpu"
        
        model.eval()
        model.to(device)
        bin_edges = torch.linspace(-cfg.pad/cfg.bins, 1+cfg.pad/cfg.bins, cfg.bins + cfg.pad*2 + 1)  # binの端
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2    # binの中心
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            
                with autocast(device):
                    output = model(input_ids, attention_mask)
                    output[:,:-2] = F.softmax(output[:,:-2],dim=1)
                    output = output[:,-1]# + (output[:,:-2] * bin_centers.to(device)).sum(dim=1))/2
                outputs[-1] += output.tolist()


outputs = torch.tensor(outputs)
print(outputs.shape)
outputs = outputs.mean(dim=0)
outputs *= train_df['target'].max() - train_df['target'].min()
outputs += train_df['target'].min()


submission = pd.DataFrame({'id':test_df['id'],'target':outputs.squeeze().tolist()})


submission.to_csv('/kaggle/working/submission.csv',index=False)
submission


submission.describe()




