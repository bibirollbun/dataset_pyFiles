class Config:
    n_fold = 5
    batch_size = 8
    train_path = '/kaggle/input/commonlitreadabilityprize/train.csv'
    test_path = '/kaggle/input/commonlitreadabilityprize/test.csv'
    bins = 64
    pad = 16
    epochs = 20
    fc_epochs = 0
    warmup_epochs = 0
    lr = 5e-5
    early_stopping_round = 5
    seed = 42

cfg = Config()


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


train_df


train_df[['id','excerpt','target','standard_error']]


train_df = train_df[['id','excerpt','target','standard_error']]
train_df = train_df.dropna()


print(train_df['target'].describe())
plt.hist(train_df['target'], bins=100)
plt.show()


train_df['target'] -= train_df['target'].min()
train_df['target'] /= train_df['target'].max()


print(train_df['target'].describe())
plt.hist(train_df['target'], bins=100)
plt.show()


print(train_df['standard_error'].describe())
plt.hist(train_df['standard_error'], bins=100)
plt.show()


train_df['standard_error'][train_df['standard_error']==0] = 0.4#外れ値を取り除く
"""print(train_df['standard_error'].min())
train_df['standard_error'] -= train_df['standard_error'].min()
train_df['standard_error'] /= train_df['standard_error'].max()"""


print(train_df['standard_error'].describe())
plt.hist(train_df['standard_error'], bins=100)
plt.show()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


bin_edges = torch.linspace(-cfg.pad/cfg.bins, 1+cfg.pad/cfg.bins, cfg.bins + cfg.pad*2 + 1)  # binの端
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2    # binの中心

def create_target_distribution(x_true, std):
    dist = torch.exp(-0.5 * ((bin_centers - x_true) / std) ** 2)
    dist = dist / dist.sum()  # 正規化
    return dist


class LitDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len, data_aug=True):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.input_ids = []
        self.attention_mask = []

        self.texts = []
        self.targets1 = []
        self.targets2 = []
        self.targets = []
        if data_aug:
            for text_num, text in enumerate(texts): #data aug
                parts = text.split('\n')
                for i in range(len(parts)):
                    self.texts.append('\n'.join(parts[:i+1]))
                    self.targets1.append(create_target_distribution(targets[text_num][0],targets[text_num][1]/5))
                    self.targets2.append(targets[text_num][1])
                    self.targets.append(targets[text_num])
            print(f'data_aug result: {len(targets)} -> {len(self.targets1)}')
        else:
            for text_num, text in enumerate(texts): #data aug
                self.texts.append(text)
                self.targets1.append(create_target_distribution(targets[text_num][0],targets[text_num][1]/5))
                self.targets2.append(targets[text_num][1])
                self.targets.append(targets[text_num])
        for text in self.texts:#あらかじめtokenizeしておく
            encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_mask.append(encoding['attention_mask'].squeeze(0))
        self.input_ids = torch.stack(self.input_ids)
        self.attention_mask = torch.stack(self.attention_mask)
        self.targets1 = torch.stack(self.targets1)
        self.targets2 = torch.tensor(self.targets2, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets1)

    def __getitems__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'target1': self.targets1[idx],
            'target2': self.targets2[idx],
            'target': self.targets[idx]
        }

def collate(x):
    return x



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
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, cfg.bins+2*cfg.pad+2)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.SiLU()

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_out = torch.mean(out.last_hidden_state, dim=1)
        out = self.fc1(pooled_out)
        out = self.act(out)
        out = self.fc2(out)
        #out[:,-2:] = self.sigmoid(out[:,-2:])
        return out
    



from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_lr = [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            # Cosine Annealing phase
            cos_anneal_lr = [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                              (self.total_epochs - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]
            return cos_anneal_lr

import torch
import numpy as np
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


train_df = train_df.sample(frac=1, random_state=1123).reset_index(drop=True)
fold_len = int(len(train_df)/cfg.n_fold)


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for fold in range(cfg.n_fold):
    seed_everything(cfg.seed)

    if fold == cfg.n_fold-1:
        train_data = train_df[:fold_len*fold]
    else:
        train_data = pd.concat([train_df[:fold_len*fold],train_df[fold_len*(fold+1):]])
    valid_data = train_df[fold_len*fold:fold_len*(fold+1)].reset_index(drop=True)
    
    train_dataset = LitDataset(train_data['excerpt'], train_data[['target','standard_error']].values, tokenizer, max_len=320, data_aug=False)
    valid_dataset = LitDataset(valid_data['excerpt'], valid_data[['target','standard_error']].values, tokenizer, max_len=320, data_aug=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate)
    
    #model = BertClassifier()
    model = AutoBertModel(model_name="FacebookAI/roberta-base")
    if torch.cuda.is_available():
        device = "cuda"
        model = nn.DataParallel(model)
    else:
        device = "cpu"
        model = nn.DataParallel(model)
    
    epochs = cfg.epochs
    fc_epochs = cfg.fc_epochs
    warmup_epochs = cfg.warmup_epochs
    
    fc_optimizer = optim.AdamW(
        list(model.module.fc1.parameters()) + list(model.module.fc2.parameters()),
        lr=cfg.lr
    )
    full_optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    loss_weight = torch.tensor([1,0,1,0],device=device,dtype=torch.float32)
    scheduler = WarmupCosineAnnealingLR(full_optimizer, warmup_epochs*len(train_dataloader), (epochs-fc_epochs)*len(train_dataloader)+1)
    scaler = GradScaler()

    best_score = 1e6
    last_progress = 0
    
    for epoch in range(epochs):
        if last_progress >= cfg.early_stopping_round:
            print('Early Stopping Occured')
            break
        last_progress += 1
        if fc_epochs > epoch:
            model.module.bert.requires_grad = False
            optimizer = fc_optimizer
        elif fc_epochs <= epoch:
            model.module.bert.requires_grad = True
            optimizer = full_optimizer
        model.train()
        model.to(device)
        total_loss = 0
        train_progress = tqdm(train_dataloader, desc=f"[Fold: {fold+1} / {cfg.n_fold},  Epoch: {epoch + 1} / {epochs}] Started")
    
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target1 = batch['target1'].to(device)
            target2 = batch['target2'].to(device)
            target = batch['target'].to(device)
    
            optimizer.zero_grad()
            with autocast(device):
                outputs = model(input_ids, attention_mask)
                loss1 = criterion1(outputs[:,:-2], target1)
                loss2 = criterion2(outputs[:,-2], target2)
                loss3 = criterion2(outputs[:,-1], target[:,0])
                loss4 = criterion2((F.softmax(outputs[:,:-2],dim=1) * bin_centers.to(device)).sum(dim=1), target[:,0])
                loss = loss1 * loss_weight[0] + loss2 * loss_weight[1] + loss3 * loss_weight[2] + loss4 * loss_weight[3]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
    
            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item(),lr=scheduler.get_lr()[0])
            if epoch >= fc_epochs:
                scheduler.step()
    
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}")
    
        model.eval()
        valid_score = 0
        valid_score1 = 0
        valid_score2 = 0
        valid_loss = 0
        eval_progress = tqdm(valid_dataloader, desc=f"[Fold: {fold+1} / {cfg.n_fold},  Epoch: {epoch + 1} / {epochs}] Evaluating")
        with torch.no_grad():
            for batch in eval_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target1 = batch['target1'].to(device)
                target2 = batch['target2'].to(device)
                target = batch['target'].to(device)
    
                with autocast(device):
                    outputs = model(input_ids, attention_mask)
                    loss1 = criterion1(outputs[:,:-2], target1)
                    loss2 = criterion2(outputs[:,-2], target2)
                    loss3 = criterion2(outputs[:,-1], target[:,0])
                    loss4 = criterion2((F.softmax(outputs[:,:-2],dim=1) * bin_centers.to(device)).sum(dim=1), target[:,0])
                    loss = loss1 * loss_weight[0] + loss2 * loss_weight[1] + loss3 * loss_weight[2] + loss4 * loss_weight[3]
    
                outputs_ = F.softmax(outputs[:,:-2],dim=1)
                pred = (bin_centers.to(device)[torch.argmax(outputs_,dim=1)])#.sum(dim=1)
                valid_score1 += criterion2(pred, batch['target'][:,0].to(device))
                valid_score2 += criterion2(outputs[:,-1], batch['target'][:,0].to(device))
                valid_score += criterion2((outputs[:,-1]+pred)/2, batch['target'][:,0].to(device))
                valid_loss += loss.item()

        if loss_weight[0] == 0:
            valid_score = valid_score2
        elif loss_weight[2] == 0:
            valid_score = valid_score1

        if valid_score < best_score:
            last_progress = 0
            torch.save(model.module.state_dict(), f'bert_finetune_fold={fold}.pth')
            best_score = valid_score
        print(f"Epoch {epoch + 1}, ValidLoss: {valid_loss / len(valid_dataloader)}, ValidScore1: {math.sqrt(valid_score1 / len(valid_dataloader))*5.387658:.4f}, ValidScore2: {math.sqrt(valid_score2 / len(valid_dataloader))*5.387658:.4f}, ValidScore: {math.sqrt(valid_score / len(valid_dataloader))*5.387658:.4f}")
    model = None
    optimizer = None
    full_optimizer = None
    fc_optimizer = None
    scheduler = None
    train_dataset = None
    valid_dataset = None
    train_loader = None
    valid_loader = None
    scaler = None





