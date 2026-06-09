!pip install lightning





import torch
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
import numpy as np
import pandas as pd
import torch.nn as nn
import lightning as L
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import pandas as pd
test_df = pd.read_csv('/kaggle/input/testiki/ml_ozon_ounterfeit_new_test.csv')



# test_df = pd.concat([test_df,new_test],axis=0)
# train_dop = pd.DataFrame({'brand_name':brand,'description':description,'name_rus':name_rus,'CommercialTypeName4':commercial})
test_df[['brand_name', 'description', 'name_rus', 'CommercialTypeName4']] = test_df[['brand_name', 'description', 'name_rus', 'CommercialTypeName4']] .fillna('')


# texts = ('Бренд: ' + train_df['brand_name'].replace('', 'не указан') + '\n' +
#          'Название: ' + train_df['name_rus'].replace('', 'не указано') + '\n' +
#          # 'Цена: ' + (2**(train_df.PriceDiscounted/69.6606)).round().astype(int).astype(str) + ' рублей\n\n' +
#          'Описание: ' + train_df['description'].replace('', 'не указано')).tolist()

texts_test = ('Бренд: ' + test_df['brand_name'].replace('', 'не указан') + '\n' +
         'Название: ' + test_df['name_rus'].replace('', 'не указано') + '\n' +
         # 'Цена: ' + (2**(test_df.PriceDiscounted.fillna(test_df.PriceDiscounted.median())/69.6606)).round().astype(int).astype(str) + ' рублей\n\n' +
         'Описание: ' + test_df['description'].replace('', 'не указано')).tolist()
# folds = list(skf.split(texts, labels))
FOLD_IDX = 1
MODEL_NAME = "deepvk/USER2-base"
pizda = MODEL_NAME.replace('/', '$')
batch_size = 29


def pool(hidden_state, mask, pooling_method="mean"):
    if pooling_method == "mean":
        input_mask_expanded = mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    elif pooling_method == "cls":
        return hidden_state[:, 0]

# Custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=396):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = labels is not None
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        if self.has_labels:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

# Lightning Data Module
class TextClassificationDataModule(L.LightningDataModule):
    def __init__(self, train_texts, train_labels, val_texts, val_labels, test_texts, 
                 tokenizer, batch_size=32, max_length=396):
        super().__init__()
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.test_texts = test_texts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TextClassificationDataset(
                self.train_texts, self.train_labels, self.tokenizer, self.max_length
            )
            self.val_dataset = TextClassificationDataset(
                self.val_texts, self.val_labels, self.tokenizer, self.max_length
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = TextClassificationDataset(
                self.test_texts, None, self.tokenizer, self.max_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,  
            shuffle=False,
            num_workers=10,
            pin_memory=False
        )

class LitTextClassification(L.LightningModule):
    def __init__(self, model_name, learning_rates=(2e-5, 1e-3), weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.learning_rates = learning_rates
        self.weight_decay = weight_decay
        
        self.model = None
        self.classifier = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_outputs = []
        
    def setup(self, stage=None):
        # Load model and classifier here to avoid CUDA initialization issues
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 300),
                nn.ReLU(),
                nn.BatchNorm1d(300),
                nn.Dropout(0.1),
                nn.Linear(300, 1)
            )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = pool(outputs.last_hidden_state, attention_mask, pooling_method="cls")
        logits = self.classifier(embeddings)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'].unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'].unsqueeze(1))
        preds = torch.sigmoid(logits) > 0.5
        labels = batch['labels']
        
        # Store outputs for later use in on_validation_epoch_end
        output = {
            'val_loss': loss,
            'preds': preds,
            'labels': labels
        }
        self.validation_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):

        if not self.validation_outputs:
            return
            
        val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        
        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        
        accuracy = accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        
        self.log("val_f1_score", f1, sync_dist=True)
        
        # Clear the outputs for the next epoch
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': self.learning_rates[0]},
            {'params': self.classifier.parameters(), 'lr': self.learning_rates[1]}
        ], weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)

    


#вот так зайти в датасет с моделью  https://www.kaggle.com/datasets/alexcolsi/userbase2psevdoplus4fold
# или все модели лежат тут https://www.kaggle.com/datasets/alexcolsi/all-models-ozon-cup


model_to_predict = ['/kaggle/input/userbase2psevdoplus4fold/checkpoints2/deepvk$USER2-base~4~epochepoch=01~val_f1_scoreval_f1_score=0.81598.ckpt']


import os
from glob import glob
import shutil
os.makedirs('preds_new_test',exist_ok=True)

for zov in model_to_predict:
    device = torch.device('cuda:0')
    checkpoint = torch.load(zov, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LitTextClassification(
            model_name=MODEL_NAME,
            learning_rates=(2e-5, 1e-3),
            weight_decay=1e-5
        )
    model.setup()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device);

    preds_test = []
    for i in tqdm(range(0, len(texts_test), 14)):
        inputs = texts_test[i:i+14]
        #ys = labels[i:i+16].unsqueeze(1).float()
        tokenized_inputs = tokenizer(inputs, max_length=396, padding=True, truncation=True, return_tensors="pt")
        # del tokenized_inputs['token_type_ids']
        with torch.no_grad():
            logits = model(**tokenized_inputs.to(device))
        preds_test.extend(F.sigmoid(logits)[:, 0].tolist())

    
    torch.save(np.array(preds_test), f'new_test.pth')




class LitTextClassification(L.LightningModule):
    def __init__(self, model_name, learning_rates=(2e-5, 1e-3), weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.learning_rates = learning_rates
        self.weight_decay = weight_decay
        
        self.model = None
        self.classifier = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_outputs = []
        
    def setup(self, stage=None):
        # Load model and classifier here to avoid CUDA initialization issues
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(312, 240),
                nn.ReLU(),
                nn.BatchNorm1d(240),
                nn.Dropout(0.2),
                nn.Linear(240, 180),
                nn.ReLU(),
                nn.BatchNorm1d(180),
                nn.Dropout(0.1),
                nn.Linear(180, 1)
            )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = pool(outputs.last_hidden_state, attention_mask, pooling_method="cls")
        logits = self.classifier(embeddings)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'].unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'].unsqueeze(1))
        preds = torch.sigmoid(logits) > 0.5
        labels = batch['labels']
        
        # Store outputs for later use in on_validation_epoch_end
        output = {
            'val_loss': loss,
            'preds': preds,
            'labels': labels
        }
        self.validation_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):


        if not self.validation_outputs:
            return
            
        val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        
        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        
        accuracy = accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        
        self.log("val_f1_score", f1, sync_dist=True)
        
        # Clear the outputs for the next epoch
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': self.learning_rates[0]},
            {'params': self.classifier.parameters(), 'lr': self.learning_rates[1]}
        ], weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)



test_df = pd.read_csv('/kaggle/input/testiki/ml_ozon_ounterfeit_test (1).csv')
test_df[['brand_name', 'description', 'name_rus', 'CommercialTypeName4']] = test_df[['brand_name', 'description', 'name_rus', 'CommercialTypeName4']] .fillna('')


texts_test = ('Бренд: ' + test_df['brand_name'].replace('', 'не указан') + '\n' +
         'Название: ' + test_df['name_rus'].replace('', 'не указано') + '\n' +
         # 'Цена: ' + (2**(test_df.PriceDiscounted.fillna(test_df.PriceDiscounted.median())/69.6606)).round().astype(int).astype(str) + ' рублей\n\n' +
         'Описание: ' + test_df['description'].replace('', 'не указано')).tolist()
MODEL_NAME = "sergeyzh/rubert-tiny-turbo"
pizda = MODEL_NAME.replace('/', '$')


[i for i in glob('/kaggle/input/*') if 'rubert' in i]


[i for i in glob('/kaggle/input/*') if 'rubert' in i]
model_to_inference = []
for i in [i for i in glob('/kaggle/input/*') if 'rubert' in i]:
    for j in glob(i+'/checkpoints2/*'):
        if 'epoch=02' in j:
            model_to_inference.append(j)
model_to_inference     


import os
from glob import glob
import shutil
os.makedirs('preds_ruberta/',exist_ok=True)

for zov in model_to_inference:
    device = torch.device('cuda:0')
    checkpoint = torch.load(zov, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LitTextClassification(
            model_name=MODEL_NAME,
            learning_rates=(2e-5, 1e-3),
            weight_decay=1e-5
        )
    model.setup()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device);

    preds_test = []
    for i in tqdm(range(0, len(texts_test), 14)):
        inputs = texts_test[i:i+14]
        #ys = labels[i:i+16].unsqueeze(1).float()
        tokenized_inputs = tokenizer(inputs, max_length=396, padding=True, truncation=True, return_tensors="pt")
        del tokenized_inputs['token_type_ids']
        with torch.no_grad():
            logits = model(**tokenized_inputs.to(device))
        preds_test.extend(F.sigmoid(logits)[:, 0].tolist())

    
    torch.save(np.array(preds_test), f'preds_ruberta/{zov.split("/")[-1][:-5]}.pth')


import torch
answers_rubert = np.zeros(22760)
for i in glob('/kaggle/working/preds_ruberta/*'):
    answers_rubert += torch.load(i,weights_only=False)
answers_rubert /= 5


(answers_rubert < 0.025).mean()





test_df.shape


answers_rubert[(~(answers_rubert < 0.025))].shape


test_to_preds = test_df.iloc[(~(answers_rubert < 0.025))]
test_to_preds.shape


texts_test = ('Бренд: ' + test_to_preds['brand_name'].replace('', 'не указан') + '\n' +
         'Название: ' + test_to_preds['name_rus'].replace('', 'не указано') + '\n' +
         # 'Цена: ' + (2**(test_df.PriceDiscounted.fillna(test_df.PriceDiscounted.median())/69.6606)).round().astype(int).astype(str) + ' рублей\n\n' +
         'Описание: ' + test_to_preds['description'].replace('', 'не указано')).tolist()
MODEL_NAME = "sergeyzh/rubert-tiny-turbo"
pizda = MODEL_NAME.replace('/', '$')


len(texts_test)


class LitTextClassification(L.LightningModule):
    def __init__(self, model_name, learning_rates=(2e-5, 1e-3), weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.learning_rates = learning_rates
        self.weight_decay = weight_decay
        
        self.model = None
        self.classifier = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_outputs = []
        
    def setup(self, stage=None):
        # Load model and classifier here to avoid CUDA initialization issues
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, 300),
                nn.ReLU(),
                nn.BatchNorm1d(300),
                nn.Dropout(0.1),
                nn.Linear(300, 1)
            )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = pool(outputs.last_hidden_state, attention_mask, pooling_method="cls")
        logits = self.classifier(embeddings)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'].unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'].unsqueeze(1))
        preds = torch.sigmoid(logits) > 0.5
        labels = batch['labels']
        
        # Store outputs for later use in on_validation_epoch_end
        output = {
            'val_loss': loss,
            'preds': preds,
            'labels': labels
        }
        self.validation_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):

        if not self.validation_outputs:
            return
            
        val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        
        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        
        accuracy = accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        
        self.log("val_f1_score", f1, sync_dist=True)
        
        # Clear the outputs for the next epoch
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': self.learning_rates[0]},
            {'params': self.classifier.parameters(), 'lr': self.learning_rates[1]}
        ], weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)


# [i for i in glob('/kaggle/input/*') if 'user' in i]
model_to_inference = []
for i in [i for i in glob('/kaggle/input/*') if 'user' in i]:
    check_dir = '/checkpoints2/*'
    if glob(i+'/checkpoints2/*') == []:
        check_dir = '/checkpoints/*'
        
    for j in glob(i+check_dir):
        if 'epoch=02' in j:
            model_to_inference.append(j)
model_to_inference


MODEL_NAME = "deepvk/USER2-base"


import os
from glob import glob
import shutil
os.makedirs('preds_user/',exist_ok=True)

for zov in model_to_inference:
    device = torch.device('cuda:0')
    checkpoint = torch.load(zov, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LitTextClassification(
            model_name=MODEL_NAME,
            learning_rates=(2e-5, 1e-3),
            weight_decay=1e-5
        )
    model.setup()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device);

    preds_test = []
    for i in tqdm(range(0, len(texts_test), 14)):
        inputs = texts_test[i:i+14]
        #ys = labels[i:i+16].unsqueeze(1).float()
        tokenized_inputs = tokenizer(inputs, max_length=396, padding=True, truncation=True, return_tensors="pt")
        # del tokenized_inputs['token_type_ids']
        with torch.no_grad():
            logits = model(**tokenized_inputs.to(device))
        preds_test.extend(F.sigmoid(logits)[:, 0].tolist())

    
    torch.save(np.array(preds_test), f'preds_user/{zov.split("/")[-1][:-5]}.pth')


# [i for i in glob('/kaggle/input/*') if 'user' in i]
model_to_inference = []
for i in [i for i in glob('/kaggle/input/*') if 'bert' in i and not 'ru' in i]:
    check_dir = '/checkpoints/*'
    if 'vmeste' in i :
        check_dir = '/checkpoints2/*'
        
    for j in glob(i+check_dir):
        if 'epoch=02' in j:
            model_to_inference.append(j)
model_to_inference,len(model_to_inference)


MODEL_NAME = "sergeyzh/BERTA"


import os
from glob import glob
import shutil
os.makedirs('preds_berta/',exist_ok=True)

for zov in model_to_inference:
    device = torch.device('cuda:0')
    checkpoint = torch.load(zov, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LitTextClassification(
            model_name=MODEL_NAME,
            learning_rates=(2e-5, 1e-3),
            weight_decay=1e-5
        )
    model.setup()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device);

    preds_test = []
    for i in tqdm(range(0, len(texts_test), 14)):
        inputs = texts_test[i:i+14]
        #ys = labels[i:i+16].unsqueeze(1).float()
        tokenized_inputs = tokenizer(inputs, max_length=396, padding=True, truncation=True, return_tensors="pt")
        del tokenized_inputs['token_type_ids']
        with torch.no_grad():
            logits = model(**tokenized_inputs.to(device))
        preds_test.extend(F.sigmoid(logits)[:, 0].tolist())

    
    torch.save(np.array(preds_test), f'preds_berta/{zov.split("/")[-1][:-5]}.pth')


other_preds = np.zeros(test_to_preds.shape[0])

berta_coef = 1/3
all_coef = 0
for i in glob('preds_berta/*'):
    all_coef += berta_coef
    other_preds += torch.load(i,weights_only=False) * berta_coef    
    
user_coef = 1
for i in glob('preds_user/*'):
    all_coef += user_coef
    other_preds += torch.load(i,weights_only=False) * user_coef  

other_preds += answers_rubert[(~(answers_rubert < 0.025))] *0.5
all_coef += 0.5 * 5
other_preds/=all_coef


ss = np.zeros(22760)
ss[(~(answers_rubert < 0.025))] = other_preds


tt = pd.read_csv('/kaggle/input/testiki/ml_ozon_ounterfeit_test (1).csv')
tt['preds'] = ss.tolist()
tt['prediction'] = (tt['preds']>0.6)


new_test = pd.read_csv('/kaggle/input/testiki/ml_ozon_ounterfeit_new_test.csv')
new_test['prediction'] = (torch.load('new_test.pth',weights_only=False) > 0.6).astype(int)


new_test['prediction'].mean()


tt.to_csv('old_test.csv')
new_test.to_csv('new_test.csv')




