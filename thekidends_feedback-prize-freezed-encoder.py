from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc

# ----------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import transformers
print('transformers version:',transformers.__version__)


config = {
    'model': 'bert-base-uncased',
    'dropout': 0.,
    'max_length': 512,
    'batch_size': 16,
    'epochs': 4,
    'freeze_lr': 4e-4,
    'unfreeze_lr': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'scheduler': 'CosineAnnealingWarmRestarts'
}


tokenizer = AutoTokenizer.from_pretrained(config['model'])


df = pd.read_csv('/kaggle/input/feedback-prize-english-language-learning-vac/public_ground_truth.csv')
test_df = pd.read_csv('/kaggle/input/feedback-prize-english-language-learning-vac/test.csv')


test_df.head()


class EssayDataset:
    def __init__(self, df, config, tokenizer=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.classes = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
        self.max_len = config['max_length']
        self.tokenizer = tokenizer
        self.is_test = is_test
        
    def __getitem__(self,idx):
        sample = self.df['full_text'][idx]
        tokenized = tokenizer.encode_plus(sample,
                                          None,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          truncation=True,
                                          padding='max_length'
                                         )
        inputs = {
            "input_ids": torch.tensor(tokenized['input_ids'], dtype=torch.long),
            "token_type_ids": torch.tensor(tokenized['token_type_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        }
        
        if self.is_test == True:
            return inputs
        
        label = self.df.loc[idx,self.classes].to_list()
        targets = {
            "labels": torch.tensor(label, dtype=torch.float32),
        }
        
        return inputs, targets
    
    def __len__(self):
        return len(self.df)



train_df, val_df = train_test_split(df,test_size=0.2,random_state=1357,shuffle=True)
print('dataframe shapes:',train_df.shape, val_df.shape)


train_ds = EssayDataset(train_df, config, tokenizer=tokenizer)
val_ds = EssayDataset(val_df, config, tokenizer=tokenizer)
test_ds = EssayDataset(test_df, config, tokenizer=tokenizer, is_test=True)


train_ds[0][0]['input_ids'].shape


train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_ds,
     batch_size=config['batch_size'],
     shuffle=True,
     num_workers=2,
     pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


print('loader shapes:',len(train_loader), len(val_loader), len(test_loader))


class FrozenEssayModel(nn.Module):
    def __init__(self,config,num_classes=6):
        super(FrozenEssayModel,self).__init__()
        self.model_name = config['model']
        self.encoder = AutoModel.from_pretrained(self.model_name)
        
        # this is how you freeze a model: the base_model is generic term for the transformer name
        for param in self.encoder.base_model.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(self.encoder.config.hidden_size,64)
        self.fc2 = nn.Linear(64,num_classes)
        
    def forward(self,inputs):
        _,outputs = self.encoder(**inputs, return_dict=False)
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        return outputs


class Trainer:
    def __init__(self, model, loaders, config,lr='unfreeze'):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.input_keys = ['input_ids','token_type_ids','attention_mask']
        
        self.loss_fn = nn.SmoothL1Loss()
        
        if lr == 'unfreeze':
            self.lr = self.config['unfreeze_lr']
        else:
            self.lr = self.config['freeze_lr']
            
        self.optim = self._get_optim()
        
        
        self.scheduler_options = {
            'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5,eta_min=1e-7),
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', min_lr=1e-7),
            'StepLR': torch.optim.lr_scheduler.StepLR(self.optim,step_size=2)
        }
        
        self.scheduler = self.scheduler_options[self.config['scheduler']]
        
        self.train_losses = []
        self.val_losses = []
        self.val_mcrmse = []
        
    def _get_optim(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        return optimizer

        
    def mcrmse(self, outputs, targets):
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss
    
    def train_one_epoch(self,epoch):
        
        running_loss = 0.
        progress = tqdm(self.train_loader, total=len(self.train_loader))
        
        for i,(inputs,targets) in enumerate(progress):
            
            self.optim.zero_grad()
            
            inputs = {k:inputs[k].to(device=config['device']) for k in inputs.keys()}
            targets = targets['labels'].to(device=config['device'])
            
            outputs = self.model(inputs)
            
            loss = self.loss_fn(outputs, targets)
            running_loss += loss.item()
            
            loss.backward()
            self.optim.step()
            
            if self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
                self.scheduler.step(epoch-1+i/len(self.train_loader)) # as per pytorch docs
            
            del inputs, targets, outputs, loss
            
        if self.config['scheduler'] == 'StepLR':
            self.scheduler.step()
            
        train_loss = running_loss/len(self.train_loader)
        self.train_losses.append(train_loss)
        
    @torch.no_grad()
    def valid_one_epoch(self,epoch):
        
        running_loss = 0.
        running_mcrmse = 0.
        progress = tqdm(self.val_loader, total=len(self.val_loader))
        
        for (inputs, targets) in progress:
            
            inputs = {k:inputs[k].to(device=config['device']) for k in inputs.keys()}
            targets = targets['labels'].to(device=config['device'])
            
            outputs = self.model(inputs)
            
            loss = self.loss_fn(outputs, targets)
            running_loss += loss.item()
            
            running_mcrmse += self.mcrmse(outputs, targets).item()
            
            del inputs, targets, outputs, loss
            
        
        val_loss = running_loss/len(self.val_loader)
        self.val_losses.append(val_loss)
        
        self.val_mcrmse.append(running_mcrmse/len(self.val_loader))
        del running_mcrmse
        
        if config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler.step(val_loss)
            
    
    def test(self, test_loader):
        
        preds = []
        for (inputs) in test_loader:
            inputs = {k:inputs[k].to(device=config['device']) for k in inputs.keys()}
            
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())
            
        preds = torch.concat(preds)
        return preds
    
    def fit(self):
        
        fit_progress = tqdm(
            range(1, self.config['epochs']+1),
            leave = True,
            desc="Training..."
        )
        
        for epoch in fit_progress:
            
            self.model.train()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self.train_one_epoch(epoch)
            self.clear()
            
            self.model.eval()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self.valid_one_epoch(epoch)
            self.clear()

            print(f"{'-'*30} EPOCH {epoch} / {self.config['epochs']} {'-'*30}")
            print(f"train loss: {self.train_losses[-1]}")
            print(f"valid loss: {self.val_losses[-1]}\n\n")
            
    
    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()


model = FrozenEssayModel(config).to(device=config['device'])
trainer_freeze = Trainer(model, (train_loader, val_loader), config, lr='freeze')


trainer_freeze.fit()


losses_df = pd.DataFrame({'epoch':list(range(1,config['epochs'] + 1)),
                          'train_loss':trainer_freeze.train_losses, 
                          'val_loss': trainer_freeze.val_losses,
                          'val_mcrmse': trainer_freeze.val_mcrmse
                         })
losses_df


plt.plot(trainer_freeze.train_losses, color='red')
plt.plot(trainer_freeze.val_losses, color='orange')
plt.title('MCRMSE Loss: Freezed Encoder')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


private_ground_truth = pd.read_csv('/kaggle/input/feedback-prize-english-language-learning-vac/private_ground_truth.csv')
private_ground_truth.head()


model.eval()
test_preds = trainer_freeze.test(test_loader)


preds_df = pd.DataFrame(test_preds.numpy(), columns=['cohesion','syntax','vocabulary','phraseology','grammar','conventions'])
preds_df['text_id'] = test_df['text_id'].values
cols = ['text_id'] + [col for col in preds_df.columns if col != 'text_id']
preds_df = preds_df[cols]
preds_df


merged_df = private_ground_truth.merge(preds_df, on='text_id', suffixes=('_true', '_pred'))
merged_df.head()


target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

true_vals = merged_df[[f"{col}_true" for col in target_cols]].values
pred_vals = merged_df[[f"{col}_pred" for col in target_cols]].values

def mcrmse(y_true, y_pred):
    colwise_mse = np.mean((y_true - y_pred) ** 2, axis=0)
    return np.mean(np.sqrt(colwise_mse))

score = mcrmse(true_vals, pred_vals)

print(f"MCRMSE Score: {score}")

