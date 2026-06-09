!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl;


import torch
import pandas as pd
import joblib
from transformers import PreTrainedModel, AutoConfig, BertModel, BertTokenizerFast, BertConfig, AutoModel, AutoTokenizer
from sklearn.metrics import mean_absolute_error
from torch import nn
from transformers.activations import ACT2FN
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
        """ self.dense = nn.Sequential(
            nn.Linear(pooler_size*2, pooler_size),
            nn.GELU("tanh"),
            nn.Dropout(0.2),
            nn.Linear(pooler_size, pooler_size*2),
        )"""
        self.dense=nn.Linear(pooler_size, pooler_size)
        
        dropout_prob = getattr(config, 'pooler_dropout', config.hidden_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.activation = getattr(config, 'pooler_hidden_act', config.hidden_act)
        self.config = config
        self.meaner=torch.nn.AdaptiveAvgPool1d(1)
    def forward(self, hidden_states,mask):
        
        """        context_token = torch.mean(hidden_states, dim=1, keepdim=True).squeeze(dim=1)
        
        avg_pool = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        pooled = torch.cat([context_token, avg_pool], dim=-1)"""
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.activation](pooled_output)
        return pooled_output

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        
        self.pooler = ContextPooler(config)

        pooler_output_dim = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.output = torch.nn.Linear(pooler_output_dim, 1) # Still predicting one label at a time. Kinda stupid

    def forward(
        self,
        input_ids,
        scaler,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        """ print(torch.mean(outputs.last_hidden_state, dim=1, keepdim=True).squeeze(dim=1))
        print(outputs.pooler_output)"""
        pooled_output = self.pooler(outputs.last_hidden_state,attention_mask)
        """print(pooled_output.shape)"""
        
        regression_output = self.output(pooled_output)

        loss = None
        true_loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()

            unscaled_labels = scaler.inverse_transform(labels.cpu().numpy())
            unscaled_outputs = scaler.inverse_transform(regression_output.cpu().detach().numpy())
            
            loss = loss_fn(regression_output, labels)
            true_loss = mean_absolute_error(unscaled_outputs, unscaled_labels)

        return {
            "loss": loss,
            "logits": regression_output,
            "true_loss": true_loss
        }


BATCH_SIZE = 8

def tokenize_smiles(seq,max_length=512):
    seq = tokenizer.cls_token + seq# If we pass a string, tokenizer will smartly think we want to create a sequence for each symbol
    tokenized = tokenizer(seq, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return tokenized

def load_model(path=None):
    config = AutoConfig.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
    model = CustomModel(config).cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model


def make_predictions(model, scaler, smiles_seq):
    aggregated_preds = []
    for smiles in smiles_seq:
        smiles = [smiles]
        smiles_tokenized = tokenize_smiles(smiles)

        input_ids = smiles_tokenized['input_ids'].cuda()
        attention_mask = smiles_tokenized['attention_mask'].cuda()
        with torch.no_grad():
            preds = model(input_ids=input_ids, scaler=scaler, attention_mask=attention_mask)['logits'].cpu().numpy()
        
        true_preds = scaler.inverse_transform(preds).flatten()
        aggregated_preds.append(true_preds.tolist())
    return np.array(aggregated_preds)


test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")

test_copy = test.copy()

smiles_test = test['SMILES'].to_list()
smiles_train=train["SMILES"].to_list()

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


scalers = joblib.load('/kaggle/input/smiles-bert-models/target_scalers.pkl')
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')


import pandas as pd
import numpy as np
from rdkit import Chem
import random
from typing import Optional, List, Union

def augment_smiles_dataset(df: pd.DataFrame,
                               smiles_column: str = 'SMILES',
                               augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
                               n_augmentations: int = 10,
                               preserve_original: bool = True,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_augmentation_strategy(smiles: str, strategy: str) -> List[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            
            if strategy == 'enumeration':
                # Standard SMILES enumeration
                for _ in range(n_augmentations):
                    enum_smiles = Chem.MolToSmiles(mol, 
                                                 canonical=False, 
                                                 doRandom=True,
                                                 isomericSmiles=True)
                    augmented.append(enum_smiles)
            
            elif strategy == 'kekulize':
                # Kekulization variants
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
            
            elif strategy == 'stereo_enum':
                # Stereochemistry enumeration
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)
            
            return list(set(augmented))  # Remove duplicates
            
        except Exception as e:
            print(f"Error in {strategy} for {smiles}: {e}")
            return [smiles]
    
    augmented_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            augmented_rows.append(original_row)
        
        for strategy in augmentation_strategies:
            strategy_smiles = apply_augmentation_strategy(original_smiles, strategy)
            
            for aug_smiles in strategy_smiles:
                if aug_smiles != original_smiles:
                    new_row = row.to_dict().copy()
                    new_row[smiles_column] = aug_smiles
                    new_row['augmentation_strategy'] = strategy
                    new_row['is_augmented'] = True
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

test = augment_smiles_dataset(test)
train= augment_smiles_dataset(train)


train


from torch.utils.data import Dataset

def tokenize_smiles(smiles, max_length=512):
    smiles = tokenizer.cls_token + smiles
    tokenized = tokenizer(
        smiles,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return tokenized

class MeinDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=512):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        tokenized = tokenize_smiles(self.smiles_list[idx], self.max_length)
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return (input_ids, attention_mask), label


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


class TestDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=512):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        tokenized = tokenize_smiles(self.smiles_list[idx], self.max_length)
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scalers = []

for target in targets:
    actual_targets = train[target]
    target_scaler = StandardScaler()
    train[target] = target_scaler.fit_transform(train[target].to_numpy().reshape(-1, 1))
    
    scalers.append(target_scaler)
labels = train[targets].values
smiles_train, smiles_test, labels_train, labels_test = train_test_split(
    train['SMILES'], labels, test_size=0.1, random_state=42)


joblib.dump(scalers, 'target_scalers.pkl')


scalers[0]


train


smiles_train


test["SMILES"].to_numpy().reshape(-1,1).shape


from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import numpy as np


class Trainer(nn.Module):
    def __init__(self,tokenizer):
        super().__init__()
        self.tokenizer=tokenizer
    
    def set_loader(self,train,labels,is_train=True):
        self.dropnan(train,labels)
        dataset=MeinDataset(self.smiles,self.targets,self.tokenizer)
        
        dataloader = DataLoader(dataset,batch_size=8,shuffle=is_train,num_workers=4,pin_memory=True)
        return dataloader

    def test_loader(self,test):
        self.test=test["smiles"].reshape(-1,1).to_list()
        dataset=TestDataset(self.test,self.tokenizer)
        dataloader = DataLoader(dataset,batch_size=1,num_workers=4,pin_memory=True)
        return dataloader

    def test_one_feature(self,model,test_loader,scaler)

        model.to(device)
        model.eval()
        test_progress = tqdm(test_dataloader, desc="Testing", leave=False)
        for batch in enumerate(test_progress):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(
                    input_ids=input_ids,
                    scaler=scaler,
                    attention_mask=attention_mask,
                    labels=labels,
                )
    def train_one_label(self,model,train_dataloader,val_dataloader,scaler, num_epochs=10, learning_rate=2e-5, device='cuda'):
        model.to(device)

        
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
        train_losses = []
        val_losses = []
        
        model.train()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            total_train_loss = 0
            total_true_train_loss = 0
            train_progress = tqdm(train_dataloader, desc="Training", leave=False)
            
            for batch_idx, batch in enumerate(train_progress):
                input_ids = batch[0][0].to(device)
                attention_mask = batch[0][1].to(device)
                labels = batch[1].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=input_ids,
                    scaler=scaler,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs['loss']
                true_loss = outputs['true_loss']
                
                
                total_train_loss += loss.item()
                total_true_train_loss += true_loss
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'true_loss': f'{true_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_true_train_loss = total_true_train_loss / len(train_dataloader)
            
            train_losses.append(avg_train_loss)
            
            model.eval()
            total_val_loss = 0
            total_true_val_loss = 0
    
            with torch.no_grad():
                val_progress = tqdm(val_dataloader, desc="Validation", leave=False)
                
                for batch in val_progress:
                    input_ids = batch[0][0].to(device)
                    attention_mask = batch[0][1].to(device)
                    labels = batch[1].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        scaler=scaler,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs['loss']
                    true_loss = outputs['true_loss']
    
                    total_val_loss += loss.item()
                    total_true_val_loss += true_loss
                    
                    val_progress.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_true_loss = total_true_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | True train loss: {avg_true_train_loss:.4f} | True val loss: {avg_val_true_loss:.4f}")
            
            model.train()
        
        return train_losses, val_losses
    def dropnan(self,smiles, targets):
        non_nan_mask = ~np.isnan(targets)
            
        self.targets = targets[non_nan_mask].reshape(-1, 1)
        self.smiles = smiles.copy()[non_nan_mask].reset_index(drop=True)

    def pipeline(self):
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        for i,lbl in enumerate(targets):
            scaler = scalers[i]
            labels_train_actual = labels_train[:, i]
            print(smiles_train.shape, 'act')
            labels_test_actual = labels_test[:, i]
            

           
            train_dataloader=self.set_loader(smiles_train, labels_train_actual,True)
            valid_dataloader=self.set_loader(smiles_test, labels_test_actual,False)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model=load_model(f'/kaggle/input/smiles-bert-models/trained_smiles_model_{lbl}_target.pth')
            
            train_losses, val_losses = self.train_one_label(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=valid_dataloader,
                scaler=scaler,
                num_epochs=10,
                learning_rate=1e-4,
                device=device
            )
            
            print('Overall loss: ', train_losses)
            torch.save(model.state_dict(), f'trained_smiles_model_{lbl}_target.pth')
            print("Model saved successfully!")
        
 


trainer=Trainer(tokenizer)


trainer.pipeline()


train.isna().sum()


def tokenize_smiles(smiles_list, max_length=512):
    smiles_with_cls = [tokenizer.cls_token + s for s in smiles_list]
    tokenized = tokenizer(
        smiles_with_cls,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return tokenized
preds_mapping = {}

for i in tqdm(range(len(targets))):
    target = targets[i]
    scaler = scalers[i]

    model_path = f'/kaggle/working/trained_smiles_model_{target}_target.pth' # Very sophisticated staff
    model = load_model(model_path)
    true_preds = []

    for i, data in test.groupby('id'):
        test_smiles = data['SMILES'].to_list()
        augmented_preds = make_predictions(model, scaler, test_smiles)
    
        average_pred = np.median(augmented_preds)
    
        true_preds.append(float(average_pred.flatten()[0]))

    preds_mapping[target] = true_preds


"""preds_mapping = {}

for i in tqdm(range(len(targets))):
    target = targets[i]
    scaler = scalers[i]

    model_path = f'/kaggle/working/trained_smiles_model_{target}_target.pth' # Very sophisticated staff
    model = load_model(model_path)
    true_preds = []
    for row in range(len(test)):
    
        test_smiles = test.iloc[row]['SMILES']
        print(test_smiles)
        pred = make_predictions(model, scaler, test_smiles)
        print(pred.shape)
    
    
        true_preds.append(float(average_pred.flatten()))

    preds_mapping[target] = true_preds"""


test


submission = pd.DataFrame(preds_mapping)
submission['id'] = test_copy['id']
submission.to_csv('submission.csv', index=False)







