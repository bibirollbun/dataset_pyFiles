!pip install /kaggle/input/neurips-opp2025-download-libraries/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import os
import json
import types
import numbers
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


import torch
import numpy as np
import polars as pl
import pandas as pd
import torch.nn as nn
from rdkit import Chem
from metric import score 
from transformers import AutoModel
from transformers import AutoConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from transformers.utils.logging import disable_progress_bar
disable_progress_bar()


class CFG:
    
    ext1_data_path = Path('/kaggle/input/tc-smiles/Tc_SMILES.csv')
    ext2_data_path = Path('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv')
    ext3_data_path = Path('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
    ext4_data_path = Path('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
    
    train_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    
    checkpoint = 'DeepChem/ChemBERTa-77M-MTR'
    
    n_splits = 5
    batch_size = 32
    hidden_size = 384
    context = 512
    seed = 42
    
    learning_rate = 3e-5
    weight_decay = 5e-3
    patience = 8
    epochs = 32


def set_seed(seed):
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)


class FE:
    
    def __init__(
        self, 
        ext1_data_path, 
        ext2_data_path,
        ext3_data_path,
        ext4_data_path
    ):
        
        self._ext1_data_path = ext1_data_path
        self._ext2_data_path = ext2_data_path
        self._ext3_data_path = ext3_data_path
        self._ext4_data_path = ext4_data_path
        
        self._batch_size = 8192        
        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    def _load_data(self, path):
        
        return pl.read_csv(path, batch_size=self._batch_size).to_pandas()
    
    def _to_canonical(self, smiles):
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True)
            
        except:
            return np.nan
    
    def _load(self, path, columns_dict, excel=False):

        if not excel:
            ext_data = pd.read_csv(path)

        else:
            ext_data = pd.read_excel(path)
        
        ext_data = ext_data.rename(columns=columns_dict)
        ext_data['SMILES'] = ext_data['SMILES'].apply(self._to_canonical)
        
        return ext_data

    def _merge(self, data, ext_data, target):
        
        shared_smiles = set(data['SMILES']) & set(ext_data['SMILES'])
        unique_smiles = set(ext_data['SMILES']) - set(data['SMILES'])
        
        populated = set(data.loc[data[target].notna(), 'SMILES'])
        shared_smiles -= populated
        
        for s in shared_smiles:
            value = ext_data.loc[ext_data['SMILES'] == s, target].iat[0]
            data.loc[data['SMILES'] == s, target] = value
        
        new_rows = ext_data[ext_data['SMILES'].isin(unique_smiles)]
        data = pd.concat([data, new_rows], axis=0).reset_index(drop=True)
        
        return data
    
    def _merge_with_ext1_data(self, df):
        
        ext_data = self._load(self._ext1_data_path, {'TC_mean': 'Tc'})        
        ext_data = ext_data[['SMILES', 'Tc']].groupby('SMILES', as_index=False)['Tc'].mean()

        df = self._merge(df, ext_data, 'Tc')
        
        return df

    def _merge_with_ext2_data(self, df):
        
        ext_data = self._load(self._ext2_data_path, {'Tg (C)': 'Tg'})
        ext_data = ext_data[['SMILES', 'Tg']] .groupby('SMILES', as_index=False)['Tg'].mean()

        df = self._merge(df, ext_data, 'Tg')
        
        return df

    def _merge_with_ext3_data(self, df):
        
        ext_data = self._load(self._ext3_data_path, {'density(g/cm3)': 'Density'}, excel=True)
        ext_data = ext_data[['SMILES', 'Density']]
        
        mask = ext_data['SMILES'].notnull() & ext_data['Density'].notnull() & (ext_data['Density'] != 'nylon')
        
        ext_data = ext_data[mask].copy()        
        ext_data['Density'] -= 0.118
        ext_data = ext_data.groupby('SMILES', as_index=False)['Density'].mean()

        df = self._merge(df, ext_data, 'Density')
        
        return df

    def _merge_with_ext4_data(self, df):
        
        ext_data = self._load(self._ext4_data_path, {'Tg [K]': 'Tg'}, excel=True)
        ext_data = ext_data[['SMILES', 'Tg']]      
        
        ext_data['Tg'] -= 273.15
        ext_data = ext_data.groupby('SMILES', as_index=False)['Tg'].mean()
        
        df = self._merge(df, ext_data,'Tg')
        
        return df
        
    def _merge_data(self, df):  
        
        df = self._merge_with_ext1_data(df)
        df = self._merge_with_ext2_data(df)
        df = self._merge_with_ext3_data(df)
        df = self._merge_with_ext4_data(df)      

        return df
    
    def _cast_datatypes(self, df):
        
        for col in df.columns:
            
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            
            val = non_null.iloc[0]
            
            if isinstance(val, numbers.Integral):
                target = 'int32'
                
            elif isinstance(val, numbers.Real):
                target = 'float32'
                
            else:
                target = 'string'
                
            df[col] = df[col].astype(target)
        
        return df

    def _stats(self, df):
        
        unique_smiles = df['SMILES'].unique()
        print(f'Dataset contains {len(unique_smiles)} unique SMILES.\n') 
        
        for p in self._properties:                
            count = df[p].notna().sum()
            print(f'Target {p} has {count} non-missing values.')        
    
    def info(self, df):
        
        print(f'\nShape of dataframe: {df.shape}')
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
        
        if 'Density' in df.columns: # Or any other target
            self._stats(df)
    
    def apply_fe(self, path):
        
        df = self._load_data(path) 
        
        if df.shape[1] > 2:
            df = self._merge_data(df)
            
        df = self._cast_datatypes(df)        
        self.info(df)
        
        cat_cols = df.select_dtypes(include=['string']).columns.tolist()
        if cat_cols:
            df[cat_cols] = df[cat_cols].fillna('Missing').astype('category')
        
        return df, cat_cols


fe = FE(
    CFG.ext1_data_path, 
    CFG.ext2_data_path,
    CFG.ext3_data_path,
    CFG.ext4_data_path
)


train_data, cat_cols = fe.apply_fe(CFG.train_path)


class SmilesDataset(Dataset):
    
    def __init__(self, smiles, targets, mode, n_aug=10):
        self.mols = [Chem.MolFromSmiles(s) for s in smiles]
        self.targets = targets
        self.mode = mode
        self.n_aug = n_aug

    def __len__(self):
        
        if self.mode == 'train':
            return len(self.mols) * self.n_aug
            
        return len(self.mols)

    def __getitem__(self, index):
        
        real_index = index % len(self.mols)        
        mol = self.mols[real_index]

        if self.mode == 'train':
            s = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True, canonical=False)
            
        else:
            s = Chem.MolToSmiles(mol, canonical=True)
        
        return s, self.targets[real_index]


class MeanPooling(nn.Module):
    
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, last_hidden_state, attention_mask):
        
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        
        return summed / counts


class ContextPooling(nn.Module):
    
    def __init__(self, hidden_size, dropout=0.4):
        super(ContextPooling, self).__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, last_hidden_state, attention_mask):
        
        cls_token = last_hidden_state[:, 0]
        
        x = self.dense(cls_token)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class ChemBERTa(nn.Module):

    def __init__(
        self,
        hidden_size,
        checkpoint,
        config_path=None,
        pretrained=True
    ):        
        super(ChemBERTa, self).__init__()

        if config_path is None:
            self._config = AutoConfig.from_pretrained(checkpoint)
            
        else:
            with open(config_path, 'r') as f:
                self._config = AutoConfig.from_dict(json.load(f))

        if pretrained:
            self.transformer = AutoModel.from_pretrained(checkpoint, config=self._config)
            
        else:
            self.transformer = AutoModel(self._config)
        
        self.mean_pool = MeanPooling()
        self.context_pool = ContextPooling(hidden_size, dropout=self._config.hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)
    
    def feature(self, input_ids, attention_mask):
        
        hidden = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        mean_features = self.mean_pool(hidden, attention_mask)
        context_features = self.context_pool(hidden, attention_mask)
        
        return mean_features + context_features

    def forward(self, inputs):
        
        feats = self.feature(inputs['input_ids'], inputs['attention_mask'])
        
        return self.fc(feats)


class MD:
    
    def __init__(
        self, 
        data,
        cat_cols,
        n_splits,
        batch_size,
        learning_rate,
        weight_decay,
        hidden_size,
        checkpoint,
        context,
        patience,
        epochs
    ):

        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self._aug_factor = 30
        
        self.data = data
        self.cat_cols = cat_cols
        self._n_splits = n_splits
        self._batch_size = batch_size
        
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._hidden_size = hidden_size
        self._checkpoint = checkpoint
        self._context = context
        self._patience = patience
        self._epochs = epochs
        
        self._config = None
        self._tokenizer = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._chemberta_oof_preds = []

    def _prepare_cv(self, df):
        
        oof_preds = np.zeros(len(df))
        
        cv = KFold(n_splits=self._n_splits, shuffle=True, random_state=42)

        return cv, oof_preds

    def _tokenize(self, X):
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        
        tokenized_data = self._tokenizer(
            list(X['SMILES']),
            padding=True,
            truncation=True,
            max_length=self._context,
            return_tensors='pt'
        )
        
        return tokenized_data

    def _initialize_model(self):
        
        if self._config is None:
            self._config = AutoConfig.from_pretrained(self._checkpoint)
    
        model = ChemBERTa(
            hidden_size=self._hidden_size, 
            checkpoint=self._checkpoint, 
            pretrained=True, 
            config_path=None
        )
        
        model.to(self._device)
    
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
        return model
    
    def _collate_fn(self, batch):
        
        smiles, targets = zip(*batch)
        
        enc = self._tokenizer(
            list(smiles),
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        
        return (enc['input_ids'], enc['attention_mask'], torch.tensor(targets, dtype=torch.float32))

    def _create_loader(self, smiles, y, mode, shuffle=True):
        
        n_aug = self._aug_factor if mode == 'train' else 1
        
        dataset = SmilesDataset(smiles, y, mode, n_aug)
        
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _train_model(self, target, title):
        
        df = self.data.dropna(subset=[target]).reset_index(drop=True)
        
        X = df.drop(['id'] + self._properties, axis=1)
        y = df[target].values.astype('float32')
        
        cv, oof_preds = self._prepare_cv(X)
        tokenized = self._tokenize(X)
        
        models = []
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
            
            train_inputs = {k: v[train_index] for k, v in tokenized.items()}
            valid_inputs = {k: v[valid_index] for k, v in tokenized.items()}
            
            y_train = torch.from_numpy(y[train_index])
            y_valid = torch.from_numpy(y[valid_index])
            
            model = self._initialize_model()
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay
            )
            
            criterion = nn.L1Loss()
            
            best_loss = float('inf')
            patience = 0
            best_state = None
            
            train_smiles = X['SMILES'].iloc[train_index].tolist()
            valid_smiles = X['SMILES'].iloc[valid_index].tolist()
            
            train_loader = self._create_loader(train_smiles, y_train, mode='train')
            valid_loader = self._create_loader(valid_smiles, y_valid, mode='valid', shuffle=False)
            
            for _ in range(self._epochs):
                
                model.train()
                
                for ids, mask, target in train_loader:
                    
                    batch = {
                        'input_ids': ids.to(self._device),
                        'attention_mask': mask.to(self._device)
                    }
                    target = target.to(self._device)
                    
                    optimizer.zero_grad()
                    preds = model(batch).squeeze()
                    
                    loss = criterion(preds, target)
                    loss.backward()
                    
                    optimizer.step()
                
                model.eval()
                valid_loss = 0.0
                
                with torch.no_grad():
                    for ids, mask, target in valid_loader:
                        
                        batch = {
                            'input_ids': ids.to(self._device),
                            'attention_mask': mask.to(self._device)
                        }
                        target = target.to(self._device)
                        preds = model(batch).squeeze()
                        valid_loss += criterion(preds, target).item()
                
                valid_loss /= len(valid_loader)
                
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    
                    patience = 0
                    
                else:
                    patience += 1
                    
                    if patience >= self._patience:
                        break
            
            model.load_state_dict(best_state)
            
            model.to(self._device)
            model.eval()
            
            valid_preds = []
            with torch.no_grad():
                for ids, mask, _ in valid_loader:
                    
                    batch = {
                        'input_ids': ids.to(self._device),
                        'attention_mask': mask.to(self._device)
                    }                    
                    preds = model(batch).squeeze().cpu().numpy()
                    preds = np.atleast_1d(preds)
                    valid_preds.extend(preds.tolist())
            
            oof_preds[valid_index] = valid_preds
            
            model.cpu()
            models.append(model)
            torch.cuda.empty_cache()
        
        return models, oof_preds

    def _save_model(self, models, oof_preds, title):
        
        base_path = os.path.join('/kaggle/working', title.capitalize())
        os.makedirs(base_path, exist_ok=True)
    
        tokenizer_path = os.path.join(base_path, 'Tokenizer')
        self._tokenizer.save_pretrained(tokenizer_path)
    
        config_path = os.path.join(base_path, 'Config')
        os.makedirs(config_path, exist_ok=True)
        
        self._config.save_pretrained(config_path)
        
        models_path = os.path.join(base_path, 'Models')
        os.makedirs(models_path, exist_ok=True)
        
        for i, model in enumerate(models):
            
            model_path = os.path.join(models_path, f'{title}-fold-{i+1}.pth')
            torch.save(model, model_path)

        name = title.replace('-', '_')
        
        oof_preds = pd.DataFrame(oof_preds, columns=['oof_preds'])
        oof_preds.to_csv(os.path.join(base_path, f'{name}_oof_preds.csv'), index=False)

    def train_and_save_model(self, title):
        
        for target in self._properties:
            
            models, oof_preds = self._train_model(target, title)
            
            valid_mask = self.data[target].notna().values
            valid_indices = np.where(valid_mask)[0]
            
            full_oof_preds = np.full(len(self.data), np.nan, dtype=float)
            full_oof_preds[valid_indices] = oof_preds
            
            self._save_model(models, full_oof_preds, f'{title.lower()}-{target.lower()}')
            
            self._chemberta_oof_preds.append(full_oof_preds)
            print(f'Finished training {title} on target: {target}')
        
        solution_data = self.data[['id'] + self._properties]
        
        oof_matrix = np.vstack(self._chemberta_oof_preds).T
        
        oof_data = pd.DataFrame(oof_matrix, columns=self._properties)
        oof_data['id'] = self.data['id'].values
        oof_data = oof_data[['id'] + self._properties]
        
        model_score = score(solution_data, oof_data, 'id')
        print(f'\nwMAE for {title}: {model_score:.3f}')   


md = MD(
    train_data,
    cat_cols,
    CFG.n_splits,
    CFG.batch_size,
    CFG.learning_rate,
    CFG.weight_decay,
    CFG.hidden_size,
    CFG.checkpoint,
    CFG.context,
    CFG.patience,
    CFG.epochs
)


md.train_and_save_model('ChemBERTa-A')

