!pip install /kaggle/input/neurips-opp2025-download-libraries/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  /kaggle/input/neurips-opp2025-download-libraries/mordred-1.2.0.tar.gz


import os
import json
import types
import joblib
import numbers
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


import numpy as np
import polars as pl
import pandas as pd
from rdkit import Chem
from mordred import Calculator
from mordred import descriptors


import lightgbm as lgb
from metric import score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


import torch
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


pd.options.display.max_columns = None


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

    chemberta_root_path = Path('/kaggle/input/neurips-opp2025-a-chemberta-t')
    fastsmiles_root_path = Path('/kaggle/input/neurips-opp2025-a-fastsmiles-t')
    
    threshold = 0.80
    
    checkpoint = 'DeepChem/ChemBERTa-77M-MTR'
    n_splits = 5
    batch_size = 1024
    hidden_size = 384
    context = 512
    seed = 42
    
    ctb_weight = 0.4
    lgb_weight = 0.3
    xgb_weight = 0.3
    
    chemberta_weight = 0.4
    fastsmiles_weight = 0.6


class FE:
    
    def __init__(
        self, 
        ext1_data_path, 
        ext2_data_path,
        ext3_data_path,
        ext4_data_path,
        threshold
    ):
        
        self._ext1_data_path = ext1_data_path
        self._ext2_data_path = ext2_data_path
        self._ext3_data_path = ext3_data_path
        self._ext4_data_path = ext4_data_path
            
        self._threshold = threshold
        self._batch_size = 16384
        
        self._ignore_3D = True   
        self._bad_features = None
        
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
    
    def _aggregate_smiles(self, df):
        
        mols = df['SMILES'].map(Chem.MolFromSmiles)
        calc = Calculator(descriptors, ignore_3D=self._ignore_3D)
        
        features = calc.pandas(mols).apply(pd.to_numeric, errors='coerce')
        
        if self._bad_features is None:
            self._bad_features = [
                c for c in features.columns 
                if features[c].nunique(dropna=False) <= 1 
                or features[c].isna().mean() > self._threshold
            ]
        
        return pd.concat([df, features], axis=1)
    
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
        
        raw_data = df.copy()
        raw_data = self._cast_datatypes(raw_data)     
            
        df = self._aggregate_smiles(df)
        df = df.drop(columns=self._bad_features, errors='ignore')        
        df = self._cast_datatypes(df)        
        self.info(df)
        
        cat_cols = df.select_dtypes(include=['string']).columns.tolist()
        if cat_cols:
            df[cat_cols] = df[cat_cols].fillna('Missing').astype('category')
        
        return df, raw_data, cat_cols


fe = FE(
    CFG.ext1_data_path, 
    CFG.ext2_data_path,
    CFG.ext3_data_path,
    CFG.ext4_data_path,
    CFG.threshold
)


train_data, raw_data, cat_cols = fe.apply_fe(CFG.train_path)


test_data, _, _ = fe.apply_fe(CFG.test_path)


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


class ChemBERTa_Inference:
    
    def __init__(
        self,
        data,
        cat_cols,
        root_path,
        n_splits,
        hidden_size,
        checkpoint,
        batch_size,
        context
    ):
        
        self.data = data
        self.cat_cols = cat_cols
        self._root_path = root_path
        self._n_splits = n_splits
        self._hidden_size = hidden_size
        self._checkpoint = checkpoint
        self._batch_size = batch_size
        self._context = context
        
        self._tokenizer = None
        self._config = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self._aug_factor = 300

    def _target_folder(self, target):
        
        for directory in os.listdir(self._root_path):
            
            full = os.path.join(self._root_path, directory)
            
            if os.path.isdir(full) and target.lower() in directory.lower():
                return full
                
        raise FileNotFoundError(f'No sub-folder for target {target} under {self._root_path}')

    def _load_models_for_target(self, target, title):
        
        base_path = self._target_folder(target)
        config_dir = os.path.join(base_path, 'Config')
        models_dir = os.path.join(base_path, 'Models')
        tokenizer_dir = os.path.join(base_path, 'Tokenizer')
        
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self._config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
        
        models = []
        
        for fold in range(1, self._n_splits + 1):
            model_path = os.path.join(models_dir, f'{title.lower()}-{target.lower()}-fold-{fold}.pth')
            
            try:
                model = torch.load(model_path, map_location=self._device, weights_only=False)
                
            except TypeError:
                model = torch.load(model_path, map_location=self._device)
                
            model.to(self._device).eval()
            models.append(model)

        title = title.replace('-', '_')
        
        csv_name = f'{title.lower()}_{target.lower()}_oof_preds.csv'
        oof_preds = pd.read_csv(os.path.join(base_path, csv_name))['oof_preds'].values
        
        return models, oof_preds

    def _predict_with_models(self, df, models):
        
        smiles_list = df['SMILES'].tolist()
        dummy_targets = [0] * len(smiles_list)
        dataset = SmilesDataset(smiles_list, dummy_targets, mode='train', n_aug=self._aug_factor)

        def tta_collate(batch):
            
            smiles_batch, _ = zip(*batch)
            
            enc = self._tokenizer(
                list(smiles_batch),
                padding=True,
                truncation=True,
                max_length=self._context,
                return_tensors='pt'
            )
            
            return enc['input_ids'], enc['attention_mask']

        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, collate_fn=tta_collate)

        fold_preds = []
        
        for model in models:
            
            preds = []
            
            with torch.no_grad():
                for ids, mask in loader:
                    
                    batch = {'input_ids': ids.to(self._device), 'attention_mask': mask.to(self._device)}
                    out = model(batch).squeeze().cpu().numpy()
                    preds.extend(np.atleast_1d(out).tolist())
            
            preds = np.array(preds).reshape(self._aug_factor, len(smiles_list)).mean(axis=0)            
            fold_preds.append(preds)
            
        return np.mean(fold_preds, axis=0)

    def create_submission(self, test_data, title):
        
        ids = test_data['id'].values
        
        sorted_idx = np.argsort(ids)
        inverse_idx = np.argsort(sorted_idx)
        
        sorted_test = test_data.iloc[sorted_idx].reset_index(drop=True)
        ensemble_preds = np.zeros((len(sorted_test), len(self._properties)), dtype='float32')
        
        for col_index, target in enumerate(self._properties):
            
            models, _ = self._load_models_for_target(target, title)
            ensemble_preds[:, col_index] = self._predict_with_models(sorted_test, models)
            
        ensemble_preds = ensemble_preds[inverse_idx]
        
        subm_data = pd.DataFrame(ensemble_preds, columns=self._properties)
        subm_data['id'] = ids
        subm_data = subm_data[['id'] + self._properties]
        subm_data.to_csv('chemberta_submission.csv', index=False)
        
        display(subm_data.head())


class FastSMILES_Inference:
    
    def __init__(
        self,
        train_data,
        test_data, 
        cat_cols,
        root_path,
        ctb_weight,
        lgb_weight,
        xgb_weight,
        n_splits
    ):

        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        self.train_data = train_data
        self.test_data = test_data
        self.cat_cols = cat_cols
        self.root_path = root_path
        
        self._ctb_weight = ctb_weight
        self._lgb_weight = lgb_weight
        self._xgb_weight = xgb_weight
        
        self._n_splits = n_splits

        self._ctb_oof_preds = []
        self._lgb_oof_preds = []
        self._xgb_oof_preds = []

        self._ctb_preds = []
        self._lgb_preds = []
        self._xgb_preds = []

    def find_folder(self, target):
        
        for d in os.listdir(self.root_path):
            
            full = os.path.join(self.root_path, d)
            
            if os.path.isdir(full) and d.lower() == target.lower():
                return full
                
        raise FileNotFoundError(f'No folder for target {target} under {self.root_path}')
    
    def load_model(self, target, title):
        
        base = os.path.join(self.find_folder(target), title)
        
        models_dir = os.path.join(base, 'Models')
        models = []
        
        for fold in range(1, self._n_splits + 1):
            path = os.path.join(models_dir, f'{title.lower()}-fold-{fold}.pkl')
            models.append(joblib.load(path))
        
        path = os.path.join(base, f'{title.lower().replace("-", "_")}_oof_preds.csv')
        oof_preds = pd.read_csv(path)['oof_preds'].values
        
        return models, oof_preds

    def infer_model(self, models):
        
        data = self.test_data.drop(['id'], axis=1)

        return np.mean([model.predict(data) for model in models], axis=0)
    
    def load_and_infer_model(self, title):
        
        for col in self.cat_cols:
            self.train_data[col] = self.train_data[col].astype('category')
            self.test_data[col] = self.test_data[col].astype('category')
        
        for target in self._properties:
            
            valid_mask = self.train_data[target].notna().values
            valid_indices = np.where(valid_mask)[0]
            
            models, oof_preds = self.load_model(target, title)
            preds = self.infer_model(models)
            
            if title.startswith('CTB'):
                self._ctb_oof_preds.append(oof_preds)
                self._ctb_preds.append(preds)          
                
            elif title.startswith('LGB'):
                self._lgb_oof_preds.append(oof_preds)
                self._lgb_preds.append(preds)           
                
            elif title.startswith('XGB'):
                self._xgb_oof_preds.append(oof_preds)
                self._xgb_preds.append(preds)
        
        solution_data = self.train_data[['id'] + self._properties]
        
        if title.startswith('CTB'):
            oof_list = self._ctb_oof_preds
        
        elif title.startswith('LGB'):
            oof_list = self._lgb_oof_preds
        
        elif title.startswith('XGB'):
            oof_list = self._xgb_oof_preds
        
        oof_matrix = np.vstack(oof_list).T
        
        oof_data = pd.DataFrame(oof_matrix, columns=self._properties)
        oof_data['id'] = self.train_data['id'].values
        oof_data = oof_data[['id'] + self._properties]
        
        model_score = score(solution_data, oof_data, 'id')
        print(f'wMAE for {title}: {model_score:.3f}')           
    
    def inference(self):
        
        ensemble_oof_preds = np.vstack([
            (self._ctb_weight * ctb +
             self._lgb_weight * lgb +
             self._xgb_weight * xgb)
            for ctb, lgb, xgb in zip(self._ctb_oof_preds, self._lgb_oof_preds, self._xgb_oof_preds)
        ]).T
        
        ensemble_preds = np.vstack([
            (self._ctb_weight * ctb +
             self._lgb_weight * lgb +
             self._xgb_weight * xgb)
            for ctb, lgb, xgb in zip(self._ctb_preds, self._lgb_preds, self._xgb_preds)
        ]).T
        
        solution_data = self.train_data[['id'] + self._properties]
        
        oof_data = pd.DataFrame(ensemble_oof_preds, columns=self._properties)
        oof_data['id'] = self.train_data['id'].values
        oof_data = oof_data[['id'] + self._properties]
        
        ensemble_score = score(solution_data, oof_data, 'id')
        print(f'wMAE for Ensemble: {ensemble_score:.3f}')
        
        subm_data = pd.DataFrame(ensemble_preds, columns=self._properties)
        subm_data['id'] = self.test_data['id'].values
        subm_data = subm_data[['id'] + self._properties]
        subm_data.to_csv('fastsmiles_submission.csv', index=False)
        
        display(subm_data.head())


chemberta_inference = ChemBERTa_Inference(
    raw_data, 
    cat_cols,
    CFG.chemberta_root_path,
    CFG.n_splits,
    CFG.hidden_size,
    CFG.checkpoint,
    CFG.batch_size,
    CFG.context
)


fastsmiles_inference = FastSMILES_Inference(
    train_data,
    test_data,
    cat_cols,
    CFG.fastsmiles_root_path,
    CFG.ctb_weight,
    CFG.lgb_weight,
    CFG.xgb_weight,
    CFG.n_splits
)


chemberta_inference.create_submission(test_data, 'ChemBERTa-A')


fastsmiles_inference.load_and_infer_model('CTB-A')


fastsmiles_inference.load_and_infer_model('LGB-A')


fastsmiles_inference.load_and_infer_model('XGB-A')


fastsmiles_inference.inference()


chemberta_submission = pd.read_csv('/kaggle/working/chemberta_submission.csv')
fastsmiles_submission = pd.read_csv('/kaggle/working/fastsmiles_submission.csv')

target_cols = [col for col in chemberta_submission.columns if col != 'id']

ensemble = chemberta_submission.copy()
for col in target_cols:
    ensemble[col] = CFG.chemberta_weight * chemberta_submission[col] + CFG.fastsmiles_weight * fastsmiles_submission[col]

ensemble.to_csv('/kaggle/working/submission.csv', index=False)
display(ensemble.head())

