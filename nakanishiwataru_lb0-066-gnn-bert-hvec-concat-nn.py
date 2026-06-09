#Test offline install torch-molecule
!pip install /kaggle/input/torch-molecule-whl/torch_molecule-0.1.3-py3-none-any.whl --no-deps --no-index --find-links=file:///kaggle/input/torch-molecule-pkg


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install /kaggle/input/torchgeometric-whl/torch_geometric-2.6.1-py3-none-any.whl


import os
import time
import argparse
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    StratifiedGroupKFold,
    GroupKFold
)
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from transformers import PreTrainedModel, AutoConfig, BertModel, BertTokenizerFast, BertConfig, AutoModel, AutoTokenizer
from transformers.activations import ACT2FN

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Draw,
    rdmolops
)
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors

from torch_molecule import GREAMolecularPredictor, GNNMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

import shutil
RDLogger.DisableLog('rdApp.*')  
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class cfg:
    mode = "Test"
    targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
    epoch = 400
    


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan



def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()

    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Remove smiles that have target value in train_df from cross_smiles
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)
            
    # Fill Nan Value with cross smiles of extra data
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]

    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES: {len(unique_smiles_extra)}')

    return df_train

train_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

data_tc = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})
data_tc = data_tc.reindex(['SMILES', "Tc"], axis=1)

data_tg2 = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})

data_tg3 = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
data_tg3['Tg'] = data_tg3['Tg'] - 273.15

data_dnst = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]

data_ffv = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')

train_data['SMILES'] = train_data['SMILES'].apply(lambda s: make_smile_canonical(s))
test_data['SMILES'] = test_data['SMILES'].apply(lambda s: make_smile_canonical(s))
data_dnst['SMILES'] = data_dnst['SMILES'].apply(lambda s: make_smile_canonical(s))
data_dnst = data_dnst[(data_dnst['SMILES'].notnull())&(data_dnst['Density'].notnull())&(data_dnst['Density'] != 'nylon')]
data_dnst['Density'] = data_dnst['Density'].astype('float64')
data_dnst['Density'] -= 0.118 # ??


train_data = add_extra_data(train_data, data_tc, 'Tc')
train_data = add_extra_data(train_data, data_tg2, 'Tg')
train_data = add_extra_data(train_data, data_tg3, 'Tg')
train_data = add_extra_data(train_data, data_ffv, 'FFV')
train_data = add_extra_data(train_data, data_dnst, 'Density')


def augment_smiles_dataset(df: pd.DataFrame,
                              smiles_column: str = 'SMILES',
                              augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
                              n_augmentations: int = 100,
                              preserve_original: bool = True,
                              random_seed: Optional[int] = None
                          ) -> pd.DataFrame:
    
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
                    enum_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
                    augmented.append(enum_smiles)
                
            elif strategy == 'kekulize':
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
                
            elif strategy == 'stereo_enum':
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)

            return list(set(augmented))

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
                    new_row['is_augmented'] = True
                    augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)

    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")

    return augmented_df

#train_data = augment_smiles_dataset(train_data)
#test_data = augment_smiles_dataset(test_data)


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        pooler_size = getattr(config, 'pooler_hidden_size', config.hidden_size)
        self.dense = nn.Linear(pooler_size, pooler_size)

        dropout_prob = getattr(config, 'pooler_dropout', config.hidden_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        self.activation = getattr(config, 'pooler_hidden_act', config.hidden_act)
        self.config = config

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0] # CLS token
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
        self.output = torch.nn.Linear(pooler_output_dim, 1)

    def forward(
        self, 
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        pooled_output = self.pooler(outputs.last_hidden_state)
        regression_output = self.output(pooled_output)


        return {
            "logits":regression_output,
            "pooled_vec":pooled_output
        }


def get_pretrained(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = CustomModel(config)

    if model_path.endswith("pytorch_model.bin"):
        model.load_state_dict(torch.load(model_path))
    else:
        model.backbone = AutoModel.from_pretrained(model_path)

    for param in model.backbone.parameters():
        param.requires_grad = True
    return model


import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import numpy as np


class SMILESDataset(Dataset):
    def __init__(self, tokenizer, smiles_list, labels=None, max_length=512, mode="Train"):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.tokenizer.cls_token + self.smiles_list[idx]

        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        if self.mode == "Train":
            label = self.labels[idx]
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float32)
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }


def load_model(path):
    config = AutoConfig.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
    model = CustomModel(config).cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model

def test_model(model, test_dl, device):
    h_vecs = []
    agg_preds = []
    model.to(device)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            h_vec = output['pooled_vec'].cpu().numpy()
            h_vecs.extend(h_vec)
            
    return h_vecs


weights = {
        "Tg": "/kaggle/input/hvec-bert-model/trained_bert_model_Tg_best.pth",
        "Tc": "/kaggle/input/hvec-bert-model/trained_bert_model_Tc_best.pth",
        "FFV": "/kaggle/input/hvec-bert-model/trained_bert_model_FFV_best.pth",
        "Density": "/kaggle/input/hvec-bert-model/trained_bert_model_Density_best.pth",
        "Rg": "/kaggle/input/hvec-bert-model/trained_bert_model_Rg_best.pth",
    }

# Predict Test dataset
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result = {}
train_bert_h_vecs = {}
valid_bert_h_vecs = {}
test_bert_h_vecs = {}
for i in tqdm(range(len(cfg.targets))):
    target = cfg.targets[i]    
    test_ds = SMILESDataset(tokenizer, test_data['SMILES'].to_list(), mode="Test")
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    notnull_train_df, notnull_valid_df = train_test_split(train_data[train_data[target].notnull()], test_size=0.1, random_state=23)
    notnull_train_df = notnull_train_df.reset_index(drop=True)
    notnull_valid_df = notnull_valid_df.reset_index(drop=True)

    smiles_train = notnull_train_df['SMILES']
    labels_train = notnull_train_df[target].values.reshape(-1,1)

    smiles_valid = notnull_valid_df['SMILES']
    labels_valid = notnull_valid_df[target].values.reshape(-1,1)

    train_ds = SMILESDataset(tokenizer, smiles_train, labels=labels_train)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    valid_ds = SMILESDataset(tokenizer, smiles_valid, labels=labels_valid)
    valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    model = load_model(weights[target])

    if cfg.mode == "Train":
        train_h_vecs = test_model(model, train_dl, device)
        valid_h_vecs = test_model(model, valid_dl, device)

        train_bert_h_vecs[target] = train_h_vecs
        valid_bert_h_vecs[target] = valid_h_vecs
        
    else:
        test_h_vecs = test_model(model, test_dl, device)
        valid_h_vecs = test_model(model, valid_dl, device)

        test_bert_h_vecs[target] = test_h_vecs
        valid_bert_h_vecs[target] = valid_h_vecs


def test_models(train_data, test_data, model_type='both'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    property_columns = ["Tg", "FFV", "Tc", "Density", "Rg"]
    print(f"Property columns: {property_columns}")

    grea_model_weights = {
        "Tg":"/kaggle/input/grea-hvec-model/GREA_BEST_Tg.pt",
        "FFV":"/kaggle/input/grea-hvec-model/GREA_BEST_FFV.pt",
        "Tc":"/kaggle/input/grea-hvec-model/GREA_BEST_Tc.pt",
        "Density":"/kaggle/input/grea-hvec-model/GREA_BEST_Density.pt",
        "Rg":"/kaggle/input/grea-hvec-model/GREA_BEST_Rg.pt"
    }

    gnn_model_weights = {
        "Tg":"/kaggle/input/hvec-gnn-model/GNN_Tg_BEST.pt",
        "FFV":"/kaggle/input/hvec-gnn-model/GNN_FFV_BEST.pt",
        "Tc":"/kaggle/input/hvec-gnn-model/GNN_Tc_BEST.pt",
        "Density":"/kaggle/input/hvec-gnn-model/GNN_Density_BEST.pt",
        "Rg":"/kaggle/input/hvec-gnn-model/GNN_Rg_BEST.pt"
    }

    # Predict GREA model
    if model_type in ['grea', 'both']:
        train_grea_h_vecs = {}
        valid_grea_h_vecs = {}
        test_grea_h_vecs = {}
        for target in property_columns:
            print(f"Test GREA model for {target} properties...")
            X_test = test_data['SMILES']
            notnull_train_data = train_data[train_data[target].notnull()]
            X = notnull_train_data['SMILES']
            X_train, X_valid = train_test_split(X, test_size=0.1, random_state=23)
            
            grea_model = GREAMolecularPredictor()

            # Load best weight
            grea_model.load_from_local(grea_model_weights[target])

            # Calc Test Prediction
            if cfg.mode == "Train":
                grea_train_output = grea_model.predict(X_train.tolist())
                train_grea_h_vec = grea_train_output["representation"]
                train_grea_h_vecs[target] = train_grea_h_vec

                grea_valid_output = grea_model.predict(X_valid.tolist())
                valid_grea_h_vec = grea_valid_output["representation"]
                valid_grea_h_vecs[target] = valid_grea_h_vec
                
            else:
                grea_test_output = grea_model.predict(X_test.tolist())
                test_grea_h_vec = grea_test_output["representation"]
                test_grea_h_vecs[target] = test_grea_h_vec
                
                grea_valid_output = grea_model.predict(X_valid.tolist())
                valid_grea_h_vec = grea_valid_output["representation"]
                valid_grea_h_vecs[target] = valid_grea_h_vec

    # Predict GNN model
    if model_type in ['gnn', 'both']:
        train_gnn_h_vecs = {}
        valid_gnn_h_vecs = {}
        test_gnn_h_vecs = {}
        
        for target in property_columns:
            print(f"Test GNN model for {target} properties...")
            X_test = test_data['SMILES']
            notnull_train_data = train_data[train_data[target].notnull()]
            X = notnull_train_data['SMILES']
            X_train, X_valid = train_test_split(X, test_size=0.1, random_state=23)
            
            gnn_model = GNNMolecularPredictor()

            # Load best weight
            gnn_model.load_from_local(gnn_model_weights[target])

            # Calc Test Prediction
            if cfg.mode == "Train":
                gnn_train_output = gnn_model.predict(X_train.tolist())
                train_gnn_h_vec = gnn_train_output["representation"]
                train_gnn_h_vecs[target] = train_gnn_h_vec

                gnn_valid_output = gnn_model.predict(X_valid.tolist())
                valid_gnn_h_vec = gnn_valid_output["representation"]
                valid_gnn_h_vecs[target] = valid_gnn_h_vec

            
            else:
                gnn_test_output = gnn_model.predict(X_test.tolist())
                test_gnn_h_vec = gnn_test_output["representation"]
                test_gnn_h_vecs[target] = test_gnn_h_vec
                
                gnn_valid_output = gnn_model.predict(X_valid.tolist())
                valid_gnn_h_vec = gnn_valid_output["representation"]
                valid_gnn_h_vecs[target] = valid_gnn_h_vec

    print('=== Output hvec Complete ===')
    return train_grea_h_vecs, valid_grea_h_vecs, test_grea_h_vecs, train_gnn_h_vecs, valid_gnn_h_vecs, test_gnn_h_vecs


train_grea_h_vecs, valid_grea_h_vecs, test_grea_h_vecs, train_gnn_h_vecs, valid_gnn_h_vecs, test_gnn_h_vecs = test_models(train_data, test_data, model_type='both')


class HVecDataset(Dataset):
    def __init__(self, dataframe, tgt_name, grea_h_vecs, gnn_h_vecs, bert_h_vecs, mode="Train"):
        self.tgt_name = tgt_name
        self.dataframe = dataframe
        self.grea_h_vecs = grea_h_vecs
        self.gnn_h_vecs = gnn_h_vecs
        self.bert_h_vecs = bert_h_vecs
        self.mode = mode
        if self.mode == "Train":
            self.train_label = dataframe[tgt_name]

    def __len__(self):
        return len(self.grea_h_vecs)

    def __getitem__(self, idx):
        grea_h_vec = torch.tensor(self.grea_h_vecs.iloc[idx], dtype=torch.float32)
        gnn_h_vec = torch.tensor(self.gnn_h_vecs.iloc[idx], dtype=torch.float32)
        bert_h_vec = torch.tensor(self.bert_h_vecs.iloc[idx], dtype=torch.float32)
        if self.mode == "Train":
            target = torch.tensor(self.train_label.iloc[idx], dtype=torch.float32)
            return grea_h_vec, gnn_h_vec, bert_h_vec, target

        else:
            return grea_h_vec, gnn_h_vec, bert_h_vec


class FusionModel(nn.Module):
    def __init__(self, dim_grea, dim_gnn, dim_bert, intgr_hidden_dim=256, output_dim=1):
        super().__init__()

        self.proj_grea = nn.Linear(dim_grea, intgr_hidden_dim)
        self.proj_gnn = nn.Linear(dim_gnn, intgr_hidden_dim)
        self.proj_bert = nn.Linear(dim_bert, intgr_hidden_dim)

        self.alpha = nn.Parameter(torch.tensor(0.25))
        self.beta = nn.Parameter(torch.tensor(0.25))
        self.gamma = nn.Parameter(torch.tensor(0.5))

        self.mlp = nn.Sequential(
            nn.Linear(intgr_hidden_dim*3, intgr_hidden_dim),
            nn.ReLU(),
            nn.Linear(intgr_hidden_dim, output_dim)
        )

    def forward(self, grea_h_vec, gnn_h_vec, bert_h_vec):
        h_grea = self.proj_grea(grea_h_vec)
        h_gnn = self.proj_gnn(gnn_h_vec)
        h_bert = self.proj_bert(bert_h_vec)

        h_all = torch.cat([self.alpha*h_grea, self.beta*h_gnn, self.gamma*h_bert], dim=1)
        out = self.mlp(h_all)

        return out


device = "cuda" if torch.cuda.is_available() else "cpu"
dratio = {
        "Tg": 0.3,
        "FFV": 0.3,
        "Tc": 0.3,
        "Density": 0.3,
        "Rg": 0.3,
    }

if cfg.mode == "Train":
    best_vals = {}
    for target in cfg.targets:
        print(f'Training {target}')
        # change hidden vecs to dataframe
        grea_train_vecs = pd.DataFrame(train_grea_h_vecs[target])
        gnn_train_vecs = pd.DataFrame(train_gnn_h_vecs[target])
        bert_train_vecs = pd.DataFrame(train_bert_h_vecs[target])

        grea_valid_vecs = pd.DataFrame(valid_grea_h_vecs[target])
        gnn_valid_vecs = pd.DataFrame(valid_gnn_h_vecs[target])
        bert_valid_vecs = pd.DataFrame(valid_bert_h_vecs[target])
    
        notnull_train_df, notnull_valid_df = train_test_split(train_data[train_data[target].notnull()], test_size=0.1, random_state=23)
        
        train_ds = HVecDataset(notnull_train_df, target, grea_train_vecs, gnn_train_vecs, bert_train_vecs)
        train_dl = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            num_workers=4
        )
    
        valid_ds = HVecDataset(notnull_valid_df, target, grea_valid_vecs, gnn_valid_vecs, bert_valid_vecs)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=64,
            num_workers=4
        )
    
        model = FusionModel(grea_train_vecs.shape[1], gnn_train_vecs.shape[1], bert_train_vecs.shape[1]).to(device)

        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=0)
    
        # Training loop
        best_val_loss = 1e6
        for epoch in range(cfg.epoch):
            tr_loss = []
            vl_loss = []
    
            model.train()
            for i, (grea_h_vec, gnn_h_vec, bert_h_vec, label) in enumerate(train_dl):
                grea_h_vec = grea_h_vec.to(device)
                gnn_h_vec = gnn_h_vec.to(device)
                bert_h_vec = bert_h_vec.to(device)
                label = label.to(device)
    
                optimizer.zero_grad()
                output = model(grea_h_vec, gnn_h_vec, bert_h_vec).squeeze()
                
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                tr_loss.append(loss.item())
                
            avg_loss = np.mean(tr_loss)
            print(f"Target {target} Epoch: {epoch} Trn Loss: {avg_loss}", flush=True)
    
    
            model.eval()
            for i, (grea_h_vec, gnn_h_vec, bert_h_vec, label) in enumerate(valid_dl):
                grea_h_vec = grea_h_vec.to(device)
                gnn_h_vec = gnn_h_vec.to(device)
                bert_h_vec = bert_h_vec.to(device)
                label = label.to(device)
    
                with torch.inference_mode():
                    output = model(grea_h_vec, gnn_h_vec, bert_h_vec).squeeze()
                    loss = criterion(output, label)
    
                vl_loss.append(loss.item())
            avg_val_loss = np.mean(vl_loss)
            scheduler.step(avg_val_loss)
            print(f"Target {target} Epoch: {epoch} Trn Loss: {avg_loss} Val Loss: {avg_val_loss}", flush=True)
    
            #Save Model
            if best_val_loss > avg_val_loss:
                print(f"New best: {best_val_loss} -> {avg_val_loss}")
                torch.save(model.state_dict(), f"{target}_best_model.pt")
                best_val_loss = avg_val_loss
        best_vals[target] = best_val_loss


    print("Training complete")
    print(best_vals)


if cfg.mode == "Test":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = {
        "Tg": "/kaggle/input/fusion-best-weight/Tg_best_model.pt",
        "FFV": "/kaggle/input/fusion-best-weight/FFV_best_model.pt",
        "Tc": "/kaggle/input/fusion-best-weight/Tc_best_model.pt",
        "Density": "/kaggle/input/fusion-best-weight/Density_best_model.pt",
        "Rg": "/kaggle/input/fusion-best-weight/Rg_best_model.pt"
    }
    dratio = {
        "Tg": 0.3,
        "FFV": 0.3,
        "Tc": 0.3,
        "Density": 0.3,
        "Rg": 0.3,
    }

    criterion = nn.L1Loss()
    result = {}
    val_mae = {}
    for target in cfg.targets:
        print(f'Test {target}')
        # change hidden vecs to dataframe
        grea_test_vecs = pd.DataFrame(test_grea_h_vecs[target])
        gnn_test_vecs = pd.DataFrame(test_gnn_h_vecs[target])
        bert_test_vecs = pd.DataFrame(test_bert_h_vecs[target])
    
        grea_valid_vecs = pd.DataFrame(valid_grea_h_vecs[target])
        gnn_valid_vecs = pd.DataFrame(valid_gnn_h_vecs[target])
        bert_valid_vecs = pd.DataFrame(valid_bert_h_vecs[target])
        
        test_ds = HVecDataset(test_data, target, grea_test_vecs, gnn_test_vecs, bert_test_vecs, mode="Test")
        test_dl = DataLoader(
            test_ds,
            batch_size=64,
            shuffle=False,
            num_workers=4
        )
        notnull_train_df, notnull_valid_df = train_test_split(train_data[train_data[target].notnull()], test_size=0.1, random_state=23)

        valid_ds = HVecDataset(notnull_valid_df, target, grea_valid_vecs, gnn_valid_vecs, bert_valid_vecs)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=64,
            num_workers=4
        )

        model = FusionModel(grea_test_vecs.shape[1], gnn_test_vecs.shape[1], bert_test_vecs.shape[1]).to(device)
        state_dict = torch.load(weights[target], map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
        preds = []
        vl_loss = []
        with torch.inference_mode():
            for i, (grea_h_vec, gnn_h_vec, bert_h_vec) in enumerate(test_dl):
                grea_h_vec = grea_h_vec.to(device)
                gnn_h_vec = gnn_h_vec.to(device)
                bert_h_vec = bert_h_vec.to(device)
        
                output = model(grea_h_vec, gnn_h_vec, bert_h_vec).flatten()
                preds.extend(output.cpu().numpy())

            for i, (grea_h_vec, gnn_h_vec, bert_h_vec, label) in enumerate(valid_dl):
                grea_h_vec = grea_h_vec.to(device)
                gnn_h_vec = gnn_h_vec.to(device)
                bert_h_vec = bert_h_vec.to(device)
                label = label.to(device)
    
                output = model(grea_h_vec, gnn_h_vec, bert_h_vec).squeeze()
                loss = criterion(output, label)
    
                vl_loss.append(loss.item())
        avg_val_loss = np.mean(vl_loss)
        val_mae[target] = avg_val_loss
        
        result[target] = preds
    
    # make submission
    sub_df = pd.DataFrame(index=test_data['id'])
    for target, arr in result.items():
        sub_df[target] = arr

    sub_df.to_csv('submission.csv')
    print(val_mae)

