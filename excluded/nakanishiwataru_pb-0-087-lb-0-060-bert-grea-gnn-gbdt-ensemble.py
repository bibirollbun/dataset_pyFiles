#Test offline install torch-molecule
!pip install /kaggle/input/torch-molecule-whl/torch_molecule-0.1.3-py3-none-any.whl --no-deps --no-index --find-links=file:///kaggle/input/torch-molecule-pkg


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


#!pip install /kaggle/input/torchgeometric-whl/torch_geometric-2.6.1-py3-none-any.whl


import os
import time
import argparse
from typing import Optional, List, Union
import gc

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

test_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
test_data['SMILES'] = test_data['SMILES'].apply(lambda s: make_smile_canonical(s))


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

aug_test_data = augment_smiles_dataset(test_data)


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
        scaler,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        pooled_output = self.pooler(outputs.last_hidden_state)
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
            "loss":loss,
            "logits":regression_output,
            "true_loss":true_loss,
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

def test_model(model, test_dl, scaler, device):
    agg_preds = []
    model.to(device)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            preds = model(input_ids=input_ids, scaler=scaler, attention_mask=attention_mask)['logits'].cpu().numpy()

            unscaled_preds = scaler.inverse_transform(preds).flatten()
            agg_preds.extend(unscaled_preds.tolist())
    return np.array(agg_preds)



weights = {
    "Tg": "/kaggle/input/bert-weights-train-same-valid-data-for-gnn/trained_bert_model_Tg_best.pth",
    "Tc": "/kaggle/input/bert-weights-train-same-valid-data-for-gnn/trained_bert_model_Tc_best.pth",
    "FFV": "/kaggle/input/bert-weights-train-same-valid-data-for-gnn/trained_bert_model_FFV_best.pth",
    "Density": "/kaggle/input/bert-weights-train-same-valid-data-for-gnn/trained_bert_model_Density_best.pth",
    "Rg": "/kaggle/input/bert-weights-train-same-valid-data-for-gnn/trained_bert_model_Rg_best.pth",
}


# Predict Test dataset
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
scalers = joblib.load('/kaggle/input/bert-scaler/bert_scaler.pkl')
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/smiles-deberta77m-tokenizer')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result = {}
for i in tqdm(range(len(targets))):
    target = targets[i]
    scaler = scalers[i]
    
    test_ds = SMILESDataset(tokenizer, aug_test_data['SMILES'].to_list(), mode="Test")
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)


    model = load_model(weights[target])

    augmented_preds = test_model(model, test_dl, scaler, device).flatten()
    result[target] = augmented_preds

sub_df = pd.DataFrame(index=aug_test_data['id'])
for target, arr in result.items():
    sub_df[target] = arr

bert_sub_df = sub_df.groupby(['id']).mean()


del tokenizer, model
gc.collect()


def test_models(aug_test_data, model_type='both'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    property_columns = ["Tg", "FFV", "Tc", "Density", "Rg"]
    print(f"Property columns: {property_columns}")
    X_test = aug_test_data['SMILES']

    grea_model_weights = {
        "Tg":"/kaggle/input/grea-best-weights/GREA_BEST_Tg.pt",
        "FFV":"/kaggle/input/grea-best-weights/GREA_BEST_FFV.pt",
        "Tc":"/kaggle/input/grea-best-weights/GREA_BEST_Tc.pt",
        "Density":"/kaggle/input/grea-best-weights/GREA_BEST_Density.pt",
        "Rg":"/kaggle/input/grea-best-weights/GREA_BEST_Rg.pt"
    }

    gnn_model_weights = {
        "Tg":"/kaggle/input/gnn-best-weight/GNN_Tg_BEST.pt",
        "FFV":"/kaggle/input/gnn-best-weight/GNN_FFV_BEST.pt",
        "Tc":"/kaggle/input/gnn-best-weight/GNN_Tc_BEST.pt",
        "Density":"/kaggle/input/gnn-best-weight/GNN_Density_BEST.pt",
        "Rg":"/kaggle/input/gnn-best-weight/GNN_Rg_BEST.pt"
    }

    # Predict GREA model
    if model_type in ['grea', 'both']:
        grea_result = {}
        grea_val_mae = {}
        grea_h_vecs = {}
        for target in property_columns:
            print(f"Test GREA model for {target} properties...")
            grea_model = GREAMolecularPredictor()

            # Load best weight
            grea_model.cpu_load_from_local(device, grea_model_weights[target])

            # Calc Test Prediction
            grea_output = grea_model.predict(X_test.tolist())
            grea_result[target] = grea_output['prediction'].flatten()

        #Make submission file
        grea_sub_df = pd.DataFrame(index=aug_test_data['id'])

        for target, arr in grea_result.items():
            grea_sub_df[target] = arr

        grea_sub_df = grea_sub_df.groupby(['id']).mean()


            

    # Predict GNN model
    if model_type in ['gnn', 'both']:
        gnn_result = {}
        gnn_val_mae = {}
        gnn_h_vecs = {}
        for target in property_columns:
            print(f"Test GNN model for {target} properties...")
            gnn_model = GNNMolecularPredictor()

            # Load best weight
            gnn_model. cpu_load_from_local(device, gnn_model_weights[target])

            # Calc Test Prediction
            gnn_output = gnn_model.predict(X_test.tolist())
            gnn_result[target] = gnn_output['prediction'].flatten()

        # Make Submission
        gnn_sub_df = pd.DataFrame(index=aug_test_data['id'])

        for target, arr in gnn_result.items():
            gnn_sub_df[target] = arr
        gnn_sub_df = gnn_sub_df.groupby(['id']).mean()

    return grea_sub_df, gnn_sub_df


grea_sub_df, gnn_sub_df = test_models(aug_test_data, model_type='both')


!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
# Show all columns
import os
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from mordred import Calculator, descriptors
pd.set_option('display.max_columns', None)

tg=pd.read_csv('/kaggle/input/modred-dataset/desc_tg.csv')
tc=pd.read_csv('/kaggle/input/modred-dataset/desc_tc.csv')
rg=pd.read_csv('/kaggle/input/modred-dataset/desc_rg.csv')
ffv=pd.read_csv('/kaggle/input/modred-dataset/desc_ffv.csv')
density=pd.read_csv('/kaggle/input/modred-dataset/desc_de.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ID=test['id']


mols_test = [Chem.MolFromSmiles(s) for s in test.SMILES]
# Initialize the Mordred Calculator
calc = Calculator(descriptors, ignore_3D=True) # ignore_3D=True for 2D descriptors
desc_test = calc.pandas(mols_test)


import networkx as nx
from rdkit.Chem import rdmolops

useless_cols = [
    "MaxPartialCharge"
    # Nan data
    'BCUT2D_MWHI',
    'BCUT2D_MWLOW',
    'BCUT2D_CHGHI',
    'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI',
    'BCUT2D_MRLOW',

    # Constant data
    'NumRadicalElectrons',
    'SMR_VSA8',
    'SlogP_VSA9',
    'fr_barbitur',
    'fr_benzodiazepine',
    'fr_dihydropyridine',
    'fr_epoxide',
    'fr_isothiocyan',
    'fr_lactam',
    'fr_nitroso',
    'fr_prisulfonamd',
    'fr_thiocyan',

    # High correlated data >0.95
    'MaxEStateIndex',
    'HeavyAtomMolWt',
    'ExactMolWt',
    'NumValenceElectrons',
    'Chi0',
    'Chi0n',
    'Chi0v',
    'Chi1',
    'Chi1n',
    'Chi1v',
    'Chi2n',
    'Kappa1',
    'LabuteASA',
    'HeavyAtomCount',
    'MolMR',
    'Chi3n',
    'BertzCT',
    'Chi2v',
    'Chi4n',
    'HallKierAlpha',
    'Chi3v',
    'Chi4v',
    'MinAbsPartialCharge',
    'MinPartialCharge',
    'MaxAbsPartialCharge',
    'FpDensityMorgan2',
    'FpDensityMorgan3',
    'Phi',
    'Kappa3',
    'fr_nitrile',
    'SlogP_VSA6',
    'NumAromaticCarbocycles',
    'NumAromaticRings',
    'fr_benzene',
    'VSA_EState6',
    'NOCount',
    'fr_C_O',
    'fr_C_O_noCOO',
    'NumHDonors',
    'fr_amide',
    'fr_Nhpyrrole',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    'fr_COO2',
    'fr_halogen',
    'fr_diazo',
    'fr_nitro_arom',
    'fr_phos_ester',
    #low shap
    'NumBridgeheadAtoms', 'NumSaturatedCarbocycles', 'NumSaturatedRings',
    'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA3', 'PEOE_VSA4',
    'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_Ar_COO', 'fr_Ar_NH',
    'fr_COO', 'fr_Imine', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
    'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_amidine', 'fr_azo', 'fr_furan',
    'fr_imidazole', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy',
    'fr_morpholine',  'fr_nitro_arom_nonortho', 'fr_piperdine',
    'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
    'fr_sulfone', 'fr_thiazole', 'fr_urea'
]

def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    counts = {
        'num_C': 0, 'num_c': 0, 'num_O': 0, 'num_N': 0, 'num_F': 0, 'num_Cl': 0,
        'num_positive_ions': 0, 'num_negative_ions': 0
    }
    if mol is None:
        return counts

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()

        if symbol == 'C':
            if atom.GetIsAromatic():
                counts['num_c'] += 1
            else:
                counts['num_C'] += 1
        elif symbol == 'Cl':
            counts['num_Cl'] += 1
        elif symbol in ['O', 'N', 'F']:
            counts[f'num_{symbol}'] += 1

        if charge > 0:
            counts['num_positive_ions'] += 1
        elif charge < 0:
            counts['num_negative_ions'] += 1

    return counts

def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]

def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))

def preprocessing(df):
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES'].to_list()]

    # 그래프 특성 추출
    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    for smile in df['SMILES']:
         compute_graph_features(smile, graph_feats)

    # 원자 개수 계산
    atom_counts = [count_atoms(smi) for smi in df['SMILES']]
    atom_df = pd.DataFrame(atom_counts)
    # 결합
    result = pd.concat(
        [
            pd.DataFrame(descriptors, columns=desc_names),
            pd.DataFrame(graph_feats),
            atom_df
        ],
        axis=1
    )
    # 후처리 및 파생변수
    result = result.replace([-np.inf, np.inf], np.nan)
    eps = 1e-6

    result["NumAromaticHeterocycles_div_NumHeteroatoms"]=result["NumAromaticHeterocycles"]/(result["NumHeteroatoms"]+eps)
    result["fr_unbrch_alkane_div_MolWt "]=result["fr_unbrch_alkane"]/(result["MolWt"]+eps)
    result["PEOE_VSA14_div_graph_diameter"] = result["PEOE_VSA14"] / (result["graph_diameter"] + eps)
    result["BalabanJ_mul_TPSA"] = result["BalabanJ"] * result["TPSA"]
    result["qed_mul_SMR_VSA5"] = result["qed"] * result["SMR_VSA5"]
    result["VSA_EState7_div_MolWt"] = result["VSA_EState7"] / (result["MolWt"] + eps)
    result["SMR_VSA10_div_MolWt"] = result["SMR_VSA10"] / (result["MolWt"] + eps)
    result["SlogP_VSA12_div_MolWt"] = result["SlogP_VSA12"] / (result["MolWt"] + eps)
    result["SMR_VSA10_div_fr_unbrch_alkane"] = result["SMR_VSA10"] / (result["fr_unbrch_alkane"] + eps)
    result["qed_mul_TPSA"] = result["qed"] * result["TPSA"]
    result["PEOE_VSA14_div_fr_unbrch_alkane"] = result["PEOE_VSA14"] / (result["fr_unbrch_alkane"] + eps)
    result["PEOE_VSA14_mul_AvgIpc"] = result["PEOE_VSA14"] * result["AvgIpc"]
    result["SMR_VSA5_div_MolWt"] = result["SMR_VSA5"] / (result["MolWt"] + eps)
    result["PEOE_VSA14_div_SlogP_VSA7"] = result["PEOE_VSA14"] / (result["SlogP_VSA7"] + eps)
    result["VSA_EState7_div_SPS"] = result["VSA_EState7"] / (result["SPS"] + eps)
    result["SlogP_VSA5_mul_FpDensityMorgan1"] = result["SlogP_VSA5"] * result["FpDensityMorgan1"]
    result["VSA_EState8_div_PEOE_VSA5"] = result["VSA_EState8"] / (result["PEOE_VSA5"] + eps)
    result["ion_ratio"] = result["num_positive_ions"] / (result["num_negative_ions"] + eps)
    result["net_ion_charge"] = result["num_positive_ions"] - result["num_negative_ions"]
    result["ion_density"] = (result["num_positive_ions"] + result["num_negative_ions"]) / (result["MolWt"] + eps)

    result['SMR_VSA5_div_MolWt_div_fr_nitro'] = (
        result['SMR_VSA5'] / (result['MolWt'] + eps)
    ) / (result['fr_nitro'] + eps)
    result['fr_unbrch_alkane_div_MolWt_div_EState_VSA11'] = (
        result['fr_unbrch_alkane'] / (result['MolWt'] + eps)
    ) / (result['EState_VSA11'] + eps)
    result['PEOE_VSA14_div_graph_diameter_div_BalabanJ'] = (
        result['PEOE_VSA14'] / (result['graph_diameter'] + eps)
    ) / (result['BalabanJ'] + eps)
    result['VSA_EState7_div_SPS_div_PEOE_VSA14'] = (
        result['VSA_EState7'] / (result['SPS'] + eps)
    ) / (result['PEOE_VSA14'] + eps)

    result.columns = ["rd_" + col for col in result.columns]
    return result

#tg = pd.concat([tg, preprocessing(tg)], axis=1)
#ffv = pd.concat([ffv, preprocessing(ffv)], axis=1)
#tc = pd.concat([tc, preprocessing(tc)], axis=1)
#density = pd.concat([density, preprocessing(density)], axis=1)
#rg = pd.concat([rg, preprocessing(rg)], axis=1)
#desc_test = pd.concat([desc_test, preprocessing(test)], axis=1)


def compute_maccskeys_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMACCSKeysFingerprint(mol)

def compute_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

def compute_rdkit_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.RDKFingerprint(mol)

def compute_atompair_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)


new_data = {}
for target, df in zip(['Tg', 'FFV', 'Tc', 'Density', 'Rg'], [tg, ffv, tc, density, rg]):
    # make finger print
    # Maccs key
    print(f"{target} Maccs Key...")
    maccs = [np.array(compute_maccskeys_fingerprint(smi), int) for smi in df['SMILES'].to_list()]
    maccs_df = pd.DataFrame(maccs, columns=[f'MACCS_{i}' for i in range(len(maccs[0]))])

    # Morgan Fingerprint
    print(f"{target} Morgan Fingerprint...")
    morgan = [np.array(compute_morgan_fingerprint(smi), int) for smi in df["SMILES"].to_list()]
    morgan_df = pd.DataFrame(morgan, columns=[f'Morgan_{i}' for i in range(len(morgan[0]))])

    # RDkit
    print(f"{target} RDkit Fingerprint...")
    rdkf = [np.array(list(compute_rdkit_fingerprint(smi).ToBitString()), int) for smi in df['SMILES'].to_list()]
    rdkf_df = pd.DataFrame(rdkf, columns=[f'RDKF_{i}' for i in range(len(rdkf[0]))])
    
    # AtomPair Fingerprint
    print(f"{target} AtomPair  Fingerprint...")
    ap = [np.array(compute_atompair_fingerprint(smi), int) for smi in df['SMILES'].to_list()]
    ap_df = pd.DataFrame(ap, columns=[f'AP_{i}' for i in range(len(ap[0]))])

    # Concat All Fingerprint
    new_data[target] = pd.concat([df, maccs_df, morgan_df, rdkf_df, ap_df], axis=1)
    new_data[target].drop(columns=[col for col in new_data[target].columns if new_data[target][col].nunique() == 1],axis=1,inplace=True)


object_cols = {
    "Tg":[],
    "FFV":[],
    "Tc":[],
    "Density":[],
    "Rg":[]
}

for tgt in ["Tg", "FFV", "Tc", "Density", "Rg"]:
    for col in new_data[tgt].columns:
        if new_data[tgt][col].dtype == 'object':
            object_cols[tgt].append(col)

    new_data[tgt][object_cols[tgt]] = new_data[tgt][object_cols[tgt]].apply(
        pd.to_numeric, errors="coerce"
    )


tg = new_data['Tg']
rg = new_data['Rg']
ffv = new_data['FFV']
tc = new_data['Tc']
density  = new_data['Density']


print("=== Add Fingerprint Dataset Shape ===")
for target, data in zip(['Tg', 'FFV', 'Tc', 'Density', 'Rg'], [tg, ffv, tc, density, rg]):
    print(f"Target: {target}, Data Length: {data.shape}")


# make finger print
# Maccs key
maccs = [np.array(compute_maccskeys_fingerprint(smi), int) for smi in test.SMILES.to_list()]
maccs_df = pd.DataFrame(maccs, columns=[f'MACCS_{i}' for i in range(len(maccs[0]))])

# Morgan Fingerprint
morgan = [np.array(compute_morgan_fingerprint(smi), int) for smi in test.SMILES.to_list()]
morgan_df = pd.DataFrame(morgan, columns=[f'Morgan_{i}' for i in range(len(morgan[0]))])

# RDkit
rdkf = [np.array(list(compute_rdkit_fingerprint(smi).ToBitString()), int) for smi in test.SMILES.to_list()]
rdkf_df = pd.DataFrame(rdkf, columns=[f'RDKF_{i}' for i in range(len(rdkf[0]))])

# AtomPair Fingerprint
ap = [np.array(compute_atompair_fingerprint(smi), int) for smi in test.SMILES.to_list()]
ap_df = pd.DataFrame(ap, columns=[f'AP_{i}' for i in range(len(ap[0]))])

# Concat All Fingerprint
desc_test = pd.concat([desc_test, maccs_df, morgan_df, rdkf_df, ap_df], axis=1)


desc_obj_cols = []

for col in desc_test.columns:
    if desc_test[col].dtype == 'object':
        desc_obj_cols.append(col)

desc_test.drop(columns=[col for col in desc_test.columns if desc_test[col].nunique() == 1],axis=1,inplace=True)
desc_test[desc_obj_cols] = desc_test[desc_obj_cols].apply(
        pd.to_numeric, errors="coerce"
    )
desc_test.dropna(axis=1, how='all', inplace=True)


def model(train_d,test_d,model,target,seed,submission=False):
    # We divide the data into training and validation sets for model evaluation
    train_cols = set(train_d.columns) - {target}
    test_cols = set(test_d.columns)
   # Intersect the feature columns
    common_cols = list(train_cols & test_cols)
    X=train_d[common_cols].copy()
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model(
        random_seed=seed
    )
    if submission==False:
       Model.fit(X_train,y_train)
       print("Fitting Done...")
       y_pred=Model.predict(X_test)
       return mean_absolute_error(y_pred,y_test) # We assess our model performance using MAE metric
        
    if submission==True:
       Model.fit(X, y, verbose=False)
       print("Fitting Done...")
       submission=Model.predict(test_d[common_cols].copy())
       return submission


from sklearn.model_selection import KFold

def model_cv(train_d, test_d, model_cls, target, n_splits=5, seed=23):
    train_cols = set(train_d.columns) - {target}
    test_cols = set(test_d.columns)
    common_cols = list(train_cols & test_cols)

    X = train_d[common_cols].copy()
    y = train_d[target].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(test_d), n_splits))

    for i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        Model = model_cls()
        Model.fit(X_train, y_train, verbose=False)

        oof_preds[val_idx] = Model.predict(X_val)
        test_preds[:, i] = Model.predict(test_d[common_cols].copy())

        print(f"Fold {i+1} done.")
        

    print(f"Target {target} MAE is {mean_absolute_error(y, oof_preds)}")
    return np.mean(test_preds, axis=1)


"""
xgb_sub={'id':ID,'Tg':model(tg,desc_test,XGBRegressor,'Tg',submission=True),
     'FFV':model(ffv,desc_test,XGBRegressor,'FFV',submission=True),
     'Tc':model(tc,desc_test,XGBRegressor,'Tc',submission=True),
     'Density':model(density,desc_test,XGBRegressor,'Density',submission=True),
     'Rg':model(rg,desc_test,XGBRegressor,'Rg',submission=True)}

xgb_sub_df = pd.DataFrame(xgb_sub)
xgb_sub_df = xgb_sub_df.set_index('id')
"""


"""
lgbm_sub={'id':ID,'Tg':model(tg,desc_test,LGBMRegressor,'Tg',submission=True),
     'FFV':model(ffv,desc_test,LGBMRegressor,'FFV',submission=True),
     'Tc':model(tc,desc_test,LGBMRegressor,'Tc',submission=True),
     'Density':model(density,desc_test,LGBMRegressor,'Density',submission=True),
     'Rg':model(rg,desc_test,LGBMRegressor,'Rg',submission=True)}

lgbm_sub_df = pd.DataFrame(lgbm_sub)
lgbm_sub_df = lgbm_sub_df.set_index('id')
"""


cat_sub={'id':ID,'Tg':model(tg,desc_test,CatBoostRegressor,'Tg', 23, submission=True),
     'FFV':model(ffv,desc_test,CatBoostRegressor,'FFV', 23, submission=True),
     'Tc':model(tc,desc_test,CatBoostRegressor,'Tc', 23, submission=True),
     'Density':model(density,desc_test,CatBoostRegressor,'Density', 23, submission=True),
     'Rg':model(rg,desc_test,CatBoostRegressor,'Rg', 23, submission=True)}


cat_sub_df = pd.DataFrame(cat_sub)
cat_sub_df = cat_sub_df.set_index('id')


print(bert_sub_df)
print(grea_sub_df)
print(gnn_sub_df)
#print(extra_sub_df)
#print(xgb_sub_df)
#print(lgbm_sub_df)
print(cat_sub_df)
#print(pfn_sub_df)
#print(hvec_sub_df)


#ensemble_df = 0.43*bert_sub_df + 0.12*grea_sub_df + 0.12*gnn_sub_df + 0.06*pfn_sub_df + 0.06*extra_sub_df + 0.06*xgb_sub_df + 0.06*lgbm_sub_df + 0.09*cat_sub_df
#ensemble_df = 0.24*bert_sub_df + 0.165*grea_sub_df + 0.165*gnn_sub_df + 0.43*cat_sub_df
ensemble_df = 0.33*bert_sub_df + 0.12*grea_sub_df + 0.12*gnn_sub_df + 0.43*cat_sub_df
ensemble_df.to_csv("submission.csv")


ensemble_df

