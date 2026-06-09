!pip install --no-index --find-links /kaggle/input/notebookf01fb1055d rdkit mordred numpy==1.26.4


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from mordred import Calculator, descriptors
from rdkit.Chem.Lipinski import RotatableBondSmarts
# from mordred import Calculator, descriptors
import networkx as nx
import time
import csv
import os
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras import losses, metrics, layers, models, callbacks


def smiles_to_descriptors(smiles):
    """
    Convert a SMILES string to Mordred descriptors with error handling.
    """
    calc = Calculator(descriptors, ignore_3D=True)
    descriptor_list = Descriptors._descList
    descriptor_names_modred = [str(d) for d in calc.descriptors]
    try:
        smiles = smiles.replace("*", "[H]")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        # RDKit descriptors
        desc_values_rdkit = {}
        for name, func in descriptor_list:
            try:
                desc_values_rdkit[name] = func(mol)
            except Exception:
                desc_values_rdkit[name] = None

        # Mordred descriptors
        desc_modred = calc(mol)                # DescriptorResult object
        desc_dict_modred = desc_modred.asdict()  # ✅ convert to plain dict

        # Merge with RDKit
        desc_dict_modred.update(desc_values_rdkit)

        return desc_dict_modred

    except Exception as e:
        print("DIKKAt")
        common_descriptors = set(descriptor_names_modred) or set(descriptor_list)
        descriptor_dict = {k: None for k in common_descriptors}
        return descriptor_dict

def build_descriptor_dataset(df):
    """
    Takes dataframe with `smiles` and `property`,
    returns dataframe with 150 descriptors added.
    Processes SMILES individually with progress bars that reset every 100 molecules.
    """
    results = []
    batch_size = 100
    total = len(df)
    
    # Initialize first progress bar
    current_batch_size = min(batch_size, total)
    pbar = tqdm(total=current_batch_size, desc="Processing SMILES batch")
    
    for i, row in enumerate(df.itertuples(), 1):
        # Process individual SMILES
        desc_dict = smiles_to_descriptors(row.SMILES)
        result = {"id": row.id, "smiles": row.SMILES}
        result.update(desc_dict)
        results.append(result)
        
        # Update progress bar
        pbar.update(1)
        
        # Check if we've completed a batch of 100 or reached the end
        if i % batch_size == 0 or i == total:
            pbar.close()
            print(f"{i} smiles done")
            
            # Start new progress bar if there are more SMILES to process
            if i < total:
                remaining = total - i
                current_batch_size = min(batch_size, remaining)
                pbar = tqdm(total=current_batch_size, desc="Processing SMILES batch")
    
    return pd.DataFrame(results)


SMILES_Tc = pd.read_csv("/kaggle/input/neurips-modredrdkit-dataset/Tc_finale_finale.csv", index_col = 0)
SMILES_Tg = pd.read_csv("/kaggle/input/neurips-modredrdkit-dataset/Tg_finale_finale.csv", index_col = 0)
SMILES_Density = pd.read_csv("/kaggle/input/neurips-modredrdkit-dataset/Density_finale_finale.csv", index_col = 0)
SMILES_FFV = pd.read_csv("/kaggle/input/neurips-modredrdkit-dataset/FFV_finale_finale.csv", index_col = 0)
SMILES_Rg = pd.read_csv("/kaggle/input/neurips-modredrdkit-dataset/Rg_finale_finale.csv", index_col = 0)
SMILES_Test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

print("All Datasets Loaded !")


SMILES_Test_Final = build_descriptor_dataset(SMILES_Test)


!pip install --no-index --find-links /kaggle/input/notebook21c6633f93 numpy
import numpy as np


descriptors = ['MINsGeH3', 'GATS6are', 'MATS7Z', 'AATS7d', 'MINdsssP', 'AATSC8Z', 'MINdS', 'MAXssssC', 'MAXssBH', 'AATS7v', 'AATSC4Z', 'MATS5dv', 'MINsOH', 'GATS4dv', 'VR2_Dt', 'MATS8s', 'MINddssS', 'MINsssSnH', 'MATS7s', 'MDEN-22', 'SpMAD_Dt', 'MINssS', 'MATS8dv', 'VE2_Dt', 'Vabc', 'GATS2d', 'MAXaaO', 'AATS5i', 'MINsssssP', 'MINaaO', 'MAXaaSe', 'MAXaaNH', 'MAXaasC', 'AXp-3d', 'MINddssSe', 'AXp-2d', 'GATS4s', 'AATSC4d', 'AATSC5p', 'MATS6i', 'AATS7s', 'AATS8p', 'DetourIndex', 'GATS6p', 'AATSC8se', 'GATS4d', 'MATS7i', 'MAXssCH2', 'AXp-6d', 'AATS4i', 'MATS4Z', 'MAXaaaC', 'MINddC', 'MATS6Z', 'GATS7c', 'MAXsssSnH', 'GATS5m', 'MDEC-14', 'GATS7are', 'AATSC7Z', 'MAXssNH2', 'MDEO-22', 'MATS4pe', 'GATS5s', 'MAXdsssP', 'MINssssPb', 'MINssSe', 'GATS5dv', 'VR3_Dt', 'AATSC4c', 'MINdNH', 'MINsSiH3', 'AATS4m', 'AATSC6dv', 'AATSC7are', 'AATSC6se', 'AATSC4pe', 'AATSC7p', 'MATS7pe', 'GATS8Z', 'AXp-7dv', 'GATS6i', 'GATS8v', 'AATS8d', 'MAXssPbH2', 'AATSC6are', 'AATS5s', 'AATSC7se', 'MAXdS', 'MINssPbH2', 'AATS8s', 'MAXssSe', 'MINsCH3', 'MATS5v', 'MDEN-13', 'AATSC5c', 'AATS6d', 'Xp-0dv', 'GATS1d', 'AATS8dv', 'GATS8m', 'MAXsCH3', 'MINssNH2', 'AATSC8v', 'AATSC6d', 'AATSC7i', 'AATS8pe', 'MINsI', 'MINsssPbH', 'MATS4are', 'MATS6pe', 'MAXtN', 'AATS6are', 'MINsSH', 'MINdssSe', 'AXp-4dv', 'AXp-5d', 'AATSC6c', 'AATSC5s', 'SpMax_Dt', 'MDEO-11', 'MINssssB', 'MATS7p', 'AATS6Z', 'AATSC7d', 'GATS6c', 'AATSC4v', 'MINdssS', 'MINsPbH3', 'MAXdsCH', 'MDEN-33', 'AATSC7v', 'MAXsssN', 'AATSC8i', 'MATS4dv', 'MAXssssBe', 'MAXsssdAs', 'MAXsNH3', 'MATS3d', 'MAXsAsH2', 'MAXsssssAs', 'GATS6v', 'MAXssssPb', 'MDEO-12', 'MINdssC', 'MINsssAs', 'MATS8pe', 'MAXsssssP', 'MINdsCH', 'MAXsSiH3', 'GATS8pe', 'AATS4v', 'GATS4se', 'MAXddsN', 'MAXsssPbH', 'AATS8v', 'MATS7dv', 'MATS4d', 'AATS4Z', 'MINdsN', 'AATS8are', 'MAXtsC', 'MINaasC', 'GATS7i', 'MATS5Z', 'AATSC7s', 'AATS6p', 'AATSC4se', 'MINssssSn', 'MATS8i', 'MATS6s', 'GATS4c', 'AXp-4d', 'MDEC-13', 'SpDiam_Dt', 'MAXsI', 'AATSC5v', 'AATSC4are', 'GATS5p', 'GATS7v', 'MATS7m', 'MATS8m', 'MATS6dv', 'AATSC7pe', 'GATS5v', 'LogEE_Dt', 'AATSC5are', 'AATSC6v', 'MATS4c', 'GATS5d', 'SM1_Dt', 'MATS8c', 'GATS8c', 'MATS4p', 'AATS7pe', 'AATS8i', 'MATS7c', 'MINsBr', 'MINssNH', 'GATS7se', 'MATS8Z', 'MAXsGeH3', 'MINsssdAs', 'AATS6pe', 'MAXsssGeH', 'MAXssssN', 'AATSC6i', 'MATS8d', 'MAXsssSiH', 'MATS1d', 'GATS8dv', 'MINssBH', 'MAXddssS', 'MAXsLi', 'AATSC8c', 'AATS5p', 'AATSC8d', 'MAXdO', 'GATS6Z', 'AATS7se', 'GATS4v', 'MATS5m', 'MAXsPbH3', 'AATS5dv', 'AATSC5dv', 'AATSC5d', 'MAXdssC', 'MAXssNH', 'MAXsNH2', 'MINtsC', 'MATS7v', 'SpAD_Dt', 'AATSC4i', 'AATSC8p', 'GATS8i', 'MAXsCl', 'MAXsSnH3', 'AXp-3dv', 'MATS8v', 'AATSC4m', 'MAXdssSe', 'MAXssssB', 'AATS7p', 'MAXsBr', 'GATS7d', 'GATS4are', 'AATS7i', 'MAXssssGe', 'MATS4m', 'GATS8d', 'AATS8m', 'AATSC5m', 'GATS4p', 'VE1_Dt', 'MAXdssS', 'ABCGG', 'MINsSeH', 'MAXssssSi', 'GATS5se', 'MINaaNH', 'GATS6se', 'MINssPH', 'GATS6d', 'MDEN-12', 'MINsLi', 'GATS7dv', 'GATS4Z', 'ABC', 'VR1_Dt', 'MAXsssP', 'AATS4d', 'MINdO', 'MINaaCH', 'MAXdsN', 'GATS8are', 'AATSC8dv', 'MATS6m', 'GATS8p', 'MATS7se', 'MAXdCH2', 'MINsNH3', 'AATS5v', 'MATS8se', 'MAXaaN', 'AATS5Z', 'AATS6i', 'MATS5d', 'MATS6d', 'MATS8are', 'MATS6are', 'MINaaS', 'AATS4pe', 'MATS4s', 'MAXddssSe', 'MINssCH2', 'MINssssN', 'MINssssGe', 'MINtCH', 'MDEC-22', 'MINssSiH2', 'AATS6dv', 'AXp-0dv', 'MAXddC', 'MAXdSe', 'MINsssN', 'AATS4are', 'AATS8Z', 'MATS6p', 'MAXaasN', 'AATS4p', 'MINaaN', 'MINsssssAs', 'MINssssSi', 'GATS5i', 'AATS6m', 'GATS5c', 'AATSC4dv', 'MDEN-11', 'MINssssBe', 'MAXsssCH', 'GATS7m', 'AATSC8pe', 'MAXsSeH', 'MINssAsH', 'GATS7pe', 'MINddsN', 'MAXsSH', 'AATSC4p', 'MATS5p', 'MAXsssNH', 'MATS4v', 'AATS6v', 'MATS7d', 'AATS4se', 'GATS6pe', 'MINsNH2', 'MATS5pe', 'MDEC-12', 'GATS7Z', 'MDEC-33', 'MAXsF', 'GATS4m', 'MAXaaS', 'MINtN', 'AATS8se', 'GATS5are', 'VE3_Dt', 'MATS6c', 'MINsPH2', 'MAXsssB', 'GATS4i', 'MATS4i', 'MAXtCH', 'AXp-2dv', 'MINssBe', 'MAXssS', 'MAXssBe', 'MATS6se', 'AATSC5i', 'AATS4s', 'GATS6dv', 'AATSC6Z', 'MINsSnH3', 'MDEC-11', 'MINsssB', 'MDEC-34', 'AATSC6s', 'AATS5m', 'GATS6s', 'AXp-5dv', 'AATSC5Z', 'AATS5pe', 'AATSC6p', 'MINsAsH2', 'MATS5c', 'AATS6se', 'AATS7dv', 'AATS4dv', 'MAXssO', 'MINdCH2', 'MINdSe', 'AATS6s', 'MDEC-24', 'AATSC4s', 'GATS8se', 'MINsCl', 'MATS5are', 'MAXssAsH', 'AATSC7m', 'MAXssssSn', 'GATS6m', 'MAXaaCH', 'GATS8s', 'MINaaaC', 'AATS5se', 'MINssssC', 'AATSC6m', 'AATSC6pe', 'MINssO', 'MAXsOH', 'MAXssPH', 'MINssSnH2', 'AATSC7c', 'AATSC5se', 'AXp-6dv', 'AATS5are', 'MATS7are', 'MINsssP', 'AXp-7d', 'AATS7Z', 'AATS5d', 'MINsssCH', 'GATS4pe', 'MDEN-23', 'MATS2d', 'MATS5s', 'MINaaSe', 'AATS7m', 'GATS3d', 'SpAbs_Dt', 'MINsssGeH', 'GATS7p', 'MATS4se', 'Kier2', 'MINsssNH', 'MAXssSiH2', 'MATS5i', 'GATS5pe', 'AATSC8m', 'AATSC8are', 'MATS6v', 'MINaasN', 'MINsssSiH', 'AATSC8s', 'MAXssGeH2', 'AATSC5pe', 'MDEC-23', 'MAXssSnH2', 'MAXsssAs', 'MAXsPH2', 'AATS7are', 'AATSC7dv', 'MATS5se', 'MINsF', 'Kier3', 'GATS7s', 'MDEC-44', 'MINssGeH2', 'MATS8p', 'GATS5Z', 'MAXdNH']
for i in SMILES_Test_Final.columns:
    if i in descriptors:
        SMILES_Test_Final = SMILES_Test_Final.drop([i], axis = 1)


import re
SMILES_Tc.columns = [re.sub(r'[^A-Za-z0-9_]', '', c) for c in SMILES_Tc.columns]
SMILES_Tg.columns = [re.sub(r'[^A-Za-z0-9_]', '', c) for c in SMILES_Tg.columns]
SMILES_Rg.columns = [re.sub(r'[^A-Za-z0-9_]', '', c) for c in SMILES_Rg.columns]
SMILES_FFV.columns = [re.sub(r'[^A-Za-z0-9_]', '', c) for c in SMILES_FFV.columns]
SMILES_Density.columns = [re.sub(r'[^A-Za-z0-9_]', '', c) for c in SMILES_Density.columns]
SMILES_Test_Final.columns = [re.sub(r'[^A-Za-z0-9_]', '', c) for c in SMILES_Test_Final.columns]


# non_numeric_indices = SMILES_Density[~pd.to_numeric(SMILES_Density['Density'], errors='coerce').notnull()].index
# print(SMILES_Density.loc[non_numeric_indices, 'Density'])
# SMILES_Density = SMILES_Density.drop(index=non_numeric_indices).reset_index(drop=True)


# cols_to_fix = ['Density']

# for col in cols_to_fix:
#     SMILES_Density[col] = pd.to_numeric(SMILES_Density[col], errors='coerce')


columns_1 = ['nAcid', 'nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A',
       'LogEE_A', 'VE1_A', 'VE2_A', 'VE3_A', 'VR1_A', 'VR2_A', 'VR3_A',
       'nAromAtom', 'nAromBond', 'nAtom', 'nHeavyAtom', 'nSpiro',
       'nBridgehead', 'nHetero', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP',
       'nF', 'nCl', 'nBr', 'nI', 'nX', 'ATS0dv', 'ATS1dv', 'ATS2dv', 'ATS3dv',
       'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS8dv', 'ATS0d', 'ATS1d',
       'ATS2d', 'ATS3d', 'ATS4d', 'ATS5d', 'ATS6d', 'ATS7d', 'ATS8d', 'ATS0s',
       'ATS1s', 'ATS2s', 'ATS3s', 'ATS4s', 'ATS5s', 'ATS6s', 'ATS7s', 'ATS8s',
       'ATS0Z', 'ATS1Z', 'ATS2Z', 'ATS3Z', 'ATS4Z', 'ATS5Z', 'ATS6Z', 'ATS7Z',
       'ATS8Z', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m',
       'ATS7m', 'ATS8m', 'ATS0v', 'ATS1v', 'ATS2v', 'ATS3v', 'ATS4v', 'ATS5v',
       'ATS6v', 'ATS7v', 'ATS8v', 'ATS0se', 'ATS1se', 'ATS2se', 'ATS3se',
       'ATS4se', 'ATS5se', 'ATS6se', 'ATS7se', 'ATS8se', 'ATS0pe', 'ATS1pe',
       'ATS2pe', 'ATS3pe', 'property']
columns_2 = ['ATS4pe', 'ATS5pe', 'ATS6pe', 'ATS7pe', 'ATS8pe', 'ATS0are', 'ATS1are',
       'ATS2are', 'ATS3are', 'ATS4are', 'ATS5are', 'ATS6are', 'ATS7are',
       'ATS8are', 'ATS0p', 'ATS1p', 'ATS2p', 'ATS3p', 'ATS4p', 'ATS5p',
       'ATS6p', 'ATS7p', 'ATS8p', 'ATS0i', 'ATS1i', 'ATS2i', 'ATS3i', 'ATS4i',
       'ATS5i', 'ATS6i', 'ATS7i', 'ATS8i', 'AATS0dv', 'AATS1dv', 'AATS2dv',
       'AATS3dv', 'AATS0d', 'AATS1d', 'AATS2d', 'AATS3d', 'AATS0s', 'AATS1s',
       'AATS2s', 'AATS3s', 'AATS0Z', 'AATS1Z', 'AATS2Z', 'AATS3Z', 'AATS0m',
       'AATS1m', 'AATS2m', 'AATS3m', 'AATS0v', 'AATS1v', 'AATS2v', 'AATS3v',
       'AATS0se', 'AATS1se', 'AATS2se', 'AATS3se', 'AATS0pe', 'AATS1pe',
       'AATS2pe', 'AATS3pe', 'AATS0are', 'AATS1are', 'AATS2are', 'AATS3are',
       'AATS0p', 'AATS1p', 'AATS2p', 'AATS3p', 'AATS0i', 'AATS1i', 'AATS2i',
       'AATS3i', 'ATSC0c', 'ATSC1c', 'ATSC2c', 'ATSC3c', 'ATSC4c', 'ATSC5c',
       'ATSC6c', 'ATSC7c', 'ATSC8c', 'ATSC0dv', 'ATSC1dv', 'ATSC2dv',
       'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv',
       'ATSC0d', 'ATSC1d', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d', 'property']
columns_3 = ['ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC0s', 'ATSC1s', 'ATSC2s', 'ATSC3s',
       'ATSC4s', 'ATSC5s', 'ATSC6s', 'ATSC7s', 'ATSC8s', 'ATSC0Z', 'ATSC1Z',
       'ATSC2Z', 'ATSC3Z', 'ATSC4Z', 'ATSC5Z', 'ATSC6Z', 'ATSC7Z', 'ATSC8Z',
       'ATSC0m', 'ATSC1m', 'ATSC2m', 'ATSC3m', 'ATSC4m', 'ATSC5m', 'ATSC6m',
       'ATSC7m', 'ATSC8m', 'ATSC0v', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v',
       'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', 'ATSC0se', 'ATSC1se', 'ATSC2se',
       'ATSC3se', 'ATSC4se', 'ATSC5se', 'ATSC6se', 'ATSC7se', 'ATSC8se',
       'ATSC0pe', 'ATSC1pe', 'ATSC2pe', 'ATSC3pe', 'ATSC4pe', 'ATSC5pe',
       'ATSC6pe', 'ATSC7pe', 'ATSC8pe', 'ATSC0are', 'ATSC1are', 'ATSC2are',
       'ATSC3are', 'ATSC4are', 'ATSC5are', 'ATSC6are', 'ATSC7are', 'ATSC8are',
       'ATSC0p', 'ATSC1p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p',
       'ATSC7p', 'ATSC8p', 'ATSC0i', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i',
       'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i', 'AATSC0c', 'AATSC1c', 'AATSC2c',
       'AATSC3c', 'AATSC0dv', 'AATSC1dv', 'AATSC2dv', 'AATSC3dv', 'AATSC0d',
       'AATSC1d', 'AATSC2d', 'AATSC3d', 'AATSC0s', 'AATSC1s', 'AATSC2s',
       'AATSC3s', 'property']
columns_4 = ['AATSC0Z', 'AATSC1Z', 'AATSC2Z', 'AATSC3Z', 'AATSC0m', 'AATSC1m',
       'AATSC2m', 'AATSC3m', 'AATSC0v', 'AATSC1v', 'AATSC2v', 'AATSC3v',
       'AATSC0se', 'AATSC1se', 'AATSC2se', 'AATSC3se', 'AATSC0pe', 'AATSC1pe',
       'AATSC2pe', 'AATSC3pe', 'AATSC0are', 'AATSC1are', 'AATSC2are',
       'AATSC3are', 'AATSC0p', 'AATSC1p', 'AATSC2p', 'AATSC3p', 'AATSC0i',
       'AATSC1i', 'AATSC2i', 'AATSC3i', 'MATS1c', 'MATS2c', 'MATS3c',
       'MATS1dv', 'MATS2dv', 'MATS3dv', 'MATS1s', 'MATS2s', 'MATS3s', 'MATS1Z',
       'MATS2Z', 'MATS3Z', 'MATS1m', 'MATS2m', 'MATS3m', 'MATS1v', 'MATS2v',
       'MATS3v', 'MATS1se', 'MATS2se', 'MATS3se', 'MATS1pe', 'MATS2pe',
       'MATS3pe', 'MATS1are', 'MATS2are', 'MATS3are', 'MATS1p', 'MATS2p',
       'MATS3p', 'MATS1i', 'MATS2i', 'MATS3i', 'GATS1c', 'GATS2c', 'GATS3c',
       'GATS1dv', 'GATS2dv', 'GATS3dv', 'GATS1s', 'GATS2s', 'GATS3s', 'GATS1Z',
       'GATS2Z', 'GATS3Z', 'GATS1m', 'GATS2m', 'GATS3m', 'GATS1v', 'GATS2v',
       'GATS3v', 'GATS1se', 'GATS2se', 'GATS3se', 'GATS1pe', 'GATS2pe',
       'GATS3pe', 'GATS1are', 'GATS2are', 'GATS3are', 'GATS1p', 'GATS2p',
       'GATS3p', 'GATS1i', 'GATS2i', 'GATS3i', 'BCUTc1h', 'BCUTc1l', 'property']
columns_5 = ['BCUTdv1h', 'BCUTdv1l', 'BCUTd1h', 'BCUTd1l', 'BCUTs1h', 'BCUTs1l',
       'BCUTZ1h', 'BCUTZ1l', 'BCUTm1h', 'BCUTm1l', 'BCUTv1h', 'BCUTv1l',
       'BCUTse1h', 'BCUTse1l', 'BCUTpe1h', 'BCUTpe1l', 'BCUTare1h',
       'BCUTare1l', 'BCUTp1h', 'BCUTp1l', 'BCUTi1h', 'BCUTi1l', 'BalabanJ',
       'SpAbs_DzZ', 'SpMax_DzZ', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ',
       'LogEE_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ', 'VR1_DzZ',
       'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm',
       'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm', 'VE1_Dzm', 'VE2_Dzm',
       'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm', 'SpAbs_Dzv', 'SpMax_Dzv',
       'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv',
       'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', 'VR2_Dzv', 'VR3_Dzv',
       'SpAbs_Dzse', 'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse',
       'LogEE_Dzse', 'SM1_Dzse', 'VE1_Dzse', 'VE2_Dzse', 'VE3_Dzse',
       'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzpe', 'SpMax_Dzpe',
       'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe',
       'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe',
       'SpAbs_Dzare', 'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare',
       'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare',
       'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'property']
columns_6 = ['VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp',
       'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp',
       'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi',
       'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi',
       'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi', 'VR3_Dzi', 'BertzCT', 'nBonds',
       'nBondsO', 'nBondsS', 'nBondsD', 'nBondsT', 'nBondsA', 'nBondsM',
       'nBondsKS', 'nBondsKD', 'RNCG', 'RPCG', 'C1SP1', 'C2SP1', 'C1SP2',
       'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'HybRatio',
       'FCSP3', 'Xch3d', 'Xch4d', 'Xch5d', 'Xch6d', 'Xch7d', 'Xch3dv',
       'Xch4dv', 'Xch5dv', 'Xch6dv', 'Xch7dv', 'Xc3d', 'Xc4d', 'Xc5d', 'Xc6d',
       'Xc3dv', 'Xc4dv', 'Xc5dv', 'Xc6dv', 'Xpc4d', 'Xpc5d', 'Xpc6d', 'Xpc4dv',
       'Xpc5dv', 'Xpc6dv', 'Xp0d', 'Xp1d', 'Xp2d', 'Xp3d', 'Xp4d', 'Xp5d',
       'Xp6d', 'Xp7d', 'AXp0d', 'AXp1d', 'Xp1dv', 'Xp2dv', 'Xp3dv', 'Xp4dv',
       'Xp5dv', 'Xp6dv', 'Xp7dv', 'AXp1dv', 'SZ', 'Sm', 'Sv', 'Sse', 'Spe',
       'Sare', 'Sp', 'Si', 'property']
columns_7 = ['MZ', 'Mm', 'Mv', 'Mse', 'Mpe', 'Mare', 'Mp', 'Mi', 'SpAbs_D',
       'SpMax_D', 'SpDiam_D', 'SpAD_D', 'SpMAD_D', 'LogEE_D', 'VE1_D', 'VE2_D',
       'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D', 'NsLi', 'NssBe', 'NssssBe', 'NssBH',
       'NsssB', 'NssssB', 'NsCH3', 'NdCH2', 'NssCH2', 'NtCH', 'NdsCH', 'NaaCH',
       'NsssCH', 'NddC', 'NtsC', 'NdssC', 'NaasC', 'NaaaC', 'NssssC', 'NsNH3',
       'NsNH2', 'NssNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NsssNH', 'NdsN',
       'NaaN', 'NsssN', 'NddsN', 'NaasN', 'NssssN', 'NsOH', 'NdO', 'NssO',
       'NaaO', 'NsF', 'NsSiH3', 'NssSiH2', 'NsssSiH', 'NssssSi', 'NsPH2',
       'NssPH', 'NsssP', 'NdsssP', 'NsssssP', 'NsSH', 'NdS', 'NssS', 'NaaS',
       'NdssS', 'NddssS', 'NsCl', 'NsGeH3', 'NssGeH2', 'NsssGeH', 'NssssGe',
       'NsAsH2', 'NssAsH', 'NsssAs', 'NsssdAs', 'NsssssAs', 'NsSeH', 'NdSe',
       'NssSe', 'NaaSe', 'NdssSe', 'NddssSe', 'NsBr', 'NsSnH3', 'NssSnH2',
       'NsssSnH', 'NssssSn', 'NsI', 'NsPbH3', 'NssPbH2', 'NsssPbH', 'NssssPb',
       'SsLi', 'property']
columns_8 = ['SssBe', 'SssssBe', 'SssBH', 'SsssB', 'SssssB', 'SsCH3', 'SdCH2',
       'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SddC', 'StsC', 'SdssC',
       'SaasC', 'SaaaC', 'SssssC', 'SsNH3', 'SsNH2', 'SssNH2', 'SdNH', 'SssNH',
       'SaaNH', 'StN', 'SsssNH', 'SdsN', 'SaaN', 'SsssN', 'SddsN', 'SaasN',
       'SssssN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SsSiH3', 'SssSiH2',
       'SsssSiH', 'SssssSi', 'SsPH2', 'SssPH', 'SsssP', 'SdsssP', 'SsssssP',
       'SsSH', 'SdS', 'SssS', 'SaaS', 'SdssS', 'SddssS', 'SsCl', 'SsGeH3',
       'SssGeH2', 'SsssGeH', 'SssssGe', 'SsAsH2', 'SssAsH', 'SsssAs',
       'SsssdAs', 'SsssssAs', 'SsSeH', 'SdSe', 'SssSe', 'SaaSe', 'SdssSe',
       'SddssSe', 'SsBr', 'SsSnH3', 'SssSnH2', 'SsssSnH', 'SssssSn', 'SsI',
       'SsPbH3', 'SssPbH2', 'SsssPbH', 'SssssPb', 'ECIndex', 'ETA_alpha',
       'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta',
       'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns', 'AETA_beta_ns',
       'ETA_beta_ns_d', 'AETA_beta_ns_d', 'ETA_eta', 'AETA_eta', 'ETA_eta_L',
       'AETA_eta_L', 'ETA_eta_R', 'AETA_eta_R', 'ETA_eta_RL', 'AETA_eta_RL', 'property']
columns_9 = ['ETA_eta_F', 'AETA_eta_F', 'ETA_eta_FL', 'AETA_eta_FL', 'ETA_eta_B',
       'AETA_eta_B', 'ETA_eta_BR', 'AETA_eta_BR', 'ETA_dAlpha_A',
       'ETA_dAlpha_B', 'ETA_epsilon_1', 'ETA_epsilon_2', 'ETA_epsilon_3',
       'ETA_epsilon_4', 'ETA_epsilon_5', 'ETA_dEpsilon_A', 'ETA_dEpsilon_B',
       'ETA_dEpsilon_C', 'ETA_dEpsilon_D', 'ETA_dBeta', 'AETA_dBeta',
       'ETA_psi_1', 'ETA_dPsi_A', 'ETA_dPsi_B', 'fragCpx', 'fMF', 'nHBAcc',
       'nHBDon', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1',
       'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4',
       'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3', 'BIC4', 'BIC5', 'CIC0', 'CIC1',
       'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4',
       'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'Kier1',
       'Lipinski', 'GhoseFilter', 'FilterItLogS', 'VMcGowan', 'LabuteASA',
       'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
       'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10',
       'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2',
       'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8',
       'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'property']
columns_10 = ['SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
       'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1',
       'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
       'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
       'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3',
       'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
       'VSA_EState8', 'VSA_EState9', 'MID', 'AMID', 'MID_h', 'AMID_h', 'MID_C',
       'AMID_C', 'MID_N', 'AMID_N', 'MID_O', 'AMID_O', 'MID_X', 'AMID_X',
       'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10',
       'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7',
       'piPC8', 'piPC9', 'piPC10', 'TpiPC10', 'apol', 'bpol', 'nRing',
       'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring', 'n7Ring', 'n8Ring', 'n9Ring',
       'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n3HRing',
       'n4HRing', 'n5HRing', 'n6HRing', 'n7HRing', 'n8HRing', 'n9HRing',
       'n10HRing', 'n11HRing', 'n12HRing', 'nG12HRing', 'naRing', 'n3aRing',
       'n4aRing', 'n5aRing', 'n6aRing', 'n7aRing', 'n8aRing', 'n9aRing',
       'n10aRing', 'n11aRing', 'n12aRing', 'nG12aRing', 'naHRing', 'property']
columns_11 = ['n3aHRing', 'n4aHRing', 'n5aHRing', 'n6aHRing', 'n7aHRing', 'n8aHRing',
       'n9aHRing', 'n10aHRing', 'n11aHRing', 'n12aHRing', 'nG12aHRing',
       'nARing', 'n3ARing', 'n4ARing', 'n5ARing', 'n6ARing', 'n7ARing',
       'n8ARing', 'n9ARing', 'n10ARing', 'n11ARing', 'n12ARing', 'nG12ARing',
       'nAHRing', 'n3AHRing', 'n4AHRing', 'n5AHRing', 'n6AHRing', 'n7AHRing',
       'n8AHRing', 'n9AHRing', 'n10AHRing', 'n11AHRing', 'n12AHRing',
       'nG12AHRing', 'nFRing', 'n4FRing', 'n5FRing', 'n6FRing', 'n7FRing',
       'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing',
       'nFHRing', 'n4FHRing', 'n5FHRing', 'n6FHRing', 'n7FHRing', 'n8FHRing',
       'n9FHRing', 'n10FHRing', 'n11FHRing', 'n12FHRing', 'nG12FHRing',
       'nFaRing', 'n4FaRing', 'n5FaRing', 'n6FaRing', 'n7FaRing', 'n8FaRing',
       'n9FaRing', 'n10FaRing', 'n11FaRing', 'n12FaRing', 'nG12FaRing',
       'nFaHRing', 'n4FaHRing', 'n5FaHRing', 'n6FaHRing', 'n7FaHRing',
       'n8FaHRing', 'n9FaHRing', 'n10FaHRing', 'n11FaHRing', 'n12FaHRing',
       'nG12FaHRing', 'nFARing', 'n4FARing', 'n5FARing', 'n6FARing',
       'n7FARing', 'n8FARing', 'n9FARing', 'n10FARing', 'n11FARing',
       'n12FARing', 'nG12FARing', 'nFAHRing', 'n4FAHRing', 'n5FAHRing',
       'n6FAHRing', 'n7FAHRing', 'n8FAHRing', 'n9FAHRing', 'n10FAHRing',
       'n11FAHRing', 'n12FAHRing', 'property']
columns_12 = ['nG12FAHRing', 'nRot', 'RotRatio', 'SLogP', 'SMR', 'TopoPSANO',
       'TopoPSA', 'GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7',
       'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6',
       'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT10', 'Diameter', 'Radius',
       'TopoShapeIndex', 'PetitjeanIndex', 'VAdjMat', 'MWC01', 'MWC02',
       'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10',
       'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08',
       'SRW09', 'SRW10', 'TSRW10', 'MW', 'AMW', 'WPath', 'WPol', 'Zagreb1',
       'Zagreb2', 'mZagreb1', 'mZagreb2', 'MaxAbsEStateIndex',
       'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS',
       'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
       'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge',
       'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
       'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW',
       'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'Chi0', 'Chi0n', 'Chi0v',
       'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'property']
columns_13 = ['Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3',
       'PEOE_VSA14', 'SMR_VSA10', 'SlogP_VSA12', 'TPSA', 'EState_VSA11',
       'VSA_EState10', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
       'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
       'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles',
       'NumAromaticHeterocycles', 'NumAromaticRings', 'NumAtomStereoCenters',
       'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
       'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
       'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms',
       'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP',
       'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
       'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
       'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
       'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
       'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',
       'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
       'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
       'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
       'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
       'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole',
       'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
       'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
       'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
       'fr_nitro_arom_nonortho', 'property']
columns_14 = ['fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation',
       'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
       'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
       'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
       'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
       'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'property']


SMILES_FFV = SMILES_FFV.drop(['smiles'], axis = 1)
SMILES_FFV = SMILES_FFV.dropna(subset = SMILES_FFV.columns)

# X_train_FFV = SMILES_FFV.drop('property', axis=1)
# y_train_FFV = SMILES_FFV['property']

# collect all your columns in a list
columns_list = [columns_1, columns_2, columns_3, columns_4, columns_5, columns_6, columns_7, columns_8, columns_9, columns_10, columns_11, columns_12, columns_13, columns_14]

# dictionary to store models, X, y
x_train_FFV = {}
y_train_FFV = {}
models_FFV = {}

for i, cols in enumerate(columns_list, start=1):
    # create dataframe
    df = pd.DataFrame(SMILES_FFV[cols], index=SMILES_FFV.index)

    # assume last column is target, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split into train/test
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # save into dicts
    x_train_FFV[i] = X
    y_train_FFV[i] = y

    # train LightGBM
    model = CatBoostRegressor(random_state=42, verbose = 0)
    model.fit(X, y)

    # store model
    models_FFV[i] = model

print("DONE!")


SMILES_Tg = SMILES_Tg.drop(['smiles'], axis = 1)
SMILES_Tg = SMILES_Tg.dropna(subset = SMILES_Tg.columns)

# X_train_Tg = SMILES_Tg.drop('property', axis=1)
# y_train_Tg = SMILES_Tg['property']

# collect all your columns in a list
columns_list = [columns_1, columns_2, columns_3, columns_4, columns_5, columns_6, columns_7, columns_8, columns_9, columns_10, columns_11, columns_12, columns_13, columns_14]

# dictionary to store models, X, y
x_train_Tg = {}
y_train_Tg = {}
models_Tg = {}

for i, cols in enumerate(columns_list, start=1):
    # create dataframe
    df = pd.DataFrame(SMILES_Tg[cols], index=SMILES_Tg.index)

    # assume last column is target, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split into train/test
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # save into dicts
    x_train_Tg[i] = X
    y_train_Tg[i] = y

    # train LightGBM
    model = CatBoostRegressor(random_state=42, verbose = 0)
    model.fit(X, y)

    # store model
    models_Tg[i] = model

print("DONE!")


SMILES_Tc = SMILES_Tc.drop(['smiles'], axis = 1)
SMILES_Tc = SMILES_Tc.dropna(subset = SMILES_Tc.columns)

# X_train_Tc = SMILES_Tc.drop('property', axis=1)
# y_train_Tc = SMILES_Tc['property']

# collect all your columns in a list
columns_list = [columns_1, columns_2, columns_3, columns_4, columns_5, columns_6, columns_7, columns_8, columns_9, columns_10, columns_11, columns_12, columns_13, columns_14]

# dictionary to store models, X, y
x_train_Tc = {}
y_train_Tc = {}
models_Tc = {}

for i, cols in enumerate(columns_list, start=1):
    # create dataframe
    df = pd.DataFrame(SMILES_Tc[cols], index=SMILES_Tc.index)

    # assume last column is target, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split into train/test
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # save into dicts
    x_train_Tc[i] = X
    y_train_Tc[i] = y

    # train LightGBM
    model = CatBoostRegressor(random_state=42, verbose = 0)
    model.fit(X, y)

    # store model
    models_Tc[i] = model

print("DONE!")


SMILES_Rg = SMILES_Rg.drop(['smiles'], axis = 1)
SMILES_Rg = SMILES_Rg.dropna(subset = SMILES_Rg.columns)

# X_train_Rg = SMILES_Rg.drop('property', axis=1)
# y_train_Rg = SMILES_Rg['property']

# collect all your columns in a list
columns_list = [columns_1, columns_2, columns_3, columns_4, columns_5, columns_6, columns_7, columns_8, columns_9, columns_10, columns_11, columns_12, columns_13, columns_14]

# dictionary to store models, X, y
x_train_Rg = {}
y_train_Rg = {}
models_Rg = {}

for i, cols in enumerate(columns_list, start=1):
    # create dataframe
    df = pd.DataFrame(SMILES_Rg[cols], index=SMILES_Rg.index)

    # assume last column is target, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split into train/test
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # save into dicts
    x_train_Rg[i] = X
    y_train_Rg[i] = y

    # train LightGBM
    model = CatBoostRegressor(random_state=42, verbose = 0)
    model.fit(X, y)

    # store model
    models_Rg[i] = model

print("DONE!")


SMILES_Density = SMILES_Density.drop(['smiles'], axis = 1)
SMILES_Density = SMILES_Density.dropna(subset = SMILES_Density.columns)

# X_train_Density = SMILES_Density.drop('property', axis=1)
# y_train_Density = SMILES_Density['property']

# collect all your columns in a list
columns_list = [columns_1, columns_2, columns_3, columns_4, columns_5, columns_6, columns_7, columns_8, columns_9, columns_10, columns_11, columns_12, columns_13, columns_14]

# dictionary to store models, X, y
x_train_Density = {}
y_train_Density = {}
models_Density = {}

for i, cols in enumerate(columns_list, start=1):
    # create dataframe
    df = pd.DataFrame(SMILES_Density[cols], index=SMILES_Density.index)

    # assume last column is target, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split into train/test
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # save into dicts
    x_train_Density[i] = X
    y_train_Density[i] = y

    # train LightGBM
    model = CatBoostRegressor(random_state=42, verbose = 0)
    model.fit(X, y)

    # store model
    models_Density[i] = model

print("DONE!")


# def build_mlp(input_dim):
#     model = models.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(512, activation="relu"),
#         layers.Dropout(0.3),
#         layers.Dense(256, activation="relu"),
#         layers.Dropout(0.3),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(1, activation="linear")  # regression output
#     ])
    
#     model.compile(
#     optimizer="adam",
#     loss=losses.MeanAbsoluteError(),   # instead of "mae"
#     metrics=[metrics.MeanAbsoluteError()]   # instead of "mae"
# )
#     return model


# xgb_model_Tc = XGBRegressor(
#     colsample_bytree = 0.7190763874791463,
#     learning_rate = 0.023773031583848494,
#     max_depth = 4,
#     n_estimators = 2002,
#     subsample = 0.8860257969776858,
#     random_state=42,
#     tree_method='hist',
#     n_jobs=-1)
# params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.44939188440490874, 'importance_type': 'split', 'learning_rate': 0.022628184820634573, 'max_depth': 7, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 717, 'n_jobs': -1, 'num_leaves': 184, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.3202421378276399, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'lambda_l1': 0.5468770981955229, 'lambda_l2': 1.2526187249841252, 'verbose': -1}

# lgb_model_Tc = lgb.LGBMRegressor(**params)

# cat_model_Tc = CatBoostRegressor(
#     iterations = 1965,
#     learning_rate = 0.014438056936402337,
#     depth = 5,
#     l2_leaf_reg = 7.658092967980456,
#     loss_function = 'RMSE',
#     border_count = 64,
#     random_strength = 0.3370571702155915,
#     bagging_temperature = 0.021952714963458617,
#     random_state = 42,
#     verbose = 0
# )

# model_Tc = build_mlp(X_train_Tc.shape[1])

# history = model_Tc.fit(
#     X_train_Tc, y_train_Tc,
#     epochs=100,
#     batch_size=64,
#     # callbacks=[es],
#     verbose=1
# )

# xgb_model_Tc.fit(X_train_Tc, y_train_Tc)
# lgb_model_Tc.fit(X_train_Tc, y_train_Tc)
# cat_model_Tc.fit(X_train_Tc, y_train_Tc)
print(f"Model Training for Tc Complete!")


# xgb_model_Tg = XGBRegressor(
#     colsample_bytree = 0.7577049157563494,
#     learning_rate = 0.05451020630556633,
#     max_depth = 9,
#     n_estimators = 628,
#     subsample = 0.7332451379558025,
#     random_state=42,
#     tree_method='hist',
#     n_jobs=-1
# )
# params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.3832370823240654, 'importance_type': 'split', 'learning_rate': 0.09952565171736023, 'max_depth': 9, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 2905, 'n_jobs': -1, 'num_leaves': 159, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.5291372793544382, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'lambda_l1': 0.8492518981046656, 'lambda_l2': 2.8912616615025994, 'verbose': -1}
# lgb_model_Tg = lgb.LGBMRegressor(**params)

# cat_model_Tg = CatBoostRegressor(
#     iterations = 2464,
#     learning_rate = 0.03949873228707747,
#     depth = 11,
#     l2_leaf_reg = 8.894594569824987,
#     loss_function = 'RMSE',
#     border_count = 32,
#     random_strength = 0.9608925989022382,
#     bagging_temperature = 0.041469553176421536,
#     random_state = 42,
#     verbose = 0
# )

# model_Tg = build_mlp(X_train_Tg.shape[1])

# history = model_Tg.fit(
#     X_train_Tg, y_train_Tg,
#     epochs=100,
#     batch_size=64,
#     # callbacks=[es],
#     verbose=1
# )

# xgb_model_Tg.fit(X_train_Tg, y_train_Tg)
# lgb_model_Tg.fit(X_train_Tg, y_train_Tg)
# # cat_model_Tg.fit(X_train_Tg, y_train_Tg)
# print(f"Model Training for Tg Complete!")


# xgb_model_Density = XGBRegressor(
#     colsample_bytree = 1.0,
#     learning_rate = 0.1,
#     max_depth = 5,
#     n_estimators = 999,
#     subsample = 0.8,
#     random_state=42,
#     tree_method='hist',
#     n_jobs=-1)
# params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.32157536792029323, 'importance_type': 'split', 'learning_rate': 0.04424832987368011, 'max_depth': 9, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1766, 'n_jobs': -1, 'num_leaves': 147, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.31592653698901324, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'lambda_l1': 0.5152217389972531, 'lambda_l2': 3.624684864186933, 'verbose': -1}
# lgb_model_Density = lgb.LGBMRegressor(**params)

# cat_model_Density = CatBoostRegressor(
#     iterations = 1988,
#     learning_rate = 0.06720419991959246,
#     depth = 9,
#     l2_leaf_reg = 6.618394329790608,
#     loss_function = 'RMSE',
#     border_count = 248,
#     random_strength = 0.7511607996589738,
#     bagging_temperature = 0.7563852191253406,
#     random_state = 42,
#     verbose = 0
# )

# model_Density = build_mlp(X_train_Density.shape[1])

# history = model_Density.fit(
#     X_train_Density, y_train_Density,
#     epochs=100,
#     batch_size=64,
#     # callbacks=[es],
#     verbose=1
# )

# xgb_model_Density.fit(X_train_Density, y_train_Density)
# lgb_model_Density.fit(X_train_Density, y_train_Density)
# # cat_model_Density.fit(X_train_Density, y_train_Density)
# print(f"Model Training for Density Complete!")


# xgb_model_FFV = XGBRegressor(
#     colsample_bytree = 0.8,
#     learning_rate = 0.01,
#     max_depth = 10,
#     n_estimators = 2000,
#     subsample = 0.8,
#     random_state=42,
#     tree_method='hist',
#     n_jobs=-1
# )
# params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.38413525380821395, 'importance_type': 'split', 'learning_rate': 0.034761657616416045, 'max_depth': 7, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 2918, 'n_jobs': -1, 'num_leaves': 86, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.5799764426845296, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'lambda_l1': 0.5098000795908426, 'lambda_l2': 0.4412097152930561, 'verbose': -1}
# lgb_model_FFV = lgb.LGBMRegressor(**params)

# cat_model_FFV = CatBoostRegressor(
#     iterations = 397,
#     learning_rate = 0.11806807371790191,
#     depth = 8,
#     l2_leaf_reg = 4.579595038056475,
#     loss_function = 'RMSE',
#     border_count = 75,
#     random_strength = 0.15703626130864143,
#     bagging_temperature = 0.3252723379135971,
#     random_state = 42,
#     verbose = 0
# )

# model_FFV = build_mlp(X_train_FFV.shape[1])

# history = model_FFV.fit(
#     X_train_FFV, y_train_FFV,
#     epochs=100,
#     batch_size=64,
#     # callbacks=[es],
#     verbose=1
# )

# xgb_model_FFV.fit(X_train_FFV, y_train_FFV)
# lgb_model_FFV.fit(X_train_FFV, y_train_FFV)
# # cat_model_FFV.fit(X_train_FFV, y_train_FFV)
# print(f"Model Training for FFV Complete!")


# xgb_model_Rg = XGBRegressor(
#     colsample_bytree = 0.8,
#     learning_rate = 0.01,
#     max_depth = 7,
#     n_estimators = 560,
#     subsample = 0.8,
#     random_state=42,
#     tree_method='hist',
#     n_jobs=-1
# )
# params = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.45409035489078536, 'importance_type': 'split', 'learning_rate': 0.09338309881127552, 'max_depth': 6, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 2131, 'n_jobs': -1, 'num_leaves': 78, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.5141708696370169, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'lambda_l1': 0.7726098502643955, 'lambda_l2': 1.2988602806545275, 'verbose': -1}
# lgb_model_Rg = lgb.LGBMRegressor(**params)

# cat_model_Rg = CatBoostRegressor(
#     iterations = 2392,
#     learning_rate = 0.010142555934976647,
#     depth = 6,
#     l2_leaf_reg = 1.6160089020018606,
#     loss_function = 'RMSE',
#     border_count = 244,
#     random_strength = 0.4542910556282948,
#     bagging_temperature = 0.6597049934702172,
#     random_state = 42,
#     verbose = 0
# )

# model_Rg = build_mlp(X_train_Rg.shape[1])

# history = model_Rg.fit(
#     X_train_Rg, y_train_Rg,
#     epochs=100,
#     batch_size=64,
#     # callbacks=[es],
#     verbose=1
# )

# xgb_model_Rg.fit(X_train_Rg, y_train_Rg)
# lgb_model_Rg.fit(X_train_Rg, y_train_Rg)
# # cat_model_Rg.fit(X_train_Rg, y_train_Rg)
# print(f"Model Training for Rg Complete!")


# from tensorflow.keras.models import load_model
# from tensorflow.keras import metrics
# model_Tc = load_model("/kaggle/input/mlp-model/full_model_Tc.keras")
# model_Tg = load_model("/kaggle/input/mlp-model/full_model_Tg.keras")
# model_Rg = load_model("/kaggle/input/mlp-model/full_model_Rg.keras")
# model_Density = load_model("/kaggle/input/mlp-model/full_model_Density.keras")
# model_FFV = load_model("/kaggle/input/mlp-model/full_model_FFV.keras")


SMILES_Test_Final.head()


SMILES_Test_Final.index = SMILES_Test_Final.iloc[:, 0]
SMILES_Test_Final = SMILES_Test_Final.drop(['smiles', 'id'], axis = 1)

SMILES_Test_Final = SMILES_Test_Final.replace([np.inf, -np.inf], np.nan)
l_Test = [k for k in SMILES_Test_Final.columns if SMILES_Test_Final[k].dtype == "object"]
if l_Test != []:
    for col in l_Test:
        converted = pd.to_numeric(SMILES_Test_Final[col], errors="coerce")
    
        # find rows where conversion failed (these caused object dtype)
        bad_rows = SMILES_Test_Final.index[converted.isna() & SMILES_Test_Final[col].notna()]
    
        if not bad_rows.empty:
            print(f"Column '{col}' has non-numeric values at rows: {bad_rows.tolist()}")
    
        # set bad values to NaN
        SMILES_Test_Final.loc[bad_rows, col] = np.nan
    
        # finally, assign back the converted numeric column
        SMILES_Test_Final[col] = pd.to_numeric(SMILES_Test_Final[col], errors="coerce")

imputer = SimpleImputer(strategy='mean')

# Fit and transform the DataFrame
SMILES_Test_Final_imputed = pd.DataFrame(imputer.fit_transform(SMILES_Test_Final), columns=SMILES_Test_Final.columns)

# (Optional) Restore original index
SMILES_Test_Final_imputed.index = SMILES_Test_Final.index

# scaler = StandardScaler()
# SMILES_Test_Final_imputed_scaled = scaler.fit_transform(SMILES_Test_Final_imputed)


submission = pd.DataFrame(columns = ["id", "Tg", "FFV", "Tc", "Density", "Rg"])

# Tg_xgb = xgb_model_Tg.predict(SMILES_Test_Final_imputed)
# Tg_lgb = lgb_model_Tg.predict(SMILES_Test_Final_imputed)
# Tg_cat = cat_model_Tg.predict(SMILES_Test_Final_imputed)
# FFV_xgb = xgb_model_FFV.predict(SMILES_Test_Final_imputed)
# FFV_lgb = lgb_model_FFV.predict(SMILES_Test_Final_imputed)
# FFV_cat = cat_model_FFV.predict(SMILES_Test_Final_imputed)
# Tc_xgb = xgb_model_Tc.predict(SMILES_Test_Final_imputed)
# Tc_lgb = lgb_model_Tc.predict(SMILES_Test_Final_imputed)
# Tc_cat = cat_model_Tc.predict(SMILES_Test_Final_imputed)
# Density_xgb = xgb_model_Density.predict(SMILES_Test_Final_imputed)
# Density_lgb = lgb_model_Density.predict(SMILES_Test_Final_imputed)
# Density_cat = cat_model_Density.predict(SMILES_Test_Final_imputed)
# Rg_xgb = xgb_model_Rg.predict(SMILES_Test_Final_imputed)
# Rg_lgb = lgb_model_Rg.predict(SMILES_Test_Final_imputed)
# Rg_cat = cat_model_Rg.predict(SMILES_Test_Final_imputed)
# Tc = model_Tc.predict(SMILES_Test_Final_imputed_scaled)
# Tg = model_Tg.predict(SMILES_Test_Final_imputed_scaled)
# Rg = model_Rg.predict(SMILES_Test_Final_imputed_scaled)
# Density = model_Density.predict(SMILES_Test_Final_imputed_scaled)
# FFV = model_FFV.predict(SMILES_Test_Final_imputed_scaled)
print("Testing Done !")


predictions_FFV = {}
predictions_Density = {}
predictions_Tc = {}
predictions_Tg = {}
predictions_Rg = {}
for i, cols in enumerate(columns_list, start=1):
    # create dataframe
    cols.remove('property')
    test_df = pd.DataFrame(SMILES_Test_Final_imputed[cols], index=SMILES_Test_Final_imputed.index)
    predictions_FFV[i] = models_FFV[i].predict(test_df)
    predictions_Density[i] = models_Density[i].predict(test_df)
    predictions_Tc[i] = models_Tc[i].predict(test_df)
    predictions_Tg[i] = models_Tg[i].predict(test_df)
    predictions_Rg[i] = models_Rg[i].predict(test_df)
print("Testing Done !")


n = len(predictions_FFV[1])
final_predictions_FFV = {}
final_predictions_Density = {}
final_predictions_Tc = {}
final_predictions_Tg = {}
final_predictions_Rg = {}
for i in range(n):
    finale_FFV = 0
    finale_Density = 0
    finale_Tc = 0
    finale_Tg = 0
    finale_Rg = 0
    for j in range(1, 15):
        finale_FFV += predictions_FFV[j][i]
        finale_Density += predictions_Density[j][i]
        finale_Tg += predictions_Tg[j][i]
        finale_Tc += predictions_Tc[j][i]
        finale_Rg += predictions_Rg[j][i]
    final_predictions_FFV[i] = finale_FFV/14
    final_predictions_Density[i] = finale_Density/14
    final_predictions_Tg[i] = finale_Tg/14
    final_predictions_Tc[i] = finale_Tc/14
    final_predictions_Rg[i] = finale_Rg/14


for i in range(SMILES_Test_Final_imputed.shape[0]):
    d = {
        "id" : SMILES_Test_Final_imputed.index[i],
        "Tg" : final_predictions_Tg[i],
        "FFV" : final_predictions_FFV[i],
        "Tc" : final_predictions_Tc[i],
        "Density" : final_predictions_Density[i],
        "Rg" : final_predictions_Rg[i]
    }
    submission.loc[len(submission)] = d

print("Submission File Completed!")


submission.head()


submission.to_csv("submission.csv", index = False)




