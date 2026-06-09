!pip install --no-index --find-links /kaggle/input/custom-packages-wheels torch tqdm transformers==4.38.2
!pip install --no-index --find-links /kaggle/input/chembert-wheel torch tqdm rdkit==2023.9.6 transformers==4.38.2


import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Lipinski import RotatableBondSmarts
import networkx as nx

import json
import time
import csv
import os
from tqdm import tqdm

import torch 
from tqdm.notebook import tqdm
from transformers import AutoModel,AutoTokenizer

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,AdaBoostRegressor

import joblib


SMILES_Tc = pd.read_csv("/kaggle/input/neurips-rdkit-descriptors/TcFinal_rdkit.csv", index_col = "index")
SMILES_Tg = pd.read_csv("/kaggle/input/neurips-rdkit-descriptors/TgFinal_rdkit.csv", index_col = "index")
SMILES_Density = pd.read_csv("/kaggle/input/neurips-rdkit-descriptors/DensityFinal_rdkit.csv", index_col = "index")
SMILES_Test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
SMILES_FFV = pd.read_csv("/kaggle/input/smiles-features-datasets/SMILES_COMPLETED_train.csv", index_col = "index")
SMILES_Train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
density_df = pd.read_csv("/kaggle/input/smiles-datasets/DensityFinal.csv")
tc_df = pd.read_csv("/kaggle/input/smiles-datasets/TcFinal.csv")
tg_df = pd.read_csv("/kaggle/input/smiles-datasets/Tg4Final.csv")

print("All Datasets Loaded !")


density_df["Density"] = pd.to_numeric(density_df['Density'],errors = 'coerce')
non_numeric_indices = SMILES_Density[~pd.to_numeric(SMILES_Density['Density'], errors='coerce').notnull()].index
print(SMILES_Density.loc[non_numeric_indices, 'Density'])
SMILES_Density = SMILES_Density.drop(index=non_numeric_indices).reset_index(drop=True)
cols_to_fix = ['Density']

for col in cols_to_fix:
    SMILES_Density[col] = pd.to_numeric(SMILES_Density[col], errors='coerce')


def rdkit_descriptor_pipeline(data):
    """
    Reads SMILES + property from CSV and generates all RDKit descriptors.
    Saves results progressively with tqdm progress bars.
    Returns a dataframe with all descriptors for each molecule.
    """
    n = data.shape[0]
    print(f"{n} SMILES left to be processed.")
    
    # Get all descriptor names
    descriptor_list = Descriptors._descList
    desc_names = [d[0] for d in descriptor_list]
    
    progress_steps = 100
    bar = tqdm(total=progress_steps, desc="Processing SMILES", unit="SMILES", leave=False)
    counter = 0
    results = []  # List to store each dictionary as a row
    
    for k in range(n):
        try:
            index = data.iloc[k, 0]
            smiles = data.iloc[k, 1]
            smiles = smiles.replace("*", "[H]")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            
            # Compute all descriptors
            desc_values = {}
            for name, func in descriptor_list:
                try:
                    desc_values[name] = func(mol)
                except Exception:
                    desc_values[name] = None
            
            d = {"index": index, "SMILES": smiles}
            d.update(desc_values)
            
        except Exception as e:
            print(f"[Error] at index {k} with SMILES {smiles}: {e}")
            d = {"index": index, "SMILES": smiles}
            d.update({name: None for name in desc_names})
        
        # Add the dictionary as a row to results
        results.append(d)
        
        # tqdm handling
        counter += 1
        bar.update(1)
        if counter == progress_steps:
            bar.close()
            print(f"{k + 1} SMILES processed so far!")
            counter = 0
            bar = tqdm(total=progress_steps, desc="Processing SMILES", unit="SMILES", leave=False)
    
    # Handle remaining progress bar updates
    if counter > 0:
        bar.update(progress_steps - counter)
    bar.close()
    print(f"All {k + 1} SMILES updated in file!")
    
    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure column order: index, SMILES, then all descriptors
    column_order = ["index", "SMILES"] + desc_names
    results_df = results_df[column_order]
    
    return results_df



SMILES_Tc = SMILES_Tc.drop(['SMILES'], axis = 1)
SMILES_Tc = SMILES_Tc.dropna(subset = SMILES_Tc.columns)

X_train_Tc = SMILES_Tc.drop('Tc', axis=1)
y_train_Tc = SMILES_Tc['Tc']

print("DONE!")


# tc_df.dropna(inplace = True)
density_df.dropna(inplace = True)


SMILES_Tg = SMILES_Tg.drop(['SMILES'], axis = 1)
SMILES_Tg = SMILES_Tg.dropna(subset = SMILES_Tg.columns)

X_train_Tg = SMILES_Tg.drop('Tg', axis=1)
y_train_Tg = SMILES_Tg['Tg']

print("DONE!")


SMILES_Density = SMILES_Density.drop(['SMILES'], axis = 1)
SMILES_Density = SMILES_Density.dropna(subset = SMILES_Density.columns)

X_train_Density = SMILES_Density.drop('Density', axis=1)
y_train_Density = SMILES_Density['Density']

print("DONE!")


# SMILES_FFV = SMILES_FFV.drop(['SMILES', 'Tg', 'T_c', 'Density', 'Rg'], axis = 1)
# SMILES_FFV = SMILES_FFV.dropna(subset = SMILES_FFV.columns)

# X_train_FFV = SMILES_FFV.drop('FFV', axis=1)
# y_train_FFV = SMILES_FFV['FFV']

# print("DONE!")


FFV_df = SMILES_Train.copy()
FFV_df = FFV_df.drop(columns = ['id','Density','Tc','Rg','Tg'])
FFV_df.dropna(inplace = True)

print("DONE!")


Rg_df = SMILES_Train.copy()
Rg_df = Rg_df.drop(columns = ['id','Density','Tc','FFV','Tg'])
Rg_df.dropna(inplace = True)

print("DONE!")


# density_df.dropna(inplace = True)


if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')


model_name_1 = "/kaggle/input/10m-mtr-chembert-wheel/ChemBERT_10M_MTR/DeepChem/ChemBERTa-10M-MTR"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1, use_fast=False)
model_1 = AutoModel.from_pretrained(model_name_1, trust_remote_code=True)
model_1 = model_1.to(device)
model_1.eval()


model_name_2 = "/kaggle/input/chembert-wheel/DeepChem/ChemBERTa-77M-MTR"
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2, use_fast=False)
model_2 = AutoModel.from_pretrained(model_name_2, trust_remote_code=True)
model_2 = model_2.to(device)
model_2.eval()


model_name_3 = "/kaggle/input/77m-mtm-chembert-wheel/ChemBERT_77M_MTM/DeepChem/ChemBERTa-77M-MLM"
tokenizer_3 = AutoTokenizer.from_pretrained(model_name_3, use_fast=False)
model_3 = AutoModel.from_pretrained(model_name_3, trust_remote_code=True)
model_3 = model_3.to(device)
model_3.eval()


def preprocess(x, model, tokenizer):
    
    embeddings = []
    
    with torch.no_grad():
        for smile in tqdm(x):
            inputs = tokenizer(smile,return_tensors = 'pt',padding = True,truncation = False).to(device)
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls)

    return np.array(embeddings)


ffv_x = preprocess(FFV_df["SMILES"], model_1, tokenizer_1)
# tc_x_1 = preprocess(tc_df["SMILES"], model_1, tokenizer_1)
# tg_x_1 = preprocess(tg_df["SMILES"], model_1, tokenizer_1)
# density_x_1 = preprocess(density_df["SMILES"], model_1, tokenizer_1)

print(f"Data Preprocessing from Model 1 Done!")


rg_x_2 = preprocess(Rg_df["SMILES"], model_2, tokenizer_2)
# tc_x_2 = preprocess(tc_df["SMILES"], model_2, tokenizer_2)
# tg_x_2 = preprocess(tg_df["SMILES"], model_2, tokenizer_2)
# density_x_2 = preprocess(density_df["SMILES"], model_2, tokenizer_2)

print(f"Data Preprocessing from Model 2 Done!")


rg_x_3 = preprocess(Rg_df["SMILES"], model_3, tokenizer_3)

print(f"Data Preprocessing from Model 3 Done!")


rg_x = rg_x_2 + rg_x_3
# tc_x = tc_x_1 + tc_x_2
# tg_x = tg_x_1 + tg_x_2
# density_x = density_x_1 + density_x_2


rg_y = np.array(Rg_df.iloc[:,-1])
# # tc_y = np.array(tc_df.iloc[:,-1])
# tg_y = np.array(tg_df.iloc[:,-1])
ffv_y = np.array(FFV_df.iloc[:,-1])
# density_y = np.array(density_df.iloc[:,-1])


# xgb_model_Tc = XGBRegressor(
#     colsample_bytree = 0.8,
#     learning_rate = 0.01,
#     max_depth = 5,
#     n_estimators = 1300,
#     subsample = 0.8,
#     random_state=42,
#     tree_method='hist',
#     n_jobs=-1
# )

# lgb_model_Tc = LGBMRegressor(
#     n_estimators = 126,
#     class_weight = None,
#     importance_type = 'split',
#     learning_rate = 0.05,
#     min_child_samples = 20,
#     min_child_weight = 0.001,
#     min_split_gain = 0.0,
#     num_leaves = 33,
#     max_depth = 11,
#     colsample_bytree = 1.0,
#     subsample = 1.0,
#     random_state=42, 
#     n_jobs=-1,
#     objective='regression',
#     boosting_type='gbdt',
#     verbose=-1
# )

cat_model_Tc = CatBoostRegressor(
    iterations = 1965,
    learning_rate = 0.014438056936402337,
    depth = 5,
    l2_leaf_reg = 7.658092967980456,
    loss_function = 'RMSE',
    border_count = 64,
    random_strength = 0.3370571702155915,
    bagging_temperature = 0.021952714963458617,
    random_state = 42,
    verbose = 0
)

# xgb_model_Tc.fit(X_train_Tc, y_train_Tc)
# lgb_model_Tc.fit(X_train_Tc, y_train_Tc)
cat_model_Tc.fit(X_train_Tc, y_train_Tc)
print(f"Model Training for Tc Complete!")


# xgb_model_Tg = XGBRegressor(
#     colsample_bytree = 0.8,
#     learning_rate = 0.1,
#     max_depth = 8,
#     n_estimators = 900,
#     subsample = 1.0,
#     random_state=42,
#     tree_method='hist',
#     n_jobs=-1
# )

lgb_model_Tg = LGBMRegressor(
    boosting_type ='gbdt',
    class_weight = None,
    colsample_bytree = 0.5411710709981165,
    importance_type = 'split',
    learning_rate = 0.07834240644853582,
    max_depth = 8,
    min_child_samples = 20,
    min_child_weight = 0.001,
    min_split_gain = 0.0,
    n_estimators = 1932,
    n_jobs =-1,
    num_leaves = 73,
    subsample = 0.5344838408125547,
    random_state =42, 
    objective ='regression',
    lambda_l1 = 0.8809202794316024,
    lambda_l2 = 3.2702363610658445,
    verbose=-1
)

# cat_model_Tg = CatBoostRegressor(
#     iterations = 913,
#     learning_rate = 0.14670306617689124,
#     depth = 9,
#     l2_leaf_reg = 6.267110409549005,
#     loss_function = 'RMSE',
#     border_count = 90,
#     random_strength = 0.7469081892120999,
#     bagging_temperature = 0.8618385100141003,
#     random_state = 42,
#     verbose = 0
# )

# xgb_model_Tg.fit(X_train_Tg, y_train_Tg)
lgb_model_Tg.fit(X_train_Tg, y_train_Tg)
# cat_model_Tg.fit(X_train_Tg, y_train_Tg)
print(f"Model Training for Tg Complete!")


# lgb_model_Density = LGBMRegressor(
#     boosting_type='gbdt',
#     class_weight = None,
#     colsample_bytree = 0.8,
#     importance_type = 'split',
#     learning_rate = 0.1,
#     max_depth = 3,
#     min_child_samples = 20,
#     min_child_weight = 0.001,
#     min_split_gain = 0.0,
#     n_estimators = 2783,
#     n_jobs=-1,
#     num_leaves = 117,
#     subsample = 0.8,
#     random_state=42, 
#     objective='regression',
#     lambda_l1 = 0.1,
#     lambda_l2 = 0.1,
#     verbose=-1
# )

cat_model_Density = CatBoostRegressor(
    iterations = 1988,
    learning_rate = 0.06720419991959246,
    depth = 9,
    l2_leaf_reg = 6.618394329790608,
    loss_function = 'RMSE',
    border_count = 248,
    random_strength = 0.7511607996589738,
    bagging_temperature = 0.7563852191253406,
    random_state = 42,
    verbose = 0
)

# lgb_model_Density.fit(X_train_Density, y_train_Density)
cat_model_Density.fit(X_train_Density, y_train_Density)
print(f"Model Training for Density Complete!")
# # ChemBERT for Density


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

# cat_model_FFV = CatBoostRegressor(
#     iterations = 1746,
#     learning_rate = 0.03718960121272398,
#     depth = 9,
#     l2_leaf_reg = 9.937714458001093,
#     loss_function = 'RMSE',
#     border_count = 56,
#     random_strength = 0.5646684138750556,
#     bagging_temperature = 0.10227124249646499,
#     random_state = 42,
#     verbose = 0
# )

# xgb_model_FFV.fit(X_train_FFV, y_train_FFV)
# cat_model_FFV.fit(X_train_FFV, y_train_FFV)
# print(f"Model Training for FFV Complete!")
# # ChemBERT for FFV


model_map = {
    
    'FFV': CatBoostRegressor(verbose=0),
    'RG': CatBoostRegressor(verbose=0)
    # 'Tc': CatBoostRegressor(verbose=0),
    # 'Tg': CatBoostRegressor(verbose=0),
    # 'Density': CatBoostRegressor(verbose=0)
}

train_data_map = {
    'RG': (rg_x, rg_y),
    # 'Tc': (tc_x, tc_y),
    # 'Tg': (tg_x, tg_y),
    'FFV': (ffv_x, ffv_y)
    # 'Density': (density_x, density_y)
}


def train_and_save_all(data_map, model_map):
    for target, (X, y) in data_map.items():
        model = model_map[target]
        print(f"Training model for {target}...")
        model.fit(X, y)
        joblib.dump(model, f"{target}.pkl")
        print(f"Saved {target}.pkl ✅")


train_and_save_all(train_data_map, model_map)
print(f"Model Training of FFV, Density and Rg Done!")


SMILES_Test_Final = rdkit_descriptor_pipeline(SMILES_Test)


descriptors = ['SMILES', 'index', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
for i in SMILES_Test_Final.columns:
    if i not in descriptors:
        SMILES_Test_Final = SMILES_Test_Final.drop([i], axis = 1)
SMILES_Test_Final.shape


SMILES_Test_Final.index = SMILES_Test_Final.iloc[:, 0]
SMILES_Test_Final = SMILES_Test_Final.drop(['SMILES', 'index'], axis = 1)

SMILES_Test_Final = SMILES_Test_Final.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy='median')

# Fit and transform the DataFrame
SMILES_Test_Final_imputed = pd.DataFrame(imputer.fit_transform(SMILES_Test_Final), columns=SMILES_Test_Final.columns)

# (Optional) Restore original index
SMILES_Test_Final_imputed.index = SMILES_Test_Final.index


SMILES_Test_Final.shape


submission = pd.DataFrame(columns = ["id", "Tg", "FFV", "Tc", "Density", "Rg"])

# Tg_xgb = xgb_model_Tg.predict(SMILES_Test_Final_imputed)
Tg_lgb = lgb_model_Tg.predict(SMILES_Test_Final_imputed)
# Tg_cat = cat_model_Tg.predict(SMILES_Test_Final_imputed)
# Tc_xgb = xgb_model_Tc.predict(SMILES_Test_Final_imputed)
# Tc_lgb = lgb_model_Tc.predict(SMILES_Test_Final_imputed)
Tc_cat = cat_model_Tc.predict(SMILES_Test_Final_imputed)
# Density_lgb = lgb_model_Density.predict(SMILES_Test_Final_imputed)
Density_cat = cat_model_Density.predict(SMILES_Test_Final_imputed)
# FFV_xgb = xgb_model_FFV.predict(SMILES_Test_Final_imputed)
# FFV_cat = cat_model_FFV.predict(SMILES_Test_Final_imputed)


# test_x_Rg = preprocess(SMILES_Test["SMILES"], model_1, tokenizer_1)
test_x_1 = preprocess(SMILES_Test["SMILES"], model_1, tokenizer_1)
test_x_2 = preprocess(SMILES_Test["SMILES"], model_2, tokenizer_2)
test_x_3 = preprocess(SMILES_Test["SMILES"], model_3, tokenizer_3)
test_x_FFV = test_x_1
# test_x_Density = test_x_1 + test_x_2
test_x_Rg = test_x_2 + test_x_3


model_file_Rg = {"Rg": "RG.pkl"}
# model_file_Tc = {"Tc": "Tc.pkl"}
# model_file_Tg = {"Tg": "Tg.pkl"}
model_file_FFV = {"FFV": "FFV.pkl"}
# model_file_Density = {"Density": "Density.pkl"}


predictions = {}

for target_name, model_file in model_file_Rg.items():
    model = joblib.load(model_file)
    predictions[target_name] = model.predict(test_x_Rg)
# for target_name, model_file in model_file_Tc.items():
#     model = joblib.load(model_file)
#     predictions[target_name] = model.predict(test_x_Tc)
# for target_name, model_file in model_file_Tg.items():
#     model = joblib.load(model_file)
#     predictions[target_name] = model.predict(test_x_Tg)
for target_name, model_file in model_file_FFV.items():
    model = joblib.load(model_file)
    predictions[target_name] = model.predict(test_x_FFV)
# for target_name, model_file in model_file_Density.items():
#     model = joblib.load(model_file)
#     predictions[target_name] = model.predict(test_x_Density)


print(predictions)


for i in range(SMILES_Test.shape[0]):
    d = {
        "id" : SMILES_Test.iloc[i, 0],
        "Tg" : Tg_lgb[i],
        "FFV" : predictions["FFV"][i],
        "Tc" : Tc_cat[i],
        "Density" : Density_cat[i],
        "Rg" : predictions["Rg"][i]
    }
    submission.loc[len(submission)] = d

print("Submission File Completed!")


submission.head(10)


submission.to_csv("submission.csv", index = False)




