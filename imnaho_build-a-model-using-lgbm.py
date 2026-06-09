# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
np.seterr(invalid='ignore')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
import re


import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module=r'seaborn\._oldcore',
    message=r'.*use_inf_as_na option is deprecated.*'
)
warnings.filterwarnings("ignore")


# Tg: Glass transition temperature
# FFV: Fractional free volume
# Tc: Thermal conductivity
# Density: Density
# Rg: Radius of gyration
path="/kaggle/input/neurips-open-polymer-prediction-2025/"
train = pd.read_csv(path+"train.csv")
print("train data shape is : " , train.shape)
train.head(3)


import kagglehub
download_path = kagglehub.dataset_download("senkin13/rdkit-2025-3-3-cp311")
print("Path to dataset files", download_path)


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem,Lipinski, rdMolDescriptors
from rdkit.Chem import Draw
from IPython.display import display


def calc_features(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "origin":mol,
    }
for i in range(5):
    smiles = train["SMILES"][i]
    mol    = calc_features(smiles)  # â†� Mol ã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆ
    print(smiles, " image is looks like below")
    display(Draw.MolToImage(mol["origin"], size=(250, 250)))
    print("==============================")


target_colums = train.columns.to_list()[2:]
print("target columns list is â†’" , target_colums , " = We need to make predictions about ",len(target_colums)," different characteristics")
contains_zero_rowwise = (train[target_colums] == 0).any(axis=1)
train[contains_zero_rowwise]


plt.figure(figsize=(12, 8))
for i, col in enumerate(target_colums):
    plt.subplot(2, 3, i + 1)
    sns.histplot(train[train[col] != 0][col], kde=False, bins=10)
    plt.title(f'Histogram of {col} (Excluding 0)')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


train.fillna(0, inplace=True)
target_cols=train.columns.to_list()[2:]
for col in target_cols:
    print( "feature " , col ,"s target row count is :", train[train[col]!=0].shape[0])


dataset1=pd.read_csv(path+"/train_supplement/dataset1.csv")
dataset2=pd.read_csv(path+"/train_supplement/dataset2.csv")
dataset3=pd.read_csv(path+"/train_supplement/dataset3.csv")
dataset4=pd.read_csv(path+"/train_supplement/dataset4.csv")
print("======================= dataset1 is below =======================")
print(dataset1.head(1))
print("======================= dataset2 is below =======================")
print(dataset2.head(1))
print("======================= dataset3 is below =======================")
print(dataset3.head(1))
print("======================= dataset4 is below =======================")
print(dataset4.head(1))


#Check for duplicates in the SMILES strings
dups = train[train['SMILES'].duplicated(keep=False)]
print("training data duplicate count is : ",dups.shape[0])
dups = dataset1[dataset1['SMILES'].duplicated(keep=False)]
print("dataset1 data duplicate count is : ",dups.shape[0])
dups = dataset3[dataset3['SMILES'].duplicated(keep=False)]
print("dataset3 data duplicate count is : ",dups.shape[0])
dups = dataset4[dataset4['SMILES'].duplicated(keep=False)]
print("dataset4 data duplicate count is : ",dups.shape[0])


dataset1.groupby("SMILES").count().sort_values("TC_mean").tail(8)


dataset1=dataset1.groupby("SMILES").mean().reset_index()


for dataset, col_name in zip([dataset1,dataset3,dataset4],["Tc","Tg","FFV"]):
    print("======================= Check if usable as '",col_name,"' feature =======================")
    same_data_cnt = 0
    different_data_cnt = 0
    differences = []
    additional_data_cnt = 0
    is_target_list=[]
    
    for i in range(dataset.shape[0]):
        if train.loc[train["SMILES"]==dataset.iloc[i,0],col_name].any()==True:
            is_target_list.append(0)
            if train.loc[train["SMILES"]==dataset.iloc[i,0],col_name].values[0] != dataset.iloc[i,1]:
                same_data_cnt+=1
                differences.append( np.abs(train.loc[train["SMILES"]==dataset.iloc[i,0],col_name].values[0] - dataset.iloc[i,1]))
            else:
                different_data_cnt+=1
        else:
            is_target_list.append(1)
            additional_data_cnt+=1
    print("SMILES in both datasets with identical values count is : ",same_data_cnt)
    print("SMILES in both datasets with differing values count is :" ,different_data_cnt)
    print("SMILES unique to the supplemental data (to add to training) count is :" ,additional_data_cnt)
    
    if different_data_cnt > 0:
        differences_arr = np.array(differences)
        median_val = np.median(differences_arr)
        print("average difference is :",median_val)

    dataset["target"]=is_target_list

    print()

print("======================= dataset1 is below =======================")
print(dataset1.head(2))
print()
print("======================= dataset3 is below =======================")
print(dataset3.head(2))
print()
print("======================= dataset4 is below =======================")
print(dataset4.head(2))


for row in dataset1[dataset1["target"]==1].iterrows():
    idx = len(train)
    train.loc[idx] = 0
    train.at[idx,"SMILES"]=row[1].SMILES
    train.at[idx,"Tc"]=row[1].TC_mean

for row in dataset3[dataset3["target"]==1].iterrows():
    idx = len(train)
    train.loc[idx] = 0
    train.at[idx,"SMILES"]=row[1].SMILES
    train.at[idx,"Tg"]=row[1].Tg

for row in dataset4[dataset4["target"]==1].iterrows():
    idx = len(train)
    train.loc[idx] = 0
    train.at[idx,"SMILES"]=row[1].SMILES
    train.at[idx,"FFV"]=row[1].FFV

print("training data shape is : ",train.shape)
print("training data duplicate SMILES row num is :" , train[train["SMILES"].duplicated(keep=False)].shape[0])


train=train.groupby("SMILES").sum().reset_index()
print("training data shape is : ",train.shape)
print("training data duplicate SMILES row num is :" , train[train["SMILES"].duplicated(keep=False)].shape[0])


import itertools
corr_mat = pd.DataFrame(np.nan, index=target_cols, columns=target_cols)
for col1, col2 in itertools.combinations(target_cols, 2):
    mask = (train[col1] != 0) & (train[col2] != 0)
    subset = train.loc[mask, [col1, col2]]
    if len(subset) >= 2:
        corr = subset[col1].corr(subset[col2])
        corr_mat.loc[col1, col2] = corr
        corr_mat.loc[col2, col1] = corr

np.fill_diagonal(corr_mat.values, 1)
plt.figure(figsize=(6,5))
sns.heatmap(corr_mat.astype(float), annot=True, vmin=-1, vmax=1, cmap='coolwarm', square=True)
plt.title('Pairwise Pearson correlation (non-zero rows only)')
plt.show()


def add_smiles_char_counts(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    s = df[smiles_col].fillna("")

    # `Common characters frequently appearing in SMILES (adjust as needed)`
    chars = list("CcNnOoSsPpFIBbH[]()=/#\\+-1234567890*@.")
    # â€» Since backslashes are escaped in Python strings, `'\\'` is used.

    # Add a safe prefix to column names (represent symbols in hexadecimal code)
    def colname_for_char(ch: str) -> str:
        return f"cnt_char_{ch}" if ch.isalnum() else f"cnt_char_0x{ord(ch):02x}"

    feat = pd.DataFrame(index=df.index)
    for ch in chars:
        feat[colname_for_char(ch)] = s.str.count(re.escape(ch))

    return pd.concat([df, feat], axis=1)

def add_smiles_token_counts(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    s = df[smiles_col].fillna("")

    # Common two-character tokens (extend as needed)
    tokens = ["Cl", "Br", "Si", "Se"]

    feat = pd.DataFrame(index=df.index)
    for tok in tokens:
        feat[f"cnt_tok_{tok}"] = s.str.count(re.escape(tok))

    return pd.concat([df, feat], axis=1)


from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def featurize_smiles(smiles: str):
    import numpy as np
    import pandas as pd

    mol = Chem.MolFromSmiles(smiles)
    base_cols = [
        "MolWt","NumAromaticRings","NumRings","FractionCSP3",
        "NumRotatableBonds","TPSA","HBA","HBD","BalabanJ","MolLogP",
        "HeavyAtomCount","NumHetero","NumAliphaticRings","NumSaturatedRings",
        # Additional features from here
        "ExactMolWt","HeavyAtomMolWt","MolMR","BertzCT","HallKierAlpha",
        "Kappa1","Kappa2","Kappa3",
        "Chi0v","Chi1v","Chi2v","Chi3v","Chi4v",
        "Chi0n","Chi1n","Chi2n","Chi3n","Chi4n",
        "NumValenceElectrons","NumRadicalElectrons",
        "NHOHCount","NOCount","NumAmideBonds",
        "NumAromaticCarbocycles","NumAromaticHeterocycles",
        "NumAliphaticCarbocycles","NumAliphaticHeterocycles",
        "NumSaturatedCarbocycles","NumSaturatedHeterocycles",
        "NumBridgeheadAtoms","NumSpiroAtoms",
        "NumAtomStereoCenters","NumUnspecifiedAtomStereoCenters",
        "MaxPartialCharge","MinPartialCharge",
        "MaxAbsPartialCharge","MinAbsPartialCharge",
        "QED",
        # Optional 3D shape descriptors (filled if calculated successfully)
        "PMI1","PMI2","PMI3","Asphericity","Eccentricity",
        "InertialShapeFactor","SpherocityIndex"
    ]
    if mol is None:
        return pd.Series(np.nan, index=base_cols)

    # Calculate partial charges (PEOE) to make Max/MinPartialCharge values more stable
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    feat = {
        # Existing features
        "MolWt": Descriptors.MolWt(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "NumHetero": Lipinski.NumHeteroatoms(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol),
        "NumSaturatedRings": Lipinski.NumSaturatedRings(mol),

        # Additional physicochemical and structural features
        "ExactMolWt": rdMolDescriptors.CalcExactMolWt(mol),
        "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt(mol),
        "MolMR": Crippen.MolMR(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),

        # Chi indices (valence/normal)
        "Chi0v": Descriptors.Chi0v(mol),
        "Chi1v": Descriptors.Chi1v(mol),
        "Chi2v": Descriptors.Chi2v(mol),
        "Chi3v": Descriptors.Chi3v(mol),
        "Chi4v": Descriptors.Chi4v(mol),
        "Chi0n": Descriptors.Chi0n(mol),
        "Chi1n": Descriptors.Chi1n(mol),
        "Chi2n": Descriptors.Chi2n(mol),
        "Chi3n": Descriptors.Chi3n(mol),
        "Chi4n": Descriptors.Chi4n(mol),

        # Electron count and radical count
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),

        # Functional group and atom-type counts
        "NHOHCount": Lipinski.NHOHCount(mol),
        "NOCount": Lipinski.NOCount(mol),
        "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),
        "NumAromaticCarbocycles": Lipinski.NumAromaticCarbocycles(mol),
        "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles(mol),
        "NumAliphaticCarbocycles": Lipinski.NumAliphaticCarbocycles(mol),
        "NumAliphaticHeterocycles": Lipinski.NumAliphaticHeterocycles(mol),
        "NumSaturatedCarbocycles": Lipinski.NumSaturatedCarbocycles(mol),
        "NumSaturatedHeterocycles": Lipinski.NumSaturatedHeterocycles(mol),

        # Ring and stereochemistry
        "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "NumAtomStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        "NumUnspecifiedAtomStereoCenters": rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),

        # Partial charges (Gasteiger)
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(mol),
        "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge(mol),

        # Overall drug-likeness score (QED)
        "QED": QED.qed(mol),
    }

    # 3D shape descriptors (generate coordinates if missing and attempt calculation)
    try:
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        feat["PMI1"] = Descriptors3D.PMI1(mol)
        feat["PMI2"] = Descriptors3D.PMI2(mol)
        feat["PMI3"] = Descriptors3D.PMI3(mol)
        feat["Asphericity"] = Descriptors3D.Asphericity(mol)
        feat["Eccentricity"] = Descriptors3D.Eccentricity(mol)
        feat["InertialShapeFactor"] = Descriptors3D.InertialShapeFactor(mol)
        feat["SpherocityIndex"] = Descriptors3D.SpherocityIndex(mol)
    except Exception:
        # If 3D calculation fails, fill with NaN
        for k in ["PMI1","PMI2","PMI3","Asphericity","Eccentricity",
                  "InertialShapeFactor","SpherocityIndex"]:
            feat[k] = np.nan

    # Return features (keeping column order fixed)
    return pd.Series({k: feat.get(k, np.nan) for k in base_cols})


features = train["SMILES"].apply(featurize_smiles)
df_feat = pd.concat([train, features], axis=1)
df_feat = add_smiles_char_counts(df_feat, smiles_col="SMILES")
df_feat = add_smiles_token_counts(df_feat, smiles_col="SMILES")
df_feat = df_feat.drop("id",axis=1)
print(df_feat.shape)
df_feat.head()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import early_stopping
from lightgbm import LGBMRegressor

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Utility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def feature_cols(df_feat_data, targets=TARGETS):
    print("***")
    num_cols = df_feat_data.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in targets]

def lgb_params_by_n(n_samples, random_state=42):
    """Scale `n_estimators` and `depth` based on the number of samples"""
    n_est = int(min(4000, max(600, 22 * np.sqrt(n_samples))))
    depth = -1 if n_samples > 3000 else 6
    return dict(
        n_estimators=n_est,
        learning_rate=0.02,
        num_leaves=2 ** 6,     # Equivalent to depth 6
        max_depth=depth,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="mae",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_single_lgbm(
    df_feat: pd.DataFrame,
    target: str,
    feat_cols: list,
    n_splits: int = 5,
    random_state: int = 42,
):
    # --- exclude 0 ---
    data = df_feat.loc[df_feat[target].notna() & (df_feat[target] != 0)].copy()
    X = data[feat_cols].values
    y = data[target].values
    n = len(y)
    if n < 2:
        raise ValueError(f"{target}: Insufficient valid samples (n={n})")

    params = lgb_params_by_n(n, random_state)
    kf = KFold(n_splits=max(2, min(n_splits, n)), shuffle=True, random_state=random_state)

    oof = np.full(n, np.nan)
    fold_mae = []

    for tr_idx, va_idx in kf.split(X):
        model = LGBMRegressor(**params)
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            eval_metric="l1",
            callbacks=[
                early_stopping(stopping_rounds=300, verbose=False)
            ],
        )
        pred = model.predict(X[va_idx])
        oof[va_idx] = pred
        fold_mae.append(mean_absolute_error(y[va_idx], pred))

    cv_mae = float(np.mean(fold_mae))

    # --- full data training ---
    final_model = LGBMRegressor(**params)
    final_model.fit(X, y)

    return {
        "target": target,
        "model": final_model,
        "cv_mae": cv_mae,
        "n_train": n,
        "oof_pred": oof,
        "feat_cols": feat_cols,  # <-- store the features actually used (for prediction time)
    }

def train_all_lgbm(
    df_feat: pd.DataFrame,
    targets=TARGETS,
    n_splits=5,
    random_state=42,
    feat_cols_by_target=None,  # dict like {"Tg": [...], "FFV": [...], ...} or None
):
    feats_default = feature_cols(df_feat, targets)
    results = {}
    for t in targets:
        # Use target-specific features if provided; otherwise, use all numeric features (default)
        feats_t = (feat_cols_by_target.get(t) if isinstance(feat_cols_by_target, dict) else None) or feats_default
        res = train_single_lgbm(df_feat, t, feats_t, n_splits, random_state)
        results[t] = res
        print(f"{t}: CV MAE={res['cv_mae']:.5f}  n={res['n_train']}")
    return results, feats_default  # keep return shape for backward compatibility

def predict_all_lgbm(results: dict, df_feat: pd.DataFrame, feat_cols=None):
    """
    - If feat_cols is a list: use it for all targets (legacy behavior).
    - If feat_cols is a dict: use per-target feature lists.
    - If feat_cols is None: use the feature list stored in results[t]['feat_cols'].
    """
    if isinstance(feat_cols, dict):
        preds = {
            t: res["model"].predict(df_feat[feat_cols.get(t, res.get("feat_cols", []))].values)
            for t, res in results.items()
        }
    else:
        if isinstance(feat_cols, list) and len(feat_cols) > 0:
            X = df_feat[feat_cols].values
            preds = {t: res["model"].predict(X) for t, res in results.items()}
        else:
            preds = {
                t: res["model"].predict(df_feat[res["feat_cols"]].values)
                for t, res in results.items()
            }
    return pd.DataFrame(preds, index=df_feat.index)



def wmae(y_true: pd.DataFrame,
         y_pred: pd.DataFrame,
         weights: dict) -> float:
    """
    Calculate the weighted MAE for each target column (ignoring NaN).
    `weights`: a dictionary like `{"Tg": 0.2, "FFV": 0.2, ...}`
    """
    props = list(weights.keys())
    w = pd.Series(weights, dtype=float)

    # Absolute error
    err = (y_pred[props] - y_true[props]).abs()

    # Weighting by column
    weighted_err = err.mul(w, axis=1)

    # "Total effective weight" per row (sum only columns that are not NaN)
    effective_w = (~err.isna()).mul(w, axis=1).sum(axis=1)

    # wMAE per row â†’ average over all rows
    row_wmae = weighted_err.sum(axis=1) / effective_w
    return float(row_wmae.mean())


def overall_wmae_with_zero_ignored(y_true_df: pd.DataFrame,
                                   y_pred_df: pd.DataFrame,
                                   weights: dict) -> float:
    """
    Replace cells with a true value of 0 with NaN before calculating wMAE.
    """
    y_true_masked = y_true_df.copy()
    for col in weights.keys():
        if col in y_true_masked.columns:
            y_true_masked.loc[y_true_masked[col] == 0, col] = np.nan
    return wmae(y_true_masked, y_pred_df, weights)


results, feature_cols_list = train_all_lgbm(df_feat.iloc[:,:-42])

pred_train = predict_all_lgbm(results, df_feat, feature_cols_list)

weights = {"Tg":0.2,"FFV":0.2,"Tc":0.2,"Density":0.2,"Rg":0.2}
overall = overall_wmae_with_zero_ignored(df_feat[TARGETS], pred_train[TARGETS], weights)
print("Overall wMAE:", overall)


results, feature_cols_list = train_all_lgbm(df_feat)

pred_train = predict_all_lgbm(results, df_feat, feature_cols_list)

overall = overall_wmae_with_zero_ignored(df_feat[TARGETS], pred_train[TARGETS], weights)
print("Overall wMAE:", overall)


from typing import List, Tuple

def _count_overlapping_regex(text: str, pattern: str) -> int:
    """Count overlapping matches using lookahead."""
    return sum(1 for _ in re.finditer(f"(?={pattern})", text))

def add_smiles_pattern_counts(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    extra_patterns: List[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Add counts of chemically meaningful SMILES units by regex.
    - Patterns are case-sensitive to preserve SMILES semantics (e.g., C vs c).
    - Overlapping matches are counted (lookahead).
    """
    s = df[smiles_col].fillna("")

    # --- Base patterns (name, regex) ---
    # Note: Keep concise but chemically useful. Extend as needed.
    patterns: List[Tuple[str, str]] = [
        # Atom tokens (multi-char first to avoid splitting)
        ("tok_Cl", r"Cl"),
        ("tok_Br", r"Br"),
        ("tok_Si", r"Si"),
        ("tok_Se", r"Se"),
        ("tok_Na_bracket", r"\[Na\]"),

        # Bracketed atoms / charges
        ("unit_bracket_atom", r"\[[^\]]+\]"),
        ("unit_formal_charge", r"\[[^\]]*[+\-]\d*[^\]]*\]"),

        # Aromatic vs aliphatic atoms
        ("atom_arom_c", r"\bc\b"),   # single aromatic carbon token
        ("atom_arom_n", r"\bn\b"),
        ("atom_arom_o", r"\bo\b"),
        ("atom_arom_s", r"\bs\b"),
        ("atom_C", r"\bC\b"),
        ("atom_N", r"\bN\b"),
        ("atom_O", r"\bO\b"),
        ("atom_S", r"\bS\b"),
        ("atom_P", r"\bP\b"),
        ("atom_F", r"\bF\b"),
        ("atom_I", r"\bI\b"),
        ("atom_B", r"\bB\b"),

        # Bonds / stereochemistry
        ("bond_double", r"="),
        ("bond_triple", r"#"),
        ("stereo_forward", r"/"),
        ("stereo_backward", r"\\"),
        
        # Branches and ring closures
        ("branch_open", r"\("),
        ("branch_close", r"\)"),
        ("ring_digit", r"[0-9]"),

        # Common substructures / functional groups
        ("fg_carbonyl", r"C=O"),
        ("fg_ester", r"C\(=O\)O"),
        ("fg_amide", r"N[Cn]?\(=O\)"),          # N(=O) or NC(=O)
        ("fg_uera_like", r"N[Cn]?C\(=O\)N"),    # urea/biuret-like (loose)
        ("fg_nitrile", r"C#N"),
        ("fg_alkyne", r"C#C"),
        ("fg_sulfonyl", r"S\(=O\)\(=O\)"),
        ("fg_sulfonate", r"O S\(=O\)\(=O\)"),   # will survive spaces removed below
        ("fg_phosphoryl", r"P\(=O\)"),
        ("fg_quat_ammonium", r"\[N\+"),         # e.g., [N+](...)
        ("fg_silyl", r"\[?Si\]?"),
        
        # Vinylene with E/Z annotations (common in your samples)
        ("unit_vinylenefwd", r"/C=C/"),
        ("unit_vinylenemix1", r"/C=C\\"),
        ("unit_vinylenemix2", r"\\C=C/"),
        ("unit_vinylenebwd", r"\\C=C\\"),
        ("unit_vinyl_generic", r"C=C"),         # fallback

        # Aromatic rings / phenyl motifs (heuristics)
        ("ring_ph_phenyl", r"c1ccccc1"),
        ("ring_arom_any", r"c1[^1]*1"),         # any 6-membered aromatic annotated as 1...1
        ("ring_bicyclic_arom", r"c1ccc2[^2]*2[^1]*1"),

        # Tert-butyl / isopropyl-like branching (common bulky substituents)
        ("frag_tBu_like", r"C\(C\)\(C\)C"),
        ("frag_iPr_like", r"C\(C\)C"),

        # Simple ethers and linkers (loose heuristics)
        ("link_ether_OC", r"O[C]"),
        ("link_ether_CO", r"C[O]"),
        ("link_oxyethylene", r"(OCC){2,}"),     # repeating OCC (PEG-like)
    ]

    # Make a sanitized copy without spaces to ease some patterns
    s_nospace = s.str.replace(r"\s+", "", regex=True)

    # Adjust a couple of patterns that were written with a space placeholder
    # to help readability above.
    patterns = [
        (name, pat.replace(" ", "")) for (name, pat) in patterns
    ]

    if extra_patterns:
        patterns.extend(extra_patterns)

    feat = pd.DataFrame(index=df.index)
    for name, pat in patterns:
        # Count on space-less string to be robust
        feat[f"cnt_{name}"] = s_nospace.apply(lambda x: _count_overlapping_regex(x, pat))

    return pd.concat([df, feat], axis=1)



df_feat = pd.concat([train, features], axis=1)
df_feat = add_smiles_pattern_counts(df_feat, smiles_col="SMILES")
df_feat = df_feat.drop("id",axis=1)
print(df_feat.shape)
df_feat.head()


results, feature_cols_list = train_all_lgbm(df_feat)

pred_train = predict_all_lgbm(results, df_feat, feature_cols_list)

overall = overall_wmae_with_zero_ignored(df_feat[TARGETS], pred_train[TARGETS], weights)
print("Overall wMAE:", overall)


# ---- Retrieve the importance DataFrame (single target) ----
def lgb_importance_df(results, feature_cols_list, target, importance_type="gain"):
    model = results[target]["model"]
    if importance_type == "gain":
        imp = model.booster_.feature_importance(importance_type="gain")
    else:
        imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_cols_list, "importance": imp})
    return df.sort_values("importance", ascending=False)

# ---- Plot the top-k items (single target)ï¼‰ ----
def plot_topk_importance(results, feature_cols_list, target, k=10, importance_type="gain"):
    imp_df = lgb_importance_df(results, feature_cols_list, target, importance_type)
    top = imp_df.head(k).iloc[::-1]
    sort_list = imp_df.iloc[::-1]["feature"].to_list()
    plt.figure(figsize=(10, max(5, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel(f"Importance ({importance_type})")
    plt.title(f"{target} - Top {min(k, len(imp_df))} LightGBM feature importances")
    plt.tight_layout()
    plt.show()
    return sort_list


Tg_list=plot_topk_importance(results, feature_cols_list, target="Tg", k=10, importance_type="gain")


FFV_list=plot_topk_importance(results, feature_cols_list, target="FFV", k=10, importance_type="gain")


Tc_list=plot_topk_importance(results, feature_cols_list, target="Tc", k=10, importance_type="gain")


Density_list=plot_topk_importance(results, feature_cols_list, target="Density", k=10, importance_type="gain")


Rg_list=plot_topk_importance(results, feature_cols_list, target="Rg", k=10, importance_type="gain")


df_feat = pd.concat([train, features], axis=1)
df_feat = add_smiles_pattern_counts(df_feat, smiles_col="SMILES")
df_feat = df_feat.drop("id",axis=1)
feat_map = {"Tg": Tg_list[58:], "FFV": FFV_list[58:], "Tc": Tc_list[58:], "Density": Density_list[58:], "Rg": Rg_list[58:]}
results, dummy  = train_all_lgbm(df_feat,feat_cols_by_target=feat_map)

pred_train = predict_all_lgbm(results, df_feat)

overall = overall_wmae_with_zero_ignored(df_feat[TARGETS], pred_train[TARGETS], weights)
print("Overall wMAE:", overall)


df_feat = pd.concat([train, features], axis=1)
df_feat = add_smiles_pattern_counts(df_feat, smiles_col="SMILES")
df_feat = df_feat.drop("id",axis=1)
feat_map = {"Tg": Tg_list[55:], "FFV": FFV_list[30:], "Tc": Tc_list[30:], "Density": Density_list[45:], "Rg": Rg_list[45:]}
results, dummy  = train_all_lgbm(df_feat,feat_cols_by_target=feat_map)

pred_train = predict_all_lgbm(results, df_feat)

overall = overall_wmae_with_zero_ignored(df_feat[TARGETS], pred_train[TARGETS], weights)
print("Overall wMAE:", overall)


df_feat = pd.concat([train, features], axis=1)
df_feat = add_smiles_pattern_counts(df_feat, smiles_col="SMILES")
df_feat = df_feat.drop("id",axis=1)
results, dummy  = train_all_lgbm(df_feat)

pred_train = predict_all_lgbm(results, df_feat)

overall = overall_wmae_with_zero_ignored(df_feat[TARGETS], pred_train[TARGETS], weights)
print("Overall wMAE:", overall)


test_df = pd.read_csv(path+"test.csv")
test_df


submit_df = pd.read_csv(path+"sample_submission.csv")
submit_df


"""Prerequisites
featurize_smiles()  â€” the same feature-generation function used during training
results             â€” dictionary of models after single-task training
feature_cols        â€” list of numeric feature names extracted during training
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
"""

# --- 1. SMILES â†’ expand into features ---
feat_test = test_df["SMILES"].apply(featurize_smiles)
test_feat = pd.concat([test_df, feat_test], axis=1)
test_feat = add_smiles_pattern_counts(test_feat, smiles_col="SMILES")

# --- 2. Ensure the feature columns present during training (add zeros for any missing ones) ---
for col in feature_cols_list:
    if col not in test_feat.columns:
        test_feat[col] = 0

# --- 3. Inference ---
preds = predict_all_lgbm(results, test_feat)   # DataFrame (len(test) Ã— 5)

# --- 4. Overwrite the TARGET column in the submission ---
submit_df.loc[:, TARGETS] = preds[TARGETS].values

# --- 5. Save as CSV for submission ---
submit_df.to_csv("submission.csv", index=False)

submit_df


