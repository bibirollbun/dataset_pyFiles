!pip install /kaggle/input/rdkit-2025-3-3/rdkit-2025.3.5-cp311-cp311-manylinux_2_28_x86_64.whl


pip freeze > requirements.txt


# ==================================================
# Core Python
# ==================================================
import gc
import pickle
import warnings

# Config warnings/pandas
warnings.filterwarnings("ignore")

# ==================================================
# Data & Utils
# ==================================================
import numpy as np
import pandas as pd
import polars as pl

pd.set_option('display.max_columns', None)

# ==================================================
# Visualization
# ==================================================
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# Machine Learning
# ==================================================
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold

# ==================================================
# Chemistry (RDKit)
# ==================================================
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdmolops,
    rdMolDescriptors
)
from rdkit.Chem import rdFingerprintGenerator as rfgs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ==================================================
# Graphs
# ==================================================
import networkx as nx



class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5
    PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'

train = pd.read_csv(CFG.PATH + 'train.csv')
test = pd.read_csv(CFG.PATH + 'test.csv')

def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))


# https://www.kaggle.com/datasets/minatoyukinaxlisa/tc-smiles
data_tc = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})

data_tg_pid = pd.read_csv('/kaggle/input/tg-of-polymer-dataset/Tg_SMILES_class_pid_polyinfo_median.csv', usecols=['SMILES', 'Tg'])

# https://springernature.figshare.com/articles/dataset/dataset_with_glass_transition_temperature/24219958?file=42507037
data_tg2 = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})

# https://www.sciencedirect.com/science/article/pii/S2590159123000377#ec0005
data_tg3 = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
data_tg3['Tg'] = data_tg3['Tg'] - 273.15

# https://github.com/Duke-MatSci/ChemProps
data_dnst = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
data_dnst['SMILES'] = data_dnst['SMILES'].apply(lambda s: make_smile_canonical(s))
data_dnst = data_dnst[(data_dnst['SMILES'].notnull())&(data_dnst['Density'].notnull())&(data_dnst['Density'] != 'nylon')]
data_dnst['Density'] = data_dnst['Density'].astype('float64')
data_dnst['Density'] -= 0.118

def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Make priority target value from competition's df
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)

    # Imput missing values for competition's SMILES
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]
    
    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES: {len(unique_smiles_extra)}')
    return df_train

train = add_extra_data(train, data_tc, 'Tc')
train = add_extra_data(train, data_tg2, 'Tg')
train = add_extra_data(train, data_tg3, 'Tg')
train = add_extra_data(train, data_tg_pid, 'Tg')
train = add_extra_data(train, data_dnst, 'Density')

print('\n'*3, '--- SMILES for training ---', )
for t in CFG.TARGETS:
    print(f'"{t}": {len(train[train[t].notnull()])}')


train_default = train.copy()


SUBSTITUENTS = ["F", "Cl", "C#N", "CF3"]
SUBS_FLEXIBLE = [
    "CH3",   
    "OCH3",
    "O",      
]
DELTA_RG = {
    "F": 0.1,
    "Cl": 0.2,
    "C#N": 0.3,
    "CF3": 0.7,
    "CH3": -0.25,
    "OCH3": -0.35,
    "O": -0.45
}

DELTA_TC = {
    "F": 0.005,
    "Cl": 0.005,
    "C#N": 0.01,
    "CF3": 0.01,
}

def generate_variations(smiles, substituents=SUBSTITUENTS):
    variations = []
    try:
        for sub in substituents:
            # case 1: aromatic ring "cc"
            if "cc" in smiles:
                new = smiles.replace("cc", f"c({sub})c", 1)
                if Chem.MolFromSmiles(new):
                    variations.append((new, sub))

            # case 2: aliphatic carbon "CC"
            if "CC" in smiles:
                new = smiles.replace("CC", f"C({sub})C", 1)
                if Chem.MolFromSmiles(new):
                    variations.append((new, sub))
    except Exception as e:
        print("Error:", e)
    return variations


def expand_train_rg(train, sub=SUBSTITUENTS, smiles_col="SMILES", rg_col="Rg"):
    new_rows = []
    for _, row in train.iterrows():
        base_smi = row[smiles_col]
        rg_val = row[rg_col]
        if pd.notna(base_smi) and pd.notna(rg_val):
            for var, sub in generate_variations(base_smi, sub):
                new_row = row.copy()
                new_row[smiles_col] = var
                # Adjust Rg based on heuristic
                noise = np.random.normal(0, 0.05)
                new_row[rg_col] = rg_val + DELTA_RG.get(sub, 0.0) + noise
                new_rows.append(new_row)
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        train = pd.concat([train, df_new], ignore_index=True)
    return train


def expand_train_tc(train, sub=SUBSTITUENTS, smiles_col="SMILES", tc_col="Tc"):
    new_rows = []
    for _, row in train.iterrows():
        base_smi = row[smiles_col]
        tc_val = row[tc_col]
        for var, sub in generate_variations(base_smi):
            # copy all row columns to preserve
            new_row = row.copy()
            new_row[smiles_col] = var
            noise = np.random.normal(0, 0.005)
            new_row[tc_col] = tc_val + DELTA_TC.get(sub, 0.0) + noise
            new_rows.append(new_row)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        train = pd.concat([train, df_new], ignore_index=True)
    return train


# ================== Exemplo ==================
train_tc = pd.DataFrame({
    "SMILES": [
        "*c1ccc2cc(*)ccc2c1",   # naftaleno substituÃ­do
        "*c1cccc2c(*)cccc12",
        "*c1ccc2ccc3c(*)cc(C#C)c4ccc1c2c34",
        "*c1ccc(*)c2ccccc12",
        "*CCCCCCCCCCCCCCCCCCCCOC(=O)CCCCCCCC(=O)O*",
        "*/C=C/c1cc(OCCCCCC)c(*)cc1OC",
        "*CCCCOC(=O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O*",
        "*c1ccc2c(c1)SC1=Nc3cc(-c4ccc5c(c4)N=C4Sc6cc(*)ccc6N=C4N5)ccc3NC1=N2"
        
    ],
    "Tc": [0.800, 0.685,0.582,0.571,0.50,0.524,0.494,0.506]
})

train_rg_value_10 = pd.DataFrame({
    "SMILES": [
        "*CC(*)c1cc(-c2ccc(C(=O)OC(C)CCCCCC)cc2)ccc1-c1ccc(C(=O)OC(C)CCCCCC)cc1",   # naftaleno substituÃ­do
        "*CC(*)c1ccc(-c2ccc3c(c2)C(CCCCCC)(CCCCCC)c2ccccc2-3)cc1",
        "*c1c(-c2ccccc2)c(-c2ccccc2)c(*)c2cc(C3(c4ccc(C#Cc5ccccc5)c(C#Cc5ccccc5)c4)c4ccccc4-c4ccccc43)ccc12",
        "*c1c(-c2ccccc2)c(-c2ccccc2)c(*)c2cc(C(c3ccc(C#Cc4ccccc4)c(C#Cc4ccccc4)c3)(C(F)(F)F)C(F)(F)F)ccc12"
        
    ],
    "Rg": [10, 10.85, 11.54, 12.31],
})

train_rg_q85 = pd.DataFrame({
    "SMILES": [
        "*c1ccc(-c2ccc3c(c2)C(CCCCCCC#N)(CCCCCCC#N)c2cc(*)ccc2-3)cc1",
        "*CCc1ccc(-c2ccc(*)cc2)cc1",
        "*c1ccc(-c2ccc3c(c2)C(CCCCCC)(CCCCCC)c2cc(*)ccc2-3)cc1",
        "*c1ccc2c(c1)SC1=Nc3cc(-c4ccc5c(c4)N=C4Sc6cc(*)ccc6N=C4N5)ccc3NC1=N2",
        "*CCCCCCCCCCSCCCCCCS*",
        "*CCCCCCCCCCOc1ccc(OC(=O)c2ccc(OCCCCCCOc3ccc(C(=O)Oc4ccc(O*)cc4)cc3)cc2)cc1",
        "*c1ccc(Cc2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2)cc1",
        "*c1ccc(-c2ccc3c(c2)C(CCCCCCBr)(CCCCCCBr)c2cc(*)ccc2-3)cc1",
        "*/C=C/*",
        "*Oc1ccc(C(C)(C)c2ccc(Oc3ccc(C(=O)Nc4ccc(-c5ccc(NC(=O)c6ccc(*)cc6)cc5C(F)(F)F)c(C(F)(F)F)c4)cc3)cc2)cc1",
        "*Oc1ccc(-c2ccc(-c3cc(-c4ccccc4)c(-c4ccc(-c5ccc(OC(=O)c6ccc(C(*)=O)cc6-c6ccccc6)cc5)cc4)c(-c4ccccc4)c3-c3ccccc3)cc2)cc1",
        "*c1ccc2ccc3c(*)cc(C#CC=C)c4ccc1c2c34",
        "*Oc1ccc(NC(=O)CCCCCCCCCC(=O)Nc2ccc(*)cc2)cc1",
        "*c1ccc(C(C)(C)c2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2)cc1",
        "*SC(*)(F)F",
        "*CCCCCCCCCCNC(=O)CCCCCCCCCCCCCCCCCCC(=O)N*"
    ],
    "Rg": [
        28.68, 30.04, 27.96, 27.53, 28.27, 26.56, 
        29.70, 29.40, 34.67, 27.69, 27.64, 34.49, 27.22,25.11,24.17,24.81
    ]
})


# Expande sÃ³ o train2
train_tc_exp = expand_train_tc(train_tc, sub=SUBSTITUENTS)
train_rg_q85_exp = expand_train_rg(train_rg_q85, sub=SUBSTITUENTS)
train_rg_10_exp = expand_train_rg(train_rg_value_10, sub=SUBS_FLEXIBLE)

# Agora concatena com o train original
train = pd.concat([train, train_tc_exp], ignore_index=True)
train = pd.concat([train, train_rg_q85_exp], ignore_index=True)
train = pd.concat([train, train_rg_10_exp], ignore_index=True)




def _star_indices(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol()=='*']

def graph_star_distance(mol):
    stars = _star_indices(mol)
    if len(stars)!=2: return None
    dmat = rdmolops.GetDistanceMatrix(mol)
    return int(dmat[stars[0], stars[1]])



def rings_between_stars(mol):
    stars = _star_indices(mol)
    if len(stars)!=2: return None
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    # quantos anÃ©is distintos interceptam qualquer caminho curto aproximado
    # proxy simples: anÃ©is que contÃªm pelo menos um dos â€œ\*â€� ou um vizinho no menor caminho
    path = Chem.rdmolops.GetShortestPath(mol, stars[0], stars[1])
    return sum(1 for r in rings if any(a in r for a in path))

def ecfp_similarity_stars(mol, radius=2, nBits=1024):
    stars = _star_indices(mol)
    if len(stars)!=2: return None
    gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    # Fingerprint apenas do entorno de cada estrela (usar invariantes baseando-se em Morgan com â€œfromAtomsâ€�)
    fp0 = gen.GetFingerprint(mol, fromAtoms=[stars[0]])
    fp1 = gen.GetFingerprint(mol, fromAtoms=[stars[1]])
    # Tanimoto manual
    on0 = set(fp0.GetOnBits()); on1 = set(fp1.GetOnBits())
    inter = len(on0 & on1); union = len(on0 | on1)
    return inter/union if union>0 else 0.0

def peri_flag(mol):
    """HeurÃ­stica para naftaleno: dois * em Ã¡tomos alfa de anÃ©is fundidos com distÃ¢ncia topolÃ³gica curta e
       muitos vizinhos comuns â†’ provÃ¡vel 'peri' (1,8-like)."""
    stars = _star_indices(mol)
    if len(stars)!=2: return 0
    d = graph_star_distance(mol)
    if d is None: return 0
    # peri costuma ter d curto (3â€“4) + vizinhos prÃ³ximos com interseÃ§Ã£o alta
    nbs0 = set([n.GetIdx() for n in mol.GetAtomWithIdx(stars[0]).GetNeighbors()])
    nbs1 = set([n.GetIdx() for n in mol.GetAtomWithIdx(stars[1]).GetNeighbors()])
    close = len(nbs0 & nbs1) > 0
    return int(d<=4 and close)

def radical_distance(mol):
    try:
        radical_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
        if len(radical_idxs) < 2:
            return 0
        path = rdmolops.GetShortestPath(mol, radical_idxs[0], radical_idxs[1])
        return len(path) - 1
    except:
        return 0

def extract_tc_positional_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return {
            "graph_star_distance": -1,
            "rings_between_stars": -1,
            "ecfp_similarity_stars": 0.0,
            "peri_flag": 0,
            "radical_distance": 0,
        }

    d1 = graph_star_distance(mol)
    rb = rings_between_stars(mol)
    ec = ecfp_similarity_stars(mol)
    pf = peri_flag(mol)
    rd = radical_distance(mol)

    return {
        "graph_star_distance": d1 if d1 is not None else -1,
        "rings_between_stars": rb if rb is not None else -1,
        "ecfp_similarity_stars": ec if ec is not None else 0.0,
        "peri_flag": pf if pf is not None else 0,
        "radical_distance": rd if rd is not None else 0,
    }

# --- exemplo de aplicaÃ§Ã£o no DataFrame ---
# supondo que seu DataFrame se chame train e a coluna de SMILES se chame "SMILES"
feature_dicts = train["SMILES"].apply(extract_tc_positional_features)
feature_dicts2 = test["SMILES"].apply(extract_tc_positional_features)

# transformar a sÃ©rie de dicionÃ¡rios em DataFrame
feature_df = pd.DataFrame(feature_dicts.tolist())
feature_df2 = pd.DataFrame(feature_dicts2.tolist())

# concatenar com o DataFrame original
train = pd.concat([train.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
test = pd.concat([test.reset_index(drop=True), feature_df2.reset_index(drop=True)], axis=1)




RDLogger.DisableLog('rdApp.*')
import time

start = time.time() 


def smiles_to_3d_features(smiles: str):
    """
    Gera descritores 3D 'seguros' para um SMILES.
    Se o SMILES contiver '*', retorna NaN (nÃ£o calcula).
    """

    smiles = smiles.replace("*", "C")

    if not isinstance(smiles, str) or "*" in smiles:
        return {
            "Asphericity": np.nan,
            "Eccentricity": np.nan,
            "InertialShapeFactor": np.nan,
            "NPR1": np.nan,
            "NPR2": np.nan,
            "SpherocityIndex": np.nan
        }

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Mol invÃ¡lido")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        feats = {
            "Asphericity": rdMolDescriptors.CalcAsphericity(mol),
            "Eccentricity": rdMolDescriptors.CalcEccentricity(mol),
            "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor(mol),
            "NPR1": rdMolDescriptors.CalcNPR1(mol),
            "NPR2": rdMolDescriptors.CalcNPR2(mol),
            "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol)
        }
        return feats
    except Exception:
        return {
            "Asphericity": np.nan,
            "Eccentricity": np.nan,
            "InertialShapeFactor": np.nan,
            "NPR1": np.nan,
            "NPR2": np.nan,
            "SpherocityIndex": np.nan
        }

def add_3d_features(df, smiles_col="SMILES"):
    features = df[smiles_col].apply(smiles_to_3d_features)
    features_df = pd.DataFrame(features.tolist(), index=df.index)

    # ğŸ”¥ forÃ§a dtype float
    features_df = features_df.astype(float)

    return pd.concat([df, features_df], axis=1)


train = add_3d_features(train)
test = add_3d_features(test)

end = time.time()  # finaliza o temporizador
print(f"Tempo de execuÃ§Ã£o: {end - start:.2f} segundos")


def ring_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return pd.Series({
                "max_ring_size": 0,
                "min_ring_size": 0,
                "atoms_in_rings_ratio": 0.0,
                "atoms_outside_rings": 0,
                **{f"has_ring_{k}": 0 for k in range(3, 11)}
            })

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        ring_sizes = [len(r) for r in atom_rings] if atom_rings else []

        # max/min tamanho do ciclo
        max_ring_size = max(ring_sizes) if ring_sizes else 0
        min_ring_size = min(ring_sizes) if ring_sizes else 0

        # presenÃ§a de ciclos 3â€“10 membros
        ring_flags = {f"has_ring_{k}": int(k in ring_sizes) for k in range(1, 11)}

        # proporÃ§Ã£o de Ã¡tomos em ciclos
        atoms_in_rings = sum(len(r) for r in atom_rings) if atom_rings else 0
        total_atoms = mol.GetNumAtoms()
        atoms_in_rings_ratio = atoms_in_rings / total_atoms if total_atoms > 0 else 0.0

        # Ã¡tomos fora de ciclos
        atoms_outside_rings = total_atoms - atoms_in_rings

        return pd.Series({
            "max_ring_size": max_ring_size,
            "min_ring_size": min_ring_size,
            "atoms_in_rings_ratio": atoms_in_rings_ratio,
            "atoms_outside_rings": atoms_outside_rings,
            **ring_flags
        })
    except:
        return pd.Series({
            "max_ring_size": 0,
            "min_ring_size": 0,
            "atoms_in_rings_ratio": 0.0,
            "atoms_outside_rings": 0,
            **{f"has_ring_{k}": 0 for k in range(3, 11)}
        })

# Exemplo de aplicaÃ§Ã£o no DataFrame
#train_feats = train["SMILES"].apply(ring_features)
#test_feats = test["SMILES"].apply(ring_features)

# Anexa no dataset
#train = pd.concat([train, train_feats], axis=1)
#test = pd.concat([test, test_feats], axis=1)



train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
train = train.drop_duplicates(subset='SMILES', keep='first').reset_index(drop=True)


def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])', 
    ]
    
    # Check for any bad patterns
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None
    except:
            return None
    
    # If RDKit not available, return cleaned SMILES
    return smiles

# Clean and validate all SMILES
print("ğŸ”„ Cleaning and validating SMILES...")
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)


train.loc[train["Tc"] > 1, "Tc"] = 0.8
train.loc[train["Rg"] > 31, "Rg"] = np.nan



# Lista de colunas inÃºteis (NANs, constantes, altamente correlacionadas)
useless_cols = [    
    # Nan data
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    
    # Constant data
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',

    # High correlated data >0.95
    'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1',
    'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v',
    'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
]


# FunÃ§Ã£o para calcular descritores
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]

# FunÃ§Ã£o para extrair caracterÃ­sticas do grafo molecular
def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    
    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    for i in range(fp.GetNumBits()):
        key = f"MorganFP_{i}"
        if key not in graph_feats:
            graph_feats[key] = []
        graph_feats[key].append(int(fp[i]))

    gen = rfgs.GetAtomPairGenerator(fpSize=2048)  # vocÃª pode ajustar min/max distance se quiser
    fp = gen.GetFingerprint(mol)  # ExplicitBitVect
    
    for i in range(fp.GetNumBits()):
        key = f"AtomPairFP_{i}"
        if key not in graph_feats:
            graph_feats[key] = []
        graph_feats[key].append(int(fp[i]))

    gen = rfgs.GetTopologicalTorsionGenerator(fpSize=2048)
    fp = gen.GetFingerprint(mol)  # ExplicitBitVect
    for i in range(fp.GetNumBits()):
        key = f"torsion_{i}"
        if key not in graph_feats:
            graph_feats[key] = []
        graph_feats[key].append(int(fp[i]))

# PrÃ©-processamento completo
def preprocessing(df):
    global desc_names
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]

    start = time.time() 
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES'].to_list()]
    end = time.time()  # finaliza o temporizador
    print(f"DESCLIST Tempo de execuÃ§Ã£o: {end - start:.2f} segundos")

    start = time.time() 
    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    for smile in df['SMILES']:
         compute_graph_features(smile, graph_feats)
        
    result = pd.concat(
        [
            pd.DataFrame(descriptors, columns=desc_names),
            pd.DataFrame(graph_feats)
        ],
        axis=1
    )

    end = time.time()  # finaliza o temporizador
    print(f" COMPUTE GRAPH Tempo de execuÃ§Ã£o: {end - start:.2f} segundos")

    result = result.replace([-np.inf, np.inf], np.nan)
    return result

train_extra = preprocessing(train)
test_extra = preprocessing(test)

# === Aplica VarianceThreshold (remove colunas com variÃ¢ncia zero) ===
selector = VarianceThreshold(threshold=0.0)
train_selected = selector.fit_transform(train_extra)
selected_cols = train_extra.columns[selector.get_support()]
test_selected = selector.transform(test_extra)

# === Converte para DataFrame com colunas preservadas ===
train_selected_df = pd.DataFrame(train_selected, columns=selected_cols, index=train.index)
test_selected_df = pd.DataFrame(test_selected, columns=selected_cols, index=test.index)

# === Junta com o DataFrame original ===
train = pd.concat([train, train_selected_df], axis=1)
test = pd.concat([test, test_selected_df], axis=1)

# SeleÃ§Ã£o de features por target
all_features = train.columns[7:].tolist()
features = {}

for target in CFG.TARGETS:
    const_descs = []
    for col in train.columns.drop(CFG.TARGETS):
        if train[train[target].notnull()][col].nunique() == 1:
            const_descs.append(col)

    valid_features = [f for f in all_features if f not in const_descs]
    features[target] = valid_features



pairs = [
    ("TPSA", "MolWt"),
    ("NumHDonors", "NumHAcceptors"),
]

new_features = []

for f1, f2 in pairs:
    if f1 in train.columns and f2 in train.columns:
        colname = f"{f1}_x_{f2}"
        train[colname] = train[f1] * train[f2]
        test[colname] = test[f1] * test[f2]
        new_features.append(colname)

# adiciona ao dicionÃ¡rio features
for target in CFG.TARGETS:
    if target in features:
        features[target] = features[target] + new_features




models = {t: [] for t in CFG.TARGETS}


def mae(y_true, y_pred):
    return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)

def base_params_f(alpha):
    base_params = {
        'device_type': 'cpu',
        'n_estimators': 10000,
        'objective': 'quantile',
        'alpha': alpha,
        'metric': 'mae',
        'verbosity': -1,

        'num_leaves': 50,
        'min_data_in_leaf': 2,
        'learning_rate': 0.01,
        'max_bin': 255,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l1': 2,
        'lambda_l2': 2,
    }
    return base_params

for target in CFG.TARGETS:
    print(f'\n\nTARGET {target}')
    train_part = train[train[target].notnull()].reset_index(drop=True)
    train[f'{target}_pred'] = 0
    test[target] = 0
    oof_lgb = np.zeros(len(train_part))
    scores = []
    
    kf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    for i, (trn_idx, val_idx) in enumerate(kf.split(train_part, train_part[target])):
        print(f"\n--- Fold {i+1} ---")
        
        x_trn = train_part.loc[trn_idx, features[target]]
        y_trn = train_part.loc[trn_idx, target]
        x_val = train_part.loc[val_idx, features[target]]
        y_val = train_part.loc[val_idx, target]

        if target in ['Tc', 'Density', 'FFV']:
            base_params = base_params_f(0.5)   # mediana
        else:
            base_params = base_params_f(0.85) 
        model_lgb = lgb.LGBMRegressor(**base_params)
        model_lgb.fit(
            x_trn, y_trn,
            eval_set=[(x_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=300,
                    verbose=False,
                ),
                lgb.log_evaluation(2500)
            ],
        )

        with open(f'/kaggle/working/lgb_{target}_fold_{i}.pkl', 'wb') as f:
            pickle.dump(model_lgb, f)

        models[target].append(model_lgb)

        val_preds = model_lgb.predict(x_val, num_iteration=model_lgb.best_iteration_)
        score = mae(y_val, val_preds)
        scores.append(score)
        print(f'MAE: {np.round(score, 5)}')
        
        oof_lgb[val_idx] = val_preds
        test[target] += model_lgb.predict(
            test[features[target]], 
            num_iteration=model_lgb.best_iteration_
        ) / CFG.FOLDS

    train.loc[train[target].notnull(), f'{target}_pred'] = oof_lgb

    print(f'\nMean MAE: {np.round(np.mean(scores), 5)}')
    print(f'Std MAE: {np.round(np.std(scores), 5)}')
    print('-'*30)

with open("/kaggle/working/features.pkl", "wb") as f:
    pickle.dump(features, f)


with open("/kaggle/working/test.pkl", "wb") as f:
    pickle.dump(test, f)




# Dictionary to store top 100 features for each target
top_features = {}

for t in CFG.TARGETS:
    if t not in models or len(models[t]) == 0:
        continue

    # Collect feature importances from all folds
    importances = pd.DataFrame()
    for m in models[t]:
        temp = pd.DataFrame({
            "feature": m.feature_name_,
            "importance": m.feature_importances_
        })
        importances = pd.concat([importances, temp])

    # Average importance across folds
    importances = (
        importances.groupby("feature", as_index=False)
        .mean()
        .sort_values("importance", ascending=False)
    )

    # Select the top 100 features for this target
    best_100 = importances["feature"].head(100).tolist()
    top_features[t] = best_100

# Save to pickle
with open("/kaggle/working/features_top100.pkl", "wb") as f:
    pickle.dump(top_features, f)

print("Top 100 features per target saved in features_top100.pkl")



print("Number of columns in train:", train.shape[1])
print("Number of rows in train:", train.shape[0])


train['starts_with_star'] = train['SMILES'].str.startswith('*')
test['starts_with_star'] = test['SMILES'].str.startswith('*')

for t in CFG.TARGETS:
    mask = train[t].notnull()
    
    preds = train.loc[mask, f'{t}_pred']
    vals = train.loc[mask, t]
    color_flag = train.loc[mask, 'starts_with_star']
    
    line_min = min(preds.min(), vals.min())
    line_max = max(preds.max(), vals.max())

    # Scatterplot with conditional color
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=preds, y=vals, hue=color_flag, palette={True: 'orange', False: 'blue'}, alpha=0.6)
    
    # Ideal line
    plt.plot(
        [line_min, line_max], 
        [line_min, line_max], 
        color='red', 
        linewidth=2, 
        linestyle='dashed'
    )

    plt.xlabel(f'Predicted {t}')
    plt.ylabel(f'True {t}')
    plt.title(f'Pred vs True for {t}')
    plt.legend(title='Starts with *', labels=['No', 'Yes'])
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()



# ==================================================
# Feature groups
# ==================================================
positional_features = [
    "graph_star_distance",
    "radical_distance",
    "rings_between_stars",
    "ecfp_similarity_stars",
    "peri_flag",
]

features_3d = [
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "SpherocityIndex",
]

feature_groups = {
    "3D Features": features_3d,
    "Positional Features": positional_features,
}

# ==================================================
# Feature importance (average across folds)
# ==================================================
for t in CFG.TARGETS:
    if t not in models or len(models[t]) == 0:
        continue

    try:
        # 1. Collect feature importances from all folds
        importances = pd.DataFrame()
        for m in models[t]:
            temp = pd.DataFrame({
                "feature": m.feature_name_,
                "importance": m.feature_importances_
            })
            importances = pd.concat([importances, temp])

        # 2. Average importance per feature
        importances = (
            importances.groupby("feature", as_index=False)
            .mean()
            .sort_values("importance", ascending=False)
        )

        # 3. Prepare smaller subplots (2x2 â†’ agora sÃ³ 1x2)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"{t} - Average Feature Importance Across Folds", fontsize=12, weight="bold")

        # --- Top 10 overall (all features)
        top10 = importances.head(10)
        sns.barplot(data=top10, x="importance", y="feature", ax=axes[0], palette="mako")
        axes[0].set_title("Top 10 Overall", fontsize=10)

        # --- Group-specific plots (agora sÃ³ 3D e Positional)
        for ax, (group_name, group_features) in zip(axes[1:], feature_groups.items()):
            subset = (
                importances[importances["feature"].isin(group_features)]
                .sort_values("importance", ascending=False)
                .head(10)
            )
            if subset.empty:
                ax.set_visible(False)
                continue
            sns.barplot(data=subset, x="importance", y="feature", ax=ax, palette="mako")
            ax.set_title(f"{group_name} (Top 10)", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"Could not extract features for {t}: {e}")



for t in CFG.TARGETS:
    df_temp = train[train[t].notnull()].copy()

    preds = df_temp[f"{t}_pred"]
    vals = df_temp[t]

    line_min = min(preds.min(), vals.min())
    line_max = max(preds.max(), vals.max())

    sns.scatterplot(x=preds, y=vals, alpha=0.3)

    plt.plot([line_min, line_max], [line_min, line_max],
             color="black", linewidth=2, linestyle="dashed")
    plt.title(f"{t} - Predicted vs Actual")
    plt.show()



MINMAX_DICT =  {
        'Tg': [-148.0297376, 472.25],
        'FFV': [0.2269924, 0.77709707],
        'Tc': [0.0465, 0.524],
        'Density': [0.748691234, 1.840998909],
        'Rg': [9.7283551, 34.672905605],
    }
NULL_FOR_SUBMISSION = -9999

def scaling_error(labels, preds, property):
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)

def get_property_weights(labels):
    property_weight = []
    for property in MINMAX_DICT.keys():
        valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION)
        property_weight.append(valid_num)
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / property_weight)
    return (property_weight / np.sum(property_weight)) * len(property_weight)

def wmae_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []
    property_weights = get_property_weights(solution[chemical_properties])
    for property in chemical_properties:
        is_labeled = solution[property] != NULL_FOR_SUBMISSION
        property_maes.append(scaling_error(solution.loc[is_labeled, property], submission.loc[is_labeled, property], property))

    if len(property_maes) == 0:
        raise RuntimeError('No labels')
    return float(np.average(property_maes, weights=property_weights))

tr_solution = train[['id'] + CFG.TARGETS]
tr_submission = train[['id'] + [t + '_pred' for t in CFG.TARGETS]]
tr_submission.columns = ['id'] + CFG.TARGETS
print(f"wMAE: {round(wmae_score(tr_solution, tr_submission, row_id_column_name='id'), 5)}")


for t in CFG.TARGETS:
    for s in train_default[train_default[t].notnull()]['SMILES']:
        if s in test['SMILES'].tolist():
            test.loc[test['SMILES']==s, t] = train_default[train_default['SMILES']==s][t].values[0]


test[['id'] + CFG.TARGETS].to_csv('submission.csv', index=False)

