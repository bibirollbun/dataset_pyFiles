#!pip install --no-index --no-deps ../input/rdkit-offline/wheelhouse/pillow-11.2.1-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install --no-index --no-deps /kaggle/input/rdkit-2023-9-6/wheelhouse/rdkit-2023.9.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


!pip install --no-index --no-deps /kaggle/input/py3dmol/wheelhouse/py3dmol-2.5.0-py2.py3-none-any.whl


! pip install /kaggle/input/mordred/mordredcommunity-2.0.6-py3-none-any.whl


#! python /kaggle/input/descriptors-3d/get_mordred_descriptors_3d.py --smiles '*Nc1ccc([C@H](CCC)c2ccc(C3(c4ccc([C@@H](CCC)c5ccc(N*)cc5)cc4)CCC(CCCCC)CC3)cc2)cc1'


from itertools import product
import warnings
from sklearn.model_selection import KFold, cross_val_score
from rdkit.Chem import AllChem, rdmolops
# check others libs
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDistGeom
import py3Dmol
from rdkit.Chem import AllChem, Descriptors3D
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from itertools import product
import numpy as np
import plotly.graph_objects as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mordred import Calculator, descriptors
import subprocess
import json


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


warnings.filterwarnings('ignore')


labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


dfs_meta = {}

train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv") #seems to be a polymer fragments dataset
train_df["source"] = "train.csv"
dfs_meta["train_df"] = { "labels": labels }

Tg_supp_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
Tg_supp_df["source"] = "dataset3.csv"
dfs_meta["Tg_supp_df"] = { "labels": ["Tg"] }

FFV_supp_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")
FFV_supp_df["source"] = "dataset4.csv"
dfs_meta["FFV_supp_df"] = { "labels": ["FFV"] }

#Tc_supp_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
#Tc_supp_df["source"] = "dataset1.csv"
#dfs_meta["Tc_supp_df"] = { "labels": ["Tc"] }
#Tc_supp_df.rename(columns={"TC_mean":"Tc"}, inplace=True)

extra_tg_df = pd.read_csv("/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv", usecols=["SMILES", "Tg (C)"])
extra_tg_df["source"] = "JCIM_sup_bigsmiles.csv"
dfs_meta["extra_tg_df"] = { "labels": ["Tg"] }
extra_tg_df.rename(columns ={"Tg (C)":"Tg"}, inplace=True)

extra_tg_3_df = pd.read_excel("/kaggle/input/smiles-extra-data/data_tg3.xlsx")
extra_tg_3_df["source"] = "data_tg3.xlsx"
dfs_meta["extra_tg_3_df"] = { "labels": ["Tg"] }
extra_tg_3_df["Tg"] = extra_tg_3_df["Tg [K]"] - 273.15

#extra_density_df = pd.read_excel("/kaggle/input/smiles-extra-data/data_dnst1.xlsx")#.describe()
#extra_density_df["source"] = "data_dnst1.xlsx"
#dfs_meta["extra_density_df"] = { "labels": ["Density"] }
#extra_density_df.rename(columns={'density(g/cm3)': 'Density'}, inplace=True)
#extra_density_df = extra_density_df[extra_density_df["Density"] != "nylon"]


def get_canonical_smiles(smiles):
    res = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        res = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True, kekuleSmiles=False)
    except: pass
    return res


for df_name in dfs_meta:
    df = globals()[df_name]
    df["canon_SMILES"] = df.SMILES.apply(lambda x: get_canonical_smiles(x) )
    cols = ["SMILES", "canon_SMILES", "source"]
    label_cols = dfs_meta[df_name]["labels"]
    cols.extend(label_cols)
    #cols = ["SMILES", "canon_SMILES"].extend(dfs_meta[df_name]["labels"])
    df = df[cols]
    globals()[df_name] = df[df["canon_SMILES"].notna()]


import duckdb
duckdb_conn = duckdb.connect(":memory:")


def get_dfs_of_label(label):
    return [dfs_name for dfs_name in dfs_meta.keys() if label in dfs_meta[dfs_name]["labels"]]


def get_label_df(label):
    union_select_clauses_str = ""
    df_names = get_dfs_of_label(label)
    if len(df_names)>0:
        select_clauses_list = [
            f"select canon_SMILES, {label} from {df_name} where {label} is not null "
            for df_name in df_names
                              ]
        #union_select_clauses_str = f' union {" union ".join(select_clauses_list)}'
        sql = f"""
        with q as 
        (
        { " union ".join(select_clauses_list) }
        )
        select canon_SMILES,
               avg({label}) as {label}
        from q
        group by canon_SMILES
               """ 
    return duckdb_conn.sql(sql).df()


for label in labels:
    df_name = f"{label}_label_df"
    globals()[f"{df_name}"] = get_label_df(label)  
    print(df_name, globals()[f"{df_name}"].shape[0], "labeled rows")


def get_smiles_df(df_names):
    select_clauses_list = [
            f"select canon_SMILES from {df_name}"
            for df_name in df_names
                              ]
    union_select_clauses_str = " union ".join(select_clauses_list)
    
    sql = f"""
with q as
(
{union_select_clauses_str}
)
select distinct canon_SMILES from q
"""
    return duckdb_conn.sql(sql).df()


df_names = [ f"{label}_label_df" for label in labels ]
SMILES_df = get_smiles_df(df_names)


def remove_dummy_atoms(mol):
    """Remove atoms with atomic number 0 (e.g., '*')"""
    emol = Chem.EditableMol(mol)
    dummy_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummy_idxs, reverse=True):
        emol.RemoveAtom(idx)
    clean_mol = emol.GetMol()
    clean_mol.UpdatePropertyCache(strict=False)
    rdmolops.FastFindRings(clean_mol)
    sanitization_status = -1
    try:
        Chem.SanitizeMol(clean_mol)
        sanitization_status = 0
    except  Exception as e:
        if "KekulizeException" in str(e) or "Can't kekulize mol" in str(e):
            try:
                Chem.SanitizeMol(clean_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                rdmolops.Kekulize(clean_mol, clearAromaticFlags=True)
                sanitization_status = 0
            except: pass
    return clean_mol, sanitization_status


def get_3d_mol(mol):
    mol = Chem.AddHs(mol)
    embedding_status = -1
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed=42
    ps.useExpTorsionAnglePrefs=True
    ps.useBasicKnowledge=True    
    ps.numThreads = 4
    ps.useRandomCoords=True
    try:
        embedding_status = AllChem.EmbedMolecule(mol, ps)
    except Exception as e:
        pass
    return mol, embedding_status


def get_optimized_3d_mol(mol):
    opt_status = -1
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            opt_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=-1)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            opt_status = AllChem.UFFOptimizeMolecule(mol, maxIters=-1)
    except Exception as e:
        pass
    return mol, opt_status 


def fix_smiles_for_embedding(smiles):
    # remove dummy atom with a double bond on the SMILES begin
    res = smiles
    substring_to_replace = "*/"
    if smiles.startswith(substring_to_replace):
        res = smiles[len(substring_to_replace):]
    return res


def get_descriptors_3d(smiles):
    res ={}
    
    res = {
           "sanitization_status": None,
           "embedding_status": None,
           "opt_status": None,
           "descriptors_3d_getting_status": None    
          }
    descriptors_3d = {}
    try:
        smiles = fix_smiles_for_embedding(smiles)
        mol = Chem.MolFromSmiles(smiles)
        mol, sanitization_status = remove_dummy_atoms(mol)
        res["sanitization_status"] = sanitization_status
        #if sanitization_status == 0:
        mol, embedding_status = get_3d_mol(mol)
        res["embedding_status"] = embedding_status
        mol, opt_status = get_optimized_3d_mol(mol)
        res["opt_status"] = opt_status
        if (embedding_status==0 and opt_status==0):
           descriptors_3d = Descriptors3D.CalcMolDescriptors3D(mol)
           res["descriptors_3d_getting_status"] = 0
    except:
        pass
    res = {**res, **descriptors_3d }
    return res


# seems to be a cause of issue on submission
def get_rdkit_descriptors_2d(smiles):
    descriptors = {}
    smiles_to_mol_status = -1
    #sanitization_status = -1
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles_to_mol_status = 0
        descriptors = Descriptors.CalcMolDescriptors(mol) 
    except:
        pass
    res = {"smiles_to_mol_status": smiles_to_mol_status }
    res = { **res, **descriptors}
    return res    


def get_rdlib_descriptors_for_list_smiles(list_smiles: list):
    descriptors_dict = { x: get_descriptors(x) for x in list_smiles }
    return pd.DataFrame.from_dict(descriptors_dict, orient='index')


from multiprocessing import Pool


#to avoid Segmentation fault
def get_descriptors_3d_subprocess(smiles, lib="rdkit"):
    if lib == "mordred":
        script = "/kaggle/input/descriptors-3d/get_mordred_descriptors_3d.py"
    else:
        script = "/kaggle/input/descriptors-3d/get_descriptors_3d.py"
    res = {}
    descriptors_3d = {}
    sproc = subprocess.run(["python",
                script,
                "--smiles",
                smiles
               ],
                      capture_output=True, text=True)
    res["subprocess_returncode"] = sproc.returncode
    try:
        descriptors_3d = json.loads(sproc.stdout.replace("'",'"'))
    except:
        pass
    return smiles, { **res, **descriptors_3d} 


from functools import partial


def get_descriptors_3d_for_list_smiles(list_smiles: list, processes: int, lib):
    #mol_descriptors_3d_dict = { x: get_descriptors_3d_subprocess(x) for x in list_smiles }
    partial_multiply = partial(get_descriptors_3d_subprocess, lib=lib)
    with Pool(processes=processes) as pool:
        descriptors_3d_list = pool.map(partial_multiply, list_smiles)
    #pool.close()
    descriptors_3d_dict = { x[0]: x[1] for x in descriptors_3d_list }
    return pd.DataFrame.from_dict(descriptors_3d_dict, orient='index')


def get_mordred_descriptors_3d_for_list_smiles(list_smiles: list, ignore_3D=True):
    #print(list_smiles)
    res = pd.DataFrame(list_smiles, columns=['SMILES'])
    #print(res)
    list_mols = [Chem.MolFromSmiles(smiles) for smiles in list_smiles]
    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    descrs_df = calc.pandas(list_mols)
    return res.merge(descrs_df, left_index=True, right_index=True)


#smiles = '*CC(*)(CC(=O)O)C(=O)O'
#mol = Chem.MolFromSmiles(smiles)
#calc = Calculator(descriptors)
#calc(mol).fill_missing(value=0).__len__()


#descriptors_df = get_rdlib_descriptors_for_list_smiles(list_smiles=list(train_df.SMILES))


import time


start = time.time()
descriptors_df = get_descriptors_3d_for_list_smiles(list_smiles=list(SMILES_df.canon_SMILES), processes=20, lib="mordred")
end = time.time()
print(end - start)
#descriptors_df.to_parquet("mordred_descriptors.parquet")


descriptors_df


#descriptors_df = pd.read_parquet("/kaggle/input/mordred-descriptors-3d/mordred_descriptors.parquet")


common_mordred_rdlib_cols = ['BalabanJ', 'BertzCT', 'TPSA', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9'] 
rdkit_descriptors_dict = { smiles: get_rdkit_descriptors_2d(smiles) for smiles in list(SMILES_df.canon_SMILES) }
rdkit_descriptors_df = pd.DataFrame.from_dict(rdkit_descriptors_dict, orient='index').drop(columns=common_mordred_rdlib_cols)


descriptors_df = descriptors_df.merge(rdkit_descriptors_df, left_index=True, right_index=True)


status_cols = [ "smiles_to_mol_status",
                "subprocess_returncode",
                "sanitization_status",
                "embedding_status", 
                "opt_status", 
                "descriptors_3d_getting_status",
                "serialization_status"
              ]


descriptors_df[status_cols].describe()


smiles_with_issues_df = descriptors_df[
(descriptors_df["smiles_to_mol_status"] != 0)
|
(descriptors_df["sanitization_status"] != 0)
|
(descriptors_df["embedding_status"] != 0)
|
(descriptors_df["opt_status"] != 0)
|
(descriptors_df["descriptors_3d_getting_status"] != 0)
|
(descriptors_df["serialization_status"] != 0)
]


for label in labels:
    globals()[f"{label}_label_df"] = globals()[f"{label}_label_df"].merge(descriptors_df, left_on="canon_SMILES", right_index=True)#.merge(descriptors_3d_df, left_on="SMILES", right_index=True)


for label in labels:
    print(label, globals()[f"{label}_label_df"].shape)


for label in labels:
    globals()[f"{label}_label_df"] = globals()[f"{label}_label_df"][
          (globals()[f"{label}_label_df"]["serialization_status"]==0)
                   ]


for label in labels:
    print(label, globals()[f"{label}_label_df"].shape)


descriptor_cols = list(descriptors_df.drop(status_cols, axis=1).columns) 


def check_feature(df, feature):
    res = None
    mock_label = pd.Series(0, index=np.arange(df.shape[0]), name='label')
    X, y = df[feature], mock_label
    try:
        dm = xgb.DMatrix(X, y, enable_categorical=False)
    except:
        res=feature
        print(f"{feature} has abnormal values")
    finally: 
        return res


def get_abnormal_value_cols(df):
    res = [ check_feature(df, col) for col in df.columns ]
    return [x for x in res if x is not None]


def get_one_value_cols(df):
    col_stats_df = df.describe().transpose()
    return list(col_stats_df[col_stats_df["min"] == col_stats_df["max"]].index.to_list())


def get_features(df):
    abnormal_value_cols = get_abnormal_value_cols(df)
    one_value_cols = get_one_value_cols(df)
    features = list(set(df.columns) - set(one_value_cols) - set(abnormal_value_cols))
    return features #, df[features]


features = get_features(descriptors_df)


for label in labels:
    print(label)
    globals()[f"{label}_features"] = get_features(globals()[f"{label}_label_df"][features])
    print(label, "features count: ", len(globals()[f"{label}_features"]))


def get_mol_3d_viewer(mol):
    block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(block, "molecule")
    viewer.setStyle({"stick": {}})
    viewer.setBackgroundColor("white")
    return viewer.zoomTo()


try:
    smiles_max_ffv = list(train_df[train_df["FFV"] == train_df["FFV"].max()].SMILES)[0]
    smiles_max_mol = Chem.MolFromSmiles(smiles_max_ffv)
except:
    smiles_max_mol = Chem.MolFromSmiles('*CCC1C[N+](C)(C)CC1*')
# seems not to be polymer itself
smiles_max_mol


try:
    smiles_min_ffv = list(train_df[train_df["FFV"] == train_df["FFV"].min()].SMILES)[0]
    smiles_min_mol = Chem.MolFromSmiles(smiles_min_ffv)
except:
    smiles_min_mol = Chem.MolFromSmiles('*CC(*)(CC(=O)O)C(=O)O')
smiles_min_mol
# seems not to be polymer itself


try:
    cleaned_smiles_min_mol, is_sanitized = remove_dummy_atoms(smiles_min_mol)
except:
    pass


mol_3d, embedding_status = get_3d_mol(cleaned_smiles_min_mol)
mol_3d, opt_status = get_optimized_3d_mol(mol_3d)
get_mol_3d_viewer(mol_3d).show()


# for sake of beauty
train_df["SMILES_len"] = train_df["SMILES"].str.len()
longest_smiles_len = train_df["SMILES_len"].max()
#longest_smiles = train_df[train_df["SMILES_len"] == longest_smiles_len].SMILES.to_list()[0]
longest_smiles = '*c1cc(C(c2ccc(OCc3cc(OCCN(C)c4ccc(C=CC5=CC(=C(C#N)C#N)CC(C)(C)C5)cc4)cc(OCCN(C)c4ccc(C=CC5=CC(=C(C#N)C#N)CC(C)(C)C5)cc4)c3)c(N3C(=O)c4ccc(Oc5ccc6c(c5)C(=O)N(*)C6=O)cc4C3=O)c2)(C(F)(F)F)C(F)(F)F)ccc1OCc1cc(OCCN(C)c2ccc(C=CC3=CC(=C(C#N)C#N)CC(C)(C)C3)cc2)cc(OCCN(C)c2ccc(C=CC3=CC(=C(C#N)C#N)CC(C)(C)C3)cc2)c1'
longest_smiles_mol = Chem.MolFromSmiles(longest_smiles)
cleaned_longest_smiles_mol, is_sanitized = remove_dummy_atoms(longest_smiles_mol)
mol_3d, embedding_status = get_3d_mol(cleaned_longest_smiles_mol)
mol_3d, opt_status = get_optimized_3d_mol(mol_3d)
get_mol_3d_viewer(mol_3d).show()


for label in labels:
    print(label)
    df = globals()[f"{label}_label_df"][[label] + globals()[f"{label}_features"]]
    #df = df.fillna(0)
    df = df.corr()[[label]].drop(label)
    df["abs_corr"] = df[label].abs()
    #corr_val = df.abs_corr.quantile(q=0.8)
    corr_vals = df.abs_corr.mean()
    cols_with_corr = list(df.sort_values("abs_corr", ascending=False)[df["abs_corr"]>0.5].index)
    locals()[f"{label}_corr_df"] = df
    locals()[f"features_{label}"] = cols_with_corr
    print(cols_with_corr[0:10])
    print(len(cols_with_corr))


import plotly.express as px
from plotly.subplots import make_subplots


for label in labels:
   corr_df = locals()[f"{label}_corr_df"]
   df = locals()[f"{label}_label_df"]
   lowest_corr_rate = 2 # fhigest is 0
   subplot_titles = corr_df.sort_values('abs_corr', ascending=False)[0:lowest_corr_rate+1].index.to_list()
   fig = make_subplots(rows=1, cols=lowest_corr_rate+1, subplot_titles=subplot_titles)
   for col_corr_rate in range(0, lowest_corr_rate+1):
       x_col_name = corr_df.sort_values('abs_corr', ascending=False)[col_corr_rate:col_corr_rate+1].index.to_list()[0]
       fig.add_trace( go.Scatter(x=df[x_col_name], y=df[label], mode='markers'), row=1, col=col_corr_rate+1 ) 
   fig.update_layout(title=f"{label}")
   fig.show() #(render="iframe")
   


smiles_with_issues_df


smiles_with_issues_df.merge(train_df, right_on="canon_SMILES", left_index=True)[["canon_SMILES"] + labels]


#smiles = "*Nc1ccc(C2(c3ccc(N*)cc3[C@]34C[C@@H]5C[C@H](C[C@@H](C5)C3)C4)c3ccccc3-c3ccccc32)c([C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)c1"
smiles = "*Nc1cc2ccccc2c2c1ccc1ccc3c(N*)cc4ccccc4c3c12"
#resembles complex alkaloids, steroids, or natural product-like frameworks with decorated aromatic systems.
mol = Chem.MolFromSmiles(smiles)
mol, is_sanitized = remove_dummy_atoms(mol)
mol, embedding_status = get_3d_mol(mol) #slow
print(str(embedding_status))
get_mol_3d_viewer(mol).show()



test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
test_df_for_visualization = test_df.head(3).copy()
#remove first dummy atom and directional bond to the first atom 
test_df_for_visualization['SMILES'] = test_df_for_visualization['SMILES'].str.replace(r'^\*\/', '', regex=True)


for smiles in test_df_for_visualization.SMILES.to_list():
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol, is_sanitized = remove_dummy_atoms(mol)
        mol, embedding_status = get_3d_mol(mol)
        mol, opt_status = get_optimized_3d_mol(mol)
        descriptors_3d = get_descriptors_3d_subprocess(smiles=smiles, lib="mordred")
        cor_feature_values = { "AATS2d": descriptors_3d[1].get("AATS2d"),
                               "FPSA5": descriptors_3d[1].get("FPSA5"),
                               "VSA_EState7": descriptors_3d[1].get("VSA_EState7"),
                               "MZ": descriptors_3d[1].get("MZ"),
                               "NsCH3": descriptors_3d[1].get("NsCH3"),
                             }
        get_mol_3d_viewer(mol).show()
        print(cor_feature_values)
        print(f"embedding status: {embedding_status}; optimization status: {opt_status}")
    except Exception as e:
        pass



def get_X_y(df, label, features):
    df_filtered = df[df[label].notna()]
    X, y = df_filtered[features].fillna(0), df_filtered[[label]]
    return X, y


import os
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"


from ray import train, tune, air
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
import ray
from ray.train.xgboost import XGBoostTrainer


def ray_tune(df, label, features, smoke_test=True):

    X, y = get_X_y(df=df, label=label, features=features)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, shuffle=True) 
    train_x_ref = ray.put(train_x)
    test_x_ref = ray.put(test_x)
    train_y_ref = ray.put(train_y)
    test_y_ref = ray.put(test_y)
       
    def xgb_train(config):
        # Access Train context for worker-specific info
        train_context = train.get_context()
        print(f"Train worker dir: {train_context.get_trial_dir()}")

        # Access Tune context for trial-specific info (if running under Tune)
        try:
            tune_context = tune.get_context()
            print(f"Tune trial dir: {tune_context.get_trial_dir()}")
        except RuntimeError as e:
            print(f"Not running under Ray Tune: {e}")
            
        #train_x, test_x, train_y, test_y = train_test_split(ray.get(X_ref), ray.get(y_ref), test_size=0.25, shuffle=True) 
        train_set = xgb.DMatrix(ray.get(train_x_ref), label=ray.get(train_y_ref))
        test_set = xgb.DMatrix(ray.get(test_x_ref), label=ray.get(test_y_ref))
        
        results = {}
        xgb.train(config,
                  train_set,
                  evals=[(test_set, "eval")],
                  evals_result=results,
                  verbose_eval=False,
                  # `TuneReportCheckpointCallback` defines the checkpointing frequency and format.
                  callbacks=[TuneReportCheckpointCallback(frequency=1, metrics={"eval-mae": "eval-mae"})],
                 )
        # Return prediction MAE
        print(f'xgb train result["eval"] {results["eval"]}')
        mae = results["eval"]["mae"][-1]
        tune.report({"eval-mae": mae, "done": True})
        

    def tune_xgboost(smoke_test=False):
        search_space = {
                        # You can mix constants with search space objects.
                        "objective": "reg:absoluteerror",
                        "eval_metric": ["mae"],
                        "tree_method": "hist",
                        "device": "cuda",
                        "max_depth": tune.randint(2, 10),
                        #"min_child_weight": tune.choice([1, 2, 3]),
                        "subsample": tune.uniform(0.5, 1.0),
                        "learning_rate": tune.loguniform(0.01, 0.3),
                        #"num_boost_round": tune.choice([x*10 for x in range(1,14)]),
                        #"n_estimators": tune.choice([x*10 for x in range(1,15)]),
                        
                       }
        scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2  # 10 training iterations
    )
       
                
        tuner = tune.Tuner(
                           tune.with_resources(xgb_train, resources={"cpu": 4}),
                           tune_config=tune.TuneConfig( 
                           
                                                       num_samples=1 if smoke_test else 500,
                                                       metric="eval-mae",
                                                       mode="min",
                                                       scheduler=scheduler,
                                                                                                             ),
                            #run_config=air.RunConfig(
                            #                         verbose=0,
                            #                         log_to_file="/dev/null"
                            #                        ),
                            param_space=search_space,
                           )
        results = tuner.fit()
        return results
    
    return tune_xgboost(smoke_test=smoke_test)    


#ray.shutdown()   


#Tg_label_df


#best_configs = {}
#best_metrics = {}
#for label in labels:
#    df = globals()[f"{label}_label_df"]
#    label_features = locals()[f"{label}_features"]
#    result_grid = ray_tune(df=df, label=label, features=label_features, smoke_test=False)
#    best_configs[f"params_{label}"] = result_grid.get_best_result(metric="eval-mae", mode="min").config
#    best_metrics[label] = result_grid.get_best_result(metric="eval-mae", mode="min").metrics["eval-mae"]


#best_metrics


best_configs = \
{'params_Tg': {'objective': 'reg:absoluteerror',
  'eval_metric': ['mae'],
  'tree_method': 'hist',
  'max_depth': 3,
  'subsample': 0.957000018504639,
  'learning_rate': 0.21458121313326117},
 'params_FFV': {
  'objective': 'reg:absoluteerror',
  'eval_metric': ['mae'],
  'max_depth': 9,
  'subsample': 0.8993916178868326,
  'learning_rate': 0.16900787837599915
      },
 'params_Tc': {'objective': 'reg:absoluteerror',
  'eval_metric': ['mae'],
  'tree_method': 'hist',
  #'device': 'cuda',
  'max_depth': 4,
  'subsample': 0.7935516775315187,
  'learning_rate': 0.24557559518304853},
 'params_Density': {'objective': 'reg:absoluteerror',
  'eval_metric': ['mae'],
  'tree_method': 'hist',
  #'device': 'cuda',
  'max_depth': 3,
  'subsample': 0.7970328202058499,
  'learning_rate': 0.2543639086241011},
 'params_Rg': {
    'objective': 'reg:absoluteerror',
    'eval_metric': ['mae'],
    'max_depth': 2,
    'subsample': 0.8156773723050599,
    'learning_rate': 0.15177055640458229
    }}


for label in labels:
    par_name = f"params_{label}"
    locals()[par_name] = best_configs[par_name]
    print(label)
    print(locals()[par_name])


def cross_val_model (X, y, params):
    params["nthread"] = 3
    model = make_pipeline(
    #StandardScaler(),  # Important if features vary in scale, but it seems not for tree
    xgb.XGBRegressor(**params, eval_set=[(X, y)])
    )
    dm = xgb.DMatrix(X, y, enable_categorical=False)
    #print(params)
    cv_results = xgb.cv(params,
                        dm,
                        nfold=5,
                        metrics="mae",
                        seed=42,
                        num_boost_round=500,
                        early_stopping_rounds=10,
                       )
    return cv_results


for label in labels:
    print(label)
    par_name = f"params_{label}"
    #label_features = locals()[f"features_{label}"]
    X, y = get_X_y(df=globals()[f"{label}_label_df"], label=label, features=globals()[f"{label}_features"])
    #locals()[par_name]["device"] =  "cuda"
    cv_results = cross_val_model(X=X, y=y, params=locals()[par_name])
    print(cv_results.tail(1))
    num_boost_round = cv_results.tail(1).index.stop + 1
    locals()[par_name]["num_boost_round"] = num_boost_round
    locals()[par_name]["n_estimators"] = num_boost_round
    print(locals()[par_name])


for label in labels:
    par_name = f"params_{label}"
    label_features = globals()[f"{label}_features"]
    var_name = f"model_{label}"
    globals()[var_name] = make_pipeline(
    #StandardScaler(),  # Important if features vary in scale
    xgb.XGBRegressor(**globals()[par_name]))
    df = globals()[f"{label}_label_df"]
    X, y = get_X_y(df=df, label=label, features=label_features)
    globals()[var_name].fit(X, y)


test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
test_descriptors_df = get_descriptors_3d_for_list_smiles(list_smiles=list(test_df.SMILES), processes=20, lib="mordred")
test_rdkit_descriptors_dict = { smiles: get_rdkit_descriptors_2d(smiles) for smiles in list(test_df.SMILES) }
test_rdkit_descriptors_df = pd.DataFrame.from_dict(test_rdkit_descriptors_dict, orient='index').drop(columns=common_mordred_rdlib_cols)
#test_descriptors_df = test_descriptors_df.merge(test_rdkit_descriptors_df, left_index=True, right_index=True)
test_df = test_df.merge(test_descriptors_df, left_on="SMILES", right_index=True).merge(test_rdkit_descriptors_df, left_on="SMILES", right_index=True)
#test_cols = test_descriptors_df.columns.to_list()
#for missing_col in set(features) -  set(test_cols):
#    test_df[missing_col] = 0


for label in labels:
    #print(label)
    model_name = f"model_{label}"
    label_features = globals()[f"{label}_features"]
    X = test_df[label_features].fillna(0)
    model = globals()[model_name]
    test_df[label] = model.predict(X)


test_df[["id", "Tg", "FFV","Tc","Density","Rg"]]


test_df[["id", "Tg", "FFV","Tc","Density","Rg"]].to_csv("submission.csv", index=False)


!cat submission.csv

