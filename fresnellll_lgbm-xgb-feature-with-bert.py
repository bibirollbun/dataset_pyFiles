# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# %% [markdown]
# # Polymer GNN - Offline Submission v2
#
# **版本更新:**
# - 使用通过 `SizeShiftReg` 正则化训练的v2模型。
# - 增加了 `torch-cluster` 库的离线安装。
# - 适配了v2模型返回多个值的 `forward` 方法。

# %%
# ===================================================================
# 1. 安装所有离线依赖 (v2 - 根据实际文件路径定制)
# ===================================================================
import os

# 定义我们存放 .whl 文件的路径
# 【正确路径】指向你提供的 `gnn-online` 数据集中的 `kaggle_wheels` 文件夹
WHEELS_PATH = "/kaggle/input/gnn-online/kaggle_wheels/" 

# 【正确路径】指向你提供的 RDKit wheel 文件
RDKIT_WHEEL_PATH = "/kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl"

print("--- 安装 RDKit ---")
# 使用 --no-index 确保离线，--no-deps 避免不必要的依赖检查
!pip install --no-index --no-deps {RDKIT_WHEEL_PATH}

print("\n--- 安装 PyTorch Geometric 及其核心依赖 ---")
# 使用 --find-links 指向我们的 wheel 文件夹
# 安装顺序很重要：先安装核心依赖，再安装主包
!pip install --no-index --find-links={WHEELS_PATH} torch_cluster
!pip install --no-index --find-links={WHEELS_PATH} torch_scatter
!pip install --no-index --find-links={WHEELS_PATH} torch_sparse
!pip install --no-index --find-links={WHEELS_PATH} pyg_lib
!pip install --no-index --find-links={WHEELS_PATH} torch_geometric

print("\n--- 安装其他特定版本的库 ---")
# 明确指定 numpy 和 scikit-learn 的 wheel 文件路径，以避免版本冲突
!pip install --no-index --no-deps {os.path.join(WHEELS_PATH, 'numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')}
!pip install --no-index --no-deps {os.path.join(WHEELS_PATH, 'scikit_learn-1.4.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')}
# ipywidgets 在提交环境中非必需，可以不装
# !pip install --no-index --no-deps {os.path.join(WHEELS_PATH, 'ipywidgets-8.1.7-py3-none-any.whl')}


print("\n\n[成功] 所有v2离线依赖安装完毕！")


# ===================================================================
# Cell 1: 精准修复核心库版本 (最终完整版)
# ===================================================================
print("--- Kaggle Notebook Environment Setup (The Definitive, Correct Version) ---")

# --- 卸载原生环境中可能冲突的版本，为我们的精准安装铺路 ---
print("\nStep 1: Uninstalling potentially conflicting native packages...")
# -y 参数可以跳过烦人的确认步骤
!pip uninstall -y scikit-learn joblib numpy lightgbm xgboost

# --- 使用精确、硬编码的路径，进行最终的强制离线安装 ---
# 这个列表包含了我们为了确保环境绝对一致而必须安装的所有核心组件
print("\nStep 2: Force-installing our exact versions from datasets...")

# ！！！！【请务必确认】！！！！
# 根据您提供的文件列表，定义whl文件所在的文件夹路径
SKLEARN_PATCH_DIR = "/kaggle/input/sklearn-1-4-2-patch/sklearn_minimal_wheels/"
EXTRA_PACKAGES_DIR = "/kaggle/input/polymer-extra-packages/wheels_for_kaggle/"
NUMPY_PATCH_DIR = "/kaggle/input/numpy-1-26-4/"
PANDARALLEL_DIR = "/kaggle/input/pandarallel-1-6-5/"
RDKIT_DIR = "/kaggle/input/rdkit-2025-3-3-cp311/"

!pip install -q --no-dependencies \
    {NUMPY_PATCH_DIR}numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    {SKLEARN_PATCH_DIR}joblib-1.4.2-py3-none-any.whl \
    {SKLEARN_PATCH_DIR}threadpoolctl-3.6.0-py3-none-any.whl \
    {SKLEARN_PATCH_DIR}scikit_learn-1.4.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    {EXTRA_PACKAGES_DIR}lightgbm-4.3.0-py3-none-manylinux_2_28_x86_64.whl \
    {EXTRA_PACKAGES_DIR}xgboost-2.0.3-py3-none-manylinux2014_x86_64.whl \
    {PANDARALLEL_DIR}pandarallel-1.6.5-py3-none-any.whl \
    {RDKIT_DIR}rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl

print("\n--- Environment patching complete! ---")
# --- 最终验证 ---
# 运行结束后，请检查下面的输出，确保版本号是我们期望的
print("\nFinal version check (Concise Output):")
!pip show scikit-learn joblib numpy rdkit lightgbm xgboost | grep -E "^Name:|^Version:"


# ===================================================================
# Cell 1 & 2: 导入所有库、配置路径并加载全局资产
# ===================================================================
#/kaggle/input/bert_smile/pytorch/default/1
# --- Section 1: 导入所有必要的库 ---
print("正在导入所有必要的库...")
import pandas as pd
import numpy as np
import warnings
import os
import gc
import re
import json # <--- 确保json被导入
import joblib # <--- 确保joblib被导入
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops, AllChem
import networkx as nx
import lightgbm as lgb
import xgboost as xgb
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
tqdm.pandas()
print("库导入完成。")


# --- Section 2: 全局配置 (CFG)与资产加载 ---
# ==================================
# Cell 2: 路径配置与加载全局资产
# ==================================
class CFG:
    COMP_DIR = "/kaggle/input/neurips-open-polymer-prediction-2025/"
    
    # --- 【【【核心修改1】】】 ---
    # 将模型路径指向新的 V2 版本
    EXPERT_MODEL_DIR = "/kaggle/input/ml-baseline-v1/ML_baseline_models/basemodel_8_22_Morgan_stack_new_env" 
    
    BERT_MODEL_PATH = "/kaggle/input/bert_smile/pytorch/default/1" 
    
    # --- 【【【新增代码】】】 ---
    # 添加GNN模型和资产的路径，用于为测试集提取特征
    GNN_MODEL_ROOT = "/kaggle/input/gnn-merge-v1/GNN_plus_models"
    GNN_MODEL_BASE_PATH = "/kaggle/input/gat-wmae/exp1_wmae_loss_model"
    BEST_FP_INDICES_PATH = "/kaggle/input/models-v9/models_v9/best_fp_indices_v9.pkl"
    
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # 与本地训练时保持一致的参数
    BERT_MAX_LEN = 128
    BERT_HIDDEN_DIM = 768
    GNN_HIDDEN_DIM = 384 # GNN嵌入的维度
    N_FOLDS = 5 # GNN模型的折数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("正在加载测试数据和全局资产...")
test = pd.read_csv(os.path.join(CFG.COMP_DIR, "test.csv"))
try:
    print(f"将从以下路径加载模型和配置: {CFG.EXPERT_MODEL_DIR}")
    assert os.path.exists(CFG.EXPERT_MODEL_DIR), f"FATAL ERROR: 模型根路径不存在! 请检查: {CFG.EXPERT_MODEL_DIR}"

    # 加载全局的PCA对象和所有的元模型
    pca = joblib.load(os.path.join(CFG.EXPERT_MODEL_DIR, 'pca_object.joblib'))
    meta_model_tg = joblib.load(os.path.join(CFG.EXPERT_MODEL_DIR, 'meta_model_tg.joblib'))
    meta_model_ffv = joblib.load(os.path.join(CFG.EXPERT_MODEL_DIR, 'meta_model_ffv.joblib'))
    meta_model_rg = joblib.load(os.path.join(CFG.EXPERT_MODEL_DIR, 'meta_model_rg.joblib'))
    meta_model_density = joblib.load(os.path.join(CFG.EXPERT_MODEL_DIR, 'meta_model_density.joblib'))
    meta_model_tc = joblib.load(os.path.join(CFG.EXPERT_MODEL_DIR, 'meta_model_tc.joblib'))
    
    print("全局PCA对象和5个元模型加载成功。")
    
    print(f"将从以下路径加载BERT模型: {CFG.BERT_MODEL_PATH}")
    assert os.path.exists(CFG.BERT_MODEL_PATH), f"FATAL ERROR: BERT模型路径不存在! 请检查: {CFG.BERT_MODEL_PATH}"
    print("BERT模型路径确认存在！")

except Exception as e:
    print(f"致命错误：加载资产文件失败！错误: {e}")
    raise


# ===================================================================
# Cell 3: 在测试集上生成通用特征池 (复现本地训练的Part 1)
# ===================================================================
print("\n--- 开始在测试集上生成通用的基础特征池 ---")

# --- Section 1a: 定义所有特征工程函数 (与训练时完全一致) ---
useless_cols = [
    "MaxPartialCharge", 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur', 'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan', 'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan', 'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1', 'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v', 'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2', 'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6', 'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6', 'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide', 'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2', 'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester',
    'NumBridgeheadAtoms', 'NumSaturatedCarbocycles', 'NumSaturatedRings', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA3', 'PEOE_VSA4', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_Ar_COO', 'fr_Ar_NH', 'fr_COO', 'fr_Imine', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_amidine', 'fr_azo', 'fr_furan', 'fr_imidazole', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitro_arom_nonortho', 'fr_piperdine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_thiazole', 'fr_urea'
]
desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]

def compute_all_descriptors_safe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]
def compute_graph_features_safe(smiles, graph_feats_dict):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        graph_feats_dict['graph_diameter'].append(0); graph_feats_dict['avg_shortest_path'].append(0); graph_feats_dict['num_cycles'].append(0)
        return
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    graph_feats_dict['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats_dict['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats_dict['num_cycles'].append(len(list(nx.cycle_basis(G))))
def count_atoms_safe(smiles):
    counts = {'num_C': 0, 'num_c': 0, 'num_O': 0, 'num_N': 0, 'num_F': 0, 'num_Cl': 0, 'num_positive_ions': 0, 'num_negative_ions': 0}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return counts
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol(); charge = atom.GetFormalCharge()
        if symbol == 'C': counts['num_c' if atom.GetIsAromatic() else 'num_C'] += 1
        elif symbol in ['O', 'N', 'F', 'Cl']: counts[f'num_{symbol}'] += 1
        if charge > 0: counts['num_positive_ions'] += 1
        elif charge < 0: counts['num_negative_ions'] += 1
    return counts

# --- Section 1b: 生成BERT + PCA特征 ---
print("  - 正在为测试集生成BERT Embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_MODEL_PATH)
bert_model = AutoModel.from_pretrained(CFG.BERT_MODEL_PATH)
bert_model.to(device); bert_model.eval()
def smiles_to_embedding(smiles, tokenizer, model):
    try:
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=CFG.BERT_MAX_LEN)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception:
        return np.zeros(CFG.BERT_HIDDEN_DIM)
test_bert_embeddings = np.array([smiles_to_embedding(s, tokenizer, bert_model) for s in tqdm(test["SMILES"], desc="Test BERT Embeddings")])
test_bert_pca = pca.transform(test_bert_embeddings) # 使用加载的pca对象进行transform
test_bert_df = pd.DataFrame(test_bert_pca, columns=[f'bert_pca_{i}' for i in range(test_bert_pca.shape[1])])
del test_bert_embeddings, bert_model, tokenizer; gc.collect(); torch.cuda.empty_cache()

# --- Section 1c: 生成手动/衍生特征 ---
print("  - 正在为测试集生成手动/衍生特征...")
descriptors_list = [compute_all_descriptors_safe(smi) for smi in tqdm(test['SMILES'], desc="RDKit Descriptors")]
rdkit_feats_df = pd.DataFrame(descriptors_list, columns=desc_names)
graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
for smi in tqdm(test['SMILES'], desc="Graph Features"):
    compute_graph_features_safe(smi, graph_feats)
graph_feats_df = pd.DataFrame(graph_feats)
atom_counts_list = [count_atoms_safe(smi) for smi in tqdm(test['SMILES'], desc="Atom Counts")]
atom_counts_df = pd.DataFrame(atom_counts_list)
manual_feats_df = pd.concat([rdkit_feats_df, graph_feats_df, atom_counts_df], axis=1)
manual_feats_df.replace([np.inf, -np.inf], np.nan, inplace=True)
with np.errstate(divide='ignore', invalid='ignore'):
    manual_feats_df["Ipc"] = np.log10(manual_feats_df["Ipc"])
eps = 1e-6
manual_feats_df["NumAromaticHeterocycles_div_NumHeteroatoms"] = manual_feats_df["NumAromaticHeterocycles"] / (manual_feats_df["NumHeteroatoms"] + eps)
manual_feats_df["fr_unbrch_alkane_div_MolWt"] = manual_feats_df["fr_unbrch_alkane"] / (manual_feats_df["MolWt"] + eps)
manual_feats_df["PEOE_VSA14_div_graph_diameter"] = manual_feats_df["PEOE_VSA14"] / (manual_feats_df["graph_diameter"] + eps)
manual_feats_df["BalabanJ_mul_TPSA"] = manual_feats_df["BalabanJ"] * manual_feats_df["TPSA"]
manual_feats_df["qed_mul_SMR_VSA5"] = manual_feats_df["qed"] * manual_feats_df["SMR_VSA5"]
manual_feats_df["VSA_EState7_div_MolWt"] = manual_feats_df["VSA_EState7"] / (manual_feats_df["MolWt"] + eps)
manual_feats_df["SMR_VSA10_div_MolWt"] = manual_feats_df["SMR_VSA10"] / (manual_feats_df["MolWt"] + eps)
manual_feats_df["SlogP_VSA12_div_MolWt"] = manual_feats_df["SlogP_VSA12"] / (manual_feats_df["MolWt"] + eps)
manual_feats_df["SMR_VSA10_div_fr_unbrch_alkane"] = manual_feats_df["SMR_VSA10"] / (manual_feats_df["fr_unbrch_alkane"] + eps)
manual_feats_df["qed_mul_TPSA"] = manual_feats_df["qed"] * manual_feats_df["TPSA"]
manual_feats_df["PEOE_VSA14_div_fr_unbrch_alkane"] = manual_feats_df["PEOE_VSA14"] / (manual_feats_df["fr_unbrch_alkane"] + eps)
manual_feats_df["PEOE_VSA14_mul_AvgIpc"] = manual_feats_df["PEOE_VSA14"] * manual_feats_df["AvgIpc"]
manual_feats_df["SMR_VSA5_div_MolWt"] = manual_feats_df["SMR_VSA5"] / (manual_feats_df["MolWt"] + eps)
manual_feats_df["PEOE_VSA14_div_SlogP_VSA7"] = manual_feats_df["PEOE_VSA14"] / (manual_feats_df["SlogP_VSA7"] + eps)
manual_feats_df["VSA_EState7_div_SPS"] = manual_feats_df["VSA_EState7"] / (manual_feats_df["SPS"] + eps)
manual_feats_df["SlogP_VSA5_mul_FpDensityMorgan1"] = manual_feats_df["SlogP_VSA5"] * manual_feats_df["FpDensityMorgan1"]
manual_feats_df["VSA_EState8_div_PEOE_VSA5"] = manual_feats_df["VSA_EState8"] / (manual_feats_df["PEOE_VSA5"] + eps)
manual_feats_df["ion_ratio"] = manual_feats_df["num_positive_ions"] / (manual_feats_df["num_negative_ions"] + eps)
manual_feats_df["net_ion_charge"] = manual_feats_df["num_positive_ions"] - manual_feats_df["num_negative_ions"]
manual_feats_df["ion_density"] = (manual_feats_df["num_positive_ions"] + manual_feats_df["num_negative_ions"]) / (manual_feats_df["MolWt"] + eps)
manual_feats_df['SMR_VSA5_div_MolWt_div_fr_nitro'] = (manual_feats_df['SMR_VSA5'] / (manual_feats_df['MolWt'] + eps)) / (manual_feats_df['fr_nitro'] + eps)
manual_feats_df['fr_unbrch_alkane_div_MolWt_div_EState_VSA11'] = (manual_feats_df['fr_unbrch_alkane'] / (manual_feats_df['MolWt'] + eps)) / (manual_feats_df['EState_VSA11'] + eps)
manual_feats_df['PEOE_VSA14_div_graph_diameter_div_BalabanJ'] = (manual_feats_df['PEOE_VSA14'] / (manual_feats_df['graph_diameter'] + eps)) / (manual_feats_df['BalabanJ'] + eps)
manual_feats_df['VSA_EState7_div_SPS_div_PEOE_VSA14'] = (manual_feats_df['VSA_EState7'] / (manual_feats_df['SPS'] + eps)) / (manual_feats_df['PEOE_VSA14'] + eps)

# --- Section 1d: 生成高维Morgan Fingerprint特征池 ---
print("  - 正在为测试集生成高维Morgan Fingerprint特征池...")
MFP_N_BITS = 1024
mfp_features = []
for smi in tqdm(test['SMILES'], desc="Generating Morgan Fingerprints Pool"):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: mfp_features.append(np.zeros(MFP_N_BITS, dtype=int)); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MFP_N_BITS)
        mfp_features.append(np.array(fp))
    except: mfp_features.append(np.zeros(MFP_N_BITS, dtype=int))

mfp_df_pool = pd.DataFrame(mfp_features, columns=[f'mfp_{i}' for i in range(MFP_N_BITS)])
# 【重要】在测试集上，我们不再做方差筛选，以保证列数与训练时筛选器的输入完全一致
print(f"    - 成功生成 {mfp_df_pool.shape[1]} 维的MFP特征池。")

# # --- Section 1e: 【新增】加载并合并GNN深度特征池 ---
# print("  - 正在加载预计算的GNN OOF嵌入特征...")
# try:
#     gnn_embed_df = pd.read_csv(CFG.GNN_EMBEDDINGS_PATH)
#     # 使用merge确保SMILES能精确对齐
#     # 注意：这里的 train DataFrame 是 Cell 3 生成的主DataFrame
#     train = pd.merge(train, gnn_embed_df, on='SMILES', how='left')
    
#     # 获取GNN特征的列名，以备后用
#     gnn_feature_cols = [col for col in gnn_embed_df.columns if 'gnn_embed_' in col]
#     print(f"    - 成功加载并合并 {len(gnn_feature_cols)} 维GNN特征。")
    
# except FileNotFoundError:
#     print(f"  - [警告] 找不到GNN嵌入文件: {CFG.GNN_EMBEDDINGS_PATH}。将不使用GNN特征。")
#     gnn_feature_cols = []

# --- Section 1e: 整合所有特征到一个大的特征池DataFrame ---
X_test_full_pool = pd.concat([test_bert_df, manual_feats_df, mfp_df_pool], axis=1)


# --- Section 1f: 【【【新增】】】为测试集生成GNN深度特征 ---
print("\n  - 正在为测试集生成GNN深度特征...")

# 导入GNN相关的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool

# --- [粘贴GNN模型定义和辅助函数] ---
# 为了让脚本独立运行，我们需要在这里包含GNN的模型定义
def get_atom_features(atom): return [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.GetHybridization(), int(atom.GetIsAromatic()), atom.GetTotalNumHs()]
def smiles_to_periodic_graph(smiles: str):
    try: mol = Chem.MolFromSmiles(smiles, sanitize=False); Chem.SanitizeMol(mol)
    except: return None
    if mol is None: return None
    x = torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_features, star_atom_indices = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); edge_indices.extend([[i, j], [j, i]])
        bond_feats = [bond.GetBondTypeAsDouble(), int(bond.GetIsConjugated())]; edge_features.extend([bond_feats, bond_feats])
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*': star_atom_indices.append(atom.GetIdx())
    if len(star_atom_indices) == 2:
        i, j = star_atom_indices[0], star_atom_indices[1]; edge_indices.extend([[i, j], [j, i]])
        periodic_bond_feats = [-1.0, 0]; edge_features.extend([periodic_bond_feats, periodic_bond_feats])
    return Data(x=x, edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(edge_features, dtype=torch.float))

class PolymerGNN_v12_Res(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_layers, tasks_fp_indices, targets, heads=4, dropout=0.2):
        super(PolymerGNN_v12_Res, self).__init__(); assert hidden_dim % heads == 0
        head_dim = hidden_dim // heads
        self.tasks_fp_indices, self.targets, self.dropout = tasks_fp_indices, targets, dropout
        self.input_conv = GATv2Conv(num_node_features, head_dim, heads=heads, dropout=self.dropout, edge_dim=num_edge_features)
        self.input_bn = BatchNorm(hidden_dim)
        self.hidden_convs = nn.ModuleList([GATv2Conv(hidden_dim, head_dim, heads=heads, dropout=self.dropout, edge_dim=num_edge_features) for _ in range(num_layers - 1)])
        self.hidden_bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - 1)])
        self.task_predictors = nn.ModuleDict()
        for task_name in self.targets:
            if task_name in self.tasks_fp_indices:
                k = len(self.tasks_fp_indices[task_name]); self.task_predictors[task_name] = nn.Sequential(nn.Linear(hidden_dim + k, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))
    def forward(self, data):
        x, edge_index, edge_attr, batch, morgan_fp = data.x, data.edge_index, data.edge_attr, data.batch, data.morgan_fp
        x = F.elu(self.input_bn(self.input_conv(x, edge_index, edge_attr=edge_attr)))
        for conv, bn in zip(self.hidden_convs, self.hidden_bns): x = F.elu(bn(conv(x, edge_index, edge_attr=edge_attr))) + x
        graph_embedding = global_mean_pool(x, batch)
        outputs = []
        for task_name in self.targets:
            if task_name in self.task_predictors:
                indices = torch.tensor(self.tasks_fp_indices[task_name], dtype=torch.long, device=morgan_fp.device); selected_fp = morgan_fp.index_select(1, indices)
                fused_embedding = torch.cat([graph_embedding, selected_fp], dim=1); outputs.append(self.task_predictors[task_name](fused_embedding))
        return torch.cat(outputs, dim=1)

class GNNFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.input_conv = original_model.input_conv
        self.input_bn = original_model.input_bn
        self.hidden_convs = original_model.hidden_convs
        self.hidden_bns = original_model.hidden_bns
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.elu(self.input_bn(self.input_conv(x, edge_index, edge_attr=edge_attr)))
        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            x = F.elu(bn(conv(x, edge_index, edge_attr=edge_attr))) + x
        return global_mean_pool(x, batch)
# --- [GNN模型定义结束] ---

# 预处理测试集SMILES
test_data_list = [smiles_to_periodic_graph(smi) for smi in tqdm(test['SMILES'], desc="GNN: Preprocessing Test SMILES")]
valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
valid_data_list = [test_data_list[i] for i in valid_indices]
test_loader = PyGDataLoader(valid_data_list, batch_size=256, shuffle=False)
best_fp_indices = joblib.load(CFG.BEST_FP_INDICES_PATH)

# 循环加载5个GNN模型，提取特征并取平均
all_fold_embeddings = []
for fold in range(CFG.N_FOLDS):
    model_path = os.path.join(CFG.GNN_MODEL_BASE_PATH, f"fold_{fold}", "final_refit_model.pth")
    original_model = PolymerGNN_v12_Res(num_node_features=7, num_edge_features=2, hidden_dim=CFG.GNN_HIDDEN_DIM, num_layers=6, tasks_fp_indices=best_fp_indices, targets=CFG.TARGETS, heads=8)
    original_model.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
    
    feature_extractor = GNNFeatureExtractor(original_model).to(CFG.DEVICE).eval()
    
    fold_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(CFG.DEVICE)
            embeddings = feature_extractor(batch)
            fold_embeddings.append(embeddings.cpu().numpy())
    all_fold_embeddings.append(np.concatenate(fold_embeddings))

# 对5个模型的特征提取结果取平均，得到更稳健的特征
mean_embeddings = np.mean(all_fold_embeddings, axis=0)
gnn_embed_df_test = pd.DataFrame(columns=[f"gnn_embed_{i}" for i in range(CFG.GNN_HIDDEN_DIM)], index=test.index)
gnn_embed_df_test.iloc[valid_indices] = mean_embeddings
gnn_embed_df_test.fillna(0, inplace=True)

# 【关键】将GNN特征加入到总的特征池中
X_test_full_pool = pd.concat([X_test_full_pool, gnn_embed_df_test], axis=1)

print(f"\n--- GNN深度特征已生成并合并。通用特征池最终维度: {X_test_full_pool.shape} ---")


# ===================================================================
# Cell 4: 专属预测、Stacking融合、查表与提交
# (此代码块已准备好使用包含GNN特征的 X_test_full_pool)
# ===================================================================

# --- Part 1: Level 0 - 生成测试集的L0预测 ---
print("\n--- Part 1: 开始Level 0预测 - 为每个目标生成基础模型预测 ---")
test_preds_l0 = {} # 用于存储每个任务的L0预测结果

# 循环处理5个目标任务
for target in CFG.TARGETS:
    print(f"\n  - 正在为 '{target}' 生成L0预测...")
    
    # 1. 加载当前任务专属的模型资产
    target_dir = os.path.join(CFG.EXPERT_MODEL_DIR, target)
    
    # 加载特征列表。这个列表是在训练时生成的，
    # 【【关键】】它已经包含了模型需要的、经过筛选的gnn_embed_*列名。
    with open(os.path.join(target_dir, "feature_cols.json"), "r") as f: 
        feature_cols_for_target = json.load(f)
    
    # 加载用于填充缺失值的特征均值
    with open(os.path.join(target_dir, "feature_means.json"), "r") as f: 
        feature_means_for_target = pd.read_json(f, typ='series')
        
    # 2. 从总特征池中，精确地抽取出当前任务需要的特征
    # X_test_full_pool 是我们在Cell 3中生成的，包含1586列的DataFrame
    # feature_cols_for_target 是一个列表，例如 ['bert_pca_0', ..., 'mfp_10', ..., 'gnn_embed_5', ...]
    X_test_target = X_test_full_pool[feature_cols_for_target].copy()
    
    # 3. 填充在测试集上可能出现的缺失值
    # 使用训练时计算的均值来填充，确保数据分布的一致性
    X_test_target = X_test_target.fillna(feature_means_for_target)
    
    # 4. 使用5折模型进行预测并取平均
    lgb_fold_preds, xgb_fold_preds = [], []
    for fold in range(5):
        # 加载LGBM和XGBoost模型
        model_lgb = joblib.load(os.path.join(target_dir, f'model_LGBM_fold{fold+1}.joblib'))
        model_xgb = joblib.load(os.path.join(target_dir, f'model_XGB_fold{fold+1}.joblib'))
        
        # 使用当前折的模型进行预测
        # 模型会自动使用 X_test_target 中正确的特征列
        lgb_fold_preds.append(model_lgb.predict(X_test_target))
        xgb_fold_preds.append(model_xgb.predict(X_test_target))
        
    # 对5个模型的预测结果取平均，得到更稳健的L0预测
    test_preds_l0[f'lgb_{target}'] = np.mean(lgb_fold_preds, axis=0)
    test_preds_l0[f'xgb_{target}'] = np.mean(xgb_fold_preds, axis=0)
    print(f"    - '{target}' 的L0预测已生成。")


# --- Part 2: Level 1 - 使用元模型进行最终的Stacking预测 ---
# 【【这部分逻辑也完全不变】】
# 它接收上面生成的L0预测，然后用L1元模型（meta_model_*.joblib）进行最终的融合
print("\n\n--- Part 2: 开始Level 1预测 - 使用元模型进行Stacking融合 ---")
submission_df = pd.DataFrame({'id': test['id']})

# Group A: Tg (独立)
X_test_meta_tg = pd.DataFrame({
    'lgb_Tg': test_preds_l0['lgb_Tg'], 
    'xgb_Tg': test_preds_l0['xgb_Tg']
})
submission_df['Tg'] = meta_model_tg.predict(X_test_meta_tg)
print("  - Group A (Tg) 预测完成。")

# Group B: FFV (独立)
X_test_meta_ffv = pd.DataFrame({
    'lgb_FFV': test_preds_l0['lgb_FFV'], 
    'xgb_FFV': test_preds_l0['xgb_FFV']
})
submission_df['FFV'] = meta_model_ffv.predict(X_test_meta_ffv)
print("  - Group B (FFV) 预测完成。")

# Group C: Rg, Density, Tc (关联组)
X_test_meta_group_c = pd.DataFrame({
    'lgb_Rg': test_preds_l0['lgb_Rg'], 'xgb_Rg': test_preds_l0['xgb_Rg'],
    'lgb_Density': test_preds_l0['lgb_Density'], 'xgb_Density': test_preds_l0['xgb_Density'],
    'lgb_Tc': test_preds_l0['lgb_Tc'], 'xgb_Tc': test_preds_l0['xgb_Tc']
})
submission_df['Rg'] = meta_model_rg.predict(X_test_meta_group_c)
submission_df['Density'] = meta_model_density.predict(X_test_meta_group_c)
submission_df['Tc'] = meta_model_tc.predict(X_test_meta_group_c)
print("  - Group C (Rg, Density, Tc) 预测完成。")

# --- Part 3 & 4: (可选) 查表提分与生成最终文件 ---
# 如果您有查表逻辑，可以放在这里

print("\n\n--- Part 4: 正在生成最终的 submission.csv 文件 ---")
submission_df.to_csv("submission.csv", index=False)
print("submission.csv 文件已成功生成！")
print("\n最终提交文件预览:")
print(submission_df.head())

