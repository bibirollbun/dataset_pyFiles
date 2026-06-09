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


# 侦察兵Cell：列出所有输入目录的内容
!ls -R /kaggle/input/


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
# 2. 导入库，定义配置、函数和模型 (CV Ensemble Submission)
# ===================================================================
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import warnings
import joblib
import os

# PyTorch and PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# --- 全局配置 (CFG) ---
class CFG:
    # 【【【修改1】】】: 路径必须指向包含所有fold文件夹的主目录
    # !!! 请将 'YOUR_CV_DATASET_NAME' 替换为你上传的Kaggle数据集的名称 !!!
    CV_MODEL_BASE_PATH = "/kaggle/input/gat-official-data/GAT_official_data"
    
    MODEL_ASSETS_PATH_V9 = "/kaggle/input/models-v9/models_v9"
    COMPETITION_DATA_PATH = "/kaggle/input/neurips-open-polymer-prediction-2025/"
    
    BEST_FP_INDICES_PATH = os.path.join(MODEL_ASSETS_PATH_V9, "best_fp_indices_v9.pkl")
    
    # 任务和特征配置
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    NUM_NODE_FEATURES = 7
    NUM_EDGE_FEATURES = 2
    MORGAN_FP_DIM = 1024
    MORGAN_FP_RADIUS = 2
    
    # --- 锁定的最佳模型架构 (来自实验10) ---
    HIDDEN_DIM = 384
    NUM_LAYERS = 6
    GAT_HEADS = 8
    
    # 交叉验证折数
    N_FOLDS = 5
    
    # 推理配置
    BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 核心数据处理函数 (无需修改) ---
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

# --- GNN模型定义 (必须使用带残差连接的版本) ---
class PolymerGNN_v12_Res(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_layers, tasks_fp_indices, targets, heads=4, dropout=0.2):
        super(PolymerGNN_v12_Res, self).__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads
        self.tasks_fp_indices = tasks_fp_indices
        self.targets = targets
        self.dropout = dropout
        self.input_conv = GATv2Conv(num_node_features, head_dim, heads=heads, dropout=self.dropout, edge_dim=num_edge_features)
        self.input_bn = BatchNorm(hidden_dim)
        self.hidden_convs = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_convs.append(GATv2Conv(hidden_dim, head_dim, heads=heads, dropout=self.dropout, edge_dim=num_edge_features))
            self.hidden_bns.append(BatchNorm(hidden_dim))
        self.task_predictors = nn.ModuleDict()
        for task_name in self.targets:
            if task_name in self.tasks_fp_indices:
                k = len(self.tasks_fp_indices[task_name])
                self.task_predictors[task_name] = nn.Sequential(
                    nn.Linear(hidden_dim + k, hidden_dim), nn.ReLU(),
                    nn.Dropout(0.2), nn.Linear(hidden_dim, 1))
    def forward(self, data):
        x, edge_index, edge_attr, batch, morgan_fp = data.x, data.edge_index, data.edge_attr, data.batch, data.morgan_fp
        x = self.input_conv(x, edge_index, edge_attr=edge_attr)
        x = self.input_bn(x)
        x = F.elu(x)
        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            residual = x
            x_out = conv(x, edge_index, edge_attr=edge_attr)
            x_out = bn(x_out)
            x_out = F.elu(x_out)
            x = x_out + residual
        graph_embedding = global_mean_pool(x, batch)
        outputs = []
        for task_name in self.targets:
            if task_name in self.task_predictors:
                indices = torch.tensor(self.tasks_fp_indices[task_name], dtype=torch.long, device=morgan_fp.device)
                selected_fp = morgan_fp.index_select(1, indices)
                fused_embedding = torch.cat([graph_embedding, selected_fp], dim=1)
                outputs.append(self.task_predictors[task_name](fused_embedding))
            else:
                outputs.append(torch.zeros(graph_embedding.size(0), 1, device=graph_embedding.device))
        return torch.cat(outputs, dim=1)

print("配置、函数和模型类定义加载完毕 (CV集成提交版)。")


# ===================================================================
# 3. 主预测流程 (CV集成)
# ===================================================================

# --- 1. 加载并预处理一次测试数据 ---
print("--- 加载并预处理测试数据 ---")
test_df = pd.read_csv(os.path.join(CFG.COMPETITION_DATA_PATH, "test.csv"))
def process_single_smiles_for_inference(smi):
    graph_data = smiles_to_periodic_graph(smi)
    if not graph_data: return None
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, CFG.MORGAN_FP_RADIUS, nBits=CFG.MORGAN_FP_DIM)
        graph_data.morgan_fp = torch.tensor(np.array(fp), dtype=torch.float).unsqueeze(0)
    else:
        graph_data.morgan_fp = torch.zeros(1, CFG.MORGAN_FP_DIM, dtype=torch.float)
    return graph_data
test_data_list = [process_single_smiles_for_inference(smi) for smi in tqdm(test_df['SMILES'], desc="处理测试集SMILES")]
print(f"测试数据预处理完成。")

def collate_fn(data_list):
    valid_data_list = [data for data in data_list if data is not None]
    if not valid_data_list: return None
    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader = PyGDataLoader(valid_data_list, batch_size=len(valid_data_list))
    return next(iter(loader))

test_loader = DataLoader(test_data_list, batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
best_fp_indices = joblib.load(CFG.BEST_FP_INDICES_PATH)

# --- 2. 循环加载每个折的模型进行预测 ---
all_fold_preds = []
print(f"\n--- 开始 {CFG.N_FOLDS}-Fold 集成预测 ---")

for fold in range(CFG.N_FOLDS):
    print(f"\n--- 正在使用 Fold-{fold} 模型进行预测 ---")
    
    # 构造当前折的文件路径
    model_path = os.path.join(CFG.CV_MODEL_BASE_PATH, f"fold_{fold}", "final_refit_model.pth")
    calibrators_path = os.path.join(CFG.CV_MODEL_BASE_PATH, f"fold_{fold}", "calibrators.pkl")
    
    # 加载模型
    model = PolymerGNN_v12_Res(
        num_node_features=CFG.NUM_NODE_FEATURES, num_edge_features=CFG.NUM_EDGE_FEATURES,
        hidden_dim=CFG.HIDDEN_DIM, num_layers=CFG.NUM_LAYERS,
        tasks_fp_indices=best_fp_indices, targets=CFG.TARGETS, heads=CFG.GAT_HEADS
    )
    model.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
    model.to(CFG.DEVICE)
    model.eval()
    
    # 加载校准器
    calibrators = joblib.load(calibrators_path)
    
    # 进行预测
    fold_raw_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold-{fold} GNN预测中", leave=False):
            if batch is None: continue
            batch = batch.to(CFG.DEVICE)
            predictions = model(batch)
            fold_raw_preds.append(predictions.cpu().numpy())
    
    if fold_raw_preds: fold_raw_preds = np.concatenate(fold_raw_preds, axis=0)
    else: fold_raw_preds = np.array([])
    
    # 进行校准
    fold_calibrated_preds = np.copy(fold_raw_preds)
    for i, task in enumerate(CFG.TARGETS):
        if task in calibrators and len(fold_raw_preds) > 0:
            task_raw_preds = fold_raw_preds[:, i].reshape(-1, 1)
            fold_calibrated_preds[:, i] = calibrators[task].predict(task_raw_preds)
            
    all_fold_preds.append(fold_calibrated_preds)
    print(f"Fold-{fold} 预测与校准完成！")

# --- 3. 平均所有折的预测结果 ---
print("\n--- 平均所有折的预测结果 ---")
final_preds_calibrated = np.mean(np.stack(all_fold_preds, axis=0), axis=0)
print("集成预测完成！")


# ===================================================================
# 4. 生成提交文件
# ===================================================================
print("\n--- 生成提交文件 ---")
submission_df = pd.read_csv(os.path.join(CFG.COMPETITION_DATA_PATH, "sample_submission.csv"))

# 处理无法解析的SMILES
valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
full_preds = np.zeros((len(test_df), len(CFG.TARGETS)))

if final_preds_calibrated.shape[0] > 0:
    full_preds[valid_indices, :] = final_preds_calibrated # <-- 使用最终的集成预测结果

# 填充到submission_df中
for i, task in enumerate(CFG.TARGETS):
    submission_df[task] = full_preds[:, i]

submission_df.to_csv("submission.csv", index=False)
print("submission.csv 文件已成功生成！")
print("提交文件预览:")
print(submission_df.head())

