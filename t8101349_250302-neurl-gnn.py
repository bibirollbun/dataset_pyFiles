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


# 1. å®‰è£� PyTorch
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# 2. å®‰è£� RDKitï¼ˆç”¨æ–¼è™•ç�†åˆ†å­�ï¼‰
!pip install rdkit-pypi



%%writefile requirements.txt
easydict
future
matplotlib
numpy
opencv-python
scikit-image
scipy
click
requests
tqdm
pyspng
ninja
imageio-ffmpeg==0.4.3
timm
psutil
scikit-learn


!pip install -r requirements.txt


# 1. å�‡ç´š PyTorch å’Œ torchvision
#!pip install torch==2.1.0 torchvision==0.15.2 torchaudio==2.0.2 torchdata==0.6.1
!pip install torch torchvision torchaudio torchdata --index-url https://download.pytorch.org/whl/cu118
!pip install pytorch-lightning


# 3. æª¢æŸ¥å®‰è£�æ˜¯å�¦æˆ�åŠŸ
import torch
import torchdata


print("Torch version:", torch.__version__)
print("Torchdata version:", torchdata.__version__)





!python --version
!nvcc --version  # å¦‚æ�œå·²ç¶“å®‰è£�äº†CUDA



!pip install duckdb


!pip install pyarrow  # æˆ–è€…ä½ ä¹Ÿå�¯ä»¥é�¸æ“‡ fastparquet




import pyarrow.parquet as pq
import pandas as pd
import gc  # å¼•å…¥å�ƒåœ¾å›�æ”¶æ¨¡çµ„

filename = "/kaggle/input/leash-BELKA/train.parquet"
columns_to_read = ["molecule_smiles", "protein_name", "binds"]

batch_size = 60000    # æ¯�æ¬¡è®€å�– 60,000 ç­†
target_rows = 1200000  # æ¯�è¼ªå„²å­˜ 1,200,000 ç­†
total_batches = 15    # ç¸½å…±åŸ·è¡Œ 15 æ¬¡
total_rows = 0        # è¨˜éŒ„ç•¶å‰�ç´¯ç©�ç­†æ•¸

parquet_file = pq.ParquetFile(filename)

# ç�²å�–ç¸½è¡Œçµ„æ•¸é‡�
num_row_groups = parquet_file.num_row_groups
print(f"Total row groups in file: {num_row_groups}")

# è¨ˆç®—æ¯�æ¬¡è®€å�–å¤šå°‘è¡Œçµ„
row_groups_per_batch = target_rows // batch_size


# é–‹å§‹é€²è¡Œæ‰¹æ¬¡è™•ç�†
for i in range(total_batches):
    chunks = []
    current_rows = 0  # æ¯�è¼ªçš„è¨ˆæ•¸å™¨
    batch_start_row_group = i * row_groups_per_batch  # ç›´æ�¥ä¾�åº�å�–
    batch_end_row_group = min(batch_start_row_group + row_groups_per_batch, num_row_groups)

    if batch_end_row_group >= num_row_groups:
        batch_end_row_group = num_row_groups  # é�¿å…�è¶…å‡ºè¡Œçµ„ç¯„åœ�

    print(f"âœ… ç¬¬ {i+1} æ¬¡è™•ç�†ï¼šå¾�è¡Œçµ„ {batch_start_row_group} åˆ°è¡Œçµ„ {batch_end_row_group}")

    # ä½¿ç”¨ pyarrow çš„ ParquetFile ç›´æ�¥è®€å�–æŒ‡å®šç¯„åœ�çš„è¡Œçµ„
    for row_group_idx in range(batch_start_row_group, batch_end_row_group):
        try:
            batch = parquet_file.read_row_groups([row_group_idx], columns=columns_to_read)
            chunk = batch.to_pandas()

            # æª¢æŸ¥æ˜¯å�¦æœ‰è³‡æ–™
            if not chunk.empty:
                chunks.append(chunk)
                current_rows += len(chunk)
                total_rows += len(chunk)

            # å¦‚æ�œè®€å�–åˆ°æŒ‡å®šç¯„åœ�çš„è³‡æ–™ï¼Œå°±å�œæ­¢
            if total_rows >= target_rows * (i + 1):
                break  # å¦‚æ�œå·²ç¶“è®€åˆ°è©²æ‰¹æ¬¡çš„çµ�å°¾å°±å�œæ­¢

        except Exception as e:
            print(f"âš ï¸� è®€å�–è¡Œçµ„ {row_group_idx} æ™‚ç™¼ç”ŸéŒ¯èª¤: {e}")

    if chunks:  # ç¢ºä¿�æœ‰è³‡æ–™æ‰�é€²è¡Œå�ˆä½µ
        # å�ˆä½µ DataFrame
        batch_df = pd.concat(chunks, ignore_index=True)

        # **å°‡æ¯� 3 åˆ—å�ˆä½µæˆ� 1 åˆ—**
        batch_df["row_idx"] = batch_df.index // 3  # æ¯� 3 åˆ—åˆ†çµ„
        batch_pivot = batch_df.pivot(index="row_idx", columns="protein_name", values="binds").reset_index()

        # **å�ˆä½µ molecule_smiles**
        smiles_df = batch_df.groupby("row_idx")["molecule_smiles"].first().reset_index()
        final_df = smiles_df.merge(batch_pivot, on="row_idx").drop(columns=["row_idx"])

        # å­˜æˆ� parquetï¼Œæ¯�æ¬¡éƒ½å­˜ä¸�å�Œçš„æª”æ¡ˆ
        output_filename = f"/kaggle/working/train_part{i+1}.parquet"
        final_df.to_parquet(output_filename, index=False)

        print(f"âœ… ç¬¬ {i+1} æ¬¡å­˜æª”ï¼š{len(final_df)} ç­†ï¼Œå·²ç´¯ç©� {total_rows} ç­†")

        # æ¸…ç�†ç„¡ç”¨çš„è®Šæ•¸ï¼Œé‡‹æ”¾è¨˜æ†¶é«”
        del batch_df, batch_pivot, smiles_df, final_df
        gc.collect()  # åŸ·è¡Œå�ƒåœ¾å›�æ”¶
    else:
        print(f"âš ï¸� ç¬¬ {i+1} æ¬¡è™•ç�†æœªè®€å�–åˆ°ä»»ä½•è³‡æ–™ï¼Œè·³é��è©²æ‰¹æ¬¡ã€‚")



# 15 å€‹ Parquet æª”æ¡ˆ
parquet_files = [f"/kaggle/working/train_part{i+1}.parquet" for i in range(0, 15)]

# åˆ�å§‹åŒ– DuckDB é€£ç·š
con = duckdb.connect()

# å­˜æ”¾æ‰€æœ‰æ‰¹æ¬¡çš„ DataFrame
all_samples = []

# é€�å€‹è™•ç�† 15 å€‹æª”æ¡ˆ
for i, file in enumerate(parquet_files):
    print(f"ğŸ“‚ æ­£åœ¨è™•ç�†æª”æ¡ˆ: {file}")

    df = con.query(f"""(SELECT * FROM parquet_scan('{file}')
                            WHERE BRD4 = 0 and HSA = 0 and sEH = 0
                            ORDER BY random()
                            LIMIT 3000)
                            UNION ALL
                            (SELECT * FROM parquet_scan('{file}')
                            WHERE BRD4 = 1 or HSA = 1 or sEH = 1
                            ORDER BY random()
                            LIMIT 13000)""").df()

# å„²å­˜è©²æ‰¹æ¬¡çµ�æ�œ
    output_filename = f"/kaggle/working/sampled_test_part{i+1}.parquet"
    df.to_parquet(output_filename, index=False)
    print(f"âœ… å·²å„²å­˜æŠ½æ¨£çµ�æ�œ: {output_filename}ï¼ˆå…± {len(df)} ç­†ï¼‰")

    all_samples.append(df)

# å�ˆä½µæ‰€æœ‰çµ�æ�œ
final_test_df = pd.concat(all_samples, ignore_index=True)

# å„²å­˜ç¸½å�ˆä½µçš„ Parquet
final_test_output = "/kaggle/working/sampled_test_all.parquet"
final_test_df.to_parquet(final_test_output, index=False)
print(f"ğŸ�¯ å…¨éƒ¨ 15 å€‹æª”æ¡ˆå·²è™•ç�†å®Œç•¢ï¼Œæœ€çµ‚å�ˆä½µæª”æ¡ˆ: {final_test_output}ï¼ˆå…± {len(final_test_df)} ç­†ï¼‰")

# é—œé–‰ DuckDB
con.close()


'''
# 15 å€‹ Parquet æª”æ¡ˆ
parquet_files = [f"/kaggle/working/train_part{i+1}.parquet" for i in range(0, 15)]

# åˆ�å§‹åŒ– DuckDB é€£ç·š
con = duckdb.connect()

# å­˜æ”¾æ‰€æœ‰æ‰¹æ¬¡çš„ DataFrame
all_samples = []

# é€�å€‹è™•ç�† 15 å€‹æª”æ¡ˆ
for i, file in enumerate(parquet_files):
    print(f"ğŸ“‚ æ­£åœ¨è™•ç�†æª”æ¡ˆ: {file}")

    df = con.query(f"""(SELECT * FROM parquet_scan('{file}')
                            ORDER BY random()
                            LIMIT 15000)
                            UNION ALL
                            (SELECT * FROM parquet_scan('{file}')
                            WHERE BRD4 = 1 or HSA = 1 or sEH = 1
                            ORDER BY random()
                            LIMIT 5000)""").df()

# å„²å­˜è©²æ‰¹æ¬¡çµ�æ�œ
    output_filename = f"/kaggle/working/sampled_test_part{i+1}.parquet"
    df.to_parquet(output_filename, index=False)
    print(f"âœ… å·²å„²å­˜æŠ½æ¨£çµ�æ�œ: {output_filename}ï¼ˆå…± {len(df)} ç­†ï¼‰")

    all_samples.append(df)

# å�ˆä½µæ‰€æœ‰çµ�æ�œ
final_test_df = pd.concat(all_samples, ignore_index=True)

# å„²å­˜ç¸½å�ˆä½µçš„ Parquet
final_test_output = "/kaggle/working/sampled_test_all.parquet"
final_test_df.to_parquet(final_test_output, index=False)
print(f"ğŸ�¯ å…¨éƒ¨ 15 å€‹æª”æ¡ˆå·²è™•ç�†å®Œç•¢ï¼Œæœ€çµ‚å�ˆä½µæª”æ¡ˆ: {final_test_output}ï¼ˆå…± {len(final_test_df)} ç­†ï¼‰")

# é—œé–‰ DuckDB
con.close()
'''
#y = df[["BRD4", "HSA", "sEH"]].values


# df.head()



output_filename = f"/kaggle/working/training_df.csv"
df.to_csv(output_filename, index=False)



!pip install torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

!pip install torch-geometric
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+${CUDA}.html


import torch
print(torch.cuda.is_available())  # True
print(torch.version.cuda)  # æ‡‰è©²é¡¯ç¤º 11.8



import torch
import torch_scatter
print("torch-scatter å®‰è£�æˆ�åŠŸï¼�")


import numpy as np

import rdkit
from rdkit import Chem

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter

import torch
import torch.nn as nn
import torch.nn.functional as F

print('import ok!')


# helper
# torch version of np unpackbits
#https://gist.github.com/vadimkantorov/30ea6d278bc492abf6ad328c6965613a

def tensor_dim_slice(tensor, dim, dim_slice):
	return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]

# @torch.jit.script
def packshape(shape, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8, pack=True):
	dim = dim if dim >= 0 else dim + len(shape)
	bits, nibble = (
		8 if dtype is torch.uint8 else 16 if dtype is torch.int16 else 32 if dtype is torch.int32 else 64 if dtype is torch.int64 else 0), (
		1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0)
	# bits = torch.iinfo(dtype).bits # does not JIT compile
	assert nibble <= bits and bits % nibble == 0
	nibbles = bits // nibble
	shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)),) + shape[1 + dim:]) if pack else (
				shape[:dim] + (shape[dim] * nibbles,) + shape[1 + dim:])
	return shape, nibbles, nibble

# @torch.jit.script
def F_unpackbits(tensor, dim: int = -1, mask: int = 0b00000001, shape=None, out=None, dtype=torch.uint8):
	dim = dim if dim >= 0 else dim + tensor.dim()
	shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=tensor.dtype, pack=False)
	shape = shape if shape is not None else shape_
	out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
	assert out.shape == shape

	if shape[dim] % nibbles == 0:
		shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
		shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
		return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

	else:
		for i in range(nibbles):
			shift = nibble * i
			sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
			sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
			torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
	return out

class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

            
print('helper ok!')


# mol to graph adopted from
# from https://github.com/LiZhang30/GPCNDTA/blob/main/utils/DrugGraph.py

PACK_NODE_DIM=9
PACK_EDGE_DIM=1
NODE_DIM=PACK_NODE_DIM*8
EDGE_DIM=PACK_EDGE_DIM*8

def one_of_k_encoding(x, allowable_set, allow_unk=False):
	if x not in allowable_set:
		if allow_unk:
			x = allowable_set[-1]
		else:
			raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
	return list(map(lambda s: x == s, allowable_set))


#Get features of an atom (one-hot encoding:)
'''
	1.atom element: 44+1 dimensions    
	2.the atom's hybridization: 5 dimensions
	3.degree of atom: 6 dimensions                        
	4.total number of H bound to atom: 6 dimensions
	5.number of implicit H bound to atom: 6 dimensions    
	6.whether the atom is on ring: 1 dimension
	7.whether the atom is aromatic: 1 dimension           
	Total: 70 dimensions
'''

ATOM_SYMBOL = [
	'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
	'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
	'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
	'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
	'Pt', 'Hg', 'Pb', 'Dy',
	#'Unknown'
]
#print('ATOM_SYMBOL', len(ATOM_SYMBOL))44
HYBRIDIZATION_TYPE = [
	Chem.rdchem.HybridizationType.S,
	Chem.rdchem.HybridizationType.SP,
	Chem.rdchem.HybridizationType.SP2,
	Chem.rdchem.HybridizationType.SP3,
	Chem.rdchem.HybridizationType.SP3D
]

def get_atom_feature(atom):
	feature = (
		 one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL)
	   + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
	   + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
	   + [atom.IsInRing()]
	   + [atom.GetIsAromatic()]
	)
	#feature = np.array(feature, dtype=np.uint8)
	feature = np.packbits(feature)
	return feature


#Get features of an edge (one-hot encoding)
'''
	1.single/double/triple/aromatic: 4 dimensions       
	2.the atom's hybridization: 1 dimensions
	3.whether the bond is on ring: 1 dimension          
	Total: 6 dimensions
'''

def get_bond_feature(bond):
	bond_type = bond.GetBondType()
	feature = [
		bond_type == Chem.rdchem.BondType.SINGLE,
		bond_type == Chem.rdchem.BondType.DOUBLE,
		bond_type == Chem.rdchem.BondType.TRIPLE,
		bond_type == Chem.rdchem.BondType.AROMATIC,
		bond.GetIsConjugated(),
		bond.IsInRing()
	]
	#feature = np.array(feature, dtype=np.uint8)
	feature = np.packbits(feature)
	return feature


def smile_to_graph(smiles):
	mol = Chem.MolFromSmiles(smiles)
	N = mol.GetNumAtoms()
	node_feature = []
	edge_feature = []
	edge = []
	for i in range(mol.GetNumAtoms()):
		atom_i = mol.GetAtomWithIdx(i)
		atom_i_features = get_atom_feature(atom_i)
		node_feature.append(atom_i_features)

		for j in range(mol.GetNumAtoms()):
			bond_ij = mol.GetBondBetweenAtoms(i, j)
			if bond_ij is not None:
				edge.append([i, j])
				bond_features_ij = get_bond_feature(bond_ij)
				edge_feature.append(bond_features_ij)
	node_feature=np.stack(node_feature)
	edge_feature=np.stack(edge_feature)
	edge = np.array(edge,dtype=np.uint8)
	return N,edge,node_feature,edge_feature

def to_pyg_format(N,edge,node_feature,edge_feature):
	graph = Data(
		idx=-1,
		edge_index = torch.from_numpy(edge.T).int(),
		x          = torch.from_numpy(node_feature).byte(),
		edge_attr  = torch.from_numpy(edge_feature).byte(),
	)
	return graph

#debug one example
g = to_pyg_format(*smile_to_graph(smiles="C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1"))
print(g)
print('[Dy] is replaced by C !!')
print('smile_to_graph() ok!')


#MODEL: simple MPNNModel
#from https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

#DEVICE='cuda'
DEVICE='cpu'

# i have removed all comments here to jepp it clean. refer to orginal link for code comments
# of MPNNModel
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        super().__init__(aggr=aggr)
    
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )
    
    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out
    
    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
    

class MPNNModel(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        super().__init__()
    
        self.lin_in = nn.Linear(in_dim, emb_dim)
    
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
    
        self.pool = global_mean_pool
    
    def forward(self, data): #PyG.Data - batch of PyG graphs
    
        h = self.lin_in(F_unpackbits(data.x,-1).float())  
    
        for conv in self.convs:
            h = h + conv(h, data.edge_index.long(), F_unpackbits(data.edge_attr,-1).float())  # (n, d) -> (n, d)
    
        h_graph = self.pool(h, data.batch)  
        return h_graph

# our prediction model here !!!!
class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
    
        self.output_type = ['infer', 'loss']
    
        graph_dim=96
        self.smile_encoder = MPNNModel(
             in_dim=NODE_DIM, edge_dim=EDGE_DIM, emb_dim=graph_dim, num_layers=4,
        )
        self.bind = nn.Sequential(
            nn.Linear(graph_dim, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )
    
    def forward(self, batch):
        graph = batch['graph']
        x = self.smile_encoder(graph) 
        bind = self.bind(x)
    
        # --------------------------
        output = {}
        if 'loss' in self.output_type:
            target = batch['bind']
            output['bce_loss'] = F.binary_cross_entropy_with_logits(bind.float(), target.float())
            """
            # Debugging loss calculationæ–°å¢�
            print("Bind Output (Sigmoid):", torch.sigmoid(bind).detach().cpu().numpy())
            print("Target:", target.cpu().numpy())
            print("Loss Value:", output['bce_loss'].item())
            """
    
        if 'infer' in self.output_type:
            output['bind'] = torch.sigmoid(bind)
    
        return output
    
#debug: make some dummy data and run
'''
def run_check_net():
    batch_size = 30
    node_dim=NODE_DIM
    edge_dim=EDGE_DIM
    
    data = []
    for b in range(batch_size):
        N = np.random.randint(5,10)
        E = np.random.randint(3,N*(N-1))
        edge_index = np.stack([
            np.random.choice(N, E, replace=True),
            np.random.choice(N, E, replace=True),
        ]).T
        edge_index = np.sort(edge_index)
        edge_index = edge_index[edge_index[:, 0].argsort()]
        edge_index[0] = [0,1] #default
        edge_index = edge_index[edge_index[:,0]!=edge_index[:,1]]
        edge_index = np.unique(edge_index, axis=0)
    
        E = len(edge_index)
        edge_index = np.ascontiguousarray(edge_index.T)
    
        d = Data(
            idx        = b,
            edge_index = torch.from_numpy(edge_index).int(),
            x          = torch.from_numpy(np.packbits(np.random.choice(2, (N, node_dim)),-1)).byte(),
            edge_attr  = torch.from_numpy(np.packbits(np.random.choice(2, (E, edge_dim)),-1)).byte(),
        )
        data.append(d)
    
    
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    epoch_indices = []  # å„²å­˜ index ä»¥æª¢æŸ¥è·³è®Š
    for batch in loader:
        epoch_indices.extend(batch.idx.tolist())
    
    
    # ğŸ“Œ è¨˜éŒ„ç•¶å‰� epoch æ‰€æœ‰ batch index
    print(f"Epoch index range: {min(epoch_indices)} â†’ {max(epoch_indices)}")
    
    
    # loader = DataLoader(data, batch_size=batch_size)
    graph = next(iter(loader))
    idx = graph.idx.tolist()  #use to index bind array
    batch = dotdict( 
        graph = graph.to(DEVICE),
        bind  = torch.from_numpy(np.random.choice(2, (batch_size, 3))).float().to(DEVICE),
    )
    zz=0
    
    net = Net().to(DEVICE)
    #print(net)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True): # dtype=torch.float16):
            output = net(batch)
            #print(output['bind'])
    
    # ---
    print('batch')
    for k, v in batch.items():
        if k=='idx':
            print(f'{k:>32} : {len(v)} ')
        elif k=='graph':
            print(f'{k:>32} : {graph} ')
        else:
            print(f'{k:>32} : {v.shape} ')
    
    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape} ')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()} ')
    
                
run_check_net()
'''
print('model ok!')



#example of parallel conversion of smiles to graph
'''
from multiprocessing import Pool
from tqdm import tqdm
import gc
from torch_geometric.loader import DataLoader as PyGDataLoader

def to_pyg_list(graph):
	L = len(graph)
	for i in tqdm(range(L)):
		N, edge, node_feature, edge_feature = graph[i]
		graph[i] = Data(
			idx=i,
			edge_index=torch.from_numpy(edge.T).int(),
			x=torch.from_numpy(node_feature).byte(),
			edge_attr=torch.from_numpy(edge_feature).byte(),
		)
	return graph


train_smiles=[ #replace [Dy] with C
    "C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1",
    "C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1",
    "C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1",
    "C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1",
    "C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1",
    "C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1",
]
train_bind =np.array([
    [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,0,0],
])
num_train= len(train_smiles)
with Pool(processes=64) as pool:
    train_graph = list(tqdm(pool.imap(smile_to_graph, train_smiles), total=num_train))

train_graph = to_pyg_list(train_graph)
train_loader = PyGDataLoader(train_graph, batch_size=3, shuffle=True)

## example training loop
scaler = torch.cuda.amp.GradScaler(enabled=True)
net = Net()
net.to(DEVICE)

optimizer =\
	torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

num_epoch=10
epoch=0
iteration=0
while epoch<num_epoch: 
	for t, graph_batch in enumerate(train_loader): 
		index = graph_batch.idx.tolist()
		B = len(index)
		batch = dotdict(
			graph  = graph_batch.to(DEVICE),
			bind   = torch.from_numpy(train_bind[index]).to(DEVICE),
		)

		net.train()
		net.output_type = ['loss', 'infer']
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)  #data_parallel(net,batch) #
			bce_loss = output['bce_loss']

		optimizer.zero_grad() 
		scaler.scale(bce_loss).backward() 
		scaler.step(optimizer)
		scaler.update()
		 
		torch.clear_autocast_cache()
		print(epoch,iteration,bce_loss.item())
		iteration +=  1
        
	epoch += 1
'''


'''
import numpy as np
import torch
from torch_geometric.data import Data
from multiprocessing import Pool
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader


# è½‰æ�›ç‚º PyG æ ¼å¼�
def to_pyg_list(graph):
	L = len(graph)
	for i in tqdm(range(L)):
		N, edge, node_feature, edge_feature = graph[i]
		graph[i] = Data(
			idx=i,
			edge_index=torch.from_numpy(edge.T).int(),
			x=torch.from_numpy(node_feature).byte(),
			edge_attr=torch.from_numpy(edge_feature).byte(),
		)
	return graph


train_file = "/kaggle/input/trainall0/sampled_train_all.parquet"
df = pd.read_parquet(train_file)


# train_loader
# å¾� df è®€å�– SMILES
train_smiles = df['molecule_smiles'].tolist()


train_bind = df[["BRD4", "HSA", "sEH"]].values
#train_bind = np.array([[1, 0, 0] if b == 1 else [0, 0, 0] for b in train_bind])  # å�‡è¨­ binds æ˜¯ 0/1

num_train = len(train_smiles)

# å¹³è¡Œè™•ç�† SMILES è½‰ Graph
with Pool(processes=8) as pool:  # è¨­ç‚º 8 æ ¸å¿ƒï¼Œé�¿å…�è¨˜æ†¶é«”çˆ†ç‚¸
    train_graph = list(tqdm(pool.imap(smile_to_graph, train_smiles), total=num_train))


train_graph = to_pyg_list(train_graph)
#train_loader = PyGDataLoader(train_graph, batch_size=3, shuffle=True)
train_loader = PyGDataLoader(train_graph, batch_size=3, shuffle=True, worker_init_fn=np.random.seed(42))



# val_loader

val_file = "/kaggle/working/sampled_train_all.parquet"
val_df = pd.read_parquet(val_file)

# å¾� df è®€å�– SMILES
val_smiles = val_df['molecule_smiles'].tolist()


val_bind = val_df[["BRD4", "HSA", "sEH"]].values


num_val = len(val_smiles)

# å¹³è¡Œè™•ç�† SMILES è½‰ Graph
with Pool(processes=8) as pool:  # è¨­ç‚º 8 æ ¸å¿ƒï¼Œé�¿å…�è¨˜æ†¶é«”çˆ†ç‚¸
    val_graph = list(tqdm(pool.imap(smile_to_graph, val_smiles), total=num_val))


val_graph = to_pyg_list(val_graph)

num_val_samples = 20000  # å�– 20,000 ç­†
total_val_samples = len(val_graph)

# éš¨æ©Ÿé�¸å�– 20,000 å€‹ç´¢å¼•ï¼ˆå¦‚æ�œæ•¸æ“šå°‘æ–¼ 20,000ï¼Œå‰‡å…¨éƒ¨é�¸å�–ï¼‰
random_indices = np.random.choice(total_val_samples, min(num_val_samples, total_val_samples), replace=False)

# é�¸å�–å°�æ‡‰çš„ Graph å’Œ Binding Data
val_graph_subset = [val_graph[i] for i in random_indices]
val_bind_subset = val_bind[random_indices] 
val_bind =val_bind_subset


# å»ºç«‹ DataLoader
val_loader = PyGDataLoader(val_graph_subset, batch_size=3, shuffle=True)



## example training loop
scaler = torch.cuda.amp.GradScaler(enabled=True)
#net = Net()
net = torch.load("/kaggle/input/gnn/pytorch/default/1/gnn_model_finish.pth")
net.to(DEVICE)

optimizer =\
	torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)

num_epoch=5
epoch=0
iteration=0


while epoch<num_epoch: 
    for t, graph_batch in enumerate(train_loader): 
        index = graph_batch.idx.tolist()
        B = len(index)
        batch = dotdict(
			graph  = graph_batch.to(DEVICE),
			bind   = torch.from_numpy(train_bind[index]).to(DEVICE),
		)

        net.train()
        net.output_type = ['loss', 'infer']
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)  #data_parallel(net,batch) 
            bce_loss = output['bce_loss']

        optimizer.zero_grad() 
        scaler.scale(bce_loss).backward() 
        scaler.step(optimizer)
        scaler.update()
		 
        torch.clear_autocast_cache()
        print(epoch,iteration,bce_loss.item())
        iteration +=  1

    # ===== Validation Step =====
    net.eval()  # åˆ‡æ�›ç‚ºè©•ä¼°æ¨¡å¼�
    val_loss = 0.0
    num_batches = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, val_graph_batch in enumerate(val_loader):  # ä½¿ç”¨ enumerate ç¢ºä¿� batch ç´¢å¼•
            if num_samples >= num_val_samples:
                break  # é�”åˆ° 50,000 ç­†å¾Œçµ�æ�Ÿé©—è­‰
    
            # é€™è£¡ä¸�èƒ½ç”¨ val_graph_batch.idx.tolist()ï¼Œæ”¹æˆ�ç›´æ�¥å�– batch_idx
            val_batch = dotdict(
                graph = val_graph_batch.to(DEVICE),
                bind = torch.from_numpy(val_bind_subset[batch_idx * 3 : (batch_idx + 1) * 3]).to(DEVICE),
            )
    
            net.output_type = ['loss', 'infer']
            val_output = net(val_batch)
            val_loss += val_output['bce_loss'].item()
            num_batches += 1
            num_samples += len(val_batch.graph)
    
    avg_val_loss = val_loss / num_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")

    epoch += 1

'''



import numpy as np
import torch
from torch_geometric.data import Data
from multiprocessing import Pool
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# è½‰æ�›ç‚º PyG æ ¼å¼�
def to_pyg_list(graph):
	L = len(graph)
	for i in tqdm(range(L)):
		N, edge, node_feature, edge_feature = graph[i]
		graph[i] = Data(
			idx=i,
			edge_index=torch.from_numpy(edge.T).int(),
			x=torch.from_numpy(node_feature).byte(),
			edge_attr=torch.from_numpy(edge_feature).byte(),
		)
	return graph


train_file = '/kaggle/input/trainall7/sampled_train_all_7.parquet'
df = pd.read_parquet(train_file)


# train_loader
# å¾� df è®€å�– SMILES
train_smiles = df['molecule_smiles'].tolist()


train_bind = df[["BRD4", "HSA", "sEH"]].values
#train_bind = np.array([[1, 0, 0] if b == 1 else [0, 0, 0] for b in train_bind])  # å�‡è¨­ binds æ˜¯ 0/1

num_train = len(train_smiles)

# å¹³è¡Œè™•ç�† SMILES è½‰ Graph
with Pool(processes=8) as pool:  # è¨­ç‚º 8 æ ¸å¿ƒï¼Œé�¿å…�è¨˜æ†¶é«”çˆ†ç‚¸
    train_graph = list(tqdm(pool.imap(smile_to_graph, train_smiles), total=num_train))


train_graph = to_pyg_list(train_graph)
#train_loader = PyGDataLoader(train_graph, batch_size=3, shuffle=True)
#train_loader = PyGDataLoader(train_graph, batch_size=3, shuffle=True, worker_init_fn=np.random.seed(42))
train_loader = PyGDataLoader(train_graph, batch_size=9, shuffle=True, drop_last=False, worker_init_fn=np.random.seed(42))


# val_loader

val_file = "/kaggle/input/trainall1/sampled_train_all_1.parquet"
val_df = pd.read_parquet(val_file)

# å¾� df è®€å�– SMILES
val_smiles = val_df['molecule_smiles'].tolist()


val_bind = val_df[["BRD4", "HSA", "sEH"]].values


num_val = len(val_smiles)

# å¹³è¡Œè™•ç�† SMILES è½‰ Graph
with Pool(processes=8) as pool:  # è¨­ç‚º 8 æ ¸å¿ƒï¼Œé�¿å…�è¨˜æ†¶é«”çˆ†ç‚¸
    val_graph = list(tqdm(pool.imap(smile_to_graph, val_smiles), total=num_val))


val_graph = to_pyg_list(val_graph)

num_val_samples = 20000  # å�– 20,000 ç­†
total_val_samples = len(val_graph)

# éš¨æ©Ÿé�¸å�– 20,000 å€‹ç´¢å¼•ï¼ˆå¦‚æ�œæ•¸æ“šå°‘æ–¼ 20,000ï¼Œå‰‡å…¨éƒ¨é�¸å�–ï¼‰
random_indices = np.random.choice(total_val_samples, min(num_val_samples, total_val_samples), replace=False)

# é�¸å�–å°�æ‡‰çš„ Graph å’Œ Binding Data
val_graph_subset = [val_graph[i] for i in random_indices]
val_bind_subset = val_bind[random_indices] 
val_bind =val_bind_subset


# å»ºç«‹ DataLoader
val_loader = PyGDataLoader(val_graph_subset, batch_size=9, shuffle=False, drop_last=False)




## example training loop
scaler = torch.cuda.amp.GradScaler(enabled=True)
# scaler = torch.amp.GradScaler(enabled=True)

#net = Net()
net = torch.load("/kaggle/input/gnn-v8-2/pytorch/default/1/gnn_model_finish8-2.pth")
net.to(DEVICE)
#net = torch.load("/kaggle/input/gnn-/pytorch/default/1/gnn_model_finish_4.pth", map_location=torch.device('cpu'))


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0004)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)

num_epoch=9
epoch=0
iteration=0


while epoch < num_epoch:
    
    epoch_loss = 0.0  # ç´€éŒ„æ•´å€‹ epoch çš„ loss

    for t, graph_batch in enumerate(train_loader): 
        epoch_indices = []  # å„²å­˜ index ä»¥æª¢æŸ¥è·³è®Š
        index = graph_batch.idx.tolist()
        epoch_indices.extend(index)
        B = len(index)

        batch = dotdict(
            graph = graph_batch.to(DEVICE),
            bind = torch.from_numpy(train_bind[index]).to(DEVICE),
        )

        # ğŸ“Œ è¨˜éŒ„ç•¶å‰� epoch æ‰€æœ‰ batch index
        print(f"Epoch {epoch+1}, Batch {t+1} - Index range: {min(epoch_indices)} â†’ {max(epoch_indices)}")

        net.train()
        net.output_type = ['loss', 'infer']
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            output = net(batch)
            bce_loss = output['bce_loss']

        optimizer.zero_grad()
        scaler.scale(bce_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()  # æ¸…é™¤ CUDA ç·©å­˜

        # ğŸ“Œ æ›´æ–° epoch ç¸½ loss
        epoch_loss += bce_loss.item()

        print(f"Epoch {epoch+1}, Iteration {iteration}, BCE Loss: {bce_loss.item()}")
        iteration += 1

    # è¨ˆç®— epoch å¹³å�‡ loss ä¸¦æ›´æ–°å­¸ç¿’ç�‡
    epoch_loss /= len(train_loader)  # è¨ˆç®— loss å¹³å�‡å€¼
    scheduler.step(epoch_loss)  # ä½¿ç”¨ ReduceLROnPlateau æ›´æ–° lr

    print(f"Epoch {epoch+1} çµ�æ�Ÿï¼Œå¹³å�‡ BCE Loss: {epoch_loss}, ç•¶å‰�å­¸ç¿’ç�‡: {optimizer.param_groups[0]['lr']}")
    epoch += 1

    
    # ===== Validation Step =====
    net.eval()  # åˆ‡æ�›ç‚ºè©•ä¼°æ¨¡å¼�
    val_loss = 0.0
    num_batches = 0
    num_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, val_graph_batch in enumerate(val_loader):  # ä½¿ç”¨ enumerate ç¢ºä¿� batch ç´¢å¼•
            if num_samples >= num_val_samples:
                break  # é�”åˆ° 50,000 ç­†å¾Œçµ�æ�Ÿé©—è­‰
    
            val_batch = dotdict(
                graph = val_graph_batch.to(DEVICE),
                bind = torch.from_numpy(val_bind_subset[batch_idx * 9 : (batch_idx + 1) * 9]).to(DEVICE),
            )

    
            net.output_type = ['loss', 'infer']
            val_output = net(val_batch)
            val_loss += val_output['bce_loss'].item()
            num_batches += 1
            num_samples += len(val_batch.graph)
    
            # å�–å¾—é �æ¸¬å€¼ï¼ˆé€šå¸¸ç‚ºæ©Ÿç�‡ï¼‰ï¼Œè½‰ç‚º 0/1
            probs = val_output['bind'].detach().cpu().numpy()  # è½‰ç‚º NumPy
            preds = (probs > 0.5).astype(int)  # è¨­å®šé–¾å€¼ 0.5
            labels = val_batch.bind.cpu().numpy()
    
            all_preds.append(preds)
            all_labels.append(labels)
    
    # è¨ˆç®—å¹³å�‡ Loss
    avg_val_loss = val_loss / num_batches
    
    # å°‡æ‰€æœ‰ batch çš„é �æ¸¬å€¼èˆ‡æ¨™ç±¤å�ˆä½µ
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # è¨ˆç®— Accuracy å’Œ F1 Score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')  # 'macro' è¨ˆç®—æ‰€æœ‰é¡�åˆ¥çš„å¹³å�‡ F1
    
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

    epoch += 1




model_path = "/kaggle/working/gnn_model_finish.pth"
torch.save(net, model_path)
print(f"æ¨¡å�‹å·²ä¿�å­˜è‡³ {model_path}")



import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from multiprocessing import Pool

# è¨­å®šè¨­å‚™ (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# è¼‰å…¥æ¨¡å�‹ä¸¦åˆ‡æ�›ç‚ºæ�¨ç�†æ¨¡å¼�
net = torch.load("/kaggle/input/gnn-v9-9-2/pytorch/default/1/gnn_model_finish9.pth", map_location=DEVICE)
net.to(DEVICE)
net.eval()


# è¨­å®šè®€å�– Parquet æ–‡ä»¶
filename = '/kaggle/input/test-110/sampled_test_all.parquet'
df = pd.read_parquet(filename)

# æ��å�–æ¸¬è©¦æ•¸æ“š
test_smiles = df['molecule_smiles'].tolist()
y_true = df[["BRD4", "HSA", "sEH"]].values  # è½‰ç‚º NumPy (Shape: [N, 3])

num_test = len(test_smiles)

# è½‰æ�› SMILES â†’ Graphï¼ˆä½¿ç”¨ 8 æ ¸å¿ƒåŠ é€Ÿï¼‰
with Pool(processes=8) as pool:
    test_graph = list(tqdm(pool.imap(smile_to_graph, test_smiles), total=num_test))

# è½‰æ�›ç‚º PyG æ ¼å¼�
test_graph = to_pyg_list(test_graph)

# å»ºç«‹ PyG DataLoader
test_loader = PyGDataLoader(test_graph, batch_size=3, shuffle=False, drop_last=False, worker_init_fn=np.random.seed(42))



import pandas as pd


y_true_df = pd.DataFrame(y_true)  
y_true_df = y_true_df.fillna(0.0)
y_true_df.isnull().sum()
y_true_df


# ===== é€²è¡Œæ�¨è«– =====
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(DEVICE)  # ç¢ºä¿�æ•¸æ“šåœ¨æ­£ç¢ºè¨­å‚™ä¸Š
        
        # æ§‹é€ æ¨¡å�‹è¼¸å…¥
        test_batch = {"graph": batch}

        # ç�²å�–æ¨¡å�‹è¼¸å‡º
        net.output_type = ['infer']  # è¨­ç½®ç‚ºæ�¨ç�†æ¨¡å¼�
        test_output = net(test_batch)

        # ç�²å�–é �æ¸¬æ©Ÿç�‡
        probs = test_output["bind"].detach().cpu().numpy()  # Shape: [batch_size, 3]
        
        # è½‰æ�›ç‚º 0/1 é �æ¸¬å€¼ (é–¾å€¼ 0.5)
        preds = (probs > 0.5).astype(int)

        # å„²å­˜çµ�æ�œ
        all_probs.append(probs)
        all_preds.append(preds)

# å�ˆä½µæ‰€æœ‰ batch çš„é �æ¸¬çµ�æ�œ
all_probs = np.vstack(all_probs)  # é �æ¸¬æ©Ÿç�‡
all_preds = np.vstack(all_preds)  # äºŒå…ƒé �æ¸¬çµ�æ�œ
all_labels = y_true_df.to_numpy()

# ===== è¨ˆç®—æŒ‡æ¨™ =====
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')  # 'macro' è¨ˆç®—æ‰€æœ‰é¡�åˆ¥çš„å¹³å�‡ F1
auc = roc_auc_score(all_labels, all_probs, average='macro')  # ç›´æ�¥ä½¿ç”¨æ©Ÿç�‡è¨ˆç®— AUC

# ===== è¼¸å‡ºçµ�æ�œ =====

print(f"Test F1 Score: {f1:.4f}")
print(f"Test AUC Score: {auc:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")



import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from multiprocessing import Pool

# è¨­å®šè¨­å‚™ (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# è¼‰å…¥æ¨¡å�‹ä¸¦åˆ‡æ�›ç‚ºæ�¨ç�†æ¨¡å¼�
net = torch.load("/kaggle/input/gnn-v9-9-2/pytorch/default/1/gnn_model_finish9.pth", map_location=DEVICE)
net.to(DEVICE)
net.eval()


import duckdb
import pandas as pd
from tqdm import tqdm
import numpy as np # linear algebra


import os

# Process the test.parquet file chunk by chunk
test_file = '/kaggle/input/leash-BELKA/test.csv'
output_file = 'submission_gnn.csv'  # Specify the path and filename for the output file

test = pd.read_csv(test_file)
test.shape


protein_mapping = {
    "BRD4": 0,
    "HSA": 1,
    "sEH": 2
}

test['protein_name'] = test['protein_name'].map(protein_mapping)

test


'''
# æ��å�–æ¸¬è©¦æ•¸æ“š
test_smiles = df['molecule_smiles'].tolist()


num_test = len(test_smiles)

# è½‰æ�› SMILES â†’ Graphï¼ˆä½¿ç”¨ 8 æ ¸å¿ƒåŠ é€Ÿï¼‰
with Pool(processes=8) as pool:
    test_graph = list(tqdm(pool.imap(smile_to_graph, test_smiles), total=num_test))

# è½‰æ�›ç‚º PyG æ ¼å¼�
test_graph = to_pyg_list(test_graph)

# å»ºç«‹ PyG DataLoader
test_loader = PyGDataLoader(test_graph, batch_size=16, shuffle=False, drop_last=False)

all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):

        batch = batch.to(DEVICE)  # ç¢ºä¿�æ•¸æ“šåœ¨æ­£ç¢ºè¨­å‚™ä¸Š
                
        # æ§‹é€ æ¨¡å�‹è¼¸å…¥
        test_batch = {"graph": batch}
        
        # ç�²å�–æ¨¡å�‹è¼¸å‡º
        net.output_type = ['infer']  # è¨­ç½®ç‚ºæ�¨ç�†æ¨¡å¼�
        test_output = net(test_batch)
        
        # ç�²å�–é �æ¸¬æ©Ÿç�‡
        probs = test_output["bind"].detach().cpu().numpy()  # Shape: [batch_size, 3]
        probs = pd.DataFrame(probs.values.flatten()
        
        # è½‰æ�›ç‚º 0/1 é �æ¸¬å€¼ (é–¾å€¼ 0.5)
        preds = (probs > 0.5).astype(int)
        
        # å„²å­˜çµ�æ�œ
        all_preds.append(preds)

# Create a DataFrame with 'id' and 'probability' columns
output_df = pd.DataFrame({'id': test['id'], 'binds': all_preds})

# Save the output DataFrame to a CSV file
output_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
'''


# è½‰æ�›ç‚º PyG æ ¼å¼�
def to_pyg_list(graph):
	L = len(graph)
	for i in tqdm(range(L)):
		N, edge, node_feature, edge_feature = graph[i]
		graph[i] = Data(
			idx=i,
			edge_index=torch.from_numpy(edge.T).int(),
			x=torch.from_numpy(node_feature).byte(),
			edge_attr=torch.from_numpy(edge_feature).byte(),
		)
	return graph


import torch
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool

BATCH_SIZE = 3  # æ¸›å°‘ batch_size ä¾†é™�ä½�è¨˜æ†¶é«”è² æ“”
CHUNK_SIZE = 50000  # æ¯�æ¬¡è™•ç�† 50000 ç­†
OUTPUT_FILE = "output.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# **ç¢ºä¿� CSV ä¸�æœƒé‡�è¤‡å¯«å…¥ header**
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)  # å…ˆåˆªé™¤ï¼Œé�¿å…�é‡�è¤‡ append

# **é€�æ‰¹è™•ç�† SMILES**
for chunk_start in range(0, len(test), CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, len(test))
    test_chunk = test.iloc[chunk_start:chunk_end]  # **å�–å‡ºä¸€å°�éƒ¨åˆ†**
    
    test_smiles = test_chunk['molecule_smiles'].tolist()
    test_protein_names = test_chunk['protein_name'].tolist()
    test_ids = test_chunk['id'].tolist()
    
    # **é€�æ‰¹è½‰æ�› SMILES â†’ Graph**
    with Pool(processes=8) as pool:
        test_graph = list(tqdm(pool.imap(smile_to_graph, test_smiles), total=len(test_smiles)))

    # **è½‰æ�›ç‚º PyG æ ¼å¼�**
    test_graph = to_pyg_list(test_graph)

    # **å»ºç«‹ DataLoader**
    test_loader = PyGDataLoader(test_graph, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    all_preds = []  # **å­˜æ”¾ç•¶å‰� chunk çš„é �æ¸¬çµ�æ�œ**
    batch_start = 0  # **æ‰¹æ¬¡ç´¢å¼•è¿½è¹¤**

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {chunk_start}/{len(test)}"):
            batch_size = batch.num_graphs
            batch = batch.to(DEVICE)

            # **æ§‹é€ æ¨¡å�‹è¼¸å…¥**
            test_batch = {"graph": batch}
            net.output_type = ['infer']
            test_output = net(test_batch)  # **è¼¸å‡ºå½¢ç‹€ [batch_size, num_proteins]**

            for i in range(batch_size):  
                protein_name = test_protein_names[batch_start + i]

                if protein_name >= test_output["bind"].size(1):
                    print(f"Warning: protein_name {protein_name} out of bounds for batch {i}")
                    continue
                
                protein_pred = test_output["bind"][i, protein_name]
                all_preds.append(1 if protein_pred > 0.5 else 0)

            batch_start += batch_size

    # **å¯«å…¥ CSVï¼Œæ¸›å°‘è¨˜æ†¶é«”ä½”ç”¨**
    output_df = pd.DataFrame({'id': test_ids, 'binds': all_preds})
    output_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=not os.path.exists(OUTPUT_FILE))

    del test_graph, test_loader, all_preds  # **é‡‹æ”¾è¨˜æ†¶é«”**
    torch.cuda.empty_cache()  # **æ¸…ç�† GPU è¨˜æ†¶é«”**



import pandas as pd

input_file = '/kaggle/working/output.csv'
output_file = '/kaggle/working/submission_gnn.csv'
# è®€å�–å·²å„²å­˜çš„ CSV
output_df = pd.read_csv(input_file)

print(len(output_df))

# ä¿®æ”¹ id æ¬„ä½�
output_df['id'] = range(295246830, 295246830 + len(output_df))

print(output_df.shape)

# å°‡ä¿®æ”¹å¾Œçš„ DataFrame å„²å­˜å›� CSV
output_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))


