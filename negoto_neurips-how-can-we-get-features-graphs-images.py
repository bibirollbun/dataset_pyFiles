import numpy as np
import pandas as pd
from tqdm import tqdm

# Load training dataset
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
# train = train[:5] # COLD RUN
display(train.head(3))


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


%%time
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

# Convert SMILES strings into "Mol" objects
train_smiles = train['SMILES'].to_list()
train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]

# Generate RDKit descriptor features
train_RDKit = pd.DataFrame([Descriptors.CalcMolDescriptors(mol) for mol in train_mols])
train_RDKit.head()


%%time
from mordred import Calculator, descriptors

# Generate mordred descriptor features (AtomCount)
descList = [
    descriptors.AcidBase,
    descriptors.Aromatic,
    descriptors.AtomCount,
#    descriptors.BertzCT, # Duplicated (even if this is not a RDKit wrapper)
    descriptors.BondCount,
#    descriptors.CarbonTypes, # Missing objects returned for some SMILES strings
    descriptors.EccentricConnectivityIndex,
#    descriptors.ExtendedTopochemicalAtom, # Missing objects returned for some SMILES strings
    descriptors.FragmentComplexity,
    descriptors.Framework,
    descriptors.InformationContent,
#    descriptors.KappaShapeIndex, # Missing objects returned for some SMILES strings
    descriptors.Lipinski,
    descriptors.McGowanVolume,
    descriptors.MolecularId,
    descriptors.PathCount, # Computationally heavy
    descriptors.Polarizability,
    descriptors.RingCount,
    descriptors.TopologicalIndex,
    descriptors.VertexAdjacencyInformation,
    descriptors.WalkCount,
    descriptors.Weight,
    descriptors.WienerIndex,
    descriptors.ZagrebIndex,
]
train_mordred = Calculator(descList).pandas(train_mols)
train_mordred.head()


%%time
from rdkit.Chem import AllChem

# Generate MACCS Keys Fingerprint Features
train_MACCS = pd.DataFrame([np.array(AllChem.GetMACCSKeysFingerprint(mol)) for mol in train_mols])
train_MACCS.columns = [f"MACCS_{i}" for i in range(167)]
train_MACCS.head()


%%time

# Generate Morgan Fingerprint Features (nBits = 512)
nBits = 512 # 1024, 2048, 4096
train_Morgan = pd.DataFrame([np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits = nBits)) for mol in train_mols])
train_Morgan.columns = [f"Morgan_{i}" for i in range(nBits)]
train_Morgan.head()


%%time
import torch
from transformers import AutoTokenizer, AutoModel

# Load ChemBERTa Model from Huggingface (https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
# path = "DeepChem/ChemBERTa-77M-MLM" # need to pre-download to use in internet disabled notebooks 
path = "/kaggle/input/c/transformers/default/1/ChemBERTa-77M-MLM"
ChemBERTa = AutoModel.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)


# Examples of tokens and model outputs
with torch.no_grad():
    for smiles in train_smiles[:3]:
        tokens = tokenizer(smiles, return_tensors = "pt", padding = True, truncation = True)
        output = ChemBERTa(**tokens)
        print(f"SMILES : {smiles}\nTokens : {tokens['input_ids'][0]}\nOutput Shape : {output[0].shape}")
        print(f"Head of [CLS] vector : {output.last_hidden_state[:, 0, :].squeeze().numpy()[:5]}\n--------")


# Generate ChemBERTa-based SMILES embeddings
CLS = []
mean_pooling = []
with torch.no_grad():
    for smiles in tqdm(train_smiles):
        tokens = tokenizer(smiles, return_tensors = "pt", padding = True, truncation = True)
        output = ChemBERTa(**tokens)
        CLS.append(output.last_hidden_state[:, 0, :].squeeze())
        mean_pooling.append(torch.mean(output.last_hidden_state, 1).squeeze())

train_ChemBERTaCLS = pd.DataFrame(np.array(CLS))
train_ChemBERTaCLS.columns = [f"BERTCLS_{i}" for i in range(train_ChemBERTaCLS.shape[-1])]
display(train_ChemBERTaCLS.head())

train_ChemBERTaMP = pd.DataFrame(np.array(mean_pooling))
train_ChemBERTaMP.columns = [f"BERTMP_{i}" for i in range(train_ChemBERTaMP.shape[-1])]
display(train_ChemBERTaMP.head())


!pip install torch_geometric --no-index --find-links=file:///kaggle/input/torch-geometric


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

def get_atom_features(atom):
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetChiralTag()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.GetHybridization()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]

def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype = torch.float)
    edge_indices = []
    if mol.GetNumBonds() > 0:
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            edge_index = torch.tensor(edge_indices, dtype = torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype = torch.long)
    data = Data(x = x, edge_index = edge_index)
    return data

# Create graphs as torch_geometric.Data objects
N = 5
graphs = [smiles_to_graph(smiles) for smiles in train_smiles[:N]]


for i in range(1):
    print(f"SMILES : {train_smiles[i]}")
    print(f"Number of Atoms (Nodes) : {graphs[0].num_nodes}")
    print("Atom (Node) Features Matrix :")
    display(graphs[i].x)
    print("Bond (Edge) Index :")
    display(graphs[i].edge_index)
    print("--------")


# Visualize SMILES as graph using networkx and matplotlib.pyplot
import networkx as nx
import matplotlib.pyplot as plt

num2char = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl', 0: '*'}

for n in range(N):
    G = nx.Graph()
    for i in range(graphs[n].num_nodes):
        G.add_node(i, feature = graphs[n].x[i])
    G.add_edges_from(graphs[n].edge_index.t().tolist())
    
    node_labels = {i: f"{num2char[int(G.nodes[i]['feature'][0])]}" for i in G.nodes}
    plt.figure(figsize = (3, 3))
    nx.draw(
        G,
        nx.spring_layout(G),
        with_labels = True,
        labels = node_labels,
        node_color = "white", 
        node_size = 200,
        font_size = 12
    )
    print(f"SMILES : {train_smiles[n]}")
    plt.show()


from rdkit.Chem import Draw
Draw.MolsToGridImage(train_mols[:4], subImgSize = (300, 160), molsPerRow = 2)


train_RDKit.to_csv("train_RDKit.csv")
train_mordred.to_csv("train_mordred.csv")
train_MACCS.to_csv("train_MACCS.csv")
train_Morgan.to_csv("train_Morgan.csv")
train_ChemBERTaCLS.to_csv("train_ChemBERTaCLS.csv")
train_ChemBERTaMP.to_csv("train_ChemBERTaMP.csv")

