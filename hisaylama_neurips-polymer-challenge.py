#Install all dependencies
#!pip install rdkit > /dev/null 2>&1
#!pip install torch torchvision torchaudio torch-geometric > /dev/null 2>&1


#Install rdkit offline
!pip install /kaggle/input/rdkit-install-wheel/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


#Install PyTorch geometric offline
!pip install --no-index\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/idna-3.4-py3-none-any.whl \
#/kaggle/input/pytorch-geometric/PyTorch-Geometric/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/Jinja2-3.1.2-py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/urllib3-1.26.12-py2.py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/torch_sparse-0.6.15-cp37-cp37m-linux_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/tqdm-4.64.1-py2.py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/charset_normalizer-2.1.1-py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/pyparsing-3.0.9-py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/joblib-1.2.0-py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/scikit_learn-1.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/threadpoolctl-3.1.0-py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/certifi-2022.9.24-py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/MarkupSafe-2.1.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/requests-2.28.1-py3-none-any.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/numpy-1.21.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl\
/kaggle/input/pytorch-geometric/PyTorch-Geometric/torch_geometric-2.1.0.post1-py3-none-any.whl


#For handling data, data visualisation, basic math and algebra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Handling directory
import os

#For statistical analysis
import seaborn as sns

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

#For Machine learning
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import random, numpy as np, torch


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Check the training datasets
df_train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
print("--- Display the training dataset top 5 rows---")
print(df_train.head())
print("-----------------------------------------------------------------------------")
#Check the test datasets
df_test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
print("--- Display the test dataset top 5 rows---")
print(df_test.head())


# Define target_variables
target_variables = ["Tg", "FFV", "Tc", "Density", "Rg"] 

# Use plain white background
sns.set_style("whitegrid")  # or "white" if you want no grid

# Clean inf values in target columns
df_train[target_variables] = df_train[target_variables].replace([np.inf, -np.inf], np.nan)

print("\n ---- Distribution of target variables ----")
num_targets = len(target_variables)
rows = (num_targets + 2) //3 #3 plots per row

plt.figure(figsize = (15, 5 * rows))

for i, col in enumerate(target_variables):
    plt.subplot(rows, 3, i + 1)
    sns.histplot(df_train[col].dropna(), kde=True, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of {col}", fontsize=12)
    plt.xlabel(col, fontsize = 10)
    plt.ylabel("Count", fontsize = 10)
    plt.xticks(fontsize = 9)
    plt.yticks(fontsize = 9)
plt.tight_layout
plt.show()


#Display all the missing entries in the dataframe pertaining to the columns 
import matplotlib.pyplot as plt
# Count missing values per column
missing_values = df_train.isna().sum()

# Bar chart of missing values per column
plt.figure(figsize=(8, 5))
missing_values.plot(kind='bar')
plt.title("Number of Missing Values per Column")
plt.ylabel("Count of NaN")
plt.xlabel("Column")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.grid(True)
plt.show()


# Correlation Matrix for Target Variables
print("\n--- Correlation Matrix of Target Variables ---")
plt.figure(figsize=(8, 6))
sns.heatmap(df_train[target_variables].corr(), annot=True, cmap='crest', fmt=".2f")
plt.title("Correlation Matrix of Target Properties", color="#FFFFFF")


# Assuming df_train is your DataFrame with a 'SMILES' column, add new descriptors
df_train['MolWt'] = df_train['SMILES'].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
df_train['TPSA'] = df_train['SMILES'].apply(lambda s: Descriptors.TPSA(Chem.MolFromSmiles(s)))
df_train['NumHDonors'] = df_train['SMILES'].apply(lambda s: Descriptors.NumHDonors(Chem.MolFromSmiles(s)))
df_train['LogP'] = df_train['SMILES'].apply(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s)))

#print(df_train.info())


new_function = ["Tg", "FFV", "Tc", "Density", "Rg", "MolWt", "TPSA", "NumHDonors", "LogP"] 
# Correlation Matrix for Target Variables
print("\n--- Correlation Matrix of Target Polymer Properties ---")
plt.figure(figsize=(8, 6))
sns.heatmap(df_train[new_function].corr(), annot=True, cmap='crest', fmt=".2f")
plt.title("Correlation Matrix of Target Properties", color="#FFFFFF")


# Assuming your DataFrame is called df
total_rows = len(df_train)

# Calculate % of non-missing and missing for each column
percent_non_missing = 100 * df_train.notnull().sum() / total_rows
percent_missing = 100 * df_train.isnull().sum() / total_rows

# Combine into a nice summary table
summary = pd.DataFrame({
    'Non-missing %': percent_non_missing.round(2),
    'Missing %': percent_missing.round(2),
    'Non-missing Count': df_train.notnull().sum(),
    'Missing Count': df_train.isnull().sum()
})

print(summary)


from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import warnings
from tqdm import TqdmWarning

# Suppress tqdm warning in notebooks
warnings.filterwarnings("ignore", category=TqdmWarning)

# --- Optional PyG import with graceful fallback -------------------------------
try:
    from torch_geometric.data import Data as _PyGData
    HAS_PYG = True
except Exception:
    HAS_PYG = False

    class _PyGData:
        """Minimal stand-in for torch_geometric.data.Data."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @property
        def num_nodes(self):
            return int(self.x.size(0)) if hasattr(self, "x") else 0

        def to(self, device):
            for k, v in list(self.__dict__.items()):
                if torch.is_tensor(v):
                    setattr(self, k, v.to(device))
            return self

        def __repr__(self):
            keys = ", ".join(sorted(self.__dict__.keys()))
            return f"Data({keys})"

# Alias so you can keep using `Data(...)` regardless of environment
Data = _PyGData

# --- Feature helpers ----------------------------------------------------------
def atom_features(atom):
    """Tensor of atomic features [Z, degree, formal charge, aromatic]."""
    return torch.tensor([
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
    ], dtype=torch.float)

def mol_descriptors(mol):
    """Global molecular descriptors as a dict."""
    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
    }

def bond_features(bond):
    """Tensor of bond features [type, conjugated, in_ring]."""
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ], dtype=torch.float)

# --- SMILES → Graph -----------------------------------------------------------
def smiles_to_graph(smiles, extra_features=None, use_mol_descriptors=False):
    """
    Convert a SMILES string into a (PyG-like) Data object with:
      - x:        [N, F_atom] node features
      - edge_index: [2, E] directed edges (undirected stored twice)
      - edge_attr:  [E, F_edge] bond features
      - extra:    optional [F_extra] global features tensor
    Returns None for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.stack(atom_feats) if atom_feats else torch.empty((0, 4), dtype=torch.float)

    # Edge list & attributes
    ei, ea = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        ei.extend([(i, j), (j, i)])  # undirected → two directed edges
        ea.extend([bf, bf])

    if ei:
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()  # [2, E]
        edge_attr  = torch.stack(ea)                                      # [E, 3]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 3), dtype=torch.float)

    # Global / extra features
    if use_mol_descriptors:
        desc_vals = list(mol_descriptors(mol).values())
        if extra_features is not None:
            extra_features = list(extra_features) + desc_vals
        else:
            extra_features = desc_vals

    # Package into Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if extra_features is not None:
        data.extra = torch.as_tensor(extra_features, dtype=torch.float)

    # Handy metadata (optional)
    data.smiles = smiles

    return data



#Graph -> tabular features 

def graph_to_features(data):
    """
    Convert a PyG Data (or one element from a batched object) to a 1D feature vector.
    Features include:
      - basic graph stats (nodes, bonds, degree stats)
      - node feature stats (mean/std per column of x)
      - edge feature stats (mean/std per column of edge_attr, if present)
      - extra graph-level features (data.extra), flattened
    Returns: (feat_vector: np.ndarray, feat_names: list[str])
    """
    feats = []
    names = []

    # Basic graph stats
    n_nodes = int(data.num_nodes)
    n_edges_dir = int(data.edge_index.size(1)) if hasattr(data, "edge_index") else 0
    n_bonds = n_edges_dir // 2  # PyG stores both directions
    feats += [n_nodes, n_bonds]
    names += ["n_nodes", "n_bonds"]

    # Degree stats (use both directions -> divide by 2)
    if hasattr(data, "edge_index") and data.edge_index is not None:
        ei = data.edge_index
        deg_out = torch.bincount(ei[0], minlength=n_nodes)
        deg_in  = torch.bincount(ei[1], minlength=n_nodes)
        deg = (deg_out + deg_in).float() / 2.0
        deg_mean = deg.mean().item()
        deg_std  = deg.std(unbiased=False).item()
        deg_max  = deg.max().item()
        feats += [deg_mean, deg_std, deg_max]
        names += ["deg_mean", "deg_std", "deg_max"]
    else:
        feats += [0.0, 0.0, 0.0]
        names += ["deg_mean", "deg_std", "deg_max"]

    # Node feature stats
    if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
        x = data.x.float()
        x_mean = x.mean(dim=0).cpu().numpy()
        x_std  = x.std(dim=0, unbiased=False).cpu().numpy()
        for j in range(x.shape[1]):
            feats.append(float(x_mean[j])); names.append(f"x_mean_{j}")
        for j in range(x.shape[1]):
            feats.append(float(x_std[j]));  names.append(f"x_std_{j}")
    else:
        # if unknown feature dim, skip
        pass

    # Edge feature stats
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
        ea = data.edge_attr.float()
        ea_mean = ea.mean(dim=0).cpu().numpy()
        ea_std  = ea.std(dim=0, unbiased=False).cpu().numpy()
        for j in range(ea.shape[1]):
            feats.append(float(ea_mean[j])); names.append(f"edge_mean_{j}")
        for j in range(ea.shape[1]):
            feats.append(float(ea_std[j]));  names.append(f"edge_std_{j}")

    # Graph-level extra features
    if hasattr(data, "extra") and data.extra is not None:
        extra = data.extra
        if torch.is_tensor(extra):
            extra = extra.detach().cpu().float().view(-1).numpy()
        else:
            extra = np.asarray(extra, dtype=np.float32).reshape(-1)
        for j, val in enumerate(extra):
            feats.append(float(val)); names.append(f"extra_{j}")

    return np.asarray(feats, dtype=np.float32), names


import xgboost as xgb
from xgboost import XGBRegressor
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

use_cuda = torch.cuda.is_available()

xgb_params = dict(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42,
    tree_method="hist",                     # keep "hist"
    device=("cuda" if use_cuda else "cpu"), # set device here (XGBoost >= 2.0)
)
if not use_cuda:
    xgb_params["n_jobs"] = -1

xgb_model = XGBRegressor(**xgb_params)
xgb_model #Print XGB _model


#Define the regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- metrics helpers ----------
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Pearson
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    # Spearman (optional)
    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        spearman = np.nan

    # MAPE (guard against zeros)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom))) * 100.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman,
        "MAPE_%": mape,
    }



def evaluate_regression(model, X_train, y_train, X_val, y_val, regression_metrics):
    """
    Evaluate a regression model with standard metrics and diagnostic plots.

    Parameters
    ----------
    model : object
        Trained regression model with a `.predict()` method.
    X_train, y_train : array-like
        Training features and targets.
    X_val, y_val : array-like
        Validation features and targets.
    regression_metrics : callable
        Function that takes (y_true, y_pred) and returns a dict of metrics.
        Expected keys: "R2", "MAE", "RMSE" (for plot titles).

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with Train and Validation metrics.
    """
    # ---------- Predictions ----------
    pred_tr = model.predict(X_train)
    pred_va = model.predict(X_val)

    # ---------- Metrics ----------
    train_metrics = regression_metrics(y_train, pred_tr)
    val_metrics   = regression_metrics(y_val,   pred_va)

    metrics_df = pd.DataFrame({"Train": train_metrics, "Validation": val_metrics})
    print("\n=== Regression Metrics ===")
    print(metrics_df.round(6).to_string())

    # ---------- Parity plot ----------
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_val, pred_va, s=18, alpha=0.75, label="Predicted vs True")
    lo = min(np.min(y_val), np.min(pred_va))
    hi = max(np.max(y_val), np.max(pred_va))
    plt.plot([lo, hi], [lo, hi], linewidth=2, label="Ideal y = x")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(
        f"Parity Plot (Validation)\n"
        f"R²={val_metrics['R2']:.3f} | MAE={val_metrics['MAE']:.4f} | RMSE={val_metrics['RMSE']:.4f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Residuals vs Fitted ----------
    res_val = pred_va - y_val
    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(pred_va, res_val, s=16, alpha=0.75)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residuals vs Fitted (Validation)")
    plt.tight_layout()
    plt.show()

    # ---------- Residual histogram ----------
    # plt.figure(figsize=(6.2, 4.2))
    # plt.hist(res_val, bins=30)
    # plt.xlabel("Residual")
    # plt.ylabel("Count")
    # plt.title("Residual Distribution (Validation)")
    # plt.tight_layout()
    # plt.show()

    return metrics_df


# --- imports you need here ---
import random
import numpy as np
import torch

# your `smiles_to_graph` from earlier must already be defined

# Prepare the training dataset
target_variable_1 = "Density"
df_train_clean = df_train[["SMILES", target_variable_1]]
df_train_new = df_train_clean.dropna()

extra_cols = []
graphs = []
for _, row in df_train_new.iterrows():
    graph = smiles_to_graph(
        smiles=row["SMILES"],
        extra_features=row[extra_cols],   # additional columns (empty is fine)
        use_mol_descriptors=True          # add RDKit descriptors
    )
    if graph is not None:
        graph.y = torch.tensor([row[target_variable_1]], dtype=torch.float)
        graph.smiles = row["SMILES"]
        graphs.append(graph)

# --- Robust DataLoader (PyG if available, fallback otherwise) ---
try:
    # Will fail in environments missing torch_sparse/torch_scatter, etc.
    from torch_geometric.loader import DataLoader as PyGDataLoader
    HAS_PYG_LOADER = True
except Exception:
    HAS_PYG_LOADER = False

    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset

    class GraphDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    def _simple_collate(batch):
        # For batch_size==1 return the single item; otherwise return a list
        return batch[0] if len(batch) == 1 else batch

    def make_fallback_dataloader(items, batch_size=1, shuffle=True, generator=None, num_workers=0):
        ds = GraphDataset(items)
        return TorchDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=num_workers,
            collate_fn=_simple_collate,
        )

# Seeding
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
g = torch.Generator().manual_seed(SEED)

# Create `loader`
if HAS_PYG_LOADER:
    loader = PyGDataLoader(graphs, batch_size=1, shuffle=True, generator=g, num_workers=0)
else:
    loader = make_fallback_dataloader(graphs, batch_size=1, shuffle=True, generator=g, num_workers=0)

# (Optional) quick sanity check
first_item = next(iter(loader))
print(type(first_item), getattr(first_item, "smiles", None), first_item.y if hasattr(first_item, "y") else None)



# Display the feautre loaded in the graph
for i, data in enumerate(loader):
    print(f"\n--- Graph {i+1} ---")
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Node feature shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge attr shape: {data.edge_attr.shape}")
    print(f"Target: {data.y}")
    if hasattr(data, "extra"):
        print(f"Extra features: {data.extra}")

    # ---- Get a proper SMILES string (PyG collates strings into lists) ----
    smiles = getattr(data, "smiles", None)
    if isinstance(smiles, (list, tuple)) and smiles:
        smiles = smiles[0]
    if (smiles is None) and hasattr(data, "row_idx"):
        idx = int(data.row_idx[0].item()) if torch.is_tensor(data.row_idx) else int(data.row_idx)
        smiles = str(df_train_new.loc[idx, "SMILES"])

    # --- Draw with atom & bond indices (robust across RDKit versions) ---
    try:
        
        # --- Clean drawing, no indices; show only molecular formula as legend ---
        from rdkit import Chem
        from rdkit.Chem import Draw, rdMolDescriptors
        from IPython.display import display

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            print(f"RDKit failed to parse SMILES: {smiles!r}")
        else:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            legend = f"{rdMolDescriptors.CalcMolFormula(mol)}   |   FFV: {float(data.y.squeeze()):.4f}"
            img = Draw.MolToImage(mol, size=(350, 300), legend=legend)
            display(img)

            #img = Draw.MolToImage(mol, size=(350, 300), legend=formula)  # legend = formula only
            #display(img)
            #display(Image(data=png_bytes))
            print("Molecular drawing with atom & bond indices shown.")
    except ImportError:
        print("(RDKit not installed — skipping molecule drawing)")

    # Stop after first (seeded) shuffled sample
    if i == 1:
        break


from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor

# --- Build feature and target arrays ---
X_rows, y_rows = [], []
feature_names = None

for batch in loader:
    # Handle single or multiple graphs per batch
    data_list = batch.to_data_list() if getattr(batch, "num_graphs", 1) > 1 else [batch]
    for d in data_list:
        x_vec, names = graph_to_features(d)
        if feature_names is None:
            feature_names = names
        X_rows.append(x_vec)

        # Expect scalar target per graph
        y_val = float(d.y.view(-1)[0].item()) if hasattr(d, "y") else np.nan
        y_rows.append(y_val)

X = np.vstack(X_rows)
y = np.asarray(y_rows, dtype=np.float32)

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fit model (GPU if available, fallback to CPU) ---
try:
    model1 = xgb_model.fit(X_train, y_train)
except xgb.core.XGBoostError as e:
    print("XGBoost CUDA not available; falling back to CPU.\n", e)
    xgb_params.update({
        "device": "cpu",
        "n_jobs": -1
    })
    xgb_model = XGBRegressor(**xgb_params)
    model1 = xgb_model.fit(X_train, y_train)

model1


metrics_df = evaluate_regression(
    model1,
    X_train, y_train,
    X_val, y_val,
    regression_metrics
)


import random
import numpy as np
import torch

# Prepare the training dataset for target_variable_2 = "Tc"
target_variable_2 = "Tc"
df_train_clean = df_train[["SMILES", target_variable_2]]
df_train_new = df_train_clean.dropna()

extra_cols = []
graphs = []
for _, row in df_train_new.iterrows():
    graph = smiles_to_graph(
        smiles=row["SMILES"],
        extra_features=row[extra_cols],   # ok even if extra_cols=[]
        use_mol_descriptors=True
    )
    if graph is not None:
        graph.y = torch.tensor([row[target_variable_2]], dtype=torch.float)
        graph.smiles = row["SMILES"]
        graphs.append(graph)

# ---- Loader alias: PyG if available, otherwise a safe fallback ---------------
try:
    from torch_geometric.loader import DataLoader  # real PyG loader
except Exception:
    from torch.utils.data import DataLoader as _TorchDataLoader
    from torch.utils.data import Dataset

    class _GraphDataset(Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]

    def _collate(batch):
        # With batch_size=1 return the single Data; otherwise return a list of Data objects
        return batch[0] if len(batch) == 1 else batch

    class DataLoader(_TorchDataLoader):  # alias name preserved
        def __init__(self, items, **kwargs):
            super().__init__(_GraphDataset(items), collate_fn=_collate, **kwargs)

# ---- Seeding and loader creation ---------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
g = torch.Generator().manual_seed(SEED)

loader = DataLoader(graphs, batch_size=1, shuffle=True, generator=g, num_workers=0)

# Optional sanity check
first_item = next(iter(loader))
print(getattr(first_item, "smiles", None), getattr(first_item, "y", None))



from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor

# --- Build feature and target arrays ---
X_rows, y_rows = [], []
feature_names = None

for batch in loader:
    # Handle single or multiple graphs per batch
    data_list = batch.to_data_list() if getattr(batch, "num_graphs", 1) > 1 else [batch]
    for d in data_list:
        x_vec, names = graph_to_features(d)
        if feature_names is None:
            feature_names = names
        X_rows.append(x_vec)

        # Expect scalar target per graph
        y_val = float(d.y.view(-1)[0].item()) if hasattr(d, "y") else np.nan
        y_rows.append(y_val)

X = np.vstack(X_rows)
y = np.asarray(y_rows, dtype=np.float32)

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fit model (GPU if available, fallback to CPU) ---
try:
    model2 = xgb_model.fit(X_train, y_train)
except xgb.core.XGBoostError as e:
    print("XGBoost CUDA not available; falling back to CPU.\n", e)
    xgb_params.update({
        "device": "cpu",
        "n_jobs": -1
    })
    xgb_model = XGBRegressor(**xgb_params)
    model2 = xgb_model.fit(X_train, y_train)

model2


metrics_df = evaluate_regression(
    model2,
    X_train, y_train,
    X_val, y_val,
    regression_metrics
)


import random
import numpy as np
import torch

# assumes `smiles_to_graph` from earlier is already defined

# ---------------- Prepare dataset for Tg ----------------
target_variable_3 = "Tg"
df_train_clean = df_train[["SMILES", target_variable_3]]
df_train_new = df_train_clean.dropna()

extra_cols = []  # if you add names later, they'll be used correctly

graphs = []
for _, row in df_train_new.iterrows():
    graph = smiles_to_graph(
        smiles=row["SMILES"],
        # IMPORTANT: avoid pandas empty-Series pitfall
        extra_features=(row[extra_cols].tolist() if extra_cols else None),
        use_mol_descriptors=True
    )
    if graph is not None:
        graph.y = torch.tensor([row[target_variable_3]], dtype=torch.float)
        graph.smiles = row["SMILES"]
        graphs.append(graph)

# --------------- Robust DataLoader alias ----------------
try:
    from torch_geometric.loader import DataLoader  # real PyG loader
except Exception:
    from torch.utils.data import DataLoader as _TorchDataLoader
    from torch.utils.data import Dataset

    class _GraphDataset(Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]

    def _collate(batch):
        # with batch_size=1 (your case) return the single Data; else a list of Data
        return batch[0] if len(batch) == 1 else batch

    class DataLoader(_TorchDataLoader):  # keep the same name for downstream code
        def __init__(self, items, **kwargs):
            super().__init__(_GraphDataset(items), collate_fn=_collate, **kwargs)

# --------------- Seeding & loader -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
g = torch.Generator().manual_seed(SEED)

loader = DataLoader(graphs, batch_size=1, shuffle=True, generator=g, num_workers=0)

# Optional sanity check
first_item = next(iter(loader))
print(getattr(first_item, "smiles", None), getattr(first_item, "y", None))



# Test and training split
# Build X, y from your (seeded) DataLoader
from sklearn.model_selection import train_test_split
X_rows, y_rows = [], []
feature_names = None

for batch in loader:
    # batch_size might be 1 or >1; handle both
    data_list = batch.to_data_list() if getattr(batch, "num_graphs", 1) > 1 else [batch]
    for d in data_list:
        x_vec, names = graph_to_features(d)
        if feature_names is None:
            feature_names = names
        X_rows.append(x_vec)
        # y: expect scalar target_variable per graph
        y_val = float(d.y.view(-1)[0].item()) if hasattr(d, "y") else np.nan
        y_rows.append(y_val)

X = np.vstack(X_rows)
y = np.asarray(y_rows, dtype=np.float32)

print("--------------------------------------------------------------")
print("Check dimensions, and first 5 feautres in the test, train splitted dataset")
# Optional: Check dimensions
print("X shape:", X.shape, "| y shape:", y.shape)
print("First 5 feature names:", feature_names[:5])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit (fallback to CPU if the wheel lacks CUDA support)
try:
    # Attempt GPU training
    model3 = xgb_model.fit(X_train, y_train)
except xgb.core.XGBoostError as e:
    print(" XGBoost CUDA not available; falling back to CPU.\n", e)
    xgb_params.update({
        "device": "cpu",  # ensure CPU mode
        "n_jobs": -1      # use all CPU cores
    })
    xgb_model = XGBRegressor(**xgb_params)
    model3 = xgb_model.fit(X_train, y_train)
model3


metrics_df = evaluate_regression(
    model3,
    X_train, y_train,
    X_val, y_val,
    regression_metrics
)


import random
import numpy as np
import torch

# assumes `smiles_to_graph` is already defined from earlier

# --------- Prepare dataset for Rg ----------
target_variable_4 = "Rg"
df_train_clean = df_train[["SMILES", target_variable_4]]
df_train_new = df_train_clean.dropna()

extra_cols = []  # add column names here if you have extra tabular features

graphs = []
for _, row in df_train_new.iterrows():
    graph = smiles_to_graph(
        smiles=row["SMILES"],
        extra_features=(row[extra_cols].tolist() if extra_cols else None),
        use_mol_descriptors=True
    )
    if graph is not None:
        graph.y = torch.tensor([row[target_variable_4]], dtype=torch.float)
        graph.smiles = row["SMILES"]
        graphs.append(graph)

# --------- Robust DataLoader alias ----------
try:
    from torch_geometric.loader import DataLoader  # real PyG loader
except Exception:
    from torch.utils.data import DataLoader as _TorchDataLoader
    from torch.utils.data import Dataset

    class _GraphDataset(Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]

    def _collate(batch):
        # With batch_size=1 return the single graph; else return a list of graphs
        return batch[0] if len(batch) == 1 else batch

    class DataLoader(_TorchDataLoader):  # keep same name for downstream code
        def __init__(self, items, **kwargs):
            super().__init__(_GraphDataset(items), collate_fn=_collate, **kwargs)

# --------- Seeding & loader ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
g = torch.Generator().manual_seed(SEED)

loader = DataLoader(graphs, batch_size=1, shuffle=True, generator=g, num_workers=0)

# Optional quick check
first_item = next(iter(loader))
print(getattr(first_item, "smiles", None), getattr(first_item, "y", None))



# Test and training split
# Build X, y from your (seeded) DataLoader
from sklearn.model_selection import train_test_split
X_rows, y_rows = [], []
feature_names = None

for batch in loader:
    # batch_size might be 1 or >1; handle both
    data_list = batch.to_data_list() if getattr(batch, "num_graphs", 1) > 1 else [batch]
    for d in data_list:
        x_vec, names = graph_to_features(d)
        if feature_names is None:
            feature_names = names
        X_rows.append(x_vec)
        # y: expect scalar target_variable per graph
        y_val = float(d.y.view(-1)[0].item()) if hasattr(d, "y") else np.nan
        y_rows.append(y_val)

X = np.vstack(X_rows)
y = np.asarray(y_rows, dtype=np.float32)

print("--------------------------------------------------------------")
print("Check dimensions, and first 5 feautres in the test, train splitted dataset")
# Optional: Check dimensions
print("X shape:", X.shape, "| y shape:", y.shape)
print("First 5 feature names:", feature_names[:5])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit (fallback to CPU if the wheel lacks CUDA support)
try:
    # Attempt GPU training
    model4 = xgb_model.fit(X_train, y_train)
except xgb.core.XGBoostError as e:
    print(" XGBoost CUDA not available; falling back to CPU.\n", e)
    xgb_params.update({
        "device": "cpu",  # ensure CPU mode
        "n_jobs": -1      # use all CPU cores
    })
    xgb_model = XGBRegressor(**xgb_params)
    model4 = xgb_model.fit(X_train, y_train)
model4


metrics_df = evaluate_regression(
    model4,
    X_train, y_train,
    X_val, y_val,
    regression_metrics
)


import random
import numpy as np
import torch

# --- Helpers ---------------------------------------------------------------
def build_graphs(df, smiles_col, target_col, extra_cols=None, use_mol_descriptors=True):
    """Return a list of Data graphs with .y and .smiles set."""
    extra_cols = extra_cols or []
    graphs = []
    for _, row in df.iterrows():
        graph = smiles_to_graph(
            smiles=row[smiles_col],
            extra_features=(row[extra_cols].tolist() if extra_cols else None),
            use_mol_descriptors=use_mol_descriptors
        )
        if graph is not None:
            graph.y = torch.tensor([row[target_col]], dtype=torch.float)
            graph.smiles = row[smiles_col]
            graphs.append(graph)
    return graphs

def get_loader(graphs, batch_size=1, shuffle=True, generator=None, num_workers=0):
    """PyG DataLoader if available; otherwise a safe fallback."""
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        return PyGDataLoader(graphs, batch_size=batch_size, shuffle=shuffle,
                             generator=generator, num_workers=num_workers)
    except Exception:
        from torch.utils.data import DataLoader as TorchDataLoader
        from torch.utils.data import Dataset

        class GraphDataset(Dataset):
            def __init__(self, items): self.items = items
            def __len__(self): return len(self.items)
            def __getitem__(self, idx): return self.items[idx]

        def _collate(batch):
            # With batch_size=1 (your case), return the single Data; else a list of Data
            return batch[0] if len(batch) == 1 else batch

        return TorchDataLoader(GraphDataset(graphs),
                               batch_size=batch_size, shuffle=shuffle,
                               generator=generator, num_workers=num_workers,
                               collate_fn=_collate)

# --- FFV dataset & loader ---------------------------------------------------
target_variable_5 = "FFV"
df_train_clean = df_train[["SMILES", target_variable_5]]
df_train_new = df_train_clean.dropna()

graphs = build_graphs(df_train_new, "SMILES", target_variable_5, extra_cols=[], use_mol_descriptors=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
g = torch.Generator().manual_seed(SEED)

loader = get_loader(graphs, batch_size=1, shuffle=True, generator=g, num_workers=0)

# Optional quick check
first_item = next(iter(loader))
print(getattr(first_item, "smiles", None), getattr(first_item, "y", None))



# Test and training split
# Build X, y from your (seeded) DataLoader
from sklearn.model_selection import train_test_split
X_rows, y_rows = [], []
feature_names = None

for batch in loader:
    # batch_size might be 1 or >1; handle both
    data_list = batch.to_data_list() if getattr(batch, "num_graphs", 1) > 1 else [batch]
    for d in data_list:
        x_vec, names = graph_to_features(d)
        if feature_names is None:
            feature_names = names
        X_rows.append(x_vec)
        # y: expect scalar target_variable per graph
        y_val = float(d.y.view(-1)[0].item()) if hasattr(d, "y") else np.nan
        y_rows.append(y_val)

X = np.vstack(X_rows)
y = np.asarray(y_rows, dtype=np.float32)

print("--------------------------------------------------------------")
print("Check dimensions, and first 5 feautres in the test, train splitted dataset")
# Optional: Check dimensions
print("X shape:", X.shape, "| y shape:", y.shape)
print("First 5 feature names:", feature_names[:5])

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit (fallback to CPU if the wheel lacks CUDA support)
try:
    # Attempt GPU training
    model5 = xgb_model.fit(X_train, y_train)
except xgb.core.XGBoostError as e:
    print(" XGBoost CUDA not available; falling back to CPU.\n", e)
    xgb_params.update({
        "device": "cpu",  # ensure CPU mode
        "n_jobs": -1      # use all CPU cores
    })
    xgb_model = XGBRegressor(**xgb_params)
    model5 = xgb_model.fit(X_train, y_train)

model5



metrics_df = evaluate_regression(
    model5,
    X_train, y_train,
    X_val, y_val,
    regression_metrics
)


#Check the test dataframe
df_test


# Convert a DataFrame of SMILES (+ optional extra columns) into a feature matrix X
def smiles_df_to_X(df, smiles_col="SMILES", extra_cols=None, use_mol_descriptors=True):
    """
    Returns
    -------
    X : np.ndarray            # (n_samples_kept, n_features)
    feature_names : list[str]
    kept_idx : list           # original df index values kept
    failed : list[tuple]      # [(idx, smiles), ...] that couldn't be parsed
    """
    extra_cols = extra_cols or []
    X_rows, kept_idx, failed = [], [], []
    feature_names = None

    for idx, row in df.iterrows():
        graph = smiles_to_graph(
            smiles=row[smiles_col],
            extra_features=row[extra_cols] if extra_cols else None,
            use_mol_descriptors=use_mol_descriptors,
        )
        if graph is None:
            failed.append((idx, row[smiles_col]))
            continue

        x_vec, names = graph_to_features(graph)
        if feature_names is None:
            feature_names = names
        X_rows.append(np.asarray(x_vec))
        kept_idx.append(idx)

    X = np.vstack(X_rows) if X_rows else np.empty((0, 0))
    return X, feature_names, kept_idx, failed


# ---- Usage on TEST data ----
extra_cols = []  # e.g., ["MolWeight", "LogP"] if you have them
df_src = df_test[["SMILES"] + extra_cols].dropna(subset=["SMILES"])

X, feature_names, kept_idx, failed = smiles_df_to_X(
    df_src,
    smiles_col="SMILES",
    extra_cols=extra_cols,
    use_mol_descriptors=True,
)

print(f"X shape: {X.shape}")
if failed:
    print(f"Skipped {len(failed)} rows that couldn't be parsed (first few): {failed[:5]}")


#All the models together
models = [model1, model2, model3, model4, model5]
heading = ["density", "Tc", "Tg", "Rg", "FFV"]


# Make sure names match number of models
assert len(models) == len(heading)

# Column-stack all predictions: shape -> (n_samples, n_targets)
Y = np.column_stack([np.asarray(m.predict(X)).ravel() for m in models])

# Build a nicely labelled DataFrame
pred_df = pd.DataFrame(Y, columns=heading, index=getattr(X, "index", None))
print(pred_df)          # or: pred_df.head()


#Prepaer the submission df
#Join the datframe
submission_df = [df_test["id"], pred_df]
submission_df = pd.concat(submission_df, axis=1)
submission_df


# simplest
print(submission_df)


# save to CSV
submission_df.to_csv('submission.csv', index=False)

