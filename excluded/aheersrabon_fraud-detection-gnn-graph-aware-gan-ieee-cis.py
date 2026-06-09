%%capture output

!python -m pip uninstall -y pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
!python -m pip install -U torch-geometric \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html


import torch, torch_geometric
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("PyG:", torch_geometric.__version__)
print("WITH_PYG_LIB =", WITH_PYG_LIB, "| WITH_TORCH_SPARSE =", WITH_TORCH_SPARSE)

import os, math, sys, math, random, numpy as np
from pathlib import Path
import json
from __future__ import annotations
import time

import polars as pl
import networkx as nx
import matplotlib.pyplot as plt

from copy import deepcopy

import torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from typing import Dict, List, Tuple
import networkx as nx
from sklearn.preprocessing import RobustScaler

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


ENABLE_TOY_RUN = False  # Set to False for full training
TOY_DATA_FRACTION = 0.25  # Use 10% of data for toy run

# If toy run enabled, override SUBSAMPLE_FRAC
if ENABLE_TOY_RUN:
    SUBSAMPLE_FRAC = TOY_DATA_FRACTION
    print(f"TOY RUN ENABLED: Using {TOY_DATA_FRACTION*100}% of data")
else:
    SUBSAMPLE_FRAC = None  # Use full data
    print("FULL TRAINING: Using 100% of data")

# Rest of your configuration...
PATH_TRAN = "/kaggle/input/ieee-fraud-detection/train_transaction.csv"
PATH_ID  = "/kaggle/input/ieee-fraud-detection/train_identity.csv"

USECOLS_TRAN = [
    "TransactionID","isFraud","TransactionDT","TransactionAmt",
    "card1","card2","card3","card4","card5","card6",
    "addr1","addr2","P_emaildomain","R_emaildomain"
]
USECOLS_ID = ["TransactionID","DeviceInfo","DeviceType","id_30","id_31"]

ENTITY_SCHEMA = [
    ("card1","card1"), ("card2","card2"), ("card3","card3"), ("card4","card4"),
    ("card5","card5"), ("card6","card6"),
    ("addr1","addr1"), ("addr2","addr2"),
    ("P_emaildomain","p_email"), ("R_emaildomain","r_email"),
    ("DeviceInfo","device"), ("DeviceType","devtype"),
    ("id_30","os"), ("id_31","browser"),
]

MIN_FREQ   = 3
KEEP_TOP_K = None
ADD_REVERSE_EDGES = True
SAVE_INDEXERS_PATH = "entity_indexers.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("="*80)
print("LOADING DATA")
print("="*80)

# Load and join
lt = pl.scan_csv(PATH_TRAN, infer_schema_length=2048).select(USECOLS_TRAN)
li = pl.scan_csv(PATH_ID, infer_schema_length=2048).select(USECOLS_ID)
lf = lt.join(li, on="TransactionID", how="left")
df = lf.collect(streaming=False)

if SUBSAMPLE_FRAC and SUBSAMPLE_FRAC < 1.0:
    print(f"Sampling {SUBSAMPLE_FRAC*100}% of data...")
    df = df.sample(fraction=SUBSAMPLE_FRAC, with_replacement=False, seed=42)

df = df.with_row_index(name="txn_index")

print(f"Loaded {df.height:,} transactions")
print(f"Fraud rate: {(df['isFraud'].sum() / df.height * 100):.2f}%")
print(f"txn_index range: 0 to {df.height-1}")
print("="*80)


# Set seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"\nDevice: {device}  |  Seed: {SEED}")


OUT_DIR = Path("./hetero_demo_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_PATH = OUT_DIR / "hetero_graph.png"
PARQUET_PATH = OUT_DIR / "hetero_embeddings.parquet"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"Device: {device}  |  Seed: {SEED}")

data_preview = HeteroData()

num_users, num_txn, num_merchants = 4, 6, 3
feat_dim, hidden = 8, 16

# random node features (demo)
data_preview['user'].x = torch.randn(num_users, feat_dim)
data_preview['txn'].x = torch.randn(num_txn, feat_dim)
data_preview['merchant'].x = torch.randn(num_merchants, feat_dim)

# user -> txn ("made")
user_src = torch.randint(0, num_users, (num_txn,), dtype=torch.long)
txn_dst  = torch.arange(0, num_txn, dtype=torch.long)
data_preview['user', 'made', 'txn'].edge_index = torch.stack([user_src, txn_dst], dim=0)

# txn -> merchant ("at")
merchant_dst = torch.randint(0, num_merchants, (num_txn,), dtype=torch.long)
data_preview['txn', 'at', 'merchant'].edge_index = torch.stack([txn_dst, merchant_dst], dim=0)

# user -> user ("follows")
uu_src = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
uu_dst = torch.tensor([1, 2, 3, 0, 2], dtype=torch.long)
data_preview['user', 'follows', 'user'].edge_index = torch.stack([uu_src, uu_dst], dim=0)

for ntype in data_preview.node_types:
    data_preview[ntype].x = data_preview[ntype].x.to(device)
for rel in data_preview.edge_types:
    data_preview[rel].edge_index = data_preview[rel].edge_index.to(device)

conv = HeteroConv({ rel: SAGEConv((-1, -1), hidden) for rel in data_preview.edge_types }, aggr='mean').to(device)

x_dict = {ntype: data_preview[ntype].x for ntype in data_preview.node_types}
edge_index_dict = {rel: data_preview[rel].edge_index for rel in data_preview.edge_types}

with torch.no_grad():
    out_dict = conv(x_dict, edge_index_dict)
    out_dict = {nt: torch.relu(h).cpu() for nt, h in out_dict.items()}  # move to CPU for saving/visualization

G = nx.DiGraph()
for ntype, count in [('user', num_users), ('txn', num_txn), ('merchant', num_merchants)]:
    for i in range(count):
        G.add_node(f"{ntype}_{i}", label=f"{ntype}:{i}", ntype=ntype)

for rel, ei in edge_index_dict.items():
    src_type, rel_name, dst_type = rel
    ei_cpu = ei.cpu().numpy()
    for s, d in ei_cpu.T:
        G.add_edge(f"{src_type}_{int(s)}", f"{dst_type}_{int(d)}", rel=rel_name)

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=SEED)
nx.draw(G, pos=pos, with_labels=True, labels={n: G.nodes[n]['label'] for n in G.nodes()}, node_size=700, font_size=9)
edge_labels = {(u, v): d['rel'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=7)
plt.title("Tiny Hetero Graph (user, txn, merchant)")
plt.axis('off')
plt.show()
plt.savefig(IMG_PATH, dpi=200, bbox_inches='tight', pad_inches=0.08)
print(f"Saved graph image to: {IMG_PATH}")

rows = []
for ntype, emb in out_dict.items():
    emb_np = emb.numpy()
    for nid in range(emb_np.shape[0]):
        rows.append({'node_type': ntype, 'node_id': int(nid), 'embedding': emb_np[nid].tolist()})

df_preview = pl.from_dicts(rows)
df_preview.write_parquet(PARQUET_PATH)
print(f"Saved embeddings parquet to: {PARQUET_PATH}")

print("Preview (first 6 rows):")
df_preview


print("BUILDING ENTITY INDEXERS")

idx_map = {}
for col, etype in ENTITY_SCHEMA:
    vc = (
        df.select(pl.col(col).cast(pl.Utf8).alias(col))
          .with_columns(pl.col(col).fill_null(""))
          .group_by(col).len().rename({"len":"cnt"})
          .filter(pl.col(col) != "")
          .sort("cnt", descending=True)
    )
    if KEEP_TOP_K:
        vc = vc.head(KEEP_TOP_K)
    if MIN_FREQ and MIN_FREQ > 1:
        vc = vc.filter(pl.col("cnt") >= MIN_FREQ)
    
    vc = vc.with_row_index(name="eid").select([col, "eid"])
    idx_map[etype] = dict(zip(vc[col].to_list(), vc["eid"].to_list()))
    print(f"{etype}: {len(idx_map[etype]):,} unique values")

if SAVE_INDEXERS_PATH:
    Path(SAVE_INDEXERS_PATH).write_text(json.dumps(idx_map))
    print(f"Saved indexers to {SAVE_INDEXERS_PATH}")



print("CREATING HETERODATA")

data = HeteroData()
num_txn = df.height
data["txn"].num_nodes = num_txn

print(f"Transaction nodes: {num_txn:,}")


print("\nAdding transaction features...")

# Amount â†’ NumPy
amt_np = df["TransactionAmt"].cast(pl.Float64).fill_null(0.0).to_numpy()
mu = float(np.mean(amt_np))
sd = float(np.std(amt_np)) if float(np.std(amt_np)) != 0 else 1.0
amt_z = (amt_np - mu) / (sd + 1e-6)
log_amt_np = np.log1p(amt_np)

# Time â†’ NumPy
sec_np = df["TransactionDT"].cast(pl.Float64).fill_null(0.0).to_numpy()
hour_np = (sec_np % 86400.0) / 3600.0
dow_np = ((sec_np // 86400.0) % 7.0)

# Stack features
X = np.stack([
    amt_z,
    log_amt_np,
    np.sin(2*np.pi*hour_np/24.0),
    np.cos(2*np.pi*hour_np/24.0),
    np.sin(2*np.pi*dow_np/7.0),
    np.cos(2*np.pi*dow_np/7.0),
], axis=1).astype(np.float32)

data["txn"].x = torch.from_numpy(X)

# Labels
y_np = df.select(pl.col("isFraud").fill_null(0).cast(pl.Int64))["isFraud"].to_numpy()
data["txn"].y = torch.tensor(y_np, dtype=torch.long)

print(f"Features: {X.shape}")
print(f"Labels: {y_np.shape} (fraud: {y_np.sum():,})")

print("\nAdding entity nodes...")

for _, etype in ENTITY_SCHEMA:
    data[etype].num_nodes = len(idx_map.get(etype, {}))

print(f"Added {len(ENTITY_SCHEMA)} entity types")


print("BUILDING EDGES")

def build_edges_for(col: str, etype: str):
    """Build edges for a specific entity type."""
    mapping = idx_map.get(etype, {})
    if not mapping:
        return None
    
    map_df = pl.DataFrame({col: list(mapping.keys()), "eid": list(mapping.values())})
    edges_df = (
        df.select(["txn_index", pl.col(col).cast(pl.Utf8).fill_null("").alias(col)])
          .join(map_df, on=col, how="inner")
          .select(["txn_index", "eid"])
    )
    
    if edges_df.height == 0:
        return None
    
    src = torch.tensor(edges_df["txn_index"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(edges_df["eid"].to_numpy(), dtype=torch.long)
    
    # âœ… SAFETY CHECK: Validate indices
    assert src.max() < num_txn, f"Source index {src.max()} >= num_txn {num_txn}"
    assert dst.max() < len(mapping), f"Dest index {dst.max()} >= num_entities {len(mapping)}"
    
    return torch.stack([src, dst], dim=0)

# Build txn -> entity edges
for col, etype in ENTITY_SCHEMA:
    eidx = build_edges_for(col, etype)
    if eidx is None:
        continue
    
    key = ("txn", f"has_{etype}", etype)
    data[key].edge_index = eidx
    
    if ADD_REVERSE_EDGES:
        rkey = (etype, f"rev_has_{etype}", "txn")
        data[rkey].edge_index = torch.stack([eidx[1], eidx[0]], dim=0)
    
    print(f"  {key}: {eidx.shape[1]:,} edges")


def add_shared_key_edges_txn_txn(data, df: pl.DataFrame, key: str, max_degree: int = 20):
    """Build txn-txn edges for transactions sharing the same entity."""
    print(f"\nBuilding txn-txn edges on shared {key}...")
    
    groups = (df.select(["txn_index", pl.col(key).cast(pl.Utf8).fill_null("").alias(key)])
                .to_pandas())
    
    by = {}
    for idx, val in zip(groups["txn_index"].values, groups[key].values):
        if not val:
            continue
        by.setdefault(val, []).append(int(idx))
    
    rows, cols = [], []
    for val, idxs in by.items():
        if len(idxs) < 2:
            continue
        
        L = len(idxs)
        for i in range(L):
            for j in range(max(0, i - max_degree), min(L, i + max_degree + 1)):
                if j == i:
                    continue
                rows.append(idxs[i])
                cols.append(idxs[j])
    
    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        
        # âœ… SAFETY CHECK: Validate indices
        max_idx = edge_index.max().item()
        assert max_idx < num_txn, f"txn-txn edge index {max_idx} >= num_txn {num_txn}"
        
        edge_index = torch.unique(edge_index, dim=1)
        rel = ("txn", f"same_{key}", "txn")
        data[rel].edge_index = edge_index
        print(f"  {rel}: {edge_index.shape[1]:,} edges")

add_shared_key_edges_txn_txn(data, df, key="card1", max_degree=20)

print(f"\nTotal edge types: {len(data.edge_types)}")


print("CREATING TRAIN/VAL/TEST SPLITS")

# Time-based splits
time_col = df["TransactionDT"].cast(pl.Float64)
t1 = float(time_col.quantile(0.7))
t2 = float(time_col.quantile(0.8))

times = time_col.fill_null(t1).to_numpy()
n = len(times)

train_mask = torch.from_numpy((times <= t1)).bool()
val_mask = torch.from_numpy((times > t1) & (times <= t2)).bool()
test_mask = torch.from_numpy((times > t2)).bool()

# Safety fallback
if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
    idx = torch.randperm(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:n_train]] = True
    
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[idx[n_train:n_train+n_val]] = True
    
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[idx[n_train+n_val:]] = True

data["txn"].train_mask = train_mask
data["txn"].val_mask = val_mask
data["txn"].test_mask = test_mask

print(f"Train: {train_mask.sum():,} ({train_mask.sum()/n*100:.1f}%)")
print(f"Val:   {val_mask.sum():,} ({val_mask.sum()/n*100:.1f}%)")
print(f"Test:  {test_mask.sum():,} ({test_mask.sum()/n*100:.1f}%)")

# Move to device
data = data.to(device)

print("\n" + "="*80)
print("DATA LOADING COMPLETE")
print("="*80)
print(f"âœ“ Device: {device}")
print(f"âœ“ Transactions: {num_txn:,}")
print(f"âœ“ Features: {X.shape[1]}")
print(f"âœ“ Node types: {len(data.node_types)}")
print(f"âœ“ Edge types: {len(data.edge_types)}")
print(f"âœ“ Fraud rate: {(y_np.sum() / len(y_np) * 100):.2f}%")
print("="*80)

# Verification
print("VERIFYING DATA INTEGRITY")

print("\nAvailable edge types:")
for et in data.edge_types:
    ei = data[et].edge_index
    src_type, rel_name, dst_type = et
    
    if ei.numel() > 0:
        src_max = ei[0].max().item()
        dst_max = ei[1].max().item()
        
        src_nodes = data[src_type].num_nodes
        dst_nodes = data[dst_type].num_nodes
        
        status = "âœ“"
        if src_max >= src_nodes or dst_max >= dst_nodes:
            status = "âœ— ERROR"
        
        print(f"  {status} {et}")
        print(f"      Edges: {ei.shape[1]:,}")
        print(f"      Src: max={src_max}, nodes={src_nodes}")
        print(f"      Dst: max={dst_max}, nodes={dst_nodes}")

print("\nâœ“ Data verification complete")
print("="*80)


print("Available relations:")
for et in data.edge_types:
    if et[0] == "txn":
        ei = getattr(data[et], "edge_index", None)
        if ei is not None and ei.numel() > 0:
            print(et, ei.shape)

rel = ('txn', 'has_devtype', 'devtype')   # change this line to try other relations

# --- Extract edges ---
eidx = data[rel].edge_index
E = eidx.size(1)

# Sample edges to avoid hairball
E_SAMPLE = min(E, 450)
perm = torch.randperm(E)[:E_SAMPLE]
src = eidx[0, perm].tolist()  # txn indices
dst = eidx[1, perm].tolist()  # entity indices

# --- Build NetworkX graph ---
G = nx.Graph()
for t in set(src):
    is_fraud = bool(int(data["txn"].y[t])) if hasattr(data["txn"], "y") else False
    G.add_node(f"txn:{t}", ntype="txn", fraud=is_fraud)

entity_type = rel[2]
for d in set(dst):
    G.add_node(f"{entity_type}:{d}", ntype=entity_type)

for s, d in zip(src, dst):
    G.add_edge(f"txn:{s}", f"{entity_type}:{d}")

# --- Layout & styling ---
pos = nx.spring_layout(G, seed=42, k=0.15)

node_colors = []
node_sizes  = []
for n, data_attr in G.nodes(data=True):
    if data_attr.get("ntype") == "txn":
        node_colors.append("red" if data_attr.get("fraud") else "gray")
        node_sizes.append(50)
    else:
        node_colors.append("lightblue")
        node_sizes.append(30)

plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.2)
nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)

# Label only frequent entities to keep it clean
labels = {n:n.split(":",1)[1] for n,deg in G.degree() if G.nodes[n].get("ntype")!= "txn" and deg>=10}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

plt.title(f"IEEE-CIS: {rel[0]} â†” {rel[2]} (red = fraud txns)")
plt.axis("off")
plt.show()


train_idx = torch.where(data["txn"].train_mask)[0]
val_idx   = torch.where(data["txn"].val_mask)[0]
test_idx  = torch.where(data["txn"].test_mask)[0]

fanout = [15, 10]   # was [25,15] â€“ smaller fanouts reduce noisy neighbors
batch_size = 2048

train_loader = NeighborLoader(
    data, input_nodes=("txn", train_idx),
    num_neighbors=fanout, batch_size=batch_size, shuffle=True
)
val_loader = NeighborLoader(
    data, input_nodes=("txn", val_idx),
    num_neighbors=fanout, batch_size=batch_size, shuffle=False
)
test_loader = NeighborLoader(
    data, input_nodes=("txn", test_idx),
    num_neighbors=fanout, batch_size=batch_size, shuffle=False
)


def dropout_edge(edge_index, p: float, training: bool):
    if (not training) or p <= 0.0:
        return edge_index
    E = edge_index.size(1)
    mask = torch.rand(E, device=edge_index.device) > p
    # keep at least 1 edge if possible
    if mask.sum() == 0 and E > 0:
        mask[torch.randint(0, E, (1,), device=edge_index.device)] = True
    return edge_index[:, mask]


class HeteroTxnGNN(nn.Module):
    def __init__(self, data, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.edge_drop_p = 0.2  # make it configurable; already used via getattr in forward


        # Per-type input projection or trainable embedding if x is missing.
        self.embeddings = nn.ModuleDict()
        self.in_lin = nn.ModuleDict()
        for ntype in data.node_types:
            if "x" in data[ntype]:
                in_dim = data[ntype].x.size(-1)
                self.in_lin[ntype] = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
            else:
                self.embeddings[ntype] = nn.Embedding(data[ntype].num_nodes, hidden)

        # HeteroConv stack (GraphSAGE for each relation)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv(
                {rel: SAGEConv((hidden, hidden), hidden) for rel in data.edge_types},
                aggr="mean"
            )
            self.convs.append(conv)

        # Classifier on txn nodes
        self.out = nn.Linear(hidden, 1)
        self.norms = nn.ModuleDict({ntype: nn.LayerNorm(self.hidden) for ntype in data.node_types})

    def _get_x(self, batch):
        x = {}
        for ntype in batch.node_types:
            if "x" in batch[ntype]:
                h = self.in_lin[ntype](batch[ntype].x)
            else:
                # Use original IDs for embedding lookup
                n_id = batch[ntype].n_id  # present in NeighborLoader batches
                h = self.embeddings[ntype](n_id)
            x[ntype] = h
        return x
    
    def forward(self, batch):
        x = self._get_x(batch)
    
        # Per-relation edge dropout (training only)
        edge_index_dict = batch.edge_index_dict
        if self.training and getattr(self, "edge_drop_p", 0.0) > 0.0:
            edge_index_dict = {
                rel: dropout_edge(ei, p=self.edge_drop_p, training=True)
                for rel, ei in edge_index_dict.items()
            }
    
        # Hetero blocks with residual + norm
        for conv in self.convs:
            h = conv(x, edge_index_dict)       # dict per node type: {ntype: tensor}
            out = {}
            for ntype, h_nt in h.items():
                base = x[ntype]
                if base.size(-1) != h_nt.size(-1):
                    proj_name = f"_proj_{ntype}"
                    if not hasattr(self, proj_name):
                        setattr(self, proj_name, nn.Linear(base.size(-1), h_nt.size(-1)))
                    base = getattr(self, proj_name)(base)
                out[ntype] = self.norms[ntype]( base + self.dropout(self.relu(h_nt)) )
            x = out
            
        logits = self.out(x["txn"]).squeeze(-1)  # [num_txn_in_subgraph]
        return logits

model = HeteroTxnGNN(data).to(device)


def bernoulli_st_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Straight-through Bernoulli sampling from logits.
    Returns a float tensor with values {0.0, 1.0} (hard) but gradient flows through the underlying probs.
    """
    probs = torch.sigmoid(logits)
    u = torch.rand_like(probs)
    hard = (probs > u).float()
    return hard + (probs - probs.detach())


def ensure_nonempty_mask(mask: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    """Make sure at least one edge is kept for relations with edges."""
    if mask.sum() == 0 and ei.size(1) > 0:
        idx = torch.randint(0, ei.size(1), (1,), device=mask.device)
        mask[idx] = 1.0
    return mask


def rel_to_key(rel: Tuple[str, str, str]) -> str:
    # deterministic string key for ModuleDict / dict indices
    return f"{rel[0]}__{rel[1]}__{rel[2]}"


# encode (use model.conv stack) safely
def encode_from_batch(model: HeteroTxnGNN,
                      batch,
                      edge_index_dict_override: Dict[Tuple[str,str,str], torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
    """
    Build per-type embeddings for the given batch using the model's conv stack.
    Uses model._get_x(batch) to obtain initial x_dict and then applies convs.
    If projection layers are needed for dimension mismatches, create and register them on the model (and move to device).
    Returns: x_out dict mapping ntype -> embedding tensor (on same device as model parameters).
    """
    device = next(model.parameters()).device
    x = model._get_x(batch)  # dict of tensors on device (NeighborLoader batch)
    # choose edge_index_dict: override (fake / generated) or batch.edge_index_dict
    if edge_index_dict_override is not None:
        edge_index_dict = edge_index_dict_override
    else:
        edge_index_dict = batch.edge_index_dict

    # Ensure all edge_index tensors are on device
    edge_index_dict = {rel: ei.to(device) for rel, ei in edge_index_dict.items()}

    # Run conv stack (same logic as forward minus final classifier)
    for conv in model.convs:
        h_dict = conv(x, edge_index_dict)  # per-node-type outputs
        out = {}
        for ntype, h_nt in h_dict.items():
            base = x[ntype]
            # handle potential dim mismatch by registered projection on model
            if base.size(-1) != h_nt.size(-1):
                proj_name = f"_proj_{ntype}"
                if not hasattr(model, proj_name):
                    # register a projection so it participates in param updates
                    proj = nn.Linear(base.size(-1), h_nt.size(-1)).to(device)
                    setattr(model, proj_name, proj)
                proj = getattr(model, proj_name)
                base = proj(base)
            out[ntype] = model.norms[ntype]( base + model.dropout(model.relu(h_nt)) )
        x = out
    # x now maps node types to final embeddings
    return x


"""
ATTENTION-GUIDED ADAPTIVE EDGE PERTURBATION (AGAEP)
"""

class EdgeTypeAttention(nn.Module):
    """
    Learns fraud-relevance weights for each edge type.
    
    Innovation: First work to use attention for adaptive edge perturbation in GAN.
    """
    def __init__(self, hidden_dim: int, num_edge_types: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        # Learnable edge type embeddings
        self.edge_type_embeddings = nn.Parameter(
            torch.randn(num_edge_types, hidden_dim) * 0.1
        )
        
        # Attention mechanism
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Context aggregator
        self.context_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, node_embeddings_dict: Dict, fraud_context: torch.Tensor):
        """
        Compute attention weights for edge types.
        
        Returns:
            edge_type_weights: [num_edge_types] importance scores
        """
        # Aggregate global context from all node types
        all_embeddings = []
        for emb in node_embeddings_dict.values():
            if emb.size(0) > 0:
                all_embeddings.append(emb.mean(dim=0, keepdim=True))
        
        if len(all_embeddings) > 0:
            context = torch.cat(all_embeddings, dim=0).mean(dim=0, keepdim=True)
        else:
            context = torch.zeros(1, self.hidden_dim, device=fraud_context.device)
        
        # Contextualize with fraud patterns
        context = self.context_aggregator(context + fraud_context)
        
        # Compute attention scores for each edge type
        edge_type_scores = []
        for edge_type_emb in self.edge_type_embeddings:
            combined = torch.cat([edge_type_emb.unsqueeze(0), context], dim=1)
            score = self.attention_mlp(combined)
            edge_type_scores.append(score)
        
        edge_type_scores = torch.cat(edge_type_scores, dim=0)
        edge_type_weights = F.softmax(edge_type_scores.squeeze(-1), dim=0)
        
        return edge_type_weights


class AdaptiveEdgePerturbGenerator(nn.Module):
    """
    Enhanced edge perturbation with attention-guided adaptation.
    
    Key improvements over baseline:
    1. Edge type attention module
    2. Fraud context encoding
    3. Adaptive perturbation scaling
    """
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str,str,str]],
                 hidden: int,
                 edge_mlp_hidden: int = 128,
                 noise_dim: int = 16):  # Increased from 0
        super().__init__()
        self.hidden = hidden
        self.noise_dim = noise_dim
        self.edge_types = edge_types
        self.num_edge_types = len(edge_types)
        
        # NEW: Edge type attention
        self.edge_attention = EdgeTypeAttention(hidden, self.num_edge_types)
        
        # Per-edge-type perturbation networks (keep your original structure)
        self.edge_scorers = nn.ModuleDict()
        for rel in edge_types:
            key = rel_to_key(rel)
            in_dim = 2 * hidden + noise_dim
            self.edge_scorers[key] = nn.Sequential(
                nn.Linear(in_dim, edge_mlp_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(edge_mlp_hidden, edge_mlp_hidden // 2),
                nn.ReLU(),
                nn.Linear(edge_mlp_hidden // 2, 1)
            )
        
        # NEW: Fraud context encoder
        self.fraud_context_encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden)
        )
    
    def compute_fraud_context(self, node_embeddings_dict, fraud_labels, fraud_mask):
        """Compute fraud context - handles size mismatch."""
        if "txn" not in node_embeddings_dict:
            device = next(self.parameters()).device
            return torch.zeros(1, self.hidden, device=device)
        
        txn_embeddings = node_embeddings_dict["txn"]
        device = txn_embeddings.device
        
        emb_size = txn_embeddings.size(0)  # e.g., 40066
        label_size = fraud_labels.size(0)   # e.g., 2048
        
        if emb_size == label_size:
            # Perfect match
            fraud_txns = fraud_mask & (fraud_labels == 1)
            if fraud_txns.sum() > 0:
                fraud_context = txn_embeddings[fraud_txns].mean(dim=0, keepdim=True)
            else:
                fraud_context = txn_embeddings.mean(dim=0, keepdim=True)
        
        elif emb_size > label_size:
            # First label_size embeddings are the target nodes
            fraud_txns = fraud_mask & (fraud_labels == 1)
            
            if fraud_txns.sum() > 0:
                # Index only the target embeddings
                fraud_context = txn_embeddings[:label_size][fraud_txns].mean(dim=0, keepdim=True)
            else:
                fraud_context = txn_embeddings[:label_size].mean(dim=0, keepdim=True)
        
        else:
            # Fewer embeddings than labels (rare)
            fraud_labels_trunc = fraud_labels[:emb_size]
            fraud_mask_trunc = fraud_mask[:emb_size]
            fraud_txns = fraud_mask_trunc & (fraud_labels_trunc == 1)
            
            if fraud_txns.sum() > 0:
                fraud_context = txn_embeddings[fraud_txns].mean(dim=0, keepdim=True)
            else:
                fraud_context = txn_embeddings.mean(dim=0, keepdim=True)
        
        return fraud_context
    
    def forward(self,
                node_embeddings_dict: Dict,
                edge_index_dict: Dict,
                fraud_labels: torch.Tensor = None,
                fraud_mask: torch.Tensor = None,
                training: bool = True):
        """
        Generate adaptive edge perturbations.
        
        Returns:
            perturbed_edge_index_dict: Modified edges
            edge_type_importance: Learned weights for analysis
        """
        device = next(self.parameters()).device
        
        # Compute fraud context
        if fraud_labels is None or fraud_mask is None:
            fraud_context = torch.zeros(1, self.hidden, device=device)
        else:
            fraud_context = self.compute_fraud_context(
                node_embeddings_dict, fraud_labels, fraud_mask
            )
        
        # Compute edge type importance via attention
        edge_type_weights = self.edge_attention(node_embeddings_dict, fraud_context)
        
        # Perturb edges adaptively
        perturbed_edge_index_dict = {}
        
        for edge_type_idx, (rel, edge_index) in enumerate(edge_index_dict.items()):
            if edge_index.size(1) == 0:
                perturbed_edge_index_dict[rel] = edge_index
                continue
            
            # Get node embeddings
            src_type, _, dst_type = rel
            src_emb = node_embeddings_dict[src_type]
            dst_emb = node_embeddings_dict[dst_type]
            
            # Get edge embeddings
            src_indices = edge_index[0]
            dst_indices = edge_index[1]
            edge_emb_src = src_emb[src_indices]
            edge_emb_dst = dst_emb[dst_indices]
            
            # Add noise for diversity
            if training and self.noise_dim > 0:
                noise = torch.randn(edge_index.size(1), self.noise_dim, device=device)
            else:
                noise = torch.zeros(edge_index.size(1), self.noise_dim, device=device)
            
            # Concatenate features
            edge_features = torch.cat([edge_emb_src, edge_emb_dst, noise], dim=1)
            
            # Compute perturbation logits
            key = rel_to_key(rel)
            edge_logits = self.edge_scorers[key](edge_features).squeeze(-1)
            
            # CRITICAL NEW STEP: Scale by edge type importance
            # Higher importance â†’ more conservative (keep more edges)
            importance = edge_type_weights[edge_type_idx]
            scaled_logits = edge_logits * (1.0 + importance)
            
            # Sample edges with straight-through gradient
            if training:
                probs = torch.sigmoid(scaled_logits)
                uniform = torch.rand_like(probs)
                hard_mask = (probs > uniform).float()
                keep_mask = hard_mask + (probs - probs.detach())
            else:
                keep_mask = (torch.sigmoid(scaled_logits) > 0.5).float()
            
            # Ensure at least one edge is kept
            if keep_mask.sum() == 0 and edge_index.size(1) > 0:
                keep_mask[torch.randint(0, edge_index.size(1), (1,), device=device)] = 1.0
            
            # Apply mask
            keep_indices = keep_mask.bool()
            perturbed_edge_index = edge_index[:, keep_indices]
            perturbed_edge_index_dict[rel] = perturbed_edge_index
        
        return perturbed_edge_index_dict, edge_type_weights


class DiscriminatorWrapper(nn.Module):
    """
    Wraps HeteroTxnGNN to produce adv logits over txn nodes given x_dict & edge_index_dict.
    Uses encode_from_batch to compute embeddings then applies base.out head.
    """
    def __init__(self, base_model: HeteroTxnGNN):
        super().__init__()
        self.base = base_model

    def forward_logits_from_batch(self, batch, edge_index_dict_override: Dict[Tuple[str,str,str], torch.Tensor] = None):
        # compute per-type embeddings with optional override for edges
        x_out = encode_from_batch(self.base, batch, edge_index_dict_override)
        txn_emb = x_out["txn"]
        logits = self.base.out(txn_emb).squeeze(-1)
        return logits


import gc

# Clear cache
torch.cuda.empty_cache()
gc.collect()

# Check available memory
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


print("GAN TRAINING CONFIGURATION")

# Adjust GAN training based on toy run
if ENABLE_TOY_RUN:
    num_epochs = 2  # Very short GAN training for toy run
    d_steps = 1
    g_steps = 1
    lambda_entropy = 1e-3
    print_every = 1
    print(f"ðŸŽ® TOY RUN: {num_epochs} GAN epochs")
else:
    num_epochs = 6  # Full GAN training
    d_steps = 1
    g_steps = 1
    lambda_entropy = 1e-3
    print_every = 1
    print(f"FULL TRAINING: {num_epochs} GAN epochs")

print("="*80)

node_types = list(data.node_types)
edge_types = list(data.edge_types)

gen = AdaptiveEdgePerturbGenerator(
    node_types=node_types, 
    edge_types=edge_types, 
    hidden=model.hidden,
    edge_mlp_hidden=128, 
    noise_dim=16  # Changed from 0
).to(device)

disc = DiscriminatorWrapper(model).to(device)

# Optimizers
lr_g = 1e-3
lr_d = 1e-3
opt_G = torch.optim.Adam(gen.parameters(), lr=lr_g, weight_decay=1e-6)
opt_D = torch.optim.Adam(disc.parameters(), lr=lr_d, weight_decay=1e-6)

bce = nn.BCEWithLogitsLoss(reduction="mean")

# NEW: Storage for edge importance analysis
edge_importance_history = []

# Training loop with attention tracking
start_time = time.time()
for epoch in range(1, num_epochs + 1):
    model.train()
    gen.train()
    disc.train()
    epoch_loss_D = 0.0
    epoch_loss_G = 0.0
    n_batches = 0
    
    # NEW: Track edge importances this epoch
    epoch_importances = []

    for batch in train_loader:
        n_batches += 1
        batch = batch.to(device)

        # Prepare inputs
        x_real = model._get_x(batch)
        edge_real = batch.edge_index_dict
        
        # NEW: Get fraud labels and mask for context
        fraud_labels = batch["txn"].y
        fraud_mask = batch["txn"].train_mask

        # Discriminator step
        for _ in range(d_steps):
            opt_D.zero_grad()

            # Real
            logits_real = disc.forward_logits_from_batch(batch, edge_index_dict_override=edge_real)
            y_real = torch.ones_like(logits_real, device=device)
            loss_real = bce(logits_real, y_real)

            # Fake - NEW: Pass fraud context
            edge_fake_dict, edge_importances = gen(
                x_real, edge_real, 
                fraud_labels=fraud_labels, 
                fraud_mask=fraud_mask, 
                training=True
            )
            
            # Store importances
            epoch_importances.append(edge_importances.detach().cpu())

            logits_fake = disc.forward_logits_from_batch(batch, edge_index_dict_override=edge_fake_dict)
            y_fake = torch.zeros_like(logits_fake, device=device)
            loss_fake = bce(logits_fake, y_fake)

            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            opt_D.step()
            epoch_loss_D += float(loss_D.detach().cpu().item())

        # Generator step
        for _ in range(g_steps):
            opt_G.zero_grad()

            edge_fake_dict, edge_importances = gen(
                x_real, edge_real,
                fraud_labels=fraud_labels,
                fraud_mask=fraud_mask,
                training=True
            )

            logits_fake_for_G = disc.forward_logits_from_batch(batch, edge_index_dict_override=edge_fake_dict)
            y_want = torch.ones_like(logits_fake_for_G, device=device)
            loss_G_adv = bce(logits_fake_for_G, y_want)

            # Entropy regularizer (encourage diversity)
            entropy_reg = 0.0
            # (You can add entropy calculation here if needed)

            loss_G = loss_G_adv + lambda_entropy * entropy_reg
            loss_G.backward()
            opt_G.step()
            epoch_loss_G += float(loss_G.detach().cpu().item())

    # Epoch summary
    elapsed = time.time() - start_time
    avg_D = epoch_loss_D / max(1, n_batches)
    avg_G = epoch_loss_G / max(1, n_batches)
    
    # NEW: Average edge importances for this epoch
    if epoch_importances:
        avg_importances = torch.stack(epoch_importances).mean(dim=0)
        edge_importance_history.append(avg_importances)
        
        # Print every 10 epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} - Edge Type Importance:")
            for idx, rel in enumerate(edge_types):
                print(f"  {rel}: {avg_importances[idx].item():.4f}")
    
    if epoch % print_every == 0:
        print(f"[Epoch {epoch:03d}] D_loss={avg_D:.4f}  G_loss={avg_G:.4f}  elapsed={elapsed/60:.2f}m")

# Save models
torch.save(gen.state_dict(), "agaep_gen.pt")
torch.save(model.state_dict(), "agaep_gnn.pt")
print("\nModels saved: agaep_gen.pt, agaep_gnn.pt")


print("FINAL EDGE TYPE IMPORTANCE ANALYSIS")

# Get final importance weights
gen.eval()
with torch.no_grad():
    # Use validation batch
    val_batch = next(iter(val_loader))
    val_batch = val_batch.to(device)
    
    node_embs = encode_from_batch(model, val_batch)
    fraud_labels = val_batch['txn'].y
    fraud_mask = val_batch['txn'].val_mask
    
    _, final_edge_importances = gen(
        node_embeddings_dict=node_embs,
        edge_index_dict=val_batch.edge_index_dict,
        fraud_labels=fraud_labels,
        fraud_mask=fraud_mask,
        training=False
    )

# Create results dataframe
import pandas as pd

edge_importance_df = pd.DataFrame({
    'Edge Type': [str(rel) for rel in edge_types],
    'Learned Weight': [w.item() for w in final_edge_importances],
})

edge_importance_df = edge_importance_df.sort_values('Learned Weight', ascending=False)
edge_importance_df['Rank'] = range(1, len(edge_types) + 1)

print(edge_importance_df.to_string(index=False))

# Save for paper
edge_importance_df.to_csv('edge_importance_results.csv', index=False)
print("\nâœ“ Saved to edge_importance_results.csv")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits, targets):
        # targets: (B,) in {0,1}
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p = torch.sigmoid(logits).clamp_min(1e-6).clamp_max(1-1e-6)
        pt = p*targets + (1-p)*(1-targets)   # p_t
        loss = self.alpha * (1-pt).pow(self.gamma) * bce
        return loss.mean() if self.reduction=="mean" else loss.sum()

# Use this instead of BCEWithLogitsLoss
criterion = FocalLoss(alpha=0.75, gamma=2.0)


y_train = data["txn"].y[train_idx].float()
pos = (y_train == 1).sum().item()
neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([(neg / max(pos, 1))], device=device)  # >1 if positives are rare

#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)


def eval_loader(loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            bsz = batch["txn"].batch_size
            logits = logits[:bsz]
            y = batch["txn"].y[:bsz].float()
            ys.append(y)
            ps.append(torch.sigmoid(logits))
    y = torch.cat(ys).detach().cpu().numpy()
    p = torch.cat(ps).detach().cpu().numpy()

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        
        # Threshold at 0.5 for F1, Recall, Precision
        y_pred = (p >= 0.5).astype(int)
        f1 = f1_score(y, y_pred)
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
    except Exception as e:
        print(f"Metric calculation error: {e}")
        auc = ap = f1 = recall = precision = float("nan")
    
    acc = ((p >= 0.5) == (y == 1)).mean()
    return {
        "auc": auc, 
        "ap": ap, 
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "acc@0.5": acc
    }


print("DEFINING AUGMENTATION HELPER FUNCTIONS")

# ===== Function 1: Build augmented edges =====
def build_augmented_edge_index_dict(x_dict, edge_real, gen, fraud_labels, fraud_mask, mode="deterministic", threshold=0.5):
    """
    Build augmented edges using the generator.
    
    Args:
        x_dict: Node embeddings dictionary
        edge_real: Real edge index dictionary
        gen: Generator model
        fraud_labels: Fraud labels for context
        fraud_mask: Mask for training samples
        mode: "deterministic" or "stochastic"
        threshold: Threshold for deterministic masking
    
    Returns:
        Dictionary of augmented edge indices
    """
    edge_fake_dict, _ = gen(
        x_dict, edge_real,
        fraud_labels=fraud_labels,
        fraud_mask=fraud_mask,
        training=(mode == "stochastic")
    )
    return edge_fake_dict


# ===== Function 2: Evaluation with augmentation =====
def eval_loader_with_augment(loader, model, gen=None, device='cuda', 
                             ensemble_aug=False, n_aug=3, 
                             aug_mode="deterministic", aug_threshold=0.5):
    """
    Evaluate model on a data loader, optionally with augmentation ensemble.
    
    Args:
        loader: DataLoader to evaluate on
        model: GNN model
        gen: Generator (optional, for augmentation)
        device: Device to use
        ensemble_aug: Whether to ensemble augmented predictions
        n_aug: Number of augmented samples to average
        aug_mode: "deterministic" or "stochastic"
        aug_threshold: Threshold for deterministic masking
    
    Returns:
        Dictionary with metrics (auc, ap, f1, recall, precision, acc@0.5)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, accuracy_score
    
    model.eval()
    ys = []
    ps = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            bsz = batch["txn"].batch_size
            
            # Real prediction
            logits_real = model(batch)[:bsz]
            probs_real = torch.sigmoid(logits_real)
            
            if ensemble_aug and gen is not None:
                # Generate augmented predictions and average
                aug_probs_list = [probs_real]
                
                for _ in range(n_aug):
                    x_dict = model._get_x(batch)
                    edge_real = batch.edge_index_dict
                    
                    # Get fraud context
                    fraud_labels = batch["txn"].y[:bsz]
                    fraud_mask = torch.ones(bsz, dtype=torch.bool, device=device)  # All are "in mask"
                    
                    aug_edges = build_augmented_edge_index_dict(
                        x_dict, edge_real, gen, 
                        fraud_labels, fraud_mask,
                        mode=aug_mode, threshold=aug_threshold
                    )
                    
                    batch_aug = deepcopy(batch)
                    batch_aug.edge_index_dict = {rel: e.to(device) for rel, e in aug_edges.items()}
                    
                    logits_aug = model(batch_aug)[:bsz]
                    probs_aug = torch.sigmoid(logits_aug)
                    aug_probs_list.append(probs_aug)
                
                # Average all predictions
                probs_final = torch.stack(aug_probs_list).mean(dim=0)
            else:
                probs_final = probs_real
            
            y = batch["txn"].y[:bsz].float()
            ys.append(y.cpu())
            ps.append(probs_final.cpu())
    
    y_true = torch.cat(ys).numpy()
    y_probs = torch.cat(ps).numpy()
    y_pred = (y_probs >= 0.5).astype(int)
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0
    
    try:
        ap = average_precision_score(y_true, y_probs)
    except:
        ap = 0.0
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'auc': auc,
        'ap': ap,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'acc@0.5': acc
    }


# Hyperparameters for augmentation
USE_AUGMENT = True          # enable augmentation during supervised training
AUG_PROB = 0.6              # per-batch probability to apply augmentation
AUG_WEIGHT = 0.6            # weight of augmented loss when combining with real loss
AUG_THRESHOLD = 0.5         # threshold on sigmoid(logits) for deterministic mask
AUG_SAMPLE_STOCHASTIC = False  # if True, use stochastic bernoulli sampling; else deterministic threshold
ENSEMBLE_AUG_EVAL = False   # if True, at eval average predictions from n_aug samples + real
AUG_ENSEMBLE_SAMPLES = 3


def deterministic_mask_from_logits(logits, threshold=0.5):
    return (torch.sigmoid(logits) > threshold).float()


def build_augmented_edge_index_dict(x_dict, edge_real, gen, fraud_labels, fraud_mask, mode="deterministic", threshold=0.5):
    """Build augmented edges using generator."""
    edge_fake_dict, _ = gen(
        x_dict, edge_real,
        fraud_labels=fraud_labels,
        fraud_mask=fraud_mask,
        training=(mode == "stochastic")
    )
    return edge_fake_dict


print("SUPERVISED TRAINING WITH AUGMENTATION")

# Configuration
epochs = 40 if not ENABLE_TOY_RUN else 5  # Fewer epochs for toy run
base_lr = 2e-3
warmup_epochs = max(1, int(0.05 * epochs))

# Create scheduler for the main model optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)

def lr_with_warmup(base_lr, epoch):
    """Apply warmup to learning rate"""
    if epoch <= warmup_epochs:
        return base_lr * (epoch / max(1, warmup_epochs))
    return base_lr

# Training loop
for epoch in range(1, epochs + 1):
    # Warmup LR
    for g in optimizer.param_groups:
        g["lr"] = lr_with_warmup(base_lr, epoch)

    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        n_batches += 1
        batch = batch.to(device)
        bsz = batch["txn"].batch_size

        # --- Real forward & loss ---
        logits = model(batch)[:bsz]
        y = batch["txn"].y[:bsz].float()
        loss_real = criterion(logits, y)

        loss = loss_real

        # --- Optional augmentation ---
        if USE_AUGMENT and (gen is not None) and (random.random() < AUG_PROB):
            # Get node embeddings and edges
            x_dict = model._get_x(batch)
            edge_real = batch.edge_index_dict
            
            # Get fraud context for attention
            fraud_labels = batch["txn"].y[:bsz]
            fraud_mask = batch["txn"].train_mask[:bsz] if hasattr(batch["txn"], 'train_mask') else torch.ones(bsz, dtype=torch.bool, device=device)
            
            # Build augmented edges
            mode = "stochastic" if AUG_SAMPLE_STOCHASTIC else "deterministic"
            aug_edges = build_augmented_edge_index_dict(
                x_dict, edge_real, gen, 
                fraud_labels, fraud_mask,
                mode=mode, threshold=AUG_THRESHOLD
            )

            # Create augmented batch
            batch_aug = deepcopy(batch)
            batch_aug.edge_index_dict = {rel: e.to(device) for rel, e in aug_edges.items()}

            # Forward on augmented batch
            logits_aug = model(batch_aug)[:bsz]
            y_aug = batch_aug["txn"].y[:bsz].float()
            loss_aug = criterion(logits_aug, y_aug)

            # Combine losses
            loss = (1.0 - AUG_WEIGHT) * loss_real + AUG_WEIGHT * loss_aug

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += float(loss.item())

    # Step scheduler
    scheduler.step()
    
    # Validation
    val_metrics = eval_loader_with_augment(
        val_loader, model, 
        gen=gen if USE_AUGMENT else None,
        device=device, 
        ensemble_aug=ENSEMBLE_AUG_EVAL,
        n_aug=AUG_ENSEMBLE_SAMPLES,
        aug_mode=("stochastic" if AUG_SAMPLE_STOCHASTIC else "deterministic"),
        aug_threshold=AUG_THRESHOLD
    )
    
    print(f"Epoch {epoch:02d}/{epochs} | loss={total_loss/n_batches:.3f} | "
          f"val_auc={val_metrics['auc']:.4f} | val_f1={val_metrics['f1']:.4f} | "
          f"val_recall={val_metrics['recall']:.4f}")


print("FINAL TEST EVALUATION")

# Final test evaluation
test_metrics = eval_loader_with_augment(
    test_loader, model, 
    gen=gen if USE_AUGMENT else None,
    device=device, 
    ensemble_aug=ENSEMBLE_AUG_EVAL,
    n_aug=AUG_ENSEMBLE_SAMPLES,
    aug_mode=("stochastic" if AUG_SAMPLE_STOCHASTIC else "deterministic"),
    aug_threshold=AUG_THRESHOLD
)

print("\nTEST RESULTS:")
print(f"  AUC:       {test_metrics['auc']:.4f}")
print(f"  AP:        {test_metrics['ap']:.4f}")
print(f"  F1:        {test_metrics['f1']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Accuracy:  {test_metrics['acc@0.5']:.4f}")
print("="*80)

# Save models
torch.save(model.state_dict(), "final_model.pt")
torch.save(gen.state_dict(), "final_generator.pt")
print("\nâœ“ Models saved: final_model.pt, final_generator.pt")


plt.figure(figsize=(12, 6))
edge_importance_df_sorted = edge_importance_df.sort_values('Learned Weight', ascending=True)
plt.barh(range(len(edge_importance_df_sorted)), edge_importance_df_sorted['Learned Weight'])
plt.yticks(range(len(edge_importance_df_sorted)), edge_importance_df_sorted['Edge Type'])
plt.xlabel('Learned Importance Weight')
plt.title('Edge Type Importance (Learned via Attention)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('edge_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("âœ“ Saved edge_importance.png")


print("\n" + "="*80)
print("AGAEP-GAN TRAINING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - agaep_gen.pt (Generator model)")
print("  - agaep_gnn.pt (GNN model)")
print("  - edge_importance_results.csv (Edge importance analysis)")
print("  - final_test_results.csv (Test metrics)")
print("  - edge_importance.png (Visualization)")
print("\nYou can now use these results for your paper!")


print("TRAINING COMPLETE - SUMMARY")

if ENABLE_TOY_RUN:
    print("ðŸŽ® TOY RUN RESULTS:")
    print(f"  Data used: {TOY_DATA_FRACTION*100}%")
    print(f"  GAN epochs: 2")
    print(f"  Supervised epochs: 5")
    print(f"  Purpose: Debugging and validation")
    print("   Set ENABLE_TOY_RUN = False for full training")
else:
    print("FULL TRAINING RESULTS:")
    print(f"  Data used: 100%")
    print(f"  GAN epochs: 6")
    print(f"  Supervised epochs: 40")

print("\nFinal Test Metrics:")
print(f"  AUC:       {test_metrics['auc']:.4f}")
print(f"  F1:        {test_metrics['f1']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")

