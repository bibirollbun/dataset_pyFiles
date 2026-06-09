import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)



seq_df = pd.read_csv('/kaggle/input/feature-extraction-global/seq_df_global.csv')



import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

csv_path = '/kaggle/input/feature-extraction-global/seq_df_global.csv'
df = pd.read_csv(csv_path)

one_d_cols = [
    "onehot","pair_flag","loop_type_onehot",
    "dG_loop_res","sa","flexibility_index"
]
two_d_cols = [
    "pp","mi_matrix","dca","covariance",
    "bp_potentials","gt_distances","contact_map","angle_matrix","motif_match_matrix"
]

def parse_array_string(x):
    if not isinstance(x, str):
        return np.array(x, dtype=float)
    rows = re.findall(r'\[([^\[\]]*\d+[^\[\]]*)\]', x)
    mat = []
    for r in rows:
        nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', r)
        mat.append([float(n) for n in nums])
    if not mat:
        return np.empty((0,0), dtype=float)
    ml = max(len(r) for r in mat)
    for r in mat:
        r.extend([0.0] * (ml - len(r)))
    return np.array(mat, dtype=float)

for c in one_d_cols + two_d_cols:
    df[c] = df[c].apply(parse_array_string)

global_numeric = ["freq_A","freq_C","freq_G","freq_U",
                  "total_dG","msa_depth","msa_diversity",
                  "gc_content","n_stems","n_loops","partner_chains_count"]
global_binary  = ["has_metal","has_ligand"]
global_cat     = ["ligand_metal_category","experimental_method"]
df_cat    = pd.get_dummies(df[global_cat].astype(str), prefix=global_cat)
df_global = pd.concat([df[global_numeric+global_binary], df_cat], axis=1)
global_cols = df_global.columns.tolist()

class RNADataset(Dataset):
    def __init__(self, df, one_d_cols, two_d_cols, global_cols):
        self.df          = df.reset_index(drop=True)
        self.one_d_cols  = one_d_cols
        self.two_d_cols  = two_d_cols
        self.global_cols = global_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        onehot = np.asarray(row[self.one_d_cols[0]], dtype=float)
        L = onehot.shape[0]

        parts = []
        for c in self.one_d_cols:
            arr = np.asarray(row[c], dtype=float)
            arr = np.atleast_2d(arr)
            if arr.shape[0] == 1 and arr.shape[1] == L:
                arr = arr.T
            h, k = arr.shape
            if h != L:
                new = np.zeros((L, k), dtype=float)
                mh = min(h, L)
                new[:mh, :] = arr[:mh, :]
                arr = new
            parts.append(torch.from_numpy(arr).float())
        X1 = torch.cat(parts, dim=1)

        mats = []
        for c in self.two_d_cols:
            mat = np.asarray(row[c], dtype=float)
            mat = np.atleast_2d(mat)
            h, w = mat.shape
            if (h, w) != (L, L):
                new = np.zeros((L, L), dtype=float)
                mh = min(h, L)
                mw = min(w, L)
                new[:mh, :mw] = mat[:mh, :mw]
                mat = new
            mats.append(torch.from_numpy(mat).float().unsqueeze(-1))
        X2 = torch.cat(mats, dim=-1)
        
        G = torch.from_numpy(df_global.iloc[idx].to_numpy(dtype=float))

        return X1, X2, G, L

def collate_fn(batch):
    lengths = [b[3] for b in batch]
    Lmax    = max(lengths)
    B       = len(batch)
    F1      = batch[0][0].shape[1]
    F2      = batch[0][1].shape[2]
    Gdim    = batch[0][2].shape[0]

    X1b = torch.zeros((B, Lmax, F1), dtype=torch.float32)
    X2b = torch.zeros((B, Lmax, Lmax, F2), dtype=torch.float32)
    Gb  = torch.zeros((B, Gdim),     dtype=torch.float32)

    for i, (X1, X2, G, L) in enumerate(batch):
        X1b[i, :L, :]     = X1.half()
        X2b[i, :L, :L, :] = X2.half()
        Gb[i]             = G.half()

    return X1b, X2b, Gb, torch.tensor(lengths)

dataset = RNADataset(df, one_d_cols, two_d_cols, global_cols)
loader  = DataLoader(
    dataset,
    batch_size=8,          
    shuffle=False,
    num_workers=0,         
    collate_fn=collate_fn,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for X1b, X2b, Gb, lengths in loader:
    X1b, X2b, Gb = X1b.to(device), X2b.to(device), Gb.to(device)
    print("X1:", X1b.shape, X1b.dtype, X1b.device)
    print("X2:", X2b.shape, X2b.dtype, X2b.device)
    print("G: ",  Gb.shape,  Gb.dtype,  Gb.device)
    print("lengths:", lengths)
    break



import numpy as np

batch_idx = 0
seq_idx   = df.index[batch_idx]
L = lengths[batch_idx].item()

print(f"Examining sequence {df.loc[seq_idx,'target_id']} (length {L})\n")

for chan, name in enumerate(two_d_cols):
    raw = np.asarray(df.loc[seq_idx, name], dtype=float)
    bat = X2b[batch_idx, :L, :L, chan].cpu().numpy()

    print(f"--- {name} ---")
    print(f" raw shape: {raw.shape}, tensor shape: {bat.shape}")
    print(f" raw[0:3,0:3]:\n{raw[:3,:3]}")
    print(f" bat[0:3,0:3]:\n{bat[:3,:3]}")
    print()



import os
import pickle
import torch

os.makedirs("/kaggle/working", exist_ok=True)

df_path = "/kaggle/working/seq_df_processed.pkl"
df.to_pickle(df_path)
print(f"✅ DataFrame saved to {df_path!r}")

ds_path = "/kaggle/working/rna_dataset.pkl"
with open(ds_path, "wb") as f:
    pickle.dump(dataset, f)
print(f"✅ Dataset pickled to {ds_path!r}")

torch_ds_path = "/kaggle/working/rna_dataset.pt"
torch.save(dataset, torch_ds_path)
print(f"✅ Dataset torch-saved to {torch_ds_path!r}")








