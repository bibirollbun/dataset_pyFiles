# ==================================
# Standard Library Imports
# ==================================
import glob
import math
import os
import random
import time
import warnings

# ==================================
# Third-Party Imports
# ==================================
# Scientific Computing & Data Handling
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# PyTorch & Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler

# Scikit-learn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Bioinformatics
import tmtools
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================
# Initial Setup
# ==================================
# Suppress specific warnings for cleaner output
warnings.simplefilter('ignore', PDBConstructionWarning)

# ==================================
# Configuration
# ==================================
# --- Input Data Locations (Read-Only) ---
CIF_DIR = "/kaggle/input/stanford-rna-3d-folding/PDB_RNA"
BASE_DIR = "/kaggle/input/3d-rna-geoformer"
METADATA_PATH = "/kaggle/input/3d-rna-geoformer/full_metadata.csv"
CACHE_DIR = "/kaggle/input/3d-rna-geoformer/cache/cache"

# --- Output Locations (Writable) ---
OUTPUT_DIR = "/kaggle/working/"
PRETRAINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_structural_model.pth")
FINETUNED_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_functional_model.pth")

# --- Device and General Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
GRAD_ACCUMULATION_STEPS = 1
MAX_LEN = 300
EPOCHS = 10
LEARNING_RATE = 5e-5
CLIP_GRAD_NORM = 1.0
WARMUP_STEPS = 100

# --- Tuned Loss Weights ---
FAPE_WEIGHT = 1.5
TORSIONAL_WEIGHT = 0.5
STERIC_CLASH_WEIGHT = 0.1
SECONDARY_STRUCTURE_WEIGHT = 0.3
TRIPLET_WEIGHT = 1.0
FAPE_CLAMP_DIST = 2.0
INITIAL_FX_WEIGHT = 0.5
FINAL_FX_WEIGHT = 4.0

# --- Model Hyperparameters ---
N_BLOCKS = 4
D_MODEL = 128
D_POINT = 4
N_HEADS = 4
D_HEAD_SCALAR = 8
D_HEAD_POINT = 2
FF_DIM = 256
DROPOUT_RATE = 0.1

# ==================================
# Verification
# ==================================
print(f"âœ… Setup complete.")
print(f"Using device: {DEVICE}")
print(f"Training with Smart Batching (Batch Size: {BATCH_SIZE})")


def extract_info_from_cif(cif_path):
    SITE_ANNOTATION_TAGS = ["_struct_site.id", "_pdbx_struct_binding_site.id"]
    try:
        has_functional_site = 0
        with open(cif_path, 'r', errors='ignore') as f:
            for line in f:
                if any(line.strip().startswith(tag) for tag in SITE_ANNOTATION_TAGS):
                    has_functional_site = 1
                    break
        parser = MMCIFParser(QUIET=True)
        target_id = os.path.basename(cif_path).replace('.cif', '')
        structure = parser.get_structure(target_id, cif_path)
        res_map = {"A": "A", "U": "U", "G": "G", "C": "C"}
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_id, res_name = residue.get_id(), residue.get_resname().strip()
                    if res_id[0] == ' ' and res_name in res_map:
                        sequence += res_name
            break
        if not sequence: return None
        return {'target_id': target_id, 'sequence': sequence, 'has_functional_site': has_functional_site}
    except Exception:
        return None

def generate_metadata_if_needed(cif_dir, output_dir):
    full_metadata_path = os.path.join(output_dir, 'full_metadata.csv')
    if os.path.exists(full_metadata_path):
        print("âœ… Metadata file already exists. Skipping generation.")
        df = pd.read_csv(full_metadata_path)
        positive_count = df['has_functional_site'].sum()
        print(f"ğŸ“„ Found {int(positive_count)} positive functional site samples in the existing metadata.")
        return df
    cif_files = glob.glob(os.path.join(cif_dir, '*.cif'))
    if not cif_files: raise FileNotFoundError(f"No .cif files found in {cif_dir}")
    print(f"â�³ Generating metadata for {len(cif_files)} CIF files...")
    data = [extract_info_from_cif(path) for path in tqdm(cif_files, desc="Scanning CIFs")]
    df = pd.DataFrame([d for d in data if d])
    df.to_csv(full_metadata_path, index=False)
    print(f"âœ… Metadata generated and saved to {full_metadata_path}")
    return df

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CIF_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
full_df = generate_metadata_if_needed(CIF_DIR, BASE_DIR)
print("\nMetadata generation complete.")


print("--- Exploratory Data Analysis ---")
full_df['seq_length'] = full_df['sequence'].str.len()

# Graph 1: Distribution of Sequence Lengths
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(full_df['seq_length'], bins=50, kde=True)
plt.title('Distribution of RNA Sequence Lengths')
plt.xlabel('Sequence Length'); plt.ylabel('Count')

# Graph 2: Class Distribution
plt.subplot(1, 2, 2)
full_df['has_functional_site'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'], labels=['Non-Functional', 'Functional'])
plt.title('Class Distribution: Functional vs. Non-Functional')
plt.ylabel('')
plt.tight_layout(); plt.show()

# Graph 3: Nucleotide Composition
nucleotide_counts = pd.Series(list(''.join(full_df['sequence']))).value_counts()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=nucleotide_counts.index, y=nucleotide_counts.values)
plt.title('Overall Nucleotide Composition')
plt.xlabel('Nucleotide'); plt.ylabel('Total Count')

# Graph 4: Sequence Length by Class
plt.subplot(1, 2, 2)
sns.boxplot(x='has_functional_site', y='seq_length', data=full_df)
plt.title('Sequence Length vs. Functional Site Presence')
plt.xlabel('Has Functional Site'); plt.ylabel('Sequence Length')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout(); plt.show()

# Graph 5: GC Content Distribution
full_df['gc_content'] = full_df['sequence'].apply(lambda x: (x.count('G') + x.count('C')) / len(x))
plt.figure(figsize=(8, 5))
sns.histplot(data=full_df, x='gc_content', hue='has_functional_site', multiple='stack', bins=30, kde=True)
plt.title('GC Content Distribution by Class')
plt.xlabel('GC Content'); plt.ylabel('Count')
plt.show()


def calculate_dihedral(p0, p1, p2, p3):
    b0 = -1.0 * (p1 - p0); b1 = p2 - p1; b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_safe = np.divide(b1, b1_norm, out=np.zeros_like(b1), where=b1_norm!=0)
    v = b0 - np.sum(b0 * b1_safe, axis=-1, keepdims=True) * b1_safe
    w = b2 - np.sum(b2 * b1_safe, axis=-1, keepdims=True) * b1_safe
    x = np.sum(v * w, axis=-1); y = np.sum(np.cross(b1_safe, v) * w, axis=-1)
    return np.arctan2(y, x)

def process_cif_file(args):
    cif_path, sequence, atom_map, num_atoms = args
    site_residues = set()
    try:
        with open(cif_path, 'r', errors='ignore') as f: lines = f.readlines()
        in_site_gen_loop, header_map = False, {}
        for line in lines:
            s_line = line.strip()
            if s_line.startswith('loop_'): in_site_gen_loop, header_map = False, {}
            elif s_line.startswith('_struct_site_gen.'):
                in_site_gen_loop = True
                header_map[s_line] = len(header_map)
            elif in_site_gen_loop and not s_line.startswith('#') and s_line:
                parts = s_line.split()
                id_col, seq_col = header_map.get('_struct_site_gen.auth_asym_id'), header_map.get('_struct_site_gen.auth_seq_id')
                if id_col is not None and seq_col is not None and len(parts) > max(id_col, seq_col):
                    try: site_residues.add((parts[id_col], int(parts[seq_col])))
                    except (ValueError, IndexError): continue
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("RNA", cif_path)
        model, seq_len = structure[0], len(sequence)
        coords, fx_sites = np.full((seq_len, num_atoms, 3), np.nan, dtype=np.float32), np.zeros(seq_len, dtype=np.float32)
        res_idx = 0
        for chain in model:
            if res_idx >= seq_len: break
            for residue in chain:
                if res_idx >= seq_len: break
                res_id_tuple = residue.get_id()
                if res_id_tuple[0] == ' ' and residue.get_resname().strip() in ['A', 'U', 'G', 'C']:
                    if (chain.id, res_id_tuple[1]) in site_residues: fx_sites[res_idx] = 1.0
                    for atom in residue:
                        atom_name = atom.get_name().replace("*", "'")
                        if atom_name in atom_map: coords[res_idx, atom_map[atom_name], :] = atom.get_coord()
                    res_idx += 1
        atom_mask = ~np.isnan(coords).any(axis=-1)
        coords[np.isnan(coords)] = 0
        return {'target_id': os.path.basename(cif_path).replace('.cif', ''), 'sequence': sequence, 'coords': coords, 'atom_mask': atom_mask, 'fx_sites': fx_sites}
    except Exception: return None

def rna_collate_fn_fx(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    keys = batch[0].keys(); collated = {k: [d[k] for d in batch] for k in keys}
    padded_sequences = pad_sequence([torch.tensor(s) for s in collated['sequence']], batch_first=True, padding_value=4).long()
    coord_mask = (padded_sequences != 4).float()
    return (padded_sequences, pad_sequence([torch.from_numpy(c) for c in collated['coords']], batch_first=True),
            pad_sequence([torch.from_numpy(m) for m in collated['atom_mask']], batch_first=True), 
            pad_sequence([torch.from_numpy(f) for f in collated['fx_sites']], batch_first=True),
            coord_mask, 
            pad_sequence([torch.from_numpy(t) for t in collated['torsionals']], batch_first=True),
            pad_sequence([torch.from_numpy(m) for m in collated['torsionals_mask']], batch_first=True), 
            collated['target_id'])

class LengthBasedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last):
        self.dataset, self.batch_size, self.drop_last = dataset, batch_size, drop_last
        self.groups = {}
        for idx in range(len(dataset)):
            length = len(dataset.metadata_df.iloc[idx]['sequence'])
            if length not in self.groups: self.groups[length] = []
            self.groups[length].append(idx)
        self.batches = self._generate_batches()
    def _generate_batches(self):
        batches = []
        for group in self.groups.values():
            random.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batch = group[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last: batches.append(batch)
        random.shuffle(batches)
        return batches
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)

class RNADataset(Dataset):
    def __init__(self, metadata_df, cif_dir, cache_dir, max_len=350, coord_mean=None, coord_std=None):
        self.metadata_df = metadata_df.copy()
        if max_len: self.metadata_df = self.metadata_df[self.metadata_df['sequence'].str.len() <= max_len].reset_index(drop=True)
        self.cif_dir, self.cache_dir = cif_dir, cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.atom_order=["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N1","C2","N3","C4","C5","C6","N7","C8","N9"]
        self.atom_map = {name: i for i, name in enumerate(self.atom_order)}
        self.num_atoms = len(self.atom_order)
        self.nuc_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        self.data_cache = [None] * len(self.metadata_df)
        self.coord_mean, self.coord_std = coord_mean, coord_std
        if self.coord_mean is None: self._calculate_normalization_stats()
    def _calculate_normalization_stats(self):
        print("Calculating normalization stats (parsing files if not cached)...")
        all_coords_list = []
        for idx in tqdm(range(len(self)), desc="Scanning for Stats"):
            data = self._get_item_data(idx)
            if data and data['atom_mask'].any(): all_coords_list.append(data['coords'][data['atom_mask']])
        if not all_coords_list: self.coord_mean, self.coord_std = 0.0, 1.0; return
        all_coords_np = np.concatenate(all_coords_list)
        self.coord_mean, self.coord_std = np.mean(all_coords_np), np.std(all_coords_np)
        print(f"Coord Mean: {self.coord_mean:.4f}, Std: {self.coord_std:.4f}")
    def _get_item_data(self, idx):
        if idx < len(self.data_cache) and self.data_cache[idx] is not None: return self.data_cache[idx]
        row = self.metadata_df.iloc[idx]
        target_id, sequence = row['target_id'], row['sequence']
        cache_path = os.path.join(self.cache_dir, f"{target_id}.pt")
        if os.path.exists(cache_path): 
            data = torch.load(cache_path, weights_only=False)
            if idx < len(self.data_cache): self.data_cache[idx] = data
            return data
        data = process_cif_file((os.path.join(self.cif_dir, f"{target_id}.cif"), sequence, self.atom_map, self.num_atoms))
        if data: 
            torch.save(data, cache_path)
            if idx < len(self.data_cache): self.data_cache[idx] = data
        return data
    def _get_torsional_angles(self, coords, atom_mask):
        seq_len,_,_ = coords.shape; torsionals=np.zeros((seq_len,7,2),dtype=np.float32); torsionals_mask=np.zeros((seq_len,7),dtype=np.float32)
        indices = self.atom_map
        p,o5p,c5p,c4p,c3p,o3p = (indices.get(n, -1) for n in ["P","O5'","C5'","C4'","C3'","O3'"])
        for i in range(seq_len):
            if i>0 and all(atom_mask[i-1,idx] for idx in [c4p,c3p,o3p] if idx!=-1) and atom_mask[i,p]:
                angle=calculate_dihedral(coords[i-1,c4p],coords[i-1,c3p],coords[i-1,o3p],coords[i,p]); torsionals[i,0,:]=[np.sin(angle),np.cos(angle)]; torsionals_mask[i,0]=1.0
            if i>0 and all(atom_mask[i-1,o3p] and atom_mask[i,idx] for idx in [p,o5p,c5p] if idx!=-1):
                angle=calculate_dihedral(coords[i-1,o3p],coords[i,p],coords[i,o5p],coords[i,c5p]); torsionals[i,1,:]=[np.sin(angle),np.cos(angle)]; torsionals_mask[i,1]=1.0
        return torsionals,torsionals_mask
    def __len__(self): return len(self.metadata_df)
    def __getitem__(self, idx):
        data = self._get_item_data(idx)
        if data is None: return self.__getitem__(np.random.randint(0, len(self)))
        seq_tokens = [self.nuc_map.get(n, 4) for n in data['sequence']]
        torsionals, torsionals_mask = self._get_torsional_angles(data['coords'], data['atom_mask'])
        coords_for_norm = np.copy(data['coords']); coords_for_norm[~data['atom_mask']] = self.coord_mean
        normalized_coords = (coords_for_norm - self.coord_mean) / (self.coord_std + 1e-8)
        normalized_coords[~data['atom_mask']] = 0
        return {"sequence":seq_tokens, "coords":normalized_coords, "atom_mask":data['atom_mask'], "fx_sites":data['fx_sites'], "torsionals":torsionals, "torsionals_mask":torsionals_mask, "target_id":data['target_id']}

class RefinementEGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(d_model*2 + 1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.update_mlp = nn.Sequential(nn.Linear(d_model*2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.coord_update_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1, bias=False))
    def forward(self, s, coords, edge_index):
        row, col = edge_index
        rel_coords = coords[row] - coords[col]
        dist = torch.norm(rel_coords, p=2, dim=-1, keepdim=True)
        edge_features = torch.cat([s[row], s[col], dist], dim=-1)
        messages = self.message_mlp(edge_features)
        coord_update_scalar = self.coord_update_mlp(messages)
        coord_shifts = (rel_coords / (dist + 1e-8)) * coord_update_scalar
        agg_messages = torch.zeros_like(s).index_add_(0, col, messages.float())
        agg_coord_shifts = torch.zeros_like(coords).index_add_(0, col, coord_shifts)
        update_input = torch.cat([s, agg_messages], dim=-1)
        s_out = s + self.update_mlp(update_input)
        coords_out = coords + agg_coord_shifts
        return s_out, coords_out

class SpatialRefinementModule(nn.Module):
    def __init__(self, d_model, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([RefinementEGNNLayer(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, s, coords, coord_mask):
        B, L, _ = s.shape
        dist_matrix = torch.cdist(coords, coords)
        mask = coord_mask.unsqueeze(1) * coord_mask.unsqueeze(2)
        dist_matrix.masked_fill_(mask == 0, float('inf'))
        k = min(16, L)
        _, edge_index_col = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        base_row = torch.arange(L, device=s.device).unsqueeze(-1).expand(-1, k)
        rows, cols = [], []
        for i in range(B):
            offset = i * L
            rows.append(base_row + offset)
            cols.append(edge_index_col[i] + offset)
        row_tensor, col_tensor = torch.cat(rows).flatten(), torch.cat(cols).flatten()
        batch_edge_index = torch.stack([row_tensor, col_tensor])
        s_flat, coords_flat = s.view(B * L, -1), coords.view(B * L, -1)
        refined_s_flat = s_flat
        for layer in self.layers:
            refined_s_flat, _ = layer(refined_s_flat, coords_flat, batch_edge_index)
        refined_s = refined_s_flat.view(B, L, -1)
        return self.norm(s + refined_s)

class InvariantPointAttention(nn.Module):
    def __init__(self, d_model, d_point, n_heads, d_head_point, d_head_scalar):
        super().__init__()
        self.n_heads,self.d_head_point=n_heads,d_head_point
        self.q_scalar,self.k_scalar,self.v_scalar = [nn.Linear(d_model, n_heads * d_head_scalar) for _ in range(3)]
        self.q_point,self.k_point,self.v_point = [nn.Linear(d_point * 3, n_heads * d_head_point * 3) for _ in range(3)]
        self.trainable_point_weights = nn.Parameter(torch.randn(n_heads))
        self.attn_out = nn.Linear(n_heads * (d_head_scalar + d_head_point * 3), d_model)
        self.gamma = 1/math.sqrt(d_head_scalar)
        self.register_buffer('virtual_points', torch.randn(d_point, 3))
    def forward(self, s, z, rotations, translations, coord_mask):
        B,L,_=s.shape
        points=torch.einsum('blij,pj->blpi',rotations,self.virtual_points)+translations.unsqueeze(-2)
        points_flat=points.reshape(B,L,-1)
        q_s,k_s,v_s=self.q_scalar(s),self.k_scalar(s),self.v_scalar(s)
        q_p,k_p,v_p=self.q_point(points_flat),self.k_point(points_flat),self.v_point(points_flat)
        q_s,k_s,v_s=[x.reshape(B,L,self.n_heads,-1) for x in [q_s,k_s,v_s]]
        q_p,k_p,v_p=[x.reshape(B,L,self.n_heads,-1,3) for x in [q_p,k_p,v_p]]
        attn_logits=self.gamma*torch.einsum('bihd,bjhd->bijh',q_s,k_s)-0.5*torch.sum((q_p.unsqueeze(2)-k_p.unsqueeze(1))**2,dim=(-1,-2))*self.trainable_point_weights
        attn_logits+=z.unsqueeze(0).permute(0,2,3,1)
        mask=coord_mask.unsqueeze(1).unsqueeze(-1)*coord_mask.unsqueeze(2).unsqueeze(-1)
        attn_logits=attn_logits.masked_fill(mask==0,-1e9)
        attn=F.softmax(attn_logits,dim=2)
        result_s=torch.einsum('bijh,bjhd->bihd',attn,v_s)
        result_p=torch.einsum('bijh,bjhdp->bihdp',attn,v_p)
        output=self.attn_out(torch.cat([result_s.reshape(B,L,-1),result_p.reshape(B,L,-1)],dim=-1))
        return output

class EGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(d_model*2 + 1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.update_mlp = nn.Sequential(nn.Linear(d_model*2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.coord_update_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1, bias=False))
    def forward(self, s, coords, edge_index):
        row, col = edge_index
        rel_coords = coords[:, row] - coords[:, col]
        dist = torch.norm(rel_coords, p=2, dim=-1, keepdim=True)
        edge_features = torch.cat([s[:, row], s[:, col], dist], dim=-1)
        messages = self.message_mlp(edge_features)
        coord_update_scalar = self.coord_update_mlp(messages)
        coord_shifts = (rel_coords / (dist + 1e-8)) * coord_update_scalar
        agg_messages = torch.zeros_like(s).index_add_(1, col, messages.float())
        agg_coord_shifts = torch.zeros_like(coords).index_add_(1, col, coord_shifts)
        update_input = torch.cat([s, agg_messages], dim=-1)
        s_out = s + self.update_mlp(update_input)
        coords_out = coords + agg_coord_shifts
        return s_out, coords_out

class GeoformerIPABlock(nn.Module):
    def __init__(self, d_model, d_point, n_heads, d_head_point, d_head_scalar, ff_dim, dropout=0.1):
        super().__init__()
        self.ipa = InvariantPointAttention(d_model, d_point, n_heads, d_head_point, d_head_scalar)
        self.ipa_norm = nn.LayerNorm(d_model)
        self.egnn = EGNNLayer(d_model)
        self.egnn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(d_model)
    def _forward_impl(self, s, z, rotations, translations, edge_index, coord_mask):
        s = s + self.ipa(self.ipa_norm(s), z, rotations, translations, coord_mask)
        s_norm, translations_norm = self.egnn_norm(s), translations
        s, translations = self.egnn(s_norm, translations_norm, edge_index)
        s = s + self.ffn(self.ffn_norm(s))
        return s, translations
    def forward(self, args):
        s, z, rotations, translations, edge_index, coord_mask = args
        return self._forward_impl(s, z, rotations, translations, edge_index, coord_mask)

class StructureModule(nn.Module):
    def __init__(self, d_model, num_atoms): super().__init__(); self.num_atoms,self.atom_predictor=num_atoms,nn.Linear(d_model,num_atoms*3)
    def forward(self, s, rotations, translations):
        local_displacements = self.atom_predictor(s).view(*s.shape[:-1], self.num_atoms, 3)
        return torch.einsum('blij,blaj->blai', rotations, local_displacements) + translations.unsqueeze(-2)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, max_samples=1024):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.max_samples = max_samples
    def forward(self, features, labels):
        if features.shape[0] == 0: return torch.tensor(0.0, device=features.device)
        pos_indices = torch.where(labels == 1)[0]
        neg_indices = torch.where(labels == 0)[0]
        num_pos, num_neg = len(pos_indices), len(neg_indices)
        if num_pos < 2 or num_neg < 2: return torch.tensor(0.0, device=features.device)
        num_each = min(self.max_samples // 2, num_pos, num_neg)
        sampled_indices = torch.cat([pos_indices[torch.randperm(num_pos)[:num_each]], neg_indices[torch.randperm(num_neg)[:num_each]]])
        features, labels = features[sampled_indices], labels[sampled_indices]
        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos[mean_log_prob_pos != 0].mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=features.device)

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
    def forward(self, x, mask):
        attention_logits = self.attention_net(x).squeeze(-1)
        mask_value = torch.finfo(attention_logits.dtype).min
        attention_logits.masked_fill_(mask == 0, mask_value)
        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(1)
        pooled_features = torch.bmm(attention_weights, x).squeeze(1)
        return pooled_features

class GeoformerRNA(nn.Module):
    def __init__(self, n_blocks, d_model, d_point, n_heads, d_head_point, d_head_scalar, ff_dim, num_atoms, dropout, rel_pos_bins=32):
        super().__init__()
        self.embedding_s = nn.Embedding(5, d_model, padding_idx=4)
        self.rel_pos_embedding = nn.Embedding(2 * rel_pos_bins + 1, n_heads)
        self.rel_pos_bins = rel_pos_bins
        self.blocks = nn.ModuleList([GeoformerIPABlock(d_model, d_point, n_heads, d_head_point, d_head_scalar, ff_dim, dropout) for _ in range(n_blocks)])
        self.structure_module = StructureModule(d_model, num_atoms)
        self.to_s_point = nn.Linear(d_model, d_point * 3)
        self.spatial_refiner = SpatialRefinementModule(d_model)
        self.early_feature_proj = nn.Linear(d_model, d_model // 2)
        fused_dim = d_model + d_model // 2
        self.gate = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.Sigmoid())
        self.functional_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 1))
        self.attention_pool = AttentionPooling(fused_dim)
        self.projection_head = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.ReLU(), nn.Linear(fused_dim, 128))
        self.torsional_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 14))
        self.secondary_structure_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.ReLU())
    def forward(self, seq, coord_mask):
        B,L=seq.shape
        s=self.embedding_s(seq)
        pos=torch.arange(L,device=seq.device); rel_pos=torch.clamp(pos[None,:]-pos[:,None]+self.rel_pos_bins,0,2*self.rel_pos_bins)
        z=self.rel_pos_embedding(rel_pos).permute(2,0,1)
        rotations=torch.eye(3,device=seq.device,dtype=s.dtype).unsqueeze(0).unsqueeze(0).expand(B,L,-1,-1)
        translations=torch.zeros(B,L,3,device=seq.device,dtype=s.dtype)
        edge_index=torch.stack([torch.arange(L-1,device=seq.device),torch.arange(1,L,device=seq.device)],dim=0)
        s_early = None
        for i, block in enumerate(self.blocks):
            s,translations=block((s,z,rotations,translations,edge_index,coord_mask))
            s_point=self.to_s_point(s).view(B,L,-1,3)
            translations=translations+torch.einsum('blij,blaj->blai',rotations,s_point).mean(dim=-2)
            if i == 0: s_early = s
        final_coords=self.structure_module(s,rotations,translations)
        refined_s = self.spatial_refiner(s, translations, coord_mask)
        s_early_proj = self.early_feature_proj(s_early)
        raw_fused_features = torch.cat([s_early_proj, refined_s], dim=-1)
        gate_values = self.gate(raw_fused_features)
        fused_features = raw_fused_features * gate_values
        functional_logits=self.functional_head(fused_features).squeeze(-1)
        prototype = self.attention_pool(fused_features, coord_mask)
        triplet_features = self.projection_head(prototype)
        torsional_preds=F.normalize(self.torsional_head(s).view(B,L,7,2),dim=-1)
        s_ss=self.secondary_structure_head(s)
        ss_logits=torch.einsum('bid,bjd->bij',s_ss,s_ss)
        return final_coords,rotations,translations,functional_logits,torsional_preds,ss_logits,triplet_features

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self,optimizer,warmup_steps,total_steps,last_epoch=-1): self.warmup_steps,self.total_steps=warmup_steps,total_steps; super().__init__(optimizer,last_epoch)
    def get_lr(self):
        if self.warmup_steps>0 and self.last_epoch<self.warmup_steps: return [base_lr*(self.last_epoch+1)/self.warmup_steps for base_lr in self.base_lrs]
        progress=(self.last_epoch-self.warmup_steps)/max(1,self.total_steps-self.warmup_steps)
        return [base_lr*0.5*(1.0+math.cos(math.pi*progress)) for base_lr in self.base_lrs]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.pos_weight = alpha, gamma, pos_weight
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.pos_weight is not None:
            weight_tensor = torch.ones_like(targets); weight_tensor[targets == 1] = self.pos_weight.item()
            BCE_loss = BCE_loss * weight_tensor
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

def FAPE_loss(pred_coords,true_coords,rotations,translations,atom_mask,coord_mask,clamp_dist=10.0):
    rotations,translations=rotations.float(),translations.float(); pred_coords,true_coords=pred_coords.float(),true_coords.float()
    inv_rots=rotations.transpose(-1,-2)
    relative_true_coords=true_coords.unsqueeze(1)-translations.unsqueeze(2).unsqueeze(3)
    local_true_coords=torch.einsum('bilk,bijak->bijal',inv_rots,relative_true_coords)
    error=torch.sqrt(torch.sum((local_true_coords-pred_coords.unsqueeze(2))**2,dim=-1)+1e-8)
    mask=(coord_mask.unsqueeze(2)*coord_mask.unsqueeze(1)).unsqueeze(-1)*atom_mask.unsqueeze(1)
    return (torch.clamp(error,max=clamp_dist)*mask).sum()/(mask.sum()+1e-8)

def torsional_loss(pred_torsionals,true_torsionals,mask): return ((1-torch.sum(pred_torsionals*true_torsionals,dim=-1))*mask).sum()/(mask.sum()+1e-8)

def steric_clash_loss(pred_coords,atom_mask,coord_mask,c4_idx=5,clash_threshold=1.5):
    B,L,_,_=pred_coords.shape
    if L<2: return torch.tensor(0.0,device=pred_coords.device)
    coords,mask=pred_coords[:,:,c4_idx,:],(atom_mask[:,:,c4_idx]*coord_mask).bool()
    total_clash_loss=torch.tensor(0.0,device=pred_coords.device)
    for b in range(B):
        valid_coords=coords[b,mask[b]]
        if valid_coords.shape[0]<2: continue
        violations=F.relu(clash_threshold-torch.pdist(valid_coords))
        total_clash_loss+=violations.mean() if violations.numel()>0 else 0.0
    return total_clash_loss/B

def secondary_structure_geometric_loss(pred_coords,ss_logits,seq_map,atom_mask,atom_map):
    with torch.no_grad():
        c1_idx=atom_map.get("C1'")
        if c1_idx is None: return torch.tensor(0.0,device=pred_coords.device)
        dist_c1=torch.cdist(pred_coords[:,:,c1_idx,:],pred_coords[:,:,c1_idx,:])
        au_mask=((seq_map==0)|(seq_map==1)); gc_mask=((seq_map==2)|(seq_map==3))
        pseudo_labels=((dist_c1<10.0)&au_mask.unsqueeze(1)&au_mask.unsqueeze(2))|((dist_c1<9.0)&gc_mask.unsqueeze(1)&gc_mask.unsqueeze(2))
        pseudo_labels=pseudo_labels.float()
        pseudo_labels.diagonal(dim1=-2,dim2=-1).zero_()
        pseudo_labels*=atom_mask[:,:,c1_idx].unsqueeze(1)&atom_mask[:,:,c1_idx].unsqueeze(2)
    ss_logits_symm=(ss_logits+ss_logits.transpose(-1,-2))/2
    return F.binary_cross_entropy_with_logits(ss_logits_symm,pseudo_labels,reduction='none').sum()/(pseudo_labels.sum()+1e-8)


def calculate_rmsd(pred_coords,true_coords,atom_masks):
    sq_dist=torch.sum((pred_coords-true_coords)**2,dim=-1)
    num_valid=atom_masks.sum()
    return torch.sqrt((sq_dist*atom_masks).sum()/(num_valid+1e-8)).item() if num_valid>0 else 0.0

def train_epoch(model, loader, optimizer, scheduler, scaler, accumulation_steps, bce_pos_weight, atom_map, current_epoch, total_epochs):
    model.train(); losses = {k: 0 for k in ['total', 'fape', 'torsion', 'clash', 'fx', 'ss', 'triplet']}
    progress = current_epoch / max(1, total_epochs - 1)
    current_fx_weight = INITIAL_FX_WEIGHT + progress * (FINAL_FX_WEIGHT - INITIAL_FX_WEIGHT)
    focal_loss_fn = FocalLoss(alpha=0.1, gamma=2.0, pos_weight=bce_pos_weight).to(DEVICE)
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.5).to(DEVICE)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(tqdm(loader, desc=f"Training Epoch {current_epoch+1}/{total_epochs}", leave=False)):
        if batch is None: continue
        seqs, true_coords, atom_masks, fx_sites, coord_masks, true_torsionals, torsionals_masks = [t.to(DEVICE) for t in batch[:-1]]
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(DEVICE.type == 'cuda')):
            pred_coords, rotations, translations, functional_logits, pred_torsionals, ss_logits, triplet_features = model(seqs, coord_masks)
            loss_fape = FAPE_WEIGHT * FAPE_loss(pred_coords, true_coords, rotations, translations, atom_masks, coord_masks, clamp_dist=FAPE_CLAMP_DIST)
            loss_torsion = TORSIONAL_WEIGHT * torsional_loss(pred_torsionals, true_torsionals, torsionals_masks)
            loss_clash = STERIC_CLASH_WEIGHT * steric_clash_loss(pred_coords, atom_masks, coord_masks)
            loss_ss = SECONDARY_STRUCTURE_WEIGHT * secondary_structure_geometric_loss(pred_coords, ss_logits, seqs, atom_masks, atom_map)
            loss_fx = current_fx_weight * (focal_loss_fn(functional_logits, fx_sites) * coord_masks).sum() / (coord_masks.sum() + 1e-8)
            batch_labels = (fx_sites.sum(dim=1) > 0).long()
            pos_indices = torch.where(batch_labels == 1)[0]
            neg_indices = torch.where(batch_labels == 0)[0]
            loss_triplet = torch.tensor(0.0, device=DEVICE)
            if len(pos_indices) >= 2 and len(neg_indices) >= 1:
                anchor_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
                positive_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
                negative_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
                anchor = triplet_features[anchor_idx]
                positive = triplet_features[positive_idx]
                negative = triplet_features[negative_idx]
                loss_triplet = TRIPLET_WEIGHT * current_fx_weight * triplet_loss_fn(anchor, positive, negative)
            loss = loss_fape + loss_torsion + loss_clash + loss_ss + loss_fx + loss_triplet
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN or Inf loss at batch {i}. Skipping."); optimizer.zero_grad(set_to_none=True); continue
        scaler.scale(loss / accumulation_steps).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        for k, v in zip(losses.keys(), [loss, loss_fape, loss_torsion, loss_clash, loss_fx, loss_ss, loss_triplet]):
            losses[k] += v.item() if torch.is_tensor(v) else v
    return {k: v/len(loader) for k, v in losses.items()}

def validate_epoch(model, loader, nuc_map):
    model.eval(); all_results = []
    c4_idx = 5
    rev_nuc_map = {v: k for k, v in nuc_map.items()}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if batch is None: continue
            tensors_to_move, target_ids = batch[:-1], batch[-1]
            seqs, true_coords, atom_masks, fx_sites, coord_masks, _, _ = [t.to(DEVICE) for t in tensors_to_move]
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(DEVICE.type == 'cuda')):
                 pred_coords, _, _, functional_logits, _, _, _ = model(seqs, coord_masks)
            for i in range(seqs.shape[0]):
                mask = coord_masks[i].bool()
                seq_len = mask.sum().item()
                tm_score = 0.0
                try:
                    true_c4 = true_coords[i, :seq_len, c4_idx].cpu().numpy()
                    pred_c4 = pred_coords[i, :seq_len, c4_idx].cpu().numpy()
                    sequence_str = "".join([rev_nuc_map.get(token.item(), 'A') for token in seqs[i, :seq_len]])
                    if true_c4.shape[0] > 0 and pred_c4.shape[0] > 0:
                        res = tmtools.tm_align(true_c4, pred_c4, sequence_str, sequence_str)
                        tm_score = res.tm_norm_chain1
                except:
                    pass
                all_results.append({
                    'target_id': target_ids[i], 
                    'rmsd': calculate_rmsd(pred_coords[i], true_coords[i], atom_masks[i]),
                    'tm_score': tm_score,
                    'fx_true': fx_sites[i][mask].cpu().numpy().flatten().astype(int),
                    'fx_preds': (torch.sigmoid(functional_logits[i][mask]).cpu().numpy() > 0.5).astype(int)
                })
    return all_results


def plot_metrics_dashboard(results_df):
    if results_df.empty: return
    plt.figure(figsize=(14, 6)); sns.set_style("whitegrid")
    
    # Plot losses
    plt.subplot(1, 2, 1)
    for col in [c for c in results_df.columns if 'loss' in c]: 
        plt.plot(results_df.index, results_df[col], label=col.replace('loss_', ''))
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Loss Components'); plt.legend(); plt.yscale('log')
    
    # Plot validation metrics (RMSD and TM-Score)
    plt.subplot(1, 2, 2)
    ax1 = plt.gca()
    ax1.plot(results_df.index, results_df['val_rmsd'], label='Validation RMSD (Ã…)', color='green', marker='o')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('RMSD (Ã…)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(results_df.index, results_df['val_tm_score'], label='Validation TM-Score', color='red', marker='.')
    ax2.set_ylabel('TM-Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    
    plt.title('Validation Metrics'); plt.tight_layout(); plt.show()


y_stratify = full_df['has_functional_site']
if y_stratify.sum() < 2:
    y_stratify = None
    print("âš ï¸� WARNING: Not enough positive samples for stratification.")

train_df, val_df = train_test_split(full_df, test_size=0.1, random_state=42, stratify=y_stratify)
print(f"ğŸ”¬ Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

train_dataset = RNADataset(train_df, CIF_DIR, CACHE_DIR, max_len=MAX_LEN)
val_dataset = RNADataset(val_df, CIF_DIR, CACHE_DIR, max_len=MAX_LEN, coord_mean=train_dataset.coord_mean, coord_std=train_dataset.coord_std)

train_batch_sampler = LengthBasedBatchSampler(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=rna_collate_fn_fx, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=rna_collate_fn_fx, num_workers=2, pin_memory=True)

num_pos_residues = sum(d['fx_sites'].sum() for d in train_dataset.data_cache if d is not None and 'fx_sites' in d)
num_total_residues = sum(len(d['fx_sites']) for d in train_dataset.data_cache if d is not None and 'fx_sites' in d)

if num_pos_residues == 0:
    print("â�Œ CRITICAL ERROR: 0 positive residues found. Cannot train.")
    bce_pos_weight = torch.tensor([1.0], device=DEVICE)
else:
    num_neg_residues = num_total_residues - num_pos_residues
    bce_pos_weight = torch.tensor([num_neg_residues / (num_pos_residues + 1e-8)], device=DEVICE)
    print(f"âš–ï¸� BCE Positive Weight: {bce_pos_weight.item():.2f} ({int(num_pos_residues)} positive residues found)")

model = GeoformerRNA(N_BLOCKS,D_MODEL,D_POINT,N_HEADS,D_HEAD_POINT,D_HEAD_SCALAR,FF_DIM,num_atoms=train_dataset.num_atoms,dropout=DROPOUT_RATE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01) # Recommended addition
scaler = GradScaler(enabled=(DEVICE.type == 'cuda')) # This line is correct
scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, len(train_loader) * EPOCHS)

history = []
start_time = time.time()
best_val_tm_score = -1.0

for epoch in range(EPOCHS):
    print(f"\n{'='*20} EPOCH {epoch + 1}/{EPOCHS} {'='*20}")
    losses = train_epoch(model, train_loader, optimizer, scheduler, scaler, GRAD_ACCUMULATION_STEPS, bce_pos_weight, train_dataset.atom_map, current_epoch=epoch, total_epochs=EPOCHS)
    results = validate_epoch(model, val_loader, train_dataset.nuc_map)

    val_rmsd = np.mean([r['rmsd'] for r in results]) if results else float('inf')
    val_tm_score = np.mean([r['tm_score'] for r in results if r['tm_score'] > 0]) if results else 0.0

    print(f"  Epoch {epoch+1:02d}/{EPOCHS} -> Train Loss: {losses['total']:.4f} | Val RMSD: {val_rmsd:.4f} | Val TM-Score: {val_tm_score:.4f}")

    if val_tm_score > best_val_tm_score:
        best_val_tm_score = val_tm_score
        MODEL_SAVE_PATH = "best_geoformer_model.pt"

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ğŸ�… New best model saved to {MODEL_SAVE_PATH} (TM-Score: {best_val_tm_score:.4f})")

    epoch_data = {**{f'loss_{k}': v for k, v in losses.items()}, 'val_rmsd': val_rmsd, 'val_tm_score': val_tm_score}
    history.append(epoch_data)

print(f"\nâœ… Training finished in {(time.time() - start_time)/60:.2f} minutes.")
history_df = pd.DataFrame(history)
plot_metrics_dashboard(history_df)

