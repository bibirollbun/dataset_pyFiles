import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)



!pip install viennarna
!pip install forgi
!pip install LinearFold
!pip install --no-deps pydca
!pip install biopython



import pandas as pd
lab_df1 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
lab_df2 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")
lab_df3 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
lab_df1['dataset'] = 'trainv1'
lab_df2['dataset'] = 'trainv2'
lab_df3['dataset'] = 'val'
lab_df = pd.concat([lab_df1, lab_df2, lab_df3], ignore_index=True)
# lab_df = lab_df1
print(lab_df.shape)
print(lab_df.head())
print(lab_df.tail())



import pandas as pd
seq_df1 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
seq_df2 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv")
seq_df3 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
seq_df4 = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
seq_df1['dataset'] = 'trainv1'
seq_df2['dataset'] = 'trainv2'
seq_df3['dataset'] = 'val'
seq_df4['dataset'] = 'test'
seq_df = pd.concat([seq_df1, seq_df2, seq_df3, seq_df4], ignore_index=True)
# seq_df = seq_df1
print(seq_df.shape)
print(seq_df.head())
print(seq_df.tail())



seq_df = pd.read_csv('/kaggle/input/seq-df/seq_df.csv')


# seq_df = seq_df.iloc[:200].reset_index(drop=True)

print(seq_df.shape)


import os
import math
import numpy as np
import pandas as pd
from Bio import AlignIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from numba import njit, prange

@njit(parallel=True)
def _mi_numba(A, q):
    N, L = A.shape
    p = np.zeros((L, q))
    for i in prange(L):
        for n in range(N):
            p[i, A[n, i]] += 1
        for sym in range(q):
            p[i, sym] /= N

    mi = np.zeros((L, L))
    for i in prange(L):
        for j in range(i, L):
            joint = np.zeros((q, q))
            for n in range(N):
                joint[A[n, i], A[n, j]] += 1
            for a in range(q):
                for b in range(q):
                    joint[a, b] /= N

            m_ij = 0.0
            for a in range(q):
                for b in range(q):
                    pij = joint[a, b]
                    if pij > 0.0:
                        m_ij += pij * math.log2(pij / (p[i, a] * p[j, b]))
            mi[i, j] = m_ij
            mi[j, i] = m_ij
    return mi

def compute_mi_numba(tid):
    """Load MSA for tid, encode it, and return (tid, mi_matrix)"""
    path = f"/kaggle/input/stanford-rna-3d-folding/MSA/{tid}.MSA.fasta"
    aln  = AlignIO.read(path, "fasta")
    seqs = [str(rec.seq) for rec in aln]
    alphabet = sorted({ch for s in seqs for ch in s})
    sym2i    = {s: i for i, s in enumerate(alphabet)}
    q        = len(alphabet)
    A = np.array([[sym2i[ch] for ch in s] for s in seqs], dtype=np.int64)

    mi = _mi_numba(A, q)
    return tid, mi

nproc = max(1, cpu_count())
with Pool(nproc) as pool:
    results = list(tqdm(
        pool.imap_unordered(compute_mi_numba, seq_df.target_id),
        total=len(seq_df),
        desc="Computing MI"
    ))

mi_dict = dict(results)
seq_df['mi_matrix'] = seq_df.target_id.map(mi_dict)

print(seq_df.loc[:, ['target_id', 'mi_matrix']].head())



import os
import numpy as np
import pandas as pd
from Bio import AlignIO
import jax
import jax.numpy as jnp
from jax import jit
from tqdm.auto import tqdm

def load_msa(target_id):
    path = f"/kaggle/input/stanford-rna-3d-folding/MSA/{target_id}.MSA.fasta"
    aln  = AlignIO.read(path, "fasta")
    seqs = [str(rec.seq) for rec in aln]
    alpha = {'A':0,'C':1,'G':2,'U':3,'-':4}
    M = np.array([[alpha.get(r,4) for r in seq] for seq in seqs], dtype=np.int32)
    return M

@jit
def mf_dca_jax(seq_matrix, theta=0.8, H=0.5, lambda_reg=0.01):
    """
    seq_matrix: int32 array (N, L) with values in [0..4]
    returns couplings (L, L) on GPU
    """
    N, L = seq_matrix.shape
    q = 5
    oh = jax.nn.one_hot(seq_matrix, q)

    eq = jnp.einsum("n l a, m l a->n m", oh, oh) / L
    w = 1.0 / jnp.sum(eq >= theta, axis=1)
    M_eff = jnp.sum(w)

    f_i = jnp.einsum("n, n l a->l a", w, oh)
    f_i = (f_i + H/q) / (M_eff + H)

    f_ij = jnp.einsum("n, n i a, n j b->i j a b", w, oh, oh)
    f_ij = (f_ij + H/(q*q)) / (M_eff + H)

    diff = f_ij - jnp.einsum("i a, j b->i j a b", f_i, f_i)
    C = diff.reshape((L*q, L*q))
    C = C + lambda_reg * jnp.eye(L*q)

    invC = jnp.linalg.inv(C)
    
    invC4 = invC.reshape((L, q, L, q))
    J = jnp.sqrt(jnp.sum(invC4**2, axis=(1,3)))
    return J

_ = mf_dca_jax(jnp.zeros((2,2), dtype=jnp.int32))

results = []
for tid in tqdm(seq_df.target_id, desc="MF-DCA (GPU)"):
    M = load_msa(tid)
    J = mf_dca_jax(M)
    results.append((tid, np.array(J)))  

dca_dict = dict(results)
seq_df["dca"] = seq_df.target_id.map(dca_dict)

tid0 = seq_df.loc[0, "target_id"]
print(tid0, seq_df.loc[0, "dca"].shape)
print(seq_df.loc[:, ['target_id', 'dca']].head())



import numpy as np
import pandas as pd
from Bio import AlignIO
from tqdm.auto import tqdm
import multiprocessing as mp

_base2num = {'A':0, 'C':1, 'G':2, 'U':3}

def compute_covariance(target_id):
    """
    For a given target_id, load its MSA and compute
    the Lﾃ有 covariance matrix on numeric窶親ncoded columns
    """
    msa_path = f"/kaggle/input/stanford-rna-3d-folding/MSA/{target_id}.MSA.fasta"
    aln = AlignIO.read(msa_path, "fasta")

    seqs = [str(rec.seq) for rec in aln]
    if len(seqs) == 0:
        return target_id, None

    numeric_rows = []
    L = len(seqs[0])
    for s in seqs:
        try:
            row = [_base2num[b] for b in s] 
        except KeyError:
            continue
        numeric_rows.append(row)

    M = np.array(numeric_rows, dtype=float)
    if M.shape[0] < 2:
        return target_id, np.zeros((L, L), dtype=float)

    Mc = M - M.mean(axis=0, keepdims=True)
    cov = (Mc.T @ Mc) / (M.shape[0] - 1)
    return target_id, cov

ids = seq_df["target_id"].tolist()
with mp.Pool(mp.cpu_count()) as pool:
    results = list(
        tqdm(pool.imap_unordered(compute_covariance, ids),
             total=len(ids),
             desc="Computing covariance")
    )

cov_dict = {tid: cov for tid, cov in results}
seq_df["covariance"] = seq_df["target_id"].map(cov_dict)

print(seq_df[["target_id","covariance"]].head())
print("Example shape:", seq_df.loc[0,"covariance"].shape)



import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

lab_df["target_id"] = lab_df["ID"].str.rsplit(pat="_", n=1).str[0]

labels = lab_df.dropna(subset=["x_1","y_1","z_1"])

bases = ["A","C","G","U"]
nt2idx = {n:i for i,n in enumerate(bases)}

pair_dists = {tuple(sorted((b1,b2))): [] for b1 in bases for b2 in bases}

for tid, grp in tqdm(labels.groupby("target_id"), desc="Gathering distances"):
    grp = grp.sort_values("resid")
    coords = grp[["x_1","y_1","z_1"]].to_numpy()
    resns  = grp["resname"].to_numpy()
    dmat = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)
    L = len(resns)
    for i in range(L):
        b1 = resns[i]
        if b1 not in nt2idx: 
            continue
        for j in range(i+1, L):
            b2 = resns[j]
            if b2 not in nt2idx:
                continue
            pair = tuple(sorted((b1, b2)))
            pair_dists[pair].append(dmat[i, j])

bp_mean = np.full((4,4), np.nan, dtype=np.float32)
for (b1, b2), dist_list in pair_dists.items():
    if dist_list:
        m = float(np.mean(dist_list))
        i, j = nt2idx[b1], nt2idx[b2]
        bp_mean[i, j] = bp_mean[j, i] = m

print("bp_mean (rows/cols = A,C,G,U):\n", bp_mean)

def compute_bp_potentials(seq: str):
    """
    Build an (L,L) matrix where entry (i,j) = bp_mean[base_i, base_j],
    skipping any non-ACGU bases
    """
    L = len(seq)
    mat = np.zeros((L, L), dtype=np.float32)
    for i, b1 in enumerate(seq):
        i1 = nt2idx.get(b1)
        if i1 is None:
            continue
        for j, b2 in enumerate(seq):
            i2 = nt2idx.get(b2)
            if i2 is None:
                continue
            mat[i, j] = bp_mean[i1, i2]
    return mat

with Pool(cpu_count()) as pool:
    bp_mats = list(
        tqdm(pool.imap(compute_bp_potentials, seq_df.sequence),
             total=len(seq_df),
             desc="Computing bp_potentials")
    )

seq_df["bp_potentials"] = bp_mats

print(seq_df.loc[:, ['target_id', 'bp_potentials']].head()) 



import numpy as np
import pandas as pd
import math
from numba import njit, prange

lab_df["target_id"] = lab_df["ID"].str.rsplit(pat="_", n=1).str[0]

coords_dict = {}
for tid, group in lab_df.groupby("target_id"):
    grp = group.sort_values("resid")
    coords_dict[tid] = grp[["x_1","y_1","z_1"]].to_numpy(dtype=np.float64)

@njit(parallel=True, cache=True)
def compute_distances_jit(coords):
    L = coords.shape[0]
    dist = np.empty((L, L), dtype=np.float64)
    for i in prange(L):
        xi, yi, zi = coords[i,0], coords[i,1], coords[i,2]
        for j in range(L):
            dx = xi - coords[j,0]
            dy = yi - coords[j,1]
            dz = zi - coords[j,2]
            dist[i, j] = math.sqrt(dx*dx + dy*dy + dz*dz)
    return dist

_ = compute_distances_jit(np.zeros((2,3), dtype=np.float64))

gt_distances = { tid: compute_distances_jit(coords)
                 for tid, coords in coords_dict.items() }

seq_df["gt_distances"] = seq_df["target_id"].map(gt_distances)

print(seq_df[["target_id","gt_distances"]].head())



import pandas as pd
import numpy as np
from numba import jit
from tqdm import tqdm
import multiprocessing as mp

lab_df["target_id"] = lab_df["ID"].str.rsplit(pat="_", n=1).str[0]
coords_df = lab_df[['target_id','resid','x_1','y_1','z_1']]
grouped = coords_df.groupby('target_id')

@jit(nopython=True)
def make_contact_map(coords: np.ndarray, thresh: float) -> np.ndarray:
    L = coords.shape[0]
    out = np.zeros((L, L), dtype=np.uint8)
    thr2 = thresh * thresh
    for i in range(L):
        xi, yi, zi = coords[i]
        for j in range(L):
            dx = xi - coords[j,0]
            dy = yi - coords[j,1]
            dz = zi - coords[j,2]
            if dx*dx + dy*dy + dz*dz <= thr2:
                out[i, j] = 1
    return out

def process_target(tid, threshold=8.0):
    grp = grouped.get_group(tid).sort_values('resid')
    coords = grp[['x_1','y_1','z_1']].values.astype(np.float64)
    cmap = make_contact_map(coords, threshold)
    return tid, cmap

targets = seq_df['target_id'].tolist()
threshold = 8.0

with mp.Pool(processes=mp.cpu_count()) as pool:
    args = [(t, threshold) for t in targets]
    results = list(tqdm(pool.starmap(process_target, args),
                        total=len(args),
                        desc="Computing contact maps"))

contact_dict = dict(results)
seq_df['contact_map'] = seq_df['target_id'].map(contact_dict)

print(seq_df[['target_id','contact_map']].head())



import numpy as np
import pandas as pd
import numba
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

coords_dict = {}
for tid, group in lab_df.groupby(lab_df["ID"].str.rsplit(pat="_", n=1).str[0]):
    g = group.sort_values('resid')
    coords = g[['x_1','y_1','z_1']].to_numpy(dtype=np.float64)
    coords_dict[tid] = coords

@numba.njit(parallel=True)
def compute_angle_matrix(coords):
    """
    Given coords[L,3], return angles[L,L] where
    angles[i,j] = angle between backbone vector at i and vector from i to j.
    """
    L = coords.shape[0]
    mat = np.zeros((L, L), dtype=np.float64)
    for i in numba.prange(L):
        if i < L-1:
            vi = coords[i+1] - coords[i]
        else:
            vi = coords[i] - coords[i-1]
        n_vi = np.sqrt(vi[0]**2 + vi[1]**2 + vi[2]**2) + 1e-8
        for j in range(L):
            r = coords[j] - coords[i]
            n_r = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) + 1e-8
            dot = vi[0]*r[0] + vi[1]*r[1] + vi[2]*r[2]
            c = dot / (n_vi * n_r)
            if c > 1: c = 1
            elif c < -1: c = -1
            mat[i, j] = np.arccos(c)
    return mat

def worker(tid):
    coords = coords_dict[tid]
    return tid, compute_angle_matrix(coords)

results = {}
with ProcessPoolExecutor() as exe:
    futures = [exe.submit(worker, tid) for tid in seq_df.target_id]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Angle matrices"):
        tid, ang = fut.result()
        results[tid] = ang

seq_df['angle_matrix'] = seq_df['target_id'].map(results)

print(seq_df[['target_id','sequence','angle_matrix']].head())



seq_df.to_csv("seq_df_2D.csv", index=False)




