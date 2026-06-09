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
#lab_df = lab_df1
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
#seq_df = seq_df1
print(seq_df.shape)
print(seq_df.head())
print(seq_df.tail())



import os
import numpy as np
import pandas as pd

nt2idx = {"A": 0, "C": 1, "G": 2, "U": 3}

def one_hot_encode(seq: str) -> np.ndarray:
    """
    Convert an RNA sequence into an (L,4) one-hot array
    Unknown bases (e.g. N) are left as all-zero rows
    """
    L = len(seq)
    arr = np.zeros((L, 4), dtype=int)
    for i, nt in enumerate(seq):
        j = nt2idx.get(nt)
        if j is not None:
            arr[i, j] = 1
    return arr

example_seq = seq_df.loc[0, "sequence"]
oh = one_hot_encode(example_seq)
print(f"First sequence length: {len(example_seq)} → one-hot shape: {oh.shape}")
print(oh[:5])

seq_df["onehot"] = seq_df["sequence"].apply(one_hot_encode)
print(seq_df[['target_id'] + ['onehot']].head())



import pandas as pd

seq_df["length"] = seq_df["sequence"].str.len()

print(seq_df.head())



import numpy as np
import pandas as pd

def compute_mononuc_freq(seq: str) -> np.ndarray:
    """
    Compute mononucleotide frequencies for A, C, G, U
    Returns a length-4 vector: [f_A, f_C, f_G, f_U]
    """
    L = len(seq)
    freqs = np.array([seq.count(nuc) / L for nuc in ["A", "C", "G", "U"]], dtype=float)
    return freqs

example_seq = seq_df.loc[0, "sequence"]
print("Frequencies for", seq_df.loc[0, "target_id"], ":", compute_mononuc_freq(example_seq))

all_freqs = np.stack(seq_df["sequence"].map(compute_mononuc_freq).values)
seq_df[["freq_A", "freq_C", "freq_G", "freq_U"]] = all_freqs

print(seq_df.head())



import itertools
import numpy as np
import pandas as pd

def get_kmer_bias(seq: str, k: int):
    alphabet = ['A','C','G','U']
    kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]
    counts = dict.fromkeys(kmers, 0)
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(nt in alphabet for nt in kmer):
            counts[kmer] += 1
            total += 1
    if total > 0:
        freqs = np.array([counts[k]/total for k in kmers], dtype=float)
    else:
        freqs = np.zeros(len(kmers), dtype=float)
    return freqs, kmers

di_freqs, di_labels = get_kmer_bias("ACGU", k=2)   # labels only
_,        tri_labels = get_kmer_bias("ACGU", k=3)

def compute_kmer_features(seq: str) -> pd.Series:
    di_freqs, _  = get_kmer_bias(seq, 2)
    tri_freqs, _ = get_kmer_bias(seq, 3)
    data = {}
    for label, freq in zip(di_labels, di_freqs):
        data[f"di_{label}"] = freq
    for label, freq in zip(tri_labels, tri_freqs):
        data[f"tri_{label}"] = freq
    return pd.Series(data)

kmer_feats = seq_df['sequence'].apply(compute_kmer_features)

seq_df = pd.concat([seq_df, kmer_feats], axis=1)

print(seq_df.columns.tolist())
print(seq_df[['target_id'] + [f"di_{l}" for l in di_labels] + [f"tri_{l}" for l in tri_labels]].head())



import pandas as pd
import RNA
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import LinearFold as lf

def mfe_and_dot(seq: str):
    """Return (dot-bracket, mfe) for a single RNA - takes much longer"""
    fc = RNA.fold_compound(seq)
    dot, mfe = fc.mfe()
    return dot, mfe

def mfe_and_dot_linear(seq):
    """Return (dot-bracket, mfe) for a single RNA - returns an approximate MFE fold"""
    dot, mfe = lf.fold(seq)
    return dot, mfe

seqs = seq_df['sequence'].tolist()
n_workers = cpu_count()

dots, mfes = [], []
with Pool(processes=n_workers) as pool:
    for dot, mfe in tqdm(pool.imap(mfe_and_dot_linear, seqs),
                         total=len(seqs),
                         desc="Folding RNAs"):
        dots.append(dot)
        mfes.append(mfe)

seq_df['dotbracket'] = dots
seq_df['mfe']        = mfes
seq_df['pair_flag']  = seq_df['dotbracket'].apply(
    lambda db: [1 if c in '()' else 0 for c in db]
)

print(seq_df.columns.tolist())
print(seq_df[['target_id'] + ['sequence'] + ['dotbracket'] + ['mfe'] + ['pair_flag']].head())



import numpy as np
import RNA
from joblib import Parallel, delayed
import forgi.graph.bulge_graph as fgb
from scipy.sparse import csr_matrix
import pandas as pd
import gc
from tqdm import tqdm

def loop_type_onehot(dot: str) -> csr_matrix:
    """Create one-hot encoded loop types using sparse matrix."""
    bg = fgb.BulgeGraph.from_dotbracket(dot)
    L = len(dot)
    data = []
    rows = []
    cols = []
    cat2idx = {"stem": 0, "loop": 1, "bulge": 2}
    for pos in range(1, L+1):
        elem = bg.get_elem(pos)
        if elem.startswith("s"):
            cat = "stem"
        elif elem.startswith("i"):
            dims = bg.get_bulge_dimensions(elem)
            cat = "bulge" if (dims[0] == 0 or dims[1] == 0) else "loop"
        else:
            cat = "loop"
        rows.append(pos-1)
        cols.append(cat2idx[cat])
        data.append(1)
    
    return csr_matrix((data, (rows, cols)), shape=(L, 3))

def compute_pp_and_onehot(seq, dot):
    """Given (sequence, dotbracket), return (pp_matrix, loop_onehot)."""
    fc = RNA.fold_compound(seq)
    fc.pf()
    pmat = np.array(fc.bpp())
    L = len(dot)
    pp = pmat[1:L+1, 1:L+1] 
    onehot = loop_type_onehot(dot)
    return pp, onehot

def process_in_chunks(inputs, chunk_size=10):
    """Process inputs in chunks to optimize memory usage."""
    pp_list, onehot_list = [], []

    for start in tqdm(range(0, len(inputs), chunk_size), desc="Processing Chunks", unit="chunk"):
        end = min(start + chunk_size, len(inputs))
        chunk = inputs[start:end]

        results = Parallel(n_jobs=-1)(delayed(compute_pp_and_onehot)(seq, dot) for seq, dot in chunk)

        for pp, onehot in results:
            pp_list.append(pp)
            onehot_list.append(onehot)

        del results
        gc.collect() 

    return pp_list, onehot_list

inputs = list(zip(seq_df['sequence'], seq_df['dotbracket']))

pp_list, onehot_list = process_in_chunks(inputs, chunk_size=500)

seq_df['pp'] = pp_list
seq_df['loop_type_onehot'] = onehot_list

print("Shapes on row 0:")
print(" pp shape:", seq_df.loc[0,'pp'].shape)
print(" loop_type_onehot shape:", seq_df.loc[0,'loop_type_onehot'].shape)

print(seq_df.columns.tolist())
print(seq_df[['target_id', 'sequence', 'pp', 'loop_type_onehot']].head())



from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import RNA

def compute_loop_dG_res(seq: str):
    seq = seq.upper()
    L = len(seq)
    fc = RNA.fold_compound(seq)
    dot, mfe = fc.mfe()
    ptable = RNA.ptable(dot)
    loop_contrib = np.zeros(L, dtype=float)
    for i in range(1, L+1):
        j = ptable[i]
        if j > i:
            E_loop = fc.eval_loop_pt(i, ptable) / 100.0
            residues = list(range(i-1, j))
            per_res = E_loop / len(residues)
            for idx in residues:
                loop_contrib[idx] += per_res
    return loop_contrib

sequences = seq_df['sequence'].tolist()

with Pool(processes=cpu_count()) as pool:
    loop_results = list(tqdm(
        pool.imap(compute_loop_dG_res, sequences),
        total=len(sequences),
        desc="Computing loop ΔG"
    ))

seq_df['dG_loop_res'] = loop_results

seq_df['total_dG']             = seq_df.dG_loop_res.map(np.sum).round(1)

print(seq_df[['target_id'] + ['sequence'] + ['total_dG']].head())



import numpy as np
import pandas as pd
import RNA
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def predict_sa(seq: str) -> np.ndarray:
    """
    Approximate per‐residue solvent accessibility as
    the probability of being unpaired:
      sa[i] = 1 − sum_j p_pair(i,j)
    where p_pair is the ViennaRNA pairing‐probability matrix.
    Returns an array of shape (L,)
    """
    fc = RNA.fold_compound(seq)
    fc.pf()
    pmat_full = np.array(fc.bpp())
    L = len(seq)
    pprob = pmat_full[1 : L+1, 1 : L+1]
    sa = 1.0 - pprob.sum(axis=1)
    return sa

sequences = seq_df['sequence'].tolist()
n_workers = max(1, cpu_count() - 1)

with Pool(n_workers) as pool:
    sa_arrays = list(
        tqdm(pool.imap(predict_sa, sequences),
             total=len(sequences),
             desc="Computing SA")
    )

seq_df['sa'] = sa_arrays

print(seq_df[['target_id', 'sequence', 'sa']].head())



import numpy as np
import pandas as pd
import math
from collections import Counter
from Bio import AlignIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def compute_flexibility_index(msa_path: str) -> np.ndarray:
    """
    Reads an MSA FASTA, computes Shannon entropy at each column,
    and returns an (L,) array of entropies.
    """
    aln = AlignIO.read(msa_path, "fasta")
    seqs = [str(rec.seq) for rec in aln]
    L = len(seqs[0])
    arr = np.array([list(s) for s in seqs])
    entropies = np.zeros(L, dtype=float)
    for i in range(L):
        col = arr[:, i]
        counts = Counter(col)
        total = sum(counts.values())
        h = 0.0
        for cnt in counts.values():
            p = cnt / total
            h -= p * math.log2(p)
        entropies[i] = h
    return entropies

base = "/kaggle/input/stanford-rna-3d-folding/MSA"
msa_paths = [f"{base}/{tid}.MSA.fasta" for tid in seq_df.target_id]

n_procs = max(1, cpu_count() - 1)
with Pool(processes=n_procs) as pool:
    flex_cols = list(tqdm(
        pool.imap(compute_flexibility_index, msa_paths),
        total=len(msa_paths),
        desc="Computing flexibility"
    ))

seq_df["flexibility_index"] = flex_cols

for tid, seq, flex in zip(seq_df.target_id, seq_df.sequence, seq_df.flexibility_index):
    assert len(flex) == len(seq), f"Length mismatch for {tid}"

print(seq_df[["target_id","sequence","flexibility_index"]].head())



!apt-get update -qq && apt-get install -qq -y infernal
!wget -q ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz
!gunzip Rfam.cm.gz
!cmpress Rfam.cm



import subprocess, tempfile
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp


MOTIF_ACCESSIONS = {
    "GNRA_tetraloop": "RF00163",
    "UNCG_tetraloop": "RF00162",
    "CUUG_tetraloop": "RF00488",
    "kink-turn":       "RF00173",
}
M = len(MOTIF_ACCESSIONS)

def scan_rfam_motifs(seq: str,
                     rfam_cm: str = "Rfam.cm",
                     evalue: float = 0.01):
    """
    Runs `cmscan` on a single-sequence FASTA and returns an (L×M) binary matrix.
    Skips any lines that don't have enough columns
    """
    L = len(seq)
    mat = np.zeros((L, M), dtype=np.int8)

    with tempfile.NamedTemporaryFile("w+", suffix=".fa") as fa:
        fa.write(f">seq\n{seq}\n")
        fa.flush()
        cmd = [
            "cmscan", "--noali", "--cut_ga",
            "--tblout", "/dev/stdout",
            rfam_cm, fa.name
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
    
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        acc   = parts[1]                
        try:
            start = int(parts[6])       
            stop  = int(parts[7])       
        except ValueError:
            continue
        if acc in MOTIF_ACCESSIONS.values():
            j = list(MOTIF_ACCESSIONS.values()).index(acc)
            mat[start-1:stop, j] = 1

    return mat

def _worker(record):
    return scan_rfam_motifs(record["sequence"])

records = seq_df[["sequence"]].to_dict(orient="records")

with mp.Pool(mp.cpu_count()) as pool:
    mats = list(tqdm(pool.imap(_worker, records),
                     total=len(records),
                     desc="Scanning Rfam motifs"))

seq_df["motif_match_matrix"] = mats

for idx, row in seq_df.head(5).iterrows():
    L = len(row.sequence)
    print(row.target_id, row.motif_match_matrix.shape, "expected", (L, M))

print(seq_df[["target_id","sequence","motif_match_matrix"]].head())



seq_df.to_csv("seq_df_1D.csv", index=False)




