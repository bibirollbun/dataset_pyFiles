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



seq_df = pd.read_csv('/kaggle/input/seq-df-2d/seq_df_2D.csv')


# seq_df = seq_df.iloc[:200].reset_index(drop=True)

print(seq_df.shape)


import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def msa_depth(target_id: str) -> int:
    path = f"/kaggle/input/stanford-rna-3d-folding/MSA/{target_id}.MSA.fasta"
    try:
        with open(path, "r") as f:
            return sum(1 for line in f if line.startswith(">"))
    except FileNotFoundError:
        return 0

target_ids = seq_df["target_id"].tolist()
depths = []
with ProcessPoolExecutor() as exec:
    for d in tqdm(exec.map(msa_depth, target_ids), total=len(target_ids), desc="MSA depth"):
        depths.append(d)

seq_df["msa_depth"] = depths
print(seq_df[["target_id","msa_depth"]].head())



import numpy as np
import pandas as pd
from Bio import AlignIO
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

def compute_msa_diversity(target_id: str):
    """
    Read MSA/{target_id}.MSA.fasta, compute average pairwise identity.
    Returns (target_id, diversity)
    """
    path = f"/kaggle/input/stanford-rna-3d-folding/MSA/{target_id}.MSA.fasta"
    try:
        aln = AlignIO.read(path, "fasta")
    except Exception:
        return target_id, np.nan

    seqs = [str(rec.seq) for rec in aln]
    n = len(seqs)
    if n < 2:
        return target_id, 1.0

    L = len(seqs[0])
    arr = np.array([list(s) for s in seqs], dtype='<U1')

    total_matches = 0
    total_positions = 0
    for i in range(n - 1):
        eq = (arr[i] == arr[i+1:])
        total_matches   += eq.sum()
        total_positions += eq.size

    diversity = total_matches / total_positions
    return target_id, float(diversity)

if __name__ == "__main__":
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(compute_msa_diversity, seq_df.target_id),
            total=len(seq_df),
            desc="Computing MSA diversity"
        ))

    diversity_dict = dict(results)
    seq_df["msa_diversity"] = seq_df["target_id"].map(diversity_dict)

    print(seq_df[["target_id","msa_diversity"]].head())



import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def compute_gc(seq: str) -> float:
    L = len(seq)
    if L == 0:
        return 0.0
    return (seq.count('G') + seq.count('C')) / L

def parallel_gc(seqs, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(compute_gc, seqs),
                            total=len(seqs),
                            desc="Computing GC"))
    return results

seqs = seq_df['sequence'].tolist()
gc_vals = parallel_gc(seqs)
seq_df['gc_content'] = gc_vals

print(seq_df[['target_id','sequence','gc_content']].head())



import pandas as pd
import numpy as np
import RNA
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

def predict_secstruct(seq: str):
    """Return dot-bracket, MFE, and trimmed pairing-probability matrix"""
    fc = RNA.fold_compound(seq)
    dotbracket, mfe = fc.mfe()
    fc.pf()
    pmat_full = fc.bpp()
    pmat = np.array(pmat_full)
    L = len(dotbracket)
    pprob = pmat[1:L+1, 1:L+1]
    return dotbracket, mfe, pprob

def count_stems_loops(dot: str):
    """
    Count the number of stems (contiguous '(' runs) and loops (contiguous '.' runs)
    in a dot-bracket string
    """
    stems = 0
    in_stem = False
    loops = 0
    in_loop = False
    for c in dot:
        if c == '(':
            if not in_stem:
                stems += 1
            in_stem = True
        else:
            in_stem = False
        if c == '.':
            if not in_loop:
                loops += 1
            in_loop = True
        else:
            in_loop = False
    return stems, loops

def process_sequence(seq: str):
    dot, mfe, pprob = predict_secstruct(seq)
    return count_stems_loops(dot)

with ProcessPoolExecutor() as executor:
    results = list(
        tqdm(
            executor.map(process_sequence, seq_df['sequence']),
            total=len(seq_df),
            desc="Counting stems/loops"
        )
    )

stems, loops = zip(*results)
seq_df['n_stems'] = stems
seq_df['n_loops'] = loops

print(seq_df[['target_id', 'n_stems', 'n_loops']].head())



import pandas as pd
import re
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

def parse_description(desc: str):
    """
    Returns (has_metal, has_ligand, category)
    where category in {'none','metal','ligand','both'}.
    """
    s = (desc or "").lower()

    metals = [
        r'\bmg\b', r'\bmagnesium\b',
        r'\bca\b', r'\bcalcium\b',
        r'\bzn\b', r'\bzinc\b',
        r'\bmn\b', r'\bmanganese\b',
        r'\bco\b', r'\bcobalt\b'
    ]
    has_metal = any(re.search(pat, s) for pat in metals)

    has_ligand = ('ligand' in s) or ('bound to' in s) or ('binding to' in s)

    if has_metal and has_ligand:
        cat = 'both'
    elif has_metal:
        cat = 'metal'
    elif has_ligand:
        cat = 'ligand'
    else:
        cat = 'none'

    return has_metal, has_ligand, cat

def _worker(desc):
    return parse_description(desc)

if __name__ == "__main__":
    n_procs = max(1, cpu_count() - 1)

    with Pool(n_procs) as pool:
        results = list(
            tqdm(
                pool.imap(_worker, seq_df["description"].fillna("")),
                total=len(seq_df),
                desc="Parsing descriptions"
            )
        )

    metals, ligs, cats = zip(*results)
    seq_df["has_metal"]             = metals
    seq_df["has_ligand"]            = ligs
    seq_df["ligand_metal_category"] = cats

    print(seq_df[["target_id","has_metal","has_ligand","ligand_metal_category"]].head())



import re
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def extract_experimental_method(all_seq_str: str) -> str:
    """
    Pulls the 'method' out of a FASTA header, e.g.
      >1SCL_A ... method: X-ray diffraction; resolution: 2.9 Å ...
    Returns the method (e.g. "X-ray diffraction") or None if not found
    """
    if not isinstance(all_seq_str, str):
        return None
    lines = all_seq_str.strip().splitlines()
    if not lines:
        return None
    header = lines[0]
    m = re.search(r"method[:=]\s*([^;|\s]+(?:\s+[^;|]+)*)", header, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

if __name__ == "__main__":
    all_seqs = seq_df["all_sequences"].fillna("").tolist()

    with Pool(processes=cpu_count()) as pool:
        methods = list(
            tqdm(pool.imap(extract_experimental_method, all_seqs),
                 total=len(all_seqs),
                 desc="Extracting expt methods")
        )

    seq_df["experimental_method"] = methods

    print(seq_df[["target_id", "experimental_method"]].head(10))

    seq_df["experimental_method"] = seq_df["experimental_method"].astype("category")



import pandas as pd
import re
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

def count_chains(fasta_str: str) -> int:
    if pd.isna(fasta_str) or fasta_str.strip() == "":
        return 0
    # count lines that start with '>'
    return len([_ for _ in fasta_str.splitlines() if _.startswith(">")])

n = len(seq_df)
with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
    counts = list(tqdm(exe.map(count_chains, seq_df["all_sequences"]),
                       total=n,
                       desc="Counting partner chains"))

seq_df["partner_chains_count"] = counts

print(seq_df[["target_id","partner_chains_count"]].head())



seq_df.to_csv("seq_df_global.csv", index=False)




























