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


import os, glob, numpy as np, pandas as pd

# --- folders with submission files ---
DATA_DIRS = [
    "/kaggle/input/dhfhdfjhd",
    # "/kaggle/input/another_folder",  # add more folders if needed
]

ID_COL = "id"
PRED_COL = "accident_risk"
CLIP01 = True

# --- collect all CSV files from all folders ---
subs = []
for data_dir in DATA_DIRS:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = [f for f in files if "sample" not in f.lower() and "oof" not in f.lower()]
    subs.extend(files)

assert len(subs) >= 2, "Need at least 2 submissions"

# --- aliases for flexible column name detection ---
ID_ALIASES   = ["id", "ID", "Id", "row_id"]
PRED_ALIASES = ["accident_risk", "y", "prediction", "pred", "target", "label", "prob", "score"]

def pick_col(cols, aliases):
    cl = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in cl:
            return cl[a.lower()]
    return None

def read_sub(path, ID_COL="id", PRED_COL="accident_risk"):
    df = pd.read_csv(path)
    id_col = pick_col(df.columns, [ID_COL] + ID_ALIASES)
    pred_col = pick_col(df.columns, [PRED_COL] + PRED_ALIASES)
    if id_col is None:
        raise ValueError(f"{os.path.basename(path)}: ID column not found")
    if pred_col is None:
        numcols = df.select_dtypes(include="number").columns.tolist()
        numcols = [c for c in numcols if c != id_col]
        if len(numcols) == 1:
            pred_col = numcols[0]
        else:
            raise ValueError(f"{os.path.basename(path)}: prediction column not found")
    return df[[id_col, pred_col]].rename(columns={id_col: ID_COL, pred_col: PRED_COL})

# --- read all submissions ---
dfs = []
for f in subs:
    d = read_sub(f, ID_COL=ID_COL, PRED_COL=PRED_COL)
    dfs.append(d)
    print(f"{os.path.basename(f)}: loaded")

# --- validate IDs ---
base_ids = dfs[0][ID_COL].values
for i, df in enumerate(dfs):
    assert np.array_equal(np.sort(base_ids), np.sort(df[ID_COL].values)), f"ID mismatch: {subs[i]}"
    dfs[i] = df.set_index(ID_COL).loc[base_ids]

# --- merge into matrix ---
mat = np.stack([d[PRED_COL].to_numpy(float) for d in dfs], axis=1)
names = [os.path.basename(f) for f in subs]
N, M = mat.shape
print(f"{M} submissions, {N} rows")

# --- correlation pruning + median ensemble ---
corr = np.corrcoef(mat.T)
keep = []
for j in range(M):
    if not keep:
        keep.append(j)
        continue
    if all(abs(corr[j, k]) < 0.995 for k in keep):
        keep.append(j)

print(f"correlation-pruned: kept {len(keep)}/{M}")

# --- final correlation-pruned median ---
arr = np.median(mat[:, keep], axis=1)
if CLIP01:
    arr = np.clip(arr, 0.0, 1.0)

out = pd.DataFrame({ID_COL: base_ids, PRED_COL: arr})
out.to_csv("sub_corrprune_median.csv", index=False)
print("saved: sub_corrprune_median.csv")





