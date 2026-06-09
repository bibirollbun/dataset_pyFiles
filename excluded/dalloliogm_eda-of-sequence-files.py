!head -n 25 /kaggle/input/stanford-rna-3d-folding/train_sequences.csv


import pandas as pd
import csv

# Read only the first four columns, ignoring "all_sequences"
df = pd.read_csv(
    "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv",
    engine="python",
    quoting=csv.QUOTE_MINIMAL,
    usecols=["target_id", "sequence", "temporal_cutoff", "description", "all_sequences"]
)

print(df.head(10))


df.loc[df.target_id.str.contains("1HMH")]


# This should contain two entries for 1HMH_1 and 1HMH_2, but it seems the latter is lost. We do not need it for the prediction anyways.
df.loc[df.target_id.str.contains("1HMH")].all_sequences


df.loc[df.target_id.str.contains(">")]


df.head(10)

