import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno   # for missing data visualization (pip install missingno)
from pathlib import Path

# Configure plots
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)


data_dir = Path("/kaggle/input/stanford-rna-3d-folding")

train_sequences_path = data_dir / "train_sequences.csv"
train_labels_path    = data_dir / "train_labels.csv"
validation_sequences_path = data_dir / "validation_sequences.csv"
validation_labels_path    = data_dir / "validation_labels.csv"
test_sequences_path       = data_dir / "test_sequences.csv"
sample_submission_path    = data_dir / "sample_submission.csv"

# Read CSVs
df_train_seq = pd.read_csv(train_sequences_path)
df_train_lbl = pd.read_csv(train_labels_path)
df_val_seq   = pd.read_csv(validation_sequences_path)
df_val_lbl   = pd.read_csv(validation_labels_path)
df_test_seq  = pd.read_csv(test_sequences_path)
df_sub_sample= pd.read_csv(sample_submission_path)



print("== train_sequences.csv ==")
display(df_train_seq.head(5))
print(df_train_seq.info())

print("\n== train_labels.csv ==")
display(df_train_lbl.head(5))
print(df_train_lbl.info())

print("\n== validation_sequences.csv ==")
display(df_val_seq.head(5))
print(df_val_seq.info())

print("\n== validation_labels.csv ==")
display(df_val_lbl.head(5))
print(df_val_lbl.info())

print("\n== test_sequences.csv ==")
display(df_test_seq.head(5))
print(df_test_seq.info())

print("\n== sample_submission.csv ==")
display(df_sub_sample.head(5))
print(df_sub_sample.info())



print("== train_labels.csv numeric summary ==")
display(df_train_lbl.describe(include=[np.number]))

print("\n== validation_labels.csv numeric summary ==")
display(df_val_lbl.describe(include=[np.number]))

print("\n== sample_submission.csv numeric summary ==")
display(df_sub_sample.describe(include=[np.number]))


print("== Missing values: train_sequences ==")
display(df_train_seq.isnull().sum())

print("\n== Missing values: train_labels ==")
display(df_train_lbl.isnull().sum())

print("\n== Missing values: validation_sequences ==")
display(df_val_seq.isnull().sum())

print("\n== Missing values: validation_labels ==")
display(df_val_lbl.isnull().sum())

print("\n== Missing values: test_sequences ==")
display(df_test_seq.isnull().sum())

print("\n== Missing values: sample_submission ==")
display(df_sub_sample.isnull().sum())


msno.bar(df_train_seq, sort="descending", figsize=(8,4), color='blue')
plt.title("Missing values in train_sequences.csv")
plt.show()

msno.bar(df_train_lbl, sort="descending", figsize=(8,4), color='green')
plt.title("Missing values in train_labels.csv")
plt.show()



def get_rna_length_stats(df, seq_col="sequence"):
    """Compute length stats from a DataFrame that has an RNA sequence column."""
    df["seq_length"] = df[seq_col].apply(len)
    return df["seq_length"].describe()

print("== Distribution of RNA sequence lengths in train_sequences ==")
display(get_rna_length_stats(df_train_seq.copy(), "sequence"))

print("\n== Distribution of RNA sequence lengths in validation_sequences ==")
display(get_rna_length_stats(df_val_seq.copy(), "sequence"))

print("\n== Distribution of RNA sequence lengths in test_sequences ==")
display(get_rna_length_stats(df_test_seq.copy(), "sequence"))


df_train_seq["seq_length"] = df_train_seq["sequence"].apply(len)
df_val_seq["seq_length"]   = df_val_seq["sequence"].apply(len)
df_test_seq["seq_length"]  = df_test_seq["sequence"].apply(len)

sns.histplot(df_train_seq["seq_length"], bins=30, color='blue', kde=True)
plt.title("Distribution of sequence lengths (train)")
plt.show()

sns.histplot(df_val_seq["seq_length"], bins=30, color='green', kde=True)
plt.title("Distribution of sequence lengths (validation)")
plt.show()

sns.histplot(df_test_seq["seq_length"], bins=30, color='orange', kde=True)
plt.title("Distribution of sequence lengths (test)")
plt.show()


def base_composition(seq):
    """Return counts/fractions of each base in an RNA sequence."""
    from collections import Counter
    c = Counter(seq)
    length = len(seq)
    return {base: c[base]/length for base in c}

train_comps = df_train_seq["sequence"].apply(base_composition)

# Flatten out into a DataFrame
df_bases_train = pd.DataFrame(train_comps.tolist()).fillna(0)  # fill missing with 0 if a base not present
print("Base composition columns in train_sequences:")
display(df_bases_train.describe())

# Plot average fraction of each base
mean_bases = df_bases_train.mean().sort_values(ascending=False)
sns.barplot(x=mean_bases.index, y=mean_bases.values, palette="Blues_d")
plt.title("Average base composition in train_sequences")
plt.xlabel("Base")
plt.ylabel("Mean fraction")
plt.show()


coords_cols = [col for col in df_train_lbl.columns if col.startswith("x_") or col.startswith("y_") or col.startswith("z_")]
print("Coordinate columns found in train_labels:", coords_cols)

# Summaries
display(df_train_lbl[coords_cols].describe())

# A quick correlation heatmap among x_1, y_1, z_1, x_2, y_2, z_2, etc.
corr = df_train_lbl[coords_cols].corr()
sns.heatmap(corr, cmap="vlag", center=0)
plt.title("Correlation among coordinate columns in train_labels")
plt.show()


coords_cols_val = [col for col in df_val_lbl.columns if col.startswith("x_") or col.startswith("y_") or col.startswith("z_")]
print("Coordinate columns found in validation_labels:", coords_cols_val)

# Summaries
display(df_val_lbl[coords_cols_val].describe())

# A quick correlation heatmap among these
corr_val = df_val_lbl[coords_cols_val].corr()
sns.heatmap(corr_val, cmap="vlag", center=0)
plt.title("Correlation among coordinate columns in validation_labels")
plt.show()


# The 'ID' in labels is of the form "target_id_resnum", so let's parse out the 'target_id' from ID if needed
# Alternatively, if train_labels has a 'target_id' column or we can do a direct merge on "ID".

df_train_lbl['target_id'] = df_train_lbl['ID'].apply(lambda x: x.split('_')[0])

# Now let's see if we can join with df_train_seq on 'target_id' if that is consistent:
merged_train = pd.merge(df_train_lbl, df_train_seq, on="target_id", how="inner")
print("Merged shape:", merged_train.shape)
display(merged_train.head(5))

# We can similarly do for validation if 'ID' format is consistent with 'target_id'
df_val_lbl['target_id'] = df_val_lbl['ID'].apply(lambda x: x.split('_')[0])
merged_val = pd.merge(df_val_lbl, df_val_seq, on="target_id", how="inner")
print("Merged shape (validation):", merged_val.shape)
display(merged_val.head(5))


if "resid" in merged_train.columns:
    # Check how resid compares to the length of the sequence
    # For each target_id, the maximum 'resid' might match the length of the sequence if it's 1-based indexing
    summary_df = merged_train.groupby("target_id").agg({
        "resid": "max",
        "sequence": lambda s: len(s.iloc[0])  # length of the first sequence in that group
    }).rename(columns={"resid": "max_resid", "sequence": "seq_length"})
    summary_df["diff"] = summary_df["seq_length"] - summary_df["max_resid"]
    print(summary_df.head(10))
    sns.histplot(summary_df["diff"], kde=True)
    plt.title("Difference between sequence length and max residue index")
    plt.show()


# Count how many distinct 'target_id' are in each set:
print("Unique target_id in train_sequences:", df_train_seq["target_id"].nunique())
print("Unique target_id in train_labels:", df_train_lbl["target_id"].nunique())
print("Unique target_id in validation_sequences:", df_val_seq["target_id"].nunique())
print("Unique target_id in validation_labels:", df_val_lbl["target_id"].nunique())
print("Unique target_id in test_sequences:", df_test_seq["target_id"].nunique())



# How many unique nucleotides appear in 'sequence' columns (like A, C, G, U, or others)?

def unique_bases(df, seq_col="sequence"):
    all_bases = set()
    for s in df[seq_col]:
        all_bases.update(list(s))
    return all_bases

print("train_sequences unique bases:", unique_bases(df_train_seq))
print("validation_sequences unique bases:", unique_bases(df_val_seq))
print("test_sequences unique bases:", unique_bases(df_test_seq))

# If there are 'N' or 'T' or 'X' in some sequences, that might require special handling.


# Create submission.csv directly from the sample_submission DataFrame
df_sub_sample.to_csv("submission.csv", index=False)




